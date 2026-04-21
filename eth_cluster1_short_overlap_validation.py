from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_practical_imitation import (
    PROXY_SETUP_SPECS,
    build_minute_arrays,
    build_proxy_scan_frame,
    build_proxy_signal_frame,
)
from eth_replication_hypothesis_test import ETH_SYMBOL, build_stage_windows, materialize_stage_bounds
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct


TARGET_CLUSTER_ID = 1
TARGET_SIDE = "Short"
TIME_BANDS_MINUTES = [5, 15, 30, 60]
PRICE_BANDS_PCT = [0.10, 0.25, 0.50, 1.00]
WINDOW_HORIZONS = [15, 30, 60]


def run_reanchored_cluster1_short_validation_analysis(
    trades_df: pd.DataFrame,
    eth_execution_reanchored_df: pd.DataFrame,
    broad_cluster1_signals_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "REANCHORED CLUSTER 1 SHORT VALIDATION - PRACTICAL ENTRY QUALITY, OVERLAP, AND IMITATION READINESS\n"
        "-----------------------------------------------------------------------------------------------\n"
        "No sufficient ETH Cluster 1 short data was available for practical overlap validation.\n"
    )

    real_trades = prepare_real_cluster1_short_trades(eth_execution_reanchored_df)
    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    if real_trades.empty or eth_trades.empty:
        return empty_result(empty, empty_section)

    minute_frame, frame_5m, stages = load_proxy_market_frames(eth_trades, logger)
    if minute_frame.empty or frame_5m.empty:
        section = (
            "REANCHORED CLUSTER 1 SHORT VALIDATION - PRACTICAL ENTRY QUALITY, OVERLAP, AND IMITATION READINESS\n"
            "-----------------------------------------------------------------------------------------------\n"
            "Practical overlap validation could not run because ETH candle data was unavailable.\n"
        )
        return empty_result(empty, section)

    model_signals = build_cluster1_short_model_signals(eth_trades, minute_frame, frame_5m, stages)
    if model_signals.empty:
        section = (
            "REANCHORED CLUSTER 1 SHORT VALIDATION - PRACTICAL ENTRY QUALITY, OVERLAP, AND IMITATION READINESS\n"
            "-----------------------------------------------------------------------------------------------\n"
            "The current Cluster 1 short proxy did not produce Stage A signals, so overlap validation could not continue.\n"
        )
        return empty_result(empty, section)

    logger.info(
        "Running reanchored Cluster 1 short overlap validation on %s real trades vs %s model signals",
        len(real_trades),
        len(model_signals),
    )

    minute_arrays = build_minute_arrays(minute_frame)
    real_trades = add_window_quality_metrics(
        real_trades,
        minute_arrays,
        time_column="effective_entry_time",
        price_column="effective_entry_price",
        side_column="side",
    )
    model_signals = add_window_quality_metrics(
        model_signals,
        minute_arrays,
        time_column="entry_time",
        price_column="entry_price",
        side_column="side",
    )

    overlap_matches = build_overlap_matches(real_trades, model_signals)
    broad_stage_a_count = extract_broad_stage_a_count(broad_cluster1_signals_df)
    overlap_summary = build_overlap_summary(real_trades, model_signals, overlap_matches, broad_stage_a_count)
    overlap_quality = build_overlap_quality(real_trades, model_signals, overlap_matches)
    section_text = build_overlap_section(real_trades, model_signals, overlap_matches, overlap_summary, overlap_quality)

    return {
        "real_trades": real_trades,
        "model_signals": model_signals,
        "overlap_matches": overlap_matches,
        "overlap_summary": overlap_summary,
        "overlap_quality": overlap_quality,
        "section_text": section_text,
    }


def empty_result(empty: pd.DataFrame, section_text: str) -> dict:
    return {
        "real_trades": empty,
        "model_signals": empty,
        "overlap_matches": empty,
        "overlap_summary": empty,
        "overlap_quality": empty,
        "section_text": section_text,
    }


def prepare_real_cluster1_short_trades(eth_execution_reanchored_df: pd.DataFrame) -> pd.DataFrame:
    if eth_execution_reanchored_df.empty:
        return pd.DataFrame()

    frame = eth_execution_reanchored_df.copy()
    for column in [
        "opened_dt_utc",
        "closed_dt_utc",
        "proxy_fill_time",
        "first_touch_time_utc",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    for column in [
        "cluster_id",
        "avgCost",
        "proxy_fill_price",
        "fill_delay_minutes",
        "reanchored_time_to_first_positive_minutes",
        "time_to_first_positive_minutes",
        "reanchored_return_after_15m_pct",
        "reanchored_return_after_30m_pct",
        "return_after_15m_pct",
        "return_after_30m_pct",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    proxy_available = frame.get("proxy_fill_available_flag", pd.Series(False, index=frame.index)).fillna(False)
    effective_time = frame.get("proxy_fill_time", pd.Series(pd.NaT, index=frame.index))
    effective_price = pd.to_numeric(frame.get("proxy_fill_price", frame.get("avgCost")), errors="coerce")

    frame["effective_entry_time"] = pd.to_datetime(
        np.where(proxy_available, effective_time, frame["opened_dt_utc"]),
        utc=True,
        errors="coerce",
    )
    frame["effective_entry_price"] = np.where(proxy_available, effective_price, pd.to_numeric(frame["avgCost"], errors="coerce"))
    frame["effective_anchor_source"] = np.where(proxy_available, "proxy_fill_time", "opened_time")
    frame["effective_time_to_first_positive_minutes"] = frame.get(
        "reanchored_time_to_first_positive_minutes",
        frame.get("time_to_first_positive_minutes"),
    )

    filtered = frame.loc[
        (frame["symbol"] == ETH_SYMBOL)
        & (pd.to_numeric(frame["cluster_id"], errors="coerce") == TARGET_CLUSTER_ID)
        & (frame["side"].astype(str).str.lower() == TARGET_SIDE.lower())
        & frame["effective_entry_time"].notna()
        & pd.to_numeric(frame["effective_entry_price"], errors="coerce").notna()
    ].copy()

    filtered["target_trade_id"] = filtered["trade_number"]
    return filtered.sort_values("effective_entry_time").reset_index(drop=True)


def load_proxy_market_frames(
    eth_trades: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    stages = build_stage_windows(eth_trades)
    fetch_start = min(stage.requested_start for stage in stages) - pd.Timedelta(hours=48)
    fetch_end = min(
        max(stage.requested_end for stage in stages) + pd.Timedelta(hours=25),
        pd.Timestamp.now(tz="UTC"),
    )

    logger.info(
        "Loading ETH candles for Cluster 1 short overlap validation from %s to %s",
        fetch_start.isoformat(),
        fetch_end.isoformat(),
    )

    minute_frame = fetch_futures_klines(
        ETH_SYMBOL,
        int(fetch_start.timestamp() * 1000),
        int(fetch_end.timestamp() * 1000),
        interval="1m",
        logger=logger,
    )
    frame_5m = fetch_futures_klines(
        ETH_SYMBOL,
        int(fetch_start.timestamp() * 1000),
        int(fetch_end.timestamp() * 1000),
        interval="5m",
        logger=logger,
    )
    return minute_frame, frame_5m, stages


def build_cluster1_short_model_signals(
    eth_trades: pd.DataFrame,
    minute_frame: pd.DataFrame,
    frame_5m: pd.DataFrame,
    stages: list,
) -> pd.DataFrame:
    scan_frame = build_proxy_scan_frame(frame_5m)
    if scan_frame.empty:
        return pd.DataFrame()

    max_signal_entry_time = scan_frame["entry_time"].max() - pd.Timedelta(hours=24)
    min_signal_entry_time = scan_frame["entry_time"].min()
    materialize_stage_bounds(stages, min_signal_entry_time, max_signal_entry_time)

    del eth_trades
    short_spec = next((spec for spec in PROXY_SETUP_SPECS if spec["setup_name"] == "eth_short_proxy"), None)
    if short_spec is None:
        return pd.DataFrame()

    raw_signals = build_proxy_signal_frame(scan_frame, minute_frame, short_spec, stages)
    if raw_signals.empty or "in_stage_a" not in raw_signals.columns:
        return pd.DataFrame()

    model = raw_signals.loc[raw_signals["in_stage_a"] == True].copy()
    model["model_signal_id"] = model["signal_id"]
    model["model_entry_time"] = pd.to_datetime(model["entry_time"], utc=True, errors="coerce")
    model["model_entry_price"] = pd.to_numeric(model["entry_price"], errors="coerce")
    return model.sort_values("model_entry_time").reset_index(drop=True)


def add_window_quality_metrics(
    frame: pd.DataFrame,
    minute_arrays: dict[str, np.ndarray],
    time_column: str,
    price_column: str,
    side_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    rows = []
    for _, row in frame.iterrows():
        metrics = compute_window_quality(
            minute_arrays,
            pd.to_datetime(row[time_column], utc=True, errors="coerce"),
            pd.to_numeric(pd.Series([row[price_column]]), errors="coerce").iloc[0],
            str(row[side_column]),
        )
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return frame.copy()
    return pd.concat([frame.reset_index(drop=True), metrics_df], axis=1)


def compute_window_quality(
    minute_arrays: dict[str, np.ndarray],
    entry_time: pd.Timestamp,
    entry_price: float,
    side: str,
) -> dict:
    row = {}
    if pd.isna(entry_time) or pd.isna(entry_price) or entry_price == 0:
        for horizon in WINDOW_HORIZONS:
            row[f"window_return_{horizon}m_pct"] = np.nan
            row[f"window_mae_{horizon}m_pct"] = np.nan
            row[f"window_mfe_{horizon}m_pct"] = np.nan
        return row

    close_times_ns = minute_arrays["close_time_ns"]
    open_times_ns = minute_arrays["open_time_ns"]
    close_prices = minute_arrays["close"]
    high_prices = minute_arrays["high"]
    low_prices = minute_arrays["low"]
    entry_ns = int(entry_time.value)

    for horizon in WINDOW_HORIZONS:
        target_ns = entry_ns + int(pd.Timedelta(minutes=horizon).value)
        close_idx = int(np.searchsorted(close_times_ns, target_ns, side="left"))
        if close_idx < len(close_times_ns):
            row[f"window_return_{horizon}m_pct"] = signed_return_pct(entry_price, float(close_prices[close_idx]), side)
        else:
            row[f"window_return_{horizon}m_pct"] = np.nan

        open_idx_start = int(np.searchsorted(open_times_ns, entry_ns, side="right"))
        open_idx_end = int(np.searchsorted(open_times_ns, target_ns, side="right") - 1)
        if open_idx_end >= open_idx_start >= 0:
            high_window = high_prices[open_idx_start : open_idx_end + 1]
            low_window = low_prices[open_idx_start : open_idx_end + 1]
            if side.lower() == "long":
                favorable = (high_window - entry_price) / entry_price * 100.0
                adverse = (entry_price - low_window) / entry_price * 100.0
            else:
                favorable = (entry_price - low_window) / entry_price * 100.0
                adverse = (high_window - entry_price) / entry_price * 100.0
            row[f"window_mfe_{horizon}m_pct"] = float(np.nanmax(favorable)) if favorable.size else np.nan
            row[f"window_mae_{horizon}m_pct"] = float(np.nanmax(adverse)) if adverse.size else np.nan
        else:
            row[f"window_mfe_{horizon}m_pct"] = np.nan
            row[f"window_mae_{horizon}m_pct"] = np.nan

    return row


def signed_return_pct(entry_price: float, future_price: float, side: str) -> float:
    if pd.isna(entry_price) or pd.isna(future_price) or entry_price == 0:
        return np.nan
    if side.lower() == "long":
        return (future_price - entry_price) / entry_price * 100.0
    return (entry_price - future_price) / entry_price * 100.0


def build_overlap_matches(real_trades: pd.DataFrame, model_signals: pd.DataFrame) -> pd.DataFrame:
    if real_trades.empty or model_signals.empty:
        return pd.DataFrame()

    real_rows = real_trades.copy()
    model_rows = model_signals.copy()
    model_lookup = model_rows.set_index("model_signal_id").to_dict("index")

    candidates = []
    diagnostics = {}
    for _, real in real_rows.iterrows():
        diagnostics[int(real["target_trade_id"])] = diagnostic_nearest_matches(real, model_rows)
        for _, model in model_rows.iterrows():
            real_time = pd.to_datetime(real["effective_entry_time"], utc=True)
            model_time = pd.to_datetime(model["model_entry_time"], utc=True)
            if pd.isna(real_time) or pd.isna(model_time):
                continue
            signed_distance = (model_time - real_time).total_seconds() / 60.0
            abs_distance = abs(signed_distance)
            if abs_distance > max(TIME_BANDS_MINUTES):
                continue

            real_price = pd.to_numeric(pd.Series([real["effective_entry_price"]]), errors="coerce").iloc[0]
            model_price = pd.to_numeric(pd.Series([model["model_entry_price"]]), errors="coerce").iloc[0]
            price_distance = abs(model_price - real_price) / real_price * 100.0 if pd.notna(real_price) and real_price != 0 else np.nan

            candidates.append(
                {
                    "target_trade_id": int(real["target_trade_id"]),
                    "model_signal_id": str(model["model_signal_id"]),
                    "signed_time_distance_minutes": signed_distance,
                    "abs_time_distance_minutes": abs_distance,
                    "price_distance_pct": price_distance,
                }
            )

    assignment = greedy_unique_assignment(candidates)
    rows = []
    for _, real in real_rows.iterrows():
        target_trade_id = int(real["target_trade_id"])
        diagnostic = diagnostics.get(target_trade_id, {})
        assigned = assignment.get(target_trade_id)
        assigned_model = model_lookup.get(assigned["model_signal_id"]) if assigned else None
        row = real.to_dict()

        nearest_any = diagnostic.get("nearest_any", {})
        nearest_same_day = diagnostic.get("nearest_same_day", {})
        row.update(
            {
                "nearest_any_signal_id": nearest_any.get("model_signal_id"),
                "nearest_any_signed_time_distance_minutes": nearest_any.get("signed_time_distance_minutes"),
                "nearest_any_abs_time_distance_minutes": nearest_any.get("abs_time_distance_minutes"),
                "nearest_any_price_distance_pct": nearest_any.get("price_distance_pct"),
                "nearest_same_day_signal_id": nearest_same_day.get("model_signal_id"),
                "nearest_same_day_signed_time_distance_minutes": nearest_same_day.get("signed_time_distance_minutes"),
                "nearest_same_day_abs_time_distance_minutes": nearest_same_day.get("abs_time_distance_minutes"),
                "nearest_same_day_price_distance_pct": nearest_same_day.get("price_distance_pct"),
            }
        )

        for minutes in TIME_BANDS_MINUTES:
            row[f"nearest_within_{minutes}m_flag"] = bool(
                pd.notna(row["nearest_any_abs_time_distance_minutes"])
                and row["nearest_any_abs_time_distance_minutes"] <= minutes
            )
        for band in PRICE_BANDS_PCT:
            key = price_band_key(band)
            row[f"nearest_price_le_{key}_flag"] = bool(
                pd.notna(row["nearest_any_price_distance_pct"])
                and row["nearest_any_price_distance_pct"] <= band
            )

        row["assigned_match_flag"] = assigned is not None
        if assigned and assigned_model:
            row["matched_model_signal_id"] = assigned["model_signal_id"]
            row["matched_time_distance_minutes"] = assigned["abs_time_distance_minutes"]
            row["matched_signed_time_distance_minutes"] = assigned["signed_time_distance_minutes"]
            row["matched_price_distance_pct"] = assigned["price_distance_pct"]
            row["matched_model_entry_time"] = assigned_model.get("model_entry_time")
            row["matched_model_entry_price"] = assigned_model.get("model_entry_price")
            row["matched_model_context_text"] = assigned_model.get("context_text")
            row["matched_model_trigger_text"] = assigned_model.get("trigger_text")
            row["matched_model_time_to_first_positive_minutes"] = assigned_model.get("time_to_first_positive_minutes")

            for horizon in WINDOW_HORIZONS:
                row[f"real_window_return_{horizon}m_pct"] = row.get(f"window_return_{horizon}m_pct")
                row[f"real_window_mae_{horizon}m_pct"] = row.get(f"window_mae_{horizon}m_pct")
                row[f"real_window_mfe_{horizon}m_pct"] = row.get(f"window_mfe_{horizon}m_pct")
                row[f"model_window_return_{horizon}m_pct"] = assigned_model.get(f"window_return_{horizon}m_pct")
                row[f"model_window_mae_{horizon}m_pct"] = assigned_model.get(f"window_mae_{horizon}m_pct")
                row[f"model_window_mfe_{horizon}m_pct"] = assigned_model.get(f"window_mfe_{horizon}m_pct")
                row[f"same_short_move_{horizon}m_flag"] = bool(
                    pd.notna(row[f"real_window_return_{horizon}m_pct"])
                    and pd.notna(row[f"model_window_return_{horizon}m_pct"])
                    and row[f"real_window_return_{horizon}m_pct"] > 0
                    and row[f"model_window_return_{horizon}m_pct"] > 0
                )
        else:
            row["matched_model_signal_id"] = pd.NA
            row["matched_time_distance_minutes"] = np.nan
            row["matched_signed_time_distance_minutes"] = np.nan
            row["matched_price_distance_pct"] = np.nan
            row["matched_model_entry_time"] = pd.NaT
            row["matched_model_entry_price"] = np.nan
            row["matched_model_context_text"] = pd.NA
            row["matched_model_trigger_text"] = pd.NA
            row["matched_model_time_to_first_positive_minutes"] = np.nan
            for horizon in WINDOW_HORIZONS:
                row[f"real_window_return_{horizon}m_pct"] = row.get(f"window_return_{horizon}m_pct")
                row[f"real_window_mae_{horizon}m_pct"] = row.get(f"window_mae_{horizon}m_pct")
                row[f"real_window_mfe_{horizon}m_pct"] = row.get(f"window_mfe_{horizon}m_pct")
                row[f"model_window_return_{horizon}m_pct"] = np.nan
                row[f"model_window_mae_{horizon}m_pct"] = np.nan
                row[f"model_window_mfe_{horizon}m_pct"] = np.nan
                row[f"same_short_move_{horizon}m_flag"] = False

        row["match_status"] = infer_match_status(row)
        row["main_failure_mode"] = infer_failure_mode(row)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("effective_entry_time").reset_index(drop=True)


def diagnostic_nearest_matches(real: pd.Series, model_rows: pd.DataFrame) -> dict:
    real_time = pd.to_datetime(real["effective_entry_time"], utc=True)
    real_price = pd.to_numeric(pd.Series([real["effective_entry_price"]]), errors="coerce").iloc[0]
    if pd.isna(real_time) or pd.isna(real_price) or real_price == 0:
        return {}

    nearest_any = None
    nearest_same_day = None
    real_day = real_time.floor("D")
    for _, model in model_rows.iterrows():
        model_time = pd.to_datetime(model["model_entry_time"], utc=True)
        model_price = pd.to_numeric(pd.Series([model["model_entry_price"]]), errors="coerce").iloc[0]
        if pd.isna(model_time) or pd.isna(model_price):
            continue

        signed_distance = (model_time - real_time).total_seconds() / 60.0
        abs_distance = abs(signed_distance)
        price_distance = abs(model_price - real_price) / real_price * 100.0
        candidate = {
            "model_signal_id": str(model["model_signal_id"]),
            "signed_time_distance_minutes": signed_distance,
            "abs_time_distance_minutes": abs_distance,
            "price_distance_pct": price_distance,
        }
        if nearest_any is None or candidate_rank(candidate) < candidate_rank(nearest_any):
            nearest_any = candidate
        if model_time.floor("D") == real_day:
            if nearest_same_day is None or candidate_rank(candidate) < candidate_rank(nearest_same_day):
                nearest_same_day = candidate
    return {"nearest_any": nearest_any or {}, "nearest_same_day": nearest_same_day or {}}


def candidate_rank(candidate: dict) -> tuple[float, float]:
    return (
        float(candidate.get("abs_time_distance_minutes", np.inf)),
        float(candidate.get("price_distance_pct", np.inf)),
    )


def greedy_unique_assignment(candidates: list[dict]) -> dict[int, dict]:
    assignment: dict[int, dict] = {}
    used_models: set[str] = set()
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            float(row["abs_time_distance_minutes"]),
            float(row["price_distance_pct"]) if pd.notna(row["price_distance_pct"]) else np.inf,
            int(row["target_trade_id"]),
            str(row["model_signal_id"]),
        ),
    )
    for candidate in sorted_candidates:
        trade_id = int(candidate["target_trade_id"])
        signal_id = str(candidate["model_signal_id"])
        if trade_id in assignment or signal_id in used_models:
            continue
        assignment[trade_id] = candidate
        used_models.add(signal_id)
    return assignment


def infer_match_status(row: dict) -> str:
    if row.get("assigned_match_flag"):
        price_distance = row.get("matched_price_distance_pct")
        time_distance = row.get("matched_time_distance_minutes")
        if pd.notna(time_distance) and pd.notna(price_distance) and time_distance <= 15 and price_distance <= 0.25:
            return "tight_match"
        if pd.notna(time_distance) and pd.notna(price_distance) and time_distance <= 30 and price_distance <= 0.50:
            return "workable_match"
        return "loose_match"
    return "unmatched"


def infer_failure_mode(row: dict) -> str:
    if row.get("assigned_match_flag"):
        return "matched"
    if pd.notna(row.get("nearest_any_abs_time_distance_minutes")) and row.get("nearest_any_abs_time_distance_minutes", np.inf) <= 60:
        if pd.notna(row.get("nearest_any_price_distance_pct")) and row.get("nearest_any_price_distance_pct", np.inf) > 1.00:
            return "wrong_price_zone"
        return "candidate_conflict_or_loose_zone"
    same_day_distance = row.get("nearest_same_day_signed_time_distance_minutes")
    if pd.notna(same_day_distance):
        return "proxy_too_early_same_day" if same_day_distance < 0 else "proxy_too_late_same_day"
    return "no_same_day_signal"


def extract_broad_stage_a_count(broad_cluster1_signals_df: pd.DataFrame) -> float:
    if broad_cluster1_signals_df.empty:
        return np.nan
    if "in_stage_a" not in broad_cluster1_signals_df.columns:
        return float(len(broad_cluster1_signals_df))
    return float(broad_cluster1_signals_df["in_stage_a"].fillna(False).sum())


def build_overlap_summary(
    real_trades: pd.DataFrame,
    model_signals: pd.DataFrame,
    overlap_matches: pd.DataFrame,
    broad_stage_a_count: float,
) -> pd.DataFrame:
    if real_trades.empty or overlap_matches.empty:
        return pd.DataFrame()

    row = {
        "real_trade_count": int(len(real_trades)),
        "model_signal_count": int(len(model_signals)),
        "broad_stage_a_signal_count": broad_stage_a_count,
        "proxy_signals_per_real_trade": safe_divide(len(model_signals), len(real_trades)),
        "matched_unique_share": overlap_matches["assigned_match_flag"].mean(),
        "median_matched_time_distance_minutes": overlap_matches.loc[
            overlap_matches["assigned_match_flag"], "matched_time_distance_minutes"
        ].median(),
        "median_matched_price_distance_pct": overlap_matches.loc[
            overlap_matches["assigned_match_flag"], "matched_price_distance_pct"
        ].median(),
        "tight_match_share": overlap_matches["match_status"].eq("tight_match").mean(),
        "workable_or_better_match_share": overlap_matches["match_status"].isin(["tight_match", "workable_match"]).mean(),
    }

    for minutes in TIME_BANDS_MINUTES:
        row[f"share_within_{minutes}m"] = overlap_matches[f"nearest_within_{minutes}m_flag"].mean()
    for band in PRICE_BANDS_PCT:
        key = price_band_key(band)
        row[f"share_nearest_price_le_{key}"] = overlap_matches[f"nearest_price_le_{key}_flag"].mean()

    row["share_within_15m_and_0p25pct"] = (
        overlap_matches["nearest_within_15m_flag"] & overlap_matches["nearest_price_le_0p25_flag"]
    ).mean()
    row["share_within_30m_and_0p50pct"] = (
        overlap_matches["nearest_within_30m_flag"] & overlap_matches["nearest_price_le_0p50_flag"]
    ).mean()
    row["share_within_60m_and_1p00pct"] = (
        overlap_matches["nearest_within_60m_flag"] & overlap_matches["nearest_price_le_1p00_flag"]
    ).mean()
    row["same_short_move_30m_share_matched"] = overlap_matches.loc[
        overlap_matches["assigned_match_flag"], "same_short_move_30m_flag"
    ].mean()
    row["same_short_move_60m_share_matched"] = overlap_matches.loc[
        overlap_matches["assigned_match_flag"], "same_short_move_60m_flag"
    ].mean()

    reason_share = overlap_matches["main_failure_mode"].value_counts(normalize=True)
    for reason, share in reason_share.items():
        row[f"failure_share__{reason}"] = share

    return pd.DataFrame([row])


def build_overlap_quality(
    real_trades: pd.DataFrame,
    model_signals: pd.DataFrame,
    overlap_matches: pd.DataFrame,
) -> pd.DataFrame:
    if overlap_matches.empty:
        return pd.DataFrame()

    matched_ids = overlap_matches.loc[overlap_matches["assigned_match_flag"], "matched_model_signal_id"].dropna().astype(str).tolist()
    matched_model = model_signals.loc[model_signals["model_signal_id"].astype(str).isin(matched_ids)].copy()
    matched_real = overlap_matches.loc[overlap_matches["assigned_match_flag"]].copy()

    rows = [
        summarize_quality_group("real_cluster1_short_trades", real_trades, time_to_first_positive_column="effective_time_to_first_positive_minutes"),
        summarize_quality_group("matched_real_cluster1_short_trades", matched_real, time_to_first_positive_column="effective_time_to_first_positive_minutes"),
        summarize_quality_group("matched_model_signals", matched_model, time_to_first_positive_column="time_to_first_positive_minutes"),
        summarize_quality_group("all_model_signals_stage_a", model_signals, time_to_first_positive_column="time_to_first_positive_minutes"),
    ]

    if not matched_real.empty:
        pair_row = {
            "group_name": "matched_pair_gaps",
            "trades": int(len(matched_real)),
            "median_time_distance_minutes": pd.to_numeric(matched_real["matched_time_distance_minutes"], errors="coerce").median(),
            "median_price_distance_pct": pd.to_numeric(matched_real["matched_price_distance_pct"], errors="coerce").median(),
            "same_short_move_15m_share": matched_real["same_short_move_15m_flag"].mean(),
            "same_short_move_30m_share": matched_real["same_short_move_30m_flag"].mean(),
            "same_short_move_60m_share": matched_real["same_short_move_60m_flag"].mean(),
        }
        for horizon in WINDOW_HORIZONS:
            pair_row[f"median_abs_gap_return_{horizon}m_pct"] = (
                pd.to_numeric(matched_real[f"model_window_return_{horizon}m_pct"], errors="coerce")
                - pd.to_numeric(matched_real[f"real_window_return_{horizon}m_pct"], errors="coerce")
            ).abs().median()
            pair_row[f"median_abs_gap_mae_{horizon}m_pct"] = (
                pd.to_numeric(matched_real[f"model_window_mae_{horizon}m_pct"], errors="coerce")
                - pd.to_numeric(matched_real[f"real_window_mae_{horizon}m_pct"], errors="coerce")
            ).abs().median()
            pair_row[f"median_abs_gap_mfe_{horizon}m_pct"] = (
                pd.to_numeric(matched_real[f"model_window_mfe_{horizon}m_pct"], errors="coerce")
                - pd.to_numeric(matched_real[f"real_window_mfe_{horizon}m_pct"], errors="coerce")
            ).abs().median()
        rows.append(pair_row)

    return pd.DataFrame(rows)


def summarize_quality_group(group_name: str, frame: pd.DataFrame, time_to_first_positive_column: str) -> dict:
    row = {
        "group_name": group_name,
        "trades": int(len(frame)),
        "median_time_to_first_positive_minutes": pd.to_numeric(frame.get(time_to_first_positive_column), errors="coerce").median(),
    }
    for horizon in WINDOW_HORIZONS:
        row[f"median_return_{horizon}m_pct"] = pd.to_numeric(frame.get(f"window_return_{horizon}m_pct"), errors="coerce").median()
        row[f"median_mae_{horizon}m_pct"] = pd.to_numeric(frame.get(f"window_mae_{horizon}m_pct"), errors="coerce").median()
        row[f"median_mfe_{horizon}m_pct"] = pd.to_numeric(frame.get(f"window_mfe_{horizon}m_pct"), errors="coerce").median()
    return row


def build_overlap_section(
    real_trades: pd.DataFrame,
    model_signals: pd.DataFrame,
    overlap_matches: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    overlap_quality: pd.DataFrame,
) -> str:
    if overlap_summary.empty or overlap_quality.empty:
        return (
            "REANCHORED CLUSTER 1 SHORT VALIDATION - PRACTICAL ENTRY QUALITY, OVERLAP, AND IMITATION READINESS\n"
            "-----------------------------------------------------------------------------------------------\n"
            "Practical overlap validation could not be completed.\n"
        )

    summary = overlap_summary.iloc[0]
    real_quality = select_quality_row(overlap_quality, "real_cluster1_short_trades")
    matched_model_quality = select_quality_row(overlap_quality, "matched_model_signals")
    all_model_quality = select_quality_row(overlap_quality, "all_model_signals_stage_a")
    pair_gaps = select_quality_row(overlap_quality, "matched_pair_gaps")

    proximity_label = determine_imitation_proximity(summary, pair_gaps)
    readiness_label = determine_imitation_readiness(summary, pair_gaps)
    failure_mode = dominant_failure_mode(overlap_matches)
    confidence = determine_overlap_confidence(real_trades, summary, pair_gaps)

    report = f"""
REANCHORED CLUSTER 1 SHORT VALIDATION - PRACTICAL ENTRY QUALITY, OVERLAP, AND IMITATION READINESS
-----------------------------------------------------------------------------------------------
This stage checks whether the current Cluster 1 short imitation candidate lands near the trader's real re-anchored ETH Cluster 1 short trades in time, price, and narrow post-entry behavior. It does not claim exact execution recovery.

Coverage:
- Real Cluster 1 short trades: {fmt_int(summary.get('real_trade_count'))}.
- Broad Cluster 1 family Stage A signals: {fmt_int(summary.get('broad_stage_a_signal_count'))}.
- Current practical Cluster 1 short proxy Stage A signals: {fmt_int(summary.get('model_signal_count'))}.
- Proxy signals per real Cluster 1 short trade: {fmt_number(summary.get('proxy_signals_per_real_trade'))}.

Time overlap:
- Real trades with a proxy signal within 5m / 15m / 30m / 60m: {pct(summary.get('share_within_5m'))} / {pct(summary.get('share_within_15m'))} / {pct(summary.get('share_within_30m'))} / {pct(summary.get('share_within_60m'))}.
- Unique assigned match share inside the 60m overlap window: {pct(summary.get('matched_unique_share'))}.
- Median matched timing distance: {fmt_number(summary.get('median_matched_time_distance_minutes'))} minutes.

Time + price overlap:
- Share within 15m and 0.25%: {pct(summary.get('share_within_15m_and_0p25pct'))}.
- Share within 30m and 0.50%: {pct(summary.get('share_within_30m_and_0p50pct'))}.
- Share within 60m and 1.00%: {pct(summary.get('share_within_60m_and_1p00pct'))}.
- Median matched price distance to real avgCost: {fmt_number(summary.get('median_matched_price_distance_pct'))}%.

Narrow-window entry quality:
- Real Cluster 1 short trades median return after 15m / 30m / 60m: {fmt_number(real_quality.get('median_return_15m_pct'))}% / {fmt_number(real_quality.get('median_return_30m_pct'))}% / {fmt_number(real_quality.get('median_return_60m_pct'))}%.
- Matched model signals median return after 15m / 30m / 60m: {fmt_number(matched_model_quality.get('median_return_15m_pct'))}% / {fmt_number(matched_model_quality.get('median_return_30m_pct'))}% / {fmt_number(matched_model_quality.get('median_return_60m_pct'))}%.
- All model Stage A signals median return after 15m / 30m / 60m: {fmt_number(all_model_quality.get('median_return_15m_pct'))}% / {fmt_number(all_model_quality.get('median_return_30m_pct'))}% / {fmt_number(all_model_quality.get('median_return_60m_pct'))}%.
- Matched-case same-short-move share at 30m / 60m: {pct(summary.get('same_short_move_30m_share_matched'))} / {pct(summary.get('same_short_move_60m_share_matched'))}.
- Matched-case median absolute gap in 30m return: {fmt_number(pair_gaps.get('median_abs_gap_return_30m_pct'))}%.

Interpretation:
- Current Cluster 1 short proxy shows {proximity_label}.
- Main failure mode at current precision: {failure_mode}.
- In narrow practical move terms relevant to roughly 10x leverage, current evidence looks {readiness_label}.
- Confidence level: {confidence}.

PRACTICAL INTERPRETATION - IS CLUSTER 1 SHORT NOW CLOSE ENOUGH TO IMITATE?
---------------------------------------------------------------------------
- The current Cluster 1 short proxy {'can already be treated as a meaningful imitation base' if readiness_label == 'close enough to support a candidate imitation base' else 'still looks more like broad-family capture than practical entry replication'}.
- {'Execution-aware branching can start from this base.' if readiness_label == 'close enough to support a candidate imitation base' else 'One more refinement step is still preferable before execution-aware branching.'}
- The main remaining weakness is {failure_mode}.
- The next logical step is {'to compare execution-aware variants only within the subset of proxy signals that already overlap the real short family reasonably well' if readiness_label == 'close enough to support a candidate imitation base' else 'to tighten practical timing or zone selection specifically around the matched and near-missed Cluster 1 short trades'}.
""".strip()
    return report + "\n"


def select_quality_row(df: pd.DataFrame, group_name: str) -> pd.Series:
    match = df.loc[df["group_name"] == group_name]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def determine_imitation_proximity(summary: pd.Series, pair_gaps: pd.Series) -> str:
    within_60 = safe_number(summary.get("share_within_60m"))
    within_30_and_price = safe_number(summary.get("share_within_30m_and_0p50pct"))
    same_move_30 = safe_number(summary.get("same_short_move_30m_share_matched"))
    gap_30 = safe_number(pair_gaps.get("median_abs_gap_return_30m_pct"))

    if within_30_and_price >= 0.40 and same_move_30 >= 0.60 and gap_30 <= 0.40:
        return "meaningful practical overlap with the real short-family entries"
    if within_60 >= 0.55 and same_move_30 >= 0.50:
        return "partial overlap with the right short family, but with loose timing or zone precision"
    return "mostly broad-family overlap rather than close practical entry alignment"


def determine_imitation_readiness(summary: pd.Series, pair_gaps: pd.Series) -> str:
    within_30_and_price = safe_number(summary.get("share_within_30m_and_0p50pct"))
    within_60_and_price = safe_number(summary.get("share_within_60m_and_1p00pct"))
    gap_60 = safe_number(pair_gaps.get("median_abs_gap_return_60m_pct"))
    if within_30_and_price >= 0.40 and within_60_and_price >= 0.60 and gap_60 <= 0.55:
        return "close enough to support a candidate imitation base"
    if within_60_and_price >= 0.40:
        return "partly aligned, but still not tight enough for strong imitation confidence"
    return "too loose for practical imitation readiness"


def dominant_failure_mode(overlap_matches: pd.DataFrame) -> str:
    if overlap_matches.empty:
        return "insufficient data"
    unmatched = overlap_matches.loc[~overlap_matches["assigned_match_flag"], "main_failure_mode"]
    if unmatched.empty:
        return "no dominant miss mode because almost everything matched"
    top_reason = unmatched.value_counts().idxmax()
    mapping = {
        "wrong_price_zone": "poor price proximity even when timing is nearby",
        "proxy_too_early_same_day": "proxy signals often appear too early relative to the real entry",
        "proxy_too_late_same_day": "proxy signals often appear too late relative to the real entry",
        "candidate_conflict_or_loose_zone": "there are nearby broad candidates, but they are too loose or collide across real trades",
        "no_same_day_signal": "the proxy often misses the real trade day entirely",
    }
    return mapping.get(top_reason, str(top_reason))


def determine_overlap_confidence(
    real_trades: pd.DataFrame,
    summary: pd.Series,
    pair_gaps: pd.Series,
) -> str:
    trade_count = len(real_trades)
    matched_share = safe_number(summary.get("matched_unique_share"))
    gap_30 = safe_number(pair_gaps.get("median_abs_gap_return_30m_pct"))
    if trade_count >= 25 and matched_share >= 0.50 and gap_30 <= 0.60:
        return "medium"
    if trade_count >= 15 and matched_share >= 0.30:
        return "low"
    return "low"


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def safe_number(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def price_band_key(band: float) -> str:
    return f"{band:.2f}".replace(".", "p")


def create_reanchored_cluster1_short_validation_plots(
    overlap_matches: pd.DataFrame,
    overlap_quality: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if overlap_matches.empty or overlap_quality.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    timing_path = plots_dir / "eth_cluster1_short_overlap_timing.png"
    price_path = plots_dir / "eth_cluster1_short_overlap_price_distance.png"
    quality_path = plots_dir / "eth_cluster1_short_overlap_quality.png"

    fig_timing = plot_overlap_timing(overlap_matches)
    fig_timing.savefig(timing_path, dpi=150, bbox_inches="tight")
    plt.close(fig_timing)

    fig_price = plot_overlap_price_distance(overlap_matches)
    fig_price.savefig(price_path, dpi=150, bbox_inches="tight")
    plt.close(fig_price)

    fig_quality = plot_overlap_quality(overlap_quality)
    fig_quality.savefig(quality_path, dpi=150, bbox_inches="tight")
    plt.close(fig_quality)

    logger.info("Saved Cluster 1 short overlap validation plots")


def plot_overlap_timing(overlap_matches: pd.DataFrame) -> plt.Figure:
    matched = overlap_matches.loc[overlap_matches["assigned_match_flag"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if matched.empty:
        ax.text(0.5, 0.5, "No matched real-to-model pairs", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.hist(pd.to_numeric(matched["matched_time_distance_minutes"], errors="coerce").dropna(), bins=12, color="#1f77b4", edgecolor="black")
    for band in TIME_BANDS_MINUTES:
        ax.axvline(band, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Cluster 1 short overlap: matched timing distance")
    ax.set_xlabel("Absolute timing distance to nearest assigned model signal (minutes)")
    ax.set_ylabel("Matched real trades")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_overlap_price_distance(overlap_matches: pd.DataFrame) -> plt.Figure:
    matched = overlap_matches.loc[overlap_matches["assigned_match_flag"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if matched.empty:
        ax.text(0.5, 0.5, "No matched real-to-model pairs", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.hist(pd.to_numeric(matched["matched_price_distance_pct"], errors="coerce").dropna(), bins=12, color="#2ca02c", edgecolor="black")
    for band in PRICE_BANDS_PCT:
        ax.axvline(band, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Cluster 1 short overlap: matched price distance")
    ax.set_xlabel("Absolute entry-price distance to real avgCost (%)")
    ax.set_ylabel("Matched real trades")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_overlap_quality(overlap_quality: pd.DataFrame) -> plt.Figure:
    groups = ["real_cluster1_short_trades", "matched_model_signals", "all_model_signals_stage_a"]
    plot_df = overlap_quality.loc[overlap_quality["group_name"].isin(groups)].copy()
    label_map = {
        "real_cluster1_short_trades": "Real target",
        "matched_model_signals": "Matched model",
        "all_model_signals_stage_a": "All model",
    }
    plot_df["plot_label"] = plot_df["group_name"].map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    draw_quality_panel(axes[0], plot_df, "median_return", "Median return %")
    draw_quality_panel(axes[1], plot_df, "median_mae", "Median MAE %")
    draw_quality_panel(axes[2], plot_df, "median_mfe", "Median MFE %")
    fig.suptitle("Cluster 1 short overlap: narrow-window quality")
    fig.tight_layout()
    return fig


def draw_quality_panel(ax: plt.Axes, plot_df: pd.DataFrame, metric_prefix: str, title: str) -> None:
    horizons = WINDOW_HORIZONS
    x = np.arange(len(plot_df))
    width = 0.22
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, horizon in enumerate(horizons):
        values = pd.to_numeric(plot_df[f"{metric_prefix}_{horizon}m_pct"], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, values, width=width, label=f"{horizon}m", color=colors[idx], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].tolist())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
