from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from market import fetch_futures_klines
from market_context_at_entry import prepare_context_features
from utils import fmt_int, fmt_number, pct, safe_divide


ETH_SYMBOL = "ETHUSDT"
REPLICATION_HORIZONS_MINUTES = [1, 5, 15, 30, 60, 240, 720, 1440]
STAGE_COLORS = {
    "stage_a": "#1f77b4",
    "stage_b": "#ff7f0e",
    "stage_c": "#2ca02c",
}


@dataclass
class ReplicationStage:
    stage_id: str
    label: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    effective_start: pd.Timestamp | None = None
    effective_end: pd.Timestamp | None = None


def run_eth_replication_hypothesis_test(
    trades_df: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_rules: list[dict],
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH REPLICATION HYPOTHESIS TEST\n"
        "-------------------------------\n"
        "No sufficient ETH rule-extraction output was available for the replication hypothesis test.\n"
    )

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    eth_clusters = cluster_frame.loc[cluster_frame["symbol"] == ETH_SYMBOL].copy()
    focus_rules = sorted(cluster_rules, key=lambda item: item.get("target_cluster", 99))[:2] if cluster_rules else []

    if eth_trades.empty or eth_clusters.empty or len(focus_rules) < 2:
        return {
            "signals_cluster1": empty,
            "signals_cluster2": empty,
            "summary": empty,
            "stage_comparison": empty,
            "section_text": empty_section,
            "combined_signals": empty,
        }

    stages = build_stage_windows(eth_trades)
    fetch_start = min(stage.requested_start for stage in stages) - pd.Timedelta(hours=24)
    fetch_end = min(
        max(stage.requested_end for stage in stages) + pd.Timedelta(hours=25),
        pd.Timestamp.now(tz="UTC"),
    )

    logger.info(
        "Running ETH replication hypothesis test from %s to %s",
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

    if minute_frame.empty or frame_5m.empty:
        section = (
            "ETH REPLICATION HYPOTHESIS TEST\n"
            "-------------------------------\n"
            "Market scan could not be completed because ETHUSDT candle data was unavailable.\n"
        )
        return {
            "signals_cluster1": empty,
            "signals_cluster2": empty,
            "summary": empty,
            "stage_comparison": empty,
            "section_text": section,
            "combined_signals": empty,
        }

    scan_frame = build_scan_frame(minute_frame, frame_5m)
    if scan_frame.empty:
        section = (
            "ETH REPLICATION HYPOTHESIS TEST\n"
            "-------------------------------\n"
            "Market scan could not be completed because the ETH scan frame was empty.\n"
        )
        return {
            "signals_cluster1": empty,
            "signals_cluster2": empty,
            "summary": empty,
            "stage_comparison": empty,
            "section_text": section,
            "combined_signals": empty,
        }

    max_signal_entry_time = scan_frame["entry_time"].max() - pd.Timedelta(hours=24)
    min_signal_entry_time = scan_frame["entry_time"].min()
    materialize_stage_bounds(stages, min_signal_entry_time, max_signal_entry_time)

    real_profiles = build_real_cluster_profiles(eth_trades, eth_clusters, cluster_summary, minute_frame)

    all_signal_frames = []
    summary_rows = []

    for rule in focus_rules:
        cluster_id = int(rule["target_cluster"])
        cluster_desc = str(rule.get("cluster_description", f"Cluster {cluster_id}"))
        logger.info("Scanning ETH market for replication signals for cluster %s", cluster_id)
        signals = build_signal_frame_for_rule(scan_frame, minute_frame, rule, cluster_id, cluster_desc)
        if signals.empty:
            signals = pd.DataFrame(columns=["cluster_id", "signal_id", "entry_time"])

        for stage in stages:
            stage_flag = stage_membership_flag(signals["entry_time"], stage)
            signals[f"in_{stage.stage_id}"] = stage_flag
            stage_signals = signals.loc[stage_flag].copy()
            summary_rows.append(
                build_stage_summary(
                    cluster_id,
                    cluster_desc,
                    rule,
                    stage,
                    stage_signals,
                    real_profiles.get(cluster_id, {}),
                )
            )

        all_signal_frames.append(signals)

    combined_signals = pd.concat(all_signal_frames, ignore_index=True, sort=False) if all_signal_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["cluster_id", "stage_sort_order"]).reset_index(drop=True)
    stage_comparison_df = build_stage_comparison(summary_df)
    section_text = build_replication_section(summary_df)

    return {
        "signals_cluster1": combined_signals.loc[combined_signals["cluster_id"] == 1].copy(),
        "signals_cluster2": combined_signals.loc[combined_signals["cluster_id"] == 2].copy(),
        "summary": summary_df,
        "stage_comparison": stage_comparison_df,
        "section_text": section_text,
        "combined_signals": combined_signals,
    }


def build_stage_windows(eth_trades: pd.DataFrame) -> list[ReplicationStage]:
    eth_start = eth_trades["opened_dt_utc"].min()
    eth_end = eth_trades["opened_dt_utc"].max()
    six_months = pd.Timedelta(days=182)
    two_years = pd.Timedelta(days=730)

    return [
        ReplicationStage("stage_a", "Stage A", eth_start, eth_end),
        ReplicationStage("stage_b", "Stage B", eth_start - six_months, eth_end + six_months),
        ReplicationStage("stage_c", "Stage C", eth_end - two_years, eth_end),
    ]


def materialize_stage_bounds(
    stages: list[ReplicationStage],
    min_available_entry_time: pd.Timestamp,
    max_available_entry_time: pd.Timestamp,
) -> None:
    for stage in stages:
        effective_start = max(stage.requested_start, min_available_entry_time)
        effective_end = min(stage.requested_end, max_available_entry_time)
        stage.effective_start = effective_start
        stage.effective_end = effective_end if effective_end >= effective_start else None


def build_scan_frame(minute_frame: pd.DataFrame, frame_5m: pd.DataFrame) -> pd.DataFrame:
    minute = minute_frame[["open_time", "close_time", "open", "high", "low", "close", "volume"]].copy()
    minute = minute.sort_values("close_time").reset_index(drop=True)
    minute["minute_index"] = np.arange(len(minute))
    minute["entry_time"] = minute["close_time"]
    minute["entry_price"] = pd.to_numeric(minute["close"], errors="coerce")
    minute["open_hour_utc"] = minute["entry_time"].dt.hour

    close_series = pd.to_numeric(minute["close"], errors="coerce")
    high_series = pd.to_numeric(minute["high"], errors="coerce")
    low_series = pd.to_numeric(minute["low"], errors="coerce")

    minute["return_pre_1m_pct"] = close_series.pct_change(1) * 100.0
    minute["return_pre_5m_pct"] = close_series.pct_change(5) * 100.0
    minute["return_pre_15m_pct"] = close_series.pct_change(15) * 100.0
    minute["return_pre_60m_pct"] = close_series.pct_change(60) * 100.0
    minute["prev_4h_high"] = high_series.shift(1).rolling(240, min_periods=2).max()
    minute["prev_4h_low"] = low_series.shift(1).rolling(240, min_periods=2).min()
    minute["distance_to_last_4h_high_pct"] = (minute["prev_4h_high"] - close_series) / close_series * 100.0
    minute["distance_to_last_4h_low_pct"] = (close_series - minute["prev_4h_low"]) / close_series * 100.0

    prepared_5m = prepare_context_features(frame_5m, "5m").copy()
    features_5m = prepared_5m[["close_time", "distance_to_ema20_pct"]].rename(
        columns={"distance_to_ema20_pct": "distance_to_ema20_pct_5m"}
    )
    features_5m = features_5m.sort_values("close_time").reset_index(drop=True)

    merged = pd.merge_asof(
        minute.sort_values("entry_time"),
        features_5m,
        left_on="entry_time",
        right_on="close_time",
        direction="backward",
    )
    return merged.drop(columns=["close_time_y"]).rename(columns={"close_time_x": "close_time"})


def build_real_cluster_profiles(
    eth_trades: pd.DataFrame,
    eth_clusters: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    minute_frame: pd.DataFrame,
) -> dict[int, dict]:
    merged = eth_clusters.merge(
        eth_trades[["trade_number", "avgCost"]],
        on="trade_number",
        how="left",
    )

    profiles: dict[int, dict] = {}
    for cluster_id in sorted(merged["cluster_id"].dropna().unique().tolist()):
        cluster_id = int(cluster_id)
        cluster_rows = merged.loc[merged["cluster_id"] == cluster_id].copy()
        metrics = evaluate_entry_profiles(
            minute_frame,
            cluster_rows.rename(columns={"avgCost": "entry_price", "opened_dt_utc": "entry_time"})[
                ["trade_number", "entry_time", "entry_price", "side"]
            ],
            id_prefix=f"real_cluster_{cluster_id}",
        )
        summary_row = cluster_summary.loc[cluster_summary["cluster_id"] == cluster_id]
        cluster_desc = str(summary_row.iloc[0]["cluster_description"]) if not summary_row.empty else f"Cluster {cluster_id}"
        profiles[cluster_id] = {
            "cluster_description": cluster_desc,
            "metrics": metrics,
            "summary": summarize_signal_like_profiles(metrics),
        }
    return profiles


def build_signal_frame_for_rule(
    scan_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    rule: dict,
    cluster_id: int,
    cluster_desc: str,
) -> pd.DataFrame:
    mask = apply_rule_to_scan_frame(scan_frame, rule)
    start_mask = mask & ~mask.shift(1, fill_value=False)
    if not start_mask.any():
        return pd.DataFrame()

    episode_ids = start_mask.cumsum().where(mask)
    episode_lengths = episode_ids.dropna().astype(int).value_counts().to_dict()
    dominant_side = infer_rule_side(rule)

    starts = scan_frame.loc[start_mask].copy()
    starts["episode_id"] = episode_ids.loc[start_mask].astype(int).to_numpy()
    starts["episode_duration_minutes"] = starts["episode_id"].map(episode_lengths).astype(float)
    starts["cluster_id"] = cluster_id
    starts["cluster_description"] = cluster_desc
    starts["side"] = dominant_side
    starts["rule_text"] = " AND ".join(condition["text"] for condition in rule.get("conditions", []))
    starts["signal_id"] = starts["entry_time"].dt.strftime(f"C{cluster_id}_%Y%m%dT%H%M%S")

    metrics = evaluate_entry_profiles(
        minute_frame,
        starts[["signal_id", "entry_time", "entry_price", "side"]],
        id_prefix=f"cluster_{cluster_id}",
    )
    merged = starts.merge(metrics, on=["signal_id", "entry_time", "entry_price", "side"], how="left")
    merged["fast_positive_reaction_flag"] = (
        (merged["return_after_1m_pct"] > 0) | (merged["return_after_5m_pct"] > 0)
    )
    return merged.sort_values("entry_time").reset_index(drop=True)


def apply_rule_to_scan_frame(scan_frame: pd.DataFrame, rule: dict) -> pd.Series:
    mask = pd.Series(True, index=scan_frame.index, dtype=bool)
    for condition in rule.get("conditions", []):
        feature_key = condition.get("feature_key")
        if feature_key == "side":
            continue
        series = pd.to_numeric(scan_frame.get(feature_key), errors="coerce")
        rule_type = condition.get("rule_type")
        if rule_type == "ge":
            cond_mask = series >= condition.get("lower_value")
        elif rule_type == "le":
            cond_mask = series <= condition.get("upper_value")
        else:
            cond_mask = series.between(
                condition.get("lower_value"),
                condition.get("upper_value"),
                inclusive="both",
            )
        mask = mask & cond_mask.fillna(False)
    return mask.fillna(False)


def infer_rule_side(rule: dict) -> str:
    if rule.get("dominant_side"):
        return str(rule["dominant_side"])
    for condition in rule.get("conditions", []):
        if condition.get("feature_key") == "side":
            return str(condition.get("value", "Long"))
    return "Long"


def evaluate_entry_profiles(
    minute_frame: pd.DataFrame,
    entries: pd.DataFrame,
    id_prefix: str,
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()

    close_times_ns = minute_frame["close_time"].astype("int64").to_numpy(dtype=np.int64)
    open_times_ns = minute_frame["open_time"].astype("int64").to_numpy(dtype=np.int64)
    close_prices = pd.to_numeric(minute_frame["close"], errors="coerce").to_numpy(dtype=float)
    high_prices = pd.to_numeric(minute_frame["high"], errors="coerce").to_numpy(dtype=float)
    low_prices = pd.to_numeric(minute_frame["low"], errors="coerce").to_numpy(dtype=float)

    rows = []
    for sequence, (_, entry) in enumerate(entries.iterrows(), start=1):
        entry_time = pd.to_datetime(entry["entry_time"], utc=True)
        entry_price = pd.to_numeric(entry["entry_price"], errors="coerce")
        side = str(entry["side"])
        signal_id = entry.get("signal_id", f"{id_prefix}_{sequence}")

        if pd.isna(entry_time) or pd.isna(entry_price) or entry_price == 0:
            rows.append(minimal_signal_metrics(signal_id, entry_time, entry_price, side))
            continue

        metrics = compute_profile_for_entry(
            close_times_ns,
            open_times_ns,
            close_prices,
            high_prices,
            low_prices,
            entry_time,
            float(entry_price),
            side,
        )
        metrics["signal_id"] = signal_id
        metrics["entry_time"] = entry_time
        metrics["entry_price"] = float(entry_price)
        metrics["side"] = side
        rows.append(metrics)

    return pd.DataFrame(rows)


def minimal_signal_metrics(signal_id: str, entry_time: pd.Timestamp, entry_price: float, side: str) -> dict:
    row = {
        "signal_id": signal_id,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "side": side,
        "MFE_pct": np.nan,
        "MAE_pct": np.nan,
        "MFE_price": np.nan,
        "MAE_price": np.nan,
        "time_to_first_positive_minutes": np.nan,
        "time_to_MFE_minutes": np.nan,
        "time_to_MAE_minutes": np.nan,
    }
    for horizon in REPLICATION_HORIZONS_MINUTES:
        row[f"return_after_{horizon}m_pct"] = np.nan
    return row


def compute_profile_for_entry(
    close_times_ns: np.ndarray,
    open_times_ns: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    entry_time: pd.Timestamp,
    entry_price: float,
    side: str,
) -> dict:
    entry_time_ns = int(entry_time.value)
    start_close_idx = int(np.searchsorted(close_times_ns, entry_time_ns, side="left"))
    if start_close_idx >= len(close_times_ns):
        return minimal_signal_metrics("unknown", entry_time, entry_price, side)

    evaluation_end_ns = entry_time_ns + int(pd.Timedelta(hours=24).value)
    end_close_idx = int(np.searchsorted(close_times_ns, evaluation_end_ns, side="right") - 1)
    if end_close_idx < start_close_idx:
        return minimal_signal_metrics("unknown", entry_time, entry_price, side)

    close_window = close_prices[start_close_idx : end_close_idx + 1]
    close_time_window = close_times_ns[start_close_idx : end_close_idx + 1]
    close_minutes = (close_time_window - entry_time_ns) / 60_000_000_000
    close_returns = signed_returns(entry_price, close_window, side)

    intrabar_start_idx = int(np.searchsorted(open_times_ns, entry_time_ns, side="left"))
    intrabar_end_idx = int(np.searchsorted(open_times_ns, evaluation_end_ns, side="right") - 1)
    if intrabar_end_idx >= intrabar_start_idx:
        high_window = high_prices[intrabar_start_idx : intrabar_end_idx + 1]
        low_window = low_prices[intrabar_start_idx : intrabar_end_idx + 1]
        intrabar_times_ns = close_times_ns[intrabar_start_idx : intrabar_end_idx + 1]
        intrabar_minutes = (intrabar_times_ns - entry_time_ns) / 60_000_000_000
        if side.lower() == "long":
            favorable_intrabar = (high_window - entry_price) / entry_price * 100.0
            adverse_intrabar = (entry_price - low_window) / entry_price * 100.0
            favorable_prices = high_window
            adverse_prices = low_window
        else:
            favorable_intrabar = (entry_price - low_window) / entry_price * 100.0
            adverse_intrabar = (high_window - entry_price) / entry_price * 100.0
            favorable_prices = low_window
            adverse_prices = high_window
    else:
        intrabar_minutes = np.array([], dtype=float)
        favorable_intrabar = np.array([], dtype=float)
        adverse_intrabar = np.array([], dtype=float)
        favorable_prices = np.array([], dtype=float)
        adverse_prices = np.array([], dtype=float)

    return_rows = {}
    for horizon in REPLICATION_HORIZONS_MINUTES:
        target_ns = entry_time_ns + int(pd.Timedelta(minutes=horizon).value)
        horizon_idx = int(np.searchsorted(close_times_ns, target_ns, side="left"))
        return_rows[f"return_after_{horizon}m_pct"] = (
            float(signed_returns(entry_price, np.array([close_prices[horizon_idx]]), side)[0])
            if horizon_idx <= end_close_idx
            else np.nan
        )

    favorable_close = np.maximum(close_returns, 0.0)
    adverse_close = np.maximum(-close_returns, 0.0)
    all_favorable = np.concatenate([favorable_close, np.maximum(favorable_intrabar, 0.0)])
    all_favorable_prices = np.concatenate([close_window, favorable_prices])
    all_favorable_minutes = np.concatenate([close_minutes, intrabar_minutes])
    all_adverse = np.concatenate([adverse_close, np.maximum(adverse_intrabar, 0.0)])
    all_adverse_prices = np.concatenate([close_window, adverse_prices])
    all_adverse_minutes = np.concatenate([close_minutes, intrabar_minutes])

    mfe_idx = int(np.nanargmax(all_favorable)) if all_favorable.size else 0
    mae_idx = int(np.nanargmax(all_adverse)) if all_adverse.size else 0
    positive_indices = np.where(close_returns > 0)[0]
    time_to_first_positive = float(close_minutes[positive_indices[0]]) if positive_indices.size else np.nan

    return {
        **return_rows,
        "MFE_pct": float(all_favorable[mfe_idx]) if all_favorable.size else np.nan,
        "MAE_pct": float(all_adverse[mae_idx]) if all_adverse.size else np.nan,
        "MFE_price": float(all_favorable_prices[mfe_idx]) if all_favorable.size else np.nan,
        "MAE_price": float(all_adverse_prices[mae_idx]) if all_adverse.size else np.nan,
        "time_to_first_positive_minutes": time_to_first_positive,
        "time_to_MFE_minutes": float(all_favorable_minutes[mfe_idx]) if all_favorable.size else np.nan,
        "time_to_MAE_minutes": float(all_adverse_minutes[mae_idx]) if all_adverse.size else np.nan,
    }


def signed_returns(entry_price: float, prices: np.ndarray, side: str) -> np.ndarray:
    if side.lower() == "long":
        return (prices - entry_price) / entry_price * 100.0
    return (entry_price - prices) / entry_price * 100.0


def stage_membership_flag(entry_times: pd.Series, stage: ReplicationStage) -> pd.Series:
    if stage.effective_start is None or stage.effective_end is None:
        return pd.Series(False, index=entry_times.index)
    return entry_times.between(stage.effective_start, stage.effective_end, inclusive="both")


def build_stage_summary(
    cluster_id: int,
    cluster_desc: str,
    rule: dict,
    stage: ReplicationStage,
    stage_signals: pd.DataFrame,
    real_profile: dict,
) -> dict:
    real_summary = real_profile.get("summary", {})
    signal_summary = summarize_signal_like_profiles(stage_signals)
    similarity = compare_profiles(real_summary, signal_summary)
    month_span = (
        (stage.effective_end - stage.effective_start).total_seconds() / (30.4375 * 24 * 3600)
        if stage.effective_start is not None and stage.effective_end is not None
        else np.nan
    )

    return {
        "cluster_id": cluster_id,
        "cluster_description": cluster_desc,
        "rule_text": " AND ".join(condition["text"] for condition in rule.get("conditions", [])),
        "stage_id": stage.stage_id,
        "stage_label": stage.label,
        "stage_sort_order": {"stage_a": 1, "stage_b": 2, "stage_c": 3}.get(stage.stage_id, 99),
        "requested_start_utc": stage.requested_start,
        "requested_end_utc": stage.requested_end,
        "effective_start_utc": stage.effective_start,
        "effective_end_utc": stage.effective_end,
        "signal_count": int(len(stage_signals)),
        "signals_per_month": safe_divide(len(stage_signals), month_span),
        "median_signal_gap_hours": signal_summary.get("median_signal_gap_hours"),
        "clustered_signal_share_lt_1h": signal_summary.get("clustered_signal_share_lt_1h"),
        "dominant_signal_hour_utc": signal_summary.get("dominant_signal_hour_utc"),
        "real_signal_count_reference": int(real_summary.get("signal_count", 0)),
        "real_pct_positive_after_1m": real_summary.get("pct_positive_after_1m"),
        "real_pct_positive_after_5m": real_summary.get("pct_positive_after_5m"),
        "real_pct_positive_after_15m": real_summary.get("pct_positive_after_15m"),
        "real_pct_positive_after_30m": real_summary.get("pct_positive_after_30m"),
        "real_pct_positive_after_60m": real_summary.get("pct_positive_after_60m"),
        "real_pct_positive_after_240m": real_summary.get("pct_positive_after_240m"),
        "real_pct_positive_after_720m": real_summary.get("pct_positive_after_720m"),
        "real_pct_positive_after_1440m": real_summary.get("pct_positive_after_1440m"),
        "real_median_MAE_pct": real_summary.get("median_MAE_pct"),
        "real_median_MFE_pct": real_summary.get("median_MFE_pct"),
        "real_median_return_after_60m_pct": real_summary.get("median_return_after_60m_pct"),
        "real_median_return_after_240m_pct": real_summary.get("median_return_after_240m_pct"),
        "real_median_return_after_720m_pct": real_summary.get("median_return_after_720m_pct"),
        "real_median_return_after_1440m_pct": real_summary.get("median_return_after_1440m_pct"),
        "pct_positive_after_1m": signal_summary.get("pct_positive_after_1m"),
        "pct_positive_after_5m": signal_summary.get("pct_positive_after_5m"),
        "pct_positive_after_15m": signal_summary.get("pct_positive_after_15m"),
        "pct_positive_after_30m": signal_summary.get("pct_positive_after_30m"),
        "pct_positive_after_60m": signal_summary.get("pct_positive_after_60m"),
        "pct_positive_after_240m": signal_summary.get("pct_positive_after_240m"),
        "pct_positive_after_720m": signal_summary.get("pct_positive_after_720m"),
        "pct_positive_after_1440m": signal_summary.get("pct_positive_after_1440m"),
        "median_MAE_pct": signal_summary.get("median_MAE_pct"),
        "median_MFE_pct": signal_summary.get("median_MFE_pct"),
        "median_return_after_60m_pct": signal_summary.get("median_return_after_60m_pct"),
        "median_return_after_240m_pct": signal_summary.get("median_return_after_240m_pct"),
        "median_return_after_720m_pct": signal_summary.get("median_return_after_720m_pct"),
        "median_return_after_1440m_pct": signal_summary.get("median_return_after_1440m_pct"),
        "median_time_to_first_profit_minutes": signal_summary.get("median_time_to_first_profit_minutes"),
        "median_time_to_MFE_minutes": signal_summary.get("median_time_to_MFE_minutes"),
        "fast_positive_reaction_share": signal_summary.get("fast_positive_reaction_share"),
        "similarity_score": similarity["score"],
        "similarity_label": similarity["label"],
        "similarity_comment": similarity["comment"],
    }


def summarize_signal_like_profiles(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {
            "signal_count": 0,
            "median_signal_gap_hours": np.nan,
            "clustered_signal_share_lt_1h": np.nan,
            "dominant_signal_hour_utc": np.nan,
            "pct_positive_after_1m": np.nan,
            "pct_positive_after_5m": np.nan,
            "pct_positive_after_15m": np.nan,
            "pct_positive_after_30m": np.nan,
            "pct_positive_after_60m": np.nan,
            "pct_positive_after_240m": np.nan,
            "pct_positive_after_720m": np.nan,
            "pct_positive_after_1440m": np.nan,
            "median_MAE_pct": np.nan,
            "median_MFE_pct": np.nan,
            "median_return_after_60m_pct": np.nan,
            "median_return_after_240m_pct": np.nan,
            "median_return_after_720m_pct": np.nan,
            "median_return_after_1440m_pct": np.nan,
            "median_time_to_first_profit_minutes": np.nan,
            "median_time_to_MFE_minutes": np.nan,
            "fast_positive_reaction_share": np.nan,
        }

    ordered = frame.sort_values("entry_time").copy()
    gap_hours = ordered["entry_time"].diff().dt.total_seconds() / 3600.0
    hour_counts = ordered["entry_time"].dt.hour.value_counts()

    summary = {
        "signal_count": int(len(ordered)),
        "median_signal_gap_hours": gap_hours.dropna().median(),
        "clustered_signal_share_lt_1h": (gap_hours < 1).mean() if gap_hours.notna().any() else np.nan,
        "dominant_signal_hour_utc": int(hour_counts.idxmax()) if not hour_counts.empty else np.nan,
        "median_MAE_pct": ordered["MAE_pct"].median(),
        "median_MFE_pct": ordered["MFE_pct"].median(),
        "median_time_to_first_profit_minutes": ordered["time_to_first_positive_minutes"].median(),
        "median_time_to_MFE_minutes": ordered["time_to_MFE_minutes"].median(),
        "fast_positive_reaction_share": ordered["fast_positive_reaction_flag"].astype(float).mean()
        if "fast_positive_reaction_flag" in ordered.columns
        else ((ordered["return_after_1m_pct"] > 0) | (ordered["return_after_5m_pct"] > 0)).astype(float).mean(),
    }

    for horizon in REPLICATION_HORIZONS_MINUTES:
        column = f"return_after_{horizon}m_pct"
        summary[f"pct_positive_after_{horizon}m"] = (ordered[column] > 0).mean() if ordered[column].notna().any() else np.nan
        if horizon in {60, 240, 720, 1440}:
            summary[f"median_return_after_{horizon}m_pct"] = ordered[column].median()
    return summary


def compare_profiles(real_summary: dict, signal_summary: dict) -> dict:
    metric_pairs = [
        ("pct_positive_after_1m", 1.0),
        ("pct_positive_after_5m", 1.0),
        ("pct_positive_after_15m", 1.0),
        ("median_MAE_pct", 1.5),
        ("median_MFE_pct", 1.5),
        ("median_return_after_60m_pct", 1.0),
        ("median_return_after_240m_pct", 1.5),
        ("median_return_after_720m_pct", 2.0),
        ("median_return_after_1440m_pct", 2.0),
    ]

    penalties = []
    for metric, scale in metric_pairs:
        real_value = real_summary.get(metric)
        signal_value = signal_summary.get(metric)
        if pd.isna(real_value) or pd.isna(signal_value):
            continue
        penalties.append(min(abs(signal_value - real_value) / scale, 1.0))

    if not penalties:
        return {
            "score": np.nan,
            "label": "insufficient",
            "comment": "Similarity could not be assessed because the post-entry profile was incomplete.",
        }

    score = max(0.0, 1.0 - float(np.mean(penalties)))
    if score >= 0.75:
        return {
            "score": score,
            "label": "strong",
            "comment": "The filter catches post-entry reactions that are fairly close to the trader's original cluster profile.",
        }
    if score >= 0.55:
        return {
            "score": score,
            "label": "moderate",
            "comment": "The filter catches partly similar reactions, but the match is incomplete.",
        }
    return {
        "score": score,
        "label": "weak",
        "comment": "The filter does not reproduce the original cluster profile closely.",
    }


def build_stage_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    for cluster_id, cluster_rows in summary_df.groupby("cluster_id", sort=True):
        row = {
            "cluster_id": int(cluster_id),
            "cluster_description": cluster_rows["cluster_description"].iloc[0],
        }
        ordered = cluster_rows.sort_values("stage_sort_order")
        if not ordered["similarity_score"].dropna().empty:
            best = ordered.sort_values(["similarity_score", "signal_count"], ascending=[False, False]).iloc[0]
            row["best_stage"] = best["stage_label"]
            row["best_stage_similarity_score"] = best["similarity_score"]
            row["best_stage_similarity_label"] = best["similarity_label"]
        else:
            row["best_stage"] = pd.NA
            row["best_stage_similarity_score"] = np.nan
            row["best_stage_similarity_label"] = "insufficient"

        for _, stage_row in ordered.iterrows():
            suffix = stage_row["stage_id"]
            row[f"{suffix}_signal_count"] = stage_row["signal_count"]
            row[f"{suffix}_signals_per_month"] = stage_row["signals_per_month"]
            row[f"{suffix}_similarity_score"] = stage_row["similarity_score"]
            row[f"{suffix}_similarity_label"] = stage_row["similarity_label"]
            row[f"{suffix}_pct_positive_after_5m"] = stage_row["pct_positive_after_5m"]
            row[f"{suffix}_median_MAE_pct"] = stage_row["median_MAE_pct"]
            row[f"{suffix}_median_return_after_240m_pct"] = stage_row["median_return_after_240m_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_replication_section(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return (
            "ETH REPLICATION HYPOTHESIS TEST\n"
            "-------------------------------\n"
            "ETH replication hypothesis test could not be completed.\n"
        )

    ordered = summary_df.sort_values(["cluster_id", "stage_sort_order"]).reset_index(drop=True)
    cluster_1 = ordered.loc[ordered["cluster_id"] == 1]
    cluster_2 = ordered.loc[ordered["cluster_id"] == 2]

    stage_a_good = False
    if not cluster_1.empty and not cluster_2.empty:
        c1_a = cluster_1.loc[cluster_1["stage_id"] == "stage_a"]
        c2_a = cluster_2.loc[cluster_2["stage_id"] == "stage_a"]
        if not c1_a.empty and not c2_a.empty:
            stage_a_good = bool(
                (c1_a.iloc[0]["similarity_score"] >= 0.60)
                and (c2_a.iloc[0]["similarity_score"] >= 0.55)
                and (c1_a.iloc[0]["signal_count"] >= 10)
                and (c2_a.iloc[0]["signal_count"] >= 10)
            )

    strongest_row = ordered.dropna(subset=["similarity_score"]).sort_values(
        ["similarity_score", "signal_count"],
        ascending=[False, False],
    )
    strongest_stage = strongest_row.iloc[0]["stage_label"] if not strongest_row.empty else "n/a"
    confidence = determine_replication_confidence(ordered)

    cluster_lines = []
    for _, row in ordered.iterrows():
        cluster_lines.append(
            f"- {row['cluster_description']} / {row['stage_label']}: {fmt_int(row['signal_count'])} signals, "
            f"{pct(row['pct_positive_after_5m'])} positive after 5m, median MAE {fmt_number(row['median_MAE_pct'])}%, "
            f"median 4h move {fmt_number(row['median_return_after_240m_pct'])}%, similarity {fmt_number(row['similarity_score'])} ({row['similarity_label']})."
        )

    exploratory = (
        "Stage A looks strong enough to justify a next-step exploratory TP/SL backtest on the replication filters."
        if stage_a_good
        else "Stage A does not yet look strong enough for a confident TP/SL backtest; the safer next step is more filter validation first."
    )

    report = f"""
ETH REPLICATION HYPOTHESIS TEST
-------------------------------
This stage scans ETHUSDT market history independently of the trader's actual entries. It checks whether the rough context filters catch post-entry reactions that resemble the trader's real ETH clusters. It is still not a full strategy backtest.

Stage-by-stage results:
{chr(10).join(cluster_lines)}

Interpretation:
- Cluster 1 filter similarity is strongest in {best_stage_label(cluster_1)}.
- Cluster 2 filter similarity is strongest in {best_stage_label(cluster_2)}.
- The strongest overall similarity appears in {strongest_stage}.
- Stage A is the main anchor because it matches the trader's original ETH trading window.
- Outside the exact trading period, filter behavior {"holds up reasonably" if survives_outside_period(ordered) else "weakens materially"}, so external robustness should be treated with caution.
- Confidence level: {confidence}.

Exploratory verdict:
- {exploratory}
""".strip()
    return report + "\n"


def best_stage_label(cluster_rows: pd.DataFrame) -> str:
    if cluster_rows.empty or cluster_rows["similarity_score"].dropna().empty:
        return "n/a"
    best = cluster_rows.sort_values(["similarity_score", "signal_count"], ascending=[False, False]).iloc[0]
    return str(best["stage_label"])


def survives_outside_period(summary_df: pd.DataFrame) -> bool:
    external = summary_df.loc[summary_df["stage_id"].isin(["stage_b", "stage_c"]), "similarity_score"].dropna()
    if external.empty:
        return False
    return bool((external >= 0.50).mean() >= 0.50)


def determine_replication_confidence(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "low"
    stage_a = summary_df.loc[summary_df["stage_id"] == "stage_a"]
    if stage_a.empty:
        return "low"

    min_stage_a_similarity = stage_a["similarity_score"].dropna().min() if stage_a["similarity_score"].notna().any() else np.nan
    min_stage_a_signals = stage_a["signal_count"].min() if not stage_a.empty else 0
    external = summary_df.loc[summary_df["stage_id"].isin(["stage_b", "stage_c"]), "similarity_score"].dropna()
    external_median = external.median() if not external.empty else np.nan

    if pd.notna(min_stage_a_similarity) and min_stage_a_similarity >= 0.70 and min_stage_a_signals >= 15 and pd.notna(external_median) and external_median >= 0.55:
        return "high"
    if pd.notna(min_stage_a_similarity) and min_stage_a_similarity >= 0.55 and min_stage_a_signals >= 8:
        return "medium"
    return "low"


def create_eth_replication_plots(
    summary_df: pd.DataFrame,
    combined_signals: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if summary_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    post_entry_path = plots_dir / "eth_replication_post_entry_profiles.png"
    density_path = plots_dir / "eth_replication_signal_density.png"

    fig_profiles = plot_replication_post_entry_profiles(summary_df)
    fig_profiles.savefig(post_entry_path, dpi=150, bbox_inches="tight")
    plt.close(fig_profiles)
    logger.info("Saved plot: %s", post_entry_path.name)

    fig_density = plot_replication_signal_density(combined_signals)
    fig_density.savefig(density_path, dpi=150, bbox_inches="tight")
    plt.close(fig_density)
    logger.info("Saved plot: %s", density_path.name)


def plot_replication_post_entry_profiles(summary_df: pd.DataFrame) -> plt.Figure:
    horizons = [1, 5, 15, 30, 60, 240, 720, 1440]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    for ax, cluster_id in zip(axes, [1, 2], strict=False):
        cluster_rows = summary_df.loc[summary_df["cluster_id"] == cluster_id].sort_values("stage_sort_order")
        if cluster_rows.empty:
            ax.set_visible(False)
            continue

        reference = cluster_rows.iloc[0]
        real_profile = [
            reference.get("real_pct_positive_after_1m", np.nan),
            reference.get("real_pct_positive_after_5m", np.nan),
            reference.get("real_pct_positive_after_15m", np.nan),
            reference.get("real_pct_positive_after_30m", np.nan),
            reference.get("real_pct_positive_after_60m", np.nan),
            reference.get("real_pct_positive_after_240m", np.nan),
            reference.get("real_pct_positive_after_720m", np.nan),
            reference.get("real_pct_positive_after_1440m", np.nan),
        ]
        ax.plot(horizons, real_profile, marker="o", linewidth=2.5, color="black", label="Real cluster trades")

        for _, row in cluster_rows.iterrows():
            profile = [
                row.get("pct_positive_after_1m", np.nan),
                row.get("pct_positive_after_5m", np.nan),
                row.get("pct_positive_after_15m", np.nan),
                row.get("pct_positive_after_30m", np.nan),
                row.get("pct_positive_after_60m", np.nan),
                row.get("pct_positive_after_240m", np.nan),
                row.get("pct_positive_after_720m", np.nan),
                row.get("pct_positive_after_1440m", np.nan),
            ]
            ax.plot(
                horizons,
                profile,
                marker="o",
                linewidth=1.8,
                color=STAGE_COLORS.get(str(row["stage_id"]), "#555555"),
                label=row["stage_label"],
            )

        ax.set_title(f"Cluster {cluster_id} post-entry profile")
        ax.set_xlabel("Minutes after signal")
        ax.set_ylabel("Share positive")
        ax.set_xscale("symlog", linthresh=15)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("ETH Replication Filters: Post-entry Reaction vs Real Cluster Trades")
    return fig


def plot_replication_signal_density(combined_signals: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    if combined_signals.empty:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No signals", ha="center", va="center")
            ax.axis("off")
        return fig

    for row_idx, cluster_id in enumerate([1, 2]):
        cluster_signals = combined_signals.loc[combined_signals["cluster_id"] == cluster_id].copy()
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]

        for stage_id in ["stage_a", "stage_b", "stage_c"]:
            flag_column = f"in_{stage_id}"
            if flag_column not in cluster_signals.columns:
                continue
            stage_signals = cluster_signals.loc[cluster_signals[flag_column]].copy()
            if stage_signals.empty:
                continue

            month_index = (
                pd.to_datetime(stage_signals["entry_time"], utc=True)
                .dt.tz_localize(None)
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            monthly_counts = (
                stage_signals.assign(month=month_index)
                .groupby("month")
                .size()
            )
            left_ax.plot(
                monthly_counts.index,
                monthly_counts.values,
                marker="o",
                linewidth=1.6,
                label=stage_id.replace("_", " ").title(),
                color=STAGE_COLORS.get(stage_id, "#555555"),
            )

            hour_counts = stage_signals["entry_time"].dt.hour.value_counts().sort_index()
            right_ax.plot(
                hour_counts.index,
                hour_counts.values,
                marker="o",
                linewidth=1.6,
                label=stage_id.replace("_", " ").title(),
                color=STAGE_COLORS.get(stage_id, "#555555"),
            )

        left_ax.set_title(f"Cluster {cluster_id} signals by month")
        left_ax.set_xlabel("Month")
        left_ax.set_ylabel("Signal count")
        left_ax.grid(alpha=0.25)
        left_ax.legend()

        right_ax.set_title(f"Cluster {cluster_id} signals by hour UTC")
        right_ax.set_xlabel("Hour UTC")
        right_ax.set_ylabel("Signal count")
        right_ax.set_xticks(range(0, 24, 2))
        right_ax.grid(alpha=0.25)
        right_ax.legend()

    fig.suptitle("ETH Replication Filters: Signal Density")
    return fig
