from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_replication_hypothesis_test import (
    ETH_SYMBOL,
    ReplicationStage,
    build_stage_windows,
    compare_profiles,
    evaluate_entry_profiles,
    materialize_stage_bounds,
    summarize_signal_like_profiles,
)
from market import fetch_futures_klines
from market_context_at_entry import prepare_context_features
from utils import fmt_int, fmt_number, pct, safe_divide


@dataclass(frozen=True)
class ExitRule:
    rule_id: str
    label: str
    rule_type: str
    horizon_minutes: int | None = None
    tp_atr: float | None = None
    sl_atr: float | None = None
    tp_pct: float | None = None
    sl_pct: float | None = None
    time_stop_minutes: int | None = None


PROXY_SETUP_SPECS = [
    {
        "setup_name": "eth_short_proxy",
        "setup_label": "ETH Short proxy setup",
        "side": "Short",
        "reference_cluster_id": 1,
        "archetype_label": "Cluster 1 short family",
        "spirit_label": "short after small bounce near upper structure",
        "context_text": "distance to 4h high <= 0.60% AND 60m return >= 0.45%",
        "trigger_text": "5m failed continuation candle: upper wick >= 0.25, 3-candle 5m push > 0, last 5m return <= 0",
    },
    {
        "setup_name": "eth_long_proxy",
        "setup_label": "ETH Long proxy setup",
        "side": "Long",
        "reference_cluster_id": 2,
        "archetype_label": "Cluster 2 long family",
        "spirit_label": "long after local flush near lower structure",
        "context_text": "distance to EMA20 5m <= -0.50% AND distance to 4h low <= 0.55%",
        "trigger_text": "5m stabilization candle: lower wick >= 0.25, 3-candle 5m move < 0, last 5m return >= 0",
    },
]

EXIT_RULES = [
    ExitRule("fixed_1h", "Fixed horizon 1h", "fixed_horizon", horizon_minutes=60),
    ExitRule("fixed_4h", "Fixed horizon 4h", "fixed_horizon", horizon_minutes=240),
    ExitRule("fixed_12h", "Fixed horizon 12h", "fixed_horizon", horizon_minutes=720),
    ExitRule("fixed_24h", "Fixed horizon 24h", "fixed_horizon", horizon_minutes=1440),
    ExitRule("atr_tp1_sl1_time24h", "TP 1 ATR / SL 1 ATR / time 24h", "atr", tp_atr=1.0, sl_atr=1.0, time_stop_minutes=1440),
    ExitRule("atr_tp1p5_sl1_time24h", "TP 1.5 ATR / SL 1 ATR / time 24h", "atr", tp_atr=1.5, sl_atr=1.0, time_stop_minutes=1440),
    ExitRule("react_tp0p75_sl0p75_time12h", "TP 0.75% / SL 0.75% / time 12h", "pct", tp_pct=0.75, sl_pct=0.75, time_stop_minutes=720),
    ExitRule("react_tp1_sl0p75_time12h", "TP 1.0% / SL 0.75% / time 12h", "pct", tp_pct=1.0, sl_pct=0.75, time_stop_minutes=720),
]


def run_eth_practical_imitation_model(
    trades_df: pd.DataFrame,
    cluster_frame: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "PRACTICAL IMITATION MODEL\n"
        "-------------------------\n"
        "No sufficient ETH data was available to build a practical imitation model.\n"
    )

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    eth_clusters = cluster_frame.loc[cluster_frame["symbol"] == ETH_SYMBOL].copy()
    if eth_trades.empty or eth_clusters.empty:
        return {
            "short_results": empty,
            "long_results": empty,
            "stage_comparison": empty,
            "exit_comparison": empty,
            "section_text": empty_section,
        }

    stages = build_stage_windows(eth_trades)
    fetch_start = min(stage.requested_start for stage in stages) - pd.Timedelta(hours=48)
    fetch_end = min(
        max(stage.requested_end for stage in stages) + pd.Timedelta(hours=25),
        pd.Timestamp.now(tz="UTC"),
    )

    logger.info(
        "Building practical imitation model on ETHUSDT from %s to %s",
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
            "PRACTICAL IMITATION MODEL\n"
            "-------------------------\n"
            "ETH proxy setup testing could not run because ETH candle data was unavailable.\n"
        )
        return {
            "short_results": empty,
            "long_results": empty,
            "stage_comparison": empty,
            "exit_comparison": empty,
            "section_text": section,
        }

    scan_frame = build_proxy_scan_frame(frame_5m)
    if scan_frame.empty:
        return {
            "short_results": empty,
            "long_results": empty,
            "stage_comparison": empty,
            "exit_comparison": empty,
            "section_text": (
                "PRACTICAL IMITATION MODEL\n"
                "-------------------------\n"
                "ETH proxy scan frame was empty, so no imitation model could be tested.\n"
            ),
        }

    max_signal_entry_time = scan_frame["entry_time"].max() - pd.Timedelta(hours=24)
    min_signal_entry_time = scan_frame["entry_time"].min()
    materialize_stage_bounds(stages, min_signal_entry_time, max_signal_entry_time)

    reference_profiles = build_reference_cluster_profiles(eth_trades, eth_clusters, minute_frame)
    minute_arrays = build_minute_arrays(minute_frame)

    setup_results: list[pd.DataFrame] = []
    stage_rows: list[dict] = []

    for spec in PROXY_SETUP_SPECS:
        raw_signals = build_proxy_signal_frame(scan_frame, minute_frame, spec, stages)
        if raw_signals.empty:
            setup_results.append(pd.DataFrame())
            continue

        reference_summary = reference_profiles.get(spec["reference_cluster_id"], {}).get("summary", {})
        setup_stage_rows, setup_executed = evaluate_proxy_setup(
            raw_signals,
            minute_arrays,
            stages,
            spec,
            reference_summary,
        )
        stage_rows.extend(setup_stage_rows)
        setup_results.append(setup_executed)

    stage_comparison = pd.DataFrame(stage_rows)
    exit_comparison = build_proxy_exit_comparison(stage_comparison)
    stage_comparison = mark_recommended_exits(stage_comparison, exit_comparison)

    short_results = combine_setup_results(setup_results, "eth_short_proxy")
    long_results = combine_setup_results(setup_results, "eth_long_proxy")
    section_text = build_practical_imitation_section(stage_comparison, exit_comparison)

    return {
        "short_results": short_results,
        "long_results": long_results,
        "stage_comparison": stage_comparison,
        "exit_comparison": exit_comparison,
        "section_text": section_text,
    }


def build_proxy_scan_frame(frame_5m: pd.DataFrame) -> pd.DataFrame:
    if frame_5m.empty:
        return pd.DataFrame()

    prepared = prepare_context_features(frame_5m, "5m").copy().sort_values("close_time").reset_index(drop=True)
    close_series = pd.to_numeric(prepared["close"], errors="coerce")
    open_series = pd.to_numeric(prepared["open"], errors="coerce")
    high_series = pd.to_numeric(prepared["high"], errors="coerce")
    low_series = pd.to_numeric(prepared["low"], errors="coerce")
    candle_range = (high_series - low_series).replace(0, np.nan)

    prepared["entry_time"] = prepared["close_time"]
    prepared["entry_price"] = close_series
    prepared["open_hour_utc"] = prepared["entry_time"].dt.hour
    prepared["return_pre_60m_pct"] = close_series.pct_change(12) * 100.0
    prepared["prev_4h_high"] = high_series.shift(1).rolling(48, min_periods=8).max()
    prepared["prev_4h_low"] = low_series.shift(1).rolling(48, min_periods=8).min()
    prepared["distance_to_last_4h_high_pct"] = (prepared["prev_4h_high"] - close_series) / close_series * 100.0
    prepared["distance_to_last_4h_low_pct"] = (close_series - prepared["prev_4h_low"]) / close_series * 100.0
    prepared["upper_wick_ratio_5m"] = (high_series - np.maximum(open_series, close_series)) / candle_range
    prepared["lower_wick_ratio_5m"] = (np.minimum(open_series, close_series) - low_series) / candle_range
    prepared["ret_last_5m_pct"] = close_series.pct_change(1) * 100.0
    prepared["ret_3c_5m_pct"] = close_series.pct_change(3) * 100.0

    keep_columns = [
        "entry_time",
        "entry_price",
        "open_hour_utc",
        "close_time",
        "distance_to_ema20_pct",
        "distance_to_last_4h_high_pct",
        "distance_to_last_4h_low_pct",
        "return_pre_60m_pct",
        "upper_wick_ratio_5m",
        "lower_wick_ratio_5m",
        "ret_last_5m_pct",
        "ret_3c_5m_pct",
        "atr14",
        "atr14_pct",
        "relative_volume",
    ]
    return prepared[keep_columns].copy()


def build_reference_cluster_profiles(
    eth_trades: pd.DataFrame,
    eth_clusters: pd.DataFrame,
    minute_frame: pd.DataFrame,
) -> dict[int, dict]:
    merged = eth_clusters.merge(
        eth_trades[["trade_number", "avgCost"]],
        on="trade_number",
        how="left",
    )

    profiles: dict[int, dict] = {}
    for cluster_id in [1, 2]:
        cluster_rows = merged.loc[merged["cluster_id"] == cluster_id].copy()
        if cluster_rows.empty:
            profiles[cluster_id] = {"metrics": pd.DataFrame(), "summary": {}}
            continue
        metrics = evaluate_entry_profiles(
            minute_frame,
            cluster_rows.rename(columns={"avgCost": "entry_price", "opened_dt_utc": "entry_time"})[
                ["trade_number", "entry_time", "entry_price", "side"]
            ],
            id_prefix=f"real_cluster_{cluster_id}",
        )
        profiles[cluster_id] = {
            "metrics": metrics,
            "summary": summarize_signal_like_profiles(metrics),
        }
    return profiles


def build_proxy_signal_frame(
    scan_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    spec: dict,
    stages: list[ReplicationStage],
) -> pd.DataFrame:
    mask = proxy_setup_mask(scan_frame, spec["setup_name"])
    start_mask = mask & ~mask.shift(1, fill_value=False)
    starts = scan_frame.loc[start_mask].copy()
    if starts.empty:
        return pd.DataFrame()

    starts["setup_name"] = spec["setup_name"]
    starts["setup_label"] = spec["setup_label"]
    starts["side"] = spec["side"]
    starts["reference_cluster_id"] = spec["reference_cluster_id"]
    starts["archetype_label"] = spec["archetype_label"]
    starts["spirit_label"] = spec["spirit_label"]
    starts["context_text"] = spec["context_text"]
    starts["trigger_text"] = spec["trigger_text"]
    starts["signal_id"] = starts["entry_time"].dt.strftime(spec["setup_name"] + "_%Y%m%dT%H%M%S")

    metrics = evaluate_entry_profiles(
        minute_frame,
        starts[["signal_id", "entry_time", "entry_price", "side"]],
        id_prefix=spec["setup_name"],
    )
    merged = starts.merge(metrics, on=["signal_id", "entry_time", "entry_price", "side"], how="left")

    for stage in stages:
        if stage.effective_start is None or stage.effective_end is None:
            merged[f"in_{stage.stage_id}"] = False
        else:
            merged[f"in_{stage.stage_id}"] = merged["entry_time"].between(stage.effective_start, stage.effective_end, inclusive="both")
    return merged.sort_values("entry_time").reset_index(drop=True)


def proxy_setup_mask(scan_frame: pd.DataFrame, setup_name: str) -> pd.Series:
    if setup_name == "eth_short_proxy":
        return (
            (scan_frame["distance_to_last_4h_high_pct"] <= 0.60)
            & (scan_frame["return_pre_60m_pct"] >= 0.45)
            & (scan_frame["upper_wick_ratio_5m"] >= 0.25)
            & (scan_frame["ret_3c_5m_pct"] > 0)
            & (scan_frame["ret_last_5m_pct"] <= 0)
        ).fillna(False)
    if setup_name == "eth_long_proxy":
        return (
            (scan_frame["distance_to_ema20_pct"] <= -0.50)
            & (scan_frame["distance_to_last_4h_low_pct"] <= 0.55)
            & (scan_frame["lower_wick_ratio_5m"] >= 0.25)
            & (scan_frame["ret_3c_5m_pct"] < 0)
            & (scan_frame["ret_last_5m_pct"] >= 0)
        ).fillna(False)
    return pd.Series(False, index=scan_frame.index)


def build_minute_arrays(minute_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = minute_frame.sort_values("open_time").reset_index(drop=True)
    return {
        "open_time_ns": ordered["open_time"].astype("int64").to_numpy(dtype=np.int64),
        "close_time_ns": ordered["close_time"].astype("int64").to_numpy(dtype=np.int64),
        "open": pd.to_numeric(ordered["open"], errors="coerce").to_numpy(dtype=float),
        "high": pd.to_numeric(ordered["high"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(ordered["low"], errors="coerce").to_numpy(dtype=float),
        "close": pd.to_numeric(ordered["close"], errors="coerce").to_numpy(dtype=float),
    }


def evaluate_proxy_setup(
    raw_signals: pd.DataFrame,
    minute_arrays: dict[str, np.ndarray],
    stages: list[ReplicationStage],
    spec: dict,
    reference_summary: dict,
) -> tuple[list[dict], pd.DataFrame]:
    stage_rows: list[dict] = []
    executed_frames: list[pd.DataFrame] = []

    for exit_rule in EXIT_RULES:
        for stage in stages:
            stage_signals = raw_signals.loc[raw_signals[f"in_{stage.stage_id}"] == True].copy()
            stage_signal_summary = summarize_signal_like_profiles(stage_signals)
            entry_similarity = compare_profiles(reference_summary, stage_signal_summary)

            executed = simulate_non_overlapping_trades(stage_signals, minute_arrays, exit_rule)
            if not executed.empty:
                executed["setup_name"] = spec["setup_name"]
                executed["setup_label"] = spec["setup_label"]
                executed["stage_id"] = stage.stage_id
                executed["stage_label"] = stage.label
                executed["exit_rule_id"] = exit_rule.rule_id
                executed["exit_rule_label"] = exit_rule.label
                executed["exit_rule_type"] = exit_rule.rule_type
                executed["reference_cluster_id"] = spec["reference_cluster_id"]
                executed["archetype_label"] = spec["archetype_label"]
                executed_frames.append(executed)

            stage_rows.append(
                build_proxy_stage_row(
                    spec,
                    stage,
                    exit_rule,
                    stage_signals,
                    executed,
                    entry_similarity,
                )
            )

    executed_df = pd.concat(executed_frames, ignore_index=True, sort=False) if executed_frames else pd.DataFrame()
    return stage_rows, executed_df


def simulate_non_overlapping_trades(
    stage_signals: pd.DataFrame,
    minute_arrays: dict[str, np.ndarray],
    exit_rule: ExitRule,
) -> pd.DataFrame:
    if stage_signals.empty:
        return pd.DataFrame()

    executed_rows = []
    current_exit_time = pd.Timestamp("1970-01-01", tz="UTC")
    for _, signal in stage_signals.sort_values("entry_time").iterrows():
        entry_time = pd.to_datetime(signal["entry_time"], utc=True)
        if entry_time < current_exit_time:
            continue
        executed = simulate_single_trade(signal, minute_arrays, exit_rule)
        if executed is None:
            continue
        executed_rows.append(executed)
        current_exit_time = executed["exit_time"]
    return pd.DataFrame(executed_rows)


def simulate_single_trade(
    signal: pd.Series,
    minute_arrays: dict[str, np.ndarray],
    exit_rule: ExitRule,
) -> dict | None:
    entry_time = pd.to_datetime(signal["entry_time"], utc=True)
    entry_price = pd.to_numeric(signal["entry_price"], errors="coerce")
    side = str(signal["side"])
    if pd.isna(entry_time) or pd.isna(entry_price) or entry_price == 0:
        return None

    if exit_rule.rule_type == "fixed_horizon":
        exit_result = resolve_fixed_exit(entry_time, float(entry_price), minute_arrays, exit_rule.horizon_minutes or 0)
    elif exit_rule.rule_type == "atr":
        atr_value = pd.to_numeric(signal.get("atr14"), errors="coerce")
        if pd.isna(atr_value) or atr_value <= 0:
            return None
        exit_result = resolve_barrier_exit(
            entry_time=entry_time,
            side=side,
            minute_arrays=minute_arrays,
            tp_price=compute_tp_price(float(entry_price), side, float(atr_value) * float(exit_rule.tp_atr or 0)),
            sl_price=compute_sl_price(float(entry_price), side, float(atr_value) * float(exit_rule.sl_atr or 0)),
            time_stop_minutes=exit_rule.time_stop_minutes or 0,
            exit_rule_label=exit_rule.label,
        )
    else:
        exit_result = resolve_barrier_exit(
            entry_time=entry_time,
            side=side,
            minute_arrays=minute_arrays,
            tp_price=compute_pct_target_price(float(entry_price), side, float(exit_rule.tp_pct or 0)),
            sl_price=compute_pct_stop_price(float(entry_price), side, float(exit_rule.sl_pct or 0)),
            time_stop_minutes=exit_rule.time_stop_minutes or 0,
            exit_rule_label=exit_rule.label,
        )

    if exit_result is None:
        return None

    exit_time = exit_result["exit_time"]
    exit_price = exit_result["exit_price"]
    open_idx_start = exit_result["open_idx_start"]
    open_idx_end = exit_result["open_idx_end"]

    high_window = minute_arrays["high"][open_idx_start : open_idx_end + 1] if open_idx_end >= open_idx_start else np.array([], dtype=float)
    low_window = minute_arrays["low"][open_idx_start : open_idx_end + 1] if open_idx_end >= open_idx_start else np.array([], dtype=float)

    mfe_pct, mae_pct = compute_trade_excursions(float(entry_price), side, high_window, low_window)
    return_pct = signed_return_pct(float(entry_price), float(exit_price), side)

    return {
        "signal_id": signal["signal_id"],
        "entry_time": entry_time,
        "entry_price": float(entry_price),
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "side": side,
        "exit_reason": exit_result["exit_reason"],
        "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
        "return_pct": return_pct,
        "MFE_pct": mfe_pct,
        "MAE_pct": mae_pct,
        "atr14_entry": pd.to_numeric(signal.get("atr14"), errors="coerce"),
        "atr14_pct_entry": pd.to_numeric(signal.get("atr14_pct"), errors="coerce"),
        "raw_signal_relative_volume_5m": pd.to_numeric(signal.get("relative_volume"), errors="coerce"),
        "open_hour_utc": signal.get("open_hour_utc", np.nan),
    }


def resolve_fixed_exit(
    entry_time: pd.Timestamp,
    entry_price: float,
    minute_arrays: dict[str, np.ndarray],
    horizon_minutes: int,
) -> dict | None:
    del entry_price
    entry_ns = int(entry_time.value)
    target_ns = entry_ns + int(pd.Timedelta(minutes=horizon_minutes).value)
    close_times_ns = minute_arrays["close_time_ns"]
    open_times_ns = minute_arrays["open_time_ns"]

    close_idx = int(np.searchsorted(close_times_ns, target_ns, side="left"))
    if close_idx >= len(close_times_ns):
        return None

    open_idx_start = int(np.searchsorted(open_times_ns, entry_ns, side="right"))
    open_idx_end = int(np.searchsorted(open_times_ns, close_times_ns[close_idx], side="right") - 1)
    return {
        "exit_time": pd.Timestamp(close_times_ns[close_idx], tz="UTC"),
        "exit_price": float(minute_arrays["close"][close_idx]),
        "exit_reason": f"time_stop_{horizon_minutes}m",
        "open_idx_start": max(open_idx_start, 0),
        "open_idx_end": max(open_idx_end, open_idx_start),
    }


def resolve_barrier_exit(
    entry_time: pd.Timestamp,
    side: str,
    minute_arrays: dict[str, np.ndarray],
    tp_price: float,
    sl_price: float,
    time_stop_minutes: int,
    exit_rule_label: str,
) -> dict | None:
    entry_ns = int(entry_time.value)
    close_times_ns = minute_arrays["close_time_ns"]
    open_times_ns = minute_arrays["open_time_ns"]
    high_prices = minute_arrays["high"]
    low_prices = minute_arrays["low"]
    close_prices = minute_arrays["close"]

    start_open_idx = int(np.searchsorted(open_times_ns, entry_ns, side="right"))
    if start_open_idx >= len(open_times_ns):
        return None

    time_stop_ns = entry_ns + int(pd.Timedelta(minutes=time_stop_minutes).value)
    last_open_idx = int(np.searchsorted(open_times_ns, time_stop_ns, side="left"))
    last_open_idx = min(last_open_idx, len(open_times_ns) - 1)

    for idx in range(start_open_idx, last_open_idx + 1):
        high_price = high_prices[idx]
        low_price = low_prices[idx]

        if side.lower() == "long":
            tp_hit = high_price >= tp_price
            sl_hit = low_price <= sl_price
        else:
            tp_hit = low_price <= tp_price
            sl_hit = high_price >= sl_price

        if tp_hit and sl_hit:
            return {
                "exit_time": pd.Timestamp(close_times_ns[idx], tz="UTC"),
                "exit_price": sl_price,
                "exit_reason": f"{exit_rule_label} stop-first same candle",
                "open_idx_start": start_open_idx,
                "open_idx_end": idx,
            }
        if sl_hit:
            return {
                "exit_time": pd.Timestamp(close_times_ns[idx], tz="UTC"),
                "exit_price": sl_price,
                "exit_reason": f"{exit_rule_label} stop",
                "open_idx_start": start_open_idx,
                "open_idx_end": idx,
            }
        if tp_hit:
            return {
                "exit_time": pd.Timestamp(close_times_ns[idx], tz="UTC"),
                "exit_price": tp_price,
                "exit_reason": f"{exit_rule_label} target",
                "open_idx_start": start_open_idx,
                "open_idx_end": idx,
            }

    close_idx = int(np.searchsorted(close_times_ns, time_stop_ns, side="left"))
    if close_idx >= len(close_times_ns):
        close_idx = len(close_times_ns) - 1
    open_idx_end = int(np.searchsorted(open_times_ns, close_times_ns[close_idx], side="right") - 1)
    return {
        "exit_time": pd.Timestamp(close_times_ns[close_idx], tz="UTC"),
        "exit_price": float(close_prices[close_idx]),
        "exit_reason": f"{exit_rule_label} time stop",
        "open_idx_start": start_open_idx,
        "open_idx_end": max(open_idx_end, start_open_idx),
    }


def compute_tp_price(entry_price: float, side: str, distance_abs: float) -> float:
    if side.lower() == "long":
        return entry_price + distance_abs
    return entry_price - distance_abs


def compute_sl_price(entry_price: float, side: str, distance_abs: float) -> float:
    if side.lower() == "long":
        return entry_price - distance_abs
    return entry_price + distance_abs


def compute_pct_target_price(entry_price: float, side: str, tp_pct: float) -> float:
    if side.lower() == "long":
        return entry_price * (1 + tp_pct / 100.0)
    return entry_price * (1 - tp_pct / 100.0)


def compute_pct_stop_price(entry_price: float, side: str, sl_pct: float) -> float:
    if side.lower() == "long":
        return entry_price * (1 - sl_pct / 100.0)
    return entry_price * (1 + sl_pct / 100.0)


def compute_trade_excursions(entry_price: float, side: str, high_window: np.ndarray, low_window: np.ndarray) -> tuple[float, float]:
    if high_window.size == 0 or low_window.size == 0:
        return np.nan, np.nan
    if side.lower() == "long":
        favorable = (high_window - entry_price) / entry_price * 100.0
        adverse = (entry_price - low_window) / entry_price * 100.0
    else:
        favorable = (entry_price - low_window) / entry_price * 100.0
        adverse = (high_window - entry_price) / entry_price * 100.0
    return float(np.nanmax(favorable)), float(np.nanmax(adverse))


def signed_return_pct(entry_price: float, exit_price: float, side: str) -> float:
    if side.lower() == "long":
        return (exit_price - entry_price) / entry_price * 100.0
    return (entry_price - exit_price) / entry_price * 100.0


def build_proxy_stage_row(
    spec: dict,
    stage: ReplicationStage,
    exit_rule: ExitRule,
    stage_signals: pd.DataFrame,
    executed: pd.DataFrame,
    entry_similarity: dict,
) -> dict:
    month_span = (
        (stage.effective_end - stage.effective_start).total_seconds() / (30.4375 * 24 * 3600)
        if stage.effective_start is not None and stage.effective_end is not None
        else np.nan
    )
    rough_drawdown = compute_rough_drawdown(executed["return_pct"]) if not executed.empty else np.nan
    cumulative_return = executed["return_pct"].sum() if not executed.empty else np.nan

    return {
        "setup_name": spec["setup_name"],
        "setup_label": spec["setup_label"],
        "side": spec["side"],
        "reference_cluster_id": spec["reference_cluster_id"],
        "archetype_label": spec["archetype_label"],
        "spirit_label": spec["spirit_label"],
        "context_text": spec["context_text"],
        "trigger_text": spec["trigger_text"],
        "stage_id": stage.stage_id,
        "stage_label": stage.label,
        "stage_sort_order": {"stage_a": 1, "stage_b": 2, "stage_c": 3}.get(stage.stage_id, 99),
        "exit_rule_id": exit_rule.rule_id,
        "exit_rule_label": exit_rule.label,
        "exit_rule_type": exit_rule.rule_type,
        "raw_signal_count": int(len(stage_signals)),
        "executed_trade_count": int(len(executed)),
        "signals_per_month": safe_divide(len(stage_signals), month_span),
        "trades_per_month": safe_divide(len(executed), month_span),
        "entry_profile_similarity": entry_similarity.get("score"),
        "entry_profile_similarity_label": entry_similarity.get("label"),
        "win_rate": safe_divide((executed["return_pct"] > 0).sum(), len(executed)),
        "avg_return_pct": executed["return_pct"].mean() if not executed.empty else np.nan,
        "median_return_pct": executed["return_pct"].median() if not executed.empty else np.nan,
        "expectancy_pct": executed["return_pct"].mean() if not executed.empty else np.nan,
        "median_MAE_pct": executed["MAE_pct"].median() if not executed.empty else np.nan,
        "median_MFE_pct": executed["MFE_pct"].median() if not executed.empty else np.nan,
        "rough_max_drawdown_pct": rough_drawdown,
        "cumulative_return_pct": cumulative_return,
    }


def compute_rough_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    cumulative = returns.fillna(0).cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    return float(drawdown.max()) if not drawdown.empty else np.nan


def build_proxy_exit_comparison(stage_comparison: pd.DataFrame) -> pd.DataFrame:
    if stage_comparison.empty:
        return pd.DataFrame()

    rows = []
    for (setup_name, exit_rule_id), group in stage_comparison.groupby(["setup_name", "exit_rule_id"], dropna=False):
        group = group.sort_values("stage_sort_order")
        stage_a = group.loc[group["stage_id"] == "stage_a"]
        stage_b = group.loc[group["stage_id"] == "stage_b"]
        stage_c = group.loc[group["stage_id"] == "stage_c"]
        positive_stage_share = (group["expectancy_pct"] > 0).mean()
        score = proxy_exit_score(group)
        rows.append(
            {
                "setup_name": setup_name,
                "setup_label": group.iloc[0]["setup_label"],
                "side": group.iloc[0]["side"],
                "exit_rule_id": exit_rule_id,
                "exit_rule_label": group.iloc[0]["exit_rule_label"],
                "exit_rule_type": group.iloc[0]["exit_rule_type"],
                "stage_a_executed_trade_count": stage_a["executed_trade_count"].iloc[0] if not stage_a.empty else 0,
                "stage_a_expectancy_pct": stage_a["expectancy_pct"].iloc[0] if not stage_a.empty else np.nan,
                "stage_a_win_rate": stage_a["win_rate"].iloc[0] if not stage_a.empty else np.nan,
                "stage_a_cumulative_return_pct": stage_a["cumulative_return_pct"].iloc[0] if not stage_a.empty else np.nan,
                "stage_a_rough_max_drawdown_pct": stage_a["rough_max_drawdown_pct"].iloc[0] if not stage_a.empty else np.nan,
                "stage_a_entry_profile_similarity": stage_a["entry_profile_similarity"].iloc[0] if not stage_a.empty else np.nan,
                "stage_b_expectancy_pct": stage_b["expectancy_pct"].iloc[0] if not stage_b.empty else np.nan,
                "stage_c_expectancy_pct": stage_c["expectancy_pct"].iloc[0] if not stage_c.empty else np.nan,
                "median_expectancy_pct_across_stages": group["expectancy_pct"].median(),
                "positive_stage_share": positive_stage_share,
                "median_drawdown_pct_across_stages": group["rough_max_drawdown_pct"].median(),
                "median_trades_per_month_across_stages": group["trades_per_month"].median(),
                "recommended_score": score,
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return comparison

    comparison["recommended_exit_flag"] = False
    for setup_name, group in comparison.groupby("setup_name"):
        best_idx = group.sort_values(
            ["recommended_score", "stage_a_expectancy_pct", "positive_stage_share", "stage_a_executed_trade_count"],
            ascending=[False, False, False, False],
        ).index[0]
        comparison.loc[best_idx, "recommended_exit_flag"] = True

    return comparison.sort_values(
        ["setup_name", "recommended_exit_flag", "recommended_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def proxy_exit_score(group: pd.DataFrame) -> float:
    stage_a = group.loc[group["stage_id"] == "stage_a"]
    stage_a_expectancy = stage_a["expectancy_pct"].iloc[0] if not stage_a.empty else np.nan
    stage_a_drawdown = stage_a["rough_max_drawdown_pct"].iloc[0] if not stage_a.empty else np.nan
    stage_a_count = stage_a["executed_trade_count"].iloc[0] if not stage_a.empty else 0
    positive_stage_share = (group["expectancy_pct"] > 0).mean()
    median_expectancy = group["expectancy_pct"].median()
    count_term = min(stage_a_count / 80.0, 1.0)
    expectancy_term = np.tanh(safe_number(stage_a_expectancy) / 0.5)
    robustness_term = np.tanh(safe_number(median_expectancy) / 0.5)
    drawdown_term = -np.tanh(safe_number(stage_a_drawdown) / 8.0)
    return (
        expectancy_term * 0.35
        + robustness_term * 0.25
        + positive_stage_share * 0.20
        + count_term * 0.10
        + drawdown_term * 0.10
    )


def mark_recommended_exits(stage_comparison: pd.DataFrame, exit_comparison: pd.DataFrame) -> pd.DataFrame:
    if stage_comparison.empty or exit_comparison.empty:
        return stage_comparison
    recommended = exit_comparison.loc[exit_comparison["recommended_exit_flag"] == True, ["setup_name", "exit_rule_id"]].copy()
    recommended["recommended_exit_flag"] = True
    merged = stage_comparison.merge(recommended, on=["setup_name", "exit_rule_id"], how="left")
    merged["recommended_exit_flag"] = merged["recommended_exit_flag"].eq(True)
    return merged


def combine_setup_results(setup_results: list[pd.DataFrame], setup_name: str) -> pd.DataFrame:
    frames = [frame for frame in setup_results if not frame.empty and str(frame.iloc[0]["setup_name"]) == setup_name]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_practical_imitation_section(stage_comparison: pd.DataFrame, exit_comparison: pd.DataFrame) -> str:
    if stage_comparison.empty or exit_comparison.empty:
        return (
            "PRACTICAL IMITATION MODEL\n"
            "-------------------------\n"
            "Proxy imitation testing could not be completed.\n"
        )

    short_best = best_exit_row(exit_comparison, "eth_short_proxy")
    long_best = best_exit_row(exit_comparison, "eth_long_proxy")
    short_stage_a = best_stage_row(stage_comparison, "eth_short_proxy", short_best.get("exit_rule_id"))
    long_stage_a = best_stage_row(stage_comparison, "eth_long_proxy", long_best.get("exit_rule_id"))

    better_setup = choose_better_setup(short_best, long_best)
    confidence = determine_proxy_confidence(short_best, long_best)
    edge_alive = (
        pd.notna(short_best.get("positive_stage_share")) and short_best.get("positive_stage_share", 0) >= (2 / 3)
    ) or (
        pd.notna(long_best.get("positive_stage_share")) and long_best.get("positive_stage_share", 0) >= (2 / 3)
    )

    final_verdict = (
        "At this point it is more practical to test simplified imitation models than to keep reverse engineering exact execution."
        if edge_alive
        else "Exact execution reverse engineering is already near its ceiling, but the imitation models still need cautious validation before they deserve capital."
    )

    report = f"""
PRACTICAL IMITATION MODEL
-------------------------
This stage stops trying to reconstruct the trader's exact execution and instead builds two honest proxy setups inspired by the ETH archetypes that were recovered earlier.

Proxy setups:
- ETH Short proxy: near upper structure after a 60m bounce, then a simple 5m failed-continuation candle.
- ETH Long proxy: near lower structure below EMA20 5m, then a simple 5m stabilization candle.

Recommended proxy exits:
- Short proxy: {short_best.get('exit_rule_label', 'n/a')}.
- Long proxy: {long_best.get('exit_rule_label', 'n/a')}.

Stage A snapshot:
- Short proxy: {fmt_int(short_stage_a.get('executed_trade_count'))} trades, win rate {pct(short_stage_a.get('win_rate'))}, expectancy {fmt_number(short_stage_a.get('expectancy_pct'))}%, rough drawdown {fmt_number(short_stage_a.get('rough_max_drawdown_pct'))}%, archetype similarity {fmt_number(short_stage_a.get('entry_profile_similarity'))}.
- Long proxy: {fmt_int(long_stage_a.get('executed_trade_count'))} trades, win rate {pct(long_stage_a.get('win_rate'))}, expectancy {fmt_number(long_stage_a.get('expectancy_pct'))}%, rough drawdown {fmt_number(long_stage_a.get('rough_max_drawdown_pct'))}%, archetype similarity {fmt_number(long_stage_a.get('entry_profile_similarity'))}.

Interpretation:
- Better setup by current evidence: {better_setup}.
- The proxy models are style-inspired setup approximations, not a reconstruction of the trader's exact rules.
- Confidence level: {confidence}.

Final verdict:
- {final_verdict}
""".strip()
    return report + "\n"


def best_exit_row(exit_comparison: pd.DataFrame, setup_name: str) -> dict:
    subset = exit_comparison.loc[exit_comparison["setup_name"] == setup_name]
    if subset.empty:
        return {}
    row = subset.loc[subset["recommended_exit_flag"] == True]
    if row.empty:
        row = subset.head(1)
    return row.iloc[0].to_dict()


def best_stage_row(stage_comparison: pd.DataFrame, setup_name: str, exit_rule_id: str | None) -> dict:
    subset = stage_comparison.loc[
        (stage_comparison["setup_name"] == setup_name)
        & (stage_comparison["stage_id"] == "stage_a")
    ]
    if exit_rule_id:
        subset = subset.loc[subset["exit_rule_id"] == exit_rule_id]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def choose_better_setup(short_best: dict, long_best: dict) -> str:
    short_score = safe_number(short_best.get("recommended_score"))
    long_score = safe_number(long_best.get("recommended_score"))
    if short_score > long_score + 0.05:
        return "Short proxy"
    if long_score > short_score + 0.05:
        return "Long proxy"
    return "No clear winner"


def determine_proxy_confidence(short_best: dict, long_best: dict) -> str:
    stage_a_expectancies = [
        safe_number(short_best.get("stage_a_expectancy_pct")),
        safe_number(long_best.get("stage_a_expectancy_pct")),
    ]
    positive_stage_shares = [
        safe_number(short_best.get("positive_stage_share")),
        safe_number(long_best.get("positive_stage_share")),
    ]
    if max(stage_a_expectancies) >= 0.20 and max(positive_stage_shares) >= 0.67:
        return "high"
    if max(stage_a_expectancies) >= 0.05 and max(positive_stage_shares) >= 0.34:
        return "medium"
    return "low"


def create_eth_proxy_plots(
    short_results: pd.DataFrame,
    long_results: pd.DataFrame,
    exit_comparison: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if exit_comparison.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    equity_path = plots_dir / "eth_proxy_equity_curves.png"
    distribution_path = plots_dir / "eth_proxy_return_distributions.png"

    fig_equity = plot_eth_proxy_equity_curves(short_results, long_results, exit_comparison)
    fig_equity.savefig(equity_path, dpi=150, bbox_inches="tight")
    plt.close(fig_equity)
    logger.info("Saved plot: %s", equity_path.name)

    fig_distribution = plot_eth_proxy_return_distributions(short_results, long_results, exit_comparison)
    fig_distribution.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig_distribution)
    logger.info("Saved plot: %s", distribution_path.name)


def plot_eth_proxy_equity_curves(
    short_results: pd.DataFrame,
    long_results: pd.DataFrame,
    exit_comparison: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    plot_proxy_equity_axis(axes[0], short_results, exit_comparison, "eth_short_proxy", "ETH Short Proxy Equity Curves")
    plot_proxy_equity_axis(axes[1], long_results, exit_comparison, "eth_long_proxy", "ETH Long Proxy Equity Curves")
    fig.suptitle("Practical Imitation Model: Equity Curves by Stage")
    return fig


def plot_proxy_equity_axis(
    ax: plt.Axes,
    results: pd.DataFrame,
    exit_comparison: pd.DataFrame,
    setup_name: str,
    title: str,
) -> None:
    best = best_exit_row(exit_comparison, setup_name)
    if results.empty or not best:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.set_axis_off()
        return

    subset = results.loc[results["exit_rule_id"] == best["exit_rule_id"]].copy()
    colors = {"stage_a": "#1f77b4", "stage_b": "#ff7f0e", "stage_c": "#2ca02c"}
    for stage_id, stage_frame in subset.groupby("stage_id"):
        ordered = stage_frame.sort_values("entry_time").reset_index(drop=True)
        equity = ordered["return_pct"].fillna(0).cumsum()
        ax.plot(np.arange(1, len(ordered) + 1), equity, label=stage_id.upper(), color=colors.get(stage_id, "#777777"))
    ax.set_title(f"{title} ({best.get('exit_rule_label', 'n/a')})")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative return %")
    ax.grid(alpha=0.25)
    ax.legend()


def plot_eth_proxy_return_distributions(
    short_results: pd.DataFrame,
    long_results: pd.DataFrame,
    exit_comparison: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_proxy_distribution_axis(axes[0], short_results, exit_comparison, "eth_short_proxy", "Short proxy")
    plot_proxy_distribution_axis(axes[1], long_results, exit_comparison, "eth_long_proxy", "Long proxy")
    fig.suptitle("Practical Imitation Model: Stage A Return Distributions")
    return fig


def plot_proxy_distribution_axis(
    ax: plt.Axes,
    results: pd.DataFrame,
    exit_comparison: pd.DataFrame,
    setup_name: str,
    title: str,
) -> None:
    best = best_exit_row(exit_comparison, setup_name)
    if results.empty or not best:
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        ax.set_axis_off()
        return
    subset = results.loc[
        (results["exit_rule_id"] == best["exit_rule_id"])
        & (results["stage_id"] == "stage_a")
    ]
    ax.hist(subset["return_pct"].dropna(), bins=30, edgecolor="black", color="#4c78a8")
    ax.set_title(f"{title} ({best.get('exit_rule_label', 'n/a')})")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Trade count")
    ax.grid(alpha=0.25)


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0
