from __future__ import annotations

import itertools
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster_rule_extraction import mann_whitney_summary, percentile_overlap_pct
from eth_filter_tightening import TIMING_FEATURE_SPECS, attach_timing_features, prepare_timing_feature_tables
from eth_replication_hypothesis_test import compare_profiles, summarize_signal_like_profiles
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


ETH_SYMBOL = "ETHUSDT"
TARGET_CLUSTER_ID = 1
TARGET_SIDE = "Short"
TARGET_STAGE_FLAG = "in_stage_a"

MICRO_FEATURE_SPECS = [
    {"key": "micro_ret_1m_last_pct", "label": "1m return last candle %", "type": "numeric", "family": "push"},
    {"key": "micro_ret_3m_pct", "label": "1m return last 3m %", "type": "numeric", "family": "push"},
    {"key": "micro_ret_5m_pct", "label": "1m return last 5m %", "type": "numeric", "family": "push"},
    {"key": "micro_acceleration_ratio", "label": "1m acceleration ratio", "type": "numeric", "family": "push"},
    {"key": "micro_upper_wick_ratio_1m", "label": "Upper wick ratio 1m", "type": "numeric", "family": "wick"},
    {"key": "micro_upper_wick_ratio_5m", "label": "Upper wick ratio 5m", "type": "numeric", "family": "wick"},
    {"key": "micro_body_to_range_ratio_1m", "label": "Body-to-range ratio 1m", "type": "numeric", "family": "candle"},
    {"key": "micro_body_to_range_ratio_5m", "label": "Body-to-range ratio 5m", "type": "numeric", "family": "candle"},
    {"key": "micro_distance_to_last_5m_high_pct", "label": "Distance to last 5m high %", "type": "numeric", "family": "structure"},
    {"key": "micro_distance_to_last_15m_high_pct", "label": "Distance to last 15m high %", "type": "numeric", "family": "structure"},
    {"key": "micro_sweep_size_above_5m_high_pct", "label": "Sweep above 5m high %", "type": "numeric", "family": "sweep"},
    {"key": "micro_sweep_size_above_15m_high_pct", "label": "Sweep above 15m high %", "type": "numeric", "family": "sweep"},
    {"key": "micro_relative_volume_1m", "label": "Relative volume 1m", "type": "numeric", "family": "volume"},
    {"key": "micro_relative_volume_prev_1m", "label": "Previous 1m relative volume", "type": "numeric", "family": "volume"},
    {"key": "micro_relative_volume_5m", "label": "Relative volume 5m", "type": "numeric", "family": "volume"},
    {"key": "micro_final_push_up_flag", "label": "Final push up before entry", "type": "boolean", "family": "push"},
    {"key": "micro_last_1m_weaker_than_push_flag", "label": "Last 1m weaker than prior push", "type": "boolean", "family": "push"},
    {"key": "micro_rejection_candle_near_local_high_flag", "label": "Rejection candle near local high", "type": "boolean", "family": "wick"},
    {"key": "micro_close_below_midpoint_after_upper_push_flag", "label": "Close below midpoint after upper push", "type": "boolean", "family": "wick"},
    {"key": "micro_failed_breakout_wick_over_5m_high_flag", "label": "Failed breakout wick over 5m high", "type": "boolean", "family": "sweep"},
    {"key": "micro_failed_breakout_wick_over_15m_high_flag", "label": "Failed breakout wick over 15m high", "type": "boolean", "family": "sweep"},
    {"key": "micro_break_recent_5m_high_flag", "label": "Break above recent 5m high", "type": "boolean", "family": "sweep"},
    {"key": "micro_break_recent_15m_high_flag", "label": "Break above recent 15m high", "type": "boolean", "family": "sweep"},
    {"key": "micro_close_back_below_5m_high_flag", "label": "Close back below 5m high", "type": "boolean", "family": "structure"},
    {"key": "micro_close_back_below_15m_high_flag", "label": "Close back below 15m high", "type": "boolean", "family": "structure"},
    {"key": "micro_return_inside_prior_micro_range_flag", "label": "Return inside prior micro-range", "type": "boolean", "family": "structure"},
    {"key": "micro_break_above_range_then_reentry_flag", "label": "Break above range then re-entry", "type": "boolean", "family": "structure"},
    {"key": "micro_volume_spike_then_weaker_continuation_flag", "label": "Volume spike then weaker continuation", "type": "boolean", "family": "volume"},
    {"key": "micro_low_volume_last_push_flag", "label": "Low-volume last push", "type": "boolean", "family": "volume"},
    {"key": "micro_price_volume_divergence_flag", "label": "Price-volume divergence on push", "type": "boolean", "family": "volume"},
    {"key": "micro_upper_wick_gt_threshold_flag", "label": "Upper wick above threshold", "type": "boolean", "family": "wick"},
    {"key": "micro_body_small_flag", "label": "Small candle body", "type": "boolean", "family": "candle"},
    {"key": "micro_bearish_close_after_push_flag", "label": "Bearish close after push", "type": "boolean", "family": "candle"},
    {"key": "micro_inside_bar_after_push_flag", "label": "Inside bar after push", "type": "boolean", "family": "candle"},
    {"key": "micro_outside_reversal_bar_flag", "label": "Outside reversal bar", "type": "boolean", "family": "candle"},
]


def run_eth_cluster1_micro_trigger_analysis(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    broad_signals_cluster1: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH CLUSTER 1 MICRO-STRUCTURE TRIGGER ANALYSIS\n"
        "----------------------------------------------\n"
        "No sufficient Cluster 1 short data was available for micro-structure trigger analysis.\n"
    )

    real_cluster1 = cluster_frame.loc[
        (cluster_frame["symbol"] == ETH_SYMBOL)
        & (cluster_frame["cluster_id"] == TARGET_CLUSTER_ID)
        & (cluster_frame["side"] == TARGET_SIDE)
    ].copy()
    broad_stage_a = broad_signals_cluster1.loc[broad_signals_cluster1[TARGET_STAGE_FLAG] == True].copy() if not broad_signals_cluster1.empty else pd.DataFrame()

    if real_cluster1.empty or broad_stage_a.empty:
        return {
            "ranking": empty,
            "candidates": empty,
            "quality": empty,
            "section_text": empty_section,
        }

    min_time, max_time = infer_micro_trigger_window(real_cluster1, broad_stage_a)
    minute_frame, frame_5m, frame_15m = load_micro_trigger_market_data(min_time, max_time, logger)
    minute_features, frame_5m_features = prepare_micro_feature_tables(minute_frame, frame_5m, frame_15m)

    reference_source = real_cluster1[
        ["trade_number", "cluster_id", "cluster_description", "side", "opened_dt_utc"]
    ].merge(
        entry_quality_df,
        on="trade_number",
        how="left",
        suffixes=("", "_entry"),
    )
    reference = attach_timing_features(
        reference_source.rename(columns={"opened_dt_utc": "entry_time"}),
        minute_features,
        frame_5m_features,
        time_column="entry_time",
    )
    reference["fast_positive_reaction_flag"] = reference["group_fast_positive_reaction"].fillna(False).astype(bool)

    broad_pool = attach_timing_features(
        broad_stage_a.copy(),
        minute_features,
        frame_5m_features,
        time_column="entry_time",
    )
    tightened_pool = build_tightened_stage_a_pool(broad_pool, tightened_summary_df)

    real_summary = summarize_signal_like_profiles(reference)
    broad_summary = summarize_signal_like_profiles(broad_pool)
    tightened_summary = summarize_signal_like_profiles(tightened_pool) if not tightened_pool.empty else {}

    ranking_df, single_candidates = rank_micro_features(reference, broad_pool, tightened_pool, real_summary, broad_summary)
    candidate_df, candidate_objects = build_candidate_micro_triggers(
        ranking_df,
        single_candidates,
        broad_pool,
        tightened_pool,
        real_summary,
        broad_summary,
    )
    quality_df = evaluate_trigger_quality(
        candidate_df,
        candidate_objects,
        broad_pool,
        tightened_pool,
        real_summary,
        broad_summary,
    )
    section_text = build_micro_trigger_section(quality_df, ranking_df, real_summary, broad_summary, tightened_summary)

    return {
        "ranking": ranking_df,
        "candidates": candidate_df,
        "quality": quality_df,
        "section_text": section_text,
    }


def infer_micro_trigger_window(real_cluster1: pd.DataFrame, broad_stage_a: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    combined = pd.concat(
        [
            pd.to_datetime(real_cluster1["opened_dt_utc"], utc=True),
            pd.to_datetime(broad_stage_a["entry_time"], utc=True),
        ],
        ignore_index=True,
    ).dropna()
    return combined.min() - pd.Timedelta(hours=6), combined.max() + pd.Timedelta(hours=1)


def load_micro_trigger_market_data(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minute_frame = fetch_futures_klines(
        ETH_SYMBOL,
        int(start_time.timestamp() * 1000),
        int(end_time.timestamp() * 1000),
        interval="1m",
        logger=logger,
    )
    frame_5m = fetch_futures_klines(
        ETH_SYMBOL,
        int(start_time.timestamp() * 1000),
        int(end_time.timestamp() * 1000),
        interval="5m",
        logger=logger,
    )
    frame_15m = fetch_futures_klines(
        ETH_SYMBOL,
        int(start_time.timestamp() * 1000),
        int(end_time.timestamp() * 1000),
        interval="15m",
        logger=logger,
    )
    return minute_frame, frame_5m, frame_15m


def prepare_micro_feature_tables(
    minute_frame: pd.DataFrame,
    frame_5m: pd.DataFrame,
    frame_15m: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_minute_features, timing_five_features = prepare_timing_feature_tables(minute_frame, frame_5m)
    minute = minute_frame[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    five = frame_5m[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    fifteen = frame_15m[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")

    close_1m = pd.to_numeric(minute["close"], errors="coerce")
    open_1m = pd.to_numeric(minute["open"], errors="coerce")
    high_1m = pd.to_numeric(minute["high"], errors="coerce")
    low_1m = pd.to_numeric(minute["low"], errors="coerce")
    volume_1m = pd.to_numeric(minute["volume"], errors="coerce")
    range_1m = (high_1m - low_1m).replace(0, np.nan)
    body_1m = (close_1m - open_1m).abs()

    minute["micro_ret_1m_last_pct"] = close_1m.pct_change(1) * 100.0
    minute["micro_ret_3m_pct"] = close_1m.pct_change(3) * 100.0
    minute["micro_ret_5m_pct"] = close_1m.pct_change(5) * 100.0
    minute["micro_acceleration_ratio"] = minute["micro_ret_1m_last_pct"].abs() / ((minute["micro_ret_3m_pct"].abs() / 3.0).replace(0, np.nan))
    minute["micro_upper_wick_ratio_1m"] = (high_1m - np.maximum(open_1m, close_1m)) / range_1m
    minute["micro_body_to_range_ratio_1m"] = body_1m / range_1m
    minute["micro_relative_volume_1m"] = volume_1m / volume_1m.rolling(20, min_periods=5).mean()
    minute["micro_relative_volume_prev_1m"] = minute["micro_relative_volume_1m"].shift(1)

    recent_5m_high = high_1m.shift(1).rolling(5, min_periods=3).max()
    recent_15m_high = high_1m.shift(1).rolling(15, min_periods=5).max()
    recent_15m_low = low_1m.shift(1).rolling(15, min_periods=5).min()
    recent_15m_range_high = high_1m.shift(1).rolling(15, min_periods=5).max()
    recent_15m_range_low = low_1m.shift(1).rolling(15, min_periods=5).min()

    minute["micro_distance_to_last_5m_high_pct"] = (recent_5m_high - close_1m) / close_1m * 100.0
    minute["micro_distance_to_last_15m_high_pct"] = (recent_15m_high - close_1m) / close_1m * 100.0
    minute["micro_sweep_size_above_5m_high_pct"] = np.maximum(high_1m - recent_5m_high, 0) / close_1m * 100.0
    minute["micro_sweep_size_above_15m_high_pct"] = np.maximum(high_1m - recent_15m_high, 0) / close_1m * 100.0

    minute["micro_final_push_up_flag"] = (minute["micro_ret_5m_pct"] > 0.12) & (minute["micro_ret_1m_last_pct"] > 0)
    minute["micro_last_1m_weaker_than_push_flag"] = minute["micro_ret_1m_last_pct"].abs() < (minute["micro_ret_3m_pct"].abs() / 3.0)
    minute["micro_rejection_candle_near_local_high_flag"] = (
        (minute["micro_upper_wick_ratio_1m"] >= 0.45)
        & (minute["micro_distance_to_last_5m_high_pct"] <= 0.12)
        & (close_1m <= open_1m)
    )
    minute["micro_close_below_midpoint_after_upper_push_flag"] = (
        (high_1m > recent_5m_high)
        & (close_1m < ((high_1m + low_1m) / 2.0))
    )
    minute["micro_failed_breakout_wick_over_5m_high_flag"] = (high_1m > recent_5m_high) & (close_1m < recent_5m_high)
    minute["micro_failed_breakout_wick_over_15m_high_flag"] = (high_1m > recent_15m_high) & (close_1m < recent_15m_high)
    minute["micro_break_recent_5m_high_flag"] = high_1m > recent_5m_high
    minute["micro_break_recent_15m_high_flag"] = high_1m > recent_15m_high
    minute["micro_close_back_below_5m_high_flag"] = close_1m < recent_5m_high
    minute["micro_close_back_below_15m_high_flag"] = close_1m < recent_15m_high
    minute["micro_return_inside_prior_micro_range_flag"] = close_1m.between(recent_15m_range_low, recent_15m_range_high, inclusive="both")
    minute["micro_break_above_range_then_reentry_flag"] = (high_1m > recent_15m_range_high) & minute["micro_return_inside_prior_micro_range_flag"]
    minute["micro_volume_spike_then_weaker_continuation_flag"] = (
        (minute["micro_relative_volume_prev_1m"] >= 1.35)
        & (minute["micro_relative_volume_1m"] < minute["micro_relative_volume_prev_1m"])
        & (minute["micro_ret_1m_last_pct"] > 0)
    )
    minute["micro_low_volume_last_push_flag"] = (minute["micro_ret_1m_last_pct"] > 0) & (minute["micro_relative_volume_1m"] <= 0.85)
    minute["micro_price_volume_divergence_flag"] = (
        (minute["micro_ret_3m_pct"] > 0.10)
        & (minute["micro_relative_volume_1m"] < minute["micro_relative_volume_prev_1m"])
    )
    minute["micro_upper_wick_gt_threshold_flag"] = minute["micro_upper_wick_ratio_1m"] >= 0.45
    minute["micro_body_small_flag"] = minute["micro_body_to_range_ratio_1m"] <= 0.25
    minute["micro_bearish_close_after_push_flag"] = (minute["micro_ret_3m_pct"] > 0.08) & (close_1m < open_1m)
    minute["micro_inside_bar_after_push_flag"] = (
        (high_1m < high_1m.shift(1))
        & (low_1m > low_1m.shift(1))
        & (minute["micro_ret_3m_pct"] > 0.08)
    )
    minute["micro_outside_reversal_bar_flag"] = (
        (high_1m > high_1m.shift(1))
        & (low_1m < low_1m.shift(1))
        & (close_1m < open_1m)
    )

    close_5m = pd.to_numeric(five["close"], errors="coerce")
    open_5m = pd.to_numeric(five["open"], errors="coerce")
    high_5m = pd.to_numeric(five["high"], errors="coerce")
    low_5m = pd.to_numeric(five["low"], errors="coerce")
    volume_5m = pd.to_numeric(five["volume"], errors="coerce")
    range_5m = (high_5m - low_5m).replace(0, np.nan)
    five["micro_upper_wick_ratio_5m"] = (high_5m - np.maximum(open_5m, close_5m)) / range_5m
    five["micro_body_to_range_ratio_5m"] = (close_5m - open_5m).abs() / range_5m
    five["micro_relative_volume_5m"] = volume_5m / volume_5m.rolling(20, min_periods=5).mean()

    minute_features = minute[
        ["close_time", *[spec["key"] for spec in MICRO_FEATURE_SPECS if spec["key"] not in {"micro_upper_wick_ratio_5m", "micro_body_to_range_ratio_5m", "micro_relative_volume_5m"}]]
    ].copy()
    minute_features = minute_features.merge(timing_minute_features, on="close_time", how="left")
    five_features = five[
        ["close_time", "micro_upper_wick_ratio_5m", "micro_body_to_range_ratio_5m", "micro_relative_volume_5m"]
    ].copy()
    five_features = five_features.merge(timing_five_features, on="close_time", how="left")

    if not fifteen.empty:
        fifteen = fifteen.copy()
        fifteen["micro_last_15m_high"] = pd.to_numeric(fifteen["high"], errors="coerce").shift(1)
        fifteen["micro_last_15m_low"] = pd.to_numeric(fifteen["low"], errors="coerce").shift(1)
        _ = fifteen

    return minute_features, five_features


def build_tightened_stage_a_pool(
    broad_pool: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if broad_pool.empty or tightened_summary_df.empty:
        return pd.DataFrame()
    row = tightened_summary_df.loc[tightened_summary_df["cluster_id"] == TARGET_CLUSTER_ID]
    if row.empty:
        return pd.DataFrame()
    timing_rule_text = str(row.iloc[0]["timing_rule_text"])
    conditions = parse_timing_rule_text(timing_rule_text)
    if not conditions:
        return pd.DataFrame()
    return apply_conditions(broad_pool, conditions)


def parse_timing_rule_text(rule_text: str) -> list[dict]:
    label_to_key = {spec["label"]: spec["key"] for spec in TIMING_FEATURE_SPECS}
    conditions = []
    for chunk in [piece.strip() for piece in rule_text.split(" AND ")]:
        if " = " in chunk:
            label, raw_value = chunk.split(" = ", 1)
            key = label_to_key.get(label.strip())
            if key is None:
                continue
            conditions.append({"feature_key": key, "rule_type": "equals", "value": raw_value.strip().lower() == "true", "text": chunk})
            continue
        match = re.match(r"(.+?)\s*(>=|<=)\s*(-?\d+(?:\.\d+)?)$", chunk)
        if not match:
            continue
        label = match.group(1).strip()
        operator = match.group(2)
        value = float(match.group(3))
        key = label_to_key.get(label)
        if key is None:
            continue
        conditions.append({"feature_key": key, "rule_type": "ge" if operator == ">=" else "le", "threshold": value, "text": chunk})
    return conditions


def rank_micro_features(
    reference: pd.DataFrame,
    broad_pool: pd.DataFrame,
    tightened_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    candidates = []
    for spec in MICRO_FEATURE_SPECS:
        candidate = derive_micro_candidate(spec, reference, broad_pool)
        if candidate is None:
            continue
        ref_hits = apply_conditions(reference, candidate["conditions"])
        broad_hits = apply_conditions(broad_pool, candidate["conditions"])
        tightened_hits = apply_conditions(tightened_pool, candidate["conditions"]) if not tightened_pool.empty else pd.DataFrame()

        filtered_summary = summarize_signal_like_profiles(broad_hits)
        filtered_similarity = compare_profiles(real_summary, filtered_summary)
        broad_similarity = compare_profiles(real_summary, broad_summary)

        coverage_real = safe_divide(len(ref_hits), len(reference))
        frequency_broad = safe_divide(len(broad_hits), len(broad_pool))
        frequency_tightened = safe_divide(len(tightened_hits), len(tightened_pool)) if not tightened_pool.empty else np.nan
        lift = safe_divide(coverage_real, frequency_broad)
        precision_proxy = safe_divide(coverage_real, coverage_real + frequency_broad)
        ranking_score = (
            safe_number(lift) * 0.35
            + max(safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"]), 0.0) * 0.35
            + max(coverage_real - frequency_broad, 0.0) * 0.20
            + max(safe_number(broad_summary.get("median_MAE_pct")) - safe_number(filtered_summary.get("median_MAE_pct")), 0.0) * 0.10
        )

        rows.append(
            {
                "feature_key": spec["key"],
                "feature_label": spec["label"],
                "feature_family": spec["family"],
                "feature_type": spec["type"],
                "condition_text": candidate["trigger_text"],
                "coverage_on_real_cluster1_trades": coverage_real,
                "frequency_on_broad_candidate_signals": frequency_broad,
                "frequency_on_tightened_candidate_signals": frequency_tightened,
                "lift": lift,
                "precision_proxy_balanced": precision_proxy,
                "recall_proxy": coverage_real,
                "effect_direction": "more_common_in_real" if coverage_real > frequency_broad else "more_common_in_broad",
                "effect_size": candidate.get("effect_size"),
                "mann_whitney_p_value": candidate.get("mann_whitney_p_value"),
                "percentile_overlap_pct": candidate.get("percentile_overlap_pct"),
                "signal_count_after_filter": int(len(broad_hits)),
                "pct_positive_1m": filtered_summary.get("pct_positive_after_1m"),
                "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
                "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
                "median_MAE": filtered_summary.get("median_MAE_pct"),
                "median_MFE": filtered_summary.get("median_MFE_pct"),
                "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
                "median_return_4h": filtered_summary.get("median_return_after_240m_pct"),
                "similarity_to_real_cluster1_profile": filtered_similarity["score"],
                "similarity_improvement_vs_broad": safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"]),
                "ranking_score": ranking_score,
            }
        )

        candidate["ranking_score"] = ranking_score
        candidates.append(candidate)

    ranking_df = pd.DataFrame(rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            ["ranking_score", "similarity_improvement_vs_broad", "lift"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return ranking_df, candidates


def derive_micro_candidate(spec: dict, reference: pd.DataFrame, broad_pool: pd.DataFrame) -> dict | None:
    key = spec["key"]
    label = spec["label"]
    if spec["type"] == "boolean":
        ref_rate = reference[key].fillna(False).astype(bool).mean()
        broad_rate = broad_pool[key].fillna(False).astype(bool).mean()
        if ref_rate <= broad_rate + 0.05:
            return None
        return {
            "candidate_id": key,
            "feature_key": key,
            "trigger_text": f"{label} = True",
            "conditions": [{"feature_key": key, "rule_type": "equals", "value": True, "text": f"{label} = True"}],
            "effect_size": ref_rate - broad_rate,
            "mann_whitney_p_value": np.nan,
            "percentile_overlap_pct": np.nan,
            "family": spec["family"],
            "type": spec["type"],
        }

    ref_values = pd.to_numeric(reference[key], errors="coerce").dropna().to_numpy(dtype=float)
    broad_values = pd.to_numeric(broad_pool[key], errors="coerce").dropna().to_numpy(dtype=float)
    if ref_values.size < 5 or broad_values.size < 25:
        return None
    ref_median = float(np.nanmedian(ref_values))
    broad_median = float(np.nanmedian(broad_values))
    if math.isclose(ref_median, broad_median, abs_tol=1e-9):
        return None
    direction = "ge" if ref_median > broad_median else "le"
    threshold = float(np.nanpercentile(ref_values, 25 if direction == "ge" else 75))
    hits = apply_numeric_condition(pd.to_numeric(broad_pool[key], errors="coerce"), direction, threshold).fillna(False)
    retained_share = hits.mean()
    if retained_share <= 0.01 or retained_share >= 0.90:
        return None
    mann_whitney = mann_whitney_summary(ref_values, broad_values)
    return {
        "candidate_id": key,
        "feature_key": key,
        "trigger_text": f"{label} {'>=' if direction == 'ge' else '<='} {format_threshold(threshold)}",
        "conditions": [{"feature_key": key, "rule_type": direction, "threshold": threshold, "text": f"{label} {'>=' if direction == 'ge' else '<='} {format_threshold(threshold)}"}],
        "effect_size": mann_whitney["rank_biserial"],
        "mann_whitney_p_value": mann_whitney["p_value"],
        "percentile_overlap_pct": percentile_overlap_pct(ref_values, broad_values),
        "family": spec["family"],
        "type": spec["type"],
    }


def apply_numeric_condition(series: pd.Series, direction: str, threshold: float) -> pd.Series:
    if direction == "ge":
        return series >= threshold
    return series <= threshold


def apply_conditions(frame: pd.DataFrame, conditions: list[dict]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        key = condition["feature_key"]
        if condition["rule_type"] == "equals":
            condition_mask = frame[key].fillna(False).astype(bool) == bool(condition["value"])
        else:
            series = pd.to_numeric(frame[key], errors="coerce")
            condition_mask = apply_numeric_condition(series, condition["rule_type"], condition["threshold"])
        mask = mask & condition_mask.fillna(False)
    return frame.loc[mask].copy()


def build_candidate_micro_triggers(
    ranking_df: pd.DataFrame,
    single_candidates: list[dict],
    broad_pool: pd.DataFrame,
    tightened_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    if ranking_df.empty:
        return pd.DataFrame(), {}

    candidate_map = {candidate["candidate_id"]: candidate for candidate in single_candidates}
    top_rows = ranking_df.head(8)
    top_candidates = [candidate_map[row["feature_key"]] for _, row in top_rows.iterrows() if row["feature_key"] in candidate_map]

    candidate_rows = []
    combined_candidates: dict[str, dict] = {}

    for candidate in top_candidates:
        row = evaluate_candidate_row(candidate, broad_pool, tightened_pool, real_summary, broad_summary)
        candidate_rows.append(row)
        combined_candidates[candidate["candidate_id"]] = candidate

    for left, right in itertools.combinations(top_candidates, 2):
        if left["family"] == right["family"]:
            continue
        combo = {
            "candidate_id": f"{left['candidate_id']}__{right['candidate_id']}",
            "trigger_text": f"{left['trigger_text']} AND {right['trigger_text']}",
            "conditions": [*left["conditions"], *right["conditions"]],
            "family": f"{left['family']} + {right['family']}",
            "type": "pair",
        }
        row = evaluate_candidate_row(combo, broad_pool, tightened_pool, real_summary, broad_summary)
        if row["signal_count"] < 8:
            continue
        candidate_rows.append(row)
        combined_candidates[combo["candidate_id"]] = combo

    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        candidates_df = candidates_df.sort_values(
            ["ranking_score", "similarity_to_real_cluster1_profile", "signal_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return candidates_df, combined_candidates


def evaluate_candidate_row(
    candidate: dict,
    broad_pool: pd.DataFrame,
    tightened_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
) -> dict:
    filtered = apply_conditions(broad_pool, candidate["conditions"])
    tightened_filtered = apply_conditions(tightened_pool, candidate["conditions"]) if not tightened_pool.empty else pd.DataFrame()
    filtered_summary = summarize_signal_like_profiles(filtered)
    filtered_similarity = compare_profiles(real_summary, filtered_summary)
    broad_similarity = compare_profiles(real_summary, broad_summary)

    reduction_vs_base = 1 - safe_divide(len(filtered), len(broad_pool))
    reduction_vs_tightened = 1 - safe_divide(len(filtered), len(tightened_pool)) if not tightened_pool.empty else np.nan
    ranking_score = (
        max(safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"]), 0.0) * 0.50
        + max(safe_number(filtered_summary.get("pct_positive_after_5m")) - safe_number(broad_summary.get("pct_positive_after_5m")), 0.0) * 0.15
        + max(safe_number(filtered_summary.get("pct_positive_after_15m")) - safe_number(broad_summary.get("pct_positive_after_15m")), 0.0) * 0.10
        + max(safe_number(broad_summary.get("median_MAE_pct")) - safe_number(filtered_summary.get("median_MAE_pct")), 0.0) * 0.15
        + max(safe_number(reduction_vs_base), 0.0) * 0.10
    )
    if len(filtered) < 12:
        ranking_score -= 0.20

    return {
        "candidate_id": candidate["candidate_id"],
        "trigger_text": candidate["trigger_text"],
        "condition_count": len(candidate["conditions"]),
        "signal_count": int(len(filtered)),
        "reduction_vs_base": reduction_vs_base,
        "reduction_vs_tightened": reduction_vs_tightened,
        "pct_positive_1m": filtered_summary.get("pct_positive_after_1m"),
        "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
        "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
        "median_MAE": filtered_summary.get("median_MAE_pct"),
        "median_MFE": filtered_summary.get("median_MFE_pct"),
        "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
        "median_return_4h": filtered_summary.get("median_return_after_240m_pct"),
        "similarity_to_real_cluster1_profile": filtered_similarity["score"],
        "similarity_label": filtered_similarity["label"],
        "similarity_improvement_vs_broad": safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"]),
        "ranking_score": ranking_score,
    }


def evaluate_trigger_quality(
    candidate_df: pd.DataFrame,
    candidate_objects: dict[str, dict],
    broad_pool: pd.DataFrame,
    tightened_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()

    quality_rows = []
    for _, row in candidate_df.head(5).iterrows():
        candidate = candidate_objects[row["candidate_id"]]
        quality_rows.append(evaluate_candidate_row(candidate, broad_pool, tightened_pool, real_summary, broad_summary))
    quality_df = pd.DataFrame(quality_rows)
    if not quality_df.empty:
        quality_df = quality_df.sort_values(
            ["ranking_score", "similarity_to_real_cluster1_profile", "signal_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return quality_df


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0


def format_threshold(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def build_micro_trigger_section(
    quality_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
    tightened_summary: dict,
) -> str:
    if quality_df.empty:
        return (
            "ETH CLUSTER 1 MICRO-STRUCTURE TRIGGER ANALYSIS\n"
            "----------------------------------------------\n"
            "Micro-structure trigger analysis could not identify a stable Cluster 1 short trigger.\n"
        )

    best = quality_df.iloc[0]
    top_features = ", ".join(ranking_df.head(5)["feature_label"].astype(str).tolist())
    confidence = determine_micro_trigger_confidence(best, quality_df)
    verdict = (
        "The best candidate micro-trigger is sharp enough to justify a small exploratory TP/SL test."
        if pd.notna(best["similarity_to_real_cluster1_profile"]) and best["similarity_to_real_cluster1_profile"] >= 0.72 and best["signal_count"] >= 24
        else "The trigger evidence is still too weak or too discretionary for a confident backtest."
    )

    report = f"""
ETH CLUSTER 1 MICRO-STRUCTURE TRIGGER ANALYSIS
----------------------------------------------
This section tests whether a repeatable micro-trigger exists inside the Cluster 1 short context family. It compares real Cluster 1 short trades against broad Stage A replication signals and against the current tightened Stage A pool.

Key findings:
- Most useful micro-features: {top_features}.
- Best candidate micro-trigger: {best['trigger_text']}.
- Broad Stage A signals: {fmt_int(broad_summary.get('signal_count'))}. Current tightened Stage A signals: {fmt_int(tightened_summary.get('signal_count')) if tightened_summary else 'n/a'}. Best micro-trigger signals: {fmt_int(best['signal_count'])}.
- Positive after 5m: broad {pct(broad_summary.get('pct_positive_after_5m'))}, tightened {pct(tightened_summary.get('pct_positive_after_5m')) if tightened_summary else 'n/a'}, micro-trigger {pct(best['pct_positive_5m'])}.
- Median MAE: broad {fmt_number(broad_summary.get('median_MAE_pct'))}%, tightened {fmt_number(tightened_summary.get('median_MAE_pct')) if tightened_summary else np.nan}%, micro-trigger {fmt_number(best['median_MAE'])}%.
- Similarity to real Cluster 1 profile: micro-trigger {fmt_number(best['similarity_to_real_cluster1_profile'])} ({best['similarity_label']}).
- Confidence level: {confidence}.

Final verdict:
- {verdict}
""".strip()
    return report + "\n"


def determine_micro_trigger_confidence(best_row: pd.Series, quality_df: pd.DataFrame) -> str:
    if best_row.empty:
        return "low"
    if (
        pd.notna(best_row["similarity_improvement_vs_broad"])
        and best_row["similarity_improvement_vs_broad"] >= 0.08
        and best_row["signal_count"] >= 20
    ):
        return "high"
    if (
        pd.notna(best_row["similarity_improvement_vs_broad"])
        and best_row["similarity_improvement_vs_broad"] >= 0.03
        and best_row["signal_count"] >= 12
    ):
        return "medium"
    return "low"


def create_eth_cluster1_micro_trigger_plots(
    ranking_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if ranking_df.empty or quality_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    importance_path = plots_dir / "eth_cluster1_micro_trigger_importance.png"
    improvement_path = plots_dir / "eth_cluster1_micro_trigger_improvement.png"

    fig_importance = plot_micro_trigger_importance(ranking_df)
    fig_importance.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig_importance)
    logger.info("Saved plot: %s", importance_path.name)

    fig_improvement = plot_micro_trigger_improvement(quality_df)
    fig_improvement.savefig(improvement_path, dpi=150, bbox_inches="tight")
    plt.close(fig_improvement)
    logger.info("Saved plot: %s", improvement_path.name)


def plot_micro_trigger_importance(ranking_df: pd.DataFrame) -> plt.Figure:
    plot_df = ranking_df.head(10).sort_values("ranking_score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["feature_label"], plot_df["ranking_score"], color="#d65f5f")
    ax.set_title("Cluster 1 Micro-trigger Feature Ranking")
    ax.set_xlabel("Ranking score")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_micro_trigger_improvement(quality_df: pd.DataFrame) -> plt.Figure:
    plot_df = quality_df.head(5).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(plot_df["trigger_text"], plot_df["signal_count"], color="#4c78a8")
    axes[0].set_title("Candidate Trigger Signal Count")
    axes[0].set_xlabel("Signals")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(plot_df["signal_count"], plot_df["similarity_to_real_cluster1_profile"], s=90, color="#f58518")
    for _, row in plot_df.iterrows():
        axes[1].annotate(row["trigger_text"], (row["signal_count"], row["similarity_to_real_cluster1_profile"]), fontsize=8)
    axes[1].set_title("Signal Count vs Similarity")
    axes[1].set_xlabel("Signals")
    axes[1].set_ylabel("Similarity to real Cluster 1 profile")
    axes[1].grid(alpha=0.25)
    return fig
