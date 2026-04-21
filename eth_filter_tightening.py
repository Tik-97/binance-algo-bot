from __future__ import annotations

import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster_rule_extraction import mann_whitney_summary, percentile_overlap_pct
from eth_replication_hypothesis_test import compare_profiles, summarize_signal_like_profiles
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


ETH_SYMBOL = "ETHUSDT"
TIMING_STAGE_ID = "stage_a"
TIMING_STAGE_LABEL = "Stage A"
FINAL_STAGE_IDS = ["stage_a", "stage_b", "stage_c"]

TIMING_FEATURE_SPECS = [
    {"key": "timing_ret_1m_last_pct", "label": "1m return last candle %", "type": "numeric", "family": "momentum"},
    {"key": "timing_ret_1m_3c_pct", "label": "1m return last 3 candles %", "type": "numeric", "family": "momentum"},
    {"key": "timing_ret_5m_last_pct", "label": "5m return last candle %", "type": "numeric", "family": "momentum"},
    {"key": "timing_ret_5m_3c_pct", "label": "5m return last 3 candles %", "type": "numeric", "family": "momentum"},
    {"key": "timing_momentum_deceleration_1m", "label": "1m momentum deceleration", "type": "numeric", "family": "momentum"},
    {"key": "timing_momentum_deceleration_5m", "label": "5m momentum deceleration", "type": "numeric", "family": "momentum"},
    {"key": "timing_upper_wick_ratio_1m", "label": "Upper wick ratio 1m", "type": "numeric", "family": "wick"},
    {"key": "timing_lower_wick_ratio_1m", "label": "Lower wick ratio 1m", "type": "numeric", "family": "wick"},
    {"key": "timing_upper_wick_ratio_5m", "label": "Upper wick ratio 5m", "type": "numeric", "family": "wick"},
    {"key": "timing_lower_wick_ratio_5m", "label": "Lower wick ratio 5m", "type": "numeric", "family": "wick"},
    {"key": "timing_distance_to_last_15m_high_pct", "label": "Distance to last 15m high %", "type": "numeric", "family": "structure"},
    {"key": "timing_distance_to_last_15m_low_pct", "label": "Distance to last 15m low %", "type": "numeric", "family": "structure"},
    {"key": "timing_relative_volume_1m", "label": "Relative volume 1m", "type": "numeric", "family": "volume"},
    {"key": "timing_relative_volume_5m", "label": "Relative volume 5m", "type": "numeric", "family": "volume"},
    {"key": "timing_sweep_high_reject_15m_flag", "label": "15m high sweep rejection", "type": "boolean", "family": "structure"},
    {"key": "timing_sweep_low_reject_15m_flag", "label": "15m low sweep rejection", "type": "boolean", "family": "structure"},
    {"key": "timing_close_inside_15m_range_flag", "label": "Close back inside 15m range", "type": "boolean", "family": "structure"},
    {"key": "timing_short_exhaustion_1m_flag", "label": "1m short exhaustion flag", "type": "boolean", "family": "momentum"},
    {"key": "timing_long_exhaustion_1m_flag", "label": "1m long exhaustion flag", "type": "boolean", "family": "momentum"},
    {"key": "timing_short_rejection_1m_flag", "label": "1m short rejection candle", "type": "boolean", "family": "wick"},
    {"key": "timing_long_rejection_1m_flag", "label": "1m long rejection candle", "type": "boolean", "family": "wick"},
]


def run_eth_filter_tightening(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    signals_cluster1: pd.DataFrame,
    signals_cluster2: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH FILTER TIGHTENING / TIMING LAYER\n"
        "------------------------------------\n"
        "No sufficient ETH replication data was available for timing-layer extraction.\n"
    )

    eth_clusters = cluster_frame.loc[cluster_frame["symbol"] == ETH_SYMBOL].copy()
    if eth_clusters.empty or (signals_cluster1.empty and signals_cluster2.empty):
        return {
            "feature_ranking": empty,
            "cluster1_candidates": empty,
            "cluster2_candidates": empty,
            "summary": empty,
            "stage_comparison": empty,
            "section_text": empty_section,
        }

    min_time, max_time = infer_eth_timing_window(eth_clusters, signals_cluster1, signals_cluster2)
    minute_frame, frame_5m = load_eth_timing_market_data(min_time, max_time, logger)
    minute_features, frame_5m_features = prepare_timing_feature_tables(minute_frame, frame_5m)

    reference_source = eth_clusters[
        ["trade_number", "cluster_id", "cluster_description", "side", "opened_dt_utc"]
    ].merge(
        entry_quality_df,
        on="trade_number",
        how="left",
        suffixes=("", "_entry"),
    )

    real_features = attach_timing_features(
        reference_source.rename(columns={"opened_dt_utc": "entry_time"}),
        minute_features,
        frame_5m_features,
        time_column="entry_time",
    )
    signal1_features = attach_timing_features(signals_cluster1.copy(), minute_features, frame_5m_features, time_column="entry_time")
    signal2_features = attach_timing_features(signals_cluster2.copy(), minute_features, frame_5m_features, time_column="entry_time")

    ranking_frames = []
    candidate_frames = []
    summary_rows = []
    stage_rows = []

    for cluster_id, signal_frame in [(1, signal1_features), (2, signal2_features)]:
        if signal_frame.empty:
            continue
        result = run_cluster_tightening(cluster_id, real_features, signal_frame, logger)
        ranking_frames.append(result["ranking"])
        candidate_frames.append(result["candidates"])
        summary_rows.append(result["summary_row"])
        stage_rows.append(result["stage_rows"])

    feature_ranking = pd.concat(ranking_frames, ignore_index=True, sort=False) if ranking_frames else pd.DataFrame()
    candidates = pd.concat(candidate_frames, ignore_index=True, sort=False) if candidate_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    stage_comparison = pd.concat(stage_rows, ignore_index=True, sort=False) if stage_rows else pd.DataFrame()
    section_text = build_tightening_section(summary_df, stage_comparison, feature_ranking)

    return {
        "feature_ranking": feature_ranking,
        "cluster1_candidates": candidates.loc[candidates["cluster_id"] == 1].copy() if not candidates.empty else empty,
        "cluster2_candidates": candidates.loc[candidates["cluster_id"] == 2].copy() if not candidates.empty else empty,
        "summary": summary_df,
        "stage_comparison": stage_comparison,
        "section_text": section_text,
    }


def infer_eth_timing_window(
    eth_clusters: pd.DataFrame,
    signals_cluster1: pd.DataFrame,
    signals_cluster2: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = [
        pd.to_datetime(eth_clusters["opened_dt_utc"], utc=True),
        pd.to_datetime(signals_cluster1.get("entry_time"), utc=True) if not signals_cluster1.empty else pd.Series(dtype="datetime64[ns, UTC]"),
        pd.to_datetime(signals_cluster2.get("entry_time"), utc=True) if not signals_cluster2.empty else pd.Series(dtype="datetime64[ns, UTC]"),
    ]
    combined = pd.concat([series.dropna() for series in timestamps if series is not None], ignore_index=True)
    return combined.min() - pd.Timedelta(hours=2), combined.max() + pd.Timedelta(hours=2)


def load_eth_timing_market_data(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return minute_frame, frame_5m


def prepare_timing_feature_tables(
    minute_frame: pd.DataFrame,
    frame_5m: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minute = minute_frame[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    frame_5m_local = frame_5m[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")

    minute = compute_candle_features(minute, prefix="timing", recent_window=15)
    frame_5m_local = compute_candle_features(frame_5m_local, prefix="timing_5m", recent_window=3)

    minute_features = minute[
        [
            "close_time",
            "timing_ret_last_pct",
            "timing_ret_3c_pct",
            "timing_upper_wick_ratio",
            "timing_lower_wick_ratio",
            "timing_relative_volume",
            "timing_distance_to_recent_high_pct",
            "timing_distance_to_recent_low_pct",
            "timing_close_inside_recent_range_flag",
            "timing_sweep_high_reject_flag",
            "timing_sweep_low_reject_flag",
            "timing_momentum_deceleration",
            "timing_short_exhaustion_flag",
            "timing_long_exhaustion_flag",
            "timing_short_rejection_flag",
            "timing_long_rejection_flag",
        ]
    ].rename(
        columns={
            "timing_ret_last_pct": "timing_ret_1m_last_pct",
            "timing_ret_3c_pct": "timing_ret_1m_3c_pct",
            "timing_upper_wick_ratio": "timing_upper_wick_ratio_1m",
            "timing_lower_wick_ratio": "timing_lower_wick_ratio_1m",
            "timing_relative_volume": "timing_relative_volume_1m",
            "timing_distance_to_recent_high_pct": "timing_distance_to_last_15m_high_pct",
            "timing_distance_to_recent_low_pct": "timing_distance_to_last_15m_low_pct",
            "timing_close_inside_recent_range_flag": "timing_close_inside_15m_range_flag",
            "timing_sweep_high_reject_flag": "timing_sweep_high_reject_15m_flag",
            "timing_sweep_low_reject_flag": "timing_sweep_low_reject_15m_flag",
            "timing_momentum_deceleration": "timing_momentum_deceleration_1m",
            "timing_short_exhaustion_flag": "timing_short_exhaustion_1m_flag",
            "timing_long_exhaustion_flag": "timing_long_exhaustion_1m_flag",
            "timing_short_rejection_flag": "timing_short_rejection_1m_flag",
            "timing_long_rejection_flag": "timing_long_rejection_1m_flag",
        }
    )

    frame_5m_features = frame_5m_local[
        [
            "close_time",
            "timing_5m_ret_last_pct",
            "timing_5m_ret_3c_pct",
            "timing_5m_upper_wick_ratio",
            "timing_5m_lower_wick_ratio",
            "timing_5m_relative_volume",
            "timing_5m_momentum_deceleration",
        ]
    ].rename(
        columns={
            "timing_5m_ret_last_pct": "timing_ret_5m_last_pct",
            "timing_5m_ret_3c_pct": "timing_ret_5m_3c_pct",
            "timing_5m_upper_wick_ratio": "timing_upper_wick_ratio_5m",
            "timing_5m_lower_wick_ratio": "timing_lower_wick_ratio_5m",
            "timing_5m_relative_volume": "timing_relative_volume_5m",
            "timing_5m_momentum_deceleration": "timing_momentum_deceleration_5m",
        }
    )
    return minute_features, frame_5m_features


def compute_candle_features(frame: pd.DataFrame, prefix: str, recent_window: int) -> pd.DataFrame:
    prepared = frame.copy()
    close_series = pd.to_numeric(prepared["close"], errors="coerce")
    open_series = pd.to_numeric(prepared["open"], errors="coerce")
    high_series = pd.to_numeric(prepared["high"], errors="coerce")
    low_series = pd.to_numeric(prepared["low"], errors="coerce")
    volume_series = pd.to_numeric(prepared["volume"], errors="coerce")

    candle_range = (high_series - low_series).replace(0, np.nan)
    prepared[f"{prefix}_ret_last_pct"] = close_series.pct_change(1) * 100.0
    prepared[f"{prefix}_ret_3c_pct"] = close_series.pct_change(3) * 100.0
    prepared[f"{prefix}_upper_wick_ratio"] = (high_series - np.maximum(open_series, close_series)) / candle_range
    prepared[f"{prefix}_lower_wick_ratio"] = (np.minimum(open_series, close_series) - low_series) / candle_range
    prepared[f"{prefix}_relative_volume"] = volume_series / volume_series.rolling(20, min_periods=5).mean()

    recent_high = high_series.shift(1).rolling(recent_window, min_periods=3).max()
    recent_low = low_series.shift(1).rolling(recent_window, min_periods=3).min()
    prepared[f"{prefix}_distance_to_recent_high_pct"] = (recent_high - close_series) / close_series * 100.0
    prepared[f"{prefix}_distance_to_recent_low_pct"] = (close_series - recent_low) / close_series * 100.0
    prepared[f"{prefix}_close_inside_recent_range_flag"] = close_series.between(recent_low, recent_high, inclusive="both")
    prepared[f"{prefix}_sweep_high_reject_flag"] = (high_series > recent_high) & (close_series < recent_high)
    prepared[f"{prefix}_sweep_low_reject_flag"] = (low_series < recent_low) & (close_series > recent_low)

    momentum_base = (prepared[f"{prefix}_ret_3c_pct"].abs() / 3.0).replace(0, np.nan)
    prepared[f"{prefix}_momentum_deceleration"] = prepared[f"{prefix}_ret_last_pct"].abs() / momentum_base
    prepared[f"{prefix}_short_exhaustion_flag"] = (
        (prepared[f"{prefix}_ret_3c_pct"] > 0)
        & (prepared[f"{prefix}_ret_last_pct"] <= 0)
    )
    prepared[f"{prefix}_long_exhaustion_flag"] = (
        (prepared[f"{prefix}_ret_3c_pct"] < 0)
        & (prepared[f"{prefix}_ret_last_pct"] >= 0)
    )
    prepared[f"{prefix}_short_rejection_flag"] = (
        (prepared[f"{prefix}_upper_wick_ratio"] >= 0.45)
        & (close_series <= open_series)
    )
    prepared[f"{prefix}_long_rejection_flag"] = (
        (prepared[f"{prefix}_lower_wick_ratio"] >= 0.45)
        & (close_series >= open_series)
    )
    return prepared


def attach_timing_features(
    entries: pd.DataFrame,
    minute_features: pd.DataFrame,
    frame_5m_features: pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()

    minute_lookup = minute_features.rename(columns={"close_time": "timing_feature_time_1m"})
    merged = pd.merge_asof(
        entries.sort_values(time_column),
        minute_lookup.sort_values("timing_feature_time_1m"),
        left_on=time_column,
        right_on="timing_feature_time_1m",
        direction="backward",
    ).drop(columns=["timing_feature_time_1m"])

    frame_5m_lookup = frame_5m_features.rename(columns={"close_time": "timing_feature_time_5m"})
    merged = pd.merge_asof(
        merged.sort_values(time_column),
        frame_5m_lookup.sort_values("timing_feature_time_5m"),
        left_on=time_column,
        right_on="timing_feature_time_5m",
        direction="backward",
    ).drop(columns=["timing_feature_time_5m"])

    boolean_columns = [spec["key"] for spec in TIMING_FEATURE_SPECS if spec["type"] == "boolean"]
    for column in boolean_columns:
        if column in merged.columns:
            merged[column] = merged[column].fillna(False).astype(bool)
    return merged


def run_cluster_tightening(
    cluster_id: int,
    real_features: pd.DataFrame,
    signal_features: pd.DataFrame,
    logger: logging.Logger,
) -> dict:
    reference = real_features.loc[real_features["cluster_id"] == cluster_id].copy()
    base_stage_signals = signal_features.loc[signal_features[f"in_{TIMING_STAGE_ID}"] == True].copy()
    if reference.empty or base_stage_signals.empty:
        empty = pd.DataFrame()
        return {
            "ranking": empty,
            "candidates": empty,
            "summary_row": {},
            "stage_rows": empty,
        }

    reference_profile = reference.copy()
    if "entry_time" not in reference_profile.columns and "opened_dt_utc" in reference_profile.columns:
        reference_profile = reference_profile.rename(columns={"opened_dt_utc": "entry_time"})
    if "fast_positive_reaction_flag" not in reference_profile.columns and "group_fast_positive_reaction" in reference_profile.columns:
        reference_profile["fast_positive_reaction_flag"] = reference_profile["group_fast_positive_reaction"].fillna(False).astype(bool)
    real_summary = summarize_signal_like_profiles(reference_profile)
    base_summary = summarize_signal_like_profiles(base_stage_signals)
    base_similarity = compare_profiles(real_summary, base_summary)

    logger.info("Extracting timing layer for ETH cluster %s from %s stage-A base signals", cluster_id, len(base_stage_signals))
    single_df, single_candidates = rank_single_feature_candidates(
        cluster_id,
        reference,
        base_stage_signals,
        real_summary,
        base_summary,
        base_similarity,
    )
    pair_df, pair_candidates = evaluate_pair_candidates(
        cluster_id,
        single_df,
        single_candidates,
        base_stage_signals,
        real_summary,
        base_summary,
        base_similarity,
    )

    candidate_objects = {candidate["candidate_id"]: candidate for candidate in [*single_candidates, *pair_candidates]}
    candidate_frames = [frame for frame in [single_df, pair_df] if not frame.empty]
    candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False) if candidate_frames else pd.DataFrame()
    candidates_df = candidates_df.sort_values(["ranking_score", "similarity_to_real_cluster_profile", "signal_count_after_filter"], ascending=[False, False, False]).reset_index(drop=True)

    best_candidate = select_best_candidate(candidates_df, candidate_objects)
    stage_rows = evaluate_candidate_across_stages(
        cluster_id,
        best_candidate,
        signal_features,
        real_summary,
    )
    summary_row = build_cluster_summary_row(
        cluster_id,
        reference,
        base_stage_signals,
        real_summary,
        base_summary,
        base_similarity,
        best_candidate,
        stage_rows,
    )
    return {
        "ranking": single_df,
        "candidates": candidates_df,
        "summary_row": summary_row,
        "stage_rows": stage_rows,
    }


def rank_single_feature_candidates(
    cluster_id: int,
    reference: pd.DataFrame,
    base_stage_signals: pd.DataFrame,
    real_summary: dict,
    base_summary: dict,
    base_similarity: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    candidates = []
    for spec in TIMING_FEATURE_SPECS:
        candidate = derive_single_feature_candidate(cluster_id, spec, reference, base_stage_signals)
        if candidate is None:
            continue
        filtered = apply_conditions(base_stage_signals, candidate["conditions"])
        if filtered.empty:
            continue
        row = build_candidate_metrics_row(
            cluster_id,
            filtered,
            base_stage_signals,
            real_summary,
            base_summary,
            base_similarity,
            candidate,
        )
        rows.append(row)
        candidates.append(candidate)

    ranking_df = pd.DataFrame(rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(["ranking_score", "similarity_improvement_vs_base", "reduction_vs_base_filter"], ascending=[False, False, False]).reset_index(drop=True)
    return ranking_df, candidates


def derive_single_feature_candidate(
    cluster_id: int,
    spec: dict,
    reference: pd.DataFrame,
    base_stage_signals: pd.DataFrame,
) -> dict | None:
    key = spec["key"]
    label = spec["label"]
    feature_type = spec["type"]
    family = spec["family"]

    if feature_type == "numeric":
        ref_values = pd.to_numeric(reference[key], errors="coerce").dropna().to_numpy(dtype=float)
        base_values = pd.to_numeric(base_stage_signals[key], errors="coerce").dropna().to_numpy(dtype=float)
        if ref_values.size < 5 or base_values.size < 20:
            return None

        ref_median = float(np.nanmedian(ref_values))
        base_median = float(np.nanmedian(base_values))
        if np.isclose(ref_median, base_median, atol=1e-9):
            return None

        direction = "ge" if ref_median > base_median else "le"
        threshold = float(np.nanpercentile(ref_values, 25 if direction == "ge" else 75))
        comparison_mask = apply_numeric_condition(pd.to_numeric(base_stage_signals[key], errors="coerce"), direction, threshold)
        retained_share = comparison_mask.fillna(False).mean()
        if retained_share <= 0.01 or retained_share >= 0.98:
            return None

        mann_whitney = mann_whitney_summary(ref_values, base_values)
        overlap = percentile_overlap_pct(ref_values, base_values)
        text = f"{label} {'>=' if direction == 'ge' else '<='} {format_threshold(threshold)}"
        return {
            "candidate_id": f"cluster_{cluster_id}_{key}",
            "cluster_id": cluster_id,
            "mode": "single",
            "feature_keys": [key],
            "families": [family],
            "conditions": [{"feature_key": key, "rule_type": direction, "threshold": threshold, "text": text}],
            "timing_rule_text": text,
            "feature_label": label,
            "feature_family": family,
            "feature_type": feature_type,
            "reference_median": ref_median,
            "base_median": base_median,
            "effect_size": mann_whitney["rank_biserial"],
            "mann_whitney_p_value": mann_whitney["p_value"],
            "percentile_overlap_pct": overlap,
        }

    ref_bool = reference[key].fillna(False).astype(bool)
    base_bool = base_stage_signals[key].fillna(False).astype(bool)
    ref_rate = float(ref_bool.mean())
    base_rate = float(base_bool.mean())
    if abs(ref_rate - base_rate) < 0.08:
        return None
    desired = ref_rate > base_rate
    retained_share = float((base_bool == desired).mean())
    if retained_share <= 0.01 or retained_share >= 0.98:
        return None
    text = f"{label} = {desired}"
    return {
        "candidate_id": f"cluster_{cluster_id}_{key}",
        "cluster_id": cluster_id,
        "mode": "single",
        "feature_keys": [key],
        "families": [family],
        "conditions": [{"feature_key": key, "rule_type": "equals", "value": desired, "text": text}],
        "timing_rule_text": text,
        "feature_label": label,
        "feature_family": family,
        "feature_type": feature_type,
        "reference_rate": ref_rate,
        "base_rate": base_rate,
        "effect_size": ref_rate - base_rate,
        "mann_whitney_p_value": np.nan,
        "percentile_overlap_pct": np.nan,
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


def build_candidate_metrics_row(
    cluster_id: int,
    filtered: pd.DataFrame,
    base_stage_signals: pd.DataFrame,
    real_summary: dict,
    base_summary: dict,
    base_similarity: dict,
    candidate: dict,
) -> dict:
    filtered_summary = summarize_signal_like_profiles(filtered)
    filtered_similarity = compare_profiles(real_summary, filtered_summary)
    ranking_score = candidate_ranking_score(
        filtered_summary,
        base_summary,
        filtered_similarity["score"],
        base_similarity["score"],
        len(filtered),
        len(base_stage_signals),
        int(real_summary.get("signal_count", 0)),
    )
    row = {
        "cluster_id": cluster_id,
        "candidate_id": candidate["candidate_id"],
        "mode": candidate["mode"],
        "condition_count": len(candidate["conditions"]),
        "timing_rule_text": candidate["timing_rule_text"],
        "signal_count_after_filter": int(len(filtered)),
        "reduction_vs_base_filter": 1 - safe_divide(len(filtered), len(base_stage_signals)),
        "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
        "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
        "pct_positive_1h": filtered_summary.get("pct_positive_after_60m"),
        "median_MAE": filtered_summary.get("median_MAE_pct"),
        "median_MFE": filtered_summary.get("median_MFE_pct"),
        "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
        "median_return_4h": filtered_summary.get("median_return_after_240m_pct"),
        "similarity_to_real_cluster_profile": filtered_similarity["score"],
        "similarity_label": filtered_similarity["label"],
        "similarity_improvement_vs_base": safe_number(filtered_similarity["score"]) - safe_number(base_similarity["score"]),
        "effect_size": candidate.get("effect_size"),
        "mann_whitney_p_value": candidate.get("mann_whitney_p_value"),
        "percentile_overlap_pct": candidate.get("percentile_overlap_pct"),
        "feature_label": candidate.get("feature_label"),
        "feature_family": candidate.get("feature_family"),
        "feature_type": candidate.get("feature_type"),
        "ranking_score": ranking_score,
    }
    if "reference_median" in candidate:
        row["reference_feature_level"] = candidate["reference_median"]
        row["base_feature_level"] = candidate["base_median"]
    if "reference_rate" in candidate:
        row["reference_feature_level"] = candidate["reference_rate"]
        row["base_feature_level"] = candidate["base_rate"]
    return row


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0


def candidate_ranking_score(
    filtered_summary: dict,
    base_summary: dict,
    filtered_similarity: float,
    base_similarity: float,
    filtered_count: int,
    base_count: int,
    real_count: int,
) -> float:
    reduction = 1 - safe_divide(filtered_count, base_count)
    pos5_improvement = safe_number(filtered_summary.get("pct_positive_after_5m")) - safe_number(base_summary.get("pct_positive_after_5m"))
    pos15_improvement = safe_number(filtered_summary.get("pct_positive_after_15m")) - safe_number(base_summary.get("pct_positive_after_15m"))
    similarity_improvement = safe_number(filtered_similarity) - safe_number(base_similarity)
    base_mae = safe_number(base_summary.get("median_MAE_pct"))
    filtered_mae = safe_number(filtered_summary.get("median_MAE_pct"))
    mae_improvement = safe_divide(base_mae - filtered_mae, base_mae if base_mae != 0 else np.nan)
    score = (
        similarity_improvement * 0.50
        + pos5_improvement * 0.15
        + pos15_improvement * 0.10
        + safe_number(mae_improvement) * 0.15
        + safe_number(reduction) * 0.10
    )
    if filtered_count < max(24, int(real_count * 0.75)):
        score -= 0.12
    if filtered_count < max(12, int(real_count * 0.40)):
        score -= 0.20
    return score


def evaluate_pair_candidates(
    cluster_id: int,
    single_df: pd.DataFrame,
    single_candidates: list[dict],
    base_stage_signals: pd.DataFrame,
    real_summary: dict,
    base_summary: dict,
    base_similarity: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    if single_df.empty or not single_candidates:
        return pd.DataFrame(), []

    ranked_ids = single_df.sort_values("ranking_score", ascending=False)["candidate_id"].head(6).tolist()
    candidate_map = {candidate["candidate_id"]: candidate for candidate in single_candidates}
    top_candidates = [candidate_map[candidate_id] for candidate_id in ranked_ids if candidate_id in candidate_map]

    rows = []
    pairs = []
    for candidate_a, candidate_b in itertools.combinations(top_candidates, 2):
        if set(candidate_a["families"]) == set(candidate_b["families"]):
            continue
        combined = {
            "candidate_id": f"{candidate_a['candidate_id']}__{candidate_b['candidate_id']}",
            "cluster_id": cluster_id,
            "mode": "pair",
            "feature_keys": [*candidate_a["feature_keys"], *candidate_b["feature_keys"]],
            "families": [*candidate_a["families"], *candidate_b["families"]],
            "conditions": [*candidate_a["conditions"], *candidate_b["conditions"]],
            "timing_rule_text": f"{candidate_a['timing_rule_text']} AND {candidate_b['timing_rule_text']}",
            "feature_label": f"{candidate_a['feature_label']} + {candidate_b['feature_label']}",
            "feature_family": f"{candidate_a['feature_family']} + {candidate_b['feature_family']}",
            "feature_type": "combined",
        }
        filtered = apply_conditions(base_stage_signals, combined["conditions"])
        if filtered.empty:
            continue
        row = build_candidate_metrics_row(
            cluster_id,
            filtered,
            base_stage_signals,
            real_summary,
            base_summary,
            base_similarity,
            combined,
        )
        rows.append(row)
        pairs.append(combined)

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["ranking_score", "similarity_improvement_vs_base", "reduction_vs_base_filter"], ascending=[False, False, False]).reset_index(drop=True)
    return pair_df, pairs


def select_best_candidate(candidates_df: pd.DataFrame, candidate_objects: dict[str, dict]) -> dict:
    if candidates_df.empty:
        return {
            "candidate_id": "none",
            "timing_rule_text": "No tightened timing layer found",
            "conditions": [],
            "mode": "none",
        }

    ordered = candidates_df.sort_values(
        ["ranking_score", "similarity_to_real_cluster_profile", "signal_count_after_filter", "condition_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    positive_singles = ordered.loc[
        (ordered["condition_count"] == 1)
        & (ordered["similarity_improvement_vs_base"] >= 0)
    ].sort_values(
        ["similarity_to_real_cluster_profile", "ranking_score", "signal_count_after_filter"],
        ascending=[False, False, False],
    )
    positive_pairs = ordered.loc[
        (ordered["condition_count"] == 2)
        & (ordered["similarity_improvement_vs_base"] >= 0)
    ].sort_values(
        ["similarity_to_real_cluster_profile", "ranking_score", "signal_count_after_filter"],
        ascending=[False, False, False],
    )
    all_singles = ordered.loc[ordered["condition_count"] == 1]

    if not positive_singles.empty:
        chosen = positive_singles.iloc[0]
        if not positive_pairs.empty:
            pair = positive_pairs.iloc[0]
            if (
                pair["similarity_to_real_cluster_profile"] >= chosen["similarity_to_real_cluster_profile"] + 0.02
                or (
                    pair["similarity_to_real_cluster_profile"] >= chosen["similarity_to_real_cluster_profile"] + 0.01
                    and pair["ranking_score"] >= chosen["ranking_score"] + 0.03
                )
            ):
                chosen = pair
    elif not all_singles.empty:
        chosen = all_singles.sort_values(
            ["similarity_to_real_cluster_profile", "ranking_score", "signal_count_after_filter"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        chosen = ordered.iloc[0]

    return candidate_objects.get(chosen["candidate_id"], {"candidate_id": chosen["candidate_id"], "timing_rule_text": chosen["timing_rule_text"], "conditions": [], "mode": chosen["mode"]})


def evaluate_candidate_across_stages(
    cluster_id: int,
    candidate: dict,
    signal_features: pd.DataFrame,
    real_summary: dict,
) -> pd.DataFrame:
    filtered_all = apply_conditions(signal_features, candidate.get("conditions", [])) if candidate.get("conditions") else signal_features.copy()
    rows = []
    for stage_id in FINAL_STAGE_IDS:
        base_stage = signal_features.loc[signal_features[f"in_{stage_id}"] == True].copy()
        filtered_stage = filtered_all.loc[filtered_all[f"in_{stage_id}"] == True].copy()
        filtered_summary = summarize_signal_like_profiles(filtered_stage)
        base_summary = summarize_signal_like_profiles(base_stage)
        similarity = compare_profiles(real_summary, filtered_summary)
        rows.append(
            {
                "cluster_id": cluster_id,
                "stage_id": stage_id,
                "stage_label": stage_id.replace("_", " ").title(),
                "timing_rule_text": candidate.get("timing_rule_text"),
                "timing_condition_count": len(candidate.get("conditions", [])),
                "signal_count_after_filter": int(len(filtered_stage)),
                "base_signal_count": int(len(base_stage)),
                "reduction_vs_base_stage": 1 - safe_divide(len(filtered_stage), len(base_stage)),
                "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
                "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
                "pct_positive_1h": filtered_summary.get("pct_positive_after_60m"),
                "median_MAE": filtered_summary.get("median_MAE_pct"),
                "median_MFE": filtered_summary.get("median_MFE_pct"),
                "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
                "median_return_4h": filtered_summary.get("median_return_after_240m_pct"),
                "similarity_to_real_cluster_profile": similarity["score"],
                "similarity_label": similarity["label"],
                "base_similarity": compare_profiles(real_summary, base_summary)["score"],
            }
        )
    return pd.DataFrame(rows)


def build_cluster_summary_row(
    cluster_id: int,
    reference: pd.DataFrame,
    base_stage_signals: pd.DataFrame,
    real_summary: dict,
    base_summary: dict,
    base_similarity: dict,
    best_candidate: dict,
    stage_rows: pd.DataFrame,
) -> dict:
    stage_a = stage_rows.loc[stage_rows["stage_id"] == TIMING_STAGE_ID].iloc[0]
    backtest_ready = bool(
        pd.notna(stage_a["similarity_to_real_cluster_profile"])
        and stage_a["similarity_to_real_cluster_profile"] >= 0.70
        and stage_a["signal_count_after_filter"] >= max(30, int(len(reference) * 0.75))
    )
    return {
        "cluster_id": cluster_id,
        "real_trade_count": int(len(reference)),
        "base_stage_signal_count": int(len(base_stage_signals)),
        "base_stage_similarity": base_similarity["score"],
        "base_stage_pct_positive_5m": base_summary.get("pct_positive_after_5m"),
        "base_stage_pct_positive_15m": base_summary.get("pct_positive_after_15m"),
        "base_stage_pct_positive_1h": base_summary.get("pct_positive_after_60m"),
        "base_stage_median_MAE": base_summary.get("median_MAE_pct"),
        "tightened_signal_count": stage_a["signal_count_after_filter"],
        "tightened_reduction_vs_base": stage_a["reduction_vs_base_stage"],
        "tightened_similarity": stage_a["similarity_to_real_cluster_profile"],
        "tightened_similarity_label": stage_a["similarity_label"],
        "tightened_pct_positive_5m": stage_a["pct_positive_5m"],
        "tightened_pct_positive_15m": stage_a["pct_positive_15m"],
        "tightened_pct_positive_1h": stage_a["pct_positive_1h"],
        "tightened_median_MAE": stage_a["median_MAE"],
        "tightened_median_MFE": stage_a["median_MFE"],
        "tightened_median_return_1h": stage_a["median_return_1h"],
        "tightened_median_return_4h": stage_a["median_return_4h"],
        "timing_rule_text": best_candidate.get("timing_rule_text"),
        "timing_condition_count": len(best_candidate.get("conditions", [])),
        "timing_mode": best_candidate.get("mode"),
        "timing_feature_keys": ", ".join(best_candidate.get("feature_keys", [])),
        "backtest_readiness_flag": backtest_ready,
    }


def format_threshold(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def build_tightening_section(
    summary_df: pd.DataFrame,
    stage_comparison: pd.DataFrame,
    feature_ranking: pd.DataFrame,
) -> str:
    if summary_df.empty:
        return (
            "ETH FILTER TIGHTENING / TIMING LAYER\n"
            "------------------------------------\n"
            "Timing-layer extraction could not be completed.\n"
        )

    lines = []
    for _, row in summary_df.sort_values("cluster_id").iterrows():
        cluster_ranking = feature_ranking.loc[feature_ranking["cluster_id"] == row["cluster_id"]].head(3)
        helpful_features = ", ".join(cluster_ranking["feature_label"].astype(str).tolist()) if not cluster_ranking.empty else "n/a"
        lines.append(
            f"- Cluster {fmt_int(row['cluster_id'])}: top timing features = {helpful_features}; "
            f"best tightened filter = {row['timing_rule_text']}; "
            f"Stage A signals {fmt_int(row['base_stage_signal_count'])} -> {fmt_int(row['tightened_signal_count'])}; "
            f"similarity {fmt_number(row['base_stage_similarity'])} -> {fmt_number(row['tightened_similarity'])}; "
            f"positive after 5m {pct(row['base_stage_pct_positive_5m'])} -> {pct(row['tightened_pct_positive_5m'])}; "
            f"median MAE {fmt_number(row['base_stage_median_MAE'])}% -> {fmt_number(row['tightened_median_MAE'])}%."
        )

    stage_a = stage_comparison.loc[stage_comparison["stage_id"] == TIMING_STAGE_ID]
    confidence = determine_tightening_confidence(summary_df, stage_comparison)
    backtest_ready = bool(summary_df["backtest_readiness_flag"].fillna(False).all()) if not summary_df.empty else False
    final_verdict = (
        "The tightened replication candidates are now close enough to justify a small exploratory TP/SL test."
        if backtest_ready
        else "The tightened filters are better timing approximations, but they still do not look close enough to the trader's real entry profile for a confident backtest."
    )

    report = f"""
ETH FILTER TIGHTENING / TIMING LAYER
------------------------------------
This stage keeps the broad ETH context filters and adds only a light timing layer on top. Feature selection is done on Stage A only, then the chosen tightened filters are checked across Stage A, Stage B, and Stage C.

Main findings:
{chr(10).join(lines)}

Interpretation:
- The most useful timing features are the ones that reduce signal density while nudging early reaction and MAE toward the real cluster trades.
- The tightened filters should be treated as timing layer approximations, not as the trader's exact entry signal.
- Confidence level: {confidence}.

Final verdict:
- {final_verdict}
""".strip()
    return report + "\n"


def determine_tightening_confidence(summary_df: pd.DataFrame, stage_comparison: pd.DataFrame) -> str:
    if summary_df.empty:
        return "low"
    improvement = (summary_df["tightened_similarity"] - summary_df["base_stage_similarity"]).median()
    reduction = summary_df["tightened_reduction_vs_base"].median()
    external = stage_comparison.loc[stage_comparison["stage_id"].isin(["stage_b", "stage_c"]), "similarity_to_real_cluster_profile"].median()
    if pd.notna(improvement) and improvement >= 0.08 and pd.notna(reduction) and reduction >= 0.50 and pd.notna(external) and external >= 0.55:
        return "high"
    if pd.notna(improvement) and improvement >= 0.03 and pd.notna(reduction) and reduction >= 0.25:
        return "medium"
    return "low"


def create_eth_tightening_plots(
    feature_ranking: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if feature_ranking.empty or summary_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_importance = plot_timing_feature_importance(feature_ranking)
    importance_path = plots_dir / "eth_timing_feature_importance.png"
    fig_importance.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig_importance)
    logger.info("Saved plot: %s", importance_path.name)

    fig_improvement = plot_tightened_filter_improvement(summary_df)
    improvement_path = plots_dir / "eth_tightened_filter_improvement.png"
    fig_improvement.savefig(improvement_path, dpi=150, bbox_inches="tight")
    plt.close(fig_improvement)
    logger.info("Saved plot: %s", improvement_path.name)


def plot_timing_feature_importance(feature_ranking: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, cluster_id in zip(axes, [1, 2], strict=False):
        subset = feature_ranking.loc[feature_ranking["cluster_id"] == cluster_id].head(8).sort_values("ranking_score")
        ax.barh(subset["feature_label"], subset["ranking_score"], color="#1f77b4")
        ax.set_title(f"Cluster {cluster_id} timing feature ranking")
        ax.set_xlabel("Ranking score")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("ETH Timing Layer: Most Useful Single Features")
    return fig


def plot_tightened_filter_improvement(summary_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ordered = summary_df.sort_values("cluster_id").reset_index(drop=True)

    for row_idx, (_, row) in enumerate(ordered.iterrows()):
        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]

        left_ax.bar(["Base", "Tightened"], [row["base_stage_signal_count"], row["tightened_signal_count"]], color=["#bbbbbb", "#1f77b4"])
        left_ax.set_title(f"Cluster {int(row['cluster_id'])}: Signal count")
        left_ax.set_ylabel("Stage A signals")
        left_ax.grid(axis="y", alpha=0.25)

        metrics = ["Similarity", "Positive 5m", "Positive 15m"]
        base_values = [row["base_stage_similarity"], row["base_stage_pct_positive_5m"], row["base_stage_pct_positive_15m"]]
        tight_values = [row["tightened_similarity"], row["tightened_pct_positive_5m"], row["tightened_pct_positive_15m"]]
        x = np.arange(len(metrics))
        width = 0.35
        right_ax.bar(x - width / 2, base_values, width=width, label="Base", color="#bbbbbb")
        right_ax.bar(x + width / 2, tight_values, width=width, label="Tightened", color="#ff7f0e")
        right_ax.set_xticks(x)
        right_ax.set_xticklabels(metrics)
        right_ax.set_ylim(0, 1)
        right_ax.set_title(f"Cluster {int(row['cluster_id'])}: Profile improvement")
        right_ax.grid(axis="y", alpha=0.25)
        right_ax.legend()

    fig.suptitle("ETH Timing Layer: Base vs Tightened Filters")
    return fig
