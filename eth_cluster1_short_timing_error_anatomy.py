from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster1_short_overlap_validation import load_proxy_market_frames
from eth_replication_hypothesis_test import ETH_SYMBOL
from utils import fmt_int, fmt_number, pct


SYNC_THRESHOLD_MINUTES = 5.0
SAME_ZONE_PRICE_THRESHOLD_PCT = 0.25
NEAR_ZONE_PRICE_THRESHOLD_PCT = 0.50
LATE_MOVE_STARTED_PRICE_THRESHOLD_PCT = -0.10
MISMATCH_TYPE_ORDER = [
    "same_moment_close_match",
    "same_zone_wrong_moment",
    "early_before_failure",
    "early_near_zone",
    "late_after_move_started",
    "late_near_zone",
    "far_mismatch",
]
TIMING_BUCKET_ORDER = ["very_early", "early", "near_sync", "late", "very_late"]


def run_cluster1_short_timing_error_anatomy_analysis(
    trades_df: pd.DataFrame,
    overlap_matches_df: pd.DataFrame,
    refinement_results_df: pd.DataFrame,
    refinement_best_matches_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "CLUSTER 1 SHORT TIMING ERROR ANATOMY - WHY THE PROXY MISSES THE MOMENT\n"
        "---------------------------------------------------------------------\n"
        "No sufficient matched Cluster 1 short overlap data was available for timing-error diagnosis.\n"
    )

    base_matches = filter_assigned_matches(overlap_matches_df)
    if base_matches.empty:
        return empty_result(empty, empty_section)

    variant_rows = [{"variant_id": "base_proxy", "variant_label": "Base proxy", "frame": base_matches}]
    best_variant = extract_best_variant_info(refinement_results_df)
    if best_variant is not None and best_variant["variant_id"] != "base_proxy":
        refined_matches = filter_assigned_matches(refinement_best_matches_df)
        if not refined_matches.empty:
            variant_rows.append(
                {
                    "variant_id": best_variant["variant_id"],
                    "variant_label": best_variant["variant_label"],
                    "frame": refined_matches,
                }
            )

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    minute_frame, frame_5m, _stages = load_proxy_market_frames(eth_trades, logger)

    if minute_frame.empty or frame_5m.empty:
        section = (
            "CLUSTER 1 SHORT TIMING ERROR ANATOMY - WHY THE PROXY MISSES THE MOMENT\n"
            "---------------------------------------------------------------------\n"
            "Timing-error diagnosis could not run because ETH candle context was unavailable.\n"
        )
        return empty_result(empty, section)

    logger.info(
        "Running Cluster 1 short timing-error anatomy on %s matched base cases%s",
        len(base_matches),
        f" and {len(variant_rows[1]['frame'])} matched refined cases" if len(variant_rows) > 1 else "",
    )

    minute_features = build_anchor_1m_features(minute_frame)
    five_minute_features = build_anchor_5m_features(frame_5m)

    timing_frames = []
    for variant in variant_rows:
        timing_frames.append(
            build_timing_error_matches(
                variant["frame"],
                minute_features,
                five_minute_features,
                variant["variant_id"],
                variant["variant_label"],
            )
        )

    timing_matches = pd.concat(timing_frames, ignore_index=True, sort=False) if timing_frames else pd.DataFrame()
    if timing_matches.empty:
        return empty_result(empty, empty_section)

    timing_summary = build_timing_error_summary(timing_matches)
    timing_quality = build_timing_error_quality(timing_matches)
    section_text = build_timing_error_section(timing_matches, timing_summary, timing_quality)

    return {
        "matches": timing_matches,
        "summary": timing_summary,
        "quality": timing_quality,
        "section_text": section_text,
    }


def empty_result(empty: pd.DataFrame, section_text: str) -> dict:
    return {
        "matches": empty,
        "summary": empty,
        "quality": empty,
        "section_text": section_text,
    }


def filter_assigned_matches(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    filtered = frame.copy()
    filtered["assigned_match_flag"] = coerce_bool_series(filtered.get("assigned_match_flag", False))
    filtered = filtered.loc[filtered["assigned_match_flag"]].copy()
    if filtered.empty:
        return filtered

    bool_columns = [
        "same_short_move_15m_flag",
        "same_short_move_30m_flag",
        "same_short_move_60m_flag",
    ]
    for column in bool_columns:
        if column in filtered.columns:
            filtered[column] = coerce_bool_series(filtered[column])

    for column in [
        "effective_entry_time",
        "matched_model_entry_time",
        "opened_dt_utc",
        "proxy_fill_time",
        "closed_dt_utc",
    ]:
        if column in filtered.columns:
            filtered[column] = pd.to_datetime(filtered[column], utc=True, errors="coerce")

    numeric_columns = [
        "matched_signed_time_distance_minutes",
        "matched_time_distance_minutes",
        "matched_price_distance_pct",
        "matched_model_entry_price",
        "effective_entry_price",
        "real_window_return_15m_pct",
        "real_window_return_30m_pct",
        "real_window_return_60m_pct",
        "model_window_return_15m_pct",
        "model_window_return_30m_pct",
        "model_window_return_60m_pct",
        "real_window_mae_15m_pct",
        "real_window_mae_30m_pct",
        "real_window_mae_60m_pct",
        "model_window_mae_15m_pct",
        "model_window_mae_30m_pct",
        "model_window_mae_60m_pct",
        "real_window_mfe_15m_pct",
        "real_window_mfe_30m_pct",
        "real_window_mfe_60m_pct",
        "model_window_mfe_15m_pct",
        "model_window_mfe_30m_pct",
        "model_window_mfe_60m_pct",
    ]
    for column in numeric_columns:
        if column in filtered.columns:
            filtered[column] = pd.to_numeric(filtered[column], errors="coerce")

    return filtered.sort_values("effective_entry_time").reset_index(drop=True)


def extract_best_variant_info(results_df: pd.DataFrame) -> dict | None:
    if results_df.empty:
        return None
    rows = results_df.copy()
    if "is_best_variant" in rows.columns:
        rows["is_best_variant"] = coerce_bool_series(rows["is_best_variant"])
        best = rows.loc[rows["is_best_variant"]]
        if not best.empty:
            row = best.iloc[0]
            return {
                "variant_id": str(row.get("variant_id", "base_proxy")),
                "variant_label": str(row.get("variant_label", "Best refined variant")),
            }
    best_row = rows.iloc[0]
    return {
        "variant_id": str(best_row.get("variant_id", "base_proxy")),
        "variant_label": str(best_row.get("variant_label", "Best refined variant")),
    }


def build_anchor_1m_features(minute_frame: pd.DataFrame) -> pd.DataFrame:
    if minute_frame.empty:
        return pd.DataFrame()

    frame = minute_frame[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["ret_candle_pct"] = (frame["close"] - frame["open"]) / frame["open"] * 100.0
    frame["ret_3m_pct"] = frame["close"].pct_change(3) * 100.0
    frame["upper_wick_ratio"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / price_range
    frame["close_position_in_range"] = (frame["close"] - frame["low"]) / price_range
    frame["feature_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    return frame[
        [
            "feature_time",
            "ret_candle_pct",
            "ret_3m_pct",
            "upper_wick_ratio",
            "close_position_in_range",
        ]
    ].copy()


def build_anchor_5m_features(frame_5m: pd.DataFrame) -> pd.DataFrame:
    if frame_5m.empty:
        return pd.DataFrame()

    frame = frame_5m[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    avg_volume_20 = frame["volume"].rolling(20, min_periods=5).mean()
    prior_5m_high = frame["high"].shift(1)
    prior_15m_high = frame["high"].shift(1).rolling(3, min_periods=1).max()

    frame["ret_candle_pct"] = (frame["close"] - frame["open"]) / frame["open"] * 100.0
    frame["ret_15m_pct"] = frame["close"].pct_change(3) * 100.0
    frame["relative_volume"] = frame["volume"] / avg_volume_20
    frame["upper_wick_ratio"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / price_range
    frame["body_range_ratio"] = (frame["close"] - frame["open"]).abs() / price_range
    frame["close_position_in_range"] = (frame["close"] - frame["low"]) / price_range
    frame["prior_5m_high"] = prior_5m_high
    frame["prior_15m_high"] = prior_15m_high
    frame["broke_prior_5m_high_flag"] = frame["high"] > prior_5m_high
    frame["broke_prior_15m_high_flag"] = frame["high"] > prior_15m_high
    frame["close_back_below_prior_5m_high_flag"] = frame["broke_prior_5m_high_flag"] & (frame["close"] < prior_5m_high)
    frame["close_back_below_prior_15m_high_flag"] = frame["broke_prior_15m_high_flag"] & (frame["close"] < prior_15m_high)
    frame["dist_below_prior_5m_high_pct"] = (prior_5m_high - frame["close"]) / frame["close"] * 100.0
    frame["dist_below_prior_15m_high_pct"] = (prior_15m_high - frame["close"]) / frame["close"] * 100.0
    frame["push_still_active_flag"] = (
        frame["ret_candle_pct"].gt(0)
        & ~frame["close_back_below_prior_5m_high_flag"].fillna(False)
        & ~frame["close_back_below_prior_15m_high_flag"].fillna(False)
    )

    failure_components = [
        frame["upper_wick_ratio"].ge(0.25).fillna(False).astype(int),
        frame["ret_candle_pct"].le(0).fillna(False).astype(int),
        frame["close_back_below_prior_5m_high_flag"].fillna(False).astype(int),
        frame["close_back_below_prior_15m_high_flag"].fillna(False).astype(int),
    ]
    frame["local_failure_score"] = sum(failure_components)
    frame["local_failure_flag"] = frame["local_failure_score"] >= 2
    frame["feature_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")

    return frame[
        [
            "feature_time",
            "ret_candle_pct",
            "ret_15m_pct",
            "relative_volume",
            "upper_wick_ratio",
            "body_range_ratio",
            "close_position_in_range",
            "prior_5m_high",
            "prior_15m_high",
            "broke_prior_5m_high_flag",
            "broke_prior_15m_high_flag",
            "close_back_below_prior_5m_high_flag",
            "close_back_below_prior_15m_high_flag",
            "dist_below_prior_5m_high_pct",
            "dist_below_prior_15m_high_pct",
            "push_still_active_flag",
            "local_failure_score",
            "local_failure_flag",
        ]
    ].copy()


def build_timing_error_matches(
    matched_df: pd.DataFrame,
    minute_features: pd.DataFrame,
    five_minute_features: pd.DataFrame,
    variant_id: str,
    variant_label: str,
) -> pd.DataFrame:
    if matched_df.empty:
        return pd.DataFrame()

    frame = matched_df.copy()
    frame["variant_id"] = variant_id
    frame["variant_label"] = variant_label
    frame["effective_entry_time"] = pd.to_datetime(frame["effective_entry_time"], utc=True, errors="coerce")
    frame["matched_model_entry_time"] = pd.to_datetime(frame["matched_model_entry_time"], utc=True, errors="coerce")
    frame["effective_entry_price"] = pd.to_numeric(frame["effective_entry_price"], errors="coerce")
    frame["matched_model_entry_price"] = pd.to_numeric(frame["matched_model_entry_price"], errors="coerce")
    frame["signed_time_delta_minutes"] = pd.to_numeric(frame["matched_signed_time_distance_minutes"], errors="coerce")
    frame["abs_time_distance_minutes"] = pd.to_numeric(frame["matched_time_distance_minutes"], errors="coerce")
    frame["matched_price_distance_pct"] = pd.to_numeric(frame["matched_price_distance_pct"], errors="coerce")
    frame["signed_price_delta_pct"] = (
        (frame["matched_model_entry_price"] - frame["effective_entry_price"]) / frame["effective_entry_price"] * 100.0
    )

    frame = attach_anchor_features(frame, minute_features, "effective_entry_time", "real_1m")
    frame = attach_anchor_features(frame, five_minute_features, "effective_entry_time", "real_5m")
    frame = attach_anchor_features(frame, minute_features, "matched_model_entry_time", "model_1m")
    frame = attach_anchor_features(frame, five_minute_features, "matched_model_entry_time", "model_5m")

    frame["timing_bucket"] = frame["signed_time_delta_minutes"].apply(classify_timing_bucket)
    frame["failure_score_gap_5m"] = pd.to_numeric(frame.get("real_5m_local_failure_score"), errors="coerce") - pd.to_numeric(
        frame.get("model_5m_local_failure_score"), errors="coerce"
    )
    frame["dist_below_prior_15m_high_gap_pct"] = pd.to_numeric(
        frame.get("real_5m_dist_below_prior_15m_high_pct"), errors="coerce"
    ) - pd.to_numeric(frame.get("model_5m_dist_below_prior_15m_high_pct"), errors="coerce")
    frame["ret_candle_5m_gap_pct"] = pd.to_numeric(frame.get("real_5m_ret_candle_pct"), errors="coerce") - pd.to_numeric(
        frame.get("model_5m_ret_candle_pct"), errors="coerce"
    )
    frame["real_after_failure_flag"] = coerce_bool_series(frame.get("real_5m_local_failure_flag", False))
    frame["model_after_failure_flag"] = coerce_bool_series(frame.get("model_5m_local_failure_flag", False))
    frame["model_on_zone_arrival_flag"] = coerce_bool_series(frame.get("model_5m_push_still_active_flag", False)) & ~frame[
        "model_after_failure_flag"
    ]
    frame["late_after_move_started_proxy_flag"] = (
        frame["signed_time_delta_minutes"].gt(SYNC_THRESHOLD_MINUTES)
        & frame["signed_price_delta_pct"].le(LATE_MOVE_STARTED_PRICE_THRESHOLD_PCT)
    )
    frame["mismatch_type"] = frame.apply(classify_timing_mismatch, axis=1)
    frame["candidate_timing_clue"] = frame.apply(infer_candidate_timing_clue, axis=1)
    return frame.sort_values(["variant_id", "effective_entry_time"]).reset_index(drop=True)


def attach_anchor_features(frame: pd.DataFrame, feature_df: pd.DataFrame, time_column: str, prefix: str) -> pd.DataFrame:
    if frame.empty or feature_df.empty or time_column not in frame.columns:
        return frame.copy()

    left = frame.sort_values(time_column).reset_index(drop=True)
    right = feature_df.copy().sort_values("feature_time")
    rename_map = {column: f"{prefix}_{column}" for column in right.columns if column != "feature_time"}
    right = right.rename(columns=rename_map)
    right = right.rename(columns={"feature_time": f"{prefix}_feature_time"})

    merged = pd.merge_asof(
        left,
        right,
        left_on=time_column,
        right_on=f"{prefix}_feature_time",
        direction="backward",
    )
    return merged


def classify_timing_bucket(value: object) -> str:
    delta = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(delta):
        return "unknown"
    if delta <= -30:
        return "very_early"
    if delta < -SYNC_THRESHOLD_MINUTES:
        return "early"
    if delta <= SYNC_THRESHOLD_MINUTES:
        return "near_sync"
    if delta < 30:
        return "late"
    return "very_late"


def classify_timing_mismatch(row: pd.Series) -> str:
    abs_time = safe_float(row.get("abs_time_distance_minutes"))
    signed_time = safe_float(row.get("signed_time_delta_minutes"))
    abs_price = safe_float(row.get("matched_price_distance_pct"))
    signed_price = safe_float(row.get("signed_price_delta_pct"))
    same_move_30 = truthy(row.get("same_short_move_30m_flag"))
    failure_gap = safe_float(row.get("failure_score_gap_5m"))
    real_after_failure = truthy(row.get("real_after_failure_flag"))
    model_on_zone_arrival = truthy(row.get("model_on_zone_arrival_flag"))

    if abs_time <= SYNC_THRESHOLD_MINUTES and abs_price <= SAME_ZONE_PRICE_THRESHOLD_PCT:
        return "same_moment_close_match"
    if abs_price <= SAME_ZONE_PRICE_THRESHOLD_PCT and abs_time > SYNC_THRESHOLD_MINUTES and not same_move_30:
        return "same_zone_wrong_moment"
    if signed_time < -SYNC_THRESHOLD_MINUTES:
        if abs_price <= NEAR_ZONE_PRICE_THRESHOLD_PCT and (failure_gap >= 1.0 or (model_on_zone_arrival and real_after_failure)):
            return "early_before_failure"
        if abs_price <= NEAR_ZONE_PRICE_THRESHOLD_PCT:
            return "early_near_zone"
    if signed_time > SYNC_THRESHOLD_MINUTES:
        if signed_price <= LATE_MOVE_STARTED_PRICE_THRESHOLD_PCT:
            return "late_after_move_started"
        if abs_price <= NEAR_ZONE_PRICE_THRESHOLD_PCT:
            return "late_near_zone"
    if abs_price <= NEAR_ZONE_PRICE_THRESHOLD_PCT:
        return "same_zone_wrong_moment"
    return "far_mismatch"


def infer_candidate_timing_clue(row: pd.Series) -> str:
    mismatch_type = str(row.get("mismatch_type", "far_mismatch"))
    if mismatch_type == "early_before_failure":
        return "model_on_zone_arrival_real_after_local_failure"
    if mismatch_type == "early_near_zone":
        return "proxy_earlier_in_zone_without_clear_failure"
    if mismatch_type == "late_after_move_started":
        return "proxy_after_short_move_already_started"
    if mismatch_type == "late_near_zone":
        return "proxy_later_but_still_in_zone"
    if mismatch_type == "same_zone_wrong_moment":
        return "same_zone_but_micro_timing_gap"
    if mismatch_type == "same_moment_close_match":
        return "close_match"
    return "loose_or_far_mismatch"


def build_timing_error_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = matches_df.groupby(["variant_id", "variant_label", "mismatch_type"], dropna=False)
    for (variant_id, variant_label, mismatch_type), group in grouped:
        variant_total = len(matches_df.loc[matches_df["variant_id"] == variant_id])
        rows.append(
            {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "mismatch_type": mismatch_type,
                "matched_cases": int(len(group)),
                "share_of_variant_matched": safe_share(len(group), variant_total),
                "median_signed_time_delta_minutes": pd.to_numeric(group["signed_time_delta_minutes"], errors="coerce").median(),
                "median_abs_time_distance_minutes": pd.to_numeric(group["abs_time_distance_minutes"], errors="coerce").median(),
                "median_price_distance_pct": pd.to_numeric(group["matched_price_distance_pct"], errors="coerce").median(),
                "median_signed_price_delta_pct": pd.to_numeric(group["signed_price_delta_pct"], errors="coerce").median(),
                "same_short_move_15m_share": coerce_bool_series(group.get("same_short_move_15m_flag", False)).mean(),
                "same_short_move_30m_share": coerce_bool_series(group.get("same_short_move_30m_flag", False)).mean(),
                "same_short_move_60m_share": coerce_bool_series(group.get("same_short_move_60m_flag", False)).mean(),
                "share_real_after_failure": coerce_bool_series(group.get("real_after_failure_flag", False)).mean(),
                "share_model_on_zone_arrival": coerce_bool_series(group.get("model_on_zone_arrival_flag", False)).mean(),
                "median_failure_score_gap_5m": pd.to_numeric(group.get("failure_score_gap_5m"), errors="coerce").median(),
                "median_return_gap_30m_pct": (
                    pd.to_numeric(group.get("model_window_return_30m_pct"), errors="coerce")
                    - pd.to_numeric(group.get("real_window_return_30m_pct"), errors="coerce")
                ).median(),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["mismatch_type"] = pd.Categorical(summary["mismatch_type"], categories=MISMATCH_TYPE_ORDER, ordered=True)
    return summary.sort_values(["variant_label", "mismatch_type"]).reset_index(drop=True)


def build_timing_error_quality(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame()

    rows = []
    for group_kind, group_column in [("timing_bucket", "timing_bucket"), ("mismatch_type", "mismatch_type")]:
        grouped = matches_df.groupby(["variant_id", "variant_label", group_column], dropna=False)
        for (variant_id, variant_label, group_value), group in grouped:
            row = {
                "variant_id": variant_id,
                "variant_label": variant_label,
                "group_kind": group_kind,
                "group_value": group_value,
                "matched_cases": int(len(group)),
                "median_signed_time_delta_minutes": pd.to_numeric(group["signed_time_delta_minutes"], errors="coerce").median(),
                "median_abs_time_distance_minutes": pd.to_numeric(group["abs_time_distance_minutes"], errors="coerce").median(),
                "median_price_distance_pct": pd.to_numeric(group["matched_price_distance_pct"], errors="coerce").median(),
                "median_signed_price_delta_pct": pd.to_numeric(group["signed_price_delta_pct"], errors="coerce").median(),
                "same_short_move_15m_share": coerce_bool_series(group.get("same_short_move_15m_flag", False)).mean(),
                "same_short_move_30m_share": coerce_bool_series(group.get("same_short_move_30m_flag", False)).mean(),
                "same_short_move_60m_share": coerce_bool_series(group.get("same_short_move_60m_flag", False)).mean(),
                "share_real_after_failure": coerce_bool_series(group.get("real_after_failure_flag", False)).mean(),
                "share_model_after_failure": coerce_bool_series(group.get("model_after_failure_flag", False)).mean(),
                "share_model_on_zone_arrival": coerce_bool_series(group.get("model_on_zone_arrival_flag", False)).mean(),
                "median_failure_score_gap_5m": pd.to_numeric(group.get("failure_score_gap_5m"), errors="coerce").median(),
                "median_dist_below_prior_15m_high_gap_pct": pd.to_numeric(
                    group.get("dist_below_prior_15m_high_gap_pct"),
                    errors="coerce",
                ).median(),
            }
            for horizon in [15, 30, 60]:
                real_return = pd.to_numeric(group.get(f"real_window_return_{horizon}m_pct"), errors="coerce")
                model_return = pd.to_numeric(group.get(f"model_window_return_{horizon}m_pct"), errors="coerce")
                real_mae = pd.to_numeric(group.get(f"real_window_mae_{horizon}m_pct"), errors="coerce")
                model_mae = pd.to_numeric(group.get(f"model_window_mae_{horizon}m_pct"), errors="coerce")
                real_mfe = pd.to_numeric(group.get(f"real_window_mfe_{horizon}m_pct"), errors="coerce")
                model_mfe = pd.to_numeric(group.get(f"model_window_mfe_{horizon}m_pct"), errors="coerce")

                row[f"median_real_return_{horizon}m_pct"] = real_return.median()
                row[f"median_model_return_{horizon}m_pct"] = model_return.median()
                row[f"median_abs_gap_return_{horizon}m_pct"] = (model_return - real_return).abs().median()
                row[f"median_real_mae_{horizon}m_pct"] = real_mae.median()
                row[f"median_model_mae_{horizon}m_pct"] = model_mae.median()
                row[f"median_real_mfe_{horizon}m_pct"] = real_mfe.median()
                row[f"median_model_mfe_{horizon}m_pct"] = model_mfe.median()
            rows.append(row)

    quality = pd.DataFrame(rows)
    if quality.empty:
        return quality
    return quality.sort_values(["variant_label", "group_kind", "group_value"]).reset_index(drop=True)


def build_timing_error_section(matches_df: pd.DataFrame, summary_df: pd.DataFrame, quality_df: pd.DataFrame) -> str:
    if matches_df.empty or summary_df.empty or quality_df.empty:
        return (
            "CLUSTER 1 SHORT TIMING ERROR ANATOMY - WHY THE PROXY MISSES THE MOMENT\n"
            "---------------------------------------------------------------------\n"
            "Timing-error diagnosis could not be completed.\n"
        )

    base_matches = matches_df.loc[matches_df["variant_id"] == "base_proxy"].copy()
    base_summary = summary_df.loc[summary_df["variant_id"] == "base_proxy"].copy()
    base_quality = quality_df.loc[quality_df["variant_id"] == "base_proxy"].copy()
    early_frame = subset_by_timing_bucket(base_matches, ["very_early", "early"])
    late_frame = subset_by_timing_bucket(base_matches, ["late", "very_late"])
    near_sync_frame = subset_by_timing_bucket(base_matches, ["near_sync"])

    base_total = len(base_matches)
    early_share = timing_bucket_share(base_matches, ["very_early", "early"])
    late_share = timing_bucket_share(base_matches, ["late", "very_late"])
    near_sync_share = timing_bucket_share(base_matches, ["near_sync"])
    early_quality = select_quality_group(base_quality, "timing_bucket", "early")
    very_early_quality = select_quality_group(base_quality, "timing_bucket", "very_early")
    late_quality = select_quality_group(base_quality, "timing_bucket", "late")
    very_late_quality = select_quality_group(base_quality, "timing_bucket", "very_late")
    near_sync_quality = select_quality_group(base_quality, "timing_bucket", "near_sync")
    dominant_type_row = dominant_summary_row(base_summary)

    early_same_move_30 = coerce_bool_series(early_frame.get("same_short_move_30m_flag", False)).mean()
    late_same_move_30 = coerce_bool_series(late_frame.get("same_short_move_30m_flag", False)).mean()
    near_sync_same_move_30 = coerce_bool_series(near_sync_frame.get("same_short_move_30m_flag", False)).mean()
    early_return_gap_30 = median_abs_gap(early_frame, "model_window_return_30m_pct", "real_window_return_30m_pct")
    late_return_gap_30 = median_abs_gap(late_frame, "model_window_return_30m_pct", "real_window_return_30m_pct")
    early_model_zone_arrival = coerce_bool_series(early_frame.get("model_on_zone_arrival_flag", False)).mean()
    early_real_after_failure = coerce_bool_series(early_frame.get("real_after_failure_flag", False)).mean()

    timing_mode = determine_timing_mode(early_share, late_share, near_sync_share, early_same_move_30, late_same_move_30)
    same_zone_issue = bool(
        not dominant_type_row.empty and str(dominant_type_row.get("mismatch_type")) in {"same_zone_wrong_moment", "early_before_failure", "early_near_zone"}
    )
    structured_direction = determine_next_refinement_direction(base_summary, base_quality, early_share, late_share)
    structured_gap = structured_direction != "not_plausible"
    confidence = determine_timing_confidence(base_total, dominant_type_row, early_same_move_30, late_same_move_30)

    refined_text = ""
    refined_matches = matches_df.loc[matches_df["variant_id"] != "base_proxy"].copy()
    if not refined_matches.empty:
        refined_variant_label = str(refined_matches["variant_label"].iloc[0])
        refined_quality = quality_df.loc[quality_df["variant_id"] != "base_proxy"].copy()
        refined_early_frame = subset_by_timing_bucket(refined_matches, ["very_early", "early"])
        refined_late_frame = subset_by_timing_bucket(refined_matches, ["late", "very_late"])
        refined_early_share = timing_bucket_share(refined_matches, ["very_early", "early"])
        refined_late_share = timing_bucket_share(refined_matches, ["late", "very_late"])
        refined_early_quality = select_quality_group(refined_quality, "timing_bucket", "early")
        refined_very_early_quality = select_quality_group(refined_quality, "timing_bucket", "very_early")
        refined_late_quality = select_quality_group(refined_quality, "timing_bucket", "late")
        refined_very_late_quality = select_quality_group(refined_quality, "timing_bucket", "very_late")
        refined_early_same_move_30 = coerce_bool_series(refined_early_frame.get("same_short_move_30m_flag", False)).mean()
        refined_late_same_move_30 = coerce_bool_series(refined_late_frame.get("same_short_move_30m_flag", False)).mean()
        refined_text = (
            f"\nSecondary comparison with best refined variant ({refined_variant_label}):\n"
            f"- Early / late matched share: {pct(refined_early_share)} / {pct(refined_late_share)}.\n"
            f"- Same-short-move share at 30m for early vs late matched cases: {pct(refined_early_same_move_30)} vs {pct(refined_late_same_move_30)}.\n"
            f"- This suggests the best refined variant {'meaningfully changes' if materially_changes_timing_anatomy(early_share, late_share, early_same_move_30, late_same_move_30, refined_early_share, refined_late_share, refined_early_same_move_30, refined_late_same_move_30) else 'does not materially change'} the base timing-mismatch anatomy.\n"
        )

    report = f"""
CLUSTER 1 SHORT TIMING ERROR ANATOMY - WHY THE PROXY MISSES THE MOMENT
---------------------------------------------------------------------
This stage diagnoses the remaining timing mismatch between the current Cluster 1 short imitation candidate and the trader's real Cluster 1 short trades. It does not claim exact execution recovery. The focus is on matched cases where the proxy lands near the same zone, but may still miss the practical short moment.

Base proxy matched cases:
- Matched cases inside the overlap window: {fmt_int(base_total)}.
- Early / near-synchronous / late matched share: {pct(early_share)} / {pct(near_sync_share)} / {pct(late_share)}.
- Median signed timing delta: {fmt_number(pd.to_numeric(base_matches["signed_time_delta_minutes"], errors="coerce").median())} minutes (negative = proxy earlier).
- Median matched price distance: {fmt_number(pd.to_numeric(base_matches["matched_price_distance_pct"], errors="coerce").median())}%.

Timing vs same-move quality:
- Same-short-move share at 30m for early vs near-synchronous vs late matched cases: {pct(early_same_move_30)} / {pct(near_sync_same_move_30)} / {pct(late_same_move_30)}.
- Median absolute 30m return gap for early vs late matched cases: {fmt_number(early_return_gap_30)}% vs {fmt_number(late_return_gap_30)}%.
- Share of early matched cases where the proxy still looked like zone-arrival while the real trade already showed local failure clues: {pct(safe_share_sum(early_model_zone_arrival, early_real_after_failure))}.

Mismatch taxonomy:
- Dominant matched-case mismatch type: {friendly_mismatch_label(dominant_type_row.get('mismatch_type') if not dominant_type_row.empty else 'far_mismatch')} ({pct(dominant_type_row.get('share_of_variant_matched') if not dominant_type_row.empty else np.nan)} of matched cases).
- Same-zone but wrong-moment issue looks {'material' if same_zone_issue else 'secondary'} in this sample.
- Overall timing mode: {timing_mode}.{refined_text}

Interpretation:
- The proxy tends to fire {describe_timing_mode(timing_mode)}.
- {'There is partial evidence that real trades sometimes appear after stronger local failure / non-continuation than the proxy signal.' if early_real_after_failure >= 0.20 and early_same_move_30 < late_same_move_30 else 'A single missing local-failure cue is not strongly confirmed by this sample.'}
- {'The remaining gap still looks structured enough to justify one more narrow timing test.' if structured_direction in {'wait_for_failure', 'wait_for_retest', 'confirmation_after_weakening', 'delay_after_zone_arrival'} else 'The remaining timing mismatch still looks fairly heterogeneous.'}
- Confidence level: {confidence}.

PRACTICAL INTERPRETATION - IS ONE MORE TIMING LAYER EVEN PLAUSIBLE?
-------------------------------------------------------------------
- {'One final narrow timing refinement still looks plausible.' if structured_gap else 'Reverse engineering now looks close to a practical ceiling rather than one simple missing layer.'}
- {'The most plausible next refinement direction is: ' + structured_direction.replace('_', ' ') + '.' if structured_gap else 'A single next layer does not yet look clearly justified by the timing-error anatomy alone.'}
- {'If one more refinement is attempted, it should be framed as a small wait-for-failure / confirmation-after-weakening test, not as recovered execution logic.' if structured_gap else 'If work continues, it should probably shift away from exact trigger recovery and toward broader imitation validation.'}
""".strip()
    return report + "\n"


def timing_bucket_share(frame: pd.DataFrame, buckets: list[str]) -> float:
    if frame.empty:
        return np.nan
    return frame["timing_bucket"].isin(buckets).mean()


def select_quality_group(quality_df: pd.DataFrame, group_kind: str, group_value: str) -> pd.Series:
    if quality_df.empty:
        return pd.Series(dtype=object)
    match = quality_df.loc[
        (quality_df["group_kind"] == group_kind) & (quality_df["group_value"].astype(str) == str(group_value))
    ]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def dominant_summary_row(summary_df: pd.DataFrame) -> pd.Series:
    if summary_df.empty:
        return pd.Series(dtype=object)
    return summary_df.sort_values(["share_of_variant_matched", "same_short_move_30m_share"], ascending=[False, False]).iloc[0]


def combine_quality_metric(rows: list[pd.Series], column: str) -> float:
    values = [safe_float(row.get(column)) for row in rows if row is not None and not row.empty]
    values = [value for value in values if not pd.isna(value)]
    if not values:
        return np.nan
    return float(np.nanmean(values))


def subset_by_timing_bucket(frame: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[frame["timing_bucket"].isin(buckets)].copy()


def median_abs_gap(frame: pd.DataFrame, left_column: str, right_column: str) -> float:
    if frame.empty:
        return np.nan
    left = pd.to_numeric(frame.get(left_column), errors="coerce")
    right = pd.to_numeric(frame.get(right_column), errors="coerce")
    return (left - right).abs().median()


def determine_timing_mode(
    early_share: float,
    late_share: float,
    near_sync_share: float,
    early_same_move_30: float,
    late_same_move_30: float,
) -> str:
    if early_share >= late_share + 0.15:
        return "mostly_too_early"
    if late_share >= early_share + 0.15:
        return "mostly_too_late"
    if safe_float(early_same_move_30) + 0.20 < safe_float(late_same_move_30):
        return "mixed_but_harmfully_early"
    if near_sync_share >= max(early_share, late_share):
        return "often_near_synchronous"
    return "mixed"


def describe_timing_mode(timing_mode: str) -> str:
    mapping = {
        "mostly_too_early": "too early relative to the trader",
        "mostly_too_late": "too late relative to the trader",
        "mixed_but_harmfully_early": "in mixed ways overall, but the damaging misses are more often early",
        "often_near_synchronous": "surprisingly close in time, with the remaining gap concentrated in local micro-structure",
        "mixed": "in mixed ways rather than from one clean timing mistake",
    }
    return mapping.get(timing_mode, "in mixed ways")


def determine_next_refinement_direction(
    summary_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    early_share: float,
    late_share: float,
) -> str:
    dominant = dominant_summary_row(summary_df)
    dominant_type = str(dominant.get("mismatch_type", "far_mismatch"))
    early_quality = select_quality_group(quality_df, "timing_bucket", "early")
    very_early_quality = select_quality_group(quality_df, "timing_bucket", "very_early")
    early_zone_arrival = combine_quality_metric([early_quality, very_early_quality], "share_model_on_zone_arrival")
    early_real_failure = combine_quality_metric([early_quality, very_early_quality], "share_real_after_failure")

    if dominant_type == "early_before_failure" or (early_share >= 0.35 and early_zone_arrival > early_real_failure):
        return "wait_for_failure"
    if dominant_type == "same_zone_wrong_moment":
        return "confirmation_after_weakening"
    if dominant_type == "early_near_zone":
        return "wait_for_retest"
    if dominant_type == "late_after_move_started" or late_share >= 0.45:
        return "delay_after_zone_arrival"
    return "not_plausible"


def determine_timing_confidence(
    matched_cases: int,
    dominant_type_row: pd.Series,
    early_same_move_30: float,
    late_same_move_30: float,
) -> str:
    dominant_share = safe_float(dominant_type_row.get("share_of_variant_matched")) if not dominant_type_row.empty else np.nan
    if matched_cases >= 20 and dominant_share >= 0.25 and safe_float(late_same_move_30) - safe_float(early_same_move_30) >= 0.20:
        return "medium"
    return "low"


def materially_changes_timing_anatomy(
    base_early_share: float,
    base_late_share: float,
    base_early_same_move_30: float,
    base_late_same_move_30: float,
    refined_early_share: float,
    refined_late_share: float,
    refined_early_same_move_30: float,
    refined_late_same_move_30: float,
) -> bool:
    return bool(
        abs(safe_float(refined_early_share) - safe_float(base_early_share)) >= 0.15
        or abs(safe_float(refined_late_share) - safe_float(base_late_share)) >= 0.15
        or abs(safe_float(refined_early_same_move_30) - safe_float(base_early_same_move_30)) >= 0.20
        or abs(safe_float(refined_late_same_move_30) - safe_float(base_late_same_move_30)) >= 0.20
    )


def friendly_mismatch_label(value: object) -> str:
    mapping = {
        "same_moment_close_match": "same-moment close match",
        "same_zone_wrong_moment": "same-zone but wrong-moment mismatch",
        "early_before_failure": "early before local failure",
        "early_near_zone": "early near-zone miss",
        "late_after_move_started": "late after move started",
        "late_near_zone": "late near-zone miss",
        "far_mismatch": "far or loose mismatch",
    }
    return mapping.get(str(value), str(value))


def safe_share(numerator: float | int, denominator: float | int) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def safe_share_sum(first: float, second: float) -> float:
    if pd.isna(first) or pd.isna(second):
        return np.nan
    return min(float(first), float(second))


def safe_float(value: object) -> float:
    if value is None:
        return np.nan
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if pd.isna(value):
        return False
    return bool(value)


def coerce_bool_series(series: object) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.map(truthy)
    return pd.Series([truthy(series)])


def create_cluster1_short_timing_error_plots(
    matches_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if matches_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    delta_path = plots_dir / "eth_cluster1_short_timing_delta_distribution.png"
    mismatch_path = plots_dir / "eth_cluster1_short_timing_mismatch_types.png"
    same_move_path = plots_dir / "eth_cluster1_short_timing_vs_same_move.png"

    fig_delta = plot_timing_delta_distribution(matches_df)
    fig_delta.savefig(delta_path, dpi=150, bbox_inches="tight")
    plt.close(fig_delta)

    fig_mismatch = plot_timing_mismatch_types(matches_df)
    fig_mismatch.savefig(mismatch_path, dpi=150, bbox_inches="tight")
    plt.close(fig_mismatch)

    fig_same_move = plot_timing_vs_same_move(matches_df)
    fig_same_move.savefig(same_move_path, dpi=150, bbox_inches="tight")
    plt.close(fig_same_move)

    logger.info("Saved Cluster 1 short timing-error anatomy plots")


def plot_timing_delta_distribution(matches_df: pd.DataFrame) -> plt.Figure:
    bins = [-60, -30, -15, -5, 5, 15, 30, 60]
    labels = ["-60:-30", "-30:-15", "-15:-5", "-5:+5", "+5:+15", "+15:+30", "+30:+60"]
    plot_df = matches_df.copy()
    plot_df["timing_bin"] = pd.cut(plot_df["signed_time_delta_minutes"], bins=bins, labels=labels, include_lowest=True)
    share_df = (
        plot_df.groupby(["variant_label", "timing_bin"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = share_df.groupby("variant_label")["count"].transform("sum")
    share_df["share"] = share_df["count"] / totals.replace(0, np.nan)

    variants = list(dict.fromkeys(share_df["variant_label"].tolist()))
    x = np.arange(len(labels))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, variant in enumerate(variants):
        subset = share_df.loc[share_df["variant_label"] == variant]
        values = [
            safe_float(subset.loc[subset["timing_bin"] == label, "share"].iloc[0])
            if not subset.loc[subset["timing_bin"] == label].empty
            else 0.0
            for label in labels
        ]
        ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            values,
            width=width,
            label=variant,
            color=palette[idx % len(palette)],
            alpha=0.88,
        )

    ax.set_title("Cluster 1 short matched timing delta distribution")
    ax.set_ylabel("Share of matched cases")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig


def plot_timing_mismatch_types(matches_df: pd.DataFrame) -> plt.Figure:
    plot_df = (
        matches_df.groupby(["variant_label", "mismatch_type"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = plot_df.groupby("variant_label")["count"].transform("sum")
    plot_df["share"] = plot_df["count"] / totals.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    variants = list(dict.fromkeys(plot_df["variant_label"].tolist()))
    bottom = np.zeros(len(variants))
    palette = {
        "same_moment_close_match": "#54a24b",
        "same_zone_wrong_moment": "#4c78a8",
        "early_before_failure": "#e45756",
        "early_near_zone": "#f58518",
        "late_after_move_started": "#b279a2",
        "late_near_zone": "#72b7b2",
        "far_mismatch": "#9d755d",
    }

    for mismatch_type in MISMATCH_TYPE_ORDER:
        shares = []
        for variant in variants:
            match = plot_df.loc[(plot_df["variant_label"] == variant) & (plot_df["mismatch_type"] == mismatch_type), "share"]
            shares.append(float(match.iloc[0]) if not match.empty else 0.0)
        ax.bar(
            variants,
            shares,
            bottom=bottom,
            label=friendly_mismatch_label(mismatch_type),
            color=palette.get(mismatch_type, "#808080"),
            alpha=0.9,
        )
        bottom += np.array(shares)

    ax.set_title("Cluster 1 short mismatch taxonomy")
    ax.set_ylabel("Share of matched cases")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_timing_vs_same_move(matches_df: pd.DataFrame) -> plt.Figure:
    buckets = ["very_early", "early", "near_sync", "late", "very_late"]
    plot_df = (
        matches_df.groupby(["variant_label", "timing_bucket"], observed=False)["same_short_move_30m_flag"]
        .mean()
        .reset_index()
    )
    variants = list(dict.fromkeys(plot_df["variant_label"].tolist()))
    x = np.arange(len(buckets))
    width = 0.8 / max(len(variants), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, variant in enumerate(variants):
        subset = plot_df.loc[plot_df["variant_label"] == variant]
        values = [
            safe_float(subset.loc[subset["timing_bucket"] == bucket, "same_short_move_30m_flag"].iloc[0])
            if not subset.loc[subset["timing_bucket"] == bucket].empty
            else 0.0
            for bucket in buckets
        ]
        ax.bar(
            x + (idx - (len(variants) - 1) / 2) * width,
            values,
            width=width,
            label=variant,
            color=palette[idx % len(palette)],
            alpha=0.88,
        )

    ax.set_title("Cluster 1 short same-move capture by timing bucket")
    ax.set_ylabel("Same-short-move share at 30m")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig
