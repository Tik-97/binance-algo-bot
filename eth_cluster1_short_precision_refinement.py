from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster1_short_overlap_validation import (
    TIME_BANDS_MINUTES,
    build_overlap_matches,
    build_overlap_quality,
    build_overlap_summary,
    load_proxy_market_frames,
    price_band_key,
)
from eth_replication_hypothesis_test import ETH_SYMBOL
from market_context_at_entry import prepare_context_features
from utils import fmt_int, fmt_number, pct


REFINEMENT_VARIANTS = [
    {
        "variant_id": "base_proxy",
        "variant_label": "Base proxy",
        "family": "base",
        "definition_text": "Current practical Cluster 1 short proxy without extra narrowing.",
    },
    {
        "variant_id": "strict_high_zone",
        "variant_label": "Strict 4h high proximity",
        "family": "price_location_tightening",
        "definition_text": "Distance to 4h high <= 0.35%.",
    },
    {
        "variant_id": "stronger_bounce",
        "variant_label": "Stronger 60m bounce context",
        "family": "context_tightening",
        "definition_text": "Pre-entry 60m return >= 0.60%.",
    },
    {
        "variant_id": "stronger_upper_wick_5m",
        "variant_label": "Stronger 5m upper wick",
        "family": "micro_rejection",
        "definition_text": "Upper wick ratio 5m >= 0.40.",
    },
    {
        "variant_id": "bearish_close_5m",
        "variant_label": "Stricter 5m non-continuation",
        "family": "timing_refinement",
        "definition_text": "Last 5m return <= -0.05%.",
    },
    {
        "variant_id": "micro_rejection_5m",
        "variant_label": "5m weak-close rejection",
        "family": "micro_rejection",
        "definition_text": "Close back in lower half of 5m candle and 5m body/range <= 0.35.",
    },
    {
        "variant_id": "hv1m_small_body",
        "variant_label": "1m effort-vs-result clue",
        "family": "effort_vs_result",
        "definition_text": "Relative volume 1m >= 1.88 and 1m body <= 0.06% of price.",
    },
    {
        "variant_id": "strict_zone_plus_bearish_close",
        "variant_label": "Strict zone + bearish 5m close",
        "family": "two_filter_combo",
        "definition_text": "Distance to 4h high <= 0.35% and last 5m return <= -0.05%.",
    },
    {
        "variant_id": "strict_zone_plus_micro_rejection",
        "variant_label": "Strict zone + 5m weak-close rejection",
        "family": "two_filter_combo",
        "definition_text": "Distance to 4h high <= 0.35% and close back in lower half with small 5m body.",
    },
    {
        "variant_id": "strict_zone_plus_hv1m_small_body",
        "variant_label": "Strict zone + 1m effort-result clue",
        "family": "two_filter_combo",
        "definition_text": "Distance to 4h high <= 0.35% and relative volume 1m >= 1.88 with 1m body <= 0.06%.",
    },
]


def run_cluster1_short_precision_refinement_analysis(
    trades_df: pd.DataFrame,
    real_trades_df: pd.DataFrame,
    model_signals_df: pd.DataFrame,
    broad_cluster1_signals_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "CLUSTER 1 SHORT PRECISION REFINEMENT - REDUCE SIGNAL DENSITY, IMPROVE OVERLAP PRECISION, PRESERVE REAL-TRADE COVERAGE\n"
        "---------------------------------------------------------------------------------------------------------------\n"
        "No sufficient Cluster 1 short overlap data was available for precision refinement.\n"
    )

    if real_trades_df.empty or model_signals_df.empty:
        return empty_result(empty, empty_section)

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    minute_frame, frame_5m, _stages = load_proxy_market_frames(eth_trades, logger)
    if minute_frame.empty or frame_5m.empty:
        section = (
            "CLUSTER 1 SHORT PRECISION REFINEMENT - REDUCE SIGNAL DENSITY, IMPROVE OVERLAP PRECISION, PRESERVE REAL-TRADE COVERAGE\n"
            "---------------------------------------------------------------------------------------------------------------\n"
            "Precision refinement could not run because ETH candle data was unavailable.\n"
        )
        return empty_result(empty, section)

    enriched_model_signals = enrich_model_signals_with_refinement_features(model_signals_df, minute_frame, frame_5m)
    broad_stage_a_count = extract_broad_stage_a_count(broad_cluster1_signals_df)

    variants_df, results_df, matches_by_variant = evaluate_refinement_variants(
        real_trades_df,
        enriched_model_signals,
        broad_stage_a_count,
    )
    if results_df.empty:
        return empty_result(empty, empty_section)

    best_variant_row = select_best_variant(results_df)
    best_variant_id = str(best_variant_row["variant_id"])
    best_matches = matches_by_variant.get(best_variant_id, pd.DataFrame()).copy()
    tradeoffs = build_tradeoff_table(results_df)
    section_text = build_refinement_section(results_df, best_variant_row, tradeoffs)

    return {
        "variants": variants_df,
        "results": results_df,
        "best_matches": best_matches,
        "tradeoffs": tradeoffs,
        "section_text": section_text,
    }


def empty_result(empty: pd.DataFrame, section_text: str) -> dict:
    return {
        "variants": empty,
        "results": empty,
        "best_matches": empty,
        "tradeoffs": empty,
        "section_text": section_text,
    }


def enrich_model_signals_with_refinement_features(model_signals_df: pd.DataFrame, minute_frame: pd.DataFrame, frame_5m: pd.DataFrame) -> pd.DataFrame:
    model = model_signals_df.copy()
    model["entry_time"] = pd.to_datetime(model["entry_time"], utc=True, errors="coerce")
    minute_features = build_minute_refinement_features(minute_frame)
    five_minute_features = build_five_minute_refinement_features(frame_5m)

    if not minute_features.empty:
        model = pd.merge_asof(
            model.sort_values("entry_time"),
            minute_features.sort_values("feature_time"),
            left_on="entry_time",
            right_on="feature_time",
            direction="backward",
        )
    if not five_minute_features.empty:
        model = model.merge(
            five_minute_features,
            on="entry_time",
            how="left",
            suffixes=("", "_five"),
        )

    model["flag_strict_high_zone"] = pd.to_numeric(model.get("distance_to_last_4h_high_pct"), errors="coerce") <= 0.35
    model["flag_stronger_bounce"] = pd.to_numeric(model.get("return_pre_60m_pct"), errors="coerce") >= 0.60
    model["flag_stronger_upper_wick_5m"] = pd.to_numeric(model.get("upper_wick_ratio_5m"), errors="coerce") >= 0.40
    model["flag_bearish_close_5m"] = pd.to_numeric(model.get("ret_last_5m_pct"), errors="coerce") <= -0.05
    model["flag_micro_rejection_5m"] = (
        (pd.to_numeric(model.get("close_position_in_range_5m"), errors="coerce") <= 0.50)
        & (pd.to_numeric(model.get("body_range_ratio_5m"), errors="coerce") <= 0.35)
    )
    model["flag_hv1m_small_body"] = (
        (pd.to_numeric(model.get("relative_volume_1m_prior20"), errors="coerce") >= 1.88)
        & (pd.to_numeric(model.get("body_pct_1m"), errors="coerce") <= 0.06)
    )
    return model.sort_values("entry_time").reset_index(drop=True)


def build_minute_refinement_features(minute_frame: pd.DataFrame) -> pd.DataFrame:
    if minute_frame.empty:
        return pd.DataFrame()

    frame = minute_frame[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    price_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["avg_volume_20_prior"] = frame["volume"].shift(1).rolling(20, min_periods=5).mean()
    frame["relative_volume_1m_prior20"] = frame["volume"] / frame["avg_volume_20_prior"]
    frame["body_pct_1m"] = (frame["close"] - frame["open"]).abs() / frame["open"] * 100.0
    frame["body_range_ratio_1m"] = (frame["close"] - frame["open"]).abs() / price_range
    frame["close_position_in_range_1m"] = (frame["close"] - frame["low"]) / price_range
    frame["upper_wick_ratio_1m"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / price_range
    frame["feature_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")

    return frame[
        [
            "feature_time",
            "relative_volume_1m_prior20",
            "body_pct_1m",
            "body_range_ratio_1m",
            "close_position_in_range_1m",
            "upper_wick_ratio_1m",
        ]
    ].copy()


def build_five_minute_refinement_features(frame_5m: pd.DataFrame) -> pd.DataFrame:
    if frame_5m.empty:
        return pd.DataFrame()

    prepared = prepare_context_features(frame_5m, "5m").copy().sort_values("close_time")
    for column in ["open", "high", "low", "close"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    price_range = (prepared["high"] - prepared["low"]).replace(0, np.nan)
    prepared["body_range_ratio_5m"] = (prepared["close"] - prepared["open"]).abs() / price_range
    prepared["close_position_in_range_5m"] = (prepared["close"] - prepared["low"]) / price_range
    prepared["entry_time"] = pd.to_datetime(prepared["close_time"], utc=True, errors="coerce")
    return prepared[["entry_time", "body_range_ratio_5m", "close_position_in_range_5m"]].copy()


def evaluate_refinement_variants(
    real_trades_df: pd.DataFrame,
    enriched_model_signals: pd.DataFrame,
    broad_stage_a_count: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    variants_df = pd.DataFrame(REFINEMENT_VARIANTS)
    matches_by_variant: dict[str, pd.DataFrame] = {}
    rows = []

    base_overlap_summary = None
    for variant in REFINEMENT_VARIANTS:
        filtered_signals = apply_variant_filter(enriched_model_signals, variant["variant_id"])
        overlap_matches = build_overlap_matches(real_trades_df, filtered_signals)
        overlap_summary = build_overlap_summary(real_trades_df, filtered_signals, overlap_matches, broad_stage_a_count)
        overlap_quality = build_overlap_quality(real_trades_df, filtered_signals, overlap_matches)

        if variant["variant_id"] == "base_proxy" and not overlap_summary.empty:
            base_overlap_summary = overlap_summary.iloc[0]

        rows.append(
            build_variant_result_row(
                variant,
                filtered_signals,
                overlap_summary,
                overlap_quality,
                base_overlap_summary,
            )
        )
        matches_by_variant[variant["variant_id"]] = overlap_matches.copy()

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return variants_df, results_df, matches_by_variant

    results_df = finalize_variant_scores(results_df)
    variants_df = variants_df.merge(
        results_df[["variant_id", "signal_count", "balanced_imitation_score", "is_best_variant"]],
        on="variant_id",
        how="left",
    )
    return variants_df, results_df.sort_values("balanced_imitation_score", ascending=False).reset_index(drop=True), matches_by_variant


def apply_variant_filter(model_signals_df: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    frame = model_signals_df.copy()
    if variant_id == "base_proxy":
        mask = pd.Series(True, index=frame.index)
    elif variant_id == "strict_high_zone":
        mask = frame["flag_strict_high_zone"]
    elif variant_id == "stronger_bounce":
        mask = frame["flag_stronger_bounce"]
    elif variant_id == "stronger_upper_wick_5m":
        mask = frame["flag_stronger_upper_wick_5m"]
    elif variant_id == "bearish_close_5m":
        mask = frame["flag_bearish_close_5m"]
    elif variant_id == "micro_rejection_5m":
        mask = frame["flag_micro_rejection_5m"]
    elif variant_id == "hv1m_small_body":
        mask = frame["flag_hv1m_small_body"]
    elif variant_id == "strict_zone_plus_bearish_close":
        mask = frame["flag_strict_high_zone"] & frame["flag_bearish_close_5m"]
    elif variant_id == "strict_zone_plus_micro_rejection":
        mask = frame["flag_strict_high_zone"] & frame["flag_micro_rejection_5m"]
    elif variant_id == "strict_zone_plus_hv1m_small_body":
        mask = frame["flag_strict_high_zone"] & frame["flag_hv1m_small_body"]
    else:
        mask = pd.Series(True, index=frame.index)
    return frame.loc[mask.fillna(False)].copy().sort_values("entry_time").reset_index(drop=True)


def build_variant_result_row(
    variant: dict,
    filtered_signals: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    overlap_quality: pd.DataFrame,
    base_overlap_summary: pd.Series | None,
) -> dict:
    summary = overlap_summary.iloc[0] if not overlap_summary.empty else pd.Series(dtype=object)
    quality_pairs = select_quality_row(overlap_quality, "matched_pair_gaps")
    matched_model = select_quality_row(overlap_quality, "matched_model_signals")

    signal_count = int(len(filtered_signals))
    base_signal_count = safe_number(base_overlap_summary.get("model_signal_count")) if base_overlap_summary is not None else np.nan

    row = {
        "variant_id": variant["variant_id"],
        "variant_label": variant["variant_label"],
        "family": variant["family"],
        "definition_text": variant["definition_text"],
        "signal_count": signal_count,
        "signals_per_real_trade": summary.get("proxy_signals_per_real_trade"),
        "reduction_vs_base": 1.0 - safe_divide(signal_count, base_signal_count),
        "recall_within_15m": summary.get("share_within_15m"),
        "recall_within_30m": summary.get("share_within_30m"),
        "recall_within_60m": summary.get("share_within_60m"),
        "share_within_15m_and_0p25pct": summary.get("share_within_15m_and_0p25pct"),
        "share_within_30m_and_0p50pct": summary.get("share_within_30m_and_0p50pct"),
        "share_within_60m_and_1p00pct": summary.get("share_within_60m_and_1p00pct"),
        "median_time_distance_minutes": summary.get("median_matched_time_distance_minutes"),
        "median_price_distance_pct": summary.get("median_matched_price_distance_pct"),
        "same_short_move_30m_share": summary.get("same_short_move_30m_share_matched"),
        "same_short_move_60m_share": summary.get("same_short_move_60m_share_matched"),
        "matched_unique_share": summary.get("matched_unique_share"),
        "tight_match_share": summary.get("tight_match_share"),
        "workable_or_better_match_share": summary.get("workable_or_better_match_share"),
        "matched_model_median_return_30m_pct": matched_model.get("median_return_30m_pct"),
        "matched_model_median_return_60m_pct": matched_model.get("median_return_60m_pct"),
        "matched_model_median_mae_30m_pct": matched_model.get("median_mae_30m_pct"),
        "matched_model_median_mae_60m_pct": matched_model.get("median_mae_60m_pct"),
        "matched_model_median_mfe_30m_pct": matched_model.get("median_mfe_30m_pct"),
        "matched_model_median_mfe_60m_pct": matched_model.get("median_mfe_60m_pct"),
        "median_abs_gap_return_30m_pct": quality_pairs.get("median_abs_gap_return_30m_pct"),
        "median_abs_gap_return_60m_pct": quality_pairs.get("median_abs_gap_return_60m_pct"),
        "median_abs_gap_mae_30m_pct": quality_pairs.get("median_abs_gap_mae_30m_pct"),
        "median_abs_gap_mfe_30m_pct": quality_pairs.get("median_abs_gap_mfe_30m_pct"),
    }
    return row


def finalize_variant_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    base_row = results_df.loc[results_df["variant_id"] == "base_proxy"].iloc[0]
    base_recall_30 = safe_number(base_row.get("recall_within_30m"))
    base_recall_60 = safe_number(base_row.get("recall_within_60m"))
    base_precision_30 = safe_number(base_row.get("share_within_30m_and_0p50pct"))
    base_same_move_30 = safe_number(base_row.get("same_short_move_30m_share"))
    base_time_distance = safe_number(base_row.get("median_time_distance_minutes"))
    base_price_distance = safe_number(base_row.get("median_price_distance_pct"))

    rows = []
    for _, row in results_df.iterrows():
        result = row.to_dict()
        recall_30_retention = safe_divide(safe_number(row.get("recall_within_30m")), base_recall_30)
        recall_60_retention = safe_divide(safe_number(row.get("recall_within_60m")), base_recall_60)
        precision_30 = safe_number(row.get("share_within_30m_and_0p50pct"))
        same_move_30 = safe_number(row.get("same_short_move_30m_share"))
        density_reduction = max(safe_number(row.get("reduction_vs_base")), 0.0)
        time_tightness = 1.0 - min(safe_number(row.get("median_time_distance_minutes")) / max(base_time_distance, 1.0), 1.5)
        price_tightness = 1.0 - min(safe_number(row.get("median_price_distance_pct")) / max(base_price_distance, 1e-6), 1.5)

        time_tightness = float(np.clip(time_tightness, 0.0, 1.0))
        price_tightness = float(np.clip(price_tightness, 0.0, 1.0))
        recall_30_retention = float(np.clip(recall_30_retention if pd.notna(recall_30_retention) else 0.0, 0.0, 1.2))
        recall_60_retention = float(np.clip(recall_60_retention if pd.notna(recall_60_retention) else 0.0, 0.0, 1.2))

        score = (
            0.25 * recall_30_retention
            + 0.15 * recall_60_retention
            + 0.20 * precision_30
            + 0.15 * same_move_30
            + 0.10 * density_reduction
            + 0.10 * time_tightness
            + 0.05 * price_tightness
        )

        result["recall_30_retention_vs_base"] = recall_30_retention
        result["recall_60_retention_vs_base"] = recall_60_retention
        result["precision_30m_0p50_improvement_vs_base"] = precision_30 - base_precision_30
        result["same_move_30m_improvement_vs_base"] = same_move_30 - base_same_move_30
        result["time_distance_improvement_minutes_vs_base"] = base_time_distance - safe_number(row.get("median_time_distance_minutes"))
        result["price_distance_improvement_pct_vs_base"] = base_price_distance - safe_number(row.get("median_price_distance_pct"))
        result["balanced_imitation_score"] = score
        rows.append(result)

    scored = pd.DataFrame(rows)
    best_idx = scored.sort_values(
        ["balanced_imitation_score", "share_within_30m_and_0p50pct", "recall_within_30m"],
        ascending=[False, False, False],
    ).index[0]
    scored["is_best_variant"] = False
    scored.loc[best_idx, "is_best_variant"] = True
    return scored


def select_quality_row(df: pd.DataFrame, group_name: str) -> pd.Series:
    match = df.loc[df["group_name"] == group_name]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def select_best_variant(results_df: pd.DataFrame) -> pd.Series:
    best = results_df.loc[results_df["is_best_variant"] == True]
    if best.empty:
        return results_df.sort_values("balanced_imitation_score", ascending=False).iloc[0]
    return best.iloc[0]


def build_tradeoff_table(results_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "variant_id",
        "variant_label",
        "family",
        "signal_count",
        "reduction_vs_base",
        "recall_within_15m",
        "recall_within_30m",
        "recall_within_60m",
        "share_within_15m_and_0p25pct",
        "share_within_30m_and_0p50pct",
        "median_time_distance_minutes",
        "median_price_distance_pct",
        "same_short_move_30m_share",
        "same_short_move_60m_share",
        "balanced_imitation_score",
        "is_best_variant",
    ]
    existing = [column for column in keep_columns if column in results_df.columns]
    return results_df[existing].sort_values("balanced_imitation_score", ascending=False).reset_index(drop=True)


def extract_broad_stage_a_count(broad_cluster1_signals_df: pd.DataFrame) -> float:
    if broad_cluster1_signals_df.empty:
        return np.nan
    if "in_stage_a" not in broad_cluster1_signals_df.columns:
        return float(len(broad_cluster1_signals_df))
    return float(broad_cluster1_signals_df["in_stage_a"].fillna(False).sum())


def build_refinement_section(results_df: pd.DataFrame, best_variant_row: pd.Series, tradeoffs_df: pd.DataFrame) -> str:
    base_row = results_df.loc[results_df["variant_id"] == "base_proxy"].iloc[0]
    best_is_base = str(best_variant_row["variant_id"]) == "base_proxy"

    material_improvement = bool(
        safe_number(best_variant_row.get("reduction_vs_base")) >= 0.25
        and safe_number(best_variant_row.get("recall_60_retention_vs_base")) >= 0.70
        and (
            safe_number(best_variant_row.get("precision_30m_0p50_improvement_vs_base")) >= 0.05
            or safe_number(best_variant_row.get("same_move_30m_improvement_vs_base")) >= 0.10
            or safe_number(best_variant_row.get("time_distance_improvement_minutes_vs_base")) >= 3.0
        )
        and not best_is_base
    )

    limiting_factor = determine_limiting_factor(best_variant_row)
    confidence = determine_refinement_confidence(best_variant_row, material_improvement)
    best_family = str(best_variant_row.get("family", "n/a"))

    report = f"""
CLUSTER 1 SHORT PRECISION REFINEMENT - REDUCE SIGNAL DENSITY, IMPROVE OVERLAP PRECISION, PRESERVE REAL-TRADE COVERAGE
---------------------------------------------------------------------------------------------------------------
This stage narrows the current Cluster 1 short imitation candidate with a small set of interpretable refinement variants. The goal is not to recover exact execution, but to reduce signal density while preserving meaningful overlap with the trader's real Cluster 1 short trades.

Base proxy:
- Signals: {fmt_int(base_row.get('signal_count'))}.
- Recall within 15m / 30m / 60m: {pct(base_row.get('recall_within_15m'))} / {pct(base_row.get('recall_within_30m'))} / {pct(base_row.get('recall_within_60m'))}.
- Median matched time distance: {fmt_number(base_row.get('median_time_distance_minutes'))} minutes.
- Median matched price distance: {fmt_number(base_row.get('median_price_distance_pct'))}%.
- Same-short-move share at 30m / 60m: {pct(base_row.get('same_short_move_30m_share'))} / {pct(base_row.get('same_short_move_60m_share'))}.

Best refinement variant:
- Variant: {best_variant_row.get('variant_label')}.
- Family: {best_family}.
- Definition: {best_variant_row.get('definition_text')}.
- Signals: {fmt_int(best_variant_row.get('signal_count'))} ({pct(best_variant_row.get('reduction_vs_base'))} reduction vs base).
- Recall within 15m / 30m / 60m: {pct(best_variant_row.get('recall_within_15m'))} / {pct(best_variant_row.get('recall_within_30m'))} / {pct(best_variant_row.get('recall_within_60m'))}.
- Time + price overlap within 30m and 0.50%: {pct(best_variant_row.get('share_within_30m_and_0p50pct'))}.
- Median matched time distance: {fmt_number(best_variant_row.get('median_time_distance_minutes'))} minutes.
- Median matched price distance: {fmt_number(best_variant_row.get('median_price_distance_pct'))}%.
- Same-short-move share at 30m / 60m: {pct(best_variant_row.get('same_short_move_30m_share'))} / {pct(best_variant_row.get('same_short_move_60m_share'))}.
- Balanced imitation score: {fmt_number(best_variant_row.get('balanced_imitation_score'))}.

Trade-off read:
- Precision improvement vs base at 30m + 0.50%: {fmt_number(best_variant_row.get('precision_30m_0p50_improvement_vs_base'))}.
- Same-short-move improvement vs base at 30m: {fmt_number(best_variant_row.get('same_move_30m_improvement_vs_base'))}.
- Time-distance improvement vs base: {fmt_number(best_variant_row.get('time_distance_improvement_minutes_vs_base'))} minutes.
- Recall retention vs base at 30m / 60m: {fmt_number(best_variant_row.get('recall_30_retention_vs_base'))} / {fmt_number(best_variant_row.get('recall_60_retention_vs_base'))}.

Interpretation:
- {'A refinement variant materially improved practical imitation precision.' if material_improvement else 'No refinement variant produced a clean material precision jump without meaningful trade-offs.'}
- The refinement family that helped most in this sample was: {best_family}.
- The main remaining limiting factor is: {limiting_factor}.
- Confidence level: {confidence}.

PRACTICAL INTERPRETATION - IS CLUSTER 1 SHORT NOW TIGHT ENOUGH FOR IMITATION WORK?
-----------------------------------------------------------------------------------
- {'The refined Cluster 1 short proxy now looks like a tighter imitation candidate, although it still remains partial replication.' if material_improvement else 'The refined Cluster 1 short proxy is still only a partial imitation candidate, not yet a tight practical replication base.'}
- {'Paper testing is now more justified than before, but only as a cautious imitation exercise.' if material_improvement else 'One more refinement or validation step is still preferable before paper testing.'}
- The biggest remaining weakness is {limiting_factor}.
""".strip()
    return report + "\n"


def determine_limiting_factor(best_variant_row: pd.Series) -> str:
    if safe_number(best_variant_row.get("recall_60_retention_vs_base")) < 0.65:
        return "remaining signal overbreadth is improved, but recall retention starts to fall too quickly"
    if safe_number(best_variant_row.get("median_time_distance_minutes")) > 10:
        return "timing precision is still the biggest weakness"
    if safe_number(best_variant_row.get("median_price_distance_pct")) > 0.25:
        return "zone precision is still the biggest weakness"
    if safe_number(best_variant_row.get("same_short_move_30m_share")) < 0.50:
        return "local move selection still does not line up tightly enough"
    return "execution ambiguity still limits how far overlap precision can be trusted"


def determine_refinement_confidence(best_variant_row: pd.Series, material_improvement: bool) -> str:
    signal_count = safe_number(best_variant_row.get("signal_count"))
    recall_30 = safe_number(best_variant_row.get("recall_within_30m"))
    if material_improvement and signal_count >= 20 and recall_30 >= 0.35:
        return "medium"
    return "low"


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def safe_number(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def create_cluster1_short_precision_refinement_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if results_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    signal_density_path = plots_dir / "eth_cluster1_short_refinement_signal_density.png"
    overlap_precision_path = plots_dir / "eth_cluster1_short_refinement_overlap_precision.png"
    frontier_path = plots_dir / "eth_cluster1_short_refinement_tradeoff_frontier.png"

    fig_density = plot_refinement_signal_density(results_df)
    fig_density.savefig(signal_density_path, dpi=150, bbox_inches="tight")
    plt.close(fig_density)

    fig_precision = plot_refinement_overlap_precision(results_df)
    fig_precision.savefig(overlap_precision_path, dpi=150, bbox_inches="tight")
    plt.close(fig_precision)

    fig_frontier = plot_refinement_tradeoff_frontier(results_df)
    fig_frontier.savefig(frontier_path, dpi=150, bbox_inches="tight")
    plt.close(fig_frontier)

    logger.info("Saved Cluster 1 short precision refinement plots")


def plot_refinement_signal_density(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.sort_values("balanced_imitation_score", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d62728" if flag else "#4c78a8" for flag in plot_df["is_best_variant"]]
    ax.bar(plot_df["variant_label"], plot_df["signal_count"], color=colors, alpha=0.88)
    ax.set_title("Cluster 1 short refinement: signal density")
    ax.set_ylabel("Signal count")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_refinement_overlap_precision(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.sort_values("balanced_imitation_score", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(plot_df))
    width = 0.25

    axes[0].bar(x - width, plot_df["recall_within_30m"], width=width, label="Recall 30m", color="#4c78a8")
    axes[0].bar(x, plot_df["share_within_30m_and_0p50pct"], width=width, label="30m + 0.50%", color="#f58518")
    axes[0].bar(x + width, plot_df["same_short_move_30m_share"], width=width, label="Same move 30m", color="#54a24b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["variant_label"], rotation=35)
    axes[0].set_title("Overlap precision vs recall")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, plot_df["median_time_distance_minutes"], width=width, label="Time distance", color="#4c78a8")
    axes[1].bar(x + width / 2, plot_df["median_price_distance_pct"], width=width, label="Price distance %", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["variant_label"], rotation=35)
    axes[1].set_title("Matched timing and price distance")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_refinement_tradeoff_frontier(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_df["recall_within_30m"],
        plot_df["share_within_30m_and_0p50pct"],
        s=np.maximum(30.0, 400.0 - plot_df["signal_count"].fillna(0).to_numpy(dtype=float) * 0.5),
        c=plot_df["balanced_imitation_score"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
    )
    for _, row in plot_df.iterrows():
        ax.annotate(str(row["variant_label"]), (row["recall_within_30m"], row["share_within_30m_and_0p50pct"]), fontsize=8)
    ax.set_title("Cluster 1 short refinement trade-off frontier")
    ax.set_xlabel("Recall within 30m")
    ax.set_ylabel("Share within 30m and 0.50%")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Balanced imitation score")
    return fig
