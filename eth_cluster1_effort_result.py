from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster1_micro_trigger import (
    ETH_SYMBOL,
    TARGET_CLUSTER_ID,
    TARGET_SIDE,
    TARGET_STAGE_FLAG,
    apply_conditions,
    build_tightened_stage_a_pool,
    infer_micro_trigger_window,
    load_micro_trigger_market_data,
)
from eth_filter_tightening import attach_timing_features
from eth_filter_tightening import prepare_timing_feature_tables
from eth_replication_hypothesis_test import compare_profiles, summarize_signal_like_profiles
from utils import fmt_int, fmt_number, pct, safe_divide


EFFORT_RESULT_BOOLEAN_COLUMNS = [
    "eff_close_below_midpoint_1m_flag",
    "eff_close_below_midpoint_5m_flag",
]


def run_eth_cluster1_effort_result_analysis(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    broad_signals_cluster1: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH CLUSTER 1 EFFORT VS RESULT HYPOTHESIS TEST\n"
        "-----------------------------------------------\n"
        "No sufficient Cluster 1 short data was available for the effort-vs-result hypothesis test.\n"
    )

    real_cluster1 = cluster_frame.loc[
        (cluster_frame["symbol"] == ETH_SYMBOL)
        & (cluster_frame["cluster_id"] == TARGET_CLUSTER_ID)
        & (cluster_frame["side"] == TARGET_SIDE)
    ].copy()
    broad_stage_a = (
        broad_signals_cluster1.loc[broad_signals_cluster1[TARGET_STAGE_FLAG] == True].copy()
        if not broad_signals_cluster1.empty
        else pd.DataFrame()
    )

    if real_cluster1.empty or broad_stage_a.empty:
        return {
            "hypothesis": empty,
            "ranking": empty,
            "quality": empty,
            "section_text": empty_section,
        }

    min_time, max_time = infer_micro_trigger_window(real_cluster1, broad_stage_a)
    minute_frame, frame_5m, _ = load_micro_trigger_market_data(min_time, max_time, logger)
    minute_features, frame_5m_features = prepare_effort_result_feature_tables(minute_frame, frame_5m)

    reference_source = real_cluster1[
        ["trade_number", "cluster_id", "cluster_description", "side", "opened_dt_utc"]
    ].merge(
        entry_quality_df,
        on="trade_number",
        how="left",
        suffixes=("", "_entry"),
    )
    reference = attach_effort_features(
        reference_source.rename(columns={"opened_dt_utc": "entry_time"}),
        minute_features,
        frame_5m_features,
        time_column="entry_time",
    )
    reference["fast_positive_reaction_flag"] = reference["group_fast_positive_reaction"].fillna(False).astype(bool)

    broad_pool = attach_effort_features(
        broad_stage_a.copy(),
        minute_features,
        frame_5m_features,
        time_column="entry_time",
    )
    tightened_pool = build_tightened_stage_a_pool(broad_pool, tightened_summary_df)

    if reference.empty or broad_pool.empty:
        return {
            "hypothesis": empty,
            "ranking": empty,
            "quality": empty,
            "section_text": empty_section,
        }

    real_summary = summarize_signal_like_profiles(reference)
    broad_summary = summarize_signal_like_profiles(broad_pool)
    tightened_summary = summarize_signal_like_profiles(tightened_pool) if not tightened_pool.empty else {}

    candidate_definitions = build_candidate_definitions(broad_pool)
    if not candidate_definitions:
        return {
            "hypothesis": empty,
            "ranking": empty,
            "quality": empty,
            "section_text": (
                "ETH CLUSTER 1 EFFORT VS RESULT HYPOTHESIS TEST\n"
                "-----------------------------------------------\n"
                "No candidate effort-vs-result definitions could be built from the available Stage A Cluster 1 short pool.\n"
            ),
        }

    hypothesis_df = evaluate_candidate_definitions(
        candidate_definitions,
        reference,
        broad_pool,
        tightened_pool,
        real_summary,
        broad_summary,
    )
    ranking_df = build_effort_result_ranking(hypothesis_df)
    quality_df = build_effort_result_quality(ranking_df)
    section_text = build_effort_result_section(quality_df, real_summary, broad_summary, tightened_summary)

    return {
        "hypothesis": hypothesis_df,
        "ranking": ranking_df,
        "quality": quality_df,
        "section_text": section_text,
    }


def prepare_effort_result_feature_tables(
    minute_frame: pd.DataFrame,
    frame_5m: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_minute_features, timing_5m_features = prepare_timing_feature_tables(minute_frame, frame_5m)
    minute = minute_frame[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")
    five = frame_5m[["close_time", "open", "high", "low", "close", "volume"]].copy().sort_values("close_time")

    minute = build_single_timeframe_effort_features(minute, suffix="1m")
    five = build_single_timeframe_effort_features(five, suffix="5m")

    minute_features = minute[
        [
            "close_time",
            "eff_relative_volume_1m",
            "eff_prev_relative_volume_1m",
            "eff_spread_pct_1m",
            "eff_body_pct_1m",
            "eff_close_to_close_pct_1m",
            "eff_prev_close_to_close_pct_1m",
            "eff_net_progress_3m_pct_1m",
            "eff_close_position_in_range_1m",
            "eff_upper_wick_ratio_1m",
            "eff_rel_volume_to_spread_1m",
            "eff_rel_volume_to_body_1m",
            "eff_close_below_midpoint_1m_flag",
        ]
    ].copy()
    five_features = five[
        [
            "close_time",
            "eff_relative_volume_5m",
            "eff_prev_relative_volume_5m",
            "eff_spread_pct_5m",
            "eff_body_pct_5m",
            "eff_close_to_close_pct_5m",
            "eff_prev_close_to_close_pct_5m",
            "eff_net_progress_3m_pct_5m",
            "eff_close_position_in_range_5m",
            "eff_upper_wick_ratio_5m",
            "eff_rel_volume_to_spread_5m",
            "eff_rel_volume_to_body_5m",
            "eff_close_below_midpoint_5m_flag",
        ]
    ].copy()
    minute_features = minute_features.merge(timing_minute_features, on="close_time", how="left")
    five_features = five_features.merge(timing_5m_features, on="close_time", how="left")
    return minute_features, five_features


def build_single_timeframe_effort_features(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    prepared = frame.copy()
    open_series = pd.to_numeric(prepared["open"], errors="coerce")
    high_series = pd.to_numeric(prepared["high"], errors="coerce")
    low_series = pd.to_numeric(prepared["low"], errors="coerce")
    close_series = pd.to_numeric(prepared["close"], errors="coerce")
    volume_series = pd.to_numeric(prepared["volume"], errors="coerce")

    range_abs = (high_series - low_series).replace(0, np.nan)
    body_abs = (close_series - open_series).abs()

    prepared[f"eff_relative_volume_{suffix}"] = volume_series / volume_series.rolling(20, min_periods=5).mean()
    prepared[f"eff_prev_relative_volume_{suffix}"] = prepared[f"eff_relative_volume_{suffix}"].shift(1)
    prepared[f"eff_spread_pct_{suffix}"] = range_abs / close_series * 100.0
    prepared[f"eff_body_pct_{suffix}"] = body_abs / close_series * 100.0
    prepared[f"eff_close_to_close_pct_{suffix}"] = close_series.pct_change(1) * 100.0
    prepared[f"eff_prev_close_to_close_pct_{suffix}"] = prepared[f"eff_close_to_close_pct_{suffix}"].shift(1)
    prepared[f"eff_net_progress_3m_pct_{suffix}"] = close_series.pct_change(3) * 100.0
    prepared[f"eff_close_position_in_range_{suffix}"] = (close_series - low_series) / range_abs
    prepared[f"eff_upper_wick_ratio_{suffix}"] = (high_series - np.maximum(open_series, close_series)) / range_abs
    prepared[f"eff_rel_volume_to_spread_{suffix}"] = prepared[f"eff_relative_volume_{suffix}"] / prepared[f"eff_spread_pct_{suffix}"].replace(0, np.nan)
    prepared[f"eff_rel_volume_to_body_{suffix}"] = prepared[f"eff_relative_volume_{suffix}"] / prepared[f"eff_body_pct_{suffix}"].replace(0, np.nan)
    prepared[f"eff_close_below_midpoint_{suffix}_flag"] = close_series < ((high_series + low_series) / 2.0)
    return prepared


def attach_effort_features(
    entries: pd.DataFrame,
    minute_features: pd.DataFrame,
    frame_5m_features: pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    merged = attach_timing_features(entries, minute_features, frame_5m_features, time_column=time_column)
    for column in EFFORT_RESULT_BOOLEAN_COLUMNS:
        if column in merged.columns:
            merged[column] = merged[column].fillna(False).astype(bool)
    return merged


def build_candidate_definitions(broad_pool: pd.DataFrame) -> list[dict]:
    hv_1m = quantile_value(broad_pool["eff_relative_volume_1m"], 0.75)
    hv_5m = quantile_value(broad_pool["eff_relative_volume_5m"], 0.75)
    small_body_1m = quantile_value(broad_pool["eff_body_pct_1m"], 0.35)
    small_spread_1m = quantile_value(broad_pool["eff_spread_pct_1m"], 0.35)
    weak_close_1m = quantile_value(broad_pool["eff_close_position_in_range_1m"], 0.45)
    no_continuation_1m = quantile_value(broad_pool["eff_close_to_close_pct_1m"], 0.45)
    small_body_5m = quantile_value(broad_pool["eff_body_pct_5m"], 0.35)
    weak_close_5m = quantile_value(broad_pool["eff_close_position_in_range_5m"], 0.45)
    wick_high_1m = quantile_value(broad_pool["eff_upper_wick_ratio_1m"], 0.65)
    effort_ratio_1m = quantile_value(broad_pool["eff_rel_volume_to_spread_1m"], 0.75)
    weak_progress_3m = quantile_value(broad_pool["eff_net_progress_3m_pct_1m"], 0.50)

    candidates = [
        {
            "candidate_id": "hv1m_small_body",
            "hypothesis_label": "High 1m volume + small 1m body",
            "definition_text": f"Relative volume 1m >= {format_threshold(hv_1m)} AND body % 1m <= {format_threshold(small_body_1m)}",
            "conditions": [
                condition_ge("eff_relative_volume_1m", hv_1m),
                condition_le("eff_body_pct_1m", small_body_1m),
            ],
        },
        {
            "candidate_id": "hv1m_small_spread",
            "hypothesis_label": "High 1m volume + small 1m spread",
            "definition_text": f"Relative volume 1m >= {format_threshold(hv_1m)} AND spread % 1m <= {format_threshold(small_spread_1m)}",
            "conditions": [
                condition_ge("eff_relative_volume_1m", hv_1m),
                condition_le("eff_spread_pct_1m", small_spread_1m),
            ],
        },
        {
            "candidate_id": "hv1m_weak_close",
            "hypothesis_label": "High 1m volume + weak 1m close",
            "definition_text": f"Relative volume 1m >= {format_threshold(hv_1m)} AND close position in range 1m <= {format_threshold(weak_close_1m)}",
            "conditions": [
                condition_ge("eff_relative_volume_1m", hv_1m),
                condition_le("eff_close_position_in_range_1m", weak_close_1m),
            ],
        },
        {
            "candidate_id": "prev_hv1m_no_continuation",
            "hypothesis_label": "Previous 1m high volume + current no continuation",
            "definition_text": (
                f"Previous relative volume 1m >= {format_threshold(hv_1m)} AND previous close-to-close % >= 0.00 "
                f"AND current close-to-close % <= {format_threshold(no_continuation_1m)}"
            ),
            "conditions": [
                condition_ge("eff_prev_relative_volume_1m", hv_1m),
                condition_ge("eff_prev_close_to_close_pct_1m", 0.0),
                condition_le("eff_close_to_close_pct_1m", no_continuation_1m),
            ],
        },
        {
            "candidate_id": "hv5m_small_body",
            "hypothesis_label": "High 5m volume + small 5m body",
            "definition_text": f"Relative volume 5m >= {format_threshold(hv_5m)} AND body % 5m <= {format_threshold(small_body_5m)}",
            "conditions": [
                condition_ge("eff_relative_volume_5m", hv_5m),
                condition_le("eff_body_pct_5m", small_body_5m),
            ],
        },
        {
            "candidate_id": "hv5m_weak_close",
            "hypothesis_label": "High 5m volume + weak 5m close",
            "definition_text": f"Relative volume 5m >= {format_threshold(hv_5m)} AND close position in range 5m <= {format_threshold(weak_close_5m)}",
            "conditions": [
                condition_ge("eff_relative_volume_5m", hv_5m),
                condition_le("eff_close_position_in_range_5m", weak_close_5m),
            ],
        },
        {
            "candidate_id": "hv1m_upper_wick_rejection",
            "hypothesis_label": "High 1m volume + upper wick rejection",
            "definition_text": f"Relative volume 1m >= {format_threshold(hv_1m)} AND upper wick ratio 1m >= {format_threshold(wick_high_1m)}",
            "conditions": [
                condition_ge("eff_relative_volume_1m", hv_1m),
                condition_ge("eff_upper_wick_ratio_1m", wick_high_1m),
            ],
        },
        {
            "candidate_id": "high_effort_ratio_weak_progress",
            "hypothesis_label": "High effort/spread ratio + weak 3m progress",
            "definition_text": (
                f"Relative volume/spread ratio 1m >= {format_threshold(effort_ratio_1m)} "
                f"AND net progress 3m % <= {format_threshold(weak_progress_3m)}"
            ),
            "conditions": [
                condition_ge("eff_rel_volume_to_spread_1m", effort_ratio_1m),
                condition_le("eff_net_progress_3m_pct_1m", weak_progress_3m),
            ],
        },
    ]

    valid_candidates: list[dict] = []
    for candidate in candidates:
        if all(is_valid_condition(condition) for condition in candidate["conditions"]):
            valid_candidates.append(candidate)
    return valid_candidates


def evaluate_candidate_definitions(
    candidate_definitions: list[dict],
    reference: pd.DataFrame,
    broad_pool: pd.DataFrame,
    tightened_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
) -> pd.DataFrame:
    rows = []
    broad_similarity = compare_profiles(real_summary, broad_summary)
    tightened_summary = summarize_signal_like_profiles(tightened_pool) if not tightened_pool.empty else {}

    for candidate in candidate_definitions:
        real_hits = apply_conditions(reference, candidate["conditions"])
        broad_hits = apply_conditions(broad_pool, candidate["conditions"])
        tightened_hits = apply_conditions(tightened_pool, candidate["conditions"]) if not tightened_pool.empty else pd.DataFrame()

        if broad_hits.empty:
            continue

        filtered_summary = summarize_signal_like_profiles(broad_hits)
        filtered_similarity = compare_profiles(real_summary, filtered_summary)

        frequency_real = safe_divide(len(real_hits), len(reference))
        frequency_broad = safe_divide(len(broad_hits), len(broad_pool))
        frequency_tightened = safe_divide(len(tightened_hits), len(tightened_pool)) if not tightened_pool.empty else np.nan
        lift = safe_divide(frequency_real, frequency_broad)
        precision_proxy = safe_divide(frequency_real, frequency_real + frequency_broad)
        recall_proxy = frequency_real

        similarity_improvement = safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"])
        pct_positive_5m_improvement = safe_number(filtered_summary.get("pct_positive_after_5m")) - safe_number(
            broad_summary.get("pct_positive_after_5m")
        )
        mae_improvement = safe_number(broad_summary.get("median_MAE_pct")) - safe_number(filtered_summary.get("median_MAE_pct"))
        count_score = min(safe_divide(len(broad_hits), max(min(24, len(broad_pool)), 1)), 1.0)
        tightened_alignment = frequency_real - safe_number(frequency_tightened) if pd.notna(frequency_tightened) else 0.0

        ranking_score = (
            max(safe_number(lift) - 1.0, 0.0) * 0.30
            + max(similarity_improvement, 0.0) * 0.30
            + max(pct_positive_5m_improvement, 0.0) * 0.15
            + max(mae_improvement, 0.0) * 0.10
            + max(tightened_alignment, 0.0) * 0.10
            + count_score * 0.05
        )

        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "hypothesis_label": candidate["hypothesis_label"],
                "definition_text": candidate["definition_text"],
                "frequency_in_real_trades": frequency_real,
                "frequency_in_broad_signals": frequency_broad,
                "frequency_in_tightened_signals": frequency_tightened,
                "lift": lift,
                "precision_proxy_balanced": precision_proxy,
                "recall_proxy": recall_proxy,
                "effect_direction": "more_common_in_real" if frequency_real > frequency_broad else "more_common_in_broad",
                "signal_count_after_filter": int(len(broad_hits)),
                "tightened_signal_count_after_filter": int(len(tightened_hits)) if not tightened_pool.empty else 0,
                "reduction_vs_broad_pool": 1 - safe_divide(len(broad_hits), len(broad_pool)),
                "reduction_vs_tightened_pool": 1 - safe_divide(len(tightened_hits), len(tightened_pool)) if not tightened_pool.empty else np.nan,
                "pct_positive_1m": filtered_summary.get("pct_positive_after_1m"),
                "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
                "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
                "median_MAE": filtered_summary.get("median_MAE_pct"),
                "median_MFE": filtered_summary.get("median_MFE_pct"),
                "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
                "similarity_to_real_cluster1_profile": filtered_similarity["score"],
                "similarity_label": filtered_similarity["label"],
                "similarity_improvement_vs_broad": similarity_improvement,
                "pct_positive_5m_improvement_vs_broad": pct_positive_5m_improvement,
                "mae_improvement_vs_broad": mae_improvement,
                "ranking_score": ranking_score,
                "broad_signal_count": int(len(broad_pool)),
                "tightened_signal_count": int(len(tightened_pool)) if not tightened_pool.empty else 0,
                "real_trade_count": int(len(reference)),
                "broad_pct_positive_5m": broad_summary.get("pct_positive_after_5m"),
                "broad_median_MAE": broad_summary.get("median_MAE_pct"),
                "tightened_pct_positive_5m": tightened_summary.get("pct_positive_after_5m"),
                "tightened_median_MAE": tightened_summary.get("median_MAE_pct"),
            }
        )

    return pd.DataFrame(rows)


def build_effort_result_ranking(hypothesis_df: pd.DataFrame) -> pd.DataFrame:
    if hypothesis_df.empty:
        return pd.DataFrame()
    return hypothesis_df.sort_values(
        ["ranking_score", "similarity_improvement_vs_broad", "lift", "signal_count_after_filter"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_effort_result_quality(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return pd.DataFrame()
    columns = [
        "candidate_id",
        "hypothesis_label",
        "definition_text",
        "frequency_in_real_trades",
        "frequency_in_broad_signals",
        "frequency_in_tightened_signals",
        "lift",
        "signal_count_after_filter",
        "tightened_signal_count_after_filter",
        "reduction_vs_broad_pool",
        "reduction_vs_tightened_pool",
        "pct_positive_1m",
        "pct_positive_5m",
        "pct_positive_15m",
        "median_MAE",
        "median_MFE",
        "median_return_1h",
        "similarity_to_real_cluster1_profile",
        "similarity_label",
        "similarity_improvement_vs_broad",
        "pct_positive_5m_improvement_vs_broad",
        "mae_improvement_vs_broad",
        "ranking_score",
    ]
    return ranking_df[columns].copy()


def build_effort_result_section(
    quality_df: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
    tightened_summary: dict,
) -> str:
    if quality_df.empty:
        return (
            "ETH CLUSTER 1 EFFORT VS RESULT HYPOTHESIS TEST\n"
            "-----------------------------------------------\n"
            "The effort-vs-result hypothesis did not produce any usable candidate definitions.\n"
        )

    best = quality_df.iloc[0]
    confidence = determine_effort_result_confidence(best)
    meaningful_clue = (
        safe_number(best["lift"]) >= 1.20
        and safe_number(best["similarity_improvement_vs_broad"]) >= 0.03
        and int(best["signal_count_after_filter"]) >= 12
    )
    verdict = (
        "The best effort-vs-result definition looks like a meaningful possible execution clue, but still not a full execution rule."
        if meaningful_clue
        else "The evidence is weak or noisy, so effort-vs-result currently looks more like a secondary clue than a reliable execution layer."
    )

    report = f"""
ETH CLUSTER 1 EFFORT VS RESULT HYPOTHESIS TEST
-----------------------------------------------
This section tests whether a high-volume / weak-price-progress pattern is statistically more common before real Cluster 1 short entries than inside the broader Cluster 1 short context family.

Key findings:
- Best candidate hypothesis: {best['hypothesis_label']}.
- Candidate definition: {best['definition_text']}.
- Frequency in real trades: {pct(best['frequency_in_real_trades'])}; broad Stage A signals: {pct(best['frequency_in_broad_signals'])}; tightened Stage A signals: {pct(best['frequency_in_tightened_signals']) if pd.notna(best['frequency_in_tightened_signals']) else 'n/a'}.
- Coverage note: this pattern appears in only {pct(best['frequency_in_real_trades'])} of real Cluster 1 short entries, so it looks like a narrow clue rather than a dominant execution layer.
- Lift vs broad pool: {fmt_number(best['lift'])}. Broad Stage A signals: {fmt_int(broad_summary.get('signal_count'))}; tightened Stage A signals: {fmt_int(tightened_summary.get('signal_count')) if tightened_summary else 'n/a'}; filtered signals: {fmt_int(best['signal_count_after_filter'])}.
- Positive after 5m: broad {pct(broad_summary.get('pct_positive_after_5m'))}, tightened {pct(tightened_summary.get('pct_positive_after_5m')) if tightened_summary else 'n/a'}, effort/result filter {pct(best['pct_positive_5m'])}.
- Median MAE: broad {fmt_number(broad_summary.get('median_MAE_pct'))}%, tightened {fmt_number(tightened_summary.get('median_MAE_pct')) if tightened_summary else np.nan}%, effort/result filter {fmt_number(best['median_MAE'])}%.
- Similarity to real Cluster 1 profile: {fmt_number(best['similarity_to_real_cluster1_profile'])} ({best['similarity_label']}).
- Confidence level: {confidence}.

Final verdict:
- {verdict}
""".strip()
    return report + "\n"


def determine_effort_result_confidence(best_row: pd.Series) -> str:
    if best_row.empty:
        return "low"
    if (
        safe_number(best_row.get("lift")) >= 1.40
        and safe_number(best_row.get("similarity_improvement_vs_broad")) >= 0.05
        and safe_number(best_row.get("frequency_in_real_trades")) >= 0.15
        and safe_number(best_row.get("signal_count_after_filter")) >= 16
    ):
        return "high"
    if (
        safe_number(best_row.get("lift")) >= 1.15
        and safe_number(best_row.get("similarity_improvement_vs_broad")) >= 0.02
        and safe_number(best_row.get("frequency_in_real_trades")) >= 0.08
        and safe_number(best_row.get("signal_count_after_filter")) >= 10
    ):
        return "medium"
    return "low"


def create_eth_cluster1_effort_result_plots(
    ranking_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if ranking_df.empty or quality_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = plots_dir / "eth_cluster1_effort_result_comparison.png"
    importance_path = plots_dir / "eth_cluster1_effort_result_feature_importance.png"

    fig_comparison = plot_effort_result_comparison(ranking_df)
    fig_comparison.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig_comparison)
    logger.info("Saved plot: %s", comparison_path.name)

    fig_importance = plot_effort_result_importance(ranking_df)
    fig_importance.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close(fig_importance)
    logger.info("Saved plot: %s", importance_path.name)


def plot_effort_result_comparison(ranking_df: pd.DataFrame) -> plt.Figure:
    plot_df = ranking_df.head(5).copy().sort_values("ranking_score")
    labels = plot_df["hypothesis_label"].astype(str).tolist()
    y = np.arange(len(plot_df))
    height = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].barh(y - height, plot_df["frequency_in_real_trades"], height=height, label="Real trades", color="#4c78a8")
    axes[0].barh(y, plot_df["frequency_in_broad_signals"], height=height, label="Broad signals", color="#9ecae9")
    axes[0].barh(y + height, plot_df["frequency_in_tightened_signals"].fillna(0), height=height, label="Tightened signals", color="#f58518")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Frequency of effort/result patterns")
    axes[0].set_xlabel("Frequency")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend()

    axes[1].scatter(plot_df["signal_count_after_filter"], plot_df["similarity_to_real_cluster1_profile"], s=90, color="#54a24b")
    for _, row in plot_df.iterrows():
        axes[1].annotate(
            row["hypothesis_label"],
            (row["signal_count_after_filter"], row["similarity_to_real_cluster1_profile"]),
            fontsize=8,
        )
    axes[1].set_title("Filtered signal count vs similarity")
    axes[1].set_xlabel("Signal count after filter")
    axes[1].set_ylabel("Similarity to real Cluster 1 profile")
    axes[1].grid(alpha=0.25)
    return fig


def plot_effort_result_importance(ranking_df: pd.DataFrame) -> plt.Figure:
    plot_df = ranking_df.head(10).copy().sort_values("ranking_score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["hypothesis_label"], plot_df["ranking_score"], color="#6f4e7c")
    ax.set_title("Cluster 1 effort-vs-result ranking")
    ax.set_xlabel("Ranking score")
    ax.grid(axis="x", alpha=0.25)
    return fig


def condition_ge(feature_key: str, threshold: float) -> dict:
    return {"feature_key": feature_key, "rule_type": "ge", "threshold": threshold}


def condition_le(feature_key: str, threshold: float) -> dict:
    return {"feature_key": feature_key, "rule_type": "le", "threshold": threshold}


def is_valid_condition(condition: dict) -> bool:
    threshold = condition.get("threshold")
    return threshold is None or pd.notna(threshold)


def quantile_value(series: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(quantile))


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
