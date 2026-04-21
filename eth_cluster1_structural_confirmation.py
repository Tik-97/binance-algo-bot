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
    infer_micro_trigger_window,
    load_micro_trigger_market_data,
    prepare_micro_feature_tables,
)
from eth_filter_tightening import attach_timing_features
from eth_replication_hypothesis_test import compare_profiles, summarize_signal_like_profiles
from utils import fmt_int, fmt_number, pct, safe_divide


BASE_TRIGGER_CONDITIONS = [
    {
        "feature_key": "micro_close_below_midpoint_after_upper_push_flag",
        "rule_type": "equals",
        "value": True,
        "text": "Close below midpoint after upper push = True",
    },
    {
        "feature_key": "micro_body_small_flag",
        "rule_type": "equals",
        "value": True,
        "text": "Small candle body = True",
    },
]
BASE_TRIGGER_TEXT = "Close below midpoint after upper push = True AND Small candle body = True"

STRUCTURAL_CONFIRMATION_SPECS = [
    {
        "key": "micro_failed_breakout_wick_over_5m_high_flag",
        "label": "Failed breakout wick over 5m high",
        "interpretability": 1.00,
    },
    {
        "key": "micro_failed_breakout_wick_over_15m_high_flag",
        "label": "Failed breakout wick over 15m high",
        "interpretability": 1.00,
    },
    {
        "key": "micro_close_back_below_5m_high_flag",
        "label": "Close back below 5m high",
        "interpretability": 0.95,
    },
    {
        "key": "micro_close_back_below_15m_high_flag",
        "label": "Close back below 15m high",
        "interpretability": 0.95,
    },
    {
        "key": "micro_return_inside_prior_micro_range_flag",
        "label": "Return inside prior micro-range",
        "interpretability": 0.95,
    },
    {
        "key": "micro_break_above_range_then_reentry_flag",
        "label": "Break above range then re-entry",
        "interpretability": 1.00,
    },
    {
        "key": "micro_rejection_candle_near_local_high_flag",
        "label": "Rejection candle near local high",
        "interpretability": 0.90,
    },
    {
        "key": "micro_upper_wick_gt_threshold_flag",
        "label": "Upper wick above threshold",
        "interpretability": 0.85,
    },
]


def run_eth_cluster1_structural_confirmation_analysis(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    broad_signals_cluster1: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH CLUSTER 1 FINAL STRUCTURAL CONFIRMATION TEST\n"
        "-------------------------------------------------\n"
        "No sufficient Cluster 1 short data was available for the final structural confirmation test.\n"
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
    base_trigger_pool = apply_conditions(broad_pool, BASE_TRIGGER_CONDITIONS)
    reference_base_hits = apply_conditions(reference, BASE_TRIGGER_CONDITIONS)

    if base_trigger_pool.empty:
        return {
            "ranking": empty,
            "candidates": empty,
            "quality": empty,
            "section_text": (
                "ETH CLUSTER 1 FINAL STRUCTURAL CONFIRMATION TEST\n"
                "-------------------------------------------------\n"
                "The current best Cluster 1 base trigger did not produce any Stage A signals, so structural confirmation could not be tested.\n"
            ),
        }

    real_summary = summarize_signal_like_profiles(reference)
    broad_summary = summarize_signal_like_profiles(broad_pool)
    base_summary = summarize_signal_like_profiles(base_trigger_pool)
    base_similarity = compare_profiles(real_summary, base_summary)

    ranking_df = rank_structural_confirmations(
        reference,
        reference_base_hits,
        broad_pool,
        base_trigger_pool,
        real_summary,
        base_summary,
        base_similarity,
    )
    candidates_df = ranking_df.head(5).copy() if not ranking_df.empty else pd.DataFrame()
    if not candidates_df.empty:
        candidates_df["is_best_final_candidate"] = False
        candidates_df.loc[candidates_df.index[0], "is_best_final_candidate"] = True

    quality_df = build_quality_frame(
        reference,
        broad_pool,
        base_trigger_pool,
        real_summary,
        broad_summary,
        base_summary,
        candidates_df,
    )
    section_text = build_structural_confirmation_section(
        ranking_df,
        quality_df,
        len(reference),
        len(reference_base_hits),
    )

    return {
        "ranking": ranking_df,
        "candidates": candidates_df,
        "quality": quality_df,
        "section_text": section_text,
    }


def rank_structural_confirmations(
    reference: pd.DataFrame,
    reference_base_hits: pd.DataFrame,
    broad_pool: pd.DataFrame,
    base_trigger_pool: pd.DataFrame,
    real_summary: dict,
    base_summary: dict,
    base_similarity: dict,
) -> pd.DataFrame:
    rows = []
    for spec in STRUCTURAL_CONFIRMATION_SPECS:
        candidate_conditions = [
            *BASE_TRIGGER_CONDITIONS,
            {
                "feature_key": spec["key"],
                "rule_type": "equals",
                "value": True,
                "text": f"{spec['label']} = True",
            },
        ]
        candidate_pool = apply_conditions(broad_pool, candidate_conditions)
        if candidate_pool.empty:
            continue
        if len(candidate_pool) >= len(base_trigger_pool):
            continue

        candidate_reference_hits = apply_conditions(reference, candidate_conditions)
        candidate_summary = summarize_signal_like_profiles(candidate_pool)
        candidate_similarity = compare_profiles(real_summary, candidate_summary)

        coverage_on_real = safe_divide(len(candidate_reference_hits), len(reference))
        coverage_on_real_base = safe_divide(len(candidate_reference_hits), len(reference_base_hits))
        frequency_on_base = safe_divide(len(candidate_pool), len(base_trigger_pool))
        lift = safe_divide(coverage_on_real, frequency_on_base)
        precision_proxy = safe_divide(coverage_on_real, coverage_on_real + frequency_on_base)
        similarity_improvement = safe_number(candidate_similarity["score"]) - safe_number(base_similarity["score"])
        pct_positive_5m_improvement = safe_number(candidate_summary.get("pct_positive_after_5m")) - safe_number(
            base_summary.get("pct_positive_after_5m")
        )
        base_mae = safe_number(base_summary.get("median_MAE_pct"))
        candidate_mae = safe_number(candidate_summary.get("median_MAE_pct"))
        mae_improvement = base_mae - candidate_mae
        mae_improvement_ratio = safe_divide(mae_improvement, base_mae if base_mae != 0 else np.nan)
        min_reasonable_signals = max(12, int(len(reference) * 0.40))
        acceptable_signal_count = len(candidate_pool) >= min_reasonable_signals
        count_score = min(safe_divide(len(candidate_pool), max(min_reasonable_signals, 1)), 1.0)

        ranking_score = (
            similarity_improvement * 0.40
            + pct_positive_5m_improvement * 0.25
            + safe_number(mae_improvement_ratio) * 0.15
            + max(safe_number(lift) - 1.0, 0.0) * 0.10
            + spec["interpretability"] * 0.05
            + count_score * 0.05
        )
        if (
            similarity_improvement <= 0
            and pct_positive_5m_improvement <= 0
            and mae_improvement <= 0
        ):
            ranking_score -= 0.08
        if len(candidate_pool) < 8:
            ranking_score -= 0.25
        elif len(candidate_pool) < min_reasonable_signals:
            ranking_score -= 0.10

        rows.append(
            {
                "confirmation_key": spec["key"],
                "confirmation_label": spec["label"],
                "base_trigger_text": BASE_TRIGGER_TEXT,
                "final_trigger_text": f"{BASE_TRIGGER_TEXT} AND {spec['label']} = True",
                "interpretability_score": spec["interpretability"],
                "coverage_on_real_cluster1_trades": coverage_on_real,
                "coverage_on_real_base_trigger_hits": coverage_on_real_base,
                "frequency_on_base_trigger_signals": frequency_on_base,
                "lift_vs_base_trigger": lift,
                "precision_proxy": precision_proxy,
                "signal_count": int(len(candidate_pool)),
                "acceptable_signal_count_flag": acceptable_signal_count,
                "reduction_vs_broad_pool": 1 - safe_divide(len(candidate_pool), len(broad_pool)),
                "reduction_vs_current_best_trigger": 1 - safe_divide(len(candidate_pool), len(base_trigger_pool)),
                "pct_positive_1m": candidate_summary.get("pct_positive_after_1m"),
                "pct_positive_5m": candidate_summary.get("pct_positive_after_5m"),
                "pct_positive_15m": candidate_summary.get("pct_positive_after_15m"),
                "median_MAE": candidate_summary.get("median_MAE_pct"),
                "median_MFE": candidate_summary.get("median_MFE_pct"),
                "median_return_1h": candidate_summary.get("median_return_after_60m_pct"),
                "median_return_4h": candidate_summary.get("median_return_after_240m_pct"),
                "similarity_to_real_cluster1_profile": candidate_similarity["score"],
                "similarity_label": candidate_similarity["label"],
                "similarity_improvement_vs_base_trigger": similarity_improvement,
                "pct_positive_5m_improvement_vs_base_trigger": pct_positive_5m_improvement,
                "median_MAE_improvement_vs_base_trigger": mae_improvement,
                "ranking_score": ranking_score,
            }
        )

    ranking_df = pd.DataFrame(rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            [
                "ranking_score",
                "similarity_improvement_vs_base_trigger",
                "pct_positive_5m_improvement_vs_base_trigger",
                "signal_count",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return ranking_df


def build_quality_frame(
    reference: pd.DataFrame,
    broad_pool: pd.DataFrame,
    base_trigger_pool: pd.DataFrame,
    real_summary: dict,
    broad_summary: dict,
    base_summary: dict,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        build_quality_row(
            scenario_type="broad_pool",
            scenario_label="Broad Stage A Cluster 1 short pool",
            trigger_text="Broad Cluster 1 short context family",
            frame=broad_pool,
            real_summary=real_summary,
            broad_count=len(broad_pool),
            base_count=len(base_trigger_pool),
        ),
        build_quality_row(
            scenario_type="base_trigger",
            scenario_label="Current best micro-trigger only",
            trigger_text=BASE_TRIGGER_TEXT,
            frame=base_trigger_pool,
            real_summary=real_summary,
            broad_count=len(broad_pool),
            base_count=len(base_trigger_pool),
        ),
    ]

    for idx, row in candidates_df.iterrows():
        candidate_conditions = [
            *BASE_TRIGGER_CONDITIONS,
            {
                "feature_key": row["confirmation_key"],
                "rule_type": "equals",
                "value": True,
                "text": f"{row['confirmation_label']} = True",
            },
        ]
        candidate_pool = apply_conditions(broad_pool, candidate_conditions)
        candidate_row = build_quality_row(
            scenario_type="final_candidate" if idx == 0 else "candidate",
            scenario_label=f"Base trigger + {row['confirmation_label']}",
            trigger_text=row["final_trigger_text"],
            frame=candidate_pool,
            real_summary=real_summary,
            broad_count=len(broad_pool),
            base_count=len(base_trigger_pool),
        )
        candidate_row["confirmation_key"] = row["confirmation_key"]
        candidate_row["confirmation_label"] = row["confirmation_label"]
        candidate_row["is_best_final_candidate"] = idx == 0
        rows.append(candidate_row)

    quality_df = pd.DataFrame(rows)
    if not quality_df.empty:
        quality_df["base_pct_positive_5m"] = base_summary.get("pct_positive_after_5m")
        quality_df["base_median_MAE"] = base_summary.get("median_MAE_pct")
        quality_df["broad_pct_positive_5m"] = broad_summary.get("pct_positive_after_5m")
        quality_df["broad_median_MAE"] = broad_summary.get("median_MAE_pct")
    return quality_df


def build_quality_row(
    scenario_type: str,
    scenario_label: str,
    trigger_text: str,
    frame: pd.DataFrame,
    real_summary: dict,
    broad_count: int,
    base_count: int,
) -> dict:
    summary = summarize_signal_like_profiles(frame)
    similarity = compare_profiles(real_summary, summary)
    return {
        "scenario_type": scenario_type,
        "scenario_label": scenario_label,
        "trigger_text": trigger_text,
        "signal_count": int(len(frame)),
        "reduction_vs_broad_pool": 1 - safe_divide(len(frame), broad_count),
        "reduction_vs_current_best_trigger": (
            np.nan if scenario_type == "broad_pool" else (1 - safe_divide(len(frame), base_count) if base_count else np.nan)
        ),
        "pct_positive_1m": summary.get("pct_positive_after_1m"),
        "pct_positive_5m": summary.get("pct_positive_after_5m"),
        "pct_positive_15m": summary.get("pct_positive_after_15m"),
        "median_MAE": summary.get("median_MAE_pct"),
        "median_MFE": summary.get("median_MFE_pct"),
        "median_return_1h": summary.get("median_return_after_60m_pct"),
        "median_return_4h": summary.get("median_return_after_240m_pct"),
        "similarity_to_real_cluster1_profile": similarity["score"],
        "similarity_label": similarity["label"],
    }


def build_structural_confirmation_section(
    ranking_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    real_trade_count: int,
    real_base_trigger_hits: int,
) -> str:
    if ranking_df.empty or quality_df.empty or not (quality_df["scenario_type"] == "final_candidate").any():
        return (
            "ETH CLUSTER 1 FINAL STRUCTURAL CONFIRMATION TEST\n"
            "-------------------------------------------------\n"
            "No extra structural confirmation produced a meaningful additional execution layer beyond the current best micro-trigger.\n"
        )

    base_row = quality_df.loc[quality_df["scenario_type"] == "base_trigger"].iloc[0]
    best_row = quality_df.loc[quality_df["scenario_type"] == "final_candidate"].iloc[0]
    best_candidate = ranking_df.iloc[0]
    top_labels = ", ".join(ranking_df.head(4)["confirmation_label"].astype(str).tolist())
    confidence = determine_structural_confirmation_confidence(best_candidate)
    backtest_ready = bool(
        pd.notna(best_row["similarity_to_real_cluster1_profile"])
        and best_row["similarity_to_real_cluster1_profile"] >= 0.70
        and pd.notna(best_candidate["pct_positive_5m_improvement_vs_base_trigger"])
        and best_candidate["pct_positive_5m_improvement_vs_base_trigger"] >= 0.02
        and best_row["signal_count"] >= 20
    )
    limit_verdict = (
        "There is still room for a very small exploratory reaction-based backtest."
        if backtest_ready
        else "This looks close to the practical ceiling of rule-based reverse engineering from public trade history alone."
    )

    report = f"""
ETH CLUSTER 1 FINAL STRUCTURAL CONFIRMATION TEST
-------------------------------------------------
This stage keeps the current best Cluster 1 short micro-trigger fixed and adds exactly one extra structural confirmation. The goal is to see whether one simple confirmation can make the short execution layer meaningfully closer to the trader's real Cluster 1 short entries.

Current base execution layer:
- {BASE_TRIGGER_TEXT}

Main findings:
- Real Cluster 1 short trades: {fmt_int(real_trade_count)}. Real trades matching the current base trigger: {fmt_int(real_base_trigger_hits)}.
- Structural confirmations tested: {top_labels}.
- Most useful structural confirmation: {best_candidate['confirmation_label']}.
- Final Cluster 1 short replication candidate: {best_candidate['final_trigger_text']}.
- Signal count: broad {fmt_int(quality_df.loc[quality_df['scenario_type'] == 'broad_pool', 'signal_count'].iloc[0])}, base trigger {fmt_int(base_row['signal_count'])}, final candidate {fmt_int(best_row['signal_count'])}.
- Positive after 5m: base {pct(base_row['pct_positive_5m'])}, final candidate {pct(best_row['pct_positive_5m'])}.
- Median MAE: base {fmt_number(base_row['median_MAE'])}%, final candidate {fmt_number(best_row['median_MAE'])}%.
- Similarity to real Cluster 1 profile: base {fmt_number(base_row['similarity_to_real_cluster1_profile'])}, final candidate {fmt_number(best_row['similarity_to_real_cluster1_profile'])} ({best_row['similarity_label']}).
- Confidence level: {confidence}.

Final verdict:
- The extra structural confirmation {"did" if safe_number(best_candidate['similarity_improvement_vs_base_trigger']) >= 0.03 else "did not"} deliver a meaningful jump over the current best micro-trigger.
- {limit_verdict}
""".strip()
    return report + "\n"


def determine_structural_confirmation_confidence(best_candidate: pd.Series) -> str:
    if best_candidate.empty:
        return "low"
    if (
        pd.notna(best_candidate["similarity_improvement_vs_base_trigger"])
        and best_candidate["similarity_improvement_vs_base_trigger"] >= 0.08
        and pd.notna(best_candidate["pct_positive_5m_improvement_vs_base_trigger"])
        and best_candidate["pct_positive_5m_improvement_vs_base_trigger"] >= 0.04
        and best_candidate["signal_count"] >= 20
    ):
        return "high"
    if (
        pd.notna(best_candidate["similarity_improvement_vs_base_trigger"])
        and best_candidate["similarity_improvement_vs_base_trigger"] >= 0.03
        and pd.notna(best_candidate["median_MAE_improvement_vs_base_trigger"])
        and best_candidate["median_MAE_improvement_vs_base_trigger"] > 0
        and best_candidate["signal_count"] >= 12
    ):
        return "medium"
    return "low"


def create_eth_cluster1_structural_confirmation_plots(
    ranking_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if ranking_df.empty or quality_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    improvement_path = plots_dir / "eth_cluster1_structural_confirmation_improvement.png"
    comparison_path = plots_dir / "eth_cluster1_final_trigger_comparison.png"

    fig_improvement = plot_structural_confirmation_improvement(ranking_df)
    fig_improvement.savefig(improvement_path, dpi=150, bbox_inches="tight")
    plt.close(fig_improvement)
    logger.info("Saved plot: %s", improvement_path.name)

    fig_comparison = plot_final_trigger_comparison(quality_df)
    fig_comparison.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig_comparison)
    logger.info("Saved plot: %s", comparison_path.name)


def plot_structural_confirmation_improvement(ranking_df: pd.DataFrame) -> plt.Figure:
    plot_df = ranking_df.head(8).sort_values("ranking_score")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].barh(plot_df["confirmation_label"], plot_df["ranking_score"], color="#2a9d8f")
    axes[0].set_title("Structural Confirmation Ranking")
    axes[0].set_xlabel("Ranking score")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(
        plot_df["signal_count"],
        plot_df["similarity_improvement_vs_base_trigger"],
        s=90,
        color="#e76f51",
    )
    for _, row in plot_df.iterrows():
        axes[1].annotate(
            row["confirmation_label"],
            (row["signal_count"], row["similarity_improvement_vs_base_trigger"]),
            fontsize=8,
        )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Signal Count vs Similarity Improvement")
    axes[1].set_xlabel("Signals after confirmation")
    axes[1].set_ylabel("Similarity improvement vs base trigger")
    axes[1].grid(alpha=0.25)
    return fig


def plot_final_trigger_comparison(quality_df: pd.DataFrame) -> plt.Figure:
    broad_row = quality_df.loc[quality_df["scenario_type"] == "broad_pool"].iloc[0]
    base_row = quality_df.loc[quality_df["scenario_type"] == "base_trigger"].iloc[0]
    final_row = quality_df.loc[quality_df["scenario_type"] == "final_candidate"].iloc[0]

    labels = ["Broad", "Base", "Final"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(labels, [broad_row["signal_count"], base_row["signal_count"], final_row["signal_count"]], color=["#bbbbbb", "#4c78a8", "#e76f51"])
    axes[0, 0].set_title("Signal Count")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(labels, [broad_row["pct_positive_5m"], base_row["pct_positive_5m"], final_row["pct_positive_5m"]], color=["#bbbbbb", "#4c78a8", "#e76f51"])
    axes[0, 1].set_title("Positive After 5m")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(labels, [broad_row["median_MAE"], base_row["median_MAE"], final_row["median_MAE"]], color=["#bbbbbb", "#4c78a8", "#e76f51"])
    axes[1, 0].set_title("Median MAE %")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].bar(
        labels,
        [
            broad_row["similarity_to_real_cluster1_profile"],
            base_row["similarity_to_real_cluster1_profile"],
            final_row["similarity_to_real_cluster1_profile"],
        ],
        color=["#bbbbbb", "#4c78a8", "#e76f51"],
    )
    axes[1, 1].set_title("Similarity to Real Cluster 1 Profile")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle("Cluster 1 Short: Broad vs Base vs Final Structural Confirmation")
    return fig


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0
