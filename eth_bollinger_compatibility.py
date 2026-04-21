from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster1_micro_trigger import (
    ETH_SYMBOL,
    TARGET_STAGE_FLAG,
    apply_conditions,
    infer_micro_trigger_window,
    load_micro_trigger_market_data,
    parse_timing_rule_text,
)
from eth_filter_tightening import attach_timing_features, prepare_timing_feature_tables
from eth_replication_hypothesis_test import compare_profiles, summarize_signal_like_profiles
from utils import fmt_int, fmt_number, pct, safe_divide


BB_PERIOD = 20
BB_STDDEV = 2.0
SHORT_CLUSTER_ID = 1
LONG_CLUSTER_ID = 2


def run_eth_bollinger_compatibility_test(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    signals_cluster1: pd.DataFrame,
    signals_cluster2: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "BOLLINGER COMPATIBILITY TEST\n"
        "----------------------------\n"
        "No sufficient ETH cluster data was available for Bollinger compatibility testing.\n"
    )

    real_short = build_real_reference(cluster_frame, entry_quality_df, SHORT_CLUSTER_ID, "Short")
    real_long = build_real_reference(cluster_frame, entry_quality_df, LONG_CLUSTER_ID, "Long")
    broad_short = build_stage_pool(signals_cluster1)
    broad_long = build_stage_pool(signals_cluster2)

    if real_short.empty and real_long.empty:
        return {
            "summary": empty,
            "short_hypotheses": empty,
            "long_hypotheses": empty,
            "section_text": empty_section,
        }

    min_time, max_time = infer_bollinger_window(real_short, real_long, broad_short, broad_long)
    minute_frame, frame_5m, frame_15m = load_micro_trigger_market_data(min_time, max_time, logger)
    minute_timing, timing_5m = prepare_timing_feature_tables(minute_frame, frame_5m)
    bb_5m = prepare_bollinger_feature_table(frame_5m, timeframe_label="5m")
    bb_15m = prepare_bollinger_feature_table(frame_15m, timeframe_label="15m")

    short_result = analyze_bollinger_family(
        family_label="short_family",
        cluster_id=SHORT_CLUSTER_ID,
        side="Short",
        real_reference=real_short,
        broad_pool=broad_short,
        tightened_summary_df=tightened_summary_df,
        minute_timing=minute_timing,
        timing_5m=timing_5m,
        bb_5m=bb_5m,
        bb_15m=bb_15m,
    )
    long_result = analyze_bollinger_family(
        family_label="long_family",
        cluster_id=LONG_CLUSTER_ID,
        side="Long",
        real_reference=real_long,
        broad_pool=broad_long,
        tightened_summary_df=tightened_summary_df,
        minute_timing=minute_timing,
        timing_5m=timing_5m,
        bb_5m=bb_5m,
        bb_15m=bb_15m,
    )

    summary_df = build_bollinger_summary(short_result, long_result)
    section_text = build_bollinger_section(summary_df, short_result, long_result)

    return {
        "summary": summary_df,
        "short_hypotheses": short_result["hypotheses"],
        "long_hypotheses": long_result["hypotheses"],
        "section_text": section_text,
    }


def build_real_reference(
    cluster_frame: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    cluster_id: int,
    side: str,
) -> pd.DataFrame:
    source = cluster_frame.loc[
        (cluster_frame["symbol"] == ETH_SYMBOL)
        & (cluster_frame["cluster_id"] == cluster_id)
        & (cluster_frame["side"] == side)
    ].copy()
    if source.empty:
        return pd.DataFrame()

    merged = source[
        ["trade_number", "cluster_id", "cluster_description", "side", "opened_dt_utc"]
    ].merge(
        entry_quality_df,
        on="trade_number",
        how="left",
        suffixes=("", "_entry"),
    )
    merged["entry_time"] = pd.to_datetime(merged["opened_dt_utc"], utc=True, errors="coerce")
    merged["fast_positive_reaction_flag"] = merged["group_fast_positive_reaction"].fillna(False).astype(bool)
    return merged


def build_stage_pool(signals_df: pd.DataFrame) -> pd.DataFrame:
    if signals_df.empty:
        return pd.DataFrame()
    if TARGET_STAGE_FLAG in signals_df.columns:
        return signals_df.loc[signals_df[TARGET_STAGE_FLAG] == True].copy()
    return signals_df.copy()


def infer_bollinger_window(
    real_short: pd.DataFrame,
    real_long: pd.DataFrame,
    broad_short: pd.DataFrame,
    broad_long: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = []
    for frame, column in [
        (real_short, "entry_time"),
        (real_long, "entry_time"),
        (broad_short, "entry_time"),
        (broad_long, "entry_time"),
    ]:
        if not frame.empty and column in frame.columns:
            timestamps.append(pd.to_datetime(frame[column], utc=True, errors="coerce"))
    if not timestamps:
        now = pd.Timestamp.now(tz="UTC")
        return now - pd.Timedelta(days=30), now
    combined = pd.concat(timestamps, ignore_index=True).dropna()
    return combined.min() - pd.Timedelta(hours=24), combined.max() + pd.Timedelta(hours=2)


def prepare_bollinger_feature_table(frame: pd.DataFrame, timeframe_label: str) -> pd.DataFrame:
    prepared = frame[["close_time", "open", "high", "low", "close"]].copy().sort_values("close_time")
    close_series = pd.to_numeric(prepared["close"], errors="coerce")
    open_series = pd.to_numeric(prepared["open"], errors="coerce")
    high_series = pd.to_numeric(prepared["high"], errors="coerce")
    low_series = pd.to_numeric(prepared["low"], errors="coerce")

    mid = close_series.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    std = close_series.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    upper = mid + BB_STDDEV * std
    lower = mid - BB_STDDEV * std
    band_range = (upper - lower).replace(0, np.nan)
    width_pct = band_range / close_series * 100.0

    prefix = f"bb_{timeframe_label}"
    prepared[f"{prefix}_mid"] = mid
    prepared[f"{prefix}_upper"] = upper
    prepared[f"{prefix}_lower"] = lower
    prepared[f"{prefix}_band_width_pct"] = width_pct
    prepared[f"{prefix}_price_relative_to_mid_pct"] = (close_series - mid) / close_series * 100.0
    prepared[f"{prefix}_distance_to_upper_band_pct"] = (upper - close_series) / close_series * 100.0
    prepared[f"{prefix}_distance_to_lower_band_pct"] = (close_series - lower) / close_series * 100.0
    prepared[f"{prefix}_band_position"] = (close_series - lower) / band_range
    prepared[f"{prefix}_close_above_upper_flag"] = close_series > upper
    prepared[f"{prefix}_close_below_lower_flag"] = close_series < lower
    prepared[f"{prefix}_wick_above_upper_flag"] = high_series > upper
    prepared[f"{prefix}_wick_below_lower_flag"] = low_series < lower
    prepared[f"{prefix}_upper_excursion_pct"] = np.maximum(high_series - upper, 0) / close_series * 100.0
    prepared[f"{prefix}_lower_excursion_pct"] = np.maximum(lower - low_series, 0) / close_series * 100.0
    prepared[f"{prefix}_band_expanding_flag"] = width_pct > width_pct.shift(1)
    prepared[f"{prefix}_band_expansion_rate"] = width_pct.diff()
    prepared[f"{prefix}_squeeze_flag"] = width_pct <= width_pct.rolling(40, min_periods=20).quantile(0.20)
    prepared[f"{prefix}_expansion_flag"] = width_pct >= width_pct.rolling(40, min_periods=20).quantile(0.80)
    prepared[f"{prefix}_close_back_inside_after_upper_touch_flag"] = (high_series > upper) & (close_series <= upper)
    prepared[f"{prefix}_close_back_inside_after_lower_touch_flag"] = (low_series < lower) & (close_series >= lower)
    prepared[f"{prefix}_close_below_mid_after_upper_touch_flag"] = (high_series > upper) & (close_series < mid)
    prepared[f"{prefix}_close_above_mid_after_lower_touch_flag"] = (low_series < lower) & (close_series > mid)
    prepared[f"{prefix}_one_bar_return_inside_from_upper_flag"] = (
        ((high_series.shift(1) > upper.shift(1)) | (close_series.shift(1) > upper.shift(1)))
        & (close_series <= upper)
    )
    prepared[f"{prefix}_one_bar_return_inside_from_lower_flag"] = (
        ((low_series.shift(1) < lower.shift(1)) | (close_series.shift(1) < lower.shift(1)))
        & (close_series >= lower)
    )
    prepared[f"{prefix}_failed_continuation_outside_upper_flag"] = (close_series.shift(1) > upper.shift(1)) & (close_series <= upper)
    prepared[f"{prefix}_failed_continuation_outside_lower_flag"] = (close_series.shift(1) < lower.shift(1)) & (close_series >= lower)
    prepared[f"{prefix}_no_follow_through_after_upper_touch_flag"] = (high_series.shift(1) > upper.shift(1)) & (close_series <= close_series.shift(1))
    prepared[f"{prefix}_no_follow_through_after_lower_touch_flag"] = (low_series.shift(1) < lower.shift(1)) & (close_series >= close_series.shift(1))
    prepared[f"{prefix}_weak_close_after_upper_touch_flag"] = (high_series > upper) & (close_series <= (open_series + close_series) / 2.0)
    prepared[f"{prefix}_stabilization_after_lower_touch_flag"] = (low_series < lower) & (close_series >= (open_series + close_series) / 2.0)

    boolean_columns = [column for column in prepared.columns if column.endswith("_flag")]
    for column in boolean_columns:
        prepared[column] = prepared[column].fillna(False).astype(bool)
    return prepared


def attach_bollinger_features(
    entries: pd.DataFrame,
    minute_timing: pd.DataFrame,
    timing_5m: pd.DataFrame,
    bb_5m: pd.DataFrame,
    bb_15m: pd.DataFrame,
    time_column: str,
) -> pd.DataFrame:
    merged = attach_timing_features(entries.copy(), minute_timing, timing_5m, time_column=time_column)

    bb_5m_lookup = bb_5m.rename(columns={"close_time": "bb_feature_time_5m"})
    merged = pd.merge_asof(
        merged.sort_values(time_column),
        bb_5m_lookup.sort_values("bb_feature_time_5m"),
        left_on=time_column,
        right_on="bb_feature_time_5m",
        direction="backward",
    ).drop(columns=["bb_feature_time_5m"])

    bb_15m_lookup = bb_15m.rename(columns={"close_time": "bb_feature_time_15m"})
    merged = pd.merge_asof(
        merged.sort_values(time_column),
        bb_15m_lookup.sort_values("bb_feature_time_15m"),
        left_on=time_column,
        right_on="bb_feature_time_15m",
        direction="backward",
    ).drop(columns=["bb_feature_time_15m"])
    return merged


def analyze_bollinger_family(
    family_label: str,
    cluster_id: int,
    side: str,
    real_reference: pd.DataFrame,
    broad_pool: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
    minute_timing: pd.DataFrame,
    timing_5m: pd.DataFrame,
    bb_5m: pd.DataFrame,
    bb_15m: pd.DataFrame,
) -> dict:
    if real_reference.empty or broad_pool.empty:
        return {
            "family_label": family_label,
            "side": side,
            "hypotheses": pd.DataFrame(),
            "best": pd.Series(dtype=object),
            "real_summary": {},
            "broad_summary": {},
            "tightened_summary": {},
        }

    reference = attach_bollinger_features(real_reference, minute_timing, timing_5m, bb_5m, bb_15m, time_column="entry_time")
    broad = attach_bollinger_features(broad_pool, minute_timing, timing_5m, bb_5m, bb_15m, time_column="entry_time")
    tightened = build_generic_tightened_pool(broad, tightened_summary_df, cluster_id)

    real_summary = summarize_signal_like_profiles(reference)
    broad_summary = summarize_signal_like_profiles(broad)
    tightened_summary = summarize_signal_like_profiles(tightened) if not tightened.empty else {}

    hypotheses = build_bollinger_hypotheses(side)
    rows = []
    broad_similarity = compare_profiles(real_summary, broad_summary)
    tightened_count = len(tightened)
    for hypothesis in hypotheses:
        real_hits = apply_conditions(reference, hypothesis["conditions"])
        broad_hits = apply_conditions(broad, hypothesis["conditions"])
        tightened_hits = apply_conditions(tightened, hypothesis["conditions"]) if not tightened.empty else pd.DataFrame()
        if broad_hits.empty:
            continue

        filtered_summary = summarize_signal_like_profiles(broad_hits)
        filtered_similarity = compare_profiles(real_summary, filtered_summary)
        frequency_real = safe_divide(len(real_hits), len(reference))
        frequency_broad = safe_divide(len(broad_hits), len(broad))
        frequency_tightened = safe_divide(len(tightened_hits), len(tightened)) if not tightened.empty else np.nan
        lift = safe_divide(frequency_real, frequency_broad)
        precision_proxy = safe_divide(frequency_real, frequency_real + frequency_broad)
        similarity_improvement = safe_number(filtered_similarity["score"]) - safe_number(broad_similarity["score"])
        mae_improvement = safe_number(broad_summary.get("median_MAE_pct")) - safe_number(filtered_summary.get("median_MAE_pct"))
        pos5_improvement = safe_number(filtered_summary.get("pct_positive_after_5m")) - safe_number(broad_summary.get("pct_positive_after_5m"))
        count_score = min(safe_divide(len(broad_hits), max(min(20, len(broad)), 1)), 1.0)
        ranking_score = (
            max(safe_number(lift) - 1.0, 0.0) * 0.30
            + max(similarity_improvement, 0.0) * 0.30
            + max(pos5_improvement, 0.0) * 0.15
            + max(mae_improvement, 0.0) * 0.10
            + max(frequency_real - safe_number(frequency_tightened), 0.0) * 0.10
            + count_score * 0.05
        )
        rows.append(
            {
                "family_label": family_label,
                "cluster_id": cluster_id,
                "side": side,
                "hypothesis_label": hypothesis["label"],
                "definition_text": hypothesis["text"],
                "timeframe": hypothesis["timeframe"],
                "logic_type": hypothesis["logic_type"],
                "frequency_in_real_trades": frequency_real,
                "frequency_in_candidate_pool": frequency_broad,
                "frequency_in_tightened_pool": frequency_tightened,
                "lift": lift,
                "precision_proxy_balanced": precision_proxy,
                "recall_proxy": frequency_real,
                "signal_count_after_filter": int(len(broad_hits)),
                "tightened_signal_count_after_filter": int(len(tightened_hits)) if not tightened.empty else 0,
                "reduction_vs_candidate_pool": 1 - safe_divide(len(broad_hits), len(broad)),
                "reduction_vs_tightened_pool": 1 - safe_divide(len(tightened_hits), tightened_count) if tightened_count else np.nan,
                "pct_positive_1m": filtered_summary.get("pct_positive_after_1m"),
                "pct_positive_5m": filtered_summary.get("pct_positive_after_5m"),
                "pct_positive_15m": filtered_summary.get("pct_positive_after_15m"),
                "median_MAE": filtered_summary.get("median_MAE_pct"),
                "median_MFE": filtered_summary.get("median_MFE_pct"),
                "median_return_1h": filtered_summary.get("median_return_after_60m_pct"),
                "median_return_4h": filtered_summary.get("median_return_after_240m_pct"),
                "similarity_to_real_cluster_profile": filtered_similarity["score"],
                "similarity_label": filtered_similarity["label"],
                "similarity_improvement_vs_candidate_pool": similarity_improvement,
                "ranking_score": ranking_score,
            }
        )

    hypotheses_df = pd.DataFrame(rows)
    if not hypotheses_df.empty:
        hypotheses_df = hypotheses_df.sort_values(
            ["ranking_score", "similarity_improvement_vs_candidate_pool", "lift", "signal_count_after_filter"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        best = hypotheses_df.iloc[0]
    else:
        best = pd.Series(dtype=object)

    return {
        "family_label": family_label,
        "side": side,
        "hypotheses": hypotheses_df,
        "best": best,
        "real_summary": real_summary,
        "broad_summary": broad_summary,
        "tightened_summary": tightened_summary,
    }


def build_generic_tightened_pool(
    broad_pool: pd.DataFrame,
    tightened_summary_df: pd.DataFrame,
    cluster_id: int,
) -> pd.DataFrame:
    if broad_pool.empty or tightened_summary_df.empty:
        return pd.DataFrame()
    row = tightened_summary_df.loc[tightened_summary_df["cluster_id"] == cluster_id]
    if row.empty:
        return pd.DataFrame()
    timing_rule_text = str(row.iloc[0]["timing_rule_text"])
    conditions = parse_timing_rule_text(timing_rule_text)
    if not conditions:
        return pd.DataFrame()
    return apply_conditions(broad_pool, conditions)


def build_bollinger_hypotheses(side: str) -> list[dict]:
    if side == "Short":
        return [
            {
                "label": "Near upper band 5m",
                "text": "BB position 5m >= 0.85",
                "timeframe": "5m",
                "logic_type": "stretch",
                "conditions": [condition_ge("bb_5m_band_position", 0.85)],
            },
            {
                "label": "Upper band touch + close back inside 5m",
                "text": "Wick above upper band 5m AND close back inside 5m",
                "timeframe": "5m",
                "logic_type": "return_inside",
                "conditions": [condition_eq("bb_5m_close_back_inside_after_upper_touch_flag", True)],
            },
            {
                "label": "Upper band touch then one-bar return inside 5m",
                "text": "Previous upper-band touch 5m AND current close inside bands 5m",
                "timeframe": "5m",
                "logic_type": "return_inside",
                "conditions": [condition_eq("bb_5m_one_bar_return_inside_from_upper_flag", True)],
            },
            {
                "label": "Upper band touch + close below mid-band 5m",
                "text": "Upper-band touch 5m AND close below mid-band 5m",
                "timeframe": "5m",
                "logic_type": "fade",
                "conditions": [condition_eq("bb_5m_close_below_mid_after_upper_touch_flag", True)],
            },
            {
                "label": "Failed continuation outside upper band 5m",
                "text": "Previous close above upper band 5m AND current close back inside",
                "timeframe": "5m",
                "logic_type": "failed_continuation",
                "conditions": [condition_eq("bb_5m_failed_continuation_outside_upper_flag", True)],
            },
            {
                "label": "Near upper band 15m",
                "text": "BB position 15m >= 0.85",
                "timeframe": "15m",
                "logic_type": "stretch",
                "conditions": [condition_ge("bb_15m_band_position", 0.85)],
            },
        ]
    return [
        {
            "label": "Near lower band 5m",
            "text": "BB position 5m <= 0.15",
            "timeframe": "5m",
            "logic_type": "stretch",
            "conditions": [condition_le("bb_5m_band_position", 0.15)],
        },
        {
            "label": "Lower band touch + close back inside 5m",
            "text": "Wick below lower band 5m AND close back inside 5m",
            "timeframe": "5m",
            "logic_type": "return_inside",
            "conditions": [condition_eq("bb_5m_close_back_inside_after_lower_touch_flag", True)],
        },
        {
            "label": "Lower band touch then one-bar return inside 5m",
            "text": "Previous lower-band touch 5m AND current close inside bands 5m",
            "timeframe": "5m",
            "logic_type": "return_inside",
            "conditions": [condition_eq("bb_5m_one_bar_return_inside_from_lower_flag", True)],
        },
        {
            "label": "Lower band touch + close above mid-band 5m",
            "text": "Lower-band touch 5m AND close above mid-band 5m",
            "timeframe": "5m",
            "logic_type": "reversion",
            "conditions": [condition_eq("bb_5m_close_above_mid_after_lower_touch_flag", True)],
        },
        {
            "label": "Failed continuation outside lower band 5m",
            "text": "Previous close below lower band 5m AND current close back inside",
            "timeframe": "5m",
            "logic_type": "failed_continuation",
            "conditions": [condition_eq("bb_5m_failed_continuation_outside_lower_flag", True)],
        },
        {
            "label": "Near lower band 15m",
            "text": "BB position 15m <= 0.15",
            "timeframe": "15m",
            "logic_type": "stretch",
            "conditions": [condition_le("bb_15m_band_position", 0.15)],
        },
    ]


def build_bollinger_summary(short_result: dict, long_result: dict) -> pd.DataFrame:
    rows = []
    for result in [short_result, long_result]:
        best = result.get("best", pd.Series(dtype=object))
        if best.empty:
            rows.append(
                {
                    "family_label": result["family_label"],
                    "side": result["side"],
                    "compatible_with_bollinger_framework": False,
                    "best_hypothesis_label": "n/a",
                    "definition_text": "n/a",
                    "lift": np.nan,
                    "similarity_to_real_cluster_profile": np.nan,
                    "similarity_improvement_vs_candidate_pool": np.nan,
                    "signal_count_after_filter": 0,
                    "confidence_level": "low",
                }
            )
            continue
        compatible = (
            safe_number(best.get("lift")) >= 1.15
            and safe_number(best.get("similarity_improvement_vs_candidate_pool")) >= 0.02
        )
        rows.append(
            {
                "family_label": result["family_label"],
                "side": result["side"],
                "compatible_with_bollinger_framework": compatible,
                "best_hypothesis_label": best["hypothesis_label"],
                "definition_text": best["definition_text"],
                "lift": best["lift"],
                "similarity_to_real_cluster_profile": best["similarity_to_real_cluster_profile"],
                "similarity_improvement_vs_candidate_pool": best["similarity_improvement_vs_candidate_pool"],
                "signal_count_after_filter": best["signal_count_after_filter"],
                "confidence_level": determine_bollinger_family_confidence(best),
            }
        )
    return pd.DataFrame(rows)


def build_bollinger_section(summary_df: pd.DataFrame, short_result: dict, long_result: dict) -> str:
    if summary_df.empty:
        return (
            "BOLLINGER COMPATIBILITY TEST\n"
            "----------------------------\n"
            "Bollinger compatibility could not be assessed.\n"
        )

    short_best = short_result.get("best", pd.Series(dtype=object))
    long_best = long_result.get("best", pd.Series(dtype=object))
    short_row = summary_df.loc[summary_df["family_label"] == "short_family"].iloc[0]
    long_row = summary_df.loc[summary_df["family_label"] == "long_family"].iloc[0]

    overall_compatible = bool(short_row["compatible_with_bollinger_framework"] or long_row["compatible_with_bollinger_framework"])
    overall_confidence = determine_bollinger_overall_confidence(summary_df)
    verdict = (
        "Behavior is compatible with a Bollinger-type stretch/reversion framework as a useful visual description, but not as proof of the trader's exact indicator logic."
        if overall_compatible
        else "Behavior is not meaningfully improved by Bollinger framing; the match looks weak or mostly coincidental."
    )

    report = f"""
BOLLINGER COMPATIBILITY TEST
----------------------------
This section checks whether the ETH archetypes are meaningfully compatible with a Bollinger-type stretch / reversion / return-inside-bands framework on standard 20/2 bands, rather than trying to claim indicator-level reconstruction.

Short family:
- Best short hypothesis: {short_row['best_hypothesis_label']}.
- Definition: {short_row['definition_text']}.
- Frequency in real Cluster 1 shorts vs candidate pool: {pct(short_best.get('frequency_in_real_trades')) if not short_best.empty else 'n/a'} vs {pct(short_best.get('frequency_in_candidate_pool')) if not short_best.empty else 'n/a'}.
- Lift: {fmt_number(short_row['lift'])}. Similarity to real short profile: {fmt_number(short_row['similarity_to_real_cluster_profile'])}. Compatibility verdict: {"compatible" if bool(short_row['compatible_with_bollinger_framework']) else "weak"}.

Long family:
- Best long hypothesis: {long_row['best_hypothesis_label']}.
- Definition: {long_row['definition_text']}.
- Frequency in real Cluster 2 longs vs candidate pool: {pct(long_best.get('frequency_in_real_trades')) if not long_best.empty else 'n/a'} vs {pct(long_best.get('frequency_in_candidate_pool')) if not long_best.empty else 'n/a'}.
- Lift: {fmt_number(long_row['lift'])}. Similarity to real long profile: {fmt_number(long_row['similarity_to_real_cluster_profile'])}. Compatibility verdict: {"compatible" if bool(long_row['compatible_with_bollinger_framework']) else "weak"}.

Overall read:
- Bollinger framing {"does" if overall_compatible else "does not"} improve the explanation of the ETH archetypes in a meaningful way.
- Confidence level: {overall_confidence}.

Final verdict:
- {verdict}
""".strip()
    return report + "\n"


def determine_bollinger_family_confidence(best_row: pd.Series) -> str:
    if best_row.empty:
        return "low"
    if (
        safe_number(best_row.get("lift")) >= 1.50
        and safe_number(best_row.get("similarity_improvement_vs_candidate_pool")) >= 0.05
        and safe_number(best_row.get("signal_count_after_filter")) >= 16
    ):
        return "high"
    if (
        safe_number(best_row.get("lift")) >= 1.15
        and safe_number(best_row.get("similarity_improvement_vs_candidate_pool")) >= 0.02
        and safe_number(best_row.get("signal_count_after_filter")) >= 10
    ):
        return "medium"
    return "low"


def determine_bollinger_overall_confidence(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "low"
    levels = summary_df["confidence_level"].astype(str).tolist()
    if levels.count("high") >= 1 and levels.count("medium") >= 1:
        return "high"
    if "medium" in levels or "high" in levels:
        return "medium"
    return "low"


def create_eth_bollinger_compatibility_plots(
    short_hypotheses: pd.DataFrame,
    long_hypotheses: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not short_hypotheses.empty:
        short_path = plots_dir / "eth_bollinger_short_comparison.png"
        fig_short = plot_bollinger_family(short_hypotheses, "Short")
        fig_short.savefig(short_path, dpi=150, bbox_inches="tight")
        plt.close(fig_short)
        logger.info("Saved plot: %s", short_path.name)

    if not long_hypotheses.empty:
        long_path = plots_dir / "eth_bollinger_long_comparison.png"
        fig_long = plot_bollinger_family(long_hypotheses, "Long")
        fig_long.savefig(long_path, dpi=150, bbox_inches="tight")
        plt.close(fig_long)
        logger.info("Saved plot: %s", long_path.name)


def plot_bollinger_family(hypotheses_df: pd.DataFrame, side_label: str) -> plt.Figure:
    plot_df = hypotheses_df.head(5).copy().sort_values("ranking_score")
    labels = plot_df["hypothesis_label"].astype(str).tolist()
    y = np.arange(len(plot_df))
    height = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].barh(y - height, plot_df["frequency_in_real_trades"], height=height, label="Real trades", color="#4c78a8")
    axes[0].barh(y, plot_df["frequency_in_candidate_pool"], height=height, label="Broad candidates", color="#9ecae9")
    axes[0].barh(y + height, plot_df["frequency_in_tightened_pool"].fillna(0), height=height, label="Tightened candidates", color="#f58518")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title(f"{side_label} Bollinger hypothesis frequency")
    axes[0].set_xlabel("Frequency")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend()

    axes[1].scatter(plot_df["signal_count_after_filter"], plot_df["similarity_to_real_cluster_profile"], s=90, color="#54a24b")
    for _, row in plot_df.iterrows():
        axes[1].annotate(
            row["hypothesis_label"],
            (row["signal_count_after_filter"], row["similarity_to_real_cluster_profile"]),
            fontsize=8,
        )
    axes[1].set_title(f"{side_label} Bollinger filtered count vs similarity")
    axes[1].set_xlabel("Signal count after filter")
    axes[1].set_ylabel("Similarity to real cluster profile")
    axes[1].grid(alpha=0.25)
    return fig


def condition_ge(feature_key: str, threshold: float) -> dict:
    return {"feature_key": feature_key, "rule_type": "ge", "threshold": threshold}


def condition_le(feature_key: str, threshold: float) -> dict:
    return {"feature_key": feature_key, "rule_type": "le", "threshold": threshold}


def condition_eq(feature_key: str, value: bool) -> dict:
    return {"feature_key": feature_key, "rule_type": "equals", "value": value}


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0
