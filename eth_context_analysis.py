from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME, resolve_output_path
from utils import fmt_int, fmt_number, pct


ETH_SYMBOL = "ETHUSDT"

NUMERIC_FEATURE_SPECS = [
    {"key": "rsi14_15m", "label": "RSI 15m", "family": "momentum"},
    {"key": "rsi14_5m", "label": "RSI 5m", "family": "momentum"},
    {"key": "stoch_rsi14_15m", "label": "Stoch RSI 15m", "family": "momentum"},
    {"key": "return_pre_1m_pct", "label": "Pre-entry return 1m %", "family": "momentum"},
    {"key": "return_pre_5m_pct", "label": "Pre-entry return 5m %", "family": "momentum"},
    {"key": "return_pre_15m_pct", "label": "Pre-entry return 15m %", "family": "momentum"},
    {"key": "return_pre_60m_pct", "label": "Pre-entry return 60m %", "family": "momentum"},
    {"key": "distance_to_ema20_pct_5m", "label": "Distance to EMA20 5m %", "family": "trend_extension"},
    {"key": "distance_to_ema20_pct_15m", "label": "Distance to EMA20 15m %", "family": "trend_extension"},
    {"key": "distance_to_ema50_pct_15m", "label": "Distance to EMA50 15m %", "family": "trend_extension"},
    {"key": "distance_to_ema20_pct_1h", "label": "Distance to EMA20 1h %", "family": "trend_extension"},
    {"key": "abs_distance_to_ema20_pct_5m", "label": "Abs distance to EMA20 5m %", "family": "trend_extension"},
    {"key": "abs_distance_to_ema20_pct_15m", "label": "Abs distance to EMA20 15m %", "family": "trend_extension"},
    {"key": "abs_distance_to_ema50_pct_15m", "label": "Abs distance to EMA50 15m %", "family": "trend_extension"},
    {"key": "abs_distance_to_ema20_pct_1h", "label": "Abs distance to EMA20 1h %", "family": "trend_extension"},
    {"key": "ema20_slope_pct_15m", "label": "EMA20 slope 15m %", "family": "trend_extension"},
    {"key": "ema20_slope_pct_1h", "label": "EMA20 slope 1h %", "family": "trend_extension"},
    {"key": "atr14_pct_5m", "label": "ATR % 5m", "family": "volatility"},
    {"key": "atr14_pct_15m", "label": "ATR % 15m", "family": "volatility"},
    {"key": "recent_range_15m_pct", "label": "Recent range 15m %", "family": "volatility"},
    {"key": "recent_range_1h_pct", "label": "Recent range 1h %", "family": "volatility"},
    {"key": "relative_volume_1m", "label": "Relative volume 1m", "family": "volume"},
    {"key": "relative_volume_5m", "label": "Relative volume 5m", "family": "volume"},
    {"key": "distance_to_last_1h_high_pct", "label": "Distance to last 1h high %", "family": "structure"},
    {"key": "distance_to_last_1h_low_pct", "label": "Distance to last 1h low %", "family": "structure"},
    {"key": "distance_to_last_4h_high_pct", "label": "Distance to last 4h high %", "family": "structure"},
    {"key": "distance_to_last_4h_low_pct", "label": "Distance to last 4h low %", "family": "structure"},
]

BOOLEAN_FEATURE_SPECS = [
    {"key": "pullback_before_entry_flag", "label": "Pullback flag", "family": "structure"},
    {"key": "impulse_before_entry_flag", "label": "Impulse flag", "family": "structure"},
    {"key": "breakout_before_entry_flag", "label": "Breakout flag", "family": "structure"},
    {"key": "pullback_or_reversal_flag", "label": "Pullback/Reversal context share", "family": "structure"},
    {"key": "breakout_or_impulse_flag", "label": "Breakout/Impulse context share", "family": "structure"},
    {"key": "structure_near_high_flag", "label": "Near-high structure share", "family": "structure"},
    {"key": "structure_near_low_flag", "label": "Near-low structure share", "family": "structure"},
    {"key": "structure_mid_range_flag", "label": "Mid-range structure share", "family": "structure"},
    {"key": "context_pullback_flag", "label": "Context label: pullback", "family": "structure"},
    {"key": "context_reversal_like_flag", "label": "Context label: reversal_like", "family": "structure"},
    {"key": "context_breakout_like_flag", "label": "Context label: breakout/continuation", "family": "structure"},
    {"key": "ema_alignment_15m_bullish_flag", "label": "15m bullish EMA alignment share", "family": "trend_extension"},
    {"key": "ema_alignment_15m_bearish_flag", "label": "15m bearish EMA alignment share", "family": "trend_extension"},
    {"key": "ema_alignment_1h_bullish_flag", "label": "1h bullish EMA alignment share", "family": "trend_extension"},
    {"key": "ema_alignment_1h_bearish_flag", "label": "1h bearish EMA alignment share", "family": "trend_extension"},
    {"key": "alignment_with_side_15m_flag", "label": "15m alignment with trade side", "family": "trend_extension"},
    {"key": "alignment_with_side_1h_flag", "label": "1h alignment with trade side", "family": "trend_extension"},
]

ETH_CONTEXT_PLOT_FEATURES = [
    ("rsi14_15m", "RSI 15m"),
    ("return_pre_15m_pct", "Pre-entry Return 15m %"),
    ("relative_volume_5m", "Relative Volume 5m"),
    ("abs_distance_to_ema20_pct_15m", "Abs Distance To EMA20 15m %"),
    ("atr14_pct_15m", "ATR % 15m"),
    ("recent_range_1h_pct", "Recent Range 1h %"),
    ("distance_to_last_4h_high_pct", "Distance To Last 4h High %"),
    ("distance_to_last_4h_low_pct", "Distance To Last 4h Low %"),
]

SIDE_COMPARE_PLOT_FEATURES = [
    ("rsi14_15m", "RSI 15m"),
    ("return_pre_15m_pct", "Pre-entry Return 15m %"),
    ("relative_volume_5m", "Relative Volume 5m"),
]

OVERALL_HYPOTHESIS_FEATURES = {
    "rsi14_15m": "lower",
    "rsi14_5m": "lower",
    "stoch_rsi14_15m": "lower",
    "return_pre_5m_pct": "lower",
    "return_pre_15m_pct": "lower",
    "relative_volume_1m": "lower",
    "relative_volume_5m": "lower",
    "abs_distance_to_ema20_pct_5m": "lower",
    "abs_distance_to_ema20_pct_15m": "lower",
    "abs_distance_to_ema50_pct_15m": "lower",
    "abs_distance_to_ema20_pct_1h": "lower",
    "pullback_or_reversal_flag": "higher",
    "breakout_or_impulse_flag": "lower",
}

LONG_HYPOTHESIS_FEATURES = {
    "rsi14_15m": "lower",
    "rsi14_5m": "lower",
    "stoch_rsi14_15m": "lower",
    "return_pre_5m_pct": "lower",
    "return_pre_15m_pct": "lower",
    "return_pre_60m_pct": "lower",
    "relative_volume_1m": "lower",
    "relative_volume_5m": "lower",
    "abs_distance_to_ema20_pct_15m": "lower",
    "abs_distance_to_ema20_pct_1h": "lower",
    "pullback_or_reversal_flag": "higher",
    "breakout_or_impulse_flag": "lower",
}

SHORT_HYPOTHESIS_FEATURES = {
    "return_pre_5m_pct": "higher",
    "return_pre_15m_pct": "higher",
    "return_pre_60m_pct": "higher",
    "relative_volume_1m": "lower",
    "relative_volume_5m": "lower",
    "abs_distance_to_ema20_pct_15m": "lower",
    "abs_distance_to_ema20_pct_1h": "lower",
    "pullback_or_reversal_flag": "higher",
    "breakout_or_impulse_flag": "lower",
    "structure_near_high_flag": "higher",
    "structure_near_low_flag": "lower",
}


def run_eth_context_analysis(context_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> dict:
    eth_df = prepare_eth_context_frame(context_df)

    empty_result = {
        "eth_context_frame": eth_df,
        "eth_entry_context_comparison": pd.DataFrame(),
        "eth_long_context_comparison": pd.DataFrame(),
        "eth_short_context_comparison": pd.DataFrame(),
        "summary_text": (
            "ETH CONTEXT ANALYSIS\n====================\n"
            "No ETHUSDT trades were available for ETH-only context analysis.\n"
        ),
    }

    if eth_df.empty:
        resolve_output_path(output_dir, "eth_context_summary").write_text(empty_result["summary_text"], encoding="utf-8")
        return empty_result

    overall_comparison = build_context_comparison(eth_df, "ETHUSDT_all")
    long_comparison = build_context_comparison(eth_df.loc[eth_df["side"] == "Long"], "ETHUSDT_Long")
    short_comparison = build_context_comparison(eth_df.loc[eth_df["side"] == "Short"], "ETHUSDT_Short")
    summary_text = build_eth_context_summary_text(eth_df, overall_comparison, long_comparison, short_comparison)

    resolve_output_path(output_dir, "eth_context_summary").write_text(summary_text, encoding="utf-8")
    return {
        "eth_context_frame": eth_df,
        "eth_entry_context_comparison": overall_comparison,
        "eth_long_context_comparison": long_comparison,
        "eth_short_context_comparison": short_comparison,
        "summary_text": summary_text,
    }


def prepare_eth_context_frame(context_df: pd.DataFrame) -> pd.DataFrame:
    if context_df.empty:
        return context_df.copy()

    eth_df = context_df.loc[context_df["symbol"] == ETH_SYMBOL].copy()
    if eth_df.empty:
        return eth_df

    for column in [
        "distance_to_ema20_pct_5m",
        "distance_to_ema20_pct_15m",
        "distance_to_ema50_pct_15m",
        "distance_to_ema20_pct_1h",
    ]:
        eth_df[f"abs_{column}"] = eth_df[column].abs() if column in eth_df.columns else np.nan

    eth_df["pullback_or_reversal_flag"] = eth_df["context_style_label"].isin(["pullback", "reversal_like"])
    eth_df["breakout_or_impulse_flag"] = eth_df["context_style_label"].isin(["breakout_continuation", "continuation_like"])
    eth_df["structure_near_high_flag"] = eth_df["structure_regime"] == "near_high"
    eth_df["structure_near_low_flag"] = eth_df["structure_regime"] == "near_low"
    eth_df["structure_mid_range_flag"] = eth_df["structure_regime"] == "mid_range"
    eth_df["context_pullback_flag"] = eth_df["context_style_label"] == "pullback"
    eth_df["context_reversal_like_flag"] = eth_df["context_style_label"] == "reversal_like"
    eth_df["context_breakout_like_flag"] = eth_df["context_style_label"].isin(["breakout_continuation", "continuation_like"])
    eth_df["ema_alignment_15m_bullish_flag"] = eth_df["ema_alignment_15m"] == "bullish"
    eth_df["ema_alignment_15m_bearish_flag"] = eth_df["ema_alignment_15m"] == "bearish"
    eth_df["ema_alignment_1h_bullish_flag"] = eth_df["ema_alignment_1h"] == "bullish"
    eth_df["ema_alignment_1h_bearish_flag"] = eth_df["ema_alignment_1h"] == "bearish"
    eth_df["alignment_with_side_15m_flag"] = np.where(
        eth_df["side"] == "Long",
        eth_df["ema_alignment_15m"] == "bullish",
        np.where(eth_df["side"] == "Short", eth_df["ema_alignment_15m"] == "bearish", False),
    )
    eth_df["alignment_with_side_1h_flag"] = np.where(
        eth_df["side"] == "Long",
        eth_df["ema_alignment_1h"] == "bullish",
        np.where(eth_df["side"] == "Short", eth_df["ema_alignment_1h"] == "bearish", False),
    )
    return eth_df


def build_context_comparison(frame: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    strong = frame.loc[frame["entry_strength_group"] == "strong_entry"]
    medium = frame.loc[frame["entry_strength_group"] == "medium_entry"]
    weak = frame.loc[frame["entry_strength_group"] == "weak_entry"]

    rows: list[dict] = []
    for spec in NUMERIC_FEATURE_SPECS:
        rows.append(compare_numeric_feature(strong, weak, len(frame), len(strong), len(medium), len(weak), subset_name, spec))
    for spec in BOOLEAN_FEATURE_SPECS:
        rows.append(compare_boolean_feature(strong, weak, len(frame), len(strong), len(medium), len(weak), subset_name, spec))
    return pd.DataFrame(rows)


def compare_numeric_feature(
    strong: pd.DataFrame,
    weak: pd.DataFrame,
    total_count: int,
    strong_count: int,
    medium_count: int,
    weak_count: int,
    subset_name: str,
    spec: dict,
) -> dict:
    key = spec["key"]
    strong_values = pd.to_numeric(strong.get(key, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    weak_values = pd.to_numeric(weak.get(key, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)

    strong_median = np.nanmedian(strong_values) if strong_values.size else np.nan
    weak_median = np.nanmedian(weak_values) if weak_values.size else np.nan
    strong_q25 = np.nanpercentile(strong_values, 25) if strong_values.size else np.nan
    strong_q75 = np.nanpercentile(strong_values, 75) if strong_values.size else np.nan
    weak_q25 = np.nanpercentile(weak_values, 25) if weak_values.size else np.nan
    weak_q75 = np.nanpercentile(weak_values, 75) if weak_values.size else np.nan

    mann_whitney = mann_whitney_summary(strong_values, weak_values)
    overlap = percentile_overlap_pct(strong_values, weak_values)
    difference = strong_median - weak_median if not (pd.isna(strong_median) or pd.isna(weak_median)) else np.nan

    return {
        "subset": subset_name,
        "family": spec["family"],
        "feature_key": key,
        "feature_label": spec["label"],
        "feature_type": "numeric",
        "total_trades": total_count,
        "strong_count": strong_count,
        "medium_count": medium_count,
        "weak_count": weak_count,
        "strong_n": int(strong_values.size),
        "weak_n": int(weak_values.size),
        "median_strong": strong_median,
        "median_weak": weak_median,
        "share_strong": np.nan,
        "share_weak": np.nan,
        "difference": difference,
        "effect_direction": direction_label(difference),
        "p_value": mann_whitney["p_value"],
        "effect_size": mann_whitney["rank_biserial"],
        "percentile_overlap_pct": overlap,
        "strong_q25": strong_q25,
        "strong_q75": strong_q75,
        "weak_q25": weak_q25,
        "weak_q75": weak_q75,
        "signal_strength": signal_strength_label(mann_whitney["rank_biserial"], mann_whitney["p_value"], overlap),
    }


def compare_boolean_feature(
    strong: pd.DataFrame,
    weak: pd.DataFrame,
    total_count: int,
    strong_count: int,
    medium_count: int,
    weak_count: int,
    subset_name: str,
    spec: dict,
) -> dict:
    key = spec["key"]
    strong_values = strong.get(key, pd.Series(dtype="boolean"))
    weak_values = weak.get(key, pd.Series(dtype="boolean"))

    strong_clean = pd.Series(strong_values).dropna().astype(bool)
    weak_clean = pd.Series(weak_values).dropna().astype(bool)
    strong_share = float(strong_clean.mean()) if not strong_clean.empty else np.nan
    weak_share = float(weak_clean.mean()) if not weak_clean.empty else np.nan
    difference = strong_share - weak_share if not (pd.isna(strong_share) or pd.isna(weak_share)) else np.nan
    p_value = proportion_p_value(int(strong_clean.sum()), int(len(strong_clean)), int(weak_clean.sum()), int(len(weak_clean)))
    overlap = 1.0 - abs(difference) if not pd.isna(difference) else np.nan

    return {
        "subset": subset_name,
        "family": spec["family"],
        "feature_key": key,
        "feature_label": spec["label"],
        "feature_type": "boolean_share",
        "total_trades": total_count,
        "strong_count": strong_count,
        "medium_count": medium_count,
        "weak_count": weak_count,
        "strong_n": int(len(strong_clean)),
        "weak_n": int(len(weak_clean)),
        "median_strong": np.nan,
        "median_weak": np.nan,
        "share_strong": strong_share,
        "share_weak": weak_share,
        "difference": difference,
        "effect_direction": direction_label(difference),
        "p_value": p_value,
        "effect_size": difference,
        "percentile_overlap_pct": overlap,
        "strong_q25": np.nan,
        "strong_q75": np.nan,
        "weak_q25": np.nan,
        "weak_q75": np.nan,
        "signal_strength": signal_strength_label(difference, p_value, overlap),
    }


def mann_whitney_summary(strong_values: np.ndarray, weak_values: np.ndarray) -> dict:
    if strong_values.size < 2 or weak_values.size < 2:
        return {"p_value": np.nan, "rank_biserial": np.nan}

    combined = np.concatenate([strong_values, weak_values])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    n1 = strong_values.size
    n2 = weak_values.size
    rank_sum_strong = ranks[:n1].sum()
    u1 = rank_sum_strong - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0

    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(tie_counts**3 - tie_counts)) if tie_counts.size else 0.0
    variance = n1 * n2 / 12.0 * ((n1 + n2 + 1) - safe_tie_adjustment(tie_term, n1 + n2))
    if variance <= 0:
        return {"p_value": np.nan, "rank_biserial": np.nan}

    z = (u1 - mean_u) / math.sqrt(variance)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    rank_biserial = (2 * u1) / (n1 * n2) - 1
    return {"p_value": p_value, "rank_biserial": rank_biserial}


def safe_tie_adjustment(tie_term: float, total_count: int) -> float:
    if total_count <= 1:
        return 0.0
    return tie_term / (total_count * (total_count - 1))


def proportion_p_value(success_1: int, total_1: int, success_2: int, total_2: int) -> float:
    if total_1 < 2 or total_2 < 2:
        return np.nan
    pooled = (success_1 + success_2) / (total_1 + total_2)
    variance = pooled * (1 - pooled) * ((1 / total_1) + (1 / total_2))
    if variance <= 0:
        return np.nan
    z = ((success_1 / total_1) - (success_2 / total_2)) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2))


def percentile_overlap_pct(strong_values: np.ndarray, weak_values: np.ndarray) -> float:
    if strong_values.size < 2 or weak_values.size < 2:
        return np.nan
    strong_low, strong_high = np.nanpercentile(strong_values, [10, 90])
    weak_low, weak_high = np.nanpercentile(weak_values, [10, 90])
    overlap_low = max(strong_low, weak_low)
    overlap_high = min(strong_high, weak_high)
    union_low = min(strong_low, weak_low)
    union_high = max(strong_high, weak_high)
    if union_high <= union_low:
        return np.nan
    overlap = max(0.0, overlap_high - overlap_low)
    return overlap / (union_high - union_low)


def direction_label(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value > 0:
        return "strong_higher"
    if value < 0:
        return "strong_lower"
    return "flat"


def signal_strength_label(effect_size: float, p_value: float, overlap: float) -> str:
    if pd.isna(effect_size):
        return "insufficient_data"
    strength = abs(effect_size)
    overlap_penalty = (1 - overlap) if pd.notna(overlap) else 0.0
    if strength >= 0.45 and (pd.isna(p_value) or p_value <= 0.10) and overlap_penalty >= 0.25:
        return "strong"
    if strength >= 0.25 and (pd.isna(p_value) or p_value <= 0.20) and overlap_penalty >= 0.10:
        return "medium"
    if strength >= 0.15:
        return "weak"
    return "minimal"


def build_eth_context_summary_text(
    eth_df: pd.DataFrame,
    overall_comparison: pd.DataFrame,
    long_comparison: pd.DataFrame,
    short_comparison: pd.DataFrame,
) -> str:
    counts_all = eth_df["entry_strength_group"].value_counts()
    long_df = eth_df.loc[eth_df["side"] == "Long"]
    short_df = eth_df.loc[eth_df["side"] == "Short"]
    counts_long = long_df["entry_strength_group"].value_counts()
    counts_short = short_df["entry_strength_group"].value_counts()

    overall_support = evaluate_hypothesis_support(overall_comparison, OVERALL_HYPOTHESIS_FEATURES)
    long_support = evaluate_hypothesis_support(long_comparison, LONG_HYPOTHESIS_FEATURES)
    short_support = evaluate_hypothesis_support(short_comparison, SHORT_HYPOTHESIS_FEATURES)

    overall_confidence = confidence_label(overall_comparison, overall_support)
    long_confidence = confidence_label(long_comparison, long_support)
    short_confidence = confidence_label(short_comparison, short_support)

    candidate_lines = build_candidate_filter_lines(overall_comparison, overall_support, overall_confidence)
    candidate_block = "\n".join(f"- {line}" for line in candidate_lines) if candidate_lines else "- No candidate context filter is justified yet from this ETH-only sample."

    summary = f"""
ETH CONTEXT ANALYSIS
====================

This section narrows the market-context analysis to ETHUSDT only. It does not infer a secret strategy. It only checks which pre-entry conditions appear more often before strong versus weak ETH entries.

ETH sample:
- Total ETH trades: {fmt_int(len(eth_df))}
- Strong ETH entries: {fmt_int(counts_all.get('strong_entry', 0))}
- Medium ETH entries: {fmt_int(counts_all.get('medium_entry', 0))}
- Weak ETH entries: {fmt_int(counts_all.get('weak_entry', 0))}
- ETH Long strong/weak counts: {fmt_int(counts_long.get('strong_entry', 0))} / {fmt_int(counts_long.get('weak_entry', 0))}
- ETH Short strong/weak counts: {fmt_int(counts_short.get('strong_entry', 0))} / {fmt_int(counts_short.get('weak_entry', 0))}

ETH strong vs weak:
- RSI 15m: {feature_sentence(overall_comparison, 'rsi14_15m')}
- Pre-entry return 15m: {feature_sentence(overall_comparison, 'return_pre_15m_pct')}
- Relative volume 5m: {feature_sentence(overall_comparison, 'relative_volume_5m')}
- Abs distance to EMA20 15m: {feature_sentence(overall_comparison, 'abs_distance_to_ema20_pct_15m')}
- Pullback/Reversal context share: {feature_sentence(overall_comparison, 'pullback_or_reversal_flag', is_share=True)}
- Breakout/Impulse context share: {feature_sentence(overall_comparison, 'breakout_or_impulse_flag', is_share=True)}
- Strongest separating ETH features: {top_feature_lines(overall_comparison, limit=4)}
- Overall ETH reading: {overall_support['verdict']}
- Overall confidence: {overall_confidence}

ETH Long strong vs weak:
- RSI 15m: {feature_sentence(long_comparison, 'rsi14_15m')}
- Pre-entry return 15m: {feature_sentence(long_comparison, 'return_pre_15m_pct')}
- Relative volume 5m: {feature_sentence(long_comparison, 'relative_volume_5m')}
- Abs distance to EMA20 15m: {feature_sentence(long_comparison, 'abs_distance_to_ema20_pct_15m')}
- Pullback/Reversal context share: {feature_sentence(long_comparison, 'pullback_or_reversal_flag', is_share=True)}
- Breakout/Impulse context share: {feature_sentence(long_comparison, 'breakout_or_impulse_flag', is_share=True)}
- Strongest separating ETH Long features: {top_feature_lines(long_comparison, limit=3)}
- ETH Long reading: {long_support['verdict']}
- ETH Long confidence: {long_confidence}

ETH Short strong vs weak:
- RSI 15m: {feature_sentence(short_comparison, 'rsi14_15m')}
- Pre-entry return 15m: {feature_sentence(short_comparison, 'return_pre_15m_pct')}
- Relative volume 5m: {feature_sentence(short_comparison, 'relative_volume_5m')}
- Abs distance to EMA20 15m: {feature_sentence(short_comparison, 'abs_distance_to_ema20_pct_15m')}
- Pullback/Reversal context share: {feature_sentence(short_comparison, 'pullback_or_reversal_flag', is_share=True)}
- Breakout/Impulse context share: {feature_sentence(short_comparison, 'breakout_or_impulse_flag', is_share=True)}
- Strongest separating ETH Short features: {top_feature_lines(short_comparison, limit=3)}
- ETH Short reading: {short_support['verdict']}
- ETH Short confidence: {short_confidence}

Interpretation:
- The main question was whether strong ETH entries tend to appear after mild cooling or pullback rather than after overheated impulse. {overall_support['answer']}
- Where the strongest separation shows up: {stronger_side_label(long_support, short_support)}.
- The most useful features in this sample are: {overall_support['most_useful_features']}.
- Side symmetry is limited: {side_difference_text(long_support, short_support)}
- Confidence is capped by sample size because weak ETH entries are only {fmt_int(counts_all.get('weak_entry', 0))} trades in total.

Candidate context filter
------------------------
Hypothesis for future testing only, not a strategy:
{candidate_block}
""".strip()
    return summary + "\n"


def evaluate_hypothesis_support(comparison_df: pd.DataFrame, hypothesis_map: dict[str, str]) -> dict:
    supportive: list[pd.Series] = []
    contradictory: list[pd.Series] = []
    descriptive: list[pd.Series] = []

    for feature_key, expected_direction in hypothesis_map.items():
        row = get_feature_row(comparison_df, feature_key)
        if row is None or row.get("signal_strength") in {"insufficient_data", "minimal"}:
            continue
        descriptive.append(row)
        observed_direction = "higher" if row.get("difference", np.nan) > 0 else "lower"
        if expected_direction == observed_direction:
            supportive.append(row)
        elif row.get("difference", np.nan) != 0:
            contradictory.append(row)

    descriptive_sorted = sort_rows_by_strength(descriptive)
    supportive_sorted = sort_rows_by_strength(supportive)
    contradictory_sorted = sort_rows_by_strength(contradictory)

    if not supportive_sorted:
        verdict = "The ETH-only sample does not give a confident cooling-vs-overheating separation yet."
        answer = "The cooling hypothesis is not strongly established from this sample alone."
    elif len(supportive_sorted) >= max(3, len(contradictory_sorted) + 1):
        verdict = "The ETH-only sample leans toward stronger entries appearing in calmer or slightly pulled-back contexts."
        answer = "The data leans toward strong ETH entries happening more often after some cooling, pullback, or lower-overheat conditions than after pure breakout chase."
    else:
        verdict = "The ETH-only sample is mixed: some cooling-style features separate strong entries, but the pattern is not clean."
        answer = "There are hints of a cooling/pullback effect, but the separation is only partial and should be treated as tentative."

    useful_features = ", ".join(row["feature_label"] for row in descriptive_sorted[:4]) if descriptive_sorted else "no feature stands out cleanly"
    return {
        "supportive": supportive_sorted,
        "contradictory": contradictory_sorted,
        "descriptive": descriptive_sorted,
        "verdict": verdict,
        "answer": answer,
        "most_useful_features": useful_features,
    }


def confidence_label(comparison_df: pd.DataFrame, support: dict) -> str:
    if comparison_df.empty:
        return "low"

    first_row = comparison_df.iloc[0]
    strong_count = int(first_row.get("strong_count", 0))
    weak_count = int(first_row.get("weak_count", 0))
    minimum_group = min(strong_count, weak_count)
    supportive_count = len(support["supportive"])
    contradictory_count = len(support["contradictory"])

    if minimum_group <= 1:
        return "low"
    if minimum_group <= 3:
        return "medium" if supportive_count >= 3 and contradictory_count <= 1 else "low"
    if supportive_count >= 5 and contradictory_count <= 1:
        return "high"
    if supportive_count >= 3:
        return "medium"
    return "low"


def top_feature_lines(comparison_df: pd.DataFrame, limit: int) -> str:
    if comparison_df.empty:
        return "n/a"
    rows = sort_rows_by_strength([row for _, row in comparison_df.iterrows()])[:limit]
    if not rows:
        return "n/a"

    parts = []
    for row in rows:
        if row["feature_type"] == "numeric":
            parts.append(
                f"{row['feature_label']} ({fmt_number(row['median_strong'])} vs {fmt_number(row['median_weak'])}, {row['effect_direction']})"
            )
        else:
            parts.append(
                f"{row['feature_label']} ({pct(row['share_strong'])} vs {pct(row['share_weak'])}, {row['effect_direction']})"
            )
    return "; ".join(parts)


def feature_sentence(comparison_df: pd.DataFrame, feature_key: str, is_share: bool = False) -> str:
    row = get_feature_row(comparison_df, feature_key)
    if row is None:
        return "n/a"

    if is_share:
        diff_text = fmt_number(row["difference"] * 100, 2) if pd.notna(row["difference"]) else "n/a"
        return (
            f"strong {pct(row['share_strong'])} vs weak {pct(row['share_weak'])} "
            f"(diff {diff_text} pp, signal {row['signal_strength']})"
        )

    p_value_text = fmt_number(row["p_value"], 3) if pd.notna(row["p_value"]) else "n/a"
    overlap_text = pct(row["percentile_overlap_pct"]) if pd.notna(row["percentile_overlap_pct"]) else "n/a"
    return (
        f"strong {fmt_number(row['median_strong'])} vs weak {fmt_number(row['median_weak'])} "
        f"(diff {fmt_number(row['difference'])}, p {p_value_text}, overlap {overlap_text}, signal {row['signal_strength']})"
    )


def build_candidate_filter_lines(comparison_df: pd.DataFrame, support: dict, confidence: str) -> list[str]:
    if confidence == "low":
        return []

    candidate_rows = []
    for feature_key in [
        "rsi14_15m",
        "return_pre_15m_pct",
        "relative_volume_5m",
        "abs_distance_to_ema20_pct_15m",
    ]:
        row = get_feature_row(comparison_df, feature_key)
        if row is None or row["signal_strength"] == "minimal":
            continue
        candidate_rows.append(row)

    lines: list[str] = []
    for row in candidate_rows:
        if pd.isna(row["median_strong"]) or pd.isna(row["median_weak"]):
            continue
        midpoint = (row["median_strong"] + row["median_weak"]) / 2.0
        if row["difference"] < 0:
            lines.append(f"{row['feature_label']} <= {fmt_number(midpoint)}")
        else:
            lines.append(f"{row['feature_label']} >= {fmt_number(midpoint)}")

    pullback_row = get_feature_row(comparison_df, "pullback_or_reversal_flag")
    breakout_row = get_feature_row(comparison_df, "breakout_or_impulse_flag")
    if pullback_row is not None and pd.notna(pullback_row["difference"]) and pullback_row["difference"] > 0:
        lines.append("prefer pullback/reversal-like context over pure breakout chase")
    if breakout_row is not None and pd.notna(breakout_row["difference"]) and breakout_row["difference"] < 0:
        lines.append("avoid the most impulsive breakout/continuation contexts")

    return lines[:5]


def stronger_side_label(long_support: dict, short_support: dict) -> str:
    long_strength = len(long_support["supportive"]) - len(long_support["contradictory"])
    short_strength = len(short_support["supportive"]) - len(short_support["contradictory"])
    if long_strength > short_strength + 1:
        return "it is more visible on ETH Long"
    if short_strength > long_strength + 1:
        return "it is more visible on ETH Short"
    return "no single side clearly dominates"


def side_difference_text(long_support: dict, short_support: dict) -> str:
    long_top = long_support["most_useful_features"]
    short_top = short_support["most_useful_features"]
    if long_top == short_top:
        return "Long and Short rely on broadly similar context features, but the weak-side samples are small."
    return (
        "Long and Short do not separate in exactly the same way. "
        f"Long differences concentrate more in {long_top}, while Short differences concentrate more in {short_top}."
    )


def sort_rows_by_strength(rows: list[pd.Series]) -> list[pd.Series]:
    def score(row: pd.Series) -> tuple[float, float, float]:
        effect_size = abs(float(row.get("effect_size", 0.0))) if pd.notna(row.get("effect_size", np.nan)) else 0.0
        p_value = float(row.get("p_value", 1.0)) if pd.notna(row.get("p_value", np.nan)) else 1.0
        overlap = float(row.get("percentile_overlap_pct", 1.0)) if pd.notna(row.get("percentile_overlap_pct", np.nan)) else 1.0
        return (effect_size, 1 - p_value, 1 - overlap)

    return sorted(rows, key=score, reverse=True)


def get_feature_row(comparison_df: pd.DataFrame, feature_key: str) -> pd.Series | None:
    matches = comparison_df.loc[comparison_df["feature_key"] == feature_key]
    if matches.empty:
        return None
    return matches.iloc[0]


def create_eth_context_plots(eth_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    if eth_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    groups = ["strong_entry", "medium_entry", "weak_entry"]
    for axis, (feature, label) in zip(axes.flatten(), ETH_CONTEXT_PLOT_FEATURES):
        data = [
            pd.to_numeric(
                eth_df.loc[eth_df["entry_strength_group"] == group, feature],
                errors="coerce",
            ).dropna().to_numpy()
            for group in groups
        ]
        axis.boxplot(data, tick_labels=groups)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2)

    fig.tight_layout()
    feature_plot_path = plots_dir / "eth_context_feature_distributions.png"
    fig.savefig(feature_plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", feature_plot_path.name)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    side_frames = [("Long", eth_df.loc[eth_df["side"] == "Long"]), ("Short", eth_df.loc[eth_df["side"] == "Short"])]
    for row_index, (side_label, side_frame) in enumerate(side_frames):
        for col_index, (feature, label) in enumerate(SIDE_COMPARE_PLOT_FEATURES):
            axis = axes[row_index, col_index]
            data = [
                pd.to_numeric(
                    side_frame.loc[side_frame["entry_strength_group"] == group, feature],
                    errors="coerce",
                ).dropna().to_numpy()
                for group in ["strong_entry", "weak_entry"]
            ]
            axis.boxplot(data, tick_labels=["strong", "weak"])
            axis.set_title(f"{side_label}: {label}")
            axis.grid(alpha=0.2)

    fig.tight_layout()
    side_plot_path = plots_dir / "eth_long_vs_short_context.png"
    fig.savefig(side_plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", side_plot_path.name)
