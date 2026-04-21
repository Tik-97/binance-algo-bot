from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME, resolve_output_path
from utils import fmt_int, fmt_number, pct, safe_divide


RULE_FEATURE_SPECS = [
    {"key": "rsi14_5m", "label": "RSI 5m", "family": "momentum"},
    {"key": "rsi14_15m", "label": "RSI 15m", "family": "momentum"},
    {"key": "stoch_rsi14_15m", "label": "Stoch RSI 15m", "family": "momentum"},
    {"key": "return_pre_1m_pct", "label": "Pre-entry return 1m %", "family": "pre_entry"},
    {"key": "return_pre_5m_pct", "label": "Pre-entry return 5m %", "family": "pre_entry"},
    {"key": "return_pre_15m_pct", "label": "Pre-entry return 15m %", "family": "pre_entry"},
    {"key": "return_pre_60m_pct", "label": "Pre-entry return 60m %", "family": "pre_entry"},
    {"key": "relative_volume_1m", "label": "Relative volume 1m", "family": "volume"},
    {"key": "relative_volume_5m", "label": "Relative volume 5m", "family": "volume"},
    {"key": "atr14_pct_15m", "label": "ATR % 15m", "family": "volatility"},
    {"key": "distance_to_ema20_pct_5m", "label": "Distance to EMA20 5m %", "family": "extension"},
    {"key": "distance_to_ema20_pct_15m", "label": "Distance to EMA20 15m %", "family": "extension"},
    {"key": "distance_to_ema50_pct_15m", "label": "Distance to EMA50 15m %", "family": "extension"},
    {"key": "distance_to_last_1h_high_pct", "label": "Distance to 1h high %", "family": "structure"},
    {"key": "distance_to_last_1h_low_pct", "label": "Distance to 1h low %", "family": "structure"},
    {"key": "distance_to_last_4h_high_pct", "label": "Distance to 4h high %", "family": "structure"},
    {"key": "distance_to_last_4h_low_pct", "label": "Distance to 4h low %", "family": "structure"},
    {"key": "recent_range_15m_pct", "label": "Recent range 15m %", "family": "volatility"},
    {"key": "recent_range_1h_pct", "label": "Recent range 1h %", "family": "volatility"},
    {"key": "open_hour_utc", "label": "Hour of day UTC", "family": "time"},
]

FAMILY_FEATURE_PRIORITY = {
    "momentum": ["rsi14_15m", "rsi14_5m", "stoch_rsi14_15m"],
    "pre_entry": ["return_pre_15m_pct", "return_pre_60m_pct", "return_pre_5m_pct", "return_pre_1m_pct"],
    "volume": ["relative_volume_5m", "relative_volume_1m"],
    "extension": ["distance_to_ema50_pct_15m", "distance_to_ema20_pct_15m", "distance_to_ema20_pct_5m"],
    "structure": ["distance_to_last_4h_high_pct", "distance_to_last_4h_low_pct", "distance_to_last_1h_high_pct", "distance_to_last_1h_low_pct"],
    "volatility": ["atr14_pct_15m", "recent_range_1h_pct", "recent_range_15m_pct"],
    "time": ["open_hour_utc"],
}


def run_eth_cluster_rule_extraction(
    cluster_frame: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    focus_clusters = sorted(cluster_frame["cluster_id"].dropna().unique().tolist())[:2] if not cluster_frame.empty else []
    empty_text = (
        "ETH CLUSTER RULE EXTRACTION\n===========================\n"
        "No sufficient ETH cluster data was available for rule extraction.\n"
    )

    if len(focus_clusters) < 2:
        resolve_output_path(output_dir, "eth_cluster_rule_extraction").write_text(empty_text, encoding="utf-8")
        return {
            "feature_ranges": pd.DataFrame(),
            "rule_quality": pd.DataFrame(),
            "text": empty_text,
            "section_text": empty_text,
            "rules": [],
        }

    logger.info("Extracting approximate rule-based ETH context filters from clusters %s", focus_clusters)
    feature_ranges = build_feature_ranges(cluster_frame, focus_clusters[0], focus_clusters[1])
    rules = []
    quality_rows = []

    for target_cluster in focus_clusters:
        other_cluster = [cluster_id for cluster_id in focus_clusters if cluster_id != target_cluster][0]
        rule = extract_cluster_rule(cluster_frame, cluster_summary, feature_ranges, target_cluster, other_cluster)
        rules.append(rule)
        quality_rows.append(evaluate_rule_quality(cluster_frame, target_cluster, rule))

    quality_df = pd.DataFrame(quality_rows)
    text = build_rule_extraction_text(cluster_summary, feature_ranges, rules, quality_df)
    section_text = build_rule_extraction_section(cluster_summary, rules, quality_df)
    resolve_output_path(output_dir, "eth_cluster_rule_extraction").write_text(text, encoding="utf-8")

    return {
        "feature_ranges": feature_ranges,
        "rule_quality": quality_df,
        "text": text,
        "section_text": section_text,
        "rules": rules,
    }


def build_feature_ranges(cluster_frame: pd.DataFrame, cluster_1: int, cluster_2: int) -> pd.DataFrame:
    rows = []
    frame_1 = cluster_frame.loc[cluster_frame["cluster_id"] == cluster_1]
    frame_2 = cluster_frame.loc[cluster_frame["cluster_id"] == cluster_2]

    for spec in RULE_FEATURE_SPECS:
        key = spec["key"]
        values_1 = pd.to_numeric(frame_1[key], errors="coerce").dropna().to_numpy(dtype=float)
        values_2 = pd.to_numeric(frame_2[key], errors="coerce").dropna().to_numpy(dtype=float)

        stats_1 = distribution_stats(values_1)
        stats_2 = distribution_stats(values_2)
        mann_whitney = mann_whitney_summary(values_1, values_2)
        overlap = percentile_overlap_pct(values_1, values_2)
        difference = stats_1["median"] - stats_2["median"] if not (pd.isna(stats_1["median"]) or pd.isna(stats_2["median"])) else np.nan

        row = {
            "feature_key": key,
            "feature_label": spec["label"],
            "family": spec["family"],
            "cluster_1_count": int(values_1.size),
            "cluster_1_mean": stats_1["mean"],
            "cluster_1_median": stats_1["median"],
            "cluster_1_std": stats_1["std"],
            "cluster_1_iqr": stats_1["iqr"],
            "cluster_1_p10": stats_1["p10"],
            "cluster_1_p25": stats_1["p25"],
            "cluster_1_p75": stats_1["p75"],
            "cluster_1_p90": stats_1["p90"],
            "cluster_2_count": int(values_2.size),
            "cluster_2_mean": stats_2["mean"],
            "cluster_2_median": stats_2["median"],
            "cluster_2_std": stats_2["std"],
            "cluster_2_iqr": stats_2["iqr"],
            "cluster_2_p10": stats_2["p10"],
            "cluster_2_p25": stats_2["p25"],
            "cluster_2_p75": stats_2["p75"],
            "cluster_2_p90": stats_2["p90"],
            "median_difference_cluster_1_minus_2": difference,
            "effect_direction": compare_direction(difference),
            "mann_whitney_p_value": mann_whitney["p_value"],
            "effect_size": mann_whitney["rank_biserial"],
            "percentile_overlap_pct": overlap,
            "signal_strength": signal_strength_label(mann_whitney["rank_biserial"], mann_whitney["p_value"], overlap),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def distribution_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "iqr": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }
    p10, p25, p75, p90 = np.nanpercentile(values, [10, 25, 75, 90])
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
        "iqr": float(p75 - p25),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
    }


def mann_whitney_summary(values_1: np.ndarray, values_2: np.ndarray) -> dict:
    if values_1.size < 2 or values_2.size < 2:
        return {"p_value": np.nan, "rank_biserial": np.nan}

    combined = np.concatenate([values_1, values_2])
    ranks = pd.Series(combined).rank(method="average").to_numpy(dtype=float)
    n1 = values_1.size
    n2 = values_2.size
    rank_sum_1 = ranks[:n1].sum()
    u1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = float(np.sum(tie_counts**3 - tie_counts)) if tie_counts.size else 0.0
    variance = n1 * n2 / 12.0 * ((n1 + n2 + 1) - safe_tie_adjustment(tie_term, n1 + n2))
    if variance <= 0:
        return {"p_value": np.nan, "rank_biserial": np.nan}

    z_score = (u1 - mean_u) / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2))
    rank_biserial = (2 * u1) / (n1 * n2) - 1
    return {"p_value": p_value, "rank_biserial": rank_biserial}


def safe_tie_adjustment(tie_term: float, total_count: int) -> float:
    if total_count <= 1:
        return 0.0
    return tie_term / (total_count * (total_count - 1))


def percentile_overlap_pct(values_1: np.ndarray, values_2: np.ndarray) -> float:
    if values_1.size < 2 or values_2.size < 2:
        return np.nan
    low_1, high_1 = np.nanpercentile(values_1, [10, 90])
    low_2, high_2 = np.nanpercentile(values_2, [10, 90])
    overlap_low = max(low_1, low_2)
    overlap_high = min(high_1, high_2)
    union_low = min(low_1, low_2)
    union_high = max(high_1, high_2)
    if union_high <= union_low:
        return np.nan
    overlap = max(0.0, overlap_high - overlap_low)
    return float(overlap / (union_high - union_low))


def compare_direction(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value > 0:
        return "cluster_1_higher"
    if value < 0:
        return "cluster_1_lower"
    return "flat"


def signal_strength_label(effect_size: float, p_value: float, overlap: float) -> str:
    if pd.isna(effect_size):
        return "insufficient_data"
    strength = abs(effect_size)
    overlap_penalty = (1 - overlap) if pd.notna(overlap) else 0.0
    if strength >= 0.45 and (pd.isna(p_value) or p_value <= 0.10) and overlap_penalty >= 0.20:
        return "strong"
    if strength >= 0.25 and (pd.isna(p_value) or p_value <= 0.20) and overlap_penalty >= 0.10:
        return "medium"
    if strength >= 0.15:
        return "weak"
    return "minimal"


def extract_cluster_rule(
    cluster_frame: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    feature_ranges: pd.DataFrame,
    target_cluster: int,
    other_cluster: int,
) -> dict:
    target_frame = cluster_frame.loc[cluster_frame["cluster_id"] == target_cluster]
    other_frame = cluster_frame.loc[cluster_frame["cluster_id"] == other_cluster]
    dominant_side = dominant_cluster_side(target_frame)

    selected_conditions = []
    current_mask = pd.Series(True, index=cluster_frame.index)

    if dominant_side is not None:
        side_mask = cluster_frame["side"] == dominant_side
        selected_conditions.append(
            {
                "feature_key": "side",
                "feature_label": "Side",
                "family": "side",
                "rule_type": "equals",
                "value": dominant_side,
                "text": f"side = {dominant_side}",
                "mask": side_mask,
            }
        )
        current_mask = current_mask & side_mask

    family_candidates = build_family_candidates(cluster_frame, feature_ranges, target_cluster, other_cluster, dominant_side)
    current_metrics = filter_metrics(cluster_frame, current_mask, target_cluster)

    for candidate in family_candidates:
        new_mask = current_mask & candidate["mask"]
        new_metrics = filter_metrics(cluster_frame, new_mask, target_cluster)
        if new_metrics["covered_trade_count"] < max(4, int(len(cluster_frame) * 0.08)):
            continue
        if new_metrics["recall"] < 0.35:
            continue
        if (
            new_metrics["precision"] >= current_metrics["precision"] - 0.02
            and new_metrics["f1_score"] >= current_metrics["f1_score"] + 0.03
        ) or (
            new_metrics["precision"] >= current_metrics["precision"] + 0.05
            and new_metrics["recall"] >= current_metrics["recall"] - 0.10
        ):
            selected_conditions.append(candidate)
            current_mask = new_mask
            current_metrics = new_metrics

    structure_candidate = next((candidate for candidate in family_candidates if candidate["family"] == "structure"), None)
    has_structure = any(condition.get("family") == "structure" for condition in selected_conditions)
    if structure_candidate is not None and not has_structure:
        new_mask = current_mask & structure_candidate["mask"]
        new_metrics = filter_metrics(cluster_frame, new_mask, target_cluster)
        if (
            new_metrics["covered_trade_count"] >= max(4, int(len(cluster_frame) * 0.08))
            and new_metrics["precision"] >= current_metrics["precision"] - 0.02
            and new_metrics["recall"] >= max(0.50, current_metrics["recall"] - 0.12)
        ):
            selected_conditions.append(structure_candidate)
            current_mask = new_mask
            current_metrics = new_metrics

    if len(selected_conditions) < 4:
        for candidate in family_candidates:
            if candidate in selected_conditions:
                continue
            new_mask = current_mask & candidate["mask"]
            new_metrics = filter_metrics(cluster_frame, new_mask, target_cluster)
            if (
                new_metrics["covered_trade_count"] >= max(4, int(len(cluster_frame) * 0.08))
                and new_metrics["precision"] >= current_metrics["precision"] - 0.02
                and new_metrics["recall"] >= 0.55
                and new_metrics["f1_score"] >= current_metrics["f1_score"] - 0.02
            ):
                selected_conditions.append(candidate)
                current_mask = new_mask
                current_metrics = new_metrics
                break

    if len(selected_conditions) < 3:
        for candidate in family_candidates:
            if candidate in selected_conditions:
                continue
            new_mask = current_mask & candidate["mask"]
            new_metrics = filter_metrics(cluster_frame, new_mask, target_cluster)
            if new_metrics["covered_trade_count"] < max(4, int(len(cluster_frame) * 0.06)):
                continue
            if new_metrics["recall"] < 0.25:
                continue
            selected_conditions.append(candidate)
            current_mask = new_mask
            current_metrics = new_metrics
            if len(selected_conditions) >= 3:
                break

    summary_row = cluster_summary.loc[cluster_summary["cluster_id"] == target_cluster].iloc[0]
    return {
        "target_cluster": target_cluster,
        "other_cluster": other_cluster,
        "cluster_description": summary_row["cluster_description"],
        "dominant_side": dominant_side,
        "conditions": selected_conditions,
        "final_mask": current_mask,
        "dominant_features": top_cluster_features(feature_ranges, target_cluster, limit=5),
    }


def dominant_cluster_side(target_frame: pd.DataFrame) -> str | None:
    short_share = safe_divide((target_frame["side"] == "Short").sum(), len(target_frame))
    long_share = safe_divide((target_frame["side"] == "Long").sum(), len(target_frame))
    if short_share >= 0.75:
        return "Short"
    if long_share >= 0.75:
        return "Long"
    return None


def build_family_candidates(
    cluster_frame: pd.DataFrame,
    feature_ranges: pd.DataFrame,
    target_cluster: int,
    other_cluster: int,
    dominant_side: str | None,
) -> list[dict]:
    candidates = []
    for family, feature_order in FAMILY_FEATURE_PRIORITY.items():
        feature_key = choose_family_feature(feature_ranges, target_cluster, family, feature_order, dominant_side)
        if feature_key is None:
            continue
        candidate = build_candidate_condition(cluster_frame, feature_ranges, target_cluster, other_cluster, feature_key)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def choose_family_feature(
    feature_ranges: pd.DataFrame,
    target_cluster: int,
    family: str,
    feature_order: list[str],
    dominant_side: str | None,
) -> str | None:
    subset = feature_ranges.loc[feature_ranges["family"] == family].copy()
    if subset.empty:
        return None

    if family == "structure" and dominant_side == "Short":
        priority = ["distance_to_last_4h_high_pct", "distance_to_last_1h_high_pct"]
    elif family == "structure" and dominant_side == "Long":
        priority = ["distance_to_last_4h_low_pct", "distance_to_last_1h_low_pct"]
    else:
        priority = feature_order

    if family == "structure" and dominant_side in {"Short", "Long"}:
        preferred_subset = subset.loc[subset["feature_key"].isin(priority)].copy()
        if not preferred_subset.empty:
            subset = preferred_subset

    subset["priority"] = subset["feature_key"].apply(lambda key: priority.index(key) if key in priority else 99)
    subset["score"] = subset.apply(feature_separation_score, axis=1)
    subset = subset.sort_values(["score", "priority"], ascending=[False, True])
    top_row = subset.iloc[0]
    return str(top_row["feature_key"]) if pd.notna(top_row["score"]) else None


def feature_separation_score(row: pd.Series) -> float:
    effect = abs(float(row["effect_size"])) if pd.notna(row["effect_size"]) else 0.0
    overlap_penalty = 1 - float(row["percentile_overlap_pct"]) if pd.notna(row["percentile_overlap_pct"]) else 0.0
    return effect * 0.75 + overlap_penalty * 0.25


def build_candidate_condition(
    cluster_frame: pd.DataFrame,
    feature_ranges: pd.DataFrame,
    target_cluster: int,
    other_cluster: int,
    feature_key: str,
) -> dict | None:
    row = feature_ranges.loc[feature_ranges["feature_key"] == feature_key]
    if row.empty:
        return None
    row = row.iloc[0]

    cluster_column_prefix = "cluster_1" if target_cluster == 1 else "cluster_2"
    other_column_prefix = "cluster_2" if target_cluster == 1 else "cluster_1"

    median_target = row[f"{cluster_column_prefix}_median"]
    median_other = row[f"{other_column_prefix}_median"]
    q10 = row[f"{cluster_column_prefix}_p10"]
    q25 = row[f"{cluster_column_prefix}_p25"]
    q75 = row[f"{cluster_column_prefix}_p75"]
    q90 = row[f"{cluster_column_prefix}_p90"]

    series = pd.to_numeric(cluster_frame[feature_key], errors="coerce")
    if pd.isna(median_target) or pd.isna(median_other):
        return None

    family = str(row["family"])
    label = str(row["feature_label"])
    rounded_q10 = round_feature_value(feature_key, q10)
    rounded_q25 = round_feature_value(feature_key, q25)
    rounded_q75 = round_feature_value(feature_key, q75)
    rounded_q90 = round_feature_value(feature_key, q90)

    rule_type = "between"
    lower_value = rounded_q10
    upper_value = rounded_q90
    if family == "momentum":
        lower_value = rounded_q25
        upper_value = rounded_q90 if median_target > median_other else rounded_q75
        rule_type = "between"
    elif family == "pre_entry":
        if median_target > median_other:
            rule_type = "ge"
            lower_value = rounded_q25
        else:
            rule_type = "le"
            upper_value = rounded_q75
    elif family == "structure":
        if "high" in feature_key and median_target < median_other:
            rule_type = "le"
            upper_value = rounded_q75
        elif "low" in feature_key and median_target < median_other:
            rule_type = "le"
            upper_value = rounded_q75
        elif median_target > median_other:
            rule_type = "ge"
            lower_value = rounded_q25
        else:
            rule_type = "between"
    elif family == "extension":
        if median_target > median_other:
            rule_type = "ge"
            lower_value = rounded_q25
        else:
            rule_type = "le"
            upper_value = rounded_q75
    elif family == "volume":
        if median_target > median_other:
            rule_type = "ge"
            lower_value = rounded_q25
        else:
            rule_type = "le"
            upper_value = rounded_q75
    elif family == "volatility":
        rule_type = "between"
        lower_value = rounded_q10
        upper_value = rounded_q90
    elif family == "time":
        rule_type = "between"
        lower_value = rounded_q25
        upper_value = rounded_q75

    mask = apply_numeric_rule(series, rule_type, lower_value, upper_value)
    metrics = filter_metrics(cluster_frame, mask, target_cluster)
    score = metrics["precision"] * 0.45 + metrics["recall"] * 0.35 + feature_separation_score(row) * 0.20

    return {
        "feature_key": feature_key,
        "feature_label": label,
        "family": family,
        "rule_type": rule_type,
        "lower_value": lower_value,
        "upper_value": upper_value,
        "text": render_rule_text(label, rule_type, lower_value, upper_value),
        "mask": mask.fillna(False),
        "score": score,
        "single_rule_precision": metrics["precision"],
        "single_rule_recall": metrics["recall"],
    }


def apply_numeric_rule(
    series: pd.Series,
    rule_type: str,
    lower_value: float | None,
    upper_value: float | None,
) -> pd.Series:
    if rule_type == "ge":
        return series >= lower_value
    if rule_type == "le":
        return series <= upper_value
    return series.between(lower_value, upper_value, inclusive="both")


def render_rule_text(label: str, rule_type: str, lower_value: float | None, upper_value: float | None) -> str:
    if rule_type == "ge":
        return f"{label} >= {fmt_threshold(lower_value)}"
    if rule_type == "le":
        return f"{label} <= {fmt_threshold(upper_value)}"
    return f"{label} between {fmt_threshold(lower_value)} and {fmt_threshold(upper_value)}"


def fmt_threshold(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def round_feature_value(feature_key: str, value: float) -> float:
    if pd.isna(value):
        return np.nan
    if "stoch" in feature_key:
        step = 0.05
    elif "rsi" in feature_key:
        step = 1.0
    elif "hour" in feature_key:
        step = 1.0
    elif "volume" in feature_key:
        step = 0.1
    else:
        step = 0.05 if abs(value) < 3 else 0.1
    return round(value / step) * step


def filter_metrics(cluster_frame: pd.DataFrame, mask: pd.Series, target_cluster: int) -> dict:
    covered = cluster_frame.loc[mask.fillna(False)]
    target_total = int((cluster_frame["cluster_id"] == target_cluster).sum())
    target_covered = int((covered["cluster_id"] == target_cluster).sum())
    covered_count = len(covered)

    precision = safe_divide(target_covered, covered_count)
    recall = safe_divide(target_covered, target_total)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "covered_trade_count": covered_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def top_cluster_features(feature_ranges: pd.DataFrame, target_cluster: int, limit: int) -> list[str]:
    subset = feature_ranges.copy()
    if target_cluster == 2:
        subset["effect_size"] = -pd.to_numeric(subset["effect_size"], errors="coerce")
        subset["median_difference_cluster_1_minus_2"] = -pd.to_numeric(
            subset["median_difference_cluster_1_minus_2"], errors="coerce"
        )
    subset["rank_score"] = subset.apply(feature_separation_score, axis=1)
    ordered = subset.sort_values(["rank_score", "mann_whitney_p_value"], ascending=[False, True]).head(limit)
    return ordered["feature_label"].astype(str).tolist()


def evaluate_rule_quality(cluster_frame: pd.DataFrame, target_cluster: int, rule: dict) -> dict:
    mask = rule["final_mask"].fillna(False)
    covered = cluster_frame.loc[mask]
    target_frame = cluster_frame.loc[cluster_frame["cluster_id"] == target_cluster]

    covered_target = covered.loc[covered["cluster_id"] == target_cluster]
    precision = safe_divide(len(covered_target), len(covered))
    recall = safe_divide(len(covered_target), len(target_frame))
    return {
        "cluster_id": target_cluster,
        "cluster_description": rule["cluster_description"],
        "condition_count": len(rule["conditions"]),
        "rule_text": " AND ".join(condition["text"] for condition in rule["conditions"]),
        "covered_trade_count": int(len(covered)),
        "coverage_share_total_eth": safe_divide(len(covered), len(cluster_frame)),
        "precision": precision,
        "recall": recall,
        "covered_total_pnl": covered["closingPnl"].sum(),
        "covered_avg_pnl": covered["closingPnl"].mean(),
        "covered_median_realized_return_pct": covered["realized_return_pct"].median(),
        "covered_median_MAE_pct": covered["MAE_pct"].median(),
        "covered_strong_reaction_share": covered["group_fast_positive_reaction"].astype(float).mean(),
        "covered_cluster_share_of_cluster_pnl": safe_divide(covered_target["closingPnl"].sum(), target_frame["closingPnl"].sum()),
        "quality_score": precision * 0.45 + recall * 0.35 + covered["group_fast_positive_reaction"].astype(float).mean() * 0.20,
    }


def build_rule_extraction_text(
    cluster_summary: pd.DataFrame,
    feature_ranges: pd.DataFrame,
    rules: list[dict],
    quality_df: pd.DataFrame,
) -> str:
    lines = [
        "ETH CLUSTER RULE EXTRACTION",
        "===========================",
        "",
        "This section translates the ETH setup clusters into rough rule-based context filters. These are replication hypotheses, not claims about the trader's exact strategy.",
        "",
    ]

    for rule in rules:
        cluster_id = rule["target_cluster"]
        summary_row = cluster_summary.loc[cluster_summary["cluster_id"] == cluster_id].iloc[0]
        quality_row = quality_df.loc[quality_df["cluster_id"] == cluster_id].iloc[0]
        lines.extend(
            [
                f"Cluster {fmt_int(cluster_id)}",
                f"---------",
                f"Summary: {summary_row['cluster_description']}",
                (
                    f"Profile: trades {fmt_int(summary_row['trade_count'])}, total pnl {fmt_number(summary_row['total_pnl'])}, "
                    f"win rate {pct(summary_row['win_rate'])}, strong post-entry reaction {pct(summary_row['strong_post_entry_reaction_share'])}"
                ),
                f"Dominant features: {', '.join(rule['dominant_features']) if rule['dominant_features'] else 'n/a'}",
                "Candidate context filter:",
            ]
        )
        for condition in rule["conditions"]:
            lines.append(f"- {condition['text']}")
        lines.extend(
            [
                (
                    f"Coverage: {fmt_int(quality_row['covered_trade_count'])} trades, "
                    f"{pct(quality_row['coverage_share_total_eth'])} of ETH sample, "
                    f"{pct(quality_row['covered_cluster_share_of_cluster_pnl'])} of cluster pnl."
                ),
                (
                    f"Precision / recall: {pct(quality_row['precision'])} / {pct(quality_row['recall'])}; "
                    f"average pnl {fmt_number(quality_row['covered_avg_pnl'])}; "
                    f"median realized return {fmt_number(quality_row['covered_median_realized_return_pct'])}%."
                ),
                (
                    f"Risk behavior on covered trades: median MAE {fmt_number(quality_row['covered_median_MAE_pct'])}%, "
                    f"strong reaction share {pct(quality_row['covered_strong_reaction_share'])}."
                ),
                "Limitations: these thresholds are approximate bands derived from cluster distributions. They are meant to preserve the archetype shape, not to reconstruct the exact live decision rule.",
                "",
            ]
        )

    better_cluster = easier_cluster_to_replicate(quality_df)
    if better_cluster is not None:
        quality_row = quality_df.loc[quality_df["cluster_id"] == better_cluster].iloc[0]
        lines.extend(
            [
                "Simple replication filter draft",
                "-------------------------------",
                (
                    f"The easiest first draft is Cluster {fmt_int(better_cluster)} because its approximate rule-set "
                    f"keeps precision {pct(quality_row['precision'])} with recall {pct(quality_row['recall'])}."
                ),
                "Treat this only as a first-pass replication filter, not as a finished strategy.",
            ]
        )

    return "\n".join(lines) + "\n"


def build_rule_extraction_section(
    cluster_summary: pd.DataFrame,
    rules: list[dict],
    quality_df: pd.DataFrame,
) -> str:
    if quality_df.empty:
        return "ETH CLUSTER RULE EXTRACTION\n===========================\nNo ETH cluster rule extraction was available.\n"

    easier_cluster = easier_cluster_to_replicate(quality_df)
    best_cluster_row = quality_df.sort_values("quality_score", ascending=False).iloc[0]
    cluster_1_row = quality_df.loc[quality_df["cluster_id"] == 1].iloc[0]
    cluster_2_row = quality_df.loc[quality_df["cluster_id"] == 2].iloc[0]

    if cluster_1_row["precision"] >= 0.75 and cluster_1_row["recall"] >= 0.45:
        cluster_1_answer = "Cluster 1 can be described by a reasonably simple filter without collapsing into noise."
    else:
        cluster_1_answer = "Cluster 1 can be sketched in broad terms, but the rough filter loses too much of the cluster to be called robust."

    if cluster_2_row["precision"] >= 0.75 and cluster_2_row["recall"] >= 0.45:
        cluster_2_answer = "Cluster 2 can also be approximated with a small set of broad context conditions."
    else:
        cluster_2_answer = "Cluster 2 is harder to compress into a short rule-set without either losing recall or becoming too narrow."

    confidence = rule_extraction_confidence(quality_df)
    overfit_risk = (
        "The rules stay broad enough to remain interpretable."
        if quality_df["condition_count"].max() <= 5 and quality_df["precision"].min() >= 0.75
        else "There is still a noticeable risk of turning the rules into an overfitted checklist."
    )

    section = f"""
ETH CLUSTER RULE EXTRACTION
===========================

This section turns the ETH setup clusters into approximate candidate context filters. The goal is not to recover a secret signal, but to create rough rule-based approximations of the ETH archetypes.

Cluster answers:
- Cluster 1: {cluster_1_answer}
- Cluster 2: {cluster_2_answer}
- Cluster 1 quality: precision {pct(cluster_1_row['precision'])}, recall {pct(cluster_1_row['recall'])}, covered trades {fmt_int(cluster_1_row['covered_trade_count'])}.
- Cluster 2 quality: precision {pct(cluster_2_row['precision'])}, recall {pct(cluster_2_row['recall'])}, covered trades {fmt_int(cluster_2_row['covered_trade_count'])}.

Replication read:
- The cluster that is easier to replicate with simple conditions is Cluster {fmt_int(easier_cluster) if easier_cluster is not None else 'n/a'}.
- The better first replication candidate is Cluster {fmt_int(best_cluster_row['cluster_id'])}, because it gives the strongest balance of precision, recall, and covered PnL.
- {overfit_risk}
- Confidence level: {confidence}
""".strip()
    return section + "\n"


def easier_cluster_to_replicate(quality_df: pd.DataFrame) -> int | None:
    if quality_df.empty:
        return None
    ordered = quality_df.sort_values(
        ["quality_score", "precision", "recall", "condition_count"],
        ascending=[False, False, False, True],
    )
    return int(ordered.iloc[0]["cluster_id"])


def rule_extraction_confidence(quality_df: pd.DataFrame) -> str:
    if quality_df.empty:
        return "low"
    min_precision = quality_df["precision"].min()
    min_recall = quality_df["recall"].min()
    max_conditions = quality_df["condition_count"].max()
    if min_precision >= 0.90 and min_recall >= 0.50 and max_conditions <= 5:
        return "high"
    if min_precision >= 0.75 and min_recall >= 0.35 and max_conditions <= 6:
        return "medium"
    return "low"


def create_eth_cluster_rule_plots(
    feature_ranges: pd.DataFrame,
    quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if feature_ranges.empty or quality_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    top_features = (
        feature_ranges.assign(abs_effect=pd.to_numeric(feature_ranges["effect_size"], errors="coerce").abs())
        .sort_values(["abs_effect", "mann_whitney_p_value"], ascending=[False, True])
        .head(6)
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (_, row) in zip(axes.flatten(), top_features.iterrows()):
        axis.errorbar(
            [1],
            [row["cluster_1_median"]],
            yerr=[[row["cluster_1_median"] - row["cluster_1_p10"]], [row["cluster_1_p90"] - row["cluster_1_median"]]],
            fmt="o",
            capsize=6,
            label="Cluster 1",
        )
        axis.errorbar(
            [2],
            [row["cluster_2_median"]],
            yerr=[[row["cluster_2_median"] - row["cluster_2_p10"]], [row["cluster_2_p90"] - row["cluster_2_median"]]],
            fmt="o",
            capsize=6,
            label="Cluster 2",
        )
        axis.set_title(row["feature_label"])
        axis.set_xticks([1, 2])
        axis.set_xticklabels(["C1", "C2"])
        axis.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    feature_plot_path = plots_dir / "eth_cluster_feature_ranges.png"
    fig.savefig(feature_plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", feature_plot_path.name)

    plot_quality = quality_df.copy()
    plot_quality["cluster_label"] = plot_quality["cluster_id"].apply(lambda value: f"Cluster {int(value)}")
    x_positions = np.arange(len(plot_quality))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_positions - width, plot_quality["coverage_share_total_eth"], width=width, label="Coverage")
    ax.bar(x_positions, plot_quality["precision"], width=width, label="Precision")
    ax.bar(x_positions + width, plot_quality["recall"], width=width, label="Recall")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_quality["cluster_label"])
    ax.set_ylim(0, 1.05)
    ax.set_title("ETH Cluster Rule Coverage")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    coverage_plot_path = plots_dir / "eth_cluster_rule_coverage.png"
    fig.tight_layout()
    fig.savefig(coverage_plot_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", coverage_plot_path.name)
