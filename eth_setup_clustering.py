from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME, resolve_output_path
from utils import fmt_int, fmt_number, pct, safe_divide


ETH_SYMBOL = "ETHUSDT"
CLUSTER_K_VALUES = [2, 3, 4, 5, 6]
KMEANS_MAX_ITER = 200
KMEANS_N_INIT = 24
CLUSTER_BOOTSTRAP_RUNS = 18
MODEL_SEED = 42

CLUSTER_FEATURE_SPECS = [
    ("rsi14_5m", "RSI 5m"),
    ("rsi14_15m", "RSI 15m"),
    ("stoch_rsi14_15m", "Stoch RSI 15m"),
    ("return_pre_1m_pct", "Pre-entry return 1m %"),
    ("return_pre_5m_pct", "Pre-entry return 5m %"),
    ("return_pre_15m_pct", "Pre-entry return 15m %"),
    ("return_pre_60m_pct", "Pre-entry return 60m %"),
    ("relative_volume_1m", "Relative volume 1m"),
    ("relative_volume_5m", "Relative volume 5m"),
    ("atr14_pct_15m", "ATR % 15m"),
    ("distance_to_ema20_pct_5m", "Distance to EMA20 5m %"),
    ("distance_to_ema20_pct_15m", "Distance to EMA20 15m %"),
    ("distance_to_ema50_pct_15m", "Distance to EMA50 15m %"),
    ("distance_to_last_1h_high_pct", "Distance to last 1h high %"),
    ("distance_to_last_1h_low_pct", "Distance to last 1h low %"),
    ("distance_to_last_4h_high_pct", "Distance to last 4h high %"),
    ("distance_to_last_4h_low_pct", "Distance to last 4h low %"),
    ("recent_range_15m_pct", "Recent range 15m %"),
    ("recent_range_1h_pct", "Recent range 1h %"),
]

CLUSTER_PLOT_LABELS = {
    "rsi14_5m": "RSI 5m",
    "rsi14_15m": "RSI 15m",
    "stoch_rsi14_15m": "Stoch RSI 15m",
    "return_pre_1m_pct": "Ret 1m",
    "return_pre_5m_pct": "Ret 5m",
    "return_pre_15m_pct": "Ret 15m",
    "return_pre_60m_pct": "Ret 60m",
    "relative_volume_1m": "RelVol 1m",
    "relative_volume_5m": "RelVol 5m",
    "atr14_pct_15m": "ATR% 15m",
    "distance_to_ema20_pct_5m": "Dist EMA20 5m",
    "distance_to_ema20_pct_15m": "Dist EMA20 15m",
    "distance_to_ema50_pct_15m": "Dist EMA50 15m",
    "distance_to_last_1h_high_pct": "Dist 1h high",
    "distance_to_last_1h_low_pct": "Dist 1h low",
    "distance_to_last_4h_high_pct": "Dist 4h high",
    "distance_to_last_4h_low_pct": "Dist 4h low",
    "recent_range_15m_pct": "Range 15m",
    "recent_range_1h_pct": "Range 1h",
    "hour_sin": "Hour sin",
    "hour_cos": "Hour cos",
    "side_short_flag": "Short side",
}


def run_eth_setup_clustering(
    context_df: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    prepared = prepare_eth_setup_frame(context_df, entry_quality_df, trades_df)
    empty_text = (
        "ETH SETUP CLUSTERING\n====================\n"
        "No sufficient ETHUSDT data was available for ETH setup clustering.\n"
    )

    if prepared.empty or len(prepared) < 12:
        resolve_output_path(output_dir, "eth_cluster_descriptions").write_text(empty_text, encoding="utf-8")
        return {
            "clusters": prepared,
            "cluster_summary": pd.DataFrame(),
            "best_cluster_profile": pd.DataFrame(),
            "description_text": empty_text,
            "section_text": empty_text,
        }

    feature_bundle = build_cluster_feature_matrix(prepared)
    X = feature_bundle["matrix"]
    feature_names = feature_bundle["feature_names"]

    if X.shape[0] < 12 or X.shape[1] < 4:
        resolve_output_path(output_dir, "eth_cluster_descriptions").write_text(empty_text, encoding="utf-8")
        return {
            "clusters": prepared,
            "cluster_summary": pd.DataFrame(),
            "best_cluster_profile": pd.DataFrame(),
            "description_text": empty_text,
            "section_text": empty_text,
        }

    logger.info("Running ETH setup clustering on %s ETH trades", len(prepared))
    evaluations = evaluate_kmeans_candidates(X)
    best_model = choose_best_model(evaluations)
    cluster_frame = prepared.copy()
    cluster_frame["cluster_raw"] = best_model["labels"]

    cluster_frame = add_cluster_projection(cluster_frame, X)
    summary = build_cluster_summary(cluster_frame, best_model)
    mapping = relabel_clusters_by_total_pnl(summary)
    cluster_frame["cluster_id"] = cluster_frame["cluster_raw"].map(mapping)
    summary["cluster_id"] = summary["cluster_raw"].map(mapping)
    summary = summary.sort_values("cluster_id").reset_index(drop=True)
    cluster_frame = cluster_frame.drop(columns=["cluster_raw"]).sort_values(["cluster_id", "trade_number"]).reset_index(drop=True)

    descriptions = build_cluster_descriptions(cluster_frame, summary)
    cluster_frame["cluster_description"] = cluster_frame["cluster_id"].map(descriptions)
    summary["cluster_description"] = summary["cluster_id"].map(descriptions)

    best_cluster_id = choose_best_cluster(summary)
    best_cluster_profile = build_best_cluster_profile(cluster_frame, best_cluster_id, feature_bundle)
    description_text = build_cluster_descriptions_text(summary, best_model, best_cluster_profile)
    section_text = build_summary_section(summary, best_model, best_cluster_profile)

    resolve_output_path(output_dir, "eth_cluster_descriptions").write_text(description_text, encoding="utf-8")
    return {
        "clusters": cluster_frame,
        "cluster_summary": summary,
        "best_cluster_profile": best_cluster_profile,
        "description_text": description_text,
        "section_text": section_text,
    }


def prepare_eth_setup_frame(
    context_df: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    eth_context = context_df.loc[context_df["symbol"] == ETH_SYMBOL].copy()
    if eth_context.empty:
        return eth_context

    quality_cols = [
        "trade_number",
        "holding_hours",
        "realized_return_pct",
        "MAE_pct",
        "MFE_pct",
        "captured_share_of_MFE",
        "group_fast_positive_reaction",
        "return_after_1m_pct",
        "return_after_5m_pct",
        "return_after_15m_pct",
        "return_after_60m_pct",
    ]
    quality = entry_quality_df.loc[entry_quality_df["symbol"] == ETH_SYMBOL, quality_cols].copy()
    trades = trades_df.loc[trades_df["symbol"] == ETH_SYMBOL, ["trade_number", "open_hour_utc"]].copy()

    merged = eth_context.merge(quality, on="trade_number", how="left", suffixes=("", "_entry"))
    merged = merged.merge(trades, on="trade_number", how="left")

    merged["open_hour_utc"] = pd.to_numeric(merged["open_hour_utc"], errors="coerce")
    merged["hour_sin"] = np.sin(2 * np.pi * merged["open_hour_utc"] / 24.0)
    merged["hour_cos"] = np.cos(2 * np.pi * merged["open_hour_utc"] / 24.0)
    merged["side_short_flag"] = (merged["side"] == "Short").astype(float)
    merged["side_long_flag"] = (merged["side"] == "Long").astype(float)
    merged["trade_positive_1m_flag"] = (merged["return_after_1m_pct"] > 0).astype(float)
    merged["trade_positive_5m_flag"] = (merged["return_after_5m_pct"] > 0).astype(float)
    merged["trade_positive_15m_flag"] = (merged["return_after_15m_pct"] > 0).astype(float)
    merged["trade_positive_60m_flag"] = (merged["return_after_60m_pct"] > 0).astype(float)

    for column, _ in CLUSTER_FEATURE_SPECS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    return merged


def build_cluster_feature_matrix(frame: pd.DataFrame) -> dict:
    feature_names = [column for column, _ in CLUSTER_FEATURE_SPECS] + ["hour_sin", "hour_cos", "side_short_flag"]
    base = frame[feature_names].copy()

    kept_columns = []
    scaled_columns = {}
    original_medians = {}
    original_stds = {}
    imputation_values = {}

    for column in feature_names:
        series = pd.to_numeric(base[column], errors="coerce")
        if series.notna().sum() < max(6, int(len(frame) * 0.35)):
            continue
        fill_value = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(fill_value)
        std = float(filled.std(ddof=0))
        if std <= 1e-9:
            continue
        mean = float(filled.mean())
        scaled_columns[column] = (filled - mean) / std
        kept_columns.append(column)
        original_medians[column] = fill_value
        original_stds[column] = std
        imputation_values[column] = fill_value

    scaled_frame = pd.DataFrame(scaled_columns, index=frame.index)
    matrix = scaled_frame.to_numpy(dtype=float)
    return {
        "matrix": matrix,
        "feature_names": kept_columns,
        "scaled_frame": scaled_frame,
        "original_medians": original_medians,
        "original_stds": original_stds,
        "imputation_values": imputation_values,
    }


def evaluate_kmeans_candidates(X: np.ndarray) -> list[dict]:
    evaluations = []
    for k in CLUSTER_K_VALUES:
        if k >= len(X):
            continue
        labels, centers, inertia = fit_kmeans(X, k, seed=MODEL_SEED)
        silhouette = silhouette_score_custom(X, labels)
        stability = bootstrap_cluster_stability(X, labels, k)
        counts = pd.Series(labels).value_counts()
        min_cluster_size = int(counts.min()) if not counts.empty else 0
        tiny_cluster_threshold = max(3, int(math.ceil(len(X) * 0.06)))
        tiny_cluster_count = int((counts < tiny_cluster_threshold).sum())
        interpretability = 1.0 - (tiny_cluster_count / max(k, 1))
        model_score = (silhouette * 0.55) + (stability * 0.35) + (interpretability * 0.10) - ((k - 2) * 0.01)

        evaluations.append(
            {
                "k": k,
                "labels": labels,
                "centers": centers,
                "inertia": inertia,
                "silhouette_score": silhouette,
                "stability_score": stability,
                "interpretability_score": interpretability,
                "tiny_cluster_count": tiny_cluster_count,
                "min_cluster_size": min_cluster_size,
                "model_score": model_score,
            }
        )
    return evaluations


def choose_best_model(evaluations: list[dict]) -> dict:
    if not evaluations:
        raise ValueError("No valid ETH clustering models were produced.")
    ordered = sorted(
        evaluations,
        key=lambda row: (
            row["model_score"],
            row["silhouette_score"],
            row["stability_score"],
            -row["k"],
        ),
        reverse=True,
    )
    return ordered[0]


def fit_kmeans(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_inertia = math.inf

    for _ in range(KMEANS_N_INIT):
        initial_indices = rng.choice(len(X), size=k, replace=False)
        centers = X[initial_indices].copy()

        for _ in range(KMEANS_MAX_ITER):
            labels = assign_nearest_centers(X, centers)
            new_centers = recompute_centers(X, labels, centers, rng)
            if np.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        labels = assign_nearest_centers(X, centers)
        inertia = cluster_inertia(X, labels, centers)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise ValueError("KMeans failed to converge.")
    return best_labels, best_centers, float(best_inertia)


def assign_nearest_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return distances.argmin(axis=1)


def recompute_centers(
    X: np.ndarray,
    labels: np.ndarray,
    current_centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new_centers = []
    for cluster_id in range(len(current_centers)):
        members = X[labels == cluster_id]
        if len(members) == 0:
            random_index = int(rng.integers(0, len(X)))
            new_centers.append(X[random_index])
        else:
            new_centers.append(members.mean(axis=0))
    return np.vstack(new_centers)


def cluster_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    total = 0.0
    for cluster_id in range(len(centers)):
        members = X[labels == cluster_id]
        if len(members) == 0:
            continue
        total += float(((members - centers[cluster_id]) ** 2).sum())
    return total


def silhouette_score_custom(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    scores = []
    for index in range(len(X)):
        own_cluster = labels[index]
        own_mask = labels == own_cluster
        own_count = int(own_mask.sum())
        if own_count <= 1:
            scores.append(0.0)
            continue

        a_value = float(distances[index, own_mask].sum() / (own_count - 1))
        b_value = math.inf
        for other_cluster in unique_labels:
            if other_cluster == own_cluster:
                continue
            other_mask = labels == other_cluster
            if not other_mask.any():
                continue
            other_distance = float(distances[index, other_mask].mean())
            b_value = min(b_value, other_distance)

        if not math.isfinite(b_value):
            scores.append(0.0)
            continue
        denominator = max(a_value, b_value)
        scores.append((b_value - a_value) / denominator if denominator > 0 else 0.0)

    return float(np.mean(scores)) if scores else np.nan


def bootstrap_cluster_stability(X: np.ndarray, base_labels: np.ndarray, k: int) -> float:
    scores = []
    rng = np.random.default_rng(MODEL_SEED + (k * 17))
    for bootstrap_index in range(CLUSTER_BOOTSTRAP_RUNS):
        sample_indices = rng.integers(0, len(X), size=len(X))
        sample = X[sample_indices]
        boot_labels, boot_centers, _ = fit_kmeans(sample, k, seed=MODEL_SEED + 100 + bootstrap_index)
        del boot_labels
        projected_labels = assign_nearest_centers(X, boot_centers)
        scores.append(adjusted_rand_index_custom(base_labels, projected_labels))
    return float(np.mean(scores)) if scores else np.nan


def adjusted_rand_index_custom(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    frame = pd.DataFrame({"a": labels_a, "b": labels_b})
    contingency = pd.crosstab(frame["a"], frame["b"])
    total = int(contingency.to_numpy().sum())
    if total <= 1:
        return np.nan

    sum_comb = sum(comb2(int(value)) for value in contingency.to_numpy().ravel())
    row_comb = sum(comb2(int(value)) for value in contingency.sum(axis=1))
    col_comb = sum(comb2(int(value)) for value in contingency.sum(axis=0))
    total_comb = comb2(total)
    if total_comb == 0:
        return np.nan

    expected = (row_comb * col_comb) / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denominator = max_index - expected
    if denominator == 0:
        return np.nan
    return float((sum_comb - expected) / denominator)


def comb2(value: int) -> int:
    if value < 2:
        return 0
    return value * (value - 1) // 2


def add_cluster_projection(frame: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    centered = X - X.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    projection = centered @ components.T
    projected = frame.copy()
    projected["cluster_pca_1"] = projection[:, 0]
    projected["cluster_pca_2"] = projection[:, 1] if projection.shape[1] > 1 else 0.0
    return projected


def build_cluster_summary(cluster_frame: pd.DataFrame, model: dict) -> pd.DataFrame:
    total_eth_pnl = cluster_frame["closingPnl"].sum()
    rows = []

    for cluster_raw, group in cluster_frame.groupby("cluster_raw"):
        trade_count = len(group)
        long_count = int((group["side"] == "Long").sum())
        short_count = int((group["side"] == "Short").sum())
        total_pnl = float(group["closingPnl"].sum())
        row = {
            "cluster_raw": int(cluster_raw),
            "trade_count": trade_count,
            "trade_share": safe_divide(trade_count, len(cluster_frame)),
            "long_count": long_count,
            "short_count": short_count,
            "long_share": safe_divide(long_count, trade_count),
            "short_share": safe_divide(short_count, trade_count),
            "total_pnl": total_pnl,
            "pnl_share_total_eth": safe_divide(total_pnl, total_eth_pnl),
            "avg_pnl": group["closingPnl"].mean(),
            "median_pnl": group["closingPnl"].median(),
            "win_rate": safe_divide((group["closingPnl"] > 0).sum(), group["closingPnl"].notna().sum()),
            "avg_holding_time_hours": group["holding_hours"].mean(),
            "median_realized_return_pct": group["realized_return_pct"].median(),
            "median_MAE_pct": group["MAE_pct"].median(),
            "median_MFE_pct": group["MFE_pct"].median(),
            "median_captured_share_of_MFE": group["captured_share_of_MFE"].median(),
            "strong_post_entry_reaction_share": group["group_fast_positive_reaction"].astype(float).mean(),
            "pct_profitable_after_1m": (group["return_after_1m_pct"] > 0).astype(float).mean(),
            "pct_profitable_after_5m": (group["return_after_5m_pct"] > 0).astype(float).mean(),
            "pct_profitable_after_15m": (group["return_after_15m_pct"] > 0).astype(float).mean(),
            "pct_profitable_after_1h": (group["return_after_60m_pct"] > 0).astype(float).mean(),
            "median_open_hour_utc": group["open_hour_utc"].median(),
            "silhouette_score": model["silhouette_score"],
            "stability_score": model["stability_score"],
            "interpretability_score": model["interpretability_score"],
            "model_score": model["model_score"],
            "selected_k": model["k"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["replication_score"] = (
        summary["pnl_share_total_eth"].clip(lower=0).fillna(0) * 0.55
        + summary["trade_share"].fillna(0) * 0.20
        + summary["win_rate"].fillna(0) * 0.10
        + summary["strong_post_entry_reaction_share"].fillna(0) * 0.10
        + summary["pct_profitable_after_15m"].fillna(0) * 0.05
    )
    return summary


def relabel_clusters_by_total_pnl(summary: pd.DataFrame) -> dict[int, int]:
    ordered = (
        summary.sort_values(["total_pnl", "trade_count"], ascending=[False, False])["cluster_raw"]
        .astype(int)
        .tolist()
    )
    return {raw_cluster: index + 1 for index, raw_cluster in enumerate(ordered)}


def build_cluster_descriptions(cluster_frame: pd.DataFrame, summary: pd.DataFrame) -> dict[int, str]:
    global_row = cluster_frame.median(numeric_only=True)
    descriptions = {}
    for _, row in summary.iterrows():
        cluster_id = int(row["cluster_id"])
        group = cluster_frame.loc[cluster_frame["cluster_id"] == cluster_id]
        descriptions[cluster_id] = describe_cluster_archetype(group, global_row)
    return descriptions


def describe_cluster_archetype(group: pd.DataFrame, global_row: pd.Series) -> str:
    medians = group.median(numeric_only=True)
    short_share = safe_divide((group["side"] == "Short").sum(), len(group))
    long_share = safe_divide((group["side"] == "Long").sum(), len(group))

    pre5 = medians.get("return_pre_5m_pct", np.nan)
    pre15 = medians.get("return_pre_15m_pct", np.nan)
    relvol = medians.get("relative_volume_5m", np.nan)
    dist_ema = medians.get("distance_to_ema20_pct_15m", np.nan)
    dist_high = medians.get("distance_to_last_4h_high_pct", np.nan)
    dist_low = medians.get("distance_to_last_4h_low_pct", np.nan)

    if short_share >= 0.70:
        if pd.notna(pre15) and pre15 > 0.15:
            base = "short after small bounce"
        elif pd.notna(pre15) and pre15 < -0.20 and pd.notna(dist_ema) and dist_ema < 0:
            base = "continuation short in weak context"
        elif pd.notna(dist_high) and dist_high < 0.70:
            base = "range-edge short near local high"
        else:
            base = "short in mixed ETH context"
    elif long_share >= 0.70:
        if pd.notna(pre15) and pre15 < -0.15:
            base = "long after local flush / reversal"
        elif pd.notna(pre15) and pre15 > 0.10 and pd.notna(pre5) and pre5 <= 0.05:
            base = "continuation long after mild pullback"
        elif pd.notna(dist_low) and dist_low < 0.70:
            base = "range-edge long near local low"
        else:
            base = "long in mixed ETH context"
    else:
        if pd.notna(pre15) and abs(pre15) < 0.15:
            base = "mixed-side mid-range ETH entry"
        elif pd.notna(relvol) and relvol > max(1.6, global_row.get("relative_volume_5m", 1.2) * 1.15):
            base = "mixed-side late momentum entry"
        else:
            base = "mixed-side contextual ETH entry"

    if pd.notna(dist_high) and dist_high < 0.70:
        structure = "near 4h high"
    elif pd.notna(dist_low) and dist_low < 0.70:
        structure = "near 4h low"
    else:
        structure = "away from range edge"

    if pd.notna(relvol) and relvol > max(1.6, global_row.get("relative_volume_5m", 1.2) * 1.10):
        heat = "with heated volume"
    elif pd.notna(relvol) and relvol < min(1.0, global_row.get("relative_volume_5m", 1.2) * 0.90):
        heat = "in calmer volume"
    else:
        heat = "in average volume"

    if pd.notna(dist_ema) and abs(dist_ema) < 0.35:
        ema_context = "close to EMA anchor"
    elif pd.notna(dist_ema) and dist_ema > 0.80:
        ema_context = "while stretched above EMA"
    elif pd.notna(dist_ema) and dist_ema < -0.80:
        ema_context = "while stretched below EMA"
    else:
        ema_context = "with mild EMA extension"

    return f"{base}, {structure}, {heat}, {ema_context}"


def choose_best_cluster(summary: pd.DataFrame) -> int:
    ordered = summary.sort_values(
        ["replication_score", "total_pnl", "trade_count"],
        ascending=[False, False, False],
    )
    return int(ordered.iloc[0]["cluster_id"])


def build_best_cluster_profile(
    cluster_frame: pd.DataFrame,
    best_cluster_id: int,
    feature_bundle: dict,
) -> pd.DataFrame:
    feature_names = feature_bundle["feature_names"]
    global_medians = feature_bundle["original_medians"]
    global_stds = feature_bundle["original_stds"]
    best_cluster = cluster_frame.loc[cluster_frame["cluster_id"] == best_cluster_id]

    rows = []
    for feature_name in feature_names:
        cluster_median = pd.to_numeric(best_cluster[feature_name], errors="coerce").median()
        global_median = global_medians.get(feature_name, np.nan)
        global_std = global_stds.get(feature_name, np.nan)
        z_diff = safe_divide(cluster_median - global_median, global_std)
        rows.append(
            {
                "cluster_id": best_cluster_id,
                "feature_key": feature_name,
                "feature_label": CLUSTER_PLOT_LABELS.get(feature_name, feature_name),
                "cluster_median": cluster_median,
                "global_median": global_median,
                "z_diff_vs_global": z_diff,
                "direction": (
                    "cluster_higher"
                    if pd.notna(z_diff) and z_diff > 0
                    else "cluster_lower"
                    if pd.notna(z_diff) and z_diff < 0
                    else "flat"
                ),
            }
        )
    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    profile["abs_z_diff"] = profile["z_diff_vs_global"].abs()
    return profile.sort_values(["abs_z_diff", "feature_label"], ascending=[False, True]).reset_index(drop=True)


def build_cluster_descriptions_text(
    summary: pd.DataFrame,
    model: dict,
    best_cluster_profile: pd.DataFrame,
) -> str:
    if summary.empty:
        return "ETH setup clustering was not available.\n"

    top_one_pnl_share = summary.sort_values("total_pnl", ascending=False)["pnl_share_total_eth"].head(1).sum()
    top_two_pnl_share = summary.sort_values("total_pnl", ascending=False)["pnl_share_total_eth"].head(2).sum()
    best_cluster = summary.sort_values("replication_score", ascending=False).iloc[0]
    top_features = best_cluster_profile.head(6)["feature_label"].tolist()
    top_features_text = ", ".join(top_features) if top_features else "n/a"

    lines = [
        "ETH SETUP CLUSTERING",
        "====================",
        "",
        "This section groups ETHUSDT entries by market context at entry. The clusters are unsupervised and do not use final PnL as an input feature.",
        "",
        f"Selected cluster count: {fmt_int(model['k'])}",
        f"Silhouette score: {fmt_number(model['silhouette_score'], 3)}",
        f"Bootstrap stability: {fmt_number(model['stability_score'], 3)}",
        f"Interpretability score: {fmt_number(model['interpretability_score'], 3)}",
        f"Top-1 cluster share of ETH PnL: {pct(top_one_pnl_share)}",
        f"Top-2 cluster share of ETH PnL: {pct(top_two_pnl_share)}",
        "",
        "Cluster descriptions:",
    ]

    for _, row in summary.sort_values("cluster_id").iterrows():
        lines.extend(
            [
                (
                    f"- Cluster {fmt_int(row['cluster_id'])}: {row['cluster_description']} | "
                    f"trades {fmt_int(row['trade_count'])}, total pnl {fmt_number(row['total_pnl'])}, "
                    f"win rate {pct(row['win_rate'])}, strong reaction {pct(row['strong_post_entry_reaction_share'])}"
                )
            ]
        )

    lines.extend(
        [
            "",
            "Best cluster candidate:",
            (
                f"- Cluster {fmt_int(best_cluster['cluster_id'])} looks like the main replication archetype. "
                f"It contributes {pct(best_cluster['pnl_share_total_eth'])} of ETH PnL across {fmt_int(best_cluster['trade_count'])} trades."
            ),
            f"- Description: {best_cluster['cluster_description']}",
            f"- Dominant differentiating features: {top_features_text}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary_section(
    summary: pd.DataFrame,
    model: dict,
    best_cluster_profile: pd.DataFrame,
) -> str:
    if summary.empty:
        return "ETH SETUP CLUSTERING\n====================\nNo sufficient ETHUSDT data was available for ETH setup clustering.\n"

    top_one_pnl_share = summary.sort_values("total_pnl", ascending=False)["pnl_share_total_eth"].head(1).sum()
    positive_clusters = summary.loc[summary["total_pnl"] > 0].sort_values("total_pnl", ascending=False)
    top_two_pnl_share = positive_clusters["pnl_share_total_eth"].head(2).sum() if not positive_clusters.empty else np.nan
    best_cluster = summary.sort_values("replication_score", ascending=False).iloc[0]
    best_cluster_by_pnl = summary.sort_values("total_pnl", ascending=False).iloc[0]
    best_features = best_cluster_profile.head(5)
    feature_text = "; ".join(
        f"{row['feature_label']} ({row['direction']}, z {fmt_number(row['z_diff_vs_global'])})"
        for _, row in best_features.iterrows()
    )

    if pd.notna(top_two_pnl_share) and top_two_pnl_share >= 0.70:
        dominance = "One or two ETH setup clusters account for most ETH profits."
    elif pd.notna(top_two_pnl_share) and top_two_pnl_share >= 0.50:
        dominance = "A few ETH setup clusters contribute a meaningful share of ETH profits, but the edge is not fully concentrated."
    else:
        dominance = "ETH profits are spread across several context clusters rather than dominated by a single archetype."

    confidence = cluster_confidence_label(summary, model)
    side_mix = (
        "The best cluster is largely side-specific."
        if best_cluster["long_share"] >= 0.70 or best_cluster["short_share"] >= 0.70
        else "The best cluster is more mixed across Long and Short."
    )

    section = f"""
ETH SETUP CLUSTERING
====================

This section groups ETHUSDT entries by repeated market context, using unsupervised clustering on pre-entry features only. Final PnL is used only after clustering to judge which setup archetypes matter most.

Selected clustering model:
- Chosen k: {fmt_int(model['k'])}
- Silhouette score: {fmt_number(model['silhouette_score'], 3)}
- Bootstrap stability: {fmt_number(model['stability_score'], 3)}
- Interpretability score: {fmt_number(model['interpretability_score'], 3)}

Cluster takeaways:
- Estimated number of reasonably stable ETH setup archetypes: {fmt_int(model['k'])}
- Highest-PnL cluster: Cluster {fmt_int(best_cluster_by_pnl['cluster_id'])} with {fmt_number(best_cluster_by_pnl['total_pnl'])} total PnL across {fmt_int(best_cluster_by_pnl['trade_count'])} trades, or {pct(top_one_pnl_share)} of ETH PnL.
- Most reproducible cluster candidate: Cluster {fmt_int(best_cluster['cluster_id'])} with description: {best_cluster['cluster_description']}.
- Top 2 clusters account for {pct(top_two_pnl_share)} of ETH PnL.
- {dominance}
- {side_mix}

Replication hypothesis:
- The cluster that best looks like a replication archetype is Cluster {fmt_int(best_cluster['cluster_id'])}.
- Candidate setup archetype: {best_cluster['cluster_description']}.
- Most characteristic features versus ETH baseline: {feature_text}.
- This is a context-based replication hypothesis, not a claim that the trader uses any specific indicator rule.

Human answer:
- ETH entries naturally group into a small number of recurring context types rather than looking fully random.
- The setup that gives the most useful replication lead is Cluster {fmt_int(best_cluster['cluster_id'])}, because it combines positive PnL contribution with non-trivial trade count and decent early trade behavior.
- This cluster is the best basis for further replication work, but it should still be tested as a context archetype rather than treated as a finished strategy.
- Confidence level: {confidence}
""".strip()
    return section + "\n"


def cluster_confidence_label(summary: pd.DataFrame, model: dict) -> str:
    if summary.empty:
        return "low"
    min_cluster = int(summary["trade_count"].min())
    if model["silhouette_score"] >= 0.22 and model["stability_score"] >= 0.50 and min_cluster >= 8:
        return "high"
    if model["silhouette_score"] >= 0.14 and model["stability_score"] >= 0.35 and min_cluster >= 4:
        return "medium"
    return "low"


def create_eth_setup_cluster_plots(
    cluster_frame: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if cluster_frame.empty or cluster_summary.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(int(cluster_summary["cluster_id"].max()), 3)))
    for cluster_id in sorted(cluster_summary["cluster_id"].unique()):
        cluster_points = cluster_frame.loc[cluster_frame["cluster_id"] == cluster_id]
        for side_label, marker in [("Long", "o"), ("Short", "^")]:
            side_points = cluster_points.loc[cluster_points["side"] == side_label]
            if side_points.empty:
                continue
            ax.scatter(
                side_points["cluster_pca_1"],
                side_points["cluster_pca_2"],
                label=f"Cluster {cluster_id} {side_label}",
                alpha=0.7,
                s=40,
                marker=marker,
                color=colors[int(cluster_id) - 1],
            )
        center_x = cluster_points["cluster_pca_1"].mean()
        center_y = cluster_points["cluster_pca_2"].mean()
        ax.text(center_x, center_y, f"C{cluster_id}", fontsize=10, fontweight="bold")

    ax.set_title("ETH Setup Cluster Projection")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    projection_path = plots_dir / "eth_cluster_projection.png"
    fig.tight_layout()
    fig.savefig(projection_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", projection_path.name)

    heatmap_source = build_heatmap_source(cluster_frame, cluster_summary)
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(heatmap_source["values"], aspect="auto", cmap="coolwarm", vmin=-1.75, vmax=1.75)
    ax.set_title("ETH Cluster Feature Heatmap")
    ax.set_xticks(np.arange(len(heatmap_source["feature_labels"])))
    ax.set_xticklabels(heatmap_source["feature_labels"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heatmap_source["cluster_labels"])))
    ax.set_yticklabels(heatmap_source["cluster_labels"])
    plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    heatmap_path = plots_dir / "eth_cluster_feature_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", heatmap_path.name)


def build_heatmap_source(cluster_frame: pd.DataFrame, cluster_summary: pd.DataFrame) -> dict:
    feature_names = [column for column in cluster_frame.columns if column in dict(CLUSTER_FEATURE_SPECS)] + ["hour_sin", "hour_cos", "side_short_flag"]
    cluster_feature_matrix = cluster_frame.groupby("cluster_id")[feature_names].median()
    standardized = (cluster_feature_matrix - cluster_feature_matrix.mean()) / cluster_feature_matrix.std(ddof=0).replace(0, np.nan)
    standardized = standardized.fillna(0.0)

    feature_rank = standardized.std(axis=0).sort_values(ascending=False)
    selected_features = feature_rank.head(min(12, len(feature_rank))).index.tolist()
    values = standardized[selected_features].to_numpy(dtype=float)
    feature_labels = [CLUSTER_PLOT_LABELS.get(feature, feature) for feature in selected_features]
    cluster_labels = [f"Cluster {int(cluster_id)}" for cluster_id in cluster_feature_matrix.index]
    return {"values": values, "feature_labels": feature_labels, "cluster_labels": cluster_labels}
