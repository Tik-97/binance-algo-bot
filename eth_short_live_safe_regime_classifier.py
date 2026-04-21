from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ethShortTrader import BASE_DIR, CANDLES_1M_PATH, build_features, load_candles
from eth_short_execution_backtest_v1 import NEAREST_CANDIDATES_PATH, load_nearest_candidates


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_live_safe_regime_classifier"
ORACLE_REGIME_PATH = BASE_DIR / "fun_backtests" / "eth_short_day_regime_split_analysis" / "day_regime_tagged.csv"

# Live-safe target regime detection tuned from the labeled session days.
BREAKDOWN_RELATIVE_VOLUME_MIN = 1.35
BREAKDOWN_MOVE_5M_MIN = 0.55
BREAKDOWN_BULL_BODY_SUM_5M_MIN = 0.65
BREAKDOWN_STRETCH_EMA20_5M_MIN = 0.90
BREAKDOWN_GREEN_COUNT_5M_MIN = 4

MIXED_RELATIVE_VOLUME_MAX = 1.20
MIXED_BOUNCE_60M_MAX = 1.50
MIXED_STRETCH_EMA20_5M_MAX = 1.40
MIXED_GREEN_COUNT_5M_MAX = 3
MIXED_DIST_1H_HIGH_MIN = 0.10

LIVE_FEATURE_COLUMNS = [
    "relative_volume_1m",
    "move_1m_pct",
    "move_3m_pct",
    "move_5m_pct",
    "body_ratio",
    "upper_wick_ratio",
    "close_position_in_range",
    "range_pct",
    "distance_above_ema7_1m_pct",
    "dist_1h_high_pct",
    "dist_4h_high_pct",
    "bounce_60m_pct",
    "stretch_above_ema20_5m_pct",
    "green_count_last_5",
    "bull_body_sum_5m_pct",
]


def collapse_oracle_regime(label: str) -> str:
    if label in {"mixed_transition", "near_entry_breakdown_noise"}:
        return label
    return "other_non_target"


def predict_live_safe_day_regime_from_bar(bar: pd.Series) -> str:
    relative_volume_1m = float(bar["relative_volume_1m"])
    move_5m_pct = float(bar["move_5m_pct"])
    bounce_60m_pct = float(bar["bounce_60m_pct"])
    stretch_above_ema20_5m_pct = float(bar["stretch_above_ema20_5m_pct"])
    green_count_last_5 = float(bar["green_count_last_5"])
    bull_body_sum_5m_pct = float(bar["bull_body_sum_5m_pct"])
    dist_1h_high_pct = float(bar["dist_1h_high_pct"])

    is_breakdown_noise = (
        relative_volume_1m >= BREAKDOWN_RELATIVE_VOLUME_MIN
        and move_5m_pct >= BREAKDOWN_MOVE_5M_MIN
        and bull_body_sum_5m_pct >= BREAKDOWN_BULL_BODY_SUM_5M_MIN
        and stretch_above_ema20_5m_pct >= BREAKDOWN_STRETCH_EMA20_5M_MIN
        and green_count_last_5 >= BREAKDOWN_GREEN_COUNT_5M_MIN
    )
    if is_breakdown_noise:
        return "near_entry_breakdown_noise"

    is_mixed_transition = (
        relative_volume_1m <= MIXED_RELATIVE_VOLUME_MAX
        and bounce_60m_pct <= MIXED_BOUNCE_60M_MAX
        and stretch_above_ema20_5m_pct <= MIXED_STRETCH_EMA20_5M_MAX
        and green_count_last_5 <= MIXED_GREEN_COUNT_5M_MAX
        and dist_1h_high_pct >= MIXED_DIST_1H_HIGH_MIN
    )
    if is_mixed_transition:
        return "mixed_transition"

    return "other_non_target"


def load_oracle_regimes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["candidate_time"] = pd.to_datetime(df["candidate_time"], utc=True, errors="coerce")
    df["oracle_regime"] = df["day_regime"].astype(str)
    df["oracle_regime_collapsed"] = df["oracle_regime"].map(collapse_oracle_regime)
    return df[["trade_date", "candidate_time", "oracle_regime", "oracle_regime_collapsed"]].dropna().drop_duplicates("trade_date")


def build_labeled_classifier_frame() -> pd.DataFrame:
    oracle = load_oracle_regimes(ORACLE_REGIME_PATH)
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH)[["trade_date", "candidate_time", "candidate_score"]].copy()
    features = build_features(load_candles(CANDLES_1M_PATH)).copy()
    features["open_time"] = pd.to_datetime(features["open_time"], utc=True)

    frame = oracle.merge(nearest_candidates, on=["trade_date", "candidate_time"], how="left")
    frame = frame.merge(features, left_on="candidate_time", right_on="open_time", how="left")
    frame["predicted_regime"] = frame.apply(predict_live_safe_day_regime_from_bar, axis=1)
    frame["predicted_is_target"] = frame["predicted_regime"].isin({"mixed_transition", "near_entry_breakdown_noise"})
    frame["oracle_is_target"] = frame["oracle_regime_collapsed"] != "other_non_target"
    return frame.sort_values("trade_date").reset_index(drop=True)


def build_confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    confusion = (
        df.groupby(["oracle_regime_collapsed", "predicted_regime"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["oracle_regime_collapsed", "predicted_regime"])
        .reset_index(drop=True)
    )
    return confusion


def build_summary(df: pd.DataFrame) -> dict:
    correct = (df["oracle_regime_collapsed"] == df["predicted_regime"]).mean() * 100.0 if len(df) else np.nan

    true_positive = int((df["oracle_is_target"] & df["predicted_is_target"]).sum())
    false_positive = int((~df["oracle_is_target"] & df["predicted_is_target"]).sum())
    false_negative = int((df["oracle_is_target"] & ~df["predicted_is_target"]).sum())

    precision = true_positive / (true_positive + false_positive) * 100.0 if (true_positive + false_positive) else np.nan
    recall = true_positive / (true_positive + false_negative) * 100.0 if (true_positive + false_negative) else np.nan

    return {
        "trade_count": int(len(df)),
        "overall_accuracy_pct": float(correct),
        "target_precision_pct": float(precision) if pd.notna(precision) else np.nan,
        "target_recall_pct": float(recall) if pd.notna(recall) else np.nan,
        "predicted_target_count": int(df["predicted_is_target"].sum()),
        "oracle_target_count": int(df["oracle_is_target"].sum()),
    }


def write_summary(summary: dict, df: pd.DataFrame, confusion: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT LIVE-SAFE REGIME CLASSIFIER",
        "=====================================",
        "",
        "predictions = mixed_transition / near_entry_breakdown_noise / other_non_target",
        "note = top-level regime prediction uses only candidate-bar features, no trader-gap oracle fields",
        f"trade_count = {summary['trade_count']}",
        f"overall_accuracy_pct = {summary['overall_accuracy_pct']:.2f}",
        f"target_precision_pct = {summary['target_precision_pct']:.2f}" if pd.notna(summary["target_precision_pct"]) else "target_precision_pct = nan",
        f"target_recall_pct = {summary['target_recall_pct']:.2f}" if pd.notna(summary["target_recall_pct"]) else "target_recall_pct = nan",
        f"predicted_target_count = {summary['predicted_target_count']}",
        f"oracle_target_count = {summary['oracle_target_count']}",
        "",
        "Rules",
        "-----",
        f"near_entry_breakdown_noise if rel_volume >= {BREAKDOWN_RELATIVE_VOLUME_MIN:.2f}, move_5m >= {BREAKDOWN_MOVE_5M_MIN:.2f}, bull_body_sum_5m >= {BREAKDOWN_BULL_BODY_SUM_5M_MIN:.2f}, stretch_ema20_5m >= {BREAKDOWN_STRETCH_EMA20_5M_MIN:.2f}, green_count_last_5 >= {BREAKDOWN_GREEN_COUNT_5M_MIN}",
        f"mixed_transition if rel_volume <= {MIXED_RELATIVE_VOLUME_MAX:.2f}, bounce_60m <= {MIXED_BOUNCE_60M_MAX:.2f}, stretch_ema20_5m <= {MIXED_STRETCH_EMA20_5M_MAX:.2f}, green_count_last_5 <= {MIXED_GREEN_COUNT_5M_MAX}, dist_1h_high >= {MIXED_DIST_1H_HIGH_MIN:.2f}",
        "",
        "Confusion",
        "---------",
    ]

    for _, row in confusion.iterrows():
        lines.append(
            f"{row['oracle_regime_collapsed']} -> {row['predicted_regime']}: {int(row['count'])}"
        )

    lines.extend(["", "Predictions", "-----------"])
    for _, row in df.iterrows():
        lines.append(
            f"{row['trade_date']}: oracle={row['oracle_regime_collapsed']} predicted={row['predicted_regime']} candidate_score={row['candidate_score']:.4f}"
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_labeled_classifier_frame()
    confusion = build_confusion_table(df)
    summary = build_summary(df)

    export_cols = [
        "trade_date",
        "oracle_regime",
        "oracle_regime_collapsed",
        "predicted_regime",
        "predicted_is_target",
        "candidate_score",
        *LIVE_FEATURE_COLUMNS,
    ]
    df[export_cols].to_csv(OUTPUT_DIR / "regime_classifier_predictions.csv", index=False)
    confusion.to_csv(OUTPUT_DIR / "confusion.csv", index=False)
    write_summary(summary, df, confusion)

    print(f"Trade count: {len(df)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
