from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eth_short_candidate_engine import (
    BASE_DIR,
    CANDLES_1M_PATH,
    JSON_TRADES_PATH,
    MICRO_FEATURES,
    MICRO_WEIGHTS,
    build_features,
    generate_candidates,
    load_candles,
    load_reference_trades,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_candidate_engine_v5"
SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16
TOP_K_VALUES = (1, 3, 5)
BLEND_WEIGHT = 0.30


def filter_session_reference_days(ref: pd.DataFrame) -> pd.DataFrame:
    session_ref = ref[
        (ref["opened_time"].dt.hour >= SESSION_START_HOUR_UTC)
        & (ref["opened_time"].dt.hour < SESSION_END_HOUR_UTC)
    ].copy()
    session_ref["trade_date"] = session_ref["opened_time"].dt.date
    return session_ref.sort_values("opened_time").reset_index(drop=True)


def filter_candidates_to_trade_days(candidates: pd.DataFrame, session_ref: pd.DataFrame) -> pd.DataFrame:
    filtered = candidates.copy()
    filtered["trade_date"] = filtered["open_time"].dt.date
    valid_days = set(session_ref["trade_date"])
    filtered = filtered.loc[filtered["trade_date"].isin(valid_days)].copy()
    return filtered.sort_values("open_time").reset_index(drop=True)


def build_training_candidates(candidates: pd.DataFrame, session_ref: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, trade in session_ref.iterrows():
        day = candidates.loc[candidates["trade_date"].eq(trade["trade_date"])].copy()
        if day.empty:
            continue
        day["trade_opened_time"] = trade["opened_time"]
        day["trade_avg_cost"] = trade["avgCost"]
        day["trade_date"] = trade["trade_date"]
        day["minute_distance_to_trade"] = (day["open_time"] - trade["opened_time"]).dt.total_seconds() / 60.0
        day["abs_minute_distance_to_trade"] = day["minute_distance_to_trade"].abs()
        day = day.sort_values(["abs_minute_distance_to_trade", "candidate_score", "open_time"], ascending=[True, False, True]).reset_index(drop=True)
        day["rank_by_distance_in_day"] = np.arange(1, len(day) + 1)
        day["is_anchor_candidate"] = False
        day.loc[0, "is_anchor_candidate"] = True
        rows.append(day)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_relative_day_features(day: pd.DataFrame) -> pd.DataFrame:
    enriched = day.copy()
    med = enriched[MICRO_FEATURES].median(numeric_only=True)
    spread = (enriched[MICRO_FEATURES].quantile(0.75) - enriched[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(enriched[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

    for feature in MICRO_FEATURES:
        enriched[f"{feature}_day_delta"] = (pd.to_numeric(enriched[feature], errors="coerce") - float(med[feature])) / float(spread[feature])

    enriched["candidate_score_day_delta"] = (
        enriched["candidate_score"] - float(enriched["candidate_score"].median())
    ) / max(float(enriched["candidate_score"].quantile(0.75) - enriched["candidate_score"].quantile(0.25)), 1e-6)
    enriched["day_candidate_count"] = len(enriched)
    return enriched


def add_relative_features(training_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, day in training_candidates.groupby("trade_opened_time", sort=True):
        rows.append(build_relative_day_features(day))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_training_profile(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    positives = train_df.loc[train_df["is_anchor_candidate"]].copy()
    if positives.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    target = positives[feature_columns].median(numeric_only=True)
    scale = (positives[feature_columns].quantile(0.75) - positives[feature_columns].quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(train_df[feature_columns].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
    return target, scale


def build_feature_weights(feature_columns: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for col in feature_columns:
        if col.endswith("_day_delta"):
            base = col.removesuffix("_day_delta")
            weights[col] = float(MICRO_WEIGHTS.get(base, 1.0))
        elif col == "candidate_score":
            weights[col] = 1.30
        elif col == "candidate_score_day_delta":
            weights[col] = 1.10
        elif col == "day_candidate_count":
            weights[col] = 0.30
        else:
            weights[col] = 1.0
    return weights


def score_day_candidates(
    day: pd.DataFrame,
    target: pd.Series,
    scale: pd.Series,
    feature_columns: list[str],
    feature_weights: dict[str, float],
) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=day.index, dtype=float)
    total_weight = 0.0
    for col in feature_columns:
        weight = float(feature_weights.get(col, 1.0))
        col_scale = float(scale.get(col, 1.0))
        if not np.isfinite(col_scale) or col_scale <= 0:
            col_scale = 1.0
        deviation = (pd.to_numeric(day[col], errors="coerce") - float(target.get(col, 0.0))).abs() / col_scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight
    if total_weight <= 0:
        return pd.Series(np.nan, index=day.index)
    return weighted_sum / total_weight


def run_leave_one_day_out(training_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if training_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_columns = [f"{feature}_day_delta" for feature in MICRO_FEATURES] + [
        "candidate_score",
        "candidate_score_day_delta",
    ]
    feature_weights = build_feature_weights(feature_columns)

    ranked_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for trade_time, holdout_day in training_df.groupby("trade_opened_time", sort=True):
        train_df = training_df.loc[~training_df["trade_opened_time"].eq(trade_time)].copy()
        target, scale = build_training_profile(train_df, feature_columns)
        scored = holdout_day.copy()
        scored["supervised_score"] = score_day_candidates(scored, target, scale, feature_columns, feature_weights)
        scored["blended_score"] = scored["supervised_score"] * (1.0 - BLEND_WEIGHT) + scored["candidate_score"] * BLEND_WEIGHT
        scored = scored.sort_values(["blended_score", "candidate_score", "open_time"], ascending=[False, False, True]).reset_index(drop=True)
        scored["rank_in_day"] = np.arange(1, len(scored) + 1)
        ranked_rows.append(scored)

        anchor_row = scored.loc[scored["is_anchor_candidate"]].iloc[0]
        best_row = scored.iloc[0]
        top5 = scored.head(5)
        summary_rows.append(
            {
                "trade_opened_time": trade_time,
                "anchor_candidate_time": anchor_row["open_time"],
                "anchor_abs_minute_distance": float(anchor_row["abs_minute_distance_to_trade"]),
                "anchor_rank_in_day": int(anchor_row["rank_in_day"]),
                "anchor_in_top_1": bool(int(anchor_row["rank_in_day"]) <= 1),
                "anchor_in_top_3": bool(int(anchor_row["rank_in_day"]) <= 3),
                "anchor_in_top_5": bool(int(anchor_row["rank_in_day"]) <= 5),
                "best_candidate_time": best_row["open_time"],
                "best_abs_minute_distance": float(best_row["abs_minute_distance_to_trade"]),
                "best_price_distance_pct": (float(best_row["close"]) - float(best_row["trade_avg_cost"])) / float(best_row["trade_avg_cost"]) * 100.0,
                "best_within_5m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 5.0),
                "best_within_15m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 15.0),
                "best_within_60m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 60.0),
                "top5_contains_within_5m": bool((top5["abs_minute_distance_to_trade"] <= 5.0).any()),
                "top5_contains_within_15m": bool((top5["abs_minute_distance_to_trade"] <= 15.0).any()),
                "day_candidate_count": int(len(scored)),
            }
        )

    ranked = pd.concat(ranked_rows, ignore_index=True) if ranked_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return ranked, summary


def select_top_k_per_day(ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
    shortlisted = (
        ranked.sort_values(["trade_opened_time", "rank_in_day", "open_time"])
        .groupby("trade_opened_time", as_index=False, sort=True)
        .head(top_k)
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    shortlisted["selected_top_k"] = top_k
    return shortlisted


def build_shortlist_match_summary(shortlisted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trade_time, day in shortlisted.groupby("trade_opened_time", sort=True):
        best = day.sort_values(["abs_minute_distance_to_trade", "candidate_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "trade_opened_time": trade_time,
                "candidate_time": best["open_time"],
                "abs_minute_distance": float(best["abs_minute_distance_to_trade"]),
                "price_distance_pct": (float(best["close"]) - float(best["trade_avg_cost"])) / float(best["trade_avg_cost"]) * 100.0,
                "within_5m": bool(float(best["abs_minute_distance_to_trade"]) <= 5.0),
                "within_15m": bool(float(best["abs_minute_distance_to_trade"]) <= 15.0),
                "within_60m": bool(float(best["abs_minute_distance_to_trade"]) <= 60.0),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    session_ref: pd.DataFrame,
    training_df: pd.DataFrame,
    summary: pd.DataFrame,
    shortlist_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    lines = [
        "ETH SHORT CANDIDATE ENGINE V5",
        "=============================",
        "",
        f"session_reference_trade_count = {len(session_ref)}",
        f"training_candidate_row_count = {len(training_df)}",
        f"blend_weight_candidate_score = {BLEND_WEIGHT:.2f}",
        "",
    ]

    if not summary.empty:
        lines.extend(
            [
                "Leave-One-Day-Out Ranking Quality",
                "---------------------------------",
                f"anchor_top1_share = {summary['anchor_in_top_1'].mean() * 100.0:.2f}%",
                f"anchor_top3_share = {summary['anchor_in_top_3'].mean() * 100.0:.2f}%",
                f"anchor_top5_share = {summary['anchor_in_top_5'].mean() * 100.0:.2f}%",
                f"best_within_5m_share = {summary['best_within_5m'].mean() * 100.0:.2f}%",
                f"best_within_15m_share = {summary['best_within_15m'].mean() * 100.0:.2f}%",
                f"best_within_60m_share = {summary['best_within_60m'].mean() * 100.0:.2f}%",
                f"top5_contains_within_5m_share = {summary['top5_contains_within_5m'].mean() * 100.0:.2f}%",
                f"top5_contains_within_15m_share = {summary['top5_contains_within_15m'].mean() * 100.0:.2f}%",
                f"median_anchor_rank = {summary['anchor_rank_in_day'].median():.2f}",
                f"median_best_abs_minute_distance = {summary['best_abs_minute_distance'].median():.2f}",
                "",
            ]
        )

    for top_k, (_, matches) in shortlist_results.items():
        lines.append(f"top_{top_k}_per_day_trade_count = {len(matches)}")
        if not matches.empty:
            lines.extend(
                [
                    f"top_{top_k}_within_5m_share = {matches['within_5m'].mean() * 100.0:.2f}%",
                    f"top_{top_k}_within_15m_share = {matches['within_15m'].mean() * 100.0:.2f}%",
                    f"top_{top_k}_within_60m_share = {matches['within_60m'].mean() * 100.0:.2f}%",
                    f"top_{top_k}_median_abs_minute_distance = {matches['abs_minute_distance'].median():.2f}",
                    f"top_{top_k}_median_abs_price_distance_pct = {matches['price_distance_pct'].abs().median():.4f}",
                ]
            )
        lines.append("")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    ref = load_reference_trades(JSON_TRADES_PATH)
    candidates = generate_candidates(build_features(candles))

    session_ref = filter_session_reference_days(ref)
    candidates_filtered = filter_candidates_to_trade_days(candidates, session_ref)
    training_candidates = build_training_candidates(candidates_filtered, session_ref)
    training_candidates = add_relative_features(training_candidates)
    ranked, summary = run_leave_one_day_out(training_candidates)

    shortlist_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for top_k in TOP_K_VALUES:
        shortlisted = select_top_k_per_day(ranked, top_k)
        matches = build_shortlist_match_summary(shortlisted)
        shortlist_results[top_k] = (shortlisted, matches)
        shortlisted.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_per_day.csv", index=False)
        matches.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_matches.csv", index=False)

    training_candidates.to_csv(OUTPUT_DIR / "training_candidates.csv", index=False)
    ranked.to_csv(OUTPUT_DIR / "ranked_candidates.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "trade_summary.csv", index=False)
    write_summary(session_ref, training_candidates, summary, shortlist_results)

    print(f"Session reference trades: {len(session_ref)}")
    print(f"Training candidate rows: {len(training_candidates)}")
    if not summary.empty:
        print(f"Anchor top-1 share: {summary['anchor_in_top_1'].mean() * 100.0:.2f}%")
        print(f"Best within 15m share: {summary['best_within_15m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
