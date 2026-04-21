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


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_candidate_engine_v4"
SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16
TRAIN_POSITIVE_MAX_MINUTES = 15.0
TOP_K_VALUES = (1, 3, 5)
BLEND_WEIGHT = 0.40


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


def attach_day_context(candidates: pd.DataFrame, session_ref: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, trade in session_ref.iterrows():
        day = candidates.loc[candidates["trade_date"].eq(trade["trade_date"])].copy()
        if day.empty:
            continue
        day["trade_opened_time"] = trade["opened_time"]
        day["trade_avg_cost"] = trade["avgCost"]
        day["minute_distance_to_trade"] = (day["open_time"] - trade["opened_time"]).dt.total_seconds() / 60.0
        day["abs_minute_distance_to_trade"] = day["minute_distance_to_trade"].abs()
        day["is_positive_anchor"] = day["abs_minute_distance_to_trade"].le(TRAIN_POSITIVE_MAX_MINUTES)
        rows.append(day)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_day_delta_profile(candidates_with_context: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    rows: list[dict] = []
    if candidates_with_context.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    grouped = candidates_with_context.groupby("trade_opened_time", sort=True)
    for trade_time, day in grouped:
        positives = day.loc[day["is_positive_anchor"]].copy()
        negatives = day.loc[~day["is_positive_anchor"]].copy()
        if positives.empty or negatives.empty:
            continue

        med = negatives[MICRO_FEATURES].median(numeric_only=True)
        spread = (negatives[MICRO_FEATURES].quantile(0.75) - negatives[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
        spread = spread.fillna(negatives[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

        anchor = positives.sort_values(["abs_minute_distance_to_trade", "candidate_score"], ascending=[True, False]).iloc[0]
        row = {
            "trade_opened_time": trade_time,
            "anchor_candidate_time": anchor["open_time"],
            "anchor_abs_minute_distance": float(anchor["abs_minute_distance_to_trade"]),
            "day_candidate_count": int(len(day)),
            "positive_candidate_count": int(len(positives)),
        }
        for feature in MICRO_FEATURES:
            row[f"{feature}_delta"] = (float(anchor[feature]) - float(med[feature])) / float(spread[feature])
        rows.append(row)

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), delta_df

    cols = [f"{feature}_delta" for feature in MICRO_FEATURES]
    target = delta_df[cols].median(numeric_only=True)
    scale = (delta_df[cols].quantile(0.75) - delta_df[cols].quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(delta_df[cols].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
    return target, scale, delta_df


def compute_day_relevance_score(day: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.Series:
    if day.empty:
        return pd.Series(dtype=float)

    negatives = day.copy()
    med = negatives[MICRO_FEATURES].median(numeric_only=True)
    spread = (negatives[MICRO_FEATURES].quantile(0.75) - negatives[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(negatives[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

    weighted_sum = pd.Series(0.0, index=day.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        current_delta = (pd.to_numeric(day[feature], errors="coerce") - float(med[feature])) / float(spread[feature])
        target = float(target_deltas.get(f"{feature}_delta", 0.0))
        scale = float(delta_scales.get(f"{feature}_delta", 1.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        deviation = (current_delta - target).abs() / scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = float(MICRO_WEIGHTS.get(feature, 1.0))
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.nan, index=day.index)
    return weighted_sum / total_weight


def rank_candidates_within_day(candidates_with_context: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trade_time, day in candidates_with_context.groupby("trade_opened_time", sort=True):
        day = day.copy()
        day["day_relevance_score"] = compute_day_relevance_score(day, target_deltas, delta_scales)
        day["blended_score"] = day["day_relevance_score"] * (1.0 - BLEND_WEIGHT) + day["candidate_score"] * BLEND_WEIGHT
        day = day.sort_values(
            ["blended_score", "candidate_score", "open_time"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        day["rank_in_day"] = np.arange(1, len(day) + 1)
        day["day_candidate_count"] = len(day)
        rows.append(day)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_trade_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for trade_time, day in ranked.groupby("trade_opened_time", sort=True):
        best = day.iloc[0]
        anchor = day.sort_values(["abs_minute_distance_to_trade", "candidate_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "trade_opened_time": trade_time,
                "trade_avg_cost": float(best["trade_avg_cost"]),
                "best_candidate_time": best["open_time"],
                "best_candidate_score": float(best["candidate_score"]),
                "best_day_relevance_score": float(best["day_relevance_score"]),
                "best_blended_score": float(best["blended_score"]),
                "best_candidate_abs_minute_distance": float(best["abs_minute_distance_to_trade"]),
                "best_candidate_price_distance_pct": (float(best["close"]) - float(best["trade_avg_cost"])) / float(best["trade_avg_cost"]) * 100.0,
                "anchor_candidate_time": anchor["open_time"],
                "anchor_rank_in_day": int(day.loc[day["open_time"].eq(anchor["open_time"])].iloc[0]["rank_in_day"]),
                "anchor_abs_minute_distance": float(anchor["abs_minute_distance_to_trade"]),
                "anchor_in_top_1": bool(int(day.loc[day["open_time"].eq(anchor["open_time"])].iloc[0]["rank_in_day"]) <= 1),
                "anchor_in_top_3": bool(int(day.loc[day["open_time"].eq(anchor["open_time"])].iloc[0]["rank_in_day"]) <= 3),
                "anchor_in_top_5": bool(int(day.loc[day["open_time"].eq(anchor["open_time"])].iloc[0]["rank_in_day"]) <= 5),
                "best_within_5m": bool(float(best["abs_minute_distance_to_trade"]) <= 5.0),
                "best_within_15m": bool(float(best["abs_minute_distance_to_trade"]) <= 15.0),
                "best_within_60m": bool(float(best["abs_minute_distance_to_trade"]) <= 60.0),
                "day_candidate_count": int(len(day)),
            }
        )
    return pd.DataFrame(rows)


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
    candidates_filtered: pd.DataFrame,
    delta_profile_rows: pd.DataFrame,
    trade_summary: pd.DataFrame,
    shortlist_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    lines = [
        "ETH SHORT CANDIDATE ENGINE V4",
        "=============================",
        "",
        f"session_reference_trade_count = {len(session_ref)}",
        f"candidate_trade_day_count = {candidates_filtered['trade_date'].nunique() if not candidates_filtered.empty else 0}",
        f"train_positive_max_minutes = {TRAIN_POSITIVE_MAX_MINUTES:.0f}",
        f"blend_weight_candidate_score = {BLEND_WEIGHT:.2f}",
        "",
    ]

    if not trade_summary.empty:
        lines.extend(
            [
                "Day Ranking Quality",
                "-------------------",
                f"anchor_top1_share = {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%",
                f"anchor_top3_share = {trade_summary['anchor_in_top_3'].mean() * 100.0:.2f}%",
                f"anchor_top5_share = {trade_summary['anchor_in_top_5'].mean() * 100.0:.2f}%",
                f"best_within_5m_share = {trade_summary['best_within_5m'].mean() * 100.0:.2f}%",
                f"best_within_15m_share = {trade_summary['best_within_15m'].mean() * 100.0:.2f}%",
                f"best_within_60m_share = {trade_summary['best_within_60m'].mean() * 100.0:.2f}%",
                f"median_anchor_rank = {trade_summary['anchor_rank_in_day'].median():.2f}",
                f"median_best_abs_minute_distance = {trade_summary['best_candidate_abs_minute_distance'].median():.2f}",
                "",
            ]
        )

    for top_k, (_, matches) in shortlist_results.items():
        lines.append(f"top_{top_k}_per_day_candidate_count = {len(matches)}")
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

    if not delta_profile_rows.empty:
        lines.extend(["Learned Day Deltas", "------------------"])
        for feature in MICRO_FEATURES:
            col = f"{feature}_delta"
            target = float(delta_profile_rows[col].median())
            spread = float(delta_profile_rows[col].quantile(0.75) - delta_profile_rows[col].quantile(0.25))
            if not np.isfinite(spread) or spread <= 0:
                spread = float(delta_profile_rows[col].std()) if np.isfinite(delta_profile_rows[col].std()) and delta_profile_rows[col].std() > 0 else 1.0
            lines.append(f"{feature}: target_delta={target:.4f}, delta_scale={spread:.4f}, weight={MICRO_WEIGHTS.get(feature, 1.0):.2f}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    ref = load_reference_trades(JSON_TRADES_PATH)
    candidates = generate_candidates(build_features(candles))

    session_ref = filter_session_reference_days(ref)
    candidates_filtered = filter_candidates_to_trade_days(candidates, session_ref)
    candidates_with_context = attach_day_context(candidates_filtered, session_ref)
    target_deltas, delta_scales, delta_profile_rows = build_day_delta_profile(candidates_with_context)
    ranked = rank_candidates_within_day(candidates_with_context, target_deltas, delta_scales)
    trade_summary = build_trade_summary(ranked)

    shortlist_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for top_k in TOP_K_VALUES:
        shortlisted = select_top_k_per_day(ranked, top_k)
        matches = build_shortlist_match_summary(shortlisted)
        shortlist_results[top_k] = (shortlisted, matches)
        shortlisted.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_per_day.csv", index=False)
        matches.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_matches.csv", index=False)

    candidates_filtered.to_csv(OUTPUT_DIR / "session_day_candidates.csv", index=False)
    candidates_with_context.to_csv(OUTPUT_DIR / "session_day_candidates_with_context.csv", index=False)
    delta_profile_rows.to_csv(OUTPUT_DIR / "day_delta_profile_rows.csv", index=False)
    ranked.to_csv(OUTPUT_DIR / "ranked_candidates.csv", index=False)
    trade_summary.to_csv(OUTPUT_DIR / "trade_summary.csv", index=False)
    write_summary(session_ref, candidates_filtered, delta_profile_rows, trade_summary, shortlist_results)

    print(f"Session reference trades: {len(session_ref)}")
    print(f"Candidate trade days: {candidates_filtered['trade_date'].nunique() if not candidates_filtered.empty else 0}")
    if not trade_summary.empty:
        print(f"Anchor top-1 share: {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%")
        print(f"Best within 15m share: {trade_summary['best_within_15m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
