from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eth_short_candidate_engine import (
    BASE_DIR,
    CANDLES_1M_PATH,
    DELTA_SCALES,
    JSON_TRADES_PATH,
    MICRO_FEATURES,
    MICRO_WEIGHTS,
    TARGET_DELTAS,
    build_features,
    generate_candidates,
    load_candles,
    load_reference_trades,
    match_candidates_to_reference,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_candidate_engine_v2"
ZONE_GAP_MINUTES = 15
TOP_K_PER_ZONE_VALUES = (1, 2)
LOCAL_RERANK_BLEND = 0.65


def assign_zone_ids(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()

    ranked = candidates.sort_values("open_time").reset_index(drop=True).copy()
    minute_gap = ranked["open_time"].diff().dt.total_seconds().div(60.0)
    ranked["zone_id"] = (minute_gap.isna() | (minute_gap > ZONE_GAP_MINUTES)).cumsum().astype(int)
    ranked["minute_gap_from_prev"] = minute_gap.fillna(np.nan)
    return ranked


def compute_local_rerank_score(zone: pd.DataFrame) -> pd.Series:
    if zone.empty:
        return pd.Series(dtype=float)
    if len(zone) == 1:
        return pd.Series([float(zone["candidate_score"].iloc[0])], index=zone.index, dtype=float)

    med = zone[MICRO_FEATURES].median(numeric_only=True)
    spread = (zone[MICRO_FEATURES].quantile(0.75) - zone[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(zone[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

    weighted_sum = pd.Series(0.0, index=zone.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        current_delta = (pd.to_numeric(zone[feature], errors="coerce") - float(med[feature])) / float(spread[feature])
        target = float(TARGET_DELTAS.get(feature, 0.0))
        scale = float(DELTA_SCALES.get(feature, 1.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        deviation = (current_delta - target).abs() / scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = float(MICRO_WEIGHTS.get(feature, 1.0))
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.nan, index=zone.index)
    return weighted_sum / total_weight


def rerank_candidates_by_zone(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()

    ranked = assign_zone_ids(candidates)
    zone_rows: list[pd.DataFrame] = []
    zone_summary_rows: list[dict] = []

    for zone_id, zone in ranked.groupby("zone_id", sort=True):
        zone = zone.copy()
        zone["local_rerank_score"] = compute_local_rerank_score(zone)
        zone["blended_zone_score"] = (
            zone["local_rerank_score"] * LOCAL_RERANK_BLEND
            + zone["candidate_score"] * (1.0 - LOCAL_RERANK_BLEND)
        )
        zone = zone.sort_values(
            ["blended_zone_score", "candidate_score", "open_time"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        zone["zone_rank"] = np.arange(1, len(zone) + 1)
        zone["zone_size"] = len(zone)
        zone["zone_start_time"] = zone["open_time"].min()
        zone["zone_end_time"] = zone["open_time"].max()
        zone_rows.append(zone)

        top_row = zone.iloc[0]
        zone_summary_rows.append(
            {
                "zone_id": int(zone_id),
                "zone_size": int(len(zone)),
                "zone_start_time": zone["open_time"].min(),
                "zone_end_time": zone["open_time"].max(),
                "top_candidate_time": top_row["open_time"],
                "top_candidate_score": float(top_row["candidate_score"]),
                "top_local_rerank_score": float(top_row["local_rerank_score"]),
                "top_blended_zone_score": float(top_row["blended_zone_score"]),
            }
        )

    zone_ranked = pd.concat(zone_rows, ignore_index=True)
    zone_summary = pd.DataFrame(zone_summary_rows).sort_values("zone_id").reset_index(drop=True)
    return zone_ranked, zone_summary


def select_top_k_per_zone(zone_ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if zone_ranked.empty:
        return zone_ranked.copy()
    shortlisted = (
        zone_ranked.sort_values(["zone_id", "zone_rank", "open_time"])
        .groupby("zone_id", as_index=False, sort=True)
        .head(top_k)
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    shortlisted["selected_top_k"] = top_k
    return shortlisted


def build_match_stats(matches: pd.DataFrame) -> dict[str, float]:
    if matches.empty:
        return {}
    return {
        "reference_trade_count": float(len(matches)),
        "within_5m_share": float(matches["within_5m"].mean() * 100.0),
        "within_15m_share": float(matches["within_15m"].mean() * 100.0),
        "within_60m_share": float(matches["within_60m"].mean() * 100.0),
        "median_abs_minute_distance": float(matches["abs_minute_distance"].median()),
        "median_abs_price_distance_pct": float(matches["price_distance_pct"].abs().median()),
    }


def write_summary(
    raw_candidates: pd.DataFrame,
    zone_ranked: pd.DataFrame,
    zone_summary: pd.DataFrame,
    topk_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    zone_sizes = zone_summary["zone_size"] if not zone_summary.empty else pd.Series(dtype=float)
    lines = [
        "ETH SHORT CANDIDATE ENGINE V2",
        "=============================",
        "",
        f"raw_candidate_count = {len(raw_candidates)}",
        f"zone_count = {len(zone_summary)}",
        f"zone_gap_minutes = {ZONE_GAP_MINUTES}",
        f"local_rerank_blend = {LOCAL_RERANK_BLEND:.2f}",
    ]

    if not zone_sizes.empty:
        lines.extend(
            [
                f"median_zone_size = {zone_sizes.median():.2f}",
                f"mean_zone_size = {zone_sizes.mean():.2f}",
                "",
            ]
        )
    else:
        lines.append("")

    for top_k, (shortlist, matches) in topk_results.items():
        stats = build_match_stats(matches)
        lines.extend(
            [
                f"top_{top_k}_per_zone_candidate_count = {len(shortlist)}",
            ]
        )
        if stats:
            lines.extend(
                [
                    f"top_{top_k}_within_5m_share = {stats['within_5m_share']:.2f}%",
                    f"top_{top_k}_within_15m_share = {stats['within_15m_share']:.2f}%",
                    f"top_{top_k}_within_60m_share = {stats['within_60m_share']:.2f}%",
                    f"top_{top_k}_median_abs_minute_distance = {stats['median_abs_minute_distance']:.2f}",
                    f"top_{top_k}_median_abs_price_distance_pct = {stats['median_abs_price_distance_pct']:.4f}",
                ]
            )
        lines.append("")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    ref = load_reference_trades(JSON_TRADES_PATH)
    features = build_features(candles)
    raw_candidates = generate_candidates(features)
    zone_ranked, zone_summary = rerank_candidates_by_zone(raw_candidates)

    topk_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for top_k in TOP_K_PER_ZONE_VALUES:
        shortlist = select_top_k_per_zone(zone_ranked, top_k)
        matches = match_candidates_to_reference(shortlist, ref)
        topk_results[top_k] = (shortlist, matches)
        shortlist.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_per_zone.csv", index=False)
        matches.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_matches.csv", index=False)

    features.head(0).to_csv(OUTPUT_DIR / "schema_features.csv", index=False)
    raw_candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    zone_ranked.to_csv(OUTPUT_DIR / "zone_ranked_candidates.csv", index=False)
    zone_summary.to_csv(OUTPUT_DIR / "zone_summary.csv", index=False)
    write_summary(raw_candidates, zone_ranked, zone_summary, topk_results)

    print(f"Raw candidates: {len(raw_candidates)}")
    print(f"Zones: {len(zone_summary)}")
    for top_k, (shortlist, matches) in topk_results.items():
        print(f"Top-{top_k} per zone candidates: {len(shortlist)}")
        if not matches.empty:
            print(f"Top-{top_k} within 15m share: {matches['within_15m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
