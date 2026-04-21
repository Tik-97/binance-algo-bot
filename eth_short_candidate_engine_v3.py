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
    match_candidates_to_reference,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_candidate_engine_v3"
ZONE_GAP_MINUTES = 15
POSITIVE_MATCH_MAX_MINUTES = 60.0
TOP_K_PER_ZONE_VALUES = (1, 2)
BLEND_WEIGHT = 0.35


def assign_zone_ids(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.sort_values("open_time").reset_index(drop=True).copy()
    minute_gap = ranked["open_time"].diff().dt.total_seconds().div(60.0)
    ranked["zone_id"] = (minute_gap.isna() | (minute_gap > ZONE_GAP_MINUTES)).cumsum().astype(int)
    ranked["minute_gap_from_prev"] = minute_gap
    return ranked


def attach_reference_matches(candidates: pd.DataFrame, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty or ref.empty:
        return candidates.copy(), pd.DataFrame()

    rows: list[dict] = []
    tagged = candidates.copy()
    tagged["is_positive_anchor"] = False
    tagged["positive_trade_opened_time"] = pd.Series(pd.NaT, index=tagged.index, dtype="datetime64[ns, UTC]")
    tagged["positive_abs_minute_distance"] = np.nan

    used_indices: set[int] = set()
    for _, trade in ref.iterrows():
        window = tagged.copy()
        window["minute_distance"] = (window["open_time"] - trade["opened_time"]).dt.total_seconds() / 60.0
        window["abs_minute_distance"] = window["minute_distance"].abs()
        if window.empty:
            continue
        best = window.sort_values(["abs_minute_distance", "candidate_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "trade_opened_time": trade["opened_time"],
                "trade_avg_cost": trade["avgCost"],
                "matched_candidate_index": int(best.name),
                "matched_candidate_time": best["open_time"],
                "matched_zone_id": int(best["zone_id"]),
                "matched_candidate_score": float(best["candidate_score"]),
                "minute_distance": float(best["minute_distance"]),
                "abs_minute_distance": float(best["abs_minute_distance"]),
                "is_positive_match": bool(float(best["abs_minute_distance"]) <= POSITIVE_MATCH_MAX_MINUTES),
            }
        )
        if float(best["abs_minute_distance"]) <= POSITIVE_MATCH_MAX_MINUTES and int(best.name) not in used_indices:
            tagged.loc[best.name, "is_positive_anchor"] = True
            tagged.loc[best.name, "positive_trade_opened_time"] = trade["opened_time"]
            tagged.loc[best.name, "positive_abs_minute_distance"] = float(best["abs_minute_distance"])
            used_indices.add(int(best.name))

    return tagged, pd.DataFrame(rows)


def build_positive_delta_profile(zone_ranked: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    rows: list[dict] = []

    positives = zone_ranked.loc[zone_ranked["is_positive_anchor"]].copy()
    for _, anchor in positives.iterrows():
        zone = zone_ranked.loc[zone_ranked["zone_id"].eq(anchor["zone_id"])].copy()
        negatives = zone.loc[~zone.index.isin([anchor.name])].copy()
        if negatives.empty:
            continue

        med = negatives[MICRO_FEATURES].median(numeric_only=True)
        spread = (negatives[MICRO_FEATURES].quantile(0.75) - negatives[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
        spread = spread.fillna(negatives[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

        row = {
            "zone_id": int(anchor["zone_id"]),
            "anchor_time": anchor["open_time"],
            "positive_abs_minute_distance": float(anchor["positive_abs_minute_distance"]),
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


def compute_zone_contrast_score(zone: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.Series:
    if zone.empty:
        return pd.Series(dtype=float)
    if len(zone) == 1:
        return pd.Series([1.0], index=zone.index, dtype=float)

    med = zone[MICRO_FEATURES].median(numeric_only=True)
    spread = (zone[MICRO_FEATURES].quantile(0.75) - zone[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(zone[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)

    weighted_sum = pd.Series(0.0, index=zone.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        current_delta = (pd.to_numeric(zone[feature], errors="coerce") - float(med[feature])) / float(spread[feature])
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
        return pd.Series(np.nan, index=zone.index)
    return weighted_sum / total_weight


def rerank_by_zone_contrast(zone_ranked: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for zone_id, zone in zone_ranked.groupby("zone_id", sort=True):
        zone = zone.copy()
        zone["zone_contrast_score"] = compute_zone_contrast_score(zone, target_deltas, delta_scales)
        zone["blended_score"] = zone["zone_contrast_score"] * (1.0 - BLEND_WEIGHT) + zone["candidate_score"] * BLEND_WEIGHT
        zone = zone.sort_values(["blended_score", "candidate_score", "open_time"], ascending=[False, False, True]).reset_index(drop=True)
        zone["zone_rank"] = np.arange(1, len(zone) + 1)
        zone["zone_size"] = len(zone)
        rows.append(zone)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_anchor_rank_summary(zone_ranked: pd.DataFrame) -> pd.DataFrame:
    if zone_ranked.empty:
        return pd.DataFrame()
    positives = zone_ranked.loc[zone_ranked["is_positive_anchor"]].copy()
    if positives.empty:
        return pd.DataFrame()

    return positives[
        [
            "positive_trade_opened_time",
            "open_time",
            "zone_id",
            "zone_rank",
            "zone_size",
            "blended_score",
            "zone_contrast_score",
            "candidate_score",
            "positive_abs_minute_distance",
        ]
    ].rename(
        columns={
            "positive_trade_opened_time": "trade_opened_time",
            "open_time": "anchor_candidate_time",
            "positive_abs_minute_distance": "anchor_abs_minute_distance",
        }
    ).sort_values("trade_opened_time").reset_index(drop=True)


def select_top_k_per_zone(zone_ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
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
        "within_5m_share": float(matches["within_5m"].mean() * 100.0),
        "within_15m_share": float(matches["within_15m"].mean() * 100.0),
        "within_60m_share": float(matches["within_60m"].mean() * 100.0),
        "median_abs_minute_distance": float(matches["abs_minute_distance"].median()),
        "median_abs_price_distance_pct": float(matches["price_distance_pct"].abs().median()),
    }


def write_summary(
    raw_candidates: pd.DataFrame,
    tagged_candidates: pd.DataFrame,
    positive_match_summary: pd.DataFrame,
    delta_profile_rows: pd.DataFrame,
    anchor_rank_summary: pd.DataFrame,
    topk_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    positive_count = int(tagged_candidates["is_positive_anchor"].sum()) if not tagged_candidates.empty else 0
    lines = [
        "ETH SHORT CANDIDATE ENGINE V3",
        "=============================",
        "",
        f"raw_candidate_count = {len(raw_candidates)}",
        f"zone_count = {int(tagged_candidates['zone_id'].nunique()) if not tagged_candidates.empty else 0}",
        f"zone_gap_minutes = {ZONE_GAP_MINUTES}",
        f"positive_match_max_minutes = {POSITIVE_MATCH_MAX_MINUTES:.0f}",
        f"blend_weight_candidate_score = {BLEND_WEIGHT:.2f}",
        "",
        f"reference_trade_count = {len(positive_match_summary)}",
        f"positive_anchor_count = {positive_count}",
        f"positive_anchor_share = {(positive_count / len(positive_match_summary) * 100.0) if len(positive_match_summary) else 0.0:.2f}%",
    ]

    if not positive_match_summary.empty:
        lines.extend(
            [
                f"nearest_raw_within_5m_share = {positive_match_summary['abs_minute_distance'].le(5.0).mean() * 100.0:.2f}%",
                f"nearest_raw_within_15m_share = {positive_match_summary['abs_minute_distance'].le(15.0).mean() * 100.0:.2f}%",
                f"nearest_raw_within_60m_share = {positive_match_summary['abs_minute_distance'].le(60.0).mean() * 100.0:.2f}%",
                "",
            ]
        )
    else:
        lines.append("")

    if not anchor_rank_summary.empty:
        lines.extend(
            [
                "Anchor Ranking Quality",
                "----------------------",
                f"anchor_top1_share = {anchor_rank_summary['zone_rank'].le(1).mean() * 100.0:.2f}%",
                f"anchor_top3_share = {anchor_rank_summary['zone_rank'].le(3).mean() * 100.0:.2f}%",
                f"anchor_top5_share = {anchor_rank_summary['zone_rank'].le(5).mean() * 100.0:.2f}%",
                f"median_anchor_rank = {anchor_rank_summary['zone_rank'].median():.2f}",
                "",
            ]
        )

    for top_k, (shortlist, matches) in topk_results.items():
        stats = build_match_stats(matches)
        lines.append(f"top_{top_k}_per_zone_candidate_count = {len(shortlist)}")
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

    if not delta_profile_rows.empty:
        lines.extend(["Learned Candidate Deltas", "------------------------"])
        delta_columns = [f"{feature}_delta" for feature in MICRO_FEATURES]
        for feature in MICRO_FEATURES:
            col = f"{feature}_delta"
            target = float(delta_profile_rows[col].median())
            spread = float((delta_profile_rows[col].quantile(0.75) - delta_profile_rows[col].quantile(0.25)))
            if not np.isfinite(spread) or spread <= 0:
                spread = float(delta_profile_rows[col].std()) if np.isfinite(delta_profile_rows[col].std()) and delta_profile_rows[col].std() > 0 else 1.0
            lines.append(f"{feature}: target_delta={target:.4f}, delta_scale={spread:.4f}, weight={MICRO_WEIGHTS.get(feature, 1.0):.2f}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    ref = load_reference_trades(JSON_TRADES_PATH)
    features = build_features(candles)
    raw_candidates = generate_candidates(features)
    zone_ranked = assign_zone_ids(raw_candidates)
    tagged_candidates, positive_match_summary = attach_reference_matches(zone_ranked, ref)
    target_deltas, delta_scales, delta_profile_rows = build_positive_delta_profile(tagged_candidates)
    reranked = rerank_by_zone_contrast(tagged_candidates, target_deltas, delta_scales)
    anchor_rank_summary = build_anchor_rank_summary(reranked)

    topk_results: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for top_k in TOP_K_PER_ZONE_VALUES:
        shortlist = select_top_k_per_zone(reranked, top_k)
        matches = match_candidates_to_reference(shortlist, ref)
        topk_results[top_k] = (shortlist, matches)
        shortlist.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_per_zone.csv", index=False)
        matches.to_csv(OUTPUT_DIR / f"shortlist_top_{top_k}_matches.csv", index=False)

    raw_candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    tagged_candidates.to_csv(OUTPUT_DIR / "tagged_candidates.csv", index=False)
    positive_match_summary.to_csv(OUTPUT_DIR / "positive_match_summary.csv", index=False)
    delta_profile_rows.to_csv(OUTPUT_DIR / "positive_delta_profile_rows.csv", index=False)
    reranked.to_csv(OUTPUT_DIR / "reranked_candidates.csv", index=False)
    anchor_rank_summary.to_csv(OUTPUT_DIR / "anchor_rank_summary.csv", index=False)
    write_summary(raw_candidates, reranked, positive_match_summary, delta_profile_rows, anchor_rank_summary, topk_results)

    print(f"Raw candidates: {len(raw_candidates)}")
    print(f"Positive anchors: {int(tagged_candidates['is_positive_anchor'].sum())}")
    if not anchor_rank_summary.empty:
        print(f"Anchor top-1 share: {anchor_rank_summary['zone_rank'].le(1).mean() * 100.0:.2f}%")
    for top_k, (_, matches) in topk_results.items():
        if not matches.empty:
            print(f"Top-{top_k} within 15m share: {matches['within_15m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
