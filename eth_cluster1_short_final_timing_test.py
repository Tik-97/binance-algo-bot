from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_cluster1_short_overlap_validation import (
    add_window_quality_metrics,
    build_overlap_matches,
    build_overlap_quality,
    build_overlap_summary,
    load_proxy_market_frames,
)
from eth_replication_hypothesis_test import ETH_SYMBOL
from market_context_at_entry import prepare_context_features
from utils import fmt_int, fmt_number, pct


FINAL_TIMING_VARIANTS = [
    {
        "variant_id": "base_proxy",
        "variant_label": "Base proxy",
        "family": "base",
        "definition_text": "Current practical Cluster 1 short proxy without extra timing delay.",
    },
    {
        "variant_id": "delay_5m_zone_hold",
        "variant_label": "Delay 5m if zone still valid",
        "family": "delay_after_zone_arrival",
        "definition_text": "Wait one completed 5m candle after the base signal and enter only if the Cluster 1 short zone still looks valid.",
    },
    {
        "variant_id": "delay_10m_no_continuation",
        "variant_label": "Delay + no meaningful continuation",
        "family": "delay_plus_no_continuation",
        "definition_text": "Wait up to two completed 5m candles and enter only if there is no meaningful continuation above the signal high while the zone still looks valid.",
    },
    {
        "variant_id": "delay_10m_weakening",
        "variant_label": "Delay + simple weakening candle",
        "family": "delay_plus_weakening_confirmation",
        "definition_text": "Wait up to two completed 5m candles and enter only on a simple weakening / failed-continuation style candle while the zone still looks valid.",
    },
]

BASE_ZONE_HIGH_THRESHOLD = 0.60
BASE_ZONE_BOUNCE_THRESHOLD = 0.45
MAX_ZONE_DROP_FROM_SIGNAL_PCT = 0.35
MEANINGFUL_NEW_HIGH_PCT = 0.15


def run_cluster1_short_final_timing_test_analysis(
    trades_df: pd.DataFrame,
    real_trades_df: pd.DataFrame,
    model_signals_df: pd.DataFrame,
    timing_error_matches_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "FINAL NARROW TIMING TEST - DELAY AFTER ZONE ARRIVAL + WEAKENING CONFIRMATION FOR CLUSTER 1 SHORT\n"
        "--------------------------------------------------------------------------------------------------\n"
        "No sufficient Cluster 1 short proxy data was available for the final narrow timing test.\n"
    )

    if real_trades_df.empty or model_signals_df.empty:
        return empty_result(empty, empty_section)

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    minute_frame, frame_5m, _stages = load_proxy_market_frames(eth_trades, logger)
    if minute_frame.empty or frame_5m.empty:
        section = (
            "FINAL NARROW TIMING TEST - DELAY AFTER ZONE ARRIVAL + WEAKENING CONFIRMATION FOR CLUSTER 1 SHORT\n"
            "--------------------------------------------------------------------------------------------------\n"
            "The final narrow timing test could not run because ETH candle context was unavailable.\n"
        )
        return empty_result(empty, section)

    logger.info(
        "Running final narrow timing test for Cluster 1 short on %s base proxy signals",
        len(model_signals_df),
    )

    enriched_signals, feature_5m = prepare_base_signals_for_timing_test(model_signals_df, frame_5m)
    if enriched_signals.empty or feature_5m.empty:
        return empty_result(empty, empty_section)

    minute_arrays = {
        "open_time_ns": minute_frame["open_time"].astype("int64").to_numpy(),
        "close_time_ns": minute_frame["close_time"].astype("int64").to_numpy(),
        "close": pd.to_numeric(minute_frame["close"], errors="coerce").to_numpy(dtype=float),
        "high": pd.to_numeric(minute_frame["high"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(minute_frame["low"], errors="coerce").to_numpy(dtype=float),
    }

    variants_df, results_df, matches_by_variant = evaluate_final_timing_variants(
        real_trades_df,
        enriched_signals,
        feature_5m,
        minute_arrays,
    )
    if results_df.empty:
        return empty_result(empty, empty_section)

    best_variant_row = select_best_variant(results_df)
    best_variant_id = str(best_variant_row["variant_id"])
    best_matches = matches_by_variant.get(best_variant_id, pd.DataFrame()).copy()
    tradeoffs_df = build_tradeoff_table(results_df)
    section_text = build_final_timing_section(results_df, best_variant_row, timing_error_matches_df)

    del output_dir
    return {
        "variants": variants_df,
        "results": results_df,
        "tradeoffs": tradeoffs_df,
        "best_matches": best_matches,
        "section_text": section_text,
    }


def empty_result(empty: pd.DataFrame, section_text: str) -> dict:
    return {
        "variants": empty,
        "results": empty,
        "tradeoffs": empty,
        "best_matches": empty,
        "section_text": section_text,
    }


def prepare_base_signals_for_timing_test(model_signals_df: pd.DataFrame, frame_5m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = model_signals_df.copy().sort_values("entry_time")
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True, errors="coerce")
    signals["entry_price"] = pd.to_numeric(signals["entry_price"], errors="coerce")

    feature_5m = build_final_timing_5m_features(frame_5m)
    if feature_5m.empty:
        return pd.DataFrame(), pd.DataFrame()

    signals = pd.merge_asof(
        signals.sort_values("entry_time"),
        feature_5m.sort_values("feature_time"),
        left_on="entry_time",
        right_on="feature_time",
        direction="backward",
    )
    signals["signal_high"] = pd.to_numeric(signals.get("high"), errors="coerce")
    return signals.reset_index(drop=True), feature_5m


def build_final_timing_5m_features(frame_5m: pd.DataFrame) -> pd.DataFrame:
    if frame_5m.empty:
        return pd.DataFrame()

    prepared = prepare_context_features(frame_5m, "5m").copy().sort_values("close_time")
    for column in ["open", "high", "low", "close", "volume"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    price_range = (prepared["high"] - prepared["low"]).replace(0, np.nan)
    prior_5m_high = prepared["high"].shift(1)

    prepared["body_range_ratio_5m"] = (prepared["close"] - prepared["open"]).abs() / price_range
    prepared["close_position_in_range_5m"] = (prepared["close"] - prepared["low"]) / price_range
    prepared["upper_wick_ratio_5m_local"] = (prepared["high"] - prepared[["open", "close"]].max(axis=1)) / price_range
    prepared["close_back_below_prior_5m_high_flag"] = (prepared["high"] > prior_5m_high) & (prepared["close"] < prior_5m_high)
    prepared["weakening_confirmation_flag"] = (
        (
            prepared["body_range_ratio_5m"].le(0.35)
            & prepared["close_position_in_range_5m"].le(0.55)
        )
        | prepared["close_back_below_prior_5m_high_flag"].fillna(False)
        | (
            prepared["upper_wick_ratio_5m_local"].ge(0.25)
            & prepared["close_position_in_range_5m"].le(0.50)
        )
    ).fillna(False)

    distance_to_high = pd.to_numeric(
        prepared.get("distance_to_last_4h_high_pct", pd.Series(np.nan, index=prepared.index)),
        errors="coerce",
    )
    pre_60m_return = pd.to_numeric(
        prepared.get("return_pre_60m_pct", pd.Series(np.nan, index=prepared.index)),
        errors="coerce",
    )
    prepared["zone_still_valid_flag"] = (
        distance_to_high.le(BASE_ZONE_HIGH_THRESHOLD)
        & pre_60m_return.ge(BASE_ZONE_BOUNCE_THRESHOLD)
    ).fillna(False)

    prepared["feature_time"] = pd.to_datetime(prepared["close_time"], utc=True, errors="coerce")
    keep_columns = [
        "feature_time",
        "open",
        "high",
        "low",
        "close",
        "distance_to_last_4h_high_pct",
        "return_pre_60m_pct",
        "ret_last_5m_pct",
        "ret_3c_5m_pct",
        "upper_wick_ratio_5m",
        "body_range_ratio_5m",
        "close_position_in_range_5m",
        "weakening_confirmation_flag",
        "zone_still_valid_flag",
    ]
    available = [column for column in keep_columns if column in prepared.columns]
    return prepared[available].copy()


def evaluate_final_timing_variants(
    real_trades_df: pd.DataFrame,
    enriched_signals: pd.DataFrame,
    feature_5m: pd.DataFrame,
    minute_arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    variants_df = pd.DataFrame(FINAL_TIMING_VARIANTS)
    rows = []
    matches_by_variant: dict[str, pd.DataFrame] = {}
    base_summary = None

    for variant in FINAL_TIMING_VARIANTS:
        variant_signals = build_variant_signal_set(enriched_signals, feature_5m, variant)
        if not variant_signals.empty:
            drop_columns = [
                column
                for column in variant_signals.columns
                if column.startswith("window_return_")
                or column.startswith("window_mfe_")
                or column.startswith("window_mae_")
            ]
            if drop_columns:
                variant_signals = variant_signals.drop(columns=drop_columns, errors="ignore")
            variant_signals = add_window_quality_metrics(
                variant_signals,
                minute_arrays,
                time_column="entry_time",
                price_column="entry_price",
                side_column="side",
            )
        overlap_matches = build_overlap_matches(real_trades_df, variant_signals)
        overlap_summary = build_overlap_summary(real_trades_df, variant_signals, overlap_matches, np.nan)
        overlap_quality = build_overlap_quality(real_trades_df, variant_signals, overlap_matches)

        if variant["variant_id"] == "base_proxy" and not overlap_summary.empty:
            base_summary = overlap_summary.iloc[0]

        rows.append(build_variant_result_row(variant, variant_signals, overlap_summary, overlap_quality, base_summary))
        matches_by_variant[variant["variant_id"]] = overlap_matches.copy()

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return variants_df, results_df, matches_by_variant

    results_df = finalize_variant_scores(results_df)
    variants_df = variants_df.merge(
        results_df[["variant_id", "signal_count", "balanced_timing_score", "is_best_variant"]],
        on="variant_id",
        how="left",
    )
    return variants_df, results_df.sort_values("balanced_timing_score", ascending=False).reset_index(drop=True), matches_by_variant


def build_variant_signal_set(
    enriched_signals: pd.DataFrame,
    feature_5m: pd.DataFrame,
    variant: dict,
) -> pd.DataFrame:
    variant_id = str(variant["variant_id"])
    if variant_id == "base_proxy":
        frame = enriched_signals.copy()
        frame["variant_entry_delay_minutes"] = 0.0
        frame["variant_confirmation_text"] = "Immediate base proxy entry"
        return finalize_variant_signal_frame(frame, variant)

    candidate_rows = []
    feature_rows = feature_5m.sort_values("feature_time").copy()
    for _, signal in enriched_signals.iterrows():
        selected = select_variant_entry(signal, feature_rows, variant_id)
        if selected is None:
            continue
        row = signal.copy()
        row["entry_time"] = selected["feature_time"]
        row["model_entry_time"] = selected["feature_time"]
        row["entry_price"] = selected["close"]
        row["model_entry_price"] = selected["close"]
        row["model_signal_id"] = f"{signal['model_signal_id']}__{variant_id}"
        row["signal_id"] = f"{signal['signal_id']}__{variant_id}"
        row["variant_entry_delay_minutes"] = selected["entry_delay_minutes"]
        row["variant_confirmation_text"] = selected["confirmation_text"]
        row["distance_to_last_4h_high_pct"] = selected.get("distance_to_last_4h_high_pct")
        row["return_pre_60m_pct"] = selected.get("return_pre_60m_pct")
        row["ret_last_5m_pct"] = selected.get("ret_last_5m_pct")
        row["ret_3c_5m_pct"] = selected.get("ret_3c_5m_pct")
        row["upper_wick_ratio_5m"] = selected.get("upper_wick_ratio_5m")
        candidate_rows.append(row)

    return finalize_variant_signal_frame(pd.DataFrame(candidate_rows), variant)


def select_variant_entry(signal: pd.Series, feature_rows: pd.DataFrame, variant_id: str) -> dict | None:
    signal_time = pd.to_datetime(signal.get("entry_time"), utc=True, errors="coerce")
    signal_price = safe_number(signal.get("entry_price"))
    signal_high = safe_number(signal.get("signal_high"))
    if pd.isna(signal_time) or signal_price <= 0:
        return None

    max_delay = 5 if variant_id == "delay_5m_zone_hold" else 10
    candidates = feature_rows.loc[
        (feature_rows["feature_time"] > signal_time)
        & (feature_rows["feature_time"] <= signal_time + pd.Timedelta(minutes=max_delay))
    ].copy()
    if candidates.empty:
        return None

    candidates["entry_delay_minutes"] = (candidates["feature_time"] - signal_time).dt.total_seconds() / 60.0
    candidates["zone_drop_from_signal_pct"] = (signal_price - pd.to_numeric(candidates["close"], errors="coerce")) / signal_price * 100.0
    reference_high = signal_high if signal_high > 0 else signal_price
    candidates["meaningful_new_high_flag"] = pd.to_numeric(candidates["high"], errors="coerce") > reference_high * (1.0 + MEANINGFUL_NEW_HIGH_PCT / 100.0)
    candidates["zone_held_after_delay_flag"] = (
        coerce_bool_series(candidates.get("zone_still_valid_flag", False))
        & candidates["zone_drop_from_signal_pct"].le(MAX_ZONE_DROP_FROM_SIGNAL_PCT)
    ).fillna(False)

    if variant_id == "delay_5m_zone_hold":
        exact = candidates.loc[candidates["entry_delay_minutes"] >= 5.0 - 1e-9].sort_values("feature_time").head(1)
        if exact.empty:
            return None
        candidate = exact.iloc[0]
        if not bool(candidate.get("zone_held_after_delay_flag")):
            return None
        return build_selected_candidate(candidate, "5m delay while short zone still valid")

    if variant_id == "delay_10m_no_continuation":
        filtered = candidates.loc[
            candidates["zone_held_after_delay_flag"]
            & ~coerce_bool_series(candidates["meaningful_new_high_flag"])
        ].sort_values("feature_time")
        if filtered.empty:
            return None
        return build_selected_candidate(filtered.iloc[0], "delay plus no meaningful continuation above signal high")

    if variant_id == "delay_10m_weakening":
        filtered = candidates.loc[
            candidates["zone_held_after_delay_flag"]
            & coerce_bool_series(candidates.get("weakening_confirmation_flag", False))
        ].sort_values("feature_time")
        if filtered.empty:
            return None
        return build_selected_candidate(filtered.iloc[0], "delay plus simple weakening / failed continuation candle")

    return None


def build_selected_candidate(candidate: pd.Series, confirmation_text: str) -> dict:
    data = candidate.to_dict()
    data["confirmation_text"] = confirmation_text
    return data


def finalize_variant_signal_frame(frame: pd.DataFrame, variant: dict) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame = frame.sort_values("entry_time")
    frame["timing_variant_id"] = variant["variant_id"]
    frame["timing_variant_label"] = variant["variant_label"]
    frame["timing_variant_family"] = variant["family"]
    frame["timing_variant_definition_text"] = variant["definition_text"]
    return frame.reset_index(drop=True)


def build_variant_result_row(
    variant: dict,
    variant_signals: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    overlap_quality: pd.DataFrame,
    base_summary: pd.Series | None,
) -> dict:
    summary = overlap_summary.iloc[0] if not overlap_summary.empty else pd.Series(dtype=object)
    pair_row = select_quality_row(overlap_quality, "matched_pair_gaps")
    matched_model_row = select_quality_row(overlap_quality, "matched_model_signals")

    signal_count = int(len(variant_signals))
    base_signal_count = safe_number(base_summary.get("model_signal_count")) if base_summary is not None else np.nan

    return {
        "variant_id": variant["variant_id"],
        "variant_label": variant["variant_label"],
        "family": variant["family"],
        "definition_text": variant["definition_text"],
        "signal_count": signal_count,
        "reduction_vs_base": 1.0 - safe_divide(signal_count, base_signal_count),
        "recall_within_15m": summary.get("share_within_15m"),
        "recall_within_30m": summary.get("share_within_30m"),
        "recall_within_60m": summary.get("share_within_60m"),
        "share_within_15m_and_0p25pct": summary.get("share_within_15m_and_0p25pct"),
        "share_within_30m_and_0p50pct": summary.get("share_within_30m_and_0p50pct"),
        "share_within_60m_and_1p00pct": summary.get("share_within_60m_and_1p00pct"),
        "median_time_distance_minutes": summary.get("median_matched_time_distance_minutes"),
        "median_price_distance_pct": summary.get("median_matched_price_distance_pct"),
        "same_short_move_30m_share": summary.get("same_short_move_30m_share_matched"),
        "same_short_move_60m_share": summary.get("same_short_move_60m_share_matched"),
        "matched_unique_share": summary.get("matched_unique_share"),
        "matched_model_median_return_15m_pct": matched_model_row.get("median_return_15m_pct"),
        "matched_model_median_return_30m_pct": matched_model_row.get("median_return_30m_pct"),
        "median_abs_gap_return_15m_pct": pair_row.get("median_abs_gap_return_15m_pct"),
        "median_abs_gap_return_30m_pct": pair_row.get("median_abs_gap_return_30m_pct"),
    }


def finalize_variant_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    base_row = results_df.loc[results_df["variant_id"] == "base_proxy"].iloc[0]
    base_recall_30 = safe_number(base_row.get("recall_within_30m"))
    base_recall_60 = safe_number(base_row.get("recall_within_60m"))
    base_combo = safe_number(base_row.get("share_within_30m_and_0p50pct"))
    base_same_move_30 = safe_number(base_row.get("same_short_move_30m_share"))
    base_time_distance = safe_number(base_row.get("median_time_distance_minutes"))
    base_price_distance = safe_number(base_row.get("median_price_distance_pct"))

    rows = []
    for _, row in results_df.iterrows():
        result = row.to_dict()
        recall_30_retention = safe_divide(safe_number(row.get("recall_within_30m")), base_recall_30)
        recall_60_retention = safe_divide(safe_number(row.get("recall_within_60m")), base_recall_60)
        combo_improvement = safe_number(row.get("share_within_30m_and_0p50pct")) - base_combo
        same_move_improvement = safe_number(row.get("same_short_move_30m_share")) - base_same_move_30
        time_improvement = base_time_distance - safe_number(row.get("median_time_distance_minutes"))
        price_improvement = base_price_distance - safe_number(row.get("median_price_distance_pct"))
        signal_reduction = max(safe_number(row.get("reduction_vs_base")), 0.0)

        score = (
            0.28 * float(np.clip(recall_30_retention if pd.notna(recall_30_retention) else 0.0, 0.0, 1.2))
            + 0.14 * float(np.clip(recall_60_retention if pd.notna(recall_60_retention) else 0.0, 0.0, 1.2))
            + 0.20 * safe_number(row.get("share_within_30m_and_0p50pct"))
            + 0.18 * safe_number(row.get("same_short_move_30m_share"))
            + 0.10 * signal_reduction
            + 0.06 * max(time_improvement, 0.0) / max(base_time_distance, 1.0)
            + 0.04 * max(price_improvement, 0.0) / max(base_price_distance, 1e-6)
        )

        result["recall_30_retention_vs_base"] = recall_30_retention
        result["recall_60_retention_vs_base"] = recall_60_retention
        result["combo_overlap_improvement_vs_base"] = combo_improvement
        result["same_move_30m_improvement_vs_base"] = same_move_improvement
        result["time_distance_improvement_minutes_vs_base"] = time_improvement
        result["price_distance_improvement_pct_vs_base"] = price_improvement
        result["balanced_timing_score"] = score
        rows.append(result)

    scored = pd.DataFrame(rows)
    best_idx = scored.sort_values(
        ["balanced_timing_score", "share_within_30m_and_0p50pct", "same_short_move_30m_share"],
        ascending=[False, False, False],
    ).index[0]
    scored["is_best_variant"] = False
    scored.loc[best_idx, "is_best_variant"] = True
    return scored


def build_tradeoff_table(results_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "variant_id",
        "variant_label",
        "family",
        "signal_count",
        "reduction_vs_base",
        "recall_within_15m",
        "recall_within_30m",
        "recall_within_60m",
        "share_within_15m_and_0p25pct",
        "share_within_30m_and_0p50pct",
        "median_time_distance_minutes",
        "median_price_distance_pct",
        "same_short_move_30m_share",
        "same_short_move_60m_share",
        "balanced_timing_score",
        "is_best_variant",
    ]
    existing = [column for column in keep_columns if column in results_df.columns]
    return results_df[existing].sort_values("balanced_timing_score", ascending=False).reset_index(drop=True)


def select_quality_row(df: pd.DataFrame, group_name: str) -> pd.Series:
    if df.empty or "group_name" not in df.columns:
        return pd.Series(dtype=object)
    match = df.loc[df["group_name"] == group_name]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def select_best_variant(results_df: pd.DataFrame) -> pd.Series:
    best = results_df.loc[results_df["is_best_variant"] == True]
    if best.empty:
        return results_df.sort_values("balanced_timing_score", ascending=False).iloc[0]
    return best.iloc[0]


def build_final_timing_section(
    results_df: pd.DataFrame,
    best_variant_row: pd.Series,
    timing_error_matches_df: pd.DataFrame,
) -> str:
    base_row = results_df.loc[results_df["variant_id"] == "base_proxy"].iloc[0]
    best_is_base = str(best_variant_row.get("variant_id")) == "base_proxy"
    material_help = bool(
        not best_is_base
        and safe_number(best_variant_row.get("reduction_vs_base")) >= 0.10
        and safe_number(best_variant_row.get("recall_30_retention_vs_base")) >= 0.85
        and (
            safe_number(best_variant_row.get("same_move_30m_improvement_vs_base")) >= 0.10
            or safe_number(best_variant_row.get("combo_overlap_improvement_vs_base")) >= 0.05
            or safe_number(best_variant_row.get("time_distance_improvement_minutes_vs_base")) >= 2.0
        )
    )

    early_share = extract_timing_bucket_share(timing_error_matches_df, "base_proxy", "early")
    late_share = extract_timing_bucket_share(timing_error_matches_df, "base_proxy", "late")
    harmful_early_pattern = early_share >= 0.35 and early_share >= late_share * 0.75
    confidence = determine_final_timing_confidence(best_variant_row, material_help)

    report = f"""
FINAL NARROW TIMING TEST - DELAY AFTER ZONE ARRIVAL + WEAKENING CONFIRMATION FOR CLUSTER 1 SHORT
--------------------------------------------------------------------------------------------------
This is one final narrow timing refinement test. The goal is not to recover exact execution logic, but to check whether a very small delay-after-zone-arrival layer plus simple weakening confirmation can improve practical imitation precision for the Cluster 1 short family.

Base proxy:
- Signals: {fmt_int(base_row.get('signal_count'))}.
- Recall within 15m / 30m / 60m: {pct(base_row.get('recall_within_15m'))} / {pct(base_row.get('recall_within_30m'))} / {pct(base_row.get('recall_within_60m'))}.
- Combined overlap within 15m + 0.25% / 30m + 0.50%: {pct(base_row.get('share_within_15m_and_0p25pct'))} / {pct(base_row.get('share_within_30m_and_0p50pct'))}.
- Median matched time distance: {fmt_number(base_row.get('median_time_distance_minutes'))} minutes.
- Median matched price distance: {fmt_number(base_row.get('median_price_distance_pct'))}%.
- Same-short-move share at 30m / 60m: {pct(base_row.get('same_short_move_30m_share'))} / {pct(base_row.get('same_short_move_60m_share'))}.

Best final timing variant:
- Variant: {best_variant_row.get('variant_label')}.
- Definition: {best_variant_row.get('definition_text')}.
- Signals: {fmt_int(best_variant_row.get('signal_count'))} ({pct(best_variant_row.get('reduction_vs_base'))} reduction vs base).
- Recall within 15m / 30m / 60m: {pct(best_variant_row.get('recall_within_15m'))} / {pct(best_variant_row.get('recall_within_30m'))} / {pct(best_variant_row.get('recall_within_60m'))}.
- Combined overlap within 15m + 0.25% / 30m + 0.50%: {pct(best_variant_row.get('share_within_15m_and_0p25pct'))} / {pct(best_variant_row.get('share_within_30m_and_0p50pct'))}.
- Median matched time distance: {fmt_number(best_variant_row.get('median_time_distance_minutes'))} minutes.
- Median matched price distance: {fmt_number(best_variant_row.get('median_price_distance_pct'))}%.
- Same-short-move share at 30m / 60m: {pct(best_variant_row.get('same_short_move_30m_share'))} / {pct(best_variant_row.get('same_short_move_60m_share'))}.

Interpretation:
- {'A small delay-after-zone-arrival layer helped materially.' if material_help else 'A small delay-after-zone-arrival layer did not produce a material practical jump.'}
- {'Simple weakening confirmation helped more than delay alone.' if (not best_is_base and 'weakening' in str(best_variant_row.get('variant_id'))) else 'Weakening confirmation did not clearly solve the remaining timing gap on its own.'}
- {'Same-short-move capture improved in a meaningful way.' if safe_number(best_variant_row.get('same_move_30m_improvement_vs_base')) >= 0.10 else 'Same-short-move capture did not improve enough to call this a clean timing fix.'}
- {'Overlap precision improved without badly breaking recall.' if material_help else 'No variant improved overlap precision enough while keeping recall comfortably intact.'}
- {'The remaining gap still looks structured enough for one more tiny refinement only if it stays very close to this delay-after-zone-arrival idea.' if material_help else 'The remaining gap now looks closer to a practical ceiling than to one clearly missing simple timing layer.'}
- {'The timing-error anatomy still supports an early-fire explanation in part.' if harmful_early_pattern else 'The timing mismatch remains too mixed for one simple delay rule to explain fully.'}
- Confidence level: {confidence}.

PRACTICAL INTERPRETATION - FINAL TIMING TEST: DID IT ACTUALLY HELP?
-------------------------------------------------------------------
- {'This final timing layer helped enough to keep one more very small refinement or paper-style imitation test on the table.' if material_help else 'This final timing layer did not help enough; reverse-engineering refinement should likely stop here.'}
- {'If work continues, the next move should be one more paper-style imitation check rather than another broad feature hunt.' if material_help else 'The next move should probably shift away from further trigger recovery and toward sandbox / paper-style imitation work.'}
- {'This remains a candidate timing refinement, not recovered execution logic.' if material_help else 'This now looks more like a practical ceiling than one more hidden trigger waiting to be recovered.'}
""".strip()
    return report + "\n"


def extract_timing_bucket_share(timing_error_matches_df: pd.DataFrame, variant_id: str, bucket_name: str) -> float:
    if timing_error_matches_df.empty:
        return 0.0
    frame = timing_error_matches_df.copy()
    if "variant_id" in frame.columns:
        frame = frame.loc[frame["variant_id"].astype(str) == variant_id].copy()
    if frame.empty or "timing_bucket" not in frame.columns:
        return 0.0
    return safe_number((frame["timing_bucket"].astype(str) == bucket_name).mean())


def determine_final_timing_confidence(best_variant_row: pd.Series, material_help: bool) -> str:
    signal_count = safe_number(best_variant_row.get("signal_count"))
    recall_30 = safe_number(best_variant_row.get("recall_within_30m"))
    if material_help and signal_count >= 20 and recall_30 >= 0.35:
        return "medium"
    return "low"


def create_cluster1_short_final_timing_test_plots(
    results_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if results_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    overlap_path = plots_dir / "eth_cluster1_short_final_timing_overlap.png"
    same_move_path = plots_dir / "eth_cluster1_short_final_timing_same_move.png"
    tradeoff_path = plots_dir / "eth_cluster1_short_final_timing_tradeoff.png"

    fig_overlap = plot_final_timing_overlap(results_df)
    fig_overlap.savefig(overlap_path, dpi=150, bbox_inches="tight")
    plt.close(fig_overlap)

    fig_same_move = plot_final_timing_same_move(results_df)
    fig_same_move.savefig(same_move_path, dpi=150, bbox_inches="tight")
    plt.close(fig_same_move)

    fig_tradeoff = plot_final_timing_tradeoff(results_df)
    fig_tradeoff.savefig(tradeoff_path, dpi=150, bbox_inches="tight")
    plt.close(fig_tradeoff)

    logger.info("Saved final narrow timing test plots")


def plot_final_timing_overlap(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.sort_values("balanced_timing_score", ascending=False).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(plot_df))
    width = 0.25

    axes[0].bar(x - width, plot_df["recall_within_30m"], width=width, label="Recall 30m", color="#4c78a8")
    axes[0].bar(x, plot_df["share_within_30m_and_0p50pct"], width=width, label="30m + 0.50%", color="#f58518")
    axes[0].bar(x + width, plot_df["share_within_15m_and_0p25pct"], width=width, label="15m + 0.25%", color="#54a24b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["variant_label"], rotation=20)
    axes[0].set_title("Final timing test: overlap precision")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, plot_df["median_time_distance_minutes"], width=width, label="Time distance", color="#4c78a8")
    axes[1].bar(x + width / 2, plot_df["median_price_distance_pct"], width=width, label="Price distance %", color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["variant_label"], rotation=20)
    axes[1].set_title("Final timing test: matched distance")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_final_timing_same_move(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.sort_values("balanced_timing_score", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width / 2, plot_df["same_short_move_30m_share"], width=width, label="Same move 30m", color="#54a24b")
    ax.bar(x + width / 2, plot_df["same_short_move_60m_share"], width=width, label="Same move 60m", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant_label"], rotation=20)
    ax.set_title("Final timing test: same-short-move capture")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_final_timing_tradeoff(results_df: pd.DataFrame) -> plt.Figure:
    plot_df = results_df.copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_df["recall_within_30m"],
        plot_df["same_short_move_30m_share"],
        s=np.maximum(60.0, 450.0 - plot_df["signal_count"].fillna(0).to_numpy(dtype=float) * 0.7),
        c=plot_df["balanced_timing_score"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
    )
    for _, row in plot_df.iterrows():
        ax.annotate(str(row["variant_label"]), (row["recall_within_30m"], row["same_short_move_30m_share"]), fontsize=8)
    ax.set_title("Final timing test trade-off")
    ax.set_xlabel("Recall within 30m")
    ax.set_ylabel("Same-short-move share at 30m")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Balanced timing score")
    return fig


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def safe_number(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def coerce_bool_series(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.fillna(False).astype(bool)
    return pd.Series(bool(series))
