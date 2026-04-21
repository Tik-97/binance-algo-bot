from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ethShortTrader import (
    BASE_DIR,
    CANDLES_1M_PATH,
    JSON_TRADES_PATH,
    STARTING_BALANCE,
    RISK_PER_TRADE_PCT,
    LEVERAGE,
    FEE_PER_SIDE_PCT,
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT,
    build_features as build_execution_features,
    get_reference_window,
    load_candles as load_execution_candles,
    load_reference_eth_shorts,
    match_to_reference,
    run_trade_exit_short,
    trim_candles_to_window,
)
from eth_short_candidate_engine import (
    build_features as build_candidate_features,
    deduplicate_candidates,
    generate_candidates,
    load_candles as load_candidate_candles,
)
from eth_short_execution_backtest_v1 import get_index_at_or_after, simulate_limit_retest_entry
from eth_short_execution_backtest_v3 import (
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT as INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT_V3,
    MAX_STOP_DISTANCE_PCT,
    run_trade_exit_short_v3,
)
from eth_short_live_safe_regime_classifier import predict_live_safe_day_regime_from_bar
from eth_short_regime_aware_prototype_v3 import classify_near_noise_subtype_live


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_regime_aware_live_like_backtest_v1"


def build_live_candidate_feed(
    candle_path: Path,
    ref_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_candles = load_candidate_candles(candle_path)
    bt_start, bt_end = get_reference_window(ref_df)
    trimmed = trim_candles_to_window(raw_candles, bt_start, bt_end)

    candidate_features = build_candidate_features(trimmed)
    raw_candidates = generate_candidates(candidate_features)
    deduped_candidates = deduplicate_candidates(raw_candidates)

    for frame in [raw_candidates, deduped_candidates]:
        if frame.empty:
            continue
        frame["candidate_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["candidate_price"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["trade_date"] = frame["candidate_time"].dt.strftime("%Y-%m-%d")

    return trimmed, raw_candidates, deduped_candidates


def run_live_like_backtest(
    execution_df: pd.DataFrame,
    deduped_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades: list[dict] = []
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    skipped_overlap = 0
    skipped_non_target = 0
    skipped_unfilled = 0
    skipped_invalid_stop = 0
    skipped_wide_stop = 0
    skipped_strong_push = 0

    predicted_target_candidates = 0

    for _, row in deduped_candidates.sort_values("candidate_time").iterrows():
        candidate_time = pd.to_datetime(row["candidate_time"], utc=True)

        if candidate_time < last_exit_time:
            skipped_overlap += 1
            continue

        anchor_idx = get_index_at_or_after(execution_df, candidate_time)
        if anchor_idx is None or anchor_idx + 1 >= len(execution_df):
            continue

        trigger_bar = execution_df.iloc[anchor_idx]
        confirm_bar = execution_df.iloc[min(anchor_idx + 1, len(execution_df) - 1)]
        row_context = trigger_bar

        predicted_regime = predict_live_safe_day_regime_from_bar(trigger_bar)
        if predicted_regime == "other_non_target":
            skipped_non_target += 1
            continue
        predicted_target_candidates += 1

        if predicted_regime == "mixed_transition":
            engine = "v1"
            subtype = "mixed_transition_v1_live_classifier"
        else:
            subtype = classify_near_noise_subtype_live(trigger_bar)
            if subtype == "strong_push_skip_live":
                skipped_strong_push += 1
                continue
            engine = "v3" if subtype == "cooling_v3_live" else "v1"

        if engine == "v1":
            failure_high = float(max(trigger_bar["high"], confirm_bar["high"]))
            stop_price_initial = failure_high * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0)

            fill_idx, order_placed_time, order_price, entry_price = simulate_limit_retest_entry(
                df=execution_df,
                anchor_idx=anchor_idx,
                stop_price_initial=stop_price_initial,
            )
            if fill_idx is None or order_placed_time is None or entry_price is None:
                skipped_unfilled += 1
                continue

            stop_distance = stop_price_initial - entry_price
            if stop_distance <= 0:
                skipped_invalid_stop += 1
                continue

            risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
            qty = risk_amount / stop_distance
            max_notional = balance * LEVERAGE
            desired_notional = qty * entry_price
            if desired_notional > max_notional:
                qty = max_notional / entry_price
            if qty <= 0:
                skipped_invalid_stop += 1
                continue

            notional = qty * entry_price
            fee_entry = notional * (FEE_PER_SIDE_PCT / 100.0)

            trade, _ = run_trade_exit_short(
                df=execution_df,
                entry_idx=fill_idx,
                entry_price=float(entry_price),
                qty=float(qty),
                fee_entry=float(fee_entry),
                balance=float(balance),
                signal_time=candidate_time,
                trigger_time=trigger_bar["open_time"],
                confirm_time=confirm_bar["open_time"],
                order_placed_time=order_placed_time,
                trigger_bar=trigger_bar,
                confirm_bar=confirm_bar,
                entry_mode=f"live_like_{subtype}",
                row_context=row_context,
                order_price=float(order_price) if order_price is not None else np.nan,
                stop_price_initial=float(stop_price_initial),
            )

            payload = asdict(trade)
            payload["stop_distance_pct"] = (stop_distance / entry_price) * 100.0
        else:
            failure_high = float(max(trigger_bar["high"], confirm_bar["high"]))
            stop_price_initial = failure_high * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT_V3 / 100.0)

            fill_idx, order_placed_time, order_price, entry_price = simulate_limit_retest_entry(
                df=execution_df,
                anchor_idx=anchor_idx,
                stop_price_initial=stop_price_initial,
            )
            if fill_idx is None or order_placed_time is None or entry_price is None:
                skipped_unfilled += 1
                continue

            stop_distance = stop_price_initial - entry_price
            if stop_distance <= 0:
                skipped_invalid_stop += 1
                continue

            stop_distance_pct = (stop_distance / entry_price) * 100.0
            if stop_distance_pct > MAX_STOP_DISTANCE_PCT:
                skipped_wide_stop += 1
                continue

            qty = (balance * LEVERAGE) / entry_price
            if qty <= 0:
                skipped_invalid_stop += 1
                continue

            notional = qty * entry_price
            fee_entry = notional * (FEE_PER_SIDE_PCT / 100.0)

            trade, _ = run_trade_exit_short_v3(
                df=execution_df,
                entry_idx=fill_idx,
                entry_price=float(entry_price),
                qty=float(qty),
                fee_entry=float(fee_entry),
                balance=float(balance),
                signal_time=candidate_time,
                trigger_time=trigger_bar["open_time"],
                confirm_time=confirm_bar["open_time"],
                order_placed_time=order_placed_time,
                trigger_bar=trigger_bar,
                confirm_bar=confirm_bar,
                row_context=row_context,
                order_price=float(order_price) if order_price is not None else np.nan,
                stop_price_initial=float(stop_price_initial),
            )

            payload = asdict(trade)
            payload["stop_distance_pct"] = stop_distance_pct

        payload["trade_date"] = candidate_time.strftime("%Y-%m-%d")
        payload["candidate_time"] = candidate_time
        payload["candidate_score"] = float(row["candidate_score"])
        payload["candidate_price"] = float(row["candidate_price"])
        payload["predicted_regime"] = predicted_regime
        payload["regime_subtype"] = subtype
        payload["regime_engine"] = engine
        trades.append(payload)

        balance = float(payload["balance_after"])
        last_exit_time = pd.to_datetime(payload["exit_time"], utc=True)

    trades_df = pd.DataFrame(trades)
    summary = {
        "raw_candidate_count": int(len(deduped_candidates)) if False else 0,
        "deduped_candidate_count": int(len(deduped_candidates)),
        "predicted_target_candidates": int(predicted_target_candidates),
        "filled_trade_count": int(len(trades_df)),
        "skipped_overlap": int(skipped_overlap),
        "skipped_non_target": int(skipped_non_target),
        "skipped_unfilled": int(skipped_unfilled),
        "skipped_invalid_stop": int(skipped_invalid_stop),
        "skipped_wide_stop": int(skipped_wide_stop),
        "skipped_strong_push": int(skipped_strong_push),
        "ending_balance": float(balance),
    }

    if trades_df.empty:
        return trades_df, summary

    wins = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losses = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]
    summary.update(
        {
            "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0),
            "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
            "expectancy_usd": float(trades_df["pnl_usd"].mean()),
            "median_hold_minutes": float(trades_df["hold_minutes"].median()),
            "median_candidate_score": float(trades_df["candidate_score"].median()),
            "median_stop_distance_pct": float(trades_df["stop_distance_pct"].median()),
        }
    )
    return trades_df, summary


def write_summary(
    summary: dict,
    raw_candidates: pd.DataFrame,
    trades_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> None:
    lines = [
        "ETH SHORT REGIME AWARE LIVE-LIKE BACKTEST V1",
        "============================================",
        "",
        "pipeline = candidate_engine_v1 -> live_safe_regime_classifier -> live_safe_micro_router -> v1/v3 execution",
        "note = no nearest_candidates oracle, no day_regime oracle",
        "note = candidate feed still uses candidate_engine_v1 session logic, so this is live-like inside the learned session regime",
        f"raw_candidate_count = {len(raw_candidates)}",
        f"deduped_candidate_count = {summary.get('deduped_candidate_count', 0)}",
        f"predicted_target_candidates = {summary.get('predicted_target_candidates', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
        f"skipped_overlap = {summary.get('skipped_overlap', 0)}",
        f"skipped_non_target = {summary.get('skipped_non_target', 0)}",
        f"skipped_unfilled = {summary.get('skipped_unfilled', 0)}",
        f"skipped_invalid_stop = {summary.get('skipped_invalid_stop', 0)}",
        f"skipped_wide_stop = {summary.get('skipped_wide_stop', 0)}",
        f"skipped_strong_push = {summary.get('skipped_strong_push', 0)}",
        f"ending_balance = {summary.get('ending_balance', np.nan):.2f}",
    ]

    if "win_rate_pct" in summary:
        lines.extend(
            [
                f"win_rate_pct = {summary['win_rate_pct']:.2f}",
                f"profit_factor = {summary['profit_factor']:.3f}" if pd.notna(summary["profit_factor"]) else "profit_factor = nan",
                f"expectancy_usd = {summary['expectancy_usd']:.4f}",
                f"median_hold_minutes = {summary['median_hold_minutes']:.2f}",
                f"median_candidate_score = {summary['median_candidate_score']:.4f}",
                f"median_stop_distance_pct = {summary['median_stop_distance_pct']:.4f}",
            ]
        )

    if not trades_df.empty:
        lines.extend(["", "Engine Split", "------------"])
        for (predicted_regime, subtype, engine), group in trades_df.groupby(["predicted_regime", "regime_subtype", "regime_engine"], sort=True):
            lines.append(
                f"{predicted_regime} / {subtype} / {engine}: trade_count={len(group)}, win_rate_pct={(group['pnl_usd'] > 0).mean() * 100.0:.2f}, median_pnl_usd={group['pnl_usd'].median():.4f}"
            )

    if not matches_df.empty:
        within = int(matches_df["matched_within_60m"].sum())
        same_dir = int(matches_df.loc[matches_df["matched_within_60m"], "same_result_direction"].sum())
        lines.extend(
            [
                "",
                f"matches_within_60m = {within}",
                f"same_result_direction = {same_dir}",
                f"agreement_rate_pct = {same_dir / within * 100.0:.2f}" if within > 0 else "agreement_rate_pct = n/a",
                f"median_time_diff_minutes = {matches_df['time_diff_minutes'].median():.2f}",
            ]
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_df = load_reference_eth_shorts(JSON_TRADES_PATH)
    trimmed_candles, raw_candidates, deduped_candidates = build_live_candidate_feed(CANDLES_1M_PATH, ref_df)
    execution_candles = load_execution_candles(CANDLES_1M_PATH)
    bt_start, bt_end = get_reference_window(ref_df)
    execution_trimmed = trim_candles_to_window(execution_candles, bt_start, bt_end)
    execution_df = build_execution_features(execution_trimmed)

    trades_df, summary = run_live_like_backtest(execution_df, deduped_candidates)
    matches_df = match_to_reference(trades_df, ref_df, max_minutes=60)

    raw_candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    deduped_candidates.to_csv(OUTPUT_DIR / "deduped_candidates.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    matches_df.to_csv(OUTPUT_DIR / "matches.csv", index=False)
    write_summary(summary, raw_candidates, trades_df, matches_df)

    print(f"Raw candidates: {len(raw_candidates)}")
    print(f"Deduped candidates: {len(deduped_candidates)}")
    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
