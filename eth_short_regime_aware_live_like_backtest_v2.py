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
from eth_short_execution_backtest_v1 import get_index_at_or_after, simulate_limit_retest_entry
from eth_short_live_safe_regime_classifier import predict_live_safe_day_regime_from_bar
from eth_short_regime_aware_live_like_backtest_v1 import build_live_candidate_feed
from eth_short_regime_aware_prototype_v3 import classify_near_noise_subtype_live


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_regime_aware_live_like_backtest_v2"
TARGET_REGIME = "near_entry_breakdown_noise"
TARGET_SUBTYPE = "mild_noise_v1_live"


def run_live_like_backtest_v2(
    execution_df: pd.DataFrame,
    deduped_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades: list[dict] = []
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    skipped_overlap = 0
    skipped_non_target = 0
    skipped_wrong_subtype = 0
    skipped_unfilled = 0
    skipped_invalid_stop = 0

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
        if predicted_regime != TARGET_REGIME:
            skipped_non_target += 1
            continue
        predicted_target_candidates += 1

        subtype = classify_near_noise_subtype_live(trigger_bar)
        if subtype != TARGET_SUBTYPE:
            skipped_wrong_subtype += 1
            continue

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
            entry_mode=f"live_like_{TARGET_SUBTYPE}",
            row_context=row_context,
            order_price=float(order_price) if order_price is not None else np.nan,
            stop_price_initial=float(stop_price_initial),
        )

        payload = asdict(trade)
        payload["stop_distance_pct"] = (stop_distance / entry_price) * 100.0
        payload["trade_date"] = candidate_time.strftime("%Y-%m-%d")
        payload["candidate_time"] = candidate_time
        payload["candidate_score"] = float(row["candidate_score"])
        payload["candidate_price"] = float(row["candidate_price"])
        payload["predicted_regime"] = predicted_regime
        payload["regime_subtype"] = subtype
        payload["regime_engine"] = "v1"
        trades.append(payload)

        balance = float(payload["balance_after"])
        last_exit_time = pd.to_datetime(payload["exit_time"], utc=True)

    trades_df = pd.DataFrame(trades)
    summary = {
        "deduped_candidate_count": int(len(deduped_candidates)),
        "predicted_target_candidates": int(predicted_target_candidates),
        "filled_trade_count": int(len(trades_df)),
        "skipped_overlap": int(skipped_overlap),
        "skipped_non_target": int(skipped_non_target),
        "skipped_wrong_subtype": int(skipped_wrong_subtype),
        "skipped_unfilled": int(skipped_unfilled),
        "skipped_invalid_stop": int(skipped_invalid_stop),
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


def write_summary(summary: dict, trades_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT REGIME AWARE LIVE-LIKE BACKTEST V2",
        "============================================",
        "",
        f"pipeline = candidate_engine_v1 -> {TARGET_REGIME} -> {TARGET_SUBTYPE} -> v1 execution",
        "note = survivor branch only; mixed_transition and cooling_v3 are intentionally disabled",
        f"deduped_candidate_count = {summary.get('deduped_candidate_count', 0)}",
        f"predicted_target_candidates = {summary.get('predicted_target_candidates', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
        f"skipped_overlap = {summary.get('skipped_overlap', 0)}",
        f"skipped_non_target = {summary.get('skipped_non_target', 0)}",
        f"skipped_wrong_subtype = {summary.get('skipped_wrong_subtype', 0)}",
        f"skipped_unfilled = {summary.get('skipped_unfilled', 0)}",
        f"skipped_invalid_stop = {summary.get('skipped_invalid_stop', 0)}",
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
    _, _, deduped_candidates = build_live_candidate_feed(CANDLES_1M_PATH, ref_df)

    execution_candles = load_execution_candles(CANDLES_1M_PATH)
    bt_start, bt_end = get_reference_window(ref_df)
    execution_trimmed = trim_candles_to_window(execution_candles, bt_start, bt_end)
    execution_df = build_execution_features(execution_trimmed)

    trades_df, summary = run_live_like_backtest_v2(execution_df, deduped_candidates)
    matches_df = match_to_reference(trades_df, ref_df, max_minutes=60)

    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    matches_df.to_csv(OUTPUT_DIR / "matches.csv", index=False)
    write_summary(summary, trades_df, matches_df)

    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
