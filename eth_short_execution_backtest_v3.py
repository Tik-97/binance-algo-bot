from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ethShortTrader import (
    BASE_DIR,
    CANDLES_1M_PATH,
    STARTING_BALANCE,
    LEVERAGE,
    FEE_PER_SIDE_PCT,
    LIMIT_FILL_SLIPPAGE_PCT,
    MARKET_SLIPPAGE_PER_SIDE_PCT,
    MAX_BARS_SCAN,
    Trade,
    build_features,
    load_candles,
    load_reference_eth_shorts,
)
from eth_short_execution_backtest_v1 import (
    NEAREST_CANDIDATES_PATH,
    ENTRY_RETEST_ABOVE_CLOSE_PCT,
    ENTRY_WAIT_BARS,
    build_match_summary,
    build_reference_lookup,
    load_nearest_candidates,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_execution_backtest_v3"

USE_FULL_MARGIN = True
ENTRY_MODE_NAME_V3 = "candidate_limit_retest_15bp_15m_full_margin_10x_filtered"

INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT = 0.25
FAST_FAIL_ENABLED = False
TAKE_PARTIAL_AT_R = 2.5
PARTIAL_CLOSE_SIZE = 0.50
ENABLE_BREAKEVEN_AFTER_PARTIAL = True
BREAKEVEN_OFFSET_PCT = 0.00
TRAIL_AFTER_R = 2.5
TRAIL_LOOKBACK_BARS = 5
TRAIL_BUFFER_PCT = 0.12

MAX_CANDIDATE_ABS_MINUTE_DISTANCE = 10.0
MIN_ENTRY_VS_TRADER_PRICE_PCT = -1.0
MAX_STOP_DISTANCE_PCT = 0.40


def get_index_at_or_after(df: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    idx = int(df["open_time"].searchsorted(ts, side="left"))
    if idx >= len(df):
        return None
    return idx


def simulate_limit_retest_entry(
    df: pd.DataFrame,
    anchor_idx: int,
    stop_price_initial: float,
) -> tuple[int | None, pd.Timestamp | None, float | None, float | None]:
    if anchor_idx is None or anchor_idx >= len(df):
        return None, None, None, None

    anchor_bar = df.iloc[anchor_idx]
    order_price = float(anchor_bar["close"]) * (1.0 + ENTRY_RETEST_ABOVE_CLOSE_PCT / 100.0)
    if order_price >= stop_price_initial:
        return None, anchor_bar["open_time"], order_price, None

    last_idx = min(anchor_idx + ENTRY_WAIT_BARS, len(df) - 1)
    for j in range(anchor_idx + 1, last_idx + 1):
        bar = df.iloc[j]
        if float(bar["high"]) >= order_price:
            entry_price = order_price * (1.0 + LIMIT_FILL_SLIPPAGE_PCT / 100.0)
            return j, bar["open_time"], order_price, entry_price
    return None, anchor_bar["open_time"], order_price, None


def run_trade_exit_short_v3(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    qty: float,
    fee_entry: float,
    balance: float,
    signal_time: pd.Timestamp,
    trigger_time: pd.Timestamp,
    confirm_time: pd.Timestamp,
    order_placed_time: pd.Timestamp,
    trigger_bar: pd.Series,
    confirm_bar: pd.Series,
    row_context: pd.Series,
    order_price: float,
    stop_price_initial: float,
) -> tuple[Trade, pd.Timestamp]:
    stop_current = stop_price_initial
    best_low = entry_price
    best_high = entry_price
    entry_time = df.iloc[entry_idx]["open_time"]

    partial_taken = False
    partial_realized_pnl = 0.0
    remaining_qty = qty
    trail_active = False

    exit_reason = "dataset_end"
    fallback_idx = min(entry_idx + MAX_BARS_SCAN, len(df) - 1)
    exit_time = df.iloc[fallback_idx]["open_time"]
    exit_price = float(df.iloc[fallback_idx]["close"])

    risk_per_unit = stop_price_initial - entry_price
    if risk_per_unit <= 0:
        raise ValueError("risk_per_unit must be positive")

    target_partial_price = entry_price - TAKE_PARTIAL_AT_R * risk_per_unit

    for j in range(entry_idx + 1, fallback_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        best_low = min(best_low, low)
        best_high = max(best_high, high)
        favorable_r_now = (entry_price - best_low) / risk_per_unit

        if (not partial_taken) and low <= target_partial_price:
            partial_qty = remaining_qty * PARTIAL_CLOSE_SIZE
            partial_exit_price = target_partial_price * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            partial_fee = (partial_qty * partial_exit_price) * (FEE_PER_SIDE_PCT / 100.0)
            partial_realized_pnl += (entry_price - partial_exit_price) * partial_qty - partial_fee
            remaining_qty -= partial_qty
            partial_taken = True
            if ENABLE_BREAKEVEN_AFTER_PARTIAL:
                stop_current = min(stop_current, entry_price * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0))

        if favorable_r_now >= TRAIL_AFTER_R:
            trail_active = True

        if trail_active and remaining_qty > 0:
            lookback_start = max(entry_idx + 1, j - TRAIL_LOOKBACK_BARS + 1)
            recent_high = float(df.iloc[lookback_start : j + 1]["high"].max())
            candidate_stop = recent_high * (1.0 + TRAIL_BUFFER_PCT / 100.0)
            stop_current = min(stop_current, candidate_stop)

        if high >= stop_current:
            exit_price = stop_current * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            exit_time = bar["open_time"]
            exit_reason = "managed_stop" if stop_current < stop_price_initial else "stop_loss"
            break

    gross_pnl_remaining = (entry_price - exit_price) * remaining_qty
    fee_exit_remaining = (remaining_qty * exit_price) * (FEE_PER_SIDE_PCT / 100.0)
    pnl = partial_realized_pnl + gross_pnl_remaining - fee_entry - fee_exit_remaining
    balance_after = balance + pnl

    trade = Trade(
        signal_time=signal_time,
        trigger_time=trigger_time,
        confirm_time=confirm_time,
        order_placed_time=order_placed_time,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_mode=ENTRY_MODE_NAME_V3,
        order_price=float(order_price) if pd.notna(order_price) else np.nan,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        side="Short",
        exit_reason=exit_reason,
        trigger_upper_wick_ratio=float(trigger_bar["upper_wick_ratio"]),
        trigger_body_ratio=float(trigger_bar["body_ratio"]),
        trigger_range_pct=float(trigger_bar["range_pct"]),
        confirm_range_pct=float(confirm_bar["range_pct"]),
        move_3m_pct=float(trigger_bar["move_3m_pct"]),
        move_5m_pct=float(trigger_bar["move_5m_pct"]),
        green_count_last_5=float(trigger_bar["green_count_last_5"]),
        bull_body_sum_5m_pct=float(trigger_bar["bull_body_sum_5m_pct"]),
        distance_above_ema7_1m_pct=float(trigger_bar["distance_above_ema7_1m_pct"]),
        rel_volume_1m=float(trigger_bar["relative_volume_1m"]),
        trigger_high=float(trigger_bar["high"]),
        trigger_low=float(trigger_bar["low"]),
        stop_price_initial=float(stop_price_initial),
        stop_price_final=float(stop_current),
        balance_before=float(balance),
        balance_after=float(balance_after),
        risk_amount_usd=float(balance),
        qty=float(qty),
        notional_usd=float(qty * entry_price),
        pnl_usd=float(pnl),
        pnl_pct_balance=float((pnl / balance) * 100.0) if balance > 0 else np.nan,
        mfe_pct=float((entry_price - best_low) / entry_price * 100.0),
        mae_pct=float((best_high - entry_price) / entry_price * 100.0),
        hold_minutes=float((exit_time - entry_time).total_seconds() / 60.0),
        context_dist_4h_high_pct=float(row_context["dist_4h_high_pct"]),
        context_bounce_60m_pct=float(row_context["bounce_60m_pct"]),
        context_stretch_above_ema20_5m_pct=float(row_context["stretch_above_ema20_5m_pct"]),
    )
    return trade, exit_time


def run_candidate_execution_backtest_v3(
    df: pd.DataFrame,
    nearest_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades = []
    skipped_unfilled = 0
    skipped_invalid_stop = 0
    skipped_far_candidate = 0
    skipped_bad_price = 0
    skipped_wide_stop = 0

    for _, row in nearest_candidates.iterrows():
        if float(row["candidate_abs_minute_distance"]) > MAX_CANDIDATE_ABS_MINUTE_DISTANCE:
            skipped_far_candidate += 1
            continue

        anchor_idx = get_index_at_or_after(df, row["candidate_time"])
        if anchor_idx is None or anchor_idx + 1 >= len(df):
            continue

        trigger_bar = df.iloc[anchor_idx]
        confirm_bar = df.iloc[min(anchor_idx + 1, len(df) - 1)]
        row_context = trigger_bar

        failure_high = float(max(trigger_bar["high"], confirm_bar["high"]))
        stop_price_initial = failure_high * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0)

        fill_idx, order_placed_time, order_price, entry_price = simulate_limit_retest_entry(
            df=df,
            anchor_idx=anchor_idx,
            stop_price_initial=stop_price_initial,
        )
        if fill_idx is None or order_placed_time is None or entry_price is None:
            skipped_unfilled += 1
            continue

        entry_vs_trader_price_pct = (float(entry_price) - float(row["trade_avg_cost"])) / float(row["trade_avg_cost"]) * 100.0
        if entry_vs_trader_price_pct < MIN_ENTRY_VS_TRADER_PRICE_PCT:
            skipped_bad_price += 1
            continue

        stop_distance = stop_price_initial - entry_price
        if stop_distance <= 0:
            skipped_invalid_stop += 1
            continue

        stop_distance_pct = stop_distance / entry_price * 100.0
        if stop_distance_pct > MAX_STOP_DISTANCE_PCT:
            skipped_wide_stop += 1
            continue

        qty = (balance * LEVERAGE) / entry_price if USE_FULL_MARGIN else (balance * LEVERAGE) / entry_price
        if qty <= 0:
            skipped_invalid_stop += 1
            continue

        notional = qty * entry_price
        fee_entry = notional * (FEE_PER_SIDE_PCT / 100.0)

        trade, _ = run_trade_exit_short_v3(
            df=df,
            entry_idx=fill_idx,
            entry_price=float(entry_price),
            qty=float(qty),
            fee_entry=float(fee_entry),
            balance=float(balance),
            signal_time=row["candidate_time"],
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
        payload["trade_date"] = row["trade_date"]
        payload["candidate_time"] = row["candidate_time"]
        payload["candidate_score"] = row["candidate_score"]
        payload["candidate_abs_minute_distance"] = row["candidate_abs_minute_distance"]
        payload["candidate_price"] = row["candidate_price"]
        payload["trade_opened_time"] = row["trade_opened_time"]
        payload["trade_closed_time"] = row["trade_closed_time"]
        payload["trade_avg_cost"] = row["trade_avg_cost"]
        payload["trade_avg_close_price"] = row["trade_avg_close_price"]
        payload["trader_return_pct"] = row["trader_return_pct"]
        payload["entry_vs_trader_price_pct"] = entry_vs_trader_price_pct
        payload["delta_vs_trader_return_pct"] = float(((trade.entry_price - trade.exit_price) / trade.entry_price * 100.0) - row["trader_return_pct"])
        payload["stop_distance_pct"] = stop_distance_pct
        trades.append(payload)
        balance = trade.balance_after

    trades_df = pd.DataFrame(trades)
    summary = {
        "candidate_count": int(len(nearest_candidates)),
        "filled_trade_count": int(len(trades_df)),
        "skipped_unfilled": int(skipped_unfilled),
        "skipped_invalid_stop": int(skipped_invalid_stop),
        "skipped_far_candidate": int(skipped_far_candidate),
        "skipped_bad_price": int(skipped_bad_price),
        "skipped_wide_stop": int(skipped_wide_stop),
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
            "median_entry_vs_trader_price_pct": float(trades_df["entry_vs_trader_price_pct"].median()),
            "median_candidate_abs_minute_distance": float(trades_df["candidate_abs_minute_distance"].median()),
            "median_stop_distance_pct": float(trades_df["stop_distance_pct"].median()),
        }
    )
    return trades_df, summary


def write_summary(summary: dict, match_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT EXECUTION BACKTEST V3",
        "===============================",
        "",
        f"entry_mode = {ENTRY_MODE_NAME_V3}",
        f"use_full_margin = {USE_FULL_MARGIN}",
        f"leverage = {LEVERAGE:.2f}",
        f"max_candidate_abs_minute_distance = {MAX_CANDIDATE_ABS_MINUTE_DISTANCE:.2f}",
        f"min_entry_vs_trader_price_pct = {MIN_ENTRY_VS_TRADER_PRICE_PCT:.2f}",
        f"max_stop_distance_pct = {MAX_STOP_DISTANCE_PCT:.2f}",
        f"candidate_count = {summary.get('candidate_count', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
        f"skipped_far_candidate = {summary.get('skipped_far_candidate', 0)}",
        f"skipped_bad_price = {summary.get('skipped_bad_price', 0)}",
        f"skipped_wide_stop = {summary.get('skipped_wide_stop', 0)}",
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
                f"median_entry_vs_trader_price_pct = {summary['median_entry_vs_trader_price_pct']:.4f}",
                f"median_candidate_abs_minute_distance = {summary['median_candidate_abs_minute_distance']:.2f}",
                f"median_stop_distance_pct = {summary['median_stop_distance_pct']:.4f}",
            ]
        )

    if not match_df.empty:
        lines.extend(
            [
                "",
                f"same_result_direction_share = {match_df['same_result_direction'].mean() * 100.0:.2f}%",
                f"median_delta_vs_trader_return_pct = {match_df['delta_vs_trader_return_pct'].median():.4f}",
            ]
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    features = build_features(candles)
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH)
    ref_lookup = build_reference_lookup(load_reference_eth_shorts(BASE_DIR / Path(r"Another Trader/another_trader.json")))

    trades_df, summary = run_candidate_execution_backtest_v3(features, nearest_candidates)
    match_df = build_match_summary(trades_df, ref_lookup)

    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "match_summary.csv", index=False)
    write_summary(summary, match_df)

    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
