from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ethShortTrader import (
    BASE_DIR,
    CANDLES_1M_PATH,
    STARTING_BALANCE,
    RISK_PER_TRADE_PCT,
    LEVERAGE,
    FEE_PER_SIDE_PCT,
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT,
    build_features,
    load_candles,
    load_reference_eth_shorts,
    run_trade_exit_short,
)
from eth_short_execution_backtest_v1 import (
    NEAREST_CANDIDATES_PATH,
    build_match_summary,
    build_reference_lookup,
    get_index_at_or_after,
    load_nearest_candidates,
    simulate_limit_retest_entry,
)
from eth_short_execution_backtest_v3 import (
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT as INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT_V3,
    MAX_STOP_DISTANCE_PCT,
    run_trade_exit_short_v3,
)
from eth_short_live_safe_regime_classifier import predict_live_safe_day_regime_from_bar
from eth_short_regime_aware_prototype_v3 import classify_near_noise_subtype_live


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_regime_aware_prototype_v4"


def run_regime_aware_backtest_v4(
    df: pd.DataFrame,
    nearest_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades: list[dict] = []

    skipped_non_target = 0
    skipped_unfilled = 0
    skipped_invalid_stop = 0
    skipped_wide_stop = 0
    skipped_strong_push = 0

    for _, row in nearest_candidates.iterrows():
        anchor_idx = get_index_at_or_after(df, row["candidate_time"])
        if anchor_idx is None or anchor_idx + 1 >= len(df):
            continue

        trigger_bar = df.iloc[anchor_idx]
        confirm_bar = df.iloc[min(anchor_idx + 1, len(df) - 1)]
        row_context = trigger_bar

        predicted_regime = predict_live_safe_day_regime_from_bar(trigger_bar)
        if predicted_regime == "other_non_target":
            skipped_non_target += 1
            continue

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
                df=df,
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
                entry_mode=f"regime_v4_{subtype}",
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
                df=df,
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
            payload["stop_distance_pct"] = stop_distance_pct

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
        payload["entry_vs_trader_price_pct"] = (float(payload["entry_price"]) - float(row["trade_avg_cost"])) / float(row["trade_avg_cost"]) * 100.0
        payload["delta_vs_trader_return_pct"] = float(((payload["entry_price"] - payload["exit_price"]) / payload["entry_price"] * 100.0) - row["trader_return_pct"])
        payload["predicted_regime"] = predicted_regime
        payload["regime_subtype"] = subtype
        payload["regime_engine"] = engine
        trades.append(payload)
        balance = float(payload["balance_after"])

    trades_df = pd.DataFrame(trades)
    summary = {
        "candidate_count": int(len(nearest_candidates)),
        "filled_trade_count": int(len(trades_df)),
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
            "median_entry_vs_trader_price_pct": float(trades_df["entry_vs_trader_price_pct"].median()),
            "median_candidate_abs_minute_distance": float(trades_df["candidate_abs_minute_distance"].median()),
            "median_stop_distance_pct": float(trades_df["stop_distance_pct"].median()),
        }
    )
    return trades_df, summary


def write_summary(summary: dict, match_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT REGIME AWARE PROTOTYPE V4",
        "===================================",
        "",
        "routing = live-safe top-level regime classifier + live-safe near-noise micro router",
        "note = removed day_regime oracle and removed trader-price pre-entry filter from the v3-style branch",
        "note = candidate set is still the nearest-candidate research baseline, so this is not yet a full standalone live engine",
        f"candidate_count = {summary.get('candidate_count', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
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
                f"median_entry_vs_trader_price_pct = {summary['median_entry_vs_trader_price_pct']:.4f}",
                f"median_candidate_abs_minute_distance = {summary['median_candidate_abs_minute_distance']:.2f}",
                f"median_stop_distance_pct = {summary['median_stop_distance_pct']:.4f}",
            ]
        )

    if not trades_df.empty:
        lines.extend(["", "Engine Split", "------------"])
        for (predicted_regime, subtype, engine), group in trades_df.groupby(["predicted_regime", "regime_subtype", "regime_engine"], sort=True):
            lines.append(
                f"{predicted_regime} / {subtype} / {engine}: trade_count={len(group)}, win_rate_pct={(group['pnl_usd'] > 0).mean() * 100.0:.2f}, median_pnl_usd={group['pnl_usd'].median():.4f}"
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

    features = build_features(load_candles(CANDLES_1M_PATH))
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH).sort_values("trade_opened_time").reset_index(drop=True)

    ref_lookup = build_reference_lookup(load_reference_eth_shorts(BASE_DIR / Path(r"Another Trader/another_trader.json")))
    trades_df, summary = run_regime_aware_backtest_v4(features, nearest_candidates)
    match_df = build_match_summary(trades_df, ref_lookup)

    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "match_summary.csv", index=False)
    write_summary(summary, match_df, trades_df)

    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
