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
    ENTRY_MODE_NAME_V3,
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT as INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT_V3,
    MAX_CANDIDATE_ABS_MINUTE_DISTANCE,
    MAX_STOP_DISTANCE_PCT,
    MIN_ENTRY_VS_TRADER_PRICE_PCT,
    run_trade_exit_short_v3,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_regime_aware_prototype"
REGIME_TAG_PATH = BASE_DIR / "fun_backtests" / "eth_short_day_regime_split_analysis" / "day_regime_tagged.csv"
TARGET_REGIMES = {"mixed_transition", "near_entry_breakdown_noise"}


def load_regime_tags(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df[["trade_date", "day_regime"]].dropna().drop_duplicates("trade_date").reset_index(drop=True)


def run_regime_aware_backtest(
    df: pd.DataFrame,
    nearest_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades: list[dict] = []

    skipped_non_target = 0
    skipped_unfilled = 0
    skipped_invalid_stop = 0
    skipped_bad_price = 0
    skipped_far_candidate = 0
    skipped_wide_stop = 0

    for _, row in nearest_candidates.iterrows():
        regime = str(row["day_regime"])
        if regime not in TARGET_REGIMES:
            skipped_non_target += 1
            continue

        anchor_idx = get_index_at_or_after(df, row["candidate_time"])
        if anchor_idx is None or anchor_idx + 1 >= len(df):
            continue

        trigger_bar = df.iloc[anchor_idx]
        confirm_bar = df.iloc[min(anchor_idx + 1, len(df) - 1)]
        row_context = trigger_bar

        if regime == "mixed_transition":
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
                entry_mode="regime_v1_mixed_transition",
                row_context=row_context,
                order_price=float(order_price) if order_price is not None else np.nan,
                stop_price_initial=float(stop_price_initial),
            )

            payload = asdict(trade)
            payload["regime"] = regime
            payload["regime_engine"] = "v1"
            payload["stop_distance_pct"] = (stop_distance / entry_price) * 100.0

        else:
            if float(row["candidate_abs_minute_distance"]) > MAX_CANDIDATE_ABS_MINUTE_DISTANCE:
                skipped_far_candidate += 1
                continue

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

            entry_vs_trader_price_pct = (float(entry_price) - float(row["trade_avg_cost"])) / float(row["trade_avg_cost"]) * 100.0
            if entry_vs_trader_price_pct < MIN_ENTRY_VS_TRADER_PRICE_PCT:
                skipped_bad_price += 1
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
            payload["regime"] = regime
            payload["regime_engine"] = "v3"
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
        trades.append(payload)
        balance = float(payload["balance_after"])

    trades_df = pd.DataFrame(trades)
    summary = {
        "candidate_count": int(len(nearest_candidates)),
        "filled_trade_count": int(len(trades_df)),
        "skipped_non_target": int(skipped_non_target),
        "skipped_unfilled": int(skipped_unfilled),
        "skipped_invalid_stop": int(skipped_invalid_stop),
        "skipped_bad_price": int(skipped_bad_price),
        "skipped_far_candidate": int(skipped_far_candidate),
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


def write_summary(summary: dict, match_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT REGIME AWARE PROTOTYPE",
        "================================",
        "",
        "routing = mixed_transition -> v1, near_entry_breakdown_noise -> v3",
        f"candidate_count = {summary.get('candidate_count', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
        f"skipped_non_target = {summary.get('skipped_non_target', 0)}",
        f"skipped_unfilled = {summary.get('skipped_unfilled', 0)}",
        f"skipped_invalid_stop = {summary.get('skipped_invalid_stop', 0)}",
        f"skipped_bad_price = {summary.get('skipped_bad_price', 0)}",
        f"skipped_far_candidate = {summary.get('skipped_far_candidate', 0)}",
        f"skipped_wide_stop = {summary.get('skipped_wide_stop', 0)}",
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
        lines.extend(
            [
                "",
                "Engine Split",
                "------------",
            ]
        )
        for engine, group in trades_df.groupby("regime_engine", sort=True):
            lines.append(
                f"{engine}: trade_count={len(group)}, win_rate_pct={(group['pnl_usd'] > 0).mean() * 100.0:.2f}, median_pnl_usd={group['pnl_usd'].median():.4f}"
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
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH)
    regime_tags = load_regime_tags(REGIME_TAG_PATH)
    nearest_candidates = nearest_candidates.merge(regime_tags, on="trade_date", how="left")
    nearest_candidates = nearest_candidates.sort_values("trade_opened_time").reset_index(drop=True)

    ref_lookup = build_reference_lookup(load_reference_eth_shorts(BASE_DIR / Path(r"Another Trader/another_trader.json")))
    trades_df, summary = run_regime_aware_backtest(features, nearest_candidates)
    match_df = build_match_summary(trades_df, ref_lookup)

    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "match_summary.csv", index=False)
    write_summary(summary, match_df, trades_df)

    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
