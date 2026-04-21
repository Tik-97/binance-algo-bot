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
    LIMIT_FILL_SLIPPAGE_PCT,
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT,
    run_trade_exit_short,
    build_features,
    load_candles,
    load_reference_eth_shorts,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_execution_backtest_v1"
NEAREST_CANDIDATES_PATH = BASE_DIR / "fun_backtests" / "eth_short_execution_prototype_v1" / "nearest_candidates.csv"

ENTRY_WAIT_BARS = 15
ENTRY_RETEST_ABOVE_CLOSE_PCT = 0.15
ENTRY_MODE_NAME = "candidate_limit_retest_15bp_15m"


def load_nearest_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["trade_opened_time", "trade_closed_time", "candidate_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    numeric_cols = [
        "trade_avg_cost",
        "trade_avg_close_price",
        "trader_return_pct",
        "candidate_price",
        "candidate_score",
        "candidate_minute_distance",
        "candidate_abs_minute_distance",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_opened_time", "trade_closed_time", "candidate_time", "trade_avg_cost"]).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.sort_values("trade_opened_time").reset_index(drop=True)


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


def build_reference_lookup(ref_df: pd.DataFrame) -> pd.DataFrame:
    lookup = ref_df.copy()
    lookup["trade_date"] = lookup["opened_time"].dt.strftime("%Y-%m-%d")
    lookup["ref_result"] = np.where(
        lookup["closingPnl"] > 0,
        "win",
        np.where(lookup["closingPnl"] < 0, "loss", "flat"),
    )
    lookup = lookup[
        ["trade_date", "opened_time", "closed_time", "avgCost", "avgClosePrice", "closingPnl", "ref_result"]
    ].rename(
        columns={
            "opened_time": "trade_opened_time",
            "closed_time": "ref_closed_time",
            "avgCost": "ref_avg_cost",
            "avgClosePrice": "ref_avg_close_price",
            "closingPnl": "ref_closing_pnl",
        }
    )
    return lookup.drop_duplicates(subset=["trade_opened_time"]).reset_index(drop=True)


def run_candidate_execution_backtest(
    df: pd.DataFrame,
    nearest_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades = []
    skipped_unfilled = 0
    skipped_invalid_stop = 0

    for _, row in nearest_candidates.iterrows():
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
            entry_mode=ENTRY_MODE_NAME,
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
        payload["entry_vs_trader_price_pct"] = (float(trade.entry_price) - float(row["trade_avg_cost"])) / float(row["trade_avg_cost"]) * 100.0
        payload["delta_vs_trader_return_pct"] = float(((trade.entry_price - trade.exit_price) / trade.entry_price * 100.0) - row["trader_return_pct"])
        trades.append(payload)
        balance = trade.balance_after

    trades_df = pd.DataFrame(trades)
    summary = {
        "candidate_count": int(len(nearest_candidates)),
        "filled_trade_count": int(len(trades_df)),
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
            "median_entry_vs_trader_price_pct": float(trades_df["entry_vs_trader_price_pct"].median()),
            "median_candidate_abs_minute_distance": float(trades_df["candidate_abs_minute_distance"].median()),
        }
    )
    return trades_df, summary


def build_match_summary(trades_df: pd.DataFrame, ref_lookup: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    merged = trades_df.merge(ref_lookup, on=["trade_date", "trade_opened_time"], how="left")
    merged["bt_result"] = np.where(
        merged["pnl_usd"] > 0,
        "win",
        np.where(merged["pnl_usd"] < 0, "loss", "flat"),
    )
    merged["same_result_direction"] = merged["bt_result"].eq(merged["ref_result"])
    return merged


def write_summary(summary: dict, match_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT EXECUTION BACKTEST V1",
        "===============================",
        "",
        f"entry_mode = {ENTRY_MODE_NAME}",
        f"candidate_count = {summary.get('candidate_count', 0)}",
        f"filled_trade_count = {summary.get('filled_trade_count', 0)}",
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

    trades_df, summary = run_candidate_execution_backtest(features, nearest_candidates)
    match_df = build_match_summary(trades_df, ref_lookup)

    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    match_df.to_csv(OUTPUT_DIR / "match_summary.csv", index=False)
    write_summary(summary, match_df)

    print(f"Filled trades: {summary.get('filled_trade_count', 0)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
