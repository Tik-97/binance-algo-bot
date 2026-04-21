from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from eth_short_candidate_engine import (
    BASE_DIR,
    CANDLES_1M_PATH,
    JSON_TRADES_PATH,
    build_features,
    generate_candidates,
    load_candles,
)


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_execution_prototype_v1"
SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16

FEE_PER_SIDE_PCT = 0.04
MARKET_SLIPPAGE_PER_SIDE_PCT = 0.025
LIMIT_FILL_SLIPPAGE_PCT = 0.00

EXECUTION_MODES = [
    {
        "mode": "market_next_open",
        "type": "market",
        "wait_bars": 0,
        "retest_above_close_pct": 0.00,
    },
    {
        "mode": "limit_retest_5bp_10m",
        "type": "limit",
        "wait_bars": 10,
        "retest_above_close_pct": 0.05,
    },
    {
        "mode": "limit_retest_10bp_10m",
        "type": "limit",
        "wait_bars": 10,
        "retest_above_close_pct": 0.10,
    },
    {
        "mode": "limit_retest_15bp_15m",
        "type": "limit",
        "wait_bars": 15,
        "retest_above_close_pct": 0.15,
    },
]


def filter_session_reference_trades(ref: pd.DataFrame) -> pd.DataFrame:
    session_ref = ref[
        (ref["opened_time"].dt.hour >= SESSION_START_HOUR_UTC)
        & (ref["opened_time"].dt.hour < SESSION_END_HOUR_UTC)
    ].copy()
    session_ref["trade_date"] = session_ref["opened_time"].dt.date
    session_ref["trader_return_pct"] = (session_ref["avgCost"] - session_ref["avgClosePrice"]) / session_ref["avgCost"] * 100.0
    return session_ref.sort_values("opened_time").reset_index(drop=True)


def load_reference_trades_full(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    trades = pd.DataFrame(raw)
    trades = trades[
        (trades["symbol"] == "ETHUSDT")
        & (trades["side"] == "Short")
        & (trades["status"] == "All Closed")
    ].copy()
    trades["opened_time"] = pd.to_datetime(trades["opened"], unit="ms", utc=True, errors="coerce")
    trades["closed_time"] = pd.to_datetime(trades["closed"], unit="ms", utc=True, errors="coerce")
    trades["avgCost"] = pd.to_numeric(trades["avgCost"], errors="coerce")
    trades["avgClosePrice"] = pd.to_numeric(trades["avgClosePrice"], errors="coerce")
    return trades.dropna(subset=["opened_time", "closed_time", "avgCost", "avgClosePrice"]).sort_values("opened_time").reset_index(drop=True)


def attach_trade_date(frame: pd.DataFrame, time_col: str) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["trade_date"] = enriched[time_col].dt.date
    return enriched


def select_nearest_day_candidates(candidates: pd.DataFrame, session_ref: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, trade in session_ref.iterrows():
        day = candidates.loc[candidates["trade_date"].eq(trade["trade_date"])].copy()
        if day.empty:
            continue
        day["minute_distance_to_trade"] = (day["open_time"] - trade["opened_time"]).dt.total_seconds() / 60.0
        day["abs_minute_distance_to_trade"] = day["minute_distance_to_trade"].abs()
        best = day.sort_values(["abs_minute_distance_to_trade", "candidate_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "trade_date": trade["trade_date"],
                "trade_opened_time": trade["opened_time"],
                "trade_closed_time": trade["closed_time"],
                "trade_avg_cost": float(trade["avgCost"]),
                "trade_avg_close_price": float(trade["avgClosePrice"]),
                "trader_return_pct": float(trade["trader_return_pct"]),
                "candidate_time": best["open_time"],
                "candidate_price": float(best["close"]),
                "candidate_score": float(best["candidate_score"]),
                "candidate_minute_distance": float(best["minute_distance_to_trade"]),
                "candidate_abs_minute_distance": float(best["abs_minute_distance_to_trade"]),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_opened_time").reset_index(drop=True)


def get_index_at_or_after(df: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    pos = int(df["open_time"].searchsorted(ts, side="left"))
    if pos >= len(df):
        return None
    return pos


def simulate_execution_mode(
    candles: pd.DataFrame,
    anchor_time: pd.Timestamp,
    mode_config: dict,
) -> tuple[pd.Timestamp | None, float | None, float | None, str]:
    anchor_idx = get_index_at_or_after(candles, anchor_time)
    if anchor_idx is None:
        return None, None, None, "anchor_not_found"

    if mode_config["type"] == "market":
        fill_idx = min(anchor_idx + 1, len(candles) - 1)
        fill_bar = candles.iloc[fill_idx]
        raw_entry = float(fill_bar["open"])
        entry_price = raw_entry * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
        return fill_bar["open_time"], raw_entry, entry_price, "filled_market"

    anchor_close = float(candles.iloc[anchor_idx]["close"])
    order_price = anchor_close * (1.0 + float(mode_config["retest_above_close_pct"]) / 100.0)
    wait_bars = int(mode_config["wait_bars"])
    last_idx = min(anchor_idx + wait_bars, len(candles) - 1)
    for idx in range(anchor_idx + 1, last_idx + 1):
        bar = candles.iloc[idx]
        if float(bar["high"]) >= order_price:
            entry_price = order_price * (1.0 + LIMIT_FILL_SLIPPAGE_PCT / 100.0)
            return bar["open_time"], order_price, entry_price, "filled_limit"
    return None, order_price, None, "unfilled_limit"


def resolve_exit_price(candles: pd.DataFrame, exit_time: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    exit_idx = get_index_at_or_after(candles, exit_time)
    if exit_idx is None:
        return None, None
    exit_bar = candles.iloc[exit_idx]
    return exit_bar["open_time"], float(exit_bar["close"])


def compute_short_net_return_pct(entry_price: float, exit_price: float) -> float:
    gross_return_pct = (entry_price - exit_price) / entry_price * 100.0
    fees_pct = FEE_PER_SIDE_PCT * 2.0
    return gross_return_pct - fees_pct


def run_execution_experiment(candles: pd.DataFrame, nearest_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    candles = candles.sort_values("open_time").reset_index(drop=True)

    for _, trade in nearest_candidates.iterrows():
        day_candles = candles.loc[candles["open_time"].dt.date.eq(trade["trade_date"])].copy()
        if day_candles.empty:
            continue

        exit_fill_time, exit_fill_price = resolve_exit_price(candles, trade["trade_closed_time"])
        if exit_fill_time is None or exit_fill_price is None:
            continue

        for mode in EXECUTION_MODES:
            fill_time, raw_entry_price, entry_price, fill_status = simulate_execution_mode(
                day_candles,
                trade["candidate_time"],
                mode,
            )
            if fill_time is None or entry_price is None:
                rows.append(
                    {
                        **trade.to_dict(),
                        "execution_mode": mode["mode"],
                        "fill_status": fill_status,
                        "fill_time": pd.NaT,
                        "raw_entry_price": raw_entry_price,
                        "entry_price": np.nan,
                        "fill_delay_minutes_from_candidate": np.nan,
                        "entry_vs_trader_price_pct": np.nan,
                        "exit_fill_time": exit_fill_time,
                        "exit_fill_price": exit_fill_price,
                        "net_return_pct_same_exit": np.nan,
                        "delta_vs_trader_return_pct": np.nan,
                    }
                )
                continue

            net_return = compute_short_net_return_pct(float(entry_price), float(exit_fill_price))
            rows.append(
                {
                    **trade.to_dict(),
                    "execution_mode": mode["mode"],
                    "fill_status": fill_status,
                    "fill_time": fill_time,
                    "raw_entry_price": raw_entry_price,
                    "entry_price": float(entry_price),
                    "fill_delay_minutes_from_candidate": (fill_time - trade["candidate_time"]).total_seconds() / 60.0,
                    "entry_vs_trader_price_pct": (float(entry_price) - float(trade["trade_avg_cost"])) / float(trade["trade_avg_cost"]) * 100.0,
                    "exit_fill_time": exit_fill_time,
                    "exit_fill_price": float(exit_fill_price),
                    "net_return_pct_same_exit": float(net_return),
                    "delta_vs_trader_return_pct": float(net_return - trade["trader_return_pct"]),
                }
            )

    return pd.DataFrame(rows)


def build_mode_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for mode, group in results.groupby("execution_mode", sort=True):
        filled = group.loc[group["entry_price"].notna()].copy()
        rows.append(
            {
                "execution_mode": mode,
                "trade_count": int(len(group)),
                "filled_count": int(len(filled)),
                "fill_rate_pct": float(len(filled) / len(group) * 100.0) if len(group) else np.nan,
                "median_fill_delay_minutes_from_candidate": float(filled["fill_delay_minutes_from_candidate"].median()) if not filled.empty else np.nan,
                "median_entry_vs_trader_price_pct": float(filled["entry_vs_trader_price_pct"].median()) if not filled.empty else np.nan,
                "median_net_return_pct_same_exit": float(filled["net_return_pct_same_exit"].median()) if not filled.empty else np.nan,
                "median_delta_vs_trader_return_pct": float(filled["delta_vs_trader_return_pct"].median()) if not filled.empty else np.nan,
                "within_25bp_of_trader_entry_share": float(filled["entry_vs_trader_price_pct"].abs().le(0.25).mean() * 100.0) if not filled.empty else np.nan,
                "positive_same_exit_share": float(filled["net_return_pct_same_exit"].gt(0).mean() * 100.0) if not filled.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("execution_mode").reset_index(drop=True)


def write_summary(session_ref: pd.DataFrame, nearest_candidates: pd.DataFrame, mode_summary: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT EXECUTION PROTOTYPE V1",
        "================================",
        "",
        f"session_reference_trade_count = {len(session_ref)}",
        f"nearest_candidate_day_count = {len(nearest_candidates)}",
        f"baseline_nearest_within_15m_share = {nearest_candidates['candidate_abs_minute_distance'].le(15.0).mean() * 100.0:.2f}%",
        f"baseline_nearest_within_5m_share = {nearest_candidates['candidate_abs_minute_distance'].le(5.0).mean() * 100.0:.2f}%",
        f"baseline_nearest_median_abs_minute_distance = {nearest_candidates['candidate_abs_minute_distance'].median():.2f}",
        "",
    ]

    for _, row in mode_summary.iterrows():
        lines.extend(
            [
                row["execution_mode"],
                "-" * len(str(row["execution_mode"])),
                f"fill_rate_pct = {row['fill_rate_pct']:.2f}%",
                f"median_fill_delay_minutes_from_candidate = {row['median_fill_delay_minutes_from_candidate']:.2f}",
                f"median_entry_vs_trader_price_pct = {row['median_entry_vs_trader_price_pct']:.4f}",
                f"median_net_return_pct_same_exit = {row['median_net_return_pct_same_exit']:.4f}",
                f"median_delta_vs_trader_return_pct = {row['median_delta_vs_trader_return_pct']:.4f}",
                f"within_25bp_of_trader_entry_share = {row['within_25bp_of_trader_entry_share']:.2f}%",
                f"positive_same_exit_share = {row['positive_same_exit_share']:.2f}%",
                "",
            ]
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = attach_trade_date(load_candles(CANDLES_1M_PATH), "open_time")
    ref = filter_session_reference_trades(load_reference_trades_full(JSON_TRADES_PATH))
    candidates = attach_trade_date(generate_candidates(build_features(candles)), "open_time")

    nearest_candidates = select_nearest_day_candidates(candidates, ref)
    results = run_execution_experiment(candles, nearest_candidates)
    mode_summary = build_mode_summary(results)

    nearest_candidates.to_csv(OUTPUT_DIR / "nearest_candidates.csv", index=False)
    results.to_csv(OUTPUT_DIR / "execution_results.csv", index=False)
    mode_summary.to_csv(OUTPUT_DIR / "execution_mode_summary.csv", index=False)
    write_summary(ref, nearest_candidates, mode_summary)

    print(f"Session reference trades: {len(ref)}")
    print(f"Nearest candidate day count: {len(nearest_candidates)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
