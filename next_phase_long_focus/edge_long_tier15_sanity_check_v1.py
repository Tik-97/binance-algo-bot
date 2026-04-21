from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1
import next_phase_long_focus.edge_long_tier15_core_plus_satellite3_v1 as strategy


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_tier15_sanity_check_v1"


def run_tier15_backtest(
    featured: pd.DataFrame,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    original_is_base_context = base.is_base_context
    original_find_trigger_index = base.find_trigger_index
    try:
        base.is_base_context = strategy.make_tier1_context_gate()
        base.find_trigger_index = strategy.make_route_aware_find_trigger()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_is_base_context
        base.find_trigger_index = original_find_trigger_index

    trades_df = trades_df.copy()
    if not trades_df.empty:
        trades_df["route"] = trades_df.apply(strategy.classify_trade_route, axis=1)
    return trades_df, summary


def compute_metrics(trades_df: pd.DataFrame, label: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    trade_count = int(len(trades_df))
    net_pnl = float(trades_df["pnl_usd"].sum()) if trade_count else 0.0
    win_rate = float((trades_df["pnl_usd"] > 0).mean() * 100.0) if trade_count else np.nan
    gross_profit = float(trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"].sum()) if trade_count else 0.0
    gross_loss = float(-trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"].sum()) if trade_count else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    ending_balance = float(base.STARTING_BALANCE + net_pnl)
    expectancy = float(net_pnl / trade_count) if trade_count else np.nan
    max_dd = 0.0
    if trade_count:
        equity = pd.Series([base.STARTING_BALANCE] + trades_df["balance_after"].tolist(), dtype=float)
        running_peak = equity.cummax()
        dd = (running_peak - equity) / running_peak.replace(0, np.nan) * 100.0
        max_dd = float(dd.max())
    return {
        "window": label,
        "bt_start": start,
        "bt_end": end,
        "trade_count": trade_count,
        "net_pnl_usd": round(net_pnl, 4),
        "win_rate_pct": round(win_rate, 2) if not pd.isna(win_rate) else np.nan,
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
        "ending_balance": round(ending_balance, 2),
        "expectancy_usd": round(expectancy, 4) if not pd.isna(expectancy) else np.nan,
        "max_drawdown_pct": round(max_dd, 2),
    }


def build_time_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    span = end - start
    midpoint = start + span / 2
    q1_end = start + span / 4
    q2_end = start + span / 2
    q3_end = start + span * 3 / 4
    one_minute = pd.Timedelta(minutes=1)
    return [
        ("first_half", start, midpoint),
        ("second_half", midpoint + one_minute, end),
        ("q1", start, q1_end),
        ("q2", q1_end + one_minute, q2_end),
        ("q3", q2_end + one_minute, q3_end),
        ("q4", q3_end + one_minute, end),
    ]


def assign_trade_periods(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df.copy()
    frame = trades_df.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["year_month"] = frame["entry_time"].dt.strftime("%Y-%m")
    midpoint = frame["entry_time"].min() + (frame["entry_time"].max() - frame["entry_time"].min()) / 2
    frame["half"] = np.where(frame["entry_time"] <= midpoint, "first_half", "second_half")

    q1 = frame["entry_time"].min() + (frame["entry_time"].max() - frame["entry_time"].min()) / 4
    q2 = frame["entry_time"].min() + (frame["entry_time"].max() - frame["entry_time"].min()) / 2
    q3 = frame["entry_time"].min() + (frame["entry_time"].max() - frame["entry_time"].min()) * 3 / 4

    def classify_quarter(ts: pd.Timestamp) -> str:
        if ts <= q1:
            return "q1"
        if ts <= q2:
            return "q2"
        if ts <= q3:
            return "q3"
        return "q4"

    frame["quarter"] = frame["entry_time"].apply(classify_quarter)
    return frame


def grouped_trade_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, part in df.groupby(group_col, dropna=False):
        trade_count = int(len(part))
        net_pnl = float(part["pnl_usd"].sum())
        win_rate = float((part["pnl_usd"] > 0).mean() * 100.0) if trade_count else 0.0
        gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        rows.append(
            {
                group_col: value,
                "trade_count": trade_count,
                "net_pnl_usd": round(net_pnl, 4),
                "win_rate_pct": round(win_rate, 2),
                "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
                "expectancy_usd": round(net_pnl / trade_count, 4) if trade_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bt_start = featured["open_time"].min()
    bt_end = featured["open_time"].max()

    full_trades, full_summary = run_tier15_backtest(featured, bt_start=bt_start, bt_end=bt_end)

    windows = build_time_windows(bt_start, bt_end)
    window_rows: list[dict[str, object]] = []
    for label, start, end in windows:
        trades_df, _ = run_tier15_backtest(featured, bt_start=start, bt_end=end)
        window_rows.append(compute_metrics(trades_df, label, start, end))

    window_summary = pd.DataFrame(window_rows)

    trades_with_periods = assign_trade_periods(full_trades)
    half_trade_summary = grouped_trade_summary(trades_with_periods, "half")
    quarter_trade_summary = grouped_trade_summary(trades_with_periods, "quarter")
    month_trade_summary = grouped_trade_summary(trades_with_periods, "year_month")
    route_half_summary = grouped_trade_summary(trades_with_periods, "route")

    positive_windows = int((window_summary["net_pnl_usd"] > 0).sum())
    nonempty_windows = int((window_summary["trade_count"] > 0).sum())
    both_halves_positive = bool(
        (window_summary.loc[window_summary["window"].isin(["first_half", "second_half"]), "net_pnl_usd"] > 0).all()
    )

    lines = [
        "EDGE LONG TIER1.5 SANITY CHECK V1",
        "=================================",
        "",
        "base_strategy = edge_long_tier15_core_plus_satellite3_v1.py",
        "",
        f"full_trade_count = {full_summary['trade_count']}",
        f"full_ending_balance = {full_summary['ending_balance']:.2f}",
        f"full_win_rate_pct = {full_summary['win_rate_pct']:.2f}" if not pd.isna(full_summary["win_rate_pct"]) else "full_win_rate_pct = n/a",
        f"full_profit_factor = {full_summary['profit_factor']:.3f}" if not pd.isna(full_summary["profit_factor"]) else "full_profit_factor = n/a",
        "",
        f"nonempty_windows = {nonempty_windows}",
        f"positive_windows = {positive_windows}",
        f"both_halves_positive = {both_halves_positive}",
        "",
        "sanity_read:",
        "- check whether first half and second half are both profitable",
        "- check whether profit is spread across more than one quarter",
        "- check whether routes are concentrated in one late pocket or appear across time",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    full_trades.to_csv(OUTPUT_DIR / "full_trades.csv", index=False)
    window_summary.to_csv(OUTPUT_DIR / "window_summary.csv", index=False)
    trades_with_periods.to_csv(OUTPUT_DIR / "trades_with_periods.csv", index=False)
    half_trade_summary.to_csv(OUTPUT_DIR / "half_trade_summary.csv", index=False)
    quarter_trade_summary.to_csv(OUTPUT_DIR / "quarter_trade_summary.csv", index=False)
    month_trade_summary.to_csv(OUTPUT_DIR / "month_trade_summary.csv", index=False)
    route_half_summary.to_csv(OUTPUT_DIR / "route_summary.csv", index=False)

    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
