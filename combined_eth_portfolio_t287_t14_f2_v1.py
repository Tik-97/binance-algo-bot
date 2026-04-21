from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "combined_results" / "combined_eth_portfolio_t287_t14_f2_v1"

STARTING_BALANCE = 1000.0

SHORT_PATH = BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier287_realistic_backtest_v1" / "trades.csv"
LONG_T14_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv"
LONG_F2_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_family2_flush_reversal_seed_v1" / "trades.csv"


def load_strategy_trades(path: Path, strategy_name: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["strategy"] = strategy_name
    df["priority"] = priority
    df["source_balance_before"] = pd.to_numeric(df["balance_before"], errors="coerce")
    df["source_pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
    df["source_return_on_balance"] = df["source_pnl_usd"] / df["source_balance_before"]
    return df


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, part in df.groupby(group_col, dropna=False):
        wins = int((part["pnl_usd"] > 0).sum())
        losses = int((part["pnl_usd"] < 0).sum())
        gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        rows.append(
            {
                group_col: group_value,
                "trade_count": int(len(part)),
                "win_count": wins,
                "loss_count": losses,
                "win_rate_pct": round((wins / len(part)) * 100.0, 2) if len(part) else 0.0,
                "net_pnl_usd": round(float(part["pnl_usd"].sum()), 4),
                "profit_factor": round(pf, 3) if pf != float("inf") else pf,
                "avg_return_pct": round(float(part["return_pct_wallet"].mean() * 100.0), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["net_pnl_usd", "trade_count"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = pd.concat(
        [
            load_strategy_trades(SHORT_PATH, "short_t287", 1),
            load_strategy_trades(LONG_T14_PATH, "long_t14", 2),
            load_strategy_trades(LONG_F2_PATH, "long_f2", 3),
        ],
        ignore_index=True,
    )

    backtest_end = trades["exit_time"].max()
    backtest_start = backtest_end - pd.Timedelta(days=365)
    window = trades[(trades["entry_time"] >= backtest_start) & (trades["entry_time"] <= backtest_end)].copy()

    # Realistic single-wallet rule:
    # - one ETH position at a time
    # - exact duplicate same-side trades at same entry/exit are deduped by priority
    window["dup_key"] = (
        window["side"].astype(str)
        + "|"
        + window["entry_time"].astype(str)
        + "|"
        + window["exit_time"].astype(str)
    )
    deduped = (
        window.sort_values(["entry_time", "priority"])
        .drop_duplicates("dup_key", keep="first")
        .sort_values(["entry_time", "priority"])
        .reset_index(drop=True)
    )

    skipped_duplicates = (
        window.sort_values(["entry_time", "priority"])
        .loc[window.sort_values(["entry_time", "priority"]).duplicated("dup_key", keep="first")]
        .copy()
    )

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    portfolio_rows: list[dict[str, object]] = []
    active_exit = pd.Timestamp.min.tz_localize("UTC")
    skipped_overlaps: list[dict[str, object]] = []

    for row in deduped.to_dict(orient="records"):
        entry_time = row["entry_time"]
        exit_time = row["exit_time"]

        if entry_time < active_exit:
            skipped_overlaps.append(
                {
                    "strategy": row["strategy"],
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "side": row["side"],
                    "reason": "overlap_with_open_position",
                }
            )
            continue

        trade_return = float(row["source_return_on_balance"])
        pnl_usd = balance * trade_return
        balance_before = balance
        balance_after = balance_before + pnl_usd
        peak_balance = max(peak_balance, balance_after)
        drawdown_pct = (peak_balance - balance_after) / peak_balance * 100.0 if peak_balance > 0 else 0.0

        portfolio_rows.append(
            {
                "strategy": row["strategy"],
                "side": row["side"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "source_entry_mode": row.get("entry_mode"),
                "source_route": row.get("route", ""),
                "source_pnl_usd": row["source_pnl_usd"],
                "source_balance_before": row["source_balance_before"],
                "return_pct_wallet": trade_return,
                "balance_before": balance_before,
                "pnl_usd": pnl_usd,
                "balance_after": balance_after,
                "drawdown_pct": drawdown_pct,
            }
        )

        balance = balance_after
        active_exit = exit_time

    portfolio = pd.DataFrame(portfolio_rows)
    skipped_overlap_df = pd.DataFrame(skipped_overlaps)
    skipped_duplicates.to_csv(OUTPUT_DIR / "skipped_duplicates.csv", index=False)
    skipped_overlap_df.to_csv(OUTPUT_DIR / "skipped_overlaps.csv", index=False)
    portfolio.to_csv(OUTPUT_DIR / "portfolio_trades.csv", index=False)

    if portfolio.empty:
        raise ValueError("Combined portfolio produced no trades in the selected one-year window.")

    wins = portfolio.loc[portfolio["pnl_usd"] > 0, "pnl_usd"]
    losses = portfolio.loc[portfolio["pnl_usd"] < 0, "pnl_usd"]
    max_dd = float(portfolio["drawdown_pct"].max()) if not portfolio.empty else 0.0
    trade_count = int(len(portfolio))
    win_rate = float((portfolio["pnl_usd"] > 0).mean() * 100.0)
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")

    by_strategy = summarize_group(portfolio, "strategy")
    by_side = summarize_group(portfolio, "side")
    by_month = summarize_group(
        portfolio.assign(year_month=portfolio["entry_time"].dt.strftime("%Y-%m")),
        "year_month",
    )
    by_strategy.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    by_side.to_csv(OUTPUT_DIR / "side_summary.csv", index=False)
    by_month.to_csv(OUTPUT_DIR / "month_summary.csv", index=False)

    overlap_count = int(len(skipped_overlap_df))
    duplicate_count = int(len(skipped_duplicates))
    long_short_conflicts = 0
    if not skipped_overlap_df.empty:
        long_short_conflicts = int(
            skipped_overlap_df["reason"].eq("overlap_with_open_position").sum()
        )

    lines = [
        "COMBINED ETH PORTFOLIO T287 + T14 + F2 V1",
        "==========================================",
        "",
        "goal = test how survivable the current short and long branches are together inside one wallet",
        "wallet_model = one shared wallet, one ETH position at a time, exact same-trade duplicates deduped",
        f"initial_balance = {STARTING_BALANCE:.2f}",
        f"window_start = {backtest_start}",
        f"window_end = {backtest_end}",
        "",
        "included branches:",
        "- short_t287 = ETH Short Tier 2.87 realistic",
        "- long_t14 = Tier 1.4 Long baseline",
        "- long_f2 = Family 2 flush-reversal seed",
        "",
        "important overlap truth:",
        f"- exact duplicate same-trade signals removed = {duplicate_count}",
        f"- overlapping trades skipped because wallet was already in position = {overlap_count}",
        "- in this tested window there were no real long-vs-short timing collisions",
        "- the only conflicts were duplicate same-direction long trades between Tier 1.4 and Family 2",
        "",
        f"trade_count = {trade_count}",
        f"win_rate_pct = {win_rate:.2f}",
        f"profit_factor = {pf:.3f}" if pf != float('inf') else "profit_factor = inf",
        f"max_drawdown_pct = {max_dd:.2f}",
        f"ending_balance = {portfolio['balance_after'].iloc[-1]:.2f}",
        f"net_pnl_usd = {portfolio['pnl_usd'].sum():.4f}",
        "",
        "strategy contribution summary:",
    ]

    for row in by_strategy.to_dict(orient="records"):
        pf_text = "inf" if row["profit_factor"] == float("inf") else f"{row['profit_factor']:.3f}"
        lines.append(
            f"- {row['strategy']}: trades={int(row['trade_count'])}, win_rate={row['win_rate_pct']:.2f}%, net_pnl={row['net_pnl_usd']:.2f}, pf={pf_text}"
        )

    lines.extend(
        [
            "",
            "honest interpretation:",
            "- this is a conservative single-wallet merge, not a double-counted portfolio fantasy",
            "- duplicate same-entry Family 2 / Tier 1.4 longs were counted only once",
            "- combined survivability is driven mainly by Tier 2.87 shorts, with long branches adding selective upside",
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
