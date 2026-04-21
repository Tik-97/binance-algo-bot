"""
Combined ETH Portfolio V2 — Full Balance × 10x Leverage Model
=============================================================
Starting balance : $1000
Position sizing  : 100% of balance × 10x leverage per trade
Fee model        : 0.04% taker both sides → 0.8% of balance per round-trip
Strategies       : short_t287 (p1) | long_t14 (p2) | long_f21 (p3) | long_bull_v3 (p4)
Arbitration      : short > long; t14 > f21 > bull_v3 on same-minute conflict
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

STARTING_BALANCE = 1000.0
LEVERAGE = 15.0
FEE_PER_SIDE_PCT = 0.04          # taker
ROUND_TRIP_FEE_PCT = 2 * FEE_PER_SIDE_PCT * LEVERAGE  # 0.8% of balance

SHORT_PATH   = BASE_DIR / "next_phase_edge_focus"  / "results" / "binance_tier287_realistic_backtest_v1" / "trades.csv"
LONG_T14_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv"
LONG_F21_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_family21_seed_plus_satellites_v1" / "trades.csv"
LONG_BV3_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_bull_continuation_v3" / "trades.csv"


def load(path: Path, strategy: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["strategy"]   = strategy
    df["priority"]   = priority
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"]  = pd.to_numeric(df["exit_price"],  errors="coerce")
    return df


def full_balance_return_pct(row: pd.Series) -> float:
    """Net % return on balance for one trade under full-balance × leverage model."""
    side = str(row["side"]).strip().upper()
    ep, xp = float(row["entry_price"]), float(row["exit_price"])
    if side in ("LONG", "BUY"):
        price_ret = (xp / ep - 1.0) * 100.0
    else:
        price_ret = (1.0 - xp / ep) * 100.0
    return price_ret * LEVERAGE - ROUND_TRIP_FEE_PCT


def summarize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat, part in df.groupby("strategy"):
        n = len(part)
        wins = (part["pnl_usd"] > 0).sum()
        gp = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        rows.append({
            "strategy":     strat,
            "trade_count":  n,
            "win_rate_pct": round(wins / n * 100, 2) if n else 0,
            "net_pnl_usd":  round(float(part["pnl_usd"].sum()), 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else pf,
        })
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for val, part in df.groupby(col, dropna=False):
        n = len(part)
        wins = (part["pnl_usd"] > 0).sum()
        gp = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        rows.append({
            col: val,
            "trade_count":  n,
            "win_rate_pct": round(wins / n * 100, 2) if n else 0,
            "net_pnl_usd":  round(float(part["pnl_usd"].sum()), 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else pf,
        })
    return pd.DataFrame(rows)


# =========================================================
# PLOTS
# =========================================================

STRATEGY_COLORS = {
    "short_t287":   "#e74c3c",
    "long_t14":     "#2ecc71",
    "long_f21":     "#3498db",
    "long_bull_v3": "#f39c12",
}


def _plot_equity_curve(portfolio: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        portfolio["exit_time"],
        portfolio["balance_after"],
        color="#1abc9c", linewidth=2, label="Portfolio",
    )
    ax.axhline(STARTING_BALANCE, color="#7f8c8d", linestyle="--", linewidth=1, label="Start")

    # Mark each trade by strategy
    for strat, color in STRATEGY_COLORS.items():
        sub = portfolio[portfolio["strategy"] == strat]
        wins = sub[sub["pnl_usd"] > 0]
        losses = sub[sub["pnl_usd"] < 0]
        ax.scatter(wins["exit_time"],   wins["balance_after"],   marker="^", color=color, s=40, alpha=0.8, label=f"{strat} win")
        ax.scatter(losses["exit_time"], losses["balance_after"], marker="v", color=color, s=40, alpha=0.8, label=f"{strat} loss")

    ax.set_title("Combined Portfolio — Equity Curve ($1000 start, 100% × 10x leverage)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Balance (USD)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_drawdown(portfolio: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(
        portfolio["exit_time"],
        portfolio["drawdown_pct"],
        0,
        color="#e74c3c", alpha=0.5, label="Drawdown %",
    )
    ax.plot(portfolio["exit_time"], portfolio["drawdown_pct"], color="#c0392b", linewidth=1)
    ax.set_title("Portfolio Drawdown from Peak", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_monthly_pnl(portfolio: pd.DataFrame, output: Path) -> None:
    portfolio = portfolio.copy()
    portfolio["year_month"] = portfolio["entry_time"].dt.strftime("%Y-%m")

    monthly = (
        portfolio.groupby(["year_month", "strategy"])["pnl_usd"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=list(STRATEGY_COLORS.keys()), fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    bottom = np.zeros(len(monthly))
    for strat in STRATEGY_COLORS:
        if strat in monthly.columns:
            vals = monthly[strat].values
            bars = ax.bar(
                monthly.index, vals, bottom=bottom,
                color=STRATEGY_COLORS[strat], label=strat, alpha=0.85,
            )
            bottom += vals

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Monthly P&L by Strategy (full-balance model)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("P&L (USD)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_per_strategy_equity(portfolio: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    strats = list(STRATEGY_COLORS.keys())

    for idx, strat in enumerate(strats):
        ax = axes[idx]
        sub = portfolio[portfolio["strategy"] == strat].copy()
        if sub.empty:
            ax.set_title(f"{strat} — no trades")
            continue
        # Cumulative P&L for this strategy (using portfolio pnl_usd, sequentially)
        sub = sub.sort_values("exit_time").reset_index(drop=True)
        cum_pnl = sub["pnl_usd"].cumsum() + STARTING_BALANCE
        ax.plot(sub["exit_time"], cum_pnl, color=STRATEGY_COLORS[strat], linewidth=2)
        wins = sub[sub["pnl_usd"] > 0]
        losses = sub[sub["pnl_usd"] < 0]
        ax.scatter(wins["exit_time"], wins["pnl_usd"].cumsum() + STARTING_BALANCE,
                   marker="^", color="green", s=30, alpha=0.8)
        ax.scatter(losses["exit_time"], losses["pnl_usd"].cumsum() + STARTING_BALANCE,
                   marker="v", color="red", s=30, alpha=0.8)
        n = len(sub)
        wr = (sub["pnl_usd"] > 0).mean() * 100
        gp = sub.loc[sub["pnl_usd"] > 0, "pnl_usd"].sum()
        gl = -sub.loc[sub["pnl_usd"] < 0, "pnl_usd"].sum()
        pf = gp / gl if gl > 0 else float("inf")
        ax.set_title(f"{strat}  |  n={n}  WR={wr:.0f}%  PF={pf:.2f}", fontsize=11)
        ax.axhline(STARTING_BALANCE, color="gray", linestyle="--", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Strategy Equity (cumulative, isolated view)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_trade_scatter(portfolio: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    for strat, color in STRATEGY_COLORS.items():
        sub = portfolio[portfolio["strategy"] == strat]
        ax.scatter(
            sub["entry_time"],
            sub["pnl_usd"],
            color=color, alpha=0.75, s=50, label=strat,
            marker="^" if strat != "short_t287" else "v",
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Trade P&L Scatter by Strategy (full-balance model)", fontsize=13)
    ax.set_xlabel("Entry Date")
    ax.set_ylabel("P&L per Trade (USD)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_trades = pd.concat(
        [
            load(SHORT_PATH,    "short_t287",   1),
            load(LONG_T14_PATH, "long_t14",     2),
            load(LONG_F21_PATH, "long_f21",     3),
            load(LONG_BV3_PATH, "long_bull_v3", 4),
        ],
        ignore_index=True,
    )

    # ── Dedup exact same-side same-entry/exit by priority ──
    all_trades["dup_key"] = (
        all_trades["side"].str.upper().str.strip()
        + "|" + all_trades["entry_time"].apply(lambda t: t.isoformat())
        + "|" + all_trades["exit_time"].apply(lambda t: t.isoformat())
    )
    sorted_all  = all_trades.sort_values(["entry_time", "priority"]).reset_index(drop=True)
    dup_mask    = sorted_all.duplicated("dup_key", keep="first")
    skipped_dups = sorted_all.loc[dup_mask].copy()
    deduped     = sorted_all.loc[~dup_mask].copy()
    exact_dup_count = int(dup_mask.sum())

    # ── Same-minute arbitration ──
    deduped["entry_minute"] = deduped["entry_time"].dt.floor("1min")
    same_min_long_conflicts = 0
    short_beat_long_count   = 0
    rows_to_drop: set[int] = set()

    for _, group in deduped.groupby("entry_minute"):
        strats = set(group["strategy"].tolist())
        has_short = "short_t287" in strats
        longs = [s for s in ["long_t14", "long_f21", "long_bull_v3"] if s in strats]

        if has_short and longs:
            rows_to_drop.update(group.loc[group["strategy"].isin(longs)].index.tolist())
            short_beat_long_count += 1
        elif len(longs) > 1:
            # Keep highest priority (lowest priority number), drop others
            keep = group.loc[group["strategy"].isin(longs)].sort_values("priority").index[0]
            drop_idx = group.loc[group["strategy"].isin(longs)].index.tolist()
            drop_idx.remove(keep)
            rows_to_drop.update(drop_idx)
            same_min_long_conflicts += 1

    arbitrated = deduped.loc[~deduped.index.isin(rows_to_drop)].copy()
    arbitrated = arbitrated.sort_values(["entry_time", "priority"]).reset_index(drop=True)

    # ── Check overlap stats ──
    print(f"\n=== OVERLAP / CONFLICT STATS ===")
    print(f"Total raw trades:          {len(all_trades)}")
    print(f"Exact duplicates removed:  {exact_dup_count}")
    print(f"Short beat long conflicts: {short_beat_long_count}")
    print(f"Same-min long conflicts:   {same_min_long_conflicts}")
    print(f"After arbitration:         {len(arbitrated)}")

    # Show any remaining time overlaps (informational)
    arbitrated_sorted = arbitrated.sort_values("entry_time").reset_index(drop=True)
    overlaps = []
    for i in range(1, len(arbitrated_sorted)):
        prev = arbitrated_sorted.iloc[i - 1]
        curr = arbitrated_sorted.iloc[i]
        if curr["entry_time"] < prev["exit_time"]:
            overlaps.append({
                "a_strategy": prev["strategy"],
                "a_entry":    prev["entry_time"],
                "a_exit":     prev["exit_time"],
                "b_strategy": curr["strategy"],
                "b_entry":    curr["entry_time"],
            })
    if overlaps:
        print(f"\nWARNING: {len(overlaps)} time-overlapping trades remain (wallet serialisation will skip them):")
        for o in overlaps[:10]:
            print(f"  {o['a_strategy']} {o['a_entry'].date()} → {o['a_exit'].date()} "
                  f"overlaps with {o['b_strategy']} {o['b_entry'].date()}")

    # ── Sequential single-wallet simulation ──
    balance      = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    portfolio_rows: list[dict] = []
    skipped_overlaps: list[dict] = []
    active_exit  = pd.Timestamp.min.tz_localize("UTC")

    for row in arbitrated.to_dict(orient="records"):
        entry_time = row["entry_time"]
        exit_time  = row["exit_time"]

        if entry_time < active_exit:
            skipped_overlaps.append({
                "strategy":   row["strategy"],
                "entry_time": entry_time,
                "exit_time":  exit_time,
                "reason":     "wallet_locked",
            })
            continue

        ret_pct = full_balance_return_pct(pd.Series(row))
        pnl_usd        = balance * ret_pct / 100.0
        balance_before = balance
        balance_after  = balance_before + pnl_usd
        peak_balance   = max(peak_balance, balance_after)
        drawdown_pct   = (peak_balance - balance_after) / peak_balance * 100.0 if peak_balance > 0 else 0.0

        portfolio_rows.append({
            "strategy":      row["strategy"],
            "side":          row["side"],
            "entry_time":    entry_time,
            "exit_time":     exit_time,
            "entry_price":   row["entry_price"],
            "exit_price":    row["exit_price"],
            "return_pct":    round(ret_pct, 4),
            "balance_before": balance_before,
            "pnl_usd":       round(pnl_usd, 4),
            "balance_after": round(balance_after, 4),
            "drawdown_pct":  round(drawdown_pct, 4),
        })
        balance    = balance_after
        active_exit = exit_time

    portfolio = pd.DataFrame(portfolio_rows)

    if portfolio.empty:
        print("No trades executed — check input paths.")
        return

    # ── Save CSVs ──
    portfolio.to_csv(OUTPUT_DIR / "portfolio_trades.csv", index=False)
    pd.DataFrame(skipped_overlaps).to_csv(OUTPUT_DIR / "skipped_overlaps.csv", index=False)
    skipped_dups.to_csv(OUTPUT_DIR / "skipped_duplicates.csv", index=False)
    summarize_strategy(portfolio).to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    summarize_group(portfolio, "side").to_csv(OUTPUT_DIR / "side_summary.csv", index=False)
    portfolio["year_month"] = portfolio["entry_time"].dt.strftime("%Y-%m")
    summarize_group(portfolio, "year_month").to_csv(OUTPUT_DIR / "month_summary.csv", index=False)

    # ── Stats ──
    n_total   = len(portfolio)
    win_rate  = float((portfolio["pnl_usd"] > 0).mean() * 100)
    gp = float(portfolio.loc[portfolio["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = float(-portfolio.loc[portfolio["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    max_dd    = float(portfolio["drawdown_pct"].max())
    end_bal   = float(portfolio["balance_after"].iloc[-1])
    net_pnl   = end_bal - STARTING_BALANCE
    win_start = portfolio["entry_time"].min()
    win_end   = portfolio["exit_time"].max()

    def _strat_row(name: str) -> dict:
        part = portfolio.loc[portfolio["strategy"] == name]
        if part.empty:
            return {"trades": 0, "wr": 0.0, "net_pnl": 0.0, "pf": 0.0}
        w = part.loc[part["pnl_usd"] > 0, "pnl_usd"]
        l = part.loc[part["pnl_usd"] < 0, "pnl_usd"]
        gp_ = float(w.sum())
        gl_ = float(-l.sum())
        return {
            "trades":  len(part),
            "wr":      round(float((part["pnl_usd"] > 0).mean() * 100), 2),
            "net_pnl": round(float(part["pnl_usd"].sum()), 2),
            "pf":      round(gp_ / gl_, 3) if gl_ > 0 else float("inf"),
        }

    s287 = _strat_row("short_t287")
    lt14 = _strat_row("long_t14")
    lf21 = _strat_row("long_f21")
    lbv3 = _strat_row("long_bull_v3")

    summary_lines = [
        "COMBINED ETH PORTFOLIO V2 — FULL BALANCE × 10x LEVERAGE",
        "=========================================================",
        "",
        "risk_model        = 100% of balance × LEVERAGE per trade",
        f"starting_balance  = {STARTING_BALANCE:.2f}",
        f"leverage          = {LEVERAGE:.0f}x",
        f"fee_round_trip    = {ROUND_TRIP_FEE_PCT:.2f}% of balance",
        f"window_start      = {win_start}",
        f"window_end        = {win_end}",
        "",
        "strategies (priority order):",
        "  1. short_t287   (SHORT)",
        "  2. long_t14     (LONG, bear flush-reversal)",
        "  3. long_f21     (LONG, bear flush-reversal)",
        "  4. long_bull_v3 (LONG, bull continuation)",
        "",
        "deduplication:",
        f"  exact_duplicates_removed     = {exact_dup_count}",
        f"  short_beat_long_conflicts    = {short_beat_long_count}",
        f"  same_minute_long_conflicts   = {same_min_long_conflicts}",
        f"  wallet_lock_skipped          = {len(skipped_overlaps)}",
        "",
        f"trade_count      = {n_total}",
        f"win_rate_pct     = {win_rate:.2f}",
        f"profit_factor    = {round(pf, 3) if pf != float('inf') else 'inf'}",
        f"max_drawdown_pct = {max_dd:.2f}",
        f"ending_balance   = {end_bal:.2f}",
        f"net_pnl_usd      = {net_pnl:.2f}",
        f"return_pct       = {net_pnl / STARTING_BALANCE * 100:.2f}%",
        "",
        "strategy_contribution:",
        f"  short_t287:   trades={s287['trades']}  WR={s287['wr']}%  net={s287['net_pnl']}  PF={s287['pf']}",
        f"  long_t14:     trades={lt14['trades']}  WR={lt14['wr']}%  net={lt14['net_pnl']}  PF={lt14['pf']}",
        f"  long_f21:     trades={lf21['trades']}  WR={lf21['wr']}%  net={lf21['net_pnl']}  PF={lf21['pf']}",
        f"  long_bull_v3: trades={lbv3['trades']}  WR={lbv3['wr']}%  net={lbv3['net_pnl']}  PF={lbv3['pf']}",
        "",
        "notes:",
        "  - Full-balance model means each trade uses the ENTIRE current balance × leverage",
        "  - P&L per trade can be large (10-50% swings per position are possible)",
        "  - This is aggressive — not a Kelly/risk-based approach",
        "  - Results do NOT reflect partial closes (blended entry/exit approximation)",
    ]
    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)

    # ── Plots ──
    print("\nGenerating plots...")
    _plot_equity_curve(portfolio, OUTPUT_DIR / "plot_equity_curve.png")
    _plot_drawdown(portfolio, OUTPUT_DIR / "plot_drawdown.png")
    _plot_monthly_pnl(portfolio, OUTPUT_DIR / "plot_monthly_pnl.png")
    _plot_per_strategy_equity(portfolio, OUTPUT_DIR / "plot_per_strategy_equity.png")
    _plot_trade_scatter(portfolio, OUTPUT_DIR / "plot_trade_scatter.png")
    print(f"Plots saved to {OUTPUT_DIR}")
    print(f"\nAll outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
