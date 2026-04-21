"""
4.5yr Portfolio Report — ETH All Strategies 2022-2026
Generates equity curve, drawdown, monthly P&L, per-strategy, trade scatter.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

BASE   = Path(__file__).resolve().parents[1]
OUT    = Path(__file__).resolve().parent / "portfolio_4yr_report"
OUT.mkdir(parents=True, exist_ok=True)

STARTING_BALANCE = 1000.0
LEVERAGE         = 20.0
FEE_RT_PCT       = 0.8   # 2 × 0.04% × 10x

# ── Load trades ───────────────────────────────────────────────────────────────
PATHS = {
    "short_t287":   BASE / "next_phase_edge_focus/results/binance_tier287_realistic_backtest_v1/trades.csv",
    "long_t14":     BASE / "next_phase_long_focus/results/edge_long_tier14_core_plus_satellite2_v1/trades.csv",
    "long_f21":     BASE / "next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/trades.csv",
    "long_bull_v3": BASE / "next_phase_long_focus/results/edge_long_bull_continuation_v3/trades.csv",
}

def load(path: Path, name: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["strategy"]   = name
    df["priority"]   = priority
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"]  = pd.to_numeric(df["exit_price"],  errors="coerce")
    return df

frames = [
    load(PATHS["short_t287"],   "short_t287",   1),
    load(PATHS["long_t14"],     "long_t14",     2),
    load(PATHS["long_f21"],     "long_f21",     3),
    load(PATHS["long_bull_v3"], "long_bull_v3", 4),
]
raw = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

# ── Deduplicate (same logic as combined backtest) ─────────────────────────────
raw = raw.drop_duplicates(subset=["entry_time","entry_price","exit_price"]).reset_index(drop=True)

# Wallet-lock arbitration: skip lower-priority if overlaps higher
kept, last_exit = [], pd.Timestamp.min.tz_localize("UTC")
for _, row in raw.iterrows():
    if row["entry_time"] < last_exit:
        continue
    kept.append(row)
    last_exit = row["exit_time"]
trades = pd.DataFrame(kept).reset_index(drop=True)

# ── Full-balance return ───────────────────────────────────────────────────────
def fb_ret(row):
    ep, xp = float(row["entry_price"]), float(row["exit_price"])
    side = str(row.get("side","")).strip().upper()
    if side in ("SHORT","SELL"):
        pr = (1.0 - xp/ep) * 100.0
    else:
        pr = (xp/ep - 1.0) * 100.0
    return pr * LEVERAGE - FEE_RT_PCT

trades["ret_pct"] = trades.apply(fb_ret, axis=1)

bal = STARTING_BALANCE
balances, drawdowns = [], []
peak = STARTING_BALANCE
for r in trades["ret_pct"]:
    bal = bal * (1 + r/100)
    peak = max(peak, bal)
    balances.append(bal)
    drawdowns.append((bal - peak) / peak * 100)

trades["balance"] = balances
trades["drawdown"] = drawdowns
trades["ym"] = trades["entry_time"].dt.strftime("%Y-%m")

print(f"Trades: {len(trades)} | WR: {(trades['ret_pct']>0).mean()*100:.1f}% | "
      f"End: ${bal:.0f} | MaxDD: {min(drawdowns):.2f}%")

# ── PLOT 1: Equity curve ──────────────────────────────────────────────────────
COLORS = {"short_t287":"#e74c3c","long_t14":"#2ecc71","long_f21":"#3498db","long_bull_v3":"#f39c12"}

fig, ax = plt.subplots(figsize=(14,5))
ax.plot(trades["entry_time"], trades["balance"], color="#2ecc71", lw=2)
ax.axhline(STARTING_BALANCE, color="gray", ls="--", lw=0.8)
for _, r in trades.iterrows():
    c = COLORS.get(r["strategy"],"gray")
    ax.scatter(r["entry_time"], r["balance"], color=c, s=40, zorder=3)
ax.fill_between(trades["entry_time"], STARTING_BALANCE, trades["balance"],
                where=trades["balance"]>=STARTING_BALANCE, alpha=0.12, color="#2ecc71")
ax.fill_between(trades["entry_time"], STARTING_BALANCE, trades["balance"],
                where=trades["balance"]<STARTING_BALANCE, alpha=0.15, color="#e74c3c")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
ax.set_title("Portfolio Equity Curve — 4.5yr (2022–2026)  |  $1000 start, 100% × 10x", fontsize=13)
ax.set_xlabel("Date"); ax.set_ylabel("Balance (USD)")
handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=8,label=n)
           for n,c in COLORS.items()]
ax.legend(handles=handles, loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUT/"plot_equity_curve.png", dpi=150)
plt.close()

# ── PLOT 2: Drawdown ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,4))
ax.fill_between(trades["entry_time"], trades["drawdown"], 0, color="#e74c3c", alpha=0.6)
ax.plot(trades["entry_time"], trades["drawdown"], color="#c0392b", lw=1)
ax.axhline(0, color="gray", ls="--", lw=0.7)
ax.set_title(f"Drawdown  |  Max: {min(drawdowns):.2f}%", fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))
ax.set_xlabel("Date"); ax.set_ylabel("Drawdown %"); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUT/"plot_drawdown.png", dpi=150)
plt.close()

# ── PLOT 3: Monthly P&L ───────────────────────────────────────────────────────
monthly = trades.groupby(["ym","strategy"])["ret_pct"].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(16,5))
bottom = np.zeros(len(monthly))
for strat in ["short_t287","long_t14","long_f21","long_bull_v3"]:
    if strat in monthly.columns:
        vals = monthly[strat].values
        bars = ax.bar(range(len(monthly)), vals, bottom=bottom,
                      label=strat, color=COLORS[strat], alpha=0.85, width=0.8)
        bottom += vals
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly.index, rotation=90, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax.set_title("Monthly Return % (balance) — stacked by strategy", fontsize=13)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig.savefig(OUT/"plot_monthly_pnl.png", dpi=150)
plt.close()

# ── PLOT 4: Per-strategy equity (2×2) ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14,9))
for ax, (strat, color) in zip(axes.flatten(), COLORS.items()):
    sub = trades[trades["strategy"]==strat].copy()
    if sub.empty:
        ax.text(0.5,0.5,"No trades",ha="center",va="center",transform=ax.transAxes)
        ax.set_title(strat); continue
    sub_bal = STARTING_BALANCE
    bals = []
    for r in sub["ret_pct"]:
        sub_bal = sub_bal * (1 + r/100)
        bals.append(sub_bal)
    n = len(sub)
    wr = (sub["ret_pct"]>0).mean()*100
    gp = sub.loc[sub["ret_pct"]>0,"ret_pct"].sum()
    gl = -sub.loc[sub["ret_pct"]<0,"ret_pct"].sum()
    pf = gp/gl if gl>0 else float("inf")
    ax.plot(sub["entry_time"], bals, color=color, lw=2)
    ax.axhline(STARTING_BALANCE, color="gray", ls="--", lw=0.8)
    ax.fill_between(sub["entry_time"], STARTING_BALANCE, bals,
                    where=[b>=STARTING_BALANCE for b in bals], alpha=0.15, color=color)
    ax.set_title(f"{strat}  n={n} WR={wr:.0f}% PF={pf:.2f}", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.grid(alpha=0.25)
plt.suptitle("Per-Strategy Equity (standalone compounding)", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUT/"plot_per_strategy.png", dpi=150, bbox_inches="tight")
plt.close()

# ── PLOT 5: Win/Loss scatter ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14,5))
for strat, color in COLORS.items():
    sub = trades[trades["strategy"]==strat]
    wins   = sub[sub["ret_pct"]>0]
    losses = sub[sub["ret_pct"]<0]
    ax.scatter(wins["entry_time"],   wins["ret_pct"],   color=color, marker="^", s=60, alpha=0.8)
    ax.scatter(losses["entry_time"], losses["ret_pct"], color=color, marker="v", s=60, alpha=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax.set_title("Trade Scatter — Return % per trade (^ win  v loss)", fontsize=13)
handles = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markersize=8,label=n)
           for n,c in COLORS.items()]
ax.legend(handles=handles, fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(OUT/"plot_trade_scatter.png", dpi=150)
plt.close()

# ── Save CSV ──────────────────────────────────────────────────────────────────
trades.to_csv(OUT/"portfolio_trades.csv", index=False)

strat_summary = trades.groupby("strategy").apply(lambda g: pd.Series({
    "trades": len(g),
    "wr_pct": round((g["ret_pct"]>0).mean()*100,1),
    "net_ret_pct": round(g["ret_pct"].sum(),2),
    "pf": round(g.loc[g["ret_pct"]>0,"ret_pct"].sum() / (-g.loc[g["ret_pct"]<0,"ret_pct"].sum())
               if (g["ret_pct"]<0).any() else float("inf"), 3),
})).reset_index()
strat_summary.to_csv(OUT/"strategy_summary.csv", index=False)

print(f"\nAll files saved to: {OUT}")
print("\nStrategy summary:")
print(strat_summary.to_string(index=False))
print(f"\nPortfolio: {len(trades)} trades | WR={( trades['ret_pct']>0).mean()*100:.1f}% | "
      f"MaxDD={min(drawdowns):.2f}% | End=${bal:.0f}")
