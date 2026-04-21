"""
Stress Backtest — Realistic Slippage Scaling
=============================================
As balance grows, notional grows (full_balance x 10x).
Larger orders = more market impact slippage + lower limit fill rate.

Slippage model (per side, on top of base 0.025%):
  notional < $10k  : +0%     (base only)
  notional $10-50k : +0.01% per $10k above $10k
  notional $50k+   : +0.05% per $50k above $50k (accelerates)

Scenarios:
  Base     — original backtest (0.025% flat)
  Moderate — x2 scaling factor
  Severe   — x4 scaling factor
  Extreme  — x8 scaling factor (worst case)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

LEVERAGE        = 10.0
STARTING        = 1000.0
BASE_SLIP_PCT   = 0.025   # already in backtest per side
OUT = Path(__file__).resolve().parent / "stress_backtest"
OUT.mkdir(parents=True, exist_ok=True)

trades = pd.read_csv(Path(__file__).resolve().parent / "portfolio_4yr_report" / "portfolio_trades.csv")
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
trades = trades.sort_values("entry_time").reset_index(drop=True)

def fb_ret(row: pd.Series) -> float:
    ep, xp = float(row["entry_price"]), float(row["exit_price"])
    side = str(row.get("side", "")).strip().upper()
    pr = (1.0 - xp/ep)*100.0 if side in ("SHORT","SELL") else (xp/ep - 1.0)*100.0
    return pr * LEVERAGE - 0.8  # slippage already in entry/exit prices

trades["_ret"] = trades.apply(fb_ret, axis=1)


def extra_slip_pct(notional: float, scale: float) -> float:
    """Extra slippage per side beyond base, scaled by scenario factor."""
    if notional <= 10_000:
        extra = 0.0
    elif notional <= 50_000:
        extra = (notional - 10_000) / 10_000 * 0.01
    else:
        extra = 0.04 + (notional - 50_000) / 50_000 * 0.05
    return extra * scale


def run_scenario(scale: float) -> tuple[list, list, float]:
    bal = STARTING
    peak = STARTING
    balances, dds = [], []
    for _, row in trades.iterrows():
        notional = bal * LEVERAGE
        extra = extra_slip_pct(notional, scale)
        # Extra cost both sides × leverage = balance impact
        extra_cost_pct = 2 * extra * LEVERAGE
        ret = float(row["_ret"]) - extra_cost_pct
        bal = bal * (1 + ret / 100)
        peak = max(peak, bal)
        balances.append(bal)
        dds.append((bal - peak) / peak * 100)
    return balances, dds, bal


SCENARIOS = {
    "Base (0.025%)":     0.0,
    "Moderate (x2)":     2.0,
    "Severe (x4)":       4.0,
    "Extreme (x8)":      8.0,
}

COLORS = {
    "Base (0.025%)":  "#2ecc71",
    "Moderate (x2)":  "#3498db",
    "Severe (x4)":    "#f39c12",
    "Extreme (x8)":   "#e74c3c",
}

results = {}
print(f"{'Scenario':<20} {'End $':>10} {'MaxDD':>8} {'WR':>7}")
print("-" * 50)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

for label, scale in SCENARIOS.items():
    bals, dds, end_bal = run_scenario(scale)
    results[label] = {"bals": bals, "dds": dds, "end": end_bal, "maxdd": min(dds)}

    # Compute per-trade notional and adjusted rets for WR
    adj_rets = []
    bal = STARTING
    for _, row in trades.iterrows():
        notional = bal * LEVERAGE
        extra = extra_slip_pct(notional, scale)
        adj = float(row["_ret"]) - 2 * extra * LEVERAGE
        adj_rets.append(adj)
        bal = bal * (1 + adj / 100)
    wr = sum(1 for r in adj_rets if r > 0) / len(adj_rets) * 100

    print(f"{label:<20} ${end_bal:>9,.0f} {min(dds):>7.2f}% {wr:>6.1f}%")

    ax1.plot(trades["entry_time"], bals, color=COLORS[label], lw=2, label=label)
    ax2.fill_between(trades["entry_time"], dds, 0, alpha=0.3, color=COLORS[label])
    ax2.plot(trades["entry_time"], dds, color=COLORS[label], lw=1, label=label)

# Also show notional size over time
notional_sizes = []
bal = STARTING
for _, row in trades.iterrows():
    notional_sizes.append(bal * LEVERAGE)
    bal = bal * (1 + float(row["_ret"]) / 100)

print(f"\nNotional size range: ${min(notional_sizes):,.0f} - ${max(notional_sizes):,.0f}")
print(f"Max extra slip (Moderate) at peak: {extra_slip_pct(max(notional_sizes), 2.0):.4f}% per side")
print(f"Max extra slip (Severe) at peak:   {extra_slip_pct(max(notional_sizes), 4.0):.4f}% per side")
print(f"Max extra slip (Extreme) at peak:  {extra_slip_pct(max(notional_sizes), 8.0):.4f}% per side")

ax1.axhline(STARTING, color="gray", ls="--", lw=0.8)
ax1.set_title("Stress Backtest — Equity Curve with Slippage Scaling  |  $1000 start, 10x", fontsize=13)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(fontsize=9); ax1.grid(alpha=0.25)

ax2.axhline(0, color="black", lw=0.8)
ax2.set_title("Drawdown by Scenario", fontsize=12)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
ax2.set_xlabel("Date")

plt.tight_layout()
fig.savefig(OUT / "stress_equity_drawdown.png", dpi=150)
plt.close()

# ── Bar chart: end balance per scenario ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
labels = list(SCENARIOS.keys())
ends = [results[l]["end"] for l in labels]
bars = ax.bar(labels, ends, color=[COLORS[l] for l in labels], alpha=0.85, width=0.5)
for bar, val in zip(bars, ends):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"${val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(STARTING, color="gray", ls="--", lw=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_title("4.5yr End Balance — Stress Scenarios", fontsize=13)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig.savefig(OUT / "stress_end_balance.png", dpi=150)
plt.close()

print(f"\nPlots saved to: {OUT}")
