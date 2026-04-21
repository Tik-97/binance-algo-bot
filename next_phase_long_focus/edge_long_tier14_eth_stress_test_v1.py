from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wickMick_long as base
from next_phase_long_focus import edge_long_tier14_core_plus_satellite2_v1 as strategy
from next_phase_long_focus import edge_long_tier1_bear_highvol_baseline_v1 as tier1


OUTPUT_DIR = ROOT / "next_phase_long_focus" / "results" / "edge_long_tier14_eth_stress_test_v1"
PLOTS_DIR = OUTPUT_DIR / "plots"


@dataclass(frozen=True)
class StressScenario:
    name: str
    title: str
    fee_per_side_pct: float
    limit_touch_buffer_pct: float
    limit_fill_slippage_pct: float
    market_entry_slippage_pct: float
    exit_slippage_pct: float


SCENARIOS = [
    StressScenario(
        name="baseline_realistic",
        title="Baseline realistic",
        fee_per_side_pct=0.04,
        limit_touch_buffer_pct=0.000,
        limit_fill_slippage_pct=0.000,
        market_entry_slippage_pct=0.025,
        exit_slippage_pct=0.025,
    ),
    StressScenario(
        name="eth_adverse_light",
        title="ETH adverse light",
        fee_per_side_pct=0.04,
        limit_touch_buffer_pct=0.005,
        limit_fill_slippage_pct=0.000,
        market_entry_slippage_pct=0.040,
        exit_slippage_pct=0.040,
    ),
    StressScenario(
        name="eth_adverse_medium",
        title="ETH adverse medium",
        fee_per_side_pct=0.04,
        limit_touch_buffer_pct=0.010,
        limit_fill_slippage_pct=0.005,
        market_entry_slippage_pct=0.060,
        exit_slippage_pct=0.060,
    ),
    StressScenario(
        name="eth_adverse_hard",
        title="ETH adverse hard",
        fee_per_side_pct=0.04,
        limit_touch_buffer_pct=0.015,
        limit_fill_slippage_pct=0.010,
        market_entry_slippage_pct=0.080,
        exit_slippage_pct=0.080,
    ),
]


def make_long_entry_simulator(
    touch_buffer_pct: float,
    limit_fill_slippage_pct: float,
    market_entry_slippage_pct: float,
):
    def simulate_long_entry_stress(
        df: pd.DataFrame,
        trigger_idx: int,
        confirm_idx: int,
        trigger_bar: pd.Series,
        confirm_bar: pd.Series,
        stop_price_initial: float,
    ) -> tuple[int | None, pd.Timestamp | None, float | None, float | None, str | None]:
        order_placed_idx = confirm_idx + 1
        if order_placed_idx >= len(df):
            return None, None, None, None, None

        order_placed_time = df.iloc[order_placed_idx]["open_time"]
        strong_signal = base.is_strong_long_signal(trigger_bar, confirm_bar)

        if strong_signal and base.ENABLE_AGGRESSIVE_MARKET_ON_STRONG_SIGNAL:
            bar = df.iloc[order_placed_idx]
            raw_entry = float(bar["open"])
            entry_price = raw_entry * (1.0 + market_entry_slippage_pct / 100.0)
            return order_placed_idx, order_placed_time, raw_entry, entry_price, "aggressive_market"

        if base.ENABLE_SHALLOW_PULLBACK_LIMIT:
            confirm_close = float(confirm_bar["close"])
            trigger_low = float(trigger_bar["low"])

            order_price = confirm_close * (1.0 - base.SHALLOW_PULLBACK_LIMIT_PCT_BELOW_CONFIRM_CLOSE / 100.0)

            if order_price < confirm_close * (1.0 - base.MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT / 100.0):
                return None, order_placed_time, order_price, None, "shallow_pullback_limit"

            if base.REJECT_IF_LIMIT_BELOW_STOP and order_price <= stop_price_initial:
                return None, order_placed_time, order_price, None, "shallow_pullback_limit"

            last_check_idx = min(order_placed_idx + base.ENTRY_WAIT_BARS - 1, len(df) - 1)
            fill_start_idx = order_placed_idx if base.LIMIT_CAN_FILL_SAME_BAR else order_placed_idx + 1
            if fill_start_idx > last_check_idx:
                return None, order_placed_time, order_price, None, "shallow_pullback_limit"

            fill_threshold = order_price * (1.0 - touch_buffer_pct / 100.0)

            for j in range(fill_start_idx, last_check_idx + 1):
                bar = df.iloc[j]
                low = float(bar["low"])
                close = float(bar["close"])

                if base.CANCEL_IF_RECLAIM_TRIGGER_LOW and low <= trigger_low:
                    return None, order_placed_time, order_price, None, "shallow_pullback_limit"

                if (
                    base.CANCEL_IF_RECLAIM_CONFIRM_CLOSE_TOO_MUCH
                    and close < confirm_close * (1.0 - base.MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT / 100.0)
                ):
                    return None, order_placed_time, order_price, None, "shallow_pullback_limit"

                if low <= fill_threshold:
                    entry_price = order_price * (1.0 + limit_fill_slippage_pct / 100.0)
                    return j, order_placed_time, order_price, entry_price, "shallow_pullback_limit"

            return None, order_placed_time, order_price, None, "shallow_pullback_limit"

        return None, order_placed_time, None, None, None

    return simulate_long_entry_stress


def compute_max_drawdown_pct(trades_df: pd.DataFrame, starting_balance: float) -> float:
    if trades_df.empty:
        return 0.0
    equity = pd.Series([starting_balance] + trades_df["balance_after"].tolist(), dtype=float)
    running_peak = equity.cummax()
    drawdown = (equity / running_peak - 1.0) * 100.0
    return abs(float(drawdown.min()))


def build_monthly_summary(trades_df: pd.DataFrame, scenario_name: str, scenario_title: str) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["scenario", "scenario_title", "month", "net_pnl_usd", "trade_count"])
    frame = trades_df.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["month"] = frame["entry_time"].dt.strftime("%Y-%m")
    monthly = (
        frame.groupby("month", as_index=False)
        .agg(net_pnl_usd=("pnl_usd", "sum"), trade_count=("pnl_usd", "size"))
    )
    monthly["scenario"] = scenario_name
    monthly["scenario_title"] = scenario_title
    return monthly[["scenario", "scenario_title", "month", "net_pnl_usd", "trade_count"]]


def build_route_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["scenario", "scenario_title", "route", "trade_count", "net_pnl_usd"])
    grouped = (
        trades_df.groupby(["scenario", "scenario_title", "route"], as_index=False)
        .agg(trade_count=("pnl_usd", "size"), net_pnl_usd=("pnl_usd", "sum"))
    )
    return grouped


def run_tier14_backtest(featured: pd.DataFrame, bt_start: pd.Timestamp, bt_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    original_is_base_context = base.is_base_context
    original_find_trigger_index = base.find_trigger_index
    try:
        base.is_base_context = strategy.make_tier1_context_gate()
        base.find_trigger_index = strategy.make_route_aware_find_trigger()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_is_base_context
        base.find_trigger_index = original_find_trigger_index
    return trades_df, summary


def run_scenario(featured: pd.DataFrame, scenario: StressScenario) -> tuple[pd.DataFrame, dict]:
    original_values = {
        "fee_per_side_pct": base.FEE_PER_SIDE_PCT,
        "market_slippage_pct": base.MARKET_SLIPPAGE_PER_SIDE_PCT,
        "limit_fill_slippage_pct": base.LIMIT_FILL_SLIPPAGE_PCT,
        "simulate_long_entry": base.simulate_long_entry,
    }
    try:
        base.FEE_PER_SIDE_PCT = scenario.fee_per_side_pct
        base.MARKET_SLIPPAGE_PER_SIDE_PCT = scenario.exit_slippage_pct
        base.LIMIT_FILL_SLIPPAGE_PCT = scenario.limit_fill_slippage_pct
        base.simulate_long_entry = make_long_entry_simulator(
            touch_buffer_pct=scenario.limit_touch_buffer_pct,
            limit_fill_slippage_pct=scenario.limit_fill_slippage_pct,
            market_entry_slippage_pct=scenario.market_entry_slippage_pct,
        )
        trades_df, summary = run_tier14_backtest(
            featured,
            bt_start=featured["open_time"].min(),
            bt_end=featured["open_time"].max(),
        )
    finally:
        base.FEE_PER_SIDE_PCT = original_values["fee_per_side_pct"]
        base.MARKET_SLIPPAGE_PER_SIDE_PCT = original_values["market_slippage_pct"]
        base.LIMIT_FILL_SLIPPAGE_PCT = original_values["limit_fill_slippage_pct"]
        base.simulate_long_entry = original_values["simulate_long_entry"]

    trades_df = trades_df.copy()
    if not trades_df.empty:
        trades_df["route"] = trades_df.apply(strategy.classify_trade_route, axis=1)
    trades_df["scenario"] = scenario.name
    trades_df["scenario_title"] = scenario.title

    summary = dict(summary)
    summary["scenario"] = scenario.name
    summary["scenario_title"] = scenario.title
    summary["filled_trade_count"] = int(len(trades_df))
    summary["net_pnl_usd"] = float(summary["ending_balance"] - base.STARTING_BALANCE)
    summary["return_on_start_pct"] = float((summary["ending_balance"] / base.STARTING_BALANCE - 1.0) * 100.0)
    summary["equiv_2000_balance"] = float(summary["ending_balance"] * (2000.0 / base.STARTING_BALANCE))
    summary["fee_per_side_pct"] = scenario.fee_per_side_pct
    summary["limit_touch_buffer_pct"] = scenario.limit_touch_buffer_pct
    summary["limit_fill_slippage_pct"] = scenario.limit_fill_slippage_pct
    summary["market_entry_slippage_pct"] = scenario.market_entry_slippage_pct
    summary["exit_slippage_pct"] = scenario.exit_slippage_pct
    summary["max_drawdown_pct"] = compute_max_drawdown_pct(trades_df, base.STARTING_BALANCE)
    summary["best_trade_usd"] = float(trades_df["pnl_usd"].max()) if not trades_df.empty else math.nan
    summary["worst_trade_usd"] = float(trades_df["pnl_usd"].min()) if not trades_df.empty else math.nan
    return trades_df, summary


def plot_equity_curves(trades_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "baseline_realistic": "#1f77b4",
        "eth_adverse_light": "#2ca02c",
        "eth_adverse_medium": "#ff7f0e",
        "eth_adverse_hard": "#d62728",
    }
    for scenario, group in trades_df.groupby("scenario", sort=False):
        title = group["scenario_title"].iloc[0]
        balances = [base.STARTING_BALANCE] + group["balance_after"].tolist()
        x = list(range(len(balances)))
        ax.plot(x, balances, marker="o", linewidth=2.2, label=title, color=colors.get(scenario))
    ax.set_title("Tier 1.4 Long ETH Stress: Equity Curves", fontsize=16, weight="bold")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Balance (USD)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_curves.png", dpi=180)
    plt.close(fig)


def plot_dashboard(summary_df: pd.DataFrame) -> None:
    order = summary_df["scenario_title"].tolist()
    x = np.arange(len(order))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tier 1.4 Long ETH/Binance Stress Dashboard", fontsize=18, weight="bold")

    metrics = [
        ("ending_balance", "Ending Balance (USD)"),
        ("win_rate_pct", "Win Rate (%)"),
        ("filled_trade_count", "Trade Count"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
    ]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        vals = summary_df[col].tolist()
        bars = ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x, order, rotation=15)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            label = f"{val:.2f}" if isinstance(val, float) else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS_DIR / "stress_dashboard.png", dpi=180)
    plt.close(fig)


def plot_monthly_net_pnl(monthly_df: pd.DataFrame) -> None:
    if monthly_df.empty:
        return
    pivot = monthly_df.pivot(index="month", columns="scenario_title", values="net_pnl_usd").fillna(0.0)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Tier 1.4 Long ETH Stress: Monthly Net PnL", fontsize=16, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net PnL (USD)")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "monthly_net_pnl.png", dpi=180)
    plt.close(fig)


def plot_trade_pnl_boxplot(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        return
    order = [scenario.title for scenario in SCENARIOS]
    groups = []
    for scenario in SCENARIOS:
        part = trades_df.loc[trades_df["scenario"] == scenario.name, "pnl_usd"]
        groups.append(part.tolist())
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(groups, tick_labels=order, showfliers=True)
    ax.set_title("Tier 1.4 Long ETH Stress: Trade PnL Distribution", fontsize=16, weight="bold")
    ax.set_ylabel("PnL (USD)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trade_pnl_boxplot.png", dpi=180)
    plt.close(fig)


def write_summary(summary_df: pd.DataFrame) -> None:
    lines = [
        "EDGE LONG TIER1.4 ETH STRESS TEST V1",
        "====================================",
        "",
        "strategy = edge_long_tier14_core_plus_satellite2_v1.py",
        "market = ETHUSDT",
        "venue = Binance-style ETH execution assumptions",
        "",
        "stress philosophy:",
        "- fee kept fixed per side",
        "- pressure only through worse limit touch / fill and worse market/exit slippage",
        "- designed to be harsher than baseline but still realistic for ETH/Binance",
        "",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.extend(
            [
                f"{row['scenario_title']}:",
                f"- trade_count = {int(row['filled_trade_count'])}",
                f"- ending_balance = {row['ending_balance']:.2f}",
                f"- net_pnl_usd = {row['net_pnl_usd']:.4f}",
                f"- win_rate_pct = {row['win_rate_pct']:.2f}" if not pd.isna(row["win_rate_pct"]) else "- win_rate_pct = n/a",
                f"- max_drawdown_pct = {row['max_drawdown_pct']:.2f}",
                f"- market_entry_slippage_pct = {row['market_entry_slippage_pct']:.3f}",
                f"- exit_slippage_pct = {row['exit_slippage_pct']:.3f}",
                f"- limit_touch_buffer_pct = {row['limit_touch_buffer_pct']:.3f}",
                f"- limit_fill_slippage_pct = {row['limit_fill_slippage_pct']:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "generated_files:",
            f"- {OUTPUT_DIR / 'scenario_summary.csv'}",
            f"- {OUTPUT_DIR / 'monthly_summary.csv'}",
            f"- {OUTPUT_DIR / 'route_summary.csv'}",
            f"- {PLOTS_DIR / 'stress_dashboard.png'}",
            f"- {PLOTS_DIR / 'equity_curves.png'}",
            f"- {PLOTS_DIR / 'monthly_net_pnl.png'}",
            f"- {PLOTS_DIR / 'trade_pnl_boxplot.png'}",
        ]
    )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)

    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    monthly_frames: list[pd.DataFrame] = []

    for scenario in SCENARIOS:
        trades_df, summary = run_scenario(featured, scenario)
        scenario_dir = OUTPUT_DIR / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(scenario_dir / "trades.csv", index=False)
        all_trades.append(trades_df)
        summary_rows.append(summary)
        monthly_frames.append(build_monthly_summary(trades_df, scenario.name, scenario.title))

    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    route_df = build_route_summary(all_trades_df)

    summary_df.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    route_df.to_csv(OUTPUT_DIR / "route_summary.csv", index=False)
    all_trades_df.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)

    plot_equity_curves(all_trades_df)
    plot_dashboard(summary_df)
    plot_monthly_net_pnl(monthly_df)
    plot_trade_pnl_boxplot(all_trades_df)
    write_summary(summary_df)

    print(f"Saved long ETH stress package to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
