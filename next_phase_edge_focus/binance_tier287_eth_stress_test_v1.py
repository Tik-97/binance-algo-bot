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

from next_phase_edge_focus import binance_tier287_realistic_backtest_v1 as strategy


OUTPUT_DIR = strategy.prev.prev.base.BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier287_eth_stress_test_v1"
PLOTS_DIR = OUTPUT_DIR / "plots"


@dataclass(frozen=True)
class StressScenario:
    name: str
    title: str
    maker_fee_pct: float
    taker_fee_pct: float
    limit_touch_buffer_pct: float
    limit_fill_slippage_pct: float
    market_entry_slippage_pct: float
    exit_slippage_pct: float


SCENARIOS = [
    StressScenario(
        name="baseline_realistic",
        title="Baseline realistic",
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        limit_touch_buffer_pct=0.000,
        limit_fill_slippage_pct=0.000,
        market_entry_slippage_pct=0.025,
        exit_slippage_pct=0.025,
    ),
    StressScenario(
        name="eth_adverse_light",
        title="ETH adverse light",
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        limit_touch_buffer_pct=0.005,
        limit_fill_slippage_pct=0.000,
        market_entry_slippage_pct=0.040,
        exit_slippage_pct=0.040,
    ),
    StressScenario(
        name="eth_adverse_medium",
        title="ETH adverse medium",
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        limit_touch_buffer_pct=0.010,
        limit_fill_slippage_pct=0.005,
        market_entry_slippage_pct=0.060,
        exit_slippage_pct=0.060,
    ),
    StressScenario(
        name="eth_adverse_hard",
        title="ETH adverse hard",
        maker_fee_pct=0.02,
        taker_fee_pct=0.05,
        limit_touch_buffer_pct=0.015,
        limit_fill_slippage_pct=0.010,
        market_entry_slippage_pct=0.080,
        exit_slippage_pct=0.080,
    ),
]


def make_limit_entry_simulator(
    touch_buffer_pct: float,
    limit_fill_slippage_pct: float,
):
    def simulate_limit_retest_entry_stress(
        df: pd.DataFrame,
        anchor_idx: int,
        stop_price_initial: float,
    ) -> tuple[int | None, pd.Timestamp | None, float | None, float | None]:
        if anchor_idx is None or anchor_idx >= len(df):
            return None, None, None, None

        anchor_bar = df.iloc[anchor_idx]
        order_price = float(anchor_bar["close"]) * (1.0 + strategy.prev.prev.base.ENTRY_RETEST_ABOVE_CLOSE_PCT / 100.0)
        if order_price >= stop_price_initial:
            return None, anchor_bar["open_time"], order_price, None

        fill_threshold = order_price * (1.0 + touch_buffer_pct / 100.0)
        last_idx = min(anchor_idx + strategy.prev.prev.base.ENTRY_WAIT_BARS, len(df) - 1)
        for j in range(anchor_idx + 1, last_idx + 1):
            bar = df.iloc[j]
            if float(bar["high"]) >= fill_threshold:
                entry_price = order_price * (1.0 + limit_fill_slippage_pct / 100.0)
                return j, bar["open_time"], order_price, entry_price
        return None, anchor_bar["open_time"], order_price, None

    return simulate_limit_retest_entry_stress


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


def run_refined_backtest_for_scenario(
    execution_df: pd.DataFrame,
    deduped_candidates: pd.DataFrame,
    market_regime_lookup: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict]:
    original_fn = strategy.prev.prev.is_other_non_target_mild_strong_bear_expansion
    try:
        strategy.prev.prev.is_other_non_target_mild_strong_bear_expansion = (
            strategy.is_other_non_target_mild_strong_bear_expansion_refined2
        )
        trades_df, summary = strategy.prev.prev.run_backtest_binance_tier286(
            execution_df,
            deduped_candidates,
            market_regime_lookup,
        )
    finally:
        strategy.prev.prev.is_other_non_target_mild_strong_bear_expansion = original_fn
    return trades_df, summary


def run_scenario(
    execution_df: pd.DataFrame,
    deduped_candidates: pd.DataFrame,
    market_regime_lookup: dict[str, dict[str, object]],
    scenario: StressScenario,
) -> tuple[pd.DataFrame, dict]:
    base = strategy.prev.prev.base
    original_values = {
        "maker": base.BINANCE_MAKER_FEE_PCT,
        "taker": base.BINANCE_TAKER_FEE_PCT,
        "limit_fill_slippage": base.BINANCE_LIMIT_FILL_SLIPPAGE_PCT,
        "market_entry_slippage": base.BINANCE_MARKET_ENTRY_SLIPPAGE_PCT,
        "exit_slippage": base.BINANCE_EXIT_SLIPPAGE_PCT,
        "simulate_limit_fn": base.simulate_limit_retest_entry_binance,
    }
    try:
        base.BINANCE_MAKER_FEE_PCT = scenario.maker_fee_pct
        base.BINANCE_TAKER_FEE_PCT = scenario.taker_fee_pct
        base.BINANCE_LIMIT_FILL_SLIPPAGE_PCT = scenario.limit_fill_slippage_pct
        base.BINANCE_MARKET_ENTRY_SLIPPAGE_PCT = scenario.market_entry_slippage_pct
        base.BINANCE_EXIT_SLIPPAGE_PCT = scenario.exit_slippage_pct
        base.simulate_limit_retest_entry_binance = make_limit_entry_simulator(
            touch_buffer_pct=scenario.limit_touch_buffer_pct,
            limit_fill_slippage_pct=scenario.limit_fill_slippage_pct,
        )
        trades_df, summary = run_refined_backtest_for_scenario(execution_df, deduped_candidates, market_regime_lookup)
    finally:
        base.BINANCE_MAKER_FEE_PCT = original_values["maker"]
        base.BINANCE_TAKER_FEE_PCT = original_values["taker"]
        base.BINANCE_LIMIT_FILL_SLIPPAGE_PCT = original_values["limit_fill_slippage"]
        base.BINANCE_MARKET_ENTRY_SLIPPAGE_PCT = original_values["market_entry_slippage"]
        base.BINANCE_EXIT_SLIPPAGE_PCT = original_values["exit_slippage"]
        base.simulate_limit_retest_entry_binance = original_values["simulate_limit_fn"]

    trades_df = trades_df.copy()
    trades_df["scenario"] = scenario.name
    trades_df["scenario_title"] = scenario.title

    summary = dict(summary)
    summary["scenario"] = scenario.name
    summary["scenario_title"] = scenario.title
    summary["limit_touch_buffer_pct"] = scenario.limit_touch_buffer_pct
    summary["max_drawdown_pct"] = compute_max_drawdown_pct(trades_df, base.STARTING_BALANCE)
    summary["net_pnl_usd"] = float(summary["ending_balance"] - base.STARTING_BALANCE)
    summary["return_on_start_pct"] = float((summary["ending_balance"] / base.STARTING_BALANCE - 1.0) * 100.0)
    summary["equiv_2000_balance"] = float(summary["ending_balance"] * (2000.0 / base.STARTING_BALANCE))
    summary["maker_fee_pct"] = scenario.maker_fee_pct
    summary["taker_fee_pct"] = scenario.taker_fee_pct
    summary["market_entry_slippage_pct"] = scenario.market_entry_slippage_pct
    summary["exit_slippage_pct"] = scenario.exit_slippage_pct
    summary["limit_fill_slippage_pct"] = scenario.limit_fill_slippage_pct
    if not trades_df.empty:
        summary["best_trade_usd"] = float(trades_df["pnl_usd"].max())
        summary["worst_trade_usd"] = float(trades_df["pnl_usd"].min())
    else:
        summary["best_trade_usd"] = math.nan
        summary["worst_trade_usd"] = math.nan
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
        balances = [strategy.prev.prev.base.STARTING_BALANCE] + group["balance_after"].tolist()
        x = list(range(len(balances)))
        ax.plot(x, balances, marker="o", linewidth=2.2, label=title, color=colors.get(scenario))
    ax.set_title("Tier 2.87 ETH Stress: Equity Curves", fontsize=16, weight="bold")
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
    fig.suptitle("Tier 2.87 ETH/Binance Stress Dashboard", fontsize=18, weight="bold")

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
    months = pivot.index.tolist()
    titles = pivot.columns.tolist()
    x = np.arange(len(months))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(titles))
    for idx, title in enumerate(titles):
        ax.bar(x + offsets[idx], pivot[title].values, width=width, label=title, color=colors[idx % len(colors)], alpha=0.9)
    ax.set_title("Tier 2.87 ETH Stress: Monthly Net PnL", fontsize=16, weight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net PnL (USD)")
    ax.set_xticks(x, months, rotation=45, ha="right")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "monthly_net_pnl.png", dpi=180)
    plt.close(fig)


def plot_trade_pnl_boxplot(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        return
    titles = []
    values = []
    for _, group in trades_df.groupby("scenario", sort=False):
        titles.append(group["scenario_title"].iloc[0])
        values.append(group["pnl_usd"].tolist())
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(values, tick_labels=titles, patch_artist=True, boxprops={"facecolor": "#a6cee3", "alpha": 0.8})
    ax.set_title("Tier 2.87 ETH Stress: Trade PnL Distribution", fontsize=16, weight="bold")
    ax.set_ylabel("Trade PnL (USD)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trade_pnl_boxplot.png", dpi=180)
    plt.close(fig)


def write_summary(summary_df: pd.DataFrame, monthly_df: pd.DataFrame) -> None:
    best = summary_df.sort_values("ending_balance", ascending=False).iloc[0]
    worst = summary_df.sort_values("ending_balance", ascending=True).iloc[0]
    lines = [
        "BINANCE TIER2.87 ETH STRESS TEST V1",
        "===================================",
        "",
        "Goal",
        "----",
        "Test Tier 2.87 under ETH/Binance execution friction assumptions that are harsher than baseline but still plausible for a small account on ETHUSDT.",
        "",
        "Design choice",
        "-------------",
        "Fees are kept fixed at Binance baseline assumptions.",
        "Stress is applied mainly through moderate slippage and slightly worse limit-touch requirements.",
        "This is intentionally more realistic for ETHUSDT than the old extreme package.",
        "",
        "Quick verdict",
        "-------------",
        f"Best ending balance scenario: {best['scenario_title']} -> {best['ending_balance']:.2f}",
        f"Worst ending balance scenario: {worst['scenario_title']} -> {worst['ending_balance']:.2f}",
        "",
        "Scenario table",
        "--------------",
    ]
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                f"{row['scenario_title']}",
                f"  filled_trade_count = {int(row['filled_trade_count'])}",
                f"  ending_balance = {row['ending_balance']:.2f}",
                f"  net_pnl_usd = {row['net_pnl_usd']:.2f}",
                f"  return_on_start_pct = {row['return_on_start_pct']:.2f}",
                f"  equiv_2000_balance = {row['equiv_2000_balance']:.2f}",
                f"  win_rate_pct = {row['win_rate_pct']:.2f}",
                f"  profit_factor = {row['profit_factor']:.3f}" if pd.notna(row["profit_factor"]) else "  profit_factor = nan",
                f"  max_drawdown_pct = {row['max_drawdown_pct']:.2f}",
                f"  worst_trade_usd = {row['worst_trade_usd']:.2f}",
                f"  maker_fee_pct = {row['maker_fee_pct']:.3f}",
                f"  taker_fee_pct = {row['taker_fee_pct']:.3f}",
                f"  market_entry_slippage_pct = {row['market_entry_slippage_pct']:.3f}",
                f"  exit_slippage_pct = {row['exit_slippage_pct']:.3f}",
                f"  limit_touch_buffer_pct = {row['limit_touch_buffer_pct']:.3f}",
                "",
            ]
        )
    if not monthly_df.empty:
        lines.extend(["Monthly coverage", "----------------"])
        monthly_pivot = monthly_df.pivot(index="month", columns="scenario_title", values="net_pnl_usd").fillna(0.0)
        for month, row in monthly_pivot.iterrows():
            parts = [f"{title}={row[title]:.2f}" for title in monthly_pivot.columns]
            lines.append(f"{month}: " + ", ".join(parts))
    lines.extend(
        [
            "",
            "Plot files",
            "----------",
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

    base = strategy.prev.prev.base
    execution_candles = base.load_execution_candles(base.CANDLES_1M_PATH)
    execution_df = base.build_execution_features(execution_candles.copy())

    candidate_features = base.build_candidate_features(execution_candles.copy())
    raw_candidates = base.generate_candidates(candidate_features)
    deduped_candidates = base.deduplicate_candidates(raw_candidates)

    daily_regime = base.build_market_regime_frame_from_candles(execution_candles.copy())
    market_regime_lookup = {
        row["trade_date"]: {
            "trend_regime": row["trend_regime"],
            "vol_regime": row["vol_regime"],
            "ret_30d_pct": float(row["ret_30d_pct"]) if pd.notna(row["ret_30d_pct"]) else np.nan,
            "ret_60d_pct": float(row["ret_60d_pct"]) if pd.notna(row["ret_60d_pct"]) else np.nan,
        }
        for _, row in daily_regime.iterrows()
    }

    ref_df = base.load_reference_eth_shorts(base.JSON_TRADES_PATH)
    ref_start, ref_end = base.get_reference_window(ref_df)

    scenario_summaries = []
    all_trades = []
    all_periods = []
    all_monthly = []

    for scenario in SCENARIOS:
        trades_df, summary = run_scenario(execution_df, deduped_candidates, market_regime_lookup, scenario)
        period_df = base.build_period_summary(trades_df, ref_start, ref_end)
        monthly_df = build_monthly_summary(trades_df, scenario.name, scenario.title)

        scenario_summaries.append(summary)
        all_trades.append(trades_df)
        if not period_df.empty:
            period_df = period_df.copy()
            period_df["scenario"] = scenario.name
            period_df["scenario_title"] = scenario.title
            all_periods.append(period_df)
        if not monthly_df.empty:
            all_monthly.append(monthly_df)

        scenario_dir = OUTPUT_DIR / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(scenario_dir / "trades.csv", index=False)
        period_df.to_csv(scenario_dir / "period_summary.csv", index=False)

    summary_df = pd.DataFrame(scenario_summaries).sort_values("ending_balance", ascending=False).reset_index(drop=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    period_df = pd.concat(all_periods, ignore_index=True) if all_periods else pd.DataFrame()
    monthly_df = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()

    summary_df.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "scenario_trades_combined.csv", index=False)
    period_df.to_csv(OUTPUT_DIR / "scenario_period_summary.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    raw_candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    deduped_candidates.to_csv(OUTPUT_DIR / "deduped_candidates.csv", index=False)
    daily_regime.to_csv(OUTPUT_DIR / "binance_daily_regime.csv", index=False)

    if not trades_df.empty:
        plot_equity_curves(trades_df)
        plot_dashboard(summary_df)
        plot_monthly_net_pnl(monthly_df)
        plot_trade_pnl_boxplot(trades_df)
    write_summary(summary_df, monthly_df)

    print(f"Saved ETH/Binance stress package to: {OUTPUT_DIR}")
    print(summary_df[["scenario_title", "filled_trade_count", "ending_balance", "win_rate_pct", "max_drawdown_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
