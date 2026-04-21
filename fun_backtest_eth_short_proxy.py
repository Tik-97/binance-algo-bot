from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Sandbox defaults. This script is intentionally standalone and is not part of
# the main reverse-analysis research pipeline.
STARTING_BALANCE = 400.0
LEVERAGE = 10.0
FULL_BALANCE_MARGIN = True
ONE_TRADE_AT_A_TIME = True
USE_STAGE = "A"
LIQUIDATION_BUFFER_PCT = 0.50

# Reused from the representative assumptions in eth_short_proxy_validation.py:
# normal_fee = 8 bps round-trip, moderate_slippage = 5 bps round-trip.
FEE_RATE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.025

SCENARIOS = [
    {"scenario_name": "tp0p5_sl0p5_12h", "tp_pct_underlying": 0.5, "sl_pct_underlying": 0.5, "max_holding_minutes": 720},
    {"scenario_name": "tp0p7_sl0p5_12h", "tp_pct_underlying": 0.7, "sl_pct_underlying": 0.5, "max_holding_minutes": 720},
    {"scenario_name": "tp1p0_sl0p7_24h", "tp_pct_underlying": 1.0, "sl_pct_underlying": 0.7, "max_holding_minutes": 1440},
]

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_proxy_short_results.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests")


@dataclass(frozen=True)
class SandboxConfig:
    starting_balance: float = STARTING_BALANCE
    leverage: float = LEVERAGE
    full_balance_margin: bool = FULL_BALANCE_MARGIN
    one_trade_at_a_time: bool = ONE_TRADE_AT_A_TIME
    use_stage: str = USE_STAGE
    liquidation_buffer_pct: float = LIQUIDATION_BUFFER_PCT
    fee_rate_per_side_pct: float = FEE_RATE_PER_SIDE_PCT
    slippage_per_side_pct: float = SLIPPAGE_PER_SIDE_PCT


def fmt_pct(value: float | int | object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric) * 100.0:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone sandbox wallet simulation for ETH short proxy signals.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS_PATH), help="Path to eth_proxy_short_results.csv")
    parser.add_argument("--candles", type=str, default=str(DEFAULT_CANDLES_PATH), help="Path to cached ETHUSDT_1m.csv")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Standalone sandbox output folder")
    parser.add_argument("--stage", type=str, default=USE_STAGE, help="Stage selection: A, B, or C")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    signals_path = (base_dir / args.signals).resolve()
    candles_path = (base_dir / args.candles).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = SandboxConfig(use_stage=args.stage.upper().strip())
    signals = load_proxy_signals(signals_path, config.use_stage)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No ETH short proxy signals found for stage `{config.use_stage}` in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No cached ETHUSDT 1m candles found in {candles_path}")

    all_trades = []
    all_equity = []
    scenario_rows = []
    summary_blocks = [build_summary_header(signals_path, candles_path, output_dir, config)]

    for scenario in SCENARIOS:
        trades, equity, summary = run_sandbox_scenario(signals, candles, config, scenario)
        all_trades.append(trades)
        all_equity.append(equity)
        scenario_rows.append(summary_to_row(summary))
        summary_blocks.append(build_scenario_summary_text(summary, config, scenario))

    trades_df = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True, sort=False) if all_equity else pd.DataFrame()
    comparison_df = pd.DataFrame(scenario_rows)

    trades_df.to_csv(output_dir / "fun_backtest_trades.csv", index=False)
    equity_df.to_csv(output_dir / "fun_backtest_equity_curve.csv", index=False)
    comparison_df.to_csv(output_dir / "fun_backtest_scenario_comparison.csv", index=False)
    (output_dir / "fun_backtest_summary.txt").write_text("\n\n".join(summary_blocks).strip() + "\n", encoding="utf-8")

    create_plots(equity_df, output_dir)

    print(f"Sandbox fun backtest completed. Outputs saved to: {output_dir}")
    return 0


def load_proxy_signals(path: Path, stage_letter: str) -> pd.DataFrame:
    stage_id = f"stage_{stage_letter.lower()}"
    frame = pd.read_csv(path)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")

    filtered = frame.loc[
        (frame["setup_name"] == "eth_short_proxy")
        & (frame["side"].str.lower() == "short")
        & (frame["stage_id"] == stage_id)
        & frame["entry_time"].notna()
        & frame["entry_price"].notna(),
        ["signal_id", "entry_time", "entry_price", "stage_id", "setup_name", "reference_cluster_id", "archetype_label"],
    ].copy()

    filtered = filtered.sort_values(["signal_id", "entry_time"]).drop_duplicates(subset=["signal_id"], keep="first")
    return filtered.sort_values("entry_time").reset_index(drop=True)


def load_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("open_time").reset_index(drop=True)


def run_sandbox_scenario(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: SandboxConfig,
    scenario: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    open_until = pd.Timestamp.min.tz_localize("UTC")
    trade_rows = []
    equity_rows = []
    liquidated = False
    liquidation_trade_index = np.nan
    liquidation_time = pd.NaT

    for idx, signal in enumerate(signals.itertuples(index=False), start=1):
        signal_time = pd.Timestamp(signal.entry_time)
        if config.one_trade_at_a_time and signal_time < open_until:
            ignored_overlaps += 1
            continue
        if balance <= 0:
            break

        trade = simulate_short_trade(
            signal,
            candles,
            balance,
            config,
            scenario,
            trade_index=len(trade_rows) + 1,
        )
        if trade is None:
            continue

        trade_rows.append(trade)
        balance = float(trade["balance_after"])
        peak_balance = max(peak_balance, balance)
        drawdown_pct = 0.0 if peak_balance <= 0 else (peak_balance - balance) / peak_balance * 100.0
        trade["drawdown_pct"] = drawdown_pct
        equity_rows.append(
            {
                "scenario_name": scenario["scenario_name"],
                "trade_index": trade["trade_index"],
                "timestamp": trade["exit_time"],
                "balance_after": balance,
                "peak_balance": peak_balance,
                "drawdown_pct": drawdown_pct,
                "exit_reason": trade["exit_reason"],
                "liquidation_flag": trade["liquidation_flag"],
            }
        )
        open_until = pd.Timestamp(trade["exit_time"])

        if bool(trade["liquidation_flag"]) or balance <= 0:
            liquidated = True
            liquidation_trade_index = trade["trade_index"]
            liquidation_time = pd.Timestamp(trade["exit_time"])
            break

    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty and "drawdown_pct" not in trades_df.columns:
        trades_df["drawdown_pct"] = np.nan
    equity_df = pd.DataFrame(equity_rows)

    summary = summarize_scenario(
        trades_df=trades_df,
        ignored_overlaps=ignored_overlaps,
        liquidated=liquidated,
        liquidation_trade_index=liquidation_trade_index,
        liquidation_time=liquidation_time,
        config=config,
        scenario=scenario,
    )
    return trades_df, equity_df, summary


def simulate_short_trade(signal, candles: pd.DataFrame, balance_before: float, config: SandboxConfig, scenario: dict, trade_index: int) -> dict | None:
    margin_used = balance_before if config.full_balance_margin else balance_before
    notional = margin_used * config.leverage
    signal_time = pd.Timestamp(signal.entry_time)
    entry_fill_price = float(signal.entry_price) * (1.0 - config.slippage_per_side_pct / 100.0)
    tp_price = entry_fill_price * (1.0 - scenario["tp_pct_underlying"] / 100.0)
    sl_price = entry_fill_price * (1.0 + scenario["sl_pct_underlying"] / 100.0)
    liquidation_move_pct = max((100.0 / config.leverage) - config.liquidation_buffer_pct, 0.01)
    liquidation_price = entry_fill_price * (1.0 + liquidation_move_pct / 100.0)

    future = candles.loc[candles["open_time"] > signal_time].copy()
    if future.empty:
        return None

    max_hold_cutoff = signal_time + pd.Timedelta(minutes=int(scenario["max_holding_minutes"]))
    future_window = future.loc[future["open_time"] <= max_hold_cutoff].copy()
    if future_window.empty:
        return None

    exit_reason = "time_stop"
    exit_time = pd.Timestamp(future_window.iloc[-1]["close_time"])
    market_exit_price = float(future_window.iloc[-1]["close"])
    liquidation_flag = False

    for candle in future_window.itertuples(index=False):
        high = float(candle.high)
        low = float(candle.low)
        if high >= liquidation_price:
            exit_reason = "liquidation"
            exit_time = pd.Timestamp(candle.close_time)
            market_exit_price = liquidation_price
            liquidation_flag = True
            break
        if high >= sl_price and low <= tp_price:
            exit_reason = "stop_loss_ambiguous_worst_case"
            exit_time = pd.Timestamp(candle.close_time)
            market_exit_price = sl_price
            break
        if high >= sl_price:
            exit_reason = "stop_loss"
            exit_time = pd.Timestamp(candle.close_time)
            market_exit_price = sl_price
            break
        if low <= tp_price:
            exit_reason = "take_profit"
            exit_time = pd.Timestamp(candle.close_time)
            market_exit_price = tp_price
            break

    exit_fill_price = market_exit_price * (1.0 + config.slippage_per_side_pct / 100.0)
    gross_underlying_return_pct = (entry_fill_price - exit_fill_price) / entry_fill_price * 100.0
    fee_cost_usd = notional * (2.0 * config.fee_rate_per_side_pct) / 100.0

    if liquidation_flag:
        pnl_usd = -balance_before
        net_account_return_pct = -100.0
        balance_after = 0.0
    else:
        pnl_usd = notional * gross_underlying_return_pct / 100.0 - fee_cost_usd
        balance_after = max(balance_before + pnl_usd, 0.0)
        net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan

    return {
        "scenario_name": scenario["scenario_name"],
        "trade_index": trade_index,
        "signal_id": signal.signal_id,
        "signal_time": signal_time,
        "entry_time": signal_time,
        "entry_price": entry_fill_price,
        "exit_time": exit_time,
        "exit_price": exit_fill_price,
        "exit_reason": exit_reason,
        "balance_before": balance_before,
        "margin_used": margin_used,
        "notional_used": notional,
        "leverage": config.leverage,
        "gross_underlying_return_pct": gross_underlying_return_pct,
        "net_account_return_pct": net_account_return_pct,
        "pnl_usd": pnl_usd,
        "balance_after": balance_after,
        "drawdown_pct": np.nan,
        "liquidation_flag": liquidation_flag,
    }


def summarize_scenario(
    trades_df: pd.DataFrame,
    ignored_overlaps: int,
    liquidated: bool,
    liquidation_trade_index: float,
    liquidation_time: pd.Timestamp,
    config: SandboxConfig,
    scenario: dict,
) -> dict:
    if trades_df.empty:
        return {
            "scenario_name": scenario["scenario_name"],
            "starting_balance": config.starting_balance,
            "ending_balance": config.starting_balance,
            "total_return_pct": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": np.nan,
            "average_win_usd": np.nan,
            "average_loss_usd": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_pct": 0.0,
            "best_trade_usd": np.nan,
            "worst_trade_usd": np.nan,
            "ignored_overlapping_signals": ignored_overlaps,
            "account_survived": True,
            "liquidation_occurred": False,
            "liquidation_trade_index": np.nan,
            "liquidation_time": pd.NaT,
        }

    winners = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losers = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]
    ending_balance = float(trades_df.iloc[-1]["balance_after"])
    max_drawdown = float(pd.to_numeric(trades_df["drawdown_pct"], errors="coerce").max())

    return {
        "scenario_name": scenario["scenario_name"],
        "starting_balance": config.starting_balance,
        "ending_balance": ending_balance,
        "total_return_pct": ((ending_balance / config.starting_balance) - 1.0) * 100.0,
        "trade_count": int(len(trades_df)),
        "win_count": int((trades_df["pnl_usd"] > 0).sum()),
        "loss_count": int((trades_df["pnl_usd"] <= 0).sum()),
        "win_rate": (trades_df["pnl_usd"] > 0).mean(),
        "average_win_usd": winners.mean() if not winners.empty else np.nan,
        "average_loss_usd": losers.mean() if not losers.empty else np.nan,
        "profit_factor": winners.sum() / abs(losers.sum()) if not losers.empty and abs(losers.sum()) > 0 else np.nan,
        "max_drawdown_pct": max_drawdown,
        "best_trade_usd": float(trades_df["pnl_usd"].max()),
        "worst_trade_usd": float(trades_df["pnl_usd"].min()),
        "ignored_overlapping_signals": ignored_overlaps,
        "account_survived": not liquidated and ending_balance > 0,
        "liquidation_occurred": liquidated,
        "liquidation_trade_index": liquidation_trade_index,
        "liquidation_time": liquidation_time,
    }


def summary_to_row(summary: dict) -> dict:
    return {
        "scenario_name": summary["scenario_name"],
        "ending_balance": summary["ending_balance"],
        "total_return_pct": summary["total_return_pct"],
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
        "max_drawdown_pct": summary["max_drawdown_pct"],
        "liquidation_occurred": summary["liquidation_occurred"],
    }


def build_summary_header(signals_path: Path, candles_path: Path, output_dir: Path, config: SandboxConfig) -> str:
    return f"""STANDALONE FUN BACKTEST - ETH SHORT PROXY WALLET SIMULATION
=====================================================
This is a standalone entertainment / exploratory wallet simulation.
It is NOT part of the main reverse-analysis research conclusions.
It is NOT proof that the ETH short proxy is safely tradable with full-balance 10x leverage.
It uses an approximate liquidation model, not exact Binance liquidation math.

Signal source used:
- Existing file: {signals_path}
- Selection: unique `eth_short_proxy` signals only, deduplicated by `signal_id`
- Stage: `stage_{config.use_stage.lower()}`
- Side: `Short`

Market data source used:
- Cached local candles only: {candles_path}
- No new market data downloads

Standalone output folder:
- {output_dir}
""".strip()


def build_scenario_summary_text(summary: dict, config: SandboxConfig, scenario: dict) -> str:
    liquidation_line = (
        f"Yes, on trade {int(summary['liquidation_trade_index'])} at {summary['liquidation_time']}."
        if summary["liquidation_occurred"] and pd.notna(summary["liquidation_time"])
        else "No."
    )
    return f"""Scenario: {summary['scenario_name']}
----------------------------------------
Starting balance: ${summary['starting_balance']:.2f}
Ending balance: ${summary['ending_balance']:.2f}
Total return: {summary['total_return_pct']:.2f}%
Trade count: {summary['trade_count']}
Win count: {summary['win_count']}
Loss count: {summary['loss_count']}
Win rate: {fmt_pct(summary['win_rate'])}
Average win: {summary['average_win_usd']:.2f}
Average loss: {summary['average_loss_usd']:.2f}
Profit factor: {summary['profit_factor']:.2f}
Max drawdown: {summary['max_drawdown_pct']:.2f}%
Best trade: ${summary['best_trade_usd']:.2f}
Worst trade: ${summary['worst_trade_usd']:.2f}
Ignored overlapping signals: {summary['ignored_overlapping_signals']}
Account survived: {'Yes' if summary['account_survived'] else 'No'}
Liquidation occurred: {liquidation_line}

Assumptions:
- leverage = {config.leverage:.1f}x
- full_balance_margin = {config.full_balance_margin}
- one_trade_at_a_time = {config.one_trade_at_a_time}
- TP = {scenario['tp_pct_underlying']:.2f}% underlying move
- SL = {scenario['sl_pct_underlying']:.2f}% underlying move
- max_hold = {int(scenario['max_holding_minutes'])} minutes
- fee_per_side = {config.fee_rate_per_side_pct:.3f}%
- slippage_per_side = {config.slippage_per_side_pct:.3f}%
- liquidation approximation = adverse move of (100/leverage - buffer)%, with buffer {config.liquidation_buffer_pct:.2f}%
- ambiguous candle ordering = adverse-first for safety (liquidation > SL > TP)
""".strip()


def create_plots(equity_df: pd.DataFrame, output_dir: Path) -> None:
    if equity_df.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    for scenario_name, group in equity_df.groupby("scenario_name"):
        ordered = group.sort_values("trade_index")
        ax1.plot(ordered["trade_index"], ordered["balance_after"], label=scenario_name, linewidth=2)
    ax1.set_title("Fun backtest equity curve")
    ax1.set_xlabel("Trade index")
    ax1.set_ylabel("Balance (USD)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.savefig(plots_dir / "fun_backtest_equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    for scenario_name, group in equity_df.groupby("scenario_name"):
        ordered = group.sort_values("trade_index")
        ax2.plot(ordered["trade_index"], ordered["drawdown_pct"], label=scenario_name, linewidth=2)
    ax2.set_title("Fun backtest drawdown")
    ax2.set_xlabel("Trade index")
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.savefig(plots_dir / "fun_backtest_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    raise SystemExit(main())
