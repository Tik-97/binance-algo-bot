from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Standalone sandbox backtest. This file is intentionally separate from the
# research pipeline and does not touch main.py / analysis.py / summary_report.txt.
STARTING_BALANCE = 400.0
LEVERAGE = 10.0
USE_STAGE = "A"
ONE_TRADE_AT_A_TIME = True
FEE_RATE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.025
LIQUIDATION_BUFFER_PCT = 0.50
ENTRY_HOUR_START_UTC = 0
ENTRY_HOUR_END_UTC = 24
MAX_HOLDING_MINUTES = 720

RISK_MODES = {
    "full_margin": 1.00,
    "half_margin": 0.50,
    "quarter_margin": 0.25,
    "tenth_margin": 0.10,
}

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_cluster1_short_model_signals.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests/step_managed_short_proxy")


@dataclass(frozen=True)
class StepManagerConfig:
    starting_balance: float = STARTING_BALANCE
    leverage: float = LEVERAGE
    stage: str = USE_STAGE
    one_trade_at_a_time: bool = ONE_TRADE_AT_A_TIME
    fee_rate_per_side_pct: float = FEE_RATE_PER_SIDE_PCT
    slippage_per_side_pct: float = SLIPPAGE_PER_SIDE_PCT
    liquidation_buffer_pct: float = LIQUIDATION_BUFFER_PCT
    entry_hour_start_utc: int = ENTRY_HOUR_START_UTC
    entry_hour_end_utc: int = ENTRY_HOUR_END_UTC
    initial_sl_pct: float = 1.5
    initial_tp_pct: float = 1.0
    first_step_trigger_pct: float = 0.5
    second_step_trigger_pct: float = 1.0
    post_step_tp_pct: float = 1.5
    second_step_locked_profit_pct: float = 0.5
    max_holding_minutes: int = MAX_HOLDING_MINUTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone stepped-management sandbox backtest for ETH short proxy signals.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS_PATH), help="Path to proxy signal source CSV")
    parser.add_argument("--candles", type=str, default=str(DEFAULT_CANDLES_PATH), help="Path to cached ETHUSDT 1m candles")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Standalone output directory")
    parser.add_argument("--stage", type=str, default=USE_STAGE, help="Stage selection if the source supports it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    signals_path = (base_dir / args.signals).resolve()
    candles_path = (base_dir / args.candles).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = StepManagerConfig(stage=args.stage.upper().strip())
    signals, source_note = load_signal_source(signals_path, config.stage)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No ETH short proxy signals found in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No cached ETHUSDT 1m candles found in {candles_path}")

    trades_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict] = []

    for risk_mode, margin_fraction in RISK_MODES.items():
        trades_df, equity_df, summary = run_backtest(
            signals=signals,
            candles=candles,
            config=config,
            risk_mode=risk_mode,
            margin_fraction=margin_fraction,
        )
        trades_frames.append(trades_df)
        equity_frames.append(equity_df)
        comparison_rows.append(summary)

    trades = pd.concat(trades_frames, ignore_index=True, sort=False) if trades_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True, sort=False) if equity_frames else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows).sort_values("margin_fraction", ascending=False).reset_index(drop=True)

    trades.to_csv(output_dir / "step_manager_trades.csv", index=False)
    equity.to_csv(output_dir / "step_manager_equity_curve.csv", index=False)
    comparison.to_csv(output_dir / "step_manager_comparison.csv", index=False)
    (output_dir / "step_manager_summary.txt").write_text(
        build_summary_text(comparison, source_note, signals_path, candles_path, config),
        encoding="utf-8",
    )
    (output_dir / "instructions.txt").write_text(build_instructions_text(config), encoding="utf-8")

    create_plots(equity, output_dir)

    print(f"Standalone step-managed sandbox backtest completed. Outputs saved to: {output_dir}")
    return 0


def load_signal_source(path: Path, stage_letter: str) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")

    if "side" in frame.columns:
        frame = frame.loc[frame["side"].astype(str).str.lower() == "short"].copy()
    if "setup_name" in frame.columns:
        frame = frame.loc[frame["setup_name"].astype(str) == "eth_short_proxy"].copy()

    source_note = ""
    stage_id = f"stage_{stage_letter.lower()}"
    if "stage_id" in frame.columns:
        frame = frame.loc[frame["stage_id"].astype(str) == stage_id].copy()
        source_note = f"Source used: `{path.name}` filtered to short `eth_short_proxy` rows and `{stage_id}`."
    else:
        source_note = f"Source used: current practical Cluster 1 short model signals from `{path.name}`."

    keep_columns = [column for column in ["signal_id", "entry_time", "entry_price", "side", "context_text", "trigger_text"] if column in frame.columns]
    filtered = frame.loc[frame["entry_time"].notna() & frame["entry_price"].notna(), keep_columns].copy()
    if "signal_id" not in filtered.columns:
        filtered["signal_id"] = [f"signal_{idx+1}" for idx in range(len(filtered))]
    filtered["entry_hour_utc"] = filtered["entry_time"].dt.hour
    filtered = filtered.sort_values(["signal_id", "entry_time"]).drop_duplicates(subset=["signal_id"], keep="first")
    filtered = filtered.sort_values("entry_time").reset_index(drop=True)
    return filtered, source_note


def load_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("open_time").reset_index(drop=True)


def entry_hour_allowed(signal_time: pd.Timestamp, config: StepManagerConfig) -> bool:
    hour = int(signal_time.hour)
    return config.entry_hour_start_utc <= hour < config.entry_hour_end_utc


def run_backtest(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: StepManagerConfig,
    risk_mode: str,
    margin_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    skipped_time_filter = 0
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    trade_rows: list[dict] = []
    equity_rows: list[dict] = []

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.entry_time)
        if not entry_hour_allowed(signal_time, config):
            skipped_time_filter += 1
            continue
        if config.one_trade_at_a_time and signal_time < last_exit_time:
            ignored_overlaps += 1
            continue
        if balance <= 0:
            break

        trade = simulate_step_managed_trade(
            signal=signal,
            candles=candles,
            balance_before=balance,
            config=config,
            risk_mode=risk_mode,
            margin_fraction=margin_fraction,
            trade_index=len(trade_rows) + 1,
        )
        if trade is None:
            continue

        balance = float(trade["balance_after"])
        peak_balance = max(peak_balance, balance)
        drawdown_pct = 0.0 if peak_balance <= 0 else (peak_balance - balance) / peak_balance * 100.0
        trade["drawdown_pct"] = drawdown_pct
        trade_rows.append(trade)
        equity_rows.append(
            {
                "risk_mode": risk_mode,
                "trade_index": trade["trade_index"],
                "timestamp": trade["exit_time"],
                "balance_after": balance,
                "peak_balance": peak_balance,
                "drawdown_pct": drawdown_pct,
                "exit_reason": trade["exit_reason"],
                "state_reached": trade["state_reached"],
            }
        )
        last_exit_time = pd.Timestamp(trade["exit_time"])

    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)
    summary = summarize_run(
        trades_df=trades_df,
        config=config,
        risk_mode=risk_mode,
        margin_fraction=margin_fraction,
        ignored_overlaps=ignored_overlaps,
        skipped_time_filter=skipped_time_filter,
    )
    return trades_df, equity_df, summary


def simulate_step_managed_trade(
    signal,
    candles: pd.DataFrame,
    balance_before: float,
    config: StepManagerConfig,
    risk_mode: str,
    margin_fraction: float,
    trade_index: int,
) -> dict | None:
    signal_time = pd.Timestamp(signal.entry_time)
    future = candles.loc[candles["open_time"] > signal_time].copy()
    if future.empty:
        return None

    max_hold_cutoff = signal_time + pd.Timedelta(minutes=config.max_holding_minutes)
    future_window = future.loc[future["open_time"] <= max_hold_cutoff].copy()
    if future_window.empty:
        return None

    margin_used = balance_before * margin_fraction
    notional_used = margin_used * config.leverage

    entry_price = float(signal.entry_price) * (1.0 - config.slippage_per_side_pct / 100.0)
    liquidation_move_pct = max((100.0 / config.leverage) - config.liquidation_buffer_pct, 0.01)
    liquidation_price = entry_price * (1.0 + liquidation_move_pct / 100.0)

    state_reached = "initial"
    active_tp_price = entry_price * (1.0 - config.initial_tp_pct / 100.0)
    active_stop_price = entry_price * (1.0 + config.initial_sl_pct / 100.0)
    first_step_trigger_price = entry_price * (1.0 - config.first_step_trigger_pct / 100.0)
    second_step_trigger_price = entry_price * (1.0 - config.second_step_trigger_pct / 100.0)
    post_step_tp_price = entry_price * (1.0 - config.post_step_tp_pct / 100.0)
    break_even_stop_price = entry_price
    locked_profit_stop_price = entry_price * (1.0 - config.second_step_locked_profit_pct / 100.0)

    exit_reason = "time_stop"
    market_exit_price = float(future_window.iloc[-1]["close"])
    exit_time = pd.Timestamp(future_window.iloc[-1]["close_time"])
    liquidation_flag = False

    for candle in future_window.itertuples(index=False):
        high = float(candle.high)
        low = float(candle.low)

        # Conservative intrabar ordering:
        # liquidation / stop first, then step advancement, then take profit.
        if high >= liquidation_price:
            exit_reason = "liquidation"
            market_exit_price = liquidation_price
            exit_time = pd.Timestamp(candle.close_time)
            liquidation_flag = True
            break

        if high >= active_stop_price:
            exit_reason = "stop_loss" if state_reached == "initial" else "managed_stop"
            market_exit_price = active_stop_price
            exit_time = pd.Timestamp(candle.close_time)
            break

        if state_reached == "initial" and low <= first_step_trigger_price:
            state_reached = "step1_breakeven"
            active_stop_price = break_even_stop_price
            active_tp_price = post_step_tp_price

        if state_reached == "step1_breakeven" and low <= second_step_trigger_price:
            state_reached = "step2_locked"
            active_stop_price = locked_profit_stop_price
            active_tp_price = post_step_tp_price

        if low <= active_tp_price:
            exit_reason = "take_profit"
            market_exit_price = active_tp_price
            exit_time = pd.Timestamp(candle.close_time)
            break

    exit_price = market_exit_price * (1.0 + config.slippage_per_side_pct / 100.0)
    gross_underlying_return_pct = (entry_price - exit_price) / entry_price * 100.0
    fee_cost_usd = notional_used * (2.0 * config.fee_rate_per_side_pct) / 100.0

    if liquidation_flag:
        pnl_usd = -margin_used
        balance_after = max(balance_before + pnl_usd, 0.0)
    else:
        pnl_usd = notional_used * gross_underlying_return_pct / 100.0 - fee_cost_usd
        balance_after = max(balance_before + pnl_usd, 0.0)

    net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan

    return {
        "risk_mode": risk_mode,
        "trade_index": trade_index,
        "signal_id": signal.signal_id,
        "signal_time": signal_time,
        "signal_hour_utc": signal_time.hour,
        "entry_time": signal_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "state_reached": state_reached,
        "balance_before": balance_before,
        "margin_fraction": margin_fraction,
        "margin_used": margin_used,
        "notional_used": notional_used,
        "leverage": config.leverage,
        "gross_underlying_return_pct": gross_underlying_return_pct,
        "net_account_return_pct": net_account_return_pct,
        "pnl_usd": pnl_usd,
        "balance_after": balance_after,
        "drawdown_pct": np.nan,
        "liquidation_flag": liquidation_flag,
        "initial_sl_pct": config.initial_sl_pct,
        "initial_tp_pct": config.initial_tp_pct,
        "first_step_trigger_pct": config.first_step_trigger_pct,
        "second_step_trigger_pct": config.second_step_trigger_pct,
        "post_step_tp_pct": config.post_step_tp_pct,
        "second_step_locked_profit_pct": config.second_step_locked_profit_pct,
    }


def summarize_run(
    trades_df: pd.DataFrame,
    config: StepManagerConfig,
    risk_mode: str,
    margin_fraction: float,
    ignored_overlaps: int,
    skipped_time_filter: int,
) -> dict:
    ending_balance = float(trades_df["balance_after"].iloc[-1]) if not trades_df.empty else config.starting_balance
    max_drawdown_pct = float(pd.to_numeric(trades_df.get("drawdown_pct"), errors="coerce").max()) if not trades_df.empty else 0.0

    pnl = pd.to_numeric(trades_df.get("pnl_usd"), errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) > 0 else np.nan

    return {
        "risk_mode": risk_mode,
        "margin_fraction": margin_fraction,
        "starting_balance": config.starting_balance,
        "ending_balance": ending_balance,
        "total_return_pct": ((ending_balance / config.starting_balance) - 1.0) * 100.0 if config.starting_balance > 0 else np.nan,
        "trade_count": int(len(trades_df)),
        "win_count": int((pnl > 0).sum()) if not trades_df.empty else 0,
        "loss_count": int((pnl < 0).sum()) if not trades_df.empty else 0,
        "win_rate": float((pnl > 0).mean()) if not trades_df.empty else np.nan,
        "average_win_usd": float(wins.mean()) if not wins.empty else np.nan,
        "average_loss_usd": float(losses.mean()) if not losses.empty else np.nan,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "best_trade_usd": float(pnl.max()) if not pnl.empty else np.nan,
        "worst_trade_usd": float(pnl.min()) if not pnl.empty else np.nan,
        "ignored_overlapping_signals": ignored_overlaps,
        "skipped_by_time_filter": skipped_time_filter,
        "survived": bool(ending_balance > 0.0),
        "liquidation_occurred": bool(not trades_df.empty and trades_df["liquidation_flag"].any()),
    }


def build_summary_text(
    comparison: pd.DataFrame,
    source_note: str,
    signals_path: Path,
    candles_path: Path,
    config: StepManagerConfig,
) -> str:
    lines = [
        "STEP-MANAGED SANDBOX BACKTEST",
        "============================",
        "",
        "This is a standalone entertainment / exploratory wallet simulation.",
        "It is NOT part of the main reverse-analysis research conclusions.",
        "It is NOT proof that the proxy is safely tradable.",
        "It uses an approximate liquidation model, not exact exchange liquidation math.",
        "",
        source_note,
        f"Signals file used: {signals_path}",
        f"Candles file used: {candles_path}",
        "",
        "Backtest settings:",
        f"- Starting balance: ${config.starting_balance:.2f}",
        f"- Leverage: {config.leverage:.1f}x",
        f"- One trade at a time: {config.one_trade_at_a_time}",
        f"- Entry hours filter: {config.entry_hour_start_utc:02d}:00 to {config.entry_hour_end_utc:02d}:00 UTC",
        f"- Max holding time: {config.max_holding_minutes} minutes",
        f"- Initial SL: {config.initial_sl_pct:.2f}%",
        f"- Initial TP: {config.initial_tp_pct:.2f}%",
        f"- First step trigger: +{config.first_step_trigger_pct:.2f}% favorable move -> SL to entry, TP to {config.post_step_tp_pct:.2f}%",
        f"- Second step trigger: +{config.second_step_trigger_pct:.2f}% favorable move -> SL to +{config.second_step_locked_profit_pct:.2f}%, TP stays {config.post_step_tp_pct:.2f}%",
        f"- Fee per side: {config.fee_rate_per_side_pct:.3f}%",
        f"- Slippage per side: {config.slippage_per_side_pct:.3f}%",
        f"- Liquidation buffer: {config.liquidation_buffer_pct:.2f}%",
        "- Intrabar ordering is conservative: liquidation / stop first, then step advancement, then take profit.",
        "",
        "Comparison by risk mode:",
        comparison.to_string(index=False),
        "",
    ]

    for row in comparison.itertuples(index=False):
        lines.append(
            f"- {row.risk_mode}: ending balance ${row.ending_balance:.2f}, total return {row.total_return_pct:.2f}%, "
            f"trades {row.trade_count}, win rate {fmt_pct_fraction(row.win_rate)}, max drawdown {row.max_drawdown_pct:.2f}%, "
            f"ignored overlaps={row.ignored_overlapping_signals}, skipped by time filter={row.skipped_by_time_filter}, "
            f"survived={row.survived}, liquidation_occurred={row.liquidation_occurred}"
        )
    return "\n".join(lines).strip() + "\n"


def build_instructions_text(config: StepManagerConfig) -> str:
    return f"""MANUAL BACKTEST INSTRUCTIONS
============================

This folder contains a standalone sandbox backtest. It is separate from the main research pipeline.

Main script:
- `sandbox_wallet_eth_short_proxy_step_manager.py`

Main files produced here:
- `step_manager_trades.csv`
- `step_manager_equity_curve.csv`
- `step_manager_comparison.csv`
- `step_manager_summary.txt`
- `plots/step_manager_wallet_balance.png`

Where to change the rules manually:
1. Open `sandbox_wallet_eth_short_proxy_step_manager.py`.
2. Edit the `StepManagerConfig` dataclass near the top of the file.

Most important parameters:
- `entry_hour_start_utc` / `entry_hour_end_utc`
  Current: {config.entry_hour_start_utc} to {config.entry_hour_end_utc}
- `initial_sl_pct`
  Current: {config.initial_sl_pct}
- `initial_tp_pct`
  Current: {config.initial_tp_pct}
- `first_step_trigger_pct`
  Current: {config.first_step_trigger_pct}
- `second_step_trigger_pct`
  Current: {config.second_step_trigger_pct}
- `post_step_tp_pct`
  Current: {config.post_step_tp_pct}
- `second_step_locked_profit_pct`
  Current: {config.second_step_locked_profit_pct}
- `max_holding_minutes`
  Current: {config.max_holding_minutes}
- `fee_rate_per_side_pct`
- `slippage_per_side_pct`
- `liquidation_buffer_pct`

Where to change wallet aggressiveness:
- Edit the `RISK_MODES` dictionary near the top of the file.
- Example:
  `\"full_margin\": 1.00`
  `\"half_margin\": 0.50`
  `\"quarter_margin\": 0.25`
  `\"tenth_margin\": 0.10`

How the step-management works:
- Trade starts with initial SL / TP.
- After the first favorable move trigger, SL moves to entry and TP extends.
- After the second favorable move trigger, SL locks profit and TP stays at the extended target.

Where to change the signal source:
- Edit `DEFAULT_SIGNALS_PATH`
- Or run:
  `python sandbox_wallet_eth_short_proxy_step_manager.py --signals "path_to_csv"`

Where to change candle source:
- Edit `DEFAULT_CANDLES_PATH`
- Or run:
  `python sandbox_wallet_eth_short_proxy_step_manager.py --candles "path_to_eth_1m.csv"`

How to run manually:
- From the project root:
  `python sandbox_wallet_eth_short_proxy_step_manager.py`

Important note:
- This is a sandbox wallet simulation only.
- It is not part of the main reverse-analysis conclusions.
- It uses approximate liquidation logic and conservative intrabar assumptions.
"""


def fmt_pct_fraction(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric) * 100.0:.2f}%"


def create_plots(equity_df: pd.DataFrame, output_dir: Path) -> None:
    if equity_df.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for risk_mode, risk_df in equity_df.groupby("risk_mode", sort=False):
        ordered = risk_df.sort_values("trade_index")
        ax.plot(ordered["trade_index"], ordered["balance_after"], marker="o", linewidth=1.5, label=risk_mode)
    ax.set_title("Step-managed sandbox wallet balance")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Balance ($)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "step_manager_wallet_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
