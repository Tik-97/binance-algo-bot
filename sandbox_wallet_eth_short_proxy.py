from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Standalone sandbox defaults. This script is intentionally separate from the
# main reverse-analysis research pipeline.
STARTING_BALANCE = 400.0
LEVERAGE = 10.0
USE_STAGE = "A"
ONE_TRADE_AT_A_TIME = True
FEE_RATE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.025
LIQUIDATION_BUFFER_PCT = 0.50
ENTRY_HOUR_START_UTC = 13
ENTRY_HOUR_END_UTC = 17
BREAKEVEN_ENABLED = True
BREAKEVEN_TRIGGER_PCT = 0.20

RISK_MODES = {
    "full_margin": 1.00,
    "half_margin": 0.50,
    "quarter_margin": 0.25,
    "tenth_margin": 0.10,
}

SCENARIOS = [
    {"name": "tp0p5_sl0p5_12h", "tp_pct": 0.5, "sl_pct": 0.5, "max_holding_minutes": 720},
    {"name": "tp0p7_sl0p5_12h", "tp_pct": 0.7, "sl_pct": 0.5, "max_holding_minutes": 720},
    {"name": "tp1p0_sl0p7_24h", "tp_pct": 1.0, "sl_pct": 0.7, "max_holding_minutes": 1440},
    {"name": "tp0p3_sl1p0_12h", "tp_pct": 0.3, "sl_pct": 1.0, "max_holding_minutes": 720},
    {"name": "tp0p3_sl1p5_12h", "tp_pct": 0.3, "sl_pct": 1.5, "max_holding_minutes": 720},
]

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_cluster1_short_model_signals.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests/sandbox_wallet_multi_risk")


@dataclass(frozen=True)
class SandboxConfig:
    starting_balance: float = STARTING_BALANCE
    leverage: float = LEVERAGE
    stage: str = USE_STAGE
    one_trade_at_a_time: bool = ONE_TRADE_AT_A_TIME
    fee_rate_per_side_pct: float = FEE_RATE_PER_SIDE_PCT
    slippage_per_side_pct: float = SLIPPAGE_PER_SIDE_PCT
    liquidation_buffer_pct: float = LIQUIDATION_BUFFER_PCT
    entry_hour_start_utc: int = ENTRY_HOUR_START_UTC
    entry_hour_end_utc: int = ENTRY_HOUR_END_UTC
    breakeven_enabled: bool = BREAKEVEN_ENABLED
    breakeven_trigger_pct: float = BREAKEVEN_TRIGGER_PCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone multi-risk sandbox wallet simulation for ETH short proxy signals.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS_PATH), help="Path to proxy signal source CSV")
    parser.add_argument("--candles", type=str, default=str(DEFAULT_CANDLES_PATH), help="Path to cached ETHUSDT 1m candles")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Standalone sandbox output directory")
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

    config = SandboxConfig(stage=args.stage.upper().strip())
    signals, source_note = load_signal_source(signals_path, config.stage)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No ETH short proxy signals found in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No cached ETHUSDT 1m candles found in {candles_path}")

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    comparison_rows: list[dict] = []

    for scenario in SCENARIOS:
        for risk_mode, margin_fraction in RISK_MODES.items():
            trades_df, equity_df, summary_row = run_wallet_simulation(
                signals=signals,
                candles=candles,
                config=config,
                scenario=scenario,
                risk_mode=risk_mode,
                margin_fraction=margin_fraction,
            )
            all_trades.append(trades_df)
            all_equity.append(equity_df)
            comparison_rows.append(summary_row)

    trades = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    equity_curve = pd.concat(all_equity, ignore_index=True, sort=False) if all_equity else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows).sort_values(["scenario_name", "margin_fraction"], ascending=[True, False]).reset_index(drop=True)

    trades.to_csv(output_dir / "sandbox_wallet_trades.csv", index=False)
    equity_curve.to_csv(output_dir / "sandbox_wallet_equity_curve.csv", index=False)
    comparison.to_csv(output_dir / "sandbox_wallet_comparison.csv", index=False)
    (output_dir / "sandbox_wallet_summary.txt").write_text(
        build_summary_text(
            comparison=comparison,
            source_note=source_note,
            signals_path=signals_path,
            candles_path=candles_path,
            output_dir=output_dir,
            config=config,
        ),
        encoding="utf-8",
    )

    create_plots(equity_curve, output_dir)

    print(f"Standalone sandbox wallet simulation completed. Outputs saved to: {output_dir}")
    return 0


def load_signal_source(path: Path, stage_letter: str) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    frame.columns = [str(column) for column in frame.columns]

    if "entry_time" not in frame.columns or "entry_price" not in frame.columns:
        raise RuntimeError(f"Signal source {path} does not contain entry_time / entry_price columns.")

    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")

    source_note = ""
    if "side" in frame.columns:
        frame = frame.loc[frame["side"].astype(str).str.lower() == "short"].copy()
    if "setup_name" in frame.columns:
        frame = frame.loc[frame["setup_name"].astype(str) == "eth_short_proxy"].copy()

    stage_id = f"stage_{stage_letter.lower()}"
    if "stage_id" in frame.columns:
        frame = frame.loc[frame["stage_id"].astype(str) == stage_id].copy()
        source_note = (
            f"Source used: executed proxy-signal style rows from `{path.name}`, filtered to side=Short and stage `{stage_id}`."
        )
    else:
        source_note = (
            f"Source used: current practical Cluster 1 short model-signal set from `{path.name}`. "
            "This file is already a Stage A-style signal source, so no extra stage filter was applied."
        )

    keep_columns = [column for column in ["signal_id", "entry_time", "entry_price", "side", "setup_name", "context_text", "trigger_text"] if column in frame.columns]
    filtered = frame.loc[
        frame["entry_time"].notna() & frame["entry_price"].notna(),
        keep_columns,
    ].copy()
    filtered["entry_hour_utc"] = filtered["entry_time"].dt.hour

    if "signal_id" not in filtered.columns:
        filtered["signal_id"] = [f"signal_{idx+1}" for idx in range(len(filtered))]

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


def entry_hour_allowed(signal_time: pd.Timestamp, config: SandboxConfig) -> bool:
    hour = int(signal_time.hour)
    return config.entry_hour_start_utc <= hour < config.entry_hour_end_utc


def run_wallet_simulation(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: SandboxConfig,
    scenario: dict,
    risk_mode: str,
    margin_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    liquidation_occurred = False
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")
    skipped_time_filter = 0

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

        trade = simulate_trade(
            signal=signal,
            candles=candles,
            balance_before=balance,
            config=config,
            scenario=scenario,
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
                "scenario_name": scenario["name"],
                "trade_index": trade["trade_index"],
                "timestamp": trade["exit_time"],
                "balance_after": balance,
                "peak_balance": peak_balance,
                "drawdown_pct": drawdown_pct,
                "exit_reason": trade["exit_reason"],
                "liquidation_flag": trade["liquidation_flag"],
            }
        )
        last_exit_time = pd.Timestamp(trade["exit_time"])

        if bool(trade["liquidation_flag"]) and balance <= 0:
            liquidation_occurred = True
            break

    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)
    summary = summarize_run(
        trades_df=trades_df,
        equity_df=equity_df,
        config=config,
        scenario=scenario,
        risk_mode=risk_mode,
        margin_fraction=margin_fraction,
        ignored_overlaps=ignored_overlaps,
        skipped_time_filter=skipped_time_filter,
        liquidation_occurred=liquidation_occurred,
    )
    return trades_df, equity_df, summary


def simulate_trade(
    signal,
    candles: pd.DataFrame,
    balance_before: float,
    config: SandboxConfig,
    scenario: dict,
    risk_mode: str,
    margin_fraction: float,
    trade_index: int,
) -> dict | None:
    signal_time = pd.Timestamp(signal.entry_time)
    future = candles.loc[candles["open_time"] > signal_time].copy()
    if future.empty:
        return None

    max_hold_cutoff = signal_time + pd.Timedelta(minutes=int(scenario["max_holding_minutes"]))
    future_window = future.loc[future["open_time"] <= max_hold_cutoff].copy()
    if future_window.empty:
        return None

    margin_used = balance_before * margin_fraction
    notional_used = margin_used * config.leverage

    # Short entry: worse fill is slightly lower on entry and slightly higher on exit.
    entry_price = float(signal.entry_price) * (1.0 - config.slippage_per_side_pct / 100.0)
    tp_price = entry_price * (1.0 - scenario["tp_pct"] / 100.0)
    sl_price = entry_price * (1.0 + scenario["sl_pct"] / 100.0)
    liquidation_move_pct = max((100.0 / config.leverage) - config.liquidation_buffer_pct, 0.01)
    liquidation_price = entry_price * (1.0 + liquidation_move_pct / 100.0)
    breakeven_price = entry_price * (1.0 + config.slippage_per_side_pct / 100.0)
    breakeven_trigger_price = entry_price * (1.0 - config.breakeven_trigger_pct / 100.0)
    breakeven_armed = False

    exit_reason = "time_stop"
    market_exit_price = float(future_window.iloc[-1]["close"])
    exit_time = pd.Timestamp(future_window.iloc[-1]["close_time"])
    liquidation_flag = False

    for candle in future_window.itertuples(index=False):
        high = float(candle.high)
        low = float(candle.low)

        if config.breakeven_enabled and not breakeven_armed and low <= breakeven_trigger_price:
            breakeven_armed = True

        # Ambiguous intrabar ordering is handled adversarially for safety:
        # liquidation > stop-loss / breakeven > take-profit.
        if high >= liquidation_price:
            exit_reason = "liquidation"
            market_exit_price = liquidation_price
            exit_time = pd.Timestamp(candle.close_time)
            liquidation_flag = True
            break
        active_stop_price = breakeven_price if breakeven_armed else sl_price
        active_stop_reason = "breakeven_stop" if breakeven_armed else "stop_loss"
        if high >= active_stop_price and low <= tp_price:
            exit_reason = f"{active_stop_reason}_ambiguous_worst_case"
            market_exit_price = active_stop_price
            exit_time = pd.Timestamp(candle.close_time)
            break
        if high >= active_stop_price:
            exit_reason = active_stop_reason
            market_exit_price = active_stop_price
            exit_time = pd.Timestamp(candle.close_time)
            break
        if low <= tp_price:
            exit_reason = "take_profit"
            market_exit_price = tp_price
            exit_time = pd.Timestamp(candle.close_time)
            break

    exit_price = market_exit_price * (1.0 + config.slippage_per_side_pct / 100.0)
    gross_underlying_return_pct = (entry_price - exit_price) / entry_price * 100.0
    fee_cost_usd = notional_used * (2.0 * config.fee_rate_per_side_pct) / 100.0

    if liquidation_flag:
        pnl_usd = -margin_used
        balance_after = max(balance_before + pnl_usd, 0.0)
        net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan
    else:
        pnl_usd = notional_used * gross_underlying_return_pct / 100.0 - fee_cost_usd
        balance_after = max(balance_before + pnl_usd, 0.0)
        net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan

    return {
        "risk_mode": risk_mode,
        "scenario_name": scenario["name"],
        "trade_index": trade_index,
        "signal_id": signal.signal_id,
        "signal_time": signal_time,
        "signal_hour_utc": signal_time.hour,
        "entry_time": signal_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
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
        "breakeven_enabled": config.breakeven_enabled,
        "breakeven_trigger_pct": config.breakeven_trigger_pct,
        "breakeven_armed_flag": breakeven_armed,
    }


def summarize_run(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    config: SandboxConfig,
    scenario: dict,
    risk_mode: str,
    margin_fraction: float,
    ignored_overlaps: int,
    skipped_time_filter: int,
    liquidation_occurred: bool,
) -> dict:
    if trades_df.empty:
        ending_balance = config.starting_balance
        max_drawdown_pct = 0.0
    else:
        ending_balance = float(trades_df["balance_after"].iloc[-1])
        max_drawdown_pct = float(pd.to_numeric(trades_df["drawdown_pct"], errors="coerce").max())

    pnl = pd.to_numeric(trades_df.get("pnl_usd"), errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) > 0 else np.nan

    return {
        "risk_mode": risk_mode,
        "scenario_name": scenario["name"],
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
        "liquidation_occurred": bool(liquidation_occurred or bool(trades_df.get("liquidation_flag", pd.Series(dtype=bool)).any())),
        "liquidation_trade_index": int(trades_df.loc[trades_df["liquidation_flag"] == True, "trade_index"].iloc[0]) if not trades_df.empty and trades_df["liquidation_flag"].any() else np.nan,
        "liquidation_time": trades_df.loc[trades_df["liquidation_flag"] == True, "exit_time"].iloc[0] if not trades_df.empty and trades_df["liquidation_flag"].any() else pd.NaT,
    }


def build_summary_text(
    comparison: pd.DataFrame,
    source_note: str,
    signals_path: Path,
    candles_path: Path,
    output_dir: Path,
    config: SandboxConfig,
) -> str:
    lines = [
        "SANDBOX WALLET SIMULATION",
        "========================",
        "",
        "This is a standalone entertainment / exploratory wallet simulation.",
        "It is NOT part of the main reverse-analysis research conclusions.",
        "It is NOT proof that the proxy is safely tradable.",
        "It uses an approximate liquidation model, not exact exchange liquidation math.",
        "The purpose is to compare how wallet behavior changes under different aggressiveness assumptions.",
        "",
        f"{source_note}",
        f"Candles used: `{candles_path.name}`.",
        f"Stage request: `{config.stage}`.",
        f"Output folder: `{output_dir}`.",
        "",
        "Core assumptions:",
        f"- Starting balance: ${config.starting_balance:.2f}",
        f"- Leverage: {config.leverage:.1f}x",
        f"- One trade at a time: {config.one_trade_at_a_time}",
        f"- Fee per side: {config.fee_rate_per_side_pct:.3f}%",
        f"- Slippage per side: {config.slippage_per_side_pct:.3f}%",
        f"- Liquidation approximation buffer: {config.liquidation_buffer_pct:.2f}% underlying move below the simple 100/leverage threshold",
        f"- Entry hours filter: {config.entry_hour_start_utc:02d}:00 to {config.entry_hour_end_utc:02d}:00 UTC",
        f"- Breakeven protection enabled: {config.breakeven_enabled}",
        f"- Breakeven trigger: +{config.breakeven_trigger_pct:.2f}% favorable underlying move from entry",
        "- Ambiguous candles are resolved adverse-first for safety: liquidation > protective stop > take-profit.",
        "- Liquidation approximation assumes the allocated trade margin is lost; with smaller margin fractions the rest of the wallet survives.",
        "- After the breakeven trigger, the active protective stop is moved to a near-entry break-even level.",
        "",
        "Risk modes:",
    ]
    for name, fraction in RISK_MODES.items():
        lines.append(f"- {name}: {fraction:.0%} of wallet used as margin per trade")
    lines.append("")
    lines.append("Scenario comparison:")
    lines.append(comparison.to_string(index=False))
    lines.append("")

    for scenario_name, scenario_frame in comparison.groupby("scenario_name", sort=False):
        lines.append(f"Scenario `{scenario_name}`:")
        for row in scenario_frame.sort_values("margin_fraction", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.risk_mode}: ending balance ${row.ending_balance:.2f}, total return {row.total_return_pct:.2f}%, "
                f"trades {row.trade_count}, win rate {fmt_pct_fraction(row.win_rate)}, max drawdown {row.max_drawdown_pct:.2f}%, "
                f"survived={row.survived}, liquidation_occurred={row.liquidation_occurred}, "
                f"ignored overlaps={row.ignored_overlapping_signals}, skipped by time filter={row.skipped_by_time_filter}"
            )
        lines.append("")

    lines.append(f"Signals file used: {signals_path}")
    lines.append(f"Candles file used: {candles_path}")
    return "\n".join(lines).strip() + "\n"


def fmt_pct_fraction(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric) * 100.0:.2f}%"


def create_plots(equity_curve: pd.DataFrame, output_dir: Path) -> None:
    if equity_curve.empty:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_equity, axes_equity = plt.subplots(len(SCENARIOS), 1, figsize=(12, 4 * len(SCENARIOS)), sharex=False)
    fig_dd, axes_dd = plt.subplots(len(SCENARIOS), 1, figsize=(12, 4 * len(SCENARIOS)), sharex=False)

    if len(SCENARIOS) == 1:
        axes_equity = [axes_equity]
        axes_dd = [axes_dd]

    for ax_eq, ax_dd, scenario in zip(axes_equity, axes_dd, SCENARIOS):
        scenario_name = scenario["name"]
        plot_df = equity_curve.loc[equity_curve["scenario_name"] == scenario_name].copy()
        for risk_mode in RISK_MODES:
            risk_df = plot_df.loc[plot_df["risk_mode"] == risk_mode].sort_values("trade_index")
            if risk_df.empty:
                continue
            ax_eq.plot(risk_df["trade_index"], risk_df["balance_after"], marker="o", linewidth=1.5, label=risk_mode)
            ax_dd.plot(risk_df["trade_index"], risk_df["drawdown_pct"], marker="o", linewidth=1.5, label=risk_mode)
        ax_eq.set_title(f"Equity curves: {scenario_name}")
        ax_eq.set_xlabel("Trade index")
        ax_eq.set_ylabel("Balance ($)")
        ax_eq.grid(alpha=0.25)
        ax_eq.legend(frameon=False)

        ax_dd.set_title(f"Drawdowns: {scenario_name}")
        ax_dd.set_xlabel("Trade index")
        ax_dd.set_ylabel("Drawdown %")
        ax_dd.grid(alpha=0.25)
        ax_dd.legend(frameon=False)

    fig_equity.tight_layout()
    fig_dd.tight_layout()
    fig_equity.savefig(plots_dir / "sandbox_wallet_equity_curves.png", dpi=150, bbox_inches="tight")
    fig_dd.savefig(plots_dir / "sandbox_wallet_drawdowns.png", dpi=150, bbox_inches="tight")
    plt.close(fig_equity)
    plt.close(fig_dd)


if __name__ == "__main__":
    raise SystemExit(main())
