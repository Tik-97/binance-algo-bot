from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STARTING_BALANCE = 400.0
LEVERAGE = 10.0
USE_STAGE = "A"
ONE_TRADE_AT_A_TIME = True
FEE_RATE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.025
LIQUIDATION_BUFFER_PCT = 0.50
ENTRY_HOUR_START_UTC = 13
ENTRY_HOUR_END_UTC = 17
MAX_HOLDING_MINUTES = 720
MIN_SIGNAL_FILTER_WICK = 0.40

RISK_MODES = {
    "full_margin": 1.00,
    "half_margin": 0.50,
    "quarter_margin": 0.25,
    "tenth_margin": 0.10,
}

EXIT_MANAGERS = [
    {
        "manager_id": "baseline_step",
        "label": "Baseline step manager (TP1.0 / SL1.5)",
        "type": "baseline_step",
    },
    {
        "manager_id": "atr_ladder",
        "label": "ATR ladder (SL=TP=0.5 ATR, shift 0.25 ATR)",
        "type": "atr_ladder",
    },
    {
        "manager_id": "pct_ladder",
        "label": "0.5% ladder (SL=TP=0.5%, shift 0.5%)",
        "type": "pct_ladder",
    },
]

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_cluster1_short_model_signals.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests/ladder_manager_cluster1_short")


@dataclass(frozen=True)
class LadderConfig:
    starting_balance: float = STARTING_BALANCE
    leverage: float = LEVERAGE
    stage: str = USE_STAGE
    one_trade_at_a_time: bool = ONE_TRADE_AT_A_TIME
    fee_rate_per_side_pct: float = FEE_RATE_PER_SIDE_PCT
    slippage_per_side_pct: float = SLIPPAGE_PER_SIDE_PCT
    liquidation_buffer_pct: float = LIQUIDATION_BUFFER_PCT
    entry_hour_start_utc: int = ENTRY_HOUR_START_UTC
    entry_hour_end_utc: int = ENTRY_HOUR_END_UTC
    min_signal_filter_wick: float = MIN_SIGNAL_FILTER_WICK
    max_holding_minutes: int = MAX_HOLDING_MINUTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ladder-manager sandbox for Cluster 1 ETH short proxy.")
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

    config = LadderConfig(stage=args.stage.upper().strip())
    signals, source_note, filter_note = load_filtered_signals(signals_path, config.stage, config)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No filtered Cluster 1 short signals found in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No cached ETHUSDT 1m candles found in {candles_path}")

    trades_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict] = []

    for manager in EXIT_MANAGERS:
        for risk_mode, margin_fraction in RISK_MODES.items():
            trades_df, equity_df, summary = run_backtest(
                signals=signals,
                candles=candles,
                config=config,
                manager=manager,
                risk_mode=risk_mode,
                margin_fraction=margin_fraction,
            )
            trades_frames.append(trades_df)
            equity_frames.append(equity_df)
            comparison_rows.append(summary)

    trades = pd.concat(trades_frames, ignore_index=True, sort=False) if trades_frames else pd.DataFrame()
    equity = pd.concat(equity_frames, ignore_index=True, sort=False) if equity_frames else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows).sort_values(["ending_balance", "trade_count"], ascending=[False, False]).reset_index(drop=True)

    trades.to_csv(output_dir / "ladder_manager_trades.csv", index=False)
    equity.to_csv(output_dir / "ladder_manager_equity_curve.csv", index=False)
    comparison.to_csv(output_dir / "ladder_manager_comparison.csv", index=False)
    (output_dir / "ladder_manager_summary.txt").write_text(
        build_summary_text(comparison, source_note, filter_note, signals_path, candles_path),
        encoding="utf-8",
    )
    (output_dir / "ladder_manager_strategy.txt").write_text(
        build_strategy_text(comparison, source_note, filter_note, signals_path, candles_path),
        encoding="utf-8",
    )

    create_plots(equity, comparison, output_dir)

    print(f"Standalone ladder-manager sandbox completed. Outputs saved to: {output_dir}")
    return 0


def load_filtered_signals(path: Path, stage_letter: str, config: LadderConfig) -> tuple[pd.DataFrame, str, str]:
    frame = pd.read_csv(path)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")
    for column in ["upper_wick_ratio_5m", "atr14_pct"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "side" in frame.columns:
        frame = frame.loc[frame["side"].astype(str).str.lower() == "short"].copy()
    if "setup_name" in frame.columns:
        frame = frame.loc[frame["setup_name"].astype(str) == "eth_short_proxy"].copy()

    stage_id = f"stage_{stage_letter.lower()}"
    if "stage_id" in frame.columns:
        frame = frame.loc[frame["stage_id"].astype(str) == stage_id].copy()
        source_note = f"Source used: `{path.name}` filtered to short `eth_short_proxy` rows and `{stage_id}`."
    else:
        source_note = f"Source used: current practical Cluster 1 short model signals from `{path.name}`."

    filtered = frame.loc[
        frame["entry_time"].notna()
        & frame["entry_price"].notna()
        & frame["upper_wick_ratio_5m"].notna(),
    ].copy()
    filtered["entry_hour_utc"] = filtered["entry_time"].dt.hour
    filtered = filtered.loc[
        (filtered["entry_hour_utc"] >= config.entry_hour_start_utc)
        & (filtered["entry_hour_utc"] < config.entry_hour_end_utc)
        & (filtered["upper_wick_ratio_5m"] >= config.min_signal_filter_wick)
    ].copy()
    filtered = filtered.sort_values(["signal_id", "entry_time"]).drop_duplicates(subset=["signal_id"], keep="first")
    filtered = filtered.sort_values("entry_time").reset_index(drop=True)

    filter_note = (
        f"Signal subset: current Cluster 1 short proxy, entries only {config.entry_hour_start_utc:02d}:00-"
        f"{config.entry_hour_end_utc:02d}:00 UTC, and upper_wick_ratio_5m >= {config.min_signal_filter_wick:.2f}."
    )
    return filtered, source_note, filter_note


def load_candles(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("open_time").reset_index(drop=True)


def run_backtest(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: LadderConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    trades: list[dict] = []
    equity: list[dict] = []

    fallback_atr_pct = pd.to_numeric(signals["atr14_pct"], errors="coerce").median()

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.entry_time)
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
            manager=manager,
            risk_mode=risk_mode,
            margin_fraction=margin_fraction,
            fallback_atr_pct=fallback_atr_pct,
            trade_index=len(trades) + 1,
        )
        if trade is None:
            continue

        balance = float(trade["balance_after"])
        peak_balance = max(peak_balance, balance)
        drawdown_pct = 0.0 if peak_balance <= 0 else (peak_balance - balance) / peak_balance * 100.0
        trade["drawdown_pct"] = drawdown_pct
        trades.append(trade)
        equity.append(
            {
                "manager_id": manager["manager_id"],
                "risk_mode": risk_mode,
                "trade_index": trade["trade_index"],
                "timestamp": trade["exit_time"],
                "balance_after": balance,
                "peak_balance": peak_balance,
                "drawdown_pct": drawdown_pct,
            }
        )
        last_exit_time = pd.Timestamp(trade["exit_time"])

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)
    summary = summarize_run(trades_df, config, manager, risk_mode, margin_fraction, ignored_overlaps)
    return trades_df, equity_df, summary


def simulate_trade(
    signal,
    candles: pd.DataFrame,
    balance_before: float,
    config: LadderConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
    fallback_atr_pct: float,
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

    atr_pct = pd.to_numeric(pd.Series([getattr(signal, "atr14_pct", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(atr_pct) or atr_pct <= 0:
        atr_pct = float(fallback_atr_pct)

    exit_state = initialize_exit_state(entry_price, manager, atr_pct)
    exit_reason = "time_stop"
    market_exit_price = float(future_window.iloc[-1]["close"])
    exit_time = pd.Timestamp(future_window.iloc[-1]["close_time"])
    liquidation_flag = False

    for candle in future_window.itertuples(index=False):
        high = float(candle.high)
        low = float(candle.low)

        if high >= liquidation_price:
            exit_reason = "liquidation"
            market_exit_price = liquidation_price
            exit_time = pd.Timestamp(candle.close_time)
            liquidation_flag = True
            break

        if high >= exit_state["active_stop_price"]:
            exit_reason = "stop_loss" if exit_state["shift_count"] == 0 else "managed_stop"
            market_exit_price = exit_state["active_stop_price"]
            exit_time = pd.Timestamp(candle.close_time)
            break

        advance_exit_state(exit_state, low, entry_price)

        if low <= exit_state["active_tp_price"]:
            exit_reason = "take_profit"
            market_exit_price = exit_state["active_tp_price"]
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
        "manager_id": manager["manager_id"],
        "manager_label": manager["label"],
        "risk_mode": risk_mode,
        "trade_index": trade_index,
        "signal_id": signal.signal_id,
        "signal_time": signal_time,
        "entry_time": signal_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "shift_count": exit_state["shift_count"],
        "active_step_pct": exit_state["step_pct"],
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
    }


def initialize_exit_state(entry_price: float, manager: dict, atr_pct: float) -> dict:
    manager_type = manager["type"]
    if manager_type == "baseline_step":
        stop_pct = 1.5
        tp_pct = 1.0
        step_pct = 0.5
        active_stop_price = entry_price * (1.0 + stop_pct / 100.0)
        active_tp_price = entry_price * (1.0 - tp_pct / 100.0)
        return {
            "type": manager_type,
            "entry_price": entry_price,
            "shift_count": 0,
            "stop_pct": stop_pct,
            "tp_pct": tp_pct,
            "step_pct": step_pct,
            "next_trigger_price": entry_price * (1.0 - step_pct / 100.0),
            "active_stop_price": active_stop_price,
            "active_tp_price": active_tp_price,
        }
    if manager_type == "atr_ladder":
        step_pct = 0.25 * atr_pct
        stop_pct = 0.50 * atr_pct
        tp_pct = 0.50 * atr_pct
    else:
        step_pct = 0.50
        stop_pct = 0.50
        tp_pct = 0.50
    return {
        "type": manager_type,
        "entry_price": entry_price,
        "shift_count": 0,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "step_pct": step_pct,
        "next_trigger_price": entry_price * (1.0 - step_pct / 100.0),
        "active_stop_price": entry_price * (1.0 + stop_pct / 100.0),
        "active_tp_price": entry_price * (1.0 - tp_pct / 100.0),
    }


def advance_exit_state(exit_state: dict, low_price: float, entry_price: float) -> None:
    while low_price <= exit_state["next_trigger_price"]:
        exit_state["shift_count"] += 1
        shift = exit_state["shift_count"] * exit_state["step_pct"]
        new_stop_pct = exit_state["stop_pct"] - shift
        new_tp_pct = exit_state["tp_pct"] + shift
        exit_state["active_stop_price"] = entry_price * (1.0 + new_stop_pct / 100.0)
        exit_state["active_tp_price"] = entry_price * (1.0 - new_tp_pct / 100.0)
        next_shift = (exit_state["shift_count"] + 1) * exit_state["step_pct"]
        exit_state["next_trigger_price"] = entry_price * (1.0 - next_shift / 100.0)


def summarize_run(
    trades_df: pd.DataFrame,
    config: LadderConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
    ignored_overlaps: int,
) -> dict:
    ending_balance = float(trades_df["balance_after"].iloc[-1]) if not trades_df.empty else config.starting_balance
    max_drawdown_pct = float(pd.to_numeric(trades_df.get("drawdown_pct"), errors="coerce").max()) if not trades_df.empty else 0.0
    pnl = pd.to_numeric(trades_df.get("pnl_usd"), errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) > 0 else np.nan

    exit_reason_share = trades_df["exit_reason"].astype(str).value_counts(normalize=True) if not trades_df.empty else pd.Series(dtype=float)
    managed_stop_share = float(exit_reason_share.get("managed_stop", 0.0))
    stop_loss_share = float(exit_reason_share.get("stop_loss", 0.0))
    take_profit_share = float(exit_reason_share.get("take_profit", 0.0))

    return {
        "manager_id": manager["manager_id"],
        "manager_label": manager["label"],
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
        "managed_stop_share": managed_stop_share,
        "stop_loss_share": stop_loss_share,
        "take_profit_share": take_profit_share,
    }


def build_summary_text(
    comparison: pd.DataFrame,
    source_note: str,
    filter_note: str,
    signals_path: Path,
    candles_path: Path,
) -> str:
    return "\n".join(
        [
            "LADDER MANAGER SANDBOX SUMMARY",
            "==============================",
            "",
            "This is a standalone entertainment / exploratory wallet simulation.",
            "It is NOT part of the main reverse-analysis research conclusions.",
            "It is NOT proof that the proxy is safely tradable.",
            "",
            source_note,
            filter_note,
            f"Signals file: {signals_path}",
            f"Candles file: {candles_path}",
            "",
            "Comparison:",
            comparison.to_string(index=False),
            "",
        ]
    ).strip() + "\n"


def build_strategy_text(
    comparison: pd.DataFrame,
    source_note: str,
    filter_note: str,
    signals_path: Path,
    candles_path: Path,
) -> str:
    best = comparison.iloc[0]
    return f"""BEST LADDER MANAGER STRATEGY
============================

This is the best standalone sandbox result from the ladder-manager comparison.
It is NOT part of the main research pipeline and it is NOT proof of tradability.

{source_note}
{filter_note}
Signals file: {signals_path}
Candles file: {candles_path}

WHAT WORKS BEST IN THIS SANDBOX
-------------------------------
Manager: {best['manager_label']}
Risk mode: {best['risk_mode']} ({best['margin_fraction']:.0%} of wallet as margin)

HOW WE GET THE SIGNAL
---------------------
- We start from the existing Cluster 1 ETH short proxy signal source.
- We only keep signals inside 13:00-17:00 UTC.
- We only keep signals where upper_wick_ratio_5m >= 0.40.

OUR ENTRY POINT
---------------
- Entry happens at the proxy signal timestamp.
- Entry price is the proxy signal entry price adjusted by sandbox slippage.
- One trade at a time only.

RESULT
------
Ending balance: ${best['ending_balance']:.2f}
Total return: {best['total_return_pct']:.2f}%
Trade count: {int(best['trade_count'])}
Win rate: {best['win_rate'] * 100.0:.2f}%
Profit factor: {best['profit_factor']:.3f}
Max drawdown: {best['max_drawdown_pct']:.2f}%
Take-profit share: {best['take_profit_share'] * 100.0:.2f}%
Managed-stop share: {best['managed_stop_share'] * 100.0:.2f}%
Plain stop-loss share: {best['stop_loss_share'] * 100.0:.2f}%

INTERPRETATION
--------------
- This is the best ladder-manager sandbox candidate in this comparison.
- It is still only a sandbox candidate, not a recovered real strategy.
- The whole point of this test was to reduce too many breakeven-style managed stops after fees.
""" + "\n"


def create_plots(equity_df: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not equity_df.empty:
        best_combo = comparison.iloc[0]
        best_equity = equity_df.loc[
            (equity_df["manager_id"] == best_combo["manager_id"])
            & (equity_df["risk_mode"] == best_combo["risk_mode"])
        ].copy()
        if not best_equity.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            ordered = best_equity.sort_values("trade_index")
            ax.plot(ordered["trade_index"], ordered["balance_after"], marker="o", linewidth=1.5)
            ax.set_title("Best ladder-manager wallet balance")
            ax.set_xlabel("Trade index")
            ax.set_ylabel("Balance ($)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "ladder_manager_wallet_balance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = comparison.copy()
        labels = [f"{row.manager_id}\n{row.risk_mode}" for row in plot_df.itertuples(index=False)]
        ax.bar(labels, plot_df["ending_balance"], color="#4c78a8", alpha=0.9)
        ax.set_title("Ladder-manager comparison by ending balance")
        ax.set_ylabel("Ending balance ($)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "ladder_manager_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
