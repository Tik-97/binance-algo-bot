from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

ENTRY_HOUR_START_UTC = 0
ENTRY_HOUR_END_UTC = 24
MIN_SIGNAL_FILTER_WICK: float | None = None

# --- New smarter entry settings ---
ENTRY_VALIDATION_MINUTES = 5
ENTRY_CANCEL_IF_NEW_HIGH_PCT = 0.15
ENTRY_REQUIRE_WEAKENING_DROP_PCT = 0.05

# --- New early fail settings ---
EARLY_FAIL_CHECK_MINUTES = 5
EARLY_FAIL_MIN_FAVORABLE_PCT = 0.10
EARLY_FAIL_MAX_ADVERSE_PCT = 0.30

RISK_MODES = {
    "full_margin": 1.00,
    "half_margin": 0.50,
    "quarter_margin": 0.25,
    "tenth_margin": 0.10,
}

EXIT_MANAGERS = [
    {
        "manager_id": "pct_partial_ladder",
        "label": "0.5% partial TP + 0.5% ladder",
        "type": "pct_partial_ladder",
    },
    {
        "manager_id": "baseline_partial_step",
        "label": "1.0% partial TP + baseline step manager",
        "type": "baseline_partial_step",
    },
    {
        "manager_id": "atr_partial_ladder",
        "label": "ATR partial TP + ATR ladder",
        "type": "atr_partial_ladder",
    },
]

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_cluster1_short_model_signals.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests/validated_entry_no_timestop")


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
    min_signal_filter_wick: float | None = MIN_SIGNAL_FILTER_WICK
    entry_validation_minutes: int = ENTRY_VALIDATION_MINUTES
    entry_cancel_if_new_high_pct: float = ENTRY_CANCEL_IF_NEW_HIGH_PCT
    entry_require_weakening_drop_pct: float = ENTRY_REQUIRE_WEAKENING_DROP_PCT
    early_fail_check_minutes: int = EARLY_FAIL_CHECK_MINUTES
    early_fail_min_favorable_pct: float = EARLY_FAIL_MIN_FAVORABLE_PCT
    early_fail_max_adverse_pct: float = EARLY_FAIL_MAX_ADVERSE_PCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone validated-entry sandbox for ETH short proxy.")
    parser.add_argument("--signals", type=str, default=str(DEFAULT_SIGNALS_PATH), help="Path to signal CSV")
    parser.add_argument("--candles", type=str, default=str(DEFAULT_CANDLES_PATH), help="Path to cached 1m candles CSV")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--stage", type=str, default=USE_STAGE, help="Stage selection")
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

    signals, source_note, filter_note = load_filtered_signals(signals_path, config)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No signals found after filtering in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No candles found in {candles_path}")

    all_trades: list[pd.DataFrame] = []
    all_equity: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

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
            all_trades.append(trades_df)
            all_equity.append(equity_df)
            comparison_rows.append(summary)

    trades = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    equity = pd.concat(all_equity, ignore_index=True, sort=False) if all_equity else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        ["ending_balance", "profit_factor", "trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    trades.to_csv(output_dir / "sandbox_wallet_trades.csv", index=False)
    equity.to_csv(output_dir / "sandbox_wallet_equity_curve.csv", index=False)
    comparison.to_csv(output_dir / "sandbox_wallet_comparison.csv", index=False)

    (output_dir / "sandbox_wallet_summary.txt").write_text(
        build_summary_text(comparison, source_note, filter_note, signals_path, candles_path, config),
        encoding="utf-8",
    )

    create_plots(equity, comparison, output_dir)

    print(f"Done. Outputs saved to: {output_dir}")
    return 0


def load_filtered_signals(path: Path, config: SandboxConfig) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)

    if "entry_time" not in df.columns or "entry_price" not in df.columns:
        raise RuntimeError("Signal file must contain at least: entry_time, entry_price")

    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")

    for col in ["upper_wick_ratio_5m", "atr14_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "side" in df.columns:
        df = df.loc[df["side"].astype(str).str.lower() == "short"].copy()

    if "setup_name" in df.columns:
        df = df.loc[df["setup_name"].astype(str) == "eth_short_proxy"].copy()

    stage_id = f"stage_{config.stage.lower()}"
    if "stage_id" in df.columns:
        df = df.loc[df["stage_id"].astype(str) == stage_id].copy()
        source_note = f"Source used: `{path.name}` filtered to short `eth_short_proxy` rows and `{stage_id}`."
    else:
        source_note = f"Source used: current practical ETH short proxy rows from `{path.name}`."

    df = df.loc[df["entry_time"].notna() & df["entry_price"].notna()].copy()

    df["entry_hour_utc"] = df["entry_time"].dt.hour
    df = df.loc[
        (df["entry_hour_utc"] >= config.entry_hour_start_utc)
        & (df["entry_hour_utc"] < config.entry_hour_end_utc)
    ].copy()

    wick_note = "no extra wick filter"
    if config.min_signal_filter_wick is not None and "upper_wick_ratio_5m" in df.columns:
        df = df.loc[df["upper_wick_ratio_5m"] >= config.min_signal_filter_wick].copy()
        wick_note = f"upper_wick_ratio_5m >= {config.min_signal_filter_wick:.2f}"

    if "signal_id" in df.columns:
        df = df.sort_values(["signal_id", "entry_time"]).drop_duplicates(subset=["signal_id"], keep="first")

    df = df.sort_values("entry_time").reset_index(drop=True)

    filter_note = (
        f"Signal subset: entries only {config.entry_hour_start_utc:02d}:00-"
        f"{config.entry_hour_end_utc:02d}:00 UTC, {wick_note}."
    )

    return df, source_note, filter_note


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["open_time", "close_time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Candles file missing columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open_time", "close_time", "open", "high", "low", "close"]).copy()
    return df.sort_values("open_time").reset_index(drop=True)


def run_backtest(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: SandboxConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    cancelled_entries = 0
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    trades: list[dict[str, Any]] = []
    equity: list[dict[str, Any]] = []

    fallback_atr_pct = pd.to_numeric(signals.get("atr14_pct"), errors="coerce").median()
    if pd.isna(fallback_atr_pct) or fallback_atr_pct <= 0:
        fallback_atr_pct = 0.5

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
            fallback_atr_pct=float(fallback_atr_pct),
            trade_index=len(trades) + 1,
        )

        if trade is None:
            cancelled_entries += 1
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
    summary = summarize_run(
        trades_df=trades_df,
        config=config,
        manager=manager,
        risk_mode=risk_mode,
        margin_fraction=margin_fraction,
        ignored_overlaps=ignored_overlaps,
        cancelled_entries=cancelled_entries,
    )
    return trades_df, equity_df, summary


def simulate_trade(
    signal,
    candles: pd.DataFrame,
    balance_before: float,
    config: SandboxConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
    fallback_atr_pct: float,
    trade_index: int,
) -> dict[str, Any] | None:
    signal_time = pd.Timestamp(signal.entry_time)

    # -------- Entry validation window --------
    validation_cutoff = signal_time + pd.Timedelta(minutes=config.entry_validation_minutes)
    validation = candles.loc[
        (candles["open_time"] > signal_time) & (candles["open_time"] <= validation_cutoff)
    ].copy()
    if validation.empty:
        return None

    signal_price = float(signal.entry_price)

    # cancel if strong continuation upward after signal
    validation_high = float(validation["high"].max())
    continuation_up_pct = (validation_high / signal_price - 1.0) * 100.0
    if continuation_up_pct >= config.entry_cancel_if_new_high_pct:
        return None

    # require at least a small weakening move downward before entry
    validation_low = float(validation["low"].min())
    weakening_down_pct = (signal_price / validation_low - 1.0) * 100.0
    if weakening_down_pct < config.entry_require_weakening_drop_pct:
        return None

    # entry at end of validation window
    entry_candle = validation.iloc[-1]
    entry_time = pd.Timestamp(entry_candle["close_time"])
    entry_price_raw = float(entry_candle["close"])

    future = candles.loc[candles["open_time"] > entry_time].copy()
    if future.empty:
        return None

    margin_used = balance_before * margin_fraction
    notional_used = margin_used * config.leverage

    entry_price = entry_price_raw * (1.0 - config.slippage_per_side_pct / 100.0)
    liquidation_move_pct = max((100.0 / config.leverage) - config.liquidation_buffer_pct, 0.01)
    liquidation_price = entry_price * (1.0 + liquidation_move_pct / 100.0)

    atr_pct = pd.to_numeric(pd.Series([getattr(signal, "atr14_pct", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(atr_pct) or atr_pct <= 0:
        atr_pct = fallback_atr_pct

    state = initialize_exit_state(entry_price, manager, atr_pct)

    exit_reason = "managed_stop"
    exit_time = pd.Timestamp(future.iloc[-1]["close_time"])
    exit_price_market = float(future.iloc[-1]["close"])
    liquidation_flag = False

    realized_underlying_return_pct = 0.0
    realized_fee_cost_usd = notional_used * (config.fee_rate_per_side_pct / 100.0)

    # -------- Early fail check --------
    early_fail_cutoff = entry_time + pd.Timedelta(minutes=config.early_fail_check_minutes)
    early_window = future.loc[future["open_time"] <= early_fail_cutoff].copy()
    if not early_window.empty:
        early_low = float(early_window["low"].min())
        early_high = float(early_window["high"].max())

        favorable_pct = (entry_price / early_low - 1.0) * 100.0
        adverse_pct = (early_high / entry_price - 1.0) * 100.0

        if favorable_pct < config.early_fail_min_favorable_pct and adverse_pct >= config.early_fail_max_adverse_pct:
            early_fail_exit_price = entry_price * (1.0 + config.early_fail_max_adverse_pct / 100.0)
            exit_price = early_fail_exit_price * (1.0 + config.slippage_per_side_pct / 100.0)
            gross_underlying_return_pct = (entry_price - exit_price) / entry_price * 100.0
            realized_fee_cost_usd += notional_used * (config.fee_rate_per_side_pct / 100.0)
            pnl_usd = notional_used * gross_underlying_return_pct / 100.0 - realized_fee_cost_usd
            balance_after = max(balance_before + pnl_usd, 0.0)
            net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan

            return {
                "manager_id": manager["manager_id"],
                "manager_label": manager["label"],
                "risk_mode": risk_mode,
                "trade_index": trade_index,
                "signal_id": getattr(signal, "signal_id", trade_index),
                "signal_time": signal_time,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": pd.Timestamp(early_window.iloc[-1]["close_time"]),
                "exit_price": exit_price,
                "exit_reason": "early_fail",
                "shift_count": 0,
                "partial_tp_done": False,
                "remaining_fraction": 1.0,
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
                "liquidation_flag": False,
            }

    # -------- Main trade loop, no time stop --------
    for candle in future.itertuples(index=False):
        high = float(candle.high)
        low = float(candle.low)
        close_time = pd.Timestamp(candle.close_time)

        if high >= liquidation_price:
            liquidation_flag = True
            exit_reason = "liquidation"
            exit_time = close_time
            exit_price_market = liquidation_price
            break

        current_stop_price = state["active_stop_price"]
        current_tp_price = state["active_tp_price"]

        event = resolve_candle_events_for_short(
            high=high,
            low=low,
            liquidation_price=liquidation_price,
            stop_price=current_stop_price,
            tp_price=current_tp_price,
        )

        if event == "stop_loss":
            exit_reason = "stop_loss" if state["shift_count"] == 0 else "managed_stop"
            exit_time = close_time
            exit_price_market = current_stop_price
            break

        if event == "take_profit":
            if not state["partial_tp_done"]:
                realized_part = state["partial_tp_fraction"]
                part_return = (entry_price - current_tp_price) / entry_price * 100.0
                realized_underlying_return_pct += realized_part * part_return
                realized_fee_cost_usd += (notional_used * realized_part) * (config.fee_rate_per_side_pct / 100.0)

                state["remaining_fraction"] -= realized_part
                state["partial_tp_done"] = True

                apply_post_partial_shift(state, entry_price)

                if state["remaining_fraction"] <= 1e-9:
                    exit_reason = "take_profit"
                    exit_time = close_time
                    exit_price_market = current_tp_price
                    break

                continue
            else:
                rem = state["remaining_fraction"]
                part_return = (entry_price - current_tp_price) / entry_price * 100.0
                realized_underlying_return_pct += rem * part_return
                realized_fee_cost_usd += (notional_used * rem) * (config.fee_rate_per_side_pct / 100.0)

                state["remaining_fraction"] = 0.0
                exit_reason = "take_profit"
                exit_time = close_time
                exit_price_market = current_tp_price
                break

        if low <= state["next_shift_trigger_price"]:
            advance_exit_state(state, entry_price)

    if liquidation_flag:
        pnl_usd = -margin_used
        balance_after = max(balance_before + pnl_usd, 0.0)
        net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan
        return {
            "manager_id": manager["manager_id"],
            "manager_label": manager["label"],
            "risk_mode": risk_mode,
            "trade_index": trade_index,
            "signal_id": getattr(signal, "signal_id", trade_index),
            "signal_time": signal_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price_market,
            "exit_reason": exit_reason,
            "shift_count": state["shift_count"],
            "partial_tp_done": state["partial_tp_done"],
            "remaining_fraction": state["remaining_fraction"],
            "balance_before": balance_before,
            "margin_fraction": margin_fraction,
            "margin_used": margin_used,
            "notional_used": notional_used,
            "leverage": config.leverage,
            "gross_underlying_return_pct": -margin_used / notional_used * 100.0 if notional_used > 0 else np.nan,
            "net_account_return_pct": net_account_return_pct,
            "pnl_usd": pnl_usd,
            "balance_after": balance_after,
            "drawdown_pct": np.nan,
            "liquidation_flag": True,
        }

    # exit remaining fraction at final market price when managed stop finally happens or dataset ends
    if state["remaining_fraction"] > 1e-9:
        exit_price = exit_price_market * (1.0 + config.slippage_per_side_pct / 100.0)
        rem = state["remaining_fraction"]
        rem_return = (entry_price - exit_price) / entry_price * 100.0
        realized_underlying_return_pct += rem * rem_return
        realized_fee_cost_usd += (notional_used * rem) * (config.fee_rate_per_side_pct / 100.0)
    else:
        exit_price = exit_price_market

    pnl_usd = notional_used * realized_underlying_return_pct / 100.0 - realized_fee_cost_usd
    balance_after = max(balance_before + pnl_usd, 0.0)
    net_account_return_pct = (pnl_usd / balance_before) * 100.0 if balance_before > 0 else np.nan

    return {
        "manager_id": manager["manager_id"],
        "manager_label": manager["label"],
        "risk_mode": risk_mode,
        "trade_index": trade_index,
        "signal_id": getattr(signal, "signal_id", trade_index),
        "signal_time": signal_time,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "shift_count": state["shift_count"],
        "partial_tp_done": state["partial_tp_done"],
        "remaining_fraction": state["remaining_fraction"],
        "balance_before": balance_before,
        "margin_fraction": margin_fraction,
        "margin_used": margin_used,
        "notional_used": notional_used,
        "leverage": config.leverage,
        "gross_underlying_return_pct": realized_underlying_return_pct,
        "net_account_return_pct": net_account_return_pct,
        "pnl_usd": pnl_usd,
        "balance_after": balance_after,
        "drawdown_pct": np.nan,
        "liquidation_flag": False,
    }


def resolve_candle_events_for_short(
    high: float,
    low: float,
    liquidation_price: float,
    stop_price: float,
    tp_price: float,
) -> str | None:
    if high >= liquidation_price:
        return "liquidation"

    stop_hit = high >= stop_price
    tp_hit = low <= tp_price

    if stop_hit and tp_hit:
        return "stop_loss"

    if stop_hit:
        return "stop_loss"

    if tp_hit:
        return "take_profit"

    return None


def initialize_exit_state(entry_price: float, manager: dict, atr_pct: float) -> dict[str, Any]:
    manager_type = manager["type"]

    if manager_type == "pct_partial_ladder":
        stop_pct = 0.50
        first_tp_pct = 0.50
        next_tp_pct = 1.00
        shift_trigger_pct = 0.25
        shift_step_pct = 0.50
        partial_tp_fraction = 0.50

    elif manager_type == "baseline_partial_step":
        stop_pct = 1.50
        first_tp_pct = 1.00
        next_tp_pct = 1.50
        shift_trigger_pct = 0.50
        shift_step_pct = 0.50
        partial_tp_fraction = 0.50

    elif manager_type == "atr_partial_ladder":
        base = max(0.50 * atr_pct, 0.10)
        stop_pct = base
        first_tp_pct = base
        next_tp_pct = base * 2.0
        shift_trigger_pct = max(0.25 * atr_pct, 0.05)
        shift_step_pct = max(0.50 * atr_pct, 0.10)
        partial_tp_fraction = 0.50

    else:
        raise ValueError(f"Unknown manager type: {manager_type}")

    return {
        "type": manager_type,
        "entry_price": entry_price,
        "shift_count": 0,
        "initial_stop_pct": stop_pct,
        "initial_first_tp_pct": first_tp_pct,
        "current_tp_pct": first_tp_pct,
        "next_tp_pct_after_partial": next_tp_pct,
        "shift_trigger_pct": shift_trigger_pct,
        "shift_step_pct": shift_step_pct,
        "active_stop_pct": stop_pct,
        "active_stop_price": entry_price * (1.0 + stop_pct / 100.0),
        "active_tp_price": entry_price * (1.0 - first_tp_pct / 100.0),
        "next_shift_trigger_price": entry_price * (1.0 - shift_trigger_pct / 100.0),
        "partial_tp_fraction": partial_tp_fraction,
        "partial_tp_done": False,
        "remaining_fraction": 1.0,
    }


def apply_post_partial_shift(state: dict[str, Any], entry_price: float) -> None:
    tightened_stop_pct = max(state["initial_stop_pct"] - state["shift_step_pct"], 0.05)
    state["active_stop_pct"] = tightened_stop_pct
    state["active_stop_price"] = entry_price * (1.0 + tightened_stop_pct / 100.0)

    state["current_tp_pct"] = state["next_tp_pct_after_partial"]
    state["active_tp_price"] = entry_price * (1.0 - state["current_tp_pct"] / 100.0)

    next_shift_total = state["shift_trigger_pct"] * (state["shift_count"] + 2)
    state["next_shift_trigger_price"] = entry_price * (1.0 - next_shift_total / 100.0)


def advance_exit_state(state: dict[str, Any], entry_price: float) -> None:
    state["shift_count"] += 1
    shift_total = state["shift_count"] * state["shift_step_pct"]

    new_stop_pct = max(state["initial_stop_pct"] - shift_total, 0.05)
    state["active_stop_pct"] = new_stop_pct
    state["active_stop_price"] = entry_price * (1.0 + new_stop_pct / 100.0)

    state["current_tp_pct"] += state["shift_step_pct"]
    state["active_tp_price"] = entry_price * (1.0 - state["current_tp_pct"] / 100.0)

    next_shift_total = state["shift_trigger_pct"] * (state["shift_count"] + 1)
    state["next_shift_trigger_price"] = entry_price * (1.0 - next_shift_total / 100.0)


def summarize_run(
    trades_df: pd.DataFrame,
    config: SandboxConfig,
    manager: dict,
    risk_mode: str,
    margin_fraction: float,
    ignored_overlaps: int,
    cancelled_entries: int,
) -> dict[str, Any]:
    ending_balance = float(trades_df["balance_after"].iloc[-1]) if not trades_df.empty else config.starting_balance
    max_drawdown_pct = float(pd.to_numeric(trades_df.get("drawdown_pct"), errors="coerce").max()) if not trades_df.empty else 0.0

    pnl = pd.to_numeric(trades_df.get("pnl_usd"), errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) > 0 else np.nan
    exit_reason_share = trades_df["exit_reason"].astype(str).value_counts(normalize=True) if not trades_df.empty else pd.Series(dtype=float)

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
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "max_drawdown_pct": max_drawdown_pct,
        "best_trade_usd": float(pnl.max()) if not pnl.empty else np.nan,
        "worst_trade_usd": float(pnl.min()) if not pnl.empty else np.nan,
        "ignored_overlapping_signals": ignored_overlaps,
        "cancelled_entries": cancelled_entries,
        "managed_stop_share": float(exit_reason_share.get("managed_stop", 0.0)),
        "stop_loss_share": float(exit_reason_share.get("stop_loss", 0.0)),
        "take_profit_share": float(exit_reason_share.get("take_profit", 0.0)),
        "early_fail_share": float(exit_reason_share.get("early_fail", 0.0)),
        "liquidation_share": float(exit_reason_share.get("liquidation", 0.0)),
        "partial_tp_share": float(pd.to_numeric(trades_df.get("partial_tp_done"), errors="coerce").fillna(False).astype(bool).mean())
        if not trades_df.empty else 0.0,
    }


def build_summary_text(
    comparison: pd.DataFrame,
    source_note: str,
    filter_note: str,
    signals_path: Path,
    candles_path: Path,
    config: SandboxConfig,
) -> str:
    lines = [
        "SANDBOX WALLET SUMMARY V3",
        "=========================",
        "",
        "This is a standalone entertainment / exploratory wallet simulation.",
        "It is NOT part of the main reverse-analysis research conclusions.",
        "It is NOT proof that the proxy is safely tradable.",
        "It uses approximate liquidation logic.",
        "",
        source_note,
        filter_note,
        f"Signals file: {signals_path}",
        f"Candles file: {candles_path}",
        "",
        "Entry logic:",
        f"- Validation window: {config.entry_validation_minutes} minutes",
        f"- Cancel if new high after signal >= {config.entry_cancel_if_new_high_pct:.2f}%",
        f"- Require weakening drop before entry >= {config.entry_require_weakening_drop_pct:.2f}%",
        "",
        "Post-entry logic:",
        f"- Early fail check window: {config.early_fail_check_minutes} minutes",
        f"- Early fail if favorable move < {config.early_fail_min_favorable_pct:.2f}% and adverse move >= {config.early_fail_max_adverse_pct:.2f}%",
        "- No time stop",
        "- Partial TP + managed stop logic",
        "",
        "Comparison:",
        comparison.to_string(index=False),
        "",
    ]
    return "\n".join(lines)


def create_plots(equity_df: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not comparison.empty and not equity_df.empty:
        best = comparison.iloc[0]
        best_equity = equity_df.loc[
            (equity_df["manager_id"] == best["manager_id"])
            & (equity_df["risk_mode"] == best["risk_mode"])
        ].copy()

        if not best_equity.empty:
            best_equity = best_equity.sort_values("trade_index")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(best_equity["trade_index"], best_equity["balance_after"], linewidth=1.5)
            ax.set_title("Best sandbox wallet balance")
            ax.set_xlabel("Trade index")
            ax.set_ylabel("Balance ($)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "sandbox_wallet_balance.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(best_equity["trade_index"], best_equity["drawdown_pct"], linewidth=1.5)
            ax.set_title("Best sandbox drawdown")
            ax.set_xlabel("Trade index")
            ax.set_ylabel("Drawdown %")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots_dir / "sandbox_wallet_drawdown.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"{row.manager_id}\n{row.risk_mode}" for row in comparison.itertuples(index=False)]
        ax.bar(labels, comparison["ending_balance"])
        ax.set_title("Sandbox comparison by ending balance")
        ax.set_ylabel("Ending balance ($)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "sandbox_wallet_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())