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
MAX_HOLDING_MINUTES = 720
MIN_TRADE_COUNT = 60

RISK_MODES = {
    "full_margin": 1.00,
    "half_margin": 0.50,
    "quarter_margin": 0.25,
    "tenth_margin": 0.10,
}

TIME_WINDOWS = [
    {"time_window_id": "all_day", "label": "00:00-24:00 UTC", "start_hour": 0, "end_hour": 24},
    {"time_window_id": "overlap_12_18", "label": "12:00-18:00 UTC", "start_hour": 12, "end_hour": 18},
    {"time_window_id": "afternoon_13_17", "label": "13:00-17:00 UTC", "start_hour": 13, "end_hour": 17},
    {"time_window_id": "afternoon_13_18", "label": "13:00-18:00 UTC", "start_hour": 13, "end_hour": 18},
    {"time_window_id": "late_overlap_14_19", "label": "14:00-19:00 UTC", "start_hour": 14, "end_hour": 19},
]

SIGNAL_FILTERS = [
    {
        "filter_id": "base",
        "label": "Base proxy",
        "description": "Current practical Cluster 1 short proxy with no extra signal narrowing.",
    },
    {
        "filter_id": "strict_high_zone",
        "label": "Strict 4h high proximity",
        "description": "Distance to 4h high <= 0.35%.",
    },
    {
        "filter_id": "stronger_bounce",
        "label": "Stronger 60m bounce",
        "description": "Pre-entry 60m return >= 0.60%.",
    },
    {
        "filter_id": "strong_upper_wick",
        "label": "Stronger 5m upper wick",
        "description": "Upper wick ratio 5m >= 0.40.",
    },
    {
        "filter_id": "bearish_close",
        "label": "Bearish 5m close",
        "description": "Last 5m return <= -0.05%.",
    },
    {
        "filter_id": "strict_plus_bearish",
        "label": "Strict zone + bearish close",
        "description": "Distance to 4h high <= 0.35% and last 5m return <= -0.05%.",
    },
    {
        "filter_id": "stronger_bounce_plus_bearish",
        "label": "Stronger bounce + bearish close",
        "description": "Pre-entry 60m return >= 0.60% and last 5m return <= -0.05%.",
    },
]

EXIT_PROFILES = [
    {
        "profile_id": "step_tp1_sl1p5",
        "label": "TP1.0 / SL1.5 with step manager",
        "initial_tp_pct": 1.0,
        "initial_sl_pct": 1.5,
        "first_trigger_pct": 0.5,
        "second_trigger_pct": 1.0,
        "post_trigger_tp_pct": 1.5,
        "locked_profit_pct": 0.5,
    },
    {
        "profile_id": "high_winrate_tp0p3_sl1p5",
        "label": "TP0.3 / SL1.5 with trailing extension",
        "initial_tp_pct": 0.3,
        "initial_sl_pct": 1.5,
        "first_trigger_pct": 0.5,
        "second_trigger_pct": 1.0,
        "post_trigger_tp_pct": 1.5,
        "locked_profit_pct": 0.5,
    },
    {
        "profile_id": "high_winrate_tp0p3_sl1p0",
        "label": "TP0.3 / SL1.0 with trailing extension",
        "initial_tp_pct": 0.3,
        "initial_sl_pct": 1.0,
        "first_trigger_pct": 0.5,
        "second_trigger_pct": 1.0,
        "post_trigger_tp_pct": 1.5,
        "locked_profit_pct": 0.5,
    },
]

DEFAULT_SIGNALS_PATH = Path("Another Trader/analysis_output/eth_cluster1_short_model_signals.csv")
DEFAULT_CANDLES_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
DEFAULT_OUTPUT_DIR = Path("fun_backtests/best_effort_cluster1_short")


@dataclass(frozen=True)
class SearchConfig:
    starting_balance: float = STARTING_BALANCE
    leverage: float = LEVERAGE
    stage: str = USE_STAGE
    one_trade_at_a_time: bool = ONE_TRADE_AT_A_TIME
    fee_rate_per_side_pct: float = FEE_RATE_PER_SIDE_PCT
    slippage_per_side_pct: float = SLIPPAGE_PER_SIDE_PCT
    liquidation_buffer_pct: float = LIQUIDATION_BUFFER_PCT
    max_holding_minutes: int = MAX_HOLDING_MINUTES
    min_trade_count: int = MIN_TRADE_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-effort standalone sandbox search for Cluster 1 ETH short proxy.")
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

    config = SearchConfig(stage=args.stage.upper().strip())
    signals, source_note = load_signal_source(signals_path, config.stage)
    candles = load_candles(candles_path)

    if signals.empty:
        raise RuntimeError(f"No ETH short proxy signals found in {signals_path}")
    if candles.empty:
        raise RuntimeError(f"No cached ETHUSDT 1m candles found in {candles_path}")

    comparison_rows: list[dict] = []
    best_trades_df = pd.DataFrame()
    best_equity_df = pd.DataFrame()

    best_key = None
    best_score = -np.inf
    best_ending_balance = -np.inf

    for time_window in TIME_WINDOWS:
        window_signals = apply_time_window(signals, time_window)
        for signal_filter in SIGNAL_FILTERS:
            filtered_signals = apply_signal_filter(window_signals, signal_filter["filter_id"])
            for exit_profile in EXIT_PROFILES:
                for risk_mode, margin_fraction in RISK_MODES.items():
                    trades_df, equity_df, summary = run_backtest(
                        signals=filtered_signals,
                        candles=candles,
                        config=config,
                        time_window=time_window,
                        signal_filter=signal_filter,
                        exit_profile=exit_profile,
                        risk_mode=risk_mode,
                        margin_fraction=margin_fraction,
                    )
                    comparison_rows.append(summary)

                    eligible = summary["trade_count"] >= config.min_trade_count
                    score = sandbox_score(summary, eligible)
                    if score > best_score or (
                        np.isclose(score, best_score) and summary["ending_balance"] > best_ending_balance
                    ):
                        best_score = score
                        best_ending_balance = summary["ending_balance"]
                        best_key = summary
                        best_trades_df = trades_df.copy()
                        best_equity_df = equity_df.copy()

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["eligible_min_trade_count", "ending_balance", "trade_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    comparison.to_csv(output_dir / "best_effort_search_results.csv", index=False)
    best_trades_df.to_csv(output_dir / "best_effort_best_trades.csv", index=False)
    best_equity_df.to_csv(output_dir / "best_effort_best_equity_curve.csv", index=False)
    (output_dir / "best_strategy.txt").write_text(
        build_best_strategy_text(
            best_summary=best_key,
            comparison=comparison,
            source_note=source_note,
            signals_path=signals_path,
            candles_path=candles_path,
            config=config,
        ),
        encoding="utf-8",
    )

    create_plots(best_equity_df, comparison, output_dir)

    print(f"Best-effort standalone sandbox search completed. Outputs saved to: {output_dir}")
    return 0


def load_signal_source(path: Path, stage_letter: str) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")

    if "side" in frame.columns:
        frame = frame.loc[frame["side"].astype(str).str.lower() == "short"].copy()
    if "setup_name" in frame.columns:
        frame = frame.loc[frame["setup_name"].astype(str) == "eth_short_proxy"].copy()

    stage_id = f"stage_{stage_letter.lower()}"
    source_note = ""
    if "stage_id" in frame.columns:
        frame = frame.loc[frame["stage_id"].astype(str) == stage_id].copy()
        source_note = f"Source used: `{path.name}` filtered to short `eth_short_proxy` rows and `{stage_id}`."
    else:
        source_note = f"Source used: current practical Cluster 1 short model signals from `{path.name}`."

    numeric_cols = [
        "distance_to_last_4h_high_pct",
        "return_pre_60m_pct",
        "upper_wick_ratio_5m",
        "ret_last_5m_pct",
        "ret_3c_5m_pct",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    keep_columns = [column for column in [
        "signal_id",
        "entry_time",
        "entry_price",
        "entry_hour_utc",
        "distance_to_last_4h_high_pct",
        "return_pre_60m_pct",
        "upper_wick_ratio_5m",
        "ret_last_5m_pct",
        "ret_3c_5m_pct",
    ] if column in frame.columns]
    filtered = frame.loc[frame["entry_time"].notna() & frame["entry_price"].notna(), keep_columns].copy()
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


def apply_time_window(signals: pd.DataFrame, time_window: dict) -> pd.DataFrame:
    start_hour = int(time_window["start_hour"])
    end_hour = int(time_window["end_hour"])
    mask = (signals["entry_hour_utc"] >= start_hour) & (signals["entry_hour_utc"] < end_hour)
    return signals.loc[mask].copy().reset_index(drop=True)


def apply_signal_filter(signals: pd.DataFrame, filter_id: str) -> pd.DataFrame:
    frame = signals.copy()
    if filter_id == "base":
        mask = pd.Series(True, index=frame.index)
    elif filter_id == "strict_high_zone":
        mask = pd.to_numeric(frame.get("distance_to_last_4h_high_pct"), errors="coerce") <= 0.35
    elif filter_id == "stronger_bounce":
        mask = pd.to_numeric(frame.get("return_pre_60m_pct"), errors="coerce") >= 0.60
    elif filter_id == "strong_upper_wick":
        mask = pd.to_numeric(frame.get("upper_wick_ratio_5m"), errors="coerce") >= 0.40
    elif filter_id == "bearish_close":
        mask = pd.to_numeric(frame.get("ret_last_5m_pct"), errors="coerce") <= -0.05
    elif filter_id == "strict_plus_bearish":
        mask = (
            pd.to_numeric(frame.get("distance_to_last_4h_high_pct"), errors="coerce") <= 0.35
        ) & (
            pd.to_numeric(frame.get("ret_last_5m_pct"), errors="coerce") <= -0.05
        )
    elif filter_id == "stronger_bounce_plus_bearish":
        mask = (
            pd.to_numeric(frame.get("return_pre_60m_pct"), errors="coerce") >= 0.60
        ) & (
            pd.to_numeric(frame.get("ret_last_5m_pct"), errors="coerce") <= -0.05
        )
    else:
        mask = pd.Series(True, index=frame.index)
    return frame.loc[mask.fillna(False)].copy().reset_index(drop=True)


def run_backtest(
    signals: pd.DataFrame,
    candles: pd.DataFrame,
    config: SearchConfig,
    time_window: dict,
    signal_filter: dict,
    exit_profile: dict,
    risk_mode: str,
    margin_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    balance = config.starting_balance
    peak_balance = balance
    ignored_overlaps = 0
    skipped_empty_filter = 0
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    trades: list[dict] = []
    equity: list[dict] = []

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.entry_time)
        if config.one_trade_at_a_time and signal_time < last_exit_time:
            ignored_overlaps += 1
            continue
        if balance <= 0:
            break

        trade = simulate_trade(signal, candles, balance, config, exit_profile, risk_mode, margin_fraction, len(trades) + 1)
        if trade is None:
            skipped_empty_filter += 1
            continue

        balance = float(trade["balance_after"])
        peak_balance = max(peak_balance, balance)
        drawdown_pct = 0.0 if peak_balance <= 0 else (peak_balance - balance) / peak_balance * 100.0
        trade["drawdown_pct"] = drawdown_pct
        trades.append(trade)
        equity.append(
            {
                "combo_id": build_combo_id(time_window, signal_filter, exit_profile, risk_mode),
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
        time_window=time_window,
        signal_filter=signal_filter,
        exit_profile=exit_profile,
        risk_mode=risk_mode,
        margin_fraction=margin_fraction,
        ignored_overlaps=ignored_overlaps,
        skipped_empty_filter=skipped_empty_filter,
    )
    return trades_df, equity_df, summary


def simulate_trade(
    signal,
    candles: pd.DataFrame,
    balance_before: float,
    config: SearchConfig,
    exit_profile: dict,
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
    active_tp_price = entry_price * (1.0 - exit_profile["initial_tp_pct"] / 100.0)
    active_stop_price = entry_price * (1.0 + exit_profile["initial_sl_pct"] / 100.0)
    first_trigger_price = entry_price * (1.0 - exit_profile["first_trigger_pct"] / 100.0)
    second_trigger_price = entry_price * (1.0 - exit_profile["second_trigger_pct"] / 100.0)
    moved_tp_price = entry_price * (1.0 - exit_profile["post_trigger_tp_pct"] / 100.0)
    breakeven_stop_price = entry_price
    locked_profit_stop_price = entry_price * (1.0 - exit_profile["locked_profit_pct"] / 100.0)

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

        if high >= active_stop_price:
            exit_reason = "stop_loss" if state_reached == "initial" else "managed_stop"
            market_exit_price = active_stop_price
            exit_time = pd.Timestamp(candle.close_time)
            break

        if state_reached == "initial" and low <= first_trigger_price:
            state_reached = "step1"
            active_stop_price = breakeven_stop_price
            active_tp_price = moved_tp_price

        if state_reached == "step1" and low <= second_trigger_price:
            state_reached = "step2"
            active_stop_price = locked_profit_stop_price
            active_tp_price = moved_tp_price

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
    }


def summarize_run(
    trades_df: pd.DataFrame,
    config: SearchConfig,
    time_window: dict,
    signal_filter: dict,
    exit_profile: dict,
    risk_mode: str,
    margin_fraction: float,
    ignored_overlaps: int,
    skipped_empty_filter: int,
) -> dict:
    ending_balance = float(trades_df["balance_after"].iloc[-1]) if not trades_df.empty else config.starting_balance
    max_drawdown_pct = float(pd.to_numeric(trades_df.get("drawdown_pct"), errors="coerce").max()) if not trades_df.empty else 0.0

    pnl = pd.to_numeric(trades_df.get("pnl_usd"), errors="coerce")
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and abs(losses.sum()) > 0 else np.nan
    trade_count = int(len(trades_df))
    eligible = trade_count >= config.min_trade_count

    return {
        "combo_id": build_combo_id(time_window, signal_filter, exit_profile, risk_mode),
        "time_window_id": time_window["time_window_id"],
        "time_window_label": time_window["label"],
        "signal_filter_id": signal_filter["filter_id"],
        "signal_filter_label": signal_filter["label"],
        "signal_filter_description": signal_filter["description"],
        "exit_profile_id": exit_profile["profile_id"],
        "exit_profile_label": exit_profile["label"],
        "risk_mode": risk_mode,
        "margin_fraction": margin_fraction,
        "starting_balance": config.starting_balance,
        "ending_balance": ending_balance,
        "total_return_pct": ((ending_balance / config.starting_balance) - 1.0) * 100.0 if config.starting_balance > 0 else np.nan,
        "trade_count": trade_count,
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
        "skipped_empty_filter": skipped_empty_filter,
        "eligible_min_trade_count": eligible,
    }


def build_combo_id(time_window: dict, signal_filter: dict, exit_profile: dict, risk_mode: str) -> str:
    return f"{time_window['time_window_id']}__{signal_filter['filter_id']}__{exit_profile['profile_id']}__{risk_mode}"


def sandbox_score(summary: dict, eligible: bool) -> float:
    if not eligible:
        return -1e12 + safe_num(summary["trade_count"])
    ending_balance = safe_num(summary["ending_balance"])
    max_dd = safe_num(summary["max_drawdown_pct"])
    profit_factor = safe_num(summary["profit_factor"])
    return ending_balance - 0.35 * max_dd + 20.0 * profit_factor


def safe_num(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0
    return float(numeric)


def build_best_strategy_text(
    best_summary: dict | None,
    comparison: pd.DataFrame,
    source_note: str,
    signals_path: Path,
    candles_path: Path,
    config: SearchConfig,
) -> str:
    if best_summary is None:
        return "No sandbox candidate was found.\n"

    top10 = comparison.head(10).copy()
    lines = [
        "BEST EFFORT SANDBOX STRATEGY",
        "============================",
        "",
        "This file describes the best standalone sandbox candidate found in a constrained search.",
        "It is NOT part of the main research pipeline and it is NOT proof of tradability.",
        "",
        source_note,
        f"Signals file: {signals_path}",
        f"Candles file: {candles_path}",
        "",
        f"Search constraint: minimum trade count = {config.min_trade_count}",
        f"Starting balance in search = ${config.starting_balance:.2f}",
        f"Leverage = {config.leverage:.1f}x",
        "",
        "BEST CANDIDATE",
        "--------------",
        f"Time window where we allow the signal: {best_summary['time_window_label']}",
        f"Signal filter that works best in this sandbox search: {best_summary['signal_filter_label']}",
        f"Filter description: {best_summary['signal_filter_description']}",
        f"Exit / management profile: {best_summary['exit_profile_label']}",
        f"Risk mode: {best_summary['risk_mode']} ({best_summary['margin_fraction']:.0%} of wallet as margin)",
        "",
        "How we get the signal:",
        "- We start from the existing Cluster 1 ETH short proxy signal source.",
        f"- We only keep signals inside the time window `{best_summary['time_window_label']}`.",
        f"- Then we apply the extra narrowing rule `{best_summary['signal_filter_label']}`.",
        "",
        "Our entry point:",
        "- Entry happens at the proxy signal timestamp.",
        "- Entry price is the proxy signal entry price from the signal file, adjusted by sandbox slippage.",
        "- One trade at a time only; overlapping later signals are skipped while a trade is open.",
        "",
        "Wallet outcome for the best sandbox candidate:",
        f"- Ending balance: ${best_summary['ending_balance']:.2f}",
        f"- Total return: {best_summary['total_return_pct']:.2f}%",
        f"- Trade count: {best_summary['trade_count']}",
        f"- Win rate: {best_summary['win_rate'] * 100.0:.2f}%" if pd.notna(best_summary['win_rate']) else "- Win rate: n/a",
        f"- Profit factor: {best_summary['profit_factor']:.3f}" if pd.notna(best_summary['profit_factor']) else "- Profit factor: n/a",
        f"- Max drawdown: {best_summary['max_drawdown_pct']:.2f}%",
        "",
        "Top sandbox candidates from this search:",
        top10.to_string(index=False),
        "",
        "Honest interpretation:",
        "- This is the best sandbox candidate from the tested combinations, not a recovered real strategy.",
        "- It is only a practical entertainment / exploration candidate built on the existing Cluster 1 short proxy.",
        "- The signal is still derived from the broad short-family proxy, then narrowed by time + one extra filter + risk/exit management.",
    ]
    return "\n".join(lines).strip() + "\n"


def create_plots(best_equity_df: pd.DataFrame, comparison: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not best_equity_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ordered = best_equity_df.sort_values("trade_index")
        ax.plot(ordered["trade_index"], ordered["balance_after"], marker="o", linewidth=1.5)
        ax.set_title("Best sandbox candidate wallet balance")
        ax.set_xlabel("Trade index")
        ax.set_ylabel("Balance ($)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "best_effort_wallet_balance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not comparison.empty:
        top = comparison.head(12).copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [
            f"{row.signal_filter_id}\n{row.time_window_id}\n{row.exit_profile_id}\n{row.risk_mode}"
            for row in top.itertuples(index=False)
        ]
        ax.bar(labels, top["ending_balance"], color="#4c78a8", alpha=0.9)
        ax.set_title("Top sandbox candidates by ending balance")
        ax.set_ylabel("Ending balance ($)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "best_effort_top_candidates.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
