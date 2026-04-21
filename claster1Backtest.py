from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
CANDLES_1M_PATH = Path(".cache/binance_klines/ETHUSDT_1m.csv")
OUTPUT_DIR = Path("fun_backtests/cluster1_short_backtest")

STARTING_BALANCE = 400.0
RISK_PER_TRADE_PCT = 1.0          # risk as % of current balance
LEVERAGE = 10.0                   # used only for notional / margin info
ONE_TRADE_AT_A_TIME = True

FEE_PER_SIDE_PCT = 0.04
SLIPPAGE_PER_SIDE_PCT = 0.025

# Cluster 1-inspired context
LOOKBACK_4H_MINUTES = 240
BOUNCE_LOOKBACK_MINUTES = 60
NEAR_4H_HIGH_MAX_PCT = 0.60       # distance to 4h high <= 0.60%
MIN_BOUNCE_60M_PCT = 0.45         # last 60m return >= 0.45%
MIN_STRETCH_ABOVE_EMA20_5M_PCT = 0.15

# Trigger
TRIGGER_MODE = "close_back_down"  # "weak_candle" | "close_back_down" | "failed_continuation" | "small_body_rejection"
TRIGGER_WINDOW_MINUTES = 5

# Stop / managed exit
INITIAL_STOP_PCT = 0.70           # hard stop from entry
BREAKEVEN_TRIGGER_PCT = 0.35      # if move goes this much in favor -> move stop tighter
LOCK_PROFIT_TRIGGER_PCT = 0.70    # if move goes this much in favor -> trail more
LOCK_PROFIT_STOP_PCT = -0.10      # stop becomes entry - 0.10% (profit lock for short)
EARLY_FAIL_WINDOW_MINUTES = 5
EARLY_FAIL_MIN_FAVORABLE_PCT = 0.08
EARLY_FAIL_MAX_ADVERSE_PCT = 0.25

MAX_BARS_SCAN = 60 * 24 * 3       # max 3 days, just to avoid infinite hold in dataset

SYMBOL = "ETHUSDT"


# =========================
# HELPERS
# =========================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str
    exit_reason: str
    stop_price_initial: float
    stop_price_final: float
    balance_before: float
    balance_after: float
    risk_amount_usd: float
    qty: float
    notional_usd: float
    pnl_usd: float
    pnl_pct_balance: float
    mfe_pct: float
    mae_pct: float
    hold_minutes: float
    signal_time: pd.Timestamp
    context_dist_4h_high_pct: float
    context_bounce_60m_pct: float
    context_stretch_above_ema20_5m_pct: float


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["open_time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open_time", "open", "high", "low", "close"]).copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_1m.set_index("open_time")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def build_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()

    # 4h high distance on 1m data
    rolling_4h_high = df["high"].rolling(LOOKBACK_4H_MINUTES, min_periods=LOOKBACK_4H_MINUTES).max()
    df["dist_4h_high_pct"] = (rolling_4h_high - df["close"]) / df["close"] * 100.0

    # 60m bounce
    close_60m_ago = df["close"].shift(BOUNCE_LOOKBACK_MINUTES)
    df["bounce_60m_pct"] = (df["close"] / close_60m_ago - 1.0) * 100.0

    # 5m EMA20 stretch
    df5 = resample_5m(df)
    df5["ema20_5m"] = ema(df5["close"], 20)
    df5["stretch_above_ema20_5m_pct"] = (df5["close"] / df5["ema20_5m"] - 1.0) * 100.0

    # merge latest completed 5m feature back to 1m
    df = pd.merge_asof(
        df.sort_values("open_time"),
        df5[["open_time", "stretch_above_ema20_5m_pct"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    # micro candle features
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["body_ratio"] = (body / rng).fillna(0.0)
    df["upper_wick_ratio"] = (upper_wick / rng).fillna(0.0)

    return df


def is_cluster1_context(row: pd.Series) -> bool:
    vals = [
        pd.notna(row["dist_4h_high_pct"]),
        pd.notna(row["bounce_60m_pct"]),
        pd.notna(row["stretch_above_ema20_5m_pct"]),
    ]
    if not all(vals):
        return False

    return (
        row["dist_4h_high_pct"] <= NEAR_4H_HIGH_MAX_PCT
        and row["bounce_60m_pct"] >= MIN_BOUNCE_60M_PCT
        and row["stretch_above_ema20_5m_pct"] >= MIN_STRETCH_ABOVE_EMA20_5M_PCT
    )


def trigger_fires(window: pd.DataFrame) -> bool:
    if len(window) < 2:
        return False

    last = window.iloc[-1]
    prev = window.iloc[-2]

    if TRIGGER_MODE == "weak_candle":
        return last["body_ratio"] <= 0.30

    if TRIGGER_MODE == "close_back_down":
        return last["close"] < prev["close"]

    if TRIGGER_MODE == "failed_continuation":
        # tried to go up, but closed weaker
        return last["high"] >= prev["high"] and last["close"] < prev["close"]

    if TRIGGER_MODE == "small_body_rejection":
        return last["body_ratio"] <= 0.35 and last["upper_wick_ratio"] >= 0.35

    return False


def intrabar_adverse_first_for_short(high: float, low: float, stop_price: float, target_price: float | None = None) -> str | None:
    stop_hit = high >= stop_price
    target_hit = target_price is not None and low <= target_price

    if stop_hit and target_hit:
        return "stop"
    if stop_hit:
        return "stop"
    if target_hit:
        return "target"
    return None


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    trades: list[Trade] = []
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    i = max(LOOKBACK_4H_MINUTES, BOUNCE_LOOKBACK_MINUTES) + 30

    while i < len(df) - TRIGGER_WINDOW_MINUTES - 2:
        row = df.iloc[i]
        now = row["open_time"]

        if ONE_TRADE_AT_A_TIME and now < last_exit_time:
            i += 1
            continue

        if not is_cluster1_context(row):
            i += 1
            continue

        trigger_window = df.iloc[i : i + TRIGGER_WINDOW_MINUTES]
        if len(trigger_window) < TRIGGER_WINDOW_MINUTES or not trigger_fires(trigger_window):
            i += 1
            continue

        signal_time = row["open_time"]
        entry_bar = df.iloc[i + TRIGGER_WINDOW_MINUTES - 1]
        entry_time = entry_bar["open_time"]
        raw_entry = float(entry_bar["close"])
        entry_price = raw_entry * (1.0 - SLIPPAGE_PER_SIDE_PCT / 100.0)

        stop_price = entry_price * (1.0 + INITIAL_STOP_PCT / 100.0)

        risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
        stop_distance = stop_price - entry_price
        if stop_distance <= 0:
            i += 1
            continue

        qty = risk_amount / stop_distance
        notional = qty * entry_price

        fee_entry = notional * (FEE_PER_SIDE_PCT / 100.0)

        best_low = entry_price
        best_high = entry_price
        stop_current = stop_price
        exit_reason = "dataset_end"
        exit_price = float(df.iloc[min(i + MAX_BARS_SCAN, len(df) - 1)]["close"])
        exit_time = df.iloc[min(i + MAX_BARS_SCAN, len(df) - 1)]["open_time"]

        partial_taken = False
        partial_realized_pnl = 0.0
        remaining_qty = qty

        # early fail check
        early_end_idx = min(i + TRIGGER_WINDOW_MINUTES - 1 + EARLY_FAIL_WINDOW_MINUTES, len(df) - 1)
        early_slice = df.iloc[i + TRIGGER_WINDOW_MINUTES : early_end_idx + 1]
        if not early_slice.empty:
            early_low = float(early_slice["low"].min())
            early_high = float(early_slice["high"].max())

            favorable_pct = (entry_price - early_low) / entry_price * 100.0
            adverse_pct = (early_high - entry_price) / entry_price * 100.0

            if favorable_pct < EARLY_FAIL_MIN_FAVORABLE_PCT and adverse_pct >= EARLY_FAIL_MAX_ADVERSE_PCT:
                exit_reason = "early_fail"
                raw_exit = entry_price * (1.0 + EARLY_FAIL_MAX_ADVERSE_PCT / 100.0)
                exit_price = raw_exit * (1.0 + SLIPPAGE_PER_SIDE_PCT / 100.0)
                exit_time = early_slice.iloc[-1]["open_time"]

                gross_pnl = (entry_price - exit_price) * remaining_qty
                fee_exit = (remaining_qty * exit_price) * (FEE_PER_SIDE_PCT / 100.0)
                pnl = gross_pnl - fee_entry - fee_exit
                balance_after = balance + pnl

                mae_pct = adverse_pct
                mfe_pct = favorable_pct

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        side="Short",
                        exit_reason=exit_reason,
                        stop_price_initial=stop_price,
                        stop_price_final=stop_current,
                        balance_before=balance,
                        balance_after=balance_after,
                        risk_amount_usd=risk_amount,
                        qty=qty,
                        notional_usd=notional,
                        pnl_usd=pnl,
                        pnl_pct_balance=(pnl / balance * 100.0) if balance > 0 else np.nan,
                        mfe_pct=mfe_pct,
                        mae_pct=mae_pct,
                        hold_minutes=(exit_time - entry_time).total_seconds() / 60.0,
                        signal_time=signal_time,
                        context_dist_4h_high_pct=float(row["dist_4h_high_pct"]),
                        context_bounce_60m_pct=float(row["bounce_60m_pct"]),
                        context_stretch_above_ema20_5m_pct=float(row["stretch_above_ema20_5m_pct"]),
                    )
                )

                balance = balance_after
                peak_balance = max(peak_balance, balance)
                last_exit_time = exit_time
                i = early_end_idx + 1
                continue

        # main managed exit loop, no time stop
        start_j = i + TRIGGER_WINDOW_MINUTES
        end_j = min(start_j + MAX_BARS_SCAN, len(df) - 1)

        for j in range(start_j, end_j + 1):
            bar = df.iloc[j]
            high = float(bar["high"])
            low = float(bar["low"])

            best_low = min(best_low, low)
            best_high = max(best_high, high)

            favorable_pct_now = (entry_price - best_low) / entry_price * 100.0

            if favorable_pct_now >= LOCK_PROFIT_TRIGGER_PCT:
                stop_current = min(stop_current, entry_price * (1.0 + LOCK_PROFIT_STOP_PCT / 100.0))
            elif favorable_pct_now >= BREAKEVEN_TRIGGER_PCT:
                stop_current = min(stop_current, entry_price)

            # partial TP only once
            if not partial_taken and low <= entry_price * (1.0 - BREAKEVEN_TRIGGER_PCT / 100.0):
                partial_qty = remaining_qty * 0.5
                partial_exit_price = entry_price * (1.0 - BREAKEVEN_TRIGGER_PCT / 100.0)
                partial_exit_price *= (1.0 + SLIPPAGE_PER_SIDE_PCT / 100.0)  # short exit worse
                partial_fee = (partial_qty * partial_exit_price) * (FEE_PER_SIDE_PCT / 100.0)
                partial_realized_pnl += (entry_price - partial_exit_price) * partial_qty - partial_fee
                remaining_qty -= partial_qty
                partial_taken = True
                stop_current = min(stop_current, entry_price)

            event = intrabar_adverse_first_for_short(high=high, low=low, stop_price=stop_current, target_price=None)
            if event == "stop":
                exit_reason = "managed_stop" if stop_current < stop_price else "stop_loss"
                raw_exit = stop_current
                exit_price = raw_exit * (1.0 + SLIPPAGE_PER_SIDE_PCT / 100.0)
                exit_time = bar["open_time"]
                break

        gross_pnl_remaining = (entry_price - exit_price) * remaining_qty
        fee_exit_remaining = (remaining_qty * exit_price) * (FEE_PER_SIDE_PCT / 100.0)
        pnl = partial_realized_pnl + gross_pnl_remaining - fee_entry - fee_exit_remaining
        balance_after = balance + pnl

        mfe_pct = (entry_price - best_low) / entry_price * 100.0
        mae_pct = (best_high - entry_price) / entry_price * 100.0

        trades.append(
            Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                side="Short",
                exit_reason=exit_reason,
                stop_price_initial=stop_price,
                stop_price_final=stop_current,
                balance_before=balance,
                balance_after=balance_after,
                risk_amount_usd=risk_amount,
                qty=qty,
                notional_usd=notional,
                pnl_usd=pnl,
                pnl_pct_balance=(pnl / balance * 100.0) if balance > 0 else np.nan,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                hold_minutes=(exit_time - entry_time).total_seconds() / 60.0,
                signal_time=signal_time,
                context_dist_4h_high_pct=float(row["dist_4h_high_pct"]),
                context_bounce_60m_pct=float(row["bounce_60m_pct"]),
                context_stretch_above_ema20_5m_pct=float(row["stretch_above_ema20_5m_pct"]),
            )
        )

        balance = balance_after
        peak_balance = max(peak_balance, balance)
        last_exit_time = exit_time
        i = end_j + 1 if exit_reason == "dataset_end" else j + 1

    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    if trades_df.empty:
        summary = {
            "trade_count": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_pct": np.nan,
            "expectancy_usd": np.nan,
            "ending_balance": STARTING_BALANCE,
        }
        return trades_df, summary

    equity = trades_df["balance_after"]
    rolling_peak = equity.cummax()
    dd = (rolling_peak - equity) / rolling_peak.replace(0, np.nan) * 100.0
    wins = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losses = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]

    summary = {
        "trade_count": int(len(trades_df)),
        "win_rate": float((trades_df["pnl_usd"] > 0).mean() * 100.0),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
        "max_drawdown_pct": float(dd.max()) if len(dd) else np.nan,
        "expectancy_usd": float(trades_df["pnl_usd"].mean()),
        "ending_balance": float(trades_df["balance_after"].iloc[-1]),
        "avg_hold_minutes": float(trades_df["hold_minutes"].mean()),
        "avg_mfe_pct": float(trades_df["mfe_pct"].mean()),
        "avg_mae_pct": float(trades_df["mae_pct"].mean()),
    }

    return trades_df, summary


def save_outputs(trades_df: pd.DataFrame, summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(OUTPUT_DIR / "cluster1_short_trades.csv", index=False)

    summary_text = "\n".join(
        [
            "CLUSTER 1 SHORT BACKTEST",
            "========================",
            "",
            "Standalone rule-based backtest inspired by recovered Cluster 1 short family.",
            "This is NOT the copy trader exact strategy.",
            "",
            "Context filter:",
            f"- near 4h high <= {NEAR_4H_HIGH_MAX_PCT:.2f}%",
            f"- 60m bounce >= {MIN_BOUNCE_60M_PCT:.2f}%",
            f"- stretch above EMA20 5m >= {MIN_STRETCH_ABOVE_EMA20_5M_PCT:.2f}%",
            "",
            f"Trigger mode: {TRIGGER_MODE}",
            f"Trigger window: {TRIGGER_WINDOW_MINUTES} minutes",
            f"Initial stop: {INITIAL_STOP_PCT:.2f}%",
            f"Risk per trade: {RISK_PER_TRADE_PCT:.2f}% of balance",
            f"Leverage info: {LEVERAGE:.1f}x",
            "",
            f"Trade count: {summary['trade_count']}",
            f"Win rate: {summary['win_rate']:.2f}%",
            f"Profit factor: {summary['profit_factor']:.3f}" if not pd.isna(summary["profit_factor"]) else "Profit factor: n/a",
            f"Max drawdown: {summary['max_drawdown_pct']:.2f}%" if not pd.isna(summary["max_drawdown_pct"]) else "Max drawdown: n/a",
            f"Expectancy: {summary['expectancy_usd']:.2f} USD",
            f"Ending balance: {summary['ending_balance']:.2f} USD",
            f"Average hold: {summary['avg_hold_minutes']:.2f} min",
            f"Average MFE: {summary['avg_mfe_pct']:.2f}%",
            f"Average MAE: {summary['avg_mae_pct']:.2f}%",
            "",
        ]
    )
    (OUTPUT_DIR / "cluster1_short_summary.txt").write_text(summary_text, encoding="utf-8")

    if not trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(trades_df["exit_time"]), trades_df["balance_after"], linewidth=1.5)
        ax.set_title("Cluster 1 short equity curve")
        ax.set_xlabel("Exit time")
        ax.set_ylabel("Balance")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plots" / "equity_curve.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        equity = trades_df["balance_after"]
        rolling_peak = equity.cummax()
        dd = (rolling_peak - equity) / rolling_peak.replace(0, np.nan) * 100.0
        ax.plot(pd.to_datetime(trades_df["exit_time"]), dd, linewidth=1.5)
        ax.set_title("Cluster 1 short drawdown")
        ax.set_xlabel("Exit time")
        ax.set_ylabel("Drawdown %")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plots" / "drawdown_curve.png", dpi=150)
        plt.close(fig)


def main() -> None:
    candles = load_candles(CANDLES_1M_PATH)
    featured = build_features(candles)
    trades_df, summary = run_backtest(featured)
    save_outputs(trades_df, summary)
    print(f"Done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()