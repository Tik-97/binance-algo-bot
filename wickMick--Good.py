from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
JSON_TRADES_PATH = BASE_DIR / Path(r"Another Trader/another_trader.json")
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "cluster1_short_climax_push_failure_rr2"


# =========================================================
# CONFIG
# =========================================================
STARTING_BALANCE = 400.0
RISK_PER_TRADE_PCT = 1.0
LEVERAGE = 10.0
ONE_TRADE_AT_A_TIME = True

FEE_PER_SIDE_PCT = 0.04
MARKET_SLIPPAGE_PER_SIDE_PCT = 0.025
LIMIT_FILL_SLIPPAGE_PCT = 0.00

# Context
LOOKBACK_4H_MINUTES = 240
BOUNCE_LOOKBACK_MINUTES = 60
NEAR_4H_HIGH_MAX_PCT = 0.35
MIN_BOUNCE_60M_PCT = 0.45
MIN_STRETCH_ABOVE_EMA20_5M_PCT = 0.15
CONTEXT_COOLDOWN_MINUTES = 240

# Trigger candle
UPPER_WICK_RATIO_MIN = 0.35
BODY_RATIO_MAX = 0.55
USE_CURRENT_OR_PREVIOUS_CANDLE = True

# Confirmation
REQUIRE_CONFIRM_CLOSE_BELOW_TRIGGER_CLOSE = True
REQUIRE_CONFIRM_LOW_BELOW_TRIGGER_LOW = True

# Climax push logic
MIN_CLIMAX_MOVE_3M_PCT = 0.45
MIN_CLIMAX_MOVE_5M_PCT = 0.70
MIN_GREEN_COUNT_LAST_5 = 3
MIN_BULL_BODY_SUM_5M_PCT = 0.35
MIN_DISTANCE_ABOVE_EMA7_1M_PCT = 0.20

# Anti-noise
MAX_CLIMAX_MOVE_5M_PCT = 2.50
MAX_TRIGGER_RANGE_PCT = 0.80
MAX_CONFIRM_RANGE_PCT = 0.80
MAX_RELATIVE_VOLUME_1M = 6.0

# -------------------------
# ENTRY LOGIC V2
# -------------------------
STRONG_SIGNAL_MIN_CLIMAX_5M_PCT = 0.95
STRONG_SIGNAL_MIN_BULL_BODY_SUM_5M_PCT = 0.45
STRONG_SIGNAL_MIN_CONFIRM_BREAK_PCT = 0.12

ENABLE_AGGRESSIVE_MARKET_ON_STRONG_SIGNAL = True
ENABLE_SHALLOW_PULLBACK_LIMIT = True

SHALLOW_PULLBACK_LIMIT_PCT_ABOVE_CONFIRM_CLOSE = 0.04
MAX_ALLOWED_PULLBACK_ABOVE_CONFIRM_CLOSE_PCT = 0.12

ENTRY_WAIT_BARS = 6
REJECT_IF_LIMIT_ABOVE_STOP = True

CANCEL_IF_RECLAIM_TRIGGER_HIGH = True
CANCEL_IF_RECLAIM_CONFIRM_CLOSE_TOO_MUCH = True

# Stop logic
USE_WIDER_STOP_OVER_CONFIRM_HIGH = True
INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT = 0.10

# -------------------------
# EXIT LOGIC: RR BASED
# -------------------------
TAKE_PARTIAL_AT_R = 2.0
PARTIAL_CLOSE_SIZE = 0.50

ENABLE_BREAKEVEN_AFTER_PARTIAL = True
BREAKEVEN_OFFSET_PCT = 0.00

TRAIL_AFTER_R = 2.0
TRAIL_LOOKBACK_BARS = 3
TRAIL_BUFFER_PCT = 0.08

ENABLE_EARLY_FAIL = False
EARLY_FAIL_WINDOW_MINUTES = 5
EARLY_FAIL_MIN_FAVORABLE_PCT = 0.08
EARLY_FAIL_MAX_ADVERSE_PCT = 0.25

MAX_BARS_SCAN = 60 * 24 * 3

# JSON matching
MATCH_WINDOW_MINUTES = 60

# Buffer around JSON trade window
JSON_PRE_BUFFER_MINUTES = max(24 * 60, LOOKBACK_4H_MINUTES + 120)
JSON_POST_BUFFER_MINUTES = 24 * 60


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class Trade:
    signal_time: pd.Timestamp
    trigger_time: pd.Timestamp
    confirm_time: pd.Timestamp
    order_placed_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    entry_mode: str
    order_price: float
    entry_price: float
    exit_price: float

    side: str
    exit_reason: str
    trigger_source: str

    trigger_upper_wick_ratio: float
    trigger_body_ratio: float
    trigger_range_pct: float
    confirm_range_pct: float

    climax_move_3m_pct: float
    climax_move_5m_pct: float
    green_count_last_5: float
    bull_body_sum_5m_pct: float
    distance_above_ema7_1m_pct: float
    rel_volume_1m: float

    preentry_move_5m_pct: float
    preentry_move_15m_pct: float

    trigger_high: float
    trigger_low: float
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

    context_dist_4h_high_pct: float
    context_bounce_60m_pct: float
    context_stretch_above_ema20_5m_pct: float


# =========================================================
# LOAD / FEATURES
# =========================================================
def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["open_time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")

    numeric_cols = ["open", "high", "low", "close"]
    if "volume" in df.columns:
        numeric_cols.append("volume")

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open_time", "open", "high", "low", "close"]).copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        df_1m.set_index("open_time")
        .resample("5min")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })
        .dropna()
        .reset_index()
    )


def build_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()

    rolling_4h_high = df["high"].rolling(
        LOOKBACK_4H_MINUTES,
        min_periods=LOOKBACK_4H_MINUTES
    ).max()
    df["dist_4h_high_pct"] = (rolling_4h_high - df["close"]) / df["close"] * 100.0

    close_60m_ago = df["close"].shift(BOUNCE_LOOKBACK_MINUTES)
    df["bounce_60m_pct"] = (df["close"] / close_60m_ago - 1.0) * 100.0

    close_3m_ago = df["close"].shift(3)
    close_5m_ago = df["close"].shift(5)
    close_15m_ago = df["close"].shift(15)

    df["move_3m_pct"] = (df["close"] / close_3m_ago - 1.0) * 100.0
    df["move_5m_pct"] = (df["close"] / close_5m_ago - 1.0) * 100.0
    df["move_15m_pct"] = (df["close"] / close_15m_ago - 1.0) * 100.0

    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        df["relative_volume_1m"] = df["volume"] / vol_ma
    else:
        df["relative_volume_1m"] = 1.0

    df["ema7_1m"] = ema(df["close"], 7)
    df["distance_above_ema7_1m_pct"] = (df["close"] / df["ema7_1m"] - 1.0) * 100.0

    df5 = resample_5m(df)
    df5["ema20_5m"] = ema(df5["close"], 20)
    df5["stretch_above_ema20_5m_pct"] = (df5["close"] / df5["ema20_5m"] - 1.0) * 100.0

    df = pd.merge_asof(
        df.sort_values("open_time"),
        df5[["open_time", "stretch_above_ema20_5m_pct"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)

    df["body_ratio"] = (body / rng).fillna(0.0)
    df["upper_wick_ratio"] = (upper_wick / rng).fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"] * 100.0).fillna(0.0)

    df["is_green"] = (df["close"] > df["open"]).astype(int)
    df["bull_body_pct"] = np.where(
        df["close"] > df["open"],
        (df["close"] - df["open"]) / df["open"] * 100.0,
        0.0,
    )
    df["green_count_last_5"] = df["is_green"].rolling(5, min_periods=5).sum()
    df["bull_body_sum_5m_pct"] = df["bull_body_pct"].rolling(5, min_periods=5).sum()

    return df


# =========================================================
# JSON / WINDOW
# =========================================================
def load_reference_trades_from_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    required = [
        "symbol", "side", "status", "opened", "closed",
        "avgCost", "avgClosePrice", "closingPnl"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JSON missing columns: {missing}")

    df = df.copy()
    df = df[
        (df["symbol"] == "ETHUSDT")
        & (df["side"] == "Short")
        & (df["status"] == "All Closed")
    ].copy()

    df["opened_time"] = pd.to_datetime(df["opened"], unit="ms", utc=True, errors="coerce")
    df["closed_time"] = pd.to_datetime(df["closed"], unit="ms", utc=True, errors="coerce")

    for col in ["avgCost", "avgClosePrice", "closingPnl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ref_result"] = np.where(
        df["closingPnl"] > 0,
        "win",
        np.where(df["closingPnl"] < 0, "loss", "flat")
    )

    df = df.dropna(subset=["opened_time", "closed_time", "avgCost", "avgClosePrice"])
    df = df.sort_values("opened_time").reset_index(drop=True)

    if df.empty:
        raise ValueError("No ETHUSDT short trades found in JSON after filtering.")

    return df


def get_json_eth_short_time_bounds(ref_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_time = ref_df["opened_time"].min()
    end_time = ref_df["closed_time"].max()
    return start_time, end_time


def trim_candles_to_json_window(
    candles: pd.DataFrame,
    json_start: pd.Timestamp,
    json_end: pd.Timestamp,
    pre_buffer_minutes: int,
    post_buffer_minutes: int,
) -> pd.DataFrame:
    start_cut = json_start - pd.Timedelta(minutes=pre_buffer_minutes)
    end_cut = json_end + pd.Timedelta(minutes=post_buffer_minutes)

    trimmed = candles[
        (candles["open_time"] >= start_cut) &
        (candles["open_time"] <= end_cut)
    ].copy()

    if trimmed.empty:
        raise ValueError(
            f"No candles remain after trimming to JSON window: "
            f"{start_cut} .. {end_cut}"
        )

    return trimmed.sort_values("open_time").reset_index(drop=True)


# =========================================================
# FILTERS
# =========================================================
def is_base_context(row: pd.Series) -> bool:
    if (
        pd.isna(row["dist_4h_high_pct"])
        or pd.isna(row["bounce_60m_pct"])
        or pd.isna(row["stretch_above_ema20_5m_pct"])
    ):
        return False

    return (
        row["dist_4h_high_pct"] <= NEAR_4H_HIGH_MAX_PCT
        and row["bounce_60m_pct"] >= MIN_BOUNCE_60M_PCT
        and row["stretch_above_ema20_5m_pct"] >= MIN_STRETCH_ABOVE_EMA20_5M_PCT
    )


def is_trigger_rejection(bar: pd.Series) -> bool:
    return (
        pd.notna(bar["upper_wick_ratio"])
        and pd.notna(bar["body_ratio"])
        and bar["upper_wick_ratio"] >= UPPER_WICK_RATIO_MIN
        and bar["body_ratio"] <= BODY_RATIO_MAX
        and bar["range_pct"] <= MAX_TRIGGER_RANGE_PCT
    )


def passes_climax_push_filter(bar: pd.Series) -> bool:
    if pd.isna(bar["move_3m_pct"]) or pd.isna(bar["move_5m_pct"]):
        return False
    if pd.isna(bar["green_count_last_5"]) or pd.isna(bar["bull_body_sum_5m_pct"]):
        return False
    if pd.isna(bar["distance_above_ema7_1m_pct"]):
        return False
    if pd.isna(bar["relative_volume_1m"]):
        return False

    return (
        bar["move_3m_pct"] >= MIN_CLIMAX_MOVE_3M_PCT
        and bar["move_5m_pct"] >= MIN_CLIMAX_MOVE_5M_PCT
        and bar["move_5m_pct"] <= MAX_CLIMAX_MOVE_5M_PCT
        and bar["green_count_last_5"] >= MIN_GREEN_COUNT_LAST_5
        and bar["bull_body_sum_5m_pct"] >= MIN_BULL_BODY_SUM_5M_PCT
        and bar["distance_above_ema7_1m_pct"] >= MIN_DISTANCE_ABOVE_EMA7_1M_PCT
        and bar["relative_volume_1m"] <= MAX_RELATIVE_VOLUME_1M
    )


def find_trigger_index(df: pd.DataFrame, i: int) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, str]] = [(i, "current")]
    if USE_CURRENT_OR_PREVIOUS_CANDLE:
        candidates.append((i - 1, "previous"))

    for trig_idx, source in candidates:
        if trig_idx < 5:
            continue
        if trig_idx + 1 >= len(df):
            continue

        trig_bar = df.iloc[trig_idx]
        confirm_bar = df.iloc[trig_idx + 1]

        if not is_trigger_rejection(trig_bar):
            continue
        if not passes_climax_push_filter(trig_bar):
            continue
        if confirm_bar["range_pct"] > MAX_CONFIRM_RANGE_PCT:
            continue
        if REQUIRE_CONFIRM_CLOSE_BELOW_TRIGGER_CLOSE and not (confirm_bar["close"] < trig_bar["close"]):
            continue
        if REQUIRE_CONFIRM_LOW_BELOW_TRIGGER_LOW and not (confirm_bar["low"] < trig_bar["low"]):
            continue

        return trig_idx, source

    return None, None


# =========================================================
# SIGNAL QUALITY
# =========================================================
def get_failure_high(trigger_bar: pd.Series, confirm_bar: pd.Series) -> float:
    if USE_WIDER_STOP_OVER_CONFIRM_HIGH:
        return float(max(trigger_bar["high"], confirm_bar["high"]))
    return float(trigger_bar["high"])


def get_confirm_break_pct(trigger_bar: pd.Series, confirm_bar: pd.Series) -> float:
    trigger_close = float(trigger_bar["close"])
    confirm_close = float(confirm_bar["close"])
    return (trigger_close - confirm_close) / trigger_close * 100.0


def is_strong_short_signal(trigger_bar: pd.Series, confirm_bar: pd.Series) -> bool:
    confirm_break_pct = get_confirm_break_pct(trigger_bar, confirm_bar)

    return (
        float(trigger_bar["move_5m_pct"]) >= STRONG_SIGNAL_MIN_CLIMAX_5M_PCT
        and float(trigger_bar["bull_body_sum_5m_pct"]) >= STRONG_SIGNAL_MIN_BULL_BODY_SUM_5M_PCT
        and confirm_break_pct >= STRONG_SIGNAL_MIN_CONFIRM_BREAK_PCT
        and float(confirm_bar["range_pct"]) <= MAX_CONFIRM_RANGE_PCT
    )


# =========================================================
# ENTRY ENGINE
# =========================================================
def simulate_short_entry(
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
    strong_signal = is_strong_short_signal(trigger_bar, confirm_bar)

    if strong_signal and ENABLE_AGGRESSIVE_MARKET_ON_STRONG_SIGNAL:
        bar = df.iloc[order_placed_idx]
        raw_entry = float(bar["open"])
        entry_price = raw_entry * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
        return order_placed_idx, order_placed_time, raw_entry, entry_price, "aggressive_market"

    if ENABLE_SHALLOW_PULLBACK_LIMIT:
        confirm_close = float(confirm_bar["close"])
        trigger_high = float(trigger_bar["high"])

        order_price = confirm_close * (1.0 + SHALLOW_PULLBACK_LIMIT_PCT_ABOVE_CONFIRM_CLOSE / 100.0)

        if order_price > confirm_close * (1.0 + MAX_ALLOWED_PULLBACK_ABOVE_CONFIRM_CLOSE_PCT / 100.0):
            return None, order_placed_time, order_price, None, "shallow_pullback_limit"

        if REJECT_IF_LIMIT_ABOVE_STOP and order_price >= stop_price_initial:
            return None, order_placed_time, order_price, None, "shallow_pullback_limit"

        last_check_idx = min(order_placed_idx + ENTRY_WAIT_BARS - 1, len(df) - 1)

        for j in range(order_placed_idx, last_check_idx + 1):
            bar = df.iloc[j]
            high = float(bar["high"])
            close = float(bar["close"])

            if CANCEL_IF_RECLAIM_TRIGGER_HIGH and high >= trigger_high:
                return None, order_placed_time, order_price, None, "shallow_pullback_limit"

            if (
                CANCEL_IF_RECLAIM_CONFIRM_CLOSE_TOO_MUCH
                and close > confirm_close * (1.0 + MAX_ALLOWED_PULLBACK_ABOVE_CONFIRM_CLOSE_PCT / 100.0)
            ):
                return None, order_placed_time, order_price, None, "shallow_pullback_limit"

            if high >= order_price:
                entry_price = order_price * (1.0 + LIMIT_FILL_SLIPPAGE_PCT / 100.0)
                return j, order_placed_time, order_price, entry_price, "shallow_pullback_limit"

        return None, order_placed_time, order_price, None, "shallow_pullback_limit"

    return None, order_placed_time, None, None, None


# =========================================================
# EXIT ENGINE
# =========================================================
def intrabar_stop_first_for_short(high: float, stop_price: float) -> bool:
    return high >= stop_price


def run_trade_exit_short(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    qty: float,
    fee_entry: float,
    balance: float,
    signal_time: pd.Timestamp,
    trigger_time: pd.Timestamp,
    confirm_time: pd.Timestamp,
    order_placed_time: pd.Timestamp,
    trigger_bar: pd.Series,
    confirm_bar: pd.Series,
    trigger_source: str,
    row_context: pd.Series,
    order_price: float,
    stop_price_initial: float,
    actual_entry_mode: str,
) -> tuple[Trade, pd.Timestamp]:
    stop_current = stop_price_initial

    best_low = entry_price
    best_high = entry_price
    entry_time = df.iloc[entry_idx]["open_time"]

    partial_taken = False
    partial_realized_pnl = 0.0
    remaining_qty = qty
    trail_active = False

    exit_reason = "dataset_end"
    fallback_idx = min(entry_idx + MAX_BARS_SCAN, len(df) - 1)
    exit_time = df.iloc[fallback_idx]["open_time"]
    exit_price = float(df.iloc[fallback_idx]["close"])

    risk_per_unit = stop_price_initial - entry_price
    if risk_per_unit <= 0:
        raise ValueError("risk_per_unit must be positive for short trades")

    target_2r_price = entry_price - TAKE_PARTIAL_AT_R * risk_per_unit

    for j in range(entry_idx + 1, fallback_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        best_low = min(best_low, low)
        best_high = max(best_high, high)

        favorable_r_now = (entry_price - best_low) / risk_per_unit

        # Partial close at 2R
        if (not partial_taken) and low <= target_2r_price:
            partial_qty = remaining_qty * PARTIAL_CLOSE_SIZE
            partial_exit_price = target_2r_price * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            partial_fee = (partial_qty * partial_exit_price) * (FEE_PER_SIDE_PCT / 100.0)

            partial_realized_pnl += (entry_price - partial_exit_price) * partial_qty - partial_fee
            remaining_qty -= partial_qty
            partial_taken = True

            if ENABLE_BREAKEVEN_AFTER_PARTIAL:
                stop_current = min(
                    stop_current,
                    entry_price * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0)
                )

        # Activate trail after 2R
        if favorable_r_now >= TRAIL_AFTER_R:
            trail_active = True

        # Trail remaining position
        if trail_active and remaining_qty > 0:
            lookback_start = max(entry_idx + 1, j - TRAIL_LOOKBACK_BARS + 1)
            recent_high = float(df.iloc[lookback_start:j + 1]["high"].max())
            candidate_stop = recent_high * (1.0 + TRAIL_BUFFER_PCT / 100.0)
            stop_current = min(stop_current, candidate_stop)

        # Stop check
        if intrabar_stop_first_for_short(high, stop_current):
            raw_exit = stop_current
            exit_price = raw_exit * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            exit_time = bar["open_time"]
            exit_reason = "managed_stop" if stop_current < stop_price_initial else "stop_loss"
            break

    gross_pnl_remaining = (entry_price - exit_price) * remaining_qty
    fee_exit_remaining = (remaining_qty * exit_price) * (FEE_PER_SIDE_PCT / 100.0)
    pnl = partial_realized_pnl + gross_pnl_remaining - fee_entry - fee_exit_remaining
    balance_after = balance + pnl

    trade = Trade(
        signal_time=signal_time,
        trigger_time=trigger_time,
        confirm_time=confirm_time,
        order_placed_time=order_placed_time,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_mode=actual_entry_mode,
        order_price=float(order_price) if pd.notna(order_price) else np.nan,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        side="Short",
        exit_reason=exit_reason,
        trigger_source=trigger_source,
        trigger_upper_wick_ratio=float(trigger_bar["upper_wick_ratio"]),
        trigger_body_ratio=float(trigger_bar["body_ratio"]),
        trigger_range_pct=float(trigger_bar["range_pct"]),
        confirm_range_pct=float(confirm_bar["range_pct"]),
        climax_move_3m_pct=float(trigger_bar["move_3m_pct"]),
        climax_move_5m_pct=float(trigger_bar["move_5m_pct"]),
        green_count_last_5=float(trigger_bar["green_count_last_5"]),
        bull_body_sum_5m_pct=float(trigger_bar["bull_body_sum_5m_pct"]),
        distance_above_ema7_1m_pct=float(trigger_bar["distance_above_ema7_1m_pct"]),
        rel_volume_1m=float(trigger_bar["relative_volume_1m"]),
        preentry_move_5m_pct=float(trigger_bar["move_5m_pct"]),
        preentry_move_15m_pct=float(trigger_bar["move_15m_pct"]),
        trigger_high=float(trigger_bar["high"]),
        trigger_low=float(trigger_bar["low"]),
        stop_price_initial=float(stop_price_initial),
        stop_price_final=float(stop_current),
        balance_before=float(balance),
        balance_after=float(balance_after),
        risk_amount_usd=float(balance * (RISK_PER_TRADE_PCT / 100.0)),
        qty=float(qty),
        notional_usd=float(qty * entry_price),
        pnl_usd=float(pnl),
        pnl_pct_balance=float((pnl / balance) * 100.0) if balance > 0 else np.nan,
        mfe_pct=float((entry_price - best_low) / entry_price * 100.0),
        mae_pct=float((best_high - entry_price) / entry_price * 100.0),
        hold_minutes=float((exit_time - entry_time).total_seconds() / 60.0),
        context_dist_4h_high_pct=float(row_context["dist_4h_high_pct"]),
        context_bounce_60m_pct=float(row_context["bounce_60m_pct"]),
        context_stretch_above_ema20_5m_pct=float(row_context["stretch_above_ema20_5m_pct"]),
    )
    return trade, exit_time


# =========================================================
# BACKTEST
# =========================================================
def run_backtest(df: pd.DataFrame, bt_start: pd.Timestamp, bt_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    trades: list[Trade] = []

    balance = STARTING_BALANCE
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")
    last_context_time = pd.Timestamp.min.tz_localize("UTC")

    context_without_trigger_count = 0
    rejected_entry_count = 0
    expired_limit_count = 0

    start_i = max(LOOKBACK_4H_MINUTES, BOUNCE_LOOKBACK_MINUTES) + 30
    i = start_i

    while i < len(df) - 5:
        row = df.iloc[i]
        now = row["open_time"]

        if now < bt_start:
            i += 1
            continue
        if now > bt_end:
            break

        if ONE_TRADE_AT_A_TIME and now < last_exit_time:
            i += 1
            continue

        if now < last_context_time + pd.Timedelta(minutes=CONTEXT_COOLDOWN_MINUTES):
            i += 1
            continue

        if not is_base_context(row):
            i += 1
            continue

        trig_idx, trig_source = find_trigger_index(df, i)
        if trig_idx is None:
            context_without_trigger_count += 1
            i += 1
            continue

        confirm_idx = trig_idx + 1
        if confirm_idx >= len(df):
            break

        trigger_bar = df.iloc[trig_idx]
        confirm_bar = df.iloc[confirm_idx]

        signal_time = row["open_time"]
        trigger_time = trigger_bar["open_time"]
        confirm_time = confirm_bar["open_time"]

        if signal_time < bt_start or signal_time > bt_end:
            i += 1
            continue

        failure_high = get_failure_high(trigger_bar, confirm_bar)
        stop_price_initial = failure_high * (
            1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0
        )

        fill_idx, order_placed_time, order_price, entry_price, actual_entry_mode = simulate_short_entry(
            df=df,
            trigger_idx=trig_idx,
            confirm_idx=confirm_idx,
            trigger_bar=trigger_bar,
            confirm_bar=confirm_bar,
            stop_price_initial=stop_price_initial,
        )

        if fill_idx is None or entry_price is None or order_placed_time is None:
            if actual_entry_mode == "aggressive_market":
                rejected_entry_count += 1
            else:
                expired_limit_count += 1
            i = confirm_idx + ENTRY_WAIT_BARS + 1
            continue

        entry_time = df.iloc[fill_idx]["open_time"]
        if entry_time < bt_start or entry_time > bt_end:
            i = fill_idx + 1
            continue

        stop_distance = stop_price_initial - entry_price
        if stop_distance <= 0:
            rejected_entry_count += 1
            i = fill_idx + 1
            continue

        risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
        qty = risk_amount / stop_distance

        max_notional = balance * LEVERAGE
        desired_notional = qty * entry_price
        if desired_notional > max_notional:
            qty = max_notional / entry_price

        if qty <= 0:
            rejected_entry_count += 1
            i = fill_idx + 1
            continue

        notional = qty * entry_price
        fee_entry = notional * (FEE_PER_SIDE_PCT / 100.0)

        trade, exit_time = run_trade_exit_short(
            df=df,
            entry_idx=fill_idx,
            entry_price=entry_price,
            qty=qty,
            fee_entry=fee_entry,
            balance=balance,
            signal_time=signal_time,
            trigger_time=trigger_time,
            confirm_time=confirm_time,
            order_placed_time=order_placed_time,
            trigger_bar=trigger_bar,
            confirm_bar=confirm_bar,
            trigger_source=trig_source,
            row_context=row,
            order_price=order_price if order_price is not None else np.nan,
            stop_price_initial=stop_price_initial,
            actual_entry_mode=actual_entry_mode if actual_entry_mode is not None else "unknown",
        )

        trades.append(trade)
        balance = trade.balance_after
        last_exit_time = exit_time
        last_context_time = signal_time

        i = fill_idx + 1

    trades_df = pd.DataFrame([asdict(t) for t in trades])

    if trades_df.empty:
        return trades_df, {
            "trade_count": 0,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_pct": np.nan,
            "expectancy_usd": np.nan,
            "ending_balance": STARTING_BALANCE,
            "context_without_trigger_count": context_without_trigger_count,
            "rejected_entry_count": rejected_entry_count,
            "expired_limit_count": expired_limit_count,
            "bt_start": bt_start,
            "bt_end": bt_end,
        }

    equity = trades_df["balance_after"]
    rolling_peak = equity.cummax()
    dd = (rolling_peak - equity) / rolling_peak.replace(0, np.nan) * 100.0

    wins = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losses = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]

    return trades_df, {
        "trade_count": int(len(trades_df)),
        "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
        "max_drawdown_pct": float(dd.max()) if len(dd) else np.nan,
        "expectancy_usd": float(trades_df["pnl_usd"].mean()),
        "ending_balance": float(trades_df["balance_after"].iloc[-1]),
        "context_without_trigger_count": context_without_trigger_count,
        "rejected_entry_count": rejected_entry_count,
        "expired_limit_count": expired_limit_count,
        "bt_start": bt_start,
        "bt_end": bt_end,
    }


# =========================================================
# MATCHING
# =========================================================
def match_backtest_to_reference(
    trades_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    max_minutes: int = MATCH_WINDOW_MINUTES,
) -> pd.DataFrame:
    if trades_df.empty or ref_df.empty:
        return pd.DataFrame()

    bt = trades_df.copy()
    bt["entry_time"] = pd.to_datetime(bt["entry_time"], utc=True, errors="coerce")
    bt["bt_result"] = np.where(
        bt["pnl_usd"] > 0,
        "win",
        np.where(bt["pnl_usd"] < 0, "loss", "flat")
    )

    rows: list[dict] = []

    for _, bt_row in bt.iterrows():
        diffs = (ref_df["opened_time"] - bt_row["entry_time"]).abs()
        nearest_idx = diffs.idxmin()
        nearest = ref_df.loc[nearest_idx]

        diff_minutes = diffs.loc[nearest_idx].total_seconds() / 60.0

        rows.append({
            "bt_entry_time": bt_row["entry_time"],
            "bt_order_placed_time": bt_row["order_placed_time"],
            "bt_entry_mode": bt_row["entry_mode"],
            "bt_entry_price": bt_row["entry_price"],
            "bt_exit_time": bt_row["exit_time"],
            "bt_pnl_usd": bt_row["pnl_usd"],
            "bt_result": bt_row["bt_result"],

            "ref_opened_time": nearest["opened_time"],
            "ref_closed_time": nearest["closed_time"],
            "ref_avgCost": nearest["avgCost"],
            "ref_avgClosePrice": nearest["avgClosePrice"],
            "ref_closingPnl": nearest["closingPnl"],
            "ref_result": nearest["ref_result"],

            "time_diff_minutes": diff_minutes,
            "matched_within_window": diff_minutes <= max_minutes,
            "same_result_direction": bt_row["bt_result"] == nearest["ref_result"],
        })

    matched_df = pd.DataFrame(rows)
    return matched_df.sort_values("bt_entry_time").reset_index(drop=True)


# =========================================================
# OUTPUT
# =========================================================
def save_outputs(trades_df: pd.DataFrame, summary: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(OUTPUT_DIR / "cluster1_short_trades.csv", index=False)

    lines = [
        "CLUSTER 1 SHORT CLIMAX PUSH FAILURE BACKTEST",
        "============================================",
        "",
        "Idea: short after abnormal upside burst, then failure, with adaptive entry style and RR-based exits.",
        "",
        f"JSON_TRADES_PATH = {JSON_TRADES_PATH}",
        f"ENTRY_WAIT_BARS = {ENTRY_WAIT_BARS}",
        "",
        f"BT_START = {summary['bt_start']}",
        f"BT_END = {summary['bt_end']}",
        "",
        f"TAKE_PARTIAL_AT_R = {TAKE_PARTIAL_AT_R}",
        f"PARTIAL_CLOSE_SIZE = {PARTIAL_CLOSE_SIZE}",
        f"TRAIL_AFTER_R = {TRAIL_AFTER_R}",
        f"TRAIL_LOOKBACK_BARS = {TRAIL_LOOKBACK_BARS}",
        "",
        "Context:",
        f"- near 4h high <= {NEAR_4H_HIGH_MAX_PCT:.2f}%",
        f"- 60m bounce >= {MIN_BOUNCE_60M_PCT:.2f}%",
        f"- stretch above EMA20 5m >= {MIN_STRETCH_ABOVE_EMA20_5M_PCT:.2f}%",
        "",
        "Climax push conditions:",
        f"- move_3m >= {MIN_CLIMAX_MOVE_3M_PCT:.2f}%",
        f"- move_5m >= {MIN_CLIMAX_MOVE_5M_PCT:.2f}%",
        f"- move_5m <= {MAX_CLIMAX_MOVE_5M_PCT:.2f}%",
        f"- green_count_last_5 >= {MIN_GREEN_COUNT_LAST_5}",
        f"- bull_body_sum_5m >= {MIN_BULL_BODY_SUM_5M_PCT:.2f}%",
        "",
        f"Trade count: {summary['trade_count']}",
        f"Win rate: {summary['win_rate_pct']:.2f}%" if not pd.isna(summary["win_rate_pct"]) else "Win rate: n/a",
        f"Profit factor: {summary['profit_factor']:.3f}" if not pd.isna(summary["profit_factor"]) else "Profit factor: n/a",
        f"Max drawdown: {summary['max_drawdown_pct']:.2f}%" if not pd.isna(summary["max_drawdown_pct"]) else "Max drawdown: n/a",
        f"Expectancy: {summary['expectancy_usd']:.2f} USD" if not pd.isna(summary["expectancy_usd"]) else "Expectancy: n/a",
        f"Ending balance: {summary['ending_balance']:.2f} USD",
        f"Context without trigger count: {summary['context_without_trigger_count']}",
        f"Rejected entry count: {summary['rejected_entry_count']}",
        f"Expired limit count: {summary['expired_limit_count']}",
        "",
    ]
    (OUTPUT_DIR / "cluster1_short_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if not trades_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(pd.to_datetime(trades_df["exit_time"]), trades_df["balance_after"], linewidth=1.5)
        ax.set_title("Cluster 1 short climax-push-failure equity curve")
        ax.set_xlabel("Exit time")
        ax.set_ylabel("Balance")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "plots" / "equity_curve.png", dpi=150)
        plt.close(fig)


def save_match_outputs(matched_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if matched_df.empty:
        (OUTPUT_DIR / "json_match_summary.txt").write_text(
            "No matches available.\n",
            encoding="utf-8",
        )
        return

    matched_df.to_csv(OUTPUT_DIR / "json_trade_matches.csv", index=False)

    within_window = matched_df["matched_within_window"].sum()
    same_result = matched_df.loc[matched_df["matched_within_window"], "same_result_direction"].sum()
    total = len(matched_df)

    lines = [
        "BACKTEST vs JSON TRADE MATCH SUMMARY",
        "===================================",
        "",
        f"Total backtest trades: {total}",
        f"Matches within {MATCH_WINDOW_MINUTES} min: {within_window}",
        f"Same result direction within matched window: {same_result}",
    ]
    if within_window > 0:
        lines.append(f"Result-direction agreement rate: {same_result / within_window * 100.0:.2f}%")

    (OUTPUT_DIR / "json_match_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    if not CANDLES_1M_PATH.exists():
        raise FileNotFoundError(f"Candles file not found: {CANDLES_1M_PATH}")

    if not JSON_TRADES_PATH.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_TRADES_PATH}")

    candles = load_candles(CANDLES_1M_PATH)
    ref_df = load_reference_trades_from_json(JSON_TRADES_PATH)

    bt_start, bt_end = get_json_eth_short_time_bounds(ref_df)

    candles = trim_candles_to_json_window(
        candles=candles,
        json_start=bt_start,
        json_end=bt_end,
        pre_buffer_minutes=JSON_PRE_BUFFER_MINUTES,
        post_buffer_minutes=JSON_POST_BUFFER_MINUTES,
    )

    print(f"JSON ETH short window: {bt_start} -> {bt_end}")
    print(f"Trimmed candle window: {candles['open_time'].min()} -> {candles['open_time'].max()}")

    featured = build_features(candles)
    trades_df, summary = run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    save_outputs(trades_df, summary)

    matched_df = match_backtest_to_reference(
        trades_df=trades_df,
        ref_df=ref_df,
        max_minutes=MATCH_WINDOW_MINUTES,
    )
    save_match_outputs(matched_df)

    if not trades_df.empty:
        print(f"Backtest trade window: {trades_df['entry_time'].min()} -> {trades_df['entry_time'].max()}")
    else:
        print("Backtest produced no trades.")

    print(f"Done. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()