from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
JSON_TRADES_PATH = BASE_DIR / Path(r"Another Trader/another_trader.json")
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_trader_style_session_v1"


# =========================================================
# CORE IDEA
# =========================================================
# Most reasonable current hypothesis:
# - ETH shorts only
# - simple reversal-like proxy, not overfitted trigger
# - bounce context kept
# - strict UTC session filter around trader cluster
# - mixed execution
# - fast fail if trade doesn't work quickly


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

# JSON window padding
JSON_PRE_BUFFER_MINUTES = 24 * 60
JSON_POST_BUFFER_MINUTES = 24 * 60

# Session filter: most reasonable current test
SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16   # inclusive of 13,14,15; exclusive of 16

# Context: keep it moderate, not too strict
LOOKBACK_4H_MINUTES = 240
LOOKBACK_1H_MINUTES = 60
BOUNCE_LOOKBACK_MINUTES = 60
NEAR_4H_HIGH_MAX_PCT = 0.60
NEAR_1H_HIGH_MAX_PCT = 0.45
MIN_BOUNCE_60M_PCT = 0.25
MAX_BOUNCE_60M_PCT = 2.30
MIN_STRETCH_ABOVE_EMA20_5M_PCT = 0.10

# Base proxy trigger
MIN_CLIMAX_MOVE_3M_PCT = 0.35
MIN_CLIMAX_MOVE_5M_PCT = 0.60
MAX_MOVE_1M_PCT = 0.35
MAX_MOVE_3M_PCT = 1.15
MAX_MOVE_5M_PCT = 1.65
MIN_GREEN_COUNT_LAST_5 = 3
MIN_BULL_BODY_SUM_5M_PCT = 0.28
MIN_DISTANCE_ABOVE_EMA7_1M_PCT = 0.10
MAX_RELATIVE_VOLUME_1M = 2.50

UPPER_WICK_RATIO_MIN = 0.18
BODY_RATIO_MAX = 0.75
MAX_TRIGGER_RANGE_PCT = 1.20
MAX_CONFIRM_RANGE_PCT = 1.20

REQUIRE_CONFIRM_CLOSE_BELOW_TRIGGER_CLOSE = True
REQUIRE_CONFIRM_LOW_BELOW_TRIGGER_LOW = True

# Execution mix
# Approx mix from earlier research: immediate-like around 39%
IMMEDIATE_MODE_BUCKET_PCT = 39
DELAYED_LIMIT_WAIT_BARS = 6
LIMIT_PULLBACK_PCT_ABOVE_CONFIRM_CLOSE = 0.05
LIMIT_CAN_FILL_SAME_BAR = False
REJECT_IF_LIMIT_ABOVE_STOP = True

# Stop / fast fail
INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT = 0.10
FAST_FAIL_WINDOW_BARS = 8
FAST_FAIL_MIN_FAVORABLE_PCT = 0.05
FAST_FAIL_MAX_ADVERSE_PCT = 0.45

# RR exit
TAKE_PARTIAL_AT_R = 2.0
PARTIAL_CLOSE_SIZE = 0.50
ENABLE_BREAKEVEN_AFTER_PARTIAL = True
BREAKEVEN_OFFSET_PCT = 0.00
TRAIL_AFTER_R = 2.0
TRAIL_LOOKBACK_BARS = 3
TRAIL_BUFFER_PCT = 0.08

MAX_BARS_SCAN = 60 * 24 * 3

# Local timing rerank derived from timing-score-v3 research.
LOCAL_TIMING_RERANK_ENABLED = True
LOCAL_TIMING_LOOKBACK_BARS = 10
LOCAL_TIMING_MAX_RANK = 3

# Research bridge: choose the best candidate from a small local trigger window
# instead of taking the first valid trigger. This improves imitation testing
# but still uses slight lookahead, so it is not yet live-ready logic.
CANDIDATE_RANKING_LAYER_ENABLED = True
CANDIDATE_RANKING_LOOKAHEAD_BARS = 5

LOCAL_TIMING_TARGET_DELTAS = {
    "move_1m_pct": 0.3092,
    "move_3m_pct": 0.6735,
    "move_5m_pct": 0.7186,
    "relative_volume_1m": 0.7294,
    "upper_wick_ratio": 0.1771,
    "body_ratio": 0.1320,
    "close_position_in_range": 0.0563,
    "range_pct": 0.5256,
    "dist_1h_high_pct": -0.1473,
    "dist_5m_high_pct": -0.2171,
    "stretch_above_ema20_5m_pct": 0.3606,
}

LOCAL_TIMING_DELTA_SCALES = {
    "move_1m_pct": 1.3136,
    "move_3m_pct": 1.8656,
    "move_5m_pct": 2.1932,
    "relative_volume_1m": 1.4532,
    "upper_wick_ratio": 0.9925,
    "body_ratio": 0.9042,
    "close_position_in_range": 1.0465,
    "range_pct": 1.3641,
    "dist_1h_high_pct": 1.3352,
    "dist_5m_high_pct": 0.8480,
    "stretch_above_ema20_5m_pct": 1.5558,
}

LOCAL_TIMING_WEIGHTS = {
    "move_1m_pct": 0.80,
    "move_3m_pct": 1.00,
    "move_5m_pct": 0.90,
    "relative_volume_1m": 0.80,
    "upper_wick_ratio": 1.10,
    "body_ratio": 1.00,
    "close_position_in_range": 1.10,
    "range_pct": 0.70,
    "dist_1h_high_pct": 1.20,
    "dist_5m_high_pct": 1.10,
    "stretch_above_ema20_5m_pct": 0.80,
}


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

    trigger_upper_wick_ratio: float
    trigger_body_ratio: float
    trigger_range_pct: float
    confirm_range_pct: float

    move_3m_pct: float
    move_5m_pct: float
    green_count_last_5: float
    bull_body_sum_5m_pct: float
    distance_above_ema7_1m_pct: float
    rel_volume_1m: float

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
# LOADERS
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


def load_reference_eth_shorts(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    required = ["symbol", "side", "status", "opened", "closed", "avgCost", "avgClosePrice", "closingPnl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JSON missing columns: {missing}")

    df = df[
        (df["symbol"] == "ETHUSDT")
        & (df["side"] == "Short")
        & (df["status"] == "All Closed")
    ].copy()

    df["opened_time"] = pd.to_datetime(df["opened"], unit="ms", utc=True, errors="coerce")
    df["closed_time"] = pd.to_datetime(df["closed"], unit="ms", utc=True, errors="coerce")

    for c in ["avgCost", "avgClosePrice", "closingPnl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ref_result"] = np.where(
        df["closingPnl"] > 0,
        "win",
        np.where(df["closingPnl"] < 0, "loss", "flat"),
    )

    df = df.dropna(subset=["opened_time", "closed_time", "avgCost", "avgClosePrice"]).copy()
    df = df.sort_values("opened_time").reset_index(drop=True)

    if df.empty:
        raise ValueError("No ETHUSDT short trades found in JSON.")

    return df


def get_reference_window(ref_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return ref_df["opened_time"].min(), ref_df["closed_time"].max()


def trim_candles_to_window(
    candles: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    start_cut = start_time - pd.Timedelta(minutes=JSON_PRE_BUFFER_MINUTES)
    end_cut = end_time + pd.Timedelta(minutes=JSON_POST_BUFFER_MINUTES)

    out = candles[
        (candles["open_time"] >= start_cut)
        & (candles["open_time"] <= end_cut)
    ].copy()

    if out.empty:
        raise ValueError("No candles left after trimming to reference window.")

    return out.sort_values("open_time").reset_index(drop=True)


# =========================================================
# FEATURES
# =========================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        df_1m.set_index("open_time")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )


def build_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()

    rolling_4h_high = df["high"].rolling(LOOKBACK_4H_MINUTES, min_periods=LOOKBACK_4H_MINUTES).max()
    df["dist_4h_high_pct"] = (rolling_4h_high - df["close"]) / df["close"] * 100.0

    rolling_5m_high_prev = df["high"].shift(1).rolling(5, min_periods=5).max()
    rolling_1h_high_prev = df["high"].shift(1).rolling(LOOKBACK_1H_MINUTES, min_periods=LOOKBACK_1H_MINUTES).max()
    df["dist_5m_high_pct"] = (rolling_5m_high_prev - df["close"]) / df["close"] * 100.0
    df["dist_1h_high_pct"] = (rolling_1h_high_prev - df["close"]) / df["close"] * 100.0

    close_1m_ago = df["close"].shift(1)
    close_60m_ago = df["close"].shift(BOUNCE_LOOKBACK_MINUTES)
    df["move_1m_pct"] = (df["close"] / close_1m_ago - 1.0) * 100.0
    df["bounce_60m_pct"] = (df["close"] / close_60m_ago - 1.0) * 100.0

    close_3m_ago = df["close"].shift(3)
    close_5m_ago = df["close"].shift(5)

    df["move_3m_pct"] = (df["close"] / close_3m_ago - 1.0) * 100.0
    df["move_5m_pct"] = (df["close"] / close_5m_ago - 1.0) * 100.0

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
    df["close_position_in_range"] = ((df["close"] - df["low"]) / rng).fillna(0.5)
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
# SIGNAL MODEL
# =========================================================
def in_session(ts: pd.Timestamp) -> bool:
    hour = ts.hour
    return SESSION_START_HOUR_UTC <= hour < SESSION_END_HOUR_UTC


def is_context_ok(row: pd.Series) -> bool:
    needed = [
        "dist_4h_high_pct",
        "dist_1h_high_pct",
        "bounce_60m_pct",
        "move_1m_pct",
        "stretch_above_ema20_5m_pct",
        "move_3m_pct",
        "move_5m_pct",
        "green_count_last_5",
        "bull_body_sum_5m_pct",
        "distance_above_ema7_1m_pct",
        "relative_volume_1m",
    ]
    if any(pd.isna(row[c]) for c in needed):
        return False

    return (
        row["dist_4h_high_pct"] <= NEAR_4H_HIGH_MAX_PCT
        and row["dist_1h_high_pct"] <= NEAR_1H_HIGH_MAX_PCT
        and row["bounce_60m_pct"] >= MIN_BOUNCE_60M_PCT
        and row["bounce_60m_pct"] <= MAX_BOUNCE_60M_PCT
        and row["move_1m_pct"] <= MAX_MOVE_1M_PCT
        and row["stretch_above_ema20_5m_pct"] >= MIN_STRETCH_ABOVE_EMA20_5M_PCT
        and row["move_3m_pct"] >= MIN_CLIMAX_MOVE_3M_PCT
        and row["move_3m_pct"] <= MAX_MOVE_3M_PCT
        and row["move_5m_pct"] >= MIN_CLIMAX_MOVE_5M_PCT
        and row["move_5m_pct"] <= MAX_MOVE_5M_PCT
        and row["green_count_last_5"] >= MIN_GREEN_COUNT_LAST_5
        and row["bull_body_sum_5m_pct"] >= MIN_BULL_BODY_SUM_5M_PCT
        and row["distance_above_ema7_1m_pct"] >= MIN_DISTANCE_ABOVE_EMA7_1M_PCT
        and row["relative_volume_1m"] <= MAX_RELATIVE_VOLUME_1M
    )


def is_trigger_bar(bar: pd.Series) -> bool:
    return (
        pd.notna(bar["upper_wick_ratio"])
        and pd.notna(bar["body_ratio"])
        and bar["upper_wick_ratio"] >= UPPER_WICK_RATIO_MIN
        and bar["body_ratio"] <= BODY_RATIO_MAX
        and bar["range_pct"] <= MAX_TRIGGER_RANGE_PCT
    )


def compute_local_timing_scores(window: pd.DataFrame) -> pd.Series:
    if window.empty:
        return pd.Series(dtype=float)

    weighted_sum = pd.Series(0.0, index=window.index, dtype=float)
    total_weight = 0.0

    for feature, target in LOCAL_TIMING_TARGET_DELTAS.items():
        if feature not in window.columns:
            continue

        series = pd.to_numeric(window[feature], errors="coerce")
        local_median = float(series.median())
        local_scale = float((series.quantile(0.75) - series.quantile(0.25)))
        if not np.isfinite(local_scale) or local_scale <= 0:
            local_scale = float(series.std()) if np.isfinite(series.std()) and series.std() > 0 else 1.0

        current_delta = (series - local_median) / local_scale
        target_scale = float(LOCAL_TIMING_DELTA_SCALES.get(feature, 1.0))
        if target_scale <= 0 or not np.isfinite(target_scale):
            target_scale = 1.0

        deviation = (current_delta - float(target)).abs() / target_scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped

        weight = float(LOCAL_TIMING_WEIGHTS.get(feature, 1.0))
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.nan, index=window.index)

    return weighted_sum / total_weight


def passes_local_timing_rerank(df: pd.DataFrame, trig_idx: int) -> bool:
    if not LOCAL_TIMING_RERANK_ENABLED:
        return True

    start_idx = max(0, trig_idx - LOCAL_TIMING_LOOKBACK_BARS + 1)
    window = df.iloc[start_idx : trig_idx + 1].copy()
    if window.empty:
        return True

    candidate_mask = window.apply(lambda row: is_context_ok(row) and is_trigger_bar(row), axis=1)
    candidates = window.loc[candidate_mask].copy()
    if candidates.empty:
        return True

    scores = compute_local_timing_scores(candidates)
    candidates["local_timing_score"] = scores
    candidates = candidates.sort_values(["local_timing_score", "open_time"], ascending=[False, True]).reset_index(drop=False)

    target_time = df.iloc[trig_idx]["open_time"]
    matched = candidates.loc[candidates["open_time"].eq(target_time)]
    if matched.empty:
        return False

    rank = int(matched.index[0]) + 1
    return rank <= LOCAL_TIMING_MAX_RANK


def is_trigger_candidate(df: pd.DataFrame, trig_idx: int) -> bool:
    if trig_idx < 5 or trig_idx + 1 >= len(df):
        return False

    trig = df.iloc[trig_idx]
    conf = df.iloc[trig_idx + 1]

    if not in_session(trig["open_time"]):
        return False
    if not is_context_ok(trig):
        return False
    if not is_trigger_bar(trig):
        return False
    if not passes_local_timing_rerank(df, trig_idx):
        return False
    if conf["range_pct"] > MAX_CONFIRM_RANGE_PCT:
        return False
    if REQUIRE_CONFIRM_CLOSE_BELOW_TRIGGER_CLOSE and not (conf["close"] < trig["close"]):
        return False
    if REQUIRE_CONFIRM_LOW_BELOW_TRIGGER_LOW and not (conf["low"] < trig["low"]):
        return False
    return True


def compute_candidate_rank_score(df: pd.DataFrame, trig_idx: int) -> float:
    trig = df.iloc[trig_idx]
    conf = df.iloc[trig_idx + 1]
    local_window_start = max(0, trig_idx - LOCAL_TIMING_LOOKBACK_BARS + 1)
    local_window = df.iloc[local_window_start : trig_idx + 1].copy()
    local_window = local_window.loc[local_window.apply(lambda row: is_context_ok(row) and is_trigger_bar(row), axis=1)].copy()

    local_timing_score = 0.0
    if not local_window.empty:
        local_scores = compute_local_timing_scores(local_window)
        local_window["local_timing_score"] = local_scores
        matched = local_window.loc[local_window["open_time"].eq(trig["open_time"]), "local_timing_score"]
        if not matched.empty:
            local_timing_score = float(matched.iloc[0])

    score = compute_signal_strength(trig, conf)
    score += local_timing_score * 3.0
    score += max(0.0, 0.5 - float(trig["dist_1h_high_pct"])) * 2.0
    score += max(0.0, 0.35 - float(trig["dist_5m_high_pct"])) * 1.5
    score += max(0.0, 0.6 - float(trig["close_position_in_range"])) * 1.0
    score += min(float(trig["upper_wick_ratio"]) / 0.4, 1.5)
    return float(score)


def select_best_candidate_trigger(df: pd.DataFrame, i: int) -> int | None:
    candidate_indices: list[int] = []

    if not CANDIDATE_RANKING_LAYER_ENABLED:
        for trig_idx in [i, i - 1]:
            if is_trigger_candidate(df, trig_idx):
                return trig_idx
        return None

    start_idx = max(5, i - 1)
    end_idx = min(len(df) - 2, i + CANDIDATE_RANKING_LOOKAHEAD_BARS)
    for trig_idx in range(start_idx, end_idx + 1):
        if is_trigger_candidate(df, trig_idx):
            candidate_indices.append(trig_idx)

    if not candidate_indices:
        return None

    scored = [
        (compute_candidate_rank_score(df, trig_idx), trig_idx)
        for trig_idx in candidate_indices
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return int(scored[0][1])


def find_trigger(df: pd.DataFrame, i: int) -> int | None:
    # Simple, moderate base proxy + bounce context
    for trig_idx in [i, i - 1]:
        if is_trigger_candidate(df, trig_idx):
            return trig_idx

    return None


def compute_signal_strength(trigger_bar: pd.Series, confirm_bar: pd.Series) -> float:
    score = 0.0
    score += min(float(trigger_bar["move_5m_pct"]) / 1.0, 2.0)
    score += min(float(trigger_bar["upper_wick_ratio"]) / 0.5, 2.0)
    score += min(float(trigger_bar["bull_body_sum_5m_pct"]) / 0.8, 2.0)
    score += max(0.0, (float(trigger_bar["close"]) - float(confirm_bar["close"])) / float(trigger_bar["close"]) * 100.0 / 0.15)
    return score


def choose_entry_mode(signal_time: pd.Timestamp, strength: float) -> str:
    bucket = int(signal_time.value // 10**9) % 100

    if strength >= 4.2:
        return "aggressive_market"
    if bucket < IMMEDIATE_MODE_BUCKET_PCT:
        return "aggressive_market"
    return "delayed_limit_like"


# =========================================================
# EXECUTION
# =========================================================
def simulate_short_entry(
    df: pd.DataFrame,
    trigger_idx: int,
    confirm_idx: int,
    stop_price_initial: float,
    entry_mode: str,
) -> tuple[int | None, pd.Timestamp | None, float | None, float | None]:
    order_placed_idx = confirm_idx + 1
    if order_placed_idx >= len(df):
        return None, None, None, None

    order_placed_time = df.iloc[order_placed_idx]["open_time"]
    trigger_bar = df.iloc[trigger_idx]
    confirm_bar = df.iloc[confirm_idx]

    if entry_mode == "aggressive_market":
        bar = df.iloc[order_placed_idx]
        raw_entry = float(bar["open"])
        entry_price = raw_entry * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
        return order_placed_idx, order_placed_time, raw_entry, entry_price

    order_price = float(confirm_bar["close"]) * (1.0 + LIMIT_PULLBACK_PCT_ABOVE_CONFIRM_CLOSE / 100.0)
    if REJECT_IF_LIMIT_ABOVE_STOP and order_price >= stop_price_initial:
        return None, order_placed_time, order_price, None

    last_check_idx = min(order_placed_idx + DELAYED_LIMIT_WAIT_BARS - 1, len(df) - 1)
    fill_start_idx = order_placed_idx if LIMIT_CAN_FILL_SAME_BAR else order_placed_idx + 1

    if fill_start_idx > last_check_idx:
        return None, order_placed_time, order_price, None

    for j in range(fill_start_idx, last_check_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        close = float(bar["close"])

        if high >= float(trigger_bar["high"]):
            return None, order_placed_time, order_price, None

        if close > float(confirm_bar["close"]) * 1.0012:
            return None, order_placed_time, order_price, None

        if high >= order_price:
            entry_price = order_price * (1.0 + LIMIT_FILL_SLIPPAGE_PCT / 100.0)
            return j, order_placed_time, order_price, entry_price

    return None, order_placed_time, order_price, None


# =========================================================
# EXIT
# =========================================================
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
    entry_mode: str,
    row_context: pd.Series,
    order_price: float,
    stop_price_initial: float,
) -> tuple[Trade, pd.Timestamp]:
    stop_current = stop_price_initial

    best_low = entry_price
    best_high = entry_price
    entry_time = df.iloc[entry_idx]["open_time"]

    # Fast fail
    ff_end = min(entry_idx + FAST_FAIL_WINDOW_BARS, len(df) - 1)
    ff_slice = df.iloc[entry_idx + 1 : ff_end + 1]
    if not ff_slice.empty:
        ff_low = float(ff_slice["low"].min())
        ff_high = float(ff_slice["high"].max())
        favorable_pct = (entry_price - ff_low) / entry_price * 100.0
        adverse_pct = (ff_high - entry_price) / entry_price * 100.0

        if favorable_pct < FAST_FAIL_MIN_FAVORABLE_PCT and adverse_pct >= FAST_FAIL_MAX_ADVERSE_PCT:
            exit_price = entry_price * (1.0 + FAST_FAIL_MAX_ADVERSE_PCT / 100.0)
            exit_price *= (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            exit_time = ff_slice.iloc[-1]["open_time"]
            exit_reason = "fast_fail"

            gross_pnl = (entry_price - exit_price) * qty
            fee_exit = (qty * exit_price) * (FEE_PER_SIDE_PCT / 100.0)
            pnl = gross_pnl - fee_entry - fee_exit
            balance_after = balance + pnl

            trade = Trade(
                signal_time=signal_time,
                trigger_time=trigger_time,
                confirm_time=confirm_time,
                order_placed_time=order_placed_time,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_mode=entry_mode,
                order_price=float(order_price) if pd.notna(order_price) else np.nan,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                side="Short",
                exit_reason=exit_reason,
                trigger_upper_wick_ratio=float(trigger_bar["upper_wick_ratio"]),
                trigger_body_ratio=float(trigger_bar["body_ratio"]),
                trigger_range_pct=float(trigger_bar["range_pct"]),
                confirm_range_pct=float(confirm_bar["range_pct"]),
                move_3m_pct=float(trigger_bar["move_3m_pct"]),
                move_5m_pct=float(trigger_bar["move_5m_pct"]),
                green_count_last_5=float(trigger_bar["green_count_last_5"]),
                bull_body_sum_5m_pct=float(trigger_bar["bull_body_sum_5m_pct"]),
                distance_above_ema7_1m_pct=float(trigger_bar["distance_above_ema7_1m_pct"]),
                rel_volume_1m=float(trigger_bar["relative_volume_1m"]),
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
                mfe_pct=float(favorable_pct),
                mae_pct=float(adverse_pct),
                hold_minutes=float((exit_time - entry_time).total_seconds() / 60.0),
                context_dist_4h_high_pct=float(row_context["dist_4h_high_pct"]),
                context_bounce_60m_pct=float(row_context["bounce_60m_pct"]),
                context_stretch_above_ema20_5m_pct=float(row_context["stretch_above_ema20_5m_pct"]),
            )
            return trade, exit_time

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
        raise ValueError("risk_per_unit must be positive")

    target_2r_price = entry_price - TAKE_PARTIAL_AT_R * risk_per_unit

    for j in range(entry_idx + 1, fallback_idx + 1):
        bar = df.iloc[j]
        high = float(bar["high"])
        low = float(bar["low"])

        best_low = min(best_low, low)
        best_high = max(best_high, high)

        favorable_r_now = (entry_price - best_low) / risk_per_unit

        if (not partial_taken) and low <= target_2r_price:
            partial_qty = remaining_qty * PARTIAL_CLOSE_SIZE
            partial_exit_price = target_2r_price * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
            partial_fee = (partial_qty * partial_exit_price) * (FEE_PER_SIDE_PCT / 100.0)

            partial_realized_pnl += (entry_price - partial_exit_price) * partial_qty - partial_fee
            remaining_qty -= partial_qty
            partial_taken = True

            if ENABLE_BREAKEVEN_AFTER_PARTIAL:
                stop_current = min(stop_current, entry_price * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0))

        if favorable_r_now >= TRAIL_AFTER_R:
            trail_active = True

        if trail_active and remaining_qty > 0:
            lookback_start = max(entry_idx + 1, j - TRAIL_LOOKBACK_BARS + 1)
            recent_high = float(df.iloc[lookback_start:j + 1]["high"].max())
            candidate_stop = recent_high * (1.0 + TRAIL_BUFFER_PCT / 100.0)
            stop_current = min(stop_current, candidate_stop)

        if high >= stop_current:
            exit_price = stop_current * (1.0 + MARKET_SLIPPAGE_PER_SIDE_PCT / 100.0)
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
        entry_mode=entry_mode,
        order_price=float(order_price) if pd.notna(order_price) else np.nan,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        side="Short",
        exit_reason=exit_reason,
        trigger_upper_wick_ratio=float(trigger_bar["upper_wick_ratio"]),
        trigger_body_ratio=float(trigger_bar["body_ratio"]),
        trigger_range_pct=float(trigger_bar["range_pct"]),
        confirm_range_pct=float(confirm_bar["range_pct"]),
        move_3m_pct=float(trigger_bar["move_3m_pct"]),
        move_5m_pct=float(trigger_bar["move_5m_pct"]),
        green_count_last_5=float(trigger_bar["green_count_last_5"]),
        bull_body_sum_5m_pct=float(trigger_bar["bull_body_sum_5m_pct"]),
        distance_above_ema7_1m_pct=float(trigger_bar["distance_above_ema7_1m_pct"]),
        rel_volume_1m=float(trigger_bar["relative_volume_1m"]),
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
def run_backtest(
    df: pd.DataFrame,
    bt_start: pd.Timestamp,
    bt_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    trades: list[Trade] = []

    balance = STARTING_BALANCE
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    expired_count = 0
    rejected_count = 0
    no_trigger_count = 0

    i = max(LOOKBACK_4H_MINUTES, BOUNCE_LOOKBACK_MINUTES, 20)

    while i < len(df) - 5:
        row = df.iloc[i]
        now = row["open_time"]

        if now < bt_start:
            i += 1
            continue
        if now > bt_end:
            break
        if not in_session(now):
            i += 1
            continue

        if ONE_TRADE_AT_A_TIME and now < last_exit_time:
            i += 1
            continue

        trig_idx = select_best_candidate_trigger(df, i)
        if trig_idx is None:
            no_trigger_count += 1
            i += 1
            continue

        conf_idx = trig_idx + 1
        trigger_bar = df.iloc[trig_idx]
        confirm_bar = df.iloc[conf_idx]

        signal_time = row["open_time"]
        trigger_time = trigger_bar["open_time"]
        confirm_time = confirm_bar["open_time"]

        failure_high = float(max(trigger_bar["high"], confirm_bar["high"]))
        stop_price_initial = failure_high * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0)

        strength = compute_signal_strength(trigger_bar, confirm_bar)
        entry_mode = choose_entry_mode(signal_time, strength)

        fill_idx, order_placed_time, order_price, entry_price = simulate_short_entry(
            df=df,
            trigger_idx=trig_idx,
            confirm_idx=conf_idx,
            stop_price_initial=stop_price_initial,
            entry_mode=entry_mode,
        )

        if fill_idx is None or order_placed_time is None or entry_price is None:
            if entry_mode == "aggressive_market":
                rejected_count += 1
            else:
                expired_count += 1
            i = max(conf_idx + 1, trig_idx + 1)
            continue

        entry_time = df.iloc[fill_idx]["open_time"]
        if entry_time < bt_start or entry_time > bt_end:
            i = max(fill_idx + 1, trig_idx + 1)
            continue

        stop_distance = stop_price_initial - entry_price
        if stop_distance <= 0:
            rejected_count += 1
            i = max(fill_idx + 1, trig_idx + 1)
            continue

        risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
        qty = risk_amount / stop_distance

        max_notional = balance * LEVERAGE
        desired_notional = qty * entry_price
        if desired_notional > max_notional:
            qty = max_notional / entry_price

        if qty <= 0:
            rejected_count += 1
            i = max(fill_idx + 1, trig_idx + 1)
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
            entry_mode=entry_mode,
            row_context=row,
            order_price=order_price if order_price is not None else np.nan,
            stop_price_initial=stop_price_initial,
        )

        trades.append(trade)
        balance = trade.balance_after
        last_exit_time = exit_time

        i = max(fill_idx + 1, trig_idx + 1)

    trades_df = pd.DataFrame([asdict(t) for t in trades])

    if trades_df.empty:
        return trades_df, {
            "trade_count": 0,
            "ending_balance": STARTING_BALANCE,
            "expired_count": expired_count,
            "rejected_count": rejected_count,
            "no_trigger_count": no_trigger_count,
            "bt_start": bt_start,
            "bt_end": bt_end,
        }

    wins = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losses = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]

    return trades_df, {
        "trade_count": int(len(trades_df)),
        "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
        "expectancy_usd": float(trades_df["pnl_usd"].mean()),
        "ending_balance": float(trades_df["balance_after"].iloc[-1]),
        "expired_count": expired_count,
        "rejected_count": rejected_count,
        "no_trigger_count": no_trigger_count,
        "bt_start": bt_start,
        "bt_end": bt_end,
    }


# =========================================================
# MATCHING
# =========================================================
def match_to_reference(trades_df: pd.DataFrame, ref_df: pd.DataFrame, max_minutes: int = 60) -> pd.DataFrame:
    if trades_df.empty or ref_df.empty:
        return pd.DataFrame()

    bt = trades_df.copy()
    bt["entry_time"] = pd.to_datetime(bt["entry_time"], utc=True, errors="coerce")
    bt["bt_result"] = np.where(bt["pnl_usd"] > 0, "win", np.where(bt["pnl_usd"] < 0, "loss", "flat"))

    rows = []
    for _, bt_row in bt.iterrows():
        diffs = (ref_df["opened_time"] - bt_row["entry_time"]).abs()
        nearest_idx = diffs.idxmin()
        ref_row = ref_df.loc[nearest_idx]
        diff_minutes = diffs.loc[nearest_idx].total_seconds() / 60.0

        rows.append({
            "bt_entry_time": bt_row["entry_time"],
            "bt_entry_mode": bt_row["entry_mode"],
            "bt_pnl_usd": bt_row["pnl_usd"],
            "bt_result": bt_row["bt_result"],
            "ref_opened_time": ref_row["opened_time"],
            "ref_result": ref_row["ref_result"],
            "time_diff_minutes": diff_minutes,
            "matched_within_60m": diff_minutes <= max_minutes,
            "same_result_direction": bt_row["bt_result"] == ref_row["ref_result"],
        })

    return pd.DataFrame(rows).sort_values("bt_entry_time").reset_index(drop=True)


# =========================================================
# OUTPUT
# =========================================================
def save_outputs(trades_df: pd.DataFrame, summary: dict, matches_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(OUTPUT_DIR / "eth_short_trades.csv", index=False)

    lines = [
        "ETH SHORT TRADER STYLE SESSION MODEL V1",
        "=======================================",
        "",
        f"BT_START = {summary['bt_start']}",
        f"BT_END = {summary['bt_end']}",
        f"session_utc = [{SESSION_START_HOUR_UTC}:00, {SESSION_END_HOUR_UTC}:00)",
        "",
        f"trade_count = {summary['trade_count']}",
        f"win_rate_pct = {summary.get('win_rate_pct', np.nan):.2f}" if "win_rate_pct" in summary else "win_rate_pct = n/a",
        f"profit_factor = {summary.get('profit_factor', np.nan):.3f}" if "profit_factor" in summary else "profit_factor = n/a",
        f"expectancy_usd = {summary.get('expectancy_usd', np.nan):.2f}" if "expectancy_usd" in summary else "expectancy_usd = n/a",
        f"ending_balance = {summary['ending_balance']:.2f}",
        "",
        f"expired_count = {summary['expired_count']}",
        f"rejected_count = {summary['rejected_count']}",
        f"no_trigger_count = {summary['no_trigger_count']}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if matches_df.empty:
        (OUTPUT_DIR / "match_summary.txt").write_text("No matches.\n", encoding="utf-8")
        return

    matches_df.to_csv(OUTPUT_DIR / "matches.csv", index=False)

    within = int(matches_df["matched_within_60m"].sum())
    same_dir = int(matches_df.loc[matches_df["matched_within_60m"], "same_result_direction"].sum())

    match_lines = [
        "MATCH SUMMARY",
        "=============",
        f"total_backtest_trades = {len(matches_df)}",
        f"matches_within_60m = {within}",
        f"same_result_direction = {same_dir}",
        f"agreement_rate_pct = {same_dir / within * 100.0:.2f}" if within > 0 else "agreement_rate_pct = n/a",
    ]
    (OUTPUT_DIR / "match_summary.txt").write_text("\n".join(match_lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    if not CANDLES_1M_PATH.exists():
        raise FileNotFoundError(f"Missing candles file: {CANDLES_1M_PATH}")
    if not JSON_TRADES_PATH.exists():
        raise FileNotFoundError(f"Missing JSON file: {JSON_TRADES_PATH}")

    candles = load_candles(CANDLES_1M_PATH)
    ref_df = load_reference_eth_shorts(JSON_TRADES_PATH)

    bt_start, bt_end = get_reference_window(ref_df)
    trimmed = trim_candles_to_window(candles, bt_start, bt_end)
    featured = build_features(trimmed)

    trades_df, summary = run_backtest(
        featured,
        bt_start=bt_start,
        bt_end=bt_end,
    )

    matches_df = match_to_reference(trades_df, ref_df, max_minutes=60)
    save_outputs(trades_df, summary, matches_df)

    print(f"Reference ETH short window: {bt_start} -> {bt_end}")
    print(f"Session filter UTC: [{SESSION_START_HOUR_UTC}:00, {SESSION_END_HOUR_UTC}:00)")
    if not trades_df.empty:
        print(f"Backtest trades: {len(trades_df)}")
        print(f"Backtest trade window: {trades_df['entry_time'].min()} -> {trades_df['entry_time'].max()}")
    else:
        print("No trades produced.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
