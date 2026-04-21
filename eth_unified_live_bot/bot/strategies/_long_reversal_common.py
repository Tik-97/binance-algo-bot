from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..models import EntryPlan, StrategyDecision


LOOKBACK_4H_MINUTES = 240
BOUNCE_LOOKBACK_MINUTES = 60
NEAR_4H_LOW_MAX_PCT = 0.35
MIN_DUMP_60M_PCT = 0.45
MIN_STRETCH_BELOW_EMA20_5M_PCT = 0.15

LOWER_WICK_RATIO_MIN = 0.35
BODY_RATIO_MAX = 0.55

MIN_CLIMAX_DROP_3M_PCT = 0.45
MIN_CLIMAX_DROP_5M_PCT = 0.70
MAX_CLIMAX_DROP_5M_PCT = 2.50
MIN_RED_COUNT_LAST_5 = 3
MIN_BEAR_BODY_SUM_5M_PCT = 0.35
MIN_DISTANCE_BELOW_EMA7_1M_PCT = 0.20
MAX_TRIGGER_RANGE_PCT = 0.80
MAX_CONFIRM_RANGE_PCT = 0.80
MAX_RELATIVE_VOLUME_1M = 6.0

STRONG_SIGNAL_MIN_CLIMAX_5M_PCT = 0.95
STRONG_SIGNAL_MIN_BEAR_BODY_SUM_5M_PCT = 0.45
STRONG_SIGNAL_MIN_CONFIRM_BREAK_PCT = 0.12

SHALLOW_PULLBACK_LIMIT_PCT_BELOW_CONFIRM_CLOSE = 0.04
MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT = 0.12
ENTRY_WAIT_BARS = 6
INITIAL_STOP_BUFFER_BELOW_FAIL_LOW_PCT = 0.10

T14_CORE_MIN_CLIMAX_DROP_5M_PCT = 0.85
T14_CORE_MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT = 1.30
T14_SAT1_MAX_CLIMAX_DROP_5M_PCT = 0.85
T14_SAT1_MAX_CONFIRM_RANGE_PCT = 0.25
T14_SAT1_MIN_RELATIVE_VOLUME_1M = 1.40
T14_SAT1_MIN_CONTEXT_DUMP_60M_PCT = 1.10
T14_SAT1_MAX_TRIGGER_BODY_RATIO = 0.45
T14_SAT2_MIN_CONFIRM_RANGE_PCT = 0.40
T14_SAT2_MAX_TRIGGER_BODY_RATIO = 0.35
T14_SAT2_MIN_TRIGGER_LOWER_WICK_RATIO = 0.45

F21_MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT = 1.10
F21_MIN_TRIGGER_BODY_RATIO = 0.14
F21_MAX_TRIGGER_BODY_RATIO = 0.26
F21_MIN_RELATIVE_VOLUME_1M = 1.20
F21_SAT_HIGHCONFIRM_MIN_TRIGGER_BODY_RATIO = 0.45
F21_SAT_HIGHCONFIRM_MIN_CONFIRM_RANGE_PCT = 0.45
F21_SAT_HIGHCONFIRM_MAX_CONTEXT_DUMP_60M_PCT = 1.05
F21_SAT_MIDBODY_MIN_TRIGGER_BODY_RATIO = 0.29
F21_SAT_MIDBODY_MAX_TRIGGER_BODY_RATIO = 0.35
F21_SAT_MIDBODY_MAX_CONFIRM_RANGE_PCT = 0.45
F21_SAT_MIDBODY_MIN_RELATIVE_VOLUME_1M = 1.20


def _normalize_1m_frame(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    return frame.sort_values("open_time").reset_index(drop=True)


def _normalize_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"])
    return frame.sort_values("open_time").reset_index(drop=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        frame.set_index("open_time")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def _build_market_regime_from_daily(daily_1d: pd.DataFrame) -> pd.DataFrame:
    daily = daily_1d.copy()
    daily["trade_date"] = daily["open_time"].dt.strftime("%Y-%m-%d")
    daily["daily_return_pct"] = daily["close"].pct_change() * 100.0
    daily["ret_30d_pct"] = daily["close"].pct_change(30) * 100.0
    daily["ret_60d_pct"] = daily["close"].pct_change(60) * 100.0
    daily["sma_30"] = daily["close"].rolling(30).mean()
    daily["sma_60"] = daily["close"].rolling(60).mean()
    daily["close_vs_sma30_pct"] = (daily["close"] / daily["sma_30"] - 1.0) * 100.0
    daily["close_vs_sma60_pct"] = (daily["close"] / daily["sma_60"] - 1.0) * 100.0
    daily["vol_14d_pct"] = daily["daily_return_pct"].rolling(14).std()
    daily["trend_regime"] = "transition"
    strong_bear = (daily["ret_60d_pct"] <= -12.0) & (daily["close_vs_sma60_pct"] <= -3.0)
    strong_bull = (daily["ret_60d_pct"] >= 12.0) & (daily["close_vs_sma60_pct"] >= 3.0)
    weak_bear = (daily["ret_30d_pct"] <= -5.0) & (daily["close_vs_sma30_pct"] <= -1.5)
    weak_bull = (daily["ret_30d_pct"] >= 5.0) & (daily["close_vs_sma30_pct"] >= 1.5)
    daily.loc[strong_bear, "trend_regime"] = "strong_bear"
    daily.loc[strong_bull, "trend_regime"] = "strong_bull"
    daily.loc[(daily["trend_regime"] == "transition") & weak_bear, "trend_regime"] = "weak_bear"
    daily.loc[(daily["trend_regime"] == "transition") & weak_bull, "trend_regime"] = "weak_bull"
    vol_threshold = float(daily["vol_14d_pct"].median(skipna=True))
    daily["vol_regime"] = np.where(daily["vol_14d_pct"] >= vol_threshold, "high_vol", "low_vol")
    return daily


def _prepare_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ema20"] = _ema(prepared["close"], 20)
    prepared["ema50"] = _ema(prepared["close"], 50)
    prepared["ema200"] = _ema(prepared["close"], 200)
    prepared["ema_alignment"] = np.where(
        (prepared["ema20"] > prepared["ema50"]) & (prepared["ema50"] > prepared["ema200"]),
        "bullish",
        np.where(
            (prepared["ema20"] < prepared["ema50"]) & (prepared["ema50"] < prepared["ema200"]),
            "bearish",
            "mixed",
        ),
    )
    return prepared


def _classify_structure_regime(close_value: float, prev_high: float, prev_low: float) -> str:
    if pd.notna(prev_high) and close_value > prev_high:
        return "breakout_up"
    if pd.notna(prev_low) and close_value < prev_low:
        return "breakout_down"
    if pd.isna(prev_high) or pd.isna(prev_low) or prev_high == prev_low:
        return "unknown"
    range_position = (close_value - prev_low) / (prev_high - prev_low)
    if range_position >= 0.8:
        return "near_high"
    if range_position <= 0.2:
        return "near_low"
    return "mid_range"


def _classify_context_style_long(
    ret_5m_pct: float,
    ret_15m_pct: float,
    ret_60m_pct: float,
    structure_regime: str,
    ema_alignment_15m: str,
    ema_alignment_1h: str,
) -> str:
    if pd.notna(ret_60m_pct) and pd.notna(ret_5m_pct) and ret_60m_pct > 0 and ret_5m_pct < 0:
        return "pullback"
    if structure_regime in {"breakout_up", "breakout_down"}:
        return "breakout_continuation"
    if ema_alignment_15m == "bearish" or ema_alignment_1h == "bearish":
        return "reversal_like"
    if pd.notna(ret_15m_pct) and pd.notna(ret_5m_pct) and ret_15m_pct > 0 and ret_5m_pct > 0:
        return "continuation_like"
    return "mid_range_or_mixed"


def _build_long_feature_frame(candles_1m: pd.DataFrame, include_family2_context: bool) -> pd.DataFrame:
    df = candles_1m.copy()

    rolling_4h_low = df["low"].rolling(LOOKBACK_4H_MINUTES, min_periods=LOOKBACK_4H_MINUTES).min()
    df["dist_4h_low_pct"] = (df["close"] - rolling_4h_low) / df["close"] * 100.0

    close_60m_ago = df["close"].shift(BOUNCE_LOOKBACK_MINUTES)
    df["dump_60m_pct"] = (1.0 - df["close"] / close_60m_ago) * 100.0

    close_3m_ago = df["close"].shift(3)
    close_5m_ago = df["close"].shift(5)
    close_15m_ago = df["close"].shift(15)
    df["drop_3m_pct"] = (1.0 - df["close"] / close_3m_ago) * 100.0
    df["drop_5m_pct"] = (1.0 - df["close"] / close_5m_ago) * 100.0
    df["move_15m_pct"] = (df["close"] / close_15m_ago - 1.0) * 100.0
    df["ret_5m_pct"] = (df["close"] / close_5m_ago - 1.0) * 100.0
    df["ret_15m_pct"] = df["move_15m_pct"]
    df["ret_60m_pct"] = (df["close"] / close_60m_ago - 1.0) * 100.0

    df["relative_volume_1m"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()
    df["ema7_1m"] = _ema(df["close"], 7)
    df["distance_below_ema7_1m_pct"] = (1.0 - df["close"] / df["ema7_1m"]) * 100.0

    df5 = _resample_ohlcv(df, "5min")
    df5["ema20_5m"] = _ema(df5["close"], 20)
    df5["stretch_below_ema20_5m_pct"] = (1.0 - df5["close"] / df5["ema20_5m"]) * 100.0
    df5["open_time"] = df5["open_time"] + pd.Timedelta(minutes=5)
    df = pd.merge_asof(
        df.sort_values("open_time"),
        df5[["open_time", "stretch_below_ema20_5m_pct"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_ratio"] = (body / rng).fillna(0.0)
    df["lower_wick_ratio"] = (lower_wick / rng).fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"] * 100.0).fillna(0.0)

    df["is_red"] = (df["close"] < df["open"]).astype(int)
    df["bear_body_pct"] = np.where(
        df["close"] < df["open"],
        (df["open"] - df["close"]) / df["open"] * 100.0,
        0.0,
    )
    df["red_count_last_5"] = df["is_red"].rolling(5, min_periods=5).sum()
    df["bear_body_sum_5m_pct"] = df["bear_body_pct"].rolling(5, min_periods=5).sum()
    df["hour_utc"] = df["open_time"].dt.hour

    if include_family2_context:
        df15 = _prepare_context_features(_resample_ohlcv(df, "15min"))
        df15 = df15[["open_time", "ema_alignment"]].rename(columns={"ema_alignment": "ema_alignment_15m"})
        df1h = _prepare_context_features(_resample_ohlcv(df, "1h"))
        df1h = df1h[["open_time", "ema_alignment"]].rename(columns={"ema_alignment": "ema_alignment_1h"})

        df = pd.merge_asof(df.sort_values("open_time"), df15.sort_values("open_time"), on="open_time", direction="backward")
        df = pd.merge_asof(df.sort_values("open_time"), df1h.sort_values("open_time"), on="open_time", direction="backward")

        prev_4h_high = df["high"].rolling(241, min_periods=2).max().shift(1)
        prev_4h_low = df["low"].rolling(241, min_periods=2).min().shift(1)
        df["structure_regime2"] = [
            _classify_structure_regime(float(close), prev_high, prev_low)
            for close, prev_high, prev_low in zip(df["close"], prev_4h_high, prev_4h_low)
        ]
        df["return_pre_60m_pct2"] = df["ret_60m_pct"]
        df["context_style_label2"] = [
            _classify_context_style_long(
                float(ret_5m) if pd.notna(ret_5m) else np.nan,
                float(ret_15m) if pd.notna(ret_15m) else np.nan,
                float(ret_60m) if pd.notna(ret_60m) else np.nan,
                str(structure_regime),
                str(align15),
                str(align1h),
            )
            for ret_5m, ret_15m, ret_60m, structure_regime, align15, align1h in zip(
                df["ret_5m_pct"],
                df["ret_15m_pct"],
                df["return_pre_60m_pct2"],
                df["structure_regime2"],
                df["ema_alignment_15m"],
                df["ema_alignment_1h"],
            )
        ]

    return df


def _in_session(candle_time: pd.Timestamp, session_start_hour_utc: int, session_end_hour_utc: int) -> bool:
    hour = int(candle_time.hour)
    if session_start_hour_utc == session_end_hour_utc:
        return True
    if session_start_hour_utc < session_end_hour_utc:
        return session_start_hour_utc <= hour < session_end_hour_utc
    return hour >= session_start_hour_utc or hour < session_end_hour_utc


def _base_long_context(row: pd.Series) -> bool:
    needed = ["dist_4h_low_pct", "dump_60m_pct", "stretch_below_ema20_5m_pct"]
    if any(pd.isna(row.get(column)) for column in needed):
        return False
    return (
        float(row["dist_4h_low_pct"]) <= NEAR_4H_LOW_MAX_PCT
        and float(row["dump_60m_pct"]) >= MIN_DUMP_60M_PCT
        and float(row["stretch_below_ema20_5m_pct"]) >= MIN_STRETCH_BELOW_EMA20_5M_PCT
    )


def _is_trigger_rejection(bar: pd.Series) -> bool:
    if pd.isna(bar["lower_wick_ratio"]) or pd.isna(bar["body_ratio"]) or pd.isna(bar["range_pct"]):
        return False
    return (
        float(bar["lower_wick_ratio"]) >= LOWER_WICK_RATIO_MIN
        and float(bar["body_ratio"]) <= BODY_RATIO_MAX
        and float(bar["range_pct"]) <= MAX_TRIGGER_RANGE_PCT
    )


def _passes_climax_dump_filter(bar: pd.Series) -> bool:
    needed = [
        "drop_3m_pct",
        "drop_5m_pct",
        "red_count_last_5",
        "bear_body_sum_5m_pct",
        "distance_below_ema7_1m_pct",
        "relative_volume_1m",
    ]
    if any(pd.isna(bar[column]) for column in needed):
        return False
    return (
        float(bar["drop_3m_pct"]) >= MIN_CLIMAX_DROP_3M_PCT
        and float(bar["drop_5m_pct"]) >= MIN_CLIMAX_DROP_5M_PCT
        and float(bar["drop_5m_pct"]) <= MAX_CLIMAX_DROP_5M_PCT
        and float(bar["red_count_last_5"]) >= MIN_RED_COUNT_LAST_5
        and float(bar["bear_body_sum_5m_pct"]) >= MIN_BEAR_BODY_SUM_5M_PCT
        and float(bar["distance_below_ema7_1m_pct"]) >= MIN_DISTANCE_BELOW_EMA7_1M_PCT
        and float(bar["relative_volume_1m"]) <= MAX_RELATIVE_VOLUME_1M
    )


def _is_valid_confirm(trigger_bar: pd.Series, confirm_bar: pd.Series) -> bool:
    return (
        float(confirm_bar["range_pct"]) <= MAX_CONFIRM_RANGE_PCT
        and float(confirm_bar["close"]) > float(trigger_bar["close"])
        and float(confirm_bar["high"]) > float(trigger_bar["high"])
    )


def _is_strong_long_signal(trigger_bar: pd.Series, confirm_bar: pd.Series) -> bool:
    confirm_break_pct = (float(confirm_bar["close"]) - float(trigger_bar["close"])) / float(trigger_bar["close"]) * 100.0
    return (
        float(trigger_bar["drop_5m_pct"]) >= STRONG_SIGNAL_MIN_CLIMAX_5M_PCT
        and float(trigger_bar["bear_body_sum_5m_pct"]) >= STRONG_SIGNAL_MIN_BEAR_BODY_SUM_5M_PCT
        and confirm_break_pct >= STRONG_SIGNAL_MIN_CONFIRM_BREAK_PCT
        and float(confirm_bar["range_pct"]) <= MAX_CONFIRM_RANGE_PCT
    )


def _build_long_entry_metadata(
    trigger_bar: pd.Series,
    confirm_bar: pd.Series,
    market_ctx: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "trigger_time": pd.to_datetime(trigger_bar["open_time"], utc=True).isoformat(),
        "confirm_time": pd.to_datetime(confirm_bar["open_time"], utc=True).isoformat(),
        "entry_cancel_if_low_le": float(trigger_bar["low"]),
        "entry_cancel_if_close_lt": float(confirm_bar["close"]) * (1.0 - MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT / 100.0),
        "trigger_snapshot": {
            "close": float(trigger_bar["close"]),
            "high": float(trigger_bar["high"]),
            "low": float(trigger_bar["low"]),
            "drop_3m_pct": float(trigger_bar["drop_3m_pct"]),
            "drop_5m_pct": float(trigger_bar["drop_5m_pct"]),
            "relative_volume_1m": float(trigger_bar["relative_volume_1m"]),
            "body_ratio": float(trigger_bar["body_ratio"]),
            "lower_wick_ratio": float(trigger_bar["lower_wick_ratio"]),
            "bear_body_sum_5m_pct": float(trigger_bar["bear_body_sum_5m_pct"]),
        },
        "confirm_snapshot": {
            "close": float(confirm_bar["close"]),
            "high": float(confirm_bar["high"]),
            "low": float(confirm_bar["low"]),
            "range_pct": float(confirm_bar["range_pct"]),
        },
        "market_ctx": market_ctx,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _make_long_entry_plan(
    *,
    strategy_name: str,
    signal_id: str,
    signal_time: datetime,
    route: str,
    regime: str,
    subtype: str,
    candidate_score: float,
    explanation: str,
    trigger_bar: pd.Series,
    confirm_bar: pd.Series,
    market_ctx: dict[str, Any],
) -> EntryPlan:
    stop_reference = min(float(trigger_bar["low"]), float(confirm_bar["low"]))
    stop_price = stop_reference * (1.0 - INITIAL_STOP_BUFFER_BELOW_FAIL_LOW_PCT / 100.0)
    strong_signal = _is_strong_long_signal(trigger_bar, confirm_bar)
    if strong_signal:
        metadata = _build_long_entry_metadata(
            trigger_bar,
            confirm_bar,
            market_ctx,
            {
                "entry_mode_research": "aggressive_market",
                "strong_signal": True,
            },
        )
        return EntryPlan(
            signal_id=signal_id,
            signal_time=signal_time,
            strategy_name=strategy_name,
            route=route,
            regime=regime,
            subtype=subtype,
            candidate_score=candidate_score,
            order_type="MARKET",
            side="BUY",
            price=None,
            stop_price=stop_price,
            ttl_seconds=0,
            explanation=explanation,
            metadata=metadata,
        )

    limit_price = float(confirm_bar["close"]) * (1.0 - SHALLOW_PULLBACK_LIMIT_PCT_BELOW_CONFIRM_CLOSE / 100.0)
    metadata = _build_long_entry_metadata(
        trigger_bar,
        confirm_bar,
        market_ctx,
        {
            "entry_mode_research": "shallow_pullback_limit",
            "strong_signal": False,
        },
    )
    return EntryPlan(
        signal_id=signal_id,
        signal_time=signal_time,
        strategy_name=strategy_name,
        route=route,
        regime=regime,
        subtype=subtype,
        candidate_score=candidate_score,
        order_type="LIMIT",
        side="BUY",
        price=limit_price,
        stop_price=stop_price,
        ttl_seconds=ENTRY_WAIT_BARS * 60,
        explanation=explanation,
        metadata=metadata,
    )


def _blocked_decision(
    *,
    strategy_name: str,
    signal_id: str,
    candle_time: datetime,
    title: str,
    explanation: str,
    route: str | None = None,
    regime: str | None = None,
    subtype: str | None = None,
    candidate_score: float | None = None,
    payload: dict[str, Any] | None = None,
) -> StrategyDecision:
    return StrategyDecision(
        signal_id=signal_id,
        candle_time=candle_time,
        status="blocked",
        title=title,
        explanation=explanation,
        strategy_name=strategy_name,
        route=route,
        regime=regime,
        subtype=subtype,
        candidate_score=candidate_score,
        payload=payload or {},
    )


def _no_signal_decision(strategy_name: str, candle_time: datetime, signal_id: str, title: str, explanation: str) -> StrategyDecision:
    return StrategyDecision(
        signal_id=signal_id,
        candle_time=candle_time,
        status="no_signal",
        title=title,
        explanation=explanation,
        strategy_name=strategy_name,
    )
