from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .models import EntryPlan, StrategyDecision


ENTRY_RETEST_ABOVE_CLOSE_PCT = 0.15
ENTRY_WAIT_BARS = 15
INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT = 0.10
DEDUP_COOLDOWN_MINUTES = 15

CORE_REGIME = "near_entry_breakdown_noise"
SATELLITE_REGIME = "mixed_transition"
EXPANSION_REGIME = "other_non_target"
MILD_SUBTYPE = "mild_noise_v1_live"
COOLING_SUBTYPE = "cooling_v3_live"

ALLOWED_CORE_TREND_REGIMES = {"strong_bear", "weak_bear"}
SATELLITE_TREND_REGIME = "strong_bear"
SATELLITE_MIN_REL_VOLUME = 0.80
SATELLITE_SECONDARY_BODY_MAX = 0.50
SATELLITE_SECONDARY_EXCLUDED_HOUR = 13

STRICT_COOLING_MOVE1M_MAX = 0.00
STRICT_COOLING_MOVE3M_MIN = 0.50
STRICT_COOLING_DIST_1H_MIN = 0.10
STRICT_COOLING_BOUNCE_60M_MAX = 1.20

SELECTIVE_COOLING_MOVE1M_MIN = -0.30
SELECTIVE_COOLING_MOVE3M_MIN = 0.00
SELECTIVE_COOLING_MOVE3M_MAX = 0.50

BREAKDOWN_RELATIVE_VOLUME_MIN = 1.35
BREAKDOWN_MOVE_5M_MIN = 0.55
BREAKDOWN_BULL_BODY_SUM_5M_MIN = 0.65
BREAKDOWN_STRETCH_EMA20_5M_MIN = 0.90
BREAKDOWN_GREEN_COUNT_5M_MIN = 4

MIXED_RELATIVE_VOLUME_MAX = 1.20
MIXED_BOUNCE_60M_MAX = 1.50
MIXED_STRETCH_EMA20_5M_MAX = 1.40
MIXED_GREEN_COUNT_5M_MAX = 3
MIXED_DIST_1H_HIGH_MIN = 0.10

LIVE_COOLING_MOVE1M_MAX = -0.05
LIVE_COOLING_CLOSE_POSITION_MAX = 0.20
LIVE_COOLING_UPPER_WICK_MIN = 0.35
LIVE_STRONG_PUSH_BODY_MAX = 0.12
LIVE_STRONG_PUSH_MOVE3M_MIN = 0.30
LIVE_STRONG_PUSH_DIST_1H_HIGH_MAX = 0.05
LIVE_STRONG_PUSH_CLOSE_POSITION_MIN = 0.70

EXPANSION_TREND_REGIME = "strong_bear"
EXPANSION_VOL_REGIME = "high_vol"
EXPANSION_MIN_SCORE = 0.84
EXPANSION_MAX_MOVE_1M_PCT = 0.25
EXPANSION_MAX_MOVE_3M_PCT = 0.28
EXPANSION_MAX_MOVE_5M_PCT = 0.55
EXPANSION_MAX_REL_VOLUME_1M = 1.50
EXPANSION_MAX_DIST_1H_HIGH_PCT = 0.08
EXPANSION_MAX_BOUNCE_60M_PCT = 0.80

MICRO_FEATURES = [
    "ret_1m_pct",
    "ret_3m_pct",
    "ret_5m_pct",
    "rel_volume_1m",
    "upper_wick_ratio",
    "body_ratio",
    "close_position_in_range",
    "range_pct",
    "dist_1h_high_pct",
    "dist_5m_high_pct",
    "close_above_ema20_pct",
]

MICRO_WEIGHTS = {
    "ret_1m_pct": 0.80,
    "ret_3m_pct": 1.00,
    "ret_5m_pct": 0.90,
    "rel_volume_1m": 0.80,
    "upper_wick_ratio": 1.10,
    "body_ratio": 1.00,
    "close_position_in_range": 1.10,
    "range_pct": 0.70,
    "dist_1h_high_pct": 1.20,
    "dist_5m_high_pct": 1.10,
    "close_above_ema20_pct": 0.80,
}

TARGET_DELTAS = {
    "ret_1m_pct": 0.3092,
    "ret_3m_pct": 0.6735,
    "ret_5m_pct": 0.7186,
    "rel_volume_1m": 0.7294,
    "upper_wick_ratio": 0.1771,
    "body_ratio": 0.1320,
    "close_position_in_range": 0.0563,
    "range_pct": 0.5256,
    "dist_1h_high_pct": -0.1473,
    "dist_5m_high_pct": -0.2171,
    "close_above_ema20_pct": 0.3606,
}

DELTA_SCALES = {
    "ret_1m_pct": 1.3136,
    "ret_3m_pct": 1.8656,
    "ret_5m_pct": 2.1932,
    "rel_volume_1m": 1.4532,
    "upper_wick_ratio": 0.9925,
    "body_ratio": 0.9042,
    "close_position_in_range": 1.0465,
    "range_pct": 1.3641,
    "dist_1h_high_pct": 1.3352,
    "dist_5m_high_pct": 0.8480,
    "close_above_ema20_pct": 1.5558,
}


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


def _resample_5m(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.set_index("open_time")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def _build_feature_frame(candles_1m: pd.DataFrame) -> pd.DataFrame:
    df = candles_1m.copy()
    df["ema20"] = _ema(df["close"], 20)
    df["ret_1m_pct"] = df["close"].pct_change(1) * 100.0
    df["ret_3m_pct"] = df["close"].pct_change(3) * 100.0
    df["ret_5m_pct"] = df["close"].pct_change(5) * 100.0
    df["ret_15m_pct"] = df["close"].pct_change(15) * 100.0
    df["ret_60m_pct"] = df["close"].pct_change(60) * 100.0
    df["move_1m_pct"] = df["ret_1m_pct"]
    df["move_3m_pct"] = df["ret_3m_pct"]
    df["move_5m_pct"] = df["ret_5m_pct"]
    df["rel_volume_1m"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()
    df["relative_volume_1m"] = df["rel_volume_1m"]

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["upper_wick_ratio"] = (upper_wick / rng).fillna(0.0)
    df["body_ratio"] = (body / rng).fillna(0.0)
    df["close_position_in_range"] = ((df["close"] - df["low"]) / rng).fillna(0.5)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"] * 100.0).fillna(0.0)

    prev_5m_high = df["high"].shift(1).rolling(5, min_periods=5).max()
    prev_1h_high = df["high"].shift(1).rolling(60, min_periods=60).max()
    prev_4h_high = df["high"].shift(1).rolling(240, min_periods=240).max()
    df["dist_5m_high_pct"] = (prev_5m_high - df["close"]) / df["close"] * 100.0
    df["dist_1h_high_pct"] = (prev_1h_high - df["close"]) / df["close"] * 100.0
    df["dist_4h_high_pct"] = (prev_4h_high - df["close"]) / df["close"] * 100.0
    df["close_above_ema20_pct"] = (df["close"] / df["ema20"] - 1.0) * 100.0

    close_60m_ago = df["close"].shift(60)
    df["bounce_60m_pct"] = (df["close"] / close_60m_ago - 1.0) * 100.0
    df["ema7_1m"] = _ema(df["close"], 7)
    df["distance_above_ema7_1m_pct"] = (df["close"] / df["ema7_1m"] - 1.0) * 100.0

    df5 = _resample_5m(df)
    df5["ema20_5m"] = _ema(df5["close"], 20)
    df5["stretch_above_ema20_5m_pct"] = (df5["close"] / df5["ema20_5m"] - 1.0) * 100.0
    df = pd.merge_asof(
        df.sort_values("open_time"),
        df5[["open_time", "stretch_above_ema20_5m_pct"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )

    df["is_green"] = (df["close"] > df["open"]).astype(int)
    df["bull_body_pct"] = np.where(
        df["close"] > df["open"],
        (df["close"] - df["open"]) / df["open"] * 100.0,
        0.0,
    )
    df["green_count_last_5"] = df["is_green"].rolling(5, min_periods=5).sum()
    df["bull_body_sum_5m_pct"] = df["bull_body_pct"].rolling(5, min_periods=5).sum()
    df["hour_utc"] = df["open_time"].dt.hour
    return df


def _candidate_gate(row: pd.Series, session_start: int, session_end: int) -> bool:
    needed = [
        "ret_1m_pct", "ret_3m_pct", "ret_5m_pct", "ret_15m_pct", "ret_60m_pct",
        "rel_volume_1m", "upper_wick_ratio", "body_ratio", "dist_1h_high_pct",
        "dist_4h_high_pct", "dist_5m_high_pct", "close_above_ema20_pct",
    ]
    if any(pd.isna(row[c]) for c in needed):
        return False
    return (
        session_start <= int(row["hour_utc"]) < session_end
        and row["dist_4h_high_pct"] <= 0.80
        and row["dist_1h_high_pct"] <= 0.60
        and row["ret_60m_pct"] >= 0.20
        and row["ret_60m_pct"] <= 3.50
        and row["ret_15m_pct"] >= 0.10
        and row["ret_5m_pct"] >= -0.10
        and row["ret_5m_pct"] <= 1.80
        and row["close_above_ema20_pct"] >= -0.10
        and row["close_above_ema20_pct"] <= 2.50
        and row["rel_volume_1m"] <= 4.50
        and (row["upper_wick_ratio"] >= 0.05 or row["body_ratio"] <= 0.85)
    )


def _score_candidates(frame: pd.DataFrame) -> pd.Series:
    weighted = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        series = pd.to_numeric(frame[feature], errors="coerce")
        med = float(series.median())
        spread = float(series.quantile(0.75) - series.quantile(0.25))
        if not np.isfinite(spread) or spread <= 0:
            spread = float(series.std()) if np.isfinite(series.std()) and series.std() > 0 else 1.0
        current_delta = (series - med) / spread
        target = float(TARGET_DELTAS[feature])
        target_scale = float(DELTA_SCALES[feature])
        deviation = (current_delta - target).abs() / target_scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = float(MICRO_WEIGHTS[feature])
        weighted = weighted.add(feature_score * weight, fill_value=0.0)
        total_weight += weight
    return weighted / total_weight if total_weight > 0 else pd.Series(np.nan, index=frame.index)


def _generate_deduped_candidates(feature_frame: pd.DataFrame, session_start: int, session_end: int) -> pd.DataFrame:
    candidates = feature_frame.loc[feature_frame.apply(lambda row: _candidate_gate(row, session_start, session_end), axis=1)].copy()
    if candidates.empty:
        return candidates
    candidates["candidate_score"] = _score_candidates(candidates)
    kept: list[pd.Series] = []
    current_bucket: list[pd.Series] = []
    bucket_start: pd.Timestamp | None = None
    for _, row in candidates.sort_values("open_time").iterrows():
        now = row["open_time"]
        if bucket_start is None:
            bucket_start = now
            current_bucket = [row]
            continue
        if (now - bucket_start).total_seconds() / 60.0 <= DEDUP_COOLDOWN_MINUTES:
            current_bucket.append(row)
        else:
            best = max(current_bucket, key=lambda item: (float(item["candidate_score"]), -item["open_time"].value))
            kept.append(best)
            bucket_start = now
            current_bucket = [row]
    if current_bucket:
        best = max(current_bucket, key=lambda item: (float(item["candidate_score"]), -item["open_time"].value))
        kept.append(best)
    return pd.DataFrame(kept).sort_values("open_time").reset_index(drop=True)


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


def _predict_live_safe_day_regime_from_bar(bar: pd.Series) -> str:
    is_breakdown_noise = (
        float(bar["relative_volume_1m"]) >= BREAKDOWN_RELATIVE_VOLUME_MIN
        and float(bar["move_5m_pct"]) >= BREAKDOWN_MOVE_5M_MIN
        and float(bar["bull_body_sum_5m_pct"]) >= BREAKDOWN_BULL_BODY_SUM_5M_MIN
        and float(bar["stretch_above_ema20_5m_pct"]) >= BREAKDOWN_STRETCH_EMA20_5M_MIN
        and float(bar["green_count_last_5"]) >= BREAKDOWN_GREEN_COUNT_5M_MIN
    )
    if is_breakdown_noise:
        return CORE_REGIME

    is_mixed_transition = (
        float(bar["relative_volume_1m"]) <= MIXED_RELATIVE_VOLUME_MAX
        and float(bar["bounce_60m_pct"]) <= MIXED_BOUNCE_60M_MAX
        and float(bar["stretch_above_ema20_5m_pct"]) <= MIXED_STRETCH_EMA20_5M_MAX
        and float(bar["green_count_last_5"]) <= MIXED_GREEN_COUNT_5M_MAX
        and float(bar["dist_1h_high_pct"]) >= MIXED_DIST_1H_HIGH_MIN
    )
    if is_mixed_transition:
        return SATELLITE_REGIME
    return EXPANSION_REGIME


def _classify_near_noise_subtype_live(trigger_bar: pd.Series) -> str:
    cooling_signal = (
        float(trigger_bar["move_1m_pct"]) <= LIVE_COOLING_MOVE1M_MAX
        or float(trigger_bar["close_position_in_range"]) <= LIVE_COOLING_CLOSE_POSITION_MAX
        or float(trigger_bar["upper_wick_ratio"]) >= LIVE_COOLING_UPPER_WICK_MIN
    )
    if cooling_signal:
        return COOLING_SUBTYPE

    strong_push_signal = (
        float(trigger_bar["body_ratio"]) <= LIVE_STRONG_PUSH_BODY_MAX
        and float(trigger_bar["move_3m_pct"]) >= LIVE_STRONG_PUSH_MOVE3M_MIN
        and float(trigger_bar["dist_1h_high_pct"]) <= LIVE_STRONG_PUSH_DIST_1H_HIGH_MAX
        and float(trigger_bar["close_position_in_range"]) >= LIVE_STRONG_PUSH_CLOSE_POSITION_MIN
    )
    if strong_push_signal:
        return "strong_push_skip_live"
    return MILD_SUBTYPE


def _is_calm_mild_transition(bar: pd.Series) -> bool:
    return (
        float(bar["move_1m_pct"]) <= 0.11
        and float(bar["move_3m_pct"]) <= 0.18
        and float(bar["move_5m_pct"]) <= 0.25
        and float(bar["relative_volume_1m"]) <= 1.20
        and float(bar["bounce_60m_pct"]) >= 0.24
        and float(bar["bounce_60m_pct"]) <= 1.20
        and float(bar["dist_1h_high_pct"]) >= 0.10
        and float(bar["dist_1h_high_pct"]) <= 0.22
        and float(bar["bull_body_sum_5m_pct"]) <= 0.32
        and float(bar["green_count_last_5"]) in {2.0, 3.0}
    )


def _is_strict_cooling_extension(trigger_bar: pd.Series) -> bool:
    return (
        float(trigger_bar["move_1m_pct"]) <= STRICT_COOLING_MOVE1M_MAX
        and float(trigger_bar["move_3m_pct"]) >= STRICT_COOLING_MOVE3M_MIN
        and float(trigger_bar["dist_1h_high_pct"]) >= STRICT_COOLING_DIST_1H_MIN
        and float(trigger_bar["bounce_60m_pct"]) <= STRICT_COOLING_BOUNCE_60M_MAX
    )


def _is_selective_relaxed_cooling(trigger_bar: pd.Series) -> bool:
    return (
        float(trigger_bar["move_1m_pct"]) >= SELECTIVE_COOLING_MOVE1M_MIN
        and float(trigger_bar["move_3m_pct"]) >= SELECTIVE_COOLING_MOVE3M_MIN
        and float(trigger_bar["move_3m_pct"]) <= SELECTIVE_COOLING_MOVE3M_MAX
    )


class Tier287StrategyAdapter:
    def __init__(self, symbol: str, session_start_hour_utc: int, session_end_hour_utc: int) -> None:
        self.symbol = symbol
        self.session_start_hour_utc = session_start_hour_utc
        self.session_end_hour_utc = session_end_hour_utc
        self.candles_1m = pd.DataFrame()
        self.daily_1d = pd.DataFrame()
        self._last_evaluated_candle: str | None = None

    def bootstrap(self, candles_1m: pd.DataFrame, daily_1d: pd.DataFrame) -> None:
        self.candles_1m = _normalize_1m_frame(candles_1m).tail(2500).copy()
        self.daily_1d = _normalize_daily_frame(daily_1d).tail(180).copy()

    def append_closed_candle(self, candle: dict[str, Any]) -> None:
        frame = _normalize_1m_frame(pd.DataFrame([candle]))
        if frame.empty:
            return
        bar = frame.iloc[0]
        self.candles_1m = (
            pd.concat([self.candles_1m, frame], ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .tail(2500)
            .reset_index(drop=True)
        )
        day_open = pd.Timestamp(bar["open_time"]).floor("1D")
        if self.daily_1d.empty or pd.Timestamp(self.daily_1d.iloc[-1]["open_time"]) != day_open:
            append_row = pd.DataFrame(
                [{
                    "open_time": day_open,
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar["volume"]),
                }]
            )
            self.daily_1d = pd.concat([self.daily_1d, append_row], ignore_index=True).tail(180).reset_index(drop=True)
        else:
            idx = self.daily_1d.index[-1]
            self.daily_1d.at[idx, "high"] = max(float(self.daily_1d.at[idx, "high"]), float(bar["high"]))
            self.daily_1d.at[idx, "low"] = min(float(self.daily_1d.at[idx, "low"]), float(bar["low"]))
            self.daily_1d.at[idx, "close"] = float(bar["close"])
            self.daily_1d.at[idx, "volume"] = float(self.daily_1d.at[idx, "volume"]) + float(bar["volume"])

    def evaluate_latest(self) -> StrategyDecision:
        if self.candles_1m.empty:
            return StrategyDecision(
                signal_id=f"no-data-{datetime.now(timezone.utc).isoformat()}",
                candle_time=datetime.now(timezone.utc),
                status="no_signal",
                title="No candles loaded",
                explanation="Strategy buffer is empty.",
            )
        latest_candle_time = self.candles_1m.iloc[-1]["open_time"]
        latest_key = latest_candle_time.isoformat()
        if self._last_evaluated_candle == latest_key:
            return StrategyDecision(
                signal_id=f"dup-{latest_key}",
                candle_time=latest_candle_time,
                status="duplicate_candle",
                title="Candle already evaluated",
                explanation="Closed candle already processed.",
            )
        self._last_evaluated_candle = latest_key

        feature_frame = _build_feature_frame(self.candles_1m.copy())
        candidates = _generate_deduped_candidates(feature_frame, self.session_start_hour_utc, self.session_end_hour_utc)
        if candidates.empty:
            return StrategyDecision(
                signal_id=f"none-{latest_key}",
                candle_time=latest_candle_time,
                status="no_signal",
                title="No candidate",
                explanation="No Tier 2.87 candidate on this closed 1m candle.",
            )
        latest_candidate = candidates.iloc[-1]
        candidate_time = pd.to_datetime(latest_candidate["open_time"], utc=True)
        if candidate_time != latest_candle_time:
            return StrategyDecision(
                signal_id=f"none-{latest_key}",
                candle_time=latest_candle_time,
                status="no_signal",
                title="No new candidate",
                explanation="Latest closed candle did not become a deduped signal candidate.",
            )
        trigger_bar = feature_frame.loc[feature_frame["open_time"] == candidate_time].iloc[0]
        market_regime_frame = _build_market_regime_from_daily(self.daily_1d)
        latest_market = market_regime_frame.iloc[-1]
        market_ctx = {
            "trade_date": str(latest_market["trade_date"]),
            "trend_regime": str(latest_market["trend_regime"]),
            "vol_regime": str(latest_market["vol_regime"]),
        }
        predicted_regime = _predict_live_safe_day_regime_from_bar(trigger_bar)
        subtype = _classify_near_noise_subtype_live(trigger_bar)
        signal_id = f"{candidate_time.strftime('%Y%m%dT%H%M%S')}-{predicted_regime}-{subtype}"
        return self._route_signal(signal_id, candidate_time, latest_candidate, trigger_bar, predicted_regime, subtype, market_ctx)

    def _route_signal(
        self,
        signal_id: str,
        candidate_time: datetime,
        latest_candidate: pd.Series,
        trigger_bar: pd.Series,
        predicted_regime: str,
        subtype: str,
        market_ctx: dict[str, Any],
    ) -> StrategyDecision:
        candidate_score = float(latest_candidate["candidate_score"])
        trigger_price = float(trigger_bar["close"])
        stop_price = float(trigger_bar["high"]) * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0)
        base_payload = {
            "candidate_score": candidate_score,
            "predicted_regime": predicted_regime,
            "subtype": subtype,
            "market_ctx": market_ctx,
            "trigger_snapshot": {
                "open_time": candidate_time.isoformat(),
                "close": trigger_price,
                "high": float(trigger_bar["high"]),
                "move_1m_pct": float(trigger_bar["move_1m_pct"]),
                "move_3m_pct": float(trigger_bar["move_3m_pct"]),
                "move_5m_pct": float(trigger_bar["move_5m_pct"]),
                "relative_volume_1m": float(trigger_bar["relative_volume_1m"]),
                "body_ratio": float(trigger_bar["body_ratio"]),
                "dist_1h_high_pct": float(trigger_bar["dist_1h_high_pct"]),
                "bounce_60m_pct": float(trigger_bar["bounce_60m_pct"]),
            },
        }

        if predicted_regime == CORE_REGIME:
            if market_ctx["trend_regime"] not in ALLOWED_CORE_TREND_REGIMES:
                return StrategyDecision(signal_id, candidate_time, "blocked", "Core signal blocked by daily trend regime", f"Predicted core setup, but daily trend regime is {market_ctx['trend_regime']}.", CORE_REGIME, CORE_REGIME, subtype, candidate_score, None, base_payload)
            if subtype == MILD_SUBTYPE:
                plan = EntryPlan(signal_id, candidate_time, "tier2p5_core_mild_limit", CORE_REGIME, subtype, candidate_score, "LIMIT", "SELL", trigger_price * (1.0 + ENTRY_RETEST_ABOVE_CLOSE_PCT / 100.0), stop_price, ENTRY_WAIT_BARS * 60, "Core mild-noise setup. Place short limit retest order above trigger close.", base_payload)
                return StrategyDecision(signal_id, candidate_time, "accepted", "Core mild-noise limit setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())
            if subtype == COOLING_SUBTYPE and _is_strict_cooling_extension(trigger_bar):
                plan = EntryPlan(signal_id, candidate_time, "tier2p5_core_strict_cooling_limit", CORE_REGIME, "strict_cooling_v1_live", candidate_score, "LIMIT", "SELL", trigger_price * (1.0 + ENTRY_RETEST_ABOVE_CLOSE_PCT / 100.0), stop_price, ENTRY_WAIT_BARS * 60, "Core strict-cooling extension. Place short limit retest order.", base_payload)
                return StrategyDecision(signal_id, candidate_time, "accepted", "Core strict-cooling limit setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())
            if subtype == COOLING_SUBTYPE and _is_selective_relaxed_cooling(trigger_bar):
                plan = EntryPlan(signal_id, candidate_time, "tier2p5_core_selective_cooling_market", CORE_REGIME, "selective_relaxed_cooling_market_v1", candidate_score, "MARKET", "SELL", None, stop_price, 0, "Core selective relaxed cooling setup. Enter short immediately by market.", base_payload)
                return StrategyDecision(signal_id, candidate_time, "accepted", "Core selective-cooling market setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())
            return StrategyDecision(signal_id, candidate_time, "blocked", "Core setup blocked by subtype", f"Subtype {subtype} did not pass core routing filters.", CORE_REGIME, CORE_REGIME, subtype, candidate_score, None, base_payload)

        if predicted_regime == SATELLITE_REGIME:
            if market_ctx["trend_regime"] != SATELLITE_TREND_REGIME:
                return StrategyDecision(signal_id, candidate_time, "blocked", "Satellite blocked by market trend", f"Satellite trades are only enabled in {SATELLITE_TREND_REGIME}, got {market_ctx['trend_regime']}.", SATELLITE_REGIME, SATELLITE_REGIME, subtype, candidate_score, None, base_payload)
            if subtype != MILD_SUBTYPE:
                return StrategyDecision(signal_id, candidate_time, "blocked", "Satellite blocked by subtype", f"Mixed-transition satellite requires {MILD_SUBTYPE}, got {subtype}.", SATELLITE_REGIME, SATELLITE_REGIME, subtype, candidate_score, None, base_payload)
            if not _is_calm_mild_transition(trigger_bar):
                return StrategyDecision(signal_id, candidate_time, "blocked", "Satellite blocked by calm-transition filter", "Mixed-transition candidate did not pass calm mild-transition filter.", SATELLITE_REGIME, SATELLITE_REGIME, subtype, candidate_score, None, base_payload)
            rel_volume = float(trigger_bar["relative_volume_1m"])
            body_ratio = float(trigger_bar["body_ratio"])
            entry_hour = int(candidate_time.hour)
            if rel_volume >= SATELLITE_MIN_REL_VOLUME:
                plan = EntryPlan(signal_id, candidate_time, "tier5_satellite_rv08_market", SATELLITE_REGIME, subtype, candidate_score, "MARKET", "SELL", None, stop_price, 0, "Mixed-transition mild setup with rel_volume >= 0.80. Market short.", base_payload)
                return StrategyDecision(signal_id, candidate_time, "accepted", "Satellite RV08 market setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())
            if body_ratio <= SATELLITE_SECONDARY_BODY_MAX and entry_hour != SATELLITE_SECONDARY_EXCLUDED_HOUR:
                plan = EntryPlan(signal_id, candidate_time, "tier5_satellite_body05_not13_market", SATELLITE_REGIME, subtype, candidate_score, "MARKET", "SELL", None, stop_price, 0, "Mixed-transition mild setup from body<=0.50 and hour!=13 bucket. Market short.", base_payload)
                return StrategyDecision(signal_id, candidate_time, "accepted", "Satellite body05/not13 market setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())
            return StrategyDecision(signal_id, candidate_time, "blocked", "Satellite blocked by secondary micro filters", "Mixed-transition candidate failed RV08 and body05/hour filters.", SATELLITE_REGIME, SATELLITE_REGIME, subtype, candidate_score, None, base_payload)

        accepted = (
            predicted_regime == EXPANSION_REGIME
            and subtype == MILD_SUBTYPE
            and market_ctx["trend_regime"] == EXPANSION_TREND_REGIME
            and market_ctx["vol_regime"] == EXPANSION_VOL_REGIME
            and candidate_score >= EXPANSION_MIN_SCORE
            and float(trigger_bar["move_1m_pct"]) <= EXPANSION_MAX_MOVE_1M_PCT
            and float(trigger_bar["move_3m_pct"]) <= EXPANSION_MAX_MOVE_3M_PCT
            and float(trigger_bar["move_5m_pct"]) <= EXPANSION_MAX_MOVE_5M_PCT
            and float(trigger_bar["relative_volume_1m"]) <= EXPANSION_MAX_REL_VOLUME_1M
            and float(trigger_bar["dist_1h_high_pct"]) <= EXPANSION_MAX_DIST_1H_HIGH_PCT
            and float(trigger_bar["bounce_60m_pct"]) <= EXPANSION_MAX_BOUNCE_60M_PCT
        )
        if accepted:
            plan = EntryPlan(signal_id, candidate_time, "tier287_other_non_target_mild_strong_bear_market", EXPANSION_REGIME, subtype, candidate_score, "MARKET", "SELL", None, stop_price, 0, "Tier 2.87 strong-bear/high-vol expansion accepted. Market short.", base_payload)
            return StrategyDecision(signal_id, candidate_time, "accepted", "Tier 2.87 expansion setup accepted.", plan.explanation, plan.route, plan.regime, plan.subtype, plan.candidate_score, plan, plan.to_payload())

        return StrategyDecision(signal_id, candidate_time, "blocked", "Candidate blocked by Tier 2.87 routing", "Candidate exists, but it does not pass active Tier 2.87 route filters.", predicted_regime, predicted_regime, subtype, candidate_score, None, base_payload)
