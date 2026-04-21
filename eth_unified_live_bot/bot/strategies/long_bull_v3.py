from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..models import EntryPlan, StrategyDecision
from ._long_reversal_common import (
    INITIAL_STOP_BUFFER_BELOW_FAIL_LOW_PCT,
    SHALLOW_PULLBACK_LIMIT_PCT_BELOW_CONFIRM_CLOSE,
    ENTRY_WAIT_BARS,
    MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT,
    _blocked_decision,
    _build_long_feature_frame,
    _build_market_regime_from_daily,
    _in_session,
    _no_signal_decision,
    _normalize_1m_frame,
    _normalize_daily_frame,
)
from .base import BaseStrategy

# =========================================================
# BULL V3 PARAMETERS
# =========================================================
_BULL_TREND_REGIMES = frozenset({"strong_bull"})
_BULL_VOL_REGIME = "high_vol"

# Intraday context (inverted vs bear flush)
_BULL_MIN_DIST_4H_LOW_PCT = 1.50       # price far above 4h rolling low
_BULL_MAX_DUMP_60M_PCT = -0.20         # price ROSE in last 60m (negative = rose)
_BULL_MAX_STRETCH_BELOW_EMA20_PCT = 0.50  # near or above 5m EMA20

# Trigger: mild pullback (not climax dump)
_BULL_MIN_DROP_5M_PCT = 0.20
_BULL_MAX_DROP_5M_PCT = 1.20
_BULL_MIN_LOWER_WICK_RATIO = 0.35
_BULL_MAX_BODY_RATIO = 0.55
_BULL_MAX_TRIGGER_RANGE_PCT = 0.70
_BULL_MAX_RED_COUNT_LAST_5 = 3.0
_BULL_MIN_REL_VOLUME = 0.80

# Confirm: TIGHT range is the key filter (stress test confirmed cliff at 0.16→0.18)
_BULL_MAX_CONFIRM_RANGE_PCT = 0.16


def _is_bull_v3_context(row: pd.Series) -> bool:
    needed = ["dist_4h_low_pct", "dump_60m_pct", "stretch_below_ema20_5m_pct"]
    if any(pd.isna(row.get(col)) for col in needed):
        return False
    return (
        float(row["dist_4h_low_pct"]) >= _BULL_MIN_DIST_4H_LOW_PCT
        and float(row["dump_60m_pct"]) <= _BULL_MAX_DUMP_60M_PCT
        and float(row["stretch_below_ema20_5m_pct"]) <= _BULL_MAX_STRETCH_BELOW_EMA20_PCT
    )


def _passes_bull_v3_trigger(trigger_bar: pd.Series, confirm_bar: pd.Series) -> bool:
    needed = [
        "drop_5m_pct", "red_count_last_5", "relative_volume_1m",
        "lower_wick_ratio", "body_ratio", "range_pct",
    ]
    if any(pd.isna(trigger_bar.get(col)) for col in needed):
        return False

    drop = float(trigger_bar["drop_5m_pct"])
    if not (_BULL_MIN_DROP_5M_PCT <= drop <= _BULL_MAX_DROP_5M_PCT):
        return False
    if float(trigger_bar["red_count_last_5"]) > _BULL_MAX_RED_COUNT_LAST_5:
        return False
    if float(trigger_bar["relative_volume_1m"]) < _BULL_MIN_REL_VOLUME:
        return False
    if float(trigger_bar["lower_wick_ratio"]) < _BULL_MIN_LOWER_WICK_RATIO:
        return False
    if float(trigger_bar["body_ratio"]) > _BULL_MAX_BODY_RATIO:
        return False
    if float(trigger_bar["range_pct"]) > _BULL_MAX_TRIGGER_RANGE_PCT:
        return False

    if pd.isna(confirm_bar.get("range_pct")) or pd.isna(confirm_bar.get("close")) or pd.isna(confirm_bar.get("high")):
        return False
    if float(confirm_bar["range_pct"]) > _BULL_MAX_CONFIRM_RANGE_PCT:
        return False
    if float(confirm_bar["close"]) <= float(trigger_bar["close"]):
        return False
    if float(confirm_bar["high"]) <= float(trigger_bar["high"]):
        return False

    return True


def _make_bull_v3_entry_plan(
    *,
    strategy_name: str,
    signal_id: str,
    signal_time: datetime,
    trigger_bar: pd.Series,
    confirm_bar: pd.Series,
    market_ctx: dict[str, Any],
    candidate_score: float,
    explanation: str,
) -> EntryPlan:
    stop_reference = min(float(trigger_bar["low"]), float(confirm_bar["low"]))
    stop_price = stop_reference * (1.0 - INITIAL_STOP_BUFFER_BELOW_FAIL_LOW_PCT / 100.0)

    # Always limit for bull continuation: market entry was worse in backtests (PF 0.733)
    limit_price = float(confirm_bar["close"]) * (
        1.0 - SHALLOW_PULLBACK_LIMIT_PCT_BELOW_CONFIRM_CLOSE / 100.0
    )

    metadata: dict[str, Any] = {
        "trigger_time": pd.to_datetime(trigger_bar["open_time"], utc=True).isoformat(),
        "confirm_time": pd.to_datetime(confirm_bar["open_time"], utc=True).isoformat(),
        "entry_cancel_if_low_le": float(trigger_bar["low"]),
        "entry_cancel_if_close_lt": float(confirm_bar["close"]) * (
            1.0 - MAX_ALLOWED_PULLBACK_BELOW_CONFIRM_CLOSE_PCT / 100.0
        ),
        "trigger_snapshot": {
            "close": float(trigger_bar["close"]),
            "high": float(trigger_bar["high"]),
            "low": float(trigger_bar["low"]),
            "drop_5m_pct": float(trigger_bar["drop_5m_pct"]),
            "relative_volume_1m": float(trigger_bar["relative_volume_1m"]),
            "lower_wick_ratio": float(trigger_bar["lower_wick_ratio"]),
            "body_ratio": float(trigger_bar["body_ratio"]),
            "red_count_last_5": float(trigger_bar["red_count_last_5"]),
        },
        "confirm_snapshot": {
            "close": float(confirm_bar["close"]),
            "high": float(confirm_bar["high"]),
            "low": float(confirm_bar["low"]),
            "range_pct": float(confirm_bar["range_pct"]),
        },
        "market_ctx": market_ctx,
        "entry_mode_research": "shallow_pullback_limit",
    }

    return EntryPlan(
        signal_id=signal_id,
        signal_time=signal_time,
        strategy_name=strategy_name,
        route="bull_mild_pullback",
        regime="bull_high_vol_continuation",
        subtype="bull_mild_pullback",
        candidate_score=candidate_score,
        order_type="LIMIT",
        side="BUY",
        price=limit_price,
        stop_price=stop_price,
        ttl_seconds=ENTRY_WAIT_BARS * 60,
        explanation=explanation,
        metadata=metadata,
    )


def _score_bull_v3(trigger_bar: pd.Series, confirm_bar: pd.Series) -> float:
    base = 0.78
    vol = float(trigger_bar.get("relative_volume_1m", 0.80))
    wick = float(trigger_bar.get("lower_wick_ratio", 0.35))
    conf_range = float(confirm_bar.get("range_pct", 0.16))

    score = (
        base
        + min(0.08, max(0.0, (vol - 0.80) * 0.10))
        + min(0.05, max(0.0, (wick - 0.35) * 0.10))
        + min(0.04, max(0.0, (_BULL_MAX_CONFIRM_RANGE_PCT - conf_range) * 0.50))
    )
    return min(0.95, round(score, 4))


class LongBullV3Strategy(BaseStrategy):
    name = "long_bull_v3"
    side = "LONG"
    priority = 4

    def __init__(self, symbol: str, session_start_hour_utc: int, session_end_hour_utc: int) -> None:
        super().__init__(symbol, session_start_hour_utc, session_end_hour_utc)
        self.candles_1m = pd.DataFrame()
        self.daily_1d = pd.DataFrame()
        self._last_evaluated_candle: str | None = None

    def bootstrap(self, candles_1m: pd.DataFrame, daily_1d: pd.DataFrame) -> None:
        self.candles_1m = _normalize_1m_frame(candles_1m).tail(2500).copy()
        self.daily_1d = _normalize_daily_frame(daily_1d).tail(180).copy()

    def ingest_candle(self, bar: dict[str, Any]) -> None:
        frame = _normalize_1m_frame(pd.DataFrame([bar]))
        if frame.empty:
            return
        latest_bar = frame.iloc[0]
        self.candles_1m = (
            pd.concat([self.candles_1m, frame], ignore_index=True)
            .drop_duplicates(subset=["open_time"], keep="last")
            .sort_values("open_time")
            .tail(2500)
            .reset_index(drop=True)
        )
        day_open = pd.Timestamp(latest_bar["open_time"]).floor("1D")
        if self.daily_1d.empty or pd.Timestamp(self.daily_1d.iloc[-1]["open_time"]) != day_open:
            self.daily_1d = pd.concat(
                [
                    self.daily_1d,
                    pd.DataFrame([{
                        "open_time": day_open,
                        "open": float(latest_bar["open"]),
                        "high": float(latest_bar["high"]),
                        "low": float(latest_bar["low"]),
                        "close": float(latest_bar["close"]),
                        "volume": float(latest_bar["volume"]),
                    }]),
                ],
                ignore_index=True,
            ).tail(180).reset_index(drop=True)
        else:
            idx = self.daily_1d.index[-1]
            self.daily_1d.at[idx, "high"] = max(float(self.daily_1d.at[idx, "high"]), float(latest_bar["high"]))
            self.daily_1d.at[idx, "low"] = min(float(self.daily_1d.at[idx, "low"]), float(latest_bar["low"]))
            self.daily_1d.at[idx, "close"] = float(latest_bar["close"])
            self.daily_1d.at[idx, "volume"] = float(self.daily_1d.at[idx, "volume"]) + float(latest_bar["volume"])

    def evaluate_latest(self) -> StrategyDecision:
        if self.candles_1m.empty:
            return _no_signal_decision(
                self.name,
                datetime.now(timezone.utc),
                f"no-data-{datetime.now(timezone.utc).isoformat()}",
                "No candles loaded",
                "Strategy buffer is empty.",
            )

        latest_candle_time = pd.to_datetime(self.candles_1m.iloc[-1]["open_time"], utc=True).to_pydatetime()
        latest_key = latest_candle_time.isoformat()
        if self._last_evaluated_candle == latest_key:
            return StrategyDecision(
                signal_id=f"dup-{latest_key}",
                candle_time=latest_candle_time,
                status="duplicate_candle",
                title="Candle already evaluated",
                explanation="Closed candle already processed.",
                strategy_name=self.name,
            )
        self._last_evaluated_candle = latest_key

        if len(self.candles_1m) < 120:
            return _no_signal_decision(
                self.name,
                latest_candle_time,
                f"warmup-{latest_key}",
                "Long Bull V3 warming up",
                "Need at least 120 closed 1m candles for features to stabilise.",
            )

        feature_frame = _build_long_feature_frame(self.candles_1m.copy(), include_family2_context=False)
        confirm_bar = feature_frame.iloc[-1]
        candle_time = pd.to_datetime(confirm_bar["open_time"], utc=True).to_pydatetime()

        if not _in_session(pd.Timestamp(candle_time), self.session_start_hour_utc, self.session_end_hour_utc):
            return _no_signal_decision(
                self.name,
                candle_time,
                f"out-of-session-{latest_key}",
                "Outside Bull V3 session",
                "Latest candle is outside the configured Bull V3 session window.",
            )

        trigger_bar = feature_frame.iloc[-2]
        signal_id = f"long-bull-v3-{pd.Timestamp(candle_time).strftime('%Y%m%dT%H%M%S')}"

        # Daily regime: must be strong_bull + high_vol
        market_regime_frame = _build_market_regime_from_daily(self.daily_1d)
        latest_market = market_regime_frame.iloc[-1]
        market_ctx = {
            "trade_date": str(latest_market["trade_date"]),
            "trend_regime": str(latest_market["trend_regime"]),
            "vol_regime": str(latest_market["vol_regime"]),
        }
        if market_ctx["trend_regime"] not in _BULL_TREND_REGIMES or market_ctx["vol_regime"] != _BULL_VOL_REGIME:
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Bull V3 blocked by daily regime",
                explanation=(
                    f"Bull V3 requires strong_bull + high_vol daily regime, "
                    f"got {market_ctx['trend_regime']} / {market_ctx['vol_regime']}."
                ),
                regime="bull_high_vol_continuation",
                subtype="regime_block",
                payload={"market_ctx": market_ctx},
            )

        # Trigger + confirm quality check
        if not _passes_bull_v3_trigger(trigger_bar, confirm_bar):
            return _no_signal_decision(
                self.name,
                candle_time,
                signal_id,
                "No Bull V3 trigger",
                "Trigger/confirm did not meet mild-pullback + tight-confirm criteria.",
            )

        # Bull intraday context (try trigger_bar first, then confirm_bar)
        context_source: str | None = None
        context_row: pd.Series | None = None
        for src, row in [("trigger_context", trigger_bar), ("confirm_context", confirm_bar)]:
            if _is_bull_v3_context(row):
                context_source = src
                context_row = row
                break

        if context_source is None or context_row is None:
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Bull V3 blocked by intraday context",
                explanation=(
                    "Trigger/confirm passed quality filters but neither bar satisfies "
                    "bull intraday context (dist_4h_low>=1.5%, rally_60m>=-0.20%, stretch<=0.50%)."
                ),
                regime="bull_high_vol_continuation",
                subtype="context_block",
                payload={
                    "market_ctx": market_ctx,
                    "trigger_dist_4h_low": float(trigger_bar.get("dist_4h_low_pct", float("nan"))),
                    "trigger_dump_60m": float(trigger_bar.get("dump_60m_pct", float("nan"))),
                    "trigger_stretch_ema20": float(trigger_bar.get("stretch_below_ema20_5m_pct", float("nan"))),
                },
            )

        candidate_score = _score_bull_v3(trigger_bar, confirm_bar)
        explanation = (
            "Bull V3: mild pullback (0.20-1.20%) in strong_bull+high_vol regime with tight confirm. "
            "Enter long via shallow limit order below confirm close."
        )

        plan = _make_bull_v3_entry_plan(
            strategy_name=self.name,
            signal_id=signal_id,
            signal_time=candle_time,
            trigger_bar=trigger_bar,
            confirm_bar=confirm_bar,
            market_ctx={**market_ctx, "context_source": context_source},
            candidate_score=candidate_score,
            explanation=explanation,
        )

        if plan.price is not None and plan.price <= plan.stop_price:
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Bull V3 blocked: invalid limit geometry",
                explanation="Planned limit price would sit at or below the stop loss.",
                regime=plan.regime,
                subtype="geometry_block",
                candidate_score=candidate_score,
                payload=plan.to_payload(),
            )

        return StrategyDecision(
            signal_id=signal_id,
            candle_time=candle_time,
            status="accepted",
            title="Long Bull V3 setup accepted",
            explanation=plan.explanation,
            strategy_name=self.name,
            route=plan.route,
            regime=plan.regime,
            subtype=plan.subtype,
            candidate_score=plan.candidate_score,
            entry_plan=plan,
            payload=plan.to_payload(),
        )
