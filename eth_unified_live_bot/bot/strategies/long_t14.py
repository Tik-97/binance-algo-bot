from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..models import StrategyDecision
from ._long_reversal_common import (
    T14_CORE_MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT,
    T14_CORE_MIN_CLIMAX_DROP_5M_PCT,
    T14_SAT1_MAX_CLIMAX_DROP_5M_PCT,
    T14_SAT1_MAX_CONFIRM_RANGE_PCT,
    T14_SAT1_MAX_TRIGGER_BODY_RATIO,
    T14_SAT1_MIN_CONTEXT_DUMP_60M_PCT,
    T14_SAT1_MIN_RELATIVE_VOLUME_1M,
    T14_SAT2_MAX_TRIGGER_BODY_RATIO,
    T14_SAT2_MIN_CONFIRM_RANGE_PCT,
    T14_SAT2_MIN_TRIGGER_LOWER_WICK_RATIO,
    _base_long_context,
    _blocked_decision,
    _build_long_feature_frame,
    _build_market_regime_from_daily,
    _in_session,
    _is_strong_long_signal,
    _is_trigger_rejection,
    _is_valid_confirm,
    _make_long_entry_plan,
    _no_signal_decision,
    _normalize_1m_frame,
    _normalize_daily_frame,
    _passes_climax_dump_filter,
)
from .base import BaseStrategy


def _score_t14(route: str, trigger_bar: pd.Series, confirm_bar: pd.Series) -> float:
    base_score = {
        "tier11_core": 0.93,
        "satellite1_soft_flush_refined": 0.89,
        "satellite2_reclaim_extension": 0.90,
    }[route]
    score = (
        base_score
        + min(0.03, max(0.0, float(trigger_bar["drop_5m_pct"]) - 0.70) * 0.02)
        + min(0.02, max(0.0, float(trigger_bar["lower_wick_ratio"]) - 0.35) * 0.05)
        + min(0.01, max(0.0, float(confirm_bar["range_pct"]) - 0.10) * 0.02)
    )
    if _is_strong_long_signal(trigger_bar, confirm_bar):
        score += 0.02
    return min(0.99, round(score, 4))


def _classify_t14_route(row: pd.Series, trigger_bar: pd.Series, confirm_bar: pd.Series) -> str | None:
    if (
        float(trigger_bar["drop_5m_pct"]) >= T14_CORE_MIN_CLIMAX_DROP_5M_PCT
        and float(row["stretch_below_ema20_5m_pct"]) <= T14_CORE_MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT
    ):
        return "tier11_core"

    if (
        float(trigger_bar["drop_5m_pct"]) < T14_SAT1_MAX_CLIMAX_DROP_5M_PCT
        and float(confirm_bar["range_pct"]) <= T14_SAT1_MAX_CONFIRM_RANGE_PCT
        and float(trigger_bar["relative_volume_1m"]) >= T14_SAT1_MIN_RELATIVE_VOLUME_1M
        and float(row["dump_60m_pct"]) >= T14_SAT1_MIN_CONTEXT_DUMP_60M_PCT
        and float(trigger_bar["body_ratio"]) <= T14_SAT1_MAX_TRIGGER_BODY_RATIO
        and not _is_strong_long_signal(trigger_bar, confirm_bar)
    ):
        return "satellite1_soft_flush_refined"

    if (
        float(confirm_bar["range_pct"]) >= T14_SAT2_MIN_CONFIRM_RANGE_PCT
        and float(trigger_bar["body_ratio"]) <= T14_SAT2_MAX_TRIGGER_BODY_RATIO
        and float(trigger_bar["lower_wick_ratio"]) >= T14_SAT2_MIN_TRIGGER_LOWER_WICK_RATIO
    ):
        return "satellite2_reclaim_extension"

    return None


class LongT14Strategy(BaseStrategy):
    name = "long_t14"
    side = "LONG"
    priority = 2

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
            append_row = pd.DataFrame(
                [{
                    "open_time": day_open,
                    "open": float(latest_bar["open"]),
                    "high": float(latest_bar["high"]),
                    "low": float(latest_bar["low"]),
                    "close": float(latest_bar["close"]),
                    "volume": float(latest_bar["volume"]),
                }]
            )
            self.daily_1d = pd.concat([self.daily_1d, append_row], ignore_index=True).tail(180).reset_index(drop=True)
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
                "Long T1.4 warming up",
                "Need more closed 1m candles before long features are stable.",
            )

        feature_frame = _build_long_feature_frame(self.candles_1m.copy(), include_family2_context=False)
        confirm_bar = feature_frame.iloc[-1]
        candle_time = pd.to_datetime(confirm_bar["open_time"], utc=True).to_pydatetime()

        if not _in_session(pd.Timestamp(candle_time), self.session_start_hour_utc, self.session_end_hour_utc):
            return _no_signal_decision(
                self.name,
                candle_time,
                f"out-of-session-{latest_key}",
                "Outside long T1.4 session",
                "Latest closed candle is outside the configured long T1.4 session window.",
            )

        trigger_bar = feature_frame.iloc[-2]
        signal_id = f"long-t14-{pd.Timestamp(candle_time).strftime('%Y%m%dT%H%M%S')}"

        market_regime_frame = _build_market_regime_from_daily(self.daily_1d)
        latest_market = market_regime_frame.iloc[-1]
        market_ctx = {
            "trade_date": str(latest_market["trade_date"]),
            "trend_regime": str(latest_market["trend_regime"]),
            "vol_regime": str(latest_market["vol_regime"]),
        }
        if market_ctx["trend_regime"] not in {"strong_bear", "weak_bear"} or market_ctx["vol_regime"] != "high_vol":
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Long T1.4 blocked by daily regime",
                explanation=(
                    f"Tier 1.4 requires bear-like daily trend and high_vol, got "
                    f"{market_ctx['trend_regime']} / {market_ctx['vol_regime']}."
                ),
                regime="bear_high_vol_flush",
                subtype="regime_block",
                payload={"market_ctx": market_ctx},
            )

        if not _is_trigger_rejection(trigger_bar):
            return _no_signal_decision(
                self.name,
                candle_time,
                signal_id,
                "No trigger rejection",
                "Previous closed candle did not form the required lower-wick rejection trigger.",
            )
        if not _passes_climax_dump_filter(trigger_bar):
            return _no_signal_decision(
                self.name,
                candle_time,
                signal_id,
                "No climax dump trigger",
                "Previous candle did not pass the long climax-dump trigger filter.",
            )
        if not _is_valid_confirm(trigger_bar, confirm_bar):
            return _no_signal_decision(
                self.name,
                candle_time,
                signal_id,
                "No confirm candle",
                "Latest candle did not confirm the trigger with a proper reclaim close/high.",
            )

        context_candidates = [
            ("confirm_context", confirm_bar),
            ("trigger_context", trigger_bar),
        ]
        selected_context_source: str | None = None
        selected_context_row: pd.Series | None = None
        route: str | None = None
        for context_source, context_row in context_candidates:
            if not _base_long_context(context_row):
                continue
            route = _classify_t14_route(context_row, trigger_bar, confirm_bar)
            if route is not None:
                selected_context_source = context_source
                selected_context_row = context_row
                break

        if route is None or selected_context_row is None or selected_context_source is None:
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Long T1.4 candidate blocked by route filters",
                explanation="Base long setup exists, but neither confirm-context nor trigger-context matches an active Tier 1.4 route.",
                regime="bear_high_vol_flush",
                subtype="route_block",
                payload={
                    "market_ctx": market_ctx,
                    "trigger_body_ratio": float(trigger_bar["body_ratio"]),
                    "trigger_lower_wick_ratio": float(trigger_bar["lower_wick_ratio"]),
                    "confirm_range_pct": float(confirm_bar["range_pct"]),
                    "confirm_context_active": bool(_base_long_context(confirm_bar)),
                    "trigger_context_active": bool(_base_long_context(trigger_bar)),
                },
            )

        candidate_score = _score_t14(route, trigger_bar, confirm_bar)
        explanation = {
            "tier11_core": "Tier 1.4 core flush-reversal setup. Enter long on market if strong, otherwise place shallow pullback limit.",
            "satellite1_soft_flush_refined": "Tier 1.4 satellite1 soft flush setup. Enter long on market if strong, otherwise place shallow pullback limit.",
            "satellite2_reclaim_extension": "Tier 1.4 satellite2 reclaim-extension setup. Enter long on market if strong, otherwise place shallow pullback limit.",
        }[route]
        plan = _make_long_entry_plan(
            strategy_name=self.name,
            signal_id=signal_id,
            signal_time=candle_time,
            route=route,
            regime="bear_high_vol_flush",
            subtype=route,
            candidate_score=candidate_score,
            explanation=explanation,
            trigger_bar=trigger_bar,
            confirm_bar=confirm_bar,
            market_ctx={**market_ctx, "context_source": selected_context_source},
        )
        if plan.price is not None and plan.price <= plan.stop_price:
            return _blocked_decision(
                strategy_name=self.name,
                signal_id=signal_id,
                candle_time=candle_time,
                title="Long T1.4 blocked by invalid limit geometry",
                explanation="Planned shallow-pullback limit would sit below the protective stop.",
                route=route,
                regime=plan.regime,
                subtype=plan.subtype,
                candidate_score=candidate_score,
                payload=plan.to_payload(),
            )

        return StrategyDecision(
            signal_id=signal_id,
            candle_time=candle_time,
            status="accepted",
            title="Long T1.4 setup accepted",
            explanation=plan.explanation,
            strategy_name=self.name,
            route=plan.route,
            regime=plan.regime,
            subtype=plan.subtype,
            candidate_score=plan.candidate_score,
            entry_plan=plan,
            payload=plan.to_payload(),
        )
