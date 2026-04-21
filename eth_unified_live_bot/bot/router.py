from __future__ import annotations

import logging

from .models import StrategyDecision
from .strategies.base import BaseStrategy


LOGGER = logging.getLogger("eth_unified_live_bot.router")


class UnifiedRouter:
    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self.strategies = sorted(strategies, key=lambda strategy: strategy.priority)
        self.last_statuses: dict[str, str] = {strategy.name: "not_loaded" for strategy in self.strategies}
        self.last_decisions: dict[str, StrategyDecision] = {}
        self.last_conflicts: dict[str, int] = {
            "short_beat_long": 0,
            "long_t14_deduped_f21": 0,
        }

    def evaluate(
        self,
        has_open_position: bool,
        has_active_signal: bool,
    ) -> StrategyDecision | None:
        self.last_statuses = {strategy.name: "not_evaluated" for strategy in self.strategies}
        self.last_decisions = {}
        self.last_conflicts = {
            "short_beat_long": 0,
            "long_t14_deduped_f21": 0,
        }

        if has_open_position or has_active_signal:
            for strategy in self.strategies:
                self.last_statuses[strategy.name] = "wallet_locked"
            LOGGER.info(
                "Router: wallet locked — open_position=%s active_signal=%s",
                has_open_position,
                has_active_signal,
            )
            self._log_result(None)
            return None

        accepted: StrategyDecision | None = None
        long_t14_decision: StrategyDecision | None = None
        long_f21_strategy = next((strategy for strategy in self.strategies if strategy.name == "long_f21"), None)

        for strategy in self.strategies:
            if strategy.name == "long_f21" and long_t14_decision is not None and long_t14_decision.status == "accepted":
                decision = strategy.evaluate_latest()
                self.last_statuses[strategy.name] = decision.status
                self.last_decisions[strategy.name] = decision
                if decision.status == "accepted":
                    self.last_statuses[strategy.name] = "deduped_by_long_t14"
                    self.last_conflicts["long_t14_deduped_f21"] = 1
                    LOGGER.info("Router: long_f21 deduped — long_t14 already accepted")
                accepted = long_t14_decision
                break

            decision = strategy.evaluate_latest()
            self.last_statuses[strategy.name] = decision.status
            self.last_decisions[strategy.name] = decision

            if decision.status != "accepted":
                continue

            if strategy.name == "short_t287":
                for lower in self.strategies:
                    if lower.priority > strategy.priority:
                        self.last_statuses[lower.name] = "skipped_higher_priority"
                if any(lower.side == "LONG" for lower in self.strategies if lower.priority > strategy.priority):
                    self.last_conflicts["short_beat_long"] = 1
                accepted = decision
                break

            if strategy.name == "long_t14" and long_f21_strategy is not None:
                long_t14_decision = decision
                continue

            accepted = decision
            break

        if accepted is None and long_t14_decision is not None:
            accepted = long_t14_decision
            if long_f21_strategy is None:
                self.last_statuses.setdefault("long_f21", "not_loaded")

        self._log_result(accepted)
        return accepted

    def _log_result(self, accepted: StrategyDecision | None) -> None:
        accepted_name = accepted.strategy_name if accepted is not None else "none"
        ordered_names = ["short_t287", "long_t14", "long_f21"]
        status_parts = []
        for name in ordered_names:
            if name in self.last_statuses:
                status_parts.append(f"{name}={self.last_statuses[name]}")
        LOGGER.info(
            "Router result: candle=%s accepted=%s %s",
            accepted.candle_time.isoformat() if accepted is not None else "none",
            accepted_name,
            " ".join(status_parts),
        )
