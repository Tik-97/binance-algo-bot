from __future__ import annotations

from .strategies.short_t287 import (
    ShortT287Strategy,
    _build_market_regime_from_daily,
)


Tier287StrategyAdapter = ShortT287Strategy

__all__ = [
    "ShortT287Strategy",
    "Tier287StrategyAdapter",
    "_build_market_regime_from_daily",
]
