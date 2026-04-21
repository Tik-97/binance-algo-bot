from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..models import StrategyDecision


class BaseStrategy(ABC):
    name: str
    side: str
    priority: int

    def __init__(self, symbol: str, session_start_hour_utc: int, session_end_hour_utc: int) -> None:
        self.symbol = symbol
        self.session_start_hour_utc = session_start_hour_utc
        self.session_end_hour_utc = session_end_hour_utc

    @abstractmethod
    def bootstrap(self, candles_1m: pd.DataFrame, daily_1d: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def ingest_candle(self, bar: dict) -> None:
        ...

    @abstractmethod
    def evaluate_latest(self) -> StrategyDecision:
        ...
