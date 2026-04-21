from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


@dataclass(slots=True)
class SymbolRules:
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float
    quantity_precision: int
    price_precision: int


@dataclass(slots=True)
class EntryPlan:
    signal_id: str
    signal_time: datetime
    route: str
    regime: str
    subtype: str
    candidate_score: float
    order_type: str
    side: str
    price: float | None
    stop_price: float
    ttl_seconds: int
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class StrategyDecision:
    signal_id: str
    candle_time: datetime
    status: str
    title: str
    explanation: str
    route: str | None = None
    regime: str | None = None
    subtype: str | None = None
    candidate_score: float | None = None
    entry_plan: EntryPlan | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        data = asdict(self)
        if self.entry_plan is not None:
            data["entry_plan"] = self.entry_plan.to_payload()
        return to_jsonable(data)


@dataclass(slots=True)
class ActiveSignal:
    signal_id: str
    candle_time: datetime
    route: str
    regime: str
    subtype: str
    candidate_score: float
    trigger_price: float
    trigger_high: float
    stop_price: float
    limit_price: float | None
    expires_at: datetime | None
    status: str
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class ManagedPosition:
    symbol: str
    side: str
    signal_id: str
    route: str
    regime: str
    subtype: str
    entry_time: datetime
    entry_price: float
    qty_initial: float
    qty_remaining: float
    stop_price_initial: float
    stop_price_current: float
    partial_taken: bool
    trail_active: bool
    bars_since_entry: int
    best_low: float
    best_high: float
    entry_fee_usd: float
    entry_order_client_id: str
    protective_stop_client_id: str | None = None
    entry_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class ClosedTrade:
    signal_id: str
    route: str
    regime: str
    subtype: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    exit_reason: str
    fees_usd: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["payload"] = to_jsonable(data["payload"])
        return to_jsonable(data)

