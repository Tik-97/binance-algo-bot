from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import websockets

from .config import AppConfig
from .exchange import BinanceFuturesClient, ExchangeError
from .models import ClosedTrade, EntryPlan, ManagedPosition
from .notifier import TelegramNotifier
from .state import StateStore
from .strategy import Tier287StrategyAdapter, _build_market_regime_from_daily


LOGGER = logging.getLogger("eth_short_demo_bot.service")

FAST_FAIL_WINDOW_BARS = 8
FAST_FAIL_MIN_FAVORABLE_PCT = 0.05
FAST_FAIL_MAX_ADVERSE_PCT = 0.45
TAKE_PARTIAL_AT_R = 2.0
PARTIAL_CLOSE_SIZE = 0.50
BREAKEVEN_OFFSET_PCT = 0.00
TRAIL_AFTER_R = 2.0
TRAIL_LOOKBACK_BARS = 3
TRAIL_BUFFER_PCT = 0.08
MAKER_FEE_PCT = 0.02
TAKER_FEE_PCT = 0.05
MARKET_ENTRY_SLIPPAGE_PCT = 0.025
MARKET_EXIT_SLIPPAGE_PCT = 0.025


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(ts: datetime | None = None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat()


def candle_trade_day(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")


def order_id(prefix: str) -> str:
    return f"esd-{prefix}-{uuid.uuid4().hex[:20]}"


class ETHShortDemoBotService:
    def __init__(
        self,
        config: AppConfig,
        store: StateStore,
        exchange: BinanceFuturesClient,
        notifier: TelegramNotifier,
    ) -> None:
        self.config = config
        self.store = store
        self.exchange = exchange
        self.notifier = notifier
        self.strategy = Tier287StrategyAdapter(
            symbol=config.symbol,
            session_start_hour_utc=config.session_start_hour_utc,
            session_end_hour_utc=config.session_end_hour_utc,
        )
        self.symbol_rules = None
        self.listen_key: str | None = None
        self.market_stream_task: asyncio.Task | None = None
        self.user_stream_task: asyncio.Task | None = None
        self.reconcile_task: asyncio.Task | None = None
        self.keepalive_task: asyncio.Task | None = None
        self.heartbeat_task: asyncio.Task | None = None
        self.report_task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self._last_pre_report_day: str | None = None
        self._last_post_report_day: str | None = None
        self._last_status_ping_at: datetime | None = None

    @staticmethod
    def _load_position_state(raw: dict[str, Any]) -> ManagedPosition:
        data = dict(raw)
        data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True).to_pydatetime()
        return ManagedPosition(**data)

    async def startup(self) -> None:
        self.store.set_state("service_status", {"status": "starting", "ts": utc_iso()})
        self.store.log_event("INFO", "startup", "Bot startup", "Service startup initiated.")
        await self.exchange.ping()
        await self.exchange.sync_time()
        await self.exchange.set_position_mode_one_way()
        await self.exchange.set_margin_type_cross(self.config.symbol)
        await self.exchange.change_leverage(self.config.symbol, self.config.leverage)

        _, self.symbol_rules = await self.exchange.exchange_info(self.config.symbol)
        candles_1m = await self.exchange.klines(self.config.symbol, "1m", limit=self.config.max_bootstrap_1m_candles)
        daily_1d = await self.exchange.klines(self.config.symbol, "1d", limit=self.config.daily_kline_history_limit)
        self.strategy.bootstrap(candles_1m=candles_1m, daily_1d=daily_1d)

        await self._recover_and_reconcile("startup")
        await self._process_catchup_candles("startup")

        self.listen_key = await self.exchange.create_listen_key()
        self.store.set_state("listen_key", {"listen_key": self.listen_key, "ts": utc_iso()})

        self.store.set_state("service_status", {"status": "running", "ts": utc_iso()})
        await self.notifier.send_event(
            (
                f"<b>ETH Short Demo Bot started</b>\n"
                f"Symbol: <code>{self.config.symbol}</code>\n"
                f"Baseline: <code>Tier 2.87</code>\n"
                f"Mode: <code>Binance Futures DEMO</code>\n"
                f"UTC: <code>{utc_iso()}</code>"
            ),
            dedupe_key=f"startup:{utc_now().strftime('%Y%m%d%H')}",
        )

    async def shutdown(self) -> None:
        self.stop_event.set()
        self.store.set_state("service_status", {"status": "stopping", "ts": utc_iso()})
        with suppress(Exception):
            if self.listen_key:
                await self.exchange.close_listen_key(self.listen_key)
        await self.notifier.send_event(
            f"<b>ETH Short Demo Bot stopped</b>\nUTC: <code>{utc_iso()}</code>",
            dedupe_key=f"stop:{utc_now().strftime('%Y%m%d%H%M')}",
            silent=True,
        )
        self.store.set_state("service_status", {"status": "stopped", "ts": utc_iso()})

    async def run_forever(self) -> None:
        await self.startup()
        self.market_stream_task = asyncio.create_task(self._market_stream_loop(), name="market-stream")
        self.user_stream_task = asyncio.create_task(self._user_stream_loop(), name="user-stream")
        self.reconcile_task = asyncio.create_task(self._reconcile_loop(), name="reconcile-loop")
        self.keepalive_task = asyncio.create_task(self._listen_key_keepalive_loop(), name="listenkey-loop")
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat-loop")
        self.report_task = asyncio.create_task(self._daily_report_loop(), name="report-loop")

        tasks = [
            self.market_stream_task,
            self.user_stream_task,
            self.reconcile_task,
            self.keepalive_task,
            self.heartbeat_task,
            self.report_task,
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if task is not None:
                    task.cancel()
            await self.shutdown()

    async def _process_catchup_candles(self, reason: str) -> None:
        raw = self.store.get_state("last_processed_candle")
        if not raw or "open_time" not in raw:
            return
        last_processed = pd.to_datetime(raw["open_time"], utc=True)
        now = utc_now()
        latest_closed = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        if last_processed >= latest_closed:
            return

        start_ms = int((last_processed + timedelta(minutes=1)).timestamp() * 1000)
        end_ms = int((latest_closed + timedelta(minutes=1)).timestamp() * 1000)
        missed = await self.exchange.klines_range(self.config.symbol, "1m", start_ms, end_ms)
        if missed.empty:
            return

        self.store.log_event(
            "INFO",
            "catchup",
            "Processing missed candles",
            f"Replaying {len(missed)} closed candles after {last_processed.isoformat()} because of {reason}.",
            payload={"reason": reason},
            trade_day=candle_trade_day(latest_closed),
        )
        for _, row in missed.iterrows():
            candle = {
                "open_time": pd.to_datetime(row["open_time"], utc=True),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            await self._handle_closed_candle(candle, source="recovery")

    async def _recover_and_reconcile(self, reason: str) -> None:
        account = await self.exchange.account_info()
        position = await self.exchange.position_risk(self.config.symbol)
        open_orders = await self.exchange.open_orders(self.config.symbol)
        open_algo_orders = await self.exchange.open_algo_orders(self.config.symbol)

        self.store.set_state(
            "last_account_snapshot",
            {
                "reason": reason,
                "ts": utc_iso(),
                "available_balance": float(account["availableBalance"]),
                "wallet_balance": float(account.get("walletBalance", account.get("totalWalletBalance", 0.0))),
            },
        )

        active_position = self.store.get_state("open_position")
        if active_position and float(position.get("positionAmt", "0")) == 0.0:
            self.store.delete_state("open_position")
            self.store.log_event("WARNING", "recovery", "Local position cleared", "Local open position existed but exchange has no live position.")

        if float(position.get("positionAmt", "0")) < 0:
            entry_price = abs(float(position.get("entryPrice", "0")))
            qty = abs(float(position.get("positionAmt", "0")))
            if not active_position:
                recovered = ManagedPosition(
                    symbol=self.config.symbol,
                    side="SHORT",
                    signal_id="recovered",
                    route="recovered_live_position",
                    regime="unknown",
                    subtype="unknown",
                    entry_time=utc_now(),
                    entry_price=entry_price,
                    qty_initial=qty,
                    qty_remaining=qty,
                    stop_price_initial=entry_price * 1.003,
                    stop_price_current=entry_price * 1.003,
                    partial_taken=False,
                    trail_active=False,
                    bars_since_entry=0,
                    best_low=entry_price,
                    best_high=entry_price,
                    entry_fee_usd=qty * entry_price * (TAKER_FEE_PCT / 100.0),
                    entry_order_client_id="recovered",
                )
                self.store.set_state("open_position", recovered.to_payload())
                self.store.log_event(
                    "WARNING",
                    "recovery",
                    "Recovered live position",
                    "Startup found an exchange short position without local state. Local position state was reconstructed.",
                    payload=recovered.to_payload(),
                )

        if open_orders:
            for order in open_orders:
                client_order_id = order.get("clientOrderId") or order.get("origClientOrderId") or order.get("c")
                if not client_order_id:
                    continue
                self.store.upsert_order(
                    client_order_id=client_order_id,
                    order_id=str(order.get("orderId")),
                    purpose=self._purpose_from_client_order_id(client_order_id),
                    status=str(order.get("status", "NEW")),
                    symbol=self.config.symbol,
                    side=str(order.get("side", "")),
                    type_=str(order.get("type", "")),
                    qty=float(order.get("origQty", 0.0)),
                    price=float(order.get("price", 0.0)) if float(order.get("price", 0.0)) > 0 else None,
                    stop_price=float(order.get("stopPrice", 0.0)) if float(order.get("stopPrice", 0.0)) > 0 else None,
                    reduce_only=bool(order.get("reduceOnly", False)),
                    payload=order,
                )

        if open_algo_orders:
            for order in open_algo_orders:
                client_algo_id = order.get("clientAlgoId")
                if not client_algo_id:
                    continue
                qty_raw = str(order.get("quantity", "")).strip()
                price_raw = str(order.get("price", "")).strip()
                trigger_raw = str(order.get("triggerPrice", "")).strip()
                self.store.upsert_order(
                    client_order_id=client_algo_id,
                    order_id=str(order.get("algoId")),
                    purpose="protective_algo",
                    status=str(order.get("algoStatus", "NEW")),
                    symbol=self.config.symbol,
                    side=str(order.get("side", "")),
                    type_=str(order.get("orderType", "")),
                    qty=float(qty_raw) if qty_raw else 0.0,
                    price=float(price_raw) if price_raw and float(price_raw) > 0 else None,
                    stop_price=float(trigger_raw) if trigger_raw and float(trigger_raw) > 0 else None,
                    reduce_only=bool(order.get("reduceOnly", False)),
                    payload=order,
                )

        active_signal = self.store.get_state("active_signal")
        entry_orders = [o for o in open_orders if not bool(o.get("reduceOnly", False)) and str(o.get("side", "")).upper() == "SELL"]
        if not active_signal and entry_orders:
            restored = entry_orders[0]
            self.store.set_state(
                "active_signal",
                {
                    "signal_id": f"recovered-{restored.get('clientOrderId', 'entry')}",
                    "route": "recovered_open_entry_order",
                    "regime": "unknown",
                    "subtype": "unknown",
                    "candidate_score": None,
                    "entry_order_client_id": restored.get("clientOrderId"),
                    "entry_order_type": restored.get("type", "LIMIT"),
                    "entry_qty": float(restored.get("origQty", 0.0)),
                    "entry_price": float(restored.get("price", 0.0)) if float(restored.get("price", 0.0)) > 0 else None,
                    "stop_price": None,
                    "expires_at": None,
                    "placed_at": utc_iso(),
                    "plan": {"recovered": True, "order": restored},
                },
            )
        if active_signal and not entry_orders and float(position.get("positionAmt", "0") or 0.0) == 0.0:
            self.store.delete_state("active_signal")

        if reason != "periodic":
            await self.notifier.send_event(
                (
                    f"<b>Bot recovery complete</b>\n"
                    f"Reason: <code>{reason}</code>\n"
                    f"Open orders: <code>{len(open_orders)}</code>\n"
                    f"Open algo orders: <code>{len(open_algo_orders)}</code>\n"
                    f"Position amt: <code>{position.get('positionAmt', '0')}</code>\n"
                    f"Available balance: <code>{float(account['availableBalance']):.2f}</code>"
                ),
                dedupe_key=f"recovery:{reason}:{utc_now().strftime('%Y%m%d%H%M')}",
                silent=True,
            )

    async def _market_stream_loop(self) -> None:
        url = f"{self.config.binance.ws_stream_base_url}/ws/{self.config.symbol.lower()}@kline_1m"
        while not self.stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, open_timeout=20) as ws:
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("e") != "kline":
                            continue
                        kline = message.get("k", {})
                        if not kline.get("x", False):
                            continue
                        candle = {
                            "open_time": pd.to_datetime(int(kline["t"]), unit="ms", utc=True),
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"]),
                        }
                        await self._handle_closed_candle(candle, source="ws")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Market stream loop failed: %s", exc)
                self.store.log_event("ERROR", "market_ws", "Market websocket disconnected", str(exc))
                await self.notifier.send_event(
                    f"<b>Market websocket disconnected</b>\n<code>{str(exc)[:300]}</code>",
                    dedupe_key=f"market-ws-error:{utc_now().strftime('%Y%m%d%H%M')}",
                )
                try:
                    await self._process_catchup_candles("market_stream_reconnect")
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_exc:  # noqa: BLE001
                    LOGGER.exception("Market reconnect catchup failed: %s", recovery_exc)
                    self.store.log_event("ERROR", "market_ws_recovery", "Market reconnect catchup failed", str(recovery_exc))
                    await self.notifier.send_event(
                        f"<b>Market reconnect catchup failed</b>\n<code>{str(recovery_exc)[:300]}</code>",
                        dedupe_key=f"market-ws-recovery:{utc_now().strftime('%Y%m%d%H%M')}",
                        silent=True,
                    )
                await asyncio.sleep(self.config.market_stream_reconnect_backoff_seconds)

    async def _user_stream_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.listen_key:
                await asyncio.sleep(1)
                continue
            url = f"{self.config.binance.ws_stream_base_url}/ws/{self.listen_key}"
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, open_timeout=20) as ws:
                    async for raw in ws:
                        message = json.loads(raw)
                        await self._handle_user_stream_message(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("User stream loop failed: %s", exc)
                self.store.log_event("ERROR", "user_ws", "User websocket disconnected", str(exc))
                await self.notifier.send_event(
                    f"<b>User websocket disconnected</b>\n<code>{str(exc)[:300]}</code>",
                    dedupe_key=f"user-ws-error:{utc_now().strftime('%Y%m%d%H%M')}",
                )
                try:
                    await self._recover_and_reconcile("user_stream_reconnect")
                except asyncio.CancelledError:
                    raise
                except Exception as recovery_exc:  # noqa: BLE001
                    LOGGER.exception("User stream recovery failed: %s", recovery_exc)
                    self.store.log_event("ERROR", "user_ws_recovery", "User stream recovery failed", str(recovery_exc))
                    await self.notifier.send_event(
                        f"<b>User stream recovery failed</b>\n<code>{str(recovery_exc)[:300]}</code>",
                        dedupe_key=f"user-ws-recovery:{utc_now().strftime('%Y%m%d%H%M')}",
                        silent=True,
                    )
                await asyncio.sleep(self.config.user_stream_reconnect_backoff_seconds)

    async def _listen_key_keepalive_loop(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.config.listen_key_keepalive_seconds)
            if self.listen_key:
                try:
                    await self.exchange.keepalive_listen_key(self.listen_key)
                    self.store.set_state("listen_key", {"listen_key": self.listen_key, "ts": utc_iso()})
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Listen key keepalive failed: %s", exc)
                    self.store.log_event("ERROR", "listen_key_keepalive", "Listen key keepalive failed", str(exc))
                    await self.notifier.send_event(
                        f"<b>Listen key keepalive failed</b>\n<code>{str(exc)[:300]}</code>",
                        dedupe_key=f"listen-key-keepalive:{utc_now().strftime('%Y%m%d%H%M')}",
                        silent=True,
                    )

    async def _reconcile_loop(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(self.config.reconcile_interval_seconds)
            try:
                await self._recover_and_reconcile("periodic")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Periodic reconcile failed: %s", exc)
                self.store.log_event("ERROR", "periodic_reconcile", "Periodic reconcile failed", str(exc))
                await self.notifier.send_event(
                    f"<b>Periodic reconcile failed</b>\n<code>{str(exc)[:300]}</code>",
                    dedupe_key=f"periodic-reconcile:{utc_now().strftime('%Y%m%d%H%M')}",
                    silent=True,
                )
                await asyncio.sleep(5)

    async def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            now = utc_now()
            self.store.set_state(
                "heartbeat",
                {
                    "ts": utc_iso(now),
                    "symbol": self.config.symbol,
                    "baseline": "Tier 2.87",
                    "status": "running",
                },
            )
            if (
                self._last_status_ping_at is None
                or (now - self._last_status_ping_at).total_seconds() >= self.config.telegram_status_interval_seconds
            ):
                await self.notifier.send_event(
                    (
                        f"<b>Bot heartbeat</b>\n"
                        f"Symbol: <code>{self.config.symbol}</code>\n"
                        f"Baseline: <code>Tier 2.87</code>\n"
                        f"Status: <code>running</code>\n"
                        f"UTC: <code>{utc_iso(now)}</code>"
                    ),
                    dedupe_key=f"heartbeat:{now.strftime('%Y%m%d%H')}",
                    silent=True,
                )
                self._last_status_ping_at = now
            await asyncio.sleep(self.config.heartbeat_interval_seconds)

    async def _daily_report_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                now = utc_now()
                current_day = candle_trade_day(now)

                if now.time() >= time(12, 45) and self._last_pre_report_day != current_day:
                    await self._send_pre_day_report(current_day)
                    self._last_pre_report_day = current_day

                if now.time() >= time(16, 5) and self._last_post_report_day != current_day:
                    await self._send_post_day_report(current_day)
                    self._last_post_report_day = current_day

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Daily report loop failed: %s", exc)
                self.store.log_event("ERROR", "daily_report_loop", "Daily report loop failed", str(exc))
                error_text = str(exc)[:300]
                await self.notifier.send_event(
                    f"<b>Daily report loop failed</b>\n<code>{error_text}</code>",
                    dedupe_key=f"daily-report-loop:{error_text}:{utc_now().strftime('%Y%m%d%H')}",
                    silent=True,
                )
                await asyncio.sleep(30)

    async def _handle_closed_candle(self, candle: dict[str, Any], source: str) -> None:
        self.strategy.append_closed_candle(candle)
        self.store.set_state("last_processed_candle", {"open_time": pd.to_datetime(candle["open_time"], utc=True).isoformat(), "source": source})

        await self._check_expired_entry_orders(pd.to_datetime(candle["open_time"], utc=True))
        await self._manage_open_position_on_candle(candle)

        open_position = self.store.get_state("open_position")
        live_open_orders = self.store.open_orders()
        if open_position or any(order["purpose"] == "entry" for order in live_open_orders):
            return

        decision = self.strategy.evaluate_latest()
        trade_day = candle_trade_day(pd.to_datetime(candle["open_time"], utc=True))
        self.store.log_event(
            "INFO",
            "strategy_decision",
            decision.title,
            decision.explanation,
            payload=decision.to_payload(),
            trade_day=trade_day,
        )

        if decision.status == "accepted" and decision.entry_plan is not None:
            await self.notifier.send_event(
                (
                    f"<b>Possible short setup</b>\n"
                    f"Time: <code>{pd.to_datetime(candle['open_time'], utc=True).isoformat()}</code>\n"
                    f"Route: <code>{decision.entry_plan.route}</code>\n"
                    f"Subtype: <code>{decision.entry_plan.subtype}</code>\n"
                    f"Score: <code>{decision.entry_plan.candidate_score:.3f}</code>\n"
                    f"Plan: <code>{decision.entry_plan.order_type}</code>"
                ),
                dedupe_key=f"preview:{decision.signal_id}",
            )
            await self._submit_entry_plan(decision.entry_plan)

    async def _submit_entry_plan(self, plan: EntryPlan) -> None:
        assert self.symbol_rules is not None
        reference_price = float(plan.price) if plan.price is not None else float(self.strategy.candles_1m.iloc[-1]["close"])
        qty, sizing = await self.exchange.compute_full_margin_short_qty(
            reference_price=reference_price,
            leverage=self.config.leverage,
            safety_buffer=self.config.order_margin_safety_buffer,
            rules=self.symbol_rules,
        )

        price = self.exchange.round_price(plan.price, self.symbol_rules) if plan.price is not None else None
        stop_price = self.exchange.round_price(plan.stop_price, self.symbol_rules)
        client_order_id = order_id("entry")
        order_type = "LIMIT" if plan.order_type == "LIMIT" else "MARKET"
        tif = "GTC" if order_type == "LIMIT" else None

        order = await self.exchange.place_order(
            symbol=self.config.symbol,
            side="SELL",
            type_=order_type,
            quantity=qty,
            client_order_id=client_order_id,
            price=price,
            time_in_force=tif,
        )

        self.store.upsert_order(
            client_order_id=client_order_id,
            order_id=str(order.get("orderId")),
            purpose="entry",
            status=str(order.get("status", "NEW")),
            symbol=self.config.symbol,
            side="SELL",
            type_=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            reduce_only=False,
            payload={"plan": plan.to_payload(), "order": order, "sizing": sizing},
        )
        self.store.set_state(
            "active_signal",
            {
                "signal_id": plan.signal_id,
                "route": plan.route,
                "regime": plan.regime,
                "subtype": plan.subtype,
                "candidate_score": plan.candidate_score,
                "entry_order_client_id": client_order_id,
                "entry_order_type": order_type,
                "entry_qty": qty,
                "entry_price": price,
                "stop_price": stop_price,
                "expires_at": utc_iso(plan.signal_time + timedelta(seconds=plan.ttl_seconds)) if plan.ttl_seconds else None,
                "placed_at": utc_iso(),
                "plan": plan.to_payload(),
            },
        )
        self.store.log_event(
            "INFO",
            "order_submit",
            "Entry order submitted",
            f"{order_type} entry order submitted for {self.config.symbol}.",
            payload={"plan": plan.to_payload(), "order": order, "sizing": sizing},
            trade_day=candle_trade_day(plan.signal_time),
        )
        await self.notifier.send_event(
            (
                f"<b>Entry order submitted</b>\n"
                f"Route: <code>{plan.route}</code>\n"
                f"Type: <code>{order_type}</code>\n"
                f"Qty: <code>{qty}</code>\n"
                f"Price: <code>{price if price is not None else 'MARKET'}</code>\n"
                f"Stop: <code>{stop_price}</code>"
            ),
            dedupe_key=f"entry-submit:{plan.signal_id}",
        )

        if str(order.get("status")) == "FILLED":
            avg_price = float(order.get("avgPrice") or order.get("price") or reference_price)
            await self._open_position_from_fill(
                signal_id=plan.signal_id,
                route=plan.route,
                regime=plan.regime,
                subtype=plan.subtype,
                entry_mode=order_type,
                client_order_id=client_order_id,
                entry_price=avg_price,
                qty=qty,
                stop_price=stop_price,
                signal_time=plan.signal_time,
            )

    async def _open_position_from_fill(
        self,
        *,
        signal_id: str,
        route: str,
        regime: str,
        subtype: str,
        entry_mode: str,
        client_order_id: str,
        entry_price: float,
        qty: float,
        stop_price: float,
        signal_time: datetime,
    ) -> None:
        fee_pct = MAKER_FEE_PCT if entry_mode == "LIMIT" else TAKER_FEE_PCT
        position = ManagedPosition(
            symbol=self.config.symbol,
            side="SHORT",
            signal_id=signal_id,
            route=route,
            regime=regime,
            subtype=subtype,
            entry_time=utc_now(),
            entry_price=entry_price,
            qty_initial=qty,
            qty_remaining=qty,
            stop_price_initial=stop_price,
            stop_price_current=stop_price,
            partial_taken=False,
            trail_active=False,
            bars_since_entry=0,
            best_low=entry_price,
            best_high=entry_price,
            entry_fee_usd=qty * entry_price * (fee_pct / 100.0),
            entry_order_client_id=client_order_id,
            entry_mode=entry_mode,
            metadata={"signal_time": signal_time.isoformat()},
        )
        self.store.set_state("open_position", position.to_payload())
        await self._place_or_replace_protective_stop(position)
        self.store.log_event(
            "INFO",
            "position_open",
            "Short position opened",
            f"Opened {self.config.symbol} short position.",
            payload=position.to_payload(),
            trade_day=candle_trade_day(position.entry_time),
        )
        await self.notifier.send_event(
            (
                f"<b>Position opened</b>\n"
                f"Route: <code>{route}</code>\n"
                f"Qty: <code>{qty}</code>\n"
                f"Entry: <code>{entry_price:.4f}</code>\n"
                f"Stop: <code>{stop_price:.4f}</code>"
            ),
            dedupe_key=f"position-open:{signal_id}",
        )

    async def _place_or_replace_protective_stop(self, position: ManagedPosition) -> None:
        if position.protective_stop_client_id:
            with suppress(Exception):
                await self.exchange.cancel_algo_order(client_algo_id=position.protective_stop_client_id)

        stop_client_id = order_id("stop")
        stop_price = self.exchange.round_price(position.stop_price_current, self.symbol_rules)
        order = await self.exchange.algo_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="STOP_MARKET",
            client_algo_id=stop_client_id,
            trigger_price=stop_price,
            close_position=True,
        )
        position.protective_stop_client_id = stop_client_id
        self.store.set_state("open_position", position.to_payload())
        self.store.upsert_order(
            client_order_id=stop_client_id,
            order_id=str(order.get("algoId")),
            purpose="protective_stop",
            status=str(order.get("algoStatus", "NEW")),
            symbol=self.config.symbol,
            side="BUY",
            type_="STOP_MARKET",
            qty=position.qty_remaining,
            price=None,
            stop_price=stop_price,
            reduce_only=True,
            payload=order,
        )

    async def _check_expired_entry_orders(self, candle_time: datetime) -> None:
        signal = self.store.get_state("active_signal")
        if not signal or not signal.get("expires_at"):
            return
        expires_at = pd.to_datetime(signal["expires_at"], utc=True)
        if candle_time < expires_at:
            return
        client_order_id = signal["entry_order_client_id"]
        with suppress(Exception):
            await self.exchange.cancel_order(self.config.symbol, client_order_id)
        self.store.log_event(
            "INFO",
            "order_cancel",
            "Entry order expired",
            f"Pending limit order {client_order_id} expired and was canceled.",
            payload=signal,
            trade_day=signal["plan"]["signal_time"][:10],
        )
        self.store.delete_state("active_signal")
        await self.notifier.send_event(
            (
                f"<b>Entry order canceled</b>\n"
                f"Reason: <code>expired</code>\n"
                f"ClientOrderId: <code>{client_order_id}</code>"
            ),
            dedupe_key=f"entry-expired:{client_order_id}",
            silent=True,
        )

    async def _manage_open_position_on_candle(self, candle: dict[str, Any]) -> None:
        raw = self.store.get_state("open_position")
        if not raw:
            return

        position = self._load_position_state(raw)
        candle_time = pd.to_datetime(candle["open_time"], utc=True)
        position.bars_since_entry += 1
        position.best_low = min(position.best_low, float(candle["low"]))
        position.best_high = max(position.best_high, float(candle["high"]))

        risk_per_unit = position.stop_price_initial - position.entry_price
        if risk_per_unit <= 0:
            return

        if position.bars_since_entry == FAST_FAIL_WINDOW_BARS:
            favorable_pct = (position.entry_price - position.best_low) / position.entry_price * 100.0
            adverse_pct = (position.best_high - position.entry_price) / position.entry_price * 100.0
            if favorable_pct < FAST_FAIL_MIN_FAVORABLE_PCT and adverse_pct >= FAST_FAIL_MAX_ADVERSE_PCT:
                await self._market_close_position(position, "fast_fail")
                return

        target_2r_price = position.entry_price - TAKE_PARTIAL_AT_R * risk_per_unit
        if not position.partial_taken and float(candle["low"]) <= target_2r_price:
            partial_qty = position.qty_remaining * PARTIAL_CLOSE_SIZE
            await self._market_close_position(position, "partial_take_profit", qty=partial_qty)
            raw = self.store.get_state("open_position")
            if not raw:
                return
            position = ManagedPosition(**raw)
            position.partial_taken = True
            position.stop_price_current = min(
                position.stop_price_current,
                position.entry_price * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0),
            )
            await self._place_or_replace_protective_stop(position)

        favorable_r_now = (position.entry_price - position.best_low) / risk_per_unit
        if favorable_r_now >= TRAIL_AFTER_R:
            position.trail_active = True

        if position.trail_active:
            recent = self.strategy.candles_1m.tail(TRAIL_LOOKBACK_BARS)
            recent_high = float(recent["high"].max())
            candidate_stop = recent_high * (1.0 + TRAIL_BUFFER_PCT / 100.0)
            new_stop = min(position.stop_price_current, candidate_stop)
            if new_stop < position.stop_price_current:
                position.stop_price_current = new_stop
                await self._place_or_replace_protective_stop(position)

        self.store.set_state("open_position", position.to_payload())

    async def _market_close_position(self, position: ManagedPosition, reason: str, qty: float | None = None) -> None:
        close_qty = qty or position.qty_remaining
        client_order_id = order_id("exit")
        order = await self.exchange.place_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="MARKET",
            quantity=close_qty,
            client_order_id=client_order_id,
            reduce_only=True,
        )
        exit_price = float(order.get("avgPrice") or order.get("price") or self.strategy.candles_1m.iloc[-1]["close"])
        fee_exit = close_qty * exit_price * (TAKER_FEE_PCT / 100.0)
        pnl = (position.entry_price - exit_price) * close_qty - fee_exit

        if close_qty < position.qty_remaining:
            position.qty_remaining -= close_qty
            self.store.set_state("open_position", position.to_payload())
            self.store.log_event(
                "INFO",
                "partial_exit",
                "Partial exit filled",
                f"Partial reduce-only market exit filled because {reason}.",
                payload={"qty": close_qty, "exit_price": exit_price, "reason": reason},
                trade_day=candle_trade_day(position.entry_time),
            )
            return

        trade = ClosedTrade(
            signal_id=position.signal_id,
            route=position.route,
            regime=position.regime,
            subtype=position.subtype,
            entry_time=position.entry_time,
            exit_time=utc_now(),
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=close_qty,
            pnl_usd=pnl - position.entry_fee_usd,
            exit_reason=reason,
            fees_usd=position.entry_fee_usd + fee_exit,
            payload={"order": order, "position": position.to_payload()},
        )
        self.store.save_trade(
            trade_id=f"{trade.signal_id}:{trade.exit_time.isoformat()}",
            opened_at=trade.entry_time.isoformat(),
            closed_at=trade.exit_time.isoformat(),
            symbol=self.config.symbol,
            side="SHORT",
            qty=trade.qty,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            pnl_usd=trade.pnl_usd,
            exit_reason=trade.exit_reason,
            payload=trade.to_payload(),
        )
        if position.protective_stop_client_id:
            with suppress(Exception):
                await self.exchange.cancel_algo_order(client_algo_id=position.protective_stop_client_id)
        self.store.delete_state("open_position")
        self.store.delete_state("active_signal")
        self.store.log_event(
            "INFO",
            "position_close",
            "Position closed",
            f"Position closed by {reason}.",
            payload=trade.to_payload(),
            trade_day=candle_trade_day(trade.entry_time),
        )
        await self.notifier.send_event(
            (
                f"<b>Position closed</b>\n"
                f"Reason: <code>{reason}</code>\n"
                f"Entry: <code>{trade.entry_price:.4f}</code>\n"
                f"Exit: <code>{trade.exit_price:.4f}</code>\n"
                f"PnL: <code>{trade.pnl_usd:.4f} USD</code>"
            ),
            dedupe_key=f"position-close:{trade.signal_id}:{reason}:{trade.exit_time.strftime('%Y%m%d%H%M')}",
        )

    async def _handle_user_stream_message(self, message: dict[str, Any]) -> None:
        event_type = message.get("e")
        if event_type == "listenKeyExpired":
            self.store.log_event("ERROR", "listen_key", "Listen key expired", "User stream listen key expired.")
            self.listen_key = await self.exchange.create_listen_key()
            return

        if event_type != "ORDER_TRADE_UPDATE":
            return

        order_data = message.get("o", {})
        client_order_id = str(order_data.get("c", ""))
        order_status = str(order_data.get("X", ""))
        order_type = str(order_data.get("o", ""))
        side = str(order_data.get("S", ""))
        exec_type = str(order_data.get("x", ""))
        avg_price = float(order_data.get("ap", "0") or 0.0)
        filled_qty = float(order_data.get("z", "0") or 0.0)
        orig_qty = float(order_data.get("q", "0") or 0.0)
        stop_price = float(order_data.get("sp", "0") or 0.0)
        reduce_only = bool(order_data.get("R", False))
        purpose = self._purpose_from_client_order_id(client_order_id)

        self.store.upsert_order(
            client_order_id=client_order_id,
            order_id=str(order_data.get("i", "")),
            purpose=purpose,
            status=order_status,
            symbol=self.config.symbol,
            side=side,
            type_=order_type,
            qty=orig_qty,
            price=avg_price if avg_price > 0 else None,
            stop_price=stop_price if stop_price > 0 else None,
            reduce_only=reduce_only,
            payload=message,
        )

        if purpose == "entry" and order_status == "FILLED":
            signal = self.store.get_state("active_signal")
            if signal:
                await self._open_position_from_fill(
                    signal_id=signal["signal_id"],
                    route=signal["route"],
                    regime=signal["regime"],
                    subtype=signal["subtype"],
                    entry_mode=signal["entry_order_type"],
                    client_order_id=client_order_id,
                    entry_price=avg_price,
                    qty=filled_qty,
                    stop_price=float(signal["stop_price"]),
                    signal_time=pd.to_datetime(signal["plan"]["signal_time"], utc=True),
                )

        if purpose == "protective_stop" and order_status == "FILLED":
            raw = self.store.get_state("open_position")
            if raw:
                position = self._load_position_state(raw)
                trade = ClosedTrade(
                    signal_id=position.signal_id,
                    route=position.route,
                    regime=position.regime,
                    subtype=position.subtype,
                    entry_time=position.entry_time,
                    exit_time=utc_now(),
                    entry_price=position.entry_price,
                    exit_price=avg_price if avg_price > 0 else position.stop_price_current,
                    qty=position.qty_remaining,
                    pnl_usd=(position.entry_price - (avg_price if avg_price > 0 else position.stop_price_current)) * position.qty_remaining - position.entry_fee_usd,
                    exit_reason="stop_loss",
                    fees_usd=position.entry_fee_usd,
                    payload={"order_update": message, "position": position.to_payload()},
                )
                self.store.save_trade(
                    trade_id=f"{trade.signal_id}:{trade.exit_time.isoformat()}",
                    opened_at=trade.entry_time.isoformat(),
                    closed_at=trade.exit_time.isoformat(),
                    symbol=self.config.symbol,
                    side="SHORT",
                    qty=trade.qty,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    pnl_usd=trade.pnl_usd,
                    exit_reason=trade.exit_reason,
                    payload=trade.to_payload(),
                )
                self.store.delete_state("open_position")
                self.store.delete_state("active_signal")
                await self.notifier.send_event(
                    (
                        f"<b>Protective stop filled</b>\n"
                        f"Exit: <code>{trade.exit_price:.4f}</code>\n"
                        f"PnL: <code>{trade.pnl_usd:.4f} USD</code>"
                    ),
                    dedupe_key=f"protective-stop:{trade.signal_id}:{trade.exit_time.strftime('%Y%m%d%H%M')}",
                )

    @staticmethod
    def _top_counts(items: list[str], limit: int = 3) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for item in items:
            if not item:
                continue
            counts[item] = counts.get(item, 0) + 1
        return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]

    @staticmethod
    def _format_top_counts(items: list[tuple[str, int]]) -> str:
        if not items:
            return "none"
        return ", ".join(f"{name} ({count})" for name, count in items)

    @staticmethod
    def _safe_event_payload(event: dict[str, Any]) -> dict[str, Any]:
        payload_raw = event.get("payload")
        if isinstance(payload_raw, str):
            try:
                return json.loads(payload_raw)
            except Exception:  # noqa: BLE001
                return {}
        if isinstance(payload_raw, dict):
            return payload_raw
        return {}

    def _current_market_context(self) -> dict[str, Any]:
        if self.strategy.daily_1d.empty:
            return {
                "trend_regime": "unknown",
                "vol_regime": "unknown",
                "ret_30d_pct": None,
                "ret_60d_pct": None,
                "close_vs_sma60_pct": None,
            }
        market_regime_frame = _build_market_regime_from_daily(self.strategy.daily_1d)
        latest = market_regime_frame.iloc[-1]
        return {
            "trend_regime": str(latest.get("trend_regime", "unknown")),
            "vol_regime": str(latest.get("vol_regime", "unknown")),
            "ret_30d_pct": float(latest["ret_30d_pct"]) if pd.notna(latest.get("ret_30d_pct")) else None,
            "ret_60d_pct": float(latest["ret_60d_pct"]) if pd.notna(latest.get("ret_60d_pct")) else None,
            "close_vs_sma60_pct": float(latest["close_vs_sma60_pct"]) if pd.notna(latest.get("close_vs_sma60_pct")) else None,
        }

    def _pre_day_expectation_text(self, market_ctx: dict[str, Any]) -> str:
        trend = market_ctx["trend_regime"]
        vol = market_ctx["vol_regime"]
        if trend == "strong_bear":
            return (
                "Favorable short backdrop. "
                + ("High volatility improves chance of a trade." if vol == "high_vol" else "Lower volatility may reduce trade frequency.")
            )
        if trend == "weak_bear":
            return "Moderately supportive short backdrop. A trade is possible, but less likely than strong_bear."
        if trend == "transition":
            return "Mixed backdrop. A trade is possible, but expectation should be lower."
        return "Unfavorable backdrop for this short model. A trade is less likely unless intraday structure becomes exceptional."

    @staticmethod
    def _route_availability_summary(market_ctx: dict[str, Any]) -> dict[str, str]:
        trend = str(market_ctx.get("trend_regime", "unknown"))
        vol = str(market_ctx.get("vol_regime", "unknown"))

        if trend in {"strong_bear", "weak_bear"}:
            core = f"enabled ({trend})"
        else:
            core = f"disabled (needs strong_bear or weak_bear, got {trend})"

        if trend == "strong_bear":
            satellite = "enabled (strong_bear)"
        else:
            satellite = f"disabled (needs strong_bear, got {trend})"

        if trend == "strong_bear" and vol == "high_vol":
            expansion = "enabled (strong_bear + high_vol)"
        else:
            expansion = f"disabled (needs strong_bear + high_vol, got {trend} + {vol})"

        return {"core": core, "satellite": satellite, "expansion": expansion}

    @staticmethod
    def _market_verdict_text(market_ctx: dict[str, Any]) -> str:
        trend = str(market_ctx.get("trend_regime", "unknown"))
        vol = str(market_ctx.get("vol_regime", "unknown"))
        if trend == "strong_bear" and vol == "high_vol":
            return "Day backdrop was strongly aligned with the short model."
        if trend in {"strong_bear", "weak_bear"}:
            return "Day backdrop was at least partially supportive for the short model."
        if trend in {"strong_bull", "weak_bull"}:
            return "Day backdrop was bullish, so active short routes were structurally restricted."
        return "Day backdrop was mixed, so only unusually strong intraday structures could pass."

    @staticmethod
    def _candidate_time_text(payload: dict[str, Any]) -> str:
        for key in ("candle_time",):
            raw = payload.get(key)
            if raw:
                return str(raw)
        trigger = payload.get("trigger_snapshot") or {}
        if trigger.get("open_time"):
            return str(trigger["open_time"])
        return "unknown"

    def _explain_candidate_block(self, payload: dict[str, Any], market_ctx: dict[str, Any]) -> str:
        regime = str(payload.get("regime") or payload.get("predicted_regime") or "unknown")
        subtype = str(payload.get("subtype") or "unknown")
        score = payload.get("candidate_score")
        score_text = f"{float(score):.3f}" if score is not None else "n/a"
        trend = str(market_ctx.get("trend_regime", "unknown"))
        vol = str(market_ctx.get("vol_regime", "unknown"))
        trigger = payload.get("trigger_snapshot") or {}

        reasons: list[str] = []
        if regime == "near_entry_breakdown_noise":
            if trend not in {"strong_bear", "weak_bear"}:
                reasons.append(f"core route disabled by {trend} day")
            else:
                reasons.append(f"core subtype {subtype} did not pass live routing")
        elif regime == "mixed_transition":
            if trend != "strong_bear":
                reasons.append(f"satellite route disabled by {trend} day")
            elif subtype != "mild_noise_v1_live":
                reasons.append(f"satellite only accepts mild_noise_v1_live, got {subtype}")
            else:
                reasons.append("satellite micro filters rejected the setup")
        elif regime == "other_non_target":
            if subtype != "mild_noise_v1_live":
                reasons.append(f"expansion only accepts mild_noise_v1_live, got {subtype}")
            if trend != "strong_bear":
                reasons.append(f"expansion disabled by {trend} day")
            if vol != "high_vol":
                reasons.append(f"expansion needs high_vol, got {vol}")
            if score is not None and float(score) < 0.84:
                reasons.append(f"score {score_text} below 0.840")
            if trigger:
                checks = [
                    ("move_1m_pct", 0.25, "move_1m"),
                    ("move_3m_pct", 0.28, "move_3m"),
                    ("move_5m_pct", 0.55, "move_5m"),
                    ("relative_volume_1m", 1.50, "rel_vol"),
                    ("dist_1h_high_pct", 0.08, "dist_1h"),
                    ("bounce_60m_pct", 0.80, "bounce_60m"),
                ]
                for field, limit, label in checks:
                    raw_value = trigger.get(field)
                    if raw_value is None:
                        continue
                    value = float(raw_value)
                    if value > limit:
                        reasons.append(f"{label} {value:.3f} > {limit:.2f}")
            if not reasons:
                reasons.append("expansion route filters rejected the setup")
        else:
            reasons.append("route filters rejected the setup")
        return "; ".join(reasons[:4])

    def _closest_candidate_summary(
        self,
        candidate_events: list[dict[str, Any]],
        market_ctx: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        candidate_payloads = [self._safe_event_payload(row) for row in candidate_events]
        scored_payloads = [payload for payload in candidate_payloads if payload.get("candidate_score") is not None]
        if not scored_payloads:
            return "none", None

        best_payload = max(scored_payloads, key=lambda payload: float(payload["candidate_score"]))
        time_text = self._candidate_time_text(best_payload)
        regime = str(best_payload.get("regime") or best_payload.get("predicted_regime") or "unknown")
        subtype = str(best_payload.get("subtype") or "unknown")
        score_text = self._fmt_score(float(best_payload["candidate_score"]))
        reason = self._explain_candidate_block(best_payload, market_ctx)
        summary = (
            f"{time_text} | {regime} | {subtype} | score {score_text}"
            + (f" | blocked because {reason}" if reason else "")
        )
        return summary, best_payload

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}%"

    @staticmethod
    def _fmt_score(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    async def _send_pre_day_report(self, trade_day: str) -> None:
        if self.store.get_daily_report(trade_day, "pre_day"):
            return
        latest_close = float(self.strategy.candles_1m.iloc[-1]["close"]) if not self.strategy.candles_1m.empty else float("nan")
        market_ctx = self._current_market_context()
        expectation = self._pre_day_expectation_text(market_ctx)
        route_status = self._route_availability_summary(market_ctx)
        markdown = (
            f"<b>Pre-day report</b>\n"
            f"Day: <code>{trade_day}</code>\n"
            f"Session: <code>{self.config.session_start_hour_utc}:00-{self.config.session_end_hour_utc}:00 UTC</code>\n"
            f"Baseline: <code>Tier 2.87</code>\n"
            f"Last close: <code>{latest_close:.4f}</code>\n"
            f"Trend regime: <code>{market_ctx['trend_regime']}</code>\n"
            f"Vol regime: <code>{market_ctx['vol_regime']}</code>\n"
            f"30d return: <code>{self._fmt_pct(market_ctx['ret_30d_pct'])}</code>\n"
            f"60d return: <code>{self._fmt_pct(market_ctx['ret_60d_pct'])}</code>\n"
            f"Close vs SMA60: <code>{self._fmt_pct(market_ctx['close_vs_sma60_pct'])}</code>\n"
            f"Core route: <code>{route_status['core']}</code>\n"
            f"Satellite route: <code>{route_status['satellite']}</code>\n"
            f"Expansion route: <code>{route_status['expansion']}</code>\n"
            f"Expectation: <code>{expectation}</code>"
        )
        self.store.save_daily_report(
            trade_day,
            "pre_day",
            markdown,
            {"last_close": latest_close, "market_ctx": market_ctx, "route_status": route_status, "expectation": expectation},
        )
        reports_dir = self.config.log_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{trade_day}_pre_day.md").write_text(markdown + "\n", encoding="utf-8")
        await self.notifier.send_report(markdown, dedupe_key=f"pre-day:{trade_day}")

    async def _send_post_day_report(self, trade_day: str) -> None:
        if self.store.get_daily_report(trade_day, "post_day"):
            return
        events = self.store.events_for_trade_day(trade_day)
        trades = self.store.trades_for_trade_day(trade_day)
        market_ctx = self._current_market_context()
        route_status = self._route_availability_summary(market_ctx)
        setup_events = [row for row in events if row["kind"] == "strategy_decision"]
        candidate_events = [
            row
            for row in setup_events
            if "no candidate" not in row["title"].lower() and "no new candidate" not in row["title"].lower()
        ]
        blocked = [row for row in candidate_events if "blocked" in row["title"].lower()]
        accepted = [row for row in candidate_events if "accepted" in row["title"].lower()]
        submits = [row for row in events if row["kind"] == "order_submit"]
        pnl = sum(float(row["pnl_usd"]) for row in trades)

        blocked_titles = [row["title"] for row in blocked]
        blocked_reasons = self._format_top_counts(self._top_counts(blocked_titles, limit=4))

        candidate_payloads = [self._safe_event_payload(row) for row in candidate_events]
        accepted_payloads = [self._safe_event_payload(row) for row in accepted]
        candidate_scores = [
            float(payload["candidate_score"])
            for payload in candidate_payloads
            if payload.get("candidate_score") is not None
        ]
        best_score = max(candidate_scores) if candidate_scores else None

        top_regimes = self._format_top_counts(
            self._top_counts(
                [str(payload.get("regime") or payload.get("predicted_regime") or "") for payload in candidate_payloads],
                limit=3,
            )
        )
        top_subtypes = self._format_top_counts(
            self._top_counts([str(payload.get("subtype") or "") for payload in candidate_payloads], limit=3)
        )
        top_routes = self._format_top_counts(
            self._top_counts([str(payload.get("route") or "") for payload in accepted_payloads], limit=3)
        )
        closest_candidate, closest_payload = self._closest_candidate_summary(candidate_events, market_ctx)
        market_verdict = self._market_verdict_text(market_ctx)

        if trades:
            no_trade_reason = "Trade day completed with at least one closed trade."
        elif submits:
            no_trade_reason = "Orders were submitted, but no trade closed during the session."
        elif accepted:
            no_trade_reason = "There were accepted setups, but no order was ultimately submitted or filled."
        elif blocked:
            no_trade_reason = (
                f"{market_verdict} Candidates existed, but routing blocked them. "
                f"Main blockers: {blocked_reasons}. Closest candidate: {closest_candidate}."
            )
        elif setup_events:
            no_trade_reason = f"{market_verdict} The bot evaluated the session, but did not find a new deduped candidate worth routing."
        else:
            no_trade_reason = "No meaningful strategy evaluations were recorded for the day."

        closeness = "Low"
        if trades or submits or accepted:
            closeness = "High"
        elif best_score is not None and best_score >= 0.90 and blocked:
            closeness = "Medium-high"
        elif best_score is not None and best_score >= 0.84 and blocked:
            closeness = "Medium"

        markdown = (
            f"<b>Post-day report</b>\n"
            f"Day: <code>{trade_day}</code>\n"
            f"Candles evaluated: <code>{len(setup_events)}</code>\n"
            f"Signal candidates: <code>{len(candidate_events)}</code>\n"
            f"Accepted setups: <code>{len(accepted)}</code>\n"
            f"Blocked setups: <code>{len(blocked)}</code>\n"
            f"Orders submitted: <code>{len(submits)}</code>\n"
            f"Trades closed: <code>{len(trades)}</code>\n"
            f"Net PnL: <code>{pnl:.4f} USD</code>\n"
            f"Best candidate score: <code>{self._fmt_score(best_score)}</code>\n"
            f"How close to a trade: <code>{closeness}</code>\n"
            f"Trend regime: <code>{market_ctx['trend_regime']}</code>\n"
            f"Vol regime: <code>{market_ctx['vol_regime']}</code>\n"
            f"Core route: <code>{route_status['core']}</code>\n"
            f"Satellite route: <code>{route_status['satellite']}</code>\n"
            f"Expansion route: <code>{route_status['expansion']}</code>\n"
            f"Market verdict: <code>{market_verdict}</code>\n"
            f"Closest candidate: <code>{closest_candidate}</code>\n"
            f"Top candidate regimes: <code>{top_regimes}</code>\n"
            f"Top candidate subtypes: <code>{top_subtypes}</code>\n"
            f"Accepted routes: <code>{top_routes}</code>\n"
            f"Main blocked reasons: <code>{blocked_reasons}</code>\n"
            f"No-trade summary: <code>{no_trade_reason}</code>"
        )
        self.store.save_daily_report(
            trade_day,
            "post_day",
            markdown,
            {
                "events": len(events),
                "candles_evaluated": len(setup_events),
                "signal_candidates": len(candidate_events),
                "accepted_setups": len(accepted),
                "blocked_setups": len(blocked),
                "orders_submitted": len(submits),
                "trades": trades,
                "best_candidate_score": best_score,
                "market_ctx": market_ctx,
                "route_status": route_status,
                "market_verdict": market_verdict,
                "closest_candidate": closest_candidate,
                "closest_candidate_payload": closest_payload,
                "top_candidate_regimes": top_regimes,
                "top_candidate_subtypes": top_subtypes,
                "accepted_routes": top_routes,
                "blocked_reasons": blocked_reasons,
                "no_trade_reason": no_trade_reason,
            },
        )
        reports_dir = self.config.log_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / f"{trade_day}_post_day.md").write_text(markdown + "\n", encoding="utf-8")
        await self.notifier.send_report(markdown, dedupe_key=f"post-day:{trade_day}")

    @staticmethod
    def _purpose_from_client_order_id(client_order_id: str) -> str:
        if "-entry-" in client_order_id:
            return "entry"
        if "-stop-" in client_order_id:
            return "protective_stop"
        if "-exit-" in client_order_id:
            return "exit"
        return "unknown"

    async def close_all_open_risk(self) -> None:
        await self.exchange.cancel_all_orders(self.config.symbol)
        await self.exchange.cancel_all_algo_orders(self.config.symbol)
        position = await self.exchange.position_risk(self.config.symbol)
        position_amt = float(position.get("positionAmt", "0") or 0.0)
        if position_amt < 0:
            qty = abs(position_amt)
            await self.exchange.place_order(
                symbol=self.config.symbol,
                side="BUY",
                type_="MARKET",
                quantity=qty,
                client_order_id=order_id("panic"),
                reduce_only=True,
            )
        self.store.delete_state("open_position")
        self.store.delete_state("active_signal")
        self.store.log_event("WARNING", "manual", "Close all open risk", "All open orders canceled and live short closed if present.")

    async def _wait_for_live_short_position(
        self,
        *,
        min_abs_qty: float,
        timeout_seconds: int = 12,
    ) -> dict[str, Any]:
        deadline = utc_now() + timedelta(seconds=timeout_seconds)
        latest: dict[str, Any] = {}
        while utc_now() < deadline:
            latest = await self.exchange.position_risk(self.config.symbol)
            position_amt = float(latest.get("positionAmt", "0") or 0.0)
            if position_amt < 0 and abs(position_amt) >= min_abs_qty:
                return latest
            await asyncio.sleep(1)
        raise ExchangeError("Entry order filled, but exchange short position was not visible in time.")

    async def smoke_market_bracket(self, test_notional_usd: float, stop_pct: float, tp_pct: float) -> dict[str, Any]:
        assert self.symbol_rules is not None
        await self.exchange.set_position_mode_one_way()
        await self.exchange.set_margin_type_cross(self.config.symbol)
        await self.exchange.change_leverage(self.config.symbol, self.config.leverage)

        mark = await self.exchange.mark_price(self.config.symbol)
        qty = self.exchange.round_quantity(test_notional_usd / mark, self.symbol_rules)
        if qty <= 0:
            raise ExchangeError("Smoke market quantity rounded to zero.")

        entry = await self.exchange.place_order(
            symbol=self.config.symbol,
            side="SELL",
            type_="MARKET",
            quantity=qty,
            client_order_id=order_id("smkmar"),
        )
        live_position = await self._wait_for_live_short_position(min_abs_qty=max(qty * 0.5, 0.001))
        avg_price = abs(float(live_position.get("entryPrice") or entry.get("avgPrice") or entry.get("price") or mark))
        stop_price = self.exchange.round_price(avg_price * (1.0 + stop_pct / 100.0), self.symbol_rules)
        take_profit = self.exchange.round_price(avg_price * (1.0 - tp_pct / 100.0), self.symbol_rules)

        stop_order = await self.exchange.algo_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="STOP_MARKET",
            client_algo_id=order_id("smkstop"),
            trigger_price=stop_price,
            close_position=True,
        )
        tp_order = await self.exchange.algo_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="TAKE_PROFIT_MARKET",
            client_algo_id=order_id("smktp"),
            trigger_price=take_profit,
            close_position=True,
        )
        return {"entry": entry, "stop_algo": stop_order, "take_profit_algo": tp_order}

    async def smoke_limit_bracket(
        self,
        test_notional_usd: float,
        entry_offset_pct: float,
        stop_pct: float,
        tp_pct: float,
        wait_seconds: int = 120,
    ) -> dict[str, Any]:
        assert self.symbol_rules is not None
        mark = await self.exchange.mark_price(self.config.symbol)
        qty = self.exchange.round_quantity(test_notional_usd / mark, self.symbol_rules)
        entry_price = self.exchange.round_price(mark * (1.0 + entry_offset_pct / 100.0), self.symbol_rules)
        entry = await self.exchange.place_order(
            symbol=self.config.symbol,
            side="SELL",
            type_="LIMIT",
            quantity=qty,
            client_order_id=order_id("smklim"),
            price=entry_price,
            time_in_force="GTC",
        )
        deadline = utc_now() + timedelta(seconds=wait_seconds)
        order_state = entry
        while utc_now() < deadline:
            await asyncio.sleep(5)
            order_state = await self.exchange.order(self.config.symbol, orig_client_order_id=entry["clientOrderId"])
            if str(order_state.get("status")) == "FILLED":
                break
        if str(order_state.get("status")) != "FILLED":
            await self.exchange.cancel_order(self.config.symbol, entry["clientOrderId"])
            return {"entry": entry, "status": "CANCELED_TIMEOUT"}

        live_position = await self._wait_for_live_short_position(min_abs_qty=max(qty * 0.5, 0.001))
        avg_price = abs(float(live_position.get("entryPrice") or order_state.get("avgPrice") or order_state.get("price") or entry_price))
        stop_price = self.exchange.round_price(avg_price * (1.0 + stop_pct / 100.0), self.symbol_rules)
        take_profit = self.exchange.round_price(avg_price * (1.0 - tp_pct / 100.0), self.symbol_rules)
        stop_order = await self.exchange.algo_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="STOP_MARKET",
            client_algo_id=order_id("smlstop"),
            trigger_price=stop_price,
            close_position=True,
        )
        tp_order = await self.exchange.algo_order(
            symbol=self.config.symbol,
            side="BUY",
            type_="TAKE_PROFIT_MARKET",
            client_algo_id=order_id("smltp"),
            trigger_price=take_profit,
            close_position=True,
        )
        return {"entry": order_state, "stop_algo": stop_order, "take_profit_algo": tp_order}
