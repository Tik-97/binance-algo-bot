from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import asdict
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import BinanceConfig
from .models import SymbolRules


class ExchangeError(RuntimeError):
    """Raised when Binance rejects a request."""


class BinanceFuturesClient:
    def __init__(self, config: BinanceConfig) -> None:
        self.config = config
        self._time_offset_ms = 0
        self._client = httpx.AsyncClient(
            base_url=self.config.rest_base_url,
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={"X-MBX-APIKEY": self.config.api_key},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError, ExchangeError)),
            reraise=True,
        ):
            with attempt:
                request_params = dict(params or {})
                if signed:
                    request_params["timestamp"] = int(time.time() * 1000) + int(self._time_offset_ms)
                    request_params["recvWindow"] = self.config.recv_window_ms
                    query = urlencode(request_params, doseq=True)
                    request_params["signature"] = hmac.new(
                        self.config.api_secret.encode("utf-8"),
                        query.encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                response = await self._client.request(method, path, params=request_params)
                if response.status_code >= 400:
                    if signed and response.status_code == 400 and "-1021" in response.text:
                        await self.sync_time()
                        raise ExchangeError(f"{method} {path} failed after time drift; retrying")
                    raise ExchangeError(f"{method} {path} failed: {response.status_code} {response.text}")
                data = response.json()
                if isinstance(data, dict) and "code" in data and data.get("code", 0) < 0:
                    raise ExchangeError(f"{method} {path} failed: {data}")
                return data
        raise ExchangeError(f"Request failed: {method} {path}")

    async def ping(self) -> None:
        await self._request("GET", "/fapi/v1/ping")

    async def sync_time(self) -> int:
        data = await self._request("GET", "/fapi/v1/time", signed=False)
        server_time = int(data["serverTime"])
        local_time = int(time.time() * 1000)
        self._time_offset_ms = server_time - local_time
        return self._time_offset_ms

    async def exchange_info(self, symbol: str) -> tuple[dict[str, Any], SymbolRules]:
        data = await self._request("GET", "/fapi/v1/exchangeInfo", params={"symbol": symbol})
        symbol_info = data["symbols"][0]
        filters = {item["filterType"]: item for item in symbol_info["filters"]}
        rules = SymbolRules(
            symbol=symbol,
            tick_size=float(filters["PRICE_FILTER"]["tickSize"]),
            step_size=float(filters["LOT_SIZE"]["stepSize"]),
            min_qty=float(filters["LOT_SIZE"]["minQty"]),
            min_notional=float(filters.get("MIN_NOTIONAL", {}).get("notional", "0")),
            quantity_precision=int(symbol_info["quantityPrecision"]),
            price_precision=int(symbol_info["pricePrecision"]),
        )
        if "demo-fapi.binance.com" in self.config.rest_base_url and symbol == "ETHUSDT":
            rules.quantity_precision = min(rules.quantity_precision, 2)
        return symbol_info, rules

    async def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        data = await self._request("GET", "/fapi/v1/klines", params=params)
        rows = []
        for item in data:
            rows.append(
                {
                    "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": pd.to_datetime(int(item[6]), unit="ms", utc=True),
                }
            )
        return pd.DataFrame(rows)

    async def klines_range(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit_per_call: int = 1000,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        cursor = start_time_ms
        while cursor < end_time_ms:
            part = await self.klines(
                symbol=symbol,
                interval=interval,
                limit=limit_per_call,
                start_time_ms=cursor,
                end_time_ms=end_time_ms,
            )
            if part.empty:
                break
            frames.append(part)
            cursor = int(part["open_time"].iloc[-1].timestamp() * 1000) + 60_000
            if len(part) < limit_per_call:
                break
        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("open_time", keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    async def change_leverage(self, symbol: str, leverage: int) -> Any:
        return await self._request("POST", "/fapi/v1/leverage", params={"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type_cross(self, symbol: str) -> None:
        try:
            await self._request("POST", "/fapi/v1/marginType", params={"symbol": symbol, "marginType": "CROSSED"}, signed=True)
        except ExchangeError as exc:
            message = str(exc)
            ignorable = (
                "No need to change margin type",
                "Margin type cannot be changed if there exists open orders",
                "Margin type cannot be changed if there exists position",
                "Position side cannot be changed if there exists open orders",
                "Position side cannot be changed if there exists position",
            )
            if not any(token in message for token in ignorable):
                raise

    async def set_position_mode_one_way(self) -> None:
        try:
            await self._request("POST", "/fapi/v1/positionSide/dual", params={"dualSidePosition": "false"}, signed=True)
        except ExchangeError as exc:
            message = str(exc)
            ignorable = (
                "No need to change position side",
                "Position side cannot be changed if there exists open orders",
                "Position side cannot be changed if there exists position",
            )
            if not any(token in message for token in ignorable):
                raise

    async def account_info(self) -> dict[str, Any]:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def position_risk(self, symbol: str) -> dict[str, Any]:
        rows = await self._request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol}, signed=True)
        return rows[0] if rows else {}

    async def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return await self._request("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)

    async def open_algo_orders(
        self,
        symbol: str | None = None,
        *,
        algo_type: str | None = None,
        algo_id: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        if algo_type is not None:
            params["algoType"] = algo_type
        if algo_id is not None:
            params["algoId"] = algo_id
        return await self._request("GET", "/fapi/v1/openAlgoOrders", params=params, signed=True)

    async def order(self, symbol: str, *, order_id: str | None = None, orig_client_order_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        return await self._request("GET", "/fapi/v1/order", params=params, signed=True)

    async def cancel_order(self, symbol: str, orig_client_order_id: str) -> dict[str, Any]:
        return await self._request("DELETE", "/fapi/v1/order", params={"symbol": symbol, "origClientOrderId": orig_client_order_id}, signed=True)

    async def cancel_all_orders(self, symbol: str) -> Any:
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)

    async def cancel_algo_order(
        self,
        *,
        algo_id: int | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        if not params:
            raise ExchangeError("Either algo_id or client_algo_id must be provided to cancel_algo_order().")
        return await self._request("DELETE", "/fapi/v1/algoOrder", params=params, signed=True)

    async def cancel_all_algo_orders(self, symbol: str) -> Any:
        return await self._request("DELETE", "/fapi/v1/algoOpenOrders", params={"symbol": symbol}, signed=True)

    async def algo_order(
        self,
        *,
        symbol: str,
        side: str,
        type_: str,
        client_algo_id: str,
        quantity: float | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        reduce_only: bool = False,
        time_in_force: str | None = None,
        working_type: str = "CONTRACT_PRICE",
        close_position: bool = False,
        price_protect: bool = False,
        new_order_resp_type: str = "ACK",
    ) -> dict[str, Any]:
        def _fmt(value: float | None) -> str | None:
            if value is None:
                return None
            return format(value, "f").rstrip("0").rstrip(".")

        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": type_,
            "clientAlgoId": client_algo_id,
            "workingType": working_type,
            "newOrderRespType": new_order_resp_type,
        }
        if quantity is not None:
            params["quantity"] = _fmt(quantity)
        if price is not None:
            params["price"] = _fmt(price)
        if trigger_price is not None:
            params["triggerPrice"] = _fmt(trigger_price)
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if reduce_only:
            params["reduceOnly"] = "true"
        if price_protect:
            params["priceProtect"] = "TRUE"
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity", None)
            params.pop("reduceOnly", None)
        return await self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)

    async def user_trades(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        return await self._request("GET", "/fapi/v1/userTrades", params={"symbol": symbol, "limit": limit}, signed=True)

    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", signed=True)
        return data["listenKey"]

    async def keepalive_listen_key(self, listen_key: str) -> None:
        await self._request("PUT", "/fapi/v1/listenKey", params={"listenKey": listen_key}, signed=True)

    async def close_listen_key(self, listen_key: str) -> None:
        await self._request("DELETE", "/fapi/v1/listenKey", params={"listenKey": listen_key}, signed=True)

    async def place_order(
        self,
        *,
        symbol: str,
        side: str,
        type_: str,
        quantity: float | None,
        client_order_id: str,
        price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
        time_in_force: str | None = None,
        working_type: str = "CONTRACT_PRICE",
        close_position: bool = False,
    ) -> dict[str, Any]:
        def _fmt(value: float | None) -> str | None:
            if value is None:
                return None
            return format(value, "f").rstrip("0").rstrip(".")

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
            "workingType": working_type,
        }
        if quantity is not None:
            params["quantity"] = _fmt(quantity)
        if price is not None:
            params["price"] = _fmt(price)
        if stop_price is not None:
            params["stopPrice"] = _fmt(stop_price)
        if reduce_only:
            params["reduceOnly"] = "true"
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity", None)
        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)

    async def mark_price(self, symbol: str) -> float:
        data = await self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})
        return float(data["markPrice"])

    @staticmethod
    def round_price(value: float, rules: SymbolRules) -> float:
        rounded = math.floor(value / rules.tick_size) * rules.tick_size
        return round(rounded, rules.price_precision)

    @staticmethod
    def round_quantity(value: float, rules: SymbolRules) -> float:
        rounded = math.floor(value / rules.step_size) * rules.step_size
        return round(rounded, rules.quantity_precision)

    async def compute_full_margin_short_qty(
        self,
        *,
        reference_price: float,
        leverage: int,
        safety_buffer: float,
        rules: SymbolRules,
    ) -> tuple[float, dict[str, Any]]:
        account = await self.account_info()
        available_balance = float(account["availableBalance"])
        max_notional = available_balance * leverage * safety_buffer
        raw_qty = max_notional / reference_price
        qty = self.round_quantity(raw_qty, rules)
        notional = qty * reference_price

        diagnostics = {
            "available_balance": available_balance,
            "max_notional": max_notional,
            "raw_qty": raw_qty,
            "qty": qty,
            "notional": notional,
            "rules": asdict(rules),
        }
        if qty <= 0:
            raise ExchangeError(f"Rounded quantity is zero: {diagnostics}")
        if qty < rules.min_qty:
            raise ExchangeError(f"Quantity below minQty: {diagnostics}")
        if notional < rules.min_notional:
            raise ExchangeError(f"Notional below minNotional: {diagnostics}")
        return qty, diagnostics
