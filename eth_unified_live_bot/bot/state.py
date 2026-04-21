from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("eth_short_demo_bot.state")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS kv_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            trade_day TEXT,
            payload TEXT
        );

        CREATE TABLE IF NOT EXISTS notifications (
            dedupe_key TEXT PRIMARY KEY,
            sent_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            client_order_id TEXT PRIMARY KEY,
            order_id TEXT,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            type TEXT NOT NULL,
            qty REAL,
            price REAL,
            stop_price REAL,
            reduce_only INTEGER NOT NULL DEFAULT 0,
            payload TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            opened_at TEXT NOT NULL,
            closed_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            strategy_name TEXT NOT NULL DEFAULT '',
            qty REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            pnl_usd REAL NOT NULL,
            exit_reason TEXT NOT NULL,
            payload TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_reports (
            trade_day TEXT NOT NULL,
            report_type TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            markdown TEXT NOT NULL,
            payload TEXT,
            PRIMARY KEY(trade_day, report_type)
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)
            conn.commit()

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_state(key, value, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, payload, utc_now()),
            )
            conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM kv_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def delete_state(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kv_state WHERE key = ?", (key,))
            conn.commit()

    def log_event(
        self,
        level: str,
        kind: str,
        title: str,
        message: str,
        payload: dict[str, Any] | None = None,
        trade_day: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO event_log(ts, level, kind, title, message, trade_day, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    level,
                    kind,
                    title,
                    message,
                    trade_day,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM event_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def events_for_trade_day(self, trade_day: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM event_log WHERE trade_day = ? ORDER BY id ASC",
                (trade_day,),
            ).fetchall()
        return [dict(row) for row in rows]

    def notification_seen(self, dedupe_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dedupe_key FROM notifications WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
        return row is not None

    def mark_notification_sent(self, dedupe_key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notifications(dedupe_key, sent_at)
                VALUES(?, ?)
                ON CONFLICT(dedupe_key) DO NOTHING
                """,
                (dedupe_key, utc_now()),
            )
            conn.commit()

    def upsert_order(
        self,
        *,
        client_order_id: str,
        order_id: str | None,
        purpose: str,
        status: str,
        symbol: str,
        side: str,
        type_: str,
        qty: float | None,
        price: float | None,
        stop_price: float | None,
        reduce_only: bool,
        payload: dict[str, Any] | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO orders(
                    client_order_id, order_id, purpose, status, symbol, side, type, qty,
                    price, stop_price, reduce_only, payload, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    order_id=excluded.order_id,
                    purpose=excluded.purpose,
                    status=excluded.status,
                    symbol=excluded.symbol,
                    side=excluded.side,
                    type=excluded.type,
                    qty=excluded.qty,
                    price=excluded.price,
                    stop_price=excluded.stop_price,
                    reduce_only=excluded.reduce_only,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    client_order_id,
                    order_id,
                    purpose,
                    status,
                    symbol,
                    side,
                    type_,
                    qty,
                    price,
                    stop_price,
                    int(reduce_only),
                    json.dumps(payload or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )
            conn.commit()

    def open_orders(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM orders
                WHERE status IN ('NEW', 'PARTIALLY_FILLED')
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def save_trade(
        self,
        *,
        trade_id: str,
        opened_at: str,
        closed_at: str,
        symbol: str,
        side: str,
        strategy_name: str = "",
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl_usd: float,
        exit_reason: str,
        payload: dict[str, Any] | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades(
                    trade_id, opened_at, closed_at, symbol, side, strategy_name, qty, entry_price,
                    exit_price, pnl_usd, exit_reason, payload
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    opened_at,
                    closed_at,
                    symbol,
                    side,
                    strategy_name,
                    qty,
                    entry_price,
                    exit_price,
                    pnl_usd,
                    exit_reason,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

    def trades_for_trade_day(self, trade_day: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE substr(opened_at, 1, 10) = ?
                ORDER BY opened_at ASC
                """,
                (trade_day,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_daily_report(
        self,
        trade_day: str,
        report_type: str,
        markdown: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_reports(trade_day, report_type, generated_at, markdown, payload)
                VALUES(?, ?, ?, ?, ?)
                """,
                (trade_day, report_type, utc_now(), markdown, json.dumps(payload or {}, ensure_ascii=False)),
            )
            conn.commit()

    def get_daily_report(self, trade_day: str, report_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM daily_reports
                WHERE trade_day = ? AND report_type = ?
                """,
                (trade_day, report_type),
            ).fetchone()
        return dict(row) if row else None

