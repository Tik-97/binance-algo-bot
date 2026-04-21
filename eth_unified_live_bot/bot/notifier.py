from __future__ import annotations

import asyncio
import logging

import httpx

from .config import TelegramConfig
from .state import StateStore


LOGGER = logging.getLogger("eth_short_demo_bot.telegram")


class TelegramNotifier:
    def __init__(self, config: TelegramConfig, store: StateStore) -> None:
        self.config = config
        self.store = store
        self._client = httpx.AsyncClient(timeout=15.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send_event(self, text: str, dedupe_key: str | None = None, silent: bool = False) -> bool:
        return await self._send(text=text, chat_id=self.config.events_chat_id, dedupe_key=dedupe_key, silent=silent)

    async def send_report(self, text: str, dedupe_key: str | None = None) -> bool:
        return await self._send(
            text=text,
            chat_id=self.config.reports_chat_id or self.config.events_chat_id,
            dedupe_key=dedupe_key,
            silent=False,
        )

    def fire_event(self, text: str, dedupe_key: str | None = None, silent: bool = False) -> None:
        asyncio.create_task(self.send_event(text, dedupe_key=dedupe_key, silent=silent))

    def fire_report(self, text: str, dedupe_key: str | None = None) -> None:
        asyncio.create_task(self.send_report(text, dedupe_key=dedupe_key))

    async def _send(
        self,
        *,
        text: str,
        chat_id: str | None,
        dedupe_key: str | None,
        silent: bool,
    ) -> bool:
        if not self.config.enabled or not self.config.bot_token or not chat_id:
            return False
        if dedupe_key and self.store.notification_seen(dedupe_key):
            return False

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "disable_notification": silent,
        }
        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            if dedupe_key:
                self.store.mark_notification_sent(dedupe_key)
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Telegram send failed: %s", exc)
            self.store.log_event(
                "WARNING",
                "telegram_error",
                "Telegram send failed",
                "Telegram API call failed.",
                payload={"error": str(exc)},
            )
            return False

