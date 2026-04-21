from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_payload") and isinstance(record.event_payload, dict):
            payload["payload"] = record.event_payload
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt_stream = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fmt_file = JsonFormatter()

    file_handler = RotatingFileHandler(
        log_dir / "bot.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt_file)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt_stream)

    for name in ("eth_short_demo_bot", "eth_unified_live_bot"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.propagate = False
        lg.addHandler(stream_handler)
        lg.addHandler(file_handler)

    return logging.getLogger("eth_unified_live_bot")

