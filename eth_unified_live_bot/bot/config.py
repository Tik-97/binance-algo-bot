from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_CANDIDATES = [
    ROOT_DIR / ".env",
    ROOT_DIR / "ETH Short Demo.txt",
    ROOT_DIR.parent / "next_phase_edge_focus" / ".env",
    ROOT_DIR.parent / "next_phase_edge_focus" / "ETH Short Demo.txt",
]


class ConfigError(RuntimeError):
    """Raised when config is missing or malformed."""


@dataclass(slots=True)
class TelegramConfig:
    bot_token: str | None
    events_chat_id: str | None
    reports_chat_id: str | None
    enabled: bool = True


@dataclass(slots=True)
class BinanceConfig:
    api_key: str
    api_secret: str
    rest_base_url: str = "https://fapi.binance.com"
    ws_stream_base_url: str = "wss://fstream.binance.com"
    recv_window_ms: int = 5000


@dataclass(slots=True)
class AppConfig:
    binance: BinanceConfig
    telegram: TelegramConfig
    symbol: str = "ETHUSDT"
    leverage: int = 10
    order_margin_safety_buffer: float = 0.985
    config_path: Path | None = None
    state_db_path: Path = ROOT_DIR / "data" / "state.db"
    log_dir: Path = ROOT_DIR / "logs"
    session_start_hour_utc: int = 13
    session_end_hour_utc: int = 16
    short_t287_session_start_utc: int = 13
    short_t287_session_end_utc: int = 16
    long_t14_session_start_utc: int = 13
    long_t14_session_end_utc: int = 16
    long_f21_session_start_utc: int = 13
    long_f21_session_end_utc: int = 16
    long_bull_v3_session_start_utc: int = 0
    long_bull_v3_session_end_utc: int = 0
    enabled_strategies: list[str] = field(
        default_factory=lambda: ["short_t287", "long_t14", "long_f21"]
    )
    trading_mode: str = "live"
    position_size_mode: str = "full_balance"
    notional_usd_per_trade: float = 50.0
    heartbeat_interval_seconds: int = 15
    telegram_status_interval_seconds: int = 3 * 60 * 60
    reconcile_interval_seconds: int = 45
    symbol_refresh_interval_seconds: int = 6 * 60 * 60
    listen_key_keepalive_seconds: int = 30 * 60
    user_stream_reconnect_backoff_seconds: int = 5
    market_stream_reconnect_backoff_seconds: int = 5
    max_bootstrap_1m_candles: int = 1500
    daily_kline_history_limit: int = 120


LEGACY_API_KEY_PATTERN = re.compile(r"^\s*(?P<value>\S+)\s*-\s*API Key\s*$", re.IGNORECASE)
LEGACY_API_SECRET_PATTERN = re.compile(r"^\s*(?P<value>\S+)\s*-\s*Secret\s*$", re.IGNORECASE)


def _read_key_value_lines(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        api_key_match = LEGACY_API_KEY_PATTERN.match(line)
        if api_key_match:
            data["BINANCE_API_KEY"] = api_key_match.group("value").strip()
            continue

        api_secret_match = LEGACY_API_SECRET_PATTERN.match(line)
        if api_secret_match:
            data["BINANCE_API_SECRET"] = api_secret_match.group("value").strip()
            continue

        if "=" not in line:
            raise ConfigError(
                f"Malformed config line in {path.name!s}: {raw_line!r}. "
                "Expected KEY=VALUE or legacy '<value> - API Key/Secret'."
            )

        key, value = line.split("=", 1)
        data[key.strip().upper()] = value.strip()

    return data


def _resolve_config_path(explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return path

    for env_key in ("ETH_UNIFIED_LIVE_CONFIG", "ETH_SHORT_DEMO_CONFIG"):
        env_path = os.getenv(env_key)
        if env_path:
            path = Path(env_path).expanduser().resolve()
            if not path.exists():
                raise ConfigError(f"Config file from {env_key} not found: {path}")
            return path

    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    raise ConfigError(
        "Could not find config file. Expected '.env' or 'ETH Short Demo.txt' "
        "in the unified bot project root or in ../next_phase_edge_focus/."
    )


def _parse_session_hour(values: dict[str, str], key: str, fallback: int) -> int:
    raw = values.get(key)
    return int(raw) if raw is not None else fallback


def load_config(explicit_path: str | None = None) -> AppConfig:
    path = _resolve_config_path(explicit_path)
    values = _read_key_value_lines(path)

    api_key = values.get("BINANCE_API_KEY")
    api_secret = values.get("BINANCE_API_SECRET")
    if not api_key:
        raise ConfigError("Missing BINANCE_API_KEY in config.")
    if not api_secret:
        raise ConfigError("Missing BINANCE_API_SECRET in config.")

    telegram_token = values.get("TELEGRAM_BOT_TOKEN")
    default_chat_id = values.get("TELEGRAM_CHAT_ID")
    events_chat_id = values.get("TELEGRAM_EVENTS_CHAT_ID", default_chat_id)
    reports_chat_id = values.get("TELEGRAM_REPORTS_CHAT_ID", default_chat_id)

    telegram_enabled = telegram_token is not None and (events_chat_id or reports_chat_id)
    telegram = TelegramConfig(
        bot_token=telegram_token,
        events_chat_id=events_chat_id,
        reports_chat_id=reports_chat_id,
        enabled=telegram_enabled,
    )

    session_start_hour_utc = int(values.get("SESSION_START_HOUR_UTC", "13"))
    session_end_hour_utc = int(values.get("SESSION_END_HOUR_UTC", "16"))

    enabled_strategies = [
        item.strip()
        for item in values.get("ENABLED_STRATEGIES", "short_t287,long_t14,long_f21").split(",")
        if item.strip()
    ]
    rest_url = values.get("BINANCE_REST_BASE_URL", "https://fapi.binance.com").strip()
    ws_url = values.get("BINANCE_WS_STREAM_BASE_URL", "wss://fstream.binance.com").strip()
    if "demo-fapi" in rest_url or "demo-fstream" in ws_url:
        trading_mode = "demo"
    else:
        trading_mode = values.get("TRADING_MODE", "live").strip().lower()
    if trading_mode not in ("live", "demo"):
        raise ConfigError(
            f"TRADING_MODE must be 'live' or 'demo', got {trading_mode!r}"
        )
    position_size_mode = values.get("POSITION_SIZE_MODE", "full_balance").strip().lower()
    if position_size_mode not in ("full_balance", "fixed_usd"):
        raise ConfigError(
            f"POSITION_SIZE_MODE must be 'full_balance' or 'fixed_usd', got {position_size_mode!r}"
        )
    notional_usd_per_trade = float(values.get("NOTIONAL_USD_PER_TRADE", "50.0"))

    config = AppConfig(
        binance=BinanceConfig(
            api_key=api_key,
            api_secret=api_secret,
            rest_base_url=rest_url,
            ws_stream_base_url=ws_url,
            recv_window_ms=int(values.get("BINANCE_RECV_WINDOW_MS", "5000")),
        ),
        telegram=telegram,
        symbol=values.get("SYMBOL", "ETHUSDT"),
        leverage=int(values.get("LEVERAGE", "10")),
        order_margin_safety_buffer=float(values.get("ORDER_MARGIN_SAFETY_BUFFER", "0.985")),
        config_path=path,
        state_db_path=Path(values.get("STATE_DB_PATH", str(ROOT_DIR / "data" / "state.db"))),
        log_dir=Path(values.get("LOG_DIR", str(ROOT_DIR / "logs"))),
        session_start_hour_utc=session_start_hour_utc,
        session_end_hour_utc=session_end_hour_utc,
        short_t287_session_start_utc=_parse_session_hour(values, "SHORT_T287_SESSION_START_UTC", session_start_hour_utc),
        short_t287_session_end_utc=_parse_session_hour(values, "SHORT_T287_SESSION_END_UTC", session_end_hour_utc),
        long_t14_session_start_utc=_parse_session_hour(values, "LONG_T14_SESSION_START_UTC", session_start_hour_utc),
        long_t14_session_end_utc=_parse_session_hour(values, "LONG_T14_SESSION_END_UTC", session_end_hour_utc),
        long_f21_session_start_utc=_parse_session_hour(values, "LONG_F21_SESSION_START_UTC", session_start_hour_utc),
        long_f21_session_end_utc=_parse_session_hour(values, "LONG_F21_SESSION_END_UTC", session_end_hour_utc),
        long_bull_v3_session_start_utc=_parse_session_hour(values, "LONG_BULL_V3_SESSION_START_UTC", 0),
        long_bull_v3_session_end_utc=_parse_session_hour(values, "LONG_BULL_V3_SESSION_END_UTC", 0),
        enabled_strategies=enabled_strategies,
        trading_mode=trading_mode,
        position_size_mode=position_size_mode,
        notional_usd_per_trade=notional_usd_per_trade,
    )

    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.state_db_path.parent.mkdir(parents=True, exist_ok=True)
    return config


def documented_config_format() -> str:
    return (
        "Preferred format:\n"
        "BINANCE_API_KEY=your_live_api_key\n"
        "BINANCE_API_SECRET=your_live_api_secret\n"
        "TELEGRAM_BOT_TOKEN=telegram_bot_token\n"
        "TELEGRAM_CHAT_ID=default_chat_id\n"
        "TELEGRAM_EVENTS_CHAT_ID=optional_separate_events_chat_id\n"
        "TELEGRAM_REPORTS_CHAT_ID=optional_separate_reports_chat_id\n"
        "SYMBOL=ETHUSDT\n"
        "LEVERAGE=10\n"
        "\n"
        "# Live (default):\n"
        "BINANCE_REST_BASE_URL=https://fapi.binance.com\n"
        "BINANCE_WS_STREAM_BASE_URL=wss://fstream.binance.com\n"
        "TRADING_MODE=live\n"
        "ENABLED_STRATEGIES=short_t287\n"
        "POSITION_SIZE_MODE=full_balance\n"
        "# POSITION_SIZE_MODE=fixed_usd\n"
        "# NOTIONAL_USD_PER_TRADE=50.0   # only used when POSITION_SIZE_MODE=fixed_usd\n"
        "\n"
        "# Demo:\n"
        "# BINANCE_REST_BASE_URL=https://demo-fapi.binance.com\n"
        "# BINANCE_WS_STREAM_BASE_URL=wss://demo-fstream.binance.com\n"
        "# TRADING_MODE=demo\n"
        "# ENABLED_STRATEGIES=short_t287,long_t14,long_f21\n"
        "\n"
        "SHORT_T287_SESSION_START_UTC=13\n"
        "SHORT_T287_SESSION_END_UTC=16\n"
        "LONG_T14_SESSION_START_UTC=13\n"
        "LONG_T14_SESSION_END_UTC=16\n"
        "LONG_F21_SESSION_START_UTC=13\n"
        "LONG_F21_SESSION_END_UTC=16\n"
        "\n"
        "Recommended file name: .env\n"
        "\n"
        "Legacy format also supported:\n"
        "<secret> - Secret\n"
        "<api_key> - API Key\n"
        "TELEGRAM_BOT_TOKEN=...\n"
        "TELEGRAM_CHAT_ID=...\n"
        "\n"
        "Live readiness checklist:\n"
        "  1. BINANCE_API_KEY and BINANCE_API_SECRET are from a LIVE Binance Futures account\n"
        "  2. The API key has Futures trading permission enabled\n"
        "  3. The API key has IP whitelist set to your server IP\n"
        "  4. LEVERAGE=10 is confirmed acceptable for your balance\n"
        "  5. POSITION_SIZE_MODE=full_balance is confirmed - bot will use 100% available margin\n"
        "  6. ENABLED_STRATEGIES=short_t287 only - long stubs must not be enabled on live\n"
        "  7. Telegram notifications are working (run smoke-market on demo first to verify)\n"
    )
