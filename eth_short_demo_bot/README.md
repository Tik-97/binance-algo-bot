# ETH Short Demo Bot

Dockerized Binance Futures DEMO bot for the current `ETHUSDT` short baseline.

## What This Project Does

- trades `ETHUSDT` only
- trades `SHORT` only
- sets `10x` leverage
- sizes with full available balance and a safety buffer
- uses closed `1m` candles, not blind high-frequency polling
- stores state in SQLite so restarts are recoverable
- sends Telegram events and daily reports
- writes readable logs and markdown daily reports
- includes manual demo smoke tests for:
  - market short + SL/TP
  - limit short + SL/TP

## Architecture

- `bot/config.py`
  - config loader
  - parses `.env`
  - legacy `ETH Short Demo.txt` still supported
  - supports separate Telegram chat IDs for events and reports
- `bot/exchange.py`
  - signed Binance Futures DEMO REST client
  - symbol rules, sizing, leverage, orders, listen key
- `bot/strategy.py`
  - self-contained live-safe `Tier 2.87` strategy adapter
  - no dependency on research folders at runtime
- `bot/service.py`
  - runtime orchestration
  - websocket consumers
  - startup recovery
  - order / position state machine
  - daily reporting
  - smoke test helpers
- `bot/state.py`
  - SQLite state store
  - event log
  - order journal
  - trade journal
  - daily reports
- `bot/notifier.py`
  - Telegram sender with dedupe
- `bot/healthcheck.py`
  - Docker healthcheck using latest heartbeat

## Binance API / WebSocket Usage Schedule

- Live market updates:
  - `wss://.../ws/ethusdt@kline_1m`
  - process only closed `1m` klines
  - this is the main signal trigger
- Candle processing:
  - on each closed `1m` kline event
  - if websocket reconnects, catch up missed candles via REST once
- User data stream:
  - dedicated listen key websocket
  - used for fills, stop fills, order status, account events
- Order status reconciliation:
  - REST every `45s`
  - also forced after websocket reconnect
- Position reconciliation:
  - REST every `45s`
  - also on startup and user-stream reconnect
- Account balance checks:
  - via `/fapi/v2/account` during reconciliation
  - plus immediately before sizing a new trade
- Symbol metadata refresh:
  - on startup
  - then every `6h`
- Listen key keepalive:
  - every `30m`
- Boot recovery:
  - restore local SQLite state
  - fetch live open orders
  - fetch live position
  - fetch account snapshot
  - recover or clear stale local state safely
  - replay missed closed candles from REST before starting normal websocket flow
- Daily analysis scheduling:
  - pre-day report at `12:45 UTC`
  - post-day report at `16:05 UTC`
- Telegram heartbeat:
  - no spammy heartbeat message loop by default
  - internal DB heartbeat every `15s`
  - healthcheck reads that heartbeat

## Startup Recovery / Reconciliation

On startup the bot does this in order:

1. initializes SQLite schema
2. validates config
3. pings Binance DEMO
4. enforces one-way mode, cross margin, `10x` leverage
5. loads recent `1m` and `1d` history
6. loads local persisted state
7. fetches live account, position, open orders
8. reconciles local state against exchange
9. restores or clears stale signal/order/position state
10. replays missed closed candles from REST
11. opens listen key
12. starts market stream, user stream, reconciliation, keepalive, reports, health heartbeat

## Project Structure

```text
eth_short_demo_bot/
├── bot/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── exchange.py
│   ├── healthcheck.py
│   ├── logging_setup.py
│   ├── models.py
│   ├── notifier.py
│   ├── service.py
│   ├── state.py
│   └── strategy.py
├── data/
├── logs/
├── Dockerfile
├── docker-compose.yml
├── ETH Short Demo.example.txt
├── requirements.txt
└── README.md
```

## Exact Config Format

Preferred format:

```text
BINANCE_API_KEY=your_demo_api_key
BINANCE_API_SECRET=your_demo_api_secret
TELEGRAM_BOT_TOKEN=telegram_bot_token
TELEGRAM_CHAT_ID=default_chat_id
TELEGRAM_EVENTS_CHAT_ID=optional_events_chat_id
TELEGRAM_REPORTS_CHAT_ID=optional_reports_chat_id
BINANCE_REST_BASE_URL=https://demo-fapi.binance.com
BINANCE_WS_STREAM_BASE_URL=wss://demo-fstream.binance.com
```

Legacy format also supported:

```text
<secret> - Secret
<api_key> - API Key
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Commands

Show config format:

```bash
python -m bot.cli config-format
```

Run the bot:

```bash
python -m bot --config "/app/.env" run
```

Manual demo test, market + SL/TP:

```bash
python -m bot --config "/app/.env" smoke-market
```

Manual demo test, limit + SL/TP:

```bash
python -m bot --config "/app/.env" smoke-limit
```

Emergency close/cancel:

```bash
python -m bot --config "/app/.env" close-all
```

## Ubuntu / Docker Deploy

1. Copy the whole `eth_short_demo_bot/` folder to the server.
2. Put your real `.env` inside that folder.
3. Review `docker-compose.yml`.
4. Run:

```bash
docker compose up -d --build
```

5. Check:

```bash
docker compose logs -f
docker compose ps
```

Detailed upload / restart / health / update instructions:

```text
See SERVER_RUNBOOK.md
```

## Operational Checklist

- before first run:
  - confirm Binance Futures DEMO API key works
  - confirm Telegram bot token and chat IDs work
  - run `config-format`
  - run `smoke-market`
  - run `smoke-limit`
- before live demo session:
  - check container is healthy
  - check latest pre-day report
  - confirm no stale open orders or stale short
- after any restart:
  - check recovery message in Telegram
  - confirm open orders / open position are reconciled correctly
- before code change:
  - stop container
  - backup `data/state.db`

## Future Improvements

- move from SQLite to Postgres only if multi-host or analytics needs grow
- add dedicated Prometheus metrics endpoint
- add mark-price websocket and latency metrics
- add exact fill-fee reconciliation from user trades
- split execution policy into configurable strategy profiles
- add integration tests against Binance DEMO sandbox workflows
