# ETH Unified Live Bot

Dockerized Binance Futures DEMO bot for one shared `ETHUSDT` wallet with:

- `short_t287` - live short baseline
- `long_t14` - stub, loaded but inactive
- `long_f21` - stub, loaded but inactive

## What This Project Does

- trades `ETHUSDT` only
- runs one process and one shared wallet
- allows one live ETH position at a time
- preserves the stable `eth_short_demo_bot` runtime architecture
- keeps the live short branch unchanged in logic
- loads long branches as stubs for clean future porting
- persists state in SQLite
- sends Telegram events and reports
- runs under Docker / Docker Compose

## Architecture

- `bot/config.py`
  - `.env` and legacy `ETH Short Demo.txt` parsing
  - global and per-strategy session windows
  - enabled strategy list
  - shared-wallet sizing mode
- `bot/strategies/`
  - `short_t287.py` - live short adapter
  - `long_t14.py` - stub
  - `long_f21.py` - stub
- `bot/router.py`
  - wallet lock
  - priority arbitration
  - long-long dedupe
- `bot/service.py`
  - websocket runtime
  - recovery
  - shared-wallet execution
  - unified daily reports
- `bot/state.py`
  - SQLite state
  - event log
  - order journal
  - trade journal
  - daily reports
  - `session_stats`

## Enabled Strategies

Default:

```text
ENABLED_STRATEGIES=short_t287,long_t14,long_f21
```

Because `long_t14` and `long_f21` are stubs, they load safely and generate zero real signals.

## Shared-Wallet Sizing

Default mode:

```text
POSITION_SIZE_MODE=full_balance
```

Meaning:

- before every entry, bot reads fresh `availableBalance`
- multiplies it by leverage
- applies existing safety buffer
- computes position size from that notional

Alternative:

```text
POSITION_SIZE_MODE=fixed_usd
NOTIONAL_USD_PER_TRADE=50.0
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

Emergency close:

```bash
python -m bot --config "/app/.env" close-all
```

Smoke market:

```bash
python -m bot --config "/app/.env" smoke-market
```

Smoke limit:

```bash
python -m bot --config "/app/.env" smoke-limit
```

## Docker Deploy

```bash
docker compose up -d --build
docker compose ps
docker inspect -f '{{.RestartCount}}' eth-unified-bot
```

Live template:

- [.env.live.example](/C:/Users/Tigran/Desktop/Binance/eth_unified_live_bot/.env.live.example)

Detailed operational steps:

- see [SERVER_RUNBOOK.md](/C:/Users/Tigran/Desktop/Binance/eth_unified_live_bot/SERVER_RUNBOOK.md)

## Trading Guides

- [HOW_TO_TRADE_SHORT_T287.md](/C:/Users/Tigran/Desktop/Binance/eth_unified_live_bot/HOW_TO_TRADE_SHORT_T287.md)
- [HOW_TO_TRADE_LONG_T14.md](/C:/Users/Tigran/Desktop/Binance/eth_unified_live_bot/HOW_TO_TRADE_LONG_T14.md)
- [HOW_TO_TRADE_LONG_F21.md](/C:/Users/Tigran/Desktop/Binance/eth_unified_live_bot/HOW_TO_TRADE_LONG_F21.md)
