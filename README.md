# Binance Algo Bot — ETHUSDT research & live trading bots

Research pipeline and Dockerized live bots for trading `ETHUSDT` perpetual futures on Binance,
with Telegram notifications and daily reports. The live bots run against the **Binance Futures DEMO**
environment.

## What's inside

| Folder | What it is |
|---|---|
| [`eth_unified_live_bot/`](eth_unified_live_bot/README.md) | Main live bot: one shared ETH wallet, multiple strategies behind a router (priority arbitration, wallet lock), SQLite state, Telegram events & daily reports, Docker |
| [`eth_short_demo_bot/`](eth_short_demo_bot/README.md) | First live bot: ETH short strategy, signed Binance Futures REST client, WebSocket runtime, recovery after restarts, smoke tests |
| [`next_phase_edge_focus/`](next_phase_edge_focus/README.md) | Short-side strategy research ("Tier 2.87"): realistic backtests, how-to-trade notes and live runbooks |
| [`next_phase_long_focus/`](next_phase_long_focus/README.md) | Long-side strategy research |
| `combined_results/` | Portfolio-level backtests: equity curve, drawdown, monthly P&L, stress tests |
| `fun_backtests/` | Experimental backtests and parameter sweeps |
| Root `*.py` scripts | Data download, feature/context analysis, setup clustering, entry-quality and execution studies |

## Live bot architecture

```
Binance WebSocket (closed 1m klines, user data stream)
            │
            ▼
   Strategy adapters ──► Router (wallet lock, priority, dedupe)
            │                         │
            ▼                         ▼
   Signed REST client ◄──── Execution & order/position state machine
            │                         │
            ▼                         ▼
     SQLite state store         Telegram notifier
 (orders, trades, events,    (entries, exits, errors,
   daily reports, stats)        daily reports)
```

- Processes only **closed** candles; catches up missed candles via REST after reconnects
- Recovers open orders/positions from state on restart
- Separate Telegram chats for events and reports are supported
- Docker / Docker Compose with a healthcheck based on the bot heartbeat

## Tech stack

Python · asyncio · Binance Futures REST & WebSocket APIs · SQLite · pandas / NumPy · matplotlib ·
Telegram Bot API · Docker

## Quick start (live demo bot)

```bash
cd eth_unified_live_bot
cp .env.live.example .env      # add your Binance DEMO API key/secret and Telegram bot token
docker compose up -d --build
```

See [`eth_unified_live_bot/SERVER_RUNBOOK.md`](eth_unified_live_bot/SERVER_RUNBOOK.md) for server setup.
API keys and tokens are read from environment variables and are never stored in the repository.

## Disclaimer

This is a personal research and engineering project. Backtest results do not guarantee future
performance, and nothing here is financial advice.
