# Unified Bot Server Runbook

Expected server path:

```bash
/root/eth_unified_live_bot
```

Live config template:

- [.env.live.example](/eth_unified_live_bot/.env.live.example)

## Quick Ops

Start:

```bash
cd /root/eth_unified_live_bot
docker compose up -d --build
```

Stop:

```bash
cd /root/eth_unified_live_bot
docker compose down
```

Restart:

```bash
cd /root/eth_unified_live_bot
docker compose restart
```

Verify:

```bash
cd /root/eth_unified_live_bot
docker compose ps
docker inspect -f '{{.RestartCount}}' eth-unified-bot
tail -n 120 /root/eth_unified_live_bot/logs/bot.log
```

## Upload Fresh Zip

```powershell
scp -P 23019 "C:\Users\Tigran\Desktop\Binance\eth_unified_live_bot.zip" root@178.160.194.52:/root/
```

## Clean Redeploy

```bash
cd /root
docker compose -f /root/eth_unified_live_bot/docker-compose.yml down --remove-orphans 2>/dev/null || true
docker rm -f eth-unified-bot 2>/dev/null || true
docker image rm eth_unified_live_bot-eth-unified-bot 2>/dev/null || true
rm -rf /root/eth_unified_live_bot
unzip -o /root/eth_unified_live_bot.zip -d /root/
cd /root/eth_unified_live_bot
docker compose up -d --build
```

## Start

```bash
cd /root/eth_unified_live_bot
docker compose up -d --build
```

## Stop

```bash
cd /root/eth_unified_live_bot
docker compose down
```

## Restart

```bash
cd /root/eth_unified_live_bot
docker compose restart
```

If code changed:

```bash
cd /root/eth_unified_live_bot
docker compose down
docker compose up -d --build
```

## Status

```bash
cd /root/eth_unified_live_bot
docker compose ps
docker inspect -f '{{.RestartCount}}' eth-unified-bot
```

## Logs

```bash
cd /root/eth_unified_live_bot
docker compose logs --tail=200
tail -f /root/eth_unified_live_bot/logs/bot.log
```

## Emergency Close

```bash
cd /root/eth_unified_live_bot
docker compose run --rm eth-unified-bot python -m bot --config /app/.env close-all
```

## Smoke Tests

```bash
cd /root/eth_unified_live_bot
docker compose run --rm eth-unified-bot python -m bot --config /app/.env smoke-market --test-notional-usd 50 --stop-pct 0.35 --tp-pct 0.70
docker compose run --rm eth-unified-bot python -m bot --config /app/.env smoke-limit --test-notional-usd 50 --entry-offset-pct -0.05 --stop-pct 0.35 --tp-pct 0.70 --wait-seconds 30
```

## Stable Result

Good deploy means:

- container is `Up`
- health becomes `healthy`
- `RestartCount` does not increase
- Telegram shows one startup recovery and then normal events only
