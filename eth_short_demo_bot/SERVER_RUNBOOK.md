# Server Runbook

This is the practical runbook for the deployed `eth_short_demo_bot` on Ubuntu.

Current expected server path:

```bash
/root/eth_short_demo_bot
```

Current server connection used during setup:

```text
host: 178.160.194.52
port: 23019
user: root
```

## 1. Upload Updated Files

Upload only one changed file:

```powershell
scp -P 23019 "C:\Users\Tigran\Desktop\Binance\eth_short_demo_bot\bot\service.py" root@178.160.194.52:/root/eth_short_demo_bot/bot/service.py
```

Upload the whole project as a fresh zip:

```powershell
scp -P 23019 "C:\Users\Tigran\Desktop\Binance\eth_short_demo_bot_fresh.zip" root@178.160.194.52:/root/
```

Then on the server:

```bash
cd /root
docker compose -f /root/eth_short_demo_bot/docker-compose.yml down --remove-orphans 2>/dev/null || true
docker rm -f eth-short-demo-bot 2>/dev/null || true
rm -rf /root/eth_short_demo_bot
unzip -o /root/eth_short_demo_bot_fresh.zip -d /root/
cd /root/eth_short_demo_bot
```

Preferred rule:

- for any meaningful bot update, prefer uploading a fresh full zip
- avoid piecemeal server edits unless the fix is tiny and urgent
- if the server starts behaving strangely, do a full clean redeploy instead of stacking more manual patches

## 2. Start The Bot

```bash
cd /root/eth_short_demo_bot
docker compose up -d --build
```

## 2A. Full Clean Redeploy From Scratch

Use this when:

- the server folder drifted from local code
- the bot restarted strangely after several manual edits
- Telegram showed old behavior after a supposed update
- you want the safest reset to a known-good local version

Upload fresh archive from Windows:

```powershell
scp -P 23019 "C:\Users\Tigran\Desktop\Binance\eth_short_demo_bot_fresh.zip" root@178.160.194.52:/root/
```

Then on the server:

```bash
cd /root
docker compose -f /root/eth_short_demo_bot/docker-compose.yml down --remove-orphans 2>/dev/null || true
docker rm -f eth-short-demo-bot 2>/dev/null || true
docker image rm eth_short_demo_bot-eth-short-demo-bot 2>/dev/null || true
rm -rf /root/eth_short_demo_bot
unzip -o /root/eth_short_demo_bot_fresh.zip -d /root/
cd /root/eth_short_demo_bot
docker compose up -d --build
docker compose ps
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
```

Then verify stability:

```bash
sleep 90
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
docker compose ps
```

Good result:

- container is `Up`
- health becomes `healthy`
- `RestartCount` does not increase

## 3. Stop The Bot

```bash
cd /root/eth_short_demo_bot
docker compose down
```

## 4. Restart The Bot

```bash
cd /root/eth_short_demo_bot
docker compose restart
```

If the image or code changed:

```bash
cd /root/eth_short_demo_bot
docker compose down
docker compose up -d --build
```

## 5. Check Status

Container status:

```bash
cd /root/eth_short_demo_bot
docker compose ps
```

Docker restart counter:

```bash
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
```

Health status only:

```bash
docker inspect --format='{{json .State.Health}}' eth-short-demo-bot
```

## 6. Check Logs

Container logs:

```bash
cd /root/eth_short_demo_bot
docker compose logs --tail=200
```

Live file log:

```bash
tail -f /root/eth_short_demo_bot/logs/bot.log
```

## 7. Safe Demo Risk Controls

Emergency close all open risk:

```bash
cd /root/eth_short_demo_bot
docker compose run --rm eth-short-demo-bot python -m bot --config /app/.env close-all
```

## 8. Manual Smoke Tests

Market smoke test:

```bash
cd /root/eth_short_demo_bot
docker compose run --rm eth-short-demo-bot python -m bot --config /app/.env smoke-market --test-notional-usd 50 --stop-pct 0.35 --tp-pct 0.70
```

Limit smoke test:

```bash
cd /root/eth_short_demo_bot
docker compose run --rm eth-short-demo-bot python -m bot --config /app/.env smoke-limit --test-notional-usd 50 --entry-offset-pct -0.05 --stop-pct 0.35 --tp-pct 0.70 --wait-seconds 30
```

## 9. Quick Health Checklist

After startup, confirm:

```bash
docker compose ps
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
sleep 90
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
```

Good result:

- container is `Up`
- health is `healthy`
- restart count does not increase

## 10. If The Bot Keeps Restarting

Run in foreground to see the real exception:

```bash
cd /root/eth_short_demo_bot
docker compose down
docker compose run --rm eth-short-demo-bot python -m bot --config /app/.env run
```

## 11. Verify Important Server Files

```bash
ls -la /root/eth_short_demo_bot
ls -la /root/eth_short_demo_bot/bot
```

Expected important files:

- `.env`
- `docker-compose.yml`
- `Dockerfile`
- `bot/service.py`
- `bot/strategy.py`
- `data/state.db` after runtime starts
- `logs/bot.log` after runtime writes logs

## 12. What Telegram Should Normally Send

Expected important messages only:

- bot started
- recovery complete on startup or real reconnect
- possible short setup
- order submitted
- position opened
- position closed
- protective stop filled
- websocket / recovery errors
- pre-day report
- post-day report
- heartbeat every 3 hours

Not expected:

- periodic recovery spam every minute
- market stream connected spam
