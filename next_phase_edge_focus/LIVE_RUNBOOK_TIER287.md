# Live Runbook: Tier 2.87

This is the active working set.

## Active files

- [edge_survivor_baseline.py](/next_phase_edge_focus/edge_survivor_baseline.py)
- [binance_tier287_realistic_backtest_v1.py](/next_phase_edge_focus/binance_tier287_realistic_backtest_v1.py)
- [binance_tier287_eth_stress_test_v1.py](/next_phase_edge_focus/binance_tier287_eth_stress_test_v1.py)
- [live_helpers.py](/next_phase_edge_focus/live_helpers.py)
- [HOW_TO_TRADE_TIER287_REALISTIC.md](/next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md)
- [ETH Short Demo.txt](/next_phase_edge_focus/ETH%20Short%20Demo.txt)

## Active result folders

- [binance_tier287_realistic_backtest_v1](/next_phase_edge_focus/results/binance_tier287_realistic_backtest_v1)
- [binance_tier287_eth_stress_test_v1](/next_phase_edge_focus/results/binance_tier287_eth_stress_test_v1)

## Default commands

Run main baseline:

```powershell
python next_phase_edge_focus/edge_survivor_baseline.py
```

Run direct realistic benchmark:

```powershell
python next_phase_edge_focus/binance_tier287_realistic_backtest_v1.py
```

Run ETH/Binance stress package:

```powershell
python next_phase_edge_focus/binance_tier287_eth_stress_test_v1.py
```

## Rule for changes

If a new idea changes signal logic:

- do not overwrite Tier 2.87
- create a new experiment file
- save outputs in a new result folder

If a change is only:

- deployment
- documentation
- logging
- alerts
- bot packaging

then Tier 2.87 stays untouched.

## Current truth

For live preparation, treat `Tier 2.87 realistic` as the only active baseline.

## Bot Deploy Rule

For the Docker bot project, prefer:

- fresh full zip upload
- full clean redeploy

Avoid relying on repeated server-side file-by-file edits unless the fix is extremely small and urgent.

Reason:

- partial edits caused version drift more than once
- Telegram behavior and runtime code can get out of sync
- a clean redeploy is safer than stacking manual patches

## Clean Redeploy Recipe

Windows upload:

```powershell
scp -P 23019 "C:\Users\Tigran\Desktop\Binance\eth_short_demo_bot_fresh.zip" root@178.160.194.52:/root/
```

Server reset:

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
sleep 90
docker inspect -f '{{.RestartCount}}' eth-short-demo-bot
```

Healthy result:

- container is `Up`
- health becomes `healthy`
- `RestartCount` stays flat
