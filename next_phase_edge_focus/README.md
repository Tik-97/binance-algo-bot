# ETH Short Edge Focus

## Current baseline

Active main baseline:

- `Tier 2.87 realistic`

This is now the source of truth for the Binance ETH short branch.

Main files:

- baseline runner: [edge_survivor_baseline.py](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/edge_survivor_baseline.py)
- baseline strategy: [binance_tier287_realistic_backtest_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/binance_tier287_realistic_backtest_v1.py)
- realistic summary: [summary.txt](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/results/binance_tier287_realistic_backtest_v1/summary.txt)
- ETH/Binance stress summary: [summary.txt](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/results/binance_tier287_eth_stress_test_v1/summary.txt)
- manual Binance playbook: [HOW_TO_TRADE_TIER287_REALISTIC.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md)
- live runbook: [LIVE_RUNBOOK_TIER287.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/LIVE_RUNBOOK_TIER287.md)

## Why Tier 2.87

Compared with `Tier 2.85` and `Tier 2.86 refined`, this version currently gives the best balance of:

- realistic Binance execution
- trade count
- win rate
- stress-test survival
- live-safe signal visibility

Current benchmark:

- `42` trades
- `ending_balance = 645.87`
- `win_rate = 88.10%`
- `profit_factor = 7.872`

Stress package:

- `ETH adverse light`: `620.00`
- `ETH adverse medium`: `562.38`
- `ETH adverse hard`: `498.29`

So this branch still degrades under worse fills, but it survives in a way that still looks tradable on ETHUSDT.

## Active project scope

This folder is now intentionally narrow.

We keep only:

- the active baseline
- the current ETH stress package
- the helper logic that baseline depends on
- the deployment config file
- the manuals needed for live transition

Older exploratory branches were removed from the active area so the project is easier to maintain.

## Market focus

We stay with:

- `Binance`
- `ETHUSDT`
- `Short only`

No BTC branch.
No SOL branch.
No long branch.

## Session focus

The candidate engine still works inside the original session window:

- `13:00-16:00 UTC`
- `17:00-20:00 Asia/Yerevan`

That matters for both manual review and live automation.

## Recommended next step

The research branch is no longer the bottleneck.

The next practical step after this cleanup is:

1. move into live/paper workflow
2. wire the strategy into the Binance demo bot
3. keep `Tier 2.87` fixed while forward data accumulates

## Rule for future changes

If a new idea changes signal logic:

- do not overwrite `Tier 2.87`
- create a new experiment file
- save outputs under `next_phase_edge_focus/results/<new_name>/`

If the change is only:

- documentation
- deployment
- monitoring
- reporting

then keep the baseline logic untouched.
