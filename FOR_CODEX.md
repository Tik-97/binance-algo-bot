# FOR CODEX

This file is the fast handoff for any new Codex chat started in this workspace.

## 1. Current Active Scope

There are two active areas:

- research / strategy baseline:
  - [next_phase_edge_focus/README.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/README.md)
  - [next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md)
  - [next_phase_edge_focus/LIVE_RUNBOOK_TIER287.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/LIVE_RUNBOOK_TIER287.md)
- long research restart:
  - [next_phase_long_focus/README.md](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/README.md)
- live demo bot:
  - [eth_short_demo_bot/README.md](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/README.md)
  - [eth_short_demo_bot/SERVER_RUNBOOK.md](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/SERVER_RUNBOOK.md)

Ignore old experimental clutter in the repo unless the user explicitly asks for it.

## 2. Current Strategy Truth

Active baseline:

- `Tier 2.87 realistic`

Main files:

- [next_phase_edge_focus/edge_survivor_baseline.py](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/edge_survivor_baseline.py)
- [next_phase_edge_focus/binance_tier287_realistic_backtest_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/binance_tier287_realistic_backtest_v1.py)
- [next_phase_edge_focus/binance_tier287_eth_stress_test_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/binance_tier287_eth_stress_test_v1.py)

Important strategy facts:

- market: `ETHUSDT`
- direction: `SHORT only`
- session: `13:00-16:00 UTC`
- Yerevan time: `17:00-20:00`
- strategy is live-safe by design: decisions come from already closed `1m` candles and current daily regime state

Current practical interpretation:

- the strategy can still evaluate candles outside the session
- actual valid trade candidates are meant for the UTC session window above

Long branch truth:

- long branch is active again under [next_phase_long_focus/README.md](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/README.md)
- it is still research, not live-ready
- current first independent long baseline:
  - [edge_long_tier1_bear_highvol_baseline_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier1_bear_highvol_baseline_v1.py)
- current first clean long core:
  - [edge_long_tier11_core_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier11_core_v1.py)
- current first expanded long candidate:
  - [edge_long_tier13_core_plus_satellite1_refined_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier13_core_plus_satellite1_refined_v1.py)
- current best structured long candidate:
  - [edge_long_tier14_core_plus_satellite2_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier14_core_plus_satellite2_v1.py)
- current widest still-clean long candidate:
  - [edge_long_tier15_core_plus_satellite3_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier15_core_plus_satellite3_v1.py)
- realistic execution stress package for the clean long baseline:
  - [edge_long_tier14_eth_stress_test_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier14_eth_stress_test_v1.py)
- fresh external-family analysis from Another Trader long trades:
  - [edge_long_another_trader_family_analysis_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_another_trader_family_analysis_v1.py)
- current sanity-check package for that candidate:
  - [edge_long_tier15_sanity_check_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier15_sanity_check_v1.py)
- direct baseline comparison package:
  - [edge_long_tier14_vs_tier15_comparison_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier14_vs_tier15_comparison_v1.py)
- residual search package from Tier 1.4:
  - [edge_long_tier14_satellite4_search_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier14_satellite4_search_v1.py)
- current long structure:
  - [edge_long_tier11_core_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_tier11_core_v1.py)
  - first viable long direction looks more like `bear-flush reversal` than `bull continuation`
  - first high-purity core inside that direction is:
    - `bear-like + high_vol`
    - `climax_drop_5m_pct >= 0.85`
    - `context_stretch_below_ema20_5m_pct <= 1.30`
    - result: `5 trades`, `100% win rate`, `ending_balance 440.56`
  - first refined winning satellite attached to that core is:
    - softer flush: `climax_drop_5m_pct < 0.85`
    - `confirm_range_pct <= 0.25`
    - `rel_volume_1m >= 1.40`
    - `context_dump_60m_pct >= 1.10`
    - `trigger_body_ratio <= 0.45`
    - combined result (`Tier 1.3`): `9 trades`, `100% win rate`, `ending_balance 483.87`
  - second winning satellite attached on top is:
    - `confirm_range_pct >= 0.40`
    - `trigger_body_ratio <= 0.35`
    - `trigger_lower_wick_ratio >= 0.45`
    - combined result (`Tier 1.4`): `12 trades`, `100% win rate`, `ending_balance 513.38`
  - third satellite exists, but it is explicitly tiny:
    - `rel_volume_1m >= 2.50`
    - `trigger_body_ratio >= 0.40`
    - combined result (`Tier 1.5`): `14 trades`, `100% win rate`, `ending_balance 529.80`
    - treat it as a micro-satellite, not a broad robust family
  - sanity-check result for `Tier 1.5`:
    - both halves positive
    - all non-empty time windows positive
    - but route concentration still matters:
      - `satellite1` is early-history heavy
      - `satellite3` is late-history only
      - `tier11_core` is the most stable cross-time route
  - current practical baseline choice:
    - `Tier 1.4` = cleaner main long baseline
    - `Tier 1.5` = wider, stronger, but more aggressive expanded version
  - realistic ETH/Binance-style execution stress result for `Tier 1.4`:
    - `Baseline realistic`: `12 trades`, `ending_balance 513.38`, `win_rate 100%`
    - `ETH adverse light`: `11 trades`, `ending_balance 494.84`, `win_rate 100%`
    - `ETH adverse medium`: `11 trades`, `ending_balance 491.40`, `win_rate 100%`
    - `ETH adverse hard`: `11 trades`, `ending_balance 487.62`, `win_rate 100%`
    - route impact:
      - `tier11_core` survives intact across all scenarios
      - `satellite2_reclaim_extension` survives intact across all scenarios
      - only `satellite1_soft_flush_refined` loses one stressed fill
  - `satellite4` search from the clean `Tier 1.4` residual found no qualified 3+ trade pocket
    - residual after `Tier 1.4`: `17` trades, only `2` wins
    - conclusion: `Tier 1.4` residual is mostly exhausted
  - external source-of-truth from `Another Trader` real ETH longs:
    - `55` ETHUSDT long reference trades
    - `100%` wins in the available sample
    - dominant regime/context:
      - `strong_bear = 46`
      - `reversal_like = 34`
      - `near_low = 33`
      - `bearish 15m EMA alignment = 43`
      - `bearish 1h EMA alignment = 39`
    - most useful named long-family pockets:
      - `family_a_flush_reversal`
        - `near_low + reversal_like + bearish_15m + bearish_1h + dump60<=-0.5`
        - `21 trades`
      - `family_b_deep_flush_reversal`
        - `near_low + reversal_like + bearish_15m + dist15<=-0.5 + dist1h<=-1.0`
        - `18 trades`
      - `family_c_breakout_down_reclaim`
        - `breakout_down + bearish_15m + bearish_1h + dump60<=-1.0`
        - `7 trades`
    - practical implication:
      - do not keep pushing `Tier 1.4` residual
      - the next long search should start as a new flush-reversal family branch
  - honest router probe result on the old `wickMick_long` engine:
    - [edge_long_another_trader_router_probe_v1/summary.txt](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/results/edge_long_another_trader_router_probe_v1/summary.txt)
    - `family_a_flush_reversal` did not become a clean new baseline through router-only changes
    - best `family_a` router on the old engine:
      - `14 trades`
      - `42.86% win rate`
      - `ending_balance 407.05`
    - `family_b` and `family_c` were also weak on the old engine
    - practical implication:
      - the next long step is not another router on top of `wickMick_long`
      - the next long step should be a new independent engine for the flush-reversal family
  - important follow-up:
    - [edge_long_family2_flush_reversal_seed_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_family2_flush_reversal_seed_v1.py)
    - [edge_long_family2_flush_reversal_seed_v1/summary.txt](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/results/edge_long_family2_flush_reversal_seed_v1/summary.txt)
    - wins/losses inside the noisy `family_a` router were not random:
      - losses tended to have too-thin trigger bodies, weaker `rel_volume_1m`, and/or more overstretched local context
    - the first clean extracted seed from that analysis is:
      - `strong_bear`
      - `high_vol`
      - `near_low`
      - `reversal_like`
      - bearish `15m` + `1h` EMA alignment
      - `return_pre_60m_pct <= -0.5`
      - `context_stretch_below_ema20_5m_pct <= 1.10`
      - `0.14 <= trigger_body_ratio <= 0.26`
      - `rel_volume_1m >= 1.20`
    - result:
      - `8 trades`
      - `100% win rate`
      - `ending_balance 466.49`
    - practical implication:
      - keep `Tier 1.4` as the main long baseline
      - treat `family2_flush_reversal_seed_v1` as the first viable new long-family seed for expansion
  - current expanded Family 2 branch:
    - [edge_long_family21_seed_plus_satellites_v1.py](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/edge_long_family21_seed_plus_satellites_v1.py)
    - [edge_long_family21_seed_plus_satellites_v1/summary.txt](/C:/Users/Tigran/Desktop/Binance/next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/summary.txt)
    - result:
      - `12 trades`
      - `100% win rate`
      - `ending_balance 497.35`
    - route split:
      - `family2_seed`: `8 trades`
      - `family2_satellite_highconfirm`: `2 trades`
      - `family2_satellite_midbody`: `2 trades`
    - practical implication:
      - `Family 2.1` is now a serious expandable long branch
      - next comparisons should treat it as a real long candidate alongside `Tier 1.4`

## 3. Current Bot Truth

Bot project path:

- [eth_short_demo_bot](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot)

Bot design:

- Binance Futures DEMO
- Docker / Docker Compose
- `.env`-based config
- Telegram notifications
- SQLite persistence
- restart-safe recovery
- `ETHUSDT` only
- `SHORT only`
- leverage `10x`

Main code:

- [eth_short_demo_bot/bot/service.py](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/bot/service.py)
- [eth_short_demo_bot/bot/strategy.py](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/bot/strategy.py)
- [eth_short_demo_bot/bot/exchange.py](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/bot/exchange.py)
- [eth_short_demo_bot/bot/config.py](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/bot/config.py)

## 4. Current Reporting Truth

Daily report was upgraded and should now include:

- trend regime
- vol regime
- route availability
- market verdict
- closest candidate
- stronger no-trade explanation

Example desired wording:

- day was bullish, short routes restricted
- closest candidate time + score + why it was blocked

If Telegram still shows repeated:

- `Daily report loop failed`
- or old weak report format

then the server is almost certainly running an old bot version and needs a fresh clean redeploy.

## 5. Server / Deploy Truth

Known server used in this project:

- host: `178.160.194.52`
- port: `23019`
- user: `root`
- expected path: `/root/eth_short_demo_bot`

Important rule:

- prefer **full clean redeploy from fresh zip**
- avoid stacking many manual file-by-file edits on the server

Fresh archive prepared locally:

- `C:\\Users\\Tigran\\Desktop\\Binance\\eth_short_demo_bot_fresh.zip`

Read first for deploy work:

- [eth_short_demo_bot/SERVER_RUNBOOK.md](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/SERVER_RUNBOOK.md)

## 6. Known Pain Points

Past failures came from:

- old server code still running after partial updates
- wrong CLI argument order in Docker command
- repeated Telegram spam from periodic recovery or daily loop errors
- missing import for `_build_market_regime_from_daily`
- server folder drift after hand-editing files directly on the server

If runtime behavior looks inconsistent with local code, assume server drift first and redeploy cleanly.

## 7. Safe Working Rules For New Codex Chats

When continuing this project:

- start by reading this file
- then read:
  - [next_phase_edge_focus/README.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/README.md)
  - [next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md](/C:/Users/Tigran/Desktop/Binance/next_phase_edge_focus/HOW_TO_TRADE_TIER287_REALISTIC.md)
  - [eth_short_demo_bot/README.md](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/README.md)
  - [eth_short_demo_bot/SERVER_RUNBOOK.md](/C:/Users/Tigran/Desktop/Binance/eth_short_demo_bot/SERVER_RUNBOOK.md)
- do not change `Tier 2.87` logic casually
- separate strategy changes from bot/deploy/logging changes
- if changing the bot, prefer updating local files, rebuilding fresh zip, and redeploying cleanly

## 8. What To Avoid

- do not treat old root-level scripts as active truth by default
- do not trust server state over local source of truth
- do not do piecemeal server surgery unless the fix is tiny and urgent
- do not reveal API keys / bot tokens / secrets in chat

## 9. Good First Diagnostic Questions

If something looks wrong, check these first:

- is the container `healthy`?
- is `RestartCount` stable?
- is the server running the latest local code?
- are Telegram messages using the new report format?
- is the issue real strategy logic, or just reporting / deployment drift?

## 10. Short Summary

Current reality:

- strategy baseline: `Tier 2.87 realistic`
- bot project: `eth_short_demo_bot`
- safest deploy method: clean redeploy from fresh zip
- primary docs: strategy README + bot runbooks
