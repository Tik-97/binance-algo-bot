# ETH Long Edge Focus

This folder is the clean restart point for the ETH long branch.

## Starting point

We are not starting from zero.

Best old long candidate found so far:

- source script: [wickMick_long.py](/wickMick_long.py)
- result folder: [cluster1_long_climax_push_failure_rr2_reentry_delayed_limit](/fun_backtests/cluster1_long_climax_push_failure_rr2_reentry_delayed_limit)
- old summary: [cluster1_long_summary.txt](/fun_backtests/cluster1_long_climax_push_failure_rr2_reentry_delayed_limit/cluster1_long_summary.txt)

Why this one:

- `trade_count = 19`
- `win_rate = 42.11%`
- `profit_factor = 1.133`
- `ending_balance = 407.35`

This is still weak, but it is a better starting survivor than the older negative long branches.

## Current goal

Use the same honest process we used for the short branch:

1. fix a clean long baseline candidate
2. measure where it actually works
3. expand only if the data justifies it
4. avoid pretending weak frequency or weak expectancy is already an edge

## First active step

First script in this branch:

- [edge_long_regime_dependency_analysis_v1.py](/next_phase_long_focus/edge_long_regime_dependency_analysis_v1.py)
- [edge_long_tier1_bear_highvol_baseline_v1.py](/next_phase_long_focus/edge_long_tier1_bear_highvol_baseline_v1.py)
- [edge_long_tier11_core_v1.py](/next_phase_long_focus/edge_long_tier11_core_v1.py)
- [edge_long_tier12_core_plus_satellite1_v1.py](/next_phase_long_focus/edge_long_tier12_core_plus_satellite1_v1.py)
- [edge_long_tier13_core_plus_satellite1_refined_v1.py](/next_phase_long_focus/edge_long_tier13_core_plus_satellite1_refined_v1.py)
- [edge_long_tier14_core_plus_satellite2_v1.py](/next_phase_long_focus/edge_long_tier14_core_plus_satellite2_v1.py)
- [edge_long_tier15_core_plus_satellite3_v1.py](/next_phase_long_focus/edge_long_tier15_core_plus_satellite3_v1.py)
- [edge_long_tier15_sanity_check_v1.py](/next_phase_long_focus/edge_long_tier15_sanity_check_v1.py)
- [edge_long_tier14_vs_tier15_comparison_v1.py](/next_phase_long_focus/edge_long_tier14_vs_tier15_comparison_v1.py)
- [edge_long_tier14_satellite4_search_v1.py](/next_phase_long_focus/edge_long_tier14_satellite4_search_v1.py)
- [edge_long_tier14_eth_stress_test_v1.py](/next_phase_long_focus/edge_long_tier14_eth_stress_test_v1.py)
- [edge_long_another_trader_family_analysis_v1.py](/next_phase_long_focus/edge_long_another_trader_family_analysis_v1.py)

First output folder:

- [edge_long_regime_dependency_analysis_v1](/next_phase_long_focus/results/edge_long_regime_dependency_analysis_v1)
- [edge_long_tier1_bear_highvol_baseline_v1](/next_phase_long_focus/results/edge_long_tier1_bear_highvol_baseline_v1)
- [edge_long_tier11_core_v1](/next_phase_long_focus/results/edge_long_tier11_core_v1)
- [edge_long_tier12_core_plus_satellite1_v1](/next_phase_long_focus/results/edge_long_tier12_core_plus_satellite1_v1)
- [edge_long_tier13_core_plus_satellite1_refined_v1](/next_phase_long_focus/results/edge_long_tier13_core_plus_satellite1_refined_v1)
- [edge_long_tier14_core_plus_satellite2_v1](/next_phase_long_focus/results/edge_long_tier14_core_plus_satellite2_v1)
- [edge_long_tier15_core_plus_satellite3_v1](/next_phase_long_focus/results/edge_long_tier15_core_plus_satellite3_v1)
- [edge_long_tier15_sanity_check_v1](/next_phase_long_focus/results/edge_long_tier15_sanity_check_v1)
- [edge_long_tier14_vs_tier15_comparison_v1](/next_phase_long_focus/results/edge_long_tier14_vs_tier15_comparison_v1)
- [edge_long_tier14_satellite4_search_v1](/next_phase_long_focus/results/edge_long_tier14_satellite4_search_v1)
- [edge_long_tier14_eth_stress_test_v1](/next_phase_long_focus/results/edge_long_tier14_eth_stress_test_v1)
- [edge_long_another_trader_family_analysis_v1](/next_phase_long_focus/results/edge_long_another_trader_family_analysis_v1)

First key result:

- this long survivor did **not** look best in bull-like daily regimes
- surprisingly, it was most alive in `strong_bear + high_vol`
- this suggests the current long candidate behaves more like a flush / reversal long than a classic bull-trend continuation long

Quick filter screen from the first analysis:

- `bear_like + high_vol`: `13` trades, `net_pnl = +20.05`, `win_rate = 53.85%`, `pf = 1.655`
- `strong_bear + high_vol`: `12` trades, `net_pnl = +13.02`, `win_rate = 50.00%`, `pf = 1.425`
- `bull_like only`: `1` trade, `net_pnl = -4.56`

So the first practical direction is already clear:

- do **not** start the long branch from `bull-only`
- start from `bear-flush reversal` conditions instead

## First active baseline

The first independent long baseline is now:

- [edge_long_tier1_bear_highvol_baseline_v1.py](/next_phase_long_focus/edge_long_tier1_bear_highvol_baseline_v1.py)

It is not a post-filter over old trades.

It is a standalone full-history backtest that:

- reuses the old long climax-dump-failure entry/exit engine
- removes the old JSON-window dependency
- adds a top-level router:
  - `old long context`
  - `bear-like daily regime`
  - `high_vol`

Current result:

- `29` trades
- `ending_balance = 438.06`
- `win_rate = 48.28%`
- `profit_factor = 1.473`

This is materially better than the old delayed-limit survivor.

## First clean core

The first small high-purity long core is now:

- [edge_long_tier11_core_v1.py](/next_phase_long_focus/edge_long_tier11_core_v1.py)

It is built from the new Tier 1 baseline, not from old clutter.

Core idea:

- stay inside `bear-like + high_vol`
- require a real local flush:
  - `climax_drop_5m_pct >= 0.85`
- but avoid the most stretched exhaustion cases:
  - `context_stretch_below_ema20_5m_pct <= 1.30`

Current result:

- `5` trades
- `ending_balance = 440.56`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Interpretation:

- this is the long equivalent of the first short core:
  - very small
  - very selective
  - high purity first
- next steps should be satellite expansion around this core, not random long ideas

## First winning satellite

The first honest satellite search produced:

- [edge_long_tier12_core_plus_satellite1_v1.py](/next_phase_long_focus/edge_long_tier12_core_plus_satellite1_v1.py)

Important lesson:

- the first pretty `CSV-only` satellite looked too clean
- when replayed as a real backtest, it added `1` loss
- that is exactly why this branch should keep using real reruns, not only post-filters

Then we refined it one step:

- [edge_long_tier13_core_plus_satellite1_refined_v1.py](/next_phase_long_focus/edge_long_tier13_core_plus_satellite1_refined_v1.py)

Refined structure:

- `Tier 1.1 core`
- plus `satellite1_soft_flush_refined`

Satellite idea:

- softer flush than the core:
  - `climax_drop_5m_pct < 0.85`
- clean confirm:
  - `confirm_range_pct <= 0.25`
- healthy local volume:
  - `rel_volume_1m >= 1.40`
- real local damage:
  - `context_dump_60m_pct >= 1.10`
- calmer trigger body:
  - `trigger_body_ratio <= 0.45`

Current combined result:

- `9` trades
- `ending_balance = 483.87`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Route split:

- `tier11_core`: `5` trades, `+43.97`
- `satellite1_soft_flush_refined`: `4` trades, `+39.90`

## Second winning satellite

The next step produced:

- [edge_long_tier14_core_plus_satellite2_v1.py](/next_phase_long_focus/edge_long_tier14_core_plus_satellite2_v1.py)

Satellite 2 idea:

- stronger confirm than the soft-flush satellite:
  - `confirm_range_pct >= 0.40`
- cleaner trigger body:
  - `trigger_body_ratio <= 0.35`
- meaningful lower-wick rejection:
  - `trigger_lower_wick_ratio >= 0.45`

Interpretation:

- this is a more obvious reclaim / continuation-after-rejection cluster
- unlike satellite 1, it is not defined mainly by softer flush + low confirm range

Current combined result after adding satellite 2:

- `12` trades
- `ending_balance = 513.38`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Route split:

- `tier11_core`: `5` trades, `+46.06`
- `satellite1_soft_flush_refined`: `4` trades, `+40.91`
- `satellite2_reclaim_extension`: `3` trades, `+26.42`

## Third satellite

There was no broad clean pocket left after `Tier 1.4`.

Only `2` winners remained in the residual set, so the next addition is explicitly a micro-satellite:

- [edge_long_tier15_core_plus_satellite3_v1.py](/next_phase_long_focus/edge_long_tier15_core_plus_satellite3_v1.py)

Satellite 3 idea:

- very high local volume:
  - `rel_volume_1m >= 2.50`
- stronger trigger body:
  - `trigger_body_ratio >= 0.40`

Interpretation:

- this is not a broad family
- it is a tiny `high-conviction micro-cluster`
- it should be treated more carefully than satellites 1 and 2

Current combined result after adding satellite 3:

- `14` trades
- `ending_balance = 529.80`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Route split:

- `tier11_core`: `5` trades, `+47.10`
- `satellite1_soft_flush_refined`: `4` trades, `+40.91`
- `satellite2_reclaim_extension`: `3` trades, `+26.94`
- `satellite3_highconviction_micro`: `2` trades, `+14.86`

## Tier 1.5 Sanity Check

We already ran a dedicated time-split sanity package:

- [edge_long_tier15_sanity_check_v1.py](/next_phase_long_focus/edge_long_tier15_sanity_check_v1.py)

Main result:

- both `first_half` and `second_half` were positive
- all `6` non-empty time windows were positive
- no single window was negative

Important nuance:

- the structure is not evenly distributed across all routes
- `satellite1` is mostly an earlier-history route
- `satellite3` is only a very late micro-route
- `tier11_core` survives in both halves and becomes stronger later

So the honest read is:

- `Tier 1.5` passed the first sanity check
- but the later satellites are still more time-local than the core

## Tier 1.4 vs Tier 1.5

Direct comparison package:

- [edge_long_tier14_vs_tier15_comparison_v1.py](/next_phase_long_focus/edge_long_tier14_vs_tier15_comparison_v1.py)

Headline result:

- `Tier 1.4`: `12` trades, `ending_balance = 513.38`
- `Tier 1.5`: `14` trades, `ending_balance = 529.80`

Honest verdict:

- `Tier 1.5` is the wider and stronger raw-history version
- `Tier 1.4` is the cleaner structural baseline
- best practical interpretation:
  - `Tier 1.4` = safer main long baseline
  - `Tier 1.5` = more aggressive expanded version

## Satellite 4 Search From Tier 1.4

Residual-search package:

- [edge_long_tier14_satellite4_search_v1.py](/next_phase_long_focus/edge_long_tier14_satellite4_search_v1.py)

Result:

- no qualified `satellite4` candidate was found from the residual after `Tier 1.4`
- residual after `Tier 1.4` had:
  - `17` trades
  - only `2` wins
  - `15` losses

Honest read:

- `Tier 1.4` residual is already mostly exhausted
- if we want more frequency, the next step should not be another clean satellite from the same residual

## Tier 1.4 Realistic ETH Stress Test

Execution-stress package:

- [edge_long_tier14_eth_stress_test_v1.py](/next_phase_long_focus/edge_long_tier14_eth_stress_test_v1.py)

Purpose:

- keep `Tier 1.4` as the clean long baseline
- keep fees fixed
- stress only the parts that are realistic for ETH / Binance style execution:
  - worse limit touch
  - worse limit fill
  - worse market entry slippage
  - worse exit slippage

Headline result:

- `Baseline realistic`: `12` trades, `ending_balance = 513.38`, `win_rate = 100.00%`
- `ETH adverse light`: `11` trades, `ending_balance = 494.84`, `win_rate = 100.00%`
- `ETH adverse medium`: `11` trades, `ending_balance = 491.40`, `win_rate = 100.00%`
- `ETH adverse hard`: `11` trades, `ending_balance = 487.62`, `win_rate = 100.00%`

What actually degrades:

- the structure stays alive in all tested scenarios
- the core survives intact:
  - `tier11_core` stays at `5` trades in every scenario
- `satellite2_reclaim_extension` also survives intact:
  - `3` trades in every scenario
- only `satellite1_soft_flush_refined` loses one fill under stressed execution:
  - `4` trades in baseline
  - `3` trades in light / medium / hard

Honest verdict:

- `Tier 1.4` survives realistic ETH-style execution pressure well
- this is a good sign for structure quality, not just a pretty frictionless curve
- the cleanest practical long baseline remains:
  - `Tier 1.4` as main baseline
  - `Tier 1.5` only as the wider aggressive variant

## Another Trader Long Family Analysis

New external-source analysis package:

- [edge_long_another_trader_family_analysis_v1.py](/next_phase_long_focus/edge_long_another_trader_family_analysis_v1.py)

Purpose:

- stop overworking the exhausted `Tier 1.4` residual
- use real `Another Trader` ETH long trades as a fresh source of family ideas
- identify the next long-family direction cleanly

Headline result:

- `55` real `ETHUSDT Long` trades
- `100%` wins in the available reference sample
- dominant regime:
  - `strong_bear = 46 trades`
- dominant context:
  - `reversal_like = 34 trades`
  - `near_low = 33 trades`
- dominant alignment:
  - `bearish 15m EMA alignment = 43 trades`
  - `bearish 1h EMA alignment = 39 trades`

Most useful named family pockets:

- `family_a_flush_reversal`
  - `near_low + reversal_like + bearish_15m + bearish_1h + dump60<=-0.5`
  - `21` trades
- `family_b_deep_flush_reversal`
  - `near_low + reversal_like + bearish_15m + dist15<=-0.5 + dist1h<=-1.0`
  - `18` trades
- `family_c_breakout_down_reclaim`
  - `breakout_down + bearish_15m + bearish_1h + dump60<=-1.0`
  - `7` trades

Honest read:

- `Another Trader` longs also point to a bearish-flush reversal direction
- this is not a classic bull-continuation family
- the next correct expansion path is:
  - keep `Tier 1.4` as main long baseline
  - build a new long-family search from these `Another Trader` flush-reversal pockets
  - stop trying to force `satellite4` out of the old `Tier 1.4` residual

## Another Trader Router Probe On Old Engine

Router probe package:

- [edge_long_another_trader_router_probe_v1/summary.txt](/next_phase_long_focus/results/edge_long_another_trader_router_probe_v1/summary.txt)

Purpose:

- test whether the `Another Trader` long-family pockets can be recovered just by adding a smarter top-level router
- keep the old `wickMick_long` entry/exit engine unchanged
- answer the honest question:
  - is the missing edge only a router problem
  - or is it a different long engine family altogether

Headline result:

- `family_a_flush_reversal` did **not** produce a clean new baseline on the old engine
- best `family_a` router:
  - `f2a_core_sb_highvol`
  - `14` trades
  - `42.86%` win rate
  - `ending_balance = 407.05`
- `family_b_deep_flush_reversal` was worse:
  - `15` trades
  - `26.67%` win rate
  - `ending_balance = 370.12`
- `family_c_breakout_down_reclaim` also failed to become a positive clean branch:
  - `5` trades
  - `40.00%` win rate
  - `ending_balance = 399.39`

Honest read:

- the `Another Trader` long-family pockets do not transfer cleanly through the old `wickMick_long` engine
- this means the problem is no longer only the top-level regime/router
- the old long engine is likely the wrong expression for this new family

Practical implication:

- keep `Tier 1.4` as the main long baseline
- stop trying to route `family_a / family_b / family_c` through `wickMick_long`
- the correct next step is:
  - build a new independent long engine for the flush-reversal family itself

## Family2 Flush-Reversal Seed

New seed branch:

- [edge_long_family2_flush_reversal_seed_v1.py](/next_phase_long_focus/edge_long_family2_flush_reversal_seed_v1.py)
- [summary.txt](/next_phase_long_focus/results/edge_long_family2_flush_reversal_seed_v1/summary.txt)
- [trades.csv](/next_phase_long_focus/results/edge_long_family2_flush_reversal_seed_v1/trades.csv)

Purpose:

- take the noisy `family_a_flush_reversal` router
- analyze what actually separated its wins from its losses
- keep only the live-safe pocket that survived honest rerun

What unified the wins vs losses in the probe:

- losses were more likely to have:
  - too-thin trigger bodies
  - weaker local sponsorship / lower `rel_volume_1m`
  - more overstretched local context
- the cleaner wins clustered around:
  - balanced trigger body
  - still-strong local volume
  - controlled stretch below `EMA20 5m`

Seed filters:

- `strong_bear`
- `high_vol`
- `near_low`
- `reversal_like`
- `bearish 15m EMA alignment`
- `bearish 1h EMA alignment`
- `return_pre_60m_pct <= -0.5`
- `context_stretch_below_ema20_5m_pct <= 1.10`
- `0.14 <= trigger_body_ratio <= 0.26`
- `rel_volume_1m >= 1.20`

Headline result:

- `8` trades
- `ending_balance = 466.49`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Honest read:

- this is not yet the new main long baseline
- but it is a real new seed branch
- unlike the broad router-only family probe, this one actually isolated a clean pocket
- the next reasonable move is to expand around this `family2` seed, not to go back to brute-force router search

## Family2.1 Seed Plus Satellites

New expansion package:

- [edge_long_family21_seed_plus_satellites_v1.py](/next_phase_long_focus/edge_long_family21_seed_plus_satellites_v1.py)
- [summary.txt](/next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/summary.txt)
- [trades.csv](/next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/trades.csv)
- [route_summary.csv](/next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/route_summary.csv)

Purpose:

- continue Family 2 expansion only from honest residual analysis
- avoid brute-force router search
- add only live-safe pockets that survived a full rerun

Added satellites:

- `family2_satellite_highconfirm`
  - `trigger_body_ratio >= 0.45`
  - `confirm_range_pct >= 0.45`
  - `context_dump_60m_pct <= 1.05`
- `family2_satellite_midbody`
  - `0.29 <= trigger_body_ratio <= 0.35`
  - `confirm_range_pct <= 0.45`
  - `rel_volume_1m >= 1.20`

Headline result:

- `12` trades
- `ending_balance = 497.35`
- `win_rate = 100.00%`
- `max_drawdown = 0.00%`

Route split:

- `family2_seed`: `8` trades, `+67.79`
- `family2_satellite_highconfirm`: `2` trades, `+16.18`
- `family2_satellite_midbody`: `2` trades, `+13.38`

Honest read:

- this is a real expansion, not just a seed
- both extra pockets survived honest rerun and stayed clean
- `Family 2.1` is now a serious second long branch, not only an exploratory hint

## Working rule

Do not overwrite old long scripts.

If a new idea appears:

- create a new script in this folder
- save outputs under `next_phase_long_focus/results/<name>/`

## Honest status

At this moment:

- `Tier 1.1 core` is the first serious long core candidate
- `Tier 1.3` is the first clean `core + satellite` candidate
- `Tier 1.4` is now the current best structured long candidate
- `Tier 1.5` is the current widest still-clean long candidate
- it is still small and research-only
- this branch is not live-ready yet
- the purpose is now:
  - keep this core clean
  - attach only profitable satellites around it
  - keep rerunning full backtests after every expansion
