# How To Trade Long T14

## Status
`STUB - not yet live. Logic not implemented.`

## One-line summary
Tier 1.4 is the current clean long baseline from research, but it is not yet ported into the live bot.

## What this strategy is (plain language, no jargon)
This long branch buys ETH after bearish flushes when price starts to reclaim from local damage. It is not a normal trend-following long. It is a reversal long built inside bearish and high-volatility conditions.

## Market conditions required (regime, vol, session)
- intended regime: `strong_bear` or `weak_bear`
- intended volatility: `high_vol`
- temporary live session assumption: `13:00-16:00 UTC`
- current bot reality: stub only, so no live signals

## What four charts to open and what to look for on each
- `1D`: confirm ETH is still in bearish or bear-recovery context
- `1H`: confirm local damage and potential reclaim from lows
- `5m`: inspect flush severity and reclaim structure
- `1m`: inspect exact trigger candle body, wick, and confirm behavior

## Signal layers (each route/subtype with exact parameter values)
Core route from research:
- `climax_drop_5m_pct >= 0.85`
- `context_stretch_below_ema20_5m_pct <= 1.30`

Satellite 1:
- `climax_drop_5m_pct < 0.85`
- `confirm_range_pct <= 0.25`
- `rel_volume_1m >= 1.40`
- `context_dump_60m_pct >= 1.10`
- `trigger_body_ratio <= 0.45`

Satellite 2:
- `confirm_range_pct >= 0.40`
- `trigger_body_ratio <= 0.35`
- `trigger_lower_wick_ratio >= 0.45`

## Entry mechanics (order type, where limit goes, where stop goes, why)
- intended research behavior comes from `edge_long_tier14_core_plus_satellite2_v1.py`
- entry mode in research is usually `shallow_pullback_limit`
- stop is placed below flush/reclaim failure level
- exact live adapter is not implemented yet

## Position sizing
- when this strategy becomes live, it will use the same unified shared-wallet sizing as the other branches
- right now it has zero effect on sizing because it is a stub

## What a good setup looks like (step by step visual checklist)
- ETH is still in bearish context, not healthy bull trend continuation
- price flushes down and then reclaims cleanly
- trigger candle does not look exhausted in the wrong direction
- local reclaim looks controlled, not random chop

## What a bad setup looks like (skip conditions, exact reasons)
- bullish continuation environment instead of bearish flush reversal
- no real reclaim, only weak sideways noise
- trigger body/wick profile does not match the route
- low quality local context despite high volatility

## How to verify each trade post-factum
Open `next_phase_long_focus/results/edge_long_tier14_core_plus_satellite2_v1/trades.csv` and check:
- `entry_mode`
- `route`
- `trend_regime`
- `vol_regime`
- `entry_time`
- `pnl_usd`

## Daily manual checklist
- confirm this file still says `STUB`
- if live bot is running, remember `long_t14` should generate zero real signals
- use research results only for reference until status changes

## Honest edge and limitations
- research result is strong and realistic-stress survival was good
- but live implementation is still missing
- until status changes to LIVE, this strategy generates zero signals and has zero effect on the bot
