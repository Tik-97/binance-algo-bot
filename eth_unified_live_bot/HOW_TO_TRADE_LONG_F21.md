# How To Trade Long F21

## Status
`STUB - not yet live. Logic not implemented.`

## One-line summary
Family 2.1 is the newer flush-reversal long family with seed plus satellites, but it is not yet ported into the live bot.

## What this strategy is (plain language, no jargon)
This is a second long family that looks for ETH near lows after heavy bearish pressure, then buys reversal-like reclaim behavior. It is a separate family from Tier 1.4, not just a minor variant.

## Market conditions required (regime, vol, session)
- regime: `strong_bear`
- volatility: `high_vol`
- structure: `near_low`
- context: `reversal_like`
- EMA alignment on `15m` and `1h`: bearish
- pre-entry `60m` return: `<= -0.5%`
- temporary live session assumption: `13:00-16:00 UTC`
- current bot reality: stub only

## What four charts to open and what to look for on each
- `1D`: confirm strong bear backdrop
- `1H`: confirm ETH is near lows and under bearish EMA structure
- `15m`: check bearish EMA alignment and reversal-like reclaim context
- `1m`: inspect exact trigger body, volume, and satellite rules

## Signal layers (each route/subtype with exact parameter values)
Family 2 seed:
- `strong_bear`
- `high_vol`
- `near_low`
- `reversal_like`
- bearish `15m` EMA alignment
- bearish `1h` EMA alignment
- `return_pre_60m_pct <= -0.5`
- `context_stretch_below_ema20_5m_pct <= 1.10`
- `0.14 <= trigger_body_ratio <= 0.26`
- `rel_volume_1m >= 1.20`

Family 2.1 satellite highconfirm:
- `trigger_body_ratio >= 0.45`
- `confirm_range_pct >= 0.45`
- `context_dump_60m_pct <= 1.05`

Family 2.1 satellite midbody:
- `0.29 <= trigger_body_ratio <= 0.35`
- `confirm_range_pct <= 0.45`
- `rel_volume_1m >= 1.20`

## Entry mechanics (order type, where limit goes, where stop goes, why)
- intended behavior comes from `edge_long_family21_seed_plus_satellites_v1.py`
- research branch uses long reversal entry around reclaim zone
- stop belongs below failed reclaim / flush continuation
- exact live adapter is not implemented yet

## Position sizing
- once ported live, it should use the same shared-wallet sizing as the rest of the unified bot
- right now it has zero effect because it is a stub

## What a good setup looks like (step by step visual checklist)
- ETH is still in strong bear context
- price is near local lows
- reclaim is reversal-like, not trend-continuation noise
- bearish EMA structure is still visible on `15m` and `1h`
- trigger candle fits either seed or one of the F2.1 satellites

## What a bad setup looks like (skip conditions, exact reasons)
- no near-low structure
- no reversal-like reclaim
- daily or intraday context is not strong bear / high vol
- trigger body sits outside seed or satellite ranges
- local context looks random rather than flush-reversal

## How to verify each trade post-factum
Open `next_phase_long_focus/results/edge_long_family21_seed_plus_satellites_v1/trades.csv` and check:
- `entry_mode`
- `route`
- `trend_regime`
- `vol_regime`
- `entry_time`
- `pnl_usd`

## Daily manual checklist
- confirm this file still says `STUB`
- if live bot is running, remember `long_f21` should generate zero real signals
- use research outputs only for reference until status changes

## Honest edge and limitations
- Family 2.1 is a real research branch and not just a tiny tweak
- it overlaps partly with Tier 1.4 but also has unique trades
- live implementation is still missing
- until status changes to LIVE, this strategy generates zero signals and has zero effect on the bot
