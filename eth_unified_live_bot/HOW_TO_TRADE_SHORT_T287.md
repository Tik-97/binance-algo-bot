# How To Trade Short T287

## Status
`LIVE - active in production`

## One-line summary
Short `ETHUSDT` only during the controlled `13:00-16:00 UTC` session when the daily backdrop supports bearish pullback failure.

## What this strategy is (plain language, no jargon)
This is a short-only ETH model that waits for price to bounce up into local resistance, then sells the failure of that bounce. It does not chase panic candles. It wants already closed `1m` candles and a bearish daily backdrop.

## Market conditions required (regime, vol, session)
- Session: `13:00-16:00 UTC`
- Core routes: daily `trend_regime` must be `strong_bear` or `weak_bear`
- Satellite route: daily `trend_regime` must be `strong_bear`
- Expansion route: daily `trend_regime` must be `strong_bear` and `vol_regime` must be `high_vol`

## What four charts to open and what to look for on each
- `1D`: confirm daily backdrop is bearish or at least not bullish against the short
- `1H`: check whether price is near recent highs and whether the bounce is stretched into resistance
- `5m`: judge whether the push up is noisy, cooling, or mixed transition
- `1m`: verify the exact trigger candle, body, wick, local volume, and distance from highs

## Signal layers (each route/subtype with exact parameter values)
Core candidate gate:
- session hour must satisfy `13 <= hour_utc < 16`
- `dist_4h_high_pct <= 0.80`
- `dist_1h_high_pct <= 0.60`
- `ret_60m_pct >= 0.20`
- `ret_60m_pct <= 3.50`
- `ret_15m_pct >= 0.10`
- `ret_5m_pct >= -0.10`
- `ret_5m_pct <= 1.80`
- `close_above_ema20_pct >= -0.10`
- `close_above_ema20_pct <= 2.50`
- `rel_volume_1m <= 4.50`
- `upper_wick_ratio >= 0.05` or `body_ratio <= 0.85`

Core regime classifier:
- `near_entry_breakdown_noise` if:
  - `relative_volume_1m >= 1.35`
  - `move_5m_pct >= 0.55`
  - `bull_body_sum_5m_pct >= 0.65`
  - `stretch_above_ema20_5m_pct >= 0.90`
  - `green_count_last_5 >= 4`

Core mild route:
- subtype = `mild_noise_v1_live`
- route = `tier2p5_core_mild_limit`
- limit retest price = `trigger_close * 1.0015`
- stop buffer above trigger high = `0.10%`
- TTL = `15` bars

Core strict cooling route:
- subtype = `cooling_v3_live`
- route = `tier2p5_core_strict_cooling_limit`
- extra filters:
  - `move_1m_pct <= 0.00`
  - `move_3m_pct >= 0.50`
  - `dist_1h_high_pct >= 0.10`
  - `bounce_60m_pct <= 1.20`

Core selective cooling route:
- subtype = `cooling_v3_live`
- route = `tier2p5_core_selective_cooling_market`
- extra filters:
  - `move_1m_pct >= -0.30`
  - `move_3m_pct >= 0.00`
  - `move_3m_pct <= 0.50`

Satellite regime classifier:
- `mixed_transition` if:
  - `relative_volume_1m <= 1.20`
  - `bounce_60m_pct <= 1.50`
  - `stretch_above_ema20_5m_pct <= 1.40`
  - `green_count_last_5 <= 3`
  - `dist_1h_high_pct >= 0.10`

Satellite calm filter:
- `move_1m_pct <= 0.11`
- `move_3m_pct <= 0.18`
- `move_5m_pct <= 0.25`
- `relative_volume_1m <= 1.20`
- `bounce_60m_pct >= 0.24`
- `bounce_60m_pct <= 1.20`
- `dist_1h_high_pct >= 0.10`
- `dist_1h_high_pct <= 0.22`
- `bull_body_sum_5m_pct <= 0.32`
- `green_count_last_5 in {2, 3}`

Satellite RV route:
- route = `tier5_satellite_rv08_market`
- `relative_volume_1m >= 0.80`

Satellite body route:
- route = `tier5_satellite_body05_not13_market`
- `body_ratio <= 0.50`
- `entry_hour != 13`

Expansion route:
- regime = `other_non_target`
- subtype must be `mild_noise_v1_live`
- route = `tier287_other_non_target_mild_strong_bear_market`
- `trend_regime = strong_bear`
- `vol_regime = high_vol`
- `candidate_score >= 0.84`
- `move_1m_pct <= 0.25`
- `move_3m_pct <= 0.28`
- `move_5m_pct <= 0.55`
- `relative_volume_1m <= 1.50`
- `dist_1h_high_pct <= 0.08`
- `bounce_60m_pct <= 0.80`

## Entry mechanics (order type, where limit goes, where stop goes, why)
- Core mild and strict cooling use `LIMIT`
  - price = trigger close plus `0.15%`
  - reason: get retest instead of selling the first candle impulse
- Core selective cooling uses `MARKET`
  - reason: cooling is mature enough and no retest is required
- Satellite and expansion use `MARKET`
  - reason: these are already filtered into narrow live-safe buckets
- Stop always starts above trigger high with `0.10%` buffer

## Position sizing
- unified bot default: `POSITION_SIZE_MODE=full_balance`
- notional is recalculated on every entry from fresh `availableBalance`
- leverage uses existing `LEVERAGE` config, default `10`
- safety buffer uses `ORDER_MARGIN_SAFETY_BUFFER`, default `0.985`

## What a good setup looks like (step by step visual checklist)
- Daily chart is bearish or at least not strongly bullish
- ETH bounced up during the last hour into local highs
- `1m` trigger appears inside `13:00-16:00 UTC`
- push looks mature, not fresh breakout continuation
- trigger candle shows either cooling or noisy failure near resistance
- route-specific filters line up exactly

## What a bad setup looks like (skip conditions, exact reasons)
- outside `13:00-16:00 UTC`
- daily trend is bullish against the short
- expansion candidate has `candidate_score < 0.84`
- expansion candidate is too hot:
  - `move_3m_pct > 0.28`
  - `relative_volume_1m > 1.50`
  - `bounce_60m_pct > 0.80`
- mixed transition candidate fails calm filter
- core candidate does not classify into mild or valid cooling path

## How to verify each trade post-factum

| Field in trades.csv | What it means | Good value for T287 short |
|---|---|---|
| `strategy_name` | Which strategy fired | `short_t287` |
| `route` | Which branch was used | one of: `tier2p5_core_mild_limit`, `tier2p5_core_strict_cooling_limit`, `tier2p5_core_selective_cooling_market`, `tier5_satellite_rv08_market`, `tier5_satellite_body05_not13_market`, `tier287_other_non_target_mild_strong_bear_market` |
| `regime` | Predicted live regime | `near_entry_breakdown_noise`, `mixed_transition`, or `other_non_target` |
| `subtype` | Regime subtype | `mild_noise_v1_live` or `cooling_v3_live` |
| `candidate_score` | Signal strength score | `>= 0.84` for expansion; no floor for core |
| `move_1m_pct` | 1m move at trigger | `<= 0.25` for expansion |
| `move_3m_pct` | 3m move at trigger | `<= 0.28` for expansion |
| `relative_volume_1m` | Relative volume at trigger | `<= 1.50` for expansion |
| `dist_1h_high_pct` | Distance from 1H high | `<= 0.08` for expansion |
| `bounce_60m_pct` | 60m bounce magnitude | `<= 0.80` for expansion |
| `trend_regime` | Daily market regime | `strong_bear` or `weak_bear` |
| `vol_regime` | Daily vol regime | `high_vol` for expansion/satellite |

## Daily manual checklist
1. Open `logs/reports/YYYY-MM-DD_post_day.md`
2. Check `no_trade_reason` - if it says regime blocked, confirm daily chart
3. For each accepted trade check `route` and `candidate_score`
4. For each blocked trade check `blocked_reason` - if score was close, note it
5. Compare `entry_price` vs candle open in TradingView to verify no slippage issue

## Honest edge and limitations
- strongest branch is still the short core plus controlled expansion
- strategy is live-safe because it only uses already closed `1m` candles
- it is still ETH-specific and session-specific
- expansion is narrower and more fragile than the core
