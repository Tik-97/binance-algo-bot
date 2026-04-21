# How to Trade: Long Bull V3 (Bull Continuation)

## Strategy ID
`long_bull_v3`  —  LONG only  —  Priority 4 (lowest, fires only when t287/t14/f21 are all inactive)

---

## Core Idea
Buy mild intraday pullbacks during confirmed bull market conditions. When ETH is in a strong
uptrend (daily), and price briefly dips 0.20–1.20% on a 1m candle with clear buyer absorption
(lower wick + tight follow-through), enter long via limit order below the confirmation close.

This is the opposite of the flush-reversal family (t14/f21): instead of catching climax
dumps in bear markets, we catch mild dips in bull markets.

---

## When It Fires (all conditions must be true)

### 1. Daily Regime Gate
| Condition | Value |
|-----------|-------|
| `trend_regime` | `strong_bull` |
| `vol_regime` | `high_vol` |

**strong_bull** = `ret_60d >= +12%` AND `close > SMA60 by >= 3%`  
**high_vol** = 14-day daily return volatility above its own median

If regime is bear, transition, or low_vol → strategy is silently blocked (no position taken).

### 2. Intraday Context Gate (checked on trigger bar)
| Condition | Meaning |
|-----------|---------|
| `dist_4h_low_pct >= 1.50%` | Price is well above the 4h rolling low — trending up |
| `dump_60m_pct <= -0.20%` | Price has RISEN 0.20%+ in the last 60 minutes |
| `stretch_below_ema20_5m_pct <= 0.50%` | Price is near or above the 5m EMA20 |

These three confirm that the short-term intraday trend is bullish before looking for a dip.

### 3. Trigger Candle (1m bar)
| Filter | Value | Why |
|--------|-------|-----|
| `drop_5m_pct` | 0.20% – 1.20% | Mild pullback only — not a panic |
| `lower_wick_ratio >= 0.35` | Wick at least 35% of total range | Buyers absorbed the dip |
| `body_ratio <= 0.55` | Body ≤ 55% of range | Not a directional momentum candle |
| `range_pct <= 0.70%` | Total bar range ≤ 0.70% | Not a chaotic/volatile bar |
| `red_count_last_5 <= 3` | ≤ 3 red candles in last 5 | Mild pullback, not a sell-off |
| `relative_volume_1m >= 0.80` | Volume ≥ 80% of 20-bar average | Sufficient participation |

### 4. Confirmation Candle (next 1m bar after trigger)
| Filter | Value | Why |
|--------|-------|-----|
| `range_pct <= 0.16%` | **CRITICAL** — tight confirm | Quick buyer absorption; no chasing |
| `close > trigger_close` | Confirm closes above trigger | Momentum confirmed |
| `high > trigger_high` | Confirm breaks trigger high | Structure reclaimed |

> **The 0.16% confirm range is the single most important filter.** Stress test showed
> relaxing to 0.18% drops WR from 75% to 68%, and 0.20% collapses WR to 54%.
> A large confirm candle means price surged violently — those setups reverse.

---

## Entry Mechanics
- **Order type**: LIMIT (always — market entry was tested and degraded results: PF 0.733)
- **Entry price**: `confirm_close × (1 − 0.04%)` — slightly below confirm close
- **TTL**: 6 minutes (6 candles). If not filled in 6 bars → order cancelled
- **Cancellation**: if price drops back below trigger low or more than 0.12% below confirm close
  before fill → order cancelled (setup invalidated)

---

## Stop Loss & Exit
Inherited from the unified bot framework (same as t14/f21):

| Parameter | Value |
|-----------|-------|
| **Stop** | `min(trigger_low, confirm_low) × (1 − 0.10%)` |
| **Partial close** | 50% position closed at 2R profit |
| **Breakeven** | Stop moves to entry after partial close |
| **Trail** | Trailing stop activates after 2R, looks back 3 bars |
| **Max hold** | 72 hours (4320 bars) |

---

## Backtest Results (2yr ETH/USDT, $400 start, 1% risk/trade, 10x leverage)

| Metric | Value |
|--------|-------|
| Trade count | 16 |
| Win rate | **75.0%** |
| Profit factor | **2.716** |
| Max drawdown | **2.04%** |
| Expectancy/trade | $2.61 |
| Ending balance | $441.74 (+10.4%) |

**Monthly activity** (strategy is silent outside strong_bull periods):

| Month | Trades | WR | PF |
|-------|--------|----|----|
| 2024-11 | 3 | 100% | ∞ |
| 2024-12 | 4 | 75% | 2.83 |
| 2025-05 | 2 | 50% | 0.64 |
| 2025-06 | 3 | 67% | 2.69 |
| 2025-08 | 4 | 75% | 2.54 |

**Exit breakdown**: 12 managed_stop (all wins, trail hit after partial), 4 stop_loss (all losses).

---

## Stress Test Summary

Parameter sensitivity — effect of loosening each filter by ±15–25%:

| Variant | n | WR | PF | Verdict |
|---------|---|----|----|---------|
| **BASE** vol=0.80, wick=0.35, conf=0.16 | 16 | **75%** | 2.716 | ✅ |
| conf → 0.18 (+12%) | 19 | 68% | 1.914 | ⚠️ Edge weakens |
| conf → 0.20 (+25%) | 24 | 54% | 1.064 | ❌ Edge collapses |
| vol → 0.70 (−12%) | 20 | 70% | 2.184 | ✅ Acceptable |
| vol → 0.60 (−25%) | 24 | 58% | 1.293 | ⚠️ Borderline |
| wick → 0.30 (−14%) | 16 | 75% | 2.716 | ✅ Identical |

**Key conclusion**: The edge lives almost entirely in `confirm_range ≤ 0.16`. Protect this filter.

---

## Live Trading Notes

1. **Frequency**: ~8 trades/year. Strategy is inactive during bear/transition/low_vol regimes.
   Silence is correct behaviour — do not force trades.

2. **In-sample caveat**: Filters were derived post-hoc on 2yr of backtest data. Treat 75% WR
   as an upper bound. Realistic live WR expectation: **60–68%**.

3. **Priority 4**: This strategy only fires when the wallet is free AND none of short_t287,
   long_t14, or long_f21 triggered on the same candle.

4. **Bull regime identification lag**: The regime classifier uses 60-day lookback. ETH could
   enter a genuine bull run but the regime won't classify as strong_bull until price has been
   12%+ above SMA60 for ~60 days. Expect delayed activation at the start of bull cycles.

5. **Do not adjust confirm_range in live config** — it is hardcoded at 0.16% in the strategy.
   If you want to experiment, modify the backtest code and re-run first.

---

## Bot Configuration (.env)
```
ENABLED_STRATEGIES=short_t287,long_t14,long_f21,long_bull_v3
LONG_BULL_V3_SESSION_START_UTC=0
LONG_BULL_V3_SESSION_END_UTC=0
```
Session `0,0` means all hours (24/7). The regime filter does the temporal gating.
