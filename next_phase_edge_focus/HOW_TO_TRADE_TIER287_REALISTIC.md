# How To Trade Tier 2.87 Realistic

This is the current main Binance baseline.

If you want the shortest version first:

- market: `ETHUSDT`
- side: `SHORT` only
- session: `13:00-16:00 UTC`
- local session in Yerevan: `17:00-20:00`
- main idea: short bearish pullbacks near local highs, not random dumps

Current benchmark:

- `42` trades
- `ending_balance = 645.87`
- `win_rate = 88.10%`
- `profit_factor = 7.872`

## 1. What charts to open

Use four charts for manual review:

1. `1D`
2. `1H`
3. `5m`
4. `1m`

If Binance lets you save a layout, keep these four open together.

## 2. What indicators to put on charts

### Daily chart

Add:

- `MA 30`
- `MA 60`

This chart is only for regime.

### 1H chart

Add:

- no complicated indicators required
- draw a horizontal line at the latest completed `1H` high
- optionally draw the latest important `4H` high too

This chart is for location.

### 5m chart

Add:

- `EMA 20`
- volume

This chart is for pullback structure.

### 1m chart

Add:

- `EMA 7`
- `EMA 20`
- volume

This chart is for the actual trigger.

## 3. What time of day to look

Do not stare at ETH all day.

This strategy is only active in:

- `13:00-16:00 UTC`
- `17:00-20:00` Yerevan time

If price looks amazing outside this window, for this baseline you still skip it.

## 4. First question: is the day bearish enough

Before you think about any short, check the daily chart.

### Strong bear

Treat the day as `strong_bear` when:

- ETH is clearly below `MA 60`
- and price is roughly `12%` or more below where it was about `60` days ago
- and it is at least about `3%` under `MA 60`

If all three are true, this is the best regime.

### Weak bear

Treat the day as `weak_bear` when:

- it is not already `strong_bear`
- ETH is below `MA 30`
- and price is roughly `5%` or more below where it was about `30` days ago
- and it is at least about `1.5%` under `MA 30`

This is allowed for the core branch, but not as strong as `strong_bear`.

### If neither is true

Do nothing.

If the daily chart is flat, bullish, or unclear:

- no trade

## 5. Second question: are we in the right location

On the `1H` and `5m` charts, the setup should look like a bearish pullback.

You do **not** want:

- a market that is already collapsing straight down
- a random chop day with no structure
- price very far under the recent local highs

You **do** want:

- the day is bearish
- ETH has bounced upward during the last hour
- price is now trading reasonably close to the recent `1H` high
- price is also not far from the local `4H` high zone

Think:

- bearish background
- local bounce up
- now the bounce is getting vulnerable

## 6. Third question: is the bounce healthy enough to short

The candidate engine only likes pullbacks that are active, but not absurd.

On the chart this means:

- last `60m` should be up, not dead
- last `15m` should also be positive or at least constructive
- last `5m` should not be a total blow-off candle
- price should be above or around `EMA20` on `5m`, but not insanely stretched

If the pullback already rolled over too much:

- skip

If the pullback is vertical and euphoric:

- also skip

## 7. The three layers inside Tier 2.87

Tier 2.87 has three practical layers:

1. `Tier 2.5 core`
2. `mixed_transition satellites`
3. `other_non_target strong_bear expansion`

You do not need to call them by name while looking at the chart, but you should know the difference.

## 8. Layer A: Tier 2.5 core

This is the safest part.

It works in:

- `strong_bear`
- `weak_bear`

What it should look like on chart:

- bearish daily regime
- price bounced up into resistance
- `1m` and `5m` are not wild
- the candle cluster near entry looks like a tired bounce, not a fresh breakout

Inside the core there are two main visual behaviors.

### Core mild-noise look

This is the most normal short.

What you want to see:

- price near local highs
- small noisy grind up
- no huge trend candle right now
- no full collapse yet
- short setup still feels “ripe”

### Core cooling look

This is a more obvious loss-of-momentum case.

What you want to see:

- `1m` trigger candle has some softness
- or upper wick rejection
- or closes lower inside its own range
- but the previous `3m` move was still decent

Simple mental picture:

- the bounce is still mature enough
- but the current minute starts to lose energy

## 9. Layer B: mixed_transition satellites

This layer is only allowed in:

- `strong_bear`

This layer is calmer and narrower.

On chart, the setup should feel like:

- a controlled grind up
- not too much impulsive volume
- not too many strong green candles in a row
- price still fairly close to recent `1H` high

Exact calm conditions in code are roughly:

- `move_1m_pct <= 0.11`
- `move_3m_pct <= 0.18`
- `move_5m_pct <= 0.25`
- `relative_volume_1m <= 1.20`
- `bounce_60m_pct` in a moderate zone
- `bull_body_sum_5m_pct <= 0.32`
- recent green count only `2` or `3`

This is the “controlled pullback” branch.

### Satellite rv08 slice

If the setup is calm, and current `1m` relative volume is at least `0.80`, this branch is allowed.

Visual meaning:

- not dead
- not explosive
- active enough to matter

### Satellite body05_not13 slice

If relative volume is below `0.80`, it can still be allowed, but only when:

- trigger body is small enough
- and the hour is not `13 UTC`

Visual meaning:

- low-volume trap style
- but not the noisy early-session version

## 10. Layer C: other_non_target strong-bear expansion

This is the aggressive part that made Tier 2.87 better.

This layer is only allowed when:

- top live-safe day classifier says `other_non_target`
- subtype is still `mild_noise_v1_live`
- market trend is `strong_bear`
- market volatility is elevated

Then on the trigger bar we still require:

- `candidate_score >= 0.84`
- `move_1m_pct <= 0.25`
- `move_3m_pct <= 0.28`
- `move_5m_pct <= 0.55`
- `relative_volume_1m <= 1.50`
- `dist_1h_high_pct <= 0.08`
- `bounce_60m_pct <= 0.80`

Visual meaning:

- still a bearish pullback
- still near the local high
- but now even the aggressive branch must stay controlled
- if it starts looking too hot, too extended, or too far from the high, skip

## 11. Exact step-by-step workflow on Binance chart

Follow this in order every time.

### Step 1

Open `1D`.

Ask:

- is ETH in `strong_bear`?
- if not, is it at least in `weak_bear`?

If no:

- stop here

### Step 2

Open `1H`.

Mark:

- latest completed `1H` high
- latest important `4H` high zone if visible

Ask:

- is current price pushing up into those highs?
- or is price already far below and late?

If late:

- skip

### Step 3

Open `5m`.

Ask:

- is the last hour a real bounce?
- is price above or around `EMA20`?
- is this a pullback, not a dump?

If yes:

- continue

If the market is already falling hard:

- skip

### Step 4

Open `1m`.

Ask:

- is current minute inside session `13:00-16:00 UTC`?
- is price still near the recent `1H` high?
- is the candle behavior either calm or softly weakening?

If yes:

- continue

### Step 5

Decide which branch this looks closest to:

- core mild-noise
- core cooling
- calm mixed-transition
- strong-bear controlled expansion

If it looks like none of them:

- skip

### Step 6

Do **not** short a pure breakout candle.

What to avoid:

- giant green body
- vertical expansion candle
- volume spike that makes it look like continuation, not exhaustion

### Step 7

Only enter if the bounce feels mature enough **and** controlled enough.

That combination matters.

Wrong entries usually happen when:

- too early
- too hot
- too far from local highs
- or outside bearish regime

## 12. Example of a good visual setup

Good example shape:

- daily chart already bearish
- last hour pushed up into resistance
- `5m` sits above `EMA20`, but not vertical
- `1m` shows a calm or slightly weakening push
- price is still very near the recent `1H` high
- session time is correct

This is the kind of chart where the strategy wants to act.

## 13. Example of a bad visual setup

Bad example shape:

- daily chart not bearish
- price already fell away from the highs
- you are trying to short a breakdown, not a pullback
- current `1m` candle is euphoric and trend-like
- or the whole thing is happening outside session

That is a skip.

## 14. If you want the safest manual version

Manual trading is always weaker than exact code.

So if you trade by eye and want the safest approximation:

- trade only `strong_bear` first
- only inside session
- only when price is clearly near recent `1H` high
- prefer calm pullbacks and obvious cooling
- skip anything that feels too hot or too late

That will miss some valid `Tier 2.87` trades, but it reduces manual mistakes.

## 15. Important truth

The exact code still sees things better than the naked chart, especially in the expansion branch.

So the most correct way to trade Tier 2.87 is:

- use the script or bot for exact signals
- use the chart to confirm context and execution quality

The chart alone is enough to understand the shape.
The code is still the final truth.

## 16. One-line version

Open `1D`, `1H`, `5m`, `1m`; make sure ETH is bearish, price is bouncing into local highs during `13:00-16:00 UTC`, and short only controlled, mature pullbacks that look calm or cooling near the high.
