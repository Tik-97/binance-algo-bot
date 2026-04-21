"""
Bull Market Spike Rejection Short — Exploration V1
====================================================
Idea: in strong_bull + high_vol regime, price sometimes makes a fast spike UP
on a 1m candle (upper wick rejection) then pulls back. Short the correction.

This is the structural mirror of bull_v3:
  bull_v3:     lower wick after mild DROP  → LONG
  spike_short: upper wick after rapid RISE → SHORT

Step 1: Run with very loose filters, save all candidate trades with features.
Step 2: Post-hoc grid search to find filters that give WR >= 65%, n >= 15.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "edge_short_spike_rejection_v1"

# ── Regime ─────────────────────────────────────────────────────────────────
ALLOWED_TREND_REGIMES = {"strong_bull"}
ALLOWED_VOL_REGIMES   = {"high_vol"}

# ── Context (loose — just confirm overall intraday uptrend) ─────────────────
CTX_MIN_DIST_4H_LOW_PCT = 1.00      # price is above 4h low (in uptrend)
CTX_MIN_RALLY_60M_PCT   = 0.10      # price rose at least 0.10% in last 60m

# ── Trigger: rapid spike UP + upper wick rejection ──────────────────────────
MIN_SPIKE_5M_PCT   = 0.20           # price went up >= 0.20% in last 5m
MAX_SPIKE_5M_PCT   = 2.00           # not a monster runaway candle
MIN_UPPER_WICK_RATIO = 0.25         # upper wick >= 25% of range (loose)
MAX_BODY_RATIO     = 0.65           # not pure directional candle
MAX_RANGE_PCT      = 0.80           # not chaotic bar
MAX_GREEN_COUNT    = 4              # at most 4 green in last 5 (came from a run)
MIN_REL_VOLUME     = 0.60           # some participation

# ── Confirmation: sellers stepped in ───────────────────────────────────────
MAX_CONFIRM_RANGE_PCT = 0.30        # loose for now — will tighten in grid search

# ── Entry / stop ────────────────────────────────────────────────────────────
ENTRY_PCT_ABOVE_CONFIRM = 0.04      # limit SHORT slightly above confirm close
ENTRY_WAIT_BARS = 6                 # 6 minutes TTL

STOP_BUFFER_ABOVE_HIGH_PCT = 0.10   # stop above max(trigger_high, confirm_high)

# ── Exit ─────────────────────────────────────────────────────────────────────
PARTIAL_CLOSE_R       = 2.0
PARTIAL_CLOSE_RATIO   = 0.50
MAX_HOLD_BARS         = 4320        # 72 hours


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_short_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features not in base build_features."""
    df = df.copy()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    rng = df["high"] - df["low"]
    df["upper_wick_ratio"] = (upper_wick / rng.replace(0, np.nan)).fillna(0.0)
    # spike_5m_pct: positive = price ROSE (inverse of drop_5m_pct)
    df["spike_5m_pct"] = -df["drop_5m_pct"]
    df["green_count_last_5"] = 5 - df["red_count_last_5"]
    # rally_60m_pct: positive = price ROSE in 60m
    df["rally_60m_pct"] = -df["dump_60m_pct"]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT / TRIGGER / CONFIRM CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _is_spike_context(row: pd.Series) -> bool:
    if str(row.get("trend_regime", "")) not in ALLOWED_TREND_REGIMES:
        return False
    if str(row.get("vol_regime", "")) not in ALLOWED_VOL_REGIMES:
        return False
    dist = row.get("dist_4h_low_pct")
    rally = row.get("rally_60m_pct")
    if pd.isna(dist) or pd.isna(rally):
        return False
    return (
        float(dist) >= CTX_MIN_DIST_4H_LOW_PCT
        and float(rally) >= CTX_MIN_RALLY_60M_PCT
    )


def _is_spike_trigger(bar: pd.Series) -> bool:
    needed = ["spike_5m_pct", "upper_wick_ratio", "body_ratio", "range_pct",
              "green_count_last_5", "relative_volume_1m"]
    if any(pd.isna(bar.get(c)) for c in needed):
        return False
    return (
        float(bar["spike_5m_pct"]) >= MIN_SPIKE_5M_PCT
        and float(bar["spike_5m_pct"]) <= MAX_SPIKE_5M_PCT
        and float(bar["upper_wick_ratio"]) >= MIN_UPPER_WICK_RATIO
        and float(bar["body_ratio"]) <= MAX_BODY_RATIO
        and float(bar["range_pct"]) <= MAX_RANGE_PCT
        and float(bar["green_count_last_5"]) <= MAX_GREEN_COUNT
        and float(bar["relative_volume_1m"]) >= MIN_REL_VOLUME
    )


def _confirm_rejects(trigger_bar: pd.Series, confirm_bar: pd.Series) -> bool:
    if float(confirm_bar["range_pct"]) > MAX_CONFIRM_RANGE_PCT:
        return False
    if float(confirm_bar["close"]) >= float(trigger_bar["close"]):
        return False
    if float(confirm_bar["low"]) >= float(trigger_bar["low"]):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# EXIT SIMULATION (SHORT)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_short_exit(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    risk_usd: float,
) -> dict:
    risk_per_unit = stop_price - entry_price          # positive: stop is above entry
    if risk_per_unit <= 0:
        return {"exit_reason": "invalid_stop", "exit_price": entry_price, "pnl_usd": 0.0, "hold_bars": 0}

    partial_target = entry_price - PARTIAL_CLOSE_R * risk_per_unit
    partial_done = False
    remaining = 1.0
    be_stop = stop_price
    trail_active = False
    trail_stop = stop_price

    for j in range(entry_idx, min(entry_idx + MAX_HOLD_BARS, len(df))):
        bar = df.iloc[j]
        lo, hi = float(bar["low"]), float(bar["high"])
        active_stop = trail_stop if trail_active else be_stop

        # Stop hit
        if hi >= active_stop:
            exit_price = active_stop
            pnl = (entry_price - exit_price) / entry_price * risk_usd / (risk_per_unit / entry_price)
            return {"exit_reason": "stop_loss" if not partial_done else "managed_stop",
                    "exit_price": exit_price, "pnl_usd": round(pnl * remaining, 6),
                    "hold_bars": j - entry_idx}

        # Partial close at 2R
        if not partial_done and lo <= partial_target:
            partial_done = True
            remaining = 1.0 - PARTIAL_CLOSE_RATIO
            be_stop = entry_price
            trail_active = True
            trail_stop = be_stop

        # Trail: look back 3 bars low (for short: trail tracks highs)
        if trail_active:
            lookback = df.iloc[max(0, j - 2): j + 1]["high"]
            trail_stop = min(trail_stop, float(lookback.max()) + stop_price * 0.001)

    # Max hold
    exit_price = float(df.iloc[min(entry_idx + MAX_HOLD_BARS - 1, len(df) - 1)]["close"])
    pnl = (entry_price - exit_price) / entry_price * risk_usd / (risk_per_unit / entry_price)
    return {"exit_reason": "max_hold", "exit_price": exit_price,
            "pnl_usd": round(pnl * remaining, 6), "hold_bars": MAX_HOLD_BARS}


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

STARTING_BALANCE = 400.0
RISK_PCT = 1.0
FEE_PCT = 0.05

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    balance = STARTING_BALANCE
    trades: list[dict] = []
    last_exit_idx = -1

    for i in range(5, len(df) - 2):
        if i <= last_exit_idx:
            continue

        row = df.iloc[i]
        if not _is_spike_context(row):
            continue

        trigger_bar = row
        confirm_bar = df.iloc[i + 1]

        if not _is_spike_trigger(trigger_bar):
            continue
        if not _confirm_rejects(trigger_bar, confirm_bar):
            continue

        # Entry: limit SHORT slightly above confirm close
        limit_price = float(confirm_bar["close"]) * (1.0 + ENTRY_PCT_ABOVE_CONFIRM / 100.0)
        stop_price = max(float(trigger_bar["high"]), float(confirm_bar["high"])) * (
            1.0 + STOP_BUFFER_ABOVE_HIGH_PCT / 100.0
        )
        if limit_price >= stop_price:
            continue

        # Try to fill limit within TTL
        entry_idx = None
        entry_price = None
        fill_start = i + 2
        for j in range(fill_start, min(fill_start + ENTRY_WAIT_BARS, len(df))):
            bar = df.iloc[j]
            if float(bar["high"]) >= limit_price:
                entry_idx = j
                entry_price = limit_price * (1.0 + FEE_PCT / 100.0)
                break
            # Cancel if drops too far from confirm
            if float(bar["low"]) < float(confirm_bar["close"]) * (1.0 - 0.12 / 100.0):
                break

        if entry_idx is None or entry_price is None:
            continue

        risk_usd = balance * RISK_PCT / 100.0
        result = simulate_short_exit(df, entry_idx + 1, entry_price, stop_price, risk_usd)

        exit_price = result["exit_price"]
        fee = FEE_PCT / 100.0 * exit_price
        pnl = result["pnl_usd"] - fee * risk_usd / (stop_price - entry_price) * 2

        balance += pnl
        last_exit_idx = entry_idx + result["hold_bars"]

        trades.append({
            "entry_time":         trigger_bar["open_time"],
            "exit_time":          df.iloc[min(last_exit_idx, len(df)-1)]["open_time"],
            "entry_price":        round(entry_price, 4),
            "exit_price":         round(exit_price, 4),
            "stop_price_initial": round(stop_price, 4),
            "side":               "Short",
            "exit_reason":        result["exit_reason"],
            "hold_bars":          result["hold_bars"],
            "pnl_usd":            round(pnl, 6),
            "pnl_pct_balance":    round(pnl / (balance - pnl) * 100, 4),
            "balance_after":      round(balance, 4),
            # Features saved for post-hoc analysis
            "spike_5m_pct":         round(float(trigger_bar["spike_5m_pct"]), 4),
            "upper_wick_ratio":     round(float(trigger_bar["upper_wick_ratio"]), 4),
            "body_ratio":           round(float(trigger_bar["body_ratio"]), 4),
            "range_pct":            round(float(trigger_bar["range_pct"]), 4),
            "green_count_last_5":   int(trigger_bar["green_count_last_5"]),
            "relative_volume_1m":   round(float(trigger_bar["relative_volume_1m"]), 4),
            "confirm_range_pct":    round(float(confirm_bar["range_pct"]), 4),
            "rally_60m_pct":        round(float(trigger_bar["rally_60m_pct"]), 4),
            "dist_4h_low_pct":      round(float(trigger_bar["dist_4h_low_pct"]), 4),
            "trend_regime":         str(trigger_bar.get("trend_regime", "")),
            "vol_regime":           str(trigger_bar.get("vol_regime", "")),
        })

    return pd.DataFrame(trades)


# ══════════════════════════════════════════════════════════════════════════════
# POST-HOC GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _stats(df: pd.DataFrame) -> dict | None:
    n = len(df)
    if n < 5:
        return None
    wr = float((df["pnl_usd"] > 0).mean() * 100)
    gp = float(df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = float(-df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 3) if pf != float("inf") else 999.0}


def grid_search(trades: pd.DataFrame) -> pd.DataFrame:
    spike_thresholds  = [0.20, 0.30, 0.40, 0.50]
    wick_thresholds   = [0.25, 0.30, 0.35, 0.40]
    confirm_thresholds = [0.14, 0.16, 0.18, 0.20, 0.25]
    vol_thresholds    = [0.60, 0.70, 0.80]

    rows = []
    for spike, wick, conf, vol in product(
        spike_thresholds, wick_thresholds, confirm_thresholds, vol_thresholds
    ):
        subset = trades[
            (trades["spike_5m_pct"]     >= spike) &
            (trades["upper_wick_ratio"] >= wick)  &
            (trades["confirm_range_pct"] <= conf) &
            (trades["relative_volume_1m"] >= vol)
        ]
        st = _stats(subset)
        if st is None:
            continue
        rows.append({
            "spike_min": spike, "wick_min": wick,
            "conf_max": conf, "vol_min": vol,
            **st,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["wr", "pf"], ascending=False).head(30)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and building features...")
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    featured = add_short_features(featured)

    # Regime distribution
    daily = featured.copy()
    daily["trade_date"] = daily["open_time"].dt.date
    day_regime = daily.drop_duplicates("trade_date")
    print("Regime days:", day_regime["trend_regime"].value_counts().to_dict())

    strong_bull_days = int((day_regime["trend_regime"] == "strong_bull").sum())
    print(f"strong_bull days available: {strong_bull_days}")
    print()

    print("Running broad backtest (loose filters)...")
    trades = run_backtest(featured)

    print(f"Raw trades found: {len(trades)}")
    if trades.empty:
        print("No trades found — check regime data or filters.")
        return

    wr = float((trades["pnl_usd"] > 0).mean() * 100)
    gp = trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum()
    gl = -trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum()
    pf = gp / gl if gl > 0 else float("inf")
    print(f"Broad stats: n={len(trades)}, WR={wr:.1f}%, PF={pf:.3f}")
    print()

    trades.to_csv(OUTPUT_DIR / "trades_broad.csv", index=False)

    # Feature distributions
    print("Feature ranges in trades:")
    for col in ["spike_5m_pct", "upper_wick_ratio", "confirm_range_pct",
                "relative_volume_1m", "body_ratio", "range_pct"]:
        if col in trades.columns:
            print(f"  {col}: min={trades[col].min():.3f} "
                  f"median={trades[col].median():.3f} "
                  f"max={trades[col].max():.3f}")
    print()

    # Monthly breakdown
    trades_copy = trades.copy()
    trades_copy["year_month"] = pd.to_datetime(trades_copy["entry_time"], utc=True).dt.strftime("%Y-%m")
    monthly = trades_copy.groupby("year_month").agg(
        n=("pnl_usd", "size"),
        wr=("pnl_usd", lambda x: f"{(x>0).mean()*100:.0f}%"),
        net=("pnl_usd", "sum"),
    )
    print("Monthly breakdown:")
    print(monthly.to_string())
    print()

    # Win/loss by exit reason
    print("Exit reasons:")
    print(trades.groupby("exit_reason").agg(
        n=("pnl_usd","size"),
        net=("pnl_usd","sum"),
        wr=("pnl_usd", lambda x: f"{(x>0).mean()*100:.0f}%"),
    ).to_string())
    print()

    print("Running post-hoc grid search...")
    grid = grid_search(trades)
    grid.to_csv(OUTPUT_DIR / "grid_search.csv", index=False)

    if grid.empty:
        print("Grid search: no combos with n>=5 found.")
    else:
        print("Top grid search results (WR >= 65%+):")
        top = grid[grid["wr"] >= 60.0].head(20)
        if top.empty:
            print("No combos with WR >= 60% found.")
            print("Best found:")
            print(grid.head(10).to_string(index=False))
        else:
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
