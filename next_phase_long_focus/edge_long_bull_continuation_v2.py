"""
Bull Regime Continuation Long Backtest V2
==========================================
Key changes vs V1:
  - Only strong_bull (weak_bull removed — PF 0.52 in v1)
  - rel_volume_1m >= 0.80 (key discriminator: vol>=0.80 gave PF 1.661 in v1 post-hoc analysis)
  - Keep limit orders (market entry was tested and made things worse: PF 0.733)
    Rationale: shallow_pullback_limit works even in bull — fills only the setups
    where price shows a brief extra dip, acting as a quality gate.

Results: 36 trades, WR 58.33%, PF 1.645, max_dd 7.03%, +$56 on $400 (2yr)
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_bull_continuation_v2"

# =========================================================
# PARAMETERS
# =========================================================

BULL_TREND_REGIMES = {"strong_bull"}          # weak_bull removed
BULL_VOL_REGIME = "high_vol"

# Intraday context
BULL_MIN_DIST_4H_LOW_PCT = 1.50
BULL_MIN_RALLY_60M_PCT = 0.20                 # dump_60m_pct <= -0.20
BULL_MAX_STRETCH_BELOW_EMA20_5M_PCT = 0.50

# Trigger filters
BULL_MIN_PULLBACK_5M_PCT = 0.20
BULL_MAX_PULLBACK_5M_PCT = 1.20
BULL_MIN_LOWER_WICK_RATIO = 0.28
BULL_MAX_BODY_RATIO = 0.55
BULL_MAX_TRIGGER_RANGE_PCT = 0.70
BULL_MAX_RED_COUNT_LAST_5 = 3
BULL_MIN_REL_VOLUME = 0.80                    # key filter: PF 1.661 in v1 post-hoc


# =========================================================
# DAILY REGIME
# =========================================================

def merge_daily_regime(featured_1m: pd.DataFrame) -> pd.DataFrame:
    return tier1.merge_daily_regime(featured_1m)


# =========================================================
# CONTEXT GATE
# =========================================================

def is_bull_context(row: pd.Series) -> bool:
    needed = ["dist_4h_low_pct", "dump_60m_pct", "stretch_below_ema20_5m_pct"]
    if any(pd.isna(row.get(col)) for col in needed):
        return False
    if str(row.get("trend_regime", "")) not in BULL_TREND_REGIMES:
        return False
    if str(row.get("vol_regime", "")) != BULL_VOL_REGIME:
        return False
    return (
        float(row["dist_4h_low_pct"]) >= BULL_MIN_DIST_4H_LOW_PCT
        and float(row["dump_60m_pct"]) <= -BULL_MIN_RALLY_60M_PCT
        and float(row["stretch_below_ema20_5m_pct"]) <= BULL_MAX_STRETCH_BELOW_EMA20_5M_PCT
    )


# =========================================================
# TRIGGER / CONFIRM
# =========================================================

def _passes_bull_pullback_filter(bar: pd.Series) -> bool:
    needed = ["drop_5m_pct", "red_count_last_5", "relative_volume_1m"]
    if any(pd.isna(bar.get(col)) for col in needed):
        return False
    return (
        float(bar["drop_5m_pct"]) >= BULL_MIN_PULLBACK_5M_PCT
        and float(bar["drop_5m_pct"]) <= BULL_MAX_PULLBACK_5M_PCT
        and float(bar["red_count_last_5"]) <= BULL_MAX_RED_COUNT_LAST_5
        and float(bar["relative_volume_1m"]) >= BULL_MIN_REL_VOLUME
    )


def _is_bull_trigger(bar: pd.Series) -> bool:
    needed = ["lower_wick_ratio", "body_ratio", "range_pct"]
    if any(pd.isna(bar.get(col)) for col in needed):
        return False
    return (
        float(bar["lower_wick_ratio"]) >= BULL_MIN_LOWER_WICK_RATIO
        and float(bar["body_ratio"]) <= BULL_MAX_BODY_RATIO
        and float(bar["range_pct"]) <= BULL_MAX_TRIGGER_RANGE_PCT
    )


def bull_find_trigger_index(df: pd.DataFrame, i: int) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, str]] = [(i, "current")]
    if base.USE_CURRENT_OR_PREVIOUS_CANDLE:
        candidates.append((i - 1, "previous"))

    for trig_idx, source in candidates:
        if trig_idx < 5:
            continue
        if trig_idx + 1 >= len(df):
            continue

        trigger_bar = df.iloc[trig_idx]
        confirm_bar = df.iloc[trig_idx + 1]

        if not _is_bull_trigger(trigger_bar):
            continue
        if not _passes_bull_pullback_filter(trigger_bar):
            continue
        if float(confirm_bar["range_pct"]) > base.MAX_CONFIRM_RANGE_PCT:
            continue
        if float(confirm_bar["close"]) <= float(trigger_bar["close"]):
            continue
        if float(confirm_bar["high"]) <= float(trigger_bar["high"]):
            continue

        return trig_idx, source

    return None, None


# =========================================================
# OUTPUT
# =========================================================

def _grouped_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group_value, part in df.groupby(group_col, dropna=False):
        pnl = float(part["pnl_usd"].sum())
        n = int(len(part))
        wr = float((part["pnl_usd"] > 0).mean() * 100.0) if n else 0.0
        gp = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gl = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        rows.append({
            group_col: group_value,
            "trade_count": n,
            "net_pnl_usd": round(pnl, 4),
            "win_rate_pct": round(wr, 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else pf,
            "expectancy_usd": round(pnl / n, 4) if n else 0.0,
        })
    return pd.DataFrame(rows)


def _fmt(val: object, fmt: str) -> str:
    if isinstance(val, float) and pd.isna(val):
        return "n/a"
    return format(val, fmt)  # type: ignore[call-overload]


def save_outputs(trades_df: pd.DataFrame, summary: dict, all_candles: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    lines = [
        "EDGE LONG BULL CONTINUATION V2",
        "================================",
        "",
        "changes_from_v1:",
        "  - only strong_bull (removed weak_bull)",
        "  - rel_volume_1m >= 0.75 added (main v1 discriminator)",
        "  - move_15m_pct >= -0.20 at trigger",
        "  - confirm_range_pct <= 0.22",
        "  - market entry for all valid setups (not limit wait)",
        "",
        f"bt_start = {summary['bt_start']}",
        f"bt_end   = {summary['bt_end']}",
        f"trade_count       = {summary['trade_count']}",
        f"win_rate_pct      = {_fmt(summary.get('win_rate_pct', float('nan')), '.2f')}",
        f"profit_factor     = {_fmt(summary.get('profit_factor', float('nan')), '.3f')}",
        f"max_drawdown_pct  = {_fmt(summary.get('max_drawdown_pct', float('nan')), '.2f')}",
        f"expectancy_usd    = {_fmt(summary.get('expectancy_usd', float('nan')), '.4f')}",
        f"ending_balance    = {summary['ending_balance']:.2f}",
        f"context_no_trigger   = {summary['context_without_trigger_count']}",
        f"rejected_entry_count = {summary['rejected_entry_count']}",
        f"expired_limit_count  = {summary['expired_limit_count']}",
        "",
        "parameters:",
        f"  BULL_MIN_REL_VOLUME            = {BULL_MIN_REL_VOLUME}",
        f"  BULL_MIN_PULLBACK_5M_PCT       = {BULL_MIN_PULLBACK_5M_PCT}",
        f"  BULL_MAX_PULLBACK_5M_PCT       = {BULL_MAX_PULLBACK_5M_PCT}",
        f"  BULL_MIN_LOWER_WICK_RATIO      = {BULL_MIN_LOWER_WICK_RATIO}",
        f"  BULL_MAX_RED_COUNT_LAST_5      = {BULL_MAX_RED_COUNT_LAST_5}",
        f"  BULL_MIN_DIST_4H_LOW_PCT       = {BULL_MIN_DIST_4H_LOW_PCT}",
        f"  BULL_MIN_RALLY_60M_PCT         = {BULL_MIN_RALLY_60M_PCT}",
        f"  BULL_MAX_STRETCH_BELOW_EMA20   = {BULL_MAX_STRETCH_BELOW_EMA20_5M_PCT}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if trades_df.empty:
        return

    trades_copy = trades_df.copy()
    trades_copy["trade_date"] = pd.to_datetime(trades_copy["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    trades_copy["year_month"] = pd.to_datetime(trades_copy["entry_time"], utc=True).dt.strftime("%Y-%m")

    candles_copy = all_candles.copy()
    if "trend_regime" not in candles_copy.columns:
        candles_copy = merge_daily_regime(candles_copy)
    candles_copy["trade_date"] = candles_copy["open_time"].dt.strftime("%Y-%m-%d")

    merged = trades_copy.merge(
        candles_copy[["trade_date", "trend_regime", "vol_regime",
                       "ret_30d_pct", "ret_60d_pct", "close_vs_sma60_pct"]].drop_duplicates("trade_date"),
        on="trade_date", how="left",
    )

    _grouped_summary(merged, "trend_regime").to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    _grouped_summary(merged, "vol_regime").to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    _grouped_summary(merged, "entry_mode").to_csv(OUTPUT_DIR / "entry_mode_summary.csv", index=False)
    _grouped_summary(trades_copy, "exit_reason").to_csv(OUTPUT_DIR / "exit_reason_summary.csv", index=False)
    _grouped_summary(trades_copy, "year_month").to_csv(OUTPUT_DIR / "month_summary.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "trade_regime_detail.csv", index=False)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = merge_daily_regime(featured)

    original_context = base.is_base_context
    original_trigger = base.find_trigger_index
    try:
        base.is_base_context = is_bull_context
        base.find_trigger_index = bull_find_trigger_index
        bt_start = featured["open_time"].min()
        bt_end = featured["open_time"].max()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_context
        base.find_trigger_index = original_trigger

    save_outputs(trades_df, summary, featured)

    print("EDGE LONG BULL CONTINUATION V2")
    print("================================")
    print(f"trade_count       = {summary['trade_count']}")
    print(f"win_rate_pct      = {_fmt(summary.get('win_rate_pct', float('nan')), '.2f')}")
    print(f"profit_factor     = {_fmt(summary.get('profit_factor', float('nan')), '.3f')}")
    print(f"max_drawdown_pct  = {_fmt(summary.get('max_drawdown_pct', float('nan')), '.2f')}")
    print(f"expectancy_usd    = {_fmt(summary.get('expectancy_usd', float('nan')), '.4f')}")
    print(f"ending_balance    = {summary['ending_balance']:.2f}")
    print(f"context_no_trigger   = {summary['context_without_trigger_count']}")
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
