"""
Bull Regime Continuation Long Backtest V4 — Relaxed Regime
===========================================================
Tests whether the bull_v3 edge holds when regime gate is relaxed:

  Variant A (baseline):  strong_bull + high_vol          (original v3)
  Variant B:             strong_bull | weak_bull + high_vol
  Variant C:             strong_bull | weak_bull + any vol

weak_bull = ret_30d >= 5% AND close > SMA30 by >= 1.5%
           (early bull, before 60-day criterion kicks in)

All other filters unchanged from v3 (confirm_range <= 0.16 is kept).
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1
import next_phase_long_focus.edge_long_bull_continuation_v3 as v3


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "edge_long_bull_continuation_v4_relaxed_regime"


# =========================================================
# VARIANT DEFINITIONS
# =========================================================

VARIANTS: list[dict] = [
    {
        "name": "A_strong_bull_highvol",
        "label": "Variant A — strong_bull + high_vol (v3 baseline)",
        "trend_regimes": {"strong_bull"},
        "vol_regimes": {"high_vol"},
    },
    {
        "name": "B_bull_highvol",
        "label": "Variant B — strong_bull|weak_bull + high_vol",
        "trend_regimes": {"strong_bull", "weak_bull"},
        "vol_regimes": {"high_vol"},
    },
    {
        "name": "C_bull_any_vol",
        "label": "Variant C — strong_bull|weak_bull + any vol",
        "trend_regimes": {"strong_bull", "weak_bull"},
        "vol_regimes": {"high_vol", "low_vol"},
    },
]


# =========================================================
# CONTEXT GATE FACTORY
# =========================================================

def make_bull_context(trend_regimes: set[str], vol_regimes: set[str]):
    def is_bull_context(row: pd.Series) -> bool:
        needed = ["dist_4h_low_pct", "dump_60m_pct", "stretch_below_ema20_5m_pct"]
        if any(pd.isna(row.get(col)) for col in needed):
            return False
        if str(row.get("trend_regime", "")) not in trend_regimes:
            return False
        if str(row.get("vol_regime", "")) not in vol_regimes:
            return False
        return (
            float(row["dist_4h_low_pct"]) >= v3.BULL_MIN_DIST_4H_LOW_PCT
            and float(row["dump_60m_pct"]) <= -v3.BULL_MIN_RALLY_60M_PCT
            and float(row["stretch_below_ema20_5m_pct"]) <= v3.BULL_MAX_STRETCH_BELOW_EMA20_5M_PCT
        )
    return is_bull_context


# =========================================================
# HELPERS
# =========================================================

def _fmt(val: object, fmt: str) -> str:
    if isinstance(val, float) and pd.isna(val):
        return "n/a"
    return format(val, fmt)  # type: ignore[call-overload]


def _stats(trades_df: pd.DataFrame) -> dict:
    n = len(trades_df)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "pf": float("nan"), "dd": float("nan"), "exp": float("nan")}
    wr = float((trades_df["pnl_usd"] > 0).mean() * 100)
    gp = float(trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = float(-trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": wr, "pf": pf, "exp": float(trades_df["pnl_usd"].mean())}


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


# =========================================================
# RUN ONE VARIANT
# =========================================================

def run_variant(featured: pd.DataFrame, variant: dict) -> tuple[pd.DataFrame, dict]:
    context_fn = make_bull_context(variant["trend_regimes"], variant["vol_regimes"])
    original_context = base.is_base_context
    original_trigger = base.find_trigger_index
    try:
        base.is_base_context = context_fn
        base.find_trigger_index = v3.bull_find_trigger_index
        bt_start = featured["open_time"].min()
        bt_end = featured["open_time"].max()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_context
        base.find_trigger_index = original_trigger
    return trades_df, summary


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading candles...")
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = v3.merge_daily_regime(featured)

    # Show regime distribution
    if "trend_regime" in featured.columns:
        regime_col = featured["trend_regime"].dropna()
        daily_regime = featured.drop_duplicates(featured["open_time"].dt.date.name if False else "open_time").copy()
        # Count days per regime
        daily = featured.copy()
        daily["trade_date"] = daily["open_time"].dt.date
        day_regime = daily.drop_duplicates("trade_date")
        print("\nRegime day count in dataset:")
        print(day_regime["trend_regime"].value_counts().to_string())
        print()

    results: list[dict] = []
    all_trades: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        print(f"Running {variant['label']}...")
        trades_df, summary = run_variant(featured, variant)
        st = _stats(trades_df)
        results.append({
            "variant": variant["name"],
            "label": variant["label"],
            "n": st["n"],
            "wr_pct": round(st["wr"], 1) if st["n"] else float("nan"),
            "pf": round(st["pf"], 3) if st["pf"] != float("inf") else float("inf"),
            "expectancy": round(st["exp"], 4) if st["n"] else float("nan"),
            "ending_balance": round(summary["ending_balance"], 2),
            "max_dd_pct": round(summary.get("max_drawdown_pct", float("nan")), 2),
        })
        all_trades[variant["name"]] = trades_df
        trades_df.to_csv(OUTPUT_DIR / f"trades_{variant['name']}.csv", index=False)

    # Comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "variant_comparison.csv", index=False)

    # Per-regime breakdown for each variant
    for variant in VARIANTS:
        trades_df = all_trades[variant["name"]]
        if trades_df.empty:
            continue
        trades_copy = trades_df.copy()
        trades_copy["trade_date"] = pd.to_datetime(trades_copy["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
        trades_copy["year_month"] = pd.to_datetime(trades_copy["entry_time"], utc=True).dt.strftime("%Y-%m")
        candles_copy = featured.copy()
        candles_copy["trade_date"] = candles_copy["open_time"].dt.strftime("%Y-%m-%d")
        merged = trades_copy.merge(
            candles_copy[["trade_date", "trend_regime", "vol_regime"]].drop_duplicates("trade_date"),
            on="trade_date", how="left",
        )
        _grouped_summary(merged, "trend_regime").to_csv(
            OUTPUT_DIR / f"regime_breakdown_{variant['name']}.csv", index=False
        )
        _grouped_summary(trades_copy, "year_month").to_csv(
            OUTPUT_DIR / f"month_summary_{variant['name']}.csv", index=False
        )

    # Print comparison
    print("\n" + "=" * 70)
    print("BULL CONTINUATION V4 — REGIME RELAXATION COMPARISON")
    print("=" * 70)
    print(f"{'Variant':<35} {'n':>4} {'WR':>7} {'PF':>7} {'MaxDD':>7} {'End Bal':>10}")
    print("-" * 70)
    for r in results:
        wr_str = f"{r['wr_pct']:.1f}%" if not pd.isna(r['wr_pct']) else "n/a"
        pf_str = f"{r['pf']:.3f}" if r['pf'] != float("inf") else "inf"
        dd_str = f"{r['max_dd_pct']:.2f}%" if not pd.isna(r['max_dd_pct']) else "n/a"
        print(f"{r['label']:<35} {r['n']:>4} {wr_str:>7} {pf_str:>7} {dd_str:>7} {r['ending_balance']:>10.2f}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")

    # Regime breakdown for variant B (the interesting one)
    b_trades = all_trades["B_bull_highvol"]
    if not b_trades.empty:
        b_copy = b_trades.copy()
        b_copy["trade_date"] = pd.to_datetime(b_copy["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
        b_merged = b_copy.merge(
            featured.copy().assign(trade_date=lambda d: d["open_time"].dt.strftime("%Y-%m-%d"))
            [["trade_date", "trend_regime", "vol_regime"]].drop_duplicates("trade_date"),
            on="trade_date", how="left",
        )
        print("\n--- Variant B regime breakdown ---")
        print(_grouped_summary(b_merged, "trend_regime").to_string(index=False))

    c_trades = all_trades["C_bull_any_vol"]
    if not c_trades.empty:
        c_copy = c_trades.copy()
        c_copy["trade_date"] = pd.to_datetime(c_copy["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
        c_merged = c_copy.merge(
            featured.copy().assign(trade_date=lambda d: d["open_time"].dt.strftime("%Y-%m-%d"))
            [["trade_date", "trend_regime", "vol_regime"]].drop_duplicates("trade_date"),
            on="trade_date", how="left",
        )
        print("\n--- Variant C regime breakdown ---")
        print(_grouped_summary(c_merged, "trend_regime").to_string(index=False))
        print(_grouped_summary(c_merged, "vol_regime").to_string(index=False))


if __name__ == "__main__":
    main()
