"""
Universal Long Post-hoc Analysis V1
====================================
Runs base wickMick flush-reversal WITHOUT regime filter (68 trades),
saves all trades with features, then exhaustive post-hoc grid search
to find filter combos with the best WR/PF/trade-count balance.

Base context (NO regime):
  - dist_4h_low_pct <= 0.35  (price near 4h rolling low)
  - dump_60m_pct >= 0.45     (price dumped in last 60m)
  - stretch_below_ema20_5m_pct >= 0.15  (extended below EMA)

Post-hoc filters tested:
  - rel_volume_1m thresholds
  - confirm_range_pct thresholds
  - trigger_lower_wick_ratio thresholds
  - session hour filters
  - combinations of the above
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_no_regime_posthoc_v1"


def _stats(subset: pd.DataFrame) -> dict | None:
    n = len(subset)
    if n == 0:
        return None
    wr = float((subset["pnl_usd"] > 0).mean() * 100)
    gp = float(subset.loc[subset["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = float(-subset.loc[subset["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 3) if pf != float("inf") else 999.0}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run base backtest (no regime monkey-patch) ---
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)

    bt_start = featured["open_time"].min()
    bt_end = featured["open_time"].max()
    trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)

    n_base = summary["trade_count"]
    wr_base = summary.get("win_rate_pct", 0) or 0
    pf_base = summary.get("profit_factor", 0) or 0
    print(f"No-regime baseline: n={n_base}  WR={wr_base:.1f}%  PF={pf_base:.3f}")

    if trades_df.empty:
        print("No trades — aborting.")
        return

    trades_df.to_csv(OUTPUT_DIR / "trades_no_regime.csv", index=False)

    # --- Prepare for post-hoc ---
    df = trades_df.copy()
    df["entry_hour"] = pd.to_datetime(df["entry_time"], utc=True).dt.hour

    vol_thresholds  = [0.0, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20]
    conf_maxes      = [9999.0, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.30]
    wick_mins       = [0.0, 0.35, 0.40, 0.45, 0.50]
    sessions = {
        "all":        None,
        "asia_0-7":   (0, 7),
        "asia_0-8":   (0, 8),
        "london_7-11":(7, 11),
        "NY_13-21":   (13, 21),
        "03-05":      (3, 5),
        "02-06":      (2, 6),
    }

    results: list[dict] = []

    for (sess_name, sess_rng), vol_min, conf_max, wick_min in product(
        sessions.items(), vol_thresholds, conf_maxes, wick_mins
    ):
        mask = pd.Series(True, index=df.index)

        if sess_rng:
            mask &= (df["entry_hour"] >= sess_rng[0]) & (df["entry_hour"] < sess_rng[1])
        if vol_min > 0:
            mask &= df["rel_volume_1m"] >= vol_min
        if conf_max < 9999:
            mask &= df["confirm_range_pct"] <= conf_max
        if wick_min > 0:
            mask &= df["trigger_lower_wick_ratio"] >= wick_min

        s = _stats(df[mask])
        if s and s["n"] >= 8:
            results.append({
                "session":   sess_name,
                "vol_min":   vol_min,
                "conf_max":  conf_max if conf_max < 9999 else "any",
                "wick_min":  wick_min,
                "n":         s["n"],
                "wr":        s["wr"],
                "pf":        s["pf"],
            })

    if not results:
        print("No combo with n>=8 found.")
        return

    res_df = pd.DataFrame(results).sort_values(["wr", "pf"], ascending=False).reset_index(drop=True)
    res_df.to_csv(OUTPUT_DIR / "posthoc_grid.csv", index=False)

    # --- Print top results ---
    print(f"\n{'session':<13} {'vol':>5} {'conf':>6} {'wick':>5}  {'n':>5}  {'WR%':>6}  {'PF':>6}")
    print("-" * 60)

    print("=== TOP 30 by WR (n>=8) ===")
    for _, r in res_df.head(30).iterrows():
        print(
            f"{r['session']:<13} {str(r['vol_min']):>5} {str(r['conf_max']):>6} {str(r['wick_min']):>5}"
            f"  {r['n']:>5}  {r['wr']:>6.1f}  {r['pf']:>6.3f}"
        )

    print("\n=== TOP 15 by WR (n>=12) ===")
    big = res_df[res_df["n"] >= 12].head(15)
    for _, r in big.iterrows():
        print(
            f"{r['session']:<13} vol>={r['vol_min']} conf<={r['conf_max']} wick>={r['wick_min']}"
            f"  n={r['n']}  WR={r['wr']}%  PF={r['pf']}"
        )

    print("\n=== TOP 10 by WR (n>=20) ===")
    large = res_df[res_df["n"] >= 20].head(10)
    for _, r in large.iterrows():
        print(
            f"{r['session']:<13} vol>={r['vol_min']} conf<={r['conf_max']} wick>={r['wick_min']}"
            f"  n={r['n']}  WR={r['wr']}%  PF={r['pf']}"
        )

    print(f"\nFull grid saved: {OUTPUT_DIR / 'posthoc_grid.csv'}")
    print(f"Trades saved:    {OUTPUT_DIR / 'trades_no_regime.csv'}")


if __name__ == "__main__":
    main()
