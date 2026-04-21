"""
BTC vs ETH — All Strategies Comparison
=======================================
Runs short_t287, long_t14, long_f21, long_bull_v3 on both ETHUSDT and BTCUSDT.
Same parameters, same logic — just different candle data.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1
import next_phase_long_focus.edge_long_tier14_core_plus_satellite2_v1 as t14_mod
import next_phase_long_focus.edge_long_family21_seed_plus_satellites_v1 as f21_mod
import next_phase_long_focus.edge_long_family2_flush_reversal_seed_v1 as family2_seed
import next_phase_long_focus.edge_long_bull_continuation_v3 as bv3

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE = REPO_ROOT / ".cache" / "binance_klines"
ETH_PATH = CACHE / "ETHUSDT_1m.csv"
BTC_PATH = CACHE / "BTCUSDT_1m.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "btc_vs_eth_comparison"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "pf": float("nan"),
                "net": 0.0, "end_bal": base.STARTING_BALANCE, "dd": float("nan")}
    wr = float((df["pnl_usd"] > 0).mean() * 100)
    gp = float(df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum())
    gl = float(-df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 3) if pf != float("inf") else float("inf"),
            "net": round(df["pnl_usd"].sum(), 2), "end_bal": round(base.STARTING_BALANCE + df["pnl_usd"].sum(), 2)}


def _run_with_path(candle_path: Path, runner_fn) -> tuple[pd.DataFrame, dict]:
    orig_path = base.CANDLES_1M_PATH
    try:
        base.CANDLES_1M_PATH = candle_path
        return runner_fn()
    finally:
        base.CANDLES_1M_PATH = orig_path


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_long_t14(candle_path: Path) -> pd.DataFrame:
    candles = base.load_candles(candle_path)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    orig_ctx = base.is_base_context
    orig_trig = base.find_trigger_index
    try:
        base.is_base_context   = t14_mod.make_tier1_context_gate()
        base.find_trigger_index = t14_mod.make_route_aware_find_trigger()
        bt_start = featured["open_time"].min()
        bt_end   = featured["open_time"].max()
        trades, _ = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context   = orig_ctx
        base.find_trigger_index = orig_trig
    return trades


def run_long_f21(candle_path: Path) -> pd.DataFrame:
    candles = base.load_candles(candle_path)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    featured = family2_seed.enrich_family2_context(featured)   # required for f21 context
    orig_ctx = base.is_base_context
    orig_trig = base.find_trigger_index
    try:
        base.is_base_context   = lambda row: True
        base.find_trigger_index = f21_mod.make_route_aware_find_trigger()
        bt_start = featured["open_time"].min()
        bt_end   = featured["open_time"].max()
        trades, _ = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context   = orig_ctx
        base.find_trigger_index = orig_trig
    return trades


def run_bull_v3(candle_path: Path) -> pd.DataFrame:
    candles = base.load_candles(candle_path)
    featured = base.build_features(candles)
    featured = bv3.merge_daily_regime(featured)
    orig_ctx = base.is_base_context
    orig_trig = base.find_trigger_index
    try:
        base.is_base_context   = bv3.is_bull_context
        base.find_trigger_index = bv3.bull_find_trigger_index
        bt_start = featured["open_time"].min()
        bt_end   = featured["open_time"].max()
        trades, _ = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context   = orig_ctx
        base.find_trigger_index = orig_trig
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies = [
        ("long_t14",    run_long_t14),
        ("long_f21",    run_long_f21),
        ("long_bull_v3", run_bull_v3),
    ]

    all_results = []

    for strat_name, runner in strategies:
        print(f"\n{'='*55}")
        print(f"  {strat_name.upper()}")
        print(f"{'='*55}")

        for symbol, candle_path in [("ETH", ETH_PATH), ("BTC", BTC_PATH)]:
            print(f"  Running {symbol}...", end=" ", flush=True)
            try:
                trades = runner(candle_path)
                st = _stats(trades)
                trades.to_csv(OUTPUT_DIR / f"trades_{strat_name}_{symbol}.csv", index=False)

                # Monthly breakdown
                if len(trades):
                    tc = trades.copy()
                    tc["ym"] = pd.to_datetime(tc["entry_time"], utc=True).dt.strftime("%Y-%m")
                    monthly = tc.groupby("ym").agg(
                        n=("pnl_usd","size"),
                        wr=("pnl_usd", lambda x: f"{(x>0).mean()*100:.0f}%"),
                        net=("pnl_usd","sum"),
                    )
                    monthly.to_csv(OUTPUT_DIR / f"monthly_{strat_name}_{symbol}.csv")

                print(f"n={st['n']} WR={st['wr']:.1f}% PF={st['pf']} net=${st['net']}")
                all_results.append({"strategy": strat_name, "symbol": symbol, **st})

            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({"strategy": strat_name, "symbol": symbol,
                                     "n": 0, "wr": float("nan"), "pf": float("nan"),
                                     "net": 0.0, "end_bal": base.STARTING_BALANCE})

    # ── Summary table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)

    print(f"\n{'='*65}")
    print("  BTC vs ETH — COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"{'Strategy':<16} {'Symbol':<6} {'n':>4} {'WR':>7} {'PF':>7} {'Net $':>9} {'End Bal':>9}")
    print("-"*65)
    for _, r in results_df.iterrows():
        wr_s = f"{r['wr']:.1f}%" if not pd.isna(r['wr']) else "n/a"
        pf_s = f"{r['pf']:.3f}" if r['pf'] != float("inf") and not pd.isna(r['pf']) else "inf"
        print(f"{r['strategy']:<16} {r['symbol']:<6} {int(r['n']):>4} {wr_s:>7} {pf_s:>7} {r['net']:>9.2f} {r['end_bal']:>9.2f}")

    # Per-strategy delta
    print(f"\n{'='*65}")
    print("  COMBINED (ETH + BTC no overlap check)")
    print(f"{'='*65}")
    for strat in results_df["strategy"].unique():
        sub = results_df[results_df["strategy"] == strat]
        total_n = sub["n"].sum()
        eth_row = sub[sub["symbol"]=="ETH"]
        btc_row = sub[sub["symbol"]=="BTC"]
        eth_n = int(eth_row["n"].values[0]) if len(eth_row) else 0
        btc_n = int(btc_row["n"].values[0]) if len(btc_row) else 0
        print(f"  {strat}: ETH={eth_n} + BTC={btc_n} = {total_n} total trades")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
