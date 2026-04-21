from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
SEED = 42
N_MC = 10_000
MC_CURVE_SAMPLES = 100          # equity curves saved for visualisation
STARTING_BALANCE = 1_000.0

BASE_DIR = Path(__file__).resolve().parents[2]
PORTFOLIO_DIR = BASE_DIR / "combined_results" / "combined_eth_portfolio_unified_v1"
OUTPUT_DIR = Path(__file__).resolve().parent

SHORT_SRC = BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier287_realistic_backtest_v1" / "trades.csv"
T14_SRC   = BASE_DIR / "next_phase_long_focus"  / "results" / "edge_long_tier14_core_plus_satellite2_v1"    / "trades.csv"
F21_SRC   = BASE_DIR / "next_phase_long_focus"  / "results" / "edge_long_family21_seed_plus_satellites_v1" / "trades.csv"

# ── scenario definitions ──────────────────────────────────────────────────────
# extra_slip  = additional slippage as % of trade notional (entry + exit combined)
# flip_win    = fraction of winning trades randomly converted to avg-loss-return
# miss_limit  = fraction of limit-entry winning trades removed entirely (no fill)
SCENARIOS = [
    dict(name="baseline",       extra_slip=0.00, flip_win=0.00, miss_limit=0.00,
         note="Exact combined backtest. No modifications."),
    dict(name="adverse_light",  extra_slip=0.05, flip_win=0.00, miss_limit=0.10,
         note="+0.05% notional slip each side, 10% limit orders don't fill."),
    dict(name="adverse_medium", extra_slip=0.10, flip_win=0.10, miss_limit=0.20,
         note="+0.10% slip, 10% wins -> avg loss, 20% limit misses."),
    dict(name="adverse_hard",   extra_slip=0.20, flip_win=0.25, miss_limit=0.35,
         note="+0.20% slip, 25% wins -> avg loss, 35% limit misses."),
]

# ── sensitivity grid params ───────────────────────────────────────────────────
SLIP_GRID  = [0.00, 0.025, 0.05, 0.10, 0.15, 0.20]   # extra slippage % of notional
FILL_GRID  = [1.00, 0.90,  0.80, 0.70, 0.60]          # limit order fill rate


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_curve(returns: np.ndarray, start: float = STARTING_BALANCE
                  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compound returns sequentially.

    Returns
    -------
    curve      : balance after each trade
    pnl_arr    : dollar pnl of each trade
    max_dd_pct : maximum drawdown over the sequence
    final_bal  : ending balance
    """
    n = len(returns)
    if n == 0:
        return np.array([]), np.array([]), 0.0, start
    curve   = np.empty(n)
    pnl_arr = np.empty(n)
    bal  = start
    peak = start
    max_dd = 0.0
    for i, r in enumerate(returns):
        p      = bal * r
        bal   += p
        pnl_arr[i] = p
        curve[i]   = bal
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return curve, pnl_arr, max_dd, bal


def metrics_from_curve(pnl_arr: np.ndarray, max_dd: float, final_bal: float) -> dict:
    n = len(pnl_arr)
    if n == 0:
        return dict(trade_count=0, win_rate_pct=0.0, profit_factor=0.0,
                    max_drawdown_pct=0.0, ending_balance=round(STARTING_BALANCE, 2),
                    net_pnl_usd=0.0)
    wins   = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    gross_profit = float(wins.sum())
    gross_loss   = float(abs(losses.sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return dict(
        trade_count      = n,
        win_rate_pct     = round(float(len(wins) / n * 100), 2),
        profit_factor    = round(pf, 3) if pf != float("inf") else pf,
        max_drawdown_pct = round(max_dd, 2),
        ending_balance   = round(final_bal, 2),
        net_pnl_usd      = round(final_bal - STARTING_BALANCE, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + enrichment
# ══════════════════════════════════════════════════════════════════════════════

def load_portfolio() -> pd.DataFrame:
    df = pd.read_csv(PORTFOLIO_DIR / "portfolio_trades.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    return df.sort_values("entry_time").reset_index(drop=True)


def load_source_meta() -> pd.DataFrame:
    """Pull entry_mode and notional_usd from the three raw backtest CSVs."""
    frames = []
    for path, strat in [(SHORT_SRC, "short_t287"), (T14_SRC, "long_t14"), (F21_SRC, "long_f21")]:
        needed = ["entry_time", "entry_mode", "notional_usd", "balance_before"]
        src = pd.read_csv(path, usecols=needed)
        src["entry_time"] = pd.to_datetime(src["entry_time"], utc=True)
        src["strategy"]   = strat
        src.rename(columns={"balance_before": "src_balance_before"}, inplace=True)
        frames.append(src)
    return pd.concat(frames, ignore_index=True)


def enrich(portfolio: pd.DataFrame, source_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge notional/leverage/entry-mode info into portfolio."""
    merged = portfolio.merge(source_meta, on=["strategy", "entry_time"], how="left")
    merged["is_limit"]    = merged["entry_mode"].fillna("").str.lower().str.contains("limit")
    lev = merged["notional_usd"] / merged["src_balance_before"]
    # clip to sensible range; fill missing with empirical midpoint
    merged["leverage_est"] = lev.fillna(3.0).clip(0.5, 20.0)
    merged["adj_return"]   = merged["return_on_balance"]   # baseline default
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Scenario application
# ══════════════════════════════════════════════════════════════════════════════

def apply_scenario(enriched: pd.DataFrame, extra_slip: float, flip_win: float,
                   miss_limit: float, avg_loss_ret: float,
                   rng: np.random.Generator) -> pd.DataFrame:
    df = enriched.copy()

    # 1. Extra slippage: proportional to leverage (cost as fraction of balance)
    df["adj_return"] = df["return_on_balance"] - (extra_slip / 100.0 * df["leverage_est"])

    # 2. Limit-order miss: drop X% of winning limit-entry trades entirely
    if miss_limit > 0:
        limit_win_idx = df.index[(df["is_limit"]) & (df["adj_return"] > 0)].tolist()
        n_miss = int(round(len(limit_win_idx) * miss_limit))
        if n_miss > 0:
            dropped = rng.choice(limit_win_idx, size=n_miss, replace=False)
            df = df.drop(index=dropped).reset_index(drop=True)

    # 3. Win flip: convert X% of remaining winners to average losing return
    if flip_win > 0:
        win_idx = df.index[df["adj_return"] > 0].tolist()
        n_flip = int(round(len(win_idx) * flip_win))
        if n_flip > 0:
            flipped = rng.choice(win_idx, size=n_flip, replace=False)
            df.loc[flipped, "adj_return"] = avg_loss_ret

    return df


def reconstruct_balance_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add adj_balance_before / adj_pnl_usd / adj_balance_after / adj_drawdown_pct."""
    bal  = STARTING_BALANCE
    peak = STARTING_BALANCE
    bal_bef, pnl_usd, bal_aft, dds = [], [], [], []
    for r in df["adj_return"].values:
        p   = bal * r
        bal += p
        peak = max(peak, bal)
        dd   = (peak - bal) / peak * 100.0 if peak > 0 else 0.0
        bal_bef.append(round(bal - p, 4))
        pnl_usd.append(round(p, 4))
        bal_aft.append(round(bal, 4))
        dds.append(round(dd, 4))
    df = df.copy()
    df["adj_balance_before"] = bal_bef
    df["adj_pnl_usd"]        = pnl_usd
    df["adj_balance_after"]  = bal_aft
    df["adj_drawdown_pct"]   = dds
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(baseline_returns: np.ndarray,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap (with replacement) MC on baseline adj_returns."""
    n = len(baseline_returns)
    final_bals = np.empty(N_MC)
    max_dds    = np.empty(N_MC)

    curve_save_idx = set(rng.choice(N_MC, size=MC_CURVE_SAMPLES, replace=False).tolist())
    saved_curves: list[list[float]] = []

    for i in range(N_MC):
        sample                = rng.choice(baseline_returns, size=n, replace=True)
        curve, _, max_dd, fb  = compute_curve(sample)
        final_bals[i] = fb
        max_dds[i]    = max_dd
        if i in curve_save_idx:
            saved_curves.append(curve.tolist())

    pcts = [5, 10, 25, 50, 75, 90, 95]

    def pct_row(metric: str, arr: np.ndarray) -> dict:
        row: dict = {"metric": metric}
        for p in pcts:
            row[f"p{p}"] = round(float(np.percentile(arr, p)), 2)
        row["mean"] = round(float(arr.mean()), 2)
        row["std"]  = round(float(arr.std()),  2)
        return row

    bal_row = pct_row("ending_balance", final_bals)
    bal_row["prob_profit_pct"]    = round(float((final_bals > STARTING_BALANCE).mean()       * 100), 1)
    bal_row["prob_double_pct"]    = round(float((final_bals > STARTING_BALANCE * 2).mean()   * 100), 1)
    bal_row["prob_lose_half_pct"] = round(float((final_bals < STARTING_BALANCE * 0.5).mean() * 100), 1)

    dd_row = pct_row("max_drawdown_pct", max_dds)

    mc_summary = pd.DataFrame([bal_row, dd_row])

    # Pad all saved curves to the same length
    max_len = max((len(c) for c in saved_curves), default=0)
    curves_df = pd.DataFrame([
        c + [None] * (max_len - len(c)) for c in saved_curves
    ])
    curves_df.index.name = "curve_id"

    return mc_summary, curves_df


# ══════════════════════════════════════════════════════════════════════════════
# Rolling window analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_rolling(enriched: pd.DataFrame, window_months: int) -> pd.DataFrame:
    df = enriched.sort_values("entry_time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    start_date = df["entry_time"].min().normalize()
    end_date   = df["exit_time"].max().normalize()
    rows: list[dict] = []

    current = start_date
    while True:
        wend = current + pd.DateOffset(months=window_months)
        if wend > end_date + pd.Timedelta(days=1):
            break
        mask = (df["entry_time"] >= current) & (df["entry_time"] < wend)
        sub  = df.loc[mask]
        n    = len(sub)
        row: dict = {
            "window_start":  current.strftime("%Y-%m-%d"),
            "window_end":    wend.strftime("%Y-%m-%d"),
            "window_months": window_months,
            "trade_count":   n,
        }
        if n >= 2:
            rets                   = sub["adj_return"].values
            _, pnl_arr, max_dd, fb = compute_curve(rets, STARTING_BALANCE)
            wins   = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr < 0]
            pf     = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else float("inf")
            row.update(dict(
                win_rate_pct    = round(float(len(wins) / n * 100), 2),
                net_pnl_pct     = round((fb - STARTING_BALANCE) / STARTING_BALANCE * 100, 2),
                profit_factor   = round(pf, 3) if pf != float("inf") else pf,
                max_drawdown_pct= round(max_dd, 2),
                strategies      = ",".join(sorted(sub["strategy"].unique())),
            ))
        else:
            row.update(dict(win_rate_pct=None, net_pnl_pct=None,
                            profit_factor=None, max_drawdown_pct=None,
                            strategies=",".join(sorted(sub["strategy"].unique())) if n else ""))
        rows.append(row)
        current += pd.DateOffset(months=1)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity grid
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity(enriched: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    for extra_slip in SLIP_GRID:
        for fill_rate in FILL_GRID:
            miss_rate = 1.0 - fill_rate
            df = enriched.copy()
            df["adj_return"] = df["return_on_balance"] - (extra_slip / 100.0 * df["leverage_est"])

            if miss_rate > 0:
                lw_idx = df.index[(df["is_limit"]) & (df["adj_return"] > 0)].tolist()
                n_miss = int(round(len(lw_idx) * miss_rate))
                if n_miss > 0:
                    dropped = rng.choice(lw_idx, size=n_miss, replace=False)
                    df = df.drop(index=dropped).reset_index(drop=True)

            rets                   = df["adj_return"].values
            _, pnl_arr, max_dd, fb = compute_curve(rets)
            n      = len(rets)
            wins   = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr < 0]
            pf     = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else float("inf")
            rows.append(dict(
                extra_slippage_pct_notional = extra_slip,
                limit_fill_rate             = fill_rate,
                trade_count                 = n,
                win_rate_pct                = round(float(len(wins) / n * 100), 2) if n > 0 else 0.0,
                profit_factor               = round(pf, 3) if pf != float("inf") else pf,
                max_drawdown_pct            = round(max_dd, 2),
                ending_balance              = round(fb, 2),
                net_pnl_usd                 = round(fb - STARTING_BALANCE, 2),
            ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Summary text
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(scenario_summary: pd.DataFrame, mc_summary: pd.DataFrame,
                  rolling_dfs: dict[int, pd.DataFrame],
                  sens_df: pd.DataFrame, avg_loss_ret: float) -> str:
    lines: list[str] = [
        "COMBINED ETH PORTFOLIO UNIFIED V1 - STRESS TEST",
        "=" * 50,
        "",
        f"initial_balance    = {STARTING_BALANCE:.2f}",
        f"n_mc_iterations    = {N_MC:,}",
        f"avg_loss_return    = {avg_loss_ret*100:.3f}%  (used for win-flip in adverse scenarios)",
        "",
    ]

    # ── Section 1: Scenarios ──────────────────────────────────────────────────
    lines += ["SECTION 1 - SCENARIO ANALYSIS", "-" * 40]
    col_w = 18
    header = f"{'':20s} {'baseline':>12} {'adv_light':>12} {'adv_medium':>12} {'adv_hard':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for field in ["trade_count", "win_rate_pct", "profit_factor",
                  "max_drawdown_pct", "ending_balance", "net_pnl_usd"]:
        parts = [f"{field:20s}"]
        for _, row in scenario_summary.iterrows():
            v = row[field]
            parts.append(f"{str(v):>12}")
        lines.append("  ".join(parts))

    lines.append("")
    for _, row in scenario_summary.iterrows():
        sc = next(s for s in SCENARIOS if s["name"] == row["scenario"])
        lines.append(f"  [{row['scenario']}]  {sc['note']}")

    # ── Section 2: Monte Carlo ────────────────────────────────────────────────
    lines += ["", "SECTION 2 - MONTE CARLO (bootstrap, with replacement)", "-" * 40]
    bal_row = mc_summary.loc[mc_summary["metric"] == "ending_balance"].iloc[0]
    dd_row  = mc_summary.loc[mc_summary["metric"] == "max_drawdown_pct"].iloc[0]

    lines += [
        "  Ending Balance distribution:",
        f"    p5   = {bal_row['p5']:.2f}    (worst 5% of reshuffled outcomes)",
        f"    p25  = {bal_row['p25']:.2f}",
        f"    p50  = {bal_row['p50']:.2f}    (median)",
        f"    p75  = {bal_row['p75']:.2f}",
        f"    p95  = {bal_row['p95']:.2f}    (best 5%)",
        f"    mean = {bal_row['mean']:.2f}   std = {bal_row['std']:.2f}",
        "",
        "  Max Drawdown distribution:",
        f"    p25  = {dd_row['p25']:.2f}%",
        f"    p50  = {dd_row['p50']:.2f}%",
        f"    p75  = {dd_row['p75']:.2f}%",
        f"    p95  = {dd_row['p95']:.2f}%",
        "",
        f"  prob_profit    = {bal_row['prob_profit_pct']:.1f}%   (end above {STARTING_BALANCE:.0f})",
        f"  prob_double    = {bal_row['prob_double_pct']:.1f}%   (end above {STARTING_BALANCE*2:.0f})",
        f"  prob_lose_half = {bal_row['prob_lose_half_pct']:.1f}%   (end below {STARTING_BALANCE*0.5:.0f})",
    ]

    # ── Section 3: Rolling windows ────────────────────────────────────────────
    lines += ["", "SECTION 3 - ROLLING WINDOW PERFORMANCE", "-" * 40]
    for months, rdf in rolling_dfs.items():
        valid = rdf.dropna(subset=["net_pnl_pct"])
        if valid.empty:
            lines.append(f"  {months}m window: no complete windows with >=2 trades")
            continue
        best  = valid.loc[valid["net_pnl_pct"].idxmax()]
        worst = valid.loc[valid["net_pnl_pct"].idxmin()]
        n_profitable = int((valid["net_pnl_pct"] > 0).sum())
        lines += [
            f"  {months}m window ({len(valid)} windows, step=1m):",
            f"    profitable_windows = {n_profitable}/{len(valid)} ({n_profitable/len(valid)*100:.0f}%)",
            f"    best:  {best['window_start']} to {best['window_end']}  "
            f"pnl={best['net_pnl_pct']:+.1f}%  trades={int(best['trade_count'])}",
            f"    worst: {worst['window_start']} to {worst['window_end']}  "
            f"pnl={worst['net_pnl_pct']:+.1f}%  trades={int(worst['trade_count'])}",
        ]

    # ── Section 4: Sensitivity digest ─────────────────────────────────────────
    lines += ["", "SECTION 4 - SENSITIVITY GRID DIGEST", "-" * 40,
              "  (full grid -> sensitivity_grid.csv)",
              "",
              f"  {'slip%':>6}  {'fill%':>6}  {'end_bal':>9}  {'max_dd':>7}  {'pf':>6}"]
    key_rows = sens_df[
        (sens_df["extra_slippage_pct_notional"].isin([0.0, 0.05, 0.10, 0.20])) &
        (sens_df["limit_fill_rate"].isin([1.0, 0.80, 0.60]))
    ]
    for _, row in key_rows.iterrows():
        pf_str = f"{row['profit_factor']:.3f}" if row["profit_factor"] != float("inf") else "  inf"
        lines.append(
            f"  {row['extra_slippage_pct_notional']:>6.3f}  "
            f"{row['limit_fill_rate']*100:>5.0f}%  "
            f"{row['ending_balance']:>9.2f}  "
            f"{row['max_drawdown_pct']:>6.2f}%  "
            f"{pf_str:>6}"
        )

    lines += [
        "",
        "honest_interpretation:",
        "  - MC reshuffles the 59-trade sequence 10,000 times; it does NOT add new trades",
        "  - small sample size (especially 5 F21 + 12 T14 trades) makes MC wide",
        "  - adverse_hard scenario is designed as a realistic live-trading worst case",
        "  - sensitivity grid tests execution quality only (no win-rate degradation)",
        "  - rolling windows expose which calendar periods drove all the profit",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── Load & enrich ─────────────────────────────────────────────────────────
    portfolio    = load_portfolio()
    source_meta  = load_source_meta()
    enriched     = enrich(portfolio, source_meta)

    losing_rets  = enriched.loc[enriched["return_on_balance"] < 0, "return_on_balance"]
    if losing_rets.empty:
        avg_win_ret  = float(enriched.loc[enriched["return_on_balance"] > 0, "return_on_balance"].mean())
        avg_loss_ret = -avg_win_ret * 1.5
    else:
        avg_loss_ret = float(losing_rets.mean())

    print(f"Loaded {len(enriched)} portfolio trades.")
    print(f"avg_loss_return = {avg_loss_ret*100:.3f}%  "
          f"({'computed from ' + str(len(losing_rets)) + ' real losers' if len(losing_rets) > 0 else 'proxy: -1.5x avg win'})")
    print(f"limit_trades = {enriched['is_limit'].sum()},  "
          f"leverage_est mean = {enriched['leverage_est'].mean():.2f}x")

    # ── Section 1: Scenarios ──────────────────────────────────────────────────
    print("\nRunning scenarios...")
    scenario_rows: list[dict] = []
    for sc in SCENARIOS:
        sdf = apply_scenario(
            enriched, sc["extra_slip"], sc["flip_win"], sc["miss_limit"],
            avg_loss_ret, rng,
        )
        rets                   = sdf["adj_return"].values
        _, pnl_arr, max_dd, fb = compute_curve(rets)
        m = metrics_from_curve(pnl_arr, max_dd, fb)
        m["scenario"]               = sc["name"]
        m["extra_slippage_pct"]     = sc["extra_slip"]
        m["flip_win_pct"]           = sc["flip_win"]
        m["limit_miss_rate"]        = sc["miss_limit"]
        scenario_rows.append(m)

        sdf = reconstruct_balance_cols(sdf)
        sdf.to_csv(OUTPUT_DIR / f"scenario_trades_{sc['name']}.csv", index=False)

    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_summary.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)

    # ── Section 2: Monte Carlo ────────────────────────────────────────────────
    print("Running Monte Carlo (10,000 iterations)...")
    baseline_returns     = enriched["return_on_balance"].values
    mc_summary, mc_curves = run_monte_carlo(baseline_returns, rng)
    mc_summary.to_csv(OUTPUT_DIR / "mc_summary.csv", index=False)
    mc_curves.to_csv(OUTPUT_DIR / "mc_sampled_curves.csv")

    # ── Section 3: Rolling windows ────────────────────────────────────────────
    print("Running rolling windows...")
    rolling_dfs: dict[int, pd.DataFrame] = {}
    for months in [3, 6, 12]:
        rdf = run_rolling(enriched, months)
        rdf.to_csv(OUTPUT_DIR / f"rolling_{months}m.csv", index=False)
        rolling_dfs[months] = rdf

    # ── Section 4: Sensitivity grid ───────────────────────────────────────────
    print("Running sensitivity grid...")
    sens_df = run_sensitivity(enriched, rng)
    sens_df.to_csv(OUTPUT_DIR / "sensitivity_grid.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = build_summary(scenario_summary, mc_summary, rolling_dfs, sens_df, avg_loss_ret)
    (OUTPUT_DIR / "stress_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
