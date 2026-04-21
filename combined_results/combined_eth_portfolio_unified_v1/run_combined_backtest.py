from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

STARTING_BALANCE = 1000.0

SHORT_PATH = BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier287_realistic_backtest_v1" / "trades.csv"
LONG_T14_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv"
LONG_F21_PATH = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_family21_seed_plus_satellites_v1" / "trades.csv"


def load_strategy_trades(path: Path, strategy_name: str, priority: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["strategy"] = strategy_name
    df["priority"] = priority
    df["source_balance_before"] = pd.to_numeric(df["balance_before"], errors="coerce")
    df["source_pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
    return df


def summarize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for strategy, part in df.groupby("strategy", dropna=False):
        wins = int((part["pnl_usd"] > 0).sum())
        losses = int((part["pnl_usd"] < 0).sum())
        gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        n = len(part)
        rows.append({
            "strategy": strategy,
            "trade_count": n,
            "win_rate_pct": round(wins / n * 100.0, 2) if n else 0.0,
            "net_pnl_usd": round(float(part["pnl_usd"].sum()), 4),
            "profit_factor": round(pf, 3) if pf != float("inf") else pf,
            "avg_return_pct": round(float(part["return_on_balance"].mean() * 100.0), 4),
        })
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group_value, part in df.groupby(group_col, dropna=False):
        wins = int((part["pnl_usd"] > 0).sum())
        losses = int((part["pnl_usd"] < 0).sum())
        gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        n = len(part)
        rows.append({
            group_col: group_value,
            "trade_count": n,
            "win_rate_pct": round(wins / n * 100.0, 2) if n else 0.0,
            "net_pnl_usd": round(float(part["pnl_usd"].sum()), 4),
            "profit_factor": round(pf, 3) if pf != float("inf") else pf,
            "avg_return_pct": round(float(part["return_on_balance"].mean() * 100.0), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_trades = pd.concat(
        [
            load_strategy_trades(SHORT_PATH, "short_t287", 1),
            load_strategy_trades(LONG_T14_PATH, "long_t14", 2),
            load_strategy_trades(LONG_F21_PATH, "long_f21", 3),
        ],
        ignore_index=True,
    )

    # --- Step 1: dedup exact same-side same-entry/exit signals by priority ---
    all_trades["dup_key"] = (
        all_trades["side"].astype(str)
        + "|"
        + all_trades["entry_time"].apply(lambda t: t.isoformat())
        + "|"
        + all_trades["exit_time"].apply(lambda t: t.isoformat())
    )

    # Sort once, assign to variable, apply mask to the same object (bug fix from old script)
    sorted_all = all_trades.sort_values(["entry_time", "priority"]).reset_index(drop=True)
    dup_mask = sorted_all.duplicated("dup_key", keep="first")
    skipped_duplicates = sorted_all.loc[dup_mask].copy()
    deduped = sorted_all.loc[~dup_mask].copy()
    exact_dup_count = int(dup_mask.sum())

    # --- Step 2: router arbitration — resolve same-minute conflicts ---
    # Rule A: short always beats long if same entry_time minute
    # Rule B: if long_t14 and long_f21 have same entry_time minute, keep long_t14 only
    deduped["entry_minute"] = deduped["entry_time"].dt.floor("1min")

    same_minute_long_conflict_count = 0
    short_beat_long_count = 0

    rows_to_drop: set[int] = set()
    for minute, group in deduped.groupby("entry_minute"):
        strats = set(group["strategy"].tolist())
        has_short = "short_t287" in strats
        has_long_t14 = "long_t14" in strats
        has_long_f21 = "long_f21" in strats

        if has_short and (has_long_t14 or has_long_f21):
            # Short beats all longs on same minute
            long_idx = group.loc[group["strategy"].isin(["long_t14", "long_f21"])].index
            rows_to_drop.update(long_idx.tolist())
            short_beat_long_count += 1

        elif has_long_t14 and has_long_f21:
            # Same-minute long conflict: keep long_t14, drop long_f21
            f21_idx = group.loc[group["strategy"] == "long_f21"].index
            rows_to_drop.update(f21_idx.tolist())
            same_minute_long_conflict_count += 1

    arbitrated = deduped.loc[~deduped.index.isin(rows_to_drop)].copy()
    arbitrated = arbitrated.sort_values(["entry_time", "priority"]).reset_index(drop=True)

    # --- Step 3: single-wallet sequential simulation ---
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    portfolio_rows: list[dict] = []
    active_exit = pd.Timestamp.min.tz_localize("UTC")
    skipped_overlaps: list[dict] = []

    for row in arbitrated.to_dict(orient="records"):
        entry_time = row["entry_time"]
        exit_time = row["exit_time"]

        if entry_time < active_exit:
            skipped_overlaps.append({
                "strategy": row["strategy"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": row["side"],
                "reason": "overlap_with_open_position",
            })
            continue

        source_pnl_usd = float(row["source_pnl_usd"])
        source_balance_before = float(row["source_balance_before"])
        return_on_balance = source_pnl_usd / source_balance_before if source_balance_before != 0 else 0.0

        pnl_usd = balance * return_on_balance
        balance_before = balance
        balance_after = balance_before + pnl_usd
        peak_balance = max(peak_balance, balance_after)
        drawdown_pct = (peak_balance - balance_after) / peak_balance * 100.0 if peak_balance > 0 else 0.0

        portfolio_rows.append({
            "strategy": row["strategy"],
            "side": row["side"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "route": row.get("route", ""),
            "source_pnl_usd": source_pnl_usd,
            "source_balance_before": source_balance_before,
            "return_on_balance": return_on_balance,
            "balance_before": balance_before,
            "pnl_usd": pnl_usd,
            "balance_after": balance_after,
            "drawdown_pct": drawdown_pct,
        })

        balance = balance_after
        active_exit = exit_time

    portfolio = pd.DataFrame(portfolio_rows)
    skipped_overlap_df = pd.DataFrame(skipped_overlaps)

    # --- Save CSVs ---
    portfolio.to_csv(OUTPUT_DIR / "portfolio_trades.csv", index=False)
    skipped_duplicates.to_csv(OUTPUT_DIR / "skipped_duplicates.csv", index=False)
    skipped_overlap_df.to_csv(OUTPUT_DIR / "skipped_overlaps.csv", index=False)

    if portfolio.empty:
        raise ValueError("Combined portfolio produced no trades.")

    # --- Summaries ---
    strategy_summary = summarize_strategy(portfolio)
    strategy_summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)

    side_summary = summarize_group(portfolio, "side")
    side_summary.to_csv(OUTPUT_DIR / "side_summary.csv", index=False)

    portfolio["year_month"] = portfolio["entry_time"].dt.strftime("%Y-%m")
    month_summary = summarize_group(portfolio, "year_month")
    month_summary.to_csv(OUTPUT_DIR / "month_summary.csv", index=False)

    # --- Overall stats ---
    wins = portfolio.loc[portfolio["pnl_usd"] > 0, "pnl_usd"]
    losses = portfolio.loc[portfolio["pnl_usd"] < 0, "pnl_usd"]
    trade_count = int(len(portfolio))
    win_rate = float((portfolio["pnl_usd"] > 0).mean() * 100.0)
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    max_dd = float(portfolio["drawdown_pct"].max())
    ending_balance = float(portfolio["balance_after"].iloc[-1])
    net_pnl = ending_balance - STARTING_BALANCE
    window_start = portfolio["entry_time"].min()
    window_end = portfolio["exit_time"].max()

    # Per-strategy contribution
    def _strat_row(name: str) -> dict:
        part = portfolio.loc[portfolio["strategy"] == name]
        if part.empty:
            return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "pf": 0.0}
        w = part.loc[part["pnl_usd"] > 0, "pnl_usd"]
        l = part.loc[part["pnl_usd"] < 0, "pnl_usd"]
        n = len(part)
        gp = float(w.sum())
        gl = float(-l.sum())
        return {
            "trades": n,
            "win_rate": round(float((part["pnl_usd"] > 0).mean() * 100.0), 2),
            "net_pnl": round(float(part["pnl_usd"].sum()), 4),
            "pf": round(gp / gl, 3) if gl > 0 else float("inf"),
        }

    s287 = _strat_row("short_t287")
    lt14 = _strat_row("long_t14")
    lf21 = _strat_row("long_f21")

    summary_lines = [
        "COMBINED ETH PORTFOLIO UNIFIED V1",
        "==================================",
        "",
        "strategies_included:",
        "- short_t287 (priority 1, SHORT)",
        "- long_t14 (priority 2, LONG)",
        "- long_f21 (priority 3, LONG)",
        "",
        "wallet_model = one shared wallet, one ETH position at a time",
        "arbitration = short beats long, long_t14 beats long_f21 on same minute",
        f"initial_balance = {STARTING_BALANCE:.2f}",
        f"window_start = {window_start}",
        f"window_end = {window_end}",
        "",
        "deduplication:",
        f"- exact_duplicate_signals_removed = {exact_dup_count}",
        f"- overlapping_trades_skipped = {len(skipped_overlaps)}",
        f"- same_minute_long_conflicts_resolved = {same_minute_long_conflict_count}",
        f"- short_beat_long_conflicts = {short_beat_long_count}",
        "",
        f"trade_count = {trade_count}",
        f"win_rate_pct = {win_rate:.2f}",
        f"profit_factor = {round(pf, 3) if pf != float('inf') else 'inf'}",
        f"max_drawdown_pct = {max_dd:.2f}",
        f"ending_balance = {ending_balance:.2f}",
        f"net_pnl_usd = {net_pnl:.2f}",
        "",
        "strategy_contribution:",
        f"- short_t287: trades={s287['trades']}, win_rate={s287['win_rate']}%, net_pnl={s287['net_pnl']}, pf={s287['pf']}",
        f"- long_t14: trades={lt14['trades']}, win_rate={lt14['win_rate']}%, net_pnl={lt14['net_pnl']}, pf={lt14['pf']}",
        f"- long_f21: trades={lf21['trades']}, win_rate={lf21['win_rate']}%, net_pnl={lf21['net_pnl']}, pf={lf21['pf']}",
        "",
        "honest_interpretation:",
        "- this is a conservative single-wallet merge",
        "- short_t287 priority means it blocks all longs when active",
        "- long_t14 and long_f21 were deduped on same-minute signals",
        "- results reflect realistic sequencing, not parallel fantasy",
    ]

    summary_text = "\n".join(summary_lines)
    (OUTPUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
