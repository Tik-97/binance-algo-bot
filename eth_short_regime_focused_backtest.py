from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ethShortTrader import BASE_DIR, CANDLES_1M_PATH, build_features, load_candles, load_reference_eth_shorts
from eth_short_execution_backtest_v1 import (
    NEAREST_CANDIDATES_PATH,
    build_match_summary,
    build_reference_lookup,
    load_nearest_candidates,
    run_candidate_execution_backtest,
)
from eth_short_execution_backtest_v2 import run_candidate_execution_backtest_v2
from eth_short_execution_backtest_v3 import run_candidate_execution_backtest_v3


OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_regime_focused_backtest"
REGIME_TAG_PATH = BASE_DIR / "fun_backtests" / "eth_short_day_regime_split_analysis" / "day_regime_tagged.csv"
TARGET_REGIMES = ["mixed_transition", "near_entry_breakdown_noise"]


def load_regime_tags(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["trade_date", "day_regime"]).copy()


def build_scopes(regime_tags: pd.DataFrame) -> list[tuple[str, set[str]]]:
    scopes: list[tuple[str, set[str]]] = []
    all_dates = set(regime_tags["trade_date"])

    target_dates = set(regime_tags.loc[regime_tags["day_regime"].isin(TARGET_REGIMES), "trade_date"])
    other_dates = all_dates - target_dates

    scopes.append(("target_regimes_combined", target_dates))
    for regime in TARGET_REGIMES:
        scopes.append((regime, set(regime_tags.loc[regime_tags["day_regime"].eq(regime), "trade_date"])))
    scopes.append(("other_regimes_combined", other_dates))
    return scopes


def run_scope_backtests(
    features: pd.DataFrame,
    nearest_candidates: pd.DataFrame,
    ref_lookup: pd.DataFrame,
    scope_name: str,
    trade_dates: set[str],
) -> pd.DataFrame:
    subset = nearest_candidates.loc[nearest_candidates["trade_date"].isin(trade_dates)].copy()
    subset = subset.sort_values("trade_opened_time").reset_index(drop=True)

    runners = {
        "v1": run_candidate_execution_backtest,
        "v2": run_candidate_execution_backtest_v2,
        "v3": run_candidate_execution_backtest_v3,
    }

    summary_rows: list[dict] = []
    for version, runner in runners.items():
        trades_df, summary = runner(features, subset)
        match_df = build_match_summary(trades_df, ref_lookup)

        trades_df.to_csv(OUTPUT_DIR / f"{scope_name}__{version}__trades.csv", index=False)
        match_df.to_csv(OUTPUT_DIR / f"{scope_name}__{version}__match_summary.csv", index=False)

        row = {
            "scope_name": scope_name,
            "version": version,
            "candidate_scope_count": int(len(subset)),
            "filled_trade_count": int(summary.get("filled_trade_count", 0)),
            "ending_balance": float(summary.get("ending_balance", np.nan)),
            "win_rate_pct": float(summary.get("win_rate_pct", np.nan)),
            "profit_factor": float(summary.get("profit_factor", np.nan)) if summary.get("profit_factor", np.nan) is not None else np.nan,
            "expectancy_usd": float(summary.get("expectancy_usd", np.nan)),
            "median_hold_minutes": float(summary.get("median_hold_minutes", np.nan)),
            "median_entry_vs_trader_price_pct": float(summary.get("median_entry_vs_trader_price_pct", np.nan)),
            "same_result_direction_share_pct": float(match_df["same_result_direction"].mean() * 100.0) if not match_df.empty else np.nan,
            "median_delta_vs_trader_return_pct": float(match_df["delta_vs_trader_return_pct"].median()) if not match_df.empty else np.nan,
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def write_summary(summary_df: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT REGIME FOCUSED BACKTEST",
        "=================================",
        "",
        f"target_regimes = {', '.join(TARGET_REGIMES)}",
        "",
    ]

    for scope_name, scope_df in summary_df.groupby("scope_name", sort=False):
        lines.extend([scope_name, "-" * len(str(scope_name))])
        for _, row in scope_df.sort_values("version").iterrows():
            lines.extend(
                [
                    f"{row['version']}: candidate_scope_count={int(row['candidate_scope_count'])}, filled_trade_count={int(row['filled_trade_count'])}, ending_balance={row['ending_balance']:.2f}, win_rate_pct={row['win_rate_pct']:.2f}, profit_factor={row['profit_factor']:.3f}" if pd.notna(row["profit_factor"]) else f"{row['version']}: candidate_scope_count={int(row['candidate_scope_count'])}, filled_trade_count={int(row['filled_trade_count'])}, ending_balance={row['ending_balance']:.2f}, win_rate_pct={row['win_rate_pct']:.2f}, profit_factor=nan",
                    f"median_entry_vs_trader_price_pct={row['median_entry_vs_trader_price_pct']:.4f}, same_result_direction_share_pct={row['same_result_direction_share_pct']:.2f}, median_delta_vs_trader_return_pct={row['median_delta_vs_trader_return_pct']:.4f}",
                ]
            )
        lines.append("")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = build_features(load_candles(CANDLES_1M_PATH))
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH)
    ref_lookup = build_reference_lookup(load_reference_eth_shorts(BASE_DIR / Path(r"Another Trader/another_trader.json")))
    regime_tags = load_regime_tags(REGIME_TAG_PATH)

    summary_parts: list[pd.DataFrame] = []
    for scope_name, trade_dates in build_scopes(regime_tags):
        summary_parts.append(run_scope_backtests(features, nearest_candidates, ref_lookup, scope_name, trade_dates))

    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    summary_df.to_csv(OUTPUT_DIR / "regime_backtest_summary.csv", index=False)
    write_summary(summary_df)

    print(f"Scopes evaluated: {summary_df['scope_name'].nunique() if not summary_df.empty else 0}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
