from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "fun_backtests" / "eth_short_post_candidate_vs_trader_analysis" / "candidate_vs_trader_comparison.csv"
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_day_regime_split_analysis"

SMALL_GAP_MAX_PCT = 0.75
EXTRA_PUSH_MIN_PCT = 0.15
LARGE_GAP_MIN_PCT = 1.25


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("trade_date").reset_index(drop=True)


def classify_regime(row: pd.Series) -> str:
    gap = float(row["entry_gap_trader_vs_candidate_pct"])
    extra_push = bool(row["extra_push_before_trader_flag"])
    pre_breakdown = bool(row["pre_breakdown_before_trader_flag"])

    if gap <= SMALL_GAP_MAX_PCT and not extra_push:
        return "near_entry_small_gap"
    if gap >= LARGE_GAP_MIN_PCT and extra_push:
        return "late_entry_extra_push"
    if gap <= SMALL_GAP_MAX_PCT and pre_breakdown:
        return "near_entry_breakdown_noise"
    return "mixed_transition"


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["day_regime"] = enriched.apply(classify_regime, axis=1)
    enriched["is_small_gap"] = enriched["entry_gap_trader_vs_candidate_pct"] <= SMALL_GAP_MAX_PCT
    enriched["is_large_gap"] = enriched["entry_gap_trader_vs_candidate_pct"] >= LARGE_GAP_MIN_PCT
    enriched["is_extra_push"] = enriched["max_up_move_before_trader_pct"] >= EXTRA_PUSH_MIN_PCT
    enriched["is_breakdown_noise"] = enriched["max_down_move_before_trader_pct"] >= EXTRA_PUSH_MIN_PCT
    return enriched


def build_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for regime, group in df.groupby("day_regime", sort=True):
        row = {
            "day_regime": regime,
            "trade_count": int(len(group)),
            "share_pct": float(len(group) / len(df) * 100.0) if len(df) else np.nan,
            "median_candidate_abs_minute_distance": float(group["candidate_abs_minute_distance"].median()),
            "median_trader_delay_from_candidate_minutes": float(group["trader_delay_from_candidate_minutes"].median()),
            "median_entry_gap_trader_vs_candidate_pct": float(group["entry_gap_trader_vs_candidate_pct"].median()),
            "extra_push_share_pct": float(group["extra_push_before_trader_flag"].mean() * 100.0),
            "breakdown_noise_share_pct": float(group["pre_breakdown_before_trader_flag"].mean() * 100.0),
            "median_trader_return_pct": float(group["trader_return_pct"].median()),
        }

        for version in ["v1", "v2", "v3"]:
            pnl_col = f"pnl_usd_{version}"
            entry_col = f"entry_vs_trader_price_pct_{version}"
            if pnl_col in group.columns:
                subset = group.loc[group[pnl_col].notna()].copy()
                row[f"{version}_trade_count"] = int(len(subset))
                row[f"{version}_win_rate_pct"] = float((subset[pnl_col] > 0).mean() * 100.0) if not subset.empty else np.nan
                row[f"{version}_median_pnl_usd"] = float(subset[pnl_col].median()) if not subset.empty else np.nan
                row[f"{version}_median_entry_vs_trader_price_pct"] = float(subset[entry_col].median()) if not subset.empty else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("trade_count", ascending=False).reset_index(drop=True)


def build_regime_detail(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trade_date",
        "day_regime",
        "candidate_abs_minute_distance",
        "trader_delay_from_candidate_minutes",
        "entry_gap_trader_vs_candidate_pct",
        "max_up_move_before_trader_pct",
        "max_down_move_before_trader_pct",
        "trader_return_pct",
        "pnl_usd_v1",
        "exit_reason_v1",
        "pnl_usd_v2",
        "exit_reason_v2",
        "pnl_usd_v3",
        "exit_reason_v3",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].sort_values(["day_regime", "trade_date"]).reset_index(drop=True)


def write_summary(df: pd.DataFrame, regime_summary: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT DAY REGIME SPLIT ANALYSIS",
        "===================================",
        "",
        f"trade_count = {len(df)}",
        f"small_gap_max_pct = {SMALL_GAP_MAX_PCT:.2f}",
        f"large_gap_min_pct = {LARGE_GAP_MIN_PCT:.2f}",
        f"extra_push_min_pct = {EXTRA_PUSH_MIN_PCT:.2f}",
        "",
    ]

    for _, row in regime_summary.iterrows():
        lines.extend(
            [
                row["day_regime"],
                "-" * len(str(row["day_regime"])),
                f"trade_count = {int(row['trade_count'])}",
                f"share_pct = {row['share_pct']:.2f}",
                f"median_candidate_abs_minute_distance = {row['median_candidate_abs_minute_distance']:.2f}",
                f"median_trader_delay_from_candidate_minutes = {row['median_trader_delay_from_candidate_minutes']:.2f}",
                f"median_entry_gap_trader_vs_candidate_pct = {row['median_entry_gap_trader_vs_candidate_pct']:.4f}",
                f"extra_push_share_pct = {row['extra_push_share_pct']:.2f}",
                f"breakdown_noise_share_pct = {row['breakdown_noise_share_pct']:.2f}",
                f"median_trader_return_pct = {row['median_trader_return_pct']:.4f}",
                f"v1_trade_count = {int(row['v1_trade_count']) if pd.notna(row['v1_trade_count']) else 0}",
                f"v1_win_rate_pct = {row['v1_win_rate_pct']:.2f}" if pd.notna(row["v1_win_rate_pct"]) else "v1_win_rate_pct = nan",
                f"v2_trade_count = {int(row['v2_trade_count']) if pd.notna(row['v2_trade_count']) else 0}",
                f"v2_win_rate_pct = {row['v2_win_rate_pct']:.2f}" if pd.notna(row["v2_win_rate_pct"]) else "v2_win_rate_pct = nan",
                f"v3_trade_count = {int(row['v3_trade_count']) if pd.notna(row['v3_trade_count']) else 0}",
                f"v3_win_rate_pct = {row['v3_win_rate_pct']:.2f}" if pd.notna(row["v3_win_rate_pct"]) else "v3_win_rate_pct = nan",
                "",
            ]
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_regime_columns(load_source(SOURCE_PATH))
    regime_summary = build_regime_summary(df)
    regime_detail = build_regime_detail(df)

    df.to_csv(OUTPUT_DIR / "day_regime_tagged.csv", index=False)
    regime_summary.to_csv(OUTPUT_DIR / "day_regime_summary.csv", index=False)
    regime_detail.to_csv(OUTPUT_DIR / "day_regime_detail.csv", index=False)
    write_summary(df, regime_summary)

    print(f"Trade count: {len(df)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
