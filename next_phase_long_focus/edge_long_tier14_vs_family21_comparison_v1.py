from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_tier14_vs_family21_comparison_v1"

TIER14_PATH = BASE_DIR / "results" / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv"
FAMILY21_PATH = BASE_DIR / "results" / "edge_long_family21_seed_plus_satellites_v1" / "trades.csv"


def load_trades(path: Path, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df["strategy"] = strategy
    df["trade_key"] = (
        df["side"].astype(str)
        + "|"
        + df["entry_time"].astype(str)
        + "|"
        + df["exit_time"].astype(str)
    )
    return df


def summarize(df: pd.DataFrame, label: str) -> dict[str, object]:
    if df.empty:
        return {
            "label": label,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "net_pnl_usd": 0.0,
            "ending_balance": None,
            "expectancy_usd": None,
            "median_hold_minutes": None,
        }

    return {
        "label": label,
        "trade_count": int(len(df)),
        "win_rate_pct": round(float((df["pnl_usd"] > 0).mean() * 100.0), 2),
        "net_pnl_usd": round(float(df["pnl_usd"].sum()), 4),
        "ending_balance": round(float(df["balance_after"].iloc[-1]), 2),
        "expectancy_usd": round(float(df["pnl_usd"].mean()), 4),
        "median_hold_minutes": round(float(df["hold_minutes"].median()), 2),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tier14 = load_trades(TIER14_PATH, "tier14")
    family21 = load_trades(FAMILY21_PATH, "family21")

    full_end = max(tier14["exit_time"].max(), family21["exit_time"].max())
    full_start = min(tier14["entry_time"].min(), family21["entry_time"].min())
    recent_start = full_end - pd.Timedelta(days=365)

    exact_overlap_keys = sorted(set(tier14["trade_key"]).intersection(set(family21["trade_key"])))
    tier14_overlap = tier14[tier14["trade_key"].isin(exact_overlap_keys)].copy()
    family21_overlap = family21[family21["trade_key"].isin(exact_overlap_keys)].copy()
    overlap_pairs = tier14_overlap.merge(
        family21_overlap,
        on="trade_key",
        suffixes=("_tier14", "_family21"),
        how="inner",
    )

    tier14_unique = tier14[~tier14["trade_key"].isin(exact_overlap_keys)].copy()
    family21_unique = family21[~family21["trade_key"].isin(exact_overlap_keys)].copy()

    recent_tier14 = tier14[tier14["entry_time"] >= recent_start].copy()
    recent_family21 = family21[family21["entry_time"] >= recent_start].copy()
    recent_tier14_unique = tier14_unique[tier14_unique["entry_time"] >= recent_start].copy()
    recent_family21_unique = family21_unique[family21_unique["entry_time"] >= recent_start].copy()

    full_summary = pd.DataFrame(
        [
            summarize(tier14, "tier14_full"),
            summarize(family21, "family21_full"),
            summarize(tier14_unique, "tier14_unique_full"),
            summarize(family21_unique, "family21_unique_full"),
            summarize(tier14_overlap, "overlap_full"),
        ]
    )
    recent_summary = pd.DataFrame(
        [
            summarize(recent_tier14, "tier14_recent_1y"),
            summarize(recent_family21, "family21_recent_1y"),
            summarize(recent_tier14_unique, "tier14_unique_recent_1y"),
            summarize(recent_family21_unique, "family21_unique_recent_1y"),
        ]
    )

    route_compare = pd.concat(
        [
            tier14.assign(route_group=tier14["route"])[["strategy", "route_group", "pnl_usd", "hold_minutes"]],
            family21.assign(route_group=family21["route"])[["strategy", "route_group", "pnl_usd", "hold_minutes"]],
        ],
        ignore_index=True,
    )
    route_summary = (
        route_compare.groupby(["strategy", "route_group"], dropna=False)
        .agg(
            trade_count=("route_group", "size"),
            net_pnl_usd=("pnl_usd", "sum"),
            avg_pnl_usd=("pnl_usd", "mean"),
            median_hold_minutes=("hold_minutes", "median"),
        )
        .reset_index()
        .sort_values(["strategy", "net_pnl_usd"], ascending=[True, False])
    )

    month_compare = pd.concat(
        [
            tier14.assign(year_month=tier14["entry_time"].dt.strftime("%Y-%m"))[["strategy", "year_month", "pnl_usd"]],
            family21.assign(year_month=family21["entry_time"].dt.strftime("%Y-%m"))[["strategy", "year_month", "pnl_usd"]],
        ],
        ignore_index=True,
    )
    month_summary = (
        month_compare.groupby(["strategy", "year_month"], dropna=False)
        .agg(trade_count=("year_month", "size"), net_pnl_usd=("pnl_usd", "sum"))
        .reset_index()
        .sort_values(["year_month", "strategy"])
    )

    overlap_pairs.to_csv(OUTPUT_DIR / "exact_overlap_pairs.csv", index=False)
    full_summary.to_csv(OUTPUT_DIR / "full_summary.csv", index=False)
    recent_summary.to_csv(OUTPUT_DIR / "recent_1y_summary.csv", index=False)
    route_summary.to_csv(OUTPUT_DIR / "route_summary.csv", index=False)
    month_summary.to_csv(OUTPUT_DIR / "month_summary.csv", index=False)
    tier14_unique.to_csv(OUTPUT_DIR / "tier14_unique_trades.csv", index=False)
    family21_unique.to_csv(OUTPUT_DIR / "family21_unique_trades.csv", index=False)

    lines = [
        "EDGE LONG TIER1.4 VS FAMILY2.1 COMPARISON V1",
        "=============================================",
        "",
        "goal = compare the current main long baseline against the expanded Family 2 branch",
        f"full_window_start = {full_start}",
        f"full_window_end = {full_end}",
        f"recent_1y_start = {recent_start}",
        "",
        "headline full-history comparison:",
        f"- Tier 1.4: trades={int(len(tier14))}, net_pnl={tier14['pnl_usd'].sum():.2f}, ending_balance={tier14['balance_after'].iloc[-1]:.2f}",
        f"- Family 2.1: trades={int(len(family21))}, net_pnl={family21['pnl_usd'].sum():.2f}, ending_balance={family21['balance_after'].iloc[-1]:.2f}",
        "",
        "headline recent-1y comparison:",
        f"- Tier 1.4 recent 1y: trades={int(len(recent_tier14))}, net_pnl={recent_tier14['pnl_usd'].sum():.2f}",
        f"- Family 2.1 recent 1y: trades={int(len(recent_family21))}, net_pnl={recent_family21['pnl_usd'].sum():.2f}",
        "",
        "overlap truth:",
        f"- exact same-trade overlaps = {len(exact_overlap_keys)}",
        f"- Tier 1.4 unique trades = {len(tier14_unique)}",
        f"- Family 2.1 unique trades = {len(family21_unique)}",
        "",
        "honest read:",
    ]

    if family21["pnl_usd"].sum() > tier14["pnl_usd"].sum():
        lines.append("- Family 2.1 is stronger on raw full-history PnL.")
    else:
        lines.append("- Tier 1.4 is stronger on raw full-history PnL.")

    if len(exact_overlap_keys) > 0:
        lines.append("- The branches are not independent; they share a meaningful set of exact same long trades.")
    else:
        lines.append("- The branches are fully distinct in exact trade timing.")

    if recent_family21["pnl_usd"].sum() > recent_tier14["pnl_usd"].sum():
        lines.append("- Family 2.1 is stronger in the most recent one-year slice.")
    else:
        lines.append("- Tier 1.4 is stronger in the most recent one-year slice.")

    lines.extend(
        [
            "- Practical interpretation should focus on unique trade contribution, not only raw standalone curves.",
            "",
            "decision framing:",
            "- Tier 1.4 = cleaner original long baseline family",
            "- Family 2.1 = newer flush-reversal family with stronger full-history curve but overlap with Tier 1.4",
        ]
    )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
