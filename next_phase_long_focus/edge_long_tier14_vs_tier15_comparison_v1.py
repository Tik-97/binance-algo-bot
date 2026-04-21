from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "edge_long_tier14_vs_tier15_comparison_v1"


STRATEGIES = [
    {
        "name": "tier14",
        "title": "Tier 1.4 core + satellite1 + satellite2",
        "summary_path": RESULTS_DIR / "edge_long_tier14_core_plus_satellite2_v1" / "summary.txt",
        "trades_path": RESULTS_DIR / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv",
        "route_path": RESULTS_DIR / "edge_long_tier14_core_plus_satellite2_v1" / "route_summary.csv",
    },
    {
        "name": "tier15",
        "title": "Tier 1.5 core + satellite1 + satellite2 + satellite3",
        "summary_path": RESULTS_DIR / "edge_long_tier15_core_plus_satellite3_v1" / "summary.txt",
        "trades_path": RESULTS_DIR / "edge_long_tier15_core_plus_satellite3_v1" / "trades.csv",
        "route_path": RESULTS_DIR / "edge_long_tier15_core_plus_satellite3_v1" / "route_summary.csv",
    },
]


SUMMARY_FLOAT_FIELDS = {
    "trade_count",
    "win_rate_pct",
    "profit_factor",
    "max_drawdown_pct",
    "expectancy_usd",
    "ending_balance",
    "context_without_trigger_count",
    "rejected_entry_count",
    "expired_limit_count",
}


def parse_summary(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() == "n/a":
            data[key] = None
            continue
        if value.lower() == "inf":
            data[key] = float("inf")
            continue
        if key in SUMMARY_FLOAT_FIELDS:
            try:
                data[key] = float(value)
                continue
            except ValueError:
                pass
        data[key] = value
    return data


def assign_periods(trades_df: pd.DataFrame) -> pd.DataFrame:
    frame = trades_df.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["year_month"] = frame["entry_time"].dt.strftime("%Y-%m")
    min_ts = frame["entry_time"].min()
    max_ts = frame["entry_time"].max()
    midpoint = min_ts + (max_ts - min_ts) / 2
    q1 = min_ts + (max_ts - min_ts) / 4
    q2 = min_ts + (max_ts - min_ts) / 2
    q3 = min_ts + (max_ts - min_ts) * 3 / 4

    frame["half"] = frame["entry_time"].apply(lambda ts: "first_half" if ts <= midpoint else "second_half")

    def classify_quarter(ts: pd.Timestamp) -> str:
        if ts <= q1:
            return "q1"
        if ts <= q2:
            return "q2"
        if ts <= q3:
            return "q3"
        return "q4"

    frame["quarter"] = frame["entry_time"].apply(classify_quarter)
    return frame


def grouped_summary(df: pd.DataFrame, group_col: str, strategy_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, part in df.groupby(group_col, dropna=False):
        trade_count = int(len(part))
        net_pnl = float(part["pnl_usd"].sum())
        win_rate = float((part["pnl_usd"] > 0).mean() * 100.0) if trade_count else 0.0
        rows.append(
            {
                "strategy": strategy_name,
                group_col: value,
                "trade_count": trade_count,
                "net_pnl_usd": round(net_pnl, 4),
                "win_rate_pct": round(win_rate, 2),
                "expectancy_usd": round(net_pnl / trade_count, 4) if trade_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_strategy(config: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = parse_summary(config["summary_path"])
    trades = pd.read_csv(config["trades_path"])
    trades = assign_periods(trades)
    route_summary = pd.read_csv(config["route_path"])

    half_summary = grouped_summary(trades, "half", config["name"])
    quarter_summary = grouped_summary(trades, "quarter", config["name"])
    month_summary = grouped_summary(trades, "year_month", config["name"])

    top_route = route_summary.sort_values(["net_pnl_usd", "trade_count"], ascending=[False, False]).iloc[0]
    top_route_name = str(top_route["route"])
    top_route_pnl = float(top_route["net_pnl_usd"])
    total_pnl = float(trades["pnl_usd"].sum())
    top_route_share_pct = (top_route_pnl / total_pnl * 100.0) if total_pnl else 0.0

    first_half_pnl = float(half_summary.loc[half_summary["half"] == "first_half", "net_pnl_usd"].sum())
    second_half_pnl = float(half_summary.loc[half_summary["half"] == "second_half", "net_pnl_usd"].sum())
    positive_quarters = int((quarter_summary["net_pnl_usd"] > 0).sum())
    nonempty_quarters = int(len(quarter_summary))

    comparison_row = {
        "strategy": config["name"],
        "title": config["title"],
        "trade_count": int(summary.get("trade_count") or 0),
        "ending_balance": float(summary.get("ending_balance") or 0.0),
        "win_rate_pct": float(summary.get("win_rate_pct") or 0.0),
        "max_drawdown_pct": float(summary.get("max_drawdown_pct") or 0.0),
        "expectancy_usd": float(summary.get("expectancy_usd") or 0.0),
        "net_pnl_usd": round(total_pnl, 4),
        "first_half_trade_count": int(half_summary.loc[half_summary["half"] == "first_half", "trade_count"].sum()),
        "first_half_net_pnl_usd": round(first_half_pnl, 4),
        "second_half_trade_count": int(half_summary.loc[half_summary["half"] == "second_half", "trade_count"].sum()),
        "second_half_net_pnl_usd": round(second_half_pnl, 4),
        "positive_quarters": positive_quarters,
        "nonempty_quarters": nonempty_quarters,
        "top_route": top_route_name,
        "top_route_net_pnl_usd": round(top_route_pnl, 4),
        "top_route_share_pct": round(top_route_share_pct, 2),
        "route_count": int(len(route_summary)),
        "latest_trade_time": str(trades["entry_time"].max()),
    }
    return comparison_row, half_summary, quarter_summary, month_summary, route_summary.assign(strategy=config["name"])


def recommendation_text(comparison_df: pd.DataFrame) -> list[str]:
    tier14 = comparison_df.loc[comparison_df["strategy"] == "tier14"].iloc[0]
    tier15 = comparison_df.loc[comparison_df["strategy"] == "tier15"].iloc[0]

    lines = []
    if tier15["trade_count"] > tier14["trade_count"] and tier15["ending_balance"] > tier14["ending_balance"]:
        lines.append(
            f"- Tier 1.5 is wider and stronger on raw history: {int(tier15['trade_count'])} trades vs {int(tier14['trade_count'])}, "
            f"ending_balance {tier15['ending_balance']:.2f} vs {tier14['ending_balance']:.2f}."
        )
    if tier15["route_count"] > tier14["route_count"]:
        lines.append(
            f"- Tier 1.5 adds one extra route (`{tier15['top_route']}` is still not dominant at {tier15['top_route_share_pct']:.2f}% share for the top route overall)."
        )
    lines.append(
        f"- Tier 1.4 is cleaner structurally: only {int(tier14['route_count'])} routes, no micro-satellite, and still {int(tier14['trade_count'])} trades at 100% win rate."
    )
    lines.append(
        f"- Tier 1.5 is still acceptable, but its extra edge comes from a tiny late micro-route; treat it as the more aggressive version, not automatically the safer baseline."
    )
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows: list[dict[str, object]] = []
    half_frames: list[pd.DataFrame] = []
    quarter_frames: list[pd.DataFrame] = []
    month_frames: list[pd.DataFrame] = []
    route_frames: list[pd.DataFrame] = []

    for config in STRATEGIES:
        row, half_df, quarter_df, month_df, route_df = summarize_strategy(config)
        comparison_rows.append(row)
        half_frames.append(half_df)
        quarter_frames.append(quarter_df)
        month_frames.append(month_df)
        route_frames.append(route_df)

    comparison_df = pd.DataFrame(comparison_rows)
    half_comparison = pd.concat(half_frames, ignore_index=True)
    quarter_comparison = pd.concat(quarter_frames, ignore_index=True)
    month_comparison = pd.concat(month_frames, ignore_index=True)
    route_comparison = pd.concat(route_frames, ignore_index=True)

    recommendation_lines = recommendation_text(comparison_df)

    lines = [
        "EDGE LONG TIER14 VS TIER15 COMPARISON V1",
        "========================================",
        "",
        "strategies_compared:",
        "- Tier 1.4 = core + satellite1 + satellite2",
        "- Tier 1.5 = Tier 1.4 + micro satellite3",
        "",
        "headline:",
        f"- Tier 1.4: {int(comparison_df.loc[comparison_df['strategy']=='tier14', 'trade_count'].iloc[0])} trades, "
        f"ending_balance = {comparison_df.loc[comparison_df['strategy']=='tier14', 'ending_balance'].iloc[0]:.2f}",
        f"- Tier 1.5: {int(comparison_df.loc[comparison_df['strategy']=='tier15', 'trade_count'].iloc[0])} trades, "
        f"ending_balance = {comparison_df.loc[comparison_df['strategy']=='tier15', 'ending_balance'].iloc[0]:.2f}",
        "",
        "recommendation:",
        *recommendation_lines,
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparison_df.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
    half_comparison.to_csv(OUTPUT_DIR / "half_comparison.csv", index=False)
    quarter_comparison.to_csv(OUTPUT_DIR / "quarter_comparison.csv", index=False)
    month_comparison.to_csv(OUTPUT_DIR / "month_comparison.csv", index=False)
    route_comparison.to_csv(OUTPUT_DIR / "route_comparison.csv", index=False)

    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
