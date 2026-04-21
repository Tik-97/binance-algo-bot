from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "edge_long_tier14_satellite4_search_v1"

TIER1_PATH = RESULTS_DIR / "edge_long_tier1_bear_highvol_baseline_v1" / "trades.csv"
TIER14_PATH = RESULTS_DIR / "edge_long_tier14_core_plus_satellite2_v1" / "trades.csv"


FEATURES = {
    "entry_mode_aggr": {"ops": ["=="], "vals": [0, 1]},
    "confirm_range_pct": {"ops": ["<=", ">="], "vals": [0.30, 0.35, 0.40, 0.45, 0.50]},
    "rel_volume_1m": {"ops": ["<=", ">="], "vals": [2.0, 2.5, 3.0, 3.5, 4.0]},
    "trigger_body_ratio": {"ops": ["<=", ">="], "vals": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]},
    "trigger_lower_wick_ratio": {"ops": ["<=", ">="], "vals": [0.40, 0.45, 0.50, 0.55]},
    "context_dump_60m_pct": {"ops": ["<=", ">="], "vals": [1.0, 1.2, 1.4, 1.6, 2.0]},
    "distance_below_ema7_1m_pct": {"ops": ["<=", ">="], "vals": [0.35, 0.40, 0.45, 0.50, 0.60]},
}


def series_for(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "entry_mode_aggr":
        return (df["entry_mode"] == "aggressive_market").astype(int)
    return df[col]


def apply_rules(df: pd.DataFrame, rules: tuple[tuple[str, str, float], ...]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, op, val in rules:
        series = series_for(df, col)
        if op == "<=":
            mask &= series <= val
        elif op == ">=":
            mask &= series >= val
        else:
            mask &= series == val
    return mask


def format_rule(rule: tuple[str, str, float]) -> str:
    col, op, val = rule
    if col == "entry_mode_aggr":
        return "entry_mode == aggressive_market" if val == 1 else "entry_mode == shallow_pullback_limit"
    return f"{col} {op} {val:.2f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = pd.read_csv(TIER1_PATH)
    tier14 = pd.read_csv(TIER14_PATH)
    residual = tier1.loc[~tier1["entry_time"].isin(set(tier14["entry_time"]))].copy()
    residual.to_csv(OUTPUT_DIR / "residual_trades.csv", index=False)

    wins = int((residual["pnl_usd"] > 0).sum())
    losses = int((residual["pnl_usd"] < 0).sum())
    net_pnl = float(residual["pnl_usd"].sum())

    candidates: list[dict[str, object]] = []
    cols = list(FEATURES)
    for rule_count in [1, 2, 3]:
        for combo in combinations(cols, rule_count):
            option_lists = []
            for col in combo:
                option_lists.append([(col, op, val) for op in FEATURES[col]["ops"] for val in FEATURES[col]["vals"]])
            for rules in product(*option_lists):
                part = residual.loc[apply_rules(residual, rules)]
                n = len(part)
                if n < 3 or n > 5:
                    continue
                pnl = float(part["pnl_usd"].sum())
                if pnl <= 0:
                    continue
                win_count = int((part["pnl_usd"] > 0).sum())
                win_rate = win_count / n * 100.0
                gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
                gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                if win_rate >= 66.67 and profit_factor >= 2.0:
                    candidates.append(
                        {
                            "trade_count": n,
                            "net_pnl_usd": round(pnl, 4),
                            "win_rate_pct": round(win_rate, 2),
                            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
                            "rules": " | ".join(format_rule(rule) for rule in rules),
                            "entry_times": " | ".join(part["entry_time"].tolist()),
                        }
                    )

    candidate_df = pd.DataFrame(candidates).sort_values(
        ["win_rate_pct", "profit_factor", "trade_count", "net_pnl_usd"],
        ascending=[False, False, False, False],
    ) if candidates else pd.DataFrame(columns=["trade_count", "net_pnl_usd", "win_rate_pct", "profit_factor", "rules", "entry_times"])
    candidate_df.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)

    lines = [
        "EDGE LONG TIER14 SATELLITE4 SEARCH V1",
        "=====================================",
        "",
        "search_scope = residual after Tier 1.4 (core + satellite1 + satellite2)",
        "search_style = constrained live-safe filter combinations only",
        "candidate_requirements = 3-5 trades, positive net pnl, win_rate >= 66.67%, profit_factor >= 2.0",
        "",
        f"residual_trade_count = {len(residual)}",
        f"residual_win_count = {wins}",
        f"residual_loss_count = {losses}",
        f"residual_net_pnl_usd = {net_pnl:.4f}",
        f"qualified_candidate_count = {len(candidate_df)}",
        "",
    ]

    if candidate_df.empty:
        lines.extend(
            [
                "result = no qualified satellite4 candidate found",
                "",
                "honest_read:",
                "- after Tier 1.4, the residual set is already mostly exhausted",
                "- only 2 residual winners remained",
                "- no 3+ trade live-safe pocket met the minimum quality threshold",
                "- this suggests the next clean step should not be satellite4 from Tier 1.4 residual",
            ]
        )
    else:
        top = candidate_df.iloc[0]
        lines.extend(
            [
                "result = at least one qualified candidate exists",
                f"top_candidate_trade_count = {int(top['trade_count'])}",
                f"top_candidate_net_pnl_usd = {float(top['net_pnl_usd']):.4f}",
                f"top_candidate_win_rate_pct = {float(top['win_rate_pct']):.2f}",
                f"top_candidate_profit_factor = {top['profit_factor']}",
                f"top_candidate_rules = {top['rules']}",
            ]
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
