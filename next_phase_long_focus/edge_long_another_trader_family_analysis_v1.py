from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_another_trader_family_analysis_v1"
CONTEXT_PATH = REPO_ROOT / "Another Trader" / "analysis_output" / "market_context_at_entry.csv"
TRADES_PATH = REPO_ROOT / "Another Trader" / "analysis_output" / "cleaned_trades.csv"


RULE_DEFINITIONS: list[tuple[str, str]] = [
    ("near_low", "structure_regime == near_low"),
    ("breakout_down", "structure_regime == breakout_down"),
    ("reversal_like", "context_style_label == reversal_like"),
    ("breakout_continuation", "context_style_label == breakout_continuation"),
    ("bearish_15m", "ema_alignment_15m == bearish"),
    ("bearish_1h", "ema_alignment_1h == bearish"),
    ("dump60_le_-0.5", "return_pre_60m_pct <= -0.5"),
    ("dump60_le_-1.0", "return_pre_60m_pct <= -1.0"),
    ("dist15_le_-0.5", "distance_to_ema20_pct_15m <= -0.5"),
    ("dist1h_le_-1.0", "distance_to_ema20_pct_1h <= -1.0"),
    ("rsi15_le_35", "rsi14_15m <= 35"),
    ("rv5_ge_1.0", "relative_volume_5m >= 1.0"),
]


def grouped_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, part in df.groupby(group_col, dropna=False):
        rows.append(
            {
                group_col: value,
                "trade_count": int(len(part)),
                "win_count": int((part["trade_outcome"] == "win").sum()),
                "net_pnl_usd": round(float(part["closingPnl"].sum()), 4),
                "median_holding_minutes": round(float(part["holding_minutes"].median()), 2),
                "median_return_pct": round(float(part["return_pct_approx"].median()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["trade_count", "net_pnl_usd"], ascending=[False, False]).reset_index(drop=True)


def load_reference_longs() -> pd.DataFrame:
    trades = pd.read_csv(TRADES_PATH)
    context = pd.read_csv(CONTEXT_PATH)

    eth_trades = trades[(trades["symbol"] == "ETHUSDT") & (trades["side"] == "Long")].copy()
    if eth_trades.empty:
        raise ValueError("No ETHUSDT long trades found in cleaned_trades.csv")

    merged = eth_trades.merge(
        context[
            [
                "id",
                "context_style_label",
                "entry_strength_group",
                "structure_regime",
                "ema_alignment_15m",
                "ema_alignment_1h",
                "rsi14_15m",
                "rsi14_5m",
                "distance_to_ema20_pct_15m",
                "distance_to_ema20_pct_1h",
                "return_pre_60m_pct",
                "relative_volume_5m",
            ]
        ],
        on="id",
        how="left",
    )
    merged["opened_dt_utc"] = pd.to_datetime(merged["opened_dt_utc"], utc=True)
    merged["trade_date"] = merged["opened_dt_utc"].dt.strftime("%Y-%m-%d")
    return merged


def attach_daily_regime(reference_df: pd.DataFrame) -> pd.DataFrame:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    daily = featured[
        ["trade_date", "trend_regime", "vol_regime", "ret_30d_pct", "ret_60d_pct", "close_vs_sma60_pct"]
    ].drop_duplicates("trade_date")
    return reference_df.merge(daily, on="trade_date", how="left")


def build_rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "near_low": df["structure_regime"].eq("near_low"),
        "breakout_down": df["structure_regime"].eq("breakout_down"),
        "reversal_like": df["context_style_label"].eq("reversal_like"),
        "breakout_continuation": df["context_style_label"].eq("breakout_continuation"),
        "bearish_15m": df["ema_alignment_15m"].eq("bearish"),
        "bearish_1h": df["ema_alignment_1h"].eq("bearish"),
        "dump60_le_-0.5": df["return_pre_60m_pct"] <= -0.5,
        "dump60_le_-1.0": df["return_pre_60m_pct"] <= -1.0,
        "dist15_le_-0.5": df["distance_to_ema20_pct_15m"] <= -0.5,
        "dist1h_le_-1.0": df["distance_to_ema20_pct_1h"] <= -1.0,
        "rsi15_le_35": df["rsi14_15m"] <= 35.0,
        "rv5_ge_1.0": df["relative_volume_5m"] >= 1.0,
    }


def search_family_combos(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rule_names = list(masks.keys())
    for rule_count in (4, 5):
        for combo in combinations(rule_names, rule_count):
            mask = pd.Series(True, index=df.index)
            for rule in combo:
                mask &= masks[rule]
            count = int(mask.sum())
            if count < 6:
                continue
            part = df.loc[mask]
            rows.append(
                {
                    "rule_count": rule_count,
                    "rule_combo": " + ".join(combo),
                    "trade_count": count,
                    "share_of_reference_pct": round(count / len(df) * 100.0, 2),
                    "net_pnl_usd": round(float(part["closingPnl"].sum()), 4),
                    "median_holding_minutes": round(float(part["holding_minutes"].median()), 2),
                    "median_return_pct": round(float(part["return_pct_approx"].median()), 4),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "rule_count",
                "rule_combo",
                "trade_count",
                "share_of_reference_pct",
                "net_pnl_usd",
                "median_holding_minutes",
                "median_return_pct",
            ]
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["trade_count", "rule_count", "net_pnl_usd"],
        ascending=[False, True, False],
    )
    return ranked.drop_duplicates("rule_combo").reset_index(drop=True)


def add_named_families(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    enriched = df.copy()
    enriched["family_a_flush_reversal"] = (
        masks["near_low"] & masks["reversal_like"] & masks["bearish_15m"] & masks["bearish_1h"] & masks["dump60_le_-0.5"]
    )
    enriched["family_b_deep_flush_reversal"] = (
        masks["near_low"] & masks["reversal_like"] & masks["bearish_15m"] & masks["dist15_le_-0.5"] & masks["dist1h_le_-1.0"]
    )
    enriched["family_c_breakout_down_reclaim"] = (
        masks["breakout_down"] & masks["bearish_15m"] & masks["bearish_1h"] & masks["dump60_le_-1.0"]
    )
    return enriched


def named_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    family_defs = {
        "family_a_flush_reversal": "near_low + reversal_like + bearish_15m + bearish_1h + dump60<=-0.5",
        "family_b_deep_flush_reversal": "near_low + reversal_like + bearish_15m + dist15<=-0.5 + dist1h<=-1.0",
        "family_c_breakout_down_reclaim": "breakout_down + bearish_15m + bearish_1h + dump60<=-1.0",
    }
    for col, definition in family_defs.items():
        part = df.loc[df[col]].copy()
        rows.append(
            {
                "family_name": col,
                "definition": definition,
                "trade_count": int(len(part)),
                "share_of_reference_pct": round(len(part) / len(df) * 100.0, 2),
                "net_pnl_usd": round(float(part["closingPnl"].sum()), 4),
                "median_holding_minutes": round(float(part["holding_minutes"].median()), 2) if not part.empty else None,
                "median_return_pct": round(float(part["return_pct_approx"].median()), 4) if not part.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["trade_count", "net_pnl_usd"], ascending=[False, False]).reset_index(drop=True)


def write_summary(
    reference_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    context_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    family_df: pd.DataFrame,
    combo_df: pd.DataFrame,
) -> None:
    top_combo_lines: list[str] = []
    for row in combo_df.head(3).to_dict(orient="records"):
        top_combo_lines.extend(
            [
                f"- {row['rule_combo']}",
                f"  trades = {int(row['trade_count'])}, share = {row['share_of_reference_pct']:.2f}%, median_hold = {row['median_holding_minutes']:.2f}m",
            ]
        )

    top_family_lines: list[str] = []
    for row in family_df.to_dict(orient="records"):
        top_family_lines.extend(
            [
                f"- {row['family_name']}: {row['definition']}",
                f"  trades = {int(row['trade_count'])}, share = {row['share_of_reference_pct']:.2f}%, net_pnl = {row['net_pnl_usd']:.2f}",
            ]
        )

    lines = [
        "EDGE LONG ANOTHER TRADER FAMILY ANALYSIS V1",
        "===========================================",
        "",
        "source = Another Trader real ETHUSDT long trades",
        "goal = identify the next clean long-family direction outside the exhausted Tier 1.4 residual",
        "",
        f"reference_trade_count = {len(reference_df)}",
        f"reference_win_rate_pct = {(reference_df['trade_outcome'] == 'win').mean() * 100.0:.2f}",
        f"reference_net_pnl_usd = {reference_df['closingPnl'].sum():.4f}",
        "",
        "headline read:",
        "- Another Trader ETH longs are not classic bull continuation longs",
        "- the dominant shape is still a bearish-flush reversal family",
        "- strongest repeated cluster is near-low + reversal-like + bearish EMA alignment + negative 60m momentum",
        "",
        "regime concentration:",
        f"- top trend regime = {trend_df.iloc[0]['trend_regime']} ({int(trend_df.iloc[0]['trade_count'])} trades)" if not trend_df.empty else "- n/a",
        f"- top vol regime = {vol_df.iloc[0]['vol_regime']} ({int(vol_df.iloc[0]['trade_count'])} trades)" if not vol_df.empty else "- n/a",
        "",
        "context concentration:",
        f"- top context_style = {context_df.iloc[0]['context_style_label']} ({int(context_df.iloc[0]['trade_count'])} trades)" if not context_df.empty else "- n/a",
        f"- top structure_regime = {structure_df.iloc[0]['structure_regime']} ({int(structure_df.iloc[0]['trade_count'])} trades)" if not structure_df.empty else "- n/a",
        "",
        "named family pockets:",
        *top_family_lines,
        "",
        "top simple rule combos:",
        *top_combo_lines,
        "",
        "next practical move:",
        "- keep Tier 1.4 as the current long baseline",
        "- build the next long-family search around the flush-reversal family from these Another Trader longs",
        "- do not search satellite4 inside Tier 1.4 residual anymore",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_df = attach_daily_regime(load_reference_longs())
    masks = build_rule_masks(reference_df)
    reference_df = add_named_families(reference_df, masks)

    trend_df = grouped_summary(reference_df, "trend_regime")
    vol_df = grouped_summary(reference_df, "vol_regime")
    hour_df = grouped_summary(reference_df, "open_hour_utc")
    context_df = grouped_summary(reference_df, "context_style_label")
    structure_df = grouped_summary(reference_df, "structure_regime")
    strength_df = grouped_summary(reference_df, "entry_strength_group")
    align15_df = grouped_summary(reference_df, "ema_alignment_15m")
    align1h_df = grouped_summary(reference_df, "ema_alignment_1h")
    family_df = named_family_summary(reference_df)
    combo_df = search_family_combos(reference_df, masks)

    reference_df.to_csv(OUTPUT_DIR / "reference_trades_enriched.csv", index=False)
    trend_df.to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    vol_df.to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    hour_df.to_csv(OUTPUT_DIR / "entry_hour_summary.csv", index=False)
    context_df.to_csv(OUTPUT_DIR / "context_style_summary.csv", index=False)
    structure_df.to_csv(OUTPUT_DIR / "structure_regime_summary.csv", index=False)
    strength_df.to_csv(OUTPUT_DIR / "entry_strength_summary.csv", index=False)
    align15_df.to_csv(OUTPUT_DIR / "ema_alignment_15m_summary.csv", index=False)
    align1h_df.to_csv(OUTPUT_DIR / "ema_alignment_1h_summary.csv", index=False)
    family_df.to_csv(OUTPUT_DIR / "named_family_summary.csv", index=False)
    combo_df.to_csv(OUTPUT_DIR / "top_rule_combos.csv", index=False)

    write_summary(reference_df, trend_df, vol_df, hour_df, context_df, structure_df, family_df, combo_df)
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
