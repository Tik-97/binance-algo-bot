from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_tier11_core_v1"

MIN_CLIMAX_DROP_5M_PCT = 0.85
MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT = 1.30


def make_tier11_core_context_gate():
    tier1_gate = tier1.make_tier1_context_gate()

    def tier11_context(row: pd.Series) -> bool:
        if not tier1_gate(row):
            return False
        climax_drop_5m_pct = float(row.get("drop_5m_pct", 0.0))
        context_stretch_below_ema20_5m_pct = float(row.get("stretch_below_ema20_5m_pct", 999.0))
        return (
            climax_drop_5m_pct >= MIN_CLIMAX_DROP_5M_PCT
            and context_stretch_below_ema20_5m_pct <= MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT
        )

    return tier11_context


def save_outputs(trades_df: pd.DataFrame, summary: dict[str, object], all_candles: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    lines = [
        "EDGE LONG TIER1.1 CORE V1",
        "==========================",
        "",
        "idea = isolate a tiny long core inside the bear-flush reversal baseline",
        "router = Tier1 bear-like high_vol long + hard flush + not-too-stretched context",
        "source_engine = wickMick_long.py",
        "",
        f"filters = climax_drop_5m_pct >= {MIN_CLIMAX_DROP_5M_PCT:.2f}",
        f"filters = context_stretch_below_ema20_5m_pct <= {MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT:.2f}",
        "",
        f"bt_start = {summary['bt_start']}",
        f"bt_end = {summary['bt_end']}",
        f"trade_count = {summary['trade_count']}",
        f"win_rate_pct = {summary['win_rate_pct']:.2f}" if not pd.isna(summary["win_rate_pct"]) else "win_rate_pct = n/a",
        f"profit_factor = {summary['profit_factor']:.3f}" if not pd.isna(summary["profit_factor"]) else "profit_factor = n/a",
        f"max_drawdown_pct = {summary['max_drawdown_pct']:.2f}" if not pd.isna(summary["max_drawdown_pct"]) else "max_drawdown_pct = n/a",
        f"expectancy_usd = {summary['expectancy_usd']:.4f}" if not pd.isna(summary["expectancy_usd"]) else "expectancy_usd = n/a",
        f"ending_balance = {summary['ending_balance']:.2f}",
        f"context_without_trigger_count = {summary['context_without_trigger_count']}",
        f"rejected_entry_count = {summary['rejected_entry_count']}",
        f"expired_limit_count = {summary['expired_limit_count']}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if trades_df.empty:
        return

    candles_with_regime = all_candles.copy()
    if "trend_regime" not in candles_with_regime.columns:
        candles_with_regime = tier1.merge_daily_regime(candles_with_regime)

    enriched = trades_df.copy()
    enriched["trade_date"] = pd.to_datetime(enriched["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    enriched = enriched.merge(
        candles_with_regime[
            [
                "trade_date",
                "trend_regime",
                "vol_regime",
                "ret_30d_pct",
                "ret_60d_pct",
                "close_vs_sma60_pct",
            ]
        ].drop_duplicates("trade_date"),
        on="trade_date",
        how="left",
    )
    tier1.grouped_summary(enriched, "trend_regime").to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    tier1.grouped_summary(enriched, "vol_regime").to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    tier1.grouped_summary(enriched, "entry_mode").to_csv(OUTPUT_DIR / "entry_mode_summary.csv", index=False)
    enriched.to_csv(OUTPUT_DIR / "trade_regime_detail.csv", index=False)


def main() -> None:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)

    original_is_base_context = base.is_base_context
    try:
        base.is_base_context = make_tier11_core_context_gate()
        bt_start = featured["open_time"].min()
        bt_end = featured["open_time"].max()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_is_base_context

    save_outputs(trades_df, summary, featured)
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
