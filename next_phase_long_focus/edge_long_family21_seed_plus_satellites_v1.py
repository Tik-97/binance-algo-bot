from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import next_phase_long_focus.edge_long_family2_flush_reversal_seed_v1 as family2_seed
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_family21_seed_plus_satellites_v1"


def is_family2_base_context(row: pd.Series) -> bool:
    return (
        row.get("trend_regime") == "strong_bear"
        and row.get("vol_regime") == "high_vol"
        and row.get("structure_regime2") == "near_low"
        and row.get("context_style_label2") == "reversal_like"
        and row.get("ema_alignment_15m") == "bearish"
        and row.get("ema_alignment_1h") == "bearish"
        and pd.notna(row.get("return_pre_60m_pct2"))
        and float(row["return_pre_60m_pct2"]) <= -0.5
    )


def classify_signal_route(row: pd.Series, trigger_bar: pd.Series, confirm_bar: pd.Series) -> str | None:
    if (
        float(row["stretch_below_ema20_5m_pct"]) <= family2_seed.MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT
        and family2_seed.MIN_TRIGGER_BODY_RATIO <= float(trigger_bar["body_ratio"]) <= family2_seed.MAX_TRIGGER_BODY_RATIO
        and float(trigger_bar["relative_volume_1m"]) >= family2_seed.MIN_RELATIVE_VOLUME_1M
    ):
        return "family2_seed"

    if (
        float(trigger_bar["body_ratio"]) >= 0.45
        and float(confirm_bar["range_pct"]) >= 0.45
        and float(row["dump_60m_pct"]) <= 1.05
    ):
        return "family2_satellite_highconfirm"

    if (
        0.29 <= float(trigger_bar["body_ratio"]) <= 0.35
        and float(confirm_bar["range_pct"]) <= 0.45
        and float(trigger_bar["relative_volume_1m"]) >= 1.20
    ):
        return "family2_satellite_midbody"

    return None


def make_route_aware_find_trigger():
    def route_aware_find_trigger(df: pd.DataFrame, i: int) -> tuple[int | None, str | None]:
        candidates: list[tuple[int, str]] = [(i, "current")]
        if base.USE_CURRENT_OR_PREVIOUS_CANDLE:
            candidates.append((i - 1, "previous"))

        row = df.iloc[i]
        for trig_idx, source in candidates:
            if trig_idx < 5 or trig_idx + 1 >= len(df):
                continue

            trigger_bar = df.iloc[trig_idx]
            confirm_bar = df.iloc[trig_idx + 1]

            if not base.is_trigger_rejection(trigger_bar):
                continue
            if not base.passes_climax_dump_filter(trigger_bar):
                continue
            if confirm_bar["range_pct"] > base.MAX_CONFIRM_RANGE_PCT:
                continue
            if base.REQUIRE_CONFIRM_CLOSE_ABOVE_TRIGGER_CLOSE and not (confirm_bar["close"] > trigger_bar["close"]):
                continue
            if base.REQUIRE_CONFIRM_HIGH_ABOVE_TRIGGER_HIGH and not (confirm_bar["high"] > trigger_bar["high"]):
                continue
            if not is_family2_base_context(row):
                continue

            route = classify_signal_route(row, trigger_bar, confirm_bar)
            if route is None:
                continue

            return trig_idx, source

        return None, None

    return route_aware_find_trigger


def classify_trade_route(trade_row: pd.Series) -> str:
    pseudo_row = {
        "stretch_below_ema20_5m_pct": float(trade_row["context_stretch_below_ema20_5m_pct"]),
        "dump_60m_pct": float(trade_row["context_dump_60m_pct"]),
    }
    pseudo_trigger = {
        "body_ratio": float(trade_row["trigger_body_ratio"]),
        "relative_volume_1m": float(trade_row["rel_volume_1m"]),
    }
    pseudo_confirm = {
        "range_pct": float(trade_row["confirm_range_pct"]),
    }
    route = classify_signal_route(pd.Series(pseudo_row), pd.Series(pseudo_trigger), pd.Series(pseudo_confirm))
    return route or "unclassified"


def save_outputs(trades_df: pd.DataFrame, summary: dict[str, object], all_candles: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades_out = trades_df.copy()
    if not trades_out.empty:
        trades_out["route"] = trades_out.apply(classify_trade_route, axis=1)
        trades_out["trade_result"] = np.where(trades_out["pnl_usd"] > 0, "win", np.where(trades_out["pnl_usd"] < 0, "loss", "flat"))
    trades_out.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    lines = [
        "EDGE LONG FAMILY2.1 SEED PLUS SATELLITES V1",
        "============================================",
        "",
        "idea = keep the Family 2 seed intact, then add two clean residual pockets from the same flush-reversal family",
        "engine = old wickMick_long execution engine, but with the new Family 2 router and route-aware expansion",
        "",
        "base family context:",
        "- strong_bear",
        "- high_vol",
        "- near_low",
        "- reversal_like",
        "- bearish 15m EMA alignment",
        "- bearish 1h EMA alignment",
        "- return_pre_60m_pct <= -0.5",
        "",
        "route split:",
        f"- family2_seed: stretch<= {family2_seed.MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT:.2f}, {family2_seed.MIN_TRIGGER_BODY_RATIO:.2f} <= body <= {family2_seed.MAX_TRIGGER_BODY_RATIO:.2f}, rel_volume_1m >= {family2_seed.MIN_RELATIVE_VOLUME_1M:.2f}",
        "- family2_satellite_highconfirm: trigger_body_ratio >= 0.45, confirm_range_pct >= 0.45, context_dump_60m_pct <= 1.05",
        "- family2_satellite_midbody: 0.29 <= trigger_body_ratio <= 0.35, confirm_range_pct <= 0.45, rel_volume_1m >= 1.20",
        "",
        "honest interpretation:",
        "- the seed already removed the obvious losses",
        "- residual analysis found two more live-safe winning pockets",
        "- one is a stronger reclaim / high-confirm cluster",
        "- the other is a mid-body supported reclaim cluster",
        "",
        f"bt_start = {summary['bt_start']}",
        f"bt_end = {summary['bt_end']}",
        f"trade_count = {summary['trade_count']}",
        f"win_rate_pct = {summary['win_rate_pct']:.2f}" if not pd.isna(summary["win_rate_pct"]) else "win_rate_pct = n/a",
        f"profit_factor = {summary['profit_factor']:.3f}" if (not pd.isna(summary['profit_factor']) and summary['profit_factor'] != float('inf')) else "profit_factor = inf",
        f"max_drawdown_pct = {summary['max_drawdown_pct']:.2f}" if not pd.isna(summary["max_drawdown_pct"]) else "max_drawdown_pct = n/a",
        f"expectancy_usd = {summary['expectancy_usd']:.4f}" if not pd.isna(summary["expectancy_usd"]) else "expectancy_usd = n/a",
        f"ending_balance = {summary['ending_balance']:.2f}",
        f"context_without_trigger_count = {summary['context_without_trigger_count']}",
        f"rejected_entry_count = {summary['rejected_entry_count']}",
        f"expired_limit_count = {summary['expired_limit_count']}",
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if trades_out.empty:
        return

    candles_with_regime = all_candles.copy()
    if "trend_regime" not in candles_with_regime.columns:
        candles_with_regime = tier1.merge_daily_regime(candles_with_regime)

    enriched = trades_out.copy()
    enriched["trade_date"] = pd.to_datetime(enriched["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    enriched = enriched.merge(
        candles_with_regime[
            ["trade_date", "trend_regime", "vol_regime", "ret_30d_pct", "ret_60d_pct", "close_vs_sma60_pct"]
        ].drop_duplicates("trade_date"),
        on="trade_date",
        how="left",
    )

    tier1.grouped_summary(enriched, "route").to_csv(OUTPUT_DIR / "route_summary.csv", index=False)
    tier1.grouped_summary(enriched, "trend_regime").to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    tier1.grouped_summary(enriched, "vol_regime").to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    tier1.grouped_summary(enriched, "entry_mode").to_csv(OUTPUT_DIR / "entry_mode_summary.csv", index=False)
    enriched.to_csv(OUTPUT_DIR / "trade_regime_detail.csv", index=False)


def main() -> None:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = tier1.merge_daily_regime(featured)
    featured = family2_seed.enrich_family2_context(featured)

    original_is_base_context = base.is_base_context
    original_find_trigger_index = base.find_trigger_index
    try:
        base.is_base_context = lambda row: True
        base.find_trigger_index = make_route_aware_find_trigger()
        bt_start = featured["open_time"].min()
        bt_end = featured["open_time"].max()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_is_base_context
        base.find_trigger_index = original_find_trigger_index

    save_outputs(trades_df, summary, featured)
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
