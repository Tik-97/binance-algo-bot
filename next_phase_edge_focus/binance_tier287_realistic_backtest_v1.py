from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from next_phase_edge_focus import binance_tier286_refined_realistic_backtest_v1 as prev


OUTPUT_DIR = prev.prev.base.BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier287_realistic_backtest_v1"
REFINED2_EXPANSION_MAX_MOVE_3M_PCT = 0.28
ORIGINAL_EXPANSION_FN = prev.is_other_non_target_mild_strong_bear_expansion_refined


def is_other_non_target_mild_strong_bear_expansion_refined2(
    trigger_bar: pd.Series,
    market_ctx: dict[str, object],
    candidate_score: float,
    predicted_regime: str,
    subtype: str,
) -> bool:
    return (
        ORIGINAL_EXPANSION_FN(
            trigger_bar=trigger_bar,
            market_ctx=market_ctx,
            candidate_score=candidate_score,
            predicted_regime=predicted_regime,
            subtype=subtype,
        )
        and float(trigger_bar["move_3m_pct"]) <= REFINED2_EXPANSION_MAX_MOVE_3M_PCT
    )


def write_summary(summary: dict, period_df: pd.DataFrame, trades_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    lines = [
        "BINANCE TIER2.87 REALISTIC BACKTEST V1",
        "======================================",
        "",
        "pipeline = Binance ETHUSDT 1m -> candidate_engine_v1 -> bear regime router -> Tier 2.85 core + refined other_non_target mild strong_bear expansion -> conservative Binance execution",
        "",
        "Refinement added on top of Tier 2.86 refined",
        "---------------------------------------------",
        f"rel_volume_1m <= {prev.REFINED_EXPANSION_MAX_REL_VOLUME:.2f}",
        f"move_5m_pct <= {prev.REFINED_EXPANSION_MAX_MOVE_5M_PCT:.2f}",
        f"move_3m_pct <= {REFINED2_EXPANSION_MAX_MOVE_3M_PCT:.2f}",
        "",
        f"filled_trade_count = {summary['filled_trade_count']}",
        f"ending_balance = {summary['ending_balance']:.2f}",
        f"accepted_other_non_target_expansion = {summary['accepted_other_non_target_expansion']}",
    ]
    if "win_rate_pct" in summary:
        lines.extend(
            [
                f"win_rate_pct = {summary['win_rate_pct']:.2f}",
                f"profit_factor = {summary['profit_factor']:.3f}" if pd.notna(summary["profit_factor"]) else "profit_factor = nan",
                f"expectancy_usd = {summary['expectancy_usd']:.4f}",
                f"median_hold_minutes = {summary['median_hold_minutes']:.2f}",
            ]
        )
    if not trades_df.empty:
        engine_stats = (
            trades_df.groupby("regime_engine")
            .agg(
                trades=("pnl_usd", "size"),
                net_pnl_usd=("pnl_usd", "sum"),
                win_rate_pct=("pnl_usd", lambda s: (s > 0).mean() * 100.0),
            )
            .reset_index()
        )
        lines.extend(["", "Engine Split", "------------"])
        for _, row in engine_stats.iterrows():
            lines.append(
                f"{row['regime_engine']}: trade_count={int(row['trades'])}, win_rate_pct={row['win_rate_pct']:.2f}, net_pnl_usd={row['net_pnl_usd']:.4f}"
            )
    if not period_df.empty:
        lines.extend(["", "Period Split", "------------"])
        for _, row in period_df.iterrows():
            pf = f"{row['profit_factor']:.3f}" if pd.notna(row["profit_factor"]) else "nan"
            lines.append(
                f"{row['period']}: trade_count={int(row['trade_count'])}, win_rate_pct={row['win_rate_pct']:.2f}, net_pnl_usd={row['net_pnl_usd']:.4f}, profit_factor={pf}"
            )
    if not matches_df.empty:
        same_dir = matches_df["same_result_direction"].mean() * 100.0 if "same_result_direction" in matches_df.columns else float("nan")
        within_60m = matches_df["matched_within_60m"].mean() * 100.0 if "matched_within_60m" in matches_df.columns else float("nan")
        lines.extend(
            [
                "",
                "Reference Comparison",
                "--------------------",
                f"match_rows = {len(matches_df)}",
                f"same_result_direction_share = {same_dir:.2f}%",
                f"within_60m_share = {within_60m:.2f}%",
            ]
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    execution_candles = prev.prev.base.load_execution_candles(prev.prev.base.CANDLES_1M_PATH)
    execution_df = prev.prev.base.build_execution_features(execution_candles.copy())

    candidate_features = prev.prev.base.build_candidate_features(execution_candles.copy())
    raw_candidates = prev.prev.base.generate_candidates(candidate_features)
    deduped_candidates = prev.prev.base.deduplicate_candidates(raw_candidates)

    daily_regime = prev.prev.base.build_market_regime_frame_from_candles(execution_candles.copy())
    market_regime_lookup = {
        row["trade_date"]: {
            "trend_regime": row["trend_regime"],
            "vol_regime": row["vol_regime"],
            "ret_30d_pct": float(row["ret_30d_pct"]) if pd.notna(row["ret_30d_pct"]) else float("nan"),
            "ret_60d_pct": float(row["ret_60d_pct"]) if pd.notna(row["ret_60d_pct"]) else float("nan"),
        }
        for _, row in daily_regime.iterrows()
    }

    original_fn = prev.prev.is_other_non_target_mild_strong_bear_expansion
    try:
        prev.prev.is_other_non_target_mild_strong_bear_expansion = is_other_non_target_mild_strong_bear_expansion_refined2
        trades_df, summary = prev.prev.run_backtest_binance_tier286(execution_df, deduped_candidates, market_regime_lookup)
    finally:
        prev.prev.is_other_non_target_mild_strong_bear_expansion = original_fn

    ref_df = prev.prev.base.load_reference_eth_shorts(prev.prev.base.JSON_TRADES_PATH)
    ref_start, ref_end = prev.prev.base.get_reference_window(ref_df)
    period_df = prev.prev.base.build_period_summary(trades_df, ref_start, ref_end)
    matches_df = prev.prev.base.match_to_reference(trades_df, ref_df) if not trades_df.empty else pd.DataFrame()

    raw_candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    deduped_candidates.to_csv(OUTPUT_DIR / "deduped_candidates.csv", index=False)
    daily_regime.to_csv(OUTPUT_DIR / "binance_daily_regime.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    period_df.to_csv(OUTPUT_DIR / "period_summary.csv", index=False)
    matches_df.to_csv(OUTPUT_DIR / "matches.csv", index=False)
    write_summary(summary, period_df, trades_df, matches_df)

    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Trades: {len(trades_df)}")
    print(f"Ending balance: {summary['ending_balance']:.2f}")
    print(f"Expansion trades: {summary['accepted_other_non_target_expansion']}")


if __name__ == "__main__":
    main()
