from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base
import market_context_at_entry as mc
import next_phase_long_focus.edge_long_tier1_bear_highvol_baseline_v1 as tier1


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_family2_flush_reversal_seed_v1"

MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT = 1.10
MIN_TRIGGER_BODY_RATIO = 0.14
MAX_TRIGGER_BODY_RATIO = 0.26
MIN_RELATIVE_VOLUME_1M = 1.20


def enrich_family2_context(featured: pd.DataFrame) -> pd.DataFrame:
    one = featured[["open_time", "open", "high", "low", "close", "volume"]].copy()
    one = one.rename(columns={"open_time": "close_time"})
    one["open_time"] = one["close_time"]

    frames: dict[str, pd.DataFrame] = {}
    for rule, label in [("5min", "5m"), ("15min", "15m"), ("1h", "1h")]:
        rs = (
            one.set_index("close_time")
            .resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        rs["open_time"] = rs["close_time"]
        rs = mc.prepare_context_features(rs, label)
        rs = rs[
            ["open_time", "ema_alignment", "distance_to_ema20_pct", "relative_volume", "rsi14"]
        ].rename(
            columns={
                "ema_alignment": f"ema_alignment_{label}",
                "distance_to_ema20_pct": f"distance_to_ema20_pct_{label}",
                "relative_volume": f"relative_volume_{label}",
                "rsi14": f"rsi14_{label}",
            }
        )
        frames[label] = rs.sort_values("open_time")

    enriched = featured.copy().sort_values("open_time")
    for label in ["5m", "15m", "1h"]:
        enriched = pd.merge_asof(
            enriched,
            frames[label],
            on="open_time",
            direction="backward",
        )

    close_series = enriched["close"].astype(float)
    high_series = enriched["high"].astype(float)
    low_series = enriched["low"].astype(float)
    prev_4h_high = high_series.rolling(241, min_periods=2).max().shift(1)
    prev_4h_low = low_series.rolling(241, min_periods=2).min().shift(1)
    ret5 = (close_series / close_series.shift(5) - 1.0) * 100.0
    ret15 = (close_series / close_series.shift(15) - 1.0) * 100.0
    ret60 = (close_series / close_series.shift(60) - 1.0) * 100.0

    structure_vals: list[str] = []
    context_vals: list[str] = []
    for idx in range(len(enriched)):
        latest_close = float(close_series.iat[idx])
        p4h = prev_4h_high.iat[idx]
        p4l = prev_4h_low.iat[idx]
        breakout_up = bool(pd.notna(p4h) and latest_close > p4h)
        breakout_down = bool(pd.notna(p4l) and latest_close < p4l)
        structure = mc.classify_structure_regime(latest_close, p4h, p4l, breakout_up, breakout_down)
        row = {
            "pullback_before_entry_flag": bool(pd.notna(ret60.iat[idx]) and pd.notna(ret5.iat[idx]) and ret60.iat[idx] > 0 and ret5.iat[idx] < 0),
            "impulse_before_entry_flag": bool(pd.notna(ret15.iat[idx]) and pd.notna(ret5.iat[idx]) and ret15.iat[idx] > 0 and ret5.iat[idx] > 0),
            "breakout_before_entry_flag": breakout_up,
            "structure_regime": structure,
            "ema_alignment_15m": enriched["ema_alignment_15m"].iat[idx],
            "ema_alignment_1h": enriched["ema_alignment_1h"].iat[idx],
        }
        structure_vals.append(structure)
        context_vals.append(mc.classify_context_style(row, "Long"))

    enriched["structure_regime2"] = structure_vals
    enriched["context_style_label2"] = context_vals
    enriched["return_pre_60m_pct2"] = ret60
    return enriched


def is_family2_seed_context(row: pd.Series) -> bool:
    return (
        row.get("trend_regime") == "strong_bear"
        and row.get("vol_regime") == "high_vol"
        and row.get("structure_regime2") == "near_low"
        and row.get("context_style_label2") == "reversal_like"
        and row.get("ema_alignment_15m") == "bearish"
        and row.get("ema_alignment_1h") == "bearish"
        and pd.notna(row.get("return_pre_60m_pct2"))
        and float(row["return_pre_60m_pct2"]) <= -0.5
        and pd.notna(row.get("stretch_below_ema20_5m_pct"))
        and float(row["stretch_below_ema20_5m_pct"]) <= MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT
    )


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
            if not is_family2_seed_context(row):
                continue
            if float(trigger_bar["body_ratio"]) < MIN_TRIGGER_BODY_RATIO:
                continue
            if float(trigger_bar["body_ratio"]) > MAX_TRIGGER_BODY_RATIO:
                continue
            if float(trigger_bar["relative_volume_1m"]) < MIN_RELATIVE_VOLUME_1M:
                continue

            return trig_idx, source

        return None, None

    return route_aware_find_trigger


def save_outputs(trades_df: pd.DataFrame, summary: dict[str, object], all_candles: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades_out = trades_df.copy()
    if not trades_out.empty:
        trades_out["route"] = "family2_flush_reversal_seed"
        trades_out["trade_result"] = np.where(trades_out["pnl_usd"] > 0, "win", np.where(trades_out["pnl_usd"] < 0, "loss", "flat"))
    trades_out.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    lines = [
        "EDGE LONG FAMILY2 FLUSH REVERSAL SEED V1",
        "=========================================",
        "",
        "idea = take the Another Trader flush-reversal family and keep only the part that separated wins from losses inside the old-engine probe",
        "engine = old wickMick_long execution engine, but with a new independent family router",
        "",
        "family router:",
        "- strong_bear",
        "- high_vol",
        "- structure_regime = near_low",
        "- context_style = reversal_like",
        "- ema_alignment_15m = bearish",
        "- ema_alignment_1h = bearish",
        "- return_pre_60m_pct <= -0.5",
        "",
        "loss-cutting layer from win/loss analysis:",
        f"- context_stretch_below_ema20_5m_pct <= {MAX_CONTEXT_STRETCH_BELOW_EMA20_5M_PCT:.2f}",
        f"- {MIN_TRIGGER_BODY_RATIO:.2f} <= trigger_body_ratio <= {MAX_TRIGGER_BODY_RATIO:.2f}",
        f"- rel_volume_1m >= {MIN_RELATIVE_VOLUME_1M:.2f}",
        "",
        "honest interpretation:",
        "- old family_a router alone was noisy",
        "- losses clustered around too-thin trigger bodies and/or weaker local volume",
        "- this seed keeps the balanced-body, properly sponsored reversal pocket",
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
    featured = enrich_family2_context(featured)

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
