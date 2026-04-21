from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import wickMick_long as base


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "edge_long_tier1_bear_highvol_baseline_v1"


def build_daily_regime_from_1m(candles_1m: pd.DataFrame) -> pd.DataFrame:
    daily = (
        candles_1m.set_index("open_time")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    daily["trade_date"] = daily["open_time"].dt.strftime("%Y-%m-%d")
    daily["daily_return_pct"] = daily["close"].pct_change() * 100.0
    daily["ret_30d_pct"] = daily["close"].pct_change(30) * 100.0
    daily["ret_60d_pct"] = daily["close"].pct_change(60) * 100.0
    daily["sma_30"] = daily["close"].rolling(30).mean()
    daily["sma_60"] = daily["close"].rolling(60).mean()
    daily["close_vs_sma30_pct"] = (daily["close"] / daily["sma_30"] - 1.0) * 100.0
    daily["close_vs_sma60_pct"] = (daily["close"] / daily["sma_60"] - 1.0) * 100.0
    daily["vol_14d_pct"] = daily["daily_return_pct"].rolling(14).std()

    daily["trend_regime"] = "transition"
    strong_bear = (daily["ret_60d_pct"] <= -12.0) & (daily["close_vs_sma60_pct"] <= -3.0)
    strong_bull = (daily["ret_60d_pct"] >= 12.0) & (daily["close_vs_sma60_pct"] >= 3.0)
    weak_bear = (daily["ret_30d_pct"] <= -5.0) & (daily["close_vs_sma30_pct"] <= -1.5)
    weak_bull = (daily["ret_30d_pct"] >= 5.0) & (daily["close_vs_sma30_pct"] >= 1.5)
    daily.loc[strong_bear, "trend_regime"] = "strong_bear"
    daily.loc[strong_bull, "trend_regime"] = "strong_bull"
    daily.loc[(daily["trend_regime"] == "transition") & weak_bear, "trend_regime"] = "weak_bear"
    daily.loc[(daily["trend_regime"] == "transition") & weak_bull, "trend_regime"] = "weak_bull"

    vol_threshold = float(daily["vol_14d_pct"].median(skipna=True))
    daily["vol_regime"] = "low_vol"
    daily.loc[daily["vol_14d_pct"] >= vol_threshold, "vol_regime"] = "high_vol"
    return daily


def merge_daily_regime(featured_1m: pd.DataFrame) -> pd.DataFrame:
    daily = build_daily_regime_from_1m(featured_1m[["open_time", "open", "high", "low", "close", "volume"]].copy())
    enriched = featured_1m.copy()
    enriched["trade_date"] = enriched["open_time"].dt.strftime("%Y-%m-%d")
    enriched = enriched.merge(
        daily[
            [
                "trade_date",
                "trend_regime",
                "vol_regime",
                "ret_30d_pct",
                "ret_60d_pct",
                "close_vs_sma60_pct",
            ]
        ],
        on="trade_date",
        how="left",
    )
    return enriched


def make_tier1_context_gate():
    original = base.is_base_context

    def tier1_context(row: pd.Series) -> bool:
        if not original(row):
            return False
        trend_regime = str(row.get("trend_regime", ""))
        vol_regime = str(row.get("vol_regime", ""))
        return trend_regime in {"strong_bear", "weak_bear"} and vol_regime == "high_vol"

    return tier1_context


def grouped_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, part in df.groupby(group_col, dropna=False):
        pnl = float(part["pnl_usd"].sum())
        trade_count = int(len(part))
        win_rate = float((part["pnl_usd"] > 0).mean() * 100.0) if trade_count else 0.0
        gross_profit = float(part.loc[part["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss = float(-part.loc[part["pnl_usd"] < 0, "pnl_usd"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        rows.append(
            {
                group_col: group_value,
                "trade_count": trade_count,
                "net_pnl_usd": round(pnl, 4),
                "win_rate_pct": round(win_rate, 2),
                "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else profit_factor,
                "expectancy_usd": round(pnl / trade_count, 4) if trade_count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["net_pnl_usd", "trade_count"], ascending=[False, False]).reset_index(drop=True)


def save_outputs(trades_df: pd.DataFrame, summary: dict[str, object], all_candles: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    lines = [
        "EDGE LONG TIER1 BEAR HIGH-VOL BASELINE V1",
        "=========================================",
        "",
        "idea = reuse the best old long climax-dump-failure engine, but make it independent",
        "router = old long context + bear-like daily regime + high_vol",
        "source_engine = wickMick_long.py",
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
        candles_with_regime = merge_daily_regime(candles_with_regime)
    trades_df = trades_df.copy()
    trades_df["trade_date"] = pd.to_datetime(trades_df["entry_time"], utc=True).dt.strftime("%Y-%m-%d")
    merged = trades_df.merge(
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
    grouped_summary(merged, "trend_regime").to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    grouped_summary(merged, "vol_regime").to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    grouped_summary(merged, "entry_mode").to_csv(OUTPUT_DIR / "entry_mode_summary.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "trade_regime_detail.csv", index=False)


def main() -> None:
    candles = base.load_candles(base.CANDLES_1M_PATH)
    featured = base.build_features(candles)
    featured = merge_daily_regime(featured)

    original_is_base_context = base.is_base_context
    try:
        base.is_base_context = make_tier1_context_gate()
        bt_start = featured["open_time"].min()
        bt_end = featured["open_time"].max()
        trades_df, summary = base.run_backtest(featured, bt_start=bt_start, bt_end=bt_end)
    finally:
        base.is_base_context = original_is_base_context

    save_outputs(trades_df, summary, featured)
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
