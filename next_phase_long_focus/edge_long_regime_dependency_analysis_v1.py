from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_PATH = (
    BASE_DIR
    / "fun_backtests"
    / "cluster1_long_climax_push_failure_rr2_reentry_delayed_limit"
    / "cluster1_long_trades.csv"
)
CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
OUTPUT_DIR = BASE_DIR / "next_phase_long_focus" / "results" / "edge_long_regime_dependency_analysis_v1"


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["entry_time", "exit_time", "signal_time", "trigger_time", "confirm_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df.sort_values("entry_time").reset_index(drop=True)


def load_candles_1m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.dropna(subset=["open_time", "open", "high", "low", "close"])
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def build_daily_from_1m(candles_1m: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = load_trades(TRADES_PATH)
    candles_1m = load_candles_1m(CANDLES_1M_PATH)
    daily = build_daily_from_1m(candles_1m)

    trades["trade_date"] = trades["entry_time"].dt.strftime("%Y-%m-%d")
    merged = trades.merge(
        daily[
            [
                "trade_date",
                "trend_regime",
                "vol_regime",
                "ret_30d_pct",
                "ret_60d_pct",
                "close_vs_sma60_pct",
                "vol_14d_pct",
            ]
        ],
        on="trade_date",
        how="left",
    )
    merged["entry_hour_utc"] = merged["entry_time"].dt.hour
    merged["holding_minutes"] = (merged["exit_time"] - merged["entry_time"]).dt.total_seconds() / 60.0

    trend_summary = grouped_summary(merged, "trend_regime")
    vol_summary = grouped_summary(merged, "vol_regime")
    hour_summary = grouped_summary(merged, "entry_hour_utc")

    overall_pnl = float(merged["pnl_usd"].sum())
    overall_win_rate = float((merged["pnl_usd"] > 0).mean() * 100.0) if len(merged) else 0.0
    overall_profit = float(merged.loc[merged["pnl_usd"] > 0, "pnl_usd"].sum())
    overall_loss = float(-merged.loc[merged["pnl_usd"] < 0, "pnl_usd"].sum())
    overall_pf = overall_profit / overall_loss if overall_loss > 0 else float("inf")

    median_ret_30 = float(merged["ret_30d_pct"].median()) if not merged["ret_30d_pct"].dropna().empty else float("nan")
    median_ret_60 = float(merged["ret_60d_pct"].median()) if not merged["ret_60d_pct"].dropna().empty else float("nan")
    median_close_vs_sma60 = (
        float(merged["close_vs_sma60_pct"].median()) if not merged["close_vs_sma60_pct"].dropna().empty else float("nan")
    )

    summary_lines = [
        "EDGE LONG REGIME DEPENDENCY ANALYSIS V1",
        "=====================================",
        "",
        "baseline_source = cluster1_long_climax_push_failure_rr2_reentry_delayed_limit",
        f"trade_count = {len(merged)}",
        f"net_pnl_usd = {overall_pnl:.4f}",
        f"win_rate_pct = {overall_win_rate:.2f}",
        f"profit_factor = {overall_pf:.3f}" if overall_pf != float("inf") else "profit_factor = inf",
        "",
        f"median_ret_30d_pct = {median_ret_30:.2f}",
        f"median_ret_60d_pct = {median_ret_60:.2f}",
        f"median_close_vs_sma60_pct = {median_close_vs_sma60:.2f}",
        "",
        "trend_regime_summary:",
    ]
    for _, row in trend_summary.iterrows():
        summary_lines.append(
            f"- {row['trend_regime']}: trades={int(row['trade_count'])}, net_pnl={row['net_pnl_usd']:.4f}, "
            f"win_rate={row['win_rate_pct']:.2f}%, pf={row['profit_factor']}"
        )
    summary_lines.extend(["", "vol_regime_summary:"])
    for _, row in vol_summary.iterrows():
        summary_lines.append(
            f"- {row['vol_regime']}: trades={int(row['trade_count'])}, net_pnl={row['net_pnl_usd']:.4f}, "
            f"win_rate={row['win_rate_pct']:.2f}%, pf={row['profit_factor']}"
        )
    summary_lines.extend(["", "entry_hour_summary:"])
    for _, row in hour_summary.iterrows():
        summary_lines.append(
            f"- hour {int(row['entry_hour_utc'])}: trades={int(row['trade_count'])}, net_pnl={row['net_pnl_usd']:.4f}, "
            f"win_rate={row['win_rate_pct']:.2f}%, pf={row['profit_factor']}"
        )

    verdict = []
    bull_like = trend_summary.loc[trend_summary["trend_regime"].isin(["strong_bull", "weak_bull"])]
    bear_like = trend_summary.loc[trend_summary["trend_regime"].isin(["strong_bear", "weak_bear"])]
    if not bull_like.empty and bull_like["net_pnl_usd"].sum() > 0:
        verdict.append("long baseline seems more alive in bull-like daily regimes")
    if not bear_like.empty and bear_like["net_pnl_usd"].sum() <= 0:
        verdict.append("bear-like daily regimes look weak or hostile for this long baseline")
    if not vol_summary.empty and "high_vol" in set(vol_summary["vol_regime"]):
        high_vol_row = vol_summary.loc[vol_summary["vol_regime"] == "high_vol"].iloc[0]
        if float(high_vol_row["net_pnl_usd"]) > 0:
            verdict.append("high-vol conditions did not kill the setup outright")
    if not verdict:
        verdict.append("no strong regime edge yet; long baseline remains weak and unstable")

    summary_lines.extend(["", "verdict:"])
    summary_lines.extend([f"- {line}" for line in verdict])

    trend_summary.to_csv(OUTPUT_DIR / "trend_regime_summary.csv", index=False)
    vol_summary.to_csv(OUTPUT_DIR / "vol_regime_summary.csv", index=False)
    hour_summary.to_csv(OUTPUT_DIR / "entry_hour_summary.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "trade_regime_detail.csv", index=False)
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
