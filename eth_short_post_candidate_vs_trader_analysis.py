from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
JSON_TRADES_PATH = BASE_DIR / "Another Trader" / "another_trader.json"
NEAREST_CANDIDATES_PATH = BASE_DIR / "fun_backtests" / "eth_short_execution_prototype_v1" / "nearest_candidates.csv"
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_post_candidate_vs_trader_analysis"

SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16
HORIZONS_MINUTES = [1, 3, 5, 10, 15]


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open_time", "open", "high", "low", "close"]).sort_values("open_time").reset_index(drop=True)


def load_session_reference_trades(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df = df[
        (df["symbol"] == "ETHUSDT")
        & (df["side"] == "Short")
        & (df["status"] == "All Closed")
    ].copy()
    df["opened_time"] = pd.to_datetime(df["opened"], unit="ms", utc=True, errors="coerce")
    df["closed_time"] = pd.to_datetime(df["closed"], unit="ms", utc=True, errors="coerce")
    df["avgCost"] = pd.to_numeric(df["avgCost"], errors="coerce")
    df["avgClosePrice"] = pd.to_numeric(df["avgClosePrice"], errors="coerce")
    df["closingPnl"] = pd.to_numeric(df["closingPnl"], errors="coerce")
    df = df[
        (df["opened_time"].dt.hour >= SESSION_START_HOUR_UTC)
        & (df["opened_time"].dt.hour < SESSION_END_HOUR_UTC)
    ].copy()
    df["trade_date"] = df["opened_time"].dt.strftime("%Y-%m-%d")
    df["trader_return_pct"] = (df["avgCost"] - df["avgClosePrice"]) / df["avgCost"] * 100.0
    df["ref_result"] = np.where(df["closingPnl"] > 0, "win", np.where(df["closingPnl"] < 0, "loss", "flat"))
    return df.sort_values("opened_time").reset_index(drop=True)


def load_nearest_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["trade_opened_time", "trade_closed_time", "candidate_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ["trade_avg_cost", "trade_avg_close_price", "trader_return_pct", "candidate_price", "candidate_score", "candidate_abs_minute_distance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.sort_values("trade_opened_time").reset_index(drop=True)


def load_execution_results(version: str) -> pd.DataFrame:
    path = BASE_DIR / "fun_backtests" / f"eth_short_execution_backtest_{version}" / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = [c for c in ["trade_date", "pnl_usd", "exit_reason", "entry_vs_trader_price_pct", "hold_minutes"] if c in df.columns]
    df = df[keep].copy()
    rename = {col: f"{col}_{version}" for col in keep if col != "trade_date"}
    return df.rename(columns=rename)


def get_index_at_or_after(df: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    idx = int(df["open_time"].searchsorted(ts, side="left"))
    if idx >= len(df):
        return None
    return idx


def future_close_return_pct(df: pd.DataFrame, start_idx: int, horizon_minutes: int) -> float:
    target_idx = start_idx + horizon_minutes
    if target_idx >= len(df):
        return np.nan
    start_price = float(df.iloc[start_idx]["close"])
    target_price = float(df.iloc[target_idx]["close"])
    return (target_price - start_price) / start_price * 100.0


def build_comparison_rows(candles: pd.DataFrame, ref: pd.DataFrame, nearest_candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    merged = nearest_candidates.merge(
        ref[["trade_date", "opened_time", "closed_time", "avgCost", "avgClosePrice", "trader_return_pct", "ref_result"]],
        left_on="trade_date",
        right_on="trade_date",
        how="inner",
        suffixes=("", "_ref"),
    )

    for _, row in merged.iterrows():
        candidate_idx = get_index_at_or_after(candles, row["candidate_time"])
        trader_idx = get_index_at_or_after(candles, row["opened_time"])
        if candidate_idx is None or trader_idx is None:
            continue

        candidate_price = float(candles.iloc[candidate_idx]["close"])
        trader_price = float(row["avgCost"])
        start_idx = min(candidate_idx, trader_idx)
        end_idx = max(candidate_idx, trader_idx)
        segment = candles.iloc[start_idx : end_idx + 1].copy()

        pre_trader_high = float(segment["high"].max()) if not segment.empty else np.nan
        pre_trader_low = float(segment["low"].min()) if not segment.empty else np.nan
        trader_delay_minutes = (row["opened_time"] - row["candidate_time"]).total_seconds() / 60.0

        payload = {
            "trade_date": row["trade_date"],
            "candidate_time": row["candidate_time"],
            "trader_opened_time": row["opened_time"],
            "trader_closed_time": row["closed_time"],
            "candidate_abs_minute_distance": float(row["candidate_abs_minute_distance"]),
            "trader_delay_from_candidate_minutes": float(trader_delay_minutes),
            "candidate_price": candidate_price,
            "trader_entry_price": trader_price,
            "trader_exit_price": float(row["avgClosePrice"]),
            "entry_gap_trader_vs_candidate_pct": (trader_price - candidate_price) / candidate_price * 100.0,
            "max_up_move_before_trader_pct": (pre_trader_high - candidate_price) / candidate_price * 100.0 if np.isfinite(pre_trader_high) else np.nan,
            "max_down_move_before_trader_pct": (candidate_price - pre_trader_low) / candidate_price * 100.0 if np.isfinite(pre_trader_low) else np.nan,
            "trader_return_pct": float(row["trader_return_pct"]),
            "ref_result": row["ref_result"],
            "extra_push_before_trader_flag": ((pre_trader_high - candidate_price) / candidate_price * 100.0) >= 0.15 if np.isfinite(pre_trader_high) else False,
            "pre_breakdown_before_trader_flag": ((candidate_price - pre_trader_low) / candidate_price * 100.0) >= 0.15 if np.isfinite(pre_trader_low) else False,
        }

        for horizon in HORIZONS_MINUTES:
            payload[f"candidate_close_return_after_{horizon}m_pct"] = future_close_return_pct(candles, candidate_idx, horizon)

        rows.append(payload)

    comparison = pd.DataFrame(rows).sort_values("trader_opened_time").reset_index(drop=True)
    for version in ["v1", "v2", "v3"]:
        exec_df = load_execution_results(version)
        if not exec_df.empty:
            comparison = comparison.merge(exec_df, on="trade_date", how="left")
    return comparison


def write_summary(comparison: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT POST-CANDIDATE VS TRADER ANALYSIS",
        "===========================================",
        "",
        f"trade_count = {len(comparison)}",
        f"median_candidate_abs_minute_distance = {comparison['candidate_abs_minute_distance'].median():.2f}",
        f"median_trader_delay_from_candidate_minutes = {comparison['trader_delay_from_candidate_minutes'].median():.2f}",
        f"median_entry_gap_trader_vs_candidate_pct = {comparison['entry_gap_trader_vs_candidate_pct'].median():.4f}",
        f"extra_push_before_trader_share = {comparison['extra_push_before_trader_flag'].mean() * 100.0:.2f}%",
        f"pre_breakdown_before_trader_share = {comparison['pre_breakdown_before_trader_flag'].mean() * 100.0:.2f}%",
        "",
        "Candidate Path Medians",
        "----------------------",
    ]

    for horizon in HORIZONS_MINUTES:
        lines.append(
            f"candidate_close_return_after_{horizon}m_pct = {comparison[f'candidate_close_return_after_{horizon}m_pct'].median():.4f}"
        )

    for version in ["v1", "v2", "v3"]:
        pnl_col = f"pnl_usd_{version}"
        if pnl_col in comparison.columns:
            subset = comparison.loc[comparison[pnl_col].notna()].copy()
            lines.extend(
                [
                    "",
                    f"{version.upper()} Execution Snapshot",
                    "-" * (len(version) + 19),
                    f"trade_count = {len(subset)}",
                    f"win_rate_pct = {(subset[pnl_col] > 0).mean() * 100.0:.2f}%" if not subset.empty else "win_rate_pct = nan",
                    f"median_pnl_usd = {subset[pnl_col].median():.4f}" if not subset.empty else "median_pnl_usd = nan",
                    f"median_entry_vs_trader_price_pct = {subset[f'entry_vs_trader_price_pct_{version}'].median():.4f}" if not subset.empty else "median_entry_vs_trader_price_pct = nan",
                ]
            )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candles = load_candles(CANDLES_1M_PATH)
    ref = load_session_reference_trades(JSON_TRADES_PATH)
    nearest_candidates = load_nearest_candidates(NEAREST_CANDIDATES_PATH)
    comparison = build_comparison_rows(candles, ref, nearest_candidates)

    comparison.to_csv(OUTPUT_DIR / "candidate_vs_trader_comparison.csv", index=False)
    write_summary(comparison)

    print(f"Trade count: {len(comparison)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
