from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ENTRY_QUALITY_INTERVALS, ENTRY_REACTION_HORIZONS_MINUTES, PLOTS_DIRNAME
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


def run_entry_quality_analysis(trades_df: pd.DataFrame, logger: logging.Logger) -> dict:
    valid = trades_df.dropna(subset=["symbol", "opened_dt_utc", "closed_dt_utc", "avgCost", "avgClosePrice"]).copy()
    if valid.empty:
        empty = pd.DataFrame()
        return {
            "per_trade": empty,
            "excursion_metrics": empty,
            "exit_efficiency": empty,
            "entry_quality_summary": empty,
            "symbol_entry_quality": empty,
            "side_entry_quality": empty,
            "section_text": "ENTRY QUALITY VS HOLDING EDGE\n-----------------------------\nNo valid trades available for entry-quality analysis.",
        }

    market_data = load_market_context(valid, logger)
    logger.info("Calculating entry quality metrics per trade")
    per_trade_records = [analyze_single_trade(row, market_data.get(str(row["symbol"]), {})) for _, row in valid.iterrows()]
    metrics = pd.DataFrame(per_trade_records).sort_values("trade_number").reset_index(drop=True)

    metrics["group_fast_positive_reaction"] = metrics["time_to_first_positive_minutes"].le(15).fillna(False)
    metrics["group_slow_development"] = (
        metrics["time_to_first_positive_minutes"].gt(60)
        | metrics["time_to_first_positive_minutes"].isna()
    )
    metrics["winner_bucket"] = np.where(metrics["closingPnl"] > 0, "winner", "loser")

    entry_quality_per_trade = build_entry_quality_per_trade(metrics)
    excursion_metrics = build_excursion_metrics(metrics)
    exit_efficiency = build_exit_efficiency(metrics)
    entry_quality_summary = build_entry_quality_summary(metrics)
    symbol_entry_quality = build_group_summary(metrics, "symbol")
    side_entry_quality = build_group_summary(metrics, "side")
    section_text = build_entry_quality_section(metrics, entry_quality_summary, symbol_entry_quality, side_entry_quality)

    return {
        "per_trade": entry_quality_per_trade,
        "excursion_metrics": excursion_metrics,
        "exit_efficiency": exit_efficiency,
        "entry_quality_summary": entry_quality_summary,
        "symbol_entry_quality": symbol_entry_quality,
        "side_entry_quality": side_entry_quality,
        "section_text": section_text,
    }


def load_market_context(valid_trades: pd.DataFrame, logger: logging.Logger) -> dict[str, dict[str, pd.DataFrame]]:
    market_data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in sorted(valid_trades["symbol"].dropna().unique()):
        symbol_trades = valid_trades.loc[valid_trades["symbol"] == symbol]
        start_ms = int((symbol_trades["opened_dt_utc"].min() - pd.Timedelta(hours=24)).timestamp() * 1000)
        coverage_end = pd.concat(
            [
                symbol_trades["closed_dt_utc"],
                symbol_trades["opened_dt_utc"] + pd.Timedelta(hours=24),
            ]
        ).max()
        end_ms = int((coverage_end + pd.Timedelta(hours=1)).timestamp() * 1000)

        market_data[str(symbol)] = {}
        for interval in ENTRY_QUALITY_INTERVALS:
            market_data[str(symbol)][interval] = fetch_futures_klines(
                str(symbol),
                start_ms,
                end_ms,
                interval=interval,
                logger=logger,
            )
    return market_data


def analyze_single_trade(row: pd.Series, symbol_market_data: dict[str, pd.DataFrame]) -> dict:
    entry_time = row["opened_dt_utc"]
    exit_time = row["closed_dt_utc"]
    entry_price = float(row["avgCost"])
    exit_price = float(row["avgClosePrice"])
    side = str(row["side"])

    minute_frame = symbol_market_data.get("1m", pd.DataFrame()).copy()
    if minute_frame.empty:
        return minimal_trade_metrics(row)

    post_entry_1m = minute_frame.loc[minute_frame["close_time"] >= entry_time].copy()
    post_entry_to_close = post_entry_1m.loc[post_entry_1m["close_time"] <= exit_time].copy()

    metrics: dict[str, object] = {
        "trade_number": row["trade_number"],
        "id": row.get("id", np.nan),
        "symbol": row["symbol"],
        "side": side,
        "opened_dt_utc": entry_time,
        "closed_dt_utc": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "closedVolume": row.get("closedVolume", np.nan),
        "closingPnl": row.get("closingPnl", np.nan),
        "holding_hours": row.get("holding_hours", np.nan),
        "trade_outcome": row.get("trade_outcome", pd.NA),
        "realized_return_pct": signed_return_pct(entry_price, exit_price, side),
    }

    for horizon in ENTRY_REACTION_HORIZONS_MINUTES:
        metrics[f"return_after_{horizon}m_pct"] = reaction_return_at_horizon(post_entry_1m, entry_time, entry_price, side, horizon)

    excursion = compute_excursion_metrics(post_entry_to_close, entry_time, entry_price, side)
    metrics.update(excursion)

    speed = compute_speed_to_profit(post_entry_to_close, entry_time, entry_price, side)
    metrics.update(speed)

    pain = compute_entry_pain(post_entry_to_close, entry_time, entry_price, side)
    metrics.update(pain)

    metrics["unrealized_best_return_pct"] = metrics["MFE_pct"]
    metrics["captured_share_of_MFE"] = safe_divide(metrics["realized_return_pct"], metrics["MFE_pct"])
    metrics["giveback_from_peak_pct"] = (
        metrics["MFE_pct"] - metrics["realized_return_pct"]
        if pd.notna(metrics["MFE_pct"]) and pd.notna(metrics["realized_return_pct"])
        else np.nan
    )
    return metrics


def minimal_trade_metrics(row: pd.Series) -> dict:
    metrics = {
        "trade_number": row["trade_number"],
        "id": row.get("id", np.nan),
        "symbol": row["symbol"],
        "side": row["side"],
        "opened_dt_utc": row["opened_dt_utc"],
        "closed_dt_utc": row["closed_dt_utc"],
        "entry_price": row.get("avgCost", np.nan),
        "exit_price": row.get("avgClosePrice", np.nan),
        "closedVolume": row.get("closedVolume", np.nan),
        "closingPnl": row.get("closingPnl", np.nan),
        "holding_hours": row.get("holding_hours", np.nan),
        "trade_outcome": row.get("trade_outcome", pd.NA),
        "realized_return_pct": np.nan,
    }
    for horizon in ENTRY_REACTION_HORIZONS_MINUTES:
        metrics[f"return_after_{horizon}m_pct"] = np.nan
    metrics.update(
        {
            "MFE_pct": np.nan,
            "MAE_pct": np.nan,
            "MFE_price": np.nan,
            "MAE_price": np.nan,
            "time_to_MFE_minutes": np.nan,
            "time_to_MAE_minutes": np.nan,
            "time_to_first_positive_minutes": np.nan,
            "time_to_first_0_25pct_profit_minutes": np.nan,
            "time_to_first_0_5pct_profit_minutes": np.nan,
            "time_to_first_1pct_profit_minutes": np.nan,
            "reached_0_25pct_before_close": False,
            "reached_0_5pct_before_close": False,
            "reached_1pct_before_close": False,
            "went_negative_first": pd.NA,
            "how_deep_before_first_profit_pct": np.nan,
            "max_drawdown_before_first_profit_pct": np.nan,
            "unrealized_best_return_pct": np.nan,
            "captured_share_of_MFE": np.nan,
            "giveback_from_peak_pct": np.nan,
        }
    )
    return metrics


def signed_return_pct(entry_price: float, price: float, side: str) -> float:
    if pd.isna(entry_price) or pd.isna(price) or entry_price == 0:
        return np.nan
    if str(side).lower() == "long":
        return (price - entry_price) / entry_price * 100.0
    return (entry_price - price) / entry_price * 100.0


def reaction_return_at_horizon(frame: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, side: str, horizon_minutes: int) -> float:
    target_time = entry_time + pd.Timedelta(minutes=horizon_minutes)
    candle = frame.loc[frame["close_time"] >= target_time].head(1)
    if candle.empty:
        return np.nan
    return signed_return_pct(entry_price, float(candle.iloc[0]["close"]), side)


def compute_excursion_metrics(frame: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, side: str) -> dict:
    if frame.empty:
        return {
            "MFE_pct": np.nan,
            "MAE_pct": np.nan,
            "MFE_price": np.nan,
            "MAE_price": np.nan,
            "time_to_MFE_minutes": np.nan,
            "time_to_MAE_minutes": np.nan,
        }

    favorable_candidates: list[tuple[float, float, float]] = []
    adverse_candidates: list[tuple[float, float, float]] = []
    full_candles = frame.loc[frame["open_time"] >= entry_time]

    for _, candle in frame.iterrows():
        close_time = candle["close_time"]
        close_price = float(candle["close"])
        minutes = (close_time - entry_time).total_seconds() / 60.0

        close_signed = signed_return_pct(entry_price, close_price, side)
        if pd.notna(close_signed):
            favorable_candidates.append((max(close_signed, 0.0), close_price, minutes))
            adverse_candidates.append((max(-close_signed, 0.0), close_price, minutes))

    for _, candle in full_candles.iterrows():
        close_time = candle["close_time"]
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        minutes = (close_time - entry_time).total_seconds() / 60.0

        if str(side).lower() == "long":
            favorable = (high_price - entry_price) / entry_price * 100.0
            adverse = (entry_price - low_price) / entry_price * 100.0
            favorable_price = high_price
            adverse_price = low_price
        else:
            favorable = (entry_price - low_price) / entry_price * 100.0
            adverse = (high_price - entry_price) / entry_price * 100.0
            favorable_price = low_price
            adverse_price = high_price

        favorable_candidates.append((max(favorable, 0.0), favorable_price, minutes))
        adverse_candidates.append((max(adverse, 0.0), adverse_price, minutes))

    best_favorable = max(favorable_candidates, key=lambda item: item[0]) if favorable_candidates else (np.nan, np.nan, np.nan)
    best_adverse = max(adverse_candidates, key=lambda item: item[0]) if adverse_candidates else (np.nan, np.nan, np.nan)

    return {
        "MFE_pct": best_favorable[0],
        "MFE_price": best_favorable[1],
        "time_to_MFE_minutes": best_favorable[2],
        "MAE_pct": best_adverse[0],
        "MAE_price": best_adverse[1],
        "time_to_MAE_minutes": best_adverse[2],
    }


def compute_speed_to_profit(frame: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, side: str) -> dict:
    if frame.empty:
        return {
            "time_to_first_positive_minutes": np.nan,
            "time_to_first_0_25pct_profit_minutes": np.nan,
            "time_to_first_0_5pct_profit_minutes": np.nan,
            "time_to_first_1pct_profit_minutes": np.nan,
            "reached_0_25pct_before_close": False,
            "reached_0_5pct_before_close": False,
            "reached_1pct_before_close": False,
        }

    close_returns = signed_close_series(frame, entry_price, side)
    close_times = frame["close_time"].reset_index(drop=True)

    def first_crossing(threshold: float) -> float:
        hit = close_times.loc[close_returns >= threshold]
        if hit.empty:
            return np.nan
        return (hit.iloc[0] - entry_time).total_seconds() / 60.0

    return {
        "time_to_first_positive_minutes": first_crossing(0.0),
        "time_to_first_0_25pct_profit_minutes": first_crossing(0.25),
        "time_to_first_0_5pct_profit_minutes": first_crossing(0.5),
        "time_to_first_1pct_profit_minutes": first_crossing(1.0),
        "reached_0_25pct_before_close": bool((close_returns >= 0.25).any()),
        "reached_0_5pct_before_close": bool((close_returns >= 0.5).any()),
        "reached_1pct_before_close": bool((close_returns >= 1.0).any()),
    }


def compute_entry_pain(frame: pd.DataFrame, entry_time: pd.Timestamp, entry_price: float, side: str) -> dict:
    if frame.empty:
        return {
            "went_negative_first": pd.NA,
            "how_deep_before_first_profit_pct": np.nan,
            "max_drawdown_before_first_profit_pct": np.nan,
        }

    close_returns = signed_close_series(frame, entry_price, side)
    close_times = frame["close_time"].reset_index(drop=True)

    first_positive_idx = close_returns[close_returns > 0].index.min()
    first_negative_idx = close_returns[close_returns < 0].index.min()
    went_negative_first = pd.NA
    if pd.notna(first_negative_idx):
        if pd.isna(first_positive_idx):
            went_negative_first = True
        else:
            went_negative_first = bool(first_negative_idx < first_positive_idx)
    elif pd.notna(first_positive_idx):
        went_negative_first = False

    if pd.isna(first_positive_idx):
        window = close_returns
        frame_window = frame
    else:
        window = close_returns.loc[:first_positive_idx]
        frame_window = frame.iloc[: first_positive_idx + 1]

    how_deep = max(-window.min(), 0.0) if not window.empty else np.nan
    adverse_intrabar = intrabar_adverse_series(frame_window.loc[frame_window["open_time"] >= entry_time], entry_price, side)
    max_drawdown = adverse_intrabar.max() if not adverse_intrabar.empty else np.nan

    return {
        "went_negative_first": went_negative_first,
        "how_deep_before_first_profit_pct": how_deep,
        "max_drawdown_before_first_profit_pct": max_drawdown,
    }


def signed_close_series(frame: pd.DataFrame, entry_price: float, side: str) -> pd.Series:
    return frame["close"].apply(lambda price: signed_return_pct(entry_price, float(price), side)).reset_index(drop=True)


def intrabar_adverse_series(frame: pd.DataFrame, entry_price: float, side: str) -> pd.Series:
    if str(side).lower() == "long":
        return ((entry_price - frame["low"].astype(float)) / entry_price * 100.0).clip(lower=0.0).reset_index(drop=True)
    return ((frame["high"].astype(float) - entry_price) / entry_price * 100.0).clip(lower=0.0).reset_index(drop=True)


def build_entry_quality_per_trade(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.copy()


def build_excursion_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_number",
        "id",
        "symbol",
        "side",
        "opened_dt_utc",
        "closed_dt_utc",
        "MFE_pct",
        "MAE_pct",
        "MFE_price",
        "MAE_price",
        "time_to_MFE_minutes",
        "time_to_MAE_minutes",
    ]
    return metrics[columns].copy()


def build_exit_efficiency(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_number",
        "id",
        "symbol",
        "side",
        "opened_dt_utc",
        "closed_dt_utc",
        "realized_return_pct",
        "unrealized_best_return_pct",
        "captured_share_of_MFE",
        "giveback_from_peak_pct",
        "MFE_pct",
        "MAE_pct",
    ]
    return metrics[columns].copy()


def build_entry_quality_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("overall", metrics),
        ("winners", metrics.loc[metrics["closingPnl"] > 0]),
        ("losers", metrics.loc[metrics["closingPnl"] <= 0]),
        ("fast_positive_reaction", metrics.loc[metrics["group_fast_positive_reaction"]]),
        ("slow_development", metrics.loc[metrics["group_slow_development"]]),
    ]
    rows = [summarize_metrics(frame, "summary_group", name) for name, frame in groups]
    return pd.DataFrame(rows)


def build_group_summary(metrics: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_name, frame in metrics.groupby(group_column, dropna=False):
        rows.append(summarize_metrics(frame, group_column, str(group_name)))
    return pd.DataFrame(rows).sort_values("trade_count", ascending=False).reset_index(drop=True)


def summarize_metrics(frame: pd.DataFrame, group_type: str, group_name: str) -> dict:
    row = {
        "group_type": group_type,
        "group_name": group_name,
        "trade_count": len(frame),
        "median_MAE_pct": series_median(frame["MAE_pct"]),
        "median_MFE_pct": series_median(frame["MFE_pct"]),
        "median_realized_return_pct": series_median(frame["realized_return_pct"]),
        "median_captured_share_of_MFE": series_median(frame["captured_share_of_MFE"]),
        "median_time_to_first_profit_minutes": series_median(frame["time_to_first_positive_minutes"]),
        "median_time_to_MFE_minutes": series_median(frame["time_to_MFE_minutes"]),
        "avg_holding_hours": series_mean(frame["holding_hours"]),
    }
    for horizon in [1, 5, 15, 30, 60, 240, 720, 1440]:
        column = f"return_after_{horizon}m_pct"
        row[f"pct_positive_{horizon}m"] = safe_divide((frame[column] > 0).sum(), frame[column].notna().sum())
        row[f"median_return_{horizon}m_pct"] = series_median(frame[column])
    return row


def build_entry_quality_section(
    metrics: pd.DataFrame,
    summary_df: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    side_summary: pd.DataFrame,
) -> str:
    overall = summary_df.loc[summary_df["group_name"] == "overall"].iloc[0] if not summary_df.empty else pd.Series(dtype=object)
    winners = summary_df.loc[summary_df["group_name"] == "winners"].iloc[0] if (summary_df["group_name"] == "winners").any() else pd.Series(dtype=object)
    losers = summary_df.loc[summary_df["group_name"] == "losers"].iloc[0] if (summary_df["group_name"] == "losers").any() else pd.Series(dtype=object)
    eth_row = _group_row(symbol_summary, "ETHUSDT")
    btc_row = _group_row(symbol_summary, "BTCUSDT")
    long_row = _group_row(side_summary, "Long")
    short_row = _group_row(side_summary, "Short")

    timing_strength = score_timing_strength(overall)
    holding_strength = score_holding_strength(overall)
    dominant_edge = classify_dominant_edge(timing_strength, holding_strength)
    confidence = confidence_level(len(metrics), timing_strength, holding_strength)

    short_immediate = compare_metric(short_row, long_row, "pct_positive_15m", threshold=0.08)
    short_holding = compare_metric(short_row, long_row, "median_captured_share_of_MFE", threshold=0.10)
    eth_edge = compare_metric(eth_row, btc_row, "pct_positive_15m", threshold=0.08)
    eth_holding = compare_metric(eth_row, btc_row, "median_captured_share_of_MFE", threshold=0.10)

    return f"""
ENTRY QUALITY VS HOLDING EDGE
-----------------------------
Overall entry-quality summary:
{_table_to_text(summary_df)}

By symbol:
{_table_to_text(symbol_summary)}

By side:
{_table_to_text(side_summary)}

Interpretation:
- Trades in profit after 1m: {pct(overall.get('pct_positive_1m'))}; after 5m: {pct(overall.get('pct_positive_5m'))}; after 15m: {pct(overall.get('pct_positive_15m'))}; after 1h: {pct(overall.get('pct_positive_60m'))}; after 24h: {pct(overall.get('pct_positive_1440m'))}.
- Median MAE: {fmt_number(overall.get('median_MAE_pct'))}%.
- Median MFE: {fmt_number(overall.get('median_MFE_pct'))}%.
- Median realized return: {fmt_number(overall.get('median_realized_return_pct'))}%.
- Median captured share of MFE: {fmt_number(overall.get('median_captured_share_of_MFE'))}.
- Median time to first profit: {fmt_number(overall.get('median_time_to_first_profit_minutes'))} minutes.
- Median time to MFE: {fmt_number(overall.get('median_time_to_MFE_minutes'))} minutes.

Human read:
- Do trades usually move in the trader's favor quickly: {'yes' if timing_strength in {'medium', 'strong'} else 'not consistently'}.
- Entry timing strength: {timing_strength}.
- Holding strength: {holding_strength}.
- Dominant edge appears to come more from: {dominant_edge}.
- ETHUSDT vs BTCUSDT: ETH immediate reaction looks {'stronger' if eth_edge else 'similar/weaker'}, while holding efficiency on ETH looks {'stronger' if eth_holding else 'similar/weaker'}.
- Short side vs Long side: short side looks {'better immediately' if short_immediate else 'not materially better immediately'}, and holding on shorts looks {'stronger' if short_holding else 'not clearly stronger'}.
- Winners vs losers: winners median MAE {fmt_number(winners.get('median_MAE_pct'))}% vs losers {fmt_number(losers.get('median_MAE_pct'))}%; winners median time to first profit {fmt_number(winners.get('median_time_to_first_profit_minutes'))} min vs losers {fmt_number(losers.get('median_time_to_first_profit_minutes'))} min.
- Confidence level: {confidence}.

Method note:
- Reaction metrics use post-entry 1-minute closes for maximal temporal precision.
- Excursion metrics use 1-minute candles to estimate favorable and adverse movement until the recorded exit time.
- This section evaluates how the market behaved after entry; it does not infer strategy indicators or exact signal logic.
""".strip()


def score_timing_strength(overall: pd.Series) -> str:
    score = 0
    if overall.get("pct_positive_5m", 0) >= 0.55:
        score += 1
    if overall.get("pct_positive_15m", 0) >= 0.60:
        score += 1
    if overall.get("pct_positive_60m", 0) >= 0.65:
        score += 1
    if pd.notna(overall.get("median_time_to_first_profit_minutes")) and overall.get("median_time_to_first_profit_minutes") <= 30:
        score += 1

    if score >= 4:
        return "strong"
    if score >= 2:
        return "medium"
    return "weak"


def score_holding_strength(overall: pd.Series) -> str:
    score = 0
    captured = overall.get("median_captured_share_of_MFE")
    if pd.notna(captured) and captured >= 0.55:
        score += 1
    if pd.notna(overall.get("median_return_1440m_pct")) and pd.notna(overall.get("median_return_60m_pct")) and overall.get("median_return_1440m_pct") > overall.get("median_return_60m_pct"):
        score += 1
    if pd.notna(overall.get("median_MFE_pct")) and pd.notna(overall.get("median_realized_return_pct")) and overall.get("median_realized_return_pct") >= 0.5 * overall.get("median_MFE_pct"):
        score += 1

    if score >= 3:
        return "strong"
    if score >= 2:
        return "medium"
    return "weak"


def classify_dominant_edge(timing_strength: str, holding_strength: str) -> str:
    if timing_strength == "strong" and holding_strength == "strong":
        return "a combination of timing and holding"
    if timing_strength in {"strong", "medium"} and holding_strength == "weak":
        return "entry timing"
    if holding_strength in {"strong", "medium"} and timing_strength == "weak":
        return "holding/supervision"
    return "a mixed or unclear combination"


def confidence_level(trade_count: int, timing_strength: str, holding_strength: str) -> str:
    if trade_count >= 100 and (timing_strength != "weak" or holding_strength != "weak"):
        return "high"
    if trade_count >= 30:
        return "medium"
    return "low"


def compare_metric(left: pd.Series, right: pd.Series, column: str, threshold: float) -> bool:
    if left.empty or right.empty:
        return False
    left_value = left.get(column, np.nan)
    right_value = right.get(column, np.nan)
    if pd.isna(left_value) or pd.isna(right_value):
        return False
    return (left_value - right_value) >= threshold


def create_entry_quality_plots(metrics: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("entry_reaction_histograms.png", plot_entry_reaction_histograms, (metrics,)),
        ("mae_histogram.png", plot_single_histogram, (metrics["MAE_pct"], "MAE Distribution", "MAE %")),
        ("mfe_histogram.png", plot_single_histogram, (metrics["MFE_pct"], "MFE Distribution", "MFE %")),
        ("time_to_first_profit_histogram.png", plot_single_histogram, (metrics["time_to_first_positive_minutes"], "Time To First Profit", "Minutes")),
        ("mae_vs_final_return.png", plot_scatter, (metrics["MAE_pct"], metrics["realized_return_pct"], "MAE vs Final Return", "MAE %", "Realized Return %")),
        ("mfe_vs_realized_return.png", plot_scatter, (metrics["MFE_pct"], metrics["realized_return_pct"], "MFE vs Realized Return", "MFE %", "Realized Return %")),
        ("cumulative_profitability_over_time.png", plot_cumulative_profitability, (metrics, None, "All Trades: Share Becoming Profitable Over Time")),
        ("cumulative_profitability_long_vs_short.png", plot_cumulative_profitability_split, (metrics, "side", ["Long", "Short"], "Long vs Short: Share Becoming Profitable")),
        ("cumulative_profitability_eth_vs_btc.png", plot_cumulative_profitability_split, (metrics, "symbol", ["ETHUSDT", "BTCUSDT"], "ETH vs BTC: Share Becoming Profitable")),
    ]

    for filename, plotter, args in plot_specs:
        fig = plotter(*args)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", filename)


def plot_entry_reaction_histograms(metrics: pd.DataFrame) -> plt.Figure:
    horizons = [1, 5, 15, 60]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, horizon in zip(axes.flatten(), horizons):
        series = metrics[f"return_after_{horizon}m_pct"].dropna()
        axis.hist(series, bins=30, edgecolor="black")
        axis.set_title(f"Return After {horizon}m")
        axis.set_xlabel("Return %")
        axis.set_ylabel("Trades")
        axis.grid(alpha=0.2)
    return fig


def plot_single_histogram(series: pd.Series, title: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(series.dropna(), bins=30, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Trades")
    ax.grid(alpha=0.2)
    return fig


def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    return fig


def plot_cumulative_profitability(metrics: pd.DataFrame, _: object, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    minutes_grid = np.arange(0, 1441, 5)
    y_values = cumulative_profitability_share(metrics["time_to_first_positive_minutes"], minutes_grid)
    ax.plot(minutes_grid, y_values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Minutes Since Entry")
    ax.set_ylabel("Share of Trades Profitable At Least Once")
    ax.grid(alpha=0.2)
    return fig


def plot_cumulative_profitability_split(metrics: pd.DataFrame, group_column: str, groups: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    minutes_grid = np.arange(0, 1441, 5)
    for group in groups:
        frame = metrics.loc[metrics[group_column] == group]
        if frame.empty:
            continue
        y_values = cumulative_profitability_share(frame["time_to_first_positive_minutes"], minutes_grid)
        ax.plot(minutes_grid, y_values, linewidth=2, label=group)
    ax.set_title(title)
    ax.set_xlabel("Minutes Since Entry")
    ax.set_ylabel("Share of Trades Profitable At Least Once")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def cumulative_profitability_share(times: pd.Series, minutes_grid: np.ndarray) -> np.ndarray:
    valid_times = times.dropna().to_numpy(dtype=float)
    if valid_times.size == 0:
        return np.zeros_like(minutes_grid, dtype=float)
    return np.array([(valid_times <= minute).mean() for minute in minutes_grid], dtype=float)


def series_median(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.median())


def series_mean(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _group_row(frame: pd.DataFrame, value: str) -> pd.Series:
    if frame.empty or "group_name" not in frame.columns:
        return pd.Series(dtype=object)
    matched = frame.loc[frame["group_name"] == value]
    if matched.empty:
        return pd.Series(dtype=object)
    return matched.iloc[0]


def _table_to_text(df: pd.DataFrame, digits: int = 2) -> str:
    if df.empty:
        return "n/a"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(digits)
    return display.to_string(index=False)
