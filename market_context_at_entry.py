from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ENTRY_QUALITY_INTERVALS, PLOTS_DIRNAME
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


MARKET_CONTEXT_INTERVALS = ["1m", "5m", "15m", "1h"]
CONTEXT_PLOT_FEATURES = [
    ("rsi14_15m", "RSI 15m"),
    ("atr14_pct_15m", "ATR % 15m"),
    ("return_pre_15m_pct", "Pre-entry Return 15m"),
    ("distance_to_ema20_pct_15m", "Distance To EMA20 15m %"),
    ("relative_volume_1m", "Relative Volume 1m"),
    ("distance_to_last_4h_high_pct", "Distance To Last 4h High %"),
]


def run_market_context_analysis(
    trades_df: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    valid = trades_df.dropna(subset=["symbol", "opened_dt_utc"]).copy()
    if valid.empty:
        empty = pd.DataFrame()
        section = "MARKET CONTEXT AT ENTRY\n-----------------------\nNo valid trades available for market context analysis.\n"
        (output_dir / "context_summary.txt").write_text(section, encoding="utf-8")
        return {
            "market_context_at_entry": empty,
            "strong_vs_weak_entry_comparison": empty,
            "context_summary_text": section,
        }

    entry_strength = classify_entry_strength(entry_quality_df)
    market_data = load_context_market_data(valid, logger)

    logger.info("Calculating market context at entry")
    rows = []
    for _, trade in valid.iterrows():
        symbol = str(trade["symbol"])
        frames = market_data.get(symbol, {})
        context_row = analyze_trade_context(trade, frames)
        rows.append(context_row)

    context_df = pd.DataFrame(rows)
    context_df = context_df.merge(
        entry_strength[["trade_number", "entry_strength_group"]],
        on="trade_number",
        how="left",
    )

    comparison_df = build_strong_vs_weak_entry_comparison(context_df)
    context_summary_text = build_context_summary_text(context_df, comparison_df)
    (output_dir / "context_summary.txt").write_text(context_summary_text, encoding="utf-8")

    return {
        "market_context_at_entry": context_df,
        "strong_vs_weak_entry_comparison": comparison_df,
        "context_summary_text": context_summary_text,
    }


def classify_entry_strength(entry_quality_df: pd.DataFrame) -> pd.DataFrame:
    metrics = entry_quality_df.copy()
    if metrics.empty:
        metrics["entry_strength_group"] = pd.Series(dtype="string")
        return metrics

    mae_median = metrics["MAE_pct"].dropna().median()
    mae_q75 = metrics["MAE_pct"].dropna().quantile(0.75)
    profit_median = metrics["time_to_first_positive_minutes"].dropna().median()
    profit_q75 = metrics["time_to_first_positive_minutes"].dropna().quantile(0.75)

    early_positive = (metrics["return_after_1m_pct"] > 0) | (metrics["return_after_5m_pct"] > 0)
    low_mae = metrics["MAE_pct"] <= mae_median
    high_mae = metrics["MAE_pct"] >= mae_q75
    fast_profit = metrics["time_to_first_positive_minutes"].le(profit_median).fillna(False)
    slow_profit = metrics["time_to_first_positive_minutes"].ge(profit_q75).fillna(False) | metrics["time_to_first_positive_minutes"].isna()

    metrics["entry_strength_group"] = np.where(
        early_positive & low_mae & fast_profit,
        "strong_entry",
        np.where(
            (~early_positive) & (high_mae | slow_profit),
            "weak_entry",
            "medium_entry",
        ),
    )
    return metrics


def load_context_market_data(valid_trades: pd.DataFrame, logger: logging.Logger) -> dict[str, dict[str, pd.DataFrame]]:
    market_data: dict[str, dict[str, pd.DataFrame]] = {}
    warmup_hours = {
        "1m": 24,
        "5m": 24,
        "15m": 72,
        "1h": 300,
    }

    for symbol in sorted(valid_trades["symbol"].dropna().unique()):
        symbol_trades = valid_trades.loc[valid_trades["symbol"] == symbol]
        market_data[str(symbol)] = {}
        min_entry = symbol_trades["opened_dt_utc"].min()
        max_entry = symbol_trades["opened_dt_utc"].max()

        for interval in MARKET_CONTEXT_INTERVALS:
            start_ms = int((min_entry - pd.Timedelta(hours=warmup_hours[interval])).timestamp() * 1000)
            end_ms = int((max_entry + pd.Timedelta(hours=1)).timestamp() * 1000)
            frame = fetch_futures_klines(
                str(symbol),
                start_ms,
                end_ms,
                interval=interval,
                logger=logger,
            )
            market_data[str(symbol)][interval] = prepare_context_features(frame, interval)
    return market_data


def prepare_context_features(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    prepared["ema20"] = prepared["close"].ewm(span=20, adjust=False).mean()
    prepared["ema50"] = prepared["close"].ewm(span=50, adjust=False).mean()
    prepared["ema200"] = prepared["close"].ewm(span=200, adjust=False).mean()
    prepared["distance_to_ema20_pct"] = (prepared["close"] - prepared["ema20"]) / prepared["ema20"] * 100.0
    prepared["distance_to_ema50_pct"] = (prepared["close"] - prepared["ema50"]) / prepared["ema50"] * 100.0
    prepared["ema20_slope_pct"] = (prepared["ema20"] - prepared["ema20"].shift(1)) / prepared["ema20"].shift(1) * 100.0
    prepared["ema50_slope_pct"] = (prepared["ema50"] - prepared["ema50"].shift(1)) / prepared["ema50"].shift(1) * 100.0
    prepared["ema_alignment"] = np.where(
        (prepared["ema20"] > prepared["ema50"]) & (prepared["ema50"] > prepared["ema200"]),
        "bullish",
        np.where(
            (prepared["ema20"] < prepared["ema50"]) & (prepared["ema50"] < prepared["ema200"]),
            "bearish",
            "mixed",
        ),
    )

    close_diff = prepared["close"].diff()
    gains = close_diff.clip(lower=0.0)
    losses = (-close_diff).clip(lower=0.0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    prepared["rsi14"] = 100 - (100 / (1 + rs))
    rsi_min = prepared["rsi14"].rolling(14, min_periods=14).min()
    rsi_max = prepared["rsi14"].rolling(14, min_periods=14).max()
    prepared["stoch_rsi14"] = (prepared["rsi14"] - rsi_min) / (rsi_max - rsi_min)

    previous_close = prepared["close"].shift(1)
    true_range = pd.concat(
        [
            prepared["high"] - prepared["low"],
            (prepared["high"] - previous_close).abs(),
            (prepared["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    prepared["atr14"] = true_range.rolling(14, min_periods=14).mean()
    prepared["atr14_pct"] = prepared["atr14"] / prepared["close"] * 100.0

    prepared["avg_volume20"] = prepared["volume"].rolling(20, min_periods=5).mean()
    prepared["relative_volume"] = prepared["volume"] / prepared["avg_volume20"]
    return prepared


def analyze_trade_context(trade: pd.Series, symbol_frames: dict[str, pd.DataFrame]) -> dict:
    entry_time = trade["opened_dt_utc"]
    row: dict[str, object] = {
        "trade_number": trade["trade_number"],
        "id": trade.get("id", np.nan),
        "symbol": trade["symbol"],
        "side": trade["side"],
        "opened_dt_utc": entry_time,
        "closed_dt_utc": trade["closed_dt_utc"],
        "closingPnl": trade.get("closingPnl", np.nan),
    }

    latest_by_interval: dict[str, pd.Series] = {}
    for interval in MARKET_CONTEXT_INTERVALS:
        frame = symbol_frames.get(interval, pd.DataFrame())
        latest = latest_context_candle(frame, entry_time)
        latest_by_interval[interval] = latest
        prefix = interval
        row.update(extract_interval_context(latest, prefix))

    one_minute = symbol_frames.get("1m", pd.DataFrame())
    row.update(extract_structure_and_preentry(one_minute, entry_time, trade["side"]))
    row["context_style_label"] = classify_context_style(row, trade["side"])
    return row


def latest_context_candle(frame: pd.DataFrame, entry_time: pd.Timestamp) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=object)
    subset = frame.loc[frame["close_time"] <= entry_time]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.iloc[-1]


def extract_interval_context(latest: pd.Series, interval: str) -> dict:
    if latest.empty:
        return {
            f"context_close_{interval}": np.nan,
            f"ema20_{interval}": np.nan,
            f"ema50_{interval}": np.nan,
            f"ema200_{interval}": np.nan,
            f"distance_to_ema20_pct_{interval}": np.nan,
            f"distance_to_ema50_pct_{interval}": np.nan,
            f"ema20_slope_pct_{interval}": np.nan,
            f"ema50_slope_pct_{interval}": np.nan,
            f"ema_alignment_{interval}": pd.NA,
            f"rsi14_{interval}": np.nan,
            f"stoch_rsi14_{interval}": np.nan,
            f"atr14_{interval}": np.nan,
            f"atr14_pct_{interval}": np.nan,
            f"current_volume_{interval}": np.nan,
            f"avg_volume20_{interval}": np.nan,
            f"relative_volume_{interval}": np.nan,
        }

    return {
        f"context_close_{interval}": latest.get("close", np.nan),
        f"ema20_{interval}": latest.get("ema20", np.nan),
        f"ema50_{interval}": latest.get("ema50", np.nan),
        f"ema200_{interval}": latest.get("ema200", np.nan),
        f"distance_to_ema20_pct_{interval}": latest.get("distance_to_ema20_pct", np.nan),
        f"distance_to_ema50_pct_{interval}": latest.get("distance_to_ema50_pct", np.nan),
        f"ema20_slope_pct_{interval}": latest.get("ema20_slope_pct", np.nan),
        f"ema50_slope_pct_{interval}": latest.get("ema50_slope_pct", np.nan),
        f"ema_alignment_{interval}": latest.get("ema_alignment", pd.NA),
        f"rsi14_{interval}": latest.get("rsi14", np.nan),
        f"stoch_rsi14_{interval}": latest.get("stoch_rsi14", np.nan),
        f"atr14_{interval}": latest.get("atr14", np.nan),
        f"atr14_pct_{interval}": latest.get("atr14_pct", np.nan),
        f"current_volume_{interval}": latest.get("volume", np.nan),
        f"avg_volume20_{interval}": latest.get("avg_volume20", np.nan),
        f"relative_volume_{interval}": latest.get("relative_volume", np.nan),
    }


def extract_structure_and_preentry(frame_1m: pd.DataFrame, entry_time: pd.Timestamp, side: str) -> dict:
    if frame_1m.empty:
        return {
            "return_pre_1m_pct": np.nan,
            "roc_5m_pct": np.nan,
            "roc_15m_pct": np.nan,
            "roc_60m_pct": np.nan,
            "recent_range_15m_pct": np.nan,
            "recent_range_1h_pct": np.nan,
            "recent_range_4h_pct": np.nan,
            "distance_to_last_1h_high_pct": np.nan,
            "distance_to_last_1h_low_pct": np.nan,
            "distance_to_last_4h_high_pct": np.nan,
            "distance_to_last_4h_low_pct": np.nan,
            "structure_regime": pd.NA,
            "return_pre_5m_pct": np.nan,
            "return_pre_15m_pct": np.nan,
            "return_pre_60m_pct": np.nan,
            "pullback_before_entry_flag": False,
            "impulse_before_entry_flag": False,
            "breakout_before_entry_flag": False,
        }

    subset = frame_1m.loc[frame_1m["close_time"] <= entry_time].copy()
    if subset.empty:
        return {
            "return_pre_1m_pct": np.nan,
            "roc_5m_pct": np.nan,
            "roc_15m_pct": np.nan,
            "roc_60m_pct": np.nan,
            "recent_range_15m_pct": np.nan,
            "recent_range_1h_pct": np.nan,
            "recent_range_4h_pct": np.nan,
            "distance_to_last_1h_high_pct": np.nan,
            "distance_to_last_1h_low_pct": np.nan,
            "distance_to_last_4h_high_pct": np.nan,
            "distance_to_last_4h_low_pct": np.nan,
            "structure_regime": pd.NA,
            "return_pre_5m_pct": np.nan,
            "return_pre_15m_pct": np.nan,
            "return_pre_60m_pct": np.nan,
            "pullback_before_entry_flag": False,
            "impulse_before_entry_flag": False,
            "breakout_before_entry_flag": False,
        }

    close_series = subset["close"].astype(float).reset_index(drop=True)
    high_series = subset["high"].astype(float).reset_index(drop=True)
    low_series = subset["low"].astype(float).reset_index(drop=True)
    latest_close = float(close_series.iloc[-1])

    prev_1h_high = high_series.tail(61).iloc[:-1].max() if len(high_series) >= 2 else np.nan
    prev_1h_low = low_series.tail(61).iloc[:-1].min() if len(low_series) >= 2 else np.nan
    prev_4h_high = high_series.tail(241).iloc[:-1].max() if len(high_series) >= 2 else np.nan
    prev_4h_low = low_series.tail(241).iloc[:-1].min() if len(low_series) >= 2 else np.nan

    return_pre_1m = rate_of_change(close_series, 1)
    return_pre_5m = rate_of_change(close_series, 5)
    return_pre_15m = rate_of_change(close_series, 15)
    return_pre_60m = rate_of_change(close_series, 60)

    recent_range_15m_pct = range_pct(high_series, low_series, latest_close, 15)
    recent_range_1h_pct = range_pct(high_series, low_series, latest_close, 60)
    recent_range_4h_pct = range_pct(high_series, low_series, latest_close, 240)

    distance_to_last_1h_high_pct = safe_divide(prev_1h_high - latest_close, latest_close) * 100 if pd.notna(prev_1h_high) else np.nan
    distance_to_last_1h_low_pct = safe_divide(latest_close - prev_1h_low, latest_close) * 100 if pd.notna(prev_1h_low) else np.nan
    distance_to_last_4h_high_pct = safe_divide(prev_4h_high - latest_close, latest_close) * 100 if pd.notna(prev_4h_high) else np.nan
    distance_to_last_4h_low_pct = safe_divide(latest_close - prev_4h_low, latest_close) * 100 if pd.notna(prev_4h_low) else np.nan

    breakout_up = bool(pd.notna(prev_4h_high) and latest_close > prev_4h_high)
    breakout_down = bool(pd.notna(prev_4h_low) and latest_close < prev_4h_low)
    structure_regime = classify_structure_regime(latest_close, prev_4h_high, prev_4h_low, breakout_up, breakout_down)

    if str(side).lower() == "long":
        pullback_flag = bool(pd.notna(return_pre_60m) and pd.notna(return_pre_5m) and return_pre_60m > 0 and return_pre_5m < 0)
        impulse_flag = bool(pd.notna(return_pre_15m) and pd.notna(return_pre_5m) and return_pre_15m > 0 and return_pre_5m > 0)
        breakout_flag = breakout_up
    else:
        pullback_flag = bool(pd.notna(return_pre_60m) and pd.notna(return_pre_5m) and return_pre_60m < 0 and return_pre_5m > 0)
        impulse_flag = bool(pd.notna(return_pre_15m) and pd.notna(return_pre_5m) and return_pre_15m < 0 and return_pre_5m < 0)
        breakout_flag = breakout_down

    return {
        "return_pre_1m_pct": return_pre_1m,
        "roc_5m_pct": return_pre_5m,
        "roc_15m_pct": return_pre_15m,
        "roc_60m_pct": return_pre_60m,
        "recent_range_15m_pct": recent_range_15m_pct,
        "recent_range_1h_pct": recent_range_1h_pct,
        "recent_range_4h_pct": recent_range_4h_pct,
        "distance_to_last_1h_high_pct": distance_to_last_1h_high_pct,
        "distance_to_last_1h_low_pct": distance_to_last_1h_low_pct,
        "distance_to_last_4h_high_pct": distance_to_last_4h_high_pct,
        "distance_to_last_4h_low_pct": distance_to_last_4h_low_pct,
        "structure_regime": structure_regime,
        "return_pre_5m_pct": return_pre_5m,
        "return_pre_15m_pct": return_pre_15m,
        "return_pre_60m_pct": return_pre_60m,
        "pullback_before_entry_flag": pullback_flag,
        "impulse_before_entry_flag": impulse_flag,
        "breakout_before_entry_flag": breakout_flag,
    }


def rate_of_change(close_series: pd.Series, periods: int) -> float:
    if len(close_series) <= periods:
        return np.nan
    previous = float(close_series.iloc[-periods - 1])
    current = float(close_series.iloc[-1])
    if previous == 0:
        return np.nan
    return (current - previous) / previous * 100.0


def range_pct(high_series: pd.Series, low_series: pd.Series, close_value: float, periods: int) -> float:
    if len(high_series) < periods:
        window_high = high_series.max()
        window_low = low_series.min()
    else:
        window_high = high_series.tail(periods).max()
        window_low = low_series.tail(periods).min()
    if pd.isna(window_high) or pd.isna(window_low) or close_value == 0:
        return np.nan
    return (window_high - window_low) / close_value * 100.0


def classify_structure_regime(close_value: float, prev_high: float, prev_low: float, breakout_up: bool, breakout_down: bool) -> str:
    if breakout_up:
        return "breakout_up"
    if breakout_down:
        return "breakout_down"
    if pd.isna(prev_high) or pd.isna(prev_low) or prev_high == prev_low:
        return "unknown"
    range_position = (close_value - prev_low) / (prev_high - prev_low)
    if range_position >= 0.8:
        return "near_high"
    if range_position <= 0.2:
        return "near_low"
    return "mid_range"


def classify_context_style(row: dict, side: str) -> str:
    if row.get("pullback_before_entry_flag"):
        return "pullback"
    if row.get("breakout_before_entry_flag") or row.get("structure_regime") in {"breakout_up", "breakout_down"}:
        return "breakout_continuation"

    alignment_15m = str(row.get("ema_alignment_15m", ""))
    alignment_1h = str(row.get("ema_alignment_1h", ""))
    if str(side).lower() == "long":
        if alignment_15m == "bearish" or alignment_1h == "bearish":
            return "reversal_like"
    else:
        if alignment_15m == "bullish" or alignment_1h == "bullish":
            return "reversal_like"
    if row.get("impulse_before_entry_flag"):
        return "continuation_like"
    return "mid_range_or_mixed"


def build_strong_vs_weak_entry_comparison(context_df: pd.DataFrame) -> pd.DataFrame:
    subsets = [
        ("all", context_df),
        ("side_Long", context_df.loc[context_df["side"] == "Long"]),
        ("side_Short", context_df.loc[context_df["side"] == "Short"]),
        ("symbol_ETHUSDT", context_df.loc[context_df["symbol"] == "ETHUSDT"]),
        ("symbol_BTCUSDT", context_df.loc[context_df["symbol"] == "BTCUSDT"]),
    ]

    rows = []
    for subset_name, subset_frame in subsets:
        strong = subset_frame.loc[subset_frame["entry_strength_group"] == "strong_entry"]
        medium = subset_frame.loc[subset_frame["entry_strength_group"] == "medium_entry"]
        weak = subset_frame.loc[subset_frame["entry_strength_group"] == "weak_entry"]
        rows.append(
            {
                "subset": subset_name,
                "strong_count": len(strong),
                "medium_count": len(medium),
                "weak_count": len(weak),
                "strong_breakout_share": bool_share(strong["context_style_label"].isin(["breakout_continuation", "continuation_like"])),
                "weak_breakout_share": bool_share(weak["context_style_label"].isin(["breakout_continuation", "continuation_like"])),
                "strong_pullback_share": bool_share(strong["pullback_before_entry_flag"]),
                "weak_pullback_share": bool_share(weak["pullback_before_entry_flag"]),
                "strong_reversal_like_share": bool_share(strong["context_style_label"] == "reversal_like"),
                "weak_reversal_like_share": bool_share(weak["context_style_label"] == "reversal_like"),
                "strong_median_rsi14_15m": median_of(strong["rsi14_15m"]),
                "weak_median_rsi14_15m": median_of(weak["rsi14_15m"]),
                "strong_median_return_pre_15m_pct": median_of(strong["return_pre_15m_pct"]),
                "weak_median_return_pre_15m_pct": median_of(weak["return_pre_15m_pct"]),
                "strong_median_atr14_pct_15m": median_of(strong["atr14_pct_15m"]),
                "weak_median_atr14_pct_15m": median_of(weak["atr14_pct_15m"]),
                "strong_median_relative_volume_1m": median_of(strong["relative_volume_1m"]),
                "weak_median_relative_volume_1m": median_of(weak["relative_volume_1m"]),
                "strong_median_distance_to_ema20_pct_15m": median_of(strong["distance_to_ema20_pct_15m"]),
                "weak_median_distance_to_ema20_pct_15m": median_of(weak["distance_to_ema20_pct_15m"]),
                "strong_bullish_alignment_15m_share": bool_share(strong["ema_alignment_15m"] == "bullish"),
                "weak_bullish_alignment_15m_share": bool_share(weak["ema_alignment_15m"] == "bullish"),
                "strong_bearish_alignment_15m_share": bool_share(strong["ema_alignment_15m"] == "bearish"),
                "weak_bearish_alignment_15m_share": bool_share(weak["ema_alignment_15m"] == "bearish"),
            }
        )
    return pd.DataFrame(rows)


def build_context_summary_text(context_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    strong = context_df.loc[context_df["entry_strength_group"] == "strong_entry"]
    medium = context_df.loc[context_df["entry_strength_group"] == "medium_entry"]
    weak = context_df.loc[context_df["entry_strength_group"] == "weak_entry"]

    strong_styles = strong["context_style_label"].value_counts(normalize=True)
    strong_top_style = strong_styles.index[0] if not strong_styles.empty else "n/a"
    strong_top_style_share = strong_styles.iloc[0] if not strong_styles.empty else np.nan

    long_strong = strong.loc[strong["side"] == "Long"]
    short_strong = strong.loc[strong["side"] == "Short"]
    long_top_style = top_style(long_strong)
    short_top_style = top_style(short_strong)

    strong_vs_weak_all = comparison_df.loc[comparison_df["subset"] == "all"].iloc[0] if not comparison_df.empty else pd.Series(dtype=object)
    eth_compare = comparison_df.loc[comparison_df["subset"] == "symbol_ETHUSDT"].iloc[0] if (comparison_df["subset"] == "symbol_ETHUSDT").any() else pd.Series(dtype=object)
    btc_compare = comparison_df.loc[comparison_df["subset"] == "symbol_BTCUSDT"].iloc[0] if (comparison_df["subset"] == "symbol_BTCUSDT").any() else pd.Series(dtype=object)
    long_compare = comparison_df.loc[comparison_df["subset"] == "side_Long"].iloc[0] if (comparison_df["subset"] == "side_Long").any() else pd.Series(dtype=object)
    short_compare = comparison_df.loc[comparison_df["subset"] == "side_Short"].iloc[0] if (comparison_df["subset"] == "side_Short").any() else pd.Series(dtype=object)

    breakout_bias = compare_values(
        strong_vs_weak_all.get("strong_breakout_share"),
        strong_vs_weak_all.get("weak_breakout_share"),
    )
    pullback_bias = compare_values(
        strong_vs_weak_all.get("strong_pullback_share"),
        strong_vs_weak_all.get("weak_pullback_share"),
    )
    reversal_bias = compare_values(
        strong_vs_weak_all.get("strong_reversal_like_share"),
        strong_vs_weak_all.get("weak_reversal_like_share"),
    )

    text = f"""
MARKET CONTEXT AT ENTRY
=======================

This section evaluates the market state at the moment of entry. It does not infer exact strategy rules; it only describes what the market looked like when strong and weak entries occurred.

Entry strength counts:
- Strong entries: {fmt_int(len(strong))}
- Medium entries: {fmt_int(len(medium))}
- Weak entries: {fmt_int(len(weak))}

Overall context observations:
- Most common context style among strong entries: {strong_top_style} ({pct(strong_top_style_share)} of strong entries).
- Strong entries vs weak entries show breakout/continuation tendency: {breakout_bias}.
- Strong entries vs weak entries show pullback tendency: {pullback_bias}.
- Strong entries vs weak entries show reversal-like tendency: {reversal_bias}.
- Strong entries median RSI 15m: {fmt_number(median_of(strong["rsi14_15m"]))}; weak entries median RSI 15m: {fmt_number(median_of(weak["rsi14_15m"]))}.
- Strong entries median pre-entry 15m move: {fmt_number(median_of(strong["return_pre_15m_pct"]))}%; weak entries: {fmt_number(median_of(weak["return_pre_15m_pct"]))}%.
- Strong entries median ATR% 15m: {fmt_number(median_of(strong["atr14_pct_15m"]))}; weak entries: {fmt_number(median_of(weak["atr14_pct_15m"]))}.
- Strong entries median relative volume 1m: {fmt_number(median_of(strong["relative_volume_1m"]))}; weak entries: {fmt_number(median_of(weak["relative_volume_1m"]))}.

By side:
- Strong long entries most often occur in contexts tagged as: {long_top_style}.
- Strong short entries most often occur in contexts tagged as: {short_top_style}.
- Long strong entries breakout share vs weak longs: {pct(long_compare.get("strong_breakout_share"))} vs {pct(long_compare.get("weak_breakout_share"))}.
- Short strong entries breakout share vs weak shorts: {pct(short_compare.get("strong_breakout_share"))} vs {pct(short_compare.get("weak_breakout_share"))}.

By symbol:
- ETHUSDT strong entries breakout share vs weak ETH entries: {pct(eth_compare.get("strong_breakout_share"))} vs {pct(eth_compare.get("weak_breakout_share"))}.
- BTCUSDT strong entries breakout share vs weak BTC entries: {pct(btc_compare.get("strong_breakout_share"))} vs {pct(btc_compare.get("weak_breakout_share"))}.
- ETHUSDT strong entries pullback share vs weak ETH entries: {pct(eth_compare.get("strong_pullback_share"))} vs {pct(eth_compare.get("weak_pullback_share"))}.
- BTCUSDT strong entries pullback share vs weak BTC entries: {pct(btc_compare.get("strong_pullback_share"))} vs {pct(btc_compare.get("weak_pullback_share"))}.

Human summary:
- The strongest entries occur more often in market regimes labeled: {strong_top_style}.
- Based on the simple regime labels, the profile looks more {dominant_style_label(strong_top_style, breakout_bias, pullback_bias, reversal_bias)} than purely random.
- Long and short entries do not appear in identical contexts: long strong entries skew toward {long_top_style}, while short strong entries skew toward {short_top_style}.
- These conclusions are about market context after aligning indicators to the last completed candle before entry; they do not imply a specific indicator-based strategy.
""".strip()
    return text + "\n"


def compare_values(strong_value: float, weak_value: float) -> str:
    if pd.isna(strong_value) or pd.isna(weak_value):
        return "unclear"
    if strong_value > weak_value + 0.10:
        return "more often"
    if weak_value > strong_value + 0.10:
        return "less often"
    return "about as often"


def dominant_style_label(top_style: str, breakout_bias: str, pullback_bias: str, reversal_bias: str) -> str:
    if top_style in {"breakout_continuation", "continuation_like"}:
        return "breakout/continuation-oriented"
    if top_style == "pullback":
        return "pullback-oriented"
    if top_style == "reversal_like":
        return "reversal-oriented"
    if breakout_bias == "more often":
        return "breakout/continuation-oriented"
    if pullback_bias == "more often":
        return "pullback-oriented"
    if reversal_bias == "more often":
        return "reversal-oriented"
    return "mixed-context"


def top_style(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "n/a"
    counts = frame["context_style_label"].value_counts()
    if counts.empty:
        return "n/a"
    return str(counts.index[0])


def create_market_context_plot(context_df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    groups = ["strong_entry", "medium_entry", "weak_entry"]
    for axis, (feature, label) in zip(axes.flatten(), CONTEXT_PLOT_FEATURES):
        data = [
            context_df.loc[context_df["entry_strength_group"] == group, feature].dropna().to_numpy()
            for group in groups
        ]
        axis.boxplot(data, tick_labels=groups)
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.2)

    fig.tight_layout()
    output_path = plots_dir / "context_feature_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", output_path.name)


def median_of(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.median())


def bool_share(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())
