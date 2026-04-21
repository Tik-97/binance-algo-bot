from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
JSON_TRADES_PATH = BASE_DIR / "Another Trader" / "another_trader.json"
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_timing_score_pipeline"

WINDOW_BEFORE_MINUTES = 60
WINDOW_AFTER_MINUTES = 60
TOP_K_PER_TRADE = 5


FEATURES = [
    "ret_1m_pct",
    "ret_5m_pct",
    "ret_15m_pct",
    "ret_60m_pct",
    "rel_volume_1m",
    "upper_wick_ratio",
    "body_ratio",
    "dist_1h_high_pct",
    "dist_4h_high_pct",
    "close_above_ema20_pct",
]

FEATURE_WEIGHTS = {
    "ret_1m_pct": 0.90,
    "ret_5m_pct": 1.10,
    "ret_15m_pct": 1.10,
    "ret_60m_pct": 1.20,
    "rel_volume_1m": 1.00,
    "upper_wick_ratio": 0.70,
    "body_ratio": 0.70,
    "dist_1h_high_pct": 1.30,
    "dist_4h_high_pct": 1.10,
    "close_above_ema20_pct": 0.90,
}

CONTRAST_OUTPUT_PREFIX = "timing_score_v2"
RERANK_OUTPUT_PREFIX = "timing_score_v3"
RERANK_SHORTLIST_K = 10

MICRO_FEATURES = [
    "ret_1m_pct",
    "ret_3m_pct",
    "ret_5m_pct",
    "rel_volume_1m",
    "upper_wick_ratio",
    "body_ratio",
    "close_position_in_range",
    "range_pct",
    "dist_1h_high_pct",
    "dist_5m_high_pct",
    "close_above_ema20_pct",
]

MICRO_WEIGHTS = {
    "ret_1m_pct": 0.80,
    "ret_3m_pct": 1.00,
    "ret_5m_pct": 0.90,
    "rel_volume_1m": 0.80,
    "upper_wick_ratio": 1.10,
    "body_ratio": 1.00,
    "close_position_in_range": 1.10,
    "range_pct": 0.70,
    "dist_1h_high_pct": 1.20,
    "dist_5m_high_pct": 1.10,
    "close_above_ema20_pct": 0.80,
}


def load_candles(path: Path) -> pd.DataFrame:
    candles = pd.read_csv(path)
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        if column in candles.columns:
            candles[column] = pd.to_numeric(candles[column], errors="coerce")
    candles = candles.dropna(subset=["open_time", "open", "high", "low", "close"]).copy()
    return candles.sort_values("open_time").reset_index(drop=True)


def load_reference_trades(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    trades = pd.DataFrame(raw)
    trades = trades[
        (trades["symbol"] == "ETHUSDT")
        & (trades["side"] == "Short")
        & (trades["status"] == "All Closed")
    ].copy()
    trades["opened_time"] = pd.to_datetime(trades["opened"], unit="ms", utc=True, errors="coerce")
    trades["closed_time"] = pd.to_datetime(trades["closed"], unit="ms", utc=True, errors="coerce")
    trades["avgCost"] = pd.to_numeric(trades["avgCost"], errors="coerce")
    trades["avgClosePrice"] = pd.to_numeric(trades["avgClosePrice"], errors="coerce")
    trades["closingPnl"] = pd.to_numeric(trades["closingPnl"], errors="coerce")
    trades = trades.dropna(subset=["opened_time", "avgCost"]).copy()
    return trades.sort_values("opened_time").reset_index(drop=True)


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    frame["ema20"] = frame["close"].ewm(span=20, adjust=False).mean()

    frame["ret_1m_pct"] = frame["close"].pct_change(1) * 100.0
    frame["ret_3m_pct"] = frame["close"].pct_change(3) * 100.0
    frame["ret_5m_pct"] = frame["close"].pct_change(5) * 100.0
    frame["ret_15m_pct"] = frame["close"].pct_change(15) * 100.0
    frame["ret_60m_pct"] = frame["close"].pct_change(60) * 100.0

    if "volume" in frame.columns:
        frame["rel_volume_1m"] = frame["volume"] / frame["volume"].rolling(20, min_periods=20).mean()
    else:
        frame["rel_volume_1m"] = 1.0

    rng = (frame["high"] - frame["low"]).replace(0, np.nan)
    body = (frame["close"] - frame["open"]).abs()
    upper_wick = frame["high"] - frame[["open", "close"]].max(axis=1)

    frame["upper_wick_ratio"] = (upper_wick / rng).fillna(0.0)
    frame["body_ratio"] = (body / rng).fillna(0.0)
    frame["close_position_in_range"] = ((frame["close"] - frame["low"]) / rng).fillna(0.5)
    frame["range_pct"] = ((frame["high"] - frame["low"]) / frame["close"] * 100.0).fillna(0.0)

    prev_1h_high = frame["high"].shift(1).rolling(60, min_periods=60).max()
    prev_4h_high = frame["high"].shift(1).rolling(240, min_periods=240).max()
    prev_5m_high = frame["high"].shift(1).rolling(5, min_periods=5).max()
    frame["dist_1h_high_pct"] = (prev_1h_high - frame["close"]) / frame["close"] * 100.0
    frame["dist_4h_high_pct"] = (prev_4h_high - frame["close"]) / frame["close"] * 100.0
    frame["dist_5m_high_pct"] = (prev_5m_high - frame["close"]) / frame["close"] * 100.0
    frame["close_above_ema20_pct"] = (frame["close"] / frame["ema20"] - 1.0) * 100.0

    return frame


def anchor_reference_minutes(features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    open_times = features["open_time"]

    for _, trade in trades.iterrows():
        idx = open_times.searchsorted(trade["opened_time"], side="right") - 1
        if idx < 0:
            continue

        feature_row = features.iloc[idx]
        payload = {
            "trade_opened_time": trade["opened_time"],
            "trade_closed_time": trade["closed_time"],
            "trade_avg_cost": trade["avgCost"],
            "trade_avg_close_price": trade["avgClosePrice"],
            "trade_closing_pnl": trade["closingPnl"],
            "anchor_candle_time": feature_row["open_time"],
            "anchor_offset_seconds": (trade["opened_time"] - feature_row["open_time"]).total_seconds(),
        }
        for feature in FEATURES:
            payload[feature] = feature_row.get(feature, np.nan)
        rows.append(payload)

    anchored = pd.DataFrame(rows)
    return anchored.dropna(subset=FEATURES).reset_index(drop=True)


def build_profile(anchored: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    medians = anchored[FEATURES].median(numeric_only=True)
    q75 = anchored[FEATURES].quantile(0.75, numeric_only=True)
    q25 = anchored[FEATURES].quantile(0.25, numeric_only=True)
    spreads = (q75 - q25).replace(0, np.nan)
    spreads = spreads.fillna(anchored[FEATURES].std(numeric_only=True))
    spreads = spreads.replace(0, 1.0).fillna(1.0)
    return medians, spreads


def compute_score(frame: pd.DataFrame, medians: pd.Series, spreads: pd.Series) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0

    for feature in FEATURES:
        weight = FEATURE_WEIGHTS.get(feature, 1.0)
        scale = float(spreads.get(feature, 1.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        deviation = (pd.to_numeric(frame[feature], errors="coerce") - float(medians[feature])).abs() / scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.nan, index=frame.index)
    return weighted_sum / total_weight


def build_window_contrast_profile(features: pd.DataFrame, anchored: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    deltas: list[dict] = []

    for _, trade in anchored.iterrows():
        start_time = trade["trade_opened_time"] - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
        end_time = trade["trade_opened_time"] + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)
        window = features[
            (features["open_time"] >= start_time) & (features["open_time"] <= end_time)
        ].dropna(subset=FEATURES).copy()
        if window.empty:
            continue

        anchor = window.loc[window["open_time"].eq(trade["anchor_candle_time"])].head(1)
        negatives = window.loc[~window["open_time"].eq(trade["anchor_candle_time"])].copy()
        if anchor.empty or negatives.empty:
            continue

        window_medians = negatives[FEATURES].median(numeric_only=True)
        window_spreads = (negatives[FEATURES].quantile(0.75) - negatives[FEATURES].quantile(0.25)).replace(0, np.nan)
        window_spreads = window_spreads.fillna(negatives[FEATURES].std(numeric_only=True))
        window_spreads = window_spreads.replace(0, 1.0).fillna(1.0)

        row = {"trade_opened_time": trade["trade_opened_time"]}
        for feature in FEATURES:
            row[f"{feature}_delta"] = (
                float(anchor.iloc[0][feature]) - float(window_medians[feature])
            ) / float(window_spreads[feature])
        deltas.append(row)

    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    delta_columns = [f"{feature}_delta" for feature in FEATURES]
    target_deltas = delta_df[delta_columns].median(numeric_only=True)
    delta_spreads = (
        delta_df[delta_columns].quantile(0.75) - delta_df[delta_columns].quantile(0.25)
    ).replace(0, np.nan)
    delta_spreads = delta_spreads.fillna(delta_df[delta_columns].std(numeric_only=True))
    delta_spreads = delta_spreads.replace(0, 1.0).fillna(1.0)
    return target_deltas, delta_spreads


def compute_window_contrast_score(
    window: pd.DataFrame,
    target_deltas: pd.Series,
    delta_scales: pd.Series,
) -> pd.Series:
    negatives = window.copy()
    window_medians = negatives[FEATURES].median(numeric_only=True)
    window_spreads = (negatives[FEATURES].quantile(0.75) - negatives[FEATURES].quantile(0.25)).replace(0, np.nan)
    window_spreads = window_spreads.fillna(negatives[FEATURES].std(numeric_only=True))
    window_spreads = window_spreads.replace(0, 1.0).fillna(1.0)

    weighted_sum = pd.Series(0.0, index=window.index, dtype=float)
    total_weight = 0.0
    for feature in FEATURES:
        target = float(target_deltas.get(f"{feature}_delta", 0.0))
        scale = float(delta_scales.get(f"{feature}_delta", 1.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        current_delta = (pd.to_numeric(window[feature], errors="coerce") - float(window_medians[feature])) / float(window_spreads[feature])
        deviation = (current_delta - target).abs() / scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = FEATURE_WEIGHTS.get(feature, 1.0)
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight

    if total_weight <= 0:
        return pd.Series(np.nan, index=window.index)
    return weighted_sum / total_weight


def build_candidate_windows(
    features: pd.DataFrame,
    anchored: pd.DataFrame,
    medians: pd.Series,
    spreads: pd.Series,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, trade in anchored.iterrows():
        start_time = trade["trade_opened_time"] - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
        end_time = trade["trade_opened_time"] + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)

        window = features[
            (features["open_time"] >= start_time) & (features["open_time"] <= end_time)
        ].copy()
        if window.empty:
            continue

        window = window.dropna(subset=FEATURES).copy()
        if window.empty:
            continue

        window["trade_opened_time"] = trade["trade_opened_time"]
        window["anchor_candle_time"] = trade["anchor_candle_time"]
        window["minute_distance_to_trade"] = (
            (window["open_time"] - trade["trade_opened_time"]).dt.total_seconds() / 60.0
        )
        window["abs_minute_distance_to_trade"] = window["minute_distance_to_trade"].abs()
        window["is_anchor_candle"] = window["open_time"].eq(trade["anchor_candle_time"])
        window["timing_score"] = compute_score(window, medians, spreads)
        window = window.sort_values(
            ["timing_score", "abs_minute_distance_to_trade"],
            ascending=[False, True],
        ).reset_index(drop=True)
        window["rank_in_window"] = np.arange(1, len(window) + 1)
        window["score_percentile_in_window"] = (
            len(window) - window["rank_in_window"] + 1
        ) / len(window)

        rows.append(window)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_candidate_windows_v2(
    features: pd.DataFrame,
    anchored: pd.DataFrame,
    target_deltas: pd.Series,
    delta_scales: pd.Series,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for _, trade in anchored.iterrows():
        start_time = trade["trade_opened_time"] - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
        end_time = trade["trade_opened_time"] + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)
        window = features[
            (features["open_time"] >= start_time) & (features["open_time"] <= end_time)
        ].dropna(subset=FEATURES).copy()
        if window.empty:
            continue

        window["trade_opened_time"] = trade["trade_opened_time"]
        window["anchor_candle_time"] = trade["anchor_candle_time"]
        window["minute_distance_to_trade"] = (
            (window["open_time"] - trade["trade_opened_time"]).dt.total_seconds() / 60.0
        )
        window["abs_minute_distance_to_trade"] = window["minute_distance_to_trade"].abs()
        window["is_anchor_candle"] = window["open_time"].eq(trade["anchor_candle_time"])
        window["timing_score_v2"] = compute_window_contrast_score(window, target_deltas, delta_scales)
        window = window.sort_values(
            ["timing_score_v2", "abs_minute_distance_to_trade"],
            ascending=[False, True],
        ).reset_index(drop=True)
        window["rank_in_window"] = np.arange(1, len(window) + 1)
        window["score_percentile_in_window"] = (len(window) - window["rank_in_window"] + 1) / len(window)
        rows.append(window)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_trade_level_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    summary_rows: list[dict] = []
    for trade_time, window in candidates.groupby("trade_opened_time", sort=True):
        window = window.sort_values(["rank_in_window"]).reset_index(drop=True)
        anchor = window.loc[window["is_anchor_candle"]].head(1)
        best = window.head(1)
        topk = window.head(TOP_K_PER_TRADE)

        if anchor.empty or best.empty:
            continue

        anchor_row = anchor.iloc[0]
        best_row = best.iloc[0]
        summary_rows.append(
            {
                "trade_opened_time": trade_time,
                "anchor_candle_time": anchor_row["open_time"],
                "anchor_rank_in_window": int(anchor_row["rank_in_window"]),
                "anchor_score": float(anchor_row["timing_score"]),
                "anchor_score_percentile": float(anchor_row["score_percentile_in_window"]),
                "best_candidate_time": best_row["open_time"],
                "best_candidate_score": float(best_row["timing_score"]),
                "best_candidate_minute_distance": float(best_row["minute_distance_to_trade"]),
                "best_candidate_abs_minute_distance": float(best_row["abs_minute_distance_to_trade"]),
                "anchor_in_top_1": bool(int(anchor_row["rank_in_window"]) <= 1),
                "anchor_in_top_3": bool(int(anchor_row["rank_in_window"]) <= 3),
                "anchor_in_top_5": bool(int(anchor_row["rank_in_window"]) <= 5),
                "best_within_5m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 5.0),
                "best_within_15m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 15.0),
                "top5_contains_within_5m": bool((topk["abs_minute_distance_to_trade"] <= 5.0).any()),
                "top5_contains_within_15m": bool((topk["abs_minute_distance_to_trade"] <= 15.0).any()),
            }
        )

    return pd.DataFrame(summary_rows)


def build_trade_level_summary_for_score(candidates: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    summary_rows: list[dict] = []
    for trade_time, window in candidates.groupby("trade_opened_time", sort=True):
        window = window.sort_values(["rank_in_window"]).reset_index(drop=True)
        anchor = window.loc[window["is_anchor_candle"]].head(1)
        best = window.head(1)
        topk = window.head(TOP_K_PER_TRADE)
        if anchor.empty or best.empty:
            continue

        anchor_row = anchor.iloc[0]
        best_row = best.iloc[0]
        summary_rows.append(
            {
                "trade_opened_time": trade_time,
                "anchor_candle_time": anchor_row["open_time"],
                "anchor_rank_in_window": int(anchor_row["rank_in_window"]),
                "anchor_score": float(anchor_row[score_column]),
                "anchor_score_percentile": float(anchor_row["score_percentile_in_window"]),
                "best_candidate_time": best_row["open_time"],
                "best_candidate_score": float(best_row[score_column]),
                "best_candidate_minute_distance": float(best_row["minute_distance_to_trade"]),
                "best_candidate_abs_minute_distance": float(best_row["abs_minute_distance_to_trade"]),
                "anchor_in_top_1": bool(int(anchor_row["rank_in_window"]) <= 1),
                "anchor_in_top_3": bool(int(anchor_row["rank_in_window"]) <= 3),
                "anchor_in_top_5": bool(int(anchor_row["rank_in_window"]) <= 5),
                "best_within_5m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 5.0),
                "best_within_15m": bool(float(best_row["abs_minute_distance_to_trade"]) <= 15.0),
                "top5_contains_within_5m": bool((topk["abs_minute_distance_to_trade"] <= 5.0).any()),
                "top5_contains_within_15m": bool((topk["abs_minute_distance_to_trade"] <= 15.0).any()),
            }
        )
    return pd.DataFrame(summary_rows)


def build_shortlist_contrast_profile(features: pd.DataFrame, anchored: pd.DataFrame, shortlist_k: int) -> tuple[pd.Series, pd.Series]:
    deltas: list[dict] = []
    for _, trade in anchored.iterrows():
        start_time = trade["trade_opened_time"] - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
        end_time = trade["trade_opened_time"] + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)
        window = features[
            (features["open_time"] >= start_time) & (features["open_time"] <= end_time)
        ].dropna(subset=MICRO_FEATURES).copy()
        if window.empty:
            continue

        window["minute_distance_to_trade"] = ((window["open_time"] - trade["trade_opened_time"]).dt.total_seconds() / 60.0)
        window["abs_minute_distance_to_trade"] = window["minute_distance_to_trade"].abs()
        shortlist = window.sort_values(["abs_minute_distance_to_trade"]).head(shortlist_k * 3).copy()
        anchor = shortlist.loc[shortlist["open_time"].eq(trade["anchor_candle_time"])].head(1)
        negatives = shortlist.loc[~shortlist["open_time"].eq(trade["anchor_candle_time"])].copy()
        if anchor.empty or negatives.empty:
            continue

        med = negatives[MICRO_FEATURES].median(numeric_only=True)
        spread = (negatives[MICRO_FEATURES].quantile(0.75) - negatives[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
        spread = spread.fillna(negatives[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
        row = {"trade_opened_time": trade["trade_opened_time"]}
        for feature in MICRO_FEATURES:
            row[f"{feature}_delta"] = (float(anchor.iloc[0][feature]) - float(med[feature])) / float(spread[feature])
        deltas.append(row)

    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    cols = [f"{feature}_delta" for feature in MICRO_FEATURES]
    target = delta_df[cols].median(numeric_only=True)
    scale = (delta_df[cols].quantile(0.75) - delta_df[cols].quantile(0.25)).replace(0, np.nan)
    scale = scale.fillna(delta_df[cols].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
    return target, scale


def compute_shortlist_rerank_score(shortlist: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.Series:
    med = shortlist[MICRO_FEATURES].median(numeric_only=True)
    spread = (shortlist[MICRO_FEATURES].quantile(0.75) - shortlist[MICRO_FEATURES].quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(shortlist[MICRO_FEATURES].std(numeric_only=True)).replace(0, 1.0).fillna(1.0)
    weighted_sum = pd.Series(0.0, index=shortlist.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        current_delta = (pd.to_numeric(shortlist[feature], errors="coerce") - float(med[feature])) / float(spread[feature])
        target = float(target_deltas.get(f"{feature}_delta", 0.0))
        scale = float(delta_scales.get(f"{feature}_delta", 1.0))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        deviation = (current_delta - target).abs() / scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = MICRO_WEIGHTS.get(feature, 1.0)
        weighted_sum = weighted_sum.add(feature_score * weight, fill_value=0.0)
        total_weight += weight
    if total_weight <= 0:
        return pd.Series(np.nan, index=shortlist.index)
    return weighted_sum / total_weight


def build_candidate_windows_v3(features: pd.DataFrame, anchored: pd.DataFrame, target_deltas: pd.Series, delta_scales: pd.Series) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, trade in anchored.iterrows():
        start_time = trade["trade_opened_time"] - pd.Timedelta(minutes=WINDOW_BEFORE_MINUTES)
        end_time = trade["trade_opened_time"] + pd.Timedelta(minutes=WINDOW_AFTER_MINUTES)
        window = features[
            (features["open_time"] >= start_time) & (features["open_time"] <= end_time)
        ].dropna(subset=MICRO_FEATURES).copy()
        if window.empty:
            continue

        window["trade_opened_time"] = trade["trade_opened_time"]
        window["anchor_candle_time"] = trade["anchor_candle_time"]
        window["minute_distance_to_trade"] = ((window["open_time"] - trade["trade_opened_time"]).dt.total_seconds() / 60.0)
        window["abs_minute_distance_to_trade"] = window["minute_distance_to_trade"].abs()
        window["is_anchor_candle"] = window["open_time"].eq(trade["anchor_candle_time"])

        shortlist = window.sort_values(["abs_minute_distance_to_trade"]).head(RERANK_SHORTLIST_K).copy()
        shortlist["timing_score_v3"] = compute_shortlist_rerank_score(shortlist, target_deltas, delta_scales)
        shortlist = shortlist.sort_values(["timing_score_v3", "abs_minute_distance_to_trade"], ascending=[False, True]).reset_index(drop=True)
        shortlist["rank_in_window"] = np.arange(1, len(shortlist) + 1)
        shortlist["score_percentile_in_window"] = (len(shortlist) - shortlist["rank_in_window"] + 1) / len(shortlist)
        rows.append(shortlist)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_summary_text(
    anchored: pd.DataFrame,
    medians: pd.Series,
    spreads: pd.Series,
    trade_summary: pd.DataFrame,
) -> None:
    if trade_summary.empty:
        text = "No timing-score candidates were produced.\n"
        (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
        return

    lines = [
        "ETH SHORT TIMING SCORE PIPELINE",
        "===============================",
        "",
        f"reference_trade_count = {len(anchored)}",
        f"window_before_minutes = {WINDOW_BEFORE_MINUTES}",
        f"window_after_minutes = {WINDOW_AFTER_MINUTES}",
        "",
        "Window ranking quality",
        "----------------------",
        f"anchor_top1_share = {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%",
        f"anchor_top3_share = {trade_summary['anchor_in_top_3'].mean() * 100.0:.2f}%",
        f"anchor_top5_share = {trade_summary['anchor_in_top_5'].mean() * 100.0:.2f}%",
        f"best_within_5m_share = {trade_summary['best_within_5m'].mean() * 100.0:.2f}%",
        f"best_within_15m_share = {trade_summary['best_within_15m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_5m_share = {trade_summary['top5_contains_within_5m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_15m_share = {trade_summary['top5_contains_within_15m'].mean() * 100.0:.2f}%",
        f"median_anchor_rank = {trade_summary['anchor_rank_in_window'].median():.2f}",
        f"median_anchor_percentile = {trade_summary['anchor_score_percentile'].median() * 100.0:.2f}%",
        "",
        "Reference medians",
        "-----------------",
    ]

    for feature in FEATURES:
        lines.append(
            f"{feature}: median={float(medians[feature]):.4f}, scale={float(spreads[feature]):.4f}, weight={FEATURE_WEIGHTS.get(feature, 1.0):.2f}"
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_text_v2(
    target_deltas: pd.Series,
    delta_scales: pd.Series,
    trade_summary: pd.DataFrame,
) -> None:
    if trade_summary.empty:
        text = "No timing-score-v2 candidates were produced.\n"
        (OUTPUT_DIR / f"{CONTRAST_OUTPUT_PREFIX}_summary.txt").write_text(text, encoding="utf-8")
        return

    lines = [
        "ETH SHORT TIMING SCORE PIPELINE V2",
        "==================================",
        "",
        f"reference_trade_count = {len(trade_summary)}",
        f"window_before_minutes = {WINDOW_BEFORE_MINUTES}",
        f"window_after_minutes = {WINDOW_AFTER_MINUTES}",
        "",
        "Window ranking quality",
        "----------------------",
        f"anchor_top1_share = {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%",
        f"anchor_top3_share = {trade_summary['anchor_in_top_3'].mean() * 100.0:.2f}%",
        f"anchor_top5_share = {trade_summary['anchor_in_top_5'].mean() * 100.0:.2f}%",
        f"best_within_5m_share = {trade_summary['best_within_5m'].mean() * 100.0:.2f}%",
        f"best_within_15m_share = {trade_summary['best_within_15m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_5m_share = {trade_summary['top5_contains_within_5m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_15m_share = {trade_summary['top5_contains_within_15m'].mean() * 100.0:.2f}%",
        f"median_anchor_rank = {trade_summary['anchor_rank_in_window'].median():.2f}",
        f"median_anchor_percentile = {trade_summary['anchor_score_percentile'].median() * 100.0:.2f}%",
        "",
        "Learned Anchor-vs-Window Deltas",
        "-------------------------------",
    ]
    for feature in FEATURES:
        lines.append(
            f"{feature}: target_delta={float(target_deltas.get(f'{feature}_delta', np.nan)):.4f}, delta_scale={float(delta_scales.get(f'{feature}_delta', np.nan)):.4f}, weight={FEATURE_WEIGHTS.get(feature, 1.0):.2f}"
        )
    (OUTPUT_DIR / f"{CONTRAST_OUTPUT_PREFIX}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_text_v3(target_deltas: pd.Series, delta_scales: pd.Series, trade_summary: pd.DataFrame) -> None:
    if trade_summary.empty:
        (OUTPUT_DIR / f"{RERANK_OUTPUT_PREFIX}_summary.txt").write_text("No timing-score-v3 candidates were produced.\n", encoding="utf-8")
        return
    lines = [
        "ETH SHORT TIMING SCORE PIPELINE V3",
        "==================================",
        "",
        f"reference_trade_count = {len(trade_summary)}",
        f"window_before_minutes = {WINDOW_BEFORE_MINUTES}",
        f"window_after_minutes = {WINDOW_AFTER_MINUTES}",
        f"rerank_shortlist_k = {RERANK_SHORTLIST_K}",
        "",
        "Window ranking quality",
        "----------------------",
        f"anchor_top1_share = {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%",
        f"anchor_top3_share = {trade_summary['anchor_in_top_3'].mean() * 100.0:.2f}%",
        f"anchor_top5_share = {trade_summary['anchor_in_top_5'].mean() * 100.0:.2f}%",
        f"best_within_5m_share = {trade_summary['best_within_5m'].mean() * 100.0:.2f}%",
        f"best_within_15m_share = {trade_summary['best_within_15m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_5m_share = {trade_summary['top5_contains_within_5m'].mean() * 100.0:.2f}%",
        f"top5_contains_within_15m_share = {trade_summary['top5_contains_within_15m'].mean() * 100.0:.2f}%",
        f"median_anchor_rank = {trade_summary['anchor_rank_in_window'].median():.2f}",
        "",
        "Learned Shortlist Deltas",
        "------------------------",
    ]
    for feature in MICRO_FEATURES:
        lines.append(
            f"{feature}: target_delta={float(target_deltas.get(f'{feature}_delta', np.nan)):.4f}, delta_scale={float(delta_scales.get(f'{feature}_delta', np.nan)):.4f}, weight={MICRO_WEIGHTS.get(feature, 1.0):.2f}"
        )
    (OUTPUT_DIR / f"{RERANK_OUTPUT_PREFIX}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candles = load_candles(CANDLES_1M_PATH)
    trades = load_reference_trades(JSON_TRADES_PATH)
    features = build_features(candles)
    anchored = anchor_reference_minutes(features, trades)
    if anchored.empty:
        raise ValueError("No anchored reference trades were available for the timing score pipeline.")

    medians, spreads = build_profile(anchored)
    candidates = build_candidate_windows(features, anchored, medians, spreads)
    trade_summary = build_trade_level_summary(candidates)
    target_deltas, delta_scales = build_window_contrast_profile(features, anchored)
    candidates_v2 = build_candidate_windows_v2(features, anchored, target_deltas, delta_scales)
    trade_summary_v2 = build_trade_level_summary_for_score(candidates_v2, "timing_score_v2")
    target_deltas_v3, delta_scales_v3 = build_shortlist_contrast_profile(features, anchored, RERANK_SHORTLIST_K)
    candidates_v3 = build_candidate_windows_v3(features, anchored, target_deltas_v3, delta_scales_v3)
    trade_summary_v3 = build_trade_level_summary_for_score(candidates_v3, "timing_score_v3")

    anchored.to_csv(OUTPUT_DIR / "reference_anchor_features.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "timing_score_candidates.csv", index=False)
    trade_summary.to_csv(OUTPUT_DIR / "timing_score_trade_summary.csv", index=False)
    candidates_v2.to_csv(OUTPUT_DIR / f"{CONTRAST_OUTPUT_PREFIX}_candidates.csv", index=False)
    trade_summary_v2.to_csv(OUTPUT_DIR / f"{CONTRAST_OUTPUT_PREFIX}_trade_summary.csv", index=False)
    candidates_v3.to_csv(OUTPUT_DIR / f"{RERANK_OUTPUT_PREFIX}_candidates.csv", index=False)
    trade_summary_v3.to_csv(OUTPUT_DIR / f"{RERANK_OUTPUT_PREFIX}_trade_summary.csv", index=False)

    top_candidates = (
        candidates.sort_values(["trade_opened_time", "rank_in_window"])
        .groupby("trade_opened_time", as_index=False)
        .head(TOP_K_PER_TRADE)
        .reset_index(drop=True)
    )
    top_candidates.to_csv(OUTPUT_DIR / "timing_score_top_candidates.csv", index=False)
    top_candidates_v2 = (
        candidates_v2.sort_values(["trade_opened_time", "rank_in_window"])
        .groupby("trade_opened_time", as_index=False)
        .head(TOP_K_PER_TRADE)
        .reset_index(drop=True)
    )
    top_candidates_v2.to_csv(OUTPUT_DIR / f"{CONTRAST_OUTPUT_PREFIX}_top_candidates.csv", index=False)
    top_candidates_v3 = (
        candidates_v3.sort_values(["trade_opened_time", "rank_in_window"])
        .groupby("trade_opened_time", as_index=False)
        .head(TOP_K_PER_TRADE)
        .reset_index(drop=True)
    )
    top_candidates_v3.to_csv(OUTPUT_DIR / f"{RERANK_OUTPUT_PREFIX}_top_candidates.csv", index=False)

    write_summary_text(anchored, medians, spreads, trade_summary)
    write_summary_text_v2(target_deltas, delta_scales, trade_summary_v2)
    write_summary_text_v3(target_deltas_v3, delta_scales_v3, trade_summary_v3)

    print(f"Anchored reference trades: {len(anchored)}")
    print(f"Candidate rows: {len(candidates)}")
    if not trade_summary.empty:
        print(f"Anchor top-1 share: {trade_summary['anchor_in_top_1'].mean() * 100.0:.2f}%")
        print(f"Anchor top-5 share: {trade_summary['anchor_in_top_5'].mean() * 100.0:.2f}%")
        print(f"Best candidate within 5m share: {trade_summary['best_within_5m'].mean() * 100.0:.2f}%")
    if not trade_summary_v2.empty:
        print(f"V2 anchor top-1 share: {trade_summary_v2['anchor_in_top_1'].mean() * 100.0:.2f}%")
        print(f"V2 anchor top-5 share: {trade_summary_v2['anchor_in_top_5'].mean() * 100.0:.2f}%")
        print(f"V2 best candidate within 5m share: {trade_summary_v2['best_within_5m'].mean() * 100.0:.2f}%")
    if not trade_summary_v3.empty:
        print(f"V3 anchor top-1 share: {trade_summary_v3['anchor_in_top_1'].mean() * 100.0:.2f}%")
        print(f"V3 anchor top-5 share: {trade_summary_v3['anchor_in_top_5'].mean() * 100.0:.2f}%")
        print(f"V3 best candidate within 5m share: {trade_summary_v3['best_within_5m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
