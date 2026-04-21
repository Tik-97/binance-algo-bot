from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CANDLES_1M_PATH = BASE_DIR / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
JSON_TRADES_PATH = BASE_DIR / "Another Trader" / "another_trader.json"
OUTPUT_DIR = BASE_DIR / "fun_backtests" / "eth_short_candidate_engine_v1"

SESSION_START_HOUR_UTC = 13
SESSION_END_HOUR_UTC = 16
DEDUP_COOLDOWN_MINUTES = 15

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

TARGET_DELTAS = {
    "ret_1m_pct": 0.3092,
    "ret_3m_pct": 0.6735,
    "ret_5m_pct": 0.7186,
    "rel_volume_1m": 0.7294,
    "upper_wick_ratio": 0.1771,
    "body_ratio": 0.1320,
    "close_position_in_range": 0.0563,
    "range_pct": 0.5256,
    "dist_1h_high_pct": -0.1473,
    "dist_5m_high_pct": -0.2171,
    "close_above_ema20_pct": 0.3606,
}

DELTA_SCALES = {
    "ret_1m_pct": 1.3136,
    "ret_3m_pct": 1.8656,
    "ret_5m_pct": 2.1932,
    "rel_volume_1m": 1.4532,
    "upper_wick_ratio": 0.9925,
    "body_ratio": 0.9042,
    "close_position_in_range": 1.0465,
    "range_pct": 1.3641,
    "dist_1h_high_pct": 1.3352,
    "dist_5m_high_pct": 0.8480,
    "close_above_ema20_pct": 1.5558,
}


def load_candles(path: Path) -> pd.DataFrame:
    candles = pd.read_csv(path)
    candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
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
    trades["avgCost"] = pd.to_numeric(trades["avgCost"], errors="coerce")
    trades["closingPnl"] = pd.to_numeric(trades["closingPnl"], errors="coerce")
    trades["ref_result"] = np.where(trades["closingPnl"] > 0, "win", np.where(trades["closingPnl"] < 0, "loss", "flat"))
    return trades.dropna(subset=["opened_time", "avgCost"]).sort_values("opened_time").reset_index(drop=True)


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    frame["ema20"] = frame["close"].ewm(span=20, adjust=False).mean()
    frame["ret_1m_pct"] = frame["close"].pct_change(1) * 100.0
    frame["ret_3m_pct"] = frame["close"].pct_change(3) * 100.0
    frame["ret_5m_pct"] = frame["close"].pct_change(5) * 100.0
    frame["ret_15m_pct"] = frame["close"].pct_change(15) * 100.0
    frame["ret_60m_pct"] = frame["close"].pct_change(60) * 100.0
    frame["rel_volume_1m"] = frame["volume"] / frame["volume"].rolling(20, min_periods=20).mean()

    rng = (frame["high"] - frame["low"]).replace(0, np.nan)
    body = (frame["close"] - frame["open"]).abs()
    upper_wick = frame["high"] - frame[["open", "close"]].max(axis=1)

    frame["upper_wick_ratio"] = (upper_wick / rng).fillna(0.0)
    frame["body_ratio"] = (body / rng).fillna(0.0)
    frame["close_position_in_range"] = ((frame["close"] - frame["low"]) / rng).fillna(0.5)
    frame["range_pct"] = ((frame["high"] - frame["low"]) / frame["close"] * 100.0).fillna(0.0)

    prev_5m_high = frame["high"].shift(1).rolling(5, min_periods=5).max()
    prev_1h_high = frame["high"].shift(1).rolling(60, min_periods=60).max()
    prev_4h_high = frame["high"].shift(1).rolling(240, min_periods=240).max()
    frame["dist_5m_high_pct"] = (prev_5m_high - frame["close"]) / frame["close"] * 100.0
    frame["dist_1h_high_pct"] = (prev_1h_high - frame["close"]) / frame["close"] * 100.0
    frame["dist_4h_high_pct"] = (prev_4h_high - frame["close"]) / frame["close"] * 100.0
    frame["close_above_ema20_pct"] = (frame["close"] / frame["ema20"] - 1.0) * 100.0
    frame["hour_utc"] = frame["open_time"].dt.hour
    return frame


def in_session(ts: pd.Timestamp) -> bool:
    return SESSION_START_HOUR_UTC <= ts.hour < SESSION_END_HOUR_UTC


def candidate_gate(row: pd.Series) -> bool:
    needed = [
        "ret_1m_pct", "ret_3m_pct", "ret_5m_pct", "ret_15m_pct", "ret_60m_pct",
        "rel_volume_1m", "upper_wick_ratio", "body_ratio", "dist_1h_high_pct",
        "dist_4h_high_pct", "dist_5m_high_pct", "close_above_ema20_pct",
    ]
    if any(pd.isna(row[c]) for c in needed):
        return False
    return (
        in_session(row["open_time"])
        and row["dist_4h_high_pct"] <= 0.80
        and row["dist_1h_high_pct"] <= 0.60
        and row["ret_60m_pct"] >= 0.20
        and row["ret_60m_pct"] <= 3.50
        and row["ret_15m_pct"] >= 0.10
        and row["ret_5m_pct"] >= -0.10
        and row["ret_5m_pct"] <= 1.80
        and row["close_above_ema20_pct"] >= -0.10
        and row["close_above_ema20_pct"] <= 2.50
        and row["rel_volume_1m"] <= 4.50
        and (row["upper_wick_ratio"] >= 0.05 or row["body_ratio"] <= 0.85)
    )


def score_candidates(frame: pd.DataFrame) -> pd.Series:
    weighted = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0
    for feature in MICRO_FEATURES:
        series = pd.to_numeric(frame[feature], errors="coerce")
        med = float(series.median())
        spread = float(series.quantile(0.75) - series.quantile(0.25))
        if not np.isfinite(spread) or spread <= 0:
            spread = float(series.std()) if np.isfinite(series.std()) and series.std() > 0 else 1.0
        current_delta = (series - med) / spread
        target = float(TARGET_DELTAS[feature])
        target_scale = float(DELTA_SCALES[feature])
        deviation = (current_delta - target).abs() / target_scale
        clipped = deviation.clip(lower=0.0, upper=3.0) / 3.0
        feature_score = 1.0 - clipped
        weight = float(MICRO_WEIGHTS[feature])
        weighted = weighted.add(feature_score * weight, fill_value=0.0)
        total_weight += weight
    return weighted / total_weight if total_weight > 0 else pd.Series(np.nan, index=frame.index)


def generate_candidates(features: pd.DataFrame) -> pd.DataFrame:
    candidates = features.loc[features.apply(candidate_gate, axis=1)].copy()
    if candidates.empty:
        return candidates
    candidates["candidate_score"] = score_candidates(candidates)
    candidates = candidates.sort_values("open_time").reset_index(drop=True)
    return candidates


def deduplicate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    kept: list[pd.Series] = []
    current_bucket: list[pd.Series] = []
    bucket_start: pd.Timestamp | None = None
    for _, row in candidates.sort_values("open_time").iterrows():
        now = row["open_time"]
        if bucket_start is None:
            bucket_start = now
            current_bucket = [row]
            continue
        if (now - bucket_start).total_seconds() / 60.0 <= DEDUP_COOLDOWN_MINUTES:
            current_bucket.append(row)
        else:
            best = max(current_bucket, key=lambda item: (float(item["candidate_score"]), -item["open_time"].value))
            kept.append(best)
            bucket_start = now
            current_bucket = [row]
    if current_bucket:
        best = max(current_bucket, key=lambda item: (float(item["candidate_score"]), -item["open_time"].value))
        kept.append(best)
    return pd.DataFrame(kept).sort_values("open_time").reset_index(drop=True)


def match_candidates_to_reference(candidates: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or ref.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, trade in ref.iterrows():
        window = candidates.copy()
        window["minute_distance"] = (window["open_time"] - trade["opened_time"]).dt.total_seconds() / 60.0
        window["abs_minute_distance"] = window["minute_distance"].abs()
        best = window.sort_values(["abs_minute_distance", "candidate_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "ref_opened_time": trade["opened_time"],
                "ref_avg_cost": trade["avgCost"],
                "ref_result": trade["ref_result"],
                "candidate_time": best["open_time"],
                "candidate_score": best["candidate_score"],
                "minute_distance": best["minute_distance"],
                "abs_minute_distance": best["abs_minute_distance"],
                "candidate_price": best["close"],
                "price_distance_pct": (float(best["close"]) - float(trade["avgCost"])) / float(trade["avgCost"]) * 100.0,
                "within_5m": float(best["abs_minute_distance"]) <= 5.0,
                "within_15m": float(best["abs_minute_distance"]) <= 15.0,
                "within_60m": float(best["abs_minute_distance"]) <= 60.0,
            }
        )
    return pd.DataFrame(rows)


def write_summary(candidates: pd.DataFrame, deduped: pd.DataFrame, matches: pd.DataFrame) -> None:
    lines = [
        "ETH SHORT CANDIDATE ENGINE V1",
        "=============================",
        "",
        f"raw_candidate_count = {len(candidates)}",
        f"deduped_candidate_count = {len(deduped)}",
        f"dedup_cooldown_minutes = {DEDUP_COOLDOWN_MINUTES}",
        "",
    ]
    if not matches.empty:
        lines.extend(
            [
                f"reference_trade_count = {len(matches)}",
                f"within_5m_share = {matches['within_5m'].mean() * 100.0:.2f}%",
                f"within_15m_share = {matches['within_15m'].mean() * 100.0:.2f}%",
                f"within_60m_share = {matches['within_60m'].mean() * 100.0:.2f}%",
                f"median_abs_minute_distance = {matches['abs_minute_distance'].median():.2f}",
                f"median_abs_price_distance_pct = {matches['price_distance_pct'].abs().median():.4f}",
            ]
        )
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candles = load_candles(CANDLES_1M_PATH)
    ref = load_reference_trades(JSON_TRADES_PATH)
    features = build_features(candles)
    candidates = generate_candidates(features)
    deduped = deduplicate_candidates(candidates)
    matches = match_candidates_to_reference(deduped, ref)

    features.head(0).to_csv(OUTPUT_DIR / "schema_features.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "raw_candidates.csv", index=False)
    deduped.to_csv(OUTPUT_DIR / "deduped_candidates.csv", index=False)
    matches.to_csv(OUTPUT_DIR / "candidate_matches.csv", index=False)
    write_summary(candidates, deduped, matches)

    print(f"Raw candidates: {len(candidates)}")
    print(f"Deduped candidates: {len(deduped)}")
    if not matches.empty:
        print(f"Within 5m share: {matches['within_5m'].mean() * 100.0:.2f}%")
        print(f"Within 15m share: {matches['within_15m'].mean() * 100.0:.2f}%")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
