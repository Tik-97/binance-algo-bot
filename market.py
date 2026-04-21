from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from config import (
    BINANCE_BACKOFF_SECONDS,
    BINANCE_MAX_RETRIES,
    EMA_INTERVAL,
    EMA_PERIOD,
    KLINES_CACHE_DIRNAME,
)
from utils import safe_divide


FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
NUMERIC_KLINE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]
INTERVAL_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass
class MarketFilterResult:
    stats: pd.DataFrame
    annotated_trades: pd.DataFrame
    note: str


def align_timestamp_for_interval(timestamp_ms: int, interval: str, mode: str = "floor") -> int:
    interval_ms = INTERVAL_TO_MS[interval]
    if mode == "floor":
        return (timestamp_ms // interval_ms) * interval_ms
    if mode == "ceil":
        return ((timestamp_ms + interval_ms - 1) // interval_ms) * interval_ms
    raise ValueError(f"Unsupported alignment mode: {mode}")


def cache_dir_for_klines(base_dir: Path | None = None) -> Path:
    root = base_dir or Path.cwd()
    cache_dir = root / KLINES_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def cache_path_for_symbol_interval(symbol: str, interval: str, base_dir: Path | None = None) -> Path:
    return cache_dir_for_klines(base_dir) / f"{symbol}_{interval}.csv"


def normalize_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS + ["open_time", "close_time"])

    normalized = frame.copy()
    for column in ["open_time_ms", "close_time_ms"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    for column in NUMERIC_KLINE_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["open_time_ms", "close_time_ms"]).copy()
    normalized["open_time"] = pd.to_datetime(normalized["open_time_ms"].astype("int64"), unit="ms", utc=True)
    normalized["close_time"] = pd.to_datetime(normalized["close_time_ms"].astype("int64"), unit="ms", utc=True)
    normalized = normalized.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms").reset_index(drop=True)
    return normalized


def load_cached_klines(symbol: str, interval: str, cache_base_dir: Path | None = None) -> pd.DataFrame:
    cache_path = cache_path_for_symbol_interval(symbol, interval, cache_base_dir)
    if not cache_path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS + ["open_time", "close_time"])
    frame = pd.read_csv(cache_path)
    return normalize_kline_frame(frame)


def save_cached_klines(symbol: str, interval: str, frame: pd.DataFrame, cache_base_dir: Path | None = None) -> None:
    cache_path = cache_path_for_symbol_interval(symbol, interval, cache_base_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_kline_frame(frame)
    normalized.to_csv(cache_path, index=False)


def _request_json_with_retry(url: str, logger: logging.Logger | None = None) -> list:
    delay = BINANCE_BACKOFF_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, BINANCE_MAX_RETRIES + 1):
        try:
            with urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            should_retry = exc.code in {418, 429, 500, 502, 503, 504}
            if logger:
                logger.warning(
                    "Binance HTTP error %s on attempt %s/%s for %s",
                    exc.code,
                    attempt,
                    BINANCE_MAX_RETRIES,
                    url,
                )
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            should_retry = True
            if logger:
                logger.warning(
                    "Binance network error on attempt %s/%s for %s: %s",
                    attempt,
                    BINANCE_MAX_RETRIES,
                    url,
                    exc,
                )

        if not should_retry or attempt == BINANCE_MAX_RETRIES:
            break
        time.sleep(delay)
        delay *= 2

    if last_error is not None:
        raise last_error
    raise RuntimeError("Binance request failed without a specific exception.")


def _fetch_futures_klines_remote(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    aligned_start = align_timestamp_for_interval(int(start_ms), interval, mode="floor")
    aligned_end = align_timestamp_for_interval(int(end_ms), interval, mode="ceil") + INTERVAL_TO_MS[interval]
    current_start = aligned_start
    all_rows: list[list] = []
    limit = 1500

    while current_start < aligned_end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": aligned_end,
            "limit": limit,
        }
        url = f"{FUTURES_KLINES_URL}?{urlencode(params)}"
        payload = _request_json_with_retry(url, logger=logger)

        if not isinstance(payload, list) or not payload:
            break

        all_rows.extend(payload)
        last_open_time = int(payload[-1][0])
        next_start = last_open_time + INTERVAL_TO_MS[interval]
        if next_start <= current_start:
            break
        current_start = next_start
        time.sleep(0.08)

    if logger:
        logger.info("Fetched %s %s candles for %s", len(all_rows), interval, symbol)

    if not all_rows:
        return pd.DataFrame(columns=KLINE_COLUMNS + ["open_time", "close_time"])
    return normalize_kline_frame(pd.DataFrame(all_rows, columns=KLINE_COLUMNS))


def fetch_futures_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
    interval: str = EMA_INTERVAL,
    logger: logging.Logger | None = None,
    use_cache: bool = True,
    cache_base_dir: Path | None = None,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    if end_ms <= start_ms:
        return pd.DataFrame(columns=KLINE_COLUMNS + ["open_time", "close_time"])

    aligned_start = align_timestamp_for_interval(int(start_ms), interval, mode="floor")
    aligned_end = align_timestamp_for_interval(int(end_ms), interval, mode="ceil") + INTERVAL_TO_MS[interval]

    if not use_cache:
        return _fetch_futures_klines_remote(symbol, aligned_start, aligned_end, interval, logger=logger)

    cached = load_cached_klines(symbol, interval, cache_base_dir)
    missing_parts: list[pd.DataFrame] = []

    if cached.empty:
        missing_parts.append(_fetch_futures_klines_remote(symbol, aligned_start, aligned_end, interval, logger=logger))
    else:
        cached_min = int(cached["open_time_ms"].min())
        cached_max = int(cached["open_time_ms"].max())
        if aligned_start < cached_min:
            missing_parts.append(
                _fetch_futures_klines_remote(symbol, aligned_start, cached_min - INTERVAL_TO_MS[interval], interval, logger=logger)
            )
        if aligned_end > cached_max + INTERVAL_TO_MS[interval]:
            missing_parts.append(
                _fetch_futures_klines_remote(symbol, cached_max + INTERVAL_TO_MS[interval], aligned_end, interval, logger=logger)
            )

    if missing_parts:
        frames_to_concat = [frame for frame in [cached, *missing_parts] if not frame.empty]
        if frames_to_concat:
            cached = normalize_kline_frame(pd.concat(frames_to_concat, ignore_index=True))
            save_cached_klines(symbol, interval, cached, cache_base_dir)

    subset = cached.loc[
        (cached["open_time_ms"] >= aligned_start - INTERVAL_TO_MS[interval])
        & (cached["open_time_ms"] <= aligned_end + INTERVAL_TO_MS[interval])
    ].reset_index(drop=True)
    return subset


def build_market_filter(trades_df: pd.DataFrame, logger: logging.Logger) -> MarketFilterResult:
    valid = trades_df.dropna(subset=["symbol", "opened_dt_utc"]).copy()
    if valid.empty:
        return MarketFilterResult(pd.DataFrame(), trades_df.copy(), "Market filter skipped: no valid timestamps.")

    start_ms = int((valid["opened_dt_utc"].min() - pd.Timedelta(hours=200)).timestamp() * 1000)
    end_ms = int((valid["opened_dt_utc"].max() + pd.Timedelta(hours=1)).timestamp() * 1000)
    annotated_parts: list[pd.DataFrame] = []

    for symbol in sorted(valid["symbol"].dropna().unique()):
        symbol_trades = valid.loc[valid["symbol"] == symbol].sort_values("opened_dt_utc").copy()
        klines = fetch_futures_klines(symbol, start_ms, end_ms, interval=EMA_INTERVAL, logger=logger)
        if klines.empty:
            symbol_trades["trend_filter_pass"] = pd.NA
            symbol_trades["trend_close"] = np.nan
            symbol_trades["trend_ema50"] = np.nan
            symbol_trades["trend_ema50_slope"] = np.nan
            annotated_parts.append(symbol_trades)
            continue

        klines["ema50"] = klines["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        klines["ema50_slope"] = klines["ema50"].diff()

        merged = pd.merge_asof(
            symbol_trades.sort_values("opened_dt_utc"),
            klines[["open_time", "close", "ema50", "ema50_slope"]].sort_values("open_time"),
            left_on="opened_dt_utc",
            right_on="open_time",
            direction="backward",
        )

        is_long = merged["side"].str.lower() == "long"
        is_short = merged["side"].str.lower() == "short"

        merged["trend_filter_pass"] = np.where(
            is_long,
            (merged["close"] >= merged["ema50"]) & (merged["ema50_slope"] >= 0),
            np.where(
                is_short,
                (merged["close"] <= merged["ema50"]) & (merged["ema50_slope"] <= 0),
                False,
            ),
        )
        merged = merged.rename(
            columns={
                "close": "trend_close",
                "ema50": "trend_ema50",
                "ema50_slope": "trend_ema50_slope",
            }
        )
        annotated_parts.append(merged.drop(columns=["open_time"]))

    annotated = pd.concat(annotated_parts, ignore_index=True).sort_values("trade_number").reset_index(drop=True)

    passed = annotated.loc[annotated["trend_filter_pass"] == True]
    failed = annotated.loc[annotated["trend_filter_pass"] == False]

    stats = pd.DataFrame(
        [
            _scenario_stats("baseline", annotated),
            _scenario_stats("trend_aligned_only", passed),
            _scenario_stats("trend_misaligned_only", failed),
        ]
    )
    note = (
        "EMA50 filter uses hourly Binance UM futures candles; Long requires close >= EMA50 and rising EMA50, "
        "Short requires close <= EMA50 and falling EMA50."
    )
    return MarketFilterResult(stats=stats, annotated_trades=annotated, note=note)


def _scenario_stats(name: str, frame: pd.DataFrame) -> dict:
    pnl = frame["closingPnl"].fillna(0)
    return {
        "scenario": name,
        "trades": len(frame),
        "total_pnl": pnl.sum(),
        "avg_pnl": pnl.mean() if len(frame) else np.nan,
        "win_rate": safe_divide((pnl > 0).sum(), pnl.notna().sum()),
        "avg_holding_hours": frame["holding_hours"].mean() if len(frame) else np.nan,
    }
