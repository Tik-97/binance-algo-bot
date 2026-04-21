"""
Download Binance Futures 1m historical klines and extend existing cache.
Usage: py download_history.py [SYMBOL]
Default: extends ETHUSDT from 2022-01-01 up to existing data start.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://fapi.binance.com"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "binance_klines"

COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def fetch_batch(symbol: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": 1500,
        "startTime": start_ms,
        "endTime": end_ms,
    }
    for attempt in range(5):
        try:
            r = httpx.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=20)
            if r.status_code == 429:
                print(" [rate limit, sleeping 10s]", end="", flush=True)
                time.sleep(10)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f" [err {attempt+1}: {e}]", end="", flush=True)
            time.sleep(3)
    return []


def ms_to_dt(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def main(symbol: str = "ETHUSDT") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{symbol}_1m.csv"

    # Load existing data to find its start
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True, errors="coerce")
        existing_start_ms = int(existing["open_time"].min().timestamp() * 1000)
        existing_end_ms   = int(existing["open_time"].max().timestamp() * 1000)
        print(f"Existing data: {existing['open_time'].min()} to {existing['open_time'].max()}")
        print(f"Rows: {len(existing):,}")
    else:
        existing = None
        existing_start_ms = None
        existing_end_ms   = None
        print("No existing data found — will download from scratch.")

    # Target: 2022-01-01 UTC
    target_start_ms = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)

    if existing_start_ms and existing_start_ms <= target_start_ms + 60_000:
        print("Data already starts at or before 2022-01-01. Nothing to download.")
        return

    end_ms = (existing_start_ms - 60_000) if existing_start_ms else int(
        pd.Timestamp("2024-04-06", tz="UTC").timestamp() * 1000
    )
    start_ms = target_start_ms

    total_ms = end_ms - start_ms
    total_bars_est = total_ms // 60_000
    print(f"\nDownloading {symbol} 1m from 2022-01-01 to {ms_to_dt(end_ms).date()}")
    print(f"Estimated ~{total_bars_est:,} candles (~{total_bars_est//1500} requests)\n")

    all_new: list[list] = []
    cursor = start_ms
    batch_num = 0

    while cursor < end_ms:
        batch_end = min(cursor + 1500 * 60_000, end_ms)
        batch = fetch_batch(symbol, cursor, batch_end)

        if not batch:
            print(f"\n  No data returned for {ms_to_dt(cursor)} — skipping window")
            cursor = batch_end + 60_000
            continue

        all_new.extend(batch)
        last_ts = int(batch[-1][0])
        cursor = last_ts + 60_000
        batch_num += 1

        if batch_num % 50 == 0:
            pct = (cursor - start_ms) / (end_ms - start_ms) * 100
            print(f"  {pct:.1f}%  {ms_to_dt(cursor).strftime('%Y-%m-%d')}  "
                  f"batches={batch_num}  rows={len(all_new):,}", flush=True)

        time.sleep(0.08)  # ~12 req/s, well under limit

    print(f"\nDownloaded {len(all_new):,} new candles in {batch_num} batches.")

    if not all_new:
        print("Nothing downloaded.")
        return

    # Build DataFrame
    new_df = pd.DataFrame(all_new, columns=COLUMNS)
    new_df["open_time"] = pd.to_datetime(new_df["open_time_ms"].astype(int), unit="ms", utc=True)
    new_df["close_time"] = pd.to_datetime(new_df["close_time_ms"].astype(int), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume",
              "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        new_df[c] = new_df[c].astype(float)
    new_df["number_of_trades"] = new_df["number_of_trades"].astype(int)

    # Merge with existing
    if existing is not None:
        # Re-read raw to preserve all columns
        existing_raw = pd.read_csv(out_path)
        existing_raw["open_time"] = pd.to_datetime(existing_raw["open_time"], utc=True, errors="coerce")
        combined = pd.concat([new_df, existing_raw], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Backup original
    if out_path.exists():
        backup = out_path.with_suffix(".csv.bak")
        out_path.rename(backup)
        print(f"Backed up original to {backup.name}")

    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined):,} total candles to {out_path}")
    print(f"Range: {combined['open_time'].min()} → {combined['open_time'].max()}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    main(symbol)
