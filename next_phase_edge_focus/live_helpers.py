from __future__ import annotations

import numpy as np
import pandas as pd


MAX_MOVE_1M_PCT = 0.11
MAX_MOVE_3M_PCT = 0.18
MAX_MOVE_5M_PCT = 0.25
MAX_RELATIVE_VOLUME_1M = 1.20
MIN_BOUNCE_60M_PCT = 0.24
MAX_BOUNCE_60M_PCT = 1.20
MIN_DIST_1H_HIGH_PCT = 0.10
MAX_DIST_1H_HIGH_PCT = 0.22
MAX_BULL_BODY_SUM_5M_PCT = 0.32
ALLOWED_GREEN_COUNTS = {2.0, 3.0}


def build_market_regime_frame_from_candles(candles_1m: pd.DataFrame) -> pd.DataFrame:
    daily = (
        candles_1m.set_index("open_time")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
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
    daily["vol_regime"] = np.where(daily["vol_14d_pct"] >= vol_threshold, "high_vol", "low_vol")
    return daily


def is_calm_mild_transition(bar: pd.Series) -> bool:
    return (
        float(bar["move_1m_pct"]) <= MAX_MOVE_1M_PCT
        and float(bar["move_3m_pct"]) <= MAX_MOVE_3M_PCT
        and float(bar["move_5m_pct"]) <= MAX_MOVE_5M_PCT
        and float(bar["relative_volume_1m"]) <= MAX_RELATIVE_VOLUME_1M
        and float(bar["bounce_60m_pct"]) >= MIN_BOUNCE_60M_PCT
        and float(bar["bounce_60m_pct"]) <= MAX_BOUNCE_60M_PCT
        and float(bar["dist_1h_high_pct"]) >= MIN_DIST_1H_HIGH_PCT
        and float(bar["dist_1h_high_pct"]) <= MAX_DIST_1H_HIGH_PCT
        and float(bar["bull_body_sum_5m_pct"]) <= MAX_BULL_BODY_SUM_5M_PCT
        and float(bar["green_count_last_5"]) in ALLOWED_GREEN_COUNTS
    )
