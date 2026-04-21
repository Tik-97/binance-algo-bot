from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from entry_quality import signed_return_pct
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct


ETH_SYMBOL = "ETHUSDT"
RAW_MODE_ORDER = ["immediate_like", "short_delay", "delayed_limit_like"]
TWO_BUCKET_ORDER = ["immediate_like", "delayed_family"]
MODE_LABELS = {
    "immediate_like": "Immediate-like",
    "short_delay": "Short-delay",
    "delayed_limit_like": "Delayed / limit-like",
    "delayed_family": "Delayed-family",
}
RETURN_HORIZONS = [1, 5, 15, 30]
SHORT_WINDOW_HORIZONS = [15, 30]


def run_eth_execution_reanchored_entry_quality_analysis(
    eth_execution_aware_df: pd.DataFrame,
    execution_mode_eth_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "EXECUTION-REANCHORED ENTRY QUALITY - ETH EARLY REACTION FROM PROXY FILL TIME\n"
        "--------------------------------------------------------------------------\n"
        "No sufficient ETH data was available for execution re-anchoring analysis.\n"
    )

    base = prepare_reanchored_base(eth_execution_aware_df, execution_mode_eth_df)
    if base.empty:
        return {
            "per_trade": empty,
            "summary": empty,
            "cluster_quality": empty,
            "distortion": empty,
            "section_text": empty_section,
        }

    market_frames = load_eth_market_frames(base, logger)
    logger.info("Running ETH execution re-anchoring on %s ETH trades", len(base))
    per_trade = recompute_reanchored_metrics(base, market_frames)
    summary = build_reanchored_summary(per_trade)
    cluster_quality = build_reanchored_cluster_quality(per_trade)
    distortion = build_reanchored_distortion(per_trade)
    section_text = build_reanchored_section(per_trade, summary, cluster_quality, distortion)

    return {
        "per_trade": per_trade,
        "summary": summary,
        "cluster_quality": cluster_quality,
        "distortion": distortion,
        "section_text": section_text,
    }


def prepare_reanchored_base(
    eth_execution_aware_df: pd.DataFrame,
    execution_mode_eth_df: pd.DataFrame,
) -> pd.DataFrame:
    if eth_execution_aware_df.empty:
        return pd.DataFrame()

    base = eth_execution_aware_df.loc[eth_execution_aware_df["symbol"] == ETH_SYMBOL].copy()
    if base.empty:
        return base

    join_columns = [
        "trade_number",
        "opened_dt_utc",
        "closed_dt_utc",
        "avgCost",
        "side",
        "execution_mode",
        "execution_mode_two_bucket",
        "fill_delay_minutes",
        "fill_delay_bucket",
        "interval_used",
        "touch_found",
        "touch_in_open_candle_flag",
        "first_touch_time_utc",
        "uncertain_flag",
    ]
    execution_lookup = select_existing_columns(execution_mode_eth_df, join_columns)
    if not execution_lookup.empty:
        base = base.merge(
            execution_lookup,
            on="trade_number",
            how="left",
            suffixes=("", "_execution"),
        )

    for column in [
        "opened_dt_utc",
        "closed_dt_utc",
        "avgCost",
        "side",
        "execution_mode",
        "execution_mode_two_bucket",
        "fill_delay_minutes",
        "fill_delay_bucket",
        "interval_used",
        "touch_found",
        "touch_in_open_candle_flag",
        "first_touch_time_utc",
        "uncertain_flag",
    ]:
        base = coalesce_duplicate_column(base, column, "_execution")

    for column in ["opened_dt_utc", "closed_dt_utc", "first_touch_time_utc"]:
        if column in base.columns:
            base[column] = pd.to_datetime(base[column], utc=True, errors="coerce")

    numeric_columns = [
        "cluster_id",
        "avgCost",
        "fill_delay_minutes",
        "time_to_first_positive_minutes",
        "return_after_1m_pct",
        "return_after_5m_pct",
        "return_after_15m_pct",
        "return_after_30m_pct",
        "MAE_pct",
        "MFE_pct",
        "captured_share_of_MFE",
        "realized_return_pct",
    ]
    for column in numeric_columns:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    base["execution_mode"] = base.get("execution_mode", pd.Series(index=base.index, dtype="object")).fillna("uncertain")
    base["execution_mode_two_bucket"] = base.get(
        "execution_mode_two_bucket",
        pd.Series(index=base.index, dtype="object"),
    ).fillna(
        base["execution_mode"].map(
            {
                "immediate_like": "immediate_like",
                "short_delay": "delayed_family",
                "delayed_limit_like": "delayed_family",
            }
        )
    ).fillna("uncertain")
    base["proxy_fill_time"] = np.where(
        base["touch_found"].fillna(False),
        base["first_touch_time_utc"],
        pd.NaT,
    )
    base["proxy_fill_time"] = pd.to_datetime(base["proxy_fill_time"], utc=True, errors="coerce")
    same_candle_mask = base["execution_mode"].eq("immediate_like") & base["proxy_fill_time"].isna()
    base.loc[same_candle_mask, "proxy_fill_time"] = base.loc[same_candle_mask, "opened_dt_utc"]
    base["proxy_fill_price"] = base["avgCost"]
    base["proxy_fill_available_flag"] = base["proxy_fill_time"].notna() & base["proxy_fill_price"].notna()
    base["strict_execution_compare_flag"] = (
        base["execution_mode_two_bucket"].isin(TWO_BUCKET_ORDER)
        & base["proxy_fill_available_flag"]
        & ~base["uncertain_flag"].fillna(False)
    )

    sort_columns = [column for column in ["cluster_id", "trade_number"] if column in base.columns]
    return base.sort_values(sort_columns).reset_index(drop=True)


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.DataFrame()
    subset_keys = [column for column in ["trade_number"] if column in existing]
    return df[existing].drop_duplicates(subset=subset_keys)


def coalesce_duplicate_column(df: pd.DataFrame, base_name: str, suffix: str) -> pd.DataFrame:
    duplicate_name = f"{base_name}{suffix}"
    if duplicate_name not in df.columns:
        return df
    if base_name not in df.columns:
        return df.rename(columns={duplicate_name: base_name})
    df[base_name] = df[base_name].combine_first(df[duplicate_name])
    return df.drop(columns=[duplicate_name])


def load_eth_market_frames(base: pd.DataFrame, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    opened_min = pd.to_datetime(base["opened_dt_utc"], utc=True, errors="coerce").min()
    closed_max = pd.to_datetime(base["closed_dt_utc"], utc=True, errors="coerce").max()
    proxy_fill_max = pd.to_datetime(base["proxy_fill_time"], utc=True, errors="coerce").max()

    if pd.isna(opened_min):
        return {"1m": pd.DataFrame(), "5m": pd.DataFrame()}

    effective_end = max(
        timestamp
        for timestamp in [
            closed_max if pd.notna(closed_max) else opened_min,
            proxy_fill_max + pd.Timedelta(minutes=max(RETURN_HORIZONS) + 5) if pd.notna(proxy_fill_max) else opened_min,
        ]
    )
    start_ms = int((opened_min - pd.Timedelta(minutes=10)).timestamp() * 1000)
    end_ms = int((effective_end + pd.Timedelta(minutes=10)).timestamp() * 1000)

    logger.info("Loading ETH candles for execution re-anchoring")
    return {
        "1m": fetch_futures_klines(ETH_SYMBOL, start_ms, end_ms, interval="1m", logger=logger),
        "5m": fetch_futures_klines(ETH_SYMBOL, start_ms, end_ms, interval="5m", logger=logger),
    }


def recompute_reanchored_metrics(base: pd.DataFrame, market_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for _, trade in base.iterrows():
        record = trade.to_dict()
        proxy_fill_time = pd.to_datetime(trade.get("proxy_fill_time"), utc=True, errors="coerce")
        proxy_fill_price = pd.to_numeric(pd.Series([trade.get("proxy_fill_price")]), errors="coerce").iloc[0]
        side = str(trade.get("side"))
        close_time = pd.to_datetime(trade.get("closed_dt_utc"), utc=True, errors="coerce")

        record["reanchored_anchor_delay_minutes"] = (
            (proxy_fill_time - pd.to_datetime(trade.get("opened_dt_utc"), utc=True, errors="coerce")).total_seconds() / 60.0
            if pd.notna(proxy_fill_time)
            else np.nan
        )

        if not bool(record.get("proxy_fill_available_flag")):
            records.append(add_empty_reanchored_metrics(record))
            continue

        for horizon in RETURN_HORIZONS:
            value, used_interval = reaction_return_from_anchor(
                market_frames,
                proxy_fill_time,
                float(proxy_fill_price),
                side,
                horizon,
            )
            record[f"reanchored_return_after_{horizon}m_pct"] = value
            record[f"reanchored_return_after_{horizon}m_interval"] = used_interval

        time_positive, time_interval = time_to_first_positive_from_anchor(
            market_frames,
            proxy_fill_time,
            float(proxy_fill_price),
            side,
            close_time,
        )
        record["reanchored_time_to_first_positive_minutes"] = time_positive
        record["reanchored_time_to_first_positive_interval"] = time_interval

        for window in SHORT_WINDOW_HORIZONS:
            excursion = window_excursion_from_anchor(
                market_frames,
                proxy_fill_time,
                float(proxy_fill_price),
                side,
                window,
            )
            record[f"reanchored_MAE_first_{window}m_pct"] = excursion["mae_pct"]
            record[f"reanchored_MFE_first_{window}m_pct"] = excursion["mfe_pct"]
            record[f"reanchored_window_{window}m_interval"] = excursion["interval_used"]

        records.append(add_reanchored_deltas(record))

    return pd.DataFrame(records)


def add_empty_reanchored_metrics(record: dict) -> dict:
    for horizon in RETURN_HORIZONS:
        record[f"reanchored_return_after_{horizon}m_pct"] = np.nan
        record[f"reanchored_return_after_{horizon}m_interval"] = pd.NA
        record[f"delta_return_after_{horizon}m_pct"] = np.nan
    record["reanchored_time_to_first_positive_minutes"] = np.nan
    record["reanchored_time_to_first_positive_interval"] = pd.NA
    record["delta_time_to_first_positive_minutes"] = np.nan
    for window in SHORT_WINDOW_HORIZONS:
        record[f"reanchored_MAE_first_{window}m_pct"] = np.nan
        record[f"reanchored_MFE_first_{window}m_pct"] = np.nan
        record[f"reanchored_window_{window}m_interval"] = pd.NA
    return record


def add_reanchored_deltas(record: dict) -> dict:
    for horizon in RETURN_HORIZONS:
        current_value = record.get(f"return_after_{horizon}m_pct")
        reanchored_value = record.get(f"reanchored_return_after_{horizon}m_pct")
        record[f"delta_return_after_{horizon}m_pct"] = diff_or_nan(reanchored_value, current_value)
    record["delta_time_to_first_positive_minutes"] = diff_or_nan(
        record.get("reanchored_time_to_first_positive_minutes"),
        record.get("time_to_first_positive_minutes"),
    )
    return record


def diff_or_nan(left: object, right: object) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left) - float(right)


def reaction_return_from_anchor(
    market_frames: dict[str, pd.DataFrame],
    anchor_time: pd.Timestamp,
    anchor_price: float,
    side: str,
    horizon_minutes: int,
) -> tuple[float, str]:
    target_time = anchor_time + pd.Timedelta(minutes=horizon_minutes)
    for interval in ["1m", "5m"]:
        frame = market_frames.get(interval, pd.DataFrame())
        if frame.empty:
            continue
        candle = frame.loc[frame["close_time"] >= target_time].head(1)
        if candle.empty:
            continue
        return signed_return_pct(anchor_price, float(candle.iloc[0]["close"]), side), interval
    return np.nan, "none"


def time_to_first_positive_from_anchor(
    market_frames: dict[str, pd.DataFrame],
    anchor_time: pd.Timestamp,
    anchor_price: float,
    side: str,
    close_time: pd.Timestamp,
) -> tuple[float, str]:
    if pd.isna(close_time):
        return np.nan, "none"

    for interval in ["1m", "5m"]:
        frame = market_frames.get(interval, pd.DataFrame())
        if frame.empty:
            continue
        subset = frame.loc[(frame["close_time"] >= anchor_time) & (frame["close_time"] <= close_time)].copy()
        if subset.empty:
            continue
        returns = subset["close"].astype(float).apply(lambda price: signed_return_pct(anchor_price, price, side))
        positive_hits = subset.loc[returns > 0, "close_time"]
        if positive_hits.empty:
            return np.nan, interval
        return (positive_hits.iloc[0] - anchor_time).total_seconds() / 60.0, interval
    return np.nan, "none"


def window_excursion_from_anchor(
    market_frames: dict[str, pd.DataFrame],
    anchor_time: pd.Timestamp,
    anchor_price: float,
    side: str,
    window_minutes: int,
) -> dict:
    window_end = anchor_time + pd.Timedelta(minutes=window_minutes)
    for interval in ["1m", "5m"]:
        frame = market_frames.get(interval, pd.DataFrame())
        if frame.empty:
            continue
        subset = frame.loc[(frame["close_time"] >= anchor_time) & (frame["close_time"] <= window_end)].copy()
        if subset.empty:
            continue

        close_candidates = subset["close"].astype(float).apply(lambda price: signed_return_pct(anchor_price, price, side))
        close_mfe = max(close_candidates.max(), 0.0) if not close_candidates.empty else np.nan
        close_mae = max(-close_candidates.min(), 0.0) if not close_candidates.empty else np.nan

        full_candles = subset.loc[subset["open_time"] >= anchor_time].copy()
        if full_candles.empty:
            return {"mfe_pct": close_mfe, "mae_pct": close_mae, "interval_used": interval}

        if side.lower() == "long":
            intrabar_mfe = ((full_candles["high"].astype(float) - anchor_price) / anchor_price * 100.0).clip(lower=0.0).max()
            intrabar_mae = ((anchor_price - full_candles["low"].astype(float)) / anchor_price * 100.0).clip(lower=0.0).max()
        else:
            intrabar_mfe = ((anchor_price - full_candles["low"].astype(float)) / anchor_price * 100.0).clip(lower=0.0).max()
            intrabar_mae = ((full_candles["high"].astype(float) - anchor_price) / anchor_price * 100.0).clip(lower=0.0).max()

        return {
            "mfe_pct": nanmax_pair(close_mfe, intrabar_mfe),
            "mae_pct": nanmax_pair(close_mae, intrabar_mae),
            "interval_used": interval,
        }

    return {"mfe_pct": np.nan, "mae_pct": np.nan, "interval_used": "none"}


def nanmax_pair(left: float, right: float) -> float:
    values = [value for value in [left, right] if pd.notna(value)]
    if not values:
        return np.nan
    return float(max(values))


def build_reanchored_summary(per_trade: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict = per_trade.loc[per_trade["strict_execution_compare_flag"]].copy()
    rows.extend(
        summarize_scope(strict.loc[strict["execution_mode"] == mode], "ETH overall", "execution_mode", mode)
        for mode in RAW_MODE_ORDER
        if not strict.loc[strict["execution_mode"] == mode].empty
    )
    rows.extend(
        summarize_scope(strict.loc[strict["execution_mode_two_bucket"] == bucket], "ETH overall", "execution_bucket", bucket)
        for bucket in TWO_BUCKET_ORDER
        if not strict.loc[strict["execution_mode_two_bucket"] == bucket].empty
    )
    return pd.DataFrame(rows)


def build_reanchored_cluster_quality(per_trade: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict = per_trade.loc[per_trade["strict_execution_compare_flag"]].copy()
    for cluster_id in sorted(strict["cluster_id"].dropna().astype(int).unique().tolist()):
        cluster_df = strict.loc[strict["cluster_id"] == cluster_id].copy()
        scope_name = f"Cluster {cluster_id}"
        rows.extend(
            summarize_scope(cluster_df.loc[cluster_df["execution_mode"] == mode], scope_name, "execution_mode", mode)
            for mode in RAW_MODE_ORDER
            if not cluster_df.loc[cluster_df["execution_mode"] == mode].empty
        )
        rows.extend(
            summarize_scope(cluster_df.loc[cluster_df["execution_mode_two_bucket"] == bucket], scope_name, "execution_bucket", bucket)
            for bucket in TWO_BUCKET_ORDER
            if not cluster_df.loc[cluster_df["execution_mode_two_bucket"] == bucket].empty
        )
    return pd.DataFrame(rows)


def summarize_scope(frame: pd.DataFrame, scope: str, group_kind: str, group_name: str) -> dict:
    row = {
        "scope": scope,
        "group_kind": group_kind,
        "group_name": group_name,
        "group_label": MODE_LABELS.get(group_name, group_name),
        "trades": int(len(frame)),
        "median_fill_delay_minutes": pd.to_numeric(frame["fill_delay_minutes"], errors="coerce").median(),
        "proxy_fill_coverage_share": frame["proxy_fill_available_flag"].mean() if not frame.empty else np.nan,
        "timing_interval_1m_share": build_interval_share(frame, "reanchored_time_to_first_positive_interval", "1m"),
        "timing_interval_5m_share": build_interval_share(frame, "reanchored_time_to_first_positive_interval", "5m"),
    }

    for horizon in RETURN_HORIZONS:
        row[f"current_median_return_after_{horizon}m_pct"] = pd.to_numeric(frame[f"return_after_{horizon}m_pct"], errors="coerce").median()
        row[f"reanchored_median_return_after_{horizon}m_pct"] = pd.to_numeric(
            frame[f"reanchored_return_after_{horizon}m_pct"], errors="coerce"
        ).median()
        row[f"delta_median_return_after_{horizon}m_pct"] = pd.to_numeric(
            frame[f"delta_return_after_{horizon}m_pct"], errors="coerce"
        ).median()

    row["current_median_time_to_first_positive_minutes"] = pd.to_numeric(
        frame["time_to_first_positive_minutes"], errors="coerce"
    ).median()
    row["reanchored_median_time_to_first_positive_minutes"] = pd.to_numeric(
        frame["reanchored_time_to_first_positive_minutes"], errors="coerce"
    ).median()
    row["delta_median_time_to_first_positive_minutes"] = pd.to_numeric(
        frame["delta_time_to_first_positive_minutes"], errors="coerce"
    ).median()

    for window in SHORT_WINDOW_HORIZONS:
        row[f"reanchored_median_MAE_first_{window}m_pct"] = pd.to_numeric(
            frame[f"reanchored_MAE_first_{window}m_pct"], errors="coerce"
        ).median()
        row[f"reanchored_median_MFE_first_{window}m_pct"] = pd.to_numeric(
            frame[f"reanchored_MFE_first_{window}m_pct"], errors="coerce"
        ).median()

    return row


def build_interval_share(frame: pd.DataFrame, column: str, value: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return frame[column].eq(value).mean()


def build_reanchored_distortion(per_trade: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strict = per_trade.loc[per_trade["strict_execution_compare_flag"]].copy()
    scopes = [("ETH overall", strict)]
    for cluster_id in sorted(strict["cluster_id"].dropna().astype(int).unique().tolist()):
        scopes.append((f"Cluster {cluster_id}", strict.loc[strict["cluster_id"] == cluster_id].copy()))

    for scope_name, scope_df in scopes:
        for group_name in TWO_BUCKET_ORDER:
            group_df = scope_df.loc[scope_df["execution_mode_two_bucket"] == group_name].copy()
            if group_df.empty:
                continue
            rows.append(
                {
                    "scope": scope_name,
                    "group_name": group_name,
                    "group_label": MODE_LABELS[group_name],
                    "trades": int(len(group_df)),
                    "median_fill_delay_minutes": pd.to_numeric(group_df["fill_delay_minutes"], errors="coerce").median(),
                    "delta_return_after_1m_pct": pd.to_numeric(group_df["delta_return_after_1m_pct"], errors="coerce").median(),
                    "delta_return_after_5m_pct": pd.to_numeric(group_df["delta_return_after_5m_pct"], errors="coerce").median(),
                    "delta_return_after_15m_pct": pd.to_numeric(group_df["delta_return_after_15m_pct"], errors="coerce").median(),
                    "delta_return_after_30m_pct": pd.to_numeric(group_df["delta_return_after_30m_pct"], errors="coerce").median(),
                    "delta_time_to_first_positive_minutes": pd.to_numeric(
                        group_df["delta_time_to_first_positive_minutes"], errors="coerce"
                    ).median(),
                }
            )
    return pd.DataFrame(rows)


def build_reanchored_section(
    per_trade: pd.DataFrame,
    summary: pd.DataFrame,
    cluster_quality: pd.DataFrame,
    distortion: pd.DataFrame,
) -> str:
    if per_trade.empty or summary.empty:
        return (
            "EXECUTION-REANCHORED ENTRY QUALITY - ETH EARLY REACTION FROM PROXY FILL TIME\n"
            "--------------------------------------------------------------------------\n"
            "Execution re-anchoring could not be completed.\n"
        )

    overall_immediate = select_summary_row(summary, "ETH overall", "execution_bucket", "immediate_like")
    overall_delayed = select_summary_row(summary, "ETH overall", "execution_bucket", "delayed_family")
    cluster1_immediate = select_summary_row(cluster_quality, "Cluster 1", "execution_bucket", "immediate_like")
    cluster1_delayed = select_summary_row(cluster_quality, "Cluster 1", "execution_bucket", "delayed_family")
    cluster2_immediate = select_summary_row(cluster_quality, "Cluster 2", "execution_bucket", "immediate_like")
    cluster2_delayed = select_summary_row(cluster_quality, "Cluster 2", "execution_bucket", "delayed_family")

    overall_distortion = select_distortion_row(distortion, "ETH overall", "delayed_family")
    overall_material = distortion_level(overall_distortion)
    cluster1_status = cluster_split_status(cluster1_immediate, cluster1_delayed)
    cluster2_status = cluster_split_status(cluster2_immediate, cluster2_delayed)
    retained_status = retained_verdict(cluster1_status, cluster2_status, overall_material)
    branching_ready = execution_branching_readiness(cluster1_immediate, cluster1_delayed, overall_distortion)
    confidence = determine_reanchored_confidence(cluster1_immediate, cluster1_delayed, cluster2_immediate, cluster2_delayed)

    section = f"""
EXECUTION-REANCHORED ENTRY QUALITY - ETH EARLY REACTION FROM PROXY FILL TIME
--------------------------------------------------------------------------
This section recomputes short-horizon ETH early-reaction metrics from a proxy fill moment: the earliest post-open market touch of `avgCost`. It does not claim an exact fill timestamp. It is a timing-corrected comparison that reduces naive opened-time distortion for delayed-fill compatible trades.

Coverage:
- ETH trades considered: {fmt_int(len(per_trade))}.
- Strict execution-comparison coverage: {pct(per_trade['strict_execution_compare_flag'].mean())}.
- Delayed-family median proxy fill delay: {fmt_number(overall_delayed.get('median_fill_delay_minutes'))} minutes.
- Re-anchored timing used 1m candles for {pct(overall_delayed.get('timing_interval_1m_share'))} of delayed-family ETH trades and 5m fallback for {pct(overall_delayed.get('timing_interval_5m_share'))}.

Overall ETH distortion check:
- Delayed-family delta after re-anchoring: return 1m {fmt_number(overall_distortion.get('delta_return_after_1m_pct'))}%, return 5m {fmt_number(overall_distortion.get('delta_return_after_5m_pct'))}%, return 15m {fmt_number(overall_distortion.get('delta_return_after_15m_pct'))}%, time to first profit {fmt_number(overall_distortion.get('delta_time_to_first_positive_minutes'))} minutes.
- Opened-time anchoring looks {overall_material} for delayed-family ETH early reaction metrics.
- After re-anchoring, delayed-family vs immediate-like return after 5m is {fmt_number(overall_delayed.get('reanchored_median_return_after_5m_pct'))}% vs {fmt_number(overall_immediate.get('reanchored_median_return_after_5m_pct'))}%.
- After re-anchoring, delayed-family vs immediate-like return after 15m is {fmt_number(overall_delayed.get('reanchored_median_return_after_15m_pct'))}% vs {fmt_number(overall_immediate.get('reanchored_median_return_after_15m_pct'))}%.
- Re-anchored time to first positive is {fmt_number(overall_delayed.get('reanchored_median_time_to_first_positive_minutes'))} minutes for delayed-family versus {fmt_number(overall_immediate.get('reanchored_median_time_to_first_positive_minutes'))} for immediate-like.

Cluster 1:
- Immediate-like current vs re-anchored return after 5m: {fmt_number(cluster1_immediate.get('current_median_return_after_5m_pct'))}% -> {fmt_number(cluster1_immediate.get('reanchored_median_return_after_5m_pct'))}%.
- Delayed-family current vs re-anchored return after 5m: {fmt_number(cluster1_delayed.get('current_median_return_after_5m_pct'))}% -> {fmt_number(cluster1_delayed.get('reanchored_median_return_after_5m_pct'))}%.
- Delayed-family current vs re-anchored time to first positive: {fmt_number(cluster1_delayed.get('current_median_time_to_first_positive_minutes'))} -> {fmt_number(cluster1_delayed.get('reanchored_median_time_to_first_positive_minutes'))} minutes.
- Cluster 1 execution-sensitive split is {cluster1_status}.

Cluster 2:
- Immediate-like current vs re-anchored return after 5m: {fmt_number(cluster2_immediate.get('current_median_return_after_5m_pct'))}% -> {fmt_number(cluster2_immediate.get('reanchored_median_return_after_5m_pct'))}%.
- Delayed-family current vs re-anchored return after 5m: {fmt_number(cluster2_delayed.get('current_median_return_after_5m_pct'))}% -> {fmt_number(cluster2_delayed.get('reanchored_median_return_after_5m_pct'))}%.
- Delayed-family current vs re-anchored time to first positive: {fmt_number(cluster2_delayed.get('current_median_time_to_first_positive_minutes'))} -> {fmt_number(cluster2_delayed.get('reanchored_median_time_to_first_positive_minutes'))} minutes.
- Cluster 2 execution-sensitive split is {cluster2_status}.

Interpretation:
- Previous execution-aware conclusions should be {retained_status}.
- The re-anchoring correction {'still supports' if branching_ready else 'does not yet fully justify'} moving toward execution-aware proxy branching for Cluster 1.
- Confidence level: {confidence}.

PRACTICAL INTERPRETATION - IS EXECUTION-AWARE BRANCHING READY YET?
------------------------------------------------------------------
- Separate Cluster 1 proxy branches for immediate-like vs delayed-family {'now look justified' if branching_ready else 'still need one more validation step before they look justified'}.
- The earlier conclusions that survived the correction are: immediate-like ETH trades remain stable under re-anchoring, while delayed-family ETH trades {'still show stronger short-horizon post-fill progress' if delayed_progress_persists(overall_immediate, overall_delayed) else 'do not clearly preserve their earlier short-horizon edge once the timing anchor is corrected'}.
- The most logical next step is {'to build execution-aware Cluster 1 proxy variants and compare them under the same exit and friction assumptions' if branching_ready else 'to run one extra validation step on re-anchored Cluster 1 behavior before branching the proxy model'}.
""".strip()
    return section + "\n"


def select_summary_row(df: pd.DataFrame, scope: str, group_kind: str, group_name: str) -> pd.Series:
    match = df.loc[
        (df["scope"] == scope)
        & (df["group_kind"] == group_kind)
        & (df["group_name"] == group_name)
    ]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def select_distortion_row(df: pd.DataFrame, scope: str, group_name: str) -> pd.Series:
    match = df.loc[(df["scope"] == scope) & (df["group_name"] == group_name)]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def safe_number(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def distortion_level(row: pd.Series) -> str:
    if row.empty:
        return "unclear because coverage is too thin"
    score = (
        abs(safe_number(row.get("delta_return_after_1m_pct")))
        + abs(safe_number(row.get("delta_return_after_5m_pct")))
        + abs(safe_number(row.get("delta_return_after_15m_pct")))
        + abs(safe_number(row.get("delta_time_to_first_positive_minutes"))) / 10.0
    )
    if score >= 1.0:
        return "material"
    if score >= 0.35:
        return "moderate"
    return "limited"


def cluster_split_status(immediate_row: pd.Series, delayed_row: pd.Series) -> str:
    if immediate_row.empty or delayed_row.empty:
        return "unclear because one subgroup is too small"

    current_gap = safe_number(delayed_row.get("current_median_return_after_5m_pct")) - safe_number(
        immediate_row.get("current_median_return_after_5m_pct")
    )
    reanchored_gap = safe_number(delayed_row.get("reanchored_median_return_after_5m_pct")) - safe_number(
        immediate_row.get("reanchored_median_return_after_5m_pct")
    )
    reanchored_time_gap = safe_number(
        immediate_row.get("reanchored_median_time_to_first_positive_minutes")
    ) - safe_number(delayed_row.get("reanchored_median_time_to_first_positive_minutes"))

    if current_gap > 0.20 and reanchored_gap > 0.20 and reanchored_time_gap >= 0:
        return "still meaningfully present after re-anchoring"
    if current_gap > 0.20 and reanchored_gap > 0.05:
        return "partially softened but still visible after re-anchoring"
    if reanchored_gap <= 0.05 and abs(reanchored_time_gap) < 0.25:
        return "mostly weakened after re-anchoring"
    return "mixed after re-anchoring"


def retained_verdict(cluster1_status: str, cluster2_status: str, overall_material: str) -> str:
    if "still meaningfully present" in cluster1_status and "still meaningfully present" in cluster2_status:
        return "mostly retained"
    if overall_material == "material" or "partially softened" in cluster1_status or "partially softened" in cluster2_status:
        return "partially softened"
    return "substantially revised"


def delayed_progress_persists(immediate_row: pd.Series, delayed_row: pd.Series) -> bool:
    if immediate_row.empty or delayed_row.empty:
        return False
    return safe_number(delayed_row.get("reanchored_median_return_after_5m_pct")) > safe_number(
        immediate_row.get("reanchored_median_return_after_5m_pct")
    ) + 0.15


def execution_branching_readiness(
    cluster1_immediate: pd.Series,
    cluster1_delayed: pd.Series,
    overall_distortion: pd.Series,
) -> bool:
    if cluster1_immediate.empty or cluster1_delayed.empty:
        return False
    reanchored_gap = safe_number(cluster1_delayed.get("reanchored_median_return_after_5m_pct")) - safe_number(
        cluster1_immediate.get("reanchored_median_return_after_5m_pct")
    )
    time_gap = safe_number(cluster1_immediate.get("reanchored_median_time_to_first_positive_minutes")) - safe_number(
        cluster1_delayed.get("reanchored_median_time_to_first_positive_minutes")
    )
    distortion_material = distortion_level(overall_distortion) in {"material", "moderate"}
    return reanchored_gap > 0.20 and time_gap >= -0.25 and distortion_material


def determine_reanchored_confidence(
    cluster1_immediate: pd.Series,
    cluster1_delayed: pd.Series,
    cluster2_immediate: pd.Series,
    cluster2_delayed: pd.Series,
) -> str:
    counts = [
        safe_number(cluster1_immediate.get("trades")),
        safe_number(cluster1_delayed.get("trades")),
        safe_number(cluster2_immediate.get("trades")),
        safe_number(cluster2_delayed.get("trades")),
    ]
    if min(counts) >= 10:
        return "medium"
    return "low"


def create_eth_execution_reanchored_plots(
    summary_df: pd.DataFrame,
    cluster_quality_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if summary_df.empty or cluster_quality_df.empty:
        return

    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    reaction_path = plots_dir / "eth_execution_reanchored_reaction_comparison.png"
    cluster_reaction_path = plots_dir / "eth_execution_reanchored_cluster_reaction.png"
    time_to_profit_path = plots_dir / "eth_execution_reanchored_time_to_profit.png"

    fig_reaction = plot_reanchored_overall_reaction(summary_df)
    fig_reaction.savefig(reaction_path, dpi=150, bbox_inches="tight")
    plt.close(fig_reaction)

    fig_cluster_reaction = plot_reanchored_cluster_reaction(cluster_quality_df)
    fig_cluster_reaction.savefig(cluster_reaction_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cluster_reaction)

    fig_time_to_profit = plot_reanchored_time_to_profit(cluster_quality_df)
    fig_time_to_profit.savefig(time_to_profit_path, dpi=150, bbox_inches="tight")
    plt.close(fig_time_to_profit)

    logger.info("Saved ETH execution re-anchored plots")


def plot_reanchored_overall_reaction(summary_df: pd.DataFrame) -> plt.Figure:
    plot_df = summary_df.loc[
        (summary_df["scope"] == "ETH overall")
        & (summary_df["group_kind"] == "execution_bucket")
        & (summary_df["group_name"].isin(TWO_BUCKET_ORDER))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    draw_current_vs_reanchored_metric(
        axes[0],
        plot_df,
        "current_median_return_after_5m_pct",
        "reanchored_median_return_after_5m_pct",
        "Median return after 5m %",
    )
    draw_current_vs_reanchored_metric(
        axes[1],
        plot_df,
        "current_median_return_after_15m_pct",
        "reanchored_median_return_after_15m_pct",
        "Median return after 15m %",
    )
    fig.tight_layout()
    return fig


def plot_reanchored_cluster_reaction(cluster_quality_df: pd.DataFrame) -> plt.Figure:
    plot_df = cluster_quality_df.loc[
        (cluster_quality_df["group_kind"] == "execution_bucket")
        & (cluster_quality_df["group_name"].isin(TWO_BUCKET_ORDER))
    ].copy()
    plot_df["plot_label"] = plot_df.apply(lambda row: f"{row['scope']}\n{MODE_LABELS.get(row['group_name'], row['group_name'])}", axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    draw_cluster_metric_bars(
        axes[0],
        plot_df,
        "reanchored_median_return_after_5m_pct",
        "Re-anchored return after 5m %",
    )
    draw_cluster_metric_bars(
        axes[1],
        plot_df,
        "reanchored_median_return_after_15m_pct",
        "Re-anchored return after 15m %",
    )
    fig.tight_layout()
    return fig


def plot_reanchored_time_to_profit(cluster_quality_df: pd.DataFrame) -> plt.Figure:
    plot_df = cluster_quality_df.loc[
        (cluster_quality_df["group_kind"] == "execution_bucket")
        & (cluster_quality_df["group_name"].isin(TWO_BUCKET_ORDER))
    ].copy()
    plot_df["plot_label"] = plot_df.apply(lambda row: f"{row['scope']}\n{MODE_LABELS.get(row['group_name'], row['group_name'])}", axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    draw_current_vs_reanchored_cluster_bars(
        ax,
        plot_df,
        "current_median_time_to_first_positive_minutes",
        "reanchored_median_time_to_first_positive_minutes",
        "Median time to first positive (minutes)",
    )
    fig.tight_layout()
    return fig


def draw_current_vs_reanchored_metric(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    current_column: str,
    reanchored_column: str,
    title: str,
) -> None:
    x = np.arange(len(plot_df))
    width = 0.35
    current_values = pd.to_numeric(plot_df[current_column], errors="coerce").to_numpy(dtype=float)
    reanchored_values = pd.to_numeric(plot_df[reanchored_column], errors="coerce").to_numpy(dtype=float)
    labels = [MODE_LABELS.get(name, name) for name in plot_df["group_name"]]

    ax.bar(x - width / 2, current_values, width=width, label="Current", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, reanchored_values, width=width, label="Re-anchored", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)


def draw_cluster_metric_bars(ax: plt.Axes, plot_df: pd.DataFrame, value_column: str, title: str) -> None:
    x = np.arange(len(plot_df))
    values = pd.to_numeric(plot_df[value_column], errors="coerce").to_numpy(dtype=float)
    colors = ["#1f77b4" if group == "immediate_like" else "#d62728" for group in plot_df["group_name"]]
    ax.bar(x, values, color=colors, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].tolist())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def draw_current_vs_reanchored_cluster_bars(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    current_column: str,
    reanchored_column: str,
    title: str,
) -> None:
    x = np.arange(len(plot_df))
    width = 0.35
    current_values = pd.to_numeric(plot_df[current_column], errors="coerce").to_numpy(dtype=float)
    reanchored_values = pd.to_numeric(plot_df[reanchored_column], errors="coerce").to_numpy(dtype=float)

    ax.bar(x - width / 2, current_values, width=width, label="Current", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, reanchored_values, width=width, label="Re-anchored", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].tolist())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
