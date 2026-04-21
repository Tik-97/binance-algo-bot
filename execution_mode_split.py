from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


PRIMARY_SYMBOL = "ETHUSDT"
SECONDARY_SYMBOL = "BTCUSDT"


def run_execution_mode_split_analysis(
    trades_df: pd.DataFrame,
    entry_quality_df: pd.DataFrame,
    eth_clusters_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "EXECUTION MODE SPLIT - IMMEDIATE VS DELAYED / LIMIT-LIKE FILLS\n"
        "---------------------------------------------------------------\n"
        "No sufficient trade data was available for execution-mode analysis.\n"
    )

    base = trades_df.copy()
    if base.empty or "opened_dt_utc" not in base.columns:
        return {
            "trade_split": empty,
            "fill_delay_summary": empty,
            "mode_comparison": empty,
            "eth_trades": empty,
            "section_text": empty_section,
        }

    base = base.merge(
        entry_quality_df[
            [
                "trade_number",
                "realized_return_pct",
                "MAE_pct",
                "MFE_pct",
                "time_to_first_positive_minutes",
            ]
        ],
        on="trade_number",
        how="left",
    )

    cluster_lookup = eth_clusters_df[["trade_number", "cluster_id", "cluster_description"]].drop_duplicates("trade_number")
    base = base.merge(cluster_lookup, on="trade_number", how="left")

    symbols = sorted(base["symbol"].dropna().astype(str).unique().tolist())
    market_cache = load_symbol_market_data(base, symbols, logger)
    trade_rows = []
    for _, trade in base.sort_values("opened_dt_utc").iterrows():
        trade_rows.append(classify_trade_execution_mode(trade, market_cache))

    trade_split = pd.DataFrame(trade_rows)
    if trade_split.empty:
        return {
            "trade_split": empty,
            "fill_delay_summary": empty,
            "mode_comparison": empty,
            "eth_trades": empty,
            "section_text": empty_section,
        }

    eth_trades = trade_split.loc[trade_split["symbol"] == PRIMARY_SYMBOL].copy()
    fill_delay_summary = build_fill_delay_summary(trade_split)
    mode_comparison = build_execution_mode_comparison(trade_split)
    section_text = build_execution_mode_section(trade_split, fill_delay_summary, mode_comparison)

    return {
        "trade_split": trade_split,
        "fill_delay_summary": fill_delay_summary,
        "mode_comparison": mode_comparison,
        "eth_trades": eth_trades,
        "section_text": section_text,
    }


def load_symbol_market_data(trades_df: pd.DataFrame, symbols: list[str], logger: logging.Logger) -> dict[str, dict[str, pd.DataFrame]]:
    market_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        symbol_trades = trades_df.loc[trades_df["symbol"] == symbol].copy()
        if symbol_trades.empty:
            continue
        start_time = pd.to_datetime(symbol_trades["opened_dt_utc"], utc=True, errors="coerce").min()
        end_time = pd.to_datetime(symbol_trades["closed_dt_utc"], utc=True, errors="coerce").max()
        if pd.isna(start_time) or pd.isna(end_time):
            market_cache[symbol] = {"1m": pd.DataFrame(), "5m": pd.DataFrame()}
            continue
        start_ms = int((start_time - pd.Timedelta(minutes=5)).timestamp() * 1000)
        end_ms = int((end_time + pd.Timedelta(minutes=5)).timestamp() * 1000)
        logger.info("Loading execution-mode candle windows for %s", symbol)
        frame_1m = fetch_futures_klines(symbol, start_ms, end_ms, interval="1m", logger=logger)
        frame_5m = fetch_futures_klines(symbol, start_ms, end_ms, interval="5m", logger=logger)
        market_cache[symbol] = {"1m": frame_1m, "5m": frame_5m}
    return market_cache


def classify_trade_execution_mode(trade: pd.Series, market_cache: dict[str, dict[str, pd.DataFrame]]) -> dict:
    symbol = str(trade.get("symbol"))
    opened = pd.to_datetime(trade.get("opened_dt_utc"), utc=True, errors="coerce")
    closed = pd.to_datetime(trade.get("closed_dt_utc"), utc=True, errors="coerce")
    avg_cost = pd.to_numeric(pd.Series([trade.get("avgCost")]), errors="coerce").iloc[0]
    side = str(trade.get("side")) if pd.notna(trade.get("side")) else "n/a"

    base_row = {
        "trade_number": trade.get("trade_number"),
        "symbol": symbol,
        "side": side,
        "opened_dt_utc": opened,
        "closed_dt_utc": closed,
        "avgCost": avg_cost,
        "avgClosePrice": trade.get("avgClosePrice"),
        "closingPnl": trade.get("closingPnl"),
        "return_pct_approx": trade.get("return_pct_approx"),
        "holding_hours": trade.get("holding_hours"),
        "open_hour_utc": trade.get("open_hour_utc"),
        "realized_return_pct": trade.get("realized_return_pct"),
        "MAE_pct": trade.get("MAE_pct"),
        "MFE_pct": trade.get("MFE_pct"),
        "time_to_first_positive_minutes": trade.get("time_to_first_positive_minutes"),
        "cluster_id": trade.get("cluster_id"),
        "cluster_description": trade.get("cluster_description"),
    }

    if pd.isna(opened) or pd.isna(avg_cost) or symbol not in market_cache:
        return {
            **base_row,
            "interval_used": "none",
            "touch_found": False,
            "touch_in_open_candle_flag": False,
            "first_touch_time_utc": pd.NaT,
            "fill_delay_minutes": np.nan,
            "fill_delay_bucket": "uncertain",
            "execution_mode": "uncertain",
            "execution_inconsistency_flag": pd.NA,
            "delayed_limit_like_proxy_flag": False,
            "uncertain_flag": True,
            "classification_note": "missing_market_data_or_trade_fields",
        }

    search_end = closed if pd.notna(closed) and closed > opened else opened + pd.Timedelta(hours=24)
    first_choice = analyze_fill_delay_in_frame(market_cache[symbol]["1m"], opened, search_end, avg_cost, interval="1m")
    if first_choice["coverage_available"]:
        chosen = first_choice
    else:
        chosen = analyze_fill_delay_in_frame(market_cache[symbol]["5m"], opened, search_end, avg_cost, interval="5m")

    if not chosen["coverage_available"]:
        return {
            **base_row,
            "interval_used": "none",
            "touch_found": False,
            "touch_in_open_candle_flag": False,
            "first_touch_time_utc": pd.NaT,
            "fill_delay_minutes": np.nan,
            "fill_delay_bucket": "uncertain",
            "execution_mode": "uncertain",
            "execution_inconsistency_flag": pd.NA,
            "delayed_limit_like_proxy_flag": False,
            "uncertain_flag": True,
            "classification_note": "no_candle_coverage",
        }

    fill_delay_minutes = chosen["fill_delay_minutes"]
    touch_found = bool(chosen["touch_found"])
    touch_in_open_candle_flag = bool(chosen["touch_in_open_candle_flag"])
    interval_used = chosen["interval_used"]
    fill_delay_bucket = classify_fill_delay_bucket(fill_delay_minutes, interval_used, touch_in_open_candle_flag, touch_found)
    execution_mode = classify_execution_mode(fill_delay_minutes, interval_used, touch_in_open_candle_flag, touch_found)

    return {
        **base_row,
        "interval_used": interval_used,
        "touch_found": touch_found,
        "touch_in_open_candle_flag": touch_in_open_candle_flag,
        "first_touch_time_utc": chosen["first_touch_time_utc"],
        "fill_delay_minutes": fill_delay_minutes,
        "fill_delay_bucket": fill_delay_bucket,
        "execution_mode": execution_mode,
        "execution_inconsistency_flag": False if touch_in_open_candle_flag else (True if touch_found else pd.NA),
        "delayed_limit_like_proxy_flag": execution_mode == "delayed_limit_like",
        "uncertain_flag": execution_mode == "uncertain",
        "classification_note": chosen["classification_note"],
    }


def analyze_fill_delay_in_frame(
    frame: pd.DataFrame,
    opened: pd.Timestamp,
    search_end: pd.Timestamp,
    avg_cost: float,
    interval: str,
) -> dict:
    if frame.empty:
        return {
            "coverage_available": False,
            "interval_used": interval,
            "touch_found": False,
            "touch_in_open_candle_flag": False,
            "first_touch_time_utc": pd.NaT,
            "fill_delay_minutes": np.nan,
            "classification_note": "no_frame_data",
        }

    subset = frame.loc[
        (frame["close_time"] >= opened - pd.Timedelta(milliseconds=1))
        & (frame["open_time"] <= search_end)
    ].copy()
    if subset.empty:
        return {
            "coverage_available": False,
            "interval_used": interval,
            "touch_found": False,
            "touch_in_open_candle_flag": False,
            "first_touch_time_utc": pd.NaT,
            "fill_delay_minutes": np.nan,
            "classification_note": "no_subset_coverage",
        }

    open_candle = subset.loc[(subset["open_time"] <= opened) & (subset["close_time"] >= opened)].head(1)
    if open_candle.empty:
        first_after = subset.loc[subset["open_time"] >= opened].head(1)
        open_candle = first_after

    touch_mask = (
        pd.to_numeric(subset["low"], errors="coerce") <= float(avg_cost)
    ) & (
        pd.to_numeric(subset["high"], errors="coerce") >= float(avg_cost)
    )

    touch_in_open_candle = False
    if not open_candle.empty:
        open_candle_low = pd.to_numeric(open_candle["low"], errors="coerce").iloc[0]
        open_candle_high = pd.to_numeric(open_candle["high"], errors="coerce").iloc[0]
        touch_in_open_candle = bool(open_candle_low <= avg_cost <= open_candle_high)

    if touch_in_open_candle:
        return {
            "coverage_available": True,
            "interval_used": interval,
            "touch_found": True,
            "touch_in_open_candle_flag": True,
            "first_touch_time_utc": opened,
            "fill_delay_minutes": 0.0,
            "classification_note": "same_candle_touch_proxy",
        }

    later_touches = subset.loc[touch_mask & (subset["open_time"] > opened)].copy()
    if later_touches.empty:
        return {
            "coverage_available": True,
            "interval_used": interval,
            "touch_found": False,
            "touch_in_open_candle_flag": False,
            "first_touch_time_utc": pd.NaT,
            "fill_delay_minutes": np.nan,
            "classification_note": "no_touch_before_close",
        }

    first_touch = later_touches.sort_values("open_time").iloc[0]
    delay_minutes = max((pd.Timestamp(first_touch["open_time"]) - opened).total_seconds() / 60.0, 0.0)
    return {
        "coverage_available": True,
        "interval_used": interval,
        "touch_found": True,
        "touch_in_open_candle_flag": False,
        "first_touch_time_utc": pd.Timestamp(first_touch["open_time"]),
        "fill_delay_minutes": delay_minutes,
        "classification_note": "later_touch_proxy",
    }


def classify_fill_delay_bucket(
    fill_delay_minutes: float | int | None,
    interval_used: str,
    touch_in_open_candle_flag: bool,
    touch_found: bool,
) -> str:
    if not touch_found or pd.isna(fill_delay_minutes):
        return "uncertain"
    if touch_in_open_candle_flag:
        return "same_1m_candle" if interval_used == "1m" else "same_5m_candle"
    delay = float(fill_delay_minutes)
    if delay <= 5:
        return "1-5m"
    if delay <= 15:
        return "5-15m"
    if delay <= 30:
        return "15-30m"
    if delay <= 60:
        return "30-60m"
    return "60m+"


def classify_execution_mode(
    fill_delay_minutes: float | int | None,
    interval_used: str,
    touch_in_open_candle_flag: bool,
    touch_found: bool,
) -> str:
    if not touch_found or pd.isna(fill_delay_minutes):
        return "uncertain"
    if touch_in_open_candle_flag:
        return "immediate_like"
    delay = float(fill_delay_minutes)
    if delay <= 1:
        return "immediate_like"
    if delay <= 5:
        return "short_delay"
    return "delayed_limit_like"


def build_fill_delay_summary(trade_split: pd.DataFrame) -> pd.DataFrame:
    scopes = build_scopes(trade_split)
    rows = []
    for scope_name, scope_df in scopes.items():
        if scope_df.empty:
            continue
        delay_values = pd.to_numeric(scope_df["fill_delay_minutes"], errors="coerce")
        interval_1m_share = (scope_df["interval_used"] == "1m").mean()
        interval_5m_share = (scope_df["interval_used"] == "5m").mean()
        immediate_share = (scope_df["execution_mode"] == "immediate_like").mean()
        short_delay_share = (scope_df["execution_mode"] == "short_delay").mean()
        delayed_share = (scope_df["execution_mode"] == "delayed_limit_like").mean()
        uncertain_share = (scope_df["execution_mode"] == "uncertain").mean()
        rows.append(
            {
                "scope": scope_name,
                "trade_count": int(len(scope_df)),
                "touch_found_share": scope_df["touch_found"].fillna(False).astype(bool).mean(),
                "same_open_candle_share": scope_df["touch_in_open_candle_flag"].fillna(False).astype(bool).mean(),
                "naive_immediate_inconsistency_share": scope_df["execution_inconsistency_flag"].eq(True).mean(),
                "immediate_like_share": immediate_share,
                "short_delay_share": short_delay_share,
                "delayed_limit_like_share": delayed_share,
                "uncertain_share": uncertain_share,
                "delayed_over_15m_share": (delay_values > 15).mean() if delay_values.notna().any() else np.nan,
                "delayed_over_60m_share": (delay_values > 60).mean() if delay_values.notna().any() else np.nan,
                "median_fill_delay_minutes": delay_values.median(),
                "p75_fill_delay_minutes": delay_values.quantile(0.75) if delay_values.notna().any() else np.nan,
                "p90_fill_delay_minutes": delay_values.quantile(0.90) if delay_values.notna().any() else np.nan,
                "interval_1m_share": interval_1m_share,
                "interval_5m_share": interval_5m_share,
                "two_execution_modes_supported_flag": bool(immediate_share >= 0.15 and delayed_share >= 0.15),
            }
        )
    return pd.DataFrame(rows)


def build_execution_mode_comparison(trade_split: pd.DataFrame) -> pd.DataFrame:
    scopes = build_scopes(trade_split)
    rows = []
    for scope_name, scope_df in scopes.items():
        if scope_df.empty:
            continue
        total = len(scope_df)
        for execution_mode, mode_df in scope_df.groupby("execution_mode", dropna=False):
            if mode_df.empty:
                continue
            rows.append(
                {
                    "scope": scope_name,
                    "execution_mode": execution_mode,
                    "trade_count": int(len(mode_df)),
                    "trade_share": safe_divide(len(mode_df), total),
                    "win_rate": (pd.to_numeric(mode_df["closingPnl"], errors="coerce") > 0).mean(),
                    "avg_pnl": pd.to_numeric(mode_df["closingPnl"], errors="coerce").mean(),
                    "median_pnl": pd.to_numeric(mode_df["closingPnl"], errors="coerce").median(),
                    "avg_return_pct": pd.to_numeric(mode_df["return_pct_approx"], errors="coerce").mean(),
                    "median_return_pct": pd.to_numeric(mode_df["return_pct_approx"], errors="coerce").median(),
                    "avg_holding_hours": pd.to_numeric(mode_df["holding_hours"], errors="coerce").mean(),
                    "median_holding_hours": pd.to_numeric(mode_df["holding_hours"], errors="coerce").median(),
                    "avg_fill_delay_minutes": pd.to_numeric(mode_df["fill_delay_minutes"], errors="coerce").mean(),
                    "median_fill_delay_minutes": pd.to_numeric(mode_df["fill_delay_minutes"], errors="coerce").median(),
                    "avg_MAE_pct": pd.to_numeric(mode_df["MAE_pct"], errors="coerce").mean(),
                    "median_MAE_pct": pd.to_numeric(mode_df["MAE_pct"], errors="coerce").median(),
                    "avg_MFE_pct": pd.to_numeric(mode_df["MFE_pct"], errors="coerce").mean(),
                    "median_MFE_pct": pd.to_numeric(mode_df["MFE_pct"], errors="coerce").median(),
                    "median_time_to_first_profit_minutes": pd.to_numeric(mode_df["time_to_first_positive_minutes"], errors="coerce").median(),
                    "long_share": (mode_df["side"] == "Long").mean(),
                    "short_share": (mode_df["side"] == "Short").mean(),
                    "median_open_hour_utc": pd.to_numeric(mode_df["open_hour_utc"], errors="coerce").median(),
                    "cluster1_share": (pd.to_numeric(mode_df["cluster_id"], errors="coerce") == 1).mean(),
                    "cluster2_share": (pd.to_numeric(mode_df["cluster_id"], errors="coerce") == 2).mean(),
                    "eth_share": (mode_df["symbol"] == PRIMARY_SYMBOL).mean(),
                    "btc_share": (mode_df["symbol"] == SECONDARY_SYMBOL).mean(),
                }
            )
    return pd.DataFrame(rows)


def build_scopes(trade_split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes = {
        "all_symbols": trade_split.copy(),
        PRIMARY_SYMBOL: trade_split.loc[trade_split["symbol"] == PRIMARY_SYMBOL].copy(),
    }
    if (trade_split["symbol"] == SECONDARY_SYMBOL).any():
        scopes[SECONDARY_SYMBOL] = trade_split.loc[trade_split["symbol"] == SECONDARY_SYMBOL].copy()
    return scopes


def build_execution_mode_section(
    trade_split: pd.DataFrame,
    fill_delay_summary: pd.DataFrame,
    mode_comparison: pd.DataFrame,
) -> str:
    if trade_split.empty or fill_delay_summary.empty:
        return (
            "EXECUTION MODE SPLIT - IMMEDIATE VS DELAYED / LIMIT-LIKE FILLS\n"
            "---------------------------------------------------------------\n"
            "Execution-mode split analysis could not be completed.\n"
        )

    eth_summary = fill_delay_summary.loc[fill_delay_summary["scope"] == PRIMARY_SYMBOL]
    all_summary = fill_delay_summary.loc[fill_delay_summary["scope"] == "all_symbols"]
    btc_summary = fill_delay_summary.loc[fill_delay_summary["scope"] == SECONDARY_SYMBOL]

    eth_row = eth_summary.iloc[0] if not eth_summary.empty else pd.Series(dtype=object)
    all_row = all_summary.iloc[0] if not all_summary.empty else pd.Series(dtype=object)
    btc_row = btc_summary.iloc[0] if not btc_summary.empty else pd.Series(dtype=object)

    eth_modes = mode_comparison.loc[mode_comparison["scope"] == PRIMARY_SYMBOL].copy()
    immediate_eth = select_mode_row(eth_modes, "immediate_like")
    delayed_eth = select_mode_row(eth_modes, "delayed_limit_like")

    confidence = determine_execution_mode_confidence(eth_row, immediate_eth, delayed_eth)
    two_modes_supported = bool(eth_row.get("two_execution_modes_supported_flag", False))
    verdict = (
        "ETH data is compatible with at least two execution modes, and this likely weakens naive immediate-fill assumptions in earlier proxy tests."
        if two_modes_supported
        else "ETH data does not cleanly separate into two strong execution modes; the signal remains weak or mixed."
    )

    report = f"""
EXECUTION MODE SPLIT - IMMEDIATE VS DELAYED / LIMIT-LIKE FILLS
---------------------------------------------------------------
This section checks whether the trade history is more compatible with one execution style or with at least two execution modes: near-immediate fills and delayed / limit-like fill proxies.

ETH focus:
- Immediate-like share: {pct(eth_row.get('immediate_like_share')) if not eth_row.empty else 'n/a'}.
- Short-delay share: {pct(eth_row.get('short_delay_share')) if not eth_row.empty else 'n/a'}.
- Delayed / limit-like share: {pct(eth_row.get('delayed_limit_like_share')) if not eth_row.empty else 'n/a'}.
- Uncertain share: {pct(eth_row.get('uncertain_share')) if not eth_row.empty else 'n/a'}.
- Naive immediate-fill inconsistency share: {pct(eth_row.get('naive_immediate_inconsistency_share')) if not eth_row.empty else 'n/a'}.
- Median fill-delay proxy: {fmt_number(eth_row.get('median_fill_delay_minutes')) if not eth_row.empty else 'n/a'} minutes.

ETH mode comparison:
- Immediate-like ETH trades: {fmt_int(immediate_eth.get('trade_count')) if not immediate_eth.empty else 'n/a'}, win rate {pct(immediate_eth.get('win_rate')) if not immediate_eth.empty else 'n/a'}, median MAE {fmt_number(immediate_eth.get('median_MAE_pct')) if not immediate_eth.empty else 'n/a'}%, median MFE {fmt_number(immediate_eth.get('median_MFE_pct')) if not immediate_eth.empty else 'n/a'}%.
- Delayed / limit-like ETH trades: {fmt_int(delayed_eth.get('trade_count')) if not delayed_eth.empty else 'n/a'}, win rate {pct(delayed_eth.get('win_rate')) if not delayed_eth.empty else 'n/a'}, median MAE {fmt_number(delayed_eth.get('median_MAE_pct')) if not delayed_eth.empty else 'n/a'}%, median MFE {fmt_number(delayed_eth.get('median_MFE_pct')) if not delayed_eth.empty else 'n/a'}%.
- Side mix and timing: immediate-like short share {pct(immediate_eth.get('short_share')) if not immediate_eth.empty else 'n/a'} vs delayed {pct(delayed_eth.get('short_share')) if not delayed_eth.empty else 'n/a'}; median open hour {fmt_number(immediate_eth.get('median_open_hour_utc'), 1) if not immediate_eth.empty else 'n/a'} vs {fmt_number(delayed_eth.get('median_open_hour_utc'), 1) if not delayed_eth.empty else 'n/a'} UTC.
- Cluster mix: immediate-like Cluster 1 / Cluster 2 shares = {pct(immediate_eth.get('cluster1_share')) if not immediate_eth.empty else 'n/a'} / {pct(immediate_eth.get('cluster2_share')) if not immediate_eth.empty else 'n/a'}; delayed = {pct(delayed_eth.get('cluster1_share')) if not delayed_eth.empty else 'n/a'} / {pct(delayed_eth.get('cluster2_share')) if not delayed_eth.empty else 'n/a'}.

Secondary summary:
- All symbols delayed / limit-like share: {pct(all_row.get('delayed_limit_like_share')) if not all_row.empty else 'n/a'}.
- BTC delayed / limit-like share: {pct(btc_row.get('delayed_limit_like_share')) if not btc_row.empty else 'n/a'}.
- Confidence level: {confidence}.

Final verdict:
- {verdict}
- This split is worth carrying forward into later analysis, because single-mode proxy models likely understate execution-sensitive parts of the trader's edge.
""".strip()
    return report + "\n"


def determine_execution_mode_confidence(
    eth_row: pd.Series,
    immediate_eth: pd.Series,
    delayed_eth: pd.Series,
) -> str:
    if eth_row.empty or immediate_eth.empty or delayed_eth.empty:
        return "low"
    if (
        safe_number(eth_row.get("immediate_like_share")) >= 0.20
        and safe_number(eth_row.get("delayed_limit_like_share")) >= 0.20
        and abs(safe_number(immediate_eth.get("median_MAE_pct")) - safe_number(delayed_eth.get("median_MAE_pct"))) >= 0.25
    ):
        return "high"
    if (
        safe_number(eth_row.get("immediate_like_share")) >= 0.10
        and safe_number(eth_row.get("delayed_limit_like_share")) >= 0.10
    ):
        return "medium"
    return "low"


def select_mode_row(mode_df: pd.DataFrame, execution_mode: str) -> pd.Series:
    row = mode_df.loc[mode_df["execution_mode"] == execution_mode]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def create_execution_mode_split_plots(
    trade_split: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if trade_split.empty:
        return
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    distribution_path = plots_dir / "execution_fill_delay_distribution.png"
    pnl_path = plots_dir / "execution_mode_pnl_comparison.png"
    time_path = plots_dir / "execution_mode_time_of_day.png"

    fig_distribution = plot_execution_fill_delay_distribution(trade_split)
    fig_distribution.savefig(distribution_path, dpi=150, bbox_inches="tight")
    plt.close(fig_distribution)
    logger.info("Saved plot: %s", distribution_path.name)

    fig_pnl = plot_execution_mode_pnl_comparison(trade_split)
    fig_pnl.savefig(pnl_path, dpi=150, bbox_inches="tight")
    plt.close(fig_pnl)
    logger.info("Saved plot: %s", pnl_path.name)

    fig_time = plot_execution_mode_time_of_day(trade_split)
    fig_time.savefig(time_path, dpi=150, bbox_inches="tight")
    plt.close(fig_time)
    logger.info("Saved plot: %s", time_path.name)


def plot_execution_fill_delay_distribution(trade_split: pd.DataFrame) -> plt.Figure:
    eth_df = trade_split.loc[trade_split["symbol"] == PRIMARY_SYMBOL].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if eth_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No ETH execution data", ha="center", va="center")
            ax.set_axis_off()
        return fig

    bucket_order = ["same_1m_candle", "same_5m_candle", "1-5m", "5-15m", "15-30m", "30-60m", "60m+", "uncertain"]
    bucket_counts = eth_df["fill_delay_bucket"].value_counts().reindex(bucket_order, fill_value=0)
    axes[0].bar(np.arange(len(bucket_counts)), bucket_counts.values, color="#4c78a8")
    axes[0].set_xticks(np.arange(len(bucket_counts)))
    axes[0].set_xticklabels(bucket_counts.index, rotation=30, ha="right")
    axes[0].set_title("ETH fill-delay bucket distribution")
    axes[0].set_ylabel("Trade count")
    axes[0].grid(axis="y", alpha=0.25)

    delay_values = pd.to_numeric(eth_df["fill_delay_minutes"], errors="coerce").dropna()
    if delay_values.empty:
        axes[1].text(0.5, 0.5, "No delay data", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        clipped = np.clip(delay_values, 0, max(60, np.nanpercentile(delay_values, 95)))
        axes[1].hist(clipped, bins=20, edgecolor="black", color="#f58518")
        axes[1].axvline(delay_values.median(), color="#d62728", linestyle="--", label="Median")
        axes[1].set_title("ETH fill-delay proxy histogram")
        axes[1].set_xlabel("Fill delay proxy (minutes, clipped)")
        axes[1].set_ylabel("Trade count")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
    return fig


def plot_execution_mode_pnl_comparison(trade_split: pd.DataFrame) -> plt.Figure:
    eth_df = trade_split.loc[
        (trade_split["symbol"] == PRIMARY_SYMBOL)
        & trade_split["execution_mode"].isin(["immediate_like", "short_delay", "delayed_limit_like", "uncertain"])
    ].copy()
    ordered = (
        eth_df.groupby("execution_mode", dropna=False)
        .agg(
            median_pnl=("closingPnl", "median"),
            median_return_pct=("return_pct_approx", "median"),
        )
        .reindex(["immediate_like", "short_delay", "delayed_limit_like", "uncertain"])
        .fillna(0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(ordered.index.astype(str), ordered["median_pnl"], color="#54a24b")
    axes[0].set_title("ETH median PnL by execution mode")
    axes[0].set_ylabel("Median PnL")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(ordered.index.astype(str), ordered["median_return_pct"], color="#e45756")
    axes[1].set_title("ETH median return % by execution mode")
    axes[1].set_ylabel("Median return %")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    return fig


def plot_execution_mode_time_of_day(trade_split: pd.DataFrame) -> plt.Figure:
    eth_df = trade_split.loc[trade_split["symbol"] == PRIMARY_SYMBOL].copy()
    order = ["immediate_like", "short_delay", "delayed_limit_like", "uncertain"]
    hourly = (
        eth_df.groupby(["open_hour_utc", "execution_mode"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    if hourly.empty:
        ax.text(0.5, 0.5, "No ETH execution timing data", ha="center", va="center")
        ax.set_axis_off()
        return fig
    bottom = np.zeros(len(hourly))
    colors = {
        "immediate_like": "#4c78a8",
        "short_delay": "#f58518",
        "delayed_limit_like": "#54a24b",
        "uncertain": "#bdbdbd",
    }
    for mode in order:
        values = hourly[mode].to_numpy(dtype=float)
        ax.bar(hourly.index.astype(int), values, bottom=bottom, label=mode, color=colors[mode])
        bottom += values
    ax.set_title("ETH execution modes by open hour (UTC)")
    ax.set_xlabel("Open hour (UTC)")
    ax.set_ylabel("Trade count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0
