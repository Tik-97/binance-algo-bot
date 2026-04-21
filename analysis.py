import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import HOLDING_BUCKETS, MONTH_ORDER, PLOTS_DIRNAME, UTC_LABEL, WEEKDAY_ORDER
from utils import coerce_numeric, ensure_columns, fmt_hours, fmt_int, fmt_number, pct, safe_divide


def normalize_trades(raw_df: pd.DataFrame, logger: logging.Logger, parse_datetime_to_utc) -> tuple[pd.DataFrame, dict]:
    df = ensure_columns(raw_df, logger).copy()

    df["opened_raw"] = df["opened"]
    df["closed_raw"] = df["closed"]
    df["updateTime_raw"] = df["updateTime"]

    for column in ["avgCost", "avgClosePrice", "closedVolume", "closingPnl", "maxOpenInterest", "id"]:
        df[column] = coerce_numeric(df[column])

    for column in ["symbol", "side", "status"]:
        df[column] = df[column].astype("string").str.strip()

    df["opened_dt_utc"] = parse_datetime_to_utc(df["opened"])
    df["closed_dt_utc"] = parse_datetime_to_utc(df["closed"])
    df["updateTime_dt_utc"] = parse_datetime_to_utc(df["updateTime"])

    df["holding_minutes"] = (df["closed_dt_utc"] - df["opened_dt_utc"]).dt.total_seconds() / 60.0
    df["holding_hours"] = df["holding_minutes"] / 60.0

    df["trade_outcome"] = np.where(
        df["closingPnl"] > 0,
        "win",
        np.where(df["closingPnl"] < 0, "loss", "flat"),
    )

    df["return_pct_approx"] = np.where(
        df["side"].str.lower() == "long",
        (df["avgClosePrice"] - df["avgCost"]) / df["avgCost"],
        np.where(
            df["side"].str.lower() == "short",
            (df["avgCost"] - df["avgClosePrice"]) / df["avgCost"],
            np.nan,
        ),
    )

    df["approximate_notional"] = df["avgCost"] * df["closedVolume"]
    df["normalized_position_notional"] = df["approximate_notional"]
    df["pnl_per_hour"] = df["closingPnl"] / df["holding_hours"].replace(0, np.nan)
    df["pnl_per_trade"] = df["closingPnl"]

    df["open_hour_utc"] = df["opened_dt_utc"].dt.hour
    df["open_weekday_utc"] = pd.Categorical(
        df["opened_dt_utc"].dt.day_name(),
        categories=WEEKDAY_ORDER,
        ordered=True,
    )
    df["open_month_utc"] = pd.Categorical(
        df["opened_dt_utc"].dt.month_name(),
        categories=MONTH_ORDER,
        ordered=True,
    )

    df["holding_bucket"] = pd.cut(
        df["holding_hours"],
        bins=[bucket[1] for bucket in HOLDING_BUCKETS] + [HOLDING_BUCKETS[-1][2]],
        labels=[bucket[0] for bucket in HOLDING_BUCKETS],
        include_lowest=True,
        right=False,
    )

    if "id" in df.columns:
        sort_keys = ["opened_dt_utc", "closed_dt_utc", "id"]
    else:
        sort_keys = ["opened_dt_utc", "closed_dt_utc"]

    df = df.sort_values(sort_keys, na_position="last").reset_index(drop=True)
    df["trade_number"] = np.arange(1, len(df) + 1)
    df["cumulative_pnl"] = df["closingPnl"].fillna(0).cumsum()

    prior_outcome = df["trade_outcome"].shift(1)
    prior_notional = df["approximate_notional"].shift(1)
    df["prev_trade_outcome"] = prior_outcome
    df["position_size_change_ratio"] = df["approximate_notional"] / prior_notional.replace(0, np.nan)

    win_streak = []
    loss_streak = []
    current_win_streak = 0
    current_loss_streak = 0
    for pnl in df["closingPnl"].fillna(0):
        if pnl > 0:
            current_win_streak += 1
            current_loss_streak = 0
        elif pnl < 0:
            current_loss_streak += 1
            current_win_streak = 0
        else:
            current_win_streak = 0
            current_loss_streak = 0
        win_streak.append(current_win_streak)
        loss_streak.append(current_loss_streak)

    df["win_streak"] = win_streak
    df["loss_streak"] = loss_streak
    df["prev_win_streak"] = df["win_streak"].shift(1).fillna(0)
    df["prev_loss_streak"] = df["loss_streak"].shift(1).fillna(0)

    quality = assess_data_quality(df)
    return df, quality


def assess_data_quality(df: pd.DataFrame) -> dict:
    quality = {
        "rows": int(len(df)),
        "missing_symbol": int(df["symbol"].isna().sum()),
        "missing_side": int(df["side"].isna().sum()),
        "missing_opened": int(df["opened_dt_utc"].isna().sum()),
        "missing_closed": int(df["closed_dt_utc"].isna().sum()),
        "missing_pnl": int(df["closingPnl"].isna().sum()),
        "missing_volume": int(df["closedVolume"].isna().sum()),
        "negative_holding_rows": int((df["holding_minutes"] < 0).fillna(False).sum()),
        "duplicate_ids": int(df["id"].dropna().duplicated().sum()) if "id" in df.columns else 0,
    }
    quality["invalid_rows_total"] = int(
        (
            df["opened_dt_utc"].isna()
            | df["closed_dt_utc"].isna()
            | df["closingPnl"].isna()
            | (df["holding_minutes"] < 0).fillna(False)
        ).sum()
    )
    return quality


def basic_stats(df: pd.DataFrame) -> dict:
    total_trades = len(df)
    win_mask = df["closingPnl"] > 0
    valid_pnl = df["closingPnl"].dropna()
    return {
        "total_trades": total_trades,
        "long_trades": int((df["side"].str.lower() == "long").sum()),
        "short_trades": int((df["side"].str.lower() == "short").sum()),
        "win_rate": safe_divide(int(win_mask.sum()), int(valid_pnl.count())),
        "avg_pnl": valid_pnl.mean(),
        "median_pnl": valid_pnl.median(),
        "total_pnl": valid_pnl.sum(),
        "avg_holding_hours": df["holding_hours"].dropna().mean(),
        "median_holding_hours": df["holding_hours"].dropna().median(),
        "avg_position_size": df["closedVolume"].dropna().mean(),
        "median_position_size": df["closedVolume"].dropna().median(),
        "avg_notional": df["approximate_notional"].dropna().mean(),
        "median_return_pct": df["return_pct_approx"].dropna().median(),
        "volume_weighted_pnl": np.average(
            df["closingPnl"].dropna(),
            weights=df.loc[df["closingPnl"].notna(), "closedVolume"].fillna(0) + 1e-9,
        )
        if not df["closingPnl"].dropna().empty
        else np.nan,
    }


def build_symbol_stats(df: pd.DataFrame) -> pd.DataFrame:
    total_trades = max(len(df), 1)
    total_pnl = df["closingPnl"].sum()

    grouped = (
        df.groupby("symbol", dropna=False)
        .agg(
            trades=("symbol", "size"),
            total_pnl=("closingPnl", "sum"),
            avg_pnl=("closingPnl", "mean"),
            median_pnl=("closingPnl", "median"),
            win_rate=("closingPnl", lambda s: safe_divide((s > 0).sum(), s.notna().sum())),
            avg_holding_hours=("holding_hours", "mean"),
            avg_volume=("closedVolume", "mean"),
        )
        .reset_index()
    )
    grouped["trade_share"] = grouped["trades"] / total_trades
    grouped["pnl_share"] = grouped["total_pnl"].apply(lambda x: safe_divide(x, total_pnl))
    return grouped.sort_values(["total_pnl", "trades"], ascending=[False, False]).reset_index(drop=True)


def build_side_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("side", dropna=False)
        .agg(
            trades=("side", "size"),
            total_pnl=("closingPnl", "sum"),
            avg_pnl=("closingPnl", "mean"),
            median_pnl=("closingPnl", "median"),
            win_rate=("closingPnl", lambda s: safe_divide((s > 0).sum(), s.notna().sum())),
            avg_holding_hours=("holding_hours", "mean"),
            avg_volume=("closedVolume", "mean"),
            avg_notional=("approximate_notional", "mean"),
            median_return_pct=("return_pct_approx", "median"),
        )
        .reset_index()
    )
    return grouped.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def build_time_stats(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    hour_stats = (
        df.groupby("open_hour_utc", dropna=False)
        .agg(trades=("trade_number", "size"), total_pnl=("closingPnl", "sum"), avg_pnl=("closingPnl", "mean"))
        .reset_index()
        .rename(columns={"open_hour_utc": "bucket"})
    )
    hour_stats["dimension"] = "open_hour_utc"

    weekday_stats = (
        df.groupby("open_weekday_utc", observed=False, dropna=False)
        .agg(trades=("trade_number", "size"), total_pnl=("closingPnl", "sum"), avg_pnl=("closingPnl", "mean"))
        .reset_index()
        .rename(columns={"open_weekday_utc": "bucket"})
    )
    weekday_stats["dimension"] = "open_weekday_utc"

    month_stats = (
        df.groupby("open_month_utc", observed=False, dropna=False)
        .agg(trades=("trade_number", "size"), total_pnl=("closingPnl", "sum"), avg_pnl=("closingPnl", "mean"))
        .reset_index()
        .rename(columns={"open_month_utc": "bucket"})
    )
    month_stats["dimension"] = "open_month_utc"

    holding_bucket_stats = (
        df.groupby("holding_bucket", observed=False, dropna=False)
        .agg(
            trades=("trade_number", "size"),
            total_pnl=("closingPnl", "sum"),
            avg_pnl=("closingPnl", "mean"),
            median_pnl=("closingPnl", "median"),
            win_rate=("closingPnl", lambda s: safe_divide((s > 0).sum(), s.notna().sum())),
            avg_holding_hours=("holding_hours", "mean"),
        )
        .reset_index()
        .rename(columns={"holding_bucket": "bucket"})
    )
    holding_bucket_stats["dimension"] = "holding_bucket"

    frames = {
        "open_hour_utc": hour_stats,
        "open_weekday_utc": weekday_stats,
        "open_month_utc": month_stats,
        "holding_bucket": holding_bucket_stats,
    }
    combined = pd.concat(frames.values(), ignore_index=True, sort=False)
    return frames, combined


def analyze_concentration(df: pd.DataFrame) -> dict:
    pnl_sorted_desc = df.sort_values("closingPnl", ascending=False)
    pnl_sorted_asc = df.sort_values("closingPnl", ascending=True)

    top_5 = pnl_sorted_desc.head(5)[
        ["trade_number", "symbol", "side", "opened_dt_utc", "closed_dt_utc", "closingPnl", "holding_hours"]
    ]
    bottom_5 = pnl_sorted_asc.head(5)[
        ["trade_number", "symbol", "side", "opened_dt_utc", "closed_dt_utc", "closingPnl", "holding_hours"]
    ]

    total_pnl = df["closingPnl"].sum()
    top_5_pnl = top_5["closingPnl"].sum()
    positive_pnl = df.loc[df["closingPnl"] > 0, "closingPnl"].sum()

    concentration_ratio_total = safe_divide(top_5_pnl, total_pnl)
    concentration_ratio_positive = safe_divide(top_5_pnl, positive_pnl)

    tail_ratio = safe_divide(
        df["closingPnl"].quantile(0.95),
        df["closingPnl"].median(),
    )

    return {
        "top_5": top_5,
        "bottom_5": bottom_5,
        "top_5_pnl": top_5_pnl,
        "top_5_pct_total_pnl": concentration_ratio_total,
        "top_5_pct_positive_pnl": concentration_ratio_positive,
        "tail_ratio_95_to_median": tail_ratio,
        "fat_tail_signal": bool(concentration_ratio_positive >= 0.5 or tail_ratio >= 4),
    }


def analyze_position_behavior(df: pd.DataFrame) -> dict:
    next_trade_df = df.copy()
    next_trade_df["prev_outcome"] = next_trade_df["trade_outcome"].shift(1)

    after_win = next_trade_df.loc[next_trade_df["prev_outcome"] == "win", "approximate_notional"]
    after_loss = next_trade_df.loc[next_trade_df["prev_outcome"] == "loss", "approximate_notional"]

    ratio_after_loss_vs_win = safe_divide(after_loss.mean(), after_win.mean())

    aggressive_after_loss = next_trade_df.loc[
        next_trade_df["prev_outcome"] == "loss", "position_size_change_ratio"
    ]
    aggressive_after_win = next_trade_df.loc[
        next_trade_df["prev_outcome"] == "win", "position_size_change_ratio"
    ]

    martingale_signal = bool(
        safe_divide((aggressive_after_loss > 1.25).sum(), aggressive_after_loss.notna().sum()) >= 0.3
        and ratio_after_loss_vs_win >= 1.1
    )

    return {
        "max_win_streak": int(df["win_streak"].max()) if not df.empty else 0,
        "max_loss_streak": int(df["loss_streak"].max()) if not df.empty else 0,
        "avg_notional_after_win": after_win.mean(),
        "avg_notional_after_loss": after_loss.mean(),
        "avg_volume_after_win": next_trade_df.loc[next_trade_df["prev_outcome"] == "win", "closedVolume"].mean(),
        "avg_volume_after_loss": next_trade_df.loc[next_trade_df["prev_outcome"] == "loss", "closedVolume"].mean(),
        "ratio_after_loss_vs_win": ratio_after_loss_vs_win,
        "avg_size_change_after_win": aggressive_after_win.mean(),
        "avg_size_change_after_loss": aggressive_after_loss.mean(),
        "martingale_signal": martingale_signal,
    }


def analyze_holding_efficiency(df: pd.DataFrame, holding_bucket_stats: pd.DataFrame) -> dict:
    winners = df.loc[df["closingPnl"] > 0]
    losers = df.loc[df["closingPnl"] < 0]

    profitable_bucket = (
        holding_bucket_stats.sort_values("total_pnl", ascending=False).head(1)["bucket"].iloc[0]
        if not holding_bucket_stats.empty
        else "n/a"
    )
    most_efficient_bucket = (
        holding_bucket_stats.sort_values("avg_pnl", ascending=False).head(1)["bucket"].iloc[0]
        if not holding_bucket_stats.empty
        else "n/a"
    )

    return {
        "avg_holding_winners": winners["holding_hours"].mean(),
        "avg_holding_losers": losers["holding_hours"].mean(),
        "median_holding_winners": winners["holding_hours"].median(),
        "median_holding_losers": losers["holding_hours"].median(),
        "profitable_bucket": profitable_bucket,
        "most_efficient_bucket": most_efficient_bucket,
    }


def classify_trader_style(df: pd.DataFrame, symbol_stats: pd.DataFrame, side_stats: pd.DataFrame, concentration: dict) -> dict:
    median_holding = df["holding_hours"].median()
    share_under_24h = safe_divide((df["holding_hours"] < 24).sum(), df["holding_hours"].notna().sum())
    top_symbol_trade_share = symbol_stats["trade_share"].max() if not symbol_stats.empty else np.nan
    top_symbol_pnl_share = symbol_stats["pnl_share"].max() if not symbol_stats.empty else np.nan

    if pd.isna(median_holding):
        style = "undetermined"
    elif median_holding < 0.25:
        style = "scalper"
    elif share_under_24h >= 0.8:
        style = "intraday trader"
    elif median_holding >= 24:
        style = "swing trader"
    else:
        style = "mixed intraday/swing trader"

    concentration_style = (
        "high concentration in a few instruments"
        if pd.notna(top_symbol_trade_share) and top_symbol_trade_share >= 0.4
        else "moderate diversification across symbols"
    )

    side_bias = "balanced"
    if not side_stats.empty and len(side_stats) >= 2:
        best_side_trades = side_stats.sort_values("trades", ascending=False).iloc[0]
        if safe_divide(best_side_trades["trades"], len(df)) >= 0.65:
            side_bias = f"strong {best_side_trades['side']} bias"

    stability = (
        "returns appear concentrated in a few outsized trades"
        if concentration["fat_tail_signal"]
        else "results look less dependent on a handful of extreme trades"
    )

    return {
        "style": style,
        "concentration_style": concentration_style,
        "side_bias": side_bias,
        "stability": stability,
        "top_symbol_trade_share": top_symbol_trade_share,
        "top_symbol_pnl_share": top_symbol_pnl_share,
    }


def best_and_worst(frame: pd.DataFrame, label_col: str) -> tuple[str, str]:
    if frame.empty:
        return "n/a", "n/a"
    ordered = frame.sort_values("total_pnl", ascending=False)
    return str(ordered.iloc[0][label_col]), str(ordered.iloc[-1][label_col])


def short_table_to_text(df: pd.DataFrame, digits: int = 2) -> str:
    if df.empty:
        return "n/a"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(digits)
    return display.to_string(index=False)


def generate_summary_report(
    df: pd.DataFrame,
    quality: dict,
    basic: dict,
    profile_stats: pd.DataFrame,
    symbol_stats: pd.DataFrame,
    side_stats: pd.DataFrame,
    time_frames: dict[str, pd.DataFrame],
    concentration: dict,
    position_behavior: dict,
    holding_efficiency: dict,
    style: dict,
    edge_checks: pd.DataFrame,
    simulation_summary: pd.DataFrame,
    market_filter_stats: pd.DataFrame | None = None,
    market_filter_note: str | None = None,
    entry_quality_section: str | None = None,
    execution_mode_section: str | None = None,
    market_context_section: str | None = None,
    eth_context_section: str | None = None,
    eth_bollinger_section: str | None = None,
    eth_setup_clustering_section: str | None = None,
    eth_execution_aware_section: str | None = None,
    eth_execution_reanchored_section: str | None = None,
    eth_cluster1_short_overlap_section: str | None = None,
    eth_cluster1_short_refinement_section: str | None = None,
    eth_cluster1_short_timing_error_section: str | None = None,
    eth_cluster1_short_final_timing_section: str | None = None,
    eth_cluster_rule_section: str | None = None,
    eth_replication_section: str | None = None,
    eth_tightening_section: str | None = None,
    eth_cluster1_micro_trigger_section: str | None = None,
    eth_cluster1_structural_confirmation_section: str | None = None,
    eth_cluster1_effort_result_section: str | None = None,
    eth_practical_imitation_section: str | None = None,
    eth_short_proxy_validation_section: str | None = None,
    core_vs_noise_section: str | None = None,
) -> str:
    best_symbol, worst_symbol = best_and_worst(symbol_stats, "symbol")
    best_side, worst_side = best_and_worst(side_stats, "side")

    symbol_distribution = (
        symbol_stats[["symbol", "trades", "trade_share", "total_pnl", "pnl_share"]]
        .copy()
        .head(10)
    )
    if not symbol_distribution.empty:
        symbol_distribution["trade_share"] = symbol_distribution["trade_share"].apply(pct)
        symbol_distribution["pnl_share"] = symbol_distribution["pnl_share"].apply(pct)

    confirmed_lines = [
        f"Trade history reconstruction points to a {style['style']} profile.",
        f"Best symbol by total PnL: {best_symbol}. Worst symbol by total PnL: {worst_symbol}.",
        f"Best side by total PnL: {best_side}. Worst side by total PnL: {worst_side}.",
        f"Average holding time: {fmt_hours(basic['avg_holding_hours'])}; median: {fmt_hours(basic['median_holding_hours'])}.",
        f"Top-5 trades contributed {pct(concentration['top_5_pct_total_pnl'])} of total PnL and {pct(concentration['top_5_pct_positive_pnl'])} of all profitable PnL.",
    ]

    hypothesis_lines = [
        f"Instrument selection looks like {style['concentration_style']}.",
        f"Directionality looks {style['side_bias']}.",
        f"PnL stability assessment: {style['stability']}.",
        (
            "There are signs of aggressive size increases after losses."
            if position_behavior["martingale_signal"]
            else "No strong martingale signal was detected from sequential position sizing alone."
        ),
    ]

    unknown_lines = [
        "Without candles, order-level history, and market context, exact entry logic cannot be reconstructed.",
        "Trend-following vs counter-trend behavior can only be inferred weakly from holding patterns and side concentration.",
        "We cannot know whether exits were discretionary, signal-driven, liquidation-driven, or copied from partial fills.",
    ]

    profile_row = profile_stats.iloc[0].to_dict() if not profile_stats.empty else {}
    simulation_median_pnl = simulation_summary["total_pnl"].median() if not simulation_summary.empty else np.nan
    simulation_positive_share = safe_divide(
        (simulation_summary["total_pnl"] > 0).sum(),
        simulation_summary["total_pnl"].notna().sum(),
    ) if not simulation_summary.empty else np.nan
    simulation_median_dd = simulation_summary["approx_max_drawdown"].median() if not simulation_summary.empty else np.nan
    simulation_describe = (
        simulation_summary.describe().transpose().reset_index().rename(columns={"index": "metric"})
        if not simulation_summary.empty
        else pd.DataFrame()
    )
    baseline_total_pnl = basic["total_pnl"]
    no_top5_total_pnl = edge_checks.loc[edge_checks["scenario"] == "without_top_5_trades", "total_pnl"]
    top5_removed_pnl = no_top5_total_pnl.iloc[0] if not no_top5_total_pnl.empty else np.nan

    report = f"""
REVERSE ANALYSIS OF EXECUTED TRADES
===================================

This is not a strict strategy backtest on OHLCV candles. It is a trade history reconstruction / behavioral trading analysis built only from executed trades. The replication layer below is a behavioral replication + probabilistic trading model, not a signal-discovery engine.

Data Quality
------------
Rows loaded: {fmt_int(quality['rows'])}
Missing symbol: {fmt_int(quality['missing_symbol'])}
Missing side: {fmt_int(quality['missing_side'])}
Missing opened timestamp: {fmt_int(quality['missing_opened'])}
Missing closed timestamp: {fmt_int(quality['missing_closed'])}
Missing PnL: {fmt_int(quality['missing_pnl'])}
Missing volume: {fmt_int(quality['missing_volume'])}
Negative holding rows: {fmt_int(quality['negative_holding_rows'])}
Duplicate IDs: {fmt_int(quality['duplicate_ids'])}
Potentially invalid rows: {fmt_int(quality['invalid_rows_total'])}

A. Basic Statistics
-------------------
Total trades: {fmt_int(basic['total_trades'])}
Long trades: {fmt_int(basic['long_trades'])}
Short trades: {fmt_int(basic['short_trades'])}
Win rate: {pct(basic['win_rate'])}
Average PnL: {fmt_number(basic['avg_pnl'])}
Median PnL: {fmt_number(basic['median_pnl'])}
Total PnL: {fmt_number(basic['total_pnl'])}
Average holding time: {fmt_hours(basic['avg_holding_hours'])}
Median holding time: {fmt_hours(basic['median_holding_hours'])}
Average position size: {fmt_number(basic['avg_position_size'])}
Median position size: {fmt_number(basic['median_position_size'])}
Average approximate notional: {fmt_number(basic['avg_notional'])}
Median approximate return: {pct(basic['median_return_pct'])}
Volume-weighted average PnL: {fmt_number(basic['volume_weighted_pnl'])}

Behavioral Profile
------------------
Trades per week, mean: {fmt_number(profile_row.get('trades_per_week_mean'))}
Trades per week, median: {fmt_number(profile_row.get('trades_per_week_median'))}
Trades per week, max: {fmt_number(profile_row.get('trades_per_week_max'))}
Average concurrent positions: {fmt_number(profile_row.get('average_concurrent_positions'))}
Max concurrent positions: {fmt_number(profile_row.get('max_concurrent_positions'))}
Approximate max drawdown (trade-level equity): {fmt_number(profile_row.get('approx_max_drawdown'))}
Mean inter-arrival time between entries: {fmt_hours(profile_row.get('mean_interarrival_hours'))}
Median inter-arrival time between entries: {fmt_hours(profile_row.get('median_interarrival_hours'))}
Clustered entries share <1h: {pct(profile_row.get('clustered_entries_share_lt_1h'))}
Clustered entries share <4h: {pct(profile_row.get('clustered_entries_share_lt_4h'))}
Top symbol trade share: {pct(profile_row.get('top_symbol_trade_share'))}
Top symbol pnl share: {pct(profile_row.get('top_symbol_pnl_share'))}

Symbol Distribution
-------------------
{short_table_to_text(symbol_distribution)}

B. Symbol Analysis
------------------
{short_table_to_text(symbol_stats)}

C. Side Analysis
----------------
{short_table_to_text(side_stats)}

D. Time Analysis ({UTC_LABEL})
-----------------------------
Open hour:
{short_table_to_text(time_frames['open_hour_utc'])}

Open weekday:
{short_table_to_text(time_frames['open_weekday_utc'])}

Open month:
{short_table_to_text(time_frames['open_month_utc'])}

Holding buckets:
{short_table_to_text(time_frames['holding_bucket'])}

E. Profit Concentration
-----------------------
Top 5 trades:
{short_table_to_text(concentration['top_5'])}

Worst 5 trades:
{short_table_to_text(concentration['bottom_5'])}

Top-5 total PnL: {fmt_number(concentration['top_5_pnl'])}
Top-5 as % of total PnL: {pct(concentration['top_5_pct_total_pnl'])}
Top-5 as % of positive PnL: {pct(concentration['top_5_pct_positive_pnl'])}
95th percentile to median PnL ratio: {fmt_number(concentration['tail_ratio_95_to_median'])}
Fat-tail signal: {concentration['fat_tail_signal']}

F. Position Size Behavior
-------------------------
Max win streak: {fmt_int(position_behavior['max_win_streak'])}
Max loss streak: {fmt_int(position_behavior['max_loss_streak'])}
Average notional after win: {fmt_number(position_behavior['avg_notional_after_win'])}
Average notional after loss: {fmt_number(position_behavior['avg_notional_after_loss'])}
Average volume after win: {fmt_number(position_behavior['avg_volume_after_win'])}
Average volume after loss: {fmt_number(position_behavior['avg_volume_after_loss'])}
Average size change ratio after win: {fmt_number(position_behavior['avg_size_change_after_win'])}
Average size change ratio after loss: {fmt_number(position_behavior['avg_size_change_after_loss'])}
Average next-trade notional after loss / after win: {fmt_number(position_behavior['ratio_after_loss_vs_win'])}
Martingale-like signal: {position_behavior['martingale_signal']}

G. Holding Efficiency
---------------------
Average holding winners: {fmt_hours(holding_efficiency['avg_holding_winners'])}
Average holding losers: {fmt_hours(holding_efficiency['avg_holding_losers'])}
Median holding winners: {fmt_hours(holding_efficiency['median_holding_winners'])}
Median holding losers: {fmt_hours(holding_efficiency['median_holding_losers'])}
Most profitable holding bucket: {holding_efficiency['profitable_bucket']}
Highest average PnL holding bucket: {holding_efficiency['most_efficient_bucket']}

I. Edge Checks
--------------
{short_table_to_text(edge_checks)}

J. Monte Carlo Replication
--------------------------
Simulation runs: {fmt_int(simulation_summary['run_id'].nunique() if not simulation_summary.empty else 0)}
Median simulated total PnL: {fmt_number(simulation_median_pnl)}
Share of profitable runs: {pct(simulation_positive_share)}
Median simulated max drawdown: {fmt_number(simulation_median_dd)}

Monte Carlo summary:
{short_table_to_text(simulation_describe, digits=2)}

K. EMA50 Trend Filter
---------------------
{market_filter_note or 'Market filter was not available.'}

{short_table_to_text(market_filter_stats if market_filter_stats is not None else pd.DataFrame())}

{entry_quality_section or ''}

{execution_mode_section or ''}

{market_context_section or ''}

{eth_context_section or ''}

{eth_bollinger_section or ''}

{eth_setup_clustering_section or ''}

{eth_execution_aware_section or ''}

{eth_execution_reanchored_section or ''}

{eth_cluster1_short_overlap_section or ''}

{eth_cluster1_short_refinement_section or ''}

{eth_cluster1_short_timing_error_section or ''}

{eth_cluster1_short_final_timing_section or ''}

{eth_cluster_rule_section or ''}

{eth_replication_section or ''}

{eth_tightening_section or ''}

{eth_cluster1_micro_trigger_section or ''}

{eth_cluster1_structural_confirmation_section or ''}

{eth_cluster1_effort_result_section or ''}

{eth_practical_imitation_section or ''}

{eth_short_proxy_validation_section or ''}

{core_vs_noise_section or ''}

H. Edge Without Fantasy
-----------------------
Confirmed by data:
- {confirmed_lines[0]}
- {confirmed_lines[1]}
- {confirmed_lines[2]}
- {confirmed_lines[3]}
- {confirmed_lines[4]}
- Removing the top-5 trades leaves total PnL at {fmt_number(top5_removed_pnl)}, versus baseline {fmt_number(baseline_total_pnl)}.
- Monte Carlo median total PnL is {fmt_number(simulation_median_pnl)} with {pct(simulation_positive_share)} profitable runs.

Reasonable hypotheses:
- {hypothesis_lines[0]}
- {hypothesis_lines[1]}
- {hypothesis_lines[2]}
- {hypothesis_lines[3]}
- If a simple EMA50 regime filter materially improves the aligned subset, the edge may be partly in market regime selection rather than entry timing alone.

Cannot be concluded from this dataset alone:
- {unknown_lines[0]}
- {unknown_lines[1]}
- {unknown_lines[2]}
- We still cannot claim exact reproducibility because entry location inside the candle, leverage handling, and copy-trading slippage remain unknown.

Human Summary
-------------
This trader most likely behaves like a {style['style']}. The strongest earnings come from {best_symbol} and the strongest directional contribution comes from {best_side}. Weak spots are concentrated in {worst_symbol} and on the {worst_side} side. The main question for robustness is whether performance survives after removing a handful of exceptional trades; here the top-5 trades account for {pct(concentration['top_5_pct_total_pnl'])} of total PnL.

On the skill-versus-luck question, the answer is mixed. There is evidence of repeatable behavior in symbol selection, side asymmetry, and holding pattern, so this is not pure randomness. But the PnL distribution is fat-tailed, and the Monte Carlo section shows how sensitive total outcome can be to the sampling of a few large trades. That means the profile is behaviorally consistent, but the realized payoff still depends heavily on outsized wins and on avoiding a few oversized losses.

The most reproducible components appear to be behavioral rather than signal-based: entry frequency, preferred symbols, side bias, and holding-time profile. The least reproducible components are precise timing, large winner capture, and risk compression on the rare but very costly trades. If top-trade concentration stays high or long-duration losers dominate drawdown, copying this profile without tighter risk controls would be difficult to justify as a low-risk replication.
""".strip()
    return report + "\n"


def save_plot(fig: plt.Figure, output_path: Path, logger: logging.Logger) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot: %s", output_path.name)


def create_plots(df: pd.DataFrame, symbol_stats: pd.DataFrame, side_stats: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("cumulative_pnl_by_trade_number.png", plot_cumulative_pnl_by_trade_number, (df,)),
        ("cumulative_pnl_over_time.png", plot_cumulative_pnl_over_time, (df,)),
        ("histogram_pnl.png", plot_histogram_pnl, (df,)),
        ("histogram_holding_time.png", plot_histogram_holding_time, (df,)),
        ("bar_chart_by_symbol.png", plot_bar_chart_by_symbol, (symbol_stats,)),
        ("bar_chart_long_vs_short.png", plot_bar_chart_long_vs_short, (side_stats,)),
        ("boxplot_pnl_by_symbol.png", plot_boxplot_pnl_by_symbol, (df,)),
        ("boxplot_holding_time_winners_vs_losers.png", plot_boxplot_holding_winners_vs_losers, (df,)),
        ("trade_count_by_hour_utc.png", plot_trade_count_by_hour, (df,)),
        ("trade_count_by_weekday_utc.png", plot_trade_count_by_weekday, (df,)),
    ]

    for filename, plotter, args in plot_specs:
        fig = plotter(*args)
        save_plot(fig, plots_dir / filename, logger)


def create_extended_plots(
    df: pd.DataFrame,
    simulation_summary: pd.DataFrame,
    simulation_paths: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("equity_curve_real.png", plot_equity_curve_real, (df,)),
        ("equity_curve_simulations.png", plot_equity_curve_simulations, (simulation_paths,)),
        ("distribution_pnl.png", plot_distribution_pnl, (df,)),
        ("drawdown_distribution.png", plot_drawdown_distribution, (simulation_summary,)),
    ]

    for filename, plotter, args in plot_specs:
        fig = plotter(*args)
        save_plot(fig, plots_dir / filename, logger)


def plot_cumulative_pnl_by_trade_number(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["trade_number"], df["cumulative_pnl"], linewidth=2)
    ax.set_title("Cumulative PnL by Trade Number")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.3)
    return fig


def plot_cumulative_pnl_over_time(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["closed_dt_utc"], df["cumulative_pnl"], linewidth=2)
    ax.set_title("Cumulative PnL Over Time")
    ax.set_xlabel("Closed Time (UTC)")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.3)
    return fig


def plot_histogram_pnl(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["closingPnl"].dropna(), bins=20, edgecolor="black")
    ax.set_title("PnL Distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Trade Count")
    ax.grid(alpha=0.2)
    return fig


def plot_histogram_holding_time(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["holding_hours"].dropna(), bins=20, edgecolor="black")
    ax.set_title("Holding Time Distribution")
    ax.set_xlabel("Holding Time (hours)")
    ax.set_ylabel("Trade Count")
    ax.grid(alpha=0.2)
    return fig


def plot_bar_chart_by_symbol(symbol_stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(symbol_stats["symbol"].astype(str), symbol_stats["total_pnl"])
    ax.set_title("Total PnL by Symbol")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Total PnL")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_bar_chart_long_vs_short(side_stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(side_stats["side"].astype(str), side_stats["total_pnl"])
    ax.set_title("Total PnL by Side")
    ax.set_xlabel("Side")
    ax.set_ylabel("Total PnL")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_boxplot_pnl_by_symbol(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped = [group["closingPnl"].dropna().values for _, group in df.groupby("symbol")]
    labels = [str(name) for name, _ in df.groupby("symbol")]
    ax.boxplot(grouped, tick_labels=labels)
    ax.set_title("PnL by Symbol")
    ax.set_xlabel("Symbol")
    ax.set_ylabel("PnL")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_boxplot_holding_winners_vs_losers(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    winners = df.loc[df["closingPnl"] > 0, "holding_hours"].dropna().values
    losers = df.loc[df["closingPnl"] < 0, "holding_hours"].dropna().values
    ax.boxplot([winners, losers], tick_labels=["Winners", "Losers"])
    ax.set_title("Holding Time: Winners vs Losers")
    ax.set_ylabel("Holding Time (hours)")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_trade_count_by_hour(df: pd.DataFrame) -> plt.Figure:
    hour_counts = df["open_hour_utc"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hour_counts.index.astype(int), hour_counts.values)
    ax.set_title("Trade Count by Open Hour (UTC)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Trade Count")
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_trade_count_by_weekday(df: pd.DataFrame) -> plt.Figure:
    weekday_counts = df["open_weekday_utc"].value_counts().reindex(WEEKDAY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(weekday_counts))
    ax.bar(positions, weekday_counts.values)
    ax.set_title("Trade Count by Weekday (UTC)")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Trade Count")
    ax.set_xticks(positions)
    ax.set_xticklabels(weekday_counts.index.astype(str))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    return fig


def plot_equity_curve_real(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = df.sort_values("closed_dt_utc")
    ax.plot(ordered["closed_dt_utc"], ordered["closingPnl"].fillna(0).cumsum(), linewidth=2)
    ax.set_title("Equity Curve (Real Trades)")
    ax.set_xlabel("Closed Time (UTC)")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.3)
    return fig


def plot_equity_curve_simulations(simulation_paths: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    if simulation_paths.empty:
        ax.text(0.5, 0.5, "No simulation paths", ha="center", va="center")
        ax.set_axis_off()
        return fig
    for run_id, run_frame in simulation_paths.groupby("run_id"):
        ax.plot(run_frame["trade_number"], run_frame["cumulative_pnl"], alpha=0.12, linewidth=1)
    ax.set_title("Equity Curves (Monte Carlo Simulations)")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    return fig


def plot_distribution_pnl(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["closingPnl"].dropna(), bins=25, edgecolor="black")
    ax.set_title("PnL Distribution")
    ax.set_xlabel("Trade PnL")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.2)
    return fig


def plot_drawdown_distribution(simulation_summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(simulation_summary["approx_max_drawdown"].dropna(), bins=25, edgecolor="black")
    ax.set_title("Drawdown Distribution (Monte Carlo)")
    ax.set_xlabel("Approximate Max Drawdown")
    ax.set_ylabel("Simulation Count")
    ax.grid(alpha=0.2)
    return fig


def console_summary(
    basic: dict,
    symbol_stats: pd.DataFrame,
    side_stats: pd.DataFrame,
    concentration: dict,
) -> str:
    best_symbol, worst_symbol = best_and_worst(symbol_stats, "symbol")
    best_side, worst_side = best_and_worst(side_stats, "side")
    return "\n".join(
        [
            f"total trades: {fmt_int(basic['total_trades'])}",
            f"win rate: {pct(basic['win_rate'])}",
            f"total pnl: {fmt_number(basic['total_pnl'])}",
            f"best symbol: {best_symbol}",
            f"worst symbol: {worst_symbol}",
            f"best side: {best_side}",
            f"worst side: {worst_side}",
            f"average holding time: {fmt_hours(basic['avg_holding_hours'])}",
            f"concentration of pnl in top trades: {pct(concentration['top_5_pct_total_pnl'])}",
        ]
    )
