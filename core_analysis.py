from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from simulation import max_drawdown_from_equity
from utils import fmt_hours, fmt_int, fmt_number, pct, safe_divide


def run_core_vs_noise_analysis(df: pd.DataFrame) -> dict:
    annotated = df.copy()
    annotated = flag_extreme_losers(annotated)
    annotated = flag_possible_faith_trades(annotated)

    scenario_frames = {
        "baseline": annotated.sort_values("closed_dt_utc").reset_index(drop=True),
        "no_extreme_losers": annotated.loc[~annotated["extreme_loser_flag"]].sort_values("closed_dt_utc").reset_index(drop=True),
        "no_faith_trades": annotated.loc[~annotated["faith_trade_hypothesis_flag"]].sort_values("closed_dt_utc").reset_index(drop=True),
    }

    summary = pd.DataFrame(
        [scenario_stats(name, frame) for name, frame in scenario_frames.items()]
    )

    flagged_extreme = (
        annotated.loc[annotated["extreme_loser_flag"]]
        .sort_values("closingPnl")
        .reset_index(drop=True)
    )
    flagged_faith = (
        annotated.loc[annotated["faith_trade_hypothesis_flag"]]
        .sort_values("closingPnl")
        .reset_index(drop=True)
    )

    section = build_core_vs_noise_section(summary, flagged_extreme, flagged_faith)
    return {
        "annotated_trades": annotated,
        "summary": summary,
        "flagged_extreme_losers": flagged_extreme,
        "flagged_faith_trades": flagged_faith,
        "scenario_frames": scenario_frames,
        "section_text": section,
    }


def flag_extreme_losers(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    annotated["extreme_loser_flag"] = False
    annotated["extreme_loser_reasons"] = ""
    annotated["loss_abs_zscore"] = np.nan

    losses = annotated.loc[annotated["closingPnl"] < 0].copy()
    if losses.empty:
        return annotated

    overall_worst_5pct = annotated["closingPnl"].quantile(0.05)
    loss_abs = losses["closingPnl"].abs()
    median_abs_loss = loss_abs.median()
    q1_abs_loss = loss_abs.quantile(0.25)
    q3_abs_loss = loss_abs.quantile(0.75)
    iqr_abs_loss = q3_abs_loss - q1_abs_loss
    iqr_threshold = q3_abs_loss + 1.5 * iqr_abs_loss if pd.notna(iqr_abs_loss) else np.nan
    mean_abs_loss = loss_abs.mean()
    std_abs_loss = loss_abs.std(ddof=0)
    zscores = (loss_abs - mean_abs_loss) / std_abs_loss if pd.notna(std_abs_loss) and std_abs_loss > 0 else pd.Series(0.0, index=losses.index)

    median_loser_holding = losses["holding_hours"].median()
    q90_loser_holding = losses["holding_hours"].quantile(0.90)
    long_holding_threshold = np.nanmax([median_loser_holding * 2 if pd.notna(median_loser_holding) else np.nan, q90_loser_holding])
    oversized_loss_threshold = np.nanmax([median_abs_loss * 2 if pd.notna(median_abs_loss) else np.nan, q3_abs_loss * 1.5 if pd.notna(q3_abs_loss) else np.nan])

    for idx, row in losses.iterrows():
        reasons: list[str] = []
        abs_loss = abs(row["closingPnl"])
        z_value = float(zscores.loc[idx]) if idx in zscores.index else np.nan

        if row["closingPnl"] <= overall_worst_5pct:
            reasons.append("worst_5pct_pnl_tail")
        if pd.notna(row["holding_hours"]) and pd.notna(long_holding_threshold) and row["holding_hours"] >= long_holding_threshold:
            reasons.append("extreme_loser_holding_time")
        if pd.notna(abs_loss) and pd.notna(oversized_loss_threshold) and abs_loss >= oversized_loss_threshold:
            reasons.append("loss_much_larger_than_median_loser")
        if (
            (pd.notna(iqr_threshold) and abs_loss >= iqr_threshold)
            or (pd.notna(z_value) and z_value >= 2.0)
        ):
            reasons.append("statistical_loss_outlier")

        if reasons:
            annotated.at[idx, "extreme_loser_flag"] = True
            annotated.at[idx, "extreme_loser_reasons"] = "; ".join(reasons)
        annotated.at[idx, "loss_abs_zscore"] = z_value

    return annotated


def flag_possible_faith_trades(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    annotated["faith_trade_hypothesis_flag"] = False
    annotated["faith_trade_hypothesis_reasons"] = ""

    losses = annotated.loc[annotated["closingPnl"] < 0].copy()
    if losses.empty:
        return annotated

    side_pnl = annotated.groupby("side", dropna=False)["closingPnl"].sum()
    dominant_side = str(side_pnl.idxmax()) if not side_pnl.empty else "n/a"
    overall_median_holding = annotated["holding_hours"].median()
    overall_q95_holding = annotated["holding_hours"].quantile(0.95)

    loss_abs = losses["closingPnl"].abs()
    median_abs_loss = loss_abs.median()
    q75_abs_loss = loss_abs.quantile(0.75)
    q90_loser_holding = losses["holding_hours"].quantile(0.90)
    median_loser_holding = losses["holding_hours"].median()
    rare_symbol_threshold = safe_divide(1, max(len(annotated), 1)) * 3
    symbol_trade_share = annotated["symbol"].value_counts(normalize=True, dropna=False)

    large_loss_threshold = np.nanmax([median_abs_loss * 1.5 if pd.notna(median_abs_loss) else np.nan, q75_abs_loss])
    extreme_holding_threshold = np.nanmax(
        [
            median_loser_holding * 1.75 if pd.notna(median_loser_holding) else np.nan,
            q90_loser_holding,
            overall_median_holding * 2 if pd.notna(overall_median_holding) else np.nan,
        ]
    )

    for idx, row in losses.iterrows():
        reasons: list[str] = []
        abs_loss = abs(row["closingPnl"])
        long_holding_loss = pd.notna(row["holding_hours"]) and pd.notna(extreme_holding_threshold) and row["holding_hours"] >= extreme_holding_threshold
        large_loss = pd.notna(abs_loss) and pd.notna(large_loss_threshold) and abs_loss >= large_loss_threshold
        profile_outlier = False
        special_btc_long = False

        if large_loss:
            reasons.append("large_final_loss")
        if long_holding_loss:
            reasons.append("extreme_holding_on_losing_trade")
        if row.get("extreme_loser_flag", False):
            reasons.append("already_flagged_as_extreme_loser")

        symbol_share = float(symbol_trade_share.get(row["symbol"], 0.0))
        if (
            str(row["side"]).lower() != str(dominant_side).lower()
            and pd.notna(row["holding_hours"])
            and pd.notna(median_loser_holding)
            and row["holding_hours"] >= median_loser_holding * 1.5
        ):
            profile_outlier = True
            reasons.append("outlier_inconsistent_with_dominant_side_profile")
        if symbol_share <= rare_symbol_threshold and large_loss:
            profile_outlier = True
            reasons.append("rare_symbol_loss_outlier")
        if (
            str(row["symbol"]) == "BTCUSDT"
            and str(row["side"]).lower() == "long"
            and large_loss
            and pd.notna(row["holding_hours"])
            and pd.notna(median_loser_holding)
            and row["holding_hours"] >= median_loser_holding * 1.5
        ):
            profile_outlier = True
            special_btc_long = True
            reasons.append("btc_long_outlier_vs_overall_profile")

        hypothesis_flag = bool(
            special_btc_long
            or (large_loss and long_holding_loss)
            or (large_loss and profile_outlier)
            or (long_holding_loss and profile_outlier)
            or (row.get("extreme_loser_flag", False) and long_holding_loss and (large_loss or profile_outlier))
        )

        if hypothesis_flag:
            annotated.at[idx, "faith_trade_hypothesis_flag"] = True
            annotated.at[idx, "faith_trade_hypothesis_reasons"] = "; ".join(dict.fromkeys(reasons))

    return annotated


def scenario_stats(name: str, frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("closed_dt_utc").reset_index(drop=True)
    pnl = ordered["closingPnl"].fillna(0)
    equity = pnl.cumsum()
    long_frame = ordered.loc[ordered["side"].astype(str).str.lower() == "long"]
    short_frame = ordered.loc[ordered["side"].astype(str).str.lower() == "short"]

    total_pnl = pnl.sum()
    top_5_pnl = ordered.nlargest(min(5, len(ordered)), "closingPnl")["closingPnl"].sum() if len(ordered) else 0.0
    symbol_counts = ordered["symbol"].value_counts(normalize=True, dropna=False)
    side_counts = ordered["side"].value_counts(normalize=True, dropna=False)

    return {
        "scenario": name,
        "trade_count": len(ordered),
        "total_pnl": total_pnl,
        "avg_pnl": pnl.mean() if len(ordered) else np.nan,
        "median_pnl": pnl.median() if len(ordered) else np.nan,
        "win_rate": safe_divide((pnl > 0).sum(), pnl.notna().sum()),
        "long_pnl": long_frame["closingPnl"].sum(),
        "short_pnl": short_frame["closingPnl"].sum(),
        "long_win_rate": safe_divide((long_frame["closingPnl"] > 0).sum(), long_frame["closingPnl"].notna().sum()),
        "short_win_rate": safe_divide((short_frame["closingPnl"] > 0).sum(), short_frame["closingPnl"].notna().sum()),
        "avg_holding_time": ordered["holding_hours"].mean(),
        "avg_holding_winners": ordered.loc[ordered["closingPnl"] > 0, "holding_hours"].mean(),
        "avg_holding_losers": ordered.loc[ordered["closingPnl"] < 0, "holding_hours"].mean(),
        "top_5_pnl_concentration": safe_divide(top_5_pnl, total_pnl),
        "approx_max_drawdown": max_drawdown_from_equity(equity),
        "symbol_concentration": symbol_counts.max() if not symbol_counts.empty else np.nan,
        "side_concentration": side_counts.max() if not side_counts.empty else np.nan,
    }


def build_core_vs_noise_section(
    summary_df: pd.DataFrame,
    flagged_extreme: pd.DataFrame,
    flagged_faith: pd.DataFrame,
) -> str:
    baseline = _scenario_row(summary_df, "baseline")
    no_extreme = _scenario_row(summary_df, "no_extreme_losers")
    no_faith = _scenario_row(summary_df, "no_faith_trades")

    cleaned_candidate_name = "no_faith_trades"
    cleaned_candidate = no_faith
    if (
        no_extreme is not None
        and no_faith is not None
        and pd.notna(no_extreme.get("approx_max_drawdown"))
        and pd.notna(no_faith.get("approx_max_drawdown"))
        and no_extreme["approx_max_drawdown"] < no_faith["approx_max_drawdown"]
    ):
        cleaned_candidate_name = "no_extreme_losers"
        cleaned_candidate = no_extreme

    if cleaned_candidate is None:
        cleaned_candidate_name = "baseline"
        cleaned_candidate = baseline

    remains_profitable_extreme = _positive_after_cleaning(no_extreme)
    remains_profitable_faith = _positive_after_cleaning(no_faith)
    smoother_extreme = _smoother_profile(baseline, no_extreme)
    smoother_faith = _smoother_profile(baseline, no_faith)
    short_edge_extreme = _strong_short_edge(no_extreme)
    short_edge_faith = _strong_short_edge(no_faith)
    long_normal_extreme = _long_side_improved(baseline, no_extreme)
    long_normal_faith = _long_side_improved(baseline, no_faith)
    fat_tail_reduced_extreme = _fat_tail_reduced(baseline, no_extreme)
    fat_tail_reduced_faith = _fat_tail_reduced(baseline, no_faith)
    cleaned_viable = (
        cleaned_candidate is not None
        and cleaned_candidate.get("total_pnl", np.nan) > 0
        and cleaned_candidate.get("approx_max_drawdown", np.nan) < baseline.get("approx_max_drawdown", np.nan)
        and cleaned_candidate.get("top_5_pnl_concentration", np.nan) <= baseline.get("top_5_pnl_concentration", np.nan)
    )

    extreme_sample = _sample_flagged_rows(flagged_extreme, "extreme_loser_reasons")
    faith_sample = _sample_flagged_rows(flagged_faith, "faith_trade_hypothesis_reasons")

    return f"""
CORE VS NOISE ANALYSIS
----------------------
Flagged extreme losers: {fmt_int(len(flagged_extreme))}
Flagged possible faith-trades: {fmt_int(len(flagged_faith))}

Scenario comparison:
{_table_to_text(summary_df)}

Flagged extreme losers sample:
{_table_to_text(extreme_sample)}

Flagged possible faith-trades sample:
{_table_to_text(faith_sample)}

Core-vs-noise interpretation:
- Profit after removing extreme losers: {'yes' if remains_profitable_extreme else 'no'} ({fmt_number(no_extreme.get('total_pnl') if no_extreme is not None else np.nan)}).
- Profit after removing possible faith-trades: {'yes' if remains_profitable_faith else 'no'} ({fmt_number(no_faith.get('total_pnl') if no_faith is not None else np.nan)}).
- Profile looks smoother after removing extreme losers: {'yes' if smoother_extreme else 'no'}.
- Profile looks smoother after removing possible faith-trades: {'yes' if smoother_faith else 'no'}.
- Strong short edge remains after removing extreme losers: {'yes' if short_edge_extreme else 'no'}.
- Strong short edge remains after removing possible faith-trades: {'yes' if short_edge_faith else 'no'}.
- Long side looks more normal after removing extreme losers: {'yes' if long_normal_extreme else 'no'}.
- Long side looks more normal after removing possible faith-trades: {'yes' if long_normal_faith else 'no'}.
- Fat-tail dependence decreases after removing extreme losers: {'yes' if fat_tail_reduced_extreme else 'no'}.
- Fat-tail dependence decreases after removing possible faith-trades: {'yes' if fat_tail_reduced_faith else 'no'}.
- Most plausible cleaned profile for further replication work: {cleaned_candidate_name}.

Human read:
- What looks like the core strategy: repeated short-side profitability, faster/cleaner holding behavior, and the part of the profile that remains profitable after removing likely outlier losers.
- What looks like noise or possible emotional contamination: likely emotional outliers with unusually long holding times, especially trades inconsistent with the dominant profitable side profile.
- Should the cleaned profile be analyzed further as a replication base: {'yes' if cleaned_viable else 'maybe, but only with caution'}.
""".strip()


def create_core_vs_noise_plots(scenario_frames: dict[str, pd.DataFrame], output_dir: Path, logger: logging.Logger) -> None:
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("core_equity_comparison.png", plot_core_equity_comparison, (scenario_frames,)),
        ("core_drawdown_comparison.png", plot_core_drawdown_comparison, (scenario_frames,)),
        ("core_pnl_distribution.png", plot_core_pnl_distribution, (scenario_frames,)),
    ]

    for filename, plotter, args in plot_specs:
        fig = plotter(*args)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", filename)


def plot_core_equity_comparison(scenario_frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, frame in scenario_frames.items():
        if frame.empty:
            continue
        ordered = frame.sort_values("closed_dt_utc")
        ax.plot(ordered["closed_dt_utc"], ordered["closingPnl"].fillna(0).cumsum(), linewidth=2, label=label)
    ax.set_title("Baseline vs Cleaned Equity Curves")
    ax.set_xlabel("Closed Time (UTC)")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_core_drawdown_comparison(scenario_frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, frame in scenario_frames.items():
        if frame.empty:
            continue
        ordered = frame.sort_values("closed_dt_utc").copy()
        equity = ordered["closingPnl"].fillna(0).cumsum()
        running_peak = equity.cummax()
        drawdown = running_peak - equity
        ax.plot(ordered["closed_dt_utc"], drawdown, linewidth=2, label=label)
    ax.set_title("Baseline vs Cleaned Drawdowns")
    ax.set_xlabel("Closed Time (UTC)")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_core_pnl_distribution(scenario_frames: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 25
    for label, frame in scenario_frames.items():
        if frame.empty:
            continue
        ax.hist(frame["closingPnl"].dropna(), bins=bins, alpha=0.4, label=label)
    ax.set_title("Baseline vs Cleaned PnL Distributions")
    ax.set_xlabel("Trade PnL")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def _scenario_row(summary_df: pd.DataFrame, name: str) -> dict | None:
    row = summary_df.loc[summary_df["scenario"] == name]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _positive_after_cleaning(row: dict | None) -> bool:
    return bool(row is not None and pd.notna(row.get("total_pnl")) and row["total_pnl"] > 0)


def _smoother_profile(baseline: dict | None, cleaned: dict | None) -> bool:
    if baseline is None or cleaned is None:
        return False
    drawdown_improved = cleaned.get("approx_max_drawdown", np.nan) < baseline.get("approx_max_drawdown", np.nan)
    concentration_improved = cleaned.get("top_5_pnl_concentration", np.nan) <= baseline.get("top_5_pnl_concentration", np.nan)
    return bool(drawdown_improved and concentration_improved)


def _strong_short_edge(row: dict | None) -> bool:
    if row is None:
        return False
    return bool(
        pd.notna(row.get("short_pnl"))
        and row["short_pnl"] > 0
        and pd.notna(row.get("short_win_rate"))
        and pd.notna(row.get("long_win_rate"))
        and row["short_win_rate"] >= row["long_win_rate"]
    )


def _long_side_improved(baseline: dict | None, cleaned: dict | None) -> bool:
    if baseline is None or cleaned is None:
        return False
    baseline_long = baseline.get("long_pnl", np.nan)
    cleaned_long = cleaned.get("long_pnl", np.nan)
    if pd.isna(baseline_long) or pd.isna(cleaned_long):
        return False
    return bool(cleaned_long >= baseline_long)


def _fat_tail_reduced(baseline: dict | None, cleaned: dict | None) -> bool:
    if baseline is None or cleaned is None:
        return False
    return bool(
        pd.notna(cleaned.get("top_5_pnl_concentration"))
        and pd.notna(baseline.get("top_5_pnl_concentration"))
        and cleaned["top_5_pnl_concentration"] <= baseline["top_5_pnl_concentration"]
    )


def _sample_flagged_rows(frame: pd.DataFrame, reason_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [
        column
        for column in ["trade_number", "symbol", "side", "closingPnl", "holding_hours", reason_column]
        if column in frame.columns
    ]
    return frame[columns].head(10).copy()


def _table_to_text(df: pd.DataFrame, digits: int = 2) -> str:
    if df.empty:
        return "n/a"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(digits)
    return display.to_string(index=False)
