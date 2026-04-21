from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOTS_DIRNAME
from eth_practical_imitation import build_minute_arrays, build_proxy_scan_frame
from eth_replication_hypothesis_test import ETH_SYMBOL, build_stage_windows, materialize_stage_bounds
from market import fetch_futures_klines
from utils import fmt_int, fmt_number, pct, safe_divide


REPRESENTATIVE_COST_ID = "normal_fee__moderate_slippage"
BASE_PARAMS = {
    "distance_to_4h_high_pct_max": 0.60,
    "pre_entry_return_60m_pct_min": 0.45,
    "upper_wick_ratio_5m_min": 0.25,
    "last_5m_return_pct_max": 0.00,
    "exit_horizon_hours": 12,
}

DISTANCE_GRID = [0.45, 0.60, 0.75]
PRE_ENTRY_GRID = [0.25, 0.45, 0.65]
WICK_GRID = [0.15, 0.25, 0.35]
HORIZON_GRID_HOURS = [8, 12, 16, 24]


@dataclass(frozen=True)
class FeeScenario:
    fee_id: str
    label: str
    round_trip_fee_bps: float


@dataclass(frozen=True)
class SlippageScenario:
    slippage_id: str
    label: str
    round_trip_slippage_bps: float


@dataclass(frozen=True)
class RiskModel:
    risk_model_id: str
    label: str
    sizing_mode: str
    max_concurrent: int
    fixed_notional: float | None = None
    risk_fraction: float | None = None
    notional_cap_fraction: float | None = None
    absolute_cap: float | None = None
    total_exposure_cap_fraction: float | None = None


FEE_SCENARIOS = [
    FeeScenario("low_fee", "Low fee", 4.0),
    FeeScenario("normal_fee", "Normal fee", 8.0),
    FeeScenario("harsh_fee", "Harsh fee", 12.0),
]

SLIPPAGE_SCENARIOS = [
    SlippageScenario("zero_slippage", "0 bps slippage", 0.0),
    SlippageScenario("mild_slippage", "Mild slippage", 2.0),
    SlippageScenario("moderate_slippage", "Moderate slippage", 5.0),
    SlippageScenario("harsher_slippage", "Harsher slippage", 8.0),
]

RISK_MODELS = [
    RiskModel(
        "fixed_notional_max1",
        "Fixed notional / max 1 open trade",
        "fixed_notional",
        max_concurrent=1,
        fixed_notional=10_000.0,
    ),
    RiskModel(
        "fixed_risk_fraction_max1",
        "Fixed risk fraction / max 1 open trade",
        "risk_fraction",
        max_concurrent=1,
        risk_fraction=0.005,
        notional_cap_fraction=0.25,
        absolute_cap=25_000.0,
    ),
    RiskModel(
        "conservative_capped_exposure_max2",
        "Conservative capped exposure / max 2 open trades",
        "capped_exposure",
        max_concurrent=2,
        fixed_notional=5_000.0,
        total_exposure_cap_fraction=0.16,
        absolute_cap=5_000.0,
    ),
]


def run_eth_short_proxy_validation(
    trades_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    empty = pd.DataFrame()
    empty_section = (
        "ETH SHORT PROXY VALIDATION\n"
        "--------------------------\n"
        "No sufficient ETH data was available for proxy validation.\n"
    )

    eth_trades = trades_df.loc[
        trades_df["symbol"] == ETH_SYMBOL,
        ["trade_number", "symbol", "side", "opened_dt_utc", "avgCost"],
    ].copy()
    if eth_trades.empty:
        return {
            "validation_summary": empty,
            "friction_comparison": empty,
            "parameter_robustness": empty,
            "stage_consistency": empty,
            "monthly_breakdown": empty,
            "net_vs_gross_curves": empty,
            "section_text": empty_section,
        }

    stages = build_stage_windows(eth_trades)
    fetch_start = min(stage.requested_start for stage in stages) - pd.Timedelta(hours=48)
    fetch_end = min(max(stage.requested_end for stage in stages) + pd.Timedelta(hours=25), pd.Timestamp.now(tz="UTC"))

    logger.info(
        "Running ETH short proxy validation from %s to %s",
        fetch_start.isoformat(),
        fetch_end.isoformat(),
    )

    minute_frame = fetch_futures_klines(
        ETH_SYMBOL,
        int(fetch_start.timestamp() * 1000),
        int(fetch_end.timestamp() * 1000),
        interval="1m",
        logger=logger,
    )
    frame_5m = fetch_futures_klines(
        ETH_SYMBOL,
        int(fetch_start.timestamp() * 1000),
        int(fetch_end.timestamp() * 1000),
        interval="5m",
        logger=logger,
    )
    if minute_frame.empty or frame_5m.empty:
        return {
            "validation_summary": empty,
            "friction_comparison": empty,
            "parameter_robustness": empty,
            "stage_consistency": empty,
            "monthly_breakdown": empty,
            "net_vs_gross_curves": empty,
            "section_text": (
                "ETH SHORT PROXY VALIDATION\n"
                "--------------------------\n"
                "Validation could not run because ETH candle data was unavailable.\n"
            ),
        }

    scan_frame = build_proxy_scan_frame(frame_5m)
    if scan_frame.empty:
        return {
            "validation_summary": empty,
            "friction_comparison": empty,
            "parameter_robustness": empty,
            "stage_consistency": empty,
            "monthly_breakdown": empty,
            "net_vs_gross_curves": empty,
            "section_text": (
                "ETH SHORT PROXY VALIDATION\n"
                "--------------------------\n"
                "Validation could not run because the ETH short scan frame was empty.\n"
            ),
        }

    max_signal_entry_time = scan_frame["entry_time"].max() - pd.Timedelta(hours=24)
    min_signal_entry_time = scan_frame["entry_time"].min()
    materialize_stage_bounds(stages, min_signal_entry_time, max_signal_entry_time)
    minute_arrays = build_minute_arrays(minute_frame)

    base_signals = build_short_proxy_signals(scan_frame, stages, BASE_PARAMS)
    friction_comparison, portfolio_curves = build_friction_comparison(base_signals, stages, minute_arrays)
    representative_rows = friction_comparison.loc[friction_comparison["cost_scenario_id"] == REPRESENTATIVE_COST_ID].copy()
    recommended_risk_model = pick_recommended_risk_model(representative_rows)

    parameter_robustness = build_parameter_robustness(scan_frame, stages, minute_arrays)
    stage_consistency, monthly_breakdown = build_stage_consistency(
        base_signals,
        stages,
        minute_arrays,
        representative_rows,
        recommended_risk_model,
    )
    validation_summary = build_validation_summary(stage_consistency, representative_rows, recommended_risk_model)
    section_text = build_validation_section(validation_summary, representative_rows, parameter_robustness)

    return {
        "validation_summary": validation_summary,
        "friction_comparison": friction_comparison,
        "parameter_robustness": parameter_robustness,
        "stage_consistency": stage_consistency,
        "monthly_breakdown": monthly_breakdown,
        "net_vs_gross_curves": portfolio_curves,
        "section_text": section_text,
    }


def build_short_proxy_signals(
    scan_frame: pd.DataFrame,
    stages: list,
    params: dict,
) -> pd.DataFrame:
    mask = (
        (scan_frame["distance_to_last_4h_high_pct"] <= params["distance_to_4h_high_pct_max"])
        & (scan_frame["return_pre_60m_pct"] >= params["pre_entry_return_60m_pct_min"])
        & (scan_frame["upper_wick_ratio_5m"] >= params["upper_wick_ratio_5m_min"])
        & (scan_frame["ret_3c_5m_pct"] > 0)
        & (scan_frame["ret_last_5m_pct"] <= params["last_5m_return_pct_max"])
    ).fillna(False)
    start_mask = mask & ~mask.shift(1, fill_value=False)
    starts = scan_frame.loc[start_mask].copy()
    if starts.empty:
        return pd.DataFrame()

    starts["setup_name"] = "eth_short_proxy"
    starts["side"] = "Short"
    starts["signal_id"] = starts["entry_time"].dt.strftime("eth_short_proxy_%Y%m%dT%H%M%S")
    starts["distance_threshold"] = params["distance_to_4h_high_pct_max"]
    starts["pre_entry_threshold"] = params["pre_entry_return_60m_pct_min"]
    starts["wick_threshold"] = params["upper_wick_ratio_5m_min"]
    starts["last_5m_ceiling"] = params["last_5m_return_pct_max"]
    starts["exit_horizon_hours"] = params["exit_horizon_hours"]
    for stage in stages:
        if stage.effective_start is None or stage.effective_end is None:
            starts[f"in_{stage.stage_id}"] = False
        else:
            starts[f"in_{stage.stage_id}"] = starts["entry_time"].between(stage.effective_start, stage.effective_end, inclusive="both")
    return starts.sort_values("entry_time").reset_index(drop=True)


def build_friction_comparison(
    base_signals: pd.DataFrame,
    stages: list,
    minute_arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_frames = []
    for stage in stages:
        stage_signals = base_signals.loc[base_signals[f"in_{stage.stage_id}"] == True].copy()
        for risk_model, fee_scenario, slippage_scenario in itertools.product(RISK_MODELS, FEE_SCENARIOS, SLIPPAGE_SCENARIOS):
            executed = simulate_portfolio(
                stage_signals,
                minute_arrays,
                horizon_hours=BASE_PARAMS["exit_horizon_hours"],
                risk_model=risk_model,
                fee_scenario=fee_scenario,
                slippage_scenario=slippage_scenario,
            )
            summary = summarize_validation_portfolio(
                executed,
                stage_signals,
                stage,
                risk_model,
                fee_scenario,
                slippage_scenario,
            )
            rows.append(summary)
            if not executed.empty:
                curve_frame = executed[["entry_time", "gross_pnl_proxy", "net_pnl_proxy"]].copy()
                curve_frame["stage_id"] = stage.stage_id
                curve_frame["risk_model_id"] = risk_model.risk_model_id
                curve_frame["cost_scenario_id"] = combined_cost_id(fee_scenario, slippage_scenario)
                curve_frame["cumulative_gross_pnl_proxy"] = curve_frame["gross_pnl_proxy"].cumsum()
                curve_frame["cumulative_net_pnl_proxy"] = curve_frame["net_pnl_proxy"].cumsum()
                curve_frames.append(curve_frame)
    friction_comparison = pd.DataFrame(rows)
    portfolio_curves = pd.concat(curve_frames, ignore_index=True, sort=False) if curve_frames else pd.DataFrame()
    return friction_comparison, portfolio_curves


def simulate_portfolio(
    stage_signals: pd.DataFrame,
    minute_arrays: dict[str, np.ndarray],
    horizon_hours: int,
    risk_model: RiskModel,
    fee_scenario: FeeScenario,
    slippage_scenario: SlippageScenario,
) -> pd.DataFrame:
    if stage_signals.empty:
        return pd.DataFrame()

    capital = 100_000.0
    active_positions: list[dict] = []
    executed_rows = []
    total_cost_pct = (fee_scenario.round_trip_fee_bps + slippage_scenario.round_trip_slippage_bps) / 100.0
    horizon_minutes = horizon_hours * 60

    for _, signal in stage_signals.sort_values("entry_time").iterrows():
        entry_time = pd.to_datetime(signal["entry_time"], utc=True)
        active_positions, capital = settle_closed_positions(active_positions, capital, entry_time)
        if len(active_positions) >= risk_model.max_concurrent:
            continue

        notional = determine_trade_notional(signal, capital, active_positions, risk_model)
        if pd.isna(notional) or notional <= 0:
            continue

        exit_result = resolve_fixed_exit(entry_time, minute_arrays, horizon_minutes)
        if exit_result is None:
            continue

        entry_price = float(signal["entry_price"])
        exit_price = float(exit_result["exit_price"])
        gross_return_pct = (entry_price - exit_price) / entry_price * 100.0
        net_return_pct = gross_return_pct - total_cost_pct

        high_window = minute_arrays["high"][exit_result["open_idx_start"] : exit_result["open_idx_end"] + 1]
        low_window = minute_arrays["low"][exit_result["open_idx_start"] : exit_result["open_idx_end"] + 1]
        mfe_pct = ((entry_price - low_window) / entry_price * 100.0).max() if low_window.size else np.nan
        mae_pct = ((high_window - entry_price) / entry_price * 100.0).max() if high_window.size else np.nan

        row = {
            "signal_id": signal["signal_id"],
            "entry_time": entry_time,
            "exit_time": exit_result["exit_time"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "holding_minutes": (exit_result["exit_time"] - entry_time).total_seconds() / 60.0,
            "notional": notional,
            "gross_return_pct": gross_return_pct,
            "net_return_pct": net_return_pct,
            "gross_pnl_proxy": notional * gross_return_pct / 100.0,
            "net_pnl_proxy": notional * net_return_pct / 100.0,
            "MFE_pct": float(mfe_pct) if pd.notna(mfe_pct) else np.nan,
            "MAE_pct": float(mae_pct) if pd.notna(mae_pct) else np.nan,
            "atr14_pct_entry": pd.to_numeric(signal.get("atr14_pct"), errors="coerce"),
            "open_hour_utc": signal.get("open_hour_utc", np.nan),
            "fee_bps_round_trip": fee_scenario.round_trip_fee_bps,
            "slippage_bps_round_trip": slippage_scenario.round_trip_slippage_bps,
            "cost_pct_total": total_cost_pct,
        }
        executed_rows.append(row)
        active_positions.append({"exit_time": exit_result["exit_time"], "net_pnl_proxy": row["net_pnl_proxy"], "notional": notional})

    active_positions, capital = settle_closed_positions(active_positions, capital, pd.Timestamp.max.tz_localize("UTC"))
    del capital
    return pd.DataFrame(executed_rows)


def settle_closed_positions(active_positions: list[dict], capital: float, now: pd.Timestamp) -> tuple[list[dict], float]:
    still_open = []
    for position in active_positions:
        if position["exit_time"] <= now:
            capital += position["net_pnl_proxy"]
        else:
            still_open.append(position)
    return still_open, capital


def determine_trade_notional(
    signal: pd.Series,
    capital: float,
    active_positions: list[dict],
    risk_model: RiskModel,
) -> float:
    if risk_model.sizing_mode == "fixed_notional":
        return float(risk_model.fixed_notional or 0.0)

    if risk_model.sizing_mode == "risk_fraction":
        atr_pct = pd.to_numeric(signal.get("atr14_pct"), errors="coerce")
        proxy_stop_pct = max(float(atr_pct) if pd.notna(atr_pct) and atr_pct > 0 else 0.0, 0.50)
        risk_budget = capital * float(risk_model.risk_fraction or 0.0)
        raw_notional = safe_divide(risk_budget, proxy_stop_pct / 100.0)
        capped_notional = min(
            raw_notional,
            capital * float(risk_model.notional_cap_fraction or 1.0),
            float(risk_model.absolute_cap or raw_notional),
        )
        return float(max(capped_notional, 0.0))

    proposed = min(
        float(risk_model.fixed_notional or 0.0),
        capital * float(risk_model.total_exposure_cap_fraction or 1.0),
        float(risk_model.absolute_cap or risk_model.fixed_notional or 0.0),
    )
    active_exposure = sum(float(position.get("notional", 0.0)) for position in active_positions)
    max_total_exposure = capital * float(risk_model.total_exposure_cap_fraction or 1.0)
    available = max(max_total_exposure - active_exposure, 0.0)
    return float(min(proposed, available))


def resolve_fixed_exit(
    entry_time: pd.Timestamp,
    minute_arrays: dict[str, np.ndarray],
    horizon_minutes: int,
) -> dict | None:
    entry_ns = int(entry_time.value)
    target_ns = entry_ns + int(pd.Timedelta(minutes=horizon_minutes).value)
    close_times_ns = minute_arrays["close_time_ns"]
    open_times_ns = minute_arrays["open_time_ns"]

    close_idx = int(np.searchsorted(close_times_ns, target_ns, side="left"))
    if close_idx >= len(close_times_ns):
        return None

    open_idx_start = int(np.searchsorted(open_times_ns, entry_ns, side="right"))
    open_idx_end = int(np.searchsorted(open_times_ns, close_times_ns[close_idx], side="right") - 1)
    return {
        "exit_time": pd.Timestamp(close_times_ns[close_idx], tz="UTC"),
        "exit_price": float(minute_arrays["close"][close_idx]),
        "open_idx_start": max(open_idx_start, 0),
        "open_idx_end": max(open_idx_end, open_idx_start),
    }


def summarize_validation_portfolio(
    executed: pd.DataFrame,
    stage_signals: pd.DataFrame,
    stage,
    risk_model: RiskModel,
    fee_scenario: FeeScenario,
    slippage_scenario: SlippageScenario,
) -> dict:
    gross_profit_factor = profit_factor(executed["gross_pnl_proxy"]) if not executed.empty else np.nan
    net_profit_factor = profit_factor(executed["net_pnl_proxy"]) if not executed.empty else np.nan
    gross_drawdown = max_drawdown(executed["gross_pnl_proxy"]) if not executed.empty else np.nan
    net_drawdown = max_drawdown(executed["net_pnl_proxy"]) if not executed.empty else np.nan
    sharpe_like = sharpe_like_metric(executed["net_pnl_proxy"]) if not executed.empty else np.nan
    month_span = (
        (stage.effective_end - stage.effective_start).total_seconds() / (30.4375 * 24 * 3600)
        if stage.effective_start is not None and stage.effective_end is not None
        else np.nan
    )

    return {
        "stage_id": stage.stage_id,
        "stage_label": stage.label,
        "risk_model_id": risk_model.risk_model_id,
        "risk_model_label": risk_model.label,
        "cost_scenario_id": combined_cost_id(fee_scenario, slippage_scenario),
        "fee_scenario_id": fee_scenario.fee_id,
        "slippage_scenario_id": slippage_scenario.slippage_id,
        "round_trip_fee_bps": fee_scenario.round_trip_fee_bps,
        "round_trip_slippage_bps": slippage_scenario.round_trip_slippage_bps,
        "trade_count": int(len(executed)),
        "raw_signal_count": int(len(stage_signals)),
        "trades_per_month": safe_divide(len(executed), month_span),
        "win_rate_gross": safe_divide((executed["gross_pnl_proxy"] > 0).sum(), len(executed)),
        "win_rate_net": safe_divide((executed["net_pnl_proxy"] > 0).sum(), len(executed)),
        "avg_return_gross_pct": executed["gross_return_pct"].mean() if not executed.empty else np.nan,
        "avg_return_net_pct": executed["net_return_pct"].mean() if not executed.empty else np.nan,
        "median_return_gross_pct": executed["gross_return_pct"].median() if not executed.empty else np.nan,
        "median_return_net_pct": executed["net_return_pct"].median() if not executed.empty else np.nan,
        "expectancy_gross_pct": executed["gross_return_pct"].mean() if not executed.empty else np.nan,
        "expectancy_net_pct": executed["net_return_pct"].mean() if not executed.empty else np.nan,
        "gross_pnl_proxy": executed["gross_pnl_proxy"].sum() if not executed.empty else np.nan,
        "net_pnl_proxy": executed["net_pnl_proxy"].sum() if not executed.empty else np.nan,
        "profit_factor_gross": gross_profit_factor,
        "profit_factor_net": net_profit_factor,
        "avg_MAE_pct": executed["MAE_pct"].mean() if not executed.empty else np.nan,
        "median_MAE_pct": executed["MAE_pct"].median() if not executed.empty else np.nan,
        "avg_MFE_pct": executed["MFE_pct"].mean() if not executed.empty else np.nan,
        "median_MFE_pct": executed["MFE_pct"].median() if not executed.empty else np.nan,
        "max_drawdown_gross_proxy": gross_drawdown,
        "max_drawdown_net_proxy": net_drawdown,
        "sharpe_like_net": sharpe_like,
        "net_expectancy_after_friction_pct": executed["net_return_pct"].mean() if not executed.empty else np.nan,
    }


def combined_cost_id(fee_scenario: FeeScenario, slippage_scenario: SlippageScenario) -> str:
    return f"{fee_scenario.fee_id}__{slippage_scenario.slippage_id}"


def profit_factor(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return np.nan
    gains = pnl_series.loc[pnl_series > 0].sum()
    losses = pnl_series.loc[pnl_series < 0].sum()
    return safe_divide(gains, abs(losses))


def max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return np.nan
    cumulative = pnl_series.fillna(0).cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    return float(drawdown.max()) if not drawdown.empty else np.nan


def sharpe_like_metric(pnl_series: pd.Series) -> float:
    if pnl_series.empty or pnl_series.std(ddof=0) == 0:
        return np.nan
    return float((pnl_series.mean() / pnl_series.std(ddof=0)) * np.sqrt(len(pnl_series)))


def pick_recommended_risk_model(representative_rows: pd.DataFrame) -> str:
    if representative_rows.empty:
        return RISK_MODELS[0].risk_model_id
    grouped = (
        representative_rows.groupby(["risk_model_id", "risk_model_label"], dropna=False)
        .agg(
            stage_a_net_expectancy_pct=("net_expectancy_after_friction_pct", lambda s: s.loc[representative_rows.loc[s.index, "stage_id"] == "stage_a"].iloc[0] if (representative_rows.loc[s.index, "stage_id"] == "stage_a").any() else np.nan),
            median_net_expectancy_pct=("net_expectancy_after_friction_pct", "median"),
            positive_stage_share=("net_expectancy_after_friction_pct", lambda s: (s > 0).mean()),
            median_drawdown=("max_drawdown_net_proxy", "median"),
            median_trade_count=("trade_count", "median"),
        )
        .reset_index()
    )
    grouped["score"] = (
        grouped["stage_a_net_expectancy_pct"].fillna(0) * 2.0
        + grouped["median_net_expectancy_pct"].fillna(0) * 1.0
        + grouped["positive_stage_share"].fillna(0) * 0.5
        - grouped["median_drawdown"].fillna(0) / 50.0
        + grouped["median_trade_count"].fillna(0) / 5000.0
    )
    return str(grouped.sort_values(["score", "stage_a_net_expectancy_pct"], ascending=[False, False]).iloc[0]["risk_model_id"])


def build_parameter_robustness(
    scan_frame: pd.DataFrame,
    stages: list,
    minute_arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    representative_fee = next(s for s in FEE_SCENARIOS if s.fee_id == "normal_fee")
    representative_slippage = next(s for s in SLIPPAGE_SCENARIOS if s.slippage_id == "moderate_slippage")
    base_risk_model = next(model for model in RISK_MODELS if model.risk_model_id == "fixed_notional_max1")

    for distance_thr, pre_entry_thr, wick_thr, horizon_h in itertools.product(
        DISTANCE_GRID,
        PRE_ENTRY_GRID,
        WICK_GRID,
        HORIZON_GRID_HOURS,
    ):
        params = {
            "distance_to_4h_high_pct_max": distance_thr,
            "pre_entry_return_60m_pct_min": pre_entry_thr,
            "upper_wick_ratio_5m_min": wick_thr,
            "last_5m_return_pct_max": BASE_PARAMS["last_5m_return_pct_max"],
            "exit_horizon_hours": horizon_h,
        }
        signals = build_short_proxy_signals(scan_frame, stages, params)
        for stage in stages:
            stage_signals = signals.loc[signals[f"in_{stage.stage_id}"] == True].copy()
            executed = simulate_portfolio(
                stage_signals,
                minute_arrays,
                horizon_hours=horizon_h,
                risk_model=base_risk_model,
                fee_scenario=representative_fee,
                slippage_scenario=representative_slippage,
            )
            summary = summarize_validation_portfolio(
                executed,
                stage_signals,
                stage,
                base_risk_model,
                representative_fee,
                representative_slippage,
            )
            summary.update(
                {
                    "distance_to_4h_high_pct_max": distance_thr,
                    "pre_entry_return_60m_pct_min": pre_entry_thr,
                    "upper_wick_ratio_5m_min": wick_thr,
                    "last_5m_return_pct_max": BASE_PARAMS["last_5m_return_pct_max"],
                    "exit_horizon_hours": horizon_h,
                }
            )
            rows.append(summary)
    return pd.DataFrame(rows)


def build_stage_consistency(
    base_signals: pd.DataFrame,
    stages: list,
    minute_arrays: dict[str, np.ndarray],
    representative_rows: pd.DataFrame,
    recommended_risk_model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee_scenario = next(s for s in FEE_SCENARIOS if s.fee_id == "normal_fee")
    slippage_scenario = next(s for s in SLIPPAGE_SCENARIOS if s.slippage_id == "moderate_slippage")
    risk_model = next(model for model in RISK_MODELS if model.risk_model_id == recommended_risk_model_id)

    stage_rows = []
    monthly_rows = []
    for stage in stages:
        stage_signals = base_signals.loc[base_signals[f"in_{stage.stage_id}"] == True].copy()
        executed = simulate_portfolio(
            stage_signals,
            minute_arrays,
            horizon_hours=BASE_PARAMS["exit_horizon_hours"],
            risk_model=risk_model,
            fee_scenario=fee_scenario,
            slippage_scenario=slippage_scenario,
        )
        if executed.empty:
            stage_rows.append({"stage_id": stage.stage_id, "stage_label": stage.label, "risk_model_id": risk_model.risk_model_id})
            continue

        daily = build_daily_curve(executed)
        monthly = build_monthly_breakdown(executed, stage)
        monthly_rows.append(monthly)
        stage_rows.append(
            {
                "stage_id": stage.stage_id,
                "stage_label": stage.label,
                "risk_model_id": risk_model.risk_model_id,
                "risk_model_label": risk_model.label,
                "trade_count": int(len(executed)),
                "trades_per_month": safe_divide(len(executed), ((stage.effective_end - stage.effective_start).total_seconds() / (30.4375 * 24 * 3600))),
                "gross_pnl_proxy": executed["gross_pnl_proxy"].sum(),
                "net_pnl_proxy": executed["net_pnl_proxy"].sum(),
                "gross_expectancy_pct": executed["gross_return_pct"].mean(),
                "net_expectancy_pct": executed["net_return_pct"].mean(),
                "profit_factor_net": profit_factor(executed["net_pnl_proxy"]),
                "max_drawdown_net_proxy": max_drawdown(executed["net_pnl_proxy"]),
                "top_5pct_trade_share_of_net_pnl": top_trade_share(executed["net_pnl_proxy"], 0.05),
                "net_pnl_without_top_5pct": remove_top_trade_tail(executed["net_pnl_proxy"], 0.05).sum(),
                "net_expectancy_without_top_5pct_pct": remove_top_trade_tail(executed["net_return_pct"], 0.05).mean(),
                "positive_month_share": (monthly["net_pnl_proxy"] > 0).mean() if not monthly.empty else np.nan,
                "median_monthly_net_pnl": monthly["net_pnl_proxy"].median() if not monthly.empty else np.nan,
                "rolling_30d_median_net_pnl": daily["rolling_30d_net_pnl"].median() if not daily.empty else np.nan,
                "rolling_60d_median_net_pnl": daily["rolling_60d_net_pnl"].median() if not daily.empty else np.nan,
                "rolling_30d_positive_share": (daily["rolling_30d_net_pnl"] > 0).mean() if not daily.empty else np.nan,
                "rolling_60d_positive_share": (daily["rolling_60d_net_pnl"] > 0).mean() if not daily.empty else np.nan,
            }
        )
    stage_consistency = pd.DataFrame(stage_rows)
    monthly_breakdown = pd.concat(monthly_rows, ignore_index=True, sort=False) if monthly_rows else pd.DataFrame()
    return stage_consistency, monthly_breakdown


def build_daily_curve(executed: pd.DataFrame) -> pd.DataFrame:
    daily = (
        executed.assign(entry_day=executed["entry_time"].dt.floor("D"))
        .groupby("entry_day", dropna=False)
        .agg(net_pnl_proxy=("net_pnl_proxy", "sum"))
        .reset_index()
        .sort_values("entry_day")
    )
    if daily.empty:
        return daily
    all_days = pd.date_range(daily["entry_day"].min(), daily["entry_day"].max(), freq="D", tz="UTC")
    daily = daily.set_index("entry_day").reindex(all_days, fill_value=0.0).rename_axis("entry_day").reset_index()
    daily["rolling_30d_net_pnl"] = daily["net_pnl_proxy"].rolling(30, min_periods=10).sum()
    daily["rolling_60d_net_pnl"] = daily["net_pnl_proxy"].rolling(60, min_periods=20).sum()
    return daily


def build_monthly_breakdown(executed: pd.DataFrame, stage) -> pd.DataFrame:
    if executed.empty:
        return pd.DataFrame()
    monthly = (
        executed.assign(
            month=executed["entry_time"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        )
        .groupby("month", dropna=False)
        .agg(
            trade_count=("signal_id", "size"),
            gross_pnl_proxy=("gross_pnl_proxy", "sum"),
            net_pnl_proxy=("net_pnl_proxy", "sum"),
            win_rate_net=("net_pnl_proxy", lambda s: safe_divide((s > 0).sum(), s.notna().sum())),
            net_expectancy_pct=("net_return_pct", "mean"),
        )
        .reset_index()
        .sort_values("month")
    )
    monthly["stage_id"] = stage.stage_id
    monthly["stage_label"] = stage.label
    return monthly


def top_trade_share(series: pd.Series, share: float) -> float:
    clean = series.dropna().sort_values(ascending=False)
    if clean.empty:
        return np.nan
    top_n = max(1, int(np.ceil(len(clean) * share)))
    positive_total = clean.loc[clean > 0].sum()
    return safe_divide(clean.head(top_n).sum(), positive_total if positive_total != 0 else clean.sum())


def remove_top_trade_tail(series: pd.Series, share: float) -> pd.Series:
    clean = series.dropna().sort_values(ascending=False)
    if clean.empty:
        return clean
    top_n = max(1, int(np.ceil(len(clean) * share)))
    remaining = clean.iloc[top_n:]
    return remaining


def build_validation_summary(
    stage_consistency: pd.DataFrame,
    representative_rows: pd.DataFrame,
    recommended_risk_model_id: str,
) -> pd.DataFrame:
    if stage_consistency.empty:
        return pd.DataFrame()
    selected = representative_rows.loc[representative_rows["risk_model_id"] == recommended_risk_model_id].copy()
    summary = stage_consistency.merge(
        selected[
            [
                "stage_id",
                "risk_model_id",
                "trade_count",
                "win_rate_net",
                "avg_return_net_pct",
                "median_return_net_pct",
                "expectancy_net_pct",
                "gross_pnl_proxy",
                "net_pnl_proxy",
                "profit_factor_net",
                "avg_MAE_pct",
                "median_MAE_pct",
                "avg_MFE_pct",
                "max_drawdown_net_proxy",
                "sharpe_like_net",
                "net_expectancy_after_friction_pct",
            ]
        ],
        on=["stage_id", "risk_model_id"],
        how="left",
        suffixes=("_consistency", ""),
    )
    summary["representative_cost_scenario_id"] = REPRESENTATIVE_COST_ID
    return summary.sort_values("stage_id").reset_index(drop=True)


def build_validation_section(
    validation_summary: pd.DataFrame,
    representative_rows: pd.DataFrame,
    parameter_robustness: pd.DataFrame,
) -> str:
    if validation_summary.empty:
        return (
            "ETH SHORT PROXY VALIDATION\n"
            "--------------------------\n"
            "Short proxy validation could not be completed.\n"
        )

    stage_a = validation_summary.loc[validation_summary["stage_id"] == "stage_a"].iloc[0]
    stage_b = validation_summary.loc[validation_summary["stage_id"] == "stage_b"].iloc[0]
    stage_c = validation_summary.loc[validation_summary["stage_id"] == "stage_c"].iloc[0]
    recommended_risk_label = str(stage_a.get("risk_model_label", "n/a"))

    robustness_base = parameter_robustness.loc[
        (parameter_robustness["stage_id"] == "stage_a")
        & (parameter_robustness["distance_to_4h_high_pct_max"] == BASE_PARAMS["distance_to_4h_high_pct_max"])
        & (parameter_robustness["pre_entry_return_60m_pct_min"] == BASE_PARAMS["pre_entry_return_60m_pct_min"])
        & (parameter_robustness["upper_wick_ratio_5m_min"] == BASE_PARAMS["upper_wick_ratio_5m_min"])
        & (parameter_robustness["exit_horizon_hours"] == BASE_PARAMS["exit_horizon_hours"])
    ]
    near_base = parameter_robustness.loc[
        (parameter_robustness["stage_id"] == "stage_a")
        & parameter_robustness["distance_to_4h_high_pct_max"].between(0.45, 0.75, inclusive="both")
        & parameter_robustness["pre_entry_return_60m_pct_min"].between(0.25, 0.65, inclusive="both")
        & parameter_robustness["upper_wick_ratio_5m_min"].between(0.15, 0.35, inclusive="both")
    ]
    robustness_positive_share = (near_base["net_expectancy_after_friction_pct"] > 0).mean() if not near_base.empty else np.nan

    alive_outside_stage_a = (
        pd.notna(stage_b.get("net_expectancy_after_friction_pct")) and stage_b.get("net_expectancy_after_friction_pct", 0) > 0
    ) or (
        pd.notna(stage_c.get("net_expectancy_after_friction_pct")) and stage_c.get("net_expectancy_after_friction_pct", 0) > 0
    )
    confidence = determine_validation_confidence(stage_a, stage_b, stage_c, robustness_positive_share)
    final_verdict = (
        "The ETH short proxy is best treated as a candidate tradable model for further paper trading / forward testing, not as a robust tradable system yet."
        if pd.notna(stage_a.get("net_expectancy_after_friction_pct")) and stage_a.get("net_expectancy_after_friction_pct", 0) > 0 and robustness_positive_share >= 0.35
        else "The edge looks fragile under realistic validation, so it should not be treated as a ready-to-trade model."
    )

    report = f"""
ETH SHORT PROXY VALIDATION
--------------------------
This section validates the ETH short proxy as a standalone style-inspired setup, with realistic friction, simple position sizing, parameter drift checks, and stage-by-stage robustness.

    Representative validation scenario:
- Cost scenario: {REPRESENTATIVE_COST_ID.replace('__', ' + ')}.
- Recommended risk model: {recommended_risk_label}.

Stage results after friction:
- Stage A: {fmt_int(stage_a.get('trade_count'))} trades, win rate {pct(stage_a.get('win_rate_net'))}, net expectancy {fmt_number(stage_a.get('net_expectancy_after_friction_pct'))}%, net PnL proxy {fmt_number(stage_a.get('net_pnl_proxy'))}, max drawdown proxy {fmt_number(stage_a.get('max_drawdown_net_proxy'))}.
- Stage B: {fmt_int(stage_b.get('trade_count'))} trades, win rate {pct(stage_b.get('win_rate_net'))}, net expectancy {fmt_number(stage_b.get('net_expectancy_after_friction_pct'))}%, net PnL proxy {fmt_number(stage_b.get('net_pnl_proxy'))}, max drawdown proxy {fmt_number(stage_b.get('max_drawdown_net_proxy'))}.
- Stage C: {fmt_int(stage_c.get('trade_count'))} trades, win rate {pct(stage_c.get('win_rate_net'))}, net expectancy {fmt_number(stage_c.get('net_expectancy_after_friction_pct'))}%, net PnL proxy {fmt_number(stage_c.get('net_pnl_proxy'))}, max drawdown proxy {fmt_number(stage_c.get('max_drawdown_net_proxy'))}.

Robustness read:
- Positive share across nearby Stage A parameter variants: {pct(robustness_positive_share)}.
- Edge outside Stage A: {"partly alive" if alive_outside_stage_a else "mostly Stage-A dependent"}.
- Confidence level: {confidence}.

Final verdict:
- {final_verdict}
""".strip()
    return report + "\n"


def determine_validation_confidence(stage_a: pd.Series, stage_b: pd.Series, stage_c: pd.Series, robustness_positive_share: float) -> str:
    if (
        safe_number(stage_a.get("net_expectancy_after_friction_pct")) > 0.10
        and robustness_positive_share >= 0.50
        and (safe_number(stage_b.get("net_expectancy_after_friction_pct")) > -0.02 or safe_number(stage_c.get("net_expectancy_after_friction_pct")) > -0.02)
    ):
        return "high"
    if (
        safe_number(stage_a.get("net_expectancy_after_friction_pct")) > 0.03
        and robustness_positive_share >= 0.25
    ):
        return "medium"
    return "low"


def create_eth_short_proxy_validation_plots(
    validation_summary: pd.DataFrame,
    net_vs_gross_curves: pd.DataFrame,
    parameter_robustness: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    if validation_summary.empty:
        return
    plots_dir = output_dir / PLOTS_DIRNAME
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_net_vs_gross(validation_summary)
    path1 = plots_dir / "eth_short_proxy_net_vs_gross.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info("Saved plot: %s", path1.name)

    fig2 = plot_short_proxy_drawdown(net_vs_gross_curves)
    path2 = plots_dir / "eth_short_proxy_drawdown.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Saved plot: %s", path2.name)

    fig3 = plot_robustness_heatmap(parameter_robustness)
    path3 = plots_dir / "eth_short_proxy_robustness_heatmap.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info("Saved plot: %s", path3.name)


def plot_net_vs_gross(validation_summary: pd.DataFrame) -> plt.Figure:
    ordered = validation_summary.sort_values("stage_id").copy()
    labels = ordered["stage_id"].str.upper().tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ordered["gross_pnl_proxy"], width=width, label="Gross", color="#bbbbbb")
    ax.bar(x + width / 2, ordered["net_pnl_proxy"], width=width, label="Net", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("ETH Short Proxy: Gross vs Net PnL Proxy")
    ax.set_ylabel("PnL proxy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig


def plot_short_proxy_drawdown(net_vs_gross_curves: pd.DataFrame) -> plt.Figure:
    representative = net_vs_gross_curves.loc[
        (net_vs_gross_curves["cost_scenario_id"] == REPRESENTATIVE_COST_ID)
        & (net_vs_gross_curves["risk_model_id"] == "fixed_notional_max1")
    ].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    stage_order = ["stage_a", "stage_b", "stage_c"]
    for ax, stage_id in zip(axes, stage_order, strict=False):
        stage_frame = representative.loc[representative["stage_id"] == stage_id].sort_values("entry_time").copy()
        if stage_frame.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        running_peak = stage_frame["cumulative_net_pnl_proxy"].cummax()
        drawdown = running_peak - stage_frame["cumulative_net_pnl_proxy"]
        ax.plot(stage_frame["entry_time"], stage_frame["cumulative_net_pnl_proxy"], label="Net equity", color="#1f77b4")
        ax.plot(stage_frame["entry_time"], -drawdown, label="Drawdown", color="#d62728", alpha=0.7)
        ax.set_title(f"{stage_id.upper()} net equity and drawdown")
        ax.grid(alpha=0.25)
        ax.legend()
    return fig


def plot_robustness_heatmap(parameter_robustness: pd.DataFrame) -> plt.Figure:
    subset = parameter_robustness.loc[
        (parameter_robustness["stage_id"] == "stage_a")
        & (parameter_robustness["upper_wick_ratio_5m_min"] == BASE_PARAMS["upper_wick_ratio_5m_min"])
        & (parameter_robustness["exit_horizon_hours"] == BASE_PARAMS["exit_horizon_hours"])
    ].copy()
    pivot = subset.pivot_table(
        index="distance_to_4h_high_pct_max",
        columns="pre_entry_return_60m_pct_min",
        values="net_expectancy_after_friction_pct",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{col:.2f}" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{idx:.2f}" for idx in pivot.index])
    ax.set_xlabel("Pre-entry 60m return threshold")
    ax.set_ylabel("Distance to 4h high threshold")
    ax.set_title("Stage A net expectancy heatmap\n(wick=0.25, horizon=12h, normal fee + moderate slippage)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Net expectancy %")
    return fig


def safe_number(value: float | int | None) -> float:
    return float(value) if value is not None and pd.notna(value) else 0.0
