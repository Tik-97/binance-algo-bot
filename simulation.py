from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import safe_divide


def max_drawdown_from_equity(equity: np.ndarray | pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    values = np.asarray(equity, dtype=float)
    running_peak = np.maximum.accumulate(values)
    drawdowns = running_peak - values
    return float(np.nanmax(drawdowns))


def compute_time_weighted_concurrency(df: pd.DataFrame) -> tuple[float, int]:
    valid = df.dropna(subset=["opened_dt_utc", "closed_dt_utc"]).copy()
    if valid.empty:
        return 0.0, 0

    events: list[tuple[pd.Timestamp, int, int]] = []
    for _, row in valid.iterrows():
        opened = row["opened_dt_utc"]
        closed = row["closed_dt_utc"]
        if closed < opened:
            continue
        # Close events get priority so touching intervals are not overcounted.
        events.append((opened, 1, 1))
        events.append((closed, 0, -1))

    if not events:
        return 0.0, 0

    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    max_active = 0
    weighted_seconds = 0.0
    total_seconds = 0.0
    previous_time = events[0][0]

    for event_time, _, delta in events:
        elapsed = (event_time - previous_time).total_seconds()
        if elapsed > 0:
            weighted_seconds += active * elapsed
            total_seconds += elapsed
            previous_time = event_time
        active += delta
        max_active = max(max_active, active)

    average_concurrent = safe_divide(weighted_seconds, total_seconds)
    return average_concurrent, max_active


def build_profile_stats(df: pd.DataFrame, concentration: dict, side_stats: pd.DataFrame, symbol_stats: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["opened_dt_utc", "closed_dt_utc"]).copy()
    avg_concurrent, max_concurrent = compute_time_weighted_concurrency(valid)

    if valid.empty:
        return pd.DataFrame([{}])

    valid["open_week"] = valid["opened_dt_utc"].dt.tz_localize(None).dt.to_period("W").astype(str)
    weekly_counts = valid.groupby("open_week")["trade_number"].size()
    interarrival_hours = valid["opened_dt_utc"].sort_values().diff().dt.total_seconds() / 3600.0

    side_bias = "balanced"
    if not side_stats.empty:
        top_side = side_stats.sort_values("trades", ascending=False).iloc[0]
        if safe_divide(top_side["trades"], len(df)) >= 0.6:
            side_bias = str(top_side["side"])

    top_symbol_trade_share = symbol_stats["trade_share"].max() if not symbol_stats.empty else np.nan
    top_symbol_pnl_share = symbol_stats["pnl_share"].max() if not symbol_stats.empty else np.nan

    profile = {
        "trades_per_week_mean": weekly_counts.mean(),
        "trades_per_week_median": weekly_counts.median(),
        "trades_per_week_max": weekly_counts.max(),
        "average_concurrent_positions": avg_concurrent,
        "max_concurrent_positions": max_concurrent,
        "holding_time_mean_hours": valid["holding_hours"].mean(),
        "holding_time_median_hours": valid["holding_hours"].median(),
        "winner_holding_mean_hours": valid.loc[valid["closingPnl"] > 0, "holding_hours"].mean(),
        "loser_holding_mean_hours": valid.loc[valid["closingPnl"] < 0, "holding_hours"].mean(),
        "wins": int((valid["closingPnl"] > 0).sum()),
        "losses": int((valid["closingPnl"] < 0).sum()),
        "flats": int((valid["closingPnl"] == 0).sum()),
        "top_5_total_pnl_share": concentration["top_5_pct_total_pnl"],
        "approx_max_drawdown": max_drawdown_from_equity(valid["cumulative_pnl"]),
        "side_bias": side_bias,
        "top_symbol_trade_share": top_symbol_trade_share,
        "top_symbol_pnl_share": top_symbol_pnl_share,
        "mean_interarrival_hours": interarrival_hours.mean(),
        "median_interarrival_hours": interarrival_hours.median(),
        "clustered_entries_share_lt_1h": safe_divide((interarrival_hours < 1).sum(), interarrival_hours.notna().sum()),
        "clustered_entries_share_lt_4h": safe_divide((interarrival_hours < 4).sum(), interarrival_hours.notna().sum()),
    }
    return pd.DataFrame([profile])


def evaluate_edge_checks(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []

    def collect(name: str, frame: pd.DataFrame) -> None:
        pnl = frame["closingPnl"].fillna(0)
        equity = pnl.cumsum()
        scenarios.append(
            {
                "scenario": name,
                "trades": len(frame),
                "total_pnl": pnl.sum(),
                "avg_pnl": pnl.mean() if len(frame) else np.nan,
                "median_pnl": pnl.median() if len(frame) else np.nan,
                "win_rate": safe_divide((pnl > 0).sum(), pnl.notna().sum()),
                "approx_max_drawdown": max_drawdown_from_equity(equity),
            }
        )

    baseline = df.sort_values("closed_dt_utc").reset_index(drop=True)
    collect("baseline", baseline)

    without_top_5 = baseline.drop(baseline.nlargest(5, "closingPnl").index).reset_index(drop=True)
    collect("without_top_5_trades", without_top_5)

    losses = baseline.loc[baseline["closingPnl"] < 0, "closingPnl"]
    if not losses.empty:
        loss_cap = abs(losses).quantile(0.75)
        capped = baseline.copy()
        capped["closingPnl"] = capped["closingPnl"].clip(lower=-loss_cap)
        collect(f"loss_capped_at_75pct_loss_{loss_cap:.2f}", capped)

    holding_limited = baseline.loc[baseline["holding_hours"] <= 24].reset_index(drop=True)
    collect("only_trades_held_24h_or_less", holding_limited)

    return pd.DataFrame(scenarios)


@dataclass
class ProbabilisticModel:
    start_time: pd.Timestamp
    horizon_hours: int
    base_lambda_per_hour: float
    hour_multipliers: np.ndarray
    side_values: np.ndarray
    side_probs: np.ndarray
    symbol_by_side: dict[str, tuple[np.ndarray, np.ndarray]]
    holding_by_side: dict[str, np.ndarray]
    return_by_side: dict[str, np.ndarray]
    notional_by_side: dict[str, np.ndarray]


def fit_probabilistic_model(df: pd.DataFrame) -> ProbabilisticModel:
    valid = df.dropna(subset=["opened_dt_utc", "closed_dt_utc", "return_pct_approx", "approximate_notional", "holding_hours"]).copy()
    if valid.empty:
        raise ValueError("Not enough valid trades to fit probabilistic model.")

    start_time = valid["opened_dt_utc"].min().floor("h")
    end_time = valid["closed_dt_utc"].max().ceil("h")
    horizon_hours = max(1, int(math.ceil((end_time - start_time).total_seconds() / 3600.0)))

    base_lambda = len(valid) / horizon_hours
    hour_counts = valid["open_hour_utc"].value_counts().reindex(range(24), fill_value=0).astype(float)
    hour_probs = hour_counts / max(hour_counts.sum(), 1.0)
    hour_multipliers = (hour_probs / (1.0 / 24.0)).to_numpy()

    side_counts = valid["side"].fillna("Unknown").value_counts(normalize=True)
    side_values = side_counts.index.to_numpy()
    side_probs = side_counts.values

    symbol_by_side: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    holding_by_side: dict[str, np.ndarray] = {}
    return_by_side: dict[str, np.ndarray] = {}
    notional_by_side: dict[str, np.ndarray] = {}

    for side in side_values:
        side_frame = valid.loc[valid["side"] == side]
        symbol_probs = side_frame["symbol"].value_counts(normalize=True)
        symbol_by_side[str(side)] = (symbol_probs.index.to_numpy(), symbol_probs.values)
        holding_by_side[str(side)] = side_frame["holding_hours"].dropna().to_numpy()
        return_by_side[str(side)] = side_frame["return_pct_approx"].dropna().to_numpy()
        notional_by_side[str(side)] = side_frame["approximate_notional"].dropna().to_numpy()

    return ProbabilisticModel(
        start_time=start_time,
        horizon_hours=horizon_hours,
        base_lambda_per_hour=base_lambda,
        hour_multipliers=hour_multipliers,
        side_values=side_values,
        side_probs=side_probs,
        symbol_by_side=symbol_by_side,
        holding_by_side=holding_by_side,
        return_by_side=return_by_side,
        notional_by_side=notional_by_side,
    )


def simulate_monte_carlo(df: pd.DataFrame, runs: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = fit_probabilistic_model(df)
    rng = np.random.default_rng(seed)

    summaries: list[dict] = []
    path_records: list[pd.DataFrame] = []

    for run_id in range(1, runs + 1):
        simulated_rows = []
        for hour_offset in range(model.horizon_hours):
            current_hour = model.start_time + pd.Timedelta(hours=hour_offset)
            lam = model.base_lambda_per_hour * model.hour_multipliers[current_hour.hour]
            entries = rng.poisson(max(lam, 0.0))
            for _ in range(entries):
                side = str(rng.choice(model.side_values, p=model.side_probs))
                symbol_values, symbol_probs = model.symbol_by_side[side]
                symbol = str(rng.choice(symbol_values, p=symbol_probs))
                holding = float(rng.choice(model.holding_by_side[side]))
                return_pct = float(rng.choice(model.return_by_side[side]))
                notional = float(rng.choice(model.notional_by_side[side]))
                pnl = notional * return_pct

                minute_offset = float(rng.uniform(0, 60))
                opened = current_hour + pd.Timedelta(minutes=minute_offset)
                closed = opened + pd.Timedelta(hours=max(holding, 0.0))

                simulated_rows.append(
                    {
                        "run_id": run_id,
                        "symbol": symbol,
                        "side": side,
                        "opened_dt_utc": opened,
                        "closed_dt_utc": closed,
                        "holding_hours": holding,
                        "return_pct_approx": return_pct,
                        "approximate_notional": notional,
                        "closingPnl": pnl,
                    }
                )

        run_df = pd.DataFrame(simulated_rows)
        if run_df.empty:
            summaries.append(
                {
                    "run_id": run_id,
                    "trades": 0,
                    "total_pnl": 0.0,
                    "win_rate": np.nan,
                    "approx_max_drawdown": 0.0,
                }
            )
            continue

        run_df = run_df.sort_values("closed_dt_utc").reset_index(drop=True)
        run_df["trade_number"] = np.arange(1, len(run_df) + 1)
        run_df["cumulative_pnl"] = run_df["closingPnl"].cumsum()
        max_dd = max_drawdown_from_equity(run_df["cumulative_pnl"])

        summaries.append(
            {
                "run_id": run_id,
                "trades": len(run_df),
                "total_pnl": run_df["closingPnl"].sum(),
                "win_rate": safe_divide((run_df["closingPnl"] > 0).sum(), len(run_df)),
                "approx_max_drawdown": max_dd,
            }
        )

        if run_id <= 100:
            path_records.append(
                run_df[["run_id", "trade_number", "cumulative_pnl"]].copy()
            )

    summary_df = pd.DataFrame(summaries)
    paths_df = pd.concat(path_records, ignore_index=True) if path_records else pd.DataFrame(columns=["run_id", "trade_number", "cumulative_pnl"])
    return summary_df, paths_df
