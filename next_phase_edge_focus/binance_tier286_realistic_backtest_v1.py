from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import ethShortTrader as eth_short_trader_module
from ethShortTrader import (
    BASE_DIR,
    CANDLES_1M_PATH,
    JSON_TRADES_PATH,
    STARTING_BALANCE,
    RISK_PER_TRADE_PCT,
    LEVERAGE,
    INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT,
    build_features as build_execution_features,
    get_reference_window,
    load_candles as load_execution_candles,
    load_reference_eth_shorts,
    match_to_reference,
    run_trade_exit_short,
)
from eth_short_candidate_engine import (
    build_features as build_candidate_features,
    deduplicate_candidates,
    generate_candidates,
)
from eth_short_execution_backtest_v1 import get_index_at_or_after
from eth_short_live_safe_regime_classifier import predict_live_safe_day_regime_from_bar
from eth_short_regime_aware_prototype_v3 import classify_near_noise_subtype_live
from eth_short_regime_aware_live_like_backtest_v4 import is_strict_cooling_extension
from next_phase_edge_focus.live_helpers import build_market_regime_frame_from_candles, is_calm_mild_transition


base = sys.modules[__name__]


OUTPUT_DIR = BASE_DIR / "next_phase_edge_focus" / "results" / "binance_tier286_realistic_backtest_v1"

CORE_REGIME = "near_entry_breakdown_noise"
SATELLITE_REGIME = "mixed_transition"
MILD_SUBTYPE = "mild_noise_v1_live"
COOLING_SUBTYPE = "cooling_v3_live"
ALLOWED_CORE_TREND_REGIMES = {"strong_bear", "weak_bear"}
SATELLITE_TREND_REGIME = "strong_bear"
SATELLITE_MIN_REL_VOLUME = 0.80
SATELLITE_SECONDARY_BODY_MAX = 0.50
SATELLITE_SECONDARY_EXCLUDED_HOUR = 13

BINANCE_MAKER_FEE_PCT = 0.02
BINANCE_TAKER_FEE_PCT = 0.05
BINANCE_LIMIT_FILL_SLIPPAGE_PCT = 0.00
BINANCE_MARKET_ENTRY_SLIPPAGE_PCT = 0.025
BINANCE_EXIT_SLIPPAGE_PCT = 0.025

SELECTIVE_COOLING_MOVE1M_MIN = -0.30
SELECTIVE_COOLING_MOVE3M_MIN = 0.00
SELECTIVE_COOLING_MOVE3M_MAX = 0.50
ENTRY_WAIT_BARS = 15
ENTRY_RETEST_ABOVE_CLOSE_PCT = 0.15

EXPANSION_REGIME = "other_non_target"
EXPANSION_SUBTYPE = MILD_SUBTYPE
EXPANSION_TREND_REGIME = "strong_bear"
EXPANSION_VOL_REGIME = "high_vol"
EXPANSION_MIN_SCORE = 0.84
EXPANSION_MAX_MOVE_1M_PCT = 0.25
EXPANSION_MAX_MOVE_3M_PCT = 0.40
EXPANSION_MAX_DIST_1H_HIGH_PCT = 0.08
EXPANSION_MAX_BOUNCE_60M_PCT = 0.80


def is_selective_relaxed_cooling(bar: pd.Series) -> bool:
    return (
        float(bar["move_1m_pct"]) >= SELECTIVE_COOLING_MOVE1M_MIN
        and float(bar["move_3m_pct"]) >= SELECTIVE_COOLING_MOVE3M_MIN
        and float(bar["move_3m_pct"]) <= SELECTIVE_COOLING_MOVE3M_MAX
    )


def simulate_limit_retest_entry_binance(
    df: pd.DataFrame,
    anchor_idx: int,
    stop_price_initial: float,
) -> tuple[int | None, pd.Timestamp | None, float | None, float | None]:
    if anchor_idx is None or anchor_idx >= len(df):
        return None, None, None, None

    anchor_bar = df.iloc[anchor_idx]
    order_price = float(anchor_bar["close"]) * (1.0 + ENTRY_RETEST_ABOVE_CLOSE_PCT / 100.0)
    if order_price >= stop_price_initial:
        return None, anchor_bar["open_time"], order_price, None

    last_idx = min(anchor_idx + ENTRY_WAIT_BARS, len(df) - 1)
    for j in range(anchor_idx + 1, last_idx + 1):
        bar = df.iloc[j]
        if float(bar["high"]) >= order_price:
            entry_price = order_price * (1.0 + BINANCE_LIMIT_FILL_SLIPPAGE_PCT / 100.0)
            return j, bar["open_time"], order_price, entry_price
    return None, anchor_bar["open_time"], order_price, None


@contextmanager
def patched_binance_execution_assumptions():
    old_fee = eth_short_trader_module.FEE_PER_SIDE_PCT
    old_market_slippage = eth_short_trader_module.MARKET_SLIPPAGE_PER_SIDE_PCT
    try:
        eth_short_trader_module.FEE_PER_SIDE_PCT = BINANCE_TAKER_FEE_PCT
        eth_short_trader_module.MARKET_SLIPPAGE_PER_SIDE_PCT = BINANCE_EXIT_SLIPPAGE_PCT
        yield
    finally:
        eth_short_trader_module.FEE_PER_SIDE_PCT = old_fee
        eth_short_trader_module.MARKET_SLIPPAGE_PER_SIDE_PCT = old_market_slippage


def build_period_summary(trades_df: pd.DataFrame, ref_start: pd.Timestamp, ref_end: pd.Timestamp) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    frame = trades_df.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["period"] = np.where(
        frame["entry_time"] < ref_start,
        "pre_reference",
        np.where(frame["entry_time"] <= ref_end, "reference_window", "post_reference"),
    )
    rows: list[dict] = []
    for period, group in frame.groupby("period", sort=False):
        wins = group.loc[group["pnl_usd"] > 0, "pnl_usd"]
        losses = group.loc[group["pnl_usd"] < 0, "pnl_usd"]
        rows.append(
            {
                "period": period,
                "trade_count": int(len(group)),
                "win_rate_pct": float((group["pnl_usd"] > 0).mean() * 100.0),
                "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
                "net_pnl_usd": float(group["pnl_usd"].sum()),
            }
        )
    return pd.DataFrame(rows)


def is_other_non_target_mild_strong_bear_expansion(
    trigger_bar: pd.Series,
    market_ctx: dict[str, object],
    candidate_score: float,
    predicted_regime: str,
    subtype: str,
) -> bool:
    return (
        predicted_regime == EXPANSION_REGIME
        and subtype == EXPANSION_SUBTYPE
        and market_ctx["trend_regime"] == EXPANSION_TREND_REGIME
        and market_ctx["vol_regime"] == EXPANSION_VOL_REGIME
        and float(candidate_score) >= EXPANSION_MIN_SCORE
        and float(trigger_bar["move_1m_pct"]) <= EXPANSION_MAX_MOVE_1M_PCT
        and float(trigger_bar["move_3m_pct"]) <= EXPANSION_MAX_MOVE_3M_PCT
        and float(trigger_bar["dist_1h_high_pct"]) <= EXPANSION_MAX_DIST_1H_HIGH_PCT
        and float(trigger_bar["bounce_60m_pct"]) <= EXPANSION_MAX_BOUNCE_60M_PCT
    )


def run_backtest_binance_tier286(
    execution_df: pd.DataFrame,
    deduped_candidates: pd.DataFrame,
    market_regime_lookup: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict]:
    balance = STARTING_BALANCE
    trades: list[dict] = []
    last_exit_time = pd.Timestamp.min.tz_localize("UTC")

    skipped_overlap = 0
    skipped_wrong_market_regime = 0
    skipped_non_target = 0
    skipped_wrong_subtype = 0
    skipped_selective_cooling = 0
    skipped_satellite_filter = 0
    skipped_satellite_low_rel_volume = 0
    skipped_satellite_secondary = 0
    skipped_unfilled = 0
    skipped_invalid_stop = 0
    fallback_market_entries = 0
    accepted_strict_cooling = 0
    accepted_selective_cooling_market = 0
    accepted_satellite_rv08 = 0
    accepted_satellite_secondary = 0
    accepted_other_non_target_expansion = 0
    predicted_core_candidates = 0
    predicted_satellite_candidates = 0
    predicted_other_non_target_candidates = 0

    with patched_binance_execution_assumptions():
        for _, row in deduped_candidates.sort_values("open_time").iterrows():
            candidate_time = pd.to_datetime(row["open_time"], utc=True)
            if candidate_time < last_exit_time:
                skipped_overlap += 1
                continue

            market_ctx = market_regime_lookup.get(candidate_time.strftime("%Y-%m-%d"))
            if not market_ctx:
                skipped_wrong_market_regime += 1
                continue

            anchor_idx = get_index_at_or_after(execution_df, candidate_time)
            if anchor_idx is None or anchor_idx + 1 >= len(execution_df):
                continue

            trigger_bar = execution_df.iloc[anchor_idx]
            confirm_bar = execution_df.iloc[min(anchor_idx + 1, len(execution_df) - 1)]
            predicted_regime = predict_live_safe_day_regime_from_bar(trigger_bar)
            subtype = classify_near_noise_subtype_live(trigger_bar)

            force_market = False
            accepted_subtype = ""
            entry_mode_suffix = ""

            if predicted_regime == CORE_REGIME:
                if market_ctx["trend_regime"] not in ALLOWED_CORE_TREND_REGIMES:
                    skipped_wrong_market_regime += 1
                    continue
                predicted_core_candidates += 1

                if subtype == MILD_SUBTYPE:
                    accepted_subtype = MILD_SUBTYPE
                    entry_mode_suffix = "tier2p5_core"
                elif subtype == COOLING_SUBTYPE and is_strict_cooling_extension(trigger_bar):
                    accepted_subtype = "strict_cooling_v1_live"
                    entry_mode_suffix = "tier2p5_core"
                    accepted_strict_cooling += 1
                elif subtype == COOLING_SUBTYPE and is_selective_relaxed_cooling(trigger_bar):
                    accepted_subtype = "selective_relaxed_cooling_market_v1"
                    entry_mode_suffix = "tier2p5_core"
                    force_market = True
                    accepted_selective_cooling_market += 1
                elif subtype == COOLING_SUBTYPE:
                    skipped_selective_cooling += 1
                    continue
                else:
                    skipped_wrong_subtype += 1
                    continue

            elif predicted_regime == SATELLITE_REGIME:
                if market_ctx["trend_regime"] != SATELLITE_TREND_REGIME:
                    skipped_wrong_market_regime += 1
                    continue
                predicted_satellite_candidates += 1

                if subtype != MILD_SUBTYPE:
                    skipped_wrong_subtype += 1
                    continue
                if not is_calm_mild_transition(trigger_bar):
                    skipped_satellite_filter += 1
                    continue

                rel_volume = float(trigger_bar["relative_volume_1m"])
                body_ratio = float(trigger_bar["body_ratio"])
                entry_hour = int(pd.to_datetime(trigger_bar["open_time"], utc=True).hour)

                if rel_volume >= SATELLITE_MIN_REL_VOLUME:
                    accepted_subtype = "mixed_transition_mild_strong_bear_rv08_market"
                    entry_mode_suffix = "tier5_satellite_rv08"
                    force_market = True
                    accepted_satellite_rv08 += 1
                elif body_ratio <= SATELLITE_SECONDARY_BODY_MAX and entry_hour != SATELLITE_SECONDARY_EXCLUDED_HOUR:
                    accepted_subtype = "mixed_transition_mild_strong_bear_body05_not13_market"
                    entry_mode_suffix = "tier5_satellite_body05_not13"
                    force_market = True
                    accepted_satellite_secondary += 1
                else:
                    if rel_volume < SATELLITE_MIN_REL_VOLUME:
                        skipped_satellite_low_rel_volume += 1
                    else:
                        skipped_satellite_secondary += 1
                    continue

            elif predicted_regime == EXPANSION_REGIME:
                predicted_other_non_target_candidates += 1
                if is_other_non_target_mild_strong_bear_expansion(
                    trigger_bar=trigger_bar,
                    market_ctx=market_ctx,
                    candidate_score=float(row["candidate_score"]),
                    predicted_regime=predicted_regime,
                    subtype=subtype,
                ):
                    accepted_subtype = "other_non_target_mild_strong_bear_v1_market"
                    entry_mode_suffix = "tier286_other_non_target"
                    force_market = True
                    accepted_other_non_target_expansion += 1
                else:
                    skipped_non_target += 1
                    continue

            else:
                skipped_non_target += 1
                continue

            failure_high = float(max(trigger_bar["high"], confirm_bar["high"]))
            stop_price_initial = failure_high * (1.0 + INITIAL_STOP_BUFFER_ABOVE_FAIL_HIGH_PCT / 100.0)

            if force_market:
                fill_idx = anchor_idx + 1
                if fill_idx >= len(execution_df):
                    skipped_unfilled += 1
                    continue
                order_placed_time = execution_df.iloc[fill_idx]["open_time"]
                raw_entry = float(execution_df.iloc[fill_idx]["open"])
                order_price = raw_entry
                entry_price = raw_entry * (1.0 + BINANCE_MARKET_ENTRY_SLIPPAGE_PCT / 100.0)
                entry_mode = f"binance_tier286_{accepted_subtype}"
                fee_entry_pct = BINANCE_TAKER_FEE_PCT
                market_entry_slippage_pct = BINANCE_MARKET_ENTRY_SLIPPAGE_PCT
            else:
                fill_idx, order_placed_time, order_price, entry_price = simulate_limit_retest_entry_binance(
                    df=execution_df,
                    anchor_idx=anchor_idx,
                    stop_price_initial=stop_price_initial,
                )
                entry_mode = f"binance_tier286_{accepted_subtype}"
                fee_entry_pct = BINANCE_MAKER_FEE_PCT
                market_entry_slippage_pct = BINANCE_LIMIT_FILL_SLIPPAGE_PCT
                if fill_idx is None or order_placed_time is None or entry_price is None:
                    fallback_idx = anchor_idx + 1
                    if fallback_idx >= len(execution_df):
                        skipped_unfilled += 1
                        continue
                    fill_idx = fallback_idx
                    order_placed_time = execution_df.iloc[fallback_idx]["open_time"]
                    raw_entry = float(execution_df.iloc[fallback_idx]["open"])
                    order_price = raw_entry
                    entry_price = raw_entry * (1.0 + BINANCE_MARKET_ENTRY_SLIPPAGE_PCT / 100.0)
                    entry_mode = f"{entry_mode}_market_fallback"
                    fee_entry_pct = BINANCE_TAKER_FEE_PCT
                    market_entry_slippage_pct = BINANCE_MARKET_ENTRY_SLIPPAGE_PCT
                    fallback_market_entries += 1

            stop_distance = stop_price_initial - entry_price
            if stop_distance <= 0:
                skipped_invalid_stop += 1
                continue

            risk_amount = balance * (RISK_PER_TRADE_PCT / 100.0)
            qty = risk_amount / stop_distance
            max_notional = balance * LEVERAGE
            desired_notional = qty * entry_price
            if desired_notional > max_notional:
                qty = max_notional / entry_price
            if qty <= 0:
                skipped_invalid_stop += 1
                continue

            notional = qty * entry_price
            fee_entry = notional * (fee_entry_pct / 100.0)

            trade, _ = run_trade_exit_short(
                df=execution_df,
                entry_idx=fill_idx,
                entry_price=float(entry_price),
                qty=float(qty),
                fee_entry=float(fee_entry),
                balance=float(balance),
                signal_time=candidate_time,
                trigger_time=trigger_bar["open_time"],
                confirm_time=confirm_bar["open_time"],
                order_placed_time=order_placed_time,
                trigger_bar=trigger_bar,
                confirm_bar=confirm_bar,
                entry_mode=entry_mode,
                row_context=trigger_bar,
                order_price=float(order_price) if order_price is not None else np.nan,
                stop_price_initial=float(stop_price_initial),
            )

            payload = asdict(trade)
            payload["exchange"] = "Binance"
            payload["symbol"] = "ETHUSDT"
            payload["trade_date"] = candidate_time.strftime("%Y-%m-%d")
            payload["candidate_time"] = candidate_time
            payload["candidate_score"] = float(row["candidate_score"])
            payload["candidate_price"] = float(row["close"])
            payload["market_trend_regime"] = market_ctx["trend_regime"]
            payload["market_vol_regime"] = market_ctx["vol_regime"]
            payload["ret_30d_pct"] = float(market_ctx["ret_30d_pct"]) if pd.notna(market_ctx["ret_30d_pct"]) else np.nan
            payload["ret_60d_pct"] = float(market_ctx["ret_60d_pct"]) if pd.notna(market_ctx["ret_60d_pct"]) else np.nan
            payload["accepted_subtype"] = accepted_subtype
            payload["regime_engine"] = entry_mode_suffix
            payload["entry_fee_pct"] = float(fee_entry_pct)
            payload["exit_fee_pct"] = float(BINANCE_TAKER_FEE_PCT)
            payload["market_entry_slippage_pct"] = float(market_entry_slippage_pct)
            trades.append(payload)

            balance = float(payload["balance_after"])
            last_exit_time = pd.to_datetime(payload["exit_time"], utc=True)

    trades_df = pd.DataFrame(trades)
    summary = {
        "exchange": "Binance",
        "symbol": "ETHUSDT",
        "deduped_candidate_count": int(len(deduped_candidates)),
        "predicted_core_candidates": int(predicted_core_candidates),
        "predicted_satellite_candidates": int(predicted_satellite_candidates),
        "predicted_other_non_target_candidates": int(predicted_other_non_target_candidates),
        "filled_trade_count": int(len(trades_df)),
        "accepted_strict_cooling": int(accepted_strict_cooling),
        "accepted_selective_cooling_market": int(accepted_selective_cooling_market),
        "accepted_satellite_rv08": int(accepted_satellite_rv08),
        "accepted_satellite_secondary": int(accepted_satellite_secondary),
        "accepted_other_non_target_expansion": int(accepted_other_non_target_expansion),
        "skipped_overlap": int(skipped_overlap),
        "skipped_wrong_market_regime": int(skipped_wrong_market_regime),
        "skipped_non_target": int(skipped_non_target),
        "skipped_wrong_subtype": int(skipped_wrong_subtype),
        "skipped_selective_cooling": int(skipped_selective_cooling),
        "skipped_satellite_filter": int(skipped_satellite_filter),
        "skipped_satellite_low_rel_volume": int(skipped_satellite_low_rel_volume),
        "skipped_satellite_secondary": int(skipped_satellite_secondary),
        "skipped_unfilled": int(skipped_unfilled),
        "skipped_invalid_stop": int(skipped_invalid_stop),
        "fallback_market_entries": int(fallback_market_entries),
        "binance_maker_fee_pct": float(BINANCE_MAKER_FEE_PCT),
        "binance_taker_fee_pct": float(BINANCE_TAKER_FEE_PCT),
        "binance_market_entry_slippage_pct": float(BINANCE_MARKET_ENTRY_SLIPPAGE_PCT),
        "binance_exit_slippage_pct": float(BINANCE_EXIT_SLIPPAGE_PCT),
        "ending_balance": float(balance),
    }

    if trades_df.empty:
        return trades_df, summary

    wins = trades_df.loc[trades_df["pnl_usd"] > 0, "pnl_usd"]
    losses = trades_df.loc[trades_df["pnl_usd"] < 0, "pnl_usd"]
    summary.update(
        {
            "win_rate_pct": float((trades_df["pnl_usd"] > 0).mean() * 100.0),
            "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan,
            "expectancy_usd": float(trades_df["pnl_usd"].mean()),
            "median_hold_minutes": float(trades_df["hold_minutes"].median()),
        }
    )
    return trades_df, summary


if __name__ == "__main__":
    print("This module is kept as a technical dependency for Tier 2.87.")
