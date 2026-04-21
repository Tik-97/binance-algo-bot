from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from next_phase_edge_focus import binance_tier286_realistic_backtest_v1 as prev


REFINED_EXPANSION_MAX_REL_VOLUME = 1.50
REFINED_EXPANSION_MAX_MOVE_5M_PCT = 0.55
ORIGINAL_EXPANSION_FN = prev.is_other_non_target_mild_strong_bear_expansion


def is_other_non_target_mild_strong_bear_expansion_refined(
    trigger_bar: pd.Series,
    market_ctx: dict[str, object],
    candidate_score: float,
    predicted_regime: str,
    subtype: str,
) -> bool:
    return (
        ORIGINAL_EXPANSION_FN(
            trigger_bar=trigger_bar,
            market_ctx=market_ctx,
            candidate_score=candidate_score,
            predicted_regime=predicted_regime,
            subtype=subtype,
        )
        and float(trigger_bar["relative_volume_1m"]) <= REFINED_EXPANSION_MAX_REL_VOLUME
        and float(trigger_bar["move_5m_pct"]) <= REFINED_EXPANSION_MAX_MOVE_5M_PCT
    )


if __name__ == "__main__":
    print("This module is kept as a technical dependency for Tier 2.87.")
