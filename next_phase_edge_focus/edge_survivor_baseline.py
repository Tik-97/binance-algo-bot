from pathlib import Path
import runpy


# Current main Binance baseline.
# Fixed baseline as of 2026-04-18:
# Tier 2.87 realistic
#
# Why this baseline:
# - ETH only
# - short only
# - bearish regime only
# - Tier 2.85 core preserved
# - plus refined other_non_target strong_bear expansion
# - conservative Binance execution assumptions
# - survived the ETH/Binance realistic stress package

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET = BASE_DIR / "next_phase_edge_focus" / "binance_tier287_realistic_backtest_v1.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
