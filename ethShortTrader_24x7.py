from pathlib import Path

import ethShortTrader as base


# Research variant: disable the UTC session gate and let the same model trade 24/7.
base.OUTPUT_DIR = base.BASE_DIR / "fun_backtests" / "eth_short_trader_style_24x7_v1"
base.SESSION_START_HOUR_UTC = 0
base.SESSION_END_HOUR_UTC = 24


if __name__ == "__main__":
    base.main()
