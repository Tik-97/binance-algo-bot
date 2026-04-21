from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eth_unified_live_bot.bot.router import UnifiedRouter
from eth_unified_live_bot.bot.strategies.long_f21 import LongF21Strategy
from eth_unified_live_bot.bot.strategies.long_t14 import LongT14Strategy
from eth_unified_live_bot.bot.strategies.short_t287 import ShortT287Strategy


CANDLES_1M_PATH = REPO_ROOT / ".cache" / "binance_klines" / "ETHUSDT_1m.csv"
OUTPUT_DIR = REPO_ROOT / "combined_results" / "eth_unified_live_bot_combined_demo_replay_check_v1"
BOOTSTRAP_1M = 1500


def _load_candles() -> pd.DataFrame:
    frame = pd.read_csv(CANDLES_1M_PATH)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close", "volume"])
    return frame.sort_values("open_time").reset_index(drop=True)


def _build_daily(candles_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        candles_1m.set_index("open_time")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candles = _load_candles()
    daily = _build_daily(candles)

    bootstrap_1m = candles.iloc[:BOOTSTRAP_1M].copy()
    bootstrap_daily = daily.loc[daily["open_time"] <= bootstrap_1m.iloc[-1]["open_time"].floor("1D")].copy()

    strategies = [
        ShortT287Strategy(symbol="ETHUSDT", session_start_hour_utc=13, session_end_hour_utc=16),
        LongT14Strategy(symbol="ETHUSDT", session_start_hour_utc=0, session_end_hour_utc=24),
        LongF21Strategy(symbol="ETHUSDT", session_start_hour_utc=0, session_end_hour_utc=24),
    ]
    for strategy in strategies:
        strategy.bootstrap(bootstrap_1m, bootstrap_daily)

    router = UnifiedRouter(strategies)
    individual_accepts = Counter()
    individual_routes = Counter()
    router_accepts = Counter()
    router_routes = Counter()
    individual_examples: dict[str, list[str]] = defaultdict(list)
    router_examples: dict[str, list[str]] = defaultdict(list)

    for _, row in candles.iloc[BOOTSTRAP_1M:].iterrows():
        candle = {
            "open_time": row["open_time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for strategy in strategies:
            strategy.ingest_candle(candle)

        accepted = router.evaluate(has_open_position=False, has_active_signal=False)
        for strategy_name, decision in router.last_decisions.items():
            if decision.status == "accepted":
                individual_accepts[strategy_name] += 1
                individual_routes[f"{strategy_name}:{decision.route}"] += 1
                if len(individual_examples[strategy_name]) < 5:
                    individual_examples[strategy_name].append(
                        f"{decision.candle_time.isoformat()} | {decision.route}"
                    )
        if accepted is not None and accepted.status == "accepted":
            router_accepts[accepted.strategy_name] += 1
            router_routes[f"{accepted.strategy_name}:{accepted.route}"] += 1
            if len(router_examples[accepted.strategy_name]) < 5:
                router_examples[accepted.strategy_name].append(
                    f"{accepted.candle_time.isoformat()} | {accepted.route}"
                )

    lines = [
        "ETH UNIFIED LIVE BOT COMBINED DEMO REPLAY CHECK V1",
        "==================================================",
        "",
        f"candles_path = {CANDLES_1M_PATH}",
        f"bootstrap_1m = {BOOTSTRAP_1M}",
        "router_sessions = short_t287 13-16 UTC, long_t14 0-24 UTC, long_f21 0-24 UTC",
        "",
        "Individual strategy accepts:",
    ]
    for name in ["short_t287", "long_t14", "long_f21"]:
        lines.append(f"- {name}: {individual_accepts.get(name, 0)}")
    lines.append("")
    lines.append("Individual route counts:")
    for key, count in sorted(individual_routes.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("Router accepted counts:")
    for name in ["short_t287", "long_t14", "long_f21"]:
        lines.append(f"- {name}: {router_accepts.get(name, 0)}")
    lines.append("")
    lines.append("Router route counts:")
    for key, count in sorted(router_routes.items()):
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("Example accepted candles per strategy:")
    for name in ["short_t287", "long_t14", "long_f21"]:
        examples = individual_examples.get(name, [])
        if examples:
            lines.append(f"- {name}:")
            for example in examples:
                lines.append(f"  {example}")
        else:
            lines.append(f"- {name}: none")
    lines.append("")
    lines.append("Example router-accepted candles:")
    for name in ["short_t287", "long_t14", "long_f21"]:
        examples = router_examples.get(name, [])
        if examples:
            lines.append(f"- {name}:")
            for example in examples:
                lines.append(f"  {example}")
        else:
            lines.append(f"- {name}: none")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.DataFrame(
        [{"strategy_name": name, "accepted_count": individual_accepts.get(name, 0)} for name in ["short_t287", "long_t14", "long_f21"]]
    ).to_csv(OUTPUT_DIR / "individual_strategy_counts.csv", index=False)
    pd.DataFrame(
        [{"strategy_name": name, "accepted_count": router_accepts.get(name, 0)} for name in ["short_t287", "long_t14", "long_f21"]]
    ).to_csv(OUTPUT_DIR / "router_strategy_counts.csv", index=False)
    pd.DataFrame(
        [{"route_key": key, "accepted_count": count} for key, count in sorted(individual_routes.items())]
    ).to_csv(OUTPUT_DIR / "individual_route_counts.csv", index=False)
    pd.DataFrame(
        [{"route_key": key, "accepted_count": count} for key, count in sorted(router_routes.items())]
    ).to_csv(OUTPUT_DIR / "router_route_counts.csv", index=False)
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
