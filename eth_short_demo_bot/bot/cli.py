from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal

from .config import documented_config_format, load_config
from .logging_setup import setup_logging
from .state import StateStore


async def _build_runtime(config_path: str | None = None):
    from .exchange import BinanceFuturesClient
    from .notifier import TelegramNotifier
    from .service import ETHShortDemoBotService

    config = load_config(config_path)
    setup_logging(config.log_dir)
    store = StateStore(config.state_db_path)
    exchange = BinanceFuturesClient(config.binance)
    notifier = TelegramNotifier(config.telegram, store)
    service = ETHShortDemoBotService(config=config, store=store, exchange=exchange, notifier=notifier)
    return service, notifier, exchange


async def run_bot(config_path: str | None = None) -> None:
    service, notifier, exchange = await _build_runtime(config_path)
    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    runner = asyncio.create_task(service.run_forever())
    waiter = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if waiter in done and not runner.done():
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
    await notifier.aclose()
    await exchange.aclose()


async def run_smoke_market(config_path: str | None, test_notional_usd: float, stop_pct: float, tp_pct: float) -> None:
    service, notifier, exchange = await _build_runtime(config_path)
    await service.startup()
    result = await service.smoke_market_bracket(test_notional_usd, stop_pct, tp_pct)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    await service.shutdown()
    await notifier.aclose()
    await exchange.aclose()


async def run_smoke_limit(
    config_path: str | None,
    test_notional_usd: float,
    entry_offset_pct: float,
    stop_pct: float,
    tp_pct: float,
    wait_seconds: int,
) -> None:
    service, notifier, exchange = await _build_runtime(config_path)
    await service.startup()
    result = await service.smoke_limit_bracket(test_notional_usd, entry_offset_pct, stop_pct, tp_pct, wait_seconds=wait_seconds)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    await service.shutdown()
    await notifier.aclose()
    await exchange.aclose()


async def run_close_all(config_path: str | None) -> None:
    service, notifier, exchange = await _build_runtime(config_path)
    await service.startup()
    await service.close_all_open_risk()
    print("Closed all open orders and live short exposure if present.")
    await service.shutdown()
    await notifier.aclose()
    await exchange.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ETH short Binance Futures DEMO bot")
    parser.add_argument("--config", help="Path to ETH Short Demo.txt", default=None)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Run the live demo bot")
    sub.add_parser("config-format", help="Print expected config format")
    sub.add_parser("close-all", help="Cancel open orders and close live short if present")

    smoke_market = sub.add_parser("smoke-market", help="Open demo market short + SL/TP bracket")
    smoke_market.add_argument("--test-notional-usd", type=float, default=50.0)
    smoke_market.add_argument("--stop-pct", type=float, default=0.35)
    smoke_market.add_argument("--tp-pct", type=float, default=0.70)

    smoke_limit = sub.add_parser("smoke-limit", help="Open demo limit short + SL/TP bracket")
    smoke_limit.add_argument("--test-notional-usd", type=float, default=50.0)
    smoke_limit.add_argument("--entry-offset-pct", type=float, default=0.20)
    smoke_limit.add_argument("--stop-pct", type=float, default=0.35)
    smoke_limit.add_argument("--tp-pct", type=float, default=0.70)
    smoke_limit.add_argument("--wait-seconds", type=int, default=120)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "config-format":
        print(documented_config_format())
        return

    if args.command == "run":
        asyncio.run(run_bot(args.config))
        return

    if args.command == "smoke-market":
        asyncio.run(run_smoke_market(args.config, args.test_notional_usd, args.stop_pct, args.tp_pct))
        return

    if args.command == "smoke-limit":
        asyncio.run(
            run_smoke_limit(
                args.config,
                args.test_notional_usd,
                args.entry_offset_pct,
                args.stop_pct,
                args.tp_pct,
                args.wait_seconds,
            )
        )
        return

    if args.command == "close-all":
        asyncio.run(run_close_all(args.config))
        return


if __name__ == "__main__":
    main()
