#!/usr/bin/env python3
"""
Native path smoke test (Phase 8.2.8, step 5).

End-to-end exerciser for the native L0-L8 stack via the production
entry point. Validates the same code path that ``main`` will use once
``production_bridge.py`` is deleted.

Modes
-----
* ``--offline`` (default): substitutes a stub exchange client. Verifies
  bootstrap → app_ctx → orchestrator → telemetry without touching the
  network. Suitable for CI and pre-credential validation.
* ``--live``: uses the real ``NativeExchangeClient``. Requires
  ``BINANCE_API_KEY`` and ``BINANCE_API_SECRET`` (testnet creds with
  ``BINANCE_TESTNET=true`` strongly recommended).

Usage
-----
::

    # offline (no creds needed):
    python3 scripts/native_smoke.py --offline --duration 10

    # live testnet:
    BINANCE_API_KEY=... BINANCE_API_SECRET=... BINANCE_TESTNET=true \\
        python3 scripts/native_smoke.py --live --duration 60

Exit codes
----------
* 0 — completed at least one cycle, no unhandled exceptions.
* 1 — bootstrap or runtime failure.
* 2 — completed but ran zero cycles (suspicious).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so ``core_engine`` resolves when
# this script is invoked directly (no ``pip install -e .``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Quiet down noisy libs unless verbose
    if not verbose:
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


# ---------------------------------------------------------------------
# Stub exchange client for --offline
# ---------------------------------------------------------------------
class _OfflineExchangeClient:
    """
    Minimal duck-typed stand-in for ``NativeExchangeClient`` used in
    ``--offline`` mode. Only implements what L1/L2 actually call.
    """

    def __init__(self) -> None:
        self._closed = False

    async def get_klines(self, symbol: str, *_a: Any, **_kw: Any) -> list:
        # Stable synthetic OHLCV; enough to exercise signal evaluation
        return [[0, "100", "101", "99", "100", "10"] for _ in range(60)]

    async def get_prices(self, *_a: Any, **_kw: Any) -> dict:
        return {
            "BTCUSDT": 50_000.0,
            "ETHUSDT": 3_000.0,
            "BNBUSDT": 400.0,
            "SOLUSDT": 100.0,
            "XRPUSDT": 0.5,
        }

    async def get_balance(self, *_a: Any, **_kw: Any) -> dict:
        return {"USDT": 1_000.0, "BTC": 0.0}

    async def place_order(self, *_a: Any, **_kw: Any) -> dict:
        return {"orderId": "offline-smoke", "status": "FILLED"}

    async def cancel_order(self, *_a: Any, **_kw: Any) -> dict:
        return {"status": "CANCELED"}

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------
# Smoke run
# ---------------------------------------------------------------------
async def _smoke(
    *,
    offline: bool,
    duration_sec: float,
    max_cycles: int | None,
    compat: bool,
) -> int:
    from core_engine.native.app_context import build_native_app_ctx
    from core_engine.native.bootstrap import (
        BootstrapConfig,
        build_components,
        shutdown_components,
    )

    log = logging.getLogger("native_smoke")

    if offline:
        # Provide synthetic creds so BootstrapConfig.from_env() doesn't
        # raise; the offline factory ignores them anyway.
        os.environ.setdefault("BINANCE_API_KEY", "offline-smoke-key")
        os.environ.setdefault("BINANCE_API_SECRET", "offline-smoke-secret")
        os.environ.setdefault("SYMBOLS", "BTCUSDT,ETHUSDT")
        factory = lambda _cfg: _OfflineExchangeClient()  # noqa: E731
        log.info("mode=offline (stub exchange client)")
    else:
        if not (os.environ.get("BINANCE_API_KEY") and os.environ.get("BINANCE_API_SECRET")):
            log.error("--live requires BINANCE_API_KEY and BINANCE_API_SECRET")
            return 1
        factory = None
        testnet = os.environ.get("BINANCE_TESTNET", "").lower() in ("true", "1", "yes")
        log.info("mode=live (testnet=%s)", testnet)
        if not testnet:
            log.warning("BINANCE_TESTNET is not set — running against MAINNET. Ctrl-C to abort.")

    try:
        cfg = BootstrapConfig.from_env()
    except ValueError as e:
        log.error("config error: %s", e)
        return 1

    log.info(
        "bootstrap: testnet=%s symbols=%d duration=%.1fs max_cycles=%s compat=%s",
        cfg.testnet,
        len(cfg.symbols),
        duration_sec,
        max_cycles,
        compat,
    )

    components = await build_components(cfg, exchange_client_factory=factory)
    app_ctx, orch = build_native_app_ctx(components, compat=compat)
    log.info("app_ctx keys: %s", sorted(app_ctx.keys()))

    try:
        metrics = await orch.run_loop(
            duration_sec=duration_sec,
            max_cycles=max_cycles,
        )
    except Exception:
        log.exception("orchestrator failure")
        await shutdown_components(components)
        return 1

    await shutdown_components(components)

    if not metrics:
        log.error("zero cycles completed")
        return 2

    # Telemetry summary
    tel = components.telemetry
    if tel is not None and len(tel) > 0:
        summary = tel.summary()
        breakdown = tel.phase_breakdown()
        log.info(
            "completed: cycles=%d successes=%d failures=%d errors=%d "
            "avg=%.2fms p50=%.2fms p95=%.2fms max=%.2fms",
            summary["count"],
            summary["total_successes"],
            summary["total_failures"],
            summary["total_errors"],
            summary["avg_duration_ms"],
            summary["p50_duration_ms"],
            summary["p95_duration_ms"],
            summary["max_duration_ms"],
        )
        log.info(
            "rates: success=%.2f%% error=%.2f%% latest_nav=%.2f",
            summary["success_rate"] * 100.0,
            summary["error_rate"] * 100.0,
            summary["latest_nav"],
        )
        for phase, avg_ms in breakdown.items():
            log.info("  phase %-12s avg=%6.2fms", phase, avg_ms)
    else:
        log.info("completed: %d cycles (no telemetry)", len(metrics))

    return 0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use stub exchange client (default)")
    mode.add_argument(
        "--live", action="store_true", help="use real NativeExchangeClient (needs creds)"
    )
    p.add_argument("--duration", type=float, default=10.0, help="seconds to run (default: 10)")
    p.add_argument("--max-cycles", type=int, default=None, help="stop after N cycles")
    p.add_argument(
        "--no-compat",
        action="store_true",
        help="deprecated no-op (G5, 8.3.12); kept for CLI compat",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    offline = not args.live  # default offline; --live opts in
    return asyncio.run(
        _smoke(
            offline=offline,
            duration_sec=args.duration,
            max_cycles=args.max_cycles,
            compat=not args.no_compat,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
