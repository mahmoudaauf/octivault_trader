"""
main.py — OctiVault Trading Bot Entry Point (Step 3)
═══════════════════════════════════════════════════════

ARCHITECTURAL CONTRACT
──────────────────────
This module talks ONLY to the 5 core engines. It MUST NOT import any
L0-L8 component directly. All access to the 145+ underlying modules
flows through the façade layer in `core_engine/`.

    BEFORE:  main → 145 modules indirectly  (god-object orchestrator)
    AFTER:   main → 5 engines               (clean façade)

         ┌────────────────────────┐
         │        main.py         │
         └───────────┬────────────┘
                     │
         ┌───────────┴───────────────────────────────┐
         │           5 ENGINES (façade only)         │
         ├───────────────────────────────────────────┤
         │ 1. MarketAccountEngine    — READ          │
         │ 2. SituationEngine        — UNDERSTAND    │
         │ 3. DecisionEngine         — DECIDE        │
         │ 4. SafeExecutionEngine    — EXECUTE       │
         │ 5. OperationsEngine       — RECOVER       │
         └───────────┬───────────────────────────────┘
                     │ (encapsulated)
                     ▼
              L0-L8 components (145+ modules)

ENFORCEMENT
───────────
Linter rule: any `import` outside `core_engine` or stdlib in this file
is a violation of the façade contract. See `STEP_3_FACADE_CONTRACT.md`.

Usage
─────
    python3 main.py --mode=paper-trade --duration=24h
    python3 main.py --mode=live --capital=1000
    python3 main.py --mode=dry-run --cycles=10
"""

from __future__ import annotations

# ── stdlib only ─────────────────────────────────────────────────────────
import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time
from typing import Any

# ── façade imports ONLY (no L0-L8 imports allowed) ──────────────────────
from core_engine import (
    DecisionEngine,
    MarketAccountEngine,
    OperationsEngine,
    SafeExecutionEngine,
    SituationEngine,
)
from core_engine.integration import setup_core_engines

# ────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
)
log = logging.getLogger("octivault.main")


# ════════════════════════════════════════════════════════════════════════
# Engine container
# ════════════════════════════════════════════════════════════════════════
class Engines:
    """Holds the 5 engines. Nothing else lives here."""

    def __init__(self, app_ctx: dict[str, Any]) -> None:
        self.market = MarketAccountEngine(app_ctx)
        self.situation = SituationEngine(app_ctx)
        self.decision = DecisionEngine(app_ctx)
        self.execution = SafeExecutionEngine(app_ctx)
        self.operations = OperationsEngine(app_ctx)

    async def initialize(self) -> None:
        """Boot all engines (operations starts the underlying system)."""
        log.info("🚀 Initializing 5 core engines…")
        await self.operations.startup_system()
        await self.market.initialize()
        await self.situation.initialize()
        await self.decision.initialize()
        await self.execution.initialize()
        await self.operations.initialize()
        log.info("✅ All 5 engines online")

    async def shutdown(self) -> None:
        """Reverse-order shutdown."""
        log.info("⏹  Shutting down 5 core engines…")
        for eng in (
            self.execution,
            self.decision,
            self.situation,
            self.market,
            self.operations,
        ):
            try:
                await eng.shutdown()
            except Exception as e:
                log.warning("shutdown error in %s: %s", eng.__class__.__name__, e)
        await self.operations.shutdown_system()
        log.info("✅ Clean shutdown complete")


# ════════════════════════════════════════════════════════════════════════
# Trading cycle — the canonical READ→UNDERSTAND→DECIDE→EXECUTE→RECOVER
# ════════════════════════════════════════════════════════════════════════
async def trading_cycle(engines: Engines, mode: str) -> dict[str, Any]:
    """
    One full trading cycle. ONLY calls the 5 engines — never anything else.
    Returns a dict of cycle telemetry.
    """
    cycle_start = time.perf_counter()

    # 1. READ ────────────────────────────────────────────────────────────
    account = await engines.market.get_account_state()
    prices = await engines.market.get_market_prices()

    # 2. UNDERSTAND ──────────────────────────────────────────────────────
    portfolio = await engines.situation.get_portfolio_snapshot()
    regime = await engines.situation.get_market_regime()
    signals = await engines.situation.get_all_signals()

    # 3. DECIDE ──────────────────────────────────────────────────────────
    decisions: list[Any] = []
    for sig in signals:
        if sig.signal_type == "BUY":
            d = await engines.decision.make_buy_decision(sig.symbol, sig.edge_score)
        elif sig.signal_type == "SELL":
            d = await engines.decision.make_sell_decision(
                sig.symbol, sig.edge_score, reason="signal"
            )
        else:
            d = None
        if d is not None:
            decisions.append(d)

    # 4. EXECUTE ─────────────────────────────────────────────────────────
    executed: list[Any] = []
    if mode != "dry-run":
        for d in decisions:
            if d.action == "BUY":
                r = await engines.execution.place_buy_order(
                    symbol=d.symbol, quantity=d.quantity, price=d.price_target
                )
            elif d.action == "SELL":
                r = await engines.execution.place_sell_order(
                    symbol=d.symbol, quantity=d.quantity, price=d.price_target
                )
            else:
                r = None
            if r is not None:
                executed.append(r)

    # 5. RECOVER / OBSERVE ───────────────────────────────────────────────
    health = await engines.operations.get_health_report()
    await engines.operations.log_event(
        "cycle_complete",
        {
            "decisions": len(decisions),
            "executed": len(executed),
            "regime": regime.overall_health,
        },
    )

    return {
        "duration_ms": (time.perf_counter() - cycle_start) * 1000,
        "prices": len(prices),
        "balances": len(account.get("balances", {})),
        "nav": portfolio.nav_usdt,
        "signals": len(signals),
        "decisions": len(decisions),
        "executed": len(executed),
        "health": health.overall_status.value
        if hasattr(health.overall_status, "value")
        else str(health.overall_status),
    }


# ════════════════════════════════════════════════════════════════════════
# Run loop
# ════════════════════════════════════════════════════════════════════════
async def run(args: argparse.Namespace) -> int:
    log.info("=" * 72)
    log.info("OctiVault Trading Bot — Façade Entry Point (Step 3)")
    log.info(
        "Mode=%s  duration=%s  capital=%s  cycles=%s",
        args.mode,
        args.duration,
        args.capital,
        args.cycles,
    )
    log.info("=" * 72)

    # Build app_ctx and wire the 5 engines (only call into core_engine)
    app_ctx = await setup_core_engines()
    app_ctx["mode"] = args.mode
    app_ctx["initial_capital"] = args.capital

    engines = Engines(app_ctx)
    await engines.initialize()

    # Graceful shutdown wiring
    stop_event = asyncio.Event()

    def _signal_handler(*_: Any) -> None:
        log.warning("⚠️  Signal received — requesting graceful shutdown")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, _signal_handler)
        # NotImplementedError/RuntimeError on Windows / non-main thread

    # Cycle budget
    deadline = (time.time() + parse_duration(args.duration)) if args.duration else None
    max_cycles = args.cycles if args.cycles > 0 else None

    cycle_no = 0
    rc = 0
    try:
        while not stop_event.is_set():
            if deadline and time.time() >= deadline:
                log.info("⏰ Duration reached — exiting loop")
                break
            if max_cycles and cycle_no >= max_cycles:
                log.info("🔢 Cycle budget exhausted — exiting loop")
                break

            cycle_no += 1
            try:
                telem = await trading_cycle(engines, args.mode)
                log.info(
                    "cycle %05d │ %5.1fms │ nav=%.2f │ sigs=%d │ dec=%d │ exec=%d │ %s",
                    cycle_no,
                    telem["duration_ms"],
                    telem["nav"],
                    telem["signals"],
                    telem["decisions"],
                    telem["executed"],
                    telem["health"],
                )
            except Exception as e:
                log.exception("cycle %d failed: %s", cycle_no, e)
                # Let OperationsEngine decide whether we recover
                plan = await engines.operations.recover_state()
                if not plan.auto_recover:
                    log.error("❌ Operations engine declined auto-recovery — aborting")
                    rc = 2
                    break
                await engines.operations.apply_recovery(plan)

            await asyncio.sleep(args.interval)

    finally:
        await engines.shutdown()

    log.info("Total cycles: %d", cycle_no)
    return rc


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
def parse_duration(s: str | None) -> float:
    """Accept '30min', '2h', '24h', '90s', or a bare number of seconds."""
    if not s:
        return 0.0
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("min") or s.endswith("m"):
        n = s[:-3] if s.endswith("min") else s[:-1]
        return float(n) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="octivault",
        description="OctiVault Trading Bot — façade entry point (talks to 5 engines only)",
    )
    p.add_argument(
        "--mode",
        choices=("dry-run", "paper-trade", "live"),
        default="paper-trade",
        help="Execution mode (default: paper-trade)",
    )
    p.add_argument(
        "--duration",
        default=None,
        help="Wall-clock budget, e.g. 30min, 2h, 24h. Omit for unlimited.",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Cycle budget (0 = unlimited).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between cycles (default: 1.0).",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital in USDT (default: 1000).",
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
