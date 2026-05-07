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

from dotenv import load_dotenv

# ── façade imports ONLY (no L0-L8 imports allowed) ──────────────────────
from core_engine import (
    DecisionEngine,
    MarketAccountEngine,
    OperationsEngine,
    SafeExecutionEngine,
    SituationEngine,
)
from core_engine.integration import setup_core_engines
from core_engine.native.cadence_scheduler import CadenceScheduler

# ────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
)
log = logging.getLogger("octivault.main")

load_dotenv()


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
        log.info("✅ Clean shutdown complete")


def _is_real_execution_result(result: Any) -> bool:
    """Count only actual placed/filled execution outcomes as executions."""
    if not result:
        return False
    success = getattr(result, "success", None)
    if success is None and isinstance(result, dict):
        success = result.get("success")
    return bool(success)


# ════════════════════════════════════════════════════════════════════════
# Trading cycle — the canonical 5-phase pattern via façade engines
# ════════════════════════════════════════════════════════════════════════
async def trading_cycle(
    engines: Engines, mode: str, app_ctx: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    One full trading cycle via the 5 core engines. ONLY calls façade methods.

    PHASE 0: DISCOVER    — Scan wallet and update symbols (optional, per-cycle)
    PHASE 1: READ        — Fetch market data and account state
    PHASE 2: UNDERSTAND  — Analyze portfolio and market regime
    PHASE 3: DECIDE      — Generate trading decisions
    PHASE 4: EXECUTE     — Place orders safely
    PHASE 5: RECOVER     — Monitor health and log events
    """
    cycle_start = time.perf_counter()
    cadence = None
    now_ts = time.time()
    if app_ctx is not None:
        cadence = app_ctx.setdefault("_cadence_scheduler", CadenceScheduler())

    # ──────────────────────────────────────────────────────────────────
    # PHASE 0: DISCOVER (optional symbol discovery per cycle)
    # ──────────────────────────────────────────────────────────────────
    if app_ctx:
        native_orch = app_ctx.get("_native_orchestrator")
        if native_orch and hasattr(native_orch, "_symbol_discovery"):
            symbol_discovery = native_orch._symbol_discovery
            if symbol_discovery:
                try:
                    symbols = await symbol_discovery.discover()
                    if symbols and hasattr(native_orch, "_market_data"):
                        md = native_orch._market_data
                        if md and hasattr(md, "_symbols"):
                            current = md._symbols or []
                            if sorted(symbols) != sorted(current):
                                md._symbols = symbols
                                log.info(f"📱 Symbols discovered: {current} → {symbols}")
                except Exception as e:
                    log.debug(f"Phase 0 discovery failed: {e}")

    # ──────────────────────────────────────────────────────────────────
    # PHASE 1: READ
    # ──────────────────────────────────────────────────────────────────
    cached_account_state = (
        app_ctx.get("_cached_account_state") if app_ctx is not None else None
    ) or {
        "balances": {},
        "positions": {},
        "open_orders": [],
    }
    if cadence is None or cadence.is_due("account", now=now_ts):
        account_state = await engines.market.get_account_state()
        if app_ctx is not None:
            app_ctx["_cached_account_state"] = account_state
        if cadence is not None:
            cadence.mark("account", now=now_ts)
    else:
        account_state = cached_account_state
    market_prices = await engines.market.get_market_prices()
    read_complete = True

    # ──────────────────────────────────────────────────────────────────
    # PHASE 2: UNDERSTAND
    # ──────────────────────────────────────────────────────────────────
    scenario_due = cadence is None or cadence.is_due("scenario", now=now_ts)
    decision_due = cadence is None or cadence.is_due("decision", now=now_ts)
    cached_portfolio = (
        app_ctx.get("_cached_portfolio_snapshot") if app_ctx is not None else None
    ) or {}
    cached_regime = (app_ctx.get("_cached_market_regime") if app_ctx is not None else None) or {}
    cached_situation = (
        app_ctx.get("_cached_situation_state") if app_ctx is not None else None
    ) or {}
    cached_signals = (app_ctx.get("_cached_signals") if app_ctx is not None else None) or []

    if scenario_due or not cached_situation:
        portfolio_snapshot = await engines.situation.get_portfolio_snapshot()
        market_regime = await engines.situation.get_market_regime()
        situation_state = await engines.situation.get_situation_state()
        if app_ctx is not None:
            app_ctx["_cached_portfolio_snapshot"] = portfolio_snapshot
            app_ctx["_cached_market_regime"] = market_regime
            app_ctx["_cached_situation_state"] = situation_state
        if cadence is not None:
            cadence.mark("scenario", now=now_ts)
    else:
        portfolio_snapshot = cached_portfolio
        market_regime = cached_regime
        situation_state = cached_situation

    if decision_due or not cached_signals:
        all_signals = await engines.situation.get_all_signals()
        if app_ctx is not None:
            app_ctx["_cached_signals"] = all_signals
        if cadence is not None:
            cadence.mark("decision", now=now_ts)
    else:
        all_signals = cached_signals
    understand_complete = True

    # Helper: safely extract value from dict or object
    def get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # ──────────────────────────────────────────────────────────────────
    # PHASE 3: DECIDE
    # ──────────────────────────────────────────────────────────────────
    trading_decisions: list[Any] = []
    if decision_due:
        for sig in all_signals:
            decision = None
            sig_type = str(
                get_value(sig, "signal_type", get_value(sig, "action", "")) or ""
            ).upper()
            symbol = get_value(sig, "symbol", "")
            edge_score = float(get_value(sig, "edge_score", get_value(sig, "edge", 0.0)) or 0.0)

            try:
                if sig_type == "BUY":
                    decision = await engines.decision.make_buy_decision(symbol, edge_score)
                elif sig_type == "SELL":
                    decision = await engines.decision.make_sell_decision(
                        symbol, edge_score, "signal"
                    )
                if decision:
                    trading_decisions.append(decision)
            except Exception as e:
                log.debug(f"Decision failed for {symbol}: {e}")
    decide_complete = True

    # ──────────────────────────────────────────────────────────────────
    # PHASE 4: EXECUTE (only in non-dry-run modes)
    # ──────────────────────────────────────────────────────────────────
    executed_orders: list[Any] = []
    if mode != "dry-run":
        for decision in trading_decisions:
            order_result = await engines.execution.execute_decision(decision)
            if _is_real_execution_result(order_result):
                executed_orders.append(order_result)
    execute_complete = True

    # ──────────────────────────────────────────────────────────────────
    # PHASE 5: RECOVER / OBSERVE
    # ──────────────────────────────────────────────────────────────────
    cached_health = (app_ctx.get("_cached_health_report") if app_ctx is not None else None) or {}
    if cadence is None or cadence.is_due("health", now=now_ts):
        health_report = await engines.operations.get_health_report()
        if app_ctx is not None:
            app_ctx["_cached_health_report"] = health_report
        if cadence is not None:
            cadence.mark("health", now=now_ts)
    else:
        health_report = cached_health
    nav_val = get_value(portfolio_snapshot, "nav_usdt", 0.0)
    regime_val = get_value(market_regime, "overall_health", "unknown")
    primary_decision = trading_decisions[0] if trading_decisions else None
    primary_execution = executed_orders[0] if executed_orders else None
    situation_metrics = get_value(situation_state, "metrics", {}) or {}
    quant_summary = {
        "timestamp": time.time(),
        "nav_usdt": float(situation_metrics.get("nav_usdt", nav_val) or 0.0),
        "free_usdt": float(situation_metrics.get("free_usdt", 0.0) or 0.0),
        "free_ratio": float(situation_metrics.get("free_ratio", 0.0) or 0.0),
        "exposure_ratio": float(situation_metrics.get("exposure_ratio", 0.0) or 0.0),
        "market_regime": get_value(situation_state, "market_regime", "UNKNOWN"),
        "portfolio_state": get_value(situation_state, "portfolio_state", "BALANCED"),
        "capital_state": get_value(situation_state, "capital_state", "HEALTHY"),
        "risk_state": get_value(situation_state, "risk_state", "NORMAL"),
        "system_state": get_value(situation_state, "system_state", "HEALTHY"),
        "playbook": get_value(primary_decision, "playbook", ""),
        "action": get_value(primary_decision, "action", "NONE"),
        "symbol": get_value(primary_decision, "symbol", ""),
        "probability_score": float(get_value(primary_decision, "probability_score", 0.0) or 0.0),
        "confidence": float(get_value(primary_decision, "confidence", 0.0) or 0.0),
        "edge_score": float(get_value(primary_decision, "edge_score", 0.0) or 0.0),
        "allowed": bool(get_value(primary_decision, "allowed", False)),
        "blocked_reason": get_value(primary_decision, "blocked_reason", ""),
        "execution_result": get_value(primary_execution, "status", "NONE"),
        "loop_duration_ms": (time.perf_counter() - cycle_start) * 1000,
    }
    summary_due = cadence is None or cadence.is_due("summary", now=now_ts)
    should_emit_summary = summary_due or bool(trading_decisions) or bool(executed_orders)
    # Extract health status (handle both dict and object)
    if isinstance(health_report, dict):
        health_status_val = health_report.get("overall_status", "UNKNOWN")
    elif hasattr(health_report, "overall_status"):
        health_status_val = (
            health_report.overall_status.value
            if hasattr(health_report.overall_status, "value")
            else str(health_report.overall_status)
        )
    else:
        health_status_val = "UNKNOWN"
    await engines.operations.log_event(
        "cycle_complete",
        {
            "read_ok": read_complete,
            "understand_ok": understand_complete,
            "decide_ok": decide_complete,
            "execute_ok": execute_complete,
            "num_signals": len(all_signals),
            "num_decisions": len(trading_decisions),
            "num_executed": len(executed_orders),
            "nav_usdt": nav_val,
            "market_regime": regime_val,
        },
    )
    if should_emit_summary:
        await engines.operations.log_event("QUANT_LOOP_SUMMARY", quant_summary)
        if app_ctx is not None:
            app_ctx["_last_summary_health_status"] = health_status_val
        if cadence is not None:
            cadence.mark("summary", now=now_ts)
    recover_complete = True

    # Return cycle telemetry
    return {
        "duration_ms": (time.perf_counter() - cycle_start) * 1000,
        "num_prices": len(market_prices),
        "num_balances": len(account_state.get("balances", {}))
        if isinstance(account_state, dict)
        else len(getattr(account_state, "balances", {})),
        "nav_usdt": nav_val,
        "num_signals": len(all_signals),
        "num_decisions": len(trading_decisions),
        "num_executed": len(executed_orders),
        "health_status": health_status_val,
        "read_phase_ok": read_complete,
        "understand_phase_ok": understand_complete,
        "decide_phase_ok": decide_complete,
        "execute_phase_ok": execute_complete,
        "recover_phase_ok": recover_complete,
    }


# ════════════════════════════════════════════════════════════════════════
# Run loop
# ════════════════════════════════════════════════════════════════════════
async def run(args: argparse.Namespace) -> int:
    native = not getattr(args, "no_native", False)
    compat = native and not getattr(args, "no_compat", False)

    log.info("=" * 72)
    log.info("OctiVault Trading Bot — Façade Entry Point (Step 3)")
    log.info(
        "Mode=%s  duration=%s  capital=%s  cycles=%s  native=%s  compat=%s",
        args.mode,
        args.duration,
        args.capital,
        args.cycles,
        native,
        compat,
    )
    log.info("=" * 72)

    # Build app_ctx and wire the 5 engines (only call into core_engine)
    app_ctx = await setup_core_engines(native=native, compat=compat)
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
                telem = await trading_cycle(engines, args.mode, app_ctx)
                phases = "".join(
                    [
                        "R" if telem["read_phase_ok"] else "✗",
                        "U" if telem["understand_phase_ok"] else "✗",
                        "D" if telem["decide_phase_ok"] else "✗",
                        "E" if telem["execute_phase_ok"] else "✗",
                        "O" if telem["recover_phase_ok"] else "✗",
                    ]
                )
                log.info(
                    "cycle %05d │ %6.1fms │ nav=%9.2f │ sigs=%2d │ dec=%2d │ exe=%2d │ [%s] │ %s",
                    cycle_no,
                    telem["duration_ms"],
                    telem["nav_usdt"],
                    telem["num_signals"],
                    telem["num_decisions"],
                    telem["num_executed"],
                    phases,
                    telem["health_status"],
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
        # Phase 8.3.1: tear down native bootstrap (background poll
        # loops, exchange-client HTTP session). No-op when running in
        # mock mode (--no-native), since _native_components is absent.
        native_components = app_ctx.get("_native_components")
        if native_components is not None:
            try:
                from core_engine.native.bootstrap import shutdown_components

                await shutdown_components(native_components)
                log.info("✅ Native bootstrap shut down")
            except Exception as e:
                log.warning("native shutdown error: %s", e)

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
        default="live",
        help="Execution mode (default: live)",
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
    p.add_argument(
        "--no-native",
        action="store_true",
        help="Skip native L0-L8 bootstrap; use empty mock app_ctx "
        "(graceful-degrade everywhere). Default: native is ON.",
    )
    p.add_argument(
        "--no-compat",
        action="store_true",
        help="Skip compat null-stubs for the 6 unmigrated façade keys. "
        "Default: compat is ON when native is ON.",
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
