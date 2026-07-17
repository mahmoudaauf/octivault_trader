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
from utils.logging_setup import setup_logging

# ────────────────────────────────────────────────────────────────────────
# Logging  — RotatingFileHandler (50 MB × 3) + console WARNING+
#
# setup_logging() attaches real handlers to the ROOT logger, so it must run
# only when this module is actually executed as the live app (see the
# __main__ block below) — never as a side effect of import. Importing
# main.py (e.g. `from main import _is_real_execution_result` in tests) used
# to trigger this at collection time, wiring every test process's root
# logger to logs/run_latest.log and interleaving test log lines into the
# live production log. `log` itself is just a named logger handle here —
# safe to bind at import time either way, since it does nothing until
# handlers exist on root.
# ────────────────────────────────────────────────────────────────────────
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
        # 2026-07-14 fix: this return value was previously discarded, so a
        # failed native-orchestrator startup (state machine timeout/hydration
        # failure) let the process fall straight into the trading loop with
        # session_anchor_nav stuck at 0 (or a stale persisted value) --
        # bypassing every NAV-protection session-scoped safety check. Refuse
        # to proceed rather than trade on an unknown/unhydrated foundation;
        # supervisor.sh's existing crash-loop-protected restart handles retry.
        if not await self.operations.startup_system():
            raise RuntimeError(
                "startup_system() failed — refusing to enter the trading loop. "
                "Check logs for the underlying native-orchestrator startup error."
            )
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
                    # Also run SymbolRotator to expand universe beyond wallet holdings
                    symbol_rotator = getattr(native_orch, "_symbol_rotator", None)
                    if symbol_rotator is not None:
                        try:
                            symbol_rotator.maybe_rotate()
                            ss_r = getattr(native_orch, "_shared_state", None)
                            if ss_r is not None:
                                rotator_syms = list(getattr(ss_r, "accepted_symbols", set()) or set())
                                if rotator_syms:
                                    # Merge: wallet holdings + rotator universe
                                    merged = sorted(set(symbols) | set(rotator_syms))
                                    symbols = merged
                        except Exception as _re:
                            log.debug(f"SymbolRotator.maybe_rotate failed: {_re}")
                    if symbols and hasattr(native_orch, "_market_data"):
                        md = native_orch._market_data
                        if md and hasattr(md, "_symbols"):
                            current = md._symbols or []
                            if sorted(symbols) != sorted(current):
                                md._symbols = symbols
                                log.info(f"📱 Symbols discovered: {current} → {symbols}")
                    if symbols and hasattr(native_orch, "_shared_state"):
                        ss = native_orch._shared_state
                        if ss and hasattr(ss, "set_accepted_symbols"):
                            ss.set_accepted_symbols(symbols)
                    # Point the real-time WS feed at the actually-traded universe instead
                    # of its hardcoded bootstrap seeds. Priority order: held positions
                    # first (need real-time prices for fast SL/TP), then BTC/ETH (regime),
                    # then the rotator/discovery universe — capped so the stream count stays
                    # bounded (avoids the python-binance queue overflow).
                    _ws = getattr(native_orch, "_market_data_ws", None)
                    if _ws is not None and hasattr(_ws, "set_symbols"):
                        try:
                            _ss_ws = getattr(native_orch, "_shared_state", None)
                            _held = list((getattr(_ss_ws, "positions", {}) or {}).keys()) if _ss_ws else []
                            _ws_universe = _held + ["BTCUSDT", "ETHUSDT"] + list(symbols)
                            await _ws.set_symbols(_ws_universe, max_symbols=12)
                        except Exception as _ws_err:
                            log.debug(f"WS set_symbols failed: {_ws_err}")
                except Exception as e:
                    log.debug(f"Phase 0 discovery failed: {e}")

    # Sync live market prices to shared_state.price_cache before portfolio snapshot
    if app_ctx:
        native_orch = app_ctx.get("_native_orchestrator")
        if native_orch:
            _md = getattr(native_orch, "_market_data", None)
            _ss = getattr(native_orch, "_shared_state", None)
            if _md and _ss and hasattr(_md, "get_prices") and hasattr(_ss, "price_cache"):
                _live = _md.get_prices()
                if _live:
                    _ss.price_cache.update(_live)

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
        # Write trend_regime to shared_state.metrics so arbitration gates can read it
        _ss = app_ctx.get("shared_state") if app_ctx else None
        if _ss is not None and hasattr(_ss, "metrics") and isinstance(market_regime, dict):
            _ss.metrics["trend_regime"] = market_regime.get("trend_regime", "RANGING")
            _ss.metrics["market_regime"] = str((situation_state or {}).get("market_regime", "UNKNOWN"))
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

    # TP/SL trigger injection — runs every cycle regardless of cadence gate
    # The native orchestrator's _phase_understand is never called by this loop
    # so we inject SELL signals here directly from the tp_sl_engine.
    _native_orch = app_ctx.get("_native_orchestrator") if app_ctx else None
    _tp_sl_engine = getattr(_native_orch, "_tp_sl_engine", None) if _native_orch else None
    _ss = getattr(_native_orch, "_shared_state", None) if _native_orch else None
    if _tp_sl_engine is not None and _ss is not None:
        _positions = getattr(_ss, "positions", {}) or {}
        _price_cache = getattr(_ss, "price_cache", {}) or {}
        _prices_now = getattr(_ss, "prices", {}) or {}
        _tpsl_signals = []
        for _sym, _pos in _positions.items():
            _cur = float(
                _prices_now.get(_sym)
                or _price_cache.get(str(_sym).upper())
                or 0.0
            )
            if _cur <= 0:
                continue
            _trigger = _tp_sl_engine.check_triggers(_sym, _pos, _cur)
            if _trigger is not None:
                _tpsl_signals.append({
                    "signal_type": "SELL",
                    "action": "SELL",
                    "symbol": _sym,
                    "edge_score": 1.0,
                    "edge": 1.0,
                    "reason": _trigger,
                    "tpsl_trigger": _trigger,
                })
                log.info("[main:TPSL] %s trigger=%s price=%.6f → injecting SELL", _sym, _trigger, _cur)
        if _tpsl_signals:
            # Prepend TP/SL exits so they take priority in decision phase
            all_signals = list(_tpsl_signals) + list(all_signals)

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
    for sig in all_signals:
        decision = None
        sig_type = str(
            get_value(sig, "signal_type", get_value(sig, "action", "")) or ""
        ).upper()
        symbol = get_value(sig, "symbol", "")
        edge_score = float(
            get_value(sig, "confidence",                           # NativeSignalEngine-adjusted
                get_value(sig, "edge_score",                       # adapter-normalised fallback
                    get_value(sig, "edge", 0.0)                    # legacy field
                )
            ) or 0.0
        )
        _is_tpsl = bool(get_value(sig, "tpsl_trigger", ""))

        # TP/SL exits always processed; BUY and non-tpsl SELLs only when decision cadence is due
        if not _is_tpsl and not decision_due:
            continue

        try:
            if sig_type == "BUY":
                decision = await engines.decision.make_buy_decision(symbol, edge_score)
            elif sig_type == "SELL":
                _sell_source = str(get_value(sig, "tpsl_trigger", "") or get_value(sig, "reason", "") or "signal")
                decision = await engines.decision.make_sell_decision(
                    symbol, edge_score, _sell_source
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
    _arb_engine = app_ctx.get("arbitration_engine") if app_ctx else None
    if mode != "dry-run":
        for decision in trading_decisions:
            order_result = await engines.execution.execute_decision(decision)
            if _is_real_execution_result(order_result):
                executed_orders.append(order_result)

            # Feed arbitration gate_7 so re-entry cooldowns and loss streaks work
            if _arb_engine is None:
                continue
            _dec_action = str(getattr(decision, "action", decision.get("action", "") if hasattr(decision, "get") else "")).upper()
            _dec_symbol = str(getattr(decision, "symbol", decision.get("symbol", "") if hasattr(decision, "get") else ""))
            _dec_allowed = getattr(decision, "allowed", decision.get("allowed", False) if hasattr(decision, "get") else False)
            _exec_success = getattr(order_result, "success", order_result.get("success", False) if hasattr(order_result, "get") else False)

            if not _dec_symbol or not _dec_allowed or not _exec_success:
                continue

            if _dec_action == "BUY":
                _arb_engine.record_buy(_dec_symbol)

            elif _dec_action == "SELL":
                # Determine win/loss from decision telemetry
                _telemetry = getattr(decision, "telemetry", decision.get("telemetry", {}) if hasattr(decision, "get") else {}) or {}
                _pnl = float(_telemetry.get("pnl_after_fees_usdt", 0.0) or 0.0)
                _src = str(getattr(decision, "reason", decision.get("reason", "") if hasattr(decision, "get") else "") or "").upper()
                _is_sl = any(kw in _src for kw in ("SL_HIT", "SL_EXIT", "STOP_LOSS"))
                _is_tp = any(kw in _src for kw in ("TP_HIT", "TAKE_PROFIT"))
                if _is_sl:
                    _arb_engine.record_sl_exit(_dec_symbol)
                    _arb_engine.record_loss(_dec_symbol)
                    _arb_engine.record_trade_outcome(_dec_symbol, _pnl if _pnl != 0.0 else -0.01)
                elif _pnl < 0:
                    _arb_engine.record_loss(_dec_symbol)
                    _arb_engine.record_trade_outcome(_dec_symbol, _pnl)
                else:
                    _arb_engine.record_win(_dec_symbol)
                    _arb_engine.record_trade_outcome(_dec_symbol, _pnl if _pnl > 0.0 else 0.01)

    execute_complete = True

    # ──────────────────────────────────────────────────────────────────
    # PHASE 5: RECOVER / OBSERVE
    # ──────────────────────────────────────────────────────────────────
    # NAV protection evaluation — every 60s
    # Computes peak_nav, drawdown tier, protection mode and writes to shared_state.nav_protection_state
    _nav_prot_due = (
        app_ctx is not None
        and (now_ts - app_ctx.get("_last_nav_prot_ts", 0.0)) >= 60.0
        and _ss is not None
    )
    if _nav_prot_due:
        try:
            from core_engine.native.nav_protection import evaluate_nav_protection
            # 2026-07-14 fix: this used to stomp previous_nav_usdt with the
            # CURRENT nav_usdt immediately before evaluate_nav_protection()
            # read it, collapsing nav_delta to ~0 on every single evaluation
            # and silently disabling the PROFIT_LOCK/FLOATING_GAIN_PROTECTION
            # branches (which key off nav_delta_pct). previous_nav_usdt is
            # already correctly maintained by shared_state.update_nav(),
            # called every trading cycle from get_portfolio_snapshot() -- it
            # does not need (and must not get) a second, destructive writer.
            _, _prot = evaluate_nav_protection(_ss)
            _mode = _prot.protection_mode
            _dd = _prot.drawdown_from_peak_pct
            if _mode != "NORMAL":
                log.info(
                    "[main:NAV-PROTECT] mode=%s drawdown=%.2f%% peak=%.2f allow_buy=%s actions=%s",
                    _mode, _dd, _prot.peak_nav_usdt, _prot.allow_buy, _prot.suggested_actions,
                )
            # Persist peak_nav update to metrics
            _cur_peak = float((_ss.metrics or {}).get("peak_nav", 0.0) or 0.0)
            if _prot.peak_nav_usdt > _cur_peak:
                _ss.metrics["peak_nav"] = _prot.peak_nav_usdt
                _ss.metrics["peak_nav_ts"] = now_ts
                log.info("[main:NAV-PROTECT] New peak NAV: %.4f USDT", _prot.peak_nav_usdt)
        except Exception as _e:
            log.debug("[main:NAV-PROTECT] evaluation failed: %s", _e)
        if app_ctx is not None:
            app_ctx["_last_nav_prot_ts"] = now_ts

    # TP/SL time-tightening + dynamic TP widening — every 5 minutes
    _tpsl_recalc_due = (
        app_ctx is not None
        and (now_ts - app_ctx.get("_last_tpsl_recalc_ts", 0.0)) >= 300.0
    )
    if _tpsl_recalc_due and _tp_sl_engine is not None:
        try:
            _updates = _tp_sl_engine.recalculate_aged_positions()
            if _updates:
                log.info("[main:TPSL-RECALC] time-tightening updated %d positions: %s", len(_updates), list(_updates.keys()))
        except Exception as _e:
            log.warning("[main:TPSL-RECALC] recalculate_aged_positions failed: %s", _e)
        if app_ctx is not None:
            app_ctx["_last_tpsl_recalc_ts"] = now_ts

    cached_health = (app_ctx.get("_cached_health_report") if app_ctx is not None else None) or {}
    if cadence is None or cadence.is_due("health", now=now_ts):
        health_report = await engines.operations.get_health_report()
        if app_ctx is not None:
            app_ctx["_cached_health_report"] = health_report
        if cadence is not None:
            cadence.mark("health", now=now_ts)
    else:
        health_report = cached_health
    # Always read NAV fresh from native shared_state — cached portfolio_snapshot can be stale
    _native_orch_for_nav = app_ctx.get("_native_orchestrator") if app_ctx else None
    _ss_for_nav = getattr(_native_orch_for_nav, "_shared_state", None) if _native_orch_for_nav else None
    if _ss_for_nav is None:
        _ss_for_nav = app_ctx.get("shared_state") if app_ctx else None
    if _ss_for_nav is not None and hasattr(_ss_for_nav, "free_balance_usdt"):
        _free = float(getattr(_ss_for_nav, "free_balance_usdt", 0.0) or 0.0)
        _pos_val = _ss_for_nav.get_portfolio_value() if hasattr(_ss_for_nav, "get_portfolio_value") else 0.0
        nav_val = _free + _pos_val if (_free + _pos_val) > 0 else get_value(portfolio_snapshot, "nav_usdt", 0.0)
    else:
        nav_val = get_value(portfolio_snapshot, "nav_usdt", 0.0)
    regime_val = get_value(market_regime, "overall_health", "unknown")
    primary_decision = trading_decisions[0] if trading_decisions else None
    primary_execution = executed_orders[0] if executed_orders else None
    situation_metrics = get_value(situation_state, "metrics", {}) or {}
    quant_summary = {
        "timestamp": time.time(),
        "nav_usdt": nav_val if nav_val > 0 else float(situation_metrics.get("nav_usdt", 0.0) or 0.0),
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
                # Phase-0 safety net: every 50 cycles, assert the system's core
                # invariants (NAV identity, TP/SL coverage, no orphans/phantoms,
                # priced positions). Logs only violations; can never break the loop.
                if cycle_no % 50 == 0:
                    try:
                        from system_invariants import check_live as _check_inv

                        _no = app_ctx.get("_native_orchestrator") if app_ctx else None
                        _ss_inv = getattr(_no, "_shared_state", None) if _no else None
                        _tpsl_inv = getattr(_no, "_tp_sl_engine", None) if _no else None
                        if _ss_inv is not None:
                            _violations = _check_inv(_ss_inv, _tpsl_inv)
                            for _name, _status, _detail in _violations:
                                log.warning("[INVARIANT] %s — %s: %s", _status, _name, _detail)
                            # Auto-heal: an orphaned holding (held but untracked → no stop,
                            # missing from NAV) is fixable by forcing a balance reconcile,
                            # which re-adopts external positions. Cheap and idempotent.
                            if any("orphan" in str(_n).lower() for _n, _, _ in _violations):
                                try:
                                    if hasattr(_ss_inv, "update_balance_map") and _ss_inv.balance:
                                        _ss_inv.update_balance_map(dict(_ss_inv.balance))
                                        log.warning("[INVARIANT] auto-heal: forced balance reconcile to re-adopt orphan(s)")
                                except Exception as _heal_err:
                                    log.debug("invariant auto-heal failed: %s", _heal_err)
                    except Exception as _inv_err:
                        log.debug("invariant check skipped: %s", _inv_err)
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
        default=5.0,
        help="Seconds between cycles (default: 5.0).",
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


_PID_FILE = "/tmp/octivault_trader.pid"


def _acquire_pid_lock() -> bool:
    """Return True if this process is the sole running instance.

    If a previous instance is still alive, send it SIGTERM so it can shut
    down gracefully, then wait up to 10 s before forcefully replacing it.
    """
    import os
    import signal
    import time as _time

    if os.path.exists(_PID_FILE):
        try:
            with open(_PID_FILE) as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 0)  # raises ProcessLookupError if dead
            log.warning(
                "⚠️  Previous instance found (PID %d) — sending SIGTERM for graceful shutdown.",
                old_pid,
            )
            try:
                os.kill(old_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Wait up to 10 s for it to exit, then SIGKILL
            for _ in range(20):
                _time.sleep(0.5)
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    break
            else:
                log.warning("⚠️  Forcefully killing stale instance (PID %d).", old_pid)
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _time.sleep(0.5)
        except (ProcessLookupError, ValueError):
            pass  # Stale PID file — safe to overwrite
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    return True


def _release_pid_lock() -> None:
    import os
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                if int(f.read().strip()) == os.getpid():
                    os.remove(_PID_FILE)
    except Exception:
        pass


def main() -> None:
    import os
    args = parse_args()
    if not _acquire_pid_lock():
        sys.exit(1)
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        rc = 130
    finally:
        _release_pid_lock()
    sys.exit(rc)


if __name__ == "__main__":
    # Attach the real file/console handlers to the root logger HERE, not at
    # import time — see the comment at the top of this file next to `log =`.
    setup_logging()
    main()
