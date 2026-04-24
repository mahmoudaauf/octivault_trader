
# core/agent_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
AgentManager — Central agent registry and orchestrator for Octivault Trader
... (see full docstring below)
"""

import logging
import os
import inspect
import time
import importlib
import asyncio as _asyncio  # single asyncio import (aliased)
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import json
from datetime import datetime

from core.component_status_logger import ComponentStatusLogger
# HealthStatus import removed - not used after P9 refactor
try:
    from core.health import update_health
except ImportError:
    try:
        from core.healthy import update_health
    except ImportError:
        async def update_health(shared_state, component_name, status, detail=""):
            """Fallback stub for update_health"""
            pass

try:
    from core.model_manager import ModelManager
except ImportError:
    ModelManager = None

# --- Canonical TradeIntent import (P9 invariant) with stub shim ---
try:
    from core.stubs import TradeIntent
except Exception as _e:
    # Bind a stub to avoid NameError during module import; any runtime use will raise clearly.
    class _MissingTradeIntent:  # type: ignore
        def __getattr__(self, name):  # pragma: no cover
            raise ImportError(
                "TradeIntent is missing. Expected in core.stubs "
                "or core.baseline_trading_kernel. Please ensure the canonical module is present."
            )
    TradeIntent = _MissingTradeIntent()  # type: ignore

try:
    from core.agent_registry import AGENT_CLASS_MAP, AGENT_IMPORT_ERRORS
except Exception:
    from core.agent_registry import AGENT_CLASS_MAP
    AGENT_IMPORT_ERRORS = {}

logger = logging.getLogger("AgentManager")
# Guard creation (not just adding) to avoid leaking FileHandler objects on reload.
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _log_path = "logs/core/agent_manager.log"
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)
    _fh = logging.FileHandler(_log_path)
    _fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(_fh)
logger.setLevel(logging.INFO)
logger.propagate = False  # avoid duplicate logs in root logger


class AgentManager:
    @staticmethod
    def _task_cancel_requested() -> bool:
        """
        True only when the *current* task has an active cancellation request.
        Helps distinguish framework shutdown from stray CancelledError in child calls.
        """
        task = _asyncio.current_task()
        if task is None:
            return False
        cancelling = getattr(task, "cancelling", None)
        if callable(cancelling):
            try:
                return bool(cancelling())
            except Exception:
                return False
        return task.cancelled()

    async def start(self):
        if self._started:
            return

        # Ensure agents are registered before marking started.
        if not self.agents:
            try:
                await self.auto_register_agents()
            except Exception as e:
                self.logger.error("AgentManager: failed to auto-register agents: %s", e, exc_info=True)
                # Do NOT set _started — allow the caller to retry start().
                return

        # ✅ BOOTSTRAP: Run discovery agents once to populate symbol universe
        try:
            await self.bootstrap_symbol_universe()
        except Exception as e:
            self.logger.warning("[Bootstrap] Symbol universe population failed: %s", e, exc_info=True)
            # Continue anyway - lazy loading in strategy agents will retry

        # Only mark started after registration has succeeded so retry is possible.
        self._started = True

        if hasattr(self, "run_loop"):
            _asyncio.create_task(self.run_loop(), name="AgentManager:run_loop")
        elif hasattr(self, "run"):
            _asyncio.create_task(self.run(), name="AgentManager:run")

    async def bootstrap_symbol_universe(self):
        """
        Run discovery agents ONCE to populate the initial symbol universe.
        If discovery fails or returns no symbols, fallback to default symbols.
        This ensures strategy agents can access symbols when they call generate_signals().
        
        ✅ Fixes: SwingTradeHunter, TrendHunter, DipSniper 0 symbols issue
        ✅ Timing: Called during startup BEFORE strategy agents run
        ✅ Fallback: Uses default symbols if discovery fails
        """
        self.logger.info("[Bootstrap] Populating symbol universe from discovery agents...")
        
        populated = False
        discovery_count = 0
        
        # Get discovery agents
        discovery_agent_instances = self.get_discovery_agents()
        
        if not discovery_agent_instances:
            self.logger.debug("[Bootstrap] No discovery agents registered")
        else:
            # Run each discovery agent once
            for agent in discovery_agent_instances:
                agent_name = getattr(agent, 'name', agent.__class__.__name__)
                if not hasattr(agent, 'run_once'):
                    self.logger.debug(f"[Bootstrap] Agent {agent_name} has no run_once() method")
                    continue
                
                try:
                    self.logger.info(f"[Bootstrap] Running discovery: {agent_name}")
                    result = agent.run_once()
                    if inspect.iscoroutine(result):
                        await result
                    self.logger.info(f"[Bootstrap] ✅ {agent_name} completed discovery")
                    populated = True
                    discovery_count += 1
                except Exception as e:
                    self.logger.warning(f"[Bootstrap] Discovery failed for {agent_name}: {e}", exc_info=True)
                    continue
        
        # Verify symbols were populated
        try:
            syms = self.shared_state.get_accepted_symbols()
            if inspect.iscoroutine(syms):
                syms = await syms
            
            if isinstance(syms, dict):
                sym_count = len(syms)
            else:
                sym_count = len(list(syms or []))
            
            # If still empty, use fallback
            if sym_count == 0:
                self.logger.warning("[Bootstrap] ⚠️ Symbol universe is EMPTY after discovery! Applying fallback...")
                try:
                    from core.bootstrap_symbols import bootstrap_default_symbols
                    result = await bootstrap_default_symbols(self.shared_state, self.logger)
                    if result:
                        syms = self.shared_state.get_accepted_symbols()
                        if inspect.iscoroutine(syms):
                            syms = await syms
                        if isinstance(syms, dict):
                            sym_count = len(syms)
                        self.logger.info(f"[Bootstrap] ✅ Fallback: Seeded {sym_count} default symbols")
                except Exception as e:
                    self.logger.error(f"[Bootstrap] Fallback failed: {e}", exc_info=True)
            else:
                self.logger.info(f"[Bootstrap] ✅ Symbol universe populated: {sym_count} symbols from {discovery_count} agents")
        except Exception as e:
            self.logger.warning(f"[Bootstrap] Could not verify symbols: {e}")

    
    def __init__(
        self,
        shared_state,
        market_data,
        execution_manager,
        config,
        symbols: Optional[List[str]] = None,
        tp_sl_engine=None,
        market_data_feed=None,
        model_manager=None,
        meta_controller=None,
        symbol_manager=None,
        exchange_client=None,
        database_manager=None,
        agent_schedule=None,
    ):
        self.shared_state = shared_state
        self.market_data = market_data
        self.execution_manager = execution_manager
        self.config = config
        self.symbols = symbols if symbols is not None else []
        self.tp_sl_engine = tp_sl_engine
        self.market_data_feed = market_data_feed
        self.model_manager = model_manager if model_manager is not None else (ModelManager(self.config) if ModelManager else None)
        self.meta_controller = meta_controller
        self.symbol_manager = symbol_manager
        self.exchange_client = exchange_client
        self.database_manager = database_manager
        self.agent_schedule = agent_schedule

        self.agents: Dict[str, Any] = {}
        self.discovery_agents = []
        self.logger = logger
        self._started = False
        self._tasks: Dict[str, _asyncio.Task] = {}
        self._manager_tasks: Dict[str, _asyncio.Task] = {}  # health/tick/run_all_agents
        self._strategies_prepared = False  # New flag
        self._strategies_started = False  # New flag
        # Tunables
        self._max_start_concurrency = int(getattr(self.config, "AGENTMGR_MAX_START_CONCURRENCY", 6))
        self._agent_timeout_s = float(getattr(self.config, "AGENTMGR_AGENT_TIMEOUT_S", 10.0))
        self._restart_on_crash = bool(getattr(self.config, "AGENTMGR_ENABLE_RESTART", True))
        self._restart_backoff_min = float(getattr(self.config, "AGENTMGR_RESTART_BACKOFF_MIN", 2.0))
        self._restart_backoff_max = float(getattr(self.config, "AGENTMGR_RESTART_BACKOFF_MAX", 60.0))
        self._market_data_ready_timeout_s = float(getattr(self.config, "AGENTMGR_MARKETDATA_READY_TIMEOUT_S", 180.0))
        raw_retrain_interval = float(
            getattr(
                self.config,
                "AGENTMGR_STRATEGY_RETRAIN_INTERVAL_S",
                os.getenv("AGENTMGR_STRATEGY_RETRAIN_INTERVAL_S", 1800.0),
            )
            or 1800.0
        )
        self._strategy_retrain_min_interval_s = float(
            getattr(
                self.config,
                "AGENTMGR_STRATEGY_RETRAIN_MIN_INTERVAL_S",
                os.getenv("AGENTMGR_STRATEGY_RETRAIN_MIN_INTERVAL_S", 300.0),
            )
            or 300.0
        )
        allow_fast_retrain = str(
            getattr(
                self.config,
                "AGENTMGR_STRATEGY_RETRAIN_ALLOW_FAST",
                os.getenv("AGENTMGR_STRATEGY_RETRAIN_ALLOW_FAST", "false"),
            )
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            not allow_fast_retrain
            and raw_retrain_interval > 0
            and raw_retrain_interval < self._strategy_retrain_min_interval_s
        ):
            self.logger.warning(
                "[Phase9:Retrain] Interval %.1fs is below architectural floor %.1fs; clamping.",
                raw_retrain_interval,
                self._strategy_retrain_min_interval_s,
            )
            raw_retrain_interval = self._strategy_retrain_min_interval_s
        self._strategy_retrain_interval_s = raw_retrain_interval
        self._strategy_retrain_timeout_s = float(
            getattr(
                self.config,
                "AGENTMGR_STRATEGY_RETRAIN_TIMEOUT_S",
                os.getenv("AGENTMGR_STRATEGY_RETRAIN_TIMEOUT_S", 3600.0),
            )
            or 3600.0
        )
        self._last_symbols_refresh_t = 0.0
        self._accepted_symbols_cache: List[str] = []
        self._last_agent_log_t: Dict[str, float] = {}  # Per-agent logging throttle (Issue 2)
        self._last_empty_intent_log_t = 0.0
        self._empty_intent_log_interval_s = float(
            getattr(self.config, "AGENTMGR_EMPTY_INTENT_LOG_INTERVAL_S", 60.0)
        )
        self._strategy_autoregister_retry_interval_s = float(
            getattr(self.config, "AGENTMGR_STRATEGY_AUTOREG_RETRY_S", 60.0)
        )
        self._last_strategy_autoregister_retry_t = 0.0
        self._optional_import_log_interval_s = float(
            getattr(self.config, "AGENTMGR_OPTIONAL_IMPORT_LOG_INTERVAL_S", 300.0)
        )
        self._last_optional_import_log_ts: Dict[str, float] = {}
        # Cached _MetaAdapter — stateless wrapper, safe to create once per meta_controller.
        self._meta_adapter: Optional[Any] = None
        
        # Watchdog periodic status reporting
        self._signal_counter = 0
        self._last_watchdog_report_ts = time.time()
        self._watchdog_report_interval_s = 30.0  # Report to watchdog every 30 seconds

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Config getter supporting dict and attribute-style configs."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _resolve_intent_planned_quote(self, intent: Dict[str, Any]) -> float:
        """Best-effort planned quote resolver for pre-publish viability checks."""
        if not isinstance(intent, dict):
            return 0.0
        for key in ("quote_hint", "planned_quote", "quote", "_planned_quote", "qty_quote"):
            try:
                val = float(intent.get(key, 0.0) or 0.0)
                if val > 0.0:
                    return val
            except Exception:
                continue
        fallback = float(
            self._cfg(
                "DEFAULT_PLANNED_QUOTE",
                self._cfg("MIN_ENTRY_USDT", self._cfg("MIN_ENTRY_QUOTE_USDT", 0.0)),
            )
            or 0.0
        )
        return max(0.0, fallback)

    def _passes_prepublish_viability_gate(self, intent: Dict[str, Any]) -> tuple[bool, str]:
        """
        Manager-side pre-publish viability gate.
        Reuses MetaController's existing pretrade effect gate instead of duplicating logic.
        """
        if not bool(self._cfg("AGENTMGR_PREPUBLISH_PRETRADE_GUARD_ENABLED", True)):
            return True, "disabled"
        if not isinstance(intent, dict):
            return False, "invalid_intent_payload"
        if self.meta_controller is None:
            return True, "meta_unavailable"

        gate_fn = getattr(self.meta_controller, "_passes_pretrade_effect_gate", None)
        if not callable(gate_fn):
            return True, "meta_gate_unavailable"

        side = str(intent.get("side") or intent.get("action") or "").upper().strip()
        if side not in {"BUY", "SELL"}:
            return False, "invalid_side"
        if side == "SELL" and not bool(self._cfg("AGENTMGR_PREPUBLISH_GUARD_SELL", False)):
            return True, "sell_bypass"

        symbol = str(intent.get("symbol") or "").replace("/", "").upper().strip()
        if not symbol:
            return False, "missing_symbol"

        planned_quote = float(self._resolve_intent_planned_quote(intent))
        if planned_quote <= 0.0:
            return False, "invalid_quote"

        signal: Dict[str, Any] = {
            "action": side,
            "confidence": float(intent.get("confidence", 0.0) or 0.0),
            "reason": str(intent.get("rationale") or intent.get("reason") or ""),
            "quote": planned_quote,
            "quote_hint": planned_quote,
        }

        passthrough_keys = (
            "_expected_move_pct",
            "expected_move_pct",
            "_regime",
            "regime",
            "_regime_scaling",
            "_tradeability_hint",
            "_break_even_prob",
            "_required_conf",
            "_atr_pct",
            "atr_pct",
            "edge",
        )
        for key in passthrough_keys:
            if key in intent and intent.get(key) is not None:
                signal[key] = intent.get(key)
        if isinstance(intent.get("policy_context"), dict):
            for key, value in intent["policy_context"].items():
                signal.setdefault(key, value)

        try:
            ok, reason = gate_fn(
                symbol=symbol,
                signal=signal,
                planned_quote=float(planned_quote),
                side=side,
            )
            if ok:
                return True, str(reason or "ok")

            reason_s = str(reason or "pretrade_gate")
            reason_u = reason_s.upper()
            if (
                side == "BUY"
                and bool(self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_RELAX_ENABLED", True))
                and reason_u in {
                    "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD",
                    "MICRO_BACKTEST_AVG_NET_BELOW_THRESHOLD",
                    "MICRO_BACKTEST_INSUFFICIENT_SAMPLES",
                    "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD",
                    "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_AVG_NET_BELOW_THRESHOLD",
                    "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_INSUFFICIENT_SAMPLES",
                }
            ):
                bt_rejections = 0
                try:
                    rej_getter = getattr(self.shared_state, "get_rejection_count", None)
                    if callable(rej_getter):
                        bt_rejections += int(
                            rej_getter(symbol, "BUY", "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD") or 0
                        )
                        bt_rejections += int(
                            rej_getter(symbol, "BUY", "MICRO_BACKTEST_AVG_NET_BELOW_THRESHOLD") or 0
                        )
                        bt_rejections += int(
                            rej_getter(symbol, "BUY", "MICRO_BACKTEST_INSUFFICIENT_SAMPLES") or 0
                        )
                        bt_rejections += int(
                            rej_getter(
                                symbol,
                                "BUY",
                                "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD",
                            )
                            or 0
                        )
                        bt_rejections += int(
                            rej_getter(
                                symbol,
                                "BUY",
                                "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_AVG_NET_BELOW_THRESHOLD",
                            )
                            or 0
                        )
                        bt_rejections += int(
                            rej_getter(
                                symbol,
                                "BUY",
                                "PRETRADE_EFFECT_GATE:MICRO_BACKTEST_INSUFFICIENT_SAMPLES",
                            )
                            or 0
                        )
                except Exception:
                    bt_rejections = 0

                relax_trigger = int(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_REJECTION_TRIGGER", 24) or 24
                )
                relax_step = int(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_CONF_STEP_REJECTIONS", 12) or 12
                )
                conf_floor = float(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_MIN_CONF_FLOOR", 0.60) or 0.60
                )
                conf_base = float(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_MIN_CONF_BASE", 0.74) or 0.74
                )
                conf_step_decay = float(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_MIN_CONF_DECAY_PER_STEP", 0.02) or 0.02
                )
                min_exp_move = float(
                    self._cfg("AGENTMGR_PREPUBLISH_DEADLOCK_MIN_EXPECTED_MOVE_PCT", 0.0030) or 0.0030
                )

                confidence = max(0.0, min(1.0, float(signal.get("confidence", 0.0) or 0.0)))
                exp_move = signal.get("_expected_move_pct", signal.get("expected_move_pct"))
                exp_move_norm = 0.0
                try:
                    if exp_move is not None:
                        exp_move_norm = float(exp_move)
                        if abs(exp_move_norm) > 1.0:
                            exp_move_norm = exp_move_norm / 100.0
                except Exception:
                    exp_move_norm = 0.0

                if bt_rejections >= max(1, relax_trigger):
                    extra = max(0, bt_rejections - relax_trigger)
                    relief_steps = 1 + int(extra // max(1, relax_step))
                    required_conf = max(conf_floor, conf_base - (conf_step_decay * relief_steps))
                    required_conf = max(conf_floor, min(0.95, required_conf))
                    if confidence >= required_conf and exp_move_norm >= min_exp_move:
                        self.logger.warning(
                            "[AgentManager:PrePublishGate:DeadlockRelief] Allowing %s BUY "
                            "after bt_rejections=%d (reason=%s conf=%.2f req=%.2f exp_move=%.3f%% min=%.3f%%)",
                            symbol,
                            int(bt_rejections),
                            reason_s,
                            confidence,
                            required_conf,
                            exp_move_norm * 100.0,
                            min_exp_move * 100.0,
                        )
                        return True, "deadlock_relief_override"

            return False, reason_s
        except Exception as exc:
            self.logger.debug(
                "[AgentManager:PrePublishGate] Meta pretrade gate error for %s %s: %s",
                symbol,
                side,
                exc,
                exc_info=True,
            )
            return True, "gate_error_allow"

    async def _filter_intents_prepublish(self, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter BUY intents through pretrade viability before publishing to Meta/event bus."""
        if not intents:
            return []

        filtered: List[Dict[str, Any]] = []
        dropped = 0
        for intent in intents:
            ok, reason = self._passes_prepublish_viability_gate(intent)
            if ok:
                filtered.append(intent)
                continue
            dropped += 1
            symbol = str(intent.get("symbol") or "").replace("/", "").upper()
            side = str(intent.get("side") or intent.get("action") or "").upper()
            self.logger.info(
                "[AgentManager:PrePublishGate] Dropped intent %s %s agent=%s reason=%s",
                symbol,
                side,
                intent.get("agent", "unknown"),
                reason,
            )
            try:
                rec = getattr(self.shared_state, "record_rejection", None)
                if callable(rec) and symbol and side:
                    rv = rec(symbol, side, str(reason or "prepublish_gate"), source="AgentManager")
                    if _asyncio.iscoroutine(rv):
                        await rv
            except Exception:
                pass

        if dropped > 0:
            self.logger.info(
                "[AgentManager:PrePublishGate] Filtered intents: in=%d out=%d dropped=%d",
                len(intents),
                len(filtered),
                dropped,
            )
        return filtered

    async def _report_watchdog_status(self, status: str = "Operational", detail: str = ""):
        """Report status to watchdog for health monitoring"""
        try:
            update_fn = getattr(self.shared_state, "update_component_status", None)
            if update_fn:
                result = update_fn("AgentManager", status, detail)
                if _asyncio.iscoroutine(result):
                    await result
        except Exception:
            pass  # Status reporting is non-critical

    # --- Meta signal adapter so all agents have a consistent API ---
    class _MetaAdapter:
        """Thin wrapper that guarantees submit_signal/submit_agent_signal exist for agents."""
        def __init__(self, meta):
            self._meta = meta
        async def submit_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any], confidence: Optional[float]=None):
            # Prefer native shim if Meta has it; fall back to receive_signal
            if hasattr(self._meta, "submit_signal"):
                return await self._meta.submit_signal(agent_name, symbol, signal, confidence)
            payload = dict(signal or {})
            if confidence is not None:
                payload["confidence"] = confidence
            return await self._meta.receive_signal(agent_name, symbol, payload)
        async def submit_agent_signal(self, agent_name: str, symbol: str, action: Dict[str, Any], confidence: float):
            if hasattr(self._meta, "submit_agent_signal"):
                return await self._meta.submit_agent_signal(agent_name, symbol, action, confidence)
            payload = dict(action or {})
            payload["confidence"] = confidence
            return await self._meta.receive_signal(agent_name, symbol, payload)
        def __getattr__(self, name):
            # Expose the rest of MetaController transparently (e.g., evaluate_signals, start, etc.)
            return getattr(self._meta, name)

    async def emit_to_meta(self, *, agent_name: str, symbol: str, action: str, confidence: float,
                           quote: Optional[float]=None, quantity: Optional[float]=None,
                           horizon_hours: Optional[float]=None, extra: Optional[Dict[str, Any]]=None):
        """
        Optional manager-level emitter agents can call instead of touching meta directly.
        Normalizes payload and forwards to meta adapter.
        """
        if not self.meta_controller:
            self.logger.warning("[AgentManager] No MetaController set; dropping signal %s %s", symbol, action)
            return
        sym = (symbol or "").replace("/", "").upper()
        payload = dict(extra or {})
        payload["action"] = (action or "").upper()
        payload["confidence"] = float(confidence)
        if quote is not None: payload["quote"] = float(quote)
        if quantity is not None: payload["quantity"] = float(quantity)
        if horizon_hours is not None: payload["horizon_hours"] = float(horizon_hours)
        try:
            if self._meta_adapter is None:
                self._meta_adapter = self._MetaAdapter(self.meta_controller)
            adapter = self._meta_adapter
            await adapter.submit_signal(agent_name, sym, payload, payload.get("confidence"))
            self.logger.info("[AgentEmit] %s %s conf=%.2f%s%s",
                             sym, payload["action"], payload["confidence"],
                             f" quote={payload.get('quote'):.2f}" if "quote" in payload else "",
                             f" qty={payload.get('quantity'):.6f}" if "quantity" in payload else "")
        except Exception as e:
            self.logger.warning("[AgentManager] emit_to_meta failed for %s: %s", sym, e, exc_info=True)

    async def submit_trade_intents(self, intents: List[Dict[str, Any]]):
        """
        Bind Agent→Meta pipe through the event bus.
        Strategists publish canonical TradeIntent payloads on:
          events.trade.intent
        """
        if not intents:
            self.logger.warning("[AgentManager:SUBMIT] submit_trade_intents called with empty list - no-op")
            return

        self.logger.warning("[AgentManager:SUBMIT] Publishing %d intents to event_bus", len(intents))

        event_bus = getattr(self.shared_state, "event_bus", None)
        publish = getattr(event_bus, "publish", None)
        if not callable(publish):
            self.logger.warning("submit_trade_intents called but shared_state.event_bus.publish is unavailable.")
            return

        published = 0
        for raw in intents:
            ti = self._coerce_trade_intent(raw)
            if ti is None:
                continue
            try:
                await publish("events.trade.intent", ti)
                published += 1
            except Exception as e:
                self.logger.warning("Failed to publish trade intent event for %s: %s", getattr(ti, "symbol", "?"), e)

        if published > 0:
            self.logger.info("[AgentManager] Published %d trade intent events", published)
            self.logger.warning("[AgentManager:SUBMIT] ✓ Published %d intents to event_bus", published)

    def _coerce_trade_intent(self, raw: Any) -> Optional[TradeIntent]:
        """Best-effort conversion into canonical TradeIntent."""
        try:
            if isinstance(TradeIntent, type) and isinstance(raw, TradeIntent):
                return raw
            if not isinstance(raw, dict):
                return None
            symbol = str(raw.get("symbol") or "").replace("/", "").upper()
            side = str(raw.get("side") or raw.get("action") or "").upper()
            if not symbol or side not in {"BUY", "SELL"}:
                return None
            qty_hint = raw.get("qty_hint", raw.get("quantity", raw.get("planned_qty")))
            quote_hint = raw.get("quote_hint", raw.get("quote", raw.get("planned_quote")))
            # Preserve extra metadata in policy_context so downstream can recover regime info
            passthrough_keys = [
                "_regime",
                "regime",
                "_regime_scaling",
                "_expected_move_pct",
                "expected_move_pct",
                "_tradeability_hint",
                "_break_even_prob",
                "_required_conf",
                "tradeability_regime",
                "volatility_regime",
                "_atr_pct",
                "atr_pct",
            ]
            policy_ctx: Dict[str, Any] = {}
            if isinstance(raw.get("policy_context"), dict):
                policy_ctx.update(raw.get("policy_context"))
            for k in passthrough_keys:
                if k in raw and k not in policy_ctx:
                    policy_ctx[k] = raw[k]
            return TradeIntent(
                symbol=symbol,
                side=side,
                qty_hint=float(qty_hint) if qty_hint is not None else None,
                quote_hint=float(quote_hint) if quote_hint is not None else None,
                agent=str(raw.get("agent") or "AgentManager"),
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                rationale=str(raw.get("rationale") or raw.get("reason") or ""),
                ttl_sec=int(raw.get("ttl_sec", 30) or 30),
                tag=str(raw.get("tag") or f"strategy/{raw.get('agent', 'AgentManager')}"),
                timeframe=(str(raw.get("timeframe")) if raw.get("timeframe") else None),
                policy_context=policy_ctx or None,
            )
        except Exception:
            return None

    # NEW: normalize any agent-returned signals into TradeIntents
    def _normalize_to_intents(self, agent_name: str, raw: Any) -> list:
        intents = []
        # Distinguish "agent returned nothing" from "agent returned an empty list".
        # `None` usually indicates an error path or a contract violation; empty collections
        # are normal when no setups are present.
        if raw is None:
            self.logger.warning("[AgentManager:NORMALIZE] None raw signals from %s", agent_name)
            return intents
        if isinstance(raw, (list, tuple, set)) and len(raw) == 0:
            self.logger.debug("[AgentManager:NORMALIZE] 0 raw signals from %s", agent_name)
            return intents
        if raw == {}:
            self.logger.debug("[AgentManager:NORMALIZE] 0 raw signals from %s (empty dict)", agent_name)
            return intents
        if raw is False or raw == 0 or raw == "":
            self.logger.debug("[AgentManager:NORMALIZE] 0 raw signals from %s (falsey=%r)", agent_name, raw)
            return intents
        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, (list, tuple, set)):
            raw = [raw]
        self.logger.debug("[AgentManager:NORMALIZE] Normalizing %d raw signals from %s", len(raw), agent_name)
        for s in raw:
            # Handle canonical TradeIntent objects directly (previously dropped silently).
            if isinstance(TradeIntent, type) and isinstance(s, TradeIntent):
                side = (getattr(s, "side", "") or "").upper()
                if side not in ("BUY", "SELL"):
                    continue
                intent_obj = {
                    "symbol": (getattr(s, "symbol", "") or "").replace("/", "").upper(),
                    "action": side,
                    "side": side,
                    "qty_hint": getattr(s, "qty_hint", None),
                    "quote_hint": getattr(s, "quote_hint", None),
                    "agent": agent_name,
                    "confidence": max(0.0, min(1.0, float(getattr(s, "confidence", 0.0) or 0.0))),
                    "rationale": getattr(s, "rationale", None),
                    "ts": float(getattr(s, "ts", None) or time.time()),
                    "ttl_sec": int(getattr(s, "ttl_sec", 30) or 30),
                    "tag": f"strategy/{agent_name}",
                    "budget_required": (side == "BUY"),
                }
                # Pass through policy_context metadata if present on the TradeIntent
                ctx = getattr(s, "policy_context", None)
                if isinstance(ctx, dict) and ctx:
                    intent_obj["policy_context"] = dict(ctx)
                intents.append(intent_obj)
                continue
            # Accept dict signals {"symbol","action","confidence",...}
            if not isinstance(s, dict):
                self.logger.debug(
                    "[_normalize_to_intents] Agent '%s' yielded non-dict, non-TradeIntent item (%s); skipping.",
                    agent_name, type(s).__name__,
                )
                continue
            sym = (s.get("symbol") or s.get("sym") or "").replace("/", "").upper()
            act = (s.get("action") or s.get("side") or "").lower()
            if not sym or act not in ("buy", "sell"):
                continue
            intent = {
                "symbol": sym,
                "action": act.upper(),
                "side": act.upper(),
                "qty_hint": s.get("qty") or s.get("qty_hint"),
                "quote_hint": s.get("quote") or s.get("quote_hint"),
                "agent": agent_name,
                "confidence": max(0.0, min(1.0, float(s.get("confidence") or 0.0))),
                "rationale": s.get("reason") or s.get("rationale"),
                "ts": float(s.get("ts") or s.get("timestamp") or time.time()),  # CRITICAL: Intent freshness
                "ttl_sec": int(s.get("ttl_sec") or 30),
                "tag": f"strategy/{agent_name}",
                "budget_required": (act == "buy"),
            }
            # Preserve regime / expected-move metadata so MetaController can build policy context
            passthrough_keys = [
                "_regime",
                "regime",
                "_regime_scaling",
                "_expected_move_pct",
                "expected_move_pct",
                "_tradeability_hint",
                "_break_even_prob",
                "_required_conf",
                "tradeability_regime",
                "volatility_regime",
                "_atr_pct",
                "atr_pct",
            ]
            # Build policy_context with passthrough keys (NOT at top level)
            policy_ctx = {}
            if isinstance(s.get("policy_context"), dict):
                policy_ctx.update(s["policy_context"])
            for k in passthrough_keys:
                if k in s and k not in policy_ctx:
                    policy_ctx[k] = s[k]
            # Store policy_context as a nested dict, not at top level
            if policy_ctx:
                intent["policy_context"] = policy_ctx
            intents.append(intent)
        if raw and not intents:
            self.logger.warning(
                "[_normalize_to_intents] Agent '%s' provided %d raw signals but NONE passed normalization. First item: %s",
                agent_name, len(raw), raw[0] if raw else "N/A",
            )
        if intents:
            self.logger.info("[AgentManager:NORMALIZE] ✓ Normalized %d intents from %s", len(intents), agent_name)
        return intents

    async def collect_and_forward_signals(self):
        """Single signal collection point - calls generate_signals() once per tick."""
        # Periodic watchdog status reporting
        self._signal_counter += 1
        now_ts = time.time()
        if (now_ts - self._last_watchdog_report_ts) >= self._watchdog_report_interval_s:
            await self._report_watchdog_status(
                "Operational",
                f"Processed {self._signal_counter} signal batches"
            )
            self._last_watchdog_report_ts = now_ts
        
        self.logger.debug("[AgentManager] Signal Collection Tick. SharedState ID: %d, Meta ID: %d", id(self.shared_state), id(self.meta_controller))
        batch = []
        strategy_agents = [
            (name, agent)
            for name, agent in list(self.agents.items())
            if getattr(agent, "agent_type", None) != "discovery" and hasattr(agent, "generate_signals")
        ]
        
        # 🔥 CRITICAL DEBUG: Log if no strategy agents found
        if not strategy_agents:
            self.logger.warning("[AgentManager] ⚠️ NO STRATEGY AGENTS FOUND! registered_agents=%s agent_types=%s", 
                               list(self.agents.keys()),
                               {n: getattr(a, "agent_type", "unknown") for n, a in self.agents.items()})
        if not strategy_agents:
            now_ts = time.time()
            retry_interval = max(5.0, float(self._strategy_autoregister_retry_interval_s or 60.0))
            if (now_ts - float(self._last_strategy_autoregister_retry_t or 0.0)) >= retry_interval:
                self._last_strategy_autoregister_retry_t = now_ts
                self.logger.warning(
                    "[AgentManager] No signal-capable strategy agents registered. Retrying strategy auto-registration."
                )
                try:
                    await self.auto_register_agents(filter_types={"strategy"})
                except Exception as e:
                    self.logger.warning(
                        "[AgentManager] Strategy auto-registration retry failed: %s",
                        e,
                        exc_info=True,
                    )
                strategy_agents = [
                    (name, agent)
                    for name, agent in list(self.agents.items())
                    if getattr(agent, "agent_type", None) != "discovery" and hasattr(agent, "generate_signals")
                ]

        for name, agent in strategy_agents:
            # ARCHITECTURAL FIX: Budget gating moved to MetaController.
            # We must allow agents to generate signals even with 0 budget
            # so they can propose EXITS (SELL signals) for open positions.

            # Call generate_signals() exactly once per tick
            try:
                fn = getattr(agent, "generate_signals")
                res = fn()
                if inspect.isawaitable(res):
                    res = await res
                if res is None:
                    raw_count = 0
                elif isinstance(res, dict):
                    raw_count = 1
                elif isinstance(res, (list, tuple, set)):
                    raw_count = len(res)
                else:
                    raw_count = 1
                self.logger.debug("[%s] generate_signals() returned %d raw signals.", name, raw_count)
                intents = self._normalize_to_intents(name, res)
                if intents:
                    batch.extend(intents)
                    symbol_count = len(getattr(agent, "symbols", []))
                    self.logger.info("[%s] Normalized %d intents (scanned %d symbols)", name, len(intents), symbol_count)
                elif res:
                    self.logger.warning("[%s] FAILED to normalize any of the %d signals.", name, len(res))
            except _asyncio.CancelledError:
                if self._task_cancel_requested():
                    raise
                self.logger.warning(
                    "[%s] Signal generation raised unexpected CancelledError; continuing tick.",
                    name,
                )
            except Exception as e:
                self.logger.warning("[%s] Signal generation failed: %s", name, e, exc_info=True)

        if batch:
            publish_batch = await self._filter_intents_prepublish(batch)
            if not publish_batch:
                self.logger.info(
                    "[AgentManager] All %d intents were filtered by pre-publish viability gate.",
                    len(batch),
                )
                return

            await self.submit_trade_intents(publish_batch)
            self.logger.info("Submitted %d TradeIntents to Meta", len(publish_batch))
            
            # 🔥 CRITICAL DEBUG: Log submission
            self.logger.warning("[AgentManager:BATCH] Submitted batch of %d intents: %s", 
                               len(publish_batch),
                               [f"{i.get('agent')}:{i.get('symbol')}" for i in publish_batch])
            
            # 🔥 CRITICAL FIX: DIRECT PATH TO METACONTROLLER
            # Don't wait for event bus drain - forward signals directly to MetaController
            # This ensures signals reach the signal_cache IMMEDIATELY
            if self.meta_controller:
                direct_count = 0
                for intent in publish_batch:
                    try:
                        symbol = intent.get("symbol")
                        agent = intent.get("agent", "AgentManager")
                        # Convert TradeIntent back to signal format for direct reception
                        signal = {
                            "action": intent.get("action") or intent.get("side"),
                            "confidence": float(intent.get("confidence", 0.0)),
                            "reason": intent.get("rationale") or intent.get("reason", ""),
                            "quote": intent.get("quote_hint") or intent.get("quote"),
                            "timestamp": time.time(),
                        }
                        passthrough_keys = [
                            "_regime",
                            "regime",
                            "_regime_scaling",
                            "_expected_move_pct",
                            "expected_move_pct",
                            "_tradeability_hint",
                            "_break_even_prob",
                            "_required_conf",
                            "tradeability_regime",
                            "volatility_regime",
                            "_atr_pct",
                            "atr_pct",
                        ]
                        for k in passthrough_keys:
                            if k in intent:
                                signal[k] = intent[k]
                        if isinstance(intent.get("policy_context"), dict):
                            # Merge policy_context fields but don't overwrite explicit signal keys
                            for k, v in intent["policy_context"].items():
                                signal.setdefault(k, v)
                        await self.meta_controller.receive_signal(agent, symbol, signal)
                        direct_count += 1
                    except Exception as e:
                        self.logger.debug("[AgentManager] Direct signal forward failed for %s from %s: %s", 
                                        intent.get("symbol"), agent, e)
                
                if direct_count > 0:
                    self.logger.info("[AgentManager:DIRECT] Forwarded %d signals directly to MetaController.signal_cache", direct_count)
        else:
            now_ts = time.time()
            interval = max(5.0, float(self._empty_intent_log_interval_s or 60.0))
            if (now_ts - float(self._last_empty_intent_log_t or 0.0)) >= interval:
                self._last_empty_intent_log_t = now_ts
                strategy_names = [n for n, _a in strategy_agents]
                self.logger.info(
                    "[AgentManager] No TradeIntents collected this tick (strategy_agents=%d names=%s registered_agents=%d).",
                    len(strategy_agents),
                    strategy_names,
                    len(self.agents),
                )


    def register_agent(self, agent):
        """Adds an agent instance to the manager's registry."""
        if not hasattr(agent, "name"):
            raise ValueError("Agent must have a 'name' attribute.")
        if not agent.name:
            raise ValueError("Agent name must not be empty.")
        self.agents[agent.name] = agent

    async def auto_register_agents(self, filter_types: Optional[set] = None):
        self._ensure_optional_agent_classes()
        ml_err = (AGENT_IMPORT_ERRORS or {}).get("MLForecaster")
        if ml_err:
            self._log_optional_import_issue(
                "MLForecaster",
                f"agent_registry import failure: {ml_err.get('error')}",
                include_trace=False,
            )
        try:
            keys = sorted(list(AGENT_CLASS_MAP.keys()))
            self.logger.info(
                "[AgentManager] Registry snapshot: total=%d has_MLForecaster=%s keys=%s",
                len(keys),
                ("MLForecaster" in AGENT_CLASS_MAP),
                keys,
            )
        except Exception:
            pass
        self.logger.info("F501 Auto-registering agents...")
        # Snapshot & deterministic order to avoid non-deterministic boot
        agent_class_items = sorted(list(AGENT_CLASS_MAP.items()), key=lambda kv: kv[0].lower())

        if filter_types is not None:
            agent_class_items = [
                (name, cls) for name, cls in agent_class_items
                if getattr(cls, "agent_type", None) in filter_types
            ]
            self.logger.info(f"Filtered to agent types: {filter_types}")

        all_dependencies = {
            "shared_state": self.shared_state,
            "market_data": self.market_data,
            "execution_manager": self.execution_manager,
            "config": self.config,
            "tp_sl_engine": self.tp_sl_engine,
            "market_data_feed": self.market_data_feed,
            "meta_controller": self.meta_controller,
            "model_manager": self.model_manager,
            "symbol_manager": self.symbol_manager,
            "exchange_client": self.exchange_client,
            "database_manager": self.database_manager,
            "agent_schedule": self.agent_schedule,
        }
        
        # Debug: Log dependency availability
        self.logger.debug(
            "[AgentManager] Dependency check: model_manager=%s (is None: %s), "
            "meta_controller=%s, symbol_manager=%s, market_data_feed=%s",
            type(self.model_manager).__name__ if self.model_manager else "None",
            self.model_manager is None,
            type(self.meta_controller).__name__ if self.meta_controller else "None",
            type(self.symbol_manager).__name__ if self.symbol_manager else "None",
            type(self.market_data_feed).__name__ if self.market_data_feed else "None",
        )

        registered = 0
        failures = 0

        for agent_name, agent_class in agent_class_items:
            injected_args = {}
            if self.symbols:
                injected_args["symbols"] = self.symbols

            try:
                self.logger.info(f"🔍 Preparing to register agent: {agent_name}")

                sig = inspect.signature(agent_class.__init__)
                accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                accepted_params = set(sig.parameters.keys()) - {"self"}
                
                # Debug: Log the signature analysis
                if agent_name == "MLForecaster":
                    self.logger.debug(
                        f"[DEBUG] MLForecaster signature analysis: "
                        f"accepts_kwargs={accepts_kwargs}, "
                        f"params={accepted_params}"
                    )

                for name in accepted_params:
                    if name in all_dependencies:
                        injected_args[name] = all_dependencies[name]

                if accepts_kwargs:
                    for k, v in all_dependencies.items():
                        if k not in injected_args:
                            injected_args[k] = v

                if getattr(agent_class, "agent_type", "") == "strategy" and "symbols" in accepted_params:
                    symbol_list = await self._get_accepted_symbols_list()
                    injected_args["symbols"] = symbol_list
                    self.logger.info(f"[AgentManager] Preparing to register agent '{agent_name}' with {len(symbol_list)} symbols")
                
                # Enhanced logging for debugging
                has_model_mgr = "model_manager" in injected_args
                self.logger.info(
                    f"🚧 Instantiating {agent_name} with args: {sorted(list(injected_args.keys()))}"
                )
                if agent_name == "MLForecaster":
                    self.logger.info(
                        f"   [MLForecaster injection details] "
                        f"has_model_manager={has_model_mgr}, "
                        f"model_manager_value={injected_args.get('model_manager')}, "
                        f"accepts_kwargs={accepts_kwargs}, "
                        f"accepted_params={accepted_params}"
                    )
                
                agent = agent_class(**injected_args)
                self.logger.info(f"✅ Instantiated agent: {agent_name}")

                # Ensure every agent gets a meta_controller with a consistent API.
                if self.meta_controller:
                    # Wrap in adapter that guarantees submit_signal/submit_agent_signal.
                    adapted = self._MetaAdapter(self.meta_controller)
                    # If agent already had a meta_controller, replace only if it's missing the expected methods.
                    existing = getattr(agent, "meta_controller", None)
                    needs_adapter = (
                        existing is None or
                        not hasattr(existing, "submit_signal") or
                        not hasattr(existing, "submit_agent_signal")
                    )
                    if needs_adapter:
                        agent.meta_controller = adapted
                        self.logger.info(f"🔗 Injected MetaAdapter into agent '{agent_name}'")
                    else:
                        self.logger.info(f"🔗 Agent '{agent_name}' already has compatible meta_controller")
                else:
                    self.logger.warning(f"⚠️ No MetaController set; agent '{agent_name}' will not be able to emit trades.")

                self.register_agent(agent)
                num_symbols = len(getattr(agent, 'symbols', []) or [])
                if num_symbols == 0 and getattr(agent_class, "agent_type", "") == "strategy":
                    self.logger.warning(f"⚠️ CRITICAL: Strategy agent '{agent_name}' registered with 0 symbols! " 
                                       f"It will not trade until tick_all_once() populates the symbol cache.")
                    self.logger.warning(f"   This usually means discovery hasn't finalized symbols yet.")
                else:
                    self.logger.info(f"📦 Registered agent {agent_name} with {num_symbols} symbols")
                registered += 1

            except Exception as e:
                # CRITICAL: Log full error details to diagnose registration failures
                self.logger.critical(
                    f"❌ REGISTRATION FAILED FOR '{agent_name}': {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
                # Also log simplified version for easier grep
                import traceback
                tb_lines = traceback.format_exc().split('\n')
                self.logger.critical(f"[AGENT_REGISTRATION_FAILURE] {agent_name}: {tb_lines[-3] if len(tb_lines) > 2 else str(e)}")
                failures += 1

        self.logger.info(f"✅ Final Registered Agents: {list(self.agents.keys())}")
        self.logger.info(f"🧩 AgentManager registration summary: registered={registered}, failures={failures}")
        if registered == 0:
            raise RuntimeError("AgentManager: No strategy agents registered")
        
        self._strategies_prepared = True  # Set flag after successful registration

    def _ensure_optional_agent_classes(self) -> None:
        """
        Recover optional agents that may have been skipped during module import time.
        This prevents a transient import issue from permanently removing a strategy
        from AGENT_CLASS_MAP for the lifetime of the process.
        """
        self._try_lazy_register_agent(
            key="MLForecaster",
            module_path="agents.ml_forecaster",
            class_name="MLForecaster",
        )

    def _try_lazy_register_agent(self, *, key: str, module_path: str, class_name: str) -> None:
        if key in AGENT_CLASS_MAP:
            return
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is None:
                self._log_optional_import_issue(
                    key,
                    f"class '{class_name}' missing after importing {module_path}",
                    include_trace=False,
                )
                return
            AGENT_CLASS_MAP[key] = cls
            self.logger.info(
                "[AgentManager] Optional agent recovered via lazy import: %s (%s.%s)",
                key,
                module_path,
                class_name,
            )
        except Exception as e:
            self._log_optional_import_issue(
                key,
                f"lazy import failed for {module_path}.{class_name}: {e}",
                include_trace=True,
            )

    def _log_optional_import_issue(self, key: str, message: str, include_trace: bool = False) -> None:
        now_ts = time.time()
        interval = max(10.0, float(self._optional_import_log_interval_s or 300.0))
        last_ts = float(self._last_optional_import_log_ts.get(key, 0.0) or 0.0)
        if (now_ts - last_ts) < interval:
            return
        self._last_optional_import_log_ts[key] = now_ts
        if str(key) == "MLForecaster":
            prefix = "MLForecaster import failed"
        else:
            prefix = f"Optional agent unavailable: {key}"
        self.logger.warning(
            "[AgentManager] %s (%s)",
            prefix,
            message,
            exc_info=bool(include_trace),
        )

    async def warmup_all(self, concurrency: int = 6):
        """
        Best-effort warmup for all agents with bounded concurrency.
        Recognized hooks: warm_up, warmup, on_warmup, prepare, initialize.
        """
        hooks = ("warm_up", "warmup", "on_warmup", "prepare", "initialize")
        sem = _asyncio.Semaphore(concurrency)
        warmed = 0

        async def _warm_one(name, agent):
            nonlocal warmed
            async with sem:
                for h in hooks:
                    fn = getattr(agent, h, None)
                    if not callable(fn):
                        continue
                    try:
                        res = fn()
                        if inspect.isawaitable(res):
                            await res
                        self.logger.info("🔥 Warmed %s via %s()", name, h)
                        warmed += 1
                        return
                    except Exception as e:
                        self.logger.warning("Warmup %s.%s failed: %s", name, h, e, exc_info=True)
                self.logger.info("ℹ️ %s: no warmup hook found", name)

        tasks = [
            _asyncio.create_task(_warm_one(n, a))
            for n, a in sorted(self.agents.items(), key=lambda kv: kv[0].lower())
        ]
        if tasks:
            await _asyncio.gather(*tasks, return_exceptions=True)

        self._strategies_prepared = True
        self.logger.info("✅ warmup_all complete: %d/%d agents warmed", warmed, len(self.agents))

    def get_discovery_agents(self):
        """
        Returns a list of registered discovery agents.
        Merges both manual discovery registry and main agent registry to avoid
        bootstrap/observability mismatches.
        """
        merged = []
        seen_names = set()
        seen_ids = set()
        for bucket in (self.discovery_agents, list(self.agents.values())):
            for agent in bucket:
                if getattr(agent, "agent_type", None) != "discovery":
                    continue
                name = str(getattr(agent, "name", "") or "").strip()
                if name and name in seen_names:
                    continue
                aid = id(agent)
                if aid in seen_ids:
                    continue
                if name:
                    seen_names.add(name)
                seen_ids.add(aid)
                merged.append(agent)
        return merged

    def register_discovery_agent(self, agent):
        """
        Allows manual registration of discovery agents (used during Phase 3).
        This is used when you directly pass agent instances from AppContext.
        """
        if not getattr(agent, "agent_type", None):
            setattr(agent, "agent_type", "discovery")

        if not hasattr(agent, "name") or not getattr(agent, "name", None):
            setattr(agent, "name", agent.__class__.__name__)

        existing_name_index = None
        for idx, existing in enumerate(self.discovery_agents):
            if str(getattr(existing, "name", "") or "") == str(agent.name):
                existing_name_index = idx
                break

        if existing_name_index is None:
            self.discovery_agents.append(agent)
        else:
            self.discovery_agents[existing_name_index] = agent

        # Keep the canonical registry consistent so startup checks, bootstrap,
        # and task launch logic see discovery agents too.
        self.agents[agent.name] = agent
        self.logger.info(f"📥 Registered discovery agent: {agent.__class__.__name__} ({agent.name})")

    async def run_discovery_agents(self):
        """
        Runs all registered discovery agents from manual injection (Phase 3 only).
        """
        self.logger.info("🚀 Starting discovery agents (Async Tasks)...")
        for agent in self.discovery_agents:
            if hasattr(agent, "run_loop") and _asyncio.iscoroutinefunction(agent.run_loop):
                self.logger.info(f"🔄 Launching discovery agent loop: {agent.__class__.__name__}")
                # Spawn as independent tasks to avoid blocking the manager
                _asyncio.create_task(agent.run_loop(), name=f"Discovery:{agent.__class__.__name__}")
            else:
                self.logger.warning(f"⚠️ Discovery Agent {agent.__class__.__name__} has no async run_loop(). Skipping.")

    async def run_discovery_agents_once(self):
        """
        Executes each discovery agent's `run_once()` method exactly once.
        Logs how many symbols were proposed and accepted per agent.
        """
        self.logger.info("🚀 Running discovery agents once (Phase 3)...")
        
        for agent in self.discovery_agents:
            agent_name = agent.__class__.__name__
            before = set(self.shared_state.symbol_proposals.keys()) 
            
            try:
                if hasattr(agent, "run_once") and _asyncio.iscoroutinefunction(agent.run_once):
                    await agent.run_once()
                    after = set(self.shared_state.symbol_proposals.keys())
                    proposed = after - before
                    self.logger.info(f"✅ {agent_name} proposed {len(proposed)} new symbols: {list(proposed)}")
                else:
                    self.logger.warning(f"⚠️ Discovery Agent {agent.__class__.__name__} does not have an async run_once() method. Skipping.")
            except Exception as e:
                self.logger.exception(f"❌ {agent_name} failed to run_once(): {e}")

    async def run_all_agents(self):
        """
        Waits for market data readiness, then launches agent tasks.
        """
        self.logger.info("⏳ Waiting for market data to be ready before running agents...")
        ready_event = getattr(self.shared_state, "market_data_ready_event", None)
        if ready_event and hasattr(ready_event, "wait"):
            try:
                await _asyncio.wait_for(ready_event.wait(), timeout=self._market_data_ready_timeout_s)
            except _asyncio.TimeoutError:
                self.logger.warning("⚠️ Market data readiness timed out after %.1fs — proceeding anyway.", self._market_data_ready_timeout_s)
        else:
            self.logger.info("ℹ️ No market_data_ready_event found; proceeding without wait.")

        # Behavioral Change 2: Allocation must precede symbol activation (Point 2)
        # We wait for OpsPlaneReady or at least an authoritative budget before starting trading agents.
        self.logger.info("⏳ Waiting for Capital Allocation (OpsPlaneReady) before activating agents...")
        ops_ready = getattr(self.shared_state, "ops_plane_ready_event", None)
        if ops_ready:
            try:
                ops_timeout = float(getattr(self.config, "AGENTMGR_OPS_READY_TIMEOUT_S", 30.0))
                await _asyncio.wait_for(ops_ready.wait(), timeout=ops_timeout)
                self.logger.info("✅ Capital assigned; proceeding with agent activation.")
            except _asyncio.TimeoutError:
                self.logger.warning("⚠️ OpsPlaneReady timed out; checking if we have any budget anyway.")
                if not hasattr(self.shared_state, "is_ops_plane_ready") or not self.shared_state.is_ops_plane_ready():
                    self.logger.warning("❌ No budget detected; agents will launch but may remain idle.")

        self.logger.info("🚀 Starting agents...")
        sem = _asyncio.Semaphore(self._max_start_concurrency)

        async def _start(name: str, agent: Any):
            async with sem:
                task = _asyncio.create_task(self._agent_entry(agent, name), name=f"Agent:{name}")
                self._tasks[name] = task
                return task

        # Launch tasks (Discovery agents ONLY)
        for n, a in sorted(self.agents.items(), key=lambda kv: kv[0].lower()):
            agent_type = getattr(a, "agent_type", None)
            
            # ARCHITECTURAL FIX: Strategy agents are TICK-DRIVEN.
            # They MUST NOT be launched as background tasks to avoid double-execution.
            if agent_type == "strategy":
                # ISSUE 3: Enforce strategy agent contract early (Startup Sanity)
                if not hasattr(a, "generate_signals"):
                    self.logger.error(f"❌ [AgentManager] Strategy agent '{n}' is MISSING generate_signals(). It will never trade.")
                else:
                    self.logger.info(f"✅ [AgentManager] Registered strategy agent: {n} (tick-driven)")
                continue
                
            await _start(n, a)

        self.logger.info("✅ Discovery agent tasks launched. Strategy agents are registered for ticking.")
        # Intentionally do NOT await agent tasks here.

    async def tick_all_once(self):
        """Central tick: refresh symbols, apply budget gating, prepare strategy agents."""
        # FIX 3: Force symbol refresh into ALL agents (critical)
        try:
            now = time.time()
            if (now - self._last_symbols_refresh_t) > 60.0:
                snap_list = await self._get_accepted_symbols_list()
                snap_list = sorted(list(set(snap_list)))  # De-duplicate and sort
                # CRITICAL FIX: Update if changed OR if cache is empty (initial bootstrap)
                if snap_list != self._accepted_symbols_cache or not self._accepted_symbols_cache:
                    if snap_list and snap_list != self._accepted_symbols_cache:
                        self.logger.info("🔄 Symbol universe changed: %d -> %d symbols. Notifying agents.",
                                         len(self._accepted_symbols_cache), len(snap_list))
                    elif snap_list and not self._accepted_symbols_cache:
                        self.logger.info("🔄 ⚡ INITIAL symbol cache population: %d symbols now available.",
                                         len(snap_list))
                    
                    self._accepted_symbols_cache = snap_list
                    self._last_symbols_refresh_t = now
                    
                    # CRITICAL FIX: FORCE symbol refresh into ALL agents (especially on first non-empty set)
                    for agent_obj in self.agents.values():
                        if hasattr(agent_obj, "load_symbols") and _asyncio.iscoroutinefunction(agent_obj.load_symbols):
                            await agent_obj.load_symbols(self._accepted_symbols_cache)
                        elif hasattr(agent_obj, "symbols"):
                            agent_obj.symbols = self._accepted_symbols_cache
                            # ISSUE 2: Log symbol visibility per agent
                            self.logger.debug("[%s] Injected %d symbols", agent_obj.__class__.__name__, len(self._accepted_symbols_cache))
        except Exception as e:
            self.logger.debug("Symbol refresh check failed: %s", e)

        # ISSUE 1 FIX: Do NOT call generate_signals here - collect_and_forward_signals does it
        # This tick prepares agents (symbol refresh, readiness checks) but doesn't execute them
        # Signal generation happens in collect_and_forward_signals() to avoid double execution
        shared_wallet_mode_raw = getattr(self.config, "CAPITAL_ALLOCATOR_SHARED_WALLET", True)
        if isinstance(shared_wallet_mode_raw, str):
            shared_wallet_mode = shared_wallet_mode_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            shared_wallet_mode = bool(shared_wallet_mode_raw)
        shared_wallet_spendable = 0.0
        if shared_wallet_mode and hasattr(self.shared_state, "get_spendable_balance"):
            try:
                quote_asset = str(getattr(self.config, "QUOTE_ASSET", "USDT") or "USDT").upper()
                spendable_res = self.shared_state.get_spendable_balance(quote_asset)
                if inspect.isawaitable(spendable_res):
                    spendable_res = await spendable_res
                shared_wallet_spendable = float(spendable_res or 0.0)
            except Exception as e:
                self.logger.debug("[AgentManager] Failed to read shared spendable budget: %s", e)
        
        for name, agent_obj in self.agents.items():
            try:
                agent_type = getattr(agent_obj, "agent_type", None)
                if agent_type == "discovery":
                    continue  # Discovery agents are self-managing

                # ARCHITECTURAL FIX: Budget gating moved to MetaController.
                # Agents are prepared (symbols synced) regardless of budget to allow exits.
                raw_budget = 0.0
                if hasattr(self.shared_state, "get_authoritative_reservation"):
                    raw_budget = float(self.shared_state.get_authoritative_reservation(name))
                budget = float(raw_budget)
                if budget <= 0.0 and shared_wallet_mode and shared_wallet_spendable > 0.0:
                    budget = float(shared_wallet_spendable)

                # ISSUE 2: Throttled symbol visibility logging (once per minute)
                now_t = time.time()
                last_log = self._last_agent_log_t.get(name, 0.0)
                symbol_count = len(getattr(agent_obj, "symbols", []))
                
                if (now_t - last_log) > 60.0:
                    status = "Active" if budget > 0 else "Active (Exit-Only/ZeroBudget)"
                    if raw_budget <= 0.0 < budget:
                        self.logger.info(
                            "[Agent:%s] %s with %d symbols (shared wallet fallback=%.2f)",
                            name,
                            status,
                            symbol_count,
                            budget,
                        )
                    else:
                        self.logger.info("[Agent:%s] %s with %d symbols", name, status, symbol_count)
                    self._last_agent_log_t[name] = now_t

                # ISSUE 3: Enforce strategy agent contract
                if not hasattr(agent_obj, "generate_signals"):
                    self.logger.warning("[%s] Missing generate_signals() - strategy agents MUST implement this", name)
                
            except Exception as e:
                self.logger.warning("[%s] Tick preparation failed: %s", name, e)

    async def run(self):
        """
        Launch all agents that have a `run()` coroutine method.
        Useful in Phase 7 when agents are expected to operate continuously.
        """
        await update_health(self.shared_state, "AgentManager", "Healthy", "Strategy agent loop running.")
        # Keep previous run() API but delegate to run_all_agents for consistency
        await self.run_all_agents()

    # CLEANUP: _safe_run() removed - dead code after centralized ticking refactor

    async def _agent_entry(self, agent: Any, name: str):
        """
        Entry wrapper with optional auto-restart on crash and a bounded startup timeout.
        """
        try:
            origin = inspect.getfile(agent.__class__)
        except Exception:
            origin = "<unknown>"
        self.logger.info("Launching agent: %s (%s from %s)", name, agent.__class__.__name__, origin)

        # Point 2: Continuous Budget Gating
        # If an agent has no budget, it should not churn ML / generate signals.
        async def _check_budget():
            if hasattr(self.shared_state, "get_authoritative_reservation"):
                budget = float(self.shared_state.get_authoritative_reservation(name))
                if budget <= 0:
                    shared_wallet_mode_raw = getattr(self.config, "CAPITAL_ALLOCATOR_SHARED_WALLET", True)
                    if isinstance(shared_wallet_mode_raw, str):
                        shared_wallet_mode = shared_wallet_mode_raw.strip().lower() in {"1", "true", "yes", "on"}
                    else:
                        shared_wallet_mode = bool(shared_wallet_mode_raw)
                    if shared_wallet_mode and hasattr(self.shared_state, "get_spendable_balance"):
                        try:
                            quote_asset = str(getattr(self.config, "QUOTE_ASSET", "USDT") or "USDT").upper()
                            spendable = self.shared_state.get_spendable_balance(quote_asset)
                            if inspect.isawaitable(spendable):
                                spendable = await spendable
                            if float(spendable or 0.0) > 0.0:
                                return True
                        except Exception as e:
                            self.logger.debug("[Agent:%s] Shared wallet budget probe failed: %s", name, e)
                    self.logger.debug("[Agent:%s] Idle: No authoritative budget allocated.", name)
                    return False
            return True

        async def _once():
            # Apply dynamic budget gating before execution (Point 2)
            if not await _check_budget():
                 await _asyncio.sleep(10) # check again later
                 return None

            # ARCHITECTURAL FIX: Only discovery agents are allowed to have a persistent background task.
            agent_type = getattr(agent, "agent_type", None)
            if agent_type == "discovery":
                if hasattr(agent, "run_loop") and _asyncio.iscoroutinefunction(agent.run_loop):
                    return await agent.run_loop()  # long-running; no timeout
                if hasattr(agent, "run") and _asyncio.iscoroutinefunction(agent.run):
                    return await agent.run()
                if hasattr(agent, "run_once") and _asyncio.iscoroutinefunction(agent.run_once):
                    return await _asyncio.wait_for(agent.run_once(), timeout=self._agent_timeout_s)
            
            # If we are here, it's either a strategy agent (which shouldn't have reached here)
            # or a discovery agent with no supported entry point.
            self.logger.debug("[AgentManager:_agent_entry] Agent %s type=%s has no background loop; exiting task.", name, agent_type)
            return None

        if not self._restart_on_crash:
            return await _once()

        backoff = self._restart_backoff_min
        while True:
            try:
                return await _once()
            except _asyncio.TimeoutError:
                self.logger.warning("⏰ Agent %s timed out (entry or run_once). Backoff %.1fs.", name, backoff)
            except _asyncio.CancelledError:
                self.logger.info("🛑 Agent %s cancelled; exiting restart loop.", name)
                raise
            except Exception as e:
                self.logger.error("🔥 Agent %s crashed: %s — restarting in %.1fs", name, e, backoff, exc_info=True)
            await _asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._restart_backoff_max)

    async def report_health_loop(self):
        """
        Periodically reports the health of the AgentManager component.
        This is a separate, persistent task to prevent the watchdog from marking the component as degraded
        if the main run loop is blocked for an extended period.
        """
        try:
            while True:
                ComponentStatusLogger.log_status(
                    component="AgentManager",
                    status="Healthy",
                    detail=f"Heartbeat ping at {datetime.now().isoformat()}"
                )
                await _asyncio.sleep(getattr(self.config, "AGENT_HEALTH_INTERVAL", 30))
        except _asyncio.CancelledError:
            self.logger.info("AgentManager.health loop cancelled.")
            raise
 
    async def run_discovery_agents_loop(self):
        """
        Periodically runs discovery agents to populate the symbol universe.
        This unblocks SymbolManager which waits for these proposals.
        """
        self.logger.info("Starting Discovery Agent loop...")
        # Launch persistent discovery agent coroutines once (outside the retry loop).
        try:
            await self.run_discovery_agents()
        except Exception as e:
            self.logger.error("run_discovery_agents failed on startup: %s", e, exc_info=True)

        # Outer retry loop: a crash in run_discovery_agents_once must NOT exit this method.
        while True:
            try:
                await self.run_discovery_agents_once()
                # Run every 10 minutes or as configured
                discovery_interval = float(getattr(self.config, "AGENTMGR_DISCOVERY_INTERVAL", 600.0))
                await _asyncio.sleep(discovery_interval)
            except _asyncio.CancelledError:
                self.logger.info("Discovery loop cancelled.")
                raise
            except Exception as e:
                self.logger.error("Discovery loop crashed: %s — retrying in 30s", e, exc_info=True)
                # Short backoff before retry; stay in the loop.
                await _asyncio.sleep(30)

    async def _tick_loop(self):  # New method for continuous ticking
        self._strategies_started = True  # Set flag when the loop starts
        self.logger.warning("🔥 [AgentManager:TICK] ✅ TICK LOOP STARTED - will collect signals every %d seconds", getattr(self.config, "AGENT_TICK_SEC", 5))
        tick_count = 0
        try:
            while True:
                try:
                    tick_count += 1
                    self.logger.debug("[AgentManager:TICK] Iteration #%d: tick_all_once", tick_count)
                    await self.tick_all_once()                 # agents do their work
                    self.logger.debug("[AgentManager:TICK] Iteration #%d: collect_and_forward_signals", tick_count)
                    await self.collect_and_forward_signals()   # NEW: forward to Meta
                except _asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("AgentManager.tick loop iteration #%d failed: %s", tick_count, e, exc_info=True)
                await _asyncio.sleep(getattr(self.config, "AGENT_TICK_SEC", 5))  # Use AGENT_TICK_SEC from config
        except _asyncio.CancelledError:
            self.logger.info("AgentManager.tick loop cancelled after %d iterations.", tick_count)
            raise

    async def run_strategy_retrain_loop(self):
        """
        Phase 9 async retrain loop for tick-driven strategy agents.
        Agents opt-in by implementing async `retrain()` (or sync callable).
        """
        interval = float(self._strategy_retrain_interval_s or 0.0)
        if interval <= 0:
            self.logger.info("[Phase9:Retrain] Strategy retrain loop disabled (interval <= 0).")
            return

        timeout_s = max(60.0, float(self._strategy_retrain_timeout_s or 3600.0))
        strategy_total = sum(1 for _n, _a in self.agents.items() if getattr(_a, "agent_type", None) == "strategy")
        retrain_capable = [
            _n for _n, _a in self.agents.items()
            if getattr(_a, "agent_type", None) == "strategy" and callable(getattr(_a, "retrain", None))
        ]
        self.logger.info(
            "[Phase9:Retrain] Strategy retrain loop started (interval=%.1fs timeout=%.1fs strategies=%d retrain_capable=%d names=%s).",
            interval,
            timeout_s,
            strategy_total,
            len(retrain_capable),
            retrain_capable,
        )
        if strategy_total > 0 and not retrain_capable:
            self.logger.warning(
                "[Phase9:Retrain] No strategy agents expose retrain(); loop will stay idle."
            )

        try:
            while True:
                loop_start = time.time()
                for name, agent in sorted(self.agents.items(), key=lambda kv: kv[0].lower()):
                    if getattr(agent, "agent_type", None) != "strategy":
                        continue
                    retrain_fn = getattr(agent, "retrain", None)
                    if not callable(retrain_fn):
                        continue

                    self.logger.info("[Phase9:Retrain] Retrain start agent=%s", name)
                    try:
                        res = retrain_fn()
                        if inspect.isawaitable(res):
                            res = await _asyncio.wait_for(res, timeout=timeout_s)
                        self.logger.info(
                            "[Phase9:Retrain] Retrain finish agent=%s status=ok result=%s",
                            name,
                            str(res)[:500],
                        )
                    except _asyncio.TimeoutError:
                        self.logger.warning(
                            "[Phase9:Retrain] Retrain finish agent=%s status=timeout timeout=%.1fs",
                            name,
                            timeout_s,
                        )
                    except _asyncio.CancelledError:
                        self.logger.warning(
                            "[Phase9:Retrain] Retrain finish agent=%s status=cancelled",
                            name,
                        )
                        raise
                    except Exception as e:
                        self.logger.warning(
                            "[Phase9:Retrain] Retrain finish agent=%s status=error err=%s",
                            name,
                            e,
                            exc_info=True,
                        )

                elapsed = max(0.0, time.time() - loop_start)
                # Sleep the remainder of the configured interval; minimum 1s guard.
                sleep_for = max(1.0, interval - elapsed)
                if elapsed > interval:
                    self.logger.warning(
                        "[Phase9:Retrain] Cycle overran interval: elapsed=%.1fs interval=%.1fs",
                        elapsed,
                        interval,
                    )
                await _asyncio.sleep(sleep_for)
        except _asyncio.CancelledError:
            self.logger.info("AgentManager.strategy_retrain loop cancelled.")
            raise

    async def run_loop(self, stop_event: Optional['_asyncio.Event'] = None):
        """
        Phase 9 compatibility: unblocked orchestration of manager tasks.
        """
        if any(not t.done() for t in self._manager_tasks.values()):
            self.logger.warning("AgentManager run_loop already active; skipping duplicate start.")
            return
        self.logger.info("🚀 AgentManager run_loop started (Unblocked Mode).")
        
        # schedule manager tasks so stop() can cancel them
        # 🔥 CRITICAL FIX: Create tick task FIRST and ensure it starts immediately
        # This unblocks the tick loop from waiting for other tasks to complete
        self._manager_tasks["tick"] = _asyncio.create_task(self._tick_loop(), name="AgentManager:tick")
        self.logger.info("🔥 [AgentManager] Tick loop scheduled - signal collection will begin immediately")
        
        self._manager_tasks["discovery"] = _asyncio.create_task(self.run_discovery_agents_loop(), name="AgentManager:discovery")
        self._manager_tasks["run_all_agents"] = _asyncio.create_task(self.run_all_agents(), name="AgentManager:run_all_agents")
        self._manager_tasks["health"] = _asyncio.create_task(self.report_health_loop(), name="AgentManager:health")
        if float(self._strategy_retrain_interval_s or 0.0) > 0:
            self._manager_tasks["strategy_retrain"] = _asyncio.create_task(
                self.run_strategy_retrain_loop(),
                name="AgentManager:strategy_retrain",
            )
        # Diagnostic hooks: make unexpected task exits visible in logs.
        for task_name, task in self._manager_tasks.items():
            def _mk_cb(_name: str):
                def _done_cb(t: _asyncio.Task):
                    try:
                        exc = t.exception()
                    except _asyncio.CancelledError:
                        self.logger.warning("AgentManager manager task cancelled: %s", _name)
                        return
                    except Exception as cb_err:
                        self.logger.warning(
                            "AgentManager manager task callback failed for %s: %s",
                            _name,
                            cb_err,
                            exc_info=True,
                        )
                        return
                    if exc is not None:
                        self.logger.error(
                            "AgentManager manager task failed: %s err=%s",
                            _name,
                            exc,
                            exc_info=True,
                        )
                    elif _name != "run_all_agents":
                        self.logger.warning("AgentManager manager task exited unexpectedly: %s", _name)
                return _done_cb
            task.add_done_callback(_mk_cb(task_name))

        ComponentStatusLogger.log_status(
            component="AgentManager",
            status="Healthy",
            detail=f"AgentManager tasks scheduled: {list(self._manager_tasks.keys())}"
        )

        # await everything
        try:
            await _asyncio.gather(*self._manager_tasks.values())
        except _asyncio.CancelledError:
            self.logger.info("AgentManager run_loop cancelled.")
            raise
        except Exception as e:
            self.logger.error("AgentManager critical manager task failure: %s", e, exc_info=True)
        finally:
            await self.stop()

    async def _get_accepted_symbols_list(self) -> List[str]:
        """
        Snapshot accepted symbols using the full active definition (Accepted + Held).
        This ensures agents like DipSniper can see positions even if they aren't in the 'accepted' list.
        """
        if hasattr(self.shared_state, "get_analysis_symbols"):
            # P9 FIX: Ask for EVERYTHING (limit=0) via robust helper
            return self.shared_state.get_analysis_symbols()

        # Fallback for legacySharedState
        snap_fn = getattr(self.shared_state, "get_accepted_symbols_snapshot", None)
        if callable(snap_fn):
            res = snap_fn()
            if inspect.isawaitable(res):
                res = await res
        else:
            get_fn = getattr(self.shared_state, "get_accepted_symbols", None)
            if callable(get_fn):
                maybe = get_fn()
                res = await maybe if inspect.isawaitable(maybe) else maybe
            else:
                res = getattr(self.shared_state, "symbols", {})
        if isinstance(res, dict):
            return list(res.keys())
        return list(res or [])

    def get_agent(self, name: str):
        """
        Retrieve a registered agent by its name.
        """
        return self.agents.get(name)

    def get_agents(self) -> Dict[str, Any]:
        """
        Return a dictionary of all registered agents by name.
        This is required by AppContext and StrategyManager.
        """
        return self.agents

    async def __call__(self):
        """Alias so the manager can be scheduled like a coroutine."""
        self.logger.info("🚀 AgentManager started in Phase 9.")
        self._started = True
        await self.run_loop()

    async def get_health(self):
        """
        Returns a dictionary representing the health status of the AgentManager.
        """
        try:
            running_agent_tasks = [n for n, t in self._tasks.items() if not t.done()]
            running_mgr_tasks = [n for n, t in self._manager_tasks.items() if not t.done()]
            return {
                "status": "Operational" if getattr(self, "_started", False) else "Initializing",
                "detail": f"agents={len(self.agents)}",
                "agents": list(self.agents.keys()),
                "running_agent_tasks": running_agent_tasks,
                "running_manager_tasks": running_mgr_tasks,
            }
        except Exception as e:
            self.logger.error("Error getting AgentManager health: %s", e, exc_info=True)
            return {"status": "Error", "detail": f"health-exception: {e}"}

    async def stop(self):
        """Gracefully cancel all agent tasks started by the manager."""
        try:
            # cancel agents
            for name, t in list(self._tasks.items()):
                if not t.done():
                    t.cancel()
            if self._tasks:
                await _asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()

            # cancel manager loops (health, tick, run_all_agents)
            for name, t in list(self._manager_tasks.items()):
                if not t.done():
                    t.cancel()
            if self._manager_tasks:
                await _asyncio.gather(*self._manager_tasks.values(), return_exceptions=True)
            self._manager_tasks.clear()

            self.logger.info("🧹 AgentManager stopped all tasks.")
        except _asyncio.CancelledError:
            # stopping while already stopping — not an error
            self.logger.info("AgentManager.stop() cancelled during shutdown.")
            raise
        except Exception:
            self.logger.exception("AgentManager.stop() failed")
            raise


# ===== AgentManager helpers =====

def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# CLEANUP: _emit_health, _wait_phase_gates, _is_fresh_intent removed — dead code, never called.
