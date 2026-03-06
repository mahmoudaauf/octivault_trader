
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

        # Only mark started after registration has succeeded so retry is possible.
        self._started = True

        if hasattr(self, "run_loop"):
            _asyncio.create_task(self.run_loop(), name="AgentManager:run_loop")
        elif hasattr(self, "run"):
            _asyncio.create_task(self.run(), name="AgentManager:run")
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
        if not raw:
            self.logger.warning("[AgentManager:NORMALIZE] Empty/None raw signals from %s", agent_name)
            return intents
        if isinstance(raw, dict):
            raw = [raw]
        elif not isinstance(raw, (list, tuple, set)):
            raw = [raw]
        self.logger.warning("[AgentManager:NORMALIZE] Normalizing %d raw signals from %s", len(raw), agent_name)
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
            for k in passthrough_keys:
                if k in s:
                    intent[k] = s[k]
            # If agent already built a policy_context, keep it
            if isinstance(s.get("policy_context"), dict):
                intent["policy_context"] = dict(s["policy_context"])
            intents.append(intent)
        if raw and not intents:
            self.logger.warning(
                "[_normalize_to_intents] Agent '%s' provided %d raw signals but NONE passed normalization. First item: %s",
                agent_name, len(raw), raw[0] if raw else "N/A",
            )
        if intents:
            self.logger.warning("[AgentManager:NORMALIZE] ✓ Successfully normalized %d intents from %s", len(intents), agent_name)
        return intents

    async def collect_and_forward_signals(self):
        """Single signal collection point - calls generate_signals() once per tick."""
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
            await self.submit_trade_intents(batch)
            self.logger.info("Submitted %d TradeIntents to Meta", len(batch))
            
            # 🔥 CRITICAL DEBUG: Log submission
            self.logger.warning("[AgentManager:BATCH] Submitted batch of %d intents: %s", 
                               len(batch),
                               [f"{i.get('agent')}:{i.get('symbol')}" for i in batch])
            
            # 🔥 CRITICAL FIX: DIRECT PATH TO METACONTROLLER
            # Don't wait for event bus drain - forward signals directly to MetaController
            # This ensures signals reach the signal_cache IMMEDIATELY
            if self.meta_controller:
                direct_count = 0
                for intent in batch:
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
                self.logger.info(f"🚧 Instantiating {agent_name} with args: {sorted(list(injected_args.keys()))}")
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
                self.logger.info(f"📦 Registered agent {agent_name} with {num_symbols} symbols")
                registered += 1

            except Exception as e:
                self.logger.warning(f"❌ Failed to register agent '{agent_name}': {str(e)}", exc_info=True)
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
        Assumes discovery agents have an 'agent_type' attribute set to 'discovery'.
        """
        return [
            agent for agent in self.agents.values()
            if getattr(agent, "agent_type", None) == "discovery"
        ]

    def register_discovery_agent(self, agent):
        """
        Allows manual registration of discovery agents (used during Phase 3).
        This is used when you directly pass agent instances from AppContext.
        """
        self.discovery_agents.append(agent)
        self.logger.info(f"📥 Registered discovery agent: {agent.__class__.__name__}")

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
                if snap_list != self._accepted_symbols_cache:
                    self.logger.info("🔄 Symbol universe changed: %d -> %d symbols. Notifying agents.",
                                     len(self._accepted_symbols_cache), len(snap_list))
                    self._accepted_symbols_cache = snap_list
                    self._last_symbols_refresh_t = now
                    # FIX 3: FORCE symbol refresh into ALL agents
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
        
        for name, agent_obj in self.agents.items():
            try:
                agent_type = getattr(agent_obj, "agent_type", None)
                if agent_type == "discovery":
                    continue  # Discovery agents are self-managing

                # ARCHITECTURAL FIX: Budget gating moved to MetaController.
                # Agents are prepared (symbols synced) regardless of budget to allow exits.
                budget = 0.0
                if hasattr(self.shared_state, "get_authoritative_reservation"):
                    budget = float(self.shared_state.get_authoritative_reservation(name))

                # ISSUE 2: Throttled symbol visibility logging (once per minute)
                now_t = time.time()
                last_log = self._last_agent_log_t.get(name, 0.0)
                symbol_count = len(getattr(agent_obj, "symbols", []))
                
                if (now_t - last_log) > 60.0:
                    status = "Active" if budget > 0 else "Active (Exit-Only/ZeroBudget)"
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
