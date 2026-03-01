"""
Legacy execution wrapper.

This module intentionally does not place orders directly.
All execution is delegated to MetaController._execute_decision so that
canonical policy/risk/liquidation routing remains centralized.
"""

import asyncio
import time
from collections import deque, defaultdict
from typing import Any, Dict, Optional, Set


class ExecutionLogic:
    def __init__(self, shared_state, execution_manager, config, logger, meta_controller):
        self.shared_state = shared_state
        self.execution_manager = execution_manager  # kept for compatibility; not used directly here
        self.config = config
        self.logger = logger
        self.meta_controller = meta_controller

        # Keep legacy counters/state for compatibility with existing references.
        self._trade_timestamps = deque(maxlen=1000)
        self._trade_timestamps_sym = defaultdict(lambda: deque(maxlen=100))
        self._trade_timestamps_agent = defaultdict(lambda: deque(maxlen=100))
        self._max_trades_per_hour = int(getattr(config, "MAX_TRADES_PER_HOUR", 10))
        self._max_trades_per_day = int(getattr(config, "MAX_TRADES_PER_DAY", 0) or 0)
        self._trade_timestamps_day = deque(maxlen=5000)
        self._last_buy_ts = {}

    async def execute_decision(self, intent: dict) -> dict:
        """
        Compatibility entrypoint that accepts an intent payload and routes it
        through the canonical MetaController decision executor.
        """
        try:
            action = str(intent.get("action", intent.get("side", ""))).upper()
            symbol = self._normalize_symbol(intent.get("symbol", ""))

            if action not in {"BUY", "SELL"} or not symbol:
                return {"ok": False, "status": "REJECTED", "reason": "invalid_intent"}

            sig = {
                "action": action,
                "side": action,
                "symbol": symbol,
                "confidence": float(intent.get("confidence", 1.0) or 1.0),
                "agent": intent.get("agent", "Meta"),
                "quote": intent.get("quote", intent.get("planned_quote", intent.get("quote_hint"))),
                "quantity": intent.get("quantity", intent.get("qty", intent.get("qty_hint"))),
                "planned_quote": intent.get("planned_quote", intent.get("quote", intent.get("quote_hint"))),
                "tag": intent.get("tag", f"strategy/{intent.get('agent', 'Meta')}"),
                "bypass_conf": bool(intent.get("bypass_conf", True)),
                "reason": intent.get("reason", intent.get("rationale", "")),
                "ts": float(intent.get("ts", time.time()) or time.time()),
                "ttl_sec": float(intent.get("ttl_sec", 30.0) or 30.0),
            }

            acc_getter = getattr(self.shared_state, "get_accepted_symbols_snapshot", None)
            accepted = {symbol}
            if callable(acc_getter):
                acc = await self._safe_await(acc_getter())
                if acc:
                    accepted = set(acc)

            result = await self._execute_decision(symbol, action, sig, accepted)
            if isinstance(result, dict):
                return result
            return {"ok": bool(result), "status": "ACCEPTED" if result else "SKIPPED"}
        except Exception as e:
            try:
                self.logger.error("execute_decision failed: %s", e, exc_info=True)
            except Exception:
                pass
            return {"ok": False, "status": "REJECTED", "reason": "internal_error"}

    def _extract_decision_id(self, signal: dict) -> Optional[str]:
        if not isinstance(signal, dict):
            return None
        for key in (
            "decision_id",
            "decisionId",
            "signal_id",
            "signalId",
            "id",
            "cache_key",
            "intent_id",
            "intentId",
        ):
            val = signal.get(key)
            if val:
                return str(val)
        return None

    async def _execute_decision(self, symbol: str, side: str, signal: dict, accepted_symbols_set: Set[str]):
        """
        Strict Meta routing:
        this wrapper never calls ExecutionManager.execute_trade/close_position directly.
        """
        symbol_u = self._normalize_symbol(symbol)
        side_u = str(side or "").upper()
        if side_u not in {"BUY", "SELL"} or not symbol_u:
            return {"ok": False, "status": "rejected", "reason": "invalid_decision"}

        meta = getattr(self, "meta_controller", None)
        exec_fn = getattr(meta, "_execute_decision", None)
        if not callable(exec_fn):
            self.logger.error(
                "[ExecutionLogic] MetaController._execute_decision unavailable; cannot route %s %s",
                side_u,
                symbol_u,
            )
            return {"ok": False, "status": "error", "reason": "meta_execute_unavailable"}

        sig = dict(signal or {})
        sig["symbol"] = symbol_u
        sig["action"] = side_u
        sig["side"] = side_u
        sig.setdefault("agent", "Meta")

        try:
            return await exec_fn(symbol_u, side_u, sig, set(accepted_symbols_set or {symbol_u}))
        except Exception as e:
            self.logger.error(
                "[ExecutionLogic] Delegated Meta execution failed for %s %s: %s",
                side_u,
                symbol_u,
                e,
                exc_info=True,
            )
            return {"ok": False, "status": "error", "reason": "meta_execution_failed"}

    async def _safe_await(self, maybe_coro: Any):
        if asyncio.iscoroutine(maybe_coro):
            return await maybe_coro
        return maybe_coro

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").upper().replace("/", "").strip()
