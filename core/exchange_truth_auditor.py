from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple


class ExchangeTruthAuditor:
    """
    Governance-only reconciliation layer between ExchangeClient and SharedState.

    Responsibilities:
    - Reconcile exchange balances vs in-memory positions.
    - Recover missed FILLED orders (idempotent).
    - Close phantom positions when exchange inventory is gone.
    - Repair open-order mirrors from exchange truth.
    - Run restart-safe bootstrap reconciliation.

    Non-responsibilities:
    - Never places orders.
    - Never makes strategy/risk decisions.
    """

    def __init__(
        self,
        config: Any,
        logger: Optional[logging.Logger] = None,
        shared_state: Optional[Any] = None,
        exchange_client: Optional[Any] = None,
        app: Optional[Any] = None,
        **_: Any,
    ) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("ExchangeTruthAuditor")
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.app = app

        self.interval_sec = float(self._cfg("TRUTH_AUDIT_INTERVAL_SEC", 5.0) or 5.0)
        self.order_history_limit = int(self._cfg("TRUTH_AUDIT_ORDER_LIMIT", 50) or 50)
        self.max_symbols_per_cycle = int(self._cfg("TRUTH_AUDIT_SYMBOLS_PER_CYCLE", 24) or 24)
        self.recovery_window_sec = int(self._cfg("TRUTH_AUDIT_RECOVERY_WINDOW_SEC", 6 * 3600) or (6 * 3600))
        self.trade_match_window_sec = float(self._cfg("TRUTH_AUDIT_TRADE_MATCH_WINDOW_SEC", 180.0) or 180.0)
        self.event_scan_limit = int(self._cfg("TRUTH_AUDIT_EVENT_SCAN_LIMIT", 3000) or 3000)
        self.position_mismatch_tol = float(self._cfg("TRUTH_AUDIT_POSITION_MISMATCH_TOL", 1e-8) or 1e-8)
        self.phantom_cooldown_sec = float(self._cfg("TRUTH_AUDIT_PHANTOM_COOLDOWN_SEC", 20.0) or 20.0)
        self.dust_threshold = float(self._cfg("DUST_POSITION_QTY", 0.00001) or 0.00001)
        self.seen_ttl_sec = float(self._cfg("TRUTH_AUDIT_SEEN_TTL_SEC", 24 * 3600) or (24 * 3600))
        self.sell_finalize_gap_warn_threshold = int(
            self._cfg("TRUTH_AUDIT_SELL_FINALIZE_GAP_WARN_THRESHOLD", 1) or 1
        )
        self.sell_finalize_warn_cooldown_sec = float(
            self._cfg("TRUTH_AUDIT_SELL_FINALIZE_WARN_COOLDOWN_SEC", 60.0) or 60.0
        )

        self.execution_manager: Optional[Any] = None
        self.trade_history_limit = int(self._cfg("TRUTH_AUDIT_TRADE_LIMIT", 50) or 50)

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._started_at = time.time()
        self._cursor = 0
        self._seen_order_ids: Dict[str, float] = {}
        self._seen_trade_ids: Dict[str, float] = {}
        self._recent_phantom_by_symbol: Dict[str, float] = {}
        self._last_signed_skip_log_ts = 0.0
        self._last_sell_finalize_warn_ts = 0.0
        self._last_sell_finalize_snapshot: Dict[str, int] = {
            "sell_finalize_fills_seen": 0,
            "sell_finalize_finalized": 0,
            "sell_finalize_pending": 0,
            "sell_finalize_duplicate": 0,
            "sell_finalize_pending_timeout": 0,
            "sell_finalize_gap": 0,
        }

        # Control-plane supervision (user-data feed health + sync debounce)
        self.user_data_health_interval_sec = float(
            self._cfg("USER_DATA_HEALTH_CHECK_SEC", 5.0) or 5.0
        )
        self.user_data_health_interval_sec = min(max(self.user_data_health_interval_sec, 1.0), 30.0)
        self.user_data_soft_gap_sec = float(self._cfg("USER_DATA_SOFT_GAP_SEC", 20.0) or 20.0)
        self.user_data_hard_gap_sec = float(self._cfg("USER_DATA_HARD_GAP_SEC", 60.0) or 60.0)
        if self.user_data_hard_gap_sec <= self.user_data_soft_gap_sec:
            self.user_data_hard_gap_sec = self.user_data_soft_gap_sec + 1.0
        self.force_sync_cooldown_sec = float(self._cfg("FORCE_SYNC_COOLDOWN_SEC", 15.0) or 15.0)
        self.open_order_verify_interval_sec = float(
            self._cfg("OPEN_ORDER_VERIFY_INTERVAL_SEC", 7.0) or 7.0
        )
        self.open_order_verify_interval_sec = min(max(self.open_order_verify_interval_sec, 5.0), 10.0)

        self.last_successful_force_sync_ts = 0.0
        self._last_user_data_health_snapshot: Dict[str, Any] = {}
        self._force_sync_lock: Optional[asyncio.Lock] = None
        self._user_data_health_task: Optional[asyncio.Task] = None
        self._open_order_verify_task: Optional[asyncio.Task] = None

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            if isinstance(self.config, dict):
                return self.config.get(key, default)
            return getattr(self.config, key, default)
        except Exception:
            return default

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _maybe_call(self, obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(obj, method, None)
        if not callable(fn):
            return None
        return await self._maybe_await(fn(*args, **kwargs))

    def set_shared_state(self, shared_state: Any) -> None:
        self.shared_state = shared_state

    def set_exchange_client(self, exchange_client: Any) -> None:
        self.exchange_client = exchange_client

    def set_execution_manager(self, execution_manager: Any) -> None:
        self.execution_manager = execution_manager

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._set_status("Initialized", "startup_reconciliation")
        await self._restart_recovery()
        self._task = asyncio.create_task(self._run_loop(), name="ExchangeTruthAuditor:loop")
        self._user_data_health_task = asyncio.create_task(
            self._user_data_health_loop(),
            name="ExchangeTruthAuditor:user_data_health",
        )
        # If full audit loop is slower than requested verification cadence, add a dedicated verifier.
        if self.interval_sec > self.open_order_verify_interval_sec:
            self._open_order_verify_task = asyncio.create_task(
                self._open_order_verify_loop(),
                name="ExchangeTruthAuditor:open_order_verify",
            )
        else:
            self._open_order_verify_task = None
        self.logger.info("ExchangeTruthAuditor started (interval=%.2fs)", self.interval_sec)
        await self._set_status("Operational", "running")

    async def stop(self) -> None:
        self._running = False
        task = self._task
        health_task = self._user_data_health_task
        open_task = self._open_order_verify_task
        self._task = None
        self._user_data_health_task = None
        self._open_order_verify_task = None
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if health_task and not health_task.done():
            health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await health_task
        if open_task and not open_task.done():
            open_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await open_task
        await self._set_status("Stopped", "shutdown")

    async def close(self) -> None:
        await self.stop()

    def health(self) -> Dict[str, Any]:
        running = bool(self._task and not self._task.done() and self._running)
        sell_map = dict(self._last_sell_finalize_snapshot or {})
        return {
            "component": "ExchangeTruthAuditor",
            "status": "Healthy" if running else "Stopped",
            "running": running,
            "interval_sec": float(self.interval_sec),
            "user_data_health_interval_sec": float(self.user_data_health_interval_sec),
            "open_order_verify_interval_sec": float(self.open_order_verify_interval_sec),
            "seen_orders": len(self._seen_order_ids),
            "seen_trades": len(self._seen_trade_ids),
            "sell_finalize_fills_seen": int(sell_map.get("sell_finalize_fills_seen", 0) or 0),
            "sell_finalize_finalized": int(sell_map.get("sell_finalize_finalized", 0) or 0),
            "sell_finalize_pending": int(sell_map.get("sell_finalize_pending", 0) or 0),
            "sell_finalize_gap": int(sell_map.get("sell_finalize_gap", 0) or 0),
            "sell_finalize_duplicate": int(sell_map.get("sell_finalize_duplicate", 0) or 0),
            "sell_finalize_pending_timeout": int(sell_map.get("sell_finalize_pending_timeout", 0) or 0),
            "last_successful_force_sync_ts": float(self.last_successful_force_sync_ts or 0.0),
            "last_user_data_health": dict(self._last_user_data_health_snapshot or {}),
        }

    async def _run_loop(self) -> None:
        while self._running:
            started = time.time()
            try:
                stats = await self._audit_cycle()
                detail = (
                    f"symbols={stats.get('symbols', 0)} "
                    f"fills_recovered={stats.get('fills_recovered', 0)} "
                    f"trades_recovered={stats.get('trades_recovered', 0)} "
                    f"trades_sell_finalized={stats.get('trades_sell_finalized', 0)} "
                    f"phantoms_closed={stats.get('phantoms_closed', 0)} "
                    f"open_order_mismatch={stats.get('open_order_mismatch', 0)} "
                    f"sell_missing_canonical={stats.get('sell_missing_canonical', 0)} "
                    f"sell_finalize_gap={stats.get('sell_finalize_gap', 0)} "
                    f"sell_finalize_pending={stats.get('sell_finalize_pending', 0)}"
                )
                await self._set_status("Operational", detail)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error("[TruthAuditor] audit cycle failed: %s", e, exc_info=True)
                await self._set_status("Degraded", f"audit_cycle_failed:{type(e).__name__}")
            elapsed = time.time() - started
            await asyncio.sleep(max(0.1, self.interval_sec - elapsed))

    async def _open_order_verify_loop(self) -> None:
        """
        Explicit periodic open-order verification (5–10s), independent of full audit cadence.
        """
        while self._running:
            try:
                symbols = await self._collect_symbols()
                stats = await self._reconcile_open_orders(symbols=symbols)
                with contextlib.suppress(Exception):
                    await self._emit_event(
                        "TRUTH_AUDIT_OPEN_ORDER_VERIFY",
                        {
                            "symbols": int(len(symbols)),
                            "open_orders": int(stats.get("open_orders", 0) or 0),
                            "open_order_mismatch": int(stats.get("open_order_mismatch", 0) or 0),
                            "ts": time.time(),
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.debug("open-order verify loop failed", exc_info=True)
            await asyncio.sleep(max(1.0, float(self.open_order_verify_interval_sec or 7.0)))

    async def _user_data_health_loop(self) -> None:
        """
        Explicit websocket/user-data health monitor.
        Soft-gap: exposure-aware + cooldown.
        Hard-gap: overrides exposure gating.
        """
        while self._running:
            try:
                await self._check_user_data_health()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.debug("user-data health loop failed", exc_info=True)
            await asyncio.sleep(max(1.0, float(self.user_data_health_interval_sec or 5.0)))

    async def _check_user_data_health(self) -> None:
        ec = self.exchange_client
        if ec is None:
            return

        snapshot = self._get_ws_health_snapshot()
        if not bool(snapshot.get("user_data_stream_enabled", True)):
            self._last_user_data_health_snapshot = dict(snapshot)
            return

        now = time.time()
        exposure_exists = await self._has_exposure()
        try:
            raw_user_gap = snapshot.get("user_data_gap_sec", -1.0)
            user_gap = float(raw_user_gap) if raw_user_gap is not None else -1.0
        except Exception:
            user_gap = -1.0
        hard_gap = max(1.0, float(self.user_data_hard_gap_sec or 60.0))
        soft_gap = max(1.0, float(self.user_data_soft_gap_sec or 20.0))
        cooldown = max(0.0, float(self.force_sync_cooldown_sec or 15.0))
        since_sync = self._force_sync_recency(now_ts=now)

        payload = {
            "ts": now,
            "exposure_exists": bool(exposure_exists),
            "soft_gap_sec": float(soft_gap),
            "hard_gap_sec": float(hard_gap),
            "force_sync_cooldown_sec": float(cooldown),
            "since_last_force_sync_sec": float(since_sync if since_sync != float("inf") else -1.0),
            **snapshot,
        }
        self._last_user_data_health_snapshot = dict(payload)

        # Hard gap overrides exposure gating.
        if user_gap > hard_gap:
            reconnected = False
            reconn = getattr(ec, "reconnect_user_data_stream", None)
            if callable(reconn):
                with contextlib.suppress(Exception):
                    reconnected = bool(await self._maybe_await(reconn(reason="HARD_GAP_TIMEOUT")))

            sync_res = await self.trigger_immediate_sync(reason="HARD_GAP_TIMEOUT")
            await self._emit_event(
                "TRUTH_AUDIT_USER_DATA_HARD_GAP",
                {
                    **payload,
                    "reconnect_requested": True,
                    "reconnected": bool(reconnected),
                    "sync_result": dict(sync_res or {}),
                },
            )
            self.logger.warning(
                "[TruthAuditor:UserData] hard gap %.1fs > %.1fs -> reconnect=%s sync=%s",
                float(user_gap),
                float(hard_gap),
                bool(reconnected),
                str((sync_res or {}).get("status", "")),
            )
            return

        # Soft gap is exposure-aware and debounced.
        if (
            exposure_exists
            and user_gap > soft_gap
            and since_sync > cooldown
        ):
            sync_res = await self.trigger_immediate_sync(reason="SOFT_GAP_EXPOSURE")
            await self._emit_event(
                "TRUTH_AUDIT_USER_DATA_SOFT_GAP",
                {
                    **payload,
                    "sync_result": dict(sync_res or {}),
                },
            )
            self.logger.info(
                "[TruthAuditor:UserData] soft gap %.1fs > %.1fs with exposure -> sync=%s",
                float(user_gap),
                float(soft_gap),
                str((sync_res or {}).get("status", "")),
            )

    def _get_ws_health_snapshot(self) -> Dict[str, Any]:
        ec = self.exchange_client
        if ec is None:
            return {
                "user_data_stream_enabled": False,
                "ws_connected": False,
                "ws_reconnect_count": 0,
                "last_user_data_event_ts": 0.0,
                "last_any_ws_event_ts": 0.0,
                "last_listenkey_refresh_ts": 0.0,
                "last_successful_force_sync_ts": float(self.last_successful_force_sync_ts or 0.0),
                "user_data_gap_sec": -1.0,
                "any_ws_gap_sec": -1.0,
                "listenkey_refresh_gap_sec": -1.0,
            }

        snap: Dict[str, Any] = {}
        getter = getattr(ec, "get_ws_health_snapshot", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                raw = getter()
                snap = raw if isinstance(raw, dict) else {}

        now = time.time()
        user_ts = float(snap.get("last_user_data_event_ts", getattr(ec, "last_user_data_event_ts", 0.0)) or 0.0)
        any_ts = float(snap.get("last_any_ws_event_ts", getattr(ec, "last_any_ws_event_ts", 0.0)) or 0.0)
        listen_ts = float(
            snap.get("last_listenkey_refresh_ts", getattr(ec, "last_listenkey_refresh_ts", 0.0)) or 0.0
        )
        sync_ts = float(
            snap.get("last_successful_force_sync_ts", getattr(ec, "last_successful_force_sync_ts", 0.0)) or 0.0
        )
        return {
            "user_data_stream_enabled": bool(
                snap.get("user_data_stream_enabled", getattr(ec, "user_data_stream_enabled", True))
            ),
            "ws_connected": bool(snap.get("ws_connected", getattr(ec, "ws_connected", False))),
            "ws_reconnect_count": int(snap.get("ws_reconnect_count", getattr(ec, "ws_reconnect_count", 0)) or 0),
            "last_user_data_event_ts": user_ts,
            "last_any_ws_event_ts": any_ts,
            "last_listenkey_refresh_ts": listen_ts,
            "last_successful_force_sync_ts": sync_ts,
            "user_data_gap_sec": (now - user_ts) if user_ts > 0 else -1.0,
            "any_ws_gap_sec": (now - any_ts) if any_ts > 0 else -1.0,
            "listenkey_refresh_gap_sec": (now - listen_ts) if listen_ts > 0 else -1.0,
        }

    async def _has_exposure(self) -> bool:
        ss = self.shared_state
        if ss is None:
            return False

        # Any open orders implies active exposure workflow.
        try:
            open_orders = getattr(ss, "open_orders", None)
            if isinstance(open_orders, dict):
                for payload in open_orders.values():
                    if isinstance(payload, dict) and payload:
                        return True
                    if isinstance(payload, list) and len(payload) > 0:
                        return True
        except Exception:
            pass

        # Any non-dust position implies exposure.
        try:
            positions = getattr(ss, "positions", None)
            if isinstance(positions, dict):
                for pos in positions.values():
                    if not isinstance(pos, dict):
                        continue
                    qty = float(pos.get("quantity", pos.get("qty", 0.0)) or 0.0)
                    if qty > max(self.dust_threshold, self.position_mismatch_tol):
                        return True
        except Exception:
            pass
        return False

    def _force_sync_recency(self, now_ts: Optional[float] = None) -> float:
        now = float(now_ts if now_ts is not None else time.time())
        ec_ts = 0.0
        with contextlib.suppress(Exception):
            ec_ts = float(getattr(self.exchange_client, "last_successful_force_sync_ts", 0.0) or 0.0)
        local_ts = float(self.last_successful_force_sync_ts or 0.0)
        ref = max(local_ts, ec_ts)
        if ref <= 0:
            return float("inf")
        return max(0.0, now - ref)

    async def trigger_immediate_sync(self, reason: str = "MANUAL_TRIGGER") -> Dict[str, Any]:
        """
        Idempotent, async-safe, debounced immediate sync trigger.
        """
        reason_u = str(reason or "MANUAL_TRIGGER").upper()
        now = time.time()
        cooldown = max(0.0, float(self.force_sync_cooldown_sec or 15.0))
        if self._force_sync_lock is None:
            self._force_sync_lock = asyncio.Lock()

        async with self._force_sync_lock:
            since_sync = self._force_sync_recency(now_ts=now)
            if since_sync <= cooldown:
                payload = {
                    "status": "skipped_cooldown",
                    "reason": reason_u,
                    "cooldown_sec": float(cooldown),
                    "since_last_sync_sec": float(since_sync),
                    "ts": now,
                }
                await self._emit_event("TRUTH_AUDIT_IMMEDIATE_SYNC", payload)
                return payload

            symbols = await self._collect_symbols()
            open_stats: Dict[str, int] = {"open_orders": 0, "open_order_mismatch": 0}
            order_stats: Dict[str, int] = {"fills_recovered": 0}
            trade_stats: Dict[str, int] = {"trades_recovered": 0, "trades_sell_finalized": 0}
            sync_balance_ok = False

            ss = self.shared_state
            if ss is not None and hasattr(ss, "sync_authoritative_balance"):
                with contextlib.suppress(Exception):
                    await self._maybe_call(ss, "sync_authoritative_balance", force=True)
                    sync_balance_ok = True

            with contextlib.suppress(Exception):
                open_stats = await self._reconcile_open_orders(symbols=symbols)
            with contextlib.suppress(Exception):
                order_stats = await self._reconcile_orders(symbols=symbols, startup=False)
            with contextlib.suppress(Exception):
                trade_stats = await self._reconcile_trades(symbols=symbols, startup=False)

            done_ts = time.time()
            self.last_successful_force_sync_ts = done_ts
            with contextlib.suppress(Exception):
                rec = getattr(self.exchange_client, "record_successful_force_sync", None)
                if callable(rec):
                    rec(reason=reason_u, ts=done_ts)
                else:
                    setattr(self.exchange_client, "last_successful_force_sync_ts", done_ts)

            payload = {
                "status": "ok",
                "reason": reason_u,
                "symbols": int(len(symbols)),
                "sync_authoritative_balance": bool(sync_balance_ok),
                "open_orders": int(open_stats.get("open_orders", 0) or 0),
                "open_order_mismatch": int(open_stats.get("open_order_mismatch", 0) or 0),
                "fills_recovered": int(order_stats.get("fills_recovered", 0) or 0),
                "trades_recovered": int(trade_stats.get("trades_recovered", 0) or 0),
                "trades_sell_finalized": int(trade_stats.get("trades_sell_finalized", 0) or 0),
                "ts": done_ts,
            }
            await self._emit_event("TRUTH_AUDIT_IMMEDIATE_SYNC", payload)
            self.logger.info(
                "[TruthAuditor:ImmediateSync] reason=%s symbols=%d open_mismatch=%d fills=%d trades=%d",
                reason_u,
                int(payload["symbols"]),
                int(payload["open_order_mismatch"]),
                int(payload["fills_recovered"]),
                int(payload["trades_recovered"]),
            )
            return payload

    async def _restart_recovery(self) -> None:
        """
        Restart safety:
        - Reconcile recent fills (idempotent).
        - Reconcile balances/phantom positions.
        - Rebuild open-order mirror from exchange truth.
        """
        symbols = await self._collect_symbols()
        # Recovery on start should scan a larger set when available.
        if len(symbols) > 0 and self.max_symbols_per_cycle > 0:
            self._cursor = 0

        fills = await self._reconcile_orders(symbols=symbols, startup=True)
        trades = await self._reconcile_trades(symbols=symbols, startup=True)
        balances = await self._reconcile_balances(symbols=symbols)
        orders = await self._reconcile_open_orders(symbols=symbols)
        sell_map = await self._validate_sell_finalize_mapping(startup=True)

        await self._emit_event(
            "TRUTH_AUDIT_RESTART_SYNC",
            {
                "symbols": len(symbols),
                "fills_recovered": fills.get("fills_recovered", 0),
                "trades_recovered": trades.get("trades_recovered", 0),
                "trades_sell_finalized": trades.get("trades_sell_finalized", 0),
                "phantoms_closed": balances.get("phantoms_closed", 0),
                "open_order_mismatch": orders.get("open_order_mismatch", 0),
                "sell_missing_canonical": fills.get("sell_missing_canonical", 0),
                "sell_finalize_fills_seen": sell_map.get("sell_finalize_fills_seen", 0),
                "sell_finalize_finalized": sell_map.get("sell_finalize_finalized", 0),
                "sell_finalize_pending": sell_map.get("sell_finalize_pending", 0),
                "sell_finalize_gap": sell_map.get("sell_finalize_gap", 0),
                "ts": time.time(),
            },
        )

    async def _audit_cycle(self) -> Dict[str, int]:
        self._prune_seen_cache()
        symbols_all = await self._collect_symbols()
        symbols = self._slice_symbols(symbols_all)

        order_stats = await self._reconcile_orders(symbols=symbols, startup=False)
        trade_stats = await self._reconcile_trades(symbols=symbols, startup=False)
        balance_stats = await self._reconcile_balances(symbols=symbols)
        open_stats = await self._reconcile_open_orders(symbols=symbols)
        sell_map = await self._validate_sell_finalize_mapping(startup=False)

        return {
            "symbols": len(symbols),
            "fills_recovered": int(order_stats.get("fills_recovered", 0)),
            "trades_recovered": int(trade_stats.get("trades_recovered", 0)),
            "trades_sell_finalized": int(trade_stats.get("trades_sell_finalized", 0)),
            "phantoms_closed": int(balance_stats.get("phantoms_closed", 0)),
            "open_order_mismatch": int(open_stats.get("open_order_mismatch", 0)),
            "sell_missing_canonical": int(order_stats.get("sell_missing_canonical", 0)),
            "sell_finalize_fills_seen": int(sell_map.get("sell_finalize_fills_seen", 0)),
            "sell_finalize_finalized": int(sell_map.get("sell_finalize_finalized", 0)),
            "sell_finalize_pending": int(sell_map.get("sell_finalize_pending", 0)),
            "sell_finalize_gap": int(sell_map.get("sell_finalize_gap", 0)),
            "sell_finalize_duplicate": int(sell_map.get("sell_finalize_duplicate", 0)),
            "sell_finalize_pending_timeout": int(sell_map.get("sell_finalize_pending_timeout", 0)),
        }

    async def _set_status(self, status: str, detail: str = "") -> None:
        ss = self.shared_state
        if ss is None:
            return
        try:
            upd = getattr(ss, "update_component_status", None) or getattr(ss, "set_component_status", None)
            if callable(upd):
                await self._maybe_await(upd("ExchangeTruthAuditor", status, detail))
            else:
                now = time.time()
                cs = getattr(ss, "component_statuses", None)
                if isinstance(cs, dict):
                    cs["ExchangeTruthAuditor"] = {
                        "status": status,
                        "message": detail,
                        "timestamp": now,
                        "ts": now,
                    }
        except Exception:
            self.logger.debug("status update failed", exc_info=True)

    async def _emit_event(self, event_name: str, payload: Dict[str, Any]) -> None:
        ss = self.shared_state
        if ss is None:
            return
        try:
            emit = getattr(ss, "emit_event", None)
            if callable(emit):
                await self._maybe_await(emit(event_name, payload))
        except Exception:
            self.logger.debug("emit_event failed: %s", event_name, exc_info=True)

    async def _validate_sell_finalize_mapping(self, startup: bool = False) -> Dict[str, int]:
        """
        Validate SELL finalization invariant by reading ExecutionManager health counters.
        Does NOT place orders. Governance-only check.
        """
        result = {
            "sell_finalize_fills_seen": 0,
            "sell_finalize_finalized": 0,
            "sell_finalize_pending": 0,
            "sell_finalize_gap": 0,
            "sell_finalize_duplicate": 0,
            "sell_finalize_pending_timeout": 0,
        }

        try:
            em = None
            with contextlib.suppress(Exception):
                em = getattr(self.app, "execution_manager", None)
            if em is None:
                with contextlib.suppress(Exception):
                    em = getattr(self.shared_state, "execution_manager", None)
            if em is None:
                self._last_sell_finalize_snapshot = dict(result)
                return result

            health_fn = getattr(em, "health", None)
            if not callable(health_fn):
                self._last_sell_finalize_snapshot = dict(result)
                return result

            health = await self._maybe_await(health_fn()) or {}

            fills_seen = int(health.get("sell_finalize_fills_seen", 0) or 0)
            finalized = int(health.get("sell_finalize_finalized", 0) or 0)
            pending = int(health.get("sell_finalize_pending", 0) or 0)
            duplicate = int(health.get("sell_finalize_duplicate", 0) or 0)
            pending_timeout = int(health.get("sell_finalize_pending_timeout", 0) or 0)

            gap = max(0, fills_seen - finalized)

            result.update(
                {
                    "sell_finalize_fills_seen": fills_seen,
                    "sell_finalize_finalized": finalized,
                    "sell_finalize_pending": pending,
                    "sell_finalize_gap": gap,
                    "sell_finalize_duplicate": duplicate,
                    "sell_finalize_pending_timeout": pending_timeout,
                }
            )

            # Snapshot for health()
            self._last_sell_finalize_snapshot = dict(result)

            # Warn if gap exceeds threshold.
            if gap >= max(1, int(self.sell_finalize_gap_warn_threshold or 1)):
                now = time.time()
                cooldown = max(5.0, float(self.sell_finalize_warn_cooldown_sec or 60.0))
                if now - float(self._last_sell_finalize_warn_ts or 0.0) > cooldown:
                    self.logger.warning(
                        "[TruthAuditor:SellMap] gap=%d fills_seen=%d finalized=%d pending=%d duplicate=%d pending_timeout=%d startup=%s",
                        gap,
                        fills_seen,
                        finalized,
                        pending,
                        duplicate,
                        pending_timeout,
                        bool(startup),
                    )
                    await self._emit_event(
                        "TRUTH_AUDIT_SELL_FINALIZE_GAP",
                        {
                            "gap": gap,
                            "fills_seen": fills_seen,
                            "finalized": finalized,
                            "pending": pending,
                            "duplicate_finalize": duplicate,
                            "pending_timeout": pending_timeout,
                            "startup_scan": bool(startup),
                            "ts": now,
                        },
                    )
                    self._last_sell_finalize_warn_ts = now

        except Exception as e:
            self.logger.error(
                "[TruthAuditor] sell finalize mapping validation failed: %s",
                e,
                exc_info=True,
            )

        return result

    def _known_quotes(self) -> List[str]:
        quotes: List[str] = []
        try:
            gkq = getattr(self.exchange_client, "get_known_quotes", None)
            if callable(gkq):
                qv = gkq()
                if isinstance(qv, (list, tuple, set)):
                    quotes.extend([str(x).upper() for x in qv if x])
        except Exception:
            pass
        if not quotes:
            quotes = ["USDT", "FDUSD", "USDC", "BUSD", "TUSD", "DAI", "BTC", "ETH"]
        quotes = sorted(set(quotes), key=len, reverse=True)
        return quotes

    def _default_quote(self) -> str:
        try:
            if isinstance(self.config, dict):
                return str(self.config.get("quote_asset", self.config.get("BASE_CURRENCY", "USDT"))).upper()
            return str(getattr(self.config, "quote_asset", getattr(self.config, "BASE_CURRENCY", "USDT"))).upper()
        except Exception:
            return "USDT"

    def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
        sym = str(symbol or "").upper().replace("/", "")
        for q in self._known_quotes():
            if sym.endswith(q) and len(sym) > len(q):
                return sym[: -len(q)], q
        dq = self._default_quote()
        if sym.endswith(dq) and len(sym) > len(dq):
            return sym[: -len(dq)], dq
        return sym[:-4], sym[-4:] if len(sym) > 4 else dq

    async def _collect_symbols(self) -> List[str]:
        out: List[str] = []
        seen = set()
        ss = self.shared_state
        if ss is None:
            return out

        try:
            active = getattr(ss, "get_active_symbols", None)
            if callable(active):
                try:
                    v = active(limit=max(0, int(self._cfg("TRUTH_AUDIT_ACTIVE_SYMBOL_LIMIT", 200) or 200)))
                except TypeError:
                    v = active()
                v = await self._maybe_await(v)
                if isinstance(v, list):
                    for s in v:
                        sym = str(s or "").upper()
                        if sym and sym not in seen:
                            out.append(sym)
                            seen.add(sym)
        except Exception:
            self.logger.debug("collect symbols from active list failed", exc_info=True)

        try:
            positions = getattr(ss, "get_open_positions", None)
            pdata = positions() if callable(positions) else getattr(ss, "positions", {})
            pdata = await self._maybe_await(pdata)
            if isinstance(pdata, dict):
                for s in pdata.keys():
                    sym = str(s or "").upper()
                    if sym and sym not in seen:
                        out.append(sym)
                        seen.add(sym)
        except Exception:
            self.logger.debug("collect symbols from positions failed", exc_info=True)

        try:
            snap_fn = getattr(ss, "get_positions_snapshot", None)
            pdata_all = snap_fn() if callable(snap_fn) else getattr(ss, "positions", {})
            pdata_all = await self._maybe_await(pdata_all)
            if isinstance(pdata_all, dict):
                for s in pdata_all.keys():
                    sym = str(s or "").upper()
                    if sym and sym not in seen:
                        out.append(sym)
                        seen.add(sym)
        except Exception:
            self.logger.debug("collect symbols from position snapshot failed", exc_info=True)

        try:
            accepted = getattr(ss, "accepted_symbols", {}) or {}
            if isinstance(accepted, dict):
                for s in accepted.keys():
                    sym = str(s or "").upper()
                    if sym and sym not in seen:
                        out.append(sym)
                        seen.add(sym)
        except Exception:
            pass

        try:
            history = list(getattr(ss, "trade_history", []) or [])
            hist_limit = max(50, int(self._cfg("TRUTH_AUDIT_HISTORY_SYMBOL_LIMIT", 300) or 300))
            for row in reversed(history[-hist_limit:]):
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol", "")).upper()
                if sym and sym not in seen:
                    out.append(sym)
                    seen.add(sym)
        except Exception:
            self.logger.debug("collect symbols from trade history failed", exc_info=True)

        max_symbols = int(self._cfg("TRUTH_AUDIT_MAX_SYMBOLS", 200) or 200)
        if max_symbols > 0 and len(out) > max_symbols:
            out = out[:max_symbols]
        return out

    def _slice_symbols(self, symbols: List[str]) -> List[str]:
        if not symbols:
            return []
        n = len(symbols)
        per_cycle = max(1, int(self.max_symbols_per_cycle or n))
        if per_cycle >= n:
            return symbols
        start = self._cursor % n
        end = start + per_cycle
        if end <= n:
            batch = symbols[start:end]
        else:
            batch = symbols[start:] + symbols[: end - n]
        self._cursor = (start + per_cycle) % n
        return batch

    async def _get_exchange_balances(self) -> Dict[str, Dict[str, float]]:
        ec = self.exchange_client
        if ec is None:
            return {}
        if not self._has_signed_access():
            self._warn_signed_unavailable("balances")
            return {}
        try:
            if hasattr(ec, "get_account_balances"):
                b = await self._maybe_await(ec.get_account_balances())
                if isinstance(b, dict):
                    return b
        except Exception:
            pass
        try:
            if hasattr(ec, "get_balances"):
                b = await self._maybe_await(ec.get_balances())
                if isinstance(b, dict):
                    return b
        except Exception:
            pass
        return {}

    def _has_signed_access(self) -> bool:
        ec = self.exchange_client
        if ec is None:
            return False
        try:
            key = str(getattr(ec, "api_key", "") or "")
            sec = str(getattr(ec, "api_secret", "") or "")
            if not key or not sec:
                return False
            if key == "paper_key" and sec == "paper_secret":
                return False
            if hasattr(ec, "is_started") and not bool(getattr(ec, "is_started")):
                return False
            return True
        except Exception:
            return False

    def _warn_signed_unavailable(self, where: str) -> None:
        now = time.time()
        if now - self._last_signed_skip_log_ts < 60.0:
            return
        self._last_signed_skip_log_ts = now
        self.logger.info(
            "[TruthAuditor] skipping %s reconciliation: signed exchange access unavailable",
            where,
        )

    def _position_qty(self, pos: Dict[str, Any]) -> float:
        try:
            return float(pos.get("quantity", pos.get("qty", 0.0)) or 0.0)
        except Exception:
            return 0.0

    async def _reconcile_balances(self, symbols: List[str]) -> Dict[str, int]:
        stats = {"phantoms_closed": 0, "mismatches": 0}
        ss = self.shared_state
        if ss is None:
            return stats

        balances = await self._get_exchange_balances()
        if not balances:
            return stats

        positions: Dict[str, Dict[str, Any]] = {}
        try:
            get_open = getattr(ss, "get_open_positions", None)
            positions = get_open() if callable(get_open) else {}
            positions = await self._maybe_await(positions)
            if not isinstance(positions, dict):
                positions = {}
        except Exception:
            positions = {}

        if not positions:
            return stats

        symbols_set = set(symbols or [])
        for sym, pos in positions.items():
            if symbols_set and sym not in symbols_set:
                continue
            state_qty = self._position_qty(pos if isinstance(pos, dict) else {})
            if state_qty <= self.position_mismatch_tol:
                continue
            base_asset, _ = self._split_base_quote(sym)
            bal = balances.get(base_asset.upper(), {})
            exchange_qty = float((bal or {}).get("free", 0.0) or 0.0) + float((bal or {}).get("locked", 0.0) or 0.0)
            if state_qty > self.dust_threshold and exchange_qty < self.dust_threshold:
                closed = await self._close_phantom_position(sym, pos, state_qty)
                if closed:
                    stats["phantoms_closed"] += 1
                continue
            if abs(state_qty - exchange_qty) > max(self.position_mismatch_tol, self.dust_threshold):
                stats["mismatches"] += 1

        if stats["mismatches"] > 0 and hasattr(ss, "sync_authoritative_balance"):
            with contextlib.suppress(Exception):
                await self._maybe_call(ss, "sync_authoritative_balance", force=True)
                await self._emit_event(
                    "TRUTH_AUDIT_BALANCE_RESYNC",
                    {
                        "mismatches": int(stats["mismatches"]),
                        "ts": time.time(),
                    },
                )
        return stats

    async def _close_phantom_position(self, symbol: str, pos: Dict[str, Any], qty: float) -> bool:
        now = time.time()
        last = float(self._recent_phantom_by_symbol.get(symbol, 0.0) or 0.0)
        if now - last < self.phantom_cooldown_sec:
            return False

        price = 0.0
        try:
            price = float((pos or {}).get("avg_price", 0.0) or (pos or {}).get("entry_price", 0.0) or 0.0)
        except Exception:
            price = 0.0
        if price <= 0:
            try:
                lp = getattr(self.shared_state, "latest_prices", {}) or {}
                price = float(lp.get(symbol, 0.0) or 0.0)
            except Exception:
                price = 0.0
        if price <= 0 and self.exchange_client and hasattr(self.exchange_client, "get_current_price"):
            with contextlib.suppress(Exception):
                price = float(await self.exchange_client.get_current_price(symbol) or 0.0)
        if price <= 0:
            price = 1.0

        synthetic_order = {
            "symbol": symbol,
            "side": "SELL",
            "status": "FILLED",
            "executedQty": float(qty),
            "price": float(price),
            "avgPrice": float(price),
            "cummulativeQuoteQty": float(qty) * float(price),
            "orderId": f"truth-phantom-{symbol}-{int(now * 1000)}",
            "updateTime": int(now * 1000),
        }
        ok = await self._apply_recovered_fill(
            synthetic_order,
            reason="exchange_truth_sync",
            synthetic=True,
        )
        if ok:
            self._recent_phantom_by_symbol[symbol] = now
            await self._emit_event(
                "TRUTH_AUDIT_PHANTOM_POSITION_CLOSED",
                {
                    "symbol": symbol,
                    "qty": float(qty),
                    "price": float(price),
                    "reason": "exchange_balance_zero_state_positive",
                    "ts": now,
                },
            )
        return ok

    # =============================
    # Trade-level (fill-level) reconciliation via /api/v3/myTrades
    # =============================

    async def _reconcile_trades(self, symbols: List[str], startup: bool) -> Dict[str, int]:
        """
        Fill-level reconciliation using exchange trade history (myTrades).

        For each symbol, fetches recent fills from the exchange and checks
        whether each trade ID has been seen. Unseen fills are applied via
        the existing _apply_recovered_fill path. For unseen SELL fills,
        additionally forces the full ExecutionManager sell lifecycle
        finalization to ensure POSITION_CLOSED events, realized PnL, and
        exit bookkeeping are complete.
        """
        stats = {"trades_recovered": 0, "trades_skipped": 0, "trades_sell_finalized": 0}
        ec = self.exchange_client
        if not symbols or ec is None:
            return stats
        if not self._has_signed_access():
            return stats
        if not hasattr(ec, "get_my_trades") or not callable(getattr(ec, "get_my_trades", None)):
            return stats

        now_s = time.time()
        for sym in symbols:
            try:
                trades = await ec.get_my_trades(sym, limit=self.trade_history_limit)
            except Exception:
                self.logger.debug("[TruthAuditor:Trades] get_my_trades failed for %s", sym, exc_info=True)
                continue
            if not isinstance(trades, list) or not trades:
                continue

            for trade in trades:
                if not isinstance(trade, dict):
                    continue

                trade_id = str(trade.get("id") or trade.get("tradeId") or "")
                if not trade_id:
                    continue

                # Dedup: skip already-seen trade IDs
                if self._is_trade_seen(trade_id):
                    continue

                # Age filter: skip old trades unless startup scan
                trade_time_ms = int(trade.get("time", 0) or 0)
                if not startup and self.recovery_window_sec > 0 and trade_time_ms > 0:
                    age_ms = max(0, int(now_s * 1000) - trade_time_ms)
                    if age_ms > (self.recovery_window_sec * 1000):
                        self._mark_trade_seen(trade_id)
                        continue

                side = "BUY" if bool(trade.get("isBuyer")) else "SELL"
                qty = 0.0
                price = 0.0
                quote_qty = 0.0
                try:
                    qty = float(trade.get("qty", 0.0) or 0.0)
                    price = float(trade.get("price", 0.0) or 0.0)
                    quote_qty = float(trade.get("quoteQty", 0.0) or 0.0)
                except Exception:
                    pass
                if qty <= 0 or price <= 0:
                    self._mark_trade_seen(trade_id)
                    continue

                # Build a synthetic order dict from the trade for _apply_recovered_fill
                order_id = str(trade.get("orderId") or "")
                commission = 0.0
                commission_asset = ""
                try:
                    commission = float(trade.get("commission", 0.0) or 0.0)
                    commission_asset = str(trade.get("commissionAsset", "") or "").upper()
                except Exception:
                    pass
                base_asset, quote_asset = self._split_base_quote(sym)
                fee_quote = commission if commission_asset == quote_asset else 0.0
                fee_base = commission if commission_asset == base_asset else 0.0

                synthetic_order = {
                    "symbol": sym,
                    "side": side,
                    "status": "FILLED",
                    "executedQty": float(qty),
                    "price": float(price),
                    "avgPrice": float(price),
                    "cummulativeQuoteQty": float(quote_qty) if quote_qty > 0 else float(qty * price),
                    "orderId": order_id,
                    "updateTime": int(trade_time_ms),
                    "fee_quote": float(fee_quote),
                    "fee_base": float(fee_base),
                    "fills": [{
                        "price": str(price),
                        "qty": str(qty),
                        "commission": str(commission),
                        "commissionAsset": str(trade.get("commissionAsset", "") or ""),
                    }],
                    "_truth_auditor_trade_id": trade_id,
                    "_truth_auditor_source": "myTrades",
                }

                # Check if this fill was already applied (matches trade_history)
                if self._is_fill_likely_already_applied(synthetic_order):
                    self._mark_trade_seen(trade_id)
                    stats["trades_skipped"] += 1
                    continue

                # Also check order-level seen cache (order may have been reconciled via _reconcile_orders)
                if order_id and self._is_order_seen(order_id):
                    self._mark_trade_seen(trade_id)
                    stats["trades_skipped"] += 1
                    continue

                # Unseen fill — recover it
                self.logger.warning(
                    "[TruthAuditor:Trades] Recovering unseen %s fill: symbol=%s trade_id=%s order_id=%s qty=%.8f price=%.8f",
                    side, sym, trade_id, order_id or "n/a", qty, price,
                )
                recovered = await self._apply_recovered_fill(
                    synthetic_order,
                    reason="trade_level_recovery",
                    synthetic=False,
                )
                self._mark_trade_seen(trade_id)
                if order_id:
                    self._mark_order_seen(order_id)

                if recovered:
                    stats["trades_recovered"] += 1

                    # For SELL fills, force the full lifecycle finalization
                    if side == "SELL":
                        finalized = await self._force_finalize_sell_lifecycle(
                            sym, synthetic_order, trade_id=trade_id,
                        )
                        if finalized:
                            stats["trades_sell_finalized"] += 1
                else:
                    stats["trades_skipped"] += 1

        return stats

    async def _force_finalize_sell_lifecycle(
        self,
        symbol: str,
        order: Dict[str, Any],
        *,
        trade_id: str = "",
    ) -> bool:
        """
        Force the full SELL lifecycle finalization through ExecutionManager.

        This ensures POSITION_CLOSED events, realized PnL computation,
        exit bookkeeping (cooldowns), and sell-finalize invariant tracking
        are all completed — even when the original execution path missed them.
        """
        em = self.execution_manager
        if em is None:
            # Try to resolve from app context
            with contextlib.suppress(Exception):
                em = getattr(self.app, "execution_manager", None)
            if em is None:
                with contextlib.suppress(Exception):
                    em = getattr(self.shared_state, "execution_manager", None)

        if em is None or not callable(getattr(em, "_finalize_sell_post_fill", None)):
            self.logger.debug(
                "[TruthAuditor:SellFinalize] ExecutionManager unavailable for %s trade_id=%s; "
                "falling back to record_trade only",
                symbol, trade_id,
            )
            return False

        try:
            # Run _ensure_post_fill_handled first (idempotent — dedupes via _post_fill_done flag)
            post_fill = None
            if callable(getattr(em, "_ensure_post_fill_handled", None)):
                post_fill = await em._ensure_post_fill_handled(
                    symbol=symbol,
                    side="SELL",
                    order=order,
                    tier=None,
                    tag="truth_auditor",
                )

            # Run the full sell finalization (close events, exit bookkeeping, finalize tracking)
            await em._finalize_sell_post_fill(
                symbol=symbol,
                order=order,
                tag="truth_auditor",
                post_fill=post_fill if isinstance(post_fill, dict) else None,
                policy_ctx={"exit_reason": "TRUTH_AUDIT_RECOVERY", "reason": "TRUTH_AUDIT_RECOVERY"},
                tier=None,
            )

            self.logger.info(
                "[TruthAuditor:SellFinalize] Forced SELL lifecycle finalization: symbol=%s trade_id=%s order_id=%s",
                symbol, trade_id, self._order_id(order) or "n/a",
            )
            await self._emit_event(
                "TRUTH_AUDIT_SELL_LIFECYCLE_FINALIZED",
                {
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "order_id": self._order_id(order),
                    "qty": float(order.get("executedQty", 0.0) or 0.0),
                    "price": self._resolve_order_price(order),
                    "source": "ExchangeTruthAuditor",
                    "ts": time.time(),
                },
            )
            return True
        except Exception as e:
            self.logger.error(
                "[TruthAuditor:SellFinalize] Failed to force SELL lifecycle for %s trade_id=%s: %s",
                symbol, trade_id, e, exc_info=True,
            )
            return False

    def _is_trade_seen(self, trade_id: str) -> bool:
        if not trade_id:
            return False
        ts = self._seen_trade_ids.get(trade_id)
        return bool(ts and (time.time() - ts) <= self.seen_ttl_sec)

    def _mark_trade_seen(self, trade_id: str) -> None:
        if trade_id:
            self._seen_trade_ids[trade_id] = time.time()

    async def _reconcile_orders(self, symbols: List[str], startup: bool) -> Dict[str, int]:
        stats = {"fills_recovered": 0, "fills_skipped": 0, "sell_missing_canonical": 0}
        if not symbols or self.exchange_client is None:
            return stats

        now_ms = int(time.time() * 1000)
        for sym in symbols:
            orders = await self._fetch_recent_orders(sym)
            if not orders:
                continue
            orders.sort(key=self._order_update_time_ms)
            for order in orders:
                if not self._is_filled_order(order):
                    continue
                order_id = self._order_id(order)
                if not order_id:
                    continue
                if self._is_order_seen(order_id):
                    continue

                age_ms = max(0, now_ms - self._order_update_time_ms(order))
                if not startup and self.recovery_window_sec > 0 and age_ms > (self.recovery_window_sec * 1000):
                    self._mark_order_seen(order_id)
                    continue

                already_applied = self._is_fill_likely_already_applied(order)
                if (
                    str(order.get("side", "")).upper() == "SELL"
                    and not already_applied
                    and not self._is_canonical_trade_event_present(order)
                ):
                    # Only emit the noisy warning for orders placed DURING this
                    # process lifetime.  Historical orders (pre-startup) will
                    # still be recovered via _apply_recovered_fill below, but
                    # the "SELL missing canonical" warning is suppressed because
                    # the in-memory event_log is empty on restart by design.
                    order_ts_s = self._order_update_time_ms(order) / 1000.0
                    is_post_startup = order_ts_s >= self._started_at
                    if is_post_startup:
                        stats["sell_missing_canonical"] += 1
                        await self._emit_missing_canonical_sell(order, startup=startup)
                    else:
                        self.logger.debug(
                            "[TruthAuditor] Pre-startup SELL (order_id=%s) missing canonical event – recovering silently",
                            self._order_id(order),
                        )

                if already_applied:
                    self._mark_order_seen(order_id)
                    stats["fills_skipped"] += 1
                    continue

                recovered = await self._apply_recovered_fill(order, reason="missed_fill_recovery", synthetic=False)
                self._mark_order_seen(order_id)
                if recovered:
                    stats["fills_recovered"] += 1
                else:
                    stats["fills_skipped"] += 1
        return stats

    async def _reconcile_open_orders(self, symbols: List[str]) -> Dict[str, int]:
        stats = {"open_orders": 0, "open_order_mismatch": 0}
        ec = self.exchange_client
        ss = self.shared_state
        if ec is None or ss is None:
            return stats

        exchange_open: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sym in symbols:
            open_orders = await self._fetch_open_orders(sym)
            if not open_orders:
                continue
            sym_map: Dict[str, Dict[str, Any]] = {}
            for order in open_orders:
                oid = self._order_id(order)
                if not oid:
                    continue
                sym_map[oid] = dict(order)
            if sym_map:
                exchange_open[sym] = sym_map
                stats["open_orders"] += len(sym_map)

        local_open = getattr(ss, "open_orders", None)
        if not isinstance(local_open, dict):
            local_open = {}

        local_map = self._normalize_open_order_map(local_open)
        stale = []
        for sym, ids in local_map.items():
            xids = set(exchange_open.get(sym, {}).keys())
            for oid in ids.keys():
                if oid not in xids:
                    stale.append((sym, oid))
        stats["open_order_mismatch"] = len(stale)

        # Keep local cache aligned to exchange truth.
        ss.open_orders = {sym: dict(orders) for sym, orders in exchange_open.items()}
        if stale:
            await self._emit_event(
                "TRUTH_AUDIT_OPEN_ORDER_MISMATCH",
                {
                    "count": len(stale),
                    "stale_orders": [{"symbol": s, "order_id": oid} for s, oid in stale[:50]],
                    "ts": time.time(),
                },
            )

        # Always publish a dedicated truth snapshot.
        try:
            setattr(ss, "truth_open_orders", {sym: dict(orders) for sym, orders in exchange_open.items()})
        except Exception:
            pass
        return stats

    def _normalize_open_order_map(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for sym, payload in (raw or {}).items():
            sym_u = str(sym or "").upper()
            if not sym_u:
                continue
            if isinstance(payload, dict):
                sym_map: Dict[str, Dict[str, Any]] = {}
                for k, v in payload.items():
                    oid = str(k)
                    if not oid and isinstance(v, dict):
                        oid = self._order_id(v)
                    if oid:
                        sym_map[oid] = dict(v) if isinstance(v, dict) else {"raw": v}
                if sym_map:
                    out[sym_u] = sym_map
            elif isinstance(payload, list):
                sym_map = {}
                for item in payload:
                    if isinstance(item, dict):
                        oid = self._order_id(item)
                        if oid:
                            sym_map[oid] = dict(item)
                if sym_map:
                    out[sym_u] = sym_map
        return out

    async def _fetch_recent_orders(self, symbol: str) -> List[Dict[str, Any]]:
        ec = self.exchange_client
        if ec is None:
            return []
        if not self._has_signed_access():
            return []
        sym = str(symbol or "").upper()
        if not sym:
            return []
        try:
            if hasattr(ec, "get_all_orders"):
                orders = await self._maybe_await(ec.get_all_orders(sym, limit=self.order_history_limit))
                if isinstance(orders, list):
                    return orders
        except Exception:
            self.logger.debug("get_all_orders failed for %s", sym, exc_info=True)
        try:
            client = getattr(ec, "client", None)
            if client and hasattr(client, "get_all_orders"):
                orders = await client.get_all_orders(symbol=sym, limit=int(self.order_history_limit))
                if isinstance(orders, list):
                    return orders
        except Exception:
            self.logger.debug("client.get_all_orders failed for %s", sym, exc_info=True)
        return []

    async def _fetch_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        ec = self.exchange_client
        if ec is None:
            return []
        if not self._has_signed_access():
            return []
        sym = str(symbol or "").upper()
        if not sym:
            return []
        try:
            if hasattr(ec, "get_open_orders"):
                orders = await self._maybe_await(ec.get_open_orders(sym))
                if isinstance(orders, list):
                    return orders
        except Exception:
            self.logger.debug("get_open_orders failed for %s", sym, exc_info=True)
        try:
            client = getattr(ec, "client", None)
            if client and hasattr(client, "get_open_orders"):
                orders = await client.get_open_orders(symbol=sym)
                if isinstance(orders, list):
                    return orders
        except Exception:
            self.logger.debug("client.get_open_orders failed for %s", sym, exc_info=True)
        return []

    def _is_filled_order(self, order: Dict[str, Any]) -> bool:
        status = str((order or {}).get("status", "")).upper()
        if status not in {"FILLED", "PARTIALLY_FILLED"}:
            return False
        try:
            return float((order or {}).get("executedQty", 0.0) or 0.0) > 0.0
        except Exception:
            return False

    def _order_id(self, order: Dict[str, Any]) -> str:
        if not isinstance(order, dict):
            return ""
        oid = order.get("orderId") or order.get("exchange_order_id") or order.get("order_id")
        if oid is None:
            return ""
        return str(oid)

    def _order_client_id(self, order: Dict[str, Any]) -> str:
        if not isinstance(order, dict):
            return ""
        cid = order.get("clientOrderId") or order.get("origClientOrderId") or order.get("client_order_id")
        if cid is None:
            return ""
        return str(cid)

    def _infer_order_source(self, order: Dict[str, Any]) -> Dict[str, str]:
        """
        Infer source/tag from client order id for provenance debugging.
        Expected internal pattern: octi-<timestamp>-<tag>.
        """
        cid = self._order_client_id(order)
        tag_hint = ""
        source_hint = "external_or_unknown"

        if cid:
            parts = str(cid).split("-", 2)
            if len(parts) >= 3 and parts[0].lower() == "octi":
                tag_hint = parts[2]
            else:
                tag_hint = str(cid)

            t = tag_hint.lower()
            if "tpsl" in t or "tp_sl" in t or "stop" in t:
                source_hint = "tp_sl_engine"
            elif "liq" in t:
                source_hint = "liquidation"
            elif "meta" in t:
                source_hint = "meta_controller"
            elif "exec" in t:
                source_hint = "execution_manager"
            elif "truth" in t:
                source_hint = "exchange_truth_auditor"

        return {
            "client_order_id": cid,
            "order_tag_hint": tag_hint,
            "order_source_hint": source_hint,
        }

    def _order_update_time_ms(self, order: Dict[str, Any]) -> int:
        if not isinstance(order, dict):
            return 0
        for key in ("updateTime", "time", "transactTime", "workingTime"):
            raw = order.get(key)
            try:
                if raw is not None:
                    return int(float(raw))
            except Exception:
                continue
        return 0

    def _is_order_seen(self, order_id: str) -> bool:
        if not order_id:
            return False
        ts = self._seen_order_ids.get(order_id)
        return bool(ts and (time.time() - ts) <= self.seen_ttl_sec)

    def _mark_order_seen(self, order_id: str) -> None:
        if order_id:
            self._seen_order_ids[order_id] = time.time()

    def _prune_seen_cache(self) -> None:
        now = time.time()
        ttl = max(60.0, float(self.seen_ttl_sec))
        if self._seen_order_ids:
            for oid, ts in list(self._seen_order_ids.items()):
                if now - ts > ttl:
                    self._seen_order_ids.pop(oid, None)
        if self._seen_trade_ids:
            for tid, ts in list(self._seen_trade_ids.items()):
                if now - ts > ttl:
                    self._seen_trade_ids.pop(tid, None)
        for sym, ts in list(self._recent_phantom_by_symbol.items()):
            if now - ts > max(self.phantom_cooldown_sec * 4.0, 60.0):
                self._recent_phantom_by_symbol.pop(sym, None)

    def _is_fill_likely_already_applied(self, order: Dict[str, Any]) -> bool:
        ss = self.shared_state
        if ss is None:
            return False
        history = getattr(ss, "trade_history", None)
        if history is None:
            return False
        try:
            rows = list(history)[-200:]
        except Exception:
            return False
        if not rows:
            return False

        sym = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "")).upper()
        try:
            qty = float(order.get("executedQty", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        price = self._resolve_order_price(order)
        order_ts = self._order_update_time_ms(order) / 1000.0 if self._order_update_time_ms(order) > 0 else 0.0

        for tr in reversed(rows):
            if not isinstance(tr, dict):
                continue
            if str(tr.get("symbol", "")).upper() != sym:
                continue
            if str(tr.get("side", "")).upper() != side:
                continue
            try:
                tr_qty = float(tr.get("qty", 0.0) or 0.0)
            except Exception:
                tr_qty = 0.0
            if tr_qty <= 0 or qty <= 0:
                continue
            if abs(tr_qty - qty) > max(self.position_mismatch_tol, qty * 0.003):
                continue
            try:
                tr_price = float(tr.get("price", 0.0) or 0.0)
            except Exception:
                tr_price = 0.0
            if price > 0 and tr_price > 0 and abs(tr_price - price) > max(1e-8, price * 0.005):
                continue
            try:
                tr_ts = float(tr.get("ts", 0.0) or 0.0)
            except Exception:
                tr_ts = 0.0
            if order_ts > 0 and tr_ts > 0 and abs(tr_ts - order_ts) > self.trade_match_window_sec:
                continue
            return True
        return False

    def _resolve_order_price(self, order: Dict[str, Any]) -> float:
        try:
            p = float(order.get("avgPrice", 0.0) or order.get("price", 0.0) or 0.0)
        except Exception:
            p = 0.0
        if p > 0:
            return p
        try:
            qty = float(order.get("executedQty", 0.0) or 0.0)
            cq = float(order.get("cummulativeQuoteQty", 0.0) or 0.0)
            if qty > 0 and cq > 0:
                return cq / qty
        except Exception:
            pass
        return 0.0

    def _is_canonical_trade_event_present(self, order: Dict[str, Any]) -> bool:
        """
        Check if a SELL fill appears in recent TRADE_EXECUTED events from canonical sources.
        Excludes truth-auditor generated events to detect off-canonical executions.
        Uses two detection layers:
          1. EM's _trade_event_emit_cache (O(1) lookup, survives event_log overflow)
          2. SharedState._event_log scan (bounded deque)
        """
        ss = self.shared_state
        if ss is None:
            return False

        sym = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "")).upper()
        if not sym or side != "SELL":
            return False
        oid = self._order_id(order)
        cid = self._order_client_id(order)

        # --- Layer 1: Check EM's emit cache (fast, survives event_log eviction) ---
        em = getattr(self.app, "execution_manager", None) if self.app else None
        if em is None:
            em = getattr(ss, "execution_manager", None)
        if em is not None:
            emit_cache = getattr(em, "_trade_event_emit_cache", None)
            if isinstance(emit_cache, dict):
                if oid:
                    cache_key = f"{sym}:SELL:OID:{oid}"
                    if emit_cache.get(cache_key):
                        return True
                if cid:
                    cache_key = f"{sym}:SELL:CID:{cid}"
                    if emit_cache.get(cache_key):
                        return True

        # --- Layer 2: Scan event_log ---
        events = getattr(ss, "_event_log", None)
        if events is None:
            return False
        try:
            rows = list(events)[-max(100, int(self.event_scan_limit or 800)):]
        except Exception:
            return False
        if not rows:
            return False
        qty = 0.0
        price = self._resolve_order_price(order)
        order_ts = self._order_update_time_ms(order) / 1000.0 if self._order_update_time_ms(order) > 0 else 0.0
        with contextlib.suppress(Exception):
            qty = float(order.get("executedQty", 0.0) or 0.0)

        qty_tol = max(self.position_mismatch_tol, qty * 0.003)
        price_tol = max(1e-8, price * 0.005) if price > 0 else 0.0

        for ev in reversed(rows):
            if not isinstance(ev, dict):
                continue
            if str(ev.get("name", "")).upper() != "TRADE_EXECUTED":
                continue
            data = ev.get("data")
            if not isinstance(data, dict):
                continue

            source = str(data.get("source", "") or "")
            if source == "ExchangeTruthAuditor":
                continue

            data_oid = str(data.get("order_id") or data.get("orderId") or data.get("exchange_order_id") or "")
            if oid and data_oid and data_oid == oid:
                return True
            data_cid = str(data.get("client_order_id") or data.get("clientOrderId") or data.get("origClientOrderId") or "")
            if cid and data_cid and data_cid == cid:
                return True

            if str(data.get("symbol", "")).upper() != sym:
                continue
            if str(data.get("side", "")).upper() != side:
                continue

            ev_ts = 0.0
            with contextlib.suppress(Exception):
                ev_ts = float(ev.get("timestamp", 0.0) or data.get("timestamp", 0.0) or data.get("ts", 0.0) or 0.0)
            if order_ts > 0 and ev_ts > 0 and abs(order_ts - ev_ts) > self.trade_match_window_sec:
                continue

            if qty > 0:
                ev_qty = 0.0
                with contextlib.suppress(Exception):
                    ev_qty = float(data.get("executed_qty", data.get("qty", 0.0)) or 0.0)
                if ev_qty > 0 and abs(ev_qty - qty) > qty_tol:
                    continue

            if price > 0:
                ev_price = 0.0
                with contextlib.suppress(Exception):
                    ev_price = float(data.get("avg_price", data.get("price", 0.0)) or 0.0)
                if ev_price > 0 and abs(ev_price - price) > price_tol:
                    continue

            return True
        return False

    async def _emit_missing_canonical_sell(self, order: Dict[str, Any], startup: bool) -> None:
        sym = str(order.get("symbol", "")).upper()
        oid = self._order_id(order)
        source_meta = self._infer_order_source(order)
        qty = float(order.get("executedQty", 0.0) or 0.0)
        px = float(self._resolve_order_price(order) or 0.0)
        payload = {
            "symbol": sym,
            "side": "SELL",
            "order_id": oid,
            "client_order_id": source_meta.get("client_order_id"),
            "order_tag_hint": source_meta.get("order_tag_hint"),
            "order_source_hint": source_meta.get("order_source_hint"),
            "status": str(order.get("status", "")).upper(),
            "executed_qty": qty,
            "price": px,
            "exchange_update_time_ms": self._order_update_time_ms(order),
            "startup_scan": bool(startup),
            "reason": "SELL_EXECUTED_WITHOUT_CANONICAL_TRADE_EXECUTED",
            "source": "ExchangeTruthAuditor",
            "timestamp": time.time(),
        }
        self.logger.warning(
            "[TruthAuditor] SELL missing canonical TRADE_EXECUTED: symbol=%s order_id=%s client_order_id=%s source_hint=%s qty=%.8f price=%.8f",
            sym,
            oid or "n/a",
            source_meta.get("client_order_id") or "n/a",
            source_meta.get("order_source_hint") or "external_or_unknown",
            qty,
            px,
        )
        await self._emit_event("TRUTH_AUDIT_SELL_MISSING_CANONICAL", payload)

    async def _apply_recovered_fill(self, order: Dict[str, Any], reason: str, synthetic: bool) -> bool:
        ss = self.shared_state
        if ss is None or not isinstance(order, dict):
            return False

        sym = str(order.get("symbol", "")).upper()
        side = str(order.get("side", "")).upper()
        if not sym or side not in {"BUY", "SELL"}:
            return False

        try:
            qty = float(order.get("executedQty", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            return False

        price = self._resolve_order_price(order)
        if price <= 0 and self.exchange_client and hasattr(self.exchange_client, "get_current_price"):
            with contextlib.suppress(Exception):
                price = float(await self.exchange_client.get_current_price(sym) or 0.0)
        if price <= 0:
            return False

        fee_quote = 0.0
        fee_base = 0.0
        try:
            base_asset, quote_asset = self._split_base_quote(sym)
            fills = order.get("fills") or []
            if isinstance(fills, list):
                for f in fills:
                    if not isinstance(f, dict):
                        continue
                    commission = 0.0
                    with contextlib.suppress(Exception):
                        commission = float(f.get("commission", 0.0) or 0.0)
                    if commission <= 0:
                        continue
                    asset = str(f.get("commissionAsset") or f.get("commission_asset") or "").upper()
                    if asset == quote_asset:
                        fee_quote += commission
                    elif asset == base_asset:
                        fee_base += commission
            if fee_quote <= 0:
                fee_quote = float(order.get("fee_quote", 0.0) or order.get("fee", 0.0) or 0.0)
            if fee_base <= 0:
                fee_base = float(order.get("fee_base", 0.0) or 0.0)
        except Exception:
            fee_quote = 0.0
            fee_base = float(order.get("fee_base", 0.0) or 0.0)

        applied = False
        try:
            if hasattr(ss, "record_trade"):
                await self._maybe_call(ss, "record_trade", sym, side, qty, price, fee_quote, fee_base, "truth_auditor")
                applied = True
            elif hasattr(ss, "record_fill"):
                await self._maybe_call(ss, "record_fill", sym, side, qty, price, fee_quote=fee_quote, fee_base=fee_base)
                applied = True
        except Exception:
            self.logger.error("[TruthAuditor] failed applying recovered fill %s %s", sym, side, exc_info=True)

        if side == "SELL":
            # Ensure stale open lots are finalised even if fill hooks were missed.
            try:
                pos_qty = 0.0
                if hasattr(ss, "get_position_qty"):
                    pos_qty = float(ss.get_position_qty(sym) or 0.0)
                if pos_qty > self.dust_threshold and hasattr(ss, "mark_position_closed"):
                    await self._maybe_call(
                        ss,
                        "mark_position_closed",
                        symbol=sym,
                        qty=qty,
                        price=price,
                        reason=f"TRUTH_AUDIT:{reason}",
                        tag="truth_auditor",
                    )
            except Exception:
                self.logger.debug("mark_position_closed fallback failed for %s", sym, exc_info=True)

        # --- New: emit canonical TRADE_EXECUTED via ExecutionManager when available ---
        try:
            em = None
            with contextlib.suppress(Exception):
                em = getattr(self.app, "execution_manager", None) or getattr(self.shared_state, "execution_manager", None)
            if em is not None and callable(getattr(em, "_emit_trade_executed_event", None)):
                # Best-effort — ExecutionManager will dedupe and honour canonical contract
                try:
                    await em._emit_trade_executed_event(sym, side, str(order.get("tag") or "truth_auditor"), order)
                except Exception:
                    self.logger.debug("ExecutionManager canonical re-emit failed for recovered fill %s", order.get("orderId") or self._order_id(order), exc_info=True)
        except Exception:
            # Nothing fatal if we cannot reach ExecutionManager
            self.logger.debug("ExchangeTruthAuditor: emit to ExecutionManager failed", exc_info=True)

        payload = {
            "symbol": sym,
            "side": side,
            "status": "reconciled" if synthetic else "filled_recovered",
            "reason": reason,
            "order_id": self._order_id(order),
            "client_order_id": self._order_client_id(order),
            "qty": float(qty),
            "price": float(price),
            "tag": "truth_auditor",
            "source": "ExchangeTruthAuditor",
            "timestamp": time.time(),
        }
        payload.update(self._infer_order_source(order))
        await self._emit_event("TRUTH_AUDIT_TRADE_EXECUTED_RECOVERED", payload)
        await self._emit_event("TRUTH_AUDIT_FILL_RECOVERED", payload)
        return applied or synthetic
