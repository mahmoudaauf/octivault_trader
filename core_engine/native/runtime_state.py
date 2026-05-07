"""
Native runtime-state persistence.

Persists a compact snapshot of the last known shared state so the bot can
restart into a meaningful read-only view during exchange throttling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _account_payload(
    *,
    nav_usdt: float,
    free_balance_usdt: float,
    invested_capital_usdt: float,
    balance: dict[str, Any],
    session_anchor_nav: float,
    ts: float,
) -> dict[str, Any]:
    return {
        "ts": float(ts or 0.0),
        "nav_usdt": float(nav_usdt or 0.0),
        "free_balance_usdt": float(free_balance_usdt or 0.0),
        "invested_capital_usdt": float(invested_capital_usdt or 0.0),
        "balance": {str(k).upper(): float(v or 0.0) for k, v in (balance or {}).items()},
        "session_anchor_nav": float(session_anchor_nav or 0.0),
    }


def _is_good_account_state(payload: dict[str, Any]) -> bool:
    nav = float(payload.get("nav_usdt", 0.0) or 0.0)
    free = float(payload.get("free_balance_usdt", 0.0) or 0.0)
    invested = float(payload.get("invested_capital_usdt", 0.0) or 0.0)
    balance = payload.get("balance", {}) or {}
    usdt = float(balance.get("USDT", 0.0) or 0.0)
    return any(v > 0.0 for v in (nav, free, invested, usdt))


class NativeRuntimeStateExporter:
    """Periodic JSON snapshot writer for NativeSharedState."""

    def __init__(self, shared_state: Any, output_path: Path, interval_sec: float = 10.0) -> None:
        if interval_sec <= 0:
            raise ValueError(f"interval_sec must be > 0, got {interval_sec}")
        self._state = shared_state
        self._output_path = Path(output_path)
        self._interval_sec = float(interval_sec)
        self._task: asyncio.Task[None] | None = None
        self._stopping: asyncio.Event | None = None

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if self._stopping is None:
            self._stopping = asyncio.Event()
        self._stopping.clear()
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._task = asyncio.create_task(self._run_loop(), name="native-runtime-state-exporter")

    async def stop(self) -> None:
        if self._stopping is None:
            self._stopping = asyncio.Event()
        self._stopping.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=self._interval_sec + 1.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._write_snapshot()

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                self._write_snapshot()
            except Exception as e:
                logger.warning("runtime-state exporter write failed: %r", e)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                continue

    def _build_payload(self) -> dict[str, Any]:
        positions_raw = getattr(self._state, "positions", {}) or {}
        positions = {
            sym: {
                "qty": float(getattr(pos, "qty", 0.0) or 0.0),
                "entry_price": float(getattr(pos, "entry_price", 0.0) or 0.0),
                "mark_price": float(getattr(pos, "mark_price", 0.0) or 0.0),
            }
            for sym, pos in positions_raw.items()
        }
        payload = {
            "ts": time.time(),
            "nav_usdt": float(getattr(self._state, "nav_usdt", 0.0) or 0.0),
            "free_balance_usdt": float(getattr(self._state, "free_balance_usdt", 0.0) or 0.0),
            "invested_capital_usdt": float(
                getattr(self._state, "invested_capital_usdt", 0.0) or 0.0
            ),
            "balance": dict(getattr(self._state, "balance", {}) or {}),
            "prices": dict(getattr(self._state, "prices", {}) or {}),
            "positions": positions,
            "session_anchor_nav": float(getattr(self._state, "session_anchor_nav", 0.0) or 0.0),
            "exchange_throttled": bool(getattr(self._state, "exchange_throttled", False)),
            "exchange_throttle_reason": str(
                getattr(self._state, "exchange_throttle_reason", "") or ""
            ),
            "exchange_throttle_until_ts": float(
                getattr(self._state, "exchange_throttle_until_ts", 0.0) or 0.0
            ),
            "metrics": dict(getattr(self._state, "metrics", {}) or {}),
        }
        payload["last_known_good_account_state"] = _account_payload(
            nav_usdt=payload["nav_usdt"],
            free_balance_usdt=payload["free_balance_usdt"],
            invested_capital_usdt=payload["invested_capital_usdt"],
            balance=payload["balance"],
            session_anchor_nav=payload["session_anchor_nav"],
            ts=payload["ts"],
        )
        if self._output_path.exists():
            try:
                previous = json.loads(self._output_path.read_text())
                previous_good = previous.get("last_known_good_account_state", {}) or {}
                if not _is_good_account_state(
                    payload["last_known_good_account_state"]
                ) and _is_good_account_state(previous_good):
                    payload["last_known_good_account_state"] = previous_good
            except Exception:
                pass
        return payload

    def _write_snapshot(self) -> None:
        payload = self._build_payload()
        target = self._output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def load_runtime_state(shared_state: Any, input_path: Path) -> bool:
    """Best-effort hydrate shared_state from the last saved runtime snapshot."""
    try:
        path = Path(input_path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text())
        account_payload = payload
        last_good = payload.get("last_known_good_account_state", {}) or {}
        if (
            bool(payload.get("exchange_throttled", False))
            and not _is_good_account_state(payload)
            and _is_good_account_state(last_good)
        ):
            account_payload = dict(last_good)
        shared_state.nav_usdt = float(account_payload.get("nav_usdt", 0.0) or 0.0)
        shared_state.free_balance_usdt = float(account_payload.get("free_balance_usdt", 0.0) or 0.0)
        shared_state.invested_capital_usdt = float(
            account_payload.get("invested_capital_usdt", 0.0) or 0.0
        )
        shared_state.balance = {
            str(k).upper(): float(v or 0.0)
            for k, v in (account_payload.get("balance", {}) or {}).items()
        }
        shared_state.prices = {
            str(k).upper(): float(v or 0.0) for k, v in (payload.get("prices", {}) or {}).items()
        }
        shared_state.price_cache = dict(shared_state.prices)
        now_ts = time.time()
        if hasattr(shared_state, "_last_tick_timestamps"):
            shared_state._last_tick_timestamps = {
                str(sym).upper(): now_ts for sym in shared_state.prices
            }
        shared_state.session_anchor_nav = float(
            account_payload.get("session_anchor_nav", 0.0) or 0.0
        )
        shared_state.set_exchange_throttle(
            bool(payload.get("exchange_throttled", False)),
            reason=str(payload.get("exchange_throttle_reason", "") or ""),
            until_ts=float(payload.get("exchange_throttle_until_ts", 0.0) or 0.0),
        )
        metrics = payload.get("metrics", {}) or {}
        if isinstance(metrics, dict):
            shared_state.metrics.update(metrics)
        positions = payload.get("positions", {}) or {}
        for sym, pos in positions.items():
            try:
                shared_state.update_position(
                    symbol=str(sym).upper(),
                    qty=float(pos.get("qty", 0.0) or 0.0),
                    entry=float(pos.get("entry_price", 0.0) or 0.0),
                    current=float(pos.get("mark_price", 0.0) or 0.0),
                )
            except Exception:
                continue
        return True
    except Exception:
        logger.debug("load_runtime_state failed", exc_info=True)
        return False


__all__ = ["NativeRuntimeStateExporter", "load_runtime_state"]
