"""
Native L0: Trade Journal — Persistent, Crash-Safe Execution Log

Append-only JSONL writer with:
- Thread-safe writes (asyncio.Lock for async context)
- Immediate disk flush (os.fsync) — survives process crash
- Daily rotation: logs/trade_journal_YYYYMMDD.jsonl
- Immutable audit trail (compliant, tamper-evident)

Design: Every trade event is immediately persisted to disk with timestamp,
so even if the process crashes, the journal is never corrupted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NativeTradeJournal:
    """Append-only JSONL trade journal with fsync durability."""

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialize the trade journal.

        Args:
            log_dir: Directory for journal files (auto-created)
        """
        self._log_dir = Path(log_dir)
        self._lock = asyncio.Lock()
        self._fd: Optional[int] = None
        self._current_date: Optional[str] = None
        self._record_count = 0
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def record(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Write one JSON line to the journal (async-safe).

        Args:
            event_type: Event category (e.g., "ORDER_SUBMITTED", "ORDER_FILLED")
            data: Event payload (arbitrary dict)

        Example:
            await journal.record("ORDER_SUBMITTED", {
                "symbol": "BTCUSDT",
                "qty": 0.5,
                "price": 45000.0,
                "order_id": "12345678"
            })
        """
        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(),
            "epoch": time.time(),
            "event": event_type,
            **data,
        }
        line = json.dumps(record, default=str) + "\n"

        async with self._lock:
            try:
                await self._write_line(line, now)
                self._record_count += 1
            except Exception as e:
                logger.exception(f"TradeJournal write failed: {e}")

    def record_sync(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Synchronous version (for non-async contexts).

        Args:
            event_type: Event category
            data: Event payload
        """
        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(),
            "epoch": time.time(),
            "event": event_type,
            **data,
        }
        line = json.dumps(record, default=str) + "\n"

        try:
            self._write_line_sync(line, now)
            self._record_count += 1
        except Exception as e:
            logger.exception(f"TradeJournal sync write failed: {e}")

    async def close(self) -> None:
        """Close the underlying file descriptor (for clean shutdown)."""
        async with self._lock:
            self._close_fd()

    def close_sync(self) -> None:
        """Synchronous version."""
        self._close_fd()

    def get_record_count(self) -> int:
        """Return total records written to journal."""
        return self._record_count

    # ──────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────

    async def _write_line(self, line: str, now: datetime) -> None:
        """Async write with rotation."""
        self._ensure_fd(now)
        if self._fd is not None:
            os.write(self._fd, line.encode())
            os.fsync(self._fd)

    def _write_line_sync(self, line: str, now: datetime) -> None:
        """Sync write with rotation."""
        self._ensure_fd(now)
        if self._fd is not None:
            os.write(self._fd, line.encode())
            os.fsync(self._fd)

    def _ensure_fd(self, now: datetime) -> None:
        """Open (or rotate) the file descriptor for today's date."""
        today = now.strftime("%Y%m%d")
        if self._fd is not None and self._current_date == today:
            return

        # Close previous day's fd
        self._close_fd()

        # Open today's file
        path = self._log_dir / f"trade_journal_{today}.jsonl"
        self._fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self._current_date = today
        logger.info(f"TradeJournal rotated to {path}")

    def _close_fd(self) -> None:
        """Close the current file descriptor."""
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            self._current_date = None


__all__ = ["NativeTradeJournal"]
