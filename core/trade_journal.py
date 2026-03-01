"""
TradeJournal — Persistent, crash-safe execution journal.

Append-only JSONL writer with:
  • Thread-safe writes (threading.Lock)
  • Immediate disk flush (os.fsync) — survives process crash
  • Date-based rotation: logs/trade_journal_YYYYMMDD.jsonl
  • Zero dependency on SharedState or Config (standalone)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("TradeJournal")


class TradeJournal:
    """Append-only JSONL trade journal with fsync durability."""

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._fd: Optional[int] = None
        self._current_date: Optional[str] = None
        os.makedirs(self._log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Write one JSON line to the journal.

        Parameters
        ----------
        event_type : str
            e.g. "ORDER_SUBMITTED", "ORDER_FILLED", "TPSL_TRIGGER"
        data : dict
            Arbitrary payload — will be merged with envelope fields.
        """
        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(),
            "epoch": time.time(),
            "event": event_type,
            **data,
        }
        line = json.dumps(record, default=str) + "\n"

        with self._lock:
            try:
                self._ensure_fd(now)
                os.write(self._fd, line.encode())
                os.fsync(self._fd)
            except Exception:
                LOGGER.exception("TradeJournal write failed")

    def close(self) -> None:
        """Close the underlying file descriptor (for clean shutdown)."""
        with self._lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
                self._current_date = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fd(self, now: datetime) -> None:
        """Open (or rotate) the file descriptor for today's date."""
        today = now.strftime("%Y%m%d")
        if self._fd is not None and self._current_date == today:
            return

        # Close previous day's fd
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass

        path = os.path.join(self._log_dir, f"trade_journal_{today}.jsonl")
        self._fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self._current_date = today
        LOGGER.info("TradeJournal opened: %s", path)
