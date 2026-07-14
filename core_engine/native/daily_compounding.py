"""Realized-equity daily compounding policy.

The policy freezes the sizing NAV during a UTC trading day. Profits may raise the
sizing base only after the day rolls over and the portfolio is flat; losses reduce
the usable sizing NAV immediately. This prevents leverage from expanding against
intraday or unrealized gains while still compounding secured net account growth.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyCompoundingState:
    sizing_date: str = ""
    sizing_nav_usdt: float = 0.0
    pending_rollover: bool = False


class DailyCompoundingPolicy:
    """Hold gains until a flat-portfolio UTC day boundary, but de-risk losses now."""

    def __init__(self, *, enabled: bool = True, state_path: str | Path | None = None) -> None:
        self.enabled = bool(enabled)
        self.state_path = Path(state_path) if state_path else None
        self.state = DailyCompoundingState()
        self._load()

    def sizing_nav(
        self,
        current_nav_usdt: float,
        *,
        has_open_positions: bool,
        now: Optional[datetime] = None,
        protection_floor_usdt: float = 0.0,
    ) -> float:
        """Return the NAV that position sizing is allowed to use.

        ``protection_floor_usdt`` is NAVProtectionEngine's drawdown-tiered floor
        (shared_state.nav_protection_state["protection_floor_usdt"]) — the
        codebase's only "protected profit reserve" concept. When positive, the
        result is additionally capped at ``current_nav - protection_floor_usdt``
        (floored at 0), so sizing never implies risking capital NAV protection
        has already flagged as locked/reserved. This is the connection
        remediation item #17 asked for: the floor now actually constrains
        position sizing instead of only being computed and logged.
        """
        current_nav = max(0.0, float(current_nav_usdt or 0.0))
        floor = max(0.0, float(protection_floor_usdt or 0.0))

        def _apply_floor(nav: float) -> float:
            if floor <= 0:
                return nav
            return max(0.0, min(nav, current_nav - floor))

        if not self.enabled or current_nav <= 0:
            return _apply_floor(current_nav)

        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=timezone.utc)
        today = instant.astimezone(timezone.utc).date().isoformat()

        if self.state.sizing_nav_usdt <= 0 or not self.state.sizing_date:
            self.state = DailyCompoundingState(
                sizing_date=today,
                sizing_nav_usdt=current_nav,
                pending_rollover=False,
            )
            self._persist()
            return _apply_floor(current_nav)

        if today > self.state.sizing_date:
            if has_open_positions:
                if not self.state.pending_rollover:
                    self.state.pending_rollover = True
                    self._persist()
                logger.info(
                    "[compounding] UTC rollover pending: portfolio is not flat; "
                    "holding sizing NAV at $%.2f",
                    self.state.sizing_nav_usdt,
                )
            else:
                previous = self.state.sizing_nav_usdt
                self.state = DailyCompoundingState(
                    sizing_date=today,
                    sizing_nav_usdt=current_nav,
                    pending_rollover=False,
                )
                self._persist()
                logger.info(
                    "[compounding] committed daily net NAV: $%.2f -> $%.2f (%+.2f%%)",
                    previous,
                    current_nav,
                    ((current_nav / previous) - 1.0) * 100.0 if previous > 0 else 0.0,
                )

        # Never wait until tomorrow to react to a loss. Positive intraday/unrealized
        # NAV is capped at the last committed daily base.
        return _apply_floor(min(current_nav, self.state.sizing_nav_usdt))

    def snapshot(self) -> dict[str, object]:
        return asdict(self.state)

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            sizing_date = str(payload.get("sizing_date", "") or "")
            sizing_nav = float(payload.get("sizing_nav_usdt", 0.0) or 0.0)
            if sizing_date and sizing_nav > 0:
                self.state = DailyCompoundingState(
                    sizing_date=sizing_date,
                    sizing_nav_usdt=sizing_nav,
                    pending_rollover=bool(payload.get("pending_rollover", False)),
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("[compounding] invalid state file ignored: %s", self.state_path)

    def _persist(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(asdict(self.state), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, self.state_path)
        except OSError:
            logger.warning("[compounding] failed to persist state: %s", self.state_path)
