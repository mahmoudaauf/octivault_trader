"""
Native L1.5-ish: Carry Strategy State — the architectural spine of the
funding-carry native-wiring plan (Phase 2).

Deliberately does NOT touch core_engine/native/shared_state.py's Position
model or NativeSharedState.positions. That model is structurally spot-only
(unsigned qty, `unrealized_pnl_pct` assumes a long, `position_value` is a
naive qty*mark spot notional) -- forcing a delta-neutral, two-leg,
spot+perpetual-futures position into it would corrupt NAV math and exposure
accounting used live by the ML-forecaster strategy right now. Instead, carry
gets its own self-contained state/NAV subsystem here, which reports into the
top-level NAV through exactly one narrow seam: locked_capital_usd() (see
NAV-integration note on CarrySharedState below).

Persistence path deviation from the original plan text (found during
implementation, not a plan-following bug): the standalone carry_paper_trader.py
daemon is still actively running against logs/carry_state.json and
logs/carry_ledger.jsonl. Reusing those exact paths here would let two
independent processes race-write the same files once this module is ever
wired into the live main.py process (Phase 7) while the standalone daemon is
still running too. This module defaults to DISTINCT paths
(logs/native_carry_state.json, logs/native_carry_ledger.jsonl) -- same JSON
schema, different files -- so the two systems can never collide on disk.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_STATE_PATH = "logs/native_carry_state.json"
DEFAULT_LEDGER_PATH = "logs/native_carry_ledger.jsonl"


@dataclass
class HedgePosition:
    """One open delta-neutral hedge: short perp + long spot on the same
    underlying (v1 is POSITIVE_ONLY -- see carry_paper_trader.py -- so
    direction is effectively always "short_perp" in practice; "long_perp" is
    kept for interface symmetry but is not currently reachable, matching the
    standalone script's own unreachable branch)."""

    symbol: str
    entry_ts: float
    entry_funding: float
    direction: str  # "short_perp" | "long_perp"
    perp_qty: float
    spot_qty: float
    notional_usd: float  # per-leg notional at entry (1:1 hedge ratio)

    def held_h(self, now: Optional[float] = None) -> float:
        return ((now or time.time()) - self.entry_ts) / 3600.0


class CarrySharedState:
    """Self-contained state for open/closed carry hedges. Pure state + disk
    persistence -- no exchange-client or network dependency, so it's testable
    without mocking I/O beyond the filesystem."""

    def __init__(
        self,
        *,
        state_path: str = DEFAULT_STATE_PATH,
        ledger_path: str = DEFAULT_LEDGER_PATH,
    ) -> None:
        self.state_path = Path(state_path)
        self.ledger_path = Path(ledger_path)
        self.open_hedges: dict[str, HedgePosition] = {}
        self._load()

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle: open / close
    # ──────────────────────────────────────────────────────────────────
    def open_hedge(
        self,
        symbol: str,
        *,
        entry_funding: float,
        perp_qty: float,
        spot_qty: float,
        notional_usd: float,
        now: Optional[float] = None,
    ) -> HedgePosition:
        if symbol in self.open_hedges:
            raise ValueError(f"{symbol} already has an open carry hedge")
        pos = HedgePosition(
            symbol=symbol,
            entry_ts=now if now is not None else time.time(),
            entry_funding=entry_funding,
            direction="short_perp" if entry_funding > 0 else "long_perp",
            perp_qty=perp_qty,
            spot_qty=spot_qty,
            notional_usd=notional_usd,
        )
        self.open_hedges[symbol] = pos
        self._persist_state()
        return pos

    def close_hedge(self, symbol: str) -> Optional[HedgePosition]:
        pos = self.open_hedges.pop(symbol, None)
        if pos is not None:
            self._persist_state()
        return pos

    def get_open_hedge(self, symbol: str) -> Optional[HedgePosition]:
        return self.open_hedges.get(symbol)

    def open_count(self) -> int:
        return len(self.open_hedges)

    def open_symbols(self) -> list[str]:
        return list(self.open_hedges.keys())

    # ──────────────────────────────────────────────────────────────────
    # NAV-integration seam (the one narrow point of contact with the rest
    # of the native runtime's capital accounting)
    # ──────────────────────────────────────────────────────────────────
    def locked_capital_usd(self) -> float:
        """Collateral tied up in open hedges (approximated as entry notional,
        not re-marked-to-market -- a delta-neutral position's notional
        shouldn't move dramatically, and re-marking would require a live
        price feed this pure-state module deliberately doesn't depend on).
        This is the term that must REDUCE capital available to the spot
        strategy's own allocator -- see the plan doc's NAV-injection note."""
        return sum(p.notional_usd for p in self.open_hedges.values())

    def net_exposure_usd(
        self, mark_prices: dict[str, dict[str, float]]
    ) -> tuple[float, bool]:
        """Net directional exposure across all open hedges, using CALLER-
        SUPPLIED mark prices (this module has no exchange-client dependency
        by design). Should be ~0 for a correctly-hedged book.

        mark_prices: {symbol: {"perp": mark_price, "spot": mark_price}}.

        Returns (net_exposure_usd, healthy) -- healthy=False if any open
        symbol is missing a price (the caller should treat the exposure
        figure as stale/unreliable in that case, not silently trust a
        partial sum).
        """
        total = 0.0
        healthy = True
        for symbol, pos in self.open_hedges.items():
            px = mark_prices.get(symbol)
            if not px or "perp" not in px or "spot" not in px:
                healthy = False
                continue
            spot_value = pos.spot_qty * px["spot"]
            perp_value = pos.perp_qty * px["perp"]
            # short_perp: the perp leg is a liability offsetting the spot
            # long (delta ~0 when spot_qty*spot_px ~= perp_qty*perp_px).
            # long_perp: not reachable under v1's POSITIVE_ONLY restriction
            # (see HedgePosition docstring) -- kept symmetric regardless.
            sign = -1.0 if pos.direction == "short_perp" else 1.0
            total += spot_value + sign * perp_value
        return total, healthy

    # ──────────────────────────────────────────────────────────────────
    # Closed-trade ledger (mirrors carry_paper_trader.py's _log_trade schema
    # exactly, so any existing analysis tooling/mental model carries over)
    # ──────────────────────────────────────────────────────────────────
    def record_closed_trade(
        self,
        symbol: str,
        *,
        held_h: float,
        accrued_funding_pct: float,
        net_pct: float,
        exit_funding: float,
        mode: str = "paper",
    ) -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "held_h": round(held_h, 1),
            "accrued_funding_pct": round(accrued_funding_pct, 4),
            "net_pct": round(net_pct, 4),
            "exit_funding": exit_funding,
            "mode": mode,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def read_ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        trades: list[dict[str, Any]] = []
        with open(self.ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return trades

    # ──────────────────────────────────────────────────────────────────
    # Disk persistence (atomic write, mirrors daily_compounding.py's pattern)
    # ──────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for symbol, row in (payload.get("open") or {}).items():
            try:
                self.open_hedges[symbol] = HedgePosition(
                    symbol=symbol,
                    entry_ts=float(row["entry_ts"]),
                    entry_funding=float(row["entry_funding"]),
                    direction=str(row.get("direction", "short_perp")),
                    perp_qty=float(row["perp_qty"]),
                    spot_qty=float(row["spot_qty"]),
                    notional_usd=float(row["notional_usd"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"open": {sym: asdict(pos) for sym, pos in self.open_hedges.items()}}
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_path)


__all__ = ["CarrySharedState", "HedgePosition", "DEFAULT_STATE_PATH", "DEFAULT_LEDGER_PATH"]
