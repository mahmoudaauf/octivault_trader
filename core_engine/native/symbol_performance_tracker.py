"""
SymbolPerformanceTracker — dynamic per-symbol edge scoring.

Scores every symbol on a rolling window of recent closed trades.
Symbols with poor recent performance are throttled (size multiplier reduced)
or soft-blocked (size multiplier = 0). They auto-recover when results improve
or after the cooldown window passes — no symbols are permanently banned.

Wired into arbitration gate_8 for BUY gating and capital_allocator for sizing.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

_STATE_PATH = os.path.join("logs", "symbol_perf.json")

_log = logging.getLogger(__name__)

# ── Tunables ─────────────────────────────────────────────────────────────────
_WINDOW           = 15      # rolling trades per symbol used for scoring
_MIN_TRADES       = 3       # need at least this many trades before throttling
# Breakeven WR with 0.2% round-trip fees is ~50%. Thresholds raised to reflect this.
_BLOCK_WIN_RATE   = 0.40    # win rate below this → soft-block (raised from 0.30)
_NO_PROBE_WIN_RATE = 0.25   # win rate below this → skip probe entirely, just re-block
_WARN_WIN_RATE    = 0.50    # win rate below this → half size (raised from 0.45)
_GOOD_WIN_RATE    = 0.60    # win rate above this → full size
_GREAT_WIN_RATE   = 0.80    # win rate above this → boosted size (1.25×)
_BLOCK_COOLDOWN_S = 4 * 3600  # soft-blocked symbol re-probes after 4 h
_PROBE_TRADES     = 2       # number of cautious probe trades on re-entry
_PROBE_SIZE_MULT  = 0.40    # size multiplier during probing
# Consecutive-loss circuit breaker: bench a symbol after this many straight losses,
# INDEPENDENT of the win-rate window (which needs _MIN_TRADES first). Catches a fresh
# losing streak fast — e.g. LUNC's 3 overnight stop-outs would bench at the 2nd.
_LOSS_STREAK_BLOCK = int(os.getenv("SYMPERF_LOSS_STREAK_BLOCK", "2") or 2)


@dataclass
class SymbolRecord:
    outcomes: deque = field(default_factory=lambda: deque(maxlen=_WINDOW))
    # outcome = (ts: float, pnl: float)
    blocked_until: float = 0.0
    probe_remaining: int = 0
    loss_streak: int = 0  # consecutive losses; resets on any win


class SymbolPerformanceTracker:
    """
    Dynamic symbol scoring. Call record_trade() after every fill,
    get_size_multiplier() before allocating capital, is_tradeable() in gate_8.
    """

    def __init__(self) -> None:
        self._records: dict[str, SymbolRecord] = {}
        self._load_state()

    def _load_state(self) -> None:
        try:
            if not os.path.exists(_STATE_PATH):
                return
            data = json.loads(open(_STATE_PATH).read())
            now = time.time()
            for sym, rec_data in data.items():
                rec = self._record(sym)
                for ts, pnl in rec_data.get("outcomes", []):
                    rec.outcomes.append((float(ts), float(pnl)))
                rec.blocked_until = float(rec_data.get("blocked_until", 0))
                rec.probe_remaining = int(rec_data.get("probe_remaining", 0))
                rec.loss_streak = int(rec_data.get("loss_streak", 0))
            _log.info("[SymPerf] Restored state for %d symbols", len(self._records))
        except Exception as e:
            _log.debug("[SymPerf] _load_state failed: %s", e)

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(_STATE_PATH) or "logs", exist_ok=True)
            payload = {}
            for sym, rec in self._records.items():
                payload[sym] = {
                    "outcomes": list(rec.outcomes),
                    "blocked_until": rec.blocked_until,
                    "probe_remaining": rec.probe_remaining,
                    "loss_streak": rec.loss_streak,
                }
            tmp = _STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, _STATE_PATH)
        except Exception as e:
            _log.debug("[SymPerf] _save_state failed: %s", e)

    def _record(self, symbol: str) -> SymbolRecord:
        if symbol not in self._records:
            self._records[symbol] = SymbolRecord()
        return self._records[symbol]

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(self, symbol: str, pnl: float) -> None:
        """Call after every confirmed closed trade (SELL fill with known PnL)."""
        rec = self._record(symbol)
        rec.outcomes.append((time.time(), pnl))

        # Consecutive-loss circuit breaker — fires BEFORE the win-rate window so a
        # fresh losing streak gets benched fast (win-rate gating needs _MIN_TRADES).
        if pnl > 0:
            rec.loss_streak = 0
        else:
            rec.loss_streak += 1
            if rec.loss_streak >= _LOSS_STREAK_BLOCK:
                rec.blocked_until = time.time() + _BLOCK_COOLDOWN_S
                rec.probe_remaining = 0
                _log.warning(
                    "[SymPerf] %s LOSS-STREAK BLOCK: %d consecutive losses — "
                    "re-probing in %.0fh",
                    symbol, rec.loss_streak, _BLOCK_COOLDOWN_S / 3600,
                )
                self._save_state()
                return

        n = len(rec.outcomes)
        if n < _MIN_TRADES:
            self._save_state()
            return

        wr = self._win_rate(rec)

        if wr < _BLOCK_WIN_RATE:
            rec.blocked_until = time.time() + _BLOCK_COOLDOWN_S
            rec.probe_remaining = 0
            _log.warning(
                "[SymPerf] %s SOFT-BLOCKED: win_rate=%.0f%% over last %d trades — "
                "re-probing in %.0fh",
                symbol, wr * 100, n, _BLOCK_COOLDOWN_S / 3600,
            )
        elif rec.probe_remaining > 0:
            if pnl > 0:
                rec.probe_remaining -= 1
                if rec.probe_remaining == 0:
                    _log.info("[SymPerf] %s probe complete — fully re-qualified", symbol)
            else:
                # Probe failed — extend block
                rec.blocked_until = time.time() + _BLOCK_COOLDOWN_S
                rec.probe_remaining = 0
                _log.warning("[SymPerf] %s probe FAILED — re-blocking for %.0fh", symbol, _BLOCK_COOLDOWN_S / 3600)
        self._save_state()

    def is_tradeable(self, symbol: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason). Called by gate_8.
        New symbols (< MIN_TRADES) always allowed at probe size.
        """
        rec = self._record(symbol)
        now = time.time()
        n = len(rec.outcomes)

        # Active block takes precedence over the new-symbol allowance — otherwise a
        # loss-streak block on a symbol with < _MIN_TRADES would be silently ignored.
        if rec.blocked_until > now:
            # Still inside the cooldown window — hard block, no trading allowed.
            remaining_h = (rec.blocked_until - now) / 3600
            return False, f"blocked({remaining_h:.1f}h_remaining)"

        if n < _MIN_TRADES:
            return True, f"new_symbol({n}/{_MIN_TRADES}_trades)"

        # Cooldown has expired — check win rate before allowing probe trades.
        # Symbols with critically low WR skip probing and get another full cooldown.
        if rec.probe_remaining == 0:
            wr_check = self._win_rate(rec)
            if wr_check < _NO_PROBE_WIN_RATE:
                rec.blocked_until = now + _BLOCK_COOLDOWN_S
                _log.warning(
                    "[SymPerf] %s cooldown expired but wr=%.0f%% < %.0f%% floor — "
                    "skipping probe, re-blocking for %.0fh",
                    symbol, wr_check * 100, _NO_PROBE_WIN_RATE * 100, _BLOCK_COOLDOWN_S / 3600,
                )
                return False, f"blocked(wr={wr_check:.0%}_below_probe_floor)"
            rec.probe_remaining = _PROBE_TRADES
            _log.info("[SymPerf] %s cooldown expired (wr=%.0f%%) — starting %d probe trades", symbol, wr_check * 100, _PROBE_TRADES)
        if rec.probe_remaining > 0:
            return True, f"probing({rec.probe_remaining}_left)"

        wr = self._win_rate(rec)
        if wr < _BLOCK_WIN_RATE:
            # Win rate still too low after probing — re-block for another cooldown.
            rec.blocked_until = now + _BLOCK_COOLDOWN_S
            rec.probe_remaining = 0
            return False, f"blocked(wr={wr:.0%}_<_{_BLOCK_WIN_RATE:.0%})"

        return True, f"ok(wr={wr:.0%})"

    def quality_score(self, symbol: str) -> float:
        """0..1 performance quality from the rolling win-rate.

        Used as `agent_quality` in the probability score so symbols with a proven
        track record score higher and chronic losers score lower. Returns a neutral
        0.5 until the symbol has enough closed trades to judge (avoids penalising or
        rewarding on noise).
        """
        rec = self._records.get(symbol)
        if rec is None or len(rec.outcomes) < _MIN_TRADES:
            return 0.5
        return float(self._win_rate(rec))

    def get_size_multiplier(self, symbol: str) -> float:
        """
        Returns a size multiplier [0.0, 1.25].
        - New symbol              → 0.75  (cautious probe)
        - Probing after block     → 0.40
        - Win rate < WARN         → 0.50
        - Win rate WARN–GOOD      → 0.75
        - Win rate GOOD–GREAT     → 1.00
        - Win rate ≥ GREAT        → 1.25  (edge confirmed, run it)
        """
        rec = self._record(symbol)
        n = len(rec.outcomes)

        if n < _MIN_TRADES:
            return 0.75  # new — allow but size down until we have data

        if rec.probe_remaining > 0:
            return _PROBE_SIZE_MULT

        wr = self._win_rate(rec)

        if wr < _BLOCK_WIN_RATE:
            return 0.0
        if wr < _WARN_WIN_RATE:
            return 0.50
        if wr < _GOOD_WIN_RATE:
            return 0.75
        if wr < _GREAT_WIN_RATE:
            return 1.00
        return 1.25

    def summary(self) -> dict[str, Any]:
        now = time.time()
        out = {}
        for sym, rec in self._records.items():
            n = len(rec.outcomes)
            out[sym] = {
                "trades": n,
                "win_rate": round(self._win_rate(rec), 3) if n >= _MIN_TRADES else None,
                "total_pnl": round(sum(p for _, p in rec.outcomes), 4),
                "size_mult": self.get_size_multiplier(sym),
                "blocked_until": rec.blocked_until if rec.blocked_until > now else None,
                "probe_remaining": rec.probe_remaining,
            }
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _win_rate(rec: SymbolRecord) -> float:
        if not rec.outcomes:
            return 0.0
        wins = sum(1 for _, p in rec.outcomes if p > 0)
        return wins / len(rec.outcomes)
