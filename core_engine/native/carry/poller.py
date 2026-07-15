"""
Native carry poller (Phase 5 of the funding-carry native-wiring plan).

A new, independent asyncio loop rather than extending NativePollingCoordinator
(polling_coordinator.py), which holds exactly one exchange-client/shared-state
pair by construction and has spot-specific assumptions baked in (its
positions-loop comment explicitly says "no separate exchange positions
endpoint," which is true for spot but false for futures -- /fapi/v2/positionRisk
is real). Bolting a second client/state shape into an already-757-line file
multiplies its complexity for a small structural win that duplication avoids
more cheaply.

Two independent cadences (a deliberate improvement over the standalone
carry_paper_trader.py, which runs everything on one shared 30-min loop):
- Funding-rate poll / entry-close scan: matches the standalone script's
  30-min cadence -- not latency-sensitive, funding settles every 8h.
- Liquidation-buffer health check: a much tighter cadence (default 5 min) --
  a real-money safety check shouldn't wait 30 minutes to notice a position
  is near liquidation.

Runs genuinely in parallel with the existing spot NativePollingCoordinator,
touching disjoint state (CarrySharedState, not NativeSharedState).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .executor import CarryLegExecutor
from .gates import CarryGateEngine
from .state import CarrySharedState

logger = logging.getLogger(__name__)


class CarryPollingLoop:
    def __init__(
        self,
        *,
        futures_client: Any,
        carry_state: CarrySharedState,
        carry_gates: CarryGateEngine,
        carry_executor: CarryLegExecutor,
        universe: Optional[set[str]] = None,
        resolve_notional_usd: Optional[Callable[[], Awaitable[float]]] = None,
        default_notional_usd: float = 10.0,
        fee_round_trip_pct: float = 0.24,
        funding_poll_interval_sec: float = 1800.0,   # 30 min, matches carry_paper_trader.py's POLL_MIN
        liq_check_interval_sec: float = 300.0,       # 5 min -- tighter than the standalone script
    ) -> None:
        self._futures = futures_client
        self._carry_state = carry_state
        self._gates = carry_gates
        self._executor = carry_executor
        self._universe: set[str] = set(universe or [])
        self._resolve_notional_usd = resolve_notional_usd
        self._default_notional_usd = float(default_notional_usd)
        self._fee_rt = float(fee_round_trip_pct) / 100.0
        self._funding_interval = max(1.0, float(funding_poll_interval_sec))
        self._liq_interval = max(1.0, float(liq_check_interval_sec))

        self._running = False
        self._funding_task: Optional[asyncio.Task] = None
        self._liq_task: Optional[asyncio.Task] = None

    def set_universe(self, symbols: set[str]) -> None:
        self._universe = set(symbols)

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._funding_task = asyncio.create_task(self._funding_loop())
        self._liq_task = asyncio.create_task(self._liquidation_loop())

    async def stop(self) -> None:
        self._running = False
        for task in (self._funding_task, self._liq_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._funding_task = None
        self._liq_task = None

    # ──────────────────────────────────────────────────────────────────
    # Funding-rate fetch (bulk, mirrors carry_paper_trader.py's current_funding())
    # ──────────────────────────────────────────────────────────────────
    async def fetch_funding(self) -> dict[str, float]:
        try:
            data = await self._futures.futures_mark_price()
        except Exception as e:
            logger.warning("[carry-poller] funding fetch failed: %s", str(e)[:100])
            return {}
        rows = data if isinstance(data, list) else []
        out: dict[str, float] = {}
        for row in rows:
            sym = row.get("symbol")
            if sym not in self._universe:
                continue
            try:
                out[sym] = float(row.get("lastFundingRate", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return out

    async def _settled_funding(self, symbol: str, since_ts: float) -> float:
        try:
            rows = await self._futures.futures_funding_rate(
                symbol, start_time=int(since_ts * 1000), limit=1000
            )
            return sum(abs(float(r.get("fundingRate", 0.0) or 0.0)) for r in rows)
        except Exception as e:
            logger.warning("[carry-poller] settled-funding fetch failed for %s: %s", symbol, str(e)[:100])
            return 0.0

    async def _resolve_notional(self) -> float:
        if self._resolve_notional_usd is not None:
            try:
                return float(await self._resolve_notional_usd())
            except Exception as e:
                logger.warning("[carry-poller] notional resolver failed, using default: %s", str(e)[:100])
        return self._default_notional_usd

    async def _finalize_close(self, symbol: str, entry_ts: float, exit_funding: float, mode: str = "paper") -> None:
        held_h = (self._now() - entry_ts) / 3600.0
        accrued = await self._settled_funding(symbol, entry_ts)
        net_pct = (accrued - self._fee_rt) * 100.0
        self._carry_state.record_closed_trade(
            symbol,
            held_h=held_h,
            accrued_funding_pct=accrued * 100.0,
            net_pct=net_pct,
            exit_funding=exit_funding,
            mode=mode,
        )

    @staticmethod
    def _now() -> float:
        import time

        return time.time()

    # ──────────────────────────────────────────────────────────────────
    # One full funding-poll cycle: close-eligible positions, then scan for
    # new opens. Public (not prefixed with _) so it's directly testable and
    # directly callable by an orchestrator cycle if a native trading-cycle
    # driven cadence is ever preferred over this class's own asyncio loop.
    # ──────────────────────────────────────────────────────────────────
    async def run_funding_cycle(self) -> dict[str, int]:
        funding = await self.fetch_funding()
        opened = closed = 0

        for symbol in list(self._carry_state.open_symbols()):
            pos = self._carry_state.get_open_hedge(symbol)
            if pos is None:
                continue
            fr = funding.get(symbol, pos.entry_funding)
            decision = self._gates.evaluate_close(symbol, fr)
            if not decision.allowed:
                continue
            entry_ts = pos.entry_ts
            result = await self._executor.close_hedge(symbol)
            if result.success:
                await self._finalize_close(symbol, entry_ts, fr)
                closed += 1
            else:
                logger.warning("[carry-poller] close failed for %s: %s", symbol, result.reason)

        self._gates.check_drawdown_halt()

        for symbol, fr in funding.items():
            decision = self._gates.evaluate_open(symbol, fr)
            if not decision.allowed:
                continue
            notional = await self._resolve_notional()
            budget = self._gates.check_notional_budget(notional)
            if not budget.allowed:
                continue
            result = await self._executor.open_hedge(symbol, fr, notional)
            if result.success:
                opened += 1
            else:
                logger.warning("[carry-poller] open failed for %s: %s", symbol, result.reason)

        return {"opened": opened, "closed": closed}

    # ──────────────────────────────────────────────────────────────────
    # Liquidation-buffer health check (tighter cadence)
    # ──────────────────────────────────────────────────────────────────
    async def run_liquidation_check(self) -> int:
        """Returns the count of positions force-closed this cycle."""
        forced = 0
        for symbol in list(self._carry_state.open_symbols()):
            pos = self._carry_state.get_open_hedge(symbol)
            if pos is None:
                continue
            try:
                positions = await self._futures.futures_position_information(symbol)
            except Exception as e:
                logger.warning("[carry-poller] position-risk fetch failed for %s: %s", symbol, str(e)[:100])
                continue
            row = next((p for p in positions if p.get("symbol") == symbol), None)
            if row is None:
                continue
            try:
                amt = float(row.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            if amt == 0.0:
                continue
            mark = float(row.get("markPrice", 0.0) or 0.0)
            liq = float(row.get("liquidationPrice", 0.0) or 0.0)
            if not self._gates.is_near_liquidation(mark_price=mark, liquidation_price=liq):
                continue
            logger.warning(
                "[carry-poller] %s within liquidation buffer (mark=%.4f liq=%.4f) — force-closing",
                symbol, mark, liq,
            )
            entry_ts = pos.entry_ts
            result = await self._executor.close_hedge(symbol)
            if result.success:
                await self._finalize_close(symbol, entry_ts, pos.entry_funding)
                forced += 1
        return forced

    # ──────────────────────────────────────────────────────────────────
    # Background loops
    # ──────────────────────────────────────────────────────────────────
    async def _funding_loop(self) -> None:
        while self._running:
            try:
                await self.run_funding_cycle()
            except Exception:
                logger.exception("[carry-poller] funding cycle error")
            await asyncio.sleep(self._funding_interval)

    async def _liquidation_loop(self) -> None:
        while self._running:
            try:
                await self.run_liquidation_check()
            except Exception:
                logger.exception("[carry-poller] liquidation-check cycle error")
            await asyncio.sleep(self._liq_interval)


__all__ = ["CarryPollingLoop"]
