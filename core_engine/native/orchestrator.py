"""
Native L8: Orchestrator (Phase 8.2.9)

Central coordinator: composes L0-L7 into the 5-phase cycle. Replaces ~1200 LOC
legacy ``MASTER_SYSTEM_ORCHESTRATOR.py`` with focused ~300-line native impl.

Design choices
--------------
* 5-phase cycle: READ (prices) -> UNDERSTAND (signals) -> DECIDE (positions)
  -> EXECUTE (orders) -> RECOVER (health).
* Stateless per-cycle. All state lives in injected dependencies (L0-L5).
* Telemetry: record cycle metrics (duration, nav, signals, executions, errors).
* Configurable cycle loop: duration or cycle count.
* Graceful shutdown on signal or exception.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Per-cycle performance telemetry."""

    cycle_num: int
    duration_ms: float
    nav: float
    signals_count: int
    decisions_count: int
    executions_count: int
    execution_successes: int
    execution_failures: int
    phase_times: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class NativeOrchestrator:
    """
    Trading cycle orchestrator. Implements the 5-phase cycle:
    1. READ - fetch market data
    2. UNDERSTAND - generate signals
    3. DECIDE - size positions
    4. EXECUTE - place orders
    5. RECOVER - health check

    Usage::

        orch = NativeOrchestrator(
            market_data=md,
            signal_engine=sig,
            decision_engine=dec,
            executor=exe,
            ...
        )
        metrics = await orch.run_cycle()  # single cycle
        # or
        await orch.run_loop(duration_sec=3600.0)  # run for 1 hour
    """

    def __init__(
        self,
        *,
        market_data: Any,  # NativeMarketData
        signal_engine: Any,  # NativeSignalEngine
        decision_engine: Any,  # NativeDecisionEngine
        executor: Any,  # NativeExecutor
        balance_sync: Any
        | None = None,  # NativeBalanceSync (optional when polling_coordinator is used)
        shared_state: Any | None = None,  # NativeSharedState
        portfolio_accessor: callable | None = None,
        telemetry: Any | None = None,  # NativeTelemetry (L6, optional)
        watchdog: Any | None = None,  # NativeWatchdog (L7, optional)
        fill_tracker: Any | None = None,  # NativeFillTracker (L3, optional)
        objective_feedback_controller: Any
        | None = None,  # ObjectiveFeedbackController (OFC, optional)
        symbol_discovery: Any | None = None,  # NativeSymbolDiscovery (optional)
        market_data_ws: Any
        | None = None,  # NativeMarketDataWebSocket (optional, zero API rate limits)
        polling_coordinator: Any
        | None = None,  # NativePollingCoordinator (optional, legacy-style staggered polling)
    ) -> None:
        self._market_data = market_data
        self._signal_engine = signal_engine
        self._decision_engine = decision_engine
        self._executor = executor
        self._balance_sync = balance_sync
        self._shared_state = shared_state
        self._portfolio_accessor = portfolio_accessor
        self._telemetry = telemetry
        self._watchdog = watchdog
        self._fill_tracker = fill_tracker
        self._ofc = objective_feedback_controller
        self._symbol_discovery = symbol_discovery
        self._market_data_ws = market_data_ws
        self._polling_coordinator = polling_coordinator

        self._cycle_count = 0
        self._stopped = True  # Use bool flag instead of asyncio.Event

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────
    def _get_balance(self) -> dict[str, float]:
        """
        Get current account balance from either balance_sync or shared_state.

        When polling_coordinator is enabled, balance is synced to shared_state.
        When polling_coordinator is disabled, use balance_sync directly.
        """
        if self._balance_sync is not None and hasattr(self._balance_sync, "get_balance"):
            return self._balance_sync.get_balance()
        elif self._shared_state is not None and hasattr(self._shared_state, "balance"):
            return dict(self._shared_state.balance)  # Copied dict for consistency
        return {}

    # ──────────────────────────────────────────────────────────────────
    # Loop control
    # ──────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Prepare orchestrator (e.g., start background tasks)."""
        self._stopped = False

        # Initialize session start time for OFC elapsed calculation
        if self._shared_state:
            self._shared_state._session_start_ts = time.time()
            # session_anchor_nav will be set on first _phase_read when balance is available

        await self._market_data.start()
        if self._market_data_ws is not None:
            await self._market_data_ws.start()

        # Start either polling coordinator (new, efficient) or balance_sync (legacy, aggressive)
        if self._polling_coordinator is not None:
            await self._polling_coordinator.start()
        elif self._balance_sync is not None:
            await self._balance_sync.start()

        # Only start fill_tracker if we're NOT using polling coordinator
        if self._fill_tracker is not None and self._polling_coordinator is None:
            await self._fill_tracker.start()

        if self._ofc is not None:
            await self._ofc.start()

        # Wait for initial data to be available (up to 5 seconds)
        # Ensures market_data and balance_sync have fetched at least once
        await self._wait_for_initial_data(max_wait_sec=5.0)

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._stopped = True
        if self._ofc is not None:
            await self._ofc.stop()

        # Stop polling coordinator or balance_sync (one or the other)
        if self._polling_coordinator is not None:
            await self._polling_coordinator.stop()
        elif self._balance_sync is not None:
            await self._balance_sync.stop()

        # Only stop fill_tracker if not using polling coordinator
        if self._fill_tracker is not None and self._polling_coordinator is None:
            await self._fill_tracker.stop()

        if self._market_data_ws is not None:
            await self._market_data_ws.stop()
        await self._market_data.stop()

    async def run_loop(
        self,
        duration_sec: float | None = None,
        max_cycles: int | None = None,
    ) -> list[CycleMetrics]:
        """
        Run trading cycles until duration exhausted or max_cycles reached.

        Returns all metrics collected.
        """
        await self.start()
        metrics: list[CycleMetrics] = []
        start_time = time.time()
        try:
            while not self._stopped:
                if duration_sec is not None and (time.time() - start_time) >= duration_sec:
                    break
                if max_cycles is not None and self._cycle_count >= max_cycles:
                    break

                m = await self.run_cycle()
                metrics.append(m)
                await asyncio.sleep(0.01)  # yield to event loop
        finally:
            await self.stop()
        return metrics

    # ──────────────────────────────────────────────────────────────────
    # Single cycle
    # ──────────────────────────────────────────────────────────────────
    async def run_cycle(self) -> CycleMetrics:
        """Execute a single 5-phase trading cycle. Returns metrics."""
        self._cycle_count += 1
        cycle_start = time.time()
        metrics = CycleMetrics(
            cycle_num=self._cycle_count,
            nav=0.0,
            signals_count=0,
            duration_ms=0.0,
            decisions_count=0,
            executions_count=0,
            execution_successes=0,
            execution_failures=0,
        )

        try:
            # Phase 0: DISCOVER (optional symbol discovery per cycle)
            if self._symbol_discovery:
                logger.debug("📍 CYCLE %d: Phase 0 DISCOVER starting", self._cycle_count)
                await self._phase_discover()
            else:
                logger.debug(
                    "📍 CYCLE %d: Phase 0 DISCOVER skipped (no symbol_discovery)", self._cycle_count
                )

            # Phase 1: READ
            t0 = time.time()
            await self._phase_read()
            metrics.phase_times["read"] = (time.time() - t0) * 1000.0

            # Phase 2: UNDERSTAND
            t0 = time.time()
            signals = await self._phase_understand()
            metrics.phase_times["understand"] = (time.time() - t0) * 1000.0
            metrics.signals_count = len(signals)

            # Phase 3: DECIDE
            t0 = time.time()
            decisions = await self._phase_decide(signals)
            metrics.phase_times["decide"] = (time.time() - t0) * 1000.0
            metrics.decisions_count = len(decisions)

            # Phase 4: EXECUTE
            t0 = time.time()
            executions = await self._phase_execute(decisions)
            metrics.phase_times["execute"] = (time.time() - t0) * 1000.0
            metrics.executions_count = len(executions)
            metrics.execution_successes = sum(1 for e in executions if e.status.value == "SUCCESS")
            metrics.execution_failures = len(executions) - metrics.execution_successes

            # Phase 5: RECOVER
            t0 = time.time()
            await self._phase_recover()
            metrics.phase_times["recover"] = (time.time() - t0) * 1000.0

            # Metrics
            # NativeSharedState exposes ``nav_usdt`` as the canonical NAV
            # field; fall back to ``nav`` for duck-typed test stubs.
            metrics.nav = getattr(
                self._shared_state, "nav_usdt", getattr(self._shared_state, "nav", 0.0)
            )
            metrics.duration_ms = (time.time() - cycle_start) * 1000.0

        except Exception as e:  # pragma: no cover - defensive
            logger.exception("cycle %05d failed: %s", self._cycle_count, e)
            metrics.errors.append(f"{type(e).__name__}: {e}")
            metrics.duration_ms = (time.time() - cycle_start) * 1000.0

        # L6: telemetry hook (optional, never raises)
        if self._telemetry is not None:
            try:
                self._telemetry.record(metrics)
            except Exception:  # pragma: no cover - defensive
                logger.exception("telemetry.record failed (cycle %05d)", self._cycle_count)

        # L7: watchdog heartbeat (optional, never raises). Records
        # ok=True iff the cycle completed without errors. Done after
        # telemetry so a watchdog crash can't poison metrics.
        if self._watchdog is not None:
            try:
                self._watchdog.record_heartbeat(ok=not metrics.errors)
            except Exception:  # pragma: no cover - defensive
                logger.exception("watchdog.record_heartbeat failed (cycle %05d)", self._cycle_count)

        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 6-phase implementation (Phase 0 optional: symbol discovery)
    # ──────────────────────────────────────────────────────────────────
    async def _phase_discover(self) -> None:
        """Phase 0: Scan wallet and update symbol list (optional, per-cycle)."""
        if not self._symbol_discovery:
            return
        try:
            symbols = await self._symbol_discovery.discover()
            if symbols and self._market_data:
                # Update market data symbols if changed
                current_symbols = self._market_data._symbols
                if sorted(symbols) != sorted(current_symbols or []):
                    logger.info("📱 Symbols discovered: %s → %s", current_symbols, symbols)
                    self._market_data._symbols = symbols
                    # Subscribe to new symbols in WebSocket if available
                    if self._market_data_ws:
                        await self._market_data_ws.subscribe(symbols)
            elif not symbols:
                logger.warning("⚠️ Symbol discovery returned empty list (wallet empty?)")
        except Exception as e:
            logger.exception("Symbol discovery failed: %s (will retry next cycle)", e)

    async def _phase_read(self) -> None:
        """Phase 1: Fetch latest market data. Sync balances."""
        # Market data is background-polled by NativeMarketData.
        # Balance sync is background-polled by NativeBalanceSync.
        # This phase is a no-op in the async model; data is always current.

        # Initialize session_anchor_nav on first cycle with real balance
        if self._shared_state:
            balance_usdt = self._get_balance().get("USDT", 0.0)
            if balance_usdt > 0:
                # Set NAV (used by capital allocator, OFC, ACE)
                self._shared_state.update_nav(balance_usdt)
                # Set anchor on first cycle (OFC needs this for pace calculation)
                if self._shared_state.session_anchor_nav <= 0:
                    self._shared_state.session_anchor_nav = balance_usdt
                    logger.info("📊 Session anchor NAV set: %.2f USDT", balance_usdt)

    async def _phase_understand(self) -> dict[str, Any]:
        """Phase 2: Generate signals from market data."""
        # Fetch current prices from L2
        prices = self._market_data.get_prices()
        if not prices:
            logger.debug("⚠️ No prices available from market_data.get_prices()")
            return {}

        logger.debug(f"🔍 Phase 2: Evaluating {len(prices)} symbols for signals")

        # For each symbol, fetch klines and evaluate signals
        signals_by_symbol: dict[str, Any] = {}
        for symbol in prices:
            try:
                klines = await self._market_data.get_klines(symbol, interval="1m", limit=100)
                agg_sig = self._signal_engine.evaluate(symbol, klines)
                if agg_sig is not None:
                    signals_by_symbol[symbol] = {
                        "direction": agg_sig.direction,
                        "score": agg_sig.score,
                        "contributions": agg_sig.contributions,
                    }
            except Exception as e:  # pragma: no cover
                logger.debug(f"signal generation failed for {symbol}: {e}")

        return signals_by_symbol

    async def _phase_decide(self, signals: dict[str, Any]) -> list[Any]:
        """Phase 3: Generate trading decisions."""
        # Build portfolio snapshot
        portfolio = self._portfolio_accessor() if self._portfolio_accessor else None
        if portfolio is None:
            logger.warning("portfolio snapshot unavailable; returning empty decisions")
            return []

        balance_usdt = self._get_balance().get("USDT", 0.0)

        # Debug: log signal details
        if signals:
            buy_sigs = [s for s, sig in signals.items() if sig.get("direction") == "BUY"]
            sell_sigs = [s for s, sig in signals.items() if sig.get("direction") == "SELL"]
            logger.info(
                "🔍 _phase_decide: %d total signals → %d BUY, %d SELL (balance=%.2f)",
                len(signals),
                len(buy_sigs),
                len(sell_sigs),
                balance_usdt,
            )

            # Gate: check if trading is halted by ObjectiveFeedbackController
            if getattr(self._shared_state, "trading_halted", False):
                logger.warning(
                    "trading_halted=True (OFC kill-switch); skipping BUY decisions " "this cycle"
                )
                return []

        decisions = self._decision_engine.decide(signals, portfolio, balance_usdt)
        return decisions

    async def _phase_execute(self, decisions: list[Any]) -> list[Any]:
        """Phase 4: Execute trading decisions."""
        results = await self._executor.execute(decisions)
        return results

    async def _phase_recover(self) -> None:
        """Phase 5: Health check and recovery."""
        # Update metrics for ObjectiveFeedbackController
        if self._shared_state:
            nav = getattr(self._shared_state, "nav_usdt", getattr(self._shared_state, "nav", 0.0))
            m = getattr(self._shared_state, "metrics", {})

            # Unrealized P&L: portfolio value - invested cost basis
            if hasattr(self._shared_state, "get_all_positions"):
                positions = self._shared_state.get_all_positions()
                total_position_value = sum(p.qty * p.mark_price for p in positions.values())
                total_entry_cost = sum(p.qty * p.entry_price for p in positions.values())
                m["unrealized_pnl"] = (
                    (total_position_value - total_entry_cost) if positions else 0.0
                )

            # Peak NAV for drawdown calculation
            m["peak_nav"] = max(m.get("peak_nav", 0.0), nav)

            # Session elapsed time
            elapsed_s = time.time() - getattr(self._shared_state, "_session_start_ts", time.time())
            m["session_elapsed_h"] = elapsed_s / 3600.0

    # ──────────────────────────────────────────────────────────────────
    # Initialization helpers
    # ──────────────────────────────────────────────────────────────────
    async def _wait_for_initial_data(self, max_wait_sec: float = 5.0) -> None:
        """
        Wait for market_data and balance_sync to fetch initial data.

        Polls both systems and returns once they have data, or after timeout.
        This ensures the first trading cycle doesn't see zero NAV/prices.
        """
        import asyncio as aio

        start = time.time()
        while (time.time() - start) < max_wait_sec:
            has_prices = False
            has_balance = False

            # Check market_data has prices
            if self._market_data and hasattr(self._market_data, "get_prices"):
                prices = self._market_data.get_prices()
                has_prices = len(prices) > 0 if prices else False

            # Check balance_sync or polling_coordinator has balance
            balance = self._get_balance()
            has_balance = bool(balance and balance.get("USDT", 0) > 0)

            if has_prices and has_balance:
                logger.info(
                    f"✅ Initial data ready (prices={len(prices)} symbols, balance=%.2f USDT)",
                    balance.get("USDT", 0.0),
                )
                return

            # Wait a bit and retry
            await aio.sleep(0.1)

        logger.warning(
            f"⚠️  Timeout waiting for initial data (waited {max_wait_sec}s). "
            f"Trading may start with zero NAV/prices."
        )
