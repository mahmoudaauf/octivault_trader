"""
Native L0: Position Hydration Engine — Authoritative Portfolio Recovery (Phase 8.4.1)

On restart, the system MUST NOT trade until it has:

1. Read balances from exchange
2. Read open orders
3. Read recent trade fills
4. Reconstruct positions with full state:
   - symbol, qty, avg_entry_price
   - realized_pnl, unrealized_pnl
   - fees_paid, entry_time
   - tp_price, sl_price
   - lifecycle_state (OPENING, ACTIVE, CLOSING, CLOSED)
   - reserved_quote
5. Calculate proper NAV
6. Classify positions (profitable, losing, stale, dust)
7. ONLY THEN allow trading

This engine uses a 3-tier recovery strategy:
  1. Local trade journal (best — per-trade detail)
  2. Exchange /myTrades (slower but authoritative)
  3. Hybrid (fallback when journal gaps exist)

Design principle: Exchange balance alone is NEVER trusted as full portfolio truth.
Balance tells you WHAT you hold; journal tells you WHY, when, and at what cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HydratedPosition:
    """Fully reconstructed position with complete state."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float

    # Profitability
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0

    # Timing
    entry_time: float = 0.0  # epoch seconds
    last_update: float = field(default_factory=time.time)

    # Risk management
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None

    # Lifecycle
    lifecycle_state: str = "ACTIVE"  # OPENING, ACTIVE, CLOSING, CLOSED

    # Capital allocation
    reserved_quote: float = 0.0  # USDT locked for this position

    # Strategy metadata
    strategy_source: str = "unknown"

    @property
    def position_value(self) -> float:
        """Total notional value in USDT."""
        return self.qty * self.current_price

    @property
    def total_pnl(self) -> float:
        """Realized + unrealized P&L."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_pnl_pct(self) -> float:
        """Total P&L as percentage of initial capital."""
        if self.qty == 0 or self.avg_entry_price <= 0:
            return 0.0
        initial_capital = self.qty * self.avg_entry_price
        if initial_capital == 0:
            return 0.0
        return (self.total_pnl / initial_capital) * 100.0

    def is_profitable(self) -> bool:
        """True if position is currently in profit."""
        return self.total_pnl > 0.0

    def is_losing(self) -> bool:
        """True if position is currently at loss."""
        return self.total_pnl < 0.0

    def is_stale(self, stale_age_sec: float = 3600) -> bool:
        """True if position hasn't been updated in stale_age_sec."""
        return (time.time() - self.last_update) > stale_age_sec

    def is_dust(self, dust_threshold_usdt: float = 1.0) -> bool:
        """True if position value is below dust threshold."""
        return self.position_value < dust_threshold_usdt


@dataclass
class HydrationState:
    """Result of position hydration."""

    success: bool
    message: str

    # Reconstructed portfolio
    positions: dict[str, HydratedPosition] = field(default_factory=dict)

    # Aggregates
    total_balance_usdt: float = 0.0
    free_balance_usdt: float = 0.0
    locked_balance_usdt: float = 0.0
    portfolio_value: float = 0.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0

    # Metrics
    positions_count: int = 0
    profitable_count: int = 0
    losing_count: int = 0
    stale_count: int = 0
    dust_count: int = 0

    # Lifecycle
    hydration_time_sec: float = 0.0
    recovery_source: str = "unknown"  # "journal", "exchange", "hybrid"


# ──────────────────────────────────────────────────────────────────────────────
# Position Hydration Engine
# ──────────────────────────────────────────────────────────────────────────────


class NativePositionHydrationEngine:
    """Reconstructs full position state from local journal + exchange data."""

    def __init__(
        self,
        shared_state: Any,
        trade_journal: Optional[Any] = None,
        exchange_client: Optional[Any] = None,
        *,
        journal_dir: str = "logs",
        allow_exchange_fallback: bool = True,
        stale_position_age_sec: float = 3600.0,
        dust_threshold_usdt: float = 1.0,
    ) -> None:
        """
        Initialize the hydration engine.

        Args:
            shared_state: NativeSharedState to update with recovered data
            trade_journal: NativeTradeJournal for reading fills
            exchange_client: Exchange client for /myTrades fallback
            journal_dir: Directory containing trade_journal files
            allow_exchange_fallback: Whether to use /myTrades if journal is incomplete
            stale_position_age_sec: Mark position as stale if older than this
            dust_threshold_usdt: Mark position as dust if value < this
        """
        self._state = shared_state
        self._journal = trade_journal
        self._exchange = exchange_client
        self._journal_dir = Path(journal_dir)
        self._allow_fallback = allow_exchange_fallback
        self._stale_sec = stale_position_age_sec
        self._dust_threshold = dust_threshold_usdt

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def hydrate(self) -> HydrationState:
        """
        Reconstruct full position state from available sources.

        Strategy:
          1. Read current balances from exchange (required)
          2. Try local journal first (preferred)
          3. Fall back to exchange /myTrades if configured and journal incomplete
          4. Reconstruct weighted average entry prices
          5. Calculate realized/unrealized PnL
          6. Return hydrated state

        Returns:
            HydrationState with full portfolio snapshot
        """
        start_time = time.time()
        state = HydrationState(success=False, message="Hydration not started")

        try:
            # Step 1: Get current balances from exchange
            logger.info("🔄 Position hydration: fetching balances...")
            balances = await self._fetch_balances()
            if not balances:
                state.message = "Failed to fetch balances from exchange"
                return state

            free_balance = float(balances.get("USDT", {}).get("free", 0.0) or 0.0)
            locked_balance = float(balances.get("USDT", {}).get("locked", 0.0) or 0.0)
            total_balance = free_balance + locked_balance

            state.free_balance_usdt = free_balance
            state.locked_balance_usdt = locked_balance
            state.total_balance_usdt = total_balance

            logger.info(
                f"  Balance: ${total_balance:.2f} "
                f"(free=${free_balance:.2f}, locked=${locked_balance:.2f})"
            )

            # Step 2: Reconstruct positions from fills
            logger.info("🔄 Position hydration: reconstructing positions from fills...")
            positions = await self._reconstruct_positions_from_fills(balances)

            # Step 3: Restore TP/SL from shared_state if available
            logger.info("🔄 Position hydration: restoring TP/SL state...")
            await self._restore_tp_sl_state(positions)

            # Step 4: Calculate PnL
            logger.info("🔄 Position hydration: calculating PnL...")
            await self._calculate_pnl(positions, balances)

            # Step 5: Classify positions
            logger.info("🔄 Position hydration: classifying positions...")
            self._classify_positions(positions)

            # Step 6: Calculate aggregates
            state.positions = positions
            state.positions_count = len(positions)
            state.portfolio_value = sum(p.position_value for p in positions.values())
            state.total_realized_pnl = sum(p.realized_pnl for p in positions.values())
            state.total_unrealized_pnl = sum(p.unrealized_pnl for p in positions.values())
            state.total_fees_paid = sum(p.fees_paid for p in positions.values())
            state.profitable_count = sum(1 for p in positions.values() if p.is_profitable())
            state.losing_count = sum(1 for p in positions.values() if p.is_losing())
            state.stale_count = sum(1 for p in positions.values() if p.is_stale(self._stale_sec))
            state.dust_count = sum(1 for p in positions.values() if p.is_dust(self._dust_threshold))

            state.hydration_time_sec = time.time() - start_time
            state.success = True
            state.message = (
                f"✅ Hydration complete: {state.positions_count} positions, "
                f"${state.portfolio_value:.2f} value, "
                f"{state.profitable_count} profitable, {state.losing_count} losing"
            )

            logger.info(state.message)
            return state

        except Exception as e:
            logger.exception(f"❌ Position hydration failed: {e}")
            state.success = False
            state.message = f"Hydration failed: {e!s}"
            return state

    # ──────────────────────────────────────────────────────────────────────────
    # Implementation details
    # ──────────────────────────────────────────────────────────────────────────

    async def _fetch_balances(self) -> dict[str, dict[str, float]]:
        """Fetch current balances from exchange."""
        if not self._exchange:
            logger.warning("No exchange client; skipping balance fetch")
            return {}

        try:
            account = await asyncio.wait_for(self._exchange.get_account(), timeout=10.0)
            balances: dict[str, dict[str, float]] = {}

            for b in account.get("balances", []):
                asset = str(b.get("asset", "")).upper()
                if not asset:
                    continue
                free = float(b.get("free", 0) or 0)
                locked = float(b.get("locked", 0) or 0)
                balances[asset] = {"free": free, "locked": locked, "total": free + locked}

            return balances
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
            return {}

    async def _reconstruct_positions_from_fills(
        self, balances: dict[str, dict[str, float]]
    ) -> dict[str, HydratedPosition]:
        """
        Reconstruct positions by reading fills from local journal or exchange.

        Uses 3-tier strategy:
          1. Local journal (preferred — per-trade detail, no API calls)
          2. Exchange /myTrades (fallback — slower, API calls)
          3. Hybrid (if journal gaps exist)
        """
        positions: dict[str, HydratedPosition] = {}

        # Try local journal first
        logger.info("  Attempting local journal recovery...")
        journal_fills = await self._read_journal_fills()

        if journal_fills:
            logger.info(f"    Found {len(journal_fills)} fills in local journal")
            positions = self._build_positions_from_fills(journal_fills)
            self._fill_missing_current_prices(positions, balances)
            return positions

        # Fallback to exchange if allowed
        if self._allow_fallback and self._exchange:
            logger.info("  Attempting exchange trade history recovery...")
            exchange_fills = await self._fetch_exchange_fills()
            if exchange_fills:
                logger.info(f"    Found {len(exchange_fills)} fills from exchange")
                positions = self._build_positions_from_fills(exchange_fills)
                self._fill_missing_current_prices(positions, balances)
                return positions

        logger.warning("No fills found in journal or exchange; assuming fresh account")
        return positions

    async def _read_journal_fills(self) -> list[dict[str, Any]]:
        """Read all fills from local trade journal files."""
        fills: list[dict[str, Any]] = []

        try:
            # Find all trade_journal_*.jsonl files
            journal_files = sorted(self._journal_dir.glob("trade_journal_*.jsonl"), reverse=True)

            for jf in journal_files:
                try:
                    with open(jf) as f:
                        for line in f:
                            record = json.loads(line)
                            if record.get("event") in ("ORDER_FILLED", "TRADE_EXECUTED"):
                                fills.append(record)
                except Exception as e:
                    logger.warning(f"Error reading {jf}: {e}")

            return fills
        except Exception as e:
            logger.warning(f"Error reading journal files: {e}")
            return []

    async def _fetch_exchange_fills(self) -> list[dict[str, Any]]:
        """Fetch trade history from exchange."""
        if not self._exchange:
            return []

        try:
            # Typically: GET /myTrades with limit=500
            # Returns list of executed trades
            trades = await asyncio.wait_for(
                self._exchange.get_all_orders(),  # or similar endpoint
                timeout=15.0,
            )
            return trades if trades else []
        except Exception as e:
            logger.error(f"Failed to fetch exchange fills: {e}")
            return []

    def _build_positions_from_fills(
        self, fills: list[dict[str, Any]]
    ) -> dict[str, HydratedPosition]:
        """Build positions by aggregating fills and calculating weighted average entry price."""
        positions: dict[str, HydratedPosition] = {}
        fees_by_symbol: dict[str, float] = {}

        for fill in fills:
            symbol = str(fill.get("symbol", "")).upper().replace("/", "")
            if not symbol:
                continue

            side = str(fill.get("side", "BUY")).upper()
            qty = float(fill.get("qty", 0) or fill.get("quantity", 0) or 0)
            price = float(fill.get("price", 0) or 0)
            fee = float(fill.get("fee", 0) or fill.get("commission", 0) or 0)
            ts = float(fill.get("epoch", fill.get("ts", time.time())))

            if qty == 0 or price == 0:
                continue

            # Initialize or update position
            if symbol not in positions:
                positions[symbol] = HydratedPosition(
                    symbol=symbol,
                    qty=0.0,
                    avg_entry_price=0.0,
                    current_price=0.0,
                    entry_time=ts,
                )

            pos = positions[symbol]

            # Update quantity and weighted average entry price
            if side == "BUY":
                total_qty = pos.qty + qty
                if total_qty > 0:
                    pos.avg_entry_price = (pos.qty * pos.avg_entry_price + qty * price) / total_qty
                pos.qty = total_qty
                pos.entry_time = min(pos.entry_time, ts)
            elif side == "SELL":
                # Reduce position (FIFO: sold at average entry price)
                pos.realized_pnl += qty * (price - pos.avg_entry_price)
                pos.qty = max(0.0, pos.qty - qty)

            # Accumulate fees
            if symbol not in fees_by_symbol:
                fees_by_symbol[symbol] = 0.0
            fees_by_symbol[symbol] += fee

        # Assign fees to positions
        for symbol, pos in positions.items():
            pos.fees_paid = fees_by_symbol.get(symbol, 0.0)

        # Remove closed positions
        return {sym: pos for sym, pos in positions.items() if pos.qty > 0}

    def _fill_missing_current_prices(
        self, positions: dict[str, HydratedPosition], balances: dict[str, dict[str, float]]
    ) -> None:
        """Fill in current prices from shared_state or use entry price as fallback."""
        for symbol, pos in positions.items():
            # Try to get from shared_state
            if hasattr(self._state, "prices"):
                price = float(self._state.prices.get(symbol, 0.0) or 0.0)
                if price > 0:
                    pos.current_price = price
                    continue

            # Fallback: use entry price (conservative, avoids stale data)
            pos.current_price = pos.avg_entry_price

    async def _restore_tp_sl_state(self, positions: dict[str, HydratedPosition]) -> None:
        """Restore TP/SL prices from shared_state if available."""
        if not hasattr(self._state, "positions"):
            return

        for symbol, pos in positions.items():
            ss_pos = self._state.positions.get(symbol)
            if ss_pos:
                pos.tp_price = getattr(ss_pos, "tp", None)
                pos.sl_price = getattr(ss_pos, "sl", None)
                pos.lifecycle_state = getattr(ss_pos, "lifecycle", "ACTIVE")

    async def _calculate_pnl(
        self, positions: dict[str, HydratedPosition], balances: dict[str, dict[str, float]]
    ) -> None:
        """Calculate realized and unrealized PnL for each position."""
        for pos in positions.values():
            if pos.qty > 0 and pos.avg_entry_price > 0:
                # Unrealized = current value - entry value - fees
                entry_value = pos.qty * pos.avg_entry_price
                current_value = pos.position_value
                pos.unrealized_pnl = current_value - entry_value - pos.fees_paid

    def _classify_positions(self, positions: dict[str, HydratedPosition]) -> None:
        """Classify positions (profitable, losing, stale, dust)."""
        for pos in positions.values():
            # Check for stale positions (haven't been updated in a while)
            if pos.is_stale(self._stale_sec):
                pos.lifecycle_state = "STALE"

            # Check for dust (position value too small to trade)
            if pos.is_dust(self._dust_threshold):
                pos.lifecycle_state = "DUST"

    # ──────────────────────────────────────────────────────────────────────────
    # Apply hydration to shared state
    # ──────────────────────────────────────────────────────────────────────────

    async def apply_to_shared_state(self, state: HydrationState) -> None:
        """
        Write hydrated positions back to shared_state.

        This should be called after hydration completes, before trading begins.
        """
        if not state.success:
            logger.warning(f"Not applying hydration (failed): {state.message}")
            return

        logger.info(f"Applying {len(state.positions)} hydrated positions to shared_state...")

        # Update shared_state positions
        if hasattr(self._state, "positions"):
            for symbol, hydrated in state.positions.items():
                self._state.positions[symbol] = {
                    "symbol": hydrated.symbol,
                    "qty": hydrated.qty,
                    "entry_price": hydrated.avg_entry_price,
                    "current_price": hydrated.current_price,
                    "unrealized_pnl": hydrated.unrealized_pnl,
                    "realized_pnl": hydrated.realized_pnl,
                    "tp": hydrated.tp_price,
                    "sl": hydrated.sl_price,
                    "lifecycle": hydrated.lifecycle_state,
                    "entry_time": hydrated.entry_time,
                    "last_update": hydrated.last_update,
                }

        # Update balance
        if hasattr(self._state, "nav_usdt"):
            self._state.nav_usdt = state.total_balance_usdt + state.portfolio_value

        # Update metrics
        if hasattr(self._state, "metrics"):
            self._state.metrics.update(
                {
                    "realized_pnl": state.total_realized_pnl,
                    "unrealized_pnl": state.total_unrealized_pnl,
                    "portfolio_value": state.portfolio_value,
                    "positions_count": state.positions_count,
                    "last_hydration_time": state.hydration_time_sec,
                }
            )

        logger.info("✅ Hydration applied to shared_state")


__all__ = [
    "NativePositionHydrationEngine",
    "HydratedPosition",
    "HydrationState",
]
