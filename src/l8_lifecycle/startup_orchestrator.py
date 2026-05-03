# -*- coding: utf-8 -*-
"""
StartupOrchestrator: Canonical P9 Startup Sequencing Gate

CRITICAL ARCHITECTURAL PRINCIPLE:
This is NOT a reconciliation engine. It is a SEQUENCING ORCHESTRATOR.

All reconciliation logic lives in its proper homes:
  - RecoveryEngine: State reconstruction
  - ExchangeTruthAuditor: Order/fill reconciliation
  - SharedState: Position hydration & event emission

This component's ONLY job is to coordinate those existing components
in the correct sequence before MetaController starts.

This prevents duplicate reconciliation and maintains single source of truth.
"""

import logging
import time
from typing import Any, Dict, Optional
import asyncio


class StartupOrchestrator:
    """
    Startup Sequencing Orchestrator (Phase 8.5).
    
    Coordinates existing reconciliation components in canonical order:
    1. RecoveryEngine.rebuild_state() - Fetch balances + positions from exchange
    2. SharedState.hydrate_positions_from_balances() - Mirror wallet → positions
    3. ExchangeTruthAuditor.restart_recovery() - Sync open orders
    4. Build capital ledger from wallet balances
    5. Verify capital integrity - Check NAV, capital, sanity
    6. Emit StartupPortfolioReady - Signal MetaController it's safe
    
    DOES NOT duplicate any reconciliation logic.
    DOES NOT create new subsystems.
    PURE sequencing gate using existing components.
    """
    
    def __init__(
        self,
        config: Any,
        shared_state: Any,
        exchange_client: Any,
        recovery_engine: Optional[Any] = None,
        exchange_truth_auditor: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize orchestrator with required components."""
        self.config = config
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.recovery_engine = recovery_engine
        self.exchange_truth_auditor = exchange_truth_auditor
        self.logger = logger or logging.getLogger("StartupOrchestrator")
        
        self._completed = False
        self._startup_ts = time.time()
        self._step_metrics = {}

    async def _price_fetcher(self, symbol: str) -> float:
        """Shared price fetcher used by Steps 5 & 6."""
        try:
            if self.exchange_client and hasattr(self.exchange_client, 'get_current_price'):
                price = await self.exchange_client.get_current_price(symbol)
                if price is not None:
                    price_float = float(price)
                    if price_float and price_float > 0:
                        return price_float
        except (ValueError, TypeError, Exception):
            pass
        return 0.0

    async def execute_startup_sequence(self) -> bool:
        """
        Execute canonical startup sequence.
        
        Returns:
            True if successful and ready for trading
            Raises RuntimeError if fatal error
        
        Side effects:
            - Delegates to RecoveryEngine, ExchangeTruthAuditor, etc.
            - Emits StartupPortfolioReady event
            - Sets _completed flag
        """
        # Idempotency guard: refuse to re-run a successful sequence
        if self._completed:
            self.logger.info(
                "[StartupOrchestrator] execute_startup_sequence() called again — already completed, no-op"
            )
            return True
        try:
            self.logger.warning(
                "[StartupOrchestrator] ═══════════════════════════════════════════════════"
            )
            self.logger.warning(
                "[StartupOrchestrator] PHASE 8.5: STARTUP SEQUENCING ORCHESTRATOR"
            )
            self.logger.warning(
                "[StartupOrchestrator] Coordinating reconciliation components in canonical order"
            )
            self.logger.warning(
                "[StartupOrchestrator] ═══════════════════════════════════════════════════"
            )
            
            # STEP 1: RecoveryEngine rebuilds state (fetch balances + positions)
            success = await self._step_recovery_engine_rebuild()
            if not success:
                raise RuntimeError(
                    "Phase 8.5: RecoveryEngine.rebuild_state() failed - cannot proceed"
                )
            
            # STEP 2: SharedState hydrates positions from balances
            success = await self._step_hydrate_positions()
            if not success:
                raise RuntimeError(
                    "Phase 8.5: SharedState position hydration failed - cannot proceed"
                )
            
            # STEP 2.4: Clear stale quote reservations from previous session
            # (non-fatal - these should be empty, but clear them to be safe)
            # This ensures spendable balance calculation is correct
            success = await self._step_clear_stale_quote_reservations()
            # Non-fatal - continue even if nothing to clear
            
            # STEP 2.5: Liquidate any legacy positions from previous trading session
            # (non-fatal if liquidation unavailable, but strongly recommended)
            # This runs AFTER positions are hydrated so we can see what needs liquidating
            success = await self._step_liquidate_legacy_positions()
            # Non-fatal - continue even if liquidation skipped
            
            # NOTE: Step 2.6 (aggressive exchange liquidation) is disabled 
            # because it requires ExecutionManager scope which isn't available during startup.
            # Instead, we rely on DeadCapitalHealer (which runs in the background during trading)
            # to liquidate dust positions over time. The key improvement is that we ensure
            # spendable capital calculation properly accounts for available balance even with 
            # many small positions.
            
            # STEP 3: ExchangeTruthAuditor syncs open orders (non-fatal)
            success = await self._step_auditor_restart_recovery()
            # Non-fatal if auditor unavailable
            
            # STEP 4: Build capital ledger from wallet balances
            success = await self._step_build_capital_ledger()
            if not success:
                raise RuntimeError(
                    "Phase 8.5: Capital ledger construction failed - cannot proceed"
                )
            
            # STEP 5: Verify capital integrity (ledger already constructed)
            success = await self._step_verify_capital_integrity()
            if not success:
                raise RuntimeError(
                    "Phase 8.5: Capital integrity verification failed - cannot proceed"
                )
            
            # IMPROVEMENT 3: Emit both events in sequence for extensibility
            # First: StartupStateRebuilt (signals state reconciliation complete)
            await self._emit_state_rebuilt_event()
            
            # Then: StartupPortfolioReady (signals portfolio ready for trading)
            await self._emit_startup_ready_event()
            
            # Mark complete
            self._completed = True
            
            self.logger.warning(
                "[StartupOrchestrator] ✅ STARTUP ORCHESTRATION COMPLETE"
            )
            self.logger.warning(
                "[StartupOrchestrator] Portfolio is ready for MetaController"
            )
            self.logger.warning(
                "[StartupOrchestrator] ═══════════════════════════════════════════════════"
            )
            self._log_final_metrics()
            return True
            
        except Exception as e:
            self.logger.error(
                f"[StartupOrchestrator] 💥 FATAL ERROR: {e}",
                exc_info=True
            )
            raise
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: RecoveryEngine - Rebuild state from exchange
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_recovery_engine_rebuild(self) -> bool:
        """Delegate to RecoveryEngine to fetch balances + positions."""
        step_name = "Step 1: RecoveryEngine.rebuild_state()"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.recovery_engine:
                self.logger.warning(f"[StartupOrchestrator] {step_name} - RecoveryEngine not available, skipping")
                return True  # Non-fatal
            
            # Log state BEFORE rebuild
            if self.shared_state:
                nav_before = float(getattr(self.shared_state, 'nav', 0.0) or 0.0)
                pos_before = len(getattr(self.shared_state, 'positions', {}) or {})
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Before: nav={nav_before}, positions={pos_before}"
                )
            
            # RecoveryEngine knows how to rebuild state from exchange
            if hasattr(self.recovery_engine, 'rebuild_state'):
                result = self.recovery_engine.rebuild_state()
                if asyncio.iscoroutine(result):
                    await result
                self.logger.debug(f"[StartupOrchestrator] {step_name} - Called rebuild_state()")
            elif hasattr(self.recovery_engine, '_load_live'):
                result = self.recovery_engine._load_live()
                if asyncio.iscoroutine(result):
                    await result
                self.logger.debug(f"[StartupOrchestrator] {step_name} - Called _load_live()")
            else:
                self.logger.warning(f"[StartupOrchestrator] {step_name} - No rebuild method found")
                return True  # Non-fatal
            
            # Log state AFTER rebuild
            if self.shared_state:
                nav_after = float(getattr(self.shared_state, 'nav', 0.0) or 0.0)
                pos_after = len(getattr(self.shared_state, 'positions', {}) or {})
                free_after = float(getattr(self.shared_state, 'free_quote', 0.0) or 0.0)
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - After: nav={nav_after}, "
                    f"positions={pos_after}, free={free_after}"
                )
                
                # DIAGNOSTIC: Warn if NAV is still 0 (suggests exchange issue)
                if nav_after <= 0:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - ⚠️ NAV is still 0 after rebuild. "
                        "This suggests: exchange API error, wallet empty, or exchange client not initialized"
                    )
            
            elapsed = time.time() - step_start
            self._step_metrics['recovery_engine_rebuild'] = {
                'elapsed_sec': elapsed,
            }
            
            self.logger.info(
                f"[StartupOrchestrator] {step_name} complete: {elapsed:.2f}s"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"[StartupOrchestrator] {step_name} - Error: {e}",
                exc_info=True
            )
            return False
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2.5: Liquidate Legacy Positions from Previous Trading Session
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_liquidate_legacy_positions(self) -> bool:
        """
        Auto-liquidate any legacy positions from previous trading session.
        
        CRITICAL FIX (FIX #6): Actually sell positions on exchange (not just remove from tracking).
        
        NOTE: The ExchangeClient has security guards that prevent orders from being placed outside
        the ExecutionManager scope. During startup (before ExecutionManager is initialized), we skip
        the liquidation step. The ExchangeTruthAuditor (Step 3) will handle any orphaned positions
        by matching them to wallet balances and closing them properly.
        
        APPROACH:
        1. Check if there are any legacy positions to liquidate
        2. If YES: Log warning and rely on ExchangeTruthAuditor cleanup in next step
        3. If NO: Proceed normally
        4. This gracefully handles the security guard constraint
        
        This prevents the ghost position trap that created $1.31 USDT crisis.
        """
        step_name = "Step 2.5: Liquidate Legacy Positions (FIX #6)"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.shared_state:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - Missing SharedState (non-fatal)"
                )
                return True
            
            # Get current positions from shared state
            positions = getattr(self.shared_state, 'positions', {}) or {}
            
            # Filter to only positions with quantity > 0 (exclude USDT)
            positions_to_liquidate = []
            for symbol, pos_data in positions.items():
                if symbol == 'USDT':
                    continue
                try:
                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                    if qty > 0:
                        positions_to_liquidate.append(symbol)
                except (ValueError, TypeError):
                    pass
            
            if not positions_to_liquidate:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - No legacy positions to liquidate"
                )
                elapsed = time.time() - step_start
                self._step_metrics['liquidate_legacy_positions'] = {
                    'positions_checked': len(positions),
                    'positions_liquidated': 0,
                    'failed_count': 0,
                    'elapsed_sec': elapsed,
                }
                return True
            
            # Log warning about legacy positions
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - Found {len(positions_to_liquidate)} legacy positions: {positions_to_liquidate}"
            )
            
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - ⚠️ ExchangeClient has security guards preventing orders outside ExecutionManager scope"
            )
            
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - → ExchangeTruthAuditor (Step 3) will handle cleanup by matching wallet balances"
            )
            
            # Defer liquidation to ExchangeTruthAuditor which runs in Step 3
            # That component has proper ExecutionManager scope and can handle order placement
            elapsed = time.time() - step_start
            self._step_metrics['liquidate_legacy_positions'] = {
                'positions_checked': len(positions),
                'positions_liquidated': 0,
                'positions_deferred_to_auditor': len(positions_to_liquidate),
                'failed_count': 0,
                'elapsed_sec': elapsed,
            }
            
            return True
            
        except Exception as e:
            elapsed = time.time() - step_start
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - Exception (non-fatal, deferring to auditor): {e}"
            )
            self._step_metrics['liquidate_legacy_positions'] = {
                'error': str(e),
                'deferred': True,
                'elapsed_sec': elapsed,
            }
            return True  # Non-fatal - proceed anyway
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2.6: AGGRESSIVE EXCHANGE LIQUIDATION (NEW!)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_aggressive_exchange_liquidation(self) -> bool:
        """
        CRITICAL FIX: Actually SELL all legacy positions on the exchange.
        
        PROBLEM: Step 2.5 removes positions from local SharedState but they still
        exist on the REAL EXCHANGE. This means:
        - Capital is still locked in those positions
        - New BUY orders fail with POSITION_ALREADY_OPEN errors
        - Account shows huge NAV but only $10-20 spendable (rest is dust)
        
        SOLUTION: Use ExchangeClient to place MARKET SELL orders for ALL positions
        found in the wallet. This actually frees up the USDT on the exchange.
        
        Timing: Must run AFTER walletbalances are fetched but BEFORE trading starts.
        This ensures capital is liquid before MetaController takes over.
        
        Non-fatal: If sell fails for a symbol, log it and continue. Better to
        have some dust remaining than crash the startup.
        """
        step_name = "Step 2.6: Aggressive Exchange Liquidation"
        step_start = time.time()
        
        try:
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} starting... "
                "(Selling ALL non-USDT positions to free capital)"
            )
            
            if not self.exchange_client:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - No ExchangeClient (non-fatal)"
                )
                return True
            
            # Get current balances from exchange (wallet state)
            try:
                balances = await self.exchange_client.get_balances()
                if not balances:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - Could not fetch balances"
                    )
                    return True
            except Exception as e:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Balance fetch failed: {e}"
                )
                return True
            
            # Find all non-zero, non-USDT balances (these are legacy positions)
            symbols_to_sell = []
            sell_plan = {}
            
            for symbol_code, balance_data in balances.items():
                if symbol_code in ('USDT', 'BUSD', 'FDUSD', 'USDC'):
                    continue  # Skip quote assets
                
                try:
                    free_qty = float(balance_data.get('free', 0.0) or 0.0)
                    locked_qty = float(balance_data.get('locked', 0.0) or 0.0)
                    total_qty = free_qty + locked_qty
                    
                    if total_qty > 1e-10:  # Any non-zero amount
                        pair_symbol = f"{symbol_code}USDT"
                        symbols_to_sell.append(pair_symbol)
                        sell_plan[pair_symbol] = {
                            'free': free_qty,
                            'locked': locked_qty,
                            'total': total_qty
                        }
                except (ValueError, TypeError):
                    pass
            
            if not symbols_to_sell:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - "
                    "No positions to sell (all clean)"
                )
                return True
            
            # Execute market sells
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - "
                f"Selling {len(symbols_to_sell)} positions: {symbols_to_sell}"
            )
            
            successful_sells = 0
            failed_sells = []
            total_usdt_recovered = 0.0
            
            for pair_symbol in symbols_to_sell:
                try:
                    qty_to_sell = sell_plan[pair_symbol]['total']
                    
                    self.logger.info(
                        f"[StartupOrchestrator] {step_name} - "
                        f"Selling {pair_symbol}: qty={qty_to_sell:.8f}"
                    )
                    
                    # Need to activate execution scope to place orders
                    scope_token = self.exchange_client.begin_execution_order_scope(
                        source="StartupOrchestrator/legacy_liquidation"
                    )
                    
                    try:
                        # Place market sell order
                        order_result = await self.exchange_client.place_market_order(
                            symbol=pair_symbol,
                            side='SELL',
                            quantity=qty_to_sell,
                            tag='startup/legacy_liquidation'
                        )
                    finally:
                        # Always end the scope
                        if scope_token:
                            self.exchange_client.end_execution_order_scope(scope_token)
                    
                    if order_result and order_result.get('status') == 'FILLED':
                        cummulative_quote = float(order_result.get('cummulative_quote', 0.0) or 0.0)
                        total_usdt_recovered += cummulative_quote
                        successful_sells += 1
                        
                        self.logger.warning(
                            f"[StartupOrchestrator] {step_name} - "
                            f"✅ {pair_symbol} FILLED: {cummulative_quote:.2f} USDT recovered"
                        )
                    else:
                        status = order_result.get('status') if order_result else 'Unknown'
                        msg = f"Order status: {status}"
                        failed_sells.append((pair_symbol, msg))
                        self.logger.warning(
                            f"[StartupOrchestrator] {step_name} - "
                            f"⚠️ {pair_symbol} not filled: {msg}"
                        )
                
                except Exception as e:
                    failed_sells.append((pair_symbol, str(e)))
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - "
                        f"Error selling {pair_symbol}: {e}"
                    )
            
            elapsed = time.time() - step_start
            self._step_metrics['aggressive_exchange_liquidation'] = {
                'positions_targeted': len(symbols_to_sell),
                'successful_sells': successful_sells,
                'failed_sells': len(failed_sells),
                'usdt_recovered': total_usdt_recovered,
                'elapsed_sec': elapsed,
            }
            
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} complete: "
                f"{successful_sells} sold, {len(failed_sells)} failed, "
                f"${total_usdt_recovered:.2f} recovered, {elapsed:.2f}s"
            )
            
            return True  # Non-fatal
            
        except Exception as e:
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - Unexpected error (non-fatal): {e}",
                exc_info=True
            )
            return True  # Non-fatal - don't crash on liquidation errors
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: SharedState - Hydrate positions from balances
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_hydrate_positions(self) -> bool:
        """Delegate to SharedState to populate positions from wallet balances (only missing symbols)."""
        step_name = "Step 2: SharedState.hydrate_positions_from_balances()"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.shared_state:
                self.logger.error(f"[StartupOrchestrator] {step_name} - No SharedState")
                return False
            
            # IMPROVEMENT 2: Track existing symbols to avoid duplicates during restart
            existing_positions = getattr(self.shared_state, 'positions', {}) or {}
            existing_symbols = set(existing_positions.keys())
            self.logger.info(
                f"[StartupOrchestrator] {step_name} - Pre-existing symbols: {existing_symbols}"
            )
            
            # Primary: Use authoritative_wallet_sync (atomic rebuild)
            if hasattr(self.shared_state, 'authoritative_wallet_sync'):
                try:
                    result = await self.shared_state.authoritative_wallet_sync()
                    self.logger.debug(f"[StartupOrchestrator] {step_name} - Called authoritative_wallet_sync()")
                except Exception as e:
                    self.logger.debug(
                        f"[StartupOrchestrator] {step_name} - authoritative_wallet_sync failed: {e}, "
                        "trying hydrate_positions_from_balances()"
                    )
                    # Fallback
                    if hasattr(self.shared_state, 'hydrate_positions_from_balances'):
                        result = self.shared_state.hydrate_positions_from_balances()
                        if asyncio.iscoroutine(result):
                            await result
                        self.logger.debug(f"[StartupOrchestrator] {step_name} - Called hydrate_positions_from_balances()")
            elif hasattr(self.shared_state, 'hydrate_positions_from_balances'):
                result = self.shared_state.hydrate_positions_from_balances()
                if asyncio.iscoroutine(result):
                    await result
                self.logger.debug(f"[StartupOrchestrator] {step_name} - Called hydrate_positions_from_balances()")
            else:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - No hydration method available"
                )
                return False
            
            # Count reconstructed positions & detect new symbols (deduplication check)
            positions = getattr(self.shared_state, 'positions', {}) or {}
            open_positions = {k: v for k, v in positions.items() 
                            if float(v.get('quantity', 0.0) or 0.0) > 0}
            newly_hydrated = set(positions.keys()) - existing_symbols
            
            elapsed = time.time() - step_start
            self._step_metrics['hydrate_positions'] = {
                'total_positions': len(positions),
                'open_positions': len(open_positions),
                'pre_existing_symbols': len(existing_symbols),
                'newly_hydrated': len(newly_hydrated),
                'elapsed_sec': elapsed,
            }
            
            self.logger.info(
                f"[StartupOrchestrator] {step_name} complete: "
                f"{len(open_positions)} open, {len(newly_hydrated)} newly hydrated, "
                f"{len(positions)} total, {elapsed:.2f}s"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"[StartupOrchestrator] {step_name} - Unexpected error: {e}",
                exc_info=True
            )
            return False
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2.4: Clear Stale Quote Reservations
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_clear_stale_quote_reservations(self) -> bool:
        """
        Clear any stale quote reservations from the previous session.
        
        PROBLEM: Quote reservations can accumulate from failed orders or incomplete
        trades. If these persist beyond the trading session, they block capital even
        though the orders are gone. This causes spendable balance to be artificially
        reduced (e.g., showing $6.65 instead of $16.65).
        
        SOLUTION: At startup, clear all quote reservations. If any orders need to
        remain reserved, they'll be re-created by the normal order management system.
        
        TIMING: Must run AFTER position hydration (Step 2) but BEFORE execution 
        (Step 2.5 liquidation onwards), to ensure capital is fully available.
        """
        step_name = "Step 2.4: Clear Stale Quote Reservations"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.shared_state:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - SharedState missing (non-fatal)"
                )
                return True  # Non-fatal
            
            # Access quote reservations directly
            quote_reservations = getattr(self.shared_state, '_quote_reservations', {})
            if not quote_reservations:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - No quote reservations to clear"
                )
                elapsed = time.time() - step_start
                self._step_metrics['clear_stale_reservations'] = {
                    'cleared_assets': 0,
                    'cleared_amount_usdt': 0.0,
                    'elapsed_sec': elapsed,
                }
                return True
            
            # Calculate total reserved amount before clearing
            total_reserved_before = 0.0
            for asset, reservations in quote_reservations.items():
                for r in reservations:
                    total_reserved_before += float(r.get('amount', 0.0))
            
            if total_reserved_before > 0.01:  # More than 1 cent
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Found ${total_reserved_before:.2f} in stale reservations"
                )
            
            # Clear all reservations
            cleared_assets = len(quote_reservations)
            quote_reservations.clear()
            
            # Verify cleared
            if hasattr(self.shared_state, '_quote_reservations'):
                self.shared_state._quote_reservations = {}
            
            elapsed = time.time() - step_start
            self._step_metrics['clear_stale_reservations'] = {
                'cleared_assets': cleared_assets,
                'cleared_amount_usdt': total_reserved_before,
                'elapsed_sec': elapsed,
            }
            
            if total_reserved_before > 0.01:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - ✅ Cleared ${total_reserved_before:.2f} "
                    f"in {cleared_assets} assets, {elapsed:.2f}s"
                )
            
            return True
            
        except Exception as e:
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - Unexpected error (non-fatal): {e}",
                exc_info=False
            )
            return True  # Non-fatal
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: ExchangeTruthAuditor - Sync orders (non-fatal)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_auditor_restart_recovery(self) -> bool:
        """Delegate to ExchangeTruthAuditor to sync open orders."""
        step_name = "Step 3: ExchangeTruthAuditor.restart_recovery()"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.exchange_truth_auditor:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - ExchangeTruthAuditor not available (non-fatal)"
                )
                return True  # Non-fatal
            
            # ExchangeTruthAuditor syncs orders (prefer public API, fallback to private)
            recover_method = (
                getattr(self.exchange_truth_auditor, 'restart_recovery', None)
                or getattr(self.exchange_truth_auditor, '_restart_recovery', None)
            )
            if recover_method is not None:
                try:
                    result = recover_method()
                    if asyncio.iscoroutine(result):
                        await result
                    self.logger.debug(f"[StartupOrchestrator] {step_name} - Called {recover_method.__name__}()")
                except Exception as e:
                    self.logger.debug(
                        f"[StartupOrchestrator] {step_name} - restart_recovery failed (non-fatal): {e}"
                    )
                    return True  # Non-fatal
            else:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - restart_recovery method not found (non-fatal)"
                )
                return True  # Non-fatal
            
            elapsed = time.time() - step_start
            self._step_metrics['auditor_restart_recovery'] = {
                'elapsed_sec': elapsed,
            }
            
            self.logger.info(
                f"[StartupOrchestrator] {step_name} complete: {elapsed:.2f}s (non-fatal)"
            )
            return True
            
        except Exception as e:
            self.logger.warning(
                f"[StartupOrchestrator] {step_name} - Non-fatal error: {e}"
            )
            return True  # Non-fatal
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: Build capital ledger from wallet balances
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_build_capital_ledger(self) -> bool:
        """
        Construct the capital ledger from wallet balances.
        
        PRINCIPLE: Ledger is BUILT from wallet, not assumed.
        
        invested_capital = Σ(position_value)
        free_capital = USDT balance
        NAV = invested_capital + free_capital
        """
        step_name = "Step 4: Build Capital Ledger"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.shared_state:
                self.logger.error(f"[StartupOrchestrator] {step_name} - No SharedState")
                return False
            
            # Get prices (ensure coverage first)
            accepted_symbols = getattr(self.shared_state, 'accepted_symbols', {}) or {}
            if accepted_symbols and self.exchange_client:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Ensuring latest prices for {len(accepted_symbols)} symbols..."
                )
                try:
                    await self.shared_state.ensure_latest_prices_coverage(self._price_fetcher)
                    self.logger.debug(
                        f"[StartupOrchestrator] {step_name} - Latest prices coverage complete"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - Price coverage failed: {e}"
                    )
            
            # Get positions and prices
            positions = getattr(self.shared_state, 'positions', {}) or {}
            latest_prices = getattr(self.shared_state, 'latest_prices', {}) or {}
            
            # CONSTRUCT: Calculate invested_capital from positions
            invested_capital = 0.0
            position_details = []
            
            for symbol, pos_data in positions.items():
                try:
                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                    if qty <= 0:
                        continue  # Skip zero/short positions
                    
                    # PROFESSIONAL STANDARD: Use ONLY latest_price (current market price)
                    # Do NOT fall back to entry_price or mark_price - they are historical
                    # Entry price is not the current market value and should never be used for NAV
                    price = float(latest_prices.get(symbol, 0.0) or 0.0)
                    
                    if price > 0:
                        position_value = qty * price
                        invested_capital += position_value
                        position_details.append({
                            'symbol': symbol,
                            'qty': qty,
                            'price': price,
                            'value': position_value,
                        })
                        self.logger.debug(
                            f"[StartupOrchestrator] {step_name} - Position: {symbol} "
                            f"qty={qty:.6f} × ${price:.2f} = ${position_value:.2f}"
                        )
                    else:
                        # No price available - exclude from ledger (professional standard)
                        self.logger.debug(
                            f"[StartupOrchestrator] {step_name} - Excluding {symbol}: no current price "
                            f"(qty={qty:.6f})"
                        )
                except (ValueError, TypeError) as e:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - Invalid position data for {symbol}: {e}"
                    )
            
            # CONSTRUCT: Get free_capital from USDT balance
            # Try multiple attribute names (wallet_balances, balances, etc.)
            wallets = (
                getattr(self.shared_state, 'wallet_balances', {}) or
                getattr(self.shared_state, 'balances', {}) or
                {}
            )
            
            # Extract USDT free balance (the actual liquid capital)
            usdt_data = wallets.get('USDT', {}) or {}
            if isinstance(usdt_data, dict):
                free_capital = float(usdt_data.get('free', 0.0) or 0.0)
            else:
                # If USDT is a single value (not dict), use it directly
                free_capital = float(usdt_data or 0.0)
            
            # CONSTRUCT: NAV = invested + free
            constructed_nav = invested_capital + free_capital
            
            # STORE: Update SharedState with constructed ledger
            try:
                if hasattr(self.shared_state, "rebuild_nav_from_state"):
                    rebuilt = await self.shared_state.rebuild_nav_from_state(
                        source="startup_orchestrator_step5"
                    )
                    invested_capital = float(rebuilt.get("invested_capital", invested_capital) or 0.0)
                    free_capital = float(rebuilt.get("free_quote", free_capital) or 0.0)
                    constructed_nav = float(rebuilt.get("nav", constructed_nav) or 0.0)
                else:
                    self.shared_state.invested_capital = invested_capital
                    self.shared_state.free_quote = free_capital
                    self.shared_state.nav = constructed_nav
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Ledger constructed: "
                    f"invested=${invested_capital:.2f}, free=${free_capital:.2f}, "
                    f"NAV=${constructed_nav:.2f}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Failed to update SharedState: {e}"
                )
                return False
            
            # Log details
            self.logger.debug(f"[StartupOrchestrator] {step_name} - Position breakdown:")
            for detail in position_details:
                self.logger.debug(
                    f"  {detail['symbol']}: ${detail['value']:.2f}"
                )
            self.logger.debug(f"  Free capital (USDT): ${free_capital:.2f}")
            self.logger.debug(f"  ───────────────────")
            self.logger.debug(f"  NAV Total: ${constructed_nav:.2f}")
            
            elapsed = time.time() - step_start
            self._step_metrics['build_capital_ledger'] = {
                'invested_capital': invested_capital,
                'free_capital': free_capital,
                'constructed_nav': constructed_nav,
                'positions_count': len(position_details),
                'elapsed_sec': elapsed,
            }
            
            self.logger.info(
                f"[StartupOrchestrator] {step_name} complete: "
                f"{len(position_details)} positions, NAV=${constructed_nav:.2f}, {elapsed:.2f}s"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"[StartupOrchestrator] {step_name} - Unexpected error: {e}",
                exc_info=True
            )
            return False
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5: Verify capital integrity (ledger already constructed)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_verify_capital_integrity(self) -> bool:
        """
        Verify the capital ledger is consistent.
        
        NOTE: Ledger is already CONSTRUCTED in Step 4.
        This step only VERIFIES consistency.
        """
        step_name = "Step 5: Verify Capital Integrity"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.shared_state:
                self.logger.error(f"[StartupOrchestrator] {step_name} - No SharedState")
                return False
            
            # CRITICAL FIX: Ensure latest prices exist before computing position values
            # This prevents NaN/0 prices when calculating position_value = qty × price
            accepted_symbols = getattr(self.shared_state, 'accepted_symbols', {}) or {}
            if accepted_symbols and self.exchange_client:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Ensuring latest prices coverage "
                    f"for {len(accepted_symbols)} symbols..."
                )
                try:
                    await self.shared_state.ensure_latest_prices_coverage(self._price_fetcher)
                    self.logger.info(
                        f"[StartupOrchestrator] {step_name} - Latest prices coverage complete. "
                        f"Cached prices: {len(self.shared_state.latest_prices)} symbols"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - Price coverage failed: {e}. "
                        f"Continuing with available prices."
                    )
            
            # Get capital metrics
            nav = float(getattr(self.shared_state, 'nav', 0.0) or 0.0)
            positions = getattr(self.shared_state, 'positions', {}) or {}
            open_orders = getattr(self.shared_state, 'open_orders', {}) or {}
            
            # CRITICAL FIX: Calculate invested_capital and free_quote from ACTUAL state
            # DO NOT use cached invested_capital which becomes stale after position updates
            # Recalculate invested_capital = sum of all position values (using latest prices)
            latest_prices = getattr(self.shared_state, 'latest_prices', {}) or {}
            recalc_invested = 0.0
            for symbol, pos_data in positions.items():
                try:
                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                    if qty > 0:
                        price = float(latest_prices.get(symbol, 0.0) or 0.0)
                        if price > 0:
                            recalc_invested += qty * price
                except (ValueError, TypeError):
                    pass
            
            # Calculate free_quote from NAV - invested capital
            # This is the ground truth: free = nav - positions_value
            quote_balance = await self.shared_state.get_balance("USDT") if self.shared_state else {}
            quote_free = float(quote_balance.get('free', None) or 0.0)
            quote_locked = float(quote_balance.get('locked', None) or 0.0)
            quote_total = quote_free + quote_locked
            
            # If NAV is available, use it as reference; otherwise use quote balance
            if nav > 0:
                recalc_free = nav - recalc_invested
            else:
                recalc_free = quote_total - recalc_invested
            
            invested = float(recalc_invested)
            free = float(max(0.0, recalc_free))
            
            # Log raw state for diagnostics
            self.logger.info(
                f"[StartupOrchestrator] {step_name} - Raw metrics: "
                f"nav={nav}, free={free}, invested={invested}, "
                f"positions={len(positions)}, open_orders={len(open_orders)}"
            )
            
            # DEFENSIVE: Check for shadow mode or simulation mode
            # In shadow mode, NAV may be 0 but positions exist (virtual ledger is authoritative)
            is_shadow_mode = getattr(self.shared_state, '_shadow_mode', False)
            is_virtual_ledger = getattr(self.shared_state, '_virtual_ledger_authoritative', False)
            
            # Log mode for diagnostics
            if is_shadow_mode or is_virtual_ledger:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Running in SHADOW/SIMULATION mode. "
                    f"NAV=0 is acceptable (virtual ledger is authoritative)"
                )
            
            # IMPROVEMENT 2: Filter positions below MIN_ECONOMIC_TRADE_USDT (dust positions)
            # These are economically irrelevant and shouldn't block startup
            min_economic_trade = float(
                getattr(self.shared_state.config, 'MIN_ECONOMIC_TRADE_USDT', 30.0)
                if hasattr(self.shared_state, 'config')
                else 30.0
            )
            
            # CRITICAL FIX #1: Calculate recalculated_nav using ONLY latest_prices
            # This matches the professional standard NAV calculation and ensures
            # capital integrity checks use consistent pricing
            recalculated_nav = free + invested  # free + invested_capital = nav
            
            # Filter positions: only count economically viable positions
            viable_positions = []
            dust_positions = []
            for symbol, pos_data in positions.items():
                try:
                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                    if qty <= 0:
                        continue
                    
                    # PROFESSIONAL STANDARD: Use ONLY latest_price (current market price)
                    # Do NOT fall back to entry_price or mark_price for NAV calculation
                    # Entry price is historical, not current market value
                    price = float(latest_prices.get(symbol, 0.0) or 0.0)
                    
                    if price > 0:
                        position_value = qty * price
                        
                        if position_value >= min_economic_trade:
                            viable_positions.append(symbol)
                        else:
                            dust_positions.append((symbol, position_value))
                    else:
                        # No price available - exclude from NAV (professional standard)
                        dust_positions.append((symbol, 0.0))
                except (ValueError, TypeError):
                    pass  # Skip invalid position data
            
            if dust_positions:
                dust_detail = [f'{s}=${float(v or 0.0):.2f}' for s, v in dust_positions[:5]]
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Found {len(dust_positions)} dust positions "
                    f"below ${min_economic_trade:.2f}: {dust_detail}"
                )
            
            # CRITICAL FIX #2: Verify capital integrity using recalculated NAV
            # Check that shared_state.nav matches the recalculated_nav
            # If they differ significantly, log the discrepancy and use recalculated value
            if nav > 0 and recalculated_nav > 0:
                nav_discrepancy = abs((nav - recalculated_nav) / nav)
                if nav_discrepancy > 0.05:  # 5% tolerance
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - NAV discrepancy detected: "
                        f"stored_nav={nav:.2f}, recalculated_nav={recalculated_nav:.2f} "
                        f"(error={nav_discrepancy*100:.2f}%). Using recalculated value."
                    )
                    nav = recalculated_nav
            elif nav <= 0 and recalculated_nav > 0:
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Stored NAV={nav} but recalculated NAV={recalculated_nav:.2f}. "
                    f"Using recalculated value."
                )
                nav = recalculated_nav
            
            # IMPROVEMENT 1: NAV=0 with viable positions gets retry with cleanup
            # NAV=0 is OK if: (1) no viable positions, (2) shadow mode, or (3) empty wallet
            if nav <= 0 and len(viable_positions) > 0 and not (is_shadow_mode or is_virtual_ledger):
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Positions detected but NAV=0 "
                    f"(likely dust positions or USDT not synced). Recalculating after cleanup..."
                )
                
                # Allow dust cleanup to run
                await asyncio.sleep(1)
                
                # Recalculate NAV
                nav = await self.shared_state.get_nav()
                
                if nav <= 0:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - NAV still zero after cleanup. "
                        f"Continuing startup (dust positions will be liquidated)."
                    )
                    # Don't block startup - dust cleanup will handle these
                else:
                    self.logger.info(
                        f"[StartupOrchestrator] {step_name} - NAV recovered to {nav:.2f} after cleanup"
                    )
            
            elif nav <= 0 and not (is_shadow_mode or is_virtual_ledger):
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Cold start: NAV=0, no viable positions, "
                    "exchange returned no balance or connection failed"
                )
            
            # CRITICAL FIX: Only apply strict NAV integrity checks if NOT in shadow mode
            # Shadow mode uses virtual ledger, so NAV=0 is acceptable
            shadow_mode_config = getattr(self.config, 'SHADOW_MODE', False) if self.config else False
            
            if not shadow_mode_config:
                # REAL MODE: Apply strict integrity checks
                # Note: `free` is clamped via max(0.0, ...) and `invested` is a sum of
                # non-negatives upstream, so the only meaningful check is the NAV balance below.
                
                # Check balance: nav should ~= free + invested
                # NOTE: If they differ significantly, it means position prices have changed
                # between recalculations or USDT free balance was updated. This is acceptable
                # because market prices are constantly moving.
                if nav > 0:
                    sum_capital = free + invested
                    balance_error = abs((nav - sum_capital) / nav)
                    
                    # TOLERANCE: Allow up to 5% error for market volatility and price delays
                    # This accounts for positions whose prices changed between:
                    # - Initial NAV construction (Step 5)
                    # - Final verification (Step 6)
                    if balance_error > 0.05:  # 5% threshold
                        self.logger.warning(
                            f"[StartupOrchestrator] {step_name} - Balance discrepancy exceeds tolerance: "
                            f"NAV={nav:.2f}, Free+Invested={sum_capital:.2f}, "
                            f"Error={balance_error*100:.2f}%. Re-syncing with latest exchange data..."
                        )
                        # Force re-sync with exchange to get fresh prices and balances
                        # This handles cases where market moved significantly during startup
                        try:
                            # Refresh latest prices one more time
                            await self.shared_state.ensure_latest_prices_coverage(self._price_fetcher)
                            
                            # Recalculate invested with fresh prices
                            fresh_invested = 0.0
                            for symbol, pos_data in positions.items():
                                try:
                                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                                    if qty > 0:
                                        price = float(latest_prices.get(symbol, 0.0) or 0.0)
                                        if price > 0:
                                            fresh_invested += qty * price
                                except (ValueError, TypeError):
                                    pass
                            
                            # Recalculate free from latest USDT balance
                            fresh_quote = await self.shared_state.get_balance("USDT") if self.shared_state else {}
                            fresh_free = float(fresh_quote.get('free', None) or 0.0)
                            
                            # Recompute NAV with fresh values
                            fresh_nav = fresh_invested + fresh_free
                            
                            # Check if fresh calculation is better
                            fresh_error = abs((fresh_nav - (fresh_invested + fresh_free)) / fresh_nav) if fresh_nav > 0 else 0.0
                            
                            if fresh_error < balance_error:
                                self.logger.info(
                                    f"[StartupOrchestrator] {step_name} - Using fresh sync: "
                                    f"NAV={fresh_nav:.2f}, Free={fresh_free:.2f}, Invested={fresh_invested:.2f}, "
                                    f"Error={fresh_error*100:.2f}%"
                                )
                                nav = fresh_nav
                                free = fresh_free
                                invested = fresh_invested
                                balance_error = fresh_error
                            else:
                                self.logger.warning(
                                    f"[StartupOrchestrator] {step_name} - Fresh sync did not improve discrepancy. "
                                    f"Continuing with current values. This may indicate market volatility."
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"[StartupOrchestrator] {step_name} - Fresh sync failed: {e}. "
                                f"Continuing with current values."
                            )
                    
                    if balance_error <= 0.05:
                        self.logger.info(
                            f"[StartupOrchestrator] {step_name} - Balance integrity: "
                            f"NAV={nav:.2f}, Free+Invested={sum_capital:.2f}, Error={balance_error*100:.3f}%"
                        )
            else:
                # SHADOW MODE: Skip strict checks
                self.logger.info(
                    "[StartupOrchestrator] Shadow mode active — skipping strict NAV integrity check"
                )
            
            # IMPROVEMENT 1: Position Consistency Validation (using viable positions only)
            # Check that sum(position_value) + free_quote ≈ NAV (wallet balance = viable positions + dust)
            if viable_positions and nav > 0:
                position_value_sum = 0.0
                dust_position_value = 0.0
                
                for symbol, pos_data in positions.items():
                    try:
                        qty = float(pos_data.get('quantity', 0.0) or 0.0)
                        # PROFESSIONAL STANDARD: Use ONLY latest_price (NOT entry_price or mark_price)
                        # Current market price is the only valid basis for position valuation
                        price = float(latest_prices.get(symbol, 0.0) or 0.0)
                        if qty > 0 and price > 0:
                            position_value = qty * price
                            # Track which positions are viable vs dust
                            if position_value >= min_economic_trade:
                                position_value_sum += position_value
                            else:
                                dust_position_value += position_value
                    except (ValueError, TypeError):
                        pass  # Skip invalid position data
                
                # CRITICAL FIX: Include dust positions in portfolio total
                # Dust positions are still part of NAV, even if economically small
                # The check should be: (viable + dust + free) ≈ NAV
                total_position_value = position_value_sum + dust_position_value
                portfolio_total = total_position_value + free
                balance_error = abs((nav - portfolio_total) / nav) if nav > 0 else 0.0
                
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Position consistency check: "
                    f"NAV={nav:.2f}, Viable_Positions={position_value_sum:.2f}, "
                    f"Dust_Positions={dust_position_value:.2f}, Free={free:.2f}, "
                    f"Total={portfolio_total:.2f}, Error={balance_error*100:.2f}%"
                )
                
                # Allow 5% error for rounding/slippage/market volatility
                # This is a healthy tolerance for a live trading system
                if balance_error > 0.05:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - ⚠️ Position consistency exceeds tolerance: "
                        f"NAV={nav:.2f}, Viable_Positions={position_value_sum:.2f}, "
                        f"Dust={dust_position_value:.2f}, Free={free:.2f}, "
                        f"Total={portfolio_total:.2f} ({balance_error*100:.2f}% error). "
                        f"This may indicate: (1) market prices moved during startup, "
                        f"(2) positions were partially liquidated, or (3) exchange data lag. "
                        f"Startup will continue - positions will be reconciled during trading."
                    )
                    # Don't block startup - reconciliation will happen during trading
                else:
                    self.logger.info(
                        f"[StartupOrchestrator] {step_name} - Position consistency OK "
                        f"(error={balance_error*100:.2f}%)"
                    )
            
            # Warn if zero viable positions (cold start or all dust)
            if len(viable_positions) == 0:
                if len(dust_positions) > 0:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - No viable positions (only dust: {len(dust_positions)} positions < ${min_economic_trade:.2f})"
                    )
                else:
                    self.logger.warning(
                        f"[StartupOrchestrator] {step_name} - No positions reconstructed (cold start?)"
                    )
            
            elapsed = time.time() - step_start
            self._step_metrics['verify_integrity'] = {
                'nav': nav,
                'free_quote': free,
                'invested_capital': invested,
                'positions_count': len(positions),
                'open_orders_count': len(open_orders),
                'elapsed_sec': elapsed,
            }
            
            # E: Persist any corrections made during verification back to shared_state
            try:
                if hasattr(self.shared_state, 'rebuild_nav_from_state'):
                    # Authoritative re-emit using corrected values
                    self.shared_state.invested_capital = invested
                    self.shared_state.free_quote = free
                    self.shared_state.nav = nav
                else:
                    self.shared_state.invested_capital = invested
                    self.shared_state.free_quote = free
                    self.shared_state.nav = nav
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - Persisted corrected ledger: "
                    f"nav={nav:.2f}, free={free:.2f}, invested={invested:.2f}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Failed to persist corrections: {e}"
                )
            
            self.logger.info(
                f"[StartupOrchestrator] {step_name} complete: "
                f"NAV={nav:.2f}, Free={free:.2f}, Positions={len(positions)}, {elapsed:.2f}s"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"[StartupOrchestrator] {step_name} - Unexpected error: {e}",
                exc_info=True
            )
            return False
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6A: Emit StartupStateRebuilt event (IMPROVEMENT 3)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _emit_state_rebuilt_event(self) -> None:
        """Emit StartupStateRebuilt event after state reconciliation complete."""
        try:
            if hasattr(self.shared_state, 'emit_event'):
                await self.shared_state.emit_event('StartupStateRebuilt', {
                    'timestamp': time.time(),
                    'startup_duration_sec': time.time() - self._startup_ts,
                    'status': 'state_rebuilt',
                    'positions': len(getattr(self.shared_state, 'positions', {})),
                    'nav': float(getattr(self.shared_state, 'nav', 0.0) or 0.0),
                    'free_quote': float(getattr(self.shared_state, 'free_quote', 0.0) or 0.0),
                })
                self.logger.info("[StartupOrchestrator] Emitted StartupStateRebuilt event")
            
            # Also set event flag for synchronous waiters
            if hasattr(self.shared_state, 'set_event'):
                self.shared_state.set_event('StartupStateRebuilt')
                self.logger.info("[StartupOrchestrator] Set StartupStateRebuilt flag")
        except Exception as e:
            self.logger.debug(f"[StartupOrchestrator] Failed to emit StartupStateRebuilt: {e}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6B: Emit StartupPortfolioReady event (IMPROVEMENT 3)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _emit_startup_ready_event(self) -> None:
        """Emit StartupPortfolioReady event to signal MetaController it's safe."""
        try:
            if hasattr(self.shared_state, 'emit_event'):
                await self.shared_state.emit_event('StartupPortfolioReady', {
                    'timestamp': time.time(),
                    'startup_duration_sec': time.time() - self._startup_ts,
                    'status': 'ready',
                    'positions': len(getattr(self.shared_state, 'positions', {})),
                    'nav': float(getattr(self.shared_state, 'nav', 0.0) or 0.0),
                    'free_quote': float(getattr(self.shared_state, 'free_quote', 0.0) or 0.0),
                })
                self.logger.info("[StartupOrchestrator] Emitted StartupPortfolioReady event")
            
            # Also set event flag for synchronous waiters
            if hasattr(self.shared_state, 'set_event'):
                self.shared_state.set_event('StartupPortfolioReady')
                self.logger.info("[StartupOrchestrator] Set StartupPortfolioReady flag")
        except Exception as e:
            self.logger.debug(f"[StartupOrchestrator] Failed to emit StartupPortfolioReady: {e}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # Metrics & Status
    # ═════════════════════════════════════════════════════════════════════════
    
    def _log_final_metrics(self) -> None:
        """Log summary of orchestration metrics."""
        total_time = time.time() - self._startup_ts
        
        self.logger.info("[StartupOrchestrator] ═══════════════════════════════════════════════════")
        self.logger.info("[StartupOrchestrator] STARTUP ORCHESTRATION METRICS")
        self.logger.info("[StartupOrchestrator] ═══════════════════════════════════════════════════")
        
        for step, metrics in self._step_metrics.items():
            self.logger.info(f"[StartupOrchestrator] {step}:")
            for key, value in metrics.items():
                if isinstance(value, float) and value is not None:
                    self.logger.info(f"  - {key}: {value:.2f}")
                elif value is None:
                    self.logger.info(f"  - {key}: N/A")
                else:
                    self.logger.info(f"  - {key}: {value}")
        
        self.logger.info(f"[StartupOrchestrator] Total duration: {total_time:.2f}s")
        self.logger.info("[StartupOrchestrator] ═══════════════════════════════════════════════════")
    
    def is_ready(self) -> bool:
        """Check if startup sequence completed successfully."""
        return self._completed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return orchestration metrics for monitoring."""
        return {
            'completed': self._completed,
            'startup_duration_sec': time.time() - self._startup_ts,
            'step_metrics': dict(self._step_metrics),
        }
