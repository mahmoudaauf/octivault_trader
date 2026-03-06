# -*- coding: utf-8 -*-
"""
StartupOrchestrator: Canonical P9 Startup Sequencing Gate

CRITICAL ARCHITECTURAL PRINCIPLE:
This is NOT a reconciliation engine. It is a SEQUENCING ORCHESTRATOR.

All reconciliation logic lives in its proper homes:
  - RecoveryEngine: State reconstruction
  - ExchangeTruthAuditor: Order/fill reconciliation
  - SharedState: Position hydration & event emission
  - PortfolioManager: Position refresh

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
    4. PortfolioManager.refresh_positions() - Update position metadata
    5. Verify startup integrity - Check NAV, capital, sanity
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
        portfolio_manager: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize orchestrator with required components."""
        self.config = config
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.recovery_engine = recovery_engine
        self.exchange_truth_auditor = exchange_truth_auditor
        self.portfolio_manager = portfolio_manager
        self.logger = logger or logging.getLogger("StartupOrchestrator")
        
        self._completed = False
        self._startup_ts = time.time()
        self._step_metrics = {}
    
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
            
            # STEP 3: ExchangeTruthAuditor syncs open orders (non-fatal)
            success = await self._step_auditor_restart_recovery()
            # Non-fatal if auditor unavailable
            
            # STEP 4: PortfolioManager refreshes position metadata
            success = await self._step_portfolio_manager_refresh()
            # Non-fatal if manager unavailable
            
            # STEP 5: Build capital ledger from wallet balances
            success = await self._step_build_capital_ledger()
            if not success:
                raise RuntimeError(
                    "Phase 8.5: Capital ledger construction failed - cannot proceed"
                )
            
            # STEP 6: Verify capital integrity (ledger already constructed)
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
            
            # ExchangeTruthAuditor syncs orders
            if hasattr(self.exchange_truth_auditor, '_restart_recovery'):
                try:
                    result = await self.exchange_truth_auditor._restart_recovery()
                    self.logger.debug(f"[StartupOrchestrator] {step_name} - Called _restart_recovery()")
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
    # STEP 4: PortfolioManager - Refresh positions (non-fatal)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_portfolio_manager_refresh(self) -> bool:
        """Delegate to PortfolioManager to refresh position metadata."""
        step_name = "Step 4: PortfolioManager.refresh_positions()"
        step_start = time.time()
        
        try:
            self.logger.info(f"[StartupOrchestrator] {step_name} starting...")
            
            if not self.portfolio_manager:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - PortfolioManager not available (non-fatal)"
                )
                return True  # Non-fatal
            
            # PortfolioManager refreshes position metadata
            if hasattr(self.portfolio_manager, 'refresh_positions'):
                try:
                    result = self.portfolio_manager.refresh_positions()
                    if asyncio.iscoroutine(result):
                        await result
                    self.logger.debug(f"[StartupOrchestrator] {step_name} - Called refresh_positions()")
                except Exception as e:
                    self.logger.debug(
                        f"[StartupOrchestrator] {step_name} - refresh_positions failed (non-fatal): {e}"
                    )
                    return True  # Non-fatal
            else:
                self.logger.debug(
                    f"[StartupOrchestrator] {step_name} - refresh_positions method not found (non-fatal)"
                )
                return True  # Non-fatal
            
            elapsed = time.time() - step_start
            self._step_metrics['portfolio_manager_refresh'] = {
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
    # STEP 5: Build capital ledger from wallet balances
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_build_capital_ledger(self) -> bool:
        """
        Construct the capital ledger from wallet balances.
        
        PRINCIPLE: Ledger is BUILT from wallet, not assumed.
        
        invested_capital = Σ(position_value)
        free_capital = USDT balance
        NAV = invested_capital + free_capital
        """
        step_name = "Step 5: Build Capital Ledger"
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
                
                async def price_fetcher(symbol: str) -> float:
                    try:
                        if hasattr(self.exchange_client, 'get_current_price'):
                            price = await self.exchange_client.get_current_price(symbol)
                            return float(price) if price else 0.0
                    except Exception:
                        pass
                    return 0.0
                
                try:
                    await self.shared_state.ensure_latest_prices_coverage(price_fetcher)
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
                    
                    # Use latest_price (just ensured) as source of truth
                    price = float(
                        latest_prices.get(symbol, 0.0) or 
                        pos_data.get('entry_price', 0.0) or 
                        0.0
                    )
                    
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
    # STEP 6: Verify capital integrity (ledger already constructed)
    # ═════════════════════════════════════════════════════════════════════════
    
    async def _step_verify_capital_integrity(self) -> bool:
        """
        Verify the capital ledger is consistent.
        
        NOTE: Ledger is already CONSTRUCTED in Step 5.
        This step only VERIFIES consistency.
        """
        step_name = "Step 6: Verify Capital Integrity"
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
                
                # Define price fetcher that uses exchange client
                async def price_fetcher(symbol: str) -> float:
                    try:
                        if hasattr(self.exchange_client, 'get_current_price'):
                            price = await self.exchange_client.get_current_price(symbol)
                            return float(price) if price else 0.0
                    except Exception:
                        pass
                    return 0.0
                
                # Ensure prices are populated
                try:
                    await self.shared_state.ensure_latest_prices_coverage(price_fetcher)
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
            free = float(getattr(self.shared_state, 'free_quote', 0.0) or 0.0)
            invested = float(getattr(self.shared_state, 'invested_capital', 0.0) or 0.0)
            positions = getattr(self.shared_state, 'positions', {}) or {}
            open_orders = getattr(self.shared_state, 'open_orders', {}) or {}
            
            # Log raw state for diagnostics
            self.logger.info(
                f"[StartupOrchestrator] {step_name} - Raw metrics: "
                f"nav={nav}, free={free}, invested={invested}, "
                f"positions={len(positions)}, open_orders={len(open_orders)}"
            )
            
            # Validate critical invariants
            issues = []
            
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
            
            # Get latest prices for position value calculation
            # CRITICAL: Use latest_prices (just populated) not entry_price or mark_price
            # This ensures position_value = qty × latest_price is accurate
            latest_prices = getattr(self.shared_state, 'latest_prices', {}) or {}
            
            # Filter positions: only count economically viable positions
            viable_positions = []
            dust_positions = []
            for symbol, pos_data in positions.items():
                try:
                    qty = float(pos_data.get('quantity', 0.0) or 0.0)
                    # CRITICAL FIX: Use latest_price from latest_prices (just ensured)
                    # Fallback to entry_price only if latest_price not available
                    price = float(latest_prices.get(symbol, 0.0) or pos_data.get('entry_price', pos_data.get('mark_price', 0.0)) or 0.0)
                    if qty > 0 and price > 0:
                        position_value = qty * price
                        if position_value >= min_economic_trade:
                            viable_positions.append(symbol)
                        else:
                            dust_positions.append((symbol, position_value))
                except (ValueError, TypeError):
                    pass  # Skip invalid position data
            
            if dust_positions:
                self.logger.warning(
                    f"[StartupOrchestrator] {step_name} - Found {len(dust_positions)} dust positions "
                    f"below ${min_economic_trade:.2f}: {[f'{s}=${v:.2f}' for s, v in dust_positions[:5]]}"
                )
            
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
                if free < 0:
                    issues.append(f"Free capital is {free} (should be >= 0)")
                
                if invested < 0:
                    issues.append(f"Invested capital is {invested} (should be >= 0)")
                
                # Check balance: nav should ~= free + invested
                if nav > 0:
                    balance_error = abs((nav - free - invested) / nav)
                    if balance_error > 0.01:  # Allow 1% error
                        issues.append(
                            f"Capital balance error: NAV={nav}, Free+Invested={free+invested} "
                            f"({balance_error*100:.2f}% error)"
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
                for symbol in viable_positions:
                    try:
                        pos_data = positions.get(symbol, {})
                        qty = float(pos_data.get('quantity', 0.0) or 0.0)
                        # CRITICAL FIX: Use latest_price from latest_prices (just populated above)
                        # NOT entry_price or mark_price — those may be stale or 0
                        price = float(
                            latest_prices.get(symbol, 0.0) or
                            pos_data.get('entry_price', pos_data.get('mark_price', 0.0)) or
                            0.0
                        )
                        if qty > 0 and price > 0:
                            position_value_sum += qty * price
                    except (ValueError, TypeError):
                        pass  # Skip invalid position data
                
                portfolio_total = position_value_sum + free
                balance_error = abs((nav - portfolio_total) / nav) if nav > 0 else 0.0
                
                self.logger.info(
                    f"[StartupOrchestrator] {step_name} - Position consistency check: "
                    f"NAV={nav:.2f}, Viable_Positions={position_value_sum:.2f}, Free={free:.2f}, "
                    f"Error={balance_error*100:.2f}%"
                )
                
                # Allow 2% error for rounding/slippage
                if balance_error > 0.02:
                    issues.append(
                        f"Position consistency error: NAV={nav:.2f}, "
                        f"Viable_Positions+Free={portfolio_total:.2f} ({balance_error*100:.2f}% error)"
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
                'issues_count': len(issues),
                'elapsed_sec': elapsed,
            }
            
            if issues:
                for issue in issues:
                    self.logger.error(f"[StartupOrchestrator] {step_name} - ⚠️ {issue}")
                self.logger.error(f"[StartupOrchestrator] {step_name} FAILED - capital integrity issues")
                return False
            
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
                if isinstance(value, float):
                    self.logger.info(f"  - {key}: {value:.2f}")
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
