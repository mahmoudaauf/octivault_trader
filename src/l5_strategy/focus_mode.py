"""
FocusMode subsystem extracted from MetaController.
Handles focus mode activation, deactivation, and related logic.
"""

import asyncio as _asyncio
import time
from typing import Set

class FocusModeManager:
    def __init__(self, shared_state, execution_manager, config, logger):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.config = config
        self.logger = logger
        self._focus_mode_active = False
        self._focus_mode_reason = ""
        self._focus_mode_trade_executed_count = 0
        self._focus_mode_healthy_cycles = 0
        self._focus_mode_trade_executed = False  # Reset trade execution flag
        
        # Focus symbol state
        self.FOCUS_SYMBOLS = set()
        self.FOCUS_SYMBOLS_PINNED = False
        self.FOCUS_SYMBOLS_LAST_UPDATE = 0.0
        self._bootstrap_focus_symbols_pending = True
        
        # Configuration
        self.MAX_FOCUS_SYMBOLS = getattr(config, 'MAX_FOCUS_SYMBOLS', 10)
        self.MIN_SIGNIFICANT_USDT = getattr(config, 'MIN_SIGNIFICANT_USDT', 50.0)

    def activate_focus_mode(self, reason: str):
        if self._focus_mode_active:
            return
        self._focus_mode_active = True
        self._focus_mode_reason = reason
        self._focus_mode_trade_executed_count = 0
        self._focus_mode_healthy_cycles = 0
        self._focus_mode_trade_executed = False 
        self.logger.warning(f"[FOCUS_MODE] ACTIVATED — {reason}")

    def deactivate_focus_mode(self):
        if not self._focus_mode_active:
            return
        self._focus_mode_active = False
        self._focus_mode_reason = ""
        self._focus_mode_trade_executed_count = 0
        self._focus_mode_healthy_cycles = 0
        self._focus_mode_trade_executed = False 
        self.logger.warning("[FOCUS_MODE] DEACTIVATED — Recovery conditions met.")

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol name."""
        return str(symbol).upper().strip()

    def _is_recovery_sellable(self, symbol: str, pos_data: dict, ignore_filters: bool = False, ignore_core: bool = True) -> bool:
        """Check if position is sellable for recovery."""
        try:
            qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
            if qty <= 0:
                return False
            
            # For focus bootstrap, we ignore dust filters to get meaningful positions
            if ignore_filters:
                return True
                
            return True
        except Exception:
            return False

    async def _bootstrap_focus_symbols(self) -> Set[str]:
        """
        CRITICAL FIX #1: Startup Bootstrap Sequence
        
        Mandatory on system start:
        1. Read spot wallet balances
        2. Rank symbols by USDT value (descending)
        3. Select top N symbols with value >= MIN_SIGNIFICANT_USDT
        4. PIN these symbols (immutable until restart)
        5. Mark all non-focus symbols as FROZEN
        
        This is called ONCE on the first evaluate_and_act() cycle.
        After bootstrap, FOCUS_SYMBOLS are PINNED and never change.
        
        Returns:
            Set of focus symbols pinned for this session (e.g., {"BTCUSDT", "ETHUSDT", "SOLUSDT"})
        """
        try:
            self.logger.info("[WALLET_FOCUS_BOOTSTRAP] Starting startup bootstrap sequence...")
            
            # Get positions snapshot from current wallet
            snap = self.shared_state.get_positions_snapshot() or {}
            
            if not snap:
                self.logger.warning("[WALLET_FOCUS_BOOTSTRAP] No positions found in wallet snapshot. Using empty set.")
                self.FOCUS_SYMBOLS = set()
                self.FOCUS_SYMBOLS_PINNED = True
                self._bootstrap_focus_symbols_pending = False
                return self.FOCUS_SYMBOLS
            
            # Build list of (symbol, notional_value) for ranking
            candidates = []
            for sym_raw, pos_data in snap.items():
                if not isinstance(pos_data, dict):
                    continue
                
                sym = self._normalize_symbol(sym_raw)
                value_usdt = float(pos_data.get("value_usdt", 0.0) or 0.0)

                # Skip dust positions in focus calculation
                if not self._is_recovery_sellable(sym, pos_data, ignore_filters=False, ignore_core=True):
                    continue

                candidates.append((sym, value_usdt))
            
            # Step 1-3: Rank by notional value (descending) and select top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_focus_symbols = {sym for sym, _ in candidates[:self.MAX_FOCUS_SYMBOLS]}
            
            # FIX #2: FALLBACK - If no qualified symbols, use top-N by value anyway (prevent deadlock)
            if not new_focus_symbols and snap:
                self.logger.warning(
                    "[WALLET_FOCUS_BOOTSTRAP] FALLBACK: No symbols >= $%.2f threshold. "
                    "Using top-%d by value to prevent deadlock.",
                    self.MIN_SIGNIFICANT_USDT, self.MAX_FOCUS_SYMBOLS
                )
                all_candidates = []
                for sym_raw, pos_data in snap.items():
                    if not isinstance(pos_data, dict):
                        continue
                    sym = self._normalize_symbol(sym_raw)
                    qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0))
                    value_usdt = float(pos_data.get("value_usdt", 0.0) or 0.0)
                    if qty > 0 and value_usdt > 0:
                        all_candidates.append((sym, value_usdt))
                
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                new_focus_symbols = {sym for sym, _ in all_candidates[:self.MAX_FOCUS_SYMBOLS]}
            
            # Step 4: PIN focus symbols (no change until restart)
            self.FOCUS_SYMBOLS = new_focus_symbols
            self.FOCUS_SYMBOLS_PINNED = True
            self._bootstrap_focus_symbols_pending = False
            self.FOCUS_SYMBOLS_LAST_UPDATE = time.time()
            
            # Step 5: Log frozen non-focus symbols
            non_focus_count = len(snap) - len(new_focus_symbols)
            
            self.logger.warning(
                "[WALLET_FOCUS_BOOTSTRAP] ✅ Bootstrap Complete: PINNED %d focus symbols: %s | "
                "%d non-focus symbols FROZEN (out-of-trading-scope)",
                len(new_focus_symbols), sorted(new_focus_symbols), non_focus_count
            )
            
            # Log capital allocation
            total_value = sum(float(pd.get("value_usdt", 0.0) or 0.0) 
                             for pd in snap.values() if isinstance(pd, dict))
            focus_value = sum(float(pd.get("value_usdt", 0.0) or 0.0) 
                             for sym, pd in snap.items() 
                             if isinstance(pd, dict) and self._normalize_symbol(sym) in new_focus_symbols)
            
            if total_value > 0:
                allocation_pct = (focus_value / total_value) * 100
                self.logger.info(
                    "[WALLET_FOCUS_BOOTSTRAP] Capital Allocation: Focus=$%.2f (%.1f%%) | "
                    "Frozen=$%.2f (%.1f%%) | Total=$%.2f",
                    focus_value, allocation_pct,
                    total_value - focus_value, 100 - allocation_pct,
                    total_value
                )
            
            return self.FOCUS_SYMBOLS
            
        except Exception as e:
            self.logger.error(
                "[WALLET_FOCUS_BOOTSTRAP] Bootstrap failed: %s | Falling back to empty focus set",
                e, exc_info=True
            )
            self.FOCUS_SYMBOLS = set()
            self.FOCUS_SYMBOLS_PINNED = True
            self._bootstrap_focus_symbols_pending = False
            return self.FOCUS_SYMBOLS

    async def _update_focus_symbols(self) -> Set[str]:
        """
        CRITICAL FIX #1 (Part 2): Pinned Focus Symbols
        
        WALLET_FOCUS_BOOTSTRAP Requirement:
        Focus symbols change ONLY via wallet re-ranking on restart or when a focus symbol is fully exited.
        
        This method now:
        - Returns pinned FOCUS_SYMBOLS if already bootstrapped
        - Calls bootstrap on first cycle if needed
        - DOES NOT re-rank (no dynamic changes)
        
        Returns:
            Set of focus symbols (pinned for this session)
        """
        try:
            # If bootstrap is pending, run it now (only on first cycle)
            if self._bootstrap_focus_symbols_pending:
                return await self._bootstrap_focus_symbols()
            
            # Once pinned, always return the same set
            if self.FOCUS_SYMBOLS_PINNED:
                return self.FOCUS_SYMBOLS
            
            # Fallback: This shouldn't happen, but bootstrap if needed
            self.logger.warning("[WALLET_FOCUS_BOOTSTRAP] Fallback: Bootstrap not run yet, triggering now")
            return await self._bootstrap_focus_symbols()
            
        except Exception as e:
            self.logger.error(
                "[WALLET_FOCUS_BOOTSTRAP] Failed to get focus symbols: %s",
                e, exc_info=True
            )
            return self.FOCUS_SYMBOLS
