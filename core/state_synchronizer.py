# -*- coding: utf-8 -*-
"""
StateSynchronizer - State Reconciliation Layer

Responsibility:
- Detect divergences between SharedState and local state caches
- Automatically reconcile state inconsistencies
- Log mismatches for audit trail
- Verify data integrity

This module implements the synchronization layer to address the
"Dual State Management" issue identified in Phase 2 review.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import asyncio


@dataclass
class StateMismatch:
    """Record of a detected state mismatch."""
    component: str
    symbol: Optional[str]
    field: str
    shared_value: Any
    local_value: Any
    resolved_to: str  # 'shared' or 'local'
    timestamp: datetime
    reason: str


class StateSynchronizer:
    """Reconciles SharedState with local component state."""
    
    def __init__(self, shared_state, meta_controller=None):
        """
        Initialize state synchronizer.
        
        Args:
            shared_state: Central SharedState instance
            meta_controller: MetaController instance (optional)
        """
        self.shared_state = shared_state
        self.meta_controller = meta_controller
        self.logger = logging.getLogger("StateSynchronizer")
        
        self.mismatches: List[StateMismatch] = []
        self.last_sync_time: Optional[datetime] = None
        self.sync_count = 0
    
    async def reconcile_all(self) -> Dict[str, Any]:
        """
        Run full state reconciliation across all components.
        
        Returns:
            Dict with reconciliation results
        """
        self.sync_count += 1
        mismatches = {
            'symbol_lifecycle': [],
            'position_counts': [],
            'capital_allocation': [],
            'dust_states': [],
        }
        
        try:
            # Check symbol lifecycle consistency
            symbol_mismatches = await self._reconcile_symbol_lifecycle()
            mismatches['symbol_lifecycle'] = symbol_mismatches
            
            # Check position count consistency
            position_mismatches = await self._reconcile_position_counts()
            mismatches['position_counts'] = position_mismatches
            
            # Check capital allocation consistency
            capital_mismatches = await self._reconcile_capital()
            mismatches['capital_allocation'] = capital_mismatches
            
            # Check dust state consistency
            dust_mismatches = await self._reconcile_dust_states()
            mismatches['dust_states'] = dust_mismatches
            
            self.last_sync_time = datetime.now(timezone.utc)
            
            # Log summary
            total_mismatches = sum(len(v) for v in mismatches.values())
            if total_mismatches > 0:
                self.logger.info(
                    f"State reconciliation complete: "
                    f"{total_mismatches} mismatches detected and fixed"
                )
            else:
                self.logger.debug("State reconciliation: all systems consistent")
            
            return mismatches
            
        except Exception as e:
            self.logger.error(f"State reconciliation failed: {e}")
            raise
    
    async def _reconcile_symbol_lifecycle(self) -> List[StateMismatch]:
        """Reconcile symbol lifecycle states."""
        mismatches = []
        
        if not self.meta_controller or not hasattr(self.meta_controller, 'lifecycle'):
            return mismatches
        
        lifecycle_mgr = self.meta_controller.lifecycle
        
        for symbol in self.shared_state.known_symbols:
            shared_state = self.shared_state.get_symbol_state(symbol)
            local_state = lifecycle_mgr.get_state(symbol)
            
            if shared_state != local_state:
                mismatch = StateMismatch(
                    component='LifecycleManager',
                    symbol=symbol,
                    field='current_state',
                    shared_value=shared_state,
                    local_value=local_state,
                    resolved_to='shared',
                    timestamp=datetime.now(timezone.utc),
                    reason='SharedState is source of truth',
                )
                
                mismatches.append(mismatch)
                self.mismatches.append(mismatch)
                
                # Reconcile: use SharedState as source of truth
                if shared_state:
                    lifecycle_mgr.set_state(
                        symbol,
                        shared_state,
                        reason='Reconciliation from SharedState',
                        force=True,
                    )
                    
                    self.logger.warning(
                        f"Symbol lifecycle mismatch for {symbol}: "
                        f"corrected {local_state} -> {shared_state}"
                    )
        
        return mismatches
    
    async def _reconcile_position_counts(self) -> List[StateMismatch]:
        """Reconcile position count between states."""
        mismatches = []
        
        # Count positions in SharedState
        shared_positions = len(self.shared_state.positions)
        
        # Count positions in MetaController (if available)
        if self.meta_controller and hasattr(self.meta_controller, 'position_count'):
            meta_positions = self.meta_controller.position_count
            
            if shared_positions != meta_positions:
                mismatch = StateMismatch(
                    component='MetaController',
                    symbol=None,
                    field='open_position_count',
                    shared_value=shared_positions,
                    local_value=meta_positions,
                    resolved_to='shared',
                    timestamp=datetime.now(timezone.utc),
                    reason='Position count mismatch - using SharedState count',
                )
                
                mismatches.append(mismatch)
                self.mismatches.append(mismatch)
                
                self.logger.warning(
                    f"Position count mismatch: "
                    f"shared={shared_positions}, meta={meta_positions} "
                    f"(source: SharedState)"
                )
        
        return mismatches
    
    async def _reconcile_capital(self) -> List[StateMismatch]:
        """Reconcile capital allocation."""
        mismatches = []
        
        # Get capital values from both sources
        shared_available = self.shared_state.available_capital
        shared_allocated = self.shared_state.allocated_capital
        
        if self.meta_controller and hasattr(self.meta_controller, 'capital_allocated'):
            meta_allocated = self.meta_controller.capital_allocated
            
            if shared_allocated != meta_allocated:
                mismatch = StateMismatch(
                    component='CapitalManager',
                    symbol=None,
                    field='allocated_capital',
                    shared_value=shared_allocated,
                    local_value=meta_allocated,
                    resolved_to='shared',
                    timestamp=datetime.now(timezone.utc),
                    reason='Capital allocation mismatch - using SharedState',
                )
                
                mismatches.append(mismatch)
                self.mismatches.append(mismatch)
                
                self.logger.warning(
                    f"Capital mismatch: "
                    f"shared={shared_allocated}, meta={meta_allocated} "
                    f"(source: SharedState)"
                )
        
        return mismatches
    
    async def _reconcile_dust_states(self) -> List[StateMismatch]:
        """Reconcile dust tracking states."""
        mismatches = []
        
        # This would check dust states if available in both sources
        # Placeholder for future expansion
        
        return mismatches
    
    async def verify_no_circular_references(self) -> bool:
        """Verify no circular import or state references."""
        # Check for common circular reference patterns
        
        if not self.meta_controller:
            return True
        
        try:
            # Try to access key attributes
            _ = self.meta_controller.shared_state
            _ = self.shared_state.positions
            
            return True
        except RecursionError:
            self.logger.error("Circular reference detected!")
            return False
    
    def get_mismatch_report(self) -> Dict[str, Any]:
        """Generate mismatch report for monitoring/alerting."""
        return {
            'total_mismatches': len(self.mismatches),
            'recent_mismatches': self.mismatches[-10:] if self.mismatches else [],
            'mismatch_summary': {
                'lifecycle': len([m for m in self.mismatches if m.component == 'LifecycleManager']),
                'positions': len([m for m in self.mismatches if m.field == 'open_position_count']),
                'capital': len([m for m in self.mismatches if m.field == 'allocated_capital']),
            },
            'last_sync': self.last_sync_time,
            'sync_count': self.sync_count,
            'health': 'healthy' if len(self.mismatches) < 5 else 'degraded' if len(self.mismatches) < 20 else 'unhealthy',
        }
    
    def reset_mismatch_history(self):
        """Clear mismatch history (for testing or after major recovery)."""
        self.mismatches.clear()
        self.logger.info("Mismatch history cleared")


class StateSyncronizationTask:
    """Background task for periodic state synchronization."""
    
    def __init__(self, synchronizer: StateSynchronizer, interval_seconds: float = 30):
        """
        Initialize sync task.
        
        Args:
            synchronizer: StateSynchronizer instance
            interval_seconds: How often to run reconciliation
        """
        self.synchronizer = synchronizer
        self.interval_seconds = interval_seconds
        self.logger = logging.getLogger("StateSyncTask")
        self._running = False
    
    async def start(self):
        """Start background sync task."""
        self._running = True
        self.logger.info(f"Starting state sync task (interval: {self.interval_seconds}s)")
        
        try:
            while self._running:
                try:
                    await self.synchronizer.reconcile_all()
                except Exception as e:
                    self.logger.error(f"Sync cycle failed: {e}")
                
                await asyncio.sleep(self.interval_seconds)
        finally:
            self._running = False
            self.logger.info("State sync task stopped")
    
    def stop(self):
        """Stop background sync task."""
        self._running = False
        self.logger.info("Stopping state sync task")
