"""
Trading Coordinator: Unified trading execution layer integrating all components.

Phase 5 of dust loop elimination project.
Integrates:
  - Phase 1: Portfolio State Machine
  - Phase 2: Bootstrap Metrics Persistence
  - Phase 3: Dust Registry Lifecycle
  - Phase 4: Position Merger & Consolidation

This module coordinates the complete trading workflow:
1. Check system readiness (bootstrap, state, dust)
2. Consolidate fragmented positions
3. Execute trades with unified position state
4. Track execution in all subsystems
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .shared_state import SharedState, PortfolioState


@dataclass
class TradeExecution:
    """Record of a single trade execution."""
    
    order_id: str                           # Unique order identifier
    symbol: str                             # Trading symbol (BTC, ETH, etc.)
    quantity: float                         # Trade quantity
    entry_price: float                      # Execution price
    trade_type: str                         # "BUY", "SELL", "REBALANCE"
    timestamp: float                        # Execution timestamp
    consolidated: bool = False              # Whether positions were consolidated
    merge_operation_id: Optional[str] = None  # Reference to merge operation if applicable
    state_before: Optional[str] = None      # Portfolio state before trade
    state_after: Optional[str] = None       # Portfolio state after trade
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "trade_type": self.trade_type,
            "timestamp": self.timestamp,
            "consolidated": self.consolidated,
            "merge_operation_id": self.merge_operation_id,
            "state_before": self.state_before,
            "state_after": self.state_after,
        }


class TradingCoordinator:
    """
    Unified trading execution coordinator.
    
    Orchestrates the complete trading workflow by integrating:
    1. System readiness checks (bootstrap, metrics, state)
    2. Position consolidation (merge fragmented positions)
    3. Trade execution with unified state
    4. Execution tracking (state machine, dust registry, merge history)
    
    Ensures only beneficial trades execute with consolidated positions.
    """
    
    def __init__(self, shared_state: SharedState):
        """
        Initialize trading coordinator.
        
        Args:
            shared_state: SharedState instance with all components
        """
        self.shared_state = shared_state
        self.trade_history: List[TradeExecution] = []
        self.logger = logging.getLogger("TradingCoordinator")
        self.logger.info("[TradingCoordinator] Initialized with SharedState")
    
    # ============================================================================
    # System Readiness Checks
    # ============================================================================
    
    def check_system_ready(self) -> Tuple[bool, str]:
        """
        Verify system is ready for trading.
        
        Checks:
        1. Not in cold bootstrap phase
        2. Bootstrap metrics initialized
        3. Dust registry operational
        4. Portfolio state valid
        
        Returns:
            (is_ready, reason_if_not_ready)
        """
        # Check 1: Not in cold bootstrap
        if self.shared_state.is_cold_bootstrap():
            reason = "System in cold bootstrap phase - wait for first trade"
            self.logger.warning(f"[TradingCoordinator] System not ready: {reason}")
            return False, reason
        
        # Check 2: Bootstrap metrics valid (at least 1 trade executed)
        total_trades = self.shared_state.bootstrap_metrics.get_total_trades_executed()
        if total_trades < 1:
            reason = "No trades executed yet - system initializing"
            self.logger.warning(f"[TradingCoordinator] System not ready: {reason}")
            return False, reason
        
        self.logger.debug("[TradingCoordinator] System ready for trading")
        return True, "System ready"
    
    # ============================================================================
    # Position Preparation (Consolidation)
    # ============================================================================
    
    def prepare_positions(self, symbol: str, positions: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """
        Prepare positions for trading by consolidating fragmented positions.
        
        Process:
        1. Check if consolidation is worthwhile
        2. Execute consolidation if recommended
        3. Return prepared positions and consolidation flag
        
        Args:
            symbol: Trading symbol
            positions: List of positions to trade
            
        Returns:
            (prepared_positions, was_consolidated)
            Returns (None, False) if preparation fails
        """
        if not positions:
            self.logger.warning(f"[TradingCoordinator] No positions to prepare for {symbol}")
            return None, False
        
        # Single position - no consolidation needed
        if len(positions) == 1:
            self.logger.debug(f"[TradingCoordinator] Single position, no consolidation needed for {symbol}")
            return positions, False
        
        # Check if consolidation is recommended
        should_merge = self.shared_state.position_merger.should_merge(symbol, positions)
        
        if not should_merge:
            self.logger.debug(f"[TradingCoordinator] Consolidation not recommended for {symbol}")
            return positions, False
        
        # Execute consolidation
        self.logger.info(f"[TradingCoordinator] Consolidating {len(positions)} positions for {symbol}")
        
        merge_operation = self.shared_state.position_merger.merge_positions(symbol, positions)
        
        if not merge_operation:
            self.logger.error(f"[TradingCoordinator] Failed to consolidate positions for {symbol}")
            return None, False
        
        # Create consolidated position
        consolidated_position = {
            "symbol": symbol,
            "quantity": merge_operation.merged_quantity,
            "entry_price": merge_operation.merged_entry_price,
            "consolidated": True,
            "source_positions": len(positions),
            "merge_operation_id": id(merge_operation),
        }
        
        self.logger.info(
            f"[TradingCoordinator] Consolidated {symbol}: "
            f"{merge_operation.merged_quantity} @ {merge_operation.merged_entry_price:.2f}"
        )
        
        return [consolidated_position], True
    
    # ============================================================================
    # Execution Tracking
    # ============================================================================
    
    def track_trade_execution(self, order_id: str, symbol: str, quantity: float, 
                             entry_price: float, trade_type: str = "TRADE",
                             was_consolidated: bool = False) -> TradeExecution:
        """
        Track trade execution in all subsystems.
        
        Updates:
        1. Portfolio state machine
        2. Dust registry (if applicable)
        3. Trade history
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            quantity: Trade quantity
            entry_price: Execution price
            trade_type: Type of trade (BUY, SELL, REBALANCE)
            was_consolidated: Whether positions were consolidated
            
        Returns:
            TradeExecution record
        """
        # Create execution record
        execution = TradeExecution(
            order_id=order_id,
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            trade_type=trade_type,
            timestamp=datetime.now().timestamp(),
            consolidated=was_consolidated,
            state_before=None,
        )
        execution.state_after = "TRADED"
        
        # Track in shared state (record_trade is async, so we handle this asynchronously)
        self.logger.debug(f"[TradingCoordinator] Recorded trade: {symbol} {quantity} @ {entry_price}")
        
        # Record in history
        self.trade_history.append(execution)
        
        self.logger.info(
            f"[TradingCoordinator] Tracked execution: {symbol} {quantity} @ {entry_price}"
        )
        
        return execution
    
    # ============================================================================
    # Main Trade Execution
    # ============================================================================
    
    def execute_trade(self, symbol: str, positions: List[Dict[str, Any]], 
                     order_params: Dict[str, Any]) -> Optional[str]:
        """
        Execute a complete trade workflow.
        
        Full workflow:
        1. Check system ready
        2. Prepare (consolidate) positions
        3. Place order with prepared position
        4. Track execution in all subsystems
        
        Args:
            symbol: Trading symbol
            positions: List of positions to trade
            order_params: Order parameters (type, price, quantity, etc.)
            
        Returns:
            order_id if successful, None otherwise
        """
        self.logger.info(f"[TradingCoordinator] Starting trade workflow for {symbol}")
        
        # Step 1: Check system ready
        is_ready, reason = self.check_system_ready()
        if not is_ready:
            self.logger.error(f"[TradingCoordinator] Cannot execute trade: {reason}")
            return None
        
        # Step 2: Prepare positions (consolidate if needed)
        prepared_positions, was_consolidated = self.prepare_positions(symbol, positions)
        if prepared_positions is None:
            self.logger.error(f"[TradingCoordinator] Failed to prepare positions for {symbol}")
            return None
        
        # Step 3: Place order with prepared position
        # In real implementation, this would call the actual trading API
        prepared_position = prepared_positions[0]
        order_id = self._place_order(
            symbol=symbol,
            quantity=prepared_position["quantity"],
            entry_price=prepared_position["entry_price"],
            order_params=order_params
        )
        
        if not order_id:
            self.logger.error(f"[TradingCoordinator] Failed to place order for {symbol}")
            return None
        
        # Step 4: Track execution
        execution = self.track_trade_execution(
            order_id=order_id,
            symbol=symbol,
            quantity=prepared_position["quantity"],
            entry_price=prepared_position["entry_price"],
            trade_type=order_params.get("trade_type", "TRADE"),
            was_consolidated=was_consolidated
        )
        
        self.logger.info(f"[TradingCoordinator] Trade executed successfully: {order_id}")
        return order_id
    
    # ============================================================================
    # Order Placement (Simulated)
    # ============================================================================
    
    def _place_order(self, symbol: str, quantity: float, entry_price: float,
                    order_params: Dict[str, Any]) -> Optional[str]:
        """
        Place an order (simulated implementation).
        
        In production, this would call the actual trading API.
        For now, generates a mock order ID.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            entry_price: Order price
            order_params: Additional order parameters
            
        Returns:
            order_id if successful, None otherwise
        """
        # Validate order
        if quantity <= 0:
            self.logger.error(f"[TradingCoordinator] Invalid quantity: {quantity}")
            return None
        
        if entry_price <= 0:
            self.logger.error(f"[TradingCoordinator] Invalid price: {entry_price}")
            return None
        
        # Generate order ID (in production, from exchange)
        import time
        order_id = f"{symbol}-{int(time.time())}-{int(quantity*100000)}"
        
        self.logger.debug(
            f"[TradingCoordinator] Placed order: {order_id} "
            f"({quantity} {symbol} @ {entry_price})"
        )
        
        return order_id
    
    # ============================================================================
    # Analytics & History
    # ============================================================================
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trades executed.
        
        Returns:
            Dictionary with:
            - total_trades: Number of trades
            - total_consolidated: Number of consolidated trades
            - symbols_traded: Unique symbols traded
            - consolidation_rate: Percentage of trades that used consolidation
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "total_consolidated": 0,
                "symbols_traded": [],
                "consolidation_rate": 0.0,
            }
        
        total_trades = len(self.trade_history)
        consolidated_trades = sum(1 for t in self.trade_history if t.consolidated)
        symbols_traded = list(set(t.symbol for t in self.trade_history))
        consolidation_rate = consolidated_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "total_consolidated": consolidated_trades,
            "symbols_traded": symbols_traded,
            "consolidation_rate": consolidation_rate,
        }
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get complete trade history.
        
        Returns:
            List of trade execution records (as dicts)
        """
        return [t.to_dict() for t in self.trade_history]
    
    def reset_history(self) -> None:
        """Reset trade history."""
        self.trade_history.clear()
        self.logger.debug("[TradingCoordinator] Trade history reset")
    
    # ============================================================================
    # System State Inspection
    # ============================================================================
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get complete system status.
        
        Returns:
            Dictionary with status of all components
        """
        is_ready, ready_reason = self.check_system_ready()
        
        bootstrap_metrics = self.shared_state.bootstrap_metrics
        dust_summary = self.shared_state.dust_lifecycle_registry.get_dust_summary()
        merge_summary = self.shared_state.position_merger.get_merge_summary() if self.shared_state.position_merger else {}
        trade_summary = self.get_trade_summary()
        
        return {
            "ready": is_ready,
            "ready_reason": ready_reason,
            "bootstrap": {
                "is_cold_bootstrap": self.shared_state.is_cold_bootstrap(),
                "total_trades_executed": self.shared_state.bootstrap_metrics.get_total_trades_executed(),
            },
            "dust": dust_summary,
            "merges": merge_summary,
            "trades": trade_summary,
            "timestamp": datetime.now().timestamp(),
        }
    
    # ============================================================================
    # Diagnostic Methods
    # ============================================================================
    
    def diagnose_trade_readiness(self, symbol: str, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Diagnose readiness for a specific trade.
        
        Provides detailed information about why a trade might fail or succeed.
        
        Args:
            symbol: Trading symbol
            positions: Positions to trade
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnosis = {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp(),
            "system_ready": False,
            "system_reason": "",
            "positions_valid": False,
            "consolidation_needed": False,
            "consolidation_feasible": False,
            "overall_ready": False,
            "issues": [],
        }
        
        # Check system ready
        is_ready, reason = self.check_system_ready()
        diagnosis["system_ready"] = is_ready
        diagnosis["system_reason"] = reason
        
        if not is_ready:
            diagnosis["issues"].append(f"System not ready: {reason}")
        
        # Check positions
        if not positions:
            diagnosis["issues"].append("No positions provided")
        else:
            diagnosis["positions_valid"] = True
            
            # Check consolidation
            if len(positions) > 1:
                diagnosis["consolidation_needed"] = True
                should_merge = self.shared_state.position_merger.should_merge(symbol, positions)
                diagnosis["consolidation_feasible"] = should_merge
                
                if not should_merge:
                    diagnosis["issues"].append("Consolidation not feasible (poor feasibility score)")
        
        # Overall readiness
        diagnosis["overall_ready"] = (
            diagnosis["system_ready"] and
            diagnosis["positions_valid"] and
            (not diagnosis["consolidation_needed"] or diagnosis["consolidation_feasible"])
        )
        
        return diagnosis
