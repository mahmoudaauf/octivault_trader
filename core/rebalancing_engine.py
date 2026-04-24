# =============================
# core/rebalancing_engine.py — Professional Portfolio Rebalancing
# =============================
"""
Advanced Rebalancing Engine for Octivault Trader

Phase 6 Implementation: Professional portfolio rebalancing with multiple strategies.

This module provides intelligent portfolio rebalancing to:
1. Maintain target allocation across asset classes
2. Rebalance on schedule or trigger-based events
3. Support multiple rebalancing strategies
4. Minimize trading costs through smart order execution
5. Enforce risk constraints during rebalancing
6. Provide comprehensive audit trail

Key Components:
- RebalancingEngine: Main rebalancing orchestrator
- AllocationTarget: Target portfolio allocation
- RebalanceStrategy: Abstract base for strategies
- EqualWeightRebalancer: Equal-weight portfolio strategy
- RiskParityRebalancer: Risk-parity allocation strategy
- DriftThresholdRebalancer: Rebalance on allocation drift
- RebalanceMetrics: Performance tracking
- RebalanceScheduler: Automatic periodic rebalancing

Thread-Safe: All operations designed for async/concurrent execution
Fail-Safe: Validates constraints before execution
Efficient: Minimizes transaction costs through smart planning
"""

from __future__ import annotations

import logging
import asyncio
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import time
from decimal import Decimal

logger = logging.getLogger(__name__)


# =============================
# Enums and Data Structures
# =============================

class RebalanceStrategy(Enum):
    """Available rebalancing strategies."""
    EQUAL_WEIGHT = "EQUAL_WEIGHT"           # Equal weight all positions
    RISK_PARITY = "RISK_PARITY"             # Equal risk contribution
    DRIFT_THRESHOLD = "DRIFT_THRESHOLD"     # Rebalance on drift > threshold
    MOMENTUM = "MOMENTUM"                   # Weight by momentum signals
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"  # Adjust for volatility
    CUSTOM = "CUSTOM"                       # Custom allocation


class RebalanceStatus(Enum):
    """Status of rebalance operation."""
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class RebalanceTrigger(Enum):
    """What triggers a rebalance."""
    SCHEDULED = "SCHEDULED"           # On schedule (e.g., daily)
    DRIFT_THRESHOLD = "DRIFT_THRESHOLD"  # Allocation drifts > threshold
    MANUAL = "MANUAL"                 # Manual request
    MARKET_EVENT = "MARKET_EVENT"     # Market event triggered
    RISK_LIMIT = "RISK_LIMIT"         # Risk limit breached


@dataclass
class AllocationTarget:
    """Target allocation for portfolio."""
    symbol: str
    target_weight: float  # 0.0 to 1.0 (as percentage of portfolio)
    min_weight: float = 0.0
    max_weight: float = 1.0
    priority: int = 0  # Higher = higher priority
    rebalance_allowed: bool = True  # Can be rebalanced?
    
    def validate(self) -> bool:
        """Validate allocation target."""
        if not (0.0 <= self.target_weight <= 1.0):
            return False
        if not (0.0 <= self.min_weight <= 1.0):
            return False
        if not (0.0 <= self.max_weight <= 1.0):
            return False
        if self.min_weight > self.target_weight or self.target_weight > self.max_weight:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PortfolioState:
    """Current state of portfolio."""
    timestamp: float
    total_value: float  # In base currency (USDT)
    positions: Dict[str, float]  # symbol -> quantity
    prices: Dict[str, float]  # symbol -> current price
    values: Dict[str, float]  # symbol -> position value in USDT
    weights: Dict[str, float]  # symbol -> weight (0.0 to 1.0)
    allocation_drift: Dict[str, float] = field(default_factory=dict)  # symbol -> drift from target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RebalanceOrder:
    """Single rebalance order."""
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    target_price: float
    reason: str
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RebalancePlan:
    """Plan for rebalancing portfolio."""
    timestamp: float
    trigger: RebalanceTrigger
    strategy: RebalanceStrategy
    portfolio_state: PortfolioState
    target_allocation: Dict[str, AllocationTarget]
    rebalance_orders: List[RebalanceOrder] = field(default_factory=list)
    total_sell_value: float = 0.0
    total_buy_value: float = 0.0
    estimated_fee_cost: float = 0.0
    expected_cost_reduction: float = 0.0
    estimated_concentration_reduction: float = 0.0
    rebalance_status: RebalanceStatus = RebalanceStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    approval_timestamp: Optional[float] = None
    execution_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["trigger"] = self.trigger.value
        d["strategy"] = self.strategy.value
        d["rebalance_status"] = self.rebalance_status.value
        d["portfolio_state"] = self.portfolio_state.to_dict()
        d["target_allocation"] = {
            k: v.to_dict() for k, v in self.target_allocation.items()
        }
        d["rebalance_orders"] = [o.to_dict() for o in self.rebalance_orders]
        return d
    
    @property
    def is_approved(self) -> bool:
        """Check if plan is approved."""
        return self.rebalance_status in (
            RebalanceStatus.APPROVED,
            RebalanceStatus.EXECUTING,
            RebalanceStatus.COMPLETED
        )
    
    @property
    def is_valid(self) -> bool:
        """Check if plan is valid."""
        return len(self.validation_errors) == 0


@dataclass
class RebalanceMetrics:
    """Metrics for rebalancing."""
    total_rebalances: int = 0
    successful_rebalances: int = 0
    failed_rebalances: int = 0
    total_concentration_reduction: float = 0.0
    total_cost_savings: float = 0.0
    total_fee_cost: float = 0.0
    average_execution_time_sec: float = 0.0
    last_rebalance_timestamp: Optional[float] = None
    last_rebalance_symbol: Optional[str] = None
    last_error: Optional[str] = None
    rebalances_by_strategy: Dict[str, int] = field(default_factory=dict)
    rebalances_by_trigger: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================
# Rebalancing Engine
# =============================

class RebalancingEngine:
    """
    Professional portfolio rebalancing engine.
    
    Supports multiple strategies with intelligent order generation,
    comprehensive validation, and detailed tracking.
    """
    
    def __init__(self, shared_state=None, exchange_client=None, config=None, meta_controller=None):
        """
        Initialize rebalancing engine.
        
        Args:
            shared_state: SharedState instance
            exchange_client: ExchangeClient instance
            config: Configuration object
            meta_controller: MetaController for order submission
        """
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config
        self.meta_controller = meta_controller
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.rebalance_interval_sec = 86400  # Daily
        self.min_rebalance_drift = 0.10  # 10% threshold
        self.max_transaction_cost_pct = 0.005  # 0.5% of portfolio
        self.enable_risk_limits = True
        self.max_concentration_ratio = 0.40  # Max weight for single position
        self.min_order_value_usd = 20.0  # Minimum order value
        
        # Operation tracking
        self.rebalance_plans: Dict[str, RebalancePlan] = {}
        self.rebalance_history: List[RebalancePlan] = []
        self.metrics = RebalanceMetrics()
        self._lock = asyncio.Lock()
        self._last_rebalance_time = 0.0
        
        # Target allocations
        self.target_allocation: Dict[str, AllocationTarget] = {}
        
        # Configure from config
        if config:
            self.rebalance_interval_sec = getattr(config, "REBALANCE_INTERVAL_SEC", 86400)
            self.min_rebalance_drift = getattr(config, "MIN_REBALANCE_DRIFT", 0.10)
            self.max_concentration_ratio = getattr(config, "MAX_CONCENTRATION_RATIO", 0.40)
    
    def set_target_allocation(self, allocation: Dict[str, AllocationTarget]) -> bool:
        """
        Set target portfolio allocation.
        
        Args:
            allocation: Dict mapping symbol -> AllocationTarget
            
        Returns:
            True if allocation is valid, False otherwise
        """
        # Validate all targets
        for symbol, target in allocation.items():
            if not target.validate():
                self.logger.error(f"[RebalancingEngine] Invalid allocation target for {symbol}")
                return False
        
        # Validate total weight
        total_weight = sum(t.target_weight for t in allocation.values())
        if not (0.95 <= total_weight <= 1.05):  # Allow 5% variance
            self.logger.error(f"[RebalancingEngine] Total allocation weight {total_weight:.2%} not ~100%")
            return False
        
        self.target_allocation = allocation
        self.logger.info(f"[RebalancingEngine] Target allocation set for {len(allocation)} symbols")
        return True
    
    async def get_portfolio_state(self, positions: Dict[str, Any], prices: Dict[str, float]) -> PortfolioState:
        """
        Get current portfolio state.
        
        Args:
            positions: Current positions
            prices: Current prices
            
        Returns:
            Portfolio state snapshot
        """
        values = {}
        weights = {}
        total_value = 0.0
        
        # Calculate position values
        for symbol, pos_data in positions.items():
            qty = pos_data.get("quantity", 0)
            price = prices.get(symbol, 0)
            
            if qty > 0 and price > 0:
                value = qty * price
                values[symbol] = value
                total_value += value
        
        # Calculate weights
        if total_value > 0:
            weights = {
                symbol: value / total_value
                for symbol, value in values.items()
            }
        
        # Calculate allocation drift
        allocation_drift = {}
        for symbol, target in self.target_allocation.items():
            current_weight = weights.get(symbol, 0.0)
            drift = current_weight - target.target_weight
            allocation_drift[symbol] = drift
        
        return PortfolioState(
            timestamp=time.time(),
            total_value=total_value,
            positions={s: pos_data.get("quantity", 0) for s, pos_data in positions.items()},
            prices=prices,
            values=values,
            weights=weights,
            allocation_drift=allocation_drift,
        )
    
    def calculate_allocation_drift(self, portfolio_state: PortfolioState) -> float:
        """
        Calculate total allocation drift.
        
        Drift = sum of absolute deviations from target weights
        """
        total_drift = 0.0
        
        for symbol, target in self.target_allocation.items():
            current_weight = portfolio_state.weights.get(symbol, 0.0)
            drift = abs(current_weight - target.target_weight)
            total_drift += drift
        
        return total_drift / 2.0  # Divide by 2 because each misalignment counts twice
    
    def should_rebalance_by_drift(self, portfolio_state: PortfolioState) -> bool:
        """
        Check if portfolio should be rebalanced based on drift.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            True if drift > threshold
        """
        drift = self.calculate_allocation_drift(portfolio_state)
        return drift > self.min_rebalance_drift
    
    async def plan_equal_weight_rebalance(self, portfolio_state: PortfolioState, 
                                         trigger: RebalanceTrigger) -> Optional[RebalancePlan]:
        """
        Plan equal-weight rebalancing.
        
        Args:
            portfolio_state: Current portfolio state
            trigger: What triggered rebalance
            
        Returns:
            Rebalance plan, or None if not needed
        """
        plan = RebalancePlan(
            timestamp=time.time(),
            trigger=trigger,
            strategy=RebalanceStrategy.EQUAL_WEIGHT,
            portfolio_state=portfolio_state,
            target_allocation=self.target_allocation,
        )
        
        try:
            # Set equal weights
            n_symbols = len(self.target_allocation)
            equal_weight = 1.0 / n_symbols if n_symbols > 0 else 0.0
            
            # Generate orders
            target_values = {}
            for symbol in self.target_allocation:
                target_value = portfolio_state.total_value * equal_weight
                target_values[symbol] = target_value
            
            await self._generate_rebalance_orders(plan, target_values, portfolio_state)
            
            # Validate plan
            await self._validate_plan(plan)
            
            return plan
        
        except Exception as e:
            plan.validation_errors.append(str(e))
            return plan
    
    async def plan_risk_parity_rebalance(self, portfolio_state: PortfolioState,
                                        volatilities: Dict[str, float],
                                        trigger: RebalanceTrigger) -> Optional[RebalancePlan]:
        """
        Plan risk-parity rebalancing.
        
        Args:
            portfolio_state: Current portfolio state
            volatilities: Symbol -> volatility mapping
            trigger: What triggered rebalance
            
        Returns:
            Rebalance plan, or None if not needed
        """
        plan = RebalancePlan(
            timestamp=time.time(),
            trigger=trigger,
            strategy=RebalanceStrategy.RISK_PARITY,
            portfolio_state=portfolio_state,
            target_allocation=self.target_allocation,
        )
        
        try:
            # Calculate weights inversely proportional to volatility
            inv_vols = {}
            for symbol in self.target_allocation:
                vol = volatilities.get(symbol, 0.20)  # Default 20% volatility
                if vol > 0:
                    inv_vols[symbol] = 1.0 / vol
                else:
                    inv_vols[symbol] = 1.0
            
            # Normalize to sum to 1
            total_inv_vol = sum(inv_vols.values())
            if total_inv_vol > 0:
                weights = {s: w / total_inv_vol for s, w in inv_vols.items()}
            else:
                weights = {s: 1.0 / len(inv_vols) for s in inv_vols}
            
            # Generate orders
            target_values = {
                symbol: portfolio_state.total_value * weights.get(symbol, 0.0)
                for symbol in self.target_allocation
            }
            
            await self._generate_rebalance_orders(plan, target_values, portfolio_state)
            
            # Validate plan
            await self._validate_plan(plan)
            
            return plan
        
        except Exception as e:
            plan.validation_errors.append(str(e))
            return plan
    
    async def plan_drift_threshold_rebalance(self, portfolio_state: PortfolioState,
                                            trigger: RebalanceTrigger) -> Optional[RebalancePlan]:
        """
        Plan rebalancing based on drift threshold.
        
        Only rebalances positions that have drifted beyond threshold.
        
        Args:
            portfolio_state: Current portfolio state
            trigger: What triggered rebalance
            
        Returns:
            Rebalance plan, or None if not needed
        """
        plan = RebalancePlan(
            timestamp=time.time(),
            trigger=trigger,
            strategy=RebalanceStrategy.DRIFT_THRESHOLD,
            portfolio_state=portfolio_state,
            target_allocation=self.target_allocation,
        )
        
        try:
            # Calculate target values
            target_values = {}
            for symbol, target in self.target_allocation.items():
                current_weight = portfolio_state.weights.get(symbol, 0.0)
                drift = abs(current_weight - target.target_weight)
                
                # Only rebalance if drifted
                if drift > self.min_rebalance_drift:
                    target_values[symbol] = portfolio_state.total_value * target.target_weight
                else:
                    # Keep current value
                    target_values[symbol] = portfolio_state.values.get(symbol, 0.0)
            
            await self._generate_rebalance_orders(plan, target_values, portfolio_state)
            
            # Validate plan
            await self._validate_plan(plan)
            
            return plan
        
        except Exception as e:
            plan.validation_errors.append(str(e))
            return plan
    
    async def _generate_rebalance_orders(self, plan: RebalancePlan, target_values: Dict[str, float],
                                         portfolio_state: PortfolioState) -> None:
        """Generate rebalance orders from target values."""
        plan.rebalance_orders = []
        plan.total_sell_value = 0.0
        plan.total_buy_value = 0.0
        
        for symbol, target_value in target_values.items():
            current_value = portfolio_state.values.get(symbol, 0.0)
            diff = target_value - current_value
            
            if diff == 0:
                continue
            
            price = portfolio_state.prices.get(symbol, 0)
            if price <= 0:
                continue
            
            # Calculate quantity
            if diff > 0:
                # BUY order
                quantity = diff / price
                if diff >= self.min_order_value_usd:
                    order = RebalanceOrder(
                        symbol=symbol,
                        side="BUY",
                        quantity=quantity,
                        target_price=price,
                        reason="Rebalance: below target",
                    )
                    plan.rebalance_orders.append(order)
                    plan.total_buy_value += diff
            else:
                # SELL order
                quantity = abs(diff) / price
                if abs(diff) >= self.min_order_value_usd:
                    order = RebalanceOrder(
                        symbol=symbol,
                        side="SELL",
                        quantity=quantity,
                        target_price=price,
                        reason="Rebalance: above target",
                    )
                    plan.rebalance_orders.append(order)
                    plan.total_sell_value += abs(diff)
        
        # Sort orders by priority and value
        plan.rebalance_orders.sort(
            key=lambda o: (
                -self.target_allocation.get(o.symbol, AllocationTarget("", 0)).priority,
                abs(o.quantity * o.target_price)
            ),
            reverse=True
        )
    
    async def _validate_plan(self, plan: RebalancePlan) -> None:
        """Validate rebalance plan."""
        plan.rebalance_status = RebalanceStatus.PLANNING
        plan.validation_errors = []
        
        # Check for orders
        if not plan.rebalance_orders:
            plan.validation_errors.append("No rebalance orders generated")
        
        # Check transaction costs
        total_notional = plan.total_sell_value + plan.total_buy_value
        if total_notional > 0:
            # Assume 0.1% taker fee for each order
            fee_cost = total_notional * 0.001
            plan.estimated_fee_cost = fee_cost
            
            cost_pct = fee_cost / plan.portfolio_state.total_value
            if cost_pct > self.max_transaction_cost_pct:
                plan.validation_errors.append(
                    f"Transaction cost {cost_pct:.2%} exceeds max {self.max_transaction_cost_pct:.2%}"
                )
        
        # Check concentration limits
        if self.enable_risk_limits:
            for symbol, target in self.target_allocation.items():
                if target.target_weight > self.max_concentration_ratio:
                    plan.validation_errors.append(
                        f"Target weight {target.target_weight:.2%} exceeds concentration limit"
                    )
        
        # Set status
        if plan.is_valid:
            plan.rebalance_status = RebalanceStatus.APPROVED
            plan.approval_timestamp = time.time()
        else:
            plan.rebalance_status = RebalanceStatus.REJECTED
    
    async def _execute_rebalancing_orders(self, plan: RebalancePlan) -> bool:
        """
        Execute rebalancing orders via the execution manager.
        
        This implements the order submission, tracking, and verification.
        
        Args:
            plan: Rebalance plan with orders to execute
            
        Returns:
            bool: True if execution succeeded, False otherwise
        """
        try:
            if not self.execution_manager:
                self.logger.error("[RebalancingEngine] No execution manager available")
                return False
            
            if not plan.rebalance_orders:
                self.logger.warning(f"[RebalancingEngine] No orders in plan {plan.plan_id}")
                return True  # Empty plan is technically successful
            
            # Submit orders to the execution manager
            submitted_orders = []
            for order in plan.rebalance_orders:
                try:
                    # Use execution manager to submit order
                    result = await self.execution_manager.submit_order(
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.order_type or "MARKET",
                        quantity=order.quantity,
                        price=order.price,
                        client_order_id=order.order_id
                    )
                    
                    if result:
                        submitted_orders.append(order.order_id)
                        self.logger.debug(f"[RebalancingEngine] Submitted order {order.order_id}: {order.symbol} {order.side}")
                    else:
                        self.logger.warning(f"[RebalancingEngine] Failed to submit order {order.order_id}")
                        return False
                        
                except Exception as e:
                    self.logger.error(f"[RebalancingEngine] Error submitting order {order.order_id}: {e}")
                    return False
            
            self.logger.info(f"[RebalancingEngine] Successfully submitted {len(submitted_orders)} orders for plan {plan.plan_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"[RebalancingEngine] Error executing rebalancing orders: {e}", exc_info=True)
            return False
    
    async def execute_rebalance(self, plan: RebalancePlan) -> bool:
        """
        Execute rebalance plan.
        
        Args:
            plan: Rebalance plan to execute
            
        Returns:
            True if successful
        """
        if not plan.is_approved:
            self.logger.warning("[RebalancingEngine] Cannot execute unapproved plan")
            return False
        
        try:
            async with self._lock:
                plan.rebalance_status = RebalanceStatus.EXECUTING
                
                # Execute rebalancing orders
                execution_success = await self._execute_rebalancing_orders(plan)
                
                if execution_success:
                    plan.rebalance_status = RebalanceStatus.COMPLETED
                    plan.execution_timestamp = time.time()
                    
                    # Update metrics
                    self.metrics.total_rebalances += 1
                    self.metrics.successful_rebalances += 1
                    self.metrics.last_rebalance_timestamp = time.time()
                    self.metrics.total_fee_cost += plan.estimated_fee_cost
                    
                    strategy_key = plan.strategy.value
                    self.metrics.rebalances_by_strategy[strategy_key] = (
                        self.metrics.rebalances_by_strategy.get(strategy_key, 0) + 1
                    )
                else:
                    plan.rebalance_status = RebalanceStatus.FAILED
                    self.logger.error(f"[RebalancingEngine] Rebalance execution failed: {plan.plan_id}")
                    return False
                
                trigger_key = plan.trigger.value
                self.metrics.rebalances_by_trigger[trigger_key] = (
                    self.metrics.rebalances_by_trigger.get(trigger_key, 0) + 1
                )
                
                self._last_rebalance_time = time.time()
                
                self.logger.info(f"[RebalancingEngine] Executed {plan.strategy.value} rebalance "
                               f"with {len(plan.rebalance_orders)} orders")
                
                # Store in history
                self.rebalance_history.append(plan)
                
                return True
        
        except Exception as e:
            plan.rebalance_status = RebalanceStatus.FAILED
            self.metrics.failed_rebalances += 1
            self.metrics.last_error = str(e)
            self.logger.error(f"[RebalancingEngine] Execution failed: {e}")
            return False
    
    async def auto_rebalance(self, positions: Dict[str, Any], prices: Dict[str, float],
                            strategy: RebalanceStrategy = RebalanceStrategy.DRIFT_THRESHOLD) -> Optional[RebalancePlan]:
        """
        Automatically rebalance portfolio if needed.
        
        Args:
            positions: Current positions
            prices: Current prices
            strategy: Rebalancing strategy to use
            
        Returns:
            Executed plan or None
        """
        # Check if enough time has passed
        time_since_rebalance = time.time() - self._last_rebalance_time
        if time_since_rebalance < self.rebalance_interval_sec:
            return None
        
        # Get portfolio state
        portfolio_state = await self.get_portfolio_state(positions, prices)
        
        # Check drift
        if not self.should_rebalance_by_drift(portfolio_state):
            return None
        
        # Plan rebalance
        plan = None
        if strategy == RebalanceStrategy.EQUAL_WEIGHT:
            plan = await self.plan_equal_weight_rebalance(
                portfolio_state, RebalanceTrigger.DRIFT_THRESHOLD
            )
        elif strategy == RebalanceStrategy.DRIFT_THRESHOLD:
            plan = await self.plan_drift_threshold_rebalance(
                portfolio_state, RebalanceTrigger.DRIFT_THRESHOLD
            )
        
        if plan and plan.is_approved:
            await self.execute_rebalance(plan)
            return plan
        
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get rebalancing summary."""
        return {
            "total_rebalances": self.metrics.total_rebalances,
            "successful_rebalances": self.metrics.successful_rebalances,
            "failed_rebalances": self.metrics.failed_rebalances,
            "success_rate": (
                self.metrics.successful_rebalances / self.metrics.total_rebalances
                if self.metrics.total_rebalances > 0 else 0.0
            ),
            "total_fee_cost": self.metrics.total_fee_cost,
            "last_rebalance": self.metrics.last_rebalance_timestamp,
            "target_allocation": {
                s: t.to_dict() for s, t in self.target_allocation.items()
            },
            "last_error": self.metrics.last_error,
        }
