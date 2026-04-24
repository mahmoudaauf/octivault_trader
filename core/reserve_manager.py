"""
Reserve Manager - Strategic Capital Reserve Policy Engine

Enforces minimum cash reserves based on market conditions and prevents
the bot from deploying all capital, preserving flexibility for opportunities.

Core Principles:
1. Never let operating cash go to zero
2. Scale reserve based on volatility (10-35% range)
3. Block entries that would breach minimum reserve
4. Auto-liquidate positions if reserve falls below threshold
5. Emit warnings when reserve approaches critical levels
"""

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from enum import Enum


class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW = "low"              # VIX < 15, calm markets
    NORMAL = "normal"        # VIX 15-20, typical
    ELEVATED = "elevated"    # VIX 20-25, moderate stress
    HIGH = "high"            # VIX 25-30, significant stress
    EXTREME = "extreme"      # VIX > 30, crisis


class ReserveStatus(Enum):
    """Reserve health status"""
    HEALTHY = "healthy"           # Above safe level
    CAUTION = "caution"           # Approaching minimum
    WARNING = "warning"           # Below minimum, corrective action needed
    CRITICAL = "critical"         # Dangerously low, urgent liquidation


@dataclass
class ReservePolicy:
    """Reserve policy configuration for different market conditions"""
    regime: VolatilityRegime
    min_ratio: float               # Minimum reserve as % of portfolio (0.10 = 10%)
    target_ratio: float            # Target reserve as % (higher for safety)
    caution_threshold: float       # Warning threshold (% of min_ratio)
    liquidation_trigger: float     # Force liquidation if below this %
    min_absolute_usdt: float       # Absolute minimum USDT regardless of ratio
    
    def __post_init__(self):
        """Validate policy parameters"""
        if not (0 < self.min_ratio <= 1):
            raise ValueError(f"min_ratio must be 0-1, got {self.min_ratio}")
        if not (0 < self.target_ratio <= 1):
            raise ValueError(f"target_ratio must be 0-1, got {self.target_ratio}")
        if self.target_ratio < self.min_ratio:
            raise ValueError("target_ratio must be >= min_ratio")
        if not (0.5 <= self.caution_threshold <= 1.0):
            raise ValueError(f"caution_threshold must be 0.5-1.0, got {self.caution_threshold}")
        if self.liquidation_trigger >= self.min_ratio:
            raise ValueError("liquidation_trigger must be < min_ratio")


@dataclass
class ReserveSnapshot:
    """Snapshot of reserve state at a point in time"""
    timestamp: datetime
    total_portfolio_usdt: float
    free_cash_usdt: float
    reserve_ratio: float           # Current cash as % of portfolio
    regime: VolatilityRegime
    policy: ReservePolicy
    status: ReserveStatus
    required_minimum: float        # Minimum required in USDT
    buffer_available: float        # How much above minimum (can be negative)
    
    # Warnings and recommendations
    can_deploy_capital: bool
    deployment_limit: float        # Maximum can safely deploy
    actions_recommended: List[str] = field(default_factory=list)


class ReserveManager:
    """
    Manages strategic capital reserves based on market volatility.
    
    Ensures the bot always maintains flexibility for opportunities by enforcing
    minimum cash reserves that scale with market conditions.
    
    Usage:
        manager = ReserveManager(config, shared_state, logger)
        
        # Check reserve status
        snapshot = await manager.get_reserve_snapshot()
        
        # Check if can deploy capital
        can_deploy, max_amount = await manager.can_deploy_capital(amount=100.0)
        
        # Force reserve restoration if needed
        liquidation_list = await manager.force_reserve_restoration()
    """
    
    def __init__(self, config, shared_state, logger):
        """
        Initialize Reserve Manager
        
        Args:
            config: Configuration object
            shared_state: Shared system state
            logger: Logger service
        """
        
        # Define reserve policies for each volatility regime
        self.policies: Dict[VolatilityRegime, ReservePolicy] = {
            VolatilityRegime.LOW: ReservePolicy(
                regime=VolatilityRegime.LOW,
                min_ratio=0.10,           # 10% minimum
                target_ratio=0.15,        # 15% target
                caution_threshold=0.80,   # Warn at 80% of minimum
                liquidation_trigger=0.05, # Force action at 5%
                min_absolute_usdt=5.0
            ),
            VolatilityRegime.NORMAL: ReservePolicy(
                regime=VolatilityRegime.NORMAL,
                min_ratio=0.15,           # 15% minimum
                target_ratio=0.20,        # 20% target
                caution_threshold=0.75,   # Warn at 75% of minimum
                liquidation_trigger=0.08, # Force action at 8%
                min_absolute_usdt=10.0
            ),
            VolatilityRegime.ELEVATED: ReservePolicy(
                regime=VolatilityRegime.ELEVATED,
                min_ratio=0.20,           # 20% minimum
                target_ratio=0.25,        # 25% target
                caution_threshold=0.70,   # Warn at 70% of minimum
                liquidation_trigger=0.12, # Force action at 12%
                min_absolute_usdt=15.0
            ),
            VolatilityRegime.HIGH: ReservePolicy(
                regime=VolatilityRegime.HIGH,
                min_ratio=0.25,           # 25% minimum
                target_ratio=0.30,        # 30% target
                caution_threshold=0.65,   # Warn at 65% of minimum
                liquidation_trigger=0.15, # Force action at 15%
                min_absolute_usdt=20.0
            ),
            VolatilityRegime.EXTREME: ReservePolicy(
                regime=VolatilityRegime.EXTREME,
                min_ratio=0.35,           # 35% minimum (preserve capital)
                target_ratio=0.40,        # 40% target
                caution_threshold=0.60,   # Warn at 60% of minimum
                liquidation_trigger=0.20, # Force action at 20%
                min_absolute_usdt=25.0
            ),
        }
        
        # Track history for analysis
        self.reserve_history: List[ReserveSnapshot] = []
        self.max_history_length = 1000
        
        self.logger.info("ReserveManager initialized with dynamic volatility-based policies")
    
    async def get_current_volatility_regime(self) -> VolatilityRegime:
        """
        Determine current market volatility regime.
        
        In production, would connect to:
        - VIX data
        - Recent drawdown magnitude
        - Price volatility metrics
        - Market microstructure signals
        
        For now, returns NORMAL as default.
        """
        # Volatility detection logic:
        # 1. Check price volatility from recent price data (if available)
        # 2. Monitor recent portfolio drawdown
        # 3. Assess recent trade outcomes and slippage
        # 4. Use heuristic thresholds to classify regime
        
        try:
            # Placeholder: In production, would analyze:
            # - Recent price swings (ATR, standard deviation)
            # - Portfolio drawdown metrics
            # - Trade execution costs (slippage)
            
            # For now, use a simple heuristic:
            # If we have excessive realizing losses recently, escalate regime
            current_cash = await self.get_current_free_cash()
            total_nav = await self.get_total_portfolio_value()
            
            if total_nav > 0:
                cash_ratio = current_cash / total_nav
                # If cash ratio drops below 8%, signal elevated volatility perception
                if cash_ratio < 0.08:
                    return VolatilityRegime.ELEVATED
            
            # Default to normal market conditions
            return VolatilityRegime.NORMAL
        except Exception as e:
            self.logger.warning(f"Error detecting volatility regime, defaulting to NORMAL: {e}")
            return VolatilityRegime.NORMAL
    
    async def get_current_free_cash(self) -> float:
        """Get current free USDT from shared state"""
        try:
            positions = self.shared_state.positions or {}
            return float(positions.get('USDT', 0))
        except Exception as e:
            self.logger.error(f"Error getting free cash: {e}")
            return 0.0
    
    async def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio NAV in USDT"""
        try:
            total = 0.0
            positions = self.shared_state.positions or {}
            
            for symbol, amount in positions.items():
                if symbol == 'USDT':
                    total += float(amount)
                else:
                    # In production, would price positions at current market
                    # For now, assume portfolio tracking in shared_state
                    pass
            
            # Get total from shared state if available
            if hasattr(self.shared_state, 'total_portfolio_value'):
                total = float(self.shared_state.total_portfolio_value)
            
            return max(total, 10.0)  # Minimum 10 USDT for calculations
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 10.0
    
    async def get_reserve_snapshot(self) -> ReserveSnapshot:
        """
        Get complete snapshot of reserve status.
        
        Returns:
            ReserveSnapshot with current reserve analysis
        """
        timestamp = datetime.now()
        free_cash = await self.get_current_free_cash()
        total_nav = await self.get_total_portfolio_value()
        regime = await self.get_current_volatility_regime()
        policy = self.policies[regime]
        
        # Calculate reserve ratio
        reserve_ratio = free_cash / total_nav if total_nav > 0 else 0.0
        
        # Determine required minimum
        min_by_ratio = total_nav * policy.min_ratio
        required_minimum = max(min_by_ratio, policy.min_absolute_usdt)
        
        # Calculate buffer
        buffer = free_cash - required_minimum
        
        # Determine status
        if free_cash < required_minimum * policy.liquidation_trigger:
            status = ReserveStatus.CRITICAL
        elif free_cash < required_minimum:
            status = ReserveStatus.WARNING
        elif free_cash < required_minimum * policy.caution_threshold:
            status = ReserveStatus.CAUTION
        else:
            status = ReserveStatus.HEALTHY
        
        # Calculate deployment limit
        max_deployable = max(0, free_cash - required_minimum)
        
        # Generate recommendations
        actions = []
        if status == ReserveStatus.CRITICAL:
            actions.append("🔴 CRITICAL: Force liquidate positions immediately")
            actions.append(f"Need to restore {required_minimum - free_cash:.2f} USDT")
        elif status == ReserveStatus.WARNING:
            actions.append("⚠️ WARNING: Reserve below minimum threshold")
            actions.append(f"Needed: {required_minimum:.2f} USDT, Current: {free_cash:.2f} USDT")
            actions.append("Reduce position sizes or liquidate low-utility holdings")
        elif status == ReserveStatus.CAUTION:
            actions.append("⚠️ CAUTION: Reserve approaching minimum")
            actions.append(f"Buffer: {buffer:.2f} USDT (target: {required_minimum * (1 - policy.caution_threshold):.2f})")
            actions.append("Monitor closely, avoid large new entries")
        else:
            actions.append("✅ HEALTHY: Reserve within safe range")
        
        can_deploy = status in [ReserveStatus.HEALTHY, ReserveStatus.CAUTION]
        
        snapshot = ReserveSnapshot(
            timestamp=timestamp,
            total_portfolio_usdt=total_nav,
            free_cash_usdt=free_cash,
            reserve_ratio=reserve_ratio,
            regime=regime,
            policy=policy,
            status=status,
            required_minimum=required_minimum,
            buffer_available=buffer,
            can_deploy_capital=can_deploy,
            deployment_limit=max_deployable,
            actions_recommended=actions
        )
        
        # Store in history
        self._store_snapshot(snapshot)
        
        return snapshot
    
    async def can_deploy_capital(self, amount: float) -> Tuple[bool, Optional[str]]:
        """
        Check if deploying capital would violate reserve policy.
        
        Args:
            amount: Amount in USDT to deploy
            
        Returns:
            Tuple of (can_deploy: bool, reason: str or None)
        """
        snapshot = await self.get_reserve_snapshot()
        
        # Would deployment violate reserve?
        remaining_after_deployment = snapshot.free_cash_usdt - amount
        would_breach = remaining_after_deployment < snapshot.required_minimum
        
        if would_breach:
            reason = (
                f"Deployment blocked: {amount:.2f} USDT would leave "
                f"{remaining_after_deployment:.2f} USDT (need {snapshot.required_minimum:.2f})"
            )
            return False, reason
        
        return True, None
    
    async def force_reserve_restoration(self) -> List[str]:
        """
        Force restore reserve if it falls below threshold.
        
        Returns:
            List of positions that would need liquidation
            
        Note: In production, this would trigger actual liquidations.
        For now, returns recommendation list.
        """
        snapshot = await self.get_reserve_snapshot()
        
        if snapshot.status not in [ReserveStatus.WARNING, ReserveStatus.CRITICAL]:
            return []
        
        deficit = snapshot.required_minimum - snapshot.free_cash_usdt
        
        liquidation_list = [
            f"LIQUIDATE: Need {deficit:.2f} USDT to restore reserve",
            f"Current free cash: {snapshot.free_cash_usdt:.2f} USDT",
            f"Required minimum: {snapshot.required_minimum:.2f} USDT",
            f"Urgency: {snapshot.status.value}",
            "",
            "Strategy:",
            "1. Liquidate lowest-utility holdings first",
            "2. Prioritize positions with smallest slippage",
            "3. Preserve high-edge positions for continuation",
            "4. Use partial fills if available"
        ]
        
        return liquidation_list
    
    async def adjust_reserve_for_volatility(self, new_regime: Optional[VolatilityRegime] = None):
        """
        Adjust reserve requirements based on volatility regime.
        
        If volatility increases, may need to force additional liquidations.
        If volatility decreases, can relax constraints.
        
        Args:
            new_regime: New regime, or None to auto-detect
        """
        if new_regime is None:
            new_regime = await self.get_current_volatility_regime()
        
        old_snapshot = await self.get_reserve_snapshot()
        old_regime = old_snapshot.regime
        
        if new_regime == old_regime:
            return  # No change needed
        
        new_policy = self.policies[new_regime]
        total_nav = old_snapshot.total_portfolio_usdt
        free_cash = old_snapshot.free_cash_usdt
        
        new_required = max(
            total_nav * new_policy.min_ratio,
            new_policy.min_absolute_usdt
        )
        old_required = old_snapshot.required_minimum
        
        change = new_required - old_required
        
        message = (
            f"Reserve adjustment: {old_regime.value} → {new_regime.value}\n"
            f"Required reserve: {old_required:.2f} → {new_required:.2f} USDT (+{change:.2f})"
        )
        
        if change > 0 and free_cash < new_required:
            message += (
                f"\n⚠️ Volatility increase requires additional reserve!\n"
                f"Need to liquidate {new_required - free_cash:.2f} USDT worth of positions"
            )
        elif change < 0:
            message += f"\n✅ Volatility decrease allows more deployment flexibility"
        
        self.logger.info(message)
    
    def _store_snapshot(self, snapshot: ReserveSnapshot):
        """Store snapshot in history with size management"""
        self.reserve_history.append(snapshot)
        
        # Trim history if too large
        if len(self.reserve_history) > self.max_history_length:
            self.reserve_history = self.reserve_history[-self.max_history_length:]
    
    async def get_reserve_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from reserve history.
        
        Returns:
            Dictionary with reserve metrics over time
        """
        if not self.reserve_history:
            return {}
        
        ratios = [s.reserve_ratio for s in self.reserve_history]
        
        return {
            "samples": len(self.reserve_history),
            "current_ratio": ratios[-1] if ratios else 0,
            "avg_ratio": sum(ratios) / len(ratios) if ratios else 0,
            "min_ratio": min(ratios) if ratios else 0,
            "max_ratio": max(ratios) if ratios else 0,
            "regime_transitions": self._count_regime_transitions(),
            "critical_events": sum(1 for s in self.reserve_history if s.status == ReserveStatus.CRITICAL),
            "warning_events": sum(1 for s in self.reserve_history if s.status == ReserveStatus.WARNING),
        }
    
    def _count_regime_transitions(self) -> int:
        """Count how many times volatility regime changed"""
        if len(self.reserve_history) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(self.reserve_history)):
            if self.reserve_history[i].regime != self.reserve_history[i-1].regime:
                transitions += 1
        return transitions
    
    async def print_reserve_status(self):
        """Pretty-print reserve status to logger"""
        snapshot = await self.get_reserve_snapshot()
        
        status_color = {
            ReserveStatus.HEALTHY: "✅",
            ReserveStatus.CAUTION: "⚠️",
            ReserveStatus.WARNING: "🔴",
            ReserveStatus.CRITICAL: "🔴🔴"
        }
        
        print("\n" + "="*80)
        print("💰 RESERVE MANAGER STATUS")
        print("="*80)
        print(f"Status: {status_color[snapshot.status]} {snapshot.status.value.upper()}")
        print(f"Volatility Regime: {snapshot.regime.value}")
        print()
        print(f"Free Cash:           {snapshot.free_cash_usdt:>10.2f} USDT")
        print(f"Total Portfolio:     {snapshot.total_portfolio_usdt:>10.2f} USDT")
        print(f"Reserve Ratio:       {snapshot.reserve_ratio*100:>9.1f}%")
        print(f"Required Minimum:    {snapshot.required_minimum:>10.2f} USDT ({snapshot.policy.min_ratio*100:.0f}%)")
        print(f"Buffer Available:    {snapshot.buffer_available:>10.2f} USDT")
        print(f"Max Deployable:      {snapshot.deployment_limit:>10.2f} USDT")
        print()
        print("Actions:")
        for action in snapshot.actions_recommended:
            print(f"  {action}")
        print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

async def test_reserve_manager():
    """Test reserve manager with mock config"""
    
    class MockConfig:
        def __init__(self):
            self.logging_level = "INFO"
    
    class MockSharedState:
        def __init__(self):
            self.positions = {'USDT': 100.0, 'BTC': 0.001}
            self.total_portfolio_value = 1000.0
    
    class MockLogger:
        def info(self, msg):
            print(f"[INFO] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")
    
    config = MockConfig()
    shared_state = MockSharedState()
    logger = MockLogger()
    
    manager = ReserveManager(config, shared_state, logger)
    
    # Test 1: Get reserve snapshot
    print("\n" + "="*80)
    print("TEST 1: Get Reserve Snapshot")
    print("="*80)
    snapshot = await manager.get_reserve_snapshot()
    print(f"Free Cash: {snapshot.free_cash_usdt} USDT")
    print(f"Reserve Ratio: {snapshot.reserve_ratio*100:.1f}%")
    print(f"Status: {snapshot.status.value}")
    print(f"Can Deploy: {snapshot.can_deploy_capital}")
    
    # Test 2: Check deployment capability
    print("\n" + "="*80)
    print("TEST 2: Check Deployment Capability")
    print("="*80)
    can_deploy, reason = await manager.can_deploy_capital(50.0)
    print(f"Can deploy 50.0 USDT: {can_deploy}")
    if reason:
        print(f"Reason: {reason}")
    
    # Test 3: Print status
    print("\n" + "="*80)
    print("TEST 3: Print Status")
    print("="*80)
    await manager.print_reserve_status()
    
    print("✅ Reserve Manager tests completed")


if __name__ == "__main__":
    asyncio.run(test_reserve_manager())
