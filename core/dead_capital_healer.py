"""
🔧 DEAD CAPITAL HEALER
======================

Ruthlessly converts dead capital back to operating cash.

This is the execution engine that liquidates dust/stale/orphaned positions
and recovers operating capital for the next cycle.

Author: Wealth Engine Architect
Date: 2026-04-17
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from core.portfolio_buckets import (
    BucketType,
    PortfolioBucketState,
    HealingEvent,
    HealingReport,
)

logger = logging.getLogger(__name__)


class DeadCapitalHealer:
    """
    Identifies and liquidates dead capital positions.
    
    This is where the wealth engine gets RUTHLESS.
    Every cycle, we identify positions that should not exist and liquidate them.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize healer with configuration.
        
        Args:
            config: Dict with configuration options
        """
        from core.portfolio_buckets import PortfolioBucketState  # Import for adaptive thresholds
        
        self.config = config or {}
        
        # 🔥 FIX 2: Use adaptive thresholds based on account size
        total_equity = self.config.get('total_equity', 500)  # Default to MICRO bracket
        thresholds = PortfolioBucketState.get_adaptive_thresholds(total_equity)
        
        # Healing thresholds
        self.min_dead_to_heal = self.config.get('min_dead_to_heal') or thresholds['min_dead_to_heal']
        self.dead_min_size = thresholds['dead_min_size']
        self.healing_urgency = thresholds['healing_urgency']
        self.batch_heal_enabled = self.config.get('batch_heal_enabled', True)     # Batch liquidations
        self.max_liquidations_per_cycle = self.config.get('max_liquidations', 10)  # Max 10 at a time
        
        # Session tracking
        self.session_healings: List[HealingEvent] = []
        self.session_recovered: float = 0.0
        self.session_id: str = f"heal_{datetime.now().isoformat()}"
        
        logger.info(f"✅ DeadCapitalHealer initialized (equity=${total_equity:.0f})")
        logger.info(f"   Min dead to heal: ${self.min_dead_to_heal:.2f}")
        logger.info(f"   Dead min size: ${self.dead_min_size:.2f}")
        logger.info(f"   Healing urgency: {self.healing_urgency}")
        logger.info(f"   Batch healing: {self.batch_heal_enabled}")
        logger.info(f"   Max liquidations/cycle: {self.max_liquidations_per_cycle}")
    
    def identify_liquidation_candidates(
        self,
        bucket_state: PortfolioBucketState,
    ) -> Tuple[List[str], float]:
        """
        Find all positions that should be liquidated.
        
        Returns:
            Tuple of (list of symbols to liquidate, total value to recover)
        """
        
        candidates = []
        total_value = 0.0
        
        logger.debug(f"🔍 Searching for liquidation candidates...")
        logger.debug(f"   Dead capital total: ${bucket_state.dead_total_value:.2f}")
        logger.debug(f"   Dead positions: {bucket_state.dead_count}")
        
        # Get liquidation priority order (largest first)
        priority_order = bucket_state.get_healing_priority_order()
        
        # Liquidate up to max per cycle
        for symbol in priority_order[:self.max_liquidations_per_cycle]:
            if symbol not in bucket_state.dead_positions:
                continue
            
            pos_data = bucket_state.dead_positions[symbol]
            value = pos_data.get('value', 0.0)
            reason = pos_data.get('reason')
            
            candidates.append(symbol)
            total_value += value
            
            logger.debug(
                f"   ✅ Candidate: {symbol:10s} | ${value:>8.2f} | Reason: {reason}"
            )
        
        logger.info(
            f"🎯 Found {len(candidates)} liquidation candidates totaling ${total_value:.2f}"
        )
        
        return candidates, total_value
    
    def create_liquidation_orders(
        self,
        candidates: List[str],
        bucket_state: PortfolioBucketState,
    ) -> List[Dict]:
        """
        Create liquidation orders for dead positions.
        
        Args:
            candidates: List of symbols to liquidate
            bucket_state: Current portfolio state
        
        Returns:
            List of liquidation orders ready to execute
        """
        
        orders = []
        
        for symbol in candidates:
            if symbol not in bucket_state.dead_positions:
                continue
            
            pos_data = bucket_state.dead_positions[symbol]
            
            order = {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': pos_data.get('qty', 0),
                'price': pos_data.get('current_price', 0),
                'expected_value': pos_data.get('value', 0),
                'reason': f"Dead capital healing ({pos_data.get('reason')})",
                'timestamp': datetime.now(),
            }
            
            orders.append(order)
            logger.debug(f"   Order: SELL {order['quantity']} {symbol} @ market")
        
        logger.info(f"📋 Created {len(orders)} liquidation orders")
        return orders
    
    def execute_liquidation_batch(
        self,
        orders: List[Dict],
        execution_callback=None,
    ) -> HealingReport:
        """
        Execute batch liquidation of dead positions.
        
        Args:
            orders: List of liquidation orders
            execution_callback: Async callback to execute orders on exchange
        
        Returns:
            HealingReport with results
        """
        
        report = HealingReport(
            session_id=self.session_id,
            timestamp=datetime.now(),
            total_positions_healed=0,
            total_amount_recovered=0.0,
        )
        
        if not orders:
            logger.info("ℹ️  No dead capital to heal this cycle")
            return report
        
        logger.info(f"🚀 Executing {len(orders)} liquidation orders...")
        
        for order in orders:
            symbol = order['symbol']
            expected_value = order['expected_value']
            
            try:
                # In real implementation, this would call exchange API
                # For now, we simulate successful execution
                if execution_callback:
                    result = execution_callback(order)
                    recovered = result.get('actual_value', expected_value)
                else:
                    # Simulate execution (assume 99% fill)
                    recovered = expected_value * 0.99
                
                # Record healing event
                event = HealingEvent(
                    symbol=symbol,
                    bucket_from=BucketType.DEAD_CAPITAL,
                    bucket_to=BucketType.OPERATING_CASH,
                    amount_recovered=recovered,
                    reason=order['reason'],
                    success=True,
                )
                
                report.healing_events.append(event)
                self.session_healings.append(event)
                
                report.total_positions_healed += 1
                report.total_amount_recovered += recovered
                self.session_recovered += recovered
                
                logger.info(
                    f"✅ Healed {symbol:10s} | Recovered: ${recovered:>8.2f}"
                )
                
            except Exception as e:
                # Record failure
                event = HealingEvent(
                    symbol=symbol,
                    bucket_from=BucketType.DEAD_CAPITAL,
                    bucket_to=BucketType.OPERATING_CASH,
                    amount_recovered=0.0,
                    reason=order['reason'],
                    success=False,
                    error_msg=str(e),
                )
                
                report.healing_events.append(event)
                report.errors.append(f"{symbol}: {str(e)}")
                
                logger.error(f"❌ Failed to heal {symbol}: {str(e)}")
        
        # Log summary
        logger.info(f"""
🎯 HEALING CYCLE COMPLETE
├─ Positions healed: {report.total_positions_healed}
├─ Amount recovered: ${report.total_amount_recovered:.2f}
├─ Success rate: {report.success_rate_pct:.0f}%
└─ Session total: ${self.session_recovered:.2f}
""")
        
        return report
    
    def should_heal(self, bucket_state: PortfolioBucketState) -> bool:
        """
        Should we prioritize healing this cycle?
        
        Args:
            bucket_state: Current portfolio state
        
        Returns:
            True if healing should be prioritized
        """
        
        # Heal if dead capital exceeds threshold
        if bucket_state.dead_total_value > self.min_dead_to_heal:
            logger.debug(
                f"💪 Healing prioritized: dead capital ${bucket_state.dead_total_value:.2f} "
                f"> threshold ${self.min_dead_to_heal:.2f}"
            )
            return True
        
        # Heal if operating cash is low
        if bucket_state.operating_cash_usdt < bucket_state.operating_cash_danger_zone:
            logger.debug(
                f"💪 Healing prioritized: operating cash low "
                f"(${bucket_state.operating_cash_usdt:.2f} < ${bucket_state.operating_cash_danger_zone:.2f})"
            )
            return True
        
        return False
    
    def get_healing_report(self) -> str:
        """Get formatted healing summary for this session"""
        
        total_recovered = self.session_recovered
        
        return f"""
🔧 DEAD CAPITAL HEALING REPORT
Session: {self.session_id}
Time: {datetime.now().isoformat()}

📊 RESULTS
├─ Total positions healed: {len(self.session_healings)}
├─ Total recovered: ${total_recovered:.2f}
└─ Success rate: {self._calculate_success_rate():.0f}%

🏆 IMPACT
├─ Operating cash recovery: ${total_recovered:.2f}
├─ Portfolio efficiency gain: {self._calculate_efficiency_gain():.1f}%
└─ Dead capital eliminated: {len([e for e in self.session_healings if e.success])} positions

📈 SESSION TIMELINE
"""
    
    def _calculate_success_rate(self) -> float:
        """Calculate % of healings that succeeded"""
        if not self.session_healings:
            return 100.0
        successful = sum(1 for e in self.session_healings if e.success)
        return (successful / len(self.session_healings)) * 100
    
    def _calculate_efficiency_gain(self) -> float:
        """Estimate portfolio efficiency gain from healing"""
        # This would need more context to calculate properly
        return 5.0  # Placeholder


class HealingOrchestrator:
    """
    Coordinates dead capital healing across multiple positions.
    
    This is the decision maker that figures out:
    1. Should we heal?
    2. How much can we safely heal?
    3. What's the execution plan?
    """
    
    def __init__(self, healer: DeadCapitalHealer, config: Optional[Dict] = None):
        """
        Initialize orchestrator.
        
        Args:
            healer: DeadCapitalHealer instance
            config: Configuration overrides
        """
        self.healer = healer
        self.config = config or {}
        
        # Decision thresholds
        self.healing_priority_threshold = self.config.get('healing_priority_threshold', 100.0)
        
        logger.info("✅ HealingOrchestrator initialized")
    
    def plan_healing_cycle(
        self,
        bucket_state: PortfolioBucketState,
    ) -> Tuple[bool, str, List[Dict]]:
        """
        Plan healing cycle and determine whether to heal.
        
        Args:
            bucket_state: Current portfolio state
        
        Returns:
            Tuple of (should_heal, reason, orders)
        """
        
        # Check if healing is needed
        should_heal = self.healer.should_heal(bucket_state)
        
        if not should_heal:
            reason = (
                f"No healing needed: dead capital ${bucket_state.dead_total_value:.2f} "
                f"< threshold ${self.healer.min_dead_to_heal:.2f}"
            )
            logger.debug(reason)
            return False, reason, []
        
        # Get candidates
        candidates, total_value = self.healer.identify_liquidation_candidates(bucket_state)
        
        if not candidates:
            reason = "No valid liquidation candidates"
            logger.info(reason)
            return False, reason, []
        
        # Create orders
        orders = self.healer.create_liquidation_orders(candidates, bucket_state)
        
        reason = f"Healing {len(candidates)} positions for ${total_value:.2f} recovery"
        logger.info(f"✅ {reason}")
        
        return True, reason, orders
