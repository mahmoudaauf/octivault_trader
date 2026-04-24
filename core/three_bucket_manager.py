"""
⚡ THREE-BUCKET INTEGRATION MODULE
===================================

Integration point for three-bucket portfolio management in the main orchestrator.

This module provides the API for:
1. Classifying portfolio into buckets
2. Healing dead capital
3. Enforcing bucket-based decision gates
4. Reporting on bucket health

Author: Wealth Engine Architect
Date: 2026-04-17
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from core.portfolio_buckets import PortfolioBucketState, BucketMetrics
from core.bucket_classifier import BucketClassifier
from core.dead_capital_healer import DeadCapitalHealer, HealingOrchestrator

logger = logging.getLogger(__name__)


class ThreeBucketPortfolioManager:
    """
    Main interface for three-bucket portfolio management.
    
    This is what the orchestrator calls to:
    1. Classify portfolio state
    2. Make decisions based on buckets
    3. Execute healing
    4. Report metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize three-bucket manager.
        
        Args:
            config: Configuration dict with threshold overrides
        """
        self.config = config or {}
        
        # Initialize components
        self.classifier = BucketClassifier(config)
        self.healer = DeadCapitalHealer(config)
        self.healing_orchestrator = HealingOrchestrator(self.healer, config)
        
        # State tracking
        self.current_bucket_state: Optional[PortfolioBucketState] = None
        self.last_healing_report = None
        
        logger.info("✅ ThreeBucketPortfolioManager initialized")
    
    def update_bucket_state(
        self,
        positions: Dict[str, Dict],
        total_equity: float,
    ) -> PortfolioBucketState:
        """
        Update the current portfolio bucket classification.
        
        This should be called at the start of each trading cycle.
        
        Args:
            positions: Dict of all positions (from exchange/state)
            total_equity: Total portfolio value
        
        Returns:
            Updated PortfolioBucketState
        """
        
        logger.debug(f"🔄 Updating bucket classification...")
        
        # Classify portfolio
        self.current_bucket_state = self.classifier.classify_portfolio(
            positions=positions,
            total_equity=total_equity,
        )
        
        return self.current_bucket_state
    
    def should_execute_healing(self) -> bool:
        """
        Check if dead capital healing should be executed this cycle.
        
        Returns:
            True if healing should proceed
        """
        
        if not self.current_bucket_state:
            return False
        
        return self.healer.should_heal(self.current_bucket_state)
    
    def plan_healing_cycle(self) -> Tuple[bool, str, list]:
        """
        Plan and return healing decisions.
        
        Returns:
            Tuple of (should_heal, reason, liquidation_orders)
        """
        
        if not self.current_bucket_state:
            return False, "No bucket state", []
        
        should_heal, reason, orders = self.healing_orchestrator.plan_healing_cycle(
            self.current_bucket_state
        )
        
        return should_heal, reason, orders
    
    def execute_healing(self, execution_callback=None):
        """
        Execute the healing cycle.
        
        Args:
            execution_callback: Callback to execute orders
        
        Returns:
            HealingReport with results
        """
        
        if not self.current_bucket_state:
            logger.warning("Cannot heal: no bucket state")
            return None
        
        # Get candidates
        candidates, total_value = self.healer.identify_liquidation_candidates(
            self.current_bucket_state
        )
        
        if not candidates:
            logger.info("No dead capital to heal")
            return None
        
        # Create orders
        orders = self.healer.create_liquidation_orders(candidates, self.current_bucket_state)
        
        # Execute
        report = self.healer.execute_liquidation_batch(orders, execution_callback)
        self.last_healing_report = report
        
        return report
    
    def can_trade_new_position(self) -> bool:
        """
        Check if it's safe to enter a new trading position.
        
        Returns:
            True if portfolio is healthy enough to trade
        """
        
        if not self.current_bucket_state:
            return False
        
        return self.current_bucket_state.can_trade_new_position()
    
    def get_trading_decision_gates(self) -> Dict[str, Tuple[bool, str]]:
        """
        Get bucket-based trading decision gates.
        
        Each gate returns (pass, reason).
        
        Returns:
            Dict of {gate_name: (pass, reason)}
        """
        
        if not self.current_bucket_state:
            return {
                'bucket_available': (False, 'No bucket state'),
                'operating_cash_healthy': (False, 'No bucket state'),
                'dead_capital_manageable': (False, 'No bucket state'),
                'portfolio_not_full': (False, 'No bucket state'),
            }
        
        state = self.current_bucket_state
        gates = {}
        
        # Gate 1: Operating cash health
        operating_cash_ok = state.is_operating_cash_healthy()
        gates['operating_cash_healthy'] = (
            operating_cash_ok,
            f"Operating cash: ${state.operating_cash_usdt:.2f} "
            f"{'✅ HEALTHY' if operating_cash_ok else '⚠️ LOW'}"
        )
        
        # Gate 2: Operating cash not critical
        not_critical = not state.is_operating_cash_critical()
        gates['operating_cash_not_critical'] = (
            not_critical,
            f"Operating cash not critical: ${state.operating_cash_usdt:.2f} "
            f"{'✅' if not_critical else '❌'}"
        )
        
        # Gate 3: Dead capital manageable
        dead_manageable = state.dead_total_value < state.operating_cash_usdt * 0.5
        gates['dead_capital_manageable'] = (
            dead_manageable,
            f"Dead capital: ${state.dead_total_value:.2f} "
            f"{'✅ MANAGEABLE' if dead_manageable else '⚠️ EXCESSIVE'}"
        )
        
        # Gate 4: Portfolio not full
        not_full = state.productive_count < state.productive_max_count
        gates['portfolio_not_full'] = (
            not_full,
            f"Portfolio: {state.productive_count}/{state.productive_max_count} positions "
            f"{'✅ HAS ROOM' if not_full else '⚠️ FULL'}"
        )
        
        # Gate 5: Overall trading health
        all_gates_pass = all(passed for passed, _ in gates.values())
        gates['all_gates_pass'] = (
            all_gates_pass,
            f"Overall: {'✅ SAFE TO TRADE' if all_gates_pass else '⚠️ RESTRICTED'}"
        )
        
        return gates
    
    def get_bucket_metrics(self) -> BucketMetrics:
        """
        Get current bucket metrics for monitoring.
        
        Returns:
            BucketMetrics with all bucket data
        """
        
        if not self.current_bucket_state:
            return BucketMetrics()
        
        state = self.current_bucket_state
        dist = state.get_bucket_distribution()
        
        metrics = BucketMetrics(
            timestamp=datetime.now(),
            operating_cash_pct=dist['operating_cash_pct'],
            productive_pct=dist['productive_pct'],
            dead_pct=dist['dead_pct'],
            operating_cash_value=state.operating_cash_usdt,
            productive_value=state.productive_total_value,
            dead_value=state.dead_total_value,
            productive_count=state.productive_count,
            dead_count=state.dead_count,
            operating_cash_health=state.operating_cash_health,
            portfolio_efficiency=state.portfolio_efficiency_pct,
            healing_potential=state.healing_potential,
        )
        
        # Add session healing metrics
        if self.last_healing_report:
            metrics.total_healed_this_session = self.last_healing_report.total_amount_recovered
            metrics.healing_events_this_session = self.last_healing_report.total_positions_healed
        
        return metrics
    
    def log_bucket_status(self) -> None:
        """Log current bucket status"""
        
        if not self.current_bucket_state:
            logger.warning("No bucket state to log")
            return
        
        metrics = self.get_bucket_metrics()
        logger.info(metrics.format_summary())
    
    def log_trading_gates(self) -> None:
        """Log current trading decision gates"""
        
        gates = self.get_trading_decision_gates()
        
        logger.info(f"""
🚦 TRADING DECISION GATES
""")
        
        for gate_name, (passed, reason) in gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{status:10s} | {gate_name:30s} | {reason}")
