"""
🎯 BUCKET CLASSIFIER
====================

Classifies each position into one of three buckets:
  A. Operating Cash - Sacred USDT reserve
  B. Productive - Active trading positions
  C. Dead Capital - Dust, stale, orphaned positions

This is the intelligent decision engine that determines portfolio structure.

Author: Wealth Engine Architect
Date: 2026-04-17
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from core.portfolio_buckets import (
    BucketType,
    DeadPositionReason,
    PositionClassification,
    PortfolioBucketState,
)

logger = logging.getLogger(__name__)


class BucketClassifier:
    """
    Intelligently classifies positions into three buckets.
    
    This is where the portfolio architecture comes alive.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize classifier with thresholds.
        
        Args:
            config: Dict with threshold overrides
        """
        self.config = config or {}
        
        # Bucket thresholds
        self.min_productive_size = self.config.get('min_productive_size', 25.0)      # $25 minimum
        self.stale_days = self.config.get('stale_days_threshold', 7)                 # 7 days
        self.performance_threshold_pct = self.config.get('performance_threshold', -15.0)  # -15%
        self.min_confidence = self.config.get('min_confidence_dead', 0.75)            # 75% to classify dead
        
        logger.info(f"✅ BucketClassifier initialized")
        logger.info(f"   Min productive size: ${self.min_productive_size:.2f}")
        logger.info(f"   Stale threshold: {self.stale_days} days")
        logger.info(f"   Performance threshold: {self.performance_threshold_pct}%")
    
    def classify_position(
        self,
        symbol: str,
        current_value: float,
        current_qty: float,
        current_price: float,
        entry_price: float,
        entry_datetime: Optional[datetime] = None,
        last_activity_datetime: Optional[datetime] = None,
        is_orphaned: bool = False,
    ) -> PositionClassification:
        """
        Classify a single position into a bucket.
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT')
            current_value: Current value in USDT ($)
            current_qty: Quantity held
            current_price: Current market price
            entry_price: Entry price
            entry_datetime: When position was entered
            last_activity_datetime: Last trade/modification
            is_orphaned: Is this a partial exit remnant?
        
        Returns:
            PositionClassification with bucket and reason
        """
        
        # Calculate performance
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Calculate staleness
        now = datetime.now()
        days_held = 0.0
        if entry_datetime:
            days_held = (now - entry_datetime).days
        
        days_since_activity = 999  # Default to very old
        if last_activity_datetime:
            days_since_activity = (now - last_activity_datetime).days
        
        # =====================================================================
        # CLASSIFICATION LOGIC
        # =====================================================================
        
        # RULE 1: Below minimum productive size → DEAD CAPITAL
        if current_value < self.min_productive_size:
            return PositionClassification(
                symbol=symbol,
                bucket=BucketType.DEAD_CAPITAL,
                reason=DeadPositionReason.BELOW_MIN_SIZE,
                current_value=current_value,
                confidence_score=0.95,  # Very confident
            )
        
        # RULE 2: Stale (no activity) → DEAD CAPITAL
        if days_since_activity > self.stale_days:
            return PositionClassification(
                symbol=symbol,
                bucket=BucketType.DEAD_CAPITAL,
                reason=DeadPositionReason.STALE,
                current_value=current_value,
                days_held=days_held,
                confidence_score=0.85,
            )
        
        # RULE 3: Orphaned position (tiny remnant from exit) → DEAD CAPITAL
        if is_orphaned:
            return PositionClassification(
                symbol=symbol,
                bucket=BucketType.DEAD_CAPITAL,
                reason=DeadPositionReason.ORPHANED,
                current_value=current_value,
                confidence_score=0.90,
            )
        
        # RULE 4: Failed performer (down > threshold) → DEAD CAPITAL
        if pnl_pct < self.performance_threshold_pct and days_held > 1:
            return PositionClassification(
                symbol=symbol,
                bucket=BucketType.DEAD_CAPITAL,
                reason=DeadPositionReason.FAILED_PERFORMER,
                current_value=current_value,
                performance_threshold_pct=pnl_pct,
                confidence_score=0.80,
            )
        
        # RULE 5: Fractional/dust share that can't trade → DEAD CAPITAL
        if current_qty < 0.0001:  # Essentially zero quantity
            return PositionClassification(
                symbol=symbol,
                bucket=BucketType.DEAD_CAPITAL,
                reason=DeadPositionReason.FRACTIONAL,
                current_value=current_value,
                confidence_score=0.95,
            )
        
        # =====================================================================
        # If no dead rules match, classify as PRODUCTIVE
        # =====================================================================
        return PositionClassification(
            symbol=symbol,
            bucket=BucketType.PRODUCTIVE,
            reason=None,
            current_value=current_value,
            entry_price=entry_price,
            entry_datetime=entry_datetime,
            days_held=days_held,
            last_activity_datetime=last_activity_datetime,
            confidence_score=0.90,
        )
    
    def classify_portfolio(
        self,
        positions: Dict[str, Dict],
        total_equity: float,
    ) -> PortfolioBucketState:
        """
        Classify entire portfolio into three buckets.
        
        Args:
            positions: Dict of all positions
                Format: {
                    'ETHUSDT': {
                        'qty': 0.01473796,
                        'entry_price': 2340.51,
                        'current_price': 2341.23,
                        'entry_time': datetime(...),
                        'value': 34.50,
                        ...
                    },
                    ...
                }
            total_equity: Total portfolio equity (cash + positions)
        
        Returns:
            PortfolioBucketState with full classification
        """
        
        bucket_state = PortfolioBucketState(
            total_equity=total_equity,
            productive_max_count=self.config.get('max_productive_positions', 5),
        )
        
        logger.info(f"🔍 Classifying {len(positions)} positions...")
        
        # Classify each position
        for symbol, pos_data in positions.items():
            
            # Skip if not a trading position (e.g., USDT cash)
            if symbol == 'USDT':
                bucket_state.operating_cash_usdt = pos_data.get('value', 0)
                continue
            
            # Get position attributes
            current_qty = pos_data.get('qty', 0)
            current_value = pos_data.get('value', 0)
            current_price = pos_data.get('current_price', 0)
            entry_price = pos_data.get('entry_price', current_price)
            entry_datetime = pos_data.get('entry_time')
            last_activity = pos_data.get('last_activity_time')
            is_orphaned = pos_data.get('is_orphaned', False)
            
            # Classify the position
            classification = self.classify_position(
                symbol=symbol,
                current_value=current_value,
                current_qty=current_qty,
                current_price=current_price,
                entry_price=entry_price,
                entry_datetime=entry_datetime,
                last_activity_datetime=last_activity,
                is_orphaned=is_orphaned,
            )
            
            # Store classification
            bucket_state.classifications[symbol] = classification
            
            # Add to appropriate bucket
            if classification.bucket == BucketType.PRODUCTIVE:
                bucket_state.productive_positions[symbol] = pos_data
                bucket_state.productive_total_value += current_value
                bucket_state.productive_count += 1
                
                logger.debug(
                    f"   ✅ {symbol}: PRODUCTIVE | Value=${current_value:.2f} | "
                    f"Confidence={classification.confidence_score:.0%}"
                )
            
            elif classification.bucket == BucketType.DEAD_CAPITAL:
                bucket_state.dead_positions[symbol] = {
                    **pos_data,
                    'reason': classification.reason,
                    'can_liquidate': True,
                    'liquidation_priority': 1,
                }
                bucket_state.dead_total_value += current_value
                bucket_state.dead_count += 1
                
                logger.debug(
                    f"   💀 {symbol}: DEAD ({classification.reason.value}) | "
                    f"Value=${current_value:.2f} | Confidence={classification.confidence_score:.0%}"
                )
        
        # Calculate derived metrics
        bucket_state.productive_avg_size = (
            bucket_state.productive_total_value / bucket_state.productive_count
            if bucket_state.productive_count > 0
            else 0.0
        )
        
        bucket_state.healing_potential = bucket_state.dead_total_value
        
        bucket_state.total_portfolio_value = (
            bucket_state.operating_cash_usdt +
            bucket_state.productive_total_value +
            bucket_state.dead_total_value
        )
        
        # Calculate health status
        bucket_state.operating_cash_health = self._assess_operating_cash_health(bucket_state)
        bucket_state.bucket_balance_score = self._calculate_balance_score(bucket_state)
        bucket_state.portfolio_efficiency_pct = (
            (bucket_state.productive_total_value / bucket_state.total_portfolio_value * 100)
            if bucket_state.total_portfolio_value > 0
            else 0.0
        )
        
        bucket_state.last_classification_time = datetime.now()
        
        # Log summary
        self._log_classification_summary(bucket_state)
        
        return bucket_state
    
    def _assess_operating_cash_health(self, state: PortfolioBucketState) -> str:
        """Assess health of operating cash bucket"""
        if state.operating_cash_usdt < state.operating_cash_floor:
            return "CRITICAL"
        elif state.operating_cash_usdt < state.operating_cash_danger_zone:
            return "LOW"
        else:
            return "HEALTHY"
    
    def _calculate_balance_score(self, state: PortfolioBucketState) -> float:
        """
        Calculate portfolio balance score (0-100).
        
        100 = Perfect balance
        Penalize for:
          - Operating cash too low or too high
          - Dead capital present
          - Portfolio efficiency off target
        """
        score = 100.0
        
        # Ideal distribution: 20% cash, 75% productive, 5% dead (or less)
        ideal_cash_pct = 20.0
        ideal_productive_pct = 75.0
        ideal_dead_pct = 5.0
        
        distribution = state.get_bucket_distribution()
        
        # Penalize deviation from ideal cash %
        cash_deviation = abs(distribution['operating_cash_pct'] - ideal_cash_pct)
        score -= min(cash_deviation, 20)  # Max -20 for cash imbalance
        
        # Penalize too much dead capital
        dead_deviation = distribution['dead_pct'] - ideal_dead_pct
        if dead_deviation > 0:
            score -= min(dead_deviation * 5, 25)  # Max -25 for dead capital
        
        # Reward high productive %
        productive_deviation = abs(distribution['productive_pct'] - ideal_productive_pct)
        if productive_deviation < 10:
            score += 5
        
        return max(0, min(100, score))
    
    def _log_classification_summary(self, state: PortfolioBucketState) -> None:
        """Log readable summary of classification"""
        dist = state.get_bucket_distribution()
        
        logger.info(f"""
📊 PORTFOLIO CLASSIFICATION COMPLETE
├─ 💵 Operating Cash:  ${state.operating_cash_usdt:>8.2f} ({dist['operating_cash_pct']:>5.1f}%) [{state.operating_cash_health}]
├─ 📈 Productive:      ${state.productive_total_value:>8.2f} ({dist['productive_pct']:>5.1f}%) in {state.productive_count} positions (avg ${state.productive_avg_size:.2f})
├─ 💀 Dead Capital:    ${state.dead_total_value:>8.2f} ({dist['dead_pct']:>5.1f}%) in {state.dead_count} positions
├─ 📌 Portfolio Total: ${state.total_portfolio_value:>8.2f}
└─ 🎯 Efficiency:      {state.portfolio_efficiency_pct:.1f}% productive | Balance score: {state.bucket_balance_score:.0f}/100
""")
        
        # Log position-by-position details if verbose
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Position classifications:")
            for symbol, classification in state.classifications.items():
                if classification.bucket == BucketType.DEAD_CAPITAL:
                    logger.debug(
                        f"  💀 {symbol:10s} → DEAD ({classification.reason.value:30s}) "
                        f"${classification.current_value:>8.2f}"
                    )
                else:
                    logger.debug(
                        f"  ✅ {symbol:10s} → PRODUCTIVE "
                        f"${classification.current_value:>8.2f}"
                    )
