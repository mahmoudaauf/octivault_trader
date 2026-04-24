"""
🎯 THREE-BUCKET PORTFOLIO ARCHITECTURE
=====================================

Core data structures and enums for three-bucket portfolio management.

Buckets:
  A. Operating Cash    - Sacred, never-zero reserve for liquidity
  B. Productive Inventory - Strategic positions generating returns
  C. Dead Capital      - Dust, stale, orphaned positions (ruthlessly liquidated)

Author: Wealth Engine Architect
Date: 2026-04-17
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


class BucketType(Enum):
    """Portfolio bucket classification"""
    OPERATING_CASH = "operating_cash"    # Sacred reserve
    PRODUCTIVE = "productive"             # Active trading positions
    DEAD_CAPITAL = "dead_capital"         # Dust/stale/orphaned


class DeadPositionReason(Enum):
    """Why a position is classified as dead"""
    BELOW_MIN_SIZE = "below_min_size"                    # < $25
    STALE = "stale"                                      # No activity > 7 days
    ORPHANED = "orphaned"                                # Partial exit remnant
    HIGH_OPPORTUNITY_COST = "high_opportunity_cost"      # Better opportunities exist
    FAILED_PERFORMER = "failed_performer"                # Down > -15% from entry
    PERMANENT_DUST = "permanent_dust"                    # Multiple failed attempts
    FRACTIONAL = "fractional"                            # Can't efficiently trade


@dataclass
class PositionClassification:
    """How a single position is classified"""
    symbol: str
    bucket: BucketType
    reason: Optional[DeadPositionReason] = None
    
    # Classification factors
    current_value: float = 0.0
    min_tradable_threshold: float = 25.0                 # $25 minimum
    stale_days_threshold: int = 7
    performance_threshold_pct: float = -15.0             # -15% from entry
    
    # Timing
    entry_price: float = 0.0
    entry_datetime: Optional[datetime] = None
    days_held: float = 0.0
    last_activity_datetime: Optional[datetime] = None
    
    # Metadata
    confidence_score: float = 0.0                        # 0-1, how confident in classification
    classification_timestamp: datetime = field(default_factory=datetime.now)
    
    def is_certain(self) -> bool:
        """Is this classification highly confident?"""
        return self.confidence_score > 0.85


@dataclass
class PortfolioBucketState:
    """
    Real-time state of the three-bucket portfolio.
    
    This is the SOURCE OF TRUTH for portfolio classification.
    """
    
    # Bucket A: Operating Cash (Sacred Reserve)
    # ============================================
    operating_cash_usdt: float = 0.0                     # Current USDT balance
    operating_cash_target_pct: float = 0.20              # Target 20% of total
    operating_cash_floor: float = 10.0                   # Absolute minimum $10
    
    # Safety zone (1.2x floor = $12)
    operating_cash_danger_zone: float = field(default_factory=lambda: 10.0 * 1.2)
    
    # Bucket B: Productive Inventory (Strategic Holdings)
    # ====================================================
    productive_positions: Dict[str, dict] = field(default_factory=dict)
    """
    Format: {
        'ETHUSDT': {
            'value': 34.50,
            'qty': 0.01473796,
            'entry_price': 2340.51,
            'entry_time': datetime(...),
            'current_price': 2341.23,
            'pnl_pct': 0.03,
        },
        ...
    }
    """
    
    productive_total_value: float = 0.0                  # Sum of all productive positions
    productive_count: int = 0                            # Number of positions
    productive_avg_size: float = 0.0                     # Average position value
    productive_max_count: int = 5                        # Max simultaneous positions
    
    # Bucket C: Dead Capital (To Be Liquidated)
    # ==========================================
    dead_positions: Dict[str, dict] = field(default_factory=dict)
    """
    Format: {
        'BTCUSDT': {
            'value': 0.0592,
            'qty': 0.00000079,
            'reason': DeadPositionReason.BELOW_MIN_SIZE,
            'can_liquidate': True,
            'liquidation_priority': 1,  # 1 = highest priority
        },
        ...
    }
    """
    
    dead_total_value: float = 0.0                        # Sum of all dead positions
    dead_count: int = 0                                  # Number of dead positions
    dead_min_size_threshold: float = 25.0                # $25 minimum for productive
    
    # Healing Metrics
    # ===============
    healing_potential: float = 0.0                       # Total $ that can be recovered
    total_healed_this_session: float = 0.0               # $ recovered in current session
    healing_rate_per_hour: float = 0.0                   # $ per hour
    avg_dead_lifetime_days: float = 0.0                  # Days before liquidation
    
    # Portfolio Totals
    # ================
    total_portfolio_value: float = 0.0                   # NAV (total invested)
    total_equity: float = 0.0                            # Total including cash
    
    # Health Score (0-100)
    # ====================
    operating_cash_health: str = "HEALTHY"               # CRITICAL, LOW, HEALTHY
    bucket_balance_score: float = 100.0                  # 0-100, ideal is ~100
    portfolio_efficiency_pct: float = 0.0                # % in productive (target 60-80%)
    
    # Classifications Log
    # ===================
    classifications: Dict[str, PositionClassification] = field(default_factory=dict)
    last_classification_time: Optional[datetime] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    def get_bucket_distribution(self) -> Dict[str, float]:
        """Get % distribution across three buckets"""
        if self.total_equity == 0:
            return {
                'operating_cash_pct': 0.0,
                'productive_pct': 0.0,
                'dead_pct': 0.0,
            }
        
        return {
            'operating_cash_pct': (self.operating_cash_usdt / self.total_equity) * 100,
            'productive_pct': (self.productive_total_value / self.total_equity) * 100,
            'dead_pct': (self.dead_total_value / self.total_equity) * 100,
        }
    
    def is_operating_cash_healthy(self) -> bool:
        """Is operating cash above danger zone?"""
        return self.operating_cash_usdt >= self.operating_cash_danger_zone
    
    def is_operating_cash_critical(self) -> bool:
        """Is operating cash below absolute floor?"""
        return self.operating_cash_usdt < self.operating_cash_floor
    
    def can_trade_new_position(self) -> bool:
        """Is it safe to enter a new position?"""
        # Gate 1: Operating cash not threatened
        if not self.is_operating_cash_healthy():
            return False
        
        # Gate 2: Portfolio not full
        if self.productive_count >= self.productive_max_count:
            return False
        
        # Gate 3: Dead capital not excessive
        if self.dead_total_value > self.operating_cash_usdt * 0.5:
            return False
        
        return True
    
    def should_heal_dead_capital(self) -> bool:
        """Should we prioritize liquidating dead capital?"""
        return self.dead_total_value > 50.0  # If > $50 in dust, heal it
    
    def get_healing_priority_order(self) -> List[str]:
        """Get order to liquidate dead positions (highest to lowest priority)"""
        # Sort by value (largest first to recover more capital quickly)
        sorted_dead = sorted(
            self.dead_positions.items(),
            key=lambda x: x[1].get('value', 0),
            reverse=True
        )
        return [symbol for symbol, _ in sorted_dead]


@dataclass
class HealingEvent:
    """Record of a healing operation"""
    symbol: str
    bucket_from: BucketType
    bucket_to: BucketType
    amount_recovered: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_msg: Optional[str] = None


@dataclass
class HealingReport:
    """Summary of healing operations"""
    session_id: str
    timestamp: datetime
    total_positions_healed: int
    total_amount_recovered: float
    healing_events: List[HealingEvent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate_pct(self) -> float:
        """What % of healing operations succeeded?"""
        if not self.healing_events:
            return 100.0
        
        successful = sum(1 for e in self.healing_events if e.success)
        return (successful / len(self.healing_events)) * 100


@dataclass
class BucketMetrics:
    """Real-time metrics for three-bucket portfolio"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Bucket sizes (%)
    operating_cash_pct: float = 0.0
    productive_pct: float = 0.0
    dead_pct: float = 0.0
    
    # Bucket sizes ($)
    operating_cash_value: float = 0.0
    productive_value: float = 0.0
    dead_value: float = 0.0
    
    # Counts
    productive_count: int = 0
    dead_count: int = 0
    
    # Health
    operating_cash_health: str = "HEALTHY"
    portfolio_efficiency: float = 0.0  # % in productive (target 60-80%)
    healing_potential: float = 0.0
    
    # Session totals
    total_healed_this_session: float = 0.0
    healing_events_this_session: int = 0
    
    def format_summary(self) -> str:
        """Format as readable summary"""
        return f"""
📊 THREE-BUCKET PORTFOLIO METRICS
├─ 💵 Operating Cash:  ${self.operating_cash_value:.2f} ({self.operating_cash_pct:.1f}%)
│  └─ Health: {self.operating_cash_health}
├─ 📈 Productive:       ${self.productive_value:.2f} ({self.productive_pct:.1f}%) in {self.productive_count} pos
├─ 💀 Dead Capital:     ${self.dead_value:.2f} ({self.dead_pct:.1f}%) in {self.dead_count} pos
│  └─ Healing potential: ${self.healing_potential:.2f}
└─ ✅ Portfolio Efficiency: {self.portfolio_efficiency:.1f}%
   └─ Healed this session: ${self.total_healed_this_session:.2f} ({self.healing_events_this_session} events)
"""
