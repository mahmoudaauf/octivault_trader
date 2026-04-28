#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
PHASE 1: PORTFOLIO SEGMENTATION COMPONENT
═══════════════════════════════════════════════════════════════════════════════

Implements the three-bucket model:
  A. Operating Cash - Strategic reserve (never deployed)
  B. Productive Inventory - Active high-conviction positions
  C. Dead Capital - Dust & stale positions (marked for liquidation)

This component transforms the system from treating the portfolio as one pool
into a structured wealth management framework.

═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class BucketType(Enum):
    """Portfolio bucket classification"""
    OPERATING_CASH = "operating_cash"      # Strategic reserve (A)
    PRODUCTIVE_INVENTORY = "productive"    # Active positions (B)
    DEAD_CAPITAL = "dead_capital"          # Dust & stale (C)


class HoldingClassification(Enum):
    """Holding utility classification"""
    TRADABLE = "tradable"                  # Above $10 USDT value
    NEAR_DUST = "near_dust"                # $5-10 USDT value
    DUST = "dust"                          # < $5 USDT value
    RECOVERABLE = "recoverable"            # Can be merged later
    WRITE_DOWN = "write_down"              # Permanent loss


@dataclass
class HoldingSegment:
    """Classification of a single holding"""
    symbol: str
    quantity: float
    current_value: float
    
    # Classification
    classification: HoldingClassification
    bucket: BucketType
    
    # Metrics
    utility_score: float          # 0-100
    days_held: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Recommendation
    recommended_action: str       # "KEEP", "MONITOR", "EXIT", "MERGE"
    priority: int                 # 1-5, lower = higher priority for exit


@dataclass
class PortfolioSegmentation:
    """Complete portfolio segmentation snapshot"""
    timestamp: datetime
    
    # Bucket totals
    operating_cash_usdt: float
    productive_value: float
    dead_capital_value: float
    
    # Percentages
    cash_ratio_pct: float
    productive_ratio_pct: float
    dead_ratio_pct: float
    
    # Health metrics
    portfolio_health_score: float  # 0-100
    dust_ratio_pct: float
    fragmentation_index: float     # 0-100
    
    # Holdings
    holdings: List[HoldingSegment]
    
    # Recommendations
    immediate_actions: List[str]
    strategic_recommendations: List[str]


class PortfolioSegmentationManager:
    """
    Manages portfolio segmentation across three buckets.
    
    Main responsibilities:
    1. Track USDT separately as operating cash
    2. Classify each holding by utility
    3. Detect dust automatically
    4. Enforce bucket policies
    5. Generate actionable recommendations
    """
    
    # Configuration
    DUST_THRESHOLD_USDT = 10.0        # Positions < $10 are dust
    NEAR_DUST_THRESHOLD_USDT = 20.0   # Near-dust threshold
    MIN_RESERVE_RATIO_NORMAL = 0.15   # 15% in normal markets
    MIN_RESERVE_RATIO_VOLATILE = 0.25 # 25% in volatile markets
    OPTIMAL_POSITION_COUNT = 4        # Ideal number of active positions
    MAX_POSITION_COUNT = 8            # Maximum before fragmentation alert
    
    def __init__(self, config=None, shared_state=None, logger=None):
        """Initialize segmentation manager"""
        self.config = config
        self.shared_state = shared_state
        self.logger = logger or print
        self.last_segmentation = None
        
    async def segment_portfolio(self) -> PortfolioSegmentation:
        """
        Analyze complete portfolio and segment into buckets.
        
        Returns:
            PortfolioSegmentation with all metrics
        """
        # Get current balances
        balances = self.shared_state.balances if self.shared_state else {}
        usdt_balance = balances.get("USDT", {}).get("free", 0)
        
        # Get positions
        positions = self.shared_state.positions if self.shared_state else {}
        
        # Calculate total NAV
        total_value = usdt_balance
        holdings = []
        
        for symbol, position in positions.items():
            if symbol == "USDT":
                continue
                
            quantity = position.get("quantity", 0)
            current_price = position.get("current_price", 0)
            entry_price = position.get("entry_price", 0)
            
            value = quantity * current_price
            total_value += value
            
            # Calculate metrics
            pnl = (current_price - entry_price) * quantity if entry_price > 0 else 0
            pnl_pct = (pnl / (entry_price * quantity)) if entry_price * quantity > 0 else 0
            
            # Classify holding
            classification = self._classify_holding(symbol, value, position)
            bucket = self._assign_bucket(symbol, classification, position)
            
            # Calculate utility
            utility = self._calculate_utility_score(
                symbol, value, classification, pnl_pct, position
            )
            
            # Get recommended action
            action, priority = self._get_recommendation(
                symbol, classification, bucket, utility, position
            )
            
            holding = HoldingSegment(
                symbol=symbol,
                quantity=quantity,
                current_value=value,
                classification=classification,
                bucket=bucket,
                utility_score=utility,
                days_held=position.get("days_held", 0),
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
                recommended_action=action,
                priority=priority
            )
            holdings.append(holding)
        
        # Calculate bucket totals
        operating_cash = usdt_balance
        productive_value = sum(h.current_value for h in holdings 
                              if h.bucket == BucketType.PRODUCTIVE_INVENTORY)
        dead_capital_value = sum(h.current_value for h in holdings 
                                if h.bucket == BucketType.DEAD_CAPITAL)
        
        # Calculate ratios
        cash_ratio = (operating_cash / total_value * 100) if total_value > 0 else 0
        productive_ratio = (productive_value / total_value * 100) if total_value > 0 else 0
        dead_ratio = (dead_capital_value / total_value * 100) if total_value > 0 else 0
        
        # Calculate health metrics
        health_score = self._calculate_health_score(
            cash_ratio, dead_ratio, len(holdings)
        )
        dust_ratio = dead_ratio
        fragmentation = self._calculate_fragmentation_index(len(holdings))
        
        # Generate recommendations
        immediate_actions = self._get_immediate_actions(
            holdings, cash_ratio, dead_ratio, len(holdings)
        )
        strategic_recs = self._get_strategic_recommendations(
            cash_ratio, dead_ratio, health_score, fragmentation
        )
        
        segmentation = PortfolioSegmentation(
            timestamp=datetime.now(),
            operating_cash_usdt=operating_cash,
            productive_value=productive_value,
            dead_capital_value=dead_capital_value,
            cash_ratio_pct=cash_ratio,
            productive_ratio_pct=productive_ratio,
            dead_ratio_pct=dead_ratio,
            portfolio_health_score=health_score,
            dust_ratio_pct=dust_ratio,
            fragmentation_index=fragmentation,
            holdings=holdings,
            immediate_actions=immediate_actions,
            strategic_recommendations=strategic_recs
        )
        
        self.last_segmentation = segmentation
        self._log_segmentation(segmentation)
        
        return segmentation
    
    def _classify_holding(self, symbol: str, value: float, position: dict) -> HoldingClassification:
        """Classify a holding into utility categories"""
        if value < 5:
            return HoldingClassification.DUST
        elif value < self.NEAR_DUST_THRESHOLD_USDT:
            return HoldingClassification.NEAR_DUST
        elif value >= self.DUST_THRESHOLD_USDT:
            return HoldingClassification.TRADABLE
        else:
            return HoldingClassification.RECOVERABLE
    
    def _assign_bucket(self, symbol: str, classification: HoldingClassification, 
                      position: dict) -> BucketType:
        """Assign holding to a bucket"""
        if classification == HoldingClassification.DUST:
            return BucketType.DEAD_CAPITAL
        elif classification == HoldingClassification.WRITE_DOWN:
            return BucketType.DEAD_CAPITAL
        
        # Check if stale (no movement in 48h)
        days_held = position.get("days_held", 0)
        if days_held > 2 and position.get("unrealized_pnl_pct", 0) < -0.05:
            return BucketType.DEAD_CAPITAL
        
        return BucketType.PRODUCTIVE_INVENTORY
    
    def _calculate_utility_score(self, symbol: str, value: float, 
                                 classification: HoldingClassification,
                                 pnl_pct: float, position: dict) -> float:
        """Calculate utility score for a holding (0-100)"""
        score = 50  # Base score
        
        # Size matters
        if value >= self.NEAR_DUST_THRESHOLD_USDT:
            score += 20
        elif value >= self.DUST_THRESHOLD_USDT:
            score += 10
        
        # PnL quality
        if pnl_pct > 0.05:
            score += 15
        elif pnl_pct > 0:
            score += 8
        elif pnl_pct > -0.05:
            score += 2
        else:
            score -= 10
        
        # Classification
        if classification == HoldingClassification.TRADABLE:
            score += 10
        elif classification == HoldingClassification.DUST:
            score -= 40
        
        return max(0, min(100, score))  # Clamp 0-100
    
    def _get_recommendation(self, symbol: str, classification: HoldingClassification,
                           bucket: BucketType, utility: float, 
                           position: dict) -> tuple[str, int]:
        """Get action recommendation and priority"""
        if classification == HoldingClassification.DUST:
            return ("EXIT_ASAP", 1)
        elif classification == HoldingClassification.NEAR_DUST:
            return ("EXIT_SOON", 2)
        elif bucket == BucketType.DEAD_CAPITAL:
            return ("EXIT", 3)
        elif utility < 40:
            return ("MONITOR", 4)
        else:
            return ("KEEP", 5)
    
    def _calculate_health_score(self, cash_ratio: float, dead_ratio: float, 
                               position_count: int) -> float:
        """Calculate portfolio health score (0-100)"""
        score = 60  # Base score
        
        # Cash ratio (should be 15-25%)
        if 15 <= cash_ratio <= 25:
            score += 20
        elif 10 <= cash_ratio <= 30:
            score += 10
        elif cash_ratio < 10:
            score -= 20
        elif cash_ratio > 35:
            score -= 5
        
        # Dead capital (should be < 2%)
        if dead_ratio < 2:
            score += 15
        elif dead_ratio < 5:
            score += 5
        elif dead_ratio > 10:
            score -= 15
        
        # Position count (optimal 2-5)
        if 2 <= position_count <= 5:
            score += 5
        elif position_count > 8:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_fragmentation_index(self, position_count: int) -> float:
        """Calculate portfolio fragmentation (0-100, lower = better)"""
        if position_count <= self.OPTIMAL_POSITION_COUNT:
            return 20
        elif position_count <= self.MAX_POSITION_COUNT:
            return 20 + (position_count - self.OPTIMAL_POSITION_COUNT) * 10
        else:
            return 80  # High fragmentation
    
    def _get_immediate_actions(self, holdings: List[HoldingSegment], 
                              cash_ratio: float, dead_ratio: float,
                              position_count: int) -> List[str]:
        """Get list of immediate actions needed"""
        actions = []
        
        # Cash below minimum?
        if cash_ratio < 10:
            actions.append("🚨 CRITICAL: Operating cash below 10% - liquidate lowest-utility holdings")
        
        # Dust above threshold?
        dust_holdings = [h for h in holdings if h.classification in [
            HoldingClassification.DUST, HoldingClassification.NEAR_DUST
        ]]
        if dust_holdings:
            actions.append(f"⚠️  Liquidate {len(dust_holdings)} dust/near-dust positions ({', '.join([h.symbol for h in dust_holdings])})")
        
        # Too fragmented?
        if position_count > self.MAX_POSITION_COUNT:
            actions.append(f"⚠️  Portfolio too fragmented ({position_count} positions) - consolidate")
        
        # Dead capital?
        dead_holdings = [h for h in holdings if h.bucket == BucketType.DEAD_CAPITAL]
        if dead_holdings:
            actions.append(f"⚠️  Exit {len(dead_holdings)} dead capital positions ({', '.join([h.symbol for h in dead_holdings])})")
        
        return actions
    
    def _get_strategic_recommendations(self, cash_ratio: float, dead_ratio: float,
                                       health_score: float, fragmentation: float) -> List[str]:
        """Get list of strategic recommendations"""
        recs = []
        
        if health_score > 80:
            recs.append("✅ Portfolio in excellent health - can consider expansion")
        elif health_score > 60:
            recs.append("✅ Portfolio healthy - maintain current position")
        else:
            recs.append("⚠️  Portfolio health declining - focus on cleanup before new entries")
        
        if cash_ratio < 15:
            recs.append("📌 Increase cash reserve through selective liquidation")
        elif cash_ratio > 30:
            recs.append("📌 Deploy excess cash into highest-conviction opportunities")
        
        if dead_ratio > 5:
            recs.append("📌 Dust ratio elevated - implement aggressive sweep policy")
        
        if fragmentation > 50:
            recs.append("📌 High fragmentation - consider reducing position count")
        
        return recs
    
    def _log_segmentation(self, seg: PortfolioSegmentation):
        """Log segmentation results"""
        self.logger(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║ 📊 PORTFOLIO SEGMENTATION SNAPSHOT                                    ║
╚═══════════════════════════════════════════════════════════════════════╝

BUCKET ALLOCATION:
  A. Operating Cash (Strategic Reserve): ${seg.operating_cash_usdt:,.2f} ({seg.cash_ratio_pct:.1f}%)
  B. Productive Inventory (Active):       ${seg.productive_value:,.2f} ({seg.productive_ratio_pct:.1f}%)
  C. Dead Capital (Dust & Stale):         ${seg.dead_capital_value:,.2f} ({seg.dead_ratio_pct:.1f}%)
  
  Total NAV: ${seg.operating_cash_usdt + seg.productive_value + seg.dead_capital_value:,.2f}

PORTFOLIO HEALTH:
  Health Score: {seg.portfolio_health_score:.0f}/100 {"🟢" if seg.portfolio_health_score > 70 else "🟡" if seg.portfolio_health_score > 50 else "🔴"}
  Fragmentation: {seg.fragmentation_index:.0f}/100 {"✅" if seg.fragmentation_index < 40 else "⚠️ "}
  Dust Ratio: {seg.dust_ratio_pct:.1f}%
  Active Positions: {len(seg.holdings)}

IMMEDIATE ACTIONS ({len(seg.immediate_actions)}):
""")
        for action in seg.immediate_actions:
            self.logger(f"  {action}")
        
        self.logger(f"\nSTRATEGIC RECOMMENDATIONS ({len(seg.strategic_recommendations)}):")
        for rec in seg.strategic_recommendations:
            self.logger(f"  {rec}")
        
        self.logger("\nHOLDINGS BY TIER:\n")
        
        # Tier 1: Keep
        tier1 = [h for h in seg.holdings if h.recommended_action == "KEEP"]
        if tier1:
            self.logger("  TIER 1 - KEEP (High utility):")
            for h in tier1:
                self.logger(f"    {h.symbol}: ${h.current_value:.2f} | Score: {h.utility_score:.0f} | PnL: {h.unrealized_pnl_pct:+.2%}")
        
        # Tier 3: Monitor
        tier3 = [h for h in seg.holdings if h.recommended_action == "MONITOR"]
        if tier3:
            self.logger("  TIER 3 - MONITOR (Declining utility):")
            for h in tier3:
                self.logger(f"    {h.symbol}: ${h.current_value:.2f} | Score: {h.utility_score:.0f} | PnL: {h.unrealized_pnl_pct:+.2%}")
        
        # Tier 4: Exit
        tier4 = [h for h in seg.holdings if h.recommended_action in ["EXIT", "EXIT_SOON", "EXIT_ASAP"]]
        if tier4:
            self.logger("  TIER 4 - EXIT (Low utility, exit priority):")
            for h in sorted(tier4, key=lambda x: x.priority):
                self.logger(f"    {h.symbol}: ${h.current_value:.2f} | Score: {h.utility_score:.0f} | Action: {h.recommended_action}")


async def main():
    """Demo segmentation manager"""
    
    # Mock shared state for demo
    class MockSharedState:
        def __init__(self):
            self.balances = {
                "USDT": {"free": 25000, "locked": 0},
                "BTC": {"free": 0.5, "locked": 0},
            }
            self.positions = {
                "BTCUSDT": {
                    "quantity": 0.5,
                    "entry_price": 42000,
                    "current_price": 45000,
                    "days_held": 5,
                    "unrealized_pnl_pct": 0.071,
                },
                "ETHUSDT": {
                    "quantity": 10,
                    "entry_price": 2500,
                    "current_price": 2300,
                    "days_held": 10,
                    "unrealized_pnl_pct": -0.08,
                },
                "DUSTCOIN": {
                    "quantity": 1000,
                    "entry_price": 0.01,
                    "current_price": 0.003,
                    "days_held": 30,
                    "unrealized_pnl_pct": -0.70,
                },
            }
    
    manager = PortfolioSegmentationManager(shared_state=MockSharedState())
    seg = await manager.segment_portfolio()
    
    print("\n✅ Portfolio segmentation complete")
    print(f"   Health Score: {seg.portfolio_health_score:.0f}/100")
    print(f"   Immediate Actions: {len(seg.immediate_actions)}")


if __name__ == "__main__":
    asyncio.run(main())
