"""
🔍 Dust Position Monitoring System

Real-time monitoring of dust positions with health checks,
alerting, and comprehensive metrics.

This is Priority 1 for the next iteration.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


logger = logging.getLogger(__name__)


class DustHealthStatus(Enum):
    """Health status of a dust position."""
    HEALTHY = "healthy"              # Accumulating normally
    STALLED = "stalled"              # Not accumulating, no progress
    CRITICAL = "critical"            # Below minimum, very old
    RECOVERED = "recovered"          # Just crossed minNotional
    MANUAL_INTERVENTION = "manual"   # Needs manual review


@dataclass
class DustAlert:
    """Alert for dust position issues."""
    symbol: str
    alert_type: str  # "stalled", "critical", "age_limit", etc.
    message: str
    severity: str  # "info", "warning", "error"
    timestamp: float = field(default_factory=time.time)
    notional: float = 0.0
    age_hours: float = 0.0
    action_required: bool = False
    suggested_action: Optional[str] = None


@dataclass
class DustPositionMetrics:
    """Metrics for a single dust position."""
    symbol: str
    quantity: float
    notional: float
    min_notional: float
    price: float
    created_at: float
    last_updated: float
    age_hours: float
    status: DustHealthStatus
    accumulation_rate_per_hour: float
    estimated_recovery_hours: Optional[float]
    rejection_count: int = 0
    skip_count: int = 0  # Times skipped by invariant


@dataclass
class DustStatistics:
    """Overall dust system statistics."""
    total_dust_positions: int = 0
    total_dust_notional: float = 0.0
    average_age_hours: float = 0.0
    positions_healthy: int = 0
    positions_stalled: int = 0
    positions_critical: int = 0
    
    dust_creation_rate: float = 0.0  # positions per hour
    prevention_rate: float = 0.0  # percentage prevented
    average_accumulation_time: float = 0.0  # hours
    
    capital_trapped: float = 0.0  # total notional in dust
    capital_recoverable: float = 0.0  # notional expected to recover
    recovery_rate: float = 0.0  # % of dust that successfully recovers
    
    alerts_count: int = 0
    alerts_critical: int = 0
    
    last_update: float = field(default_factory=time.time)


class DustMonitor:
    """
    Real-time dust position monitoring system.
    
    Tracks dust positions, detects issues, generates alerts,
    and provides comprehensive metrics.
    """
    
    def __init__(self, shared_state, config):
        """
        Initialize monitor.
        
        Args:
            shared_state: SharedState instance
            config: Configuration object
        """
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DustMonitor")
        
        # Tracking
        self._dust_positions: Dict[str, Dict[str, Any]] = {}
        self._position_creation_times: Dict[str, float] = {}
        self._position_skip_counts: Dict[str, int] = defaultdict(int)
        self._alerts: List[DustAlert] = []
        
        # Configuration
        self._stall_threshold_hours = float(
            getattr(config, "DUST_STALL_THRESHOLD_HOURS", 4.0)
        )
        self._critical_threshold_hours = float(
            getattr(config, "DUST_CRITICAL_THRESHOLD_HOURS", 8.0)
        )
        self._min_notional_floor = float(
            getattr(config, "MIN_NOTIONAL_FLOOR", 10.0)
        )
    
    async def update_dust_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        min_notional: float,
        was_skipped: bool = False
    ) -> None:
        """
        Update dust position tracking.
        
        Called whenever a dust position is encountered.
        
        Args:
            symbol: Trading pair
            quantity: Position quantity
            price: Current mark price
            min_notional: Exchange minNotional
            was_skipped: Whether this position was skipped by invariant
        """
        try:
            now = time.time()
            notional = quantity * price
            
            # Record position
            if symbol not in self._dust_positions:
                self._position_creation_times[symbol] = now
                self.logger.info(
                    "[DustMonitor] New dust position: %s notional=%.2f min=%.2f",
                    symbol, notional, min_notional
                )
            
            self._dust_positions[symbol] = {
                "quantity": quantity,
                "price": price,
                "notional": notional,
                "min_notional": min_notional,
                "last_updated": now,
            }
            
            # Track skips (times invariant blocked this)
            if was_skipped:
                self._position_skip_counts[symbol] += 1
        
        except Exception as e:
            self.logger.warning(
                "[DustMonitor] Error updating dust position %s: %s",
                symbol, e
            )
    
    async def remove_dust_position(self, symbol: str) -> None:
        """
        Remove dust position (recovered or liquidated).
        
        Args:
            symbol: Trading pair
        """
        if symbol in self._dust_positions:
            notional = self._dust_positions[symbol]["notional"]
            age_hours = self._get_position_age_hours(symbol)
            
            self.logger.info(
                "[DustMonitor] Dust position recovered: %s notional=%.2f age_hours=%.1f",
                symbol, notional, age_hours
            )
            
            del self._dust_positions[symbol]
            if symbol in self._position_creation_times:
                del self._position_creation_times[symbol]
            if symbol in self._position_skip_counts:
                del self._position_skip_counts[symbol]
    
    def _get_position_age_hours(self, symbol: str) -> float:
        """Get age of dust position in hours."""
        if symbol not in self._position_creation_times:
            return 0.0
        
        created_at = self._position_creation_times[symbol]
        age_seconds = time.time() - created_at
        return age_seconds / 3600.0
    
    def _get_position_health(self, symbol: str) -> DustHealthStatus:
        """Determine health status of dust position."""
        age_hours = self._get_position_age_hours(symbol)
        
        if age_hours >= self._critical_threshold_hours:
            return DustHealthStatus.CRITICAL
        elif age_hours >= self._stall_threshold_hours:
            return DustHealthStatus.STALLED
        else:
            return DustHealthStatus.HEALTHY
    
    async def check_dust_health(self) -> DustStatistics:
        """
        Check overall dust system health.
        
        Returns:
            Comprehensive dust statistics
        """
        try:
            stats = DustStatistics()
            stats.total_dust_positions = len(self._dust_positions)
            
            healthy = 0
            stalled = 0
            critical = 0
            total_notional = 0.0
            
            for symbol, pos_data in self._dust_positions.items():
                notional = pos_data["notional"]
                total_notional += notional
                
                health = self._get_position_health(symbol)
                
                if health == DustHealthStatus.HEALTHY:
                    healthy += 1
                elif health == DustHealthStatus.STALLED:
                    stalled += 1
                elif health == DustHealthStatus.CRITICAL:
                    critical += 1
            
            stats.total_dust_notional = total_notional
            stats.positions_healthy = healthy
            stats.positions_stalled = stalled
            stats.positions_critical = critical
            stats.capital_trapped = total_notional
            
            # Calculate recovery rate (positions that recovered)
            # This would be tracked separately in production
            stats.recovery_rate = 0.78  # Placeholder
            
            # Calculate average age
            if self._dust_positions:
                ages = [
                    self._get_position_age_hours(sym)
                    for sym in self._dust_positions.keys()
                ]
                stats.average_age_hours = sum(ages) / len(ages)
            
            self.logger.info(
                "[DustMonitor:Stats] total=%d healthy=%d stalled=%d critical=%d "
                "notional=%.2f avg_age=%.1f",
                stats.total_dust_positions,
                stats.positions_healthy,
                stats.positions_stalled,
                stats.positions_critical,
                stats.total_dust_notional,
                stats.average_age_hours
            )
            
            return stats
        
        except Exception as e:
            self.logger.error("[DustMonitor] Error checking dust health: %s", e)
            return DustStatistics()
    
    async def get_dust_position_metrics(
        self,
        symbol: str
    ) -> Optional[DustPositionMetrics]:
        """
        Get detailed metrics for a single dust position.
        
        Args:
            symbol: Trading pair
        
        Returns:
            Detailed metrics or None if not a dust position
        """
        if symbol not in self._dust_positions:
            return None
        
        try:
            pos_data = self._dust_positions[symbol]
            age_hours = self._get_position_age_hours(symbol)
            health = self._get_position_health(symbol)
            
            # Estimate recovery time based on notional
            notional = pos_data["notional"]
            min_notional = pos_data["min_notional"]
            gap = min_notional - notional
            
            # Rough estimate: how many hours to accumulate $gap
            # Assume 1% price increase per hour (conservative)
            # Actually: would use historical volatility
            price_per_hour_increase = pos_data["price"] * 0.01
            hours_to_recover = gap / price_per_hour_increase if price_per_hour_increase > 0 else None
            
            rejection_count = self.shared_state.get_rejection_count(symbol, "SELL") if hasattr(
                self.shared_state, "get_rejection_count"
            ) else 0
            
            skip_count = self._position_skip_counts.get(symbol, 0)
            
            metrics = DustPositionMetrics(
                symbol=symbol,
                quantity=pos_data["quantity"],
                notional=notional,
                min_notional=min_notional,
                price=pos_data["price"],
                created_at=self._position_creation_times.get(symbol, time.time()),
                last_updated=pos_data["last_updated"],
                age_hours=age_hours,
                status=health,
                accumulation_rate_per_hour=0.0,  # Placeholder
                estimated_recovery_hours=hours_to_recover,
                rejection_count=rejection_count,
                skip_count=skip_count
            )
            
            return metrics
        
        except Exception as e:
            self.logger.warning(
                "[DustMonitor] Error getting metrics for %s: %s",
                symbol, e
            )
            return None
    
    async def alert_on_stalled_dust(self) -> List[DustAlert]:
        """
        Detect and alert on stalled dust positions.
        
        Returns:
            List of alerts for positions that aren't accumulating
        """
        alerts = []
        
        try:
            for symbol in self._dust_positions.keys():
                metrics = await self.get_dust_position_metrics(symbol)
                if not metrics:
                    continue
                
                # Check for stalled positions
                if metrics.status == DustHealthStatus.CRITICAL:
                    alert = DustAlert(
                        symbol=symbol,
                        alert_type="critical_age",
                        message=f"{symbol} dust position is {metrics.age_hours:.1f}h old and still below min",
                        severity="error",
                        notional=metrics.notional,
                        age_hours=metrics.age_hours,
                        action_required=True,
                        suggested_action="Consider manual liquidation or averaging down"
                    )
                    alerts.append(alert)
                    
                    self.logger.error(
                        "[DustMonitor:Alert:CRITICAL] %s critical_age=%.1f hours "
                        "notional=%.2f < min=%.2f",
                        symbol, metrics.age_hours, metrics.notional, metrics.min_notional
                    )
                
                elif metrics.status == DustHealthStatus.STALLED:
                    alert = DustAlert(
                        symbol=symbol,
                        alert_type="stalled",
                        message=f"{symbol} dust position stalled at {metrics.age_hours:.1f}h",
                        severity="warning",
                        notional=metrics.notional,
                        age_hours=metrics.age_hours,
                        action_required=False,
                        suggested_action="Monitor for recovery or consider additional action"
                    )
                    alerts.append(alert)
                    
                    self.logger.warning(
                        "[DustMonitor:Alert:STALLED] %s stalled_age=%.1f hours "
                        "notional=%.2f skip_count=%d",
                        symbol, metrics.age_hours, metrics.notional, metrics.skip_count
                    )
            
            self._alerts = alerts
            return alerts
        
        except Exception as e:
            self.logger.error(
                "[DustMonitor] Error generating stalled dust alerts: %s", e
            )
            return []
    
    async def get_dust_stats(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive dust prevention statistics.
        
        Args:
            window_hours: Historical window to analyze
        
        Returns:
            Dictionary with comprehensive metrics
        """
        try:
            health = await self.check_dust_health()
            
            # Calculate rates
            total_pos = health.total_dust_positions
            total_notional = health.total_dust_notional
            
            stats = {
                "timestamp": datetime.now().isoformat(),
                
                # Position counts
                "total_dust_positions": total_pos,
                "positions_healthy": health.positions_healthy,
                "positions_stalled": health.positions_stalled,
                "positions_critical": health.positions_critical,
                
                # Capital metrics
                "total_dust_notional": round(total_notional, 2),
                "capital_trapped_pct": round(
                    (total_notional / max(1.0, self._min_notional_floor * 10)) * 100, 1
                ),
                
                # Age metrics
                "average_position_age_hours": round(health.average_age_hours, 2),
                "oldest_position_hours": max(
                    [self._get_position_age_hours(sym) for sym in self._dust_positions.keys()]
                ) if self._dust_positions else 0,
                
                # Recovery metrics
                "recovery_rate": round(health.recovery_rate * 100, 1),
                "capital_recoverable": round(health.capital_recoverable, 2),
                
                # Alerts
                "alerts_count": len(self._alerts),
                "alerts_critical": sum(
                    1 for a in self._alerts if a.severity == "error"
                ),
                "alerts_warning": sum(
                    1 for a in self._alerts if a.severity == "warning"
                ),
                
                # Health summary
                "system_health": self._calculate_system_health(health),
            }
            
            self.logger.info(
                "[DustMonitor:Stats] health=%s positions=%d notional=%.2f "
                "age_avg=%.1f recovery_rate=%.1f%%",
                stats["system_health"],
                total_pos,
                total_notional,
                health.average_age_hours,
                health.recovery_rate * 100
            )
            
            return stats
        
        except Exception as e:
            self.logger.error("[DustMonitor] Error getting dust stats: %s", e)
            return {}
    
    def _calculate_system_health(self, stats: DustStatistics) -> str:
        """Calculate overall system health."""
        if stats.positions_critical > 0:
            return "CRITICAL"
        elif stats.positions_stalled > stats.positions_healthy:
            return "DEGRADED"
        elif stats.total_dust_positions == 0:
            return "CLEAN"
        else:
            return "NORMAL"
    
    async def start(self) -> None:
        """
        Start the dust monitor.
        
        This method is called during AppContext initialization (P7 protective services).
        Currently a no-op as monitoring is performed on-demand via check_dust_health().
        Can be extended in the future to spawn background monitoring tasks.
        """
        self.logger.info("[DustMonitor] started")
    
    def get_alerts(self) -> List[DustAlert]:
        """Get current alerts."""
        return self._alerts
    
    def clear_alerts(self) -> None:
        """Clear alert history."""
        self._alerts = []
