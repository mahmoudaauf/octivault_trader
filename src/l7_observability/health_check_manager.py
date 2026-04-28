# -*- coding: utf-8 -*-
"""
HealthCheckManager - Startup and Continuous Health Verification

Responsibility:
- Run health checks on all critical components
- Block startup if critical checks fail
- Monitor component health continuously
- Generate health reports for observability

This module implements health checks to address the
"No Health Checks on Startup" issue identified in Phase 2 review.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


class HealthCheckManager:
    """Manages health checks for all system components."""
    
    def __init__(self, app_context):
        """
        Initialize health check manager.
        
        Args:
            app_context: AppContext instance with all components
        """
        self.app_context = app_context
        self.logger = logging.getLogger("HealthCheckManager")
        
        self.results: List[HealthCheckResult] = []
        self.last_check_time: Optional[datetime] = None
    
    async def check_all_critical(self) -> bool:
        """
        Run all critical health checks.
        
        Returns:
            True if all critical checks pass, False otherwise
        """
        self.logger.info("🏥 Starting critical health checks...")
        
        checks = [
            ("DatabaseManager", self._check_database),
            ("ExchangeClient", self._check_exchange_client),
            ("SharedState", self._check_shared_state),
            ("AgentManager", self._check_agent_manager),
            ("MarketDataFeed", self._check_market_data),
        ]
        
        results = []
        
        for component_name, check_func in checks:
            try:
                start_time = datetime.now(timezone.utc)
                result = await check_func()
                duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                result.duration_ms = duration
                
                results.append(result)
                self.results.append(result)
                
                # Log result
                status_icon = "✅" if result.status == HealthStatus.HEALTHY else (
                    "⚠️" if result.status == HealthStatus.DEGRADED else "❌"
                )
                
                self.logger.info(
                    f"{status_icon} {result.component}: {result.message} "
                    f"({result.duration_ms:.0f}ms)"
                )
                
                # Fail fast on critical components
                if result.status == HealthStatus.UNHEALTHY:
                    self.logger.error(
                        f"❌ CRITICAL: {result.component} is unhealthy. "
                        f"Blocking startup."
                    )
                    return False
                
            except Exception as e:
                self.logger.error(
                    f"❌ Health check for {component_name} raised exception: {e}"
                )
                return False
        
        self.last_check_time = datetime.now(timezone.utc)
        
        # All critical checks passed
        self.logger.info(
            "✅ All critical health checks passed. System ready to start."
        )
        
        return True
    
    async def check_all_optional(self) -> Dict[str, HealthCheckResult]:
        """
        Run optional health checks (don't block startup).
        
        Returns:
            Dict of component -> HealthCheckResult
        """
        self.logger.info("🏥 Starting optional health checks...")
        
        checks = {
            "Logger": self._check_logging,
            "Configuration": self._check_configuration,
            "DiskSpace": self._check_disk_space,
        }
        
        results = {}
        
        for component_name, check_func in checks.items():
            try:
                result = await check_func()
                results[component_name] = result
                
                status_icon = "✅" if result.status == HealthStatus.HEALTHY else "⚠️"
                self.logger.debug(
                    f"{status_icon} {component_name}: {result.message}"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Optional health check {component_name} failed: {e}"
                )
        
        return results
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and responsiveness."""
        try:
            db_manager = self.app_context.database_manager
            
            if not db_manager:
                return HealthCheckResult(
                    component="DatabaseManager",
                    status=HealthStatus.UNHEALTHY,
                    message="DatabaseManager not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Try a simple ping/query
            if hasattr(db_manager, 'ping'):
                await db_manager.ping()
            elif hasattr(db_manager, 'test_connection'):
                await db_manager.test_connection()
            else:
                # Fallback: try to access database
                _ = db_manager.connection
            
            return HealthCheckResult(
                component="DatabaseManager",
                status=HealthStatus.HEALTHY,
                message="Database connected and responsive",
                timestamp=datetime.now(timezone.utc),
                details={'status': 'connected'},
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="DatabaseManager",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc),
                details={'error': str(e)},
            )
    
    async def _check_exchange_client(self) -> HealthCheckResult:
        """Check exchange API connectivity and authentication."""
        try:
            exchange = self.app_context.exchange_client
            
            if not exchange:
                return HealthCheckResult(
                    component="ExchangeClient",
                    status=HealthStatus.UNHEALTHY,
                    message="ExchangeClient not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Get account info (validates API key and connectivity)
            if hasattr(exchange, 'get_account'):
                account = await exchange.get_account()
            else:
                account = exchange.get_account()
            
            if not account:
                return HealthCheckResult(
                    component="ExchangeClient",
                    status=HealthStatus.UNHEALTHY,
                    message="No account data returned from exchange",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Check account has balances
            balances = account.get('balances', [])
            
            return HealthCheckResult(
                component="ExchangeClient",
                status=HealthStatus.HEALTHY,
                message=f"Exchange connected ({len(balances)} assets available)",
                timestamp=datetime.now(timezone.utc),
                details={
                    'status': 'connected',
                    'assets_count': len(balances),
                },
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="ExchangeClient",
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange check failed: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc),
                details={'error': str(e)},
            )
    
    async def _check_shared_state(self) -> HealthCheckResult:
        """Check shared state initialization."""
        try:
            shared_state = self.app_context.shared_state
            
            if not shared_state:
                return HealthCheckResult(
                    component="SharedState",
                    status=HealthStatus.UNHEALTHY,
                    message="SharedState not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Check required state is populated
            nav = getattr(shared_state, 'nav', None)
            if nav is None or nav <= 0:
                return HealthCheckResult(
                    component="SharedState",
                    status=HealthStatus.DEGRADED,
                    message="NAV not initialized or zero",
                    timestamp=datetime.now(timezone.utc),
                    details={'nav': nav},
                )
            
            # Check positions structure
            positions = getattr(shared_state, 'positions', None)
            if positions is None:
                return HealthCheckResult(
                    component="SharedState",
                    status=HealthStatus.DEGRADED,
                    message="Positions not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            return HealthCheckResult(
                component="SharedState",
                status=HealthStatus.HEALTHY,
                message=f"State initialized (NAV: ${nav:.2f}, {len(positions)} positions)",
                timestamp=datetime.now(timezone.utc),
                details={'nav': nav, 'position_count': len(positions)},
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="SharedState",
                status=HealthStatus.UNHEALTHY,
                message=f"State check failed: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc),
                details={'error': str(e)},
            )
    
    async def _check_agent_manager(self) -> HealthCheckResult:
        """Check agent manager initialization."""
        try:
            agent_mgr = self.app_context.agent_manager
            
            if not agent_mgr:
                return HealthCheckResult(
                    component="AgentManager",
                    status=HealthStatus.UNHEALTHY,
                    message="AgentManager not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Count registered agents
            agents = getattr(agent_mgr, 'agents', [])
            agent_count = len(agents)
            
            if agent_count == 0:
                return HealthCheckResult(
                    component="AgentManager",
                    status=HealthStatus.DEGRADED,
                    message="No agents registered yet",
                    timestamp=datetime.now(timezone.utc),
                    details={'agent_count': 0},
                )
            
            return HealthCheckResult(
                component="AgentManager",
                status=HealthStatus.HEALTHY,
                message=f"{agent_count} agents registered and ready",
                timestamp=datetime.now(timezone.utc),
                details={'agent_count': agent_count},
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="AgentManager",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent check failed: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc),
                details={'error': str(e)},
            )
    
    async def _check_market_data(self) -> HealthCheckResult:
        """Check market data feed initialization."""
        try:
            market_data = self.app_context.market_data_feed
            
            if not market_data:
                return HealthCheckResult(
                    component="MarketDataFeed",
                    status=HealthStatus.UNHEALTHY,
                    message="MarketDataFeed not initialized",
                    timestamp=datetime.now(timezone.utc),
                )
            
            # Check if any prices are loaded
            latest_prices = getattr(market_data, 'latest_prices', {})
            price_count = len(latest_prices)
            
            if price_count == 0:
                return HealthCheckResult(
                    component="MarketDataFeed",
                    status=HealthStatus.DEGRADED,
                    message="No market prices available yet (loading...)",
                    timestamp=datetime.now(timezone.utc),
                    details={'prices_loaded': 0},
                )
            
            return HealthCheckResult(
                component="MarketDataFeed",
                status=HealthStatus.HEALTHY,
                message=f"Market data available for {price_count} symbols",
                timestamp=datetime.now(timezone.utc),
                details={'symbols_with_prices': price_count},
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="MarketDataFeed",
                status=HealthStatus.UNHEALTHY,
                message=f"Market data check failed: {str(e)[:100]}",
                timestamp=datetime.now(timezone.utc),
                details={'error': str(e)},
            )
    
    async def _check_logging(self) -> HealthCheckResult:
        """Check logging system."""
        try:
            # Verify logger is configured
            test_logger = logging.getLogger("HealthCheck.Test")
            test_logger.debug("Health check test message")
            
            return HealthCheckResult(
                component="Logger",
                status=HealthStatus.HEALTHY,
                message="Logging system operational",
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            return HealthCheckResult(
                component="Logger",
                status=HealthStatus.UNHEALTHY,
                message=f"Logging check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )
    
    async def _check_configuration(self) -> HealthCheckResult:
        """Check configuration loading."""
        try:
            # Verify key config values are set
            config = self.app_context.config
            
            if not config:
                return HealthCheckResult(
                    component="Configuration",
                    status=HealthStatus.UNHEALTHY,
                    message="Configuration not loaded",
                    timestamp=datetime.now(timezone.utc),
                )
            
            return HealthCheckResult(
                component="Configuration",
                status=HealthStatus.HEALTHY,
                message="Configuration loaded successfully",
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            return HealthCheckResult(
                component="Configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            import shutil
            import os
            
            # Check disk space in current directory
            stat = shutil.disk_usage(".")
            percent_free = (stat.free / stat.total) * 100
            
            if percent_free < 5:
                status = HealthStatus.UNHEALTHY
            elif percent_free < 10:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                component="DiskSpace",
                status=status,
                message=f"Disk {percent_free:.1f}% free ({stat.free / 1e9:.1f}GB available)",
                timestamp=datetime.now(timezone.utc),
                details={
                    'percent_free': percent_free,
                    'free_gb': stat.free / 1e9,
                    'total_gb': stat.total / 1e9,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                component="DiskSpace",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
            )
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        critical_results = [r for r in self.results if r.component in [
            "DatabaseManager", "ExchangeClient", "SharedState"
        ]]
        
        unhealthy = [r for r in critical_results if r.status == HealthStatus.UNHEALTHY]
        degraded = [r for r in critical_results if r.status == HealthStatus.DEGRADED]
        healthy = [r for r in critical_results if r.status == HealthStatus.HEALTHY]
        
        return {
            'overall_status': (
                HealthStatus.UNHEALTHY.value if unhealthy else (
                    HealthStatus.DEGRADED.value if degraded else HealthStatus.HEALTHY.value
                )
            ),
            'last_check': self.last_check_time,
            'summary': {
                'healthy': len(healthy),
                'degraded': len(degraded),
                'unhealthy': len(unhealthy),
            },
            'details': {r.component: r for r in self.results},
        }
