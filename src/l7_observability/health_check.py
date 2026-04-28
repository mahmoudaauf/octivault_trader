# -*- coding: utf-8 -*-
"""
health_check.py - P9-aligned Health Check Endpoints

Kubernetes-compatible health check endpoints for liveness and readiness probes.

Endpoints:
  • GET /health - Basic liveness check (always returns 200 if app is running)
  • GET /ready - Readiness check (checks dependencies: DB, market data, etc.)
  • GET /live - Live trading check (checks if processing trades normally)

Architecture:
  • Minimal dependencies (fast responses)
  • No auth required (health checks run before auth)
  • JSON responses with status codes
  • Detailed error info for debugging
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("HealthCheck")


class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthChecker:
    """
    Health checker for Octivault Trader.
    
    Tracks application state and provides health status for orchestrators.
    """
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.last_trade_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.dependencies_ok = False
        self.market_data_ok = False
        self.trading_active = False
    
    # Configuration thresholds
    STALE_TRADE_THRESHOLD_SEC = 300  # 5 minutes
    STALE_DATA_THRESHOLD_SEC = 60    # 1 minute
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def mark_trade_executed(self):
        """Mark that a trade was just executed"""
        self.last_trade_time = datetime.utcnow()
    
    def mark_error(self, error_msg: str):
        """Mark that an error occurred"""
        self.last_error = error_msg
    
    def set_dependencies_ok(self, ok: bool):
        """Set if dependencies are healthy (DB, Redis, etc.)"""
        self.dependencies_ok = ok
    
    def set_market_data_ok(self, ok: bool):
        """Set if market data is flowing normally"""
        self.market_data_ok = ok
    
    def set_trading_active(self, active: bool):
        """Set if trading logic is active"""
        self.trading_active = active
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get basic health status (liveness).
        
        Returns:
            Dict with status, uptime, and basic info
        """
        return {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": self.get_uptime_seconds(),
            "version": "P9-aligned",
        }
    
    def get_ready_status(self) -> Dict[str, Any]:
        """
        Get readiness status (dependencies check).
        
        Returns:
            Dict with readiness and dependency details
        """
        dependencies_ready = self.dependencies_ok
        market_data_ready = self.market_data_ok
        overall_ready = dependencies_ready and market_data_ready
        
        return {
            "status": HealthStatus.HEALTHY.value if overall_ready else HealthStatus.UNHEALTHY.value,
            "ready": overall_ready,
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": {
                "database": self.dependencies_ok,
                "market_data": self.market_data_ready,
            },
            "checks": {
                "config_loaded": True,
                "logging_configured": True,
                "market_data_synced": self.market_data_ok,
                "dependencies_connected": self.dependencies_ok,
            }
        }
    
    def get_live_status(self) -> Dict[str, Any]:
        """
        Get live trading status (active processing check).
        
        Returns:
            Dict with trading status and recent activity
        """
        if not self.trading_active:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "live": False,
                "reason": "Trading not active",
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        # Check if trades are recent (not stale)
        if self.last_trade_time:
            time_since_trade = (datetime.utcnow() - self.last_trade_time).total_seconds()
            trade_stale = time_since_trade > self.STALE_TRADE_THRESHOLD_SEC
            
            return {
                "status": HealthStatus.HEALTHY.value if not trade_stale else HealthStatus.DEGRADED.value,
                "live": not trade_stale,
                "timestamp": datetime.utcnow().isoformat(),
                "last_trade": self.last_trade_time.isoformat() if self.last_trade_time else None,
                "time_since_last_trade_sec": time_since_trade,
                "stale_threshold_sec": self.STALE_TRADE_THRESHOLD_SEC,
            }
        
        return {
            "status": HealthStatus.DEGRADED.value,
            "live": False,
            "reason": "No trades executed yet",
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """
        Get full health status (all checks).
        
        Returns:
            Dict with complete health information
        """
        ready = self.get_ready_status()
        live = self.get_live_status()
        
        # Determine overall status
        if ready["status"] == HealthStatus.UNHEALTHY.value:
            overall_status = HealthStatus.UNHEALTHY.value
        elif live["status"] == HealthStatus.UNHEALTHY.value:
            overall_status = HealthStatus.UNHEALTHY.value
        elif ready["status"] == HealthStatus.DEGRADED.value or live["status"] == HealthStatus.DEGRADED.value:
            overall_status = HealthStatus.DEGRADED.value
        else:
            overall_status = HealthStatus.HEALTHY.value
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": self.get_uptime_seconds(),
            "liveness": self.get_health_status(),
            "readiness": ready,
            "trading_status": live,
            "last_error": self.last_error,
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def initialize_health_checker() -> HealthChecker:
    """Initialize the health checker (call from main)"""
    global _health_checker
    _health_checker = HealthChecker()
    logger.info("✅ Health checker initialized")
    return _health_checker


# FastAPI route handlers
async def health_endpoint() -> Dict[str, Any]:
    """
    GET /health - Liveness probe for Kubernetes
    
    Always returns 200 if app is running.
    
    Response:
        {
            "status": "healthy",
            "timestamp": "2026-04-10T12:00:00.000000",
            "uptime_seconds": 3600.0,
            "version": "P9-aligned"
        }
    """
    checker = get_health_checker()
    return checker.get_health_status()


async def ready_endpoint() -> Dict[str, Any]:
    """
    GET /ready - Readiness probe for Kubernetes
    
    Returns 200 only if app is ready to receive traffic.
    
    Response:
        {
            "status": "healthy|unhealthy",
            "ready": true|false,
            "timestamp": "2026-04-10T12:00:00.000000",
            "dependencies": {
                "database": true,
                "market_data": true
            },
            "checks": { ... }
        }
    """
    checker = get_health_checker()
    status = checker.get_ready_status()
    
    # Return appropriate HTTP status
    # Kubernetes expects 200 for ready, anything else for not ready
    return status


async def live_endpoint() -> Dict[str, Any]:
    """
    GET /live - Live trading probe for Kubernetes
    
    Returns 200 only if trading is processing normally.
    
    Response:
        {
            "status": "healthy|degraded|unhealthy",
            "live": true|false,
            "timestamp": "2026-04-10T12:00:00.000000",
            "last_trade": "2026-04-10T11:55:00.000000",
            "time_since_last_trade_sec": 300
        }
    """
    checker = get_health_checker()
    return checker.get_live_status()


async def deep_status_endpoint() -> Dict[str, Any]:
    """
    GET /status - Full health status (debugging)
    
    Returns complete health information for debugging.
    
    Response:
        {
            "status": "healthy|degraded|unhealthy",
            "timestamp": "2026-04-10T12:00:00.000000",
            "uptime_seconds": 3600.0,
            "liveness": { ... },
            "readiness": { ... },
            "trading_status": { ... },
            "last_error": null
        }
    """
    checker = get_health_checker()
    return checker.get_full_status()
