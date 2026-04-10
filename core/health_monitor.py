# -*- coding: utf-8 -*-
"""
Health Check Endpoints - Issue #20

Kubernetes-compatible health monitoring endpoints for the trading bot.

Endpoints:
1. /health - Overall system health (JSON status)
2. /ready - Readiness probe (can accept traffic)
3. /live - Liveness probe (process is alive)

All endpoints return JSON with:
- status: "healthy", "degraded", "unhealthy"
- timestamp: ISO 8601 timestamp
- checks: Individual component status
- details: Additional information

Integration with:
- MetaController: Core trading logic health
- Guards: Safety guard status (5 guards)
- Database: Connection pool health
- APIs: Exchange connection status
- Memory: Resource usage monitoring
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import psutil
import os

logger = logging.getLogger(__name__)


class HealthCheckStatus:
    """Enumeration of health check statuses."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Individual component health status."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthCheckStatus.HEALTHY
        self.message = "OK"
        self.last_check = None
        self.check_count = 0
        self.error_count = 0
    
    def mark_healthy(self, message: str = "OK"):
        """Mark component as healthy."""
        self.status = HealthCheckStatus.HEALTHY
        self.message = message
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.check_count += 1
    
    def mark_degraded(self, message: str = "Degraded"):
        """Mark component as degraded."""
        self.status = HealthCheckStatus.DEGRADED
        self.message = message
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.check_count += 1
        self.error_count += 1
    
    def mark_unhealthy(self, message: str = "Unhealthy"):
        """Mark component as unhealthy."""
        self.status = HealthCheckStatus.UNHEALTHY
        self.message = message
        self.last_check = datetime.now(timezone.utc).isoformat()
        self.check_count += 1
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "last_check": self.last_check,
            "check_count": self.check_count,
            "error_count": self.error_count,
        }


class HealthMonitor:
    """Centralized health monitoring for all system components."""
    
    _instance: Optional['HealthMonitor'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Core components
        self.meta_controller = ComponentHealth("meta_controller")
        self.database = ComponentHealth("database")
        self.exchange_api = ComponentHealth("exchange_api")
        self.memory = ComponentHealth("memory")
        self.cpu = ComponentHealth("cpu")
        
        # Safety guards (5 guards)
        self.guard_balance = ComponentHealth("guard_balance")
        self.guard_leverage = ComponentHealth("guard_leverage")
        self.guard_hours = ComponentHealth("guard_hours")
        self.guard_anomaly = ComponentHealth("guard_anomaly")
        self.guard_correlation = ComponentHealth("guard_correlation")
        
        # Observability
        self.prometheus = ComponentHealth("prometheus_metrics")
        self.traces = ComponentHealth("traces")
        self.alerts = ComponentHealth("alerts")
        
        self.components = {
            # Core
            "meta_controller": self.meta_controller,
            "database": self.database,
            "exchange_api": self.exchange_api,
            "memory": self.memory,
            "cpu": self.cpu,
            # Guards
            "guard_balance": self.guard_balance,
            "guard_leverage": self.guard_leverage,
            "guard_hours": self.guard_hours,
            "guard_anomaly": self.guard_anomaly,
            "guard_correlation": self.guard_correlation,
            # Observability
            "prometheus": self.prometheus,
            "traces": self.traces,
            "alerts": self.alerts,
        }
        
        self.start_time = datetime.now(timezone.utc)
        self._initialized = True
    
    def check_memory(self, threshold_percent: float = 80.0) -> None:
        """Check memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            if memory_percent > threshold_percent:
                self.memory.mark_unhealthy(
                    f"Memory usage {memory_percent:.1f}% exceeds threshold {threshold_percent}%"
                )
            elif memory_percent > (threshold_percent * 0.8):
                self.memory.mark_degraded(
                    f"Memory usage {memory_percent:.1f}% is high"
                )
            else:
                self.memory.mark_healthy(f"Memory usage {memory_percent:.1f}%")
        except Exception as e:
            self.memory.mark_degraded(f"Error checking memory: {e}")
    
    def check_cpu(self, threshold_percent: float = 90.0) -> None:
        """Check CPU usage."""
        try:
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent(interval=0.1)
            
            if cpu_percent > threshold_percent:
                self.cpu.mark_unhealthy(
                    f"CPU usage {cpu_percent:.1f}% exceeds threshold {threshold_percent}%"
                )
            elif cpu_percent > (threshold_percent * 0.8):
                self.cpu.mark_degraded(
                    f"CPU usage {cpu_percent:.1f}% is high"
                )
            else:
                self.cpu.mark_healthy(f"CPU usage {cpu_percent:.1f}%")
        except Exception as e:
            self.cpu.mark_degraded(f"Error checking CPU: {e}")
    
    def update_component(self, component_name: str, status: str, message: str) -> None:
        """Update component health status."""
        if component_name not in self.components:
            logger.warning(f"Unknown component: {component_name}")
            return
        
        component = self.components[component_name]
        
        if status == HealthCheckStatus.HEALTHY:
            component.mark_healthy(message)
        elif status == HealthCheckStatus.DEGRADED:
            component.mark_degraded(message)
        elif status == HealthCheckStatus.UNHEALTHY:
            component.mark_unhealthy(message)
    
    def get_overall_status(self) -> str:
        """Determine overall system status."""
        # Critical components (any unhealthy = system unhealthy)
        critical = ["meta_controller", "database"]
        
        for comp_name in critical:
            if comp_name in self.components:
                if self.components[comp_name].status == HealthCheckStatus.UNHEALTHY:
                    return HealthCheckStatus.UNHEALTHY
        
        # Check for degraded components
        has_degraded = any(
            comp.status == HealthCheckStatus.DEGRADED
            for comp in self.components.values()
        )
        
        if has_degraded:
            return HealthCheckStatus.DEGRADED
        
        return HealthCheckStatus.HEALTHY
    
    def is_ready(self) -> bool:
        """Check if system is ready to accept traffic."""
        # Ready if: meta_controller + guards all healthy, database accessible
        required_ready = [
            "meta_controller",
            "database",
            "exchange_api",
            "guard_balance",
            "guard_leverage",
            "guard_hours",
        ]
        
        for comp_name in required_ready:
            if comp_name in self.components:
                comp = self.components[comp_name]
                if comp.status == HealthCheckStatus.UNHEALTHY:
                    return False
        
        return True
    
    def is_alive(self) -> bool:
        """Check if system process is alive."""
        # Alive if process is running (always true for this check)
        # In practice, this is checked at process level by Kubernetes
        return True
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        self.check_memory()
        self.check_cpu()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "status": self.get_overall_status(),
            "is_ready": self.is_ready(),
            "is_alive": self.is_alive(),
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "summary": {
                "total_checks": sum(comp.check_count for comp in self.components.values()),
                "total_errors": sum(comp.error_count for comp in self.components.values()),
                "healthy_components": sum(
                    1 for comp in self.components.values()
                    if comp.status == HealthCheckStatus.HEALTHY
                ),
                "degraded_components": sum(
                    1 for comp in self.components.values()
                    if comp.status == HealthCheckStatus.DEGRADED
                ),
                "unhealthy_components": sum(
                    1 for comp in self.components.values()
                    if comp.status == HealthCheckStatus.UNHEALTHY
                ),
            },
        }
    
    def get_readiness_report(self) -> Dict[str, Any]:
        """Generate readiness probe report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ready": self.is_ready(),
            "status": "ready" if self.is_ready() else "not_ready",
            "checks": {
                "meta_controller": self.meta_controller.status,
                "database": self.database.status,
                "exchange_api": self.exchange_api.status,
                "guards": {
                    "balance": self.guard_balance.status,
                    "leverage": self.guard_leverage.status,
                    "hours": self.guard_hours.status,
                },
            },
        }
    
    def get_liveness_report(self) -> Dict[str, Any]:
        """Generate liveness probe report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alive": self.is_alive(),
            "status": "alive",
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "pid": os.getpid(),
        }


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor."""
    return HealthMonitor()
