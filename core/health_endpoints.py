# -*- coding: utf-8 -*-
"""
Health Check HTTP Endpoints - Issue #20

FastAPI endpoints for Kubernetes health checks:
- GET /health - Overall health status (liveness + readiness)
- GET /ready - Readiness probe (can accept traffic)
- GET /live - Liveness probe (process is alive)

HTTP Status Codes:
- 200: Healthy/Ready/Alive
- 503: Unhealthy/Not Ready/Dead

JSON Response Format:
{
    "status": "healthy|degraded|unhealthy",
    "timestamp": "2026-04-10T20:30:00Z",
    "uptime_seconds": 3600,
    "checks": {...}
}
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, Response, status
from starlette.responses import JSONResponse
import json

try:
    from core.health_monitor import get_health_monitor
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthEndpoints:
    """Health check endpoints for Kubernetes integration."""
    
    def __init__(self, app: Optional[FastAPI] = None):
        self.app = app
        self.health_monitor = None
        
        if HEALTH_MONITOR_AVAILABLE:
            try:
                self.health_monitor = get_health_monitor()
            except Exception as e:
                logger.warning(f"Could not initialize health monitor: {e}")
    
    def register_endpoints(self, app: FastAPI) -> None:
        """Register health check endpoints with FastAPI app."""
        self.app = app
        
        app.add_api_route("/health", self.health_check, methods=["GET"])
        app.add_api_route("/ready", self.readiness_probe, methods=["GET"])
        app.add_api_route("/live", self.liveness_probe, methods=["GET"])
        app.add_api_route("/healthz", self.health_check, methods=["GET"])  # Kubernetes alias
        
        logger.info("✅ Health check endpoints registered")
    
    async def health_check(self, response: Response) -> Dict[str, Any]:
        """
        GET /health
        
        Overall health status endpoint.
        Returns 200 if healthy, 503 if degraded/unhealthy.
        """
        if not self.health_monitor:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "unhealthy",
                "message": "Health monitor not available",
            }
        
        try:
            report = self.health_monitor.get_health_report()
            
            if report["status"] == "unhealthy":
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            elif report["status"] == "degraded":
                # Degraded but still serving (200 response)
                response.status_code = status.HTTP_200_OK
            else:
                response.status_code = status.HTTP_200_OK
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "unhealthy",
                "message": f"Error checking health: {e}",
            }
    
    async def readiness_probe(self, response: Response) -> Dict[str, Any]:
        """
        GET /ready
        
        Kubernetes readiness probe endpoint.
        Returns 200 if ready to accept traffic, 503 if not.
        """
        if not self.health_monitor:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "ready": False,
                "status": "not_ready",
                "message": "Health monitor not available",
            }
        
        try:
            report = self.health_monitor.get_readiness_report()
            
            if report["ready"]:
                response.status_code = status.HTTP_200_OK
            else:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            
            return report
        
        except Exception as e:
            logger.error(f"Error checking readiness: {e}")
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "ready": False,
                "status": "not_ready",
                "message": f"Error checking readiness: {e}",
            }
    
    async def liveness_probe(self, response: Response) -> Dict[str, Any]:
        """
        GET /live
        
        Kubernetes liveness probe endpoint.
        Returns 200 if alive, 503 if dead.
        Triggers container restart if returns 503.
        """
        if not self.health_monitor:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "alive": False,
                "status": "dead",
                "message": "Health monitor not available",
            }
        
        try:
            report = self.health_monitor.get_liveness_report()
            
            if report["alive"]:
                response.status_code = status.HTTP_200_OK
            else:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            
            return report
        
        except Exception as e:
            logger.error(f"Error checking liveness: {e}")
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "alive": False,
                "status": "dead",
                "message": f"Error checking liveness: {e}",
            }


# Global instance
_health_endpoints = None


def get_health_endpoints() -> HealthEndpoints:
    """Get or create health endpoints instance."""
    global _health_endpoints
    if _health_endpoints is None:
        _health_endpoints = HealthEndpoints()
    return _health_endpoints


def register_health_endpoints(app: FastAPI) -> None:
    """Register health endpoints with FastAPI app."""
    endpoints = get_health_endpoints()
    endpoints.register_endpoints(app)
