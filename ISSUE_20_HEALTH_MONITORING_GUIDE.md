# Issue #20: Health Monitoring Endpoints - Implementation Guide

**Status:** 🚀 READY TO START  
**Priority:** HIGH  
**Estimated Effort:** 2 hours  
**Expected Tests:** 5-8 (target 100%)  
**Timeline:** Friday, April 11, 2026 (9 AM - 12 PM)  
**Prerequisite:** Issue #19 (APM) - ✅ COMPLETE

---

## Overview

This issue adds comprehensive health monitoring endpoints to the trading bot for Kubernetes integration, liveness detection, and operational monitoring.

**Current Status:** 
- ❌ No health endpoints
- ❌ No Kubernetes probe support
- ❌ No component status visibility
- ❌ No automated health tracking

**After Implementation:**
- ✅ `/health` - Overall system health
- ✅ `/ready` - Readiness probe (Kubernetes)
- ✅ `/live` - Liveness probe (Kubernetes)
- ✅ Component status aggregation
- ✅ Prometheus health metrics export

---

## Task Breakdown

### Task 1: Health Endpoint Infrastructure (40 minutes)

**Goal:** Create health check framework in FastAPI

**Location:** `core/health_monitor.py` (NEW)

**Implementation:**
```python
class HealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    """Individual component health."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    latency_ms: float

class HealthMonitor:
    """Centralized health monitoring."""
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.last_update = time.time()
    
    def set_component_health(self, name: str, status: HealthStatus, message: str = ""):
        """Update component health status."""
        ...
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health."""
        # DEGRADED if any component degraded
        # UNHEALTHY if any component unhealthy
        # HEALTHY if all components healthy
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for API response."""
        ...
```

**Components to Monitor:**
- ✅ Database connection (SQLite)
- ✅ Exchange API (Binance)
- ✅ Market data feed (WebSocket)
- ✅ Jaeger APM backend (from Issue #19)
- ✅ Bootstrap module readiness
- ✅ Signal manager status

**Validation:**
- Test with missing components
- Test with degraded components
- Verify cascade logic (1 unhealthy = system unhealthy)

### Task 2: Kubernetes Probes (40 minutes)

**Goal:** Implement `/health`, `/ready`, `/live` endpoints

**Location:** `main.py` (MODIFIED)

**Implementation Pattern:**
```python
# In main.py or new routes.py
from fastapi import FastAPI, HTTPException

@app.get("/health", tags=["health"])
async def health() -> Dict[str, Any]:
    """
    Overall system health check.
    
    Returns:
    - status: "healthy" | "degraded" | "unhealthy"
    - timestamp: Current UTC timestamp
    - components: {component_name: {status, message}}
    - uptime_seconds: Time since startup
    - version: Trading bot version
    """
    monitor = get_health_monitor()
    return {
        "status": monitor.get_overall_status().value,
        "timestamp": time.time(),
        "components": {
            name: {
                "status": comp.status.value,
                "message": comp.message,
                "latency_ms": comp.latency_ms
            }
            for name, comp in monitor.components.items()
        },
        "uptime_seconds": time.time() - START_TIME,
        "version": "1.0.0"
    }

@app.get("/ready", tags=["health"])
async def ready() -> Dict[str, bool]:
    """
    Kubernetes readiness probe.
    
    Returns 200 if system is ready to serve traffic.
    Returns 503 if not ready.
    
    Conditions for READY:
    - All critical components HEALTHY
    - Bootstrap complete or not required
    - APM backend reachable
    - Database connected
    """
    monitor = get_health_monitor()
    
    # Check critical components
    critical = ["database", "exchange_api", "jaeger"]
    for component in critical:
        if component in monitor.components:
            comp = monitor.components[component]
            if comp.status == HealthStatus.UNHEALTHY:
                raise HTTPException(status_code=503, detail=f"{component} unhealthy")
    
    return {"ready": True}

@app.get("/live", tags=["health"])
async def live() -> Dict[str, bool]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if process is alive.
    Returns 503 if process should be restarted.
    
    Conditions for LIVE:
    - Process running
    - Event loop responsive
    - Memory not exhausted
    - No deadlocks detected
    """
    # Check if main loop is responsive
    if not hasattr(app.state, 'heartbeat_time'):
        raise HTTPException(status_code=503, detail="Heartbeat not initialized")
    
    last_beat = getattr(app.state, 'heartbeat_time', 0)
    now = time.time()
    
    # Liveness timeout: 30 seconds without heartbeat = dead
    if now - last_beat > 30.0:
        raise HTTPException(status_code=503, detail="Heartbeat timeout")
    
    return {"live": True}
```

**Expected Responses:**

```
GET /health
───────────────────────────────────────────────
200 OK
{
  "status": "healthy",
  "timestamp": 1712756400.123,
  "components": {
    "database": {"status": "healthy", "message": "", "latency_ms": 2.3},
    "exchange_api": {"status": "healthy", "message": "Binance connected", "latency_ms": 150},
    "jaeger": {"status": "healthy", "message": "APM backend reachable", "latency_ms": 5}
  },
  "uptime_seconds": 3600,
  "version": "1.0.0"
}

GET /ready
───────────────────────────────────────────────
200 OK
{"ready": true}

OR

503 Service Unavailable
{"detail": "database unhealthy"}

GET /live
───────────────────────────────────────────────
200 OK
{"live": true}

OR

503 Service Unavailable
{"detail": "Heartbeat timeout"}
```

**Validation:**
- Test all three endpoints
- Verify response codes
- Verify response format
- Test with degraded components
- Test timeout scenarios

### Task 3: Integration with Metrics (30 minutes)

**Goal:** Export health metrics to Prometheus

**Location:** `core/health_monitor.py` (ENHANCED)

**Implementation:**
```python
# In HealthMonitor class
def update_prometheus_metrics(self):
    """Export health status as Prometheus metrics."""
    # Create metrics if not exist
    if not hasattr(self, '_health_gauge'):
        from prometheus_client import Gauge
        self._health_gauge = Gauge(
            'system_health_status',
            'System health status (1=healthy, 2=degraded, 3=unhealthy)'
        )
        self._component_health = Gauge(
            'component_health_status',
            'Component health status',
            ['component']
        )
    
    # Update metrics
    status_value = {
        HealthStatus.HEALTHY: 1,
        HealthStatus.DEGRADED: 2,
        HealthStatus.UNHEALTHY: 3
    }
    
    self._health_gauge.set(status_value[self.get_overall_status()])
    
    for name, comp in self.components.items():
        self._component_health.labels(component=name).set(
            status_value[comp.status]
        )
```

**Prometheus Queries:**
```
# Overall health
system_health_status == 1  # Healthy

# Component health
component_health_status{component="database"} == 1

# Alert on degraded system
alert: SystemDegraded
  if: system_health_status > 1
  for: 2m
  annotations:
    summary: "System health degraded"
```

### Task 4: Heartbeat Integration (20 minutes)

**Goal:** Track main loop heartbeat for liveness detection

**Location:** `main.py` (run_loop method)

**Implementation:**
```python
async def run_loop(self):
    """Main event loop with heartbeat tracking."""
    app.state.heartbeat_time = time.time()
    
    while self.running:
        try:
            # Main logic
            await self.meta_controller.evaluate_and_act()
            
            # Update heartbeat
            app.state.heartbeat_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Loop error: {e}")
            app.state.heartbeat_time = time.time()  # Still alive, just errored
```

**Validation:**
- Heartbeat updates correctly
- Liveness probe detects timeout
- Liveness probe recovers when heartbeat resumes

---

## Test Requirements

### Test File: `tests/test_health_monitoring.py`

**Test Coverage (8 tests):**

1. **Test HealthMonitor initialization**
   ```python
   def test_health_monitor_initialization():
       monitor = HealthMonitor()
       assert monitor.get_overall_status() == HealthStatus.HEALTHY
   ```

2. **Test component health updates**
   ```python
   def test_set_component_health():
       monitor = HealthMonitor()
       monitor.set_component_health("database", HealthStatus.DEGRADED, "Slow")
       assert monitor.components["database"].status == HealthStatus.DEGRADED
   ```

3. **Test overall health cascade**
   ```python
   def test_overall_health_cascade():
       # Add healthy components
       monitor = HealthMonitor()
       monitor.set_component_health("db", HealthStatus.HEALTHY)
       monitor.set_component_health("api", HealthStatus.DEGRADED)
       # Overall should be DEGRADED
       assert monitor.get_overall_status() == HealthStatus.DEGRADED
   ```

4. **Test /health endpoint**
   ```python
   @pytest.mark.asyncio
   async def test_health_endpoint():
       response = client.get("/health")
       assert response.status_code == 200
       assert "status" in response.json()
       assert "components" in response.json()
   ```

5. **Test /ready endpoint (ready)**
   ```python
   @pytest.mark.asyncio
   async def test_ready_endpoint_success():
       # Setup all components as healthy
       response = client.get("/ready")
       assert response.status_code == 200
       assert response.json()["ready"] == True
   ```

6. **Test /ready endpoint (not ready)**
   ```python
   @pytest.mark.asyncio
   async def test_ready_endpoint_failure():
       # Simulate unhealthy database
       response = client.get("/ready")
       assert response.status_code == 503
   ```

7. **Test /live endpoint (alive)**
   ```python
   @pytest.mark.asyncio
   async def test_live_endpoint_success():
       app.state.heartbeat_time = time.time()
       response = client.get("/live")
       assert response.status_code == 200
   ```

8. **Test /live endpoint (dead)**
   ```python
   @pytest.mark.asyncio
   async def test_live_endpoint_timeout():
       app.state.heartbeat_time = time.time() - 60  # 60s ago
       response = client.get("/live")
       assert response.status_code == 503
   ```

---

## Success Criteria

✅ **All endpoints functional:**
- `/health` returns complete system status
- `/ready` returns Kubernetes-compatible response
- `/live` detects heartbeat timeout

✅ **All tests passing:**
- 8/8 new tests passing
- Integration tests with existing code
- No regressions

✅ **Prometheus integration:**
- Metrics exported correctly
- Grafana dashboard updated
- Alerts configured

✅ **Documentation complete:**
- API documentation
- Kubernetes configuration
- Monitoring guide

---

## Deployment Configuration

### Kubernetes Probes

```yaml
# In deployment spec
containers:
- name: trading-bot
  image: octavault/trader:latest
  livenessProbe:
    httpGet:
      path: /live
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: /ready
      port: 8000
    initialDelaySeconds: 10
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 3
```

---

## Timeline (Friday 9 AM - 12 PM)

```
9:00 - 9:15  Implement HealthMonitor class
9:15 - 9:40  Add endpoints (/health, /ready, /live)
9:40 - 10:00 Integrate heartbeat tracking
10:00 - 10:15 Add Prometheus metrics export
10:15 - 11:00 Write 8 test cases
11:00 - 11:30 Run tests & fix issues
11:30 - 12:00 Documentation & Kubernetes config
```

---

## Next Issue (After #20)

**Sprint 2:** Performance optimization and advanced monitoring
- Issue #21: MetaController refactoring
- Issue #22: Performance profiling
- Issue #23: Database optimization
- Issue #24: Cache optimization
- Issue #25: Multi-instance coordination

---

## Quick Reference

**Files to Create/Modify:**
- ✏️ `core/health_monitor.py` (NEW - 200 lines)
- ✏️ `main.py` (MODIFIED - endpoints + heartbeat)
- ✏️ `tests/test_health_monitoring.py` (NEW - 300 lines)
- ✏️ `deployment/kubernetes-health.yaml` (NEW)

**Dependencies:**
- ✅ FastAPI (already installed)
- ✅ Prometheus client (from Issue #16)
- ✅ APM infrastructure (from Issue #19)

**Estimated Code Lines:**
- HealthMonitor: 200 lines
- Endpoints: 100 lines
- Tests: 300 lines
- Configuration: 50 lines
- **Total: ~650 lines**

---

**Ready to implement! Starting Friday 9 AM**
