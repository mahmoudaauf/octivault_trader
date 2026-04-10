# Issue #19: APM Instrumentation (Jaeger) - Implementation Guide

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH  
**Estimated Effort:** 4-6 hours  
**Expected Tests:** 8/8 (100%)  
**Deadline:** Wednesday, April 10, 2026

---

## Overview

This issue adds Jaeger/OpenTelemetry distributed tracing to the trading bot system for complete request tracing, bottleneck identification, and performance monitoring across all guards and trading operations.

**Current Status:** ✅ Infrastructure ready, need integration into core modules

---

## Implementation Tasks

### Task 1: Integrate Tracing into MetaController (Target: 2 hours)

**File:** `core/meta_controller.py`  
**Target:** Add tracing to main decision-making loop

**Changes Required:**
1. Import APM instrumentation
2. Wrap `evaluate_and_act()` main loop with trace spans
3. Add guard evaluation tracing (7 guards)
4. Track signal processing flow
5. Monitor execution decisions

**Validation:** 3 new test files (guard tracing, loop iteration, trade flow)

### Task 2: Integrate Tracing into Guard System (Target: 1.5 hours)

**Files:** `core/integration_guard.py`, guard implementations  
**Target:** Trace each guard evaluation with detailed attributes

**Changes Required:**
1. Wrap each guard's evaluation with `trace_guard_evaluation()`
2. Add rejection reason tracking
3. Monitor approval/rejection rates per guard
4. Track guard latency metrics

**Guards to Instrument:**
- ✅ balance_guard
- ✅ leverage_guard
- ✅ hours_guard
- ✅ anomaly_guard
- ✅ correlation_guard
- ✅ execution_guard
- ✅ concentration_guard

**Validation:** 2 test files (guard tracing attributes, multi-guard correlation)

### Task 3: Integrate Tracing into Execution Manager (Target: 1 hour)

**File:** `core/execution_manager.py`  
**Target:** Trace order placement and settlement

**Changes Required:**
1. Trace trade execution flow
2. Add order submission tracking
3. Monitor fill confirmation
4. Track execution latency

**Validation:** 2 test files (execution flow, fill confirmation)

### Task 4: Set Up Jaeger Backend Service (Target: 30 minutes)

**Files:** `deployment/jaeger-deployment.yaml`, `docker-compose.yml`  
**Target:** Make Jaeger discoverable by trading bot

**Changes Required:**
1. Update docker-compose to include Jaeger service
2. Configure OpenTelemetry environment variables
3. Add health check for Jaeger availability
4. Update service discovery

**Validation:** 1 test file (Jaeger connectivity)

### Task 5: Create Trace Visualization Dashboard (Target: 30 minutes)

**Files:** `dashboards/jaeger_apm_dashboard.json`  
**Target:** Pre-configured dashboard for Grafana

**Content:**
- Trade flow visualization
- Guard evaluation timeline
- Execution latency heatmap
- Error rate tracking by span type

**Validation:** Dashboard loads in Grafana without errors

---

## Implementation Checklist

### Phase 1: MetaController Instrumentation
- [ ] Add APM instrument imports to `meta_controller.py`
- [ ] Initialize APM instrument in `__init__()`
- [ ] Wrap main loop iteration with trace context
- [ ] Add guard evaluation tracing (with results)
- [ ] Add signal processing tracing
- [ ] Add trade decision tracing
- [ ] Write 3 test files for coverage
- [ ] Validate trace attribute completeness

### Phase 2: Guard System Instrumentation  
- [ ] Add tracing to balance_guard
- [ ] Add tracing to leverage_guard
- [ ] Add tracing to hours_guard
- [ ] Add tracing to anomaly_guard
- [ ] Add tracing to correlation_guard
- [ ] Add tracing to execution_guard
- [ ] Add tracing to concentration_guard
- [ ] Track approval/rejection metrics
- [ ] Write 2 test files for coverage
- [ ] Verify cross-guard correlation tracing

### Phase 3: ExecutionManager Instrumentation
- [ ] Add tracing to order submission
- [ ] Add tracing to order confirmation
- [ ] Track execution latency
- [ ] Monitor position management
- [ ] Write 2 test files for coverage
- [ ] Verify end-to-end execution flow

### Phase 4: Backend & Deployment
- [ ] Configure Jaeger in docker-compose
- [ ] Set OpenTelemetry environment variables
- [ ] Add Jaeger health check
- [ ] Update deployment docs
- [ ] Write 1 test file for Jaeger connectivity

### Phase 5: Visualization
- [ ] Create Jaeger dashboards
- [ ] Configure Grafana panel links
- [ ] Document trace interpretation
- [ ] Validate dashboard functionality

---

## Code Examples

### Example 1: MetaController Loop Instrumentation

```python
# In core/meta_controller.py
from core.apm_instrument import get_apm_instrument

class MetaController:
    def __init__(self, ...):
        ...
        self.apm = get_apm_instrument()
        self.cycle_number = 0
    
    async def evaluate_and_act(self):
        """Main controller loop with tracing."""
        self.cycle_number += 1
        
        async with self.apm.tracer.span(
            "evaluate_and_act",
            {
                "cycle.number": self.cycle_number,
                "symbols.count": len(self._position_state),
            }
        ) as span:
            try:
                # Guard evaluations...
                results = await self._evaluate_guards()
                
                # Signal processing...
                signals = await self._process_signals()
                
                # Trading decisions...
                decisions = await self._make_decisions(results, signals)
                
                return decisions
                
            except Exception as e:
                self.apm.tracer.set_span_status_error(span, str(e))
                raise
```

### Example 2: Guard Instrumentation

```python
# In core/integration_guard.py
async def _evaluate_balance_guard(self, symbol: str, ...):
    """Trace balance guard evaluation."""
    coro = self._do_evaluate_balance_guard(symbol, ...)
    
    result = await self.apm.trace_guard_evaluation(
        guard_name="balance_guard",
        symbol=symbol,
        coro=coro,
        additional_attrs={
            "balance.threshold": self.balance_threshold,
            "required.balance": required_balance,
            "current.balance": current_balance,
        }
    )
    
    return result
```

### Example 3: ExecutionManager Instrumentation

```python
# In core/execution_manager.py
async def place_order(self, order: Order, ...):
    """Trace order placement."""
    async def _do_place():
        order_id = await self._submit_order(order)
        confirmation = await self._wait_confirmation(order_id)
        return confirmation
    
    result = await self.apm.trace_execution(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        coro=_do_place(),
        metadata={
            "price": order.price,
            "client_id": order.client_order_id,
        }
    )
    
    return result
```

---

## Test Requirements

### Test File 1: MetaController Tracing (test_meta_controller_tracing.py)

```python
@pytest.mark.asyncio
async def test_evaluate_and_act_creates_span():
    """Verify evaluate_and_act creates Jaeger span."""
    # Setup
    # Execute
    # Assert: Span created with correct attributes

@pytest.mark.asyncio
async def test_guard_evaluation_tracing():
    """Verify guard evaluations are traced."""
    # Setup
    # Execute
    # Assert: 7 guard spans created with proper hierarchy

@pytest.mark.asyncio
async def test_signal_processing_tracing():
    """Verify signal flow is traced."""
    # Setup
    # Execute
    # Assert: Signal processing spans captured
```

### Test File 2: Guard Tracing (test_guard_tracing.py)

```python
@pytest.mark.asyncio
async def test_balance_guard_tracing():
    """Verify balance guard creates trace span."""
    # Setup
    # Execute
    # Assert: Guard span includes rejection_reason

@pytest.mark.asyncio
async def test_guard_correlation_in_traces():
    """Verify guard spans are correlated."""
    # Setup
    # Execute
    # Assert: Multiple guards share parent span
```

### Test File 3: Execution Tracing (test_execution_tracing.py)

```python
@pytest.mark.asyncio
async def test_order_placement_tracing():
    """Verify order placement is traced."""
    # Setup
    # Execute
    # Assert: Trade execution span created

@pytest.mark.asyncio
async def test_execution_latency_tracking():
    """Verify execution latency is recorded."""
    # Setup
    # Execute
    # Assert: Span duration reflects actual latency
```

---

## Success Criteria

✅ **All guards traced** - Every guard evaluation generates a span  
✅ **End-to-end flow visible** - Full trade flow traceable in Jaeger UI  
✅ **Attributes captured** - All relevant metadata included in spans  
✅ **Performance impact minimal** - <100ms added per cycle  
✅ **8/8 tests passing** - 100% test coverage for tracing  
✅ **Dashboard functional** - Pre-configured visualization ready  

---

## Deployment

```bash
# 1. Update requirements.txt with OpenTelemetry packages
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-jaeger \
            opentelemetry-instrumentation

# 2. Start Jaeger backend
docker-compose up jaeger

# 3. Set environment variables
export OTEL_ENABLED=true
export OTEL_SERVICE_NAME=octivault-trader
export JAEGER_HOST=localhost
export JAEGER_PORT=6831

# 4. Start trading bot
python main.py

# 5. View traces in Jaeger UI
# Open http://localhost:16686
```

---

## Validation Steps

1. **Backend Connectivity** ✅
   - Jaeger service running
   - UDP port 6831 listening
   - Health check passing

2. **Span Creation** ✅
   - MetaController spans visible in UI
   - Guard spans properly nested
   - Execution spans tracked

3. **Attributes Completeness** ✅
   - All required attributes present
   - Guard decisions recorded
   - Latency accurate

4. **Performance** ✅
   - Trace overhead < 100ms/cycle
   - No memory leaks
   - Batch processor efficient

---

## Timeline

- **Start:** Wednesday morning (April 10)
- **MetaController integration:** 2 hours
- **Guard integration:** 1.5 hours  
- **Execution integration:** 1 hour
- **Backend setup:** 30 minutes
- **Visualization:** 30 minutes
- **Testing & validation:** 1 hour
- **Buffer:** 1 hour

**Total:** ~8 hours (doable in full day with breaks)

---

## Notes

- All infrastructure already in place (jaeger_tracer.py, apm_instrument.py, deployment YAML)
- Just need to integrate into core modules
- Tests ensure tracing works correctly
- Dashboard provides immediate visibility
- Can be deployed incrementally (feature flag available)

---

**Next:** Start with Task 1 - MetaController integration
