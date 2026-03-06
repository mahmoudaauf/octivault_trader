# Phase 3: Dust Registry Lifecycle - Design & Implementation Plan

## Objective

Implement DustRegistry to track dust positions through their healing lifecycle and prevent repeated entry into healing for the same dust position.

## Root Problems Being Solved

**Dust Loop Root Issues #3, #4:**
- **#3**: Dust positions not properly tracked through lifecycle (creation → detection → healing attempt → resolution/cleanup)
- **#4**: Repeated entry into same healing strategy for the same dust position (circuit breaker needed)

## Design Overview

### DustRegistry Class

A persistent registry that tracks:
1. **Dust Position Lifecycle**: When dust was created, when healing started, healing progress
2. **Circuit Breaker**: Prevents repeated healing attempts for same position
3. **Status Tracking**: Current state of each dust position (NEW, HEALING, HEALED, ABANDONED)

### Data Structure

```json
{
  "dust_positions": {
    "{symbol}": {
      "quantity": 0.00123,
      "notional_usd": 0.98,
      "created_at": 1772826690.55,
      "status": "HEALING",
      "healing_attempts": 3,
      "last_healing_attempt_at": 1772826720.55,
      "first_healing_attempt_at": 1772826700.55,
      "healing_days_elapsed": 1.5,
      "max_healing_days": 30,
      "circuit_breaker_enabled": true,
      "circuit_breaker_tripped_at": 1772826750.55
    }
  },
  "metadata": {
    "last_cleanup_at": 1772826690.55,
    "total_dust_symbols": 5,
    "total_dust_notional": 4.95
  }
}
```

### Key Methods

#### Tracking
- `mark_position_as_dust(symbol, quantity, notional)` - Record new dust position
- `mark_healing_started(symbol)` - Record when healing attempt begins
- `record_healing_attempt(symbol)` - Increment attempt counter
- `mark_healing_complete(symbol)` - Record successful healing

#### Status Queries
- `is_dust_position(symbol)` - Check if position is tracked as dust
- `get_dust_status(symbol)` - Get current status
- `get_healing_attempts(symbol)` - Get attempt count
- `get_dust_position_info(symbol)` - Get all tracking data

#### Circuit Breaker
- `should_attempt_healing(symbol)` - Check if healing should be attempted
- `trip_circuit_breaker(symbol)` - Prevent further healing attempts
- `reset_circuit_breaker(symbol)` - Allow healing to resume
- `is_circuit_breaker_tripped(symbol)` - Check breaker status

#### Cleanup
- `mark_healed(symbol)` - Remove from tracking after successful healing
- `cleanup_abandoned_dust(days_threshold)` - Remove dust not improved in N days
- `get_all_dust_positions()` - Get all tracked dust
- `get_dust_summary()` - Get metrics

### Persistence

- Stored in `dust_registry.json` (same directory as bootstrap_metrics.json)
- Atomic writes using temp file + rename
- Survives restart

### Integration with is_cold_bootstrap()

The circuit breaker status can be checked during bootstrap detection to prevent repeated healing attempts:

```python
# In is_cold_bootstrap():
if self.dust_registry.is_circuit_breaker_tripped(symbol):
    # Don't attempt healing for this symbol
    continue
```

## Implementation Timeline

1. Create DustRegistry class (100 lines)
2. Create DustPosition dataclass (20 lines)
3. Integrate with SharedState (10 lines)
4. Create comprehensive tests (300+ lines, 20+ tests)

Total: ~150 lines of production code + ~300 lines of tests

## Test Coverage

### Test Classes
1. **TestDustRegistryBasics** (5 tests)
   - Initialization
   - File creation
   - Empty registry

2. **TestDustPositionTracking** (8 tests)
   - Mark as dust
   - Record attempts
   - Update status
   - Persist to disk

3. **TestCircuitBreaker** (6 tests)
   - Trip circuit breaker
   - Check breaker status
   - Prevent healing when tripped
   - Reset breaker

4. **TestDustLifecycle** (5 tests)
   - Full lifecycle: new → healing → complete
   - Circuit breaker during lifecycle
   - Cleanup scenarios

5. **TestDustRegistryCleanup** (4 tests)
   - Mark healed
   - Cleanup abandoned
   - Summary statistics

6. **TestDustRegistryIntegration** (3 tests)
   - Integration with SharedState
   - Persistence across restarts

**Total: 31 tests**

## Success Criteria

- ✅ All 31 tests passing
- ✅ Dust positions properly tracked
- ✅ Circuit breaker prevents repeated healing
- ✅ Cleanup removes abandoned dust
- ✅ Persistent across restarts
- ✅ No cross-test interference
- ✅ Production-ready code

## Files to Create/Modify

1. **core/shared_state.py**
   - Add DustPosition dataclass
   - Add DustRegistry class
   - Integrate into SharedState.__init__()
   - Add DustRegistry to __all__ exports

2. **test_dust_registry_lifecycle.py** (NEW)
   - 31 comprehensive tests
   - 100% coverage of DustRegistry functionality

## Expected Outcomes

After Phase 3 completion:
- ✅ Dust positions have persistent lifecycle tracking
- ✅ Circuit breaker prevents repeated healing attempts
- ✅ Abandoned dust can be cleaned up
- ✅ System won't get stuck in healing loop for same position
- ✅ 60/70 total tests passing (Phase 1, 2, 3)
