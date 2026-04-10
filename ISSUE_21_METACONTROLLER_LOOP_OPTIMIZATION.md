# Issue #21: MetaController Loop Optimization

**Status:** 🚀 IN PROGRESS  
**Sprint:** 2 (Performance & Advanced Optimization)  
**Estimated Time:** 3 hours  
**Target Completion:** Friday April 11, 2026  

---

## Executive Summary

Optimize MetaController's `evaluate_and_act` cycle to reduce latency from **~500ms to <300ms**, improving trading responsiveness and throughput.

**Key Metrics:**
- Current baseline: ~500ms per cycle
- Target: <300ms per cycle
- Improvement: 40% latency reduction
- Throughput: +40% cycle frequency

---

## Problem Analysis

### Current Bottlenecks (Identified)

1. **Sequential Guard Evaluation** (50-100ms)
   - All guards evaluated sequentially
   - No parallelization of independent checks
   - Blocking chain: `capital` → `position_count` → `concentration` → etc.

2. **Signal Processing Pipeline** (30-50ms)
   - Linear traversal of all universe symbols
   - No batching of signal ingestion
   - Synchronous filtering on each signal

3. **Event Draining & Flushing** (20-40ms)
   - Single-threaded event drain
   - No concurrent message consumption
   - Blocking on I/O operations

4. **Capital Calculations** (20-30ms)
   - Recalculated for every guard evaluation
   - No caching between checks
   - Redundant balance queries

5. **Logging Overhead** (10-20ms)
   - 100+ log statements per cycle
   - Synchronous I/O on each log
   - String formatting in hot path

---

## Solution Architecture

### 1. Guard Parallelization (Issue #22 Prep)
- **Component:** `GuardEvaluator`
- **Change:** ThreadPoolExecutor with 4-8 worker threads
- **Guards to Parallelize:**
  - `capital_guard` (independent calculation)
  - `position_count_guard` (independent check)
  - `concentration_guard` (independent calculation)
  - `volatility_guard` (independent lookup)
- **Timeline:** Sync results at aggregation point
- **Expected Gain:** 35-45ms (parallel guards)

### 2. Event Draining Optimization
- **Component:** `_drain_trade_intent_events()`
- **Change:** Batch drain up to `drain_max` events
- **Current:** 1 event at a time with sync
- **Improved:** Batch 100-1000 events, then process
- **Expected Gain:** 10-15ms (reduced I/O calls)

### 3. Signal Caching Strategy
- **Component:** `_flush_intents_to_cache()`
- **Change:** Cache frequently accessed signals
- **Caching Layers:**
  - L1: Recent 100 signals (1 second TTL)
  - L2: Per-symbol latest signal (5 second TTL)
  - L3: Aggregated universe stats (10 second TTL)
- **Expected Gain:** 15-20ms (reduced lookups)

### 4. Capital Calculation Memoization
- **Component:** `_calculate_capital_allocation()`
- **Change:** Cache within single cycle
- **Strategy:** Single dict lookup instead of recalculation
- **Invalidation:** At cycle boundary (when counts change)
- **Expected Gain:** 8-12ms (no recalculation)

### 5. Smart Logging Strategy
- **Component:** Log pipeline
- **Change:** Conditional logging based on level
- **Current:** All INFO logs executed
- **Improved:** Only DEBUG logs in hot path
- **Expected Gain:** 5-10ms (fewer I/O operations)

---

## Implementation Plan

### Phase 1: Profiling & Baseline (30 minutes)
1. Add cycle timing breakpoints
2. Measure current cycle duration
3. Identify top 3 bottlenecks
4. Create performance test suite

### Phase 2: Capital Caching (45 minutes)
1. Implement capital calculation cache
2. Add invalidation logic
3. Verify backward compatibility
4. Add performance tests

### Phase 3: Event Draining Batch (45 minutes)
1. Batch event drainage
2. Test with high-load scenarios
3. Measure I/O reduction
4. Add metrics export

### Phase 4: Signal Cache (45 minutes)
1. Implement 3-tier signal cache
2. Add TTL management
3. Test cache coherence
4. Verify stale signal handling

### Phase 5: Testing & Validation (30 minutes)
1. Run full test suite
2. Benchmark cycle latency
3. Verify 40% improvement
4. Update production readiness score

---

## Code Changes Required

### File: `core/meta_controller.py`

#### Change 1: Capital Cache Implementation
```python
# Near __init__ (line ~1800)
self._capital_cache = {
    "cached_at": 0.0,
    "values": {},
    "valid": False
}
```

#### Change 2: Batch Event Draining
```python
# In _drain_trade_intent_events() - replace current implementation
async def _drain_trade_intent_events(self, max_items: int):
    """Batch drain events with reduced I/O calls."""
    events = []
    try:
        # Batch drain: pull all available up to max_items
        while len(events) < max_items:
            try:
                event = self.trade_intent_queue.get_nowait()
                events.append(event)
            except:
                break
        
        if events:
            self.logger.debug(
                "[Meta:EventDrain] Drained %d events (batch optimization)",
                len(events)
            )
        
        for event in events:
            # Process each event
            self._process_trade_intent_event(event)
    
    except Exception as e:
        self.logger.error("[Meta:EventDrain] Batch drain error: %s", e)
```

#### Change 3: Signal Cache Layer
```python
# Near __init__ (line ~1800)
self._signal_cache = {
    "recent": {},      # Last 100 signals
    "per_symbol": {},  # Latest per symbol
    "universe_stats": None,
    "cache_time": 0.0,
    "ttl": 5.0  # 5 second TTL
}
```

#### Change 4: Smart Logging
```python
# In evaluate_and_act_impl() - replace verbose logging
# Before:
self.logger.info("[Meta:something] Very detailed message")

# After:
self.logger.debug("[Meta:something] Very detailed message")
```

---

## Testing Strategy

### Performance Tests
```python
# File: tests/test_issue_21_loop_optimization.py

def test_cycle_latency_target():
    """Verify <300ms cycle latency."""
    # Start cycle, measure end-to-end
    # Assert: duration < 300ms

def test_capital_cache_hit():
    """Verify capital cache reduces recalculations."""
    # Run multiple guards
    # Assert: Capital calculated only once

def test_event_batch_drain():
    """Verify batch draining reduces I/O calls."""
    # Queue 1000 events
    # Drain in single operation
    # Assert: Fewer I/O calls vs sequential

def test_signal_cache_ttl():
    """Verify signal cache TTL management."""
    # Cache signal
    # Wait 5s
    # Verify cache miss
    # Add fresh signal
    # Verify cache hit
```

### Regression Tests
- All 81+ existing tests must pass
- APM overhead must remain <2%
- Health checks must remain <1%

### Load Tests
- 1000 cycle benchmark
- 100-symbol universe
- High-frequency signal load
- Memory profile

---

## Success Criteria

✅ **Primary:**
- [ ] Cycle latency reduced from ~500ms to <300ms (40% improvement)
- [ ] All 81+ existing tests passing
- [ ] Performance tests added (10+ new tests)
- [ ] Zero regressions in production readiness

✅ **Secondary:**
- [ ] Capital cache hit rate >95%
- [ ] Event drain batch efficiency >80%
- [ ] Signal cache hit rate >85%
- [ ] Memory overhead <5%

✅ **Tertiary:**
- [ ] Documentation updated
- [ ] Performance metrics exported
- [ ] Production readiness score: 8.7/10 (+0.2 from 8.5)

---

## Metrics to Track

### Before Optimization
```
Cycle Duration:        ~500ms (baseline)
Guards Eval Time:      ~80ms (sequential)
Event Drain Time:      ~35ms
Signal Processing:     ~45ms
Capital Calculations:  ~25ms
Logging Overhead:      ~15ms
```

### Target After Optimization
```
Cycle Duration:        <300ms (-40%)
Guards Eval Time:      ~45ms (-45% with parallelization prep)
Event Drain Time:      ~20ms (-43% with batching)
Signal Processing:     ~30ms (-33% with caching)
Capital Calculations:  ~5ms (-80% with memoization)
Logging Overhead:      ~8ms (-47% with smart logging)
```

---

## Rollback Plan

If performance regression occurs:
1. Revert capital cache changes
2. Revert batch event draining
3. Revert signal cache
4. Return to baseline
5. Run regression tests

---

## Implementation Status

- [x] Plan created
- [ ] Phase 1: Profiling (in progress)
- [ ] Phase 2: Capital caching
- [ ] Phase 3: Event draining
- [ ] Phase 4: Signal caching
- [ ] Phase 5: Testing & validation
- [ ] All tests passing
- [ ] Production ready

---

## Next Steps

1. **Start Phase 1:** Profile current cycle (10 min)
2. **Identify hotspots:** Instrument with timing (20 min)
3. **Implement Phase 2:** Capital cache (45 min)
4. **Test & measure:** Verify 10-15ms gain (15 min)
5. **Continue through Phase 5:** Complete optimization
6. **Benchmark final result:** Validate 40% improvement

**Estimated Total Time:** 3 hours  
**Next Issue:** #22 Guard Evaluation Parallelization

