# UURE Integration with AppContext: Complete Guide

## Overview

The Unified Universe Rotation Engine (UURE) is now fully integrated into AppContext with:

✅ **Component Instantiation** - UURE created during bootstrap
✅ **Background Task** - Periodic rotation every 5 minutes
✅ **Shared State Propagation** - UURE receives shared_state updates
✅ **Graceful Shutdown** - UURE loop stopped cleanly during teardown

---

## Integration Points

### 1. Module Import (Line 71)

```python
_uure_mod = _import_strict("core.universe_rotation_engine")
```

**Purpose:** Strict import of UniverseRotationEngine class
**Behavior:** Fails fast if module not found (no fallbacks)

---

### 2. Component Registration in __init__ (Line 1000)

```python
self.universe_rotation_engine: Optional[Any] = None
```

**Purpose:** Register UURE as an optional component
**Behavior:** Can be injected externally or constructed during bootstrap

---

### 3. Bootstrap Construction (Line 3335-3346)

```python
# UniverseRotationEngine
if self.universe_rotation_engine is None:
    UniverseRotationEngine = _get_cls(_uure_mod, "UniverseRotationEngine")
    self.universe_rotation_engine = _try_construct(
        UniverseRotationEngine,
        config=self.config,
        logger=self.logger,
        shared_state=self.shared_state,
        capital_symbol_governor=self.capital_symbol_governor,
        execution_manager=self.execution_manager,
    )
```

**Purpose:** Construct UURE instance during AppContext bootstrap
**Dependencies Passed:**
- `config` - System configuration
- `logger` - Structured logging
- `shared_state` - Symbol universe and balance state
- `capital_symbol_governor` - Safety constraints
- `execution_manager` - Trade execution for liquidation

**Construction Guard:** `if self.universe_rotation_engine is None` allows external injection

---

### 4. Shared State Propagation (Line 436)

Added to `_components_for_shared_state()`:

```python
self.universe_rotation_engine,
```

**Purpose:** UURE receives updates when SharedState is swapped/reinjected
**Mechanism:** Calls `set_shared_state()` if available, otherwise assigns attribute

---

### 5. Shutdown Ordering (Line 451)

Added to `_components_for_shutdown()`:

```python
self.universe_rotation_engine,
```

**Order:** Before `tp_sl_engine` (UURE must liquidate before price targets apply)

---

### 6. Background Loop Initialization (Line 1047)

```python
self._uure_task: Optional[asyncio.Task] = None
```

**Purpose:** Persistent reference to background rotation task
**Usage:** Idempotence guard and graceful cleanup

---

### 7. Loop Startup (Line 1820-1825)

```python
try:
    self._start_uure_loop()
except Exception:
    self.logger.debug("failed to start UURE loop after gates clear", exc_info=True)
```

**Trigger:** After all readiness gates clear (market data ready, balances synced, etc.)
**Timing:** Only starts when exchange is ready, shared_state is populated
**Error Handling:** Logs non-fatal errors, allows bootstrap to continue

---

### 8. Loop Implementation (Lines 2818-2883)

```python
async def _uure_loop(self) -> None:
    """
    Background Universe Rotation Engine loop:
      - Periodically calls UURE to evaluate and update symbol universe
      - Runs every UURE_INTERVAL_SEC seconds (default: 300, i.e., 5 minutes)
      - Graceful error handling with debug logging
    Config (defaults):
      UURE_ENABLE=True
      UURE_INTERVAL_SEC=300 (5 minutes)
    """
```

**Core Logic:**
1. Check if UURE is enabled
2. Sleep for UURE_INTERVAL_SEC
3. Call `universe_rotation_engine.compute_and_apply_universe()`
4. Log rotation result (added/removed/kept symbols)
5. Emit UNIVERSE_ROTATION summary event
6. Repeat

**Error Handling:**
- Catches `CancelledError` for graceful shutdown
- Catches and logs other exceptions without stopping loop
- Continues rotation even if one cycle fails

---

### 9. Loop Startup Guard (Lines 2884-2905)

```python
def _start_uure_loop(self) -> None:
    """
    Idempotently start the background UURE loop (if enabled).
    """
```

**Features:**
- ✅ Idempotent: Won't create duplicate tasks
- ✅ Guard checks: Respects UURE_ENABLE config flag
- ✅ Graceful:** Handles no-running-loop case
- ✅ Debug logs:** Records all failures for troubleshooting

**Idempotence Check:**
```python
if getattr(self, "_uure_task", None) and not self._uure_task.done():
    return  # Already running, skip
```

---

### 10. Loop Shutdown (Lines 2906-2918)

```python
async def _stop_uure_loop(self) -> None:
    """
    Stop the background UURE loop if running.
    """
```

**Process:**
1. Get reference to running task
2. Cancel the task
3. Await cancellation completion
4. Clean up reference

---

### 11. Shutdown Integration (Line 2213-2217)

```python
try:
    await self._stop_uure_loop()
except Exception:
    self.logger.debug("shutdown: stop UURE loop failed", exc_info=True)
```

**Purpose:** Clean shutdown of UURE loop before component teardown
**Timing:** Before component shutdown (allows final rotation if needed)

---

## Configuration

### UURE_ENABLE (Default: True)

```python
UURE_ENABLE = True
```

**Purpose:** Master switch for UURE rotation
**Options:**
- `True` - Run background rotation
- `False` - Skip rotation (universe frozen)

---

### UURE_INTERVAL_SEC (Default: 300)

```python
UURE_INTERVAL_SEC = 300  # 5 minutes
```

**Purpose:** Seconds between rotation cycles
**Recommended Values:**
- `60` - Testing (every minute)
- `300` - Normal ops (5 minutes)
- `600` - Conservative (10 minutes)
- `1200` - Stable portfolio (20 minutes)

**Trade-offs:**
- Shorter interval = faster weak symbol exit
- Longer interval = fewer liquidations, lower fees

---

## Data Flow

### Universe Rotation Cycle

```
1. UURE Loop Sleep (UURE_INTERVAL_SEC)
   └─ Wait 300 seconds (default)

2. UURE.compute_and_apply_universe()
   ├─ Collect all candidates
   ├─ Score each candidate
   ├─ Rank by score (descending)
   ├─ Apply smart cap
   ├─ Identify rotation (added/removed/kept)
   ├─ Hard-replace SharedState.accepted_symbols
   └─ Liquidate removed symbols

3. Liquidation (if any symbols removed)
   ├─ Create sell intents
   ├─ Submit to MetaController
   └─ Free up capital

4. Next cycle rebalance
   ├─ PortfolioBalancer sizes new universe
   ├─ Buys new symbols
   └─ Adjusts position weights

5. Loop sleep
   └─ Next rotation in 5 min
```

### Example: Hour 1→2 Transition

**Hour 1 State:**
```
Candidates discovered: [BTC, ETH, ADA, SOL, ...]
Accepted universe: {BTCUSDT, ETHUSDT}  (cap=2)
Positions: BTC=$86, ETH=$86
```

**Hour 2 Discovery:**
```
New candidates: {ETH (0.82), ADA (0.75), SOL (0.72), BTC (0.65), ...}
Scores change due to market conditions
```

**UURE Evaluation:**
```
Ranked: [ETH (0.82), ADA (0.75), SOL (0.72), BTC (0.65), ...]
Cap: 2 (from governor + capital)
New universe: {ETHUSDT, ADAUSDT}

Rotation:
  - Added: [ADAUSDT]
  - Removed: [BTCUSDT]  <- Weak, exits automatically
  - Kept: [ETHUSDT]
```

**Action:**
```
1. Hard-replace: SharedState.accepted_symbols = {ETHUSDT, ADAUSDT}
2. Liquidate: MetaController.execute_sell(BTCUSDT, qty=0.0019)
3. Rebalance: PortfolioBalancer.compute_targets() → {ETH: $86, ADA: $86}
4. Result: Portfolio rotated to stronger symbols
```

---

## Integration Summary

| Component | Role | Status |
|-----------|------|--------|
| AppContext.__init__() | Register & construct UURE | ✅ |
| _propagate_* methods | Inject dependencies | ✅ Auto |
| _components_for_shared_state() | Receive SharedState updates | ✅ |
| _components_for_shutdown() | Ordered shutdown | ✅ |
| _start_uure_loop() | Launch background task | ✅ |
| _uure_loop() | Periodic rotation logic | ✅ |
| _stop_uure_loop() | Clean shutdown | ✅ |
| graceful_shutdown() | Stop loop before teardown | ✅ |

---

## Testing

### Unit Test: UURE Instantiation

```python
import asyncio
from core.app_context import AppContext

async def test_uure_construction():
    """Verify UURE is constructed during AppContext bootstrap."""
    ctx = AppContext(config={})
    
    # Construct components
    await ctx.public_bootstrap()
    
    # Assert UURE is created
    assert ctx.universe_rotation_engine is not None
    assert hasattr(ctx.universe_rotation_engine, 'compute_and_apply_universe')
    
    # Cleanup
    await ctx.graceful_shutdown()

asyncio.run(test_uure_construction())
```

### Unit Test: UURE Loop Task

```python
async def test_uure_loop_starts():
    """Verify UURE background loop starts after gates clear."""
    ctx = AppContext(config={})
    await ctx.public_bootstrap()
    
    # Assert loop task is running
    assert ctx._uure_task is not None
    assert not ctx._uure_task.done()
    
    # Cleanup
    await ctx.graceful_shutdown()
    await asyncio.sleep(0.1)  # Let cancellation complete
    assert ctx._uure_task.done()

asyncio.run(test_uure_loop_starts())
```

### Unit Test: Rotation Summary

```python
async def test_universe_rotation_summary():
    """Verify UURE emits UNIVERSE_ROTATION summary events."""
    summaries = []
    
    ctx = AppContext(config={
        'UURE_INTERVAL_SEC': 1,  # Fast rotation for testing
    })
    
    # Hook summary emission
    original_emit = ctx._emit_summary
    async def track_summary(event, **kvs):
        summaries.append(event)
        return await original_emit(event, **kvs)
    
    ctx._emit_summary = track_summary
    await ctx.public_bootstrap()
    
    # Wait 2 cycles
    await asyncio.sleep(3)
    
    # Assert summaries logged
    assert 'UNIVERSE_ROTATION' in summaries
    
    await ctx.graceful_shutdown()

asyncio.run(test_universe_rotation_summary())
```

---

## Troubleshooting

### UURE Loop Not Starting

**Symptoms:**
- `_uure_task` is `None`
- No rotation happening

**Debug Steps:**
1. Check config: `UURE_ENABLE=True`
2. Check logs: Look for "[UURE] background loop started"
3. Verify UURE constructed: `ctx.universe_rotation_engine is not None`
4. Verify gates cleared: "readiness gates cleared" in logs

**Solution:**
```python
# Manually start loop if gates cleared
await ctx.public_bootstrap()
ctx._start_uure_loop()  # Explicit start if needed
```

---

### UURE Loop Stopping Unexpectedly

**Symptoms:**
- Loop runs once, then stops
- `_uure_task.done()` returns `True`

**Debug Steps:**
1. Check exception: Look for "[UURE] loop iteration failed"
2. Check dependencies: governor, shared_state, execution_manager not None
3. Check SharedState: Is it available and updated?

**Solution:**
```python
# Add detailed logging
ctx.logger.setLevel(logging.DEBUG)
# Re-start loop
await ctx._stop_uure_loop()
ctx._start_uure_loop()
```

---

### Universe Not Changing

**Symptoms:**
- UURE runs but accepted_symbols unchanged
- No rotation happening

**Debug Steps:**
1. Check discovery: Are new candidates being found?
2. Check scores: Are scores changing between cycles?
3. Check cap: Is smart cap allowing rotation?

**Solution:**
```python
# Manually check universe evaluation
result = await ctx.universe_rotation_engine.compute_and_apply_universe()
print(f"Rotation: added={result['added']}, removed={result['removed']}")
```

---

## Production Checklist

- [x] UURE module imported (strict, no fallbacks)
- [x] UURE component registered in AppContext
- [x] UURE constructed during bootstrap
- [x] UURE receives shared_state updates
- [x] UURE included in shutdown ordering
- [x] Background loop created with idempotence guard
- [x] Loop startup after readiness gates
- [x] Loop shutdown before component teardown
- [x] Config keys documented (UURE_ENABLE, UURE_INTERVAL_SEC)
- [x] Summary events emitted (UNIVERSE_ROTATION)
- [x] Error handling comprehensive (no naked exceptions)
- [x] Logging integrated (debug + info)
- [x] Documentation complete

---

## Summary

UURE is now **production-ready** with:

✅ **Deterministic** - Same inputs → same universe every time
✅ **Autonomous** - Runs automatically every 5 minutes (configurable)
✅ **Safe** - Respects capital constraints, governor rules, execution availability
✅ **Integrated** - AppContext owns lifecycle, lifecycle owns AppContext
✅ **Recoverable** - Graceful error handling, persistent operation even if one cycle fails
✅ **Observable** - Summary events, debug logs, component status

**The unified universe is now the canonical symbol authority.** 🏛️
