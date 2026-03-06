# Why TradeIntent Events Are NOT Being Emitted - Root Cause Analysis

## The Issue

Your logs show:
```
[TrendHunter] Buffered BUY for ETHUSDT      ✓ Signal generated
[Meta] Signal cache contains 2 signals       ✓ Signal cached
[Meta:POST_BUILD] decisions_count=1          ✓ ONE decision built
[Meta:POST_BUILD] decisions_count=0          ✗ Then NO MORE decisions
[execute_trade]                              ✗ NO execution
[TradeIntent]                                ✗ NO event emitted
```

## Root Cause: Signal Batcher Deferral

Signals are being generated and cached, but **signal batching is deferring execution indefinitely**.

### The Signal Flow

```
TrendHunter generates signal
    ↓
Signal cached in MetaController._signal_cache
    ↓
_build_decisions() processes cache → calls signal_batcher.add_signal()
    ↓
signal_batcher.should_flush() checks:
    - Has batch window elapsed? (default 5 seconds)
    - Is batch full? (default batch size)
    - Is there a SELL signal? (critical flush)
    ↓
IF should_flush() returns TRUE:
    ↓
    Signals are flushed and converted to decisions
    ↓
    decisions = [(...), (...)]  ← decisions_count > 0
    ↓
    TradeIntent events emitted ✓
    ↓
    _execute_decision() called ✓
ELSE:
    ↓
    decisions = []  ← decisions_count = 0  ✗
    ↓
    NO TradeIntent emitted
    ↓
    NO execution
```

## Why Batcher Isn't Flushing

Looking at line 6010 in meta_controller.py:

```python
should_flush = self.signal_batcher.should_flush()
if should_flush:
    # Flush and execute
    batched_signals = await self.signal_batcher.flush()
    decisions = [...]  ← decisions list populated
else:
    # DEFER - batch window not elapsed yet
    decisions = []  ← decisions list EMPTY
    self.logger.debug("[Meta:Batching] Batch not ready...")
```

### Timeline Analysis from Your Logs

```
23:40:43,795 - Signal buffered in TrendHunter
23:40:44,862 - Signal in cache (0.067 seconds later)
23:40:44,940 - decisions_count=1 (one decision made!)
                 This is FIRST flush - must have met condition
23:40:47,621 - decisions_count=0 (batch window reset?)
23:40:49,709 - decisions_count=0 (still waiting)
23:40:51,800 - decisions_count=0 (still waiting)
```

**Theory:** Batch window is **5 seconds** (default). After first flush at 23:40:44, next flush would be around 23:40:49-23:40:50, but the signal might not generate new ones in that window, OR new signals are arriving but batcher is still deferring.

## Configuration Issue

**Default Setting:**
```python
SIGNAL_BATCH_WINDOW_SEC = 5.0  # 5 second window
```

**What This Means:**
- Signals are collected for 5 seconds
- Every 5 seconds, batch flushes and executes
- BETWEEN flushes, signals accumulate but don't execute
- In shadow mode with ONE BUY signal every 5-10 seconds, this causes execution gaps

## Why TradeIntent Events Aren't Emitted

TradeIntent events are ONLY emitted in the execution path (lines 6163 and 6208):

```python
if decisions:  # ← decisions list must be non-empty
    for sym, side, sig in decisions:
        # ONLY HERE is TradeIntent emitted
        await self.shared_state.emit_event("TradeIntent", {...})
```

**But decisions are empty when:**
- Batcher hasn't flushed yet
- Batch window not elapsed
- Next batch not ready

## The Direct Cause: Agents Call receive_signal(), Not emit TradeIntent

You asked: "Why are agents calling `meta_controller.receive_signal(...)` instead of emitting TradeIntent?"

**Answer:** 

1. **Agents generate signals** → call `MetaController.receive_signal()`
2. **MetaController caches signals** → calls `signal_batcher.add_signal()`
3. **Batcher defers** → decisions_count = 0
4. **No decisions** → TradeIntent NEVER emitted
5. **Execution blocked** → No order sent

The architecture expects:
- Agents emit signals (correct ✓)
- MetaController batches them (working ✓)
- Batcher flushes on schedule (BROKEN ✗)
- Decisions built from batch (blocked)
- TradeIntent emitted (never reached)
- Orders executed (never reached)

## Evidence

**From meta_controller.py line 5968:**

```python
# Add to batcher (de-duplicates on symbol+side)
self.signal_batcher.add_signal(batched)
```

**From meta_controller.py line 6010:**

```python
should_flush = self.signal_batcher.should_flush()
if should_flush:
    # 3 conditions for flush:
    # 1. Batch window elapsed (default 5 sec)
    # 2. Batch size full (default varies)
    # 3. Critical signal present (SELL)
else:
    # NO FLUSH → decisions = []
    decisions = []  ← Empty!
```

**Result:** If `should_flush()` returns FALSE, execution is deferred and TradeIntent events are never emitted.

## Logs Confirming This

Your logs show:

```
[Meta:Batching] Batch not ready (pending=2, window_elapsed=0.50s, threshold=5.00s)
```

Translation:
- **pending=2** → 2 signals waiting in batch
- **window_elapsed=0.50s** → Only 0.5 seconds have passed
- **threshold=5.00s** → Need 5.0 seconds to flush
- **Status:** WAITING, NOT FLUSHING ✗

This confirms the batcher is deferring because the 5-second window hasn't elapsed!

## Solution

There are three ways to fix this:

### Option 1: Reduce Batch Window
```python
SIGNAL_BATCH_WINDOW_SEC = 0.5  # 500ms instead of 5s
```
**Pros:** Faster execution
**Cons:** More fragmentation, less batching benefit

### Option 2: Force Flush on Critical Signals
```python
# In signal_batcher.should_flush():
if any(sig.side == "BUY" for sig in self._pending_signals):
    return True  # Force flush on BUY signals
```
**Pros:** BUY signals execute immediately
**Cons:** Less batching of SELL signals

### Option 3: Disable Batching in Shadow Mode
```python
if trading_mode == "shadow":
    should_flush = True  # Always flush immediately
else:
    should_flush = self.signal_batcher.should_flush()
```
**Pros:** Shadow mode executes immediately, live mode benefits from batching
**Cons:** Slightly more code complexity

## Recommended Fix

**Option 3** is best because:
1. Shadow mode is for testing - needs fast execution
2. Live mode benefits from batching friction reduction
3. Maintains architecture intent
4. No API changes needed

Implementation:
```python
# In _build_decisions(), around line 6010
if self.trading_mode == "shadow":
    # Shadow mode: disable batching, execute immediately
    should_flush = True
else:
    # Live mode: use configured batch window
    should_flush = self.signal_batcher.should_flush()

if should_flush:
    # Execute as normal
```

## Summary

**Why TradeIntent events aren't emitted:**
1. Agents call `receive_signal()` → signals cached ✓
2. Signals added to batcher ✓
3. **Batcher defers because 5-second window not elapsed** ✗
4. Decisions remain empty (decisions_count = 0)
5. Execution skipped (no decisions to execute)
6. **TradeIntent events never emitted** (only emitted with decisions)

**Root Cause:** Signal batching design works in live trading (reduces friction) but blocks execution in shadow mode (needs immediate testing feedback).

**Fix:** Disable batching (use Option 3) or reduce batch window for shadow mode.
