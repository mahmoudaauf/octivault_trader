# 🔍 Diagnostic Guide: Signal Batching Deadlock Fix

## Quick Diagnosis

### **Symptom Checklist**

Check your logs for these patterns:

```
✅ Pattern 1: Signal Successfully Buffered
[SignalManager] Signal ACCEPTED and cached: AAPL from Strategy (confidence=0.75)
[Meta:Batching] Added signal to batcher: AAPL BUY (confidence=0.750)

✅ Pattern 2: Batch Deferral (Normal, 5 seconds max)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=0.50s, threshold=5.00s)
[LOOP_SUMMARY] decision=NONE exec_attempted=False

❌ Pattern 3: DEADLOCK - Signals Stuck (Should NOT see more than 5-10 times in a row)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=0.50s, threshold=5.00s)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=1.00s, threshold=5.00s)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=1.50s, threshold=5.00s)
...
[Meta:Batching] Batch not ready (pending=1, window_elapsed=4.50s, threshold=5.00s)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=4.99s, threshold=5.00s)
[Meta:Batching] Batch not ready (pending=1, window_elapsed=5.49s, threshold=5.00s)  ⚠️ NOW STALE!
```

---

## What the Fix Does

### **Before the Fix: Deadlock Scenario**

```
Cycle 1 (t=0s):
  └─ Signal: BUY AAPL (confidence=0.75)
     └─ Added to batcher
        └─ window_elapsed = 0.0s < threshold=5.0s
           └─ Batch NOT flushed ⚠️
              └─ decisions = []
                 └─ return (NO EXECUTION) ❌

Cycle 2 (t=2s):  ← 2 seconds later
  └─ No new signals (agent not generating)
     └─ _build_decisions() returns empty list
        └─ Batcher NEVER checked again
           └─ Old signal still in _pending_signals
              └─ DEADLOCKED FOREVER ❌

Cycle 3, 4, 5... (t=4s, 6s, 8s...):
  └─ Same as Cycle 2
     └─ Batcher never wakes up
        └─ Signals trapped ❌
```

### **After the Fix: Forced Flush**

```
Cycle 1 (t=0s):
  └─ Signal: BUY AAPL (confidence=0.75)
     └─ Added to batcher
        └─ window_elapsed = 0.0s < threshold=5.0s
           └─ Batch NOT flushed yet (normal) ⚠️
              └─ decisions = []
                 └─ return (deferred, not stuck) ✅

Cycle 2 (t=2s):
  └─ BEFORE adding new signals:
     └─ Check: _pending_signals has 1 signal, age=2.0s
        └─ age (2.0s) < threshold (5.0s)
           └─ Still waiting for window ⏳
              └─ No forced flush yet ✅

Cycle 3 (t=4.5s):
  └─ BEFORE adding new signals:
     └─ Check: _pending_signals has 1 signal, age=4.5s
        └─ age (4.5s) < threshold (5.0s)
           └─ Almost there... ⏳

Cycle 4 (t=5.1s):
  └─ BEFORE adding new signals:
     └─ Check: _pending_signals has 1 signal, age=5.1s
        └─ age (5.1s) >= threshold (5.0s) 
           └─ 🔥 FORCED FLUSH TRIGGERED!
              └─ [Meta:BatchFlush] ✓ Flushed 1 STALE signals
                 └─ decisions = [reconstructed from stale]
                    └─ Execute immediately ✅
                       └─ [Atomic:BUY] ✓ Order submitted AAPL ✅
```

---

## Log Analysis Guide

### **1. Check Batch Initialization**

```
[Meta:Init] Signal batcher initialized: window=5.0s, max_batch=10
```

**What it means:** Batcher initialized with 5-second window and max 10 signals per batch

---

### **2. Check Signal Ingestion**

```
[Meta:DRAIN] ⚠️ DRAINED 0 events from event_bus
[Meta:DRAIN] ⚠️ DRAINED 1 events from event_bus
```

**What it means:** Event bus checked for incoming trade intents from external agents

---

### **3. Check Signal Caching**

```
[SignalManager] Signal ACCEPTED and cached: AAPL from Strategy (confidence=0.75)
[Meta:Batching] Added signal to batcher: AAPL BUY (confidence=0.750)
```

**What it means:** Signal successfully moved from cache to batcher queue

---

### **4. Normal Batch Deferral (NOT DEADLOCK)**

```
[Meta:Batching] Batch not ready (pending=1, window_elapsed=0.50s, threshold=5.00s)
```

**Status:** ✅ NORMAL - Batch intentionally deferred, waiting for window

**When to expect:** When signals arrive and batch window hasn't elapsed yet

**How long:** Typically 5 seconds or until batch fills up

---

### **5. Forced Flush (FIX ACTIVATED)**

```
[Meta:BatchFlush] 🔥 FORCED FLUSH: Batch aged 5.12s >= window 5.00s with 1 pending signals
[Meta:BatchFlush] ✓ Flushed 1 STALE signals (age=5.12s)
[Meta:BatchFlush] ⚠️ EXECUTING 1 DEFERRED signals before new decisions
```

**Status:** ✅ FIX WORKING - Old batch automatically flushed after window expired

**When to expect:** ~5 seconds after signals are buffered

**What happens next:** Signals reconstructed and passed to `_execute_decision()`

---

### **6. Deferred Execution**

```
[Meta:StaleExecution] 🚀 EXECUTING 1 DEFERRED stale signals NOW
```

**Status:** ✅ EXECUTION PROCEEDING - Old signals being executed on fresh cycle

**Expected flow:**
```
[Meta:StaleExecution] 🚀 EXECUTING 1 DEFERRED stale signals NOW
[Meta:P9-GATE] Readiness check...
[Meta:CapitalGovernor] Position limit OK...
[Atomic:BUY] ✓ Order submitted AAPL: qty=1.234, quote=50.00 USDT
```

---

### **7. Final Confirmation**

```
[LOOP_SUMMARY] decision=BUY exec_attempted=True exec_result=SUCCESS trade_opened=True
```

**Status:** ✅ TRADE COMPLETE - Signal successfully converted to execution

---

## Diagnosing Problems

### **Issue: "Batch not ready" appears many times (>10 times)**

**Diagnosis:**
```
Cycle 1: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.02s, threshold=5.00s)
Cycle 2: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.04s, threshold=5.00s)
Cycle 3: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.06s, threshold=5.00s)
...
Cycle 100: [Meta:Batching] Batch not ready (pending=1, window_elapsed=2.00s, threshold=5.00s)
```

**Possible causes:**
1. **Batch window is too long** - Set `SIGNAL_BATCH_WINDOW_SEC` to a shorter value
2. **_build_decisions() not being called** - Check that `evaluate_and_act()` is running
3. **Timing issue** - Check system clock is correct

**Solution:**
```python
# In config:
SIGNAL_BATCH_WINDOW_SEC=2.0  # Reduce from 5.0 to 2.0 seconds
```

---

### **Issue: "Batch not ready" then nothing happens (no forced flush)**

**Diagnosis:**
```
Cycle 1: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.50s, threshold=5.00s)
Cycle 2: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.52s, threshold=5.00s)
Cycle 3: [Meta:Batching] Batch not ready (pending=1, window_elapsed=0.54s, threshold=5.00s)
...
Cycle 100: [Meta:Batching] Batch not ready (pending=1, window_elapsed=2.00s, threshold=5.00s)
❌ NO FORCED FLUSH SEEN (should appear at ~5 seconds)
```

**Possible causes:**
1. **Batch window timer not incrementing** - Check system time
2. **Cycles not running** - MetaController.run() might be stuck
3. **Batcher state corrupted** - `_batch_start_time` might be wrong

**Solution:**
```python
# In logs, look for:
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #X starting

# If iteration counter not increasing, MetaController loop is stuck
```

---

### **Issue: Forced flush happens but no trade executes**

**Diagnosis:**
```
[Meta:BatchFlush] 🔥 FORCED FLUSH: Batch aged 5.12s >= window 5.00s with 1 pending signals
[Meta:BatchFlush] ✓ Flushed 1 STALE signals (age=5.12s)
❌ NO "EXECUTING DEFERRED" message
❌ NO "Order submitted" message
```

**Possible causes:**
1. **Signal reconstruction failed** - Check `sig.extra.get("original_signal")`
2. **Empty flush** - Batcher.flush() returned empty list (race condition?)
3. **Readiness gates blocking** - Execution gates preventing trade

**Debug steps:**
```python
# Add to meta_controller.py temporarily:
if stale_signals:
    self.logger.info(f"[DEBUG] stale_signals={stale_signals}")
    for sig in stale_signals:
        self.logger.info(f"[DEBUG] signal={sig.symbol} side={sig.side} extra={sig.extra.keys()}")
```

---

## Performance Monitoring

### **Metrics to Watch**

**1. Batch Flush Rate**
```
[Meta:BatchFlush] 🔥 FORCED FLUSH: ... with 1 pending signals
[Meta:BatchFlush] 🔥 FORCED FLUSH: ... with 3 pending signals
```

Expected: ~1 forced flush per 5 seconds per active signal stream

---

**2. Deferred Signal Count**
```
[Meta:BatchFlush] ✓ Flushed 1 STALE signals
[Meta:BatchFlush] ✓ Flushed 2 STALE signals
[Meta:BatchFlush] ✓ Flushed 5 STALE signals
```

Expected: ~1-5 signals per flush (higher = better batching)

---

**3. Execution Latency**
```
t=0.0s: [Meta:Batching] Added signal to batcher: AAPL BUY
t=5.1s: [Meta:BatchFlush] 🔥 FORCED FLUSH
t=5.1s: [Atomic:BUY] ✓ Order submitted AAPL
```

Expected: ~5.0-5.5 seconds (batch window + small overhead)

---

## Testing the Fix

### **Manual Test 1: Send Single Signal**

```python
# In a test script:
meta = MetaController(...)
await meta.start()

# Cycle 1 (t=0s): Send signal
await meta.receive_signal("TestAgent", "AAPL", {
    "side": "BUY",
    "confidence": 0.75
})

# Wait 5 seconds
time.sleep(5)

# Cycle ~5 (t=5s): Check logs for FORCED FLUSH
# Should see:
# [Meta:BatchFlush] 🔥 FORCED FLUSH
# [Atomic:BUY] ✓ Order submitted AAPL
```

---

### **Manual Test 2: Disable Batching**

```python
# In config:
SIGNAL_BATCH_WINDOW_SEC=0.0  # Disable batching

# Should execute immediately:
# [Meta:Batching] Added signal to batcher: AAPL BUY
# [Meta:Batching] Batch ready (should_flush=True, critical=False)
# [Meta:Batching] ✓ Flush triggered: 1 signals batched
# [Atomic:BUY] ✓ Order submitted AAPL
```

---

## Configuration Reference

| Setting | Default | Meaning | Impact |
|---------|---------|---------|--------|
| `SIGNAL_BATCH_WINDOW_SEC` | 5.0 | Max time to wait before flushing batch | ↑ = delay execution, ↓ = more trades/day |
| `SIGNAL_BATCH_MAX_SIZE` | 10 | Max signals per batch | ↑ = group more signals, ↓ = execute sooner |

---

## Summary

The fix ensures that:

1. ✅ Signals are never trapped in the batch buffer forever
2. ✅ After batch window expires, forced flush automatically executes
3. ✅ Deferred signals execute on the next evaluation cycle
4. ✅ Clear logging shows exactly when and why flushing happens

**Key log lines to look for:**
- `[Meta:BatchFlush] 🔥 FORCED FLUSH` - Fix is working
- `[Meta:StaleExecution] 🚀 EXECUTING` - Deferred signals executing
- `[Atomic:BUY] ✓ Order submitted` - Trade executed successfully

