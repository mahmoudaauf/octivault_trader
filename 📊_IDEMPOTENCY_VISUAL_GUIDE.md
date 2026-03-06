# 📊 IDEMPOTENCY FIX — Visual Guide

## The Problem: Stale Cache Forever

```
Timeline of the broken behavior:

Time=0s:   Signal arrives for ETHUSDT BUY
           └─> ExecutionManager adds to cache
               _active_symbol_side_orders = {(ETHUSDT, BUY)}
           └─> Tries to place order... DEADLOCK ❌

Time=1s:   Signal 2 arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in cache? YES
               └─> return SKIPPED ❌ (entry never cleared)

Time=10s:  Signal 3 arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in cache? YES
               └─> return SKIPPED ❌ (still there!)

Time=1h:   Signal N arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in cache? YES
               └─> return SKIPPED ❌ (PERMANENTLY BLOCKED)

Time=∞:    Manual restart required to recover 😞
```

---

## The Solution: Time-Scoped Auto-Clearing

```
Timeline of the fixed behavior:

Time=0s:   Signal arrives for ETHUSDT BUY
           └─> Add to dict with timestamp
               _active_symbol_side_orders = {(ETHUSDT, BUY): 0}
           └─> Try to place... DEADLOCK ❌
               (but timestamp is recorded!)

Time=1s:   Signal 2 arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in dict? YES
               └─> Elapsed: 1 - 0 = 1s < 30s ✓
               └─> return SKIPPED ✓ (legitimate duplicate)

Time=15s:  Signal 3 arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in dict? YES
               └─> Elapsed: 15 - 0 = 15s < 30s ✓
               └─> return SKIPPED ✓ (still within window)

Time=35s:  Signal N arrives for ETHUSDT BUY
           └─> Check: (ETHUSDT, BUY) in dict? YES
               └─> Elapsed: 35 - 0 = 35s > 30s ✓✓✓
               └─> AUTO-CLEAR! 🔥
               └─> Log: "Order stuck for 35s; forcibly clearing"
               └─> Delete from cache
               └─> NEW ORDER ATTEMPT
               └─> SUCCESS! ✅

Result: AUTOMATIC DEADLOCK RECOVERY 🎉
```

---

## Before vs After: Visual Comparison

### ❌ BEFORE (Broken)

```
┌─────────────────────────────────────────────┐
│ Order 1: ETHUSDT BUY                        │
├─────────────────────────────────────────────┤
│ ✓ Received signal                           │
│ ✓ Added to cache: (ETHUSDT, BUY)            │
│ ✗ DEADLOCK during placement                 │
│ ✗ Cache entry NEVER cleared                 │
└─────────────────────────────────────────────┘
                      ↓
                      ↓
┌─────────────────────────────────────────────┐
│ Order 2: ETHUSDT BUY (retry)                │
├─────────────────────────────────────────────┤
│ ✓ Received signal                           │
│ ✗ Check cache: (ETHUSDT, BUY) EXISTS        │
│ ✗ REJECTED ❌                               │
│ ✗ Forever blocked                           │
└─────────────────────────────────────────────┘
                      ↓
                      ↓
         🛑 PERMANENT BLOCK FOREVER 🛑
         Manual restart required
```

### ✅ AFTER (Fixed)

```
┌─────────────────────────────────────────────┐
│ Order 1: ETHUSDT BUY                        │
├─────────────────────────────────────────────┤
│ ✓ Received signal                           │
│ ✓ Added to cache: (ETHUSDT, BUY) @ 0s       │
│ ✗ DEADLOCK during placement                 │
│ ⏱ Timestamp recorded                        │
└─────────────────────────────────────────────┘
                      ↓
                      ↓
┌─────────────────────────────────────────────┐
│ Order 2: ETHUSDT BUY (retry at 2s)          │
├─────────────────────────────────────────────┤
│ ✓ Received signal                           │
│ ✓ Check cache: (ETHUSDT, BUY) EXISTS        │
│ ✓ Check time: 2s - 0s = 2s < 30s            │
│ ✗ REJECTED (still within window) ✓          │
└─────────────────────────────────────────────┘
                      ↓
      (waits... retries... waits...)
                      ↓
┌─────────────────────────────────────────────┐
│ Order N: ETHUSDT BUY (retry at 35s)         │
├─────────────────────────────────────────────┤
│ ✓ Received signal                           │
│ ✓ Check cache: (ETHUSDT, BUY) EXISTS        │
│ ✓ Check time: 35s - 0s = 35s > 30s ✓        │
│ 🔥 AUTO-CLEAR STALE ENTRY!                  │
│ ✓ Delete from cache                         │
│ ✓ Retry order placement                     │
│ ✅ SUCCESS!                                 │
└─────────────────────────────────────────────┘

Result: AUTOMATIC RECOVERY ✨
```

---

## State Diagram: Order Lifecycle

### Cache State Transitions

```
                    Normal Path
                    ───────────
                           │
                    Signal arrives
                           │
                           ↓
        ┌──────────────────────────────────┐
        │ ADD to cache with timestamp      │
        │ _active_symbol_side_orders[key]  │
        │            = now()               │
        └──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ↓             ↓
              SUCCESS       FAILURE/TIMEOUT
              (fast)           (slow)
                    │             │
                    │             ├─→ ❌ Deadlock
                    │             │   (entry stuck)
                    │             │
                    │         Wait for retry...
                    │             │
                    ↓             ↓
        ┌──────────────────────────────────┐
        │ Check age of cache entry         │
        │ if now() - entry < 30s:          │
        │     REJECT (still in window)     │
        │ else:                            │
        │     DELETE & RETRY 🔥            │
        └──────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   YOUNG (<30s)           OLD (>30s)
        │                       │
        ↓                       ↓
     REJECT            AUTO-CLEAR & RETRY
   (legitimate)           (recovery)
        │                       │
        │                       ↓
        │            ┌──────────────────┐
        │            │ Finally block:   │
        │            │ DELETE from cache│
        │            └──────────────────┘
        │                       │
        ↓                       ↓
    ┌────────┐         ┌────────────────┐
    │ CLEAN  │         │ ORDER SUCCESS  │
    │ STATE  │         │ + CLEAN STATE  │
    └────────┘         └────────────────┘
```

---

## Code Flow Diagram

```
submit_order(ETHUSDT, BUY)
    │
    ↓
build_client_order_id()
    │
    ↓
check_bootstrap()
    │
    ↓
order_key = (ETHUSDT, BUY)
now = time.time()
    │
    ├─→ 🔥 NEW LOGIC 🔥
    │   if order_key in _active_symbol_side_orders:
    │       last_attempt = dict[order_key]
    │       elapsed = now - last_attempt
    │       │
    │       ├─→ if elapsed < 30s:
    │       │       return SKIPPED (ACTIVE_ORDER)
    │       │
    │       └─→ else:  ← 🔥 AUTO-CLEAR HAPPENS HERE
    │               del dict[order_key]
    │               log "STALE_CLEARED"
    │               # Fall through to normal flow
    │
    ├─→ dict[order_key] = now  ← Record attempt
    │
    ↓
try:
    acquire_semaphore()
    place_order()
finally:
    dict.pop(order_key, None)  ← Clean up
```

---

## Metric Changes

### Cache Size Over Time

```
BEFORE FIX:
active_symbol_side_orders
│
│   ╱╲         (stale entry never cleared)
│  ╱  ╲╱╲      ╱╲
│ ╱      ╲    ╱  ╲    
├────────────→ Grows unbounded
│              Eventually fills memory

AFTER FIX:
active_symbol_side_orders
│     ╭─╮ ╭─╮
│    ╭─╯  ╰─╮ ╭─╮
│───┤        ╰─╯  ╰─╮
│    ╰──────────────╯
└─────────────────────→ Stable low level
                       Auto-cleared when stale
```

---

## Retry Pattern

### BEFORE: No Recovery
```
Request 1: ────→ FAIL (entry added)
Request 2: ────→ REJECT (ACTIVE_ORDER)
Request 3: ────→ REJECT (ACTIVE_ORDER)
Request 4: ────→ REJECT (ACTIVE_ORDER)
...
Request N: ────→ REJECT (ACTIVE_ORDER) ❌ FOREVER

Duration: ∞ (until manual restart)
```

### AFTER: Auto-Recovery
```
Request 1: ────→ FAIL (entry added, t=0s)
Request 2: ────→ REJECT (ACTIVE_ORDER, 2s elapsed)
Request 3: ────→ REJECT (ACTIVE_ORDER, 10s elapsed)
Request 4: ────→ REJECT (ACTIVE_ORDER, 20s elapsed)
Request 5: ────→ REJECT (ACTIVE_ORDER, 28s elapsed)
Request 6: ────→ STALE CLEARED! (32s elapsed) 🔥
            ────→ RETRY
            ────→ SUCCESS ✅

Duration: 32 seconds (automatic recovery)
```

---

## Impact Summary

```
┌─────────────────┬──────────────┬──────────────┐
│     Metric      │   BEFORE     │    AFTER     │
├─────────────────┼──────────────┼──────────────┤
│ Orders blocked  │  Forever ∞   │ <30 seconds  │
│ Recovery method │  Restart     │  Automatic   │
│ Cache cleanup   │  Manual ❌   │  Auto 🔥     │
│ Downtime        │  Hours 😞    │  Seconds ✨  │
│ Buy signals     │  Fail 🛑     │  Succeed ✅  │
│ Reliability     │  Low         │  High        │
└─────────────────┴──────────────┴──────────────┘
```

---

## Key Insight

```
┌────────────────────────────────────────────┐
│ BEFORE: Set tracks PRESENCE only           │
│ ├─ (symbol, side) in set?                  │
│ │  └─ YES → REJECT (forever)               │
│ └─ No time info → Can't expire             │
│                                            │
│ AFTER: Dict tracks PRESENCE + TIMESTAMP    │
│ ├─ (symbol, side) in dict?                 │
│ │  └─ YES → Check age                      │
│ │     ├─ Young (<30s) → REJECT             │
│ │     └─ Old (>30s) → AUTO-CLEAR & RETRY   │
│ └─ Time info → Can expire automatically    │
└────────────────────────────────────────────┘
```

---

## Success Indicators

✅ See `[EM:STALE_CLEARED]` messages in logs  
✅ `active_symbol_side_orders` stays low (<5)  
✅ Orders eventually succeed instead of failing  
✅ No permanent `IDEMPOTENT` rejections  
✅ Buy signal success rate improves  

🎉 **All fixed!**
