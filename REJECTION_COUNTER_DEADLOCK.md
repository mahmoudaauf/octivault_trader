# 🔍 ROOT CAUSE ANALYSIS - REJECTION COUNTER DEADLOCK

## The REAL Issue (Not Liquidation!)

**The system is stuck in a REJECTION COUNTER DEADLOCK**, not a liquidation issue!

### What's Happening:

```
Signal Generated (ETHUSDT BUY)
  ↓
Affordability Check: INSUFFICIENT_QUOTE
  ↓
Reject Signal (CAPITAL_INSUFFICIENT reason)
  ↓
Increment rejection counter: count=1
  ↓
Signal rejected again next cycle
  ↓
...repeat...
  ↓
After 10 rejections: count=10
  ↓
MetaController: "RejectionThreshold >= 10, SKIPPING ETHUSDT"
  ↓
✋ SIGNAL NEVER REACHES _execute_decision()
  ↓
❌ RULE5_ESCALATION NEVER TRIGGERED
  ↓
❌ LIQUIDATION NEVER CALLED
  ↓
🔴 DEADLOCK: Infinite rejection loop
```

### Evidence from Logs:

```
2026-05-02 20:52:23,580 INFO [Meta:Block:RejectionThreshold] 
  Skipping ETHUSDT BUY: rejected 10 times >= threshold 10 (micro=True)

2026-05-02 20:52:25,588 CRITICAL [Deadlock:TRIGGER] 
  ❌ REPEATED FAILURES DETECTED: CAPITAL_INSUFFICIENT count=10 >= threshold=10
```

---

## Why Liquidation Isn't Being Called

**The code never reaches line 19583-19620** (where I added the liquidation fix) because:

1. Signal gets rejected due to insufficient capital
2. Rejection counter incremented
3. After 10 rejections, signal is PRE-REJECTED before reaching _execute_decision()
4. RULE5_ESCALATION code never gets hit
5. Liquidation is never invoked

**The rejection counter SHORT-CIRCUITS the entire escalation chain!**

---

## The Fix Needed

We need to trigger liquidation **EARLIER** - when the **rejection counter hits the threshold**, not later in _execute_decision().

Two options:

### Option A: Clear Rejection Counters (Quick Fix)
- Reset `shared_state.rejection_counters` to 0
- Allows signals to reach _execute_decision()
- Then my liquidation trigger code will work
- System becomes unstable again after (not a real fix)

### Option B: Proper Fix - Escalate Before Threshold
- Monitor for CAPITAL_INSUFFICIENT rejections
- When count reaches 5 (half threshold) → trigger liquidation
- Free capital
- Counter resets naturally (capital available again)
- Signals execute normally
- **This is the architectural fix**

---

## Where to Make the Fix

**File:** `src/l8_lifecycle/meta_controller.py`  
**Function:** Any rejection counter management function (need to find it)  
**Logic:** 

```python
if rejection_reason == "CAPITAL_INSUFFICIENT" and count == 5:
    # Trigger liquidation BEFORE hitting threshold
    liquidation_triggered = await _trigger_liquidation()
    if liquidation_triggered:
        # Reset counter to allow retry
        shared_state.rejection_counters[symbol] = 0
```

---

## Why This Matters

Current architecture:
```
Too many rejections → BLOCK SIGNAL → No escalation → No liquidation ❌
```

Needed architecture:
```
Capital insufficient → ESCALATE → Free capital → Retry → Execute ✅
```

---

## Action Items

1. ✅ Found real blocker: Rejection counter threshold
2. ✅ Added liquidation trigger (but it wasn't reached)
3. ❌ Need to move liquidation trigger EARLIER
4. ❌ Trigger on CAPITAL_INSUFFICIENT rejection (not at threshold)

---

## Temporary Workaround (While We Fix Properly)

Clear the rejection counters manually:

```bash
# In orchestrator, before signals get rejected 10 times:
self.shared_state.rejection_counters.clear()
```

This would unstick the current deadlock BUT would need to be done periodically.

---

## Permanent Fix Strategy

Modify the rejection counter logic to:
1. Detect CAPITAL_INSUFFICIENT rejections specifically
2. Trigger liquidation at rejection count=5 (not 10)
3. Reset counter after liquidation
4. Allow signal to retry with freed capital

This way:
- Capital insufficient → Liquidation triggered → Capital freed → Retry → Execute ✅

---

**Root Cause**: Rejection threshold (10) blocking signals before escalation point  
**Impact**: System stuck in infinite rejection loop  
**Liquidation Fix Applied**: ✅ (but unreachable due to rejection threshold)  
**Proper Fix Needed**: Escalate on CAPITAL_INSUFFICIENT before hitting threshold
