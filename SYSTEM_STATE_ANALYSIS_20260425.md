# System State Analysis - April 25, 2026

## Executive Summary

Your system is **STUCK in an infinite rejection loop** with a phantom position, NOT a dust position problem as initially diagnosed.

**Key Finding**: System tries to SELL ETHUSDT repeatedly with `amount=0.0`, triggering the balance guard rejection "Amount must be positive, got 0.0".

---

## Evidence from Logs

### Pattern: 100+ Consecutive ETHUSDT Sell Rejections
```
2026-04-24 15:42:07 [EXEC_REJECT] symbol=ETHUSDT side=SELL 
reason=balance_guard:invalid_amount:Amount must be positive, got 0.0
status=rejected
```

**Frequency**: Every 10-20 minutes for 24+ hours
**Duration**: Loop frozen at cycle 1195

### What This Tells Us

1. **Position exists in system state** (otherwise wouldn't attempt SELL)
2. **Position quantity is 0.0 or evaluates to 0.0** (being calculated before SELL execution)
3. **System is in retry loop** (same SELL rejection repeating indefinitely)
4. **Previous balance loss (104→103.89 USDT)** created this condition

---

## Root Cause Analysis

### Timeline Hypothesis

**Previous Session (before restart)**:
1. System had ETHUSDT position (some quantity > 0)
2. Executed SELL order with quantity that got rounded DOWN to 0.0001 or 0.0
3. Due to MIN_QTY or step_size filtering, position became unsellable
4. Session ended with frozen position

**Current Session (restart)**:
1. System loads existing position from exchange/state
2. Position state shows: `qty=0.0` or `qty < min_sellable`
3. System knows position exists but can't determine valid sell quantity
4. Enters infinite retry: "Try to sell → Calculate qty=0.0 → Rejected → Retry"

---

## Why My Previous Dust Fixes Didn't Solve This

### What I Fixed
✅ Enhanced dust detection (Lines 9460-9500 in execution_manager.py)
✅ Added stuck detection tracker (Lines 3462-3500)
✅ Made dust prevention 3-layer

### What I Didn't Address
❌ **Position initialization on restart** - System loads state but doesn't validate sellable qty exists
❌ **Zero-quantity position handling** - No circuit breaker for qty=0 cases
❌ **Exchange→State mismatch** - Position may exist on exchange but sync shows 0.0
❌ **Restart impact on position state** - Previous session's phantom position persists

---

## Current System State

### Known Facts
- Process running: YES (loop 1195 frozen, not crashing)
- Orchestrator stable: YES (no crash after 26s anymore)
- Trading loops executing: YES (every 30 seconds)
- Trades being made: NO (all NONE decisions or REJECTED)
- Capital trapped: YES (free balance stuck at ~50 USDT)
- PnL: 0.00 (no progress)

### Position State (Inferred)
```
Symbol: ETHUSDT
Qty: ~0.0 (or < 0.0001 min_qty)
Status: STUCK (can't exit, can't trade new)
Blocking: ALL trading on this symbol
```

### Why Loop Frozen at 1195
- System locks on ETHUSDT symbol
- Attempts SELL → Gets 0.0 qty → Rejected
- Waits 30s
- Loops back, same attempt
- **No error handling to skip/liquidate phantom position**

---

## Restart Effect Analysis

### Restart #1: Added min_positive_edge_bps increase
- **Effect**: Prevented some bad trades BUT
- **Side Effect**: If position was partially exited before restart, phantom qty persists

### Restart #2: Applied dust prevention logic
- **Effect**: Enhanced SELL dust detection BUT
- **Problem**: Doesn't run if qty=0.0 from the start
- **Why**: My code checks `remainder > 0` - if qty already calculated as 0.0, never enters dust logic

---

## The Real Problem

**Your system has a "phantom position" - a position that:**
1. Exists in system state (shows in position list)
2. Has qty=0.0 (or rounded to 0.0)
3. Can't be exited (qty too small for exchange)
4. Can't be skipped (system insists on exiting before new trades)
5. Blocks entire symbol from trading

This is **not a dust remainder problem** (which happens AFTER a valid exit).
This is a **position deletion by rounding problem** (happens DURING initial exit calculation).

---

## Why This Happens

### The Execution Flow

```
SELL Signal Generated
  ↓
Calculate Qty: qty = await _get_sellable_qty(symbol)
  ↓ [Returns 0.0 due to rounding or sync error]
  ↓
Validate Amount: balance_mgr.allocate(amount=0.0)
  ↓
Guard Rejects: "Amount must be positive, got 0.0"
  ↓
Order Rejected
  ↓
Position NOT updated (still shows qty > 0 somewhere)
  ↓
Next cycle: Retry SELL for same position
  ↓ [infinite loop]
```

---

## Critical Questions to Answer Before Fix

Before I propose a solution, I need to understand:

### 1. **Exchange State vs Local State**
   - Does Binance actually have an open ETHUSDT position?
   - Or is it only in your local state?
   - **Check**: `curl -X GET "https://fapi.binance.com/fapi/v2/openOrders?symbol=ETHUSDT"` OR spot equivalent

### 2. **Position Quantity Origin**
   - Where is the 0.0 qty coming from? 
   - Is `_get_sellable_qty()` returning 0.0?
   - Or is `round_step()` rounding something down to 0.0?

### 3. **Previous Session's Partial Exit**
   - In the 104→103.89 USDT loss session, what was the SELL order for ETHUSDT?
   - Did it partially fill?
   - Did it leave a remainder?

### 4. **Restart Impact**
   - After restart, does system sync positions from exchange?
   - Or use cached/persisted state?
   - Is there a stale position entry somewhere?

---

## Proposed Solution Framework

Once we understand the above, the fix should:

1. **Detect phantom positions on startup**
   - Check if qty syncs from exchange
   - If not, mark as phantom

2. **Handle zero-qty positions gracefully**
   - Don't retry SELL forever
   - Either:
     a) Force liquidate as 0 value (close position cleanly)
     b) Skip this symbol temporarily
     c) Force sync from exchange

3. **Prevent future phantom creation**
   - Validate qty > min_qty BEFORE sell execution
   - If qty would round to < min_qty, reject early
   - Don't let order get to balance_guard with 0.0

4. **Add escape hatch for stuck positions**
   - If same symbol rejected 5+ times consecutively
   - Force clear position state
   - Mark symbol for manual review

---

## Immediate Action Plan

**STEP 1**: Scan portfolio to answer critical questions
**STEP 2**: Understand phantom position origin  
**STEP 3**: Propose targeted fix (not general dust prevention)
**STEP 4**: Implement position repair logic
**STEP 5**: Add guards to prevent restart from creating new phantoms

---

## What NOT to Do

❌ Don't blindly increase dust thresholds (masks real problem)
❌ Don't just liquidate everything (loses valid positions)
❌ Don't restart again without fix (creates new restarts in infinite loop)
❌ Don't assume dust fix from previous session solved this (it didn't - this is different issue)

