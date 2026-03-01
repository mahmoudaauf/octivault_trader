# TP/SL SELL Canonicality Fix - Implementation Plan

**Issue:** Non-canonical finalization paths breaking 100% canonical coverage  
**Root Cause:** Duplicate `_finalize_sell_post_fill()` call + fallback position_manager calls  
**Severity:** CRITICAL  
**File:** `core/execution_manager.py`  
**Lines:** 5689-5760 (duplicate finalization)

---

## Problem Summary

### Current (BROKEN) Flow

```
Non-liquidation SELL (tag="tp_sl", is_liquidation=False):

Line 5689: await self._finalize_sell_post_fill(...)  ✅ CANONICAL
    └─ Emits POSITION_CLOSED event
    └─ Emits RealizedPnlUpdated event
    └─ Full EM accounting

Line 5700-5750: FALLBACK finalization ❌ NON-CANONICAL
    ├─ pm.close_position(...)
    ├─ pm.finalize_position(...)
    ├─ shared_state.close_position(...)
    └─ shared_state.mark_position_closed(...)
    
    These bypass the ExecutionManager event paths
    These are NOT observed by governance/auditing
    These duplicate the canonical finalization
```

### Why This Breaks Canonicality

The problem is **dual finalization attempts**:

1. **First attempt (canonical):** `_finalize_sell_post_fill()` at line 5689
   - ✅ Routes through ExecutionManager event emission
   - ✅ Calls `_emit_close_events()` (which we just fixed!)
   - ✅ Emits POSITION_CLOSED event
   - ✅ Records in EM accounting

2. **Second attempt (fallback):** Lines 5700-5750
   - ❌ Calls SharedState methods directly
   - ❌ BYPASSES ExecutionManager event paths
   - ❌ Governs doesn't see this finalization
   - ❌ Duplicate accounting possible

**Result:** TP/SL SELLs are NOT 100% canonical because they have a fallback path that bypasses EM.

---

## Evidence

### Code Evidence: Exact Location

**File:** `core/execution_manager.py`

**Line 5689-5700: Canonical Path (1st)**
```python
if side == "sell":
    await self._finalize_sell_post_fill(
        symbol=sym,
        order=raw,
        tag=str(tag_raw or ""),
        post_fill=post_fill,
        policy_ctx=policy_ctx,
        tier=tier,
    )
```

**Line 5700-5750: Fallback/Duplicate Path (2nd)**
```python
# Finalize position on SELL fills
if side == "sell":
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        exec_qty = float(raw.get("executedQty", 0.0))
        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
        fee_quote = float(raw.get("fee_quote", 0.0) or raw.get("fee", 0.0) or 0.0)
        
        # TRY 1: position_manager.close_position()
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(
                symbol=sym,
                executed_qty=exec_qty,
                executed_price=exec_px,
                fee_quote=fee_quote,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
            )
        
        # TRY 2: position_manager.finalize_position()
        elif pm and hasattr(pm, "finalize_position"):
            await pm.finalize_position(
                symbol=sym,
                executed_qty=exec_qty,
                executed_price=exec_px,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
            )
        
        # TRY 3: shared_state.close_position() ← DIRECT BYPASS!
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(
                sym,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
            )
        
        # Then mark closed
        self._journal("POSITION_CLOSURE_VIA_MARK", {...})
        if hasattr(self.shared_state, "mark_position_closed"):
            await self.shared_state.mark_position_closed(...)
    except Exception:
        self.logger.debug("[EM] finalize_position failed for %s", sym, exc_info=True)
```

---

## Root Cause Analysis

### Why The Fallback Exists

The fallback path (lines 5700-5750) exists to handle:

1. **Position Manager Finalization** - Some implementations of SharedState have position_manager
2. **Journaling** - Tracks "POSITION_CLOSURE_VIA_MARK" events
3. **Backward Compatibility** - Older code paths called position_manager directly

**But it should NOT exist as a separate path.**

### Why It's A Problem

**These fallback calls are NOT going through the canonical EM event path:**

```python
# The fallback calls:
await pm.close_position(...)        # ← SharedState method, NOT EM
await pm.finalize_position(...)     # ← SharedState method, NOT EM
await self.shared_state.close_position(...)  # ← Direct SharedState, NOT EM
await self.shared_state.mark_position_closed(...)  # ← Direct SharedState, NOT EM

# What's missing:
# - No TRADE_EXECUTED event (could be skipped if fallback takes over)
# - No POSITION_CLOSED event (emitted by _finalize_sell_post_fill() but may be overridden)
# - No RealizedPnlUpdated event (canonical path only)
# - No governance audit trail (fallback is silent)
```

---

## Comparison: Liquidation vs Non-Liquidation

### Liquidation Path (is_liquidation=True) ✅ CORRECT

**Lines 5025-5076:**
```python
if is_filled:
    try:
        if not merged.get("_post_fill_done"):
            pf_result = await self._handle_post_fill(...)
            merged["_post_fill_done"] = True
            merged["_post_fill_result"] = pf_result
    except Exception:
        ...

    try:
        await self._finalize_sell_post_fill(
            symbol=sym,
            order=merged,
            tag=str(clean_tag or ""),
            post_fill=merged.get("_post_fill_result") or {},
            policy_ctx=policy_ctx,
            tier=tier,
        )
    except Exception:
        ...
```

**Result:** ✅ ONLY canonical path, NO fallback

### Non-Liquidation Path (is_liquidation=False, tag="tp_sl") ❌ BROKEN

**Lines 5689-5750:**
```python
# First: canonical path ✅
if side == "sell":
    await self._finalize_sell_post_fill(...)

# Second: fallback paths ❌ (PROBLEM!)
if side == "sell":
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(...)  # ← FALLBACK
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(...)  # ← FALLBACK
        # ...mark_position_closed()...
```

**Result:** ❌ CANONICAL path called, BUT then fallback overrides it

---

## The Fix

### Option A: Remove Fallback (RECOMMENDED)

**Delete lines 5700-5750 entirely.**

The canonical `_finalize_sell_post_fill()` should be the ONLY finalization.

```python
# DELETE from line 5700 to 5750

# Keep only:
if side == "sell":
    await self._finalize_sell_post_fill(
        symbol=sym,
        order=raw,
        tag=str(tag_raw or ""),
        post_fill=post_fill,
        policy_ctx=policy_ctx,
        tier=tier,
    )

# That's it. No fallback.
```

**Why this works:**
- ✅ Single canonical path only
- ✅ No duplicate finalization
- ✅ All events go through EM
- ✅ Governance has full visibility
- ✅ 100% canonical guaranteed

**Risks:**
- If SharedState position_manager finalization is needed, it must be called from `_finalize_sell_post_fill()`
- Check if any code depends on the fallback behavior

---

### Option B: Move Fallback into Canonical (ALTERNATIVE)

If position_manager finalization is genuinely needed, move it **inside** `_finalize_sell_post_fill()`:

**Modify:** `_finalize_sell_post_fill()` (line 1391)

```python
async def _finalize_sell_post_fill(
    self,
    *,
    symbol: str,
    order: Optional[Dict[str, Any]],
    tag: str = "",
    post_fill: Optional[Dict[str, Any]] = None,
    policy_ctx: Optional[Dict[str, Any]] = None,
    tier: Optional[str] = None,
) -> None:
    """Canonical SELL post-fill finalizer."""
    
    # ... existing code ...
    
    # NEW: Position manager finalization (moved from lines 5700-5750)
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        exec_qty = float(order.get("executedQty", 0.0))
        exec_px = float(order.get("avgPrice", order.get("price", 0.0)) or 0.0)
        fee_quote = float(order.get("fee_quote", 0.0) or order.get("fee", 0.0) or 0.0)
        
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(
                symbol=symbol,
                executed_qty=exec_qty,
                executed_price=exec_px,
                fee_quote=fee_quote,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED") if policy_ctx else "SELL_FILLED",
            )
        elif pm and hasattr(pm, "finalize_position"):
            await pm.finalize_position(
                symbol=symbol,
                executed_qty=exec_qty,
                executed_price=exec_px,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED") if policy_ctx else "SELL_FILLED",
            )
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(
                symbol,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED") if policy_ctx else "SELL_FILLED",
            )
        
        # Journal closure
        self._journal("POSITION_CLOSURE_VIA_MARK", {
            "symbol": symbol,
            "executed_qty": exec_qty,
            "executed_price": exec_px,
            "reason": str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED") if policy_ctx else "SELL_FILLED",
            "tag": tag,
            "timestamp": time.time(),
        })
        
        if hasattr(self.shared_state, "mark_position_closed"):
            await self.shared_state.mark_position_closed(
                symbol=symbol,
                qty=exec_qty,
                price=exec_px,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED") if policy_ctx else "SELL_FILLED",
                tag=tag,
            )
    except Exception:
        self.logger.debug("[EM] position_manager finalize failed for %s", symbol, exc_info=True)
```

**Then delete lines 5700-5750 from execute_trade().**

**Why this works:**
- ✅ All finalization in one canonical method
- ✅ Position manager still called, but through canonical path
- ✅ Single point of execution control
- ✅ Clear responsibility segregation

---

## Implementation Steps

### Step 1: Identify Dependencies

Check if code depends on the fallback behavior:

```bash
grep -n "position_manager.close_position\|mark_position_closed" \
  core/execution_manager.py \
  core/shared_state.py \
  agents/*.py
```

### Step 2: Apply Fix (Recommended: Option A)

1. Delete lines 5700-5750 from `execute_trade()`
2. Keep the canonical `_finalize_sell_post_fill()` call

### Step 3: Verify

Run tests:
- ✅ TP/SL SELL execution
- ✅ POSITION_CLOSED events always emitted
- ✅ RealizedPnlUpdated events emitted
- ✅ No duplicate finalization
- ✅ Governance sees complete event chain

### Step 4: Document

Update documentation to reflect:
- ✅ TP/SL SELL now 100% canonical
- ✅ Single finalization path only
- ✅ Full event emission guaranteed

---

## Impact Analysis

### What Gets Fixed

```
BEFORE (BROKEN):
TP/SL SELL execution:
├─ _finalize_sell_post_fill() called ✅ CANONICAL
├─ POSITION_CLOSED event emitted ✅
└─ fallback pm.close_position() called ❌ NON-CANONICAL
   ├─ May override canonical finalization
   └─ May skip EM accounting

AFTER (FIXED):
TP/SL SELL execution:
└─ _finalize_sell_post_fill() called ✅ CANONICAL
   ├─ POSITION_CLOSED event emitted ✅
   ├─ RealizedPnlUpdated event emitted ✅
   ├─ Position manager finalization called ✅
   └─ All through canonical EM path ✅
```

### Affected Operations

- ✅ TP/SL SELL fills (non-liquidation)
- ✅ Regular SELL fills (will now be more canonical)
- ✅ Dust position closes (will work correctly with our earlier fix)

### Event Emission Impact

```
POSITION_CLOSED event:
  BEFORE: May be emitted by canonical path OR skipped by fallback
  AFTER: Always emitted by canonical path ✅

RealizedPnlUpdated event:
  BEFORE: May be missing if fallback takes over
  AFTER: Always emitted by canonical path ✅

TRADE_EXECUTED event:
  BEFORE: Depends on canonical path execution
  AFTER: Guaranteed by canonical path ✅
```

---

## Related Issues

This fix complements the dust emission fix from earlier:

| Fix | Location | Issue | Solution |
|-----|----------|-------|----------|
| Dust Emission | `_emit_close_events()` | Early return for dust | Use correct qty metric |
| TP/SL Bypass | `execute_trade()` lines 5700-5750 | Duplicate finalization | Remove fallback paths |

Together, these ensure:
- ✅ 100% canonical execution for all SELL closes
- ✅ 100% event emission for all closes (including dust)
- ✅ Complete governance visibility
- ✅ P9 observability contract preserved

---

## Recommendation

**Apply Option A (Remove Fallback)** because:

1. **Canonical Clarity:** Single clear finalization path
2. **No Duplication:** Eliminates redundant calls
3. **Event Guarantee:** All events through EM only
4. **Governance Compliance:** Full audit visibility
5. **Dust Fix Compatibility:** Works seamlessly with earlier fix

**Estimated Impact:** Low risk, high compliance gain
**Estimated Effort:** 5 minutes to delete lines 5700-5750
**Testing Required:** TP/SL SELL execution tests

---

## Verification Checklist

After applying the fix:

- [ ] Lines 5700-5750 deleted from `execute_trade()`
- [ ] `_finalize_sell_post_fill()` still called for SELL fills
- [ ] Test TP/SL SELL execution (trigger take-profit)
- [ ] Verify POSITION_CLOSED event emitted
- [ ] Verify RealizedPnlUpdated event emitted
- [ ] Verify no duplicate finalization in logs
- [ ] Check governance audit sees complete chain
- [ ] Verify dust fixes still work
- [ ] Run full test suite

---

**Status:** READY FOR IMPLEMENTATION
