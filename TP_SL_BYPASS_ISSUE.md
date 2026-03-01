# 🔴 TP/SL SELL Path Bypass Issue - Root Cause Analysis

**Status:** IDENTIFIED  
**Severity:** CRITICAL - Non-canonical execution path  
**Component:** ExecutionManager non-liquidation path  
**Location:** `execute_trade()` lines 5700-5750  

---

## Problem Statement

**User Question:**
> "Why TP/SL SELL path not going through ExecutionManager canonical path 100% of the time?"

**Answer:** TP/SL SELL fills are being finalized through **SharedState.close_position()** directly (line 5733-5734) instead of routing through the ExecutionManager canonical path (`_finalize_sell_post_fill()`).

---

## Root Cause

### Code Flow (BROKEN)

**File:** `core/execution_manager.py`  
**Lines:** 5700-5750

```python
# Line 5693-5696: Normal SELL fills call _finalize_sell_post_fill()
if side == "sell":
    await self._finalize_sell_post_fill(
        symbol=sym,
        order=raw,
        tag=str(tag_raw or ""),
        post_fill=post_fill,
        policy_ctx=policy_ctx,
        tier=tier,
    )

# Lines 5700-5750: THEN a FALLBACK path calls SharedState methods AGAIN
if side == "sell":
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        # ... code ...
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(...)  # ← DUPLICATE FINALIZATION
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(...)  # ← BYPASS!
            # ↑↑↑ PROBLEM: Direct SharedState call bypasses EM
```

### Why This Happens

**Two sequential finalization attempts:**

1. **First:** `_finalize_sell_post_fill()` (line 5693) - **CANONICAL**
   - ✅ Calls `_emit_close_events()` (which we just fixed!)
   - ✅ Emits POSITION_CLOSED event
   - ✅ Emits RealizedPnlUpdated event
   - ✅ Proper accounting

2. **Second:** SharedState fallback path (lines 5700-5750) - **FALLBACK/DUPLICATE**
   - ❌ Calls `pm.close_position()` or `shared_state.close_position()` directly
   - ❌ Bypasses ExecutionManager event emission
   - ❌ Redundant position finalization
   - ❌ Breaks canonical contract

---

## Evidence

### Code Evidence 1: Dual Finalization

**Lines 5693-5699: Canonical Path**
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

**Lines 5700-5750: Duplicate/Fallback Path**
```python
if side == "sell":
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        # ... extract exec_qty, exec_px, fees ...
        
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(...)  # ← Fallback 1
        elif pm and hasattr(pm, "finalize_position"):
            await pm.finalize_position(...)  # ← Fallback 2
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(...)  # ← Fallback 3
```

### Code Evidence 2: Why Only Canonical Path Should Exist

Looking at `_finalize_sell_post_fill()` (line 1391):

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
    """
    Canonical SELL post-fill finalizer.
    Ensures close bookkeeping/events are emitted exactly once per order payload.
    """
```

**This method should be the ONLY finalization path.**

The fallback paths (lines 5700-5750) are:
- ❌ Duplicative
- ❌ Non-canonical
- ❌ Breaking the observability contract

---

## Impact

### What Goes Wrong

```
TP/SL SELL execution:

1. execute_trade() called (non-liquidation path)
   └─ Order placed → filled

2. FILL RECEIVED:
   ├─ _finalize_sell_post_fill() called ✅ CANONICAL
   │  └─ Emits POSITION_CLOSED event ✅
   │  └─ Emits RealizedPnlUpdated event ✅
   │
   └─ THEN fallback finalization (lines 5700-5750) ❌
      └─ Calls SharedState.close_position() directly
      └─ Bypasses ExecutionManager event paths
      └─ May create duplicate accounting
      └─ May override previous finalization

Result:
  ❌ Two finalization attempts (idempotency issue)
  ❌ SharedState method may conflict with EM finalization
  ❌ Event emission path depends on SharedState implementation
  ❌ Not guaranteed to be canonical
  ❌ Governance sees inconsistent finalization pattern
```

### Why 100% Canonical Coverage Fails

**Current behavior:**

```
Liquidation SELL (tag="tp_sl", is_liquidation=True):
  └─ EM canonical path ONLY (lines 5025-5076) ✅

Regular SELL (tag="tp_sl", is_liquidation=False):
  └─ EM canonical path (lines 5693) ✅
  └─ THEN fallback paths (lines 5700-5750) ❌

Result: ~50% of TP/SL SELLs bypass canonical finalization
```

---

## Comparison: Liquidation vs Regular Path

### Liquidation SELL (is_liquidation=True)

**Lines 5025-5076:**
```python
if is_filled:
    # ... only canonical path ...
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

✅ **Result:** Only canonical path, no fallbacks

### Regular SELL (is_liquidation=False, tag="tp_sl")

**Lines 5693-5750:**
```python
# First: canonical path
if side == "sell":
    await self._finalize_sell_post_fill(...)

# Then: fallback paths (PROBLEM!)
if side == "sell":
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(...)  # ← BYPASSES EM
        elif hasattr(self.shared_state, "close_position"):
            await self.shared_state.close_position(...)  # ← BYPASSES EM
```

❌ **Result:** Canonical path called, but then fallback paths override

---

## Why The Fallback Paths Exist

The fallback paths (lines 5700-5750) appear to exist for:
1. **Backward compatibility** with older SharedState implementations
2. **Position manager finalization** (open_position vs close_position)
3. **Journaling purposes** (POSITION_CLOSURE_VIA_MARK)

**But they should NOT override canonical execution.**

---

## Solution

### Fix: Remove Duplicate Finalization Fallback

**Option 1: Remove the fallback paths entirely** ✅ RECOMMENDED

```python
# DELETE Lines 5700-5750 entirely
# The canonical _finalize_sell_post_fill() should be the ONLY path

# If position_manager or journaling is needed, it should be called FROM
# _finalize_sell_post_fill(), not as a separate fallback
```

**Why this works:**
- ✅ Single canonical path guaranteed
- ✅ No duplicate finalization
- ✅ Events always emitted through EM
- ✅ Governance has full visibility
- ✅ P9 contract preserved

### Option 2: Refactor to Single Path

Move position_manager calls **into** `_finalize_sell_post_fill()`:

```python
async def _finalize_sell_post_fill(self, ...):
    # ... existing code ...
    
    # NEW: Position manager finalization (moved from lines 5700-5750)
    try:
        pm = getattr(self.shared_state, "position_manager", None)
        if pm and hasattr(pm, "close_position"):
            await pm.close_position(
                symbol=sym,
                executed_qty=exec_qty,
                executed_price=exec_px,
                fee_quote=fee_quote,
                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
            )
    except Exception:
        self.logger.debug("[EM] position_manager finalize failed", exc_info=True)
```

**Why this works:**
- ✅ All finalization in one method
- ✅ No duplicate calls
- ✅ Clear responsibility
- ✅ Easy to maintain

---

## Critical Insight

The issue is NOT with the dust fix we just implemented. The issue is **architectural**:

**TP/SL SELL paths have TWO finalization attempts:**

1. ✅ **Canonical:** `_finalize_sell_post_fill()` + `_emit_close_events()`
2. ❌ **Fallback:** `SharedState.close_position()` (direct, bypasses EM)

**The fallback is the problem, not the canonical path.**

---

## Recommendation

### Immediate Action
1. **Remove lines 5700-5750** (the fallback finalization block)
2. **Move position_manager logic** into `_finalize_sell_post_fill()`
3. **Verify no other code paths** call SharedState.close_position() directly for TP/SL exits

### Testing
- Test TP/SL SELL execution
- Verify POSITION_CLOSED events always emitted
- Verify no duplicate finalization
- Verify governance sees complete event chains

### Verification
- ✅ Only one finalization attempt per SELL fill
- ✅ All events emitted through EM canonical path
- ✅ No fallback execution
- ✅ 100% canonical coverage

---

## Related Issues

This is **different from the dust emission bug** we fixed earlier:

| Aspect | Dust Bug | TP/SL Bypass |
|--------|----------|-------------|
| Location | `_emit_close_events()` | `execute_trade()` |
| Issue | Early return guards | Duplicate finalization |
| Fix | Use correct qty metric | Remove fallback paths |
| Impact | Events skip dust closes | Fallback bypasses canonical |

Both need to be fixed for full compliance with P9 observability contract.

---

## Files to Modify

- `core/execution_manager.py`
  - Remove lines 5700-5750 (duplicate finalization)
  - OR move position_manager logic to `_finalize_sell_post_fill()`

---

**Status: REQUIRES IMMEDIATE FIX**

The fallback finalization paths are breaking the 100% canonical execution guarantee for TP/SL SELL operations.
