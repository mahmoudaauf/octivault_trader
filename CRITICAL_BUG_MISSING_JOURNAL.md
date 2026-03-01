# CRITICAL BUG: Missing ORDER_FILLED Journal in Quote Path

**Severity**: 🔴 CRITICAL - State Synchronization Invariant Violation

**Detection**: During state sync verification (2025-01-XX)

---

## The Problem

When orders are placed via the **quote path** (`_place_market_order_quote()`), the system:

1. ✅ Places the order at Binance
2. ✅ Receives the filled order response
3. ✅ Updates the position in SharedState via `_update_position_from_fill()`
4. ✅ Performs post-fill handling
5. ❌ **NEVER JOURNALS THE ORDER_FILLED EVENT**

This violates the critical invariant: **"All state mutations must be tracked via the journal"**

## Exact Location

**File**: `core/execution_manager.py`
**Method**: `_place_market_order_quote()` (lines 6626-6790)
**Missing**: `self._journal("ORDER_FILLED", {...})`

### Code Path
```python
async def _place_market_order_quote(...):
    # Line 6686: Place order at Binance
    raw_order = await self.exchange_client.place_market_order(...)
    
    # Lines 6701-6703: Check if order was placed
    if not raw_order or not raw_order.get("orderId"):
        return {"ok": False, "reason": "order_not_placed"}
    
    # Line 6706: Determine if filled
    status = str(raw_order.get("status", "")).upper()
    is_filled = status in ("FILLED", "PARTIALLY_FILLED")
    
    # Lines 6708-6718: UPDATE POSITION IN SHAREDSTATE
    if is_filled:
        position_updated = await self._update_position_from_fill(
            symbol=symbol,
            side=side,
            order=raw_order,
            tag=str(tag or "")
        )
    
    # Lines 6739-6750: Handle post-fill
    if raw_order and is_filled:
        post_fill = await self._ensure_post_fill_handled(...)
        # ... finalize handling ...
    
    # Line 6775: Return raw order
    return raw_order
    
    # ❌ NO JOURNAL CREATED - BUG!
```

## Why This Breaks Invariants

### Failure Mode
1. ExecutionManager calls `_place_market_order_quote()`
2. Quote path places order → receives FILLED status → updates position
3. **But no ORDER_FILLED journal entry is created**
4. Later, TruthAuditor runs and checks journals for ORDER_FILLED events
5. **TruthAuditor sees position in SharedState but no journal entry**
6. If position came from ExchangeClient callback (orders through event stream):
   - ExchangeClient logs ORDER_FILLED via `_journal()`
   - But two separate journal entries (confusing)
7. If position only from quote path:
   - **SharedState has position change, but no journal entry at all**
   - **This violates the single source of truth invariant**

### Consequences
- **Audit trail is incomplete**: Can't reconstruct state from journals
- **TruthAuditor confusion**: Position exists but no ORDER_FILLED entry to match it
- **Orphan detection failures**: Orphan detection relies on ORDER_FILLED journal events
- **Recovery impossible**: System can't replay state from journals if crashed

## Comparison: Other Paths

All other order placement paths journal ORDER_FILLED:

### Bootstrap Path (Line 7061)
```python
if self._is_order_fill_confirmed(order):
    self._journal("ORDER_FILLED", {
        "symbol": symbol,
        "side": "BUY",
        "executed_qty": ...,
        "avg_price": ...,
        "order_id": ...,
        "status": ...,
        "tag": safe_tag,
        "path": "bootstrap_quote",
    })
```

### Standard Execute Path (Line 7329)
```python
if self._is_order_fill_confirmed(order):
    self._journal("ORDER_FILLED", {
        "symbol": symbol,
        "side": side.upper(),
        "executed_qty": ...,
        "avg_price": ...,
        "order_id": ...,
        "status": ...,
        "tag": safe_tag,
    })
```

### Quote Path (Line 6626)
```python
# ❌ MISSING - NO JOURNAL
```

## The Fix

Add ORDER_FILLED journaling to `_place_market_order_quote()` after position update succeeds.

### Location to Add Fix
After line 6718 (after position update), add:
```python
if position_updated:  # Only journal if position was actually updated
    self._journal("ORDER_FILLED", {
        "symbol": symbol,
        "side": side.upper(),
        "executed_qty": float(raw_order.get("executedQty", 0.0) or 0.0),
        "avg_price": self._resolve_post_fill_price(
            raw_order,
            float(raw_order.get("executedQty", 0.0) or 0.0)
        ),
        "cumm_quote": float(raw_order.get("cummulativeQuoteQty", quote) or quote),
        "order_id": str(raw_order.get("orderId", "")),
        "status": str(raw_order.get("status", "")),
        "tag": str(tag or ""),
        "path": "quote_based",
    })
```

## State Synchronization Timeline

### Current (BROKEN)
```
1. ExecutionManager → _place_market_order_quote()
2. → ExchangeClient.place_market_order() [returns FILLED order]
3. → _update_position_from_fill() [position updated in SharedState] ✅
4. → Post-fill handling ✅
5. → Return to caller ❌ NO JOURNAL
6. Later: TruthAuditor runs
7. → Finds position in SharedState
8. → Searches journals for matching ORDER_FILLED
9. → NOT FOUND ❌ INVARIANT VIOLATION
```

### Fixed
```
1. ExecutionManager → _place_market_order_quote()
2. → ExchangeClient.place_market_order() [returns FILLED order]
3. → _update_position_from_fill() [position updated in SharedState] ✅
4. → self._journal("ORDER_FILLED", {...}) [journal entry created] ✅ FIX
5. → Post-fill handling ✅
6. → Return to caller ✅
7. Later: TruthAuditor runs
8. → Finds position in SharedState
9. → Searches journals for matching ORDER_FILLED
10. → FOUND ✅ INVARIANT MAINTAINED
```

## Testing After Fix

1. **Unit Test**: Verify ORDER_FILLED journal is created for filled quote orders
2. **Integration Test**: Execute quote order, verify:
   - Position updated in SharedState
   - ORDER_FILLED journal entry created
   - Journal entry matches position update
3. **TruthAuditor Test**: Run auditor after quote order, verify no invariant violations
4. **Replay Test**: Reconstruct state from journals, verify completeness

## Files Affected
- `core/execution_manager.py` - Missing journal in `_place_market_order_quote()`
- `core/shared_state.py` - Uses journals for validation (TruthAuditor context)
- `core/truth_auditor.py` - Searches journals for ORDER_FILLED events

## Priority
🔴 **CRITICAL** - This violates a core architectural invariant. Must be fixed before any quote-based orders are placed in production.

---

**Detection**: Initial state sync verification
**Status**: Ready for fix application
**Impact**: All quote-based market orders (BUY/SELL via `quote_order_qty`)
