# 🔥 PHASE 2 & 3: Fill Reconciliation & ExecutionManager Integration

**Status**: 🚧 PENDING IMPLEMENTATION  
**Consultant Recommendation**: Wire ExecutionManager to use place_market_order() with fill-aware liquidity release

---

## 📋 Scope of Work

### Phase 2: Fill Reconciliation
Implement authoritative order status checking and reconciliation BEFORE liquidity release.

### Phase 3: ExecutionManager Integration
Wire the three-step execution scope pattern into ExecutionManager.

---

## 🎯 Implementation Plan: Phase 2 (Fill Reconciliation)

### Current Problem
ExecutionManager releases liquidity **immediately after order submission**:

```python
# WRONG: Releases liquidity before fill confirmation
await self._place_market_order_internal(...)  # Returns immediately
await self.shared_state.release_liquidity(...)  # Assumes filled!
```

### The Fix
Check actual Binance order status before releasing:

```python
# CORRECT: Only release after fill confirmation
order = await exchange_client.place_market_order(...)

# Query Binance for ACTUAL fill status
fill_status = order.get("status")  # "FILLED", "PARTIALLY_FILLED", "PENDING", etc.

if fill_status in ["FILLED", "PARTIALLY_FILLED"]:
    # Only NOW release liquidity - we know it was actually used
    spent = float(order.get("cummulativeQuoteQty", planned_quote))
    await self.shared_state.release_liquidity(quote_asset, reservation_id)
else:
    # Order not filled - rollback reservation
    await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
```

### Where This Happens in Code

**File**: `core/execution_manager.py`

**Current locations that need fixing**:
1. Lines ~6413-6456 (`_place_market_order_qty`)
2. Lines ~6520-6570 (`_place_market_order_quote`)

**Pattern to replace**:

**BEFORE (Wrong)**:
```python
raw_order = await self._place_market_order_internal(
    symbol=symbol,
    side=side.upper(),
    quantity=float(qty),
    current_price=current_price,
    planned_quote=float(planned_quote),
    comment=self._sanitize_tag(tag or "meta"),
    ...
)
if not raw_order:
    # Release on error
    await self.shared_state.release_liquidity(quote_asset, reservation_id)
    return {"ok": False, "reason": "order_not_placed"}

# WRONG: Release immediately without checking fill status
if reservation_id:
    spent = float(raw_order.get("cummulativeQuoteQty") or planned_quote)
    await self.shared_state.release_liquidity(quote_asset, reservation_id)
```

**AFTER (Correct)**:
```python
# PHASE 3: Use place_market_order (not _place_market_order_internal)
token = exchange_client.begin_execution_order_scope("ExecutionManager")
try:
    order = await exchange_client.place_market_order(
        symbol=symbol,
        side=side.upper(),
        quantity=qty if qty > 0 else None,
        quote_order_qty=planned_quote if planned_quote > 0 else None,
        tag=self._sanitize_tag(tag or "meta"),
    )
finally:
    exchange_client.end_execution_order_scope(token)

# Check if order was actually placed
if not order or not order.get("orderId"):
    # Rollback: order wasn't placed
    if reservation_id:
        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
        await self._log_execution_event("liquidity_rolled_back", symbol, {
            "asset": quote_asset,
            "amount": float(planned_quote),
            "scope": "buy_by_qty",
            "reason": "order_not_placed",
            "reservation_id": reservation_id
        })
    await self._log_execution_event("order_skip", symbol, {"side": side.upper(), "reason": "order_not_placed"})
    return {"ok": False, "reason": "order_not_placed"}

# PHASE 2: Check fill status BEFORE releasing liquidity
status = str(order.get("status", "")).upper()
filled = status in ("FILLED", "PARTIALLY_FILLED")

if reservation_id:
    if filled:
        # Release: Order was filled (or partially filled)
        spent = float(order.get("cummulativeQuoteQty", planned_quote))
        await self.shared_state.release_liquidity(quote_asset, reservation_id)
        await self._log_execution_event("liquidity_released", symbol, {
            "asset": quote_asset,
            "amount": float(spent),
            "scope": "buy_by_qty",
            "reason": "order_filled",
            "reservation_id": reservation_id,
            "actual_status": status,
        })
    else:
        # Rollback: Order not filled yet (pending, expired, etc.)
        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
        await self._log_execution_event("liquidity_rolled_back", symbol, {
            "asset": quote_asset,
            "amount": float(planned_quote),
            "scope": "buy_by_qty",
            "reason": f"order_not_filled",
            "reservation_id": reservation_id,
            "actual_status": status,
        })

# Continue with post-fill handling (only if filled)
if filled:
    try:
        post_fill = await self._ensure_post_fill_handled(
            symbol,
            side.upper(),
            order,
            tier=None,
            tag=str(tag or ""),
        )
        if side.upper() == "SELL":
            await self._finalize_sell_post_fill(
                symbol=symbol,
                order=order,
                tag=str(tag or ""),
                post_fill=post_fill,
                policy_ctx=None,
                tier=None,
            )
    except Exception as e:
        self.logger.error(f"[POST_FILL_CRASH_DIRECT_PATH] {symbol}: {e}", exc_info=True)

return order
```

---

## 🎯 Implementation Plan: Phase 3 (ExecutionManager Integration)

### Scope Management
ExecutionManager must manage the execution scope:

```python
# INSIDE ExecutionManager._place_market_order_qty() and _place_market_order_quote()

async def _place_market_order_qty(self, symbol: str, qty: float, ...):
    # ... existing code ...
    
    # START SCOPE
    token = self.exchange_client.begin_execution_order_scope("ExecutionManager")
    try:
        # Place order using new method
        order = await self.exchange_client.place_market_order(
            symbol=symbol,
            side=side.upper(),
            quantity=float(qty),
            tag=self._sanitize_tag(tag or "meta"),
        )
    finally:
        # END SCOPE (always, even on exception)
        self.exchange_client.end_execution_order_scope(token)
    
    # ... rest of logic ...
```

### Three-Step Pattern

```
Step 1: begin_execution_order_scope("ExecutionManager")
    ↓
Step 2: await exchange_client.place_market_order(...)
    ↓
Step 3: end_execution_order_scope(token)
```

**Why try/finally?**
- Scope MUST be released even if place_market_order() raises an exception
- Prevents scope from remaining "stuck" active

---

## 📊 Decision Table: When to Release/Rollback Liquidity

| Order Status | Action | Reason |
|--------------|--------|--------|
| `FILLED` | Release | Order 100% filled, liquidity spent |
| `PARTIALLY_FILLED` | Release | Part of order filled, release used portion |
| `PENDING_CANCEL` | Rollback | User cancelled, order not filling |
| `CANCELED` | Rollback | Order was canceled, no fill |
| `EXPIRED` | Rollback | Order timed out, no fill |
| `REJECTED` | Rollback | Binance rejected, never submitted |
| `NEW` | ? | Order accepted but not yet filled - **query again** |

---

## ⚠️ Edge Cases to Handle

### Case 1: NEW Status (Order Queued)
```python
if status == "NEW":
    # Order accepted but not filled yet
    # Option A: Poll for update (simple)
    # Option B: Assume fill is coming, release liquidity on timely basis
    # RECOMMENDATION: Keep reservation active, poll in background
    
    # For now: treat as "not confirmed filled" → Rollback
    await self.shared_state.rollback_liquidity(...)
```

### Case 2: Partial Fill During Submission
```python
if status == "PARTIALLY_FILLED":
    executed_qty = float(order.get("executedQty", 0))
    planned_qty = float(qty)
    
    spent = float(order.get("cummulativeQuoteQty"))
    
    # Release only what was spent
    await self.shared_state.release_liquidity(
        quote_asset,
        reservation_id,
        amount=spent  # Release proportionally
    )
    
    # For remaining qty: Should we re-place?
    # RECOMMENDATION: Let caller decide via async polling
```

### Case 3: Exception During Placement
```python
try:
    order = await self.exchange_client.place_market_order(...)
except Exception as e:
    # Order not placed (network error, auth error, etc.)
    # Rollback reservation
    if reservation_id:
        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
    raise
```

---

## 🔌 Implementation Checklist

### Phase 2: Fill Reconciliation
- [ ] Update `_place_market_order_qty()` to check fill status before release
- [ ] Update `_place_market_order_quote()` to check fill status before release
- [ ] Add `rollback_liquidity()` method to SharedState (if not exists)
- [ ] Log liquidity_released with actual_status
- [ ] Log liquidity_rolled_back with actual_status
- [ ] Test with orders that don't fill immediately

### Phase 3: ExecutionManager Integration
- [ ] Replace `_place_market_order_internal()` calls with `place_market_order()`
- [ ] Add begin_execution_order_scope/end_execution_order_scope pattern
- [ ] Verify try/finally structure
- [ ] Test scope enforcement (should reject orders from outside)
- [ ] Verify client order IDs are unique
- [ ] Check summary events are emitted

### Phase 4: Position Integrity (Separate PR)
- [ ] Update position calculations to use `order["executedQty"]` not `planned_qty`
- [ ] Update risk checks to use actual fills
- [ ] Update capital allocation based on actual spending

---

## 🧪 Testing Strategy

### Unit Tests
```python
async def test_phase2_fill_status_check():
    """Verify liquidity is only released if order filled"""
    
    # Mock order with FILLED status
    order_filled = {"status": "FILLED", "executedQty": 0.1, "cummulativeQuoteQty": 5000}
    # Should release liquidity ✓
    
    # Mock order with NEW status
    order_pending = {"status": "NEW", "executedQty": 0, "cummulativeQuoteQty": 0}
    # Should rollback liquidity ✓
    
    # Mock order with CANCELED status
    order_canceled = {"status": "CANCELED", "executedQty": 0, "cummulativeQuoteQty": 0}
    # Should rollback liquidity ✓
```

### Integration Tests
```python
async def test_phase3_scope_enforcement():
    """Verify ExecutionManager scope is enforced"""
    
    # Should work: place_market_order inside scope
    token = exchange_client.begin_execution_order_scope("ExecutionManager")
    order = await exchange_client.place_market_order(...)  # ✓ works
    exchange_client.end_execution_order_scope(token)
    
    # Should fail: place_market_order outside scope
    order = await exchange_client.place_market_order(...)  # ✗ raises PermissionError
```

### Live Tests
```python
async def test_phase3_paper_live():
    """Run in paper trading to verify end-to-end flow"""
    
    # 1. Place market order (BUY 0.001 BTC)
    # 2. Verify order reaches Binance
    # 3. Check fill status after 1 second
    # 4. If filled: verify liquidity released
    # 5. If pending: verify liquidity rolled back
    # 6. Verify summary events logged
```

---

## 📚 Related Files to Modify

1. **core/execution_manager.py**
   - `_place_market_order_qty()` → Add scope + fill check
   - `_place_market_order_quote()` → Add scope + fill check
   - `_place_market_order_internal()` → Can be removed or deprecated
   - `_place_market_order_core()` → Verify still called

2. **core/shared_state.py**
   - Add `rollback_liquidity()` method (if not exists)
   - Update `release_liquidity()` to support partial amounts

3. **core/exchange_client.py**
   - ✅ `place_market_order()` - Already added (Phase 1)
   - Already has `begin_execution_order_scope()` and `end_execution_order_scope()`

---

## 🚨 Critical Design Decision: Scope-Protected Orders

**Why ExecutionManager scope?**
1. **Single Responsibility** - Only ExecutionManager decides on order timing
2. **Fail Closed** - Accidentally calling place_market_order raises error
3. **Audit Trail** - Scope owner tracked in logs
4. **Circuit Breakers** - ExecutionManager enforces daily loss limits before scope

**Why not allow any caller?**
- Risk of duplicate orders from different code paths
- Risk of orders placed without capital reservation
- Risk of orders placed without risk checks

---

## 📖 Documentation to Create

- [ ] PHASE2_FILL_RECONCILIATION.md
- [ ] PHASE3_EXECUTIONMANAGER_INTEGRATION.md
- [ ] PHASE4_POSITION_INTEGRITY.md

---

## ✅ Success Criteria

Phase 2 & 3 are complete when:

1. ✅ ExecutionManager uses `place_market_order()` (not internal method)
2. ✅ Liquidity is released ONLY after fill confirmation
3. ✅ Non-filled orders cause liquidity rollback
4. ✅ Scope enforcement prevents orders from outside ExecutionManager
5. ✅ All trades reconcile correctly with Binance
6. ✅ Trade journal includes actual_status (not assumed)
7. ✅ Summary events logged with complete metadata
8. ✅ Paper and live trading modes both work correctly

---

## 🎓 Key Principles

| Principle | Implementation |
|-----------|-----------------|
| **Orders are signed** | `signed=True` in place_market_order |
| **Liquidity is reserved before placement** | SharedState.reserve_liquidity() |
| **Liquidity is released after confirmation** | Check fill status first |
| **Scope protects order placement** | Fail closed if outside ExecutionManager |
| **Events are authoritative** | Order status from Binance, not assumptions |
| **Retries are transparent** | Exponential backoff handled by _request() |

