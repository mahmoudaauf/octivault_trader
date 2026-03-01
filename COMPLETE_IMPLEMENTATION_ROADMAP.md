# 🚀 COMPLETE IMPLEMENTATION ROADMAP: Live-Safe Order Execution

**Consultant Level Recommendation**  
**Date**: February 25, 2026  
**Status**: Phase 1 ✅ COMPLETE | Phase 2-3 🚧 IMPLEMENTATION GUIDE | Phase 4 📋 PLANNED

---

## 🎯 Executive Summary

We are implementing a three-phase refactor to achieve **live-safe order execution** with proper scope enforcement and fill-aware liquidity management.

**Problem**: Orders are placed without proper scope guards, and liquidity is released before fill confirmation.

**Solution**: 
1. ✅ **Phase 1**: Add `place_market_order()` method to ExchangeClient with scope enforcement
2. 🚧 **Phase 2**: Implement fill-aware liquidity release in ExecutionManager
3. 🚧 **Phase 3**: Wire ExecutionManager to use place_market_order with scope pattern
4. 📋 **Phase 4**: Validate position updates use actual fills (executedQty)

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ExecutionManager                         │
│                                                                 │
│  Reserve Liquidity → Begin Scope → Place Order → Check Fill    │
│       ↓                   ↓            ↓            ↓           │
│  SharedState       Scope Token    Order Placed   Fill Check     │
│       ✓                   ✓            ✓            ✓           │
│                                                                 │
│                   IF FILLED: Release Liquidity                 │
│                   ELSE: Rollback Reservation                   │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    End Scope
                         ↓
            ┌────────────────────────────┐
            │   ExchangeClient Methods   │
            ├────────────────────────────┤
            │ place_market_order()       │
            │ begin_execution_order_scope│
            │ end_execution_order_scope()│
            │ _guard_execution_path()    │
            │ _request() [signed]        │
            └────────────────────────────┘
                         ↓
              ┌──────────────────────┐
              │  Binance /api/v3/order  │
              │  (Signed POST Request)  │
              └──────────────────────┘
```

---

## 📋 PHASE 1: Order Placement Method Restoration

**Status**: ✅ **COMPLETE**

### What Was Done
Added `async def place_market_order()` to ExchangeClient with:
- Execution scope enforcement via `_guard_execution_path()`
- Parameter validation (quantity OR quote_order_qty)
- Client order ID generation (octi-<timestamp>-<tag>)
- Signed API request to /api/v3/order
- Summary event emission (ORDER_SUBMITTED / ORDER_FAILED)
- No liquidity release (by design)

### Key Code
**File**: `core/exchange_client.py` (lines ~1042-1168)

```python
async def place_market_order(
    self,
    symbol: str,
    side: str,
    *,
    quantity: Optional[float] = None,
    quote_order_qty: Optional[float] = None,
    tag: str = "",
) -> Dict[str, Any]:
    """Guards execution path, calls _request("POST", "/api/v3/order", ...)"""
    sym = self._norm_symbol(symbol)
    side = side.upper()
    
    # Guard 1: Enforce scope
    await self._guard_execution_path(method="place_market_order", symbol=sym, side=side, tag=tag)
    
    # Guard 2: Require quantity or quote_order_qty
    if not quantity and not quote_order_qty:
        raise ValueError("Either quantity or quote_order_qty must be provided")
    
    # Build params with newClientOrderId
    params = {
        "symbol": sym,
        "side": side,
        "type": "MARKET",
        "newClientOrderId": f"octi-{int(time.time()*1000)}-{str(tag)[:16]}",
    }
    if quantity and quantity > 0:
        params["quantity"] = float(quantity)
    if quote_order_qty and quote_order_qty > 0:
        params["quoteOrderQty"] = float(quote_order_qty)
    
    try:
        # Execute signed POST to /api/v3/order
        response = await self._request("POST", "/api/v3/order", params, signed=True, api="spot_api")
        
        # Emit success event
        await self._emit_summary("ORDER_SUBMITTED", symbol=sym, side=side, status=response.get("status"), ...)
        
        return response
    except Exception as e:
        # Emit failure event
        await self._emit_summary("ORDER_FAILED", symbol=sym, side=side, status="ERROR", ...)
        raise
```

### Validation
✅ No syntax errors  
✅ Scope enforcement in place  
✅ Exponential backoff via _request()  
✅ Summary events emitted  
✅ Raises loudly on failure  

**Documentation**: `PHASE1_ORDER_PLACEMENT_RESTORATION.md`

---

## 🚧 PHASE 2: Fill Reconciliation

**Status**: 📋 **IMPLEMENTATION GUIDE READY**

### What Needs to Change
Replace premature liquidity release with **fill-aware** release logic.

**Current Problem** (WRONG):
```python
# Order submitted, immediately release liquidity (assumes filled)
order = await self._place_market_order_internal(...)
await self.shared_state.release_liquidity(...)  # ❌ Assumes filled!
```

**Solution** (CORRECT):
```python
# Order submitted, check fill status BEFORE releasing
order = await exchange_client.place_market_order(...)
status = order.get("status")  # "FILLED", "PARTIALLY_FILLED", "NEW", etc.

if status in ["FILLED", "PARTIALLY_FILLED"]:
    # Release: order actually filled
    await self.shared_state.release_liquidity(...)  # ✓ Confirmed filled
else:
    # Rollback: order not filled
    await self.shared_state.rollback_liquidity(...)  # ✓ No fill confirmed
```

### Files to Modify
1. `core/execution_manager.py` (lines ~6413-6570)
   - `_place_market_order_qty()`
   - `_place_market_order_quote()`
2. `core/shared_state.py`
   - Add `rollback_liquidity()` method

### Decision Table
```
Order Status          | Action       | Reason
─────────────────────┼──────────────┼────────────────────
FILLED                | Release      | 100% filled
PARTIALLY_FILLED      | Release      | Partial fill
NEW / PENDING_CANCEL  | Rollback     | Not filled yet
CANCELED              | Rollback     | User canceled
EXPIRED               | Rollback     | Timed out
REJECTED              | Rollback     | Never submitted
```

**Documentation**: `PHASE2_3_FILL_RECONCILIATION_INTEGRATION.md` (Phase 2 section)

---

## 🚧 PHASE 3: ExecutionManager Integration

**Status**: 📋 **IMPLEMENTATION GUIDE READY**

### What Needs to Change
Wire ExecutionManager to use the three-step scope pattern:

```
Step 1: token = exchange_client.begin_execution_order_scope("ExecutionManager")
Step 2: order = await exchange_client.place_market_order(...)
Step 3: exchange_client.end_execution_order_scope(token)
```

### Pattern to Implement

**File**: `core/execution_manager.py`

```python
async def _place_market_order_qty(self, symbol: str, qty: float, ...):
    # ... reserve liquidity ...
    reservation_id = await self.shared_state.reserve_liquidity(quote_asset, quote)
    
    # PHASE 3: Scope pattern
    token = self.exchange_client.begin_execution_order_scope("ExecutionManager")
    try:
        order = await self.exchange_client.place_market_order(
            symbol=symbol,
            side=side.upper(),
            quantity=float(qty),
            tag=self._sanitize_tag(tag or "meta"),
        )
    finally:
        self.exchange_client.end_execution_order_scope(token)
    
    # PHASE 2: Fill-aware release
    if order and order.get("orderId"):
        status = str(order.get("status", "")).upper()
        if status in ["FILLED", "PARTIALLY_FILLED"]:
            spent = float(order.get("cummulativeQuoteQty", quote))
            await self.shared_state.release_liquidity(quote_asset, reservation_id)
        else:
            await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
    else:
        await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
    
    # ... post-fill handling (only if filled) ...
    return order
```

### Key Points
- ✅ try/finally ensures scope is always released
- ✅ place_market_order is called INSIDE scope
- ✅ Liquidity release depends on fill status
- ✅ Order response is authoritative (from Binance)

**Documentation**: `PHASE2_3_FILL_RECONCILIATION_INTEGRATION.md` (Phase 3 section)

---

## 📋 PHASE 4: Position Integrity

**Status**: 🔮 **PLANNED**

### What Needs to Change
Use actual fills (executedQty) for position updates, not planned quantities.

**Current Problem** (WRONG):
```python
# Position updated with PLANNED amount, not actual fill
position_qty = planned_qty  # ❌ What we wanted to buy
```

**Solution** (CORRECT):
```python
# Position updated with ACTUAL fill amount
position_qty = float(order.get("executedQty", 0))  # ✓ What actually filled
```

### Files to Modify
1. `core/position_manager.py` (or similar)
   - Update `update_position()` to use executedQty
2. `core/execution_manager.py`
   - Update position calculation logic
3. `core/capital_allocator.py`
   - Update capital usage to actual spending

### Details
```python
# Get actual fill data
executed_qty = float(order.get("executedQty", 0))
actual_quote_spent = float(order.get("cummulativeQuoteQty", 0))
average_fill_price = actual_quote_spent / executed_qty if executed_qty > 0 else current_price

# Update position with actual values
await self._update_position(
    symbol=symbol,
    actual_quantity=executed_qty,  # Use actual fill
    actual_price=average_fill_price,
    order_id=order["orderId"],
)

# Adjust capital allocation by actual spending
await self._adjust_capital_allocation(
    spent=actual_quote_spent,  # Not planned amount
    remaining=planned_quote - actual_quote_spent,
)
```

---

## 🔄 Complete Execution Flow

```python
# EXECUTION MANAGER
async def execute_buy(self, symbol: str, quote: float):
    symbol = self._norm_symbol(symbol)
    quote_asset = self._split_symbol_quote(symbol)
    
    # ─────────────────────────────────────────────────────
    # PHASE 0: Pre-checks
    # ─────────────────────────────────────────────────────
    ok, why = await self.explain_afford_market_buy(symbol, Decimal(str(quote)))
    if not ok:
        return {"ok": False, "reason": why}
    
    # ─────────────────────────────────────────────────────
    # PHASE 1a: Reserve Liquidity
    # ─────────────────────────────────────────────────────
    reservation_id = await self.shared_state.reserve_liquidity(
        quote_asset,
        float(quote),
        ttl_seconds=30
    )
    
    try:
        # ─────────────────────────────────────────────────
        # PHASE 1b: Begin Execution Scope
        # ─────────────────────────────────────────────────
        token = self.exchange_client.begin_execution_order_scope("ExecutionManager")
        
        try:
            # ─────────────────────────────────────────────
            # PHASE 1c: Place Market Order (Signed)
            # ─────────────────────────────────────────────
            order = await self.exchange_client.place_market_order(
                symbol=symbol,
                side="BUY",
                quote_order_qty=float(quote),
                tag=f"buy-{symbol}",
            )
            
            if not order or not order.get("orderId"):
                # Order not placed
                raise BinanceAPIException("Order not placed", code=-1)
        
        finally:
            # ─────────────────────────────────────────────
            # PHASE 1d: End Execution Scope
            # ─────────────────────────────────────────────
            self.exchange_client.end_execution_order_scope(token)
        
        # ─────────────────────────────────────────────────
        # PHASE 2: Check Fill Status (Authoritative)
        # ─────────────────────────────────────────────────
        status = str(order.get("status", "")).upper()
        executed_qty = float(order.get("executedQty", 0))
        actual_quote_spent = float(order.get("cummulativeQuoteQty", 0))
        
        is_filled = status in ["FILLED", "PARTIALLY_FILLED"] and executed_qty > 0
        
        if is_filled:
            # ─────────────────────────────────────────────
            # PHASE 2a: Release Liquidity (Confirmed Filled)
            # ─────────────────────────────────────────────
            await self.shared_state.release_liquidity(
                quote_asset,
                reservation_id,
                amount=actual_quote_spent
            )
            
            # ─────────────────────────────────────────────
            # PHASE 3: Update Position (Actual Fill)
            # ─────────────────────────────────────────────
            average_price = actual_quote_spent / executed_qty if executed_qty > 0 else 0
            await self._update_position(
                symbol=symbol,
                quantity=executed_qty,  # Use actual fill!
                price=average_price,
                order_id=order["orderId"],
            )
            
            # ─────────────────────────────────────────────
            # PHASE 4: Post-Fill Handling
            # ─────────────────────────────────────────────
            await self._ensure_post_fill_handled(
                symbol, "BUY", order,
                tier=None, tag=f"buy-{symbol}"
            )
            
            return {"ok": True, "order": order}
        else:
            # ─────────────────────────────────────────────
            # PHASE 2b: Rollback Liquidity (Not Filled)
            # ─────────────────────────────────────────────
            await self.shared_state.rollback_liquidity(
                quote_asset,
                reservation_id,
            )
            
            return {"ok": False, "reason": "order_not_filled", "status": status}
    
    except Exception as e:
        # Rollback on any exception
        if reservation_id:
            with contextlib.suppress(Exception):
                await self.shared_state.rollback_liquidity(quote_asset, reservation_id)
        
        self.logger.error(f"Order placement failed: {e}", exc_info=True)
        raise
```

---

## ✅ Success Criteria

### Phase 1 ✅
- [x] place_market_order() method added to ExchangeClient
- [x] Scope enforcement via _guard_execution_path()
- [x] Signed requests to /api/v3/order
- [x] Summary events emitted
- [x] No syntax errors

### Phase 2 🚧
- [ ] ExecutionManager checks fill status before release
- [ ] Liquidity rollback on non-filled orders
- [ ] Proper logging of release/rollback events
- [ ] Edge cases handled (NEW, PARTIAL, CANCELED)

### Phase 3 🚧
- [ ] Three-step scope pattern implemented
- [ ] Try/finally ensures scope cleanup
- [ ] Scope enforcement prevents orders from outside ExecutionManager
- [ ] All trades reconcile with Binance

### Phase 4 📋
- [ ] Positions updated with executedQty
- [ ] Capital allocation tracks actual spending
- [ ] Risk checks use actual fills
- [ ] Complete audit trail in trade journal

---

## 📚 Documentation Files

✅ `PHASE1_ORDER_PLACEMENT_RESTORATION.md` - Phase 1 complete  
🚧 `PHASE2_3_FILL_RECONCILIATION_INTEGRATION.md` - Phases 2-3 guides  
📋 (To be created) `PHASE4_POSITION_INTEGRITY.md` - Phase 4 guide

---

## 🛠️ Next Steps

### Immediate (Next PR)
1. Review Phase 1 implementation (`place_market_order()`)
2. Verify scope enforcement in action
3. Test with paper trading

### Short Term (Week 1)
1. Implement Phase 2 (fill-aware liquidity release)
2. Add rollback_liquidity() to SharedState
3. Update both _place_market_order_qty() and _place_market_order_quote()

### Medium Term (Week 2)
1. Implement Phase 3 (scope pattern in ExecutionManager)
2. Run integration tests
3. Paper trading with full flow

### Long Term (Week 3)
1. Implement Phase 4 (position integrity)
2. Validate all positions use actual fills
3. Live trading with risk monitoring

---

## 🎓 Key Principles

| Principle | Implementation |
|-----------|-----------------|
| **Orders are scoped** | begin/end_execution_order_scope() |
| **Orders are signed** | signed=True in _request() |
| **Fills are authoritative** | Check order["status"] from Binance |
| **Liquidity is reserved first** | reserve_liquidity() before placement |
| **Liquidity is released after confirmation** | release_liquidity() only if filled |
| **Positions are accurate** | Use executedQty, not planned |
| **Events are complete** | Summary events with all metadata |

---

## ❓ FAQ

**Q: Why not just assume orders are filled?**  
A: Binance returns status=ACCEPTED immediately, but fill happens asynchronously. Order could expire, be canceled, or partially fill. We must check authoritative source (Binance).

**Q: Why scope enforcement?**  
A: Prevents accidental orders from other code paths. Single responsibility principle - only ExecutionManager decides when to place orders.

**Q: Why try/finally for scope?**  
A: Scope MUST be cleaned up even if place_market_order() raises an exception. Otherwise scope remains "stuck" active.

**Q: What if order fails due to insufficient balance?**  
A: Binance returns error, place_market_order() raises BinanceAPIException, ExecutionManager catches it, rollback_liquidity() is called, exception propagates up.

**Q: What if order is placed but Binance times out?**  
A: place_market_order() will have placed the order (newClientOrderId ensures deduplication), retry logic in _request() handles transient network errors, caller must check order status via separate query.

---

## 📞 Support

For questions about this roadmap:
1. Read the phase-specific documentation
2. Check the FAQ
3. Review the code examples
4. Consult the decision tables

---

**Status**: Phase 1 ✅ Complete | Phase 2-3 Ready for Implementation | Phase 4 Planned

