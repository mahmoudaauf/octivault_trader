# TP/SL SELL Path - Before & After Comparison

**File:** `core/execution_manager.py`  
**Lines:** 5689-5760  
**Change Type:** Remove duplicate finalization fallback  

---

## BEFORE (BROKEN)

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

                # Finalize position on SELL fills  ← ❌ PROBLEM: Duplicate finalization
                if side == "sell":
                    try:
                        pm = getattr(self.shared_state, "position_manager", None)
                        exec_qty = float(raw.get("executedQty", 0.0))
                        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
                        fee_quote = float(raw.get("fee_quote", 0.0) or raw.get("fee", 0.0) or 0.0)
                        try:
                            _, quote_asset = self._split_base_quote(sym)
                            fills = raw.get("fills") or []
                            if isinstance(fills, list):
                                fee_quote = sum(
                                    float(f.get("commission", 0.0) or 0.0)
                                    for f in fills
                                    if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == quote_asset
                                ) or fee_quote
                        except Exception:
                            pass
                        if pm and hasattr(pm, "close_position"):
                            await pm.close_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                fee_quote=fee_quote,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif pm and hasattr(pm, "finalize_position"):
                            await pm.finalize_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif hasattr(self.shared_state, "close_position"):
                            await self.shared_state.close_position(
                                sym,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        # 🔥 MANDATORY: Journal position closure BEFORE mark_position_closed
                        self._journal("POSITION_CLOSURE_VIA_MARK", {
                            "symbol": sym,
                            "executed_qty": exec_qty,
                            "executed_price": exec_px,
                            "reason": str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            "tag": str(tag_raw or ""),
                            "timestamp": time.time(),
                        })
                        if hasattr(self.shared_state, "mark_position_closed"):
                            await self.shared_state.mark_position_closed(
                                symbol=sym,
                                qty=exec_qty,
                                price=exec_px,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                tag=str(tag_raw or ""),
                            )
                    except Exception:
                        self.logger.debug("[EM] finalize_position failed for %s", sym, exc_info=True)

                try:
                    await self._audit_post_fill_accounting(
                        symbol=sym,
                        ...
```

### Problem Analysis

1. **Line 5689-5697:** `_finalize_sell_post_fill()` called ✅ CANONICAL
   - Emits POSITION_CLOSED event
   - Emits RealizedPnlUpdated event
   - Full EM accounting

2. **Line 5700-5750:** **DUPLICATE** finalization block ❌
   - Calls `pm.close_position()` - bypasses EM
   - Calls `pm.finalize_position()` - bypasses EM
   - Calls `shared_state.close_position()` - direct bypass
   - Calls `mark_position_closed()` - direct bypass
   - These are NOT going through ExecutionManager event paths
   - These duplicate the canonical finalization
   - These break the 100% canonical guarantee

3. **Result:** Non-canonical execution path
   - Fallback may override canonical finalization
   - Events may be skipped
   - Governance visibility broken
   - P9 contract violated

---

## AFTER (FIXED)

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

                try:
                    await self._audit_post_fill_accounting(
                        symbol=sym,
                        ...
```

### Solution Analysis

1. **Line 5689-5697:** `_finalize_sell_post_fill()` called ✅ CANONICAL
   - Emits POSITION_CLOSED event
   - Emits RealizedPnlUpdated event
   - Full EM accounting
   - **This is the ONLY path**

2. **Lines 5700-5750:** **DELETED** ❌→✅
   - No duplicate finalization
   - No fallback bypass
   - No SharedState direct calls
   - No governance visibility gap

3. **Result:** 100% canonical execution
   - Single finalization path only
   - All events through EM guaranteed
   - Complete governance visibility
   - P9 contract maintained

---

## Key Changes

| Aspect | BEFORE | AFTER |
|--------|--------|-------|
| Finalization paths | 2 (canonical + fallback) | 1 (canonical only) |
| Event emission | Conditional (canonical or fallback) | Always canonical |
| EM visibility | Partial (fallback skipped) | Complete (all events) |
| Governance audit | Incomplete (fallback not tracked) | Complete (all tracked) |
| Dust fix compatibility | Potential issues | Fully compatible |
| P9 compliance | Violated (non-100% canonical) | Maintained (100% canonical) |

---

## Exact Deletion

**Delete the following section from `core/execution_manager.py`:**

**Start:** Line 5700 (comment: `# Finalize position on SELL fills`)  
**End:** Line 5750 (closing brace of try/except block)

**Total lines to delete:** 51 lines

```python
# DELETE LINES 5700-5750:

                # Finalize position on SELL fills
                if side == "sell":
                    try:
                        pm = getattr(self.shared_state, "position_manager", None)
                        exec_qty = float(raw.get("executedQty", 0.0))
                        exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
                        fee_quote = float(raw.get("fee_quote", 0.0) or raw.get("fee", 0.0) or 0.0)
                        try:
                            _, quote_asset = self._split_base_quote(sym)
                            fills = raw.get("fills") or []
                            if isinstance(fills, list):
                                fee_quote = sum(
                                    float(f.get("commission", 0.0) or 0.0)
                                    for f in fills
                                    if str(f.get("commissionAsset") or f.get("commission_asset") or "").upper() == quote_asset
                                ) or fee_quote
                        except Exception:
                            pass
                        if pm and hasattr(pm, "close_position"):
                            await pm.close_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                fee_quote=fee_quote,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif pm and hasattr(pm, "finalize_position"):
                            await pm.finalize_position(
                                symbol=sym,
                                executed_qty=exec_qty,
                                executed_price=exec_px,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        elif hasattr(self.shared_state, "close_position"):
                            await self.shared_state.close_position(
                                sym,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            )
                        # 🔥 MANDATORY: Journal position closure BEFORE mark_position_closed
                        self._journal("POSITION_CLOSURE_VIA_MARK", {
                            "symbol": sym,
                            "executed_qty": exec_qty,
                            "executed_price": exec_px,
                            "reason": str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                            "tag": str(tag_raw or ""),
                            "timestamp": time.time(),
                        })
                        if hasattr(self.shared_state, "mark_position_closed"):
                            await self.shared_state.mark_position_closed(
                                symbol=sym,
                                qty=exec_qty,
                                price=exec_px,
                                reason=str(policy_ctx.get("exit_reason") or policy_ctx.get("reason") or "SELL_FILLED"),
                                tag=str(tag_raw or ""),
                            )
                    except Exception:
                        self.logger.debug("[EM] finalize_position failed for %s", sym, exc_info=True)
```

---

## Verification

After deletion, verify:

1. **Line 5689-5697** should still be there (canonical finalization)
2. **Comment "# Finalize position on SELL fills"** should be gone
3. **Next line after deletion** should be `try:` followed by `await self._audit_post_fill_accounting(`
4. **No syntax errors** in the file

---

## Testing Plan

After applying the fix:

```python
# Test 1: TP/SL SELL execution
async def test_tp_sl_sell_execution():
    # Trigger take-profit order
    order = await em.execute_trade("BTC/USDT", "sell", qty=1.0, tag="tp_sl")
    
    # Verify:
    assert order["status"] == "filled"
    
    # Verify POSITION_CLOSED event was emitted (only once)
    events = await em.get_events("POSITION_CLOSED", symbol="BTC/USDT")
    assert len(events) == 1  # Exactly one
    assert events[0]["symbol"] == "BTC/USDT"

# Test 2: Regular SELL with TP/SL tag
async def test_regular_sell_with_tp_sl_tag():
    order = await em.close_position("BTC/USDT", reason="manual", tag="tp_sl")
    
    # Verify:
    assert order["side"] == "sell"
    
    # Verify events emitted through canonical path
    events = await em.get_events("RealizedPnlUpdated", symbol="BTC/USDT")
    assert len(events) >= 1
    assert all(e["source"] == "ExecutionManager" for e in events)

# Test 3: Dust position SELL (tests our earlier fix too)
async def test_dust_sell_with_tp_sl():
    # Create dust position (0.001 BTC)
    pos = await em.open_position("BTC/USDT", qty=0.001)
    
    # Trigger SELL with TP/SL tag
    order = await em.close_position("BTC/USDT", reason="dust", tag="tp_sl")
    
    # Verify POSITION_CLOSED event emitted (even for dust)
    events = await em.get_events("POSITION_CLOSED", symbol="BTC/USDT")
    assert any(e["executed_qty"] == 0.001 for e in events)

# Test 4: Governance sees complete event chain
async def test_governance_event_chain():
    order = await em.close_position("BTC/USDT", reason="tp_sl_exit", tag="tp_sl")
    
    # Get audit trail
    events = await auditor.get_event_trail("BTC/USDT")
    
    # Verify complete chain:
    # - SELL order placed
    # - SELL order filled
    # - TRADE_EXECUTED emitted
    # - POSITION_CLOSED emitted
    # - RealizedPnlUpdated emitted
    # - All from ExecutionManager (canonical)
    
    assert all(e["source"] == "ExecutionManager" for e in events[-4:])
```

---

## Summary

| Aspect | Value |
|--------|-------|
| File | `core/execution_manager.py` |
| Lines to delete | 5700-5750 (51 lines) |
| Complexity | Simple deletion |
| Risk | Very Low |
| Impact | High (fixes canonicality issue) |
| Testing | TP/SL SELL execution |
| Backward compatibility | Full (removes fallback only) |

---

**Ready to implement. This fix is low-risk and high-impact.**
