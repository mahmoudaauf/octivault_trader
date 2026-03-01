# Testing Plan: SELL Post-Fill Fix Validation

**Status:** Ready for Phase 13  
**Date:** February 23, 2026  

---

## 🧪 Unit Tests

### Test 1: `_reconcile_delayed_fill()` Returns Order Without Post-Fill Flags

```python
async def test_reconcile_returns_order_without_post_fill_flags():
    """Verify reconcile no longer sets _post_fill_done or _post_fill_result."""
    
    # Arrange
    order = {
        "symbol": "BTCUSDT",
        "orderId": "12345",
        "status": "FILLED",
        "executedQty": 1.0,
        "price": 50000.0,
    }
    
    # Act
    result = await em._reconcile_delayed_fill(
        symbol="BTCUSDT",
        side="SELL",
        order=order,
        tag="test",
    )
    
    # Assert
    assert result is not None
    assert result.get("status") == "FILLED"  # Data merged
    assert result.get("executedQty") == 1.0
    assert "_post_fill_done" not in result or not result.get("_post_fill_done")
    assert "_post_fill_result" not in result
```

### Test 2: `close_position()` Calls `_ensure_post_fill_handled()` Exactly Once

```python
async def test_close_position_calls_ensure_post_fill_once():
    """Verify close_position doesn't call post-fill twice."""
    
    # Mock to count calls
    call_count = 0
    original_ensure = em._ensure_post_fill_handled
    
    async def mock_ensure(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await original_ensure(*args, **kwargs)
    
    em._ensure_post_fill_handled = mock_ensure
    
    try:
        # Act
        await em.close_position(symbol="BTCUSDT", reason="TEST")
        
        # Assert
        assert call_count == 1, f"Expected 1 call, got {call_count}"
    finally:
        em._ensure_post_fill_handled = original_ensure
```

### Test 3: `_finalize_sell_post_fill()` Idempotency

```python
async def test_finalize_idempotency_with_pre_set_flags():
    """Verify finalize skips post-fill when _post_fill_done is set."""
    
    # Arrange
    order = {
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "executedQty": 1.0,
        "price": 50000.0,
        "_post_fill_done": True,  # Already processed
        "_post_fill_result": {
            "delta": 100.0,
            "realized_committed": True,
            "emitted": True,
        }
    }
    
    mock_ensure_count = 0
    original_ensure = em._ensure_post_fill_handled
    
    async def mock_ensure(*args, **kwargs):
        nonlocal mock_ensure_count
        mock_ensure_count += 1
        return {}
    
    em._ensure_post_fill_handled = mock_ensure
    
    try:
        # Act
        await em._finalize_sell_post_fill(
            symbol="BTCUSDT",
            order=order,
            post_fill=order.get("_post_fill_result"),
        )
        
        # Assert
        assert mock_ensure_count == 0, f"Expected 0 calls (idempotent), got {mock_ensure_count}"
    finally:
        em._ensure_post_fill_handled = original_ensure
```

---

## 🔗 Integration Tests

### Test 4: SELL Order Flow → POSITION_CLOSED Event

```python
async def test_sell_flow_emits_position_closed():
    """Verify entire SELL flow emits POSITION_CLOSED and updates position."""
    
    # Arrange
    symbol = "ETHUSDT"
    initial_qty = 2.5
    
    # Setup position in SharedState
    ss.positions[symbol] = {
        "quantity": initial_qty,
        "entry_price": 2000.0,
        "status": "OPEN",
    }
    
    # Mock exchange to return filled SELL
    async def mock_place_order(*args, **kwargs):
        return {
            "symbol": symbol,
            "orderId": "999",
            "status": "FILLED",
            "executedQty": initial_qty,
            "avgPrice": 2100.0,  # Profit
            "cummulativeQuoteQty": 5250.0,
        }
    
    em.exchange_client.place_market_order = mock_place_order
    
    # Track events
    events_emitted = []
    original_emit = ss.emit_event
    
    async def mock_emit(event_name, data):
        events_emitted.append((event_name, data))
        return await original_emit(event_name, data)
    
    ss.emit_event = mock_emit
    
    try:
        # Act
        result = await em.close_position(
            symbol=symbol,
            reason="TEST_CLOSE",
        )
        
        # Assert
        assert result.get("ok") or result.get("status") == "FILLED"
        assert ("POSITION_CLOSED", ...) in events_emitted or \
               any(e[0] == "POSITION_CLOSED" for e in events_emitted)
        assert ss.positions[symbol]["quantity"] == 0.0
        assert ss.positions[symbol]["status"] == "CLOSED"
    finally:
        ss.emit_event = original_emit
```

### Test 5: Liquidation Path → Position Reduced

```python
async def test_liquidation_reduces_position():
    """Verify liquidation path correctly marks position as closed."""
    
    # Arrange
    symbol = "BNBUSDT"
    ss.positions[symbol] = {"quantity": 3.0, "status": "OPEN"}
    
    async def mock_place_order(*args, **kwargs):
        return {
            "symbol": symbol,
            "orderId": "888",
            "status": "FILLED",
            "executedQty": 3.0,
            "price": 600.0,
        }
    
    em.exchange_client.place_market_order = mock_place_order
    em._liquidate_symbols([symbol], reason="LIQUIDATION")
    
    # Assert
    assert ss.positions[symbol]["quantity"] == 0.0
```

### Test 6: Delayed Fill Reconciliation → Finalization Works

```python
async def test_delayed_fill_reconciliation_then_finalize():
    """Verify delayed fill path still finalizes correctly."""
    
    # Arrange: Order initially returns NOT_FILLED
    order = {
        "symbol": "ADAUSDT",
        "orderId": "777",
        "status": "PARTIALLY_FILLED",
        "executedQty": 0.0,  # Not filled yet
    }
    
    # Mock get_order to return filled on retry
    async def mock_get_order(symbol, order_id=None, client_order_id=None):
        return {
            "symbol": symbol,
            "orderId": order_id,
            "status": "FILLED",
            "executedQty": 5.0,  # Now filled
            "price": 1.5,
        }
    
    em.exchange_client.get_order = mock_get_order
    
    # Act
    merged = await em._reconcile_delayed_fill(
        symbol="ADAUSDT",
        side="SELL",
        order=order,
    )
    
    # Assert: reconcile returns merged data
    assert merged["executedQty"] == 5.0
    assert merged.get("status") == "FILLED"
    # But NO post-fill flags set yet
    assert "_post_fill_done" not in merged or not merged.get("_post_fill_done")
    
    # Now caller must handle post-fill
    post_fill = await em._ensure_post_fill_handled(
        symbol="ADAUSDT",
        side="SELL",
        order=merged,
    )
    
    # Assert
    assert post_fill.get("emitted") or post_fill.get("trade_event_emitted")
    assert merged.get("_post_fill_done")
```

---

## 🎯 System Tests

### Test 7: Backtest Run → Positions Close Correctly

```bash
python backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-31
```

**Verify:**
- [ ] Every SELL order that fills has position.quantity = 0
- [ ] POSITION_CLOSED event logged for each close
- [ ] Realized PnL correctly recorded
- [ ] No "position_closed_but_not_reduced" warnings in logs

### Test 8: Integration Test → Full Trade Cycle

```python
async def test_full_trade_cycle():
    """BUY → target move achieved → SELL → position reduced."""
    
    symbol = "XRPUSDT"
    
    # 1. BUY
    buy_result = await em.execute_trade(
        symbol=symbol,
        side="BUY",
        quantity=10.0,
        tag="test_buy",
    )
    assert buy_result.get("executedQty") > 0
    assert ss.positions[symbol]["quantity"] > 0
    
    # 2. SELL (via TP or SL)
    sell_result = await em.close_position(
        symbol=symbol,
        reason="TP_HIT",
        tag="tp_sl",
    )
    assert sell_result.get("executedQty") > 0
    
    # 3. Verify position closed
    assert ss.positions[symbol]["quantity"] == 0.0
    assert ss.positions[symbol]["status"] == "CLOSED"
```

---

## 📊 Live Validation

### Test 9: Dry Run → Watch Position Close Logs

```bash
python dry_run_test.py --duration 60 --watch close_position
```

**Watch for:**
```
[EM:CLOSE_ATTEMPT] symbol=BTCUSDT qty=1.5 reason=TP_HIT
[EM:CLOSE_RESULT] symbol=BTCUSDT ok=True status=FILLED executed_qty=1.5
[EM:DelayedFill] Reconciled delayed fill symbol=BTCUSDT ... qty=1.5
[POSITION_CLOSED] symbol=BTCUSDT entry_price=50000 exit_price=51000 qty=1.5 realized_pnl=1500
```

**Verify:**
- [ ] Each CLOSE_ATTEMPT has matching CLOSE_RESULT
- [ ] Reconcile merges order without "post_fill" in log
- [ ] POSITION_CLOSED always emitted when filled
- [ ] No double POSITION_CLOSED events

### Test 10: Live Small Trade → Check SharedState

```python
# Execute small closing trade, then immediately check:
pos = await ss.get_position("BTCUSDT")
assert pos["quantity"] == 0.0
assert pos["closed_at"] is not None
print(f"✅ Position closed at {pos['closed_at']}")
```

---

## ✅ Checklist

**Before Testing:**
- [ ] Syntax verified (no errors)
- [ ] All 4 code locations updated
- [ ] Documentation written

**Unit Testing:**
- [ ] Test 1: Reconcile doesn't set flags
- [ ] Test 2: Close calls post-fill once
- [ ] Test 3: Finalize idempotency works
- [ ] Test 4: Full flow emits POSITION_CLOSED
- [ ] Test 5: Liquidation reduces position
- [ ] Test 6: Delayed fill finalizes

**Integration Testing:**
- [ ] Test 7: Backtest completes cleanly
- [ ] Test 8: Full trade cycle works
- [ ] Test 9: Dry run logs show correct flow
- [ ] Test 10: SharedState updates verified

**Success Criteria:**
- ✅ SELL orders close positions
- ✅ POSITION_CLOSED emitted every time
- ✅ SharedState.positions[symbol].quantity = 0
- ✅ No duplicate events or double accounting
- ✅ No "position_closed_but_not_reduced" errors

---

## 📝 Next Steps

1. **Run Unit Tests** → Verify method-level behavior
2. **Run Integration Tests** → Verify flow integrity
3. **Run Backtest** → Verify system behavior
4. **Run Dry Run** → Verify log output matches expectations
5. **Deploy to Test Environment** → Monitor live execution
6. **Verify SharedState Updates** → Check position tracking

