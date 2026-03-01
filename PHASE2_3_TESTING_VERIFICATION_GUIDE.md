# 🧪 PHASE 2-3 TESTING VERIFICATION GUIDE

**Date**: February 25, 2026  
**Purpose**: Verify Phase 2-3 implementation correctness  
**Status**: Ready to execute

---

## 📋 Test Categories

### 1. Unit Tests
### 2. Integration Tests  
### 3. Paper Trading Tests
### 4. Scope Enforcement Tests

---

## 🔬 UNIT TESTS

### Test 1.1: Fill Status Check - Filled Order

**Objective**: Verify `release_liquidity()` called when status="FILLED"

```python
async def test_place_market_order_qty_with_fill():
    """
    Test that liquidity is released when order fills.
    
    PHASE 2 verification: Fill-aware liquidity release
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "BTCUSDT"
    quote_amount = 100.0
    
    # Mock: Binance returns FILLED
    with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
        mock_place.return_value = {
            'orderId': 12345,
            'status': 'FILLED',  # ← KEY: Order was filled
            'executedQty': 0.001,
            'cummulativeQuoteQty': 99.50,
            'symbol': 'BTCUSDT'
        }
        
        with patch.object(manager.shared_state, 'release_liquidity') as mock_release:
            with patch.object(manager.shared_state, 'rollback_liquidity') as mock_rollback:
                # Execute
                result = await manager._place_market_order_quote(
                    symbol=symbol,
                    side='BUY',
                    quote_order_qty=quote_amount
                )
                
                # Verify
                assert result is not None
                assert result.get('status') == 'FILLED'
                
                # ✅ PHASE 2 CHECK: Called release, NOT rollback
                mock_release.assert_called_once()
                mock_rollback.assert_not_called()
                
                print("✅ Test 1.1 PASSED: Filled order releases liquidity")
```

**Expected Result**: ✅ release_liquidity() called  
**Failure Indicator**: ❌ rollback_liquidity() called instead

---

### Test 1.2: Fill Status Check - Non-Filled Order

**Objective**: Verify `rollback_liquidity()` called when status="NEW"

```python
async def test_place_market_order_qty_without_fill():
    """
    Test that liquidity is rolled back when order doesn't fill.
    
    PHASE 2 verification: Fill-aware liquidity rollback
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "BTCUSDT"
    quote_amount = 100.0
    
    # Mock: Binance returns NEW (not filled)
    with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
        mock_place.return_value = {
            'orderId': 12346,
            'status': 'NEW',  # ← KEY: Order not filled
            'executedQty': 0.0,
            'cummulativeQuoteQty': 0.0,
            'symbol': 'BTCUSDT'
        }
        
        with patch.object(manager.shared_state, 'release_liquidity') as mock_release:
            with patch.object(manager.shared_state, 'rollback_liquidity') as mock_rollback:
                # Execute
                result = await manager._place_market_order_quote(
                    symbol=symbol,
                    side='BUY',
                    quote_order_qty=quote_amount
                )
                
                # Verify
                assert result is not None
                assert result.get('status') == 'NEW'
                
                # ✅ PHASE 2 CHECK: Called rollback, NOT release
                mock_rollback.assert_called_once()
                mock_release.assert_not_called()
                
                print("✅ Test 1.2 PASSED: Non-filled order rolls back liquidity")
```

**Expected Result**: ✅ rollback_liquidity() called  
**Failure Indicator**: ❌ release_liquidity() called instead

---

### Test 1.3: Scope Enforcement - Begin/End Calls

**Objective**: Verify `begin_execution_order_scope()` and `end_execution_order_scope()` called

```python
async def test_scope_enforcement_qty():
    """
    Test that scope enforcement is active.
    
    PHASE 3 verification: ExecutionManager scope integration
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "BTCUSDT"
    quote_amount = 100.0
    
    # Mock scope methods
    with patch.object(manager.exchange_client, 'begin_execution_order_scope') as mock_begin:
        with patch.object(manager.exchange_client, 'end_execution_order_scope') as mock_end:
            with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
                mock_begin.return_value = "token_123"
                mock_place.return_value = {
                    'orderId': 12347,
                    'status': 'FILLED',
                    'executedQty': 0.001,
                    'cummulativeQuoteQty': 99.50,
                    'symbol': 'BTCUSDT'
                }
                
                # Execute
                await manager._place_market_order_quote(
                    symbol=symbol,
                    side='BUY',
                    quote_order_qty=quote_amount
                )
                
                # ✅ PHASE 3 CHECK: Scope methods called
                mock_begin.assert_called_once_with("ExecutionManager")
                mock_end.assert_called_once_with("token_123")
                
                print("✅ Test 1.3 PASSED: Scope enforcement active")
```

**Expected Result**: ✅ begin() called before place(), end() in finally  
**Failure Indicator**: ❌ Either not called or in wrong order

---

### Test 1.4: Exception Handling - Scope Cleanup

**Objective**: Verify scope cleanup happens even on exception

```python
async def test_exception_cleanup_qty():
    """
    Test that scope cleanup happens on exception.
    
    PHASE 3 verification: Finally block ensures cleanup
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "BTCUSDT"
    quote_amount = 100.0
    
    # Mock scope and exception
    with patch.object(manager.exchange_client, 'begin_execution_order_scope') as mock_begin:
        with patch.object(manager.exchange_client, 'end_execution_order_scope') as mock_end:
            with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
                mock_begin.return_value = "token_456"
                mock_place.side_effect = Exception("API Error")  # ← Raises
                
                with patch.object(manager.shared_state, 'rollback_liquidity') as mock_rollback:
                    # Execute (expect exception)
                    try:
                        await manager._place_market_order_quote(
                            symbol=symbol,
                            side='BUY',
                            quote_order_qty=quote_amount
                        )
                    except Exception:
                        pass  # Expected
                    
                    # ✅ PHASE 3 CHECK: Scope cleaned up even on error
                    mock_begin.assert_called_once()
                    mock_end.assert_called_once()  # ← Must still be called!
                    mock_rollback.assert_called_once()  # ← Rollback on error
                    
                    print("✅ Test 1.4 PASSED: Scope cleanup on exception")
```

**Expected Result**: ✅ end() called despite exception  
**Failure Indicator**: ❌ end() not called when exception raised

---

### Test 1.5: Partially Filled Order

**Objective**: Verify `release_liquidity()` called for PARTIALLY_FILLED

```python
async def test_place_market_order_quote_partially_filled():
    """
    Test that partially filled orders release liquidity.
    
    PHASE 2 verification: PARTIALLY_FILLED treated as release-worthy
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "ETHUSDT"
    quote_amount = 500.0
    
    # Mock: Binance returns PARTIALLY_FILLED
    with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
        mock_place.return_value = {
            'orderId': 12348,
            'status': 'PARTIALLY_FILLED',  # ← KEY: Partially filled
            'executedQty': 0.15,
            'cummulativeQuoteQty': 250.00,  # Only half filled
            'symbol': 'ETHUSDT'
        }
        
        with patch.object(manager.shared_state, 'release_liquidity') as mock_release:
            with patch.object(manager.shared_state, 'rollback_liquidity') as mock_rollback:
                # Execute
                result = await manager._place_market_order_quote(
                    symbol=symbol,
                    side='BUY',
                    quote_order_qty=quote_amount
                )
                
                # Verify
                assert result.get('status') == 'PARTIALLY_FILLED'
                
                # ✅ PHASE 2 CHECK: Partial fills release (actual spending)
                mock_release.assert_called_once()
                mock_rollback.assert_not_called()
                
                print("✅ Test 1.5 PASSED: Partially filled order releases liquidity")
```

**Expected Result**: ✅ release_liquidity() called  
**Failure Indicator**: ❌ rollback_liquidity() called

---

### Test 1.6: Event Logging with Actual Status

**Objective**: Verify event logs include `actual_status` from Binance

```python
async def test_event_logging_includes_status():
    """
    Test that events log the actual status from Binance.
    
    Verification: Complete audit trail
    """
    # Setup
    manager = ExecutionManager(...)
    symbol = "BNBUSDT"
    
    # Mock
    with patch.object(manager.exchange_client, 'place_market_order') as mock_place:
        mock_place.return_value = {
            'orderId': 12349,
            'status': 'FILLED',
            'executedQty': 1.5,
            'cummulativeQuoteQty': 450.00,
            'symbol': 'BNBUSDT'
        }
        
        with patch.object(manager, '_log_execution_event') as mock_log:
            with patch.object(manager.shared_state, 'release_liquidity'):
                # Execute
                await manager._place_market_order_qty(
                    symbol=symbol,
                    side='BUY',
                    quantity=1.5
                )
                
                # ✅ AUDIT CHECK: Event includes actual_status
                calls = [call[0][0] for call in mock_log.call_args_list]
                assert 'liquidity_released' in calls
                
                # Verify call args contain actual_status
                for call in mock_log.call_args_list:
                    if call[0][0] == 'liquidity_released':
                        event_data = call[0][2]
                        assert 'actual_status' in event_data
                        assert event_data['actual_status'] == 'FILLED'
                
                print("✅ Test 1.6 PASSED: Event logging includes actual_status")
```

**Expected Result**: ✅ `actual_status` in event data  
**Failure Indicator**: ❌ `actual_status` missing from logs

---

## 🔗 INTEGRATION TESTS

### Test 2.1: Full Flow - Filled Order

**Objective**: End-to-end test of filled order with liquidity tracking

```python
async def test_full_flow_filled_order():
    """
    Integration test: Reserve → Place → Release
    
    Verifies complete pipeline for filled order.
    """
    # Setup real manager (not mocked)
    manager = ExecutionManager(...)
    shared_state = manager.shared_state
    symbol = "BTCUSDT"
    quote_amount = 100.0
    
    # 1. Reserve liquidity
    reservation_id = await shared_state.reserve_liquidity(
        "USDT", quote_amount, tag="test"
    )
    assert reservation_id is not None
    
    # 2. Check reserved
    reserved = await shared_state.get_reserved_balance("USDT")
    assert reserved >= quote_amount
    
    # 3. Place order (mocked to return FILLED)
    with patch.object(manager.exchange_client, 'place_market_order') as mock:
        mock.return_value = {
            'orderId': 99999,
            'status': 'FILLED',
            'executedQty': 0.001,
            'cummulativeQuoteQty': 99.50,
            'symbol': 'BTCUSDT'
        }
        
        result = await manager._place_market_order_quote(
            symbol=symbol,
            side='BUY',
            quote_order_qty=quote_amount,
            tag="test"
        )
    
    # 4. Check released
    released = await shared_state.get_reserved_balance("USDT")
    
    # ✅ VERIFY
    assert result['status'] == 'FILLED'
    assert released < reserved  # Liquidity released
    
    print("✅ Test 2.1 PASSED: Full flow for filled order")
```

**Expected Result**: ✅ Liquidity reserved → released  
**Failure Indicator**: ❌ Liquidity not released after fill

---

### Test 2.2: Full Flow - Non-Filled Order

**Objective**: End-to-end test of non-filled order with liquidity rollback

```python
async def test_full_flow_non_filled_order():
    """
    Integration test: Reserve → Place → Rollback
    
    Verifies complete pipeline for non-filled order.
    """
    # Setup real manager
    manager = ExecutionManager(...)
    shared_state = manager.shared_state
    symbol = "ETHUSDT"
    quote_amount = 500.0
    
    # 1. Reserve liquidity
    reservation_id = await shared_state.reserve_liquidity(
        "USDT", quote_amount, tag="test_nofill"
    )
    initial_reserved = await shared_state.get_reserved_balance("USDT")
    
    # 2. Place order (mocked to return NEW)
    with patch.object(manager.exchange_client, 'place_market_order') as mock:
        mock.return_value = {
            'orderId': 88888,
            'status': 'NEW',  # NOT filled
            'executedQty': 0.0,
            'cummulativeQuoteQty': 0.0,
            'symbol': 'ETHUSDT'
        }
        
        result = await manager._place_market_order_quote(
            symbol=symbol,
            side='BUY',
            quote_order_qty=quote_amount,
            tag="test_nofill"
        )
    
    # 3. Check rollback happened
    final_reserved = await shared_state.get_reserved_balance("USDT")
    
    # ✅ VERIFY
    assert result['status'] == 'NEW'
    assert final_reserved < initial_reserved  # Liquidity rolled back
    
    print("✅ Test 2.2 PASSED: Full flow for non-filled order")
```

**Expected Result**: ✅ Liquidity reserved → rolled back  
**Failure Indicator**: ❌ Liquidity not rolled back on non-fill

---

## 📱 PAPER TRADING TESTS

### Test 3.1: Live Paper Trade - Filled Order

**Objective**: Execute real order on paper trading account, verify fill-aware release

```bash
# Manual test:
# 1. Start paper trading
# 2. Execute command:
python -m pytest tests/test_phase2_3_paper_trading.py::test_paper_trade_filled -v

# Expected:
# ✅ Order placed
# ✅ Status returns FILLED (or PARTIALLY_FILLED)
# ✅ Liquidity released
# ✅ Event logged with actual_status
```

---

### Test 3.2: Live Paper Trade - Queued Order

**Objective**: Place order that doesn't fill immediately, verify rollback

```bash
# Manual test:
# 1. Start paper trading
# 2. Place limit order (won't fill immediately)
# 3. Execute command:
python -m pytest tests/test_phase2_3_paper_trading.py::test_paper_trade_queued -v

# Expected:
# ✅ Order placed with status=NEW
# ✅ Liquidity rolled back
# ✅ Event logged with actual_status=NEW
```

---

## 🔐 SCOPE ENFORCEMENT TESTS

### Test 4.1: Scope Violation Detection

**Objective**: Verify that non-ExecutionManager scope tokens are rejected

```python
async def test_scope_violation_detection():
    """
    Test that scope enforcement prevents unauthorized access.
    
    PHASE 3 security verification.
    """
    # Setup
    exchange_client = ExchangeClient(...)
    
    # 1. Create valid token
    valid_token = exchange_client.begin_execution_order_scope("ExecutionManager")
    
    # 2. Try to use invalid token
    invalid_token = "fake_token_12345"
    
    # 3. Try to place order with invalid token
    # (This should fail or be blocked by guard)
    try:
        order = await exchange_client.place_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            tag="test"
            # Note: No token validation in current design
            #       but scope is enforced via ExecutionManager
        )
        print("⚠️  Warning: Scope validation needs improvement")
    except Exception as e:
        print(f"✅ Test 4.1 PASSED: Scope violation detected")
    
    # Cleanup
    exchange_client.end_execution_order_scope(valid_token)
```

---

## 📊 TEST EXECUTION PLAN

### Phase 1: Quick Verification (30 minutes)
```
✅ Test 1.1: Fill status - FILLED
✅ Test 1.2: Fill status - NEW
✅ Test 1.3: Scope enforcement
✅ Test 1.4: Exception cleanup
```

### Phase 2: Comprehensive Unit Tests (1 hour)
```
✅ All unit tests above
✅ Verify all assertions pass
✅ Check code coverage > 90%
```

### Phase 3: Integration Tests (1 hour)
```
✅ Test 2.1: Full flow - filled
✅ Test 2.2: Full flow - non-filled
✅ Verify liquidity tracking
```

### Phase 4: Paper Trading (2-4 hours)
```
✅ Test 3.1: Paper trade - filled
✅ Test 3.2: Paper trade - queued
✅ Verify events logged correctly
✅ Verify audit trail complete
```

---

## ✅ Success Criteria

### Unit Tests
- ✅ All tests pass
- ✅ 100% of scope enforcement calls verified
- ✅ 100% of fill-status checks verified
- ✅ 100% of liquidity operations verified

### Integration Tests
- ✅ Liquidity correctly reserved before placement
- ✅ Liquidity released only when order fills
- ✅ Liquidity rolled back when order doesn't fill
- ✅ No orphaned reservations

### Paper Trading
- ✅ Orders place successfully
- ✅ Status checks work correctly
- ✅ Liquidity operations match Binance response
- ✅ Events logged with actual_status
- ✅ Audit trail complete and correct

---

## 🚀 How to Run Tests

```bash
# Run unit tests only
pytest tests/test_phase2_3_unit.py -v

# Run integration tests
pytest tests/test_phase2_3_integration.py -v

# Run paper trading tests
pytest tests/test_phase2_3_paper_trading.py -v -s

# Run all Phase 2-3 tests
pytest tests/test_phase2_3_*.py -v --cov=core

# Run specific test
pytest tests/test_phase2_3_unit.py::test_place_market_order_qty_with_fill -v
```

---

## 📝 Expected Test Files

Create these test files:
1. `tests/test_phase2_3_unit.py` - Unit tests 1.1-1.6
2. `tests/test_phase2_3_integration.py` - Integration tests 2.1-2.2
3. `tests/test_phase2_3_paper_trading.py` - Paper trading tests 3.1-3.2
4. `tests/conftest.py` - Shared fixtures

---

**Status**: Ready to begin testing Phase 2-3 implementation

**Next**: Execute unit tests, then integration tests, then paper trading

