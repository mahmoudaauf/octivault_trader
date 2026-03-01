# CompoundingEngine Protective Gates - Test Specification

## Overview

Test specification for validating the three protective gates implementation in `CompoundingEngine`. These tests ensure:
1. Each gate correctly filters symbols
2. Gates integrate properly with execution flow  
3. Logging/monitoring works correctly
4. Backward compatibility maintained

---

## Unit Tests

### Test Suite 1: Volatility Gate (`_validate_volatility_gate`)

#### Test 1.1: Volatility Above Threshold (PASS)
```python
async def test_volatility_gate_pass_high_volatility():
    """Symbol with 0.52% volatility should PASS (> 0.45% threshold)"""
    engine = CompoundingEngine(...)
    
    # Mock: Symbol with high volatility
    mock_ohlcv = [
        (0, 100.0, 102.0, 98.0, 101.0, 1000),  # High variation
        (0, 101.0, 103.5, 99.5, 102.0, 1000),
        # ... 23 more candles with std ~0.52%
    ]
    
    shared_state.get_symbol_ohlcv = AsyncMock(return_value=mock_ohlcv)
    
    result = await engine._validate_volatility_gate("ETHUSDT")
    assert result == True  # Should PASS
```

#### Test 1.2: Volatility Below Threshold (FAIL)
```python
async def test_volatility_gate_fail_low_volatility():
    """Symbol with 0.15% volatility should FAIL (< 0.45% threshold)"""
    engine = CompoundingEngine(...)
    
    # Mock: Calm symbol
    mock_ohlcv = [
        (0, 100.0, 100.05, 99.95, 100.02, 1000),  # Minimal variation
        (0, 100.02, 100.07, 99.97, 100.04, 1000),
        # ... 23 more calm candles
    ]
    
    shared_state.get_symbol_ohlcv = AsyncMock(return_value=mock_ohlcv)
    
    result = await engine._validate_volatility_gate("STABLECOIN")
    assert result == False  # Should FAIL (too calm)
```

#### Test 1.3: Volatility At Boundary
```python
async def test_volatility_gate_at_boundary():
    """Symbol at exactly 0.45% should PASS (>=, not >)"""
    engine = CompoundingEngine(...)
    
    # Create OHLCV with volatility = exactly 0.0045
    # (Would need proper calculation, roughly 0.45% std)
    
    result = await engine._validate_volatility_gate("BOUNDARY")
    assert result == True  # Should PASS at boundary
```

#### Test 1.4: Missing Data Handling
```python
async def test_volatility_gate_no_ohlcv():
    """Missing OHLCV should conservatively SKIP (return False)"""
    engine = CompoundingEngine(...)
    
    shared_state.get_symbol_ohlcv = AsyncMock(return_value=None)
    
    result = await engine._validate_volatility_gate("UNKNOWN")
    assert result == False  # Conservative: skip on missing data
```

---

### Test Suite 2: Edge Validation Gate (`_validate_edge_gate`)

#### Test 2.1: Not at Local High (PASS)
```python
async def test_edge_gate_pass_away_from_high():
    """Symbol 0.5% from high should PASS (distance > 0.1%)"""
    engine = CompoundingEngine(...)
    
    # Build OHLCV: high=100.0, current=99.5
    mock_ohlcv = [
        # ... 20 candles with highs up to 100.0
        (0, 99.0, 100.0, 98.0, 99.5, 1000),  # At high
        (0, 99.5, 99.8, 99.2, 99.4, 1000),   # Moving down
        (0, 99.4, 99.6, 99.0, 99.5, 1000),   # Still down (current)
    ]
    
    result = await engine._validate_edge_gate("ETHUSDT")
    assert result == True  # Should PASS (good distance from high)
```

#### Test 2.2: At Local High (FAIL)
```python
async def test_edge_gate_fail_at_local_high():
    """Symbol within 0.1% of 20-candle high should FAIL"""
    engine = CompoundingEngine(...)
    
    # Build OHLCV: high=100.0, current=99.95 (0.05% away)
    mock_ohlcv = [
        # ... 20 candles with highs up to 100.0
        (0, 99.9, 100.0, 99.8, 99.95, 1000),  # Current very close to high
    ]
    
    result = await engine._validate_edge_gate("ETHUSDT")
    assert result == False  # Should FAIL (at local top)
```

#### Test 2.3: Post-Momentum (FAIL)
```python
async def test_edge_gate_fail_post_momentum():
    """Symbol with >0.5% recent rally should FAIL (momentum fired)"""
    engine = CompoundingEngine(...)
    
    # Build OHLCV: 0.8% move in last 5 candles
    mock_ohlcv = [
        # ... candles up to 5 ago
        (0, 99.0, 99.1, 98.9, 99.0, 1000),   # 5 candles ago
        (0, 99.0, 99.3, 98.9, 99.2, 1000),
        (0, 99.2, 99.5, 99.0, 99.4, 1000),
        (0, 99.4, 99.7, 99.2, 99.6, 1000),
        (0, 99.6, 99.9, 99.4, 99.8, 1000),   # Current
    ]
    # Move = 99.8 / 99.0 - 1 = +0.81% (exceeds 0.5%)
    
    result = await engine._validate_edge_gate("ETHUSDT")
    assert result == False  # Should FAIL (momentum fired)
```

#### Test 2.4: Clear Momentum (PASS)
```python
async def test_edge_gate_pass_clear_momentum():
    """Symbol with <0.5% recent move should PASS"""
    engine = CompoundingEngine(...)
    
    # Build OHLCV: 0.2% move in last 5 candles
    mock_ohlcv = [
        # ... candles up to 5 ago
        (0, 100.0, 100.1, 99.9, 100.0, 1000),   # 5 candles ago
        (0, 100.0, 100.1, 99.9, 100.0, 1000),
        (0, 100.0, 100.1, 99.9, 100.0, 1000),
        (0, 100.0, 100.1, 99.9, 100.0, 1000),
        (0, 100.0, 100.1, 99.9, 100.2, 1000),   # Current (+0.2%)
    ]
    
    result = await engine._validate_edge_gate("ETHUSDT")
    assert result == True  # Should PASS (momentum clear)
```

---

### Test Suite 3: Economic Threshold Gate (`_validate_economic_gate`)

#### Test 3.1: Profit Sufficient (PASS)
```python
async def test_economic_gate_pass_sufficient_profit():
    """With profit > (fees + buffer), should PASS"""
    engine = CompoundingEngine(...)
    shared_state.metrics = {"realized_pnl": 100.0}
    
    # $100 profit, 5 buys @ $10 each
    # Fees: 5 × $10 × 0.225% = $0.11
    # Buffer: $50
    # Available: $100 - $0.11 - $50 = $49.89 > 0 ✅
    
    result = await engine._validate_economic_gate(50.0, 5)
    assert result == True  # Should PASS (sufficient profit)
```

#### Test 3.2: Profit Insufficient (FAIL)
```python
async def test_economic_gate_fail_insufficient_profit():
    """With profit < (fees + buffer), should FAIL"""
    engine = CompoundingEngine(...)
    shared_state.metrics = {"realized_pnl": 20.0}
    
    # $20 profit, 5 buys @ $10 each
    # Fees: $0.11
    # Buffer: $50
    # Available: $20 - $0.11 - $50 = -$30.11 < 0 ❌
    
    result = await engine._validate_economic_gate(50.0, 5)
    assert result == False  # Should FAIL (insufficient profit)
```

#### Test 3.3: At Break-Even (PASS)
```python
async def test_economic_gate_at_breakeven():
    """Exactly at break-even should PASS (profit > 0)"""
    engine = CompoundingEngine(...)
    
    # $50.11 profit, 5 buys @ $10 each
    # Fees: $0.11
    # Buffer: $50
    # Available: $50.11 - $0.11 - $50 = $0.00 >= 0 ✅
    
    shared_state.metrics = {"realized_pnl": 50.11}
    
    result = await engine._validate_economic_gate(50.0, 5)
    assert result == True  # PASS at break-even
```

#### Test 3.4: Just Below Break-Even (FAIL)
```python
async def test_economic_gate_just_below_breakeven():
    """Slightly below break-even should FAIL"""
    engine = CompoundingEngine(...)
    
    # $50.10 profit (just under)
    # Fees: $0.11
    # Buffer: $50
    # Available: $50.10 - $0.11 - $50 = -$0.01 < 0 ❌
    
    shared_state.metrics = {"realized_pnl": 50.10}
    
    result = await engine._validate_economic_gate(50.0, 5)
    assert result == False  # Should FAIL (insufficient by $0.01)
```

---

### Test Suite 4: Integration (_pick_symbols with Gates 1 & 2)

#### Test 4.1: All Gates Pass
```python
async def test_pick_symbols_all_gates_pass():
    """All symbols pass both gates -> return top symbols"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - 5 candidate symbols (ETHUSDT, LTCUSDT, ADAUSDT, DOGEUSDT, XRPUSDT)
    # - Mock volatility: all > 0.45%
    # - Mock edge: all have good entry points
    
    shared_state.get_accepted_symbols_snapshot.return_value = {
        'ETHUSDT': {}, 'LTCUSDT': {}, 'ADAUSDT': {}, 'DOGEUSDT': {}, 'XRPUSDT': {}
    }
    shared_state.get_symbol_scores.return_value = {
        'ETHUSDT': 50, 'LTCUSDT': 45, 'ADAUSDT': 40, 'DOGEUSDT': 35, 'XRPUSDT': 30
    }
    
    # All pass volatility and edge gates
    symbols = await engine._pick_symbols()
    
    assert len(symbols) == 5  # All returned (or up to max_symbols)
    assert symbols[0] == 'ETHUSDT'  # Highest score first
```

#### Test 4.2: Some Symbols Filtered by Gate 1
```python
async def test_pick_symbols_gate1_filters():
    """Volatile symbols pass, calm ones filtered out"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - ETHUSDT: 0.52% volatility -> PASS
    # - LTCUSDT: 0.15% volatility -> FAIL (too calm)
    # - ADAUSDT: 0.48% volatility -> PASS
    
    # Mock volatility returns accordingly
    # (would set up mock_ohlcv or market_data)
    
    symbols = await engine._pick_symbols()
    
    assert 'ETHUSDT' in symbols  # Volatile, passed
    assert 'LTCUSDT' not in symbols  # Calm, filtered out
    assert 'ADAUSDT' in symbols  # Volatile, passed
```

#### Test 4.3: Some Symbols Filtered by Gate 2
```python
async def test_pick_symbols_gate2_filters():
    """Good entries pass, poor ones filtered out"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - ETHUSDT: 0.25% from high, clear momentum -> PASS
    # - LTCUSDT: at local high (0.05% away) -> FAIL
    # - ADAUSDT: 0.3% from high, clear momentum -> PASS
    
    # Mock edge validation returns accordingly
    
    symbols = await engine._pick_symbols()
    
    assert 'ETHUSDT' in symbols  # Good entry, passed
    assert 'LTCUSDT' not in symbols  # At top, filtered out
    assert 'ADAUSDT' in symbols  # Good entry, passed
```

#### Test 4.4: All Symbols Filtered (Return Empty)
```python
async def test_pick_symbols_all_filtered():
    """When all symbols filtered, return empty list"""
    engine = CompoundingEngine(...)
    
    # Setup: All symbols fail one or both gates
    # (either too calm or at bad entry points)
    
    symbols = await engine._pick_symbols()
    
    assert symbols == []  # No symbols pass both gates
```

---

## Integration Tests

### Test 5.1: Complete Compounding Cycle (All Gates Pass)
```python
async def test_complete_cycle_all_gates_pass():
    """Full cycle with favorable conditions"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - Circuit breaker: Open? No
    # - Realized PnL: $100
    # - Free balance: $75
    # - Gate 1: 2 of 5 symbols volatile enough
    # - Gate 2: Both symbols have good entries  
    # - Gate 3: $100 > fees + $50 buffer? Yes
    
    await engine._check_and_compound()
    
    # Verify:
    # - Orders placed for the 2 high-quality symbols
    # - Logs show all gates passed
    # - No symbols filtered
    assert execution_manager.execute_trade.call_count == 2
```

### Test 5.2: Blocked by Gate 3 (Economic)
```python
async def test_cycle_blocked_by_economic_gate():
    """Economic gate blocks compounding when profit thin"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - Realized PnL: $20 (only)
    # - Gate 3: $20 > fees + $50 buffer? No
    
    await engine._check_and_compound()
    
    # Verify:
    # - No orders placed
    # - Log shows "Compounding blocked by economic gate"
    # - _execute_compounding_strategy() not called
    assert execution_manager.execute_trade.call_count == 0
```

### Test 5.3: Symbols Filtered, But Some Pass
```python
async def test_cycle_some_symbols_pass_filters():
    """Out of 5 candidates, 2 pass volatility, 1 passes edge"""
    engine = CompoundingEngine(...)
    
    # Setup:
    # - 5 candidate symbols by score
    # - Gate 1 filters: 5 -> 2 (low vol symbols removed)
    # - Gate 2 filters: 2 -> 1 (one at local high)
    # - Gate 3 passes
    
    await engine._check_and_compound()
    
    # Verify:
    # - Only 1 symbol ordered
    # - Logs show filtering progression
    assert execution_manager.execute_trade.call_count == 1
```

---

## Logging Validation Tests

### Test 6.1: Gate 1 Logging
```python
def test_gate1_logging():
    """Verify volatility gate logs correctly"""
    
    # PASS scenario:
    # ✅ ETHUSDT volatility 0.52% >= 0.45% (Gate 1: PASS)
    
    # FAIL scenario:
    # ❌ STABLECOIN volatility 0.15% < 0.45% (Gate 1: FAIL - too calm)
    
    # All gates filtered scenario:
    # ⚠️ All symbols filtered by volatility gate (none volatile enough)
```

### Test 6.2: Gate 2 Logging
```python
def test_gate2_logging():
    """Verify edge validation gate logs correctly"""
    
    # PASS scenario:
    # ✅ ETHUSDT edge is valid - not at high, momentum clear (Gate 2: PASS)
    
    # FAIL (at high) scenario:
    # ❌ LTCUSDT at local high (current=2105.5, high=2106.3, dist=0.95%) (Gate 2: FAIL)
    
    # FAIL (momentum) scenario:
    # ❌ ADAUSDT momentum fired recently (+0.62% move in last 5 candles) (Gate 2: FAIL)
```

### Test 6.3: Gate 3 Logging
```python
def test_gate3_logging():
    """Verify economic gate logs correctly"""
    
    # PASS scenario:
    # ✅ Economic gate PASS: $100 - $2.50 - $50 = $47.50 available (Gate 3: PASS)
    
    # FAIL scenario:
    # ❌ Economic gate FAIL: profit too thin ($20 < $52.50 fees+buffer) (Gate 3: FAIL)
```

---

## Backward Compatibility Tests

### Test 7.1: Config Not Set (Use Defaults)
```python
def test_config_defaults():
    """Missing config uses safe defaults"""
    
    engine = CompoundingEngine(config={})  # Empty config
    
    # Defaults:
    # - COMPOUNDING_MIN_VOLATILITY: 0.0045
    # - COMPOUNDING_ECONOMIC_BUFFER: 50.0
    
    assert engine._get_volatility_filter() == 0.0045
```

### Test 7.2: Config Override
```python
def test_config_override():
    """Can tune thresholds via config"""
    
    config = {
        'COMPOUNDING_MIN_VOLATILITY': 0.0060,
        'COMPOUNDING_ECONOMIC_BUFFER': 100.0
    }
    
    engine = CompoundingEngine(config=config)
    
    assert engine._get_volatility_filter() == 0.0060
```

### Test 7.3: No Breaking Changes
```python
def test_backward_compatibility():
    """Existing code still works with gates enabled"""
    
    # Old code that doesn't know about gates should still work
    # Gates are transparent to existing caller
    
    # Example: If old code calls execute_trade directly,
    # it should still work (gates only filter in _pick_symbols)
```

---

## Performance Tests

### Test 8.1: Gate Execution Time
```python
async def test_gate_performance():
    """Each gate should complete quickly"""
    
    import time
    
    engine = CompoundingEngine(...)
    
    # Gate 1: Should complete in < 100ms
    start = time.time()
    await engine._validate_volatility_gate("ETHUSDT")
    assert time.time() - start < 0.1
    
    # Gate 2: Should complete in < 50ms
    start = time.time()
    await engine._validate_edge_gate("ETHUSDT")
    assert time.time() - start < 0.05
    
    # Gate 3: Should complete in < 1ms
    start = time.time()
    await engine._validate_economic_gate(100, 5)
    assert time.time() - start < 0.001
```

### Test 8.2: Batch Symbol Processing
```python
async def test_batch_performance():
    """Filtering 20 symbols should complete in < 1s"""
    
    engine = CompoundingEngine(...)
    
    # Create 20 candidate symbols
    symbols = [f"SYM{i}USDT" for i in range(20)]
    
    import time
    start = time.time()
    filtered = await engine._pick_symbols()
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast
```

---

## Test Execution Plan

### Phase 1: Unit Tests (Quick)
- Test each gate individually
- Run in isolation
- Duration: ~5 minutes

### Phase 2: Integration Tests (Moderate)
- Test gates together
- Test with full compounding cycle
- Duration: ~10 minutes

### Phase 3: Logging Tests (Validation)
- Verify messages are informative
- Check for clarity in gate decisions
- Duration: ~5 minutes

### Phase 4: Backward Compatibility (Verification)
- Ensure no breaking changes
- Verify config handling
- Duration: ~5 minutes

### Phase 5: Performance Tests (Optional)
- Verify gate overhead < 5% of cycle time
- Check memory usage stable
- Duration: ~10 minutes

**Total**: ~35 minutes for comprehensive test coverage

---

## Success Criteria

✅ **All Unit Tests Pass**: Each gate individually works correctly
✅ **All Integration Tests Pass**: Gates work together in real cycle
✅ **Logging Clear**: Every gate decision logged with reasoning  
✅ **Backward Compatible**: Existing code unaffected
✅ **Performance**: Gate overhead < 100ms per cycle
✅ **Edge Cases**: Missing data handled gracefully

---

## Regression Prevention

Once tests pass, add to CI/CD pipeline:

```bash
# Before deployment
pytest tests/test_compounding_gates.py -v

# Check coverage
pytest --cov=core.compounding_engine tests/test_compounding_gates.py
```

Target coverage: > 90% of gate logic

---

## Known Test Challenges

1. **OHLCV Mocking**: Creating realistic price data for volatility calculation
   - Solution: Use synthetic OHLCV with controlled std

2. **Timing Dependencies**: Tests might be flaky if wall-clock dependent
   - Solution: Mock time.time() for deterministic tests

3. **Async Context**: Need proper async test fixtures
   - Solution: Use pytest-asyncio with proper fixtures

---

## Test Data Templates

### High Volatility Template (0.52%)
```python
VOLATILE_OHLCV = [
    (ts, 100.0, 102.0, 98.0, 101.0, 1000),
    (ts, 101.0, 103.5, 99.5, 102.0, 1000),
    (ts, 102.0, 104.0, 100.0, 103.0, 1000),
    # ... 21 more with similar variation
]
```

### Low Volatility Template (0.15%)
```python
CALM_OHLCV = [
    (ts, 100.0, 100.05, 99.95, 100.02, 1000),
    (ts, 100.02, 100.07, 99.97, 100.04, 1000),
    # ... 22 more with minimal variation
]
```

---

**This test specification ensures comprehensive validation of the three protective gates implementation.**

