# ✅ VERIFICATION GUIDE: Shadow Mode TRADE_EXECUTED Fix

**Date:** March 2, 2026  
**Component:** ExecutionManager (`core/execution_manager.py`)  
**Lines Modified:** 7900-8000 (shadow mode path in `_place_with_client_id`)

---

## Quick Verification (< 5 minutes)

### 1. Code Location

```bash
# Find the modified method
grep -n "_place_with_client_id" core/execution_manager.py
# Should show around line 7902
```

### 2. Check Event Emission

```bash
# Look for the new TRADE_EXECUTED emission in shadow path
grep -n "EM:ShadowMode:Canonical" core/execution_manager.py
# Should find the log statement
```

### 3. Check Post-Fill Call

```bash
# Verify _handle_post_fill is called in shadow path
grep -n "EM:ShadowMode:PostFill" core/execution_manager.py
# Should find the post-fill completion log
```

---

## Functional Verification (5-15 minutes)

### Test 1: Shadow Mode Emits TRADE_EXECUTED

**Setup:**
```python
# In test or manual run:
config.trading_mode = "shadow"
app.execution_manager.shared_state.trading_mode = "shadow"
```

**Execute:**
```python
# Place a simulated order
result = await execution_manager.execute_trade(
    symbol="ETHUSDT",
    side="BUY",
    quantity=1.0,
    tag="test_shadow"
)
```

**Verify:**
1. Check logs for: `[EM:ShadowMode:Canonical] ETHUSDT BUY TRADE_EXECUTED event emitted`
2. Check event log: `shared_state._event_log` should contain TRADE_EXECUTED event
3. Check dedup cache: `execution_manager._trade_event_emit_cache` should have entry

### Test 2: Virtual Balances Are Updated

**Setup:**
```python
# Initialize shadow mode virtual portfolio
await shared_state.initialize_virtual_portfolio_from_real()
initial_quote = shared_state.virtual_balances.get("USDT", {}).get("free", 0.0)
```

**Execute:**
```python
# Place BUY order (should reduce quote balance)
result = await execution_manager.execute_trade(
    symbol="ETHUSDT",
    side="BUY",
    quantity=0.5,
    tag="test_shadow"
)
```

**Verify:**
1. Quote balance decreased: `virtual_balances["USDT"]["free"] < initial_quote`
2. Position created: `virtual_positions["ETHUSDT"]["qty"] == 0.5`
3. Check logs for: `[EM:ShadowMode:PostFill] ETHUSDT BUY post-fill accounting complete`

### Test 3: Sell Order Updates PnL

**Setup:**
```python
# After BUY order in previous test
# Position should have qty=0.5, avg_price=<buy_price>
initial_pnl = shared_state.virtual_realized_pnl
```

**Execute:**
```python
# Place SELL order
result = await execution_manager.execute_trade(
    symbol="ETHUSDT",
    side="SELL",
    quantity=0.5,
    tag="test_shadow"
)
```

**Verify:**
1. Position qty reduced: `virtual_positions["ETHUSDT"]["qty"] == 0.0`
2. Quote balance increased
3. PnL updated: `virtual_realized_pnl >= initial_pnl` (or less if loss)
4. Check logs for: `[EM:ShadowMode:PostFill] ETHUSDT SELL post-fill accounting complete`

### Test 4: Event Log Contains TRADE_EXECUTED

**Execute:**
```python
# After any simulated trade
events = await shared_state.get_recent_events(limit=50)
trade_executed_events = [e for e in events if e["name"] == "TRADE_EXECUTED"]
```

**Verify:**
```python
assert len(trade_executed_events) > 0, "No TRADE_EXECUTED events found"
event = trade_executed_events[-1]
assert event["data"]["symbol"] == "ETHUSDT"
assert event["data"]["side"] in ("BUY", "SELL")
assert event["data"]["source"] == "ExecutionManager"
```

### Test 5: Dedup Cache Prevents Duplicate Events

**Setup:**
```python
# After a trade
event_count_before = len([e for e in shared_state._event_log if e["name"] == "TRADE_EXECUTED"])
```

**Execute:**
```python
# Try to emit the same TRADE_EXECUTED again
await execution_manager._emit_trade_executed_event(
    symbol="ETHUSDT",
    side="BUY",
    tag="test_shadow",
    order=simulated_order  # Same order as before
)
```

**Verify:**
```python
event_count_after = len([e for e in shared_state._event_log if e["name"] == "TRADE_EXECUTED"])
assert event_count_after == event_count_before, "Duplicate event was added (dedup failed)"
# Check logs for: "[TRADE_EXECUTED_DEDUP] Skip duplicate"
```

---

## Log Output Verification

### Expected Log Lines (Shadow Mode Fill)

```log
[EM:ShadowMode] ETHUSDT BUY FILLED (simulated). qty=1.00000000, price=1234.56, quote=1234.56
[EM:ShadowMode:Canonical] ETHUSDT BUY TRADE_EXECUTED event emitted. qty=1.00000000, shadow_order_id=SHADOW-abc123def456
[EM:ShadowMode:PostFill] ETHUSDT BUY post-fill accounting complete
```

### Grep Commands to Verify

```bash
# All shadow mode logs
grep "\[EM:ShadowMode" logs/clean_run.log

# Only TRADE_EXECUTED emissions
grep "\[EM:ShadowMode:Canonical\]" logs/clean_run.log

# Post-fill accounting
grep "\[EM:ShadowMode:PostFill\]" logs/clean_run.log

# Verify NO direct portfolio updates
# (should NOT see the old _update_virtual_portfolio_on_fill messages)
grep "\[EM:ShadowMode:UpdateVirtual\]" logs/clean_run.log
# ^ Should return NO results after fix
```

---

## Integration Testing

### End-to-End Shadow Mode Test

```python
async def test_shadow_mode_full_cycle():
    """Test shadow mode respects canonical accounting path."""
    
    # Setup
    config.trading_mode = "shadow"
    await app.initialize()
    
    # Initialize virtual portfolio
    await shared_state.initialize_virtual_portfolio_from_real()
    
    # BUY 1 ETHUSDT at 2000 USDT
    buy_result = await execution_manager.execute_trade(
        symbol="ETHUSDT",
        side="BUY",
        quantity=0.5,
        tag="test_shadow"
    )
    assert buy_result["ok"] == True
    
    # Check event was emitted
    events = await shared_state.get_recent_events(limit=100)
    buy_events = [e for e in events if e["name"] == "TRADE_EXECUTED" and "BUY" in str(e.get("data", {}))]
    assert len(buy_events) > 0, "BUY TRADE_EXECUTED not found"
    
    # Check position was opened
    assert "ETHUSDT" in shared_state.virtual_positions
    assert shared_state.virtual_positions["ETHUSDT"]["qty"] == 0.5
    
    # SELL 0.5 ETHUSDT at 2100 USDT (profit = 50)
    sell_result = await execution_manager.execute_trade(
        symbol="ETHUSDT",
        side="SELL",
        quantity=0.5,
        tag="test_shadow"
    )
    assert sell_result["ok"] == True
    
    # Check position was closed
    assert shared_state.virtual_positions["ETHUSDT"]["qty"] == 0.0
    
    # Check PnL was recorded
    assert shared_state.virtual_realized_pnl > 0, "No PnL recorded"
    
    # Check SELL event was emitted
    sell_events = [e for e in events if e["name"] == "TRADE_EXECUTED" and "SELL" in str(e.get("data", {}))]
    assert len(sell_events) > 0, "SELL TRADE_EXECUTED not found"
    
    print("✅ Shadow mode canonical path test PASSED")
```

---

## Comparison: Before vs After

### Before Fix

```
Shadow Mode:
├─ _simulate_fill() ✅
├─ Log fill info ✅
├─ Call _update_virtual_portfolio_on_fill() ✅ (direct mutation)
├─ Return result
└─ NO TRADE_EXECUTED EVENT ❌

Problems:
- No dedup cache populated
- TruthAuditor can't validate
- Direct balance mutation
- Missing canonical event
```

### After Fix

```
Shadow Mode:
├─ _simulate_fill() ✅
├─ Emit TRADE_EXECUTED event ✅ (canonical)
│  └─ Dedup cache populated ✅
├─ Call _handle_post_fill() ✅ (canonical)
│  ├─ Update virtual balances ✅
│  ├─ Record positions ✅
│  └─ Calculate PnL ✅
└─ Return result

Benefits:
+ Dedup cache populated
+ TruthAuditor can validate
+ Single-source-of-truth accounting
+ Full canonical path
```

---

## Troubleshooting

### Issue: No TRADE_EXECUTED Event Appears

**Check:**
1. Is `trading_mode == "shadow"`?
   ```python
   assert shared_state.trading_mode == "shadow"
   ```

2. Did the simulated fill succeed?
   ```python
   assert simulated.get("ok") == True
   assert simulated.get("executedQty", 0.0) > 0
   ```

3. Are logs at INFO level?
   ```python
   import logging
   logging.getLogger("ExecutionManager").setLevel(logging.INFO)
   ```

### Issue: Virtual Balances Not Updated

**Check:**
1. Did post-fill handler complete?
   ```python
   # Look for: "[EM:ShadowMode:PostFill] ... complete"
   ```

2. Is virtual portfolio initialized?
   ```python
   assert len(shared_state.virtual_balances) > 0
   ```

3. Check exception logs:
   ```python
   # Look for: "[EM:ShadowMode:PostFillFail]"
   ```

### Issue: STRICT_OBSERVABILITY_EVENTS Flag Causes Exception

**Solution:**
```python
# If STRICT_OBSERVABILITY_EVENTS is True, emit errors raise
# Set to False for testing:
app.config.STRICT_OBSERVABILITY_EVENTS = False
```

---

## Performance Verification

### Latency

Shadow mode with canonical path should have **negligible overhead**:
- `_emit_trade_executed_event`: O(1) with dedup cache hit
- `_handle_post_fill`: O(n) where n = number of positions (same as live)

### Memory

No additional memory overhead (uses existing post-fill infrastructure).

---

## Approval Checklist

- [ ] Code changes applied to `core/execution_manager.py`
- [ ] Shadow mode emits TRADE_EXECUTED events
- [ ] Virtual balances update via `_handle_post_fill()`
- [ ] No direct portfolio mutations in shadow path
- [ ] Dedup cache is populated
- [ ] Error handling works correctly
- [ ] Logs show canonical path taken
- [ ] Tests pass
- [ ] No regressions in live mode

---

## Next Steps

1. **Deploy to staging** - Run with extended shadow mode testing
2. **Monitor logs** - Look for canonical path evidence
3. **Run TruthAuditor** - Validate shadow fills against this schema
4. **Stress test** - Extended shadow mode with high activity
5. **Switch to live** - Once shadow passes full audit trail
