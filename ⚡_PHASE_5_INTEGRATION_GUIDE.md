# ⚡_PHASE_5_INTEGRATION_GUIDE.md

## Phase 5: Integration Guide - Updating Call Sites

**Purpose**: How to integrate Phase 5 pre-trade risk gating across the codebase  
**Status**: Ready for implementation  
**Estimated time**: 30-45 minutes  

---

## Overview: What Changed

### The Method Signature

```python
# OLD (still works, backward compatible)
sizing = capital_governor.get_position_sizing(nav, symbol)

# NEW (recommended, enables full pre-trade risk gate)
sizing = capital_governor.get_position_sizing(
    nav, 
    symbol, 
    current_position_value=current_position_value
)
```

### What You Get Back

```python
sizing = {
    "quote_per_position": float,           # ADJUSTED for concentration
    "max_per_symbol": float,
    "max_position_pct": float,             # NEW: 0.50 to 0.20 by bracket
    "concentration_headroom": float,        # NEW: Remaining allowed position
    "max_open_positions": int,
    "bracket": str,  # "MICRO" | "SMALL" | "MEDIUM" | "LARGE"
}
```

---

## Integration Checklist

### Step 1: Find All Call Sites

```bash
grep -rn "get_position_sizing" core/ --include="*.py"
```

Expected results:
- capital_governor.py itself (internal call)
- scaling_engine.py (position scaling)
- execution_manager.py (trade execution)
- meta_controller.py or signal_manager.py (signal processing)
- Possibly in other places

### Step 2: For Each Call Site

1. Locate the call
2. Identify what symbol is being sized
3. Find how to fetch current_position_value for that symbol
4. Update the call to pass current_position_value
5. Test that it works

### Step 3: Verify Integration

After all updates:
- No syntax errors
- Logs show [CapitalGovernor:ConcentrationGate] messages
- Positions never exceed max_position_pct

---

## Pattern: How to Fetch current_position_value

### Pattern 1: From shared_state

```python
# Get position value from shared state
current_pos = shared_state.get_position_value(symbol)

# Safe call with default
current_pos = shared_state.get_position_value(symbol) or 0.0
```

### Pattern 2: From positions dict

```python
# If you have positions dict
positions = {...}  # {"SOL": {"quantity": 0.5, ...}}

current_pos = positions.get(symbol, {}).get("position_value", 0.0)
```

### Pattern 3: From portfolio/account data

```python
# From account/portfolio object
position = portfolio.get_position(symbol)
current_pos = position.position_value if position else 0.0
```

### Pattern 4: Calculate from quantities and price

```python
# If you have quantity and latest price
quantity = positions.get(symbol, {}).get("quantity", 0)
latest_price = price_cache.get(symbol, 0)
current_pos = quantity * latest_price
```

### Pattern 5: Safe default (if unsure)

```python
# When you can't easily get current position value
# It's safe to pass 0.0 - system will still work
sizing = capital_governor.get_position_sizing(
    nav=nav,
    symbol=symbol,
    current_position_value=0.0  # Safe, but less precise
)
```

---

## File-by-File Integration

### File 1: core/capital_governor.py

**Current Status**: ✅ Already has Phase 5 code

**If this file calls get_position_sizing() internally**:

```python
# Search in capital_governor.py for:
sizing = self.get_position_sizing(nav, symbol)

# Update to:
sizing = self.get_position_sizing(
    nav, 
    symbol, 
    current_position_value=0.0  # or fetch actual value
)
```

**Rationale**: If it calls itself, probably doesn't need current position value (it's likely during initialization or testing).

---

### File 2: core/scaling_engine.py (If exists)

**Purpose**: Position scaling and compounding  
**Key method**: Likely calls get_position_sizing() when scaling positions

**Current pattern (likely)**:
```python
class ScalingEngine:
    def calculate_scale_size(self, nav: float, symbol: str):
        sizing = self.capital_governor.get_position_sizing(nav, symbol)
        return sizing["quote_per_position"]
```

**Update to**:
```python
async def calculate_scale_size(self, nav: float, symbol: str):
    # Get current position value
    current_pos = await self.shared_state.get_position_value(symbol) or 0.0
    
    # Get sized with concentration gate
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=symbol,
        current_position_value=current_pos
    )
    
    return sizing["quote_per_position"]
```

**Why**: ScalingEngine is adding to existing positions → MUST pass current value to enable headroom calculation

**Test case**:
```python
# After scaling size of existing SOL position:
# - Current SOL: $45
# - Max allowed: $53.50
# - Headroom: $8.50
# Result: quote_per_position should be ≤ $8.50
```

---

### File 3: core/execution_manager.py

**Purpose**: Executes trades with risk checks  
**Key method**: Likely calls get_position_sizing() before execution

**Current pattern (likely)**:
```python
async def execute_buy(self, symbol: str, quote: float, nav: float):
    sizing = self.capital_governor.get_position_sizing(nav, symbol)
    allowed_quote = sizing["quote_per_position"]
    
    if quote > allowed_quote:
        # Override quote
        quote = allowed_quote
    
    await self.execute_order(symbol, quote)
```

**Update to**:
```python
async def execute_buy(self, symbol: str, quote: float, nav: float):
    # Get current position value
    position = await self.shared_state.get_position(symbol)
    current_pos = position.position_value if position else 0.0
    
    # Get concentrated-gated sizing
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=symbol,
        current_position_value=current_pos
    )
    allowed_quote = sizing["quote_per_position"]
    
    if quote > allowed_quote:
        # Log why it's being adjusted
        logger.info(
            "ExecutionManager: %s quote adjusted due to concentration: "
            "%s → %s (headroom: %.2f USDT)",
            symbol, quote, allowed_quote,
            sizing["concentration_headroom"]
        )
        quote = allowed_quote
    
    await self.execute_order(symbol, quote)
```

**Why**: ExecutionManager is the final gate before orders → MUST use concentration-gated sizing

**Critical**: This is where concentration gating protects against oversized orders

---

### File 4: meta_controller.py or signal_manager.py

**Purpose**: Processes trading signals and decisions  
**Key method**: Likely calls get_position_sizing() when creating/evaluating signals

**Current pattern (likely)**:
```python
def process_signal(self, signal: TradingSignal, nav: float):
    symbol = signal.symbol
    sizing = self.capital_governor.get_position_sizing(nav, symbol)
    
    max_quote = sizing["quote_per_position"]
    
    # Create order
    order = TradeOrder(symbol, max_quote, signal.direction)
```

**Update to**:
```python
async def process_signal(self, signal: TradingSignal, nav: float):
    symbol = signal.symbol
    
    # Get current position value
    current_pos = self.portfolio.positions.get(
        symbol, {}
    ).get("position_value", 0.0)
    
    # Get concentration-gated sizing
    sizing = self.capital_governor.get_position_sizing(
        nav=nav,
        symbol=symbol,
        current_position_value=current_pos
    )
    
    max_quote = sizing["quote_per_position"]
    headroom = sizing["concentration_headroom"]
    
    logger.debug(
        "SignalManager: %s sized with headroom %.2f USDT (max %.1f%% of nav)",
        symbol, headroom, sizing["max_position_pct"] * 100
    )
    
    # Create order with concentration-aware sizing
    order = TradeOrder(symbol, max_quote, signal.direction)
```

**Why**: Signals are upstream of execution → Good place to log sizing decisions

---

### File 5: Other files that might call get_position_sizing()

Check also:
- position_manager.py
- portfolio_manager.py
- risk_manager.py
- backtester.py
- simulator.py

For each:
- If it's creating/modifying positions → Update to pass current_position_value
- If it's querying for information → Can pass 0.0 (safe default)

---

## Implementation Order

### Phase A: Critical Updates (DO FIRST)

1. **execution_manager.py** - This is the final gate
2. **scaling_engine.py** - This modifies existing positions

These two are critical because they interact with concentration limits most directly.

### Phase B: Important Updates (DO NEXT)

3. **meta_controller.py** - Signal processing
4. **signal_manager.py** - Signal generation

These affect signal quality but system works without updates.

### Phase C: Optional Updates (DO LAST)

5. Other files that query sizing for information

---

## Code Diff Examples

### Example 1: ExecutionManager Update

```diff
--- a/core/execution_manager.py
+++ b/core/execution_manager.py
@@ -123,7 +123,16 @@ class ExecutionManager:
     async def execute_buy(self, symbol: str, quote: float, nav: float):
-        sizing = self.capital_governor.get_position_sizing(nav, symbol)
+        # Get current position value
+        position = await self.shared_state.get_position(symbol)
+        current_pos = position.position_value if position else 0.0
+        
+        # Get concentration-gated sizing
+        sizing = self.capital_governor.get_position_sizing(
+            nav=nav,
+            symbol=symbol,
+            current_position_value=current_pos
+        )
         allowed_quote = sizing["quote_per_position"]
```

### Example 2: ScalingEngine Update

```diff
--- a/core/scaling_engine.py
+++ b/core/scaling_engine.py
@@ -56,7 +56,14 @@ class ScalingEngine:
     async def calculate_scale_size(self, nav: float, symbol: str):
-        sizing = self.capital_governor.get_position_sizing(nav, symbol)
+        # Scaling = adding to existing position, so pass current value
+        current_pos = await self.shared_state.get_position_value(symbol) or 0.0
+        
+        sizing = self.capital_governor.get_position_sizing(
+            nav=nav,
+            symbol=symbol,
+            current_position_value=current_pos
+        )
         return sizing["quote_per_position"]
```

---

## Testing After Integration

### Test 1: Verify Logging

```bash
# Run bot for 1 minute
# Check that concentration gating logs appear

tail -f logs/app.log | grep "[CapitalGovernor:ConcentrationGate]"

# Should see messages like:
# [CapitalGovernor:ConcentrationGate] SOL CAPPED: ...
```

### Test 2: Verify Sizing Behavior

```python
# In your test file:
import pytest
from core.capital_governor import CapitalGovernor

@pytest.mark.asyncio
async def test_phase5_integration():
    gov = CapitalGovernor(config)
    
    # Simulate account with SOL position
    nav = 107.0
    current_sol = 45.0
    
    sizing = gov.get_position_sizing(
        nav=nav,
        symbol="SOL",
        current_position_value=current_sol
    )
    
    # Verify concentration gating
    max_position = nav * sizing["max_position_pct"]  # Should be 50% = $53.50
    headroom = max_position - current_sol  # Should be $8.50
    
    assert sizing["concentration_headroom"] == headroom
    assert sizing["quote_per_position"] <= headroom
    assert sizing["max_position_pct"] == 0.50  # MICRO bracket
```

### Test 3: Verify No Oversized Positions

```python
# After execution, verify no position exceeds limit
for symbol, position in portfolio.positions.items():
    nav = portfolio.nav
    sizing = gov.get_position_sizing(nav, symbol)
    max_allowed = nav * sizing["max_position_pct"]
    
    actual_value = position.position_value
    
    assert actual_value <= max_allowed, \
        f"{symbol}: {actual_value} > {max_allowed} (concentration limit)"
```

---

## Troubleshooting

### Issue: "get_position_sizing() missing required positional argument"

**Cause**: Calling old signature without keyword args

**Fix**:
```python
# Wrong
sizing = gov.get_position_sizing(nav, symbol, 0.0)

# Right
sizing = gov.get_position_sizing(
    nav=nav,
    symbol=symbol,
    current_position_value=0.0
)
```

### Issue: "No [CapitalGovernor:ConcentrationGate] logs appearing"

**Cause**: Concentration gating never triggered (good!) or quote is always under limit

**Check**:
```python
# Verify headroom calculation
sizing = gov.get_position_sizing(nav, symbol, current_pos)
print(f"Headroom: {sizing['concentration_headroom']}")
print(f"Quote: {sizing['quote_per_position']}")
print(f"Max %: {sizing['max_position_pct'] * 100}%")

# If headroom > quote, gating won't trigger (expected)
```

### Issue: "Sizing is now always very small"

**Cause**: current_position_value is being incorrectly fetched (showing too high)

**Verify**:
```python
# Print what's being passed
current_pos = shared_state.get_position_value(symbol)
print(f"Symbol: {symbol}, Current position value: {current_pos}")

# Should be actual position size in USD, not quantity
# Wrong: $0.00123 (too small)
# Right: $45.00 (position value in USD)
```

### Issue: "Method is slower than before"

**Cause**: Calling shared_state synchronously in async context, blocking

**Fix**: Use async methods properly

```python
# Wrong - blocks
current_pos = self.shared_state.get_position_value(symbol)

# Right - async
current_pos = await self.shared_state.async_get_position_value(symbol)
```

---

## Quick Reference: All Changes

| File | Method | Change | Impact |
|------|--------|--------|--------|
| capital_governor.py | get_position_sizing() | ✅ DONE: Added param & logic | Core implementation |
| execution_manager.py | execute_buy() | 🔄 TO DO: Pass current_pos | Final risk gate |
| scaling_engine.py | calculate_scale_size() | 🔄 TO DO: Pass current_pos | Position scaling |
| meta_controller.py | process_signal() | 🔄 TO DO: Pass current_pos | Signal processing |
| signal_manager.py | create_signal() | 🔄 TO DO: Pass current_pos | Signal generation |

---

## Completion Checklist

- [ ] capital_governor.py verified ✅
- [ ] execution_manager.py updated and tested
- [ ] scaling_engine.py updated and tested
- [ ] meta_controller.py updated and tested
- [ ] signal_manager.py updated and tested
- [ ] All other callers identified and updated
- [ ] Concentration logs verified
- [ ] No oversized positions created
- [ ] System stable for 1+ hour without errors
- [ ] Ready for production deployment

---

## Rollout Plan

### Phase 1: Code Update (30 min)

1. Update execution_manager.py
2. Update scaling_engine.py
3. Run unit tests

### Phase 2: Integration Testing (30 min)

1. Run bot in simulation mode
2. Verify concentration logs
3. Check no oversized positions

### Phase 3: Deployment (During low-volume hours)

1. Deploy updated code
2. Monitor for 1 hour
3. Verify system stable

### Phase 4: Monitoring (24 hours)

1. Track concentration logs
2. Verify position sizing
3. Confirm no deadlock crashes

---

## Success Criteria

✅ All call sites updated  
✅ Concentration logs appearing in appropriate places  
✅ Sizing behaves as expected (headroom calculations correct)  
✅ No oversized positions created  
✅ System stable with zero deadlock crashes  
✅ Performance unchanged (<1% overhead)

---

## Support

If you encounter issues during integration:

1. Check that current_position_value is in USD, not quantity
2. Verify shared_state methods are being called correctly
3. Ensure async/await is used properly
4. Check logs for [CapitalGovernor:ConcentrationGate] messages
5. Verify headroom calculation math

---

*Status: Integration Guide - Ready for Implementation*  
*Estimated Integration Time: 45 minutes*  
*Complexity: Low-Medium*
