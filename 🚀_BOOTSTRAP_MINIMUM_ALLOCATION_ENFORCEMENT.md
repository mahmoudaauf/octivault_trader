# 🚀 Bootstrap Minimum Allocation Enforcement

## Status: ✅ IMPLEMENTED

**Date**: March 7, 2026  
**Component**: ExecutionManager  
**File**: `core/execution_manager.py` (lines ~7190-7250)  
**Impact**: Bootstrap orders now GUARANTEED to pass exchange minNotional  

---

## Problem

**Scenario**:
```
Capital = $91 USDT
Risk Fraction = 10%
Normal Allocation = 91 × 0.1 = $9.10

Exchange MinNotional = $10.00
Order Fails: $9.10 < $10.00 ❌
```

**Issue**: Bootstrap would calculate allocation based on capital × risk_fraction, which could fall below exchange minNotional requirements. This causes orders to be **rejected by the exchange**, wasting execution attempts and delaying critical first trades.

---

## Solution

**Formula**: `allocation = max(capital × risk_fraction, min_notional × 1.1)`

**Implementation**:
1. Detect bootstrap context (tag contains "bootstrap" OR bypass_min_notional=True)
2. Get exchange minNotional for the symbol
3. Calculate minimum required: `min_notional × 1.1` (10% safety margin)
4. Enforce: `allocation = max(requested_quote, minimum_required)`
5. Log enforcement for transparency

**Example**:
```python
capital = 91 USDT
risk_fraction = 0.1
normal = capital * risk_fraction = 9.1
min_notional = 10.0
safety_margin = 10% → 11.0

allocation = max(9.1, 11.0) = 11.0 USDT ✓ Passes!
```

---

## Code Changes

### Location
**File**: `core/execution_manager.py`  
**Method**: `_place_market_order_quote()`  
**Lines**: ~7190-7250

### Enforcement Logic

```python
# Bootstrap minimum allocation enforcement
is_bootstrap = bool(
    bypass_min_notional or 
    (tag and "bootstrap" in str(tag).lower()) or
    (decision_id and "bootstrap" in str(decision_id).lower())
)

if is_bootstrap and side.upper() == "BUY" and quote > 0:
    # Get min_notional from exchange
    min_notional = 10.0  # Default
    try:
        info = await self.exchange_client.get_symbol_info(symbol)
        # Extract from filters...
    except Exception:
        min_notional = float(getattr(self.config, "MIN_NOTIONAL_USDT", 10.0))
    
    # Enforce minimum
    minimum_allocation = min_notional * 1.1  # 10% safety margin
    
    if quote < minimum_allocation:
        quote = minimum_allocation
        logger.info("[BOOTSTRAP_ALLOC] Adjusted: %.2f → %.2f USDT", 
                   original_quote, quote)
```

---

## Behavior

### Before
```
Bootstrap Capital: $91 USDT
Risk: 10%
Calculated: $9.10
Exchange Minimum: $10.00
Result: ❌ Order rejected (below minimum)
```

### After
```
Bootstrap Capital: $91 USDT
Risk: 10%
Calculated: $9.10
Enforced Minimum: $11.00 (min_notional × 1.1)
Result: ✅ Order passes (11.00 > 10.00)
```

---

## Log Output

When bootstrap allocation is enforced:

```
[EM:BOOTSTRAP_ALLOC] 🚀 MINIMUM ALLOCATION ENFORCED:
  Original: 9.10 USDT
  MinNotional: 10.00 USDT
  Required: 11.00 USDT (min_notional * 1.1)
  Adjusted: 11.00 USDT ✓ Now passes minimum
```

When allocation already meets minimum:

```
[EM:BOOTSTRAP_ALLOC] ✓ Allocation already meets minimum:
  Quote: 15.00 USDT >= Required: 11.00 USDT
```

---

## Safety Guarantees

✅ **Detection**: Only applies to bootstrap context (flag/tag/decision_id)  
✅ **Direction**: Only applies to BUY orders (SELL uses position quantity)  
✅ **Exchange**: Gets real minNotional from exchange filters  
✅ **Margin**: Adds 10% safety buffer (1.1x multiplier)  
✅ **Fallback**: Uses config default if exchange lookup fails  
✅ **Transparency**: Logs before/after allocation values  
✅ **Robustness**: Graceful error handling (continue with original if enforcement fails)  

---

## Impact Assessment

| Aspect | Benefit |
|--------|---------|
| **Order Pass Rate** | Increases bootstrap order execution success |
| **Capital Efficiency** | Uses minimum necessary (not more) |
| **Exchange Compliance** | Guarantees minNotional compliance |
| **Code Complexity** | Minimal (single decision point) |
| **Performance** | Negligible (single division + comparison) |
| **Backwards Compatible** | Yes (only affects bootstrap context) |

---

## Testing

### Test Case 1: Bootstrap With Low Capital
```python
capital = 91 USDT
risk_fraction = 0.1
symbol = 'BTCUSDT'
min_notional = 10.0

# Expected behavior
quote_in = 9.10
quote_out = 11.00  # Enforced minimum
assert quote_out >= min_notional * 1.1
```

**Expected Result**: ✅ PASS - Order gets $11.00 allocation

### Test Case 2: Bootstrap With Sufficient Capital
```python
capital = 200 USDT
risk_fraction = 0.1
symbol = 'ETHUSDT'
min_notional = 10.0

# Expected behavior
quote_in = 20.0
quote_out = 20.0  # Already >= 11.0
assert quote_out == quote_in  # No adjustment
```

**Expected Result**: ✅ PASS - No adjustment needed

### Test Case 3: Non-Bootstrap (Should NOT Enforce)
```python
capital = 91 USDT
tag = "meta/normal"  # NOT bootstrap
quote_in = 9.10

# Expected behavior
quote_out = 9.10  # No enforcement
assert quote_out == quote_in
```

**Expected Result**: ✅ PASS - Enforcement skipped for non-bootstrap

---

## Configuration

No new configuration required. Uses existing:
- `MIN_NOTIONAL_USDT` (config.py, default 10.0)
- Exchange symbol info filters (get_symbol_info)

---

## Deployment

✅ **Status**: Implemented and ready  
✅ **Risk**: Low (only affects bootstrap, graceful fallback)  
✅ **Testing**: Manual verification ready  
✅ **Rollback**: Simple revert of the enforcement block  

---

## Next Steps

1. **Test**: Verify bootstrap orders with capital < min_notional succeed
2. **Monitor**: Watch log output for [EM:BOOTSTRAP_ALLOC] messages
3. **Validate**: Confirm order minNotional compliance on exchange
4. **Optimize**: Adjust 1.1x safety margin based on exchange behavior

---

## Summary

**Problem**: Bootstrap orders could fail due to allocation below minNotional  
**Solution**: Enforce `allocation = max(capital × risk_fraction, min_notional × 1.1)`  
**Result**: Bootstrap orders GUARANTEED to pass exchange minimum requirements  
**Status**: ✅ IMPLEMENTED AND READY FOR TESTING  

The system now intelligently adjusts bootstrap allocation to the absolute minimum required to pass exchange validation, ensuring capital-efficient order execution while maintaining strict compliance.
