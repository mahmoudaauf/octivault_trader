# ⚡ Bootstrap Minimum Allocation - Quick Reference

## The Fix in One Sentence

Bootstrap executor now forces allocation to `max(capital * risk_fraction, min_notional * 1.1)` so orders pass exchange minimums.

## The Problem

```
Capital = $91
Risk = 10%
Calculated = $9.10
Exchange Minimum = $10.00
Result = ❌ REJECTED
```

## The Solution

```python
allocation = max(
    capital * risk_fraction,        # What we calculated
    min_notional * 1.1              # What exchange requires + 10% buffer
)

# Example:
allocation = max(9.10, 10.0 * 1.1) = max(9.10, 11.0) = $11.00 ✅
```

## Where It's Implemented

**File**: `core/execution_manager.py`  
**Method**: `_place_market_order_quote()` (lines ~7190-7250)  
**Context**: Bootstrap phase only (detected via tag/flag)

## How It Works

1. **Detect Bootstrap**: Check if `bypass_min_notional=True` or tag contains "bootstrap"
2. **Get Exchange Min**: Fetch minNotional from `exchange_client.get_symbol_info()`
3. **Calculate Floor**: `minimum = min_notional × 1.1` (10% safety margin)
4. **Enforce Allocation**: `quote = max(quote, minimum)`
5. **Log Decision**: Show before/after values for transparency

## Expected Behavior

### Scenario 1: Low Capital (91 USDT)
```
input_quote = 9.10 USDT
min_notional = 10.0
minimum_required = 10.0 * 1.1 = 11.0
output_quote = max(9.10, 11.0) = 11.0 USDT ✅
```

### Scenario 2: Sufficient Capital (200 USDT)
```
input_quote = 20.0 USDT
min_notional = 10.0
minimum_required = 10.0 * 1.1 = 11.0
output_quote = max(20.0, 11.0) = 20.0 USDT ✅ (no change)
```

### Scenario 3: Non-Bootstrap Trade
```
tag = "meta/normal"  (NOT bootstrap)
input_quote = 9.10 USDT
enforcement_active = FALSE (skipped for non-bootstrap)
output_quote = 9.10 USDT (no change)
```

## Log Messages

**When Enforcement Applied**:
```
[EM:BOOTSTRAP_ALLOC] 🚀 MINIMUM ALLOCATION ENFORCED:
  Original: 9.10 USDT
  MinNotional: 10.00 USDT
  Required: 11.00 USDT (min_notional * 1.1)
  Adjusted: 11.00 USDT ✓ Now passes minimum
```

**When Not Needed**:
```
[EM:BOOTSTRAP_ALLOC] ✓ Allocation already meets minimum:
  Quote: 20.00 USDT >= Required: 11.00 USDT
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_notional` | 10.0 USDT | Exchange minimum from filters |
| `safety_margin` | 1.1x | Enforce at 110% of minimum |
| `detection` | tag, flag | Bootstrap context detection |
| `scope` | BUY only | SELL uses position quantity, not allocation |

## Benefits

✅ **Passes Exchange**: Orders guaranteed to meet minNotional  
✅ **Efficient Capital**: Uses minimum necessary, not more  
✅ **Transparent**: Logs show exactly what changed and why  
✅ **Safe**: Only affects bootstrap context  
✅ **Graceful**: Falls back to original if enforcement fails  

## Testing the Fix

### Verify Enforcement Works
```bash
# Watch logs for bootstrap trades
grep "BOOTSTRAP_ALLOC" logs/meta_controller.log

# Check allocation was enforced
grep "MINIMUM ALLOCATION ENFORCED" logs/meta_controller.log
```

### Check Exchange Behavior
1. Send bootstrap order with capital < min_notional
2. Verify it gets adjusted to minimum_required
3. Confirm order accepted by exchange
4. Verify fill executes successfully

## Configuration

No new config needed. Uses existing:
- `MIN_NOTIONAL_USDT` in config.py (default: 10.0)
- Exchange symbol filters (get_symbol_info)

## Rollback

If issues arise:
1. Comment out lines 7200-7250 in execution_manager.py
2. Restart system
3. Orders will use original allocation (may fail on minNotional)

## Summary

Bootstrap orders are now guaranteed to pass exchange minNotional checks by enforcing a minimum allocation of `min_notional × 1.1`. This simple but critical fix ensures the system can always bootstrap, even with very small capital amounts.

**Status**: ✅ Implemented and ready for testing  
**Risk**: Low (only affects bootstrap, graceful fallback)  
**Impact**: High (enables reliable bootstrap execution)
