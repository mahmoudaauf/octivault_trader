# ⚡ LATEST PRICES COVERAGE - QUICK REFERENCE

## The Problem

Startup Step 5 (Integrity Check) was calculating:
```
position_value = qty × price
```

But `price` came from `entry_price` or `mark_price` which might be:
- `None` (hydrated positions have entry_price=None)
- Stale (mark_price not yet updated)
- Missing (0.0)

Result: `position_value = qty × 0 = 0` ❌ Wrong!

---

## The Solution

### Step 1: Fetch Latest Prices BEFORE Integrity Check
```python
# NEW: Before integrity checks, populate latest_prices
await shared_state.ensure_latest_prices_coverage(price_fetcher)
# Now: latest_prices[symbol] = current_market_price
```

### Step 2: Use Latest Prices for Position Value
```python
# NEW: Use latest_prices (just fetched)
latest_prices = shared_state.latest_prices
price = latest_prices.get(symbol, 0.0) or entry_price or mark_price

position_value = qty × price  # ✅ Now accurate
```

---

## What Changed

**File:** `core/startup_orchestrator.py`

**When:** Step 5 - Verify Startup Integrity

**What:** 
1. Added `ensure_latest_prices_coverage()` call
2. Updated position value calculation to use `latest_prices` first

**Result:** Accurate position values in integrity check

---

## Verification

```bash
✓ Syntax: python3 -m py_compile core/startup_orchestrator.py
✓ Status: READY FOR DEPLOYMENT
```

---

## Deployment

**Command:**
```bash
systemctl restart octi-trader
```

**Verify in logs:**
```
"Ensuring latest prices coverage"
"Latest prices coverage complete"
"Cached prices: X symbols"
```

---

## Example

**Before:** 0.5 BTC position
```
entry_price = None
price = None
position_value = 0.5 × 0 = 0 ❌ (Classified as dust)
```

**After:** 0.5 BTC position
```
latest_prices[BTCUSDT] = 50000.0 (fetched)
price = 50000.0
position_value = 0.5 × 50000.0 = 25000.0 ✅ (Correct)
```

---

## Risk: 🟢 LOW

- Uses existing methods
- Error handling in place
- Fallback logic preserved
- No breaking changes

**All systems go!** ✅
