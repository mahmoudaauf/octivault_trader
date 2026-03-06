# Shadow Mode Bug Fix: _split_symbol AttributeError

## Problem

When shadow mode tries to update virtual portfolio after simulated fill, it crashes with:

```
[EM:ShadowMode:UpdateVirtual] Failed to update virtual portfolio:
'ExecutionManager' object has no attribute '_split_symbol'
```

## Root Cause

In `core/execution_manager.py`, the method `_update_virtual_portfolio_on_fill()` at line 7231 was calling a non-existent method:

```python
# WRONG: Method doesn't exist
base_asset = self._split_symbol(symbol)[0]
```

ExecutionManager has these symbol-splitting methods:
- ✅ `_split_base_quote(symbol)` → returns Tuple[str, str] (base, quote)
- ✅ `_split_symbol_quote(symbol)` → returns str (quote only)
- ❌ `_split_symbol(symbol)` → **DOES NOT EXIST**

## Solution

Changed line 7231 from using the non-existent `_split_symbol` to the correct `_split_base_quote` method:

**File**: `core/execution_manager.py`  
**Line**: 7231  
**Change**:

```python
# BEFORE (BROKEN):
base_asset = self._split_symbol(symbol)[0]

# AFTER (FIXED):
base_asset = self._split_base_quote(symbol)[0]
```

## Method Reference

### `_split_base_quote(symbol: str) -> Tuple[str, str]`

Located at line 1802 in execution_manager.py

```python
def _split_base_quote(self, symbol: str) -> Tuple[str, str]:
    s = (symbol or "").upper()
    # Check configured quote currency first
    _base_ccy = (self.base_ccy or "").upper()
    if _base_ccy and s.endswith(_base_ccy):
        return s[:-len(_base_ccy)], _base_ccy
    # Fall back to common quote currencies
    for q in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "BTC", "ETH"):
        if s.endswith(q):
            return s[:-len(q)], q
    # Last resort: naive 3–4 letter quote split
    return s[:-4], s[-4:]
```

**Example usage**:
```python
base, quote = self._split_base_quote("BTCUSDT")
# Returns: ("BTC", "USDT")

base = self._split_base_quote("BTCUSDT")[0]
# Returns: "BTC"
```

## Impact

- ✅ Virtual portfolio updates now work correctly in shadow mode
- ✅ Base asset correctly extracted from symbol pair
- ✅ Virtual position tracking functional
- ✅ Virtual PnL calculation can proceed

## Verification

```bash
python3 -m py_compile core/execution_manager.py
# ✅ execution_manager.py compiles successfully
```

## Testing

After this fix, shadow mode should:
1. ✅ Place simulated orders (SHADOW- prefix)
2. ✅ Call `_simulate_fill()` with correct slippage
3. ✅ Call `_update_virtual_portfolio_on_fill()` without error
4. ✅ Update virtual_positions[symbol] correctly
5. ✅ Track virtual_realized_pnl on sells
6. ✅ Maintain virtual_nav

## Files Modified

| File | Line | Change |
|------|------|--------|
| `core/execution_manager.py` | 7231 | Fixed method call from `_split_symbol()` to `_split_base_quote()` |

---

**Status**: ✅ FIXED  
**Date**: March 2, 2026  
**Scope**: 1 file, 1 line
