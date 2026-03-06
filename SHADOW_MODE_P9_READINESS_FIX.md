# Shadow Mode P9 Readiness Gate Fix

## Problem

In shadow mode, the trading bot was **blocking all BUY signals** at the P9 readiness gate in `MetaController._execute_decision()`.

The gate requires two events to be set:
1. `market_data_ready_event` - Signals that live market data is available
2. `accepted_symbols_ready_event` - Signals that trading symbols are ready

### Root Cause

In shadow mode with synthetic market data:
- **Live market data doesn't exist** - Shadow mode uses simulated prices and OHLCV data
- **Market data stream may not trigger** the `market_data_ready_event` because there's no continuous WebSocket stream  
- **P9 gate blocks ALL BUYs** unless BOTH events are set (live mode requirement)

Result: Even when signals are generated and decisions are built, they are **skipped at execution time**.

## Solution

### 1. Shadow Mode Bypass in `_execute_decision()` (Line ~12730)

Changed the readiness check to:

**Before:**
```python
# Live mode: Require BOTH market_data_ready AND accepted_symbols_ready
if not (md_ready and as_ready):
    return {"ok": False, "reason": "p9_readiness_gate"}
```

**After:**
```python
if is_shadow_mode:
    # Shadow mode: Only require accepted_symbols (via event OR actual population)
    readiness_ok = as_ready or has_accepted_symbols
    if not readiness_ok:
        return {"ok": False, "reason": "p9_readiness_gate_shadow"}
else:
    # Live mode: Require BOTH market_data AND accepted_symbols
    if not (md_ready and as_ready):
        return {"ok": False, "reason": "p9_readiness_gate"}
```

### 2. Bootstrap Seed Bypass (Line ~8420)

Applied the same logic to bootstrap seed execution:
- Check if symbols are actually populated in `shared_state.accepted_symbols` as a fallback
- In shadow mode, accept either the event being set OR symbols being present
- In live mode, keep the strict requirement

### 3. Fallback Check

Added fallback detection for accepted symbols:
```python
has_accepted_symbols = bool(getattr(self.shared_state, "accepted_symbols", {}))
```

This handles edge cases where the `accepted_symbols_ready_event` hasn't been set yet, but the symbols are actually populated in the dict.

## Changes Made

**File: `core/meta_controller.py`**

1. **Line ~12730-12762**: Updated P9 readiness gate in `_execute_decision()` method
   - Added `is_shadow_mode` detection
   - Added fallback check for actual symbol population
   - Implemented conditional readiness logic (shadow vs live)

2. **Line ~8420-8455**: Updated bootstrap seed readiness gate  
   - Added same shadow mode bypass logic
   - Added fallback symbol population check

## Behavior

### In Shadow Mode
- ✅ BUY signals execute if symbols are configured (event OR actual population)
- ✅ Market data readiness is NOT required (synthetic data)
- ✅ Virtual positions and fills are simulated correctly

### In Live Mode
- ✅ Strict P9 gate: Both `market_data_ready_event` AND `accepted_symbols_ready_event` required
- ✅ No change to existing behavior

## Testing

To test this fix:

```bash
export TRADING_MODE=shadow
nohup python -u main_phased.py > logs/clean_run.log 2>&1 &
```

Check logs for:
- `[Meta:P9-GATE]` - Should show "has_symbols=True" or event set
- `[Meta:POST_BUILD]` - Should show `decisions_count > 0`
- `execute_trade` - Should see actual trade execution attempts

## Impact

- ✅ Fixes BUY signal execution in shadow mode
- ✅ Maintains strict validation in live mode
- ✅ Adds robustness with fallback symbol checking
- ✅ No breaking changes to live trading behavior
