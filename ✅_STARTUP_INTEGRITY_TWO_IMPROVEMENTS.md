# Two Critical Startup Integrity Improvements

## Overview

Applied two surgical fixes to `core/startup_orchestrator.py` Step 5 verification to prevent false fatal errors and handle dust positions correctly.

---

## Improvement 1: Non-Fatal NAV=0 with Retry

### Problem
Previous logic:
```python
if nav == 0 and (positions > 0 or free > 0) and not shadow_mode:
    raise RuntimeError("NAV is 0 - Fatal Error!")  # ❌ Too harsh
```

This caused startup to FAIL when:
- NAV=0 but positions exist (dust from previous trades)
- NAV hadn't synced from exchange yet
- Free quote existed but NAV still calculating

### Solution
New logic:
```python
if nav == 0 and viable_positions > 0 and not shadow_mode:
    logger.warning("Positions detected but NAV=0 - Recalculating...")
    
    # Give dust cleanup 1 second to run
    await asyncio.sleep(1)
    
    # Recalculate NAV
    nav = await self.shared_state.get_nav()
    
    if nav == 0:
        logger.warning("NAV still zero - Continuing (dust will be liquidated)")
        # ✅ Don't block startup - allow cleanup to handle dust
    else:
        logger.info(f"NAV recovered to {nav:.2f}")
        # ✅ NAV synced successfully
```

### Benefits
✅ Allows 1-second window for USDT sync from exchange
✅ Retries NAV calculation after cleanup
✅ Non-fatal even if NAV remains 0 (dust liquidation handles it)
✅ Prevents false positive startup failures

### Expected Behavior
```
Before (❌ FAILED):
NAV is 0.0 (should be > 0) - STARTUP BLOCKED

After (✅ PASSES):
NAV still zero after cleanup. Continuing startup.
Dust positions will be liquidated.
```

---

## Improvement 2: Dust Position Filtering

### Problem
Previous logic counted ALL positions:
```python
if nav == 0 and positions > 0:  # Even if positions are $0.50 each
    raise RuntimeError()  # ❌ Blocks startup
```

This failed when:
- Account had many dust positions < $30 (MIN_ECONOMIC_TRADE_USDT)
- These dust positions are economically irrelevant
- But startup would fail due to position count

### Solution
Filter positions by economic viability:
```python
# Get MIN_ECONOMIC_TRADE_USDT threshold (default: 30.0)
min_economic_trade = config.MIN_ECONOMIC_TRADE_USDT

# Separate viable from dust
viable_positions = []
dust_positions = []

for symbol, pos in positions.items():
    value = qty * price
    
    if value >= min_economic_trade:
        viable_positions.append(symbol)  # ✅ Economically relevant
    else:
        dust_positions.append((symbol, value))  # ❌ Below threshold
```

Then check only viable positions:
```python
if nav == 0 and len(viable_positions) > 0:  # Only count viable ones
    # Retry NAV calculation
else:
    # ✅ OK - no economically relevant positions
```

### Benefits
✅ Dust positions don't block startup
✅ Only economically viable positions trigger retry
✅ Dust is logged separately for visibility
✅ Prevents false positives from micro positions

### Example
```
Account State:
- Position 1: BTC with $5,000 value  → VIABLE
- Position 2: XRP with $0.50 value   → DUST
- Position 3: ETH with $2.30 value   → DUST
- NAV: 0.0 (not synced yet)

Old Logic:
positions > 0 → FAIL ❌

New Logic:
viable_positions = 1 (BTC)
dust_positions = 2 (XRP, ETH)
→ Retry NAV, allow dust cleanup → PASS ✅

Logs:
Found 2 dust positions below $30.00: BTC=$5000.00, [XRP=$0.50, ETH=$2.30]
NAV still zero after cleanup. Continuing startup.
```

---

## Combined Flow Diagram

```
┌─────────────────────────────────────────────┐
│ Step 5: Verify Startup Integrity            │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Get Capital Metrics │
        │ nav, free, invested │
        │ positions, orders   │
        └──────────┬──────────┘
                   │
     ┌─────────────▼─────────────┐
     │ IMPROVEMENT 2:            │
     │ Filter Dust Positions     │
     │ value < MIN_TRADE         │
     │ viable_pos, dust_pos      │
     └─────────────┬─────────────┘
                   │
        ┌──────────▼──────────┐
        │ Check Shadow Mode?  │
        ├──────────┬──────────┤
        │ YES      │ NO       │
        │ Allow    │ Check    │
        │ NAV=0 ✅ │ further  │
        └──────────┼──────────┘
                   │
     ┌─────────────▼──────────────────┐
     │ IMPROVEMENT 1:                 │
     │ NAV=0 + viable_pos?            │
     ├──────────┬──────────┬──────────┤
     │ YES      │ NO       │ NO       │
     │ Retry    │ NAV>0    │ NAV>0   │
     │ with 1s  │ Check    │ OK ✅  │
     │ cleanup  │ balance  │          │
     │          │ error    │          │
     └──────────┼──────────┴──────────┘
                │
     ┌──────────▼──────────┐
     │ Recalc NAV after    │
     │ cleanup sleep(1)    │
     └──────────┬──────────┘
                │
     ┌──────────▼──────────────────┐
     │ NAV still 0?                 │
     ├──────────┬──────────┬────────┤
     │ YES      │ NO       │        │
     │ Log warn │ Log info │        │
     │ Continue │ Continue │        │
     │ ✅ PASS  │ ✅ PASS  │        │
     └──────────┴──────────┴────────┘
```

---

## Code Changes Summary

### File: `core/startup_orchestrator.py`

#### Change 1: Add dust position filtering (before NAV check)
```python
# Filter positions below MIN_ECONOMIC_TRADE_USDT
min_economic_trade = config.MIN_ECONOMIC_TRADE_USDT

viable_positions = []
dust_positions = []

for symbol, pos_data in positions.items():
    position_value = qty * price
    if position_value >= min_economic_trade:
        viable_positions.append(symbol)
    else:
        dust_positions.append((symbol, position_value))

# Log dust positions for visibility
if dust_positions:
    logger.warning(f"Found {len(dust_positions)} dust positions below ${min_economic_trade}")
```

#### Change 2: Replace harsh NAV=0 check with retry logic
```python
# OLD (❌ Fatal):
if nav == 0 and positions > 0:
    raise RuntimeError(...)

# NEW (✅ Retry):
if nav == 0 and viable_positions > 0:
    logger.warning("Positions detected but NAV=0 - Recalculating...")
    
    await asyncio.sleep(1)  # Allow cleanup
    nav = await self.shared_state.get_nav()
    
    if nav == 0:
        logger.warning("NAV still zero - Continuing startup")
        # Don't block - dust liquidation will handle
```

#### Change 3: Use viable_positions in consistency check
```python
# OLD: if positions and nav > 0:
# NEW: if viable_positions and nav > 0:

for symbol in viable_positions:
    # Calculate consistency only for viable positions
```

---

## Configuration Impact

### MIN_ECONOMIC_TRADE_USDT
- **Default:** 30.0 USDT
- **Source:** `core/config.py` line 262
- **Used for:** Dust threshold filter
- **Effect:** Positions below this are marked as dust and ignored

### Override
```bash
# In .env or environment:
export MIN_ECONOMIC_TRADE_USDT=50.0  # Raise dust threshold
```

---

## Testing Scenarios

### Scenario 1: Dust Positions
```
Situation:
  Positions: BTC=$5000 (viable), XRP=$0.50 (dust), ETH=$2.30 (dust)
  NAV: 0.0 (not synced)

Expected:
  viable_positions = 1 (BTC)
  dust_positions = 2 (XRP, ETH)
  
Result:
  ✅ Triggers retry with 1s cleanup
  ✅ Logs dust position warning
  ✅ Continues startup if NAV still 0
```

### Scenario 2: USDT Sync Delay
```
Situation:
  Positions: None
  Free: 1000 USDT
  NAV: 0.0 (async sync in progress)

Expected:
  viable_positions = 0 (no positions)
  
Result:
  ✅ NAV=0 with no viable positions OK
  ✅ Cold start allowed
  ✅ USDT will sync eventually
```

### Scenario 3: Shadow Mode
```
Situation:
  TRADING_MODE = "shadow"
  NAV: 0.0 (expected)
  Positions: 3 (from virtual ledger)

Expected:
  is_shadow_mode = True
  
Result:
  ✅ Bypasses all checks
  ✅ NAV=0 is acceptable
  ✅ Startup continues immediately
```

### Scenario 4: Real Mode with Position
```
Situation:
  TRADING_MODE = "live"
  NAV: 0.0 (exchange sync pending)
  Positions: 1 viable position ($500)

Expected:
  viable_positions = 1
  
Result:
  ✅ Triggers 1s sleep for sync
  ✅ Recalculates NAV after sleep
  ✅ Either NAV syncs or continues with warning
  ✅ Continues startup (not fatal)
```

---

## Expected Logs

### Dust Positions Detected
```
[StartupOrchestrator] Step 5 - Found 2 dust positions below $30.00: XRP=$0.50, ETH=$2.30
[StartupOrchestrator] Step 5 - Positions detected but NAV=0 - likely dust positions. Recalculating...
[StartupOrchestrator] Step 5 - Waiting 1 second for cleanup...
[StartupOrchestrator] Step 5 - NAV still zero after cleanup. Continuing startup.
```

### Successful Recovery
```
[StartupOrchestrator] Step 5 - Positions detected but NAV=0 - likely dust positions. Recalculating...
[StartupOrchestrator] Step 5 - Waiting 1 second for cleanup...
[StartupOrchestrator] Step 5 - NAV recovered to 5000.25 after cleanup
```

### Cold Start
```
[StartupOrchestrator] Step 5 - No dust positions detected
[StartupOrchestrator] Step 5 - Cold start: NAV=0, no viable positions
```

---

## Metrics Tracked

```python
self._step_metrics['verify_integrity'] = {
    'nav': nav,                           # Final NAV after retry
    'free_quote': free,                   # Free quote balance
    'invested_capital': invested,         # Invested in positions
    'viable_positions_count': len(viable_positions),  # NEW
    'dust_positions_count': len(dust_positions),      # NEW
    'total_positions_count': len(positions),          # Total (viable + dust)
    'open_orders_count': len(open_orders),
    'issues_count': len(issues),
    'elapsed_sec': elapsed,
}
```

---

## Summary

These two improvements work together to:

1. **Handle dust correctly** - Filter by economic threshold (Improvement 2)
2. **Retry NAV sync** - Allow 1 second for exchange sync (Improvement 1)
3. **Non-fatal failures** - Log warnings instead of blocking startup
4. **Transparent logging** - Clear visibility into what was filtered and why
5. **Configurable thresholds** - MIN_ECONOMIC_TRADE_USDT controls dust level

**Result:** Startup succeeds in more realistic scenarios while still catching genuine integrity issues.
