# ⚡ DUST RECOVERY INVARIANTS - QUICK REFERENCE

## The Three Critical Rules

### Rule #1: Dust Must NOT Block BUY Signals

```
IF violated:
├─ BUY signal appears
├─ Dust position exists on same symbol
├─ Signal gets rejected
└─ Deadlock: Dust forever blocks recovery

MUST enforce:
├─ Only SIGNIFICANT positions block entry
├─ Dust below floor ALLOWS entry
├─ Unhealable dust ALLOWS entry
└─ Result: P0 promotion can execute
```

**Status**: ❌ **VIOLATED** (Lines 9902-9930, `meta_controller.py`)

---

### Rule #2: Dust Must NOT Count Toward Position Limits

```
IF violated:
├─ Position limit = N (e.g., 8 positions)
├─ Portfolio has 7 significant + 1 dust
├─ Dust fills the limit
└─ No new signals can execute

MUST enforce:
├─ Position limit counts ONLY significant positions
├─ Dust excluded from count
├─ Result: Can still add positions despite dust
```

**Status**: ❌ **VIOLATED** (Consequence of Rule #1 violation)

---

### Rule #3: Dust Must Be REUSABLE When Signal Appears

```
IF violated:
├─ Dust position: $5 notional
├─ BUY signal appears: confidence 0.90
├─ System rejects signal (dust blocks)
└─ Dust never grows to viability

MUST enforce:
├─ Allow BUY to merge with existing dust
├─ Dust $5 + new capital $25 = Position $30
├─ Dust graduates to viable status
└─ Capital recovered, P0 success
```

**Status**: ❌ **VIOLATED** (Consequence of Rule #1 violation)

---

## Where the Bug Lives

### File: `core/meta_controller.py`

**Lines**: 9902-9930  
**Function**: `_build_decisions()`

**Current Code** (BROKEN):
```python
if existing_qty > 0:
    # ❌ Rejects ALL positions, treating dust same as viable
    self.logger.info("[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY...")
    continue  # Drops the signal
```

**Should Be** (FIXED):
```python
if existing_qty > 0:
    blocks, pos_value, sig_floor, reason = await self._position_blocks_new_buy(sym, existing_qty)
    if blocks:
        # ✅ Only skip if position is SIGNIFICANT
        self.logger.info("[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY...")
        continue
    else:
        # ✅ Dust doesn't block - allow entry for promotion
        pass  # Continue processing signal
```

---

## Why This Breaks Dust Recovery

### The Recovery Pipeline

```
Stage 1: Position Becomes Dust
    Capital drops → Position falls below minNotional
    Result: Dust position created
    Status: ✅ Works (not blocked)

Stage 2: DustMonitor Detects It
    Monitors position health (age, notional, recovery potential)
    Status: ✅ Works (not blocked)

Stage 3: Strong Signal Appears ← 🚫 BLOCKED HERE
    High-confidence BUY signal triggers
    P0 promotion logic checks: "Can we promote this dust?"
    Should answer: YES if signal + dust exist
    
    But gets rejected at ONE_POSITION_GATE first! ❌
    
    Result: P0 promotion never reaches decision point

Stage 4: P0 Dust Promotion (BLOCKED)
    Cannot execute because signal never made it through gate
    Dust stays as dust
    Capital never recovered

Stage 5: Accumulation Resolution (BLOCKED)
    Accumulation would grow dust toward minNotional
    But signal was rejected, so no accumulation
    Dust becomes static

Stage 6: Capital Starvation
    Dust blocks everything
    Free capital decreases
    Eventually: Capital floor breach
    No escape (dust promotion blocked)
    Result: ☠️ SYSTEM DEADLOCK
```

### Impact

| Stage | Expected | Actual | Impact |
|-------|----------|--------|--------|
| 1. Dust Created | ✅ | ✅ | Position marked dust |
| 2. Monitoring | ✅ | ✅ | Health tracked |
| 3. Signal → P0 | ✅ | ❌ | Signal rejected at gate |
| 4. Promotion | ✅ | ❌ | Never executes |
| 5. Accumulation | ✅ | ❌ | Blocked by gate |
| 6. Recovery | ✅ | ❌ | Dust permanent |

---

## The Fix is Simple

### What Needs to Change

**1 Line** to change (conceptually):
```python
# FROM:
if existing_qty > 0:  # ❌ Crude check

# TO:
if existing_qty > 0 and await self._position_blocks_new_buy(sym, existing_qty)[0]:  # ✅ Smart check
```

### What Already Exists

The method `_position_blocks_new_buy()` at lines 1771-1809:
- ✅ Checks if position is PERMANENT_DUST (< $1.0)
- ✅ Checks if position is below SIGNIFICANT_FLOOR
- ✅ Checks if position is UNHEALABLE
- ✅ Returns False for all dust cases
- ✅ Returns True only for significant positions

**It has all the logic. Just needs to be called.**

### After the Fix

```
Dust position exists
    ↓
BUY signal appears
    ↓
Check: _position_blocks_new_buy()
    ├─ Value: $5
    ├─ Floor: $10
    ├─ Value < Floor? YES
    └─ Return: False (doesn't block)
    ↓
✅ Signal allowed through
    ↓
P0 Promotion checks: "Can we promote?"
    ├─ Dust exists? YES
    ├─ Signal exists? YES
    └─ Both conditions met → EXECUTE
    ↓
✅ P0 Promotion runs
    ├─ Scale dust from $5 → $30
    ├─ Capital recovered
    └─ System escapes starvation
```

---

## Verification Tests

### Test 1: Dust Should Allow Entry

```python
# Setup
positions['ETHUSDT'] = {'qty': 0.00133, 'price': 3.00, 'value': $4}
signal = {'symbol': 'ETHUSDT', 'action': 'BUY', 'confidence': 0.90}

# Build decisions
decisions = await meta._build_decisions([signal])

# Verify
assert len(decisions) > 0, "Dust should NOT block entry!"
assert decisions[0]['action'] == 'BUY'
```

**Current Result**: ❌ FAILS (signal rejected)  
**After Fix**: ✅ PASSES (signal allowed)

### Test 2: Significant Should Still Block

```python
# Setup
positions['BTCUSDT'] = {'qty': 0.001, 'price': 45000, 'value': $45}
signal = {'symbol': 'BTCUSDT', 'action': 'BUY', 'confidence': 0.90}

# Build decisions
decisions = await meta._build_decisions([signal])

# Verify
assert len(decisions) == 0, "Significant position SHOULD block!"
```

**Current Result**: ✅ PASSES (signal rejected)  
**After Fix**: ✅ PASSES (signal rejected)

---

## Timeline

| Date | Event | Status |
|------|-------|--------|
| (TBD) | Bug discovered | ✅ DETECTED |
| (TBD) | Root cause identified | ✅ ANALYZED |
| (TBD) | Fix implemented | ⏳ PENDING |
| (TBD) | Tests pass | ⏳ PENDING |
| (TBD) | Deployed to production | ⏳ PENDING |
| (TBD) | Verified in live trading | ⏳ PENDING |

---

## Related Documents

1. **`DUST_AS_RECOVERABLE_CAPITAL_SUMMARY.md`** - Overview of dust recovery system
2. **`⚠️_CRITICAL_DUST_BLOCKING_BUG_ANALYSIS.md`** - Detailed bug analysis
3. **`🔧_DUST_BLOCKING_FIX_IMPLEMENTATION.md`** - Step-by-step fix guide

---

## Bottom Line

**Current State**: ❌ Dust blocks entry (deadlock)  
**Required State**: ✅ Dust allows entry (recovery)  
**Effort**: Minimal (call existing method)  
**Impact**: Critical (enables capital recovery)  
**Urgency**: 🚨 HIGH (blocks key safety mechanism)

The dust recovery system is sophisticated and well-designed. It just needs this one connection to work: call `_position_blocks_new_buy()` instead of checking `existing_qty > 0`.
