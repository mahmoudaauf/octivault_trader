# ⚠️ CRITICAL OPERATIONAL RULE: Dust Must NOT Block New Trades

## The Rule

```
INVARIANT:
┌────────────────────────────────────────────────────────────────┐
│  1. Dust must NOT block BUY signals                            │
│  2. Dust must NOT count toward position limits                 │
│  3. Dust must be REUSED when a signal appears                  │
│                                                                │
│  IF VIOLATED → System deadlocks (dust blocks entry forever)    │
└────────────────────────────────────────────────────────────────┘
```

## Why This Rule Exists

Your dust recovery system **requires dust positions to be reusable**:

### The Recovery Flow
```
Dust Position Created (price falls below minNotional)
    ↓
DustMonitor tracks it as RECOVERABLE
    ↓
Strong BUY signal appears (confidence ≥ 0.55)
    ↓
P0 DUST PROMOTION: Add capital to scale dust upward
    ↓
Dust graduates to viable position
    ↓
System recovers capital, avoids death spiral

BUT IF DUST BLOCKS NEW BUY:
    ↓
❌ New BUY signal rejected (position already exists)
❌ P0 Promotion can never execute
❌ Dust becomes permanent prison
❌ Capital never recovers
❌ SYSTEM DEADLOCK
```

---

## Current Implementation Status

### ✅ Partial Fix Exists

Your system **HAS attempted to fix this** in `_position_blocks_new_buy()`:

```python
# meta_controller.py, lines 1771-1809

async def _position_blocks_new_buy(self, symbol: str, existing_qty: float) -> Tuple[bool, float, float, str]:
    """
    Determine whether an existing position should block a new BUY 
    under one-position-per-symbol rules.
    """
    
    # ✅ FIX #1: Permanent dust (< $1.0) doesn't block
    permanent_dust_threshold = float(self._cfg("PERMANENT_DUST_USDT_THRESHOLD", 1.0))
    if pos_value > 0 and pos_value < permanent_dust_threshold:
        return False, pos_value, significant_floor, "permanent_dust_invisible"
    
    # ✅ FIX #2: Dust below significant floor doesn't block
    if pos_value > 0 and pos_value < significant_floor:
        return False, pos_value, significant_floor, "dust_below_significant_floor"
    
    # ✅ FIX #3: Unhealable dust doesn't block (marked as UNHEALABLE_LT_MIN_NOTIONAL)
    dust_unhealable = getattr(self.shared_state, "dust_unhealable", {}) or {}
    if str(dust_unhealable.get(sym, "") or "") == "UNHEALABLE_LT_MIN_NOTIONAL":
        return False, 0.0, 0.0, "unhealable_dust"
    
    return True, pos_value, significant_floor, "significant_position"
```

### 🔴 BUT: This Fix is NOT Being Used!

The critical problem is at **lines 9902-9930**:

```python
# meta_controller.py, lines 9902-9930

# 🚫 CRITICAL FIX: ONE_POSITION_PER_SYMBOL ENFORCEMENT
if existing_qty > 0:
    # ❌ WRONG: Checks raw quantity, not dust status!
    self.logger.info(
        "[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY: existing position blocks entry "
        "(qty=%.6f, ONE_POSITION_PER_SYMBOL rule enforced)",
        sym, existing_qty
    )
    # ... drops the signal immediately
    return
```

### The Bug

```python
# WRONG (current code, line 9910):
if existing_qty > 0:
    # ❌ Rejects if ANY quantity exists, including dust
    # ❌ Doesn't check if position is dust
    # ❌ Doesn't call _position_blocks_new_buy()
    skip_signal()

# CORRECT (what it should do):
blocks, pos_value, sig_floor, reason = await self._position_blocks_new_buy(sym, existing_qty)
if blocks:
    # ✅ Only reject if position is SIGNIFICANT
    # ✅ Allow dust to be reused/promoted
    skip_signal()
else:
    # ✅ Dust position: allow BUY to proceed
    # ✅ Enable P0 promotion path
    allow_signal()
```

---

## The Three Violations

### Violation #1: Dust BLOCKS BUY Signals

**Current behavior**:
```python
if existing_qty > 0:  # ANY quantity, including dust
    reject_buy_signal()  # ❌ WRONG
```

**Should be**:
```python
blocks = await self._position_blocks_new_buy(sym, existing_qty)
if blocks:  # Only if SIGNIFICANT
    reject_buy_signal()  # ✅ CORRECT
else:
    allow_buy_signal()  # Allow dust to be reused
```

### Violation #2: Dust COUNTS Toward Position Limit

The one-position-per-symbol rule treats dust the same as viable positions:
```
Position Limit Calculation:
├─ Significant Position A: $100 ✓ Counts
├─ Dust Position B: $5 ❌ Should NOT count
├─ Signal for C: Strong BUY ❌ Rejected (B blocks it)
└─ Result: Deadlock
```

**Fix location**: Same as Violation #1

### Violation #3: Dust Cannot Be REUSED

When a strong BUY signal appears and dust exists on the same symbol:

```python
# Current (BROKEN):
Dust ETHUSDT = $5
BUY Signal ETHUSDT = confidence 0.90
Result: ❌ Rejected by one-position-gate
Problem: Can't promote dust with new capital

# Should be (FIXED):
Dust ETHUSDT = $5
BUY Signal ETHUSDT = confidence 0.90
Result: ✅ Allowed through
Action: Scale dust with new capital
Outcome: Dust → $30 (viable), capital recovered
```

---

## Root Cause Analysis

### The Conflict

There are **two different position-locking strategies** that conflict:

**Strategy A** (`_position_blocks_new_buy()`, lines 1771+):
```
Intelligent position locking:
├─ Permanent dust < $1.0 → Allow new BUY
├─ Dust < significant floor → Allow new BUY  
├─ Unhealable dust → Allow new BUY
└─ Significant positions → Block new BUY
```

**Strategy B** (lines 9902+):
```
Crude position locking:
├─ Any position exists (qty > 0) → Block all new BUY
└─ Result: Dust gets treated same as viable positions
```

### Why This Happened

Looking at the code history:
1. `_position_blocks_new_buy()` was implemented to handle dust exceptions
2. But at line 9910, the **simpler check** `if existing_qty > 0` is used instead
3. The sophisticated check was never integrated into the decision gate
4. Result: Dust-aware logic is implemented but never called

---

## Where the Bug Lives

### File: `core/meta_controller.py`

**Location**: Lines 9902-9930

**Function**: `_build_decisions()` (main decision engine)

**Current code**:
```python
def _build_decisions(self):
    # ... build list of BUY signals ...
    
    for sig in buy_signals:
        sym = sig['symbol']
        existing_qty = float(self.shared_state.get_position_qty(sym) or 0.0)
        
        # ❌ BUG HERE: Lines 9902-9930
        if existing_qty > 0:
            # Rejects ALL positions, including dust
            # Should use _position_blocks_new_buy() instead
            self.logger.info(
                "[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY: "
                "existing position blocks entry (qty=%.6f)",
                sym, existing_qty
            )
            # Records why trade was skipped
            await self._record_why_no_trade(...)
            continue  # ❌ Skip this signal
```

**Related code**:
```python
# Lines 1771-1809: _position_blocks_new_buy() exists but is NEVER CALLED
# This method has all the dust-aware logic but isn't used
```

---

## Impact on Dust Recovery

### Scenario: Dust Promotion Deadlock

```
Timeline:
────────

T=0:  ETHUSDT position created
      ├─ Buy $10 ETHUSDT
      ├─ Price drops to $4 (below $10 minNotional)
      └─ Position becomes DUST (tracked by DustMonitor)

T+2h: Market condition improves, strong BUY signal appears
      ├─ Confidence: 0.90 (strong)
      ├─ Signal intent: Scale dust from $4 → $30
      ├─ Available capital: $25
      └─ Should execute? ✅ YES (dust recovery opportunity)

      CURRENT BEHAVIOR (BROKEN):
      ├─ Check: position exists? YES (qty > 0)
      ├─ Action: REJECT signal at ONE_POSITION_GATE
      ├─ Reason: "ONE_POSITION_PER_SYMBOL rule enforced"
      └─ Result: ❌ DEADLOCK - Dust can never be promoted

      CORRECT BEHAVIOR (FIXED):
      ├─ Check: _position_blocks_new_buy(ETHUSDT, 0.00133)
      ├─ Price: $4, so value = $4 < $10 (significant floor)
      ├─ Return: (False, $4, $10, "dust_below_significant_floor")
      ├─ Action: ALLOW signal
      └─ Result: ✅ P0 PROMOTION executes, dust recovered
```

### Capital Impact

**With Bug** (dust blocks):
```
Day 1: Capital $100, position ETHUSDT = $4 (dust)
       Signal to promote dust appears
       ❌ Rejected by ONE_POSITION_GATE
       Capital remains $100 (dust unrecovered)
       
Day 2: Capital $95 (market loss)
       Position ETHUSDT still $4 (dust)
       ❌ Dust still blocks any new entries
       System slowly starves

Result: Dust becomes permanent prison, capital degradation
```

**Without Bug** (dust allows entry):
```
Day 1: Capital $100, position ETHUSDT = $4 (dust)
       Signal to promote dust appears (confidence 0.90)
       ✅ Allowed through
       P0 DUST PROMOTION: Add $25 to scale dust
       Position ETHUSDT becomes $29 (viable)
       
Day 2: Position ETHUSDT = $29 (now tradeable)
       Can be exited or scaled further
       
Result: Dust recovered, capital system stays healthy
```

---

## Fix Required

### Step 1: Use Existing Dust-Aware Logic

Replace lines 9902-9930 with:

```python
# ✅ FIXED: Use dust-aware position blocking logic

# Get detailed block status (handles dust exceptions)
blocks, pos_value, sig_floor, reason = await self._position_blocks_new_buy(sym, existing_qty)

if blocks:
    # Position is SIGNIFICANT and blocks entry
    self.logger.info(
        "[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY: "
        "existing SIGNIFICANT position blocks entry "
        "(value=%.2f >= floor=%.2f, reason=%s)",
        sym, pos_value, sig_floor, reason
    )
    await self._record_why_no_trade(
        sym, "POSITION_ALREADY_OPEN",
        f"Significant position blocks entry (value=${pos_value:.2f}, reason={reason})",
        side="BUY", signal=sig
    )
    continue  # Skip this signal

# ✅ If we reach here, position is either:
#    - Dust (< $1.0 permanent_dust_threshold), or
#    - Below significant floor, or
#    - Unhealable dust
# All of these are ALLOWED to be reused/promoted
# Do NOT skip this signal
```

### Step 2: Handle Reentry Flags Properly

Keep the existing reentry logic but make it **dust-aware**:

```python
# Check for reentry permissions (scale-in, promotion, etc.)
if has_position and not is_permanent_dust(sym):
    # Non-permanent dust can be promoted
    sig["_allow_reentry"] = True

if has_open and allow_scale_in:
    sig["_allow_reentry"] = True

# Important: Dust positions should have _allow_reentry=True
# so they can be scaled/promoted without blocking
```

### Step 3: Add Guard Checks

Ensure dust positions aren't being treated as normal positions elsewhere:

```python
# In execute_buy() or similar:
if not await self._position_blocks_new_buy(sym, existing_qty):
    # Dust position - use different handling
    # (promotion logic, not normal BUY)
```

---

## Verification Checklist

After applying the fix, verify:

- [ ] Dust positions (< significant floor) do NOT block BUY signals
- [ ] `_position_blocks_new_buy()` is called at decision gate
- [ ] Return value is checked: `if blocks:` (not `if existing_qty > 0:`)
- [ ] P0 dust promotion can execute when signal + dust exist
- [ ] Non-permanent dust can be reused when signals appear
- [ ] Significant positions still block (normal rule intact)
- [ ] Unhealable dust is properly exempted
- [ ] Permanent dust (< $1.0) is exempted
- [ ] Test: Create dust → trigger BUY signal → verify allowed through

---

## Test Case: Dust Promotion Should Work

```python
async def test_dust_position_allows_new_buy():
    """
    REQUIREMENT: Dust positions must not block new BUY signals.
    This is critical for P0 DUST PROMOTION to function.
    """
    
    # Setup: Position becomes dust
    meta.shared_state.positions['ETHUSDT'] = {
        'quantity': 0.00133,
        'price': 3.00,  # $4 notional < $10 min
        'status': 'DUST_LOCKED'
    }
    
    # Signal: Strong BUY appears
    signal = {
        'symbol': 'ETHUSDT',
        'action': 'BUY',
        'confidence': 0.90  # Strong
    }
    
    # Build decisions
    decisions = await meta._build_decisions([signal], ...)
    
    # VERIFY: Signal is NOT rejected at ONE_POSITION_GATE
    assert len(decisions) > 0, "Dust should not block BUY signal!"
    assert decisions[0]['symbol'] == 'ETHUSDT'
    assert decisions[0]['action'] == 'BUY'
    
    # VERIFY: P0 promotion can execute
    can_promote = await meta._check_p0_dust_promotion()
    assert can_promote == True, "P0 promotion should be possible!"
```

---

## Summary

**The Critical Operational Rule is Violated:**
- ❌ Dust DOES block BUY signals (bug at line 9902-9930)
- ❌ Dust DOES count toward position limits (same bug)
- ❌ Dust CANNOT be reused (consequence of bug)

**The Fix Already Exists:**
- ✅ `_position_blocks_new_buy()` (lines 1771-1809) has correct logic
- ✅ It correctly handles dust exceptions
- ✅ It's just never called from the decision gate

**What Needs to Happen:**
1. Replace crude `if existing_qty > 0:` check with dust-aware `_position_blocks_new_buy()` call
2. Only skip signals if `blocks == True` (not on quantity existence)
3. Verify dust positions can now be promoted when signals appear
4. Test: Dust should graduate to viable positions, recovering capital

**Impact of Not Fixing:**
- System deadlocks when dust is created
- P0 Dust Promotion escape hatch never works
- Capital floor crisis cannot be escaped
- Dust becomes permanent (never recovers)

**Once Fixed:**
- Dust positions are reusable
- P0 Promotion can execute
- Dust recovery pathway is open
- Capital floor escape is possible
