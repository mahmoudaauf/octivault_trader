# 🔧 FIX IMPLEMENTATION: Dust Must Not Block New Trades

## Quick Reference

```
FILE: core/meta_controller.py
LINES: 9902-9930
FUNCTION: _build_decisions()

CURRENT (BROKEN):
    if existing_qty > 0:
        skip_signal()  # ❌ Dust blocks even if below min

FIXED:
    blocks = await self._position_blocks_new_buy(sym, existing_qty)
    if blocks:
        skip_signal()  # ✅ Only skip if position is significant
    # else: allow signal through (dust can be reused)
```

---

## The Fix in Detail

### Current Broken Code (meta_controller.py, lines 9895-9935)

```python
existing_qty = float(self.shared_state.get_position_qty(sym) or 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# 🚫 CRITICAL FIX: ONE_POSITION_PER_SYMBOL ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════
# Professional rule: If position exists for symbol, REJECT all new BUY signals
# INVARIANT: max_exposure_per_symbol = 1 position (no stacking, no scaling, no accumulation)
#
# This prevents risk doubling and enforces strict position isolation.
# ═══════════════════════════════════════════════════════════════════════════════

if existing_qty > 0:  # ❌ WRONG: Checks raw quantity, includes dust!
    # Position exists - REJECT BUY signal regardless of any flag/exception
    self.logger.info(
        "[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY: existing position blocks entry "
        "(qty=%.6f, ONE_POSITION_PER_SYMBOL rule enforced)",
        sym, existing_qty
    )
    self.logger.warning(
        "[Meta:GATE_DROP_ONE_POSITION] %s BUY dropped at ONE_POSITION_PER_SYMBOL gate (qty=%.6f)",
        sym, existing_qty
    )
    self.logger.warning(
        "[WHY_NO_TRADE] symbol=%s reason=POSITION_ALREADY_OPEN details=ONE_POSITION_PER_SYMBOL "
        "qty=%.6f", sym, existing_qty
    )
    await self._record_why_no_trade(
        sym,
        "POSITION_ALREADY_OPEN",
        f"ONE_POSITION_PER_SYMBOL qty={existing_qty:.6f}",
        side="BUY",
        signal=sig,
    )
    continue  # ❌ Skips signal, including dust recovery paths
```

### Fixed Code (Replace with this)

```python
existing_qty = float(self.shared_state.get_position_qty(sym) or 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# ✅ DUST-AWARE: ONE_POSITION_PER_SYMBOL ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════
# Intelligent position locking rule:
# - Significant positions BLOCK new BUY signals (prevent risk doubling)
# - Dust positions ALLOW new BUY signals (enable dust promotion/reuse)
# - Unhealable dust ALLOWS new BUY (prevents deadlock)
#
# This enables:
# 1. P0 DUST PROMOTION (scale dust with freed capital)
# 2. Dust recovery (dust → viable when signal appears)
# 3. Normal position isolation (significant blocks entry)
# ═══════════════════════════════════════════════════════════════════════════════

if existing_qty > 0:
    # ✅ FIXED: Use dust-aware blocking logic instead of crude qty check
    blocks, pos_value, sig_floor, reason = await self._position_blocks_new_buy(sym, existing_qty)
    
    if blocks:
        # Position is SIGNIFICANT (value >= floor) - blocks entry
        self.logger.info(
            "[Meta:ONE_POSITION_GATE] 🚫 Skipping %s BUY: existing SIGNIFICANT position blocks entry "
            "(value=%.2f >= floor=%.2f, reason=%s, ONE_POSITION_PER_SYMBOL enforced)",
            sym, pos_value, sig_floor, reason
        )
        self.logger.warning(
            "[Meta:GATE_DROP_ONE_POSITION] %s BUY dropped at ONE_POSITION_PER_SYMBOL gate "
            "(value=%.2f, reason=%s)",
            sym, pos_value, reason
        )
        self.logger.warning(
            "[WHY_NO_TRADE] symbol=%s reason=POSITION_ALREADY_OPEN details=ONE_POSITION_PER_SYMBOL "
            "value=%.2f reason=%s", sym, pos_value, reason
        )
        await self._record_why_no_trade(
            sym,
            "POSITION_ALREADY_OPEN",
            f"Significant position blocks entry (value=${pos_value:.2f}, reason={reason})",
            side="BUY",
            signal=sig,
        )
        continue  # ✅ Skip only SIGNIFICANT positions
    else:
        # Position is DUST or UNHEALABLE_DUST - ALLOW signal through
        # This enables:
        # - P0 DUST PROMOTION when strong signals exist
        # - Dust recovery (reuse dust with new capital)
        # - Bootstrap entry (dust doesn't block new entries)
        self.logger.info(
            "[Meta:DUST_REENTRY_ALLOWED] ✅ Allowing %s BUY: existing dust position permits entry "
            "(value=%.2f < floor=%.2f, reason=%s)",
            sym, pos_value, sig_floor, reason
        )
        # Continue processing signal - don't skip
```

---

## Why This Fix Works

### Before (Broken)
```
Check: if existing_qty > 0
       ├─ ETHUSDT has qty 0.00133 → TRUE
       └─ Signal REJECTED immediately
       
Result: Dust blocks BUY
        ❌ P0 promotion fails
        ❌ Dust never recovers
        ❌ Deadlock
```

### After (Fixed)
```
Check: await self._position_blocks_new_buy(sym, existing_qty)
       ├─ Check if position is SIGNIFICANT
       ├─ Check if it's PERMANENT_DUST (< $1.0)
       ├─ Check if it's UNHEALABLE_DUST
       └─ Return: (blocks=False, value=$4, reason="dust_below_significant_floor")

if blocks == False:
       ├─ Signal is ALLOWED through
       └─ Result: ✅ P0 promotion can execute
       
Result: Dust doesn't block BUY
        ✅ P0 promotion works
        ✅ Dust recovers
        ✅ System escapes deadlock
```

---

## The Existing Dust-Aware Logic (Already Implemented)

Your code at **lines 1771-1809** already has the correct logic:

```python
async def _position_blocks_new_buy(self, symbol: str, existing_qty: float) -> Tuple[bool, float, float, str]:
    """
    Determine whether an existing position should block a new BUY 
    under one-position-per-symbol rules.
    
    Returns: (blocks, position_value, significant_floor, reason)
    
    ✅ Returns False for:
       - Permanent dust (< $1.0)
       - Dust below significant floor
       - Unhealable dust
    
    ✅ Returns True for:
       - Significant positions (value >= floor)
    """
    
    sym = self._normalize_symbol(symbol)
    qty = float(existing_qty or 0.0)
    if qty <= 0:
        return False, 0.0, 0.0, "no_position"

    # ✅ Check if unhealable (marked as UNHEALABLE_LT_MIN_NOTIONAL)
    dust_unhealable = getattr(self.shared_state, "dust_unhealable", {}) or {}
    if str(dust_unhealable.get(sym, "") or "") == "UNHEALABLE_LT_MIN_NOTIONAL":
        return False, 0.0, 0.0, "unhealable_dust"

    significant_floor = await self._canonical_significant_floor(sym)

    # ... get price ...
    pos_value = qty * price if price > 0 else 0.0
    
    # ✅ Permanent dust (< $1.0) doesn't block
    permanent_dust_threshold = float(self._cfg("PERMANENT_DUST_USDT_THRESHOLD", 1.0) or 1.0)
    if pos_value > 0 and pos_value < permanent_dust_threshold:
        return False, pos_value, significant_floor, "permanent_dust_invisible"
    
    # ✅ Dust below significant floor doesn't block
    if pos_value > 0 and pos_value < significant_floor:
        return False, pos_value, significant_floor, "dust_below_significant_floor"
    
    # ✅ Only return True for significant positions
    return True, pos_value, significant_floor, "significant_position"
```

**The fix is just connecting the dots:** Call this method at the decision gate instead of using the crude `if existing_qty > 0` check.

---

## Integration Points

### Location 1: Main Decision Gate (Primary Fix)

**File**: `core/meta_controller.py`  
**Lines**: 9902-9930  
**Function**: `_build_decisions()`

Replace the crude check with the dust-aware version shown above.

### Location 2: Verify Reentry Flags

**File**: `core/meta_controller.py`  
**Lines**: 9880-9895 (before the gate)

Ensure reentry flags are set correctly for dust:

```python
# Before the ONE_POSITION_GATE check, ensure dust gets reentry permission:

if existing_qty > 0:
    has_open = await self._is_position_open(sym)
    is_permanent_dust = await self._is_permanent_dust(sym)
    
    # ✅ Non-permanent dust can be promoted/reused
    if not is_permanent_dust:
        sig["_allow_reentry"] = True
    
    if has_open and allow_scale_in:
        sig["_allow_reentry"] = True
```

### Location 3: Check Downstream Handlers

Any code that **uses** the `_allow_reentry` flag:

```python
# In execute_buy() or similar, if _allow_reentry is True:
if sig.get("_allow_reentry"):
    # Could be dust promotion - use different handling
    can_promote = await self._check_p0_dust_promotion()
    if can_promote:
        # Execute P0 promotion instead of normal BUY
        return await self._execute_p0_dust_promotion(sym, ...)
    else:
        # Fall through to normal BUY
        return await execute_buy(sym, ...)
```

---

## Testing the Fix

### Test 1: Dust Doesn't Block BUY

```python
async def test_dust_allows_buy_signal():
    """Dust positions should not block BUY signals."""
    
    # Setup: Create dust position
    meta.shared_state.positions['ETHUSDT'] = {
        'quantity': 0.00133,
        'price': 3.00,  # Value = $3.99 < $10 floor
        'status': 'DUST_LOCKED'
    }
    
    # BUY signal appears
    signal = {
        'symbol': 'ETHUSDT',
        'action': 'BUY',
        'confidence': 0.90
    }
    
    # Build decisions
    decisions = await meta._build_decisions([signal])
    
    # VERIFY: Signal is NOT rejected
    assert len(decisions) > 0, "Dust should not block BUY!"
    assert decisions[0]['symbol'] == 'ETHUSDT'
    assert decisions[0]['action'] == 'BUY'
```

### Test 2: Significant Position Still Blocks

```python
async def test_significant_position_blocks_buy():
    """Significant positions SHOULD block new BUY signals."""
    
    # Setup: Significant position
    meta.shared_state.positions['BTCUSDT'] = {
        'quantity': 0.001,
        'price': 45000.00,  # Value = $45 > $10 floor
        'status': 'OPEN'
    }
    
    # BUY signal appears
    signal = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'confidence': 0.90
    }
    
    # Build decisions
    decisions = await meta._build_decisions([signal])
    
    # VERIFY: Signal IS rejected
    assert len(decisions) == 0, "Significant position should block BUY!"
```

### Test 3: P0 Promotion Can Execute

```python
async def test_p0_dust_promotion_executes():
    """With dust unblocked, P0 promotion should execute."""
    
    # Setup: Dust position + strong signal
    meta.shared_state.positions['ETHUSDT'] = {
        'quantity': 0.00133,
        'price': 3.00,  # Dust
        'status': 'DUST_LOCKED'
    }
    
    signal = {
        'symbol': 'ETHUSDT',
        'action': 'BUY',
        'confidence': 0.90  # Strong
    }
    
    # Build decisions
    decisions = await meta._build_decisions([signal])
    
    # VERIFY: Signal goes through
    assert len(decisions) > 0
    decision = decisions[0]
    
    # VERIFY: Could be P0 promotion (or normal BUY for reentry)
    assert decision['symbol'] == 'ETHUSDT'
    assert decision['action'] == 'BUY'
    assert decision.get('_allow_reentry') == True
```

---

## Verification Checklist

After implementing the fix:

- [ ] Code change: Replace `if existing_qty > 0:` with `_position_blocks_new_buy()` call
- [ ] Logic change: Only skip if `blocks == True` (not on qty existence)
- [ ] Logging: Updated to show reason (dust_below_floor, etc.)
- [ ] Reentry flags: Dust positions get `_allow_reentry = True`
- [ ] P0 Promotion: Can now execute when signal + dust exist
- [ ] Test 1 passes: Dust allows BUY through
- [ ] Test 2 passes: Significant position still blocks
- [ ] Test 3 passes: P0 promotion can execute
- [ ] Integration: No downstream code breaks
- [ ] Edge case: Permanent dust (< $1) is handled
- [ ] Edge case: Unhealable dust is exempted

---

## Rollback Plan

If issues arise, revert to:

```python
if existing_qty > 0:
    # Temporary fallback
    await self._record_why_no_trade(...)
    continue
```

But this will recreate the deadlock. Better to fix than fallback.

---

## Related Code to Review

Before/after the fix, review these areas for consistency:

1. **Dust checking in execute_buy()** - Make sure it allows dust to be reused
2. **P0 promotion logic** - Should execute when signal + dust exist (after fix)
3. **Accumulation resolution** - Should still work for rejected trades
4. **Position status marking** - Dust should be marked correctly in shared_state
5. **DustMonitor** - Health tracking should continue to work
6. **Permanent dust handling** - Should exclude from all metrics

---

## Summary

**The Fix:**
1. Replace `if existing_qty > 0:` with `await self._position_blocks_new_buy(sym, qty)`
2. Only skip signal if `blocks == True`
3. Allow dust to proceed to P0 promotion logic

**Why It Works:**
- Dust-aware logic already exists (`_position_blocks_new_buy()`)
- Just needs to be called at the decision gate
- Returns False for dust (allowing entry)
- Returns True for significant (blocking entry)

**Result:**
- ✅ Dust doesn't block BUY signals
- ✅ Dust doesn't count toward position limits
- ✅ Dust can be reused for recovery
- ✅ P0 Dust Promotion can execute
- ✅ Capital floor escape is possible

**Impact:**
- **Without fix**: System deadlocks, dust never recovers, capital starves
- **With fix**: Dust recovery pathway opens, system can escape capital floor crises
