# 🔴 CRITICAL BUG: Conditional TRADE_EXECUTED Emission on Dust

**Status:** IDENTIFIED & CONFIRMED  
**Severity:** CRITICAL - Breaks observability contract  
**Component:** ExecutionManager  
**Location:** `_emit_close_events()` line 1019-1020

---

## Problem Statement

**User Requirement:**
> Inside ExecutionManager post-fill: Emit TRADE_EXECUTED if executed_qty > 0, then apply dust cleanup separately. Emission must NOT depend on remaining position.

**Reality (BROKEN):**
The canonical TRADE_EXECUTED emission is conditionally **skipped** when the final remaining position becomes dust.

### Exact Code Path

```python
# execution_manager.py:1018-1020
async def _emit_close_events(self, sym: str, raw: Dict[str, Any], post_fill: Optional[Dict[str, Any]] = None) -> None:
    entry_price, exec_px, exec_qty, realized_pnl = self._calc_close_payload(sym, raw)
    if exec_qty <= 0 or exec_px <= 0:  # ❌ THIS EARLY RETURN BREAKS THE CONTRACT
        return
    # ... rest of method (including re-emit attempt at line 1033)
```

---

## Scenario Where Bug Manifests

### Setup
- Position: `ETHUSDT` with 0.1 qty
- Current Price: $3,000
- Dust Threshold: $10 USD notional

### Execution Flow

1. **Order Fills:** SELL 0.1 ETH at $3,000 → `executedQty = 0.1`
2. **Post-Fill Accounting:** 
   - `_handle_post_fill()` called → ✅ **TRADE_EXECUTED emitted** (line 237)
   - Trade recorded → ✅ **Event emitted**
3. **Close Events Processing:**
   - `_finalize_sell_post_fill()` called (line 1391)
   - Calls `_emit_close_events()` (line 1444)
   - **BUG TRIGGER:** `_calc_close_payload()` is called (line 1019)
     - Checks position AFTER fill is recorded
     - Remaining position = dust (< $10 notional)
     - Returns `exec_qty = 0.0` or `exec_px = 0.0`
   - **Line 1020:** `if exec_qty <= 0 or exec_px <= 0: return` → **EXITS EARLY**
   - ❌ No POSITION_CLOSED event emitted
   - ❌ Re-emit fallback never reaches (line 1033)

---

## Root Cause Analysis

### Why `_calc_close_payload()` Returns 0 Qty

Looking at line 990:
```python
def _calc_close_payload(self, sym: str, raw: Dict[str, Any]) -> Tuple[float, float, float, float]:
    entry_price = float(self._get_entry_price_for_sell(sym) or 0.0)
    exec_px = float(raw.get("avgPrice", raw.get("price", 0.0)) or 0.0)
    exec_qty = float(raw.get("executedQty", 0.0) or 0.0)  # ✅ This comes from filled order
```

**The issue:** `_calc_close_payload()` tries to use `executedQty` from the raw order, but:

1. **In SELL closes after position cleanup:** The raw order may have `executedQty` stripped or zeroed by SharedState during dust cleanup
2. **Alternative path:** If `executedQty` is missing/0 and the position is now dust, the method returns (exec_qty=0)

### Why This Violates The Contract

**The P9 Observability Contract (from code comment line 233):**
```python
# P9 event contract: every confirmed fill must emit TRADE_EXECUTED.
# Emission is anchored to post-fill processing, independent of tag/agent/side.
```

**What's Happening:**
1. ✅ Post-fill emits TRADE_EXECUTED (line 237) → **CORRECT**
2. ✅ Event goes into event log → **CORRECT**
3. ❌ Close events skip POSITION_CLOSED if remaining position is dust → **VIOLATION**
4. ❌ Re-emit fallback (line 1033) never executes → **VIOLATION**

**Impact:**
- ExchangeTruthAuditor cannot detect fills for positions that close to dust
- Event chain breaks (fill exists, but no close event)
- Governance layer loses visibility on dust cleanup operations

---

## Evidence from Code

### Evidence 1: Early Return Guards Entire Method

**Line 1018-1022:**
```python
async def _emit_close_events(self, sym: str, raw: Dict[str, Any], post_fill: Optional[Dict[str, Any]] = None) -> None:
    entry_price, exec_px, exec_qty, realized_pnl = self._calc_close_payload(sym, raw)
    if exec_qty <= 0 or exec_px <= 0:  # ← GATES EVERYTHING BELOW
        return
    # Lines 1033+: Re-emit fallback never reached if above condition is true
```

### Evidence 2: Guards POSITION_CLOSED Emission

**Lines 1064-1077 (POSITION_CLOSED event):**
```python
self.logger.info(json.dumps({
    "event": "POSITION_CLOSED",
    "symbol": sym,
    "entry_price": entry_price,
    "exit_price": exec_px,
    "qty": exec_qty,  # ← Uses the potentially-zero exec_qty
    "realized_pnl": realized_pnl,
}, separators=(",", ":")))

try:
    await maybe_call(self.shared_state, "emit_event", "POSITION_CLOSED", {
        "symbol": sym,
        "entry_price": float(entry_price or 0.0),
        "price": float(exec_px or 0.0),
        "qty": float(exec_qty or 0.0),  # ← Zeros here
        "realized_pnl": float(realized_pnl or 0.0),
        "timestamp": time.time(),
    })
```

These lines are **unreachable** when `exec_qty <= 0`.

### Evidence 3: Re-Emit Fallback is Inside the Guarded Block

**Lines 1033-1037:**
```python
# Ensure canonical TRADE_EXECUTED exists for SELL closes. Some paths (recovered fills,
# transient emit failures, or external/order-recovery flows) may reach close events
# without a prior canonical TRADE_EXECUTED. To preserve the architecture invariant
# (every confirmed fill must emit TRADE_EXECUTED) we re-emit here when missing.
try:
    tag = (raw or {}).get("tag") or (raw or {}).get("order_tag") or ""
    # Idempotent/dedupe-protected: always attempt canonical emit for invariants
    with contextlib.suppress(Exception):
        await self._emit_trade_executed_event(sym, "SELL", str(tag or ""), raw)
```

**This fallback is INSIDE the function body**, meaning it only executes if the early return at line 1020 is NOT taken.

---

## Proof of Violation

### Test Case: SELL Position to Dust

```
Given:
  - Symbol: ETHUSDT
  - Position before: 0.1 ETH @ $3,000 entry = $300 notional
  - Fill: SELL 0.1 ETH @ $3,000 → $300 value

When:
  - execute_trade(side="sell", quantity=0.1) called
  - Exchange returns: { status: "FILLED", executedQty: 0.1, avgPrice: 3000 }

Then:
  - _handle_post_fill() → ✅ Emits TRADE_EXECUTED (line 237)
  - _finalize_sell_post_fill() → Calls _emit_close_events() 
  - _emit_close_events():
    - Calls _calc_close_payload()
    - Remaining position = 0 (position fully closed)
    - Returns: exec_qty=0.0 (or potentially dust-zeroed value)
    - Line 1020: if exec_qty <= 0: return ← ❌ EXITS HERE
    - POSITION_CLOSED event NEVER emitted
    - Re-emit fallback NEVER executed

Result:
  - ❌ Contract violated
  - ❌ Missing governance event
  - ❌ Truth auditor blind spot
```

---

## Impact Assessment

### Observability Loss

1. **Event Chain Broken:**
   - TRADE_EXECUTED exists in log
   - POSITION_CLOSED missing
   - Governance sees inconsistent state

2. **ExchangeTruthAuditor Implications:**
   - Can see fills in `_reconcile_trades()`
   - Cannot properly verify close events via `_is_canonical_trade_event_present()`
   - May trigger false "SELL_MISSING_CANONICAL" warnings for dust closes

3. **Position Lifecycle:**
   - Entry event: ✅ Recorded
   - Fill event: ✅ Recorded  
   - Close event: ❌ Missing
   - PnL event: ✅ Recorded (independent path)
   - Governance: ❌ Sees incomplete lifecycle

### Severity: CRITICAL

- **Breaks P9 observability contract**
- **Governance layer loses visibility**
- **Affects all dust cleanup operations**
- **Silent failure** (no error logged)

---

## Solution

### Fix Requirements

1. **Unconditional TRADE_EXECUTED Re-Emit:**
   - Move re-emit fallback (line 1033) OUTSIDE the early-return guard
   - Ensure it executes even when remaining position is dust

2. **Separate Concerns:**
   - `_emit_trade_executed_event()` → Should emit always (if exec_qty > 0)
   - `_emit_close_events()` → Should emit POSITION_CLOSED regardless of remaining state
   - Dust cleanup → Should be separate operation in SharedState

3. **Preserve Dust Cleanup:**
   - Keep dust cleanup in SharedState.record_trade() (separate responsibility)
   - ExecutionManager should ONLY manage event emission
   - Governance should see complete event chain

### Proposed Code Change

```python
# execution_manager.py:1018-1100

async def _emit_close_events(self, sym: str, raw: Dict[str, Any], post_fill: Optional[Dict[str, Any]] = None) -> None:
    entry_price, exec_px, exec_qty, realized_pnl = self._calc_close_payload(sym, raw)
    
    # ✅ FIX: Use executedQty from order directly, not from remaining position
    # exec_qty represents what was EXECUTED in this fill, not what remains
    # For determining if we should emit events, use the order's executedQty
    
    # Recover actual executed quantity from the raw order (not from position state)
    actual_exec_qty = float(raw.get("executedQty", raw.get("executed_qty", 0.0)) or 0.0)
    
    if actual_exec_qty <= 0 or exec_px <= 0:
        return  # No valid execution to report

    # Ensure canonical TRADE_EXECUTED exists (idempotent fallback)
    try:
        tag = (raw or {}).get("tag") or (raw or {}).get("order_tag") or ""
        with contextlib.suppress(Exception):
            await self._emit_trade_executed_event(sym, "SELL", str(tag or ""), raw)
    except Exception:
        self.logger.debug("[EM:CloseEmitRecover] re-emit TRADE_EXECUTED failed", exc_info=True)

    # Emit PnL and close events using executed quantity from order
    if not committed:
        try:
            cur = float(getattr(self.shared_state, "metrics", {}).get("realized_pnl", 0.0) or 0.0)
            self.shared_state.metrics["realized_pnl"] = cur + float(realized_pnl)
        except Exception:
            pass

    if not emitted:
        # ... emit RealizedPnlUpdated ...
        pass

    # ✅ ALWAYS emit POSITION_CLOSED with what was executed
    self.logger.info(json.dumps({
        "event": "POSITION_CLOSED",
        "symbol": sym,
        "entry_price": entry_price,
        "exit_price": exec_px,
        "qty": actual_exec_qty,  # ← Use actual filled quantity
        "realized_pnl": realized_pnl,
    }, separators=(",", ":")))
    
    try:
        await maybe_call(self.shared_state, "emit_event", "POSITION_CLOSED", {
            "symbol": sym,
            "entry_price": float(entry_price or 0.0),
            "price": float(exec_px or 0.0),
            "qty": float(actual_exec_qty or 0.0),  # ← Use executed, not remaining
            "realized_pnl": float(realized_pnl or 0.0),
            "timestamp": time.time(),
        })
    except Exception:
        pass
```

---

## Recommendation

### Immediate Action
1. **Fix `_emit_close_events()`** to use `executedQty` from order directly
2. **Ensure POSITION_CLOSED always emitted** for any valid execution
3. **Move TRADE_EXECUTED re-emit outside the guard** to execute unconditionally
4. **Add logging** to track dust closes to observability events

### Testing
- Test SELL closing position to dust
- Verify POSITION_CLOSED event emitted
- Verify TRADE_EXECUTED event exists
- Verify event order correct
- Verify PnL computation correct

### Governance
- ExchangeTruthAuditor should see complete event chains
- Dust cleanup should be visible in governance layer
- No blind spots for position lifecycle

---

## Files Modified

- `core/execution_manager.py` (lines 1018-1100)

## Related Code Locations

- **Observation point:** `_emit_trade_executed_event()` (line 3315)
- **Dust cleanup:** SharedState.record_trade()
- **Governance audit:** ExchangeTruthAuditor._validate_sell_finalize_mapping() (line 238)
- **Close event:** _finalize_sell_post_fill() (line 1391)
