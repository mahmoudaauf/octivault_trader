# Post-Fill Emission Contract - ExecutionManager

**Date:** February 24, 2026  
**Status:** ✅ VERIFIED - Emission is unconditional on remaining position

---

## Executive Summary

The ExecutionManager `_handle_post_fill()` method **ALWAYS emits TRADE_EXECUTED** for any fill with `executed_qty > 0`, **independent of remaining position value or dust floor checks**.

Dust cleanup is a **separate, downstream operation** in SharedState, not a blocker to emission.

---

## Code Path Analysis

### 1. TRADE_EXECUTED Emission (Unconditional)

**Location:** `core/execution_manager.py` lines 190-420  
**Method:** `async def _handle_post_fill(...)`

```python
# Line 214-223: Early return ONLY if exec_qty <= 0
exec_qty = self._safe_float(order.get("executedQty") or order.get("executed_qty"), 0.0)
if exec_qty <= 0:
    return {
        "delta": delta_f,
        "realized_committed": realized_committed,
        "emitted": emitted,
        "trade_event_emitted": trade_event_emitted,
    }

# Line 227: Resolve price (validation only)
price = self._resolve_post_fill_price(order, exec_qty)
if price > 0:
    order.setdefault("avgPrice", float(price))
    if self._safe_float(order.get("price"), 0.0) <= 0:
        order["price"] = float(price)

# 🔥 CRITICAL: Line 234-240 - UNCONDITIONAL EMISSION ✅
# P9 event contract: every confirmed fill must emit TRADE_EXECUTED.
# Emission is anchored to post-fill processing, independent of tag/agent/side.
trade_event_emitted = bool(
    await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
)
self.logger.debug(f"[DEBUG] Trade executed event emitted: symbol={sym} side={side_u} tag={tag} emitted={trade_event_emitted}")

# Line 244-251: Validate price (AFTER emission, won't block it)
if price <= 0:
    self.logger.warning("[POST_FILL_PRICE_MISSING] ...")
    return {
        "delta": delta_f,
        "realized_committed": realized_committed,
        "emitted": emitted,
        "trade_event_emitted": trade_event_emitted,  # ✅ Already set above
    }

# Rest of method: PnL computation, tracking, etc. (all AFTER emission)
```

**Key invariant:**
- ✅ If `executed_qty > 0` → **TRADE_EXECUTED is emitted**
- ❌ No floor checks, no remaining position checks, no dust thresholds gate the emission
- ✅ Emission is **the first observable action** after price validation

---

### 2. Dust Cleanup (Separate Responsibility)

**Dust threshold checks happen in:**
1. **SharedState.record_trade()** - Position manager handles dust on its own
2. **TP/SL engine** - May skip if remaining position is insignificant
3. **Balance reconciliation** - ExchangeTruthAuditor closes phantom positions

**NOT in ExecutionManager post-fill** - EM never blocks emission based on dust.

---

### 3. Call Chain: Emission → PnL → Finalization

```
execute_trade() [SELL fill]
    ↓
_place_market_order_qty() [exchange order]
    ↓
Order filled (executedQty > 0)
    ↓
_ensure_post_fill_handled()
    ↓
_handle_post_fill()
    ├─ ✅ emit TRADE_EXECUTED (unconditional, Line 241)
    ├─ Compute realized PnL delta
    ├─ Update metrics
    ├─ Emit RealizedPnlUpdated
    └─ Return {delta, emitted, trade_event_emitted}
    ↓
_finalize_sell_post_fill()
    ├─ Ensure post-fill was run (call _ensure_post_fill_handled if needed)
    ├─ Record exit bookkeeping
    ├─ Emit POSITION_CLOSED events
    ├─ Sync remaining position in SharedState
    └─ Mark order as finalized
    ↓
SharedState.record_trade()
    ├─ Update positions (may mark as dust)
    ├─ Record in trade_history
    └─ Maybe arm TP/SL (if remaining position significant)
```

**Observation:** Dust cleanup happens in SharedState, AFTER ExecutionManager has already emitted.

---

## Verification: No Conditional Emission

### Scenario 1: Partial SELL that leaves dust

```
Position: 0.01 BTC (worth $500)
SELL order: 0.009 BTC at $50,000 = $450
Remaining: 0.001 BTC (worth $50 - below dust floor)
```

**Flow:**
1. ✅ `_handle_post_fill()` called with `executedQty=0.009`
2. ✅ **TRADE_EXECUTED emitted** (no floor check blocks it)
3. ✅ PnL computed: (0.009 × $50k - fees)
4. ✅ `_finalize_sell_post_fill()` completes
5. ✅ SharedState marks remaining 0.001 BTC as dust
6. ✅ TP/SL doesn't arm (remaining insignificant)

**Result:** Emission happens **regardless** of remaining position.

### Scenario 2: SELL execution below minimum quote

```
Position: 0.00001 BTC (worth $0.50)
SELL order: 0.00001 BTC at $50,000 = $0.50
NO_REMAINDER_BELOW_QUOTE: $10.00
```

**Flow:**
1. ✅ Order execution might be skipped **before placement** if remaining < floor
2. BUT: If order **does execute**, emission is **unconditional**
3. ✅ `_handle_post_fill()` still runs
4. ✅ TRADE_EXECUTED is emitted
5. ✅ Finalization completes
6. ✅ SharedState may mark as fully dust-closed

**Result:** **Execution may be skipped, but if it executes, emission always happens.**

---

## Configurable Parameters (Reference)

| Config | Default | Purpose |
|--------|---------|---------|
| `NO_REMAINDER_BELOW_QUOTE` | 0.0 | Minimum remaining notional (blocks entry, not emission) |
| `DUST_POSITION_QTY` | 0.00001 | Qty threshold for dust classification |
| `SIGNIFICANT_POSITION_FLOOR` | 25.0 | Min USDT to arm TP/SL (doesn't block emission) |
| `MIN_ECONOMIC_TRADE_USDT` | 10.0 | Min notional for BUY entry (post-fill unaffected) |

**None of these gate TRADE_EXECUTED emission.**

---

## Where Floor Checks Actually Happen

### 1. **Pre-Execution (Entry Gate)**
```python
# core/execution_manager.py ~line 4900
if remaining_value < floor:
    return {"ok": False, "status": "skipped", "reason": "TERMINAL_DUST", ...}
    # ← Execution never starts
```

### 2. **TP/SL Arming (Post-Emission)**
```python
# core/execution_manager.py ~line 303-317
if notional < min_notional:
    self.logger.info("[TPSL_SKIPPED_ECONOMIC] %s notional=%.4f", sym, notional)
    # ← TP/SL not armed, but trade already recorded + emitted
```

### 3. **Dust Marking (Post-Emission)**
```python
# core/shared_state.py (position manager)
if remaining_qty <= dust_threshold:
    position.is_dust = True
    # ← Marked after emission, doesn't block it
```

**Emission happens at level 2 or 3, not 1.**

---

## Test Cases: Verify Emission

### Test 1: Partial SELL with dust residual
```python
async def test_sell_partial_leaves_dust():
    # Execute SELL of 0.009 BTC, leaving 0.001 BTC
    order = {
        "executedQty": 0.009,
        "status": "FILLED",
        "avgPrice": 50000,
    }
    
    result = await em._handle_post_fill("BTCUSDT", "SELL", order)
    
    assert result["trade_event_emitted"] == True  # ✅ Must be True
    assert result["emitted"] == True  # PnL event also emitted
```

### Test 2: SELL below min notional
```python
async def test_sell_below_notional():
    # Execute SELL of $2.00 (below $10 MIN_ECONOMIC_TRADE_USDT)
    order = {
        "executedQty": 0.00004,  # 0.00004 BTC × $50k = $2
        "status": "FILLED",
        "avgPrice": 50000,
    }
    
    result = await em._handle_post_fill("BTCUSDT", "SELL", order)
    
    assert result["trade_event_emitted"] == True  # ✅ Still emitted
    # (TP/SL just won't arm for next trade)
```

### Test 3: Zero execution (no emission)
```python
async def test_no_execution_no_emission():
    # Order did NOT fill
    order = {
        "executedQty": 0.0,  # No execution
        "status": "PENDING",
    }
    
    result = await em._handle_post_fill("BTCUSDT", "SELL", order)
    
    assert result["trade_event_emitted"] == False  # ✅ Correct (no fill)
    assert result["emitted"] == False
```

---

## Conclusion

✅ **The code is correct as-is.**

**TRADE_EXECUTED is emitted unconditionally for any fill with `executed_qty > 0`.**

Dust handling is a separate, downstream responsibility of SharedState/PositionManager.

**No changes needed.**

---

## References

- **Emission code:** `core/execution_manager.py:236-240` (unconditional TRADE_EXECUTED)
- **Early return (no fill):** `core/execution_manager.py:214-223` (if exec_qty <= 0)
- **Post-fill method:** `core/execution_manager.py:190-420`
- **Finalization:** `core/execution_manager.py:1391-1476`
- **ExchangeTruthAuditor:** `core/exchange_truth_auditor.py` (governance layer)
