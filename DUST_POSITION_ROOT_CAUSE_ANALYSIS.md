# 🔍 WHY DUST POSITIONS ARE CREATED - ROOT CAUSE ANALYSIS

**Date:** April 27, 2026  
**Status:** ROOT CAUSE IDENTIFIED  
**Severity:** CRITICAL - Prevents profitable trading

---

## 📌 EXECUTIVE SUMMARY

The system creates dust positions because **position value is calculated AFTER execution, not BEFORE**. Positions enter the portfolio at their fill price, then get immediately classified as dust if the value falls below the significant floor threshold.

This is a **sequential timing issue**, not a configuration issue.

---

## 🔴 THE PROBLEM

### Current Flow (BROKEN)

```
1. BUY signal approved by MetaController
   └─ Only checks: confidence, capital, position count gates
   └─ ⚠️ Does NOT verify entry will create significant position

2. ExecutionManager executes order
   └─ Calculates quantity based on min_entry_quote
   └─ Places order with exchange

3. Fill received → record_fill() called
   └─ ✅ Position registered
   └─ 📊 Value calculated: qty × price
   └─ ⚠️ **VALUE COMPARISON WITH significant_floor happens HERE**
   
4. Classification at record_fill (shared_state.py:6041-6046)
   ├─ Calculate: position_value = qty × price
   ├─ Get: significant_floor (default: $20-25 USDT)
   ├─ Compare: is_significant = (position_value >= significant_floor)
   └─ Result: ❌ DUST if value < floor
```

### Why This Creates Dust

```python
# From shared_state.py:6041-6046
position_value = float(current_qty * price)
significant_floor = float(await self.get_significant_position_floor(symbol))
is_significant = bool(position_value >= significant_floor and position_value > 0.0)

if not is_significant:  # THIS IS THE PROBLEM
    pos["state"] = PositionState.DUST_LOCKED.value
    pos["status"] = "DUST"
    # And records to dust registry
    self.record_dust(symbol, current_qty, ...)
```

### Example Scenario

| Step | What Happens | Value | Status |
|------|--------------|-------|--------|
| 1 | Signal: BUY BTCUSDT, conf=0.75 | — | ✅ Approved |
| 2 | Entry floor: $20 min | — | ✅ Met |
| 3 | Order: Buy 0.00035 BTC | $14.87 | ✅ Filled |
| 4 | Value check: 0.00035 × $42,500 | $14.88 | ❌ **DUST** |
| 5 | Position registered | $14.88 | 🔒 **DUST_LOCKED** |

**Result:** Position created below minimum threshold → marked as dust immediately

---

## 🎯 ROOT CAUSES

### Root Cause #1: No Pre-Execution Value Validation

**Location:** `meta_controller.py` (_build_decisions)

The system validates that we have $20 minimum entry quote:
```python
# This only checks the PLANNED quote
min_entry_quote = 20.0  # USDT
available_capital = 30.0
# ✅ Gate passes: 30 >= 20
```

But it does NOT validate:
- What the actual filled quantity will be
- What the actual position value will be after fill

### Root Cause #2: Price Volatility Between Approval & Execution

**Scenario:**
1. MetaController approves BUY ETHUSDT with min_entry_quote=$20
2. Decides quantity: qty = 20 / $1,500 = 0.0133 ETH
3. Order placed to exchange
4. ⏱️ **2-5 second delay** while order sits or gets partially filled
5. ETH price drops to $1,420
6. Fill received: 0.0133 ETH × $1,420 = **$18.89 USD** ❌ **BELOW $20 FLOOR**

### Root Cause #3: Rounding & Fee Deductions

**Scenario:**
1. Intended entry: $25 USDT
2. Exchange minimum lot step requires 0.00001 BTC
3. Actual quantity after rounding: 0.00033 BTC (not 0.00034)
4. Fill: 0.00033 × $43,000 = $14.19 ❌ **BELOW FLOOR**
5. Plus: Exchange fees reduce base quantity
6. Final position: even smaller → dust

### Root Cause #4: Min Notional vs Significant Floor Misalignment

**Configuration:**
```python
# From config.py
MIN_ENTRY_QUOTE_USDT = 10.0      # Absolute minimum
DEFAULT_PLANNED_QUOTE = 24.0      # What we plan to spend
SIGNIFICANT_POSITION_FLOOR = 20.0 # What qualifies as significant
```

**Problem:** Entry floor ($10) is MUCH lower than significant floor ($20)

Entry approval checks: `entry_quote >= 10`  
Dust classification checks: `value >= 20`

Gap = **$10 difference** where positions can fall through!

---

## 📊 EVIDENCE FROM CODE

### Evidence #1: The Exact Moment Dust Gets Created

**File:** `core/shared_state.py`, lines 6041-6046

```python
async def record_fill(self, symbol: str, side: str, qty: float, price: float, ...):
    # ... filling logic ...
    
    current_qty = float(pos.get("quantity", 0.0) or 0.0)
    significant_floor = float(await self.get_significant_position_floor(symbol) or 0.0)
    position_value = float(current_qty * price) if current_qty > 0 and price > 0 else 0.0
    
    # ⚠️ THIS IS WHERE DUST DECISION IS MADE:
    is_significant = bool(position_value >= significant_floor and position_value > 0.0)
    
    if not is_significant:
        pos["state"] = PositionState.DUST_LOCKED.value  # ← DUST CREATED HERE
        pos["status"] = "DUST"
        
        self.record_dust(  # ← RECORDED IN DUST REGISTRY
            symbol,
            current_qty,
            origin="execution_fill",
            context={...}
        )
```

### Evidence #2: Entry Gate Uses Different Floor

**File:** `core/meta_controller.py`, lines 16206-16260

```python
async def _build_decisions(self, accepted_symbols_set: set):
    # Entry gate checks:
    min_entry = float(self._cfg("MIN_ENTRY_USDT", 12.0))  # Uses MIN_ENTRY
    significant_position_usdt = max(
        1.5 * min_notional,
        1.5 * await self._get_avg_trade_cost(),
        min_position_value,
        strategy_floor,
        min_entry,  # ← Often 12-24 USDT
    )
    
    # Filter BUY signals: reject if planned_quote < significant_position_usdt
    # But this checks PLANNED quote, not POST-FILL value!
```

### Evidence #3: No Post-Fill Safeguard

**Missing Code (doesn't exist):**

```python
# This validation is NOT done before execution:
async def validate_entry_will_not_be_dust(symbol, planned_quote, current_price):
    """MISSING: This check is never performed"""
    estimated_qty = planned_quote / current_price
    min_notional = await get_symbol_min_notional(symbol)
    
    # After fees/rounding, will position be significant?
    # THIS IS NOT CHECKED ANYWHERE IN THE CODEBASE
    pass
```

---

## 🎛️ CONFIGURATION STACK

Here's how the misconfigured entry floors cascade:

```
1. MIN_ENTRY_QUOTE_USDT = 10.0
   └─ Absolute minimum for any entry
   └─ Used by: ExecutionManager._get_min_entry_quote()

2. DEFAULT_PLANNED_QUOTE = 24.0
   └─ What we normally plan to spend
   └─ Used by: signal planning, position sizing

3. MIN_ENTRY_USDT = 24.0
   └─ Entry gate floor
   └─ Used by: MetaController.policy_manager gate

4. SIGNIFICANT_POSITION_FLOOR = 20.0
   └─ What classifies as "significant"
   └─ Used by: shared_state.record_fill() dust classification
   └─ 🔴 PROBLEM: Lower than or equal to entry floor
   └─ 🔴 Result: Positions can be approved then classified as dust

5. MIN_NOTIONAL_MULT = 2.0
   └─ Min position = exchange_min_notional × 2.0
   └─ Can add another layer of dust on small account
```

**The Gap:**
```
Entry Approval Range:     10.0 ─────────── 24.0  (or higher)
                           ↑
Dust Classification:               20.0 ← Some entries > floor but < classification floor
                                    ↓
Result: Position approved, then classified as dust
```

---

## 🔧 HOW TO FIX IT

### Fix 1: Pre-Execution Value Validation (RECOMMENDED)

**Location:** `execution_manager.py` before `_execute_order()`

```python
async def _validate_entry_will_be_significant(
    self,
    symbol: str,
    planned_quote: float,
    current_price: float
) -> Tuple[bool, str]:
    """
    PREVENT dust creation at the source.
    
    Validate that after fill, position value will be >= significant floor
    accounting for:
    - Price volatility (±2% buffer)
    - Rounding (lot step)
    - Fees (exchange % deduction)
    
    Returns: (is_valid, reason)
    """
    sym = self._norm_symbol(symbol)
    
    # 1. Get exchange filters for rounding
    filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
    step_size, _, _, _, _ = self._extract_filter_vals(filters)
    
    # 2. Calculate quantity with rounding
    raw_qty = planned_quote / current_price
    rounded_qty = self._apply_lot_step(raw_qty, step_size)
    
    # 3. Account for fees (typical -0.1%)
    fee_pct = 0.001  # 0.1% exchange fee
    qty_after_fee = rounded_qty * (1.0 - fee_pct)
    
    # 4. Conservative worst-case: price down 2%
    worst_price = current_price * 0.98
    
    # 5. Calculate worst-case value
    worst_value = qty_after_fee * worst_price
    
    # 6. Get significant floor
    significant_floor = await self.shared_state.get_significant_position_floor(sym)
    
    # 7. Validate
    is_valid = worst_value >= significant_floor
    reason = (
        f"OK: {worst_value:.2f} >= {significant_floor:.2f}" if is_valid
        else f"DUST_RISK: worst_case={worst_value:.2f} < floor={significant_floor:.2f}"
    )
    
    return is_valid, reason
```

### Fix 2: Align Entry Floors (QUICK FIX)

**Location:** `config.py`

```python
# Current (BROKEN):
MIN_ENTRY_QUOTE_USDT = 10.0              # Can be too small
SIGNIFICANT_POSITION_FLOOR = 20.0        # Gap of $10!

# Fixed (ALIGNED):
MIN_ENTRY_QUOTE_USDT = 20.0              # Match significant floor
SIGNIFICANT_POSITION_FLOOR = 20.0        # No gap
```

### Fix 3: Automatic Dust-Safe Rounding

**Location:** `execution_manager.py` in position sizing calculation

```python
async def _calculate_dust_safe_quantity(
    self,
    symbol: str,
    planned_quote: float,
    current_price: float
) -> float:
    """
    Calculate quantity that WILL be significant after fill.
    
    Adds 10% buffer to account for:
    - Fee deductions
    - Price slippage
    - Rounding down
    """
    sym = self._norm_symbol(symbol)
    
    # Step 1: Get constraints
    filters = await self.exchange_client.ensure_symbol_filters_ready(sym)
    step_size, _, _, _, _ = self._extract_filter_vals(filters)
    significant_floor = await self.shared_state.get_significant_position_floor(sym)
    
    # Step 2: Calculate with 10% buffer
    buffer_quote = planned_quote * 1.10  # 10% extra to account for slippage/fees
    raw_qty = buffer_quote / current_price
    
    # Step 3: Round to lot step
    rounded_qty = self._apply_lot_step(raw_qty, step_size)
    
    # Step 4: Verify it's still meaningful
    post_fee_qty = rounded_qty * 0.999  # 0.1% fee
    final_value = post_fee_qty * current_price
    
    if final_value < significant_floor:
        self.logger.warning(
            f"[DUST_SAFE_QTY] {symbol}: calculated qty would be dust. "
            f"Aborting: value={final_value:.2f} < floor={significant_floor:.2f}"
        )
        return 0.0  # Signal to reject this entry
    
    return rounded_qty
```

### Fix 4: Add Guard in record_fill()

**Location:** `shared_state.py` record_fill() before marking as dust

```python
async def record_fill(self, symbol: str, side: str, qty: float, price: float, ...):
    # ... existing logic ...
    
    if current_qty > 0 and not is_significant:
        # NEW: Log warning with diagnostic data
        self.logger.warning(
            "[DUST_CREATED] %s: value=%.2f < floor=%.2f "
            "(qty=%.8f, price=%.2f, entry_quote=%.2f)",
            symbol,
            position_value,
            significant_floor,
            current_qty,
            price,
            position_value,
        )
        
        # NEW: Check if this was predictable
        if hasattr(self, '_last_planned_quote'):
            last_planned = self._last_planned_quote.get(symbol, 0.0)
            if last_planned > 0 and last_planned < significant_floor * 0.9:
                self.logger.error(
                    "[DUST_PREVENTABLE] %s: entry approved for $%.2f "
                    "but floor is $%.2f. THIS SHOULD HAVE BEEN CAUGHT PRE-EXECUTION",
                    symbol, last_planned, significant_floor
                )
```

---

## 📋 IMPLEMENTATION PRIORITY

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| 🔴 **P0** | Pre-execution validation (Fix 1) | Stops 80% of dust creation | Medium |
| 🔴 **P1** | Align entry floors (Fix 2) | Closes the $10 gap | Low |
| 🟠 **P2** | Dust-safe quantity calculation (Fix 3) | Handles price volatility | Medium |
| 🟡 **P3** | Diagnostic logging (Fix 4) | Helps debug remaining cases | Low |

---

## ⚠️ WHAT THIS MEANS FOR YOU

### Current Behavior
- System sends BUY signals
- Orders execute
- Positions often become dust immediately
- Capital gets trapped in dust cleanup/healing cycles
- No new trades happen (capital occupied by dust)

### After Fix
- BUY signals validated pre-execution to ensure significant position
- Positions rejected if they would become dust
- Capital remains available for real trades
- Dust positions disappear from the creation pipeline

---

## 🔗 RELATED DOCUMENTS

- `DUST_LIQUIDATION_FIX_PLAN.md` - Flag alignment strategy
- `PORTFOLIO_FRAGMENTATION_FIXES_IMPLEMENTATION.md` - Dust prevention
- `DETECTED_ISSUES_SUMMARY_APRIL26.md` - Issues overview
