# ⚡ DUST POSITIONS - QUICK ANSWER

## TL;DR: Why Dust Positions Are Created

**The system creates dust positions because it validates entry AMOUNTS but not POST-FILL VALUES.**

### The Issue in 3 Steps

1. **MetaController approves:** "Buy with $20 minimum" ✅
2. **ExecutionManager executes:** Places order for ~0.0033 BTC ✅  
3. **Fill arrives:** "0.0033 BTC × $42,000 = $14.70" ❌ **BELOW $20 = DUST**

---

## Why It Happens

| Check Point | What Gets Validated | Problem |
|---|---|---|
| **Pre-execution** | Min entry quote ($10-24) | ✅ Checked |
| **Pre-execution** | Capital available | ✅ Checked |
| **Pre-execution** | Position count | ✅ Checked |
| **Pre-execution** | **Will position be significant after fill?** | ❌ **NOT CHECKED** |
| **Post-execution** | Dust classification | Position already created |

---

## The Root Cause

**File:** `core/shared_state.py` lines 6041-6046

```python
async def record_fill(self, symbol, side, qty, price, ...):
    # Calculate position value AFTER fill
    position_value = qty * price
    significant_floor = 20.0  # From config
    
    # If value < $20 = DUST (TOO LATE - position already created!)
    if position_value < significant_floor:
        mark_as_dust(symbol)  # ← Dust created here, after execution
```

---

## How to Prevent It

Add validation **before execution** to check:

```python
# Before ExecutionManager places order:
worst_case_value = (
    (planned_quote / price) *     # qty calculation
    (1.0 - 0.001) *               # minus 0.1% fee
    (0.98)                         # minus 2% price slippage
)

if worst_case_value < significant_floor:
    reject_entry()  # Stop it before it becomes dust!
```

---

## Configuration Problem

The entry floor is too low relative to the dust threshold:

```
MIN_ENTRY_QUOTE_USDT = 10.0 ← Can approve entries as low as $10
SIGNIFICANT_POSITION_FLOOR = 20.0 ← But dust at below $20
                         ↑ $10 gap = positions fall through!
```

---

## Quick Fixes (Ranked)

1. **Align configs:** MIN_ENTRY_QUOTE_USDT = 20.0 (match floor)
2. **Add pre-execution check:** Validate position will be significant before order
3. **Add price volatility buffer:** Plan for 2% slippage + 0.1% fees in qty calc
4. **Improve entry rounding:** Account for exchange lot step rounding

---

## Impact

**Without fix:**
- Every small entry → potential dust
- Capital trapped in dust → no new trades
- System stuck in dust healing loops

**With fix:**
- Entries rejected if they'd become dust
- Capital freed up for real positions
- Clean trading with significant positions only

---

See: `DUST_POSITION_ROOT_CAUSE_ANALYSIS.md` for full technical details
