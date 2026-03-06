# 🔧 Minimal Patch: Entry Price Reconstruction During Hydration

**Date:** 2026-03-06  
**Status:** READY FOR IMPLEMENTATION  
**Component:** SharedState (hydrate_positions_from_balances method)  
**Scope:** Minimal, non-breaking change

---

## Problem

Positions loaded from wallet balances have:
- ✅ `quantity` (from wallet)
- ✅ `latest_price` (fetched in Step 5)
- ❌ `entry_price` (missing - needed for P&L calculations)

Without `entry_price`, P&L calculations fail:
```
net_pnl = (latest_price - entry_price) × quantity
         = (latest_price - None) × quantity  ← ERROR
```

---

## Solution: Minimal Patch

**Location:** `SharedState.hydrate_positions_from_balances()` method

**Concept:**
```python
if position.entry_price is None:
    position.entry_price = avg_price or latest_price
```

**Priority Order:**
1. `avg_price` - if available from exchange fills (most accurate)
2. `latest_price` - fallback to current price (assumes breakeven entry)
3. Do nothing - if neither available (acceptable for startup)

---

## Implementation Details

### Where to Apply
**File:** `/core/shared_state.py` (or wherever hydrate_positions_from_balances is defined)

**Method:** `hydrate_positions_from_balances()`

**After:** Position object creation/update

### Code Pattern
```python
def hydrate_positions_from_balances(self):
    """Hydrate positions from wallet balances."""
    
    for symbol, balance in balances.items():
        # ... existing hydration code ...
        position = self.positions.get(symbol) or create_new_position(symbol)
        position.quantity = balance['free']
        
        # MINIMAL PATCH: Reconstruct entry_price if missing
        if position.entry_price is None:
            # Try: avg_price (from fills) → latest_price (fallback)
            position.entry_price = (
                getattr(position, 'avg_price', None) or
                self.latest_prices.get(symbol, None)
            )
            if position.entry_price:
                self.logger.debug(
                    f"[Hydration] Reconstructed entry_price for {symbol}: "
                    f"${position.entry_price:.2f}"
                )
        
        self.positions[symbol] = position
```

---

## Why This Is Minimal

✅ **Single responsibility:** Only fills missing `entry_price`  
✅ **Non-breaking:** Doesn't change existing positions with entry_price  
✅ **Safe fallback:** Uses latest_price if avg_price unavailable  
✅ **Idempotent:** Safe to call multiple times (checks None first)  
✅ **No side effects:** Doesn't modify hydration logic

---

## Impact

### Before Patch
```
Position hydrated: SOLUSDT
  quantity: 1.239
  latest_price: 89.82
  entry_price: None ← P&L calculation fails
```

### After Patch
```
Position hydrated: SOLUSDT
  quantity: 1.239
  latest_price: 89.82
  entry_price: 89.82 ← P&L calculation works (breakeven assumption)
```

---

## Testing

1. **Find hydrate_positions_from_balances()** in SharedState
2. **Add the patch** after position object creation
3. **Run startup:**
   ```bash
   python main_phased.py
   ```
4. **Check logs for:**
   ```
   [Hydration] Reconstructed entry_price for SOLUSDT: $89.82
   [Hydration] Reconstructed entry_price for BTCUSDT: $71526.19
   ```
5. **Verify P&L calculations** work without errors

---

## Next Steps

1. **Locate the file:** Find where `hydrate_positions_from_balances()` is defined (likely in `shared_state.py`)
2. **Apply the patch:** Add the 5-line reconstruction logic
3. **Deploy:** Restart bot and monitor logs
4. **Verify:** Confirm P&L calculations no longer fail

---

## FAQ

**Q: Why not use initial_price?**  
A: We want entry_price (actual purchase price). If unavailable, latest_price is better than None.

**Q: What if both are None?**  
A: Position is marked for liquidation in dust cleanup anyway. No harm leaving entry_price = None.

**Q: Will this break existing entry_price values?**  
A: No—patch only sets if `entry_price is None`. Existing values are preserved.

**Q: Is this temporary or permanent?**  
A: Permanent. Positions from wallet balances legitimately lack entry_price. This is the correct place to reconstruct it.

---

## Architectural Note

This patch belongs in **SharedState**, not StartupOrchestrator, because:
- SharedState is responsible for **position hydration**
- StartupOrchestrator only **coordinates** existing components
- Keeping logic in the right place maintains single responsibility principle

✅ StartupOrchestrator: Sequencing orchestrator (fixed)  
⏳ SharedState: Position hydration (needs this patch)
