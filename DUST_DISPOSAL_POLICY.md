# Dust-Remnant Disposal Policy (Phase 3, May 2026)

## Overview

**Problem:** Positions below the min recovery notional ($10) were classified as DUST but never exited because they were marked unsellable based solely on notional value, ignoring the actual Binance exchange filters (`min_qty`, `min_notional`). This left dead capital stranded.

**Solution:** Three-layer dust disposal policy:
1. **Classify weak dust as disposal candidates** — check real exchange filters to determine if dust is actually sellable
2. **Optionally promote the smallest amount needed** — if qty is too small but promotion cost is cheap, flag for micro-BUY top-up
3. **Exit through the normal recovery path** — sellable dust becomes tier-0 in capital unlock ranking

---

## Implementation

### 1. Dust Sellability Classification (`portfolio_recovery.py`)

**Before:** Dust marked unsellable if `notional_usdt < min_recovery_notional_usdt` (binary threshold).

**After:** Two-step classification:

1. **`_is_dust_sellable(pos)`** — Check if dust meets Binance filter requirements:
   - Fetches LOT_SIZE (`min_qty`, `step_size`) and MIN_NOTIONAL filters from exchange
   - Returns `True` if `qty >= min_qty AND notional >= min_notional`
   - Returns `False` otherwise
   - Sets `pos.suggested_action = "SELL_DUST"` if sellable

2. **`_compute_dust_top_up_need(pos)`** — For unsellable dust, compute promotion cost:
   - Calculates how much additional qty is needed to reach `min_qty`
   - If `qty * price < $2.0` (configurable), flags `needs_top_up = True` and stores `top_up_qty`
   - Otherwise leaves dust unsellable (too expensive to promote)
   - Logs decision for debugging

### 2. Symbol Filters Caching (`portfolio_recovery.py`)

**New method: `_get_symbol_filters(symbol)`**
- Fetches Binance exchange info for LOT_SIZE and MIN_NOTIONAL filters
- Caches results to avoid repeated API calls
- Falls back to conservative defaults on fetch failure: `min_qty=0, min_notional=10`

### 3. Position Data Enhancement (`portfolio_recovery.py`)

**`RecoveryPositionRecord` new fields:**
```python
needs_top_up: bool = False      # True if dust needs qty promotion
top_up_qty: float = 0.0         # Quantity to buy for promotion
```

### 4. Dust Promotion in Decisions (`decisions.py`)

When recovery candidate is selected and `needs_top_up=True`:
1. Emit a micro-BUY decision first: `Action.OPEN, qty=top_up_qty`
   - Risk score: 0.05 (very safe)
   - Reason: `"dust_promotion:{symbol}"`
2. Then emit the SELL decision normally
3. Log both actions in recovery decision context

---

## Ranking Behavior

**Capital Unlock Candidate Ranking** (`rank_capital_unlock_candidates`):

| Status | Sellable | Tier | Priority |
|--------|----------|------|----------|
| DUST | ✅ Yes | 0 | Highest — remove first |
| PROFITABLE | ✅ Yes | 1 | Good opportunity |
| STALE | ✅ Yes | 2 | Liquidate stale |
| WEAK | ✅ Yes | 3 | P&L recovery |
| ENTRY_UNKNOWN | ✅ Yes (if critical) | 4 | Under duress only |
| DUST | ❌ No | 9 | Never — unsellable |
| Other | ❌ No | 9 | Never — unsellable |

**Result:** Sellable dust naturally floats to tier 0 and gets picked first for capital unlock.

---

## Edge Cases & Behavior

### Case 1: Dust meets exchange min_qty and min_notional
```
qty: 0.0001 BTC @ $45,000 = $4.50
exchange filter: min_qty=0.0001, min_notional=$2
→ Dust is SELLABLE: pos.sellable=True, suggested_action="SELL_DUST"
```

### Case 2: Dust below min_qty, but promotion is cheap
```
qty: 3 ADA @ $0.90 = $2.70
exchange filter: min_qty=3.5, min_notional=$5
promotion need: 0.5 * $0.90 = $0.45 < $2.0 limit
→ Dust NEEDS TOP-UP: pos.needs_top_up=True, top_up_qty=0.5
→ Emit BUY 0.5 ADA, then SELL 3.5 ADA
```

### Case 3: Dust below min_qty, promotion is too expensive
```
qty: 0.0001 BTC @ $45,000 = $4.50
exchange filter: min_qty=0.01, min_notional=$500
promotion need: 0.0099 * $45,000 = $445.50 > $2.0 limit
→ Dust REMAINS UNSELLABLE: pos.sellable=False, needs_top_up=False
→ Left in wallet, but doesn't block new positions
```

### Case 4: No exchange client (paper mode / testnet)
- Falls back to conservative defaults: `min_qty=0, min_notional=$10`
- Most dust becomes unsellable
- Promotion logic still functions if cost is negligible

---

## Integration with Existing Systems

### PortfolioRecoveryEngine
- Called during `refresh()` when hydrating positions from wallet
- Automatically classifies all positions via `_classify_position()`

### NativeDecisionEngine
- Recovery candidate selected by `PortfolioRecoveryEngine.rank_capital_unlock_candidates()`
- Checked in `get_decisions()` at line ~473
- If `needs_top_up=True`, promotion BUY emitted before SELL

### NativeExecutor
- Already validates LOT_SIZE and MIN_NOTIONAL before placing any order
- Promotion BUY will be validated like any other order
- SELL will be validated with dust qty (now guaranteed to pass filters)

---

## Testing

**New test file:** `tests/test_dust_disposal_policy.py` (6 tests)

1. ✅ Dust with sufficient qty and notional is sellable
2. ✅ Dust below exchange min_qty is not sellable
3. ✅ Dust promotion flagged if cost is small
4. ✅ Dust promotion not flagged if cost is high
5. ✅ Sellable dust ranks tier 0 for capital unlock
6. ✅ Unsellable dust does not rank

**Existing test coverage:** All 700 existing tests pass + 6 new tests = **706 passing**

---

## Logging & Debugging

### Key log lines:

**Dust classified as sellable:**
```
suggested_action: "SELL_DUST"
```

**Dust needing promotion:**
```
DEBUG: Dust ADAUSDT: qty=3.0 < min=3.5; can promote for $0.45
```

**Dust too expensive to promote:**
```
DEBUG: Dust BTCUSDT: qty=0.0001 notional=$4.50; too expensive to promote (would cost $445.50)
```

**Promotion BUY emitted:**
```
INFO: 💰 DUST PROMOTION BUY: ADAUSDT qty=0.5 to reach min_qty threshold
```

**Dust disposal via recovery SELL:**
```
INFO: CAPITAL_UNLOCK_DECISION {..., dust_promotion=True, promotion_qty=0.5, ...}
```

---

## Parameters & Configuration

### PortfolioRecoveryEngine init:
```python
min_recovery_notional_usdt: float = 10.0  # Classification threshold for DUST status
```

### Promotion cost limit (hard-coded):
```python
promotion_cost_limit = 2.0  # Max $2 to promote dust
```

### Decision engine:
```python
risk_score=0.05  # for promotion BUY (very safe)
risk_score=0.1   # for recovery SELL (low risk)
```

---

## Future Enhancements

1. **Async exchange filter fetching:** Currently synchronous; could cache in SharedState and pre-fetch on startup
2. **Adaptive promotion limits:** Could base $2 limit on account NAV
3. **Dust consolidation:** Could batch multiple dust positions into single BUY if combined cost is good
4. **Multi-leg promotion:** Could emit USDT borrow + BUY + SELL as a single composite action

---

## Summary

The dust-remnant disposal policy transforms stranded dust from dead capital into:
- **Sellable dust:** Identified via real exchange filters → ranked tier 0 for capital unlock → exited immediately
- **Promotable dust:** Flagged for cheap top-up → micro-BUY + SELL executed as recovery cycle
- **Unpromotable dust:** Left in portfolio but doesn't block new positions (no slot tax)

Result: Dead capital freed, account NAV normalized, position capacity available for fresh opportunities.
