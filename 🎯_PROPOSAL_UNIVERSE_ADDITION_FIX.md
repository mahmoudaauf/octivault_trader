# 🎯 PROPOSAL UNIVERSE: REPLACEMENT → ADDITION FIX

## Problem Statement
The system currently treats each new proposal as a **full replacement universe** instead of an **addition** to the existing universe. This causes:
- Symbol universe shrinkage when proposals are made
- Loss of existing symbols when SymbolScreener proposes new ones
- Inefficient re-screening of the same symbols repeatedly
- Cap enforcement happening at write time rather than maintaining a growing set

## Root Cause
**File**: `core/shared_state.py`, lines 2096-2200 in `set_accepted_symbols()`

```python
# === HARD REPLACE MODE ===
# Build wanted set from incoming symbols
wanted = { self._norm_sym(k) for k in symbols.keys() }

# Remove everything not wanted (but protect wallet_force from non-wallet sources)
current_keys = set(self.accepted_symbols.keys())
for s in (current_keys - wanted):
    meta = self.accepted_symbols.get(s, {})
    # Wallet-force symbols are sticky: only remove if source is WalletScannerAgent
    if meta.get("accept_policy") == "wallet_force" and source != "WalletScannerAgent":
        self.logger.debug("🛡️ Protected wallet_force symbol %s from removal", s)
        continue
        
    self.accepted_symbols.pop(s, None)  # ← HARD DELETE
    self.symbols.pop(s, None)
```

This logic **removes all current symbols** that are not in the incoming proposal, effectively replacing the universe.

## Solution: Additive Mode for Discovery Agents

### Change Strategy
1. **For Discovery Agent proposals** (SymbolScreener, IPOChaser, etc.):
   - **MERGE** incoming symbols with existing ones (additive)
   - Only remove if explicitly told to do so
   - This allows multiple discovery passes to accumulate symbols

2. **For WalletScannerAgent**:
   - Retain current behavior (wallet-forced symbols are authoritative)
   
3. **Cap enforcement**:
   - Still apply cap, but on the merged set
   - Ensure we don't exceed limits after merging

### Implementation Details

#### Step 1: Add `merge_mode` parameter to `set_accepted_symbols()`
```python
async def set_accepted_symbols(
    self, 
    symbols: Dict[str, Dict[str, Any]], 
    *, 
    allow_shrink: bool = False,
    merge_mode: bool = False,  # NEW: True = add to existing, False = replace
    source: Optional[str] = None
) -> None:
```

#### Step 2: Update logic based on merge_mode
```python
if merge_mode:
    # ADDITIVE: Merge incoming with existing
    merged = dict(self.accepted_symbols)  # Start with current
    merged.update(symbols)  # Add/update incoming
    final = merged
else:
    # REPLACEMENT: Use only incoming (current behavior)
    final = dict(symbols)
```

#### Step 3: Update SymbolManager to use merge_mode
In `core/symbol_manager.py`, `propose_symbol()` and `propose_symbols()`:
```python
# When proposing from discovery agents, use merge_mode=True
await self._safe_set_accepted_symbols(
    final_map, 
    allow_shrink=False, 
    merge_mode=True,  # NEW: Additive for discovery
    source=source
)
```

## Files to Modify

1. **core/shared_state.py**
   - Add `merge_mode` parameter to `set_accepted_symbols()`
   - Implement additive merging logic when `merge_mode=True`

2. **core/symbol_manager.py**
   - Update `_safe_set_accepted_symbols()` to accept and pass `merge_mode`
   - Update `propose_symbol()` to use `merge_mode=True`
   - Update `propose_symbols()` to use `merge_mode=True`
   - Update `add_symbol()` to use `merge_mode=True`

3. **core/app_context.py** (if needed)
   - Verify any direct calls to `set_accepted_symbols()` use appropriate `merge_mode`

## Expected Behavior After Fix

**Before**:
- Proposal Pass 1: Accept symbols [A, B]
- Proposal Pass 2: Accept symbols [C, D] → Universe becomes [C, D] (lost A, B)

**After**:
- Proposal Pass 1: Accept symbols [A, B] → Universe = [A, B]
- Proposal Pass 2: Accept symbols [C, D] → Universe = [A, B, C, D] (additive)
- Cap enforcement: If cap=2, reduce to [A, B, C] or [B, C, D] (prioritization logic)

## Testing Checklist
- [ ] Symbol universe grows with multiple proposals
- [ ] Cap still enforced after merging
- [ ] Duplicates properly handled (no duplication on re-proposal)
- [ ] WalletScannerAgent still works correctly
- [ ] Shrink protection still active
- [ ] Logs show "merge" vs "replace" operations

## Backward Compatibility
- Default behavior (merge_mode=False) maintains current replace semantics
- Explicit opt-in to merge mode for discovery agents only
- No breaking changes to API signatures (new parameter is optional)
