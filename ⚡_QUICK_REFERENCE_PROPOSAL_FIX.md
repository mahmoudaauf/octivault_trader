# ⚡ QUICK REFERENCE: PROPOSAL UNIVERSE ADDITION FIX

## The Problem (Fixed ✅)
Each discovery agent proposal **replaced** the entire symbol universe instead of **adding** to it.

Example:
- Pass 1: SymbolScreener proposes [A, B, C] → Universe = [A, B, C]
- Pass 2: SymbolScreener proposes [D, E, F] → Universe = [D, E, F] ❌ Lost A, B, C!

## The Solution
Added `merge_mode` parameter to control proposal behavior:
- **merge_mode=True**: Add symbols to existing universe (additive)
- **merge_mode=False**: Replace universe with incoming symbols (legacy)

## What Changed

### File 1: core/shared_state.py
```python
# Before
async def set_accepted_symbols(self, symbols, *, allow_shrink=False, source=None):
    # Hard replace logic only

# After
async def set_accepted_symbols(self, symbols, *, allow_shrink=False, merge_mode=False, source=None):
    if merge_mode:
        # MERGE incoming with existing (additive)
    else:
        # REPLACE with incoming (legacy)
```

### File 2: core/symbol_manager.py
```python
# Three key methods now use merge_mode=True for discovery:

# 1. Single symbol proposal (from SymbolScreener, IPOChaser, etc.)
async def add_symbol(self, symbol, source="unknown", **kwargs):
    await self._safe_set_accepted_symbols(
        final_map, 
        allow_shrink=False,
        merge_mode=True,  # ← ADDITIVE
        source=source
    )

# 2. Batch proposal
async def propose_symbols(self, symbols, source="unknown", **kwargs):
    await self._safe_set_accepted_symbols(
        final_map, 
        allow_shrink=False,
        merge_mode=True,  # ← ADDITIVE
        source=source
    )

# 3. Signature updated
async def _safe_set_accepted_symbols(
    self, 
    symbols_map, 
    *, 
    allow_shrink=False, 
    merge_mode=False,  # ← NEW
    source=None
):
```

## Expected Behavior After Fix

### ✅ Multiple Discovery Passes
```
Discovery Pass 1: [A, B, C]   Universe: [A, B, C]
Discovery Pass 2: [D, E, F]   Universe: [A, B, C, D, E, F]  ← Growing!
Discovery Pass 3: [G, H, I]   Universe: [A, B, C, D, E, F, G, H, I]  ← Still growing!
```

### ✅ Cap Still Works
```
Cap = 50 symbols
After Pass 1: 30 symbols
After Pass 2: 45 symbols
After Pass 3: 65 symbols → Trimmed to 50  ✅
```

### ✅ No Breaking Changes
```
Existing code without merge_mode parameter: Works unchanged (merge_mode=False by default)
New discovery code: Uses merge_mode=True for additive behavior
Startup initialization: Still uses replace mode (as intended)
```

## Log Output Examples

### MERGE MODE (Discovery Agents)
```
[SS] 🔄 MERGE MODE: 2 + 50 = 52 symbols (source=SymbolScreener)
🎛️ CANONICAL GOVERNOR: 52 → 50 symbols (at SharedState)
[SS] 🔄 MERGE MODE: 50 + 10 = 60 symbols (source=SymbolScreener)
🎛️ CANONICAL GOVERNOR: 60 → 50 symbols (at SharedState)
```

### REPLACE MODE (Startup/Finalization)
```
[SS] 🔄 REPLACE MODE: 0 → 100 symbols (source=discovery)
[SS] 🔄 REPLACE MODE: 100 → 50 symbols (source=finalize)
```

## Backward Compatibility
✅ **100% Backward Compatible**
- Default behavior (merge_mode=False) preserves original semantics
- No breaking changes to public APIs
- Graceful fallback for older code

## Testing Strategy

### Unit Tests to Add
```python
# Test 1: Single symbol addition (additive)
async def test_propose_symbol_additive():
    state = ["A", "B"]
    add("C", merge_mode=True)
    assert state == ["A", "B", "C"]  # ✓ Additive

# Test 2: Batch proposal (additive)
async def test_propose_symbols_additive():
    state = ["A", "B"]
    add_batch(["C", "D"], merge_mode=True)
    assert state == ["A", "B", "C", "D"]  # ✓ Additive

# Test 3: Legacy replace mode still works
async def test_set_accepted_symbols_replace():
    state = ["A", "B"]
    set_accepted(["C", "D"], merge_mode=False)
    assert state == ["C", "D"]  # ✓ Replaced

# Test 4: Cap applied after merge
async def test_merge_with_cap():
    state = ["A", "B"] (cap=3)
    add_batch(["C", "D", "E"], merge_mode=True)
    assert len(state) == 3  # ✓ Capped at 3

# Test 5: Shrink rejection in replace mode
async def test_shrink_rejection_replace_mode():
    state = ["A", "B", "C"]
    result = set_accepted(["D"], allow_shrink=False, merge_mode=False)
    assert state == ["A", "B", "C"]  # ✓ Rejected shrink

# Test 6: Shrink allowed in merge mode (no operation)
async def test_no_shrink_in_merge_mode():
    state = ["A", "B", "C"]
    result = set_accepted(["D"], allow_shrink=False, merge_mode=True)
    assert state == ["A", "B", "C", "D"]  # ✓ Added, no shrink
```

### Integration Tests
```
1. Run SymbolScreener → Verify universe grows, doesn't replace
2. Run IPOChaser → Verify additions stack
3. Multiple passes → Verify cumulative effect
4. With cap enforcement → Verify cap applied after merging
5. WalletScannerAgent → Verify still works (uses replace mode)
6. Startup sequence → Verify initialization still works
```

## Monitoring Metrics

### Add to Dashboard/Logs
```
- Universe growth rate: symbols/minute
- Merge vs replace operations: count
- Cap enforcement frequency: times/period
- Discovery agent throughput: proposals/minute
- Symbol re-proposals: count (duplicates handled correctly)
```

## Performance Impact
✅ **Negligible**
- Merge operation: O(n) where n = current symbols (~50-200)
- Cap enforcement: O(n log n) sorting, only when over cap
- No new database queries or external calls

## Migration Checklist

- [x] Code changes implemented
- [x] Syntax validation passed
- [x] No breaking changes to APIs
- [x] Backward compatibility maintained
- [x] Logs updated for debugging
- [x] Documentation created
- [ ] Unit tests added (if using testing framework)
- [ ] Integration tests added (if using testing framework)
- [ ] Deployed and monitored (after deployment)
- [ ] Log monitoring for "MERGE MODE" vs "REPLACE MODE" operations

## Common Questions

**Q: Will existing code break?**
A: No. Default is merge_mode=False, preserving original behavior.

**Q: What about WalletScannerAgent?**
A: It uses the default merge_mode=False and still replaces (as intended).

**Q: How does cap work with merge?**
A: Cap applied AFTER merge. If merged > cap, oldest/lowest-priority symbols trimmed.

**Q: Can I manually set merge_mode=False for discovery?**
A: Yes, but not recommended. It defeats the purpose of additive discovery.

**Q: What if cap=0 (unlimited)?**
A: Cap enforcement skipped. Universe can grow unbounded.

## Support

If issues arise:
1. Check logs for "MERGE MODE" messages
2. Verify cap enforcement is working (look for "CANONICAL GOVERNOR" logs)
3. Ensure discovery agents are calling propose_symbol/propose_symbols
4. Confirm SharedState has set_accepted_symbols method
