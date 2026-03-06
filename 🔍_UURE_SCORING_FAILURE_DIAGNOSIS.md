# 🔍 UURE Scoring Failure Diagnosis

**Issue**: "logs show zero score logs: `grep "score="` → nothing"  
**Root Cause**: All symbols fail the pre-scoring gate  
**Impact**: Scoring never executes because candidates don't make it to `_score_all()`

---

## The Problem

Your logs show:
```
grep "score=" → nothing
```

This means **no scoring is happening at all**.

### Timeline of Execution

```
compute_and_apply_universe()
  ├─ Step 1: _collect_candidates()
  │   └─ Returns: all_candidates = [sym1, sym2, ...]
  │
  ├─ Step 2: _score_all(all_candidates)  ← SCORING HAPPENS HERE
  │   ├─ Calls: get_unified_score(sym) for each
  │   └─ Logs: "[UURE] Scored N candidates"  ← YOU'RE NOT SEEING THIS
  │
  └─ Rest of pipeline...
```

**If no scoring logs appear**, it means one of these is true:
1. `all_candidates` is empty (no candidates collected)
2. `all_candidates` contains invalid types (filtered out in _score_all)
3. Scoring is raising exceptions (caught silently)
4. Log level is DEBUG (not visible at INFO level)

---

## Finding the Actual Problem

### Step 1: Check if Candidates Are Being Collected

Add this logging to diagnose:

**Location**: `core/universe_rotation_engine.py`, line 475

```python
all_candidates = await self._collect_candidates()
print(f"[DEBUG] all_candidates type: {type(all_candidates)}")
print(f"[DEBUG] all_candidates length: {len(all_candidates) if all_candidates else 'NONE'}")
if all_candidates:
    print(f"[DEBUG] first 5: {all_candidates[:5]}")
else:
    print(f"[DEBUG] EMPTY - no candidates!")
```

**Expected Output**:
```
[DEBUG] all_candidates type: <class 'list'>
[DEBUG] all_candidates length: 45
[DEBUG] first 5: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', ...]
```

**If you see**:
```
[DEBUG] all_candidates length: 0
[DEBUG] EMPTY - no candidates!
```

Then the problem is in `_collect_candidates()` — it's returning an empty list.

---

### Step 2: If Candidates Are Empty, Check Sources

**Location**: `core/universe_rotation_engine.py`, lines 545-549

```python
accepted = await self._maybe_await(self.ss.get_accepted_symbols())
accepted_syms = set(accepted.keys())

positions = await self._maybe_await(self.ss.get_positions_snapshot())
position_syms = set(positions.keys())

print(f"[DEBUG] accepted_syms: {len(accepted_syms)} - {list(accepted_syms)[:5]}")
print(f"[DEBUG] position_syms: {len(position_syms)} - {list(position_syms)[:5]}")
```

**Expected**:
```
[DEBUG] accepted_syms: 10 - ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', ...]
[DEBUG] position_syms: 5 - ['BTC/USDT', 'ETH/USDT', ...]
```

**If both are 0**:
- `get_accepted_symbols()` returning empty dict
- `get_positions_snapshot()` returning empty dict

This is the real pre-scoring gate: **data availability from SharedState**.

---

### Step 3: Check SharedState Population

If candidates are empty, the issue is **upstream**:

```python
# In your bootstrap/startup code:
print(f"[DEBUG] SharedState accepted: {ctx.shared_state.get_accepted_symbol_list()}")
print(f"[DEBUG] SharedState positions: {await ctx.shared_state.get_positions_snapshot()}")
```

**Possible reasons accepted is empty**:
1. ✅ Discovery agents haven't run yet
2. ✅ Discovery agents found no symbols
3. ✅ All discovered symbols were filtered out
4. ❌ `set_accepted_symbols()` was never called

---

## The Actual Pre-Scoring Gate Chain

It's NOT in `_score_all()` itself. The real "pre-scoring gate" is:

```
compute_and_apply_universe()
  │
  ├─ _collect_candidates()
  │   ├─ Gate: get_accepted_symbols() has symbols
  │   └─ Gate: get_positions_snapshot() has positions
  │   └─ Result: all_candidates = union of both
  │
  ├─ if not all_candidates: return early ← GATE FAILURE
  │
  └─ _score_all(all_candidates)
      ├─ For each symbol, call: get_unified_score()
      └─ Logs: "[UURE] Scored N candidates"
```

**The pre-scoring gate is: `if not all_candidates`**

If this triggers, you see:
```
[UURE] No candidates found
```

Not:
```
[UURE] Scored 0 candidates
```

---

## How to Fix

### Fix 1: Ensure Discovery Populates Symbols

```python
# Before UURE runs, ensure discovery has populated accepted symbols
async def bootstrap():
    ctx = AppContext(...)
    
    # Phase 1: Ensure symbols are discovered
    await ctx.discovery_engine.run_discovery()
    # or
    await ctx.symbol_screener.refresh_cache()
    
    # Phase 2: Now UURE can score them
    ctx._start_uure_loop()
```

### Fix 2: Manually Set Symbols if Discovery is Slow

```python
# Immediate fallback in SharedState
ctx.shared_state.set_accepted_symbols({
    "BTC/USDT": {...},
    "ETH/USDT": {...},
    # ... at least some symbols
})
```

### Fix 3: Add Logging to Verify Flow

In `_collect_candidates()`:

```python
accepted = await self._maybe_await(self.ss.get_accepted_symbols())
accepted_syms = set(accepted.keys())

positions = await self._maybe_await(self.ss.get_positions_snapshot())
position_syms = set(positions.keys())

all_syms = accepted_syms | position_syms

# ADD THIS:
self.logger.info(  # Changed from debug to info
    f"[UURE] Candidates: {len(accepted_syms)} accepted, "
    f"{len(position_syms)} positions, {len(all_syms)} total"
)
if not all_syms:
    self.logger.warning(f"[UURE] No candidates! Accepted source has {len(accepted)} items, positions has {len(positions)} items")

return list(all_syms)
```

---

## Expected Log Sequence (Working)

```
[UURE] Starting universe rotation cycle
[UURE] Candidates: 50 accepted, 10 positions, 60 total         ← _collect_candidates()
[UURE] Scored 60 candidates. Mean: 0.642                       ← _score_all()
[UURE] Ranked 60 candidates. Top 5: [('BTC/USDT', 0.95), ...] ← _rank_by_score()
[UURE] Governor cap applied: 60 → 5                             ← _apply_governor_cap()
[UURE] Profitability filter applied: 5 → 5                     ← _apply_profitability_filter()
[UURE] Relative rule: weakest_edge=0.45, required=0.35, qualifiers=5  ← relative rule
[UURE] Rotation: added=3, removed=2, kept=2                    ← rotation logic
```

**If you see nothing after "Starting universe rotation cycle"**, it means:
- `_collect_candidates()` returned empty
- Gate `if not all_candidates: return` triggers
- No further processing

---

## Diagnostic Script

Add this to your startup to trace the issue:

```python
async def diagnose_uure():
    """Diagnose why UURE scoring fails."""
    ctx = AppContext(...)
    await ctx.public_bootstrap()
    
    # Check 1: Are symbols accepted?
    accepted = await ctx.shared_state.get_accepted_symbols()
    print(f"✓ Accepted symbols: {len(accepted)} - {list(accepted.keys())[:5]}")
    
    # Check 2: Are there positions?
    positions = await ctx.shared_state.get_positions_snapshot()
    print(f"✓ Positions: {len(positions)} - {list(positions.keys())[:5]}")
    
    # Check 3: Can UURE collect?
    if ctx.universe_rotation_engine:
        candidates = await ctx.universe_rotation_engine._collect_candidates()
        print(f"✓ Candidates: {len(candidates)} - {candidates[:5] if candidates else 'EMPTY'}")
        
        # Check 4: Can UURE score?
        if candidates:
            scored = await ctx.universe_rotation_engine._score_all(candidates)
            print(f"✓ Scores: {len(scored)} - mean={sum(scored.values())/len(scored):.3f if scored else 'EMPTY'}")
        else:
            print("✗ No candidates to score!")
    
    await ctx.graceful_shutdown()

asyncio.run(diagnose_uure())
```

---

## Summary

**The "pre-scoring gate" is**: Empty candidates list before `_score_all()` is called.

**Root cause is upstream**: Discovery/SharedState not populating `accepted_symbols` or `positions`.

**Fix**: Ensure discovery runs before UURE, or manually seed symbols.

**Next**: Run the diagnostic script above to identify exactly where the chain breaks.
