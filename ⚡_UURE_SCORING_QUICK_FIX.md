# ⚡ UURE Scoring: Quick Fix Reference

**Status**: Zero score logs appearing  
**Cause**: Pre-scoring gate failing (no candidates collected)  
**Solution**: Ensure SharedState has symbols before UURE runs  

---

## The Issue

```
grep "score=" → nothing
```

Translation: UURE never reaches the scoring step.

---

## Why Scoring Doesn't Happen

**Flow**:
```
compute_and_apply_universe()
  ├─ Step 1: Collect candidates
  │   ├─ Get accepted_symbols
  │   ├─ Get positions
  │   └─ Union both
  │
  ├─ if candidates is empty: RETURN EARLY ← YOU ARE HERE
  │
  └─ (Scoring never happens)
```

**Pre-scoring gate**: `if not all_candidates: return`

---

## Quick Diagnostic

1. **Check if symbols exist**:
```python
ctx.shared_state.get_accepted_symbol_list()
# If empty → discovery hasn't run
```

2. **Check if UURE is being called**:
```
grep "[UURE] Starting universe rotation" → appears?
grep "[UURE] No candidates found" → appears?
```

3. **If "No candidates found" appears**:
   - Problem: No symbols in SharedState
   - Fix: Run discovery before UURE

4. **If neither appears**:
   - Problem: UURE loop never started
   - Fix: Check readiness gates cleared

---

## The Fix

### Option A: Seed Symbols Immediately

```python
# In bootstrap, before starting UURE:
await ctx.shared_state.set_accepted_symbols({
    "BTC/USDT": {"status": "TRADING"},
    "ETH/USDT": {"status": "TRADING"},
    # ... minimum 3-5 symbols
})

# Now UURE can score them
ctx._start_uure_loop()
```

### Option B: Run Discovery First

```python
# Discovery populates symbols
await ctx.discovery_engine.run_discovery()

# Check result
symbols = ctx.shared_state.get_accepted_symbol_list()
print(f"Discovery found: {len(symbols)} symbols")

# Now UURE can score
ctx._start_uure_loop()
```

### Option C: Add Logging to Find Exact Break

In `core/universe_rotation_engine.py`, line 476:

```python
# Change from debug to info
self.logger.info(
    f"[UURE] Candidates: {len(accepted_syms)} accepted, "
    f"{len(position_syms)} positions, {len(all_syms)} total"
)

if not all_syms:
    self.logger.error(f"[UURE] PRE-SCORING GATE FAILED - no candidates!")
    # Log why
    self.logger.error(f"  Accepted: {len(accepted)} items - {list(accepted.keys())[:5]}")
    self.logger.error(f"  Positions: {len(positions)} items - {list(positions.keys())[:5]}")
```

---

## After Fix

You should see:
```
[UURE] Candidates: 50 accepted, 10 positions, 60 total
[UURE] Scored 60 candidates. Mean: 0.642
[UURE] Ranked 60 candidates. Top 5: [...]
```

The `score=` logs appear in the **unified_score calculation** (get_unified_score method), which you won't see directly but will see the results in the ranking logs.

---

## Scoring Log Locations

1. **Calculation**: `shared_state.py:1862` → `get_unified_score()`
   - Computes: 40% conviction + 20% vol + 20% momentum + 20% liquidity
   - Doesn't log individual values (just calculates)

2. **Summary**: `universe_rotation_engine.py:601` → `_score_all()` 
   - Logs: `"[UURE] Scored N candidates. Mean: X.XXX"`
   - This is the score summary you're looking for

3. **Ranking**: `universe_rotation_engine.py:487` → `_rank_by_score()`
   - Logs: `"[UURE] Ranked N candidates. Top 5: [...]"`
   - Shows scored symbols with their values

---

## Working Example

```python
# Bootstrap with symbols seeded
async def bootstrap_with_uure():
    ctx = AppContext(config={"UURE_ENABLE": True})
    await ctx.public_bootstrap()
    
    # Seed symbols immediately
    ctx.shared_state.set_accepted_symbols({
        "BTC/USDT": {"status": "TRADING"},
        "ETH/USDT": {"status": "TRADING"},
    })
    
    # Trigger UURE once manually to test
    result = await ctx.universe_rotation_engine.compute_and_apply_universe()
    print(f"UURE result: {result['rotation']}")
    
    # Should see logs:
    # [UURE] Candidates: 2 accepted, 0 positions, 2 total
    # [UURE] Scored 2 candidates. Mean: 0.X
    # [UURE] Ranked 2 candidates. Top 5: [...]
```

---

## Key Files

| File | Method | Issue |
|------|--------|-------|
| `universe_rotation_engine.py` | `_collect_candidates()` | Empty candidates |
| `shared_state.py` | `get_accepted_symbols()` | Returns empty dict |
| `shared_state.py` | `get_unified_score()` | Scoring calculation |
| `app_context.py` | `_uure_loop()` | Loop never starts |

---

## Next Steps

1. Check if `[UURE] No candidates found` appears in logs
2. If yes → Fix: Seed symbols before UURE
3. If no → Fix: Check if UURE loop is starting (`[UURE] background loop started`)
4. Run diagnostic script to trace exact break point

**Most likely**: Discovery hasn't populated symbols yet when UURE first runs.

**Quick fix**: Add 1-2 seed symbols immediately in bootstrap.
