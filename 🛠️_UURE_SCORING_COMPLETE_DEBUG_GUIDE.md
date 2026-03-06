# 🛠️ UURE Scoring Failure: Complete Debugging & Fix Guide

**Symptom**: `grep "score="` returns nothing  
**Root Cause**: Pre-scoring gate failure - no candidates passed to scoring function  
**Impact**: UURE loop runs but scoring never executes  

---

## Problem Analysis

### What's Happening

When UURE runs, it follows this flow:

```python
async def compute_and_apply_universe(self) -> Dict[str, Any]:
    # Step 1: Collect candidates
    all_candidates = await self._collect_candidates()
    
    # GATE CHECK (Pre-Scoring Gate)
    if not all_candidates:
        self.logger.warning("[UURE] No candidates found")
        return result  # ← RETURNS EARLY, SCORING NEVER RUNS
    
    # Step 2: Score all candidates (NEVER REACHED)
    scored = await self._score_all(all_candidates)
```

**The pre-scoring gate is**: `if not all_candidates: return`

### Why You See Zero Score Logs

The scoring log appears in `_score_all()`:

```python
async def _score_all(self, candidates: List[str]) -> Dict[str, float]:
    # ... scoring logic ...
    
    if scores:
        self.logger.debug(  # ← This is never executed
            f"[UURE] Scored {len(scores)} candidates. "
            f"Mean: {sum(scores.values())/len(scores):.3f}"
        )
    else:
        self.logger.warning(f"[UURE] No candidates scored")
    return scores
```

**If `_score_all()` is never called**, you never see this log.

---

## Root Cause: Where Candidates Come From

The pre-scoring gate depends on `_collect_candidates()`:

```python
async def _collect_candidates(self) -> List[str]:
    """Step 1: Collect all candidate symbols from discovery & current positions."""
    try:
        # Get symbols from accepted set
        accepted = await self._maybe_await(self.ss.get_accepted_symbols())
        accepted_syms = set(accepted.keys())
        
        # Get symbols from positions
        positions = await self._maybe_await(self.ss.get_positions_snapshot())
        position_syms = set(positions.keys())
        
        # Union of both (all candidates)
        all_syms = accepted_syms | position_syms
        
        self.logger.debug(
            f"[UURE] Candidates: {len(accepted_syms)} accepted, "
            f"{len(position_syms)} positions, {len(all_syms)} total"
        )
        return list(all_syms)
```

**If both sources are empty**, `all_syms` is empty, gate fails.

### Two Sources Can Fail

1. **`get_accepted_symbols()` returns empty dict**
   - Cause: Discovery hasn't run / hasn't found symbols
   - State: `SharedState.accepted_symbols = {}`

2. **`get_positions_snapshot()` returns empty dict**
   - Cause: No open positions (normal during bootstrap)
   - State: `SharedState.positions = {}`

**If BOTH are empty**, candidates is empty, pre-scoring gate fails.

---

## Diagnosis: Finding the Exact Problem

### Step 1: Check What Appears in Logs

Run a UURE cycle and search for:

```bash
# Are candidates being collected?
grep "UURE.*Candidates" app.log

# Are symbols being scored?
grep "UURE.*Scored" app.log

# Is the pre-scoring gate failing?
grep "No candidates found" app.log
```

**Scenario A**: If you see `[UURE] No candidates found`
- Problem: Pre-scoring gate triggered
- Cause: `accepted_symbols` and `positions` both empty
- Fix: See "Fix A" below

**Scenario B**: If you see nothing after `[UURE] Starting universe rotation cycle`
- Problem: `_collect_candidates()` isn't being reached
- Cause: Earlier error or gate
- Fix: Check readiness gates, UURE loop startup

**Scenario C**: If you see `[UURE] Scored 0 candidates`
- Problem: Candidates collected but scoring failed for all
- Cause: Invalid candidate types or scoring exceptions
- Fix: Add type checking in `_score_all()`

---

## Fix A: Seed Symbols in Bootstrap

**Problem**: Discovery hasn't populated `accepted_symbols` yet when UURE first runs.

**Solution**: Manually seed minimum symbols before starting UURE.

**Location**: In your `AppContext.public_bootstrap()` or equivalent:

```python
async def public_bootstrap(self):
    # ... existing bootstrap code ...
    
    # After all initialization, BEFORE starting UURE:
    # Seed minimum symbols
    if self.shared_state:
        current = await self.shared_state.get_accepted_symbols()
        if not current:  # Only seed if empty
            self.logger.info("[Bootstrap] Seeding initial universe...")
            
            # Minimum viable set of symbols
            seed_symbols = {
                "BTCUSDT": {"status": "TRADING", "notional": 10},
                "ETHUSDT": {"status": "TRADING", "notional": 10},
                "BNBUSDT": {"status": "TRADING", "notional": 10},
                "SOLUSDT": {"status": "TRADING", "notional": 10},
                "ADAUSDT": {"status": "TRADING", "notional": 10},
            }
            
            await self.shared_state.set_accepted_symbols(seed_symbols)
            self.logger.info(f"[Bootstrap] Seeded {len(seed_symbols)} symbols")
    
    # Now start UURE - it will find these symbols
    # ... rest of bootstrap ...
```

---

## Fix B: Add Verbose Logging to _collect_candidates()

**Problem**: You're not seeing why candidates are empty.

**Solution**: Change debug logs to info and add detailed breakdown.

**Location**: `core/universe_rotation_engine.py`, lines 541-563

```python
async def _collect_candidates(self) -> List[str]:
    """Step 1: Collect all candidate symbols from discovery & current positions."""
    try:
        # Get symbols from accepted set
        accepted = await self._maybe_await(self.ss.get_accepted_symbols())
        accepted_syms = set(accepted.keys())
        
        # Get symbols from positions
        positions = await self._maybe_await(self.ss.get_positions_snapshot())
        position_syms = set(positions.keys())
        
        # Union of both (all candidates)
        all_syms = accepted_syms | position_syms
        
        # CHANGE: Log at INFO level with detailed breakdown
        self.logger.info(
            f"[UURE] Candidates: {len(accepted_syms)} accepted, "
            f"{len(position_syms)} positions, {len(all_syms)} total"
        )
        
        # ADD: If empty, log why
        if not all_syms:
            self.logger.error(
                f"[UURE] PRE-SCORING GATE: No candidates found!"
                f"\n  accepted source: {len(accepted)} items, keys={list(accepted.keys())[:3]}"
                f"\n  positions source: {len(positions)} items, keys={list(positions.keys())[:3]}"
            )
        
        # ADD: If found, log samples
        if all_syms:
            self.logger.debug(f"[UURE] Sample candidates: {list(all_syms)[:5]}")
        
        return list(all_syms)

    except Exception as e:
        self.logger.error(f"[UURE] Error collecting candidates: {e}", exc_info=True)
        return []
```

---

## Fix C: Add Pre-Scoring Gate Diagnostics

**Problem**: Pre-scoring gate fails silently.

**Solution**: Make it explicit with detailed logging.

**Location**: `core/universe_rotation_engine.py`, lines 475-485

```python
# Step 1: Collect all candidate symbols
self.logger.info("[UURE] Starting universe rotation cycle")
all_candidates = await self._collect_candidates()

# ADD: Explicit gate check with diagnostics
if not all_candidates:
    self.logger.error(
        "[UURE] PRE-SCORING GATE FAILED: Aborting rotation cycle. "
        "No candidates available from accepted_symbols or positions. "
        "Check SharedState population."
    )
    return result

# Now we know we have candidates
self.logger.info(f"[UURE] PRE-SCORING GATE PASSED: {len(all_candidates)} candidates ready for scoring")

# Step 2: Score all candidates
scored = await self._score_all(all_candidates)
```

---

## Fix D: Verify Scoring Actually Runs

**Problem**: Add explicit marker to verify scoring is executing.

**Solution**: Add "before" and "after" logging.

**Location**: `core/universe_rotation_engine.py`, lines 565-610

```python
async def _score_all(
    self, candidates: List[str]
) -> Dict[str, float]:
    """Step 2: Unified score for all candidates."""
    try:
        # ADD: Entry marker
        self.logger.info(f"[UURE:_score_all] ENTERING with {len(candidates)} candidates")
        
        scores = {}
        for candidate in candidates:
            # FIX #2: Safe handling of mixed candidate types
            if isinstance(candidate, str):
                symbol = candidate
            elif isinstance(candidate, dict):
                symbol = candidate.get("symbol")
                if not symbol:
                    self.logger.debug(f"[UURE] Skipping candidate dict without symbol: {candidate}")
                    continue
                symbol = str(symbol).upper()
            else:
                self.logger.debug(f"[UURE] Skipping non-string/non-dict candidate: {type(candidate).__name__} = {candidate}")
                continue
            
            symbol = str(symbol).upper()
            
            try:
                score = self.ss.get_unified_score(symbol)
                scores[symbol] = score
                # ADD: Debug each scoring result
                self.logger.debug(f"[UURE:_score_all] scored {symbol} = {score:.4f}")
            except Exception as score_err:
                self.logger.debug(f"[UURE] Failed to score {symbol}: {score_err}")
                continue

        # CHANGE: Info level + detailed summary
        if scores:
            mean_score = sum(scores.values()) / len(scores)
            self.logger.info(
                f"[UURE] Scored {len(scores)} candidates. "
                f"Mean: {mean_score:.4f}, Min: {min(scores.values()):.4f}, Max: {max(scores.values()):.4f}"
            )
            # ADD: Top 5 candidates by score
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.debug(f"[UURE] Top 5 scores: {[(s, f) for s, f in top_5]}")
        else:
            self.logger.error(f"[UURE] Scoring failed: {len(candidates)} candidates input, 0 scored")
        
        # ADD: Exit marker
        self.logger.info(f"[UURE:_score_all] EXITING with {len(scores)} successful scores")
        return scores

    except Exception as e:
        self.logger.error(f"[UURE] Error scoring candidates: {e}", exc_info=True)
        return {}
```

---

## Comprehensive Diagnostic Script

**Add this to test UURE independently:**

```python
async def diagnose_uure_scoring():
    """Complete UURE scoring diagnosis."""
    
    print("\n=== UURE SCORING DIAGNOSIS ===\n")
    
    ctx = AppContext(config={"UURE_ENABLE": True})
    await ctx.public_bootstrap()
    
    # Check 1: UURE engine exists
    print(f"✓ UURE engine exists: {ctx.universe_rotation_engine is not None}")
    if not ctx.universe_rotation_engine:
        print("✗ UURE engine is None - cannot diagnose")
        await ctx.graceful_shutdown()
        return
    
    uure = ctx.universe_rotation_engine
    
    # Check 2: SharedState has data
    accepted = await ctx.shared_state.get_accepted_symbols()
    positions = await ctx.shared_state.get_positions_snapshot()
    print(f"✓ Accepted symbols: {len(accepted)} - {list(accepted.keys())[:3]}")
    print(f"✓ Positions: {len(positions)} - {list(positions.keys())[:3]}")
    
    if not accepted and not positions:
        print("\n✗ PROBLEM: Both accepted and positions are empty!")
        print("  → Cause: No symbol sources available")
        print("  → Fix: Seed symbols before UURE runs")
        print("\n  Quick fix:")
        print("""
        ctx.shared_state.set_accepted_symbols({
            "BTCUSDT": {"status": "TRADING"},
            "ETHUSDT": {"status": "TRADING"},
        })
        """)
        await ctx.graceful_shutdown()
        return
    
    # Check 3: Can UURE collect candidates?
    print("\n→ Testing _collect_candidates()...")
    candidates = await uure._collect_candidates()
    print(f"✓ Collected {len(candidates)} candidates: {candidates[:5] if candidates else 'EMPTY'}")
    
    if not candidates:
        print("✗ PROBLEM: _collect_candidates() returned empty")
        print("  → Even though we saw accepted/positions above")
        print("  → Check: _maybe_await() handling")
        await ctx.graceful_shutdown()
        return
    
    # Check 4: Can UURE score?
    print("\n→ Testing _score_all()...")
    try:
        scored = await uure._score_all(candidates)
        print(f"✓ Scored {len(scored)} candidates")
        if scored:
            mean = sum(scored.values()) / len(scored)
            print(f"  Mean score: {mean:.4f}")
            print(f"  Scores: {list(scored.items())[:3]}")
        else:
            print("✗ PROBLEM: _score_all() returned empty dict")
            print("  → Candidates were collected but none scored")
            print("  → Check: get_unified_score() exceptions")
    except Exception as e:
        print(f"✗ PROBLEM: _score_all() raised exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Check 5: Full cycle
    print("\n→ Testing full compute_and_apply_universe()...")
    result = await uure.compute_and_apply_universe()
    print(f"✓ Full cycle result:")
    print(f"  new_universe: {len(result.get('new_universe', []))} symbols")
    print(f"  score_info keys: {len(result.get('score_info', {}))}")
    print(f"  rotation: {result.get('rotation', {})}")
    print(f"  error: {result.get('error')}")
    
    print("\n=== END DIAGNOSIS ===\n")
    await ctx.graceful_shutdown()

# Run it
asyncio.run(diagnose_uure_scoring())
```

---

## Quick Fix Priority

**Highest Priority** (do first):
1. Run diagnostic script to find exact break point
2. Apply Fix B (verbose logging) to see actual values
3. If `accepted_symbols` is empty → Apply Fix A (seed symbols)

**Second Priority** (for production):
1. Apply Fix C (explicit gate diagnostics)
2. Apply Fix D (detailed scoring logging)
3. Monitor logs after deployment

---

## After Fix: Expected Logs

```
[UURE] Starting universe rotation cycle
[UURE:_collect_candidates] ENTERING
[UURE] Candidates: 5 accepted, 2 positions, 7 total
[UURE:_collect_candidates] EXITING with 7 candidates
[UURE] PRE-SCORING GATE PASSED: 7 candidates ready for scoring
[UURE:_score_all] ENTERING with 7 candidates
[UURE:_score_all] scored BTCUSDT = 0.7654
[UURE:_score_all] scored ETHUSDT = 0.6432
...
[UURE] Scored 7 candidates. Mean: 0.6821, Min: 0.4200, Max: 0.9102
[UURE:_score_all] EXITING with 7 successful scores
[UURE] Ranked 7 candidates. Top 5: [('BTCUSDT', 0.9102), ...]
...
```

If you see these logs, scoring is working correctly.

---

## Summary

| Issue | Fix |
|-------|-----|
| `grep "score="` returns nothing | Apply Fix B (verbose logging) to find why |
| `[UURE] No candidates found` in logs | Apply Fix A (seed symbols in bootstrap) |
| Pre-scoring gate fails silently | Apply Fix C (explicit gate diagnostics) |
| Want production visibility | Apply Fix D (detailed scoring logging) |
| Need to verify the fix | Run diagnostic script above |

**Start with**: Diagnostic script to find exact break point.
