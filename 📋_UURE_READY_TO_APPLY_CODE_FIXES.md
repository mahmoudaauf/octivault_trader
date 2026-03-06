# 📋 UURE Scoring: Ready-to-Apply Code Fixes

**Problem**: No score logs = pre-scoring gate failing = no candidates  
**Status**: Code fixes prepared and ready to deploy  

---

## What to Apply

Choose based on your situation:

| Fix | What | When | Impact |
|-----|------|------|--------|
| **A** | Seed symbols | No symbols in SharedState | Unblocks scoring immediately |
| **B** | Verbose logging | Finding root cause | Shows why candidates are empty |
| **C** | Gate diagnostics | Production visibility | Clear gate failure logs |
| **D** | Scoring detail logs | Detailed tracing | Shows each score calculation |

---

## Fix A: Seed Symbols in Bootstrap

**File**: Your bootstrap code or `app_context.py`  
**When**: After SharedState initialized, before UURE loop starts  
**Impact**: Immediate - UURE will have candidates to score

**Apply This**:

```python
# In public_bootstrap() or equivalent, after initialization:

# Seed minimum symbols if none discovered yet
if self.shared_state:
    current = await self.shared_state.get_accepted_symbols()
    if not current or len(current) < 3:
        self.logger.info("[Bootstrap] Seeding initial universe (discovery slow)...")
        
        seed_symbols = {
            "BTCUSDT": {"status": "TRADING", "notional": 10},
            "ETHUSDT": {"status": "TRADING", "notional": 10},
            "BNBUSDT": {"status": "TRADING", "notional": 10},
            "SOLUSDT": {"status": "TRADING", "notional": 10},
            "ADAUSDT": {"status": "TRADING", "notional": 10},
        }
        
        await self.shared_state.set_accepted_symbols(seed_symbols)
        self.logger.info(f"[Bootstrap] Seeded {len(seed_symbols)} symbols for UURE")
```

**Check**: After applying, you should see `[UURE] Candidates: 5 accepted` instead of 0.

---

## Fix B: Verbose Logging in _collect_candidates()

**File**: `core/universe_rotation_engine.py`  
**Lines**: 541-563  
**When**: During debugging to find exact issue  
**Impact**: Shows why candidates are empty

**Replace This** (lines 541-563):

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

    except Exception as e:
        self.logger.error(f"[UURE] Error collecting candidates: {e}")
        return []
```

**With This** (verbose version):

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

        # Log at INFO level with detail
        self.logger.info(
            f"[UURE] Candidates: {len(accepted_syms)} accepted, "
            f"{len(position_syms)} positions, {len(all_syms)} total"
        )
        
        # If empty, log why
        if not all_syms:
            self.logger.error(
                f"[UURE] PRE-SCORING GATE FAILURE: "
                f"accepted={len(accepted)} items, positions={len(positions)} items. "
                f"Accepted keys: {list(accepted.keys())[:5]}. "
                f"Positions keys: {list(positions.keys())[:5]}"
            )
        else:
            # If found, log sample
            self.logger.debug(f"[UURE] Sample candidates: {list(all_syms)[:5]}")
        
        return list(all_syms)

    except Exception as e:
        self.logger.error(f"[UURE] Error collecting candidates: {e}", exc_info=True)
        return []
```

**Check**: You should see either:
- `[UURE] Candidates: X accepted, Y positions, Z total` (normal)
- `[UURE] PRE-SCORING GATE FAILURE: ...` (problem found!)

---

## Fix C: Explicit Gate Diagnostics

**File**: `core/universe_rotation_engine.py`  
**Lines**: 472-486  
**When**: For production - make gate failures obvious  
**Impact**: Clear logs when pre-scoring gate fails

**Replace This**:

```python
        try:
            # Step 1: Collect all candidate symbols
            self.logger.info("[UURE] Starting universe rotation cycle")
            all_candidates = await self._collect_candidates()
            if not all_candidates:
                self.logger.warning("[UURE] No candidates found")
                return result

            # Step 2: Score all candidates
            scored = await self._score_all(all_candidates)
```

**With This**:

```python
        try:
            # Step 1: Collect all candidate symbols
            self.logger.info("[UURE] Starting universe rotation cycle")
            all_candidates = await self._collect_candidates()
            
            # Pre-scoring gate check
            if not all_candidates:
                self.logger.error(
                    "[UURE] PRE-SCORING GATE FAILED: No candidates collected. "
                    "Check SharedState.accepted_symbols and positions. "
                    "UURE will skip this cycle."
                )
                return result
            
            self.logger.info(
                f"[UURE] PRE-SCORING GATE PASSED: {len(all_candidates)} candidates "
                f"ready for scoring"
            )

            # Step 2: Score all candidates
            scored = await self._score_all(all_candidates)
```

**Check**: When gate fails, you'll see:
```
[UURE] PRE-SCORING GATE FAILED: No candidates collected...
```

Instead of generic:
```
[UURE] No candidates found
```

---

## Fix D: Detailed Scoring Logs

**File**: `core/universe_rotation_engine.py`  
**Lines**: 565-615  
**When**: For detailed tracing - optional  
**Impact**: See each symbol's score + summary stats

**Replace This**:

```python
    async def _score_all(
        self, candidates: List[str]
    ) -> Dict[str, float]:
        """Step 2: Unified score for all candidates."""
        try:
            scores = {}
            for candidate in candidates:
                # FIX #2: Safe handling of mixed candidate types
                # Handle string candidates, dict candidates, and float/invalid candidates
                if isinstance(candidate, str):
                    symbol = candidate
                elif isinstance(candidate, dict):
                    # If candidate is a dict, extract symbol
                    symbol = candidate.get("symbol")
                    if not symbol:
                        self.logger.debug(f"[UURE] Skipping candidate dict without symbol: {candidate}")
                        continue
                    symbol = str(symbol).upper()
                else:
                    # Skip invalid types (float, int, None, etc.)
                    self.logger.debug(f"[UURE] Skipping non-string/non-dict candidate: {type(candidate).__name__} = {candidate}")
                    continue
                
                # Ensure symbol is string and uppercase
                symbol = str(symbol).upper()
                
                try:
                    score = self.ss.get_unified_score(symbol)
                    scores[symbol] = score
                except Exception as score_err:
                    self.logger.debug(f"[UURE] Failed to score {symbol}: {score_err}")
                    # Continue with next candidate
                    continue

            if scores:
                self.logger.debug(
                    f"[UURE] Scored {len(scores)} candidates. "
                    f"Mean: {sum(scores.values())/len(scores):.3f}"
                )
            else:
                self.logger.warning(f"[UURE] No candidates scored (processed {len(candidates)} inputs)")
            return scores

        except Exception as e:
            self.logger.error(f"[UURE] Error scoring candidates: {e}", exc_info=True)
            return {}
```

**With This** (detailed version):

```python
    async def _score_all(
        self, candidates: List[str]
    ) -> Dict[str, float]:
        """Step 2: Unified score for all candidates."""
        try:
            self.logger.debug(f"[UURE:_score_all] ENTERING with {len(candidates)} candidates")
            
            scores = {}
            for candidate in candidates:
                # FIX #2: Safe handling of mixed candidate types
                # Handle string candidates, dict candidates, and float/invalid candidates
                if isinstance(candidate, str):
                    symbol = candidate
                elif isinstance(candidate, dict):
                    # If candidate is a dict, extract symbol
                    symbol = candidate.get("symbol")
                    if not symbol:
                        self.logger.debug(f"[UURE] Skipping candidate dict without symbol: {candidate}")
                        continue
                    symbol = str(symbol).upper()
                else:
                    # Skip invalid types (float, int, None, etc.)
                    self.logger.debug(f"[UURE] Skipping non-string/non-dict candidate: {type(candidate).__name__} = {candidate}")
                    continue
                
                # Ensure symbol is string and uppercase
                symbol = str(symbol).upper()
                
                try:
                    score = self.ss.get_unified_score(symbol)
                    scores[symbol] = score
                    # Detailed logging of each score
                    self.logger.debug(f"[UURE:_score_all] scored {symbol}={score:.4f}")
                except Exception as score_err:
                    self.logger.debug(f"[UURE] Failed to score {symbol}: {score_err}")
                    # Continue with next candidate
                    continue

            if scores:
                mean_score = sum(scores.values()) / len(scores)
                min_score = min(scores.values())
                max_score = max(scores.values())
                
                self.logger.info(
                    f"[UURE] Scored {len(scores)} candidates. "
                    f"Mean={mean_score:.4f}, Min={min_score:.4f}, Max={max_score:.4f}"
                )
                
                # Log top 5 candidates
                top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.debug(f"[UURE] Top 5 scores: {[(s, f'{f:.4f}') for s, f in top_5]}")
            else:
                self.logger.error(
                    f"[UURE] Scoring failed: {len(candidates)} candidates input, "
                    f"0 scored successfully. Check get_unified_score() exceptions."
                )
            
            self.logger.debug(f"[UURE:_score_all] EXITING with {len(scores)} successful scores")
            return scores

        except Exception as e:
            self.logger.error(f"[UURE] Error scoring candidates: {e}", exc_info=True)
            return {}
```

**Check**: You'll see detailed logs like:
```
[UURE:_score_all] ENTERING with 7 candidates
[UURE:_score_all] scored BTCUSDT=0.8432
[UURE:_score_all] scored ETHUSDT=0.6721
...
[UURE] Scored 7 candidates. Mean=0.6842, Min=0.4100, Max=0.9102
[UURE] Top 5 scores: [('BTCUSDT', '0.8432'), ...]
[UURE:_score_all] EXITING with 7 successful scores
```

---

## Application Order

1. **Start with Fix A** (seed symbols)
   - Easiest, most likely to fix the issue
   - 5 minute implementation

2. **Then apply Fix B** (verbose logging)
   - Shows you exactly what's happening
   - Helps confirm Fix A worked

3. **Then for production**, apply:
   - Fix C (gate diagnostics) - makes failures obvious
   - Fix D (optional, only if needed for tracing)

---

## Testing After Fixes

```python
# Quick test after applying fixes
async def test_uure_scoring_fixed():
    ctx = AppContext(config={"UURE_ENABLE": True})
    await ctx.public_bootstrap()
    
    # Trigger UURE manually
    result = await ctx.universe_rotation_engine.compute_and_apply_universe()
    
    # Check results
    assert result["new_universe"], "Universe should not be empty"
    assert result["score_info"], "Should have scores"
    assert len(result["score_info"]) > 0, "Should have scored at least one symbol"
    
    print(f"✓ UURE scoring working: {len(result['score_info'])} symbols scored")
    print(f"  New universe: {result['new_universe']}")
    
    await ctx.graceful_shutdown()

asyncio.run(test_uure_scoring_fixed())
```

**Expected output**:
```
✓ UURE scoring working: 7 symbols scored
  New universe: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT']
```

---

## Rollback Plan

If any fix causes issues:

1. Revert the file to original
2. Keep Fix A (seeding) - that's purely additive
3. Re-apply fixes one at a time

**Quick rollback command**:
```bash
git checkout core/universe_rotation_engine.py  # Revert all logging changes
```

---

## Summary

| Fix | Code | File | Lines | Effort | Impact |
|-----|------|------|-------|--------|--------|
| A | Seed symbols | bootstrap | 10 lines | 5 min | ⭐⭐⭐ High |
| B | Verbose logging | UURE | 20 lines | 5 min | ⭐⭐⭐ High |
| C | Gate diagnostics | UURE | 15 lines | 5 min | ⭐⭐ Medium |
| D | Score detail logs | UURE | 25 lines | 5 min | ⭐ Low (optional) |

**Recommended**: Apply A + B immediately. Very low risk, high visibility.

---

## Next Steps

1. Copy Fix A code above
2. Apply to your bootstrap
3. Restart system
4. Check logs for scoring messages
5. If still failing, apply Fix B for diagnostics
6. Share diagnostic output for further debugging
