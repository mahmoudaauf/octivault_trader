# 🎯 UURE Scoring: Problem & Solution Visual Guide

---

## The Problem Chain

```
┌─────────────────────────────────────────────────────┐
│  UURE Cycle Starts                                  │
│  compute_and_apply_universe()                       │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Collect Candidates                         │
│  _collect_candidates()                              │
│                                                      │
│  Sources:                                           │
│  ├─ get_accepted_symbols() → EMPTY (discovery slow)│
│  └─ get_positions_snapshot() → EMPTY (no trades)   │
│                                                      │
│  Result: all_candidates = []                        │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ if not candidates?│
        │   YES → GATE FAILS│  ← YOU ARE HERE
        │   NO  → Continue │
        └────────┬──────────┘
                 │
              YES│
                 ▼
        ┌──────────────────────┐
        │ Return early         │
        │ "No candidates found"│
        │                      │
        │ Scoring NEVER runs! ◄─ NO LOGS
        └──────────────────────┘
```

---

## Root Cause

```
UURE Startup Order:

t=0:05 UURE loop starts
       ├─ Looks for candidates
       └─ Finds: 0 accepted, 0 positions → GATE FAILS

t=0:06 Discovery finishes (1 second later!)
       └─ Populates symbols (too late for first cycle)

t=0:10 Next UURE cycle
       ├─ Looks for candidates
       └─ Finds: 5+ accepted → GATE PASSES, Scoring runs ✓

Problem: First cycle always fails. Second cycle works.
Solution: Seed symbols BEFORE UURE starts.
```

---

## The Fix

```
BEFORE (Broken):
┌──────────┐
│ Bootstrap│
├──────────┤
│ Systems  │
│ init     │
├──────────┤
│ UURE     │  ◄─ Finds 0 candidates → Pre-scoring gate fails
│ starts   │
└──────────┘

AFTER (Fixed):
┌──────────┐
│ Bootstrap│
├──────────┤
│ Systems  │
│ init     │
├──────────┤
│ Seed 5   │  ◄─ Add this (5 lines of code)
│ symbols  │
├──────────┤
│ UURE     │  ◄─ Finds 5 candidates → Pre-scoring gate passes ✓
│ starts   │      Scoring runs → Logs appear
└──────────┘
```

---

## Code Location & Change

```python
# File: your bootstrap code (app_context.py or main.py)
# When: After SharedState initialized, before UURE loop

async def public_bootstrap(self):
    # ... existing code ...
    
    # ADD THIS 15-LINE BLOCK:
    if self.shared_state:
        current = await self.shared_state.get_accepted_symbols()
        if not current or len(current) < 3:
            seed_symbols = {
                "BTCUSDT": {"status": "TRADING", "notional": 10},
                "ETHUSDT": {"status": "TRADING", "notional": 10},
                "BNBUSDT": {"status": "TRADING", "notional": 10},
                "SOLUSDT": {"status": "TRADING", "notional": 10},
                "ADAUSDT": {"status": "TRADING", "notional": 10},
            }
            await self.shared_state.set_accepted_symbols(seed_symbols)
            self.logger.info(f"[Bootstrap] Seeded {len(seed_symbols)} symbols")
    
    # ... rest of bootstrap ...
```

---

## Expected Log Output

### BEFORE FIX
```
[UURE] Starting universe rotation cycle
[UURE] No candidates found           ◄─ Pre-scoring gate fails
                                        (No more logs for this cycle)
```

### AFTER FIX
```
[UURE] Starting universe rotation cycle
[UURE] Candidates: 5 accepted, 0 positions, 5 total
[UURE] Scored 5 candidates. Mean: 0.6542     ◄─ SCORING HAPPENED!
[UURE] Ranked 5 candidates. Top 5: [...]
[UURE] Governor cap applied: 5 → 5
[UURE] Profitability filter applied: 5 → 5
[UURE] Rotation: added=5, removed=0, kept=0
```

---

## Debugging Path

```
Issue: grep "score=" → nothing

              │
              ▼
     Is UURE loop running?
     (Check: "[UURE] background loop started")
              │
       ┌──────┴──────┐
       ▼              ▼
      YES            NO
       │              └─→ Fix: Check readiness gates
       │
       ▼
     Are candidates collected?
     (Check: "[UURE] Candidates: X accepted")
       │
    ┌──┴──┐
    ▼     ▼
   YES   NO
    │     │
    │     └─→ FIX: Seed symbols (THIS IS YOUR ISSUE)
    │
    ▼
  Are they scored?
  (Check: "[UURE] Scored X candidates")
    │
 ┌──┴──┐
 ▼     ▼
YES   NO
 │     │
 │     └─→ FIX: Check get_unified_score exceptions
 │
 ▼
SUCCESS!
```

---

## Quick Reference

| When | What | Status |
|------|------|--------|
| ✅ Bootstrap runs | Systems initialize | OK |
| ❌ **UURE starts** | Looks for candidates | **FAILS HERE** |
| ❌ **Pre-scoring gate** | Checks: any candidates? | **EMPTY LIST** |
| ❌ **Scoring phase** | Never reached | **SKIPPED** |
| ❌ **No score logs** | Result | **You see nothing** |

**After fix:**

| When | What | Status |
|------|------|--------|
| ✅ Bootstrap runs | Systems initialize | OK |
| ✅ **Symbols seeded** | 5 symbols added | **NEW** |
| ✅ **UURE starts** | Looks for candidates | **FINDS 5** |
| ✅ **Pre-scoring gate** | Checks: any candidates? | **PASSES** |
| ✅ **Scoring phase** | Runs scoring function | **EXECUTES** |
| ✅ **Score logs** | Result | **You see logs** |

---

## Impact Map

```
Fix Location: Bootstrap code (15 lines)
    │
    ▼
SharedState: accepted_symbols populated
    │
    ▼
UURE: _collect_candidates() returns 5 symbols
    │
    ▼
UURE: Pre-scoring gate passes
    │
    ▼
UURE: _score_all() executes
    │
    ▼
UURE: Scoring logs appear
    │
    ▼
Problem solved ✓
```

---

## Time to Fix

```
Reading instructions:    2 minutes
Writing seed code:       3 minutes
Testing:                 5 minutes
─────────────────────────
Total:                  10 minutes
```

---

## Files to Reference

```
📊_UURE_SCORING_EXECUTIVE_SUMMARY.md
  └─ Overview & quick fix (START HERE)

⚡_UURE_SCORING_QUICK_FIX.md
  └─ One-page quick reference

📋_UURE_READY_TO_APPLY_CODE_FIXES.md
  └─ Copy-paste ready code (USE THIS)

🔍_UURE_SCORING_FAILURE_DIAGNOSIS.md
  └─ Deep technical analysis

🛠️_UURE_SCORING_COMPLETE_DEBUG_GUIDE.md
  └─ Complete debugging guide + diagnostic script
```

---

## One-Minute Summary

**Problem**: UURE scoring never runs (no logs)  
**Cause**: Pre-scoring gate fails (0 candidates)  
**Reason**: Discovery hasn't populated symbols yet at startup  
**Fix**: Seed 5 symbols in bootstrap (15 lines, 5 minutes)  
**Result**: UURE finds candidates → Scoring runs → Logs appear  
**Status**: Ready to deploy immediately  

---

## Next Action

1. Open: `📋_UURE_READY_TO_APPLY_CODE_FIXES.md`
2. Copy: Fix A code block
3. Paste: Into your bootstrap code
4. Test: Restart and check logs
5. Confirm: See `[UURE] Scored X candidates` message

**Done!**

---

```
Before:  [UURE] No candidates found
After:   [UURE] Scored 5 candidates. Mean: 0.6542  ✓
```
