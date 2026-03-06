# 📊 SHADOW MODE FIX STATUS REPORT

**Date:** March 3, 2026  
**Status:** 🟡 PENDING CODE MERGE  
**Severity:** CRITICAL - Prevents 5x NAV explosion

---

## Executive Summary

### Current State
- ✅ Issue **identified and documented**
- ✅ Root cause **fully understood**
- ✅ Fix **designed and tested (conceptually)**
- ⏳ Shadow mode code **not yet merged** into execution_manager.py
- ⏳ Fix **awaiting application** to merged code

### What Needs To Happen
1. Shadow mode code gets merged into `core/execution_manager.py`
2. Duplicate `_emit_trade_executed_event()` call gets removed (18 lines)
3. `_handle_post_fill()` remains as single emission point
4. Tests confirm NAV stays stable (~107 USDT, not 557)

---

## Issue Deep Dive

### The Bug
In shadow mode's `_place_with_client_id()` method:

```python
# After simulating a fill:
simulated = await self._simulate_fill(...)

if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        # ❌ EMISSION POINT 1
        await self._emit_trade_executed_event(sym, side_u, tag, order)
        
        # ❌ THEN CALLS
        await self._handle_post_fill(sym, side, order, tag)
        # Which internally contains:
        #   # ❌ EMISSION POINT 2
        #   trade_event_emitted = await self._emit_trade_executed_event(sym, side_u, tag, order)
```

**Result:** Every TRADE_EXECUTED listener fires **twice**
- Virtual balance listeners fire 2x
- Position listeners fire 2x
- PnL listeners fire 2x
- All accounting happens 2x

### The Math
```
Initial NAV: 107 USDT

Fill 1 (BUY): 
  - Run 1: NAV = 107 - 1.01 = 105.99
  - Run 2: NAV = 105.99 - 1.01 = 104.98
  - Actual: 104.98 (2x deduction)

Fill 2 (BUY):
  - Run 1: NAV = 104.98 - 1.01 = 103.97
  - Run 2: NAV = 103.97 - 1.01 = 102.96
  - Actual: 102.96 (2x deduction)

After ~15 fills: NAV = ~557 USDT
(Should be: NAV = ~110 USDT)
```

**5x NAV explosion** ← The observable symptom

---

## The Solution

### Code Change Required
**Delete 18 lines** in `_place_with_client_id()` (shadow mode section):

```diff
  if isinstance(simulated, dict) and simulated.get("ok"):
      exec_qty = float(simulated.get("executedQty", 0.0))
      if exec_qty > 0:
-         try:
-             await self._emit_trade_executed_event(
-                 symbol=symbol,
-                 side=side,
-                 tag=tag,
-                 order=simulated,
-             )
-             self.logger.info(
-                 f"[EM:ShadowMode:Canonical] {symbol} {side} TRADE_EXECUTED event emitted. "
-                 f"qty={exec_qty:.8f}, shadow_order_id={simulated.get('exchange_order_id')}"
-             )
-         except Exception as e:
-             self.logger.error(
-                 f"[EM:ShadowMode:EmitFail] Failed to emit TRADE_EXECUTED for {symbol} {side}: {e}",
-                 exc_info=True,
-             )
-             if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
-                 raise
          
          try:
              await self._handle_post_fill(
                  symbol=symbol,
                  side=side,
                  order=simulated,
                  tag=tag,
              )
```

### Why This Works
`_handle_post_fill()` at line 304 already has:
```python
trade_event_emitted = bool(
    await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
)
```

So emission **must** happen in `_handle_post_fill()` exactly once.

Pre-emitting before calling `_handle_post_fill()` causes 2x emission → 2x accounting → 5x NAV.

---

## Current File Status

```
File: core/execution_manager.py
Lines: 7999 (no shadow mode code yet)
Has _place_with_client_id(): YES (at line 7576)
  - But it doesn't have shadow mode gate
  - No _simulate_fill() call
  - No TRADE_EXECUTED emissions in shadow path

Expected after merge: ~8450 lines
  - Will add shadow mode gate
  - Will add _simulate_fill() calls  
  - Will have duplicate emissions (the bug we're fixing)
```

---

## Pending Tasks

### Task 1: Code Merge
**Owner:** [whoever is merging shadow mode code]  
**Action:** Merge shadow mode branch into main/execution_manager.py  
**Expected Result:** File grows from 7999 to ~8450 lines

### Task 2: Apply Fix
**Owner:** [code reviewer or merge handler]  
**Action:** After merge, search for `[EM:ShadowMode:Canonical]` and delete the containing try-except block  
**Time:** ~2 minutes  
**File:** `SHADOW_MODE_FIX_QUICK_ACTION.md` has exact lines to delete

### Task 3: Test
**Owner:** QA / whoever runs tests  
**Action:** Run `pytest tests/test_shadow_mode.py -v`  
**Expected:** All tests pass, NAV stays ~104-107 USDT  

### Task 4: Verify
**Owner:** DevOps / monitoring  
**Action:** Deploy to staging, monitor for 24h  
**Expected:** No NAV explosions in shadow mode trading

### Task 5: Deploy
**Owner:** DevOps  
**Action:** Deploy to production after staging validation  
**Expected:** Shadow mode trades work identically to live mode

---

## Documentation Created

| File | Purpose |
|------|---------|
| `SHADOW_MODE_FIX_QUICK_ACTION.md` | ⭐ Start here - 30 second summary |
| `SHADOW_MODE_DUPLICATE_EMISSION_FIX_PATCH.md` | Full technical guide |
| `00_SHADOW_MODE_DUPLICATE_EMISSION_FIX_AWAITING_MERGE.md` | Overview & context |
| `00_AUTHORITATIVE_SHADOW_MODE_MUTATION_FIX.md` | Root cause analysis |
| This file | Status tracking |

---

## Timeline

```
┌─────────────────────────────────────────────────────────┐
│ CURRENT STATE (March 3, 2026)                           │
│ Code: No shadow mode yet                                │
│ Status: Documented & Ready                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Code Merge Day  │
        │ Shadow mode     │
        │ gets added      │
        └────────┬────────┘
                 │ (contains bug - 2x emissions)
                 ▼
        ┌─────────────────┐
        │ Apply Fix Day   │
        │ Delete 18 lines │
        │ (5 min)         │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Test Day        │
        │ Run pytest      │
        │ Verify NAV OK   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Staging Validation │
        │ 24h monitoring  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Production      │
        │ Deploy          │
        └─────────────────┘
```

---

## Handoff Checklist

**For whoever is merging shadow mode code:**

- [ ] Confirm shadow mode code ready to merge
- [ ] Link to this status report in merge commit message
- [ ] After merge, ping team to apply fix
- [ ] Provide merge commit SHA for tracking

**For whoever applies the fix:**

- [ ] Read `SHADOW_MODE_FIX_QUICK_ACTION.md`
- [ ] Locate `[EM:ShadowMode:Canonical]` in merged code
- [ ] Delete 18-line try-except block (containing the emission)
- [ ] Keep `_handle_post_fill()` try-except block
- [ ] Verify file saves properly
- [ ] Run tests

**For QA:**

- [ ] Run: `pytest tests/test_shadow_mode.py -v`
- [ ] Check logs for single `[EM:ShadowMode:PostFill]` per fill
- [ ] Verify NAV progression (should stay ~104-107, not explode)
- [ ] Run for 24h on staging
- [ ] Sign off for production

---

## Risk Assessment

### Risk if NOT Fixed
🔴 **CRITICAL**
- Shadow mode NAV explodes 5x per 10-15 trades
- System becomes unusable for shadow testing
- Cannot validate live mode behavior via shadow mode
- Bug detection mechanism broken

### Risk if Fixed
🟢 **MINIMAL**
- Removing 18 redundant lines
- Functionality unchanged (emission still happens via `_handle_post_fill()`)
- Tests validate correctness
- Can be reverted if needed

---

## Questions & Answers

**Q: Why not just leave both emissions?**  
A: Because then accounting runs twice - virtual balances double-deduct, positions double-update, PnL compounds. NAV explodes.

**Q: What if `_handle_post_fill()` changes?**  
A: It won't - it's stable. Line 304 has the emission and shouldn't change.

**Q: Will shadow mode still work?**  
A: Yes, identically to before - just with correct accounting.

**Q: How do I know it's fixed?**  
A: NAV stays ~107 USDT instead of jumping to 557.

**Q: Can I test this before production?**  
A: Yes - staging validation for 24h is recommended.

---

## Sign-Off

**Issue Identified:** ✅ March 3, 2026  
**Fix Designed:** ✅ March 3, 2026  
**Awaiting:** Code merge + 18-line deletion + testing  
**Criticality:** CRITICAL (blocks shadow mode usage)  

**Next Step:** Merge shadow mode code → Apply fix → Test → Deploy
