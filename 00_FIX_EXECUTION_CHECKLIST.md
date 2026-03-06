# ✅ Signal Pipeline Fix - Execution Checklist

**Status:** Ready for diagnostic testing  
**Last Updated:** Session complete  
**Expected Completion Time:** 40-60 minutes

---

## Pre-Diagnostic Checklist ✅

- [x] Architecture documented (SIGNAL_PIPELINE_TRACE.md)
- [x] Root cause analyzed (SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md)
- [x] Code instrumented with diagnostic logs (4 locations)
- [x] Diagnostic guides created (3 documents + index)
- [x] Troubleshooting matrix documented
- [x] Commands ready to execute

**Status:** ✅ Ready to proceed

---

## Diagnostic Execution Checklist

### Phase 1: Run Diagnostic Test
- [ ] Open terminal
- [ ] Navigate to project directory:
  ```bash
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ```
- [ ] Run diagnostic test:
  ```bash
  python -m pytest tests/test_clean_run.py -xvs > logs/diagnostic_run.log 2>&1
  ```
- [ ] Wait for test to complete (2-3 minutes)
- [ ] Verify log file created:
  ```bash
  ls -la logs/diagnostic_run.log
  ```

**Checkpoint:** Test completed ✅

---

### Phase 2: Extract Diagnostic Logs
- [ ] Run extraction command:
  ```bash
  grep -E "\[AgentManager:NORMALIZE\]|\[AgentManager:SUBMIT\]|\[MetaController:RECV_SIGNAL\]|\[Meta:DRAIN" logs/diagnostic_run.log
  ```
- [ ] Save output to file for review:
  ```bash
  grep -E "\[AgentManager:NORMALIZE\]|\[AgentManager:SUBMIT\]|\[MetaController:RECV_SIGNAL\]|\[Meta:DRAIN" logs/diagnostic_run.log > logs/diagnostic_summary.txt
  ```
- [ ] View the output:
  ```bash
  cat logs/diagnostic_summary.txt
  ```

**Checkpoint:** Logs extracted ✅

---

### Phase 3: Check by Layer
- [ ] Check Layer 2 (Normalization):
  ```bash
  echo "=== Normalization ===" && grep "\[AgentManager:NORMALIZE\]" logs/diagnostic_run.log
  ```
  - [ ] See `Normalizing X signals`?
  - [ ] See `✓ Successfully normalized X`?
  - **Result:** ✅ Working / ❌ Broken / ⚠️ Unknown

- [ ] Check Layer 3 (Publishing):
  ```bash
  echo "=== Publishing ===" && grep "\[AgentManager:SUBMIT\]" logs/diagnostic_run.log
  ```
  - [ ] See `Publishing X intents`?
  - [ ] See `✓ Published X intents`?
  - **Result:** ✅ Working / ❌ Broken / ⚠️ Unknown

- [ ] Check Layer 4 (Draining):
  ```bash
  echo "=== Draining ===" && grep "\[Meta:DRAIN" logs/diagnostic_run.log
  ```
  - [ ] See `ENTERING _drain_trade_intent_events`?
  - [ ] See `DRAINED X events` (X > 0)?
  - **Result:** ✅ Working / ❌ Broken / ⚠️ Unknown

- [ ] Check Layer 5 (Reception):
  ```bash
  echo "=== Reception ===" && grep "\[MetaController:RECV_SIGNAL\]" logs/diagnostic_run.log
  ```
  - [ ] See `Received signal`?
  - [ ] See `✓ Signal cached`?
  - **Result:** ✅ Working / ❌ Broken / ⚠️ Unknown

**Checkpoint:** All layers checked ✅

---

### Phase 4: Identify Broken Link

Based on the logs above, identify which is missing:

**Use this decision tree:**

```
See NORMALIZE logs?
├─ NO  → Problem: Agent not calling normalization
│        Location: agent_manager.py line 410 (collect_and_forward_signals)
│        Action: Check if called from evaluate_and_act
│
└─ YES, see ✓ Successfully normalized?
   ├─ NO  → Problem: Signals failing validation
   │        Location: agent_manager.py line 340-360 (validation logic)
   │        Action: Check confidence, symbol, action filters
   │
   └─ YES → See SUBMIT logs?
      ├─ NO  → Problem: Not calling submit_trade_intents
      │        Location: agent_manager.py line 450 (submit call)
      │        Action: Check control flow after normalization
      │
      └─ YES, see ✓ Published?
         ├─ NO  → Problem: Event bus publishing failed
         │        Location: agent_manager.py line 265 (event_bus.publish)
         │        Action: Check event bus connection
         │
         └─ YES → See DRAIN logs?
            ├─ NO  → Problem: _drain not being called
            │        Location: meta_controller.py line 5840 (in evaluate_and_act)
            │        Action: Check lifecycle loop
            │
            └─ YES, see DRAINED X (X > 0)?
               ├─ NO  → Problem: Queue empty despite publishing
               │        Location: Check event bus subscription
               │        Action: Debug event bus state
               │
               └─ YES → See ✓ Signal cached?
                  ├─ NO  → Problem: SignalManager rejecting signals
                  │        Location: signal_manager.py line 60-100
                  │        Action: Check symbol/confidence validation
                  │
                  └─ YES → ✅ PIPELINE WORKING!
                           Check why no decisions being made
```

**Result:** Broken link identified at ________________

---

### Phase 5: Review Code Location

- [ ] Write down the broken code location:
  ```
  File: ___________________
  Function: ___________________
  Line: ___________________
  Problem: ___________________
  ```

- [ ] Open the file in VS Code:
  ```bash
  code core/[filename].py:[line_number]
  ```

- [ ] Read the surrounding code (10 lines before and after)

- [ ] Understand what's wrong based on:
  - The missing log indicates where control flow should go
  - The diagnostic guide (DIAGNOSTIC_FIXES_APPLIED.md) explains each location
  - The root cause analysis provides context

**Checkpoint:** Code location reviewed ✅

---

### Phase 6: Apply Fix

- [ ] Open document: `DIAGNOSTIC_FIXES_APPLIED.md`

- [ ] Find section for your broken code location

- [ ] Read the explanation of what's wrong

- [ ] Review the suggested fix

- [ ] Implement the fix in VS Code

- [ ] Save the file

- [ ] Verify no syntax errors

**Checkpoint:** Fix applied ✅

---

### Phase 7: Verify Fix

- [ ] Run diagnostic test again:
  ```bash
  python -m pytest tests/test_clean_run.py -xvs > logs/diagnostic_run_v2.log 2>&1
  ```

- [ ] Extract new logs:
  ```bash
  grep -E "\[AgentManager:NORMALIZE\]|\[AgentManager:SUBMIT\]|\[MetaController:RECV_SIGNAL\]|\[Meta:DRAIN" logs/diagnostic_run_v2.log
  ```

- [ ] Check signal cache:
  ```bash
  grep "Signal cache contains" logs/diagnostic_run_v2.log | tail -1
  ```
  - [ ] Should show: `Signal cache contains X signals` (X > 0)?

- [ ] Check decisions:
  ```bash
  grep "decisions_count" logs/diagnostic_run_v2.log | tail -1
  ```
  - [ ] Should show: `decisions_count=X` (X > 0)?

- [ ] Run full system test:
  ```bash
  python -m pytest tests/ -xvs > logs/full_test.log 2>&1
  ```

- [ ] Verify no errors in full test

**Checkpoint:** Fix verified ✅

---

## Success Criteria

### ✅ System is Fixed When:

- [ ] All 4 diagnostic logs appear in the test output
- [ ] `Signal cache contains X signals` (X > 0) in logs
- [ ] `decisions_count=X` (X > 0) in logs
- [ ] No `✗ Signal rejected` or `✗ Validation failed` logs
- [ ] No exceptions in test output
- [ ] All tests pass
- [ ] Manual test shows trades executing

### 🎉 Final Verification:

- [ ] Run system with real signals
- [ ] Observe trades executing normally
- [ ] Monitor logs for any errors
- [ ] Performance is acceptable

---

## Document Reference

| When You Need | Read This |
|---------------|-----------|
| Understand architecture | SIGNAL_PIPELINE_TRACE.md |
| Understand the problem | SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md |
| Diagnostic steps | SIGNAL_PIPELINE_QUICK_START.md |
| Detailed guidance | DIAGNOSTIC_FIXES_APPLIED.md |
| Overview | ANALYSIS_COMPLETE_SUMMARY.md |
| Navigation | 00_SIGNAL_PIPELINE_INDEX.md |

---

## Time Tracking

```
Phase 1: Run diagnostic test      [  3 min]
Phase 2: Extract logs            [  1 min]
Phase 3: Check by layer          [  3 min]
Phase 4: Identify broken link    [  2 min]
Phase 5: Review code             [  5 min]
Phase 6: Apply fix               [ 10 min] (variable)
Phase 7: Verify fix              [  3 min]
                                 ─────────
Total Expected                   [ 27 min] (minimum)
                                 [ 50 min] (with uncertainty)
```

---

## Backup & Safety

Before applying fixes:

- [ ] Create backup of modified file:
  ```bash
  cp core/[filename].py core/[filename].py.backup
  ```

- [ ] Commit current state to git:
  ```bash
  git add -A && git commit -m "Checkpoint before signal pipeline fix"
  ```

- [ ] Create backup of logs:
  ```bash
  cp logs/diagnostic_run.log logs/diagnostic_run.backup.log
  ```

If fix breaks something:
- [ ] Restore backup:
  ```bash
  cp core/[filename].py.backup core/[filename].py
  ```
- [ ] Review the fix more carefully
- [ ] Ask for additional guidance

---

## Troubleshooting

**Q: Diagnostic test fails with error**
- A: Check error message in logs/diagnostic_run.log
- A: Document the error and refer to root cause analysis

**Q: Diagnostic logs not appearing**
- A: Verify instrumentation was applied (grep for WARNING logs)
- A: Check if pytest is capturing logs correctly

**Q: Can't identify broken link from matrix**
- A: All 4 logs present? → Problem upstream (check TrendHunter)
- A: All 4 logs missing? → Problem in test setup
- A: Some logs present? → Use matrix to identify missing stage

**Q: Fix didn't work**
- A: Re-check the problem diagnosis
- A: Verify fix was applied correctly (diff against documentation)
- A: Check for related issues in adjacent code

---

## Contact / Escalation

If stuck:
1. Check DIAGNOSTIC_FIXES_APPLIED.md for your code location
2. Review SIGNAL_PIPELINE_TRACE.md for architecture context
3. Check diagnostic log output for error messages
4. Review git diff to see what code changed
5. Run with more verbose logging if needed

---

## Summary

**Goal:** Fix signal pipeline breakage  
**Method:** Diagnostic logging to identify broken link  
**Status:** Ready to execute  
**Time:** 27-50 minutes estimated  
**Difficulty:** Medium (follow checklists)  

**Next Step:** Execute Phase 1 → Run Diagnostic Test

Good luck! 🚀
