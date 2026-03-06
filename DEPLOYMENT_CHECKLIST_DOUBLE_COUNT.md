# ✅ DEPLOYMENT CHECKLIST: Double-Count Fix

## Pre-Deployment

- [ ] **Code Review**
  - [ ] Read `core/shared_state.py` lines 3437-3459
  - [ ] Verify reconciliation logic makes sense
  - [ ] Check error handling is in place
  - [ ] Confirm no breaking changes

- [ ] **Technical Validation**
  - [ ] Run: `python -m py_compile core/shared_state.py`
  - [ ] Verify: No syntax errors
  - [ ] Check: Logger import is available
  - [ ] Confirm: Exception handling won't crash

- [ ] **Documentation Review**
  - [ ] Read: DOUBLE_COUNT_SIMPLE_EXPLANATION.md
  - [ ] Read: CRITICAL_FIX_DOUBLE_COUNT_DEPLOYED.md
  - [ ] Review: Testing procedures documented
  - [ ] Confirm: Rollback procedure clear

---

## Staging Deployment

- [ ] **Environment Setup**
  - [ ] Code deployed to staging
  - [ ] Trading bot started on staging
  - [ ] Logs accessible and monitoring
  - [ ] Exchange client connected (test API call succeeds)

- [ ] **Initial Test**
  - [ ] Bot starts without errors
  - [ ] `get_portfolio_snapshot()` can be called
  - [ ] No exceptions in logs
  - [ ] Reconciliation logic doesn't error

- [ ] **Functional Test: One BUY**
  - [ ] Execute one BUY order
  - [ ] Wait for fill confirmation
  - [ ] Call `get_portfolio_snapshot()`
  - [ ] Check for reconciliation log messages

- [ ] **Data Validation**
  - [ ] positions qty visible and correct
  - [ ] open_trades qty visible and correct
  - [ ] positions qty ≈ open_trades qty
  - [ ] NAV = cash + (position value)

---

## Staging Validation

### Check 1: Reconciliation Logging
```
Look in logs for either:

Option A (data was consistent):
  [No reconciliation message - data was already sync]
  ✓ This is fine

Option B (data was fixed):
  [WARNING] [RECONCILE] BTCUSDT: open_trade qty=X → balance qty=Y
  ✓ This is fine - fix applied

Option C (error):
  [WARNING] Failed to reconcile open_trades: ...
  ⚠️ This might indicate an issue
```

### Check 2: Portfolio Values
```
After BUY, verify in logs:
[INFO] Total portfolio value: NNN.NN USDT

Then manually calculate:
NAV = (USDT balance) + (position qty × current price)

Should match reported value!
```

### Check 3: Position Consistency
```
Execute:
  snapshot = await shared_state.get_portfolio_snapshot()
  pos_qty = snapshot["positions"]["BTCUSDT"]["quantity"]
  ot_qty = shared_state.open_trades["BTCUSDT"]["quantity"]

Verify:
  abs(pos_qty - ot_qty) < 0.00000001
  ✓ Should be true
```

### Check 4: No Exceptions
```
Monitor logs for:
  - Python exceptions
  - Assertion errors
  - Uncaught errors in reconciliation

Expected:
  Only [INFO] and [WARNING] logs, no errors
  ✓ Should see no [ERROR] level logs from reconciliation
```

---

## Pre-Live Validation

- [ ] **Staging Tests Complete**
  - [ ] One BUY tested successfully
  - [ ] Reconciliation either ran or not needed (both OK)
  - [ ] No exceptions in logs
  - [ ] NAV math verified
  - [ ] Position consistency verified

- [ ] **Code Ready**
  - [ ] All changes in one file (shared_state.py)
  - [ ] No dependencies on other changes
  - [ ] Rollback procedure is simple (one file)

- [ ] **Team Ready**
  - [ ] Team understands the fix
  - [ ] Team knows what to monitor
  - [ ] Team has rollback procedure
  - [ ] On-call person identified for first 2 hours

---

## Live Deployment

### Phase 1: Deploy
- [ ] Backup current version
- [ ] Deploy new code to production
- [ ] Restart trading bot
- [ ] Verify startup successful (no errors)

### Phase 2: Initial Monitoring (15 minutes)
- [ ] Check that bot is running
- [ ] Monitor for any exceptions
- [ ] Verify logs are being written
- [ ] Check CPU/memory are normal

### Phase 3: Trade Test (30 minutes)
- [ ] Execute one test BUY order
- [ ] Wait for fill
- [ ] Check reconciliation logs
- [ ] Verify NAV consistency
- [ ] Verify position tracking

### Phase 4: Extended Monitoring (1 hour)
- [ ] Continue normal trading operations
- [ ] Monitor for any issues
- [ ] Look for reconciliation messages
  - First hour: might see a few (expected)
  - After that: should be none or very rare
- [ ] Check for any exceptions

---

## What to Monitor

### Green Lights ✅
- [ ] Bot trading normally
- [ ] No exceptions in logs
- [ ] Reconciliation messages appear only occasionally
- [ ] NAV values make sense
- [ ] Positions tracking correctly

### Yellow Lights ⚠️
- [ ] Reconciliation messages appearing frequently
  - **Cause**: Something keeps putting data out of sync
  - **Action**: Investigate what's causing mismatches
- [ ] Prices seem stale
  - **Cause**: Exchange client price fetch issue
  - **Action**: Check if `get_ticker()` working
- [ ] NAV doesn't match math
  - **Cause**: Balance refresh or price issue
  - **Action**: Check balance and price API calls

### Red Lights 🔴
- [ ] Exceptions in logs from reconciliation
  - **Action**: Immediate rollback (see below)
- [ ] Bot crashes
  - **Action**: Immediate rollback
- [ ] Consistent failures to calculate NAV
  - **Action**: Investigate or rollback

---

## Rollback Procedure

If anything goes wrong:

```bash
# Step 1: Get previous version
git checkout HEAD~1 -- core/shared_state.py

# Step 2: Restart bot
# (It will reload the file)

# Result: Fix is disabled, normal operation resumes
```

**Time to rollback**: <2 minutes  
**Risk of rollback**: None (removes feature, bot still works)

---

## Post-Deployment

- [ ] **First Hour**
  - [ ] Monitor logs actively
  - [ ] Ready to rollback if needed
  - [ ] Team on alert

- [ ] **First Day**
  - [ ] Check for any issues
  - [ ] Verify reconciliation messages are normal
  - [ ] Ensure no cascading errors

- [ ] **First Week**
  - [ ] Normal operations
  - [ ] Reconciliation should be rare
  - [ ] Document any unexpected behavior

---

## Success Criteria

All of these should be TRUE after deployment:

- [ ] Bot starts and runs normally
- [ ] No exceptions from reconciliation logic
- [ ] Position quantities consistent (positions qty ≈ open_trades qty)
- [ ] NAV math validates: NAV = USDT + (position value)
- [ ] Reconciliation messages appear only occasionally
- [ ] No impact on trading logic or performance

---

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Developer | | | Reviewed ☐ |
| QA | | | Tested ☐ |
| DevOps | | | Ready to deploy ☐ |
| Approval | | | Approved ☐ |

---

## Documentation Links

| Document | Purpose |
|----------|---------|
| VISUAL_GUIDE_DOUBLE_COUNT_FIX.md | Visual explanation |
| DOUBLE_COUNT_SIMPLE_EXPLANATION.md | User-friendly overview |
| DOUBLE_COUNT_BUG_FINAL_DIAGNOSIS.md | Technical details |
| CRITICAL_FIX_DOUBLE_COUNT_DEPLOYED.md | Implementation guide |
| INDEX_DOUBLE_COUNT_FIX.md | Complete index |

---

## Quick Reference

**File Changed**: `core/shared_state.py` (lines 3437-3459)  
**Change Type**: Enhancement (add reconciliation)  
**Risk Level**: Low  
**Rollback Time**: 2 minutes  
**Testing Time**: 30 minutes  
**Monitoring Time**: 1 hour  

---

## Go/No-Go Decision

**Ready to deploy?** ☐ YES / ☐ NO

**If NO, reason:**
```
_________________________________________
_________________________________________
```

**Date: ____________**  
**Decision made by: ____________**

