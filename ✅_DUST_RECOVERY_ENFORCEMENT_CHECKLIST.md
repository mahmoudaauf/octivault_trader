# ✅ DUST RECOVERY CRITICAL RULE ENFORCEMENT CHECKLIST

## The Rule

```
INVARIANT: Dust must NOT block trades

1. Dust must NOT block BUY signals
2. Dust must NOT count toward position limits  
3. Dust must be REUSABLE when signal appears

IF ANY OF THESE FAIL → SYSTEM DEADLOCKS
```

---

## Status Summary

| Rule | Status | Evidence | Fix Location |
|------|--------|----------|--------------|
| #1: Dust doesn't block BUY | ❌ VIOLATED | Lines 9902-9930, `meta_controller.py` | Replace with `_position_blocks_new_buy()` |
| #2: Dust doesn't count limit | ❌ VIOLATED | Same location (consequence) | Same fix |
| #3: Dust is reusable | ❌ VIOLATED | Same location (consequence) | Same fix |

---

## Rule #1 Enforcement Checklist

### ❌ Current Implementation

```python
# meta_controller.py, lines 9902-9930
if existing_qty > 0:
    # Reject signal - treats dust same as viable
    skip_signal()
```

**Problem**: Dust blocks BUY unconditionally  
**Impact**: P0 Promotion cannot execute

### ✅ Required Implementation

```python
# What should happen:
blocks = await self._position_blocks_new_buy(sym, existing_qty)
if blocks:  # Only if SIGNIFICANT
    skip_signal()
else:
    allow_signal()  # Dust allowed through
```

**Solution**: Use existing dust-aware method  
**Status**: ⏳ PENDING IMPLEMENTATION

### Verification Tests

- [ ] **Test 1a**: Dust position ($5) allows BUY through
  ```python
  pos = create_dust_position('ETHUSDT', value=5.00)
  sig = create_buy_signal('ETHUSDT', conf=0.90)
  decisions = await meta._build_decisions([sig])
  assert len(decisions) > 0, "Dust should allow entry"
  ```

- [ ] **Test 1b**: Significant position ($50) blocks BUY
  ```python
  pos = create_position('BTCUSDT', value=50.00)
  sig = create_buy_signal('BTCUSDT', conf=0.90)
  decisions = await meta._build_decisions([sig])
  assert len(decisions) == 0, "Significant should block"
  ```

- [ ] **Test 1c**: Permanent dust ($0.50) allows BUY
  ```python
  pos = create_permanent_dust('ADAUSDT', value=0.50)
  sig = create_buy_signal('ADAUSDT', conf=0.90)
  decisions = await meta._build_decisions([sig])
  assert len(decisions) > 0, "Permanent dust should allow"
  ```

---

## Rule #2 Enforcement Checklist

### ❌ Current State

```
Position limit = 8 positions

Portfolio:
├─ Significant position A: $100
├─ Significant position B: $80
├─ Dust position C: $5  ← Should NOT count!
├─ Dust position D: $3
├─ (4 more positions)
└─ Total: 6 significant + 2 dust = "8 positions"

New signal E appears:
├─ Available slot in limit? NO (8/8)
├─ Dust is counting toward limit
└─ Signal rejected even though dust shouldn't count
```

**Problem**: Dust fills position slots  
**Impact**: New signals blocked due to dust

### ✅ Required Implementation

```
Position limit = 8 SIGNIFICANT positions (dust excluded)

Portfolio:
├─ Significant A: $100
├─ Significant B: $80
├─ Dust C: $5  ← Doesn't count!
├─ Dust D: $3  ← Doesn't count!
├─ (4 more significant)
└─ Count: 6 significant (not 8)

New signal E appears:
├─ Available significant slot? YES (6/8)
├─ Dust not counted
└─ Signal allowed
```

**Solution**: Count only significant positions  
**Status**: ⏳ DEPENDS ON RULE #1 FIX

### Verification Tests

- [ ] **Test 2a**: Position count excludes dust
  ```python
  meta.config.N_MAX_POSITIONS = 3
  
  # Add 2 significant + 1 dust
  add_position('BTCUSDT', value=50.00)  # Significant
  add_position('ETHUSDT', value=40.00)  # Significant
  add_position('ADAUSDT', value=2.00)   # Dust
  
  sig_count = await meta.shared_state.get_significant_position_count()
  assert sig_count == 2, "Dust should not be counted"
  ```

- [ ] **Test 2b**: Can add new position if dust filling slots
  ```python
  meta.config.N_MAX_POSITIONS = 3
  
  # Add 2 significant + 1 dust = 3 total, but only 2 significant
  add_position('BTCUSDT', value=50.00)
  add_position('ETHUSDT', value=40.00)
  add_position('ADAUSDT', value=2.00)
  
  sig = create_buy_signal('XRPUSDT', conf=0.90)
  decisions = await meta._build_decisions([sig])
  
  assert len(decisions) > 0, "Should allow - only 2 of 3 significant filled"
  ```

---

## Rule #3 Enforcement Checklist

### ❌ Current State

```
Dust ETHUSDT exists: value = $5

Strong BUY signal ETHUSDT: confidence = 0.95
├─ P0 promotion should execute:
│  ├─ Add freed capital: $25
│  ├─ Scale dust: $5 → $30
│  └─ Dust graduates to viable
│
But:
├─ Signal rejected at ONE_POSITION_GATE
├─ P0 never gets to evaluate
└─ Dust cannot be reused
```

**Problem**: Dust cannot be reused for recovery  
**Impact**: Recovery mechanism blocked

### ✅ Required Implementation

```
Dust ETHUSDT exists: value = $5

Strong BUY signal ETHUSDT: confidence = 0.95
├─ Signal allowed through (dust < floor)
├─ P0 evaluation:
│  ├─ Dust exists? YES
│  ├─ Signal exists? YES
│  ├─ Can add capital? YES
│  └─ Execute promotion: dust scales $5 → $30
├─ Dust recovers, position becomes viable
└─ Capital redistributed (not lost)
```

**Solution**: Allow signal through so P0 can execute  
**Status**: ⏳ DEPENDS ON RULE #1 FIX

### Verification Tests

- [ ] **Test 3a**: P0 Promotion executes when dust + signal exist
  ```python
  # Create dust
  meta.shared_state.positions['ETHUSDT'] = {
      'qty': 0.00133,
      'price': 3.00,
      'value': 4.00  # Dust
  }
  
  # Signal appears
  sig = create_strong_buy_signal('ETHUSDT', conf=0.95)
  
  # Can P0 execute?
  can_promote = await meta._check_p0_dust_promotion()
  assert can_promote == True, "P0 should be able to promote"
  
  # Does it execute?
  decisions = await meta._build_decisions([sig])
  assert len(decisions) > 0, "Signal should reach P0 decision"
  ```

- [ ] **Test 3b**: Dust grows toward viability after signal
  ```python
  # Setup dust + signal
  dust_value_before = 5.00
  signal = create_signal('ETHUSDT', confidence=0.90)
  
  # Execute
  decisions = await meta._build_decisions([signal])
  
  # Check: Did dust get capital added?
  dust_after = await get_position_value('ETHUSDT')
  assert dust_after > dust_value_before, "Capital should be added to dust"
  ```

---

## Implementation Checklist

### Phase 1: Code Change

- [ ] Open `core/meta_controller.py`
- [ ] Navigate to lines 9902-9930
- [ ] Identify the `if existing_qty > 0:` block
- [ ] Replace with dust-aware check:
  ```python
  if existing_qty > 0:
      blocks, pos_value, sig_floor, reason = await self._position_blocks_new_buy(sym, existing_qty)
      if blocks:
          # Skip only if SIGNIFICANT
          await self._record_why_no_trade(...)
          continue
      # else: allow signal through
  ```
- [ ] Update logging to show reason (dust_below_floor, etc.)
- [ ] Review changes for correctness

### Phase 2: Testing

- [ ] Run Test 1a: Dust allows BUY
  - Expected: ✅ Signal goes through
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 1b: Significant blocks BUY
  - Expected: ✅ Signal rejected
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 1c: Permanent dust allows
  - Expected: ✅ Signal goes through
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 2a: Position count correct
  - Expected: ✅ Dust not counted
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 2b: Can add new despite dust
  - Expected: ✅ Signal allowed
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 3a: P0 can execute
  - Expected: ✅ P0 decision made
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

- [ ] Run Test 3b: Dust grows toward viability
  - Expected: ✅ Capital added
  - Actual: `_______________`
  - Status: [ ] PASS [ ] FAIL

### Phase 3: Verification

- [ ] All 7 tests pass
- [ ] No regressions in other decision logic
- [ ] Logs show correct reason for skipped signals
- [ ] Logs show dust being allowed through
- [ ] No deadlocks observed in test runs

### Phase 4: Integration

- [ ] Code review passed
- [ ] Integration tests passed
- [ ] P0 promotions working in staging
- [ ] Dust positions tracked correctly
- [ ] Recovery metrics accurate

### Phase 5: Deployment

- [ ] Merge to main
- [ ] Deploy to production
- [ ] Monitor logs in live trading
- [ ] Verify P0 promotions executing
- [ ] Confirm dust recovery working

---

## Success Criteria

### Rule #1: Dust Doesn't Block BUY
- [ ] Dust positions (value < floor) allow signals through
- [ ] Significant positions still block signals
- [ ] Logging clearly shows "dust_below_significant_floor"
- [ ] Decision gate calls `_position_blocks_new_buy()`

### Rule #2: Dust Doesn't Count Limit
- [ ] Position count excludes dust positions
- [ ] New signals can enter despite dust-filled slots
- [ ] Portfolio has correct significant position count
- [ ] Dust positions don't affect `N_MAX_POSITIONS` check

### Rule #3: Dust is Reusable
- [ ] P0 Dust Promotion evaluates when signal + dust exist
- [ ] Dust can be scaled with freed capital
- [ ] Dust graduates to viable positions
- [ ] Capital is recovered (not permanently lost)

### Overall
- [ ] All three rules enforced
- [ ] All tests passing
- [ ] No regressions
- [ ] System survives stress tests without deadlock
- [ ] Capital recovery works in live trading

---

## Rollback Plan

If unexpected issues arise:

1. Revert the code change
   ```bash
   git revert <commit_hash>
   ```

2. Go back to simple check:
   ```python
   if existing_qty > 0:
       skip_signal()  # Original behavior
   ```

3. This restores the deadlock, but at least system won't crash
4. Post-incident analysis to understand what went wrong

**Note**: Rolling back leaves the deadlock in place, so this is TEMPORARY only.

---

## Timeline

| Date | Task | Owner | Status |
|------|------|-------|--------|
| (TBD) | Code review | (TBD) | ⏳ |
| (TBD) | Implementation | (TBD) | ⏳ |
| (TBD) | Testing phase 1-3 | (TBD) | ⏳ |
| (TBD) | Integration test | (TBD) | ⏳ |
| (TBD) | Staging deployment | (TBD) | ⏳ |
| (TBD) | Production deployment | (TBD) | ⏳ |
| (TBD) | Live trading verification | (TBD) | ⏳ |

---

## Sign-Off

- [ ] Developer: Implemented fix
- [ ] Tester: Tests pass (7/7)
- [ ] Reviewer: Code approved
- [ ] PM: Deployed to production
- [ ] Ops: Monitoring confirms working

---

## Notes

```
Add notes here during implementation:
_________________________________________________________________

_________________________________________________________________

_________________________________________________________________
```

---

## Contact

For questions about this fix:
- See: `⚠️_CRITICAL_DUST_BLOCKING_BUG_ANALYSIS.md`
- See: `🔧_DUST_BLOCKING_FIX_IMPLEMENTATION.md`
- See: `⚡_DUST_RECOVERY_INVARIANTS_QUICK_REFERENCE.md`

**This is a CRITICAL safety mechanism. Fix with urgency but care.**
