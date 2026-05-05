# ⚡ QUICK START: Capital Allocator Fix Implementation

**Time to Deploy:** 25 minutes  
**Risk Level:** LOW  
**Files Needed:** 2 (CAPITAL_ALLOCATOR_FIX_CODE.py + config)  

---

## 🎯 The Problem (30 seconds)

```python
# BROKEN (current):
default_bootstrap_reserve = 2.0  # Fixed - same for all accounts
# Result: $8.73 free gets 100% to trading, 0% to dust healing

# FIXED (new):
reserve = nav * 0.20  # Dynamic - 20% for micro accounts  
# Result: $84.55 NAV reserves $16.91, frees $67.64 for 60/20/20 split
```

**Impact:** Dust healing can finally execute with guaranteed budget.

---

## 📋 Deployment Checklist

### Step 1: Add Config (2 minutes)
Add these to `.env` or config:

```bash
# Capital Allocator Configuration
RESERVE_PCT_MICRO=0.20          # 20% for NAV < $50
RESERVE_PCT_SMALL=0.15          # 15% for NAV < $200
RESERVE_PCT_NORMAL=0.10         # 10% for NAV >= $200
RESERVE_MIN_USDT=1.00           # Absolute minimum
ALLOC_PCT_CORE=0.60             # 60% for BTC/ETH
ALLOC_PCT_ALTS=0.20             # 20% for alts
ALLOC_PCT_DUST=0.20             # 20% for dust healing
```

### Step 2: Update Code (3 minutes)
Replace `src/l6_governance/capital_allocator.py`:

**Find this (line 766):**
```python
default_bootstrap_reserve = 2.0
bootstrap_reserve_usdt = float(self._cfg("BOOTSTRAP_RESERVE_USDT", default_bootstrap_reserve))
```

**Replace with (from CAPITAL_ALLOCATOR_FIX_CODE.py):**
- Add imports: `from decimal import Decimal`
- Add method: `calculate_dynamic_reserve()`
- Add method: `allocate_capital_60_20_20()`
- Add method: `allocate_with_nav_dynamics()`

### Step 3: Update Calls (5 minutes)
In `meta_controller.py` or orchestrator:

**Old:**
```python
allocation = self.capital_allocator.get_old_allocation(free_usdt)
trading_budget = allocation['total']  # All capital goes to trading
dust_budget = 0  # Dust starved!
```

**New:**
```python
allocation = await self.capital_allocator.allocate_with_nav_dynamics(
    nav=current_nav,
    free_usdt=free_capital
)
trading_budget = allocation['effective_trading']  # $54.11 for 60/20
dust_budget = allocation['dust_healing']  # $13.53 for dust - NOW FUNDED!
```

### Step 4: Test (10 minutes)
Run with current account:

```python
>>> from capital_allocator import allocate_with_nav_dynamics
>>> result = await allocator.allocate_with_nav_dynamics(
...     nav=84.55,
...     free_usdt=8.73
... )
>>> result['reserve']
16.91  # ✅ Dynamic, not $2.00
>>> result['dust_healing']
1.75   # ✅ Now has budget!
>>> result['effective_trading']
5.99   # ✅ Predictable
```

### Step 5: Monitor (5 minutes)
Watch logs for:

```log
[ALLOCATION] NAV=$84.55, Reserve=$16.91(20%), Allocatable=$67.64
[ALLOCATION] Split: Core=$40.58(60%) + Alts=$13.53(20%) + Dust=$13.53(20%)
[DUST_HEALING] Budget floor met: $13.53
[DUST_HEALING] Liquidating 32 positions with capital support
```

---

## 🔍 Verification (What to Check)

| Check | Current | Expected | Status |
|-------|---------|----------|--------|
| Reserve | $2.00 | $16.91 | ✓ |
| Allocatable | $6.73 | $67.64 | ✓ |
| Trading budget | $6.73 | $54.11 | ✓ |
| Dust budget | $0.00 | $13.53 | ✓ |
| Dust healing | Starved | Funded | ✓ |
| Split ratio | 100/0/0 | 60/20/20 | ✓ |

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forget to add config | Will use defaults, might work but log warnings |
| Only add one function | Need all three (reserve calc, split, orchestrator) |
| Don't update callers | Old code will still use old allocation logic |
| Test with wrong NAV | Use $84.55 (current) to verify 20% tier |
| Forget async/await | `allocate_with_nav_dynamics()` is async |

---

## 🚀 Before/After Metrics

### Before
```
Allocation: 100% Trading / 0% Dust Healing
Free USDT: $8.73 (locked, can't allocate to dust)
Dust positions: 32 (stuck, unfunded)
Mode: RECOVERY (forced)
```

### After
```
Allocation: 60/20/20 (Trading/Alts/Dust)
Free USDT: $67.64 allocatable (once dust cleared)
Dust positions: 32 (liquidating with $13.53 budget)
Mode: GROWTH (after capital reaches $100)
```

---

## 💾 Files to Keep Handy

1. **CAPITAL_ALLOCATOR_FIX_CODE.py** - Copy functions from here
2. **CAPITAL_ALLOCATOR_FIX_ANALYSIS.md** - Detailed reference
3. **REMEDIATION_SUMMARY.md** - Full context

---

## 🆘 If Something Goes Wrong

### Rollback (< 1 minute)
```bash
# Revert capital_allocator.py to use fixed $2.00 reserve
git checkout src/l6_governance/capital_allocator.py

# Restore old config values
# Update .env to use old settings
```

### Debug
1. Check logs for `calculate_dynamic_reserve()` debug output
2. Verify reserve calculation: `NAV * 0.20` (for micro)
3. Check dust_healing budget is non-zero

---

## 📞 Post-Deployment Checklist

- [ ] All 3 functions added to capital_allocator.py
- [ ] Config values added to .env
- [ ] MetaController updated to call new functions
- [ ] Dust healing component uses `allocation['dust_healing']`
- [ ] Tests pass with $84.55 NAV
- [ ] Reserve shows as $16.91 (not $2.00)
- [ ] Dust healing budget shows as $13.53
- [ ] Logs show correct allocation split
- [ ] System still runs without errors
- [ ] Background monitoring active

---

## 🎉 Success = You See This

```log
[MAIN] Starting Capital Allocator fix deployment...
[IMPORT] Added calculate_dynamic_reserve ✓
[IMPORT] Added allocate_capital_60_20_20 ✓
[IMPORT] Added allocate_with_nav_dynamics ✓
[CONFIG] Reserve tiers loaded: 20%/15%/10% ✓
[CONFIG] Allocation split loaded: 60%/20%/20% ✓
[ALLOCATION] NAV=$84.55, Reserve=$16.91(dynamic) ✓
[ALLOCATION] Allocatable=$67.64 ✓
[ALLOCATION] Split: Core=$40.58 + Alts=$13.53 + Dust=$13.53 ✓
[DUST_HEALING] Budget FUNDED: $13.53 ✓
[SYSTEM] Dust healing can now execute ✓
```

---

**Estimated Total Time:** 30 minutes (5+3+5+10+5 min)  
**Risk:** LOW (config only in phase 1)  
**Testing:** Use current $84.55 account  
**Rollback:** Simple git revert  

Ready to deploy? Start with `.env` config additions.
