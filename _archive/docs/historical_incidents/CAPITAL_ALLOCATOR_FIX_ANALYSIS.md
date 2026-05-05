# 🔧 ROOT CAUSE FIX: Capital Allocator 60/20/20 Implementation

**Date:** May 4, 2026
**Issue:** Fixed $2.00 reserve breaks 60/20/20 allocation strategy
**Status:** READY TO IMPLEMENT

---

## 🔴 Root Cause Chain

### Problem #1: Fixed Reserve ($2.00)
**Location:** `src/l6_governance/capital_allocator.py:766`

```python
# CURRENT (BROKEN):
default_bootstrap_reserve = 2.0  # EMERGENCY: Reduced from 15.0
bootstrap_reserve_usdt = float(self._cfg("BOOTSTRAP_RESERVE_USDT", default_bootstrap_reserve))
```

**Impact:**
- For $84.55 NAV account: Locks $2.00 for reserve
- Leaves $82.55 for allocation
- But system behaves like it's locking MUCH more
- Why? Dust positions ($75.82) are not liquid

---

### Problem #2: No Dust Healing Allocation Tier
**Effect:** Dust healing operations fight with trading signals for same $7.73 pool

**Current allocation structure:**
- ✅ Trading signals get: Anything they want (no ceiling)
- ❌ Dust healing gets: Leftover (no floor)
- ❌ Reserve gets: Fixed $2.00 (no dynamics)

**Result:** 85/0/15 split (Trading/Dust/Reserve)

---

### Problem #3: Competing Capital Pools
**Issue:** Two concurrent processes pulling from same source:
1. **MetaController** → Trading signals (MLForecaster, SwingTradeHunter, etc.)
2. **DustHealing** → Recovery operations (forced liquidations)

**Conflict:** When both trigger simultaneously, smaller pool loses.

---

## ✅ Solution: Three-Part Fix

### PART 1: Dynamic Reserve Calculation

**Replace:**
```python
default_bootstrap_reserve = 2.0  # Fixed amount - WRONG
```

**With:**
```python
def calculate_dynamic_reserve(nav: float, cfg: dict) -> float:
    """
    NAV-based reserve tiers:
    - NAV < $50: 20% reserve
    - NAV $50-$200: 15% reserve
    - NAV > $200: 10% reserve
    """
    if nav < 50:
        reserve_pct = cfg.get("RESERVE_PCT_MICRO", 0.20)
    elif nav < 200:
        reserve_pct = cfg.get("RESERVE_PCT_SMALL", 0.15)
    else:
        reserve_pct = cfg.get("RESERVE_PCT_NORMAL", 0.10)

    dynamic_reserve = nav * reserve_pct
    min_reserve = cfg.get("RESERVE_MIN_USDT", 1.00)
    max_reserve = cfg.get("RESERVE_MAX_USDT", nav * 0.40)

    return max(min_reserve, min(max_reserve, dynamic_reserve))
```

**Example:**
- Current: Reserve = $2.00 (2.4% of $84.55)
- Dynamic: Reserve = $84.55 × 20% = $16.91 (micro tier)
- **Difference: +$14.91 earmarked for healing**

---

### PART 2: Implement 60/20/20 Split

**New allocation function:**
```python
def allocate_capital_60_20_20(
    free_usdt: float,
    cfg: dict
) -> dict:
    """
    Split allocatable capital (after reserve):
    - 60% to core trading (BTC/ETH)
    - 20% to alt trading (growth/emerging)
    - 20% to dust healing & recovery
    """
    allocatable = free_usdt  # Already after reserve

    return {
        'trading_core': allocatable * 0.60,
        'trading_alts': allocatable * 0.20,
        'dust_healing': allocatable * 0.20,
    }
```

**Example with $84.55 NAV:**
- Reserve: $16.91 (20% for micro tier)
- Allocatable: $67.64
- Trading Core: $40.58 (60%)
- Trading Alts: $13.53 (20%)
- Dust Healing: $13.53 (20%)

---

### PART 3: Orchestrator Integration

**Update MetaController to use new allocations:**

```python
# In MetaController or CapitalGovernor:

allocation = await self.capital_allocator.allocate_with_nav_dynamics(
    nav=current_nav,
    free_usdt=free_capital
)

# Now pass allocations to each component:
self.meta_controller.trading_budget = allocation['effective_trading']
self.dust_healer.budget = allocation['dust_healing']
self.reserve = allocation['reserve']
```

---

## 📊 Quantitative Impact

### Before Fix (Current State)
```
NAV: $84.55
Free: $8.73 (effective, after hidden locks)
Reserve: $2.00 (fixed)
Allocatable: $6.73 actual
Trading: $6.73 (100% of what's available)
Dust Healing: $0 (starved)
Split: 100/0/0 (broken)
```

### After Fix
```
NAV: $84.55
Free: $84.55 (all liquid once dust is cleared)
Reserve: $16.91 (20% of NAV, dynamic)
Allocatable: $67.64
Trading Core: $40.58 (60%)
Trading Alts: $13.53 (20%)
Dust Healing: $13.53 (20%)
Split: 60/20/20 (correct)
```

---

## 🔧 Implementation Steps

### Step 1: Update Config (.env or config.py)
```python
# Capital Allocator Configuration
RESERVE_PCT_MICRO = 0.20      # 20% for NAV < $50
RESERVE_PCT_SMALL = 0.15      # 15% for NAV < $200
RESERVE_PCT_NORMAL = 0.10     # 10% for NAV >= $200
RESERVE_MIN_USDT = 1.00       # Absolute minimum
RESERVE_MAX_USDT = None       # Will be 40% of NAV

ALLOC_PCT_CORE = 0.60         # 60% for BTC/ETH
ALLOC_PCT_ALTS = 0.20         # 20% for growth alts
ALLOC_PCT_DUST = 0.20         # 20% for dust healing
```

### Step 2: Modify capital_allocator.py
**Replace lines 766-790 with dynamic reserve calculation**

### Step 3: Add allocation orchestrator method
**New method `allocate_with_nav_dynamics()` returns complete allocation dict**

### Step 4: Update callers in meta_controller.py
**Use new `allocation['dust_healing']` budget for dust operations**

### Step 5: Test with current $84.55 NAV
**Verify:**
- ✅ Reserve calculated as 20% = $16.91
- ✅ Allocatable = $67.64
- ✅ Dust healing gets $13.53 budget
- ✅ Trading gets $54.11 budget

---

## 🚀 Expected Outcomes

### Immediate (After Fix)
1. **Dust healing gets floor budget:** $13.53 minimum
2. **Trading gets predictable budget:** $54.11 maximum
3. **Reserve is dynamic:** Scales with NAV

### Short Term (5-10 minutes)
1. Dust positions liquidated more aggressively (has budget)
2. Capital freed up faster
3. Free USDT crosses $10 threshold
4. System enters GROWTH mode

### Medium Term (1-2 hours)
1. 60/20/20 split observed in production
2. Stable trading with proper allocation
3. Systematic dust prevention

---

## ⚠️ Critical Notes

1. **Don't ship with fixed reserves:** Causes 85/0/15 splits
2. **Reserve must scale with NAV:** Not account size
3. **Dust healing needs floor budget:** Prevent starvation
4. **Test with current micro account:** $84.55 is good test case

---

## 📝 Code Location

- **File:** `src/l6_governance/capital_allocator.py`
- **Lines to replace:** 766-790 (fixed reserve logic)
- **New functions:** `calculate_dynamic_reserve()`, `allocate_capital_60_20_20()`
- **Integration:** `meta_controller.py` (allocation consumer)

---

**Status:** Ready for implementation
**Risk:** LOW (config-only in first phase)
**Rollback:** Simple config revert
**Testing:** Use current $84.55 account as test case
