# 📊 TP Profitability Gate - Quick Reference

## The Problem
Your logs showed: **"Expected move = 0.99%"** but this doesn't account for fees.

**Economics:** If you need 0.7% to cover friction (fees + slippage), a 0.99% move = only 0.29% actual profit.

---

## The Fix

### Updated MIN_PROFITABLE_MOVE_PCT by Account Size

```
ACCOUNT SIZE              OLD     NEW     TARGET TP RANGE
─────────────────────────────────────────────────────────
< $1,000   (MICRO)        0.55%   2.0%    1.8% - 2.5%
$1-5K      (STANDARD)     0.55%   1.2%    1.2% - 1.5%
≥ $5,000   (MULTI)        0.55%   0.8%    0.8% - 1.0%
```

---

## What Changed

**File:** `/core/nav_regime.py` (lines 100-171)

All 3 regime classes (MicroSniperConfig, StandardConfig, MultiAgentConfig) now have realistic minimum profitable move thresholds that account for **0.7% total friction** (fees + slippage).

---

## Expected Behavior

### Before Fix
```
Signal: Expected move 0.99%
Decision: ✅ ACCEPT (0.99% > 0.55% threshold)
Result: Take trade, earn 0.29% after fees
Problem: ❌ Fees dominate the profit
```

### After Fix (MICRO account)
```
Signal: Expected move 0.99%
Decision: ❌ REJECT (0.99% < 2.0% threshold)
Reason: move=0.99% < profitable_min=2.0% (fees will dominate)
Benefit: ✅ Wait for signals with real edge (2%+ moves)
```

---

## Economics Explained

### Total Transaction Friction
```
Entry Fee:     0.2% (Binance taker)
Exit Fee:      0.2% (Binance taker)
Slippage:      0.3% (market impact + spread)
───────────────────────
TOTAL:         0.7% friction
```

### Break-Even Analysis
```
If your signal targets a 2% TP move:
  Actual profit = 2.0% move - 0.7% friction = 1.3% net gain ✅
  
If your signal targets a 0.99% TP move:
  Actual profit = 0.99% move - 0.7% friction = 0.29% net gain ❌
  (Risk/reward ratio becomes terrible)
```

---

## Files Modified

✅ `/core/nav_regime.py`
- MicroSniperConfig: 0.55% → **2.0%**
- StandardConfig: 0.55% → **1.2%**
- MultiAgentConfig: 0.55% → **0.8%**

---

## Impact Summary

| Metric | Impact |
|--------|--------|
| **Trade Quality** | ↑ Only high-edge signals accepted |
| **Trade Count** | ↓ Fewer trades (intentional) |
| **Avg Win Size** | ↑ Larger average profit per trade |
| **Win Rate** | ← Unchanged (filter-independent) |
| **PnL/Trade** | ↑ Significantly better |

---

## Validation

After deployment, verify:

1. **Check logs** for rejection messages:
   ```
   grep "fees will dominate" logs/
   ```

2. **Expect to see:**
   - MICRO accounts: More rejections on 0.8%-1.5% moves
   - STANDARD: Some rejections on 0.6%-1.1% moves
   - MULTI: Rare rejections (threshold is low)

3. **Result:**
   - ✅ Fewer total trades
   - ✅ Higher average profit per trade
   - ✅ No fee-dominated losses

---

## Key Insight

**The old 0.55% threshold was economically impossible for small accounts.**

- On a $100 account, even a $1 move (1%) only nets $0.30 after fees
- The engine was literally trying to trade fractions of cents
- This fix forces realistic profit targets based on account size

**Now: Small accounts wait for 2%+ signals. Larger accounts can trade tighter.**

