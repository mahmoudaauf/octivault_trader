# Quick Reference: Model Trainer Debug & Improvements

## What Was Added

### 1️⃣ Training Label Distribution Debug
**Shows:** How many BUY vs HOLD signals exist in training data
```
[ML DEBUG] Label distribution for BTC/USDT: {0: 850, 1: 150}
```
- 0 = HOLD/SELL (negative class)
- 1 = BUY (positive class)
- Appears BEFORE class weight computation

### 2️⃣ Validation Label Distribution Debug  
**Shows:** How many BUY vs HOLD signals in validation split
```
[ML DEBUG] Validation distribution for BTC/USDT: {0: 85, 1: 18}
```
- Appears right after train/validation split
- Helps detect label leakage or imbalanced splits

### 3️⃣ Triple Barrier Method (Professional Standard)
**Instead of:** `future_return > 0.05%`
**Now uses:** Profit must exceed transaction costs + volatility buffer WITHIN N bars

**Calculation:**
```
threshold = fee(0.1%) + slippage(0.05%) + buffer(0.05%) + vol(ATR×0.5)
BUY if: profit_pct > threshold within 5 bars
```

**Result:** More realistic, achievable labels that match real trading

---

## How to Monitor

### In Logs, Look For:
```
1. [ML DEBUG] Label distribution for {symbol}: {0: X, 1: Y}
   → Training distribution (raw labels)

2. [ML DEBUG] Triple Barrier Labels: fee=0.001 slippage=0.0005 ...
   → Confirmation triple barrier is active
   → Shows actual distribution after filtering

3. [ML DEBUG] Validation distribution for {symbol}: {0: X, 1: Y}
   → Validation distribution (should be similar %)

4. Applied balanced class weights for {symbol}: {0: W1, 1: W2}
   → Class 1 (BUY) should have higher weight
```

---

## Configuration

### To Enable/Disable Triple Barrier:
```bash
# Enable (default - RECOMMENDED)
export ML_USE_TRIPLE_BARRIER_LABELS=true

# Disable (fallback to regime-aware)
export ML_USE_TRIPLE_BARRIER_LABELS=false
```

### To Adjust Costs (if your actual costs differ):
```bash
# Fee percentage per trade
export ML_TRIPLE_BARRIER_FEE_PCT=0.002

# Slippage/market impact
export ML_TRIPLE_BARRIER_SLIPPAGE_PCT=0.0010

# Safety buffer
export ML_TRIPLE_BARRIER_BUFFER_PCT=0.0010

# How many bars ahead to look for profit
export ML_TRIPLE_BARRIER_LOOKFORWARD_BARS=10
```

---

## Expected Log Flow

```
Starting training for BTC/USDT...
├─ Training with 5 features (OHLCV)
├─ [ML DEBUG] Label distribution: {0: 850, 1: 150}  ← Raw count
├─ Using Triple Barrier Labeling (improved method)
├─ [ML DEBUG] Triple Barrier Labels: ... dist={0: 820, 1: 180}  ← After filtering
├─ [ML DEBUG] Validation distribution: {0: 85, 1: 18}  ← Val set check
├─ Applied balanced class weights: {0: 0.54, 1: 3.06}  ← BUY weighted 6× higher
├─ Training progress model: epoch=1/15 loss=0.456 val_loss=0.412 ...
└─ Training complete. Model saved.
```

---

## Interpretation Guide

### Good Signs:
✅ BUY count increases after Triple Barrier (more realistic labels)
✅ Validation distribution % matches training (no leakage)
✅ Class weight for BUY > 1.0 (BUY gets more importance)
✅ Both train and val have reasonable accuracy (not 99%)

### Warning Signs:
⚠️ BUY count becomes 0 after Triple Barrier → costs too high
⚠️ Val distribution % very different from train → possible leakage
⚠️ Class weight < 1.0 for both classes → label math issue
⚠️ Training loss stays flat → imbalanced classes not working

---

## Files to Check

- `core/model_trainer.py` - Main implementation
  - Lines 70-79: Config parameters
  - Lines 222-281: Triple Barrier method
  - Lines 440-502: Label generation logic
  - Lines 545-548: Validation debug
  - Lines 520-525: Training debug

---

## Recovery / Rollback

If issues occur, you can revert labeling to previous method:
```bash
# Set environment variable
export ML_USE_TRIPLE_BARRIER_LABELS=false

# Or in code, change line ~77:
self.use_triple_barrier_labels = False
```

This will fall back to regime-aware → simple threshold method.

---

## Real Impact on Model

### Before Changes:
```
Threshold: 0.05%
Train labels: {0: 950, 1: 50} (5% BUY)
Class weight BUY: 0.05
Model learns: "Most things are HOLD"
Result: Model mostly predicts HOLD
```

### After Changes:
```
Triple Barrier: cost=0.2% + vol=1.0% = 1.2%
Train labels: {0: 820, 1: 180} (18% BUY)
Class weight BUY: 3.06
Model learns: "These are profitable BUYs"
Result: Model identifies edge cases
```

The key: **More realistic labels → Better trading**

---

## Support

If logs show unexpected distributions:
1. Check all three debug lines appear
2. Verify BUY count increases (not decreases) after Triple Barrier
3. Confirm validation % similar to training
4. Monitor actual trading results to see if profit improved
