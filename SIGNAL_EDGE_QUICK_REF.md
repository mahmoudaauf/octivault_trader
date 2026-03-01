# 🎯 Signal Edge Tuning Quick Reference

## One-Liner Summary
**Realized Edge = Price Movement % - Roundtrip Cost % (0.12% default)**

## Thresholds

```
realized_edge > 0.4%  → ⚠️  TOO_CONSERVATIVE
              0.2-0.4% → ✅  OPTIMAL  
              < 0.2%  → ❌  INSUFFICIENT
```

## What to Do

### ⚠️ TOO_CONSERVATIVE (edge > 0.4%)
You're leaving money on the table. Model says "not confident" too often.

**Quick Fix:**
```python
# Lower confidence requirement
# Option 1: In config
confidence_threshold = 0.65  # was 0.70

# Option 2: In model
# Retrain with relaxed output threshold
```

### ✅ OPTIMAL (edge 0.2-0.4%)
Model is well-calibrated. Keep current settings.

### ❌ INSUFFICIENT (edge < 0.2%)
Costs exceed potential gains. Model is either weak or entry timing is bad.

**Quick Fix:**
```python
# Option 1: Don't trade this asset
# In agent config
DISABLED_SYMBOLS = {'ETHUSDT'}  # Skip until retrained

# Option 2: Retrain with better features
# Add lookback, add indicators, add regime context
lookback = 50  # was 20
features = ['close', 'rsi', 'atr', 'vwap']  # add more
```

## Reading the Logs

```
[SIGNAL_OUTCOME:5m]  SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅ agent=MLForecaster
[SIGNAL_OUTCOME:15m] SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅ agent=MLForecaster
[SIGNAL_OUTCOME:30m] SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅ agent=MLForecaster
[SIGNAL_TUNING]     SYMBOL avg_edge=X.XX% → RECOMMENDATION
```

**Fields:**
- `ret` = Actual price movement (market paid)
- `cost` = Roundtrip trading cost (what we pay)
- `edge` = Profit leftover after costs (what we keep)
- `conf` = Model confidence in signal
- `✅/⚠️/❌` = Tuning assessment

## Tuning Workflow

1. **Collect 100+ signals** (let system run 1-2 weeks)
2. **Identify pattern**
   ```bash
   # Too many ⚠️ → too conservative
   # Too many ❌ → insufficient edge
   # Mixed ✅ → model well-tuned
   ```
3. **Apply ONE change** (don't change multiple things at once)
4. **Collect 30+ more signals** with new setting
5. **Compare metrics** — repeat from step 3

## Example Tuning Session

### Day 1: Baseline
```
Signal 1-50:  25 ⚠️ TOO_CONSERVATIVE, 20 ✅ OPTIMAL, 5 ❌ INSUFFICIENT
Average edge: 0.38% → TOO CONSERVATIVE
Action: Lower confidence_threshold from 0.70 to 0.65
```

### Day 2: First Adjustment
```
Signal 51-80:  10 ⚠️ TOO_CONSERVATIVE, 18 ✅ OPTIMAL, 2 ❌ INSUFFICIENT
Average edge: 0.28% → Still slightly conservative
Action: Lower confidence_threshold from 0.65 to 0.60
```

### Day 3: Second Adjustment
```
Signal 81-110: 5 ⚠️ TOO_CONSERVATIVE, 22 ✅ OPTIMAL, 3 ❌ INSUFFICIENT
Average edge: 0.24% → OPTIMAL
Action: Keep confidence_threshold at 0.60, monitor
```

## Config Locations

**MLForecaster tuning:**
```python
# agents/ml_forecaster.py
self.confidence_threshold = 0.70  # Lower to be more aggressive
self.sentiment_threshold = -0.5   # Relax sentiment gate
self.volatility_regime_block = True  # Disable to trade more often
```

**MetaController tuning:**
```python
# core/meta_controller.py
TIER_A_CONF = 0.80  # Lower for more tier-A trades
TIER_B_CONF = 0.50  # Lower for more tier-B trades
MIN_AGENTS = 1      # Lower for single-agent consensus
```

## Red Flags

🚩 **Negative edge** — Signal consistently loses vs fees
- Don't trade this pair
- Retrain model immediately

🚩 **Extreme edge (>1%)** — Too good to be true
- Check for data quality issues
- Verify price feeds aren't stale
- May indicate look-ahead bias in backtest

🚩 **High volatility in edge** (std > 0.2%)
- Model inconsistent across market conditions
- May need regime-specific tuning
- Consider adding regime awareness to features

## Remember

> **In trading, edge is everything. Measure it. Optimize it. Repeat.**

This system helps you see exactly how much edge your models generate. Use it.
