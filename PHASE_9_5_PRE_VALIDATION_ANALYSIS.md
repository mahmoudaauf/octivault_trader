# Phase 9.5: Pre-Validation Analysis Layer
## Confidence Bucket Separation Test (Before Step 1)

**Status:** Pre-Validation Hypothesis Test  
**Date:** February 21, 2026  
**Purpose:** Determine if 30m horizon increases signal separation  
**Method:** Hedge fund-grade statistical analysis  

---

## 🎯 The Core Question

**Does longer horizon increase separation between high-confidence and low-confidence predictions?**

```
HYPOTHESIS:
  If 30m horizon is better than 5m,
  then confidence bucketing should show clearer separation.
  
  High confidence (>0.8) should have materially different
  forward move distribution than low confidence (<0.6).
  
  If separation is weak or absent,
  you're just stretching noise, not improving signal.
```

---

## 📊 Analysis Plan: Confidence Bucket Separation

### This is NOT the full Phase 9.5 validation.

This is the **hypothesis test** that determines if full validation is worth running.

**Timeline:** 2-3 hours (quick statistical check)  
**Decision Point:** Should we proceed to Step 1 (full validation)?

---

## 🔬 The Test Framework

### Step 0a: Load 5m OHLCV with ML Confidence Scores

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load your historical OHLCV data
df = pd.read_csv('your_ohlcv_data.csv', parse_dates=['time'])
df = df.sort_values('time').reset_index(drop=True)

# Load ML confidence scores
# (These are the existing Phase 8 signal confidences)
ml_scores = pd.read_csv('ml_forecaster_confidences.csv', parse_dates=['time'])

# Merge
df = df.merge(ml_scores[['time', 'confidence']], on='time', how='left')

print(f"Total rows: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Confidence range: {df['confidence'].min():.3f} to {df['confidence'].max():.3f}")
```

### Step 0b: Calculate Forward Returns (5m and 30m)

```python
def calculate_forward_returns(df):
    """
    Calculate forward returns at multiple horizons.
    This does NOT create targets yet.
    Just measures actual realized moves.
    """
    
    # 5m forward return (1 candle ahead)
    df['forward_return_5m'] = (
        df['close'].shift(-1) / df['close'] - 1
    )
    
    # 30m forward return (6 candles ahead)
    df['forward_return_30m'] = (
        df['close'].shift(-6) / df['close'] - 1
    )
    
    # Absolute moves (magnitude regardless of direction)
    df['abs_move_5m'] = df['forward_return_5m'].abs()
    df['abs_move_30m'] = df['forward_return_30m'].abs()
    
    # Remove last few rows (can't calculate forward return)
    df = df.iloc[:-6].copy()
    
    return df

df = calculate_forward_returns(df)

print("\nForward Return Statistics:")
print(f"\n5m Forward Returns:")
print(df['forward_return_5m'].describe())
print(f"\n30m Forward Returns:")
print(df['forward_return_30m'].describe())
```

---

## 🎯 The Key Analysis: Confidence Bucket Separation

### Step 1: Define Confidence Buckets

```python
def define_confidence_buckets(df):
    """
    Split data into three confidence buckets.
    """
    
    df['confidence_bucket'] = 'medium'
    df.loc[df['confidence'] < 0.60, 'confidence_bucket'] = 'low'
    df.loc[df['confidence'] > 0.80, 'confidence_bucket'] = 'high'
    
    # Count
    bucket_counts = df['confidence_bucket'].value_counts()
    
    print("=" * 70)
    print("CONFIDENCE BUCKET DISTRIBUTION")
    print("=" * 70)
    print(f"\nLow    (<0.60): {bucket_counts.get('low', 0):6d} ({bucket_counts.get('low', 0)/len(df)*100:5.1f}%)")
    print(f"Medium (0.60-0.80): {bucket_counts.get('medium', 0):6d} ({bucket_counts.get('medium', 0)/len(df)*100:5.1f}%)")
    print(f"High   (>0.80): {bucket_counts.get('high', 0):6d} ({bucket_counts.get('high', 0)/len(df)*100:5.1f}%)")
    print(f"Total:  {len(df):6d}")
    
    return df

df = define_confidence_buckets(df)
```

### Step 2: Analyze 5m Horizon (Phase 8 Baseline)

```python
def analyze_horizon_by_confidence(df, horizon_col, horizon_name):
    """
    For each confidence bucket:
    - Median move
    - 75th percentile move
    - Standard deviation
    - Separation ratio
    """
    
    print("\n" + "=" * 70)
    print(f"ANALYSIS: {horizon_name} Forward Returns")
    print("=" * 70)
    
    results = {}
    
    for bucket in ['low', 'medium', 'high']:
        bucket_data = df[df['confidence_bucket'] == bucket][horizon_col]
        
        if len(bucket_data) > 0:
            median = bucket_data.abs().median()
            p75 = bucket_data.abs().quantile(0.75)
            std = bucket_data.std()
            mean = bucket_data.mean()
            count = len(bucket_data)
            
            results[bucket] = {
                'median': median,
                'p75': p75,
                'std': std,
                'mean': mean,
                'count': count
            }
            
            print(f"\n{bucket.upper()} Confidence:")
            print(f"  Count:              {count}")
            print(f"  Mean:               {mean:+.4%}")
            print(f"  Median (abs):       {median:.4%}")
            print(f"  75th percentile:    {p75:.4%}")
            print(f"  Std Dev:            {std:.4%}")
    
    # Calculate separation metrics
    print(f"\n" + "-" * 70)
    print("SEPARATION ANALYSIS:")
    
    if 'low' in results and 'high' in results:
        median_ratio = results['high']['median'] / results['low']['median'] if results['low']['median'] > 0 else np.inf
        p75_ratio = results['high']['p75'] / results['low']['p75'] if results['low']['p75'] > 0 else np.inf
        
        print(f"\nMedian Separation Ratio (High/Low):")
        print(f"  {median_ratio:.2f}x")
        print(f"\n75th Percentile Ratio (High/Low):")
        print(f"  {p75_ratio:.2f}x")
        
        # Statistical test (t-test)
        low_moves = df[df['confidence_bucket'] == 'low'][horizon_col].abs()
        high_moves = df[df['confidence_bucket'] == 'high'][horizon_col].abs()
        
        t_stat, p_value = stats.ttest_ind(high_moves, low_moves)
        
        print(f"\nT-Test (High vs Low):")
        print(f"  t-statistic:        {t_stat:.3f}")
        print(f"  p-value:            {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  Result:             ✅ SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Result:             ⚠️  NOT SIGNIFICANT (p >= 0.05)")
    
    print("\n" + "=" * 70)
    
    return results

results_5m = analyze_horizon_by_confidence(df, 'forward_return_5m', '5m Horizon (Phase 8)')
results_30m = analyze_horizon_by_confidence(df, 'forward_return_30m', '30m Horizon (Phase 9.5)')
```

### Step 3: Compare Separation Between Horizons

```python
def compare_separation_quality(results_5m, results_30m):
    """
    Does 30m show BETTER separation than 5m?
    """
    
    print("\n" + "=" * 70)
    print("COMPARISON: 5m vs 30m Separation Quality")
    print("=" * 70)
    
    if 'high' in results_5m and 'low' in results_5m:
        sep_5m = results_5m['high']['median'] / results_5m['low']['median']
        print(f"\n5m Separation Ratio:   {sep_5m:.2f}x")
    else:
        sep_5m = 0
        print(f"\n5m Separation Ratio:   (insufficient data)")
    
    if 'high' in results_30m and 'low' in results_30m:
        sep_30m = results_30m['high']['median'] / results_30m['low']['median']
        print(f"30m Separation Ratio:  {sep_30m:.2f}x")
    else:
        sep_30m = 0
        print(f"30m Separation Ratio:  (insufficient data)")
    
    if sep_5m > 0 and sep_30m > 0:
        improvement = (sep_30m / sep_5m - 1) * 100
        print(f"\nImprovement:           {improvement:+.1f}%")
        
        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        
        if improvement > 20:
            print("✅ 30m shows SIGNIFICANTLY better separation")
            print("   Confidence bucket is more predictive at 30m")
            print("   Signal quality improves with longer horizon")
            print("   → PROCEED to Step 1 (full validation)")
        
        elif improvement > 5:
            print("⚠️  30m shows MODEST improvement in separation")
            print("   Some signal quality improvement visible")
            print("   But not compelling")
            print("   → CONSIDER running Step 1, but skeptical")
        
        else:
            print("❌ 30m shows LITTLE to NO separation improvement")
            print("   Confidence bucket equally weak at both horizons")
            print("   Likely stretching noise, not improving signal")
            print("   → DO NOT PROCEED to Step 1")
            print("   → Investigate different horizon (15m? 45m?)")

compare_separation_quality(results_5m, results_30m)
```

---

## 💰 Signal-to-Noise Ratio Analysis

### Step 4: SNR Comparison by Horizon

```python
def calculate_snr_by_confidence(df, horizon_col, horizon_name):
    """
    Signal-to-Noise Ratio for each confidence bucket.
    
    SNR = Signal / Noise
    where:
      Signal = Expected move (median of high confidence)
      Noise = Variability (std dev of low confidence)
    """
    
    print("\n" + "=" * 70)
    print(f"SIGNAL-TO-NOISE RATIO: {horizon_name}")
    print("=" * 70)
    
    low_data = df[df['confidence_bucket'] == 'low'][horizon_col].abs()
    high_data = df[df['confidence_bucket'] == 'high'][horizon_col].abs()
    
    if len(low_data) > 0 and len(high_data) > 0:
        signal = high_data.median()
        noise = low_data.std()
        snr = signal / noise if noise > 0 else 0
        
        print(f"\nSignal (high conf median):  {signal:.4%}")
        print(f"Noise (low conf std):       {noise:.4%}")
        print(f"SNR:                        {snr:.3f}")
        
        return snr
    
    return 0

snr_5m = calculate_snr_by_confidence(df, 'forward_return_5m', '5m Horizon')
snr_30m = calculate_snr_by_confidence(df, 'forward_return_30m', '30m Horizon')

print("\n" + "=" * 70)
print("SNR IMPROVEMENT:")
print("=" * 70)
if snr_5m > 0:
    snr_improvement = (snr_30m / snr_5m - 1) * 100
    print(f"\n5m SNR:    {snr_5m:.3f}")
    print(f"30m SNR:   {snr_30m:.3f}")
    print(f"Improvement: {snr_improvement:+.1f}%")
    
    if snr_improvement > 50:
        print("\n✅ SNR improves significantly with 30m horizon")
    elif snr_improvement > 10:
        print("\n⚠️  SNR improves modestly with 30m horizon")
    else:
        print("\n❌ SNR does not improve with 30m horizon")
```

---

## 💡 Expected Value Comparison by Confidence

### Step 5: EV Analysis

```python
def analyze_ev_by_confidence(df, horizon_col, cost_pct, horizon_name):
    """
    For each confidence bucket:
    - Calculate expected value
    - Can we actually gate trades with confidence?
    """
    
    print("\n" + "=" * 70)
    print(f"EXPECTED VALUE ANALYSIS: {horizon_name}")
    print("=" * 70)
    
    for bucket in ['low', 'medium', 'high']:
        bucket_data = df[df['confidence_bucket'] == bucket][horizon_col].abs()
        
        if len(bucket_data) > 0:
            median_move = bucket_data.median()
            edge = max(0, median_move - cost_pct)
            ev = edge / cost_pct if cost_pct > 0 else 0
            
            print(f"\n{bucket.upper()} Confidence ({bucket_data.shape[0]} samples):")
            print(f"  Median move:        {median_move:.4%}")
            print(f"  Cost:               {cost_pct:.4%}")
            print(f"  Edge:               {edge:+.4%}")
            print(f"  EV (edge/cost):     {ev:.2f}")
            
            if ev > 1.0:
                print(f"  Status:             ✅ Profitable")
            elif ev > 0.5:
                print(f"  Status:             ⚠️  Marginal")
            else:
                print(f"  Status:             ❌ Unprofitable")
    
    print("\n" + "=" * 70)

# 5m analysis (with 0.12% cost assumption)
analyze_ev_by_confidence(df, 'forward_return_5m', cost_pct=0.0012, horizon_name='5m Horizon')

# 30m analysis (with 0.15% cost assumption)
analyze_ev_by_confidence(df, 'forward_return_30m', cost_pct=0.0015, horizon_name='30m Horizon')
```

---

## 🎯 The Decision Checkpoint

### What to Look For

```
PROCEED TO STEP 1 IF:

✅ Separation Ratio (High/Low):
   30m separation > 5m separation by > 20%
   (Confidence bucket predicts 30m better)

✅ SNR Improvement:
   30m SNR > 5m SNR by > 50%
   (Signal rises above noise with longer horizon)

✅ EV by Confidence:
   High confidence bucket shows EV > 1.0 at 30m
   (Confidence actually gates profitable trades)

✅ Statistical Significance:
   t-test p-value < 0.05 for high vs low separation
   (Difference is real, not random)

STOP IF ANY OF:

❌ Separation Ratio worsens or unchanged
   (30m doesn't improve separation)

❌ SNR improves < 10%
   (Not much signal quality improvement)

❌ All confidence buckets show EV < 1.0
   (Edge doesn't materialize)

❌ High and low confidence buckets have similar moves
   (Confidence is useless at both horizons)
```

---

## 📋 Full Pre-Validation Script

```python
"""
PHASE 9.5 PRE-VALIDATION: Confidence Bucket Separation Test

This script answers: Does 30m horizon increase separation?

Run time: 30 minutes
Output: Statistical results guiding Step 1 decision
"""

import pandas as pd
import numpy as np
from scipy import stats

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('your_ohlcv_data.csv', parse_dates=['time'])
ml_scores = pd.read_csv('ml_forecaster_confidences.csv', parse_dates=['time'])

df = df.sort_values('time').reset_index(drop=True)
df = df.merge(ml_scores[['time', 'confidence']], on='time', how='left')

print("Data loaded:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

# ============================================================================
# CALCULATE FORWARD RETURNS
# ============================================================================

df['forward_return_5m'] = (df['close'].shift(-1) / df['close'] - 1)
df['forward_return_30m'] = (df['close'].shift(-6) / df['close'] - 1)
df['abs_move_5m'] = df['forward_return_5m'].abs()
df['abs_move_30m'] = df['forward_return_30m'].abs()

df = df.iloc[:-6].copy()

# ============================================================================
# DEFINE CONFIDENCE BUCKETS
# ============================================================================

df['confidence_bucket'] = 'medium'
df.loc[df['confidence'] < 0.60, 'confidence_bucket'] = 'low'
df.loc[df['confidence'] > 0.80, 'confidence_bucket'] = 'high'

# ============================================================================
# ANALYSIS SECTION 1: 5m HORIZON (BASELINE)
# ============================================================================

print("\n" + "="*70)
print("SECTION 1: 5M HORIZON (PHASE 8 BASELINE)")
print("="*70)

results_5m = {}
for bucket in ['low', 'medium', 'high']:
    data = df[df['confidence_bucket'] == bucket]['forward_return_5m'].abs()
    if len(data) > 0:
        results_5m[bucket] = {
            'median': data.median(),
            'p75': data.quantile(0.75),
            'std': data.std(),
            'count': len(data),
        }
        print(f"\n{bucket.upper()}:")
        print(f"  Median:  {results_5m[bucket]['median']:.4%}")
        print(f"  P75:     {results_5m[bucket]['p75']:.4%}")
        print(f"  Std:     {results_5m[bucket]['std']:.4%}")

# Separation
if 'high' in results_5m and 'low' in results_5m:
    sep_5m = results_5m['high']['median'] / results_5m['low']['median']
    print(f"\nSeparation (High/Low): {sep_5m:.2f}x")

# ============================================================================
# ANALYSIS SECTION 2: 30M HORIZON (PROPOSED)
# ============================================================================

print("\n" + "="*70)
print("SECTION 2: 30M HORIZON (PHASE 9.5 PROPOSED)")
print("="*70)

results_30m = {}
for bucket in ['low', 'medium', 'high']:
    data = df[df['confidence_bucket'] == bucket]['forward_return_30m'].abs()
    if len(data) > 0:
        results_30m[bucket] = {
            'median': data.median(),
            'p75': data.quantile(0.75),
            'std': data.std(),
            'count': len(data),
        }
        print(f"\n{bucket.upper()}:")
        print(f"  Median:  {results_30m[bucket]['median']:.4%}")
        print(f"  P75:     {results_30m[bucket]['p75']:.4%}")
        print(f"  Std:     {results_30m[bucket]['std']:.4%}")

# Separation
if 'high' in results_30m and 'low' in results_30m:
    sep_30m = results_30m['high']['median'] / results_30m['low']['median']
    print(f"\nSeparation (High/Low): {sep_30m:.2f}x")

# ============================================================================
# ANALYSIS SECTION 3: SEPARATION IMPROVEMENT
# ============================================================================

print("\n" + "="*70)
print("SECTION 3: SEPARATION IMPROVEMENT")
print("="*70)

if sep_5m > 0 and sep_30m > 0:
    sep_improvement = (sep_30m / sep_5m - 1) * 100
    print(f"\n5m Separation:   {sep_5m:.2f}x")
    print(f"30m Separation:  {sep_30m:.2f}x")
    print(f"Improvement:     {sep_improvement:+.1f}%")
    
    # Statistical test
    low_moves_5m = df[df['confidence_bucket'] == 'low']['forward_return_5m'].abs()
    high_moves_5m = df[df['confidence_bucket'] == 'high']['forward_return_5m'].abs()
    t_5m, p_5m = stats.ttest_ind(high_moves_5m, low_moves_5m)
    
    low_moves_30m = df[df['confidence_bucket'] == 'low']['forward_return_30m'].abs()
    high_moves_30m = df[df['confidence_bucket'] == 'high']['forward_return_30m'].abs()
    t_30m, p_30m = stats.ttest_ind(high_moves_30m, low_moves_30m)
    
    print(f"\nT-Test Results:")
    print(f"  5m (High vs Low):  t={t_5m:.2f}, p={p_5m:.6f} {'✅' if p_5m < 0.05 else '⚠️'}")
    print(f"  30m (High vs Low): t={t_30m:.2f}, p={p_30m:.6f} {'✅' if p_30m < 0.05 else '⚠️'}")

# ============================================================================
# DECISION
# ============================================================================

print("\n" + "="*70)
print("DECISION: SHOULD WE PROCEED TO STEP 1?")
print("="*70)

decision_score = 0

if sep_improvement > 20:
    print("\n✅ Separation improves > 20%")
    decision_score += 1
elif sep_improvement > 5:
    print("\n⚠️  Separation improves 5-20%")
    decision_score += 0.5
else:
    print("\n❌ Separation improves < 5%")

if p_30m < 0.05:
    print("✅ 30m separation is statistically significant")
    decision_score += 1
else:
    print("⚠️  30m separation lacks statistical significance")

snr_5m = results_5m['high']['median'] / results_5m['low']['std'] if results_5m['low']['std'] > 0 else 0
snr_30m = results_30m['high']['median'] / results_30m['low']['std'] if results_30m['low']['std'] > 0 else 0

if snr_30m / snr_5m > 1.5:
    print("✅ SNR improves > 50%")
    decision_score += 1
elif snr_30m / snr_5m > 1.1:
    print("⚠️  SNR improves 10-50%")
    decision_score += 0.5
else:
    print("❌ SNR improves < 10%")

print("\n" + "="*70)
print(f"Decision Score: {decision_score:.1f}/3")

if decision_score >= 2.5:
    print("\n🚀 RECOMMENDATION: PROCEED TO STEP 1 (Full Validation)")
    print("\n30m horizon shows clear advantages over 5m:")
    print("  • Better separation by confidence")
    print("  • Higher signal-to-noise ratio")
    print("  • Statistically significant differences")
elif decision_score >= 1.5:
    print("\n⚠️  RECOMMENDATION: CAUTIOUS PROCEED TO STEP 1")
    print("\n30m shows some promise but check carefully:")
    print("  • Investigate weak metrics further")
    print("  • Consider alternative horizons (15m, 45m)")
else:
    print("\n❌ RECOMMENDATION: DO NOT PROCEED TO STEP 1")
    print("\n30m horizon does not show compelling advantages:")
    print("  • Separation not better than 5m")
    print("  • SNR not significantly improved")
    print("  • Likely stretching noise")
    print("\nNext Steps:")
    print("  • Try 15m (3-candle) or 45m (9-candle)")
    print("  • Investigate data quality")
    print("  • Re-evaluate Phase 8 baseline")

print("\n" + "="*70)
```

---

## 📌 Key Metrics to Report Back

Once you run this pre-validation, report:

1. **Separation Ratio**
   - 5m: X.XXx
   - 30m: X.XXx
   - Improvement: +XX%

2. **SNR Comparison**
   - 5m: X.XXX
   - 30m: X.XXX
   - Improvement: +XX%

3. **Statistical Significance**
   - 5m p-value: X.XXXXX (significant? Y/N)
   - 30m p-value: X.XXXXX (significant? Y/N)

4. **EV by Confidence (30m)**
   - High confidence EV: X.XX
   - Medium confidence EV: X.XX
   - Low confidence EV: X.XX

5. **Final Recommendation**
   - Proceed to Step 1? YES/NO
   - Why?

---

## 🎯 This is the Hedge Fund Way

No enthusiasm. Just statistics.

Either 30m horizon shows clear advantages in this pre-validation, or you don't proceed.

That's the discipline that separates winners from noise traders.

