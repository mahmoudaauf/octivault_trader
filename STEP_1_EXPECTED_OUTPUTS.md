# STEP 1: WHAT YOU'LL GET

This document shows exactly what Step 1 produces and how to interpret it.

## 📄 OUTPUT FILES

After running `python step1_construct_30m_labels.py`, you'll have:

```
validation_outputs/
├── BTCUSDT_5m_with_30m_labels.csv      ← Main output file
├── BTCUSDT_30m_label_analysis.json     ← Statistics summary
├── ETHUSDT_5m_with_30m_labels.csv
├── ETHUSDT_30m_label_analysis.json
├── [other symbols...]
└── step1_results.json                  ← Overall summary
```

---

## 📊 FILE 1: CSV WITH LABELS

**Filename:** `BTCUSDT_5m_with_30m_labels.csv`

This is your original 5m OHLCV data with 3 new columns added:

```csv
timestamp,open,high,low,close,volume,forward_return_30m,target_30m,forward_return_valid
1516064400,10800.00,10810.00,10790.00,10805.00,127.50,0.002456,1,True
1516065000,10805.00,10850.00,10800.00,10840.00,135.20,0.001234,1,True
1516065600,10840.00,10880.00,10820.00,10850.00,142.80,-0.000567,0,True
1516066200,10850.00,10900.00,10840.00,10880.00,138.90,0.000123,0,True
1516066800,10880.00,10950.00,10870.00,10920.00,156.30,0.003456,1,True
...
```

### Column Explanations:

| Column | Meaning | Example |
|--------|---------|---------|
| `timestamp` | Unix timestamp of candle open | 1516064400 |
| `open` | 5m candle open price | 10800.00 |
| `high` | 5m candle high price | 10810.00 |
| `low` | 5m candle low price | 10790.00 |
| `close` | 5m candle close price | 10805.00 |
| `volume` | 5m candle volume in quote asset | 127.50 |
| `forward_return_30m` | **NEW**: Actual 30m return (cumulative) | 0.002456 |
| `target_30m` | **NEW**: Binary label (1=positive) | 1 |
| `forward_return_valid` | **NEW**: Is this label valid? | True |

### How Labels Are Created:

For each row `i`:
```
entry_price = close[i]
exit_price = close[i + 6]  (30m = 6 × 5m candles)

forward_return_30m = (exit_price - entry_price) / entry_price
target_30m = 1 if forward_return_30m > 0.0020 else 0
forward_return_valid = True (always true for rows with valid forward data)
```

### Using This File:

```python
import pandas as pd

# Load labeled data
df = pd.read_csv('validation_outputs/BTCUSDT_5m_with_30m_labels.csv')

# You can now use this for ML training
X = df[['open', 'high', 'low', 'close', 'volume', ...]]  # Features
y = df['target_30m']  # Labels (1 = win, 0 = loss)

# Calculate win rate
win_rate = df['target_30m'].mean()
print(f"Win rate: {win_rate:.1%}")  # Should be ~30-40%
```

---

## 📋 FILE 2: JSON ANALYSIS SUMMARY

**Filename:** `BTCUSDT_30m_label_analysis.json`

Contains statistical summary of the labels. Example:

```json
{
  "total_observations": 9994,
  "mean": 0.0014823,
  "median": 0.0008234,
  "std": 0.00452341,
  "min": -0.052341,
  "max": 0.067823,
  "percentiles": {
    "p05": -0.008932,
    "p10": -0.006234,
    "p25": -0.002456,
    "p50": 0.0008234,
    "p75": 0.003456,
    "p90": 0.006789,
    "p95": 0.009234
  },
  "target_distribution": {
    "negative_or_equal": 0.648,
    "positive": 0.352
  },
  "absolute_move_stats": {
    "mean": 0.00412,
    "median": 0.00413,
    "std": 0.00245,
    "p25": 0.00234,
    "p50": 0.00413,
    "p75": 0.00623,
    "p90": 0.00912,
    "p95": 0.01234
  }
}
```

### Key Metrics to Check:

```
METRIC                          VALUE      NEED              STATUS
────────────────────────────────────────────────────────────────────
total_observations             9994       >= 100            ✅ PASS
Positive target ratio          35.2%      25-45%             ✅ PASS
Negative target ratio          64.8%      55-75%             ✅ PASS
Median absolute move           0.413%     >= 0.30%           ✅ PASS
Mean return                    0.148%     > 0%               ✅ PASS
Std dev of returns             0.452%     < 1%               ✅ PASS
────────────────────────────────────────────────────────────────────
Overall:                                   All checks pass    ✅ PASS
```

---

## 📈 FILE 3: OVERALL RESULTS SUMMARY

**Filename:** `step1_results.json`

Shows status for all symbols processed:

```json
{
  "BTCUSDT": {
    "status": "SUCCESS",
    "csv_path": "/path/to/BTCUSDT_5m_with_30m_labels.csv",
    "analysis_path": "/path/to/BTCUSDT_30m_label_analysis.json",
    "stats": {
      "total_observations": 9994,
      "target_distribution": {
        "positive": 0.352
      },
      "absolute_move_stats": {
        "median": 0.00413
      }
    }
  },
  "ETHUSDT": {
    "status": "SUCCESS",
    "csv_path": "/path/to/ETHUSDT_5m_with_30m_labels.csv",
    "analysis_path": "/path/to/ETHUSDT_30m_label_analysis.json",
    "stats": { ... }
  },
  "SOLUSDT": {
    "status": "FAILED",
    "error": "Historical data file not found"
  }
}
```

---

## 🔍 INTERPRETING THE RESULTS

### Example 1: EXCELLENT Results ✅✅✅

```
Target Distribution:
  Positive: 35.2%              ✅ Perfect (in 30-40% zone)
  
Absolute Move:
  Median: 0.413%               ✅ Excellent (well above 0.30%)
  
Statistics:
  Mean return: 0.148%          ✅ Positive
  Std dev: 0.452%              ✅ Reasonable volatility
  
Decision: PASS STEP 1
→ Proceed to Step 2 immediately
```

### Example 2: GOOD Results, Proceed ✅✅

```
Target Distribution:
  Positive: 28.5%              ✅ Acceptable (25-45% range)
  
Absolute Move:
  Median: 0.310%               ✅ Minimum viable (just above 0.30%)
  
Statistics:
  Mean return: 0.089%          ✅ Slightly positive
  
Decision: PASS STEP 1 (with caution)
→ Proceed to Step 2 but monitor carefully
→ Expect tighter margins on later steps
```

### Example 3: MARGINAL Results, Proceed Cautiously ⚠️

```
Target Distribution:
  Positive: 24.3%              ⚠️ Just below 25% threshold
  
Absolute Move:
  Median: 0.287%               ⚠️ Just below 0.30% threshold
  
Statistics:
  Mean return: -0.005%         ⚠️ Slightly negative
  
Decision: MAYBE
→ Could proceed to Step 2, but with high skepticism
→ Plan to retry with adjusted parameters
→ Collect more data if possible
```

### Example 4: FAILED Results ❌❌❌

```
Target Distribution:
  Positive: 52.3%              ❌ Too high (should be < 45%)
  
Absolute Move:
  Median: 0.245%               ❌ Too low (should be >= 0.30%)
  
Statistics:
  Mean return: 0.234%          ❌ Way too high (suggests bad distribution)
  
Decision: FAIL STEP 1
→ STOP here, do not proceed to Step 2
→ Investigate root cause
→ Adjust edge_threshold_pct or prediction_horizon_candles
→ Retry Step 1 with new parameters
```

---

## 🎯 SPECIFIC METRIC MEANINGS

### 1. Positive Target Ratio (35.2%)

**What it means:** Out of 9,994 forward 30m moves, 35.2% exceeded the 0.20% threshold.

**Why it matters:** 
- If too low (< 25%): Edge threshold is unrealistic for this market
- If too high (> 45%): Market is trending too much (edge is easy)
- Goldilocks zone (25-45%): Balanced problem

**If you see 60% positive:** Try raising edge threshold to 0.30% or 0.40%
**If you see 15% positive:** Try lowering edge threshold to 0.10% or 0.05%

### 2. Median Absolute Move (0.413%)

**What it means:** The typical 30m move (in absolute value) is 0.413%.

**Why it matters:**
- Theory predicts: √6 × 0.15% = 0.367%
- Reality: 0.413% (slightly higher than theory)
- Ratio: 0.413% / 0.367% = 1.12x (realistic)

**If you see 0.25%:** Moves too small, try longer horizon (45m or 60m)
**If you see 0.80%:** Moves too large, try shorter horizon (15m) or different market

### 3. Mean Return (0.148%)

**What it means:** Average 30m move is +0.148% (winners and losers combined).

**Why it matters:**
- Should be positive (market has slight upward bias in this period)
- Should be small relative to std dev
- Positive mean indicates tradeable edge exists

**If you see negative:** Market was bearish in this period, try different data

### 4. Standard Deviation (0.452%)

**What it means:** Typical deviation from mean return is 0.452%.

**Why it matters:**
- Shows variability/noise in the market
- Signal-to-noise ratio = mean / std = 0.148% / 0.452% = 0.33
- Lower is better (less noise means cleaner signal)

---

## 📊 WHAT HAPPENS NEXT

### If ALL metrics look good:
```
✅ Step 1: PASS
   ↓
→ Proceed to Step 2: Expected Move Validation
→ Train test model on 30m targets
→ Calculate break-even probability
```

### If SOME metrics are questionable:
```
⚠️ Step 1: PASS with concerns
   ↓
→ Proceed to Step 2 but watch carefully
→ Expect marginal results in later steps
→ Plan to re-run validation with adjusted parameters
```

### If metrics are BAD:
```
❌ Step 1: FAIL
   ↓
→ STOP: Do not proceed to Step 2
→ Analyze what went wrong
→ Options:
   a) Adjust edge_threshold_pct
   b) Try different prediction_horizon_candles
   c) Use different time period of data
   d) Check data quality
→ Re-run Step 1
```

---

## 💾 USING THE DATA FOR STEP 2

Once Step 1 passes, you'll use the CSV files for Step 2:

```python
# Step 2 code will do:
df = pd.read_csv('validation_outputs/BTCUSDT_5m_with_30m_labels.csv')

# Use forward_return_30m to validate expected move assumptions
absolute_moves = df['forward_return_30m'].abs()
print(f"Median move: {absolute_moves.median():.4%}")  # Should match our report

# Use target_30m to train ML model
y_true = df['target_30m']
# (Model predictions would be trained against this)
```

---

## 🚀 YOU'RE READY

Run Step 1:
```zsh
python step1_construct_30m_labels.py
```

Check the output files and come back with:
1. **Positive target ratio** (%) 
2. **Median absolute move** (%)
3. **Pass/Fail/Maybe status**

Then proceed to Step 2.
