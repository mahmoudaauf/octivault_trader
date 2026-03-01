# PHASE 9.5 VALIDATION STEP 1: EXECUTION GUIDE
## Construct 30m Cumulative Return Labels

---

## 📋 WHAT THIS SCRIPT DOES

**Input:** 5m OHLCV data (your existing CSV files)
```
data/historical/BTCUSDT_5m.csv
data/historical/ETHUSDT_5m.csv
...
```

**Process:** For each timestep, looks forward 6 candles (30m) and calculates:
```
cumulative_return = (close[i+6] - close[i]) / close[i]
target = 1 if cumulative_return > 0.20% else 0
```

**Output:** CSV files with added columns:
```
forward_return_30m: The actual realized return over next 30m
target_30m:        Binary label (1 = profitable, 0 = not profitable)
```

**Time Required:** 5-10 minutes execution

---

## 🚀 HOW TO RUN

### Step 1: Verify Data Exists
Check that you have 5m OHLCV files in `data/historical/`:

```zsh
ls -lh data/historical/*_5m.csv
```

Expected output:
```
data/historical/BTCUSDT_5m.csv
data/historical/ETHUSDT_5m.csv
data/historical/SOLUSDT_5m.csv
...
```

If files don't exist, you need to generate them first using your data loader.

### Step 2: Run the Script

From the octivault_trader directory:

```zsh
python step1_construct_30m_labels.py
```

**Expected output:**
```
================================================================================
PHASE 9.5 VALIDATION - STEP 1: CONSTRUCT 30M LABELS
================================================================================

Processing symbols: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

================================================================================
Processing: BTCUSDT
================================================================================
✅ Loaded 10000 rows for BTCUSDT
Constructing 30m labels (horizon=6 candles, threshold=0.0020)...
✅ Created 9994 valid labels

================================================================================
30M CUMULATIVE RETURN ANALYSIS - BTCUSDT
================================================================================

Total observations:      9994

Return Statistics:
  Mean:                    0.0015%
  Median:                  0.0008%
  Std Dev:                 0.4521%
  Min:                    -5.2341%
  Max:                     6.7823%

...

Target Distribution (threshold=0.2000%):
  Positive (> threshold):   35.2%
  Negative (≤ threshold):   64.8%

Validation: ✅ REALISTIC (30-40% positive expected)

Absolute Move Distribution:
  ...

Expected Move Validation (√6 × 0.15%):
  Theoretical:             0.3674%
  Actual Median:           0.4125%
  Ratio:                   1.12x

Validation: ✅ VIABLE (supports 30m horizon)
================================================================================
```

### Step 3: Check Output Files

```zsh
ls -lh validation_outputs/
```

You should see:
```
BTCUSDT_5m_with_30m_labels.csv       (large file with labeled data)
BTCUSDT_30m_label_analysis.json      (summary statistics)
ETHUSDT_5m_with_30m_labels.csv
ETHUSDT_30m_label_analysis.json
...
step1_results.json                   (overall results)
```

---

## 📊 DECISION RULES: WHEN DO WE PASS STEP 1?

Check the output for these metrics:

### ✅ PASS Criteria (All must be TRUE):

1. **Target Distribution:**
   ```
   Need: 25% < positive_ratio < 45%
   Why: If too few positives (<25%), threshold too high
        If too many positives (>45%), threshold too low
   ```

2. **Absolute Move Distribution:**
   ```
   Need: median_absolute_move >= 0.30%
   Why: √6 × 0.15% = 0.37%, but 0.30% is acceptable floor
   ```

3. **Data Quality:**
   ```
   Need: Successfully constructed labels for your symbols
        No NaN or error messages
   ```

### ❌ FAIL Criteria (Any of these is a blocker):

1. **Scripts fails with error**
   - Check data file format: needs columns [timestamp, open, high, low, close, volume]
   - Check CSV files aren't corrupted
   - Run with smaller dataset first (e.g., 1000 rows)

2. **Target distribution < 20% positive**
   - Edge threshold (0.20%) is too high
   - Market isn't moving enough in 30m
   - Solution: Try lower threshold (0.10%) or longer horizon (45m)

3. **Target distribution > 50% positive**
   - Edge threshold is too low
   - Almost everything is "positive"
   - Solution: Try higher threshold (0.30%) or shorter horizon (15m)

4. **Median absolute move < 0.25%**
   - 30m isn't enough time for meaningful moves
   - Solution: Try longer horizon (45m or 60m)

---

## 🔍 WHAT TO CHECK IN THE RESULTS

### File 1: JSON Analysis
```zsh
cat validation_outputs/BTCUSDT_30m_label_analysis.json
```

Look for:
```json
{
  "total_observations": 9994,
  "mean": 0.00148,
  "median": 0.00082,
  "std": 0.00452,
  "target_distribution": {
    "negative_or_equal": 0.648,  // ← Need between 0.55-0.75
    "positive": 0.352             // ← Need between 0.25-0.45
  },
  "absolute_move_stats": {
    "median": 0.00413,             // ← Need >= 0.0030
    ...
  }
}
```

### File 2: CSV Data Sample
```zsh
head validation_outputs/BTCUSDT_5m_with_30m_labels.csv
```

Should show:
```
timestamp,open,high,low,close,volume,forward_return_30m,target_30m,forward_return_valid
1234567890,40000.50,40050.00,39950.00,40025.00,125.3,0.00245,1,True
1234568490,40025.00,40075.00,40000.00,40050.00,118.7,0.00089,0,True
...
```

---

## 💡 INTERPRETATION GUIDE

### If You SEE This → LIKELY RESULT

**Example 1: Good Results**
```
Target Distribution:
  Positive: 35.2%        ✅ Perfect (30-40% range)
Absolute Move:
  Median: 0.413%         ✅ Excellent (> 0.30%)
Decision: PASS → Proceed to Step 2
```

**Example 2: Marginal Results (Proceed with caution)**
```
Target Distribution:
  Positive: 24.8%        ⚠️ Slightly low (just below 25%)
Absolute Move:
  Median: 0.32%          ⚠️ Okay (above 0.30% floor)
Decision: MAYBE → Proceed to Step 2 but re-evaluate
```

**Example 3: Fail Results (Stop and retry)**
```
Target Distribution:
  Positive: 52.3%        ❌ Too high (above 45%)
Absolute Move:
  Median: 0.28%          ❌ Too low (below 0.30%)
Decision: FAIL → Adjust threshold and retry
```

---

## 🔧 TROUBLESHOOTING

### Issue: Script fails to find data files

**Error:** `Data directory data/historical not found`

**Fix:**
```zsh
# Check where your data actually is
find . -name "*_5m.csv" | head -5

# Update the config in step1_construct_30m_labels.py
# Find this line (around line 95):
    data_dir: str = "data/historical"
# Change to:
    data_dir: str = "your/actual/data/path"
```

### Issue: Script runs but produces no labels

**Error:** `Only 10 valid labels, need 100`

**Fix:**
- Your CSV file is too small
- Solution: Use larger dataset (10,000+ rows recommended)

### Issue: Target distribution is wildly off

**Options:**
1. **Too few positive (< 20%):** Lower the edge_threshold_pct
   ```python
   edge_threshold_pct: float = 0.0010   # Try 0.10% instead of 0.20%
   ```

2. **Too many positive (> 50%):** Raise the edge_threshold_pct
   ```python
   edge_threshold_pct: float = 0.0030   # Try 0.30% instead of 0.20%
   ```

3. **Absolute move too small:** Try longer horizon
   ```python
   prediction_horizon_candles: int = 9   # Try 45m instead of 30m
   ```

### Issue: Memory error with large files

**Fix:**
- Process symbols one at a time
- Or modify script to downsample data

---

## 📈 WHAT HAPPENS NEXT

### If Step 1 PASSES:
→ Proceed to **Step 2: Expected Move Validation**
  - Verify forward move distribution matches theory
  - Confirm break-even probability calculation

### If Step 1 FAILS:
→ **Stop here and investigate**
  1. Review output statistics
  2. Adjust threshold or horizon
  3. Re-run Step 1
  4. Repeat until passing

### If Step 1 Shows MARGINAL Results:
→ **Proceed cautiously to Step 2**
  1. Collect more data if possible
  2. Test on different symbol
  3. Be skeptical of final results

---

## 📝 REPORTING RESULTS

Once Step 1 completes, collect:

```
Symbol: BTCUSDT

1. Total Valid Labels: ___________
2. Target Distribution (Positive): ____%
3. Median Absolute Move: ______%
4. Target Distribution Range: [___%, ____%]
5. Pass/Fail/Maybe: _________

Decision: Proceed to Step 2? YES / NO / MAYBE
```

Then proceed to Step 2 if results look good.

---

## ⏱️ TIMELINE

- **Setup:** 2 minutes
- **Execution:** 5-10 minutes (depends on data size)
- **Analysis:** 3-5 minutes
- **Total:** ~15 minutes

**If you need to retry:** Add 10-15 minutes per retry

---

## 🎯 KEY DECISION POINT

**Question:** Can we construct valid 30m cumulative return labels?

**Success Metric:** 
- Labels exist ✅
- Distribution is realistic (25-45% positive) ✅
- Forward moves are meaningful (>= 0.30% median) ✅

**Next Gate:** Step 2 validates that the ML model can actually learn this pattern.

---

Ready to run?

```zsh
python step1_construct_30m_labels.py
```

Come back with the results from:
- Target distribution percentage (positive)
- Median absolute move
- Pass/Fail status

Then we move to Step 2.
