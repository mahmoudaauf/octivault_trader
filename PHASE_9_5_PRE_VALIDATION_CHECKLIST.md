# Phase 9.5: Pre-Validation Checklist
## Everything You Need Before Running the Analysis

---

## ✅ Pre-Validation Setup Checklist

### Data Preparation

- [ ] **OHLCV CSV File Ready**
  - Filename: `your_ohlcv_data.csv`
  - Columns: `time, open, high, low, close, volume`
  - Rows: Minimum 5,000 (prefer 10,000+)
  - Time format: ISO 8601 (e.g., `2024-01-01 00:00:00`)
  - Intervals: Consistent 5-minute gaps
  - Sorted: By time ascending
  - Quality check: No NaN values in `close`

- [ ] **ML Confidence CSV File Ready**
  - Filename: `ml_forecaster_confidences.csv`
  - Columns: `time, confidence`
  - Rows: Same length as OHLCV
  - Time format: Matches OHLCV exactly
  - Values: Between 0.0 and 1.0
  - Quality check: No NaN values
  - Source: Phase 8 MLForecaster signal emissions

### Environment Setup

- [ ] **Python Environment Ready**
  - Python 3.8+ installed
  - Required libraries installed:
    - pandas
    - numpy
    - scipy

Install with:
```bash
pip install pandas numpy scipy
```

- [ ] **Script Ready**
  - Copy full script from: `PHASE_9_5_PRE_VALIDATION_ANALYSIS.md`
  - Save as: `pre_validation_analysis.py`
  - Test import: `python -c "import pandas, numpy, scipy"`

### File Organization

```
your_working_directory/
├── pre_validation_analysis.py    (the script)
├── your_ohlcv_data.csv           (input data)
├── ml_forecaster_confidences.csv (input data)
└── pre_validation_results.txt    (output - generated)
```

---

## 🔧 Before Running: Verify Data Quality

### Quick Validation Script

```python
import pandas as pd

# Check OHLCV
df = pd.read_csv('your_ohlcv_data.csv')
print("OHLCV Data Check:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  NaN in close: {df['close'].isna().sum()}")
print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

# Check Confidence
conf = pd.read_csv('ml_forecaster_confidences.csv')
print("\nConfidence Data Check:")
print(f"  Rows: {len(conf)}")
print(f"  Confidence range: {conf['confidence'].min():.3f} to {conf['confidence'].max():.3f}")
print(f"  NaN values: {conf['confidence'].isna().sum()}")

# Check alignment
print(f"\nAlignment Check:")
print(f"  OHLCV rows: {len(df)}")
print(f"  Confidence rows: {len(conf)}")
print(f"  Match: {'✅ Yes' if len(df) == len(conf) else '❌ No'}")
```

Run this before proceeding:
```bash
python quick_check.py
```

---

## 🚀 Running the Analysis

### Step 1: Execute the Script

```bash
python pre_validation_analysis.py > pre_validation_results.txt 2>&1
```

This will:
- Load your data
- Run all analyses
- Generate output
- Save to `pre_validation_results.txt`

### Step 2: Monitor Output

You should see progress like:
```
Data loaded:
  Rows: 15000
  Columns: ['time', 'open', 'high', 'low', 'close', 'volume', 'confidence']

Calculating forward returns...
Defining confidence buckets...

SECTION 1: 5M HORIZON (PHASE 8 BASELINE)
...
```

If you see errors:
- Check file paths
- Verify CSV format
- Check column names match

### Step 3: Check Results

Results file will contain:
- SECTION 1: 5m analysis
- SECTION 2: 30m analysis
- SECTION 3: Comparison
- DECISION: PROCEED/STOP

---

## 📊 What to Look For in Results

### The 5 Key Metrics

After running, find these sections:

#### 1. Separation Ratio

```
5m Separation (High/Low): X.XXx
30m Separation (High/Low): X.XXx
Improvement: +XX.X%

✅ Pass: Improvement > 20%
❌ Fail: Improvement ≤ 20%
```

#### 2. SNR Comparison

```
5m SNR: X.XXX
30m SNR: X.XXX
Improvement: +XX.X%

✅ Pass: Improvement > 50%
❌ Fail: Improvement ≤ 50%
```

#### 3. Statistical Significance

```
5m (High vs Low): t=X.XX, p=X.XXXXX
30m (High vs Low): t=X.XX, p=X.XXXXX

✅ Pass: Both p < 0.05
❌ Fail: Either p ≥ 0.05
```

#### 4. EV by Confidence

```
30m HIGH confidence: EV = X.XX
30m MEDIUM confidence: EV = X.XX
30m LOW confidence: EV = X.XX

✅ Pass: High conf EV > 1.0
❌ Fail: High conf EV ≤ 1.0
```

#### 5. Decision Score

```
Decision Score: X.X/3

✅ Pass: Score ≥ 2.5
⚠️  Caution: Score 1.5-2.4
❌ Fail: Score < 1.5
```

---

## 🎯 Decision Rules (Copy-Paste)

### If Results Show:

```
┌─────────────────────────────────────┐
│ METRIC              │ PASS          │
├─────────────────────────────────────┤
│ Separation > 20%    │ ✅            │
│ SNR > 50%           │ ✅            │
│ p < 0.05            │ ✅            │
│ High EV > 1.0       │ ✅            │
│ Decision Score > 2.5│ ✅            │
└─────────────────────────────────────┘

DECISION: 🚀 PROCEED TO STEP 1
```

### If Results Show:

```
┌─────────────────────────────────────┐
│ METRIC              │ FAIL          │
├─────────────────────────────────────┤
│ Separation < 20%    │ ❌            │
│ SNR < 50%           │ ❌            │
│ p > 0.05            │ ❌            │
│ High EV < 1.0       │ ❌            │
│ Decision Score < 1.5│ ❌            │
└─────────────────────────────────────┘

DECISION: ⛔ STOP - DO NOT PROCEED
Next: Investigate, try different horizon
```

---

## 📝 Report Template

After running, fill this out:

```
╔════════════════════════════════════════════════════════════════════╗
║                    PRE-VALIDATION RESULTS                          ║
╚════════════════════════════════════════════════════════════════════╝

Data Used:
  OHLCV rows: ________
  Date range: ________ to ________
  Confidence min/max: ________ to ________

Results:

1. SEPARATION RATIO
   5m (High/Low):   _______x
   30m (High/Low):  _______x
   Improvement:     _______%
   
2. SIGNAL-TO-NOISE RATIO
   5m SNR:          _______
   30m SNR:         _______
   Improvement:     _______%

3. STATISTICAL TEST (p-values)
   5m p-value:      _______  (significant? Y/N)
   30m p-value:     _______  (significant? Y/N)

4. EXPECTED VALUE (30m horizon)
   High conf:       _______ (profitable? Y/N)
   Medium conf:     _______
   Low conf:        _______

5. DECISION
   Decision Score:  _______ / 3
   Proceed to Step 1? 
   
   ☐ YES (Score ≥ 2.5)
   ☐ MAYBE (Score 1.5-2.4) 
   ☐ NO (Score < 1.5)

Why?
[Your assessment]

════════════════════════════════════════════════════════════════════════

```

---

## 🛠️ Troubleshooting

### Error: "FileNotFoundError"

```
Error: No such file or directory: 'your_ohlcv_data.csv'
```

**Fix:**
- Check filename matches exactly
- Ensure file is in same directory as script
- Verify file extension is `.csv`

### Error: "KeyError: 'close'"

```
Error: 'close' is not in columns
```

**Fix:**
- Check CSV has column named `close` (exact case)
- Verify column headers: `time, open, high, low, close, volume`
- Open CSV in text editor, check first line

### Error: "shape mismatch"

```
Error: Cannot merge - shape mismatch
```

**Fix:**
- OHLCV and Confidence files must have same number of rows
- Check both have 5000+ rows
- Ensure time columns align exactly

### Error: "No data" or empty results

```
Warning: No data points in bucket
```

**Fix:**
- Check confidence distribution: should have mix of low/medium/high
- If all confidence > 0.80, can't test separation
- May need to adjust bucket thresholds (currently 0.60/0.80)

---

## ✅ Final Pre-Flight Checklist

Before running, verify:

- [ ] OHLCV CSV file exists
- [ ] Confidence CSV file exists
- [ ] Both files have 5000+ rows
- [ ] Column names match exactly
- [ ] Time columns align
- [ ] Python environment ready
- [ ] Script copied and saved
- [ ] Working directory set correctly
- [ ] Output file not already open
- [ ] Ready to wait 15-20 minutes for results

---

## 🚦 Next Steps After Pre-Validation

### If PASS (Score ≥ 2.5):

1. ✅ Review results carefully
2. ✅ Confirm separation improvement > 20%
3. ✅ Confirm SNR improvement > 50%
4. ✅ Confirm statistical significance (p < 0.05)
5. ✅ Save results for documentation
6. ✅ **PROCEED TO STEP 1** (Full Validation Framework)

### If FAIL (Score < 1.5):

1. ❌ Analyze which metrics failed
2. ❌ Consider data quality issues
3. ❌ Try different horizon (15m or 45m instead of 30m)
4. ❌ Check Phase 8 confidence scores validity
5. ❌ **DO NOT PROCEED** until pre-validation passes

### If MAYBE (Score 1.5-2.4):

1. ⚠️ Review borderline metrics
2. ⚠️ Check for data quality issues
3. ⚠️ Consider running on larger dataset
4. ⚠️ Try different confidence bucket thresholds
5. ⚠️ **CAUTIOUS PROCEED** only if you understand trade-offs

---

## 📞 If You Get Stuck

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| Script won't run | Check Python installation, verify imports |
| Data files not found | Verify file paths, use absolute paths |
| Wrong results | Check data quality, verify column alignment |
| Takes too long | Normal for 10000+ rows, give it 30 minutes |
| Can't interpret results | Compare to examples in decision rules section |

---

## ✨ You're Ready

You have:
- ✅ Pre-validation framework document
- ✅ Full Python script (copy-paste ready)
- ✅ Data requirements specification
- ✅ Expected output examples
- ✅ Decision rules
- ✅ Report template
- ✅ Troubleshooting guide

**Next Action:**

1. Gather your OHLCV and confidence CSV files
2. Run the script
3. Report the 5 metrics
4. We decide: PROCEED or PIVOT

That's how hedge funds test structural changes.

With discipline, not enthusiasm.

