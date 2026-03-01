# STEP 1: PRE-FLIGHT CHECKLIST

Before running `step1_construct_30m_labels.py`, verify:

## 📦 DATA REQUIREMENTS

- [ ] Data directory exists: `data/historical/`
- [ ] At least one 5m CSV file exists: `*_5m.csv`
- [ ] Each CSV has columns: `timestamp, open, high, low, close, volume`
- [ ] CSV files have >= 1000 rows each (more is better)

**Check:**
```zsh
ls -lh data/historical/*_5m.csv
head -3 data/historical/BTCUSDT_5m.csv
```

## 🐍 PYTHON ENVIRONMENT

- [ ] Python 3.8+ installed
- [ ] pandas library available
- [ ] numpy library available

**Check:**
```zsh
python --version
python -c "import pandas; print('pandas OK')"
python -c "import numpy; print('numpy OK')"
```

## 📁 DIRECTORIES

- [ ] `validation_outputs/` directory will be created automatically
- [ ] Write permissions in current directory
- [ ] Enough disk space for output files (~10MB per symbol)

## ⚙️ CONFIGURATION (in script)

These are set to defaults, modify if needed:

```python
data_dir: str = "data/historical"           # Where input CSV files are
output_dir: str = "validation_outputs"      # Where to save results

prediction_horizon_candles: int = 6         # 6 × 5m = 30m
edge_threshold_pct: float = 0.0020          # 0.20% minimum target return
min_data_points: int = 100                  # Minimum valid labels needed
```

## 🏃 READY TO EXECUTE

When all checkboxes above are marked:

```zsh
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python step1_construct_30m_labels.py
```

## ✅ EXPECTED SUCCESS INDICATORS

Look for these in the output:

```
✅ Loaded 10000 rows for BTCUSDT
✅ Created 9994 valid labels
✅ Saved labeled data: validation_outputs/BTCUSDT_5m_with_30m_labels.csv
✅ Saved analysis: validation_outputs/BTCUSDT_30m_label_analysis.json

30M CUMULATIVE RETURN ANALYSIS - BTCUSDT
================================================
Total observations:      9994

Target Distribution:
  Positive (> threshold):   35.2%    ← Need 25-45%
  Negative (≤ threshold):   64.8%

Validation: ✅ REALISTIC

Absolute Move Distribution:
  Median:                    0.4125%   ← Need >= 0.30%

Validation: ✅ VIABLE
```

## ❌ COMMON FAILURE MODES

**Error: "Historical data file not found"**
- [ ] Check file naming: must be `{SYMBOL}_5m.csv`
- [ ] Check directory path matches `data_dir` config

**Error: "Missing columns"**
- [ ] CSV must have: timestamp, open, high, low, close, volume
- [ ] Column names must be lowercase

**Error: "Only X valid labels, need 100"**
- [ ] Your CSV file is too small (< 106 rows)
- [ ] Need at least 100+ rows of historical data

**Warning: "Too few positive targets"**
- [ ] Target distribution < 25% positive
- [ ] Try lowering `edge_threshold_pct` from 0.0020 to 0.0010
- [ ] Or try longer horizon: change `prediction_horizon_candles` to 9 or 12

**Warning: "Too many positive targets"**
- [ ] Target distribution > 45% positive
- [ ] Try raising `edge_threshold_pct` from 0.0020 to 0.0030
- [ ] Or try shorter horizon: change `prediction_horizon_candles` to 4 or 3

## 📊 NEXT STEPS AFTER SUCCESS

Once Step 1 completes successfully:

1. **Review the JSON analysis file:**
   ```zsh
   cat validation_outputs/BTCUSDT_30m_label_analysis.json
   ```

2. **Check key metrics:**
   - Positive target ratio
   - Median absolute move
   - Distribution statistics

3. **Verify with decision rules:**
   - ✅ PASS: 25% < positive < 45% AND median move >= 0.30%
   - ❌ FAIL: Outside ranges
   - ⚠️ MAYBE: Edge cases (e.g., 24.8% positive)

4. **Proceed to Step 2** (if passing):
   - Review `PHASE_9_5_VALIDATION_FRAMEWORK.md` Step 2
   - Create expected move validation script
   - Confirm theoretical assumptions match reality

## 🕐 ESTIMATED TIME

- Setup/verification: 5 minutes
- Script execution: 5-10 minutes
- Result analysis: 5 minutes
- **Total: 15-25 minutes**

---

**You're ready. Run the script when data is verified above.**
