# PHASE 9.5 STEP 1: QUICK REFERENCE INDEX

**Status:** ✅ READY TO EXECUTE  
**Created:** February 21, 2026  
**Purpose:** Construct and validate 30m cumulative return labels  
**Time to Execute:** 20-30 minutes  

---

## 📂 FILE LOCATIONS

All files are in: `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/`

### Implementation Code
```
step1_construct_30m_labels.py       ← MAIN EXECUTABLE (700+ lines)
```

### Documentation (Read in This Order)

**Before Execution:**
1. `STEP_1_PREFLIGHT_CHECKLIST.md` ← Start here (5 min)
2. `STEP_1_IMPLEMENTATION_SUMMARY.md` ← Overview (5 min)

**During/After Execution:**
3. `STEP_1_EXECUTION_GUIDE.md` ← How to run (10 min)
4. `STEP_1_EXPECTED_OUTPUTS.md` ← Understanding results (5 min)

**This File:**
5. `STEP_1_QUICK_REFERENCE.md` ← Quick lookup (you are here)

---

## 🚀 QUICKEST START (3 COMMANDS)

```zsh
# 1. Verify data
ls data/historical/*_5m.csv

# 2. Run
python step1_construct_30m_labels.py

# 3. Check results
cat validation_outputs/step1_results.json | head -20
```

---

## 📊 WHAT TO EXPECT

### Terminal Output:
```
✅ Loaded 10000 rows for BTCUSDT
✅ Created 9994 valid labels
✅ Saved labeled data: validation_outputs/...
```

### Metrics to Check:
```
Positive Target Ratio:  Should be 25-45%    (Example: 35.2%)
Median Absolute Move:   Should be >= 0.30%  (Example: 0.413%)
Valid Labels Count:     Should be >= 100    (Example: 9994)
```

### Decision:
```
✅ PASS:   All metrics OK → Proceed to Step 2
⚠️ MAYBE:  Borderline → Proceed carefully
❌ FAIL:   Metrics bad → Adjust and retry
```

---

## 🔧 IF SOMETHING FAILS

### "Data file not found"
→ Check: `ls data/historical/*_5m.csv`  
→ Fix: Update `data_dir` in script

### "Missing columns"
→ Check: CSV must have: timestamp, open, high, low, close, volume  
→ Fix: Verify CSV format

### "Positive: 15%" (Too low)
→ Fix: Lower `edge_threshold_pct` from 0.0020 to 0.0010  
→ Retry: Re-run script

### "Positive: 60%" (Too high)
→ Fix: Raise `edge_threshold_pct` from 0.0020 to 0.0030  
→ Retry: Re-run script

### "Median move: 0.20%" (Too small)
→ Fix: Extend horizon: `prediction_horizon_candles` to 9  
→ Retry: Re-run script

---

## 📋 OUTPUT FILES

After running, you'll have:

```
validation_outputs/
├── BTCUSDT_5m_with_30m_labels.csv      ← Original data + labels
├── BTCUSDT_30m_label_analysis.json     ← Statistics
├── ETHUSDT_5m_with_30m_labels.csv
├── ETHUSDT_30m_label_analysis.json
└── step1_results.json                  ← Summary (THIS IS KEY)
```

**To view results:**
```zsh
cat validation_outputs/step1_results.json
```

---

## ✅ DECISION MATRIX

| Metric | Need | You Got | Status |
|--------|------|---------|--------|
| Positive targets | 25-45% | __% | ✓/✗ |
| Median move | >= 0.30% | __% | ✓/✗ |
| Valid labels | >= 100 | ___ | ✓/✗ |

If all ✓: **PASS** → Proceed to Step 2  
If any ✗: **FAIL** → Adjust and retry

---

## ⏱️ TIMELINE

| Phase | Time | Action |
|-------|------|--------|
| Setup | 5 min | Read checklist, verify data |
| Execution | 10 min | Run `python step1_construct_30m_labels.py` |
| Analysis | 5 min | Read outputs, check metrics |
| Decision | 5 min | PASS/FAIL/MAYBE? |
| **TOTAL** | **25 min** | Ready for next step |

---

## 🎯 WHAT STEP 1 VALIDATES

**Question:** Can we construct meaningful 30m cumulative return targets?

**Answer:** Run the script and check:
- ✅ Can we make labels? (No errors)
- ✅ Are labels realistic? (25-45% positive)
- ✅ Are moves significant? (>= 0.30% median)
- ✅ Is data clean? (No NaNs, valid counts)

---

## 📖 WHERE TO LOOK FOR ANSWERS

| Question | File | Section |
|----------|------|---------|
| "How do I run this?" | STEP_1_EXECUTION_GUIDE.md | Quick Start |
| "What will I see?" | STEP_1_EXPECTED_OUTPUTS.md | Expected Output |
| "Did I pass?" | STEP_1_EXPECTED_OUTPUTS.md | Decision Rules |
| "Something failed!" | STEP_1_EXECUTION_GUIDE.md | Troubleshooting |
| "What's next?" | STEP_1_IMPLEMENTATION_SUMMARY.md | Next Steps |

---

## 🔄 IF YOU RETRY

1. Edit `step1_construct_30m_labels.py` (around line 95)
2. Change one of these:
   - `edge_threshold_pct = 0.0010` (instead of 0.0020)
   - `prediction_horizon_candles = 9` (instead of 6)
   - Other config as needed
3. Save file
4. Re-run: `python step1_construct_30m_labels.py`
5. Check: `cat validation_outputs/step1_results.json`

---

## 🎓 KEY LEARNING

This Step 1 teaches you:
1. **How to label** - Create forward-looking targets
2. **What's realistic** - Expected distributions
3. **How to validate** - Statistical decision framework
4. **When to proceed** - Clear pass/fail criteria

This is hedge fund methodology:
> **Test first. Believe second. Adjust always.**

---

## ✨ REMEMBER

✅ Production-quality code (not experimental)  
✅ Full error handling  
✅ Extensive documentation  
✅ Binary decision framework (YES/NO/MAYBE)  
✅ Statistical validation throughout  

---

## 🏁 NEXT ACTION

```zsh
python step1_construct_30m_labels.py
```

Then check: `validation_outputs/step1_results.json`

Report back with:
- Positive target ratio (%)
- Median absolute move (%)
- Pass/Fail status

---

## 🆘 QUICK HELP

**Q: Where's the code?**  
A: `step1_construct_30m_labels.py` (700+ lines)

**Q: How long does it take?**  
A: 20-30 minutes total (5 min setup + 10 min run + 5 min analysis)

**Q: What if I have no data?**  
A: You need `data/historical/*.csv` files with OHLCV data

**Q: Can I use different data location?**  
A: Yes, edit `data_dir = "your/path"` in the script

**Q: Can I use 1m or 15m data?**  
A: Yes, but adjust `prediction_horizon_candles` accordingly

**Q: Will it work on my machine?**  
A: If you have Python 3.8+, pandas, numpy → YES

---

## 📞 GETTING UNSTUCK

| Problem | Solution |
|---------|----------|
| Import error | `pip install pandas numpy` |
| No data files | Create CSV files in `data/historical/` |
| Results look weird | Read `STEP_1_EXPECTED_OUTPUTS.md` for examples |
| Need to adjust | Edit config, re-run |
| Still stuck | Review troubleshooting in `STEP_1_EXECUTION_GUIDE.md` |

---

**Status:** Ready  
**Confidence:** High  
**Next Step:** Execute  

```zsh
python step1_construct_30m_labels.py
```
