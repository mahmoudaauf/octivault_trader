# Three-Case Diagnostic Guide: Complete Index

## 📋 What This Guide Covers

You asked about three possible scenarios:
1. **Case 1:** Indicators not computed (OHLCV present but EMA/MACD missing)
2. **Case 2:** Wrong timeframe data (fetching one timeframe, using another)
3. **Case 3:** Price reference bug (close prices extracted wrong)

This guide helps you identify which case (if any) applies to your system.

---

## 🚀 Quick Start (5 Minutes)

### Read First
👉 **[`CASE_DIAGNOSIS_START_HERE.md`](CASE_DIAGNOSIS_START_HERE.md)**
- 3-question diagnosis flowchart
- Quick bash commands to identify your case
- Expected behaviors for each case

### Run This
```bash
# Automated diagnostic
python3 run_diagnostic.py

# Or manual checks
grep "EMA_S=" logs/clean_run.log | head -5
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT"
```

---

## 📚 Complete Documentation

### 1. **Executive Summary** ⭐ START HERE
📄 **[`CASE_DIAGNOSIS_START_HERE.md`](CASE_DIAGNOSIS_START_HERE.md)**
- Best for: Quick diagnosis in 5 minutes
- Contains: Flowchart, quick checks, immediate actions
- Read this first

### 2. **Detailed Overview**
📄 **[`OHLCV_INDICATOR_SUMMARY.md`](OHLCV_INDICATOR_SUMMARY.md)**
- Best for: Understanding what each case means
- Contains: Case descriptions, log analysis, decision tree
- Read this second

### 3. **Step-by-Step Flowchart**
📄 **[`QUICK_DIAGNOSTIC_FLOWCHART.md`](QUICK_DIAGNOSTIC_FLOWCHART.md)**
- Best for: Methodical diagnosis
- Contains: Questions to answer, commands to run, red flags
- Use this to walk through diagnosis

### 4. **Technical Deep Dive**
📄 **[`THREE_CASE_ANALYSIS.md`](THREE_CASE_ANALYSIS.md)**
- Best for: Understanding code flow
- Contains: Code locations, data flow verification, root causes
- Read if you need technical details

### 5. **Comprehensive Reference**
📄 **[`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`](DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md)**
- Best for: Complete symptom analysis
- Contains: Detailed case-by-case breakdown, diagnostics
- Reference when stuck

### 6. **Debug Code Snippets**
📄 **[`DEBUG_SNIPPETS_BY_CASE.md`](DEBUG_SNIPPETS_BY_CASE.md)**
- Best for: Adding temporary logging
- Contains: Copy-paste code to add for each case, expected output
- Use if diagnostics aren't clear

### 7. **Diagnostic Script**
📄 **[`run_diagnostic.py`](run_diagnostic.py)**
- Best for: Automated checking
- Run: `python3 run_diagnostic.py`
- Reports all three cases automatically

---

## 🎯 Which Document to Read?

### Scenario: "I have 5 minutes"
1. Read: [`CASE_DIAGNOSIS_START_HERE.md`](CASE_DIAGNOSIS_START_HERE.md)
2. Run: Quick diagnostic commands

### Scenario: "I want to understand the issue"
1. Read: [`OHLCV_INDICATOR_SUMMARY.md`](OHLCV_INDICATOR_SUMMARY.md)
2. Review: [`QUICK_DIAGNOSTIC_FLOWCHART.md`](QUICK_DIAGNOSTIC_FLOWCHART.md)
3. Run: `python3 run_diagnostic.py`

### Scenario: "I need technical details"
1. Read: [`THREE_CASE_ANALYSIS.md`](THREE_CASE_ANALYSIS.md)
2. Reference: [`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`](DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md)
3. Debug: [`DEBUG_SNIPPETS_BY_CASE.md`](DEBUG_SNIPPETS_BY_CASE.md)

### Scenario: "I'm stuck on diagnosis"
1. Add debug code from: [`DEBUG_SNIPPETS_BY_CASE.md`](DEBUG_SNIPPETS_BY_CASE.md)
2. Run system and capture output
3. Review diagnostic output in detail

---

## 🔍 Case Identification Cheat Sheet

| Case | Symptom | Check | Fix Time |
|------|---------|-------|----------|
| **Case 1** | "Insufficient OHLCV" in logs | `len(rows) < 50` | 1-5 min |
| **Case 1** | "Indicator error" in logs | Check exception logs | 2-10 min |
| **Case 2** | Wrong timeframe in config | `fetching OHLCV for X-1m` vs `expected 5m` | 5-15 min |
| **Case 3** | EMA ≠ Binance price | `EMA=50000, price=0.12` | 2-5 min |
| **None** | Everything working! | All checks pass | ✅ Done |

---

## 💻 Quick Diagnostic Commands

### Check for Case 1
```bash
# Look for explicit errors
grep -E "Insufficient|Indicator error|Indicator NaN" logs/clean_run.log

# Check data volume
python3 -c "
from core.shared_state import SharedState
from core.config import Config
import logging

config = Config('config/config.json')
ss = SharedState({}, config, logging.getLogger(), 'Check')
for key, rows in list(ss.market_data.items())[:5]:
    sym, tf = key
    print(f'{sym:12s} {tf:5s} → {len(rows):4d} bars')
"
```

### Check for Case 2
```bash
# See what timeframes are being fetched
grep "fetching OHLCV" logs/clean_run.log | head -5

# Compare with TrendHunter config
cat config/config.json | grep -i timeframe
```

### Check for Case 3
```bash
# Get actual market price
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" | jq '.price'

# Check EMA in logs
grep "EMA_S=" logs/clean_run.log | head -1
# Should be similar value
```

---

## 📊 Understanding Your Logs

### Healthy System Example
```
[Time 1] WARNING [DEBUG_MDF] fetching OHLCV for ENAUSDT
[Time 2] DEBUG [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051
         ↓
         ✅ Both happening = system working
         ✅ 11-second gap is normal
         ✅ EMA value (0.12) matches price (verify with Binance)
```

### Case 1 Example
```
[Time 1] WARNING [DEBUG_MDF] fetching OHLCV for ENAUSDT
         (but no heuristic check)
         ↓
         ❌ Case 1: Indicators not computed
         Check: len(rows) < 50?
```

### Case 2 Example
```
[Time 1] WARNING [DEBUG_MDF] fetching OHLCV for ENAUSDT-1m (wrong timeframe!)
[Time 2] WARNING [DEBUG_MDF] fetching OHLCV for ETHUSDT-1m
         ↓
         ❌ Case 2: Wrong timeframe
         Expected: 5m, Got: 1m
```

---

## ✅ Success Criteria

You'll know the system is working when you see:

1. ✅ OHLCV fetching logs appear regularly
2. ✅ EMA values print in debug output
3. ✅ EMA values ≈ Binance price (within 5%)
4. ✅ HISTOGRAM changes with market moves
5. ✅ BUY/SELL signals generated (not always HOLD)
6. ✅ Signal confidence > 0

---

## 🆘 If You're Still Stuck

### Step 1: Collect Information
```bash
# Run automated diagnostic
python3 run_diagnostic.py > diagnostic_output.txt

# Collect logs
tail -100 logs/clean_run.log > recent_logs.txt

# Check Binance
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" > binance_price.txt
```

### Step 2: Add Debug Logging
Copy snippets from [`DEBUG_SNIPPETS_BY_CASE.md`](DEBUG_SNIPPETS_BY_CASE.md):
- For Case 1: Add to `agents/trend_hunter.py`
- For Case 2: Add to `core/market_data_feed.py`
- For Case 3: Add to data extraction

### Step 3: Review Detailed Docs
- Reference [`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`](DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md)
- Review [`THREE_CASE_ANALYSIS.md`](THREE_CASE_ANALYSIS.md)

---

## 📖 Document Relationships

```
START HERE
    ↓
CASE_DIAGNOSIS_START_HERE.md (5 min read)
    ↓
    ├─→ OHLCV_INDICATOR_SUMMARY.md (understand the cases)
    │       ↓
    │   QUICK_DIAGNOSTIC_FLOWCHART.md (step-by-step)
    │       ↓
    │   THREE_CASE_ANALYSIS.md (technical details)
    │
    └─→ run_diagnostic.py (automated check)
            ↓
        DEBUG_SNIPPETS_BY_CASE.md (if more info needed)
```

---

## 🎓 Key Learnings

### The Data Flow (Correct in Your System)
```
Binance API                    [timestamp, open, high, low, close, volume]
                                           ↓
market_data_feed.py stores      {"ts": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4], "v": r[5]}
                                           ↓
_std_row() converts to          [open, high, low, close, volume]
                                 ↓
compute_ema uses                closes = [r[3], r[3], ...]  ← Correct!
```

All indices are correct throughout your codebase. If there's an issue, it's not a Case 3 bug.

### Why Case 3 is Unlikely
- ✅ Code verified end-to-end
- ✅ All indices checked and correct
- ✅ Multiple verification points
- ✅ System has been through many iterations

---

## 📝 Notes

- **Case 1** is most likely if you see explicit error messages
- **Case 2** would show as signal quality issues, not computation errors
- **Case 3** is essentially ruled out (code verified correct)
- **Most likely** scenario: System is working, need to check signal generation

---

## 🔗 Quick Links

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [`CASE_DIAGNOSIS_START_HERE.md`](CASE_DIAGNOSIS_START_HERE.md) | Quick diagnosis | 5 min |
| [`OHLCV_INDICATOR_SUMMARY.md`](OHLCV_INDICATOR_SUMMARY.md) | Case overview | 10 min |
| [`QUICK_DIAGNOSTIC_FLOWCHART.md`](QUICK_DIAGNOSTIC_FLOWCHART.md) | Step-by-step | 15 min |
| [`THREE_CASE_ANALYSIS.md`](THREE_CASE_ANALYSIS.md) | Technical deep dive | 20 min |
| [`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`](DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md) | Comprehensive ref | 25 min |
| [`DEBUG_SNIPPETS_BY_CASE.md`](DEBUG_SNIPPETS_BY_CASE.md) | Add logging | 10 min |
| [`run_diagnostic.py`](run_diagnostic.py) | Automated check | < 1 min |

---

## 💡 TL;DR

**Your situation:** OHLCV is fetched, indicators compute, values look reasonable

**Most likely:** Everything is working correctly ✅

**Action:** Investigate signal generation and execution layer instead

**If stuck:** Run `python3 run_diagnostic.py` and review appropriate document based on output
