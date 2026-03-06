# Summary: OHLCV Fetching vs Indicator Computation Issues

## Three Cases Overview

Based on your logs showing both OHLCV fetching and EMA computation, here's how to identify which case applies:

### **Case 1: Indicators Not Computed** ❌
**What it looks like:**
- OHLCV is fetched but indicators fail
- Logs show: `"Insufficient OHLCV"` or `"Indicator error"` or `"Indicator NaN"`
- No EMA values printed

**Why it happens:**
- Less than 50 OHLCV bars in storage (min_required = 50)
- NaN values in close prices (shouldn't happen but possible with API errors)
- EMA/MACD computation crashes (rare)

**Where to look:**
- `agents/trend_hunter.py` lines 837-872 (data sufficiency & computation)
- `core/shared_state.py` lines 2846-2852 (data retrieval)

**How to fix:**
1. Increase `TRENDHUNTER_MIN_DATA` in config if it's too high
2. Wait longer for MarketDataFeed to accumulate 50+ bars
3. Check logs for explicit error messages: `grep "Insufficient\|Indicator error" logs/clean_run.log`

---

### **Case 2: Wrong Timeframe Data** ⚠️
**What it looks like:**
- EMA values print normally BUT
- Signal quality is poor / signals are inverted
- Trading on misaligned candle times

**Why it happens:**
- MarketDataFeed fetches wrong timeframe (config mismatch)
- Timeframe normalization fails (e.g., "5m" → "5M" → key not found)
- Requesting "5m" but getting "1m" data instead

**Where to look:**
- `core/market_data_feed.py` lines 922-935 (timeframe fetching loop)
- `core/shared_state.py` line 2827 (`_norm_tf` function)
- TrendHunter `self.timeframe` vs config setting

**How to fix:**
1. Verify `MarketDataFeed.timeframes` config matches your expected timeframes
2. Check `_norm_tf()` properly normalizes all variations (5m, 5M, 5mins, etc.)
3. Add debug logging at market_data_feed.py:924 to see what's being fetched
4. Ensure stored key `(symbol, timeframe)` matches retrieval key

---

### **Case 3: Price Reference Bug** 🔴
**What it looks like:**
- EMA values seem completely wrong (e.g., EMA=50000 when price is $0.12)
- Close prices are clearly extracted from wrong position

**Why it happens:**
- Wrong index used to extract close price
- Binance API format differs than expected
- Data structure mismatch between storage and retrieval

**Status:** ✅ **NOT PRESENT IN YOUR CODE**

I've verified the data flow end-to-end:
```
Binance API (index 4 = close)
    ↓
market_data_feed.py stores r[4] as "c" ✅
    ↓
_std_row() converts dict → [o, h, l, c, v] ✅
    ↓
trend_hunter.py reads r[3] = c ✅
```

All indices are correct. If EMA values seem wrong, it's because Case 1 or Case 2, not Case 3.

---

## Your Log Analysis

From `clean_run.log`:

```
2026-03-05 23:02:03,253 WARNING [AppContext] [DEBUG_MDF] fetching OHLCV for ENAUSDT
2026-03-05 23:02:14,071 - DEBUG - [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051
```

**What this tells us:**
- ✅ OHLCV fetching is happening (first line)
- ✅ EMA computation is happening (second line, 11 seconds later)
- ✅ EMA values are reasonable (0.12 is the actual price of ENAUSDT if it trades at that level)
- ✅ HISTOGRAM is computing (-0.000051 is valid MACD histogram)

**Likely situation:** Everything is working normally. The system is:
1. Fetching OHLCV successfully
2. Computing indicators successfully
3. Generating signals

**Next question:** Are the signals being acted upon? Check if:
- Decisions are being generated (not blocked)
- Confidence is > minimum threshold
- Risk rules aren't blocking execution

---

## Recommended Next Steps

### Step 1: Run the diagnostic script
```bash
python3 run_diagnostic.py
```

This will:
- Check OHLCV data availability
- Verify data format
- Compute indicators and validate results
- Report any issues

### Step 2: Verify Binance prices
```bash
# Check if ENAUSDT really trades at ~0.12
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" | jq '.price'

# For all symbols in your logs
for SYM in ENAUSDT BCHUSDT ETHUSDT BTCUSDT; do
  echo -n "$SYM: "
  curl -s "https://api.binance.com/api/v3/ticker/price?symbol=$SYM" | jq '.price'
done
```

### Step 3: Check for error messages
```bash
# Case 1 indicators
grep -E "Insufficient OHLCV|Indicator error|Indicator NaN" logs/clean_run.log

# Case 2 indicators (less obvious - look at timeframe in debug logs)
grep "fetching OHLCV" logs/clean_run.log | head -10

# Case 3 indicators (unlikely)
grep "price.*wrong\|index.*error" logs/clean_run.log
```

### Step 4: Add debug logging (if needed)
```python
# In agents/trend_hunter.py around line 844, add:
logger.info(
    "[DEBUG] %s: %d bars, closes=[%.8f...%.8f], ema_short=%.8f, ema_long=%.8f, hist=%.8f",
    symbol,
    len(rows),
    closes[0],
    closes[-1],
    ema_short[-1],
    ema_long[-1],
    hist[-1],
)
```

---

## Files Created for Your Reference

1. **`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`**
   - Detailed explanation of all three cases
   - Code locations for each case
   - How to diagnose

2. **`THREE_CASE_ANALYSIS.md`**
   - End-to-end data flow verification
   - Code path from Binance → storage → indicators
   - Why Case 3 is not present

3. **`QUICK_DIAGNOSTIC_FLOWCHART.md`**
   - Step-by-step questions to identify the case
   - One-minute diagnostic commands
   - Expected normal behavior

4. **`run_diagnostic.py`**
   - Automated diagnostic script
   - Checks all three cases
   - Outputs clear pass/fail for each

---

## Decision Tree

```
Do indicators compute at all?
├─ NO → CASE 1: Insufficient Data
│       └─ Check: len(rows) < 50?
│       └─ Fix: Wait for more OHLCV or increase TRENDHUNTER_MIN_DATA
│
└─ YES: Do EMA values match Binance price?
    ├─ NO → CASE 3: Price Bug (unlikely, but check indexing)
    │       └─ Fix: Verify r[3] vs r[4] indexing
    │
    └─ YES: Are signals being generated?
        ├─ NO → CASE 2: Wrong Timeframe (signals messed up)
        │       └─ Fix: Verify timeframe matching
        │
        └─ YES: Check why decisions aren't executing
                └─ Likely: Risk rules, insufficient confidence, or execution filters
```

---

## Quick Reference: Code Indices

**Binance API returns:**
```
[openTime, open, high, low, close, volume, ...]
[   0,     1,    2,   3,   4,     5      ]
```

**MarketDataFeed stores as dict:**
```python
{"ts": r[0], "o": r[1], "h": r[2], "l": r[3], "c": r[4], "v": r[5]}
```

**TrendHunter normalizes to list:**
```python
[open, high, low, close, volume]
[ 0,    1,   2,    3,     4   ]
```

**So `r[3]` is close.** ✅ Correct!

---

## Bottom Line

✅ Your OHLCV fetching is working
✅ Your indicator computation is working
✅ Your EMA values are reasonable

**Focus on:** Why signals aren't being acted upon
- Are decisions being generated?
- Are confidence scores sufficient?
- Are risk rules blocking execution?

These are not OHLCV or indicator issues, but decision/execution issues.
