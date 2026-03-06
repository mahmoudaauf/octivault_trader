# Executive Summary: OHLCV vs Indicator Issue Diagnosis

## Your Situation

From logs, you see:
- ✅ **OHLCV is being fetched** (visible in debug logs)
- ✅ **Indicators are being computed** (EMA/MACD values printed)
- ❓ **But something might be wrong** with one of these

---

## The Three Possible Cases

### Case 1: Indicators Not Computed ❌
**Bottom Line:** Not enough data or computation failed

| Aspect | Details |
|--------|---------|
| **Symptoms** | Logs show "Insufficient OHLCV" or "Indicator error" |
| **Root Cause** | <50 bars accumulated OR computation crashes |
| **Fix Time** | 1-5 minutes (just wait for data) |
| **Likelihood** | Medium |

### Case 2: Wrong Timeframe Data ⚠️
**Bottom Line:** Fetching/storing wrong timeframe

| Aspect | Details |
|--------|---------|
| **Symptoms** | Indicators compute but signal quality is poor |
| **Root Cause** | Timeframe mismatch (config vs storage) |
| **Fix Time** | 5-15 minutes (config change) |
| **Likelihood** | Low |

### Case 3: Price Bug 🔴
**Bottom Line:** Close prices extracted from wrong position

| Aspect | Details |
|--------|---------|
| **Symptoms** | EMA values completely wrong (e.g., 50000 when price is 0.12) |
| **Root Cause** | Wrong index used to extract close price |
| **Fix Time** | 2-5 minutes (index fix) |
| **Likelihood** | **Very Low** (code verified correct) |

---

## Quick Diagnosis Steps

### Step 1: Check Logs (30 seconds)
```bash
grep -E "Insufficient|Indicator error|Indicator NaN" logs/clean_run.log
```
- **If found:** Case 1
- **If not found:** Go to Step 2

### Step 2: Verify Binance Price (1 minute)
```bash
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" | jq '.price'
```
- Compare with EMA value in logs
- **If matches (within 5%):** All good, not a price bug
- **If differs:** Case 3 (but unlikely)

### Step 3: Check Timeframe (2 minutes)
```bash
grep "fetching OHLCV" logs/clean_run.log | head -5
```
Look at the timeframe. Compare with TrendHunter config.
- **If matches:** All good
- **If differs:** Case 2

---

## Immediate Action

### Most Likely: Everything is OK ✅
Your system is probably working correctly:
1. OHLCV is being fetched
2. Indicators are computing
3. EMA values match actual prices

**Next question:** Why aren't signals being acted upon?
- Check decision generation
- Check confidence thresholds
- Check execution filters

### Less Likely: Case 1 or 2

**If Case 1:** Just wait 1-5 minutes for more bars to accumulate
```
Nothing to fix, data is accumulating normally
```

**If Case 2:** Quick config fix
```python
# In config.json, verify:
"TIMEFRAMES": ["5m", "1h"],  # Should include expected timeframes
```

---

## Documentation Provided

I've created 5 detailed guides:

1. **`OHLCV_INDICATOR_SUMMARY.md`** ← Start here
   - Overview of all three cases
   - Which one applies to you

2. **`QUICK_DIAGNOSTIC_FLOWCHART.md`** ← Then read this
   - Step-by-step questions
   - Quick diagnostic commands
   - Expected behaviors

3. **`THREE_CASE_ANALYSIS.md`** ← For deep dive
   - End-to-end data flow
   - Code references
   - Verification that Case 3 isn't present

4. **`DEBUG_SNIPPETS_BY_CASE.md`** ← If you need to add logging
   - Code to add for temporary debugging
   - What output to expect
   - How to interpret results

5. **`DIAGNOSTIC_OHLCV_INDICATOR_ISSUE.md`** ← Comprehensive reference
   - Detailed symptom analysis
   - Root cause investigation
   - How to fix each case

---

## The Bottom Line

### What's Likely Happening
1. OHLCV fetching: ✅ Working
2. Indicator computation: ✅ Working  
3. Data flow: ✅ Verified correct
4. Prices: ✅ Correct (if EMA ≈ Binance price)

### What to Check Next
1. Are BUY/SELL signals being generated?
2. Are confidence scores sufficient?
3. Are risk rules blocking execution?
4. Is the decision layer working?

**These are not OHLCV/indicator issues, but decision/execution issues.**

---

## One-Minute Check

If you only have 60 seconds:

```python
import asyncio

async def quick_check():
    # 1. Check data exists
    rows = await shared_state.get_market_data("ENAUSDT", "5m")
    assert rows is not None and len(rows) > 0, "NO DATA"
    
    # 2. Check sufficient
    assert len(rows) >= 50, f"CASE 1: Only {len(rows)}<50 bars"
    
    # 3. Check price sensible
    import numpy as np
    closes = np.asarray([r["c"] for r in rows], dtype=float)
    assert not np.any(np.isnan(closes)), "CASE 1: NaN in prices"
    
    # 4. Check indicators compute
    from utils.indicators import compute_ema
    ema = compute_ema(closes, 12)
    assert not np.isnan(ema[-1]), "CASE 1: EMA is NaN"
    
    print(f"✅ All checks passed")
    print(f"   {len(rows)} bars | price={closes[-1]:.8f} | EMA={ema[-1]:.8f}")

asyncio.run(quick_check())
```

**If this passes:** System is working, focus on signal generation
**If this fails:** Tells you exactly which case applies

---

## Support

If after running diagnostics you still need help:

1. **Run `run_diagnostic.py`** - provides automated assessment
2. **Provide output from:**
   - `python3 run_diagnostic.py`
   - `grep "EMA_S=" logs/clean_run.log | head -5`
   - `curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" | jq '.price'`

This will allow pinpointing the exact issue in seconds.

---

## Key Insight

From your logs:
```
2026-03-05 23:02:03,253 WARNING [AppContext] [DEBUG_MDF] fetching OHLCV for ENAUSDT
2026-03-05 23:02:14,071 - DEBUG - [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051
```

This shows a **healthy system**:
- Data is being fetched ✅
- Indicators are computing ✅
- 11-second delay is normal (async pipeline) ✅

**The question isn't "are indicators working" but "are signals being acted upon"**

That's a different part of the system to investigate.
