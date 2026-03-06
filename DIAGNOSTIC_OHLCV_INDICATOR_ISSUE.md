# Diagnostic: OHLCV vs EMA/Indicator Mismatch

## Issue Summary
From logs (`clean_run.log`), we see:
- ✅ OHLCV fetching is logged: `3. **Indicator code uses wrong index:**
   - ✅ **VERIFIED CORRECT**: The indexing is actually correct:
     - `exchange_client.get_ohlcv()` returns: `[openTime, open, high, low, close, volume]`
     - `_std_row()` converts dicts to: `[open, high, low, close, volume]`
     - `closes = np.asarray([r[3] for r in rows])` correctly gets index 3 = close
   - **This is NOT the culprit** - indexing is verified correctDF] fetching OHLCV for ENAUSDT` (line 924 in `market_data_feed.py`)
- ✅ EMA heuristic checks are logged: `[TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051`

**However**, the question is: Are the EMA values being computed **correctly** from the OHLCV data?

---

## Three Cases to Investigate

### **Case 1: Indicators Not Computed** ❌
**Symptoms:**
- OHLCV candles are present in `shared_state.market_data`
- But EMA/MACD/HISTOGRAM values are **NaN** or **missing**
- TrendHunter returns `"HOLD", 0.0, "Indicator error"` or `"Indicator NaN"`

**Root Causes:**
1. **Insufficient data rows:** Less than 50 candles required for stable indicators
   - Check: Line 848-850 in `agents/trend_hunter.py`
   ```python
   if len(rows) < min_required:
       return "HOLD", 0.1, f"Insufficient OHLCV ({len(rows)}<{min_required})"
   ```

2. **NaN values in close prices:**
   - Check line 870-872 in `agents/trend_hunter.py`
   ```python
   if any(np.isnan(float(v)) for v in tail_vals):
       return "HOLD", 0.0, "Indicator NaN"
   ```

3. **Exception during EMA/MACD computation:**
   - Check line 861-865 in `agents/trend_hunter.py`
   ```python
   except Exception as e:
       logger.error("[%s] Indicator calc failed for %s: %s", self.name, symbol, e, exc_info=True)
       return "HOLD", 0.0, "Indicator error"
   ```

**How to Diagnose:**
```python
# Check data availability
rows = shared_state.get_market_data("ENAUSDT", "5m")
print(f"OHLCV rows: {len(rows) if rows else 0}")
if rows:
    print(f"Last row: {rows[-1]}")
    print(f"All closes: {[r['c'] for r in rows[-10:]]}")
```

---

### **Case 2: Wrong Timeframe Data** ⚠️
**Symptoms:**
- OHLCV is fetched but from the **wrong timeframe**
- Example: Request `5m` data but get `1m` or `1h` data instead
- Indicators are computed but from incorrect time windows
- Signal quality is poor (high false positives/negatives)

**Root Causes:**
1. **Timeframe normalization mismatch:**
   - Check line 2833 in `core/shared_state.py`
   ```python
   def _norm_tf(self, timeframe: str) -> str:
       # If this doesn't properly normalize "5m" → "5m", lookups fail
   ```

2. **Market data feed fetching wrong timeframe:**
   - Check line 924-935 in `core/market_data_feed.py`
   ```python
   for tf in self.timeframes:
       try:
           rows = await ec.get_ohlcv(sym, tf, limit=3)
   ```
   If `self.timeframes` doesn't match config or requests wrong intervals

3. **SharedState key mismatch:**
   - OHLCV stored under key `(symbol, "5m")`
   - But lookup tries `(symbol, "5M")` or `(symbol, "5mins")`
   - See line 2846-2852 in `core/shared_state.py`

**How to Diagnose:**
```python
# Check what timeframes are available
print("Available market data keys:")
for key in shared_state.market_data.keys():
    symbol, timeframe = key
    data = shared_state.market_data[key]
    print(f"  {symbol:12s} {timeframe:5s} → {len(data)} bars")

# Check what TrendHunter is requesting
print(f"\nTrendHunter.timeframe = {trend_hunter.timeframe}")
# Should match what's stored in market_data
```

---

### **Case 3: Price Reference Bug** 🔴
**Symptoms:**
- OHLCV candles are fetched
- Indicators are computed
- **But the close prices are stale, zero, or misaligned**
- Example: EMA shows `0.12` when price is `65000` (clearly wrong)

**Root Causes:**

1. **OHLCV normalization strips price data:**
   - Check `_sanitize_ohlcv()` in `market_data_feed.py` line 926-938
   ```python
   bar = {
       "ts": float(r[0]),  # r[0] = open_time (ms)
       "o": float(r[1]),   # r[1] = open
       "h": float(r[2]),   # r[2] = high
       "l": float(r[3]),   # r[3] = low  ← WRONG INDEX!
       "c": float(r[4]),   # r[4] = close
       "v": float(r[5]),   # r[5] = volume
   }
   ```
   
   **BUG HYPOTHESIS:** If Binance API returns:
   ```
   [timestamp, open, high, low, close, volume, ...]
   [  0,       1,    2,    3,   4,     5       ]
   ```
   But code extracts as:
   ```
   bar["l"] = float(r[3])  # Should be "low", but what if indexed wrong?
   bar["c"] = float(r[4])  # Should be "close"
   ```

2. **Indicator code uses wrong index:**
   - Check line 844 in `agents/trend_hunter.py`
   ```python
   closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] is labeled 'c'
   ```
   But if `rows` is a list of dicts, this should be `r["c"]`, not `r[3]`!
   
   **This is the likely culprit!**

3. **Price injection not updating properly:**
   - Check line 918-920 in `core/market_data_feed.py`
   ```python
   price_f = self._coerce_positive_price(price)
   if price_f > 0:
       try:
           price_updated = bool(await self._inject_latest_price(sym, price_f))
   ```

**How to Diagnose:**
```python
# Check if OHLCV has correct format
rows = await shared_state.get_market_data("ENAUSDT", "5m")
if rows:
    # Verify it's a dict
    last = rows[-1]
    print(f"Last OHLCV type: {type(last)}")
    print(f"Last OHLCV keys: {last.keys() if isinstance(last, dict) else 'NOT A DICT'}")
    
    # Try both access methods
    try:
        c_method1 = last[3]  # List/tuple access
        print(f"  last[3] = {c_method1}")
    except:
        print(f"  last[3] failed (not a list/tuple)")
    
    try:
        c_method2 = last["c"]  # Dict access
        print(f"  last['c'] = {c_method2}")
    except:
        print(f"  last['c'] failed (not a dict)")
    
    # Print all fields
    print(f"All fields: {last}")
```

---

## Quick Test to Identify Which Case

Add this to your code to run a quick diagnostic:

```python
import asyncio
from core.shared_state import SharedState

async def diagnose():
    ss = shared_state  # Your shared_state instance
    symbol = "ENAUSDT"
    timeframe = "5m"
    
    # Get OHLCV
    rows = await ss.get_market_data(symbol, timeframe)
    
    if rows is None:
        print(f"❌ CASE 2: No data for {symbol}-{timeframe}")
        print(f"Available keys: {list(ss.market_data.keys())[:10]}")
        return
    
    print(f"✅ Found {len(rows)} OHLCV bars for {symbol}-{timeframe}")
    
    if len(rows) < 50:
        print(f"❌ CASE 1: Insufficient data ({len(rows)} < 50)")
        return
    
    # Check data format
    last_row = rows[-1]
    print(f"Last OHLCV type: {type(last_row)}")
    print(f"Last OHLCV: {last_row}")
    
    # Try to extract close price
    try:
        if isinstance(last_row, dict):
            close = last_row.get("c") or last_row.get("close")
        else:
            close = last_row[4]  # List/tuple index
        print(f"✅ Extracted close: {close}")
    except Exception as e:
        print(f"❌ CASE 3: Failed to extract close: {e}")
        return
    
    # Check EMA computation
    try:
        import numpy as np
        closes = np.asarray([r["c"] if isinstance(r, dict) else r[4] for r in rows], dtype=float)
        print(f"✅ EMA input (closes): min={closes.min():.2f}, max={closes.max():.2f}, last={closes[-1]:.2f}")
    except Exception as e:
        print(f"❌ CASE 3: Failed to compute EMA input: {e}")

# Run diagnostic
# asyncio.run(diagnose())
```

---

## Recommended Fixes by Case

### If Case 1 (Insufficient Data):
- Increase `TRENDHUNTER_MIN_DATA` in config
- Or wait for more market data to accumulate
- Check `market_data_feed.min_bars_required` setting

### If Case 2 (Wrong Timeframe):
- Verify `self.timeframes` in `MarketDataFeed.__init__()`
- Check config for `TIMEFRAMES` setting
- Ensure `_norm_tf()` properly handles all timeframe formats
- Add logging at line 924 in `market_data_feed.py`:
  ```python
  self._logger.warning(f"[DEBUG] Fetching OHLCV for {sym}/{tf}")
  ```

### If Case 3 (Price Reference Bug):
- **FIX:** In `agents/trend_hunter.py` line 844-845, change:
  ```python
  # WRONG:
  closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] is low, not close!
  
  # RIGHT:
  closes = np.asarray([r["c"] if isinstance(r, dict) else r[4] for r in rows], dtype=float)
  ```
- Verify OHLCV dict key consistency across all components
- Check `_sanitize_ohlcv()` returns correct dict format

---

## Log Evidence Analysis

From your logs:
```
2026-03-05 23:02:03,253 WARNING [AppContext] [DEBUG_MDF] fetching OHLCV for ENAUSDT
2026-03-05 23:02:14,071 - DEBUG - [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051
```

**Key Observation:**
- Time gap: **11 seconds** between OHLCV fetch and EMA computation
- EMA values are extremely small: **0.12** (likely a price in the range of 0.01-0.1)
- This could be a **Case 3: Price Reference Bug** if ENAUSDT's actual price is much higher

**Action:** Check Binance for ENAUSDT's current price:
- If ENAUSDT price ≈ 0.12 → All good (normal price)
- If ENAUSDT price is much higher (e.g., 65000) → **Case 3 confirmed!**

---

## Next Steps

1. **Run the diagnostic code above** to identify which case
2. **Post the output** and I'll provide a targeted fix
3. **Check the price** of suspicious symbols (EMA values vs actual market price)
4. **Review logs** for `"Insufficient OHLCV"` or `"Indicator error"` messages
