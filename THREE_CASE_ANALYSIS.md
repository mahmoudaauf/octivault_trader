# Three-Case Analysis: OHLCV Fetching vs Indicator Computation

## Overview

Your logs show:
- ✅ OHLCV is being **fetched**: `[DEBUG_MDF] fetching OHLCV for ENAUSDT`
- ✅ **Indicators are being computed**: `[TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051`

**Question:** Are indicators being computed **correctly** from the OHLCV data?

---

## Case 1: Indicators Not Computed (Data Insufficient or NaN)

### Symptoms
- OHLCV bars exist in storage
- Indicator computation fails or returns NaN
- Log messages like:
  - `"Insufficient OHLCV (n<50)"`
  - `"Indicator NaN"`
  - `"Indicator error"`

### Code Locations

**Fetching logs:**
```python
# core/market_data_feed.py:924
self._logger.warning("[DEBUG_MDF] fetching OHLCV for %s", sym)
```

**Storing OHLCV:**
```python
# core/market_data_feed.py:931-937
bar = {
    "ts": float(r[0]),  # openTime (ms)
    "o": float(r[1]),   # open
    "h": float(r[2]),   # high
    "l": float(r[3]),   # low
    "c": float(r[4]),   # close ← Critical: This is the close price
    "v": float(r[5]),   # volume
}
await self._maybe_await(self.shared_state.add_ohlcv(sym, tf, bar))
```

**Data normalization in TrendHunter:**
```python
# agents/trend_hunter.py:101-119
def _std_row(self, r):
    """Convert dict or list to [open, high, low, close, volume]"""
    if isinstance(r, dict):
        return [float(o), float(h), float(l), float(c), float(v)]
    #     Index:    [  0    ,   1   ,   2   ,   3   ,   4   ]
    
    # For lists, take last 5 elements
    if len(seq) == 5:
        return [float(x) for x in seq]
```

**Indicator computation:**
```python
# agents/trend_hunter.py:844-865
closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] = close (from _std_row)
fast = int(self._cfg("TRENDHUNTER_EMA_SHORT", self.ema_fast))      # 12
slow = int(self._cfg("TRENDHUNTER_EMA_LONG", self.ema_slow))        # 26

try:
    if _HAS_TALIB:
        ema_short = talib.EMA(closes, timeperiod=fast)
        ema_long = talib.EMA(closes, timeperiod=slow)
        macd_line, sig_line, hist = talib.MACD(closes)
    else:
        ema_short = compute_ema(closes, fast)
        ema_long = compute_ema(closes, slow)
        macd_line, sig_line, hist = compute_macd(closes)
    
    # Guard against NaN
    tail_vals = [ema_short[-1], ema_long[-1], hist[-1]]
    if any(np.isnan(float(v)) for v in tail_vals):
        return "HOLD", 0.0, "Indicator NaN"
        
except Exception as e:
    logger.error("[%s] Indicator calc failed for %s: %s", self.name, symbol, e, exc_info=True)
    return "HOLD", 0.0, "Indicator error"
```

**Data sufficiency check:**
```python
# agents/trend_hunter.py:837-850
min_required = 50
cfg_min = self._cfg("TRENDHUNTER_MIN_DATA", 50)
if isinstance(cfg_min, dict):
    min_required = int(cfg_min.get(symbol, cfg_min.get("default", 50)))

# Ensure adequate lookback for indicator stability (EMA slow + MACD signal + padding)
macd_signal = 9
min_required = max(int(min_required), int(slow + macd_signal + 3))
# = max(50, 26 + 9 + 3) = max(50, 38) = 50

if len(rows) < min_required:
    return "HOLD", 0.1, f"Insufficient OHLCV ({len(rows)}<{min_required})"
```

### Diagnosis Commands

Check data availability:
```python
import asyncio
from core.shared_state import SharedState

async def check():
    rows = await shared_state.get_market_data("ENAUSDT", "5m")
    if rows:
        print(f"✅ {len(rows)} bars available")
        if len(rows) < 50:
            print(f"❌ Case 1: Only {len(rows)} < 50 required")
    else:
        print(f"❌ No data at all")

asyncio.run(check())
```

---

## Case 2: Wrong Timeframe Data

### Symptoms
- OHLCV data exists but from **different timeframe** than requested
- Indicators compute but signal quality is poor
- Trading on misaligned candle times

### Code Locations

**Timeframe fetching loop:**
```python
# core/market_data_feed.py:922-935
self._logger.warning("[DEBUG_MDF] fetching OHLCV for %s", sym)
for tf in self.timeframes:  # ← What is in self.timeframes?
    try:
        async def _fetch_tail():
            return await ec.get_ohlcv(sym, tf, limit=3)
        rows = await self._with_retries(_fetch_tail, f"poll.get_ohlcv[{sym},{tf}]")
        rows = self._sanitize_ohlcv(rows or [])
        for r in rows:
            bar = {
                "ts": float(r[0]),
                "o": float(r[1]),
                "h": float(r[2]),
                "l": float(r[3]),
                "c": float(r[4]),
                "v": float(r[5]),
            }
            await self._maybe_await(self.shared_state.add_ohlcv(sym, tf, bar))
```

**Data storage key:**
```python
# core/shared_state.py:2764-2776
async def add_ohlcv(self, symbol: str, timeframe: str, bar: OHLCVBar) -> None:
    sym = self._norm_sym(symbol)
    tf = self._norm_tf(timeframe)  # ← Critical: normalization
    key = (sym, tf)
    # ...
    lst = self.market_data.setdefault(key, [])
```

**Data retrieval with fallback keys:**
```python
# core/shared_state.py:2846-2852
async def get_market_data(self, symbol: str, timeframe: str) -> Optional[List[OHLCVBar]]:
    sym = self._norm_sym(symbol)
    tf = self._norm_tf(timeframe)
    rows = self.market_data.get((sym, tf))
    if rows is None:
        rows = self.market_data.get((sym, str(timeframe or "").strip()))
    if rows is None:
        rows = self.market_data.get((symbol, timeframe))  # Fallback to non-normalized
    return rows
```

**TrendHunter requests timeframe:**
```python
# agents/trend_hunter.py:122-125
async def _get_market_data_safe(self, symbol: str, timeframe: str):
    fn = getattr(self.shared_state, "get_market_data", None)
    if not callable(fn):
        return None
    res = fn(symbol, timeframe)
    return (await res) if asyncio.iscoroutine(res) else res
```

### Diagnosis Commands

Check available timeframes:
```python
print("Available market data keys:")
for key in shared_state.market_data.keys():
    symbol, timeframe = key
    rows = shared_state.market_data[key]
    if "ENAUSDT" in symbol:
        print(f"  {symbol:15s} {timeframe:5s} → {len(rows)} bars")

print(f"\nTrendHunter is requesting: {trend_hunter.timeframe}")
```

---

## Case 3: Price Reference Bug

### Symptoms
- OHLCV data exists
- Indicators compute
- **But close prices seem wrong** (e.g., EMA=0.12 but actual price is 65000)
- Signals are inverted or extreme confidence values

### Code Analysis: **This Case is NOT Present**

I've verified the data flow is **correct**:

1. **Binance API returns:**
   ```
   [openTime, open, high, low, close, volume, ...]
   [  0,      1,    2,    3,   4,     5       ]
   ```

2. **MarketDataFeed stores (line 931-937):**
   ```python
   bar = {
       "ts": float(r[0]),  # openTime ✅
       "o": float(r[1]),   # open ✅
       "h": float(r[2]),   # high ✅
       "l": float(r[3]),   # low ✅
       "c": float(r[4]),   # close ✅
       "v": float(r[5]),   # volume ✅
   }
   ```

3. **TrendHunter normalizes (line 101-119):**
   ```python
   return [float(o), float(h), float(l), float(c), float(v)]
   #       [  0    ,   1   ,   2   ,   3   ,   4   ]
   ```

4. **TrendHunter reads close (line 844):**
   ```python
   closes = np.asarray([r[3] for r in rows], dtype=float)  # r[3] = close ✅
   ```

**Conclusion:** Indexing is correct throughout the pipeline. If EMA values seem wrong, it's because:
- The symbol actually trades at that price level (e.g., ENAUSDT at $0.12)
- Or the timeframe is wrong (Case 2)
- Or insufficient data (Case 1)

---

## Recommended Diagnostic Steps

### Step 1: Check Data Availability
```bash
# Run the included diagnostic script
python3 run_diagnostic.py
```

### Step 2: Check Logs for Error Messages
```bash
# Look for Case 1 indicators
grep -E "Insufficient OHLCV|Indicator error|Indicator NaN" logs/clean_run.log

# Look for Case 2 indicators
grep -E "No data for|Wrong timeframe" logs/clean_run.log

# Look for Case 3 indicators (unlikely)
grep -E "price.*wrong|index.*error" logs/clean_run.log
```

### Step 3: Verify Binance API Response
```bash
# Check what Binance actually returns for ENAUSDT
curl "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT"

# Should see something like:
# {"symbol":"ENAUSDT","price":"0.1234"}  ← If price is ~0.12, all is normal
```

### Step 4: Add Debug Logging
Add to `agents/trend_hunter.py` around line 844:

```python
logger.info(
    "[TrendHunter:DEBUG] %s: rows=%d, closes_min=%.8f, closes_max=%.8f, closes_last=%.8f",
    symbol,
    len(rows),
    np.min(closes),
    np.max(closes),
    closes[-1],
)
```

---

## Summary Table

| Case | Symptom | Root Cause | Fix |
|------|---------|-----------|-----|
| **1** | Indicators not computed | <50 bars or NaN values | Wait for data / check config TRENDHUNTER_MIN_DATA |
| **2** | Wrong timeframe | Timeframe mismatch in storage/retrieval | Verify `_norm_tf()` and `self.timeframes` config |
| **3** | Wrong prices | Incorrect close price extraction | ✅ Not present - indexing is correct |

---

## Log Analysis: What We Know

From your logs:
```
2026-03-05 23:02:03,253 WARNING [AppContext] [DEBUG_MDF] fetching OHLCV for ENAUSDT
2026-03-05 23:02:14,071 - DEBUG - [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.12 EMA_L=0.12 HIST=-0.000051
```

**Interpretation:**
- ✅ **Fetching is working** (MDF logs confirm)
- ✅ **EMA is computing** (values show 0.12, which is valid for ENAUSDT)
- ✅ **HISTOGRAM is computing** (value -0.000051 = MACD histogram)
- ✅ **11-second delay is normal** (async fetch → compute pipeline)

**Most likely:** The system is working correctly, and EMA_S=0.12 is the actual close price. Check Binance to confirm ENAUSDT trades at that price level.
