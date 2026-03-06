# Quick Diagnostic Flowchart

## Answer These Questions to Identify the Case

### Question 1: Are indicators being computed at all?
```
Check logs for: "Heuristic check for" OR "Indicator error" OR "Indicator NaN"

YES → Go to Question 2
NO  → CASE 1 (Insufficient Data or NaN)
      Action: Wait for more OHLCV bars or check config TRENDHUNTER_MIN_DATA
```

### Question 2: Are the computed EMA values reasonable?
```
From logs: "EMA_S=0.12 EMA_L=0.12 HIST=-0.000051"

To verify: Check Binance API
$ curl "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT"

If result shows price ≈ 0.12
  → Go to Question 3 (All is normal, skip to Case verification)

If result shows price ≠ 0.12 (e.g., 65000)
  → CASE 3 (Price Reference Bug)
     Action: Check close price extraction
```

### Question 3: Is the timeframe correct?
```
From logs, note the fetch: "[DEBUG_MDF] fetching OHLCV for ENAUSDT"

Check what timeframe was requested:
- Look at market_data_feed.py self.timeframes config
- Look at trend_hunter.py self.timeframe setting
- Compare with the key being stored in shared_state.market_data

Do they match?

YES → All good! System is working. Check signal generation.
NO  → CASE 2 (Wrong Timeframe)
      Action: Fix timeframe normalization or config
```

---

## One-Minute Diagnostic Commands

### Check 1: Verify OHLCV is being stored
```python
import asyncio
from core.shared_state import SharedState
from core.config import Config
import logging

config = Config("config/config.json")
ss = SharedState({}, config, logging.getLogger(), "Check1")

# This would show what's in storage (if you run it after MarketDataFeed warmup)
print(f"Keys in market_data: {list(ss.market_data.keys())[:5]}")
```

### Check 2: Verify Binance price
```bash
# For ENAUSDT
curl -s "https://api.binance.com/api/v3/ticker/price?symbol=ENAUSDT" | jq '.price'

# For multiple symbols from your logs
for SYM in ENAUSDT BCHUSDT PHAUSDT VIRTUALUSDT; do
  PRICE=$(curl -s "https://api.binance.com/api/v3/ticker/price?symbol=$SYM" | jq '.price')
  echo "$SYM: $PRICE"
done
```

### Check 3: Inspect raw log entries
```bash
# Find all OHLCV fetch logs
grep "fetching OHLCV" logs/clean_run.log | head -5

# Find all EMA heuristic logs
grep "Heuristic check for" logs/clean_run.log | head -5

# Look for error messages
grep -E "Insufficient|Indicator error|Indicator NaN" logs/clean_run.log
```

---

## Expected Normal Behavior

When everything is working correctly:

### Log Sequence
```
[Time 1] WARNING [AppContext] [DEBUG_MDF] fetching OHLCV for ENAUSDT
[Time 1] WARNING [AppContext] price update ENAUSDT = 0.123
[Time 2] DEBUG [TrendHunter] Heuristic check for ENAUSDT: EMA_S=0.123 EMA_L=0.121 HIST=-0.000042
```

### Expected Values
- **EMA_S** should be close to current price ± some small divergence
- **EMA_L** should be close to EMA_S for trends, or diverge in ranging markets
- **HIST** should be small (close to 0) unless strong trend

### Time Expectations
- MDF fetches: Every poll cycle (e.g., every 5-10 seconds)
- EMA computes: ~100ms later when signal_engine runs
- Safe to ignore gaps of several minutes between fetches

---

## Data Format Validation

### What OHLCV should look like in shared_state
```python
# After MarketDataFeed warmup:

shared_state.market_data = {
    ("ENAUSDT", "5m"): [
        {"ts": 1741200000.0, "o": 0.120, "h": 0.125, "l": 0.119, "c": 0.123, "v": 1000000.0},
        {"ts": 1741200300.0, "o": 0.123, "h": 0.126, "l": 0.122, "c": 0.124, "v": 1100000.0},
        # ... more bars
    ],
    ("BCHUSDT", "5m"): [
        # ... bars
    ],
}
```

### How TrendHunter processes it
```python
rows = await shared_state.get_market_data("ENAUSDT", "5m")
# rows = [
#   {"ts": ..., "o": 0.120, ...},
#   ...
# ]

normalized_rows = [_std_row(r) for r in rows]
# normalized_rows = [
#   [0.120, 0.125, 0.119, 0.123, 1000000.0],  # [open, high, low, close, volume]
#   ...
# ]

closes = np.asarray([r[3] for r in normalized_rows], dtype=float)
# closes = array([0.123, 0.124, ...])  # All close prices

ema_short = compute_ema(closes, 12)
ema_long = compute_ema(closes, 26)
# ema_short ≈ [nan, ..., 0.1229]  (first 11 are NaN due to period)
# ema_long ≈ [nan, ..., 0.1219]   (first 25 are NaN due to period)
```

---

## Red Flags Indicating a Case

### Red Flag for Case 1
- ❌ Logs show: `"Insufficient OHLCV (n<50)"`
- ❌ Logs show: `"Indicator NaN"`
- ❌ Logs show: `"Indicator error"`
- ❌ No EMA values printed at all for a symbol

### Red Flag for Case 2
- ⚠️ EMA values seem fine but signals are inverted
- ⚠️ Different timeframe than you expected in log message
- ⚠️ Data updates slowly (e.g., one bar per minute when expecting 5m candles)

### Red Flag for Case 3 (Very Unlikely)
- 🔴 EMA value completely wrong (e.g., EMA=50000 when price is 0.12)
- 🔴 Close price extracted from wrong position in data structure
- 🔴 Index mismatch in array access

---

## Action Plan Based on Findings

### If Case 1: Insufficient Data
1. Check `TRENDHUNTER_MIN_DATA` in config → increase if needed
2. Run MarketDataFeed warmup longer
3. Monitor logs for "Insufficient OHLCV" messages
4. Once you see enough bars, indicators will start computing

### If Case 2: Wrong Timeframe
1. Check `_norm_tf()` in `core/shared_state.py` line ~2827
2. Verify `MarketDataFeed.timeframes` matches request
3. Add logging at market_data_feed.py:924 to see what timeframes are being fetched
4. Ensure config has correct TIMEFRAMES setting

### If Case 3: Price Bug (Unlikely)
1. Check `r[3]` vs `r[4]` indexing in market_data_feed.py:931-937
2. Verify `_std_row()` in trend_hunter.py:101-119 order is correct
3. Add debug logging to print raw Binance response vs stored values

---

## Success Criteria

You'll know it's working when:

1. ✅ OHLCV fetching logs appear regularly
2. ✅ EMA values print in logs
3. ✅ EMA values match Binance price within 5%
4. ✅ HISTOGRAM changes sign with market moves
5. ✅ BUY/SELL signals are generated (not just HOLD)
6. ✅ Signal confidence > 0 (not always 0)
