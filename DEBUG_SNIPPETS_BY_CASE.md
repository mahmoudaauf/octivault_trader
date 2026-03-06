# Debug Logging: Case-Specific Code Snippets

Use these code additions to identify which case is happening. Add them temporarily to gather diagnostic info.

---

## Case 1 Detection: Add to `agents/trend_hunter.py`

### Location: After line 835 in `_generate_signal()` method

```python
# ADD THIS BLOCK for Case 1 diagnosis:
async def _generate_signal(self, symbol: str, is_ml_capable: bool = False) -> Tuple[str, float, str]:
    data = await self._get_market_data_safe(symbol, self.timeframe)
    if data is None:
        return "HOLD", 0.0, "OHLCV None"

    rows = [self._std_row(r) for r in data]
    rows = [r for r in rows if r is not None]

    # ===== CASE 1 DIAGNOSIS =====
    logger.info(
        "[DEBUG:CASE1] %s: found_rows=%d (fetched=%d)",
        symbol,
        len(rows),
        len(data)
    )
    # ============================

    min_required = 50
    cfg_min = self._cfg("TRENDHUNTER_MIN_DATA", 50)
    if isinstance(cfg_min, dict):
        min_required = int(cfg_min.get(symbol, cfg_min.get("default", 50)))
    elif isinstance(cfg_min, int):
        min_required = cfg_min

    # ===== CASE 1 DIAGNOSIS =====
    if len(rows) < min_required:
        logger.warning(
            "[DEBUG:CASE1] %s: INSUFFICIENT DATA - %d rows < %d required",
            symbol,
            len(rows),
            min_required,
        )
        return "HOLD", 0.1, f"Insufficient OHLCV ({len(rows)}<{min_required})"
    # ============================

    closes = np.asarray([r[3] for r in rows], dtype=float)

    # ===== CASE 1 DIAGNOSIS =====
    logger.info(
        "[DEBUG:CASE1] %s: closes=[%.8f...%.8f], len=%d",
        symbol,
        closes[0],
        closes[-1],
        len(closes),
    )
    # ============================

    # ... rest of method
```

### Location: After line 865 (after indicator computation)

```python
    try:
        if _HAS_TALIB:
            ema_short = talib.EMA(closes, timeperiod=fast)
            ema_long = talib.EMA(closes, timeperiod=slow)
            macd_line, sig_line, hist = talib.MACD(closes)
        else:
            ema_short = compute_ema(closes, fast)
            ema_long = compute_ema(closes, slow)
            macd_line, sig_line, hist = compute_macd(closes)
        
        # ===== CASE 1 DIAGNOSIS =====
        logger.info(
            "[DEBUG:CASE1] %s: indicator_arrays - ema_short_len=%d, ema_long_len=%d, hist_len=%d",
            symbol,
            len(ema_short) if ema_short is not None else 0,
            len(ema_long) if ema_long is not None else 0,
            len(hist) if hist is not None else 0,
        )
        # ============================
        
        # Guard against short series/NaNs at the tail
        tail_vals = [np.asarray(ema_short)[-1], np.asarray(ema_long)[-1], np.asarray(hist)[-1]]
        
        # ===== CASE 1 DIAGNOSIS =====
        has_nans = [np.isnan(float(v)) for v in tail_vals]
        logger.info(
            "[DEBUG:CASE1] %s: tail_values - ema_s=%.8f (nan=%s), ema_l=%.8f (nan=%s), hist=%.8f (nan=%s)",
            symbol,
            tail_vals[0],
            has_nans[0],
            tail_vals[1],
            has_nans[1],
            tail_vals[2],
            has_nans[2],
        )
        # ============================
        
        if any(np.isnan(float(v)) for v in tail_vals):
            logger.error("[DEBUG:CASE1] %s: INDICATOR NaN detected", symbol)
            return "HOLD", 0.0, "Indicator NaN"
    except Exception as e:
        logger.error("[DEBUG:CASE1] %s: INDICATOR ERROR: %s", symbol, e, exc_info=True)
        return "HOLD", 0.0, "Indicator error"
```

---

## Case 2 Detection: Add to `core/market_data_feed.py`

### Location: Around line 924-935 (OHLCV fetching loop)

```python
    # Lightweight tail refresh on each timeframe
    # 🔎 2) Log before fetching OHLCV
    self._logger.warning("[DEBUG_MDF] fetching OHLCV for %s", sym)
    
    # ===== CASE 2 DIAGNOSIS =====
    self._logger.warning(
        "[DEBUG:CASE2] MarketDataFeed timeframes=%s, about to fetch for %s",
        self.timeframes,
        sym,
    )
    # ============================
    
    for tf in self.timeframes:
        try:
            # ===== CASE 2 DIAGNOSIS =====
            self._logger.warning("[DEBUG:CASE2] Fetching %s-%s", sym, tf)
            # ============================
            
            async def _fetch_tail():
                return await ec.get_ohlcv(sym, tf, limit=3)
            rows = await self._with_retries(_fetch_tail, f"poll.get_ohlcv[{sym},{tf}]")
            rows = self._sanitize_ohlcv(rows or [])
            
            # ===== CASE 2 DIAGNOSIS =====
            if rows:
                self._logger.warning(
                    "[DEBUG:CASE2] Fetched %s-%s: %d rows, closes=[%.8f...%.8f]",
                    sym,
                    tf,
                    len(rows),
                    float(rows[-1][4]),
                    float(rows[-1][4]),  # Most recent close
                )
            # ============================
            
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
                bars_added += 1
        except Exception:
            # ===== CASE 2 DIAGNOSIS =====
            self._logger.warning("[DEBUG:CASE2] Failed to fetch %s-%s", sym, tf, exc_info=True)
            # ============================
            self._logger.debug("poll.get_ohlcv failed for %s %s", sym, tf, exc_info=True)
```

### Location: Add to `core/shared_state.py` `add_ohlcv()` method (around line 2764)

```python
    async def add_ohlcv(self, symbol: str, timeframe: str, bar: OHLCVBar) -> None:
        """
        Append/merge a single OHLCV bar ensuring ascending ts and 6-field hygiene.
        bar keys: ts,o,h,l,c,v  (epoch seconds float)
        """
        sym = self._norm_sym(symbol)
        tf = self._norm_tf(timeframe)
        key = (sym, tf)
        
        # ===== CASE 2 DIAGNOSIS =====
        logger.info(
            "[DEBUG:CASE2] add_ohlcv: symbol=%s→%s, tf=%s→%s, key=(%s,%s), c=%.8f",
            symbol,
            sym,
            timeframe,
            tf,
            sym,
            tf,
            bar.get("c", 0),
        )
        # ============================
```

---

## Case 3 Detection: Add to `agents/trend_hunter.py`

### Location: In `_std_row()` method (around line 101)

```python
    def _std_row(self, r):
        try:
            if isinstance(r, dict):
                d = r
                o = d.get("o", d.get("open"))
                h = d.get("h", d.get("high"))
                l = d.get("l", d.get("low"))
                c = d.get("c", d.get("close", d.get("last", d.get("price"))))
                v = d.get("v", d.get("volume"))
                
                # ===== CASE 3 DIAGNOSIS =====
                if c and o and h and l:
                    if c > h or c < l:
                        logger.warning(
                            "[DEBUG:CASE3] Malformed dict: c=%.8f not in [l=%.8f, h=%.8f]",
                            c, l, h,
                        )
                # ============================
                
                if None in (o, h, l, c, v):
                    return None
                return [float(o), float(h), float(l), float(c), float(v)]
            
            seq = list(r)
            
            # ===== CASE 3 DIAGNOSIS =====
            if len(seq) >= 5:
                # Assuming Binance format: [time, o, h, l, c, v, ...]
                #                        [  0,  1, 2, 3, 4, 5  ]
                logger.debug(
                    "[DEBUG:CASE3] Raw sequence from API: len=%d, c_at_idx4=%.8f, "
                    "vals=[%.8f(0), %.8f(1), %.8f(2), %.8f(3), %.8f(4), %.8f(5)]",
                    len(seq),
                    float(seq[4]) if len(seq) > 4 else 0,
                    float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]),
                    float(seq[4]), float(seq[5]) if len(seq) > 5 else 0,
                )
            # ============================
            
            if len(seq) >= 6:
                seq = seq[-5:]
            if len(seq) == 5:
                return [float(x) for x in seq]
        except Exception:
            return None
        return None
```

---

## Master Diagnostic Function

Add this to any component to run all checks at once:

```python
async def diagnose_ohlcv_indicators():
    """Run all Case 1, 2, 3 diagnostics."""
    import logging
    logger = logging.getLogger("Diagnostic")
    
    # Assumes you have access to:
    # - shared_state
    # - trend_hunter
    # - exchange_client
    
    print("\n" + "="*80)
    print("OHLCV vs Indicator Diagnostic")
    print("="*80)
    
    # CASE 1: Data Availability
    print("\n[CASE 1] Checking data availability...")
    test_symbol = "ENAUSDT"
    test_tf = "5m"
    
    rows = await shared_state.get_market_data(test_symbol, test_tf)
    if not rows:
        print(f"  ❌ CASE 1 DETECTED: No data for {test_symbol}-{test_tf}")
        print(f"     Available keys: {list(shared_state.market_data.keys())[:5]}")
        return
    
    print(f"  ✅ Found {len(rows)} bars for {test_symbol}-{test_tf}")
    
    if len(rows) < 50:
        print(f"  ❌ CASE 1 DETECTED: Only {len(rows)} < 50 required")
        return
    
    # CASE 2: Timeframe Matching
    print("\n[CASE 2] Checking timeframe matching...")
    if test_tf in [key[1] for key in shared_state.market_data.keys()]:
        print(f"  ✅ Timeframe {test_tf} found in storage")
    else:
        print(f"  ❌ CASE 2 DETECTED: Timeframe {test_tf} not in storage")
        print(f"     Available: {set(key[1] for key in shared_state.market_data.keys())}")
        return
    
    # CASE 3: Price Extraction
    print("\n[CASE 3] Checking price extraction...")
    try:
        # Get current price from Binance
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={test_symbol}") as resp:
                data = await resp.json()
                binance_price = float(data["price"])
        
        # Get last OHLCV close
        last_close = float(rows[-1]["c"])
        
        diff_pct = abs(last_close - binance_price) / binance_price * 100
        
        if diff_pct < 1:
            print(f"  ✅ Price matches: OHLCV={last_close:.8f}, Binance={binance_price:.8f} (diff={diff_pct:.2f}%)")
        else:
            print(f"  ⚠️ CASE 3 POSSIBLE: OHLCV={last_close:.8f}, Binance={binance_price:.8f} (diff={diff_pct:.2f}%)")
    except Exception as e:
        print(f"  ⚠️ Could not verify: {e}")
    
    # Indicator Computation
    print("\n[INDICATORS] Computing indicators...")
    try:
        import numpy as np
        from utils.indicators import compute_ema, compute_macd
        
        normalized = [trend_hunter._std_row(r) for r in rows]
        normalized = [r for r in normalized if r is not None]
        
        closes = np.asarray([r[3] for r in normalized], dtype=float)
        
        ema_s = compute_ema(closes, 12)
        ema_l = compute_ema(closes, 26)
        _, _, hist = compute_macd(closes)
        
        print(f"  ✅ Indicators computed successfully")
        print(f"     EMA_SHORT[-1]: {float(ema_s[-1]):.8f}")
        print(f"     EMA_LONG[-1]:  {float(ema_l[-1]):.8f}")
        print(f"     HISTOGRAM[-1]: {float(hist[-1]):.8f}")
    except Exception as e:
        print(f"  ❌ Indicator computation failed: {e}")
    
    print("\n" + "="*80)
    print("Diagnostic complete")
    print("="*80)

# Run it:
# asyncio.run(diagnose_ohlcv_indicators())
```

---

## How to Use These Snippets

1. **For Case 1 diagnosis:**
   - Add the logging blocks to trend_hunter.py
   - Run the system
   - Look for `[DEBUG:CASE1]` lines in logs
   - Check if `found_rows` < `min_required`

2. **For Case 2 diagnosis:**
   - Add the logging blocks to market_data_feed.py and shared_state.py
   - Run the system
   - Look for `[DEBUG:CASE2]` lines
   - Verify timeframe in key matches what you requested

3. **For Case 3 diagnosis:**
   - Add the logging blocks to trend_hunter.py
   - Compare close price in logs with Binance API price
   - Should be within 1% (small lag is ok)

4. **For all cases:**
   - Use the master diagnostic function
   - Runs all checks in sequence
   - Identifies which case applies

---

## Expected Diagnostic Output

### Case 1 Output:
```
[DEBUG:CASE1] ENAUSDT: found_rows=25 (fetched=25)
[DEBUG:CASE1] ENAUSDT: INSUFFICIENT DATA - 25 rows < 50 required
```

### Case 2 Output:
```
[DEBUG:CASE2] MarketDataFeed timeframes=['5m', '1h'], about to fetch for ENAUSDT
[DEBUG:CASE2] Fetching ENAUSDT-5m
[DEBUG:CASE2] Fetched ENAUSDT-5m: 3 rows, closes=[0.12345678...0.12345678]
[DEBUG:CASE2] add_ohlcv: symbol=ENAUSDT→ENAUSDT, tf=5m→5m, key=(ENAUSDT,5m), c=0.12345678
```

### Case 3 Output:
```
[DEBUG:CASE3] Raw sequence from API: len=12, c_at_idx4=0.12345678, vals=[timestamp, 0.12, 0.125, 0.119, 0.123, 1000000]
[DEBUG:CASE3] Malformed dict: c=65000.00 not in [l=0.1, h=0.2]  ← This would indicate the bug
```

### All Normal Output:
```
[DEBUG:CASE1] ENAUSDT: found_rows=150 (fetched=150)
[DEBUG:CASE1] ENAUSDT: closes=[0.11234567...0.12345678], len=150
[DEBUG:CASE1] ENAUSDT: indicator_arrays - ema_short_len=150, ema_long_len=150, hist_len=150
[DEBUG:CASE1] ENAUSDT: tail_values - ema_s=0.12345678 (nan=False), ema_l=0.12123456 (nan=False), hist=-0.00005100 (nan=False)
```

---

## After Diagnosis

Once you've added these snippets and identified the case:

1. **Remove the debug logging** to reduce log volume
2. **Implement the fix** for that specific case
3. **Test with the diagnostic again** to confirm fix worked

These snippets are meant for temporary debugging, not production code.
