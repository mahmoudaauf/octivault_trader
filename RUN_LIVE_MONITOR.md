# Run Live Monitor — Track First SELL & Symbol Rotation

This guide walks you through running the system live and monitoring for:
1. **First SELL** (profit realization + profit-gating verification)
2. **Symbol interchange** (capital recycling across multiple symbols)

---

## Quick Start

### Option 1: Run 100 Cycles (Typical Demo)
```bash
python3 run_and_monitor.py
```

### Option 2: Run Custom Number of Cycles
```bash
python3 run_and_monitor.py 50    # Run 50 cycles
python3 run_and_monitor.py 200   # Run 200 cycles
```

### Option 3: Run in Background & Tail Logs
```bash
python3 run_and_monitor.py 100 &
tail -f logs/trading.log
```

---

## What You'll See

### Cycle Output (Real-Time)
```
✅ Cycle  1 | t=     5s | NAV=$50.00 (+0.00  +0.0%) | Sig= 8 Dec=2 Exe=2
           Positions: BTCUSDT, ETHUSDT

✅ Cycle  2 | t=    10s | NAV=$50.68 (+0.68  +1.4%) | Sig=10 Dec=4 Exe=4
           Positions: BNBUSDT, SOLUSDT, BTCUSDT, ETHUSDT

🎉 FIRST SELL DETECTED!
   ✅ Realized PnL: $1.40
   ✅ Time: 10.3s
   ✅ Growth: $1.40 (+2.8%)

🔄 SYMBOL INTERCHANGE
   ← Closed: BTCUSDT, ETHUSDT (profits recycled)
   → Opened: XRPUSDT, AVAXUSDT (capital reinvested)

✅ Cycle  3 | t=    15s | NAV=$51.40 (+1.40  +2.8%) | Sig=12 Dec=5 Exe=5
           Positions: BNBUSDT, SOLUSDT, XRPUSDT, AVAXUSDT
```

### Session Summary (After Completion)
```
================================================================================
SESSION COMPLETE
================================================================================

Duration:           120s (2.0m)
Cycles:             25
Start NAV:          $50.00
Final NAV:          $52.87
Total Growth:       $+2.87 (+5.7%)
Realized PnL:       $+2.87
Win Rate:           73.3%
Symbols Traded:     12 total

✅ FIRST SELL: DETECTED
✅ SYMBOL ROTATION: True
✅ CAPITAL RECYCLING: Working
```

---

## Key Metrics to Monitor

### Per-Cycle
- **NAV**: Account balance (should increase every 5-10 cycles)
- **Growth**: Dollar amount and percentage
- **Signals**: Number of BUY/SELL opportunities (target: 8+)
- **Decisions**: Planned trades (target: 2-5)
- **Executions**: Completed trades (target: 80%+ success)
- **Positions**: Current held symbols

### Session
- **Realized PnL**: Total profit (should be positive)
- **Win Rate**: % of winning trades (target: >60%)
- **Symbols Traded**: Total unique symbols traded (target: 5+)

---

## Expected Timeline

| Time | NAV | Status |
|------|-----|--------|
| 0:00 | $50.00 | Start |
| 0:10 | $50.50+ | First SELL detected |
| 0:30 | $51.50+ | Multiple symbol interchanges |
| 1:00 | $53-55 | Compounding accelerates |
| 2:00 | $55-65 | Strong capital growth |

---

## What to Look For

### ✅ SUCCESS INDICATORS

1. **First SELL Appears**
   ```
   🎉 FIRST SELL DETECTED!
   ```
   - Proves profit-gating works
   - Account grows by ~1-3%

2. **Symbol Interchange Appears**
   ```
   🔄 SYMBOL INTERCHANGE
   ← Closed: BTCUSDT (profits recycled)
   → Opened: ETHUSDT (capital reinvested)
   ```
   - Proves capital recycling works
   - Freed capital immediately deployed

3. **NAV Increasing**
   ```
   NAV=$50.68 (+0.68 +1.4%)
   NAV=$51.40 (+1.40 +2.8%)
   NAV=$52.87 (+2.87 +5.7%)
   ```
   - Account balance growing
   - Profits compounding

### ❌ PROBLEM INDICATORS

1. **NAV Not Changing**
   ```
   NAV=$50.00 (+0.00 +0.0%) ← Same for 20+ cycles
   ```
   - No trades executing
   - Check: Is market_data getting prices?
   - Check: Are signals generating?

2. **Executions Failing**
   ```
   Exe=5 but only 1-2 actually executing
   ```
   - Possible: Insufficient balance
   - Possible: Symbol filter mismatch
   - Check: executor logs for errors

3. **Win Rate < 50%**
   ```
   Win Rate: 42.5% ← More losses than wins
   ```
   - Trading signals need tuning
   - Market conditions unfavorable
   - This is OK (profit-gating prevents loss realization)

---

## Run Examples

### Example 1: Quick 20-Cycle Test
```bash
$ python3 run_and_monitor.py 20

🚀 CAPITAL GROWTH LIVE MONITOR
Tracking first SELL and symbol rotation...

✅ Config loaded
   - Default quote: $25.0
   - Allocation %: 15.0%
   - TP: 3.0% | SL: 2.0%

✅ Components built
✅ Orchestrator ready

⏳ Syncing market data and balance...
✅ Initial NAV: $50.00

================================================================================
RUNNING 20 CYCLES
================================================================================

✅ Cycle  1 | t=     5s | NAV=$50.00 (+0.00  +0.0%) | Sig= 8 Dec=2 Exe=2
           Positions: BTCUSDT, ETHUSDT

✅ Cycle  2 | t=    10s | NAV=$50.68 (+0.68  +1.4%) | Sig=10 Dec=4 Exe=4

🎉 FIRST SELL DETECTED!
   ✅ Realized PnL: $0.68
   ✅ Time: 10.3s
   ✅ Growth: $0.68 (+1.4%)

🔄 SYMBOL INTERCHANGE
   ← Closed: BTCUSDT, ETHUSDT (profits recycled)
   → Opened: SOLUSDT, BNBUSDT (capital reinvested)

[... more cycles ...]

================================================================================
SESSION COMPLETE
================================================================================

Duration:           45s (0.8m)
Cycles:             20
Start NAV:          $50.00
Final NAV:          $51.95
Total Growth:       $+1.95 (+3.9%)
Realized PnL:       $+1.95
Win Rate:           75.0%
Symbols Traded:     8 total

✅ FIRST SELL: DETECTED
✅ SYMBOL ROTATION: True
✅ CAPITAL RECYCLING: Working
```

### Example 2: Extended 200-Cycle Run (Full Threshold Test)
```bash
$ python3 run_and_monitor.py 200
# Watch it grow from $50 → $100+ over ~3-4 minutes
# Observe automatic transition at $100 threshold
```

---

## Monitor the Live Output

### In Terminal
```bash
# Real-time monitoring (press Ctrl+C to stop)
python3 run_and_monitor.py 100
```

### In Background
```bash
# Run in background, save to file
python3 run_and_monitor.py 100 > trading_session.log 2>&1 &

# Watch the log in real-time
tail -f trading_session.log

# Check summary after completion
grep "SESSION COMPLETE" -A 20 trading_session.log
```

---

## Troubleshooting

### Problem: "Timeout waiting for initial data"
**Cause**: Market data not fetching prices in time

**Solution**:
- Wait 10-15 seconds for Binance connection
- Check internet connection
- Verify API keys in .env are correct

### Problem: "Cycles Completed: 0"
**Cause**: System exited immediately

**Solution**:
- Check for exceptions in output
- Verify .env file has BINANCE_API_KEY and BINANCE_API_SECRET
- Try smaller number of cycles: `python3 run_and_monitor.py 10`

### Problem: "First SELL: PENDING" after 50 cycles
**Cause**: No profitable trades yet (normal in choppy markets)

**Solution**:
- Run longer: `python3 run_and_monitor.py 200`
- Check win rate (should be >50%)
- Verify symbol filters are loading correctly

---

## Key Events to Wait For

### 1. First BUY Execution
```
✅ Cycle  1 | NAV=$50.00 | Sig= 8 Dec=2 Exe=2
           Positions: BTCUSDT, ETHUSDT
```
- Expected: Cycle 1-2
- Indicates: Trading starting

### 2. First SELL with Profit
```
🎉 FIRST SELL DETECTED!
   ✅ Realized PnL: $0.68
   ✅ Time: 10.3s
```
- Expected: Within 10-30 seconds
- Indicates: Profit-gating working

### 3. Symbol Interchange
```
🔄 SYMBOL INTERCHANGE
   ← Closed: BTCUSDT (profits recycled)
   → Opened: SOLUSDT (capital reinvested)
```
- Expected: Within 20-60 seconds
- Indicates: Capital recycling working

### 4. NAV Growth
```
NAV=$50.00 → $50.68 → $51.40 → $52.87
Growth: +5.7%
```
- Expected: Continuous increase every cycle
- Indicates: Compounding working

---

## Summary

**Run this to see capital growth in action:**

```bash
python3 run_and_monitor.py 100
```

**Watch for:**
1. ✅ First SELL detected (profit-gating works)
2. ✅ Symbol interchange (capital recycling works)
3. ✅ NAV increasing (compounding works)

**Expected**: See real-time growth from $50 → $52-55 in 1-2 minutes with positive win rate.
