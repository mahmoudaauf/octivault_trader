# 6-Hour Trading Session - Quick Reference Guide

## 📊 Session Information

- **Duration:** 6 hours (21,600 seconds)
- **Start Time:** Fri May 1, 17:17:46 EEST 2026
- **End Time:** Fri May 1, 23:17:46 EEST 2026
- **Process PID:** 62926
- **Status:** ✅ RUNNING

---

## 🎯 Session Configuration

All checkpoints have been cleared for a fresh start:
- ✅ `state/checkpoint.json` - Removed
- ✅ `checkpoint_metrics.json` - Removed
- ✅ Checkpoint logs - Removed
- ✅ Live log - Cleared
- ✅ System fully restarted with `TRADING_DURATION_HOURS=6`

---

## 📈 Real-Time Monitoring Commands

### Watch Live Trading Activity
```bash
tail -f /tmp/octivault_live.log
```

### Watch Session Initialization & Phases
```bash
tail -f /tmp/octivault_6hour_session.log
```

### Monitor Key Metrics Only
```bash
tail -f /tmp/octivault_live.log | grep -E "LOOP_SUMMARY|TRADE_AUDIT|NAV="
```

### Watch Loop Iterations
```bash
tail -f /tmp/octivault_live.log | grep "LOOP_SUMMARY"
```

### Watch Trades Executed
```bash
tail -f /tmp/octivault_live.log | grep "TRADE_AUDIT"
```

### Watch NAV Updates
```bash
tail -f /tmp/octivault_live.log | grep "NAV="
```

### Check Process Status
```bash
ps aux | grep MASTER_SYSTEM_ORCHESTRATOR | grep -v grep
```

### Get Process Statistics
```bash
ps aux | grep MASTER_SYSTEM_ORCHESTRATOR | grep -v grep | awk '{print "PID: " $2 ", CPU: " $3 "%, Memory: " $4 "%, Time: " $11 ":" $12}'
```

---

## 🛠️ Control Commands

### Stop the Session (Early Exit)
```bash
kill 62926
```

### Restart Session (New Checkpoints)
```bash
/tmp/run_6hour_session.sh
```

### Check If Still Running
```bash
ps aux | grep 62926 | grep -v grep && echo "Running" || echo "Stopped"
```

---

## 📊 Analysis Commands

### Count Total Loop Iterations
```bash
grep -c "LOOP_SUMMARY" /tmp/octivault_live.log
```

### Count Trades Executed
```bash
grep -c "TRADE_AUDIT" /tmp/octivault_live.log
```

### Extract Latest NAV Value
```bash
tail -1 /tmp/octivault_live.log | grep -oP 'NAV=\K[0-9.]+'
```

### Get Initial NAV
```bash
grep "NAV=" /tmp/octivault_live.log | head -1 | grep -oP 'NAV=\K[0-9.]+'
```

### Show Error Count
```bash
grep -c "ERROR\|CRITICAL" /tmp/octivault_live.log
```

### Show Warning Count
```bash
grep -c "WARNING" /tmp/octivault_live.log
```

### Get Symbols Being Tracked
```bash
grep "accepted_symbols=" /tmp/octivault_6hour_session.log | head -1
```

### View Portfolio State
```bash
tail -f /tmp/octivault_live.log | grep "Portfolio\|FLAT\|LONG\|MIXED"
```

---

## 🔍 Detailed Log Analysis

### Show All Trades with Details
```bash
grep "TRADE_AUDIT" /tmp/octivault_live.log
```

### Show Signal Generation
```bash
grep "SwingTradeHunter\|IPO\|DipSniper" /tmp/octivault_live.log
```

### Show Position Management
```bash
grep "POSITION_OPENED\|POSITION_CLOSED" /tmp/octivault_live.log
```

### Show Bootstrap Events
```bash
grep -i "bootstrap" /tmp/octivault_6hour_session.log
```

### Show Health Status
```bash
grep "health\|HEALTHY\|DEGRADED" /tmp/octivault_live.log | tail -20
```

### Show Capital Changes
```bash
grep "capital_free\|reserved" /tmp/octivault_live.log | tail -10
```

---

## ⏰ Timeline Reference

```
17:17:46  ← Session Started (checkpoint cleared)
17:20:30  ← System fully bootstrapped & trading
18:17:46  ← 1 hour mark
19:17:46  ← 2 hours mark
20:17:46  ← 3 hours mark
21:17:46  ← 4 hours mark
22:17:46  ← 5 hours mark
23:17:46  ← Session Ends (auto-shutdown)
```

---

## 📁 Log Files

| File | Purpose | Location |
|------|---------|----------|
| Main Session Log | Initialization & phases | `/tmp/octivault_6hour_session.log` |
| Live Activity Log | Real-time trading data | `/tmp/octivault_live.log` |
| Master Orchestrator Log | System orchestration | `/tmp/octivault_master_orchestrator.log` |

---

## 🚀 What the System is Doing

Every 2-3 seconds, the system:
1. ✅ Syncs authoritative balance from Binance
2. ✅ Recalculates NAV (cash + positions)
3. ✅ Monitors all 10 symbols for signals
4. ✅ Generates buy/sell intents (SwingTradeHunter, etc.)
5. ✅ Executes trades if conditions met
6. ✅ Manages positions (TP/SL enforcement)
7. ✅ Compounds profits (reinvestment)
8. ✅ Checks system health

Every 15 minutes, the system:
1. ✅ Calculates performance metrics
2. ✅ Runs PI control feedback algorithm
3. ✅ Adjusts confidence_floor, size_multiplier, etc.
4. ✅ Tunes toward 2%/day objective

---

## 🎯 Expected Outcomes After 6 Hours

- ✅ 20,000+ loop iterations completed
- ✅ 50-200+ trades executed
- ✅ Portfolio growth tracked and logged
- ✅ Real-time balance tracking verified
- ✅ All 10 symbols actively monitored
- ✅ System health maintained at HEALTHY
- ✅ Feedback system running 24 tuning cycles

---

## 📋 How to Generate Final Report

After the 6-hour session completes:

```bash
/Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/generate_session_report.sh
```

This creates a markdown report with:
- Capital performance (NAV start → end)
- Trading activity (# trades, win rate, etc.)
- Portfolio health metrics
- System uptime & reliability
- Detailed recommendations

---

## 🆘 Troubleshooting

### Process Not Running?
```bash
ps aux | grep 62926
# If not running, restart with:
/tmp/run_6hour_session.sh
```

### Live Log Not Updating?
```bash
# Check if log file exists
ls -lah /tmp/octivault_live.log

# If empty, process might have crashed
tail -f /tmp/octivault_6hour_session.log
```

### High CPU Usage?
```bash
# This is normal during active trading
# System should stabilize at 3-5% CPU
ps aux | grep 62926
```

### Many Errors in Logs?
```bash
# Check error count
grep "ERROR\|CRITICAL" /tmp/octivault_live.log | wc -l

# See actual errors
grep "ERROR\|CRITICAL" /tmp/octivault_live.log | head -20
```

---

## 📞 Key Metrics to Track

As the session progresses, monitor:

1. **NAV (Net Asset Value)** - Should show steady growth
2. **Capital Free** - Cash available for new trades
3. **Trade Count** - Should increase consistently
4. **Portfolio State** - Should alternate between FLAT and trading states
5. **Health Status** - Should remain HEALTHY
6. **Loop Frequency** - Should be consistent ~2 sec intervals
7. **Symbol Diversity** - All 10 symbols should be actively processed

---

**Session Created:** 2026-05-01 17:17:46 EEST  
**Status:** ✅ 6-HOUR SESSION RUNNING  
**Auto-Shutdown:** 2026-05-01 23:17:46 EEST
