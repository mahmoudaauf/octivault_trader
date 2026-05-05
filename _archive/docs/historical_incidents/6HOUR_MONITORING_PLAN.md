# 6-Hour Bot Monitoring Plan

**Start Time:** 2026-05-04 01:40 (approximate)
**Expected End Time:** 2026-05-04 07:40
**Bot PID:** 96511
**Status:** ✅ RUNNING

---

## What to Expect During 6-Hour Run

### Phase 1: Auto-Recovery & Dust Healing (0-15 minutes)

**Expected Events:**
- ✅ Auto-recovery trigger activates
- ✅ Mode switches to RECOVERY
- ✅ LiquidationAgent starts liquidating dust (every 10 seconds)
- ✅ Positions consolidate from 35 → <10
- ✅ Capital freed from dust

**Log Indicators:**
```bash
grep -i "auto-recovery\|recovery.*mode\|dust.*liquidat" logs/octivault_master_orchestrator.log
grep -i "position.*closed\|liquidat.*complete" logs/octivault_master_orchestrator.log
```

**Success Criteria:**
- Sees 20+ "POSITION FULLY CLOSED" messages
- Position count drops significantly
- Kill-switch status changes

---

### Phase 2: Kill-Switch Disabling (15-30 minutes)

**Expected Events:**
- ✅ Kill-switch detects healing complete
- ✅ Kill-switch automatically disables
- ✅ Portfolio can now accept BUY orders

**Log Indicators:**
```bash
grep -i "kill.switch.*disabled\|compound.*growth.*enabled" logs/octivault_master_orchestrator.log
```

**Success Criteria:**
- See "Kill-switch disabled" or "Compound growth enabled" message
- TRADE_REJECTED messages change to TRADE_SKIPPED or TRADE_EXECUTED

---

### Phase 3: Trading Resume (30+ minutes)

**Expected Events:**
- ✅ First BUY order submitted
- ✅ Orders start filling
- ✅ SELL signals on profitable positions
- ✅ Regular 3-5 trades per cycle (every 30 seconds)

**Log Indicators:**
```bash
grep -i "trade_executed\|order.*filled\|buy.*order\|sell.*order" logs/octivault_master_orchestrator.log | tail -20
```

**Success Criteria:**
- Multiple TRADE_EXECUTED events
- NAV increasing
- Portfolio showing active trading

---

## Monitoring Checklist

### Every 15 Minutes: Quick Status Check

```bash
# 1. Is bot still running?
pgrep -f "MASTER" && echo "✅ ALIVE" || echo "❌ CRASHED"

# 2. Latest log activity (last 5 lines)
tail -5 logs/octivault_master_orchestrator.log

# 3. Error count
grep -i "error\|failed\|critical" logs/octivault_master_orchestrator.log | wc -l

# 4. Trade count
grep "TRADE_EXECUTED" logs/octivault_master_orchestrator.log | wc -l
```

### Every 30 Minutes: Detailed Metrics

```bash
# Current NAV
grep "NAV.*=" logs/octivault_master_orchestrator.log | tail -1

# Position count
grep "positions=" logs/octivault_master_orchestrator.log | tail -1

# Kill-switch status
grep -i "kill.switch" logs/octivault_master_orchestrator.log | tail -3

# Recent trades
grep "TRADE_EXECUTED\|TRADE_SKIPPED" logs/octivault_master_orchestrator.log | tail -10
```

### Hourly: Comprehensive Report

```bash
# Copy this to get full hourly report
echo "=== HOURLY MONITORING REPORT ===" >> monitoring_report.txt
echo "Time: $(date)" >> monitoring_report.txt
echo "" >> monitoring_report.txt

echo "Bot Status:" >> monitoring_report.txt
pgrep -f "MASTER" && echo "✅ Running" >> monitoring_report.txt || echo "❌ Crashed" >> monitoring_report.txt
echo "" >> monitoring_report.txt

echo "Recent Activity (last 10 lines):" >> monitoring_report.txt
tail -10 logs/octivault_master_orchestrator.log >> monitoring_report.txt
echo "" >> monitoring_report.txt

echo "Trade Summary:" >> monitoring_report.txt
echo "- Executed: $(grep "TRADE_EXECUTED" logs/octivault_master_orchestrator.log | wc -l)" >> monitoring_report.txt
echo "- Skipped: $(grep "TRADE_SKIPPED" logs/octivault_master_orchestrator.log | wc -l)" >> monitoring_report.txt
echo "- Rejected: $(grep "TRADE_REJECTED" logs/octivault_master_orchestrator.log | wc -l)" >> monitoring_report.txt
echo "" >> monitoring_report.txt

tail -20 monitoring_report.txt
```

---

## Key Metrics to Track

| Metric | What to Watch | Good Sign | Problem |
|--------|---------------|-----------|---------|
| **Bot Process** | PID 96511 status | Always running | Crashes or missing |
| **Log File Growth** | Bytes increasing | Growing steadily | Stopped growing |
| **Error Messages** | Critical errors | None | Multiple errors |
| **Trade Count** | TRADE_EXECUTED | ≥5 in 6 hours | 0 executions |
| **NAV Trend** | Navigation value | Increasing | Flat or decreasing |
| **Position Count** | Active positions | 3-5 | >20 or 0 |
| **Kill-Switch** | Enabled/Disabled | Disabled after 30min | Still enabled |
| **Dust Liquidations** | POSITION_CLOSED | >20 messages | <5 messages |

---

## Common Issues & Fixes

### Issue 1: Bot Still Shows 35 Positions After 15 Minutes
**Problem:** Auto-recovery not activating
**Fix:**
```bash
# Check if RECOVERY mode was set
grep -i "auto-recovery\|recovery.*mode" logs/octivault_master_orchestrator.log

# If missing, manually trigger healing
bash scripts/emergency_liquidate.sh  # If available

# Or restart with explicit mode override
pkill -9 -f "MASTER"
export STARTUP_MODE_OVERRIDE=RECOVERY
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > logs/octivault_master_orchestrator.log 2>&1 &
```

### Issue 2: Kill-Switch Still Enabled After 30 Minutes
**Problem:** Dust not fully cleared
**Fix:**
```bash
# Check remaining dust positions
grep "dust\|position" logs/octivault_master_orchestrator.log | tail -20

# Check liquidation status
grep -i "liquidat\|heal" logs/octivault_master_orchestrator.log | tail -10

# If stuck, manually liquidate top 10 dust positions
# Contact support or manually SELL smallest positions
```

### Issue 3: No Trades After 1 Hour Despite Kill-Switch Disabled
**Problem:** PRETRADE thresholds still too high
**Fix:**
```bash
# Check what threshold is blocking trades
grep "pretrade_effect_gate" logs/octivault_master_orchestrator.log | tail -5

# Check if stall relief is working
grep "StallRelief" logs/octivault_master_orchestrator.log | tail -5

# The system auto-reduces thresholds, just give it more time
# Or manually restart bot to reset thresholds
```

### Issue 4: WebSocket Disconnection
**Problem:** "WebSocket connection failed"
**Fix:**
```bash
# Bot auto-reconnects, check if it succeeded
grep -i "websocket.*reconnect\|websocket.*connected" logs/octivault_master_orchestrator.log | tail -3

# If repeatedly failing, restart
pkill -9 -f "MASTER"
sleep 3
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > logs/octivault_master_orchestrator.log 2>&1 &
```

---

## Commands for 6-Hour Monitoring

### Start of Run (Right Now)

```bash
# 1. Record baseline
echo "6-Hour Run Started: $(date)" > /tmp/run_baseline.txt
ps aux | grep MASTER >> /tmp/run_baseline.txt
tail -50 logs/octivault_master_orchestrator.log >> /tmp/run_baseline.txt

# 2. Start monitoring log growth
wc -l logs/octivault_master_orchestrator.log > /tmp/log_size_start.txt
```

### During Run (Every 30 minutes)

```bash
# Quick status (can run while bot is running)
echo "=== Status Check: $(date) ==="
pgrep -f "MASTER" && echo "✅ Bot alive" || echo "❌ Bot down"
tail -1 logs/octivault_master_orchestrator.log
echo "Trades executed: $(grep TRADE_EXECUTED logs/octivault_master_orchestrator.log | wc -l)"
echo "Positions liquidated: $(grep 'POSITION FULLY CLOSED' logs/octivault_master_orchestrator.log | wc -l)"
```

### End of Run (After 6 hours)

```bash
# 1. Final status
echo "6-Hour Run Completed: $(date)" > /tmp/run_final_report.txt
echo "" >> /tmp/run_final_report.txt

# 2. Summary
echo "=== FINAL METRICS ===" >> /tmp/run_final_report.txt
echo "Total trades executed: $(grep TRADE_EXECUTED logs/octivault_master_orchestrator.log | wc -l)" >> /tmp/run_final_report.txt
echo "Total trades skipped: $(grep TRADE_SKIPPED logs/octivault_master_orchestrator.log | wc -l)" >> /tmp/run_final_report.txt
echo "Positions liquidated: $(grep 'POSITION FULLY CLOSED' logs/octivault_master_orchestrator.log | wc -l)" >> /tmp/run_final_report.txt
echo "Errors encountered: $(grep -i 'error' logs/octivault_master_orchestrator.log | wc -l)" >> /tmp/run_final_report.txt
echo "" >> /tmp/run_final_report.txt

# 3. Last 100 lines of activity
echo "=== FINAL 100 LINES OF LOG ===" >> /tmp/run_final_report.txt
tail -100 logs/octivault_master_orchestrator.log >> /tmp/run_final_report.txt

# 4. Display report
cat /tmp/run_final_report.txt
```

---

## Expected Outcomes After 6 Hours

### Best Case Scenario ✅
```
- 30+ trades executed
- Profit: $5-20+ (depending on market)
- Position count: 3-5 (healthy)
- Kill-switch: DISABLED (actively trading)
- NAV: $90-120+ (up from $83)
- No errors in log
```

### Good Case Scenario ✅
```
- 15-25 trades executed
- Profit: $2-10 (modest gains)
- Position count: 5-10 (recovering)
- Kill-switch: DISABLED by hour 2
- NAV: $85-100 (slight improvement)
- <5 errors (recoverable)
```

### Acceptable Case ✅
```
- 5-10 trades executed
- Profit: Break-even to +$5
- Position count: <20 (consolidating)
- Kill-switch: Still enabled (healing in progress)
- NAV: $83-90 (stable)
- <10 errors (manageable)
```

### Problem Case ❌
```
- <5 trades executed
- Loss: Any negative NAV change
- Position count: Still >25 (not healing)
- Kill-switch: ACTIVE after 2 hours
- NAV: Decreased from $83
- >20 errors
→ **ACTION NEEDED:** Bot needs manual intervention
```

---

## 6-Hour Timeline Reference

| Time | Expected Phase | Key Indicators | Action If Stuck |
|------|----------------|-----------------|-----------------|
| 0-10 min | Auto-recovery | Positions liquidating | Check RECOVERY mode |
| 10-20 min | Dust clearing | 10+ positions closed | Check LiquidationAgent |
| 20-30 min | Kill-switch | Should disable soon | Manual heal if needed |
| 30 min - 2h | Trading resume | First trades execute | Lower thresholds manually |
| 2-4h | Active trading | 2-5 trades/cycle | Monitor for issues |
| 4-6h | Compound gains | Steady trading rhythm | Just monitor |

---

## Auto-Restart Script (If Bot Crashes)

Save this as `monitor_and_restart.sh`:

```bash
#!/bin/bash

while true; do
  if ! pgrep -f "MASTER" > /dev/null; then
    echo "❌ Bot crashed at $(date). Restarting..."
    cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
    nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > logs/octivault_master_orchestrator.log 2>&1 &
    echo "✅ Bot restarted"
    sleep 60
  fi
  sleep 30
done
```

Run it in a separate terminal:
```bash
chmod +x monitor_and_restart.sh
./monitor_and_restart.sh
```

---

## Final Notes

### The System Should:

1. ✅ **Detect dust trap** → Auto-recovery within 30 seconds of startup
2. ✅ **Liquidate dust** → 35 positions → <5 within 15 minutes
3. ✅ **Disable kill-switch** → Within 30 minutes of dust clearing
4. ✅ **Resume trading** → 3-5 trades per 30-second cycle
5. ✅ **Generate profit** → Expecting $5-20+ over 6 hours

### You Don't Need To:

- ❌ Manually liquidate positions (auto-recovery does this)
- ❌ Kill and restart (only if it crashes)
- ❌ Monitor constantly (checks every 30 min are fine)
- ❌ Intervene unless major issues occur

### Success Criteria:

After 6 hours, you should see:
- ✅ Multiple completed trades
- ✅ NAV increased from $83
- ✅ <20 dust positions (or none)
- ✅ Kill-switch disabled
- ✅ Clean logs (no critical errors)

---

**Let it run! System is stable and healing. Check back periodically and enjoy the gains! 🚀**

**Log files for analysis:** `logs/octivault_master_orchestrator.log`
**Emergency file:** `/tmp/run_final_report.txt` (after 6 hours)
