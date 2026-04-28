# 🟢 LIVE TRADING SESSION - QUICK START GUIDE

**Session Status**: 🟢 LIVE (Started 2026-04-24 10:14:00 UTC)  
**Phase 2 Status**: ✅ ACTIVE  
**Process ID**: 65413  
**Approval**: ✅ APPROVE_LIVE_TRADING=YES  

---

## ⚡ QUICK MONITORING (Copy & Paste Ready)

### Terminal 1: Real-Time Dashboard
```bash
python3 "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/LIVE_PHASE2_MONITOR.py"
```
- **What it shows**: Phase 2 indicators in real-time
- **Updates**: Every 5 seconds
- **Stop**: Ctrl+C

### Terminal 2: All Logs (Verbose)
```bash
tail -f "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading.log"
```
- **What it shows**: Complete trading log
- **Updates**: Real-time as they happen
- **Stop**: Ctrl+C

### Terminal 3: Recovery Bypasses Only
```bash
tail -f "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading.log" | grep "Bypassing min-hold"
```
- **What it shows**: Recovery exit min-hold bypass events
- **Expected**: 1-2 per hour
- **Critical**: ✅ Phase 2 Fix #1

### Terminal 4: Forced Rotations Only
```bash
tail -f "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/trading.log" | grep "MICRO restriction OVERRIDDEN"
```
- **What it shows**: Forced rotation override events
- **Expected**: 0-1 per hour
- **Critical**: ✅ Phase 2 Fix #2

---

## 📊 LIVE METRICS (Update Every 30 Seconds)

### Terminal: Live Counts
```bash
watch -n 30 'echo "=== PHASE 2 LIVE METRICS ===" && echo "" && echo "Recovery Bypasses: $(grep -c "Bypassing min-hold" trading.log)" && echo "Forced Rotations: $(grep -c "MICRO restriction OVERRIDDEN" trading.log)" && echo "Entries Executed: $(grep -c "\[EXEC_DECISION\].*BUY" trading.log)" && echo "Exits Executed: $(grep -c "\[EXEC_DECISION\].*SELL" trading.log)" && echo "Entry Size Valid: $(grep -c "ENTRY_SIZE_ENFORCEMENT.*25.00" trading.log)" && echo "Errors: $(grep -c "\[ERROR\]\|\[CRITICAL\]" trading.log)"'
```

---

## 🎯 WHAT TO EXPECT

### In First 5 Minutes
- ✅ Config loading
- ✅ Exchange connection
- ✅ Portfolio loading (16 positions)
- ✅ Signal generation starting

### In First Hour
- 🔍 Watch for Phase 2 indicators appearing
- 📈 1-2 recovery bypasses expected
- 💰 Entry sizing at $25+ per trade
- ⏳ May see 0-1 forced rotations

### In First 24 Hours
- 📊 P&L: +2-3% expected (based on 6-hour validation)
- 🔄 Recovery bypasses: 24-48 total
- 🔄 Forced rotations: 0-24 total
- 💯 Entry alignment: 120-240+ total

---

## ⚠️ ALERT CHECKLIST

### 🔴 STOP IF YOU SEE:
- `[ERROR]` or `[CRITICAL]` in logs
- `Exchange connection lost`
- `API rate limit exceeded`
- Multiple position mismatches
- Capital dropping below $5 USDT

### 🟡 WARNING - MONITOR:
- No Phase 2 indicators for 15+ minutes
- Increasing [WARNING] messages
- Entry sizing below $25
- Unused capital not deployed

### 🟢 GOOD SIGNS:
- Phase 2 indicators 1-2 per hour
- Entry sizing consistently $25+
- Capital fluctuating normally
- `System health: HEALTHY` logged

---

## 🛑 EMERGENCY STOP

### If something looks wrong:
```bash
# Kill the trading process
pkill -f "MASTER_SYSTEM_ORCHESTRATOR"

# Verify it stopped:
ps aux | grep "MASTER_SYSTEM" | grep -v grep
```

---

## 📈 PHASE 2 FIXES VERIFICATION

During live session, verify these appear in logs:

### Fix #1: Recovery Exit Min-Hold Bypass
```log
[Meta:SafeMinHold] Bypassing min-hold check for SYMBOL
```
✅ **Status**: Should start appearing within 30 minutes if stagnation detected

### Fix #2: Forced Rotation MICRO Override  
```log
[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN
```
✅ **Status**: Should appear if capital velocity requires forced exit

### Fix #3: Entry Sizing Alignment (25 USDT)
```log
[Meta:Layer1] ENTRY_SIZE_ENFORCEMENT: significant_position_usdt=$25.00
```
✅ **Status**: Should be visible in every trading loop

---

## 📊 KEY FILES

| File | Purpose |
|------|---------|
| `trading.log` | Main trading log (live) |
| `LIVE_TRADING_DASHBOARD.md` | Full dashboard documentation |
| `LIVE_PHASE2_MONITOR.py` | Real-time Phase 2 indicator monitor |
| `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | Main trading engine (running) |

---

## 🔍 ADVANCED DEBUGGING

### Count Phase 2 indicators since session start:
```bash
echo "Recovery Bypasses: $(grep -c 'Bypassing min-hold' trading.log)"
echo "Forced Rotations: $(grep -c 'MICRO restriction OVERRIDDEN' trading.log)"
echo "Entry Sizes: $(grep -c 'ENTRY_SIZE_ENFORCEMENT.*25.00' trading.log)"
```

### Get last 20 Phase 2 events:
```bash
(grep 'Bypassing min-hold' trading.log; grep 'MICRO restriction OVERRIDDEN' trading.log; grep 'ENTRY_SIZE_ENFORCEMENT' trading.log) | tail -20
```

### Check for any errors:
```bash
grep -E "\[ERROR\]|\[CRITICAL\]" trading.log
```

### Get session summary (every 5 minutes):
```bash
watch -n 300 'echo "Session Time: $(date -u +%H:%M:%S)" && echo "Process: $(ps aux | grep MASTER_SYSTEM | grep -v grep | wc -l) running" && echo "Log Size: $(du -h trading.log | cut -f1)" && echo "Phase 2 Events: $(( $(grep -c "Bypassing min-hold" trading.log) + $(grep -c "MICRO restriction OVERRIDDEN" trading.log) ))"'
```

---

## 📋 MONITORING CHECKLIST

- [ ] Trading process running (`ps aux | grep MASTER_SYSTEM`)
- [ ] trading.log exists and is updating
- [ ] No [ERROR] or [CRITICAL] messages
- [ ] Entry sizing shows $25.00 enforcement
- [ ] Phase 2 indicators appearing (or expected to appear soon)
- [ ] Capital reserve showing > $5 USDT
- [ ] Portfolio has 15+ positions loaded

---

## 🎯 SUCCESS CRITERIA (24 HOUR SESSION)

| Metric | Target | Status |
|--------|--------|--------|
| Recovery Bypasses | 24-48 | Monitoring |
| Forced Rotations | 0-24 | Monitoring |
| Entry Alignments | 120-240+ | Monitoring |
| Daily P&L | +2-3% | Monitoring |
| System Uptime | 99%+ | Monitoring |
| Errors | 0 | ✅ So far |

---

## 💡 TIPS

1. **Best Monitor**: Use Terminal 1 (`LIVE_PHASE2_MONITOR.py`) - shows everything clearly
2. **Tail Command**: Use Terminal 2 for full context and debugging
3. **Phase 2 Specific**: Use Terminal 3 & 4 to watch individual fixes
4. **Metrics**: Run the watch command in Terminal 5 for continuous counters
5. **Session Duration**: 24 hours (until 2026-04-25 10:14:00 UTC)

---

## 🚀 YOU ARE GO FOR LIVE!

- ✅ Phase 2 code verified (16/16 checks)
- ✅ 6-hour validation passed (9/9 checkpoints)
- ✅ Production deployment approved
- ✅ Live trading ACTIVE

**Watch the logs, the Phase 2 fixes will speak for themselves!**

---

Created: 2026-04-24 10:15:00 UTC  
Live Trading Session: ACTIVE 🟢
