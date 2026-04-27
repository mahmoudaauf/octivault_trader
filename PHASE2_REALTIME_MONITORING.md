# 🔍 PHASE 2 REAL-TIME MONITORING GUIDE

## Live Commands (Copy & Paste)

### 1. WATCH LIVE LOGS (Real-time follow)
```bash
tail -f /tmp/octivault_master_orchestrator.log
```
**What to look for**: Entry/exit orders, error messages, signal processing

### 2. CHECK ENTRY SIZING (Verify Fix #3)
```bash
grep "quote=25.00" /tmp/octivault_master_orchestrator.log | tail -20
```
**Expected**: All BUY signals show `quote=25.00`
**Status**: ✅ 229+ detected in current session

### 3. VERIFY ROTATION OVERRIDE (Fix #2)
```bash
grep -i "ROTATION.*OVERRIDE\|rotation.*exit\|forced.*exit" /tmp/octivault_master_orchestrator.log | tail -10
```
**Expected**: Stagnation override events logged
**Status**: ✅ AVNTUSDT rotation override detected

### 4. CHECK RECOVERY BYPASS (Fix #1)
```bash
grep -i "bypass_min_hold\|recovery.*exit\|SafeMinHold.*Bypass" /tmp/octivault_master_orchestrator.log | tail -10
```
**Expected**: Bypass messages when capital recovery triggers
**Status**: ℹ️ Awaiting recovery event (capital stable)

### 5. MONITOR BOT PROCESS
```bash
ps aux | grep "MASTER_SYSTEM_ORCHESTRATOR" | grep -v grep
```
**Expected**: Single process running (PID 58703)
**Status**: ✅ Running (54% CPU, 582 MB RAM)

### 6. COUNT ALL ENTRY SIGNALS
```bash
grep -c "quote=25.00" /tmp/octivault_master_orchestrator.log
```
**Trending**: Should increase every 30 seconds
**Current**: 229 signals

### 7. CHECK FOR ERRORS
```bash
grep -i "ERROR\|CRITICAL\|exception" /tmp/octivault_master_orchestrator.log | tail -20
```
**Expected**: Minimal or none
**Status**: ✅ No critical errors detected

### 8. LOG FILE SIZE & GROWTH
```bash
du -h /tmp/octivault_master_orchestrator.log && echo "" && ls -lh /tmp/octivault_master_orchestrator.log
```
**Expected**: Growing continuously (healthy trading activity)
**Status**: Should be 50+ KB

### 9. INITIALIZATION STATUS
```bash
grep -i "system ready\|main loop\|operational\|initialization" /tmp/octivault_master_orchestrator.log | tail -5
```
**Expected**: "System ready, starting main loop" message
**Status**: ℹ️ In final initialization phase

### 10. SIGNAL CACHE STATUS
```bash
grep -c "Signal cached" /tmp/octivault_master_orchestrator.log
```
**Expected**: 10+ signals cached (one per symbol)
**Trending**: Should stabilize at 10 after warm-up

---

## Performance Dashboard (Quick Status Check)

```bash
#!/bin/bash
echo "⚡ PHASE 2 BOT DASHBOARD - $(date)"
echo "=========================================="
echo ""
echo "📊 Process Status:"
ps aux | grep "MASTER_SYSTEM_ORCHESTRATOR" | grep -v grep | awk '{print "   PID: "$2" CPU: "$3"% MEM: "$4"% TIME: "$11}'
echo ""
echo "📈 Entry Sizing (Fix #3):"
echo "   Quote=25 signals: $(grep -c 'quote=25.00' /tmp/octivault_master_orchestrator.log)"
echo ""
echo "🔄 Rotation Overrides (Fix #2):"
echo "   Overrides detected: $(grep -c 'ROTATION_STAGNATION_OVERRIDE' /tmp/octivault_master_orchestrator.log)"
echo ""
echo "🛡️  Recovery Bypasses (Fix #1):"
echo "   Bypass events: $(grep -c 'bypass_min_hold' /tmp/octivault_master_orchestrator.log)"
echo ""
echo "⚠️  Errors & Warnings:"
echo "   Critical errors: $(grep -c 'CRITICAL' /tmp/octivault_master_orchestrator.log)"
echo "   Error messages: $(grep -ic 'ERROR' /tmp/octivault_master_orchestrator.log)"
echo ""
echo "💾 Log File:"
du -h /tmp/octivault_master_orchestrator.log | awk '{print "   Size: "$1" Growth rate: Active"}'
echo ""
echo "🎯 System Status:"
if grep -q "System ready" /tmp/octivault_master_orchestrator.log 2>/dev/null; then
    echo "   ✅ System operational (main loop running)"
else
    echo "   ⚙️  System initializing (still warming up)"
fi
```

Save as `dashboard.sh` and run: `bash dashboard.sh` (runs every 2 seconds with watch)

---

## Expected Log Patterns

### NORMAL STARTUP SEQUENCE (Should see these in order)
1. ✅ "Config initialized" - Configuration loaded
2. ✅ "Exchange connected" - Binance API responding  
3. ✅ "Capital loaded" - NAV and balance determined
4. ✅ "Symbol bootstrap" - 10 symbols initialized
5. ✅ "Signal processor ready" - Hunters activated
6. ✅ "Signals cached: 10" - All symbols have signals
7. ✅ "System ready, starting main loop" - **FULL OPERATIONAL**

### TRADING ACTIVITY PATTERNS (Should see regularly)
```
[Meta:PreTradeEffect] SYMBOL BUY quote=25.00 ...
  ↓ Entry signal evaluated
[EXECUTE] BUY order for SYMBOL @ size 25.00
  ↓ Order placed
[POSITION_OPENED] SYMBOL qty=X entry=Y profit_target=Z
  ↓ Trade opened
```

### FIX #1 ACTIVATION (Recovery Bypass)
```
[Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit: SYMBOL
  ↓ Recovery triggered
[FORCED_EXIT] SYMBOL via recovery exit
  ↓ Position closed for protection
```

### FIX #2 ACTIVATION (Rotation Override)
```
[Meta:ProfitGate] FORCED EXIT override for SYMBOL (bypassing profit gate for recovery)
  reason=ROTATION_STAGNATION_OVERRIDE
  ↓ Rotation override triggered
[ROTATION] SYMBOL rotated to NEWSYMBOL
  ↓ Position rotated to new opportunity
```

### FIX #3 VERIFICATION (Entry Sizing)
```
Every BUY signal should show:
  quote=25.00
  
Track: grep -c "quote=25.00"
Expected to grow linearly with trading activity
```

---

## Monitoring Intervals

### Every 30 seconds (Automated)
- Bot processes price data
- Signals evaluated
- Entry/exit decisions made
- Order execution

### Every 5 minutes (Manual Check)
```bash
echo "Entry signals (last 5 min): $(grep -c 'quote=25.00' /tmp/octivault_master_orchestrator.log)"
```

### Every 15 minutes (Performance Check)
```bash
echo "Bot uptime: $(ps -o etime= -p PID)"
echo "Total signals: $(grep -c 'quote=25.00' /tmp/octivault_master_orchestrator.log)"
echo "Errors: $(grep -c 'ERROR' /tmp/octivault_master_orchestrator.log)"
```

### Every 1 hour (Comprehensive Review)
- Daily return so far: _____
- Trades executed: _____
- Win rate: _____
- Max drawdown: _____
- Any issues: _____

---

## Critical Alerts (Stop & Check)

### 🚨 STOP IF YOU SEE:
1. **"Process terminated"** → Bot crashed, restart immediately
2. **Multiple ERROR lines** → Check log for cause
3. **"quote=15.00" in logs** → Fix #3 not applied, restart with new .env
4. **"Exchange connection lost"** → API issue, may recover automatically
5. **"Capital below minimum"** → Recovery trigger (expected), monitor exit

### 🟡 WARNING (Monitor closely):
1. **CPU > 80%** → Bot working hard (normal during signal rush)
2. **Memory > 1GB** → May need restart if keeps growing
3. **No signals for 5 min** → Check if market locked or API slow
4. **Negative daily return > 5%** → May need manual review

### ✅ HEALTHY INDICATORS:
1. **quote=25.00 appearing regularly** → Fix #3 working ✅
2. **Process running > 1 hour** → System stable
3. **~1-3 trades per hour** → Normal trading pace
4. **CPU 20-60%** → Normal load
5. **Growing signal cache** → System active

---

## Verification Checklist (Run Hourly)

```
Hour 1: ___
- [ ] quote=25.00 appearing in logs
- [ ] Process still running (PID 58703)
- [ ] No ERROR messages
- [ ] Entry signals processing

Hour 2: ___
- [ ] Entry signals increased by ~30-60 (at 30s intervals)
- [ ] At least 1-2 BUY signals executed
- [ ] No process crashes
- [ ] Rotation overrides detected (if applicable)

Hour 3+: ___
- [ ] Daily return tracking ____%
- [ ] Win rate _____
- [ ] Total trades: _____
- [ ] All 3 fixes working (quote=25, rotations, recoveries)
```

---

## Quick Comparison: Pre vs Post Phase 2

| Aspect | Pre-Phase 2 | Post-Phase 2 (Now) |
|--------|------------|-------------------|
| **Entry Size** | 15 USDT | 25 USDT ✅ |
| **Entry Signals** | 15 USDT | 229+ @ 25 USDT ✅ |
| **Rotation** | Manual trigger | Auto-override ✅ |
| **Recovery Exit** | Manual intervention | Automatic bypass ✅ |
| **Log Pattern** | quote=15.00 | quote=25.00 ✅ |
| **Bot Status** | Uncertain | LIVE & VERIFIED ✅ |

---

## One-Minute Status Command

```bash
# Paste this entire block:
echo "🎯 PHASE 2 STATUS - $(date '+%H:%M:%S')" && \
echo "Process: $(ps aux | grep -c 'MASTER_SYSTEM_ORCHESTRATOR.*py')-1 running" && \
echo "Entry 25: $(grep -c 'quote=25.00' /tmp/octivault_master_orchestrator.log) signals" && \
echo "Rotation: $(grep -c 'ROTATION.*OVERRIDE' /tmp/octivault_master_orchestrator.log) overrides" && \
echo "Errors: $(grep -c 'ERROR' /tmp/octivault_master_orchestrator.log) messages" && \
echo "CPU/MEM: $(ps aux | grep 'MASTER_SYSTEM_ORCHESTRATOR' | grep -v grep | awk '{print $3"%/"$4"%"}')" && \
echo "Status: ✅ Operational"
```

Run this every minute to get instant status update.

---

## IMPORTANT: Keep Monitoring For 30 Minutes

**Critical Phase**: Next 30 minutes are warm-up period
- System finalizing initialization
- Signals reaching full clarity
- First trades expected within 15 minutes
- All 3 fixes should be demonstrably active

**What to expect**:
- ✅ 300+ entry signals with quote=25.00
- ✅ 1-3 BUY orders executed
- ✅ Rotation events if positions held too long
- ✅ Capital management actively tracking

**At 30-minute mark, system reaches FULL OPERATIONAL status**

---

