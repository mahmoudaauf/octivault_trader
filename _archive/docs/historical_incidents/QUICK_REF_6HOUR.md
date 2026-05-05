# ⏱️ QUICK REFERENCE - 6 Hour Run

**Start:** NOW
**End:** +6 hours
**Bot PID:** 96511 ✅

---

## 📋 30-Second Status Check (Do Every 30 Min)

```bash
# Copy & paste this:
echo "=== CHECK $(date) ===" && \
pgrep -f "MASTER" && echo "✅ Bot OK" || echo "❌ CRASHED" && \
echo "Trades: $(grep TRADE_EXECUTED logs/octivault_master_orchestrator.log | wc -l)" && \
echo "Liquidated: $(grep 'POSITION.*CLOSED' logs/octivault_master_orchestrator.log | wc -l)" && \
tail -1 logs/octivault_master_orchestrator.log
```

---

## 🎯 What Should Happen

| Time | What | How to Check |
|------|------|-------------|
| 0-10 min | Auto-heal starts | `grep "RECOVERY" logs/...log` |
| 10-20 min | Dust clearing | `grep "CLOSED" logs/...log` (should grow) |
| 20-30 min | Kill-switch off | `grep "disabled" logs/...log` |
| 30m+ | Trading active | `grep "EXECUTED" logs/...log` |

---

## 🚨 If Something Goes Wrong

### Bot Crashed?
```bash
pkill -9 -f "MASTER"
sleep 3
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > logs/octivault_master_orchestrator.log 2>&1 &
```

### Stuck in Dust Trap After 30 Min?
```bash
# Check what happened
grep -i "recovery\|liquidat" logs/octivault_master_orchestrator.log | tail -10

# Force manual healing
bash scripts/emergency_liquidate.sh  # If available
```

### Still No Trades After 1 Hour?
- Don't panic - system auto-reduces thresholds
- Wait until kill-switch shows as "DISABLED"
- Check: `grep "kill.switch\|compound" logs/octivault_master_orchestrator.log`

---

## ✅ Success Metrics (After 6 Hours)

| Metric | Target | You're Good If |
|--------|--------|---|
| Trades Executed | 15+ | ✅ See 15+ "TRADE_EXECUTED" lines |
| Positions Liquidated | 20+ | ✅ See 20+ "POSITION CLOSED" lines |
| NAV Change | +$5-20 | ✅ NAV > $88 |
| Position Count | <5 | ✅ From 35 → <5 |
| Errors | <10 | ✅ Less than 10 error lines |

---

## 📊 Real-Time Monitoring (Optional)

Watch logs in real-time:
```bash
tail -f logs/octivault_master_orchestrator.log | grep -E "TRADE|CLOSED|disabled|RECOVERY"
```

---

## 🔄 Full Hourly Report

After each hour, run:
```bash
echo "Hour $(date +%H): Trades=$(grep TRADE_EXECUTED logs/octivault_master_orchestrator.log | wc -l) Liquidated=$(grep 'CLOSED' logs/octivault_master_orchestrator.log | wc -l) Status=$(pgrep -f MASTER && echo 'OK' || echo 'DOWN')"
```

---

## 🛑 STOP If...

- ❌ Bot crashes more than 3 times
- ❌ NAV goes negative (loss >$10)
- ❌ 100+ error lines in log
- ❌ Bot stuck for >1 hour with 0 activity

---

**Default:** Let it run! System is stable. Just check status occasionally.
**Full Plan:** See `6HOUR_MONITORING_PLAN.md`
