# ⚡ LIVE TRADING WITH BALANCE MONITORING - QUICK START

## 🚀 START LIVE TRADING NOW

### Simplest (Recommended):
```bash
bash start_live_trading.sh
```

### Just Trading:
```bash
python3 LIVE_TRADING_WITH_BALANCE_MONITOR.py
```

### Just Dashboard:
```bash
python3 balance_dashboard.py
```

---

## 💰 WHAT YOU GET

✅ **Real-Time Balance Tracking**
- Current balance updates every cycle
- Peak balance monitoring
- Lowest balance tracking
- Total profit/loss calculation
- Performance status (📈📉➡️)

✅ **Live Dashboard**
- Refreshes every 5 seconds
- Shows all performance metrics
- System status display
- Terminal UI

✅ **Persistent Memory**
- Balance data saved to disk
- Survives system restart
- Checkpoints every 5 minutes
- Complete history preserved

✅ **Auto-Recovery**
- Automatic restart detection
- Balance checkpoint recovery
- Resume from checkpoint
- Zero data loss

---

## 📊 MONITORING

### Dashboard Display:
```
Current Balance: $11,850.00
📈 GAINING - Change: +$1,850.00 (+18.50%)

Initial:   $10,000.00
Peak:      $12,500.00 (+25.00%)
Low:       $ 9,200.00 (-8.00%)
Current:   $11,850.00 (+18.50%)

Updates: 250 | Last: 2026-04-26T15:45:30
Phase: live_trading | Task: cycle_250 | Recovery: ✅
```

### Multi-Terminal Monitoring:
```bash
# Terminal 1: Trading system
python3 LIVE_TRADING_WITH_BALANCE_MONITOR.py

# Terminal 2: Balance dashboard
python3 balance_dashboard.py

# Terminal 3: Watch state files
watch -n 5 'ls -lh state/'

# Terminal 4: Check history
tail -f state/operational_state.json | python3 -m json.tool
```

---

## 📁 FILES DEPLOYED

| File | Size | Purpose |
|------|------|---------|
| LIVE_TRADING_WITH_BALANCE_MONITOR.py | 7.8 KB | Trading + balance |
| balance_dashboard.py | 3.3 KB | Dashboard |
| start_live_trading.sh | 3.3 KB | Startup script |
| LIVE_TRADING_BALANCE_MONITOR_GUIDE.md | 11 KB | Full guide |

---

## ✅ STATUS

- ✅ Phase 4: COMPLETE (30-min test passed, state recovery verified)
- ✅ Phase 5: READY (live trading with balance monitoring enabled)
- ✅ Balance Tracking: ACTIVE
- ✅ Auto-Recovery: READY
- ✅ All Systems: OPERATIONAL

---

## 🎯 KEY COMMANDS

```bash
# Start complete live trading
bash start_live_trading.sh

# Just trading
python3 LIVE_TRADING_WITH_BALANCE_MONITOR.py

# Just dashboard
python3 balance_dashboard.py

# Single dashboard snapshot
python3 balance_dashboard.py --once

# Verify deployment
python3 verify_deployment.py

# Check balance history
python3 -c "from system_state_manager import SystemStateManager; import json; mgr = SystemStateManager(); ctx = mgr.get_system_context(); print(json.dumps(ctx.get('system_status', {}).get('balance_tracking', {}), indent=2))"
```

---

## 💡 TIPS

1. **First Time?**
   - Run: `bash start_live_trading.sh`
   - Watch dashboard update
   - See balance change in real-time

2. **Monitor While Running?**
   - Open new terminal
   - Run: `python3 balance_dashboard.py`
   - Dashboard shows current performance

3. **Quick Check?**
   - Run: `python3 balance_dashboard.py --once`
   - See balance snapshot
   - Exit automatically

4. **System Crashes?**
   - Just restart: `python3 LIVE_TRADING_WITH_BALANCE_MONITOR.py`
   - System auto-detects restart
   - Loads previous balance checkpoint
   - Continues from where it left off

---

## 📈 FEATURES

- **Real-Time Updates**: Balance tracked every cycle
- **Performance Metrics**: Initial, peak, low, current balances
- **Status Indicators**: 📈 GAINING / 📉 LOSING / ➡️ STABLE
- **Persistent Storage**: All data saved to disk
- **Auto-Recovery**: Zero manual intervention needed
- **Live Dashboard**: Terminal-based UI refreshing every 5 seconds
- **Checkpoint Snapshots**: Balance saved every 5 minutes

---

**Status: READY FOR LIVE TRADING** ✅

**Start Now:**
```bash
bash start_live_trading.sh
```
