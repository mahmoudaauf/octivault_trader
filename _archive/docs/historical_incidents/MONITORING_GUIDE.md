# 🎯 ACTIVE CAPITAL GROWTH MONITORING SYSTEM

Complete guide to monitoring, tracking capital growth, and automatically fixing issues.

## 📋 Overview

The monitoring system provides:

1. **Real-Time Capital Tracking** - Live NAV, free capital, and invested positions
2. **Active Health Checks** - Continuous assessment of balance sync, execution, and positions
3. **Automatic Issue Detection** - Identifies problems before they impact trading
4. **Auto-Fix Engine** - Applies fixes automatically when issues are detected
5. **Live Dashboard** - Beautiful real-time visualization of capital growth

## 🚀 Quick Start

### Option 1: Integrated Launch (Recommended)

```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Run everything with one command
./start_trading_with_monitoring.sh --duration 6 --monitor-interval 10
```

This automatically:
- ✅ Clears state files for fresh start
- ✅ Starts trading orchestrator
- ✅ Starts active monitoring with auto-fix engine
- ✅ Opens real-time dashboard
- ✅ Runs for 6 hours
- ✅ Cleans up on exit

### Option 2: Manual Components

**Terminal 1 - Start Trading:**
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json
env TRADING_DURATION_HOURS=6 APPROVE_LIVE_TRADING=YES python 🎯_MASTER_SYSTEM_ORCHESTRATOR.py
```

**Terminal 2 - Start Active Monitor:**
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python -m monitoring.active_capital_monitor --duration 360 --interval 10
```

**Terminal 3 - Start Dashboard:**
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python monitoring/real_time_dashboard.py --refresh 30
```

## 📊 Component Details

### 1. Active Capital Monitor (`monitoring/active_capital_monitor.py`)

Continuously monitors system health and detects issues.

**Features:**
- Capital growth tracking
- Health scoring (0-100) for:
  - Balance sync reliability
  - Position alignment
  - Execution performance
- Issue detection
- Automatic fix application

**Running:**
```bash
python -m monitoring.active_capital_monitor \
    --duration 360 \           # Monitor for 6 hours
    --interval 10              # Check every 10 seconds
```

**What It Tracks:**
```
💰 CAPITAL STATUS
   NAV:              $101.70
   Free USDT:        $97.86
   Invested:         $3.84
   Positions:        1
   Loop Count:       42

📈 GROWTH METRICS
   Growth Rate (Ann):   1235.50%
   Volatility:          2.35%
   Drawdown:            0.15%
   Elapsed Time:        0h 5m

🏥 SYSTEM HEALTH
   BALANCE_SYNC       🟢 95.0/100
   POSITIONS          🟢 90.0/100
   EXECUTION          🟢 88.0/100

⚠️  ACTIVE ALERTS
   STALE_BALANCE      [MEDIUM] ⏳ PENDING
      Balance cache appears stale or sync failing
```

### 2. Real-Time Dashboard (`monitoring/real_time_dashboard.py`)

Beautiful real-time visualization of capital growth.

**Features:**
- Live NAV with change indicator
- Capital breakdown
- Returns analysis
- Sparkline history chart
- Health indicator lights
- Updates every 30 seconds

**Running:**
```bash
python monitoring/real_time_dashboard.py --refresh 30
```

**Display:**
```
════════════════════════════════════════════════════════════════════════════════════════════════════
📊 REAL-TIME CAPITAL GROWTH DASHBOARD
════════════════════════════════════════════════════════════════════════════════════════════════════

⏰ 2026-05-02 14:35:22 | ⏱️  Elapsed: 0h 5m 30s | 📍 Loop: 42

💰 CAPITAL STATUS
─────────────────────────────────────────────────────────────────────────────────────────────────
  Current NAV:        $101.70  [  +0.55]
  Free Capital:       $97.86
  Invested:           $3.84
  Positions:                 1

📈 RETURNS ANALYSIS
─────────────────────────────────────────────────────────────────────────────────────────────────
  Total Return:           0.55%
  Hourly Return (Ann):    6.60%
  Max Drawdown:          -0.12%
  History:            ▁▂▄▅▆▆▇█▆▅▄▃▂▂▂▁▁▁▂▂

🏥 SYSTEM HEALTH
─────────────────────────────────────────────────────────────────────────────────────────────────
  Balance Sync:       🟢
  Execution:          🟢
  Positions:          🟢

════════════════════════════════════════════════════════════════════════════════════════════════════
```

### 3. Integrated Launcher (`launch_with_monitor.py`)

Coordinates orchestrator and monitor startup.

**Features:**
- Starts orchestrator with trading environment
- Runs active monitor concurrently
- Handles graceful shutdown
- Manages state cleanup

**Running:**
```bash
python launch_with_monitor.py \
    --duration 6 \             # 6 hours of trading
    --monitor-interval 10      # Check every 10 seconds
```

## 🔍 Issue Detection & Auto-Fix

### Detected Issues

| Issue Type | Detection | Severity | Auto-Fix |
|-----------|-----------|----------|----------|
| **Stale Balance Cache** | Balance sync score < 60 | HIGH | Force fresh sync |
| **Position Misalignment** | Wallet Guard filters | MEDIUM | NAV rebuild |
| **Capital Stagnation** | No growth in 15min | MEDIUM | Reset throttles |
| **Execution Slowdown** | Exec health score < 70 | MEDIUM | Check constraints |
| **Sync Failure** | Repeated sync errors | HIGH | Retry with backoff |
| **Liquidity Warning** | Free capital too low | MEDIUM | Alert operator |

### Auto-Fix Cooldown

- Same fix type can't run more than once per 5 minutes
- Prevents rapid oscillation
- Allows system to stabilize

### Fix Examples

**Stale Balance Cache:**
```python
# Auto-fix calls:
await shared_state.sync_authoritative_balance(force=True)
# This bypasses TTL throttle, fetches fresh from Binance
```

**Position Misalignment:**
```python
# Auto-fix rebuilds NAV:
await rebuild_nav_from_state(source="auto_fix")
# Recalibrates wallet guard with current data
```

## 📈 Monitoring Metrics

### Capital Metrics
- **NAV (Net Asset Value)** - Total portfolio value
- **Free USDT** - Available capital for new trades
- **Invested** - Value locked in positions
- **Positions Count** - Number of active trades

### Performance Metrics
- **Total Return %** - Since session start
- **Hourly Annualized Return** - Extrapolated hourly rate
- **Maximum Drawdown** - Peak-to-trough decline
- **Volatility** - Standard deviation of returns

### Health Scores (0-100)
- **Balance Sync** - API call success rate
- **Position Alignment** - Wallet vs position agreement
- **Execution** - Order success rate

## 🎯 Typical Monitoring Session

### Hour 0 (Session Start)
```
✅ Fresh state files cleared
✅ Orchestrator started with force=True balance sync
✅ Monitor begins tracking from checkpoint 0
✅ Dashboard initializes with clean data
🟢 All health indicators GREEN
```

### Hour 1-3 (Steady State)
```
📈 Capital growing steadily (+2-3% per hour)
🟢 Health scores 85-95/100
⚡ Trade execution rate: 2-3 per minute
💰 Free capital increasing with profitability
```

### Hour 3-6 (Optimization)
```
📊 Growth rate may stabilize or accelerate
🔄 TTL throttle causes periodic cache reuse (normal)
🟡 Occasional warnings can trigger auto-fixes
✅ Auto-fixes applied automatically
```

## 🚨 Troubleshooting

### Dashboard Shows No Data
**Cause:** Orchestrator hasn't logged metrics yet
**Fix:** Wait 30 seconds for first cycle

### Monitor Shows Low Health Scores
**Cause:** Too many API errors or failed syncs
**Fix:** Monitor will auto-apply fixes; check Binance API status

### Capital Not Growing
**Cause:** Could be capital floor constraints or market conditions
**Fix:** Monitor logs, check if orders are being rejected

### Auto-Fix Not Applied
**Cause:** Same fix type ran recently (5min cooldown)
**Fix:** Wait or manually restart to force fresh sync

## 🔧 Configuration

### Adjust Monitoring Frequency
```bash
# More frequent checks (every 5 seconds)
python -m monitoring.active_capital_monitor --interval 5

# Less frequent (every 30 seconds)
python -m monitoring.active_capital_monitor --interval 30
```

### Adjust Dashboard Refresh
```bash
# Fast updates (every 10 seconds)
python monitoring/real_time_dashboard.py --refresh 10

# Slower updates (every 60 seconds)
python monitoring/real_time_dashboard.py --refresh 60
```

### Change Log Path
```bash
# Monitor custom log
python -m monitoring.active_capital_monitor \
    --log-path logs/custom_run.log
```

## 📊 Output Files

The system generates several output files for analysis:

```
monitoring/
├── dashboard_metrics.json      # Latest metrics snapshot
├── active_capital_monitor.py   # Main monitor script
├── real_time_dashboard.py      # Dashboard script
└── [other support files]
```

**dashboard_metrics.json** format:
```json
{
  "timestamp": 1714743322.5,
  "nav": 101.70,
  "free": 97.86,
  "invested": 3.84,
  "total_return_pct": 0.55,
  "hourly_return_pct": 6.60,
  "max_drawdown_pct": -0.12,
  "loop": 42,
  "health": {
    "balance_sync": "🟢",
    "execution": "🟢",
    "positions": "🟢"
  }
}
```

## 🎯 Best Practices

1. **Always clear state before new session**
   ```bash
   rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json
   ```

2. **Monitor in background, don't interrupt**
   - Use `nohup` or screen/tmux for long sessions
   - Let auto-fixes run automatically
   - Don't manually restart unless critical issue

3. **Check metrics.json periodically**
   - Provides machine-readable status
   - Can be integrated with other tools
   - Updates every 30 seconds

4. **Watch for WARNING status**
   - 🟡 Means potential issue detected
   - Monitor will auto-fix if possible
   - Check logs if problem persists

5. **Trust the auto-fix engine**
   - Designed to handle common issues
   - Has 5-minute cooldown to prevent oscillation
   - Safe to apply (read-only operations + state resets)

## 🔄 Session Workflow

```
1. Clear State (✅ Fresh start)
   ↓
2. Start Orchestrator (✅ Begin trading)
   ↓
3. Start Monitor (✅ Begin tracking)
   ↓
4. Start Dashboard (✅ Live visualization)
   ↓
5. Monitor Runs for Duration
   ├─ Every 10s: Health check + issue detection
   ├─ Every 30s: Dashboard update
   ├─ On issue: Auto-fix application
   ├─ Every 5m: Detailed log analysis
   └─ Continuous: Capital tracking
   ↓
6. Session Complete (✅ Clean exit)
   ├─ Orchestrator terminated
   ├─ Monitor finalized
   └─ Summary printed
```

## 📞 Support

For issues:
1. Check if orchestrator is running: `pgrep -f MASTER_SYSTEM_ORCHESTRATOR`
2. Check orchestrator logs: `cat /tmp/octivault_orchestrator.log`
3. Check monitor health scores (< 60 = issue)
4. Review `monitoring/dashboard_metrics.json` for current state
5. Restart with fresh state if needed

---

**Created:** May 2, 2026
**Version:** 1.0
**Status:** Production Ready ✅
