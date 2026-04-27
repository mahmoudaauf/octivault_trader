# 🚀 OCTI AI TRADING BOT - OPERATIONAL QUICK START

**Status:** ✅ READY TO DEPLOY  
**Last Updated:** 2026-02-14  
**For:** Traders & Operations Team

---

## 1️⃣ STARTUP SEQUENCE (5 STEPS)

### Step 1: Prerequisites Check
```bash
# Verify environment is ready
✓ Python 3.9+ installed
✓ Binance API key & secret in .env
✓ Network connectivity to Binance
✓ Minimum 100 USDT balance

# Check
python3 -c "from core.config import Config; print('✓ Config loaded')"
```

### Step 2: Approve Live Trading
```bash
# Enable live trading (REQUIRED)
export APPROVE_LIVE_TRADING=YES

# Optional: Start in paper trading mode
export PAPER_TRADING=true
```

### Step 3: Start the System
```bash
# Start Master Orchestrator
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# Expected output:
# ✅ Prerequisite checks passed
# ✅ Configuration loaded
# ✅ Exchange connected
# ✅ All layers initialized
# ✅ Main loop started
```

### Step 4: Monitor System Health
```bash
# In another terminal, watch logs
tail -f /tmp/octivault_master_orchestrator.log | grep "✓\|✅\|⚠️\|ERROR"

# Watch for these indicators:
✓ [Meta:Init] layers started
✓ [Cycle] evaluation completed
✓ [Guard] All checks passed
⚠️ [Guard] Market data not ready
ERROR [Cycle] Critical failure
```

### Step 5: Verify Trading Active
```bash
# Should see signals and trades in logs:
[Meta:Decision] BUY BTCUSDT conf=0.72
[Exec:Order] Placed BUY order id=123456
[Trade] Filled: BTC +0.0006 @ 42500
```

---

## 2️⃣ NORMAL OPERATION

### Main Loop Cycle (Every 100-500ms)
```
Tick ↓
  ├─ Poll prices ← Market data
  ├─ Drain events ← Execution fills
  ├─ Guard check ← Safety gates
  │  ├─ Market data ready?
  │  ├─ Balances available?
  │  ├─ Exchange connected?
  │  └─ Trading hours valid?
  ├─ Signal intake ← Agent signals
  │  ├─ Apply confidence floor (0.50)
  │  ├─ Age filter (< 60s)
  │  └─ Deduplicate (1 BUY, 1 SELL per symbol)
  ├─ Arbitration ← 6-layer gating
  │  ├─ Confidence check
  │  ├─ Expected move check
  │  ├─ Position limit check
  │  ├─ Daily trade limit check
  │  ├─ Regime check
  │  └─ Economic gate check
  ├─ Decision ← Execution route
  │  ├─ BUY? → Check position, reserve symbol, place order
  │  ├─ SELL? → Check lifecycle, check profit gate, close position
  │  └─ SKIP? → Next cycle
  ├─ Execution ← Order submitted
  ├─ Bookkeeping ← Metrics updated
  └─ Next tick →

Expected Frequency:
├─ Market data: ~100ms (every tick)
├─ Signals: ~500ms-2s (new agent signals)
├─ Trades: ~5-60s (per decision + execution)
├─ Fills: ~1-10s (order latency + market)
└─ Metrics update: ~100ms (per tick)
```

### What You'll See in Logs

**🟢 Normal Trading**
```
[Cycle] 1234567: Market data ✓ balance ✓ ops ✓
[Signal] BTCUSDT BUY conf=0.72 expected_move=0.35%
[Arbitration] BTCUSDT BUY passes all gates
[Meta:Decision] BUY BTCUSDT quote=30 USDT
[Exec:Order] Placed BUY order id=123456
[Trade] Filled: BTCUSDT +0.0006 @ 42500
[Metrics] NAV=10120 +1.2% PnL=120 Trades=1
```

**🟡 Guard Blocked (Safe)**
```
[Guard] Market data ready? ✗ (< 1 price update)
[Cycle] Skipping cycle - waiting for market data
[Guard] Will retry next cycle...
[Cycle] 1234568: Market data ✓ ready to trade
```

**🔴 Error (Investigate)**
```
[Guard] Exchange connection failed: timeout
[Alert] Ops plane not ready - stopping trades
[Error] Order placement failed: insufficient balance
[Cycle] PAUSED - waiting for manual intervention
```

---

## 3️⃣ MODE CHANGES & REACTIONS

### Automatic Mode Transitions

**BOOTSTRAP → NORMAL**
```
Trigger: First SELL executed
├─ Position recovered to base capital
├─ Unlock to multi-position trading
├─ Enable rotation
└─ Enable dust healing
```

**NORMAL → SAFE**
```
Trigger: Drawdown > 5%
├─ Reduce position size: 25 USDT → 12.5 USDT
├─ Increase confidence floor: 0.50 → 0.65
├─ Allow existing positions to recover
└─ Resume after recovery: drawdown < 3%
```

**NORMAL → PROTECTIVE**
```
Trigger: Volatility spike or signal quality drop
├─ Increase confidence floor: 0.50 → 0.70
├─ Require longer hold times
├─ Reduce trading frequency
└─ Focus on preservation
```

**ANY → RECOVERY**
```
Trigger: Drawdown > 20%
├─ HALT all new entries
├─ Force SELL weak positions
├─ Focus on capital preservation
├─ Manual intervention may be required
└─ Can continue exiting positions
```

**ANY → PAUSED**
```
Trigger: Manual pause or critical failure
├─ STOP all trading immediately
├─ Continue monitoring
├─ Manual unpause required
└─ Investigate error before resuming
```

### Checking Current Mode

```bash
# View logs for mode info
grep "Mode" /tmp/octivault_master_orchestrator.log | tail -5

# Expected output:
# [PolicyManager] Mode changed: NORMAL → SAFE (drawdown=6.2%)
# [Mode] Current mode: SAFE, position_size_multiplier=0.5x
```

---

## 4️⃣ COMMON ISSUES & SOLUTIONS

### Issue: No Trades Executed (Everything Guard-Blocked)

**Symptoms:**
```
[Guard] Market data ready? ✗
[Guard] Balances available? ✗
[Guard] Exchange connected? ✗
[Cycle] Skipping - guards not ready
```

**Diagnosis:**
```bash
# Check exchange connection
curl -s "https://api.binance.com/api/v3/account" \
  -H "X-MBX-APIKEY: $BINANCE_KEY" | jq '.balances | .[0]'

# Check if Binance is down
curl -s https://status.binance.com | jq '.statusCode'
# Should return 0 (OK) or wait if != 0
```

**Solution:**
```
1. Wait for Binance system recovery (if down)
2. Verify API key/secret in .env
3. Check network connectivity: ping api.binance.com
4. Restart system: Ctrl+C, then python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py
```

### Issue: Stuck in MICRO_SNIPER Mode (Can't Open New Positions)

**Symptoms:**
```
[Regime] Current: MICRO_SNIPER (NAV < 1000)
[WHY_NO_TRADE] Max positions reached: open=1/limit=1
[Meta:PositionLimit] Blocking trade - at regime max
```

**Solution:**
```
Option 1: Close existing position
├─ Wait for TP hit (2% profit)
├─ OR manually trigger SELL via command
└─ Frees up position slot for new entry

Option 2: Deposit capital (move to STANDARD regime)
├─ Deposit to bring NAV > 1000
├─ STANDARD allows 2 positions max
└─ More trading opportunities
```

### Issue: Trades Executing But Not Profitable

**Symptoms:**
```
[Trade] Filled: BTCUSDT -0.0006 @ 42500 (LOSS -$0.50)
[Metrics] PnL=-50 USDT after 10 trades (fees eating gains)
```

**Diagnosis:**
```bash
# Check signal quality
grep "confidence" /tmp/octivault_master_orchestrator.log | tail -10
# Should be > 0.65 for profitable trades

# Check economic edge
grep "edge_bps\|expected_alpha" /tmp/octivault_master_orchestrator.log
# Should be > 35-50 bps

# Check execution slippage
grep "Entry price\|Filled @" /tmp/octivault_master_orchestrator.log
# Entry should match signal price within 0.1%
```

**Solution:**
```
1. Increase MIN_SIGNAL_CONF to 0.60 (quality filter)
2. Increase MIN_EXPECTED_EDGE_BPS to 20 (profitability gate)
3. Wait for higher-confidence signals (skip low-conf)
4. Check agent configuration for bias/calibration
```

### Issue: Dust Accumulating (Many Small Positions)

**Symptoms:**
```
[PortfolioHealth] Fragmentation: SEVERE
├─ Active positions: 15
├─ Dust positions: 12 (< $25)
├─ Dust ratio: 80%
└─ Portfolio needs consolidation
```

**Solution:**
```
Option 1: Automatic healing (default)
├─ Wait 5-10 min for price recovery
├─ System triggers consolidation automatically
└─ Dust exits at break-even or better

Option 2: Manual emergency liquidation
├─ Trigger dust exit policy (set DUST_EXIT_ENABLED=true)
├─ Sacrifice meme coins: DOGE, SHIB, PEPE
├─ Recover capital to base
└─ All in logs

Option 3: Pause trading, manual cleanup
├─ Set mode to PAUSED
├─ Manually SELL dust positions
├─ Focus on largest positions only
```

### Issue: System Crashes or Disconnects

**Symptoms:**
```
ERROR [Network] Connection lost to Binance
ERROR [Memory] Out of memory error
ERROR [Core] Unhandled exception in main loop
```

**Recovery:**
```bash
# Restart the system
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# System auto-recovers:
├─ Re-fetches position snapshot
├─ Reconciles with open_trades
├─ Re-calibrates metrics
├─ Resumes trading with no data loss
```

---

## 5️⃣ PERFORMANCE MONITORING

### Key Metrics to Track

```
Daily Checklist:
├─ NAV trend: Should be +0.5% to +2.0% per day
├─ Win rate: Should be 55-65%
├─ Max drawdown: Should stay < 15%
├─ Sharpe ratio: Should be > 1.0
└─ Trades per hour: Should be 3-12 (normal regime)

Weekly Checklist:
├─ Cumulative return: +3% to +14%
├─ Capital preserved: Still have 95%+ of base
├─ No major drawdown: Recovering quickly
├─ Dust ratio: Should stay < 30%
└─ Position turnover: Active rotation happening

Monthly Checklist:
├─ Cumulative return: +12% to +56% (quarterly target)
├─ Sharpe ratio: Track trend (should increase over time)
├─ Max drawdown: Analyze peak-to-trough recovery time
├─ Capital efficiency: Check % deployed vs reserved
└─ Mode distribution: Time in each mode
```

### Pulling Live Metrics

```bash
# View current metrics
grep "Metrics" /tmp/octivault_master_orchestrator.log | tail -3

# Output:
# [Metrics] NAV=10250 +2.5% PnL=250 Win=6/10 MaxDD=-4.2% Sharpe=1.45

# Count trades today
grep "\[Trade\] Filled:" /tmp/octivault_master_orchestrator.log | wc -l

# Sum realized P&L
grep "realized_pnl" /tmp/octivault_master_orchestrator.log | tail -1

# Check portfolio composition
grep "Position:" /tmp/octivault_master_orchestrator.log | tail -10
```

### Daily Report Script

```bash
#!/bin/bash
# daily_report.sh

echo "=== OCTI BOT DAILY REPORT ==="
echo "Date: $(date)"
echo ""

echo "💰 PERFORMANCE:"
tail -100 /tmp/octivault_master_orchestrator.log | \
  grep "Metrics\|NAV\|PnL" | tail -3

echo ""
echo "📊 TRADES:"
grep "\[Trade\] Filled:" /tmp/octivault_master_orchestrator.log | \
  tail -10 | cut -d' ' -f1-6

echo ""
echo "🎯 PORTFOLIO:"
grep "Position:\|Dust\|Concentration" /tmp/octivault_master_orchestrator.log | tail -5

echo ""
echo "⚠️ ALERTS:"
grep "Guard\|Error\|Warning" /tmp/octivault_master_orchestrator.log | \
  tail -5 | grep -v "✓"
```

---

## 6️⃣ EMERGENCY PROCEDURES

### Manual Stop (Graceful Shutdown)

```bash
# Press Ctrl+C in terminal running bot
# System will:
# ├─ Stop accepting new trade signals
# ├─ Allow open positions to close naturally
# ├─ Save metrics to disk
# ├─ Clean shutdown in 5-30 seconds
# └─ Exit code 0 (success)

# Or trigger remote shutdown
curl -X POST http://localhost:8000/stop

# Verify stopped
ps aux | grep "MASTER_SYSTEM_ORCHESTRATOR" || echo "✓ Stopped"
```

### Forced Pause (Freeze Trading)

```bash
# Pause without stopping system
# In logs:
# [Mode] Changed to PAUSED
# [Cycle] No trading - manual pause active

# To resume:
# [Mode] Changed from PAUSED to NORMAL
```

### Position Recovery (After Crash)

```bash
# System automatically reconciles on restart:
# ├─ Fetches current positions from Binance
# ├─ Compares with local open_trades
# ├─ Updates metrics to current state
# ├─ Clears any stale references
# └─ Ready to trade

# Verify reconciliation in logs:
grep "reconcile\|Recovery" /tmp/octivault_master_orchestrator.log
```

### Emergency Liquidation (Full Portfolio Close)

```bash
# If system is critically damaged:
# 1. Pause the bot (Ctrl+C)
# 2. Export to exchange for manual close:
python3 << 'EOF'
from core.exchange_client import ExchangeClient
from core.config import Config

config = Config()
client = ExchangeClient(config)

# Get all open positions
positions = client.fetch_open_positions()
for pos in positions:
    print(f"Position: {pos['symbol']} qty={pos['qty']}")
    # Manual close: client.sell(pos['symbol'], pos['qty'], 'MARKET')

EOF
```

---

## 7️⃣ CONFIGURATION QUICK REFERENCE

### Most Important Settings

```
.env file (configure these first):

# API Connection
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Trading Parameters
DEFAULT_PLANNED_QUOTE=25          # Base position size (USDT)
MAX_SPEND_PER_TRADE_USDT=50       # Max position size (USDT)
MIN_SIGNAL_CONF=0.50              # Signal ingestion threshold
MIN_ENTRY_QUOTE_USDT=25           # Minimum to trade
TP_PERCENT=2.0                    # Take profit %
SL_PERCENT=-1.0                   # Stop loss %

# Risk Management
MAX_POSITIONS_STANDARD=2          # Max open positions
MAX_TRADES_PER_HOUR=12            # Rate limit
DAILY_DRAWDOWN_HARD_STOP=30       # Emergency stop

# Features
FOCUS_MODE_ENABLED=true           # Restrict to top symbols
DUST_EXIT_ENABLED=true            # Auto-exit dust
ECONOMIC_GUARD_ENABLED=true       # Fee-aware gating
BOOTSTRAP_SEED_ENABLED=false      # First-trade seed

# Logging
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
LOG_FILE=/tmp/octivault.log       # Log output path
```

### Changing Settings On-The-Fly

```bash
# Increase position size
sed -i 's/DEFAULT_PLANNED_QUOTE=25/DEFAULT_PLANNED_QUOTE=50/' .env

# Reduce trading frequency
sed -i 's/MAX_TRADES_PER_HOUR=12/MAX_TRADES_PER_HOUR=6/' .env

# Lower confidence floor (more aggressive)
sed -i 's/MIN_SIGNAL_CONF=0.50/MIN_SIGNAL_CONF=0.45/' .env

# Restart to apply changes
kill %1  # Stop current bot
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py  # Restart
```

---

## ✅ STARTUP CHECKLIST

Before running, verify:

```
INFRASTRUCTURE
□ Python 3.9+ installed: python3 --version
□ Dependencies installed: pip list | grep -E "python-binance|aiohttp|redis"
□ .env file exists with valid API keys
□ Network connectivity: ping api.binance.com
□ Sufficient disk space: df -h (at least 1GB)

ACCOUNT SETUP
□ Binance account verified (not banned/liquidated)
□ API key has trading permissions (not read-only)
□ Account has > 100 USDT balance
□ No existing positions stuck from previous runs
□ Order history is clear (no stuck pending orders)

CONFIGURATION
□ APPROVE_LIVE_TRADING=YES set
□ Trading pair symbols configured
□ Position size matches account (not over-sized)
□ TP/SL percentages reasonable (2%/1%)
□ Risk limits defined

FINAL CHECK
□ Read latest system summary
□ Review recent fixes (Phase 2 fixes applied)
□ No obvious errors in logs from previous run
□ Team notified of bot startup
□ Ready to monitor live performance
```

---

**Questions?** Check `COMPREHENSIVE_SYSTEM_SUMMARY.md` for detailed documentation.  
**Issues?** Review error logs: `tail -f /tmp/octivault_master_orchestrator.log`  
**Emergency?** Press Ctrl+C to stop gracefully.

**Happy Trading! 🚀**
