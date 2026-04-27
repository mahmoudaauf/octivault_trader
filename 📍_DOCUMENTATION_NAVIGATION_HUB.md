# 📍 OCTI AI TRADING BOT - DOCUMENTATION NAVIGATION HUB

**Your Complete Guide to the System | Production Ready | Last Updated: 2026-02-14**

---

## 🎯 WHERE TO START

### 👤 I'm a **TRADER / OPERATIONS PERSON**
**Time needed:** 15 minutes to get running  
**Read this first:**
- **[OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md)** ⭐
  - ✅ 5-step startup procedure
  - ✅ What to expect during normal operation
  - ✅ Common problems and instant fixes
  - ✅ Daily monitoring checklist
  - ✅ Emergency procedures

**Then reference:**
- [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) → "Monitoring & Observability" section
- [TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md) → "Error Handling & Recovery" section

---

### 👨‍💻 I'm a **DEVELOPER / ARCHITECT**
**Time needed:** 45-60 minutes to understand everything  
**Read this first:**
- **[COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md)** ⭐
  - ✅ Complete 7-layer system architecture
  - ✅ How each component fits together
  - ✅ Decision-making pipeline
  - ✅ Capital management system
  - ✅ Risk management framework
  - ✅ Configuration parameters

**Then deep dive:**
- [TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md)
  - Step-by-step decision flows with ASCII diagrams
  - Exact gating logic and order of operations
  - Policy nudge calculations
  - Error handling classification

---

### 🔧 I want to **UNDERSTAND A SPECIFIC PROCESS**
Jump to: [Quick Topic Lookup](#-quick-topic-lookup-by-question)

---

### 🚨 **SYSTEM IS DOWN / BROKEN**
Jump to: [Emergency Procedures](#-emergency-procedures)

---

## 📋 QUICK TOPIC LOOKUP BY QUESTION

### Starting & Running the System
```
Q: How do I start the bot?
A: OPERATIONAL_QUICK_START.md → Section 1 (Startup Sequence)
   Time: 5 min

Q: What happens on startup?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Core Operational Flows → Bootstrap
   Time: 10 min

Q: What does each layer do?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → System Architecture Overview
   Time: 15 min
```

### Trading & Position Management
```
Q: How are trading decisions made?
A: TECHNICAL_DECISION_FLOWS.md → Main Decision Flow
   Time: 15 min

Q: Why did/didn't a trade execute?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Decision-Making Pipeline → 6-Layer Gates
   Time: 10 min

Q: How are position sizes calculated?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Capital Management System
   Time: 10 min

Q: What are all the ways a trade can be blocked?
A: TECHNICAL_DECISION_FLOWS.md → Arbitration Section
   Time: 10 min
```

### Profit & Loss Management
```
Q: When does the bot close winning trades?
A: TECHNICAL_DECISION_FLOWS.md → SELL Order Flow
   Time: 10 min

Q: How are stop-losses enforced?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → TP/SL Management
   Time: 5 min

Q: Why am I holding a losing position?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Position Lifecycle
   Time: 5 min
```

### Portfolio & Capital
```
Q: What is "dust" and how is it handled?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Dust & Portfolio Recovery
   Time: 15 min

Q: Why did the system stop trading?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Risk Management Layers
   Time: 10 min

Q: How does regime switching work?
A: TECHNICAL_DECISION_FLOWS.md → Regime Determination Flow
   Time: 10 min

Q: What's my current position limit?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Capital Management → Position Limits
   Time: 5 min
```

### System Health & Monitoring
```
Q: What metrics should I check daily?
A: OPERATIONAL_QUICK_START.md → Section 5 (Performance Monitoring)
   Time: 5 min

Q: How do I know the system is working?
A: OPERATIONAL_QUICK_START.md → Section 2 (Normal Operation)
   Time: 5 min

Q: What's a healthy daily return?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Performance Targets
   Time: 3 min

Q: How do I read the logs?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Monitoring & Observability
   Time: 10 min
```

### Configuration & Tuning
```
Q: What parameters can I adjust?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Configuration & Mode System
   Time: 15 min

Q: Should I change the defaults?
A: OPERATIONAL_QUICK_START.md → Section 7 (Configuration Reference)
   Time: 10 min

Q: What's the safe range for each parameter?
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Configuration Ranges
   Time: 10 min
```

### Problems & Troubleshooting
```
Q: No trades are executing - why?
A: OPERATIONAL_QUICK_START.md → Section 4 (Common Issues)
   Time: 5 min

Q: System is stuck in MICRO_SNIPER mode
A: OPERATIONAL_QUICK_START.md → Common Issues #2
   Time: 5 min

Q: Dust is accumulating too fast
A: COMPREHENSIVE_SYSTEM_SUMMARY.md → Dust Recovery Strategies
   Time: 10 min

Q: System keeps crashing
A: OPERATIONAL_QUICK_START.md → Emergency Procedures
   Time: 5 min

Q: I see errors in the logs
A: TECHNICAL_DECISION_FLOWS.md → Error Handling & Recovery
   Time: 15 min
```

---

## 🏗️ SYSTEM ARCHITECTURE (AT A GLANCE)

```
LAYER 7: Master Orchestrator
  └─ Startup, lifecycle, shutdown

LAYER 6: Watchdog & Monitoring
  └─ Health checks, metrics, alerts

LAYER 5: Signal Processing
  ├─ Market data → Signals
  ├─ Multi-agent consensus
  └─ Deduplication & batching

LAYER 4: Decision-Making
  ├─ Main evaluation loop
  ├─ 6-layer arbitration gates
  ├─ Policy matrix evaluation
  └─ State machine management

LAYER 3: Execution
  ├─ Order routing & submission
  ├─ Position sizing
  ├─ Take-profit & stop-loss
  └─ Rotation authorization

LAYER 2: Capital Management
  ├─ Balance validation
  ├─ Position limits
  ├─ Quota enforcement
  └─ Dust consolidation

LAYER 1: Exchange & State
  ├─ Binance API interface
  ├─ Real-time data feeds
  ├─ Position tracking
  └─ Order reconciliation
```

**📖 Full architecture details:** [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md)

---

## 🎯 KEY CONCEPTS (UNDERSTAND THESE FIRST)

### The 6 Decision Gates (ALL must pass)
1. **Lifecycle State** - Is the position in a valid state for trading?
2. **Portfolio Health** - Do I have room for more positions?
3. **Capital Availability** - Do I have free USDT to allocate?
4. **Economic Gate** - Will this trade cover its costs?
5. **Signal Confidence** - Is the signal high-confidence enough?
6. **Regime Gating** - Am I within position limits for my capital level?

**📖 Details:** [TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md) → Main Decision Flow

### The 4 NAV Regimes
- **MICRO_SNIPER** (< $1K) - 1 position, strict capital preservation
- **STANDARD** ($1K-$5K) - 2 positions, balanced trading
- **MULTI_AGENT** (>= $5K) - 3+ positions, growth mode

**📖 Details:** [TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md) → Regime Determination

### The 8 Operational Modes
- **BOOTSTRAP** - First position (special rules)
- **NORMAL** - Steady-state (default)
- **SAFE** - Drawdown recovery (reduced size)
- **PROTECTIVE** - Volatility spike (higher confidence)
- **AGGRESSIVE** - Capital abundant (larger size)
- **RECOVERY** - Major drawdown (no new entries)
- **PAUSED** - Manual pause (no trading)

**📖 Details:** [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) → Configuration & Mode System

### The Dust Classification
- **PERMANENT_DUST** (< $1) - Too small to trade
- **DUST** ($1-$25) - Recoverable via healing
- **MICRO** ($25-$100) - Small but trackable
- **SIGNIFICANT** (> $100) - Normal position

**📖 Details:** [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) → Dust & Portfolio Recovery

---

## 📊 OPERATIONAL FLOWS (THE MAIN PROCESSES)

### Per-Cycle Trading Flow (300-500ms)
```
1. Market data update (100ms)
   └─ Fetch current prices, volumes, conditions
2. Signal intake (50ms)
   └─ Collect signals from agents
3. Signal filtering (50ms)
   └─ Deduplicates, removes old signals
4. Decision arbitration (100ms)
   └─ Applies all 6 gates to each signal
5. Order submission (50-100ms)
   └─ Places winning signal as order
6. Bookkeeping (50ms)
   └─ Updates metrics, records trade

→ All 6 gates documented: TECHNICAL_DECISION_FLOWS.md
```

### TP/SL Management (Every 50-200ms)
```
1. Poll open positions
2. Fetch current price
3. Check if +2% (take-profit)
4. Check if -1% (stop-loss)
5. Generate exit signal if needed
6. Process exit signal through normal flow
```

### Dust Recovery (Every 60s)
```
1. Detect dust accumulation
2. Attempt consolidation merge
3. Monitor for break-even escape
4. Emergency liquidation if > 60% dust
5. Update portfolio health metrics
```

### Policy Nudge Application (Per mode change)
```
1. Evaluate system state
2. Determine current mode
3. Load mode's policy weights
4. Calculate adjustments (size, confidence, cooldown)
5. Apply to active and future trades
```

---

## 🚨 EMERGENCY PROCEDURES

### System Not Trading
1. Check if in RECOVERY mode (drawdown > 20%) → Expected, wait for recovery
2. Check if in PAUSED mode → Manual pause, restart to resume
3. Check log for errors → Common Issues guide
4. Verify API connectivity → Can you access binance.com?
5. Last resort → Restart system, it auto-reconciles

**📖 Guide:** [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) → Section 4 & 6

### System Crashed
1. Note the time of crash
2. Check position reconciliation with exchange
3. Restart the system → Auto-reconciles from Binance
4. Monitor for 2-3 cycles to confirm stability
5. Review crash logs for root cause

**📖 Guide:** [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) → Emergency Procedures

### Extreme Loss (> 20% drawdown)
1. System enters RECOVERY mode automatically
2. No new entries until recovery completes
3. Monitor closely for stabilization
4. DO NOT panic-close positions (may break strategy)
5. If position goes to -5%, stop-loss exits automatically

**📖 Guide:** [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) → Risk Management

### Manual Stop Required
1. Ctrl+C in terminal where bot is running
2. Wait for graceful shutdown (30-60 seconds)
3. Check that all positions are recorded
4. Close positions manually via Binance if needed
5. Restart system when ready

---

## ✅ PRE-DEPLOYMENT CHECKLIST

- [ ] API keys configured in `.env`
- [ ] Network connectivity verified
- [ ] Binance account status checked
- [ ] Read [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md)
- [ ] Reviewed [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) → System Architecture
- [ ] Understood the 6 decision gates
- [ ] Familiar with the 8 operational modes
- [ ] Have emergency procedures handy
- [ ] Set up log monitoring (tail -f /tmp/octivault_master_orchestrator.log)
- [ ] Team aware of system status

---

## 📈 PERFORMANCE EXPECTATIONS

### Daily
- Return: +0.5% to +2.0%
- Win rate: 55-65%
- Number of trades: 3-12 (regime-dependent)
- Maximum intraday drawdown: <5%

### Weekly
- Cumulative return: +3% to +14%
- Capital preservation: >95%
- Position turnover: Active rotation
- Dust ratio: <30%

### Monthly
- Cumulative return: +12% to +56%
- Sharpe ratio: >1.0
- Maximum drawdown (recovered): <20%
- Capital efficiency: 70-80% deployed

---

## 🔍 WHERE TO FIND THINGS

### Core System Files
```
core/meta_controller.py          → Main decision loop (13,000+ lines)
core/policy_manager.py           → Policy & mode system (780 lines)
core/arbitration_engine.py       → 6-layer gating logic
core/execution_manager.py        → Order submission
core/capital_governor.py         → Position limits
core/signal_fusion.py            → Multi-agent consensus
```

### Configuration
```
.env                             → API keys & trading parameters
core/config.py                   → Static configuration
```

### Logs & Monitoring
```
/tmp/octivault_master_orchestrator.log  → Main system logs
Metrics API: http://localhost:8000/metrics
Positions API: http://localhost:8000/positions
```

### Documentation You're Using Now
```
OPERATIONAL_QUICK_START.md        → Trader's playbook
COMPREHENSIVE_SYSTEM_SUMMARY.md   → Architecture & design
TECHNICAL_DECISION_FLOWS.md       → Decision flowcharts
📍_DOCUMENTATION_NAVIGATION_HUB.md → You are here
```

---

## 📞 QUICK REFERENCE

### Most Important Configuration Parameters
```
DEFAULT_PLANNED_QUOTE=25          # Base position size (USDT)
MAX_SPEND_PER_TRADE_USDT=50       # Maximum position (USDT)
MIN_SIGNAL_CONF=0.50              # Minimum signal confidence
TP_PERCENT=2.0                    # Take-profit at +2%
SL_PERCENT=-1.0                   # Stop-loss at -1%
MAX_POSITIONS_STANDARD=2          # Max open positions
FOCUS_MODE_ENABLED=true           # Restrict to top symbols
DUST_EXIT_ENABLED=true            # Auto-exit dust
```

### Most Important Commands
```bash
# Start the system
python 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# Monitor logs (new terminal)
tail -f /tmp/octivault_master_orchestrator.log

# Check system health
curl http://localhost:8000/metrics

# View open positions
curl http://localhost:8000/positions

# Stop the system
Ctrl+C
```

### Most Important Metrics to Track
- **NAV** - Current portfolio value
- **Active Positions** - Number of open trades
- **Dust Ratio** - % of capital in dust
- **Daily Return** - Today's P&L %
- **Win Rate** - % of profitable trades
- **Max Drawdown** - Largest daily loss

---

## 🎓 LEARNING PATHS

### Fast Track (Get Running Today) - 30 minutes
1. Read [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) → Sections 1-2
2. Start system per Section 1
3. Monitor logs per Section 2
4. Bookmark Section 4 (Common Issues) for quick reference
5. Done! System is running

### Comprehensive Path (Understand Everything) - 2-3 hours
1. Read this file (📍_DOCUMENTATION_NAVIGATION_HUB.md) - 10 min
2. Read [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) - 20 min
3. Read [COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md) - 45 min
4. Read [TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md) - 30 min
5. Review code: core/meta_controller.py - 30 min
6. Run system, observe, ask questions

### Daily Operations Path - 1 hour/day
- Morning: Review [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) → Monitoring checklist
- During trading: Tail the logs
- Evening: Aggregate daily metrics
- As needed: Jump to [Quick Topic Lookup](#-quick-topic-lookup-by-question)

---

## 📚 DOCUMENT MAP & PURPOSES

| Document | Purpose | Audience | Read Time | Format |
|----------|---------|----------|-----------|--------|
| **[OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md)** | How to run & operate the system | Traders/Operators | 15 min | Step-by-step guide |
| **[COMPREHENSIVE_SYSTEM_SUMMARY.md](COMPREHENSIVE_SYSTEM_SUMMARY.md)** | How the system is architected | Developers | 45 min | Architecture guide |
| **[TECHNICAL_DECISION_FLOWS.md](TECHNICAL_DECISION_FLOWS.md)** | How decisions are made | Engineers | 30 min | Decision flowcharts |
| **📍_DOCUMENTATION_NAVIGATION_HUB.md** | Navigation & quick reference | Everyone | 10 min | Index & navigation |

---

## 🎯 NEXT STEPS

### If you haven't started yet:
→ Read [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) NOW
→ Follow Section 1 (Startup Sequence)
→ Come back here if you have questions

### If the system is running:
→ Check Section 5 (Performance Monitoring) in [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md)
→ Monitor daily returns vs targets
→ Use Quick Topic Lookup above for specific questions

### If there's a problem:
→ Jump to [Emergency Procedures](#-emergency-procedures) above
→ Or search [Quick Topic Lookup](#-quick-topic-lookup-by-question) for your issue
→ Or read [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) → Section 4

---

**Version:** 1.0 | **Status:** ✅ READY FOR PRODUCTION | **Last Updated:** 2026-02-14

**Start with:** [OPERATIONAL_QUICK_START.md](OPERATIONAL_QUICK_START.md) ⭐
