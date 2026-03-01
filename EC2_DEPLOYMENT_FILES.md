---
title: EC2 Deployment - Required Files Checklist
date: 2026-02-22
version: 1.0
status: Complete
---

# 📦 EC2 DEPLOYMENT - COMPLETE FILE LIST

## Overview

This document lists **ALL files required** for deploying the regime trading system to EC2 (Ubuntu server at `ip-172-31-37-246`).

**Total Files:** ~80+ files  
**Total Code:** ~1,700+ lines (core) + dependencies  
**Estimated Size:** ~500MB+ (with venv + dependencies)

---

## ⚡ QUICK DEPLOYMENT SCRIPT

```bash
#!/bin/bash
# Deploy to EC2 - Fast Version
TARGET_IP="ubuntu@ip-172-31-37-246"
TARGET_DIR="/home/ubuntu/octivault_trader"

# 1. Sync all core files
rsync -avz --delete \
  /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/ \
  $TARGET_IP:$TARGET_DIR/ \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="logs/*" \
  --exclude="data/*" \
  --exclude="*.pyc" \
  --exclude="__pycache__"

# 2. Create venv and install dependencies on EC2
ssh $TARGET_IP "cd $TARGET_DIR && python3 -m venv .venv && \
  .venv/bin/pip install -r requirements.txt && \
  .venv/bin/python launch_regime_trading.py --mode paper --duration 0.1"
```

---

## 📋 REQUIRED FILES - CATEGORIZED

### CATEGORY 1: CORE REGIME TRADING SYSTEM (ESSENTIAL)

#### A. Integration & Adapter Layer
```
✅ core/regime_trading_integration.py (600 lines, 23KB)
   └─ RegimeTradingAdapter main class
   └─ State synchronization
   └─ Trade execution routing
```

#### B. Live Trading System Architecture
```
✅ live_trading_system_architecture.py (400 lines, 18KB)
   └─ RegimeDetectionEngine
   └─ ExposureController
   └─ PositionSizer
   └─ UniverseManager
   └─ LiveTradingOrchestrator
```

#### C. Data Pipeline
```
✅ live_data_pipeline.py (350 lines, 16KB)
   └─ LiveDataFetcher
   └─ LivePositionManager
```

#### D. Launcher & CLI
```
✅ launch_regime_trading.py (450 lines, 20KB)
   └─ Command-line interface
   └─ Paper/Live/Backtest modes
   └─ Component initialization
```

#### E. Validation & Testing
```
✅ extended_walk_forward_validator.py (280 lines, 14KB)
   └─ 24-month backtest framework
   └─ Performance metrics
```

```
✅ extended_historical_ingestion.py (200 lines, 12KB)
   └─ Historical data fetching
   └─ Data preprocessing
```

---

### CATEGORY 2: EXISTING CORE OCTIVAULT FILES (CRITICAL)

These files from the existing codebase **MUST** be on EC2 (already there or need sync):

```
CORE TRADING ENGINE:
  ✅ main.py (modified ~57 lines added)
  ✅ main_live.py
  ✅ run_full_system.py
  ✅ dry_run_test.py

SHARED STATE & MANAGERS:
  ✅ core/shared_state.py
  ✅ core/exchange_client.py
  ✅ core/execution_manager.py (CRITICAL - has recent fixes)
  ✅ core/market_data_feed.py
  ✅ core/portfolio_manager.py
  ✅ core/risk_manager.py
  ✅ core/agent_manager.py
  ✅ core/capital_allocator.py
  ✅ core/position_manager.py
  ✅ core/tp_sl_engine.py
  ✅ core/meta_controller.py
  ✅ core/recovery_engine.py
  ✅ core/alert_system.py
  ✅ core/watchdog.py
  ✅ core/heartbeat.py
  ✅ core/config.py
  ✅ core/database_manager.py
  ✅ core/strategy_manager.py

AGENTS:
  ✅ agents/cot_assistant.py
  ✅ agents/symbol_screener.py
  ✅ agents/swing_trade_hunter.py
  ✅ agents/*.py (all agent files)

UTILITIES:
  ✅ utils/logging_setup.py
  ✅ utils/shared_state_tools.py
  ✅ utils/indicators.py
  ✅ utils/pnl_calculator.py
  ✅ utils/*.py (all utility files)

MODELS:
  ✅ portfolio/
  ✅ tests/
```

---

### CATEGORY 3: CONFIGURATION FILES (REQUIRED)

```
ENVIRONMENT:
  ✅ .env (CRITICAL - API keys, paper trading flags)
  ✅ .env.example (for reference)

PYTHON:
  ✅ requirements.txt (Python dependencies)
  ✅ setup.py (if using)

DOCKER (Optional, but recommended):
  ✅ Dockerfile
  ✅ docker-compose.yml
  ⚠️  Only if running in containers
```

---

### CATEGORY 4: DOCUMENTATION (REFERENCE)

These files help with deployment and monitoring:

```
DEPLOYMENT DOCS:
  ✅ IMPLEMENTATION_STATUS.md (current status)
  ✅ CODEBASE_MODIFICATIONS.md (what was changed)
  ✅ DEPLOYMENT_CHECKLIST.md (pre/post deployment)
  ✅ DEPLOYMENT_FIX.md (known issues)
  ✅ deployment_guide.py (step-by-step)

SYSTEM DOCS:
  ✅ README_LIVE_TRADING.md (system overview)
  ✅ SYSTEM_ARCHITECTURE.md (technical details)
  ✅ QUICKSTART.md (quick start guide)

REFERENCE:
  ✅ PROJECT_INDEX.md (project structure)
```

---

### CATEGORY 5: DATA & LOGS (CREATE ON EC2)

These directories should be created on EC2 (don't sync):

```
DATA:
  📁 data/ (historical data, will be fetched fresh)
  📁 validation_outputs/ (backtest results)

LOGS:
  📁 logs/ (trading logs, should be on EC2 only)
     ├─ trader.log
     ├─ regime.log
     └─ system.log

CACHE:
  📁 .cache/ (model cache, if using)
  📁 __pycache__/ (Python cache - auto-created)
```

---

### CATEGORY 6: TESTING FILES (OPTIONAL)

If running tests on EC2:

```
TESTS:
  ✅ tests/test_regime_system.py (600 lines, comprehensive)
  ✅ tests/*.py (all test files)

TEST DATA:
  ✅ test_data/ (if included)
```

---

## 📊 DEPLOYMENT MATRIX

| Category | Files | Size | Critical | Notes |
|----------|-------|------|----------|-------|
| **Regime Trading** | 4 | ~70KB | ✅ YES | New system core |
| **Octivault Core** | 20+ | ~500KB | ✅ YES | Existing system |
| **Agents** | 15+ | ~200KB | ✅ YES | Trading logic |
| **Utils** | 10+ | ~100KB | ✅ YES | Support code |
| **Config** | 3 | ~50KB | ✅ YES | API keys, settings |
| **Documentation** | 10+ | ~200KB | ⚠️ OPTIONAL | Reference only |
| **Tests** | 5+ | ~80KB | ⚠️ OPTIONAL | Development use |
| **Data/Logs** | — | Dynamic | ✅ Create | Created at runtime |
| **TOTAL** | ~80+ | ~500MB+ | | With venv + deps |

---

## 🔄 DEPLOYMENT ORDER

### Phase 1: Upload Core Code
```bash
# 1. Sync all Python code
rsync -avz core/ agents/ utils/ portfolio/ tests/ \
  $TARGET:~/octivault_trader/

# 2. Sync entry points
rsync -avz *.py .env .env.example requirements.txt \
  $TARGET:~/octivault_trader/
```

### Phase 2: Install Dependencies
```bash
ssh $TARGET "cd ~/octivault_trader && \
  python3 -m venv .venv && \
  .venv/bin/pip install -r requirements.txt"
```

### Phase 3: Verify Installation
```bash
ssh $TARGET ".venv/bin/python -c 'from core.regime_trading_integration import RegimeTradingAdapter; print(OK)'"
```

### Phase 4: Test Paper Trading
```bash
ssh $TARGET "cd ~/octivault_trader && \
  .venv/bin/python launch_regime_trading.py --mode paper --duration 0.1"
```

### Phase 5: Setup Cron (Optional)
```bash
ssh $TARGET "crontab -e"
# Add: 0 * * * * ~/octivault_trader/.venv/bin/python ~/octivault_trader/launch_regime_trading.py >> ~/octivault_trader/logs/trader.log 2>&1
```

---

## 🚨 CRITICAL FILES (DO NOT MISS)

These files **MUST** be on EC2 for the system to work:

```
ABSOLUTELY CRITICAL:
  1. ✅ core/regime_trading_integration.py
  2. ✅ core/execution_manager.py (MUST be latest version - has recent fixes)
  3. ✅ core/shared_state.py
  4. ✅ core/exchange_client.py (API connectivity)
  5. ✅ live_trading_system_architecture.py
  6. ✅ live_data_pipeline.py
  7. ✅ launch_regime_trading.py
  8. ✅ main.py (with regime trading additions)
  9. ✅ .env (with API keys)
  10. ✅ requirements.txt

IMPORTANT BUT OPTIONAL:
  • Tests (not needed for production)
  • Documentation (useful for reference)
  • Data files (will be fetched fresh from Binance)
```

---

## ✅ PRE-DEPLOYMENT CHECKLIST

**On Local Machine (before deploying):**

- [ ] All Python files syntax-checked
  ```bash
  python -m py_compile core/regime_trading_integration.py
  ```

- [ ] All imports verified locally
  ```bash
  python -c "from core.regime_trading_integration import RegimeTradingAdapter; print('OK')"
  ```

- [ ] `.env` has correct API keys
  ```bash
  grep "BINANCE_API_KEY" .env
  ```

- [ ] `requirements.txt` is current
  ```bash
  pip freeze | head -20
  ```

**On EC2 (after deploying):**

- [ ] Files copied correctly
  ```bash
  ls -la ~/octivault_trader/core/regime_trading_integration.py
  ```

- [ ] venv created successfully
  ```bash
  ls -la ~/octivault_trader/.venv/bin/python
  ```

- [ ] All imports work on EC2
  ```bash
  .venv/bin/python -c "from core.regime_trading_integration import RegimeTradingAdapter; print('OK')"
  ```

- [ ] Paper trading test runs
  ```bash
  .venv/bin/python launch_regime_trading.py --mode paper --duration 0.1
  ```

- [ ] Logs created successfully
  ```bash
  ls -la ~/octivault_trader/logs/
  ```

---

## 📝 DEPLOYMENT SCRIPTS

### Script 1: Full Deployment (Recommended)
```bash
#!/bin/bash
set -e

TARGET="ubuntu@ip-172-31-37-246"
REMOTE_DIR="/home/ubuntu/octivault_trader"

echo "🚀 Starting deployment to EC2..."

# 1. Create remote directory
ssh $TARGET "mkdir -p $REMOTE_DIR"

# 2. Sync all code
echo "📦 Syncing code..."
rsync -avz --delete \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude="logs/*" \
  --exclude="data/*" \
  /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/ \
  $TARGET:$REMOTE_DIR/

# 3. Create venv
echo "📦 Creating Python environment..."
ssh $TARGET "cd $REMOTE_DIR && \
  python3 -m venv .venv && \
  .venv/bin/pip install --upgrade pip"

# 4. Install dependencies
echo "📦 Installing dependencies..."
ssh $TARGET "cd $REMOTE_DIR && \
  .venv/bin/pip install -r requirements.txt"

# 5. Test imports
echo "✅ Testing imports..."
ssh $TARGET "cd $REMOTE_DIR && \
  .venv/bin/python -c 'from core.regime_trading_integration import RegimeTradingAdapter; print(\"✅ IMPORTS OK\")'"

# 6. Create directories
echo "📁 Creating directories..."
ssh $TARGET "cd $REMOTE_DIR && mkdir -p logs data validation_outputs"

# 7. Brief test
echo "🧪 Running brief test..."
ssh $TARGET "cd $REMOTE_DIR && \
  .venv/bin/python launch_regime_trading.py --mode paper --duration 0.01"

echo "✅ Deployment complete!"
```

### Script 2: Quick Sync (For Updates)
```bash
#!/bin/bash
# Quick sync after code changes
TARGET="ubuntu@ip-172-31-37-246"
REMOTE_DIR="/home/ubuntu/octivault_trader"

rsync -avz \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude="logs/*" \
  /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/ \
  $TARGET:$REMOTE_DIR/

echo "✅ Code synced. Remember to restart services if running."
```

### Script 3: Start Cron Job
```bash
#!/bin/bash
# Setup hourly trading
TARGET="ubuntu@ip-172-31-37-246"
CRON_CMD="0 * * * * cd /home/ubuntu/octivault_trader && .venv/bin/python launch_regime_trading.py >> logs/trader.log 2>&1"

ssh $TARGET "crontab -l | grep -q 'launch_regime_trading' || (crontab -l 2>/dev/null; echo '$CRON_CMD') | crontab -"

echo "✅ Cron job installed"
```

---

## 🎯 FILE SIZE SUMMARY

```
Core Regime Trading System:     ~70 KB
  ├─ regime_trading_integration.py: 23 KB
  ├─ live_trading_system_architecture.py: 18 KB
  ├─ live_data_pipeline.py: 16 KB
  └─ launch_regime_trading.py: 20 KB

Existing Octivault Core:         ~500 KB
  ├─ core/: ~300 KB
  ├─ agents/: ~150 KB
  └─ utils/: ~50 KB

Configuration & Dependencies:    ~50 KB
  ├─ .env: ~2 KB
  ├─ requirements.txt: ~10 KB
  └─ other configs: ~38 KB

Python Virtual Environment:      ~300-400 MB
  ├─ Site packages (ccxt, pandas, numpy, etc.)
  └─ Will be created on EC2

TOTAL TO SYNC: ~620 KB (all code)
TOTAL WITH VENV: ~350 MB+
```

---

## 🔗 DEPENDENCIES (In requirements.txt)

Key packages needed on EC2:

```
TRADING:
  - ccxt>=1.91.x (Binance API)
  - pandas>=1.3.x (Data processing)
  - numpy>=1.21.x (Numerical computing)

ASYNC:
  - asyncio (built-in)
  - aiohttp>=3.8.x (async HTTP)

ML (Optional):
  - scikit-learn>=0.24.x (ML models)

MONITORING:
  - python-dotenv (environment variables)
  - logging (built-in)
```

---

## 🚨 KNOWN ISSUES & FIXES

### Issue 1: Old Files on EC2
**Problem:** EC2 has older version of `execution_manager.py` with syntax errors

**Solution:**
```bash
rsync -avz /path/to/core/execution_manager.py ubuntu@ip-172-31-37-246:~/octivault_trader/core/
```

**Files to always update:**
  - `core/execution_manager.py`
  - `core/shared_state.py`
  - `core/meta_controller.py`

### Issue 2: Python Version
**Problem:** Python 3.7 might be default on Ubuntu

**Solution:**
```bash
ssh ubuntu@ip-172-31-37-246
python3 --version  # Should be >= 3.8
sudo apt update && sudo apt install python3.10
python3.10 -m venv .venv
```

### Issue 3: API Keys Not Set
**Problem:** `.env` missing on EC2

**Solution:**
```bash
# Copy .env with real API keys (don't commit to git!)
scp -r .env ubuntu@ip-172-31-37-246:~/octivault_trader/
```

---

## 📊 MONITORING AFTER DEPLOYMENT

**Check logs on EC2:**
```bash
ssh ubuntu@ip-172-31-37-246 "tail -f ~/octivault_trader/logs/trader.log"
```

**Monitor cron execution:**
```bash
ssh ubuntu@ip-172-31-37-246 "tail /var/log/syslog | grep CRON"
```

**Check system health:**
```bash
ssh ubuntu@ip-172-31-37-246 "df -h && free -h && ps aux | grep python"
```

---

## 📞 SUPPORT

**If deployment fails:**

1. Check EC2 logs: `cat ~/octivault_trader/logs/trader.log`
2. Verify Python: `.venv/bin/python --version`
3. Test imports: `.venv/bin/python -c "import ccxt"`
4. Check API keys: `grep BINANCE ~/.env`
5. Review DEPLOYMENT_CHECKLIST.md

---

**Last Updated:** February 22, 2026  
**Status:** ✅ Ready for Deployment  
**Next Step:** Execute deployment scripts
