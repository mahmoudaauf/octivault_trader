#!/usr/bin/env python3
"""
🚀 AUTONOMOUS SYSTEM STARTUP GUIDE
==================================
Complete guide to run the trading system autonomously on your machine.
"""

import subprocess
from pathlib import Path
import sys

def print_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")

def print_section(text):
    print(f"\n{text}")
    print("-" * 80)

def main():
    project_root = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
    
    print_header("🤖 OCTIVAULT AUTONOMOUS TRADING SYSTEM - STARTUP GUIDE")
    
    print("""
This system is designed to trade autonomously 24/7 on the Binance exchange.

KEY FEATURES:
✅ Fully autonomous - no manual intervention needed
✅ Error recovery - auto-restarts on failures
✅ LIVE trading - real money mode
✅ Real-time monitoring - track performance live
✅ Multi-agent system - 6+ trading strategies
✅ Automatic profit compounding - reinvest gains
✅ Position management - take-profit/stop-loss
✅ Signal generation - AI-driven trade signals
""")
    
    print_section("📋 QUICK START - THREE OPTIONS")
    
    print("""
OPTION 1: Simple (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ./QUICK_START_AUTONOMOUS.sh

  This runs the system in the foreground with full output.
  Perfect for development and testing.


OPTION 2: Background (Production)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ./AUTONOMOUS_START.sh --background

  This runs in the background. Monitor with:
  ./AUTONOMOUS_START.sh --logs


OPTION 3: With Real-time Monitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Terminal 1 - Start system:
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ./QUICK_START_AUTONOMOUS.sh

  Terminal 2 - Monitor system:
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ./AUTONOMOUS_START.sh --monitor

  This gives you a real-time dashboard of system status.
""")
    
    print_section("🔧 CONFIGURATION")
    
    print("""
The system reads configuration from: .env

Current settings:
""")
    
    # Read .env to show current config
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            # Extract key settings
            for line in content.split('\n'):
                if any(x in line for x in ['BINANCE_TESTNET', 'TRADING_MODE', 'LIVE_MODE', 'TESTNET_MODE', 'BASE_TARGET', 'CAPITAL_TARGET']):
                    if not line.startswith('#') and '=' in line:
                        print(f"  {line}")
    
    print("""
To change settings:
  1. Open .env in your text editor
  2. Modify the BINANCE_TESTNET setting:
     - false = LIVE trading (real money)
     - true = TESTNET (fake money)
  3. Save and restart the system

⚠️  WARNING: LIVE MODE is currently ENABLED
    The system will trade with REAL MONEY
    Only run in LIVE mode if you have funded your account
""")
    
    print_section("📊 MONITORING")
    
    print("""
While the system is running, you can:

1. View logs:
   tail -f logs/octivault_trader.log

2. Monitor system:
   python3 REALTIME_MONITOR.py

3. Check process:
   ps aux | grep "LIVE_ED25519"

4. Stop system:
   pkill -f "LIVE_ED25519"
""")
    
    print_section("🔄 SYSTEM ARCHITECTURE")
    
    print("""
The autonomous system consists of:

1. 🔗 Exchange Client
   └─ Connects to Binance API
   └─ Handles WebSocket authentication
   └─ Manages user data streams

2. 📊 Market Data
   ├─ Real-time price feeds (WebSocket)
   └─ Historical OHLCV candles (REST)

3. 🤖 AI Agents (6+ strategies)
   ├─ Trend Hunter - trend following
   ├─ ML Forecaster - machine learning
   ├─ DIP Sniper - bounce trading
   ├─ IPO Chaser - new listing detection
   ├─ RL Strategist - reinforcement learning
   └─ News Reactor - news-based signals

4. 🧠 Signal Manager
   ├─ Receives signals from agents
   ├─ Validates signals
   └─ Caches for execution

5. 🎯 Execution Manager
   ├─ Validates risk constraints
   ├─ Places orders on exchange
   └─ Manages fills and fees

6. 📈 TP/SL Engine
   ├─ Monitors open positions
   ├─ Executes take-profit targets
   ├─ Executes stop-loss orders
   └─ Manages position exits

7. 🔄 Meta Controller
   ├─ Orchestrates all components
   ├─ Runs trading cycles
   └─ Handles recovery and restarts

All components run continuously and automatically.
""")
    
    print_section("✅ VERIFICATION CHECKLIST")
    
    print("""
Before running in LIVE mode, verify:

□ API Keys are correct (test with testnet first)
□ Account is funded with real USDT
□ Network connection is stable
□ Computer won't shut down during trading
□ Logs are being written correctly
□ System can connect to Binance

Quick verification test:

  python3 -c "
from core.config import Config
from core.exchange_client import ExchangeClient
import asyncio

async def test():
    config = Config()
    exchange = ExchangeClient()
    await exchange.start()
    balances = await exchange.get_spot_balances()
    print(f'USDT Balance: {balances.get(\"USDT\", {}).get(\"free\", 0):.2f}')
    await exchange.close()

asyncio.run(test())
  "
""")
    
    print_section("📞 TROUBLESHOOTING")
    
    print("""
Issue: "Cannot import core modules"
Solution: Make sure you're in the project directory and venv is activated
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  source venv/bin/activate

Issue: "API key not valid"
Solution: Check .env file - make sure BINANCE_TESTNET=false for live trading

Issue: "No signals being generated"
Solution: The system needs to warm up - agents take time to analyze markets
  Wait at least 5-10 minutes for first signals

Issue: "Process crashes immediately"
Solution: Check logs for errors
  tail -100 logs/octivault_trader.log

Issue: "Not enough balance"
Solution: Minimum 10 USDT required for first trade
  Check your account at https://www.binance.com
""")
    
    print_section("💡 TIPS FOR SUCCESS")
    
    print("""
1. START SMALL - Test with 10-50 USDT first
2. MONITOR LOGS - Watch logs/octivault_trader.log for activity
3. USE TESTNET - Test with testnet mode first (BINANCE_TESTNET=true in .env)
4. PATIENCE - First trades may take 5-15 minutes as agents analyze
5. STABILITY - Keep computer/connection running continuously
6. BACKUPS - System maintains trade journal in logs/
7. COMPOUND - Enable auto-compounding for exponential growth
""")
    
    print_section("🚀 LAUNCHING NOW")
    
    response = input("Ready to start the autonomous system? (yes/no): ").strip().lower()
    
    if response == "yes":
        print("\n✅ Starting autonomous system...\n")
        
        # Change to project directory and activate venv
        import os
        os.chdir(str(project_root))
        
        # Try to run the quick start script
        script_path = project_root / "QUICK_START_AUTONOMOUS.sh"
        if script_path.exists():
            subprocess.run(["bash", str(script_path)])
        else:
            print("Script not found, running directly...")
            subprocess.run([
                f"{project_root}/venv/bin/python3",
                f"{project_root}/🚀_LIVE_ED25519_TRADING.py"
            ])
    else:
        print("\n✅ Setup complete. When ready, run:")
        print(f"  cd {project_root}")
        print("  ./QUICK_START_AUTONOMOUS.sh")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
