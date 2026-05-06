#!/usr/bin/env bash
#
# 🎯 QUICK REFERENCE - MONITORING COMMANDS
#
# Copy & paste these commands to quickly manage your monitoring system

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                🎯 OCTIVAULT MONITORING - QUICK REFERENCE                  ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 STATUS CHECK
────────────────────────────────────────────────────────────────────────────

  # Check system health in 3 seconds
  python check_status.py

  # Watch logs in real-time
  tail -f logs/active_15m_run.log | grep -E "NAV|Loop|Health|Error"

  # Check if orchestrator is running
  pgrep -f MASTER_SYSTEM_ORCHESTRATOR

  # View latest metrics (JSON)
  cat monitoring/dashboard_metrics.json | jq '.'


🚀 START TRADING WITH MONITORING
────────────────────────────────────────────────────────────────────────────

  # ONE COMMAND - Everything automated
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
  ./start_trading_with_monitoring.sh --duration 6 --monitor-interval 10

  # Or manually in separate terminals:

  # Terminal 1: Trading Orchestrator
    rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json
    env TRADING_DURATION_HOURS=6 APPROVE_LIVE_TRADING=YES \
      python 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

  # Terminal 2: Active Monitor
    python -m monitoring.active_capital_monitor \
      --duration 360 --interval 10

  # Terminal 3: Real-Time Dashboard
    python monitoring/real_time_dashboard.py --refresh 30


🧹 CLEANUP & STATE MANAGEMENT
────────────────────────────────────────────────────────────────────────────

  # Clear state for fresh start
  rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json state/nav_cache.json

  # Kill stuck processes
  pkill -f MASTER_SYSTEM_ORCHESTRATOR
  pkill -f active_capital_monitor
  pkill -f real_time_dashboard

  # Kill specific PID
  kill -9 <PID>

  # Force kill everything and clean
  pkill -9 -f 'MASTER_SYSTEM_ORCHESTRATOR|active_capital_monitor|real_time_dashboard' || true
  sleep 2
  rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json


📊 MONITORING COMMANDS
────────────────────────────────────────────────────────────────────────────

  # Active monitor (checks every 10s, applies fixes)
  python -m monitoring.active_capital_monitor --duration 360 --interval 10

  # Active monitor (faster checks - every 5s)
  python -m monitoring.active_capital_monitor --duration 360 --interval 5

  # Real-time dashboard (updates every 30s)
  python monitoring/real_time_dashboard.py --refresh 30

  # Real-time dashboard (updates every 10s - fast)
  python monitoring/real_time_dashboard.py --refresh 10

  # Integrated launcher (coordinates both)
  python launch_with_monitor.py --duration 6 --monitor-interval 10


📈 METRICS & ANALYSIS
────────────────────────────────────────────────────────────────────────────

  # View current metrics (formatted)
  cat monitoring/dashboard_metrics.json | jq '{nav, free, invested, total_return_pct, hourly_return_pct}'

  # Monitor metrics updates in real-time
  watch -n 1 'cat monitoring/dashboard_metrics.json | jq ".nav, .free, .invested"'

  # Extract NAV from metrics every 5 seconds
  while true; do cat monitoring/dashboard_metrics.json | jq '.nav'; sleep 5; done

  # Calculate total return percentage
  cat monitoring/dashboard_metrics.json | jq '.total_return_pct'


🔍 TROUBLESHOOTING COMMANDS
────────────────────────────────────────────────────────────────────────────

  # Show orchestrator logs
  cat /tmp/octivault_orchestrator.log | tail -50

  # Show orchestrator errors only
  grep ERROR /tmp/octivault_orchestrator.log

  # Check balance sync health in logs
  grep "sync_authoritative_balance" logs/active_15m_run.log | tail -20

  # Check for position misalignment
  grep "WalletGuard\|position qty\|wallet qty" logs/active_15m_run.log | tail -20

  # Check trade execution
  grep "TRADE:\|order placed\|order rejected" logs/active_15m_run.log | tail -20

  # Show capital growth over time
  grep "NAV:\|Free:\|Invested:" logs/active_15m_run.log | tail -20

  # Count loop iterations
  grep "Loop:" logs/active_15m_run.log | wc -l

  # Show last 30 seconds of logs
  tail -n 100 logs/active_15m_run.log


⚡ PERFORMANCE CHECKS
────────────────────────────────────────────────────────────────────────────

  # CPU usage
  ps aux | grep -E 'MASTER_SYSTEM|active_capital|real_time' | grep -v grep

  # Memory usage
  top -l 1 | grep 'MASTER_SYSTEM_ORCHESTRATOR'

  # Log file size
  ls -lh logs/active_15m_run.log

  # Recent modification time
  stat logs/active_15m_run.log | grep Modify


🎯 AUTO-FIX MONITORING
────────────────────────────────────────────────────────────────────────────

  # Check if auto-fixes are being applied
  grep "auto_fix\|Auto-fix\|Applying fix" logs/active_15m_run.log

  # Monitor health scores changing
  tail -f monitoring/dashboard_metrics.json | jq '.health'

  # Check issue alerts
  grep "Alert\|CRITICAL\|HIGH\|WARNING" logs/active_15m_run.log


📚 DOCUMENTATION
────────────────────────────────────────────────────────────────────────────

  # Read monitoring guide
  cat MONITORING_GUIDE.md

  # Read summary
  cat ACTIVE_MONITORING_SUMMARY.md

  # Read balance sync flow (from earlier session)
  cat /tmp/balance_sync_flow.md


💡 COMMON PATTERNS
────────────────────────────────────────────────────────────────────────────

  # Complete fresh session (recommended)
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader && \
  rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json && \
  ./start_trading_with_monitoring.sh --duration 6

  # Monitor-only session (trading already running)
  python -m monitoring.active_capital_monitor --duration 360 --interval 10

  # Dashboard-only session (just watch metrics)
  python monitoring/real_time_dashboard.py --refresh 30

  # Kill everything and restart
  pkill -9 -f 'MASTER_SYSTEM_ORCHESTRATOR|active_capital_monitor' || true && \
  sleep 2 && \
  cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader && \
  ./start_trading_with_monitoring.sh --duration 6


🎓 UNDERSTANDING THE OUTPUT
────────────────────────────────────────────────────────────────────────────

Dashboard shows:
  💰 NAV              = Total portfolio value
  💰 Free Capital     = Available for new trades
  💰 Invested         = Locked in positions
  📈 Total Return %   = Profit since session start
  📈 Hourly Return %  = Annualized hourly rate
  📉 Max Drawdown %   = Biggest peak-to-trough drop
  🟢 Health Status    = System operational status

Health Scores (0-100):
  🟢 85-100   = Healthy (operating normally)
  🟡 60-84    = Warning (minor issues, auto-fixing)
  🔴 < 60     = Critical (issues detected, fixes applied)


📞 QUICK HELP
────────────────────────────────────────────────────────────────────────────

  # If metrics not updating
  → Check: pgrep -f MASTER_SYSTEM_ORCHESTRATOR
  → Fix: Restart orchestrator with fresh state

  # If health score declining
  → Check: grep ERROR logs/active_15m_run.log
  → Fix: Monitor auto-fix should handle it

  # If no capital growth
  → Check: grep "capital floor\|insufficient" logs/active_15m_run.log
  → Fix: May need manual investigation

  # If stuck or unresponsive
  → Kill: pkill -9 -f MASTER_SYSTEM_ORCHESTRATOR
  → Clean: rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json
  → Restart: ./start_trading_with_monitoring.sh --duration 6

═══════════════════════════════════════════════════════════════════════════════

Created: May 2, 2026 | Status: Production Ready ✅

EOF
