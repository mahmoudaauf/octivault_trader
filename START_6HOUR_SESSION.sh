#!/bin/bash
# 6-Hour Extended Trading Session Launcher
# ========================================
# Starts the trading system and monitors for 6 hours with checkpoints

set -e

BASE_DIR="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
cd "$BASE_DIR"

echo "🎯 STARTING 6-HOUR EXTENDED TRADING SESSION"
echo "==========================================="
echo ""
echo "📅 Date: $(date '+%Y-%m-%d')"
echo "🕐 Time: $(date '+%H:%M:%S')"
echo "⏱️  Duration: 6 hours"
echo "📊 Checkpoints: Every 30 minutes (12 total)"
echo ""

# Clean up old log files
rm -f 6HOUR_SESSION_MONITOR.log 6HOUR_SESSION_REPORT.md

# Export environment variables
export APPROVE_LIVE_TRADING=YES
export TRADING_MODE=live

# Start the 6-hour session in background
echo "🚀 Launching session monitor..."
nohup python3 -u RUN_6HOUR_SESSION.py > 6HOUR_SESSION_MONITOR.log 2>&1 &
SESSION_PID=$!
echo "✅ Session monitor started (PID: $SESSION_PID)"

# Give it a moment to start
sleep 2

# Get the trading system PID from the log
SYSTEM_PID=$(ps aux | grep "MASTER_SYSTEM_ORCHESTRATOR" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$SYSTEM_PID" ]; then
    echo "✅ Trading system running (PID: $SYSTEM_PID)"
else
    echo "⚠️  Trading system not found yet (still starting)"
fi

echo ""
echo "📊 MONITORING COMMANDS:"
echo "======================"
echo ""
echo "Real-time checkpoint updates:"
echo "  tail -f 6HOUR_SESSION_MONITOR.log | grep CHECKPOINT"
echo ""
echo "Full session log:"
echo "  tail -f 6HOUR_SESSION_MONITOR.log"
echo ""
echo "Process status:"
echo "  ps aux | grep -E 'MASTER_SYSTEM|RUN_6HOUR' | grep -v grep"
echo ""
echo "Session info:"
echo "  cat 6HOUR_SESSION_LIVE_STATUS.md"
echo ""
echo "==========================================="
echo "✅ Session started successfully!"
echo "📁 Session PID: $SESSION_PID"
echo "🕐 Session will run until: $(date -j -v+6H '+%Y-%m-%d %H:%M:%S')"
echo "==========================================="
echo ""
