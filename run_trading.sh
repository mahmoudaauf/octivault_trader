#!/bin/bash
# Run trading system and diagnostics

cd '/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader'

LOGFILE="trade_execution_$(date +%s).log"

echo "🎯 TRADE EXECUTION DIAGNOSTICS" >> "$LOGFILE"
echo "Start time: $(date)" >> "$LOGFILE"
echo "================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# Export environment
export APPROVE_LIVE_TRADING=YES
export PYTHONUNBUFFERED=1

# Run Python trading system for 120 seconds max
(sleep 120; pkill -f "MASTER_SYSTEM_ORCHESTRATOR") &
KILLER_PID=$!

python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py >> "$LOGFILE" 2>&1

# Kill the killer process if still running
kill $KILLER_PID 2>/dev/null

echo "" >> "$LOGFILE"
echo "================================" >> "$LOGFILE"
echo "End time: $(date)" >> "$LOGFILE"

# Display results
echo "✅ Log saved to: $LOGFILE"
echo ""
echo "📋 Last 100 lines of output:"
tail -100 "$LOGFILE"
