#!/bin/zsh
# Monitor Octi AI Trading Bot for trade execution
# Usage: ./monitor_trades.sh [interval_seconds]

INTERVAL=${1:-30}
LOG_FILE="orchestrator_latest.log"

echo "🎯 Octi AI Trading Bot - Trade Execution Monitor"
echo "=================================================="
echo "Monitoring interval: ${INTERVAL}s"
echo "Log file: $LOG_FILE"
echo ""

while true; do
    echo "⏱️  $(date '+%Y-%m-%d %H:%M:%S') - Status Check"
    echo "---"
    
    # Check system is running
    RUNNING=$(ps aux | grep -c "[p]ython.*MASTER_SYSTEM_ORCHESTRATOR")
    if [ $RUNNING -gt 0 ]; then
        echo "✅ System: RUNNING (1 process)"
    else
        echo "❌ System: NOT RUNNING"
    fi
    
    # Check for trade decisions in last N seconds
    TRADES=$(tail -500 "$LOG_FILE" | grep -c "decision=" || echo 0)
    echo "📊 Trade decisions: $TRADES found in recent logs"
    
    # Check for gate evaluations
    GATES=$(tail -500 "$LOG_FILE" | grep -c "_passes_buy_gate\|confidence_floor" || echo 0)
    echo "🔒 Gate evaluations: $GATES"
    
    # Check for errors
    ERRORS=$(tail -500 "$LOG_FILE" | grep -c "ERROR\|REJECT" || echo 0)
    echo "⚠️  Errors/Rejects: $ERRORS"
    
    # Show latest trade decision if any
    LATEST_TRADE=$(tail -500 "$LOG_FILE" | grep "decision=" | tail -1)
    if [ -n "$LATEST_TRADE" ]; then
        echo "🎯 Latest: ${LATEST_TRADE:0:120}"
    fi
    
    # Check confidence floor in recent logs
    CONF_FLOOR=$(tail -200 "$LOG_FILE" | grep "confidence_floor" | tail -1)
    if [ -n "$CONF_FLOOR" ]; then
        echo "📈 Floor: ${CONF_FLOOR:0:120}"
    fi
    
    echo ""
    echo "Waiting ${INTERVAL}s... (Ctrl+C to exit)"
    sleep "$INTERVAL"
done
