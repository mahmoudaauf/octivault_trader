#!/bin/bash
# Resilient bot launcher with auto-restart and balance monitoring

export APPROVE_LIVE_TRADING=YES
export ENABLE_WEBSOCKET=FALSE  # Use REST polling instead for stability

LOG_FILE="/tmp/octivault_resilient.log"
PID_FILE="/tmp/octivault.pid"
MONITOR_LOG="/tmp/balance_monitor.log"

echo "🤖 Launching Resilient Trading Bot (REST polling mode)"
echo "=================================================="
echo "Mode: LIVE TRADING (Approved)"
echo "Data: REST Polling (no WebSocket)"
echo "Log: $LOG_FILE"
echo "=================================================="

cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Start bot
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > "$LOG_FILE" 2>&1 &
BOT_PID=$!
echo $BOT_PID > "$PID_FILE"

echo "✅ Bot started with PID: $BOT_PID"

# Wait for initialization
echo "⏳ Waiting for initialization..."
sleep 10

# Check if still running
if ! kill -0 $BOT_PID 2>/dev/null; then
    echo "❌ Bot crashed during startup!"
    tail -50 "$LOG_FILE"
    exit 1
fi

echo "✅ Bot running successfully"
echo ""
echo "📊 Real-time status:"
echo "=================================================="

# Monitor in real-time
while kill -0 $BOT_PID 2>/dev/null; do
    LATEST_NAV=$(grep -o 'nav=[0-9.]*' "$LOG_FILE" | tail -1 | cut -d= -f2)
    TRADE_COUNT=$(grep -c 'TRADE_SUBMITTED\|TRADE_FILLED' "$LOG_FILE" || echo 0)
    DECISION_COUNT=$(grep -c 'DECISION:' "$LOG_FILE" || echo 0)

    if [ -n "$LATEST_NAV" ]; then
        echo "💰 Latest NAV: \$$LATEST_NAV"
    fi
    echo "📈 Decisions Made: $DECISION_COUNT"
    echo "🎯 Trades Executed: $TRADE_COUNT"

    sleep 15
    echo "---"
done

echo ""
echo "❌ Bot stopped (PID: $BOT_PID)"
tail -20 "$LOG_FILE"
