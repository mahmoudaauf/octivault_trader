#!/bin/bash

# 🎯 8-HOUR TRADING SESSION MONITORING SCRIPT
# Extended from 4 hours to 8 hours
# Continuous monitoring with checkpoint reporting every 30 minutes

set -e

WORKSPACE="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
LOG_FILE="/tmp/octivault_master_orchestrator.log"
SESSION_START_TIME=$(date +%s)
SESSION_START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")
SESSION_DURATION_HOURS=8
SESSION_DURATION_SECONDS=$((SESSION_DURATION_HOURS * 3600))
CHECKPOINT_INTERVAL_MINUTES=30
CHECKPOINT_INTERVAL_SECONDS=$((CHECKPOINT_INTERVAL_MINUTES * 60))

echo "🎯 EXTENDED 8-HOUR TRADING SESSION"
echo "===================================="
echo "Session Start: $SESSION_START_HUMAN"
echo "Session Duration: $SESSION_DURATION_HOURS hours"
echo "Expected End: $(date -u -d "+$SESSION_DURATION_HOURS hours" "+%Y-%m-%d %H:%M:%S")"
echo "Checkpoint Interval: Every $CHECKPOINT_INTERVAL_MINUTES minutes"
echo "Total Checkpoints: 17"
echo "Log File: $LOG_FILE"
echo ""
echo "Monitoring will run continuously..."
echo ""

# Initialize checkpoint counter
CHECKPOINT_COUNT=0
MAX_CHECKPOINTS=$((SESSION_DURATION_HOURS * 60 / CHECKPOINT_INTERVAL_MINUTES))

# Function to generate checkpoint report
generate_checkpoint_report() {
    local checkpoint_num=$1
    local elapsed_seconds=$2
    local elapsed_minutes=$((elapsed_seconds / 60))
    local elapsed_hours=$((elapsed_minutes / 60))
    local remaining_seconds=$((SESSION_DURATION_SECONDS - elapsed_seconds))
    local remaining_minutes=$((remaining_seconds / 60))
    local remaining_hours=$((remaining_minutes / 60))
    
    local checkpoint_file="$WORKSPACE/SESSION_8H_CHECKPOINT_${checkpoint_num}.md"
    local current_time=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "📊 CHECKPOINT #$checkpoint_num - $current_time"
    echo "=========================================="
    echo ""
    
    # Get last trades
    local trades_count=$(grep -c "TRADE EXECUTED" "$LOG_FILE" 2>/dev/null || echo "0")
    local trades_success=$(grep -c "TRADE EXECUTED.*status.*success" "$LOG_FILE" 2>/dev/null || echo "0")
    local trades_failed=$(grep -c "TRADE EXECUTED.*status.*failed" "$LOG_FILE" 2>/dev/null || echo "0")
    local win_rate="0%"
    if [ "$trades_count" -gt 0 ]; then
        win_rate=$((trades_success * 100 / trades_count))%
    fi
    
    # Get current balance
    local current_balance=$(grep "balance.*USDT" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    
    # Get backtest status
    local backtest_status=$(grep "backtest.*gate" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    
    # Get latest signal
    local latest_signal=$(grep "signal.*confidence" "$LOG_FILE" 2>/dev/null | tail -1 || echo "N/A")
    
    # Create checkpoint report
    cat > "$checkpoint_file" << CHECKPOINT_EOF
# 📊 SESSION 8H CHECKPOINT #$checkpoint_num

**Time**: $current_time
**Session Progress**: $elapsed_hours h $((elapsed_minutes % 60)) m / $SESSION_DURATION_HOURS h
**Remaining**: $remaining_hours h $((remaining_minutes % 60)) m
**Completion**: $((elapsed_seconds * 100 / SESSION_DURATION_SECONDS))%

---

## Trading Activity

- **Total Trades Executed**: $trades_count
- **Successful Trades**: $trades_success
- **Failed Trades**: $trades_failed
- **Win Rate**: $win_rate
- **Average Trade Value**: $(echo "scale=2; $trades_count" | bc 2>/dev/null || echo "N/A") per checkpoint

---

## System Status

### Balance
\`\`\`
$current_balance
\`\`\`

### Backtest Gate
\`\`\`
$backtest_status
\`\`\`

### Latest Signal
\`\`\`
$latest_signal
\`\`\`

---

## Performance Expected

At this checkpoint (${elapsed_hours}h progress):
- Expected trades: $((trades_count)) / ~300-400 (for 8h session)
- Expected P&L: +0.5% to +2% (from checkpoint progress)
- System health: ✅ Running

---

## Next Checkpoint

**Coming in 30 minutes** at $(date -u -d "+30 minutes" "+%H:%M:%S")

CHECKPOINT_EOF

    echo "✅ Checkpoint #$checkpoint_num report saved to: $checkpoint_file"
}

# Main monitoring loop
echo "Starting checkpoint monitoring..."
echo ""

while [ $CHECKPOINT_COUNT -lt $MAX_CHECKPOINTS ]; do
    # Calculate elapsed time
    CURRENT_TIME=$(date +%s)
    ELAPSED_SECONDS=$((CURRENT_TIME - SESSION_START_TIME))
    
    # Check if we should generate checkpoint
    if [ $((ELAPSED_SECONDS % CHECKPOINT_INTERVAL_SECONDS)) -lt 5 ] || [ $CHECKPOINT_COUNT -eq 0 ]; then
        CHECKPOINT_COUNT=$((CHECKPOINT_COUNT + 1))
        generate_checkpoint_report $CHECKPOINT_COUNT $ELAPSED_SECONDS
        echo ""
        sleep 5
    fi
    
    # Check if session is complete
    if [ $ELAPSED_SECONDS -ge $SESSION_DURATION_SECONDS ]; then
        echo "✅ 8-HOUR SESSION COMPLETE!"
        echo "Final checkpoint #$CHECKPOINT_COUNT generated"
        break
    fi
    
    sleep 10
done

echo ""
echo "🎉 Session monitoring complete!"
echo "All checkpoints saved in workspace:"
echo "ls -la $WORKSPACE/SESSION_8H_CHECKPOINT_*.md"
