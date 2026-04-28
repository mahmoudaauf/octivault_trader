#!/bin/bash

# 🎯 4-HOUR TRADING SESSION MONITORING SCRIPT
# Extended from 2 hours to 4 hours (20:34 to 00:34)
# Continuous monitoring with checkpoint reporting every 30 minutes

set -e

WORKSPACE="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
LOG_FILE="/tmp/octivault_master_orchestrator.log"
SESSION_START_TIME=$(date +%s)
SESSION_START_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")
SESSION_DURATION_HOURS=4
SESSION_DURATION_SECONDS=$((SESSION_DURATION_HOURS * 3600))
CHECKPOINT_INTERVAL_MINUTES=30
CHECKPOINT_INTERVAL_SECONDS=$((CHECKPOINT_INTERVAL_MINUTES * 60))

echo "🎯 EXTENDED 4-HOUR TRADING SESSION"
echo "=================================="
echo "Session Start: $SESSION_START_HUMAN"
echo "Session Duration: $SESSION_DURATION_HOURS hours"
echo "Expected End: $(date -u -d "+$SESSION_DURATION_HOURS hours" "+%Y-%m-%d %H:%M:%S")"
echo "Checkpoint Interval: Every $CHECKPOINT_INTERVAL_MINUTES minutes"
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
    
    local checkpoint_file="$WORKSPACE/SESSION_4H_CHECKPOINT_${checkpoint_num}.md"
    local current_time=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "📊 CHECKPOINT #$checkpoint_num - $current_time"
    echo "=========================================="
    echo ""
    
    # Get last trades
    local trades=$(tail -300 "$LOG_FILE" 2>/dev/null | grep -E "TRADE EXECUTED|POSITION_OPENED|POSITION_CLOSED" | tail -10)
    
    # Get current balance
    local balance=$(tail -100 "$LOG_FILE" 2>/dev/null | grep "balance\|NAV" | tail -1)
    
    # Get rejection counts
    local win_rate_rejections=$(grep "MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD" "$LOG_FILE" 2>/dev/null | wc -l)
    local capital_rejections=$(grep "NET_USDT_BELOW_THRESHOLD" "$LOG_FILE" 2>/dev/null | wc -l)
    
    # Get active symbols
    local active_symbols=$(tail -200 "$LOG_FILE" 2>/dev/null | grep -E "confidence.*\[0-9\]" | sed 's/.*symbol=//' | sed 's/ .*//' | sort | uniq | head -15)
    
    cat > "$checkpoint_file" << EOF
# 🎯 4-HOUR SESSION CHECKPOINT #$checkpoint_num

**Checkpoint Time**: $current_time  
**Session Elapsed**: ${elapsed_hours}h ${elapsed_minutes}m ${elapsed_seconds}s  
**Session Remaining**: ${remaining_hours}h ${remaining_minutes}m ${remaining_seconds}s  
**Total Progress**: $((elapsed_seconds * 100 / SESSION_DURATION_SECONDS))%

---

## Session Overview

| Metric | Value |
|--------|-------|
| **Start Time** | $SESSION_START_HUMAN |
| **Current Time** | $current_time |
| **Total Duration** | $SESSION_DURATION_HOURS hours |
| **Elapsed** | ${elapsed_hours}h ${elapsed_minutes}m |
| **Remaining** | ${remaining_hours}h ${remaining_minutes}m |
| **Progress** | $((elapsed_seconds * 100 / SESSION_DURATION_SECONDS))% |

---

## System Status

### Performance Metrics
\`\`\`
Win-Rate Rejections: $win_rate_rejections (backtests building)
Capital Rejections: $capital_rejections (gates checking)
Active Symbols Being Tested: $(echo "$active_symbols" | wc -w)
\`\`\`

### Current Balance
\`\`\`
$balance
\`\`\`

### Active Symbols
\`\`\`
$active_symbols
\`\`\`

### Recent Trades
\`\`\`
$trades
\`\`\`

---

## Next Checkpoint

**Next Checkpoint**: In $CHECKPOINT_INTERVAL_MINUTES minutes  
**Expected Time**: $(date -u -d "+$CHECKPOINT_INTERVAL_SECONDS seconds" "+%Y-%m-%d %H:%M:%S")

---

## Notes

- System is running continuously for 4-hour extended session
- Backtest progress continues to build confidence scores
- Trades expected to increase as gates clear
- Monitoring active and reports updating every 30 minutes

EOF
    
    echo "✅ Checkpoint report saved: $checkpoint_file"
}

# Main monitoring loop
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - SESSION_START_TIME))
    
    # Check if session duration exceeded
    if [ $elapsed -ge $SESSION_DURATION_SECONDS ]; then
        echo ""
        echo "🏁 SESSION COMPLETE!"
        echo "===================="
        echo "Session ended after $SESSION_DURATION_HOURS hours"
        echo "Total elapsed: $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
        
        # Generate final report
        generate_checkpoint_report $MAX_CHECKPOINTS $elapsed
        
        echo ""
        echo "Final checkpoint report saved!"
        break
    fi
    
    # Generate checkpoint reports every CHECKPOINT_INTERVAL_SECONDS
    if [ $((elapsed % CHECKPOINT_INTERVAL_SECONDS)) -lt 5 ] && [ $CHECKPOINT_COUNT -eq 0 ]; then
        CHECKPOINT_COUNT=1
        checkpoint_num=$((elapsed / CHECKPOINT_INTERVAL_SECONDS + 1))
        generate_checkpoint_report $checkpoint_num $elapsed
        sleep 5
        CHECKPOINT_COUNT=0
    elif [ $((elapsed % CHECKPOINT_INTERVAL_SECONDS)) -ge 5 ]; then
        CHECKPOINT_COUNT=0
    fi
    
    # Sleep for 1 minute between checks
    sleep 60
done

echo "🎯 4-HOUR SESSION MONITORING COMPLETE"
echo "====================================="
echo "All checkpoints generated successfully!"
echo ""
echo "To view all checkpoints, run:"
echo "  ls -lah $WORKSPACE/SESSION_4H_CHECKPOINT_*.md"
echo ""
echo "To view latest checkpoint:"
echo "  tail -50 $WORKSPACE/SESSION_4H_CHECKPOINT_*.md | tail -50"
