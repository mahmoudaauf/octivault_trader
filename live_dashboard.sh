#!/bin/bash
# Real-time dashboard for live trading system

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    OCTIVAULT LIVE TRADING DASHBOARD                            ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # System status
    echo "SYSTEM STATUS:"
    if pgrep -f "python main.py" > /dev/null; then
        echo "  ✅ Main trading system: RUNNING"
    else
        echo "  ❌ Main trading system: STOPPED"
    fi

    if pgrep -f "checkpoint_monitor.py" > /dev/null; then
        echo "  ✅ Checkpoint monitor: RUNNING"
    else
        echo "  ❌ Checkpoint monitor: STOPPED"
    fi
    echo ""

    # Latest log entries (last 5 minutes)
    echo "RECENT ACTIVITY (last 30 seconds):"
    echo "────────────────────────────────────────────────────────────────────────────────"
    tail -20 monitor_live_trading.log 2>/dev/null | grep -E "(cycle|nav=|READY|COMPLETE|ERROR)" | tail -10
    echo ""

    # Checkpoints achieved
    echo "CHECKPOINTS ACHIEVED:"
    echo "────────────────────────────────────────────────────────────────────────────────"
    if [ -f checkpoints.jsonl ]; then
        cat checkpoints.jsonl | jq -r '.target, .actual, .gain_pct' 2>/dev/null | paste - - - | while read target actual gain; do
            printf "  🎯 %-10s NAV: %-10s Gain: %-8s\n" "$target" "$actual" "$gain"
        done
    else
        echo "  (no checkpoints yet)"
    fi
    echo ""

    # Estimated time to next checkpoint
    current_nav=$(tail -5 monitor_live_trading.log 2>/dev/null | grep "nav=" | tail -1 | sed 's/.*nav=\s*\([0-9.]*\).*/\1/')
    echo "CURRENT NAV: \$$current_nav"
    echo ""

    # Uptime
    uptime_sec=$(($(date +%s) - $(stat -f%Sm -t%s monitor_live_trading.log 2>/dev/null || date +%s)))
    uptime_min=$((uptime_sec / 60))
    echo "UPTIME: ${uptime_min} minutes"
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Press Ctrl+C to exit | Refreshing every 10 seconds..."
    echo ""

    sleep 10
done
