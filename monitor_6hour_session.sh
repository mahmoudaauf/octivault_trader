#!/bin/bash

# ============================================================================
# 6-HOUR SESSION LIVE MONITOR
# ============================================================================
# Monitor the 6-hour trading session in real-time

clear

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║              6-HOUR SESSION LIVE MONITOR                               ║"
echo "║              Monitoring: tail -f /tmp/octivault_live.log               ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "Live Activity Stream:"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""

# Monitor with key metrics highlighted
tail -f /tmp/octivault_live.log | while IFS= read -r line; do
    # Highlight key events with colors
    if echo "$line" | grep -q "LOOP_SUMMARY"; then
        echo -e "\033[1;36m$line\033[0m"  # Cyan - Loop iteration
    elif echo "$line" | grep -q "TRADE_AUDIT"; then
        echo -e "\033[1;32m$line\033[0m"  # Green - Trade executed
    elif echo "$line" | grep -q "NAV="; then
        echo -e "\033[1;33m$line\033[0m"  # Yellow - NAV update
    elif echo "$line" | grep -q "ERROR\|CRITICAL"; then
        echo -e "\033[1;31m$line\033[0m"  # Red - Errors
    elif echo "$line" | grep -q "WARNING"; then
        echo -e "\033[1;35m$line\033[0m"  # Magenta - Warnings
    else
        echo "$line"
    fi
done
