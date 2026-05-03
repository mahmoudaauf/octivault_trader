#!/bin/bash
# Monitor real-time balance growth from BalanceSync logs

echo "🚀 REAL-TIME BALANCE MONITORING"
echo "================================="
echo ""

log_file="/tmp/octivault_balance_sync.log"

# Get first NAV
first_nav=$(grep "BalanceSync.*💰" "$log_file" | head -1 | grep -oE '\$[0-9.]+' | head -1 | sed 's/\$//')
first_timestamp=$(grep "BalanceSync.*💰" "$log_file" | head -1 | awk '{print $1}')

# Get latest NAV
latest_nav=$(grep "BalanceSync.*💰" "$log_file" | tail -1 | grep -oE '\$[0-9.]+' | head -1 | sed 's/\$//')
latest_timestamp=$(grep "BalanceSync.*💰" "$log_file" | tail -1 | awk '{print $1}')

echo "Starting Balance: \$$first_nav (at $first_timestamp)"
echo "Current Balance:  \$$latest_nav (at $latest_timestamp)"
echo ""

if (( $(echo "$latest_nav > $first_nav" | bc -l) )); then
    delta=$(echo "scale=2; $latest_nav - $first_nav" | bc)
    pct=$(echo "scale=2; ($latest_nav - $first_nav) / $first_nav * 100" | bc)
    echo "✅ GROWING: +\$$delta (+$pct%)"
elif (( $(echo "$latest_nav < $first_nav" | bc -l) )); then
    delta=$(echo "scale=2; $first_nav - $latest_nav" | bc)
    pct=$(echo "scale=2; ($first_nav - $latest_nav) / $first_nav * 100" | bc)
    echo "⚠️  DECAYING: -\$$delta (-$pct%)"
else
    echo "➡️  STABLE: No change"
fi

echo ""
echo "Recent Updates (last 5):"
grep "BalanceSync.*💰" "$log_file" | tail -5 | sed 's/.*NAV updated: /  • /'
