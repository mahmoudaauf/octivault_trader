#!/bin/bash
# Real-time growth monitoring

LOG_FILE="/tmp/octivault_growth_mode.log"

while true; do
  clear
  echo "📊 GROWTH MODE MONITOR"
  echo "======================"
  echo ""
  
  # Get latest NAV
  LATEST_NAV=$(grep "BalanceSync.*💰" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '\$[0-9.]+' | head -1 | sed 's/\$//')
  if [ -n "$LATEST_NAV" ]; then
    echo "💰 Current NAV: \$$LATEST_NAV"
  fi
  
  # Get latest trade
  LATEST_TRADE=$(grep "Execution Event" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '"event": "[^"]*' | sed 's/"event": "//')
  if [ -n "$LATEST_TRADE" ]; then
    echo "📈 Last execution: $LATEST_TRADE"
  fi
  
  # Count trades
  TRADE_COUNT=$(grep -c "TRADE_SUBMITTED\|TRADE_FILLED" "$LOG_FILE" 2>/dev/null || echo "0")
  echo "📊 Total trades: $TRADE_COUNT"
  
  # Check for errors
  ERROR_COUNT=$(grep -c "❌\|ERROR\|FAILED" "$LOG_FILE" 2>/dev/null || echo "0")
  if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "⚠️  Errors detected: $ERROR_COUNT"
  fi
  
  echo ""
  echo "Recent activity:"
  grep "BalanceSync.*💰\|TRADE_\|GROWTH" "$LOG_FILE" 2>/dev/null | tail -5 | sed 's/^/  /'
  
  echo ""
  echo "Press Ctrl+C to exit, or wait for update..."
  sleep 5
done
