#!/bin/bash
TARGET=10.0
CHECK_INTERVAL=10
COUNT=0

while true; do
  COUNT=$((COUNT + 1))
  
  # Find latest log file
  LOG_FILE=$(ls -t logs/trading_run_*.log 2>/dev/null | head -1)
  
  if [ -z "$LOG_FILE" ]; then
    echo "[$COUNT] Waiting for log file..."
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Get last LOOP_SUMMARY
  LAST_SUMMARY=$(tail -50 "$LOG_FILE" | grep "LOOP_SUMMARY" | tail -1)
  
  if [ -z "$LAST_SUMMARY" ]; then
    echo "[$COUNT] Waiting for trading data..."
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Extract metrics
  PNL=$(echo "$LAST_SUMMARY" | grep -oE "pnl=[0-9\.\-]+" | cut -d= -f2)
  if [ -z "$PNL" ]; then PNL="0.00"; fi
  
  # Calculate progress
  PROGRESS=$(echo "scale=1; $PNL * 100 / $TARGET" | bc 2>/dev/null || echo "0")
  FILLED=$(echo "scale=0; $PROGRESS / 5" | bc 2>/dev/null || echo "0")
  
  BAR=$(printf '█%.0s' $(seq 1 ${FILLED%.*}))$(printf '░%.0s' $(seq 1 $((20 - ${FILLED%.*}))))
  
  printf "[$COUNT] PnL: %+6.2f USDT (%5.1f%%) [%s]\n" "$PNL" "$PROGRESS" "$BAR"
  
  # Check target
  if (( $(echo "$PNL >= $TARGET" | bc -l 2>/dev/null || echo "0") )); then
    echo ""
    echo "✅ SUCCESS! Reached $PNL USDT profit!"
    break
  fi
  
  sleep $CHECK_INTERVAL
done
