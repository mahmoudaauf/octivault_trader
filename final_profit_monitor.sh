#!/bin/bash
# Final comprehensive profit monitor
# Runs until 10 USDT profit target is reached or 4 hours pass

TARGET_PNL=10.0
MAX_TIME=14400  # 4 hours in seconds
CHECK_INTERVAL=20
START_TIME=$(date +%s)
COUNT=0

echo "======================================================"
echo "PROFIT ACCUMULATOR - 10 USDT TARGET"
echo "======================================================"
echo "Start: $(date)"
echo "Target: $TARGET_PNL USDT"
echo ""

while true; do
  COUNT=$((COUNT + 1))
  ELAPSED=$(($(date +%s) - START_TIME))
  
  # Check if time limit reached
  if [ $ELAPSED -gt $MAX_TIME ]; then
    echo "[${COUNT}] Time limit (4h) reached. Stopping."
    break
  fi
  
  # Find latest log
  LOG=$(ls -t logs/trading_run_*.log 2>/dev/null | head -1)
  
  if [ -z "$LOG" ]; then
    echo "[${COUNT}] $(date +%H:%M:%S) - Waiting for log file..."
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Extract latest LOOP_SUMMARY
  SUMMARY=$(grep "LOOP_SUMMARY" "$LOG" 2>/dev/null | tail -1)
  
  if [ -z "$SUMMARY" ]; then
    echo "[${COUNT}] $(date +%H:%M:%S) - Initializing..."
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Parse metrics
  PNL=$(echo "$SUMMARY" | grep -oE "pnl=[0-9\.\-]+" | cut -d= -f2 || echo "0.00")
  CAPITAL=$(echo "$SUMMARY" | grep -oE "capital_free=[0-9\.\-]+" | cut -d= -f2 || echo "0.00")
  LOOP=$(echo "$SUMMARY" | grep -oE "loop_id=[0-9]+" | cut -d= -f2 || echo "0")
  TRADE=$(echo "$SUMMARY" | grep -oE "trade_opened=[a-zA-Z]+" | cut -d= -f2 || echo "False")
  
  # Display
  printf "[${COUNT:0:3}] $(date +%H:%M:%S) | PnL: %+6.2f USDT | Capital: %7.2f | Loop: %4d | Trade: %s | Elapsed: %02d:%02d:%02d\n" \
    "$PNL" "$CAPITAL" "$LOOP" "$TRADE" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
  
  # Check if target reached
  if (( $(echo "$PNL >= $TARGET_PNL" | bc -l 2>/dev/null || echo 0) )); then
    echo ""
    echo "🎉 SUCCESS! Reached $PNL USDT profit!"
    echo "Target: $TARGET_PNL USDT ✅"
    echo "Time taken: $((ELAPSED/60)) minutes"
    break
  fi
  
  sleep $CHECK_INTERVAL
done

echo "======================================================"
echo "End: $(date)"
echo "======================================================"
