#!/bin/bash
# Real-time profit monitor with continuous tracking

TARGET=10.0
CHECK_INTERVAL=15
MAX_DURATION=$((24 * 3600))  # 24 hours in seconds
START_TIME=$(date +%s)

echo "=========================================="
echo "🚀 CONTINUOUS PROFIT MONITOR"
echo "=========================================="
echo "Target: $TARGET USDT"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "Max Duration: 24 hours"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

count=0
while true; do
  count=$((count + 1))
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
  MINS=$(echo "scale=0; ($ELAPSED % 3600) / 60" | bc)
  SECS=$(echo "scale=0; $ELAPSED % 60" | bc)
  
  # Check if max duration reached
  if [ $ELAPSED -gt $MAX_DURATION ]; then
    echo "[TIMEOUT] 24 hours elapsed. Stopping monitor."
    break
  fi
  
  # Find latest log file
  LOG_FILE=$(ls -t logs/trading_run_*.log 2>/dev/null | head -1)
  
  if [ -z "$LOG_FILE" ]; then
    printf "[%02d:%02d:%02d] Waiting for log file...\n" "$HOURS" "$MINS" "$SECS" 2>/dev/null
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Extract last LOOP_SUMMARY line
  LAST_LINE=$(grep "LOOP_SUMMARY" "$LOG_FILE" 2>/dev/null | tail -1)
  
  if [ -z "$LAST_LINE" ]; then
    printf "[%02d:%02d:%02d] Initializing system...\n" "$HOURS" "$MINS" "$SECS" 2>/dev/null
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Parse metrics
  PNL=$(echo "$LAST_LINE" | grep -oE "pnl=[0-9\.\-]+" | cut -d= -f2 || echo "0.00")
  TRADES=$(echo "$LAST_LINE" | grep -oE "trade_opened=[a-zA-Z]+" | cut -d= -f2 || echo "False")
  LOOP=$(echo "$LAST_LINE" | grep -oE "loop_id=[0-9]+" | cut -d= -f2 || echo "0")
  EXEC=$(echo "$LAST_LINE" | grep -oE "exec_attempted=[a-zA-Z]+" | cut -d= -f2 || echo "False")
  
  [ -z "$PNL" ] && PNL="0.00"
  
  # Calculate progress percentage and bar
  PCTL=$(echo "scale=1; $PNL * 100 / $TARGET" | bc 2>/dev/null || echo "0")
  FILLED=$(echo "scale=0; $PCTL / 5" | bc 2>/dev/null | cut -d. -f1)
  [ "$FILLED" -lt 0 ] && FILLED=0
  [ "$FILLED" -gt 20 ] && FILLED=20
  EMPTY=$((20 - FILLED))
  
  BAR=$(printf '█%.0s' $(seq 1 $FILLED))$(printf '░%.0s' $(seq 1 $EMPTY))
  
  # Display status
  printf "[%02d:%02d:%02d] [%s] PnL: %+7.2f USDT (%5.1f%%) | Loop: %5s | Trade: %s | Exec: %s\n" \
    "$HOURS" "$MINS" "$SECS" "$BAR" "$PNL" "$PCTL" "$LOOP" "$TRADES" "$EXEC" 2>/dev/null
  
  # Check if target reached
  if (( $(echo "$PNL >= $TARGET" | bc -l 2>/dev/null || echo 0) )); then
    echo ""
    echo "✅ ============================================"
    echo "✅ SUCCESS! TARGET REACHED!"
    echo "✅ Final PnL: $PNL USDT"
    echo "✅ Target was: $TARGET USDT"
    echo "✅ Time to target: ${HOURS}h ${MINS}m ${SECS}s"
    echo "✅ ============================================"
    exit 0
  fi
  
  sleep $CHECK_INTERVAL
done
