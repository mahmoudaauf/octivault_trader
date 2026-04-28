#!/bin/bash
# Real-time profit tracker - continuously monitor accumulated profits

TARGET=10.0
LOG_FILE="logs/trading_run_20260424T203944Z.log"
CHECK_INTERVAL=10

echo "=========================================="
echo "REAL-TIME PROFIT TRACKER"
echo "=========================================="
echo "Target: $TARGET USDT"
echo "Log: $LOG_FILE"
echo "Update interval: ${CHECK_INTERVAL}s"
echo ""

COUNT=0
while true; do
  COUNT=$((COUNT + 1))
  
  # Get last LOOP_SUMMARY line to extract pnl
  LAST_SUMMARY=$(tail -50 "$LOG_FILE" | grep "LOOP_SUMMARY" | tail -1)
  
  if [ -z "$LAST_SUMMARY" ]; then
    echo "[$COUNT] Waiting for trading data..."
    sleep $CHECK_INTERVAL
    continue
  fi
  
  # Extract pnl value using grep/sed
  PNL=$(echo "$LAST_SUMMARY" | grep -oE "pnl=[0-9\.\-]+" | cut -d= -f2)
  CAPITAL=$(echo "$LAST_SUMMARY" | grep -oE "capital_free=[0-9\.\-]+" | cut -d= -f2)
  LOOP=$(echo "$LAST_SUMMARY" | grep -oE "loop_id=[0-9]+" | cut -d= -f2)
  
  if [ -z "$PNL" ]; then
    PNL="0.00"
  fi
  
  if [ -z "$CAPITAL" ]; then
    CAPITAL="0.00"
  fi
  
  # Calculate progress
  PROGRESS=$(echo "scale=1; $PNL * 100 / $TARGET" | bc)
  
  # Build progress bar (20 chars)
  FILLED=$(echo "scale=0; $PROGRESS / 5" | bc)
  FILLED=${FILLED%.*}  # Remove decimals
  if [ "$FILLED" -lt 0 ]; then FILLED=0; fi
  if [ "$FILLED" -gt 20 ]; then FILLED=20; fi
  
  EMPTY=$((20 - FILLED))
  BAR=$(printf '█%.0s' $(seq 1 $FILLED))$(printf '░%.0s' $(seq 1 $EMPTY))
  
  # Display status
  printf "[$COUNT] PnL: %+6.2f/%6.2f USDT (%5.1f%%) [%s] | Capital: %7.2f | Loop: %d\n" \
    "$PNL" "$TARGET" "$PROGRESS" "$BAR" "$CAPITAL" "$LOOP"
  
  # Check if target reached
  if (( $(echo "$PNL >= $TARGET" | bc -l) )); then
    echo ""
    echo "✅ SUCCESS! Reached $PNL USDT profit (target: $TARGET USDT)"
    echo ""
    
    # Stop orchestrator
    PID=$(cat orchestrator.pid 2>/dev/null)
    if [ -n "$PID" ]; then
      echo "Stopping orchestrator (PID $PID)..."
      kill -TERM "$PID" 2>/dev/null || true
      sleep 2
      kill -9 "$PID" 2>/dev/null || true
      echo "✅ Orchestrator stopped"
    fi
    
    echo "=========================================="
    exit 0
  fi
  
  sleep $CHECK_INTERVAL
done
