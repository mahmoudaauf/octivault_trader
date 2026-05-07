#!/bin/bash
# Restart the live trading system with balance sync fix

set -e

echo "=========================================="
echo "Stopping old instances..."
echo "=========================================="
pkill -f "python.*main.py" || true
pkill -f "octivault" || true
sleep 3

echo "=========================================="
echo "Starting live trading system..."
echo "=========================================="
cd "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"

# Run with output to both console and log file
python3 main.py 2>&1 | tee -a /tmp/live_trading_restart.log

echo "=========================================="
echo "System started. Monitor logs with:"
echo "tail -f /tmp/live_trading.log"
echo "=========================================="
