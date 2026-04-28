#!/bin/bash
# Test Python and run trading system

cd '/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader'

# Test Python
echo "Testing Python..." > test_result.txt
python3 --version >> test_result.txt 2>&1

# Try to run system
echo "Starting system..." >> test_result.txt
export APPROVE_LIVE_TRADING=YES
timeout 120 python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py >> test_result.txt 2>&1 &

# Wait and check result
sleep 5
tail -100 test_result.txt >> test_result.txt 2>&1
