#!/bin/bash
# 📊 Quick Commands to Review Your 3-Hour Trading Session

echo "🎯 3-HOUR TRADING SESSION - QUICK REFERENCE"
echo "============================================"
echo ""

# Display key files
echo "📋 KEY DOCUMENTS:"
echo "  1. Final Report: 3HOUR_SESSION_FINAL_REPORT.md"
echo "  2. Setup Guide: README_3HOUR_SESSION.md"
echo "  3. Status Doc:  SESSION_3HOUR_LIVE_STATUS.md"
echo ""

# Check logs
echo "📊 REVIEW LOGS:"
echo "  # See all signals generated:"
echo "  grep -i 'signal' /tmp/octivault_master_orchestrator.log | tail -20"
echo ""
echo "  # Count total trades:"
echo "  grep -c 'EXECUTED\\|FILLED' /tmp/octivault_master_orchestrator.log"
echo ""
echo "  # Check P&L progression:"
echo "  grep 'total_equity' /tmp/octivault_master_orchestrator.log | tail -5"
echo ""
echo "  # Monitor specific pair:"
echo "  grep 'ETHUSDT\\|BTCUSDT' /tmp/octivault_master_orchestrator.log | grep -E 'SELL|BUY' | tail -10"
echo ""

# Statistics
echo "📈 SESSION STATISTICS:"
TOTAL_SIGNALS=$(grep -c "Signal\|signal" /tmp/octivault_master_orchestrator.log 2>/dev/null || echo "N/A")
TOTAL_TRADES=$(grep -c "EXECUTED" /tmp/octivault_master_orchestrator.log 2>/dev/null || echo "0")
REJECTED=$(grep -c "REJECTED\|blocked" /tmp/octivault_master_orchestrator.log 2>/dev/null || echo "0")

echo "  ✅ Total Signals: $TOTAL_SIGNALS"
echo "  ✅ Executed Trades: $TOTAL_TRADES"
echo "  ⚠️  Rejected Trades: $REJECTED"
echo ""

# Performance metrics
echo "💰 PERFORMANCE REVIEW:"
echo "  # Get latest portfolio value:"
echo "  grep 'total_value' /tmp/octivault_master_orchestrator.log | tail -1"
echo ""
echo "  # Get latest P&L:"
echo "  grep 'total_equity' /tmp/octivault_master_orchestrator.log | tail -1"
echo ""
echo "  # Get profit factor:"
echo "  tail -100 /tmp/octivault_master_orchestrator.log | grep 'Profit\\|profit\\|P&L'"
echo ""

# Risk checks
echo "🛡️  RISK MANAGEMENT REVIEW:"
echo "  # Check dynamic edge enforcements:"
echo "  grep 'EDGE' /tmp/octivault_master_orchestrator.log | tail -10"
echo ""
echo "  # Check position limits:"
echo "  grep 'max_pos\\|ONE_POSITION' /tmp/octivault_master_orchestrator.log | tail -5"
echo ""
echo "  # Check capital floor:"
echo "  grep 'FLOOR\\|RESERVE' /tmp/octivault_master_orchestrator.log | tail -5"
echo ""

echo "🎯 NEXT STEPS:"
echo "  1. Review the Final Report: README_3HOUR_SESSION.md"
echo "  2. Check detailed logs: /tmp/octivault_master_orchestrator.log"
echo "  3. Run another 3-hour session:"
echo "     export APPROVE_LIVE_TRADING=YES"
echo "     python3 RUN_3HOUR_SESSION.py --paper"
echo ""
echo "✅ Session Complete! All data preserved in logs."
