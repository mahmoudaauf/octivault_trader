#!/bin/bash
# Launch bot in GROWTH MODE with all fixes applied

set -e

REPO_PATH="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
cd "$REPO_PATH"

# Kill any existing instances
pkill -9 -f "MASTER_SYSTEM_ORCHESTRATOR" || true
sleep 2

# Apply growth configuration
export APPROVE_LIVE_TRADING=YES
export BOOTSTRAP_REENTRY_MIN_CONFIDENCE=0.62
export MIN_ENTRY_QUOTE_USDT=5.0
export DEFAULT_PLANNED_QUOTE=10.0
export MAX_PLANNED_QUOTE_USDT=30.0
export MIN_HOLD_SECONDS=30
export PROFIT_LOCK_TARGET_PERCENT=0.01

echo "🚀 Starting bot in GROWTH MODE..."
echo "  • Bootstrap confidence gate: 0.62 (relaxed from 0.68)"
echo "  • Minimum trade size: $5.00 (adaptive)"
echo "  • Planned trade size: $10.00 (from $25)"
echo "  • Maximum trade size: $30.00"
echo "  • Profit lock: 1% (from 1.5%)"
echo ""

nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/octivault_growth_mode.log 2>&1 &
PID=$!

echo "✅ Bot started with PID $PID"
echo "📊 Log: tail -f /tmp/octivault_growth_mode.log"
echo ""
echo "🎯 Watch for:"
echo "   ✅ [BalanceSync] 💰 NAV updated: ... GROWING"
echo "   ✅ Execution Event: TRADE_SUBMITTED"
echo "   ✅ Execution Event: TRADE_FILLED"
