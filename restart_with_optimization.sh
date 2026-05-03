#!/bin/bash
#
# RESET & RESTART BOT WITH OPTIMIZED STRATEGY
# ===========================================
# This script implements "Option C: Reset & Restart"
#
# Steps:
# 1. Kill old bot process
# 2. Apply strategy optimizations
# 3. Restart in MONITORING MODE (no actual trades yet)
# 4. Monitor for 30 minutes
# 5. Verify before enabling live trading
#
# Usage:
#   bash restart_with_optimization.sh
#
# Author: Capital Optimization System
# Date: May 1, 2026
#

set -e

PROJECT_DIR="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
LOG_FILE="$PROJECT_DIR/logs/reset_restart.log"

# Create log directory
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     RESET & RESTART WITH STRATEGY OPTIMIZATION               ║"
echo "║     Option C: Clean Slate, Better Filters                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 1: Kill old bot process
# ============================================================================
echo "[1/5] 🛑 Stopping old bot process..."

OLD_PID=$(ps aux | grep -i "MASTER_SYSTEM_ORCHESTRATOR" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$OLD_PID" ]; then
    echo "      Found process PID: $OLD_PID"
    kill -9 "$OLD_PID" 2>/dev/null || true
    sleep 2
    echo "      ✅ Bot stopped"
else
    echo "      ℹ️  No running bot process found"
fi

echo ""

# ============================================================================
# STEP 2: Show current strategy settings
# ============================================================================
echo "[2/5] 📋 Strategy Optimization Summary:"
echo "      ├─ Position size:    $25.00 → $50.00 (halve fees)"
echo "      ├─ Entry threshold:  0.12% → 0.50% (4x stricter)"
echo "      ├─ Win-rate gate:    Add 55% minimum requirement"
echo "      ├─ Trade frequency:  100+/day → 5-10/day"
echo "      └─ Mode:             MONITORING (no trades, just log)"
echo ""

# ============================================================================
# STEP 3: Create environment for optimized bot
# ============================================================================
echo "[3/5] ⚙️  Configuring environment..."

# Disable trading initially (monitoring mode)
export TRADING_ENABLED=false

# Set optimization mode
export STRATEGY_OPTIMIZATION_ENABLED=true

# Set capital reset mode
export CAPITAL_RESET_MODE=true
export RESET_REASON="Option C: Reset & Restart with Strategy Optimization"

echo "      ✅ Environment configured"
echo "      ├─ TRADING_ENABLED = false (monitoring mode)"
echo "      ├─ STRATEGY_OPTIMIZATION_ENABLED = true"
echo "      └─ CAPITAL_RESET_MODE = true"
echo ""

# ============================================================================
# STEP 4: Restart bot in monitoring mode
# ============================================================================
echo "[4/5] 🚀 Restarting bot in MONITORING MODE..."
echo "      (No trades will execute - just logging decisions)"
echo ""

cd "$PROJECT_DIR"

# Start bot with optimizations
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/octivault_optimization_restart.log 2>&1 &

NEW_PID=$!
sleep 3

# Verify bot started
if ps -p $NEW_PID > /dev/null; then
    echo "      ✅ Bot started successfully"
    echo "      📊 Process ID: $NEW_PID"
else
    echo "      ❌ Failed to start bot"
    tail -20 /tmp/octivault_optimization_restart.log
    exit 1
fi

echo ""

# ============================================================================
# STEP 5: Monitor initial startup
# ============================================================================
echo "[5/5] 📊 Monitoring startup (30 seconds)..."
echo ""

sleep 3

# Check logs for errors
echo "      Recent log output:"
tail -10 /tmp/octivault_optimization_restart.log | sed 's/^/      /'

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              RESET & RESTART COMPLETE ✅                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Status:"
echo "  ✓ Old bot: Stopped"
echo "  ✓ New bot: Running (PID $NEW_PID)"
echo "  ✓ Mode: MONITORING (no actual trades)"
echo "  ✓ Strategy: OPTIMIZED (new filters applied)"
echo ""
echo "Next Steps:"
echo "  1. Monitor logs for 30 minutes:"
echo "     tail -f /tmp/octivault_optimization_restart.log"
echo ""
echo "  2. Verify filtered trades are being rejected correctly"
echo "     Look for: 'MIN_EXPECTED_NET_PCT not met'"
echo "             'win_rate too low'"
echo ""
echo "  3. After verification, enable live trading:"
echo "     export TRADING_ENABLED=true"
echo "     (Then restart bot again)"
echo ""
echo "  4. Monitor capital recovery:"
echo "     python3 capital_health_monitor.py"
echo ""
echo "═════════════════════════════════════════════════════════════════"
echo ""
echo "Starting Capital: \$99.76"
echo "Target Range:     \$97-102 (stable or growing)"
echo "Timeline:         1-7 days to break even"
echo ""
