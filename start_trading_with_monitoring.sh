#!/bin/bash
# === OCTIVAULT FREEZE BANNER ===
# STATUS:    LEGACY
# CANONICAL: main.py
# REASON:    Duplicate of START_TRADING.sh + LAUNCH_MONITOR.sh combo
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# ===============================

#
# 🎯 INTEGRATED TRADING + MONITORING WITH ACTIVE FIXES
#
# This script:
# 1. Clears old state for fresh start
# 2. Starts the trading orchestrator
# 3. Starts active monitoring with auto-fix engine
# 4. Runs real-time dashboard
#
# Usage:
#   ./start_trading_with_monitoring.sh [--duration 6] [--monitor-interval 10]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DURATION_HOURS=6
MONITOR_INTERVAL=10
TRADING_DIR="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION_HOURS="$2"
            shift 2
            ;;
        --monitor-interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Change to trading directory
cd "$TRADING_DIR"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🎯 OCTIVAULT INTEGRATED TRADING + MONITORING SYSTEM${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Step 1: Clean state
echo -e "\n${YELLOW}[1/4] Cleaning state files...${NC}"
rm -f state/checkpoint.json state/active_trades.db state/portfolio_state.json state/nav_cache.json 2>/dev/null || true
echo -e "${GREEN}✅ State cleared${NC}"

# Step 2: Start orchestrator in background
echo -e "\n${YELLOW}[2/4] Starting Trading Orchestrator...${NC}"
export TRADING_DURATION_HOURS=$DURATION_HOURS
export APPROVE_LIVE_TRADING=YES

nohup env TRADING_DURATION_HOURS=$DURATION_HOURS APPROVE_LIVE_TRADING=YES \
    python "🎯_MASTER_SYSTEM_ORCHESTRATOR.py" > /tmp/octivault_orchestrator.log 2>&1 &

ORCHESTRATOR_PID=$!
echo -e "${GREEN}✅ Orchestrator started (PID: $ORCHESTRATOR_PID)${NC}"
echo -e "   Log: /tmp/octivault_orchestrator.log"

# Wait for orchestrator to initialize
echo -e "\n${YELLOW}[3/4] Waiting for orchestrator to initialize...${NC}"
sleep 5
echo -e "${GREEN}✅ Initialization complete${NC}"

# Step 3: Start dashboard in new terminal window (macOS)
echo -e "\n${YELLOW}[4/4] Starting Real-Time Dashboard...${NC}"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - open in new Terminal window
    osascript <<EOF
tell application "Terminal"
    do script "cd '$TRADING_DIR' && python monitoring/real_time_dashboard.py --refresh 30"
end tell
EOF
    echo -e "${GREEN}✅ Dashboard opened in new Terminal window${NC}"
else
    # Linux - try tmux or just print instructions
    echo -e "${YELLOW}⚠️  Please run in another terminal:${NC}"
    echo -e "   cd '$TRADING_DIR' && python monitoring/real_time_dashboard.py --refresh 30"
fi

# Step 4: Run active monitor (this blocks until duration is complete)
echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 SYSTEM READY - STARTING ACTIVE MONITORING${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"

python launch_with_monitor.py \
    --duration $DURATION_HOURS \
    --monitor-interval $MONITOR_INTERVAL \
    --no-trading

# Cleanup
echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
kill $ORCHESTRATOR_PID 2>/dev/null || true
sleep 2

echo -e "\n${GREEN}✅ Session complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
