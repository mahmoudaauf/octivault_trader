#!/bin/bash
#
# LIVE TRADING WITH BALANCE MONITORING STARTUP
# Starts Octivault Trading Bot with real-time balance tracking
#

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║          OCTIVAULT TRADER - LIVE TRADING WITH BALANCE MONITORING          ║"
echo "║                                                                            ║"
echo "║     Real-Time Portfolio Performance Tracking & Auto-Recovery Enabled       ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}Current Directory:${NC} $(pwd)"
echo ""

# Step 1: Verify deployment
echo -e "${YELLOW}📋 Step 1: Verifying Deployment...${NC}"
python3 verify_deployment.py
echo ""

# Check if verification passed
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Deployment verification failed!${NC}"
    echo "Please fix the issues above before starting production."
    exit 1
fi

echo -e "${GREEN}✅ Deployment verification passed!${NC}"
echo ""

# Step 2: Display system information
echo -e "${YELLOW}📊 Step 2: System Information${NC}"
echo -e "  Python Version: $(python3 --version)"
echo -e "  State Directory: $(pwd)/state"
echo -e "  State Files:"
ls -lh state/*.json 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
echo ""

# Step 3: Start trading system in background
echo -e "${YELLOW}🚀 Step 3: Starting Live Trading System${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Start trading system
python3 LIVE_TRADING_WITH_BALANCE_MONITOR.py &
TRADING_PID=$!

echo -e "${GREEN}✅ Trading system started (PID: $TRADING_PID)${NC}"
echo ""

# Give trading system time to start
sleep 3

# Step 4: Start balance monitoring dashboard
echo -e "${YELLOW}📊 Step 4: Starting Balance Monitoring Dashboard${NC}"
echo ""

# Start dashboard in a new terminal or in the background
python3 balance_dashboard.py

# Cleanup on exit
trap "kill $TRADING_PID 2>/dev/null || true" EXIT

echo ""
echo -e "${YELLOW}⚠️  Live trading system shutdown${NC}"
echo -e "${BLUE}State files saved to:${NC} $(pwd)/state/"
echo ""

# Display final state
echo -e "${YELLOW}📊 Final System State:${NC}"
if [ -f "state/checkpoint.json" ]; then
    echo "Last checkpoint:"
    cat state/checkpoint.json | python3 -m json.tool | sed 's/^/  /'
fi

echo ""
echo -e "${GREEN}Thank you for running Octivault Trader with Balance Monitoring!${NC}"
