#!/bin/bash
#
# LIVE PRODUCTION STARTUP SCRIPT
# Starts the Octivault Trading Bot in production with state recovery enabled
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
echo "║              OCTIVAULT TRADER - LIVE PRODUCTION STARTUP                   ║"
echo "║                                                                            ║"
echo "║                 State Recovery & Auto-Recovery Enabled                     ║"
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

# Step 3: Start production system
echo -e "${YELLOW}🚀 Step 3: Starting Production System${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

python3 PRODUCTION_STARTUP.py

# This line is only reached if the process terminates
echo ""
echo -e "${YELLOW}⚠️  Production system shutdown${NC}"
echo -e "${BLUE}State files saved to:${NC} $(pwd)/state/"
echo ""

# Display final state
echo -e "${YELLOW}📊 Final System State:${NC}"
if [ -f "state/checkpoint.json" ]; then
    echo "Last checkpoint:"
    cat state/checkpoint.json | python3 -m json.tool | sed 's/^/  /'
fi

echo ""
echo -e "${GREEN}Thank you for running Octivault Trader!${NC}"
