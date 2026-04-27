#!/bin/bash
# PHANTOM POSITION FIX - DEPLOYMENT SCRIPT
# Run this to deploy the phantom fix and restart the system

echo "=========================================="
echo "🎯 PHANTOM POSITION FIX - DEPLOYMENT"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[1/5] Verifying implementation...${NC}"
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Verify code is there
if grep -q "_phantom_positions" core/execution_manager.py; then
    echo -e "${GREEN}✅ Phantom code found in execution_manager.py${NC}"
else
    echo -e "${RED}❌ ERROR: Phantom code NOT found${NC}"
    exit 1
fi

# Verify syntax
echo -e "${YELLOW}[2/5] Checking Python syntax...${NC}"
if python3 -m py_compile core/execution_manager.py 2>&1; then
    echo -e "${GREEN}✅ Syntax check passed${NC}"
else
    echo -e "${RED}❌ Syntax error detected${NC}"
    exit 1
fi

echo -e "${YELLOW}[3/5] Stopping current system...${NC}"
pkill -f "MASTER_SYSTEM_ORCHESTRATOR\|phase3_live\|2HOUR_SESSION" 2>/dev/null || true
pkill -f "python.*octivault" 2>/dev/null || true
sleep 3
echo -e "${GREEN}✅ System stopped${NC}"

# Verify no processes
REMAINING=$(ps aux | grep -i "octi\|master\|trader" | grep -v grep | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo -e "${GREEN}✅ No trader processes running${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING: $REMAINING trader processes still running${NC}"
    echo "   Waiting 2 more seconds..."
    sleep 2
fi

echo -e "${YELLOW}[4/5] Starting system with phantom fix...${NC}"
echo "   Command: python 🎯_MASTER_SYSTEM_ORCHESTRATOR.py"
echo ""

nohup python 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > deploy_startup.log 2>&1 &
DEPLOY_PID=$!
echo $DEPLOY_PID > octi_trader.pid
echo -e "${GREEN}✅ System started (PID: $DEPLOY_PID)${NC}"
echo ""

echo -e "${YELLOW}[5/5] Waiting for system to boot...${NC}"
sleep 5

# Show startup output
echo ""
echo "=========================================="
echo "📋 SYSTEM STARTUP OUTPUT (last 20 lines):"
echo "=========================================="
tail -20 deploy_startup.log
echo ""

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "📊 NEXT STEPS:"
echo "1. Monitor the system:"
echo "   tail -f deploy_startup.log"
echo ""
echo "2. Watch for PHANTOM repairs:"
echo "   grep 'PHANTOM_REPAIR' deploy_startup.log"
echo ""
echo "3. Check loop counter (should pass 1195):"
echo "   tail -f deploy_startup.log | grep 'Loop:'"
echo ""
echo "4. Verify success (after 100 loops ~1-2 min):"
echo "   grep -c 'Amount must be positive' deploy_startup.log"
echo "   # Should be 0"
echo ""
echo "📖 DOCUMENTATION:"
echo "   - QUICK_START_PHANTOM_FIX.md (read now!)"
echo "   - PHANTOM_FIX_DEPLOYMENT_GUIDE.md (detailed)"
echo "   - IMPLEMENTATION_STATUS.md (status)"
echo ""
echo "=========================================="
