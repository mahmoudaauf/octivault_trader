#!/bin/bash
# 🎯 Monitor 3-Hour Trading Session
# Real-time profit and reinvestment tracking

WORKSPACE="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
LOG_FILE="/tmp/octivault_3h_monitor.log"
ORCHESTRATOR_LOG="/tmp/octivault_master_orchestrator.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🎯 3-HOUR TRADING SESSION LIVE MONITORING${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}\n"

# Check if process is running
if pgrep -f "RUN_3HOUR_SESSION.py" > /dev/null; then
    echo -e "${GREEN}✅ Session RUNNING${NC}"
else
    echo -e "${RED}❌ Session NOT RUNNING${NC}"
fi

echo -e "\n${BLUE}═══ KEY METRICS ===${NC}\n"

# Extract latest portfolio value
PORTFOLIO_VALUE=$(grep -o "portfolio_value\": [0-9.]*" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | grep -o "[0-9.]*")
echo -e "💰 Portfolio Value: ${GREEN}${PORTFOLIO_VALUE:-N/A} USDT${NC}"

# Extract latest trade info
TRADES=$(grep -o "trades_executed: [0-9]*" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]*")
echo -e "📊 Trades Executed: ${YELLOW}${TRADES:-0}${NC}"

# Extract latest P&L
PNL=$(grep -o "profit_loss\": [0-9.-]*" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | grep -o "[0-9.-]*")
echo -e "📈 Current P&L: ${PNL:-0} USDT"

# Extract latest NAV
NAV=$(grep -i "total_equity.*:" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | grep -o "[0-9.]*$")
echo -e "🎯 Current NAV: ${GREEN}${NAV:-N/A}${NC}"

echo -e "\n${BLUE}═══ ACTIVE POSITIONS ===${NC}\n"

# Show position summary
grep -i "debug:classify" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -5 | while read line; do
    SYMBOL=$(echo "$line" | grep -o "[A-Z]*USDT" | head -1)
    VALUE=$(echo "$line" | grep -o "value=[0-9.]*" | grep -o "[0-9.]*")
    QTY=$(echo "$line" | grep -o "qty=[0-9.]*" | grep -o "[0-9.]*")
    
    if [ ! -z "$VALUE" ]; then
        printf "  📍 %-10s Qty=%.8f Value=\$%-8s\n" "$SYMBOL" "$QTY" "$VALUE"
    fi
done

echo -e "\n${BLUE}═══ RECENT ACTIVITY ===${NC}\n"

# Show latest trade attempts
echo -e "${CYAN}Latest Trade Signals:${NC}"
grep -i "signal.*sell\|signal.*buy" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -3 | while read line; do
    echo "  • $(echo $line | sed 's/.*\[\(.*\)\]/\1/')"
done

echo -e "\n${CYAN}Recent Execution Events:${NC}"
grep -i "exec\|trade_rejected\|blocked" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -3 | while read line; do
    if echo "$line" | grep -q "REJECTED\|blocked"; then
        echo -e "  ${RED}✗ $(echo $line | sed 's/.*\[\(.*\)\]/\1/')${NC}"
    else
        echo -e "  ${GREEN}✓ $(echo $line | sed 's/.*\[\(.*\)\]/\1/')${NC}"
    fi
done

echo -e "\n${BLUE}═══ SYSTEM HEALTH ===${NC}\n"

# Check component status
if grep -q "HEALTHY\|Running" "$ORCHESTRATOR_LOG" 2>/dev/null; then
    echo -e "${GREEN}✅ All Components: Healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Components: Check logs${NC}"
fi

# Show elapsed time
START_TIME=$(grep -m1 "Session.*started" "$LOG_FILE" 2>/dev/null | head -1)
if [ ! -z "$START_TIME" ]; then
    echo -e "⏱️  $START_TIME"
fi

echo -e "\n${BLUE}═══ COMMAND TIPS ===${NC}\n"

echo -e "Real-time log tail:"
echo -e "  ${CYAN}tail -f /tmp/octivault_master_orchestrator.log${NC}\n"

echo -e "Search for trades:"
echo -e "  ${CYAN}grep -i 'executed\|filled\|trade' /tmp/octivault_master_orchestrator.log${NC}\n"

echo -e "Monitor every 10 seconds:"
echo -e "  ${CYAN}watch -n 10 'tail -20 /tmp/octivault_master_orchestrator.log'${NC}\n"

echo -e "Check session status:"
echo -e "  ${CYAN}pgrep -a 'RUN_3HOUR_SESSION'${NC}\n"

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "Session will run until completion. Press Ctrl+C to stop monitoring.${NC}\n"

# Keep monitoring
echo -e "${YELLOW}Live monitoring (updates every 10 seconds):${NC}\n"

while true; do
    clear
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] 3-HOUR TRADING SESSION MONITOR${NC}\n"
    
    # Portfolio value
    PORTFOLIO=$(grep -o '"total_value": [0-9.]*' "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | grep -o "[0-9.]*")
    if [ ! -z "$PORTFOLIO" ]; then
        echo -e "${GREEN}Portfolio Value: \$$PORTFOLIO${NC}"
    fi
    
    # Latest signals
    echo -e "\n${BLUE}Recent Signals (last 5):${NC}"
    grep -i "received.*signal\|submitted.*signal" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -5 | nl -w2 -s'. '
    
    # Latest P&L
    echo -e "\n${BLUE}Latest P&L Update:${NC}"
    grep "total_equity" "$ORCHESTRATOR_LOG" 2>/dev/null | tail -1 | sed 's/^/  /'
    
    # Status
    echo -e "\n${BLUE}Process Status:${NC}"
    if pgrep -f "RUN_3HOUR_SESSION.py" > /dev/null; then
        echo -e "  ${GREEN}✅ Running${NC}"
    else
        echo -e "  ${RED}❌ Stopped${NC}"
    fi
    
    sleep 10
done
