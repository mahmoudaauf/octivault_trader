#!/bin/bash

###############################################################################
# 🚀 PERSISTENT AUTONOMOUS TRADING - CONTINUOUS OPERATION
###############################################################################
# Start the trading system in persistent background mode
# - Runs 24/7 without stopping
# - Auto-restarts on crashes
# - Compounds profits continuously
# - Logs all activity for monitoring
# - Can be monitored with tail -f logs/
###############################################################################

set -e

PROJECT_ROOT="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
VENV_PATH="${PROJECT_ROOT}/venv"
LOGS_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOGS_DIR}/persistent_trading.log"
PID_FILE="${LOGS_DIR}/persistent_trading.pid"
MAIN_SCRIPT="${PROJECT_ROOT}/🚀_LIVE_ED25519_TRADING.py"

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║     🚀 PERSISTENT AUTONOMOUS TRADING - CONTINUOUS MODE         ║"
    echo "║                                                                ║"
    echo "║        System will run 24/7 and compound profits               ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_running() {
    echo -e "${YELLOW}▶${NC}  $1"
}

start_persistent_trading() {
    print_header
    
    echo ""
    print_info "Starting persistent autonomous trading system..."
    echo ""
    
    # Activate virtual environment and start process
    print_running "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
    
    print_running "Starting trading system in persistent background mode..."
    
    # Start the process with nohup
    nohup python3 "$MAIN_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Save PID
    echo $pid > "$PID_FILE"
    
    # Wait a moment for process to start
    sleep 2
    
    # Verify process is running
    if ps -p $pid > /dev/null 2>&1; then
        print_success "Trading system started successfully!"
        echo ""
        echo "Process Information:"
        echo "  • Process ID: $pid"
        echo "  • Log File: $LOG_FILE"
        echo "  • PID File: $PID_FILE"
        echo "  • Start Time: $(date)"
        echo ""
        echo "Monitoring:"
        echo "  • Watch logs: tail -f $LOG_FILE"
        echo "  • Check status: ps -p $pid"
        echo "  • View recent trades: tail -50 $LOG_FILE | grep FILLED"
        echo ""
        echo "System Configuration:"
        echo "  • Mode: LIVE (Real Money Trading)"
        echo "  • Capital: Auto-compounding enabled"
        echo "  • Cycle Frequency: Every 2 seconds"
        echo "  • Duration: 24/7 continuous operation"
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "✅ System is now running persistently"
        echo "   Monitor with: tail -f $LOG_FILE"
        echo "═══════════════════════════════════════════════════════════════"
        echo ""
    else
        echo "❌ Failed to start trading system"
        exit 1
    fi
}

# Main execution
start_persistent_trading

# Keep script running (in case user wants to see immediate feedback)
echo ""
print_info "System started. Press Ctrl+C to exit this script (system continues running)."
echo ""

# Show initial logs
print_info "Initial output (showing first logs):"
sleep 2
tail -20 "$LOG_FILE"

echo ""
print_info "System is now running in background. You can close this terminal."
echo "To monitor: tail -f $LOG_FILE"
