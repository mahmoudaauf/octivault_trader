#!/bin/bash

###############################################################################
# 🚀 OCTIVAULT AUTONOMOUS TRADING SYSTEM - STARTUP SCRIPT
###############################################################################
# Start the autonomous trading system with full monitoring and error recovery
# Usage: ./AUTONOMOUS_START.sh [option]
#   --background   Run in background
#   --monitor      Run with real-time monitor
#   --testnet      Run in testnet mode
#   --help         Show help
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
VENV_PATH="${PROJECT_ROOT}/venv"
LOG_DIR="${PROJECT_ROOT}/logs"
PYTHON_BIN="${VENV_PATH}/bin/python3"
PID_FILE="${LOG_DIR}/autonomous.pid"

# Functions
print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  🚀 OCTIVAULT AUTONOMOUS TRADING SYSTEM"
    echo -e "${BLUE}║${NC}  Autonomous Live Trading with Error Recovery"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

check_requirements() {
    echo ""
    print_info "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found"
        exit 1
    fi
    print_success "Python3 found"
    
    # Check venv
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        exit 1
    fi
    print_success "Virtual environment found"
    
    # Check project root
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        print_error ".env file not found"
        exit 1
    fi
    print_success ".env configuration found"
    
    # Create logs directory
    mkdir -p "$LOG_DIR"
    print_success "Logs directory ready: $LOG_DIR"
}

activate_venv() {
    echo ""
    print_info "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
    print_success "Virtual environment activated"
}

verify_dependencies() {
    echo ""
    print_info "Verifying Python dependencies..."
    
    $PYTHON_BIN -c "import pandas; import numpy; import aiohttp; import ccxt; import dotenv" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "All dependencies verified"
    else
        print_warning "Some dependencies may be missing, attempting install..."
        pip install -r "${PROJECT_ROOT}/requirements.txt" -q
    fi
}

start_system() {
    local mode=$1
    
    echo ""
    print_info "Starting autonomous trading system..."
    
    # Export environment variables
    export APPROVE_LIVE_TRADING=YES
    export TRADING_MODE=live
    export PYTHONUNBUFFERED=1
    
    if [ "$mode" = "background" ]; then
        print_info "Running in background mode..."
        nohup $PYTHON_BIN "${PROJECT_ROOT}/AUTONOMOUS_SYSTEM_STARTUP.py" > "${LOG_DIR}/autonomous_system.log" 2>&1 &
        local pid=$!
        echo $pid > "$PID_FILE"
        print_success "System started with PID: $pid"
        print_info "Logs available at: ${LOG_DIR}/autonomous_system.log"
        echo ""
        echo "Monitor with:"
        echo "  tail -f ${LOG_DIR}/autonomous_system.log"
        echo ""
    else
        print_info "Running in foreground mode..."
        $PYTHON_BIN "${PROJECT_ROOT}/AUTONOMOUS_SYSTEM_STARTUP.py"
    fi
}

start_monitor() {
    echo ""
    print_info "Starting real-time monitor..."
    $PYTHON_BIN "${PROJECT_ROOT}/REALTIME_MONITOR.py"
}

stop_system() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null; then
            print_info "Stopping process $pid..."
            kill $pid
            print_success "System stopped"
        fi
        rm -f "$PID_FILE"
    fi
}

show_help() {
    cat << EOF

Usage: $0 [OPTIONS]

OPTIONS:
    --background    Run system in background (recommended for production)
    --monitor       Start with real-time monitor
    --testnet       Run in testnet mode
    --stop          Stop running background process
    --logs          Show live logs from background process
    --status        Check if system is running
    --help          Show this help message

EXAMPLES:
    # Run in background (production)
    $0 --background
    
    # Run in foreground with full output (development)
    $0
    
    # Run with monitor in another terminal
    $0 --monitor
    
    # Check logs
    $0 --logs
    
    # Stop background process
    $0 --stop

ENVIRONMENT:
    The system reads configuration from:
    - .env file in project root
    - Environment variables (override .env)
    
    Key settings:
    - BINANCE_TESTNET: false (for live trading)
    - BINANCE_API_KEY: Your API key
    - TRADING_MODE: live

EOF
}

status_check() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p $pid > /dev/null; then
            print_success "System is running (PID: $pid)"
            return 0
        fi
    fi
    print_warning "System is not running"
    return 1
}

show_logs() {
    if [ -f "${LOG_DIR}/autonomous_system.log" ]; then
        tail -f "${LOG_DIR}/autonomous_system.log"
    else
        print_error "Log file not found"
    fi
}

# Main execution
print_header

# Parse arguments
case "${1:-}" in
    --background)
        check_requirements
        activate_venv
        verify_dependencies
        start_system "background"
        ;;
    --monitor)
        check_requirements
        activate_venv
        verify_dependencies
        start_monitor
        ;;
    --stop)
        stop_system
        ;;
    --logs)
        show_logs
        ;;
    --status)
        check_requirements
        activate_venv
        status_check
        ;;
    --testnet)
        export BINANCE_TESTNET=true
        check_requirements
        activate_venv
        verify_dependencies
        start_system "foreground"
        ;;
    --help)
        show_help
        ;;
    *)
        # Default: run in foreground
        check_requirements
        activate_venv
        verify_dependencies
        start_system "foreground"
        ;;
esac

echo ""
