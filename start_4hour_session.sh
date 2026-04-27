#!/bin/bash
# 4-HOUR LIVE TRADING SESSION WITH CHECKPOINTS
# Starts both trading system and monitoring dashboard

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print headers
print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $1"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Cleanup function
cleanup() {
    echo ""
    print_warning "Received interrupt signal - shutting down gracefully..."
    
    if [ ! -z "$TRADING_PID" ]; then
        print_info "Stopping trading system (PID: $TRADING_PID)..."
        kill $TRADING_PID 2>/dev/null || true
        sleep 2
    fi
    
    if [ ! -z "$MONITOR_PID" ]; then
        print_info "Stopping monitor (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null || true
        sleep 1
    fi
    
    print_success "Session terminated gracefully"
    exit 0
}

# Set trap for Ctrl+C
trap cleanup SIGINT SIGTERM

# Main script
print_header "🚀 4-HOUR LIVE TRADING SESSION WITH CHECKPOINTS"

# Check if verify_deployment.py exists
if [ ! -f "verify_deployment.py" ]; then
    print_error "verify_deployment.py not found!"
    exit 1
fi

# Run verification
print_info "Running deployment verification..."
if ! python3 verify_deployment.py > /dev/null 2>&1; then
    print_error "Deployment verification failed!"
    python3 verify_deployment.py
    exit 1
fi
print_success "Deployment verified (5/5 checks passed)"

# Display session information
echo ""
print_info "Session Configuration:"
print_info "  Duration: 4 hours (240 minutes)"
print_info "  Checkpoint Interval: Every 15 minutes"
print_info "  Expected Checkpoints: 16"
print_info ""

# Check state directory
if [ ! -d "state" ]; then
    print_warning "Creating state directory..."
    mkdir -p state
fi
print_success "State directory ready"

# Clear previous logs if requested
if [ "$1" = "--fresh" ]; then
    print_warning "Clearing previous session data..."
    rm -f state/4hour_session*.json state/4hour_session.log state/checkpoint_*.json
    print_success "Previous session data cleared"
fi

echo ""
print_info "Starting systems in background..."
echo ""

# Start trading system
print_info "Starting trading system..."
python3 run_4hour_session.py > trading_4hour.log 2>&1 &
TRADING_PID=$!
print_success "Trading system started (PID: $TRADING_PID)"

# Give trading system a moment to initialize
sleep 2

# Start monitor dashboard
print_info "Starting monitor dashboard..."
python3 monitor_4hour_session.py &
MONITOR_PID=$!
print_success "Monitor dashboard started (PID: $MONITOR_PID)"

echo ""
print_header "📊 SESSION RUNNING"
echo ""
print_info "Trading System: PID $TRADING_PID"
print_info "Monitor Dashboard: PID $MONITOR_PID"
echo ""
print_info "Monitor will refresh every 5 seconds"
print_info "Checkpoints will be saved every 15 minutes"
print_info "Session will run for 4 hours (240 minutes)"
echo ""
print_warning "Press Ctrl+C to stop session early"
echo ""

# Wait for monitor to complete or user interrupt
wait $MONITOR_PID

# If monitor exits before trading, trading might still be running
if kill -0 $TRADING_PID 2>/dev/null; then
    print_info "Monitor finished, but trading is still running..."
    print_info "Waiting for trading to complete..."
    wait $TRADING_PID
fi

echo ""
print_header "✅ SESSION COMPLETE"
echo ""

# Display final summary
if [ -f "state/4hour_session_final.json" ]; then
    print_success "Final session summary saved"
    echo ""
    print_info "View final summary:"
    echo "   cat state/4hour_session_final.json | python3 -m json.tool"
fi

echo ""
print_info "Session logs and checkpoints saved in state/ directory"
echo ""

print_success "All systems stopped gracefully"
