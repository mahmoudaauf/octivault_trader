#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# OctiVault Trading Bot - Local Execution Script
# Quick commands for common tasks
# ═══════════════════════════════════════════════════════════════════════════

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

setup_environment() {
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null
    fi
    
    # Set PYTHONPATH
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
}

show_status() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          🎯 OctiVault Trading Bot - Local Runner 🎯          ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${BLUE}System Information:${NC}"
    echo "  Python:      $(python3 --version 2>&1 | cut -d' ' -f2)"
    echo "  Location:    $(pwd)"
    echo "  Environment: $([ -d venv ] && echo 'venv' || echo 'system')"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

cmd_setup() {
    echo -e "${YELLOW}🔧 Running infrastructure setup...${NC}"
    echo ""
    
    if [ ! -f "setup.sh" ]; then
        echo "❌ setup.sh not found"
        return 1
    fi
    
    bash setup.sh
    return $?
}

cmd_test() {
    setup_environment
    show_status
    
    echo -e "${BLUE}🧪 Running Full Test Suite${NC}"
    echo "   Tests: $(python3 -m pytest tests/ --collect-only -q 2>/dev/null | grep -E "^<" | wc -l)"
    echo ""
    
    python3 -m pytest tests/ -v --tb=short
    return $?
}

cmd_quick_test() {
    setup_environment
    show_status
    
    echo -e "${BLUE}⚡ Running Quick Test Suite${NC}"
    echo "   Testing core components only..."
    echo ""
    
    python3 -m pytest tests/test_arbitration_engine.py \
                      tests/test_risk_manager.py \
                      tests/test_error_handler.py \
                      tests/test_config_constants.py \
                      -v --tb=short
    return $?
}

cmd_core_test() {
    setup_environment
    show_status
    
    echo -e "${BLUE}✅ Running Core Component Tests${NC}"
    echo "   Testing business logic only (no mocks)..."
    echo ""
    
    python3 -m pytest tests/test_arbitration_engine.py \
                      tests/test_error_types.py \
                      tests/test_config_constants.py \
                      -v --tb=short
    return $?
}

cmd_integration_test() {
    setup_environment
    show_status
    
    echo -e "${BLUE}🔗 Running Integration Tests${NC}"
    echo "   Testing component interactions..."
    echo ""
    
    python3 -m pytest tests/test_bootstrap_manager.py \
                      tests/test_balance_integration.py \
                      tests/test_websocket_integration.py \
                      -v --tb=short 2>&1 | head -200
    return $?
}

cmd_phase_run() {
    setup_environment
    show_status
    
    echo -e "${BLUE}🚀 Starting Phased Execution${NC}"
    echo ""
    
    if [ ! -f "🎯_PHASED_RUN.sh" ]; then
        echo "❌ 🎯_PHASED_RUN.sh not found"
        return 1
    fi
    
    bash 🎯_PHASED_RUN.sh
    return $?
}

cmd_validate() {
    setup_environment
    show_status
    
    echo -e "${BLUE}✔️ Validating System Setup${NC}"
    echo ""
    
    python3 << 'VALIDATE_EOF'
import sys
import os

status = True

# Check Python
print(f"✅ Python {sys.version.split()[0]}")

# Check directories
for d in ['core', 'tests', 'logs', 'data', 'models', 'config', 'scripts']:
    if os.path.isdir(d):
        print(f"✅ Directory: {d}/")
    else:
        print(f"❌ Directory: {d}/ MISSING")
        status = False

# Check files
for f in ['pytest.ini', '.env']:
    if os.path.isfile(f):
        print(f"✅ File: {f}")
    else:
        if f == '.env':
            print(f"⚠️  File: {f} (optional)")
        else:
            print(f"❌ File: {f} MISSING")
            status = False

# Check imports
print("")
print("Dependencies:")
deps = ['pytest', 'pytest_asyncio', 'pandas', 'numpy', 'ccxt', 'aiohttp']
for dep in deps:
    try:
        mod = __import__(dep)
        v = getattr(mod, '__version__', 'installed')
        print(f"✅ {dep}: {v}")
    except ImportError:
        print(f"❌ {dep}: MISSING")
        status = False

print("")
if status:
    print("🟢 System is ready for execution")
    sys.exit(0)
else:
    print("🔴 Please run: ./run-local.sh setup")
    sys.exit(1)
VALIDATE_EOF
    
    return $?
}

cmd_status() {
    setup_environment
    show_status
    
    echo -e "${BLUE}📊 System Status:${NC}"
    echo ""
    
    # Python
    echo -e "  ${YELLOW}Python:${NC}"
    python3 -c "import sys; print(f'    Version: {sys.version.split()[0]}')"
    python3 -c "import sys; print(f'    Executable: {sys.executable}')"
    
    # Virtual Environment
    echo ""
    echo -e "  ${YELLOW}Virtual Environment:${NC}"
    if [ -d "venv" ]; then
        echo "    Status: Active ✅"
        echo "    Location: $(pwd)/venv"
    else
        echo "    Status: Not created (run: ./run-local.sh setup)"
    fi
    
    # Directories
    echo ""
    echo -e "  ${YELLOW}Directories:${NC}"
    for d in core tests logs data models config scripts; do
        [ -d "$d" ] && echo "    ✅ $d/" || echo "    ❌ $d/"
    done
    
    # Tests
    echo ""
    echo -e "  ${YELLOW}Tests:${NC}"
    TESTS=$(python3 -m pytest tests/ --collect-only -q 2>/dev/null | grep -E "^<" | wc -l)
    echo "    Total: $TESTS tests"
    
    echo ""
    return 0
}

cmd_help() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║           OctiVault Trading Bot - Command Reference           ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${BLUE}Available Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}setup${NC}           - Set up virtual environment and install dependencies"
    echo -e "  ${YELLOW}validate${NC}        - Validate system setup and dependencies"
    echo -e "  ${YELLOW}status${NC}          - Show system status"
    echo ""
    echo -e "${BLUE}Testing Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}test${NC}            - Run full test suite"
    echo -e "  ${YELLOW}quick-test${NC}      - Run quick tests (arbitration, risk, error, config)"
    echo -e "  ${YELLOW}core-test${NC}       - Run core component tests (no mocks)"
    echo -e "  ${YELLOW}integration-test${NC} - Run integration tests"
    echo ""
    echo -e "${BLUE}Execution Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}phase-run${NC}       - Start phased execution (Phase 0-9)"
    echo ""
    echo -e "${BLUE}Other Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}help${NC}            - Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo ""
    echo "  # First time setup"
    echo "  ./run-local.sh setup"
    echo ""
    echo "  # Verify everything works"
    echo "  ./run-local.sh validate"
    echo ""
    echo "  # Run quick tests"
    echo "  ./run-local.sh quick-test"
    echo ""
    echo "  # Start phased execution"
    echo "  ./run-local.sh phase-run"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

# Get command (default to help)
COMMAND="${1:-help}"

# Execute command
case "$COMMAND" in
    setup)
        cmd_setup
        ;;
    validate)
        cmd_validate
        ;;
    status)
        cmd_status
        ;;
    test)
        cmd_test
        ;;
    quick-test)
        cmd_quick_test
        ;;
    core-test)
        cmd_core_test
        ;;
    integration-test)
        cmd_integration_test
        ;;
    phase-run)
        cmd_phase_run
        ;;
    help|-h|--help)
        cmd_help
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        echo ""
        cmd_help
        exit 1
        ;;
esac

EXIT_CODE=$?
exit $EXIT_CODE
