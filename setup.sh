#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# OctiVault Trading Bot - Infrastructure Setup Script
# Automated setup for macOS/Linux
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║    🚀 OctiVault Trading Bot - Infrastructure Setup 🚀        ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Check Python Version
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 1: Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_PATCH=$(python3 --version 2>&1 | awk '{print $2}')

if (( $(echo "$PYTHON_VERSION < 3.9" | bc -l) )); then
    echo -e "${RED}❌ Python 3.9+ required, found $PYTHON_PATCH${NC}"
    echo "   Install Python 3.9+ and try again."
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_PATCH detected${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Check Working Directory
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 2: Checking working directory...${NC}"

if [ ! -f "pytest.ini" ]; then
    echo -e "${RED}❌ Not in OctiVault project directory${NC}"
    echo "   Please run this script from the project root."
    exit 1
fi

echo -e "${GREEN}✅ Project directory verified${NC}"
echo "   Working directory: $(pwd)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Create Virtual Environment
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 3: Setting up virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "   Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "   Deleted existing venv"
    else
        echo "   Using existing venv"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment ready${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Activate Virtual Environment
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 4: Activating virtual environment...${NC}"

source venv/bin/activate

PYTHON_CHECK=$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null)
if [[ $PYTHON_CHECK == *"venv"* ]]; then
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment might not be active${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Upgrade pip
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 5: Upgrading pip...${NC}"

python3 -m pip install --quiet --upgrade pip setuptools wheel
echo -e "${GREEN}✅ pip upgraded${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Install Dependencies
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 6: Installing dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    echo "   Installing from requirements.txt..."
    pip install --quiet -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo "   requirements.txt not found, installing core packages..."
    pip install --quiet \
        pytest==8.4.2 \
        pytest-asyncio==1.2.0 \
        pytest-cov==7.1.0 \
        pandas==2.3.3 \
        numpy==2.0.2 \
        ccxt==4.5.48 \
        aiohttp==3.13.3 \
        python-dotenv==1.0.0
    echo -e "${GREEN}✅ Core dependencies installed${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Verify Installation
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 7: Verifying installation...${NC}"

python3 << 'VERIFY_EOF'
import sys
errors = []

try:
    import pytest
    print(f"  ✅ pytest {pytest.__version__}")
except ImportError:
    errors.append("pytest")

try:
    import pytest_asyncio
    print("  ✅ pytest_asyncio")
except ImportError:
    errors.append("pytest_asyncio")

try:
    import pandas
    print(f"  ✅ pandas {pandas.__version__}")
except ImportError:
    errors.append("pandas")

try:
    import numpy
    print(f"  ✅ numpy {numpy.__version__}")
except ImportError:
    errors.append("numpy")

try:
    import ccxt
    print(f"  ✅ ccxt {ccxt.__version__}")
except ImportError:
    errors.append("ccxt")

try:
    import aiohttp
    print(f"  ✅ aiohttp {aiohttp.__version__}")
except ImportError:
    errors.append("aiohttp")

if errors:
    print(f"\n❌ Missing packages: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✅ All dependencies verified")
VERIFY_EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Verification failed${NC}"
    exit 1
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Ensure Directories Exist
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 8: Ensuring required directories...${NC}"

for dir in core tests logs data models config scripts; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        mkdir -p "$dir"
        echo "  ✅ $dir/ (created)"
    fi
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Check Configuration
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 9: Checking configuration files...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file found${NC}"
else
    echo -e "${YELLOW}⚠️  .env file not found (optional)${NC}"
fi

if [ -f "pytest.ini" ]; then
    echo -e "${GREEN}✅ pytest.ini configured${NC}"
else
    echo -e "${RED}❌ pytest.ini missing (required)${NC}"
fi

if [ -f "tests/conftest.py" ]; then
    echo -e "${GREEN}✅ tests/conftest.py found${NC}"
else
    echo -e "${YELLOW}⚠️  tests/conftest.py not found (should exist)${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Set PYTHONPATH
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 10: Configuring Python path...${NC}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo -e "${GREEN}✅ PYTHONPATH set${NC}"
echo "   PYTHONPATH=$PYTHONPATH"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: Quick Test
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}📍 Step 11: Running quick test...${NC}"

TESTS_COUNT=$(python3 -m pytest tests/ --collect-only -q 2>/dev/null | grep -E "^<" | wc -l)

if [ $TESTS_COUNT -gt 0 ]; then
    echo -e "${GREEN}✅ Found $TESTS_COUNT tests${NC}"
else
    echo -e "${YELLOW}⚠️  Could not collect tests${NC}"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo -e "║${GREEN}    ✅ Setup Complete!${NC}"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${BLUE}📋 Next Steps:${NC}"
echo ""
echo "1️⃣  To activate the virtual environment in the future:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "2️⃣  Run tests:"
echo -e "   ${YELLOW}python3 -m pytest tests/ -v${NC}"
echo ""
echo "3️⃣  Run phased execution:"
echo -e "   ${YELLOW}bash 🎯_PHASED_RUN.sh${NC}"
echo ""
echo "4️⃣  Or use the convenience script:"
echo -e "   ${YELLOW}./run-local.sh test${NC}"
echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "   See 🔧_INFRASTRUCTURE_SETUP_GUIDE.md for troubleshooting"
echo ""
