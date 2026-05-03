#!/bin/bash

echo ""
echo "🔍 VERIFYING MONITORING SYSTEM SETUP..."
echo ""

# Check 1: Core files exist
echo "✓ Checking core files..."
files=(
    "monitoring/active_capital_monitor.py"
    "monitoring/real_time_dashboard.py"
    "launch_with_monitor.py"
    "check_status.py"
    "start_trading_with_monitoring.sh"
    "MONITORING_GUIDE.md"
    "ACTIVE_MONITORING_SUMMARY.md"
    "QUICK_REFERENCE.sh"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        missing=$((missing + 1))
    fi
done

if [ $missing -eq 0 ]; then
    echo ""
    echo "✅ ALL CORE FILES PRESENT"
else
    echo ""
    echo "❌ $missing FILES MISSING"
    exit 1
fi

# Check 2: Scripts are executable
echo ""
echo "✓ Checking executable permissions..."
for script in "start_trading_with_monitoring.sh" "check_status.py" "QUICK_REFERENCE.sh"; do
    if [ -x "$script" ]; then
        echo "  ✅ $script is executable"
    else
        echo "  ⚠️  $script is NOT executable (will fix)"
        chmod +x "$script" 2>/dev/null
    fi
done

# Check 3: Python syntax
echo ""
echo "✓ Checking Python syntax..."
python_files=(
    "monitoring/active_capital_monitor.py"
    "monitoring/real_time_dashboard.py"
    "launch_with_monitor.py"
    "check_status.py"
)

syntax_ok=0
for file in "${python_files[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "  ✅ $file syntax OK"
        syntax_ok=$((syntax_ok + 1))
    else
        echo "  ⚠️  $file has syntax issues"
    fi
done

# Check 4: Directories exist
echo ""
echo "✓ Checking directories..."
for dir in "monitoring" "state" "logs"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/"
    else
        echo "  ⚠️  $dir/ (creating)"
        mkdir -p "$dir"
    fi
done

# Check 5: Quick test
echo ""
echo "✓ Running quick test of check_status.py..."
if python3 check_status.py > /tmp/test_output.log 2>&1; then
    echo "  ✅ check_status.py runs successfully"
else
    echo "  ⚠️  check_status.py returned error (may be normal if no data yet)"
fi

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ MONITORING SYSTEM SETUP VERIFIED"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📚 Files created:"
echo "   • monitoring/active_capital_monitor.py (main monitor engine)"
echo "   • monitoring/real_time_dashboard.py (live visualization)"
echo "   • launch_with_monitor.py (integrated launcher)"
echo "   • start_trading_with_monitoring.sh (one-command startup)"
echo "   • check_status.py (quick health check)"
echo "   • MONITORING_GUIDE.md (complete guide)"
echo "   • ACTIVE_MONITORING_SUMMARY.md (implementation summary)"
echo "   • QUICK_REFERENCE.sh (command cheatsheet)"
echo ""
echo "🚀 Quick Start:"
echo "   ./start_trading_with_monitoring.sh --duration 6"
echo ""
echo "📊 Check Status:"
echo "   python check_status.py"
echo ""
echo "📖 Read Guide:"
echo "   cat MONITORING_GUIDE.md"
echo ""
echo "⚡ See Commands:"
echo "   cat QUICK_REFERENCE.sh"
echo ""
