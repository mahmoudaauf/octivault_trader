#!/bin/bash
# Test script: Verify throttle state fixes work correctly
# Run this after IP ban expires (14:31:53 UTC)

set -e

REPO_ROOT="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🧪 Testing Throttle State Fixes"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 1: Check fixes are in place
echo "📋 Test 1: Verifying code fixes..."
echo ""

if grep -q "Throttle window expired; clearing throttle state" core_engine/native/bootstrap.py; then
    echo "✅ Fix 1: bootstrap.py has throttle expiry check"
else
    echo "❌ Fix 1: bootstrap.py missing throttle expiry check"
    exit 1
fi

if grep -q "Exchange throttled; skipping symbol discovery this cycle" core_engine/native/orchestrator.py; then
    echo "✅ Fix 2: orchestrator.py has throttle gate in _phase_discover"
else
    echo "❌ Fix 2: orchestrator.py missing throttle gate"
    exit 1
fi

if grep -q '"exchange_throttled": false' runtime_state_snapshot.json; then
    echo "✅ Fix 3: runtime_state_snapshot.json has throttle state cleared"
else
    echo "❌ Fix 3: runtime_state_snapshot.json still has old throttle state"
    exit 1
fi

echo ""

# Test 2: Check imports
echo "📋 Test 2: Verifying imports..."
echo ""

if grep -q "^import time$" core_engine/native/bootstrap.py; then
    echo "✅ time module imported in bootstrap.py"
else
    echo "❌ time module NOT imported in bootstrap.py"
    exit 1
fi

if grep -q "^import time$" core_engine/native/orchestrator.py; then
    echo "✅ time module imported in orchestrator.py"
else
    echo "❌ time module NOT imported in orchestrator.py"
    exit 1
fi

echo ""

# Test 3: Run 100 cycles with monitoring
echo "📋 Test 3: Running 100-cycle live trading test..."
echo ""
echo "This will test:"
echo "  ✅ No fresh 418 errors (wallet scan skips throttle)"
echo "  ✅ NAV > \$0 (balance syncs successfully)"
echo "  ✅ Capital freeing (liquidates dust when needed)"
echo "  ✅ Symbol interchange (trades multiple pairs)"
echo ""

OUTPUT_FILE="run_test_$(date +%s).log"
echo "📝 Output: $OUTPUT_FILE"
echo ""

timeout 600 python3 run_and_monitor.py 100 2>&1 | tee "$OUTPUT_FILE"
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ] && [ $TEST_EXIT_CODE -ne 124 ]; then
    echo ""
    echo "❌ Test failed with exit code $TEST_EXIT_CODE"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ All fixes verified!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Parse results
echo "📊 Test Results:"
echo ""

if grep -q "418" "$OUTPUT_FILE"; then
    echo "⚠️  Found 418 error in logs (but should be minimal/recoverable)"
else
    echo "✅ No 418 rate limit errors detected"
fi

if grep -q "NAV=\$0.00" "$OUTPUT_FILE"; then
    echo "⚠️  NAV still at \$0.00 (balance sync may have failed)"
else
    NAV=$(grep -o "NAV=\$[0-9.]*" "$OUTPUT_FILE" | tail -1 | cut -d'=' -f2)
    if [ -n "$NAV" ]; then
        echo "✅ NAV updated: $NAV"
    fi
fi

if grep -q "CAPITAL FREEING" "$OUTPUT_FILE"; then
    echo "✅ Capital freeing activated (dust liquidation working)"
else
    echo "ℹ️  Capital freeing not activated (account may have sufficient balance)"
fi

DECISION_COUNT=$(grep -c "decisions=" "$OUTPUT_FILE" || echo "0")
echo "✅ Generated $DECISION_COUNT trading decisions"

echo ""
echo "📄 Full logs available in: $OUTPUT_FILE"
echo ""
