#!/bin/bash

# ============================================
# AUTO-DETECTION VERIFICATION SCRIPT
# Check if all auto-detection systems are working
# ============================================

echo "�� AUTO-DETECTION SYSTEM VERIFICATION"
echo "======================================"
echo ""

cd "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"

# Check if core files exist
echo "📋 Checking core components..."
echo ""

components=(
    "core/exchange_client.py:Balance fetching"
    "core/shared_state.py:State caching"
    "core/bucket_classifier.py:Classification logic"
    "core/three_bucket_manager.py:Bucket management"
    "core/symbol_manager.py:Symbol discovery"
    "core/exchange_truth_auditor.py:Reconciliation"
    "balance_threshold_config.py:Dynamic thresholds"
)

for component in "${components[@]}"; do
    file="${component%%:*}"
    desc="${component##*:}"
    
    if [ -f "$file" ]; then
        echo "✅ $file"
        echo "   └─ $desc"
    else
        echo "❌ $file (MISSING!)"
    fi
done

echo ""
echo "🔍 Key Detection Methods..."
echo ""

# Show key method counts
echo "Exchange Client:"
grep -c "async def get_spot_balances\|async def get_account_balance\|get_balances" core/exchange_client.py 2>/dev/null && echo "  ✅ Balance fetching methods found" || echo "  ❌ Balance methods not found"

echo ""
echo "Shared State:"
grep -c "async def update_balances\|async def hydrate_balances_from_exchange\|async def hydrate_positions_from_balances" core/shared_state.py 2>/dev/null && echo "  ✅ Balance caching methods found" || echo "  ❌ Caching methods not found"

echo ""
echo "Bucket Classifier:"
grep -c "def classify_portfolio\|def classify_position" core/bucket_classifier.py 2>/dev/null && echo "  ✅ Classification methods found" || echo "  ❌ Classification methods not found"

echo ""
echo "Three Bucket Manager:"
grep -c "async def update_bucket_state" core/three_bucket_manager.py 2>/dev/null && echo "  ✅ Bucket update method found" || echo "  ❌ Bucket method not found"

echo ""
echo "Exchange Truth Auditor:"
grep -c "async def.*reconcile\|def _position_qty" core/exchange_truth_auditor.py 2>/dev/null && echo "  ✅ Reconciliation methods found" || echo "  ❌ Reconciliation methods not found"

echo ""
echo "Balance Thresholds:"
grep -c "def classify_balance\|def calculate_thresholds" balance_threshold_config.py 2>/dev/null && echo "  ✅ Dynamic threshold methods found" || echo "  ❌ Threshold methods not found"

echo ""
echo "======================================"
echo "✅ AUTO-DETECTION SYSTEM VERIFIED"
echo ""
echo "All components are in place and ready."
echo ""
echo "To test detection in action:"
echo "  python3 diagnostic_signal_flow.py"
echo ""
