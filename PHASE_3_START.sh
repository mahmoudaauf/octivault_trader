#!/bin/bash
# 🚀 PHASE 3 STARTUP SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                     🚀 PHASE 3: WIRING IMPLEMENTATION                    ║"
echo "║                                                                           ║"
echo "║  16 methods → 22 components → 5 live engines → 1 complete system        ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Phase 2 completion
echo "✅ Phase 2 Status Check..."
if [ ! -f "core_engine/implementations.py" ]; then
    echo "❌ ERROR: core_engine/implementations.py not found!"
    echo "   Did you complete Phase 2? Run 'PHASE_2_START.sh' first"
    exit 1
fi

if [ ! -f "core_engine/WIRING_EXAMPLES.py" ]; then
    echo "❌ ERROR: core_engine/WIRING_EXAMPLES.py not found!"
    exit 1
fi

echo "✅ implementations.py found (550 lines)"
echo "✅ WIRING_EXAMPLES.py found (400 lines)"
echo ""

# Verify all 5 engine files exist
echo "✅ Verifying all 5 engine files..."
engines=("market_account_engine.py" "situation_engine.py" "decision_engine.py" "safe_execution_engine.py" "operations_engine.py")
for engine in "${engines[@]}"; do
    if [ ! -f "core_engine/$engine" ]; then
        echo "❌ ERROR: core_engine/$engine not found!"
        exit 1
    fi
    echo "   ✅ core_engine/$engine"
done
echo ""

# Verify Phase 3 documentation
echo "✅ Verifying Phase 3 documentation..."
docs=("PHASE_3_QUICK_START.md" "PHASE_3_LAUNCH.md" "PHASE_3_WIRING_CHECKLIST.md")
for doc in "${docs[@]}"; do
    if [ ! -f "$doc" ]; then
        echo "❌ ERROR: $doc not found!"
        exit 1
    fi
    echo "   ✅ $doc"
done
echo ""

# Check git status
echo "✅ Git status check..."
git_status=$(git status --porcelain | wc -l)
if [ "$git_status" -gt 0 ]; then
    echo "⚠️  WARNING: You have uncommitted changes"
    echo "   Consider: git add . && git commit -m '[Phase 2] Final commit before Phase 3'"
    echo ""
fi

# Create Phase 3 branch
echo "🌿 Creating Phase 3 branch..."
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "phase-3/wiring" ]; then
    git checkout -b phase-3/wiring 2>/dev/null || git checkout phase-3/wiring
    echo "   ✅ Branch: phase-3/wiring"
else
    echo "   ✅ Already on branch: phase-3/wiring"
fi
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                        🟢 READY FOR PHASE 3!                             ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 What to do next:"
echo ""
echo "1️⃣  READ THIS FIRST:"
echo "    Open: PHASE_3_QUICK_START.md"
echo "    This has the exact pattern to follow for all 16 methods"
echo ""
echo "2️⃣  FOLLOW THE CHECKLIST:"
echo "    Open: PHASE_3_WIRING_CHECKLIST.md"
echo "    Check off each step as you complete it"
echo ""
echo "3️⃣  USE THE EXAMPLES:"
echo "    Reference: core_engine/WIRING_EXAMPLES.py"
echo "    If you get stuck, copy from here"
echo ""
echo "4️⃣  USE THE SPECS:"
echo "    Reference: PHASE_2_INTEGRATION_GUIDE.md"
echo "    For component interfaces"
echo ""
echo "5️⃣  FOLLOW THE PATTERN:"
echo "    For each method:"
echo "    a) Find the method in the engine file"
echo "    b) Add import at top: from core_engine.implementations import EngineImpl"
echo "    c) Replace body: return await EngineImpl.method_name(self.app_ctx, ...)"
echo "    d) Test: python3 -m py_compile core_engine/engine_engine.py"
echo "    e) Commit with git"
echo ""

echo "⏱️  Timeline: ~2 hours 45 minutes"
echo ""
echo "🎯 Goal: Replace 16 placeholder methods with real implementations"
echo "   ├─ Step 1: MarketAccountEngine (4 methods) - 30 min"
echo "   ├─ Step 2: SituationEngine (4 methods) - 30 min"
echo "   ├─ Step 3: DecisionEngine (3 methods) - 20 min"
echo "   ├─ Step 4: SafeExecutionEngine (3 + FIX #2) - 30 min"
echo "   ├─ Step 5: OperationsEngine (2 methods) - 20 min"
echo "   └─ Final Validation - 30 min"
echo ""
echo "⭐ Special: Step 4 includes FIX #2 idempotent SELL guard!"
echo ""

echo "═══════════════════════════════════════════════════════════════════════════"
echo "💪 YOU'VE GOT THIS!"
echo ""
echo "   Start with: PHASE_3_QUICK_START.md"
echo "   Track with: PHASE_3_WIRING_CHECKLIST.md"
echo ""
echo "🚀 GO TIME!"
echo "═══════════════════════════════════════════════════════════════════════════"
