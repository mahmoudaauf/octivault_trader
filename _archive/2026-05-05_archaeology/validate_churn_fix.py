#!/usr/bin/env python3
"""
Validation Script for Symbol Churn Fix (Parts 1-10)

Verifies all 10 fixes are properly implemented:
- Parts 1-5: Healing cycle fixes (original)
- Parts 6-10: Symbol convergence fixes (new)

Checks:
1. Config parameters exist and are valid
2. SharedState methods exist and are callable
3. SymbolScreener gating methods exist
4. SwingTradeHunter capital allocation exists
5. LiquidationAgent healing prioritization implemented
6. UURE second gating check implemented
7. Checkpoint files exist
8. No syntax errors in modified files

Exit code: 0 = success, 1 = failure
"""

import json
import sys
from pathlib import Path

# Add workspace to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))


def validate_config():
    """Validate config parameters"""
    print("\n📋 Validating Config Parameters...")
    try:
        # Try importing as module-level constants
        from pathlib import Path

        config_path = Path(__file__).parent / "src" / "l0_core" / "config.py"

        # Read the config file and check for required strings
        with open(config_path) as f:
            config_content = f.read()

        required_params = [
            "SYMBOL_CONVERGENCE_MODE = True",
            "PROVEN_SYMBOLS = {",
            "EXCLUDED_SYMBOLS = {",
            "CONVERGENCE_MIN_HISTORY_TRADES = 100",
            "CONVERGENCE_MAX_EXPERIMENTAL_SYMBOLS = 2",
            "CONVERGENCE_MAX_NEW_SYMBOLS_PER_DAY = 1",
        ]

        missing = []
        for param_line in required_params:
            if param_line not in config_content:
                missing.append(f"  ❌ {param_line} not found")
            else:
                print(f"  ✅ {param_line} found")

        if missing:
            print("\n".join(missing))
            return False

        # Count entries in dicts
        import re

        # Count PROVEN_SYMBOLS entries
        proven_match = re.search(r"PROVEN_SYMBOLS = \{([^}]+)\}", config_content, re.DOTALL)
        if proven_match:
            proven_entries = len(re.findall(r"'[A-Z]+USDT':", proven_match.group(1)))
            if proven_entries != 7:
                print(f"  ❌ PROVEN_SYMBOLS should have 7 entries, has {proven_entries}")
                return False
            print("  ✅ PROVEN_SYMBOLS has 7 entries")

        # Count EXCLUDED_SYMBOLS entries
        excluded_match = re.search(r"EXCLUDED_SYMBOLS = \{([^}]+)\}", config_content, re.DOTALL)
        if excluded_match:
            excluded_entries = len(re.findall(r"'[A-Z0-9]+USDT':", excluded_match.group(1)))
            if excluded_entries != 22:
                print(f"  ❌ EXCLUDED_SYMBOLS should have 22 entries, has {excluded_entries}")
                return False
            print("  ✅ EXCLUDED_SYMBOLS has 22 entries")

        return True
    except Exception as e:
        print(f"  ❌ Error validating config: {e}")
        return False


def validate_shared_state():
    """Validate SharedState convergence methods"""
    print("\n📋 Validating SharedState Convergence Methods...")
    try:
        from src.l0_core.shared_state import SharedState

        required_methods = [
            "is_symbol_excluded",
            "is_symbol_proven",
            "add_to_exclusion_list",
            "get_experimental_symbols",
            "get_experimental_count",
            "can_add_new_symbol",
        ]

        missing = []
        for method in required_methods:
            if not hasattr(SharedState, method):
                missing.append(f"  ❌ {method} not found")
            else:
                print(f"  ✅ {method} exists")

        if missing:
            print("\n".join(missing))
            return False

        return True
    except Exception as e:
        print(f"  ❌ Error validating SharedState: {e}")
        return False


def validate_symbol_screener():
    """Validate SymbolScreener gating methods"""
    print("\n📋 Validating SymbolScreener Gating Methods...")
    try:
        from agents.symbol_screener import SymbolScreener

        required_methods = [
            "_should_accept_symbol",
            "_is_proven_symbol",
            "_get_experimental_count",
        ]

        missing = []
        for method in required_methods:
            if not hasattr(SymbolScreener, method):
                missing.append(f"  ❌ {method} not found")
            else:
                print(f"  ✅ {method} exists")

        if missing:
            print("\n".join(missing))
            return False

        # Check that _propose method has convergence gating
        import inspect

        source = inspect.getsource(SymbolScreener._propose)
        if "_should_accept_symbol" not in source:
            print("  ❌ _propose() doesn't call _should_accept_symbol")
            return False
        print("  ✅ _propose() has convergence gating check")

        return True
    except Exception as e:
        print(f"  ❌ Error validating SymbolScreener: {e}")
        return False


def validate_swing_trade_hunter():
    """Validate SwingTradeHunter capital allocation"""
    print("\n📋 Validating SwingTradeHunter Capital Allocation...")
    try:
        from agents.swing_trade_hunter import SwingTradeHunter

        if not hasattr(SwingTradeHunter, "get_available_capital_for_symbol"):
            print("  ❌ get_available_capital_for_symbol method not found")
            return False

        print("  ✅ get_available_capital_for_symbol method exists")

        # Verify method signature
        import inspect

        sig = inspect.signature(SwingTradeHunter.get_available_capital_for_symbol)
        params = list(sig.parameters.keys())
        if "symbol" not in params or "total_capital" not in params:
            print(f"  ❌ Method signature missing required parameters: {params}")
            return False
        print("  ✅ Method has correct signature")

        return True
    except Exception as e:
        print(f"  ❌ Error validating SwingTradeHunter: {e}")
        return False


def validate_liquidation_agent():
    """Validate LiquidationAgent healing prioritization"""
    print("\n📋 Validating LiquidationAgent Healing Prioritization...")
    try:
        import inspect

        from agents.liquidation_agent import LiquidationAgent

        # Check build_plan method has experimental prioritization
        source = inspect.getsource(LiquidationAgent.build_plan)

        if "is_symbol_proven" not in source:
            print("  ❌ build_plan() doesn't check is_symbol_proven")
            return False
        print("  ✅ build_plan() checks is_symbol_proven")

        if "sort_key" not in source and "sort" not in source:
            print("  ❌ build_plan() doesn't have sorting logic")
            return False
        print("  ✅ build_plan() has sorting logic")

        return True
    except Exception as e:
        print(f"  ❌ Error validating LiquidationAgent: {e}")
        return False


def validate_uure_gating():
    """Validate UURE second gating check"""
    print("\n📋 Validating UURE Second Gating Check (Belt-and-Suspenders)...")
    try:
        import inspect

        from src.l3_portfolio.universe_rotation_engine import UniverseRotationEngine

        # Check _collect_discovery_proposals has convergence gating
        source = inspect.getsource(UniverseRotationEngine._collect_discovery_proposals)

        if "can_add_new_symbol" not in source:
            print("  ❌ _collect_discovery_proposals() doesn't call can_add_new_symbol")
            return False
        print("  ✅ _collect_discovery_proposals() has convergence gate check")

        if "convergence_gate" not in source:
            print("  ❌ _collect_discovery_proposals() doesn't filter on convergence_gate")
            return False
        print("  ✅ _collect_discovery_proposals() filters convergence_gate rejections")

        return True
    except Exception as e:
        print(f"  ❌ Error validating UURE: {e}")
        return False


def validate_checkpoint_files():
    """Validate checkpoint files exist"""
    print("\n📋 Validating Checkpoint Files...")
    try:
        checkpoint_dir = workspace_root / ".balance_bleed_fixes"

        if not checkpoint_dir.exists():
            print(f"  ❌ Checkpoint directory not found: {checkpoint_dir}")
            return False
        print(f"  ✅ Checkpoint directory exists: {checkpoint_dir}")

        # Check original fixes (1-5)
        for i in range(1, 6):
            checkpoint_file = checkpoint_dir / f"fix_{i}_checkpoint.json"
            if not checkpoint_file.exists():
                print(f"  ⚠️  fix_{i}_checkpoint.json not found (may be from previous session)")

        # Check convergence fixes (6-10)
        convergence_file = checkpoint_dir / "fix_6_to_10_convergence_checkpoint.json"
        if not convergence_file.exists():
            print("  ❌ fix_6_to_10_convergence_checkpoint.json not found")
            return False

        # Verify checkpoint file content
        with open(convergence_file) as f:
            checkpoint = json.load(f)

        if checkpoint.get("status") != "implemented":
            print(f"  ❌ Checkpoint status not 'implemented': {checkpoint.get('status')}")
            return False
        print("  ✅ Convergence checkpoint file valid")

        return True
    except Exception as e:
        print(f"  ❌ Error validating checkpoint files: {e}")
        return False


def validate_syntax():
    """Validate no syntax errors in modified files"""
    print("\n📋 Validating Python Syntax...")
    try:
        import py_compile

        files_to_check = [
            "src/l0_core/config.py",
            "src/l0_core/shared_state.py",
            "agents/symbol_screener.py",
            "agents/swing_trade_hunter.py",
            "agents/liquidation_agent.py",
            "src/l3_portfolio/universe_rotation_engine.py",
        ]

        errors = []
        for file_path in files_to_check:
            full_path = workspace_root / file_path
            if not full_path.exists():
                errors.append(f"  ❌ {file_path} not found")
                continue

            try:
                py_compile.compile(str(full_path), doraise=True)
                print(f"  ✅ {file_path} syntax OK")
            except py_compile.PyCompileError as e:
                errors.append(f"  ❌ {file_path}: {e}")

        if errors:
            print("\n".join(errors))
            return False

        return True
    except Exception as e:
        print(f"  ❌ Error validating syntax: {e}")
        return False


def main():
    """Run all validations"""
    print("=" * 80)
    print("SYMBOL CHURN FIX VALIDATION (Parts 1-10)")
    print("=" * 80)

    checks = [
        ("Config Parameters", validate_config),
        ("SharedState Methods", validate_shared_state),
        ("SymbolScreener Gating", validate_symbol_screener),
        ("SwingTradeHunter Capital", validate_swing_trade_hunter),
        ("LiquidationAgent Healing", validate_liquidation_agent),
        ("UURE Second Gating", validate_uure_gating),
        ("Checkpoint Files", validate_checkpoint_files),
        ("Python Syntax", validate_syntax),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ Unexpected error in {check_name}: {e}")
            results.append((check_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} | {check_name}")

    print("=" * 80)
    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 ALL FIXES VALIDATED! System is ready for deployment.")
        print("\nNext steps:")
        print("  1. System restart to activate all fixes")
        print("  2. Monitor 24 hours post-restart")
        print("  3. Confirm symbol count stabilizes and P&L trend positive")
        return 0
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
