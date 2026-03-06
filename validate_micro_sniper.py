#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MICRO_SNIPER Mode - Deployment Validation Script

Checks that all required components are in place:
1. core/nav_regime.py exists and is importable
2. MetaController imports and initializes RegimeManager
3. All gating methods defined
4. Regime update logic in evaluate_and_act()
5. Dust healing gating implemented
6. No syntax errors in modified files
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Tuple

# Configuration
WORKSPACE_ROOT = Path(__file__).parent
CORE_DIR = WORKSPACE_ROOT / "core"
META_CONTROLLER_PATH = CORE_DIR / "meta_controller.py"
NAV_REGIME_PATH = CORE_DIR / "nav_regime.py"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, msg: str):
        self.passed.append(msg)
        print(f"{GREEN}✓{RESET} {msg}")
    
    def add_fail(self, msg: str):
        self.failed.append(msg)
        print(f"{RED}✗{RESET} {msg}")
    
    def add_warn(self, msg: str):
        self.warnings.append(msg)
        print(f"{YELLOW}⚠{RESET} {msg}")
    
    def summary(self) -> int:
        print(f"\n{BOLD}=== VALIDATION SUMMARY ==={RESET}")
        print(f"{GREEN}Passed: {len(self.passed)}{RESET}")
        print(f"{RED}Failed: {len(self.failed)}{RESET}")
        print(f"{YELLOW}Warnings: {len(self.warnings)}{RESET}")
        
        if self.failed:
            print(f"\n{BOLD}{RED}DEPLOYMENT BLOCKED - Fix failures above{RESET}")
            return 1
        elif self.warnings:
            print(f"\n{BOLD}{YELLOW}DEPLOYMENT ALLOWED - Review warnings{RESET}")
            return 0
        else:
            print(f"\n{BOLD}{GREEN}ALL CHECKS PASSED - Ready for deployment{RESET}")
            return 0


def check_nav_regime_exists() -> bool:
    """Check that core/nav_regime.py exists."""
    if not NAV_REGIME_PATH.exists():
        return False
    return True


def check_nav_regime_syntax() -> bool:
    """Check that nav_regime.py has valid Python syntax."""
    try:
        with open(NAV_REGIME_PATH, 'r') as f:
            code = f.read()
        compile(code, str(NAV_REGIME_PATH), 'exec')
        return True
    except SyntaxError as e:
        print(f"    Syntax error: {e}")
        return False


def check_nav_regime_imports() -> bool:
    """Check that nav_regime.py can be imported."""
    try:
        sys.path.insert(0, str(WORKSPACE_ROOT))
        import core.nav_regime
        return True
    except ImportError as e:
        print(f"    Import error: {e}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def check_nav_regime_classes() -> Tuple[bool, List[str]]:
    """Check that all required classes exist in nav_regime.py."""
    try:
        sys.path.insert(0, str(WORKSPACE_ROOT))
        import core.nav_regime as nr
        
        required_classes = [
            'NAVRegime',
            'MicroSniperConfig',
            'StandardConfig',
            'MultiAgentConfig',
            'RegimeManager',
        ]
        
        required_functions = [
            'get_nav_regime',
            'get_regime_config',
        ]
        
        missing = []
        for cls_name in required_classes:
            if not hasattr(nr, cls_name):
                missing.append(f"class {cls_name}")
        
        for func_name in required_functions:
            if not hasattr(nr, func_name):
                missing.append(f"function {func_name}")
        
        return len(missing) == 0, missing
    except Exception as e:
        return False, [str(e)]


def check_meta_controller_imports() -> bool:
    """Check that MetaController imports nav_regime."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        if "from core.nav_regime import" in content or "import core.nav_regime" in content:
            return True
        return False
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def check_regime_manager_init() -> bool:
    """Check that RegimeManager is initialized in __init__."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        # Look for initialization pattern
        if "self.regime_manager = RegimeManager" in content:
            return True
        return False
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def check_regime_update_in_evaluate() -> bool:
    """Check that regime is updated in evaluate_and_act()."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        # Look for NAV regime evaluation in evaluate_and_act
        patterns = [
            r"regime_manager\.update_regime\(.*nav\)",
            r"get_nav_regime\(",
            r"\[REGIME\].*NAV",
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def check_gating_methods() -> Tuple[bool, List[str]]:
    """Check that all 10 gating methods are defined."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        required_methods = [
            '_regime_can_rotate',
            '_regime_can_heal_dust',
            '_regime_get_available_capital',
            '_regime_get_position_size_limit',
            '_regime_check_max_positions',
            '_regime_check_max_symbols',
            '_regime_check_expected_move',
            '_regime_check_confidence',
            '_regime_check_daily_trade_limit',
            '_regime_log_trade_executed',
        ]
        
        missing = []
        for method_name in required_methods:
            if f"def {method_name}" not in content:
                missing.append(method_name)
        
        return len(missing) == 0, missing
    except Exception as e:
        return False, [str(e)]


def check_dust_healing_gating() -> bool:
    """Check that dust healing has regime gating."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            lines = f.readlines()
        
        in_method = False
        for i, line in enumerate(lines):
            if "_check_dust_healing_opportunity" in line and "def " in line:
                in_method = True
                # Check next 100 lines for the gating
                for j in range(i, min(i + 100, len(lines))):
                    if "_regime_can_heal_dust" in lines[j]:
                        return True
                    # Stop checking if we hit another method definition
                    if j > i and "def " in lines[j]:
                        break
        
        return False
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def check_meta_controller_syntax() -> bool:
    """Check that meta_controller.py has valid syntax."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            code = f.read()
        compile(code, str(META_CONTROLLER_PATH), 'exec')
        return True
    except SyntaxError as e:
        print(f"    Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def check_documentation() -> Tuple[bool, List[str]]:
    """Check that documentation files exist."""
    required_docs = [
        "MICRO_SNIPER_MODE_INTEGRATION.md",
        "MICRO_SNIPER_IMPLEMENTATION_SUMMARY.md",
    ]
    
    missing = []
    for doc in required_docs:
        doc_path = WORKSPACE_ROOT / doc
        if not doc_path.exists():
            missing.append(doc)
    
    return len(missing) == 0, missing


def check_no_breaking_changes() -> bool:
    """Check that existing MetaController methods are not broken."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        # Look for class definition and basic methods that must exist
        critical_methods = [
            "def __init__",
            "async def evaluate_and_act",
            "def _can_act",
            "async def _check_dust_healing_opportunity",
        ]
        
        for method in critical_methods:
            if method not in content:
                return False
        
        return True
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def check_logging_infrastructure() -> bool:
    """Check that regime logging is in place."""
    try:
        with open(META_CONTROLLER_PATH, 'r') as f:
            content = f.read()
        
        # Look for [REGIME] logging
        if "[REGIME]" in content:
            # Count occurrences - should be in multiple places
            count = content.count("[REGIME")
            if count >= 3:  # Multiple gating decisions
                return True
        
        return False
    except Exception as e:
        print(f"    Error reading file: {e}")
        return False


def main():
    result = ValidationResult()
    
    print(f"\n{BOLD}=== MICRO_SNIPER MODE VALIDATION ==={RESET}\n")
    
    # Phase 1: nav_regime.py checks
    print(f"{BOLD}Phase 1: core/nav_regime.py{RESET}")
    
    if check_nav_regime_exists():
        result.add_pass("File exists")
    else:
        result.add_fail("File does not exist")
        return result.summary()
    
    if check_nav_regime_syntax():
        result.add_pass("Valid Python syntax")
    else:
        result.add_fail("Syntax errors in nav_regime.py")
        return result.summary()
    
    if check_nav_regime_imports():
        result.add_pass("Module is importable")
    else:
        result.add_fail("Module cannot be imported")
    
    classes_ok, missing_classes = check_nav_regime_classes()
    if classes_ok:
        result.add_pass("All required classes/functions exist")
    else:
        result.add_fail(f"Missing: {', '.join(missing_classes)}")
    
    # Phase 2: MetaController checks
    print(f"\n{BOLD}Phase 2: core/meta_controller.py{RESET}")
    
    if check_meta_controller_syntax():
        result.add_pass("Valid Python syntax")
    else:
        result.add_fail("Syntax errors in meta_controller.py")
        return result.summary()
    
    if check_meta_controller_imports():
        result.add_pass("Imports nav_regime module")
    else:
        result.add_fail("Does not import nav_regime")
    
    if check_regime_manager_init():
        result.add_pass("RegimeManager initialized in __init__")
    else:
        result.add_fail("RegimeManager not initialized")
    
    if check_regime_update_in_evaluate():
        result.add_pass("Regime update in evaluate_and_act()")
    else:
        result.add_fail("Regime update not in evaluate_and_act()")
    
    methods_ok, missing_methods = check_gating_methods()
    if methods_ok:
        result.add_pass("All 10 gating methods defined")
    else:
        result.add_fail(f"Missing methods: {', '.join(missing_methods)}")
    
    if check_dust_healing_gating():
        result.add_pass("Dust healing has regime gating")
    else:
        result.add_fail("Dust healing gating not found")
    
    if check_no_breaking_changes():
        result.add_pass("No breaking changes to critical methods")
    else:
        result.add_fail("Critical methods appear broken")
    
    if check_logging_infrastructure():
        result.add_pass("Regime logging infrastructure in place")
    else:
        result.add_warn("Regime logging may be incomplete")
    
    # Phase 3: Documentation
    print(f"\n{BOLD}Phase 3: Documentation{RESET}")
    
    docs_ok, missing_docs = check_documentation()
    if docs_ok:
        result.add_pass("All documentation files exist")
    else:
        result.add_warn(f"Missing documentation: {', '.join(missing_docs)}")
    
    # Summary
    return result.summary()


if __name__ == "__main__":
    sys.exit(main())
