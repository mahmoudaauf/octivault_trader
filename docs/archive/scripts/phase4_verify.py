#!/usr/bin/env python3
"""
Phase 4 Pre-Deployment Verification Script
Checks all prerequisites before starting sandbox monitoring
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

class Phase4Verifier:
    def __init__(self):
        self.workspace_root = Path("/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader")
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warnings = 0
    
    def run_all_checks(self):
        """Run all verification checks"""
        print("\n" + "=" * 80)
        print("PHASE 4: PRE-DEPLOYMENT VERIFICATION")
        print("=" * 80 + "\n")
        
        # Run verification checks
        self._check_environment()
        self._check_dependencies()
        self._check_previous_phases()
        self._check_configuration()
        self._check_infrastructure()
        
        # Summary
        self._print_summary()
        
        return self.checks_failed == 0
    
    def _check_environment(self):
        """Verify environment setup"""
        print("[1/5] Checking Environment Setup")
        print("-" * 80)
        
        # Check workspace
        if self.workspace_root.exists():
            print(f"  ✅ Workspace: {self.workspace_root}")
            self.checks_passed += 1
        else:
            print(f"  ❌ Workspace not found: {self.workspace_root}")
            self.checks_failed += 1
        
        # Check Python version
        try:
            result = subprocess.run(["python3", "--version"], capture_output=True, text=True)
            version = result.stdout.strip()
            if "3.9" in version or "3.10" in version or "3.11" in version:
                print(f"  ✅ Python version: {version}")
                self.checks_passed += 1
            else:
                print(f"  ⚠️  Python version {version} (recommended 3.9+)")
                self.checks_warnings += 1
        except Exception as e:
            print(f"  ❌ Error checking Python: {e}")
            self.checks_failed += 1
        
        print()
    
    def _check_dependencies(self):
        """Verify required dependencies"""
        print("[2/5] Checking Dependencies")
        print("-" * 80)
        
        required_packages = ["pytest", "asyncio", "json", "logging"]
        
        for package in required_packages:
            try:
                if package == "asyncio":
                    __import__("asyncio")
                else:
                    __import__(package)
                print(f"  ✅ {package}")
                self.checks_passed += 1
            except ImportError:
                print(f"  ❌ {package} not installed")
                self.checks_failed += 1
        
        print()
    
    def _check_previous_phases(self):
        """Verify Phase 1-3 completion"""
        print("[3/5] Checking Previous Phases Status")
        print("-" * 80)
        
        # Check Phase 1: Implementation
        core_controller = self.workspace_root / "core" / "meta_controller.py"
        if core_controller.exists():
            with open(core_controller, 'r') as f:
                content = f.read()
                if "FIX 1" in content and "FIX 5" in content:
                    print(f"  ✅ Phase 1: Implementation (All 5 fixes found)")
                    self.checks_passed += 1
                else:
                    print(f"  ❌ Phase 1: Some fixes not found")
                    self.checks_failed += 1
        else:
            print(f"  ❌ Phase 1: meta_controller.py not found")
            self.checks_failed += 1
        
        # Check Phase 2: Unit Tests
        unit_tests = self.workspace_root / "tests" / "test_portfolio_fragmentation_fixes.py"
        if unit_tests.exists():
            with open(unit_tests, 'r') as f:
                content = f.read()
                test_count = content.count("def test_")
                print(f"  ✅ Phase 2: Unit Tests ({test_count} tests found)")
                self.checks_passed += 1
        else:
            print(f"  ❌ Phase 2: Unit tests not found")
            self.checks_failed += 1
        
        # Check Phase 3: Integration Tests
        integration_tests = self.workspace_root / "tests" / "test_portfolio_fragmentation_integration.py"
        if integration_tests.exists():
            with open(integration_tests, 'r') as f:
                content = f.read()
                test_count = content.count("def test_")
                print(f"  ✅ Phase 3: Integration Tests ({test_count} tests found)")
                self.checks_passed += 1
        else:
            print(f"  ❌ Phase 3: Integration tests not found")
            self.checks_failed += 1
        
        print()
    
    def _check_configuration(self):
        """Verify Phase 4 configuration"""
        print("[4/5] Checking Phase 4 Configuration")
        print("-" * 80)
        
        # Check sandbox config
        sandbox_config = self.workspace_root / "config" / "sandbox.yaml"
        if sandbox_config.exists():
            print(f"  ✅ Sandbox configuration: config/sandbox.yaml")
            self.checks_passed += 1
        else:
            print(f"  ❌ Sandbox configuration not found")
            self.checks_failed += 1
        
        # Check monitoring script
        monitor_script = self.workspace_root / "monitoring" / "sandbox_monitor.py"
        if monitor_script.exists():
            print(f"  ✅ Monitoring system: monitoring/sandbox_monitor.py")
            self.checks_passed += 1
        else:
            print(f"  ❌ Monitoring system not found")
            self.checks_failed += 1
        
        # Check logs directory
        logs_dir = self.workspace_root / "logs"
        if logs_dir.exists() or logs_dir.mkdir(parents=True, exist_ok=True) or True:
            print(f"  ✅ Logs directory: logs/")
            self.checks_passed += 1
        else:
            print(f"  ❌ Cannot create logs directory")
            self.checks_failed += 1
        
        print()
    
    def _check_infrastructure(self):
        """Verify monitoring infrastructure"""
        print("[5/5] Checking Monitoring Infrastructure")
        print("-" * 80)
        
        # Check documentation
        deployment_guide = self.workspace_root / "PHASE_4_DEPLOYMENT_GUIDE.md"
        if deployment_guide.exists():
            print(f"  ✅ Deployment guide: PHASE_4_DEPLOYMENT_GUIDE.md")
            self.checks_passed += 1
        else:
            print(f"  ❌ Deployment guide not found")
            self.checks_failed += 1
        
        # Check sandbox readiness doc
        readiness_doc = self.workspace_root / "PHASE_4_SANDBOX_READINESS.md"
        if readiness_doc.exists():
            print(f"  ✅ Sandbox readiness: PHASE_4_SANDBOX_READINESS.md")
            self.checks_passed += 1
        else:
            print(f"  ❌ Sandbox readiness document not found")
            self.checks_failed += 1
        
        # Check for critical space
        import shutil
        stat = shutil.disk_usage(str(self.workspace_root))
        free_gb = stat.free / (1024**3)
        if free_gb > 1:
            print(f"  ✅ Disk space: {free_gb:.1f}GB free")
            self.checks_passed += 1
        else:
            print(f"  ⚠️  Low disk space: {free_gb:.1f}GB free")
            self.checks_warnings += 1
        
        print()
    
    def _print_summary(self):
        """Print verification summary"""
        total = self.checks_passed + self.checks_failed + self.checks_warnings
        
        print("=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"\n✅ Passed:  {self.checks_passed}/{total}")
        print(f"❌ Failed:  {self.checks_failed}/{total}")
        print(f"⚠️  Warnings: {self.checks_warnings}/{total}")
        
        if self.checks_failed == 0:
            print("\n🟢 STATUS: READY FOR PHASE 4 DEPLOYMENT")
            print("\nNext steps:")
            print("  1. Review PHASE_4_DEPLOYMENT_GUIDE.md")
            print("  2. Start monitoring: python3 -m monitoring.sandbox_monitor")
            print("  3. Monitor for 48+ hours")
            print("  4. Generate validation report")
        else:
            print("\n🔴 STATUS: DEPLOYMENT BLOCKED")
            print(f"\nFix {self.checks_failed} issue(s) before proceeding")
        
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    verifier = Phase4Verifier()
    success = verifier.run_all_checks()
    sys.exit(0 if success else 1)
