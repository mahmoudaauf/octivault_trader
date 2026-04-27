"""
Component Health & Initialization System
Validates all system components and their dependencies
Date: April 11, 2026
"""

import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComponentStatus:
    """Status of a component"""
    name: str
    status: str  # 'OK', 'WARNING', 'ERROR'
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dependencies: List[str] = field(default_factory=list)
    initialization_time: float = 0.0


class ComponentValidator:
    """Validates and initializes system components"""
    
    # Core required components
    CORE_COMPONENTS = {
        'AppContext': 'core.app_context',
        'HealthCheckManager': 'core.health_check_manager',
        'PositionManager': 'core.position_manager',
        'PortfolioManager': 'core.portfolio_manager',
        'RiskManager': 'core.risk_manager',
        'Bootstrap Manager': 'core.bootstrap_manager',
        'Market Data Feed': 'core.market_data_feed',
        'Trading Coordinator': 'core.trading_coordinator',
        'Execution Logic': 'core.execution_logic',
        'Config Constants': 'core.config_constants',
    }
    
    # Optional components (nice to have)
    OPTIONAL_COMPONENTS = {
        'Dashboard Server': 'dashboard_server',
        'Prometheus Exporter': 'core.prometheus_exporter',
        'Health Endpoints': 'core.health_endpoints',
        'Retraining Engine': 'core.retraining_engine',
        'Symbol Manager': 'core.symbol_manager',
    }
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.sys_path = str(self.workspace_path)
        if self.sys_path not in sys.path:
            sys.path.insert(0, self.sys_path)
        
        self.results: Dict[str, ComponentStatus] = {}
    
    def validate_core_components(self) -> Tuple[int, int]:
        """Validate all core components"""
        logger.info("=" * 70)
        logger.info("VALIDATING CORE COMPONENTS")
        logger.info("=" * 70)
        
        passed = 0
        failed = 0
        
        for component_name, module_path in self.CORE_COMPONENTS.items():
            try:
                start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop() else 0
                __import__(module_path)
                end_time = asyncio.get_event_loop().time() if asyncio.get_event_loop() else 0
                
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='OK',
                    message=f"Module '{module_path}' loaded successfully",
                    initialization_time=end_time - start_time if start_time > 0 else 0
                )
                logger.info(f"✅ {component_name}: OK ({module_path})")
                passed += 1
                
            except ImportError as e:
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='ERROR',
                    message=f"Import failed: {str(e)[:100]}"
                )
                logger.error(f"❌ {component_name}: IMPORT ERROR - {str(e)[:100]}")
                failed += 1
                
            except Exception as e:
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='WARNING',
                    message=f"Validation warning: {str(e)[:100]}"
                )
                logger.warning(f"⚠️  {component_name}: WARNING - {str(e)[:100]}")
                failed += 1
        
        return passed, failed
    
    def validate_optional_components(self) -> Tuple[int, int]:
        """Validate optional components"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING OPTIONAL COMPONENTS")
        logger.info("=" * 70)
        
        passed = 0
        skipped = 0
        
        for component_name, module_path in self.OPTIONAL_COMPONENTS.items():
            try:
                __import__(module_path)
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='OK',
                    message=f"Module '{module_path}' loaded successfully"
                )
                logger.info(f"✅ {component_name}: AVAILABLE ({module_path})")
                passed += 1
                
            except ImportError:
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='WARNING',
                    message="Optional component not available"
                )
                logger.info(f"ℹ️  {component_name}: NOT AVAILABLE (optional)")
                skipped += 1
                
            except Exception as e:
                self.results[component_name] = ComponentStatus(
                    name=component_name,
                    status='WARNING',
                    message=f"Optional component skipped: {str(e)[:80]}"
                )
                logger.info(f"ℹ️  {component_name}: SKIPPED - {str(e)[:80]}")
                skipped += 1
        
        return passed, skipped
    
    def validate_file_structure(self) -> Tuple[int, int]:
        """Validate project file structure"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING PROJECT STRUCTURE")
        logger.info("=" * 70)
        
        required_dirs = ['core', 'tests', 'logs', 'data']
        optional_dirs = ['models', 'config', 'scripts', 'docs']
        
        valid = 0
        missing = 0
        
        for dir_name in required_dirs:
            dir_path = self.workspace_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"✅ Directory '{dir_name}': EXISTS")
                valid += 1
            else:
                logger.warning(f"❌ Directory '{dir_name}': MISSING")
                missing += 1
        
        # Check optional directories
        for dir_name in optional_dirs:
            dir_path = self.workspace_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"ℹ️  Optional directory '{dir_name}': EXISTS")
            else:
                logger.info(f"ℹ️  Optional directory '{dir_name}': NOT FOUND")
        
        return valid, missing
    
    def validate_configuration(self) -> bool:
        """Validate configuration files"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATING CONFIGURATION")
        logger.info("=" * 70)
        
        config_files = {
            '.env': 'Environment variables',
            'pytest.ini': 'Pytest configuration',
            'setup.py': 'Package setup',
            'requirements.txt': 'Dependencies',
        }
        
        all_valid = True
        
        for filename, description in config_files.items():
            file_path = self.workspace_path / filename
            if file_path.exists():
                logger.info(f"✅ {description} ({filename}): FOUND")
            else:
                logger.info(f"ℹ️  {description} ({filename}): NOT FOUND (optional)")
        
        return all_valid
    
    def check_dependencies(self) -> Tuple[int, int]:
        """Check if required Python packages are installed"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING DEPENDENCIES")
        logger.info("=" * 70)
        
        required_packages = {
            'pytest': 'Testing framework',
            'pytest_asyncio': 'Async test support',
            'asyncio': 'Async I/O',
            'pandas': 'Data analysis',
            'numpy': 'Numerical computing',
            'ccxt': 'Exchange connectivity',
        }
        
        installed = 0
        missing = 0
        
        for package_name, description in required_packages.items():
            try:
                __import__(package_name)
                logger.info(f"✅ {description} ({package_name}): INSTALLED")
                installed += 1
            except ImportError:
                logger.warning(f"❌ {description} ({package_name}): MISSING")
                missing += 1
        
        return installed, missing
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report = "\n" + "=" * 70 + "\n"
        report += "COMPONENT VALIDATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Summary
        core_ok = sum(1 for s in self.results.values() if s.status == 'OK' and s.name in self.CORE_COMPONENTS)
        core_total = len(self.CORE_COMPONENTS)
        
        report += f"Core Components: {core_ok}/{core_total} OK\n"
        
        # Component details
        report += "\nComponent Status:\n"
        report += "-" * 70 + "\n"
        
        for comp_name, status in self.results.items():
            icon = "✅" if status.status == 'OK' else "⚠️" if status.status == 'WARNING' else "❌"
            report += f"{icon} {comp_name}: {status.status}\n"
            if status.message:
                report += f"   {status.message}\n"
        
        report += "\n" + "=" * 70 + "\n"
        
        return report
    
    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        logger.info("\n")
        logger.info("╔" + "=" * 68 + "╗")
        logger.info("║" + " " * 68 + "║")
        logger.info("║" + "  🔍 OCTIVAULT COMPONENT VALIDATION SYSTEM  ".center(68) + "║")
        logger.info("║" + " " * 68 + "║")
        logger.info("╚" + "=" * 68 + "╝")
        
        # Run validations
        file_valid, file_missing = self.validate_file_structure()
        core_pass, core_fail = self.validate_core_components()
        opt_pass, opt_skip = self.validate_optional_components()
        dep_installed, dep_missing = self.check_dependencies()
        self.validate_configuration()
        
        # Generate report
        report = self.generate_report()
        logger.info(report)
        
        # Final verdict
        all_healthy = (file_missing == 0 and core_fail == 0 and dep_missing == 0)
        
        logger.info("\n" + "=" * 70)
        if all_healthy:
            logger.info("✅ SYSTEM HEALTH: EXCELLENT - ALL CRITICAL COMPONENTS OK")
            logger.info("=" * 70)
            return True
        elif core_fail == 0:
            logger.info("🟡 SYSTEM HEALTH: GOOD - SOME OPTIONAL COMPONENTS MISSING")
            logger.info("=" * 70)
            return True
        else:
            logger.error("❌ SYSTEM HEALTH: CRITICAL - ESSENTIAL COMPONENTS FAILED")
            logger.info("=" * 70)
            return False


def main():
    """Main validation routine"""
    workspace_path = Path(__file__).parent
    
    validator = ComponentValidator(str(workspace_path))
    success = validator.run_full_validation()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
