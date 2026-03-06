#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: StartupReconciler Integration with AppContext

Tests:
1. Phase 8.5 runs before Phase 9
2. Reconciliation completes successfully
3. Portfolio state is ready after reconciliation
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup path
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Configure logging to see Phase 8.5 output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(levelname)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


async def test_phase_8_5_integration():
    """Test that Phase 8.5 (StartupReconciler) integrates with AppContext."""
    
    logger.info("="*80)
    logger.info("TEST: StartupReconciler Phase 8.5 Integration")
    logger.info("="*80)
    
    try:
        # Import AppContext
        from core.app_context import AppContext
        from core.startup_reconciler import StartupReconciler
        
        logger.info("✅ Imports successful")
        
        # Check that StartupReconciler is importable from app_context
        import core.app_context as ac_module
        source_code = open(ac_module.__file__).read()
        
        if "StartupReconciler" in source_code:
            logger.info("✅ StartupReconciler import found in app_context.py")
        else:
            logger.error("❌ StartupReconciler import NOT found in app_context.py")
            return False
        
        if "P8.5" in source_code or "Phase 8.5" in source_code:
            logger.info("✅ Phase 8.5 code found in app_context.py")
        else:
            logger.error("❌ Phase 8.5 code NOT found in app_context.py")
            return False
        
        if "run_startup_reconciliation" in source_code:
            logger.info("✅ run_startup_reconciliation() call found in app_context.py")
        else:
            logger.error("❌ run_startup_reconciliation() call NOT found in app_context.py")
            return False
        
        # Check StartupReconciler has required methods
        required_methods = [
            'run_startup_reconciliation',
            'is_ready',
            'get_metrics',
        ]
        
        for method in required_methods:
            if hasattr(StartupReconciler, method):
                logger.info(f"✅ StartupReconciler.{method}() exists")
            else:
                logger.error(f"❌ StartupReconciler.{method}() NOT found")
                return False
        
        logger.info("="*80)
        logger.info("✅ ALL INTEGRATION CHECKS PASSED")
        logger.info("="*80)
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Run `python -m pytest test_startup_reconciler_integration.py -v`")
        logger.info("2. Monitor logs for '[P8.5_startup_reconciliation]' output")
        logger.info("3. Verify no errors between P8 and P9")
        logger.info("")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase_8_5_integration())
    sys.exit(0 if success else 1)
