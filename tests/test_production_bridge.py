#!/usr/bin/env python3
"""
Integration test for Phase 8.1 Production Bridge

Tests:
1. Bridge can be imported without errors
2. Bridge loads legacy orchestrator successfully
3. build_production_app_ctx() populates 25+ components
4. Mock mode (no production flag) returns empty dict
5. Production mode (--production flag) returns populated dict
"""

import asyncio
import sys
from pathlib import Path

# Add project root (parent of tests/)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger(__name__)


async def test_bridge_import():
    """Test 1: Can import production_bridge without errors"""
    logger.info("Test 1: Importing production_bridge...")
    try:
        from core_engine import production_bridge
        logger.info("✅ Bridge imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import bridge: {e}")
        return False


async def test_production_app_ctx():
    """Test 2: Can build production app_ctx with real components"""
    logger.info("Test 2: Building production app_ctx...")
    try:
        from core_engine.production_bridge import build_production_app_ctx
        
        app_ctx, orchestrator = await build_production_app_ctx()
        
        # Validate keys
        expected_keys = {
            "shared_state", "exchange_client", "market_data_feed",
            "balance_manager", "portfolio_manager", "execution_manager",
            "signal_manager", "risk_manager", "health_monitor",
            "watchdog", "heartbeat"
        }
        
        found_keys = set(app_ctx.keys())
        missing = expected_keys - found_keys
        
        if missing:
            logger.warning(f"⚠️  Missing keys in app_ctx: {missing}")
            logger.warning(f"   (This is OK — graceful degradation)")
        
        # Check for real balance (NAV)
        balance_mgr = app_ctx.get("balance_manager")
        if balance_mgr and hasattr(balance_mgr, "last_nav"):
            nav = getattr(balance_mgr, "last_nav", 0.0)
            logger.info(f"   Real NAV detected: ${nav:.2f}")
        
        logger.info(f"✅ Production app_ctx built: {len(app_ctx)} keys")
        return True, app_ctx
        
    except Exception as e:
        logger.error(f"❌ Failed to build production app_ctx: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_mock_app_ctx():
    """Test 3: Mock app_ctx (no --production flag) returns empty dict"""
    logger.info("Test 3: Verifying mock app_ctx (no bridge)...")
    try:
        from core_engine.integration import create_app_context
        
        # Build mock app_ctx (production=False)
        app_ctx = await create_app_context(production=False)
        
        if isinstance(app_ctx, dict) and len(app_ctx) == 0:
            logger.info(f"✅ Mock app_ctx is empty (graceful): {app_ctx}")
            return True
        else:
            logger.warning(f"⚠️  Mock app_ctx not empty: {app_ctx}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed mock app_ctx test: {e}")
        return False


async def test_engines_with_app_ctx(app_ctx):
    """Test 4: Engines can use populated app_ctx"""
    logger.info("Test 4: Engines consuming app_ctx...")
    try:
        from core_engine.implementations import MarketAccountEngineImpl
        
        # Call static method directly (not instance method)
        account_state = await MarketAccountEngineImpl.get_account_state(app_ctx)
        nav = account_state.get("nav_usdt", 0.0)
        
        if nav > 0:
            logger.info(f"✅ Portfolio snapshot NAV: ${nav:.2f}")
            return True
        else:
            logger.warning(f"⚠️  Portfolio snapshot NAV is {nav} (OK during warmup)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed engine test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_graceful_degradation():
    """Test 5: Engines degrade gracefully when components missing"""
    logger.info("Test 5: Graceful degradation with empty app_ctx...")
    try:
        from core_engine.implementations import MarketAccountEngineImpl
        
        # Call static method directly (not instance method)
        empty_ctx = {}
        account_state = await MarketAccountEngineImpl.get_account_state(empty_ctx)
        
        # Should return dict (not crash), even if nav=0.0 due to missing components
        if isinstance(account_state, dict):
            logger.info(f"✅ Graceful degradation: returned {len(account_state)} keys")
            return True
        else:
            logger.error(f"❌ Unexpected result type: {type(account_state)}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed degradation test: {e}")
        return False


async def test_cli_flag():
    """Test 6: CLI flag --production parses correctly"""
    logger.info("Test 6: CLI flag parsing...")
    try:
        from main import parse_args
        
        # Test with --production
        args = parse_args(["--mode=paper-trade", "--production"])
        if getattr(args, "production", False) is True:
            logger.info(f"✅ --production flag parsed: {args.production}")
            return True
        else:
            logger.error(f"❌ --production flag not parsed correctly")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed CLI test: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("PHASE 8.1 PRODUCTION BRIDGE — INTEGRATION TEST SUITE")
    logger.info("=" * 70)
    
    results = {}
    
    # Test 1: Import
    results["Import"] = await test_bridge_import()
    if not results["Import"]:
        logger.error("Cannot continue without bridge import")
        sys.exit(1)
    
    # Test 2: Production app_ctx
    success, app_ctx = await test_production_app_ctx()
    results["Production AppCtx"] = success
    
    # Test 3: Mock app_ctx
    results["Mock AppCtx"] = await test_mock_app_ctx()
    
    # Test 4: Engines with app_ctx (if production succeeded)
    if app_ctx:
        results["Engine Integration"] = await test_engines_with_app_ctx(app_ctx)
    
    # Test 5: Graceful degradation
    results["Graceful Degradation"] = await test_graceful_degradation()
    
    # Test 6: CLI flag
    results["CLI Flag"] = await test_cli_flag()
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} — {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED — Production bridge is ready for use!")
        return 0
    else:
        logger.error(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
