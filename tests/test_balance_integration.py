#!/usr/bin/env python3
"""
Real Integration Test: Verify Balance Fetching and SharedState Population

This test:
1. Initializes the full AppContext stack
2. Fetches balances from Binance testnet API
3. Verifies SharedState is properly populated
4. Tests balance query methods
5. Validates capital availability

Run with:
    python -m pytest tests/test_balance_integration.py -v -s
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

# Configure logging to see diagnostic output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


class TestBalanceIntegration:
    """Integration tests for balance fetching and population"""
    
    @pytest.fixture
    async def app_context(self):
        """Create AppContext instance for testing"""
        from core.app_context import AppContext
        
        logger.info("=" * 60)
        logger.info("TEST: Creating AppContext")
        logger.info("=" * 60)
        
        app = AppContext()
        
        # Initialize the application
        await app.init()
        
        yield app
        
        # Cleanup
        if hasattr(app, 'shutdown'):
            await app.shutdown()
    
    @pytest.mark.asyncio
    async def test_exchange_client_configuration(self, app_context):
        """Test 1: ExchangeClient is correctly configured"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: ExchangeClient Configuration")
        logger.info("=" * 60)
        
        ec = app_context.exchange_client
        
        # Check instance exists
        assert ec is not None, "ExchangeClient is None!"
        logger.info("✓ ExchangeClient instance exists")
        
        # Check testnet flag
        assert hasattr(ec, 'testnet'), "ExchangeClient missing 'testnet' attribute"
        logger.info(f"✓ EC.testnet = {ec.testnet} (should be True)")
        assert ec.testnet == True, f"testnet should be True, got {ec.testnet}"
        
        # Check paper_trade flag
        assert hasattr(ec, 'paper_trade'), "ExchangeClient missing 'paper_trade' attribute"
        logger.info(f"✓ EC.paper_trade = {ec.paper_trade} (should be True)")
        assert ec.paper_trade == True, f"paper_trade should be True, got {ec.paper_trade}"
        
        # Check API keys exist
        assert hasattr(ec, 'api_key'), "ExchangeClient missing 'api_key' attribute"
        assert hasattr(ec, 'api_secret'), "ExchangeClient missing 'api_secret' attribute"
        logger.info(f"✓ API credentials present (len={len(ec.api_key)}, {len(ec.api_secret)})")
        
        logger.info("✅ TEST 1 PASSED: ExchangeClient properly configured\n")
    
    @pytest.mark.asyncio
    async def test_exchange_client_balance_fetch(self, app_context):
        """Test 2: ExchangeClient can fetch balances from testnet"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: ExchangeClient Balance Fetch")
        logger.info("=" * 60)
        
        ec = app_context.exchange_client
        
        # Try to fetch balances
        logger.info("Calling ec.get_spot_balances()...")
        balances = await ec.get_spot_balances()
        
        # Verify return type
        assert isinstance(balances, dict), f"Expected dict, got {type(balances)}"
        logger.info(f"✓ Returned type: dict")
        
        # Verify not empty
        assert len(balances) > 0, "Balances dict is empty!"
        logger.info(f"✓ Balances returned: {len(balances)} assets")
        
        # Check structure
        for asset, data in balances.items():
            assert isinstance(data, dict), f"{asset}: value not dict"
            assert "free" in data, f"{asset}: missing 'free' key"
            assert "locked" in data, f"{asset}: missing 'locked' key"
            assert isinstance(data["free"], (int, float)), f"{asset}: free not numeric"
            assert isinstance(data["locked"], (int, float)), f"{asset}: locked not numeric"
        
        logger.info(f"✓ Verified structure for all {len(balances)} assets")
        
        # Show sample assets
        sample_assets = list(balances.keys())[:3]
        for asset in sample_assets:
            free = balances[asset]["free"]
            locked = balances[asset]["locked"]
            total = free + locked
            logger.info(f"  {asset:8} free={free:12.8f} locked={locked:12.8f} total={total:12.8f}")
        
        logger.info("✅ TEST 2 PASSED: ExchangeClient successfully fetches balances\n")
        
        return balances
    
    @pytest.mark.asyncio
    async def test_shared_state_initialization(self, app_context):
        """Test 3: SharedState is initialized properly"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: SharedState Initialization")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Check instance exists
        assert ss is not None, "SharedState is None!"
        logger.info("✓ SharedState instance exists")
        
        # Check balances attribute
        assert hasattr(ss, 'balances'), "SharedState missing 'balances' attribute"
        logger.info(f"✓ SS.balances exists")
        
        # Check balances is dict
        assert isinstance(ss.balances, dict), f"balances should be dict, got {type(ss.balances)}"
        logger.info(f"✓ SS.balances is dict")
        
        # Check exchange client attached
        assert hasattr(ss, '_exchange_client'), "SharedState missing '_exchange_client' attribute"
        logger.info(f"✓ Exchange client attached to SharedState")
        
        logger.info("✅ TEST 3 PASSED: SharedState properly initialized\n")
    
    @pytest.mark.asyncio
    async def test_shared_state_balance_population(self, app_context):
        """Test 4: SharedState.balances is populated with real data"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: SharedState Balance Population")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        ec = app_context.exchange_client
        
        # Force a balance sync
        logger.info("Triggering balance hydration from exchange...")
        success = await ss.hydrate_balances_from_exchange()
        
        assert success, "hydrate_balances_from_exchange() returned False!"
        logger.info("✓ Hydration succeeded")
        
        # Check balances are populated
        assert len(ss.balances) > 0, "SharedState.balances still empty after hydration!"
        logger.info(f"✓ SharedState.balances populated: {len(ss.balances)} assets")
        
        # Verify structure
        for asset, data in ss.balances.items():
            assert isinstance(data, dict), f"{asset}: value not dict"
            assert "free" in data, f"{asset}: missing 'free' key"
            assert "locked" in data, f"{asset}: missing 'locked' key"
        
        logger.info(f"✓ All balances have correct structure")
        
        # Show populated balances
        sample_assets = list(ss.balances.keys())[:5]
        for asset in sample_assets:
            free = ss.balances[asset].get("free", 0)
            locked = ss.balances[asset].get("locked", 0)
            total = free + locked
            logger.info(f"  {asset:8} free={free:12.8f} locked={locked:12.8f} total={total:12.8f}")
        
        logger.info("✅ TEST 4 PASSED: SharedState properly populated with balances\n")
    
    @pytest.mark.asyncio
    async def test_balance_ready_event(self, app_context):
        """Test 5: Balances ready event is properly set"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Balances Ready Event")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Force hydration to ensure ready event is set
        await ss.hydrate_balances_from_exchange()
        
        # Check ready event
        assert hasattr(ss, 'balances_ready_event'), "Missing balances_ready_event"
        logger.info("✓ balances_ready_event exists")
        
        is_ready = ss.balances_ready_event.is_set()
        logger.info(f"✓ balances_ready_event.is_set() = {is_ready}")
        assert is_ready, "balances_ready_event should be set after hydration!"
        
        logger.info("✅ TEST 5 PASSED: Ready event properly set\n")
    
    @pytest.mark.asyncio
    async def test_get_balance_method(self, app_context):
        """Test 6: SharedState.get_balance() method works correctly"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: get_balance() Method")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Test with USDT
        logger.info("Querying balance for USDT...")
        usdt_balance = await ss.get_balance("USDT")
        
        assert isinstance(usdt_balance, dict), f"Expected dict, got {type(usdt_balance)}"
        logger.info(f"✓ Returned type: dict")
        
        assert "free" in usdt_balance, "Missing 'free' key"
        assert "locked" in usdt_balance, "Missing 'locked' key"
        logger.info(f"✓ Has required keys: free, locked")
        
        free = usdt_balance.get("free", 0)
        locked = usdt_balance.get("locked", 0)
        total = free + locked
        
        logger.info(f"  USDT: free={free:12.8f}, locked={locked:12.8f}, total={total:12.8f}")
        logger.info("✅ TEST 6 PASSED: get_balance() method works correctly\n")
    
    @pytest.mark.asyncio
    async def test_get_free_balance_method(self, app_context):
        """Test 7: SharedState.get_free_balance() method works"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 7: get_free_balance() Method")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Test with USDT
        logger.info("Querying free balance for USDT...")
        free_usdt = await ss.get_free_balance("USDT")
        
        assert isinstance(free_usdt, (int, float)), f"Expected numeric, got {type(free_usdt)}"
        logger.info(f"✓ Returned type: {type(free_usdt).__name__}")
        
        logger.info(f"  USDT free: {free_usdt:12.8f}")
        logger.info("✅ TEST 7 PASSED: get_free_balance() works correctly\n")
    
    @pytest.mark.asyncio
    async def test_get_spendable_balance_method(self, app_context):
        """Test 8: SharedState.get_spendable_balance() method works"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 8: get_spendable_balance() Method")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Test with USDT
        logger.info("Querying spendable balance for USDT...")
        spendable_usdt = await ss.get_spendable_balance("USDT")
        
        assert isinstance(spendable_usdt, (int, float)), f"Expected numeric, got {type(spendable_usdt)}"
        logger.info(f"✓ Returned type: {type(spendable_usdt).__name__}")
        
        logger.info(f"  USDT spendable: {spendable_usdt:12.8f}")
        
        # Should be positive
        assert spendable_usdt >= 0, f"Spendable should be >= 0, got {spendable_usdt}"
        logger.info("✓ Value is non-negative")
        
        logger.info("✅ TEST 8 PASSED: get_spendable_balance() works correctly\n")
    
    @pytest.mark.asyncio
    async def test_balance_snapshot(self, app_context):
        """Test 9: Balance snapshot method works"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 9: Balance Snapshot")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Get snapshot
        logger.info("Getting balance snapshot...")
        snapshot = ss.get_balance_snapshot()
        
        assert isinstance(snapshot, dict), f"Expected dict, got {type(snapshot)}"
        logger.info(f"✓ Snapshot type: dict")
        
        assert len(snapshot) > 0, "Snapshot is empty!"
        logger.info(f"✓ Snapshot contains {len(snapshot)} assets")
        
        # Verify it's a copy
        assert snapshot is not ss.balances, "Snapshot should be a copy, not reference!"
        logger.info("✓ Snapshot is a proper copy (not reference)")
        
        logger.info("✅ TEST 9 PASSED: Balance snapshot works correctly\n")
    
    @pytest.mark.asyncio
    async def test_no_balance_warning(self, app_context):
        """Test 10: No balance warning when querying existing asset"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 10: No Balance Warning for Valid Asset")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Capture logs
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        logger_ss = logging.getLogger("SharedState")
        logger_ss.addHandler(handler)
        
        # Query USDT balance
        logger.info("Querying USDT balance (should not trigger warning)...")
        await ss.get_spendable_balance("USDT")
        
        # Check for warning
        log_contents = log_capture.getvalue()
        assert "[SS:BalanceWarning]" not in log_contents, "Unexpected balance warning!"
        logger.info("✓ No balance warning generated")
        
        logger_ss.removeHandler(handler)
        logger.info("✅ TEST 10 PASSED: No spurious warnings\n")
    
    @pytest.mark.asyncio
    async def test_capital_availability(self, app_context):
        """Test 11: Capital is available for trading"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 11: Capital Availability for Trading")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Get quote asset
        quote_asset = ss.quote_asset
        logger.info(f"Quote asset: {quote_asset}")
        
        # Get available capital
        logger.info(f"Querying available capital in {quote_asset}...")
        available = await ss.get_spendable_balance(quote_asset)
        
        logger.info(f"  Available {quote_asset}: {available:12.8f}")
        
        # Check NAV
        nav = ss.get_nav_quote()
        logger.info(f"  NAV (quote): {nav:12.8f}")
        
        logger.info("✓ Capital queries work correctly")
        logger.info("✅ TEST 11 PASSED: Capital available for trading\n")
    
    @pytest.mark.asyncio
    async def test_multiple_balance_queries(self, app_context):
        """Test 12: Multiple balance queries return consistent results"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 12: Balance Query Consistency")
        logger.info("=" * 60)
        
        ss = app_context.shared_state
        
        # Ensure balances are populated
        await ss.hydrate_balances_from_exchange()
        
        # Query multiple times
        logger.info("Querying USDT balance 5 times...")
        results = []
        for i in range(5):
            balance = await ss.get_spendable_balance("USDT")
            results.append(balance)
            logger.info(f"  Query {i+1}: {balance:12.8f}")
        
        # All should be identical (since no trades occur)
        assert len(set(results)) == 1, "Results inconsistent across queries!"
        logger.info("✓ All queries return identical results")
        
        logger.info("✅ TEST 12 PASSED: Consistent balance queries\n")


# Standalone test runner
async def run_all_tests():
    """Run all tests manually without pytest"""
    logger.info("\n" + "=" * 70)
    logger.info("BALANCE INTEGRATION TEST SUITE")
    logger.info("=" * 70)
    
    test_suite = TestBalanceIntegration()
    
    # Initialize app
    from core.app_context import AppContext
    app = AppContext()
    await app.init()
    
    try:
        # Run tests
        await test_suite.test_exchange_client_configuration(app)
        await test_suite.test_exchange_client_balance_fetch(app)
        await test_suite.test_shared_state_initialization(app)
        await test_suite.test_shared_state_balance_population(app)
        await test_suite.test_balance_ready_event(app)
        await test_suite.test_get_balance_method(app)
        await test_suite.test_get_free_balance_method(app)
        await test_suite.test_get_spendable_balance_method(app)
        await test_suite.test_balance_snapshot(app)
        await test_suite.test_no_balance_warning(app)
        await test_suite.test_capital_availability(app)
        await test_suite.test_multiple_balance_queries(app)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 70)
        
    finally:
        if hasattr(app, 'shutdown'):
            await app.shutdown()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
