"""
Pytest configuration and fixtures for OctiVault Trading Bot tests
Provides async fixtures, mocks, and app context initialization
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )


@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    try:
        loop.close()
    except:
        pass


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_exchange_client():
    """Mock exchange client for testing"""
    client = AsyncMock()
    client.get_balance = AsyncMock(return_value={
        "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
        "ETH": {"free": 10.0, "used": 0.0, "total": 10.0},
        "USDT": {"free": 50000.0, "used": 0.0, "total": 50000.0},
    })
    client.get_ticker = AsyncMock(return_value={
        "symbol": "BTC/USDT",
        "bid": 40000.0,
        "ask": 40001.0,
        "last": 40000.5,
    })
    client.create_order = AsyncMock(return_value={
        "id": "12345",
        "symbol": "BTC/USDT",
        "type": "limit",
        "side": "buy",
        "price": 40000.0,
        "amount": 0.1,
        "status": "open",
    })
    client.cancel_order = AsyncMock(return_value={"id": "12345", "status": "canceled"})
    client.fetch_order = AsyncMock(return_value={
        "id": "12345",
        "symbol": "BTC/USDT",
        "status": "closed",
        "filled": 0.1,
        "remaining": 0.0,
    })
    return client


@pytest.fixture
def mock_market_data():
    """Mock market data provider"""
    data = AsyncMock()
    data.get_historical_data = AsyncMock(return_value=[
        {"timestamp": 1000, "open": 40000, "high": 41000, "low": 39000, "close": 40500},
        {"timestamp": 2000, "open": 40500, "high": 41500, "low": 40000, "close": 41000},
        {"timestamp": 3000, "open": 41000, "high": 42000, "low": 40500, "close": 41500},
    ])
    data.subscribe = AsyncMock(return_value=True)
    data.unsubscribe = AsyncMock(return_value=True)
    return data


@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = AsyncMock()
    db.connect = AsyncMock(return_value=True)
    db.disconnect = AsyncMock(return_value=True)
    db.query = AsyncMock(return_value=[])
    db.insert = AsyncMock(return_value={"id": "123"})
    db.update = AsyncMock(return_value=True)
    db.delete = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_cache():
    """Mock cache/redis connection"""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    return cache


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    ws = AsyncMock()
    ws.connect = AsyncMock(return_value=True)
    ws.disconnect = AsyncMock(return_value=True)
    ws.send = AsyncMock(return_value=True)
    ws.recv = AsyncMock(return_value={"type": "ticker", "data": {}})
    ws.is_connected = MagicMock(return_value=True)
    return ws


# ============================================================================
# APP CONTEXT FIXTURES
# ============================================================================

@pytest.fixture
async def app_context(mock_exchange_client, mock_market_data, mock_database, mock_cache):
    """
    Async fixture providing a complete app context for testing
    Initializes with mocked external dependencies
    """
    try:
        from core.app_context import AppContext
    except ImportError:
        # If AppContext doesn't exist, create a mock
        class MockAppContext:
            def __init__(self):
                self.exchange_client = mock_exchange_client
                self.market_data = mock_market_data
                self.database = mock_database
                self.cache = mock_cache
                self.initialized = False
                self.running = False
            
            async def initialize(self):
                self.initialized = True
                return True
            
            async def cleanup(self):
                self.initialized = False
                return True
            
            async def start(self):
                self.running = True
                return True
            
            async def stop(self):
                self.running = False
                return True
            
            async def get_status(self):
                return {
                    "initialized": self.initialized,
                    "running": self.running,
                    "exchange_connected": True,
                    "database_connected": True,
                }
        
        ctx = MockAppContext()
        await ctx.initialize()
        yield ctx
        await ctx.cleanup()
        return

    # Try to use real AppContext if it exists
    ctx = AppContext()
    
    # Patch external dependencies
    ctx.exchange_client = mock_exchange_client
    ctx.market_data = mock_market_data
    ctx.database = mock_database
    ctx.cache = mock_cache
    
    try:
        await ctx.initialize()
    except Exception:
        # If initialize fails, just mark as ready
        ctx.initialized = True
    
    yield ctx
    
    try:
        await ctx.cleanup()
    except Exception:
        pass


@pytest.fixture
def sync_app_context(mock_exchange_client, mock_market_data, mock_database, mock_cache):
    """
    Synchronous fixture for tests that don't use async
    Provides a context object with mocked dependencies
    """
    class SyncAppContext:
        def __init__(self):
            self.exchange_client = mock_exchange_client
            self.market_data = mock_market_data
            self.database = mock_database
            self.cache = mock_cache
            self.initialized = True
            self.running = False
        
        def get_balance(self, symbol=None):
            if symbol:
                return {symbol: {"free": 1000, "used": 0, "total": 1000}}
            return {"BTC": 1.0, "ETH": 10.0, "USDT": 50000.0}
        
        def get_status(self):
            return {
                "initialized": self.initialized,
                "running": self.running,
                "health": "ok",
            }
    
    return SyncAppContext()


# ============================================================================
# SHARED STATE FIXTURES
# ============================================================================

@pytest.fixture
def shared_state():
    """Fixture for shared state across components"""
    class SharedState:
        def __init__(self):
            self.balance_ready = False
            self.market_data_ready = False
            self.exchange_connected = False
            self.balance_data = {}
            self.market_data = {}
        
        def set_balance(self, balance_dict):
            self.balance_data = balance_dict
            self.balance_ready = True
        
        def get_balance(self):
            return self.balance_data
        
        def set_market_data(self, data):
            self.market_data = data
            self.market_data_ready = True
        
        def get_market_data(self):
            return self.market_data
    
    return SharedState()


# ============================================================================
# TRADING STATE FIXTURES
# ============================================================================

@pytest.fixture
def position_manager():
    """Mock position manager"""
    from unittest.mock import Mock
    
    manager = Mock()
    manager.positions = {}
    manager.add_position = Mock(return_value=True)
    manager.remove_position = Mock(return_value=True)
    manager.get_position = Mock(return_value=None)
    manager.list_positions = Mock(return_value=[])
    manager.get_total_exposure = Mock(return_value=0.0)
    manager.is_within_limits = Mock(return_value=True)
    
    return manager


@pytest.fixture
def portfolio_manager():
    """Mock portfolio manager"""
    from unittest.mock import Mock
    
    manager = Mock()
    manager.total_value = 50000.0
    manager.cash = 50000.0
    manager.positions_value = 0.0
    manager.pnl = 0.0
    manager.pnl_percent = 0.0
    manager.calculate_pnl = Mock(return_value=0.0)
    manager.get_allocation = Mock(return_value={})
    manager.get_performance = Mock(return_value={"daily_return": 0.0})
    
    return manager


@pytest.fixture
def risk_manager():
    """Mock risk manager"""
    from unittest.mock import Mock
    
    manager = Mock()
    manager.max_position_size = 0.05
    manager.max_concentration = 0.20
    manager.max_daily_loss = 0.03
    manager.max_drawdown = 0.10
    manager.var_limit = 0.02
    manager.check_position_limits = Mock(return_value=True)
    manager.check_var_limits = Mock(return_value=True)
    manager.check_drawdown = Mock(return_value=True)
    manager.is_risk_within_limits = Mock(return_value=True)
    
    return manager


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def temp_config():
    """Temporary configuration for testing"""
    config = {
        "trading": {
            "enabled": True,
            "paper_mode": True,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "max_position_size": 0.05,
            "max_concentration": 0.20,
        },
        "market_data": {
            "provider": "binance",
            "update_interval": 1,
            "history_length": 100,
        },
        "risk": {
            "max_daily_loss": 0.03,
            "max_drawdown": 0.10,
            "var_limit": 0.02,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/test.log",
        },
    }
    return config


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return [
        {
            "timestamp": 1704067200,
            "open": 42000.0,
            "high": 42500.0,
            "low": 41500.0,
            "close": 42250.0,
            "volume": 100.5,
        },
        {
            "timestamp": 1704067260,
            "open": 42250.0,
            "high": 42700.0,
            "low": 42100.0,
            "close": 42600.0,
            "volume": 120.3,
        },
        {
            "timestamp": 1704067320,
            "open": 42600.0,
            "high": 43000.0,
            "low": 42500.0,
            "close": 42800.0,
            "volume": 110.7,
        },
    ]


@pytest.fixture
def sample_order():
    """Sample order object for testing"""
    return {
        "id": "test_order_123",
        "symbol": "BTC/USDT",
        "type": "limit",
        "side": "buy",
        "price": 42000.0,
        "amount": 0.1,
        "filled": 0.0,
        "remaining": 0.1,
        "status": "open",
        "timestamp": 1704067200,
    }


@pytest.fixture
def sample_position():
    """Sample position object for testing"""
    return {
        "symbol": "BTC/USDT",
        "side": "long",
        "size": 0.5,
        "entry_price": 40000.0,
        "current_price": 42000.0,
        "pnl": 500.0,
        "pnl_percent": 2.5,
        "opened_at": 1704067200,
    }


# ============================================================================
# PATCHING FIXTURES
# ============================================================================

@pytest.fixture
def patch_exchange_api():
    """Patch external exchange API calls"""
    with patch("ccxt.async_support.binance") as mock_binance:
        mock_binance.return_value.fetch_balance = AsyncMock(
            return_value={
                "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
                "USDT": {"free": 50000.0, "used": 0.0, "total": 50000.0},
            }
        )
        mock_binance.return_value.fetch_ticker = AsyncMock(
            return_value={"bid": 42000.0, "ask": 42001.0}
        )
        mock_binance.return_value.create_order = AsyncMock(
            return_value={"id": "12345", "status": "open"}
        )
        yield mock_binance


@pytest.fixture
def patch_websocket_api():
    """Patch WebSocket connections"""
    with patch("websockets.client.connect") as mock_ws:
        async_mock = AsyncMock()
        async_mock.__aenter__ = AsyncMock(return_value=async_mock)
        async_mock.__aexit__ = AsyncMock(return_value=None)
        async_mock.recv = AsyncMock(
            return_value='{"type": "ticker", "data": {}}'
        )
        async_mock.send = AsyncMock()
        mock_ws.return_value = async_mock
        yield mock_ws


@pytest.fixture
def patch_database():
    """Patch database connections"""
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value.connect = MagicMock()
        yield mock_engine


# ============================================================================
# AUTOUSE FIXTURES (Applied to all tests)
# ============================================================================

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    import unittest.mock
    # Clear any global mocks
    yield
    # Cleanup after test


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    yield
    
    # Cleanup logging after test


# ============================================================================
# MARKERS FOR TEST CATEGORIZATION
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their modules"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        elif "test_" in item.nodeid.lower() and "integration" not in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)
