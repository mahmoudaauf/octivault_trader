"""
Concrete Wiring Examples - Copy/Paste Ready
═════════════════════════════════════════════════════════════════════════════

This file contains ready-to-use code snippets for wiring each engine.
Copy these directly into your engine files.
"""


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Wire MarketAccountEngine.get_account_state()
# ═════════════════════════════════════════════════════════════════════════════

MARKET_ACCOUNT_ENGINE_WIRING = '''
# In core_engine/market_account_engine.py, replace the get_account_state method:

from core_engine.implementations import MarketAccountEngineImpl

class MarketAccountEngine:
    """..existing docstring..."""

    async def get_account_state(self) -> Dict[str, Any]:
        """
        Fetch account state from exchange_client (L1).
        Real implementation delegates to L1 components.
        """
        return await MarketAccountEngineImpl.get_account_state(self.app_ctx)

    async def get_market_prices(self, symbols: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Fetch prices from market_data_feed or exchange_client (L1/L2).
        """
        return await MarketAccountEngineImpl.get_market_prices(self.app_ctx, symbols)

    async def get_wallet_balance(self) -> Dict[str, Any]:
        """
        Get wallet balance from balance_manager (L2).
        """
        return await MarketAccountEngineImpl.get_wallet_balance(self.app_ctx)
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Wire SituationEngine.get_portfolio_snapshot()
# ═════════════════════════════════════════════════════════════════════════════

SITUATION_ENGINE_WIRING = '''
# In core_engine/situation_engine.py, replace the get_portfolio_snapshot method:

from core_engine.implementations import SituationEngineImpl

class SituationEngine:
    """..existing docstring..."""

    async def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        Get portfolio state from portfolio_manager (L3).
        Real implementation calls L3 component.
        """
        return await SituationEngineImpl.get_portfolio_snapshot(self.app_ctx)

    async def get_all_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get signals from signal_manager (L5).
        """
        return await SituationEngineImpl.get_all_signals(self.app_ctx, symbol)

    async def get_fused_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fused signal from signal_fusion (L5).
        """
        return await SituationEngineImpl.get_fused_signal(self.app_ctx, symbol)

    async def get_market_regime(self) -> Dict[str, str]:
        """
        Get market regime from regime_detector (L2).
        """
        return await SituationEngineImpl.get_market_regime(self.app_ctx)
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Wire DecisionEngine.make_buy_decision()
# ═════════════════════════════════════════════════════════════════════════════

DECISION_ENGINE_WIRING = '''
# In core_engine/decision_engine.py, replace the make_buy_decision method:

from core_engine.implementations import DecisionEngineImpl

class DecisionEngine:
    """..existing docstring..."""

    async def get_current_mode(self) -> str:
        """
        Get current trading mode from mode_manager (L5).
        """
        return await DecisionEngineImpl.get_current_mode(self.app_ctx)

    async def evaluate_signal(
        self, symbol: str, signal_type: str, edge_score: float
    ) -> Dict[str, Any]:
        """
        Evaluate signal through 6-layer arbitration gates.
        """
        return await DecisionEngineImpl.evaluate_signal(
            self.app_ctx, symbol, signal_type, edge_score
        )

    async def make_buy_decision(self, symbol: str, edge_score: float) -> Optional[Dict[str, Any]]:
        """
        Make buy decision with capital allocation (L5/L6).
        """
        return await DecisionEngineImpl.make_buy_decision(self.app_ctx, symbol, edge_score)

    async def make_sell_decision(self, symbol: str, edge_score: float) -> Optional[Dict[str, Any]]:
        """
        Make sell decision (symmetric to make_buy_decision).
        """
        return await DecisionEngineImpl.make_sell_decision(self.app_ctx, symbol, edge_score)
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Wire SafeExecutionEngine.place_sell_order() [FIX #2]
# ═════════════════════════════════════════════════════════════════════════════

SAFE_EXECUTION_ENGINE_WIRING = '''
# In core_engine/safe_execution_engine.py, replace the place_sell_order method:

from core_engine.implementations import SafeExecutionEngineImpl

class SafeExecutionEngine:
    """..existing docstring..."""

    async def validate_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate order with comprehensive checks.
        """
        return await SafeExecutionEngineImpl.validate_order(
            self.app_ctx, symbol, action, quantity, price
        )

    async def place_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        """
        Place BUY order via execution_manager (L4).
        """
        return await SafeExecutionEngineImpl.place_buy_order(
            self.app_ctx, symbol, quantity, price, order_type
        )

    async def place_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        """
        Place SELL order with FIX #2 idempotent guard.

        ⭐ CRITICAL: This method includes FIX #2 guard implementation!
           - Checks bounded_cache for duplicate sell prevention
           - Prevents double-sells on system recovery
           - Marks completion in cache with 5-minute TTL
        """
        return await SafeExecutionEngineImpl.place_sell_order(
            self.app_ctx, symbol, quantity, price, order_type
        )
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Wire OperationsEngine.startup_system()
# ═════════════════════════════════════════════════════════════════════════════

OPERATIONS_ENGINE_WIRING = '''
# In core_engine/operations_engine.py, replace the startup_system method:

from core_engine.implementations import OperationsEngineImpl

class OperationsEngine:
    """..existing docstring..."""

    async def startup_system(self) -> bool:
        """
        Execute system startup (L0→L8).
        """
        return await OperationsEngineImpl.startup_system(self.app_ctx)

    async def get_health_report(self) -> Dict[str, Any]:
        """
        Get health status from health_monitor (L7).
        """
        return await OperationsEngineImpl.get_health_report(self.app_ctx)
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Create app_ctx with Real Components
# ═════════════════════════════════════════════════════════════════════════════

APP_CONTEXT_SETUP = '''
# In your orchestrator or main startup file:

import asyncio
from core_engine import (
    MarketAccountEngine,
    SituationEngine,
    DecisionEngine,
    SafeExecutionEngine,
    OperationsEngine,
)


async def create_app_context_with_real_components():
    """
    Create app_ctx dictionary with real component instances.
    This wires the 5 engines to actual L0-L8 components.
    """

    # Import real components from src/
    from src.l0_core.shared_state import SharedState
    from src.l1_exchange.exchange_client import ExchangeClient
    from src.l2_marketdata.market_data_feed import MarketDataFeed
    from src.l2_marketdata.balance_manager import BalanceManager
    from src.l3_portfolio.portfolio_manager import PortfolioManager
    from src.l4_execution.execution_manager import ExecutionManager
    from src.l5_strategy.signal_fusion import SignalFusion
    from src.l5_strategy.arbitration_engine import ArbitrationEngine
    from src.l5_strategy.mode_manager import ModeManager
    from src.l6_governance.capital_allocator import CapitalAllocator
    from src.l7_observability.health_monitor import HealthMonitor
    from src.l8_lifecycle.startup_orchestrator import StartupOrchestrator

    # Initialize components
    shared_state = SharedState()
    exchange_client = ExchangeClient(api_key="...", api_secret="...")
    market_data_feed = MarketDataFeed()
    balance_manager = BalanceManager(exchange_client)
    portfolio_manager = PortfolioManager(shared_state)
    signal_fusion = SignalFusion()
    arbitration_engine = ArbitrationEngine()
    execution_manager = ExecutionManager(exchange_client)
    mode_manager = ModeManager(shared_state)
    capital_allocator = CapitalAllocator(portfolio_manager)
    health_monitor = HealthMonitor()
    startup_orchestrator = StartupOrchestrator()

    # Create app_ctx
    app_ctx = {
        "shared_state": shared_state,
        "exchange_client": exchange_client,
        "market_data_feed": market_data_feed,
        "balance_manager": balance_manager,
        "portfolio_manager": portfolio_manager,
        "signal_fusion": signal_fusion,
        "arbitration_engine": arbitration_engine,
        "execution_manager": execution_manager,
        "mode_manager": mode_manager,
        "capital_allocator": capital_allocator,
        "health_monitor": health_monitor,
        "startup_orchestrator": startup_orchestrator,
    }

    return app_ctx


async def initialize_engines():
    """
    Create and initialize all 5 engines with real components.
    """
    app_ctx = await create_app_context_with_real_components()

    # Create engines
    engines = {
        "market_account": MarketAccountEngine(app_ctx),
        "situation": SituationEngine(app_ctx),
        "decision": DecisionEngine(app_ctx),
        "safe_execution": SafeExecutionEngine(app_ctx),
        "operations": OperationsEngine(app_ctx),
    }

    # Start system
    await engines["operations"].startup_system()

    return engines


async def main():
    """Main entry point - demonstrates full engine initialization."""
    engines = await initialize_engines()

    # Test READ function
    account_state = await engines["market_account"].get_account_state()
    print(f"✅ Account state: {account_state}")

    # Test UNDERSTAND function
    portfolio = await engines["situation"].get_portfolio_snapshot()
    print(f"✅ Portfolio: {portfolio}")

    # Test DECIDE function
    buy_decision = await engines["decision"].make_buy_decision("BTCUSDT", edge_score=0.7)
    print(f"✅ Buy decision: {buy_decision}")

    # Test EXECUTE function
    if buy_decision:
        result = await engines["safe_execution"].place_buy_order(
            symbol=buy_decision["symbol"],
            quantity=buy_decision["quantity"],
        )
        print(f"✅ Execute result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
'''


# ═════════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: Integration Test
# ═════════════════════════════════════════════════════════════════════════════

INTEGRATION_TEST = '''
# File: core_engine/tests/test_engines_integration.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from core_engine import (
    MarketAccountEngine,
    SituationEngine,
    DecisionEngine,
    SafeExecutionEngine,
    OperationsEngine,
)


@pytest.fixture
def mock_app_ctx():
    """Create mock app context with all required components."""
    return {
        "exchange_client": AsyncMock(),
        "market_data_feed": AsyncMock(),
        "balance_manager": AsyncMock(),
        "portfolio_manager": AsyncMock(),
        "signal_fusion": AsyncMock(),
        "arbitration_engine": AsyncMock(),
        "execution_manager": AsyncMock(),
        "mode_manager": AsyncMock(),
        "capital_allocator": AsyncMock(),
        "health_monitor": AsyncMock(),
        "startup_orchestrator": AsyncMock(),
        "bounded_cache": AsyncMock(),
    }


@pytest.mark.asyncio
async def test_market_account_engine_get_account_state(mock_app_ctx):
    """Test MarketAccountEngine.get_account_state() calls exchange_client."""
    mock_app_ctx["exchange_client"].get_account = AsyncMock(return_value={
        "balances": [{"asset": "BTC", "free": "1.0", "locked": "0.0"}]
    })

    engine = MarketAccountEngine(mock_app_ctx)
    result = await engine.get_account_state()

    assert result is not None
    assert "timestamp" in result
    mock_app_ctx["exchange_client"].get_account.assert_called_once()


@pytest.mark.asyncio
async def test_situation_engine_get_portfolio_snapshot(mock_app_ctx):
    """Test SituationEngine.get_portfolio_snapshot() calls portfolio_manager."""
    mock_app_ctx["portfolio_manager"].get_nav = AsyncMock(return_value=10000.0)
    mock_app_ctx["portfolio_manager"].get_positions = AsyncMock(return_value=[])

    engine = SituationEngine(mock_app_ctx)
    result = await engine.get_portfolio_snapshot()

    assert result["nav_usdt"] == 10000.0
    mock_app_ctx["portfolio_manager"].get_nav.assert_called_once()


@pytest.mark.asyncio
async def test_decision_engine_make_buy_decision(mock_app_ctx):
    """Test DecisionEngine.make_buy_decision() with arbitration."""
    mock_app_ctx["arbitration_engine"].evaluate = AsyncMock(return_value={
        "passed": True,
        "gates_status": {},
        "blocking_gates": [],
    })
    mock_app_ctx["capital_allocator"].allocate_for_buy = AsyncMock(return_value=0.1)
    mock_app_ctx["mode_manager"].get_current_mode = AsyncMock(return_value="GROWTH")

    engine = DecisionEngine(mock_app_ctx)
    result = await engine.make_buy_decision("BTCUSDT", edge_score=0.8)

    assert result is not None
    assert result["action"] == "BUY"
    assert result["quantity"] == 0.1


@pytest.mark.asyncio
async def test_safe_execution_engine_place_sell_order_with_fix2(mock_app_ctx):
    """Test SafeExecutionEngine.place_sell_order() with FIX #2 guard."""
    mock_app_ctx["bounded_cache"].get = AsyncMock(return_value=None)  # Not finalized
    mock_app_ctx["bounded_cache"].set = AsyncMock(return_value=True)
    mock_app_ctx["execution_manager"].place_order = AsyncMock(return_value={
        "orderId": "12345",
        "price": 50000.0,
        "executedQty": 0.1,
    })

    engine = SafeExecutionEngine(mock_app_ctx)
    result = await engine.place_sell_order("BTCUSDT", quantity=0.1, price=50000.0)

    assert result["success"] is True
    assert result["order_id"] == "12345"
    # Verify FIX #2 cache set was called
    mock_app_ctx["bounded_cache"].set.assert_called_once()


@pytest.mark.asyncio
async def test_fix2_duplicate_sell_prevention(mock_app_ctx):
    """Test FIX #2: Prevent duplicate SELL on crash recovery."""
    # First call: normal execution
    mock_app_ctx["bounded_cache"].get = AsyncMock(return_value=None)
    mock_app_ctx["bounded_cache"].set = AsyncMock(return_value=True)

    engine = SafeExecutionEngine(mock_app_ctx)
    result1 = await engine.place_sell_order("BTCUSDT", quantity=0.1, price=50000.0)
    assert result1["success"] is True

    # Second call: already finalized (duplicate)
    mock_app_ctx["bounded_cache"].get = AsyncMock(return_value=True)  # Already cached

    result2 = await engine.place_sell_order("BTCUSDT", quantity=0.1, price=50000.0)
    assert result2["status"] == "ALREADY_FINALIZED"
    print("✅ FIX #2 guard working: duplicate SELL prevented")


@pytest.mark.asyncio
async def test_operations_engine_startup_system(mock_app_ctx):
    """Test OperationsEngine.startup_system() initializes L0→L8."""
    mock_app_ctx["startup_orchestrator"].startup = AsyncMock(return_value=True)

    engine = OperationsEngine(mock_app_ctx)
    result = await engine.startup_system()

    assert result is True
    mock_app_ctx["startup_orchestrator"].startup.assert_called_once()
'''


# Print all examples
if __name__ == "__main__":
    print("=" * 80)
    print("CONCRETE WIRING EXAMPLES - COPY/PASTE READY")
    print("=" * 80)
    print("\n1. MARKET_ACCOUNT_ENGINE_WIRING:")
    print(MARKET_ACCOUNT_ENGINE_WIRING)
    print("\n" + "=" * 80)
    print("2. SITUATION_ENGINE_WIRING:")
    print(SITUATION_ENGINE_WIRING)
    print("\n" + "=" * 80)
    print("3. DECISION_ENGINE_WIRING:")
    print(DECISION_ENGINE_WIRING)
    print("\n" + "=" * 80)
    print("4. SAFE_EXECUTION_ENGINE_WIRING:")
    print(SAFE_EXECUTION_ENGINE_WIRING)
    print("\n" + "=" * 80)
    print("5. OPERATIONS_ENGINE_WIRING:")
    print(OPERATIONS_ENGINE_WIRING)
    print("\n" + "=" * 80)
    print("6. APP_CONTEXT_SETUP:")
    print(APP_CONTEXT_SETUP)
    print("\n" + "=" * 80)
    print("7. INTEGRATION_TEST:")
    print(INTEGRATION_TEST)
