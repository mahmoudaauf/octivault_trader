"""
Core Engines Façade
═══════════════════

5 Abstraction Layers for the 5 Core Functions

1. MarketAccountEngine     — READ market/account
2. SituationEngine         — UNDERSTAND situation
3. DecisionEngine          — DECIDE what to do
4. SafeExecutionEngine     — EXECUTE safely
5. OperationsEngine        — RECOVER/monitor

This package provides high-level abstraction over the entire system.
Each engine coordinates multiple layers (L0-L8) to fulfill a single
core responsibility.

## Architecture

### Traditional View (By Layer)
    L0: Core infrastructure
    L1: Exchange I/O
    L2: Market data & wallet
    L3: Portfolio state
    L4: Order execution
    L5: Strategy & signals
    L6: Governance & policy
    L7: Observability
    L8: Lifecycle

### Functional View (By Core Function)
    ┌──────────────────────────────────────────────┐
    │ MARKET_ACCOUNT_ENGINE (1. READ)              │
    │ - exchange_client (L1)                       │
    │ - market_data_feed (L2)                      │
    │ - balance_manager (L2)                       │
    │ - ws_market_data (L1)                        │
    └──────────────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────────────┐
    │ SITUATION_ENGINE (2. UNDERSTAND)             │
    │ - portfolio_manager (L3)                     │
    │ - signal_manager (L5)                        │
    │ - signal_fusion (L5)                         │
    │ - market_regime_detector (L2)                │
    │ - anomaly_detection (L2)                     │
    │ - all agents (L5)                            │
    └──────────────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────────────┐
    │ DECISION_ENGINE (3. DECIDE)                  │
    │ - meta_controller (L8)                       │
    │ - arbitration_engine (L5)                    │
    │ - mode_manager (L5)                          │
    │ - capital_allocator (L6)                     │
    │ - policy_manager (L6)                        │
    │ - execution_logic (L4)                       │
    └──────────────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────────────┐
    │ SAFE_EXECUTION_ENGINE (4. EXECUTE)           │
    │ - execution_manager (L4)                     │
    │ - exchange_client (L1)                       │
    │ - bounded_cache (L0) ← FIX #2                │
    │ - safety_order_manager (L4)                  │
    │ - leverage_manager (L4)                      │
    │ - error_handler (L0)                         │
    └──────────────────────────────────────────────┘
              ↓
    ┌──────────────────────────────────────────────┐
    │ OPERATIONS_ENGINE (5. RECOVER/MONITOR)       │
    │ - health_monitor (L7)                        │
    │ - watchdog (L7)                              │
    │ - state_manager (L3)                         │
    │ - recovery_engine (L3)                       │
    │ - startup_orchestrator (L8)                  │
    │ - event_store (L3)                           │
    │ - prometheus_exporter (L7)                   │
    └──────────────────────────────────────────────┘

## Usage

```python
from core_engine import (
    MarketAccountEngine,
    SituationEngine,
    DecisionEngine,
    SafeExecutionEngine,
    OperationsEngine,
)

# Initialize all engines
app_ctx = {...}  # Application context with all components

market_engine = MarketAccountEngine(app_ctx)
situation_engine = SituationEngine(app_ctx)
decision_engine = DecisionEngine(app_ctx)
execution_engine = SafeExecutionEngine(app_ctx)
operations_engine = OperationsEngine(app_ctx)

# Startup
await operations_engine.startup_system()

# Main trading loop
while True:
    # 1. Read market/account
    account = await market_engine.get_account_state()
    prices = await market_engine.get_market_prices()

    # 2. Understand situation
    portfolio = await situation_engine.get_portfolio_snapshot()
    signals = await situation_engine.get_all_signals()
    regime = await situation_engine.get_market_regime()

    # 3. Decide what to do
    decision = await decision_engine.make_buy_decision("BTCUSDT", 0.45)

    # 4. Execute safely
    if decision:
        result = await execution_engine.place_buy_order(
            decision.symbol,
            decision.quantity,
            decision.price_target
        )

    # 5. Monitor health
    health = await operations_engine.get_health_report()
    if health.overall_status != HealthStatus.OK:
        await operations_engine.recover_state()

# Shutdown
await operations_engine.shutdown_system()
```

## Benefits

1. **Separation of Concerns**: Each engine has one responsibility
2. **Composability**: Engines work together cleanly
3. **Testability**: Each engine can be tested independently
4. **Clarity**: Clear data flows and responsibilities
5. **Maintainability**: Adding features is predictable
6. **Scalability**: Easy to add observability, caching, etc.
"""

from __future__ import annotations

from core_engine.decision_engine import DecisionEngine
from core_engine.market_account_engine import MarketAccountEngine
from core_engine.operations_engine import HealthReport, HealthStatus, OperationsEngine
from core_engine.safe_execution_engine import SafeExecutionEngine
from core_engine.situation_engine import SituationEngine

__all__ = [
    # Engines
    "MarketAccountEngine",
    "SituationEngine",
    "DecisionEngine",
    "SafeExecutionEngine",
    "OperationsEngine",
    # Status enums
    "HealthStatus",
    "HealthReport",
]

__version__ = "1.0.0"
__description__ = "5-Layer Façade for Core Trading Functions"
