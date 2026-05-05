"""
╔════════════════════════════════════════════════════════════════════════════╗
║              QUICK REFERENCE: 5 CORE_ENGINE FAÇADE ENGINES                ║
║                                                                            ║
║  Created: May 5, 2026                                                     ║
║  Location: /core_engine/                                                  ║
║  Status: ✅ READY FOR INTEGRATION                                         ║
╚════════════════════════════════════════════════════════════════════════════╝


📚 QUICK IMPORT

from core_engine import (
    MarketAccountEngine,
    SituationEngine,
    DecisionEngine,
    SafeExecutionEngine,
    OperationsEngine,
    HealthStatus,
    HealthReport,
)


🏗️  ARCHITECTURE OVERVIEW

Traditional View (By Layer):
    L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8

Functional View (By Core Function):
    [READ] → [UNDERSTAND] → [DECIDE] → [EXECUTE] → [RECOVER/MONITOR] → Loop

Each engine abstracts multiple layers into a single responsibility.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🔴 ENGINE #1: MarketAccountEngine (READ market/account)
─────────────────────────────────────────────────────────

Purpose:     Data ingestion from exchange
Coordinates: 4 components (exchange_client, market_data_feed, balance_manager, ws_market_data)

Key Methods:
  • get_account_state()              → {balances, positions, orders, timestamp}
  • get_market_prices(symbols)       → {symbol: price}
  • get_ohlcv_data(symbol, tf)       → [candles]
  • get_wallet_balance()             → {total, available, locked}
  • subscribe_to_market_updates()    → stream updates
  • sync_balance_with_exchange()     → bool

Data Returned:
  • account_state: Dict with balances, positions, open orders
  • prices: Dict[symbol] = float
  • ohlcv: List of candle dicts with open, high, low, close, volume
  • wallet: Dict with USDT equivalents and asset balances

Example:
  market = MarketAccountEngine(app_ctx)
  account = await market.get_account_state()
  prices = await market.get_market_prices(["BTCUSDT", "ETHUSDT"])
  await market.sync_balance_with_exchange()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🟡 ENGINE #2: SituationEngine (UNDERSTAND situation)
────────────────────────────────────────────────────

Purpose:     Analysis and synthesis of portfolio & market state
Coordinates: 7 components (portfolio_manager, signal_manager, signal_fusion, etc.)

Key Methods:
  • get_portfolio_snapshot()         → PortfolioSnapshot
  • get_all_signals(symbol)          → [SignalScore]
  • get_fused_signal(symbol)         → SignalScore
  • get_market_regime()              → RegimeState
  • detect_anomalies()               → {spikes, liquidations, volume}
  • get_position_analysis(symbol)    → {qty, entry, current, P&L, risk}
  • get_capital_efficiency()         → {total, active, reserve, util%}
  • get_risk_assessment()            → {overall, liquidation, concentration}

Data Classes:
  • PortfolioSnapshot:
      - nav_usdt, available_capital, locked_capital
      - active_positions, total_p_and_l, total_p_and_l_pct

  • SignalScore:
      - symbol, signal_type (BUY/SELL/HOLD)
      - edge_score (-1.0 to +1.0)
      - confidence (0.0 to 1.0)
      - agent_name, timestamp

  • RegimeState:
      - volatility_regime (LOW/NORMAL/HIGH)
      - trend_regime (UPTREND/DOWNTREND/RANGING)
      - nav_regime (GROWTH/DECAY)
      - overall_health (OK/WARN/CRISIS)

Example:
  situation = SituationEngine(app_ctx)
  portfolio = await situation.get_portfolio_snapshot()
  signals = await situation.get_all_signals()
  regime = await situation.get_market_regime()
  fused = await situation.get_fused_signal("BTCUSDT")
  anomalies = await situation.detect_anomalies()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🟢 ENGINE #3: DecisionEngine (DECIDE what to do)
─────────────────────────────────────────────────

Purpose:     Trading decision logic with multi-layer safety gates
Coordinates: 6 components (meta_controller, arbitration_engine, mode_manager, etc.)

Key Methods:
  • get_current_mode()               → mode_string
  • set_mode(mode)                   → bool
  • evaluate_signal(symbol, type, edge) → ArbitrationResult
  • make_buy_decision(symbol, edge)  → TradeDecision
  • make_sell_decision(symbol, edge) → TradeDecision
  • evaluate_exit_signals(symbol)    → TradeDecision
  • apply_policy_constraints()       → bool
  • get_mode_constraints()           → {mode, allow_new, max_pos, conf_floor}

Multi-Layer Arbitration Gates (6 layers):
  1. Symbol format validation
  2. Confidence floor (mode-dependent)
  3. Market regime check
  4. Position limit check
  5. Capital available check
  6. Risk manager approval

Data Classes:
  • TradeDecision:
      - symbol, action (BUY/SELL/HOLD/FORCE_EXIT)
      - quantity, price_target, stop_loss, take_profit
      - reason, confidence, timestamp, mode

  • ArbitrationResult:
      - passed: bool
      - gates_status: Dict[gate_name: bool]
      - blocking_gates: [gate_names]

Trading Modes:
  • PAUSED: No trading allowed
  • PROTECTIVE: Minimal trades, small sizes
  • BOOTSTRAP: Building initial positions
  • GROWTH: Active trading with growth focus
  • CRISIS: Emergency mode, close positions only

Example:
  decision = DecisionEngine(app_ctx)

  # Check arbitration gates
  result = await decision.evaluate_signal("BTCUSDT", "BUY", 0.45)
  if result.passed:
      # Make buy decision
      buy_decision = await decision.make_buy_decision("BTCUSDT", 0.45)
      if buy_decision:
          execute_order(buy_decision)

  # Get current mode
  mode = await decision.get_current_mode()

  # Change mode
  await decision.set_mode("GROWTH")


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🔵 ENGINE #4: SafeExecutionEngine (EXECUTE safely)
──────────────────────────────────────────────────

Purpose:     Safe order placement with comprehensive guards
Coordinates: 6 components (execution_manager, exchange_client, bounded_cache, etc.)

Key Methods:
  • validate_order(symbol, action, qty, price) → OrderValidation
  • place_buy_order(symbol, qty, price, type)   → ExecutionResult
  • place_sell_order(symbol, qty, price, type)  → ExecutionResult ⚠️ FIX #2!
  • place_safety_order(symbol, qty, TP, SL)    → ExecutionResult
  • get_order_status(symbol, order_id)          → {status, filled, price}
  • cancel_order(symbol, order_id)              → bool

Safety Checks (in validate_order):
  ✓ Price > 0
  ✓ Quantity > 0
  ✓ Notional (qty × price) >= MIN_NOTIONAL (10 USDT)
  ✓ Step size alignment (ROUND UP for conservative)
  ✓ Slippage modeling (10 bps default: worse fill)
  ✓ Margin/leverage within limits

FIX #2: Idempotent SELL Guard
  • Located in: place_sell_order()
  • Mechanism: BoundedCache (L0)
  • Key: "sell_finalize_{symbol}_{order_id}"
  • Prevents: Duplicate SELL finalization
  • Redundancy: Called 10x by execution_manager

Data Classes:
  • OrderValidation:
      - valid: bool
      - errors: [str]
      - warnings: [str]

  • ExecutionResult:
      - success: bool
      - order_id, symbol, action (BUY/SELL)
      - quantity, filled_quantity, average_price
      - status (PENDING/PARTIALLY_FILLED/FILLED/FAILED)
      - error_message, timestamp

Example:
  execution = SafeExecutionEngine(app_ctx)

  # Validate order
  validation = await execution.validate_order("BTCUSDT", "BUY", 0.1, 42000)
  if not validation.valid:
      print(f"Validation errors: {validation.errors}")
      return

  # Place BUY order
  result = await execution.place_buy_order("BTCUSDT", 0.1, 42000, order_type="LIMIT")
  if result.success:
      print(f"✅ Order placed: {result.order_id}")
  else:
      print(f"❌ Order failed: {result.error_message}")

  # Place SELL order (with FIX #2 guard)
  sell_result = await execution.place_sell_order("BTCUSDT", 0.1, 43000, "LIMIT")
  # The guard prevents duplicate finalization even if called 10x


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


🟣 ENGINE #5: OperationsEngine (RECOVER/MONITOR)
────────────────────────────────────────────────

Purpose:     System health, recovery, and observability
Coordinates: 7 components (health_monitor, watchdog, state_manager, etc.)

Key Methods:
  • startup_system()                 → bool
  • shutdown_system()                → bool
  • get_health_report()              → HealthReport
  • check_liveness()                 → bool
  • detect_anomalies()               → [anomaly_strings]
  • save_state()                     → bool
  • recover_state()                  → RecoveryPlan
  • apply_recovery(plan)             → bool
  • export_metrics()                 → {metrics}
  • log_event(type, details)         → None
  • get_event_history(type, limit)   → [events]
  • get_uptime()                     → seconds
  • get_performance_stats()          → {latency, orders, trades, errors}

Startup Sequence (L0→L8):
  0. Core infrastructure (L0)
  1. Exchange I/O (L1)
  2. Market data & wallet (L2)
  3. Portfolio state (L3)
  4. Order execution (L4)
  5. Strategy & signals (L5)
  6. Governance & policy (L6)
  7. Observability (L7)
  8. Lifecycle & orchestration (L8)

Data Classes:
  • ComponentStatus:
      - name, status (HealthStatus enum)
      - uptime_seconds, last_update
      - error_count, warning_count
      - details: Dict

  • HealthReport:
      - timestamp, overall_status (HealthStatus enum)
      - components: Dict[name: ComponentStatus]
      - critical_issues: [str]
      - warnings: [str]
      - suggestions: [str]

  • RecoveryPlan:
      - issues: [str]
      - recovery_steps: [str]
      - estimated_recovery_time_sec: float
      - priority (IMMEDIATE/URGENT/HIGH/NORMAL)
      - auto_recover: bool

Enums:
  • HealthStatus: OK, WARN, ERROR, CRITICAL

Example:
  ops = OperationsEngine(app_ctx)

  # Startup system
  if not await ops.startup_system():
      print("❌ Startup failed")
      return

  # Check health
  health = await ops.get_health_report()
  print(f"Overall health: {health.overall_status}")

  if health.overall_status == HealthStatus.CRITICAL:
      # Generate and apply recovery
      plan = await ops.recover_state()
      await ops.apply_recovery(plan)

  # Save state
  await ops.save_state()

  # Get metrics
  metrics = await ops.export_metrics()

  # Get uptime
  uptime = await ops.get_uptime()
  print(f"System uptime: {uptime:.0f}s")

  # Shutdown
  await ops.shutdown_system()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


📦 COMPLETE WORKFLOW EXAMPLE

import asyncio
from core_engine import (
    MarketAccountEngine,
    SituationEngine,
    DecisionEngine,
    SafeExecutionEngine,
    OperationsEngine,
    HealthStatus,
)

async def main():
    # Setup
    app_ctx = get_app_context()  # Your app context

    # Initialize engines
    market = MarketAccountEngine(app_ctx)
    situation = SituationEngine(app_ctx)
    decision = DecisionEngine(app_ctx)
    execution = SafeExecutionEngine(app_ctx)
    ops = OperationsEngine(app_ctx)

    # Startup
    if not await ops.startup_system():
        print("❌ Startup failed")
        return

    try:
        # Main trading loop
        while True:
            # 1. READ market/account
            account = await market.get_account_state()
            prices = await market.get_market_prices()

            # 2. UNDERSTAND situation
            portfolio = await situation.get_portfolio_snapshot()
            signals = await situation.get_all_signals()
            regime = await situation.get_market_regime()

            # 3. DECIDE what to do
            for signal in signals[:5]:  # Top 5 signals
                if signal.signal_type == "BUY":
                    buy_decision = await decision.make_buy_decision(
                        signal.symbol,
                        signal.edge_score
                    )

                    if buy_decision:
                        # 4. EXECUTE safely
                        result = await execution.place_buy_order(
                            buy_decision.symbol,
                            buy_decision.quantity,
                            buy_decision.price_target
                        )

                        # 5. MONITOR & RECOVER
                        if result.success:
                            await ops.log_event("BUY_ORDER", {
                                "symbol": result.symbol,
                                "quantity": result.quantity,
                                "order_id": result.order_id,
                            })

            # Health check
            health = await ops.get_health_report()
            if health.overall_status == HealthStatus.CRITICAL:
                print("🚨 CRITICAL: Attempting recovery...")
                plan = await ops.recover_state()
                await ops.apply_recovery(plan)

            await asyncio.sleep(2)  # 2-second cycle

    except KeyboardInterrupt:
        print("⏹️  Shutting down...")
    finally:
        await ops.shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


📊 FILE STATISTICS

Location:               /core_engine/
Total Files:           6
Total Lines of Code:   2,140
Documentation:         Comprehensive

Breakdown:
  market_account_engine.py:   ~230 lines
  situation_engine.py:        ~350 lines
  decision_engine.py:         ~380 lines
  safe_execution_engine.py:   ~420 lines
  operations_engine.py:       ~520 lines
  __init__.py:                ~130 lines


✅ VALIDATION

Syntax Check:           ✅ PASSED
  └─ All 6 files compile successfully

Type Hints:             ✅ Complete
  └─ Full type annotations for clarity

Documentation:          ✅ Comprehensive
  └─ Docstrings, examples, data class docs

Import Test:            ✅ Ready
  └─ from core_engine import *


🚀 NEXT STEPS

1. Integration: Wire engines to actual L0-L8 components
2. Testing: Create integration and unit tests
3. Performance: Benchmark and profile
4. Deployment: Roll out to production
5. Monitoring: Add observability metrics


═══════════════════════════════════════════════════════════════════════════════

                         STATUS: ✅ PHASE 1 COMPLETE
                            Ready for integration
                          Created: May 5, 2026

═══════════════════════════════════════════════════════════════════════════════
"""
