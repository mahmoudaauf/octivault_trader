"""
PHASE 2 Integration Guide
═════════════════════════════════════════════════════════════════════════════

HOW TO INTEGRATE: 5 Engines → Real Components

This file provides step-by-step integration instructions.
"""

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Wire MarketAccountEngine to L1/L2
# ═════════════════════════════════════════════════════════════════════════════

"""
TARGET: Connect MarketAccountEngine to exchange data sources

CURRENT STATE:
  File: core_engine/market_account_engine.py
  Methods: get_account_state, get_market_prices, get_wallet_balance, etc.
  Implementation: Uses placeholder asyncio.sleep()

WIRING STEPS:

  1. Replace placeholder in get_account_state():
     FROM: await asyncio.sleep(0.1)
     TO:   Use MarketAccountEngineImpl.get_account_state(app_ctx)

     # Copy this into market_account_engine.py:
     from core_engine.implementations import MarketAccountEngineImpl

     async def get_account_state(self):
         return await MarketAccountEngineImpl.get_account_state(self.app_ctx)

  2. Replace placeholder in get_market_prices():
     FROM: await asyncio.sleep(0.1)
     TO:   Use MarketAccountEngineImpl.get_market_prices()

     async def get_market_prices(self, symbols):
         return await MarketAccountEngineImpl.get_market_prices(self.app_ctx, symbols)

  3. Replace placeholder in get_wallet_balance():
     FROM: await asyncio.sleep(0.1)
     TO:   Use MarketAccountEngineImpl.get_wallet_balance()

     async def get_wallet_balance(self):
         return await MarketAccountEngineImpl.get_wallet_balance(self.app_ctx)

COMPONENT INTERFACES REQUIRED:

  exchange_client (L1):
    - async get_account() → Dict with "balances" and "positions"
    - async get_prices(symbols) → Dict[symbol: price]
    - async get_open_orders() → List[order]

  market_data_feed (L2):
    - async get_prices(symbols) → Dict[symbol: price]
    - async get_ohlcv(symbol) → List[OHLCV]

  balance_manager (L2):
    - async get_balance() → Dict with wallet summary

TESTING:
  pytest core_engine/tests/test_market_account_integration.py -v
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Wire SituationEngine to L3/L5
# ═════════════════════════════════════════════════════════════════════════════

"""
TARGET: Connect SituationEngine to portfolio and signal fusion

CURRENT STATE:
  File: core_engine/situation_engine.py
  Methods: get_portfolio_snapshot, get_all_signals, get_fused_signal, get_market_regime
  Implementation: Uses placeholder asyncio.sleep()

WIRING STEPS:

  1. Replace placeholder in get_portfolio_snapshot():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SituationEngineImpl.get_portfolio_snapshot()

     # Copy this into situation_engine.py:
     from core_engine.implementations import SituationEngineImpl

     async def get_portfolio_snapshot(self):
         return await SituationEngineImpl.get_portfolio_snapshot(self.app_ctx)

  2. Replace placeholder in get_all_signals():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SituationEngineImpl.get_all_signals()

     async def get_all_signals(self, symbol=None):
         return await SituationEngineImpl.get_all_signals(self.app_ctx, symbol)

  3. Replace placeholder in get_fused_signal():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SituationEngineImpl.get_fused_signal()

     async def get_fused_signal(self, symbol):
         return await SituationEngineImpl.get_fused_signal(self.app_ctx, symbol)

  4. Replace placeholder in get_market_regime():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SituationEngineImpl.get_market_regime()

     async def get_market_regime(self):
         return await SituationEngineImpl.get_market_regime(self.app_ctx)

COMPONENT INTERFACES REQUIRED:

  portfolio_manager (L3):
    - async get_nav() → float (portfolio NAV in USDT)
    - async get_positions() → List[position]
    - async get_pnl() → float (total P&L)
    - async get_capital_allocated() → float
    - async get_capital_available() → float

  signal_manager (L5):
    - async get_signals(symbol) → List[signal]
    - async get_all_signals() → List[signal]

  signal_fusion (L5):
    - async fuse_signal(symbol) → Dict with fused edge/confidence

  market_regime_detector (L2):
    - async get_regime() → Dict with volatility/trend regimes

TESTING:
  pytest core_engine/tests/test_situation_integration.py -v
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Wire DecisionEngine to L5/L6
# ═════════════════════════════════════════════════════════════════════════════

"""
TARGET: Connect DecisionEngine to arbitration and capital allocation

CURRENT STATE:
  File: core_engine/decision_engine.py
  Methods: get_current_mode, evaluate_signal, make_buy_decision, make_sell_decision
  Implementation: Uses placeholder asyncio.sleep()

WIRING STEPS:

  1. Replace placeholder in get_current_mode():
     FROM: await asyncio.sleep(0.1)
     TO:   Use DecisionEngineImpl.get_current_mode()

     # Copy this into decision_engine.py:
     from core_engine.implementations import DecisionEngineImpl

     async def get_current_mode(self):
         return await DecisionEngineImpl.get_current_mode(self.app_ctx)

  2. Replace placeholder in evaluate_signal():
     FROM: await asyncio.sleep(0.1)
     TO:   Use DecisionEngineImpl.evaluate_signal()

     async def evaluate_signal(self, symbol, signal_type, edge_score):
         return await DecisionEngineImpl.evaluate_signal(
             self.app_ctx, symbol, signal_type, edge_score
         )

  3. Replace placeholder in make_buy_decision():
     FROM: await asyncio.sleep(0.1)
     TO:   Use DecisionEngineImpl.make_buy_decision()

     async def make_buy_decision(self, symbol, edge_score):
         return await DecisionEngineImpl.make_buy_decision(self.app_ctx, symbol, edge_score)

  4. Replace placeholder in make_sell_decision():
     FROM: await asyncio.sleep(0.1)
     TO:   Similar pattern

COMPONENT INTERFACES REQUIRED:

  arbitration_engine (L5):
    - async evaluate(symbol, signal_type, edge_score) → Dict with gates_status

  mode_manager (L5):
    - async get_current_mode() → str ("PROTECTIVE", "GROWTH", etc.)
    - async get_constraints() → Dict with mode constraints

  capital_allocator (L6):
    - async allocate_for_buy(symbol) → float (quantity to buy)
    - async allocate_for_sell(symbol) → float (quantity to sell)

TESTING:
  pytest core_engine/tests/test_decision_integration.py -v
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Wire SafeExecutionEngine to L4 + L0 [FIX #2]
# ═════════════════════════════════════════════════════════════════════════════

"""
TARGET: Connect SafeExecutionEngine to execution manager + bounded cache (FIX #2)

CURRENT STATE:
  File: core_engine/safe_execution_engine.py
  Methods: validate_order, place_buy_order, place_sell_order (with FIX #2)
  Implementation: Partial implementation with FIX #2 guard pattern

WIRING STEPS:

  1. Replace placeholder in validate_order():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SafeExecutionEngineImpl.validate_order()

     # Copy this into safe_execution_engine.py:
     from core_engine.implementations import SafeExecutionEngineImpl

     async def validate_order(self, symbol, action, quantity, price=None):
         return await SafeExecutionEngineImpl.validate_order(
             self.app_ctx, symbol, action, quantity, price
         )

  2. Replace placeholder in place_buy_order():
     FROM: await asyncio.sleep(0.1)
     TO:   Use SafeExecutionEngineImpl.place_buy_order()

     async def place_buy_order(self, symbol, quantity, price=None, order_type="LIMIT"):
         return await SafeExecutionEngineImpl.place_buy_order(
             self.app_ctx, symbol, quantity, price, order_type
         )

  3. Replace placeholder in place_sell_order() [CRITICAL - FIX #2]:
     FROM: await asyncio.sleep(0.1)
     TO:   Use SafeExecutionEngineImpl.place_sell_order()

     CRITICAL: This includes FIX #2 idempotent guard!

     async def place_sell_order(self, symbol, quantity, price=None, order_type="LIMIT"):
         return await SafeExecutionEngineImpl.place_sell_order(
             self.app_ctx, symbol, quantity, price, order_type
         )

     FIX #2 GUARD VERIFICATION:
     ✅ Checks bounded_cache for existing sell finalization
     ✅ Uses cache_key: f"sell_finalize_{symbol}_{order_id}"
     ✅ Sets TTL=300s (5 minutes)
     ✅ Returns ALREADY_FINALIZED if duplicate detected
     ✅ Prevents double-sells in event of system crash recovery

COMPONENT INTERFACES REQUIRED:

  execution_manager (L4):
    - async place_order(symbol, quantity, price, action, order_type) → Dict with orderId
    - async get_order_status(order_id) → str ("FILLED", "PARTIAL", "PENDING")
    - async cancel_order(symbol, order_id) → bool

  bounded_cache (L0):
    - async get(key) → value or None
    - async set(key, value, ttl=None) → bool
    - async contains(key) → bool

TESTING:
  pytest core_engine/tests/test_safe_execution_integration.py -v
  pytest core_engine/tests/test_fix2_idempotent_guard.py -v
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Wire OperationsEngine to L7/L8
# ═════════════════════════════════════════════════════════════════════════════

"""
TARGET: Connect OperationsEngine to health monitoring and startup orchestration

CURRENT STATE:
  File: core_engine/operations_engine.py
  Methods: startup_system, get_health_report, recover_state
  Implementation: Uses placeholder asyncio.sleep()

WIRING STEPS:

  1. Replace placeholder in startup_system():
     FROM: await asyncio.sleep(0.1)
     TO:   Use OperationsEngineImpl.startup_system()

     # Copy this into operations_engine.py:
     from core_engine.implementations import OperationsEngineImpl

     async def startup_system(self):
         return await OperationsEngineImpl.startup_system(self.app_ctx)

  2. Replace placeholder in get_health_report():
     FROM: await asyncio.sleep(0.1)
     TO:   Use OperationsEngineImpl.get_health_report()

     async def get_health_report(self):
         return await OperationsEngineImpl.get_health_report(self.app_ctx)

  3. Similar replacements for shutdown_system, check_liveness, etc.

COMPONENT INTERFACES REQUIRED:

  startup_orchestrator (L8):
    - async startup() → bool
    - async shutdown() → bool

  health_monitor (L7):
    - async get_report() → Dict with full health report
    - async get_component_status(component) → Dict
    - async get_overall_health() → str ("OK", "WARN", "ERROR", "CRITICAL")

TESTING:
  pytest core_engine/tests/test_operations_integration.py -v
"""


# ═════════════════════════════════════════════════════════════════════════════
# WIRING SEQUENCE CHECKLIST
# ═════════════════════════════════════════════════════════════════════════════

WIRING_CHECKLIST = """
PHASE 2 INTEGRATION CHECKLIST
────────────────────────────────

[ ] Step 1: MarketAccountEngine (L1/L2)
    [ ] Import MarketAccountEngineImpl
    [ ] Replace get_account_state()
    [ ] Replace get_market_prices()
    [ ] Replace get_wallet_balance()
    [ ] Replace get_ohlcv_data()
    [ ] Test with real exchange_client
    [ ] Validate error handling

[ ] Step 2: SituationEngine (L3/L5)
    [ ] Import SituationEngineImpl
    [ ] Replace get_portfolio_snapshot()
    [ ] Replace get_all_signals()
    [ ] Replace get_fused_signal()
    [ ] Replace get_market_regime()
    [ ] Test with real portfolio_manager
    [ ] Validate signal fusion

[ ] Step 3: DecisionEngine (L5/L6)
    [ ] Import DecisionEngineImpl
    [ ] Replace get_current_mode()
    [ ] Replace evaluate_signal()
    [ ] Replace make_buy_decision()
    [ ] Replace make_sell_decision()
    [ ] Test 6-layer arbitration gates
    [ ] Validate capital allocation

[ ] Step 4: SafeExecutionEngine (L4 + L0 FIX #2)
    [ ] Import SafeExecutionEngineImpl
    [ ] Replace validate_order()
    [ ] Replace place_buy_order()
    [ ] Replace place_sell_order() [WITH FIX #2]
    [ ] Test FIX #2 idempotent guard
    [ ] Verify bounded_cache integration
    [ ] Test duplicate sell prevention
    [ ] Validate error handling

[ ] Step 5: OperationsEngine (L7/L8)
    [ ] Import OperationsEngineImpl
    [ ] Replace startup_system()
    [ ] Replace get_health_report()
    [ ] Replace recover_state()
    [ ] Test full startup sequence L0→L8
    [ ] Validate health monitoring

[ ] Integration Testing
    [ ] Test full READ→UNDERSTAND→DECIDE→EXECUTE→RECOVER cycle
    [ ] Test with real market data
    [ ] Test with real portfolio state
    [ ] Test FIX #2 guard (10x redundancy)
    [ ] Load testing (2-second loop)
    [ ] Error recovery testing

[ ] Documentation
    [ ] Update CORE_ENGINE_SUMMARY.md with real component names
    [ ] Add component initialization guide
    [ ] Add troubleshooting guide
    [ ] Create integration examples

[ ] Deployment
    [ ] Stage 1: Paper trading (no real orders)
    [ ] Stage 2: Live trading with tiny position sizes
    [ ] Stage 3: Scale to production volume
    [ ] Monitor FIX #2 guard (99% confidence requirement)
"""

print(WIRING_CHECKLIST)
