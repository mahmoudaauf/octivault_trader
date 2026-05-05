"""
Complete System Architecture Flow
═════════════════════════════════════════════════════════════════════════════

This document shows how the 5 façade engines connect to all L0-L8 components.
"""

ARCHITECTURE_OVERVIEW = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FAÇADE LAYER (PHASE 2)                                 │
│                    5 Engines Abstract All Complexity                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   READ       │  │ UNDERSTAND   │  │   DECIDE     │  │  EXECUTE     │   │
│  │              │  │              │  │              │  │              │   │
│  │ Market       │  │ Situation    │  │ Decision     │  │ Safe         │   │
│  │ Account      │  │ Engine       │  │ Engine       │  │ Execution    │   │
│  │ Engine       │  │              │  │              │  │ Engine       │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                 │                 │                 │              │
│         │                 │                 │                 │              │
│         └─────────────────┴─────────────────┴─────────────────┘              │
│                           │                                                  │
│                    ┌──────▼──────┐                                          │
│                    │ Operations  │                                          │
│                    │ Engine      │                                          │
│                    │ (RECOVER)   │                                          │
│                    └─────────────┘                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │  L0: Core      │  │  L1: Exchange  │  │  L2: Market    │
        │  State         │  │  Client        │  │  Data          │
        │  Management    │  │                │  │                │
        └────────────────┘  └────────────────┘  └────────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
    │  L3: Portfolio │     │  L4: Execution │     │  L5: Strategy  │
    │  Management    │     │  Management    │     │  & Signals     │
    │                │     │                │     │                │
    └────────────────┘     └────────────────┘     └────────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
        ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
        │  L6: Governance│  │  L7: Health &  │  │  L8: Lifecycle │
        │  & Policy      │  │  Observability │  │  Management    │
        │                │  │                │  │                │
        └────────────────┘  └────────────────┘  └────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │  System Monitoring, Health Reports, Event Logging      │
        └────────────────────────────────────────────────────────┘
"""

DETAILED_FLOW = """
═════════════════════════════════════════════════════════════════════════════
FUNCTION 1: READ (MarketAccountEngine)
═════════════════════════════════════════════════════════════════════════════

Input:  (None - runs on interval)
Output: Account state, prices, balances

Flow:
  MarketAccountEngine
    ↓
  MarketAccountEngineImpl.get_account_state()
    ├→ exchange_client (L1).get_account()
    │   └→ Returns: {balances: [...], positions: [...]}
    │
    ├→ market_data_feed (L2).get_prices(symbols)
    │   └→ Returns: {BTCUSDT: 45000, ETHUSDT: 3000, ...}
    │
    ├→ balance_manager (L2).get_balance()
    │   └→ Returns: {total_usdt: 10000, available_usdt: 8000, ...}
    │
    └→ SharedState (L0) ← Cache prices/balances
        └→ Returns: Account snapshot

Component Dependencies:
  PRIMARY:
    - exchange_client (L1): Binance async client wrapper
    - market_data_feed (L2): Price cache with WebSocket updates
    - balance_manager (L2): Wallet state tracker

  FALLBACK:
    - If market_data_feed fails, use exchange_client directly
    - If balance_manager fails, derive from account balances


═════════════════════════════════════════════════════════════════════════════
FUNCTION 2: UNDERSTAND (SituationEngine)
═════════════════════════════════════════════════════════════════════════════

Input:  Account state (from READ)
Output: Portfolio snapshot, signals, regime state

Flow:
  SituationEngine
    ↓
  SituationEngineImpl.get_portfolio_snapshot()
    ├→ portfolio_manager (L3).get_nav()
    │   └→ Returns: 10000.0 (total portfolio value in USDT)
    │
    ├→ portfolio_manager (L3).get_positions()
    │   └→ Returns: [{symbol: "BTCUSDT", qty: 0.1, cost: 4500, ...}, ...]
    │
    ├→ portfolio_manager (L3).get_pnl()
    │   └→ Returns: 250.0 (total profit/loss in USDT)
    │
    └→ SharedState (L0) ← Cache portfolio metrics
        └→ Returns: PortfolioSnapshot

  SituationEngineImpl.get_all_signals()
    ├→ signal_manager (L5).get_signals(symbol)
    │   └→ Returns: [
    │       {agent: "MA_cross", symbol: "BTCUSDT", edge: 0.65, ...},
    │       {agent: "RSI", symbol: "BTCUSDT", edge: 0.55, ...},
    │       ...
    │     ]
    │
    └→ Returns: List of all active signals

  SituationEngineImpl.get_fused_signal()
    ├→ signal_fusion (L5).fuse_signal(symbol)
    │   ├→ Reads: All signals for symbol
    │   ├→ Applies: Weighted fusion (best agents)
    │   ├→ Calculates: Combined edge score
    │   └→ Returns: {fused_edge: 0.68, confidence: 0.92, ...}
    │
    └→ Returns: SignalScore dataclass

  SituationEngineImpl.get_market_regime()
    ├→ market_regime_detector (L2).get_regime()
    │   ├→ Analyzes: Volatility (NORMAL, HIGH, EXTREME)
    │   ├→ Analyzes: Trend (UPTREND, DOWNTREND, RANGING)
    │   └→ Returns: {volatility: "NORMAL", trend: "UPTREND", ...}
    │
    └→ Returns: RegimeState

Component Dependencies:
  PRIMARY:
    - portfolio_manager (L3): Position & capital tracking
    - signal_manager (L5): Individual agent signals
    - signal_fusion (L5): Weighted signal combination
    - market_regime_detector (L2): Market state analysis

  READS FROM:
    - agents (L5): Individual trading signals
    - SharedState (L0): Current positions & capital


═════════════════════════════════════════════════════════════════════════════
FUNCTION 3: DECIDE (DecisionEngine)
═════════════════════════════════════════════════════════════════════════════

Input:  Situation snapshot + fused signal
Output: Trade decision (BUY/SELL/HOLD)

Flow:
  DecisionEngine
    ↓
  DecisionEngineImpl.make_buy_decision(symbol, edge_score)
    │
    ├─→ Step 1: Get current mode
    │   ├→ mode_manager (L5).get_current_mode()
    │   │   └→ Returns: "PROTECTIVE", "GROWTH", "ACCUMULATION", ...
    │   │
    │   └→ Returns: Trading mode constraints
    │
    ├─→ Step 2: Evaluate through arbitration gates (6-layer)
    │   ├→ arbitration_engine (L5).evaluate(symbol, "BUY", edge_score)
    │   │   ├→ Gate 1: Symbol format validation (must end with USDT)
    │   │   ├→ Gate 2: Confidence floor (edge_score ≥ threshold)
    │   │   ├→ Gate 3: Market regime check (no trades in EXTREME vol)
    │   │   ├→ Gate 4: Position limit (max 10 active positions)
    │   │   ├→ Gate 5: Capital available (at least $100 for BUY)
    │   │   ├→ Gate 6: Risk manager approval (volatility-adjusted position size)
    │   │   │
    │   │   └→ Returns: {passed: True/False, gates_status: {...}, blocking_gates: [...]}
    │   │
    │   └→ Returns: ArbitrationResult
    │
    ├─→ Step 3: If passed, allocate capital
    │   ├→ capital_allocator (L6).allocate_for_buy(symbol)
    │   │   ├→ Checks: Available capital ($8000)
    │   │   ├→ Checks: Position size limits
    │   │   ├→ Checks: Leverage constraints
    │   │   │
    │   │   └→ Returns: 0.1 BTC (quantity to buy)
    │   │
    │   └→ Returns: Quantity
    │
    └─→ Step 4: Build TradeDecision
        └→ Returns: {
            symbol: "BTCUSDT",
            action: "BUY",
            quantity: 0.1,
            confidence: 0.68,
            reason: "Signal edge: 0.68"
          }

Component Dependencies:
  PRIMARY:
    - arbitration_engine (L5): 6-layer gate validation
    - mode_manager (L5): Trading mode + constraints
    - capital_allocator (L6): Position size calculation

  CONSULTED:
    - policy_manager (L6): Trading policies
    - leverage_manager (L4): Leverage limits
    - tp_sl_engine (L4): Take profit/stop loss calculation


═════════════════════════════════════════════════════════════════════════════
FUNCTION 4: EXECUTE (SafeExecutionEngine)
═════════════════════════════════════════════════════════════════════════════

Input:  Trade decision (BUY/SELL/HOLD)
Output: ExecutionResult (success/failure)

Flow:
  SafeExecutionEngine
    ↓
  SafeExecutionEngineImpl.place_buy_order()
    │
    ├─→ Step 1: Validate order
    │   ├→ Check: Symbol format (must end with USDT)
    │   ├→ Check: Quantity > 0
    │   ├→ Check: Price > 0 (if limit order)
    │   ├→ Check: Notional ≥ 10 USDT (Binance minimum)
    │   ├→ Check: Step size alignment (BTC = 0.00001)
    │   │
    │   └→ Returns: OrderValidation {valid: True/False, errors: [...]}
    │
    ├─→ Step 2: Place order on exchange
    │   ├→ execution_manager (L4).place_order(
    │   │     symbol="BTCUSDT",
    │   │     quantity=0.1,
    │   │     price=45000,
    │   │     action="BUY",
    │   │     order_type="LIMIT"
    │   │   )
    │   │
    │   ├→ Returns: {orderId: 12345, executedQty: 0.1, avgPrice: 45000, ...}
    │   │
    │   └→ Store: In SharedState (L0)
    │
    └─→ Returns: ExecutionResult {
          success: True,
          order_id: 12345,
          symbol: "BTCUSDT",
          quantity: 0.1,
          average_price: 45000.0,
          status: "FILLED"
        }

  SafeExecutionEngineImpl.place_sell_order() ⭐ [WITH FIX #2]
    │
    ├─→ Step 1: Validate order (same as BUY)
    │   └→ Returns: OrderValidation
    │
    ├─→ Step 2: FIX #2 CHECK - Prevent duplicate SELL
    │   ├→ bounded_cache (L0).get(f"sell_finalize_BTCUSDT_{order_id}")
    │   │   └→ Returns: None (not cached) or True (already finalized)
    │   │
    │   ├→ If already cached:
    │   │   └→ REJECT: Return {status: "ALREADY_FINALIZED", ...}
    │   │       (Prevents duplicate sell on crash recovery)
    │   │
    │   └→ Continue if not cached
    │
    ├─→ Step 3: Place order on exchange
    │   ├→ execution_manager (L4).place_order(...)
    │   │   └→ Returns: {orderId: 12346, executedQty: 0.1, ...}
    │   │
    │   └→ Store: In SharedState (L0)
    │
    ├─→ Step 4: FIX #2 MARK - Mark as finalized
    │   ├→ bounded_cache (L0).set(
    │   │     key=f"sell_finalize_BTCUSDT_{order_id}",
    │   │     value=True,
    │   │     ttl=300  # 5-minute expiry
    │   │   )
    │   │
    │   └→ Cache set successfully
    │
    └─→ Returns: ExecutionResult {
          success: True,
          order_id: 12346,
          symbol: "BTCUSDT",
          quantity: 0.1,
          average_price: 45100.0,
          status: "FILLED"
        }

Component Dependencies:
  PRIMARY:
    - execution_manager (L4): Order placement on exchange
    - bounded_cache (L0): FIX #2 duplicate prevention cache

  CONSULTED:
    - exchange_client (L1): Direct order placement (fallback)
    - safety_order_manager (L4): Safety order placement
    - leverage_manager (L4): Leverage adjustment


═════════════════════════════════════════════════════════════════════════════
FUNCTION 5: RECOVER/MONITOR (OperationsEngine)
═════════════════════════════════════════════════════════════════════════════

Input:  System state
Output: Health status, recovery actions

Flow:
  OperationsEngine
    ↓
  OperationsEngineImpl.startup_system()
    ├→ startup_orchestrator (L8).startup()
    │   ├→ Initialize: L0_core (SharedState, BoundedCache)
    │   ├→ Initialize: L1_exchange (ExchangeClient)
    │   ├→ Initialize: L2_marketdata (MarketDataFeed, BalanceManager)
    │   ├→ Initialize: L3_portfolio (PortfolioManager)
    │   ├→ Initialize: L4_execution (ExecutionManager)
    │   ├→ Initialize: L5_strategy (SignalFusion, Arbitration)
    │   ├→ Initialize: L6_governance (PolicyManager, CapitalAllocator)
    │   ├→ Initialize: L7_observability (HealthMonitor, Watchdog)
    │   ├→ Initialize: L8_lifecycle (EventStore)
    │   │
    │   └→ Returns: True (success) or False (failure)
    │
    └→ Returns: bool

  OperationsEngineImpl.get_health_report()
    ├→ health_monitor (L7).get_report()
    │   ├→ Check: Each component status (OK, WARN, ERROR, CRITICAL)
    │   ├→ Check: Memory usage, CPU, network
    │   ├→ Check: Last heartbeat from each layer
    │   ├→ Check: Error counts and types
    │   │
    │   └→ Returns: {
    │       overall_status: "OK" | "WARN" | "ERROR" | "CRITICAL",
    │       components: {
    │         "L0_core": {status: "OK", uptime: 3600},
    │         "L1_exchange": {status: "OK", uptime: 3600},
    │         ...
    │       },
    │       critical_issues: [],
    │       warnings: ["Memory usage at 85%"],
    │       suggestions: ["Consider cleaning up old positions"]
    │     }
    │
    └→ Returns: HealthReport

Component Dependencies:
  PRIMARY:
    - startup_orchestrator (L8): Full L0→L8 initialization
    - health_monitor (L7): Component health tracking
    - watchdog (L7): Process monitoring

  CONSULTED:
    - state_manager (L3): Persistent state
    - recovery_engine (L3): State recovery
    - event_store (L3): Event history


═════════════════════════════════════════════════════════════════════════════
COMPONENT MATRIX: Which Engine Calls Which Components
═════════════════════════════════════════════════════════════════════════════

Component           | READ | UNDERSTAND | DECIDE | EXECUTE | RECOVER
──────────────────────────────────────────────────────────────────────────
L0: SharedState     | R/W  | R/W        | R/W    | R/W     | R/W
   BoundedCache     |      |            |        | R/W*    |
──────────────────────────────────────────────────────────────────────────
L1: ExchangeClient  | ✓    |            |        | ✓       |
   WebSocket        | ✓    |            |        |         |
──────────────────────────────────────────────────────────────────────────
L2: MarketDataFeed  | ✓    | ✓          |        |         |
   BalanceManager   | ✓    |            |        |         |
   RegimeDetector   |      | ✓          |        |         |
──────────────────────────────────────────────────────────────────────────
L3: PortfolioMgr    |      | ✓          |        |         |
   StateManager     |      |            |        |         | ✓
──────────────────────────────────────────────────────────────────────────
L4: ExecutionMgr    |      |            |        | ✓       |
   SafeOrderMgr     |      |            |        | ✓       |
   LeverageMgr      |      |            | ✓      | ✓       |
──────────────────────────────────────────────────────────────────────────
L5: SignalMgr       |      | ✓          |        |         |
   SignalFusion     |      | ✓          |        |         |
   Arbitration      |      |            | ✓      |         |
   ModeManager      |      |            | ✓      |         |
──────────────────────────────────────────────────────────────────────────
L6: PolicyMgr       |      |            | ✓      |         |
   CapitalAllocator |      |            | ✓      |         |
──────────────────────────────────────────────────────────────────────────
L7: HealthMonitor   |      |            |        |         | ✓
   Watchdog         |      |            |        |         | ✓
──────────────────────────────────────────────────────────────────────────
L8: StartupOrch     |      |            |        |         | ✓
   EventStore       |      |            |        |         | ✓

Legend:
  ✓ = Uses
  R/W = Read/Write
  * = FIX #2 specific (SELL guard)


═════════════════════════════════════════════════════════════════════════════
OPERATION LOOP SEQUENCE (2-second cycle)
═════════════════════════════════════════════════════════════════════════════

Second 0.0: READ
  ├─ MarketAccountEngine.get_account_state()
  ├─ Fetch: exchange_client.get_account()
  ├─ Cache: Prices, balances, positions
  └─ Time: ~200ms

Second 0.2: UNDERSTAND
  ├─ SituationEngine.get_portfolio_snapshot()
  ├─ Analyze: All signals, fusion
  ├─ Detect: Anomalies, regime
  └─ Time: ~300ms

Second 0.5: DECIDE
  ├─ DecisionEngine.evaluate_signal()
  ├─ Filter: 6-layer arbitration
  ├─ Allocate: Capital
  └─ Time: ~200ms

Second 0.7: EXECUTE
  ├─ SafeExecutionEngine.validate_order()
  ├─ SafeExecutionEngine.place_order()
  ├─ Record: FIX #2 guard (for SELL)
  └─ Time: ~400ms (exchange latency)

Second 1.1: RECOVER/MONITOR
  ├─ OperationsEngine.check_liveness()
  ├─ OperationsEngine.get_health_report()
  ├─ Log: Metrics, events
  └─ Time: ~100ms

Second 1.2: ← CYCLE COMPLETE, REPEAT

Total cycle time: ~2 seconds
Tolerance: ±500ms (allows for network latency)


═════════════════════════════════════════════════════════════════════════════
DATA FLOW DIAGRAM
═════════════════════════════════════════════════════════════════════════════

        ┌─────────────────────────────────────────────────┐
        │          MARKET DATA SOURCES                    │
        │  (Binance, Exchange APIs, WebSockets)          │
        └────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────────┐
        │   1. READ: MarketAccountEngine                  │
        │   Output: AccountState                          │
        │   {balances, prices, open_orders}             │
        └────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────────┐
        │   2. UNDERSTAND: SituationEngine               │
        │   Output: PortfolioSnapshot                    │
        │   {nav, positions, signals, regime}           │
        └────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────────┐
        │   3. DECIDE: DecisionEngine                    │
        │   Output: TradeDecision                        │
        │   {symbol, action, quantity, confidence}      │
        └────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────────┐
        │   4. EXECUTE: SafeExecutionEngine              │
        │   Output: ExecutionResult                      │
        │   {success, order_id, filled_qty}             │
        │   ⭐ FIX #2: Double-sell prevention           │
        └────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────────────────────┐
        │   5. RECOVER: OperationsEngine                 │
        │   Output: HealthReport                         │
        │   {status, components, issues}                │
        │   ← Loops back to READ                         │
        └─────────────────────────────────────────────────┘


═════════════════════════════════════════════════════════════════════════════
COMPONENT INITIALIZATION ORDER (L0 → L8)
═════════════════════════════════════════════════════════════════════════════

Layer | Component                    | Dependencies        | Order
──────────────────────────────────────────────────────────────────────
L0    | SharedState                  | None                | 1st
L0    | BoundedCache                 | SharedState         | 2nd
L1    | ExchangeClient               | (API key)           | 3rd
L1    | WebSocketClient              | ExchangeClient      | 4th
L2    | MarketDataFeed               | WebSocketClient     | 5th
L2    | BalanceManager               | ExchangeClient      | 6th
L2    | RegimeDetector               | MarketDataFeed      | 7th
L3    | PortfolioManager             | SharedState         | 8th
L3    | StateManager                 | SharedState         | 9th
L4    | ExecutionManager             | ExchangeClient      | 10th
L4    | SafeOrderManager             | ExecutionManager    | 11th
L4    | LeverageManager              | PortfolioManager    | 12th
L5    | SignalManager                | SharedState         | 13th
L5    | SignalFusion                 | SignalManager       | 14th
L5    | ArbitrationEngine            | PolicyManager       | 15th
L5    | ModeManager                  | SharedState         | 16th
L6    | PolicyManager                | (config)            | 17th
L6    | CapitalAllocator             | PortfolioManager    | 18th
L7    | HealthMonitor                | All layers          | 19th
L7    | Watchdog                     | HealthMonitor       | 20th
L8    | EventStore                   | SharedState         | 21st
L8    | StartupOrchestrator          | All layers          | 22nd

Startup sequence: 22 components in strict order L0→L8
"""

if __name__ == "__main__":
    print(ARCHITECTURE_OVERVIEW)
    print("\n\n")
    print(DETAILED_FLOW)
