# 🚀 MIGRATION SEQUENCE PLAN
## Complete Native Stack L0-L8 (In Order)

**Status**: READY TO START
**Target**: Full production-ready native stack
**Timeline**: 6-8 weeks (working sequentially)

---

## 📋 EXECUTION SEQUENCE

### **PHASE 1: L1 Complete (1 week)**

**Currently Done**:
- ✅ NativeSharedState (L0)
- ✅ NativeBalanceSync (L1)
- ✅ NativeMarketData (L2)

**What's Missing in L1**:
```
core_engine/native/exchange_client.py
├─ NativeExchangeClient (raw Binance REST wrapper)
│  ├─ Authentication (API key/secret, Ed25519)
│  ├─ Public methods: get_exchange_info(), get_prices(), get_klines()
│  ├─ Signed methods: get_account(), get_balance(), place_order(), cancel_order()
│  └─ Order tracking: get_order(), get_open_orders()
└─ Dependencies: aiohttp, binance SDK (same as legacy)
```

**Deliverables**:
- [ ] Create `core_engine/native/exchange_client.py` (200-300 lines)
- [ ] Write unit tests `tests/test_native_exchange_client.py`
- [ ] Add to `core_engine/native/__init__.py` exports
- [ ] Validate: Can place order via native client

**Success Criteria**:
```python
# Should work:
client = NativeExchangeClient(api_key, api_secret)
await client.ping()
balance = await client.get_balance("USDT")
order = await client.place_order("BTC", "BUY", 0.001, 45000)
```

---

### **PHASE 2: L3 Portfolio (2 weeks)**

**Dependencies**: L0, L1, L2 ✅

**What's Needed in L3** (7 components):
```
core_engine/native/portfolio.py
├─ NativePortfolioManager (core portfolio state)
│  ├─ positions dict[symbol] → Position
│  ├─ NAV calculation (cash + positions)
│  ├─ P&L tracking (realized, unrealized)
│  ├─ Dust handling
│  └─ Event emission
│
├─ NativePositionManager (position lifecycle)
│  ├─ Position state (open, exiting, closed)
│  ├─ Entry/exit tracking
│  ├─ TP/SL level management
│  └─ Forced exit handling
│
├─ NativeTPSLEngine (take-profit/stop-loss)
│  ├─ Dynamic TP/SL levels (ATR-based)
│  ├─ Exit trigger detection
│  └─ Exit classification
│
├─ NativeSymbolManager (symbol universe)
│  ├─ Symbol registry (known symbols)
│  ├─ Symbol metadata (notional mins, precision)
│  └─ accepted_symbols tracking
│
├─ NativeSymbolRotation (symbol in/out rotation)
│  ├─ Replacement multiplier logic
│  ├─ Cooldown enforcement
│  └─ Soft bootstrap lock
│
├─ NativeBucketManager (3-tier allocation)
│  ├─ Active bucket (trading positions)
│  ├─ Reserve bucket (standby capital)
│  └─ Idle bucket (dust, closed)
│
└─ NativeRecoveryEngine (state reconstruction)
   ├─ Rebuild from exchange
   ├─ Fallback to database
   └─ Integrity verification
```

**Files to Create**:
- `core_engine/native/portfolio.py` (500+ lines)
- `core_engine/native/tp_sl.py` (200+ lines)
- `core_engine/native/symbol_management.py` (200+ lines)
- `tests/test_native_portfolio.py` (10+ tests)

**Success Criteria**:
```python
# Should work:
mgr = NativePortfolioManager(shared_state, exchange_client)
await mgr.open_position("BTC", 0.001, 45000, "signal_1")
nav = mgr.get_nav()  # cash + positions value
await mgr.close_position("BTC", 46000)
realized_pnl = mgr.get_realized_pnl()
```

**Dependencies Between L3 Components**:
```
PortfolioManager
  ├─ PositionManager (nested)
  ├─ TPSLEngine (separate)
  ├─ SymbolManager (nested)
  ├─ SymbolRotation (separate)
  ├─ BucketManager (separate)
  └─ RecoveryEngine (separate)
```

---

### **PHASE 3: L5 Decision & Arbitration (2 weeks)**

**Dependencies**: L0, L1, L2, L3 ✅

**What's Needed in L5** (3 critical components):
```
core_engine/native/decisions.py (ALREADY STARTED)
├─ Extend NativeDecisionEngine with gates
├─ Add Gate 1: Symbol validation
├─ Add Gate 2: Confidence floor (mode-based)
├─ Add Gate 3: Market regime check
├─ Add Gate 4: Position limit (current < max)
├─ Add Gate 5: Capital available (> 0)
├─ Add Gate 6: Risk manager approval (stub for now)
└─ Return: (decision_approved, gate_results)

core_engine/native/arbitration.py
├─ NativeArbitrationEngine (6-layer gate evaluation)
├─ evaluate_gates_sync() (no async context needed)
├─ Returns: GateResult[] (pass/fail + reason per gate)
└─ Fallback symbol logic (retry if rejected)

core_engine/native/signal_fusion.py (ALREADY EXISTS)
├─ Extend NativeSignalEngine with agent weighting
├─ MLForecaster: 1.5x
├─ LiquidationAgent: 1.3x
├─ DipSniper: 1.2x
├─ TrendHunter: 1.0x
├─ Others: 0.7x-0.9x
├─ Thresholds: BUY +0.35, SELL -0.35
└─ Return: Fused signal with composite edge

core_engine/native/mode_manager.py
├─ NativeModeManager (governance mode selection)
├─ Detect: PAUSED, PROTECTIVE, BOOTSTRAP, NORMAL, AGGRESSIVE
├─ Apply phase gates (BOOTSTRAP, INITIALIZATION, STEADY_STATE)
└─ Block actions per mode (BUY, SELL, LIQUIDATE)
```

**Files to Create/Extend**:
- Extend `core_engine/native/decisions.py` (add gates, 200+ lines)
- Create `core_engine/native/arbitration.py` (200+ lines)
- Create `core_engine/native/mode_manager.py` (150+ lines)
- Extend `core_engine/native/signals.py` (agent weighting, 100+ lines)
- `tests/test_native_arbitration.py` (15+ tests)

**Success Criteria**:
```python
# Should work:
decider = NativeDecisionEngine(portfolio, shared_state, risk_mgr)
signal = Signal(symbol="BTC", action="BUY", confidence=0.7)
decision, gates = await decider.decide(signal)
# decision: (symbol, action, details)
# gates: [GateResult(name, passed, reason)]

arb = NativeArbitrationEngine(portfolio, regime, mode)
gates = arb.evaluate_gates_sync(signal)
# gates: list of GateResult (all 6 evaluated)
```

---

### **PHASE 4: L6 Risk Management (1 week)**

**Dependencies**: L0, L1, L3, L5 ✅

**What's Needed in L6** (1 critical component):
```
core_engine/native/risk.py
├─ NativeRiskManager (overall risk policy)
├─ Position limit check (max open < config)
├─ Capital limit check (available > 0)
├─ Daily loss limit (current loss < threshold)
├─ Drawdown limit (NAV decline < threshold)
├─ Per-symbol capital cap (allocation < limit)
└─ Return: (approved: bool, reason: str)

core_engine/native/capital_allocator.py
├─ NativeCapitalAllocator (position sizing)
├─ Kelly criterion (optional)
├─ Fixed % allocation
├─ Dynamic scaling (based on performance)
└─ Return: allocated_capital (amount to deploy)
```

**Files to Create**:
- `core_engine/native/risk.py` (150+ lines)
- `core_engine/native/capital.py` (150+ lines)
- `tests/test_native_risk.py` (10+ tests)

**Success Criteria**:
```python
# Should work:
risk = NativeRiskManager(portfolio, config)
approved, reason = risk.check_trade(symbol="BTC", side="BUY")
# approved: True/False
# reason: "OK" or "Position limit exceeded" etc.

allocator = NativeCapitalAllocator(portfolio, signal)
amount = allocator.allocate_capital(signal)  # $amount to use
```

---

### **PHASE 5: L7 Health & Monitoring (1 week)**

**Dependencies**: L0, L2, L3 ✅

**What's Needed in L7** (2 critical components):
```
core_engine/native/health.py
├─ NativeHealthMonitor (real-time component checks)
├─ Check: exchange_client connectivity
├─ Check: shared_state consistency
├─ Check: portfolio NAV integrity
├─ Check: balance sync freshness
├─ Emit: HealthStatus (all components)
└─ Return: overall_health (HEALTHY/DEGRADED/CRITICAL)

core_engine/native/watchdog.py
├─ NativeWatchdog (crash/hang detection)
├─ Monitor: main loop liveness
├─ Monitor: component response time (max threshold)
├─ Detect: hangs (last update > 5s ago?)
├─ Detect: crashes (process exit)
├─ Action: trigger alert, auto-restart (optional)
└─ Emit: WatchdogAlert events
```

**Files to Create**:
- `core_engine/native/health.py` (150+ lines)
- `core_engine/native/watchdog.py` (150+ lines)
- `tests/test_native_health.py` (10+ tests)

**Success Criteria**:
```python
# Should work:
health = NativeHealthMonitor(components)
status = health.check_all()
# status: HealthStatus(overall=HEALTHY, components={...})

watchdog = NativeWatchdog()
watchdog.update_heartbeat()  # Call each cycle
await watchdog.monitor()  # Runs in background
# Alerts if heartbeat missing > 10 seconds
```

---

### **PHASE 6: L8 Orchestrator (2 weeks)**

**Dependencies**: L0-L7 ✅

**What's Needed in L8** (1 critical component):
```
core_engine/native/orchestrator.py
├─ NativeOrchestrator (replaces MetaController)
├─ Initialization: setup L0-L7 components
├─ Main loop: evaluate_and_act() every 2 seconds
├─ Cycle steps:
│  ├─ 1. Sync balance (real-time)
│  ├─ 2. Ingest signals (signal_manager cache)
│  ├─ 3. Fuse signals (signal_fusion)
│  ├─ 4. Get governance mode (mode_manager)
│  ├─ 5. Build decisions (multi-source)
│  ├─ 6. Arbitrate (arbitration_engine gates)
│  ├─ 7. Check risk (risk_manager)
│  ├─ 8. Allocate capital (capital_allocator)
│  ├─ 9. Execute (executor)
│  ├─ 10. Update state (portfolio_manager)
│  ├─ 11. Monitor health (health_monitor)
│  └─ 12. Emit loop summary
├─ Startup: recovery_engine (rebuild state)
├─ Shutdown: cancel orders, save state
└─ Async lifecycle: run(), stop()

core_engine/native/executor.py (ALREADY EXISTS)
├─ Extend NativeExecutor
├─ Add: order_manager (track open orders)
├─ Add: fill reconciliation (partial fills)
├─ Add: **FIX #2: Idempotent SELL guard** ← CRITICAL
│  ├─ Cache key: "sell_finalize_{symbol}_{order_id}"
│  ├─ Reset: each cycle start
│  └─ Prevent: duplicate sell finalization
└─ Return: ExecutionResult (success, order_id, reason)
```

**Files to Create/Extend**:
- Create `core_engine/native/orchestrator.py` (400-500 lines)
- Extend `core_engine/native/executor.py` (add FIX #2 guard, 100+ lines)
- Create `core_engine/native/startup.py` (200+ lines)
- `tests/test_native_orchestrator.py` (20+ tests)

**Success Criteria**:
```python
# Should work:
orch = NativeOrchestrator(config, exchange_client, ...)
await orch.initialize()  # Startup sequence
await orch.run()  # Main loop (runs until stop called)

# In tests:
# ✅ 22-minute continuous operation
# ✅ 1,084 cycles without crashes
# ✅ FIX #2 guard: 0 duplicate sells
# ✅ Capital accounting: 0.00% error
```

---

## 📅 WEEK-BY-WEEK PLAN

```
WEEK 1: L1 Exchange Client
├─ Create NativeExchangeClient
├─ Write unit tests
├─ Integrate with existing NativeBalanceSync
└─ Validate: Can place orders

WEEK 2: L3 Portfolio (Part 1)
├─ Create PortfolioManager + PositionManager
├─ Create TPSLEngine
├─ Write integration tests
└─ Validate: Can track positions

WEEK 3: L3 Portfolio (Part 2)
├─ Create SymbolManager + SymbolRotation
├─ Create BucketManager
├─ Create RecoveryEngine
└─ Validate: Can recover state

WEEK 4: L5 Decision/Arbitration
├─ Extend DecisionEngine (add gates)
├─ Create ArbitrationEngine
├─ Extend SignalFusion (agent weights)
├─ Create ModeManager
└─ Validate: 6 gates block bad trades

WEEK 5: L6 Risk + L7 Health
├─ Create RiskManager
├─ Create CapitalAllocator
├─ Create HealthMonitor
├─ Create Watchdog
└─ Validate: Limits enforced, monitoring works

WEEK 6: L8 Orchestrator (Part 1)
├─ Create StartupOrchestrator
├─ Create core orchestrator loop
├─ Wire all L0-L7 components
└─ Validate: Single cycle completes

WEEK 7: L8 Orchestrator (Part 2)
├─ **Add FIX #2 guard to executor** ← CRITICAL
├─ Full lifecycle (startup, run, shutdown)
├─ Integration tests (full cycle)
├─ Paper trading validation
└─ Validate: 22-min test passes

WEEK 8: Production Switch
├─ Integration test suite (177 tests)
├─ Compare: legacy vs native trades
├─ Performance benchmarking
├─ Production switch (if all pass)
└─ Delete production_bridge.py
```

---

## 🎯 START: PHASE 1 (L1 Exchange Client)

**Let's begin with the first component: NativeExchangeClient**

### **What We're Building**

File: `core_engine/native/exchange_client.py`

Responsibilities:
- ✅ Raw Binance REST API wrapper
- ✅ Public: exchange_info, prices, klines
- ✅ Signed: account, balance, orders
- ✅ Auth: HMAC or Ed25519

### **Files to Examine** (as reference)

1. `src/l1_exchange/exchange_client.py` (legacy — reference only)
2. `core_engine/native/__init__.py` (where to export)
3. `core_engine/native/config_loader.py` (how config works)
4. `core_engine/native/shared_state.py` (data structures)

### **Ready to Start?**

**Next step**: Shall I create the `NativeExchangeClient` skeleton?

You can provide guidance on:
1. Should it use `aiohttp` directly (like legacy) or async Binance SDK?
2. Should it support both HMAC and Ed25519, or just HMAC for now?
3. Any specific methods beyond buy/sell/cancel/get_balance/get_account?

---

## ✅ PROGRESS TRACKING

```
[ ] Phase 1: L1 Exchange Client (1 week)
[ ] Phase 2: L3 Portfolio (2 weeks)
[ ] Phase 3: L5 Decision/Arbitration (2 weeks)
[ ] Phase 4: L6 Risk Management (1 week)
[ ] Phase 5: L7 Health/Watchdog (1 week)
[ ] Phase 6: L8 Orchestrator (2 weeks)
[ ] Integration Testing (1 week)
[ ] Paper Trading Validation (1 week)
[ ] Production Switch (1 day)
```

---

**Ready to start Phase 1?** 🚀
