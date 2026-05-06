# 🔀 LEGACY vs CORE ENGINE — INTEGRATION ANALYSIS

**Date**: May 6, 2026  
**Status**: TWO SEPARATE SYSTEMS in gradual migration  
**Integration Level**: Minimal (shims only, not merged)

---

## ⚙️ ACTUAL ARCHITECTURE (What's Really Running)

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (NEW ENTRY)                     │
│                                                             │
│  setup_core_engines(production=True)                        │
│  └─→ production_bridge.build_production_app_ctx()           │
│      └─→ Initializes LEGACY orchestrator                    │
│          └─→ 9 concurrent tasks (including MetaController)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
   LEGACY SYSTEM                  CORE ENGINE
   (Active/Production)            (Shim Layer)
   
   🎯_MASTER_SYSTEM_              core_engine/
   ORCHESTRATOR.py                ├─ market_account_engine.py
   ├─ L0-L8 (all)                 ├─ situation_engine.py
   ├─ MetaController              ├─ decision_engine.py
   ├─ AgentManager                ├─ safe_execution_engine.py
   ├─ PollingCoordinator          ├─ operations_engine.py
   ├─ MarketDataFeed              │
   ├─ BalanceSync                 └─ implementations.py
   ├─ TPSLEngine                      └─ Wraps legacy via app_ctx
   ├─ SafetyOrderManager
   ├─ ExecutionManager
   ├─ Watchdog
   └─ HealthMonitor

        │                         │
        └────────────┬────────────┘
                     │
         production_bridge (BRIDGE)
         ├─ Maps legacy to app_ctx
         ├─ Provides app_ctx dict
         └─ Enables façade delegation
```

---

## 📊 CONNECTION MATRIX

| Component | Legacy | Core Engine | MetaController | Type |
|-----------|--------|-------------|----------------|------|
| **MetaController** | ✅ Direct import | ❌ None | - | Direct L8 |
| **ExchangeClient** | ✅ Direct | ✅ Via app_ctx | ✅ Via execution_manager | Bridge |
| **SharedState** | ✅ Direct | ✅ Via app_ctx | ✅ Direct | Bridge |
| **SignalFusion** | ✅ Direct | ✅ Via app_ctx | ✅ Direct | Bridge |
| **PortfolioManager** | ✅ Direct | ✅ Via app_ctx | ✅ Via position_manager | Bridge |

---

## 🔍 DETAILED INTEGRATION ANALYSIS

### **LEGACY SYSTEM** (Primary, Active)

**Entry Point**: `🎯_MASTER_SYSTEM_ORCHESTRATOR.run_system()`

**What it does**:
1. Initializes ALL L0-L8 components sequentially
2. Wires them together directly (no indirection)
3. Launches 9+ concurrent async tasks:
   - `PollingCoordinator` (API requests)
   - `MarketDataFeed` (WebSocket streaming)
   - `TPSLEngine` (TP/SL monitoring)
   - `SafetyOrderManager` (OCO handling)
   - **`MetaController.run()`** ← MASTER DECISION LOOP
   - `AgentManager` (signal generation)
   - `Watchdog` (crash detection)
   - `Heartbeat` (liveness)
   - `ThreeBucketManagement` (portfolio rebalancing)

**MetaController's Role**:
```python
# In MetaController.run() main loop (every 2 seconds):
1. Ingest signals → signal_manager (L5)
2. Get governance decision → mode_manager (L5)
3. Build decisions → multi-source (L3-L5)
4. Arbitrate → arbitration_engine (L5) ← **6-layer gates**
5. Execute → execution_manager (L4) ← **FIX #2 guard called 10x**
6. Update state → portfolio_manager (L3)
7. Emit loop summary → health_monitor (L7)
```

**Status**: **FROZEN** per header comment but ACTIVELY RUNNING as production system

---

### **CORE ENGINE SYSTEM** (New, Wrapping)

**Entry Point**: `main.py` → `setup_core_engines(production=True)`

**5 Façade Engines** (what user code sees):
```python
market_account_engine.get_account_state()
  └─→ implementations.py wrapper
      └─→ app_ctx["exchange_client"].get_account()
          └─→ Legacy ExchangeClient (L1)

situation_engine.analyze()
  └─→ implementations.py wrapper
      └─→ app_ctx["portfolio_manager"].get_nav()
          └─→ Legacy PortfolioManager (L3)

decision_engine.decide_trades()
  └─→ implementations.py wrapper
      └─→ app_ctx["arbitration_engine"].evaluate_gates()
          └─→ Legacy ArbitrationEngine (L5)

safe_execution_engine.execute_order()
  └─→ implementations.py wrapper
      └─→ app_ctx["execution_manager"].place_order()
          └─→ Legacy ExecutionManager (L4) ← **FIX #2 guard**

operations_engine.monitor_health()
  └─→ implementations.py wrapper
      └─→ app_ctx["health_monitor"].check_all()
          └─→ Legacy HealthMonitor (L7)
```

**Status**: **Shim layer** — not integrated into main loop, just wraps legacy components

---

## 🔌 INTEGRATION POINTS (Actual Connections)

### **1. Production Bridge** (`production_bridge.py`)

**File**: `core_engine/production_bridge.py` lines 141-150

```python
# Load legacy orchestrator class
legacy_mod = importlib.import_module("src.l8_lifecycle.master_orchestrator")
Orchestrator = getattr(legacy_mod, "MasterSystemOrchestrator", None)

# Instantiate and initialize
orch = Orchestrator()
orch.check_prerequisites()
await orch.initialize_components()  # ← Constructs ALL L0-L8
```

**Result**: Returns `app_ctx` dict mapping:
- `"exchange_client"` → Legacy ExchangeClient instance
- `"meta_controller"` → Legacy MetaController instance
- `"portfolio_manager"` → Legacy PortfolioManager instance
- etc. (50+ mappings)

### **2. App Context** (Central Dictionary)

**File**: `core_engine/integration.py`

```python
app_ctx = {
    "config": Config,
    "shared_state": SharedState,
    "exchange_client": ExchangeClient,  # L1
    "market_data_feed": MarketDataFeed,  # L2
    "portfolio_manager": PortfolioManager,  # L3
    "execution_manager": ExecutionManager,  # L4
    "signal_fusion": SignalFusion,  # L5
    "arbitration_engine": ArbitrationEngine,  # L5
    "meta_controller": MetaController,  # L8
    "health_monitor": HealthMonitor,  # L7
    # ... 50+ more
}
```

The **only connection** between systems is this dictionary.

### **3. Façade Implementations** (`implementations.py`)

```python
class MarketAccountEngine:
    async def get_account_state(self):
        if not self._app_ctx:
            return None  # Mock mode
        client = self._app_ctx.get("exchange_client")
        return await client.get_account()  # ← Delegates to LEGACY
```

**Key pattern**: `implementations.py` is just an **async wrapper** + async helper (`_maybe_await()`)

---

## 🚫 WHAT'S NOT CONNECTED

### **MetaController ↔ Core Engine**

| Direction | Status | Details |
|-----------|--------|---------|
| MetaController → Core Engine | ❌ NO | MetaController never imports or calls core_engine |
| Core Engine → MetaController | ✅ YES | Via app_ctx["meta_controller"] |
| **Direct dependency?** | ❌ NO | Only through app_ctx bridge |

**Proof**: MetaController doesn't know it's being wrapped:
```python
# In meta_controller.py
# No imports from core_engine/
# No references to market_account_engine, situation_engine, etc.
# Runs normally as if it's the only system
```

### **Core Engine ↔ Main Loop**

| Integration | Status | Details |
|-------------|--------|---------|
| **Is Core Engine in run_system()?** | ❌ NO | Legacy orchestrator runs; façades are separate |
| **Does MetaController call façades?** | ❌ NO | MetaController calls L0-L8 directly |
| **Are façades called anywhere?** | ❌ NOT YET | They exist but aren't wired into production loop |

---

## 📈 MIGRATION STRATEGY (Phase 8.2)

### **Current Phase (8.1 Transitional)**

```
Step 1: Build Core Engine shims (DONE)
  └─→ market_account_engine.py, etc.
  └─→ implementations.py (async wrappers)
  └─→ production_bridge.py (legacy reuse)

Step 2: Test Core Engine in isolation (IN PROGRESS?)
  └─→ Write tests for façades
  └─→ Verify delegations work
  └─→ Benchmark vs legacy

Step 3: Integrate native subsystem (PLANNED for Phase 8.2.3+)
  └─→ Replace production_bridge (legacy construction)
  └─→ Wire native/ modules instead of legacy L0-L8
  └─→ Gradually migrate MetaController
```

### **Phase 8.2 Goals** (Per comments)

1. **Build native L0-L4** ✅ DONE (13 files, 1,800 LOC)
2. **Create façade API** ✅ DONE (5 engines)
3. **Wire via production_bridge** ✅ DONE (legacy reuse)
4. **Plan native integration** 🔄 IN PROGRESS
5. **Full replacement** 📅 FUTURE (Phase 8.2.3+)

---

## 🎯 CURRENT OPERATIONAL MODEL

### **What Runs When You Start**

```
$ python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py --mode=live

1. main.py entry
2. setup_core_engines(production=True)
3. production_bridge builds app_ctx from LEGACY orchestrator
4. 5 façades created (unused in production)
5. Legacy orchestrator.run_system() executes
6. MetaController.run() starts main loop (2s cycles)
7. 9+ tasks run (all legacy)
8. Façades available but NOT called
```

### **Who's Actually Trading?**

- **LEGACY MetaController** (L8) ← Decision maker
- **LEGACY execution_manager** (L4) ← Order placer (FIX #2 guard)
- **LEGACY signal_fusion** (L5) ← Signal aggregator
- **LEGACY arbitration_engine** (L5) ← Gate evaluator

**Core Engine is just sitting there, ready but unused.**

---

## ✅ KEY FINDINGS

### **1. Two Separate Systems**
- ✅ Legacy L0-L8 is fully independent
- ✅ Core Engine façades are independent wrappers
- ✅ MetaController is oblivious to Core Engine
- ✅ Only connection is app_ctx dictionary

### **2. Production is Legacy**
- ✅ Live trading uses legacy orchestrator
- ✅ MetaController runs actual decision loop
- ✅ FIX #2 guard in legacy execution_manager (10 calls)
- ✅ Core Engine is not in production path

### **3. Migration in Progress**
- ✅ Phase 8.1: Built shims (Core Engine façades)
- 🔄 Phase 8.2: Building native replacement (L0-L4)
- 📅 Phase 8.3: Plan full integration
- 📅 Phase 9.0: Full replacement of legacy

### **4. Current Risk Profile**
- ✅ Core Engine changes don't affect production (isolated)
- ✅ Legacy system fully functional and deployed
- ⚠️ Native subsystem not yet integrated (Phase 8.2.3+)
- ✅ FIX #2 guard in active legacy execution_manager

---

## 🚀 NEXT STEPS FOR INTEGRATION

### **To Connect Core Engine to Production Loop**

1. **Modify main.py** to inject façades into MetaController
2. **Route MetaController → façade engines** for each decision step
3. **Gradually replace** legacy L0-L8 calls with native subsystem calls
4. **Test equivalence** (legacy behavior vs new behavior)
5. **Switch production_bridge** to use native instead of legacy

### **To Verify Integration**

- [ ] Core Engine façades tested in isolation
- [ ] Production_bridge can load native modules
- [ ] MetaController can call façades
- [ ] FIX #2 guard still active in native executor
- [ ] 22-min test passes with hybrid system

---

## 📋 SUMMARY

| Aspect | Legacy | Core Engine | Status |
|--------|--------|-------------|--------|
| **Production ready** | ✅ YES | ❌ SHIM ONLY | Legacy running |
| **In main loop** | ✅ YES | ❌ NO | Not integrated |
| **Independent** | ✅ YES | ✅ YES | Both work alone |
| **Connected** | - | ⚠️ VIA APP_CTX | Minimal bridge |
| **MetaController integrated** | ✅ YES | ❌ NO | Legacy only |
| **FIX #2 guard active** | ✅ YES (legacy exec_mgr) | ❌ NOT YET (native executor not wired) | Production safe |

---

**VERDICT: Two separate systems. Legacy is production. Core Engine is prepared but awaiting Phase 8.2.3 integration.** 🚀
