# AppContext Architecture Diagram & Quick Reference

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPCONTEXT (Orchestrator)                  │
│  • Phased initialization (P3→P9)                               │
│  • Component lifecycle management                              │
│  • Readiness gating & health diagnostics                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
         ┌──────────────────────┬───────────────────────────┐
         ↓                      ↓                           ↓
    ┌─────────┐           ┌──────────┐            ┌─────────────┐
    │   P3    │           │   P4     │            │   P5→P9     │
    │Exchange │           │ Market   │            │  Decision   │
    │ & Univ. │           │ Data &   │            │  & Control  │
    │Bootstrap│           │ Filters  │            │   Engines   │
    └────┬────┘           └──────────┘            └─────┬───────┘
         │                     │                        │
         └─────────────────────┴────────────────────────┘
                               ↓
    ┌──────────────────────────────────────────────────────────────┐
    │              SHARED STATE (Central Hub)                      │
    │  • Balances, NAV, Positions, Intents                        │
    │  • Dynamic config, Events, Metrics                          │
    │  • All 23 components reference this                         │
    └──────────────────────────────────────────────────────────────┘
                               ↓
    ┌──────────────────────────────────────────────────────────────┐
    │                  30+ System Components                        │
    ├──────────────────────────────────────────────────────────────┤
    │  CORE INFRASTRUCTURE (P3-P5)                                │
    │  ├─ ExchangeClient (Binance API)                            │
    │  ├─ SymbolManager (Universe)                                │
    │  ├─ CapitalSymbolGovernor (Capital limits per symbol)       │
    │  ├─ MarketDataFeed (OHLCV, tick prices)                     │
    │  ├─ ExecutionManager (Order execution, filters, affordability)
    │  └─ ExchangeTruthAuditor (P3, reconciliation)               │
    ├──────────────────────────────────────────────────────────────┤
    │  DECISION ENGINES (P6)                                       │
    │  ├─ MetaController (Primary trading orchestrator)           │
    │  ├─ StrategyManager (Signal generation, caching)            │
    │  ├─ AgentManager (Multi-agent coordination)                 │
    │  └─ RiskManager (Capital limits, drawdown tracking)         │
    ├──────────────────────────────────────────────────────────────┤
    │  PROTECTIVE SERVICES (P7)                                    │
    │  ├─ TPSLEngine (Take-profit/Stop-loss automation)           │
    │  ├─ PnLCalculator (Portfolio accounting)                    │
    │  ├─ Watchdog (Safety circuit breaker)                       │
    │  ├─ Heartbeat (Liveness detection)                          │
    │  ├─ AlertSystem (Anomaly notifications)                     │
    │  └─ DustMonitor (Dust position tracking)                    │
    ├──────────────────────────────────────────────────────────────┤
    │  ANALYTICS & ORCHESTRATION (P8)                              │
    │  ├─ PerformanceMonitor (Trade metrics, analytics)           │
    │  ├─ CompoundingEngine (Dynamic equity growth)               │
    │  ├─ VolatilityRegime (Market regime detection)              │
    │  ├─ PortfolioBalancer (Multi-asset rebalancing)             │
    │  ├─ PerformanceEvaluator (Win rate, Sharpe, max DD)         │
    │  └─ LiquidationOrchestrator (Coordinated liquidation)       │
    ├──────────────────────────────────────────────────────────────┤
    │  LIQUIDITY & CAPITAL (P8)                                    │
    │  ├─ LiquidationAgent (Smart position liquidation)           │
    │  ├─ CashRouter (Liquidity routing)                          │
    │  ├─ CapitalAllocator (P9 wealth allocation)                 │
    │  └─ ProfitTargetEngine (P9 global profit guarding)          │
    ├──────────────────────────────────────────────────────────────┤
    │  BACKGROUND SUPPORT                                          │
    │  ├─ RecoveryEngine (Crash recovery, snapshots)              │
    │  ├─ UniverseRotationEngine (Universe rotation)              │
    │  ├─ AdaptiveCapitalEngine (Dynamic sizing)                  │
    │  ├─ DashboardServer (Web UI)                                │
    │  └─ WalletScannerAgent (P3, one-shot universe seed)         │
    └──────────────────────────────────────────────────────────────┘
```

---

## Initialization Phases Flowchart

```
START
  ↓
[P3] Exchange & Universe Bootstrap
  ├─ _gate_exchange_ready() → public session started, exchangeInfo warmed
  ├─ _ensure_exchange_signed_ready() → API keys loaded, signed mode attempted
  ├─ _attempt_fetch_balances() → early NAV/balances population
  ├─ _ensure_universe_bootstrap() → seed from config.SYMBOLS
  ├─ WalletScannerAgent.run_once() → populate from live balances (timeout 25s)
  ├─ _detect_restart_mode() → check for existing positions/intents/history
  └─ Declare startup policy: RECONCILIATION_ONLY or COLD_START
         ↓ (if restart or LIVE_MODE)
      [RECONCILIATION_ONLY] → no bootstrap, no forced trades, observe existing
      [COLD_START] → allows bootstrap if configured
         ↓
[P4] Market Data Feed (HARD GATE for P5)
  ├─ Start MarketDataFeed (warmup + background loop)
  ├─ Wait for: exchange_ready + universe_non_empty + market_data_ready
  ├─ Timeout: 180s; if exceeds → log warning, continue background
  └─ If gate fails: abort P5+ to avoid trading with stale data
         ↓ [P4 Gate Check]
      [PASS] → continue to P5
      [FAIL] → return (abort P5+)
         ↓
[P5] Execution Manager
  ├─ Preconditions: balances available, universe non-empty, exchange connected
  ├─ Start ExecutionManager (warmup symbol filters, balances)
  └─ Start AdaptiveCapitalEngine background monitor (5-min evaluation loop)
         ↓
[P6] Decision Engines & MetaController
  ├─ Authoritative wallet sync (exchange = source of truth)
  ├─ Start MetaController (primary trading decision engine)
  │  └─ Sanity check: verify _running flag; abort if fails
  ├─ Start StrategyManager (signal generation, caching)
  ├─ Start AgentManager (multi-agent coordination)
  └─ Start RiskManager (capital limits, drawdown tracking)
         ↓
[P7] Protective Services
  ├─ Start PnLCalculator (portfolio accounting)
  ├─ Start Heartbeat (liveness detection)
  ├─ Start Watchdog (safety circuit breaker)
  ├─ Start AlertSystem (anomaly notifications)
  ├─ Start TPSLEngine (take-profit/stop-loss, single instance, idempotent)
  └─ Start DustMonitor (dust position tracking)
         ↓
[P8] Analytics & Orchestration
  ├─ Start PerformanceMonitor (trade analytics)
  ├─ Start CompoundingEngine (dynamic equity growth)
  ├─ Start VolatilityRegime (market regime detection)
  ├─ Start PortfolioBalancer (multi-asset rebalancing)
  ├─ Start LiquidationAgent (smart liquidation)
  ├─ Start LiquidationOrchestrator (coordinated orchestration, one mode)
  ├─ Start PerformanceEvaluator (win rate, Sharpe, max DD)
  ├─ Start CapitalAllocator (P9 wealth allocation)
  ├─ Start ProfitTargetEngine (P9 global profit guarding)
  └─ Start DashboardServer (web UI, optional config disable)
         ↓
[P9] Finalization & Health Reporting
  ├─ Announce runtime mode (Live / Paper / Testnet / Signal-Only)
  ├─ Execution probe (can we afford a market buy?)
  ├─ Optional readiness gating (WAIT_READY_SECS > 0)
  │  └─ Wait for gates: market_data, execution, capital, exchange, startup_sanity
  ├─ Start background loops:
  │  ├─ Affordability scout (round-robin symbol probing, 15s interval)
  │  ├─ UURE loop (universe rotation, 5-min interval, immediate startup run)
  │  └─ Periodic readiness logger (30s interval)
  └─ Mark _init_completed=True, emit INIT_COMPLETE, READINESS_TICK events
         ↓
READY ✅ (Trading can commence)
```

---

## Component Dependency Graph

```
Legend:
  → = "receives" or "requires"
  ↔ = "bi-directional communication"

TIER 0: Infrastructure (Exchange & Data)
  ┌─────────────────────────────────────────┐
  │  ExchangeClient                         │
  │  ├─ → SymbolManager                     │
  │  ├─ → MarketDataFeed                    │
  │  ├─ → ExecutionManager                  │
  │  └─ → LiquidationAgent / Orchestrator   │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  SharedState (Central Hub)              │
  │  ├─ ↔ MetaController                    │
  │  ├─ ↔ ExecutionManager                  │
  │  ├─ ↔ RiskManager                       │
  │  ├─ ↔ StrategyManager                   │
  │  ├─ ← TPSLEngine                        │
  │  ├─ ← PerformanceMonitor                │
  │  ├─ ← Watchdog / Heartbeat              │
  │  └─ → 23 total components               │
  └─────────────────────────────────────────┘

TIER 1: Market Data & Universe
  ┌─────────────────────────────────────────┐
  │  MarketDataFeed                         │
  │  ├─ → MetaController (prices)           │
  │  ├─ → VolatilityRegime (OHLCV)          │
  │  └─ → CompoundingEngine (momentum)      │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  SymbolManager                          │
  │  ├─ → UniverseRotationEngine            │
  │  ├─ → LiquidationAgent (universe set)   │
  │  └─ → ExecutionManager (symbol filters) │
  └─────────────────────────────────────────┘

TIER 2: Execution Core
  ┌─────────────────────────────────────────┐
  │  ExecutionManager                       │
  │  ├─ ↔ MetaController                    │
  │  ├─ → TPSLEngine (order execution)      │
  │  ├─ → RiskManager (pre-flight checks)   │
  │  └─ → PerformanceMonitor (order events) │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  MetaController                         │
  │  ├─ → StrategyManager (signal queries)  │
  │  ├─ → AgentManager (agent queries)      │
  │  ├─ → RiskManager (capital checks)      │
  │  ├─ → TPSLEngine (exit management)      │
  │  ├─ → LiquidationOrchestrator (liq)    │
  │  ├─ → CashRouter (cash management)      │
  │  └─ → PerformanceEvaluator (metrics)    │
  └─────────────────────────────────────────┘

TIER 3: Decision & Risk
  ┌─────────────────────────────────────────┐
  │  StrategyManager                        │
  │  ├─ ↔ MetaController (signals)          │
  │  └─ → PerformanceMonitor (backtest)     │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  AgentManager                           │
  │  ├─ ↔ MetaController (decisions)        │
  │  └─ → LiquidationAgent (opportunities)  │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  RiskManager                            │
  │  ├─ ↔ MetaController (limits)           │
  │  ├─ ↔ ExecutionManager (validation)     │
  │  ├─ → CapitalSymbolGovernor (per-sym)   │
  │  └─ → Watchdog (circuit break)          │
  └─────────────────────────────────────────┘

TIER 4: Protective & Analytics
  ┌─────────────────────────────────────────┐
  │  TPSLEngine                             │
  │  ├─ ↔ ExecutionManager (mandatory exits)│
  │  └─ ↔ MetaController (exit coordination)│
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  Watchdog + Heartbeat                   │
  │  ├─ ↔ SharedState (health status)       │
  │  └─ → RiskManager (circuit breaks)      │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  PerformanceMonitor                     │
  │  ├─ ← ExecutionManager (order events)   │
  │  ├─ ← MetaController (trade loops)      │
  │  └─ → PerformanceEvaluator (metrics)    │
  └─────────────────────────────────────────┘

TIER 5: Advanced Control
  ┌─────────────────────────────────────────┐
  │  LiquidationOrchestrator                │
  │  ├─ → LiquidationAgent (coordination)   │
  │  ├─ ↔ MetaController (callback refresh) │
  │  └─ → CashRouter (opportunity capture)  │
  └─────────────────────────────────────────┘
         ↓
  ┌─────────────────────────────────────────┐
  │  CapitalAllocator                       │
  │  ├─ → StrategyManager (discovery)       │
  │  ├─ → AgentManager (discovery)          │
  │  ├─ → ExecutionManager (validation)     │
  │  ├─ → LiquidationAgent (opportunity)    │
  │  └─ → RiskManager (limits)              │
  └─────────────────────────────────────────┘
```

---

## Readiness Gate Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│  P3: EXCHANGE_READY Gate                                   │
│  ├─ ExchangeClient.public_session_started = True           │
│  ├─ exchangeInfo cache warmed                              │
│  └─ [PASS] → Proceed to P4                                 │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  P3: UNIVERSE_READY Gate                                   │
│  ├─ SharedState.get_accepted_symbols() returns non-empty   │
│  └─ [PASS] → Proceed to P4                                 │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  P4: MARKET_DATA_READY Gate (HARD BLOCK)                  │
│  ├─ SharedState.is_market_data_ready() = True              │
│  ├─ Timeout: 180s                                          │
│  └─ [FAIL] → Abort P5+ (safety: no trading with bad MDF)  │
└─────────────────────────────────────────────────────────────┘
         ↓ [P4 PASS]
┌─────────────────────────────────────────────────────────────┐
│  P5: BALANCES_READY Gate (Soft)                            │
│  ├─ shared_state.balances_ready OR free_usdt > 0           │
│  └─ [FAIL] → Proceed anyway, watch for capital gates       │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│  P9: Optional Readiness Gating (if WAIT_READY_SECS > 0)   │
│  ├─ market_data_ready (MarketDataNotReady)                 │
│  ├─ execution_ready (ExecutionManagerNotReady)             │
│  ├─ capital_ready (BalancesNotReady, NAVNotReady)          │
│  ├─ exchange_ready (ExchangeClientNotReady)                │
│  ├─ startup_sanity (FiltersCoverageLow, FreeQuoteBelowFloor)
│  └─ Timeout: WAIT_READY_SECS (e.g., 120 = 2 min)          │
└─────────────────────────────────────────────────────────────┘
```

---

## SharedState Component Wiring (23 Total)

```
SET SHARED_STATE PROPAGATION
set_shared_state(ss)
  ├─ [Symbols & Universe]
  │  ├─ SymbolManager
  │  ├─ CapitalSymbolGovernor
  │  └─ UniverseRotationEngine
  ├─ [Market Data & Execution]
  │  ├─ MarketDataFeed
  │  ├─ ExecutionManager
  │  ├─ StrategyManager
  │  ├─ AgentManager
  │  └─ RiskManager
  ├─ [Portfolio & Liquidity]
  │  ├─ LiquidationAgent
  │  ├─ LiquidationOrchestrator
  │  ├─ CashRouter
  │  ├─ TPSLEngine
  │  └─ MetaController
  ├─ [Analytics & Monitoring]
  │  ├─ PerformanceMonitor
  │  ├─ AlertSystem
  │  ├─ Watchdog
  │  ├─ Heartbeat
  │  ├─ CompoundingEngine
  │  ├─ RecoveryEngine
  │  ├─ VolatilityRegime
  │  ├─ PerformanceEvaluator
  │  ├─ PortfolioBalancer
  │  └─ DustMonitor
  └─ [Central Hub]
     └─ ComponentStatusLogger (bind_shared_state)

RESULT: All 23 components reference the single canonical SharedState
```

---

## Configuration Hierarchy

```
┌────────────────────────────────────────┐
│  1. SharedState.dynamic_config (live)  │ ← Highest priority
└────────────────────────────────────────┘
         ↑
┌────────────────────────────────────────┐
│  2. Config object attributes           │ ← Python config file / object
└────────────────────────────────────────┘
         ↑
┌────────────────────────────────────────┐
│  3. Config dict (config.SYMBOLS, etc)  │ ← Dict-like config
└────────────────────────────────────────┘
         ↑
┌────────────────────────────────────────┐
│  4. Environment variables (os.getenv)  │ ← Lowest priority
└────────────────────────────────────────┘

LOOKUP METHOD:
  try:
    return shared_state.dynamic_config.get(key)
  except:
    pass
  try:
    return config.key  # attribute
  except:
    pass
  try:
    return config[key]  # dict
  except:
    pass
  return os.getenv(key)  # env var
```

---

## Key Methods Quick Reference

| Category | Method | Purpose |
|----------|--------|---------|
| **Config** | `_cfg(key, default)` | Hierarchical config lookup |
| | `_cfg_bool(key, default)` | Boolean with flexible parsing |
| | `_cfg_float(key, default)` | Float with safe conversion |
| | `_cfg_int(key, default)` | Integer with safe conversion |
| **Time** | `_loop_time()` | Monotonic time from event loop |
| **Tasks** | `_ff(awaitable, name)` | Fire-and-forget background task |
| | `_spawn(name, awaitable)` | Spawn and track task |
| **Calls** | `_try_call(obj, methods, args)` | Sync best-effort method call |
| | `_try_call_async(obj, methods)` | Async best-effort with await |
| **Attributes** | `_set_attr_if_missing(obj, name, val)` | Safe attribute assignment |
| **Startup** | `_ensure_components_built()` | Construct missing components |
| | `initialize_all(up_to_phase)` | Phased initialization (P3→P9) |
| **Health** | `_ops_plane_snapshot()` | Readiness check with issues list |
| | `_emit_summary(event, **kv)` | Structured logging + events |
| | `_emit_health_status(level, detail)` | Health status event |
| **Shutdown** | `shutdown(save_snapshot)` | Graceful teardown |

---

## Startup Policy Matrix

```
CONDITION                                → POLICY
═══════════════════════════════════════════════════════════════════
restart=True (existing positions)         → RECONCILIATION_ONLY
OR LIVE_MODE=True (always pure recon)     → (no forced entries)
                                           → (no seed trades)
                                           → (no capital overrides)
                                           → (observe existing portfolio)
───────────────────────────────────────────────────────────────────
restart=False (cold-start)                → COLD_START
AND LIVE_MODE=False (allows bootstrap)    → (bootstrap allowed)
                                           → (initial trades permitted)
                                           → (subject to BOOTSTRAP_SEED_ENABLED)
───────────────────────────────────────────────────────────────────
DEFAULT (no config override)              → COLD_START
                                           → (most permissive)
```

---

**Quick Reference Complete** ✅
