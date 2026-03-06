# AppContext Comprehensive Review
**Date:** 2024  
**File:** `core/app_context.py` (4,536 lines)  
**Purpose:** Central orchestrator for entire Octivault Trader system lifecycle  
**Status:** ✅ Well-architected, production-ready with clear phased initialization  

---

## 1. Executive Summary

**AppContext** is the core bootstrapping and lifecycle management engine for Octivault Trader. It:

- **Orchestrates 30+ system components** through phased initialization (P3→P9)
- **Implements strict vs. optional module loading** (fail-fast for required, graceful skip for optional)
- **Manages startup readiness gates** with comprehensive health diagnostics
- **Handles graceful shutdown** with ordered component teardown
- **Implements key fixes** for restart detection, restart-aware startup policy, and shared state identity validation
- **Provides extensive observability** through structured logging and summary events

**Key Design Principle:** *Single clean definition, defensive construction, automatic dependency wiring*

---

## 2. Architecture Overview

### 2.1 Core Philosophy

```
┌─────────────────────────────────────────┐
│         AppContext (Orchestrator)        │
│  - Phased init (P3→P9) with gates      │
│  - Graceful component lifecycle         │
│  - Readiness validation                │
│  - Health status tracking               │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│      30+ System Components               │
│  - Core: Exchange, SharedState, Exec    │
│  - Data: MarketDataFeed, SymbolManager  │
│  - Control: MetaController, Agents      │
│  - Protective: Watchdog, Heartbeat      │
│  - Analytics: Performance, Compound     │
│  - Liquidity: Orchestrator, Agent       │
└─────────────────────────────────────────┘
```

### 2.2 Module Import Strategy

**Strict Imports (Required):**
- ExchangeClient, MarketDataFeed, SharedState, ExecutionManager, RiskManager
- TPSLEngine, RecoveryEngine, Watchdog, Heartbeat
- SymbolManager, CapitalSymbolGovernor, UniverseRotationEngine
- ComponentStatusLogger, AgentManager, StrategyManager

**Optional Imports (Graceful Skip):**
- PerformanceMonitor, AlertSystem, CompoundingEngine, VolatilityRegime
- PortfolioBalancer, PnLCalculator, PerformanceEvaluator
- LiquidationOrchestrator, CashRouter, LiquidationAgent
- AdaptiveCapitalEngine, ExchangeTruthAuditor, DustMonitor
- WalletScannerAgent, DashboardServer, CapitalAllocator, ProfitTargetEngine

**Strategy:** _import_strict() fails fast on required modules, _import_optional() silently skips unavailable optional modules.

---

## 3. Initialization Phases (P3→P9)

### 3.1 Phase P3: Exchange & Universe Bootstrap

**Purpose:** Establish exchange connectivity and seed the trading universe

**Steps:**
1. **Exchange Public Ready** (`_gate_exchange_ready`)
   - Ensure ExchangeClient exists and public session started
   - Warm exchangeInfo cache for early consumers
   
2. **API Keys & Signed Mode** (`_ensure_exchange_signed_ready`)
   - Attempt to load API keys from environment (BINANCE_API_KEY, BINANCE_API_SECRET)
   - Elevate to signed mode if keys present
   - Emit PHASE_SKIP if keys absent (stay public-only)
   
3. **Balances Probe** (`_attempt_fetch_balances`)
   - Best-effort: fetch balances to clear BalancesNotReady gate early
   - Non-fatal: logs at DEBUG if fails
   
4. **Universe Bootstrap** (`_ensure_universe_bootstrap`)
   - If SharedState universe empty, seed from config.SYMBOLS
   - Prevents hardcoded defaults; config-driven only
   
5. **Wallet Scan** (Optional, P3.7)
   - One-shot wallet balance scan to populate universe from live portfolio
   - Timeout: 25s; non-fatal if exceeds
   - Emits P3_WALLET_SCAN_COMPLETED summary
   
6. **Restart Mode Detection** (Critical Fix #3, P3.62)
   - Detects if this is a restart with existing positions/intents/history
   - Enables proper restart-aware startup policy (reconciliation vs. cold-start)
   
7. **Startup Policy Declaration** (P3.65)
   - If restart OR live_mode: **PURE RECONCILIATION** (no forced entries, no seed trades)
   - If cold-start: **COLD_START** mode (allows bootstrap seed if configured)

**Readiness Gates:**
- ✅ ExchangeClientReady (required)
- ✅ Universe non-empty (required)
- ✅ Balances/NAV populated (required if attempting orders)

---

### 3.2 Phase P4: Market Data Feed

**Purpose:** Start real-time market data collection; hard gate for P5+

**Key Features:**
- **Warmup + Background Loop Contract**
  - Detects warmup/loop entrypoints (start_warmup, run_loop)
  - Runs warmup with timeout, spawns loop in background
  - Non-blocking after warmup completes
  
- **Timeout Handling**
  - Default: 90 seconds (P4_MARKET_DATA_START_TIMEOUT_SEC)
  - If exceeds: logs warning, continues in background
  - Tracked with _phase_timeout() summary
  
- **Readiness Gate (Hard Block)**
  - Waits for: exchange ready, universe non-empty, market data ready
  - Timeout: 180s (P4_MARKET_DATA_READY_TIMEOUT_SEC)
  - If fails: logs error, aborts P5+ to avoid trading with bad market data

**Status Check:**
- MetaController, ExecutionManager cannot start if market data unavailable
- Guard: `if not _p4_gate_ok: return`

---

### 3.3 Phase P5: Execution Manager

**Purpose:** Initialize trading execution paths and symbol filters

**Prerequisites:**
- ✅ P4 gate passed (market data ready)
- ✅ Balances available (balances_ready OR free_usdt > 0)
- ✅ Universe non-empty
- ✅ ExchangeClient available

**Actions:**
1. Start ExecutionManager with timeout (30s default)
2. Warm symbol filters and balances
3. Start AdaptiveCapitalEngine background monitoring task
4. Emit P5_EXECUTION_STARTED summary

**Adaptive Capital Engine:**
- Monitors capital-sizing decisions every 5 minutes
- Evaluates volatility, drawdown, fee/slippage impact
- Emits health status with current risk fraction and min trade quote
- Graceful on errors (logs and retries after 1 minute)

---

### 3.4 Phase P6: Strategy, Agents, Risk & MetaController

**Purpose:** Start decision engines and trading logic

**Critical Sequence:**
1. **Authoritative Wallet Sync** (Pre-MetaController)
   - `shared_state.authoritative_wallet_sync()` — exchange is source of truth
   - Falls back to `hard_reset_capital_state()` if unavailable
   - **Non-negotiable:** Must complete before MetaController starts
   
2. **MetaController Startup** (P6_meta_controller)
   - Primary trading decision engine
   - Sanity check: verify `_running` flag after start
   - Abort downstream if fails (prevents trading with broken meta controller)
   
3. **StrategyManager** (P6_strategy)
   - Signal generation and caching
   - Enhanced with NAV source and position count source (recent fix)
   
4. **AgentManager** (P6_agent_manager)
   - Multi-agent orchestration
   - Discovery and lifecycle management
   
5. **RiskManager** (P6_risk_manager)
   - Capital limits, position sizing
   - Drawdown tracking and protective actions

**Component Registration:**
- MetaController registered with SharedState before start
- Status updated as "Initialized" → "Running"

---

### 3.5 Phase P7: Protective Services

**Purpose:** Start safety mechanisms and telemetry

**Components (In Order):**
1. **PnLCalculator** — Portfolio accounting and P&L tracking
2. **Heartbeat** — Liveness detection and failover signals
3. **Watchdog** — Safety circuit breaker
4. **AlertSystem** — Anomaly notifications
5. **TPSLEngine** — Take-profit/stop-loss automation
   - Enforces single instance (idempotent guard)
   - Injected with TradeJournal and session_id
   - Wired into ExecutionManager and MetaController
6. **DustMonitor** — Real-time dust position monitoring

**Critical Injections:**
- TPSLEngine → ExecutionManager (mandatory exit paths)
- TPSLEngine → MetaController (exit coordination)

---

### 3.6 Phase P8: Analytics, Portfolio & Orchestration

**Purpose:** Enable advanced portfolio management and liquidity coordination

**Components (In Order):**
1. **PerformanceMonitor** — Trade analytics and metrics
2. **CompoundingEngine** — Dynamic equity compounding
3. **VolatilityRegime** — Market regime detection
4. **PortfolioBalancer** — Multi-asset rebalancing
5. **LiquidationAgent** — Smart position liquidation (optional)
6. **LiquidationOrchestrator** — Coordinated liquidity orchestration (optional)
7. **PerformanceEvaluator** — Win rate, Sharpe, max DD analytics
8. **DashboardServer** — Web UI (optional, can disable via config)
9. **CapitalAllocator** — P9 wealth allocation (optional, can disable via config)
10. **ProfitTargetEngine** — P9 global profit guarding

**Liquidity Mode Enforcement:**
- Only ONE of: cash_router, orchestrator, agent, event_bus
- Others nulled to prevent ambiguous paths
- Mode set via LIQUIDITY_ORCHESTRATION_MODE config

**Optional Wiring:**
- LiquidationOrchestrator completion callback → MetaController.refresh_cash_router()
- PerformanceEvaluator registered with SharedState before start

---

### 3.7 Phase P9: Finalization & Health Reporting

**Purpose:** Announce readiness and begin monitoring

**Steps:**
1. **Announce Runtime Mode**
   - Live / Paper / Testnet / Signal-Only
   - Emit RuntimeModeChanged event to SharedState
   
2. **Execution Probe** (`_dry_probe_execution`)
   - Quick affordability check: can we execute a market buy?
   - Uses SharedState.affordability_snapshot() or ExecutionManager.can_afford_market_buy()
   - Determines min-notional floor dynamically
   
3. **Readiness Gates** (Optional blocking, if WAIT_READY_SECS > 0)
   - Wait for: market_data, execution, capital, exchange, startup_sanity
   - Timeout: WAIT_READY_SECS (0 = non-blocking)
   - Default gates if timeout set: all 5 gates
   
4. **Periodic Readiness Logger** (Background, 30s interval)
   - Emits READINESS_TICK every 30 seconds
   - Shows issues, liquidity gaps, dust metrics
   - Health level: OK / DEGRADED / ERROR based on blockers
   
5. **Initialization Complete**
   - Mark `_init_completed = True`
   - Record `_init_highest_phase = max(9, prior)`
   - Emit INIT_COMPLETE summary
   - Emit health status (OK if ready, DEGRADED otherwise)

---

## 4. Key Components & Dependencies

### 4.1 Core Execution Path

```
ExchangeClient
    ↓
SharedState (balances, positions, NAV)
    ↓
SymbolManager (universe)
    ↓
MarketDataFeed (prices, OHLCV)
    ↓
ExecutionManager (order execution)
    ↓
MetaController (trading decisions)
    ↓
TPSLEngine (mandatory exits)
    ↓
RiskManager (limits, circuit breaks)
```

### 4.2 Shared State Propagation

**Components receiving SharedState (23 total):**
- Data: SymbolManager, CapitalSymbolGovernor, UniverseRotationEngine, MarketDataFeed
- Execution: ExecutionManager, StrategyManager, AgentManager, RiskManager, LiquidationAgent
- Portfolio: LiquidationOrchestrator, CashRouter, TPSLEngine, PerformanceMonitor
- Analytics: AlertSystem, Watchdog, Heartbeat, CompoundingEngine, RecoveryEngine
- Advanced: VolatilityRegime, PerformanceEvaluator, PortfolioBalancer, DustMonitor
- Central: MetaController

**Method:** `_components_for_shared_state()` returns complete list; `set_shared_state()` propagates via method call or attribute assignment.

### 4.3 Exchange Client Propagation

**Components receiving ExchangeClient (11 total):**
- SymbolManager, MarketDataFeed, ExecutionManager, StrategyManager, AgentManager
- RiskManager, LiquidationAgent, LiquidationOrchestrator, CashRouter, ExchangeTruthAuditor

**Method:** `_propagate_exchange_client()` runs after exchange client construction/start.

---

## 5. Critical Fixes & Design Decisions

### 5.1 Restart Mode Detection (Critical Fix #3)

**Problem:** On restart, system saw existing positions as errors needing liquidation.

**Solution:** `_detect_restart_mode()` checks 5 conditions:
1. Existing positions in portfolio
2. Pending accumulation intents
3. Prior execution history (total_trades > 0)
4. NAV history (more than just current snapshot)
5. Database/snapshot files exist

**Impact:**
- If restart detected: **PURE RECONCILIATION** mode (no forced entries, no seed trades)
- If cold-start: **COLD_START** mode (allows bootstrap seed if configured)
- Sets `shared_state._is_restart` for downstream code

---

### 5.2 Startup Policy Declaration (P3.65)

**Three Modes:**
1. **RECONCILIATION_ONLY** (restart or LIVE_MODE)
   - No bootstrap seed
   - No forced entry trades
   - No capital overrides
   - No confidence bypasses
   - Existing positions observed and managed per strategy

2. **COLD_START** (first-ever launch)
   - Bootstrap seed allowed if BOOTSTRAP_SEED_ENABLED=True
   - Can fire initial trades to test system
   - Flexible confidence requirements

3. **LIVE_MODE** (always pure reconciliation)
   - Regardless of restart detection
   - Emphasis: "No forced entries. No seed trades. No capital overrides."

---

### 5.3 Shared State Identity Enforcement

**Method:** `_enforce_shared_state_identity()`

**Purpose:** Ensure MetaController, TPSLEngine, ExecutionManager, RiskManager all reference the single canonical SharedState.

**Action:**
- Checks each core component's `shared_state` attribute
- If mismatch detected: rewires to canonical instance
- Logs rewired components
- If STRICT_SHARED_STATE_IDENTITY=True: raises RuntimeError on any mismatch

**Critical:** Prevents subtle bugs from component-specific SharedState copies.

---

### 5.4 TP/SL Engine Wiring (P7)

**Problem:** Multiple TP/SL engines or missing wiring → trading with incomplete exit coverage.

**Solution:**
- Idempotent single-instance guard in P7
- Explicit wiring into ExecutionManager: `exec.set_tp_sl_engine(tp_sl_engine)`
- Explicit wiring into MetaController for exit coordination
- TradeJournal and session_id injected for audit trail

---

### 5.5 Liquidity Plane Enforcement (P8)

**Problem:** Multiple liquidity components with ambiguous precedence.

**Solution:** `enforce_single_liquidity_mode()`
- Reads LIQUIDITY_ORCHESTRATION_MODE config
- Keeps ONE of: cash_router, orchestrator, agent, event_bus
- Nulls others to prevent ambiguous paths
- Validates mode has required interface before proceeding

---

### 5.6 Authoritative Wallet Sync (P6, Pre-MetaController)

**Problem:** Stale balances from initialization caused capital gate failures.

**Solution:**
- Hard requirement: `shared_state.authoritative_wallet_sync()` before MetaController.start()
- Falls back to `shared_state.hard_reset_capital_state()`
- Logs with warning level: "[BOOT] Authoritative wallet sync (exchange is source of truth)"
- Non-negotiable guard: `if hasattr(...) await ...`

---

## 6. Observability & Health Diagnostics

### 6.1 Summary Events (Structured Logging)

All events go through `_emit_summary(event, **kvs)` and hit logs + SharedState bus:

**Categories:**
- **PHASE_***: PHASE_START, PHASE_DONE, PHASE_TIMEOUT, PHASE_ERROR, PHASE_SKIP
- **INIT_***: INIT_START, INIT_COMPLETE, INIT_EXCEPTION, INIT_GATES_TIMEOUT, INIT_ISSUES, INIT_BLOCKED_CORE_GATES
- **BOOT_***: BOOT_INVENTORY
- **P3_***: P3_EXCHANGE_PUBLIC_READY, P3_WALLET_SCAN_COMPLETED, P3_TRUTH_AUDITOR
- **P4_***: P4_MARKET_DATA_DECISION
- **P5_***: P5_EXECUTION_STARTED, P5_EXECUTION_SKIPPED
- **P6_***: P6_STACK_STARTED
- **P7_***: P7_PROTECTIVE_STARTED
- **P8_***: P8_ANALYTICS_STARTED
- **LIQUIDITY_***: LIQUIDITY_NEEDED, LIQUIDITY_TRIGGERED, LIQUIDITY_MODE_VALIDATED, LIQUIDITY_RESULT, LIQUIDITY_ORCH_MISSING
- **READINESS_***: READINESS_TICK
- **STARTUP_POLICY**: Mode declaration (RECONCILIATION_ONLY or COLD_START)
- **SHARED_STATE_IDENTITY**: Identity verification result

### 6.2 Health Status Events

**Levels:**
- OK: All systems nominal
- DEGRADED: Core systems up, but capital/liquidity constrained
- ERROR: Infrastructure failure (no exchange, no market data)
- STARTING: Initialization in progress
- SHUTDOWN: Graceful teardown

**Events:**
- `_emit_health_status(level, details_dict)`
- Includes issues list, component status, liquidity gaps, dust metrics

### 6.3 Readiness Snapshot (`_ops_plane_snapshot`)

**Returns:** `{ready: bool, issues: List[str], detail: dict}`

**Issues Checked:**
- ExchangeClientNotReady
- MarketDataNotReady
- SymbolsUniverseEmpty
- ExecutionManagerNotReady
- BalancesNotReady / NAVNotReady
- FiltersCoverageLow
- FreeQuoteBelowFloor
- ZeroQuantityExecution (Rule 2)

**Detail Fields:**
- AffordabilityProbe: `{symbol, ok, amount, code, planned_quote, required_min_quote}`
- Liquidity: Per-symbol gaps and consecutive tick counts
- Dust: Registry size, origin breakdown, policy conflicts

---

## 7. Helper Methods & Utilities

### 7.1 Configuration & Loop Helpers

| Method | Purpose |
|--------|---------|
| `_cfg(key, default)` | Lookup with SharedState.dynamic_config priority, then config attrs, dict, env |
| `_cfg_bool(key, default)` | Boolean config with flexible parsing |
| `_cfg_float(key, default)` | Float config with safe conversion |
| `_cfg_int(key, default)` | Integer config with safe conversion |
| `_loop_time()` | Monotonic time from event loop (or manual fallback) |
| `_start_timeout_sec(phase_key)` | Per-phase timeout lookup with defaults |

### 7.2 Component Helpers

| Method | Purpose |
|--------|---------|
| `_try_call(obj, method_names, *args, **kwargs)` | Sync best-effort call of first available method |
| `_try_call_async(obj, method_names, *args, **kwargs)` | Async best-effort call with await support |
| `_maybe_await(value)` | Awaits coroutine or returns value directly |
| `_set_attr_if_missing(obj, name, value)` | Safe attribute assignment (only if missing) |
| `_symbols_list_to_dict(symbols)` | Convert list to `{SYMBOL: {'enabled': True, 'meta': {}}}` |

### 7.3 Fire-and-Forget Task Scheduling

```python
def _ff(self, aw: Awaitable, *, name: str | None = None):
    """Schedule coroutine as background task; never raises; handles no-loop."""
    try:
        loop = asyncio.get_running_loop()
        if name:
            return loop.create_task(aw, name=name)
        return loop.create_task(aw)
    except RuntimeError:
        return None  # No running loop yet
```

**Usage:** `self._ff(self._delayed_resync_liq_symbols(30.0))`

---

## 8. Background Tasks & Loops

### 8.1 Affordability Scout Loop (`_affordability_scout_loop`)

**Purpose:** Round-robin symbol affordability probing to detect liquidity gaps

**Behavior:**
- Iterates over accepted symbols (optionally USDT-quoted only)
- Calls ExecutionManager.can_afford_market_buy() per symbol
- Emits LIQUIDITY_NEEDED throttled summaries
- Tracks per-symbol deficit state (consecutive ticks, gap amount)

**Configuration:**
- AFFORD_SCOUT_ENABLE (default: True)
- AFFORD_SCOUT_INTERVAL_SEC (default: 15s)
- AFFORD_SCOUT_JITTER_PCT (default: 10%)
- AFFORD_SCOUT_ONLY_USDT (default: True)

**Startup:** Triggered after P9 readiness gates clear

---

### 8.2 Universe Rotation Engine Loop (`_uure_loop`)

**Purpose:** Periodic universe rotation (add/remove symbols based on performance)

**Behavior:**
- Runs immediately once at P9 startup (critical for universe population)
- Then periodic calls every UURE_INTERVAL_SEC (default: 300s = 5 min)
- Calls `universe_rotation_engine.compute_and_apply_universe()`
- Logs rotation results: added, removed, kept counts

**Configuration:**
- UURE_ENABLE (default: True)
- UURE_INTERVAL_SEC (default: 300s)

**Startup:** Triggered after P9 readiness gates clear

---

### 8.3 Periodic Readiness Logger (`_periodic_readiness_log`)

**Purpose:** Emit READINESS_TICK every 30 seconds for visibility

**Output:**
- Issues list (blockers and warnings)
- Liquidity gaps per symbol
- Dust metrics (registry size, origin breakdown, external %)
- Health level determination (OK / DEGRADED / ERROR)

**Lifecycle:** Spawned in P9; runs until shutdown

---

### 8.4 Adaptive Capital Engine Monitor (`_run_adaptive_capital_monitor`)

**Purpose:** Background monitoring of adaptive sizing decisions

**Behavior:**
- Evaluates every 5 minutes
- Gathers: NAV, free capital, volatility, drawdown, fee/slippage, throughput
- Calls `adaptive_capital_engine.evaluate(...)`
- Logs: risk_fraction, min_trade_quote, win_rate, avg_r_multiple
- Emits health status with current adaptive metrics

**Lifecycle:** Started in P5 after ExecutionManager ready

---

## 9. Graceful Shutdown

**Method:** `async def shutdown(self, save_snapshot: bool = False)`

**Sequence:**
1. Stop background scout (`_stop_affordability_scout`)
2. Stop UURE loop (`_stop_uure_loop`)
3. Ordered component stop (controllers → data/infra → liquidity)
   - See `_components_for_shutdown()` for order
4. Stop exchange client (last, network connections)
5. Optional snapshot save (SharedState or RecoveryEngine)
6. Cancel stray tasks
7. Emit SHUTDOWN_DONE summary

**Contract:** Never raises; logs errors at DEBUG level for robustness.

---

## 10. Integration Points with SignalManager

**Recent Enhancement:** NAV source and position count source parameters

**Integration in AppContext:**
1. SignalManager constructed in P6 (via StrategyManager) or earlier
2. Receives `shared_state` parameter (NAV source)
3. Optionally receives `position_count_source` callable
4. Methods: `get_current_nav()`, `get_position_count()`

**NAV Retrieval Fallback Chain:**
1. shared_state.nav → portfolio_nav → total_equity_usdt
2. Safe default: 0.0 on any error

**Position Count Retrieval Fallback Chain:**
1. position_count_source() callable (if provided)
2. shared_state.get_positions_snapshot()
3. Safe default: 0 on any error

---

## 11. Configuration Files & Defaults

### 11.1 Key Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| VOLATILITY_REGIME_TIMEFRAME | "1h" | Brain: slow, strategic regime detection |
| ohlcv_timeframes | ["5m", "1h"] | Hands: 5m execution, Brain: 1h regime |
| START_TIMEOUT_SEC | 30.0 | Component start timeout |
| P4_MARKET_DATA_START_TIMEOUT_SEC | 90.0 | MDF warmup timeout |
| P4_MARKET_DATA_READY_TIMEOUT_SEC | 180.0 | MDF readiness gate timeout |
| WAIT_READY_SECS | 0 | P9 blocking readiness timeout (0 = non-blocking) |
| GATE_READY_ON | "" | P9 gate list (comma-separated) |
| LIQUIDITY_ORCHESTRATION_MODE | "agent" | One of: cash_router, orchestrator, agent, event_bus |
| LIQ_ORCH_ENABLE | True | Enable liquidity orchestration |
| AFFORD_SCOUT_ENABLE | True | Enable affordability scout |
| AFFORD_SCOUT_INTERVAL_SEC | 15 | Scout polling interval |
| UURE_ENABLE | True | Enable universe rotation engine |
| UURE_INTERVAL_SEC | 300 | UURE polling interval (5 min) |
| STRICT_SHARED_STATE_IDENTITY | False | Raise on SharedState identity mismatch |
| SYMBOLS / SEED_SYMBOLS | [] | Initial universe seeds |
| TESTNET_MODE | False | Testnet vs. live |
| PAPER_MODE | False | Paper trading (no actual orders) |
| SIGNAL_ONLY | False | Signal monitoring without trading |
| LIVE_MODE | False | Always use pure reconciliation startup |
| BOOTSTRAP_SEED_ENABLED | True | Allow bootstrap seed trades (cold-start only) |
| COLD_BOOTSTRAP_ENABLED | True | Enable cold bootstrap mode |

### 11.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| BINANCE_API_KEY | Exchange API key |
| BINANCE_API_SECRET | Exchange API secret |
| APP_LOG_FILE | Optional log file path |
| TF_CPP_MIN_LOG_LEVEL | TensorFlow logging level (0-3) |
| OMP_NUM_THREADS | BLAS thread count |
| MKL_NUM_THREADS | Intel MKL thread count |

---

## 12. Design Patterns & Best Practices

### 12.1 Signature-Aware Construction

```python
def _try_construct(cls: Optional[type], **candidate_kwargs) -> Optional[Any]:
    sig = inspect.signature(cls.__init__)
    allowed = {k for k in sig.parameters.keys() if k != "self"}
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}
    return cls(**kwargs)
```

**Benefit:** Automatically filters kwargs to match target __init__ signature; no manual parameter lists.

### 12.2 Module Import Pattern

```python
_exchange_mod = _import_strict("core.exchange_client")  # Fail-fast required
_perf_mod = _import_optional("core.performance_monitor")  # Graceful skip
```

**Benefit:** Clear distinction between required (must succeed) and optional (nice-to-have) modules.

### 12.3 Timeout-Aware Startup

```python
try:
    await asyncio.wait_for(asyncio.shield(start_task), timeout=timeout)
except asyncio.TimeoutError:
    self.logger.warning("[%s] timed out at %.1fs — continuing in background", phase_key, timeout)
    # Task continues; non-blocking after timeout
```

**Benefit:** Doesn't block startup on slow components; logs timeout but continues.

### 12.4 Readiness Gating

```python
def _blocked(snap: dict) -> List[str]:
    issues = set(snap.get("issues", []))
    blocks: List[str] = []
    if "market_data" in want and "MarketDataNotReady" in issues:
        blocks.append("market_data")
    # ... more gates ...
    return blocks

# Wait until gates clear or timeout
while blocked and self._loop_time() < deadline:
    await asyncio.sleep(poll_sec)
    snap = await self._ops_plane_snapshot()
    blocked = _blocked(snap)
```

**Benefit:** Flexible gating with configurable timeout and polling interval.

---

## 13. Known Limitations & Future Improvements

### 13.1 Current Limitations

1. **Sequential Phase Execution:** P3→P9 runs serially; no parallel phase startup
   - *Mitigation:* Background loops (scout, UURE) provide concurrency after phases complete

2. **No Automatic Retry:** Phase failures not retried; user must restart
   - *Mitigation:* Graceful degradation; optional components skip silently

3. **Limited Component Introspection:** Dynamic method detection via hasattr()
   - *Mitigation:* Explicit _phase_key attributes on components help

4. **Shared State as Single Canary:** All capital/nav flows through one object
   - *Mitigation:* Identity enforcement prevents copies; periodic snapshots logged

### 13.2 Future Improvements

1. **Phase Parallelization:** Start independent phases concurrently
2. **Automatic Retry Logic:** Configurable retries with exponential backoff
3. **Component Registry:** Explicit registration instead of dynamic discovery
4. **Health-Based Cascading:** Prevent P5+ if P4 gate unhealthy
5. **Warm Shutdown:** Graceful position close before shutdown

---

## 14. Testing & Validation Checklist

### 14.1 Unit Tests

- [ ] _import_strict() fails on missing required modules
- [ ] _import_optional() skips gracefully on missing modules
- [ ] _try_construct() filters kwargs correctly
- [ ] _enforce_shared_state_identity() detects mismatches
- [ ] _detect_restart_mode() returns True/False correctly
- [ ] set_shared_state() propagates to all 23 target components
- [ ] _propagate_exchange_client() updates all 11 target components

### 14.2 Integration Tests

- [ ] Phase P3→P9 runs without exceptions
- [ ] Readiness gates block until clear
- [ ] Affordability scout detects liquidity gaps
- [ ] UURE loop rotates universe
- [ ] Background tasks start and stop cleanly
- [ ] Shutdown completes without hanging

### 14.3 Scenario Tests

- [ ] Cold-start from empty universe
- [ ] Restart with existing positions (RECONCILIATION_ONLY mode)
- [ ] Live mode forces pure reconciliation
- [ ] Missing optional component skips gracefully
- [ ] Phase timeout non-blocking but logged
- [ ] Health status transitions correctly (STARTING → OK → DEGRADED)

---

## 15. Conclusion

**AppContext** is a well-designed, production-ready orchestration engine with:

✅ **Clear phased architecture** (P3→P9) with explicit gates  
✅ **Defensive construction** (strict vs. optional, signature-aware)  
✅ **Comprehensive observability** (structured logging, health status, readiness snapshots)  
✅ **Critical fixes** (restart detection, startup policy, shared state identity)  
✅ **Graceful degradation** (optional components skip, timeouts non-blocking)  
✅ **Full lifecycle management** (construction → startup → monitoring → shutdown)  

**Recommended Deployment:**
- Ensure BINANCE_API_KEY and BINANCE_API_SECRET in environment
- Set SYMBOLS or SEED_SYMBOLS in config for initial universe
- Configure WAIT_READY_SECS and GATE_READY_ON for startup blocking
- Monitor READINESS_TICK and health status events for visibility
- Use LIVE_MODE=True for production reconciliation-only startup

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Review Status:** ✅ Complete
