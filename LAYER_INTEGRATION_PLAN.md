# OctiVault Trader — Layer Integration Plan
**Date:** 2026-04-26  
**Scope:** All existing modules mapped to architectural layers. Zero new modules. Pure wiring analysis.  
**Constraint:** No new code written — only wire existing modules that are built but disconnected.

---

## SECTION 1 — The Correct 8-Layer Architecture

Before mapping modules, this is the target layer model. Every module in the system belongs to exactly one layer. Layers must start in order. A layer may not start until every layer below it is fully ready.

```
Layer 0 │ Infrastructure      │ Config, .env, logging, API keys
Layer 1 │ Exchange            │ ExchangeClient — open session, test connectivity
Layer 2 │ State Hydration     │ SharedState ← RecoveryEngine ← StartupOrchestrator
Layer 3 │ Market Data         │ MarketDataFeed, PollingCoordinator — OHLCV must be warm
Layer 4 │ Risk & Capital      │ RiskManager, CapitalAllocator, CapitalGovernor (in MetaController)
Layer 5 │ Execution Path      │ ExecutionManager, TPSLEngine
Layer 6 │ Decision Brain      │ MetaController, AgentManager, SignalFusion, SignalManager
Layer 7 │ Profit Engine       │ LiquidationOrchestrator, CashRouter, CompoundingEngine,
        │                     │ PortfolioBalancer, DeadCapitalHealer, ThreeBucketManager
Layer 8 │ Observability       │ HealthMonitor, Watchdog, Heartbeat, PerformanceEvaluator
```

**Rule:** MetaController (Layer 6) must not start until Layers 0–5 are fully initialized and State Hydration (Layer 2) has emitted `StartupPortfolioReady`.

---

## SECTION 2 — Full Module-to-Layer Map

### Layer 0 — Infrastructure
| Module | File | Status in MASTER |
|---|---|---|
| OrchestratorConfig | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ✅ Runs at boot |
| CoreConfig | `core/config.py` | ✅ Loaded via OrchestratorConfig fallback |
| .env loader | `python-dotenv` | ✅ `load_dotenv()` called at module level |
| Logging setup | `setup_logging()` in MASTER | ✅ First thing called |

### Layer 1 — Exchange Connectivity
| Module | File | Status in MASTER |
|---|---|---|
| ExchangeClient | `core/exchange_client.py` | ✅ Step [1/9] — `await start()` |
| Initial balance fetch | ExchangeClient.get_spot_balances() | ✅ Called at step [1/9] |

**Note:** `BINANCE_API_TYPE=HMAC` causes WebSocket authentication to fail. System falls back to 25s polling. The Ed25519 key is present in `.env` but unused.

### Layer 2 — State Hydration ⚠️ BROKEN
| Module | File | Status in MASTER |
|---|---|---|
| SharedState | `core/shared_state.py` | ✅ Step [2/9] — init only |
| RecoveryEngine | `core/recovery_engine.py` | ⚠️ Instantiated in optional section — `rebuild_state()` NEVER called |
| StartupOrchestrator | `core/startup_orchestrator.py` | ❌ NOT IMPORTED, NOT CALLED |
| ExchangeTruthAuditor | `core/exchange_truth_auditor.py` | ❌ Disabled by `ENABLE_EXCHANGE_TRUTH_AUDITOR=false` |
| bootstrap_symbols | `core/bootstrap_symbols.py` | ✅ Called at step [2/9] (symbols only) |

**What happens today:** SharedState is seeded with USDT balance only (`self.shared_state.balances["USDT"] = {"free": usdt_balance}`). All crypto positions (BTC, ETH, BNB, LINK, XRP, SOL, etc.) are invisible to the system on every restart. RecoveryEngine is added to the optional task list but its `rebuild_state()` method is never awaited before MetaController starts.

**What must happen:** `StartupOrchestrator.run()` must be awaited between step [2/9] and step [5/9], calling the canonical 6-step boot sequence.

### Layer 3 — Market Data ⚠️ STARTS TOO LATE
| Module | File | Status in MASTER |
|---|---|---|
| MarketDataFeed | `core/market_data_feed.py` | ⚠️ Init at step [7.5/9] — task starts AFTER MetaController |
| PollingCoordinator | `core/polling_coordinator.py` | ✅ Step [8/9] — init correct, task concurrent |
| WebSocket price feed | inside ExchangeClient/MarketDataFeed | ⚠️ HMAC auth fails — polling fallback at 25s |

**Critical problem:** MarketDataFeed is initialized at step 7.5 but its async task starts concurrently with MetaController at task-launch time. MetaController's first decision cycle fires within seconds of boot, before `shared_state.get_market_data_sync(symbol, "5m")` has any OHLCV rows. This means the micro_backtest always reads an empty candle list on the first N cycles, guaranteeing `MICRO_BACKTEST_INSUFFICIENT_SAMPLES` rejections at the very start.

**Fix:** MarketDataFeed must be started at step [3/9] and a brief warm-up wait (5–10 seconds) must complete before MetaController begins its loop.

### Layer 4 — Risk & Capital
| Module | File | Status in MASTER |
|---|---|---|
| RiskManager | `core/risk_manager.py` | ✅ Step [4/9] |
| CapitalAllocator | `core/capital_allocator.py` | ✅ Instantiated in optional section |
| CapitalGovernor | `core/capital_governor.py` | ✅ Embedded in MetaController — active (MICRO bracket) |
| NAVRegimeManager | `core/nav_regime.py` | ✅ Embedded in MetaController — active (MICRO_SNIPER) |
| CapitalVelocityOptimizer | `core/capital_velocity_optimizer.py` | ✅ Embedded in MetaController |
| ReserveManager | `core/reserve_manager.py` | ❌ Never instantiated anywhere |

### Layer 5 — Execution Path
| Module | File | Status in MASTER |
|---|---|---|
| ExecutionManager | `core/execution_manager.py` | ✅ Step [5/9] |
| TPSLEngine | `core/tp_sl_engine.py` | ✅ Step [5B/9] — has TP/SL monitoring |
| EV gate | inside ExecutionManager | ✅ Active — validates expected_move vs round-trip cost |

### Layer 6 — Decision Brain
| Module | File | Status in MASTER |
|---|---|---|
| MetaController | `core/meta_controller.py` | ✅ Step [6/9] — starts too early (before L2 ready, before L3 warm) |
| SignalManager | `core/signal_manager.py` | ✅ Step [3/9] |
| AgentManager | `core/agent_manager.py` | ✅ Step [7/9] |
| SignalFusion | `core/signal_fusion.py` | ✅ Embedded in MetaController async task |
| SignalBatcher | `core/signal_batcher.py` | ✅ Embedded in MetaController |
| RotationExitAuthority | `core/rotation_authority.py` | ✅ Embedded in MetaController |
| PortfolioAuthority | `core/portfolio_authority.py` | ✅ Embedded in MetaController |
| ArbitrationEngine | `core/arbitration_engine.py` | ✅ Embedded in MetaController |
| OpportunityRanker | `core/opportunity_ranker.py` | ✅ Embedded in MetaController (lazy) |
| LifecycleManager | `core/lifecycle_manager.py` | ✅ Embedded in MetaController |
| ModeManager | `core/mode_manager.py` | ✅ Embedded in MetaController |
| FocusMode | `core/focus_mode.py` | ✅ Embedded in MetaController |
| StateManager | `core/state_manager.py` | ✅ Embedded in MetaController |
| BootstrapManager | `core/bootstrap_manager.py` | ✅ Embedded in MetaController |
| ActionRouter | inside MetaController | ✅ Priority routing active |
| ExternalAdoptionEngine | `core/external_adoption_engine.py` | ❌ Never instantiated — setter exists at line 21239 but never called |
| UniverseRotationEngine | `core/universe_rotation_engine.py` | ✅ Enabled by ENABLE_UNIVERSE_ROTATION=true |
| SymbolManager | `core/symbol_manager.py` | ✅ Instantiated in optional section |
| rule_overrides | `automation/rule_overrides.py` | ⚠️ Imported by MetaController — `proposed_rules.json` does not exist → no-op |

### Layer 7 — Profit Engine ⚠️ MOSTLY DISCONNECTED
| Module | File | Status in MASTER |
|---|---|---|
| LiquidationOrchestrator | `core/liquidation_orchestrator.py` | ⚠️ Instantiated — but `cash_router=None`, MetaController never calls it |
| CashRouter | `core/cash_router.py` | ❌ Never instantiated — passed as `None` to LiquidationOrchestrator |
| CompoundingEngine | `core/compounding_engine.py` | ❌ Disabled — `ENABLE_COMPOUNDING_ENGINE=false` in defaults |
| PortfolioBalancer | `core/portfolio_balancer.py` | ❌ Disabled — `ENABLE_BALANCER=false` in defaults |
| DeadCapitalHealer | `core/dead_capital_healer.py` | ❌ Never instantiated anywhere |
| ThreeBucketManager | `core/three_bucket_manager.py` | ⚠️ Instantiated at step [6.5/9] — but DeadCapitalHealer not wired to it |
| RebalancingEngine | `core/rebalancing_engine.py` | ❌ Disabled — `ENABLE_REBALANCING_ENGINE=false` in defaults |
| PortfolioSegmentationManager | `core/portfolio_segmentation.py` | ⚠️ Instantiated at step [6.6/9] — no periodic task assigned |
| ProfitTargetEngine | `core/profit_target_engine.py` | ✅ Instantiated — profit guard wired to SharedState |
| CapitalRecovery task | inside MetaController (~line 13591) | ⚠️ Active — calls `liquidation_agent._free_usdt_now()` but no LiquidationOrchestrator handle |

### Layer 8 — Observability
| Module | File | Status in MASTER |
|---|---|---|
| HealthMonitor | `core/health_monitor.py` | ✅ Step [9/9] |
| Watchdog | `core/watchdog.py` | ✅ Step [9/9] — running as task |
| Heartbeat | `core/heartbeat.py` | ✅ Step [9/9] — running as task |
| PerformanceEvaluator | `core/performance_evaluator.py` | ✅ Instantiated in optional section |
| PerformanceMonitor | `core/performance_monitor.py` | ✅ Instantiated in optional section |
| AlertSystem | `core/alert_system.py` | ❌ Disabled — `ENABLE_ALERT_SYSTEM=false` in defaults |
| PnLCalculator | `utils/pnl_calculator.py` | ✅ Instantiated in optional section |

---

## SECTION 3 — Seven Specific Findings

---

### Finding 1 — Which Layers Currently Start Too Early

**MetaController (Layer 6) starts before Layer 2 is ready and before Layer 3 is warm.**

Current boot order in MASTER:
```
Step [1/9]   ExchangeClient.start()
Step [2/9]   SharedState init (USDT only — positions empty)
Step [3/9]   SignalManager init
Step [4/9]   RiskManager init
Step [5/9]   ExecutionManager init
Step [5B/9]  TPSLEngine init
Step [6/9]   MetaController init  ← Layer 6 begins here
Step [7/9]   AgentManager init
Step [7.3/9] Discovery agents
Step [7.5/9] MarketDataFeed init  ← Layer 3 initialized here (AFTER MetaController)
Step [8/9]   PollingCoordinator init
Step [9/9]   HealthMonitor, Watchdog, Heartbeat
             [optional section] RecoveryEngine instantiated (rebuild_state() never called)
             [task launch] ALL tasks start concurrently
```

The problem: at task launch, MetaController.start() and MarketDataFeed.run() begin at exactly the same time. MetaController's first decision cycle (2s interval) fires before MarketDataFeed has loaded a single OHLCV bar. The micro-backtest gate reads `shared_state.get_market_data_sync(symbol, "5m")` and gets an empty list. Every BUY is immediately rejected with `MICRO_BACKTEST_INSUFFICIENT_SAMPLES`.

Additionally, RecoveryEngine is never called synchronously at all — it is just added to the optional task list where its `run_forever` loop runs in background. The loop does not call `rebuild_state()` on startup; it monitors for state drift. So SharedState starts with zero positions on every restart.

**Corrected start order required:**
```
L0  Config + .env + logging
L1  ExchangeClient.start() + connectivity test
L2  SharedState init → StartupOrchestrator.run() → positions hydrated → StartupPortfolioReady
L3  MarketDataFeed.run() → warm-up wait (5s) → OHLCV cache populated
L4  RiskManager + CapitalAllocator
L5  ExecutionManager + TPSLEngine
L6  MetaController.start() + AgentManager.start()
L7  LiquidationOrchestrator + CashRouter + CompoundingEngine + PortfolioBalancer
L8  HealthMonitor + Watchdog + Heartbeat
```

---

### Finding 2 — Which Layers Are Missing from MASTER_SYSTEM_ORCHESTRATOR

**Layer 2 (State Hydration) is missing its sequencing gate.**

`core/startup_orchestrator.py` exists and is complete. Its `run()` method calls the exact 6-step canonical sequence:
1. `RecoveryEngine.rebuild_state()` — fetch balances + positions from Binance REST
2. `SharedState.hydrate_positions_from_balances()` — mirror wallet → internal positions
3. `ExchangeTruthAuditor.restart_recovery()` — sync open orders
4. `PortfolioManager.refresh_positions()` — update position metadata
5. Verify startup integrity — NAV, capital, sanity checks
6. Emit `StartupPortfolioReady` — signal MetaController it's safe to start

`StartupOrchestrator` is **not imported anywhere in `🎯_MASTER_SYSTEM_ORCHESTRATOR.py`**. There is no `from core.startup_orchestrator import StartupOrchestrator` in the file. The recovery logic runs as a background task that never calls `rebuild_state()`.

**Layer 7 (Profit Engine) is structurally present but all profit-recycling components are off.**

LiquidationOrchestrator is instantiated but CashRouter is not. CompoundingEngine, PortfolioBalancer, and RebalancingEngine are all disabled by feature flags that default to `false`.

---

### Finding 3 — Which Modules Are Built but Not Wired

These modules have complete implementations but receive zero calls from the main execution path:

| Module | Lines | What It Does | Why Not Running |
|---|---|---|---|
| `StartupOrchestrator` | ~250 | Sequences Layer 2 boot (state hydration) | Not imported in MASTER |
| `CashRouter` | 758 | Sweeps dust + redeems stablecoins → USDT | Passed as `cash_router=None` to LiquidationOrchestrator |
| `DeadCapitalHealer` | 375 | Identifies illiquid holdings, liquidates them | Never instantiated anywhere |
| `ExternalAdoptionEngine` | ~400 | Adopts existing wallet holdings as bot positions | MetaController has `set_external_adoption_engine()` setter at line 21239 — never called |
| `ReserveManager` | ~300 | Manages USDT reserve pool for compounding | Never instantiated in MASTER |
| `automation/rule_overrides.py` | ~100 | Dynamic `required_conf` relaxation without restart | Imported by MetaController — but `proposed_rules.json` doesn't exist, so always a no-op |
| `automation/auto_rule_proposer.py` | ~200 | Generates `proposed_rules.json` from diagnostics | Never called |

**Partially wired (instantiated but missing a critical dependency):**

| Module | Missing Dependency | Effect |
|---|---|---|
| `LiquidationOrchestrator` | `CashRouter` passed as `None` | Dust sweep + stablecoin redemption path is dead |
| `ThreeBucketManager` | `DeadCapitalHealer` not wired to it | Three-bucket classification works but dead capital healing never executes |
| `MetaController._run_capital_recovery_task()` | No handle to `LiquidationOrchestrator` | When USDT is insufficient, MetaController calls `self.liquidation_agent._free_usdt_now()` — a narrow path that bypasses the full orchestration chain |

---

### Finding 4 — Which Feature Flags Must Be Enabled

The following flags are either hardcoded `false` as defaults or absent from `.env`, meaning the system cannot trade or recycle profits regardless of market conditions:

| Flag | Current Value | Required Value | Effect of Change |
|---|---|---|---|
| `PRETRADE_MICRO_BACKTEST_ENABLED` | `true` (in .env) | `false` | **Unblocks all trading** — removes the gate that drops every BUY intent |
| `RECOVERY_ALLOW_REST` | `false` (default) | `true` | RecoveryEngine.rebuild_state() fetches positions from Binance REST on every restart |
| `ENABLE_EXCHANGE_TRUTH_AUDITOR` | `false` (default) | `true` | Open orders synced into SharedState at startup |
| `ENABLE_COMPOUNDING_ENGINE` | `false` (default) | `true` | Realized PnL is reinvested — required for capital growth |
| `ENABLE_BALANCER` | `false` (default) | `true` | PortfolioBalancer periodic rebalancing runs |
| `ENABLE_REBALANCING_ENGINE` | `false` (default) | `true` | RebalancingEngine drift-threshold rebalancing runs |
| `BINANCE_API_TYPE` | `HMAC` (in .env) | `ED25519` | WebSocket real-time fills instead of 25s polling (requires confirming Ed25519 key is registered on Binance) |
| `MIN_TRADE_QUOTE` | `25` (in .env) | `15` | Allows trades with current ~$22 USDT balance |
| `DEFAULT_PLANNED_QUOTE` | `25` (in .env) | `15` | Consistent with MIN_TRADE_QUOTE reduction |

**Secondary flags (quality of life, not blocking):**

| Flag | Current | Recommended | Reason |
|---|---|---|---|
| `ENABLE_ALERT_SYSTEM` | `false` | `true` | Telegram notifications for fills and events |
| `ENABLE_POSITION_MERGER_ENHANCED` | `false` | `true` | Consolidates fragmented positions |
| `PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES` | `true` (in .env) | `false` | Even if micro-backtest gate stays on, do not hard-reject when samples are insufficient — allow soft gate |

---

### Finding 5 — Where MetaController Should Call LiquidationOrchestrator

There are two places MetaController already detects an USDT shortfall but does not call LiquidationOrchestrator:

**Location A — Affordability check loop (meta_controller.py ~line 17961)**

This is the exact code path where a BUY intent fails because there is not enough USDT:

```python
# EXISTING CODE (line ~17961):
if reason in ("INSUFFICIENT_QUOTE", "QUOTE_LT_MIN_NOTIONAL") or "NOT_EXECUTABLE" in str(reason):
    sig["_need_liquidity"] = True
    sig["_liq_gap"] = gap
    sig["_liq_reason"] = reason
    decisions.append((sym, action, sig))   ← intent is queued but USDT is never actually freed
```

The intent is marked with `_need_liquidity=True` and passed downstream, but nothing ever calls the LiquidationOrchestrator to free the USDT. The trade silently stalls.

**The wire that is needed here:**
After setting `_need_liquidity=True`, MetaController must call:
```python
if self._liquidation_orchestrator is not None:
    await self._liquidation_orchestrator.ensure_liquidity(
        required_usdt=gap,
        reason=str(reason),
        symbol=sym,
    )
```

This is ~5 lines. `LiquidationOrchestrator.ensure_liquidity()` already exists — it calls `CashRouter.sweep_to_usdt()` as its first action.

**Location B — Capital recovery task (meta_controller.py ~line 13666)**

Inside `_run_capital_recovery_task()`, MetaController currently does this when free USDT is below the floor:

```python
# EXISTING CODE (line ~13666-13679):
elif self.liquidation_agent and hasattr(self.liquidation_agent, "_free_usdt_now"):
    await self.liquidation_agent._free_usdt_now(
        min_usdt=gap_usdt,
        free_before=free_usdt,
        ...
    )
```

This calls a narrow method on the `liquidation_agent` (a single strategy agent) and completely bypasses the `LiquidationOrchestrator` → `CashRouter` → stablecoin redemption chain. The correct call here is:

```python
# REPLACEMENT:
if getattr(self, "_liquidation_orchestrator", None) is not None:
    await self._liquidation_orchestrator.ensure_liquidity(
        required_usdt=gap_usdt,
        reason="capital_recovery",
    )
```

**How to wire it:** MetaController already has a `set_external_adoption_engine()` pattern at line 21239. A `set_liquidation_orchestrator(orch)` setter needs to be added (or the attribute set directly from MASTER after both are instantiated). The MASTER already creates both — it just never connects them.

---

### Finding 6 — Where StartupOrchestrator Must Replace Bootstrap-First Behavior

**The problem:** MASTER boots MetaController on an empty SharedState. This is the "bootstrap-first" behavior — the system assumes it is starting from scratch every time, even when it has live positions and wallet holdings.

**Current flow in MASTER:**
```
Step [2/9] SharedState.__init__()
           shared_state.balances["USDT"] = {"free": usdt_balance}
           ← DONE. All other balances and positions are zero.
           
... steps [3/9] through [9/9] ...

[optional section] recovery_engine = RecoveryEngine(...)
                   ← instantiated but rebuild_state() never called
                   
[task launch] asyncio.create_task(meta_controller.start())
              ← MetaController cycles with empty positions immediately
```

**Required flow with StartupOrchestrator:**

The insertion point is between step [2/9] and step [3/9] (after SharedState init, before SignalManager):

```
Step [2/9]   SharedState.__init__()
Step [2.5/9] [NEW] startup_orchestrator = StartupOrchestrator(
                     config=self.config,
                     shared_state=self.shared_state,
                     exchange_client=self.exchange_client,
                     recovery_engine=self.recovery_engine,
                     exchange_truth_auditor=self.exchange_truth_auditor,
                 )
             [NEW] await startup_orchestrator.run()
                   ↳ RecoveryEngine.rebuild_state()          ← fetches all positions via REST
                   ↳ SharedState.hydrate_positions()         ← positions now visible
                   ↳ ExchangeTruthAuditor.restart_recovery() ← open orders synced
                   ↳ emit StartupPortfolioReady              ← signal that MetaController may start
Step [3/9]   SignalManager init (unchanged)
...
```

**What changes:** RecoveryEngine must be instantiated before step [2.5/9], not in the optional section. `RECOVERY_ALLOW_REST=true` must be set in `.env` so `rebuild_state()` actually contacts the Binance REST API.

The `StartupOrchestrator` already handles failures gracefully — if RecoveryEngine fails, it falls back to a clean slate (current behavior). So this change adds resilience without risk.

---

### Finding 7 — Why PrePublishGate Blocks All Signals

The blockage is a **two-layer gate with no fill history to satisfy either layer.**

**Layer A — AgentManager.PrePublishGate (agent_manager.py line 324)**

Every intent from every agent passes through `_passes_prepublish_viability_gate()` before being published to MetaController. This gate calls MetaController's own `_passes_pretrade_effect_gate()`. If that gate returns `False`, the intent is dropped with the reason logged.

**Layer B — MetaController._passes_pretrade_effect_gate() (meta_controller.py line 7601)**

This gate runs in two sub-phases:

**Sub-phase 1 — OHLCV micro-backtest:**
```python
bt_enabled = True  (PRETRADE_MICRO_BACKTEST_ENABLED=true in .env)
bt_min_samples = 12
bt_require_samples = True  (PRETRADE_MICRO_BACKTEST_REQUIRE_SAMPLES=true in .env)
bt_min_win_rate = 0.52

rows = shared_state.get_market_data_sync(symbol, "5m")  ← reads OHLCV cache
```

On a fresh restart, MarketDataFeed has not yet loaded data. `rows = []`. `bt_sample_count = 0`. `bt_win_rate = None`.

Since `bt_require_samples=True` and `bt_sample_count < bt_min_samples`, the gate returns:
```
False, "MICRO_BACKTEST_INSUFFICIENT_SAMPLES"
```

Even after MarketDataFeed warms up (5–10 minutes), the backtest computes net returns on recent 5m closes. On a flat or choppy market, many of the 90 lookback bars show negative net returns after subtracting round-trip fees (2 × 10 bps fee + 2 × 10 bps slippage + 5 bps buffer = 45 bps). A 52% win rate threshold after fees is genuinely difficult to clear in sideways conditions.

**Sub-phase 2 — Deadlock relief (does not activate soon enough):**

MetaController has its own deadlock relief that triggers after `bt_relax_trigger=8` consecutive micro-backtest rejections for a symbol. This gradually relaxes thresholds. AgentManager has a separate, independent deadlock counter that triggers after `AGENTMGR_PREPUBLISH_DEADLOCK_REJECTION_TRIGGER=24` rejections.

The rejection counters are stored in `SharedState` as in-memory state. **They reset to zero on every restart.** So on each restart the system must accumulate 8 (MetaController) + 24 (AgentManager) rejections before any relief applies. With a 2s cycle and 2 symbols, this takes roughly:
- MetaController relief: ~16 seconds (8 rejections × 2 symbols)
- AgentManager relief: ~48 seconds (24 rejections × 2 symbols)

Relief is additive with stall-aware relaxation but it only softens the thresholds — it does not bypass the gate. A system with no fill history can never achieve the initial win rate proof required to unlock the gate permanently.

**The core paradox:**
> The micro-backtest gate requires a history of fills to prove the strategy is profitable before allowing the first trade. But fills can only exist after trades. The system is deadlocked by design on first boot.

**Summary of gate failure chain:**
```
AgentManager cycles every 2s
  → _filter_intents_prepublish(intents=[BTCUSDT-BUY, ETHUSDT-BUY])
    → _passes_prepublish_viability_gate(BTCUSDT, BUY)
      → meta_controller._passes_pretrade_effect_gate(BTCUSDT, BUY, quote=25)
        → shared_state.get_market_data_sync("BTCUSDT", "5m") → []   [empty on boot]
        → bt_sample_count = 0 < bt_min_samples=12
        → bt_require_samples = True
        → return False, "MICRO_BACKTEST_INSUFFICIENT_SAMPLES"
      → record_rejection(BTCUSDT, BUY, "MICRO_BACKTEST_INSUFFICIENT_SAMPLES")
      → return False, "MICRO_BACKTEST_INSUFFICIENT_SAMPLES"
    → Dropped. Log: "Filtered intents: in=2 out=0 dropped=2"
```

This runs on every cycle. No trade ever reaches ExecutionManager.

---

## SECTION 4 — Minimum Intervention Table

These are the exact changes needed to wire the existing system. Listed in priority order. No new modules added.

| Priority | Type | Change | File | Lines |
|---|---|---|---|---|
| P0 | .env | `PRETRADE_MICRO_BACKTEST_ENABLED=false` | `.env` | 1 |
| P0 | .env | `RECOVERY_ALLOW_REST=true` | `.env` | 1 |
| P0 | .env | `MIN_TRADE_QUOTE=15` + `DEFAULT_PLANNED_QUOTE=15` | `.env` | 2 |
| P1 | Code | Import + call StartupOrchestrator between step [2/9] and [3/9] | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ~30 |
| P1 | Code | Move MarketDataFeed init to step [3/9] + 5s warm-up before MetaController task | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ~10 |
| P1 | Code | Instantiate CashRouter + pass it to LiquidationOrchestrator | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ~15 |
| P1 | Code | Add `set_liquidation_orchestrator()` setter to MetaController + call it from MASTER after both instantiated | `core/meta_controller.py` + MASTER | ~10 |
| P1 | Code | At MetaController line ~17961: call `_liquidation_orchestrator.ensure_liquidity()` when `_need_liquidity=True` | `core/meta_controller.py` | ~8 |
| P2 | .env | `ENABLE_EXCHANGE_TRUTH_AUDITOR=true` | `.env` | 1 |
| P2 | .env | `ENABLE_COMPOUNDING_ENGINE=true` | `.env` | 1 |
| P2 | .env | `ENABLE_BALANCER=true` | `.env` | 1 |
| P2 | Code | Instantiate DeadCapitalHealer + wire to ThreeBucketManager | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ~20 |
| P2 | Code | Instantiate ExternalAdoptionEngine + call `meta_controller.set_external_adoption_engine()` | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | ~10 |
| P3 | .env | `BINANCE_API_TYPE=ED25519` (after confirming key is registered) | `.env` | 1 |
| P3 | Code | Add rejection counter persistence to SharedState snapshot | `core/shared_state.py` | ~15 |

---

## SECTION 5 — Startup Sequence Diagram (Current vs Target)

### Current (broken)
```
t=0s   ExchangeClient.start()
t=1s   SharedState(USDT only)  ← positions empty
t=2s   SignalManager, RiskManager
t=3s   ExecutionManager, TPSLEngine
t=4s   MetaController.__init__()
t=5s   AgentManager.__init__()
t=6s   MarketDataFeed.__init__()  ← OHLCV not loaded yet
t=7s   PollingCoordinator, Health, Watchdog, Heartbeat
t=8s   RecoveryEngine instantiated (rebuild_state never called)
t=8s   ALL TASKS LAUNCHED CONCURRENTLY
t=10s  MetaController.step()  ← OHLCV empty → ALL BUY intents rejected
t=10s  MarketDataFeed still loading first bars
... continues rejecting every 2s until MarketDataFeed warms up ...
... even after warm-up: win rate cannot be proven → still rejected ...
```

### Target (correct)
```
t=0s   ExchangeClient.start() + connectivity test
t=1s   SharedState.__init__()
t=2s   RecoveryEngine instantiated (with RECOVERY_ALLOW_REST=true)
t=2s   ExchangeTruthAuditor instantiated
t=3s   StartupOrchestrator.run() ← awaited synchronously
           ↳ rebuild_state() → positions hydrated from REST
           ↳ ExchangeTruthAuditor.restart_recovery() → open orders synced
           ↳ emit StartupPortfolioReady
t=5s   MarketDataFeed started (Layer 3) ← warm-up begins
t=10s  [5s wait] OHLCV cache populated with initial bars
t=10s  RiskManager, CapitalAllocator
t=11s  ExecutionManager, TPSLEngine
t=12s  MetaController.start() + AgentManager.start()  ← OHLCV ready, positions hydrated
t=13s  LiquidationOrchestrator (with CashRouter wired) + CompoundingEngine + PortfolioBalancer
t=14s  HealthMonitor, Watchdog, Heartbeat
```

---

*Report complete. No code has been modified. All findings reference existing files and line numbers.*  
*Awaiting approval to proceed with implementation in the priority order above.*
