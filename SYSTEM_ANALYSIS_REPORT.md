# OctiVault Trader — Complete System Analysis (All 226 Scripts)
**Date:** 2026-04-26  
**Scope:** Every .py file read, all logs analyzed, full gap report and fix plan

---

## STEP 1 — Full Repository Map (All 226 Scripts Classified)

### Category A — Active Trading Core (runs every cycle)
These modules are instantiated and running in every live session:

| Module | Role | Status |
|---|---|---|
| `core/shared_state.py` | Authoritative memory — balances, positions, signals, PnL | ✅ Running |
| `core/meta_controller.py` | Brain — ~20k lines, decision loop every 2s | ✅ Running |
| `core/execution_manager.py` | Single order path → ExchangeClient | ✅ Running |
| `core/exchange_client.py` | Binance connection (HMAC+polling fallback) | ✅ Running |
| `core/risk_manager.py` | Advisory gate — daily loss, exposure, notional | ✅ Running |
| `core/agent_manager.py` | Agent lifecycle + PrePublishGate | ✅ Running (gate blocks all) |
| `core/market_data_feed.py` | OHLCV + WebSocket price feed | ✅ Running |
| `core/polling_coordinator.py` | Balance/order polling every 30s | ✅ Running |
| `core/signal_manager.py` | Signal cache and routing | ✅ Running |
| `core/tp_sl_engine.py` | TP/SL monitoring | ✅ Running (no positions to monitor) |
| `core/capital_allocator.py` | Capital allocation per symbol | ✅ Running |
| `core/symbol_manager.py` | Symbol lifecycle | ✅ Running |
| `core/universe_rotation_engine.py` | Symbol universe rotation | ✅ Running |
| `core/heartbeat.py` | Health ping every 30s | ✅ Running |
| `core/watchdog.py` | Component health monitor | ✅ Running |
| `core/health_monitor.py` | Health status | ✅ Running |
| `core/pnl_calculator.py` | PnL tracking | ✅ Running |
| `core/performance_evaluator.py` | KPI analytics | ✅ Running |

### Category B — Embedded Inside MetaController (active logic, instantiated in meta_controller.py)

| Module | Role | Status |
|---|---|---|
| `core/signal_batcher.py` | Batch signals to reduce friction | ✅ Active |
| `core/signal_fusion.py` | Multi-agent edge aggregation (alpha amplifier) | ✅ Active async task |
| `core/capital_governor.py` | Position limits by NAV bracket | ✅ Active (MICRO bracket) |
| `core/nav_regime.py` | NAV regime detection (MICRO_SNIPER mode) | ✅ Active |
| `core/rotation_authority.py` | Stagnation/concentration/liquidity exit authority | ✅ Active |
| `core/portfolio_authority.py` | Velocity/rebalance/profit-recycling exits | ✅ Active |
| `core/capital_velocity_optimizer.py` | Proactive capital allocation | ✅ Active |
| `core/opportunity_ranker.py` | Capital-first opportunity scoring | ✅ Active (lazy init) |
| `core/lifecycle_manager.py` | Symbol lifecycle state machine | ✅ Active |
| `core/bootstrap_manager.py` | Bootstrap mode / dust bypass | ✅ Active |
| `core/arbitration_engine.py` | Multi-layer gate evaluation | ✅ Active |
| `core/mode_manager.py` | Mode switching (NORMAL/FOCUS/RECOVERY/BOOTSTRAP) | ✅ Active |
| `core/focus_mode.py` | Focus mode subsystem | ✅ Active |
| `core/state_manager.py` | Liveness, cooldowns, health transitions | ✅ Active |
| `core/intent_manager.py` | Intent sink and drain | ✅ Active |
| `core/policy_manager.py` | Policy evaluation and decision logic | ✅ Active |
| `core/adaptive_capital_engine.py` | Dynamic sizing envelope per symbol | ✅ Active |
| `core/scaling.py` | Scale-in / compounding logic | ✅ Active |
| `core/action_router.py` | Priority-based routing to ExecutionManager | ⚠️ Wired but only via `set_external_adoption_engine` path |

### Category C — Disabled by Feature Flag (built, tested, awaiting env var)

| Module | Enable Flag | Current Value | Effect |
|---|---|---|---|
| `core/compounding_engine.py` | `ENABLE_COMPOUNDING_ENGINE` | `false` (not in .env) | No PnL reinvestment |
| `core/portfolio_balancer.py` | `ENABLE_BALANCER` | `false` (not in .env) | No rebalancing |
| `core/exchange_truth_auditor.py` | `ENABLE_EXCHANGE_TRUTH_AUDITOR` | `false` (not in .env) | No order reconciliation at startup |
| `core/position_merger_enhanced.py` | `ENABLE_POSITION_MERGER_ENHANCED` | `false` | No position consolidation |
| `core/rebalancing_engine.py` | `ENABLE_REBALANCING_ENGINE` | `false` | No engine-driven rebalancing |
| `core/volatility_regime.py` | `ENABLE_VOLATILITY_REGIME` | `false` | No volatility regime detection |
| `core/alert_system.py` | `ENABLE_ALERT_SYSTEM` | `false` | No Telegram/external alerts |
| `core/liquidation_orchestrator.py` | `LIQ_ORCH_ENABLE=true` ✅ | Initialized | MetaController doesn't call it |

### Category D — Built but Never Connected to Main Path

These modules exist, are well-written, contain important logic, but are **never instantiated or called** by `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` or `MetaController`:

| Module | Lines | Purpose | Connection Status |
|---|---|---|---|
| `core/cash_router.py` | 758 | Converts non-core balances (stables, dust) to USDT | ⚠️ Referenced by `LiquidationOrchestrator` internally, but `LiquidationOrchestrator` itself is never called by MetaController |
| `core/dead_capital_healer.py` | 375 | Liquidates dust/stale/orphaned positions → operating cash | ❌ Never instantiated in live path |
| `core/three_bucket_manager.py` | 281 | Orchestrates bucket classification + healing | ❌ Never instantiated |
| `core/bucket_classifier.py` | 361 | Classifies positions into Operating Cash / Productive / Dead Capital | ❌ Never instantiated |
| `core/portfolio_buckets.py` | 317 | Data structures for three-bucket model | ❌ Only used internally |
| `core/reserve_manager.py` | 542 | Enforces minimum cash reserve (10–35% of portfolio) | ❌ Never instantiated in live path |
| `core/external_adoption_engine.py` | 499 | Handles pre-existing wallet assets (adopt/liquidate/hedge) | ❌ Setter exists in MetaController but AppContext (not MASTER) would wire it |
| `core/layer_orchestrator.py` | — | Three-layer architecture orchestrator | ❌ Not wired |
| `core/layer_contracts.py` | — | Formal layer boundary contracts | ❌ Not wired |
| `core/startup_orchestrator.py` | — | Sequencing gate: RecoveryEngine → ExchangeTruthAuditor → PortfolioManager | ❌ Not called by MASTER_SYSTEM_ORCHESTRATOR |
| `config/EV_ALIGNMENT_CONFIG.py` | — | Canonical EV formula to align UURE and ExecutionManager | ❌ Never imported |
| `core/portfolio_segmentation.py` | 488 | Segments portfolio into productive/dead/cash | ⚠️ Initialized optionally in MASTER, no active loop |
| `core/correlation_manager.py` | — | Correlation-based diversification | ❌ Not wired |
| `core/chaos_monkey.py` | — | Fault injection for testing | ❌ Not enabled |
| `core/replay_engine.py` | — | Event replay for auditing | ❌ Not active |
| `automation/auto_rule_proposer.py` | — | Proposes required_conf relaxations from diagnostics | ❌ Never run automatically |
| `core/app_context.py` | — | Alternative initialization path (Phase 3–9 style) | ❌ Bypassed; MASTER_SYSTEM_ORCHESTRATOR does its own init |

### Category E — Broken / Dead References

| File | Problem |
|---|---|
| `RUN_AUTONOMOUS_LIVE.py` | Calls `🚀_LIVE_ED25519_TRADING.py` — **file does not exist** |
| `AUTONOMOUS_SYSTEM_STARTUP.py` | Calls `signal_manager.generate_signals()` — **method does not exist** |
| `PRODUCTION_STARTUP.py` | Reads from `state/operational_state.json` — **directory doesn't exist** (state is in `data/`) |

### Category F — Agents

| Agent | Type | Status |
|---|---|---|
| `agents/trend_hunter.py` | Strategy (ML) | ✅ Running — signals dropped at PrePublishGate |
| `agents/swing_trade_hunter.py` | Strategy | ✅ Running — signals dropped at PrePublishGate |
| `agents/dip_sniper.py` | Strategy | ✅ Running — signals dropped at PrePublishGate |
| `agents/ml_forecaster.py` | Strategy (ML+TF) | ✅ Running — signals dropped at PrePublishGate |
| `agents/symbol_screener.py` | Discovery | ✅ Running — proposing 0 new symbols |
| `agents/ipo_chaser.py` | Discovery | ✅ Running |
| `agents/wallet_scanner_agent.py` | Discovery | ✅ Running — finds 13 wallet assets, none added |
| `agents/liquidation_agent.py` | Infrastructure | ✅ Registered — never triggered |
| `agents/edge_calculator.py` | Utility | ✅ Used by agents for edge scoring |

### Category G — Diagnostic / Monitoring Scripts (not part of trading loop, 40+ scripts)

All `MONITOR_*.py`, `REALTIME_*.py`, `CONTINUOUS_*.py`, `SESSION_*.py`, `PERIODIC_*.py`, `PROFIT_*.py`, `balance_dashboard.py`, `diagnostic_signal_flow.py`, `extract_rejections.py`, `GATING_WATCHDOG.py`, `FAST_DIAGNOSTICS.py`, `SIGNAL_FLOW_DIAGNOSTIC.py` — read-only observers. None place orders. Kept for operational visibility.

### Category H — Development / Verification Scripts

`verify_*.py`, `phase4_*.py`, `deploy_phase2_production.py`, `component_validator.py`, `TEST_BOOTSTRAP.py`, `TEST_FALLBACK.py`, `FORCE_SIGNALS_INJECTOR.py`, `tests/` directory — diagnostics, validation, and testing tools.

### Category I — Support / Infrastructure

`core/stubs.py`, `core/error_types.py`, `core/error_handler.py`, `core/contracts.py`, `core/core_utils.py`, `core/logger_utils.py`, `core/exit_utils.py`, `utils/indicators.py`, `utils/ta_indicators.py`, `utils/ohlcv_cache.py`, `utils/shared_state_tools.py`, `utils/pnl_calculator.py`, `utils/symbol_filter_pipeline.py`, `utils/hyg_guards.py`, `utils/volatility_adjusted_confidence.py` — support libraries. All healthy and in use.

---

## STEP 2 — Startup Flow (Authoritative)

```
🎯_MASTER_SYSTEM_ORCHESTRATOR.py
│
├── Core init:
│     Config → ExchangeClient (HMAC, WS auth fails → polling fallback)
│     SharedState (empty) → SignalManager → ExecutionManager → RiskManager
│
├── RecoveryEngine.run():
│     Reads empty SharedState memory → RECOVERY_ALLOW_REST=False → skips REST
│     Result: 0 positions recovered. ← BROKEN
│
├── MarketDataFeed → WebSocket mini-ticker + OHLCV backfill (BTCUSDT, ETHUSDT)
│
├── MetaController.start():
│     Instantiates embedded: CapitalGovernor, NAVRegimeManager, RotationAuthority,
│     PortfolioAuthority, SignalFusion (async), SignalBatcher, OpportunityRanker,
│     CapitalVelocityOptimizer, LifecycleManager, BootstrapManager...
│     Decision loop starts (every 2s)
│
├── AgentManager.start():
│     Registers 7 agents → discovery agents launch → strategy agents tick-driven
│     PrePublishGate: BLOCKS all BUY intents (micro_backtest gate) ← PRIMARY BLOCKER
│
├── TPSLEngine → waits for gates → opens → monitors 0 positions
├── CapitalAllocator → plans capital per symbol
├── SymbolManager → manages BTCUSDT, ETHUSDT
├── UniverseRotationEngine → scores and rotates 2 symbols every 5 min
├── PnLCalculator → running
├── PerformanceEvaluator → running
├── PortfolioBalancer → ENABLE_BALANCER=false → exits immediately ← DISABLED
├── LiquidationOrchestrator → initialized, sits idle (MetaController doesn't call it)
├── CompoundingEngine → ENABLE_COMPOUNDING_ENGINE=false → never starts ← DISABLED
│
└── All 17 tasks running. Net result: system runs perfectly, trades 0.
```

---

## STEP 3 — Runtime Log Analysis

### Boot
System boots cleanly every time. All 17 tasks start. No import errors. No crashes.

### WebSocket Auth Failure
```
[EC:UserDataWS:v3] Policy violation error (1008) — HMAC key rejected by WS API v3
[EC:ListenKey] Got 410 Gone — listenKey unavailable
→ Falls back to Tier 3 polling (25s interval)
```
Balance/order updates are 25 seconds stale at all times.  
**Root cause:** `.env` has `BINANCE_API_TYPE=HMAC`. The Ed25519 key is present in `.env` as `BINANCE_API_SECRET_ED25519` but is unused.

### Signal Pipeline — Complete Blockage (Current)
```
TrendHunter: BUY ETHUSDT (conf=0.80, exp_move=2.06%, edge=0.369) → Published TradeIntent
AgentManager:PrePublishGate → DROPPED: micro_backtest_win_rate_below_threshold
                                       (ALL 3 intents, every cycle, zero reach MetaController)
```
The micro_backtest gate checks 12+ historical fills for ≥52% win rate. The system has essentially zero fill history. Gate is permanently locked.

The deadlock relief mechanism exists (relaxes gate after 24 rejections per symbol) but requires the per-symbol rejection counter to accumulate. That counter appears to reset on restart.

### MetaController Behavior When Signals Arrive (April 16 session)
In one earlier session when signals bypassed the gate, a trade executed but was blocked at ExecutionManager's EV hard gate:
```
[EM:EV_HARD_GATE] Blocked BUY MATICUSDT:
  expected_move=0.2500% (raw=0.0000% key=emergency_fallback)
  <= required=0.3610%
```
`raw=0.0000% key=emergency_fallback` means the `_expected_move_pct` from the agent signal was not correctly propagated to ExecutionManager. The agents do calculate it (ETHUSDT: 2.06%) but the field was lost in transit through MetaController. The current AgentManager PrePublishGate code now correctly passes `_expected_move_pct` through — this specific issue may be fixed but can't be confirmed since no signals currently reach MetaController.

### One Trade That Executed
```
Cycle 1: ETHUSDT $75.49 → $46.06 | P&L: $-29.43 (-38.99%) | 35s
```
No TP/SL triggered an exit. System restarted, forgot the position. Capital fell from $75 to $46.

### NAV Regime
With NAV ~$22–52, the system is in **MICRO_SNIPER mode** (NAV < $1000):
- max_open_positions: 1–2 (configurable, CAPITAL_MICRO_MAX_CONCURRENT_POSITIONS=2)
- max_active_symbols: 3
- min_expected_move: 1.0% (hard gate)
- min_confidence: 0.50 (lowered from 0.70 in code)
- max_trades_day: 3

This is appropriate behavior, not a bug. But the 1.0% expected_move requirement is an additional gate that signals must pass even after the micro_backtest gate is cleared.

### Recovery at Every Restart
```
RecoveryEngine → reads empty SharedState → RECOVERY_ALLOW_REST=False → skips REST
RestartPositionClassifier → Found 0 symbols to classify → registered 0 positions
```
Every restart is a fresh start. No positions, no avg_prices, no cooldowns, no PnL history restored.

---

## STEP 4 — State Recovery Audit

| State Component | Persisted To File? | Recovered On Restart? | Verdict |
|---|---|---|---|
| USDT balance | ❌ No | ❌ Not fetched (REST disabled) | 🔴 BROKEN |
| Open positions | ❌ No | ❌ Not rebuilt | 🔴 BROKEN |
| Average entry prices | ❌ No | ❌ Not restored | 🔴 BROKEN |
| Open orders | ❌ No | ❌ Not synced (ExchangeTruthAuditor disabled) | 🔴 BROKEN |
| Reservations | In-memory only | ❌ Resets on restart | 🔴 BROKEN |
| Cooldowns | In-memory only | ❌ Resets on restart | 🔴 BROKEN |
| Realized PnL | `data/event_store.db` (36KB, minimal) | ❌ RecoveryEngine doesn't load it | 🔴 BROKEN |
| Accepted symbols | `.env` SYMBOLS= | ✅ Hardcoded | ✅ OK |
| Micro_backtest rejection counter | In-memory only | ❌ Resets on restart (deadlock relief never triggers) | 🔴 BROKEN |
| Trade journal | `logs/trade_journal_*.jsonl` | ❌ Not loaded into SharedState | ⚠️ PARTIAL |

**State files that exist but are not used by RecoveryEngine:**
- `data/event_store.db` — event log, not loaded
- `data/operational_state.json` — session state, RecoveryEngine doesn't read it
- `state/` directory — PRODUCTION_STARTUP.py checks here but MASTER uses `data/`

---

## STEP 5 — Architecture Gap Report (Complete, All 226 Scripts)

### GAP #1 — P0: micro_backtest gate permanently blocks all trades
**The primary reason the system has not traded since April 16.**

`.env` has `PRETRADE_MICRO_BACKTEST_ENABLED=true`, `MIN_WIN_RATE=0.52`, `REQUIRE_SAMPLES=true`, `MIN_SAMPLES=12`. With no fill history, every signal is rejected. The deadlock relief (relaxes after 24 rejections) resets on restart, so it never accumulates enough to trigger.

### GAP #2 — P0: Recovery does nothing — fresh start on every restart
`RECOVERY_ALLOW_REST=False` (default) + empty SharedState at boot = RecoveryEngine reads nothing. The ETH position opened at $75 was forgotten on restart. TP/SL never managed it. Capital effectively missing.

### GAP #3 — P1: USDT below minimum trade size
Wallet USDT ≈ $22. `MIN_TRADE_QUOTE=25`. Every BUY is blocked by affordability check. The `LiquidationOrchestrator` (`LIQ_ORCH_ENABLE=true`) and `CashRouter` (758 lines) both exist to solve this, but MetaController does not call them when USDT is insufficient.

### GAP #4 — P1: CompoundingEngine disabled
`ENABLE_COMPOUNDING_ENGINE` not in `.env` → defaults to `false`. No realized PnL is ever reinvested.

### GAP #5 — P1: PortfolioBalancer disabled
`ENABLE_BALANCER` not in `.env` → `enabled` property returns `False` → exits immediately on start. No rebalancing.

### GAP #6 — P1: ExchangeTruthAuditor disabled — open orders not reconciled at startup
`ENABLE_EXCHANGE_TRUTH_AUDITOR=false` → open orders from Binance are never synced into SharedState at boot. Even if RecoveryEngine is fixed, open orders from before the restart would still be missed.

### GAP #7 — P1: StartupOrchestrator not called by MASTER_SYSTEM_ORCHESTRATOR
`core/startup_orchestrator.py` implements the correct canonical boot sequence (RecoveryEngine → ExchangeTruthAuditor → PortfolioManager → capital verification → emit StartupPortfolioReady). It is well-written and ready to use. The MASTER_SYSTEM_ORCHESTRATOR runs components independently instead, missing this coordination layer.

### GAP #8 — P2: 21 modules built but never connected to main path
See Category D above. Key lost capabilities:
- **CashRouter** (758 lines): would sweep stable/dust balances to USDT for liquidity. Referenced by LiquidationOrchestrator but LiquidationOrchestrator is never called.
- **DeadCapitalHealer** (375 lines): would liquidate orphaned/dust positions. Never instantiated.
- **ThreeBucketManager** (281 lines): entire portfolio structure framework. Never called.
- **ReserveManager** (542 lines): would enforce minimum cash reserve. Never instantiated.
- **ExternalAdoptionEngine** (499 lines): would classify and manage pre-existing wallet assets. Setter exists in MetaController but AppContext is bypassed.
- **config/EV_ALIGNMENT_CONFIG.py**: canonical EV formula to prevent UURE admitting symbols that EM then rejects. Never imported.

### GAP #9 — P2: WebSocket uses HMAC instead of Ed25519
`BINANCE_API_TYPE=HMAC` → WS API v3 rejects with 1008 → falls back to 25-second polling. The Ed25519 key is already in `.env` as `BINANCE_API_SECRET_ED25519`.

### GAP #10 — P2: Only 2 symbols in universe
`SYMBOLS=BTCUSDT,ETHUSDT` hardcoded. Discovery agents propose 0 new symbols (symbol proposals not accepted). 2 symbols = very low trade frequency.

### GAP #11 — P2: MetaController → LiquidationOrchestrator call missing
When MetaController detects INSUFFICIENT_QUOTE (not enough USDT to buy), it should request `LiquidationOrchestrator.ensure_free_usdt()` → which calls `CashRouter.ensure_free_usdt()` → which sells smallest/most stable non-core balance → frees USDT → retries once. This entire chain exists in the code but is never triggered from MetaController.

### GAP #12 — P2: micro_backtest rejection counter resets on restart
The deadlock relief mechanism in `AgentManager._passes_prepublish_viability_gate()` requires 24+ accumulated rejections to kick in. It calls `shared_state.get_rejection_count()` which is in-memory only. Every restart resets to 0, so the system never accumulates enough to trigger relief.

### GAP #13 — P3: automation/rule_overrides.py is wired but inactive
MetaController imports `get_required_conf_override` from `automation.rule_overrides`. This dynamically relaxes required_conf without restart. But `automation/proposed_rules.json` doesn't exist, so it's a no-op. Running `automation/auto_rule_proposer.py` would create it from diagnostics.

### GAP #14 — P3: RUN_AUTONOMOUS_LIVE.py calls a non-existent file
References `🚀_LIVE_ED25519_TRADING.py` which does not exist. Crashes immediately if launched.

### GAP #15 — P3: Async consensus check bug in MetaController
`Failed to check consensus for MATICUSDT: object bool can't be used in 'await' expression`  
A method returning `bool` is being awaited. Minor noise, not a blocker.

### GAP #16 — P3: SignalFusion output unclear
`SignalFusion` runs as an independent async task (composite_edge mode) and aggregates agent edge scores. It's well-written. Whether its fused signal is actually consumed by MetaController's primary decision path needs verification.

---

## STEP 6 — Minimal Fix Plan

**Principle:** No rewrites. Surgical `.env` changes first. Then targeted code patches.

### Fix 1 — Unblock all trading (P0) — 1 line in `.env`
```
PRETRADE_MICRO_BACKTEST_ENABLED=false
```
This single change allows every agent signal to pass the PrePublishGate and reach MetaController. Re-enable after 50+ fills have accumulated.

### Fix 2 — Fix recovery (P0) — 1 line in `.env`
```
RECOVERY_ALLOW_REST=true
```
Allows RecoveryEngine to use Binance REST at startup to fetch real balances, open orders, and positions. Populates SharedState correctly instead of starting empty.

### Fix 3 — Enable ExchangeTruthAuditor (P1) — 1 line in `.env`
```
ENABLE_EXCHANGE_TRUTH_AUDITOR=true
```
Syncs open orders from Binance into SharedState at startup. Works with Fix 2 for complete recovery.

### Fix 4 — Fix USDT insufficiency — lower minimums (P1) — `.env` changes
With ~$22 available USDT, current `MIN_TRADE_QUOTE=25` blocks everything:
```
MIN_TRADE_QUOTE=15
DEFAULT_PLANNED_QUOTE=15
CAPITAL_TARGET_FREE_USDT=15
MIN_ENTRY_USDT=15
TRADE_AMOUNT_USDT=15
TIER_B_MAX_QUOTE=30
META_MICRO_SIZE_USDT=15
ADAPTIVE_MIN_TRADE_QUOTE=14.0
EMIT_BUY_QUOTE=15
```

### Fix 5 — Wire MetaController → LiquidationOrchestrator (P1) — ~20 lines of code
In `core/meta_controller.py`, in the section where affordability check fails with INSUFFICIENT_QUOTE, add a call to `self.liquidation_orchestrator.ensure_free_usdt()` if the orchestrator is available, then retry once. This wires the existing `LiquidationOrchestrator → CashRouter` chain into the actual decision loop.

### Fix 6 — Enable CompoundingEngine (P1) — 1 line in `.env`
```
ENABLE_COMPOUNDING_ENGINE=true
```

### Fix 7 — Enable PortfolioBalancer (P1) — 1 line in `.env`
```
ENABLE_BALANCER=true
```

### Fix 8 — Switch to Ed25519 for real-time WS (P2) — 1 line in `.env`
```
BINANCE_API_TYPE=ED25519
```
The Ed25519 key is already in `.env` as `BINANCE_API_SECRET_ED25519`. Verify it is activated on your Binance account before switching.

### Fix 9 — Expand symbol universe (P2) — 1 line in `.env`
```
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,ADAUSDT,XRPUSDT
```

### Fix 10 — Call StartupOrchestrator from MASTER_SYSTEM_ORCHESTRATOR (P2) — ~30 lines of code
Replace the independent `RecoveryEngine.run()` task with a `StartupOrchestrator.execute_startup_sequence()` call before launching MetaController. This uses the correct sequencing: RecoveryEngine → ExchangeTruthAuditor → PortfolioManager → capital verification → emit StartupPortfolioReady.

### Fix 11 — Persist micro_backtest rejection counter (P2) — ~15 lines of code
In `core/shared_state.py`, persist the per-symbol rejection counter to `data/rejection_counters.json` on update, and load it on startup. This allows the deadlock relief mechanism to accumulate across restarts and eventually trigger even with the gate enabled.

### Fix 12 — Wire DeadCapitalHealer into TP/SL post-exit or periodic loop (P3) — ~20 lines
After every TP/SL exit, or on a 5-minute timer, call `ThreeBucketPortfolioManager.plan_healing_cycle()` and execute the healing plan. This converts orphaned dust positions back to USDT automatically.

### Fix 13 — Import and apply EV_ALIGNMENT_CONFIG (P3) — ~5 lines
In `core/config.py`, import `CANONICAL_EV_CONFIG` from `config/EV_ALIGNMENT_CONFIG.py` and apply it. This aligns the EV threshold used by UURE when admitting symbols with the threshold used by ExecutionManager when accepting trades. Eliminates the case where UURE says "good edge" but EM says "insufficient edge".

### Fix 14 — Fix async consensus bug (P3) — 1–3 lines of code
In `core/meta_controller.py`, find the consensus check call that is being awaited but returns `bool`. Remove the `await`.

---

## STEP 7 — Implementation Order

| # | Fix | Type | Risk | Change Size |
|---|---|---|---|---|
| 1 | `PRETRADE_MICRO_BACKTEST_ENABLED=false` | `.env` | Low | 1 line |
| 2 | `RECOVERY_ALLOW_REST=true` | `.env` | Low | 1 line |
| 3 | `ENABLE_EXCHANGE_TRUTH_AUDITOR=true` | `.env` | Low | 1 line |
| 4 | Lower trade minimums to $15 | `.env` | Low | 8 lines |
| 5 | `ENABLE_COMPOUNDING_ENGINE=true` | `.env` | Low | 1 line |
| 6 | `ENABLE_BALANCER=true` | `.env` | Low | 1 line |
| 7 | `BINANCE_API_TYPE=ED25519` | `.env` | Medium | 1 line |
| 8 | Expand SYMBOLS | `.env` | Low | 1 line |
| 9 | Wire MetaController → LiquidationOrchestrator | Code | Medium | ~20 lines |
| 10 | Call StartupOrchestrator from MASTER | Code | Medium | ~30 lines |
| 11 | Persist rejection counter across restarts | Code | Low | ~15 lines |
| 12 | Wire DeadCapitalHealer into periodic loop | Code | Medium | ~20 lines |
| 13 | Apply EV_ALIGNMENT_CONFIG | Code | Low | ~5 lines |
| 14 | Fix async consensus check bug | Code | Low | 1–3 lines |

**Fixes 1–8 are pure `.env` changes.** Apply all of them, restart once, verify behavior. Then move to code changes.

---

## Expected System Behavior After All Fixes

```
Restart:
  StartupOrchestrator →
    RecoveryEngine (REST) → fetches live Binance balances + positions → SharedState populated
    ExchangeTruthAuditor → syncs open orders → clean state
    reject_counter restored from data/rejection_counters.json
    StaleReservation cleanup → emit StartupPortfolioReady

Trading loop (every 2s):
  Agents → TradeIntents (conf=0.80, exp_move=2.06%, _expected_move_pct set) →
  AgentManager (micro_backtest gate OFF) → signals pass →
  MetaController receives intents → dedup → cooldown check → affordability →
    If USDT < MIN_TRADE_QUOTE:
      → LiquidationOrchestrator.ensure_free_usdt() → CashRouter frees dust/stables → retry once
  RiskManager → exposure check → approve →
  NAV Regime MICRO_SNIPER: min_expected_move=1.0% check (2.06% passes) →
  CapitalGovernor: position limit check (max 2 concurrent) →
  ExecutionManager → EV gate (exp_move 2.06% > required ~0.36%) → order placed →
  ExchangeClient → fill → SharedState updated (balance, position, avg_price) →
  TPSLEngine monitors position → exits at TP or SL →
  post_exit_bookkeeping → cooldown applied → PnL credited →
  CompoundingEngine reinvests realized PnL →
  PortfolioBalancer rebalances every 5 min →
  DeadCapitalHealer converts dust → USDT →
  loop repeats
```

---

## Questions for Mahmoud Before Code Changes

1. **micro_backtest gate**: Disable entirely (`PRETRADE_MICRO_BACKTEST_ENABLED=false`) or keep but with lower rejection trigger (`AGENTMGR_PREPUBLISH_DEADLOCK_REJECTION_TRIGGER=3`)? Disabling is cleanest short-term.

2. **Ed25519 switch**: Is the Ed25519 key in `.env` (`BINANCE_API_SECRET_ED25519`) actively registered on your Binance account? If yes, switching `BINANCE_API_TYPE=ED25519` gives real-time fills. If not sure, skip this for now.

3. **USDT level**: With ~$22 available USDT, even with `MIN_TRADE_QUOTE=15`, trades will be small ($15). Do you want to first liquidate some of the smaller crypto holdings (BNB, LINK, ADA, etc.) through the exchange UI to bring USDT to $50+ before restarting the bot?

4. **Symbol universe**: Which additional symbols do you want the bot to trade beyond BTCUSDT/ETHUSDT?

5. **ExternalAdoptionEngine**: Your wallet holds BTC, ETH, BNB, LINK, XRP, ADA, DOGE, SOL, PAXG, AVAX, PEPE, CHIP. Do you want the bot to adopt and manage these as its own positions (assign TP/SL to them), or leave them untouched as manual holdings?
