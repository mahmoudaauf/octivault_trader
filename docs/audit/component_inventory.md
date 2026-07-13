# Component Inventory — core_engine/native/

Evidence-based. "Instantiated" requires a confirmed constructor call site (file:line or
named caller), not just an import. "Status" is assigned only when runtime or wiring
evidence supports it — imports alone never justify "Active."

## Tier A — Execution / Risk Path (full depth)

These are the components on the hot path from signal → decision → order → fill → TP/SL.

| Component | Path | Instantiated | Instantiated by | Started/Triggered | Background loop | Tests | Status |
|---|---|---|---|---|---|---|---|
| `NativeDecisionEngine` | decisions.py | Yes — `bootstrap.py:723` | `build_components()` | Called synchronously per cycle: `orchestrator.py:598` `self._decision_engine.decide(signals, portfolio, balance_usdt)` | No | `test_nav_protection.py`, `test_native_l4.py`, `test_portfolio_recovery_mode.py`, `test_native_bootstrap.py` | **Active** |
| `NativeExecutor` | executor.py | Yes — `bootstrap.py:748`; `_tp_sl_engine` injected post-hoc at `bootstrap.py:878` | `build_components()` | Called synchronously per cycle: `orchestrator.py:603` `await self._executor.execute(decisions)` | No | `test_native_l5.py`, `test_race_conditions_and_growth.py` | **Active**. Deliberately duplicates fill/position-registration logic (comments at lines 497, 690) to compensate for `fill_tracker` being `None` in default polling mode — intentional, not dead code |
| `NativeCapitalAllocator` | capital_allocator.py | Yes — `bootstrap.py:818` | `build_components()` | Called synchronously via `allocate_for_buy(symbol)` per BUY decision | No | `test_race_conditions_and_growth.py`, `test_daily_compounding.py` | **Active**. Contains one TODO: `_compute_volatility_pct` (line ~400) returns a hardcoded 0.008 placeholder, real rolling-volatility-from-klines not implemented |
| `AdaptiveCapitalEngine` (`NativeAdaptiveCapitalEngine`) | adaptive_capital_engine.py | Conditional — `bootstrap.py:799`, gated by `adaptive_capital_engine_enabled` (default `True`) | `build_components()` | Invoked synchronously inside `capital_allocator.allocate_for_buy()` (`capital_allocator.py:264`), only on the NAV≥$100 branch and only if enabled | No | `test_objective_quality_guards.py` | **Active** (default config) |
| `DailyCompoundingPolicy` / `DailyCompoundingState` | daily_compounding.py (untracked) | Yes, but NOT in bootstrap.py directly — one level down: `capital_allocator.py:22` imports it, `capital_allocator.py:83-86` constructs `DailyCompoundingPolicy(enabled=..., state_path=...)`, which constructs `DailyCompoundingState()` internally | `NativeCapitalAllocator.__init__` | `sizing_nav()` called every `allocate_for_buy()` (`capital_allocator.py:151`) | No | `test_daily_compounding.py` | **Active**. Fully wired despite not appearing in bootstrap.py's own instantiation list — the earlier exploration concern about this is resolved |
| `NativeFillTracker` | fill_tracker.py | **Conditional, off by default** — legacy branch `bootstrap.py:529` or fallback `bootstrap.py:638`, both gated on `polling_enabled=False`. Default `polling_enabled=True` (`bootstrap.py:117,227`) means `fill_tracker` stays `None` | `build_components()` | `.start()` only called by `orchestrator.py:161-162` when `self._fill_tracker is not None and self._polling_coordinator is None` — under default config, `NativePollingCoordinator` substitutes | Yes, when active (poll loop, `fill_tracker.py:81/97`, min 1.0s interval) | `test_native_fill_tracker.py` | **Idle by default config** (intentional design, not broken — `NativePollingCoordinator` covers fill detection instead). One TODO: `hold_sec` hardcoded to `0.0` (line 338), real entry-timestamp tracking not implemented. Side effect: `fill_tracker.set_tp_sl_engine(...)` injection at `bootstrap.py:876-877` is also skipped in default mode |
| `NativeTPSLEngine` | tp_sl_engine.py | Yes — `bootstrap.py:870`; injected into executor at `bootstrap.py:878` | `build_components()` | `.start()`/`.stop()` via orchestrator lifecycle; `.check_triggers()` per position per cycle (`orchestrator.py:531`); `.recalculate_aged_positions()` periodically (`orchestrator.py:614`, every 300s) | Yes (via orchestrator's periodic recovery phase) | `test_objective_quality_guards.py`, `test_race_conditions_and_growth.py`, `test_native_tpsl_engine.py` | **Active** |
| `SymbolRotator` | symbol_rotator.py | Yes, non-fatal — `bootstrap.py:563`, wrapped in try/except (logs warning on failure, line 572) | `build_components()` | `orchestrator.py:379` and `:415` call `.maybe_rotate()` per cycle, only if not `None` | No (called per cycle, not its own loop) | `test_symbol_rotator_quality.py` | **Active** |
| `NativeArbitrationEngine` | arbitration_engine.py | Yes — `bootstrap.py:841`, given `shared_state`, `decision_engine`, `signal_fusion`, `mode_manager`, `ml_forecaster` | `build_components()` | **CORRECTED (Phase 4 follow-up):** called via `core_engine/implementations.py::DecisionEngineImpl.make_buy_decision`/`make_sell_decision` → `DecisionEngineImpl.evaluate_signal` → `arbitration_engine.evaluate()` — this is the actual production decision path (`main.py` → façade `DecisionEngine` → `implementations.py`), NOT `core_engine/native/decisions.py::NativeDecisionEngine.decide()` (which is only reachable via the inert `NativeOrchestrator.run_loop()`). The original Phase 1 static trace missed this because it only searched `orchestrator.py`, `decisions.py`, `executor.py` — not `implementations.py` | No confirmed loop | `test_native_l4.py`, `test_race_conditions_and_growth.py` | **Live and wired.** Not exercised in the Phase 2 observation session only because zero signals reached the decision stage that session (upstream `PERSIST_GATE`/confidence-floor gating), not because arbitration itself is unreachable |

## Tier B — Supporting / Observability (inventory depth)

| File | Class/Fn | Purpose | In bootstrap.py? | Instantiated elsewhere? | Test? |
|---|---|---|---|---|---|
| concentration_guard.py | `NativeConcentrationGuard` | Blocks BUY decisions over-concentrating in one theme cluster | No | Yes — `decisions.py:127` | No |
| health_monitor.py | `NativeHealthMonitor` | Native system health monitor | Yes (`bootstrap.py:927`) | — | Indirect (`test_native_health_regime.py`, `test_native_bootstrap.py`) |
| cadence_scheduler.py | `CadenceScheduler` | Multi-speed scheduler for engine phases within one loop | No | Yes — `main.py:148` | `test_cadence_scheduler.py` |
| model_manager.py | `ModelManager` | Loads/caches/validates TF Keras models, quarantines corrupt artifacts | Yes (`bootstrap.py:675`, inline import) | — | `test_native_model_manager.py` |
| nav_protection.py | `NAVProtectionEngine` / `NAVAttributionEngine` | NAV-based attribution & protection (drawdown/PnL) | No | Only self-referential default args inside its own module (`nav_protection.py:427-428`) — not found instantiated in bootstrap.py or elsewhere | `test_nav_protection.py` |
| fear_greed.py | `FearGreedFetcher` | Fetches Fear & Greed Index from alternative.me | Yes (`bootstrap.py:883`) | — | No |
| market_regime_detector.py | `NativeMarketRegimeDetector` | Per-symbol trend/regime classification | Yes (`bootstrap.py:848`) | — | Indirect |
| polling_coordinator.py | `NativePollingCoordinator` | Staggered, active-trades-gated polling (substitutes for fill_tracker) | Yes (`bootstrap.py:503`) | — | `test_native_polling_reconciliation.py` |
| mode_manager.py | `NativeModeManager` | Durable operating-mode state machine | Yes (`bootstrap.py:738`) | — | `test_native_l4.py`, `test_native_l8.py` |
| regime_gate.py | `NativeRegimeGate` | Filters new-position opens by market condition | No | Yes — `decisions.py:131`, `arbitration_engine.py:45` | `test_native_l4.py` |
| bounded_cache.py | `NativeBoundedCache` | TTL cache for execution idempotency guards | Yes (`bootstrap.py:935`) | — | Indirect |
| compat.py | `make_compat_stubs`/`register_compat_stubs`/`_NullStub` | Null-stub compat shims for legacy app_ctx keys | Unclear — no confirmed call in bootstrap.py | Not found via grep | `test_native_compat.py` |
| quant_reasoning.py | `select_playbook`/`classify_market_regime`/`compute_probability_score` | Situation classification helpers | No (function-based) | Not directly grepped | `test_overextension_guard.py` |
| symbol_performance_tracker.py | `SymbolPerformanceTracker` | Rolling per-symbol edge scoring; throttles poor performers | No | Yes — `arbitration_engine.py:53` | No |
| paper_signal_generator.py | `PaperModeSignalGenerator` | Synthetic BUY/SELL signals for paper-mode testing | Yes (`bootstrap.py:652`) | — | No |
| capital_policy.py | `compute_spendable_quote`/`prune_reservations` | Shared reserve/spendable-capital math | No (function-based) | Not confirmed this pass — needs follow-up | No |
| legacy_signal_adapter.py | `LegacySignalAdapter` | Bridges legacy signal agents into native pipeline | No | Yes — `signal_manager_bridge.py:67` (lazy import) | `test_quant_terminal_layer.py` |
| signal_manager_bridge.py | `SignalManagerBridge` | Bridges legacy SignalManager + native paper generator | Yes (`bootstrap.py:711-717`) | — | `test_native_bootstrap.py` |

### Anomaly flags (Tier B)
- `concentration_guard.py`, `symbol_performance_tracker.py`: wired into `decisions.py`/`arbitration_engine.py` but not in `bootstrap.py` and **no test file** — untested code on a live path.
- `paper_signal_generator.py`, `fear_greed.py`: instantiated but **no dedicated test file**.
- `nav_protection.py`: has tests but **no confirmed instantiation site** outside its own module — candidate unwired feature, needs Phase 2 runtime confirmation.
- `capital_policy.py`, `compat.py`, `quant_reasoning.py`: function-only modules; call-site wiring not confirmed by class-name grep, needs follow-up.

## Other native/ files (inventory only, not independently deep-dived)

| File | Purpose | In bootstrap.py |
|---|---|---|
| app_context.py | Assembles app_ctx dict, wires `NativeOrchestrator` | N/A (caller of bootstrap) |
| orchestrator.py | `NativeOrchestrator` — top-level L8 orchestrator; starts all pollers | N/A (caller of bootstrap) |
| config_loader.py | `ConfigLoader`/`ConfigGroup` — second, parallel config system | No (separate from `BootstrapConfig`) |
| error_types.py | `OctiError`, `BootstrapError`, severity/category/recovery enums | No |
| math_utils.py | sharpe/sortino/calmar/max_drawdown functions | No |
| time_utils.py | `NativeTimeUtils` | No |
| retry_manager.py | `NativeRetryManager` | No |
| symbol_discovery.py | `NativeSymbolDiscovery` — wallet-scan discovery | Yes |
| startup_state_machine.py | `NativeStartupStateMachine` — gated startup sequence | Yes |
| trade_journal.py | `NativeTradeJournal` | Yes |
| position_hydration_engine.py | `NativePositionHydrationEngine` | Yes |
| exchange_client.py | `NativeExchangeClient` | Yes |
| order_execution.py | `NativeOrderExecution` | Yes |
| balance_sync.py | `NativeBalanceSync` (legacy poller) | Yes (constructed, started only if `polling_coordinator` absent) |
| market_data.py | `NativeMarketData` (REST poller) | Yes (constructed, not started in bootstrap — orchestrator starts it) |
| market_data_websocket.py | `NativeMarketDataWebSocket` | Yes (constructed, not started in bootstrap) |
| portfolio_manager.py | `NativePortfolioManager` | Yes |
| position_manager.py | `NativePositionManager` | Yes |
| recovery_engine.py | `NativeRecoveryEngine` | Yes |
| safety_order_manager.py | `NativeSafetyOrderManager` | Yes |
| observability.py | `NativeTelemetry` | Yes |
| watchdog.py | `NativeWatchdog` | Yes |
| prometheus_exporter.py | `NativePrometheusExporter` | Yes |
| portfolio_recovery.py | Portfolio recovery / dust disposal | Not independently confirmed this pass |
| balance_validator.py | Balance validator | Not independently confirmed this pass |
| signal_fusion.py | Signal-fusion adapter (consumed by arbitration_engine) | Not independently confirmed this pass |
| signals.py | L3 signal engine (`NativeSignalEngine`, cross-checks ML confidence) | Consumed by `signal_manager_bridge.py` |
| telemetry_export.py | L6 telemetry exporter | Yes (`telemetry_exporter.start()` in bootstrap, gated by `TELEMETRY_EXPORT_PATH`) |
| objective_feedback_controller.py | `ObjectiveFeedbackController` (`NativeObjectiveFeedbackController`) | Yes (`bootstrap.py:804-805`, gated by `ofc_enabled`, default `True`); `.start()` via orchestrator (`orchestrator.py:227-228`); heartbeat loop `_run_loop` (min 60s) |
| runtime_state.py | `NativeRuntimeStateExporter` | Yes (`bootstrap.py`, `.start()` called directly in `build_components`, writes `runtime_state_snapshot.json` periodically) |
| shared_state.py | `NativeSharedState` | Yes (`bootstrap.py:465`) — de-facto singleton, instantiated once, passed by reference everywhere; not an enforced singleton pattern |
