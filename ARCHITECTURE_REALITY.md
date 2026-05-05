# ARCHITECTURE REALITY
_Generated 2026-05-05T17:16:24_  
_Total Python files scanned: **316**_

## 1. Status summary

| Status | Count |
|---|---|
| 🟢 LIBRARY | 155 |
| 🟢 ENTRY-POINT | 81 |
| 🔴 ORPHAN | 61 |
| 🟠 PATCH-ARTIFACT | 19 |

## 2. Location summary

| Location | Count |
|---|---|
| `src/` (clean architecture) | 0 |
| Repo root (likely chaos)    | 31 |
| Other folders               | 285 |

## 3. 🟢 Entry-point candidates (has `__main__`, not imported)

| File | LOC | Last commit |
|---|---:|---|
| `master_orchestrator.py` | 2966 | 2026-04-28 |
| `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | 2966 | 2026-05-04 |
| `tests/test_portfolio_fragmentation_integration.py` | 772 | 2026-04-27 |
| `docs/archive/scripts/2HOUR_CHECKPOINT_SESSION.py` | 568 | 2026-04-28 |
| `tools/detect_balance_symbols.py` | 498 | 2026-05-04 |
| `docs/archive/scripts/RUN_6HOUR_SESSION_MONITORED.py` | 481 | 2026-04-28 |
| `docs/archive/scripts/MONITOR_3HOUR_TRADING_SESSION.py` | 474 | 2026-04-28 |
| `scripts/check_layer_imports.py` | 474 | 2026-05-03 |
| `docs/archive/scripts/phase3_live_trading.py` | 441 | 2026-04-28 |
| `docs/archive/scripts/UNIT_TEST_EXECUTION_GUIDE.py` | 374 | 2026-04-28 |
| `docs/archive/scripts/CONTINUOUS_OPERATION_GUIDE.py` | 359 | 2026-04-28 |
| `docs/archive/scripts/RUN_6HOUR_SESSION.py` | 356 | 2026-04-28 |
| `docs/archive/scripts/phase2_paper_trading.py` | 355 | 2026-04-28 |
| `docs/archive/scripts/run_4hour_session.py` | 350 | 2026-04-28 |
| `docs/archive/scripts/LIVE_MONITOR.py` | 324 | 2026-04-28 |
| `src/l8_lifecycle/runners/apply_recovery_to_live.py` | 324 | 2026-04-28 |
| `docs/archive/scripts/MONITOR_15MIN_SESSION.py` | 313 | 2026-04-28 |
| `monitoring/real_time_dashboard.py` | 313 | 2026-05-03 |
| `src/l8_lifecycle/runners/component_validator.py` | 312 | 2026-04-28 |
| `docs/archive/scripts/LIVE_PHASE2_MONITOR.py` | 307 | 2026-04-28 |
| `docs/archive/scripts/CONTINUOUS_ACTIVE_MONITOR.py` | 301 | 2026-04-28 |
| `config/EV_ALIGNMENT_CONFIG.py` | 289 | 2026-04-27 |
| `tools/monitor_6h_session.py` | 278 | 2026-04-27 |
| `src/l8_lifecycle/runners/objective_tracker.py` | 277 | 2026-05-03 |
| `tools/next_level_tpsl_analysis.py` | 276 | 2026-03-01 |
| `docs/archive/scripts/AUTONOMOUS_STARTUP_GUIDE.py` | 275 | 2026-04-28 |
| `docs/archive/scripts/MONITOR_15MIN_REALTIME.py` | 272 | 2026-04-28 |
| `phase1_recover_large.py` | 246 | 2026-05-04 |
| `src/l7_observability/monitors/monitor_4hour_session.py` | 242 | 2026-04-28 |
| `_test_failure_modes.py` | 238 | untracked |
| `docs/archive/scripts/phase4_verify.py` | 233 | 2026-04-28 |
| `docs/archive/scripts/6HOUR_MONITORING_DASHBOARD.py` | 226 | 2026-04-28 |
| `force_liquidate_dust.py` | 223 | 2026-05-04 |
| `_test_safety_order_manager.py` | 220 | untracked |
| `capital_health_monitor.py` | 216 | 2026-05-03 |
| `docs/archive/scripts/PROFIT_ACCUMULATOR_MONITOR.py` | 207 | 2026-04-28 |
| `phase1_dust_convert.py` | 207 | 2026-05-04 |
| `src/l7_observability/monitors/phase2_monitoring.py` | 204 | 2026-04-28 |
| `docs/archive/scripts/TEST_EXIT_FIRST_VALIDATION.py` | 200 | 2026-04-28 |
| `docs/archive/scripts/PERSISTENT_TRADING_WATCHDOG.py` | 198 | 2026-04-28 |
| _...41 more_ | | |

## 4. 🟠 Patch-artifacts (filename contains FIX/OLD/BACKUP/v2/etc.)

| File | LOC | Imported by | Last commit |
|---|---:|---:|---|
| `CAPITAL_ALLOCATOR_FIX_CODE.py` | 349 | 0 | 2026-05-04 |
| `FIX_BALANCE_BLEED_IMPLEMENTATION.py` | 546 | 0 | 2026-05-03 |
| `balance_threshold_config.py` | 155 | 1 | 2026-04-27 |
| `docs/archive/scripts/test_rounding_fix.py` | 230 | 0 | 2026-04-28 |
| `fix_execution_deadlock.py` | 126 | 0 | 2026-05-03 |
| `src/l3_portfolio/holding_utility.py` | 229 | 2 | 2026-04-28 |
| `src/l8_lifecycle/runners/verify_dust_fix.py` | 120 | 0 | 2026-04-28 |
| `src/l8_lifecycle/runners/verify_fixes.py` | 199 | 0 | 2026-04-28 |
| `src/l8_lifecycle/runners/verify_fixes_detailed.py` | 309 | 0 | 2026-04-28 |
| `tests/test_consolidation_exception_fix.py` | 164 | 0 | 2026-04-28 |
| `tests/test_portfolio_fragmentation_fixes.py` | 688 | 0 | 2026-04-27 |
| `tools/advanced_fix_python_indentation.py` | 62 | 0 | 2026-02-11 |
| `tools/fix_class_decorator_indentation.py` | 40 | 0 | 2026-02-11 |
| `tools/fix_indentation.py` | 46 | 0 | 2026-02-11 |
| `tools/fix_python_indentation.py` | 47 | 0 | 2026-02-11 |
| `tools/smart_python_indentation_fixer.py` | 72 | 0 | 2026-02-11 |
| `validate_churn_fix.py` | 361 | 0 | 2026-05-03 |
| `verify_capital_allocator_fix.py` | 316 | 0 | 2026-05-04 |
| `verify_ws_fix.py` | 164 | 0 | 2026-05-03 |

## 5. 🔴 Orphans (no importers, no `__main__`) — top quarantine candidates

_Total orphans: **61**_

| File | LOC | Last commit |
|---|---:|---|
| `tests/conftest.py` | 514 | 2026-04-28 |
| `src/l5_strategy/objective_feedback_controller.py` | 490 | 2026-04-28 |
| `tests/test_self_healing_controller.py` | 436 | 2026-04-28 |
| `docs/archive/scripts/PHASE_2_STATUS_REPORT.py` | 185 | 2026-04-28 |
| `tests/test_portfolio_target_size_enforcer.py` | 155 | 2026-05-03 |
| `tests/test_strict_cap_count_tradable.py` | 154 | 2026-04-28 |
| `tests/test_truth_audit_wallet_guard.py` | 154 | 2026-04-28 |
| `config/STRATEGY_OPTIMIZATION_v2.py` | 129 | 2026-05-03 |
| `tests/test_insuff_bal_circuit_breaker.py` | 108 | 2026-04-28 |
| `docs/archive/scripts/SIGNAL_FLOW_DIAGNOSTIC.py` | 100 | 2026-04-28 |
| `tests/test_live_order_recovery_guards.py` | 100 | 2026-05-03 |
| `tests/test_layered_architecture.py` | 97 | 2026-04-28 |
| `tests/test_layer_namespace.py` | 83 | 2026-04-28 |
| `_validate_failure_modes.py` | 77 | 2026-05-04 |
| `_check_real_balance.py` | 73 | 2026-05-04 |
| `tests/test_sell_finalize_idempotency.py` | 71 | 2026-05-03 |
| `tests/layers/test_l4_execution.py` | 68 | 2026-04-28 |
| `show_detected_symbols.py` | 65 | 2026-04-28 |
| `tests/layers/test_l1_exchange.py` | 62 | 2026-04-28 |
| `tests/test_dust_exit_candidate_selection.py` | 58 | 2026-05-03 |
| `docs/archive/scripts/test_trendhunter_import.py` | 55 | 2026-04-28 |
| `tests/layers/test_l0_cross_cutting.py` | 45 | 2026-04-28 |
| `tests/layers/test_l3_portfolio.py` | 44 | 2026-04-28 |
| `tests/layers/test_l5_strategy.py` | 42 | 2026-04-28 |
| `tests/layers/test_l6_governance.py` | 41 | 2026-04-28 |
| `src/l7_observability/monitors/extract_rejections.py` | 39 | 2026-04-28 |
| `tests/layers/test_l7_observability.py` | 36 | 2026-04-28 |
| `tests/layers/test_l2_wallet.py` | 33 | 2026-04-28 |
| `src/__init__.py` | 28 | 2026-04-28 |
| `config/__init__.py` | 20 | 2026-04-10 |
| `dashboards/__init__.py` | 20 | 2026-04-10 |
| `deployment/__init__.py` | 20 | 2026-04-10 |
| `models/__init__.py` | 20 | 2026-04-10 |
| `portfolio/__init__.py` | 20 | 2026-04-10 |
| `scripts/__init__.py` | 20 | 2026-04-10 |
| `stream/__init__.py` | 20 | 2026-04-10 |
| `tools/__init__.py` | 20 | 2026-04-10 |
| `agents/__init__.py` | 19 | 2026-04-10 |
| `tests/__init__.py` | 19 | 2026-04-10 |
| `src/l0_core/__init__.py` | 8 | 2026-04-28 |
| `src/l1_exchange/__init__.py` | 4 | 2026-04-28 |
| `src/l2_marketdata/__init__.py` | 4 | 2026-04-28 |
| `src/l3_portfolio/__init__.py` | 4 | 2026-04-28 |
| `src/l4_execution/__init__.py` | 4 | 2026-04-28 |
| `src/l5_strategy/__init__.py` | 4 | 2026-04-28 |
| `src/l6_governance/__init__.py` | 4 | 2026-04-28 |
| `src/l7_observability/__init__.py` | 4 | 2026-04-28 |
| `src/l8_lifecycle/__init__.py` | 4 | 2026-04-28 |
| `tests/layers/__init__.py` | 2 | 2026-04-28 |
| `docs/archive/scripts/phase4_quick_validation.py` | 1 | 2026-04-28 |
| `monitoring/__init__.py` | 1 | 2026-05-03 |
| `monitoring/capital_dashboard.py` | 1 | 2026-05-03 |
| `monitoring/capital_growth_dashboard.py` | 1 | 2026-05-03 |
| `monitoring/capital_growth_monitor.py` | 1 | 2026-05-03 |
| `monitoring/monitor_integration.py` | 1 | 2026-05-03 |
| `monitoring/verify_monitor_setup.py` | 1 | 2026-05-03 |
| `src/l7_observability/diagnostics/__init__.py` | 1 | 2026-04-28 |
| `src/l7_observability/monitors/__init__.py` | 1 | 2026-04-28 |
| `src/l8_lifecycle/runners/__init__.py` | 1 | 2026-04-28 |
| `tests/test_nav_no_double_count.py` | 1 | 2026-04-28 |
| _...1 more (see orphans_full.txt)_ | | |

## 6. Suspected duplicate groups (similar normalized names)

### Group `init`

- `agents/__init__.py` — 19 LOC, 🔴 ORPHAN, imported_by=0
- `config/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `dashboards/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `deployment/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `models/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `monitoring/__init__.py` — 1 LOC, 🔴 ORPHAN, imported_by=0
- `portfolio/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `scripts/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `src/__init__.py` — 28 LOC, 🔴 ORPHAN, imported_by=0
- `src/l0_core/__init__.py` — 8 LOC, 🔴 ORPHAN, imported_by=0
- `src/l1_exchange/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l2_marketdata/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l3_portfolio/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l4_execution/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l5_strategy/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l6_governance/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l7_observability/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l7_observability/diagnostics/__init__.py` — 1 LOC, 🔴 ORPHAN, imported_by=0
- `src/l7_observability/monitors/__init__.py` — 1 LOC, 🔴 ORPHAN, imported_by=0
- `src/l8_lifecycle/__init__.py` — 4 LOC, 🔴 ORPHAN, imported_by=0
- `src/l8_lifecycle/runners/__init__.py` — 1 LOC, 🔴 ORPHAN, imported_by=0
- `stream/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `tests/__init__.py` — 19 LOC, 🔴 ORPHAN, imported_by=0
- `tests/layers/__init__.py` — 2 LOC, 🔴 ORPHAN, imported_by=0
- `tools/__init__.py` — 20 LOC, 🔴 ORPHAN, imported_by=0
- `utils/__init__.py` — 1 LOC, 🔴 ORPHAN, imported_by=0

### Group `autorecovery`

- `auto_recovery.py` — 26 LOC, 🟢 LIBRARY, imported_by=3
- `src/l8_lifecycle/runners/auto_recovery.py` — 229 LOC, 🟢 LIBRARY, imported_by=4

### Group `liveintegration`

- `docs/archive/scripts/live_integration.py` — 25 LOC, 🟢 LIBRARY, imported_by=1
- `src/l8_lifecycle/runners/live_integration.py` — 64 LOC, 🟢 LIBRARY, imported_by=2

## 7. Top 20 LIBRARIES by fan-in (most-imported = most critical)

| File | Imported by N files | Sample importers |
|---|---:|---|
| `src/l0_core/shared_state.py` | 28 | `agents/wallet_scanner_agent.py`, `diagnose_healing.py`, `docs/archive/scripts/FORCE_SIGNALS_INJECTOR.py` |
| `src/l1_exchange/exchange_client.py` | 17 | `diagnose_healing.py`, `docs/archive/scripts/FORCE_SIGNALS_INJECTOR.py`, `docs/archive/scripts/REALTIME_MONITOR.py` |
| `src/l0_core/stubs.py` | 16 | `agents/dip_sniper.py`, `agents/liquidation_agent.py`, `agents/ml_forecaster.py` |
| `src/l0_core/config.py` | 14 | `diagnose_healing.py`, `docs/archive/scripts/FORCE_SIGNALS_INJECTOR.py`, `docs/archive/scripts/TEST_BOOTSTRAP.py` |
| `src/l4_execution/execution_manager.py` | 11 | `docs/archive/scripts/TEST_EXIT_FIRST_VALIDATION.py`, `force_liquidate_dust.py`, `master_orchestrator.py` |
| `src/l8_lifecycle/meta_controller.py` | 11 | `docs/archive/scripts/FORCE_SIGNALS_INJECTOR.py`, `docs/archive/scripts/LIVE_TRADING_WITH_BALANCE_MONITOR.py`, `docs/archive/scripts/TEST_EXIT_FIRST_VALIDATION.py` |
| `src/l0_core/component_status_logger.py` | 10 | `agents/dip_sniper.py`, `agents/liquidation_agent.py`, `agents/ml_forecaster.py` |
| `src/l3_portfolio/bootstrap_symbols.py` | 10 | `agents/dip_sniper.py`, `agents/ml_forecaster.py`, `agents/swing_trade_hunter.py` |
| `src/_lazy.py` | 9 | `src/l0_core/__init__.py`, `src/l1_exchange/__init__.py`, `src/l2_marketdata/__init__.py` |
| `src/l0_core/layer_contracts.py` | 9 | `master_orchestrator.py`, `scripts/check_layer_imports.py`, `tests/layers/test_l1_exchange.py` |
| `tests/layers/fakes.py` | 7 | `tests/layers/test_l1_exchange.py`, `tests/layers/test_l2_wallet.py`, `tests/layers/test_l3_portfolio.py` |
| `utils/shared_state_tools.py` | 7 | `agents/ml_forecaster.py`, `agents/swing_trade_hunter.py`, `src/l3_portfolio/holding_utility.py` |
| `system_state_manager.py` | 6 | `docs/archive/scripts/LIVE_TRADING_WITH_BALANCE_MONITOR.py`, `docs/archive/scripts/phase4_30min_test.py`, `docs/archive/scripts/run_4hour_session.py` |
| `utils/ta_indicators.py` | 6 | `agents/dip_sniper.py`, `agents/swing_trade_hunter.py`, `agents/trend_hunter.py` |
| `utils/tuned_params.py` | 6 | `agents/dip_sniper.py`, `agents/swing_trade_hunter.py`, `agents/trend_hunter.py` |
| `src/l3_portfolio/event_store.py` | 5 | `master_orchestrator.py`, `src/l3_portfolio/replay_engine.py`, `src/l4_execution/execution_manager.py` |
| `src/l5_strategy/model_manager.py` | 5 | `agents/ipo_chaser.py`, `agents/swing_trade_hunter.py`, `agents/trend_hunter.py` |
| `src/l5_strategy/signal_manager.py` | 5 | `docs/archive/scripts/FORCE_SIGNALS_INJECTOR.py`, `docs/archive/scripts/diagnostic_signal_flow.py`, `master_orchestrator.py` |
| `utils/indicators.py` | 5 | `agents/dip_sniper.py`, `agents/swing_trade_hunter.py`, `agents/trend_hunter.py` |
| `src/_layer_index.py` | 4 | `scripts/migrate_consumers.py`, `src/__init__.py`, `src/_lazy.py` |
