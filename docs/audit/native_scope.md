# Native Solution — Scope Map

**Audit baseline:** branch `phase-3/wiring`, HEAD `cecdf363590a09b31ff528372ec0a314c7c0ab5f`
("fix: raise TOP_N 8→13 so BTC/ETH/SOL/XRP fit alongside best alts", 2026-06-20).

**Working tree state at time of audit: DIRTY.** 28 tracked files modified, 6 untracked
files present. This audit observes the system *as it currently sits*, not a clean
commit — every finding below should be read as "true of this exact working tree,"
not "true of the last commit." Per user instruction, the tree was not stashed.

Modified: `agents/ml_forecaster.py`, `backtest_edge.py`, `config/EV_ALIGNMENT_CONFIG.py`,
`core_engine/implementations.py`, `core_engine/native/__init__.py`,
`core_engine/native/adaptive_capital_engine.py`, `core_engine/native/bootstrap.py`,
`core_engine/native/capital_allocator.py`, `core_engine/native/decisions.py`,
`core_engine/native/executor.py`, `core_engine/native/fill_tracker.py`,
`core_engine/native/objective_feedback_controller.py`, `core_engine/native/runtime_state.py`,
`core_engine/native/shared_state.py`, `core_engine/native/symbol_rotator.py`,
`core_engine/native/tp_sl_engine.py`, `objective_controller_state.json`,
`retrain_weekly.py`, `runtime_state_snapshot.json`, `src/l5_strategy/model_trainer.py`,
`supervisor.sh`, `tests/test_model_trainer_bootstrap_labels.py`, `tests/test_native_bootstrap.py`,
`tests/test_native_fill_tracker.py`, `tests/test_native_l4.py`, `tests/test_native_l5.py`,
`tests/test_native_runtime_state.py`, `trade_monitor.py`.

Untracked: `core_engine/native/daily_compounding.py`, `strategy_validation.py`,
`tests/test_daily_compounding.py`, `tests/test_objective_quality_guards.py`,
`tests/test_strategy_validation.py`, `tests/test_symbol_rotator_quality.py`.

## Entry points

| Entry point | Path | Role |
|---|---|---|
| Production supervisor | `supervisor.sh` | Sole production launch path. Runs `.venv/bin/python3 main.py`, handles crash-loop backoff, log rotation, and a stall watchdog that force-restarts `main.py` if no new "cycle N" log line appears for 180s. Started via `nohup ./supervisor.sh >> logs/supervisor.out 2>&1 &`; stopped via `touch logs/supervisor.stop`. |
| Application entry point | `main.py` | Invoked as `python3 main.py --mode=live\|paper-trade\|dry-run ...` (default mode: `live`). Docstring states an explicit architectural contract: `main.py` may only import the 5 façade engines in `core_engine/` — never L0-L8 native modules directly. |
| Composition root | `core_engine/native/bootstrap.py::build_components(cfg)` (async, ~lines 438-1005) | Constructs the entire native L0-L8 stack. `BootstrapConfig.from_env()` reads ~50 env vars. Not called directly by `main.py` — reached via `core_engine/integration.py`. |

## Real startup call chain

1. `supervisor.sh` → `.venv/bin/python3 main.py`
2. `main.py:555` → `native = not args.no_native` (default `True`)
3. `main.py:572` → `app_ctx = await setup_core_engines(native=native, compat=compat)`
4. `core_engine/integration.py::setup_core_engines()` → `create_app_context(native=True)` → imports `BootstrapConfig`, `build_components` from `core_engine.native.bootstrap`; `cfg = BootstrapConfig.from_env()`; `components = await build_components(cfg)`
5. `core_engine/native/app_context.py::build_native_app_ctx(components, compat=compat)` — wraps the un-started components into a `NativeOrchestrator`, stores it as `app_ctx["_native_orchestrator"]`
6. `main.py:93` → `Engines.initialize()` calls `await self.operations.startup_system()` unconditionally on every boot → `core_engine/implementations.py::OperationsEngineImpl.startup_system()` → `native_orch = app_ctx.get("_native_orchestrator"); await native_orch.start()` — this is what actually starts the background pollers (see `runtime_task_inventory.md`, Phase 2).
7. `main.py:676-681` → `core_engine.native.bootstrap.shutdown_components()` on exit.

Other scripts that import `bootstrap.build_components` directly but are **not** part of the
supervised runtime (ad-hoc/dev tooling, not launched by `supervisor.sh`): `monitor_live_trading.py`,
`monitor_capital_growth.py`, `run_and_monitor.py`, `wait_for_throttle_expiry_and_test.py`,
`test_after_throttle_expires.py`, `backtest_edge.py`, `scripts/native_smoke.py`, `tests/test_native_*.py`.

`trade_monitor.py` is a companion monitoring script, not a launcher — it does not call
`bootstrap` itself; it reads native state (imports `cadence_scheduler`, `nav_protection`).

## Scope classification

| Area | Path | Classification | Reason |
|---|---|---|---|
| Native | `core_engine/native/` (58 files, L0-L8) | Included | Current implementation; composition root; file headers explicitly say things like "replaces 300-line legacy ConfigConstants," "replaces 1,200-line legacy SharedState" |
| Native façade | `core_engine/{market_account,situation,decision,safe_execution,operations}_engine.py`, `implementations.py` | Included | The only modules `main.py` is contractually allowed to import; `implementations.py` is a native/legacy bridge (see below) |
| Legacy | `src/l0_core/`, `src/l3_portfolio/`, `src/l5_strategy/` | Excluded (superseded) | Older layered architecture. `src/l5_strategy/model_trainer.py` is still actively invoked (by `agents/ml_forecaster.py` and `retrain_weekly.py`), so "excluded" means "not part of the native package," not "unused" |
| Shared/bridge | `core_engine/native/legacy_signal_adapter.py`, `signal_manager_bridge.py`, `core_engine/implementations.py`, `agents/ml_forecaster.py`, `agents/symbol_screener.py` | Needs review | Formal bridges between native and legacy signal generation; actively wired into the live signal path (see `feature_ignition_matrix.md`) |
| Duplicate config system | `core_engine/native/bootstrap.py` (`BootstrapConfig`) vs `core_engine/native/config_loader.py` (`ConfigLoader`) | Needs review | Two independent env-var-driven config systems coexist, with naming collisions on the same settings (see `configuration_map.md`) |
| Unknown / offline tooling | `strategy_validation.py`, `retrain_weekly.py`, `config/STRATEGY_OPTIMIZATION_v2.py`, `_archaeology/` | Investigate | `strategy_validation.py` is new, untracked, and not imported anywhere except its own test — appears to be an unwired offline validation script. `retrain_weekly.py` is a standalone cron script, intentionally decoupled from the live runtime. `config/STRATEGY_OPTIMIZATION_v2.py` is a dated one-off patch file not imported by any active code. `_archaeology/` contains a prior dead-code audit (`inventory.py`, `orphans_full.txt`, `unreached_from_entry.txt`, `entry_points.txt`, `live_dependency_closure.txt`) — reused as a starting point for this audit's import-graph work, though it predates this branch's untracked files |
| Root clutter | ~20 top-level `.sh` scripts (`monitor_growth.sh`, `enable_growth.sh`, `carry_supervisor.sh`, `LAUNCH_MONITOR.sh`, etc.), dozens of root-level `.md` status/fix reports, multiple `.env*` variants | Excluded / not part of production path | Only `supervisor.sh` is the canonical launcher; the rest appear to be one-off session/monitor scripts from prior development rounds — flagged as candidate clutter, not verified dead |

## Environment / dependencies

- No Dockerfile, docker-compose, or `.github/workflows` found — no containerization, no CI pipeline.
- Manifests: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml` at repo root.
- `deployment/` contains only `jaeger-deployment.yaml` (tracing) — not a deployment pipeline.
- `tests/` has ~65+ files, mostly `test_native_*.py` (one per L0-L8 layer) plus `tests/layers/`.
- No pre-existing `docs/audit/` directory (this audit creates it).
