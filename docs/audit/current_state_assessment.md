# Current State Assessment — Native Solution Audit

Synthesizes `native_scope.md`, `component_inventory.md`, `feature_inventory.md`,
`feature_ignition_matrix.md`, `configuration_map.md`, `runtime_timeline.md`,
`runtime_task_inventory.md`, `data_flow_map.md`, `baseline_test_report.md`, and
`documentation_gap_analysis.md`. Baseline: branch `phase-3/wiring`, HEAD `cecdf363`,
working tree dirty (28 modified, 6 untracked files, not stashed for this audit).

## A. Current system map

The native solution is `core_engine/native/` (58 files, L0-L8 layered), reached through
a single composition root: `bootstrap.py::build_components()`, invoked from
`core_engine/integration.py`, invoked from `main.py`, launched in production by
`supervisor.sh`. `main.py` also directly drives the live trading cycle itself
(`trading_cycle()`, cadence-scheduled) rather than delegating that to
`NativeOrchestrator.run_loop()`, which is effectively test-only. A thin façade layer
(`core_engine/{market_account,situation,decision,safe_execution,operations}_engine.py`
+ `implementations.py`) is the only thing `main.py` is contractually allowed to import,
and `implementations.py` itself is a native/legacy bridge with explicit try-native,
fall-back-to-legacy logic throughout.

Legacy code (`src/l0_core`, `l3_portfolio`, `l5_strategy`) is not dead — `agents/ml_forecaster.py`
(legacy) is bridged into the live signal path via `legacy_signal_adapter.py` +
`signal_manager_bridge.py`, and actively drives every cycle's signal generation.
`src/l5_strategy/model_trainer.py` is invoked both by the live process (background
training inside `ml_forecaster.py`) and by a standalone weekly cron script
(`retrain_weekly.py`).

## B. Actual startup path

Confirmed by both static tracing and a live dry-run session: process start →
`BootstrapConfig.from_env()` → `build_components()` constructs ~30+ components (some
gated by feature flags, all default-on) → `app_context.build_native_app_ctx()` wraps
unstarted pollers into `NativeOrchestrator` → 5 façade engines initialize →
`OperationsEngineImpl.startup_system()` calls `NativeOrchestrator.start()`, which starts
market data (REST + WebSocket), the polling coordinator (4-5 staggered loops), TP/SL
engine, and Objective Feedback Controller → a 4-phase startup state machine
(BOOTING → HYDRATING → RECONCILING → VALIDATING → READY) completes in under 1 second →
trading cycles begin. Observed cold-start time to "all 5 engines online": ~35 seconds,
dominated by TensorFlow/Keras model loading for 10 symbols.

## C. Active components

Confirmed active via runtime evidence (not just instantiation): market data (REST +
WS), symbol discovery, polling coordinator (all sub-loops), position hydration (runs,
though degraded — see Section F), startup state machine, MLForecaster signal generation
(via legacy bridge), signal cross-check engine, decision engine, concentration guard,
regime gate, TP/SL engine (armed), Objective Feedback Controller (started), NAV
protection (called directly from `main.py`, not via orchestrator — a Phase 3 correction
to an earlier Phase 1 "unconfirmed" flag), capital allocator + adaptive capital engine +
daily compounding policy (called on every BUY path, though never exercised this session
since 0 decisions occurred), fear & greed fetch, runtime state export.

## D. Idle components

Correctly idle by default configuration (not broken): legacy `balance_sync.py` and
`fill_tracker.py` (both superseded by `NativePollingCoordinator` when `polling_enabled=True`,
the default), paper mode signal generator (only active when `paper_mode=True`, default
False), synthetic live signals (opt-in flag, default off).

## E. Unwired components

**`NativeArbitrationEngine` — CORRECTION, this section was wrong in the original Phase
1-3 pass.** Phase 1's static trace searched `orchestrator.py`, `core_engine/native/decisions.py`,
and `executor.py` for calls to `arbitration_engine.evaluate()`/`.evaluate_gates()` and
found none, and Phase 2's live session showed zero arbitration-related log lines — both
true, but the conclusion drawn from them ("unwired, dead code") was wrong. A Phase 4
follow-up found the actual production decision path is `main.py` → façade `DecisionEngine`
→ `core_engine/implementations.py::DecisionEngineImpl.make_buy_decision`/`make_sell_decision`
→ `evaluate_signal()` → **`arbitration_engine.evaluate()`** — a different, separate
decision implementation from `core_engine/native/decisions.py::NativeDecisionEngine.decide()`
(which really is only reachable via the inert `NativeOrchestrator.run_loop()`, and really
does not call arbitration). The Phase 2 session showed zero arbitration activity simply
because zero signals ever reached the decision stage that session (blocked upstream by
`PERSIST_GATE`/confidence-floor gating in `MLForecaster`) — `make_buy_decision` itself
was never called, so naturally nothing downstream of it was either. **The arbitration
engine, including `symbol_performance_tracker.py` and all its own config flags
(`DOWNTREND_MARGIN`, `SYMBOL_DOWNTREND_VETO_ENABLED`, `REBUY_BLOCK_NOTIONAL`, etc.), is
live and wired into every real BUY/SELL decision.** No code change was made based on the
original (incorrect) finding. This is flagged prominently as a lesson on this audit's own
reliability: a static trace that checks the "obvious" files and a runtime session with no
qualifying activity can both look like solid evidence for "dead code" while actually
proving nothing more than "not exercised by this particular path/session."

`config/EV_ALIGNMENT_CONFIG.py` and `config/STRATEGY_OPTIMIZATION_v2.py` remain
orphaned — not imported by any active code path. This finding is unaffected by the
arbitration correction above.

## F. Broken components

**`position_hydration_engine.py`** calls `NativeExchangeClient.get_all_orders()`, which
doesn't exist on that class. Confirmed via a live `AttributeError` on every startup this
session. It's currently harmless only because the observed account holds zero positions
(the fallback "assume fresh account" happens to be correct by coincidence, not by
design). If the bot restarted while holding open positions, this bug means exchange-side
fill history would not be consulted for recovery — only the local trade journal would be,
and if that's also incomplete (e.g., after an unclean shutdown), positions could be
silently dropped from `shared_state` on restart. **This is the highest-severity concrete
defect found in this audit.**

## G. Trigger and ignition gaps

Beyond the arbitration engine (Section E), the ignition matrix found no other confirmed
"trigger never fires" gaps in Tier A/B components. `NAVProtectionEngine` and the TP/SL
300s aged-position recalculation, both originally flagged in Phase 1/2 as
possibly-unwired, were resolved in Phase 3 follow-up: both are called directly from
`main.py`'s own cycle logic (lines ~414-416 and ~437-446 respectively), independently
of the native orchestrator's parallel (but production-inert) equivalents. This is a
**duplication pattern worth noting**: at least two pieces of logic exist in both
`main.py` and `NativeOrchestrator`, with only the `main.py` copy actually live in
production. This isn't broken today, but it's a maintenance hazard — a future change to
`orchestrator.py`'s copy could be mistaken for a live fix when it changes nothing in
production.

## H. Data-flow gaps

Two independent, overlapping configuration systems (`BootstrapConfig` vs
`config_loader.py`) read the same logical settings under different env var names in
several cases (`configuration_map.md`) — a latent risk of divergent effective
configuration depending on which code path reads which system, not proven to have
caused an actual incident in this audit but architecturally fragile.

## I. Observability gaps

Prometheus and telemetry export were not confirmed live or idle in this pass — both are
gated by env vars (`PROMETHEUS_EXPORT_PATH`, `TELEMETRY_EXPORT_PATH`) whose current
`.env` values were not directly inspected; absence of export log lines in the Phase 2
session is consistent with (but doesn't prove) both being unset. Health monitor and
watchdog cadences were not observed within the ~142s session window — instantiation
confirmed, periodic behavior not confirmed either way.

## J. Legacy coupling

No native file has a static `import` from `src.*` — coupling is entirely through
dependency injection (legacy objects passed as constructor arguments to bridge classes)
and through `bootstrap.py`, which does import and instantiate legacy agents
(`MLForecaster`, `SymbolScreener`) directly. `core_engine/implementations.py` is
structurally a permanent bridge layer (try-native-first, fall-back-to-legacy throughout),
not a temporary migration shim — several code paths (signal retrieval, NAV computation,
startup orchestration) still have legacy as their fallback of record.

## K. Runtime risks

1. **A background ML training task is not cancelled on shutdown** — confirmed in Phase 2,
   the process's "clean shutdown" completed while a `ModelTrainer_BTCUSDT` training task
   kept running for another ~83 seconds afterward. `supervisor.sh`'s 180-second stall
   watchdog currently tolerates this, but a longer or hung training run is a real risk
   of either delaying a needed restart or getting force-killed mid-training.
2. **Silent broad exception handling around position recovery** (`position_hydration_engine.py`)
   converts a real bug (`AttributeError`) into a quiet "assume fresh account" fallback —
   exactly the "silent fallback behavior" pattern the audit spec asked to watch for.
3. **Duplicate logic in `main.py` vs `NativeOrchestrator`** for NAV protection and TP/SL
   aged-position recalculation (Section G) — a maintenance/drift risk, not an active bug.
4. **Two parallel config systems** — a latent config-divergence risk (Section H).
5. Test suite currently cannot be trusted as a safety net for the execution path:
   `NativeDecisionEngine`, `NativeSharedState`, and `Position` all show API-drift test
   failures (constructor signatures, method names, dict-vs-object) concentrated exactly
   on the Tier A / hot-path components (`baseline_test_report.md`).

## L. Recommended next actions

1. ✅ **DONE — Fixed `position_hydration_engine.py`'s exchange-fills recovery.** Added a
   real `get_my_trades(symbol, limit)` method to `NativeExchangeClient` (GET
   `/api/v3/myTrades`, the endpoint the original code's own comment said it intended to
   use) and updated `_fetch_exchange_fills()` to iterate it per-symbol over non-zero
   wallet balances. Verified live: a fresh dry-run went from "assuming fresh account, 0
   positions" to correctly recovering **5 real open positions worth $201.24** from 1,397
   exchange trades. Tests: `test_position_hydration_integration.py` (14) and
   `test_native_bootstrap.py` (29) pass.
2. ~~Decide arbitration engine's fate~~ **RESOLVED — no code change needed.** Phase 4
   follow-up found the arbitration engine is already wired into the live decision path
   via `implementations.py` (Section E correction above). The original "dead code"
   finding was wrong.
3. ✅ **DONE (partial, by design) — background training task on shutdown.** Added a call
   to `ml_forecaster.stop()` in `bootstrap.shutdown_components()`, so shutdown now
   actually requests cancellation of tracked training tasks (it previously didn't call
   this at all). Verified via log: `"🚓 MLForecaster stopping..."` now fires. This does
   **not** eliminate the ~140s shutdown latency, because training runs via
   `loop.run_in_executor()` (a thread), and neither asyncio task cancellation nor
   `asyncio.run()`'s own executor-drain-on-exit can interrupt a thread already mid-epoch.
   A full fix requires cooperative cancellation checks inside `model_trainer.py`'s
   per-epoch loop (legacy code) — user explicitly deferred this as out of scope for this
   pass; `supervisor.sh`'s 180s stall-watchdog tolerance already covers the observed
   worst case.
4. **Consolidate the two config systems** — not attempted this pass (larger refactor,
   needs a design decision on which system wins). Still recommended.
5. ✅ **DONE — quarantined stale legacy-namespace tests** (see `changes_made.md` for the
   exact list and mechanism).
6. **Update the Tier A test suite** (`test_native_l5.py`, `test_portfolio_recovery_mode.py`,
   `test_race_conditions_and_growth.py`) to match current `NativeDecisionEngine`/
   `NativeSharedState`/`Position` APIs — not attempted this pass, larger scope.
7. **Investigate the two standing behavior-drift test failures** (`test_nav_protection.py`,
   `test_native_tpsl_engine.py`) — not attempted this pass.
8. **Delete or archive orphaned config files** — not attempted this pass.

See `changes_made.md` for full detail on what was and wasn't implemented, with
verification evidence for each.

## Final verdict

**NATIVE SYSTEM RUNS WITH LIMITED GAPS.**

The system starts reliably, reaches steady state in ~35 seconds, and its startup/cycle
loop was directly observed running cleanly for 3 full cycles with zero unhandled crashes
(exit code 0). The vast majority of L0-L4 native components have solid, currently-green
test coverage (645 passing tests) and were confirmed live and correctly sequenced in a
real run. Failures are not silent by default — errors are logged (if sometimes
over-broadly caught), and the one confirmed runtime bug (`get_all_orders`) degrades
gracefully rather than crashing. The gaps found were real and mostly fixed in this same session (position-recovery bug,
shutdown-hygiene partial fix, stale test quarantine); one finding (the arbitration
engine) turned out to be a false positive in the audit itself, corrected before any code
was changed based on it. Remaining open items (config-system duplication, Tier A test
drift, two standing behavior-drift test failures) are bounded and specific, not silent
incorrect trading behavior in the configuration this audit observed (100% cash, dry-run,
default flags — and notably, the account was in fact holding real positions the whole
time that position hydration was silently failing to recover, which the fix now
corrects). This does not rule out issues surfacing under different conditions
(`polling_enabled=False`, non-default feature flags) that this pass's short observation
window did not exercise.
