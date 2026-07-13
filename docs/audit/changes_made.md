# Changes Made — Remediation Phase

Follows the diagnostic-only Phase 0-3 audit (`current_state_assessment.md` etc.). Every
change below was made with explicit user sign-off on the specific tradeoff involved, per
the audit's remediation rules. No live trades were placed; all verification used
`--mode=dry-run` against real (non-testnet) credentials.

## 1. Fixed: position hydration exchange-fills recovery

**Observed problem:** `position_hydration_engine.py::_fetch_exchange_fills()` called
`self._exchange.get_all_orders()`, a method that doesn't exist on `NativeExchangeClient`
— confirmed via a live `AttributeError` on every process startup (`docs/audit/runtime_timeline.md`
Finding #3). The bug was silently caught and degraded to "assuming fresh account."

**Root cause:** `NativeExchangeClient` (a deliberately minimal ~400-line rewrite per its
own docstring, "Public surface (8 methods)") never implemented a trade-history endpoint.
The calling code's own comment ("Typically: GET /myTrades with limit=500 ... or similar
endpoint") shows the original author knew the intended endpoint but never wired it.

**Fix:**
- `core_engine/native/exchange_client.py`: added `EP_MY_TRADES = "/api/v3/myTrades"` and
  a new `get_my_trades(symbol, *, limit=500)` method (signed GET, matching the existing
  `get_order`/`cancel_order` pattern), plus updated the class docstring's method list.
- `core_engine/native/position_hydration_engine.py`: `_fetch_exchange_fills()` now takes
  `balances` (already available at its one call site), derives candidate symbols from
  every non-zero non-USDT balance (same `f"{asset}USDT"` convention already used by
  `_merge_balance_holdings()` in the same file), calls `get_my_trades()` per symbol, and
  maps Binance's trade-response fields (`isBuyer`, `qty`, `price`, `commission`, `time`
  in ms) into the fill-dict shape `_build_positions_from_fills()` already expects.

**Files changed:** `core_engine/native/exchange_client.py`,
`core_engine/native/position_hydration_engine.py`

**Risk:** Low. Additive (new method, no existing method signatures changed) plus a
targeted fix to a function that was already non-functional (always returned `[]` via the
caught exception). No behavior that was previously working could regress.

**Tests:** `tests/test_position_hydration_integration.py` (14 tests) and
`tests/test_native_bootstrap.py` (29 tests) — all pass.

**Verification result:** Ran `python3 main.py --mode=dry-run --cycles=1 --interval=5
--duration=30s` before and after. Before: `Failed to fetch exchange fills:
'NativeExchangeClient' object has no attribute 'get_all_orders'` → `Hydration complete: 0
positions`. After: no error, `Found 1397 fills from exchange` → `Hydration complete: 5
positions, $201.24 value, 1 profitable, 4 losing`. **The account genuinely holds 5 open
positions that every previous restart was silently failing to recover.** This was the
single highest-confidence, highest-impact finding of the entire audit.

## 2. Fixed (partial, by design): background training task not cancelled on shutdown

**Observed problem:** A background `ModelTrainer_BTCUSDT` training task, queued by
`agents/ml_forecaster.py` during startup, kept running ~83 seconds after the process
logged "Clean shutdown complete" — confirmed in `docs/audit/runtime_timeline.md` Finding
#1 and `docs/audit/runtime_task_inventory.md`.

**Root cause:** `MLForecaster.stop()` (which cancels tracked tasks in
`self._train_tasks`) already existed but was never called anywhere in the shutdown
chain. `core_engine/native/bootstrap.py::shutdown_components()` stopped market data,
balance sync, and fill tracker, but not `components.ml_forecaster`.

**Fix:** Added a call to `ml_forecaster.stop()` in `shutdown_components()`, wrapped in
the same defensive try/except pattern as the other component-stop calls.

**Files changed:** `core_engine/native/bootstrap.py`

**Known limitation (explicitly accepted by user, not fixed):** Training runs via
`loop.run_in_executor()` (a thread pool). Neither `asyncio.Task.cancel()` on the
awaiting coroutine nor `asyncio.run()`'s own executor-drain-on-exit can interrupt a
thread already executing a training epoch — cancellation is requested but not
cooperative. Verified: shutdown latency was unchanged (~144s) after this fix, but the
`stop()` call now correctly fires (`"🚓 MLForecaster stopping..."` now appears in the
log at shutdown, which it never did before). A full fix requires cooperative
cancellation checks inside `src/l5_strategy/model_trainer.py`'s per-epoch loop (legacy
code) — user explicitly deferred this, judging it out of scope for a native-solution
audit and higher-risk to touch without dedicated testing. `supervisor.sh`'s 180s
stall-watchdog tolerance already covers the observed worst case.

**Tests:** No dedicated test exists for shutdown task-cancellation; verified via direct
runtime observation (log timestamps) instead.

## 3. No change: arbitration engine — audit correction, not a code fix

**What happened:** Phase 1-3 of this audit concluded `NativeArbitrationEngine` was
instantiated but never invoked ("dead wiring"), based on a static trace of
`orchestrator.py`/`core_engine/native/decisions.py`/`executor.py` and a live session with
zero arbitration-related log lines. The user approved "wire it into the decision path
now" based on that finding. Before implementing, a Phase 4 re-verification of the actual
call chain found the conclusion was **wrong**: `core_engine/implementations.py::DecisionEngineImpl.make_buy_decision`/`make_sell_decision`
already calls `arbitration_engine.evaluate()` via `evaluate_signal()`, and this — not
`NativeDecisionEngine.decide()` — is the real production decision path (`main.py` → the
façade `DecisionEngine` → `implementations.py`). The Phase 2 session's silence was
because zero signals ever reached the decision stage that session, not because
arbitration is unreachable.

**Action taken:** Reported the correction to the user before writing any code. User
confirmed no change needed. Corrected `docs/audit/component_inventory.md`,
`feature_ignition_matrix.md`, `data_flow_map.md`, `current_state_assessment.md`, and
`documentation_gap_analysis.md` in place (not deleted — corrections are visible inline,
each marked "CORRECTED") to reflect the accurate wiring status and document the audit's
own error transparently.

**Files changed:** documentation only, no source code.

## 4. Quarantined: 12 stale legacy-namespace test files

**Observed problem:** 12 test files fail at collection with `ModuleNotFoundError` for a
layer structure (`src.l1_exchange`, `src.l4_execution`, `src.l6_governance`,
`src.l7_observability`, `src.l8_lifecycle`) that doesn't exist anywhere in the current
repo — confirmed in `docs/audit/baseline_test_report.md`. This is a different, older
structure than either the current `src/{l0_core,l3_portfolio,l5_strategy}` or
`core_engine/native/`.

**User consultation:** 4 of the 12 have safety-critical names (`test_insuff_bal_circuit_breaker.py`,
`test_live_order_recovery_guards.py`, `test_sell_finalize_idempotency.py`,
`test_truth_audit_wallet_guard.py`). Flagged this explicitly before acting, since
quarantining them means giving up whatever safety-net coverage they represent. User
confirmed quarantining all 12, on the basis that these tests have been unable to run at
all (collection error, not a passing-then-broken test) for as long as the underlying
`src.l1_exchange`/`src.l4_execution` modules have been absent, and that equivalent
safety-relevant logic now lives in and is covered by `core_engine/native/executor.py` /
`capital_allocator.py` via `test_native_l5.py` and others.

**Mechanism:** Added a `collect_ignore` list to `tests/conftest.py`, with a comment
explaining the rationale and pointing to `baseline_test_report.md`. This is reversible
(delete the list) and non-destructive (files are untouched on disk, git history intact)
— it only stops pytest from attempting to import them.

**Scope discipline:** Deliberately limited to the 12 files that fail at **collection**
(the entire file cannot be imported — zero ambiguity that quarantining loses no currently
passing test). Two other files with the same root-cause pattern
(`test_layer_namespace.py`, `test_layered_architecture.py`) were explicitly **not**
quarantined, because their failure output shows a mix of passing (`.`) and failing (`F`)
cases — meaning some of their tests exercise something other than the missing legacy
package and quarantining the whole file risked hiding real, currently-passing coverage
without individually reading each parametrized case. Left as a follow-up recommendation
instead.

**Files changed:** `tests/conftest.py`

**Verification result:** Before: `66 failed, 645 passed, 2 xfailed, 2 warnings, 12 errors
in 52.87s`. After: `66 failed, 645 passed, 2 xfailed, 2 warnings in 56.49s` — confirms
the 12 quarantined files were purely collection-error noise (already excluded from the
"66 failed" count even before quarantine) and nothing else changed. The 66 known
failures remain fully visible, not hidden.

## 5. Fixed: standing behavior-drift test failures (both root-caused to test drift, not production bugs)

**`test_nav_protection.py::test_freeze_buy_blocks_new_entries`** — expected
`protection_mode == "FREEZE_BUY"` but got `"DEFENSIVE"`. Root cause: the test's NAV
scenario (session anchor 97.0 → 94.0, a ~3.09% drawdown) never crossed the
`drawdown_freeze_buy_pct` threshold, which git history shows was deliberately lowered
from 5%→4% in commit `66fd382c` ("fix: close all 8 system gaps") — and the test's
original 3.09%-drawdown scenario wouldn't have crossed even the older 5% threshold
either. **Fix:** changed the NAV drop to 97.0 → 93.0 (~4.12% drawdown), which correctly
crosses the current 4% FREEZE_BUY threshold. No production code changed — the threshold
itself is intentional, recent, deliberate calibration.

**`test_native_tpsl_engine.py::test_force_exit_triggers_on_aged_position_object`** —
expected `"TIME_FORCE_EXIT"` but got `"SL_HIT"`. Root cause: the test aged the position
2 hours and its own docstring claimed "after 1.5h," but `_AGE_FORCE_EXIT_SEC` defaults to
3 hours (`TPSL_FORCE_EXIT_H` env var, explicitly commented "force exit after rotator
immunity (2h) expires") — so at 2h old, the time-based force-exit layer correctly didn't
fire, and control fell through to the static SL check, which correctly fired instead
since the test's price was also below the SL level. **Fix:** aged the test position to
3.5 hours (past the 3h default) so the intended layer-2 (time-force-exit) branch, which
is checked before layer-1 (static SL) in `check_triggers()`, actually fires. No
production code changed.

**Files changed:** `tests/test_nav_protection.py`, `tests/test_native_tpsl_engine.py`

**Risk:** None — test-only changes, no production code touched, both root causes
confirmed against actual (and in the NAV case, git-history-confirmed intentional)
threshold values rather than assumed.

**Tests:** `tests/test_nav_protection.py` (11/11 pass), `tests/test_native_tpsl_engine.py`
(22/22 pass).

## 6. No change: orphaned config files — user chose to leave all three alone

Comprehensively re-confirmed reference counts before asking: `config/STRATEGY_OPTIMIZATION_v2.py`
has zero references anywhere in the repo; `config/EV_ALIGNMENT_CONFIG.py` is referenced
only as a static layer-boundary label string in `scripts/check_layer_imports.py`, never
actually imported; `strategy_validation.py` (untracked, new this branch) is referenced
only by its own test. Presented each individually (deletion is destructive) and the user
chose "leave it alone" for all three — `EV_ALIGNMENT_CONFIG.py` may still be worth wiring
in later rather than being truly abandoned, `STRATEGY_OPTIMIZATION_v2.py` is harmless
dead weight not worth the deletion in this pass, and `strategy_validation.py` is
in-progress work not this audit's call to remove.

## Updated baseline after this remediation session

Before any fixes: `66 failed, 645 passed, 2 xfailed, 2 warnings, 12 errors in 52.87s`.
After: `64 failed, 647 passed, 2 xfailed, 2 warnings in 53.36s` — 12 collection errors
eliminated (quarantined, not hidden — visible in `tests/conftest.py`'s `collect_ignore`
with rationale), 2 fewer failures (both root-caused and fixed as test drift), 2 more
tests passing. All remaining 64 failures are pre-existing and documented in
`baseline_test_report.md`'s classification table (mostly stale `src.l*` namespace tests
and Tier A API-signature drift on `NativeDecisionEngine`/`NativeSharedState`/`Position`).

## Not attempted this pass (see `current_state_assessment.md` Section L)

- Consolidating the two parallel config systems (`BootstrapConfig` vs `config_loader.py`)
- Updating the Tier A test suite (`test_native_l5.py`, `test_portfolio_recovery_mode.py`,
  `test_race_conditions_and_growth.py`) for current `NativeDecisionEngine`/
  `NativeSharedState`/`Position` API signatures
- Rewriting `test_layer_namespace.py`/`test_layered_architecture.py` against current
  architecture (flagged, not quarantined — see item 4 above)

These remain open, prioritized recommendations, not silently dropped.
