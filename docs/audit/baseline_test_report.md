# Baseline Test Report

**Command:** `.venv/bin/python3 -m pytest tests/ -q` (run against current dirty `phase-3/wiring` tree, no fixes applied)

**Result:** `66 failed, 645 passed, 2 xfailed, 12 errors in 52.87s` (exit code 1)

Read-only baseline — nothing below was fixed as part of this audit. Root-caused via a
representative sample per failure cluster (grep + traceback inspection), not an
exhaustive per-test debug.

## Collection errors (12) — test suite references a layer structure that no longer exists

All 12 collection errors are `ModuleNotFoundError` for `src.l0_core.layer_contracts`,
`src.l1_exchange.*`, `src.l3_portfolio.portfolio_target_size_enforcer`, `src.l4_execution.*`,
`src.l6_governance`, `src.l7_observability`, `src.l8_lifecycle.*`. Per `native_scope.md`,
the current `src/` tree only contains `l0_core`, `l3_portfolio`, `l5_strategy` —
`l1_exchange`, `l4_execution`, `l6_governance`, `l7_observability`, `l8_lifecycle` do not
exist anywhere in the repository. These tests target an even older layer structure than
either the current `src/` legacy code or `core_engine/native/` — **classification: test
drift, stale legacy tests, not reflecting any currently-live architecture.**

```
tests/layers/test_l1_exchange.py
tests/layers/test_l2_wallet.py
tests/layers/test_l4_execution.py
tests/layers/test_l6_governance.py
tests/layers/test_l7_observability.py
tests/test_dust_exit_candidate_selection.py
tests/test_insuff_bal_circuit_breaker.py
tests/test_live_order_recovery_guards.py
tests/test_portfolio_target_size_enforcer.py
tests/test_sell_finalize_idempotency.py
tests/test_strict_cap_count_tradable.py
tests/test_truth_audit_wallet_guard.py
```

## Failures by file (66 total)

| File | Failures | Sampled root cause | Classification |
|---|---|---|---|
| `test_layer_namespace.py` | 15 | `ModuleNotFoundError: No module named 'src.l1_exchange'` (and similar) | **Same as collection errors** — stale legacy layer-namespace tests |
| `test_native_l5.py` (NativeExecutor) | 11 | `ExecutionResult.status == TERMINAL` when test expects `SUCCESS`; error message `"price unavailable ... cannot size order"`; also `AttributeError`-style dedup-count mismatches | **Test/fixture drift from active development.** `executor.py` is one of the dirty files (115 lines changed this branch) — its price-fetch/validation path changed and the test mocks/stubs were not updated to match. Tier A / execution-path component — this is the most consequential cluster since it's on the hot path |
| `test_overextension_guard.py` | 7 | `assert not dec.allowed` fails — decision comes back `allowed=True` when the test expects it blocked; log shows `market_regime_detector not available`, `health_monitor not available` | **Test fixture gap** — the test's mock `app_ctx` doesn't wire `market_regime_detector`/`health_monitor`, so the guard silently no-ops instead of blocking. Whether this reflects a real "guard doesn't fail closed" risk in production (where these components ARE wired) needs a follow-up read, not confirmed as a production defect from this failure alone |
| `test_portfolio_recovery_mode.py` | 6 | `TypeError: __init__() got an unexpected keyword argument 'shared_state'`; `AttributeError: type object 'NativeDecisionEngine' has no attribute '_map_signal_sell_reason'` | **API drift.** `NativeDecisionEngine`'s constructor signature and internal method names changed; tests still call the old signature/method names |
| `test_layered_architecture.py` | 6 | Same `src.l*` namespace pattern | Same as collection errors — stale legacy tests |
| `test_self_healing_controller.py` | 4 | `ModuleNotFoundError: No module named 'src.l8_lifecycle'` | Same as collection errors — stale legacy tests |
| `test_race_conditions_and_growth.py` | 3 | `AttributeError: 'NativeSharedState' object has no attribute 'deduct_free_balance'`; `assert 'HOLD' == 'SELL_PROFIT'` | **API drift** (method renamed/removed on `NativeSharedState`) plus one **behavior drift** (profit-taking threshold logic no longer matches test expectation — needs follow-up to determine if this is an intentional strategy change or a regression) |
| `test_nav_truthfulness_and_capital_clamp.py` | 3 | `FileNotFoundError` reading `src/l4_execution/execution_manager.py` (doesn't exist); `ModuleNotFoundError: src.l0_core.shared_state` | Same as collection errors — stale legacy tests, plus one test that reads legacy source files directly by path (brittle test design, independent of the module-removal issue) |
| `test_native_health_regime.py` | 2 | Not sampled individually this pass | Needs follow-up |
| `test_consolidation_exception_fix.py` | 2 | Not sampled individually this pass | Needs follow-up |
| `tests/layers/test_l0_cross_cutting.py` | 2 | `ModuleNotFoundError` on l0 import path variant | Same as collection errors |
| `test_quant_terminal_layer.py` | 1 | `test_situation_state_classifies_dust_heavy` | Needs follow-up |
| `test_nav_protection.py` | 1 | `assert protection.protection_mode == 'FREEZE_BUY'` — actual value `'DEFENSIVE'` | **Behavior/naming drift**, standing (not part of this branch's dirty files) — either the protection-mode enum was renamed and the test wasn't updated, or the freeze-buy threshold logic changed. Genuine candidate defect, needs a source read of `nav_protection.py`'s protection-mode state machine before classifying further |
| `test_native_tpsl_engine.py` | 1 | `assert result == "TIME_FORCE_EXIT"` — actual `"SL_HIT"` | **Behavior drift** — TP/SL engine now resolves aged positions to `SL_HIT` before reaching the time-based force-exit branch; needs follow-up to confirm intentional vs regression |
| `test_native_polling_reconciliation.py` | 1 | `TypeError: 'Position' object is not subscriptable` — test does `state.positions["XRPUSDT"]["qty"]` | **API drift** — `Position` changed from dict-like to a dataclass/object; test wasn't updated |
| `test_native_l3.py` | 1 | Not sampled individually this pass | Needs follow-up |

## Aggregate classification

| Category | Count (approx.) | Meaning |
|---|---|---|
| Stale legacy layer-namespace tests (`src.l1_exchange`, `l4_execution`, `l6_governance`, `l7_observability`, `l8_lifecycle`, plus `l0_core.layer_contracts`/`l0_core.shared_state`) | 12 collection errors + ~27 of the 66 failures (`test_layer_namespace.py`, `test_layered_architecture.py`, most of `test_self_healing_controller.py`, part of `test_nav_truthfulness_and_capital_clamp.py`, `tests/layers/test_l0_cross_cutting.py`) | **Test drift — safe to either delete, quarantine, or rewrite against the current `core_engine/native/` architecture.** These do not indicate any defect in currently-live code; they test a layer structure that was fully removed at some earlier point and never cleaned up |
| API drift on Tier A components (`NativeDecisionEngine` constructor/methods, `NativeSharedState.deduct_free_balance`, `Position` dict→object) | ~20 (`test_native_l5.py`, `test_portfolio_recovery_mode.py`, part of `test_race_conditions_and_growth.py`, `test_native_polling_reconciliation.py`) | **Test coverage on the execution/risk path is currently unreliable.** Given `executor.py`, `decisions.py`, `capital_allocator.py`, `shared_state.py` are all dirty files on this branch, this is consistent with in-progress refactoring outpacing test updates — but it means the test suite cannot currently be trusted as a safety net for the hottest path in the system |
| Standing behavior drift, not tied to this branch's dirty files (`test_nav_protection.py`, `test_native_tpsl_engine.py`, one case in `test_race_conditions_and_growth.py`) | 3 | Candidate real defects or intentional-but-undocumented behavior changes — need targeted source review before remediation, flagged in `current_state_assessment.md` |
| Not yet individually classified | ~5 (`test_native_health_regime.py`, `test_consolidation_exception_fix.py`, `test_quant_terminal_layer.py`, `test_native_l3.py`, one `test_overextension_guard.py` variant) | Follow-up needed in a future pass |

## What passed

645 tests passed, 2 xfailed (expected failures). Notably, all of `test_native_bootstrap.py`
(29), `test_native_l0.py` (31), `test_native_l1.py` (30), `test_native_l2.py` (18),
`test_native_l4.py` (31), `test_native_app_context.py` (10), `test_native_compat.py` (15),
and `test_native_fill_tracker.py` (13) passed cleanly — the core L0-L4 native layer and
bootstrap wiring have solid, currently-green coverage. The failures concentrate in L5
(executor) and above, plus the fully-legacy layer-namespace tests.
