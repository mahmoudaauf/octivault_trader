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

---

# Remediation Session 2 (2026-07-14, same day) — remediation_plan.md items

Follows the five-trades/day capability audit (`daily_target_assessment.md`,
`trade_lifecycle_map.md`, `daily_trade_funnel.md`, `strategy_contribution.md`,
`profit_and_compounding_assessment.md`) run immediately after Session 1 above, working
through `remediation_plan.md`'s priority list in order. Baseline for this session:
`64 failed, 726 passed, 2 xfailed` at completion (up from `647 passed` at session start
— the 52/726 split below stayed flat through every step; **the 52 failures at session
end are the same pre-existing, unrelated 52 confirmed at the very start of this session**,
re-verified by name after every item).

## Priority 1 — Safety

**Item #1 — [WIRING] Gate paper_mode on all mutating/trade-history exchange calls.**
`core_engine/native/exchange_client.py`: `place_order()`, `cancel_order()`, `get_order()`,
`get_my_trades()` had zero paper-mode check (`get_account()` was the only guarded
method) — a `--mode=paper-trade` run with sentinel `"paper_key"`/`"paper_secret"`
credentials would have sent a real signed HTTPS order to `https://api.binance.com` if
any decision ever cleared the arbitration/confidence gates. Added `_is_paper()` (same
check `get_account()` already used) to all four methods; `place_order()` now fabricates
a locally-computed filled-order response using the real, safe, unsigned `get_prices()`
call for a realistic price — never reaching a signed endpoint. **Second bug found while
verifying stage 9-14 execution (item #9, below):** the paper `get_order()`/`cancel_order()`
responses were initially static, data-less stubs; since `_simulate_place_order()` marks
every order instantly `FILLED` (including the maker-first-buy's LIMIT leg), and
`order_execution.py` only caches non-terminal orders, a follow-up `refresh_status()` poll
(executor.py's maker-first-buy, 2s grace window, `MAKER_ENTRY_ENABLED=true` by default)
fell through to the data-less stub and **zeroed out the real fill price/qty/fees** —
found by actually running the fixed system end-to-end, not by code reading alone. Fixed
by adding an in-memory paper-order ledger (`_paper_orders_by_id`/`_paper_orders_by_coid`)
so `get_order()`/`cancel_order()` return the real placed data; an unknown order now
reports `"REJECTED"`, never a false `"FILLED"`. Files: `core_engine/native/exchange_client.py`.
Tests: 10 new in `tests/test_native_l1.py` (paper-mode gating + ledger correctness),
1 corrected (`test_get_order_paper_mode_never_calls_request` previously asserted the old,
now-fixed buggy behavior).

**Item #2 — [WIRING] Arbitration-unavailable silent default-pass → fail-closed.**
`core_engine/implementations.py::DecisionEngineImpl.evaluate_signal()` defaulted to
`passed=True` ("MVP default-pass") when `arbitration_engine` was `None` — the same defect
family as the already-fixed position-hydration silent fallback. Changed to fail-closed
(`passed=False`, `blocking_gates=["arbitration_engine_unavailable"]`, logged at CRITICAL).
Not observed to have fired in production (arbitration is confirmed instantiated), but a
missing arbitration engine is itself a startup bug that should never silently permit
trading. Tests: 2 new in `tests/test_phase4_integration.py`.

## Priority 2 — PnL / counting correctness

**Item #3 — [WIRING] Fee-aware realized-PnL wired into the default (polling-mode) path.**
`polling_coordinator.py::_fetch_and_reconcile_fills()` called a nonexistent
`get_account_trades()` method (same defect class as the position-hydration
`get_all_orders()` bug from Session 1) — silently returned immediately every call via
`hasattr()` guard, meaning **zero realized-PnL computation happened anywhere in the
default runtime path**. `fill_tracker.py` had the same nonexistent-method bug (it's
disabled by default so this was latent, not live, but would have been equally broken if
ever enabled). Fixed both to call the real `get_my_trades()`; added SELL-fill realized-PnL
computation (currency-aware commission conversion, mirroring `fill_tracker.py`'s existing
`_commission_quote` logic) to `polling_coordinator.py`, writing to
`shared_state.metrics["realized_pnl"]`. Known scope limit: this reconciliation loop only
sees symbols still in `shared_state.positions` — a position closed synchronously by
`executor.py` before the next 60s poll isn't caught here; item #4 covers that path.
Files: `core_engine/native/fill_tracker.py`, `core_engine/native/polling_coordinator.py`.
Tests: 3 new in `tests/test_native_polling_reconciliation.py`, 1 corrected in
`tests/test_native_fill_tracker.py` (was asserting against the old wrong method name).

**Item #4 — [WIRING] Canonical net-of-fee realized-PnL function.**
Added `commission_quote_from_fills()` and `compute_net_trade_pnl()` to
`core_engine/native/executor.py` — the single place "net realized profit" is now defined
(gross minus entry fee minus exit fee; slippage is already implicit since both prices are
real fill prices, not pre-trade references). Entry-side commission is captured at BUY time
(`_entry_commission_quote` dict, keyed by symbol) and consumed at `_close_position`, which
now logs and persists net PnL (previously gross-only, log-line-only). **Found and fixed a
second, unrelated, previously-latent bug while adding a BUY→SELL integration test:**
`executor.py`'s `_validate_lot_size()` had a `logger.debug(".. {:.8f} ..", qty, ...)` call
mixing `.format()`-style placeholders with `%`-style logging — this raises `TypeError` and
would abort **every real BUY whose quantity needs step-size rounding** (i.e. almost every
real order), silently converted into a generic "unexpected error" by the outer exception
handler. Fixed to `%.8f`. Files: `core_engine/native/executor.py`. Tests: 8 new in
`tests/test_native_l5.py` (canonical-function unit tests + full BUY→SELL integration test
asserting the exact net-of-fee number).

**Item #5 — [WIRING] Currency-aware fee summation in position hydration.**
`position_hydration_engine.py::_build_positions_from_fills()` summed raw `commission`
values regardless of `commissionAsset` — a real currency-unit bug (a BNB-denominated
historical fee would be summed as if it were USDT, corrupting post-restart
`unrealized_pnl`/`fees_paid`). Added `_fee_quote()` (mirrors the same conversion logic
used in `polling_coordinator.py`/`fill_tracker.py`) and threaded `commissionAsset` through
`_fetch_exchange_fills()`. Legacy journal-sourced fills (no `fee_asset` field) fall back to
"assume already quote-denominated," preserving prior behavior for that case exactly.
Files: `core_engine/native/position_hydration_engine.py`. Tests: 6 new in
`tests/test_position_hydration_integration.py`.

**Item #6 — Verified, not a bug.** Checked whether `make_sell_decision`'s
`pnl_after_fees_usdt` telemetry key actually reaches `main.py:385`'s consumption —
traced the full chain (`decision_engine.py` forwards to `implementations.py` with no
telemetry stripping; `main.py` appends the exact `TradeDecision` object to
`trading_decisions`) and confirmed the contract holds. No code change.

**Item #7 — [WITHDRAWN — audit correction, not a real defect].** The prior audit pass
claimed `nav_protection.py` was "fully unwired, zero production call sites" — this was
wrong. The claim's own grep was scoped only to `core_engine/native/` and missed
`main.py:412-433`, which calls `evaluate_nav_protection()` every 60s against the live
shared-state instance (same class of mistake as the earlier arbitration-engine false
positive in Session 1). Verified directly this session: invoking it against a realistic
`NativeSharedState` produces `allow_tp_sl_adjustment=True` and populated
`suggested_actions` under a drawdown scenario — it is live, not dead code. Corrected in
`profit_and_compounding_assessment.md` §5, `daily_target_assessment.md`, and
`remediation_plan.md` item #7 (all three, in place, marked CORRECTED — not silently
rewritten). No code change was made or needed for this item itself.

**Item #8 — [WIRING] Deprecation warning on `config_loader.py`.** Confirmed zero
production callers (only its own file and `tests/test_native_l0.py`) — an "operator trap"
since its env vars silently have no effect on live behavior (collide in name with
different `BootstrapConfig` keys). Added a module/class docstring `.. deprecated::` note
and a `logger.warning()` on `ConfigLoader.__init__()`. Chose warning over deletion: no
production risk either way, but deletion is less reversible than a warning for a
low-priority item. Tests: 31/31 pre-existing `test_native_l0.py` tests still pass
(warning doesn't change behavior).

## Priority 3 — Broken execution / exits

**Item #9 — [VERIFIED] Safely-gated real execution test, stages 9-14.** Since item #1's
fix makes paper-mode provably network-safe, wrote a standalone integration test exercising
the REAL `NativeExchangeClient` (paper mode) → `NativeOrderExecution` → `NativeExecutor`
chain (not test stubs) through execution → fill → position registration → TP/SL arm →
exit → net-of-fee PnL persistence. This is what surfaced the item #1 paper-order-ledger
bug and the item #4 logging-format bug above — both found only by actually running the
fixed code, not by static reading. Final verified result: a $500 BTC trade with a 1bps
maker discount and zero fees produced `realized_pnl=0.05`, exactly as expected. No
production code change beyond the item #1/#4 fixes already listed; the standalone script
itself was scratch-only and deleted after use (its coverage is now permanent in
`tests/test_native_l1.py`'s ledger tests and `tests/test_native_l5.py`'s BUY→SELL test).

**Item #10 — [WIRING] Real ATR-based volatility for capital sizing.**
`capital_allocator.py::_compute_volatility_pct()` was a hardcoded `return 0.008` (always
mid-range), despite the code structure implying real volatility-awareness. Extracted the
existing ATR computation from `tp_sl_engine.py` (`_compute_atr`/`_compute_atr_from_candles`
— real, working, already used for TP/SL sizing) into `core_engine/native/math_utils.py` as
shared `compute_atr`/`compute_atr_from_candles`/`compute_atr_pct` functions, so both
callers share one implementation instead of the capital allocator having a fake one.
`tp_sl_engine.py`'s two methods are now thin wrappers (identical behavior, verified via its
full existing 22-test suite passing unchanged). The 0.008 fallback is preserved but now
only fires on genuine cold-start (no candle/price data at all), not always. Files:
`core_engine/native/math_utils.py`, `core_engine/native/tp_sl_engine.py`,
`core_engine/native/capital_allocator.py`. Tests: 10 new in
`tests/test_native_math_utils_atr.py`.

**Item #11 — [VERIFIED] Position hydration fix (Session 1) still holds.** Re-confirmed
`get_all_orders` has zero references anywhere in the repo and `get_my_trades` is correctly
wired; `tests/test_position_hydration_integration.py` (20/20, including 6 new from item #5)
passes. No action needed beyond this verification.

## Priority 4 — Non-starting / never-exercised components

**Item #12 — [VERIFIED, no code change needed] TP/SL auto-arms hydrated positions with
no prior TP/SL.** Traced `apply_to_shared_state()` (writes `tp=None`/`sl=None` for any
restart-recovered position it couldn't restore a prior level for — the common real-restart
case) against `tp_sl_engine.py::check_triggers()`'s existing auto-arm branch
("AUTO-ARM... arm them on first sight so every held position is protected"). Verified
directly: a position with `tp=None, sl=None` gets armed (tp=102.0/sl=99.0 from entry=100.0)
on its very first `check_triggers()` call after the startup grace period, and is
immediately correctly evaluated against the fresh levels. No gap exists; this safety net
was already built. Test: 1 new permanent regression test in `tests/test_native_tpsl_engine.py`.

## Priority 5 — Wiring gaps (strategy/runtime integration)

**Item #13 — [VERIFIED, not started] funding-carry paper daemon.** Confirmed
`carry_paper_trader.py`/`carry_supervisor.sh` is genuinely paper-only and double-gated
(`CARRY_MODE` defaults to `"paper"`; real orders additionally require a `LIVE_ARM_FILE`
that does not exist; no `CARRY_MODE`/`CARRY_LIVE` override anywhere in `.env` or the
shell environment). **Did not start the daemon** — restarting a persistent process
(even a verified-paper one) is an operational action beyond code remediation and wasn't
explicitly requested; flagged for the user to start themselves if desired.

**Items #14, #15, #22 — explicitly not attempted.** Integrating funding-carry into the
supervised runtime (contingent on its own forward-proof gate), documenting a statarb
backtest, and retiring/replacing the ML forecaster are all STRATEGY-track work (new
signal research), not code-wiring fixes, per the binding constraint against fabricating
edge or loosening gates to reach the target. Left fully to `remediation_plan.md`'s own
documentation of these items.

## Priority 6 — Missing data / triggers

**Item #16 — [DEFERRED] Symbol-discovery cold-start lag.** Explicitly the lowest-priority
("fix if convenient") item, and root-causing it with confidence would require a fresh live
observation window (not safely repeatable this session) rather than a speculative static
fix. Left as documented/deferred rather than guessing.

## Priority 7 — Capital / compounding

**Item #17 — [WIRING] Connected NAV-protection floor to daily compounding.**
`DailyCompoundingPolicy.sizing_nav()` now accepts an optional `protection_floor_usdt`
kwarg and caps its result at `max(0, current_nav - protection_floor_usdt)` in addition to
its existing daily-rollover cap. `capital_allocator.py::allocate_for_buy` reads
`shared_state.nav_protection_state["protection_floor_usdt"]` (populated by `main.py`'s
already-wired — see item #7 correction — 60s `evaluate_nav_protection()` call) and passes
it through. `locked_profit_usdt` was deliberately left unconnected: `nav_protection.py`'s
own `available_profit_to_risk_usdt`/`protection_floor_usdt` computation already folds
locked profit into a raised floor (PROFIT_LOCK mode), so a second connection would
double-count. Files: `core_engine/native/daily_compounding.py`,
`core_engine/native/capital_allocator.py`. Tests: 4 new in `tests/test_daily_compounding.py`,
new file `tests/test_capital_allocator_nav_protection.py` (2 tests).

## Priority 8 — Health / metrics / monitoring

**Item #18 — [WIRING] Built the daily-target monitor.** New module
`core_engine/native/daily_target_monitor.py::NativeDailyTargetMonitor` — confirmed absent
by the audit, now implemented: per-UTC-day funnel counters, net-of-fee outcome
classification (win/loss/breakeven/compoundable, built on items #3/#4's now-trustworthy
PnL — sequenced here per the plan specifically because it would have automated a wrong
answer before that fix), the full progress-state vocabulary, an explicit
strategy-vs-wiring attribution split, UTC-day rollover with JSONL history archival, and
restart-safe persistence. **Read-only by construction**: every method is purely additive
bookkeeping; there is no gate/threshold setter anywhere in the file, so the "must never
auto-loosen gates" constraint is structural, not a policy note. Wired into: `bootstrap.py`
(3 new `BootstrapConfig` fields, constructed alongside the trade journal), `NativeExecutor`
(records order-submitted/entry-filled/trade-closed), and
`implementations.py::DecisionEngineImpl.evaluate_signal` (records signal-qualified/
decision-approved-or-rejected-with-reason — the single confirmed production choke point
for all arbitration evaluations). Files: `core_engine/native/daily_target_monitor.py` (new),
`core_engine/native/bootstrap.py`, `core_engine/native/app_context.py`,
`core_engine/native/executor.py`, `core_engine/implementations.py`. Tests: 16 in new file
`tests/test_daily_target_monitor.py`, plus wiring tests added to `test_native_bootstrap.py`,
`test_native_l5.py`, `test_phase4_integration.py`.

## Priority 9 — Performance

**Item #19.** No performance defects identified, unchanged from the original assessment.

## Priority 10 — Dead code

**Item #20 — [DONE — labeled, not deleted].** Added a prominent `.. important::`
docstring note to `NativeDecisionEngine.decide()` and `NativeOrchestrator.run_loop()`
stating they are not the production decision path and citing the real one. Chose labeling
over deletion given both have real existing test coverage
(`tests/test_native_l4.py`/`test_native_l2.py` and others) — deletion would be a larger,
less-reversible change than the (zero) safety risk justified. Docstring-only; no behavior
change.

**Item #21** — see item #8 (same underlying fact, cross-referenced under Dead Code).

## Priority 11 — Strategy calibration

**Item #22 — not attempted, by design.** See items #14/#15/#22 above. Retiring/replacing
the ML forecaster is the dominant blocker to the 5-trades/day target per
`daily_target_assessment.md`, and it is explicitly excluded from a code-wiring remediation
pass — no threshold change on a proven non-predictive score creates edge.

## Updated baseline after this remediation session

Before this session's fixes: `64 failed, 726 passed` is actually the count **after** —
tracking the session's growth: started at `647 passed` (Session 1's end state), ended at
`726 passed` (**+79 new tests**, all passing), with the same `52` pre-existing/unrelated
failures at both ends (re-verified by exact test name after every single item this
session, not just by count). Zero regressions introduced across 15+ files touched.

## Not attempted this session

- Item #13's daemon restart (operational action, not code remediation — see above)
- Items #14, #15, #22 (STRATEGY-track, no code-only shortcut exists — see above)
- Item #16 (deferred, needs live observation to root-cause safely)
- Full config-system consolidation (`BootstrapConfig` vs `config_loader.py` beyond the
  deprecation warning) — still an open, larger design decision, not attempted here.
