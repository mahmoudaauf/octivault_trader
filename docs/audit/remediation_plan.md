# Remediation Plan — Native OctiVault Trader

Date: 2026-07-14. Prioritized per the standard order: safety violations first, then
PnL/counting correctness, then broken execution/exits, then non-starting components,
then wiring, then missing data/triggers, then capital/compounding, then health/metrics,
then performance, then dead code, then strategy calibration last.

Every item states **Owner class**: `WIRING` (native/execution team, code-only fix, no
new research) or `STRATEGY` (requires new signal research — new model, new backtest, or
retiring/replacing the current signal source — different timeline and skillset).

**Binding constraint on this entire plan** (repeated from `daily_target_assessment.md`
§7 because it applies here specifically): **no item below recommends loosening
`ConfFloor`, `PERSIST_GATE`, or any risk/regime gate, and no item recommends forcing
trade volume to approach the 5/day target.** The daily-target's own safety rules — no
forced trades, no threshold loosening without new evidence — apply to this remediation
plan's own recommendations, not only to the live system. Where an item touches a gate,
it is a correctness fix (e.g., fixing a mis-wired config key) never a loosening.

---

## Priority 1 — Safety violations

1. **[WIRING] Gate `paper_mode`/`paper_key` on all mutating and trade-history exchange
   calls.** `core_engine/native/exchange_client.py:397-484` — `place_order`,
   `cancel_order`, `get_order`, `get_my_trades` have zero paper-mode check; only
   `get_account()` (line 369) is simulated. Today the only thing preventing
   `--mode=paper-trade` from sending real signed orders to `https://api.binance.com` is
   `main.py:362`'s `if mode != "dry-run"` check — a single, narrow guard standing in for
   what should be an exchange-client-level safety boundary. Fix: make every mutating/
   trade-history method check a `paper_mode`/`paper_key` flag and route to a simulated
   response when set, independent of what `main.py`'s mode string happens to be. This is
   the highest-severity open item in this plan because it is a live blast-radius risk
   for anyone running or testing `--mode=paper-trade` without first re-reading this
   exact audit finding.
2. **[WIRING] `implementations.py:706-710`'s silent "arbitration engine is `None` →
   default-pass" fallback.** Not observed to have fired (arbitration engine is
   confirmed instantiated in production), but this is the same defect *family* as the
   already-fixed `position_hydration_engine.py` "assume fresh account" fallback — a
   safety-relevant gate silently defaulting to permissive behavior instead of failing
   loud. Fix: replace the silent default-pass with an explicit fail-closed path (block
   the trade, log at ERROR/CRITICAL) if `arbitration_engine` is ever `None` at call
   time, since a `None` arbitration engine is itself a startup/wiring bug that should
   never be silently tolerated in a live trading path.

## Priority 2 — PnL / counting correctness

3. **[WIRING] Enable or replicate `fill_tracker.py`'s fee-aware realized-PnL
   calculation under default config.** `polling_enabled=True` (default) routes fill
   handling through `polling_coordinator.py`, which has zero realized-PnL/commission
   logic, leaving `shared_state.metrics["realized_pnl"]` permanently `0.0`. This is read
   by `NAVAttributionEngine`, `ObjectiveFeedbackController`, and
   `AdaptiveCapitalEngine` — all three are currently operating on a dead signal. Fix:
   either flip the default to use `fill_tracker.py`, or port its `_commission_quote`
   currency-correct commission-conversion logic into `polling_coordinator.py`'s
   fill-handling path.
4. **[WIRING] Add a single canonical `compute_net_trade_pnl(entry_fill, exit_fill)`
   function** that subtracts entry fee + exit fee + observed slippage from gross
   proceeds, call it from `executor.py::_close_position` (which currently only logs an
   explicitly-labeled `gross_pnl` and persists nothing), and write the result to
   `shared_state.metrics["realized_pnl"]` and the trade journal. Without this, no
   "net-profitable trade" claim from this system's own state is verifiable without
   manual Binance `myTrades` reconciliation.
5. **[WIRING] Fix `position_hydration_engine.py`'s fee summation to check
   `commissionAsset`** the same way `fill_tracker.py`'s `_commission_quote` does
   (currently sums raw numeric `commission` regardless of denominating asset — a real
   currency-unit bug that silently corrupts post-restart `unrealized_pnl` for any
   position with a BNB-denominated historical fee).
6. **[WIRING] Verify/fix the `pnl_after_fees_usdt` telemetry-key contract** between
   `make_sell_decision` (which computes a local `pnl_after_fees` float,
   `implementations.py` around line 1030) and `main.py:385`'s
   `telemetry.get("pnl_after_fees_usdt", 0.0)` consumption. Flagged as a candidate
   mismatch, not confirmed broken — read the full function body past line 1030 to
   settle whether the key is actually set before treating this as fixed either way.
7. **[WITHDRAWN 2026-07-14 — audit correction, not a real defect.]** ~~Wire
   `evaluate_nav_protection()` into the orchestrator's periodic cycle~~. The original
   claim (zero production call sites, confirmed "via grep") was wrong: that grep was
   scoped only to `core_engine/native/` and missed `main.py:412-433`, which calls
   `evaluate_nav_protection()` every 60s against the live shared-state instance —
   same class of mistake as the earlier arbitration-engine false positive in this
   audit series. Verified directly this pass (not just re-read): invoking it against
   a realistic `NativeSharedState` produces `allow_tp_sl_adjustment=True` and
   populated `suggested_actions` under a drawdown scenario — it is live, not dead
   code. See `docs/audit/profit_and_compounding_assessment.md` §5 for the full
   correction. No code change was made or needed for this item.
8. **[WIRING] Delete or clearly deprecate `core_engine/native/config_loader.py`'s
   fee/TP/SL/compounding keys** (`EXIT_FEE_BPS`, `TAKE_PROFIT_PCT`, `STOP_LOSS_PCT`,
   `COMPOUNDING_ENABLED`) or wire them as aliases into `BootstrapConfig` with a startup
   warning if both are set and disagree. Confirmed this pass: `config_loader.py` is not
   called from any live path (only `_archive/` and tests reference it), so this is
   currently an "operator trap" (dead config that looks live) rather than an active
   dual-system disagreement — downgrade from the prior audit's "silent divergence"
   framing to this narrower but still real risk, and fix it before an operator wastes
   time tuning a dead knob.

## Priority 3 — Broken execution / exits

9. **[WIRING] Run one safely-gated real execution test** (after item 1 above is fixed)
   to observe stages 9-14 of `trade_lifecycle_map.md` (execution, fill confirmation,
   TP/SL, exit gate, realized-PnL recording, reconciliation) actually fire at least
   once. These stages are "wired by code evidence" but have **never been
   runtime-exercised in this audit's history** — not because of a known defect, but
   because no candidate has ever survived to reach them. This is not urgent relative to
   Priority 1/2, but it is a real gap: a live-capital decision should not rely on
   stages that have literally never been observed to execute correctly.
10. **[WIRING] Fix `capital_allocator.py`'s hardcoded `_compute_volatility_pct`
    placeholder** (returns `0.008` always, `capital_allocator.py:395`) — position sizing
    is not actually volatility-aware despite the code structure implying it is. Real
    (if currently unexercised) risk-sizing defect: volatile symbols are not sized down
    relative to calm ones.
11. **[WIRING, already fixed this session — verify]** `position_hydration_engine.py`'s
    `get_all_orders()` AttributeError (nonexistent method, silently fell back to
    "assume fresh account" on every restart) — fixed via `get_my_trades()` +
    pagination fix (commits in this branch's history). No action needed beyond
    confirming this stays fixed; listed here for completeness since it was the
    highest-severity defect found in the prior pass.

## Priority 4 — Non-starting / never-exercised components

12. **[WIRING] Confirm TP/SL engine is tracking the same live position set found by
    position hydration.** `trade_lifecycle_map.md` Stage 11 flags that the 5 real
    positions recovered by the hydration fix were confirmed in a separate check, not
    necessarily verified loaded into the *same* live session's `NativeSharedState` at
    that moment. Low effort, should be closed out as a direct follow-up to the
    hydration fix rather than left as an open assumption.

## Priority 5 — Wiring gaps (strategy/runtime integration)

13. **[WIRING, lowest-effort item toward any real positive-edge volume] Restart
    `carry_supervisor.sh`** (the funding-carry paper-trading keep-alive daemon).
    Currently stopped; `logs/carry_state.json` shows no activity for ~9 days, stalled
    at 2 of the required ≥30 forward-proof trades. This is pure infra/wiring work —
    the strategy's positive edge is already backtested (+1.22%/trade, 80% win, 361
    spot-hedgeable perps) and execution is already testnet-validated. Restarting the
    daemon costs nothing and is the fastest lever available toward accumulating
    evidence for the one strategy in this codebase with a proven positive edge.
14. **[STRATEGY, contingent on item 13's forward-proof gate passing] Integrate
    funding-carry execution into the supervised native runtime** (`bootstrap.py`/
    `NativeOrchestrator`), replacing its current fully-separate-process/script
    architecture (own env vars, arm file, kill-switch). Do this only after the ≥30-trade
    forward-proof gate is met net-positive — do not wire it into live capital before
    that gate closes, per the strategy's own documented caveats (survivorship bias,
    illiquid-alt slippage, liquidation/basis risk unmodeled). Even fully wired, expect
    an intermittent trickle of trades on high-funding-dislocation days, not a
    dependable daily-5 quota by itself.
15. **[STRATEGY] Produce a documented backtest for `statarb_discover.py`** before any
    wiring decision. Its "tested & DEAD" status is currently asserted only in memory
    (a one-line cross-reference in `funding-carry-edge-candidate.md`), with no on-disk
    backtest artifact in this repo to independently verify sample size or magnitude —
    unlike (a) and (b), which have `backtest_edge.py`/`edge_report.py` and
    `funding_carry_backtest.py`/`carry_ledger.jsonl` respectively. Low priority since
    the strategy is already believed dead, but the documentation gap itself should be
    closed so future audits aren't reasoning from an unverifiable one-liner.

## Priority 6 — Missing data / triggers

16. **[WIRING] Reduce symbol-discovery cold-start lag and WS reconnect churn.**
    `symbol_discovery.py` returned 0 symbols on the first scan and took a full extra
    cycle to populate 5 real holdings, forcing a fallback to 10 hardcoded
    `DEFAULT_SYMBOLS` and triggering 2 WebSocket reconnects in a 54-second window. Minor
    multiplier on missed opportunity during cold-start only; not evidenced to persist
    after warm-up. Low priority — fix if convenient, not urgent.

## Priority 7 — Capital / compounding

17. **[DONE 2026-07-14]** Connected `nav_protection.py`'s `protection_floor_usdt` to
    `daily_compounding.py`: `DailyCompoundingPolicy.sizing_nav()` now accepts an
    optional `protection_floor_usdt` kwarg and caps its result at
    `max(0, current_nav - protection_floor_usdt)` in addition to its existing
    daily-rollover cap; `capital_allocator.py::allocate_for_buy` reads
    `shared_state.nav_protection_state["protection_floor_usdt"]` (populated by
    `main.py`'s already-wired 60s `evaluate_nav_protection()` call, see item 7) and
    passes it through. Tests: `tests/test_daily_compounding.py` (4 new),
    `tests/test_capital_allocator_nav_protection.py` (new file, 2 tests) — all pass.
    `locked_profit_usdt` was intentionally left unconnected: `available_profit_to_risk_usdt`
    already folds it into a lower `protection_floor_usdt` inside `nav_protection.py`
    itself (PROFIT_LOCK mode raises the floor by the locked amount), so wiring
    `protection_floor_usdt` alone is sufficient — a second `locked_profit_usdt`
    connection would double-count the same protection.

## Priority 8 — Health / metrics / monitoring

18. **[DONE 2026-07-14]** Built `core_engine/native/daily_target_monitor.py` —
    `NativeDailyTargetMonitor`: per-UTC-day counters (signals generated/qualified/
    risk-approved/rejected with reasons, orders submitted, entries filled, trades
    closed), net-of-fee outcome classification (win/loss/breakeven/compoundable)
    built on items #3/#4's now-trustworthy `compute_net_trade_pnl`, the full
    progress-state vocabulary (`progress_state()`), an explicit
    `strategy_vs_wiring_summary()` split, UTC-day rollover with a JSONL history
    archive, and restart-safe JSON persistence (same pattern as
    `daily_compounding.py`). **Read-only by construction**: every `record_*` method
    is purely additive bookkeeping; there is no setter for any gate/threshold
    anywhere in the file, so the binding constraint is enforced structurally, not
    just as a policy note. Wired into `bootstrap.py` (new `BootstrapConfig` fields
    `daily_target_trades`/`daily_target_state_path`/`daily_target_history_path`),
    injected into `NativeExecutor` (records order-submitted/entry-filled/
    trade-closed) and exposed via `app_ctx["daily_target_monitor"]`, consumed in
    `implementations.py::DecisionEngineImpl.evaluate_signal` (records
    signal-qualified/decision-approved-or-rejected-with-reason — the single
    production choke point for all arbitration evaluations). Tests: 16 in
    `tests/test_daily_target_monitor.py`, plus wiring/regression tests in
    `test_native_bootstrap.py`, `test_native_l5.py`, `test_phase4_integration.py`
    — all pass, no regressions in the full suite (52 pre-existing/unrelated
    failures unchanged, 726 passing up from 685).

## Priority 9 — Performance

19. No performance defects were identified in this audit pass distinct from the
    correctness/wiring items already listed above (e.g., cold-start ~35s dominated by
    TensorFlow/Keras model loading is a known, accepted startup cost, not flagged as a
    defect). Nothing to list here beyond noting the category was considered and found
    empty this pass.

## Priority 10 — Dead code

20. **[DONE 2026-07-14 — labeled, not deleted]** `core_engine/native/decisions.py`'s
    `NativeDecisionEngine.decide()` and `core_engine/native/orchestrator.py`'s
    `NativeOrchestrator.run_loop()` both got a prominent docstring
    `.. important::` note stating they are not the production decision path and
    citing the exact confirmed live path instead. Chose labeling over deletion:
    both have real, non-trivial existing test coverage
    (`tests/test_native_l4.py`, `tests/test_native_l2.py`, and others construct
    `NativeDecisionEngine` directly), so deleting would be a larger, less
    reversible change than the safety/correctness risk (none — it's unreachable
    in production) justified. No behavior change; docstring-only edit, verified
    via `ast.parse` + full test suite (49/49 in the two directly-affected test
    files, no change to the broader 726-passing/52-pre-existing-failing count).
21. **[WIRING] `config_loader.py`'s dead keys** — see item 8 (listed there under
    PnL-correctness since the "operator trap" framing is the more urgent aspect of the
    same underlying dead-code fact; cross-referenced here for completeness under the
    Dead Code category).

## Priority 11 — Strategy calibration (last, and bounded by the safety constraint above)

22. **[STRATEGY] Retire or replace the legacy ML forecaster as the live signal
    source.** This is the dominant blocker to the 5-net-profitable-trades/day target
    (see `daily_target_assessment.md` §6 item 1) and it is listed last per the required
    priority order precisely because it is the one item that cannot be resolved by the
    native/wiring team — it requires new signal research: either a retrained model
    family independently re-backtested for positive net expectancy (not merely
    retrained on the same feature set, which the existing evidence gives no reason to
    expect will change the verdict), or a structural replacement with a different alpha
    thesis (funding-carry, once forward-proven, being the only current candidate; the
    memory file separately names stat-arb (dead per item 15), cross-exchange, and
    event-driven as other candidates never built out in this repo).
    **Explicitly excluded from this item, per the binding safety constraint**: raising
    or lowering `ConfFloor`, adjusting `PERSIST_GATE`'s bar count, or re-tuning
    `gate_2_confidence`/`gate_3_regime`/`gate_11_symbol_downtrend` thresholds is NOT a
    valid way to execute this item. The proven finding is that confidence does not
    rank-order outcomes for this model family — no threshold placement on a
    non-predictive score creates edge. Any change here must be justified by a fresh,
    independently-run backtest on the replacement/retrained signal showing positive net
    expectancy, not by an attempt to hit the 5/day count.

---

## Summary: WIRING vs STRATEGY item count

- **WIRING items (native team, no new research needed):** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  11 (already fixed, verify only), 12, 13, 16, 17, 18, 20, 21 — 17 of 22 items.
- **STRATEGY items (new signal research, different owner/timeline):** 14 (contingent
  wiring step, but gated on a strategy-side proof), 15, 22 — 3 of 22 items, plus item 14
  which is a hybrid (wiring work gated on a strategy-side forward-proof result).

The volume imbalance is itself a finding: most of what's broken in this codebase is
fixable by the native/execution team without new research. But the single largest
item — replacing or retiring the only live signal source — is strategy work with no
code-only shortcut, and it is the item that most directly determines whether "5
net-profitable trades/day" is reachable at all. Completing every wiring item without
addressing item 22 (or wiring in item 14 as a supplement) produces a system that
correctly, verifiably, and safely executes a proven-losing strategy — an improvement in
trustworthiness and safety, not in profitability.
