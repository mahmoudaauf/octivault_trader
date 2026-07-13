# Documentation Gap Analysis

Compares Phase 1-3 findings against existing architecture documentation
(`docs/architecture/`, `.LEGACY_TO_NATIVE_MAP.md`, `LEGACY_SIGNAL_INTEGRATION.md`,
`LEGACY_VS_CORE_ENGINE_INTEGRATION.md`). Docs are dated April 28 - May 7, 2026; current
HEAD is June 20, 2026, with substantial additional uncommitted work on top (July 14,
2026 observation date) — roughly 6-10 weeks of undocumented drift is expected and found.

This comparison is diagnostic, not prescriptive — it identifies divergence and lets you
decide whether code or docs need to change, per the audit's working rules.

## Arbitration Engine — the clearest, highest-confidence gap

- **Documented (legacy):** `docs/architecture/COMPREHENSIVE_SYSTEM_SUMMARY.md` describes
  `core/arbitration_engine.py` as doing "6-layer gating, signal evaluation, pass/fail
  logic," embedded in `MetaController`, central to the DECIDE phase
  ("3. DECIDES: 6-layer arbitration with regime-based gating").
  `docs/architecture/LAYER_INTEGRATION_PLAN.md` marks it "✅ Embedded in MetaController."
- **`.LEGACY_TO_NATIVE_MAP.md` (2026-05-06):** explicitly lists the native port of
  Arbitration Engine as **`❌ TODO`** — i.e., as of six weeks before this audit, the
  native version did not exist yet.
- **Current reality (this audit, 2026-07-14):** `core_engine/native/arbitration_engine.py`
  now exists, is instantiated in `bootstrap.py:841`, and is wired into `app_ctx`. The
  Phase 1-3 passes of this audit concluded it was never called, based on a static trace
  that checked `orchestrator.py`/`core_engine/native/decisions.py`/`executor.py` (all
  negative) and a live session with zero arbitration log lines. **That conclusion was
  wrong — corrected in Phase 4.** The actual production call is
  `core_engine/implementations.py::DecisionEngineImpl.make_buy_decision`/`make_sell_decision`
  → `evaluate_signal()` → `arbitration_engine.evaluate()`, reached via `main.py`'s façade
  `DecisionEngine`, not via `NativeOrchestrator`/`NativeDecisionEngine` at all. The Phase 2
  session's zero arbitration activity was simply because no signal reached the decision
  stage that session (upstream gating in `MLForecaster`), not because arbitration is
  unreachable.
- **Classification: Documented and implemented and wired — matches reality.** The TODO
  from `.LEGACY_TO_NATIVE_MAP.md` was in fact fully closed, both at the code level and
  the integration level. **This entry is left in this document, corrected in place,
  as a record of how the audit itself got something wrong and why** — a static trace
  that checks the "obvious" files (the ones an orchestrator-centric mental model
  suggests) missed the actual façade/implementations.py call site, and a live session
  with no qualifying activity looked like confirming evidence for "dead" when it was
  really just "not exercised by this path." No code was changed based on the original
  finding — see `current_state_assessment.md` Section E.

## Two config systems — undocumented divergence

- **Documented:** No architecture doc found that describes `core_engine/native/config_loader.py`
  as a distinct system from `BootstrapConfig`. `docs/architecture/LOGICAL_LAYERED_ARCHITECTURE.md`
  and related docs describe a single logical L0 config layer.
- **Current reality:** Two independently-parsed, overlapping env-var config systems
  coexist (`configuration_map.md`), with real naming collisions on live settings
  (`TP_PCT`/`TAKE_PROFIT_PCT`, `DAILY_COMPOUNDING_ENABLED`/`COMPOUNDING_ENABLED`, etc.).
- **Classification: Undocumented divergence, code likely outdated (or at minimum,
  needs consolidation).** Neither doc describes this as an intentional two-system design;
  it reads as incomplete migration residue (config_loader.py likely predates or was
  meant to be replaced by BootstrapConfig, or vice versa) that was never cleaned up.

## Legacy signal bridge — documented and confirmed accurate

- **Documented:** `LEGACY_SIGNAL_INTEGRATION.md` and `.LEGACY_TO_NATIVE_MAP.md` describe
  bridging legacy `MLForecaster`/`SymbolScreener`/`SignalManager` into the native pipeline
  via adapter classes.
- **Current reality:** `legacy_signal_adapter.py` and `signal_manager_bridge.py` do
  exactly this, confirmed wired in `bootstrap.py` and confirmed live at runtime (every
  cycle in the Phase 2 observation ran signal generation through this exact path).
- **Classification: Documented and fully implemented, matches reality.** This is the
  one major subsystem where documentation, static wiring, and runtime observation all
  agree cleanly.

## Layer naming — legacy docs describe a structure that no longer exists in `src/`

- **Documented:** `docs/architecture/LOGICAL_LAYERED_ARCHITECTURE.md` describes an L0-L8
  logical layering with modules under `core/` (not `src/l*_*` or `core_engine/native/`) —
  e.g. `core/meta_controller.py`, `core/signal_manager.py`, `core/arbitration_engine.py`.
  This appears to predate even the `src/l0_core`/`l3_portfolio`/`l5_strategy` structure
  found in the current repo.
- **Current reality:** `src/` only contains `l0_core`, `l3_portfolio`, `l5_strategy`.
  `core_engine/native/` uses its own L0-L8 labeling convention with entirely different
  module names and a `Native*` class-naming prefix. Neither matches the `core/*.py`
  paths in the oldest docs.
- **Classification: Documentation outdated (multiple generations behind).** This
  explains the baseline test suite's 12 collection errors and ~27 additional failures
  referencing `src.l1_exchange`, `src.l4_execution`, `src.l6_governance`,
  `src.l7_observability`, `src.l8_lifecycle` (`baseline_test_report.md`) — those tests
  were written against this even-older `core/`-era or an intermediate `src/l*`-era
  structure that has since been superseded twice over, and neither the tests nor these
  docs were updated to track the current `core_engine/native/` reality.

## Position hydration / exchange fills recovery — no dedicated doc found, but a real bug

- **Documented:** `docs/architecture/STATE_RECOVERY_SYSTEM.md` exists (not deep-read
  this pass) and presumably describes intended recovery behavior; not cross-checked
  line-by-line against `position_hydration_engine.py` in this audit.
- **Current reality:** `position_hydration_engine.py` calls
  `NativeExchangeClient.get_all_orders()`, a method that doesn't exist on that class —
  confirmed via runtime `AttributeError` in Phase 2.
- **Classification: Needs a follow-up read of `STATE_RECOVERY_SYSTEM.md` against the
  actual `exchange_client.py` method list** to determine whether this is a simple
  rename-drift bug (most likely, given the rest of the codebase's pattern of method
  renames outpacing callers) or a deeper design gap. Flagged here rather than fully
  resolved — out of this pass's depth budget.

## Root-level fix/status docs — informal changelog, not architecture

The dozens of root-level `.md` files (`.CAPITAL_ALLOCATION_FIX.md`,
`.ROOT_CAUSE_NO_TRADES.md`, `LEGACY_REUSABLE_COMPONENTS.md`, `LEGACY_TPSL_ANALYSIS.md`,
etc.) function as an informal, chronological changelog of prior fixes and investigations
rather than living architecture documentation. They were not systematically cross-checked
against current code in this pass — treat them as historical record, not a source of
truth for current wiring. Recommend they eventually move under `docs/archive/` (which
already holds hundreds of similar historical documents) if not already effectively
superseded.

## Summary table

| Area | Documented? | Implemented? | Wired? | Verdict |
|---|---|---|---|---|
| Legacy signal bridge (MLForecaster/SymbolScreener → native) | Yes | Yes | Yes | Matches reality |
| Arbitration engine (6-layer gating) | Yes (as central to DECIDE phase) | Yes (native port exists) | **Yes** (corrected in Phase 4 — original "not wired" finding was an audit error, see full entry above) | Matches reality |
| Native config system | No (docs assume one config layer) | Yes (two systems: BootstrapConfig, config_loader) | Both partially wired, diverging | Undocumented divergence, needs consolidation decision |
| L0-L8 layer naming under `core/` | Yes (oldest docs) | No — superseded twice (by `src/l*`, then `core_engine/native/`) | N/A | Documentation multiple generations outdated; drives ~40 stale test failures |
| Position hydration exchange-fills recovery | Presumably (STATE_RECOVERY_SYSTEM.md, not deep-read) | Partially — method call is broken | Broken at runtime | Needs follow-up doc-vs-code read |
| Daily compounding policy | No dedicated doc found | Yes | Yes | Implemented but undocumented — recent addition (untracked file this session) |
| Two-tier polling (gated coordinator vs legacy poller) | No dedicated doc found | Yes | Yes (coordinator is default) | Implemented but undocumented |
