# Archived Phase 8 documentation — 2026-05-06

Historical planning, status, and per-sub-phase completion notes from the
Phase 8.1 (production bridge) and Phase 8.2.x (native L0–L8 migration) work.

These files are kept for the audit trail only. They reflect the state of the
codebase at the time they were written; **most claims about file paths,
APIs, and "next steps" are now stale** (the bridge has been deleted, the
legacy orchestrator file is gone, native is the default boot path).

For the current truth, see the three active docs at the repo root:

| Doc | Purpose |
| --- | ------- |
| `PHASE_8_3_PLAN.md` | Active Stabilization & Hardening plan |
| `PHASE_8_2_8_PREP.md` | Bridge-removal prep (closed; left for context) |
| `PHASE_8_2_8_TRIAGE.md` | Per-key compat-stub triage referenced by `core_engine/integration.py` |

## Why these were archived

Per acceptance gate **G4** of `PHASE_8_3_PLAN.md`: *"repo root has ≤3
PHASE_* docs."* Twenty-two had accumulated. Nineteen were status snapshots,
duplicate executive summaries, or per-sub-phase completion reports whose
information is captured in git history and in the keeper docs above.

## Files

- `PHASE_8_2_1_COMPLETION.md` — L0 (NativeSharedState) recap
- `PHASE_8_2_1_L0_NATIVE_SPEC.md` — L0 spec
- `PHASE_8_2_2_L1_NATIVE_SPEC.md` — L1 (NativeExchangeClient) spec
- `PHASE_8_2_3_PLUS_L2_L8_PLAN.md` — Combined L2–L8 plan
- `PHASE_8_2_9_COMPLETION.md` — L8 lint/observability close-out
- `PHASE_8_2_NATIVE_MIGRATION_ROADMAP.md` — Original native migration roadmap
- `PHASE_8_BRIDGE_VALIDATION.md` — Bridge acceptance test results
- `PHASE_8_CODE_REVIEW.md` — Mid-migration code review
- `PHASE_8_DECISION_TREE.md` — "What to do next" decision matrix
- `PHASE_8_EQUIVALENCE_TEST_RESULTS.md` — Bridge ↔ legacy equivalence tests
- `PHASE_8_EXECUTIVE_SUMMARY.md` — Executive overview snapshot
- `PHASE_8_MASTER_INDEX.md` — Index of all Phase 8 docs (now superseded)
- `PHASE_8_PRODUCTION_WIRING_PLAN.md` — Phase 8.1 wiring plan
- `PHASE_8_PROGRESS_DASHBOARD.md` — Status dashboard
- `PHASE_8_QUICK_REFERENCE.md` — Quick reference (superseded by root `QUICK_REFERENCE.md`)
- `PHASE_8_SESSION_COMPLETE.md` — Per-session completion note
- `PHASE_8_STATUS.md` — Status checkpoint
- `PHASE_8_SUMMARY.md` — Phase 8.1 summary
- `PHASE_8_WHATS_NEXT.md` — "What to do next" snapshot
