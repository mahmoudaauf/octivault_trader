# Phase 8.3 — Stabilization & Hardening

**Started:** 2026-05-06
**Predecessor:** Phase 8.2.8 — bridge deletion landed, native is the only path
**Status:** ✅ Substantively complete — all six compat stubs replaced with native
impls (8.3.7-8.3.12), G5 acceptance gate satisfied + codified (G5 cleanup),
heartbeat wired into the orchestrator cycle (8.3.12-followup). Suite: **549/549
passing in 17.6s**. Remaining sub-steps (8.3.2 live testnet smoke, 8.3.6 perf
baseline, 8.3.13 Py 3.11) are optional follow-ups, not migration blockers.

---

## Context

Phase 8.2 delivered the native L0–L8 stack and tore the legacy
orchestrator out by the roots. The native suite is green (204 tests),
the offline smoke is clean, `main.py` defaults to native+compat. The
"migration" milestone is therefore **complete**.

But the system isn't *production-grade* yet. Several known gaps were
parked behind "land the migration first":

1. **Resource lifecycle** — `main.py` graceful shutdown calls
   `engines.shutdown()` but **does not** call
   `shutdown_components()` on the native bootstrap. Background polling
   tasks (`market_data._poll_loop`, `balance_sync._poll_loop`) and
   the exchange-client HTTP session can leak across signal-driven
   exits.

2. **Live testnet smoke (step 5b)** — `scripts/native_smoke.py --live`
   has never been run. We have offline coverage and unit coverage, but
   no end-to-end real-network signal-of-life.

3. **The 6 unmigrated façade keys** — currently null-stubs via
   `compat=True`. Fine for boot, but the 5 façade engines fall back
   to graceful-degrade for: `portfolio_manager`, `position_manager`,
   `tp_sl_engine`, `safety_order_manager`, `recovery_engine`,
   `watchdog`. Real implementations are needed before live trading.

4. **Observability surface** — `NativeTelemetry` exists with summary
   stats and percentiles, but nothing exports them. No JSON dump, no
   `/healthz`, no log rollup. Operators have no read path.

5. **Doc graveyard** — ~20 `PHASE_8_*.md` files at repo root, most
   stale (planning docs, status snapshots, completion reports). They
   confuse anyone arriving fresh.

6. **Pre-commit / CI** — the local `vulture` hook references a binary
   that isn't installed. `bash -c '… || true'` makes it harmless on
   exit, but we've been blanket-using `--no-verify` anyway. Time to
   audit and either fix or remove.

7. **Python 3.9 event-loop pollution in full-suite runs** — observed
   ~19 unrelated failures from asyncio loop teardown when `tests/` is
   run in one process. Native subset is clean. Fixable with
   per-test loop scoping or moving to 3.11.

8. **Performance baseline** — original Phase 8.2 roadmap targeted
   ~180-200ms cycles vs. ~300ms for the legacy bridge. We have **zero**
   measured numbers because the legacy bridge is gone. Native offline
   smoke shows ~1ms/cycle but that's against a stub client. We need
   real-network numbers.

---

## Sub-step priority matrix

| # | Title | Effort | Risk | Value |
|---|---|---|---|---|
| **8.3.1** | **Native shutdown wiring in `main.py`** | done — `c3fc3a2` | low | high — real leak |
| 8.3.2 | Live testnet smoke (parked 5b) | 30 min | low | high — first end-to-end signal |
| **8.3.3** | **Telemetry export — periodic atomic JSON snapshot** | done — this commit | low | medium |
| **8.3.4** | **Doc graveyard sweep → single `PHASE_8_FINAL.md`** | done — `7856404` + `913c543` | none | medium |
| **8.3.5** | **Pre-commit audit (vulture removed, ruff scoped, 10 lint fixes)** | done — `10c39c4` | none | low — DX |
| 8.3.6 | Real-network performance baseline (avg/p50/p95) | 1 h | low | medium |
| **8.3.7** | **Native `portfolio_manager` (read-only aggregator; replaces compat stub)** | done — this commit | medium | high — needed for live |
| **8.3.8** | **Native `position_manager` (read-only per-symbol accessor; replaces compat stub)** | done — this commit | medium | high |
| **8.3.9** | **Native `tp_sl_engine` (per-symbol target store + crossing detection)** | done — this commit | medium | high |
| **8.3.10** | **Native `safety_order_manager` (OCO intent store + best-effort TP-leg placement)** | done — this commit | medium | medium |
| 8.3.11 | **Native `recovery_engine` (in-process self-diagnosis: orphan OCO / stale prices / NAV drift / zero entry — with apply dispatcher)** | done — `8ec9a51` | medium | medium |
| 8.3.12 | **Native `watchdog` (heartbeat tracking + 5-detector anomaly sweep) — FINAL compat-stub replacement; G5 unlocked** | done — `5f37566` | low | medium |
| 8.3.12-followup | **Wire `NativeWatchdog.record_heartbeat()` into orchestrator cycle (5 wiring tests)** | done — `d0cf533` | low | medium |
| **G5 cleanup** | **Retire `core_engine.native.compat` plumbing — `compat: bool` kwarg becomes deprecated no-op everywhere; production code no longer imports `register_compat_stubs`. Module file kept (orphaned) per user override.** | done — `88787d4` | none | low |
| 8.3.13 | Python 3.9 → 3.11 migration | 1 day | medium | low — DX |

**Recommended sequence**: 8.3.1 → 8.3.4 → 8.3.5 → 8.3.6 → 8.3.2 (when
creds available) → 8.3.3 → 8.3.7 onwards.

The first four are **cheap** and unblock everything else. 8.3.7-8.3.12
are the genuine engineering work — building real implementations of
the six compat-stubbed components.

---

## Acceptance gates

| Gate | When | What | Status |
|---|---|---|---|
| **G1: clean shutdown** | after 8.3.1 | `python main.py --cycles=2 --no-native`, then SIGINT, exits 0 with no orphan tasks | ✅ closed (`c3fc3a2`) |
| **G2: live signal** | after 8.3.2 | live testnet smoke completes 60s with non-zero NAV | ⏸ deferred — needs creds; orthogonal to migration |
| **G3: observability** | after 8.3.3 | telemetry summary written to disk every N cycles, machine-readable | ✅ closed |
| **G4: clean repo** | after 8.3.4 | repo root has ≤3 `PHASE_*` docs | ✅ closed (`7856404` + `913c543`) |
| **G5: full parity** | after 8.3.7-8.3.12 | `compat=False` boots cleanly with full façade-engine coverage | ✅ closed — all 6 stubs replaced (8.3.7 → 8.3.12), heartbeat wired (8.3.12-followup), production code de-compatted (G5 cleanup, `88787d4`) |

**Migration-completeness gates (G1, G3, G4, G5) are all closed.** G2 is the
sole remaining gate and is gated on testnet credentials — not on code work.

---

## Out of scope for 8.3

- Multi-exchange support
- New strategy engines
- UI / dashboards
- Distributed deployment

These belong to Phase 9+.

---

**Owner:** @mauf
**Branch:** `phase-3/wiring`
**Last updated:** 2026-05-06 (post G5 cleanup, suite 549/549 green)

---

## Known follow-ups (post-migration, non-blocking)

These surfaced during 8.3 work but are explicitly **not** Phase 8.3 deliverables.
Tracked here so they don't get lost:

1. **`.claude/worktrees/` orchestrator hygiene** — two worktrees
   (`competent-yonath-ba6f13`, `frosty-bhaskara-651c14`) contain a
   `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` script that has **no `argparse`**,
   ignores `--mode` / `--duration`, and defaults to LIVE + 24h. Three
   zombie processes accumulated from these worktrees on 2026-05-06 (all
   running mainnet on a $72 wallet despite `--mode=paper-trade` flags;
   killed via SIGKILL after they ignored SIGTERM for 10s+). The script
   is **not** in the main tree (never committed to `phase-3/wiring`).
   **Mitigation**: either delete the worktrees or audit and fix the
   script if anyone still launches from there. Real entry point
   `main.py` is unaffected — it has proper `argparse`, signal handlers
   for SIGINT+SIGTERM, calls `shutdown_components()` on exit, and
   defaults to `paper-trade`.
2. **`SymbolScreener` convergence gating** — observed blocking 100% of
   proposals during the 2026-05-06 zombie run. Productivity issue, not a
   safety issue, but worth investigating before any real launch.
3. **`core_engine.native.compat` orphan module** — restored by user override
   in `49619d4`; production code no longer imports it but the file +
   `tests/test_native_compat.py` (15 self-contained tests) remain. Harmless
   dead code; can be deleted in a future cleanup pass when convenient.
