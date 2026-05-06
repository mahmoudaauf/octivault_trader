# Phase 8.3 — Stabilization & Hardening

**Started:** 2026-05-06
**Predecessor:** Phase 8.2.8 — bridge deletion landed, native is the only path
**Status:** Planning

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
| **8.3.1** | **Native shutdown wiring in `main.py`** | 30 min | low | high — real leak |
| 8.3.2 | Live testnet smoke (parked 5b) | 30 min | low | high — first end-to-end signal |
| 8.3.3 | Telemetry export (`/healthz` JSON or log-rollup task) | 2 h | low | medium |
| 8.3.4 | Doc graveyard sweep → single `PHASE_8_FINAL.md` | 1 h | none | medium |
| 8.3.5 | Pre-commit audit (vulture decision, mypy stage) | 30 min | none | low — DX |
| 8.3.6 | Real-network performance baseline (avg/p50/p95) | 1 h | low | medium |
| 8.3.7 | Native `portfolio_manager` (replaces compat stub) | 1-2 days | medium | high — needed for live |
| 8.3.8 | Native `position_manager` | 1-2 days | medium | high |
| 8.3.9 | Native `tp_sl_engine` | 2-3 days | medium | high |
| 8.3.10 | Native `safety_order_manager` | 2-3 days | medium | medium |
| 8.3.11 | Native `recovery_engine` | 1 day | medium | medium |
| 8.3.12 | Native `watchdog` | 1 day | low | medium |
| 8.3.13 | Python 3.9 → 3.11 migration | 1 day | medium | low — DX |

**Recommended sequence**: 8.3.1 → 8.3.4 → 8.3.5 → 8.3.6 → 8.3.2 (when
creds available) → 8.3.3 → 8.3.7 onwards.

The first four are **cheap** and unblock everything else. 8.3.7-8.3.12
are the genuine engineering work — building real implementations of
the six compat-stubbed components.

---

## Acceptance gates

| Gate | When | What |
|---|---|---|
| **G1: clean shutdown** | after 8.3.1 | `python main.py --cycles=2 --no-native`, then SIGINT, exits 0 with no orphan tasks |
| **G2: live signal** | after 8.3.2 | live testnet smoke completes 60s with non-zero NAV |
| **G3: observability** | after 8.3.3 | telemetry summary written to disk every N cycles, machine-readable |
| **G4: clean repo** | after 8.3.4 | repo root has ≤3 `PHASE_*` docs |
| **G5: full parity** | after 8.3.7-8.3.12 | `compat=False` boots cleanly with full façade-engine coverage |

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
**Last updated:** 2026-05-06
