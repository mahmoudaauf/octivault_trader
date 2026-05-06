# Phase 8 — Production Wiring Plan

**Goal:** Connect the 5 façade engines to real L0–L8 components so the
runtime loop produces real telemetry (NAV, signals, decisions, executions)
instead of mock zeros.

**Branch:** `phase-3/wiring`
**Date opened:** 2026-05-06

---

## Background

After Phase 7 Steps 2–5, `main.py` is a clean façade-only entry point that
runs a 5-phase loop (READ→UNDERSTAND→DECIDE→EXECUTE→RECOVER) through the
five engines. However, every cycle currently emits:

```
nav=0.00  sigs=0  dec=0  exe=0
⚠️ exchange_client / portfolio_manager / signal_manager / health_monitor not available
```

Reason: `core_engine/integration.py::create_app_context()` returns an empty
dict — every L0–L8 import is commented out. The engines fall back to mock
behaviour (graceful degradation, by design).

The **legacy** `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` (3,306 lines) already
constructs all ~50 components correctly. Phase 8 reuses that construction
logic instead of duplicating it.

---

## Strategy: "Adopt, then Refactor"

Two-tier approach:

1. **8.1 — Bridge (this PR)**
   Wrap the legacy orchestrator: instantiate it, run its
   `initialize_components()`, then expose its `self.<component>` attributes
   as `app_ctx[<key>]` keys. The 5 engines immediately get real components.

2. **8.2 — Native (later PR)**
   Slowly migrate construction into `core_engine/integration.py`
   layer-by-layer (L0 → L8), one component at a time. The bridge becomes
   a thin shim until it can be removed.

Phase 8.1 is what unlocks production telemetry. Phase 8.2 is hygiene.

---

## Component Map (legacy attr → app_ctx key)

| Legacy attribute              | app_ctx key              | Engine consumer       | Layer |
| ----------------------------- | ------------------------ | --------------------- | ----- |
| `exchange_client`             | `exchange_client`        | MarketAccount         | L1    |
| `market_data_feed`            | `market_data_feed`       | MarketAccount         | L2    |
| `balance_sync` / `…_updater`  | `balance_manager`        | MarketAccount         | L2    |
| `market_regime_detector`      | `market_regime_detector` | Situation             | L2    |
| `portfolio_manager`           | `portfolio_manager`      | Situation             | L3    |
| `signal_manager`              | `signal_manager`         | Situation             | L5    |
| `agent_manager` (signal hub)  | `signal_fusion`          | Situation             | L5    |
| `meta_controller`             | `arbitration_engine`     | Decision              | L5/L8 |
| (config flag)                 | `mode_manager`           | Decision              | L5    |
| `execution_manager`           | `execution_manager`      | SafeExecution         | L4    |
| (none — global)               | `bounded_cache`          | SafeExecution (FIX#2) | L0    |
| `risk_manager`                | `risk_manager`           | Decision/Operations   | L6    |
| `health_monitor`              | `health_monitor`         | Operations            | L7    |
| `startup_orchestrator`        | `startup_orchestrator`   | Operations            | L8    |
| `shared_state`                | `shared_state`           | All (read-only)       | L0    |

Engines already implement graceful degradation per key, so missing
mappings will simply log a warning.

---

## Phase 8.1 — Implementation Steps

| # | Action | File | Status |
|---|--------|------|--------|
| 1 | Create `production_bridge.py` that imports `MasterSystemOrchestrator` and exposes `build_production_app_ctx()` | `core_engine/production_bridge.py` | TODO |
| 2 | Patch `integration.py::create_app_context()` to call bridge when `OCTIVAULT_MODE=production` env var set (default = mocks) | `core_engine/integration.py` | TODO |
| 3 | Add `--production` CLI flag to `main.py` to toggle bridge | `main.py` | TODO |
| 4 | Smoke test: `python3 main.py --mode=paper-trade --duration=2min --production` — verify `nav>0`, no crashes | `production_smoke.log` | TODO |
| 5 | Document results in `PHASE_8_BRIDGE_VALIDATION.md` | `PHASE_8_BRIDGE_VALIDATION.md` | TODO |
| 6 | Commit with `[phase-8.1] production bridge` message | git | TODO |

### Acceptance Criteria

- `paper-trade --production` runs ≥ 60 cycles without errors.
- Cycle telemetry shows `nav > 0` (real balance from Binance).
- At least one of `sigs`, `dec`, `exe` becomes non-zero within 5 minutes
  (depending on whether market gives signals during the test window).
- Old mock mode still works when `--production` is not passed.

### Failure-Mode Decisions

- If a legacy component fails to construct, **log + skip** — engines
  degrade gracefully. Phase 8.2 will harden these one by one.
- If `live` mode is requested without `APPROVE_LIVE_TRADING=YES`,
  prerequisite check fails — bridge aborts (existing legacy behaviour).
- If bridge import fails entirely, fall back to empty app_ctx (current
  behaviour) and log a clear error.

---

## Phase 8.2 — Native Migration (later)

Once 8.1 is stable, gradually replace the bridge with native construction.
Order of migration (low risk → high risk):

1. **L0 first**: `SharedState`, `BoundedCache` — pure state, no I/O.
2. **L7/L8**: `HealthMonitor`, `StartupOrchestrator` — observability/lifecycle.
3. **L2**: `MarketDataFeed`, `BalanceManager` — read-only market I/O.
4. **L3**: `PortfolioManager` — depends on shared_state + balance.
5. **L5**: `SignalManager`, `SignalFusion`, `ArbitrationEngine`, `ModeManager`.
6. **L4 last**: `ExecutionManager` — touches real money, do last with careful tests.
7. **L1**: `ExchangeClient` — keep as-is via legacy until everything else is native.

Each migration step: extract the construction code from the legacy
orchestrator, place it in `integration.py`, swap the bridge mapping,
verify, commit.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Legacy orchestrator pulls in 145+ modules → import explosion | Already the case for the legacy entry point; bridge doesn't change footprint |
| Live trading triggered accidentally | Bridge respects `APPROVE_LIVE_TRADING` env var; default mode=paper-trade |
| Side effects from `initialize_components()` (real API calls, balance polling, WS connections) | Acceptable for paper-trade; document network calls in bridge docstring |
| Mode confusion between `--mode=paper-trade` (façade) and orchestrator's `live_mode` | Bridge passes through mode arg; both must agree (assertion) |
| Frozen modules in `MODULE_FREEZE_MANIFEST.json` may be re-imported | Acceptable — freeze prevents *modification*, not import |

---

## Success Definition

Phase 8.1 is **done** when:
1. `main.py --production` produces real NAV from Binance balance.
2. The 5-phase loop runs ≥ 60 cycles cleanly.
3. The bridge can be toggled off (`--production` removed) and old mock
   mode still passes its smoke test.
4. `PHASE_8_BRIDGE_VALIDATION.md` is committed.
