# Phase 8.2.8 — Legacy `app_ctx` Key Triage

**Date:** 2026-05-06
**Branch:** `phase-3/wiring`
**Status:** Decision document — drives the final deletion step.

This document closes the open question from `PHASE_8_2_8_PREP.md`:
*"Decide per remaining key: port to native, keep legacy via compat shim,
or drop."*

The triage is grounded in the **actual demand surface** of the 5 façade
engines (`situation_engine`, `decision_engine`, `safe_execution_engine`,
`operations_engine`, plus `setup_core_engines` in `integration.py`).
That surface was extracted by static grep of every
`app_ctx.get(...)` / `app_ctx[...]` read.

---

## 1. Demand vs supply matrix

The bridge (`production_bridge.ATTR_TO_CTX_KEY`) supplies 28 keys. The
façade engines read 23 keys. The intersection is what actually matters.

### 1a. Bridge supplies, façade reads — **REAL gap if missing in native**

| key | reader(s) | native status | decision |
|---|---|---|---|
| `shared_state` | (none direct, used by orchestrator) | ✅ `NativeSharedState` | covered |
| `exchange_client` | `safe_execution_engine` ×2, `setup_core_engines` | ✅ now exposed via `NativeComponents.exchange_client` (this commit) | covered |
| `market_data_feed` | `setup_core_engines` | ✅ `NativeMarketData` | covered |
| `balance_manager` | (none direct) | ✅ `NativeBalanceSync` | covered |
| `signal_manager` | (none in façades; orchestrator only) | ✅ `NativeSignalEngine` | covered |
| `execution_manager` | `setup_core_engines` | ✅ `NativeExecutor` | covered |
| `portfolio_manager` | `situation_engine`, `setup_core_engines` | ✅ `NativePortfolioManager` (8.3.7) | **native** |
| `position_manager` | `situation_engine`, `decision_engine` | ✅ `NativePositionManager` (8.3.8) | **native** |
| `risk_manager` | `situation_engine`, `decision_engine` | partial — risk caps live in `NativeDecisionEngine` | **drop key**, behavior already in DE |
| `tp_sl_engine` | `decision_engine` | ✅ `NativeTPSLEngine` (8.3.9) | **native** |
| `safety_order_manager` | `safe_execution_engine` | ✅ `NativeSafetyOrderManager` (8.3.10) | **native** |
| `recovery_engine` | `operations_engine` ×2 | ❌ not in native | **shim** (TBD) |
| `signal_fusion` | `setup_core_engines` | absorbed | **drop key** — `NativeSignalEngine` is the fusion |
| `arbitration_engine` | `setup_core_engines` | absorbed | **drop key** — `NativeDecisionEngine` is the gate |
| `health_monitor` | `setup_core_engines` | ❌ not in native | **shim** (TBD) |
| `performance_monitor` | `operations_engine` | partial — `NativeTelemetry` covers latency | **map** `telemetry` → also accept this key, or drop |
| `startup_orchestrator` | `setup_core_engines` | n/a — bootstrap *is* the startup | **drop key** |
| `watchdog` | `operations_engine` ×2 | ❌ not in native | **shim** (TBD) |

### 1b. Bridge supplies, **nobody reads** — dead weight

These were mapped historically but no façade engine consumes them. The
native stack does **not** need to provide them.

| key | decision |
|---|---|
| `order_cache_manager` | **drop** |
| `balance_cache_updater` | **drop** |
| `market_regime_detector` | **drop** |
| `volatility_regime` | **drop** |
| `heartbeat` | **drop** |
| `symbol_manager` | **drop** |
| `three_bucket_manager` | **drop** |
| `alert_system` | **drop** |

### 1c. Façade reads, **bridge does not supply** — already optional in production

These are read with `.get()` and silently skipped today even on the
legacy production path. They're already in graceful-degrade state and
no native action is required.

| key | reader |
|---|---|
| `anomaly_detection` | `situation_engine` |
| `mode_manager` | `decision_engine` ×2, `setup_core_engines` |
| `fourth_slot_tracker` | `decision_engine` |
| `policy_manager` | `decision_engine` |
| `bounded_cache` | `setup_core_engines` |
| `lifecycle_manager` | `operations_engine` |
| `state_manager` | `operations_engine` |
| `prometheus_exporter` | `operations_engine` |
| `event_store` | `operations_engine` ×2 |

---

## 2. Final decision summary

| bucket | count | action |
|---|---|---|
| Native already covers | 7 | none |
| Newly covered this commit (`exchange_client`) | 1 | done |
| Drop bridge mapping (dead weight or absorbed) | 11 | delete from `ATTR_TO_CTX_KEY` |
| Already optional (graceful-degrade) | 9 | none — keep `.get()` reads, document |
| **True remaining shims required** | **6** | see §3 |

The remaining six (`portfolio_manager`, `position_manager`, `tp_sl_engine`,
`safety_order_manager`, `recovery_engine`, `health_monitor`, `watchdog`)
are the only items blocking final deletion of `production_bridge.py`.

> Note: `risk_manager`, `signal_fusion`, `arbitration_engine`,
> `startup_orchestrator`, and `performance_monitor` are **decided as
> drops** — their behavior is either already in native or genuinely
> not needed at the façade layer.

---

## 3. Shim plan for the remaining six

For each, we have three options:

1. **Port to native** — write a `NativeXxx` class.
2. **Compat shim** — thin adapter that wraps a legacy instance and
   exposes only the methods the façade actually calls.
3. **Stub** — null-object that returns sensible defaults (the façade
   already graceful-degrades, so a stub is cheap).

### Recommendation per item

| key | reader behavior | recommendation |
|---|---|---|
| `portfolio_manager` | `situation_engine.observe()` calls `get_nav()` + `get_positions()` | ✅ resolved in **8.3.7** — `NativePortfolioManager` (read-only aggregator over `NativeSharedState` + `NativeBalanceSync`) |
| `position_manager` | `situation_engine` reads list; `decision_engine` checks for existing position before opening | ✅ resolved in **8.3.8** — `NativePositionManager` (read-only per-symbol accessor over `NativeSharedState.positions`; provides `get_position` + `analyze_position`) |
| `tp_sl_engine` | `decision_engine` consults for protective-order placement decisions | **stub** returning "no action" — decision engine already graceful-degrades |
| `safety_order_manager` | `safe_execution_engine` calls `place_safety_orders(order)` | **stub** returning `[]` |
| `recovery_engine` | `operations_engine` calls `check_recovery_needed()` and `recover()` | **stub** returning `False` / no-op |
| `health_monitor` | `setup_core_engines` only references for wiring (not invoked in cycle) | **drop** — wiring path can tolerate `None` |
| `watchdog` | `operations_engine` calls `heartbeat()` and `check_liveness()` | **stub** returning `True` |

### Why stubs over ports

Phase 8.2 was scoped as a **bottom-up rebuild of the trading core**
(L0–L8). Operational concerns (recovery, watchdog, health) and the
position/portfolio model are explicit non-goals for 8.2 — they're owned
by Phase 8.3+. A stub layer:

* unblocks `production_bridge.py` deletion **today**,
* keeps the native `app_ctx` honest about what's missing (stubs log
  `debug` on each call so triage is rediscoverable),
* leaves the façade engines unchanged (they already handle missing
  components gracefully).

### Where the stubs live

Proposed: `core_engine/native/compat.py` — a single ~120 LOC file with
six small classes and a `register_compat_stubs(app_ctx)` helper. The
caller (`build_native_app_ctx` or a separate opt-in flag) decides
whether to install them. Default: **opt-in via `compat=True` argument**
so unit tests of the pure factory stay clean.

---

## 4. What lands in this commit

- `NativeComponents.exchange_client` field (+ exposed in `app_ctx`).
- `NATIVE_CTX_KEYS` updated.
- `bootstrap.build_components()` populates the new field; `shutdown_components()`
  prefers it for HTTP close.
- 2 new tests:
  - `test_build_native_app_ctx_omits_exchange_client_when_none`
  - `test_build_native_app_ctx_includes_exchange_client_when_provided`
- 1 augmented test: `test_build_components_returns_wired_native_components`
  now asserts `components.exchange_client is not None`.
- This document.

**Test gate:** 199/199 passing.

---

## 5. Next concrete step

Implement `core_engine/native/compat.py` with the six stubs from §3.
Once landed, all keys the façades actually read will be present (or
deliberately absent with documented graceful-degrade behavior), and the
final deletion of `production_bridge.py` is unblocked pending only the
paper-trade smoke run.
