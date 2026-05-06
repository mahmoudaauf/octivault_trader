# Phase 8.2.8 — Bridge Deprecation: Preparation

**Status:** preparation only — `production_bridge.py` is **not deleted**.
Doing so today would break `create_app_context(production=True)`, which
is the only path that wires the production `app_ctx`.

## What landed in this step

1. **Deprecation notice** in `core_engine/production_bridge.py`:
   - `build_production_app_ctx()` now emits a `DeprecationWarning`
     (once per process) pointing at the native replacement seam.
   - Behavior unchanged: legacy orchestrator is still loaded and mapped.

2. **Native app-context seam** in `core_engine/native/app_context.py`:
   - `NativeComponents` dataclass — pre-constructed L0-L6 instances.
   - `build_native_app_ctx(components) -> (app_ctx, NativeOrchestrator)`.
   - Stable key contract published as `NATIVE_CTX_KEYS`.
   - Pure assembly, no I/O, no credentials. Fully unit-testable.

3. **Native bootstrap** in `core_engine/native/bootstrap.py`:
   - `BootstrapConfig` (frozen dataclass) — explicit credential + tuning
     surface, plus a `from_env()` classmethod with safe coercion.
   - `build_components(cfg, *, exchange_client_factory=None)` — async
     builder that constructs all L0-L6 native instances ready to be
     handed to `build_native_app_ctx`. Tests can inject a stub exchange
     client to avoid network setup.
   - `shutdown_components(components)` — best-effort, idempotent
     teardown of pollers + HTTP session.

4. **Tests**:
   - `tests/test_native_app_context.py` (9 tests) — factory + warning.
   - `tests/test_native_bootstrap.py` (13 tests) — config parsing,
     wiring, injection seam, shutdown idempotence, end-to-end cycle.

## What still blocks final deletion

The bridge populates ~25 `app_ctx` keys from the legacy orchestrator.
The native stack provides 6 of these today:

| ctx key | provided by native | status |
|---|---|---|
| `shared_state` | `NativeSharedState` | ✅ |
| `balance_manager` | `NativeBalanceSync` | ✅ |
| `market_data_feed` | `NativeMarketData` | ✅ |
| `signal_manager` | `NativeSignalEngine` | ✅ |
| `decision_engine` | `NativeDecisionEngine` | ✅ (native-only key) |
| `execution_manager` | `NativeExecutor` | ✅ |
| `telemetry` | `NativeTelemetry` | ✅ (native-only key) |
| `exchange_client` | — | ❌ legacy-only |
| `order_cache_manager` | — | ❌ legacy-only |
| `balance_cache_updater` | — | ❌ legacy-only |
| `market_regime_detector` | — | ❌ legacy-only |
| `volatility_regime` | — | ❌ legacy-only |
| `heartbeat` | — | ❌ legacy-only |
| `portfolio_manager` | — | ❌ legacy-only |
| `position_manager` | — | ❌ legacy-only |
| `symbol_manager` | — | ❌ legacy-only |
| `three_bucket_manager` | — | ❌ legacy-only |
| `tp_sl_engine` | — | ❌ legacy-only |
| `safety_order_manager` | — | ❌ legacy-only |
| `recovery_engine` | — | ❌ legacy-only |
| `signal_fusion` | — | ❌ legacy-only |
| `arbitration_engine` | — | ❌ legacy-only |
| `risk_manager` | — | ❌ legacy-only |
| `health_monitor` | — | ❌ legacy-only |
| `performance_monitor` | — | ❌ legacy-only |
| `alert_system` | — | ❌ legacy-only |
| `startup_orchestrator` | — | ❌ legacy-only |
| `watchdog` | — | ❌ legacy-only |

Some of these are real layers that Phase 8.2 has not yet ported
(`risk_manager`, `health_monitor`, lifecycle), and some are strategy
features the 5 façade engines treat as optional (graceful degradation).

## Required before deletion

1. Decide per remaining key: **port to native**, **keep legacy via
   compat shim**, or **drop** (graceful-degrade indefinitely).
2. ~~Build a dedicated `core_engine/native/bootstrap.py` that:~~ ✅ done.
3. Update `create_app_context(production=True, native=True)` to call
   `bootstrap.build_components()` then `build_native_app_ctx(...)`,
   without touching the legacy bridge.
4. Run a paper-trading session against the native path end-to-end.
5. Only then: delete `production_bridge.py`, the
   `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` legacy loader, and the legacy
   integration branch in `core_engine/integration.py`.

## Tests gate

After this prep step:

```
pytest tests/test_native_l0..6.py tests/test_native_l8.py \
       tests/test_integration_full_cycle.py \
       tests/test_native_app_context.py tests/test_native_bootstrap.py -q
# 190 passed
```
