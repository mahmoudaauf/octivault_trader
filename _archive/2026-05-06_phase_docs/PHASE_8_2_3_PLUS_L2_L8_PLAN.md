# Phase 8.2.3+ — Native Layers L2–L8 Plan

**Status:** 📋 Planning (parallel-ready stubs)
**Predecessor:** Phase 8.2.2 (L1 Native: ExchangeClient + BalanceSync + OrderExecution) ✅
**Successor:** Phase 8.3 (Native cutover; legacy retired)

---

## Overview

This document scopes the remaining native layers (L2 → L8) at sufficient
fidelity to be picked up in parallel work streams. Each layer follows the
same template established by Phase 8.2.1 / 8.2.2:

1. Legacy analysis (what we're replacing, LOC, pain points)
2. Native target (modules, public surface, dependencies)
3. Implementation phases (typically 3–5 sub-phases)
4. Integration points (which existing core_engine façades touch this)
5. Tests (target count, fixture style)
6. Success criteria
7. Rollback plan

Each spec, when authored as a standalone `PHASE_8_2_X_LY_NATIVE_SPEC.md`,
should be ~300–500 lines. The stubs below establish scope only.

---

## L2 — Market Data (Phase 8.2.3)

**Why now:** Trading decisions need price + candle history. Currently the
legacy `data_engines/market_data.py` (~600 LOC) handles caching,
multi-symbol batching, and stale-quote detection.

**Native target:** `core_engine/native/market_data.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeMarketData` | ~250 | `start()`, `stop()`, `get_price(sym)`, `get_klines(sym, iv, n)`, `latest_quote_age(sym)`, `prime(symbols)` |

**Dependencies:** L1 `NativeExchangeClient.get_prices` / `get_klines`.

**Implementation phases:**
1. In-memory price cache with monotonic timestamp
2. Bulk-pull poller using `get_prices(None)` once per cycle
3. Klines on-demand with bounded LRU per (symbol, interval)
4. Stale-quote detection (age > N × poll_interval ⇒ flag)
5. Optional async WebSocket upgrade path (deferred; REST suffices)

**Tests:** ~12 tests — cache hit/miss, staleness threshold, klines LRU
eviction, multi-symbol fetch, callback fan-out (none — pure read API).

**Integration:** `core_engine/production_bridge.py` swaps the legacy
`MarketData` reference with `NativeMarketData` behind the same accessor
methods. No call-site changes required.

**Success criteria:**
- All 12 tests pass
- Mock cycle still ≤ 1ms median when L2 native active
- No regressions in 6h paper-trade soak

**Rollback:** Single-line revert in `production_bridge.py` (factory swap).

---

## L3 — Signal Engine (Phase 8.2.4)

**Legacy:** `signal_engine/` directory, ~1500 LOC across multiple
strategies (RSI, MACD, MA crossover). High duplication, mixed sync/async.

**Native target:** `core_engine/native/signals.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeSignalEngine` | ~300 | `evaluate(symbol, klines) → list[Signal]`, `register_strategy(name, fn)`, `enabled_strategies` |
| `Signal` (dataclass) | — | `symbol`, `direction` (BUY/SELL/HOLD), `score` (0..1), `strategy`, `meta` |

**Dependencies:** L2 `NativeMarketData` (klines).

**Implementation phases:**
1. Strategy plugin registry (function-based; no class hierarchy)
2. Pure-numpy implementations of RSI, MACD, MA crossover (no pandas)
3. Aggregation (weighted average score across enabled strategies)
4. Hysteresis / cooldown to prevent flip-flopping

**Tests:** ~15 tests — golden-vector tests against known indicator outputs
(RSI(14) on a known series, MACD on EOD data), aggregation math, cooldown
behavior, plugin registration.

**Success criteria:**
- Numpy-only (no pandas dependency in core hot path)
- Signals produced with score ∈ [0, 1] always
- Per-cycle evaluation < 5 ms for 30-symbol universe

---

## L4 — Decision Engine (Phase 8.2.5)

**Legacy:** `decision_engine/` ~800 LOC. Mixes risk gating with order sizing.

**Native target:** `core_engine/native/decisions.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeDecisionEngine` | ~250 | `decide(signals, portfolio, balance) → list[Decision]` |
| `Decision` (dataclass) | — | `symbol`, `action` (OPEN/CLOSE/HOLD), `quantity`, `reason`, `risk_score` |

**Dependencies:** L0 `NativeSharedState`, L3 signals.

**Implementation phases:**
1. Position-sizing math (Kelly fraction, capped exposure, per-symbol caps)
2. Risk gates (max-drawdown, daily-loss, concurrent-positions)
3. Action ranking (highest-conviction first, capital-aware)
4. Idempotency tagging (decision UUID for downstream dedup)

**Tests:** ~12 tests — sizing edge cases (zero balance, dust, max
exposure), gate trips, ranking stability.

**Success criteria:**
- Pure function (decide is deterministic given identical inputs)
- All decisions reversible (action enum is closed set)

---

## L5 — Execution Coordinator (Phase 8.2.6)

**Legacy:** `execution_coordinator.py` ~700 LOC.

**Native target:** `core_engine/native/executor.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeExecutor` | ~200 | `execute(decisions) → list[ExecutionResult]` |

**Dependencies:** L1 `NativeOrderExecution`, L0 `NativeSharedState`.

**Implementation phases:**
1. Sequential execution with per-symbol bulkhead
2. Idempotency dedup using decision UUID
3. Partial-fill reconciliation against `NativeBalanceSync`
4. Failure classification (retryable vs terminal)

**Tests:** ~10 tests — happy path, partial fills, exchange rejection,
duplicate decision rejection.

---

## L6 — Recovery / Health (Phase 8.2.7)

**Legacy:** `auto_recovery.py`, `capital_health_monitor.py` (~500 LOC combined).

**Native target:** `core_engine/native/health.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeHealthMonitor` | ~200 | `check() → HealthReport`, `is_healthy()`, `last_alert` |
| `HealthReport` (dataclass) | — | per-component status + summary |

**Implementation phases:**
1. Component probes (exchange ping, balance freshness, NAV drift)
2. Alert thresholds (configurable via L0 ConfigLoader)
3. Auto-pause hooks into L4 / L5 (return BUSY ⇒ skip cycle)

**Tests:** ~10 tests — each probe under healthy/degraded/critical.

---

## L7 — Telemetry / Logging (Phase 8.2.8)

**Legacy:** Scattered `logger.info` / metrics across the codebase, plus
`monitoring/`.

**Native target:** `core_engine/native/telemetry.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeTelemetry` | ~150 | `record_cycle(metrics)`, `record_event(name, **kw)`, `snapshot()` |

**Implementation phases:**
1. In-memory ring buffer (last N=1000 cycles)
2. Lightweight Prometheus-compatible exposition (text format, optional)
3. JSONL log sink (append-only, rotated by external `log_rotator.sh`)

**Tests:** ~8 tests — buffer wrap, exposition format, sink write ordering.

---

## L8 — Orchestration (Phase 8.2.9)

**Legacy:** `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` (~1200 LOC) — the final
boss. Currently delegated to via `production_bridge`.

**Native target:** `core_engine/native/orchestrator.py`

| Module | LOC target | Public surface |
|---|---|---|
| `NativeOrchestrator` | ~300 | `run_cycle()`, `run_loop(duration_or_cycles)`, `shutdown()` |

This is the layer that **retires** the legacy orchestrator. It composes
L0–L7 native components and exposes the same `run_cycle` shape `main.py`
already calls.

**Implementation phases:**
1. Compose all native layers via `setup_core_engines(production=True, native=True)`
2. Implement the 5-phase cycle (READ/UNDERSTAND/DECIDE/EXECUTE/RECOVER)
   in pure native code
3. Parity validation against legacy (Phase 8.3 equivalence test)
4. Cutover flag wiring + feature-flag rollout

**Tests:** ~15 integration tests — full cycle in mock mode, error paths,
graceful shutdown, hot-reload of config.

**Success criteria (the big one):**
- All ≥ 110 native tests pass (L0..L8)
- 6-hour paper-trade soak: 0 errors, NAV drift < 0.5% vs legacy
- Median cycle ≤ legacy median; p99 ≤ 1.5× legacy p99

**Rollback:** Feature flag `native_full=False` reverts to bridge mode in
one commit.

---

## Sequencing & Parallelism

```
                      ┌── L2 (data) ──┐
                      │               │
   L0 ✅  →  L1 ✅  →  ├── L3 (sig) ───┼──→ L4 (dec) ──→ L5 (exe) ─┐
                      │               │                            │
                      └── L6 (health) ┴── L7 (telemetry) ──────────┴── L8 (orch)
```

* **L2, L6, L7 are parallel-safe** once L1 lands.
* **L3 → L4 → L5** is the critical path.
* **L8 is the join point.**

Recommended cadence: 1 layer per ~1 week, with L2/L3 in parallel.

---

## Cross-cutting concerns

* **Type discipline.** `from __future__ import annotations` everywhere.
  No `Any` returns from public surface (use TypedDicts or dataclasses).
* **No global state.** All native classes accept dependencies via
  `__init__`; module-level singletons are forbidden (L0 `ConfigLoader` is
  the documented exception).
* **Async by default.** Every method that touches I/O is `async`. CPU-only
  helpers stay sync.
* **Logging.** Logger name = `core_engine.native.<module>`. No print().
* **Tests.** Each layer ships with `tests/test_native_l<N>.py`. Use
  pytest-asyncio. Mock the layer below; never call the real exchange.

---

## Status table

| Layer | Phase | Status | Owner | Tests |
|---|---|---|---|---|
| L0 utilities | 8.2.1 | ✅ done | — | 29/29 |
| L1 exchange  | 8.2.2 | ✅ done | — | 20/20 |
| L2 data      | 8.2.3 | 📋 spec'd | tbd | 0/12 |
| L3 signals   | 8.2.4 | 📋 spec'd | tbd | 0/15 |
| L4 decisions | 8.2.5 | 📋 spec'd | tbd | 0/12 |
| L5 execution | 8.2.6 | 📋 spec'd | tbd | 0/10 |
| L6 health    | 8.2.7 | 📋 spec'd | tbd | 0/10 |
| L7 telemetry | 8.2.8 | 📋 spec'd | tbd | 0/8  |
| L8 orchestr. | 8.2.9 | 📋 spec'd | tbd | 0/15 |

**Total native test target: ~131.**
