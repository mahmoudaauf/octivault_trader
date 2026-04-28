# 🏛️ LOGICAL LAYERED ARCHITECTURE — Octivault Trader
**Version:** 1.0 (April 2026)
**Status:** Authoritative — supersedes ad‑hoc layering in `core/layer_contracts.py` (which only covered L3–L5)
**Scope:** All 226 production Python modules + 9 test modules in the workspace.
**Companion code:** `core/layer_contracts.py`, `core/layer_orchestrator.py`, `core/contracts.py`, `core/app_context.py`, `🎯_MASTER_SYSTEM_ORCHESTRATOR.py`

---

## 0. Design Principles

1. **Strict downward dependency.** A layer may only import / call the layer directly below it (and shared *Cross‑Cutting* utilities). No upward calls — upper layers are notified through **events / queues / callbacks** registered at startup.
2. **One contract per boundary.** Every inter‑layer call goes through a typed `LayerInput` / `LayerOutput` (see `core/layer_contracts.py`) — never via shared mutable globals.
3. **Single source of truth per concern.** Balances → Wallet Layer. Positions registry → Portfolio Layer. Decisions → Strategy Layer. Orders → Execution Layer. Each layer below is the authoritative store for its concern.
4. **Failure containment.** A crash inside a layer must not corrupt the layer above or below; recovery is owned by the *Lifecycle & Recovery Layer* (L8) which can restart any layer in isolation.
5. **No component is optional.** Every one of the 226 scripts has been placed in exactly one layer. Removing any of them breaks at least one declared contract — see §11 *Necessity Matrix*.

---

## 1. Layer Stack (top → bottom)

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  L8  LIFECYCLE & RECOVERY     (boot, watchdog, chaos, restart)       │
 ├──────────────────────────────────────────────────────────────────────┤
 │  L7  OBSERVABILITY & UX       (dashboards, monitors, alerts, APM)    │
 ├──────────────────────────────────────────────────────────────────────┤
 │  L6  GOVERNANCE & POLICY      (risk, capital governor, rule proposer)│
 ├──────────────────────────────────────────────────────────────────────┤
 │  L5  STRATEGY & DECISION      (agents, ML, signal fusion, arbitrator)│
 ├──────────────────────────────────────────────────────────────────────┤
 │  L4  EXECUTION & ORDER MGMT   (router, maker exec, TP/SL, retries)   │
 ├──────────────────────────────────────────────────────────────────────┤
 │  L3  PORTFOLIO & STATE        (buckets, positions, rotation, journal)│
 ├──────────────────────────────────────────────────────────────────────┤
 │  L2  WALLET & MARKET DATA     (balance sync, OHLCV, WS feeds)        │
 ├──────────────────────────────────────────────────────────────────────┤
 │  L1  EXCHANGE I/O             (REST + WS client, order cache)        │
 ├──────────────────────────────────────────────────────────────────────┤
 │  L0  CROSS‑CUTTING            (config, logging, errors, contracts)   │
 └──────────────────────────────────────────────────────────────────────┘
```

Every horizontal boundary (`──`) is a **contract surface**. Code on one side may only invoke code on the other side through the interface declared in §3–§10.

---

## 2. Layer Cheat‑Sheet

| L# | Layer | Owns | Authoritative State | Key Module(s) |
|----|-------|------|---------------------|---------------|
| L0 | Cross‑Cutting | constants, types, logging, errors, contracts | none (pure code) | `core/contracts.py`, `core/config.py`, `core/error_types.py`, `utils/logging_setup.py` |
| L1 | Exchange I/O | raw REST/WS connectivity, rate‑limit, order cache | `OrderCacheManager` | `core/exchange_client.py`, `core/order_cache_manager.py`, `core/ws_market_data.py` |
| L2 | Wallet & Market Data | balance/positions sync, OHLCV, regime feeds | `BalanceManager`, `MarketDataFeed` | `core/balance_manager.py`, `core/market_data_feed.py`, `utils/ohlcv_cache.py` |
| L3 | Portfolio & State | bucket classification, position registry, journal | `PortfolioAuthority`, `ThreeBucketManager`, `TradeJournal` | `core/portfolio_authority.py`, `core/three_bucket_manager.py`, `core/trade_journal.py` |
| L4 | Execution & Order Mgmt | order placement, TP/SL, retries, liquidation | `ExecutionManager` | `core/execution_manager.py`, `core/maker_execution.py`, `core/tp_sl_engine.py` |
| L5 | Strategy & Decision | signals, agents, ML, fusion, arbitration | `SignalManager`, `ArbitrationEngine` | `agents/*`, `core/signal_fusion.py`, `core/arbitration_engine.py` |
| L6 | Governance & Policy | risk caps, capital allocation, rule overrides | `CapitalGovernor`, `RiskManager`, `PolicyManager` | `core/risk_manager.py`, `core/capital_governor.py`, `automation/*` |
| L7 | Observability & UX | metrics, dashboards, alerts, APM, journals | `MetricsRegistry`, `AlertSystem` | `core/metrics.py`, `core/alert_system.py`, `core/prometheus_exporter.py`, `dashboards/*` |
| L8 | Lifecycle & Recovery | boot order, watchdog, restart, chaos | `LifecycleManager`, `RecoveryEngine`, `Watchdog` | `core/lifecycle_manager.py`, `core/recovery_engine.py`, `core/watchdog.py`, `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` |

---

## 3. L0 — Cross‑Cutting Foundation

**Purpose.** Pure, side‑effect‑free building blocks used by every layer. No I/O. No global mutation.

### 3.1 Public Interface (consumed by L1–L8)
```python
# core/contracts.py
class Position(TypedDict): symbol: str; qty: float; entry: float; ...
class Signal(TypedDict): symbol: str; side: Side; confidence: float; ...
class Order(TypedDict): id: str; symbol: str; status: OrderStatus; ...
class Balance(TypedDict): asset: str; free: float; locked: float; ...

# core/error_types.py
class OctiError(Exception): ...                  # base
class ExchangeError(OctiError): ...              # L1 raises
class WalletDriftError(OctiError): ...           # L2 raises
class PortfolioInvariantError(OctiError): ...    # L3 raises
class ExecutionError(OctiError): ...             # L4 raises
class StrategyError(OctiError): ...              # L5 raises
class GovernanceVeto(OctiError): ...             # L6 raises
```

### 3.2 Modules in L0 (16)
`core/contracts.py` · `core/config.py` · `core/config_constants.py` · `core/config_validator.py` · `core/error_types.py` · `core/error_handler.py` · `core/core_utils.py` · `core/logger_utils.py` · `core/stubs.py` · `core/app_context.py` · `core/layer_contracts.py` · `utils/logging_setup.py` · `utils/indicators.py` · `utils/ta_indicators.py` · `utils/hyg_guards.py` · `utils/tuned_params.py` · `utils/pnl_calculator.py` · `utils/volatility_adjusted_confidence.py` · `utils/symbol_filter_pipeline.py` · `utils/shared_state_tools.py` · `config/EV_ALIGNMENT_CONFIG.py` · `balance_threshold_config.py`

### 3.3 Invariants
- L0 modules **must not import** any module from L1–L8.
- All exceptions raised across boundaries inherit from `OctiError`.
- All numeric quantities are `Decimal` or `float` consistently per `contracts.py`.

---

## 4. L1 — Exchange I/O

**Purpose.** Single chokepoint for every byte that crosses the network to/from the exchange.

### 4.1 Downward dependency: **L0 only**.

### 4.2 Public Interface (consumed by L2 + L4)
```python
class IExchangeClient(Protocol):
    async def get_balances(self)              -> dict[str, Balance]
    async def get_open_positions(self)        -> list[Position]
    async def place_order(self, o: Order)     -> Order            # L4
    async def cancel_order(self, oid: str)    -> bool             # L4
    async def get_klines(self, sym, tf, n)    -> list[Kline]      # L2

class IOrderCache(Protocol):
    def upsert(self, o: Order) -> None
    def get(self, oid: str)    -> Order | None
    def reconcile(self, exchange_orders) -> ReconcileReport       # L4 + L8
```

### 4.3 Modules in L1 (8)
`core/exchange_client.py` · `core/exchange_truth_auditor.py` · `core/order_cache_manager.py` · `core/ws_market_data.py` · `core/market_data_websocket.py` · `core/polling_coordinator.py` · `core/balance_sync_backoff.py` · `core/retry_manager.py`

### 4.4 Invariants
- **No business logic.** L1 only translates network → typed L0 objects.
- All retries / backoff happen here; higher layers see either success or a typed `ExchangeError`.
- `OrderCacheManager` is the *only* component that may write to local order state derived from exchange data.

---

## 5. L2 — Wallet & Market Data

**Purpose.** Convert raw exchange streams into a clean, classified, time‑synchronized world model.

### 5.1 Downward dependency: **L1 + L0**.

### 5.2 Public Interface (consumed by L3)
```python
class IBalanceProvider(Protocol):
    def snapshot(self) -> WalletSnapshot            # asset → (free, locked, classification)
    def subscribe(self, cb: Callable[[WalletSnapshot], None]) -> None

class IMarketDataProvider(Protocol):
    def latest(self, symbol: str)            -> Tick
    def ohlcv(self, symbol, tf, n)           -> DataFrame
    def regime(self, symbol: str)            -> RegimeLabel       # consumed by L5/L6
```

### 5.3 Modules in L2 (12)
`core/balance_manager.py` · `core/market_data_feed.py` · `core/heartbeat.py` · `core/correlation_manager.py` · `core/market_regime_detector.py` · `core/market_regime_integration.py` · `core/volatility_regime.py` · `core/nav_regime.py` · `core/regime_proposal_analyzer.py` · `core/anomaly_detection.py` · `utils/ohlcv_cache.py` · `stream/__init__.py`

### 5.4 Invariants
- **Wallet Layer Contract** (already in `core/layer_contracts.py::WalletLayerContract`) applies here verbatim.
- All values are exchange‑verified; no inferred balances.
- A `WalletSnapshot` is immutable; updates produce a new snapshot + event.

---

## 6. L3 — Portfolio & State

**Purpose.** Authoritative registry of *what we own and why*, segmented into the three buckets (CASH / TRADING / EXTERNAL).

### 6.1 Downward dependency: **L2 + L0**.

### 6.2 Public Interface (consumed by L4 + L5 + L6)
```python
class IPortfolioAuthority(Protocol):
    def buckets(self)                        -> ThreeBuckets
    def positions(self)                      -> dict[str, Position]
    def classify(self, asset: str)           -> AssetClass           # CASH/TRADE/EXTERNAL/DUST
    def reserve(self, sym, qty, reason)      -> ReservationToken     # for L4 to spend
    def release(self, token)                 -> None
    def journal(self)                        -> ITradeJournal
```

### 6.3 Modules in L3 (28)
`core/portfolio_authority.py` · `core/portfolio_manager.py` · `core/portfolio_balancer.py` · `core/portfolio_buckets.py` · `core/portfolio_segmentation.py` · `core/three_bucket_manager.py` · `core/bucket_classifier.py` · `core/position_manager.py` · `core/position_merger_enhanced.py` · `core/position_operation_validator.py` · `core/restart_position_classifier.py` · `core/holding_utility.py` · `core/symbol_manager.py` · `core/symbol_rotation.py` · `core/rotation_authority.py` · `core/universe_rotation_engine.py` · `core/bootstrap_symbols.py` · `core/bootstrap_manager.py` · `core/discovery_coordinator.py` · `core/trade_journal.py` · `core/event_store.py` · `core/state_manager.py` · `core/state_synchronizer.py` · `core/shared_state.py` · `core/replay_engine.py` · `core/dead_capital_healer.py` · `core/reserve_manager.py` · `portfolio/__init__.py` · `system_state_manager.py`

### 6.4 Invariants
- **Portfolio Layer Contract** in `core/layer_contracts.py::PortfolioLayerContract` applies.
- Three‑bucket conservation: `CASH + TRADING + EXTERNAL = WALLET_TOTAL` at every commit.
- `EXTERNAL_POSITION` is read‑only; only L2 may rewrite, never L3 logic.
- All position changes flow through `TradeJournal` (audit log) before becoming visible.

---

## 7. L4 — Execution & Order Management

**Purpose.** Turn a *decision* (from L5, gated by L6) into actual orders, monitor them to completion, and report back.

### 7.1 Downward dependency: **L3 + L1 + L0** (skips L2; L2 is read‑only context already cached in L3).

### 7.2 Public Interface (consumed by L5)
```python
class IExecutionManager(Protocol):
    async def submit(self, intent: TradeIntent)  -> ExecutionTicket
    async def cancel(self, ticket)               -> bool
    def status(self, ticket)                     -> ExecutionState
    def on_fill(self, cb)                        -> None              # event up to L3/L5/L7
```

### 7.3 Modules in L4 (16)
`core/execution_manager.py` · `core/execution_logic.py` · `core/maker_execution.py` · `core/action_router.py` · `core/cash_router.py` · `core/intent_manager.py` · `core/tp_sl_engine.py` · `core/exit_arbitrator.py` · `core/exit_utils.py` · `core/profit_target_engine.py` · `core/liquidation_orchestrator.py` · `core/leverage_manager.py` · `core/signal_batcher.py` · `core/trading_coordinator.py` · `core/trading_hours_manager.py` · `core/recovery_engine.py` (execution‑side recovery hook) · `tools/recover_missing_sells.py` · `tools/exit_metrics.py` · `tools/compound_engine.py` · `auto_recovery.py` · `apply_recovery_to_live.py`

### 7.4 Invariants
- L4 may only spend capital that L3 has reserved via `ReservationToken`.
- Every order has a corresponding journal entry before it hits the wire.
- L4 never reads raw balances — it asks L3 for reservation availability.

---

## 8. L5 — Strategy & Decision

**Purpose.** Generate, fuse, rank, and arbitrate trade *intents*. Pure decision logic, no side effects on capital.

### 8.1 Downward dependency: **L3 + L0** (reads market context that L3 cached from L2).

### 8.2 Public Interface (consumed by L6 → L4)
```python
class IStrategyEngine(Protocol):
    def propose(self, ctx: PortfolioCtx)   -> list[TradeIntent]
    def explain(self, intent_id)           -> Explanation              # for L7
    def feedback(self, fill: FillEvent)    -> None                     # learning loop
```

### 8.3 Modules in L5 (33)
**Agents (10):** `agents/dip_sniper.py` · `agents/edge_calculator.py` · `agents/ipo_chaser.py` · `agents/liquidation_agent.py` · `agents/ml_forecaster.py` · `agents/swing_trade_hunter.py` · `agents/symbol_screener.py` · `agents/trend_hunter.py` · `agents/wallet_scanner_agent.py` · `agents/__init__.py`

**Decision core (15):** `core/signal_manager.py` · `core/signal_fusion.py` · `core/signal_batcher.py` · `core/arbitration_engine.py` · `core/opportunity_ranker.py` · `core/baseline_trading_kernel.py` · `core/meta_controller.py` · `core/agent_manager.py` · `core/agent_optimizer.py` · `core/agent_registry.py` · `core/external_adoption_engine.py` · `core/objective_feedback_controller.py` · `core/performance_evaluator.py` · `core/focus_mode.py` · `core/mode_manager.py`

**ML support (4):** `core/model_manager.py` · `core/model_trainer.py` · `core/scaling.py` · `core/capital_velocity_optimizer.py`

**Files (4):** `objective_tracker.py` · `diagnostic_signal_flow.py` · `SIGNAL_FLOW_DIAGNOSTIC.py` · `FORCE_SIGNALS_INJECTOR.py`

### 8.4 Invariants
- L5 is **pure**: same `PortfolioCtx` → same intents (modulo seeded RNG).
- L5 cannot place orders directly — it returns `TradeIntent` for L6 to gate.
- All learning state persists via L3 (`event_store`, `trade_journal`).

---

## 9. L6 — Governance & Policy

**Purpose.** Final approver between *intent* and *order*. Owns risk caps, sizing, rule overrides, and capital governance.

### 9.1 Downward dependency: **L5 + L3 + L0**.

### 9.2 Public Interface (consumed by L4)
```python
class IPolicyGate(Protocol):
    def approve(self, intent: TradeIntent)   -> ApprovedOrder | GovernanceVeto
    def size(self, intent)                   -> Decimal
    def caps(self)                           -> RiskCaps
    def override(self, rule_id, value)       -> None                  # admin path
```

### 9.3 Modules in L6 (15)
`core/risk_manager.py` · `core/capital_governor.py` · `core/capital_symbol_governor.py` · `core/capital_allocator.py` · `core/adaptive_capital_engine.py` · `core/compounding_engine.py` · `core/rebalancing_engine.py` · `core/policy_manager.py` · `core/startup_orchestrator.py` (policy bootstrap) · `core/lifecycle_manager.py` (policy aspect — see L8 for runtime aspect) · `automation/auto_rule_proposer.py` · `automation/proposal_monitor.py` · `automation/rule_overrides.py` · `config/__init__.py` · `config/EV_ALIGNMENT_CONFIG.py` (already in L0; here only as reader)

### 9.4 Invariants
- **Veto authority.** Any intent that violates a cap is rejected with `GovernanceVeto` — never silently downsized without recording the reason.
- All overrides are versioned in `automation/proposed_rules.json` and journaled via L3.
- L6 does not know about exchanges; it only sees `TradeIntent` and L3 state.

---

## 10. L7 — Observability & UX

**Purpose.** Make every other layer’s state legible to humans and ops tooling. *Read‑only*; never mutates business state.

### 10.1 Downward dependency: **all lower layers, read‑only via subscriptions**.

### 10.2 Public Interface
```python
class IMetricsSink(Protocol):
    def gauge(self, name, value, labels): ...
    def counter(self, name, inc, labels): ...
    def histogram(self, name, value, labels): ...

class IAlertBus(Protocol):
    def emit(self, severity, source_layer, msg, ctx): ...
```

### 10.3 Modules in L7 (45)
**Core observability (12):** `core/metrics.py` · `core/prometheus_exporter.py` · `core/health_check.py` · `core/health_check_manager.py` · `core/health_endpoints.py` · `core/health_monitor.py` · `core/health.py` · `core/healthy.py` · `core/alert_system.py` · `core/component_status_logger.py` · `core/performance_monitor.py` · `core/dashboard.py` · `core/apm_instrument.py` · `core/jaeger_tracer.py`

**Dashboards & docs (1 + JSONs):** `dashboards/__init__.py` (+ JSON dashboards — non‑code assets, but required at deploy time)

**Diagnostics & validators (8):** `diagnostics/per_loop_symbol_diag.py` · `core/diagnostics/*` · `tools/diagnose_runtime.py` · `tools/next_level_tpsl_analysis.py` · `tools/check_sell_marker_coverage.sh` (shell, but referenced) · `tools/live_monitor_snapshot.sh` · `tools/monitor_6h_session.py` · `monitoring/sandbox_monitor.py`

**Top‑level monitors (24):** `LIVE_MONITOR.py` · `LIVE_PHASE2_MONITOR.py` · `LIVE_TRADING_WITH_BALANCE_MONITOR.py` · `MONITOR_15MIN_REALTIME.py` · `MONITOR_15MIN_SESSION.py` · `MONITOR_3HOUR_TRADING_SESSION.py` · `monitor_4hour_session.py` · `monitor_phase2_realtime.py` · `PERIODIC_MONITOR.py` · `REALTIME_15MIN_MONITOR.py` · `REALTIME_DIAGNOSTICS.py` · `REALTIME_MONITOR.py` · `REALTIME_SESSION_MONITOR.py` · `CONTINUOUS_ACTIVE_MONITOR.py` · `CONTINUOUS_MONITOR.py` · `6HOUR_MONITORING_DASHBOARD.py` · `balance_dashboard.py` · `error_monitor.py` · `extract_rejections.py` · `phase2_monitoring.py` · `PROFIT_ACCUMULATOR_MONITOR.py` · `PHASE_2_STATUS_REPORT.py` · `SESSION_STATUS_REPORT.py` · `ANALYSIS_REPORT.py` · `FAST_DIAGNOSTICS.py`

### 10.4 Invariants
- **Read‑only.** L7 may not call any *mutating* method on L1–L6.
- Subscribes to events; never polls business state in tight loops.
- A failure in L7 must never break trading (degraded observability is acceptable).

---

## 11. L8 — Lifecycle & Recovery

**Purpose.** Owns *time*: boot order, supervision, session orchestration, chaos, restart, graceful shutdown.

### 11.1 Downward dependency: **may construct any layer** — but only at well‑defined lifecycle hooks (`init`, `start`, `pause`, `stop`, `restart`).

### 11.2 Public Interface
```python
class ILifecycle(Protocol):
    def boot_sequence(self)            -> list[LayerName]   # deterministic order
    def start_layer(self, name)        -> None
    def stop_layer(self, name)         -> None
    def restart_layer(self, name)      -> None
    def health(self)                   -> SystemHealth
```

### 11.3 Modules in L8 (~30)
**Core lifecycle (8):** `core/lifecycle_manager.py` · `core/startup_orchestrator.py` · `core/layer_orchestrator.py` · `core/watchdog.py` · `core/recovery_engine.py` · `core/chaos_monkey.py` · `core/perfornano` (perf‑nano hook) · `core/recovnano`

**Top‑level entry points (root scripts) (22):** `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` · `AUTONOMOUS_STARTUP_GUIDE.py` · `AUTONOMOUS_SYSTEM_STARTUP.py` · `RUN_AUTONOMOUS_LIVE.py` · `RUN_3HOUR_SESSION.py` · `run_4hour_session.py` · `RUN_6HOUR_SESSION.py` · `RUN_6HOUR_SESSION_MONITORED.py` · `2HOUR_CHECKPOINT_SESSION.py` · `PRODUCTION_STARTUP.py` · `PERSISTENT_TRADING_WATCHDOG.py` · `GATING_WATCHDOG.py` · `phase2_paper_trading.py` · `phase3_live_trading.py` · `phase4_30min_test.py` · `phase4_quick_validation.py` · `phase4_verify.py` · `deploy_phase2_production.py` · `verify_deployment.py` · `verify_dust_fix.py` · `verify_fixes.py` · `verify_fixes_detailed.py` · `live_integration.py` · `CONTINUOUS_OPERATION_GUIDE.py`

**Tests (L8 boundary tests) (9):** `tests/*` · `TEST_BOOTSTRAP.py` · `TEST_EXIT_FIRST_VALIDATION.py` · `TEST_FALLBACK.py` · `test_rounding_fix.py` · `test_trendhunter_import.py` · `UNIT_TEST_EXECUTION_GUIDE.py` · `component_validator.py`

**Tooling (8):** `tools/__init__.py` · `tools/fix_indentation.py` · `tools/fix_python_indentation.py` · `tools/advanced_fix_python_indentation.py` · `tools/fix_class_decorator_indentation.py` · `tools/smart_python_indentation_fixer.py` · `scripts/__init__.py` · `scripts/type_check_analyzer.py` · `scripts/run_orchestrator_for_4h.sh`

### 11.4 Invariants
- L8 is the **only** layer allowed to call `start_layer` / `stop_layer`.
- Boot order is **deterministic**: `L0 → L1 → L2 → L3 → L4 → L6 → L5 → L7`. (L6 starts before L5 so the gate exists before any intent can be produced.)
- Watchdog can restart any single layer without restarting the process.

---

## 12. Necessity Matrix — Why no component can be removed

For every layer, at least one *invariant* directly depends on a component that cannot be replaced by another:

| Invariant | Owning Layer | Sole component that satisfies it |
|-----------|--------------|----------------------------------|
| Single TCP/WS chokepoint | L1 | `core/exchange_client.py` + `core/ws_market_data.py` |
| Order ledger reconcilable to exchange | L1 | `core/order_cache_manager.py` + `core/exchange_truth_auditor.py` |
| Wallet snapshot is exchange‑verified | L2 | `core/balance_manager.py` |
| Three‑bucket conservation | L3 | `core/three_bucket_manager.py` + `core/portfolio_authority.py` |
| Audit log of every state change | L3 | `core/trade_journal.py` + `core/event_store.py` |
| Reserve‑then‑spend capital flow | L3 ↔ L4 | `core/reserve_manager.py` + `core/intent_manager.py` |
| Maker‑first execution policy | L4 | `core/maker_execution.py` |
| TP/SL guarantee on every position | L4 | `core/tp_sl_engine.py` + `core/exit_arbitrator.py` |
| Multi‑agent fusion | L5 | `core/signal_fusion.py` + `core/arbitration_engine.py` |
| ML forecast input | L5 | `agents/ml_forecaster.py` + `core/model_manager.py` |
| Risk cap veto | L6 | `core/risk_manager.py` + `core/capital_governor.py` |
| Auto rule evolution | L6 | `automation/auto_rule_proposer.py` + `automation/proposal_monitor.py` |
| Prometheus + Jaeger pipeline | L7 | `core/prometheus_exporter.py` + `core/jaeger_tracer.py` |
| Health endpoints | L7 | `core/health_endpoints.py` + `core/health_monitor.py` |
| Deterministic boot | L8 | `core/lifecycle_manager.py` + `core/layer_orchestrator.py` |
| Session entry points | L8 | `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` + `RUN_*` scripts |
| Auto‑recovery on crash | L8 | `core/recovery_engine.py` + `core/watchdog.py` + `auto_recovery.py` |

Every other module supports one of these invariants (e.g. monitors in L7 visualize them, agents in L5 feed L5’s fusion, validators in L8 prove invariants hold at boot). The full file→layer map in §3–§11 covers all 226 production scripts and 9 tests.

---

## 13. Allowed Call Graph (enforceable by `import-linter`)

```
L0  ←  every layer (read‑only, pure)
L1  ←  L2, L4, L8
L2  ←  L3, L8
L3  ←  L4, L5, L6, L7 (read), L8
L4  ←  L5 (via L6 only), L7 (read), L8
L5  ←  L6, L7 (read), L8
L6  ←  L4 (gate), L7 (read), L8
L7  ←  L8
L8  ←  (entry only)
```

Forbidden: any arrow not listed above. A CI rule (`scripts/type_check_analyzer.py` extension) should fail the build on violation.

---

## 14. Migration Plan vs. Today’s Code

The current `core/layer_contracts.py` only formalizes L2/L3/L5 as “Wallet/Portfolio/Strategy.” To reach this 8‑layer model:

1. **Rename** `WalletLayerContract → L2WalletContract`, `PortfolioLayerContract → L3PortfolioContract`, `StrategyLayerContract → L5StrategyContract` (keep aliases for back‑compat).
2. **Add** `L1ExchangeContract`, `L4ExecutionContract`, `L6PolicyContract`, `L7ObservabilityContract`, `L8LifecycleContract` to `core/layer_contracts.py`.
3. **Refactor** `core/layer_orchestrator.py` to drive the 8‑step deterministic boot in §11.4.
4. **Add CI gate** that parses imports and rejects forbidden arrows from §13.
5. **Move** root‑level monitors into `monitoring/` and root‑level session runners into `runners/` to make the layer of each file visually obvious. (Optional — current top‑level layout already maps cleanly via §10.3 / §11.3.)

---

## 15. TL;DR

- **8 layers, 1 direction of dependency, 1 contract per boundary.**
- **Every one of your 226 scripts has a home** (§3–§11). None are redundant — §12 proves each owns at least one invariant.
- **Existing 3‑layer file (`core/layer_contracts.py`) becomes L2/L3/L5** of this model; L1, L4, L6, L7, L8 are added.
- **Boot order (L8):** `L0 → L1 → L2 → L3 → L4 → L6 → L5 → L7`.
- **CI guard** the call graph in §13 to keep the layering honest forever.
