# OctiVault Trader — Complete System Architecture

**Version:** 1.0 | **Date:** 2026-05-04 | **Based on:** Verified source code cross-reference

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Layer Stack & Dependency Model](#2-layer-stack--dependency-model)
3. [Component Map (all modules per layer)](#3-component-map)
4. [End-to-End Data Flow](#4-end-to-end-data-flow)
5. [Exchange Integration](#5-exchange-integration)
6. [Signal Generation Pipeline](#6-signal-generation-pipeline)
7. [Capital Management Architecture](#7-capital-management-architecture)
8. [Execution Pipeline](#8-execution-pipeline)
9. [Risk & Governance Architecture](#9-risk--governance-architecture)
10. [State & Recovery Architecture](#10-state--recovery-architecture)
11. [Observability Architecture](#11-observability-architecture)
12. [Event Architecture](#12-event-architecture)
13. [ML Model Architecture](#13-ml-model-architecture)
14. [Boot Sequence](#14-boot-sequence)
15. [Trading Cycle (Per-Tick)](#15-trading-cycle-per-tick)
16. [Exit-First Strategy](#16-exit-first-strategy)
17. [Operational Topology](#17-operational-topology)

---

## 1. System Overview

OctiVault Trader is a fully autonomous cryptocurrency trading bot running on Binance Spot. It operates 24/7 using a multi-agent signal architecture fused with Keras ML forecasters, gated by a six-layer safety guard chain, and governed by a strict 60/20/20 capital allocation model.

**Runtime profile**

| Property | Value |
|----------|-------|
| Language | Python 3.9 |
| I/O model | Fully async — `asyncio` + `aiohttp` |
| Exchange | Binance Spot (REST + WebSocket) |
| Quote currency | USDT |
| Max active positions | 5 (configurable) |
| Capital strategy | 60 / 20 / 20 (compound / healing / buffer) |
| ML framework | Keras / TensorFlow 2.15 |
| Metrics | Prometheus (SafetyGuardMetrics) |
| Tracing | Jaeger distributed tracing |

**Operating modes**

| Mode | Trigger | Orders hit exchange? |
|------|---------|---------------------|
| Paper | `PAPER_TRADING=true` or sentinel keys `paper_key` / `paper_secret` | No — fills simulated |
| Testnet | `USE_TESTNET=true` | Yes — Binance testnet |
| Live | Signed credentials, neither flag set | Yes — real funds |

---

## 2. Layer Stack & Dependency Model

### 2.1 The 8-Layer Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│  L8  LIFECYCLE & RECOVERY    boot, watchdog, chaos, restart          │
├──────────────────────────────────────────────────────────────────────┤
│  L7  OBSERVABILITY & UX      dashboards, metrics, alerts, tracing    │
├──────────────────────────────────────────────────────────────────────┤
│  L6  GOVERNANCE & POLICY     risk caps, capital governor, approver   │
├──────────────────────────────────────────────────────────────────────┤
│  L5  STRATEGY & DECISION     agents, ML, signal fusion, arbitration  │
├──────────────────────────────────────────────────────────────────────┤
│  L4  EXECUTION & ORDER MGMT  order placement, TP/SL, retries, liq.  │
├──────────────────────────────────────────────────────────────────────┤
│  L3  PORTFOLIO & STATE       buckets, positions, rotation, journal   │
├──────────────────────────────────────────────────────────────────────┤
│  L2  WALLET & MARKET DATA    balance sync, OHLCV feeds, regimes      │
├──────────────────────────────────────────────────────────────────────┤
│  L1  EXCHANGE I/O            REST + WebSocket client, order cache    │
├──────────────────────────────────────────────────────────────────────┤
│  L0  CROSS-CUTTING           config, contracts, errors, shared state │
└──────────────────────────────────────────────────────────────────────┘
```

Every horizontal line is a **contract surface**. Code may only call downward through declared interfaces.

### 2.2 Allowed Dependencies (`src/l0_core/layer_contracts.py`)

```python
ALLOWED_DEPENDENCIES = {
    "L0": set(),                                         # pure utilities — no imports
    "L1": {"L0"},
    "L2": {"L0", "L1"},
    "L3": {"L0", "L2"},          # skips L1: exchange I/O is cached inside L3 state
    "L4": {"L0", "L1", "L3"},    # skips L2: market data read from L3, not live feeds
    "L5": {"L0", "L3"},          # pure decisions: zero I/O dependency
    "L6": {"L0", "L3", "L5"},    # read-only governance
    "L7": {"L0", "L1", "L2", "L3", "L4", "L5", "L6"},  # read-only observability
    "L8": {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"},
}
```

**Non-obvious skips**
- L3 skips L1: Portfolio never calls the exchange directly. All raw data has been normalised by L2 before L3 sees it.
- L4 skips L2: Execution reads market context from L3's cached state, not live feeds.
- L5 skips L1/L2/L4: Strategy is pure signal computation — same inputs always produce the same outputs.

### 2.3 Enforced call graph

```
L0  ←  all layers (read-only, pure imports)
L1  ←  L2, L4, L8
L2  ←  L3, L8
L3  ←  L4, L5, L6, L7 (read), L8
L4  ←  L6 (gate only), L7 (read), L8
L5  ←  L6, L7 (read), L8
L6  ←  L4 (gate), L7 (read), L8
L7  ←  L8
L8  ←  (entry point only)
```

### 2.4 Boot order

L6 starts **before** L5 so the policy gate always exists before any intent can be produced.

```
L0 → L1 → L2 → L3 → L4 → L6 → L5 → L7 → L8 (main loop)
```

---

## 3. Component Map

### L0 — Cross-Cutting Foundation
*Purpose: pure, side-effect-free building blocks. No I/O. No global mutation.*

| Module | Role |
|--------|------|
| `src/l0_core/config.py` | 150+ config keys, loaded from `.env` |
| `src/l0_core/contracts.py` | `TradeIntent`, `OrderSide`, type contracts |
| `src/l0_core/shared_state.py` | Central state store + event bus; all core enums |
| `src/l0_core/error_types.py` | Typed exception hierarchy (`OctiError` → all errors) |
| `src/l0_core/layer_contracts.py` | `ALLOWED_DEPENDENCIES`, CI-enforceable boundary |
| `src/l0_core/logger_utils.py` | Structured logging setup |
| `src/l0_core/metrics.py` | KPI counter/gauge wrappers |
| `src/l0_core/health.py` / `healthy.py` | `HealthCode` enum + status helpers |
| `src/l0_core/bounded_cache.py` | TTL cache with max-size bound |
| `src/l0_core/core_utils.py` | Common helpers |
| `src/l0_core/time_utils.py` | Timestamp parse/format |
| `src/l0_core/config_validator.py` | Config consistency checks at boot |
| `src/l0_core/error_handler.py` | Error recovery logic |
| `src/l0_core/stubs.py` | Fallback shims for optional deps |
| `src/l0_core/component_status_logger.py` | Per-component heartbeat logging |
| `utils/indicators.py` / `ta_indicators.py` | Technical indicator primitives |
| `utils/pnl_calculator.py` | P&L calculations |
| `utils/symbol_filter_pipeline.py` | Symbol screening helpers |
| `utils/volatility_adjusted_confidence.py` | Confidence scaling by regime |
| `config/EV_ALIGNMENT_CONFIG.py` | Expected-value alignment constants |

**Key enums (all in `shared_state.py`)**

```python
HealthCode:         OK | WARN | ERROR
DustClass:          TRADABLE | NEAR_DUST | DUST | RECOVERABLE_DUST | PERMANENT_WRITE_DOWN_DUST
PositionState:      ACTIVE | DUST_LOCKED | LIQUIDATING
AssetClassification: BOT_POSITION | EXTERNAL_POSITION | DUST | STABLE | RECOVERY
ExecutionResult:    FILLED | PARTIAL | REJECTED | BLOCKED
Component:          MARKET_DATA_FEED | EXECUTION_MANAGER | META_CONTROLLER |
                    AGENT_MANAGER | RISK_MANAGER | PNL_CALCULATOR |
                    PERFORMANCE_MON | APP_CONTEXT
```

---

### L1 — Exchange I/O
*Purpose: single chokepoint for every byte crossing the network.*

| Module | Role |
|--------|------|
| `src/l1_exchange/exchange_client.py` | All Binance REST + WebSocket; 3-tier user-data stream |
| `src/l1_exchange/order_cache_manager.py` | Local order ledger; reconcilable to exchange |
| `src/l1_exchange/exchange_truth_auditor.py` | Audits local order state vs. exchange truth |
| `src/l1_exchange/ws_market_data.py` | Market-data WebSocket feed (ticker, klines) |
| `src/l1_exchange/market_data_websocket.py` | Alternative WS implementation |
| `src/l1_exchange/polling_coordinator.py` | REST polling coordinator |
| `src/l1_exchange/balance_sync_backoff.py` | Balance sync with exponential backoff |
| `src/l1_exchange/retry_manager.py` | Exponential backoff + jitter for all API calls |

**WebSocket user-data stream — 3-tier fallback**

| Tier | URL | Auth | Notes |
|------|-----|------|-------|
| 1 | `wss://ws-api.binance.com:443/ws-api/v3` | HMAC signature (default) or Ed25519 session.logon | Set `BINANCE_API_TYPE=ED25519` for Ed25519 |
| 2 | `wss://stream.binance.com:9443/ws/{listenKey}` | listenKey in URL | Auto-fallback if Tier 1 fails |
| 3 | REST polling loop | `_user_data_polling_loop` | Last resort |

**Invariants**
- L1 translates network responses → typed L0 objects only. No business logic.
- All retries are handled here; higher layers receive either success or a typed `ExchangeError`.
- `OrderCacheManager` is the only component that may write local order state derived from exchange data.

---

### L2 — Wallet & Market Data
*Purpose: convert raw exchange streams into a clean, classified, time-synchronised world model.*

| Module | Role |
|--------|------|
| `src/l2_marketdata/market_data_feed.py` | OHLCV streaming; feeds `SharedState.market_data` |
| `src/l2_marketdata/balance_manager.py` | Live balance tracking; `WalletSnapshot` |
| `src/l2_marketdata/balance_sync.py` | Real-time balance update loop |
| `src/l2_marketdata/balance_cache_updater.py` | Caches balance snapshots |
| `src/l2_marketdata/volatility_regime.py` | Low / Normal / High regime from ATR |
| `src/l2_marketdata/market_regime_detector.py` | Overall market regime |
| `src/l2_marketdata/market_regime_integration.py` | Injects regime into trading logic |
| `src/l2_marketdata/nav_regime.py` | NAV-based regime: MICRO_SNIPER / STANDARD / MULTI_AGENT |
| `src/l2_marketdata/regime_proposal_analyzer.py` | Analyses proposed regime changes |
| `src/l2_marketdata/correlation_manager.py` | Tracks pairwise asset correlation matrix |
| `src/l2_marketdata/anomaly_detection.py` | Detects market data anomalies |
| `src/l2_marketdata/heartbeat.py` | Periodic system liveness signal |

**NAV Regimes**

| Regime | NAV range | Max positions | Confidence floor |
|--------|-----------|---------------|-----------------|
| `MICRO_SNIPER` | < $1 000 | 1 | 0.50 |
| `STANDARD` | $1 000 – $5 000 | 2 | 0.55 |
| `MULTI_AGENT` | > $5 000 | `MAX_ACTIVE_SYMBOLS` | 0.60 |

---

### L3 — Portfolio & State
*Purpose: authoritative registry of what the bot owns and why, segmented into three capital buckets.*

| Module | Role |
|--------|------|
| `src/l3_portfolio/portfolio_authority.py` | Single source of portfolio truth |
| `src/l3_portfolio/portfolio_manager.py` | Position management + dust classification |
| `src/l3_portfolio/position_manager.py` | Position create / update / close lifecycle |
| `src/l3_portfolio/three_bucket_manager.py` | 60/20/20 bucket accounting |
| `src/l3_portfolio/portfolio_buckets.py` | Bucket operations |
| `src/l3_portfolio/portfolio_balancer.py` | Rebalances allocations across buckets |
| `src/l3_portfolio/portfolio_target_size_enforcer.py` | Enforces per-position size limits |
| `src/l3_portfolio/portfolio_segmentation.py` | Segmentation analysis |
| `src/l3_portfolio/bucket_classifier.py` | Classifies positions into buckets |
| `src/l3_portfolio/symbol_manager.py` | Trading universe management |
| `src/l3_portfolio/symbol_rotation.py` | Symbol swap mechanics |
| `src/l3_portfolio/universe_rotation_engine.py` | Dynamic universe optimisation |
| `src/l3_portfolio/bootstrap_manager.py` | `BootstrapOrchestrator` — initial buys |
| `src/l3_portfolio/bootstrap_symbols.py` | Bootstrap symbol selection |
| `src/l3_portfolio/discovery_coordinator.py` | Symbol discovery coordination |
| `src/l3_portfolio/rotation_authority.py` | Authority decisions for rotation |
| `src/l3_portfolio/dead_capital_healer.py` | Recovers locked / dust capital |
| `src/l3_portfolio/reserve_manager.py` | Manages cash reserve allocations |
| `src/l3_portfolio/position_merger_enhanced.py` | Merges fragmented positions |
| `src/l3_portfolio/position_operation_validator.py` | Validates position operations |
| `src/l3_portfolio/holding_utility.py` | Holding age / utility analysis |
| `src/l3_portfolio/restart_position_classifier.py` | Classifies positions after restart |
| `src/l3_portfolio/state_manager.py` | Persists / loads portfolio state |
| `src/l3_portfolio/state_synchronizer.py` | Syncs local state with exchange |
| `src/l3_portfolio/event_store.py` | Event-sourcing log (all state changes) |
| `src/l3_portfolio/trade_journal.py` | Human-readable audit log of every trade |
| `src/l3_portfolio/replay_engine.py` | Replays event log for state reconstruction |

**L3 invariants**
- Three-bucket conservation: `COMPOUND + HEALING + BUFFER = deployable_NAV` at every commit.
- `EXTERNAL_POSITION` assets are read-only — only L2 may reclassify them.
- Every position change flows through `TradeJournal` before becoming visible to L4+.
- Capital reservation (`ReservationToken`) is the only mechanism by which L4 may spend L3 capital.

---

### L4 — Execution & Order Management
*Purpose: turn a gated decision into actual exchange orders, monitor them to completion.*

| Module | Role |
|--------|------|
| `src/l4_execution/execution_manager.py` | Main execution pipeline; `begin_execution_order_scope` guard |
| `src/l4_execution/execution_logic.py` | Core execution routing |
| `src/l4_execution/intent_manager.py` | Processes `TradeIntent` objects |
| `src/l4_execution/action_router.py` | Routes intents to appropriate handlers |
| `src/l4_execution/cash_router.py` | Resolves cash for buys |
| `src/l4_execution/trading_coordinator.py` | Coordinates multi-step trade operations |
| `src/l4_execution/maker_execution.py` | Maker-order (limit) execution strategy |
| `src/l4_execution/tp_sl_engine.py` | Take-profit / stop-loss monitoring + exit |
| `src/l4_execution/profit_target_engine.py` | Explicit profit-target exit management |
| `src/l4_execution/exit_arbitrator.py` | Decides which exit pathway fires first |
| `src/l4_execution/exit_utils.py` | Exit helper functions |
| `src/l4_execution/liquidation_orchestrator.py` | Coordinated position liquidations |
| `src/l4_execution/recovery_engine.py` | Stuck-order detection and re-placement |
| `src/l4_execution/leverage_manager.py` | Leverage control (capped at 1× for spot) |
| `src/l4_execution/signal_batcher.py` | Batches signals to reduce churn |
| `src/l4_execution/trading_hours_manager.py` | Enforces trading-hours policy |

**Order path guard**
```
ENFORCE_EXECUTION_MANAGER_PATH=True  (default)
ALLOW_UNSAFE_DIRECT_ORDER_PATH=True  (testing only — disables guard)
```
`begin_execution_order_scope()` is a context manager. Any call to `ExchangeClient.place_*` outside this scope raises at runtime when the guard is active.

---

### L5 — Strategy & Decision
*Purpose: generate, fuse, rank, and arbitrate trade intents. Pure logic — no side effects on capital.*

**Agents (10 active)**

| Agent | Strategy | Weight |
|-------|----------|--------|
| `TrendHunter` | MACD histogram crossover | 1.0× |
| `DipSniper` | RSI oversold + volume spike | 1.2× |
| `LiquidationAgent` | Liquidation-cascade momentum | 1.3× |
| `MLForecaster` | Keras per-symbol 5m model | 1.5× (highest) |
| `SymbolScreener` | Cross-symbol opportunity ranking | 0.8× |
| `IPOChaser` | New listing momentum | 0.9× |
| `WalletScannerAgent` | Wallet flow anomaly detection | 0.7× |
| `SwingTradeHunter` | Swing pattern recognition (ML) | 1.0× |
| `EdgeCalculator` | Edge scoring per signal | — |
| `BaselineTradingKernel` | MA crossover + volume rules | — |

**Decision modules**

| Module | Role |
|--------|------|
| `src/l5_strategy/signal_manager.py` | Signal cache, validation, deduplication |
| `src/l5_strategy/signal_fusion.py` | Composite-edge score fusion |
| `src/l5_strategy/arbitration_engine.py` | 6-gate arbitration (see §15) |
| `src/l5_strategy/opportunity_ranker.py` | Cross-symbol EV ranking |
| `src/l5_strategy/agent_manager.py` | Agent lifecycle coordination |
| `src/l5_strategy/agent_optimizer.py` | Rolling performance-based weight tuning |
| `src/l5_strategy/agent_registry.py` | Agent discovery and registration |
| `src/l5_strategy/model_manager.py` | ML model loading + warm-up |
| `src/l5_strategy/model_trainer.py` | Online retraining on new candles |
| `src/l5_strategy/mode_manager.py` | Strategy mode switching |
| `src/l5_strategy/performance_evaluator.py` | Per-agent win-rate + PnL factor |
| `src/l5_strategy/focus_mode.py` | Focus portfolio on highest-EV symbols |
| `src/l5_strategy/capital_velocity_optimizer.py` | Capital turnover optimisation |
| `src/l5_strategy/objective_feedback_controller.py` | Feedback loop for continuous improvement |
| `src/l5_strategy/external_adoption_engine.py` | Pluggable external signal sources |

**Composite edge score**
```
composite_edge = Σ (agent_signal × agent_weight × confidence_adjustment)
  BUY  if composite_edge >=  0.35
  SELL if composite_edge <= -0.35
  HOLD otherwise
```
Then filtered by MetaController confidence gate (≥ 0.89 in current tuning — rejects ~95% of signals).

---

### L6 — Governance & Policy
*Purpose: final approver between intent and order. Owns all risk caps.*

| Module | Role |
|--------|------|
| `src/l6_governance/risk_manager.py` | Risk limit enforcement |
| `src/l6_governance/capital_allocator.py` | Capital allocation planning |
| `src/l6_governance/capital_governor.py` | Aggregate capital exposure limits |
| `src/l6_governance/capital_symbol_governor.py` | Per-symbol capital limits |
| `src/l6_governance/policy_manager.py` | Higher-level trading rules |
| `src/l6_governance/rebalancing_engine.py` | Triggered portfolio rebalancing |
| `src/l6_governance/scaling.py` | Dynamic position sizing |
| `src/l6_governance/adaptive_capital_engine.py` | Adapts allocation as NAV changes |
| `src/l6_governance/compounding_engine.py` | 60/20/20 profit reinvestment |

**Safety guard chain** (every intent passes through all 6 in sequence — any rejection halts the intent)

| Guard # | Guard | Rejection outcome |
|---------|-------|-------------------|
| 1 | Balance validation — free USDT ≥ required | `REJECTED` |
| 2 | Leverage validation — leverage ≤ MAX_LEVERAGE (1×) | `REJECTED` |
| 3 | Trading hours — within allowed window, no maintenance | `SKIP` |
| 4 | Anomaly detection — signal not a statistical outlier | `QUARANTINED` |
| 5 | Correlation guard — portfolio concentration ≤ limit | `REJECTED` |
| 6 | Capital adequacy — `HealthCode.ERROR` on any component | `HALT` |

---

### L7 — Observability & UX
*Purpose: make every layer's state legible. Read-only — never mutates business state.*

| Module | Role |
|--------|------|
| `src/l7_observability/prometheus_exporter.py` | `SafetyGuardMetrics` — 20 Prometheus metrics |
| `src/l7_observability/health_monitor.py` | Aggregates `Component` → system `HealthCode` |
| `src/l7_observability/health_check.py` | Individual component health checks |
| `src/l7_observability/health_check_manager.py` | Coordinates all health checks |
| `src/l7_observability/health_endpoints.py` | HTTP `/health`, `/ready` endpoints |
| `src/l7_observability/performance_monitor.py` | Sharpe, Sortino, return % tracking |
| `src/l7_observability/nav_attribution_monitor.py` | Attributes NAV changes to sources |
| `src/l7_observability/alert_system.py` | Webhook / email alerts |
| `src/l7_observability/dashboard.py` | Real-time web dashboard |
| `src/l7_observability/jaeger_tracer.py` | Distributed trace spans per `trace_id` |
| `src/l7_observability/apm_instrument.py` | APM instrumentation hooks |

**Prometheus metrics (SafetyGuardMetrics)**

| Metric | Type | Labels |
|--------|------|--------|
| `balance_validation_total` | Counter | symbol, status |
| `leverage_validation_total` | Counter | symbol, status |
| `hours_validation_total` | Counter | symbol, status |
| `anomaly_detection_total` | Counter | status |
| `correlation_validation_total` | Counter | symbol, status |
| `trades_executed_total` | Counter | symbol, side, guard_status |
| `balance_validation_approval_rate` | Gauge | — |
| `leverage_validation_approval_rate` | Gauge | — |
| `anomaly_detection_rate` | Gauge | — |
| `overall_approval_rate` | Gauge | — |
| `active_positions_count` | Gauge | — |
| `max_concentration_ratio` | Gauge | — |
| `current_max_leverage` | Gauge | — |
| `execution_latency_seconds` | Histogram | — |
| `guard_latency_seconds` | Histogram | — |
| `*_latency_seconds` (per guard) | Histogram | — |

---

### L8 — Lifecycle & Recovery
*Purpose: owns time — boot order, supervision, session orchestration, restart, graceful shutdown.*

| Module | Role |
|--------|------|
| `src/l8_lifecycle/meta_controller.py` | Main trading loop; fires intents to L6 |
| `src/l8_lifecycle/lifecycle_manager.py` | Layer start / stop / restart |
| `src/l8_lifecycle/startup_orchestrator.py` | 10-step boot sequence (see §14) |
| `src/l8_lifecycle/watchdog.py` | Health monitor; can restart any layer in isolation |
| `src/l8_lifecycle/fourth_slot_tracker.py` | Tracks the aggressive 4th position slot |
| `src/l8_lifecycle/chaos_monkey.py` | Injects deliberate faults for resilience testing |
| `🎯_MASTER_SYSTEM_ORCHESTRATOR.py` | Primary entry point — launches all layers |
| `launch_with_monitor.py` | Launch with real-time monitoring |

---

## 4. End-to-End Data Flow

```
EXCHANGE (Binance)
│
├── WebSocket market data ─────────────────────────────────────────────┐
│   ticker / kline events                                              │
│                                                                      ▼
│                                                            L2: MarketDataFeed
│                                                            └─ SharedState.prices
│                                                            └─ SharedState.market_data
│                                                            └─ VolatilityRegime.regime
│
├── WebSocket user-data ────────────────────────────────────────────────┐
│   executionReport / balanceUpdate / outboundAccountPosition           │
│                                                                       ▼
│                                                             L1: ExchangeClient
│                                                             └─ _ingest_user_data_ws_payload()
│                                                             └─ loop.create_task(emit_event(...))
│                                                             └─ SharedState.emit_event()
│                                                                       │
│                                                               ┌───────┴──────────┐
│                                                               ▼                  ▼
│                                                        L3: PositionManager  L2: BalanceManager
│
└── REST API (polling / on-demand)
    account, balances, order status, exchange info
    ▼
    L1: ExchangeClient → L2: BalanceManager → L3: PortfolioAuthority


SIGNAL GENERATION
│
├── L2 sends RegimeLabel ──────────────────────────────────────────────┐
│                                                                      │
├── L3 sends PortfolioCtx ──────────────────────────────────────────── ▼
│                                                               L5: AgentManager
│                                                               ├─ 10 agents generate signals
│                                                               └─ SignalManager caches + deduplicates
│                                                                       │
│                                                               L5: SignalFusion
│                                                               └─ composite_edge score
│                                                                       │
│                                                               L5: ArbitrationEngine
│                                                               └─ 6 gates → TradeIntent[]
│                                                                       │
│                                                               L6: PolicyGate.approve()
│                                                               └─ 6 safety guards
│                                                               └─ ApprovedOrder | GovernanceVeto
│                                                                       │
│                                                               L8: MetaController
│                                                               └─ stamps trace_id + tier + planned_quote
│                                                                       │
EXECUTION                                                               │
│                                                                       ▼
├────────────────────────────────────────────────────────  L4: ExecutionManager
│                                                          ├─ begin_execution_order_scope()
│                                                          ├─ L3.reserve(qty, reason) → ReservationToken
│                                                          ├─ L1.place_market_buy/sell()
│                                                          ├─ OrderCacheManager.track(order)
│                                                          └─ EventStore.record(fill)
│                                                                       │
│                                                          L3: PositionManager.update(fill)
│                                                          L7: metrics.trades_executed.inc()
│                                                          L7: jaeger_tracer.finish_span(trace_id)
│
└── FILL CONFIRMED ──────────────────────────────────────────────────────
    SharedState.emit_event("fill", payload)
    → L3 updates position registry
    → L2 triggers balance re-sync
    → L7 records P&L attribution
    → L5 receives feedback for agent learning
```

---

## 5. Exchange Integration

### REST API calls

| Operation | Endpoint | Layer |
|-----------|----------|-------|
| Place market order | `POST /api/v3/order` | L4 via L1 |
| Cancel order | `DELETE /api/v3/order` | L4 via L1 |
| Query order status | `GET /api/v3/order` | L4 via L1 |
| Account balances | `GET /api/v3/account` | L2 via L1 |
| Ticker price | `GET /api/v3/ticker/price` | L2 via L1 |
| 24hr stats | `GET /api/v3/ticker/24hr` | L5 via L1 |
| Klines / OHLCV | `GET /api/v3/klines` | L2 via L1 |
| Exchange info (filters) | `GET /api/v3/exchangeInfo` | L1 (cached) |
| listenKey create | `POST /api/v3/userDataStream` | L1 |
| listenKey keepalive | `PUT /api/v3/userDataStream` | L1 |

### WebSocket event routing

`_ingest_user_data_ws_payload` dispatches via `loop.create_task(SharedState.emit_event(...))`:

| WS event | `emit_event` name | Consumer |
|----------|-------------------|----------|
| `executionReport` | `order_update` | L4 RecoveryEngine, L3 PositionManager |
| `balanceUpdate` | `balance_update` | L2 BalanceManager |
| `outboundAccountPosition` | `account_position` | L2 BalanceManager |
| `listStatus` | `list_status` | L4 ExecutionManager |
| `listenKeyExpired` | `listen_key_expired` | L1 (triggers Tier 2 fallback) |

Dispatch is always `loop.create_task(...)` — never `await` directly from the WS handler, which would block the event loop and drop subsequent events.

### Rate limiting

`RetryManager` (L1): exponential backoff + jitter on all API calls.
- Binance `429` → wait `Retry-After` header value
- After 5 consecutive `429`s → circuit breaker opens, `ExecutionManager` halts for 10 min

---

## 6. Signal Generation Pipeline

### Volume

At peak, the pipeline processes approximately:

```
8 agents × 10 symbols × 60 cycles/min = ~5,000 signal evaluations/session
↓
SignalManager deduplication + age filter (< 60 s)
↓
SignalFusion composite_edge scoring
↓
ArbitrationEngine 6-gate filter
↓
MetaController confidence gate (≥ 0.89) → ~5% pass rate → ~1 approved intent per 20 s
```

### Agent signal schema

```python
{
    "symbol":     str,          # e.g. "BTCUSDT"
    "action":     str,          # "BUY" | "SELL" | "HOLD"
    "confidence": float,        # 0.0–1.0 per agent
    "reason":     str,
    "agent":      str,
    "ts_ms":      int,
    # optional
    "entry_price":  float,
    "stop_loss":    float,
    "take_profit":  float,
    "expected_move": float,     # expected alpha % used in Gate 4 economic check
}
```

### Fusion formula

```
weighted_signal = Σ (signal_i × weight_i × perf_score_i)
agent_perf_score = rolling_win_rate × pnl_factor   (updated by ObjectiveFeedbackController)

composite_edge ≥  0.35 → BUY intent
composite_edge ≤ -0.35 → SELL intent
```

### Agent weights (current tuning)

| Agent | Weight | Rationale |
|-------|--------|-----------|
| MLForecaster | 1.5× | Highest historical accuracy |
| LiquidationAgent | 1.3× | High-conviction event-driven |
| DipSniper | 1.2× | Strong mean-reversion edge |
| TrendHunter | 1.0× | Baseline |
| SwingTradeHunter | 1.0× | Pattern-based |
| IPOChaser | 0.9× | Higher variance |
| SymbolScreener | 0.8× | Macro filter, not entry signal |
| WalletScannerAgent | 0.7× | Weak signal, high noise |

---

## 7. Capital Management Architecture

### Three-Bucket Model

```
Total Deployable USDT (NAV − reserve)
│
├── 60%  COMPOUND BUCKET
│         └─ Top 3 positions by EV score
│         └─ Best compounding candidates
│         └─ Managed by ThreeBucketManager + CompoundingEngine
│
├── 20%  HEALING BUCKET
│         └─ Recovery trades on positions in drawdown
│         └─ Dead capital healer operations
│         └─ Managed by DeadCapitalHealer + RebalancingEngine
│
└── 20%  BUFFER BUCKET
          └─ Emergency liquidity
          └─ Source for 4th Slot (aggressive position)
          └─ Managed by ReserveManager
```

**Conservation invariant**: `COMPOUND + HEALING + BUFFER = deployable_NAV` at every state commit.

**Rebalancing triggers**: NAV change > 5% | position open/close | daily at configured time.

### 4th Slot (aggressive profit hunting)

The 4th slot runs in parallel with the 3 compound positions:

| Property | Value |
|----------|-------|
| Capital source | BUFFER bucket |
| Entry trigger | Highest-confidence arbitration signal |
| Take-profit | +15% (`FIX8_4TH_SLOT_PROFIT_TARGET_PCT = 0.15`) |
| Stop-loss | −3% (`FIX8_4TH_SLOT_STOP_LOSS_PCT = -0.03`) |
| Max hold | 2 hours (`FIX8_4TH_SLOT_MAX_HOLD_MINUTES = 120`) |
| Exit: first of | TP fires \| SL fires \| timeout |
| Tracker | `FourthSlotTracker` |

### Capital reservation flow (reserve-then-spend)

```
L5 proposes TradeIntent (quote = desired USDT)
  ↓
L6 CapitalGovernor.approve()  →  checks bucket has free capital
  ↓
L3 ReserveManager.reserve(symbol, qty, reason)  →  returns ReservationToken
  ↓
L4 ExecutionManager.place_order()  →  spends token on wire
  ↓
On fill: L3 PositionManager records position, releases token
On failure: token released, capital returned to bucket
```

L4 never reads raw balances — it only spends reserved capital via `ReservationToken`.

### Dust lifecycle

```
TRADABLE ──→ (price falls) ──→ NEAR_DUST
NEAR_DUST ──→ (price falls further) ──→ DUST
DUST ──→ (recovery possible) ──→ RECOVERABLE_DUST
DUST ──→ (economically stuck) ──→ PERMANENT_WRITE_DOWN_DUST

DeadCapitalHealer sweep:
  DUST + RECOVERABLE_DUST → attempt market sell → capital freed
  PERMANENT_WRITE_DOWN_DUST → write off, no action
```

`DeadCapitalHealer` runs every `DEAD_CAPITAL_HEAL_INTERVAL_SEC`, up to `max_liquidations=50` per sweep cycle.

---

## 8. Execution Pipeline

### Full order lifecycle

```
TradeIntent (from L5, stamped by L8)
  │
  ├─ L6 PolicyManager.check()         # trading hours, churn cooldown, regime
  ├─ L6 RiskManager.validate()        # size, notional, drawdown
  ├─ L6 CapitalAllocator.approve()    # bucket has capital
  ├─ L3 ReserveManager.reserve()      # token issued
  │
  └─ L4 ExecutionManager.place_order()
         begin_execution_order_scope()
         ├─ ExchangeClient.place_market_buy/sell()
         ├─ OrderCacheManager.track(order)    # idempotency check on trace_id
         └─ EventStore.record(fill)
         │
         On fill confirmed:
         ├─ PositionManager.update(fill)
         ├─ ReserveManager.release(token)
         ├─ TPSLEngine.register(position)    # register exit pathways
         ├─ SharedState.emit_event("fill")
         └─ metrics.trades_executed.labels(...).inc()
```

### TP/SL engine

`TPSLEngine` monitors every open position on each tick:

| Exit type | Trigger | Action |
|-----------|---------|--------|
| Take-profit | `current_price ≥ entry × (1 + tp_pct)` | market SELL |
| Stop-loss | `current_price ≤ entry × (1 - sl_pct)` | market SELL |
| Trailing stop | `current_price ≤ peak_price × (1 - trail_pct)` | market SELL |
| Time exit | `position_age > max_hold_minutes` | market SELL |

### Idempotency guarantee

Every order is tagged with `trace_id` (UUID4). `OrderCacheManager` rejects duplicate `trace_id` submissions. `sell_finalize` with the same `trace_id` is a no-op.

### Stuck order recovery

```
RecoveryEngine monitors orders:
1. Poll status after ORDER_STUCK_TIMEOUT_SEC
2. If still NEW or PARTIALLY_FILLED → cancel
3. If cancellation succeeds → re-place with new trace_id
4. After 3 failed recoveries → mark for manual review, emit alert
```

---

## 9. Risk & Governance Architecture

### Safety guard chain (in sequence, any veto halts the intent)

```
TradeIntent
  │
  ▼ GUARD 1: Balance validation
  │   free_USDT ≥ quote_required?  →  APPROVED / REJECTED_INSUFFICIENT
  │
  ▼ GUARD 2: Leverage validation
  │   leverage ≤ MAX_LEVERAGE (1×)?  →  APPROVED / REJECTED_OVER_LEVERAGE
  │
  ▼ GUARD 3: Trading hours
  │   within allowed window, not in maintenance?  →  ALLOWED / REJECTED_MARKET_CLOSED
  │
  ▼ GUARD 4: Anomaly detection
  │   signal not statistical outlier?  →  ACCEPTED / QUARANTINED
  │
  ▼ GUARD 5: Correlation guard
  │   concentration_ratio ≤ MAX_CONCENTRATION?  →  APPROVED / REJECTED_CONCENTRATION
  │
  ▼ GUARD 6: Capital adequacy / system health
      HealthCode ≠ ERROR on tracked components?  →  PROCEED / HALT
      │
      ▼
    ApprovedOrder (reaches L4)
```

### Risk limits

| Limit | Key | Scope |
|-------|-----|-------|
| Max drawdown per day | `MAX_DAILY_DRAWDOWN_PCT` | Portfolio |
| Max position size | `MAX_POSITION_SIZE_PCT` | Per symbol |
| Max open positions | `MAX_ACTIVE_SYMBOLS` | Portfolio |
| Min order size | `MIN_NOTIONAL_USDT` | Per order |
| Max leverage | `MAX_LEVERAGE = 1` | Spot only |

### Auto-rule evolution

`AutoRuleProposer` (L6) analyses recent trade outcomes and proposes parameter adjustments (confidence thresholds, allocation percentages). Changes are stored in `automation/proposed_rules.json`, reviewed by `ProposalMonitor`, and applied via `rule_overrides.py` — all journaled in L3.

---

## 10. State & Recovery Architecture

### Persistent state files

```
state/
├── operational_state.json    # current phase, active processes, last activity
├── session_memory.json       # session ID, task history, error log
├── checkpoint.json           # full portfolio + capital snapshot
├── recovery_state.json       # recovery status and pending actions
└── context.json              # full context for agent reconstruction
```

### Startup recovery sequence (L8 StartupOrchestrator)

```
_step_recovery_engine_rebuild()
  └─ RecoveryEngine reads EventStore log → rebuilds position registry

_step_hydrate_positions()
  └─ SharedState loads positions from exchange balance snapshot

_step_clear_stale_quote_reservations()
  └─ Releases any orphaned ReservationTokens from previous session

_step_liquidate_legacy_positions()
  └─ Sells positions not in current strategy universe

_step_aggressive_exchange_liquidation()
  └─ Force-exits any orders stuck in NEW/PARTIALLY_FILLED state

_step_auditor_restart_recovery()
  └─ ExchangeTruthAuditor syncs open orders vs. local OrderCache

_step_build_capital_ledger()
  └─ Reconstructs 60/20/20 ledger from fresh wallet snapshot

_step_verify_capital_integrity()
  └─ Assert: computed NAV == exchange-reported NAV (no double-count)

_emit_state_rebuilt_event()
  └─ Broadcasts "portfolio_ready" to all L3/L4/L5 subscribers

_emit_startup_ready_event()
  └─ System enters live trading mode
```

### Recovery invariants

- System never enters trading mode until `_step_verify_capital_integrity()` passes.
- A corrupt `state/` directory is not fatal — `RecoveryEngine` rebuilds from `EventStore` log.
- If `EventStore` is also corrupted, the system rebuilds from exchange truth via `ExchangeTruthAuditor` (no local history required).

---

## 11. Observability Architecture

### Health model

```
Component enum member → HealthCode (OK | WARN | ERROR)
                              │
                       HealthMonitor aggregates
                              │
                       System HealthCode
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
      HTTP /health      Alert system     MetaController
      endpoint          (webhook/email)  halts trading
                                         if ERROR
```

### Distributed tracing

Every `TradeIntent` carries a `trace_id` (UUID4) stamped by `MetaController`. Jaeger spans are emitted at:
- Signal receipt
- Arbitration decision
- Policy gate result
- Order placement
- Fill confirmed

This gives a complete end-to-end trace per trade.

### Dashboard components

| Component | Description |
|-----------|-------------|
| Real-time NAV | Live portfolio value |
| Position table | All open positions with P&L |
| Signal heatmap | Per-agent, per-symbol confidence |
| Guard approval rates | Each guard's pass/reject rate |
| Health status | Per-component `HealthCode` |
| Trade log | Rolling fill history |
| Regime indicator | Current NAV regime |

---

## 12. Event Architecture

`SharedState` is the system's event bus. All cross-layer notifications flow through it.

### Subscription model

```python
# Subscribe
shared_state.subscribe_events("fill", handler_fn)

# Publish (always non-blocking)
loop.create_task(shared_state.emit_event("fill", payload))
# Never: await shared_state.emit_event(...)  ← blocks WS handler → drops events
```

### Event catalogue

| Event name | Producer | Consumers |
|------------|----------|-----------|
| `order_update` | L1 WS handler | L4 RecoveryEngine, L3 PositionManager |
| `balance_update` | L1 WS handler | L2 BalanceManager |
| `account_position` | L1 WS handler | L2 BalanceManager |
| `fill` | L4 ExecutionManager | L3 PositionManager, L2 BalanceManager, L5 FeedbackController, L7 Metrics |
| `position_opened` | L3 PositionManager | L4 TPSLEngine, L7 Dashboard |
| `position_closed` | L3 PositionManager | L7 NAV attribution, L5 Feedback |
| `listen_key_expired` | L1 WS handler | L1 (triggers Tier 2 fallback) |
| `portfolio_ready` | L8 StartupOrchestrator | L4, L5, L6 (unlock trading) |
| `health_change` | L7 HealthMonitor | L8 MetaController (halt/resume) |
| `dust_detected` | L3 PortfolioManager | L3 DeadCapitalHealer |

---

## 13. ML Model Architecture

### Model inventory

- **Format**: `.keras` model + `.pkl` StandardScaler per symbol
- **Location**: `src/models/`
- **Count**: 50+ per-symbol models
- **Symbols**: AAVEUSDT, ADAUSDT, APTUSDT, ARBUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOTUSDT, ETHUSDT, LINKUSDT, MATICUSDT, NEARUSDT, SOLUSDT, UNIUSDT, XRPUSDT … (full list in `src/models/`)

### Input pipeline

```
OHLCV candles (5m timeframe, last 150 bars)
  │
  ├─ Derived features:
  │   EMA-9, EMA-21, ATR-14, RSI-14, MACD(12,26,9), volume z-score
  │
  ├─ Normalised using stored StandardScaler (per symbol)
  │
  └─ Shape: (1, 150, n_features)  →  Keras model
```

### Model output

```python
{
    "action":        "BUY" | "SELL" | "HOLD",
    "confidence":    float,   # 0.0–1.0
    "horizon_bars":  int,     # expected hold duration in 5m bars
}
```

### Online retraining

`ModelTrainer` triggers retraining when:
- `ONLINE_TRAINING=true` in config
- ≥ 500 new candles accumulated since last train
- Retraining runs in a background thread; `ModelManager` hot-swaps the model on completion
- Old model is kept as fallback for 1 retraining cycle

---

## 14. Boot Sequence

### Phase 1 — Layer initialisation (ordered by dependency)

```
L0  Config.load()  →  Config.validate()
L0  SharedState.__init__()
L1  ExchangeClient.__init__()  →  REST auth check  →  exchangeInfo cache
L2  BalanceManager.start()    →  first snapshot
L2  MarketDataFeed.start()    →  WebSocket subscriptions
L3  EventStore.open()
L3  PortfolioAuthority.__init__()
L4  ExecutionManager.__init__()
L6  RiskManager.__init__()
L6  CapitalAllocator.__init__()
L5  ModelManager.load_models()    (warm-up: all 50+ Keras models)
L5  AgentManager.__init__()       (agents registered)
L7  HealthMonitor.start()
L7  PrometheusExporter.start()    (HTTP /metrics endpoint)
L7  health_endpoints.start()      (HTTP /health, /ready)
```

### Phase 2 — State recovery (StartupOrchestrator)

```
_step_recovery_engine_rebuild()
_step_hydrate_positions()
_step_clear_stale_quote_reservations()
_step_liquidate_legacy_positions()
_step_aggressive_exchange_liquidation()
_step_auditor_restart_recovery()
_step_build_capital_ledger()
_step_verify_capital_integrity()          ← system halts here if NAV mismatch
_emit_state_rebuilt_event()
_emit_startup_ready_event()
```

### Phase 3 — Main trading loop (MetaController)

System enters the per-tick trading cycle described in §15.

---

## 15. Trading Cycle (Per-Tick)

`MetaController` runs a tight async loop. Each iteration:

```
[1] Tick counter + cycle timer

[2] Drain market events
    Process: price updates, fills, position snapshots, metric recalcs

[3] Guard evaluation (short-circuits on first failure)
    Guard 1: Market data fresh?          price age < 5 s          → else SKIP
    Guard 2: Balances available?         USDT > 0, free > MIN     → else RECOVER
    Guard 3: Ops plane ready?            ExchangeClient healthy    → else HALT
    Guard 4: Trading hours valid?        within window             → else SKIP
    Guard 5: Position constraints met?   open < regime_max         → else SKIP
    Guard 6: Capital adequate?           free ≥ MIN_CAPITAL        → else RECOVER

[4] Signal intake
    AgentManager.tick() → signals from all active agents
    SignalManager: validate confidence ≥ 0.50, age < 60 s, deduplicate

[5] Batch + sort
    Collect up to 50 signals, sort by composite_edge descending

[6] Per-signal arbitration (ArbitrationEngine)
    Gate 1: Lifecycle state    no ROTATION_PENDING / DUST_HEALING conflict
    Gate 2: Portfolio health   position count < regime_max, dust ratio < limit
    Gate 3: Capital            free_quote = (balance − reserve) − allocated ≥ MIN_ENTRY_QUOTE
    Gate 4: Economic (anti-churn)
             round_trip_cost ≈ 0.50%  (2 × taker_fee + 2 × slippage)
             expected_alpha > round_trip_cost + 0.05%  → pass
    Gate 5: Confidence         MICRO_SNIPER ≥ 0.50 | STANDARD ≥ 0.55 | MULTI_AGENT ≥ 0.60
    Gate 6: Regime             MICRO_SNIPER: max 1 pos, no rotation, no dust healing
                                STANDARD: max 2 pos
                                MULTI_AGENT: up to MAX_ACTIVE_SYMBOLS

[7] MetaController confidence gate
    composite_edge confidence ≥ 0.89 (current tuning)
    Rejects ~95% of intents at this stage

[8] MetaController stamps TradeIntent
    intent.trace_id = UUID4
    intent.tier = "BOT_POSITION" | "RECOVERY" | "DUST_RECOVERY"
    intent.planned_quote = resolved from CapitalAllocator

[9] L6 safety guard chain (§9)
    6 guards in sequence → ApprovedOrder or veto

[10] L4 ExecutionManager.place_order()
     → exchange wire → fill → position update → metrics
```

---

## 16. Exit-First Strategy

Every entry is blocked until three exit pathways are pre-registered. This prevents the deadlock pattern where capital becomes permanently locked in deteriorating positions.

### Pre-entry checklist (enforced in ExecutionManager)

```
Before any BUY order fires:
1. planned_quote set on TradeIntent           (capital committed)
2. TPSLEngine.register(symbol, tp, sl)        (price exits)
3. FourthSlotTracker or PositionManager       (time exit registered)

If any of 1/2/3 fails → BUY rejected
```

### Exit pathway priority

```
ExitArbitrator evaluates on each tick:

1. TP price hit?    → immediate market SELL (profit locked)
2. SL price hit?    → immediate market SELL (loss limited)
3. Trailing stop?   → market SELL (if configured)
4. Time exceeded?   → market SELL (prevents indefinite holding)
5. Dust detected?   → DeadCapitalHealer handles separately
```

### Capital flow guarantee

```
Entry → capital reserved (ReservationToken)
  ↓
Position live → capital tied to position
  ↓
Exit fires (TP/SL/time/dust) → market SELL executed
  ↓
Fill confirmed → USDT returned to bucket → available for next trade
  ↓
NO EXIT possible → system blocked from this capital forever (prevented by pre-entry checks)
```

---

## 17. Operational Topology

### Process model

```
Single Python process
├── asyncio event loop (main thread)
│   ├── MetaController (trading loop)
│   ├── MarketDataFeed (WS coroutine)
│   ├── ExchangeClient (WS coroutines)
│   ├── BalanceManager (sync loop)
│   ├── HealthMonitor (check loop)
│   └── HTTP endpoints (aiohttp server)
│
└── Background threads
    ├── ModelTrainer (online retraining — runs in ThreadPoolExecutor)
    └── Logger (file handlers)
```

### Deployment configuration

```
Recommended launch:
  ./start_trading_with_monitoring.sh

Direct launch:
  python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py --duration 24
  python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py --paper --duration 8
```

### Environment hierarchy

```
.env.template    (canonical reference — all keys documented)
.env             (active config — loaded at runtime)
.env.bak         (auto-backup before major changes)
.env.growth      (growth-mode override profile)
```

### State directory

```
state/                  (runtime state — survives restarts)
logs/                   (structured logs — trading.log, ws_events.log, health.log, pnl.log)
src/models/             (Keras .keras + .pkl files — 50+ symbol models)
snapshots/              (periodic portfolio snapshots)
```

---

*End of OctiVault Trader System Architecture v1.0*
