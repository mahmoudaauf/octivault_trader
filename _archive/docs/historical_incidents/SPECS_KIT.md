# OctiVault Trader — Specs Kit

**Version:** 1.1 | **Date:** 2026-05-04 | **Author:** Verified against source — cross-checked all enums, layer deps, metrics, boot sequence

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Specification](#2-architecture-specification)
3. [Layer Contracts](#3-layer-contracts)
4. [Core Data Models](#4-core-data-models)
5. [Configuration Reference](#5-configuration-reference)
6. [Exchange Integration](#6-exchange-integration)
7. [Strategy & Signal Pipeline](#7-strategy--signal-pipeline)
8. [Portfolio & Capital Management](#8-portfolio--capital-management)
9. [Execution Pipeline](#9-execution-pipeline)
10. [Governance & Risk](#10-governance--risk)
11. [Observability & Health](#11-observability--health)
12. [Testing Specification](#12-testing-specification)
13. [Deployment & Operations](#13-deployment--operations)
14. [ML Model Specification](#14-ml-model-specification)
15. [Known Invariants & Guard Rails](#15-known-invariants--guard-rails)
16. [Trading Cycle Decision Flow](#16-trading-cycle-decision-flow)
17. [Exit-First Strategy](#17-exit-first-strategy)
18. [Incident Runbooks](#18-incident-runbooks)
19. [Tools Reference](#19-tools-reference)

---

## 1. System Overview

OctiVault Trader is a production-grade autonomous cryptocurrency trading bot built on Binance Spot. It uses a multi-agent signal architecture with ML forecasting, a strict 8-layer dependency model, and a 60/20/20 capital allocation strategy.

**Runtime characteristics**
- Language: Python 3.9
- I/O model: fully async (`asyncio` + `aiohttp`)
- Exchange: Binance Spot (REST + WebSocket)
- Quote currency: USDT
- Max active symbols: 5 (configurable)
- Capital strategy: 60/20/20 (compound / healing / buffer)

**Operating modes**

| Mode | Trigger | Behavior |
|------|---------|----------|
| Paper | `PAPER_TRADING=true` or sentinel API keys | Simulates fills, no real orders |
| Testnet | `USE_TESTNET=true` | Real orders against Binance testnet |
| Live | Signed credentials, neither flag set | Real funds, full execution |

---

## 2. Architecture Specification

### 2.1 Layer Stack

```
L8  Lifecycle        meta_controller, lifecycle_manager, startup_orchestrator, watchdog
L7  Observability    health_monitor, performance_monitor, alert_system, prometheus_exporter
L6  Governance       risk_manager, capital_allocator, capital_governor, policy_manager
L5  Strategy         signal_manager, agent_manager, arbitration_engine, model_manager
L4  Execution        execution_manager, tp_sl_engine, liquidation_orchestrator
L3  Portfolio        portfolio_manager, position_manager, state_synchronizer, symbol_manager
L2  Market Data      market_data_feed, balance_manager, volatility_regime
L1  Exchange I/O     exchange_client, ws_market_data, order_cache_manager
L0  Core             config, contracts, shared_state, error_types, layer_contracts
```

### 2.2 Dependency Rules

Higher layers may call lower layers. Lateral calls (same layer) are allowed only within L0. Cross-layer upward calls (low calling high) are **forbidden**. Enforced at import time via `src/l0_core/layer_contracts.py`.

```python
# src/l0_core/layer_contracts.py — exact values
ALLOWED_DEPENDENCIES = {
    "L0": set(),                                          # pure utilities
    "L1": {"L0"},
    "L2": {"L0", "L1"},
    "L3": {"L0", "L2"},          # skips L1 — exchange I/O cached inside L3
    "L4": {"L0", "L1", "L3"},    # skips L2 — market data cached in L3
    "L5": {"L0", "L3"},          # pure decisions — no market/exchange calls
    "L6": {"L0", "L3", "L5"},    # read-only governance
    "L7": {"L0", "L1", "L2", "L3", "L4", "L5", "L6"},   # read-only observability
    "L8": {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"},  # boots everything
}
```

Critical non-obvious skips:
- **L3 skips L1**: Portfolio never calls exchange directly; uses data already fetched by L2
- **L4 skips L2**: Execution reads market data from L3 state, not live feeds
- **L5 skips L1/L2/L4**: Strategy is pure signal logic — zero I/O dependency

### 2.3 Boot Sequence

```
# High-level layer init order
1. L0  Config loaded & validated
2. L0  SharedState instantiated
3. L1  ExchangeClient connected (REST auth verified)
4. L2  MarketDataFeed started, BalanceManager synced
5. L3  PortfolioManager bootstrapped (BootstrapOrchestrator)
6. L4  ExecutionManager initialized
7. L5  AgentManager loaded (ML models warmed)
8. L6  RiskManager, CapitalAllocator initialized
9. L7  HealthMonitor, PerformanceMonitor started
10. L8  MetaController enters main trading loop
```

**Detailed startup steps** (`StartupOrchestrator.execute_startup_sequence`):

```
_step_recovery_engine_rebuild()        # RecoveryEngine rebuilds positions from event log
_step_hydrate_positions()              # SharedState loads positions from exchange balances
_step_clear_stale_quote_reservations() # Clears any orphaned quote locks
_step_liquidate_legacy_positions()     # Sells positions from previous run not in current strategy
_step_aggressive_exchange_liquidation()# Force-exits stuck orders if needed
_step_auditor_restart_recovery()       # ExchangeTruthAuditor syncs open orders vs. local state
_step_build_capital_ledger()           # Reconstructs capital ledger from wallet snapshot
_step_verify_capital_integrity()       # Asserts NAV matches exchange truth
_emit_state_rebuilt_event()            # Broadcasts portfolio-ready to all subscribers
_emit_startup_ready_event()            # System enters trading mode
```

---

## 3. Layer Contracts

### L0 — Core

**`SharedState`** — central event bus and state store

```python
class SharedState:
    # State
    nav: Decimal
    prices: dict[str, Decimal]               # symbol → last price
    positions: dict[str, Position]
    market_data: dict[tuple[str,str], OHLCV]  # (symbol, tf) → candles
    balances: dict[str, Decimal]

    # Event bus
    async def emit_event(event: str, payload: dict) -> None
    def subscribe_events(event: str, handler: Callable) -> None

    # Health
    health_code: HealthCode
    component_status: dict[str, ComponentStatus]
```

**`HealthCode`** — `src/l0_core/shared_state.py`
```python
class HealthCode(Enum):
    OK    = "ok"
    WARN  = "warn"
    ERROR = "error"
```

**`DustClass`** — formal dust lifecycle taxonomy
```python
class DustClass(Enum):
    TRADABLE                 = "TRADABLE"                  # tradeable position
    NEAR_DUST                = "NEAR_DUST"                 # approaching dust threshold
    DUST                     = "DUST"                      # below min notional
    RECOVERABLE_DUST         = "RECOVERABLE_DUST"          # can be sold if price recovers
    PERMANENT_WRITE_DOWN_DUST = "PERMANENT_WRITE_DOWN_DUST" # economically unrecoverable
```

**`PositionState`** — position lifecycle state
```python
class PositionState(Enum):
    ACTIVE      = "ACTIVE"
    DUST_LOCKED = "DUST_LOCKED"
    LIQUIDATING = "LIQUIDATING"
```

**`AssetClassification`** — three-layer capital accounting
```python
class AssetClassification(Enum):
    BOT_POSITION      = "BOT_POSITION"      # created by trading strategy
    EXTERNAL_POSITION = "EXTERNAL_POSITION" # pre-existing / external deposit
    DUST              = "DUST"              # below MIN_ECONOMIC_TRADE_USDT
    STABLE            = "STABLE"            # stablecoins (USDT, FDUSD, etc.)
    RECOVERY          = "RECOVERY"          # from previous run restart
```

**`ExecutionResult`**
```python
class ExecutionResult(Enum):
    FILLED   = "FILLED"
    PARTIAL  = "PARTIAL"
    REJECTED = "REJECTED"
    BLOCKED  = "BLOCKED"
```

**`Component`** — registered health-check components
```python
class Component(Enum):
    MARKET_DATA_FEED = "MarketDataFeed"
    EXECUTION_MANAGER = "ExecutionManager"
    META_CONTROLLER   = "MetaController"
    AGENT_MANAGER     = "AgentManager"
    RISK_MANAGER      = "RiskManager"
    PNL_CALCULATOR    = "PnLCalculator"
    PERFORMANCE_MON   = "PerformanceEvaluator"
    APP_CONTEXT       = "AppContext"
```

---

### L1 — Exchange I/O

**`ExchangeClient`** public interface

```python
# Orders
async def place_market_buy(symbol, quote_amount, trace_id) -> dict
async def place_market_sell(symbol, quantity, trace_id) -> dict
async def cancel_order(symbol, order_id) -> dict
async def get_order_status(symbol, order_id) -> dict

# Account
async def get_balances() -> dict[str, Decimal]
async def get_account_snapshot() -> dict

# Market
async def get_ticker(symbol) -> dict
async def get_klines(symbol, interval, limit) -> list[OHLCV]
async def get_exchange_info(symbol) -> dict     # LOT_SIZE, MIN_NOTIONAL filters

# Auth
def _has_signed_credentials() -> bool           # False in paper mode
async def reconnect_user_data_stream() -> None

# WebSocket
async def start_user_data_stream() -> None      # 3-tier: WS-API → listenKey → polling
```

**WebSocket tiers** (user-data stream, ordered by preference)

| Tier | URL | Auth |
|------|-----|------|
| 1 | `wss://ws-api.binance.com:443/ws-api/v3` | HMAC signature (default) / Ed25519 session.logon |
| 2 | `wss://stream.binance.com:9443/ws/{listenKey}` | listenKey in URL |
| 3 | REST polling | `_user_data_polling_loop` fallback |

Set `BINANCE_API_TYPE=ED25519` to use session.logon (tier 1, Ed25519 path).

---

### L4 — Execution

**Order path guard** — all orders must flow through `ExecutionManager`:

```
ENFORCE_EXECUTION_MANAGER_PATH=True   (default, enforced)
ALLOW_UNSAFE_DIRECT_ORDER_PATH=True   (override, testing only)
```

`begin_execution_order_scope()` context manager gates every order. Direct calls to `ExchangeClient.place_*` outside this scope raise at runtime when the guard is active.

**`ExecutionManager.place_order`** contract

```python
async def place_order(
    symbol: str,
    side: str,                        # "BUY" | "SELL"
    quote_amount: Optional[float],    # specify either quote_amount or quantity
    quantity: Optional[float],
    order_type: str = "MARKET",
    trace_id: Optional[str] = None,
    tag: str = "",
) -> dict                             # exchange order response
```

Raises `InsufficientBalanceError` (circuit-broken after N failures), `OrderValidationError`, `ExecutionDeadlockError`.

---

### L5 — Strategy

**Signal schema** (dict passed to `SignalManager.receive_signal`)

```python
{
    "symbol":     str,          # e.g. "BTCUSDT"
    "action":     str,          # "BUY" | "SELL" | "HOLD"
    "confidence": float,        # 0.0–1.0
    "reason":     str,
    "agent":      str,          # agent name
    "ts_ms":      int,          # Unix ms
    # optional extras
    "entry_price": float,
    "stop_loss":   float,
    "take_profit": float,
}
```

**`ArbitrationEngine`** fusion rules
- Requires ≥ 2 agents to agree before emitting a `TradeIntent`
- Weighted by `agent_performance_score` (rolling win-rate × PnL factor)
- Suppressed if `RiskManager.is_trading_halted() == True`

---

## 4. Core Data Models

### `TradeIntent` (L0 contract)

```python
# src/l0_core/contracts.py
@dataclass
class TradeIntent:
    # --- Set by agents ---
    symbol:         str
    side:           Union[OrderSide, str] = OrderSide.BUY   # normalised to OrderSide on init
    quote:          Optional[float] = None    # USDT amount to spend (buy side)
    quantity:       Optional[float] = None    # asset qty to sell (sell side)
    confidence:     Optional[float] = None    # 0.0–1.0
    reason:         str = ""
    agent:          str = ""
    tag:            str = ""
    ts_ms:          int = field(default_factory=lambda: int(time.time() * 1000))

    # --- Stamped by MetaController / ExecutionManager (agents must NOT set these) ---
    planned_quote:  Optional[float] = None    # resolved quote after capital allocation
    trace_id:       Optional[str] = None
    tier:           Optional[str] = None      # "BOT_POSITION"|"RECOVERY"|"DUST_RECOVERY"
    is_liquidation: bool = False
    policy_context: Optional[dict] = None
```

Note: `side` accepts both `OrderSide` enum and plain string; the constructor normalises to `OrderSide.BUY`/`OrderSide.SELL` (lowercase values `"buy"`/`"sell"`).

### `Position`

```python
@dataclass
class Position:
    symbol:           str
    quantity:         Decimal
    avg_entry_price:  Decimal
    unrealized_pnl:   Decimal
    bucket:           str          # "compound"|"healing"|"buffer"
    dust_class:       DustClass
    opened_at:        datetime
    last_updated:     datetime
    trace_ids:        list[str]    # all order trace IDs
```

### `OHLCV` (candle)

```python
@dataclass
class OHLCV:
    symbol:    str
    tf:        str           # e.g. "5m"
    open_time: int           # Unix ms
    open:      Decimal
    high:      Decimal
    low:       Decimal
    close:     Decimal
    volume:    Decimal
    closed:    bool
```

### Error hierarchy

```
TraderException
├── ExchangeError
│   ├── InsufficientBalanceError
│   ├── OrderValidationError
│   └── RateLimitError
├── ExecutionError
│   ├── ExecutionDeadlockError
│   └── DuplicateOrderError
├── PortfolioError
│   ├── PositionNotFoundError
│   └── CapitalAllocationError
├── ConfigError
└── HealthError
    └── CircuitBreakerOpen
```

Each exception carries:
```python
ErrorContext(
    severity:  ErrorSeverity,   # INFO | WARN | ERROR | CRITICAL
    category:  ErrorCategory,   # EXCHANGE | EXECUTION | PORTFOLIO | …
    recovery:  ErrorRecovery,   # RETRY | SKIP | HALT | NOTIFY
    trace_id:  Optional[str],
)
```

---

## 5. Configuration Reference

All config is loaded from `.env` via `python-dotenv` into `src/l0_core/config.py`.

### Exchange

| Key | Default | Notes |
|-----|---------|-------|
| `BINANCE_API_KEY` | — | Real/testnet key; `paper_key` for paper mode |
| `BINANCE_API_SECRET` | — | Real/testnet secret; `paper_secret` for paper mode |
| `BINANCE_API_TYPE` | `HMAC` | `HMAC` or `ED25519` |
| `USE_TESTNET` | `false` | Switch to Binance testnet endpoints |
| `PAPER_TRADING` | `false` | Full paper simulation |

### Capital

| Key | Default | Notes |
|-----|---------|-------|
| `INITIAL_CAPITAL_USDT` | — | Starting USDT |
| `BOOTSTRAP_BUDGET_USDT` | — | Budget for initial symbol buys |
| `BOOTSTRAP_NUM_SYMBOLS` | `3` | Symbols to buy at boot |
| `MAX_ACTIVE_SYMBOLS` | `5` | Hard cap on open positions |
| `MIN_ACTIVE_SYMBOLS` | `3` | Min before new buys are triggered |
| `MAX_UNIVERSE_SYMBOLS` | `30` | Watchlist size |
| `FIX8_COMPOUND_ALLOCATION_PCT` | `0.60` | Top-3 bucket share |
| `FIX8_HEALING_ALLOCATION_PCT` | `0.20` | Healing bucket share |
| `FIX8_BUFFER_ALLOCATION_PCT` | `0.20` | Emergency buffer share |

### 4th Slot (aggressive profit hunting)

| Key | Default | Notes |
|-----|---------|-------|
| `FIX8_4TH_SLOT_ENABLED` | `true` | Enable 4th rotating slot |
| `FIX8_4TH_SLOT_PROFIT_TARGET_PCT` | `0.15` | Exit at +15% |
| `FIX8_4TH_SLOT_STOP_LOSS_PCT` | `-0.03` | Exit at -3% |
| `FIX8_4TH_SLOT_MAX_HOLD_MINUTES` | `120` | Auto-close after 2 h |

### Risk

| Key | Default | Notes |
|-----|---------|-------|
| `MAX_POSITION_SIZE_PCT` | — | Max % of NAV per position |
| `MAX_DAILY_DRAWDOWN_PCT` | — | Halt if NAV drops this % intraday |
| `MAX_LEVERAGE` | `1` | 1 = spot only |
| `MIN_NOTIONAL_USDT` | — | Min order size (exchange minimum) |

### Timeframes

| Key | Default | Notes |
|-----|---------|-------|
| `TIMEFRAMES` | `["1m","5m"]` | Candle intervals subscribed |
| `MIN_BARS_FOR_MARKET_READY` | `150` | Bars required before trading begins |

### Execution guards

| Key | Default | Notes |
|-----|---------|-------|
| `ENFORCE_EXECUTION_MANAGER_PATH` | `true` | Block direct order calls |
| `ALLOW_UNSAFE_DIRECT_ORDER_PATH` | `false` | Override (testing only) |

### Volatility regimes

| Key | Default | Notes |
|-----|---------|-------|
| `VOLATILITY_REGIME_LOW_PCT` | `0.0025` | Below = low regime |
| `VOLATILITY_REGIME_HIGH_PCT` | `0.006` | Above = high regime |

### Bootstrap

| Key | Default | Notes |
|-----|---------|-------|
| `BOOTSTRAP_SOFT_LOCK_DURATION_SEC` | `3600` | 1-hour cooldown after bootstrap trade |
| `SYMBOL_REPLACEMENT_MULTIPLIER` | `1.10` | New symbol must be 10% better to rotate |

---

## 6. Exchange Integration

### REST endpoints used

| Operation | Endpoint |
|-----------|----------|
| Place order | `POST /api/v3/order` |
| Cancel order | `DELETE /api/v3/order` |
| Query order | `GET /api/v3/order` |
| Account info | `GET /api/v3/account` |
| Ticker price | `GET /api/v3/ticker/price` |
| Klines | `GET /api/v3/klines` |
| Exchange info | `GET /api/v3/exchangeInfo` |
| listenKey create | `POST /api/v3/userDataStream` |
| listenKey keepalive | `PUT /api/v3/userDataStream` |

### WebSocket event → SharedState routing

`_ingest_user_data_ws_payload` dispatches WS payloads to `SharedState.emit_event`:

| WS event type | `emit_event` name |
|---------------|-------------------|
| `executionReport` | `order_update` |
| `balanceUpdate` | `balance_update` |
| `outboundAccountPosition` | `account_position` |
| `listStatus` | `list_status` |
| `listenKeyExpired` | `listen_key_expired` |

Dispatch uses `loop.create_task(ss.emit_event(...))` — never `await` directly from the WS handler to avoid blocking the event loop.

### Rate limit handling

`RetryManager` applies exponential backoff with jitter. Binance `429` responses trigger a `RateLimitError`; `RetryManager` waits `Retry-After` seconds before retrying. After 5 consecutive rate-limit errors, the circuit breaker opens and `ExecutionManager` halts new orders.

---

## 7. Strategy & Signal Pipeline

### Agent roster

| Agent | Type | Signal source |
|-------|------|---------------|
| `SwingTradeHunter` | ML (Keras) | Per-symbol 5m model |
| `TrendHunter` | ML (Keras) | Per-symbol 5m model |
| `MLForecaster` | ML (Keras) | Ensemble over trained symbols |
| `BaselineTradingKernel` | Rules | MA crossover + volume |
| `VolatilityRegimeAgent` | Rules | Regime-adjusted signals |
| `OpportunityRanker` | Rules | Cross-symbol ranking |
| `CapitalVelocityOptimizer` | Rules | Turnover optimization |
| `ExternalAdoptionEngine` | External | Pluggable external feeds |

### Signal flow

```
Agent.generate_signal()
    → SignalManager.receive_signal()       # validate, cache, deduplicate
    → ArbitrationEngine.arbitrate()        # fuse, weight, threshold
    → MetaController.on_trade_intent()     # policy check, trace_id stamp
    → ExecutionManager.place_order()       # order path guard
    → ExchangeClient.place_market_*()      # Binance API
```

### Arbitration thresholds

- Minimum signal agreement: 2 agents
- Minimum weighted confidence: 0.55 (configurable)
- Signals older than 60 s are stale and excluded
- `HOLD` signals from any agent reduce composite confidence by 0.1 per agent

---

## 8. Portfolio & Capital Management

### 60/20/20 Three-Bucket Model

```
Total Deployable USDT
├── 60%  Compound bucket   → top 3 positions by EV score
├── 20%  Healing bucket    → recovery trades for losing positions
└── 20%  Buffer bucket     → emergency liquidity + 4th slot
```

Managed by `ThreeBucketManager` + `CompoundingEngine`. Rebalancing triggered on:
- NAV change > 5%
- Position close/open
- Daily at configurable time

### 4th Slot

The 4th slot (in addition to the 3 compound positions) is a high-turnover aggressive position:
- Source: buffer bucket
- Entry: highest-confidence arbitration signal
- Exit: first of → +15% profit | -3% stop | 2-hour timeout
- Tracked by `FourthSlotTracker`

### Dust classification

Uses `DustClass` enum (see §3 L0). Policy mapping:

| `DustClass` value | Condition | Action |
|-------------------|-----------|--------|
| `TRADABLE` | value ≥ min notional, no lock | normal trading |
| `NEAR_DUST` | approaching dust threshold | monitor, no new buy |
| `DUST` | below min notional | `DeadCapitalHealer` sweep |
| `RECOVERABLE_DUST` | can be sold if price recovers slightly | schedule retry sell |
| `PERMANENT_WRITE_DOWN_DUST` | economically unrecoverable | write off, no action |

`DeadCapitalHealer` runs a sweep cycle every `DEAD_CAPITAL_HEAL_INTERVAL_SEC` seconds, attempting to liquidate up to `max_liquidations` positions per cycle (default: 50 after Fix #5).

### Symbol rotation

`UniverseRotationEngine` rotates symbols when a candidate is `SYMBOL_REPLACEMENT_MULTIPLIER` (1.10×) better by EV score than the worst current position. Rotation is gated by `BootstrapManager` soft-lock (1 h after any bootstrap trade).

---

## 9. Execution Pipeline

### Order lifecycle

```
TradeIntent
  → PolicyManager.check(intent)           # governance veto
  → RiskManager.validate(intent)          # size, notional, drawdown
  → CapitalAllocator.approve(intent)      # bucket has capital
  → ExecutionManager.place_order()        # begin_execution_order_scope
      → ExchangeClient.place_market_*()
      → OrderCacheManager.track(order)
      → EventStore.record(fill)
  → PositionManager.update(fill)
  → SharedState.emit_event("fill", ...)
```

### TP/SL engine

`TPSLEngine` monitors open positions every tick:
- Take-profit: market sell at `entry × (1 + tp_pct)`
- Stop-loss: market sell at `entry × (1 - sl_pct)`
- Trailing stop: optional, updated on each new high

### Recovery

`RecoveryEngine` handles stuck orders:
1. Poll order status after `ORDER_STUCK_TIMEOUT_SEC`
2. Cancel if still `NEW` or `PARTIALLY_FILLED`
3. Re-place if cancellation succeeds
4. Mark position for manual review after 3 failed recoveries

### Idempotency

Every order is tagged with a `trace_id` (UUID4). `OrderCacheManager` rejects duplicate `trace_id` submissions. `sell_finalize` operations are idempotent — re-running with the same `trace_id` is a no-op.

---

## 10. Governance & Risk

### Risk gates (ordered, any failure halts intent)

1. `MAX_DAILY_DRAWDOWN_PCT` — NAV vs. SOD NAV
2. `MAX_POSITION_SIZE_PCT` — new position would exceed limit
3. `MAX_ACTIVE_SYMBOLS` — already at cap
4. `MIN_NOTIONAL_USDT` — order too small for exchange
5. `InsufficientBalanceError` circuit breaker — N consecutive failures → halt 10 min
6. `HealthCode.ERROR` on any tracked `Component` — all trading suspended

### Capital governor

`CapitalGovernor` enforces per-symbol and aggregate capital limits. `CapitalSymbolGovernor` tracks per-symbol exposure in USDT. Both are checked before `CapitalAllocator.approve`.

### Policy manager

`PolicyManager` applies higher-level rules:
- Trading hours filter (`TradingHoursManager`)
- Regime filter (no buys in `HIGH_VOLATILITY` unless confidence > 0.80)
- Churn prevention (no sell+rebuy of same symbol within `CHURN_COOLDOWN_SEC`)

---

## 11. Observability & Health

### Health check matrix

| Component | Check interval | Critical threshold |
|-----------|---------------|-------------------|
| ExchangeClient | 30 s | No REST response in 60 s |
| UserDataStream | 60 s | No event in 120 s |
| BalanceManager | 60 s | Balance stale > 90 s |
| MarketDataFeed | 30 s | No candle in 60 s |
| MetaController | 15 s | Loop stalled > 30 s |

`HealthMonitor` aggregates `Component` statuses into system `HealthCode`. Transitions:
- Any component → `HealthCode.WARN` → system `WARN`
- Any component → `HealthCode.ERROR` → system `ERROR`, trading suspended

### Metrics exported (Prometheus)

`SafetyGuardMetrics` — `src/l7_observability/prometheus_exporter.py`

**Safety Guard metrics**

| Metric | Type | Labels |
|--------|------|--------|
| `balance_validation_total` | Counter | symbol, status |
| `balance_validation_latency_seconds` | Histogram | — |
| `balance_validation_approval_rate` | Gauge | — |
| `leverage_validation_total` | Counter | symbol, status |
| `leverage_validation_latency_seconds` | Histogram | — |
| `leverage_validation_approval_rate` | Gauge | — |
| `current_max_leverage` | Gauge | — |
| `hours_validation_total` | Counter | symbol, status |
| `hours_validation_latency_seconds` | Histogram | — |
| `hours_validation_approval_rate` | Gauge | — |
| `anomaly_detection_total` | Counter | status |
| `anomaly_detection_latency_seconds` | Histogram | — |
| `anomaly_detection_rate` | Gauge | — |
| `anomaly_signals_in_history` | Gauge | — |
| `correlation_validation_total` | Counter | symbol, status |
| `correlation_validation_latency_seconds` | Histogram | — |
| `correlation_validation_approval_rate` | Gauge | — |
| `active_positions_count` | Gauge | — |
| `max_concentration_ratio` | Gauge | — |

**Execution metrics**

| Metric | Type | Labels |
|--------|------|--------|
| `trades_executed_total` | Counter | symbol, side, guard_status |
| `execution_latency_seconds` | Histogram | — |
| `guard_latency_seconds` | Histogram | — |
| `overall_approval_rate` | Gauge | — |

`guard_status` label values: `all_passed`, `rejected_by_balance`, `rejected_by_leverage`, `rejected_by_hours`, `rejected_by_anomaly`, `rejected_by_correlation`

### Distributed tracing

Every `TradeIntent` carries a `trace_id` propagated through all layers. Jaeger exporter (`jaeger_tracer.py`) emits spans for: signal receipt → arbitration → policy → execution → fill.

---

## 12. Testing Specification

### Test framework

- pytest 8.4.2 with `asyncio_mode = "strict"` (every async test needs `@pytest.mark.asyncio`)
- Fixtures in `tests/conftest.py`
- Coverage target: all L0–L8 modules

### Test modules

| File | Covers | Count |
|------|--------|-------|
| `test_layered_architecture.py` | Layer contracts, boot order, call graph | — |
| `test_l0_cross_cutting.py` | Config, error types, SharedState, contracts | — |
| `test_l1_exchange.py` | ExchangeClient, retry, WS tiers | — |
| `test_l2_wallet.py` | BalanceManager, volatility regime | — |
| `test_l3_portfolio.py` | PortfolioManager, PositionManager | — |
| `test_l4_execution.py` | ExecutionManager, TP/SL, recovery | — |
| `test_l5_strategy.py` | SignalManager, ArbitrationEngine, agents | — |
| `test_l6_governance.py` | RiskManager, CapitalAllocator | — |
| `test_l7_observability.py` | HealthMonitor, metrics | — |
| `test_websocket_integration.py` | WS event ingestion, SharedState routing (78 tests) | 78 |
| `test_nav_truthfulness_and_capital_clamp.py` | NAV double-count prevention | — |
| `test_nav_no_double_count.py` | NAV idempotency | — |
| `test_sell_finalize_idempotency.py` | Idempotent sell | — |
| `test_insuff_bal_circuit_breaker.py` | Balance circuit breaker | — |
| `test_live_order_recovery_guards.py` | Stuck order recovery | — |
| `test_portfolio_fragmentation_integration.py` | Fragment handling | — |
| `test_dust_exit_candidate_selection.py` | Dust selection | — |
| `test_self_healing_controller.py` | Auto-healing | — |
| `test_truth_audit_wallet_guard.py` | Wallet guard | — |
| `test_strict_cap_count_tradable.py` | Position cap | — |
| `test_exchange_client_paper_mode.py` | Paper mode | — |
| `test_exchange_client_testnet_configuration.py` | Testnet keys | — |

### Mock patterns

**Do not** use bare `Mock()` for fixtures that rely on attribute *absence* (e.g., `_lock_context` in `WebSocketMarketData`). Use plain classes or `Mock(spec=[])` instead — `Mock()` returns truthy for any `hasattr()`.

**Do not** mock the exchange database in integration tests. Use Binance testnet or VCR cassettes.

### Running tests

```bash
# Full suite
pytest tests/ -v

# Single layer
pytest tests/test_l4_execution.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# WS integration only
pytest tests/test_websocket_integration.py -v
```

---

## 13. Deployment & Operations

### Launch (recommended)

```bash
./start_trading_with_monitoring.sh
# or directly:
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py --duration 24
# paper mode:
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py --paper --duration 8
```

### Health check

```bash
python3 check_status.py
python3 check_balance.py
```

### Emergency tools

| Script | Use |
|--------|-----|
| `force_liquidate_dust.py` | Force-sell all hard-dust positions |
| `force_balance_sync.py` | Re-sync balance from exchange |
| `fix_execution_deadlock.py` | Clear stuck execution state |
| `diagnose_healing.py` | Debug dead-capital healer |
| `diagnose_execution_blocker.py` | Debug execution blocks |
| `validate_churn_fix.py` | Verify churn guard is working |

### Monitoring stack

| Tool | Access |
|------|--------|
| Real-time dashboard | `python3 launch_with_monitor.py` |
| Prometheus | `src/l7_observability/prometheus_exporter.py` |
| Jaeger | configured via `JAEGER_HOST` env var |
| Health HTTP endpoint | `src/l7_observability/health_endpoints.py` |

### Environment file management

| File | Purpose |
|------|---------|
| `.env` | Active configuration |
| `.env.template` | Canonical reference (all keys documented) |
| `.env.bak` | Auto-backup before major changes |
| `.env.growth` | Growth-mode overrides |

### Log files

All logs under `logs/`. Key log names:
- `trading.log` — main execution log
- `ws_events.log` — WebSocket event stream
- `health.log` — component health transitions
- `pnl.log` — fill and P&L records

---

## 14. ML Model Specification

### Model format

- Framework: Keras / TensorFlow 2.15
- File format: `.keras` (model) + `.pkl` (metadata/scaler)
- Location: `src/models/`

### Covered symbols (50+ models)

AAVEUSDT, ADAUSDT, APTUSDT, ARBUSDT, ASTERUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOTUSDT, ETHUSDT, LINKUSDT, LTCUSDT, MATICUSDT, NEARUSDT, SOLUSDT, UNIUSDT, XRPUSDT, … (full list in `src/models/`)

### Input features (per model)

- OHLCV candles, 5m timeframe, last 150 bars
- Derived: EMA-9, EMA-21, ATR-14, RSI-14, MACD, volume z-score
- Normalized per-symbol using stored `StandardScaler`

### Output

```python
{
    "action":     "BUY" | "SELL" | "HOLD",
    "confidence": float,   # 0.0–1.0
    "horizon_bars": int,   # expected hold in bars
}
```

### Model lifecycle

`ModelManager` loads models at startup. `ModelTrainer` can retrain on-the-fly if `ONLINE_TRAINING=true` and sufficient new bars are available (>= 500 new candles since last train).

---

## 15. Known Invariants & Guard Rails

These must hold at all times or the system will malfunction silently.

### Critical invariants

1. **NAV single-source**: NAV must be computed from exchange balance snapshot only. Never add `unrealized_pnl` to `USDT_balance` — that double-counts allocated capital. Enforced by `test_nav_no_double_count.py`.

2. **Order path guard**: All real orders must pass through `ExecutionManager.begin_execution_order_scope()`. Direct calls to `ExchangeClient.place_*` outside this scope are illegal when `ENFORCE_EXECUTION_MANAGER_PATH=True`.

3. **Event dispatch is non-blocking**: WS event handlers must not `await` `SharedState.emit_event` directly. Always use `loop.create_task(...)`. Blocking the WS handler drops subsequent events.

4. **CancelledError propagation**: In Python 3.9, `asyncio.CancelledError` is NOT a subclass of `Exception`. Bare `except Exception` does NOT catch it. Every long-running coroutine must handle `CancelledError` explicitly for clean shutdown.

5. **Idempotent sells**: Selling a position twice with the same `trace_id` must be a no-op. `sell_finalize` checks `OrderCacheManager` before placing. Never remove this guard.

6. **Balance sync before orders**: Balance must be re-fetched from exchange before any order placement if last sync is > 30 s old. Stale balance leads to `InsufficientBalance` errors and circuit-breaker trips.

7. **Ed25519 vs HMAC**: Session.logon (WS-API tier 1) requires Ed25519 keys. HMAC keys fall back to listenKey (tier 2) automatically. Setting `BINANCE_API_TYPE=ED25519` with HMAC keys will cause WS auth failures.

8. **pytest-asyncio strict mode**: All async tests require `@pytest.mark.asyncio`. Missing the decorator causes the test to be collected but not awaited, silently passing without executing.

9. **Dust classification before rotation**: `DeadCapitalHealer` and `PortfolioManager` must classify dust (`DustClass`) before any rotation decision. Skipping classification allows `DUST`/`NEAR_DUST` positions to be counted as tradable capital.

10. **Reconnect counter**: `reconnect_user_data_stream` has an early return if `_has_signed_credentials()` is False (paper mode). The reconnect counter is NOT incremented in that case. Do not rely on the counter in paper mode.

---

## 16. Trading Cycle Decision Flow

`MetaController` runs a tight loop. Each tick executes these stages in order. Any failed guard short-circuits to END without executing later stages.

```
START CYCLE
│
├─ [1] Tick increment + cycle timer
│
├─ [2] Drain market events
│       Process price updates, fills, position snapshots, metrics
│
├─ [3] Guard evaluation (all 6 must pass)
│   ├─ Guard 1 — Market Data Ready?    price exists + age < 5 s  → else SKIP CYCLE
│   ├─ Guard 2 — Balances Available?   USDT > 0 + free > MIN_CAPITAL  → else TRIGGER RECOVERY
│   ├─ Guard 3 — Ops Plane Ready?      ExchangeClient functional + orders succeeding  → else HALT
│   ├─ Guard 4 — Trading Hours Valid?  within allowed window, no maintenance  → else SKIP CYCLE
│   ├─ Guard 5 — Position Constraints? total open < regime_max + concentration < limit  → else SKIP
│   └─ Guard 6 — Capital Adequacy?     free capital ≥ MIN_CAPITAL, no forced recovery active  → else TRIGGER RECOVERY
│
├─ [4] Signal intake & filtering
│       confidence ≥ 0.50 + age < 60 s + deduplicate (1 BUY + 1 SELL per symbol, highest conf)
│
├─ [5] Batch collection & sorting
│       Collect up to 50 signals, sort by confidence descending
│
└─ [6] Per-signal arbitration (6 gates)
    ├─ GATE 1 — Lifecycle state check   symbol not in ROTATION_PENDING/DUST_HEALING conflict
    ├─ GATE 2 — Portfolio health check  position count < regime_max, dust ratio < threshold
    ├─ GATE 3 — Capital availability    free_quote = (balance - reserve) - allocated ≥ MIN_ENTRY_QUOTE
    ├─ GATE 4 — Economic gate (anti-churn)
    │           round_trip_cost = (2×taker_fee) + (2×slippage) ≈ 0.50%
    │           min_profitable_move = 0.55%  →  expected_alpha must exceed this
    ├─ GATE 5 — Signal confidence gate
    │           MICRO_SNIPER: ≥ 0.50 | STANDARD: ≥ 0.55 | MULTI_AGENT: ≥ 0.60
    └─ GATE 6 — Regime gating
                MICRO_SNIPER (NAV < $1 000): max 1 position, no rotation, no dust healing
                STANDARD (NAV $1 000–$5 000): max 2 positions
                MULTI_AGENT (NAV > $5 000): max active symbols = MAX_ACTIVE_SYMBOLS
```

---

## 17. Exit-First Strategy

**Core principle**: The system must plan the full exit before entering any position. Entering without a defined exit is how capital becomes permanently locked.

### Exit pathways (every position must have all three)

| Pathway | Trigger | Notes |
|---------|---------|-------|
| Take-profit (TP) | price ≥ entry × (1 + tp_pct) | captured by `TPSLEngine` |
| Stop-loss (SL) | price ≤ entry × (1 - sl_pct) | mandatory, not optional |
| Time exit | position age > max_hold_minutes | prevents indefinite holding |

### Capital release guarantee

Before `ExecutionManager.place_order()` fires a BUY:
1. `planned_quote` is set on `TradeIntent` (capital committed)
2. TP/SL parameters are recorded in `TPSLEngine` state
3. Time-exit deadline is registered in `FourthSlotTracker` (4th slot) or `PositionManager`

If any of these three registrations fail, the BUY is rejected.

### Anti-deadlock invariant

Capital must always have a path back to USDT. The sequence:

```
TP fires → market SELL → USDT freed → available for next trade     ✓
SL fires → market SELL → USDT freed (at a loss)                    ✓
Time exit fires → market SELL → USDT freed                         ✓
No exit registered → position can grow to DUST → capital locked     ✗  (blocked by entry guard)
```

---

## 18. Incident Runbooks

### System Hung / Trading Stalled

**Symptoms**: No fills for > 15 min, logs show `MetaController stalled`, `HealthCode.ERROR`

```bash
# 1. Check system status
python3 check_status.py

# 2. Check for execution deadlock
python3 diagnose_execution_blocker.py

# 3. If deadlock confirmed:
python3 fix_execution_deadlock.py

# 4. Restart with monitoring
./start_trading_with_monitoring.sh
```

### Capital Stuck as Dust

**Symptoms**: NAV far below USDT balance, many `DUST` / `RECOVERABLE_DUST` positions, healer not recovering

```bash
# 1. Diagnose healer state
python3 diagnose_healing.py

# 2. Force manual sweep (up to 50 positions)
python3 force_liquidate_dust.py

# 3. Force balance re-sync after liquidations
python3 force_balance_sync.py
```

### Balance Out of Sync

**Symptoms**: `InsufficientBalance` errors despite sufficient NAV, circuit breaker tripping

```bash
python3 force_balance_sync.py
python3 check_balance.py
```

### Database / State Corruption

**Symptoms**: `sqlite3` errors, `/ready` returns `"database": false`, state file corrupted

```bash
# Check disk space first (most common cause)
df -h .

# Check state file integrity
ls -lh state/
sqlite3 state/octivault.db "PRAGMA integrity_check;"

# If corrupted: system will rebuild from exchange truth on restart
# StartupOrchestrator._step_recovery_engine_rebuild() handles this automatically
./start_trading_with_monitoring.sh
```

### Symbol Churn (Excessive Rotation)

**Symptoms**: Same symbol being bought and sold repeatedly within minutes, PnL bleeding to fees

```bash
python3 validate_churn_fix.py
# Check CHURN_COOLDOWN_SEC in .env — should be ≥ 300
```

---

## 19. Tools Reference

All tools under `tools/`. Run from project root.

| Tool | Purpose | Usage |
|------|---------|-------|
| `detect_balance_symbols.py` | Scan wallet + classify all positions | `python3 tools/detect_balance_symbols.py` (live) or `--mock` |
| `diagnose_runtime.py` | Full smoke-test of all layers against live exchange | `python3 tools/diagnose_runtime.py` |
| `exit_metrics.py` | Analyse TP/SL exit performance | `python3 tools/exit_metrics.py` |
| `compound_engine.py` | Inspect 60/20/20 capital allocation state | `python3 tools/compound_engine.py` |
| `recover_missing_sells.py` | Detect and re-issue missing sell orders | `python3 tools/recover_missing_sells.py` |
| `next_level_tpsl_analysis.py` | Deep-dive TP/SL parameter analysis | `python3 tools/next_level_tpsl_analysis.py` |
| `monitor_6h_session.py` | Real-time 6-hour session monitor | `python3 tools/monitor_6h_session.py` |
| `live_monitor_snapshot.sh` | One-shot system snapshot | `bash tools/live_monitor_snapshot.sh` |
| `check_sell_marker_coverage.sh` | Verify all positions have sell markers | `bash tools/check_sell_marker_coverage.sh` |

**Tool pattern** (how to add a new tool that needs real data):

```python
from src.l0_core.config import Config
from src.l0_core.shared_state import SharedState
from src.l1_exchange.exchange_client import ExchangeClient

config = Config()
shared_state = SharedState(config=config)
exchange = ExchangeClient(config=config, shared_state=shared_state)

# then use exchange.get_account_balances(), exchange.get_all_tickers(), etc.
```

---

*End of OctiVault Trader Specs Kit v1.1*
