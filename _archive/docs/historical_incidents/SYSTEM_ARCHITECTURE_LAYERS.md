# System Architecture & Communication Layers

## 🏗️ 9-Layer System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ L8: LIFECYCLE (Meta-Control)                                               │
│ ├─ MetaController: Main orchestrator, fires trades, applies policy         │
│ ├─ LifecycleManager: System startup/shutdown                              │
│ └─ StartupOrchestrator: Bootstrap sequence                                 │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ Policy Decisions & Trade Firing
                       │ (TRADE_EXECUTED / TRADE_SKIPPED events)
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L7: OBSERVABILITY (Monitoring & Metrics)                                    │
│ ├─ PerformanceMonitor: Tracks agent KPIs                                    │
│ ├─ Dashboard: Real-time metrics                                             │
│ ├─ HealthCheckManager: System vitals                                        │
│ ├─ APMInstrument: Distributed tracing                                       │
│ └─ ComponentStatusLogger: Component heartbeats                              │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ KPI Updates, Health Events
                       │ (emit_event calls)
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L6: GOVERNANCE (Risk & Compliance)                                          │
│ ├─ RiskManager: Portfolio risk limits                                       │
│ ├─ CapitalAllocator: 60/20/20 distribution                                  │
│ ├─ Scaling: Dynamic equity tiers                                            │
│ ├─ CapitalGovernor: Position limits (NAV calculation)                       │
│ └─ CompoundingEngine: Reinvestment logic                                    │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ Risk Constraints & Capital Budgets
                       │ (gating decisions)
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L5: STRATEGY (Signal Generation)                                            │
│ ├─ SwingTradeHunter: Main momentum agent (generates BUY/SELL)              │
│ ├─ TrendHunter: Trend following                                             │
│ ├─ DipSniper: Counter-trend micro trades                                    │
│ ├─ LiquidationAgent: Dust healing & exits                                   │
│ ├─ AgentManager: Manages all 8+ agents                                      │
│ ├─ SignalManager: Signal aggregation                                        │
│ ├─ ModeManager: Market regime detection                                     │
│ └─ ObjectiveFeedbackController: Performance feedback                        │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ TradeIntent Events
                       │ (event_bus.publish('events.trade.intent', ...))
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L4: EXECUTION (Order Management)                                            │
│ ├─ ExecutionManager: Order lifecycle (submit, fill, settle)                │
│ ├─ IntentManager: Translates TradeIntent → executable orders               │
│ ├─ LiquidationOrchestrator: Coordinates exits                              │
│ ├─ TPSLEngine: Take-profit & stop-loss                                     │
│ └─ DustHealer: Converts dust to USDT                                        │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ Order Placement Requests to Binance
                       │ (REST API calls)
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L3: PORTFOLIO (State Management)                                            │
│ ├─ StateManager: Position tracking                                          │
│ ├─ PortfolioBalancer: Portfolio rebalancing                                 │
│ ├─ ReserveManager: Capital reserves                                         │
│ ├─ PnLCalculator: Profit/loss valuation                                     │
│ ├─ PortfolioBuckets: Tier allocation tracking                               │
│ └─ TradeJournal: Historical trade log                                       │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ Position Updates & P&L
                       │ (StateManager.update_position())
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L2: MARKET DATA (Feed & Analysis)                                           │
│ ├─ MarketRegimeDetector: Trending / Ranging analysis                        │
│ ├─ MarketRegimeIntegrator: Regime-aware signals                             │
│ ├─ OHLCVFeed: Candlestick data (1h, 4h, 1d)                                │
│ ├─ WebSocketManager: Real-time price updates                               │
│ └─ SymbolScreener: Universe of symbols                                      │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ OHLCV Data, Price Updates
                       │ (price feed API calls)
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L1: EXCHANGE (Binance API)                                                  │
│ ├─ RestClient: HTTP requests (orders, balances)                            │
│ ├─ WebSocketClient: Real-time streams                                      │
│ └─ MarketDataConnector: Historical candles                                  │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ Account Data, Price Ticks
                       │
┌──────────────────────▼──────────────────────────────────────────────────────┐
│ L0: CORE (Central State & Communication)                                     │
│ ├─ SharedState: Singleton holding all system state                          │
│ ├─ EventBus: Pub/Sub messaging (events.trade.intent, HealthStatus, etc.)   │
│ ├─ ConfigManager: System parameters                                         │
│ ├─ Logger: Structured logging                                              │
│ └─ Readiness Gates: Phase gates (AcceptedSymbolsReady, MarketDataReady)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow & Communication Patterns

### Pattern 1: Signal → Trade Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: SIGNAL GENERATION (L5 - Strategy)                                  │
│                                                                              │
│ SwingTradeHunter._generate_signal(SOLUSDT):                                │
│  ├─ Fetch 300 candles from MarketData (L2)                                 │
│  ├─ Calculate EMA20, EMA50, RSI, MACD                                       │
│  ├─ Logic: if ema20 > ema50 and rsi < 75 → BUY signal                      │
│  ├─ Calculate confidence: 0.65 (base) or 0.85 (with volume confirmation)   │
│  └─ confidence=0.65 ✅ PASS (>= 0.35 minimum)                              │
│                                                                              │
│ Output: TradeIntent(symbol=SOLUSDT, side=BUY, confidence=0.65)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EVENT PUBLICATION (L0 - EventBus)                                  │
│                                                                              │
│ SwingTradeHunter._publish_trade_intent():                                  │
│  ├─ Get EventBus from shared_state                                          │
│  ├─ Call: event_bus.publish('events.trade.intent', TradeIntent(...))       │
│  └─ Event enters queue for all subscribers                                  │
│                                                                              │
│ Subscribers: MetaController, SignalManager, Dashboard                       │
│                                                                              │
│ Output: Event queued in system                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SIGNAL CACHING (L5 - SignalManager)                                │
│                                                                              │
│ MetaController.recv_signal(signal):                                        │
│  ├─ Validate: confidence 0.65 < 0.75? → Check if 0.65 >= 0.75             │
│  ├─ Cache in signal_cache for 30 seconds                                    │
│  └─ ✅ Signal cached (confidence=0.65) [TOO LOW - NEEDS 0.75+]             │
│                                                                              │
│ Status: "✓ Signal cached for SOLUSDT (confidence=0.65)"                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: TRADE FIRING ATTEMPT (L8 - MetaController)                         │
│                                                                              │
│ MetaController._fire_trade_intent():                                       │
│  ├─ Loop through cached signals                                             │
│  ├─ Check gates:                                                            │
│  │  ├─ Confidence gate: 0.65 < 0.75 → Could block [SEE BELOW]             │
│  │  ├─ Risk gates: portfolio exposure, position limits                      │
│  │  └─ PreTrade effect: net_pct_below_threshold check                      │
│  ├─ Check expected profit: 0.04% > 0.06%? → NO ❌                          │
│  └─ Decision: SKIP (blocked by net_pct_below_threshold gate)               │
│                                                                              │
│ Output: TRADE_SKIPPED event                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: EXECUTION (L4 - ExecutionManager) [IF GATES PASS]                  │
│                                                                              │
│ ExecutionManager.submit_order():                                           │
│  ├─ Calculate order quantity & quote size                                   │
│  ├─ Place order on Binance (REST API)                                       │
│  ├─ Monitor fills                                                           │
│  └─ Update portfolio state                                                  │
│                                                                              │
│ Output: Order ID, filled qty, average price                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Pattern 2: Event Bus Communication (L0 Core)

```
PUBLISH SIDE:                          EVENT BUS (L0)                 SUBSCRIBE SIDE:
─────────────────────                  ──────────────                 ────────────────

SwingTradeHunter                                                       MetaController
  ├─ _publish_trade_intent()    →     emit_event()              →      recv_signal()
  │                                    └─ Queue event                  └─ Cache & validate
  └─ [events.trade.intent]                │
                                          ├─ Distribute to all
TrendHunter                              │  subscribers
  ├─ _publish_trade_intent()    →     ├─ Notify listeners
  │                                    │
  └─ [events.trade.intent]            └─ Store in history
                                          
ExecutionManager                                                       Dashboard
  ├─ _emit_execution_event()    →     [Various Event Types]      →    _consume_events()
  │                                    ├─ TradeIntent                  └─ Display metrics
  └─ [TRADE_EXECUTED]                 ├─ TRADE_EXECUTED
                                       ├─ TRADE_SKIPPED
PnLCalculator                          ├─ HealthStatus
  ├─ emit_event()               →     ├─ NavReady
  │                                    └─ [100+ more types]
  └─ [valuation_cycle]
```

---

### Pattern 3: Data State Flow (L3 Portfolio)

```
INCOMING DATA:                      STATEMANAGER (L3)              OUTGOING STATE:
──────────────────                  ──────────────────             ────────────────

Order Fills from Binance  →  update_position()         →  SharedState.positions
  ├─ symbol                   ├─ Add/modify/remove pos    ├─ {symbol: {qty, avg_price}}
  ├─ qty                      └─ Update NAV               └─ Updated every fill
  └─ price                    

Price Updates (L2)         →  recalculate_nav()         →  SharedState.nav
  ├─ BTCUSDT: 78,500           ├─ Sum all positions        ├─ NAV = $87.29
  ├─ ETHUSDT: 2,330            ├─ Mark-to-market           └─ Updated every price tick
  └─ More...                   └─ Add USDT cash balance

Historical Trades          →  pnl_calculator()          →  SharedState.pnl_metrics
  ├─ Closed trades            ├─ realized_pnl = -$94.68   ├─ Realized P&L
  ├─ Entry/exit prices        ├─ unrealized_pnl = $0      └─ Unrealized P&L
  └─ Fees                      └─ total_equity = $83.85
```

---

## 📡 Communication Mechanisms

### Type 1: Event Bus (Async Pub/Sub)
```python
# PUBLISH (from any layer):
await shared_state.emit_event("TradeIntent", {
    "symbol": "SOLUSDT",
    "side": "BUY",
    "confidence": 0.65
})

# SUBSCRIBE (in any layer):
async def handle_events():
    queue = await shared_state.subscribe_events("MetaController")
    while True:
        event = await queue.get()
        if event["type"] == "TradeIntent":
            await process_signal(event)
```

### Type 2: Shared State (Sync Read/Write)
```python
# WRITE (from ExecutionManager):
shared_state.positions[symbol] = {
    "qty": 1.5,
    "avg_price": 100.0,
    "entry_time": 1234567890
}
shared_state.nav = 87.29

# READ (from MetaController):
nav = shared_state.nav
positions = shared_state.positions
equity = shared_state.total_equity
```

### Type 3: Dependency Injection
```python
# Layer 8 passes Layer 7 to Layer 6:
MetaController(
    config=config,
    shared_state=shared_state,        # L0
    risk_manager=risk_manager,         # L6
    market_data=market_data_feed,      # L2
    execution_manager=execution_mgr,   # L4
    strategy_manager=strategy_mgr,     # L5
)
```

---

## 🎯 Current Signal Flow Example (Real from Logs)

```
TIMESTAMP: 2026-05-03 22:59:03,063

[1] SwingTradeHunter generates signal
    └─ Logic: EMA20 (84.0442) > EMA50 (83.9604) ✅
              RSI (68.24) < 75 ✅
              Result: BUY confidence=0.65 ✅

[2] Signal published via EventBus
    └─ event_bus.publish('events.trade.intent', TradeIntent(...))
    └─ Event queued for subscribers

[3] MetaController receives signal
    └─ ✓ Signal cached for SOLUSDT (confidence=0.65)
    └─ Cache entry: {symbol: SOLUSDT, confidence: 0.65, age: 0s}

[4] Trade firing loop runs
    └─ Check: 0.65 < 0.75? → YES, below minimum ❌
    └─ Check: net_pct_below_threshold? → YES ❌
    └─ Decision: TRADE_SKIPPED

[5] Event emitted
    └─ event: TRADE_SKIPPED
    └─ reason: pretrade_effect_gate:net_pct_below_threshold
    └─ confidence: 0.667324189706613

[6] Dashboard updates
    └─ TRADE_SKIPPED count: +1
    └─ Display: "Blocked 113 trades today"
    └─ Status: "Waiting for better market conditions"
```

---

## 📊 Layer Coupling & Responsibilities

| Layer | Responsibility | Depends On | Feeds To |
|-------|-----------------|-----------|----------|
| **L8** | Policy & firing | L6, L5, L3 | L7 metrics |
| **L7** | Monitoring | L6, L5, L4, L3 | Dashboard |
| **L6** | Risk & gating | L3, L2 | L8 decisions |
| **L5** | Signal generation | L2, L0 | L4, L8 |
| **L4** | Order lifecycle | L3, L1 | L3 state |
| **L3** | Portfolio state | L1, L4 | L6, L8 |
| **L2** | Market data | L1 | L5, L6 |
| **L1** | Binance API | - | L2, L3 |
| **L0** | Central state & events | - | All layers |

---

## 🔌 Current Communication Issues

### Issue 1: Confidence Value Mismatch
```
L5 → L8: Confidence = 0.65 (OLD CODE)
L8 Check: Requires >= 0.75 minimum
Result: MISMATCH → Signal rejected at validation stage

Fix: Restart system to load new code (confidence = 0.80)
```

### Issue 2: PreTrade Effect Gate (Expected Profit Check)
```
L8 → L6: "Check if this trade has good expected profit"
L6: Expected profit = 0.04%
L6 Check: 0.04% > 0.06%? NO ❌
Result: Gate blocks trade, sends TRADE_SKIPPED

Reason: Market conditions unfavorable (tight spreads)
Solution: Wait for volatility to increase naturally
```

### Issue 3: Event Ordering
```
Sequence (correct):
1. SwingTradeHunter publishes TradeIntent
2. MetaController receives (via EventBus)
3. MetaController validates
4. MetaController fires (or skips)

If async timing off:
- MetaController might fire before signal fully cached
- Could cause "signal not found" errors
- Currently: Working correctly (proper async/await)
```

---

## ✅ Health Metrics

| Component | Status | Communication |
|-----------|--------|---|
| **EventBus** | ✅ Healthy | All events flowing |
| **SharedState** | ✅ Healthy | Read/write working |
| **MetaController** | ✅ Healthy | Processing every cycle |
| **SwingTradeHunter** | ✅ Healthy | Generating signals |
| **ExecutionManager** | ⏳ Idle | Waiting for gate approval |
| **PerformanceMonitor** | ✅ Healthy | Tracking events |
| **RiskManager** | ✅ Healthy | Enforcing gates |

---

## 🎯 Summary: How Layers Communicate

```
       ASYNC             (Event-based, non-blocking)
         ↓
    ┌────────────┐
    │  EventBus  │  ← Pub/Sub messaging (TradeIntent, HealthStatus, etc.)
    └────────────┘
         ↑
    SYNC READ        (Shared memory, immediate)
    ASYNC WRITE      (Event-driven updates)
         ↑
    ┌────────────┐
    │ SharedState│  ← Central state holder
    └────────────┘
         ↑
    FUNCTIONAL      (Dependency injection)
    CALLS
         ↑
    ┌────────────┐
    │   Layers   │  ← Strategy, Execution, Governance
    │   1-8      │     calling methods on each other
    └────────────┘
```

**Result:** Loose coupling (EventBus), tight data consistency (SharedState), fast execution (sync where needed), non-blocking where possible (async where safe).
