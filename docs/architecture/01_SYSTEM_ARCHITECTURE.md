# 🏗️ System Architecture

**Document Version:** 2.0  
**Last Updated:** April 10, 2026  
**Status:** Complete

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         OCTI AI TRADING BOT - SYSTEM ARCHITECTURE           │
└─────────────────────────────────────────────────────────────┘

Layer 1: Market Access
┌─────────────────────────────────────────────────────────────┐
│ Exchange Connections (Binance, etc.)                        │
│ ├─ WebSocket Streams (price, trades, orders)               │
│ ├─ REST APIs (order placement, account info)               │
│ └─ Data Feeds (tickers, candles, order book)              │
└─────────────────────────────────────────────────────────────┘
                            ↑
Layer 2: Data Processing
┌─────────────────────────────────────────────────────────────┐
│ Market Data Feed Module                                     │
│ ├─ Data Normalization                                       │
│ ├─ Event Processing                                         │
│ ├─ Stream Management                                        │
│ └─ Cache Management                                         │
└─────────────────────────────────────────────────────────────┘
                            ↑
Layer 3: Core Trading Engine
┌─────────────────────────────────────────────────────────────┐
│ Core Module - Central Orchestration                         │
│ ├─ Agent Manager (Coordination)                             │
│ ├─ Execution Manager (Trade Execution)                      │
│ ├─ Position Manager (Tracking)                              │
│ ├─ Risk Manager (Safety)                                    │
│ ├─ Cash Router (Capital Management)                         │
│ └─ Policy Manager (Rules & Constraints)                     │
└─────────────────────────────────────────────────────────────┘
                            ↑
Layer 4: Intelligence Layer
┌─────────────────────────────────────────────────────────────┐
│ Agents Module - Decision Making                             │
│ ├─ Research Agent (Analysis)                                │
│ ├─ DIP Sniper (Dips)                                        │
│ ├─ IPA Chaser (IPOs)                                        │
│ ├─ ML Forecaster (Predictions)                              │
│ ├─ Swing Trade Hunter (Swings)                              │
│ ├─ Trend Hunter (Trends)                                    │
│ ├─ Liquidation Agent (Liquidations)                         │
│ └─ Symbol Screener (Symbol Selection)                       │
└─────────────────────────────────────────────────────────────┘
                            ↑
Layer 5: Support Systems
┌─────────────────────────────────────────────────────────────┐
│ Supporting Modules                                          │
│ ├─ Models (ML Models)                                       │
│ ├─ Config (Configuration)                                   │
│ ├─ Utils (Utilities)                                        │
│ ├─ Dashboards (Visualization)                               │
│ └─ Portfolio (Performance Tracking)                         │
└─────────────────────────────────────────────────────────────┘

```

---

## Module Breakdown

### Layer 1: Market Access (`core/exchange_client.py`)

**Responsibilities:**
- Connect to exchange APIs
- Manage WebSocket connections
- Handle authentication
- Manage rate limiting

**Key Classes:**
- `ExchangeClient` - Main exchange interface
- `StreamManager` - WebSocket management

---

### Layer 2: Data Processing (`core/market_data_feed.py`)

**Responsibilities:**
- Normalize market data
- Process real-time updates
- Manage data caches
- Emit data events

**Key Classes:**
- `MarketDataFeed` - Data distribution
- `PriceCache` - Price caching
- `DataProcessor` - Normalization

---

### Layer 3: Core Trading Engine

#### Agent Manager (`core/agent_manager.py`)
- Coordinates all agents
- Manages agent lifecycle
- Routes actions
- Handles priorities

#### Execution Manager (`core/execution_manager.py`)
- Executes trades
- Manages orders
- Handles order lifecycle
- Provides order status

#### Position Manager (`core/position_manager.py`)
- Tracks positions
- Calculates P&L
- Manages closes
- Reports positions

#### Risk Manager (`core/risk_manager.py`)
- Enforces position limits
- Checks drawdown
- Validates orders
- Manages exposure

#### Cash Router (`core/cash_router.py`)
- Allocates capital
- Manages wallet
- Rebalances positions
- Tracks cash flow

---

### Layer 4: Intelligence Layer (`agents/`)

Each agent implements `Agent` interface:

```python
class Agent:
    async def analyze()  # Market analysis
    async def signal()   # Generate signals
    async def execute()  # Execute strategy
```

**8 Agents:**
1. **DIP Sniper** - Buy significant dips
2. **IPO Chaser** - Catch IPO momentum
3. **ML Forecaster** - ML-based predictions
4. **Swing Trade Hunter** - Swing trading
5. **Trend Hunter** - Ride trends
6. **Liquidation Agent** - Liquidation events
7. **Symbol Screener** - Find opportunities
8. **Wallet Scanner** - Wallet movements

---

### Layer 5: Support Systems

#### Models (`models/`)
- Price prediction
- Signal generation
- Risk assessment
- Feature engineering

#### Config (`config/`)
- API credentials
- Trading parameters
- Risk limits
- System settings

#### Utils (`utils/`)
- Data fetching
- Data processing
- Monitoring
- Logging

#### Portfolio (`portfolio/`)
- Performance tracking
- Analytics
- Reporting

---

## Data Flow

```
Market Updates (WebSocket)
        ↓
Market Data Feed
    ├─ Cache prices
    ├─ Emit events
        ↓
Agents (receive events)
    ├─ Analyze market
    ├─ Generate signals
    ├─ Create orders
        ↓
Execution Manager (receive orders)
    ├─ Validate with Risk Manager
    ├─ Check capital availability
    ├─ Place orders
    ├─ Update positions
        ↓
Position Manager (update tracking)
    ├─ Update P&L
    ├─ Track performance
        ↓
Dashboards & Reports
    ├─ Display status
    ├─ Show performance
```

---

## Key Design Patterns

### 1. **Agent Pattern**
- Multiple autonomous agents
- Each with specific strategy
- Coordinated through manager

### 2. **Event-Driven Architecture**
- Market data triggers updates
- Async/await throughout
- Non-blocking operations

### 3. **Layered Architecture**
- Clear separation of concerns
- Modular components
- Easy to test and maintain

### 4. **Manager/Worker Pattern**
- Central manager coordinates
- Workers execute tasks
- Communication through APIs

### 5. **Risk Management Pattern**
- Checks at every step
- Multiple safeguards
- Fail-safe defaults

---

## Configuration Management

```
Config Module
├── API Keys (from .env)
├── Trading Parameters
│   ├─ Position sizes
│   ├─ Risk limits
│   ├─ Stop losses
│   └─ Take profits
├── System Settings
│   ├─ Logging level
│   ├─ Cache settings
│   ├─ Retry policies
│   └─ Timeouts
└── Agent Configuration
    ├─ Enabled/disabled
    ├─ Strategies
    └─ Parameters
```

---

## Scalability Considerations

### Horizontal Scaling
- Multiple agents can run independently
- Data feed can be sharded
- Execution can be parallel

### Performance Optimizations
- Caching frequently accessed data
- Async I/O throughout
- Connection pooling
- Batch operations

### Monitoring & Observability
- Structured logging
- Performance metrics
- System health checks
- Alert mechanisms

---

## Security Architecture

### API Security
- API keys in environment variables
- Rate limiting enforcement
- Request validation
- Response verification

### Data Security
- Order encryption
- Position tracking
- Cash tracking
- Audit logging

### Risk Controls
- Position limits per agent
- Portfolio limits
- Daily loss limits
- Circuit breakers

---

## Deployment Architecture

```
Production Environment
├── Main Bot Instance
│   ├─ Agents
│   ├─ Core Engine
│   └─ Connections
├── Monitoring
│   ├─ Dashboards
│   ├─ Logging
│   └─ Alerts
└── Support
    ├─ Database (optional)
    ├─ Cache (Redis)
    └─ Message Queue (optional)
```

---

## Related Documentation

- [Module Structure](./02_MODULE_STRUCTURE.md)
- [Agent Interactions](./03_AGENT_INTERACTIONS.md)
- [Data Flow Details](./04_DATA_FLOW.md)
- [Configuration Reference](../reference/02_CONFIG_REFERENCE.md)

---

**Next:** → [Module Structure](./02_MODULE_STRUCTURE.md)
