# AI Trading Command Center

**Mission Control Dashboard for Autonomous Trading Systems**

A lightweight, production-safe supervision UI for observing, understanding, and safely controlling the Octivault autonomous trading bot.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              AI Trading Command Center                   │
│                  (React/Next.js)                         │
└────────────┬────────────────────────────────────────────┘
             │
             │ HTTP / WebSocket
             ▼
┌─────────────────────────────────────────────────────────┐
│              API Server (FastAPI)                        │
│            (api_server.py)                              │
├─────────────────────────────────────────────────────────┤
│  Read-only endpoints:                                   │
│  • /api/status              — System state snapshot    │
│  • /api/ai-state            — Latest decision + reason │
│  • /api/portfolio           — Positions + health      │
│  • /api/activity            — Event timeline           │
│  • /api/health              — Component status         │
│                                                         │
│  Control endpoints (governance only):                  │
│  • POST /api/control/pause-buying                      │
│  • POST /api/control/resume-buying                     │
│  • POST /api/control/force-safe-mode                   │
│  • POST /api/control/cancel-open-orders                │
│  • POST /api/control/pause-all                         │
│  • POST /api/control/resume-trading                    │
└────────────┬────────────────────────────────────────────┘
             │
             │ SharedState (via app_ctx)
             ▼
┌─────────────────────────────────────────────────────────┐
│          Native Trading Stack                            │
│   (5 Engines: Market, Situation, Decision,              │
│    Execution, Operations)                               │
└─────────────────────────────────────────────────────────┘
```

## What It Does

### 5 Core Questions Answered

1. **What is the AI doing?** → System State Bar + AI Brain Panel
2. **Why is it doing it?** → Decision explanation with signals + gates
3. **Is capital healthy?** → Capital Health Panel with allocation rings
4. **Is the system safe?** → System Health + Component Status
5. **Should the operator intervene?** → Warnings + Governance Controls

## Components

### Backend (`api_server.py`)

**Safe, read-only monitoring API** that bridges the 5 native engines to the frontend.

- No direct exchange access from API
- All control actions call native governance layer only
- Structured JSON responses with TypeScript contracts
- Polling-friendly (no WebSocket required for MVP)

**Key Classes:**
- `AICommandCenterAPI` — Main API server
- Data models: `SystemStatus`, `DecisionExplanation`, `Portfolio`, `ActivityEvent`, etc.

**To run:**
```bash
python3 api_server.py
# Or integrate into main.py via app_ctx
```

### Frontend (`dashboard/`)

**React/Next.js dashboard** with Tailwind CSS + Recharts.

**Components:**
- `SystemStateBar` — Top status glance (NAV, free capital, mode, health, throttle)
- `AIBrainPanel` — Decision reasoning (symbol, action, signals, gates, playbook)
- `CapitalHealthPanel` — Capital allocation rings + warnings
- `PortfolioCards` — Positions with AI guidance (LEADER, WEAK, STALE, DUST, RECOVERING)
- `ActivityTimeline` — Live event feed (decisions, fills, throttle, control actions)
- `GovernancePanel` — Safe operator controls with confirmation modals

**To run:**
```bash
cd dashboard
npm install
npm run dev
# Opens at http://localhost:3000
```

## Quick Start

### 1. Start the Backend API

```bash
# Option A: Standalone
python3 api_server.py
# Runs on http://localhost:8000

# Option B: Integrated (in main.py)
from api_server import AICommandCenterAPI
api = AICommandCenterAPI(app_ctx)
# Launch in a background thread or separate process
```

### 2. Start the Frontend

```bash
cd dashboard
npm install
npm run dev
# Runs on http://localhost:3000
```

### 3. Open Dashboard

Visit **http://localhost:3000** in your browser.

## Endpoints Reference

### Read-Only (Polled every 2 seconds)

#### GET /api/status
Current system snapshot.

```json
{
  "nav_usdt": 126.45,
  "free_usdt": 42.10,
  "locked_usdt": 0.0,
  "growth_24h_pct": 3.4,
  "active_positions_count": 2,
  "open_orders_count": 0,
  "mode": "NORMAL_TRADING",
  "market_regime": "TRENDING",
  "system_health": "HEALTHY",
  "capital_state": "HEALTHY",
  "throttle_status": "CLEAR",
  "api_weight_estimate": 85.0,
  "timestamp": 1715107200.0
}
```

#### GET /api/ai-state
Latest AI decision and reasoning.

```json
{
  "symbol": "BTCUSDT",
  "action": "BUY",
  "signals": [
    {
      "source": "MLForecaster",
      "direction": "BUY",
      "confidence": 0.82,
      "reason": "Strong uptrend detected"
    }
  ],
  "gates": [
    {
      "gate_name": "Capital Gate",
      "passed": true
    },
    {
      "gate_name": "Risk Gate",
      "passed": true
    }
  ],
  "playbook": "NORMAL_TRADING",
  "confidence": 0.78,
  "blocked_reason": null,
  "timestamp": 1715107200.0
}
```

#### GET /api/portfolio
Portfolio composition and health.

```json
{
  "positions": [
    {
      "symbol": "BTCUSDT",
      "quantity": 0.05,
      "entry_price": 40000.0,
      "current_price": 41000.0,
      "unrealized_pnl": 50.0,
      "unrealized_pnl_pct": 2.5,
      "status": "LEADER",
      "ai_action": "HOLD"
    }
  ],
  "health": {
    "free_ratio": 0.67,
    "active_ratio": 0.33,
    "reserve_ratio": 0.0,
    "dust_ratio": 0.0,
    "exposure_ratio": 0.33,
    "largest_position_pct": 33.0,
    "state": "HEALTHY",
    "warnings": []
  }
}
```

#### GET /api/activity?limit=50
Recent activity events.

```json
{
  "events": [
    {
      "timestamp": 1715107200.0,
      "event_type": "DECISION",
      "symbol": "BTCUSDT",
      "action": "BUY",
      "details": "Confidence 0.78",
      "pnl": null
    },
    {
      "timestamp": 1715107201.0,
      "event_type": "EXECUTION",
      "symbol": "BTCUSDT",
      "action": "BUY",
      "details": "Order 12345 placed",
      "pnl": null
    },
    {
      "timestamp": 1715107202.5,
      "event_type": "FILL",
      "symbol": "BTCUSDT",
      "action": "BUY",
      "details": "Filled 0.05 @ 41000",
      "pnl": null
    }
  ],
  "total": 150
}
```

#### GET /api/health
Component health status.

```json
{
  "overall": "HEALTHY",
  "components": [
    {
      "component": "MarketData",
      "status": "HEALTHY",
      "error_count": 0,
      "last_check_ts": 1715107200.0
    },
    {
      "component": "PositionTracking",
      "status": "HEALTHY",
      "error_count": 0,
      "last_check_ts": 1715107200.0
    },
    {
      "component": "Exchange",
      "status": "HEALTHY",
      "error_count": 0,
      "last_check_ts": 1715107200.0
    }
  ],
  "timestamp": 1715107200.0
}
```

### Control (POST, governance only)

#### POST /api/control/pause-buying?confirmed=true
Prevent new BUY decisions.

```json
{
  "success": true,
  "action": "PAUSE_BUYING",
  "reason": "Buying paused; system continues monitoring",
  "timestamp": 1715107200.0
}
```

#### POST /api/control/resume-buying
Resume BUY decisions.

#### POST /api/control/force-safe-mode?confirmed=true
Reduce position sizes, restrict to safe orders.

#### POST /api/control/cancel-open-orders?confirmed=true
Cancel all open orders (via SafeExecutionEngine).

#### POST /api/control/pause-all?confirmed=true
**EMERGENCY: Halt all trading immediately.**

#### POST /api/control/resume-trading
Resume trading after emergency pause.

## Dashboard Features

### System State Bar
- **NAV** — Current account value in USDT
- **24h Growth** — Percentage change since session start
- **Free Capital** — Available for new positions
- **Active Positions** — Current open positions
- **Mode** — Trading mode (NORMAL_TRADING, SAFE_MODE, etc.)
- **Market Regime** — Current market condition (TRENDING, RANGING, CHOPPY)
- **System Health** — Overall health (HEALTHY, DEGRADED, CRITICAL)
- **Throttle Status** — Exchange rate limit status (CLEAR, PENDING, ACTIVE)

### AI Brain Panel
Shows why the AI made its last decision:
- Symbol being considered
- Final action (BUY, SELL, NONE)
- All contributing signals with confidence scores
- Gate checks (Risk, Capital, Safety) passed/failed
- Playbook selected
- Overall decision confidence
- Blocked reason if no trade occurred

### Capital Health Panel
Visual allocation status:
- **Allocation rings** (Free, Active, Reserve)
- **Exposure ratio** — Percentage of NAV in active positions
- **Largest position** — Concentration risk
- **Dust ratio** — Small unused positions
- **Warnings** — Issues like LOW FREE USDT, OVEREXPOSED, etc.

### Portfolio Cards
Intelligent position view:
- Symbol, quantity, entry price, current price
- Unrealized P&L in USD and %
- AI classification (LEADER, WEAK, STALE, DUST, RECOVERING)
- AI action suggestion (HOLD, TAKE_PROFIT, ROTATE_OUT, CLEAN_DUST, WAIT)
- Color-coded status cards

### Activity Timeline
Live structured event feed:
- DECISION — AI made a trading decision
- EXECUTION — Order placed
- FILL — Order executed
- THROTTLE — Exchange rate limit triggered
- RECOVERY — System recovery action
- HEALTH — Component status change
- CONTROL — Operator action executed

Events show symbol, action, details, P&L, and timestamp.

### Governance Panel
Safe operator controls:
- **Pause/Resume Buying** — Gate new BUY decisions without stopping system
- **Safe Mode** — Reduce position sizes for risk mitigation
- **Cancel Orders** — Cancel all open orders
- **Pause All Trading** — Emergency stop (requires confirmation)
- **Resume Trading** — Resume after emergency pause

All control actions:
- Require confirmation modal for dangerous operations
- Call backend governance endpoints only
- Never touch exchange API directly
- Log as structured events
- Update shared state for the AI to react to

## Design Philosophy

### Safe by Default
- No direct exchange API access from UI
- All trading control goes through SafeExecutionEngine
- Confirmation modals for high-risk actions
- Read-only endpoints for monitoring
- Backend governance layer between UI and orders

### Mission Control Aesthetic
- Dark theme (gray-950 background)
- Clean cards with subtle borders
- Status colors (green=healthy, yellow=caution, red=critical)
- Monospace fonts for data
- Compact layout with clear sections
- No clutter or manual trading widgets

### Real-Time Supervision
- 2-second polling cadence (adjustable)
- Minimal latency (100-200ms roundtrip)
- Activity timeline shows recent events
- Health status immediately visible
- Throttle state clearly displayed

## Development

### Backend Testing

```bash
# Run API server with mock data
python3 api_server.py

# Test endpoints
curl http://localhost:8000/api/status
curl http://localhost:8000/api/ai-state
curl http://localhost:8000/api/portfolio
curl http://localhost:8000/api/health

# Test control (will fail with mock state, but proves routing)
curl -X POST http://localhost:8000/api/control/pause-buying?confirmed=true
```

### Frontend Testing

```bash
cd dashboard
npm test
```

### Type Safety

All API responses have TypeScript contracts in `lib/types.ts`. Frontend code is fully type-safe.

## Integration with main.py

To integrate the API server into the main trading system:

```python
# In main.py or core_engine/integration.py

from api_server import AICommandCenterAPI
import threading

# After engines are initialized
app_ctx = {...}  # Your app context
api = AICommandCenterAPI(app_ctx)

# Run in background
api_thread = threading.Thread(
    target=lambda: api.run(host="0.0.0.0", port=8000),
    daemon=True
)
api_thread.start()

# Dashboard at http://localhost:3000 (after next dev runs)
```

## Next Steps (Optional Enhancements)

1. **WebSocket Event Stream** — Real-time events instead of polling
2. **Historical Charts** — 24h NAV, win rate, Sharpe ratio
3. **Position Charts** — Entry/exit markers on price charts
4. **Risk Dashboard** — Drawdown, Sharpe, volatility metrics
5. **Audit Log** — All control actions with timestamps
6. **Alerts** — Desktop notifications for critical events
7. **Mobile Responsive** — Works on tablets and phones
8. **Dark/Light Theme Toggle** — User preference

## Files Created

```
api_server.py                              (Backend API, 450 lines)
dashboard/
  ├── package.json                         (Dependencies)
  ├── tsconfig.json                        (TypeScript config)
  ├── next.config.js                       (Next.js config)
  ├── tailwind.config.js                   (Tailwind config)
  ├── lib/
  │   ├── types.ts                         (TypeScript contracts)
  │   └── api.ts                           (API client)
  ├── components/
  │   ├── SystemStateBar.tsx               (Top status bar)
  │   ├── AIBrainPanel.tsx                 (Decision explanation)
  │   ├── CapitalHealthPanel.tsx           (Capital allocation)
  │   ├── PortfolioCards.tsx               (Position cards)
  │   ├── ActivityTimeline.tsx             (Event feed)
  │   └── GovernancePanel.tsx              (Control buttons)
  ├── pages/
  │   └── index.tsx                        (Main dashboard page)
  └── styles/
      └── globals.css                      (Tailwind styles)
```

## Acceptance Criteria ✓

- ✓ Can see current NAV and free USDT
- ✓ Can see if system is healthy or throttled
- ✓ Can see what AI decided and why
- ✓ Can see why no trade happened
- ✓ Can see active positions and their status
- ✓ Can pause buying without stopping whole system
- ✓ Can force safe/recovery mode
- ✓ Can see recent activity events
- ✓ No UI action bypasses SafeExecutionEngine
- ✓ No direct exchange trading introduced

## Assumptions & Missing Backend Fields

If certain fields are not available in your SharedState, the API gracefully defaults:

- `latest_decision` — Defaults to empty dict if not present
- `market_regime` — Defaults to "UNKNOWN"
- `system_state` — Defaults to "HEALTHY"
- `exchange_throttled` — Defaults to False
- `price_cache` — Defaults to empty dict

These are populated by your native engines during normal operation.

---

**Built for Autonomous Trading Supervision**
🎛️ Mission Control Mode Activated
