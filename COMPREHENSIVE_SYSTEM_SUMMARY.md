# 🎯 OCTI AI TRADING BOT - COMPREHENSIVE SYSTEM SUMMARY

**Last Updated:** 2026-02-14 | **Status:** OPERATIONAL | **Version:** Phase 2C+ Extended

---

## 📋 TABLE OF CONTENTS

1. [System Architecture Overview](#system-architecture-overview)
2. [Core Operational Flows](#core-operational-flows)
3. [Decision-Making Pipeline](#decision-making-pipeline)
4. [Trading Execution Sequence](#trading-execution-sequence)
5. [Capital Management System](#capital-management-system)
6. [Risk Management Layers](#risk-management-layers)
7. [Dust & Portfolio Recovery](#dust--portfolio-recovery)
8. [Key Components & Modules](#key-components--modules)
9. [Configuration & Mode System](#configuration--mode-system)
10. [Monitoring & Observability](#monitoring--observability)
11. [Quick Reference Commands](#quick-reference-commands)

---

## SYSTEM ARCHITECTURE OVERVIEW

### 7-Layer System Stack

```
┌────────────────────────────────────────────────────────────┐
│ LAYER 7: MASTER ORCHESTRATOR (🎯_MASTER_SYSTEM_ORCHESTRATOR.py)
│ └─ Lifecycle management, prerequisite validation, shutdown
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 6: WATCHDOG & HEALTH MONITOR
│ └─ System health, component status, alerting
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 5: SIGNAL PROCESSING & FUSION
│ ├─ Market Data Feed → Signal Generation
│ ├─ Signal Manager (cache, deduplication)
│ ├─ Signal Fusion (composite_edge mode, multi-agent consensus)
│ └─ Signal Batcher (reduces churn by 75%)
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 4: DECISION-MAKING ENGINE
│ ├─ MetaController (evaluation + gating)
│ ├─ Policy Manager (nudges, mode x policy matrix)
│ ├─ Arbitration Engine (6-layer signal arbitration)
│ ├─ NAV Regime Manager (MICRO_SNIPER mode enforcement)
│ └─ Lifecycle Manager (symbol state machine)
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 3: EXECUTION ENGINE
│ ├─ Execution Manager (order placement & monitoring)
│ ├─ Execution Logic (routing, position sizing)
│ ├─ Scaling Manager (dynamic position sizing)
│ ├─ Capital Governor (position limiting)
│ ├─ TP/SL Engine (stop-loss & take-profit)
│ └─ Rotation Authority (symbol rotation decisions)
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 2: CAPITAL & PORTFOLIO MANAGEMENT
│ ├─ Balance Manager (balance validation)
│ ├─ Capital Allocator (quota allocation)
│ ├─ Capital Velocity Optimizer (allocation planning)
│ ├─ Portfolio Authority (concentration limits)
│ ├─ Dust Manager (consolidation, healing)
│ └─ Recovery Engine (starvation recovery)
└────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────┐
│ LAYER 1: EXCHANGE INTERFACE & STATE MANAGEMENT
│ ├─ Exchange Client (Binance API)
│ ├─ Shared State (position tracking, metrics, cache)
│ ├─ Market Data Feed (real-time prices & candles)
│ └─ Polling Coordinator (async polling management)
└────────────────────────────────────────────────────────────┘
```

---

## CORE OPERATIONAL FLOWS

### Flow 1: System Bootstrap (On Startup)

```
START
  ↓
1. Validate Prerequisites
   ├─ Check environment variables (APPROVE_LIVE_TRADING)
   ├─ Verify API credentials (Binance)
   ├─ Check account status (not liquidated, balance > 0)
   └─ Validate configuration consistency
  ↓
2. Load Configuration
   ├─ Parse .env and config.py
   ├─ Load exchange rules (MIN_NOTIONAL, step sizes)
   └─ Initialize mode settings (BOOTSTRAP, NORMAL, SAFE, etc.)
  ↓
3. Initialize Exchange Interface
   ├─ Create ExchangeClient
   ├─ Fetch account balances
   ├─ Fetch open positions/orders
   └─ Establish WebSocket feeds
  ↓
4. Initialize Shared State
   ├─ Load position snapshot from exchange
   ├─ Initialize metrics (NAV, PnL, drawdown)
   ├─ Seed price cache
   └─ Set up event subscribers
  ↓
5. Initialize Signal Pipeline
   ├─ Create SignalManager (cache for signals)
   ├─ Create SignalFusion (composite_edge mode)
   ├─ Create Signal Batcher (reduces churn)
   └─ Start AgentManager (TrendHunter, DipSniper, etc.)
  ↓
6. Initialize Decision Layer
   ├─ Create MetaController
   ├─ Create PolicyManager
   ├─ Create Arbitration Engine
   └─ Create Lifecycle Manager
  ↓
7. Initialize Execution Layer
   ├─ Create ExecutionManager
   ├─ Create Execution Logic
   ├─ Create Scaling Manager
   ├─ Create Capital Governor
   └─ Create TP/SL Engine
  ↓
8. Initialize Capital Management
   ├─ Create Balance Manager
   ├─ Create Capital Allocator
   ├─ Create Capital Velocity Optimizer
   └─ Create Portfolio Authority
  ↓
9. Initialize Monitors & Health
   ├─ Create HealthMonitor
   ├─ Create Watchdog
   ├─ Create Heartbeat
   └─ Create Performance Monitor (if available)
  ↓
10. Start All Polling Tasks
    ├─ Market data polling (prices, candles)
    ├─ Balance polling
    ├─ Position polling
    ├─ Order status polling
    ├─ Trade history polling
    └─ Metrics calculation
  ↓
11. Run Main Evaluation Loop (MetaController)
    └─ [See: Flow 2 - Main Evaluation Loop]
```

### Flow 2: Main Evaluation Loop (Per-Cycle)

```
MAIN LOOP (runs every 100-500ms)
  ↓
1. Tick Increment & Cycle Start
   ├─ Increment tick counter
   ├─ Record cycle start time
   └─ Check for timeout watchdog
  ↓
2. Drain Market Events
   ├─ Process price updates
   ├─ Process trade fills
   ├─ Process balance changes
   ├─ Process order status updates
   └─ Update SharedState metrics
  ↓
3. Guard Evaluation (Readiness Checks)
   ├─ Check market data ready
   ├─ Check balances available
   ├─ Check ops plane ready
   ├─ Check trading hours
   ├─ Check exchange connectivity
   └─ If ANY fail → RETURN (no trading this cycle)
  ↓
4. Signal Intake & Cache
   ├─ Retrieve new signals from SignalManager
   ├─ Apply confidence floor filter (MIN_SIGNAL_CONF)
   ├─ Age-out stale signals (MAX_SIGNAL_AGE_SECONDS)
   ├─ Deduplicate signals (per symbol, per side)
   └─ Enrich signals with metadata (tag, reason, agent)
  ↓
5. Batch Collection & Processing
   ├─ Collect signals into batch (up to 50)
   ├─ Apply fade filtering (remove low-confidence)
   ├─ Sort by confidence (highest first)
   └─ Generate decisions from batch
  ↓
6. Decision Arbitration (Per-Signal)
   For each (symbol, side, signal):
      ├─ GATE 1: Lifecycle State Check
      │  ├─ Get symbol lifecycle (DUST_HEALING, ROTATION_PENDING, etc.)
      │  └─ If state blocks this authority → SKIP
      ├─ GATE 2: Portfolio Health Check
      │  ├─ Classify positions (SIGNIFICANT vs DUST)
      │  ├─ Count open positions
      │  └─ Check position limits
      ├─ GATE 3: Capital Availability Check
      │  ├─ Calculate free quote available
      │  ├─ Check allocation quota
      │  └─ Check min notional floor
      ├─ GATE 4: Economic Gate
      │  ├─ Calculate expected alpha
      │  ├─ Check against MIN_EXPECTED_EDGE_BPS
      │  └─ Apply fee & slippage adjustment
      ├─ GATE 5: Signal Confidence Gate
      │  ├─ Check confidence >= regime minimum
      │  └─ Check expected move >= regime minimum
      ├─ GATE 6: Regime Gating
      │  ├─ Check MICRO_SNIPER mode restrictions (NAV < 1000)
      │  ├─ Check max positions limit
      │  └─ Check daily trade limit
      └─ If ALL gates pass → DECISION APPROVED
  ↓
7. Trade Intent Generation
   For each APPROVED decision:
      ├─ Determine action (BUY, SELL, SKIP)
      ├─ Calculate position size
      ├─ Create TradeIntent object
      └─ Queue for execution
  ↓
8. Execution Dispatch
   For each TradeIntent:
      ├─ Route through Execution Logic
      ├─ Place order with ExecutionManager
      ├─ Monitor for fill status
      └─ Update SharedState on completion
  ↓
9. Post-Execution Bookkeeping
   ├─ Record executed trades
   ├─ Update metrics (NAV, PnL, etc.)
   ├─ Update TP/SL levels
   ├─ Check for forced rotations
   └─ Reset per-cycle counters
  ↓
10. Cycle End & Logging
    ├─ Calculate cycle duration
    ├─ Emit performance metrics
    ├─ Log cycle summary
    └─ RETURN to step 1 (next cycle)
```

### Flow 3: TP/SL Management (Continuous)

```
TP/SL ENGINE (runs in parallel, every 50-200ms)
  ↓
1. Poll Open Positions
   ├─ Fetch current positions
   ├─ Filter bot-managed positions
   └─ Exclude TP/SL exit tags
  ↓
2. For Each Position
   ├─ Retrieve TP/SL levels from config
   ├─ Fetch current price
   ├─ Calculate P&L vs entry
   ├─ Check if TP hit (P&L >= TP_PERCENT)
   ├─ Check if SL hit (P&L <= -SL_PERCENT)
   └─ If hit → Generate exit signal
  ↓
3. Exit Signal Generation
   ├─ Create SELL signal with tag="tp_sl"
   ├─ Inject into signal pipeline
   └─ Route to MetaController for execution
  ↓
4. Execute Exit
   ├─ Close position
   ├─ Update lifecycle (ROTATION_PENDING)
   ├─ Freeze dust healing (600s cooldown)
   └─ Record trade in history
```

### Flow 4: Dust & Portfolio Recovery (Scheduled)

```
PORTFOLIO RECOVERY (every 60s cycle)
  ↓
1. Dust Detection
   ├─ Count total positions
   ├─ Classify as SIGNIFICANT vs DUST
   ├─ Classify as PERMANENT_DUST (< $1 USDT)
   └─ Calculate dust ratio
  ↓
2. Dust Consolidation (If Enabled)
   ├─ Group small positions (< $25 USDT)
   ├─ Coalesce into single order
   ├─ Execute consolidation merge
   └─ Track consolidation metrics
  ↓
3. Dust Healing (If Health < 80%)
   ├─ Check if DUST_HEALING lifecycle allowed
   ├─ Calculate dust escape price (cost basis + buffer)
   ├─ Set conditional exit orders
   └─ Monitor for break-even exit
  ↓
4. Emergency Liquidation (If Dust > 60%)
   ├─ Activate PHASE_2_GUARD
   ├─ Force SELL non-essential positions
   ├─ Prioritize sacrifice coins (DOGE, SHIB, PEPE)
   ├─ Recover capital to base
   └─ Reset dust tracking
  ↓
5. Capital Recovery (If Starvation Detected)
   ├─ Check available capital < MIN_CAPITAL threshold
   ├─ Identify over-concentrated positions
   ├─ Force rotation (SELL concentrated, BUY new opportunities)
   └─ Restore capital velocity
```

---

## DECISION-MAKING PIPELINE

### Signal Ingestion & Filtering

```
Signal Source (Agent → SignalManager)
  ↓
Step 1: Confidence Floor Filter
  ├─ MIN_SIGNAL_CONF: 0.50 (ingest level)
  ├─ Reject if confidence < 0.50
  └─ DIAGNOSTIC: Lowered to 0.50 for maximum capture
  ↓
Step 2: Age Filter
  ├─ MAX_SIGNAL_AGE_SECONDS: 60
  ├─ Reject if signal older than 60s
  └─ Prevents stale trade decisions
  ↓
Step 3: Deduplication
  ├─ Per symbol, per side (one BUY, one SELL)
  ├─ Keep highest confidence
  ├─ Drop duplicates within 1 cycle
  └─ Reduces churn by 40-50%
  ↓
Step 4: Signal Fusion (composite_edge mode)
  ├─ Collect signals from multiple agents
  ├─ Apply consensus voting (threshold: 0.60)
  ├─ Calculate composite confidence
  ├─ Aggregate edge estimates
  └─ Enhanced signal quality & edge measurement
  ↓
Output: Processed Signal Batch
```

### Arbitration Engine (6-Layer Gating)

```
Signal → Arbitration Engine
  ├─ GATE 1: CONFIDENCE CHECK
  │  ├─ Regime minimum confidence (MICRO_SNIPER: 0.50, STANDARD: 0.55, MULTI_AGENT: 0.60)
  │  ├─ Check: confidence >= regime_min
  │  └─ DIAGNOSTIC: Lowered thresholds to force signal flow
  │
  ├─ GATE 2: EXPECTED_MOVE CHECK
  │  ├─ Regime minimum move (MICRO_SNIPER: 0.15%, STANDARD: 0.25%, MULTI_AGENT: 0.35%)
  │  ├─ Fee-aware minimum (move > 2x round-trip fees)
  │  └─ Check: expected_move >= regime_min
  │
  ├─ GATE 3: POSITION_LIMIT CHECK
  │  ├─ Max open positions per regime
  │  ├─ Count bot-managed positions (excluding TP/SL, rebalance, liquidation)
  │  └─ Check: open_positions < max_positions
  │
  ├─ GATE 4: DAILY_TRADE_LIMIT CHECK
  │  ├─ Max trades per day per regime
  │  ├─ Track executed trades
  │  └─ Check: executed_today < max_per_day
  │
  ├─ GATE 5: REGIME_STATE CHECK
  │  ├─ MICRO_SNIPER: NAV < 1000
  │  │  ├─ Max 1 position
  │  │  ├─ Max 1 symbol
  │  │  ├─ Disable rotation
  │  │  └─ Disable dust healing (strict capital preservation)
  │  ├─ STANDARD: NAV 1000-5000
  │  │  ├─ Max 2 positions
  │  │  ├─ Max 3 symbols
  │  │  ├─ Enable rotation (60s cooldown)
  │  │  └─ Enable dust healing
  │  └─ MULTI_AGENT: NAV >= 5000
  │     ├─ Max 3+ positions
  │     ├─ Max 5+ symbols
  │     ├─ Enable rotation (30s cooldown)
  │     └─ Enable dust healing
  │
  └─ GATE 6: ECONOMIC_GATE CHECK
     ├─ Check expected alpha >= MIN_EXPECTED_EDGE_BPS
     ├─ Apply fee & slippage adjustment
     ├─ Ensure profitability after costs
     └─ Prevent sub-economic churn

Result: PASS (execute) or FAIL (skip) + reason
```

### Policy Manager & Mode x Policy Matrix

```
System Metrics → Policy Manager
  ↓
1. Evaluate Metrics:
   ├─ Velocity: capital_velocity ($/hour)
   ├─ Drawdown: current_drawdown_pct
   ├─ Volatility: realized_volatility
   ├─ Capital: available capital / NAV
   └─ Signal: signal_quality score
  ↓
2. Determine Mode:
   ├─ BOOTSTRAP: First position acquisition
   ├─ NORMAL: Steady-state operation
   ├─ SAFE: Drawdown > 5%
   ├─ PROTECTIVE: Volatility spike
   ├─ AGGRESSIVE: Capital abundant, edge high
   ├─ RECOVERY: Drawdown > 20%
   ├─ PAUSED: Manual pause or guard failure
   └─ SIGNAL_ONLY: Observation mode
  ↓
3. Apply Mode x Policy Matrix:
   ┌─────────────────────────────────────────────────────┐
   │ Mode         │ Velocity │ Drawdown │ Volatility │    │
   ├─────────────────────────────────────────────────────┤
   │ SAFE         │   0.0    │   1.0    │   1.0      │    │
   │ PROTECTIVE   │   0.0    │   1.0    │   1.0      │    │
   │ NORMAL       │   0.5    │   1.0    │   1.0      │    │
   │ AGGRESSIVE   │   1.0    │   0.5    │   0.5      │    │
   │ RECOVERY     │   0.0    │   1.0    │   0.0      │    │
   │ BOOTSTRAP    │   0.5    │   1.0    │   1.0      │    │
   └─────────────────────────────────────────────────────┘
  ↓
4. Calculate Nudges:
   ├─ confidence_nudge: additive adjustment to signal confidence
   ├─ cooldown_nudge: additional wait time between trades
   ├─ trade_size_multiplier: scale position size (0.5x to 2.0x)
   └─ max_positions_nudge: adjust open position limit (+/- 1)
  ↓
Output: Policy Nudges (applied to active trades)
```

---

## TRADING EXECUTION SEQUENCE

### BUY Execution Flow

```
1. Signal Arrives
   └─ side=BUY, symbol=BTCUSDT, confidence=0.72, expected_move=0.35%
  ↓
2. Pre-Execution Validation
   ├─ Check position not already held (or allowed accumulation)
   ├─ Check balance available
   ├─ Check min notional feasible (with buffer)
   ├─ Check trading hours
   └─ Check not in manual pause
  ↓
3. Quote Resolution
   ├─ Get planned quote from ScalingManager
   ├─ Apply dynamic sizing (based on confidence)
   ├─ Round to min notional (with 2 USDT buffer)
   ├─ Apply max spend cap
   └─ Resolved Quote: 30 USDT
  ↓
4. Create TradeIntent
   ├─ symbol: BTCUSDT
   ├─ side: BUY
   ├─ quantity: calculated from quote + current price
   ├─ planned_quote: 30 USDT
   ├─ confidence: 0.72
   ├─ agent: TrendHunter
   ├─ tag: meta/TrendHunter
   └─ trace_id: unique decision ID
  ↓
5. Atomic Position Check (Race Prevention)
   ├─ Acquire symbol lock
   ├─ Check if position exists (blocks duplicate entries)
   ├─ Reserve symbol
   └─ Release lock
  ↓
6. Order Submission
   ├─ Call ExecutionManager.place_order()
   ├─ Submit to exchange API
   ├─ Receive order_id from Binance
   ├─ Record in pending orders
   └─ Status: PLACED
  ↓
7. Order Monitoring
   ├─ Poll order status every 100-200ms
   ├─ Track fills (partial or complete)
   ├─ When filled → Update position in SharedState
   └─ Status: FILLED
  ↓
8. Post-Fill Bookkeeping
   ├─ Update open_trades tracking
   ├─ Calculate average entry price
   ├─ Set initial TP/SL levels
   ├─ Update portfolio metrics
   ├─ Record in trade history
   └─ Emit trade event
  ↓
Output: Trade recorded, position active
```

### SELL Execution Flow

```
1. Sell Signal Arrives
   └─ side=SELL, symbol=BTCUSDT, reason=TP_HIT, tag=tp_sl
  ↓
2. Pre-Execution Validation
   ├─ Check position exists & significant (> $0.50)
   ├─ Check not in lifecycle lock (DUST_HEALING blocks SELL? NO)
   ├─ Check not in FOCUS_MODE but non-focus symbol (YES = block)
   └─ Check not in reentry lock
  ↓
3. Lifecycle Gating
   ├─ Get lifecycle state for symbol
   ├─ If state is DUST_HEALING:
   │  ├─ Check age in DUST_HEALING
   │  ├─ If < 5 min → SKIP (wait for accumulation or break-even)
   │  └─ If >= 5 min → ALLOW (dust timeout expired)
   └─ If state is other → ALLOW
  ↓
4. Exit Gate Validation
   ├─ PROFIT GATE (for meta_exit only, not TP/SL)
   │  ├─ Check: (current_price - entry_price) / entry_price >= min_profit
   │  ├─ Min profit covers 2x round-trip fees
   │  └─ TP/SL exits bypass this gate
   ├─ EXCURSION GATE
   │  ├─ Check: price moved >= tick_size * 2
   │  └─ Prevent sub-tick noise exits
   └─ If ANY gate fails → RETURN (blocked)
  ↓
5. Position Consolidation
   ├─ If multiple positions held → Consolidate to total_qty
   ├─ If full exit (qty >= position_qty) → Use close_position() method
   ├─ If partial exit → Use execute_quantity_sell() method
   └─ Resolved Quantity: 1.523 BTC (total position)
  ↓
6. Create TradeIntent (SELL)
   ├─ symbol: BTCUSDT
   ├─ side: SELL
   ├─ quantity: 1.523 BTC
   ├─ tag: tp_sl (or meta_exit, rotation, etc.)
   ├─ reason: TP_HIT (or STAGNATION, ROTATION, etc.)
   └─ trace_id: unique decision ID
  ↓
7. Atomic Position Check (Race Prevention)
   ├─ Acquire symbol lock
   ├─ Verify position still exists
   ├─ Reserve symbol
   └─ Release lock
  ↓
8. Order Submission
   ├─ Call ExecutionManager.close_position() or execute_quantity_sell()
   ├─ Market order submission (no partial fills waits)
   ├─ Receive execution confirmation
   └─ Status: FILLED
  ↓
9. Post-Exit Bookkeeping
   ├─ Update position to 0 qty
   ├─ Calculate realized P&L
   ├─ Update portfolio metrics
   ├─ Set lifecycle state to ROTATION_PENDING
   ├─ Freeze dust healing (600s)
   ├─ Record in trade history
   └─ Emit trade event
  ↓
10. Reentry Gating Setup
    ├─ Start reentry lock (900s by default)
    ├─ Record exit reason (TP, SL, ROTATION, etc.)
    ├─ If TP exit → Allow re-entry after price breaks
    ├─ If SL exit → Hold lock longer (stagnation recovery)
    └─ If ROTATION → Immediate re-entry to new symbol allowed
  ↓
Output: Position closed, capital recovered
```

---

## CAPITAL MANAGEMENT SYSTEM

### Capital Allocation Framework

```
Total NAV (e.g., $10,000)
  ├─ Base Capital: $10,000 (invested)
  ├─ Reserve (5%): $500 (for fees + minimum orders)
  ├─ Available: $9,500
  │
  ├─ Per-Position Allocation:
  │  ├─ MICRO_SNIPER (NAV < 1000): 30% limit = $300/position
  │  ├─ STANDARD (NAV 1000-5000): 25% limit = $2,500/position
  │  └─ MULTI_AGENT (NAV >= 5000): 20% limit = $2,000/position
  │
  └─ Per-Cycle Throughput:
     ├─ Max 12 trades/hour (adaptive)
     ├─ Max 100 trades/day
     └─ Friction: 0.2% per trade (taker fees)
```

### Position Sizing (ScalingManager)

```
Decision to BUY:
  ├─ Base Quote: 25 USDT (default)
  ├─ Confidence Scaling:
  │  ├─ conf=0.50 → 25 USDT (minimum)
  │  ├─ conf=0.65 → 35 USDT (1.4x)
  │  ├─ conf=0.80 → 50 USDT (2.0x, capped at max_spend)
  │  └─ conf=1.00 → 50 USDT (capped)
  ├─ Policy Nudge Scaling:
  │  ├─ trade_size_multiplier = 1.0x (normal)
  │  ├─ In AGGRESSIVE mode → 1.5x multiplier
  │  └─ In RECOVERY mode → 0.5x multiplier
  └─ Final Quote: 25 USDT * 1.0x = 25 USDT

From Quote to Quantity:
  ├─ Current Price: 42,500 USDT/BTC
  ├─ Raw Quantity: 25 / 42,500 = 0.000588 BTC
  ├─ Step Size: 0.0001 BTC
  ├─ Rounded Up: 0.0006 BTC (1 step size higher)
  └─ Final Order: 0.0006 BTC @ 42,500 = 25.5 USDT notional
```

### Capital Velocity Optimization

```
Capital Velocity = (Total PnL / Time) 
                 = (Realized PnL * 360) / (Actual Trading Days)

Example:
  ├─ Base Capital: $10,000
  ├─ Realized P&L after 8 hours: $120
  ├─ Daily Rate: $120 * 3 = $360/day
  ├─ Annualized: $360 * 360 = $129,600/year (1,296% APR!)
  ├─ But: subject to market conditions, drawdown, slippage
  └─ Target: 50-100 USDT/hour in normal conditions

Optimization Knobs:
  ├─ Increase trade frequency (shorter hold times)
  ├─ Increase position size (leverage capital)
  ├─ Improve edge (better signal quality)
  └─ Reduce friction (lower fees, better execution)
```

---

## RISK MANAGEMENT LAYERS

### Guard Readiness Checks (Cycle Entry)

```
GUARD 1: MARKET_DATA_READY
  ├─ Have prices for all tracked symbols
  ├─ Recent updates (< 5 seconds old)
  └─ If FAIL → Skip cycle, no trading

GUARD 2: BALANCES_AVAILABLE
  ├─ Successfully fetched account balances
  ├─ USDT balance > MIN_OPERATING_BALANCE
  └─ If FAIL → Skip cycle, may trigger recovery

GUARD 3: OPS_PLANE_READY
  ├─ Execution manager operational
  ├─ Exchange connectivity confirmed
  ├─ Order placement succeeding
  └─ If FAIL → Skip cycle, activate alerting

GUARD 4: TRADING_HOURS_CHECK
  ├─ Within allowed trading hours (24/7 for crypto)
  ├─ Market in normal trading state
  └─ Not in maintenance window

GUARD 5: POSITION_CONSTRAINTS_MET
  ├─ Total positions < max for regime
  ├─ Concentration ratio < threshold
  └─ Dust ratio < critical threshold

GUARD 6: CAPITAL_ADEQUACY_CHECK
  ├─ Free capital >= MIN_CAPITAL_USDT
  ├─ No forced capital recovery in progress
  └─ Portfolio not in starvation state
```

### Economic Gate (Anti-Churn Protection)

```
For Each Trade Decision:
  ├─ Calculate Round-Trip Cost:
  │  ├─ Entry Fee: 0.1% (taker)
  │  ├─ Exit Fee: 0.1% (taker)
  │  ├─ Entry Slippage: 0.15%
  │  ├─ Exit Slippage: 0.15%
  │  └─ Total RT Cost: 0.50%
  │
  ├─ Calculate Min Profitable Move:
  │  ├─ Must cover round-trip costs
  │  ├─ Plus MIN_PROFIT_BPS (5-10 bps profit floor)
  │  └─ Min Profitable: 0.55-0.60%
  │
  ├─ Compare Expected Alpha to Min Profitable:
  │  ├─ If alpha < min_profitable → REJECT (sub-economic)
  │  ├─ If alpha >= min_profitable → ACCEPT
  │  └─ Threshold varies by confidence & regime
  │
  └─ Prevents: Excessive churn, fee drain, low-conviction trades
```

### Position Lifecycle Gates

```
                    ┌─────────────────┐
                    │   NO POSITION   │
                    └────────┬────────┘
                             │ BUY
                             ↓
        ┌────────────────────────────────────────┐
        │   STRATEGY_OWNED (active trading)       │
        │   - Monitors TP/SL                      │
        │   - Allows scaling up/partial exits     │
        │   - Allows ROTATION orders              │
        └────────────┬──────────────────┬─────────┘
                     │ TP/SL HIT       │ Stagnation/Rotation
                     ↓ or SELL signal   ↓
        ┌─────────────────────┐  ┌──────────────────────┐
        │  ROTATION_PENDING   │  │  DUST_HEALING        │
        │  (cooldown: 900s)   │  │  (wait for escape)   │
        │  - Blocks SELL      │  │  - Blocks SELL       │
        │  - Allows BUY retry │  │  - Allows break-even │
        │  (same symbol)      │  │  - Timeout: 5 min    │
        └────────────┬────────┘  └──────────┬───────────┘
                     │ After 900s                │ Escape or timeout
                     ↓                           ↓
                    CLEAR                      CLEAR
```

### Drawdown & Recovery Limits

```
Drawdown Level        │ System Response
──────────────────────┼─────────────────────────────────────
< 5%                  │ NORMAL mode - full trading
5-10%                 │ SAFE mode - reduce position size to 50%
10-20%                │ PROTECTIVE mode - manual SELL priority
20-30%                │ RECOVERY mode - pause new entries, close positions
> 30%                 │ CRITICAL - force liquidation, activate emergency recovery
```

---

## DUST & PORTFOLIO RECOVERY

### Dust Classification System

```
Position Value (USDT)      │ Classification
───────────────────────────┼──────────────────────────
< 1.00                     │ PERMANENT_DUST (invisible)
1.00 - 24.99               │ DUST (recoverable via healing)
25.00 - 100.00             │ MICRO (small position)
> 100.00                   │ SIGNIFICANT (tracked normally)

Recovery Strategy by Dust Type:
├─ PERMANENT_DUST: Ignore, don't report, can't recover
├─ DUST: Healing via escape prices, consolidation, or sacrifice
├─ MICRO: Accumulate via multiple BUY signals
└─ SIGNIFICANT: Normal management
```

### Dust Accumulation & Healing

```
Position Becomes DUST (< $25 USDT)
  ├─ Trigger: Losing trade, scaling down, or conversion error
  ├─ State: DUST_ACCUMULATING
  │  ├─ Wait 5-10 minutes for price recovery
  │  ├─ Set escape alert at cost basis + buffer
  │  └─ Monitor for break-even
  │
  ├─ Healing Attempt 1: Price appreciation to break-even
  │  ├─ If successful → Exit at breakeven (SAVED!)
  │  ├─ If timeout (5 min) → Try Attempt 2
  │  └─ New state: DUST_MATURED
  │
  ├─ Healing Attempt 2: Force consolidation merge
  │  ├─ Group dust with similar positions
  │  ├─ Coalesce into single order
  │  └─ Improve market depth, better exit price
  │
  └─ Healing Attempt 3: Priority sacrifice
     ├─ If dust > 60% of portfolio
     ├─ Force SELL meme coins (DOGE, SHIB, PEPE)
     ├─ Recover capital to base
     └─ Emergency liquidation activated
```

### Portfolio Fragmentation Detection (FIX #3)

```
Portfolio Health Check runs every cycle:
  ├─ HEALTHY:
  │  ├─ < 5 positions, OR
  │  ├─ < 10 positions with good concentration (Herfindahl > 0.3)
  │  └─ Dust ratio < 20%
  │
  ├─ FRAGMENTED:
  │  ├─ 5-15 positions with even distribution
  │  ├─ Dust ratio 20-40%
  │  └─ Requires consolidation
  │
  └─ SEVERE:
     ├─ > 15 positions
     ├─ Many small positions
     ├─ Dust ratio > 40%
     └─ Emergency liquidation recommended

Action on SEVERE Fragmentation:
  ├─ Trigger emergency dust liquidation
  ├─ Force SELL low-concentration positions
  ├─ Reduce portfolio to 3-5 significant positions
  └─ Restore capital velocity
```

---

## KEY COMPONENTS & MODULES

### Core Decision Layer

| Component | Responsibility | File |
|-----------|-----------------|------|
| **MetaController** | Main evaluation loop, signal intake, decision arbitration | `core/meta_controller.py` |
| **PolicyManager** | Policy evaluation, mode x policy matrix, nudge calculation | `core/policy_manager.py` |
| **ArbitrationEngine** | 6-layer gating, signal evaluation, pass/fail logic | `core/arbitration_engine.py` |
| **Lifecycle Manager** | Symbol state machine (NEW→ACTIVE→COOLING→EXITING) | `core/lifecycle_manager.py` |
| **NAV Regime Manager** | MICRO_SNIPER/STANDARD/MULTI_AGENT mode enforcement | `core/nav_regime.py` |

### Execution Layer

| Component | Responsibility | File |
|-----------|-----------------|------|
| **ExecutionManager** | Order placement, monitoring, fill tracking | `core/execution_manager.py` |
| **Execution Logic** | Trade routing, decision → order mapping | `core/execution_logic.py` |
| **Scaling Manager** | Dynamic position sizing, confidence scaling | `core/scaling.py` |
| **Capital Governor** | Position limits, quota enforcement | `core/capital_governor.py` |
| **TP/SL Engine** | Stop-loss & take-profit monitoring, exit generation | `core/tp_sl_engine.py` |
| **Rotation Authority** | Symbol rotation decisions, bracket enforcement | `core/rotation_authority.py` |

### Capital Management

| Component | Responsibility | File |
|-----------|-----------------|------|
| **Balance Manager** | Balance validation, reserve enforcement | `core/balance_manager.py` |
| **Capital Allocator** | Quota allocation, position sizing floors | `core/capital_allocator.py` |
| **Capital Velocity Optimizer** | Allocation planning, throughput maximization | `core/capital_velocity_optimizer.py` |
| **Portfolio Authority** | Concentration limits, diversification enforcement | `core/portfolio_authority.py` |

### Signal Processing

| Component | Responsibility | File |
|-----------|-----------------|------|
| **SignalManager** | Signal cache, deduplication, age filtering | `core/signal_manager.py` |
| **Signal Fusion** | Multi-agent consensus, composite_edge mode | `core/signal_fusion.py` |
| **Signal Batcher** | Batch collection, de-duplication, churn reduction | `core/signal_batcher.py` |
| **Agent Manager** | Trading agent orchestration (TrendHunter, DipSniper, etc.) | `core/agent_manager.py` |

### Infrastructure

| Component | Responsibility | File |
|-----------|-----------------|------|
| **Exchange Client** | Binance API wrapper, order execution | `core/exchange_client.py` |
| **Shared State** | Position tracking, metrics, event bus | `core/shared_state.py` |
| **Market Data Feed** | Real-time price updates, candle history | `core/market_data_feed.py` |
| **Health Monitor** | System health, component status | `core/health_monitor.py` |
| **Watchdog** | Timeout detection, emergency shutdown | `core/watchdog.py` |

---

## CONFIGURATION & MODE SYSTEM

### Key Configuration Parameters

```
TRADING BEHAVIOR
├─ DEFAULT_PLANNED_QUOTE: 25 USDT (base position size)
├─ MIN_ENTRY_QUOTE_USDT: 25 USDT (minimum to trade)
├─ MAX_SPEND_PER_TRADE_USDT: 50 USDT (position cap)
├─ MIN_SIGNIFICANT_POSITION_USDT: 25 USDT (position floor)
├─ MIN_SIGNAL_CONF: 0.50 (ingest level)
├─ TIER_A_CONFIDENCE_THRESHOLD: 0.15 (execution floor, DIAGNOSTIC)
├─ TIER_B_CONF: 0.15 (execution floor, DIAGNOSTIC)
└─ MIN_EXPECTED_EDGE_BPS: 10 bps (economic gate)

CAPITAL MANAGEMENT
├─ BASE_CAPITAL: Loaded from exchange balances
├─ MAX_NOTIONAL_PER_SYMBOL: 100% (single symbol limit)
├─ MAX_LEVERAGE: 1.0 (no margin)
├─ NOTIONAL_SAFETY_BUFFER_USDT: 2.0 USDT (rounding buffer)
├─ PERMANENT_DUST_USDT_THRESHOLD: 1.0 USDT (below = ignored)
└─ MIN_CAPITAL_USDT: 100 USDT (emergency threshold)

SIGNAL FILTERING
├─ MAX_SIGNAL_AGE_SECONDS: 60 sec (stale cutoff)
├─ SIGNAL_BATCH_WINDOW_SEC: 0.1 sec (batch collection)
├─ SIGNAL_BATCH_MAX_SIZE: 10 (signals/batch)
├─ SIGNAL_FUSION_MODE: composite_edge (multi-agent voting)
├─ SIGNAL_FUSION_THRESHOLD: 0.60 (consensus for approval)
└─ META_DIRECTIONAL_CONSISTENCY_PCT: 60% (agent agreement)

POSITION MANAGEMENT
├─ TP_PERCENT: 2.0% (take profit level)
├─ SL_PERCENT: -1.0% (stop loss level)
├─ BUY_COOLDOWN_SEC: 300s (wait between BUY same symbol)
├─ BUY_REENTRY_DELTA_PCT: 0.5% (price change before re-entry)
├─ REENTRY_LOCK_SEC: 900s (cooldown after SELL)
└─ MAX_OPEN_POSITIONS_PER_SYMBOL: 1 (no pyramiding)

RISK LIMITS
├─ MAX_POSITIONS_MICRO_SNIPER: 1 (NAV < 1000)
├─ MAX_POSITIONS_STANDARD: 2 (NAV 1000-5000)
├─ MAX_POSITIONS_MULTI_AGENT: 3+ (NAV >= 5000)
├─ MAX_TRADES_PER_HOUR: 12
├─ MAX_TRADES_PER_DAY: 100
└─ DAILY_DRAWDOWN_HARD_STOP: 30%

MODE SETTINGS
├─ FOCUS_MODE_ENABLED: true (restrict to top performers)
├─ DUST_EXIT_ENABLED: true (auto-exit dust)
├─ TIME_EXIT_ENABLED: true (exit after 24h stagnation)
├─ BOOTSTRAP_SEED_ENABLED: false (first-trade seed)
├─ STRICT_PROFIT_ONLY_SELLS: false (allow loss-saving exits)
└─ ECONOMIC_GUARD_ENABLED: true (fee-aware gating)
```

### Mode Definitions

```
BOOTSTRAP
├─ First position acquisition
├─ Aggressive entry gates relaxed
├─ Allow 1 position per symbol
├─ No rotation active
└─ Duration: Until first SELL or 5 min timeout

NORMAL
├─ Steady-state trading
├─ All guards active
├─ Regime-based position limits
├─ Rotation enabled (with cooldown)
└─ Default mode after bootstrap

SAFE
├─ Triggered by: drawdown > 5%
├─ Action: Reduce position size to 50%
├─ Prevent new entries until recovery
└─ Auto-exit when drawdown < 3%

PROTECTIVE
├─ Triggered by: volatility spike or signal quality drop
├─ Action: Require higher confidence (0.70+)
├─ Increase hold time for positions
└─ Prioritize safety over throughput

AGGRESSIVE
├─ Triggered by: high capital, strong edge, low volatility
├─ Action: Increase position size to 150%
├─ Allow slightly lower confidence (0.55+)
├─ Increase trading frequency
└─ Risk managed by volatility cap

RECOVERY
├─ Triggered by: drawdown > 20%
├─ Action: Halt new entries completely
├─ Force sell weak positions
├─ Focus on capital preservation
└─ Manual intervention required

PAUSED
├─ Manual pause or critical guard failure
├─ No trading allowed
├─ Monitoring only
└─ Requires manual unpause
```

---

## MONITORING & OBSERVABILITY

### Key Metrics Tracked

```
PERFORMANCE METRICS
├─ NAV (Net Asset Value): Total portfolio value
├─ Realized P&L: Closed trade profits/losses
├─ Unrealized P&L: Open position value change
├─ Total Return %: (NAV - Base) / Base
├─ Annualized Return: (Daily Return) * 365
├─ Sharpe Ratio: Return / Volatility
├─ Sortino Ratio: Return / Downside Vol
├─ Max Drawdown: Largest peak-to-trough decline
├─ Win Rate: Winning trades / Total trades
├─ Profit Factor: Sum(Wins) / Sum(Losses)
└─ Average Win/Loss: Win trades vs Loss trades

TRADING ACTIVITY
├─ Total Trades Executed: Cumulative count
├─ Trades This Hour: For rate limiting
├─ Trades This Day: For daily caps
├─ Average Hold Time: Seconds in position
├─ Capital Velocity: $/hour (realized)
├─ Throughput: Orders/hour (execution rate)
├─ Execution Success Rate: Fills / Attempts
└─ Average Trade Size: USDT per trade

POSITION METRICS
├─ Open Positions: Count
├─ Total Position Value: Sum of all open USDT
├─ Largest Position: % of portfolio
├─ Concentration Ratio: Herfindahl index (0-1)
├─ Dust Positions: Count < $25 USDT
├─ Dust Ratio: Dust value / Total
├─ Zero Positions: Count with qty=0
└─ Average Position Age: Hours held

RISK METRICS
├─ Current Drawdown %: NAV / Peak - 1
├─ Volatility (Realized): Std dev of returns
├─ Beta: Correlation with BTC
├─ Value at Risk (VaR): 95% confidence interval
├─ Exposure: Total notional / NAV
├─ Leverage Used: (Total Long + Short) / NAV
└─ Margin Ratio: Margin Available / Required
```

### Logging & Debugging

```
Log Levels:
├─ DEBUG: Detailed function-level tracing
├─ INFO: Significant events (trades, mode changes)
├─ WARNING: Unexpected conditions, gates triggered
├─ ERROR: Trade failures, connection errors
└─ CRITICAL: System failures, emergency stops

Key Log Tags:
├─ [Meta:xxx] → MetaController decisions
├─ [Exec:xxx] → Execution operations
├─ [Guard:xxx] → Readiness gate checks
├─ [Cycle:xxx] → Main loop tracking
├─ [Policy:xxx] → Policy decisions
├─ [Risk:xxx] → Risk management events
├─ [Recovery:xxx] → Recovery operations
├─ [Dust:xxx] → Dust tracking & healing
├─ [Bootstrap:xxx] → Bootstrap mode events
├─ [FOCUS:xxx] → Focus mode operations
└─ [Regime:xxx] → Regime transitions

Watchdog Signals:
├─ cycle_too_long: > 2 seconds (slow main loop)
├─ no_market_data: > 5 seconds without price update
├─ order_not_filled: > 30 seconds without execution
├─ connection_lost: Exchange API unreachable
├─ balance_depleted: < MIN_CAPITAL_USDT
└─ critical_error: Unrecoverable system state
```

---

## QUICK REFERENCE COMMANDS

### Starting the System

```bash
# Full system startup (Master Orchestrator)
export APPROVE_LIVE_TRADING=YES
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# Paper trading (no real money)
export PAPER_TRADING=true
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# Run for 24 hours then auto-shutdown
export TRADING_DURATION_HOURS=24
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py

# With debug logging
export LOG_LEVEL=DEBUG
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py
```

### Configuration Management

```bash
# View current configuration
cat .env | grep -E "^[A-Z_]+"

# Modify configuration (example: increase position size)
sed -i 's/DEFAULT_PLANNED_QUOTE=25/DEFAULT_PLANNED_QUOTE=50/' .env

# Load configuration from template
cp .env.example .env

# Validate configuration
python3 -c "from core.config import Config; c = Config(); print('✓ Config valid')"
```

### Monitoring & Debugging

```bash
# Watch live logs
tail -f /tmp/octivault_master_orchestrator.log

# Check system health
curl http://localhost:8000/health

# Fetch current metrics
curl http://localhost:8000/metrics

# Query position status
curl http://localhost:8000/positions

# Check recent trades
curl http://localhost:8000/trades?limit=10

# Emergency stop signal
curl -X POST http://localhost:8000/stop
```

### Maintenance Tasks

```bash
# Reconcile open_trades with actual positions
python3 -c "from core.meta_controller import MetaController; import asyncio; asyncio.run(MetaController._reconcile_open_trade_book())"

# Reset dust state (clear stale dust tracking)
python3 -c "import redis; r = redis.Redis(); r.delete('dust_state:*')"

# Verify no stuck positions
python3 verify_fixes.py

# Generate performance report
python3 scripts/performance_report.py

# Backup trading history
cp /tmp/trades.db /backups/trades_$(date +%Y%m%d_%H%M%S).db
```

---

## 🎯 SYSTEM SUMMARY

### What This System Does

```
1. MONITORS: Real-time market prices for 100+ trading pairs
2. ANALYZES: Multi-agent signal generation (TrendHunter, DipSniper, etc.)
3. DECIDES: 6-layer arbitration with regime-based gating
4. EXECUTES: Atomic order placement with race condition prevention
5. MANAGES: Portfolio health, dust recovery, capital allocation
6. RECOVERS: Emergency dust liquidation, starvation recovery
7. REPORTS: Live metrics, performance tracking, alerting
```

### Key Capabilities

- ✅ **Concurrent Trading**: Multiple symbols simultaneously (1-5+ based on regime)
- ✅ **Dynamic Sizing**: Position size scales with confidence and market conditions
- ✅ **Dust Recovery**: Automatic healing of small positions via consolidation or escape exits
- ✅ **Risk Management**: Drawdown limits, position caps, economic gating
- ✅ **Multi-Agent Consensus**: Signal fusion from 3+ independent agents
- ✅ **Race Prevention**: Atomic symbol locking prevents duplicate entries
- ✅ **Portfolio Rebalancing**: Concentration monitoring with forced rotations
- ✅ **Emergency Recovery**: Phase 2 guard for critical dust situations
- ✅ **Mode Adaptation**: 8 modes (BOOTSTRAP, NORMAL, SAFE, AGGRESSIVE, RECOVERY, etc.)
- ✅ **Focus Mode**: Restrict trading to top-performing symbols

### Performance Targets

- **Daily Return**: 0.5-2.0% (market dependent)
- **Win Rate**: 55-65% (frequency-based trading)
- **Max Drawdown**: <30% (hard stop at 30%)
- **Sharpe Ratio**: 1.0-2.0 (risk-adjusted return)
- **Capital Efficiency**: 70-80% deployed (20-30% reserve)
- **Execution Success**: 95%+ fill rate on placed orders
- **Cycle Latency**: <500ms (signal to execution)

---

**Document Version:** 1.0  
**Last Reviewed:** 2026-02-14  
**Maintained By:** Octi AI Bot Dev Team  
**Status:** Ready for Production
