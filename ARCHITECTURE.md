# Octivault Trader — P9 Master Architecture & Design (Consolidated, Final v2.1)

**Last updated: February 26, 2026**

This v2.1 supersedes v2.0 and incorporates institutional-grade capital management fixes: Global systemic degradation guards, performance-based capital allocation, budget validation enforcement, and automated position wind-down for capital efficiency during stress scenarios.

---

## 1. Context & Goals

Octivault Trader (P9) is a self-healing wealth engine that continuously discovers opportunities, manages risk-first execution, and optimizes portfolio performance through intelligent signal arbitration and automated exit management.

**Core Principles:**
- **Signal Arbitration First**: Meta Controller as central decision authority with multi-layer blocking gates
- **Self-Healing Execution**: Bootstrap mechanisms, liquidity retry, and dust accumulation
- **Risk-First Architecture**: All trades pass through Risk Manager assessment before execution
- **Observability Everywhere**: Comprehensive health monitoring, structured logging, and performance tracking
- **Continuous Discovery**: Dynamic symbol discovery and wallet scanning
- **Layer Separation (Universe/Signal/Allocation)**: Watchlist breadth is decoupled from capital limits; allocation gates constrain execution capacity, not discovery breadth

---

## Table of Contents

1. [Context & Goals](#1-context--goals)
2. [Canonical Architecture Diagram (P9+ v2.0)](#2-canonical-architecture-diagram-p9-v20)
   - 2.1 [Detailed Component Architecture Diagram](#21-detailed-component-architecture-diagram)
   - 2.2 [Signal Processing Flow Diagram](#22-signal-processing-flow-diagram)
   - 2.3 [Dust Management State Machine](#23-dust-management-state-machine)
   - 2.4 [Bootstrap & Recovery Flow](#24-bootstrap--recovery-flow)
   - 2.5 [How to Read the Architecture Diagrams](#25-how-to-read-the-architecture-diagrams)
3. [Core Component Architecture](#3-core-component-architecture)
   - 3.1 [Signal Generation Layer (Agents)](#31-signal-generation-layer-agents)
   - 3.2 [Central Arbitration Engine (Meta Controller)](#32-central-arbitration-engine-meta-controller)
   - 3.3 [Execution Layer](#33-execution-layer)
   - 3.4 [Data Architecture (SharedState)](#34-data-architecture-sharedstate)
4. [Advanced Features Discovered](#4-advanced-features-discovered)
5. [Runtime Flows (Updated)](#5-runtime-flows-updated)
6. [Component Inventory](#6-component-inventory)
7. [Configuration & Safety](#7-configuration--safety)
8. [Observability & Health](#8-observability--health)
9. [Evolution from v1.4](#9-evolution-from-v14)
10. [Developer Experience](#10-developer-experience)
11. [Maintenance Guidelines](#11-maintenance-guidelines)

---

## 2. Canonical Architecture Diagram (P9+ v2.0)

```
========================================================================================================================
|                                               OCTIVAULT TRADER (P9+)                                                 |
========================================================================================================================
|  Config/Secrets | AppContext/Phases | Watchdog | Heartbeat | Notifications | SummaryLog | ADRs  | CI/CD Gates |
========================================================================================================================

                                           ┌──────────────────────────────────────────┐
                                           │              BINANCE EXCHANGE            │
                                           │   (Symbols, Prices/OHLCV, Orders, Bal.) │
                                           └──────────────────────────────────────────┘
                                               ▲ fills/balances/filters     │ prices/ohlcv/symbols
                                               │                            ▼
+---------------------------+  C/Q  +------------------------------+
|      ExchangeClient       |<----->|       MarketDataFeed         |
|  post_only_limit/market   |       |  publish MarketDataUpdate    |
|  get_price/ohlcv/balances |       |  VolatilityRegimeDetector    |
|  filters cache + CBs      |       |  ready_event()               |
+---------------------------+       +------------------------------+
                                             updates/events
                                                    ▼
+--------------------------------------------------------------------------------------------------------+
|                                          SharedState                                                   |
| accepted_symbols | prices/ohlcv | balances | positions | filters | exposure_target | dust              |
| cooldowns | reservations (quote ledger v2) | spendable USDT | latest_signals bus                    |
| KPIs: realized/unrealized PnL | run_rate | reservation_util_pct | dust_nav_pct | health               |
| Events: AcceptedSymbolsReady | MarketDataReady | RealizedPnlUpdated | HealthStatus | ModelUpdated     |
+--------------------------------------------------------------------------------------------------------+
  ▲         ▲             ▲             ▲              ▲              ▲             ▲          ▲
  │         │             │             │              │              │             │          │
  │   AgentRegistry   AgentManager  MetaController  ExecutionManager  PortfolioMgr  TP/SL     PnL
  │   (discovery &     (runs all)    (arbiter,      (RESERVE→HYG v2  (NAV/Snap)    Engine   Calculator
  │   strategy lists)               affordability,   → Maker→Taker)                             │
  │         │                         liquidity)      ^ risk‑approved                          │
  │         │       TradeIntent(s)      │              |                                       │
  │         │             │             └───────▶ RiskManager ◀────────────────────────────────┘
  │   StrategyManager      │                    (ALLOW/ADJUST/DENY)
  │   (enable/weights)     │
  │         │              │
  │         │              └─────────────────────▶ HYG v2 (final gate)
  │
  │                 +-------------------+        +-------------------------+
  │                 |   Cash Router     |        | Liquidation Orchestrator|
  │                 |  (planner/budgets)|        |   (executor via SELL)   |
  │                 +-------------------+        +-------------------------+
  │                         │                                 │
  │     spendable per agent/symbol                 LiquidityPlan(policy)  (tag=liquidation)
  │                         ▼                                 ▼
  │                    MetaController  ──────────────── single order path ───────────┘
  │
  │    +-------------------+        +----------------------+       +----------------------+
  │    | CompoundingEngine |        | ProfitTargetGuard    |       | CapitalAllocator     |
  │    | (realized PnL)    |        | (≥20 USDT/h KPI)     |       | (per-tier/agent)    |
  │    +-------------------+        +----------------------+       +----------------------+
  │             │ ExposureDirective            │ Risk nudges                  │ allocations
  │             ▼                               ▼                             ▼
  │      SharedState.exposure_target      RiskManager                   StrategyManager
  │
  ├──────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                             PortfolioBalancer (incl. DustSweeper)                                     │
  │  Offline planes: PerformanceMonitor | Evaluator | ModelTrainer | AgentOptimizer | Recovery            │
  │  Ops: Watchdog | Heartbeat | Notifier | SummaryLog | Metrics API                                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 Detailed Component Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       OCTIVAULT TRADER - DETAILED VIEW                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           EXTERNAL SYSTEMS                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐             │
│  │            BINANCE EXCHANGE              │  │         EXTERNAL DATA SOURCES           │             │
│  │  • REST API (orders, balances, filters)  │  │  • News feeds                          │             │
│  │  • WebSocket (prices, OHLCV)             │  │  • Social sentiment                    │             │
│  │  • Account info & positions              │  │  • On-chain data                       │             │
│  └──────────────────────────────────────────┘  └──────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       DATA INGESTION LAYER                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   ExchangeClient    │  │   MarketDataFeed    │  │   SymbolManager     │  │   VolatilityRegime  │   │
│  │  • Order placement   │  │  • Price streams    │  │  • Universe curation│  │   Detector         │   │
│  │  • Balance queries   │  │  • OHLCV warmup     │  │  • Discovery        │  │  • ATR calculations │   │
│  │  • Filter caching    │  │  • Regime detection │  │  • Filtering        │  └─────────────────────┘   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       CORE DATA STORE                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                     SharedState                                                     │ │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
│  │  STATE MANAGEMENT:                                                                       EVENTS:    │ │
│  │  • accepted_symbols (trading universe)                                                  • market_data│ │
│  │  • prices/ohlcv (real-time + historical)                                               • trade_intent│ │
│  │  • balances/positions (portfolio state)                                                • decisions   │ │
│  │  • reservations (quote ledger v2)                                                      • executions  │ │
│  │  • cooldowns (trading restrictions)                                                    • health      │ │
│  │  • exposure_target (compounding directives)                                            • snapshots   │ │
│  │  • dust classifications                                                                • model_update│ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SIGNAL GENERATION LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   AgentRegistry     │  │   StrategyManager   │  │   AgentManager      │  │   SignalManager     │   │
│  │  • Agent discovery   │  │  • Enable/disable    │  │  • Orchestration    │  │  • Aggregation      │   │
│  │  • Configuration     │  │  • Weights/allocs    │  │  • Scheduling       │  │  • Fusion logic     │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
│                                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   TrendHunter       │  │   MLForecaster      │  │   DipSniper         │  │   ArbitrageHunter   │   │
│  │  • Technical analysis│  │  • ML predictions   │  │  • Mean reversion   │  │  • Cross-exchange    │   │
│  │  • Pattern recognition│  │  • Feature eng.    │  │  • Support/resist.  │  │  • Price diffs       │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
│                                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   LiquidationAgent  │  │   WalletScanner     │  │   NewsReactor       │  │   SignalFusion      │   │
│  │  • Cascade detection │  │  • Large holders    │  │  • Sentiment        │  │  • Correlation      │   │
│  │  • Volume spikes     │  │  • Whale tracking   │  │  • News impact      │  │  • Consensus        │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       ARBITRATION & CONTROL LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                 MetaController (_build_decisions)                                  │ │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
│  │  ARBITRATION PIPELINE:                                                                  GATES:       │ │
│  │  1. Signal Ingestion ← agents                                                        • Dust Prevention│ │
│  │  2. Multi-layer Filtering (Dust/Portfolio/Rotation/Fee/Capital)                      • Portfolio Auth │ │
│  │  3. Consensus Building & Confidence Weighting                                        • Rotation Auth  │ │
│  │  4. Risk Assessment (ALLOW/ADJUST/DENY)                                              • Fee Validation │ │
│  │  5. Reservation → HYG v2 → Dual-Queue Execution                                      • Capital Alloc  │ │
│  │  6. Result → State Update → TP/SL Arming → Snapshot                                  • Bootstrap Guard│ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   RiskManager       │  │   CashRouter        │  │   CapitalAllocator  │  │   PolicyManager     │   │
│  │  • Trade assessment  │  │  • Budget planning  │  │  • Per-agent allocs │  │  • Business rules   │   │
│  │  • Caps/freezes      │  │  • Spendable calc   │  │  • Tier management  │  │  • Mode switching   │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       EXECUTION & RISK MANAGEMENT                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                ExecutionManager (Single Order Path)                                │ │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ │
│  │  EXECUTION FLOW:                                                                     SAFETY:         │ │
│  │  1. RESERVE quote (ledger v2)                                                       • HYG v2 gate    │ │
│  │  2. Risk approval (ALLOW/ADJUST/DENY)                                               • Circuit breakers│ │
│  │  3. Dual-Queue: Maker Try (post-only) → Taker Fallback (market IOC)                 • Reservation TTL │ │
│  │  4. Result processing & state updates                                               • Idempotent ops  │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   Liquidation       │  │   TP/SL Engine      │  │   PortfolioAuth     │  │   RotationAuth      │   │
│  │   Orchestrator      │  │  • ATR-based exits   │  │  • Position limits  │  │  • Replacement      │   │
│  │  • Capacity recovery│  │  • Dynamic multipliers│  │  • Concentration   │  │  • Opportunity eval │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PORTFOLIO MANAGEMENT                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   PortfolioManager  │  │   CompoundingEngine │  │   ProfitTargetGuard │  │   PortfolioBalancer │   │
│  │  • NAV/PnL tracking │  │  • Exposure control  │  │  • KPI enforcement  │  │  • Rebalancing      │   │
│  │  • Snapshots         │  │  • Hysteresis       │  │  • ≥20 USDT/h target│  │  • Dust sweeping    │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       OFFLINE & OPERATIONS                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   PerformanceMon    │  │   ModelTrainer      │  │   AgentOptimizer    │  │   RecoveryEngine    │   │
│  │  • Backtesting       │  │  • ML training      │  │  • Parameter tuning │  │  • State restoration│   │
│  │  • Strategy eval     │  │  • Feature eng      │  │  • Genetic algos    │  │  • Order reconciliation│  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
│                                                                                                         │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │   Watchdog          │  │   Heartbeat         │  │   Notifier          │  │   SummaryLog        │   │
│  │  • Health monitoring │  │  • Component status │  │  • Alerts           │  │  • Structured logs  │   │
│  │  • Escalation        │  │  • Liveness checks  │  │  • Notifications    │  │  • Rollups          │   │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       DATA FLOWS & INTERACTIONS                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  SIGNAL FLOW: Agents → SignalManager → MetaController → RiskManager → ExecutionManager → Exchange     │
│                                                                                                         │
│  STATE SYNC: All components read from SharedState, write via events or direct updates                  │
│                                                                                                         │
│  CONTROL FLOW: AppContext → PhaseManager → Component initialization → Readiness events                 │
│                                                                                                         │
│  FEEDBACK LOOP: Execution results → SharedState → Portfolio updates → Compounding → Exposure changes   │
│                                                                                                         │
│  HEALTH MONITORING: Watchdog → Component status checks → Escalation → Notifications                    │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Signal Processing Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  SIGNAL PROCESSING PIPELINE                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐
│   AGENTS    │───▶│ SIGNAL FUSION│───▶│   META ARBITER  │───▶│ RISK CHECK  │───▶│ EXECUTION   │
│  (Signals)  │    │  (Consensus) │    │ (_build_decisions)│    │ (ALLOW/DENY)│    │ (HYG v2)   │
└─────────────┘    └─────────────┘    └─────────────────┘    └─────────────┘    └─────────────┘
       │                   │                       │                       │             │
       ▼                   ▼                       ▼                       ▼             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐
│ CONFIDENCE  │    │ WEIGHTING   │    │ DUST PREVENTION │    │ CAPITAL     │    │ MAKER TRY   │
│ SCORING     │    │ & AGGREGATION│    │ STATE MACHINE   │    │ ALLOCATION  │    │ (POST-ONLY) │
└─────────────┘    └─────────────┘    └─────────────────┘    └─────────────┘    └─────────────┘
                                                                                       │
                                                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TAKER FALLBACK (MARKET IOC)                                  │
│                              WITHIN BPS LIMITS                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ RESULT PROC │───▶│ STATE UPDATE│───▶│ TP/SL ARMING│───▶│ PORTFOLIO   │───▶│ COMPOUNDING │
│ (COMMIT/REL)│    │ (POSITIONS) │    │ (ATR DYNAMIC)│    │ SNAPSHOT     │    │ ENGINE      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## 2.3 Dust Management State Machine

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               DUST MANAGEMENT STATE MACHINE                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ACTIVE        │────▶│ DUST_LOCKED     │────▶│ LIQUIDATING     │────▶│ TERMINAL        │
│   (Normal ops)  │     │ (Below minNot)  │     │ (Forced sell)   │     │ (Untradeable)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
       ▲                       │                       │                       │
       │                       ▼                       ▼                       │
       │              ┌─────────────────┐     ┌─────────────────┐             │
       │              │ ACCUMULATION    │     │ EXECUTABLE      │             │
       │              │ (Rejected buys) │     │ (Can sell)      │             │
       │              └─────────────────┘     └─────────────────┘             │
       │                       │                       │                       │
       │                       ▼                       ▼                       │
       │              ┌─────────────────┐     ┌─────────────────┐             │
       │              │ THRESHOLD CROSS │     │ LIQUIDATION     │             │
       │              │ (Auto BUY emit) │     │ COMPLETE        │             │
       │              └─────────────────┘     └─────────────────┘             │
       │                       │                       │                       │
       │                       ▼                       ▼                       │
       └───────────────────────┼───────────────────────┼───────────────────────┘
                               ▼                       ▼
                      ┌─────────────────┐     ┌─────────────────┐
                      │ PROMOTION       │     │ STATE RESET     │
                      │ (Strong BUY)    │     │ (Normal ops)    │
                      └─────────────────┘     └─────────────────┘
```

---

## 2.4 Bootstrap & Recovery Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           BOOTSTRAP & RECOVERY SEQUENCE                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  NO POSITIONS│───▶│ THROUGHPUT  │───▶│ MICRO BUDGET │───▶│ TIER-B FORCE │───▶│ SINGLE TRADE │
│  DETECTED    │    │ GAP ACTIVE   │    │ GRANT        │    │ ELIGIBILITY   │    │ EXECUTION    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                       │                       │             │
       ▼                   ▼                       ▼                       ▼             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ CAPITAL     │    │ STARVATION  │    │ LIQUIDITY   │    │ ACCUMULATION │    │ NORMAL OPS  │
│ STARVED     │    │ MODE        │    │ RETRY        │    │ RESOLUTION   │    │ RESUME      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                                       │
                                                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM STABILIZATION                                         │
│                              FULL PORTFOLIO MANAGEMENT                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2.5 How to Read the Architecture Diagrams

### Legend & Symbols
- **Boxes**: Components or systems
- **Arrows (→)**: Data flow direction
- **Dotted Arrows (···)**: Event notifications
- **Double Lines (══)**: Critical boundaries or single paths
- **Grouped Boxes**: Related functionality clusters

### Flow Types
- **Signal Flow**: Agent signals through arbitration to execution
- **State Sync**: Components reading/writing to SharedState
- **Control Flow**: System initialization and phase management
- **Feedback Loop**: Results driving portfolio and compounding decisions

### Layer Organization
1. **External Systems**: Binance API and data sources
2. **Data Ingestion**: Raw data collection and processing
3. **Core Data Store**: SharedState as central authority
4. **Signal Generation**: Agent-based opportunity detection
5. **Arbitration & Control**: Meta Controller decision making
6. **Execution & Risk**: Order placement and safety gates
7. **Portfolio Management**: NAV, compounding, and balancing
8. **Offline & Operations**: Training, monitoring, and maintenance

---

## 3. Core Component Architecture

### 3.1 Signal Generation Layer (Agents)

**Agent Registry & Manager:**
- **Purpose**: Manages agent lifecycle, discovery, and coordination
- **Key Agents**:
  - `trend_hunter.py`: Technical analysis signals
  - `ml_forecaster.py`: Machine learning predictions
  - `dip_sniper.py`: Mean-reversion opportunities
  - `arbitrage_hunter_agent.py`: Cross-exchange arbitrage
  - `liquidation_agent.py`: Liquidation cascade detection
  - `wallet_scanner_agent.py`: Large wallet monitoring
  - `signal_fusion_agent.py`: Multi-signal correlation

**Signal Characteristics:**
- Confidence scores (0.0-1.0)
- Action types: BUY/SELL/HOLD
- Agent attribution and reasoning
- TTL-based expiration

### 3.2 Central Arbitration Engine (Meta Controller)

**Core Function**: `_build_decisions()` - The central decision authority

**Arbitration Pipeline:**
1. **Signal Ingestion**: Collect from all agents via `signal_manager.get_all_signals()`
2. **Dust Prevention**: 4-layer dust protection system
3. **Portfolio Authority**: Position limits, concentration controls
4. **Rotation Authority**: Position replacement optimization
5. **Fee Validation**: Profit gates, excursion controls
6. **Capital Allocation**: Budget validation and reallocation
7. **Decision Trace Injection**: Attach canonical `decision_id` and `trace_id` on every Meta decision before execution (including bootstrap and gated paths)

**Key Arbitration Logic:**
- `_apply_sell_arbiter()`: Single SELL per cycle priority ranking
- `_wind_down_positions()`: Automated position reduction for agents with zero capital allocation
- Capacity management when portfolio full
- Mandatory sell logic for over-capacity
- Agent-indicated sell processing
- Bootstrap and throughput guard mechanisms
- Capital efficiency through degradation response

### 3.3 Execution Layer

**Execution Manager:**
- **Single Order Path**: Only component that can place/cancel orders
- **Dual-Queue Execution**: Maker Try → Taker Fallback
- **HYG v2**: Final hygiene gate (rounding, minNotional, fee-safety)
- **Hygiene Contract Enforcement**: ExecutionManager enforces validator return contract `(ok, qty, adjusted_quote, reason)` and raises explicit `HYGIENE_INTERFACE_MISMATCH` on adapter mismatch.
- **Reservation System**: Quote ledger v2 with TTL
- **NAV-Tier Economic Floor**: BUY affordability uses dynamic floor by total equity
- **Canonical Floor Resolver**: `_get_min_entry_quote(symbol, ...)` is the single source for BUY economic-floor resolution and combines:
  - NAV-tier floor
  - exchange `min_notional`
  - exit-feasibility floor (`compute_symbol_exit_floor`)
  - SharedState dynamic floor (`compute_min_entry_quote`)
  - `< $500 NAV`: `max(min_notional, $10)`
  - `$500-$2000 NAV`: `8% of NAV`
  - `> $2000 NAV`: `5% of NAV`

**TP/SL Engine:**
- **Dynamic ATR-based exits**: ATR(14) with sentiment/regime multipliers
- **Multiple exit strategies**: Time-based, profit targets, trailing stops
- **Concurrent rate-limited exits**: With debounce protection
- **Snowball asymmetry**: Different TP/SL for compounding phases

### 3.4 Capital Allocation System

**CapitalAllocator:**
- **Performance-Based Allocation**: Tiered weighting system (A/B/C) based on agent performance metrics
- **Global Degradation Response**: Automatically reduces usable capital pool during systemic performance degradation
- **Budget Enforcement**: Provides authoritative agent budgets that RiskManager validates against
- **Capital Regime Logging**: Comprehensive logging for allocation decisions and degradation events
- **Institutional Safety**: Configurable degradation thresholds with risk reduction factors

**Key Allocation Logic:**
- `allocate_capital()`: Performance-weighted distribution with global degradation scaling
- `get_agent_budget()`: Returns authoritative budget limits for RiskManager validation
- `check_degradation()`: Monitors system-wide performance metrics for capital reduction triggers
- `log_allocation_regime()`: Detailed logging of capital allocation decisions and constraints

**Separation Contract (Structural):**
- **Universe Layer** (`SymbolManager`): Curates a broad watchlist via `DISCOVERY.TOP_N_SYMBOLS` / `MAX_UNIVERSE_SYMBOLS`
- **Signal Layer** (agents + Meta): Evaluates opportunities on accepted universe
- **Allocation Layer** (Meta envelope + execution affordability): Enforces `max_positions`, capital floors, and spendable balance
- **Invariant**: Allocation constraints must not shrink discovery breadth by coupling universe cap to `max_positions`

### 3.5 Data Architecture (SharedState)

**Authoritative Memory:**
- `accepted_symbols`: Curated trading universe
- `prices/ohlcv`: Real-time and historical market data
- `balances/positions`: Portfolio state with dust classification
- `reservations`: Quote ledger v2 with TTL
- `cooldowns`: Per-symbol trading restrictions
- `exposure_target`: Compounding directives
- `valuation bootstrap invariant`: wallet-restored non-zero quantities receive non-zero provisional `value_usdt` until pricing is available; significance/dust demotion is deferred during market-data warmup

**Event Bus:**
- `events.market_data.update`
- `events.shared_state.ready.*`
- `events.trade.intent`
- `events.meta.decision`
- `events.exec.order/result`
- `events.health.status`

---

## 4. Advanced Features Discovered

### 4.1 Dust Management System

**4-Layer Dust Prevention:**
1. **Dust State Machine**: ACTIVE → DUST_LOCKED → LIQUIDATING
2. **Accumulation Logic**: Tracks rejected trades, auto-emits when threshold crossed
3. **Dust Promotion**: Strong BUY signals can "promote" dust positions
4. **Terminal State**: Prevents infinite liquidation retries on permanently untradeable dust

**Institutional Residual Tiers (Implemented):**
- **Tier 1 (Economically Relevant Residual)**: `>= PERMANENT_DUST_USDT_THRESHOLD` and below significant floor; eligible for healing/liquidation policies.
- **Tier 2 (Marginal Dust)**: small but still above permanent threshold; only acted on when capital/governance allows.
- **Tier 3 (Permanent Dust)**: `< PERMANENT_DUST_USDT_THRESHOLD` (default `$1.0`); treated as terminal residual and excluded from:
  - bootstrap flat blocking
  - dust healing/forced dust liquidation
  - rotation/concentration/starvation exits
  - occupied-capital and risk/capacity pressure metrics

**Dust Liquidation Policies:**
- `dust_first`: Prioritize smallest positions
- `low_impact`: Minimize portfolio disruption
- `fast_profit`: Target losing positions

### 4.2 Bootstrap & Throughput Mechanisms

**Bootstrap Override:**
- Forces Tier-B eligibility for first trades
- Bypasses normal affordability checks
- Auto-adjusts minimum notional requirements
- Bootstrap seed arming invariant:
  - Requires `AcceptedSymbolsReady == True`
  - Requires `MarketDataReady == True`
  - Requires symbol data readiness (`is_symbol_data_ready(symbol)` when available)
  - Requires `latest_prices[symbol] > 0` before seed BUY can be armed

**Throughput Guard:**
- Prevents system starvation when no positions exist
- Grants micro-budget to highest-confidence signals
- Single-trade bootstrap mode

### 4.3 Accumulation Resolution

**Rejected Trade Tracking:**
- Accumulates intended quote amounts across rejections
- Monitors minNotional threshold crossing
- Auto-emits BUY when accumulation succeeds

### 4.4 Portfolio Authority Systems

**Rotation Authority:**
- Evaluates position replacement opportunities
- Considers opportunity scores vs. current holdings
- Manages capacity recovery through intelligent exits

**Concentration Control:**
- Symbol-level position limits
- Portfolio-wide diversification rules
- Dynamic rebalancing triggers

---

## 5. Runtime Flows (Updated)

### 5.1 Signal to Trade Pipeline

```
Agent Signals → Signal Manager → Meta Controller (_build_decisions)
    ↓
Multi-layer Filtering (Dust/Portfolio/Rotation/Fee/Capital)
    ↓
Consensus Building & Confidence Weighting
    ↓
Risk Assessment (ALLOW/ADJUST/DENY)
    ↓
Reservation → HYG v2 → Dual-Queue Execution
    ↓
Result → State Update → TP/SL Arming → Snapshot
```

### 5.2 Dust Accumulation Flow

```
Trade Rejected (minNotional)
    ↓
Accumulation Resolution Check
    ↓
Accumulate Quote Amount
    ↓
Monitor Threshold Crossing
    ↓
Auto-Emit Accumulated BUY
```

### 5.3 Bootstrap Flow

```
No Positions + Throughput Gap
    ↓
Grant Micro-Budget Override
    ↓
Force Tier-B Eligibility
    ↓
Execute Single Bootstrap Trade
    ↓
Resume Normal Operation
```

---

## 6. Component Inventory

### Core Components

| Component | Purpose | Key Methods | Dependencies |
|-----------|---------|-------------|--------------|
| **MetaController** | Central signal arbiter & position management | `_build_decisions()`, `_apply_sell_arbiter()`, `_wind_down_positions()` | SharedState, SignalManager, RiskManager, CapitalAllocator |
| **ExecutionManager** | Order execution authority | `execute_buy()`, `dual_queue_buy()` | ExchangeClient, SharedState, HYG |
| **SharedState** | Authoritative data store | `get_positions()`, `reserve_quote()` | Event bus, persistence |
| **TP/SLEngine** | Dynamic exit management | `set_initial_tp_sl()`, `check_exits()` | SharedState, MarketDataFeed |
| **AgentManager** | Agent orchestration | `run_agents()`, `collect_intents()` | AgentRegistry, StrategyManager |
| **RiskManager** | Trade risk assessment & budget validation | `assess()`, `validate_budget()`, `freeze/unfreeze()` | SharedState, Config, CapitalAllocator |
| **CapitalAllocator** | Performance-based capital allocation | `allocate_capital()`, `get_agent_budget()` | SharedState, PerformanceEvaluator |
| **PortfolioManager** | NAV/PnL tracking | `calculate_nav()`, `snapshot()` | SharedState |

### Supporting Components

| Component | Purpose | Key Features |
|-----------|---------|---------------|
| **MarketDataFeed** | Price/OHLCV streaming | Volatility regime detection, warmup |
| **ExchangeClient** | API communication | Circuit breakers, idempotent orders |
| **CashRouter** | Budget allocation | Per-agent/symbol spendable calculation |
| **PerformanceEvaluator** | KPI monitoring & degradation detection | Sharpe ratio, drawdown monitoring, global systemic guards |
| **LiquidationOrchestrator** | Capacity recovery | Policy-based SELL generation |
| **CompoundingEngine** | Exposure management | Hysteresis-based target adjustment |
| **PortfolioBalancer** | Position optimization | Dust sweeping, rebalancing |
| **RecoveryEngine** | State restoration | Snapshot replay, order reconciliation |

---

## 7. Configuration & Safety

### Key Safety Mechanisms

- **HYG v2**: Rounding, minNotional × safety, remainder guard
- **Circuit Breakers**: Per-endpoint rate limiting with half-open recovery
- **Reservation TTL**: Auto-release stale quote commitments
- **Dust State Machine**: Prevents infinite liquidation retries
- **Accumulation Guards**: Threshold-based auto-emission prevents quote loss
- **Global Systemic Degradation Guard**: Performance-based capital reduction with configurable thresholds
- **Budget Validation**: RiskManager enforcement of CapitalAllocator limits
- **Position Wind-down**: Automated capital efficiency for degraded agents

### Performance Targets

- **Decision→Fill p95**: < 900ms @ 30 symbols
- **HYG Pass Rate**: ≥ 95% for valid affordability
- **Liquidity Retry Success**: ≥ 70%
- **Dust NAV %**: < 1.5% after 24h
- **Reservation Utilization**: ≥ 70%

---

## 8. Observability & Health

### Health Monitoring

- **Heartbeat**: 5-10s component status
- **Watchdog**: 30s supervisor with escalation
- **Component Status**: OK/WARN/ERROR with details
- **Structured Logging**: Per-action summary logs

### Key Metrics

- **Run Rate**: USDT/h profit velocity
- **Fill Ratio**: Executed vs. requested quantity %
- **Slippage BPS**: Price impact measurement
- **Decision Latency**: Intent to execution time
- **Dust Ratio**: % NAV in untradeable positions

---

## 9. Evolution from v1.4

### Major Additions

1. **Advanced Meta Controller**: Complex arbitration with dust promotion, accumulation resolution
2. **TP/SL Engine**: Full ATR-based dynamic exits with snowball asymmetry
3. **Dust State Machine**: Sophisticated dust lifecycle management
4. **Bootstrap Mechanisms**: Throughput guards and first-trade overrides
5. **Accumulation Logic**: Rejected trade quote accumulation with auto-emission
6. **Portfolio Authority**: Rotation and concentration control systems
7. **Multi-Agent Fusion**: Signal correlation and consensus building
8. **Institutional Capital Management**: Performance-based allocation with global degradation response
9. **Global Systemic Degradation Guard**: Multi-threshold performance monitoring with capital reduction
10. **Budget Validation System**: RiskManager enforcement of CapitalAllocator limits
11. **Position Wind-down Logic**: Automated capital efficiency for degraded agents

### Architectural Improvements

- **Single Order Path**: Strict ExecutionManager monopoly on trading
- **Exchange Guardrail**: `ExchangeClient` order APIs reject direct callers unless an `ExecutionManager` scope is active (`ENFORCE_EXECUTION_MANAGER_PATH=true` by default)
- **CashRouter Compliance**: Liquidity/balancer sells are routed through `ExecutionManager.execute_trade(..., is_liquidation=True)`; no direct exchange sell fallback
- **Liquidation SELL Finalization**: Liquidation success path returns immediately after post-fill accounting/close events, preventing fallthrough into the generic SELL pipeline and preserving canonical fill mapping.
- **Canonical Fill Emission Hardening**: `TRADE_EXECUTED` emission is now explicit/verified in post-fill handling with SELL fallback emission when the first attempt is not confirmed.
- **Quote/Qty Post-Fill Symmetry**: Both `_place_market_order_qty` and `_place_market_order_quote` invoke the same post-fill finalization hook to avoid path-dependent observability gaps.
- **Immediate SELL Close Finalizer**: Every SELL filled path now runs `_finalize_sell_post_fill(...)` directly after `_ensure_post_fill_handled(...)` (including delayed-fill reconciliation), guaranteeing immediate close-event emission with per-order idempotency.
- **SELL Finalization Invariant Telemetry**: ExecutionManager tracks per-SELL-fill runtime assertions (`fills_seen`, `finalized`, `pending`, duplicate/missing violations) and emits periodic `[EM:SellFinalizeCounter]` logs to prove one filled SELL maps to one close finalization under TP/SL churn.
- **Delayed Fill Reconciliation**: `_place_market_order_core` performs a short post-submit recheck via `ExchangeClient.get_order(...)` (order id then client order id) and runs canonical post-fill handling when a late `FILLED/PARTIALLY_FILLED` arrives.
- **Delayed Fill Retry Window**: `ExecutionManager._reconcile_delayed_fill(...)` now retries post-submit reconciliation (`POST_SUBMIT_RECHECK_ATTEMPTS`) before classifying a close as non-filled, reducing false-negative TP/SL closes during exchange lag.
- **Post-Fill Idempotency Wrapper**: `_ensure_post_fill_handled(...)` caches per-order post-fill results to prevent duplicate realized-PnL/event side effects across overlapping execution paths.
- **SELL Qty Core Finalization**: `_place_market_order_core` SELL qty return path performs a filled-status post-reconcile finalization call before returning, ensuring canonical observability even for direct core callers.
- **Force-Finalize Fill Contract**: `ExecutionManager.close_position(..., force_finalize=True)` now re-runs delayed-fill reconciliation and only force-finalizes local closure after a confirmed filled/partially-filled SELL with `executedQty > 0`.
- **TP/SL Re-entry Cooldown (Hard Lock)**: Meta now enforces a symbol-level lock window after TP/SL exits (`TP_SL_REENTRY_LOCK_SEC` fallback to `REENTRY_LOCK_SEC`) before allowing another BUY on the same symbol.
- **Flat-Bypass Lock Guard**: Meta flat-portfolio cooldown bypass now defers to TP/SL re-entry lock when a recent TP/SL exit exists, preventing immediate same-symbol churn re-entry.
- **Execution-Side Exit Bookkeeping Backup**: ExecutionManager SELL fill path now records canonical exit reason + cooldown for TP/SL/liquidation contexts, so Meta re-entry governance remains informed even if upper-layer bookkeeping is skipped.
- **Hard Expected-Move Entry Gate**: ExecutionManager BUY affordability now rejects entries when `expected_move_pct <= round_trip_cost_pct * safety_multiplier` (`EV_HARD_SAFETY_MULT` / `ENTRY_EXPECTED_MOVE_FEE_MULT`).
- **Minimum TP Distance Tightening**: Meta TP/SL guard enforces `min_tp_pct >= TP_MIN_ROUND_TRIP_COST_MULT * round_trip_cost_pct` (default 2.0x), preventing sub-fee churn setups.
- **TP/SL Anti-Churn Gates**: TP/SL now enforces explicit round-trip-cost profitability/EV gates (`_passes_profit_gate`, `_passes_net_exit_gate`, `_passes_tp_distance_gate`) and refreshes `open_trades` entry metadata on every new fill to prevent stale-age/stale-entry churn loops.
- **Configurable Asymmetric TP Design**: TP distance now includes a dedicated TP-only asymmetry bias layer (`TPSL_ASYMMETRIC_TP_ENABLED`, `TPSL_ASYM_TP_*`) driven by regime/volatility/sentiment/phase, enabling upside expansion without widening SL.
- **TP/SL Execution Routing Contract**: `TPSLEngine` closes positions only through `ExecutionManager.close_position(...)` (`TPSL_ENFORCE_EXECUTION_MANAGER_ONLY=true`), with runtime guard rails to block direct exchange execution paths.
- **Risk-First Flow**: All decisions pass through Risk Manager
- **Event-Driven State**: SharedState as central event bus
- **Self-Healing**: Bootstrap, retry, and recovery mechanisms
- **Comprehensive Observability**: Health status, structured logging, metrics
- **Institutional Safety**: Global degradation guards, budget validation, position wind-down
- **Capital Efficiency**: Performance-based allocation with automatic risk reduction
- **Cross-Component Governance**: CapitalAllocator-RiskManager-MetaController integration

---

## 10. Developer Experience

### Code Organization

```
octivault_trader/
├── core/                    # Core system components
│   ├── meta_controller.py   # Central arbitration engine
│   ├── execution_manager.py # Order execution authority
│   ├── shared_state.py      # Authoritative data store
│   ├── tp_sl_engine.py      # Dynamic exit management
│   └── [20+ core modules]
├── agents/                  # Signal generation agents
│   ├── trend_hunter.py      # Technical analysis
│   ├── ml_forecaster.py     # ML predictions
│   └── [15+ agent modules]
├── meta/                    # Meta-layer coordination
├── offline/                 # Performance & training
├── ops/                     # Operations & monitoring
└── tests/                   # Comprehensive test suite
```

### Key Contracts

- **TradeIntent**: Agent → Meta signal format
- **MetaDecision**: Arbitration result with affordability and canonical trace metadata (`decision_id`, `trace_id`)
- **LiquidityPlan**: Capacity recovery directives
- **ReservationEvent**: Quote ledger transactions
- **ExecOrder/Result**: Order execution contracts
- **TRADE_EXECUTED Event**: Canonical fill telemetry now includes provenance fields (`order_id`, `client_order_id`, `tag`, `source`) so off-canonical SELLs can be traced and reconciled.

---

## 11. Maintenance Guidelines

### When to Update This Document

This architecture document should be updated whenever:

1. **New Components Added**: When new core components, agents, or systems are introduced
2. **Major Logic Changes**: When arbitration logic, execution flows, or safety mechanisms change
3. **Contract Modifications**: When data structures or API contracts between components change
4. **Performance Targets Updated**: When KPIs or performance requirements change
5. **Safety Mechanisms Enhanced**: When new guards, circuit breakers, or risk controls are added

### Update Process

1. **Code Review Trigger**: Architecture changes should trigger document updates
2. **Version Bumping**: Increment version number (e.g., v2.1, v2.2) for significant changes
3. **Date Update**: Update "Last updated" timestamp
4. **Change Log**: Document what changed in the evolution section
5. **Review**: Have changes reviewed by team members familiar with the system

### Validation Checklist

- [ ] All component relationships accurately reflected
- [ ] Data flow diagrams current and correct
- [ ] Performance targets up-to-date
- [ ] Safety mechanisms documented
- [ ] New features integrated into appropriate sections
- [ ] Code organization matches actual structure

---

This v2.1 architecture reflects the institutional-grade, production-ready trading system that has evolved significantly beyond the original v1.4 specification, incorporating advanced risk management, self-healing capabilities, comprehensive observability, and enterprise-level capital management with global degradation guards and automated efficiency mechanisms.
