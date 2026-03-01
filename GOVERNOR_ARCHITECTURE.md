╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🎛️ CAPITAL SYMBOL GOVERNOR — ARCHITECTURE DIAGRAM            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════════════

SYSTEM ARCHITECTURE

                          ┌──────────────────────────┐
                          │      AppContext          │
                          │   (Orchestrator)         │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
        ┌─────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ SharedState         │ │ CapitalSymbol    │ │ SymbolManager    │
        │ (Central State)     │ │ Governor         │ │ (Discovery)      │
        │                     │ │ (Constraints)    │ │                  │
        │ - balances          │ │                  │ │ - discovers      │
        │ - positions         │ │ Rules:           │ │ - validates      │
        │ - metrics           │ │ 1. Capital Floor │ │ - caps symbols   │
        │ - equity            │ │ 2. API Health    │ │ - finalizes      │
        │ - drawdown          │ │ 3. Retrain Stability    │              │
        │                     │ │ 4. Drawdown Guard       │              │
        └─────────────────────┘ └──────────────────┘ └──────────────────┘
                    │                  ▲                  │
                    │                  │                  │
                    │ reads equity     │ reads drawdown   │ calls governor
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌──────────────────────┐            ┌──────────────────────┐
        │  MarketDataFeed      │            │   MLForecaster       │
        │  (Data Ingestion)    │            │   (Signal Gen)       │
        │                      │            │                      │
        │ - Polls OHLCV data   │            │ - Scans symbols      │
        │ - Detects rate limit │            │ - Generates signals  │
        │ - Notifies governor  │            │ - (optional: track   │
        │   on RateLimit       │            │   retrain skips)     │
        └──────────────────────┘            └──────────────────────┘
                    │                                     │
                    │ if RateLimit                        │ optional: skip tracking
                    ▼                                     ▼
        ┌──────────────────────────────────────────────────────┐
        │   Governor._classify_error()                         │
        │   - Detect error code                                │
        │   - If code in {-1003, -1015, -1021}:               │
        │     call mark_api_rate_limited()                     │
        └──────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────────────────┐
        │   Governor (Dynamic Rules Applied)                   │
        │                                                      │
        │   compute_symbol_cap():                             │
        │   1. Fetch equity                                    │
        │   2. capital_floor_cap(equity) → base cap            │
        │   3. IF api_rate_limited: cap -= 1                  │
        │   4. IF retrain_skips > max: cap -= 1               │
        │   5. IF drawdown > threshold: cap = 1               │
        │   6. Return max(1, cap)                             │
        └──────────────────────────────────────────────────────┘
                    │
                    │ Returns: symbol cap (int)
                    ▼
        ┌──────────────────────────────────────────────────────┐
        │   SymbolManager.initialize_symbols()                 │
        │                                                      │
        │   validated = [50 symbols from discovery]            │
        │   cap = await governor.compute_symbol_cap()          │
        │   validated = validated[:cap]  ← Apply cap here      │
        │   SharedState.set_accepted_symbols(validated)        │
        └──────────────────────────────────────────────────────┘
                    │
                    │ Finalized symbols list
                    ▼
        ┌──────────────────────────────────────────────────────┐
        │   SharedState.accepted_symbols = [2 symbols]         │
        │   (e.g., [BTCUSDT, ETHUSDT])                        │
        └──────────────────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┬────────────┬──────────────┐
        │                        │            │              │
        ▼                        ▼            ▼              ▼
   ┌────────────┐  ┌────────────────┐  ┌──────────┐  ┌───────────────┐
   │ MarketData │  │ ExecutionMgr   │  │ RiskMgr  │  │ Agent Suite   │
   │ Feed       │  │ (2 symbols)    │  │          │  │ (2 symbols)   │
   │ (2 polls)  │  │                │  │          │  │               │
   │            │  │ Can execute    │  │ Enforces │  │ ML scan       │
   │ Polls only │  │ trades for 2   │  │ limits   │  │ 2 symbols     │
   │ 2 symbols  │  │ symbols max    │  │ on 2     │  │               │
   └────────────┘  └────────────────┘  └──────────┘  └───────────────┘
                    │
                    │ Executes trades
                    ▼
                ┌─────────────┐
                │  Journal    │
                │ (Trades)    │
                └─────────────┘


═══════════════════════════════════════════════════════════════════════════════════════

FLOW DIAGRAM: Bootstrap Initialization

START (System Boot)
    │
    ├─ AppContext.__init__()
    │   ├─ Create SharedState (single instance)
    │   ├─ Create CapitalSymbolGovernor(shared_state, config)
    │   │   └─ Governor ready, no state yet
    │   └─ Create SymbolManager(app=self)
    │       └─ SymbolManager has governor reference via self._app
    │
    ├─ AppContext._ensure_components_built()
    │   ├─ MarketDataFeed created
    │   ├─ ExecutionManager created
    │   └─ Other components...
    │
    └─ Phase transitions start
        │
        ├─ Phase 3: Wait for balances
        │   └─ SharedState.balances populated
        │
        ├─ Phase 4: Symbol Discovery
        │   │
        │   ├─ SymbolManager.initialize_symbols()
        │   │   │
        │   │   ├─ await SymbolDiscoverer.run()
        │   │   │   └─ Agents find 50+ symbols
        │   │   │
        │   │   ├─ filter_pipeline(discovered)
        │   │   │   └─ Reduce to 50 candidates
        │   │   │
        │   │   ├─ Validate all 50 (async, bounded)
        │   │   │   └─ All pass: BTCUSDT, ETHUSDT, ..., LTOUSDT
        │   │   │
        │   │   ├─ 🎛️ GOVERNOR INTEGRATION POINT
        │   │   │   │
        │   │   │   ├─ cap = await governor.compute_symbol_cap()
        │   │   │   │   │
        │   │   │   │   ├─ equity = SharedState.balances[USDT] = 172
        │   │   │   │   ├─ Rule 1: 172 < 250 → base_cap = 2
        │   │   │   │   ├─ Rule 2: no rate limit → cap stays 2
        │   │   │   │   ├─ Rule 3: no retrain skips → cap stays 2
        │   │   │   │   ├─ Rule 4: drawdown = 0% < 8% → cap stays 2
        │   │   │   │   └─ Return: 2
        │   │   │   │
        │   │   │   ├─ Apply cap: validated = validated[:2]
        │   │   │   │   └─ validated = {BTCUSDT, ETHUSDT}
        │   │   │   │
        │   │   │   └─ Log: "🎛️ Governor capped symbols: 2 (was 50)"
        │   │   │
        │   │   ├─ PANIC_GUARD check (collapse prevention)
        │   │   │   └─ Not triggered (2 symbols is healthy)
        │   │   │
        │   │   └─ SharedState.set_accepted_symbols({BTCUSDT, ETHUSDT})
        │   │       └─ Emit: AcceptedSymbolsReady event
        │   │
        │   └─ SymbolDiscovery complete ✅
        │
        ├─ Phase 5: Warm up data
        │   │
        │   └─ MarketDataFeed starts polling
        │       ├─ Polls BTCUSDT (not all 50)
        │       ├─ Polls ETHUSDT (not all 50)
        │       └─ Builds 50-bar history for each
        │           └─ Emit: MarketDataReady after 50 bars
        │
        ├─ Phase 6: Regime detection
        │   │
        │   └─ VolatilityRegime analyzes 2 symbols
        │       └─ Emits regime signals (only for 2)
        │
        ├─ Phase 7: ML warmup
        │   │
        │   └─ MLForecaster trains on 2 symbols
        │       ├─ Less data to process
        │       ├─ Faster training
        │       └─ Emit: MLReady
        │
        ├─ Phase 8: Ready gate
        │   │
        │   └─ Check all readiness flags
        │       └─ Emit: P8Ready
        │
        └─ Phase 9: Live trading begins
            │
            ├─ MarketDataFeed continues polling 2 symbols
            ├─ MLForecaster generates signals for 2 symbols
            ├─ ExecutionManager executes trades for 2 symbols
            └─ Journal records all trades


═══════════════════════════════════════════════════════════════════════════════════════

RULE APPLICATION LOGIC

┌─────────────────────────────────────────────────────────────────┐
│  compute_symbol_cap() Decision Tree                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  equity = await _get_equity()                                  │
│  │                                                              │
│  ├─ Rule 1: Capital Floor                                      │
│  │   if equity < 250:      cap = 2                             │
│  │   elif equity < 800:    cap = 3                             │
│  │   elif equity < 2000:   cap = 4                             │
│  │   else:                 cap = max(2, floor(usable / min))   │
│  │                                                              │
│  ├─ Rule 2: API Health Guard                                   │
│  │   if _api_rate_limited:                                     │
│  │       cap = max(1, cap - 1)                                 │
│  │                                                              │
│  ├─ Rule 3: Retrain Stability Guard                            │
│  │   if _retrain_skipped_count > MAX_RETRAIN_SKIPS:           │
│  │       cap = max(1, cap - 1)                                 │
│  │                                                              │
│  ├─ Rule 4: Drawdown Guard                                     │
│  │   drawdown = await _get_drawdown_pct()                      │
│  │   if drawdown and drawdown > MAX_DRAWDOWN_PCT:             │
│  │       cap = 1                                               │
│  │                                                              │
│  └─ Return: max(1, cap)  [never below 1]                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════

EXAMPLE: $172 ACCOUNT CAP CALCULATION

┌─────────────────────────────────────────────────────────────────┐
│  Step-by-Step Cap Calculation                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Fetch Equity                                               │
│     SharedState.balances = {                                   │
│       "USDT": {"free": 172.0, "locked": 0.0},                 │
│       ...                                                       │
│     }                                                           │
│     equity = 172.0 + 0.0 = 172.0 USDT                         │
│                                                                 │
│  2. Rule 1: Capital Floor                                      │
│     172 < 250?  YES                                            │
│     → cap = 2                                                  │
│                                                                 │
│  3. Rule 2: API Health                                         │
│     _api_rate_limited = False                                  │
│     → cap stays 2                                              │
│                                                                 │
│  4. Rule 3: Retrain Stability                                  │
│     _retrain_skipped_count = 0                                 │
│     0 > 2?  NO                                                 │
│     → cap stays 2                                              │
│                                                                 │
│  5. Rule 4: Drawdown Guard                                     │
│     drawdown = 0.0%                                            │
│     0.0 > 8.0?  NO                                             │
│     → cap stays 2                                              │
│                                                                 │
│  6. Final Cap                                                  │
│     return max(1, 2) = 2 ✅                                    │
│                                                                 │
│  Result: Can trade 2 symbols maximum                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════

EXAMPLE: RATE LIMIT SCENARIO

Before Rate Limit:
  - cap = 2 (from capital floor)
  - System trading BTCUSDT, ETHUSDT

Rate Limit Occurs:
  - MarketDataFeed.get_ohlcv() hits error code -1003
  - _classify_error() detects "RateLimit"
  - Calls governor.mark_api_rate_limited()
  - Sets: _api_rate_limited = True

Next Symbol Discovery:
  - SymbolManager.initialize_symbols() called
  - cap = await governor.compute_symbol_cap()
    ├─ equity = 172
    ├─ Rule 1: 172 < 250 → cap = 2
    ├─ Rule 2: _api_rate_limited = True
    │   cap = max(1, 2 - 1) = 1  ← Reduced!
    ├─ Rule 3: no skips → cap stays 1
    ├─ Rule 4: no drawdown → cap stays 1
    └─ return 1
  - validated = validated[:1]  # Only first symbol
  - Result: System now trades only BTCUSDT

Log Output:
  [MarketDataFeed] ⚠️ RateLimit error detected
  [CapitalSymbolGovernor] ⚠️ API Rate Limited → reduce cap to 1
  [SymbolManager] 🎛️ Governor capped symbols: 1 (was 2)


═══════════════════════════════════════════════════════════════════════════════════════

EXAMPLE: DRAWDOWN SCENARIO

Normal Trading:
  - System profitable
  - cap = 2 (BTCUSDT, ETHUSDT)

Account Loses Money:
  - Drawdown = 9.5%
  - SharedState.current_drawdown = 9.5

Next compute_symbol_cap():
  - Rule 4: drawdown = 9.5%
  - 9.5 > 8.0?  YES
  - cap = 1  ← Forced to defensive

Log Output:
  [CapitalSymbolGovernor] 🛡️ Drawdown 9.5% > 8% → DEFENSIVE (cap=1)
  [SymbolManager] 🎛️ Governor capped symbols: 1 (was 2)

Effect:
  - Stops trading one symbol
  - Concentrates on single most reliable symbol
  - Reduces risk during drawdown
  - Allows recovery

Recovery:
  - After drawdown < 8%
  - Next discovery allows cap to increase
  - System can gradually add symbols back


═══════════════════════════════════════════════════════════════════════════════════════
