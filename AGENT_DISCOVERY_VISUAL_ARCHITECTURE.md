# Agent Discovery Mechanism - Visual Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OCTIVAULT TRADING BOT                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────┐
│        AppContext Bootstrap           │
│                                       │
│  1. Create AgentManager               │
│  2. Register agents (6 total)         │
│  3. Create CapitalAllocator           │
│  4. Wire agent_manager → allocator    │
└──────┬───────────────────────────────┘
       │
       ├─────────────────────────────────────────────────────────────┐
       │                                                             │
       v                                                             v
┌─────────────────────────────────┐               ┌─────────────────────────────┐
│     AgentManager                │               │   CapitalAllocator          │
│     (core/agent_manager.py)     │               │   (core/capital_allocator)  │
│                                 │               │                             │
│  ┌───────────────────────────┐  │               │  runs_forever() loop        │
│  │ agents: Dict              │  │               │  every ~5-10 seconds        │
│  │ ├─ DipSniper             │  │               │                             │
│  │ ├─ IPOChaser             │  │◄──────────────┤  Calls:                     │
│  │ ├─ LiquidationAgent       │  │  agent_manager │  _snapshot_performance()   │
│  │ ├─ MLForecaster          │  │   reference    │                             │
│  │ ├─ SymbolScreener        │  │               │  Discovers agents by:       │
│  │ └─ WalletScannerAgent    │  │               │  ├─ agent_mgr.get_agents()  │
│  └───────────────────────────┘  │               │  └─ agent_mgr.agents dict   │
│                                 │               │                             │
│  Methods:                       │               │  Returns:                   │
│  ├─ register_agent()            │               │  ├─ perf_map (6 entries)    │
│  ├─ get_agents()                │               │  └─ perf_ok (boolean)       │
│  ├─ get_discovery_agents()      │               │                             │
│  ├─ collect_and_forward_signals │               │  Logs:                      │
│  └─ run_discovery_agents_once() │               │  "[Allocator] Agent         │
│                                 │               │   Discovery: found 6        │
│                                 │               │   agents: [list]"           │
└─────────────────────────────────┘               └─────────────────────────────┘
       │                                                    │
       │                                                    v
       │                                           ┌──────────────────────┐
       │                                           │  Build Allocation    │
       │                                           │  Plan                │
       │                                           │                      │
       │                                           │  For each agent:     │
       │                                           │  tier="growth"       │
       │                                           │  roi=0.0             │
       │                                           │  win_rate=0.5        │
       │                                           │  weight=1.0/6=0.167  │
       │                                           │                      │
       │                                           │  Allocates capital:  │
       │                                           │  100 USDT / 6 = 16.7 │
       │                                           └──────────────────────┘
       │
       v
┌────────────────────────────────────────────────────────┐
│     SharedState (core/shared_state.py)                │
│                                                       │
│  Stores:                                             │
│  ├─ per_agent_budgets: Dict[agent, budget]           │
│  ├─ per_agent_metrics: Dict[agent, stats]            │
│  └─ accepted_symbols: Set of symbols to trade        │
│                                                       │
│  Updated by:                                         │
│  └─ CapitalAllocator.set_authoritative_reservations()│
└────────────────────────────────────────────────────────┘
```

## Agent Registry Diagram

```
AgentManager.agents
├── "DipSniper" → DipSniper() instance
│   ├─ agent_type: (not set, default="strategy")
│   ├─ generate_signals(): ✓ (included in signal loop)
│   └─ allocation: 16.7%
│
├── "IPOChaser" → IPOChaser() instance
│   ├─ is_discovery_agent: True
│   ├─ agent_type: (implied="discovery")
│   ├─ propose_symbol(): ✓ (discovers new IPOs)
│   └─ allocation: 16.7%
│
├── "LiquidationAgent" → LiquidationAgent() instance
│   ├─ agent_type: "discovery"
│   ├─ Purpose: Clean up weak positions
│   └─ allocation: 16.7%
│
├── "MLForecaster" → MLForecaster() instance
│   ├─ agent_type: (not set, default="strategy")
│   ├─ generate_signals(): ✓ (generates predictions)
│   └─ allocation: 16.7%
│
├── "SymbolScreener" → SymbolScreener() instance
│   ├─ agent_type: (implied="discovery")
│   ├─ propose_symbol(): ✓ (screens market)
│   └─ allocation: 16.7%
│
└── "WalletScannerAgent" → WalletScannerAgent() instance
    ├─ is_discovery_agent: True
    ├─ agent_type: (implied="discovery")
    ├─ Purpose: Scan wallet holdings
    └─ allocation: 16.7%
```

## Discovery Cycle Timeline

```
Timeline: Every 5-10 seconds

T0: CapitalAllocator.run_forever() wakes up
    │
    ├─ T1: Call _snapshot_performance()
    │   │
    │   ├─ T2: Access self.agent_manager.agents
    │   │   └─ Returns Dict[str, Agent] with 6 agents
    │   │
    │   ├─ T3: Build perf_map
    │   │   ├─ For each agent in agents dict:
    │   │   │   └─ perf_map[agent_name] = default_perf
    │   │   │       ├─ tier: "growth"
    │   │   │       ├─ roi: 0.0
    │   │   │       ├─ win_rate: 0.5
    │   │   │       └─ ...
    │   │   └─ perf_map now has 6 entries
    │   │
    │   └─ T4: Return (perf_map, True)
    │
    ├─ T5: Log discovery
    │   └─ "[Allocator] Agent Discovery: found 6 agents: [list]"
    │
    ├─ T6: Build allocation plan
    │   ├─ For each agent in perf_map (6 total):
    │   │   └─ weight = 1.0 / 6 = 0.167
    │   │
    │   └─ per_agent_usdt[agent] = 0.167 * available_capital
    │
    ├─ T7: Update SharedState
    │   └─ ss.per_agent_budgets = {agent: budget for each agent}
    │
    └─ T8: Sleep 5-10 seconds, repeat

Result:
├─ DipSniper:        16.7% of capital
├─ IPOChaser:        16.7% of capital
├─ LiquidationAgent:  16.7% of capital
├─ MLForecaster:      16.7% of capital
├─ SymbolScreener:    16.7% of capital
└─ WalletScannerAgent:16.7% of capital
```

## Signal Flow (AgentManager)

```
AgentManager.collect_and_forward_signals()
│
├─ Filter agents by type
│  ├─ strategy_agents = agents where agent_type != "discovery"
│  │   ├─ DipSniper ✓
│  │   └─ MLForecaster ✓
│  │
│  └─ discovery_agents = agents where agent_type == "discovery"
│      ├─ IPOChaser ✓
│      ├─ LiquidationAgent ✓
│      ├─ SymbolScreener ✓
│      └─ WalletScannerAgent ✓
│
├─ For each strategy agent:
│  ├─ Call agent.generate_signals()
│  ├─ Receive signals: List[{symbol, action, confidence, ...}]
│  ├─ Normalize to TradeIntent objects
│  └─ Forward to MetaController
│
└─ Discovery agents run separately (run_discovery_agents_once)
   └─ Propose symbols via symbol_manager
```

## Agent Type Classification

```
Current Implementation:

┌─────────────────────┐
│   Agent Instance    │
├─────────────────────┤
│                     │
│  Attributes:        │
│  ├─ agent_type      │ (INCONSISTENT!)
│  ├─ is_discovery_... │ (LEGACY!)
│  └─ generate_signals │ (TYPE HINT)
│                     │
│  Detection Logic:   │
│  if agent_type == "discovery": → Discovery
│  elif hasattr(agent, "generate_signals"): → Strategy
│  elif is_discovery_agent: → Discovery (fallback)
│  else: → Unknown
│
└─────────────────────┘

Problem: Three different ways to classify agents!
├─ agent_type attribute
├─ is_discovery_agent flag
└─ presence of generate_signals method

Suggested Fix:
┌─────────────────────┐
│   Agent Instance    │
├─────────────────────┤
│                     │
│  STANDARDIZED:      │
│  ├─ agent_type      │ ONLY THIS (Enum)
│  ├─ agent_name      │ REQUIRED
│  ├─ interval_sec    │ How often to run
│  └─ run_loop() or   │ Implementation
│     run_once()      │
│                     │
│  Enum Values:       │
│  ├─ "signal"        │ Generate buy/sell
│  ├─ "discovery"     │ Propose symbols
│  └─ "infrastructure"│ System maintenance
│
└─────────────────────┘
```

## Capital Allocation Model

```
┌────────────────────────────────────────┐
│  Available Capital: 100 USDT            │
└────────┬─────────────────────────────┬─┘
         │                             │
    Current Model              Proposed Model
    (Equal Split)             (Role-Based)
         │                             │
    ┌────v────────────────┐      ┌────v────────────────┐
    │ 6 agents            │      │ 2 signal agents:80%  │
    │ 1.0 / 6 = 0.167     │      │ 3 discovery:15%      │
    │ = 16.7% each        │      │ 1 infrastructure:5%  │
    │                     │      │                      │
    │ Per agent: 16.7 USD │      │ Signal agent:  40 USD│
    │                     │      │ Discovery:    5 USD  │
    │ Problem:            │      │ Infrastructure: 5 USD│
    │ - Capital spread    │      │                      │
    │   too thin          │      │ Benefit:             │
    │ - Discovery agents  │      │ - Focused allocation │
    │   shouldn't trade   │      │ - Better returns     │
    │                     │      │ - Clear role sep     │
    └─────────────────────┘      └──────────────────────┘
```

## Code Reference Map

```
Agent Discovery Related Files:

core/
├── agent_manager.py
│   ├─ Line 125-135: Agent registry init
│   ├─ Line 438-465: Strategy agent filtering
│   ├─ Line 788-798: get_discovery_agents()
│   ├─ Line 806-829: Discovery agent management
│   └─ Line 440-530: Signal collection loop
│
├── capital_allocator.py
│   ├─ Line 280-340: _snapshot_performance()
│   ├─ Line 1098: Discovery log message
│   ├─ Line 450-550: _build_plan()
│   ├─ Line 310-360: Agent lookup logic
│   └─ Line 70-100: Constructor (receives agent_manager)
│
├── app_context.py
│   ├─ Line 3810-3830: CapitalAllocator construction
│   ├─ Line 3819: agent_manager wiring
│   └─ Line 2500+: Bootstrap phases
│
└── shared_state.py
    ├─ Line 3200+: per_agent_metrics storage
    ├─ Line 3300+: per_agent_budgets storage
    └─ Line 4550+: set_authoritative_reservations()

agents/
├── dip_sniper.py (strategy, no explicit type)
├── ml_forecaster.py (strategy, no explicit type)
├── ipo_chaser.py (discovery, is_discovery_agent=True)
├── liquidation_agent.py (infrastructure, agent_type="discovery")
├── symbol_screener.py (discovery, implied)
└── wallet_scanner_agent.py (discovery, is_discovery_agent=True)
```

---

## Key Observations from Live Logs

### What Worked ✅
```
2026-03-04 22:36:00,965 - INFO - 🔭 SymbolScreener initialized
2026-03-04 22:36:00,965 - INFO - SymbolScreener wired SymbolManager; has propose_symbol=True
2026-03-04 22:36:03,921 WARNING [AppContext] [Allocator] Agent Discovery: found 6 agents
2026-03-04 22:36:03,921 INFO [AppContext] Strategy weight set: SymbolScreener → 0.167
```

Agent discovery mechanism **successfully finds and registers** all 6 agents.

### What Could Improve ⚠️
```
- SymbolScreener getting allocation weight (discovery agent shouldn't trade)
- Equal weight to all 6 agents (should weight by type/role)
- No visible performance tier differentiation (all "growth" tier by default)
- Mixed agent type classification (some use is_discovery_agent, some implicit)
```

---

## Next Steps

1. **Audit Agent Type Attributes**
   - Review each agent class for consistent agent_type
   - Standardize on single enum-based approach

2. **Implement Role-Based Allocation**
   - Separate capital allocation by agent purpose
   - Signal generators: 80%, Discovery: 15%, Infrastructure: 5%

3. **Add Type-Aware Filtering in CapitalAllocator**
   - Filter out discovery agents from allocation
   - Only allocate to signal-generating agents

4. **Test Discovery Mechanism**
   - Unit test: verify discovery finds all registered agents
   - Integration test: verify allocation uses discovered agents
   - E2E test: verify agents execute within allocated budget
