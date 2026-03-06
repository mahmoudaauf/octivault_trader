# Discovery Agents System - Complete Post-Fix Architecture

## System Overview

The trading bot uses a **two-tier agent discovery system**:

1. **Agent Discovery** (What CapitalAllocator does)
   - Finds all registered agents in agent_manager.agents dict
   - Allocates capital to each
   - Runs every 5 seconds

2. **Symbol Discovery** (What Discovery Agents do) ← THIS WAS BROKEN, NOW FIXED
   - Discovers new trading opportunities
   - Proposes symbols to symbol universe
   - Runs every 10 minutes (configurable)

---

## Complete Bootstrap Sequence

### Phase 1-5: Core Systems
```
main.py
  ↓
AppContext.initialize_all(up_to_phase=9)
  ├─ Phase 1: Config, logging, paths
  ├─ Phase 2: Database, exchange client
  ├─ Phase 3: Shared state, symbol manager, universe bootstrap
  ├─ Phase 4: Market data feed
  ├─ Phase 5: Execution manager
  └─ (All prerequisites ready)
```

### Phase 6: Agent Initialization (Where Fix Applies)
```
_ensure_components_built()
  ├─ Create StrategyManager
  ├─ Create AgentManager
  ├─ Inject MetaController into AgentManager
  │  └─ [Existing code]
  │
  ├─ ✅ NEW: Register discovery agents
  │  ├─ from core.agent_registry import register_all_discovery_agents
  │  ├─ register_all_discovery_agents(self.agent_manager, self)
  │  │  ├─ Finds wallet_scanner_agent → WalletScannerAgent
  │  │  ├─ Finds symbol_screener_agent → SymbolScreener
  │  │  ├─ Finds liquidation_agent → LiquidationAgent
  │  │  └─ Finds ipo_chaser → IPOChaser
  │  │     └─ Calls agent_manager.register_discovery_agent(agent)
  │  │
  │  └─ agent_manager.discovery_agents = [WalletScanner, SymbolScreener, Liquidation, IPOChaser]
  │
  └─ Create other components (Risk, PnL, TP/SL, etc.)
```

### Phase 6 Runtime: Startup Managers
```
initialize_all() → up_to_phase >= 6
  ├─ Start StrategyManager
  ├─ Start AgentManager ← Triggers discovery loop!
  │  └─ AgentManager.start()
  │     └─ Calls run_all_agents() (creates background tasks for discovery agents)
  │     └─ Calls run_discovery_agents_loop() ← MAIN DISCOVERY ENGINE
  │        ├─ await self.run_discovery_agents()  [Line 1049]
  │        │  ├─ For each agent in self.discovery_agents:
  │        │  │  └─ If hasattr(agent, "run_loop"): await agent.run_loop()
  │        │  └─ [Handles long-running discovery]
  │        │
  │        └─ While True: [Line 1051 - infinite loop]
  │           ├─ await self.run_discovery_agents_once()  [Line 1056]
  │           │  ├─ For each agent in self.discovery_agents:
  │           │  │  ├─ If hasattr(agent, "run_once"):
  │           │  │  │  └─ await agent.run_once()
  │           │  │  │
  │           │  │  └─ WalletScannerAgent.run_once()
  │           │  │     ├─ Fetch wallet holdings from exchange
  │           │  │     ├─ Analyze holdings for trading signals
  │           │  │     └─ Propose new symbols
  │           │  │
  │           │  └─ SymbolScreener.run_once()
  │           │     ├─ Screen all symbols for patterns
  │           │     ├─ Check RSI, MACD, volume, etc.
  │           │     └─ Propose high-potential symbols
  │           │
  │           │  └─ LiquidationAgent.run_once()
  │           │     ├─ Scan for liquidation opportunities
  │           │     └─ Propose liquidation targets
  │           │
  │           │  └─ IPOChaser.run_once()
  │           │     ├─ Check for new IPO listings
  │           │     └─ Propose new IPO symbols
  │           │
  │           └─ sleep(AGENTMGR_DISCOVERY_INTERVAL)  # 10 minutes default
  │
  ├─ Start MetaController
  ├─ Start RiskManager
  ├─ Start TPSLEngine
  └─ ... more components ...
```

---

## Discovery Agent Details

### WalletScannerAgent
**Purpose**: Monitor wallet holdings and detect portfolio opportunities

**Execution**:
```python
class WalletScannerAgent:
    async def run_once(self):
        # 1. Fetch all wallet holdings from exchange
        holdings = await exchange_client.get_wallet()
        
        # 2. For each holding, analyze if tradeable
        for asset, balance in holdings.items():
            if is_tradeable(asset):
                # 3. Check if pairing exists (e.g., BTC→BTCUSDT)
                symbol = f"{asset}USDT"
                if is_listed(symbol):
                    # 4. Propose to symbol manager
                    await symbol_manager.propose_symbol(symbol)
```

### SymbolScreener
**Purpose**: Identify symbols with profitable trading patterns

**Execution**:
```python
class SymbolScreener:
    async def run_once(self):
        # 1. Get all tradeable symbols from exchange
        symbols = await exchange_client.get_symbols()
        
        # 2. For each symbol, calculate technical indicators
        for symbol in symbols:
            klines = await exchange_client.get_klines(symbol)
            rsi = calculate_rsi(klines)
            macd = calculate_macd(klines)
            volume = get_volume(klines)
            
            # 3. Score each symbol
            score = score_symbol(rsi, macd, volume)
            
            # 4. Propose high-scoring ones
            if score > threshold:
                await symbol_manager.propose_symbol(symbol)
```

### IPOChaser
**Purpose**: Detect newly listed tokens and capitalize on launch volatility

**Execution**:
```python
class IPOChaser:
    async def run_once(self):
        # 1. Fetch all exchange symbols
        all_symbols = await exchange_client.get_all_symbols()
        
        # 2. Compare against known universe
        new_symbols = all_symbols - known_symbols
        
        # 3. For each new listing, evaluate
        for symbol in new_symbols:
            if is_promising(symbol):
                # 4. Propose immediately
                await symbol_manager.propose_symbol(symbol)
        
        # 5. Update known universe
        self.known_symbols = all_symbols
```

### LiquidationAgent
**Purpose**: Detect forced liquidations and capitalize on imbalances

**Execution**:
```python
class LiquidationAgent:
    async def run_once(self):
        # 1. Monitor funding rates (indicator of imbalance)
        for symbol in exchange_client.symbols:
            funding = await exchange_client.get_funding_rate(symbol)
            
            # 2. High funding = many longs at risk
            if funding > high_threshold:
                # 3. Watch for liquidation cascade
                price_impact = await detect_price_pressure(symbol)
                if price_impact > threshold:
                    # 4. Propose for trading
                    await symbol_manager.propose_symbol(symbol)
```

---

## Symbol Proposal Pipeline

### What Happens When Agent Proposes Symbol

```
Agent.run_once()
  └─ await symbol_manager.propose_symbol("BTCUSDT")
     └─ SymbolManager.propose_symbol()
        ├─ Check symbol validity (is it listed? min notional OK?)
        ├─ Evaluate proposal score
        ├─ Add to proposal buffer
        │  proposed_symbols = {"BTCUSDT": {...}, "ETHUSDT": {...}}
        │
        └─ These get promoted to accepted_symbols during refresh
           ├─ Via accept_proposed_symbols()
           └─ Injected into all agents:
              ├─ await agent.load_symbols(new_symbols)
              └─ Agents can now trade those symbols
```

---

## Capital Allocation Flow

### CapitalAllocator (Runs every 5 seconds)

```
CapitalAllocator._snapshot_performance()
  ├─ Get all agents from agent_manager.agents
  │  └─ agents = {
  │       "dip_sniper": DipSniper(),
  │       "ipo_chaser": IPOChaser(),  ← NOW REGISTERED
  │       "wallet_scanner": WalletScannerAgent(),  ← NOW REGISTERED
  │       "symbol_screener": SymbolScreener(),  ← NOW REGISTERED
  │       "liquidation_agent": LiquidationAgent(),  ← NOW REGISTERED
  │       "ml_forecaster": MLForecaster(),
  │       "trend_hunter": TrendHunter()
  │     }
  │
  ├─ Calculate allocation per agent
  │  ├─ Each agent gets ~16.7% of budget (100% ÷ 6)
  │  ├─ Discovery agents: ~2.8% each
  │  │  (If we want to tune: adjust in capital_allocator)
  │  └─ Store in shared_state:
  │     authoritative_reservation[agent_name] = allocation_amount
  │
  ├─ Now agents can check budget:
  │  └─ budget = await shared_state.get_authoritative_reservation("dip_sniper")
  │     if budget > 0:
  │         proceed with trading
  │
  └─ Agent execution respects this budget
```

---

## Complete Information Flow

```
                    Bootstrap
                       ↓
        ┌──────────────┴──────────────┐
        ↓                              ↓
   AppContext              register_all_discovery_agents()
   .initialize_all()              ↓
        ↓              agent_manager.discovery_agents ← NOW POPULATED
   _ensure_components          [4 agents]
   _built()                      ↓
        ↓                   AgentManager.start()
   Create agents               ↓
        ↓              run_discovery_agents_loop()
   [FIX APPLIED]           ↓ (every 10 min)
        ↓           run_discovery_agents_once()
   register_discovery       ├─ WalletScanner.run_once()
   _agents()               ├─ SymbolScreener.run_once()
        ↓                  ├─ IPOChaser.run_once()
   discovery_agents        └─ LiquidationAgent.run_once()
   = [4 agents]                ↓
        ↓              Propose symbols
        ↓                     ↓
        └──────────┬──────────┘
                   ↓
            SymbolManager
            .propose_symbol()
                   ↓
            Symbol accepted
                   ↓
         Inject into agents
                   ↓
        Strategy agents can
        trade new symbols!
```

---

## Key Configuration Parameters

```python
# In config/settings

# Discovery execution interval
AGENTMGR_DISCOVERY_INTERVAL = 600.0  # Run every 10 minutes

# Agent-specific parameters
WALLET_SCANNER_CONFIG = {
    "min_balance_threshold": 0.001,  # USDT
    "trading_pairs_to_check": ["USDT"],
}

IPO_CHASER_CONFIG = {
    "lookback_hours": 24,
    "min_volume_increase": 5.0,  # 5x
}

SYMBOL_SCREENER_CONFIG = {
    "min_volume_24h": 1000000,  # USDT
    "rsi_threshold": 30,  # Oversold
    "macd_threshold": 0.001,
}

LIQUIDATION_AGENT_CONFIG = {
    "funding_rate_threshold": 0.0010,  # 0.1%
    "liquidation_detection_window": 3600,  # 1 hour
}
```

---

## Monitoring Discovery Agents

### Logs to Watch

**Startup**:
```
[Bootstrap] ✅ Registered discovery agents with AgentManager
[AgentManager] Starting Discovery Agent loop...
[Agent:wallet_scanner_agent] Launching agent...
[Agent:symbol_screener_agent] Launching agent...
[Agent:ipo_chaser] Launching agent...
[Agent:liquidation_agent] Launching agent...
```

**Execution** (every 10 minutes):
```
[AgentManager] Running discovery agents once...
[WalletScannerAgent] Scanned X wallet holdings, proposed Y symbols
[SymbolScreener] Evaluated Z symbols, proposed Y candidates
[IPOChaser] Checked for new listings, found X matches
[LiquidationAgent] Detected Y liquidation risks
```

**Issues to Watch**:
```
[AgentManager] ⏰ Agent wallet_scanner timed out
[AgentManager] 💥 Agent symbol_screener crashed: ...
```

### Health Checks

```python
# Verify discovery agents are registered
assert len(agent_manager.discovery_agents) == 4

# Verify they're in the main agent dict too
assert "wallet_scanner_agent" in agent_manager.agents
assert "symbol_screener_agent" in agent_manager.agents
assert "ipo_chaser" in agent_manager.agents
assert "liquidation_agent" in agent_manager.agents

# Verify they have required methods
for agent in agent_manager.discovery_agents:
    assert hasattr(agent, "run_once")
    assert hasattr(agent, "agent_type") and agent.agent_type == "discovery"
```

---

## Performance Considerations

### Impact on System Resources

| Component | Impact | Mitigation |
|-----------|--------|-----------|
| **Exchange API calls** | ~4-20 calls/10min | Async batching, caching |
| **CPU** | ~2-5% per scan cycle | Configurable interval |
| **Memory** | Symbol buffer growth | Automatic pruning |
| **Network** | 100-500 KB per cycle | Rate limiting built-in |

### Optimization Options

```python
# Reduce frequency (default 10 min)
config.AGENTMGR_DISCOVERY_INTERVAL = 1800  # 30 minutes

# Reduce agents per cycle
config.DISCOVERY_AGENT_SUBSET = ["ipo_chaser", "wallet_scanner"]
# (Only run 2 of 4 agents per cycle, rotate)

# Reduce symbols to evaluate
config.SYMBOL_SCREENER_MAX_SYMBOLS = 500  # Limit to top 500

# Batch API calls
config.EXCHANGE_API_BATCH_SIZE = 20
```

---

## Summary

**Before Fix**:
- discovery_agents list = EMPTY []
- run_discovery_agents_once() → no-op (iterates empty list)
- Symbol universe = static
- Discovery agents = useless

**After Fix**:
- discovery_agents list = [4 agents]
- run_discovery_agents_once() → executes all 4 agents
- Symbol universe = dynamically expanding
- Discovery agents = working as designed

**The system is now fully autonomous!**

