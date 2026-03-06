# 🔍 Agent Discovery Mechanism - Architecture Analysis

## Executive Summary
The agent discovery mechanism is a **two-tier system**:
1. **Capital Allocator Discovery** - Finds agents via AgentManager for allocation planning
2. **Agent Manager Registry** - Maintains agent inventory with type classification

Status from logs:
```
Found 6 agents: ['DipSniper', 'IPOChaser', 'LiquidationAgent', 'MLForecaster', 'SymbolScreener', 'WalletScannerAgent']
Each allocated 16.7% weight (6 agents × 0.167 = 1.0)
```

---

## Architecture Flow

### Phase 1: Agent Registration
```
AppContext Bootstrap
  ├─ Initialize AgentManager (core.agent_manager.py)
  ├─ Register all agents via register_agent()
  │  ├─ DipSniper (strategy agent)
  │  ├─ IPOChaser (discovery agent)
  │  ├─ LiquidationAgent (infrastructure agent)
  │  ├─ MLForecaster (strategy agent)
  │  ├─ SymbolScreener (discovery agent)
  │  └─ WalletScannerAgent (discovery agent)
  └─ agents: Dict[str, Agent] populated
```

**Key File:** `core/agent_manager.py` lines 125-135
```python
self.agents: Dict[str, Any] = {}          # Main registry
self.discovery_agents = []                 # Manual injection list
self._started = False
self._tasks: Dict[str, Task] = {}         # Active tasks
```

### Phase 2: Discovery Loop (CapitalAllocator)
```
CapitalAllocator.run_forever()
  │
  ├─ Every ~5-10 seconds
  │
  ├─ Call _snapshot_performance()
  │  │
  │  └─ Find agents via AgentManager
  │     ├─ Check if agent_manager has get_agents()
  │     ├─ Check if agent_manager has agents attribute
  │     └─ Extract all agents into perf_map
  │
  └─ Build allocation plan
     └─ Log: "[Allocator] Agent Discovery: found N agents: [list]"
```

**Key File:** `core/capital_allocator.py` lines 280-340

```python
# Step 1: Try to get agents from agent_manager
source = self.agent_manager
if source:
    if hasattr(source, "get_agents"):
        agents = source.get_agents() or {}
    elif hasattr(source, "agents"):
        agents = source.agents or {}

# Step 2: For each agent, create perf_map entry
if agents:
    for name in agents:
        if name not in perf_map:
            perf_map[name] = {
                "tier": "growth",
                "roi": 0.0,
                "win_rate": 0.5,
                "drawdown": 0.0,
                "tph": 0.0,
            }
            perf_ok = True

# Step 3: Log discovery
self.logger.warning(
    f"[Allocator] Agent Discovery: found {len(perf_map)} agents: {list(perf_map.keys())}"
)
```

---

## Component Interactions

### 1. **AgentManager** (Core Registry)
**Location:** `core/agent_manager.py`

**Responsibilities:**
- Maintain `agents` dict: `{name: Agent}`
- Register/unregister agents dynamically
- Filter agents by type (strategy vs discovery)
- Start/stop agent loops
- Normalize signals from agents

**Key Methods:**
```python
def register_agent(self, name: str, agent: Any)
    # Adds agent to self.agents dict
    # Validates agent_type if present

def get_agents() -> Dict[str, Any]:
    # Returns all registered agents (assumed but not shown)

def get_discovery_agents() -> List[Agent]:
    # Filters self.agents where agent_type == "discovery"

def get_agents_by_type(agent_type: str) -> List[Agent]:
    # Returns agents matching given type
```

---

### 2. **CapitalAllocator** (Discovery Engine)
**Location:** `core/capital_allocator.py`

**Responsibilities:**
- Discover agents from AgentManager every cycle
- Build performance snapshots
- Allocate capital based on agent tiers
- Track per-agent budgets

**Discovery Method: `_snapshot_performance()`**

```python
async def _snapshot_performance(self):
    """
    Discovers all agents and builds performance profile.
    
    Returns:
        (perf_map, perf_ok)
        where perf_map = {agent_name: {tier, roi, win_rate, ...}}
    """
```

**Flow:**
1. Initialize empty `perf_map = {}`
2. Try to read from `shared_state.per_agent_metrics` (if available)
3. Fallback: Query `self.agent_manager.agents`
4. For each agent:
   - If not in perf_map, create bootstrap entry with default stats
   - Set tier="growth", roi=0.0, win_rate=0.5
5. Return perf_map and success flag

---

### 3. **Agent Registration Checkpoint**
**Location:** Agent classes themselves

**Agent Type Classification:**
```python
# Strategy agents (generate signals, require allocation)
class DipSniper:
    # agent_type not explicitly set → defaults to "strategy"
    # has generate_signals() → included in strategy cycle

class MLForecaster:
    # agent_type not explicitly set → defaults to "strategy"
    # has generate_signals() → included in strategy cycle

# Discovery agents (discover symbols, optional allocation)
class IPOChaser:
    is_discovery_agent = True  # or agent_type = "discovery"
    # Proposes symbols via symbol_manager

class WalletScannerAgent:
    is_discovery_agent = True
    # Scans wallet holdings

class SymbolScreener:
    # Implied discovery agent
    # Scans market for candidates

# Infrastructure agents
class LiquidationAgent:
    agent_type = "discovery"  # Infrastructure, background-driven
    # Cleans up weak positions
```

---

## Discovery Mechanism Details

### Current Implementation (From Logs)

**What We See:**
```
2026-03-04 22:36:03,921 WARNING [AppContext] [Allocator] Agent Discovery: found 6 agents: 
['DipSniper', 'IPOChaser', 'LiquidationAgent', 'MLForecaster', 'SymbolScreener', 'WalletScannerAgent']

2026-03-04 22:36:03,921 INFO [AppContext] Strategy weight set: SymbolScreener -> 0.167
2026-03-04 22:36:03,921 INFO [AppContext] Strategy weight set: IPOChaser -> 0.167
2026-03-04 22:36:03,921 INFO [AppContext] Strategy weight set: WalletScannerAgent -> 0.167
...
```

**Algorithm (Implied):**
1. CapitalAllocator discovers 6 agents
2. Equal weight allocation: 1.0 / 6 = 0.167 per agent
3. Each agent gets "Strategy weight → 0.167"
4. Note: All 6 treated as strategy agents (no type filtering visible)

### Problem: Type Confusion
The logs show **discovery agents being allocated weight**. This might indicate:

**Hypothesis 1: No type filtering**
- CapitalAllocator allocates to ALL agents equally
- Strategy agents (DipSniper, MLForecaster) + Discovery agents (SymbolScreener, etc.)
- Result: Capital spread across 6 agents instead of 2 strategy agents

**Hypothesis 2: Agent type classification is missing**
- `agent_type` attribute not consistently set
- Discovery agents detected as "strategy" by default
- Result: Over-allocation to discovery agents

**Hypothesis 3: Intentional mixed allocation**
- Design intends capital allocation to discovery agents
- Allows discovery agents to participate in positioning (e.g., IPOChaser buys new tokens)
- Capital allocation enables discovery agents to execute trades

---

## Allocation Model

### Current State (From Logs)
```
Total agents: 6
Allocation per agent: 0.167 (16.7%)
Total allocated: 1.0 (100%)

Capital breakdown (example with 100 USDT):
├─ DipSniper:        16.7 USDT
├─ IPOChaser:        16.7 USDT
├─ LiquidationAgent:  16.7 USDT
├─ MLForecaster:      16.7 USDT
├─ SymbolScreener:    16.7 USDT
└─ WalletScannerAgent:16.7 USDT
```

### Allocation Algorithm

**File:** `core/capital_allocator.py` lines 450-550

```python
async def _build_plan(self, perf_map: Dict[str, Dict]):
    """
    Build capital allocation plan based on agent performance tiers.
    
    Logic:
    1. Stratify agents by tier (growth, mature, proven)
    2. Compute tier weights
    3. Apply risk constraints
    4. Validate total allocation = 100%
    """
    
    # Group agents by tier
    tier_counts = {}
    for agent_name, perf in perf_map.items():
        tier = perf.get("tier", "growth")
        tier_counts.setdefault(tier, []).append(agent_name)
    
    # Assign weights per tier
    tier_weights = {
        "growth": 0.50,   # 50% to new agents
        "mature": 0.30,   # 30% to proven agents
        "proven": 0.20,   # 20% to best agents
    }
    
    # Per-agent allocation
    plan = {}
    for tier, agents in tier_counts.items():
        weight = tier_weights.get(tier, 0.25)
        per_agent = weight / len(agents)
        for agent in agents:
            plan[agent] = per_agent * available_capital
```

---

## Discovery Agent Management

### Registration Paths

**Path 1: Automatic (AgentManager)**
```python
# In AppContext.setup_agents()
am = AgentManager(...)
am.register_agent("DipSniper", dip_sniper_instance)
am.register_agent("MLForecaster", ml_forecaster_instance)
# ...
self.agent_manager = am
```

**Path 2: Manual Discovery (Phase 3)**
```python
# In AppContext.bootstrap_phase_3()
for agent in discovery_agents:
    am.register_discovery_agent(agent)
    
# Then run discovery loop
await am.run_discovery_agents_once()
```

### Discovery Agent Methods

**File:** `core/agent_manager.py` lines 788-850

```python
def get_discovery_agents(self) -> List[Agent]:
    """
    Filter agents by agent_type == "discovery"
    """
    return [
        agent for agent in self.agents.values()
        if getattr(agent, "agent_type", None) == "discovery"
    ]

def register_discovery_agent(self, agent: Agent):
    """
    Manual registration for Phase 3 discovery agents.
    """
    self.discovery_agents.append(agent)

async def run_discovery_agents(self):
    """
    Launch discovery agent loops as background tasks.
    """
    for agent in self.discovery_agents:
        if hasattr(agent, "run_loop"):
            asyncio.create_task(agent.run_loop(), name=f"Discovery:{agent.__class__.__name__}")

async def run_discovery_agents_once(self):
    """
    Run discovery agents exactly once (one-shot execution).
    """
    for agent in self.discovery_agents:
        before = set(self.shared_state.symbol_proposals.keys())
        await agent.run_once()
        after = set(self.shared_state.symbol_proposals.keys())
        proposed = after - before
        self.logger.info(f"✅ {agent.__class__.__name__} proposed {len(proposed)} symbols")
```

---

## Current System State

### From Logs (2026-03-04 22:36:03)

**Discovery Success:**
✅ All 6 agents found and registered
✅ AgentManager discovery working
✅ CapitalAllocator allocation computed
✅ Each agent got 16.7% weight

**Execution Flow:**
```
1. SymbolScreener proposed symbols (has propose_symbol=True)
2. Symbols accepted: BTCUSDT, ETHUSDT (2 symbols)
3. Signals built for 2 symbols
4. No BUY signals reached confidence threshold
5. Result: FLAT_PORTFOLIO, no trades
```

---

## Configuration Keys

**Agent Discovery Settings:**

| Key | Default | Purpose |
|-----|---------|---------|
| `AGENTMGR_MAX_START_CONCURRENCY` | 6 | Max concurrent agent starts |
| `AGENTMGR_AGENT_TIMEOUT_S` | 10.0 | Agent startup timeout |
| `AGENTMGR_ENABLE_RESTART` | True | Auto-restart crashed agents |
| `AGENTMGR_RESTART_BACKOFF_MIN` | 2.0 | Min backoff on crash |
| `AGENTMGR_RESTART_BACKOFF_MAX` | 60.0 | Max backoff on crash |
| `AGENTMGR_MARKETDATA_READY_TIMEOUT_S` | 180.0 | Wait for market data |
| `AGENTMGR_EMPTY_INTENT_LOG_INTERVAL_S` | 60.0 | Throttle empty intent logs |

**Capital Allocator Settings:**
| Key | Default | Purpose |
|-----|---------|---------|
| `CAPITAL_ALLOCATOR.UNIFIED_WALLET_MODE` | True | Shared wallet across agents |
| `CAPITAL_ALLOCATOR.TARGET_EXPOSURE_PCT` | 0.20 | Max portfolio exposure |
| `CAPITAL_ALLOCATOR.REQUIRE_PERF_SOURCE` | False | Allow bootstrap without history |

---

## Potential Issues & Observations

### 1. **Equal Weight Allocation to Mixed Agent Types**
- **Issue:** Discovery agents (SymbolScreener, IPOChaser) get same weight as strategy agents (DipSniper, MLForecaster)
- **Risk:** Capital spread thin across 6 agents instead of focused on 2 signal generators
- **Impact:** Each agent only gets ~$16 of capital (from ~100 USDT base)
- **Recommendation:** Filter discovery agents from allocation or use separate allocation tiers

### 2. **No Performance History Bootstrap**
- **Status:** Allocator uses default tier="growth", roi=0.0, win_rate=0.5
- **Observation:** All agents treated equally until trading history builds
- **Risk:** Good agents don't get capital preference initially
- **Recommendation:** Seed initial allocation based on agent design/class, not just type

### 3. **Agent Type Classification Inconsistency**
- **DipSniper, MLForecaster:** No explicit `agent_type` (default to strategy)
- **IPOChaser, WalletScannerAgent:** Have `is_discovery_agent = True`
- **LiquidationAgent:** `agent_type = "discovery"`
- **Risk:** Inconsistent attribute names cause filtering bugs
- **Recommendation:** Standardize to `agent_type` attribute across all agents

### 4. **Discovery Agents in Signal Pipeline**
- **Observation:** SymbolScreener has `propose_symbol=True` but still gets allocation
- **Question:** Should SymbolScreener generate buy/sell signals or just propose symbols?
- **Current:** Acting as both discovery AND signal generator
- **Risk:** Adds noise to signal pipeline if not properly configured

---

## Recommended Improvements

### 1. **Separate Agent Roles**
```python
# Clear classification
class AgentRole(Enum):
    SIGNAL_GENERATOR = "signal"      # Generate buy/sell signals
    SYMBOL_DISCOVERY = "discovery"    # Propose new symbols
    INFRASTRUCTURE = "infrastructure" # Liquidation, cleanup, monitoring

# Usage
class DipSniper:
    agent_type = AgentRole.SIGNAL_GENERATOR

class IPOChaser:
    agent_type = AgentRole.SYMBOL_DISCOVERY
```

### 2. **Differential Allocation Model**
```python
async def _build_plan(self, perf_map, agent_types):
    """
    Allocate based on agent role, not mixed type.
    
    Signal generators: Get 80% of capital
    Symbol discovery: Get 15% of capital
    Infrastructure: Get 5% of capital
    """
    allocation = {}
    
    # Group by role
    signal_agents = [a for a in perf_map if agent_types[a] == SIGNAL_GENERATOR]
    discovery_agents = [a for a in perf_map if agent_types[a] == SYMBOL_DISCOVERY]
    
    # Allocate differently
    for agent in signal_agents:
        allocation[agent] = 0.80 / len(signal_agents)
    for agent in discovery_agents:
        allocation[agent] = 0.15 / len(discovery_agents)
    
    return allocation
```

### 3. **Standardized Attribute Names**
```python
# ALL agents should have these attributes:
class BaseAgent:
    agent_type: str = "signal"  # or "discovery" or "infrastructure"
    name: str = "BaseAgent"
    interval_sec: float = 5.0   # Run every 5 seconds
```

---

## Testing & Validation

### How to Verify Discovery is Working

**1. Check Agent Registration:**
```bash
grep "Agent Discovery: found" logs/clean_run.log
# Should show: "found 6 agents: [list]"
```

**2. Check Allocation Plan:**
```bash
grep "Strategy weight set:" logs/clean_run.log
# Should show weight for each agent
```

**3. Check Agent Type Filtering:**
```bash
grep "agent_type" logs/clean_run.log
# Should show consistent classification
```

### How to Test Changes

```python
# Test 1: Verify discovery mechanism
async def test_agent_discovery():
    am = AgentManager(...)
    am.register_agent("Test1", mock_agent)
    agents = am.get_agents()
    assert len(agents) == 1
    assert "Test1" in agents

# Test 2: Verify allocation calculation
async def test_allocation():
    ca = CapitalAllocator(...)
    perf_map, ok = await ca._snapshot_performance()
    assert len(perf_map) == 6  # Our agents
    assert ok == True

# Test 3: Verify type filtering
async def test_type_filtering():
    am = AgentManager(...)
    discovery = am.get_discovery_agents()
    assert all(getattr(a, "agent_type", None) == "discovery" for a in discovery)
```

---

## Summary

**Current State:**
- ✅ Agent discovery mechanism is **working**
- ✅ 6 agents discovered and registered
- ✅ Capital allocation computed and assigned
- ⚠️ Mixed agent types treated equally
- ⚠️ Capital spread thin across discovery + strategy agents

**Next Steps:**
1. Review if discovery agents should receive capital allocation
2. Implement role-based differential allocation
3. Standardize agent type attributes
4. Add agent tier recommendations based on design quality
