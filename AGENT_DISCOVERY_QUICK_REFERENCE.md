# Agent Discovery Mechanism - Quick Reference

## At a Glance

**What:** System that automatically discovers and registers all agents for capital allocation and execution

**Where:** 
- **Discovery:** `core/capital_allocator.py` line 1098
- **Registration:** `core/agent_manager.py` line 125+
- **Wiring:** `core/app_context.py` line 3819

**How:** CapitalAllocator queries AgentManager every 5-10 seconds to build allocation plan

**Current Status:** ✅ Working - Found 6 agents, allocated capital equal-weight

---

## The Discovery Process (5-Step)

```
Step 1: Agent Registration (Bootstrap)
  └─ AppContext.setup_agents() → AgentManager.register_agent() × 6

Step 2: Agent Storage
  └─ AgentManager.agents = {name: instance} with 6 entries

Step 3: Periodic Discovery (Every 5-10 sec)
  └─ CapitalAllocator._snapshot_performance() queries agent_manager.agents

Step 4: Performance Mapping
  └─ Build perf_map with default stats for each discovered agent

Step 5: Allocation
  └─ Assign equal weight (0.167) to each of 6 agents
```

---

## Agent List (From Logs)

| Agent | Type | Role | Allocation | Status |
|-------|------|------|------------|--------|
| DipSniper | Strategy | Signal generation | 16.7% | ✅ Active |
| MLForecaster | Strategy | ML predictions | 16.7% | ✅ Active |
| IPOChaser | Discovery | Find new tokens | 16.7% | ✅ Active |
| SymbolScreener | Discovery | Market scanning | 16.7% | ✅ Active |
| WalletScannerAgent | Discovery | Wallet monitoring | 16.7% | ✅ Active |
| LiquidationAgent | Infrastructure | Position cleanup | 16.7% | ✅ Active |

---

## Key Code Locations

### Discovery Entry Point
**File:** `core/capital_allocator.py`, lines 280-340

```python
# The method that discovers agents
async def _snapshot_performance(self):
    perf_map = {}
    
    # Look for agents via agent_manager
    source = self.agent_manager
    if source and hasattr(source, "agents"):
        agents = source.agents  # Dict[name, Agent]
        for name in agents:
            perf_map[name] = {
                "tier": "growth",
                "roi": 0.0,
                "win_rate": 0.5,
                "drawdown": 0.0,
                "tph": 0.0,
            }
    
    # Log discovery
    self.logger.warning(
        f"[Allocator] Agent Discovery: found {len(perf_map)} agents: "
        f"{list(perf_map.keys())}"
    )
    
    return perf_map, True
```

### Agent Registration
**File:** `core/agent_manager.py`, lines 650-700+

```python
def register_agent(self, name: str, agent: Any):
    """Add agent to registry"""
    self.agents[name] = agent
    self.logger.info(f"Registered agent: {name}")

def get_agents(self) -> Dict[str, Any]:
    """Return all registered agents"""
    return self.agents
```

### Wiring (AppContext)
**File:** `core/app_context.py`, lines 3810-3830

```python
# Create CapitalAllocator with agent_manager reference
self.capital_allocator = CapitalAllocator(
    shared_state=self.shared_state,
    config=self.config,
    agent_manager=self.agent_manager,  # ← KEY WIRE
    # ...
)
```

---

## How Discovery Works (Detailed)

### Step 1: CapitalAllocator Initialization
```python
class CapitalAllocator:
    def __init__(self, shared_state, config, agent_manager=None):
        self.agent_manager = agent_manager  # Receives reference
        self.ss = shared_state              # For storage
        self.config = config                # For settings
```

### Step 2: Background Loop
```python
async def run_forever(self):
    while True:
        # Every cycle
        await self._run_cycle()
        await asyncio.sleep(5.0)  # ~5 seconds
```

### Step 3: Cycle Execution
```python
async def _run_cycle(self):
    # ... checks and gates ...
    
    # THE DISCOVERY CALL
    perf_map, perf_ok = self._snapshot_performance()
    
    # Use discovered agents to build plan
    plan = await self._build_plan(perf_map)
    
    # Apply allocations to shared_state
    ss.set_authoritative_reservations(plan)
```

### Step 4: Performance Snapshot
```python
async def _snapshot_performance(self):
    perf_map = {}
    
    # Query agent_manager
    if self.agent_manager and hasattr(self.agent_manager, "agents"):
        agents = self.agent_manager.agents
        
        # Add each agent to perf map
        for agent_name in agents:
            perf_map[agent_name] = default_perf_profile()
    
    # Log the discovery
    self.logger.warning(
        f"[Allocator] Agent Discovery: found {len(perf_map)} agents: "
        f"{list(perf_map.keys())}"
    )
    
    return perf_map, True
```

### Step 5: Build Allocation Plan
```python
async def _build_plan(self, perf_map):
    plan = {}
    num_agents = len(perf_map)
    weight_per_agent = 1.0 / num_agents  # Equal split
    
    for agent_name in perf_map:
        plan[agent_name] = weight_per_agent * available_capital
    
    return plan
```

---

## Common Questions

### Q: What triggers agent discovery?
**A:** CapitalAllocator._snapshot_performance() runs every cycle (~5 sec). It queries agent_manager.agents dict to see what agents are registered.

### Q: How many times does discovery run?
**A:** Continuously! Every cycle of the CapitalAllocator loop (~5-10 seconds between cycles). But it's idempotent - same agents = same allocations.

### Q: Can agents be added/removed at runtime?
**A:** Yes! If you call `agent_manager.register_agent()` or modify `agent_manager.agents` dict, the next cycle's discovery will see the changes.

### Q: Does discovery check agent type/role?
**A:** Currently NO - all discovered agents get equal treatment. Discovery doesn't filter by agent_type. This is a gap (see "Issues" section).

### Q: What if agent_manager is None?
**A:** CapitalAllocator has null checks:
```python
if self.agent_manager and hasattr(self.agent_manager, "agents"):
    # Proceed with discovery
```
If agent_manager is None, discovery returns empty perf_map and continues.

### Q: How is allocation calculated?
**A:** Simple equal-weight model:
```python
weight_per_agent = 1.0 / len(perf_map)
per_agent_capital = weight_per_agent * available_capital
```
With 6 agents and 100 USDT: each gets 16.7 USDT.

### Q: Can allocation be weighted by agent performance?
**A:** Yes! The system has this built in but it's not used initially. Agents have performance tiers (growth/mature/proven), but bootstrap defaults to "growth" for all.

### Q: Are discovery agents supposed to get capital allocation?
**A:** Currently YES, but this might be a design issue. Discovery agents (IPOChaser, SymbolScreener) shouldn't need execution capital - only signal generators (DipSniper, MLForecaster) should.

---

## Troubleshooting

### Issue: "Agent Discovery: found 0 agents"
**Causes:**
1. agent_manager not wired to CapitalAllocator
2. No agents registered in agent_manager.agents dict
3. agent_manager is None

**Fix:**
```python
# Verify in logs during bootstrap:
# "[AgentManager] Registered agent: ..."  ← Should see 6 of these

# Check CapitalAllocator construction:
# capital_allocator = CapitalAllocator(..., agent_manager=self.agent_manager, ...)
#                                                     ↑ MUST BE PASSED
```

### Issue: "Agent Discovery: found 6 agents" but wrong names
**Likely Cause:** Agents registered with different names than expected

**Debug:**
```python
# Add to logs:
print(f"Agent registry: {self.agents.keys()}")

# Also check config:
# Some agents might be disabled via config flags
```

### Issue: Discovery runs but allocations don't update
**Likely Cause:** SharedState update not working

**Debug:**
```python
# Check for errors in _build_plan()
# Check CapitalAllocator.set_authoritative_reservations() call

# Verify in logs:
# "[Allocator] per_agent_usdt = {...}"  ← Should show allocation map
```

### Issue: Agent gets 0% allocation despite being discovered
**Likely Cause:** Bootstrap gate blocking allocation

**Check logs:**
```
[Allocator] BOOTSTRAP MODE: No performance data available
[EXEC_BLOCK] gate=BALANCES_READY reason=STALE_DATA
```

**Fix:** Wait for system to stabilize (data to update), or override REQUIRE_PERF_SOURCE=false

---

## Configuration Knobs

### CapitalAllocator Configuration
**Location:** `config.py` > `CAPITAL_ALLOCATOR` dict

```python
CAPITAL_ALLOCATOR = {
    "UNIFIED_WALLET_MODE": True,      # Shared wallet across agents
    "TARGET_EXPOSURE_PCT": 0.20,      # Max 20% of NAV exposed
    "REQUIRE_PERF_SOURCE": False,     # Allow bootstrap without history
    "INITIAL_TIER_BOOTSTRAP": "growth", # Default tier for new agents
    "MAX_PORTFOLIO_ALLOCATION_PCT": 100, # Can use up to 100% of capital
}
```

### AgentManager Configuration
**Location:** `config.py` > Agent-specific settings

```python
AGENTMGR_MAX_START_CONCURRENCY = 6       # Start up to 6 agents in parallel
AGENTMGR_AGENT_TIMEOUT_S = 10.0          # Agent startup timeout
AGENTMGR_EMPTY_INTENT_LOG_INTERVAL_S = 60.0  # Throttle empty signal logs
```

### Discovery Interval
**Implicit:** CapitalAllocator runs every ~5 seconds (hardcoded in run_forever)

**To change:** Modify `core/capital_allocator.py` line ~240:
```python
async def run_forever(self):
    while True:
        await self._run_cycle()
        await asyncio.sleep(5.0)  # ← CHANGE THIS VALUE
```

---

## Testing Checklist

### ✓ Discovery is Working
- [ ] Logs show: "[Allocator] Agent Discovery: found 6 agents"
- [ ] Agent names list all 6 expected agents
- [ ] Discovery runs periodically (multiple log entries ~5 sec apart)

### ✓ Allocation is Calculated
- [ ] Logs show: "[Allocator] Strategy weight set: [agent] → 0.167"
- [ ] 6 agents × 0.167 = 1.0 (100% allocated)
- [ ] Total capital distributed matches available NAV

### ✓ Agents Execute Within Budget
- [ ] Each agent's trades don't exceed allocated capital
- [ ] MetaController respects per_agent_budgets from SharedState
- [ ] ExecutionManager checks budget before executing orders

### ✓ Discovery Handles Failures Gracefully
- [ ] If agent_manager is None → discovery skips (no crash)
- [ ] If agent dict is empty → perf_map is empty (no crash)
- [ ] If SharedState update fails → log warning (no crash)

---

## Known Issues

| Issue | Severity | Impact | Recommendation |
|-------|----------|--------|-----------------|
| Discovery agents get capital allocation | Medium | Capital spread thin across discovery agents | Filter discovery agents from allocation |
| Equal-weight allocation to all agents | Medium | New agents don't get preference over bad agents | Implement tier-based weighting |
| Inconsistent agent_type attributes | Low | Type filtering breaks, role unclear | Standardize to enum-based classification |
| No performance history used initially | Low | Bootstrap doesn't seed smart allocation | Add initial tier overrides per agent class |

---

## Recent Changes (March 2026)

### Fixed in UURE Scoring
- Fixed nested volatility_regimes structure bug
- Regime extraction now properly navigates timeframe level

### Expected Next (From Code Review)
- Agent type standardization
- Role-based allocation model
- Discovery agent capital filter

---

## Quick Start: Adding a New Agent

```python
# 1. Create agent class
class MyAgent:
    agent_type = "signal"  # or "discovery" or "infrastructure"
    agent_name = "MyAgent"
    
    async def generate_signals(self):
        # Return signals
        return [...]

# 2. Register in AppContext bootstrap
agent = MyAgent(...)
self.agent_manager.register_agent("MyAgent", agent)

# 3. Discovery will find it automatically
# Next cycle of CapitalAllocator._snapshot_performance():
# → perf_map["MyAgent"] = default_perf_profile

# 4. Allocation computed
# → per_agent_usdt["MyAgent"] = capital / num_agents
```

---

## Related Documentation

- `AGENT_DISCOVERY_ANALYSIS.md` - Deep technical analysis
- `AGENT_DISCOVERY_VISUAL_ARCHITECTURE.md` - Diagrams and flow charts
- `core/capital_allocator.py` - Implementation source
- `core/agent_manager.py` - Agent registry source
- `core/shared_state.py` - Allocation storage source
