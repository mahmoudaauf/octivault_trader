# Agent Discovery Mechanism - Analysis Complete ✅

## Summary

I've completed a comprehensive analysis of the agent discovery mechanism in your trading bot. Here's what I found:

---

## Key Findings

### ✅ Discovery Mechanism is Working
The system successfully:
- Discovers 6 agents: DipSniper, IPOChaser, LiquidationAgent, MLForecaster, SymbolScreener, WalletScannerAgent
- Allocates capital to each agent (16.7% each)
- Runs discovery cycle every ~5 seconds
- Logs discovery events to confirm operation

**Evidence from logs:**
```
2026-03-04 22:36:03,921 WARNING [AppContext] [Allocator] Agent Discovery: 
found 6 agents: ['DipSniper', 'IPOChaser', 'LiquidationAgent', 
'MLForecaster', 'SymbolScreener', 'WalletScannerAgent']
```

### ⚠️ Design Issues Identified

**1. Mixed Agent Types in Allocation**
- Discovery agents (SymbolScreener, IPOChaser) receive capital allocation
- These should only propose symbols, not execute trades
- Result: Capital spread thin across 6 agents instead of focused on 2 signal generators

**2. Equal-Weight Allocation Model**
- All agents get exactly 16.7% (1.0 / 6)
- No differentiation by agent role or quality
- Bootstrap default tier="growth" for all agents

**3. Inconsistent Agent Type Classification**
- Some agents use `agent_type` attribute
- Some use `is_discovery_agent` flag
- Some have no explicit type (default to strategy)
- Inconsistency causes filtering bugs and ambiguity

**4. No Role-Based Capital Separation**
- Signal generators and discovery agents get same treatment
- Should use separate allocation tiers:
  - Signal agents: 80% of capital
  - Discovery agents: 15% of capital  
  - Infrastructure: 5% of capital

---

## Architecture

### Discovery Flow (5 Steps)
```
1. AppContext Bootstrap
   └─ Register 6 agents in AgentManager

2. AgentManager Stores
   └─ agents: Dict[name: Agent] (6 entries)

3. CapitalAllocator Loop (Every ~5 sec)
   └─ Calls _snapshot_performance()

4. Discovery Query
   └─ Reads agent_manager.agents dict

5. Allocation
   └─ Equal weight: 0.167 per agent
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **AgentManager** | `core/agent_manager.py` | Maintains agent registry |
| **CapitalAllocator** | `core/capital_allocator.py` | Discovers agents & allocates capital |
| **AppContext** | `core/app_context.py` | Wires components together |
| **SharedState** | `core/shared_state.py` | Stores allocation results |

---

## Current Allocation Model

With ~100 USDT available capital:

```
DipSniper (Strategy):        16.7 USDT ← Should be 40+ USDT
IPOChaser (Discovery):       16.7 USDT ← Should be ~5 USDT (no trades)
LiquidationAgent (Infra):    16.7 USDT ← Should be ~5 USDT
MLForecaster (Strategy):     16.7 USDT ← Should be 40+ USDT
SymbolScreener (Discovery):  16.7 USDT ← Should be ~5 USDT (no trades)
WalletScannerAgent (Disc):   16.7 USDT ← Should be ~5 USDT
                            ─────────
                            100 USDT total

Problem: Signal generators (2 agents) starved for capital,
         Discovery agents (3 agents) over-allocated but can't use it
```

---

## Recommended Improvements

### 1. **Standardize Agent Type Classification**
```python
from enum import Enum

class AgentRole(Enum):
    SIGNAL = "signal"              # Generate buy/sell signals
    DISCOVERY = "discovery"        # Propose symbols to trade
    INFRASTRUCTURE = "infrastructure"  # System maintenance

# ALL agents should have:
class BaseAgent:
    agent_role: AgentRole = AgentRole.SIGNAL
    agent_name: str = "BaseAgent"
    interval_sec: float = 5.0
```

### 2. **Implement Role-Based Allocation**
```python
async def _build_plan(self, perf_map, agent_roles):
    """Allocate capital based on agent role"""
    
    # Separate by role
    signal_agents = [a for a in perf_map if agent_roles[a] == SIGNAL]
    discovery_agents = [a for a in perf_map if agent_roles[a] == DISCOVERY]
    infra_agents = [a for a in perf_map if agent_roles[a] == INFRASTRUCTURE]
    
    plan = {}
    
    # 80% to signal generators
    signal_weight = 0.80 / len(signal_agents) if signal_agents else 0
    for agent in signal_agents:
        plan[agent] = signal_weight * available_capital
    
    # 15% to discovery agents
    discovery_weight = 0.15 / len(discovery_agents) if discovery_agents else 0
    for agent in discovery_agents:
        plan[agent] = discovery_weight * available_capital
    
    # 5% to infrastructure
    infra_weight = 0.05 / len(infra_agents) if infra_agents else 0
    for agent in infra_agents:
        plan[agent] = infra_weight * available_capital
    
    return plan
```

### 3. **Filter Discovery Agents from Execution**
```python
# In MetaController or ExecutionManager:
if agent_role == AgentRole.DISCOVERY:
    # Only allow symbol proposals, not trades
    return await self.symbol_manager.propose(symbol)
else:
    # Allow normal execution
    return await self.execute_trade(intent)
```

### 4. **Add Agent-Specific Tier Overrides**
```python
# In config.py:
AGENT_INITIAL_TIERS = {
    "DipSniper": "mature",           # Proven strategy
    "MLForecaster": "growth",        # Newer model
    "IPOChaser": "discovery",        # Discovery only
    "SymbolScreener": "discovery",   # Discovery only
    # ...
}

# In CapitalAllocator:
tier = AGENT_INITIAL_TIERS.get(agent_name, "growth")
```

---

## Files Created for Documentation

I've created comprehensive documentation in your workspace:

1. **AGENT_DISCOVERY_ANALYSIS.md**
   - Deep technical analysis of the discovery mechanism
   - Architecture flow diagrams
   - Current state assessment
   - Potential issues and recommendations

2. **AGENT_DISCOVERY_VISUAL_ARCHITECTURE.md**
   - System diagrams with ASCII art
   - Timeline visualization
   - Agent registry structure
   - Capital allocation models

3. **AGENT_DISCOVERY_QUICK_REFERENCE.md**
   - Quick lookup guide
   - Common questions and answers
   - Troubleshooting checklist
   - Configuration reference
   - Testing checklist

All files are in: `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/`

---

## Related Fixes

Earlier in this session, I also fixed:
- **UURE Scoring Error** - Fixed nested volatility_regimes structure bug
  - File: `core/shared_state.py`, `get_unified_score()` method
  - Issue: Type mismatch when accessing regime data
  - Fix: Added proper nested dict navigation with defensive type checking

---

## Next Steps Recommended

### Immediate (1-2 hours)
1. Review the analysis documents
2. Verify current behavior against findings
3. Check if discovery agents should really get capital

### Short Term (1-2 days)
1. Standardize `agent_type` attributes across all agents
2. Add role-based allocation logic
3. Test allocation changes with mock agents

### Medium Term (1 week)
1. Implement agent-specific tier overrides
2. Add discovery agent execution filter
3. Monitor allocation distribution in production

---

## Questions to Consider

1. **Should discovery agents get capital allocation?**
   - Current: YES (16.7% each)
   - Suggested: NO (only symbol proposals)
   - Depends on: Whether IPOChaser/SymbolScreener should execute trades

2. **Should all agents start with equal weight?**
   - Current: YES (0.167 each)
   - Suggested: Role-based (0.40 signal agents, 0.05 discovery)
   - Depends on: System design goals

3. **How to handle agent additions at runtime?**
   - Current: Auto-discovered in next cycle
   - Consideration: Need tier assignment for new agents
   - Solution: Config-based tier mapping or human assignment

4. **Should allocation be dynamic based on performance?**
   - Current: Equal weight until history builds
   - Consideration: Might create positive feedback (good agents starved initially)
   - Solution: Blend initial allocation with learned performance

---

## Conclusion

Your agent discovery mechanism is **functionally operational** and successfully finds and registers agents. However, there are **design opportunities** to improve capital allocation efficiency by:

1. Separating capital allocation by agent role
2. Standardizing agent type classification
3. Filtering discovery agents from execution capital
4. Implementing performance-based tier differentiation

The discovery system itself is solid; it's the allocation model that needs refinement for optimal capital utilization.

---

## Documentation Map

```
Agent Discovery Analysis Complete
├── AGENT_DISCOVERY_ANALYSIS.md (This file)
│   └─ Technical deep-dive
├── AGENT_DISCOVERY_VISUAL_ARCHITECTURE.md
│   └─ Diagrams and flows
├── AGENT_DISCOVERY_QUICK_REFERENCE.md
│   └─ Quick lookup guide
└── UURE_SCORING_ERROR_FIX.md (Related)
    └─ Earlier fix for type error
```

All analysis complete. Ready for implementation! ✅
