# Capital Governor vs Capital Allocator - Detailed Comparison

## Quick Summary

| Aspect | Capital Governor | Capital Allocator |
|--------|------------------|-------------------|
| **Purpose** | Controls **position structure & limits** | Distributes **executable capital budgets** |
| **Scope** | **Bracket-based limits** (micro, small, medium, large) | **Dynamic budget distribution** per agent/strategy |
| **Timing** | Applied **per-trade** (on signal arrival) | Applied **periodically** (every 15+ minutes) |
| **Decision Type** | **Yes/No permission** (can you trade this?) | **How much capital** to allocate? |
| **Output** | Position limits & sizing constraints | Budget per agent, exposure percentages |
| **Operates On** | Account equity bracket | Current free capital & performance |

---

## Detailed Breakdown

### 1. Capital Governor: "What's Allowed?"

**Location**: `core/capital_governor.py` (400+ lines)

**Purpose**: Enforce best-practice position limits based on account size.

**Key Responsibility**:
- Classify account into bracket (MICRO, SMALL, MEDIUM, LARGE)
- Define what's **allowed** to trade (position structure, sizing, rotation)
- Answer: "Can I trade this symbol? How much? Can I rotate?"

**Core Methods**:
```python
get_bracket(nav)                    # < $500? → MICRO
get_position_limits(nav)            # Max symbols, positions, rotating slots
get_position_sizing(nav, symbol)    # Quote per position, EV multiplier
should_restrict_rotation(nav)       # Can rotate? (MICRO = NO)
validate_symbol_for_bracket(nav, symbol)  # Is this symbol allowed?
```

**Input**: 
- Current NAV (account equity: $350)

**Output**:
```python
{
    "bracket": "micro",
    "max_active_symbols": 2,
    "core_pairs": 2,
    "max_rotating_slots": 0,        # ← NO rotation in MICRO!
    "max_concurrent_positions": 1,
    "quote_per_position": 12.0,
    "ev_multiplier": 1.4,
    "allow_rotation": False
}
```

**Decision Point**: When a BUY signal arrives
```
Signal arrives → Check Capital Governor:
  ✅ "BTCUSDT allowed? Yes (core pair)"
  ✅ "Can I trade 1 position? Yes (0 open now)"
  ✅ "Size $12? Yes (within limits)"
  → PROCEED with trade
```

**Key Characteristics**:
- ✅ **Static per bracket** (same for all $350 accounts)
- ✅ **Focused on structure** (limits, rotation, position count)
- ✅ **Prevents overtrading** (1 position max in MICRO)
- ✅ **Enforced before trading** (gate keeper)
- ✅ **Enables learning** (focused on 2 symbols in MICRO)

---

### 2. Capital Allocator: "How Much Capital?"

**Location**: `core/capital_allocator.py` (1210 lines)

**Purpose**: Dynamically distribute available capital across agents/strategies based on performance.

**Key Responsibility**:
- Calculate total available capital (free USDT, NAV-based)
- Distribute across agents in tiers (core, growth, experimental)
- Account for risk constraints
- Emit allocation plan for StrategyManager
- Answer: "How much capital does AGENT_A get? AGENT_B?"

**Core Methods**:
```python
plan_allocation()                   # Calculate full allocation
_free_usdt()                        # Get spendable balance
_nav_quote()                        # Get total NAV
_snapshot_performance()             # Assess agent ROI/Sharpe
_exposure_target()                  # Calculate global exposure limit
apply_allocation_to_strategies()    # Set budgets for agents
```

**Input**: 
- Current free USDT ($350)
- Recent agent performance (ROI, win rate, drawdown)
- Risk constraints (max exposure, max per-agent allocation)

**Output**:
```python
AllocationPlan {
    "total_free": 350.0,
    "allocations": {
        "btc_dca": {
            "tier": "core",
            "budget": 175.0,      # 50% to core tier
            "max_per_symbol": 175.0,
            "exposure": 0.50
        },
        "eth_momentum": {
            "tier": "growth",
            "budget": 122.5,      # 35% to growth tier
            "max_per_symbol": 75.0,
            "exposure": 0.35
        },
        "sol_experimental": {
            "tier": "experimental",
            "budget": 52.5,       # 15% to experimental tier
            "max_per_symbol": 30.0,
            "exposure": 0.15
        }
    },
    "timestamp": "2026-03-01T12:00:00Z"
}
```

**Decision Point**: Periodic allocation cycle (every 15 minutes)
```
Every 15 minutes → Capital Allocator runs:
  1. Check free USDT: $350
  2. Get agent performance scores
  3. Calculate: 50% to core, 35% growth, 15% experimental
  4. Emit budget plan
  5. StrategyManager applies budgets
```

**Key Characteristics**:
- ✅ **Dynamic per agent** (changes based on performance)
- ✅ **Focused on distribution** (how to split capital)
- ✅ **Respects performance** (better agents get more capital)
- ✅ **Risk-aware** (checks exposure caps)
- ✅ **Continuous optimization** (rebalances every 15 min)

---

## Side-by-Side Comparison

### Architecture Layer

```
User's BUY Signal
       ↓
   ┌───┴────────────────────────────────────┐
   ↓                                         ↓
Capital Governor                    Capital Allocator
(Permission Gate)                   (Budget Distributor)
   ↓                                         ↓
"Can I trade?"                      "How much per agent?"
   ↓                                         ↓
   ├─ Bracket check (MICRO)         ├─ Perf snapshot
   ├─ Position limit (1 max)        ├─ Calculate pool
   ├─ Size limit ($12)              ├─ Apply tier splits
   ├─ Rotation? NO                  └─ Emit allocation
   └─ PROCEED/BLOCK
       ↓
   Execution Manager
   (Actually places order)
```

### Temporal Scope

```
Timeline:

Capital Governor:
  ├─ Applied: Per signal arrival (milliseconds)
  ├─ Duration: Single trade decision
  ├─ Updates: Changes only when NAV crosses bracket threshold
  └─ Example: "At $350, max 1 position allowed"

Capital Allocator:
  ├─ Applied: Periodic cycle (15 min by default)
  ├─ Duration: Affects multiple trades over 15 minutes
  ├─ Updates: Recalculates every cycle based on latest perf
  └─ Example: "Allocate $175 to core agents this cycle"
```

### Decision Criteria

```
Capital Governor asks:
  1. What bracket am I in? (NAV-based)
  2. How many symbols allowed? (Fixed per bracket)
  3. How many positions open? (Current state)
  4. Can I trade this symbol? (Core vs rotating)
  5. What's the max size? (Bracket limit)

Capital Allocator asks:
  1. How much free USDT? (Market state)
  2. How are agents performing? (Performance history)
  3. What's the global exposure limit? (Risk config)
  4. How to split across tiers? (Strategic allocation)
  5. What budget per agent? (Optimization)
```

### Use Cases

```
Capital Governor handles:
  ✓ Block BUY if position limit reached
    (MICRO: max 1 position, already have 1 open)
  
  ✓ Prevent rotation in learning phase
    (MICRO: rotation disabled)
  
  ✓ Adjust position size to bracket
    (MICRO: $12 per trade)
  
  ✓ Enforce core vs rotating limits
    (MICRO: 2 core, 0 rotating)

Capital Allocator handles:
  ✓ Distribute $350 across strategies
    (Core agents: $175, Growth: $122.5, Experimental: $52.5)
  
  ✓ Rebalance based on performance
    (High-ROI agents get more budget next cycle)
  
  ✓ Enforce per-agent exposure caps
    (No single agent gets > 65% of capital)
  
  ✓ Manage risk across portfolio
    (Total allocation ≤ NAV * max_ratio)
```

---

## Integration Points

### Where Capital Governor Is Used

```python
# MetaController.orchestrate()
limits = self.capital_governor.get_position_limits(nav)
if len(open_positions) >= limits["max_concurrent_positions"]:
    return {"action": "REJECT", "reason": "Position limit"}

# SymbolRotationManager.can_rotate_symbol()
if self.capital_governor.should_restrict_rotation(nav):
    return False  # Rotation disabled in MICRO

# PositionManager.calculate_position_size()
sizing = self.capital_governor.get_position_sizing(nav)
order_size = sizing["quote_per_position"]  # $12
```

### Where Capital Allocator Is Used

```python
# AppContext P8 (periodic cycle)
allocation_plan = await self.capital_allocator.plan_allocation()

# StrategyManager applies budgets
for agent, budget in allocation_plan["allocations"].items():
    strategy_manager.set_agent_budget(agent, budget["budget"])

# ExecutionManager checks before order placement
if order_size > allocation_plan["allocations"][agent]["max_per_symbol"]:
    return {"action": "REJECT", "reason": "Exceeds allocated budget"}
```

---

## Complementary Relationship

### They Work Together

```
SCENARIO: $350 account, 2 agents (BTC_DCA, ETH_MOMENTUM), BUY signal arrives

Step 1: Capital Governor (permission)
  ├─ Check: Bracket = MICRO ✓
  ├─ Check: Max 1 position, 0 open ✓
  ├─ Check: BTCUSDT is core symbol ✓
  └─ Decision: ALLOWED

Step 2: Capital Allocator (budget)
  ├─ Free USDT: $350
  ├─ BTC_DCA (core): $175 allocated
  ├─ Check: $12 order << $175 budget ✓
  └─ Decision: APPROVED

Step 3: Execution
  ├─ Order size: $12 (from Capital Governor)
  ├─ Budget check: $12 ≤ $175 allocated (from Capital Allocator)
  └─ Execute: BUY 0.00017 BTC @ $70,000 = $12
```

### Without One or the Other

**Without Capital Governor** (no position limits):
- ❌ Account could open 2+ positions (fragmented capital)
- ❌ Could trade 10 symbols (unfocused learning)
- ❌ Could rotate constantly (distracted from core edge)
- ✅ But: Budget still managed by Capital Allocator

**Without Capital Allocator** (no budget distribution):
- ✓ Position limits enforced (Governor)
- ✓ Position sizes limited (Governor)
- ❌ But: No intelligent budget distribution across agents
- ❌ All agents compete for same pool (unfair)

---

## When They Conflict (Priority)

If both apply different constraints, **Capital Governor wins**:

```python
# Example: $350 MICRO account

# Capital Governor says:
max_positions = 1
quote_per_position = 12.0

# Capital Allocator says:
btc_agent_budget = 175.0

# Actual execution uses BOTH as constraints:
order_size = min(
    12.0,           # ← Capital Governor limit (winner)
    175.0,          # Capital Allocator budget
    available_usdt  # Physical limit
)
# Result: $12 order (most conservative wins)
```

**Reason**: Governor enforces learning phase safety, Allocator optimizes within those safe boundaries.

---

## Configuration

### Capital Governor Config

```python
# No special config - uses only:
- NAV (from PnLCalculator)
- .env if FORCE_CAPITAL_BRACKET set

# Brackets are hardcoded (safe defaults):
MICRO < $500
SMALL $500-$2000
MEDIUM $2000-$10000
LARGE ≥ $10000
```

### Capital Allocator Config

```python
# From core/config.py CAPITAL_ALLOCATOR section:

ENABLED: true
INTERVAL_MIN: 15          # How often to replan

TIERS:
  core: 0.50              # 50% to stable agents
  growth: 0.35            # 35% to medium-risk agents
  experimental: 0.15      # 15% to new agents

MAX_GLOBAL_ALLOC_RATIO: 0.65   # Never allocate > 65% of capital
MIN_AGENT_BUDGET: 10.0         # Minimum per agent ($10)
TARGET_EXPOSURE_PCT: 0.20      # Global exposure target

IPO_POOL_RATIO: 0.10    # Optional carve-out for IPO agents
REQUIRE_PERF_SOURCE: false # Don't block on missing perf data
```

---

## Lifecycle & Timing

### Capital Governor
```
Initialization:
  → Created once in MetaController.__init__()
  → No async operations

Per Signal:
  → get_bracket(nav) - instant lookup
  → get_position_limits(nav) - instant lookup
  → get_position_sizing(nav) - instant lookup
  → Response time: < 1ms

When NAV crosses threshold:
  → Bracket changes (e.g., $350 → $520 triggers MICRO → SMALL)
  → Next signal gets new limits automatically
```

### Capital Allocator
```
Initialization:
  → Created in AppContext P7/P8
  → Async lifecycle (start/stop)

Periodic Cycle (every 15 min):
  → Calls plan_allocation()
  → Reads free USDT, agent performance
  → Emits AllocationPlan event
  → StrategyManager applies new budgets
  → Response time: ~500ms per cycle

Performance Snapshot:
  → Reads from PerformanceEvaluator/Monitor
  → Falls back to agent_scores
  → Recalculates tier assignments
```

---

## Summary Table

| Feature | Capital Governor | Capital Allocator |
|---------|------------------|-------------------|
| **Type** | Permission system | Distribution system |
| **Scope** | Position structure | Capital allocation |
| **Timing** | Per-trade | Every 15 minutes |
| **Input** | Account NAV | Free USDT + perf metrics |
| **Output** | Limits & constraints | Budget per agent |
| **Decision** | Yes/No allowed? | How much per agent? |
| **Update Freq** | On bracket change | Every 15 min cycle |
| **Complexity** | Low (lookup tables) | High (optimization) |
| **Async** | No | Yes |
| **Primary Goal** | Safe learning | Optimal growth |
| **Risk Focus** | Structure risk | Capital risk |

---

## Recommended Reading Order

1. **Capital Governor** (if building position structure):
   - Read: `CAPITAL_GOVERNOR_QUICK_REF.md`
   - Focus: Understanding your bracket limits

2. **Capital Allocator** (if optimizing budget distribution):
   - Read: Docstring in `core/capital_allocator.py` (~60 lines)
   - Focus: Understanding tier allocation

3. **Both Together**:
   - MetaController integration (Phase B of Governor roadmap)
   - StrategyManager integration (with Allocator)

---

## Your Scenario ($350 MICRO Account)

**Capital Governor Role**:
```
"Your $350 micro account gets:
  - 2 symbols max (BTCUSDT, ETHUSDT)
  - 1 position at a time
  - $12 per trade
  - NO rotation (deep learning)
  - 1.4x EV gate (permissive)"
```

**Capital Allocator Role**:
```
"Your $350 free USDT is split:
  - BTC_DCA (core): $175
  - ETH_MOMENTUM (growth): $122.50
  - Reserve: $52.50
  This rebalances every 15 min based on performance."
```

**Together**:
```
When BUY signal arrives:
  1. Governor checks: "BTCUSDT allowed?" → YES (core)
  2. Governor checks: "Position count OK?" → YES (0 < 1)
  3. Allocator checks: "Budget available?" → YES ($175 for BTC)
  4. Governor says: "Size this $12" ✓
  5. Allocator says: "You have $175 budget" ✓
  6. Execute: $12 BUY of BTC
```

---

## Key Differences at a Glance

**Capital Governor** = "**Am I allowed?**"
- Bracket-based permissions
- Structural limits
- Yes/No decisions

**Capital Allocator** = "**How much?**"
- Performance-based optimization
- Budget distribution
- Quantity decisions

**Together** = **Smart, safe, scalable trading**

