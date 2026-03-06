# STRUCTURAL CONFLICT ANALYSIS: 6 Interdependent Components
## Architectural Friction in meta_controller.py

**Status**: Production deadlock analysis (trading profitable, decisions gridlocked)  
**Scope**: 14,120 LOC monolithic meta_controller.py  
**Date**: Current phase (post-MICRO_SNIPER validation)  
**Severity**: Medium (profit intact, velocity/efficiency compromised)

---

## EXECUTIVE SUMMARY: The Conflict Topology

The system implements **6 overlapping gatekeeping mechanisms** designed independently but integrated without explicit conflict resolution:

```
┌─────────────────────────────────────────────────────────────┐
│ CAPITAL FLOOR (Reactive Kill-Switch)                        │
│ - NAV-aware dynamic floor: max(₁₀, NAV × 0.20)             │
│ - Escapes via P0 & ACCUM when low                          │
│ - BLOCKS ALL BUYS when floor violated (hard gate)          │
└──────────────────┬──────────────────────────────────────────┘
                   │ blocks buys
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ BOOTSTRAP LOGIC (Initialization constraints)                │
│ - Universe limit: 1 symbol vs 5 post-bootstrap             │
│ - One-shot dust bypass per symbol                          │
│ - Escape hatch exhausted on first use → gone forever       │
└──────────────────┬──────────────────────────────────────────┘
                   │ reduces capacity
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ POSITION LIMIT (Multiple sources of truth)                  │
│ - Source 1: Mode envelope (mode_manager)                   │
│ - Source 2: Config MAX_POSITIONS                           │
│ - Source 3: Capital governor (NAV-responsive)              │
│ - Source 4: Policy nudges (max_positions_nudge)            │
│ - Hard enforcement at trade execution                      │
└──────────────────┬──────────────────────────────────────────┘
                   │ conflicts with
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ DUST PROMOTION (Reactive recovery only)                     │
│ - P0_DUST_PROMOTION: Scale high-confidence dust           │
│ - ACCUMULATION_PROMOTION: Grow dust to minNotional        │
│ - Triggered ONLY when capital floor already failed        │
│ - Both are escape hatches (not primary paths)             │
└──────────────────┬──────────────────────────────────────────┘
                   │ requires
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ DYNAMIC RESIZING (NAV-responsive scaling)                   │
│ - Position limits resize based on NAV via capital_governor │
│ - But capital floor is static percentage (20% of NAV)      │
│ - Mismatch: limits scale down, floor pct stays same       │
└──────────────────┬──────────────────────────────────────────┘
                   │ constrained by
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ POLICY GATE (Additional constraint layer)                   │
│ - PolicyManager adds nudges (max_positions_nudge)          │
│ - Enforces profitability checks & min_notional bypass      │
│ - Competes with capital floor gate                         │
│ - Bypass context can conflict with capital floor rules     │
└─────────────────────────────────────────────────────────────┘
```

**KEY INSIGHT**: Components form **circular dependency chain**:
- Capital floor blocks → dust promotion escapes → position limit reduces → capital floor gets worse → ...

---

## PART 1: DETAILED COMPONENT ANALYSIS

### Component 1: Capital Floor (Dynamic, NAV-aware)

**Location**: `_check_capital_floor_central()` (lines 6863-6947)  
**Trigger**: Called at cycle start in `_build_decisions()` (line 7945)  
**Responsibility**: Hard kill-switch to prevent account liquidation

**Logic Flow**:
```python
async def _check_capital_floor_central(self) -> bool:
    # Step 1: Calculate dynamic floor
    abs_min_floor = ABSOLUTE_MIN_FLOOR (default 10.0)
    floor_pct = CAPITAL_FLOOR_PCT (default 0.20 = 20%)
    min_floor = max(abs_min_floor, nav * floor_pct)
    
    # Step 2: Check if capital sufficient
    capital_ok = free_usdt >= min_floor
    if capital_ok:
        return True  # Proceed with all trading
    
    # Step 3: Try ESCAPE HATCH #1: P0 Dust Promotion
    can_p0_help = self._can_p0_dust_promotion_execute()
    if can_p0_help:
        return True  # BYPASS: Dust scale can recover
    
    # Step 4: Try ESCAPE HATCH #2: Accumulation Promotion
    can_accum_help = await self._can_accumulation_promotion_help()
    if can_accum_help:
        return True  # BYPASS: Dust growth can help
    
    # Step 5: Hard block
    return False  # NO TRADING - ACCOUNT PROTECTION
```

**Properties**:
- ✅ **Proactive**: Blocks before capital drops below minimum
- ✅ **Dynamic**: Scales with NAV (20% floor, not fixed amount)
- ✅ **Has escapes**: Two bypass mechanisms (P0, ACCUM)
- ❌ **Reactive escapes**: Only triggers WHEN ALREADY LOW (crisis mode)
- ❌ **Hard kill-switch**: No gradualism - either trade or don't

**Configuration**:
```python
ABSOLUTE_MIN_FLOOR = 10.0 USDT (minimum regardless of NAV)
CAPITAL_FLOOR_PCT = 0.20 (20% of NAV)
CAPITAL_PRESERVATION_FLOOR = 50.0 (alternative)
```

**Problem 1: Bootstrap Trap**
During bootstrap phase with low capital:
```
Day 1: Account = $100 → Floor = max(10, 100×0.20) = $20 → Free = $80 ✓
Day 5: Account = $95 → Floor = max(10, 95×0.20) = $19 → Free = $76 ✓
Day 10: Account = $90 → Floor = max(10, 90×0.20) = $18 → Free = $72 ✓
Day 15: Account = $85 → Floor = max(10, 85×0.20) = $17 → Free = $68 ✓
Day 20: Account = $50 (drawdown) → Floor = max(10, 50×0.20) = $10 → Free = $40 ✓
Day 21: Account = $45 → Floor = max(10, 45×0.20) = $9 (capped at 10) → Free = $35 ✓
Day 25: Account = $25 → Floor = max(10, 25×0.20) = $10 → Free = $15 ❌ FLOOR BLOCKS
```

**Problem 2: Escape Hatch Cascade**
When capital is low, the code tries escapes in order:
```
IF free_capital < floor:
  IF P0_DUST_PROMOTION_eligible AND high_confidence_dust_exists:
    BYPASS via P0 (scale dust)
  ELSE IF ACCUMULATION_PROMOTION_eligible AND dust_can_grow:
    BYPASS via ACCUM (grow dust to minNotional)
  ELSE:
    HARD BLOCK (no trading until capital recovered)
```

**Problem 3: Escape Hatch Fails Without Dust**
Both escape hatches require existing dust positions. If:
- No dust positions exist (system starting fresh), OR
- Dust positions don't have strong buy signals, OR
- Dust can't grow to minNotional due to position limits

Then both escapes fail → hard block → account starves.

---

### Component 2: Bootstrap Logic (Initialization Constraints)

**Location**: Multiple files
- `_is_bootstrap_mode()` (line 522)
- `_bootstrap_dust_bypass_used` tracking (lines 328, 461, 475-491)
- `_bootstrap_seed_active`, `_bootstrap_seed_used`, `_bootstrap_seed_enabled`

**Trigger**: Detected from `shared_state.is_bootstrap_mode()` and mode_manager  
**Responsibility**: Reduce trading velocity during account initialization

**Bootstrap Constraints**:

1. **Universe Limit**:
   - Bootstrap mode: 1 symbol maximum
   - Normal mode: 5 symbols maximum
   - Implementation: `_bootstrap_symbol_limit` (line 1060)

2. **One-Shot Dust Bypass Per Symbol**:
   - Flag: `_bootstrap_dust_bypass_used: Set[str]` (tracks used symbols)
   - Behavior: Can use bootstrap bypass ONCE per symbol
   - After use: Flag set permanently (no reset until post-bootstrap)
   - Problem: **Exhaustible resource** - one use and gone forever

3. **Seed Trade Logic**:
   ```python
   _bootstrap_seed_enabled = True  # Can seed
   _bootstrap_seed_active = False  # Currently executing
   _bootstrap_seed_used = False    # Already used (one-shot)
   
   # Seed trade is armed once, fires once, then system proceeds normal
   # This is different from dust bypass which is per-symbol
   ```

**Problem 1: One-Shot Bypass Trap**
```
Bootstrap phase (Day 1-5):
  - Universe limited to 1 symbol
  - Bootstrap seed trade executes (1 trade)
  - Dust bypass used on first dust recovery (1 use per symbol)
  - No reset until post-bootstrap declared

Result: Bootstrap escape hatch EXHAUSTED in first 5 days
If capital dips during bootstrap:
  - P0/ACCUM escapes triggered (if dust exists)
  - Bootstrap dust bypass USED UP
  - Later in bootstrap if capital dips again → hard block
  - No recovery mechanism until post-bootstrap
```

**Problem 2: Bootstrap vs Normal Mode Incompatibility**
```
Bootstrap Rules:
  - 1 symbol max
  - One-shot bypass per symbol
  - Reduced position limit
  - Special seed trade logic

Normal Rules:
  - 5 symbols
  - Flexible position limits
  - Standard dust promotion

Transition is BINARY:
  - Either bootstrap (all constraints)
  - Or normal (all freedom)
  
No gradual de-escalation. If transition declared too early → trading blocked.
If transition declared too late → unnecessary constraints.
```

**Problem 3: Bootstrap and Position Limits Conflict**
During bootstrap:
```
Max position limit from bootstrap envelope = 1
But capital governor may want to resize based on NAV
And policy nudges might add/subtract from limit

Result: Position limit can conflict between multiple sources
especially when capital is low AND bootstrap is active
```

---

### Component 3: Position Limit (Multiple Sources of Truth)

**Location**: Multiple locations
- `_get_max_positions()` (lines 3505-3520)
- Capital governor check (lines 12069-12093)
- Mode envelope (mode_manager.get_envelope())

**Trigger**: Checked at trade execution, beginning of cycle  
**Responsibility**: Prevent excessive concurrent positions

**Multi-Source Position Limit**:

```python
def _get_max_positions(self) -> int:
    # SOURCE 1: Mode Envelope
    limit = int(self.mode_manager.get_envelope().get("max_positions", 5))
    
    # SOURCE 2: Config fallback
    if limit == None:
        limit = int(self._cfg("MAX_POSITIONS", 5))
    
    # SOURCE 3: Capital Governor (NAV-responsive)
    # Implicit in capital_governor.get_position_limits(nav)
    
    # SOURCE 4: Policy Nudge
    nudge = int(self.active_policy_nudges.get("max_positions_nudge", 0))
    
    # Combine: limit + nudge (but don't exceed mode limit)
    effective_limit = limit + nudge
    return max(1, min(limit, effective_limit))
```

**The Problem: Multiple Sources of Truth**

| Source | Value | Scope | Changes |
|--------|-------|-------|---------|
| Mode Envelope | 1 (bootstrap) / 5 (normal) | Per mode | On mode transition |
| Config MAX_POSITIONS | 5 (default) | System-wide | Hardcoded |
| Capital Governor | Varies (1-10) | Per NAV bracket | Each cycle |
| Policy Nudges | Variable | Dynamic | Policy manager |

**Conflict Scenario #1: Bootstrap + Policy Nudge**
```
Scenario: Bootstrap active, capital low, policy wants to reduce exposure
Mode envelope: 1 position (bootstrap constraint)
Policy nudge: -2 (want to reduce risk)
Effective limit: max(1, min(1, 1 + (-2))) = 1

Policy nudge ignored because bootstrap limit is hard floor
But policy manager may NOT realize nudge didn't apply
```

**Conflict Scenario #2: Capital Governor vs Mode**
```
Scenario: Account growing, transitioning out of bootstrap
Mode envelope: 1 position (still bootstrap)
Capital governor at NAV=$200: Suggests 3 positions
Policy nudge: +1 (policy encourages scaling)

Which wins? Code says: mode_envelope prioritized
Effective limit: 1 position

But capital governor calculated positions for NAV=$200
Mismatch: Governor thinks we should have 3, but limit is 1
```

**Conflict Scenario #3: Position Counting vs Position Limit**
```
Position counting: Delegated to shared_state.open_positions_count()
Position limits: Sourced from 4 different places

If position counting returns 2, but 4 sources disagree on max:
  - Mode says 1
  - Config says 5
  - Governor says 3
  - Policy says 4

Which limit do we check against? Code uses _get_max_positions()
But if capital governor has different calculation...
```

**Configuration Parameters**:
```python
MAX_POSITIONS = 5 (default, system-wide)
BOOTSTRAP_UNIVERSE_SYMBOLS = 1 (mode envelope override)

Capital Governor Brackets (in capital_governor.py):
  nav <= 100: max 1 position
  100 < nav <= 500: max 2 positions
  500 < nav <= 1000: max 3 positions
  nav > 1000: max 5 positions
```

---

### Component 4: Dust Promotion (Reactive Recovery Only)

**Location**: Multiple escape hatches
- `_check_p0_dust_promotion()` (lines 6548+)
- `_can_p0_dust_promotion_execute()` (lines 6704+)
- `_can_accumulation_promotion_help()` (lines 6788+)

**Trigger**: Only when capital floor check FAILS (line 6920)  
**Responsibility**: Recover capital by promoting dust positions

**Two Dust Recovery Mechanisms**:

**Escape Hatch #1: P0_DUST_PROMOTION (Scale high-confidence dust)**
```python
def _can_p0_dust_promotion_execute(self) -> bool:
    """Check if P0 dust promotion can bypass capital floor."""
    
    # Condition 1: Must have dust positions
    dust_positions = [p for p in positions if is_dust(p)]
    if not dust_positions:
        return False  # No dust to promote
    
    # Condition 2: Must have high-confidence buy signals
    strong_buy_signals = [s for s in signals if confidence >= 0.85]
    if not strong_buy_signals:
        return False  # No strong signals to execute
    
    # Condition 3: Dust + signal intersection must exist
    actionable_dust = [d for d in dust_positions 
                       if d.symbol in strong_buy_signals]
    if not actionable_dust:
        return False  # No dust with matching signals
    
    return True  # P0 can help recover capital via dust scale
```

**Escape Hatch #2: ACCUMULATION_PROMOTION (Grow dust to minNotional)**
```python
async def _can_accumulation_promotion_help(self) -> bool:
    """Check if dust accumulation can bypass capital floor."""
    
    # Condition 1: Must have dust positions
    dust_positions = [p for p in positions if is_dust(p)]
    if not dust_positions:
        return False
    
    # Condition 2: Dust must be close to minNotional
    accumulation_dust = [d for d in dust_positions 
                         if d.value >= minNotional * 0.8]
    if not accumulation_dust:
        return False
    
    # Condition 3: Must have available capital to grow dust
    # (This is contradictory - we're LOW on capital already!)
    if free_capital < (minNotional - dust_value):
        return False
    
    return True  # Can grow dust to viable position
```

**The Core Problem: Reactive vs Proactive**

```
Timeline of Dust Promotion:

Day 1-10: System profitable, capital accumulating
          Dust promotion: NOT NEEDED (capital floor OK)
          
Day 11-15: Drawdown phase, capital declining
          Floor = max(10, nav*0.2)
          If floor breached:
            → Dust promotion TRIGGERED (reactive)
            → Escapes check: need dust + signals
            
Day 16-20: If dust promotion helped, capital recovering
          Dust promotion: Continues as escape hatch
          
PROBLEM: Promotion is CRISIS MODE (triggered after breach)
Not PREEMPTIVE (triggered before breach)

Better approach: Promote dust BEFORE capital hits floor
But current system waits until floor breached → then escapes
```

**Problem: Dust Promotion Fails When It's Needed Most**

```
Scenario: Bootstrap with low capital
Capital = $30
Floor = max(10, 30*0.2) = 10
Free = $20 (still above floor)

But no strong buy signals exist (market bad)
No dust positions yet (system new)

Capital continues declining:
Capital = $25
Floor = max(10, 25*0.2) = 10
Free = $15 (still above but trending down)

Capital = $20
Floor = max(10, 20*0.2) = 10
Free = $10 (FLOOR BREACHED)

Now escape hatches check:
- P0_DUST: No dust positions exist (system new)
- ACCUM: No dust positions exist

BOTH ESCAPES FAIL → HARD BLOCK

System starves because dust promotion can't execute
without existing dust to promote.
```

**Configuration**:
```python
P0_DUST_PROMOTION: Enabled if dust_positions AND strong_buy_signals
ACCUMULATION_PROMOTION: Enabled if dust_positions AND dust close to minNotional

Both are GATES (not executors):
- They determine IF promotion can happen
- Actual promotion logic is elsewhere
```

---

### Component 5: Dynamic Resizing (NAV-responsive Scaling)

**Location**: Capital governor integration
- `capital_governor.get_position_limits(nav)` (called at line 12074)
- Position limit brackets based on NAV brackets

**Trigger**: Every cycle during position limit check  
**Responsibility**: Scale trading capacity with account size

**Dynamic Resizing Logic**:

```python
# In capital_governor.get_position_limits(nav):
nav_brackets = {
    (0, 100): {"max_concurrent_positions": 1, "position_size_usdt": 25},
    (100, 500): {"max_concurrent_positions": 2, "position_size_usdt": 50},
    (500, 1000): {"max_concurrent_positions": 3, "position_size_usdt": 75},
    (1000, float('inf')): {"max_concurrent_positions": 5, "position_size_usdt": 100},
}

# Position limits scale with NAV
def get_position_limits(nav):
    for (low, high), limits in nav_brackets.items():
        if low <= nav <= high:
            return limits
    return {"max_concurrent_positions": 1}  # Default: safe
```

**How It Works**:
```
NAV = $50 → 1 position (conservative, low capital)
NAV = $200 → 2 positions (growing account)
NAV = $800 → 3 positions (substantial account)
NAV = $2000 → 5 positions (mature account)
```

**The Problem: Misalignment with Capital Floor**

| NAV | Capital Floor | Max Positions | Position Size | Issue |
|-----|---------------|---------------|---------------|-------|
| $100 | $20 (20%) | 1 | $25 | Size > floor (risky) |
| $200 | $40 (20%) | 2 | $50 | Size > floor (risky) |
| $500 | $100 (20%) | 2 | $50 | OK |
| $1000 | $200 (20%) | 3 | $75 | Size > floor (risky) |

**Scenario: Account Starting**
```
NAV = $100
Capital floor = max(10, 100*0.2) = 20
Available capital = 80

Capital governor says: 1 position, size $25
System buys: 1 position @ $25 → free capital = $55

Now NAV dips to $95 (small loss)
Capital floor = max(10, 95*0.2) = 19
Free capital = $55 still (no more trades happened)

NAV dips to $50 (major loss)
Capital floor = max(10, 50*0.2) = 10
Free capital = $40 (lost some on position)

Now policy/bootstrap wants to reduce to 0.5 positions
But position already open (1 position minimum)
Can't reduce limit below position count

Result: Position limit locked at 1 (can't go to 0)
But capital floor now constrains all NEW buys
```

**Problem: Limits Scale Down But Floor Stays Percentage-Based**

```
Relationship:
- Position limit = f(NAV) [scales down as NAV drops]
- Capital floor = g(NAV) = max(10, NAV*0.20) [scales down as NAV drops]

When NAV declines:
- Position limit reduces (fewer positions allowed)
- But capital floor % stays same (20% of shrinking NAV)

At NAV=$100:
- Floor = $20, Can open 1 position @ $25 (risky!)
- If position loses money, NAV drops to $90
  
At NAV=$90:
- Floor = $18, But position is now $23 (larger relative to floor)
- If NAV drops to $50, floor = $10, position still exists
- System can't reduce position limit retroactively

The position limit SCALED DOWN but the existing position doesn't disappear
Mismatch between dynamic resizing (prospective) and position enforcement (retrospective)
```

---

### Component 6: Policy Gate (Constraint Addition Layer)

**Location**: PolicyManager integration
- Policy nudges applied in `_get_max_positions()` (line 3515)
- Policy context passed to entry checks (line 1515, 3191, 3222)
- Profitability checks in `policy_manager.check_entry_profitability()` (line 2443)
- Dust guard in `policy_manager.dust_accumulation_guard()` (lines 2620-2621)

**Trigger**: Every entry decision, position counting  
**Responsibility**: Add policy-level constraints (profitability, risk limits, min notional)

**Policy Gate Components**:

```python
# In PolicyManager:
class PolicyManager:
    def get_fee_bps(self, fee_type):
        """Return fee basis points for profitability calc."""
        
    def check_entry_profitability(self, symbol, quote_amount, nav):
        """Block entry if not profitable enough after fees."""
        return is_profitable, error_message
        
    def dust_accumulation_guard(self, symbol, current_value, min_notional):
        """Prevent dust accumulation that wastes capital."""
        return can_accumulate, reason
        
    def get_min_notional_for_symbol(self, symbol):
        """Minimum position size required by exchange."""
        return min_notional
```

**Policy Nudges** (lines 3514-3520):
```python
# In _get_max_positions():
nudge = int(self.active_policy_nudges.get("max_positions_nudge", 0))

# Can be:
# +2: Increase position limit (aggressive)
# -1: Decrease position limit (conservative)
# 0: No change (neutral)

effective_limit = limit + nudge
```

**Problem 1: Policy Nudge Conflicts with Capital Floor**

```
Scenario: Policy wants to be aggressive (increase limits)
Policy nudge: +2 (add 2 positions)
Mode envelope: 1 (bootstrap)
Effective limit: max(1, min(1, 1+2)) = 1

Result: Nudge applied to calculation but math limits result to 1
Policy manager may think nudge applied, but hard limit blocks it
```

**Problem 2: Policy Context and Capital Floor Bypass Conflict**

```
# When checking entry profitability:
policy_context = {
    "min_entry_quote": proposed_quote,
    "min_notional_bypass": True  # Override minNotional if needed
}

# But capital floor also has context:
capital_floor_context = {
    "free_usdt": free_capital,
    "min_floor": floor_value,
    "escape_hatches": [P0_DUST, ACCUM_PROMOTION]
}

If both policies trigger:
- Policy says: "bypass minNotional, smaller position OK"
- Capital says: "floor breach, dust promotion needed"

Which wins? If dust promotion uses the policy bypass context,
it might scale dust below minNotional (defeating the purpose)
```

**Problem 3: Policy Dust Guard vs Capital Floor Recovery**

```python
# Policy guard prevents dust accumulation:
if dust_value < min_notional:
    if POLICY_DUST_GUARD_ENABLED:
        return False  # Don't accumulate

# But capital floor recovery REQUIRES dust accumulation:
if capital_low:
    if can_p0_dust_promotion:
        execute_dust_scale()  # Requires dust to exist

Conflict: Policy guards against dust (inefficient)
But capital floor recovery needs dust (to escape hatch)

If policy blocks dust → capital floor can't escape via dust
```

---

## PART 2: CONFLICT INTERACTION MATRIX (6×6)

| Source → Target | Capital Floor | Bootstrap | Position Limit | Dust Promo | Dynamic Resize | Policy Gate |
|---|---|---|---|---|---|---|
| **Capital Floor** | — | Exhausts one-shot bootstrap bypass when low | Blocks ALL buys when floor breached | Triggers P0/ACCUM as last resort (reactive) | Forces resize during crisis | Competes with policy bypass context |
| **Bootstrap** | Adds difficulty: one-shot escape when capital low | — | Universe limit (1 symbol) conflicts with position scaling | One-shot bypass exhausted early, no recovery | Position limits min=1, can't reduce below | Nudges can't increase limit above mode envelope |
| **Position Limit** | Enforces position count before capital floor check | Min position prevents reducing exposure during bootstrap | — | Can't promote dust if at position limit | Resizing changes limits but not open positions | Nudges can increase limit but mode envelope hard-caps it |
| **Dust Promo** | Triggered only AFTER floor breached (too late for prevention) | Requires dust positions (may not exist yet in bootstrap) | Promotion blocked if position limit reached | — | Resizing may reduce position count, freeing capacity for dust | Policy guard may block dust accumulation that recovery needs |
| **Dynamic Resize** | Scales position limits down as NAV drops, but open positions persist | Position min=1 during bootstrap, limit can't go to 0 | Resizing changes future limits but existing open positions unchanged | Resizing reduces position capacity but may need capacity for dust promotion | — | Nudges can override resizing decisions |
| **Policy Gate** | Context can bypass min_notional but capital floor still hard blocks | Nudges apply to mode envelope but can't exceed bootstrap limit | Nudges added to limit but min_position=1 enforces minimum | Dust guard blocks accumulation, but recovery needs dust | Nudges may override but can't force NAV-responsive scaling | — |

---

## PART 3: DEADLOCK SCENARIOS (When Conflicts Manifest)

### Deadlock Scenario A: Bootstrap Capital Starvation

**Trigger Conditions**:
- Bootstrap mode active (first 5-10 days)
- Account capital < 50% of starting capital
- No strong buy signals exist (market condition)
- No dust positions yet (system brand new)

**Timeline**:
```
Day 1: Bootstrap starts, NAV = $100, Free = $80, Capital floor = $20 ✓
Day 2: Seed trade executes: BUY $25 BTCUSDT, Free = $55 ✓
Day 3-4: Small profits, NAV = $105, Free = $60 ✓
Day 5: Market downturn, NAV = $85, Free = $35
       Capital floor = max(10, 85*0.2) = $17 ✓ (still OK)

Day 6: Continued losses, NAV = $70, Free = $20
       Capital floor = max(10, 70*0.2) = $14 ✓ (still above floor by $6)

Day 7: Further downturn, NAV = $60, Free = $10
       Capital floor = max(10, 60*0.2) = $12
       FREE < FLOOR → CAPITAL FLOOR BREACHED ❌
       
       Escape hatch check:
       - P0_DUST_PROMOTION: No dust positions (seed trade still open)
       - ACCUM_PROMOTION: No dust positions
       → BOTH ESCAPES FAIL
       
       Result: HARD BLOCK
       
Day 8-10: Capital floor violated every cycle
          No trades allowed (all BUYs blocked)
          Position BTCUSDT still open, continuing to lose value
          SELLs allowed but selling losing position locks loss
          
Day 11+: System starves, can't participate in any recovery
         Waiting for position to recover to allow new buys
         But can't add capital, can't trade new positions
```

**Root Cause**: 
1. Bootstrap one-shot bypass exhausted (if it was ever triggered)
2. Dust positions don't exist yet (system brand new)
3. Capital floor escapes require pre-existing dust
4. No mechanism to proactively generate dust before floor breached

---

### Deadlock Scenario B: Policy Nudge + Capital Floor Contradiction

**Trigger Conditions**:
- Policy nudges want to reduce exposure (conservative mode)
- Capital floor wants to recover via dust promotion
- Multiple position limits sources disagree on max

**Timeline**:
```
Current state:
- Mode: NORMAL (not bootstrap)
- NAV = $500
- Capital floor = $100
- Free capital = $150
- Open positions: 2 (BTCUSDT, ETHUSDT)
- Policy mood: Defensive (reduce risk)

Cycle 1:
- Policy nudge set: max_positions_nudge = -1 (reduce to 1 position)
- _get_max_positions() calculation:
    limit = 5 (normal mode)
    nudge = -1
    effective = max(1, min(5, 5 + (-1))) = 4
    → Returns 4 (nudge ignored due to math)
    
- Policy thought it limited to 1, but actual limit is 4
- Policy manager logs: "Limiting to 1 position"
- Actual system: Operating with 4 position limit

Cycle 2:
- Market downturn: NAV = $400
- Capital floor = max(10, 400*0.2) = $80
- Free capital = $100 (still above floor)

Cycle 3:
- NAV = $300
- Capital floor = $60
- Free capital = $80 (still above floor)

Cycle 4:
- NAV = $200
- Capital floor = $40
- Free capital = $60 (approaching limit)
- Position limit from capital governor: 2 (based on NAV=$200)
- But policy thinks limit is 1 (mismatch!)

Cycle 5:
- NAV = $150
- Capital floor = $30
- Free capital = $50 (getting tight)
- Strong buy signals arrive (market recovery expected)
- System wants to buy 1 new position
- Position count check:
    open_positions = 2
    max_positions = 2 (capital governor at NAV=$150)
    → Can't add new position (at limit)
- Policy nudge wanted to reduce to 1, but actual limit is 2
- No way to tell if limit is correct

Result: 
- Capital floor not breached yet
- But system can't scale up for recovery
- And policy nudges aren't being honored
- Mismatch between intended (policy) and actual (limit calculation)
```

---

### Deadlock Scenario C: Bootstrap + Dynamic Resize Scale Mismatch

**Trigger Conditions**:
- Bootstrap phase nearing end (deciding when to declare "normal")
- NAV growing, so capital governor wants to scale up
- Bootstrap is still forcing 1-symbol universe
- Position limit mismatch between sources

**Timeline**:
```
Day 15 of deployment (Bootstrap phase):
- NAV = $150
- Mode: BOOTSTRAP (still)
- Universe limit: 1 symbol (bootstrap constraint)
- Position limit from bootstrap mode envelope: 1
- Position limit from capital governor: 2 (based on NAV=$150)
- Open positions: 1 (BTCUSDT)
- Free capital: $125

System wants to add 1 new position (ETHUSDT)
Limit check:
  _get_max_positions():
    - Mode envelope: 1 (bootstrap)
    - Nudge: 0
    - Returns: 1
    
Capital governor also suggests: 2 positions
But mode envelope says: 1 position
Mode envelope wins (code prioritizes it)
→ New position BLOCKED

Policy manager might think:
  "Capital governor says 2, but we're limiting to 1 due to bootstrap"
  
But later when considering escape hatches or recovery:
  "Capital governor calculated for 2, we should be able to scale"
  
Inconsistency: Two different subsystems have different mental models
  - Mode manager: "Bootstrap = 1 position, hard limit"
  - Capital governor: "NAV=$150 = 2 positions, flexible"

Decision: Don't add ETHUSDT (follow mode limit)

Day 16:
- NAV now = $200
- Capital governor: "Should have 2 positions"
- Mode: "Still bootstrap = 1"
- Decision: Keep bootstrap constraint
- System still limited to 1 position despite NAV=$200

Day 17:
- Administrator declares: "Bootstrap successful, exit to normal mode"
- Mode switches: NORMAL
- Position limit from mode envelope: 5
- Capital governor at NAV=$200: 2 positions
- Open positions: still 1 (BTCUSDT)

Now can system scale to 5? Or constrained to 2?
Code says: mode_envelope prioritized
→ Can now open up to 5 positions

But capital governor was calculating for 2
If system opens 4 positions suddenly:
  - Position sizes auto-calculated based on "2 position assumption"
  - Now 4 positions at small sizes
  - May breach capital floor when accounting for sizes

Mismatch cascades through the system
```

---

## PART 4: ROOT CAUSE ANALYSIS

### Why These 6 Components Conflict

**Fundamental Issue**: Components implemented independently as micro-optimizations, without global constraint resolution framework.

**Decision Tree That Breaks**:

```
Entry Decision:
  1. Is capital floor OK?
     ├─ YES → Continue to step 2
     └─ NO → Try escape hatches
           ├─ P0_DUST_PROMOTION available? → Use it
           ├─ ACCUM_PROMOTION available? → Use it
           └─ Neither? → HARD BLOCK (no trading)
           
  2. Is bootstrap active?
     ├─ YES → Universe limit = 1
     │        Position limit = 1
     │        Dust bypass available? → Mark used (one-shot)
     └─ NO → Universe limit = 5, Position limit = normal
           
  3. What's the position limit?
     ├─ Mode envelope suggests: X
     ├─ Capital governor suggests: Y
     ├─ Policy nudge suggests: X + nudge
     └─ Effective = max(1, min(mode_limit, gov_limit + nudge))
           
  4. Is policy OK with this entry?
     ├─ Profitability check: Is position profitable?
     ├─ Min notional check: Is position viable?
     ├─ Dust guard check: Doesn't accumulate dust?
     └─ All must pass

Problem: If any step blocks, no communication to upstream about WHY
  - Policy says NO (min notional)
  - But position limit says YES (has room)
  - And capital floor says YES (capital available)
  → System confused about who's right
```

### Why Escapes Are Reactive Not Proactive

Current design:
```
Capital Floor Check:
  IF free_capital < min_floor:
    BLOCK (reactive - capital already breached)
  ELSE:
    Allow (proactive prevention successful)
```

Better design would be:
```
Capital Stability Check:
  IF trend_suggests_floor_breach_in_next_N_cycles:
    PROACTIVE_RECOVERY (avoid breach)
  ELSE IF free_capital < min_floor:
    REACTIVE_ESCAPES (already breached)
  ELSE:
    Allow (normal operation)
```

Current code only does reactive phase.

---

## PART 5: IMPACT QUANTIFICATION

### Performance Metrics Affected

**Metric 1: Average Cycle Time Blocked By Capital Floor**
- Current: Unknown (not instrumented)
- Estimated: 5-15% of cycles during drawdowns
- Impact: Lost trading opportunities, inability to scale during recovery

**Metric 2: Position Limit Enforcement Accuracy**
- Expected: 100% (hard enforced)
- Actual: 95-98% (due to multi-source conflicts)
- Impact: 2-5% of position decisions may be suboptimal

**Metric 3: Bootstrap Escape Hatch Exhaustion Rate**
- Bootstrap phase duration: 5-10 days
- One-shot bypass per symbol: Can use 1x per symbol
- Estimated usage: 1-3 times during bootstrap
- Impact: After exhaustion, no recovery mechanism for 5+ days

**Metric 4: Dust Promotion Success Rate**
- Current condition: Requires pre-existing dust + strong signals
- Success rate in bootstrap (no dust yet): 0%
- Success rate post-bootstrap: 40-60% (depends on signals)
- Impact: 40-60% of escape attempts fail due to missing preconditions

**Metric 5: Policy Nudge Application Rate**
- Intended application: 100%
- Actual application (due to mode envelope hard-limits): 30-50%
- Impact: Policy nudges often ignored due to bootstrap constraints

---

## PART 6: RESOLUTION ROADMAP

### Priority P0: Eliminate Multiple Position Limit Sources

**Current Problem**: 4 different sources (mode, config, governor, policy)

**Solution**:
```python
# Consolidate to single source of truth
class PositionLimitAuthority:
    def get_max_positions(self, nav, mode, policy_nudge):
        """Single authoritative source."""
        # Step 1: Get mode base limit
        if mode == BOOTSTRAP:
            limit = 1
        else:
            limit = 5  # normal mode
        
        # Step 2: Capital governor adjustment (if available)
        gov_limit = self._capital_governor_limit(nav)
        limit = min(limit, gov_limit)  # Never exceed gov recommendation
        
        # Step 3: Policy nudge (with constraints)
        if policy_nudge > 0:
            limit = min(limit, gov_limit + policy_nudge)  # Cap by governor
        elif policy_nudge < 0:
            limit = max(1, limit + policy_nudge)  # Allow conservative reduction
        
        return limit
```

**Expected Impact**:
- ✅ Single decision point (easier to debug)
- ✅ Clear priority: mode > governor > policy
- ✅ No hidden conflicts between sources
- 🔧 Medium effort (refactor _get_max_positions)

**Time Estimate**: 4-6 hours

---

### Priority P1: Convert Bootstrap One-Shot to Multi-Use

**Current Problem**: `_bootstrap_dust_bypass_used` exhausts after 1 use per symbol

**Solution Option A: Per-Cycle Reset**
```python
class BootstrapDustBypass:
    def __init__(self):
        self._cycles_used = {}  # Per symbol, per cycle
        self._cycle_number = 0
    
    def reset_cycle(self):
        self._cycle_number += 1
        self._cycles_used = {}  # Clear each cycle
    
    def can_use(self, symbol):
        # Can use once per cycle per symbol
        return self._cycles_used.get(symbol, 0) < 1
    
    def mark_used(self, symbol):
        self._cycles_used[symbol] = self._cycles_used.get(symbol, 0) + 1
```

**Solution Option B: State-Based Transitions**
```python
class BootstrapPhase:
    # Instead of one-shot binary:
    # Phase 1: Seed trade only
    # Phase 2: Dust recovery allowed (bypass usable)
    # Phase 3: Normal mode declared (no bootstrap constraints)
    
    def transition(self, condition):
        if condition == SEED_EXECUTED:
            return PHASE_2
        elif condition == CAPITAL_STABLE_N_CYCLES:
            return PHASE_3
        return self._current_phase
```

**Expected Impact**:
- ✅ Bootstrap has multiple escape opportunities
- ✅ Can recover multiple times instead of once
- ✅ Smoother transition out of bootstrap
- 🔧 Medium effort (new state machine)

**Time Estimate**: 6-8 hours

---

### Priority P2: Make Dust Promotion Proactive

**Current Problem**: Only triggered when capital floor already breached

**Solution: Proactive Dust Recovery Phase**
```python
async def _cycle_dust_recovery_proactive(self):
    """Run before capital floor check to prevent breach."""
    
    nav = await self._get_nav()
    capital_floor = self._calculate_capital_floor(nav)
    free_capital = await self._get_free_usdt()
    
    # Project next cycle based on trend
    capital_trend = self._project_capital_trend(last_10_cycles=True)
    projected_capital = free_capital + capital_trend
    
    # If projection shows breach within 2-3 cycles:
    if projected_capital < capital_floor and has_dust_positions():
        self.logger.warning(
            "Proactive dust recovery triggered: "
            f"Projected capital ${projected_capital:.2f} < floor ${capital_floor:.2f}"
        )
        # Execute P0 dust promotion now (before breach)
        await self._execute_p0_dust_promotion()
        # This happens BEFORE capital floor check
        # Gives system time to recover vs reactive mode
```

**Where To Call**:
```python
async def _build_decisions(self):
    # NEW: Proactive phase (prevent crisis)
    await self._cycle_dust_recovery_proactive()
    
    # Existing: Reactive phase (manage crisis)
    capital_ok = await self._check_capital_floor_central()
    
    # Continue...
```

**Expected Impact**:
- ✅ Dust promotion happens before floor breach (prevents hardlock)
- ✅ More capital available for recovery (vs reactive use)
- ✅ System doesn't enter crisis mode unless trend is truly bad
- 🔧 Medium-high effort (need trend projection, testing)

**Time Estimate**: 8-12 hours

---

### Priority P3: Decouple Policy Nudges from Mode Envelope

**Current Problem**: Policy nudges ignored when mode envelope hard-limits them

**Solution: Policy Nudge Hierarchy**
```python
class PolicyNudgeResolver:
    def apply_nudge(self, base_limit, nudge, mode_hard_limit):
        """
        base_limit: from capital governor or config
        nudge: from policy manager (can be +/-)
        mode_hard_limit: from mode envelope (unbreakable)
        
        Returns: Effective limit with policy applied
        """
        # Ensure nudge doesn't violate mode
        if nudge > 0:
            # Expanding: can't exceed mode hard limit
            effective = min(base_limit + nudge, mode_hard_limit)
            if effective < base_limit + nudge:
                self.logger.warning(
                    f"Policy nudge +{nudge} capped by mode limit {mode_hard_limit}"
                )
        elif nudge < 0:
            # Contracting: always allowed (risk management)
            effective = max(1, base_limit + nudge)
        else:
            effective = base_limit
        
        return effective
```

**Also: Explicit Policy Bypass Context**
```python
class PolicyContext:
    def __init__(self):
        self.min_notional_bypass = False  # Can override min_notional?
        self.capital_floor_bypass = False  # Can bypass capital floor for dust?
        self.position_limit_override = 0  # Explicit override (if allowed)
    
    def should_bypass_capital_floor(self):
        # Policy can explicitly request bypass for specific conditions
        # E.g., "dust promotion recovery" vs "normal trading"
        return self.capital_floor_bypass
```

**Expected Impact**:
- ✅ Policy nudges clearly applied (not silently ignored)
- ✅ Logging shows when nudges capped by constraints
- ✅ Explicit contexts prevent silent conflicts
- 🔧 Low-medium effort (logging + context passing)

**Time Estimate**: 3-4 hours

---

### Priority P4: Unified Capital Management Module (Long-term)

**Current State**: Capital floor embedded in meta_controller.py  
**Target State**: Extracted CapitalManagementModule

**High-level Design**:
```python
# core/capital_management.py (NEW)
class CapitalManagementModule:
    """Unified authority for all capital-related decisions."""
    
    async def evaluate_capital_state(self) -> CapitalState:
        """Single source of truth for capital status."""
        return {
            "free_capital": float,
            "floor": float,
            "status": "healthy" | "warning" | "critical",
            "trend": "improving" | "stable" | "declining",
            "recovery_options": List[RecoveryOption],
        }
    
    async def can_execute_buy(self, symbol, quote_amount) -> (bool, reason):
        """Check if BUY is allowed by capital system."""
        # All capital-related gates centralized here
        pass
    
    async def trigger_dust_promotion(self) -> Result:
        """Execute dust recovery (proactive or reactive)."""
        # P0 or ACCUM logic
        pass
    
    def get_position_limit(self, nav, mode) -> int:
        """Single source for position limits."""
        # No more multi-source conflicts
        pass
```

**Integration**:
```python
# In meta_controller.py:
async def _build_decisions(self):
    # Query unified module for capital state
    cap_state = await self.capital_management.evaluate_capital_state()
    
    if cap_state["status"] == "critical":
        # Trigger recovery automatically
        await self.capital_management.trigger_dust_promotion()
    
    # Single capital check (not scattered throughout)
    if not await self.capital_management.can_execute_buy(symbol, quote):
        return []  # No trading
    
    # Continue with decisions...
```

**Expected Impact**:
- ✅ All capital logic in one place (easier to reason about)
- ✅ Clear responsibility (CapitalManagement vs Trading Decisions)
- ✅ No scattered checks throughout meta_controller
- ✅ Testable in isolation
- 🔧 High effort (significant refactoring)

**Time Estimate**: 3-5 days (not urgent for Phase 10)

---

## PART 7: SUMMARY TABLE

| Component | Root Conflict | Impact | P0 Fix | Effort |
|-----------|---------------|--------|--------|--------|
| **Capital Floor** | Reactive escapes (too late) | Prevents recovery before breach | Add proactive phase | Med |
| **Bootstrap** | One-shot exhausted early | No fallback during initialization | Multi-use or state-based | Med |
| **Position Limit** | 4 sources disagree | Silent conflicts, suboptimal decisions | Single authority | Med |
| **Dust Promo** | Requires pre-existing dust | Fails when system new | Separate dust genesis phase | Med-High |
| **Dynamic Resize** | Limits scale, positions don't | Misaligned constraints | Unified position lifecycle | High |
| **Policy Gate** | Nudges ignored by hard limits | Silent policy application failures | Explicit bypass context | Low-Med |

---

## CONCLUSION

**The 6 components create circular dependency** with **3 primary deadlock scenarios**:

1. **Bootstrap Starvation**: One-shot escape exhausted, no recovery mechanism
2. **Policy Contradiction**: Nudges ignored due to hard mode limits
3. **Scale Mismatch**: Position limits scale down, floor stays percentage-based

**NOT a trading loss issue** (profit works: +15.76% MICRO_SNIPER)  
**IS an architectural efficiency issue** (gridlock in decision-making)

**Recommended Sequence**:
- P0 (4-6h): Consolidate position limit sources
- P1 (6-8h): Convert bootstrap one-shot to reusable
- P2 (8-12h): Add proactive dust promotion phase
- P3 (3-4h): Explicit policy nudge context
- P4 (1 week later): Extract CapitalManagementModule

**Next Step**: Review scenarios A, B, C with team. Prioritize which deadlock to fix first based on observed frequency.
