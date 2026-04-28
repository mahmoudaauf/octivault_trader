# Technical Deep-Dive: Symbol Entry/Exit Logic Implementation

## 1. Current Code Architecture Analysis

### Location: core/meta_controller.py

#### 1.1 Position Blocking Logic (Lines 2977-3050)

```python
async def _position_blocks_new_buy(self, symbol: str, existing_qty: float) -> Tuple[bool, float, float, str]:
    """
    Returns: (blocks, position_value, significant_floor, reason)
    """
    # RULE: If existing position is "significant", blocks new entry
    # RULE: If existing position is "dust", doesn't block
    # RULE: If existing position is "unhealable", doesn't block
```

**Current Logic:**
```
Position Value = Quantity × Current_Price

if position_value < $1 (PERMANENT_DUST_THRESHOLD):
    → Does NOT block new entry

elif position_value < significant_floor ($10-25):
    → Does NOT block new entry

elif dust_classification == "UNHEALABLE_LT_MIN_NOTIONAL":
    → Does NOT block new entry

else:  # position_value > significant_floor
    → BLOCKS new entry (significant position)
```

**Problem:** Even micro positions ($5-10) that are losing block new entries

#### 1.2 Regime Position Limits (Lines 1590-1620)

```python
def _regime_check_max_positions(self) -> bool:
    """
    Check if we've reached max open positions for regime.
    
    MICRO_SNIPER: Max 1 open position    ← Current regime
    STANDARD: Max 2 open positions
    MULTI_AGENT: Max 3+ open positions
    """
    
    max_pos = self._get_max_positions()
    active_pos_count = count_active_positions()
    
    if active_pos_count >= max_pos:
        return False  # Blocks trade
    return True  # Allows trade
```

**Problem:** All positions count equally toward limit; no tier classification

---

## 2. Proposed Implementation: Tier-Based System

### 2.1 Position Classification Engine

```python
class PositionTier(Enum):
    """Position tier classification"""
    TIER1 = "primary_active"      # Significant, blocks new entry
    TIER2 = "secondary_micro"      # Small, parallel, doesn't block
    TIER3 = "dust_liquidation"     # < $1, passive consolidation
    UNHEALABLE = "unhealable_dust" # Exchange minimum, never blocked


async def classify_position(symbol: str, qty: float) -> PositionTier:
    """
    Classify position into tier based on multiple factors
    """
    
    # Get position value
    price = await safe_price(symbol)
    position_value = qty * price
    
    # Check 1: Dust floor
    if position_value < DUST_FLOOR_USDT:  # $1
        return PositionTier.TIER3
    
    # Check 2: Unhealable dust
    _, min_notional = await compute_symbol_trade_rules(symbol)
    if position_value < min_notional:  # $10-50 exchange minimum
        return PositionTier.UNHEALABLE
    
    # Check 3: Secondary micro threshold
    if position_value < SECONDARY_THRESHOLD_USDT:  # $10-15
        return PositionTier.TIER2
    
    # Check 4: Primary active position
    if position_value >= PRIMARY_THRESHOLD_USDT:  # $25+
        return PositionTier.TIER1
    
    # Check 5: Time-based classification
    # Position > 4 hours old → may be marked TIER2 (aging)
    if position_age > 4 * 3600:
        return PositionTier.TIER2
    
    return PositionTier.TIER1  # Default: primary


# Configuration
DUST_FLOOR_USDT = 1.0
SECONDARY_THRESHOLD_USDT = 10.0
PRIMARY_THRESHOLD_USDT = 25.0
UNHEALABLE_THRESHOLD_USDT = 50.0
```

### 2.2 Enhanced Blocking Logic

```python
async def position_blocks_new_buy(self, symbol: str, qty: float) -> Tuple[bool, str]:
    """
    Enhanced blocking logic with tier support
    
    Returns: (is_blocked, reason)
    """
    
    if qty <= 0:
        return False, "no_position"
    
    # Classify existing position
    tier = await classify_position(symbol, qty)
    
    # Decision matrix
    if tier == PositionTier.TIER1:
        return True, f"tier1_active_position_blocks"
    
    elif tier == PositionTier.TIER2:
        # Check if Tier2 slot available
        tier2_count = count_tier2_positions()
        tier2_max = get_tier2_max()
        if tier2_count >= tier2_max:
            return True, f"tier2_slots_full"
        return False, "tier2_allows_parallel"
    
    elif tier == PositionTier.TIER3:
        return False, "dust_never_blocks"
    
    elif tier == PositionTier.UNHEALABLE:
        return False, "unhealable_never_blocks"
    
    return True, "unknown_tier"
```

### 2.3 Capital Allocation Formula

```python
async def calculate_new_position_size(
    symbol: str,
    config_entry_size: float,
    regime: str
) -> float:
    """
    Calculate position size considering:
    - Available capital
    - Active positions
    - Profitability history
    - Tier requirements
    """
    
    # Get available capital
    total_balance = await get_balance()
    allocated = sum_tier1_allocations()
    reserved = sum_tier2_allocations()
    available = total_balance - allocated - reserved
    
    # Base size from config
    base_size = config_entry_size
    
    # Regime adjustment
    regime_factor = {
        "MICRO_SNIPER": 0.8,   # Conservative: 80% of config
        "STANDARD": 0.9,        # Moderate: 90% of config
        "MULTI_AGENT": 1.0,     # Aggressive: 100% of config
    }.get(regime, 1.0)
    base_size *= regime_factor
    
    # Capital constraint
    max_by_capital = available / (regime_max_positions + 1)
    
    # Profitability multiplier (compounding)
    cumulative_pnl = await get_cumulative_pnl()
    starting_balance = 103.89  # Historical starting point
    if cumulative_pnl > 0:
        compounding_mult = 1.0 + (cumulative_pnl / starting_balance)
    else:
        compounding_mult = 1.0  # No boost on losses
    
    # Final size
    final_size = min(
        base_size,
        max_by_capital,
        available
    )
    
    # Apply compounding multiplier (gradual increase)
    if compounding_mult > 1.0:
        # Only boost if still within constraints
        boosted_size = final_size * min(compounding_mult, 1.15)  # Cap at 15% boost
        final_size = min(boosted_size, available)
    
    return max(final_size, MIN_POSITION_USDT)  # Floor at minimum
```

---

## 3. Code Changes Required

### 3.1 core/meta_controller.py Changes

#### Addition: Position Tier Tracking

```python
# Add to MetaController.__init__()
self.position_tiers: Dict[str, PositionTier] = {}  # symbol -> tier
self.tier2_allocations: Dict[str, float] = {}      # symbol -> allocated $
self.tier1_allocations: Dict[str, float] = {}      # symbol -> allocated $
```

#### Modification: Entry Decision

```python
# CURRENT (Line ~1070):
pos_ok = self._regime_check_max_positions()
if not pos_ok:
    reason = f"Max positions reached for regime"

# PROPOSED:
pos_ok, tier_reason = await self._tier_aware_position_check(signal_symbol)
if not pos_ok:
    reason = tier_reason  # More specific reason
```

#### Addition: Position Classification Method

```python
async def _classify_position_tier(self, symbol: str, qty: float) -> PositionTier:
    """Classify position into appropriate tier"""
    # Implement logic from 2.1 above
    pass
```

#### Modification: Entry Size Calculation

```python
# CURRENT (Line ~920):
entry_size = self._cfg("TRADE_AMOUNT_USDT", 25.0)

# PROPOSED:
entry_size = await self.calculate_new_position_size(
    symbol=signal_symbol,
    config_entry_size=self._cfg("TRADE_AMOUNT_USDT", 25.0),
    regime=self.current_regime
)
```

### 3.2 core/shared_state.py Changes

#### Addition: Tier Tracking in Position State

```python
# Add to PositionRecord dataclass (or equivalent)
@dataclass
class PositionRecord:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    tier: PositionTier  # NEW
    is_primary_lock: bool  # NEW
    can_scale: bool  # NEW (for Tier2)
```

#### Addition: Tier Update on Position Change

```python
async def update_position_tier(self, symbol: str, new_qty: float) -> None:
    """Update position tier when quantity changes"""
    tier = await classify_position(symbol, new_qty)
    self.position_tiers[symbol] = tier
    self.logger.info(f"[Tier] {symbol} reclassified to {tier.value}")
```

### 3.3 core/balance_manager.py Changes

#### Modification: Tier-Aware Allocation

```python
# CURRENT:
available = total_balance - allocated

# PROPOSED:
tier1_allocated = sum(tier1_allocations.values())
tier2_allocated = sum(tier2_allocations.values())
dust_value = compute_dust_total()
available = total_balance - tier1_allocated - tier2_allocated - dust_value
```

#### Addition: Tier Slot Management

```python
def get_tier1_slots_available(self, regime: str) -> int:
    """How many Tier1 slots remain?"""
    max_slots = self._get_regime_max_positions(regime)
    used_slots = len([p for p in positions if p.tier == TIER1])
    return max(0, max_slots - used_slots)

def get_tier2_slots_available(self) -> int:
    """How many Tier2 slots remain?"""
    max_tier2 = self._cfg("MAX_TIER2_CONCURRENT", 2)
    used_tier2 = len([p for p in positions if p.tier == TIER2])
    return max(0, max_tier2 - used_tier2)
```

---

## 4. Behavioral Changes

### 4.1 Entry Behavior

**Before (Restrictive):**
```
New signal for BTC/USDT arrives
├─ Check: Is there ANY BTC position? YES (0.0005 BTC = $15)
├─ Decision: BLOCKED
└─ Result: Signal ignored, capital idle
```

**After (Tiered):**
```
New signal for BTC/USDT arrives
├─ Check: What tier is existing position?
│  └─ Classify: 0.0005 BTC @ $2,500 = $12.50 → TIER2
├─ Decision: Check Tier2 slots (2 available, 0 used)
├─ Approve: Can open as secondary Tier2 or new Tier1
└─ Result: Signal executed, capital deployed
```

### 4.2 Exit Behavior

**Before (Passive):**
```
Position in BTC/USDT @ $12.50 (Tier2)
├─ Status: Running for 4+ hours
├─ Decision: Might exit, might not
└─ Result: Capital potentially locked long-term
```

**After (Time-Enforced):**
```
Position in BTC/USDT @ $12.50 (Tier2)
├─ Check 1: TP signal? → Close if triggered
├─ Check 2: SL signal? → Close if triggered
├─ Check 3: Age > 4 hours? → Force close
├─ Check 4: Trend reversal? → Close
└─ Result: Capital released predictably
```

### 4.3 Capital Utilization

**Before (Frozen):**
```
$103.89 total
├─ Deployed: $25 (24%)
├─ Trapped: $68 (dust)
├─ Available: $10.89 (insufficient for new entry)
└─ Utilization: 24%
```

**After (Efficient):**
```
$130 total (after liquidations)
├─ Tier1: $50 (2 positions × $25)
├─ Tier2: $10 (2 positions × $5)
├─ Dust: $5 (consolidating)
├─ Available: $65 (ready for next entry)
└─ Utilization: 80%
```

---

## 5. Risk Controls Built-In

### 5.1 Tier1 Position Lock

```python
# Tier1 positions CANNOT be:
- Closed prematurely (unless TP/SL/manual)
- Averaged down (no additional buys)
- Scaled up (size is fixed at entry)

# Enforced by:
- Entry check: Returns True if Tier1 exists
- Exit check: Allows only TP/SL/time-based exit
```

### 5.2 Tier2 Position Scaling

```python
# Tier2 positions CAN be:
- Run in parallel with Tier1
- Smaller size (20-50% of Tier1)
- Auto-closed on time limit

# Constrained by:
- Max Tier2 concurrent: 2 positions
- Size cap: $5-10 per position
- Duration: Same as Tier1 (2-4 hours)
```

### 5.3 Dust Auto-Liquidation

```python
# Dust positions WILL:
- Auto-consolidate when < $1
- Never block new entries
- Liquidate when 28 signals fire
- Be retried if initial attempt fails

# Prevented by:
- Exchange min notional check
- Capital requirement for fees
- Liquidation agent governance
```

---

## 6. Backward Compatibility

### 6.1 Configuration File Compatibility

```
# Old .env files still work
DEFAULT_PLANNED_QUOTE=25
MIN_TRADE_QUOTE=25
# ... (all existing parameters)

# New optional parameters (with defaults)
TIER2_MAX_CONCURRENT=2           # (default: 2)
TIER2_ENTRY_SIZE_RATIO=0.5       # (default: 50% of tier1)
POSITION_AGE_TIER2_THRESHOLD=240 # (default: 240 min = 4h)
DUST_FLOOR_USDT=1.0              # (default: $1)
```

### 6.2 Logging Integration

```
# New log messages preserve existing format
[TIER:Classify] BTC qty=0.0005 price=2500 value=$12.50 → TIER2

# Existing log filters still work
grep "BLOCKED" logs  # Still catches position blocks
grep "entry_attempted" logs  # Still catches attempts
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# Test: Tier classification
async def test_tier_classification():
    assert classify_position(qty=0.0002, price=2500) == TIER3  # $0.50
    assert classify_position(qty=0.005, price=2500) == TIER2   # $12.50
    assert classify_position(qty=0.02, price=2500) == TIER1    # $50

# Test: Entry blocking
async def test_entry_blocking():
    existing = PositionRecord(qty=0.005, tier=TIER2)
    assert position_blocks(existing) == False  # TIER2 doesn't block
    
    existing = PositionRecord(qty=0.02, tier=TIER1)
    assert position_blocks(existing) == True   # TIER1 blocks

# Test: Capital allocation
async def test_capital_allocation():
    available = 100
    regime = "STANDARD"  # Max 2 positions
    size = calculate_position_size(available, regime)
    assert size <= 50  # Half available
```

### 7.2 Integration Tests

```python
# Test: Full entry/exit cycle
async def test_tier1_entry_to_exit():
    # Enter Tier1 position
    await execute_signal(symbol="BTC", tier=TIER1)
    assert count_tier1() == 1
    assert available_capital < 100
    
    # Try to enter second Tier1 (should be blocked)
    ok = await can_enter_new_symbol("ETH")
    assert ok == True  # Different symbol, should allow
    
    # Exit Tier1 position
    await exit_on_tp_signal()
    assert count_tier1() == 0
    assert available_capital > previous_available

# Test: Tier2 parallel execution
async def test_tier2_parallel():
    # Enter Tier1
    await execute_signal(symbol="BTC", tier=TIER1)
    
    # Enter Tier2 in different symbol
    await execute_signal(symbol="ETH", tier=TIER2)
    
    # Both running simultaneously
    assert count_positions() == 2
    
    # Exit Tier2 (doesn't affect Tier1)
    await exit_on_sl_signal("ETH")
    assert count_tier2() == 0
    assert count_tier1() == 1  # Tier1 still running
```

### 7.3 Stress Tests

```python
# Test: Capital fragmentation
async def test_capital_fragmentation():
    for i in range(10):
        signal = await get_next_signal()
        await execute_signal(signal)
    
    # Verify no deadlock
    assert count_positions() <= regime_max + tier2_max
    assert available_capital > MIN_OPERATIONAL_RESERVE
    
    # Force liquidate dust
    dust_value = await liquidate_all_dust()
    assert available_capital >= previous + dust_value

# Test: Continuous trading
async def test_continuous_trading_48h():
    """Run system for 48 hours, verify health"""
    start_time = now()
    while now() - start_time < 48 * 3600:
        await process_signals()
        await manage_exits()
        await consolidate_dust()
        await log_metrics()
    
    assert error_count == 0
    assert capital_recovered > starting_capital
```

---

## 8. Rollout Plan

### 8.1 Phase 1: Code Review & Testing (Day 1)
- [ ] Code changes reviewed
- [ ] Unit tests passing (100%)
- [ ] Integration tests passing
- [ ] Stress tests completed
- [ ] Documentation approved

### 8.2 Phase 2: Staged Deployment (Day 2)
- [ ] Deploy to test environment
- [ ] Run 2-hour smoke test
- [ ] Monitor logs for warnings
- [ ] Verify tier classification working
- [ ] Test entry/exit flow

### 8.3 Phase 3: Live Deployment (Day 2-3)
- [ ] Deploy to production
- [ ] Reduce entry size to $5 first
- [ ] Monitor liquidation execution
- [ ] Track position classifications
- [ ] Verify capital freed ($50-80)
- [ ] Monitor profitability over 8+ hours

### 8.4 Phase 4: Validation & Scaling (Day 3+)
- [ ] Confirm > 50% win rate
- [ ] Scale to $10 entry size
- [ ] Enable Tier2 positions (2-4 concurrent)
- [ ] Monitor account growth trajectory
- [ ] Proceed to compounding phase

---

## 9. Success Metrics

```
METRIC 1: Position Tier Classification
├─ Target: 100% of positions correctly classified
├─ Measure: Log parsing (TIER1/TIER2/TIER3/UNHEALABLE count)
├─ Success: < 1% misclassification

METRIC 2: Entry Success Rate
├─ Target: 90%+ of signals result in executed trades
├─ Measure: signals_generated vs exec_attempted
├─ Success: From current 2% → 90%+

METRIC 3: Capital Utilization
├─ Target: 80-95% of capital deployed
├─ Measure: (allocated / total) × 100
├─ Success: From current 24% → 80%+

METRIC 4: Position Duration (Time-to-Exit)
├─ Target: Average position 2-4 hours
├─ Measure: Average (exit_time - entry_time)
├─ Success: Predictable, no indefinite holds

METRIC 5: Dust Ratio
├─ Target: < 10% of capital in dust
├─ Measure: dust_value / total_balance
├─ Success: From current 96.8% → < 10%

METRIC 6: Compounding Frequency
├─ Target: 8+ cycles per month
├─ Measure: Number of profit-reinvestment events
├─ Success: Accelerating account growth

METRIC 7: Account Recovery
├─ Target: $103 → $500+ in 48 hours
├─ Measure: Total balance over time
├─ Success: Exponential recovery curve
```

