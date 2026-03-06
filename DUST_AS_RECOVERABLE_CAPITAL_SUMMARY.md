# 🔄 How Dust is Treated as Recoverable Capital

## Executive Summary

Your system treats **Dust** (small positions below minimum order size) as **recoverable capital** rather than lost capital. Dust positions are systematically monitored, tracked, and recovered through multiple coordinated mechanisms. This is a sophisticated capital recovery strategy that prevents permanent capital loss and enables the system to escape capital floor crises.

---

## Part 1: What is Dust?

### Classification Tiers

Your system defines dust in **3 tiers** based on position notional (USD value):

| **Tier** | **Name** | **Range** | **Treatment** | **Examples** |
|----------|----------|-----------|---------------|--------------|
| **Tier 1** | Economic Dust | $10 - $25 | Tracked for recovery, counted in health metrics | Position worth $12 due to price movement |
| **Tier 2** | Significant Dust | $1 - $10 | Recoverable but requires more aggressive tactics | Position worth $5 after loss |
| **Tier 3** | Permanent Dust | < $1.0 | Terminal residual, excluded from all recovery efforts | Position worth $0.50, written off |

### How Dust is Created

Positions become dust through:
1. **Market Losses**: Price drop reduces position notional below min_notional
2. **Incomplete Fills**: Partial order execution leaves remainder below minNotional
3. **Accumulation Rejection**: Buy intended for minNotional succeeds only partially
4. **Trade Slippage**: Position value drops below threshold after execution

---

## Part 2: Capital Recovery Mechanisms

### Mechanism 1: Dust Promotion (P0 Scaling)

**When triggered**: Capital floor breach, system in crisis mode

**How it works**:
```python
P0_DUST_PROMOTION (Escape Hatch #1):
├─ Condition 1: Must have dust positions
├─ Condition 2: Must have high-confidence BUY signals (confidence ≥ 0.55)
└─ Condition 3: Dust + signal intersection must exist

Action: Scale dust positions upward using freed capital
Result: Dust becomes viable position → capital recovery achieved
```

**Example**: 
- Dust position: BTCUSDT, quantity=0.00005 BTC, value=$2.00
- Strong BUY signal detected (confidence=0.85)
- System adds $25 to scale position → value becomes $27.00
- Position graduates from dust status → now tradeable
- Capital locked in dust is now "free" for other operations

### Mechanism 2: Accumulation Resolution

**When triggered**: Rejected trades accumulate intended quote amounts

**How it works**:
```
Trade Rejected (minNotional not met)
    ↓
Quote amount accumulates across rejections
    ↓
Monitor threshold crossing
    ↓
When accumulated ≥ minNotional:
    ├─ Auto-emit BUY with full accumulated quote
    ├─ Transform dust → viable position
    └─ Capital recovery via growth
```

**Example**:
- Attempt 1: BUY for $8 → rejected (minNotional=$10)
- Attempt 2: BUY for $7 → accumulated=$15
- Threshold crossed! → Auto-emit BUY for $15
- Dust grows → becomes viable position

### Mechanism 3: Dust Monitoring & Health Tracking

**Location**: `core/dust_monitor.py`

**What it tracks**:
```python
For each dust position:
├─ Notional value (USD)
├─ Age (hours since created)
├─ Health status (HEALTHY/STALLED/CRITICAL)
├─ Recovery rate (% expected to recover)
├─ Rejection count (failed exit attempts)
└─ Skip count (times skipped by invariant)
```

**Health Classification**:
- **HEALTHY** (< 4 hours old): Accumulating normally, recovery likely
- **STALLED** (4-8 hours old): Not accumulating, needs intervention
- **CRITICAL** (> 8 hours old): Below minimum for extended period, consider liquidation

**Recovery Metrics Calculated**:
```python
capital_recoverable = sum(dust_notional for all HEALTHY dust)
recovery_rate = (recovered_dust_positions / total_dust_positions)
capital_trapped = total_dust_notional  # Locked until promoted/recovered
```

### Mechanism 4: Bootstrap Dust Scale Bypass

**When triggered**: First-time dust position recovery during bootstrap

**How it works**:
```python
BOOTSTRAP_DUST_SCALE_BYPASS:
├─ Enabled on FIRST dust recovery per symbol (one-time use)
├─ Allows scaling dust below minNotional temporarily
├─ Purpose: Get dust to viable accumulation threshold
├─ Circuit breaker: Prevents infinite dust recycling
└─ Tracked in: _consolidated_dust_symbols or symbol_dust_state
```

---

## Part 3: Capital Calculations

### How Dust Affects Available Capital

**Capital Ledger**:
```python
Total Capital = Free USDT + Occupied Positions + DUST_POSITIONS

Free USDT Calculation:
├─ Start: USDT Balance
├─ Subtract: Occupied in Non-Dust Positions
├─ Subtract: Occupied in Dust Positions (LOCKED)
└─ Result: Free USDT available for new trades

Example (NAV = $100):
├─ USDT Balance: $100
├─ Position A (Non-Dust): $25 (locked in active trade)
├─ Position B (Dust): $5 (locked, recoverable)
├─ Position C (Dust): $2 (locked, recoverable)
├─ Free USDT: $68
└─ Recoverable Capital: $7 (if dust recovered)
```

### Capital Floor Check with Dust

**Capital Floor Formula**:
```python
CAPITAL_FLOOR = max(
    MIN_CAPITAL_USDT,           # e.g., $10
    NAV * CAPITAL_FLOOR_RATIO   # e.g., NAV * 0.20
)

# Example:
NAV = $100
Floor = max(10, 100 * 0.20) = $20

For Floor Check:
├─ Total Equity = $100
├─ Free Capital = $68
├─ Capital Status: OK (equity > floor)
└─ Recovery Options: Available (dust can be promoted)

# But when breached:
NAV = $50 (after losses)
Floor = max(10, 50 * 0.20) = $10
├─ Total Equity = $50 (below $100 floor)
├─ Status: CRITICAL
└─ Action: Trigger P0_DUST_PROMOTION escape hatch
```

### Dust Contribution to Capital Recovery

**When Dust is Promoted**:
```
Before Promotion:
├─ Total Capital: $50
├─ Free Capital: $43
├─ Dust Positions: $7 (LOCKED)
├─ Floor: $10
└─ Status: BREACHED (floor $10 > free $43)

After P0 Dust Promotion:
├─ Scaled dust position from $7 → $32
├─ Used freed capital: $25
├─ New Free Capital: $18
├─ New Position Value: $32 (now viable!)
├─ Total Capital: $50 (same)
└─ Status: CRITICAL but RECOVERABLE

Capital Freed for Other Operations:
├─ Previous: Locked $7 in dust
├─ Now: Locked $32 in viable position
├─ Result: Capital redistributed, not created
```

---

## Part 4: Dust Recovery Lifecycle

### State Machine

```
ACTIVE (Normal Trade)
    ↓
    [Price falls below minNotional]
    ↓
DUST_LOCKED (Below minimum, cannot sell)
    ├─ Health = HEALTHY (< 4h old)
    │   ├─ Option A: P0 Scaling (if strong signals)
    │   └─ Option B: Accumulation (grow with additional buys)
    ├─ Health = STALLED (4-8h old)
    │   ├─ Flag for monitoring
    │   └─ Consider intervention
    └─ Health = CRITICAL (> 8h old)
        └─ Alert for potential liquidation

[From any state with viable recovery path]
    ↓
ACCUMULATION (Growing via additional buys)
    ├─ Track rejected trade accumulation
    ├─ Monitor threshold crossing
    └─ Auto-emit when minNotional reached

[Successful recovery]
    ↓
PROMOTION (Scaled to viability)
    ├─ Added sufficient capital
    ├─ Position now > minNotional
    └─ Graduates from dust tracking

[Or]
    ↓
LIQUIDATION (Forced exit)
    ├─ Position too old/stalled
    ├─ Sweep at market price
    └─ Capital recovered (minus slippage)

[Or]
    ↓
TERMINAL (Abandoned)
    ├─ Below $1.0 threshold
    ├─ Permanent dust (written off)
    └─ Excluded from recovery metrics
```

---

## Part 5: Why Dust Matters for Capital Management

### The Dust Paradox

**Without Dust Recovery**:
```
Capital = $100
Trade A: Partial fill (only bought $8 when intended $10)
Result: Stuck with dust → Capital unavailable → Effective capital = $92
```

**With Dust Recovery** (your system):
```
Capital = $100
Trade A: Partial fill (bought $8, dust created)
Accumulation Resolution: Next trade adds $2 more → Position reaches $10
Result: Dust becomes viable → Capital effective = $100
```

### Avoiding Capital Death Spirals

Your system prevents this scenario:

```
❌ Without dust recovery:
Day 1: Capital $100, Trade 1: Lose $20 → $80 remaining
       But $7 stuck in dust → effective capital $73
Day 2: Capital $80 (includes dust), Can only trade with $73
       Trade 2: Lose $10 → $70 remaining
       More dust created → effective capital drops to $60
Day 3: Capital $70, effective $50 due to dust
       Trading becomes impossible
       DEATH SPIRAL

✅ With dust recovery (your system):
Day 1: Capital $100, Trade 1: Lose $20 → $80 remaining
       Dust created: $7
       P0 Promotion triggered: Scale dust → $32 (viable)
       Capital not lost, redistributed
Day 2: Capital $80, effective $80 (dust recovered)
       Trade 2: Lose $10 → $70 remaining
       Recovery mechanisms available
       SYSTEM REMAINS VIABLE
```

---

## Part 6: Configuration Parameters

### Dust Thresholds

```python
# In config files:

MIN_NOTIONAL_USDT = 10.0           # Exchange minimum
PERMANENT_DUST_USDT_THRESHOLD = 1.0 # Below this = terminal
DUST_POSITION_THRESHOLD = 25.0      # Below this = trackable dust

# Dust health timeouts
DUST_STALL_THRESHOLD_HOURS = 4.0    # When position stalls
DUST_CRITICAL_THRESHOLD_HOURS = 8.0 # When critical alerts

# Dust promotion constraints
DUST_PROMOTION_MIN_QUOTE = 25.0     # Minimum capital to promote
DUST_OVERRIDE_THRESHOLD = 0.70      # Allow if 70% of portfolio is dust
```

### Escape Hatch Configuration

```python
P0_DUST_PROMOTION = True            # Enable dust scaling
ACCUMULATION_PROMOTION = True       # Enable accumulation resolution
DUST_OVERRIDE_ENABLED = True        # Allow dust to override position limit
```

---

## Part 7: How It All Integrates

### Capital Floor Check Flow

```python
async def _check_capital_floor_central(self) -> bool:
    """Check if capital floor is OK."""
    
    # 1. Get current state
    nav = await self._get_nav()
    free_capital = await self._get_free_usdt()
    floor = self._calculate_capital_floor(nav)
    
    # 2. Check against floor
    if free_capital >= floor:
        return True  # ✅ HEALTHY
    
    # 3. Floor breached - try escape hatches
    # Escape Hatch #1: P0 Dust Promotion
    if await self._check_p0_dust_promotion():
        return True  # ✅ Can execute P0, bypass check
    
    # Escape Hatch #2: Accumulation Promotion
    if await self._can_accumulation_promotion_help():
        return True  # ✅ Can grow dust, bypass check
    
    # 4. Both escapes failed - hard block
    return False  # ❌ CRITICAL: No recovery available
```

### Dust Health Monitoring

```python
# Every cycle, DustMonitor tracks:

health = await dust_monitor.check_dust_health()

# Returns:
{
    "total_dust_positions": 3,           # How many dust positions
    "total_dust_notional": 12.50,        # Total USD locked in dust
    "positions_healthy": 2,              # Ready for recovery
    "positions_stalled": 1,              # Needs intervention
    "positions_critical": 0,             # Abandoned
    "capital_trapped": 12.50,            # USD locked
    "capital_recoverable": 10.00,        # Expected to recover
    "recovery_rate": 0.78,               # % that will recover
    "system_health": "NORMAL"            # Overall assessment
}
```

---

## Part 8: Key Insights

### 1. Dust is Capital, Not Loss

```
Dust ≠ Permanent Loss
Dust = Temporary Capital Reallocation
```

Your system tracks every dust position and measures its recovery potential. Each dollar in dust is treated as a potential recovery opportunity, not a sunk cost.

### 2. Multi-Layer Recovery

Your system doesn't rely on a single recovery mechanism:
- **Layer 1** (Accumulation): Grow dust through additional purchases
- **Layer 2** (P0 Promotion): Scale dust with freed capital
- **Layer 3** (Bootstrap Bypass): One-time scaling assist per symbol
- **Layer 4** (Liquidation): Last resort to recover any residual value

### 3. Capital Efficiency

By treating dust as recoverable:
```
Total Effective Capital = Free USDT + Recoverable Dust + Positions
Whereas without it:     = Free USDT + (Lost Dust) + Positions
                        (Lower effective capital)
```

This allows the system to:
- Maintain higher trading capacity
- Avoid starvation faster
- Escape capital floor crises
- Survive longer drawdowns

### 4. Health-Based Decision Making

Your system makes recovery decisions based on dust health:
- **HEALTHY** (young): Try accumulation/promotion
- **STALLED** (aging): Monitor closely, consider intervention
- **CRITICAL** (old): Prepare for liquidation or write-off
- **TERMINAL** (< $1): Exclude from calculations, ignore

---

## Part 9: Limitations & Safeguards

### When Dust Recovery Might Fail

```
Scenario 1: Bootstrap with No Dust
├─ Problem: No existing dust to promote
├─ Trigger: Capital floor breach immediately
└─ Solution: THROUGHPUT_GRANT mechanism (separate)

Scenario 2: No Strong Buy Signals
├─ Problem: P0 requires high-confidence signals
├─ Trigger: Capital low, but no buying opportunity
└─ Solution: Accumulation resolution for ongoing positions

Scenario 3: Excessive Dust Accumulation
├─ Problem: Too many small positions
├─ Trigger: Portfolio limit reached
└─ Solution: DUST_OVERRIDE allows promotion despite position limit

Scenario 4: Permanent Dust Below $1
├─ Problem: Dust too small to recover
├─ Trigger: Position degraded to terminal level
└─ Solution: Exclude from metrics, write off as loss
```

### Safeguards Built In

1. **Circuit Breaker**: Don't repeatedly heal same dust
2. **Age Limits**: Critical dust abandoned after max age
3. **Minimum Viable Promotion**: Only promote if $25+ freed capital available
4. **Permanent Dust Threshold**: Below $1.0 excluded from recovery
5. **Dust Inflation Prevention**: Don't promote unless position will be viable
6. **Symbol-Scoped Bypass Tracking**: One bypass per symbol per bootstrap

---

## Part 10: Monitoring & Observability

### Key Metrics to Watch

```python
# From DustMonitor stats:

1. capital_trapped: How much USD is locked in dust positions
   - Normal: < 5% of NAV
   - Warning: 5-20% of NAV
   - Critical: > 20% of NAV

2. capital_recoverable: How much dust can be recovered
   - Should be > 60% of capital_trapped
   - Indicates health of dust positions

3. recovery_rate: % of dust that successfully recovers
   - Healthy: > 70%
   - Marginal: 40-70%
   - Unhealthy: < 40%

4. average_position_age_hours: How long dust positions exist
   - Healthy: < 2 hours (quick recovery)
   - Marginal: 2-6 hours
   - Critical: > 6 hours (needs intervention)

5. positions_critical / total_dust_positions:
   - Healthy: 0%
   - Warning: > 10%
   - Critical: > 25%
```

### Example Monitoring Dashboard

```
📊 DUST MONITOR STATUS
┌────────────────────────────────────────────┐
│ Total Positions:         3                 │
│ Status: HEALTHY → STALLED → CRITICAL       │
│                  (1)      (1)      (1)     │
├────────────────────────────────────────────┤
│ Capital Trapped:      $12.50               │
│ Capital Recoverable:  $10.00 (80%)         │
│ Recovery Rate:        78%                  │
├────────────────────────────────────────────┤
│ Avg Position Age:     3.2 hours            │
│ Oldest Position:      8.5 hours (CRITICAL) │
├────────────────────────────────────────────┤
│ System Health:        NORMAL ✅            │
└────────────────────────────────────────────┘
```

---

## Summary

Your system treats **Dust as Recoverable Capital** through:

1. ✅ **Tracking**: DustMonitor watches every dust position for recovery opportunities
2. ✅ **Classification**: Health-based categorization (HEALTHY/STALLED/CRITICAL)
3. ✅ **Promotion**: P0 Scaling upgrades dust with freed capital
4. ✅ **Accumulation**: Rejected trades grow dust toward minNotional
5. ✅ **Bootstrap Override**: One-time scaling assist per symbol
6. ✅ **Capital Efficiency**: Prevents death spirals and starvation
7. ✅ **Safeguards**: Circuit breakers, age limits, and permanent dust write-offs
8. ✅ **Observability**: Comprehensive metrics for monitoring recovery health

This sophisticated approach means **every dollar trapped in dust is treated as a potential recovery opportunity, not a permanent loss**, enabling the system to survive longer drawdowns and escape capital floor crises more effectively.

---

## ⚠️ CRITICAL OPERATIONAL RULE (MUST ENFORCE)

For dust recovery to work, three invariants **MUST be enforced**:

### Invariant 1: Dust Must NOT Block BUY Signals
```
✓ CORRECT: Dust position + strong BUY signal → Allow entry (promote dust)
✗ WRONG:   Dust position + strong BUY signal → Reject entry (deadlock)
```

### Invariant 2: Dust Must NOT Count Toward Position Limits
```
✓ CORRECT: Position limit = significant positions only
           Dust positions excluded from count
✗ WRONG:   Position limit = all positions including dust
           Dust fills the limit, blocks new entries
```

### Invariant 3: Dust Must Be REUSABLE When Signal Appears
```
✓ CORRECT: Dust $5 + BUY signal → Merge with new capital
           Dust $5 + capital $25 → Position $30 (viable)
✗ WRONG:   Dust $5 + BUY signal → Rejected (blocked)
           Dust never reused, never recovers
```

**If any of these are violated → System deadlocks and dust never recovers.**

---

## Status: Critical Bug Found

**File**: `core/meta_controller.py`, lines 9902-9930  
**Issue**: Dust IS currently blocking BUY signals (violates Invariant #1)  
**Impact**: P0 Dust Promotion cannot execute, dust becomes permanent prison  
**Solution**: See `🔧_DUST_BLOCKING_FIX_IMPLEMENTATION.md`

**Root Cause**: 
- Method `_position_blocks_new_buy()` exists with correct dust-aware logic
- But it's NOT called at the decision gate
- Instead, crude `if existing_qty > 0:` check is used (treats dust same as viable)

**Fix**: Replace crude check with dust-aware method call (see implementation doc)
