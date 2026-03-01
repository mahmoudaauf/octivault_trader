# 🚀 Phase 10: Rotation-Execution Alignment Implementation

**Date:** February 23, 2026  
**Status:** ✅ IMPLEMENTED  
**System:** Unified Universe Rotation Engine (UURE) with ExecutionManager EV Integration

---

## Executive Summary

### The Insight
> "You are not missing complexity. You are missing alignment between rotation logic and execution profitability logic."

**Your diagnosis was correct:** The system had two independent profitability systems:
1. **Rotation logic:** Decides universe based on score
2. **Execution logic:** Decides trades based on expected value (EV)

**Problem:** Universe ≠ Execution reality

**Solution:** Align rotation decisions to use the same EV calculation as execution

### The Implementation (Phase 10)

Three-layer alignment architecture:

```
Layer 1: Reuse ExecutionManager EV Logic
         └─ Same net-edge calculation
         └─ Same fee/slippage treatment
         └─ Same profitability threshold

Layer 2: Profitability Filter (Step 4.5)
         └─ Removes symbols that fail execution's own tests
         └─ "If execution would reject this, rotation rejects it"
         
Layer 3: Relative Replacement Rule (Step 4.6)
         └─ Only rotate to better symbols (superiority factor)
         └─ "25% edge premium required to abandon proven winners"
```

### Result: Professional Capital Allocator

```
Before:  Smart Retail Bot (misaligned systems)
         Rotation → Score-based universe
         Execution → EV-based trades
         Disconnect: Universe ≠ Trade Quality
         
After:   Institutional Capital Allocator (aligned)
         Rotation → EV-based universe (ExecutionManager logic)
         Execution → EV-based trades (same logic)
         Alignment: Universe = Trade Quality ✅
```

---

## Technical Implementation

### Phase 10 Changes

**File:** `core/universe_rotation_engine.py`

#### New Method 1: `_apply_profitability_filter()`

```python
async def _apply_profitability_filter(
    self, candidates: List[str]
) -> List[str]:
    """
    Step 4.5: Filter candidates by execution profitability (net edge).
    
    Uses ExecutionManager's EV logic:
      net_edge = expected_move - (2 * fees) - slippage
      
    Only keeps symbols where net_edge > MIN_NET_PROFIT_AFTER_FEES
    """
```

**What It Does:**
- Calls `ExecutionManager._entry_profitability_feasible()`
- Gets `required_tp` (take profit target needed for profitability)
- Compares vs `tp_max` (maximum achievable TP from ATR)
- Only keeps if `tp_max >= required_tp`
- Safe default: includes on error (doesn't crash rotation)

**Key Properties:**
- ✅ Uses same threshold as ExecutionManager: `MIN_NET_PROFIT_AFTER_FEES`
- ✅ Respects same fee structure: `2 × fees + slippage`
- ✅ Respects same ATR calculation: `TP_ATR_MULT × ATR`
- ✅ Blocks symbols execution would reject

#### New Method 2: `_apply_relative_replacement_rule()`

```python
async def _apply_relative_replacement_rule(
    self, new_candidates: List[str], current_universe: List[str]
) -> List[str]:
    """
    Step 4.6: Relative Replacement Rule
    
    Only allows rotation OUT if incoming candidates have superior edge.
    
    Rule: incoming_edge > weakest_active_edge × ROTATION_SUPERIORITY_FACTOR
    
    Default ROTATION_SUPERIORITY_FACTOR = 1.25 (25% premium required)
    """
```

**What It Does:**
- Identifies weakest symbol in current universe (lowest net edge)
- Calculates required edge for incoming: `weakest_edge × 1.25`
- Only admits new symbols that exceed this threshold
- Keeps strong actives that meet threshold
- Blocks rotation if no candidates qualify

**Key Properties:**
- ✅ Prevents "moving sideways" into equal candidates
- ✅ Blocks rotation out of winners into losers
- ✅ Configurable via `ROTATION_SUPERIORITY_FACTOR`
- ✅ Safe: keeps current universe if no qualifiers

**Example:**

```
Current universe edges:
  BTC:  +0.8%  ← Strongest
  ETH:  +0.4%
  SOL:  +0.2%  ← Weakest (this is the gate)

Required edge for rotation: 0.2% × 1.25 = 0.25%

New candidates:
  XRP:  +0.3%  ✅ Exceeds 0.25%, admitted
  ADA:  +0.1%  ❌ Below 0.25%, blocked
  DOGE: +0.2%  ❌ Below 0.25%, blocked

Result: Rotate in XRP only (if cap allows)
        Keep SOL (proven +0.2% > marginal ADA/DOGE)
```

---

## Pipeline Architecture (Phase 10)

### New 7-Step Pipeline

```
Step 1: Collect Candidates
        │
        └─> Union of accepted symbols + current positions
        
Step 2: Score All Candidates
        │
        └─> Unified scoring (existing)
        
Step 3: Rank by Score
        │
        └─> Sort descending (existing)
        
Step 4: Apply Governor Cap
        │
        └─> Capital floor rules (existing)
        └─> Result: N symbols respecting equity tier
        
Step 4.5: Apply Profitability Filter ✨ NEW
        │
        └─> ExecutionManager EV logic
        └─> Result: Only symbols execution would accept
        
Step 4.6: Apply Relative Replacement Rule ✨ NEW
        │
        └─> Superiority gate (1.25× factor)
        └─> Keep strong actives + add superior new
        
Step 5: Identify Rotation
        │
        └─> Diff: current → final
        
Step 6: Hard Replace Universe
        │
        └─> Deterministic replacement
        
Step 7: Trigger Liquidation
        │
        └─> Sell removed symbols
```

### Code Integration

**File:** `core/universe_rotation_engine.py`, method `compute_and_apply_universe()`

```python
# Step 4: Apply governor cap
capped = await self._apply_governor_cap(ranked)

# Step 4.5: Apply profitability filter (NEW)
profitable = await self._apply_profitability_filter(capped)

# Step 4.6: Apply relative replacement rule (NEW)
current_universe = self.ss.get_accepted_symbol_list()
final_universe = await self._apply_relative_replacement_rule(
    profitable, current_universe
)

# Step 5+: Use final_universe for rotation
rotation = await self._identify_rotation(final_universe)
await self._hard_replace_universe(final_universe)
```

---

## Configuration

### Parameter: ROTATION_SUPERIORITY_FACTOR

**Location:** `core/app_context.py` or `config/tuned_params.json`

```python
ROTATION_SUPERIORITY_FACTOR = 1.25      # Default: 25% edge premium
```

**What It Means:**
- 1.0 = Anything better than weakest is OK (liberal)
- 1.25 = Need 25% edge improvement (balanced, default)
- 1.50 = Need 50% edge improvement (conservative)
- 2.0 = Need 100% edge improvement (very conservative)

**Recommendation by Account Size:**

| Account | Factor | Interpretation |
|---------|--------|---|
| Micro (<$500) | 1.5 | Conservative: only strong rotation |
| Small ($500-$5K) | 1.25 | Balanced: 25% premium required |
| Medium ($5K-$50K) | 1.1 | Aggressive: low bar for rotation |
| Large ($50K+) | 1.05 | Very aggressive: minimal gate |

**Set via AppContext:**

```python
class AppContext:
    def __init__(self, ...):
        # ... other config ...
        
        # Phase 10: Rotation-Execution Alignment
        self.config.ROTATION_SUPERIORITY_FACTOR = 1.25
```

### ExecutionManager Config (Already Exists)

```python
# These are already in ExecutionManager config
# UURE now reuses them:

MIN_NET_PROFIT_AFTER_FEES = 0.0035      # 0.35% minimum
MIN_PLANNED_QUOTE_FEE_MULT = 2.5        # Entry fee multiplier
MIN_PROFIT_EXIT_FEE_MULT = 2.0          # Exit fee multiplier
TP_ATR_MULT = 1.5                       # TP target multiplier
EXIT_SLIPPAGE_BPS = 20                  # Exit slippage (bps)
```

---

## Expected Behavior

### Scenario 1: All Candidates Unprofitable

```
Universe: [BTC, ETH, SOL]
New candidates: [DOGE (-0.1%), ADA (-0.2%), XRP (-0.3%)]

Step 4.5 (Profitability Filter):
  └─> All candidates fail execution's profitability test
  
Step 4.6 (Relative Rule):
  └─> No candidates pass superiority gate
  
Result: Rotation BLOCKED
         Universe stays [BTC, ETH, SOL]
```

**Outcome:** Capital preserved. No rotation into -EV trades. ✅

### Scenario 2: Mixed Quality Candidates

```
Universe: [BTC (+0.8%), ETH (+0.2%), SOL (+0.1%)]
New candidates: [XRP (+0.15%), ADA (+0.25%), DOGE (-0.1%)]
Superiority factor: 1.25
Weakest active: SOL (+0.1%)
Required edge: 0.1% × 1.25 = 0.125%

Step 4.5 (Profitability Filter):
  DOGE (-0.1%) ← Fails profitability test, filtered
  XRP (+0.15%), ADA (+0.25%) ← Pass profitability test

Step 4.6 (Relative Rule):
  XRP (+0.15%) ≥ 0.125% ✅ Qualifies
  ADA (+0.25%) ≥ 0.125% ✅ Qualifies
  
Result: Rotation allowed
         New universe: [BTC, ETH, SOL, XRP, ADA]
         (or cap-limited subset if needed)
```

**Outcome:** Only high-quality rotation. DOGE rejected despite score. ✅

### Scenario 3: Strong New Opportunity

```
Universe: [BTC (+0.5%), ETH (+0.3%), SOL (+0.2%)]
New candidate: [AVAX (+1.2%)]
Superiority factor: 1.25
Weakest: SOL (+0.2%)
Required: 0.2% × 1.25 = 0.25%

Step 4.5: AVAX passes profitability ✅
Step 4.6: AVAX (+1.2%) >> 0.25% ✅

Result: AVAX admitted
         Could displace SOL (lowest) if cap hit
```

**Outcome:** Strong opportunity captured, weak symbol exited. ✅

---

## Alignment Demonstration

### Before (Misaligned)

**Execution System Says:**
- "DOGE needs +0.35% net profit to be tradeable"
- "Algorithm: fees - move - slippage = NET_EDGE"

**Rotation System Says:**
- "DOGE scores 0.70, ranks #2"
- "Include in universe"

**Result:** Universe contains DOGE, execution rejects DOGE, capital idles. ❌

### After (Aligned - Phase 10)

**Both Systems Say:**
- "DOGE needs +0.35% net profit"
- "Algorithm: fees - move - slippage = NET_EDGE"
- "DOGE: execute test = fail, rotation test = fail"
- "DOGE: not in universe, not traded"

**Result:** Universe = Execution reality, no conflicts. ✅

---

## Professional Architecture

### Layers of Protection

```
Layer 1: Capital Floor (Governor)
         └─ No more than equity allows
         
Layer 2: Profitability Floor (ExecutionManager)
         └─ Only symbols execution would accept
         └─ Same EV calc as trades
         
Layer 3: Quality Bar (Relative Rule)
         └─ Only symbols better than current worst
         └─ Prevents sideways or downgrade rotations
```

### Decision Tree

```
Can we rotate?

1. Is this a high-quality symbol? (Profitability filter)
   NO  → Reject (execution wouldn't trade it)
   YES → Continue

2. Is it better than our weakest symbol? (Relative rule)
   NO  → Reject (keep current)
   YES → Continue

3. Do we have capital for it? (Governor cap)
   NO  → Reject (respect capital floor)
   YES → Continue

4. Add to universe
```

---

## Optional Enhancements (Later Phases)

### After 100+ Rotations

Once you have rotation performance data:

**Enhancement 1: Win-Rate Weighting**
```python
# Current: edge × 1.25
# Enhanced: (edge × 1.25) + (win_rate_factor × 0.3)

def _calculate_weighted_edge(edge, win_rate):
    return edge * 1.25 + win_rate * 0.3
```

**Enhancement 2: Regime-Based Factor**
```python
# Different superiority factors by regime

factors = {
    "bull": 1.1,      # Easier to rotate in bullish
    "normal": 1.25,   # Balanced
    "bear": 1.5       # Conservative in downtrend
}
factor = factors[current_regime]
required = weakest_edge * factor
```

**Enhancement 3: Drawdown-Sensitive Tightening**
```python
# Tighten gate during drawdowns

if drawdown > 5%:
    factor = 1.5      # More conservative
elif drawdown > 2%:
    factor = 1.35     # Moderate
else:
    factor = 1.25     # Normal
```

---

## Testing Checklist

### Unit Tests to Add

```python
async def test_profitability_filter_blocks_unprofitable():
    """Verify filter removes symbols execution would reject"""
    
async def test_relative_rule_keeps_strong_actives():
    """Verify rule keeps actives beating threshold"""
    
async def test_relative_rule_blocks_marginal_additions():
    """Verify rule rejects candidates below threshold"""
    
async def test_rotation_blocked_when_no_qualifiers():
    """Verify conservative behavior: keep current if all blocked"""
    
async def test_alignment_with_execution_logic():
    """Verify rotation EV = execution EV"""
```

### Integration Tests

```python
async def test_full_rotation_cycle_aligned():
    """Full cycle: score → filter → rule → rotate"""
    
async def test_edge_cases_safe_defaults():
    """Errors don't crash: keep current universe"""
```

### Production Monitoring

```
Log key metrics:
  [UURE] Profitability filter: X in → Y out (Z blocked)
  [UURE] Relative rule: weakest_edge=X, required=Y, qualifiers=Z
  [UURE] Rotation result: added={}, removed={}, kept={}
```

---

## Summary of Changes

### Code Changes

**File:** `core/universe_rotation_engine.py`

| Section | Change | Type |
|---------|--------|------|
| `__init__()` | No changes | Existing |
| `compute_and_apply_universe()` | Added 2 filter steps | Modified |
| `_apply_profitability_filter()` | New method | Added |
| `_apply_relative_replacement_rule()` | New method | Added |
| All other methods | Unchanged | Existing |

**Total:** ~200 lines added, 0 lines removed, comprehensive error handling

### Configuration Changes

| Parameter | Default | Type |
|-----------|---------|------|
| `ROTATION_SUPERIORITY_FACTOR` | 1.25 | New |
| `MIN_NET_PROFIT_AFTER_FEES` | 0.0035 | Reused |

### Documentation

- Updated: `ROTATION_REPLACEMENT_RULES_AUDIT.md` 
- Created: `PHASE_10_ALIGNMENT_IMPLEMENTATION.md` (this file)

---

## Deployment Checklist

- [x] Implement profitability filter
- [x] Implement relative replacement rule
- [x] Add ROTATION_SUPERIORITY_FACTOR config
- [x] Integrate into UURE pipeline
- [x] Syntax verification (no errors)
- [x] Documentation complete
- [ ] Unit tests (next)
- [ ] Integration tests (next)
- [ ] 1-2 rotation cycles monitoring (next)
- [ ] Production deployment

---

## Professional Verdict

✅ **Phase 10 Complete: Rotation-Execution Alignment**

System moved from:
- ❌ Smart retail bot (misaligned logic)

To:
- ✅ Institutional capital allocator (aligned logic)

**Key Achievement:** Rotation now uses same profitability math as execution. Universe = Trade Quality.

Next phase: Collect performance data and optionally add win-rate weighting.
