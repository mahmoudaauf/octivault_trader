# Capital Velocity Optimizer - Integration Guide

## Overview

The **Capital Velocity Optimizer** is a coordination layer that bridges velocity governance (PortfolioAuthority, RotationAuthority) with proactive capital allocation planning.

**Key Point**: It does NOT execute exits. It RECOMMENDS rotations based on institutional capital velocity metrics.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     MetaController                              │
└──────────────────────────────────────────────────────────────────┘
                             ↑
                             │ Uses recommendations
                             │
       ┌─────────────────────────────────────────────────┐
       │   Capital Velocity Optimizer (THIS MODULE)      │
       └─────────────────────────────────────────────────┘
                             ↑
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    Position Velocity    Opportunity Velocity   Portfolio
    (PortfolioAuthority) (MLForecaster)         Metrics
```

### What It Reads From

1. **SharedState** (position data, latest prices)
2. **MLForecaster** (signals with confidence and expected_move_pct)
3. **Position metrics** (age, PnL, entry time)

### What It Outputs

- `VelocityOptimizationPlan` with:
  - Current portfolio velocity (realized, per hour)
  - Best opportunity velocity (forward-looking)
  - Velocity gap (opportunity - portfolio)
  - Rotation recommendations (structured)

---

## Integration Steps

### Step 1: Add to MetaController `__init__`

**Location**: `meta_controller.py`, around line ~500-750 in `__init__`

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

class MetaController:
    def __init__(self, ...):
        # ... existing code ...
        
        # ═══════════════════════════════════════════════════════════════
        # CAPITAL VELOCITY OPTIMIZER (Proactive allocation planning)
        # ═══════════════════════════════════════════════════════════════
        self.capital_velocity_optimizer = CapitalVelocityOptimizer(
            config=self.config,
            shared_state=self.shared_state,
            logger=self.logger
        )
        
        self.logger.info("[Meta:Init] Capital Velocity Optimizer initialized")
```

### Step 2: Add Velocity Planning Call

**Location**: In `orchestrate()` or decision loop, after signal collection but before final execution decision

```python
async def orchestrate(self, accepted_symbols_set: set = None) -> Dict[str, Any]:
    """Main orchestration loop."""
    
    # ... existing code ...
    
    # Measure capital velocity and get optimization plan
    try:
        owned = self._owned_positions_for_rea()  # Your method
        candidates = list(accepted_symbols_set or [])
        
        velocity_plan = await self.capital_velocity_optimizer.optimize_capital_velocity(
            owned_positions=owned,
            candidate_symbols=candidates,
        )
        
        # Log the plan for transparency
        self.logger.info(
            "[Meta:VelocityOpt] Portfolio: %.2f%%/hr | Opportunity: %.2f%%/hr | Gap: %.2f%%/hr | Rotations: %d",
            velocity_plan.portfolio_velocity_pct_per_hour,
            velocity_plan.opportunity_velocity_pct_per_hour,
            velocity_plan.velocity_gap,
            len(velocity_plan.rotations_recommended),
        )
        
        # OPTIONAL: Feed rotations to decision logic
        # (See "How to Use Recommendations" below)
        
    except Exception as e:
        self.logger.warning("[Meta:VelocityOpt] Error in velocity optimization: %s", e)
        velocity_plan = None
    
    # ... continue with existing orchestration ...
```

### Step 3: Optional - Use Velocity Recommendations

If you want MetaController to ACT on velocity recommendations, add this decision layer:

```python
# Inside decision loop, after PortfolioAuthority checks
if velocity_plan and velocity_plan.rotations_recommended:
    for rotation in velocity_plan.rotations_recommended:
        exit_symbol = rotation.get("exit_symbol")
        opportunity_symbol = rotation.get("opportunity_symbol")
        velocity_gap = rotation.get("velocity_gap_pct_per_hour", 0.0)
        
        # Gate: Only rotate if gap is material and confidence is high
        if (velocity_gap > 0.5 and  # > 0.5% per hour improvement
            rotation.get("confidence", 0.0) > 0.65):
            
            self.logger.warning(
                "[Meta:VelocityOpt] ROTATION RECOMMENDATION: Exit %s (%.2f%%/hr) for %s (%.2f%%/hr)",
                exit_symbol,
                rotation.get("current_velocity_pct", 0.0),
                opportunity_symbol,
                rotation.get("opportunity_velocity_pct", 0.0),
            )
            
            # Forward to rotation authority or emit as signal
            # (Don't execute directly - let existing authorities decide)
```

---

## Configuration Parameters

Add to your `config.py`:

```python
# Capital Velocity Optimizer Configuration
ENABLE_CAPITAL_VELOCITY_OPTIMIZATION = True          # Master switch
VELOCITY_GAP_THRESHOLD_PCT = 0.5                     # Min % per hour improvement to rotate
VELOCITY_MIN_POSITION_AGE_HOURS = 0.25               # Min hold time before recyclable (15 min)
VELOCITY_HOLDING_COST_FEE_BPS = 10.0                 # Estimated round-trip fee (bps)
VELOCITY_CONFIDENCE_MIN = 0.55                       # Min ML confidence to consider opportunity
```

---

## Key Design Decisions

### 1. No Direct Exit Authority

The optimizer **recommends** rotations but does NOT execute them. All actions flow through:
- PortfolioAuthority (velocity-based exits)
- RotationAuthority (opportunity-based swaps)
- MetaController (orchestration)

This preserves the existing governance hierarchy.

### 2. Uses Existing Module Outputs

Instead of reimplementing velocity metrics:

| Metric | Source | Module |
|--------|--------|--------|
| Position velocity (realized) | SharedState positions | (measured by optimizer) |
| Opportunity velocity (ML forecast) | MLForecaster signals | (estimated by optimizer) |
| Portfolio governance | PortfolioAuthority | (existing authority) |
| Rotation governance | RotationAuthority | (existing authority) |

### 3. Two Velocity Types

**Position Velocity (Realized)**:
```
velocity = (unrealized_pnl_pct) / (age_hours) - holding_cost
```
Measured in real positions. Can be negative (losing money).

**Opportunity Velocity (Forecasted)**:
```
velocity = (ml_confidence * expected_move_pct) / time_to_achieve
```
Forward-looking from ML signals. Pure estimate.

### 4. Rotation Recommendations

Optimizer recommends a rotation ONLY if:
1. Current position is **recyclable** (low/negative velocity, aged enough)
2. Opportunity exists with **significantly higher velocity**
3. Gap exceeds threshold (configurable)
4. ML confidence is adequate

This avoids churn and honors the existing rotation authority.

---

## Example Output (VelocityOptimizationPlan)

```python
VelocityOptimizationPlan(
    timestamp=1703001600.123,
    portfolio_velocity_pct_per_hour=0.35,          # Current weighted average
    opportunity_velocity_pct_per_hour=1.20,        # Best available
    velocity_gap=0.85,                             # Opportunity - Portfolio
    rotations_recommended=[
        {
            "exit_symbol": "BTC",
            "opportunity_symbol": "SOL",
            "velocity_gap_pct_per_hour": 0.95,
            "current_velocity_pct": 0.10,
            "opportunity_velocity_pct": 1.05,
            "reason": "VELOCITY_OPTIMIZATION_GAP",
            "confidence": 0.72,
            "position_age_hours": 2.5,
        }
    ],
    hold_positions=["ETH", "LINK"],
    analysis={
        "position_count": 3,
        "candidate_count": 12,
        "recyclable_count": 1,
        "opportunity_count": 6,
        "position_metrics": {...},
        "opportunity_metrics": {...},
    }
)
```

---

## Integration Checklist

- [ ] Add `CapitalVelocityOptimizer` import in `meta_controller.py`
- [ ] Initialize in `__init__` method
- [ ] Call `optimize_capital_velocity()` in orchestration loop
- [ ] Log velocity plan metrics
- [ ] (Optional) Use rotation recommendations in decision flow
- [ ] Add config parameters to `core/config.py`
- [ ] Test with small portfolio (3-5 positions)
- [ ] Validate velocity calculations against realized metrics
- [ ] Document in runbook

---

## Testing Recommendations

### Unit Test: Position Velocity

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

optimizer = CapitalVelocityOptimizer(config, shared_state, logger)

position = {
    "unrealized_pnl_pct": 0.02,  # 2% profit
    "entry_time": time.time() - 3600,  # 1 hour ago
    "value_usdt": 100.0,
}

metric = optimizer.evaluate_position_velocity("BTC", position, time.time())
assert metric.pnl_per_hour > 0
assert metric.age_hours > 0.99
```

### Unit Test: Opportunity Velocity

```python
ml_signal = {
    "confidence": 0.75,
    "_expected_move_pct": 0.015,  # 1.5% expected move
    "action": "BUY",
}

opp_metric = optimizer.estimate_opportunity_velocity("SOL", ml_signal)
expected_return = 0.75 * 0.015  # 1.125%
assert abs(opp_metric.expected_return_pct - 0.01125) < 0.0001
```

### Integration Test: Full Plan

```python
owned_positions = {
    "BTC": {"unrealized_pnl_pct": 0.01, "entry_time": ...},
    "ETH": {"unrealized_pnl_pct": -0.005, "entry_time": ...},
}

candidates = ["SOL", "LINK", "AVAX"]

plan = await optimizer.optimize_capital_velocity(owned_positions, candidates)
assert plan.portfolio_velocity_pct_per_hour >= 0
assert isinstance(plan.rotations_recommended, list)
```

---

## Troubleshooting

### Rotations Not Recommended?
1. Check `VELOCITY_GAP_THRESHOLD_PCT` - may be too high
2. Verify ML signals have `_expected_move_pct` populated
3. Check ML confidence floor: `VELOCITY_CONFIDENCE_MIN`
4. Ensure positions are old enough: `VELOCITY_MIN_POSITION_AGE_HOURS`

### Rotations Too Frequent?
1. Increase `VELOCITY_GAP_THRESHOLD_PCT` (currently 0.5%)
2. Increase `VELOCITY_MIN_POSITION_AGE_HOURS` (currently 15 min)
3. Reduce `ENABLE_CAPITAL_VELOCITY_OPTIMIZATION` to advisory-only

### Velocity Metrics Seem Wrong?
1. Verify position `entry_time` is correctly set
2. Confirm `unrealized_pnl_pct` calculations are accurate
3. Check that `holding_cost_fee_bps` matches actual trading fees
4. Validate against manual calculation: `(pnl_pct / age_hours) - fee`

---

## Advanced Customization

### Custom Opportunity Estimation

If you want to feed additional metrics (volatility, liquidity, etc.):

```python
async def my_custom_opportunity_estimate(self, symbol: str) -> float:
    # Your logic here
    volatility = await self.ss.get_volatility(symbol)
    liquidity = await self.ss.get_liquidity_score(symbol)
    return volatility * liquidity

# Subclass and override:
class CustomVelocityOptimizer(CapitalVelocityOptimizer):
    async def estimate_universe_opportunity(self, candidates):
        # Use custom estimation
        pass
```

### Dynamic Thresholds

You can make velocity gap threshold adaptive:

```python
def _get_dynamic_velocity_threshold(self, market_regime: str) -> float:
    thresholds = {
        "BULL": 0.3,      # Aggressive rotation
        "NORMAL": 0.5,    # Balanced
        "BEAR": 1.0,      # Conservative
    }
    return thresholds.get(market_regime, 0.5)
```

---

## Summary

The **Capital Velocity Optimizer** is a lightweight (~200 lines) coordination layer that:

✅ Measures real position velocity  
✅ Estimates opportunity velocity from ML signals  
✅ Identifies capital rotation opportunities  
✅ Recommends (does not execute)  
✅ Integrates with existing authorities  
✅ Provides institutional-grade metrics  

It answers the question: **"Can we do better with this capital?"**
