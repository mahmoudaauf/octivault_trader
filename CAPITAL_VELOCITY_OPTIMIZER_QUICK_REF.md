# Capital Velocity Optimizer - Architect's Quick Reference

## Problem Statement

Your system has **velocity governance** (reactive: exits slow positions) but lacks **velocity optimization** (proactive: allocates capital to faster opportunities).

**Governance** ≠ **Optimization**

```
Governance:    If profit_rate < target → exit               [REACTIVE]
Optimization:  Is this capital used better elsewhere? → rotate [PROACTIVE]
```

---

## What This Module Does

**4-Step Coordination:**

1. **MEASURE** - Real-time position velocity (P&L per hour)
2. **ESTIMATE** - Opportunity velocity from ML signals
3. **IDENTIFY** - Capital rotation candidates
4. **RECOMMEND** - Structured rotation suggestions

**Does NOT:**
- Execute exits directly
- Override existing authorities
- Change exit logic
- Bypass PortfolioAuthority or RotationAuthority

---

## Module Components

### Core Classes

| Class | Purpose |
|-------|---------|
| `CapitalVelocityOptimizer` | Main coordinator |
| `PositionVelocityMetric` | Real position velocity snapshot |
| `OpportunityVelocityMetric` | Forecasted opportunity velocity |
| `VelocityOptimizationPlan` | Structured recommendation output |

### Key Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `evaluate_position_velocity()` | position dict, timestamp | `PositionVelocityMetric` | Measure realized velocity |
| `estimate_opportunity_velocity()` | ML signal dict | `OpportunityVelocityMetric` | Forecast opportunity velocity |
| `recommend_rotation()` | position metrics, opportunity metrics | `List[Dict]` | Generate rotation suggestions |
| `optimize_capital_velocity()` | owned positions, candidates | `VelocityOptimizationPlan` | Main entry point |

---

## Formulas

### Position Velocity (Realized)

```
velocity = (unrealized_pnl_pct / age_hours) - holding_cost_per_hour

Example:
  Position: BTC, +2% PnL, held 1 hour
  Holding cost: ~10 bps per trade amortized = 0.001%/hr
  
  velocity = (0.02 / 1.0) - 0.0001 = 0.0199 = 1.99% per hour
```

### Opportunity Velocity (Forecasted)

```
velocity = (ml_confidence * expected_move_pct) / time_to_achieve

Example:
  ML signal: confidence=0.72, expected_move=1.5%, time=1hr
  
  velocity = (0.72 * 0.015) / 1.0 = 0.0108 = 1.08% per hour
```

### Velocity Gap (Why Rotate)

```
gap = opportunity_velocity - position_velocity

If gap > threshold (default 0.5%/hr):
  → Recommend rotation
  
Example:
  Opportunity: 1.08%/hr
  Position: 0.20%/hr
  Gap: 0.88%/hr > 0.50%/hr ✓ Rotate
```

---

## Integration Points (Minimal)

### In MetaController `__init__`

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

self.capital_velocity_optimizer = CapitalVelocityOptimizer(
    config=self.config,
    shared_state=self.shared_state,
    logger=self.logger
)
```

### In Orchestration Loop

```python
velocity_plan = await self.capital_velocity_optimizer.optimize_capital_velocity(
    owned_positions=owned_positions,
    candidate_symbols=candidate_list
)

# Log metrics
self.logger.info(
    "[Velocity] Portfolio: %.2f%%/hr | Opportunity: %.2f%%/hr | Gap: %.2f%%/hr",
    velocity_plan.portfolio_velocity_pct_per_hour,
    velocity_plan.opportunity_velocity_pct_per_hour,
    velocity_plan.velocity_gap
)

# Optionally act on recommendations (but don't bypass authorities)
if velocity_plan.velocity_gap > 1.0:
    self.logger.warning("[Velocity] Significant gap detected: %s", velocity_plan.rotations_recommended)
```

---

## Configuration

```python
# In core/config.py

# Master switch
ENABLE_CAPITAL_VELOCITY_OPTIMIZATION = True

# Rotation thresholds
VELOCITY_GAP_THRESHOLD_PCT = 0.5                # Min % per hour to consider rotating
VELOCITY_MIN_POSITION_AGE_HOURS = 0.25          # Min hold time (15 min)
VELOCITY_HOLDING_COST_FEE_BPS = 10.0            # Estimated round-trip fee

# Opportunity filtering
VELOCITY_CONFIDENCE_MIN = 0.55                  # Min ML confidence to estimate velocity
```

---

## Output Format (VelocityOptimizationPlan)

```python
@dataclass
class VelocityOptimizationPlan:
    timestamp: float                              # When plan was created
    portfolio_velocity_pct_per_hour: float       # Current (realized) %/hr
    opportunity_velocity_pct_per_hour: float     # Best available (forecasted) %/hr
    velocity_gap: float                          # Gap = opportunity - portfolio
    rotations_recommended: List[Dict]            # Structured recommendations
    hold_positions: List[str]                    # Symbols to keep
    analysis: Dict                               # Debug data
```

### Example Rotation Recommendation

```python
{
    "exit_symbol": "BTC",
    "opportunity_symbol": "SOL", 
    "velocity_gap_pct_per_hour": 0.88,         # Why: 88 bps/hr improvement
    "current_velocity_pct": 0.20,              # Current P&L rate
    "opportunity_velocity_pct": 1.08,          # Forecasted rate
    "reason": "VELOCITY_OPTIMIZATION_GAP",
    "confidence": 0.72,                         # ML confidence in SOL signal
    "position_age_hours": 2.5,                 # BTC has been held 2.5 hours
}
```

---

## Interaction with Existing Modules

### PortfolioAuthority (Layer 3 Exit Governance)

| Scenario | PortfolioAuth | VelocityOpt |
|----------|---------------|------------|
| Position has negative velocity | ✅ Exits (underperforming) | ⚠️ Flags as recyclable |
| Position lacks velocity but positive PnL | ✅ Holds | ⚠️ Watches for exit signal |
| Capital underutilized (low run rate) | ✅ Forces exits to recycle | ✅ Recommends better targets |

### RotationAuthority (Opportunity-Based Swaps)

| Scenario | RotationAuth | VelocityOpt |
|----------|-------------|------------|
| New signal beats old position | ✅ Can rotate | ✅ Quantifies improvement |
| Restricted by bracket limits | ✅ Enforces | ⚠️ Respects boundaries |
| Multi-symbol opportunity | ⚠️ Handles one swap | ✅ Ranks all opportunities |

### MLForecaster (Signal Generation)

| Signal Property | MLF | VelocityOpt |
|-----------------|-----|-------------|
| `confidence` | ✅ Outputs | ✅ Uses for filtering |
| `_expected_move_pct` | ✅ Outputs | ✅ Uses for velocity estimation |
| `action` (BUY/SELL) | ✅ Outputs | ✅ Uses for direction |

**Flow**: MLForecaster → Signal Bus → VelocityOpt reads latest signals

---

## Decision Tree

```
Position velocity < 0?
  YES → PortfolioAuthority handles exit
  NO  ↓
  
Position recyclable (old enough)?
  YES ↓
  NO  → Keep holding
  
Opportunity opportunity velocity > threshold?
  YES → Recommend rotation
  NO  → Keep position
```

---

## Real Example

### Portfolio State

```
BTC:  Entry 2 hrs ago, +0.2% PnL    → velocity = +0.10%/hr - fee = +0.09%/hr
ETH:  Entry 1 hr ago, +0.5% PnL     → velocity = +0.50%/hr - fee = +0.49%/hr
LINK: Entry 30min ago, -0.1% PnL    → velocity = -0.20%/hr - fee = -0.20%/hr (TOO YOUNG)

Portfolio weighted average: ~0.25%/hr
```

### ML Signals Available

```
SOL:   confidence=0.72, expected_move=1.5%  → velocity = 0.72 * 1.5% = 1.08%/hr
AVAX:  confidence=0.68, expected_move=1.0%  → velocity = 0.68 * 1.0% = 0.68%/hr
DOGE:  confidence=0.51, expected_move=2.0%  → filtered (confidence too low)
```

### Optimization Plan Output

```
portfolio_velocity_pct_per_hour = 0.25
opportunity_velocity_pct_per_hour = 1.08 (SOL)
velocity_gap = 0.83%/hr

rotations_recommended = [
    {
        "exit_symbol": "BTC",
        "opportunity_symbol": "SOL",
        "velocity_gap_pct_per_hour": 0.99,    (1.08 - 0.09)
        "confidence": 0.72,
    },
    {
        "exit_symbol": "ETH",
        "opportunity_symbol": "SOL",
        "velocity_gap_pct_per_hour": 0.59,    (1.08 - 0.49)
        "confidence": 0.72,
    }
]

# Action: MetaController sees BTC would improve by 99 bps/hr if rotated to SOL
# But RotationAuthority gates it, and PortfolioAuthority decides if BTC stays
```

---

## Why This Works

1. **Complements Governance** - Doesn't override, just informs
2. **Minimal Code** - ~200 lines, reuses existing modules
3. **Institutional Grade** - Uses professional velocity metrics
4. **Forward-Looking** - Bridges realized performance with forecasted opportunities
5. **Non-Prescriptive** - Recommends but doesn't execute

---

## Testing Checklist

- [ ] `evaluate_position_velocity()` matches manual calculations
- [ ] `estimate_opportunity_velocity()` correctly applies ML confidence
- [ ] `optimize_capital_velocity()` handles empty portfolios gracefully
- [ ] Rotation recommendations only for sufficiently aged positions
- [ ] Gap threshold correctly filters weak rotation opportunities
- [ ] Integration with MetaController doesn't break existing flow
- [ ] Logging provides visibility into velocity decisions
- [ ] Config params can disable module safely

---

## Performance Profile

- **CPU**: O(n*m) where n=positions, m=candidates (typically <100 total)
- **Memory**: Minimal (metrics are small dataclasses)
- **Latency**: <50ms typical (no ML inference, just signal reading)
- **Frequency**: Can run every orchestration cycle (~1-2 sec)

---

## Limitations

1. **Forward-Looking Only**: Opportunity velocity is purely ML-based (no order book analysis)
2. **Single Timeframe**: Assumes ~1 hour planning horizon (configurable)
3. **No Execution**: Provides recommendations only
4. **ML Dependency**: Requires MLForecaster signals with expected_move_pct
5. **Fee Estimate**: Uses static holding cost; doesn't adjust for pair-specific fees

---

## Future Enhancements

1. **Dynamic Time Horizons**: Adjust planning horizon based on market regime
2. **Order Book Integration**: Estimate fill quality for liquidity weighting
3. **Correlation Awareness**: Avoid rotating into correlated pairs
4. **Historical Backtest**: Validate velocity gap improvements post-execution
5. **Ensemble Opportunity**: Weight multiple signal sources, not just ML

---

## Questions?

**"Can it execute rotations?"**
No. It recommends via `VelocityOptimizationPlan.rotations_recommended`. MetaController decides whether to act.

**"What if ML signals are wrong?"**
Velocity optimizer filters by confidence threshold. Low-confidence signals are ignored. Existing governance still applies.

**"Does it work with existing PortfolioAuthority logic?"**
Yes. PortfolioAuthority handles underperforming exits. VelocityOpt suggests better targets. Both serve different purposes.

**"Is it safe in production?"**
Yes. Advisory-only. Can disable with `ENABLE_CAPITAL_VELOCITY_OPTIMIZATION = False`.

**"How often should it run?"**
Every MetaController orchestration cycle (typically 1-2 seconds), but rotations gate on minimum position age anyway.
