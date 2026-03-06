# Capital Velocity Optimizer - Minimal Integration Example

This document shows the exact 3 code changes needed to add velocity optimization to MetaController.

---

## Change 1: Initialization (Lines ~500-750 in `__init__`)

**File**: `core/meta_controller.py`

**Location**: In the `__init__` method, after `PortfolioAuthority` is initialized

**FIND THIS** (existing code):
```python
# ... existing code ...
# Portfolio Authority (Layer 3) - Capital utilization governance
from core.portfolio_authority import PortfolioAuthority

self.portfolio_authority = PortfolioAuthority(
    logger=self.logger,
    config=self.config,
    shared_state=self.shared_state
)
self.logger.info("[Meta:Init] PortfolioAuthority initialized for velocity governance")
```

**ADD AFTER** (~20 lines):
```python
# ═══════════════════════════════════════════════════════════════════════════════
# CAPITAL VELOCITY OPTIMIZER (Proactive allocation planning)
# ═══════════════════════════════════════════════════════════════════════════════
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

self.capital_velocity_optimizer = CapitalVelocityOptimizer(
    config=self.config,
    shared_state=self.shared_state,
    logger=self.logger
)
self.logger.info("[Meta:Init] Capital Velocity Optimizer initialized for velocity planning")
```

---

## Change 2: Velocity Planning Call (In Orchestration Loop)

**File**: `core/meta_controller.py`

**Location**: In `orchestrate()` or `_build_decisions()`, after collecting ML signals but before final execution decisions

**FIND THIS** (existing code pattern):
```python
async def orchestrate(self, accepted_symbols_set: set = None) -> Dict[str, Any]:
    """Main orchestration loop."""
    
    # ... existing signal collection ...
    
    # Portfolio Authority velocity check
    vel_exit_sig = self.portfolio_authority.authorize_velocity_exit(
        owned_positions=owned_positions_for_rea,
        current_metrics=metrics
    )
    
    # ... more decisions ...
```

**ADD AFTER the Portfolio Authority check** (~30 lines):
```python
# ═══════════════════════════════════════════════════════════════════════════════
# CAPITAL VELOCITY OPTIMIZATION (Forward-looking capital allocation)
# ═══════════════════════════════════════════════════════════════════════════════
velocity_plan = None
try:
    # Get candidate symbols (union of universe, discovery, etc.)
    candidate_symbols = list(accepted_symbols_set or [])
    
    # Run optimization
    velocity_plan = await self.capital_velocity_optimizer.optimize_capital_velocity(
        owned_positions=owned_positions_for_rea,
        candidate_symbols=candidate_symbols,
    )
    
    # Log metrics for visibility
    self.logger.info(
        "[Meta:VelocityOpt] Portfolio: %.2f%%/hr | Opportunity: %.2f%%/hr | Gap: %.2f%%/hr | Rotations: %d",
        velocity_plan.portfolio_velocity_pct_per_hour,
        velocity_plan.opportunity_velocity_pct_per_hour,
        velocity_plan.velocity_gap,
        len(velocity_plan.rotations_recommended),
    )
    
    # Optional: Store plan in SharedState for other consumers
    if hasattr(self.ss, "latest_velocity_plan"):
        self.ss.latest_velocity_plan = velocity_plan
        
except Exception as e:
    self.logger.warning("[Meta:VelocityOpt] Error in velocity optimization: %s", e)
    velocity_plan = None
```

---

## Change 3: Optional - Use Recommendations

**File**: `core/meta_controller.py`

**Location**: In decision logic, when deciding whether to approve rotations

**ADD THIS** (if you want MetaController to consider velocity gap):
```python
# Inside decision loop, AFTER existing rotation checks
if velocity_plan and velocity_plan.rotations_recommended:
    for rotation in velocity_plan.rotations_recommended:
        exit_symbol = rotation.get("exit_symbol")
        opportunity_symbol = rotation.get("opportunity_symbol")
        velocity_gap = rotation.get("velocity_gap_pct_per_hour", 0.0)
        ml_confidence = rotation.get("confidence", 0.0)
        
        # Gate: Only consider if gap is material and ML signal is strong
        if velocity_gap > 0.5 and ml_confidence > 0.65:  # Tunable thresholds
            self.logger.warning(
                "[Meta:VelocityOpt] Velocity gap identified: "
                "Exit %s (%.2f%%/hr) for %s (%.2f%%/hr) - Gap: %.2f%%/hr | ML Conf: %.2f",
                exit_symbol,
                rotation.get("current_velocity_pct", 0.0),
                opportunity_symbol,
                rotation.get("opportunity_velocity_pct", 0.0),
                velocity_gap,
                ml_confidence,
            )
            
            # NOTE: Don't execute directly. Instead:
            # 1. Let RotationAuthority evaluate the swap
            # 2. Let PortfolioAuthority apply velocity exit rules
            # 3. Use this as SIGNAL to increase priority of rotation
            
            # Example: Boost rotation score if velocity gap supports it
            if hasattr(self.ss, "velocity_gap_override"):
                self.ss.velocity_gap_override = {
                    "exit_symbol": exit_symbol,
                    "opportunity_symbol": opportunity_symbol,
                    "gap_pct": velocity_gap,
                }
```

---

## Configuration (Add to `core/config.py`)

**Location**: In the Configuration class, add a new section

```python
# ═══════════════════════════════════════════════════════════════════════════════
# CAPITAL VELOCITY OPTIMIZER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Master switch: Enable/disable velocity optimization
ENABLE_CAPITAL_VELOCITY_OPTIMIZATION = True

# Velocity gap threshold: Minimum improvement (% per hour) to recommend rotation
# Example: 0.5% per hour means we only rotate if opportunity is 0.5% better per hour
VELOCITY_GAP_THRESHOLD_PCT = 0.5

# Minimum position age (hours) before it's eligible for rotation
# Example: 0.25 = 15 minutes (don't rotate fresh trades)
VELOCITY_MIN_POSITION_AGE_HOURS = 0.25

# Estimated round-trip fee (basis points)
# Used to calculate net velocity (PnL - fee cost)
# Example: 10.0 = 10 bps round-trip (entry + exit)
VELOCITY_HOLDING_COST_FEE_BPS = 10.0

# Minimum ML confidence to consider an opportunity
# Example: 0.55 = only estimate velocity for 55%+ confidence signals
VELOCITY_CONFIDENCE_MIN = 0.55
```

---

## That's It!

The three changes above:

1. **Initialize** the optimizer in `__init__` (~10 lines)
2. **Call** the optimizer in orchestration loop (~25 lines)
3. **Optionally use** recommendations in decision logic (~30 lines)
4. **Configure** parameters in config (~20 lines)

**Total: ~85 lines of code** (much shorter than the ~200 line optimizer module itself)

---

## Verification Checklist

After integration:

- [ ] `meta_controller.py` imports `CapitalVelocityOptimizer` without error
- [ ] `self.capital_velocity_optimizer` exists after `__init__`
- [ ] Log shows `"[Meta:Init] Capital Velocity Optimizer initialized..."`
- [ ] `orchestrate()` calls `optimize_capital_velocity()` and logs velocity metrics
- [ ] No errors in velocity plan computation
- [ ] Config parameters are recognized
- [ ] Recommendations appear in logs
- [ ] MetaController continues to function normally (no breaking changes)

---

## Example Log Output

After integration, you should see logs like:

```
[Meta:Init] Capital Velocity Optimizer initialized for velocity planning
[Meta:VelocityOpt] Portfolio: 0.35%/hr | Opportunity: 1.20%/hr | Gap: 0.85%/hr | Rotations: 2
[Meta:VelocityOpt] Velocity gap identified: Exit BTC (0.10%/hr) for SOL (1.08%/hr) - Gap: 0.98%/hr | ML Conf: 0.72
```

---

## Minimal Test (After Integration)

Add this to a test script:

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

# Test 1: Initialize
optimizer = CapitalVelocityOptimizer(config, shared_state, logger)
print(f"✓ Optimizer created: {optimizer}")

# Test 2: Measure position velocity
position = {
    "unrealized_pnl_pct": 0.02,
    "entry_time": time.time() - 3600,
}
metric = optimizer.evaluate_position_velocity("BTC", position, time.time())
print(f"✓ Position velocity: {metric.net_velocity * 100:.2f}%/hr")

# Test 3: Estimate opportunity velocity
signal = {
    "confidence": 0.75,
    "_expected_move_pct": 0.015,
}
opp = optimizer.estimate_opportunity_velocity("SOL", signal)
print(f"✓ Opportunity velocity: {opp.estimated_velocity_pct * 100:.2f}%/hr")

# Test 4: Full plan
plan = await optimizer.optimize_capital_velocity(
    owned_positions={"BTC": position},
    candidate_symbols=["SOL", "ETH"]
)
print(f"✓ Optimization plan: {len(plan.rotations_recommended)} recommendations")
```

---

## Next Steps

1. **Copy the module**: Save `capital_velocity_optimizer.py` to `core/`
2. **Integrate**: Add the 3 changes to `meta_controller.py`
3. **Configure**: Add config parameters to `config.py`
4. **Test**: Run minimal test above
5. **Monitor**: Check logs for velocity metrics
6. **Tune**: Adjust `VELOCITY_GAP_THRESHOLD_PCT` based on rotation frequency

That's all! 🚀
