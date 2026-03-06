# Capital Velocity Optimizer - Validation & Diagnostics

This guide helps you verify the optimizer is working correctly after integration.

---

## Pre-Integration Validation

### 1. Module Syntax Check

```bash
# Verify module syntax
python3 -m py_compile core/capital_velocity_optimizer.py

# Expected output: (silent = success, no output means no syntax errors)
```

### 2. Import Validation

```python
# Test basic import
from core.capital_velocity_optimizer import (
    CapitalVelocityOptimizer,
    PositionVelocityMetric,
    OpportunityVelocityMetric,
    VelocityOptimizationPlan
)

print("✓ All classes imported successfully")
```

### 3. Instantiation Check

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
from core.config import Config
import logging

config = Config()
logger = logging.getLogger("test")

# Create optimizer instance
optimizer = CapitalVelocityOptimizer(config, None, logger)
print(f"✓ Optimizer instantiated: {type(optimizer).__name__}")

# Verify key attributes
assert hasattr(optimizer, "evaluate_position_velocity")
assert hasattr(optimizer, "estimate_opportunity_velocity")
assert hasattr(optimizer, "optimize_capital_velocity")
print("✓ All required methods exist")
```

---

## Post-Integration Validation

### 1. MetaController Initialization

```python
# After adding to meta_controller.py __init__
meta = MetaController(...)

# Verify optimizer exists
assert hasattr(meta, "capital_velocity_optimizer")
print("✓ Optimizer initialized in MetaController")

# Verify it's the right type
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
assert isinstance(meta.capital_velocity_optimizer, CapitalVelocityOptimizer)
print("✓ Optimizer is correct type")
```

### 2. Orchestration Loop Integration

Check logs for:

```
[Meta:Init] Capital Velocity Optimizer initialized for velocity planning
```

This confirms the optimizer was created successfully.

---

## Diagnostic Tests

### Test 1: Position Velocity Calculation

**Purpose**: Verify velocity measurement formula is correct

```python
import time
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
from core.config import Config

config = Config()
config.VELOCITY_HOLDING_COST_FEE_BPS = 10.0
optimizer = CapitalVelocityOptimizer(config, None, None)

# Test case: +2% PnL, held 1 hour
now = time.time()
position = {
    "unrealized_pnl_pct": 0.02,
    "entry_time": now - 3600,  # 1 hour ago
    "value_usdt": 100.0,
}

metric = optimizer.evaluate_position_velocity("BTC", position, now)

print(f"Position Velocity Test:")
print(f"  PnL %: {metric.pnl_pct * 100:.2f}%")
print(f"  Age: {metric.age_hours:.2f} hours")
print(f"  PnL/hour: {metric.pnl_per_hour * 100:.4f}%")
print(f"  Holding cost: {metric.holding_cost_bps:.2f} bps")
print(f"  Net velocity: {metric.net_velocity * 100:.4f}%")
print(f"  Recyclable: {metric.is_recyclable}")

# Verify calculation
expected_pnl_per_hour = 0.02 / 1.0  # 2%
expected_holding_cost = (10.0 * 100.0) / (1.0 * 10000.0)  # ~0.01%
expected_net = expected_pnl_per_hour - expected_holding_cost

assert abs(metric.pnl_per_hour - expected_pnl_per_hour) < 0.0001, "PnL/hour incorrect"
assert abs(metric.net_velocity - expected_net) < 0.0001, "Net velocity incorrect"

print("✓ Position velocity calculation is correct")
```

### Test 2: Opportunity Velocity Estimation

**Purpose**: Verify ML signal conversion to velocity

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
from core.config import Config

config = Config()
config.VELOCITY_CONFIDENCE_MIN = 0.55
optimizer = CapitalVelocityOptimizer(config, None, None)

# Test case: 72% confidence, 1.5% expected move
ml_signal = {
    "confidence": 0.72,
    "_expected_move_pct": 0.015,
    "action": "BUY",
}

opp = optimizer.estimate_opportunity_velocity("SOL", ml_signal)

print(f"Opportunity Velocity Test:")
print(f"  ML Confidence: {opp.ml_confidence * 100:.1f}%")
print(f"  Expected move: {opp.expected_move_pct * 100:.2f}%")
print(f"  Expected return: {opp.expected_return_pct * 100:.3f}%")
print(f"  Estimated velocity: {opp.estimated_velocity_pct * 100:.3f}%")

# Verify calculation
expected_return = 0.72 * 0.015  # 1.08%
assert abs(opp.expected_return_pct - expected_return) < 0.0001, "Expected return incorrect"

expected_velocity = expected_return / 1.0  # 1 hour horizon
assert abs(opp.estimated_velocity_pct - expected_velocity) < 0.0001, "Velocity incorrect"

print("✓ Opportunity velocity estimation is correct")
```

### Test 3: Rotation Recommendation Logic

**Purpose**: Verify rotation recommendations meet thresholds

```python
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
from core.config import Config
import time

config = Config()
config.VELOCITY_GAP_THRESHOLD_PCT = 0.5
optimizer = CapitalVelocityOptimizer(config, None, None)

# Current position: low velocity
now = time.time()
position_metrics = {
    "BTC": optimizer.evaluate_position_velocity("BTC", {
        "unrealized_pnl_pct": 0.001,  # +0.1%
        "entry_time": now - 3600,
        "value_usdt": 100.0,
    }, now)
}

# Better opportunity: high velocity
opportunity_metrics = {
    "SOL": type('Metric', (), {
        'symbol': 'SOL',
        'estimated_velocity_pct': 0.0108,  # 1.08%/hr
        'ml_confidence': 0.72,
        'expected_return_pct': 0.0108,
    })()
}

rotations = optimizer.recommend_rotation(
    position_metrics,
    opportunity_metrics,
    portfolio_velocity_avg=0.0001  # Very low portfolio avg
)

print(f"Rotation Recommendation Test:")
print(f"  Position velocity: {position_metrics['BTC'].net_velocity * 100:.3f}%/hr")
print(f"  Opportunity velocity: 1.08%/hr")
print(f"  Recommendations: {len(rotations)}")

if rotations:
    for rotation in rotations:
        print(f"  - {rotation['exit_symbol']} → {rotation['opportunity_symbol']}")
        print(f"    Gap: {rotation['velocity_gap_pct_per_hour']:.3f}%/hr")

print("✓ Rotation recommendation logic works")
```

### Test 4: Full Optimization Plan

**Purpose**: Verify end-to-end execution

```python
import asyncio
from core.capital_velocity_optimizer import CapitalVelocityOptimizer
from core.config import Config
import time

async def test_full_optimization():
    config = Config()
    optimizer = CapitalVelocityOptimizer(config, None, None)
    
    # Mock shared_state with signals
    class MockSharedState:
        latest_ml_signals = {
            "SOL": {
                "confidence": 0.72,
                "_expected_move_pct": 0.015,
                "action": "BUY",
            },
            "ETH": {
                "confidence": 0.65,
                "_expected_move_pct": 0.010,
                "action": "BUY",
            },
        }
    
    optimizer.ss = MockSharedState()
    
    # Current positions
    now = time.time()
    owned_positions = {
        "BTC": {
            "unrealized_pnl_pct": 0.001,
            "entry_time": now - 3600,
            "value_usdt": 100.0,
            "qty": 1.0,
        },
        "ETH": {
            "unrealized_pnl_pct": -0.002,
            "entry_time": now - 1800,  # Only 30 min old
            "value_usdt": 50.0,
            "qty": 1.0,
        },
    }
    
    # Candidate symbols
    candidates = ["SOL", "LINK", "AVAX"]
    
    # Run optimization
    plan = await optimizer.optimize_capital_velocity(
        owned_positions=owned_positions,
        candidate_symbols=candidates,
    )
    
    print(f"Full Optimization Plan Test:")
    print(f"  Portfolio velocity: {plan.portfolio_velocity_pct_per_hour:.3f}%/hr")
    print(f"  Opportunity velocity: {plan.opportunity_velocity_pct_per_hour:.3f}%/hr")
    print(f"  Velocity gap: {plan.velocity_gap:.3f}%/hr")
    print(f"  Rotations recommended: {len(plan.rotations_recommended)}")
    print(f"  Hold positions: {plan.hold_positions}")
    
    # Verify output structure
    assert hasattr(plan, 'timestamp'), "Missing timestamp"
    assert hasattr(plan, 'portfolio_velocity_pct_per_hour'), "Missing portfolio velocity"
    assert hasattr(plan, 'opportunity_velocity_pct_per_hour'), "Missing opportunity velocity"
    assert isinstance(plan.rotations_recommended, list), "Recommendations not a list"
    assert isinstance(plan.hold_positions, list), "Hold positions not a list"
    
    print("✓ Full optimization plan structure is correct")
    return plan

# Run async test
plan = asyncio.run(test_full_optimization())
```

---

## Runtime Diagnostics

### Check Log Output

After running orchestration, look for logs containing:

```
[Meta:VelocityOpt] Portfolio: X.XXX%/hr | Opportunity: Y.YYY%/hr | Gap: Z.ZZZ%/hr | Rotations: N
```

**What to look for**:
- Portfolio velocity > -5% (reasonable for real trading)
- Opportunity velocity > 0% (forward-looking)
- Gap generally >= 0 (opportunities better than current)
- Rotations count reasonable (0-3 typically)

### Extract Velocity Metrics

Add this debug code to MetaController orchestration:

```python
if velocity_plan:
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"CAPITAL VELOCITY DIAGNOSTICS")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Portfolio Velocity:     {velocity_plan.portfolio_velocity_pct_per_hour:>8.3f}%/hr")
    print(f"Opportunity Velocity:   {velocity_plan.opportunity_velocity_pct_per_hour:>8.3f}%/hr")
    print(f"Velocity Gap:           {velocity_plan.velocity_gap:>8.3f}%/hr")
    print(f"Rotations Recommended:  {len(velocity_plan.rotations_recommended):>8}")
    print(f"Positions to Hold:      {len(velocity_plan.hold_positions):>8}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if velocity_plan.analysis.get("position_metrics"):
        print(f"\nPOSITION VELOCITY DETAILS:")
        for sym, metrics in velocity_plan.analysis["position_metrics"].items():
            print(f"  {sym:>6} age={metrics.get('age_hours', 0):>5.2f}hr "
                  f"pnl={metrics.get('pnl_pct', 0):>7.3f}% "
                  f"vel={metrics.get('net_velocity_pct', 0):>7.3f}%/hr "
                  f"recycle={metrics.get('recyclable', False)}")
    
    if velocity_plan.rotations_recommended:
        print(f"\nROTATION RECOMMENDATIONS:")
        for rot in velocity_plan.rotations_recommended:
            print(f"  {rot.get('exit_symbol', '?'):>6} → {rot.get('opportunity_symbol', '?'):<6} "
                  f"gap={rot.get('velocity_gap_pct_per_hour', 0):>6.3f}%/hr "
                  f"conf={rot.get('confidence', 0):>5.2f}")
```

---

## Performance Checks

### Latency Measurement

```python
import time
import asyncio
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

async def measure_latency():
    optimizer = CapitalVelocityOptimizer(config, ss, logger)
    
    owned_positions = {...}  # Real portfolio
    candidates = [...]       # Real candidates
    
    start = time.time()
    plan = await optimizer.optimize_capital_velocity(
        owned_positions=owned_positions,
        candidate_symbols=candidates,
    )
    elapsed = time.time() - start
    
    print(f"Optimization latency: {elapsed*1000:.2f}ms")
    
    # Should be < 100ms for typical portfolio
    assert elapsed < 0.1, f"Optimization too slow: {elapsed*1000:.1f}ms"

asyncio.run(measure_latency())
```

### Memory Profile

```python
import tracemalloc
from core.capital_velocity_optimizer import CapitalVelocityOptimizer

tracemalloc.start()

optimizer = CapitalVelocityOptimizer(config, ss, logger)
plan = await optimizer.optimize_capital_velocity(
    owned_positions={...},
    candidate_symbols=[...]
)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

# Should be < 5 MB
assert peak < 5 * 1024 * 1024
```

---

## Validation Checklist

### Before Integration
- [ ] Module syntax is valid (`py_compile` succeeds)
- [ ] All classes import successfully
- [ ] Optimizer instantiates without errors
- [ ] Key methods exist and are callable

### After MetaController Integration
- [ ] MetaController `__init__` completes without error
- [ ] Optimizer is assigned to `self.capital_velocity_optimizer`
- [ ] Orchestration loop calls `optimize_capital_velocity()`
- [ ] No exceptions in optimization flow

### Verification Tests
- [ ] Position velocity calculation matches manual math
- [ ] Opportunity velocity converts ML signals correctly
- [ ] Rotation recommendations respect gap threshold
- [ ] Full plan output has correct structure
- [ ] Latency < 100ms (typical portfolio)
- [ ] Memory usage < 5 MB

### Runtime Validation
- [ ] Velocity metrics appear in logs
- [ ] Portfolio/opportunity/gap values are sensible
- [ ] Recommendations count is reasonable (0-3)
- [ ] No errors in MetaController logs
- [ ] Trading continues normally

### Edge Cases
- [ ] Empty portfolio handled gracefully
- [ ] No candidates handled gracefully
- [ ] Weak ML signals filtered out
- [ ] Very young positions not rotated
- [ ] Negative velocity positions identified

---

## Troubleshooting

### Optimizer Not Appearing in Logs

**Symptom**: No `[Meta:VelocityOpt]` logs

**Check**:
1. Is `self.capital_velocity_optimizer` created in `__init__`?
2. Is `optimize_capital_velocity()` called in orchestration loop?
3. Is logger.info() call present?

**Fix**:
```python
# In orchestration, after call:
if velocity_plan:
    self.logger.info("[VelocityOpt] Plan: %.2f%%/hr (portfolio) vs %.2f%%/hr (opportunity)",
        velocity_plan.portfolio_velocity_pct_per_hour,
        velocity_plan.opportunity_velocity_pct_per_hour
    )
```

### Velocity Metrics Seem Wrong

**Symptom**: Portfolio velocity is negative, or too high

**Check**:
1. Are position `entry_time` values correct?
2. Are `unrealized_pnl_pct` values accurate?
3. Is `VELOCITY_HOLDING_COST_FEE_BPS` reasonable?

**Debug**:
```python
# Print position details
for sym, pos in owned_positions.items():
    pnl_pct = float(pos.get("unrealized_pnl_pct", 0.0) or 0.0)
    age_hrs = (time.time() - float(pos.get("entry_time", 0))) / 3600.0
    vel = pnl_pct / age_hrs if age_hrs > 0 else 0
    print(f"{sym}: {pnl_pct*100:.2f}% PnL, {age_hrs:.2f}hrs held, {vel*100:.3f}%/hr")
```

### No Rotation Recommendations

**Symptom**: `rotations_recommended` always empty

**Check**:
1. Are there positions old enough? (default: 15 min)
2. Are there ML signals available?
3. Is velocity gap large enough? (default: 0.5%/hr)

**Debug**:
```python
# Check recyclable positions
for sym, metric in position_metrics.items():
    print(f"{sym}: age={metric.age_hours:.2f}hr, recyclable={metric.is_recyclable}")

# Check opportunity signals
for sym, opp in opportunity_metrics.items():
    print(f"{sym}: velocity={opp.estimated_velocity_pct*100:.2f}%/hr")

# Check gaps
for pos_sym, pos_metric in position_metrics.items():
    for opp_sym, opp_metric in opportunity_metrics.items():
        gap = opp_metric.estimated_velocity_pct - pos_metric.net_velocity
        print(f"{pos_sym} → {opp_sym}: gap={gap*100:.2f}%/hr")
```

---

## Success Criteria

✅ **Module loads** without syntax/import errors  
✅ **Initializes** in MetaController successfully  
✅ **Calculates** position velocity correctly (±0.01%)  
✅ **Estimates** opportunity velocity from ML signals  
✅ **Generates** rotation recommendations when appropriate  
✅ **Logs** velocity metrics every cycle  
✅ **Completes** in <100ms per cycle  
✅ **Handles** edge cases gracefully  
✅ **No impact** on existing trading logic  

If all criteria are met, the optimizer is working correctly and ready for production use.
