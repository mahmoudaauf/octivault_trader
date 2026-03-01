# Capital Governor - Best Practice Decision Tree Implementation Guide

## Overview

This document describes the implementation of the **Best Practice Decision Tree for Symbol Rotation & Position Sizing** based on account equity bracket.

**Status**: ✅ IMPLEMENTED in `core/capital_governor.py`

---

## Best Practice Decision Tree

### Decision Logic

```
If equity < $500:
    Fix 1–2 core pairs (no rotation)
    Allow 1 rotating slot max
    → MICRO Bracket

Else if equity < $2000:
    Fix 2 core pairs
    Allow 1 rotating slot
    → SMALL Bracket

Else if equity < $10000:
    Fix 3 core pairs
    Allow 5 rotating slots
    → MEDIUM Bracket

Else (equity >= $10000):
    Fix 5 core pairs
    Allow 5-10 rotating slots
    → LARGE Bracket
```

---

## Capital Brackets

### MICRO Bracket (< $500 USDT)

**Objective**: Learning phase, unblock engine, enable compounding

**Position Limits**:
- Max Active Symbols: **2**
- Core Pairs (no rotation): **2**
- Rotating Slots: **0** (NO ROTATION)
- Max Concurrent Positions: **1**
- Symbol Replacement Multiplier: **2.0** (100% improvement needed - unreachable)
- Soft Lock Duration: **24 hours** (prevent rotation)

**Position Sizing**:
- Quote per Position: **$12.00**
- Max per Symbol: **$24.00**
- Portfolio Allocation: **5%** per position
- EV Multiplier: **1.4** (permissive gate)
- Profit Lock: **Disabled**

**Use Case**:
```python
nav = 350.0  # Your current equity
from core.capital_governor import CapitalGovernor

gov = CapitalGovernor(config)
limits = gov.get_position_limits(nav)
# Returns:
# {
#   "bracket": "micro",
#   "max_active_symbols": 2,
#   "core_pairs": 2,
#   "max_rotating_slots": 0,
#   "allow_rotation": False,
#   ...
# }

# Sizing
sizing = gov.get_position_sizing(nav)
# Returns:
# {
#   "quote_per_position": 12.0,
#   "enable_profit_lock": False,
#   "ev_multiplier": 1.4,
#   ...
# }
```

**Recommendation**: Pick your 2 best symbols (highest volume/stability) and focus on learning.

---

### SMALL Bracket ($500-$2000 USDT)

**Objective**: Stable growth with limited rotation

**Position Limits**:
- Max Active Symbols: **5**
- Core Pairs: **2**
- Rotating Slots: **1**
- Max Concurrent Positions: **2**
- Symbol Replacement Multiplier: **1.50** (50% improvement needed)
- Soft Lock Duration: **1 hour**

**Position Sizing**:
- Quote per Position: **$15.00**
- Max per Symbol: **$30.00**
- Portfolio Allocation: **3%** per position
- EV Multiplier: **1.6**
- Profit Lock: **Disabled** (still learning)

**Composition**: 2 core + 1 rotating = 3 active symbols at a time

---

### MEDIUM Bracket ($2000-$10000 USDT)

**Objective**: Scaling phase with institutional discipline emerging

**Position Limits**:
- Max Active Symbols: **10**
- Core Pairs: **3**
- Rotating Slots: **5**
- Max Concurrent Positions: **3**
- Symbol Replacement Multiplier: **1.25** (25% improvement needed)
- Soft Lock Duration: **30 minutes**

**Position Sizing**:
- Quote per Position: **$25.00**
- Max per Symbol: **$75.00**
- Portfolio Allocation: **2%** per position
- EV Multiplier: **1.8**
- Profit Lock: **Enabled** (start harvesting edge)

**Composition**: 3 core + 5 rotating = 8 active symbols

---

### LARGE Bracket (≥ $10000 USDT)

**Objective**: Institutional diversification with strict discipline

**Position Limits**:
- Max Active Symbols: **20**
- Core Pairs: **5**
- Rotating Slots: **10**
- Max Concurrent Positions: **5**
- Symbol Replacement Multiplier: **1.10** (10% improvement needed)
- Soft Lock Duration: **5 minutes**

**Position Sizing**:
- Quote per Position: **$50.00**
- Max per Symbol: **$150.00**
- Portfolio Allocation: **1%** per position
- EV Multiplier: **2.0** (strict gate)
- Profit Lock: **Enabled**

**Composition**: 5 core + 10 rotating = 15 active symbols

---

## Integration Points

### 1. In MetaController (Health Check & Arbitration)

```python
from core.capital_governor import CapitalGovernor

class MetaController:
    def __init__(self, config, ...):
        self.capital_governor = CapitalGovernor(config)
        
    async def orchestrate(self, ...):
        # Get current NAV
        nav = self.portfolio_tracker.nav
        
        # Apply capital brackets to position limits
        limits = self.capital_governor.get_position_limits(nav)
        sizing = self.capital_governor.get_position_sizing(nav)
        
        # Use limits for arbitration
        max_concurrent = limits["max_concurrent_positions"]
        allow_rotation = limits["allow_rotation"]
        
        # Check if we should block a BUY based on position limits
        if len(current_positions) >= max_concurrent:
            reason = f"MAX_POSITIONS_REACHED ({max_concurrent})"
            return {...}  # Reject BUY
```

### 2. In SymbolRotationManager (Rotation Eligibility)

```python
from core.capital_governor import CapitalGovernor

class SymbolRotationManager:
    def __init__(self, config):
        self.capital_governor = CapitalGovernor(config)
        
    def can_rotate_symbol(self, current, candidate, ...):
        nav = self.shared_state.nav  # Get current equity
        
        # Check if bracket allows rotation at all
        if self.capital_governor.should_restrict_rotation(nav):
            logger.info(f"[RotationMgr] Rotation disabled (MICRO bracket)")
            return False
        
        # Apply bracket-specific multiplier
        limits = self.capital_governor.get_position_limits(nav)
        multiplier = limits["symbol_replacement_multiplier"]
        
        # Now use multiplier for score check
        threshold = current_score * multiplier
        return candidate_score > threshold
```

### 3. In PositionManager (Position Sizing & Allocation)

```python
from core.capital_governor import CapitalGovernor

class PositionManager:
    def __init__(self, config, ...):
        self.capital_governor = CapitalGovernor(config)
        
    def calculate_position_size(self, nav, symbol, ...):
        sizing = self.capital_governor.get_position_sizing(nav, symbol)
        
        quote_per_position = sizing["quote_per_position"]
        max_per_symbol = sizing["max_per_symbol"]
        
        # Respect bracket limits
        position_size = min(quote_per_position, nav * sizing["portfolio_allocation_pct"] / 100)
        
        return position_size
    
    def is_valid_position(self, nav, symbol, is_core=False):
        return self.capital_governor.validate_symbol_for_bracket(nav, symbol, is_core)
```

### 4. In ConfigManager (Dynamic Profile Selection)

```python
def get_capital_profile(self, nav: float) -> dict:
    """
    NEW APPROACH: Use CapitalGovernor for intelligent bracket selection.
    """
    from core.capital_governor import CapitalGovernor
    
    governor = CapitalGovernor(self)
    limits = governor.get_position_limits(nav)
    sizing = governor.get_position_sizing(nav)
    
    # Build profile from governor recommendations
    profile = {
        "ev_multiplier": sizing["ev_multiplier"],
        "default_planned_quote": sizing["quote_per_position"],
        "min_order_usdt": sizing["min_order_usdt"],
        "enable_profit_lock": sizing["enable_profit_lock"],
        "max_active_symbols": limits["max_active_symbols"],
        "max_concurrent_positions": limits["max_concurrent_positions"],
        "description": limits["reason"],
    }
    
    logger.info(f"[Profile] {limits['reason']}: {profile}")
    return profile
```

---

## Usage Examples

### Example 1: Small Account ($350 USDT) - MICRO Bracket

```python
from core.capital_governor import CapitalGovernor

config = Config()
gov = CapitalGovernor(config)

nav = 350.0

# Get position limits
limits = gov.get_position_limits(nav)
print(f"Max active symbols: {limits['max_active_symbols']}")         # 2
print(f"Max rotating slots: {limits['max_rotating_slots']}")         # 0
print(f"Allow rotation: {limits['allow_rotation']}")                 # False

# Get sizing
sizing = gov.get_position_sizing(nav)
print(f"Quote per position: ${sizing['quote_per_position']}")        # $12.00
print(f"EV Multiplier: {sizing['ev_multiplier']}")                   # 1.4

# Display full report
print(gov.format_limits_for_display(nav))
# Output:
# [CapitalGovernor Report]
#   Equity: $350.00
#   Bracket: MICRO
#   Position Limits:
#     - Max Active Symbols: 2
#     - Core Pairs: 2
#     - Rotating Slots: 0
#     - Max Concurrent Positions: 1
#     - Rotation Allowed: False
#   Position Sizing:
#     - Per Position: $12.00
#     - Portfolio Allocation: 5.0%
#     - EV Multiplier: 1.4x
#     - Profit Lock: False
#   Reason: MICRO_BRACKET: Focus on 2 core pairs for learning
```

### Example 2: Growing Account ($1500 USDT) - SMALL Bracket

```python
nav = 1500.0

limits = gov.get_position_limits(nav)
print(f"Max active symbols: {limits['max_active_symbols']}")         # 5
print(f"Core pairs: {limits['core_pairs']}")                         # 2
print(f"Max rotating slots: {limits['max_rotating_slots']}")         # 1
print(f"Allow rotation: {limits['allow_rotation']}")                 # True

sizing = gov.get_position_sizing(nav)
print(f"Quote per position: ${sizing['quote_per_position']}")        # $15.00
print(f"Portfolio allocation: {sizing['portfolio_allocation_pct']}%") # 3.0%
```

### Example 3: Institutional Account ($15000 USDT) - LARGE Bracket

```python
nav = 15000.0

limits = gov.get_position_limits(nav)
print(f"Max active symbols: {limits['max_active_symbols']}")         # 20
print(f"Max rotating slots: {limits['max_rotating_slots']}")         # 10
print(f"Max concurrent positions: {limits['max_concurrent_positions']}")  # 5

sizing = gov.get_position_sizing(nav)
print(f"Quote per position: ${sizing['quote_per_position']}")        # $50.00
print(f"EV Multiplier: {sizing['ev_multiplier']}")                   # 2.0
print(f"Profit Lock: {sizing['enable_profit_lock']}")                # True
```

---

## Best Practices for Your Account

### For MICRO Bracket ($<500)

1. **Pick 2 core pairs**: Choose highest volume/most stable pairs (e.g., BTCUSDT, ETHUSDT)
2. **No rotation**: Don't rotate symbols at all
3. **1 position at a time**: Stay focused
4. **Learn the system**: Focus on understanding your alpha, not on profit
5. **Avoid overtrading**: Stick to signal quality > frequency
6. **Expected PnL**: Small per trade ($5-20), but consistent learning

**Sample Configuration**:
```
CAPITAL_PROFILE_NAV_THRESHOLD=500
BOOTSTRAP_SOFT_LOCK_ENABLED=True
BOOTSTRAP_SOFT_LOCK_DURATION_SEC=86400  # 24 hours for MICRO
SYMBOL_REPLACEMENT_MULTIPLIER=2.0       # Unreachable (100% needed)
MAX_ACTIVE_SYMBOLS=2
MAX_POSITIONS_TOTAL=1
```

### For SMALL Bracket ($500-$2000)

1. **2 core + 1 rotating**: Stabilize with 2 core, experiment with 1 rotating
2. **Gentle rotation**: Only rotate if new symbol is 50% better
3. **2 concurrent positions**: Slightly more aggressive
4. **Enable learning**: Still focus on edge validation
5. **Monitor rotation**: Track if rotating symbols actually improve results

**Sample Configuration**:
```
BOOTSTRAP_SOFT_LOCK_DURATION_SEC=3600   # 1 hour
SYMBOL_REPLACEMENT_MULTIPLIER=1.50      # Require 50% improvement
MAX_ACTIVE_SYMBOLS=5
MAX_POSITIONS_TOTAL=2
```

### For MEDIUM+ Brackets ($2000+)

1. **Profit lock enabled**: Start harvesting your edge
2. **More rotation**: Can rotate if 10-25% improvement
3. **Diversification**: 3-5 core pairs + multiple rotating slots
4. **Institutional gates**: Full P9 constraints applied
5. **Risk management**: Use strict position sizing

---

## Testing the Implementation

### Unit Test

```python
import unittest
from core.capital_governor import CapitalGovernor
from core.config import Config

class TestCapitalGovernor(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.gov = CapitalGovernor(self.config)
    
    def test_micro_bracket_limits(self):
        nav = 350.0
        limits = self.gov.get_position_limits(nav)
        
        self.assertEqual(limits["bracket"], "micro")
        self.assertEqual(limits["max_active_symbols"], 2)
        self.assertEqual(limits["max_rotating_slots"], 0)
        self.assertFalse(limits["allow_rotation"])
    
    def test_small_bracket_limits(self):
        nav = 1500.0
        limits = self.gov.get_position_limits(nav)
        
        self.assertEqual(limits["bracket"], "small")
        self.assertEqual(limits["max_active_symbols"], 5)
        self.assertEqual(limits["max_rotating_slots"], 1)
        self.assertTrue(limits["allow_rotation"])
    
    def test_sizing_scales_with_nav(self):
        micro_sizing = self.gov.get_position_sizing(350.0)
        small_sizing = self.gov.get_position_sizing(1500.0)
        large_sizing = self.gov.get_position_sizing(15000.0)
        
        self.assertLess(
            micro_sizing["quote_per_position"],
            small_sizing["quote_per_position"]
        )
        self.assertLess(
            small_sizing["quote_per_position"],
            large_sizing["quote_per_position"]
        )

if __name__ == "__main__":
    unittest.main()
```

### Integration Test

```python
# In meta_controller.py or test suite
async def test_capital_governor_integration():
    config = Config()
    governor = CapitalGovernor(config)
    
    # Simulate different equity levels
    for nav in [350.0, 1500.0, 5000.0, 15000.0]:
        limits = governor.get_position_limits(nav)
        sizing = governor.get_position_sizing(nav)
        
        # Verify consistency
        assert limits["core_pairs"] + limits["max_rotating_slots"] <= limits["max_active_symbols"]
        assert sizing["quote_per_position"] <= nav * sizing["portfolio_allocation_pct"] / 100 * 10
        
        print(f"NAV ${nav}: {limits['reason']}")
    
    print("✅ All capital governor tests passed")
```

---

## Logging & Monitoring

### Key Log Lines

```python
[CapitalGovernor] Initialized with brackets: MICRO=<$500, SMALL=$500-$2000, MEDIUM=$2000-$10000, LARGE>=$10000
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max positions, rotation=False
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[CapitalGovernor:CorePairs] NAV=$350.00 → Recommended 2 core pairs: BTCUSDT, ETHUSDT
[WHY_NO_TRADE] symbol=BNBUSDT reason=SYMBOL_NOT_ALLOWED_IN_BRACKET details=rotating_slot_exhausted
```

### Expected Output (MICRO Account)

```
[AppContext] P7: Starting PnLCalculator...
[CapitalGovernor] Initialized with brackets: MICRO=<$500, SMALL=$500-$2000, MEDIUM=$2000-$10000, LARGE>=$10000
[Portfolio] NAV=350.00, Equity gain=0.00, Unrealized PnL=-0.02
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max position, rotation=False
[MetaController] Health check: ExecutionManager=running → ✅ READY
[Signal] BTCUSDT: BUY signal strength=0.87 (0.87 > 0.75 threshold) → EXECUTE
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[Execution] BTCUSDT: BUY 0.00025 @ 70000.00 USDT = $17.50 (✓ valid)
✅ READY for trading
```

---

## Migration Path

### Step 1: Implement CapitalGovernor (DONE ✅)
- Create `core/capital_governor.py`
- Define 4 bracket types (MICRO, SMALL, MEDIUM, LARGE)
- Implement limits and sizing methods

### Step 2: Integrate with MetaController
- Add `self.capital_governor = CapitalGovernor(config)` in `__init__`
- Use `get_position_limits()` in health check and arbitration
- Check `max_concurrent_positions` before allowing BUY

### Step 3: Integrate with SymbolRotationManager
- Add rotation check: `capital_governor.should_restrict_rotation(nav)`
- Use bracket-specific multiplier for score evaluation
- Set soft lock duration from `get_position_limits()`

### Step 4: Update PositionManager
- Use `get_position_sizing()` for sizing calculations
- Validate symbols with `validate_symbol_for_bracket()`
- Respect `max_per_symbol` limits

### Step 5: Test & Monitor
- Run unit tests (see Testing section)
- Monitor logs for bracket transitions
- Verify no trades exceed limits
- Track profitability by bracket

---

## Decision Tree Summary

```
NAV < $500 (MICRO)
├─ Max Symbols: 2
├─ Core Pairs: 2
├─ Rotating Slots: 0 ✗ NO ROTATION
├─ Max Concurrent: 1
├─ Position Size: $12/trade
├─ EV Gate: 1.4x (permissive)
└─ Profit Lock: ✗ OFF

NAV $500-$2000 (SMALL)
├─ Max Symbols: 5
├─ Core Pairs: 2
├─ Rotating Slots: 1
├─ Max Concurrent: 2
├─ Position Size: $15/trade
├─ EV Gate: 1.6x
└─ Profit Lock: ✗ OFF

NAV $2000-$10000 (MEDIUM)
├─ Max Symbols: 10
├─ Core Pairs: 3
├─ Rotating Slots: 5
├─ Max Concurrent: 3
├─ Position Size: $25/trade
├─ EV Gate: 1.8x
└─ Profit Lock: ✓ ON

NAV ≥ $10000 (LARGE)
├─ Max Symbols: 20
├─ Core Pairs: 5
├─ Rotating Slots: 10
├─ Max Concurrent: 5
├─ Position Size: $50/trade
├─ EV Gate: 2.0x (strict)
└─ Profit Lock: ✓ ON
```

---

## Questions?

For integration help, refer to the inline docstrings in `core/capital_governor.py` or post in the architecture discussion.
