# Capital Governor Implementation - Complete Summary

**Status**: ✅ **IMPLEMENTED AND READY FOR INTEGRATION**

**Date**: March 1, 2026

**Files Created**:
1. `core/capital_governor.py` (400+ lines, ✅ syntax verified)
2. `CAPITAL_GOVERNOR_GUIDE.md` (Comprehensive integration guide)
3. `CAPITAL_GOVERNOR_QUICK_REF.md` (Quick reference card)

---

## What Was Implemented

### Best Practice Decision Tree

```
If equity < $500:
    Fix 1–2 core pairs (no rotation)
    Allow 1 rotating slot max
    → MICRO Bracket

Else if equity < $2000:
    Allow 2 core + 1 rotating
    → SMALL Bracket

Else if equity < $10000:
    Allow 3 core + 5 rotating
    → MEDIUM Bracket

Else (equity >= $10000):
    Allow 5 core + 10 rotating
    → LARGE Bracket
```

### Capital Governor Module (`core/capital_governor.py`)

A new module that provides:

**Classes**:
- `CapitalBracket` (Enum): MICRO, SMALL, MEDIUM, LARGE
- `CapitalGovernor`: Main class with decision logic

**Key Methods**:
```python
def get_bracket(nav: float) -> CapitalBracket
    # Returns current bracket based on NAV

def get_position_limits(nav: float) -> Dict[str, Any]
    # Returns position limits (max_active_symbols, max_rotating_slots, etc)

def get_position_sizing(nav: float, symbol: str) -> Dict[str, float]
    # Returns position sizing (quote_per_position, ev_multiplier, etc)

def should_restrict_rotation(nav: float) -> bool
    # Check if rotation should be disabled (MICRO bracket only)

def validate_symbol_for_bracket(nav: float, symbol: str, is_core: bool) -> bool
    # Check if symbol is allowed in this bracket

def get_recommended_core_pairs(nav: float, available_symbols: List[str]) -> List[str]
    # Get list of recommended core pairs

def format_limits_for_display(nav: float) -> str
    # Human-readable report for logging
```

---

## Bracket Details

### MICRO Bracket (< $500 USDT)

| Metric | Value |
|--------|-------|
| **Max Active Symbols** | 2 |
| **Core Pairs** | 2 |
| **Rotating Slots** | **0 (NO ROTATION)** |
| **Max Concurrent Positions** | 1 |
| **Position Size** | $12.00 |
| **EV Multiplier** | 1.4 (permissive) |
| **Soft Lock Duration** | 24 hours |
| **Profit Lock** | ❌ Disabled |
| **Reason** | Focus on 2 core pairs for learning |

**Use Case**:
```python
nav = 350.0  # Your current equity
gov = CapitalGovernor(config)

limits = gov.get_position_limits(nav)
# {
#   "max_active_symbols": 2,
#   "max_rotating_slots": 0,      # ← NO rotation!
#   "core_pairs": 2,
#   "allow_rotation": False,       # ← Disabled!
#   ...
# }

sizing = gov.get_position_sizing(nav)
# {
#   "quote_per_position": 12.0,    # ← Small positions
#   "ev_multiplier": 1.4,          # ← Permissive gate
#   "enable_profit_lock": False,   # ← Learning mode
#   ...
# }
```

### SMALL Bracket ($500-$2000 USDT)

| Metric | Value |
|--------|-------|
| **Max Active Symbols** | 5 |
| **Core Pairs** | 2 |
| **Rotating Slots** | **1** |
| **Max Concurrent Positions** | 2 |
| **Position Size** | $15.00 |
| **EV Multiplier** | 1.6 |
| **Soft Lock Duration** | 1 hour |
| **Profit Lock** | ❌ Disabled |
| **Reason** | 2 core + 1 rotating = stable growth |

### MEDIUM Bracket ($2000-$10000 USDT)

| Metric | Value |
|--------|-------|
| **Max Active Symbols** | 10 |
| **Core Pairs** | 3 |
| **Rotating Slots** | **5** |
| **Max Concurrent Positions** | 3 |
| **Position Size** | $25.00 |
| **EV Multiplier** | 1.8 |
| **Soft Lock Duration** | 30 minutes |
| **Profit Lock** | ✅ **ENABLED** |
| **Reason** | 3 core + 5 rotating = scaling phase |

### LARGE Bracket (≥ $10000 USDT)

| Metric | Value |
|--------|-------|
| **Max Active Symbols** | 20 |
| **Core Pairs** | 5 |
| **Rotating Slots** | **10** |
| **Max Concurrent Positions** | 5 |
| **Position Size** | $50.00 |
| **EV Multiplier** | 2.0 (strict) |
| **Soft Lock Duration** | 5 minutes |
| **Profit Lock** | ✅ **ENABLED** |
| **Reason** | 5 core + 10 rotating = institutional diversification |

---

## Integration Points

### 1. **MetaController** (Position Limit Checking)

Location: `core/meta_controller.py` → `orchestrate()` method

**Before**:
```python
# No position limit checking (was relying on simple global cap)
max_concurrent = 2  # Hardcoded
```

**After**:
```python
from core.capital_governor import CapitalGovernor

class MetaController:
    def __init__(self, config, ...):
        self.capital_governor = CapitalGovernor(config)
    
    async def orchestrate(self, ...):
        nav = self.portfolio_tracker.nav
        limits = self.capital_governor.get_position_limits(nav)
        
        # Check position count before BUY
        if len(current_positions) >= limits["max_concurrent_positions"]:
            logger.info(f"[Meta] Position limit reached: {len(current_positions)} >= {limits['max_concurrent_positions']}")
            return {...}  # Reject BUY
        
        # Continue with normal flow
        ...
```

**Expected Log**:
```
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max positions, rotation=False
[Meta] NAV=$350.00 → MICRO bracket (max_positions=1, rotation=False)
```

### 2. **SymbolRotationManager** (Rotation Eligibility)

Location: `core/symbol_rotation.py` → `can_rotate_symbol()` method

**Before**:
```python
def can_rotate_symbol(self, current_symbol, candidate_symbol, current_score, candidate_score):
    # Used fixed multiplier
    threshold = current_score * self.replacement_multiplier  # 1.10 fixed
    return candidate_score > threshold
```

**After**:
```python
from core.capital_governor import CapitalGovernor

class SymbolRotationManager:
    def __init__(self, config):
        self.capital_governor = CapitalGovernor(config)
    
    def can_rotate_symbol(self, current_symbol, candidate_symbol, current_score, candidate_score):
        nav = self.shared_state.nav
        
        # Check if rotation allowed at all (MICRO bracket disables)
        if self.capital_governor.should_restrict_rotation(nav):
            logger.info(f"[RotationMgr] Rotation disabled in MICRO bracket")
            return False
        
        # Use bracket-specific multiplier
        limits = self.capital_governor.get_position_limits(nav)
        multiplier = limits["symbol_replacement_multiplier"]
        
        threshold = current_score * multiplier
        return candidate_score > threshold
```

**Expected Log**:
```
[SymbolRotation] Cannot rotate BTCUSDT → BNBUSDT: rotation disabled in MICRO bracket
[WHY_NO_TRADE] symbol=BNBUSDT reason=SYMBOL_NOT_ALLOWED_IN_BRACKET
```

### 3. **PositionManager** (Position Sizing & Validation)

Location: `core/position_manager.py` (or wherever position sizing happens)

**Before**:
```python
def calculate_position_size(nav, symbol):
    # Fixed sizing
    return 24.0  # Always $24
```

**After**:
```python
from core.capital_governor import CapitalGovernor

class PositionManager:
    def __init__(self, config):
        self.capital_governor = CapitalGovernor(config)
    
    def calculate_position_size(self, nav, symbol, is_core=False):
        # Get bracket-specific sizing
        sizing = self.capital_governor.get_position_sizing(nav, symbol)
        quote_per_position = sizing["quote_per_position"]
        max_per_symbol = sizing["max_per_symbol"]
        
        # Respect bracket limits
        position_size = min(quote_per_position, max_per_symbol)
        
        logger.debug(f"[PositionMgr] {symbol}: ${position_size} (bracket: {self.capital_governor.get_bracket(nav).value})")
        return position_size
    
    def is_valid_position(self, nav, symbol, is_core=False):
        # Validate against bracket
        return self.capital_governor.validate_symbol_for_bracket(nav, symbol, is_core)
```

**Expected Log**:
```
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[PositionMgr] BTCUSDT: $12.00 (bracket: micro)
```

### 4. **Config Manager** (Dynamic Profile Selection)

Location: `core/config.py` → `get_capital_profile()` method

**Optional Enhancement**:
```python
def get_capital_profile(self, nav: float) -> dict:
    """
    Enhanced: Use CapitalGovernor for intelligent bracket selection.
    """
    from core.capital_governor import CapitalGovernor
    
    governor = CapitalGovernor(self)
    limits = governor.get_position_limits(nav)
    sizing = governor.get_position_sizing(nav)
    
    # Build profile from governor
    profile = {
        "ev_multiplier": sizing["ev_multiplier"],
        "default_planned_quote": sizing["quote_per_position"],
        "min_order_usdt": sizing["min_order_usdt"],
        "enable_profit_lock": sizing["enable_profit_lock"],
        "max_active_symbols": limits["max_active_symbols"],
        "description": limits["reason"],
    }
    
    logger.info(f"[Profile] {limits['reason']} → EV×{sizing['ev_multiplier']}")
    return profile
```

---

## Integration Roadmap

### Phase A: Foundation (Today)
- ✅ Create `core/capital_governor.py`
- ✅ Create documentation
- ⏳ Add to MetaController `__init__`
- ⏳ Test bracket classification

### Phase B: MetaController Integration (Next)
- [ ] Add `self.capital_governor = CapitalGovernor(config)` in `__init__`
- [ ] In `orchestrate()`, check position limits before BUY
- [ ] Log bracket info on each arbitration
- [ ] Monitor position count against `max_concurrent_positions`

### Phase C: SymbolRotationManager Integration
- [ ] Add `self.capital_governor = CapitalGovernor(config)` in `__init__`
- [ ] In `can_rotate_symbol()`, check `should_restrict_rotation(nav)` first
- [ ] Use bracket-specific `symbol_replacement_multiplier`
- [ ] Set soft lock duration from governor

### Phase D: PositionManager Integration
- [ ] Use `get_position_sizing()` for all sizing calculations
- [ ] Use `validate_symbol_for_bracket()` for symbol validation
- [ ] Enforce `max_per_symbol` limits
- [ ] Log sizing decisions with bracket info

### Phase E: Testing & Monitoring
- [ ] Unit tests for bracket classification
- [ ] Unit tests for position limit edge cases
- [ ] Integration test: verify limits are enforced
- [ ] Monitor logs for bracket transitions
- [ ] Track compliance with limits

---

## Usage Examples

### Example 1: Get Bracket & Limits

```python
from core.capital_governor import CapitalGovernor
from core.config import Config

config = Config()
gov = CapitalGovernor(config)

# Example: $350 USDT account (MICRO bracket)
nav = 350.0
limits = gov.get_position_limits(nav)

print(f"Bracket: {limits['bracket']}")                      # micro
print(f"Max active symbols: {limits['max_active_symbols']}") # 2
print(f"Max rotating slots: {limits['max_rotating_slots']}")  # 0
print(f"Max concurrent positions: {limits['max_concurrent_positions']}") # 1
print(f"Allow rotation: {limits['allow_rotation']}")          # False
print(f"Replacement multiplier: {limits['symbol_replacement_multiplier']}") # 2.0
```

### Example 2: Check Position Sizing

```python
sizing = gov.get_position_sizing(nav)

print(f"Quote per position: ${sizing['quote_per_position']}")         # $12.00
print(f"Max per symbol: ${sizing['max_per_symbol']}")                 # $24.00
print(f"Portfolio allocation: {sizing['portfolio_allocation_pct']}%") # 5.0%
print(f"EV multiplier: {sizing['ev_multiplier']}")                    # 1.4
print(f"Profit lock enabled: {sizing['enable_profit_lock']}")         # False
```

### Example 3: Check Rotation Eligibility

```python
should_restrict = gov.should_restrict_rotation(nav)
if should_restrict:
    print("Rotation is DISABLED (MICRO bracket)")
else:
    print("Rotation is ALLOWED")
```

### Example 4: Display Full Report

```python
report = gov.format_limits_for_display(nav)
print(report)

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

---

## Testing

### Unit Test Example

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
        micro = self.gov.get_position_sizing(350.0)
        large = self.gov.get_position_sizing(15000.0)
        
        self.assertLess(micro["quote_per_position"], large["quote_per_position"])
        self.assertLess(micro["ev_multiplier"], large["ev_multiplier"])

if __name__ == "__main__":
    unittest.main()
```

**Run Tests**:
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 -m unittest test_capital_governor.TestCapitalGovernor -v
```

---

## Logging & Monitoring

### Key Log Lines to Expect

```
[CapitalGovernor] Initialized with brackets: MICRO=<$500, SMALL=$500-$2000, MEDIUM=$2000-$10000, LARGE>=$10000
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max positions, rotation=False
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[CapitalGovernor:CorePairs] NAV=$350.00 → Recommended 2 core pairs: BTCUSDT, ETHUSDT
[Meta] Health check: ExecutionManager=running, PnLCalculator=running → ✅ READY
[SymbolRotation] Cannot rotate BTCUSDT → BNBUSDT: rotation disabled in MICRO bracket
[WHY_NO_TRADE] symbol=BNBUSDT reason=SYMBOL_NOT_ALLOWED_IN_BRACKET details=micro_bracket_no_rotation
```

### Bracket Transition Logs

When you grow from $500 (MICRO) → $525 (SMALL):
```
[CapitalGovernor:PositionLimits] NAV=$525.00 → small bracket: 5 active symbols (2 core + 1 rotating), 2 max positions, rotation=True
[CapitalGovernor:Sizing] NAV=$525.00: Quote per position = $15.00 (up from $12.00)
[Meta] Bracket upgraded: MICRO → SMALL, enabling rotation with 1.5x multiplier threshold
```

---

## Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `core/capital_governor.py` | Implementation | ✅ Created, syntax verified |
| `CAPITAL_GOVERNOR_GUIDE.md` | Detailed integration guide | ✅ Created |
| `CAPITAL_GOVERNOR_QUICK_REF.md` | Quick reference card | ✅ Created |
| `core/meta_controller.py` | ⏳ Needs integration | Pending |
| `core/symbol_rotation.py` | ⏳ Needs integration | Pending |
| `core/position_manager.py` | ⏳ Needs integration | Pending |

---

## Summary

### What This Solves

✅ **For MICRO accounts** ($<500):
- Prevents overtrading by limiting to 2 symbols, 1 position
- Disables rotation entirely (focus on core pairs)
- Ensures small position sizes ($12) appropriate for capital level
- Uses permissive gates (1.4x) to enable learning

✅ **For SMALL accounts** ($500-2K):
- Allows controlled rotation (1 slot) while keeping 2 core
- Scales position size ($15) appropriately
- Enables growth without recklessness

✅ **For MEDIUM+ accounts** ($2K+):
- Full diversification capabilities (5-10 rotating slots)
- Strict institutional discipline (2.0x EV gates)
- Profit lock enabled to harvest earned edge
- Proper risk management

### Key Benefits

1. **Capital-aware**: Adapts to account size automatically
2. **Learning-focused**: MICRO bracket prioritizes learning over profit
3. **Gradual scaling**: Each bracket allows more complexity as capital grows
4. **Safety-first**: Position limits prevent overcommitment
5. **Profit harvesting**: Profit lock enables at MEDIUM+ bracket
6. **Institutional discipline**: Strict gates for large accounts

---

## Next Steps

1. **Review** this document and the detailed guides
2. **Add to MetaController** (Phase B)
3. **Test bracket transitions** with growing NAV
4. **Monitor logs** for compliance
5. **Scale confidently** knowing your account is protected

---

**Questions?** Check:
- `CAPITAL_GOVERNOR_GUIDE.md` - Detailed integration guide
- `CAPITAL_GOVERNOR_QUICK_REF.md` - Quick reference
- `core/capital_governor.py` - Source code & docstrings
