# Capital Governor - Quick Reference Card

## One-Line Decision Tree

```
IF equity < $500:  Fix 2 core pairs, NO rotation, 1 position max
ELSE IF < $2000:   2 core + 1 rotating, 2 positions max
ELSE IF < $10000:  3 core + 5 rotating, 3 positions max
ELSE:              5 core + 10 rotating, 5 positions max
```

---

## Bracket Comparison Table

| Metric | MICRO (<$500) | SMALL ($500-$2K) | MEDIUM ($2K-$10K) | LARGE (≥$10K) |
|--------|---|---|---|---|
| **Max Active Symbols** | 2 | 5 | 10 | 20 |
| **Core Pairs** | 2 | 2 | 3 | 5 |
| **Rotating Slots** | 0 | 1 | 5 | 10 |
| **Max Concurrent Positions** | 1 | 2 | 3 | 5 |
| **Position Size** | $12 | $15 | $25 | $50 |
| **Portfolio Allocation** | 5% | 3% | 2% | 1% |
| **Rotation Allowed** | ❌ NO | ✅ YES | ✅ YES | ✅ YES |
| **Replacement Multiplier** | 2.0 (100%) | 1.5 (50%) | 1.25 (25%) | 1.1 (10%) |
| **Soft Lock Duration** | 24h | 1h | 30m | 5m |
| **EV Multiplier** | 1.4 | 1.6 | 1.8 | 2.0 |
| **Profit Lock** | ❌ OFF | ❌ OFF | ✅ ON | ✅ ON |

---

## How to Use in Code

### Get Position Limits

```python
from core.capital_governor import CapitalGovernor

gov = CapitalGovernor(config)
nav = 350.0  # Current account equity

limits = gov.get_position_limits(nav)

# Access limits
print(limits["max_active_symbols"])          # 2
print(limits["max_rotating_slots"])          # 0
print(limits["allow_rotation"])              # False
print(limits["max_concurrent_positions"])    # 1
print(limits["symbol_replacement_multiplier"]) # 2.0
```

### Get Position Sizing

```python
sizing = gov.get_position_sizing(nav, symbol="BTCUSDT")

# Access sizing
print(sizing["quote_per_position"])    # 12.0
print(sizing["max_per_symbol"])        # 24.0
print(sizing["ev_multiplier"])         # 1.4
print(sizing["enable_profit_lock"])    # False
```

### Check if Rotation is Allowed

```python
if gov.should_restrict_rotation(nav):
    # Rotation is disabled (MICRO bracket)
    pass
else:
    # Rotation is allowed
    pass
```

### Validate Symbol for Bracket

```python
# Core pairs are always allowed
valid = gov.validate_symbol_for_bracket(nav, "BTCUSDT", is_core=True)  # True

# Non-core symbols check against bracket
valid = gov.validate_symbol_for_bracket(nav, "BNBUSDT", is_core=False)
if nav < 500:  # MICRO bracket
    # valid = False (no rotation allowed)
else:  # SMALL+
    # valid = True (rotation allowed)
```

### Display Full Report

```python
print(gov.format_limits_for_display(nav))
# Outputs:
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

## Integration Checklist

### MetaController

- [ ] Import `CapitalGovernor` in `__init__`
- [ ] Add `self.capital_governor = CapitalGovernor(config)`
- [ ] In health check: use `get_position_limits()` to validate position count
- [ ] In arbitration: check `max_concurrent_positions` before BUY
- [ ] Log: `[Meta] NAV=$%.2f → %s bracket (max_positions=%d, rotation=%s)`

### SymbolRotationManager

- [ ] Import `CapitalGovernor` in `__init__`
- [ ] Add `self.capital_governor = CapitalGovernor(config)`
- [ ] In `can_rotate_symbol()`: check `should_restrict_rotation(nav)` first
- [ ] Apply bracket-specific `symbol_replacement_multiplier` from `get_position_limits()`
- [ ] Set `soft_lock_duration` from `get_position_limits()`

### PositionManager

- [ ] Import `CapitalGovernor` in `__init__`
- [ ] Add `self.capital_governor = CapitalGovernor(config)`
- [ ] In sizing: use `get_position_sizing(nav, symbol)["quote_per_position"]`
- [ ] In validation: use `validate_symbol_for_bracket(nav, symbol, is_core)`
- [ ] Enforce `max_per_symbol` limit from sizing

### Config

- [ ] Optional: Update `get_capital_profile()` to use governor recommendations
- [ ] Log bracket transitions when NAV crosses thresholds

---

## Key Design Principles

1. **MICRO accounts are learning accounts**
   - No rotation (focus on 2 pairs)
   - Small position size ($12)
   - Permissive gates (1.4x EV)
   - No profit lock (priority = learning)

2. **SMALL accounts are growth accounts**
   - Limited rotation (1 rotating slot)
   - Slightly larger positions ($15)
   - Higher gates (1.6x EV)
   - Still no profit lock

3. **MEDIUM accounts start professionalization**
   - Multiple rotating slots (5)
   - Standard sizing ($25)
   - Strict gates (1.8x EV)
   - **Profit lock ENABLED**

4. **LARGE accounts are institutional**
   - Full diversification (5+10 slots)
   - Large positions ($50)
   - Strictest gates (2.0x EV)
   - Full P9 constraints

---

## Common Questions

### Q: How does this affect my trading?

**A**: 
- **MICRO**: You trade 1 pair at a time, no surprises, pure learning
- **SMALL**: You can experiment with 1 rotating pair while keeping 2 stable
- **MEDIUM**: You have room to diversify while maintaining discipline
- **LARGE**: You get full power with institutional safeguards

### Q: What if I'm at the threshold ($500)?

**A**: Governor uses `<` so $500.00 exactly is SMALL bracket. You need $500.01+ to cross. Consider it a soft boundary. You can request a manual override if you're close.

### Q: Can I override the bracket?

**A**: Yes, in `.env`:
```
# Force a bracket regardless of NAV
FORCE_CAPITAL_BRACKET=SMALL  # micro, small, medium, large
```

### Q: How often is the bracket recalculated?

**A**: Every time `get_position_limits()` or `get_position_sizing()` is called. Typically:
- Once per P7/P8 phase (on startup)
- On every arbitration call (MetaController)
- On every position size request

### Q: Do my existing positions get closed if I drop below bracket threshold?

**A**: No. The governor only prevents **new** positions from exceeding the limit. Existing positions are left alone for graceful degradation.

### Q: What's the "symbol_replacement_multiplier"?

**A**: It's the quality bar for rotation. For example:
- MICRO: 2.0 means candidate must score **2x current** (impossible, no rotation)
- SMALL: 1.5 means candidate must score **50% higher** than current
- LARGE: 1.1 means candidate must score **10% higher** than current

---

## Example: Growing from MICRO to SMALL

**Day 1** (NAV = $350):
```
[CapitalGovernor] MICRO bracket
- Active symbols: BTCUSDT, ETHUSDT
- Max positions: 1
- Position size: $12
- No rotation allowed
```

**After 5 profitable trades** (NAV = $520):
```
[CapitalGovernor] Bracket changed: MICRO → SMALL
- Max active symbols: 5 (up from 2)
- Core pairs: 2 (BTCUSDT, ETHUSDT)
- Rotating slots: 1 (can now add 1 experimental pair)
- Max positions: 2 (up from 1)
- Position size: $15 (up from $12)
- Rotation allowed: YES (with 50% improvement threshold)
```

**You now have:**
1. 2 trusted core pairs (BTCUSDT, ETHUSDT)
2. 1 experimental rotating slot (can test new pairs)
3. More capital per position ($15 instead of $12)
4. Can have 2 positions open simultaneously

---

## Logging Examples

### MICRO Account Starting Up

```
[CapitalGovernor] Initialized with brackets: MICRO=<$500, SMALL=$500-$2000, MEDIUM=$2000-$10000, LARGE>=$10000
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max positions, rotation=False
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[CapitalGovernor:CorePairs] NAV=$350.00 → Recommended 2 core pairs: BTCUSDT, ETHUSDT
[Meta] Health check: ExecutionManager=running, PnLCalculator=running → ✅ READY
```

### Rotation Attempt in MICRO

```
[SymbolRotation] Cannot rotate BTCUSDT → BNBUSDT: rotation disabled in MICRO bracket
[WHY_NO_TRADE] symbol=BNBUSDT reason=SYMBOL_NOT_ALLOWED_IN_BRACKET details=micro_bracket_no_rotation
```

### Bracket Transition ($500+ NAV)

```
[CapitalGovernor:PositionLimits] NAV=$525.00 → small bracket: 5 active symbols (2 core + 1 rotating), 2 max positions, rotation=True
[CapitalGovernor:Sizing] NAV=$525.00: Quote per position = $15.00 (up from $12.00)
[Meta] Bracket upgraded: MICRO → SMALL, enabling rotation with 1.5x multiplier
```

---

## Files Reference

- **Implementation**: `core/capital_governor.py` (400+ lines)
- **Guide**: `CAPITAL_GOVERNOR_GUIDE.md` (this directory)
- **Quick Ref**: `CAPITAL_GOVERNOR_QUICK_REF.md` (you are here)

---

## TL;DR

| Your Equity | Max Symbols | Core | Rotating | Position Size | Can Rotate? |
|---|---|---|---|---|---|
| < $500 | 2 | 2 | 0 | $12 | ❌ NO |
| $500-2K | 5 | 2 | 1 | $15 | ✅ YES (hard) |
| $2K-10K | 10 | 3 | 5 | $25 | ✅ YES (medium) |
| ≥ $10K | 20 | 5 | 10 | $50 | ✅ YES (easy) |

That's it! Use `CapitalGovernor` and let it do the rest.
