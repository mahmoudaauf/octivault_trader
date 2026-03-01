# Capital Governor - Master Index

**Status**: ✅ **FULLY IMPLEMENTED & COMMITTED**

**Commit**: `bc7614c` (Pushed to main)

**Date**: March 1, 2026

---

## Quick Navigation

### 🚀 Just Getting Started? Start Here
1. **[CAPITAL_GOVERNOR_QUICK_REF.md](CAPITAL_GOVERNOR_QUICK_REF.md)** ← **READ FIRST** (5 min)
   - One-liner decision tree
   - Bracket comparison table
   - Your account's bracket immediately
   - Common Q&A

### 📖 Want Deep Understanding?
2. **[CAPITAL_GOVERNOR_GUIDE.md](CAPITAL_GOVERNOR_GUIDE.md)** ← **Comprehensive Guide** (15 min)
   - Each bracket explained in detail
   - Integration code examples
   - Best practices for your bracket
   - Testing & monitoring setup

### 🛠️ Ready to Integrate?
3. **[CAPITAL_GOVERNOR_IMPLEMENTATION.md](CAPITAL_GOVERNOR_IMPLEMENTATION.md)** ← **Integration Guide** (20 min)
   - Complete architecture overview
   - 5-phase integration roadmap
   - Code examples for each component
   - Testing checklist

### 💻 Need the Code?
4. **[core/capital_governor.py](core/capital_governor.py)** ← **Source Code** (400+ lines)
   - Main CapitalGovernor class
   - All bracket logic
   - Detailed docstrings
   - Ready to integrate

---

## What Is Capital Governor?

Capital Governor implements the **Best Practice Decision Tree for Capital-Aware Trading**:

```
If equity < $500:
    Fix 1–2 core pairs (no rotation)
    Allow 1 rotating slot max
Else:
    Allow 5–10 rotating symbols (scaled by bracket)
```

It automatically adjusts:
- ✅ Position limits (max concurrent positions, max symbols)
- ✅ Position sizing ($12-$50 per trade based on capital)
- ✅ Rotation eligibility (disabled for small accounts)
- ✅ EV multiplier gates (1.4x to 2.0x based on bracket)
- ✅ Profit lock behavior (disabled for learning, enabled for profit)

---

## Your Account's Bracket

### **Current**: MICRO ($<500)

| Setting | Your Limit |
|---------|-----------|
| **Max Symbols** | 2 |
| **Core Pairs** | 2 (both core, no rotation) |
| **Rotating Slots** | 0 (❌ NO rotation) |
| **Max Positions** | 1 at a time |
| **Position Size** | $12.00 per trade |
| **EV Multiplier** | 1.4x (permissive for learning) |
| **Profit Lock** | OFF (accumulate, don't lock) |
| **Mode** | **Learning Phase** |

**Your Focus**: Pick 2 great symbols (BTCUSDT, ETHUSDT). Trade them over and over. Deep learning = faster edge validation.

### **Next Tier**: SMALL ($500-$2000)

When you reach $500+:
- Max symbols increases to 5
- Can add 1 rotating slot
- Positions increase to 2 max
- Position size $15 per trade
- Still learning (no profit lock)
- Rotation enabled (50% improvement needed)

### **Future**: MEDIUM ($2000-$10000) & LARGE (≥$10000)

Full scaling, diversification, profit lock, institutional gates.

---

## How to Use in Your Code

### Basic Usage

```python
from core.capital_governor import CapitalGovernor

# Initialize once
gov = CapitalGovernor(config)

# Get your bracket at any time
nav = 350.0  # Current account equity
limits = gov.get_position_limits(nav)
sizing = gov.get_position_sizing(nav)

print(limits["max_concurrent_positions"])  # 1
print(sizing["quote_per_position"])        # $12.00
```

### Integration Points

**MetaController** (Block BUYs if position limit reached):
```python
limits = self.capital_governor.get_position_limits(nav)
if len(current_positions) >= limits["max_concurrent_positions"]:
    return {"action": "REJECT", "reason": "Position limit reached"}
```

**SymbolRotationManager** (Disable rotation for MICRO):
```python
if self.capital_governor.should_restrict_rotation(nav):
    return False  # Rotation not allowed
```

**PositionManager** (Use bracket-specific sizing):
```python
sizing = self.capital_governor.get_position_sizing(nav)
position_size = sizing["quote_per_position"]  # $12.00 for MICRO
```

---

## Documentation Map

| Document | Purpose | Read Time | Audience |
|----------|---------|-----------|----------|
| **QUICK_REF.md** | Fast facts, tables, checklists | 5 min | Everyone |
| **GUIDE.md** | Detailed integration walkthrough | 15 min | Developers |
| **IMPLEMENTATION.md** | Complete architecture & roadmap | 20 min | Architects |
| **capital_governor.py** | Source code & docstrings | 20 min | Developers |
| **This file** | Navigation index | 3 min | Everyone |

---

## Integration Phases

### ✅ Phase A: Foundation (DONE)
- ✅ Implement `core/capital_governor.py` (400+ lines)
- ✅ Create documentation (4 files)
- ✅ Verify syntax (0 errors)
- ✅ Commit & push (bc7614c)

### ⏳ Phase B: MetaController Integration (NEXT)
- [ ] Add `self.capital_governor = CapitalGovernor(config)`
- [ ] Check position limits before BUY
- [ ] Estimated: 30 min, 20 lines of code

### ⏳ Phase C: SymbolRotationManager Integration
- [ ] Check `should_restrict_rotation(nav)`
- [ ] Use bracket-specific multiplier
- [ ] Estimated: 30 min, 10 lines of code

### ⏳ Phase D: PositionManager Integration
- [ ] Use `get_position_sizing()` everywhere
- [ ] Validate with `validate_symbol_for_bracket()`
- [ ] Estimated: 1 hour, 30 lines of code

### ⏳ Phase E: Testing & Monitoring
- [ ] Unit tests (examples provided)
- [ ] Monitor logs for bracket transitions
- [ ] Estimated: 1 hour

**Total Integration Time**: ~3 hours (spread across phases)

---

## Key Concepts

### Capital Bracket
Your account falls into one of 4 brackets based on NAV:
- **MICRO**: < $500 (you are here)
- **SMALL**: $500-$2000
- **MEDIUM**: $2000-$10000
- **LARGE**: ≥ $10000

Each bracket has different limits and sizing.

### Position Limits
**Constraints on how many trades you can have open**:
- `max_active_symbols`: Total symbols you track (2 for MICRO)
- `max_rotating_slots`: How many can rotate (0 for MICRO)
- `max_concurrent_positions`: Open positions at once (1 for MICRO)
- `allow_rotation`: Can you rotate? (NO for MICRO)

### Position Sizing
**How much capital per trade**:
- `quote_per_position`: Recommended size ($12 for MICRO)
- `max_per_symbol`: Maximum per symbol ($24 for MICRO)
- `ev_multiplier`: How strict the gates are (1.4x for MICRO)
- `enable_profit_lock`: Lock winning trades? (NO for MICRO)

### Symbol Replacement Multiplier
**How much better a candidate must be to rotate**:
- **MICRO**: 2.0 (100% better needed = impossible, no rotation)
- **SMALL**: 1.5 (50% better)
- **MEDIUM**: 1.25 (25% better)
- **LARGE**: 1.1 (10% better)

---

## Common Questions

### Q: What if I'm exactly at $500?
**A**: Brackets use `<` so $500.00 is SMALL. You need $500.01+ to cross. The boundary is soft - you can request an override if you're close.

### Q: When is the bracket recalculated?
**A**: Every time you call `get_position_limits()` or `get_position_sizing()`. In practice, this happens:
- Once per startup (P7/P8)
- On every arbitration call (MetaController)
- On every position sizing request

### Q: What if I drop below my bracket threshold?
**A**: Existing positions aren't closed. Only new positions are blocked. This allows graceful degradation if your account shrinks.

### Q: Can I override my bracket?
**A**: Yes, via `.env`:
```
FORCE_CAPITAL_BRACKET=SMALL  # Override to SMALL (micro, small, medium, large)
```

### Q: Why can't MICRO accounts rotate?
**A**: 
1. Learning phase: You should master 2 symbols before trying 5
2. Risk: Rotation analysis adds complexity you don't need yet
3. Capital protection: Prevent distraction from core edge
4. Simplicity: Clear focus = faster learning

### Q: What happens at $2000+?
**A**: Profit lock gets enabled. Instead of compounding all profits, you lock them in. This transitions from "learning" to "harvesting."

---

## For Your Account Right Now ($350 MICRO)

### Do This
✅ Pick 2 symbols (BTCUSDT, ETHUSDT recommended)
✅ Trade them exclusively 
✅ Position size: $12 per trade
✅ Study what works across 30-50 trades
✅ Focus on consistency, not compounding

### Don't Do This
❌ Try to rotate to other symbols (disabled)
❌ Open 2 positions at once (max is 1)
❌ Trade more than $12 per position (bracket limit)
❌ Chase other symbols (focus on your 2)
❌ Lock profits (learning phase = accumulate)

### Expected Behavior
- BUY signal comes → check position count
- If 0 positions: ✅ EXECUTE (allowed)
- If 1 position: ❌ BLOCK (position limit reached)
- Try to rotate: ❌ BLOCKED (rotation disabled in MICRO)
- Request $24 position: ❌ REJECTED (size limit)

---

## Integration Checklist

### MetaController
- [ ] Import CapitalGovernor
- [ ] Initialize in __init__
- [ ] Get position limits in orchestrate()
- [ ] Check len(positions) >= max before BUY
- [ ] Test: Trade, verify 1 position limit enforced

### SymbolRotationManager
- [ ] Import CapitalGovernor
- [ ] Initialize in __init__
- [ ] Check should_restrict_rotation(nav)
- [ ] Use bracket multiplier
- [ ] Test: Try to rotate, verify it's blocked

### PositionManager
- [ ] Import CapitalGovernor
- [ ] Initialize in __init__
- [ ] Use get_position_sizing() for all sizes
- [ ] Use validate_symbol_for_bracket()
- [ ] Test: Sizing $12, validation blocks non-core

### Config (Optional)
- [ ] Update get_capital_profile()
- [ ] Use governor recommendations
- [ ] Dynamic profile selection
- [ ] Commit: Integration complete

### Testing
- [ ] Run unit tests
- [ ] Monitor [CapitalGovernor] logs
- [ ] Verify limits enforced
- [ ] Test bracket transitions
- [ ] Commit: All tests passing

---

## Expected Logs

When you run `python3 main_phased.py`:

```
[CapitalGovernor] Initialized with brackets: MICRO=<$500, SMALL=$500-$2000, MEDIUM=$2000-$10000, LARGE>=$10000
[CapitalGovernor:PositionLimits] NAV=$350.00 → micro bracket: 2 active symbols (2 core + 0 rotating), 1 max positions, rotation=False
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position, EV×1.4, profit_lock=False
[Meta] NAV=$350.00 → MICRO bracket (max_positions=1, rotation=False)
[AppContext] P7: PnLCalculator status → Running ✅
[Main] ✅ Runtime plane is live (P9)
```

When you try to trade:

```
[BUY] BTCUSDT: Signal strength 0.87 → EXECUTE
[CapitalGovernor:Sizing] NAV=$350.00 → BTCUSDT: $12.00 per position
[Execution] BTCUSDT: BUY 0.00025 @ 70000.00 = $12.00 ✅
[Meta] Position opened: 1/1 (at limit)

[BUY] ETHUSDT: Signal strength 0.92 → BLOCKED
[Meta] Position limit reached: 1 >= 1 (MICRO bracket)
[WHY_NO_TRADE] symbol=ETHUSDT reason=POSITION_LIMIT_REACHED
```

When you try to rotate (MICRO):

```
[SymbolRotation] Cannot rotate BTCUSDT → BNBUSDT
[SymbolRotation] Rotation disabled: MICRO bracket
[WHY_NO_TRADE] symbol=BNBUSDT reason=SYMBOL_NOT_ALLOWED_IN_BRACKET
```

---

## File Locations

```
octivault_trader/
├── core/
│   └── capital_governor.py ............................ Main implementation
├── CAPITAL_GOVERNOR_QUICK_REF.md ...................... Quick reference
├── CAPITAL_GOVERNOR_GUIDE.md .......................... Detailed guide
├── CAPITAL_GOVERNOR_IMPLEMENTATION.md ................ Complete architecture
└── CAPITAL_GOVERNOR_INDEX.md .......................... This file (navigation)
```

---

## Learning Path

### 1. Understand Your Bracket (5 min)
Read: `CAPITAL_GOVERNOR_QUICK_REF.md`
- Learn: You're in MICRO bracket
- Limits: 2 symbols, 1 position, $12 per trade, NO rotation
- Action: Identify your 2 core symbols

### 2. Deep Dive Integration (15 min)
Read: `CAPITAL_GOVERNOR_GUIDE.md`
- Learn: How each bracket works
- Study: Integration code examples
- Plan: Which components to update first

### 3. Integration Roadmap (20 min)
Read: `CAPITAL_GOVERNOR_IMPLEMENTATION.md`
- Learn: 5-phase integration plan
- Study: Complete code examples
- Plan: Timeline and priorities

### 4. Start Integration (30 min)
Code: `MetaController` Phase B
- Add CapitalGovernor initialization
- Add position limit check
- Test with your account
- Verify 1 position limit enforced

### 5. Complete Integration (2-3 hours)
Code: Phases C, D, E
- SymbolRotationManager: Disable rotation
- PositionManager: Use bracket sizing
- Testing: Verify all limits work

---

## Getting Help

### "How do I use this in my code?"
→ See `CAPITAL_GOVERNOR_GUIDE.md` → "Integration Points"

### "What are the exact limits for my bracket?"
→ See `CAPITAL_GOVERNOR_QUICK_REF.md` → "Bracket Comparison Table"

### "How do I integrate this step-by-step?"
→ See `CAPITAL_GOVERNOR_IMPLEMENTATION.md` → "Integration Roadmap"

### "What should I expect to see in logs?"
→ See `CAPITAL_GOVERNOR_GUIDE.md` → "Logging & Monitoring"

### "Can I see the code?"
→ See `core/capital_governor.py` (400+ lines with docstrings)

---

## Summary

**Capital Governor** automatically enforces best-practice position limits based on account size:

| Bracket | Your Account | Max Symbols | Core | Rotating | Positions | Size | Mode |
|---------|---|---|---|---|---|---|---|
| MICRO | **< $500** | 2 | 2 | 0 | 1 | $12 | Learning |
| SMALL | $500-2K | 5 | 2 | 1 | 2 | $15 | Growth |
| MEDIUM | $2K-10K | 10 | 3 | 5 | 3 | $25 | Scaling |
| LARGE | ≥$10K | 20 | 5 | 10 | 5 | $50 | Institutional |

**Your Account**: MICRO ($350) → 2 symbols, 1 position, $12 per trade, NO rotation.

**Next Steps**: 
1. Review `CAPITAL_GOVERNOR_QUICK_REF.md` (5 min)
2. Plan integration with MetaController, SymbolRotationManager, PositionManager
3. Implement Phase B (MetaController) - 30 min
4. Test & verify position limits enforced

Ready? Start with `CAPITAL_GOVERNOR_QUICK_REF.md`! 🚀

---

**Questions?** Check the relevant guide above or review the source code docstrings in `core/capital_governor.py`.
