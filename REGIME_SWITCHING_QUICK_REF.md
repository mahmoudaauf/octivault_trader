# Regime Switching - Quick Reference Card

## ✅ YES - The System Can Switch Models

The system uses **automatic NAV-based regime switching** between three models.

---

## The Three Models

### Model 1: MICRO_SNIPER (NAV < $1,000)
**Retail Micro / Beginner Model**

```
Max Positions:        1 (single trade only)
Max Assets:          1 (one symbol at a time)
Min Expected Move:   1.0% (hardened requirement)
Min Confidence:      70%
Max Trades/Day:      3 (hourly discipline)
Position Size:       30% of NAV
```

✓ Ultra-conservative
✓ Learning phase
✓ Capital preservation

---

### Model 2: STANDARD ($1,000 - $5,000)
**Growth / Transition Model**

```
Max Positions:        2 (limited pairs)
Max Assets:          3 (conservative universe)
Min Expected Move:   0.50% (balanced)
Min Confidence:      65%
Max Trades/Day:      6 (moderate pace)
Position Size:       25% of NAV
```

✓ Controlled expansion
✓ Feature unlock begins
✓ Skill development

---

### Model 3: MULTI_AGENT (NAV ≥ $5,000)
**Professional / Hedge Fund Model**

```
Max Positions:        3+ (portfolio diversification)
Max Assets:          5+ (aggressive coverage)
Min Expected Move:   0.30% (full sensitivity)
Min Confidence:      60% (standard threshold)
Max Trades/Day:      20+ (unrestricted)
Position Size:       Standard scaling
```

✓ Full professional system
✓ All features enabled
✓ Maximum optimization

---

## How It Works

### Automatic Switching
```
Every cycle:
1. Check live NAV from SharedState
2. Determine regime: MICRO_SNIPER | STANDARD | MULTI_AGENT
3. Apply regime rules to all decisions
4. Switch regimes seamlessly if NAV crosses threshold
```

### Example Progression
```
Start       → Grow        → Scale      → Drawdown
$500 MICRO   $1,200 STD    $5,500 MULTI  $800 MICRO
```

---

## Regime Enforcement

### ❌ Signals Rejected if:
- Confidence below regime floor
- Expected move below regime minimum
- Position count at regime limit
- Feature disabled for regime

### ✅ Enforcement Points:
- Signal validation (MetaController)
- Position limiting (CapitalGovernor)
- Feature availability (Components)
- Trade frequency (ModeManager)

---

## Key Differences

| Feature | Micro | Standard | Hedge Fund |
|---------|-------|----------|------------|
| Positions | 1 | 2 | 3+ |
| Assets | 1 | 3 | 5+ |
| Move Req | 1.0% | 0.5% | 0.3% |
| Trades/Day | 3 | 6 | 20+ |
| Rotation | ❌ | ✅ | ✅ |
| Dust Heal | ❌ | ✅ | ✅ |

---

## No Manual Configuration Needed

✅ Automatic based on NAV
✅ No restart required
✅ Seamless transitions
✅ Logged for audit
✅ Reversible on drawdown

---

## Monitor Current Regime

```python
nav = signal_manager.get_current_nav()
regime = nav_regime.get_nav_regime(nav)

if regime == NAVRegime.MICRO_SNIPER:
    print("Retail Micro Model Active")
elif regime == NAVRegime.STANDARD:
    print("Growth Model Active")
elif regime == NAVRegime.MULTI_AGENT:
    print("Hedge Fund Model Active")
```

---

## Real Example: Growing Account

```
Week 1:  NAV = $300 → MICRO_SNIPER
         Max 1 position, need 1.0% move
         Result: Safe learning phase

Week 2:  NAV = $1,100 → Auto-switches to STANDARD
         Max 2 positions, need 0.5% move
         Result: Controlled expansion unlocked

Month 2: NAV = $5,200 → Auto-switches to MULTI_AGENT
         Max 3+ positions, need 0.3% move
         Result: Professional system activated

Month 3: Drawdown to NAV = $900 → Auto-switches back to MICRO_SNIPER
         Protection mode: Max 1 position
         Result: Capital preservation engaged
```

---

## Status: ✅ FULLY IMPLEMENTED

All three models:
- ✅ Implemented
- ✅ Integrated
- ✅ Tested
- ✅ Automatic
- ✅ Production ready

---

## Files to Reference

- `core/nav_regime.py` - Regime definitions
- `core/capital_governor.py` - Limit enforcement
- `core/meta_controller.py` - Decision enforcement
- `core/regime_manager.py` - State management

---

## Bottom Line

The system doesn't manually toggle between "Hedge Fund Mode" and "Retail Micro Mode" - instead, it **automatically adapts to whatever NAV level your account has**, providing appropriate constraints and unlocking features gradually as capital grows.

This is more sophisticated and safer than a binary toggle because it ensures you're never under-constrained (risking big) or over-constrained (missing opportunities) relative to your account size.
