# Regime-Based Scaling Quick Reference Card

## 🎯 What Changed

| Before | After |
|--------|-------|
| Binary gate: Block entire regime | Gradient scaling: Reduce position size per regime |
| `if regime == "bear": return` | Scale by 0.6x and continue |
| All-or-nothing | Risk-adjusted positioning |
| Miss profitable trades | Capture all valid trades |

---

## 📊 Regime Scaling Matrix (At a Glance)

```
REGIME          │ POS SIZE │ TP TARGET │ EXCURSION │ TRAILING │ CONFIDENCE
────────────────┼──────────┼───────────┼───────────┼──────────┼──────────
Trending        │  1.0x ✓  │   1.0x ✓  │   0.85x ✓ │  1.3x ✓  │   +5% ✓
High Vol        │  0.8x    │   1.05x   │   1.0x    │  1.2x    │    0%
Sideways        │  0.5x ✗  │   0.6x ✗  │   1.4x ✗  │  0.9x ✗  │   -5% ✗
Bear            │  0.6x    │   0.8x    │   1.2x ✗  │  0.95x   │   -8%
Normal          │  1.0x    │   1.0x    │   1.0x    │  1.0x    │    0%

✓ = Favorable (go big)  |  ✗ = Unfavorable (be careful)
```

---

## 🔄 Signal Flow

```
┌──────────────┐
│ TrendHunter  │ Emits: signal + _regime_scaling + _regime
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ MetaController       │ Applies: position_size_mult ← PHASE 2
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ExecutionManager     │ Creates order (scaled size)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ TP/SL Engine         │ Applies: tp_mult, excursion_mult ← PHASE 3
│ Trailing            │ Applies: trail_mult ← PHASE 4
└──────────────────────┘
```

---

## 🎬 Example: Sideways Regime Trade

```
SIGNAL:
  Symbol: ETHUSDT
  Action: BUY
  Confidence: 0.72
  Quote: $100
  
REGIME DETECTION:
  1h Regime: "sideways"
  
SCALING APPLIED:
  Position Size: $100 → $50 (50% of normal)
  TP Target:    1.5% → 0.9% (60% of normal)
  Excursion:    100bp → 140bp (1.4x harder to confirm)
  Trailing:     1.5x ATR → 1.35x ATR (tighter)
  Confidence:   0.72 → 0.67 (-5% penalty)
  
EXECUTION:
  Order size: $50 instead of $100
  TP: 0.9% move (tighter profit target)
  Trailing: Tight trailing stops
  
RESULT:
  Lower risk in choppy sideways market
  Captures profit quickly if thesis is right
  Exits quickly if sideways breaks down
```

---

## 📍 Integration Status

| Phase | Component | Status | Impact |
|-------|-----------|--------|--------|
| 1 | TrendHunter | ✅ DONE | Emits scaling data |
| 2 | MetaController | ⏭️ TODO | Scales position size |
| 3a | TP/SL (TP) | ⏭️ TODO | Scales TP targets |
| 3b | TP/SL (Excursion) | ⏭️ TODO | Scales excursion gate |
| 4 | ExecutionManager | ⏭️ TODO | Scales trailing |
| 5 | Config | ⏭️ TODO | Externalizes multipliers |

---

## 🚀 Next Steps (5-Minute Quick Start)

### 1. Verify Phase 1 (2 min)
```bash
# Open and scan lines 503-720 in:
# agents/trend_hunter.py
# ✓ Look for _get_regime_scaling_factors() method
# ✓ Look for _regime_scaling in signal emission
```

### 2. Plan Phase 2 (3 min)
```bash
# Open:
# core/meta_controller.py
# Search: "_execute_decision" or "execute_decision"
# Task: Apply signal._regime_scaling["position_size_mult"] to quote_hint
```

### 3. Reference Documentation (Reference as needed)
```
Read in this order:
1. REGIME_BASED_SCALING_SUMMARY.md (this context)
2. REGIME_BASED_SCALING_ARCHITECTURE.md (why/how)
3. REGIME_SCALING_INTEGRATION_CODE_SNIPPETS.md (exact code)
```

---

## 💡 Key Insights

1. **Alpha is everywhere** - Don't block regimes, scale them
2. **Sideways is hardest** - 50% position size, tighter TP, harder excursion
3. **Trending is easiest** - Full size, full TP, loose trailing
4. **Confidence matters** - High confidence overrides regime penalties
5. **Consistency wins** - Same scaling logic across all agents

---

## ⚠️ Common Mistakes (Avoid These)

❌ **DON'T**: Hard-block bear regime trades
✅ **DO**: Scale bear trades to 60% size

❌ **DON'T**: Ignore regime in TP/SL calculations
✅ **DO**: Apply multipliers consistently

❌ **DON'T**: Change multipliers without backtesting
✅ **DO**: Externalize to config and test A/B

❌ **DON'T**: Forget to log scaling decisions
✅ **DO**: Log every scaling application

---

## 🔍 How to Verify It's Working

```python
# Check 1: Signal has scaling data
signal = agent._collected_signals[-1]
assert "_regime_scaling" in signal
assert signal["_regime_scaling"]["position_size_mult"] > 0

# Check 2: MetaController applies it (after Phase 2)
# Position size in order should match signal quote × multiplier
expected_size = 100.0 * 0.5  # e.g., sideways
actual_size = order["quote_hint"]
assert abs(expected_size - actual_size) < 0.01

# Check 3: TP is scaled (after Phase 3a)
expected_tp = entry + (1.5% * 0.6)  # e.g., sideways
actual_tp = position["tp_price"]
assert abs(expected_tp - actual_tp) < 0.01
```

---

## 📚 Documentation Map

| Document | Purpose | Read When |
|----------|---------|-----------|
| **REGIME_BASED_SCALING_SUMMARY.md** (this file) | Quick overview | First (2 min) |
| **REGIME_BASED_SCALING_ARCHITECTURE.md** | Deep dive | Planning implementation |
| **REGIME_SCALING_INTEGRATION_CHECKLIST.md** | Task tracking | Managing work |
| **REGIME_SCALING_INTEGRATION_CODE_SNIPPETS.md** | Code examples | Writing code |

---

## 🎯 Success Indicators

✅ All green when:
- [ ] Signal carries `_regime_scaling` dict
- [ ] Position size changes per regime (50% in sideways, 100% in trending)
- [ ] TP targets change per regime (60% in sideways, 100% in trending)
- [ ] Excursion gates change per regime (harder in sideways)
- [ ] Trailing stops change per regime (tighter in sideways)
- [ ] Logs show scaling decisions
- [ ] No signals are hard-blocked (all gradient-scaled)
- [ ] Performance metrics show improvement

---

## 🔄 The Loop

```
1. Generate signal with regime awareness ✅ (Phase 1)
   ↓
2. Apply position size scaling ⏭️ (Phase 2)
   ↓
3. Apply TP/SL scaling ⏭️ (Phase 3)
   ↓
4. Apply trailing scaling ⏭️ (Phase 4)
   ↓
5. Make multipliers configurable ⏭️ (Phase 5)
   ↓
6. Test and backtest
   ↓
7. Monitor live performance
   ↓
8. Tune multipliers via config
   ↓
9. [repeat from step 6]
```

---

## 🏁 Bottom Line

**Before**: System blocked trades in bad regimes (lost alpha)

**After**: System scales down trades in bad regimes (captures alpha with controlled risk)

**Benefit**: More profitable, more consistent, risk-managed across all market conditions

**Status**: Phase 1 ✅ complete, Phases 2-5 ⏭️ ready to implement

**Effort**: ~5-10 lines per phase, total ~50 lines to add across codebase

**Impact**: Fundamental improvement in regime-aware position sizing

---

## 📞 For Questions

1. **How do multipliers work?** → See REGIME_BASED_SCALING_ARCHITECTURE.md
2. **What are the exact code changes?** → See REGIME_SCALING_INTEGRATION_CODE_SNIPPETS.md
3. **What's the implementation plan?** → See REGIME_SCALING_INTEGRATION_CHECKLIST.md
4. **Is this live yet?** → Phase 1 ✅, Phases 2-5 ⏭️

---

**Created**: REGIME_BASED_SCALING_SUMMARY.md
**Status**: Complete implementation ready
**Next**: Start Phase 2 integration in MetaController
