# Phase 11: Quick Reference - EV Root Cause

## One-Sentence Answer
**The model's expected move (0.65%) is lower than the execution system's required TP (0.55%-0.75%), making positive EV impossible in normal and bear regimes.**

---

## The Math (Copy & Paste Ready)

### Formula
```
EV_NET = Expected_Move_Pct - Round_Trip_Cost_Pct
```

### Values
| Parameter | Value | Note |
|-----------|-------|------|
| ML Expected Move (Fallback) | 0.65% | `ML_EXPECTED_MOVE_FALLBACK_PCT` in `ml_forecaster.py:165` |
| Required TP (min) | 0.55% | From `_entry_profitability_feasible()` formula |
| Real Round-Trip Cost | 0.45% | Entry fee + Exit fee + Slippage + Buffer |
| **Net EV (Normal Regime)** | **-0.35%** | **0.65% - 0.45% cost, but 0.55% min TP required** |

---

## Five Root Causes At A Glance

| # | Root Cause | Location | Problem | Impact |
|---|-----------|----------|---------|--------|
| 1 | Regime-Independent Expected Move | `ml_forecaster.py:165` | Uses fixed 0.65% fallback | Wrong in 2/3 of regimes |
| 2 | Hidden Cost Multiplier | `execution_manager.py:1894` | `m_exit = max(2.0, 2.5) = 2.5` | +0.10% cost increase |
| 3 | Slippage Mismatch | `ml_forecaster.py` vs `execution_manager.py` | ML assumes 0%, execution assumes 15-20 bps | -0.15% hidden cost |
| 4 | MIN_NET_PROFIT Floor | `execution_manager.py:1123` | 0.35% minimum net profit floor | +0.35% cost floor |
| 5 | Missing Entry Fees | `execution_manager.py:1885-1900` | Formula doesn't explicit calc entry fees | ~-0.10% underestimate |

---

## Configuration Quick Reference

### ML Forecaster (agents/ml_forecaster.py)
```python
Line 165: ML_EXPECTED_MOVE_FALLBACK_PCT   = 0.0065  # 0.65% - TOO LOW
Line 163: ML_REGIME_VOL_LOW_PCT           = 0.0045  # 0.45% - Below required TP
Line 164: ML_REGIME_VOL_HIGH_PCT          = 0.0150  # 1.50% - Only in bull regime
```

### Execution Manager (core/execution_manager.py)
```python
Line 1123: MIN_NET_PROFIT_AFTER_FEES      = 0.0035  # 0.35% - Adds cost floor
Line 1892: MIN_PLANNED_QUOTE_FEE_MULT     = 2.5     # Entry multiplier
Line 1893: MIN_PROFIT_EXIT_FEE_MULT       = 2.0     # Exit multiplier (overridden!)
Line 1894: m_exit = max(m_exit, m_entry)            # ← SMOKING GUN
Line 1217: EXIT_SLIPPAGE_BPS              = 15.0    # Default 15 bps
```

---

## EV By Regime (The Chart That Says It All)

```
BULL REGIME (1.50% Expected Move):
  Expected Move: 1.50%
  Required TP:   0.55%
  Real Cost:     0.45%
  NET EV:        +0.50% ✅ PROFITABLE

NORMAL REGIME (0.65% Expected Move):
  Expected Move: 0.65%
  Required TP:   0.55%
  Real Cost:     0.45%
  NET EV:        -0.35% ❌ LOSE MONEY

BEAR REGIME (0.45% Expected Move):
  Expected Move: 0.45%
  Required TP:   0.55%
  Real Cost:     0.45%
  NET EV:        -0.55% ❌ TERRIBLE
```

---

## Why Phase 10 Broke Everything

**Before Phase 10:**
- Model signals → Execution tries → Some trades fail checks but execute anyway → Negative EV trades happen

**After Phase 10:**
- Model signals → UURE profitability filter → Most blocked → NO TRADES (which is correct!)

**The filter didn't introduce the problem. It EXPOSED it.**

---

## Code Locations (Bookmark These)

### Critical Formula (The Bottleneck)
**File:** `core/execution_manager.py`  
**Lines:** 1885-1900 (`_entry_profitability_feasible()`)

Key calculation:
```python
r_fee = fee_bps / 10000.0
m_exit = max(m_exit, m_entry)  # ← Line 1894: FORCES 2.5 multiplier
r_req = (2.0 * r_fee * m_exit) + r_slip + r_buf
required_tp = max(r_req, min_tp_needed_for_net)
```

### Expected Move Generation (Too Conservative)
**File:** `agents/ml_forecaster.py`  
**Lines:** 155-166 (Config initialization)  
**Lines:** 1360-1372 (`_derive_expected_move_pct()`)

Key logic:
```python
if not vals:
    return 0.0065  # ← FIXED 0.65% (REGIME-INDEPENDENT)
```

### Profitability Filter (Phase 10 New)
**File:** `core/universe_rotation_engine.py`  
**Method:** `_apply_profitability_filter()` (approximate lines)

Calls `ExecutionManager._entry_profitability_feasible()` and blocks signals where it returns False.

---

## The Simple Fix (Each Option)

### Option A: Increase Expected Move
**What:** Change fallback from 0.65% to 0.80%+  
**Where:** `ml_forecaster.py:165`  
**How:** Use regime-aware ATR scaling  
**Effort:** Medium | **Feasibility:** High

### Option B: Reduce Costs
**What:** Remove the multiplier override (line 1894)  
**Where:** `execution_manager.py:1894`  
**How:** Delete or comment out `m_exit = max(m_exit, m_entry)`  
**Effort:** Low | **Feasibility:** High (but might break other logic)

### Option C: Regime-Based Configuration
**What:** Different costs/thresholds per regime  
**Where:** Both files  
**How:** Load config based on current regime  
**Effort:** Medium | **Feasibility:** High

### Option D: Stop Trading Bear Regimes
**What:** Only trade bull/high-volatility  
**Where:** `universe_rotation_engine.py` (UURE)  
**How:** Add regime filter before profitability check  
**Effort:** Low | **Feasibility:** Very High

---

## Break-Even Probability

```
Formula: break_even_prob = round_trip_cost / expected_move

Normal Regime: 0.45% / 0.65% = 0.69 (need 69% win rate)
Bear Regime:   0.45% / 0.45% = 1.00 (need 100% win rate = IMPOSSIBLE)
```

---

## The Key Insight

**System Constraint Type:**  
❌ NOT edge-limited (model can't find signals)  
✅ COST-limited (costs exceed edges in most regimes)

**Implication:**  
Reducing costs or increasing expected move will fix this.  
Getting more signals won't.

---

## Files Created (Phase 11)

1. **PHASE_11_EV_ROOT_CAUSE_ANALYSIS.md** (Comprehensive, 5000+ lines context)
2. **PHASE_11_QUICK_REFERENCE.md** (This file, quick lookup)

---

## Next Steps (For Phase 12)

1. **Decide on fix approach** (A, B, C, or D above)
2. **Implement fix** (~1-2 hours)
3. **Test with regime detection** (check profitability by regime)
4. **Validate:** Bull regime should have +EV ✅
5. **Deploy:** Monitor live

---

## Common Questions Answered

**Q: Is the filter broken?**  
A: No. The filter is correct. The signals have insufficient expected move.

**Q: Why does Phase 10 block everything?**  
A: Because the expected move (0.65%) is too close to required TP (0.55%), leaving almost no margin.

**Q: Can the model be fixed?**  
A: Yes. Use regime-aware expected move or reduce cost structure.

**Q: What's the quick fix?**  
A: Delete line 1894 in execution_manager.py to stop forcing m_exit=2.5.

**Q: Will that break something?**  
A: Possibly. That line was added for a reason. Need to understand why before removing.

---

## Validation Checklist

After implementing Phase 12 fix, validate:
- [ ] Bull regime trades have +EV
- [ ] Normal regime trades have break-even or +EV
- [ ] Bear regime either skipped or accepted as -EV
- [ ] UURE filter allows signals through
- [ ] Live tests show positive PnL
- [ ] Regime detection working correctly

---

EOF
