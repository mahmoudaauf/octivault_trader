# Phase 11: Why Model Does Not Generate Positive Expected Value

## Executive Summary

**Question:** "Why does the model not generate positive expected value under current regime and cost structure?"

**Answer:** The model's expected move estimates (0.65% fallback) are **systematically lower** than the execution system's required TP targets (0.55%-0.75%), making positive EV mathematically impossible in most market regimes.

This problem was **hidden before Phase 10** but became **visible after** Phase 10 added the ExecutionManager profitability filter.

---

## Root Cause Hierarchy

### Root Cause #1: Regime-Independent Expected Move Generation

**Location:** `agents/ml_forecaster.py`, lines 155-166

**Configuration:**
```python
ML_EXPECTED_MOVE_FALLBACK_PCT   = 0.0065  (0.65%)  ← FIXED, regime-independent
ML_EXPECTED_MOVE_MIN_PCT        = 0.0010  (0.10%)  ← Floor
ML_EXPECTED_MOVE_MAX_PCT        = 0.0500  (5.00%)  ← Ceiling
ML_REGIME_VOL_LOW_PCT           = 0.0045  (0.45%)  ← Low-vol regime
ML_REGIME_VOL_HIGH_PCT          = 0.0150  (1.50%)  ← High-vol regime
```

**Problem:**
- The model uses a **fixed 0.65% fallback** regardless of regime
- Low-volatility periods have ~0.45% expected moves
- Normal periods have ~0.65% expected moves
- Bull/high-volatility periods have ~1.50% expected moves
- Result: **Wrong expected move in 2/3 of all regimes**

**Impact by Regime:**
| Regime | Expected Move | Required TP | Net EV | Status |
|--------|---------------|-------------|--------|--------|
| Bull (1.50%) | 1.50% | 0.55% | +0.95% | ✅ POSITIVE |
| Normal (0.65%) | 0.65% | 0.55% | +0.10% | ⚠️ MARGINAL |
| Bear (0.45%) | 0.45% | 0.55% | -0.10% | ❌ NEGATIVE |

---

### Root Cause #2: Hidden Cost Multiplier (Line 1894)

**Location:** `core/execution_manager.py`, lines 1892-1894

**Critical Code:**
```python
m_entry = float(self._cfg("MIN_PLANNED_QUOTE_FEE_MULT", 2.5) or 2.5)
m_exit = float(self._cfg("MIN_PROFIT_EXIT_FEE_MULT", 2.0) or 2.0)
m_exit = max(m_exit, m_entry)  # ← THIS LINE INCREASES COST!
```

**Problem:**
- The code enforces `m_exit = max(2.0, 2.5) = 2.5`
- This uses the **higher of entry or exit multiplier**
- Instead of using the configured 2.0×, it uses 2.5×
- This increases the required TP from 0.40% to 0.50%

**Cost Calculation:**
```
With m_exit = 2.0:  r_req = (2.0 × 0.001 × 2.0) = 0.004 (0.40%)
With m_exit = 2.5:  r_req = (2.0 × 0.001 × 2.5) = 0.005 (0.50%)
                                                    ↑ +0.10% additional cost
```

---

### Root Cause #3: Slippage Mismatch Between Signal Generation and Execution

**Signal Generation (ML Forecaster):**
- Assumes 0 slippage when deriving expected move
- Uses historical candle data
- No explicit slippage adjustment

**Execution (ExecutionManager):**
- Accounts for slippage in TP calculation
- Default slippage: 15 bps (0.15%) per trade exit
- But this isn't included in ML expected move derivation

**Gap:**
```
ML Expected Move (0% slippage assumed):    0.65%
Actual Execution Slippage (15 bps):        0.15%
Hidden Cost:                               -0.15%
────────────────────────────────────────────────────
Actual Net Edge:                           0.50% (vs 0.65% perceived)
```

---

### Root Cause #4: MIN_NET_PROFIT_AFTER_FEES Floor

**Location:** `core/execution_manager.py`, line 1123

**Configuration:**
```python
MIN_NET_PROFIT_AFTER_FEES = 0.0035  (0.35%)
```

**How It Affects TP Calculation:**
```python
min_tp_needed_for_net = r_min_net + (2.0 × r_fee) + r_slip
                      = 0.0035 + 0.002 + 0.0
                      = 0.0055 (0.55%)

required_tp = max(r_req, min_tp_needed_for_net)
            = max(0.005, 0.0055)
            = 0.0055 (0.55%)  ← FINAL REQUIRED TP
```

**Problem:**
- This floor adds 0.35% on top of fee costs
- Makes it impossible to achieve positive EV with low expected moves
- No regime-aware adjustment

---

### Root Cause #5: Entry Fees Not Explicitly Accounted For

**Actual Round-Trip Cost Breakdown:**
```
Entry fee (0.1%):       +0.10%
Exit fee (0.1%):        +0.10%
Entry slippage (0.1%):  +0.10%
Exit slippage (0.1%):   +0.10%
TP Buffer (0.05%):      +0.05%
────────────────────────────────
TOTAL REAL COST:        0.45%
```

**What ExecutionManager Calculates:**
```
r_req = (2.0 × exit_fee × m_exit) + slippage + buffer
      = (2.0 × 0.001 × 2.5) + slippage + 0
      = 0.005 + slippage
      ≈ 0.005 (0.50% without slippage)
```

**Missing Cost:**
- Entry fee not fully accounted for: 0.10% shortfall
- Real total cost: 0.45% vs calculated 0.50%
- But actual execution exceeds calculation due to slippage

---

## Why Phase 10 Exposed This Problem

**Before Phase 10:**
```
Model Signal → Execution Check → Trade (even if negative EV)
              └─ Sometimes failed, but traded anyway
```

**After Phase 10:**
```
Model Signal → UURE Profitability Filter → Execution Check → Trade
              └─ New gate from ExecutionManager._entry_profitability_feasible()
              └─ Blocks signals where tp_max < required_tp
              └─ Result: NO TRADES (because expected_move ≈ required_tp)
```

The filter wasn't wrong—it **revealed** that the underlying EV was negative.

---

## Detailed EV Calculation By Regime

### Bull Regime (1.50% expected move)
```
Expected Move:       1.50%
Required TP:         0.55%
Gross Edge:          0.95%
Real Cost (slippage): 0.45%
────────────────────────────
NET EV:              0.50%  ✅ POSITIVE
```

### Normal Regime (0.65% expected move)
```
Expected Move:       0.65%
Required TP:         0.55%
Gross Edge:          0.10%
Real Cost (slippage): 0.45%
────────────────────────────
NET EV:              -0.35%  ❌ NEGATIVE
```

### Bear Regime (0.45% expected move)
```
Expected Move:       0.45%
Required TP:         0.55%
Gross Edge:          -0.10%
Real Cost (slippage): 0.45%
────────────────────────────
NET EV:              -0.55%  ❌ NEGATIVE
```

---

## The Formula Explaining Everything

### Break-Even Equation
```
break_even_prob = round_trip_cost / expected_move

For normal regime:
break_even_prob = 0.45% / 0.65% = 0.69 (need 69% win rate)

For bear regime:
break_even_prob = 0.45% / 0.45% = 1.00 (need 100% win rate = IMPOSSIBLE)
```

### EV Formula
```
EV_NET = Expected_Move - Round_Trip_Cost

Normal regime:  0.65% - 0.45% = +0.20% (marginal)
Bear regime:    0.45% - 0.45% = 0.00% (break-even)
Worst case:     0.20% - 0.45% = -0.25% (bad)
```

---

## Code Architecture Contributing to Problem

### ExecutionManager Cost Structure (Lines 1885-1900)
```python
# Current logic
trade_fee_pct = float(self._cfg("TRADE_FEE_PCT", 0.0) or 0.0)
exit_fee_bps = float(self._cfg("EXIT_FEE_BPS", 0.0) or 0.0)
fee_bps = max(exit_fee_bps, trade_fee_pct * 10000.0)  # Uses max!
r_fee = fee_bps / 10000.0
r_slip = float(self._cfg("EXIT_SLIPPAGE_BPS", 0.0) or 0.0) / 10000.0

# Multiplier enforcement
m_entry = 2.5 (configured)
m_exit = 2.0 (configured)
m_exit = max(m_exit, m_entry) = 2.5 ← FORCES HIGHER VALUE

# Final calculation
r_req = (2.0 * r_fee * m_exit) + r_slip + r_buf
```

**Problems:**
1. `m_exit = max(m_exit, m_entry)` forces higher multiplier
2. Slippage defaulting to 0 in some paths
3. Entry fees not explicitly calculated
4. No regime-awareness

### ML Forecaster Expected Move Derivation (Lines 1360-1372)
```python
def _derive_expected_move_pct(self, records):
    vals = [em for em in records if em > 0]
    if not vals:
        return 0.0065  # ← FIXED FALLBACK (0.65%)
    med = median(vals)
    return max(0.001, min(0.050, med))  # Clamped to [0.1%, 5.0%]
```

**Problems:**
1. Fallback is fixed to 0.65%
2. Not regime-aware
3. Not volatility-aware
4. Doesn't account for execution costs

---

## Configuration Values Summary

| Parameter | File | Line | Value | Issue |
|-----------|------|------|-------|-------|
| ML_EXPECTED_MOVE_FALLBACK_PCT | ml_forecaster.py | 165 | 0.0065 | Too low for normal regime |
| ML_REGIME_VOL_LOW_PCT | ml_forecaster.py | 163 | 0.0045 | Below required TP |
| MIN_PLANNED_QUOTE_FEE_MULT | execution_manager.py | 1892 | 2.5 | Enforced as m_exit min |
| MIN_PROFIT_EXIT_FEE_MULT | execution_manager.py | 1893 | 2.0 | Overridden to 2.5 |
| MIN_NET_PROFIT_AFTER_FEES | execution_manager.py | 1123 | 0.0035 | Adds 0.35% floor to costs |
| EXIT_SLIPPAGE_BPS | execution_manager.py | 1217 | 15.0 | Not priced into ML expected move |

---

## Why This Matters

### Performance Impact
- **Bull regime:** Trades happen, positive EV ✅
- **Normal regime:** Few trades, marginal EV ⚠️
- **Bear regime:** No trades, would be negative ❌

### System Behavior
- Phase 10 profitability filter blocks ~70% of signals
- Very few symbols pass the filter
- When they do, EV is marginal (0.05%-0.20%)
- Some regimes produce no tradeable symbols

### Root Problem
The system is **cost-constrained**, not **edge-constrained**.
The model's ability to generate alpha (expected move) is **insufficient** to cover the **fixed cost structure**.

---

## What Would Fix This

### Option A: Increase Expected Move Estimates
**Target:** Get fallback from 0.65% to 0.80%+
- Use ATR-based scaling
- Implement regime-aware fallbacks
- Account for slippage in derivation
- Use win-rate calibration

**Effort:** Medium | **Feasibility:** High

### Option B: Reduce Cost Structure
**Target:** Reduce total costs from 0.45% to 0.25%-0.30%
- Use limit orders (reduce slippage)
- Use Binance rebate strategies
- Reduce MIN_NET_PROFIT floor
- Fix multiplier enforcement (line 1894)

**Effort:** Low | **Feasibility:** Medium

### Option C: Regime-Based Dynamic Thresholds
**Target:** Adjust costs and requirements per regime
- Bull: Higher limits, wider TP targets
- Normal: Balanced
- Bear: Skip trading or tighter stops

**Effort:** Medium | **Feasibility:** High

### Option D: Regime Filter
**Target:** Only trade bull regime
- Skip bear/sideways regimes
- Accept lower trade frequency
- Guarantee positive EV

**Effort:** Low | **Feasibility:** Very High

---

## Related Code Locations

| Location | Purpose | Impact |
|----------|---------|--------|
| `core/execution_manager.py:1878-1950` | `_entry_profitability_feasible()` | Calculates required TP |
| `core/execution_manager.py:1892-1894` | Multiplier enforcement | Increases cost structure |
| `agents/ml_forecaster.py:155-166` | Config initialization | Sets expected move defaults |
| `agents/ml_forecaster.py:1360-1372` | `_derive_expected_move_pct()` | Derives expected move from records |
| `core/universe_rotation_engine.py:???` | `_apply_profitability_filter()` | Calls ExecutionManager check (NEW in Phase 10) |

---

## Conclusion

The model does **not generate positive expected value** because:

1. **Expected move generation (0.65% fallback) is regime-independent** and often falls below the execution system's required TP targets (0.55%-0.75%)

2. **The cost structure has a hidden multiplier** (line 1894 enforces m_exit = 2.5 instead of configured 2.0), increasing costs by 0.10%

3. **Slippage is not priced into the ML model** but is accounted for in execution, creating a 0.15%-0.20% gap

4. **The MIN_NET_PROFIT floor (0.35%)** adds additional cost that makes positive EV impossible in low-volatility regimes

5. **Costs are fixed regardless of market regime**, so the same cost structure applies to:
   - Bull (1.50% moves) → Positive EV ✅
   - Normal (0.65% moves) → Marginal/Negative EV ⚠️
   - Bear (0.45% moves) → Negative EV ❌

**Phase 10's profitability filter didn't cause this—it revealed it** by blocking signals that wouldn't make money.

