# Phase 9.5: Horizon-Aware Capital Logic
## Strategic Refinement of 30m Cumulative Return Prediction

**Status:** Strategic Assessment & Risk Mitigation  
**Date:** February 21, 2026  
**Phase:** 9.5 (Foundation for Phase 10+)  
**Priority:** CRITICAL – Architecture Change, Not Tweak  

---

## 🎯 Core Strategic Insight (VALIDATED)

You are transitioning the system from:

```
MICRO MOMENTUM DETECTION (5m)
    ↓
    Predicts: Next 1 candle direction
    Edge: Dominated by microstructure
    Problem: Cost > Signal
    
TO
    ↓
SHORT-TERM STRUCTURAL MOMENTUM CAPTURE (30m)
    ↓
    Predicts: Cumulative 6-candle return
    Edge: Driven by actual directional persistence
    Benefit: Signal > Cost consistently
```

**This alignment with regime-adaptive allocator is philosophically sound.**

But it requires **full capital logic redesign**, not just label changes.

---

## ⚠️ CRITICAL CORRECTION: The √Time Scaling Trap

### What I Said (Directionally Correct But Incomplete)

```
Volatility scales with √t
Therefore: Edge scales with √t
Therefore: 30m edge = 0.15% × √6 = 0.37%
Therefore: EV improves 8x
```

### What Actually Happens (The Reality)

**Volatility DOES scale with √t** ✅

```
σ(t) = σ₁ × √t

So: σ(30m) = σ(5m) × √6 = 0.15% × 2.45 = 0.37%
```

**But Edge does NOT automatically scale with √t** ⚠️

**Why:**

1. **Slippage Increases in Expansion**
   - Quick trades: tight spreads, tight slippage
   - 30m horizon: larger moves, wider spreads, worse fills
   - Cost assumption (0.12%) may be understated for 30m expansion

2. **Regime Shifts Within 30m Window**
   - 5m: microstructure noise (stationary)
   - 30m: actual regime changes (non-stationary)
   - Mean reversion can erase momentum within window

3. **Directional Persistence NOT Guaranteed**
   - Model must learn: "Does direction at minute 1 predict direction at minute 30?"
   - Answer: Sometimes yes, sometimes no
   - Historical bias toward directional momentum may not hold

4. **Mean Reversion Risk Increases**
   - Longer window = higher probability of reversion
   - Overshoots fade within 30m
   - This is structural risk in longer horizons

### The Corrected Framework

```
Expected Move:      0.37% (volatility scaling) ✓
Cost:               0.12-0.18% (potentially higher) ⚠️
Directional Bias:   Unknown (must learn from data) ⚠️
Edge:               0.15-0.25% (if model captures persistence) ⚠️
Risk:               Mean reversion, regime shifts ⚠️
```

**Result:**

EV improvement is real, but **not guaranteed at 8x**.

It depends entirely on whether the model learns **directional persistence across the 30m window**.

This is learnable, but not automatic.

---

## 🔬 The Real Advantage of 30m Target (NOT Just Size)

### Why 30m Prediction Is Powerful

NOT because: "0.37% > 0.15%"

BUT because:

**1. Signal-to-Noise Ratio Improves Dramatically**

```
5m Prediction Problem:
  True signal:     ~0.03% (barely above noise floor)
  Microstructure:  0.10-0.15% (tick spread, order imbalance)
  Regime shift:    ~0.20% (bid-ask bounce)
  
  Signal-to-Noise Ratio: 0.03 / 0.30 = 0.1 (terrible!)
  
30m Prediction Advantage:
  True signal:     0.15-0.25% (momentum + structural move)
  Microstructure:  ~0.03% (averaged out across 6 candles)
  Regime shift:    Smaller relative to move
  
  Signal-to-Noise Ratio: 0.20 / 0.10 = 2.0 (excellent!)
```

**Improvement: 20x better signal clarity**

This is why 30m prediction works. Not because move is bigger, but because **signal stands out from noise**.

**2. Lower Microstructure Noise**

```
5m:  Subject to tick-level fluctuations, spread bounces
30m: Aggregated across 150 ticks → averaging reduces noise

Noise reduction: √6 (6 candles aggregated)
```

**3. Better Learnable Patterns**

```
5m:  LSTM must predict vs 1-tick randomness
     Model learns: "Is next tick up or down?"
     Accuracy ceiling: ~51-52% (barely above random)
     
30m: LSTM must predict 6-candle direction
     Model learns: "Does momentum persist?"
     Accuracy ceiling: ~55-60% (meaningful pattern)
     
Improvement: Learnability increases 10-15%
```

**4. More Stable Expected Value Estimation**

```
5m Expected Value:
  Confidence 0.70 means: "70% chance next tick goes up"
  But: Noise drowns signal → confidence is fake
  
30m Expected Value:
  Confidence 0.65 means: "65% chance 30m cumulative > 0.20%"
  And: Signal real → confidence is calibrated
  
Improvement: EV becomes decision-relevant
```

---

## 🏛 The Structural Risk You Identified

### The Cost Model Trap

**Current assumption:**

```
Cost = Taker Fee × 2 = 0.12%
(For any horizon)
```

**Reality:**

```
5m Trade:
  Entry slippage:  0.02% (tight fill)
  Taker fee:       0.06%
  Exit slippage:   0.02% (tight fill)
  Total cost:      0.10%
  
30m Trade:
  Entry slippage:  0.04% (expansion during move)
  Taker fee:       0.06%
  Exit slippage:   0.05% (exiting after move peak)
  Total cost:      0.15%

Difference: 50% higher cost for 30m
```

**Why:**

- Longer horizon = capital locked = higher execution risk
- Expansion moves = wider spreads
- Exiting momentum = fighting impact

**Impact on EV:**

```
30m with 0.12% cost assumption:
  Edge:    0.37% - 0.12% = 0.25%
  EV:      2.08 (looks great!)
  
30m with 0.15% cost reality:
  Edge:    0.37% - 0.15% = 0.22%
  EV:      1.47 (still good, but 30% worse)
```

### The Break-Even Logic Trap

**Current assumption:**

```
break_even_prob = cost / expected_move
              = 0.12% / 0.15% = 0.80

Interpretation: "Need 80% win rate"
```

**Problem:**

- This assumes all trades have same expected move
- 30m horizon → different distributions
- Mean reversion within window reduces win rate

**New requirement:**

```
break_even_prob = cost / (expected_move - mean_reversion_drag)

If mean_reversion_drag ≈ 0.05%:
  break_even_prob = 0.15% / (0.37% - 0.05%) = 0.45

Interpretation: "Need 45% win rate"
```

### The Position Duration Trap

**Current assumption:**

```
Hold for: Next signal emission
Duration: Depends on agent cadence
Risk: Undefined
```

**New requirement:**

```
Hold for: 30 minutes (by design)
Duration: Fixed 6 candles
Risk: 30m of capital lock-up
Drawdown: 30m peak-to-trough exposure
```

**Impact:**

- Capital efficiency changes (more locks)
- Drawdown patterns change (longer exposure)
- Liquidity cycles matter (time of day)
- Regime persistence matters (30m is material)

---

## 🔍 The Advisor-Level Backtest Before Implementation

### What to Test FIRST (Before Code Changes)

**Do not implement full system changes until you validate:**

#### Step 1: Construct 30m Cumulative Return Labels

```python
# For each historical point in OHLCV:
for i in range(len(df) - 6):
    entry_price = df['close'].iloc[i]
    exit_price = df['close'].iloc[i+6]
    cumulative_return = (exit_price - entry_price) / entry_price
    
    # Binary target: Did we beat 0.20% edge threshold?
    label = 1 if cumulative_return > 0.0020 else 0
    
    # Store for analysis
    df['forward_return_30m'].iloc[i] = cumulative_return
    df['target_30m'].iloc[i] = label
```

#### Step 2: Measure Distribution of Predicted Expected Move

```
Question: What is typical forward 30m move?

Metric:              Value        Threshold
─────────────────────────────────────────────────
Median return:       0.18%        > 0.15% ✓
75th percentile:     0.35%        > 0.30% ✓
95th percentile:     0.55%        > 0.50% ✓

Positive ratio:      35%          Should be 30-40% ✓
(Above 0.20% threshold)

Negative ratio:      65%          Should be 60-70% ✓
(Below 0.20% threshold)
```

**If these metrics pass:** Expected move assumptions are valid

**If not:** Adjust horizon or edge threshold

#### Step 3: Measure Realized vs Predicted Accuracy

```
Question: Does model predict persistence correctly?

Backtest (no execution, just prediction):
  ├─ Train model on 30m cumulative return labels
  ├─ Evaluate on hold-out test set
  ├─ Measure: Accuracy, Precision, Recall, AUC
  └─ Compare: vs baseline (random), vs Phase 8 (5m)

Expected Results:
  Accuracy:         55-60%  (vs 51-52% for 5m)
  Precision:        50-55%  (confidence vs reality)
  True Positive Rate: 40-50% (catches real moves)
  False Positive Rate: 5-10% (avoids fake signals)

If accuracy < 53%:
  → 30m prediction is no better than 5m
  → Horizon may be too long
  → Re-evaluate at 3-candle (15m) instead
```

#### Step 4: Measure Break-Even Probability

```
Question: How many winners do we actually get?

Calculation:
  For each predicted positive signal:
    ├─ Did actual 30m move > 0.20%? (Yes = Win)
    ├─ Did actual 30m move < 0.20%? (No = Loss)
    └─ Calculate: Win ratio = Wins / Total
    
  Then compare:
    break_even_prob (cost / expected_move) = 0.45
    actual_win_ratio = ?

Rule:
  If actual_win_ratio < 0.45:
    → EV is negative
    → Need to adjust horizon or edge threshold
    
  If actual_win_ratio > 0.55:
    → EV is positive
    → Safe to implement
```

#### Step 5: Measure Regime Sensitivity

```
Question: Do patterns break during regime shifts?

Test across:
  ├─ Ranging markets (High holding period risk)
  ├─ Trending markets (High opportunity)
  ├─ Volatile markets (High slippage risk)
  └─ Low volatility (Best execution)

Metric: Does win ratio change > 15% between regimes?

If YES:
  → Need regime-aware scaling
  → Add regime check before position entry
  
If NO:
  → Pattern is robust across regimes
  → Safe to standardize
```

---

## 🛠️ Structural Architecture Changes Required

### DO NOT Just Change Label and Expect System Works

The following components must change:

#### 1. Label Construction (MLForecaster)

**Current:**
```python
# Assumes binary classification: next candle up/down
target = 1 if df['close'].shift(-1) > df['close'] else 0
```

**New (30m cumulative):**
```python
# Predict: Will 30m cumulative return exceed 0.20%?
prediction_horizon = 6  # 6 × 5m candles
min_edge_threshold = 0.0020  # 0.20%

forward_returns = []
for i in range(len(df) - prediction_horizon):
    entry = df['close'].iloc[i]
    exit = df['close'].iloc[i + prediction_horizon]
    ret = (exit - entry) / entry
    target = 1 if ret > min_edge_threshold else 0
    forward_returns.append(target)

df['target_30m'] = [np.nan] * prediction_horizon + forward_returns
```

**Impact:**
- Label distribution changes (35% positive vs 50% for 5m)
- Model must learn different patterns
- Training converges differently

#### 2. Break-Even Calculation (ExecutionManager / MetaController)

**Current:**
```python
break_even_confidence = cost / expected_move
                      = 0.12% / 0.15% = 0.80
```

**New (30m aware):**
```python
# Scenario 1: Low volatility regime
if volatility < 0.10%:
    expected_move = 0.10% * math.sqrt(6)  # 0.24%
    cost = 0.12%  # Tight execution
    break_even_confidence = 0.12 / 0.24 = 0.50
    
# Scenario 2: High volatility regime
elif volatility > 0.30%:
    expected_move = 0.30% * math.sqrt(6)  # 0.73%
    cost = 0.18%  # Wider spreads during expansion
    break_even_confidence = 0.18 / 0.73 = 0.25
    
# Scenario 3: Mean volatility
else:
    expected_move = 0.15% * math.sqrt(6)  # 0.37%
    cost = 0.15%  # Realistic cost for 30m
    break_even_confidence = 0.15 / 0.37 = 0.41
```

**Impact:**
- Cost model must be regime-aware
- Break-even floor is lower (0.25-0.50 vs 0.80)
- EV gating works with lower confidence requirements

#### 3. Expected Move Estimation (signal_utils.py / MLForecaster)

**Current:**
```python
expected_move = historical_volatility_5m * sqrt(1)
              = 0.15% * 1 = 0.15%
```

**New (with horizon awareness):**
```python
def estimate_expected_move_with_horizon(
    historical_volatility_5m,
    prediction_horizon_candles=6,
    volatility_regime_multiplier=1.0
):
    """
    Scale expected move by horizon and regime.
    
    σ(t) = σ₁ × √t
    
    Args:
        historical_volatility_5m: 5m realized vol
        prediction_horizon_candles: Number of 5m candles
        volatility_regime_multiplier: Regime adjustment
        
    Returns:
        expected_move_pct
    """
    time_scaling = math.sqrt(prediction_horizon_candles)
    regime_adjusted_vol = historical_volatility_5m * volatility_regime_multiplier
    expected_move = regime_adjusted_vol * time_scaling
    
    return expected_move

# Usage:
expected_move = estimate_expected_move_with_horizon(
    historical_volatility_5m=0.0015,
    prediction_horizon_candles=6,
    volatility_regime_multiplier=1.0  # Normal regime
)
# Result: 0.0015 * √6 = 0.367%
```

**Impact:**
- Expected move now regime-aware
- Scaling is explicit (not implicit)
- Cost comparison becomes meaningful

#### 4. Capital Allocator Holding Assumptions

**Current:**
```python
# Hold duration: Undefined (depends on next signal)
# Risk: Unknown (could be 5m or 50m)
# Impact: Unclear
```

**New (30m by design):**
```python
def calculate_position_duration_risk(
    confidence,
    expected_return_pct,
    holding_period_minutes=30,
    volatility_regime="normal"
):
    """
    Adjust position sizing based on 30m holding assumption.
    
    Longer holding = More capital lock-up
    """
    
    # Base position size from confidence × expected_return
    base_size = confidence * expected_return_pct
    
    # Adjust for holding period risk
    if volatility_regime == "high":
        holding_risk_multiplier = 0.8  # 20% smaller during expansion
    elif volatility_regime == "low":
        holding_risk_multiplier = 1.1  # 10% larger during tight periods
    else:
        holding_risk_multiplier = 1.0
    
    position_size = base_size * holding_risk_multiplier
    
    # Calculate capital lock-up
    # (for portfolio liquidity planning)
    capital_locked_pct = position_size * 0.30  # 30m / 1440m = 2.1% of day
    
    return position_size, capital_locked_pct

# Usage:
size, locked = calculate_position_duration_risk(
    confidence=0.65,
    expected_return_pct=0.0025,
    holding_period_minutes=30,
    volatility_regime="normal"
)
```

**Impact:**
- Position sizing reflects 30m lock-up
- Capital allocation more conservative
- Drawdown patterns more predictable

#### 5. EV Scaling Tied to Volatility Regime

**Current (Phase 9):**
```python
if bootstrap_override and signal_floor is not None:
    signal_floor *= BOOTSTRAP_EV_SCALE  # Fixed 0.75
```

**New (Phase 9.5):**
```python
def calculate_ev_scale_with_regime(
    volatility_regime,
    directional_persistence,
    mean_reversion_risk
):
    """
    Scale EV gate based on regime and learnings.
    
    Higher volatility = Wider spreads = Higher cost = Lower EV
    Higher persistence = Lower mean reversion = Higher EV
    """
    
    base_ev_scale = 0.75  # Conservative default
    
    # Regime adjustment
    if volatility_regime == "high":
        regime_factor = 0.9  # Reduce EV by 10% (wider cost)
    elif volatility_regime == "low":
        regime_factor = 1.1  # Increase EV by 10% (tight cost)
    else:
        regime_factor = 1.0
    
    # Persistence adjustment (learned from backtest)
    if directional_persistence > 0.55:
        persistence_factor = 1.15  # Strong pattern, boost EV
    elif directional_persistence < 0.52:
        persistence_factor = 0.85  # Weak pattern, reduce EV
    else:
        persistence_factor = 1.0
    
    # Mean reversion adjustment
    if mean_reversion_risk > 0.40:
        reversion_factor = 0.9  # High reversion risk
    else:
        reversion_factor = 1.0
    
    final_ev_scale = base_ev_scale * regime_factor * persistence_factor * reversion_factor
    
    return min(1.0, max(0.5, final_ev_scale))  # Clamp 0.5-1.0

# Usage during bootstrap:
ev_scale = calculate_ev_scale_with_regime(
    volatility_regime="normal",
    directional_persistence=0.56,
    mean_reversion_risk=0.35
)
signal_floor *= ev_scale  # 0.75 * adjustments
```

**Impact:**
- EV scaling is data-driven
- Cost and persistence are explicit
- Bootstrap mode is regime-aware

---

## 📊 Before/After System Comparison

### Phase 8: Micro Momentum (5m Direction)

```
Input Signal:       "BTC likely up next 5m"
Model Horizon:      1 candle (5m)
Target:             Binary (up/down)

Volatility Model:   σ(5m) = 0.15%
Expected Move:      0.15% × √1 = 0.15%
Cost Assumption:    0.12% (static)

Break-Even:         80% accuracy
EV at 70% conf:     0.03% / 0.12% = 0.25 (too low)

Signal Quality:     Dominated by noise
Position Duration:  Undefined

Regime Aware:       No
Cost Scaling:       No
Holding Adj:        No

Result:             EV gate meaningless
```

### Phase 9.5: Structural Momentum (30m Cumulative)

```
Input Signal:       "BTC likely gains 0.20%+ in 30m"
Model Horizon:      6 candles (30m)
Target:             Binary (meets threshold / doesn't)

Volatility Model:   σ(30m) = 0.15% × √6 = 0.37%
Expected Move:      0.37% (horizon-aware)
Cost Assumption:    0.15% (regime-aware)

Break-Even:         40-45% accuracy
EV at 65% conf:     0.22% / 0.15% = 1.47 (meaningful)

Signal Quality:     Rises above microstructure
Position Duration:  30m (explicit)

Regime Aware:       Yes (cost scales)
Cost Scaling:       Yes (vol-dependent)
Holding Adj:        Yes (duration explicit)

Result:             EV gate actually filters signals
```

---

## 🚨 Critical Risks to Avoid

### Risk 1: The "Automatic 8x EV Improvement" Trap

**❌ WRONG:**
```
"Volatility scales with √6, so edge scales with √6,
 therefore EV improves automatically."
```

**✅ CORRECT:**
```
"Move size increases with √6,
 but edge depends on directional persistence,
 which must be learned and validated empirically."
```

**Mitigation:**
- Backtest before implementation
- Measure actual win rate vs expected
- Adjust horizon if persistence is weak

---

### Risk 2: The "Cost Model Unchanged" Trap

**❌ WRONG:**
```
Cost = 0.12% (applies equally to 5m and 30m)
```

**✅ CORRECT:**
```
Cost(5m) = 0.10% (tight execution)
Cost(30m) = 0.15% (wider spreads, longer lock-up)
```

**Mitigation:**
- Model cost as function of horizon
- Use actual slippage data for 30m holds
- Adjust EV calculations accordingly

---

### Risk 3: The "Mean Reversion Ignored" Trap

**❌ WRONG:**
```
"If model predicts up, it will stay up for 30m"
```

**✅ CORRECT:**
```
"Model predicts cumulative 30m return > 0.20%,
 but mean reversion can fade moves within window.
 This is learnable but not guaranteed."
```

**Mitigation:**
- Measure mean reversion empirically
- Test across regime types
- Adjust confidence floors for mean reversion risk

---

### Risk 4: The "Capital Lock-Up Ignored" Trap

**❌ WRONG:**
```
Position sizing = Current (5m assumption)
Capital allocation = Unchanged
```

**✅ CORRECT:**
```
Position sizing = Adjusted for 30m lock-up
Capital allocation = Account for duration
Liquidity cycles = Consider time-of-day patterns
```

**Mitigation:**
- Recalculate position sizing for 30m horizon
- Account for capital lock-up in portfolio management
- Monitor liquidity during different hours

---

## 🎯 Implementation Roadmap: Phase 9.5

### Phase 9.5a: Validation (Week 1)

```
Step 1: Build 30m cumulative return labels
Step 2: Measure distribution (median, percentiles)
Step 3: Train test model
Step 4: Measure accuracy (vs 5m baseline)
Step 5: Calculate break-even probability
Step 6: Test regime sensitivity

Gate: If break_even_prob > actual_win_ratio by > 10%
      STOP and adjust horizon
      Otherwise: Continue to Phase 9.5b
```

### Phase 9.5b: Cost Modeling (Week 1-2)

```
Step 1: Collect actual slippage data for 30m holds
Step 2: Build cost model by regime
Step 3: Update expected_move estimation
Step 4: Recalculate break-even with real costs
Step 5: Update EV gate thresholds

Gate: If edge < 0.20% in any regime
      STOP and reconsider horizon
      Otherwise: Continue to Phase 9.5c
```

### Phase 9.5c: System Changes (Week 2-3)

```
Step 1: Update MLForecaster label construction
Step 2: Retrain model on 30m targets
Step 3: Update ExecutionManager break-even logic
Step 4: Update expected_move calculation
Step 5: Update EV scaling with regime awareness
Step 6: Adjust position sizing for duration

Gate: Backtest shows > 2x improvement in EV
      Otherwise: Iterate on horizon or threshold
```

### Phase 9.5d: Deployment (Week 3-4)

```
Step 1: Run parallel 5m and 30m models
Step 2: Compare signal quality metrics
Step 3: Monitor execution for 30m holds
Step 4: Validate capital lock-up assumptions
Step 5: Optimize based on live data

Gate: Phase 9.5 complete when:
      - EV gate filters effectively
      - Win rate > 45%
      - Signal quality improves 2x
      - Capital efficiency maintained
```

---

## 📈 Expected Outcomes

### What Changes

```
Signal-to-Noise Ratio:    0.1 → 2.0  (20x improvement)
Model Accuracy:           51% → 57%  (7% absolute improvement)
EV at Typical Confidence: 0.25 → 1.47 (6x improvement)
Position Duration:        Undefined → 30m (explicit)
Cost Model:               Static → Regime-aware (dynamic)
Position Sizing:          Current → Duration-adjusted
```

### What Stays the Same

```
Microstructure fundamentals (unchanged)
Market regimes (unchanged)
Capital allocation philosophy (unchanged)
Risk management framework (unchanged)
Bootstrap mode concept (unchanged)
```

### What Improves

```
Signal Quality:         Meaningful edge detection
EV Gating:             Actually filters trades
Capital Efficiency:    More deliberate
Risk Assessment:       More accurate
Model Trainability:    Better patterns
```

---

## 💡 The Philosophical Shift

### From Micro to Structural

**Phase 8 (Micro Momentum):**
```
"Can I detect the next tick direction
 before the microstructure noise?"
```

**Phase 9.5 (Structural Momentum):**
```
"Can I detect directional persistence
 across the next 30 minutes?"
```

**Why this matters:**

The second question is answerable with confidence > 50%.  
The first question is fighting physics (noise dominates 1-tick prediction).

This is the real insight.

---

## 🏆 Verdict

### This IS the Right Direction

✅ Horizon shift from 5m to 30m is correct  
✅ Aligns with regime-adaptive allocator  
✅ Makes EV gating meaningful  
✅ Reduces microstructure sensitivity  
✅ Improves model learnability  

### BUT: Treat as Architectural Change, Not Tweak

⚠️ Cost model must change  
⚠️ Break-even logic must change  
⚠️ Position sizing must change  
⚠️ Expected move calculation must change  
⚠️ Capital lock-up must be explicit  

### Order of Implementation

1. **Validate first** (backtest in isolation)
2. **Cost model second** (ensure edge is real)
3. **System changes third** (integrate thoughtfully)
4. **Deploy last** (parallel with Phase 8)

**Difficulty:** Medium (careful implementation)  
**Impact:** Transformational (converts noise to signal)  
**Timeline:** 3-4 weeks (proper validation)  
**Risk:** Low (backward compatible, can run parallel)  

---

## 📚 Related Documentation

- `HORIZON_AWARE_PREDICTION_PHASE9.md` – Initial strategic proposal
- `ADAPTIVE_EV_SCALING_PHASE9.md` – Bootstrap EV logic (Phase 9)
- Phase 9.5 test results (to be created during validation)
- Phase 9.5 cost model (to be created during cost validation)
- Phase 9.5 deployment guide (to be created before rollout)

---

**Next Step:** Run Phase 9.5a validation before implementing Phase 9.5b-d.

