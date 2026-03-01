# 🎯 Horizon-Aware Prediction: Phase 9 Strategic Enhancement

**Date:** February 21, 2026  
**Status:** ✅ ANALYSIS & RECOMMENDATION  
**Impact:** Transforms EV gating from theoretical to practically meaningful  

---

## 🚀 The Insight

**Current Problem:**
- MLForecaster predicts: "Next 1 candle (5m) direction"
- Expected move: ~0.15% (too small to cover costs 0.12%)
- EV gate meaningless (expected_move barely covers cost)
- Signal confidence disconnected from reality

**Your Solution:**
- Predict: "Cumulative return over 3-6 candles (15m-30m)"
- Expected move: 0.30%-0.60% (exceeds cost by 2.5-5x)
- EV gate becomes meaningful (~100 bps edge is realistic)
- Signal confidence reflects time-adjusted volatility

**Physics:**
```
Volatility scales with sqrt(time)
─────────────────────────────────

5m move  ≈ 0.15%
20m move ≈ 0.15% × √4 = 0.30%
30m move ≈ 0.15% × √6 = 0.37%
60m move ≈ 0.15% × √12 = 0.52%

Cost = 0.12%
Edge at 5m = 0.15% - 0.12% = 0.03%  ← Tight!
Edge at 30m = 0.37% - 0.12% = 0.25% ← Breathing room!
```

---

## 📊 Current Architecture (Phase 8)

### Model Training
**File:** `model_trainer.py`  
**Input:** 50 timesteps × 29 features  
**Target:** Binary classification (0/1)  
**Meaning:** ??? (Unclear what determines the target)  

```python
# Current (unclear target definition)
y.append(targets[i + self.timesteps])  # What is this?
```

**Problems:**
- Target definition not shown in provided code
- Likely based on next 1-candle direction
- Expected move too small
- EV gate doesn't provide meaningful gating

### Prediction Horizon
```
Timeline:
─────────────────────────────
Lookback: 50 × 5m = 250m (4.2h)
Predict:  1 candle = 5m (next)
Horizon:  5m into future
```

---

## ✅ Proposed Architecture (Phase 9+)

### Model Training with Multi-Horizon Targets

**Concept:** Train on cumulative return over multiple horizons

```python
def _build_multi_horizon_targets(
    df: pd.DataFrame,
    horizon_candles: int = 6  # 30m on 5m candles
) -> np.ndarray:
    """
    Create binary targets for cumulative move over horizon.
    
    horizon_candles = 6 means:
    - Look forward 6 candles (30 minutes on 5m data)
    - Calculate cumulative return
    - Target = 1 if return > threshold, else 0
    """
    close = df["close"].values
    targets = []
    
    for i in range(len(close) - horizon_candles):
        entry = close[i]
        exit_price = close[i + horizon_candles]
        return_pct = (exit_price - entry) / entry
        
        # Threshold = cost + minimal edge (0.12% + 0.08% = 0.20%)
        min_edge = 0.0020  # 20 bps minimum edge
        target = 1 if return_pct > min_edge else 0
        targets.append(target)
    
    return np.array(targets, dtype=np.int32)
```

### Key Changes

| Aspect | Phase 8 (Current) | Phase 9 (Proposed) |
|--------|-------------------|-------------------|
| **Prediction** | Next 1 candle | Next 6 candles |
| **Horizon** | 5 minutes | 30 minutes |
| **Expected Move** | 0.15% | 0.30%-0.60% |
| **EV Threshold** | 0.03% (tight) | 0.25%-0.50% (clear) |
| **Cost Coverage** | 1.25x | 2.5-5x |
| **Signal Confidence** | Miscalibrated | Realistic |
| **Model Target** | ??? | Clear (cumulative return) |

---

## 🔧 Implementation Roadmap

### Step 1: Add Horizon Configuration

**File:** `agents/ml_forecaster.py`

```python
class MLForecaster:
    def __init__(self, ...):
        # NEW: Horizon configuration
        self.prediction_horizon_candles = int(
            kwargs.get("horizon_candles", 6)  # 6 × 5m = 30m
        )
        self.horizon_min_edge_pct = float(
            self._cfg("ML_HORIZON_MIN_EDGE_PCT", 0.0020)  # 20 bps
        )
        
        # Keep existing for compatibility
        self.timeframe = kwargs.get("timeframe", "5m")
```

### Step 2: Update Target Generation

**File:** `agents/ml_forecaster.py` → Add new method

```python
def _build_edge_feature_frame_with_horizon_target(
    self,
    ohlcv: List[Any],
    horizon_candles: int = 6,
    min_edge_pct: float = 0.0020,
) -> pd.DataFrame:
    """
    Build feature frame AND horizon-aware target column.
    
    Steps:
    1. Convert OHLCV to DataFrame with features (existing)
    2. Calculate cumulative returns over horizon
    3. Create binary target (1 if meets min_edge, else 0)
    """
    # Step 1: Build existing feature frame
    df = self._build_edge_feature_frame(ohlcv)
    if df.empty:
        return df
    
    # Step 2: Calculate forward-looking cumulative return
    close = pd.to_numeric(df["close"], errors="coerce")
    targets = []
    
    for i in range(len(df)):
        if i + horizon_candles >= len(df):
            # Not enough future data (edge case)
            targets.append(0)  # Default to no signal
        else:
            entry_price = float(close.iloc[i])
            exit_price = float(close.iloc[i + horizon_candles])
            
            if entry_price <= 0:
                targets.append(0)
            else:
                cumulative_return = (exit_price - entry_price) / entry_price
                target = 1 if cumulative_return > min_edge_pct else 0
                targets.append(target)
    
    df["target"] = targets
    
    # Log target distribution
    target_arr = np.array(targets)
    pos_pct = (target_arr.sum() / len(target_arr) * 100) if len(target_arr) > 0 else 0
    self.logger.info(
        f"[{self.name}] Horizon target: candles={horizon_candles} "
        f"min_edge={min_edge_pct*100:.2f}% "
        f"positive_ratio={pos_pct:.1f}%"
    )
    
    return df
```

### Step 3: Update Retrain to Use New Targets

**File:** `agents/ml_forecaster.py` → Modify `retrain()` method

```python
async def retrain(self) -> Dict[str, Any]:
    # ... existing code ...
    
    for idx, sym in enumerate(symbols):
        # ... fetch OHLCV ...
        
        # NEW: Use horizon-aware target generation
        feature_df = self._build_edge_feature_frame_with_horizon_target(
            ohlcv,
            horizon_candles=self.prediction_horizon_candles,
            min_edge_pct=self.horizon_min_edge_pct,
        )
        
        # ... rest of training code ...
```

### Step 4: Update Expected Move Calculation

**File:** `agents/ml_forecaster.py` → Modify existing method

```python
def _estimate_expected_move_pct_from_rows(
    self,
    rows: List[List[float]],
    horizon_steps: int = None  # Will derive from horizon_candles
) -> float:
    """
    Estimate expected move for the PREDICTION HORIZON.
    
    Before: horizon_steps was arbitrary (conf_horizon_min)
    Now:    horizon_steps = self.prediction_horizon_candles
    """
    if horizon_steps is None:
        horizon_steps = self.prediction_horizon_candles
    
    if not rows:
        return float(self._expected_move_fallback_pct)
    
    close = float(rows[-1][3] or 0.0)
    if close <= 0:
        return float(self._expected_move_fallback_pct)
    
    atr = self._atr_from_numeric_rows(rows, lookback=14)
    atr_pct = (atr / close) if atr > 0 else 0.0
    rv_pct = self._rv_pct_from_numeric_rows(rows, lookback=20)
    
    # CHANGED: Use actual prediction horizon
    h = max(1.0, float(horizon_steps))  # 6 for 30m on 5m data
    h_sqrt = float(np.sqrt(h))
    
    atr_move = atr_pct * max(1.0, min(3.0, h_sqrt))  # sqrt(6) ≈ 2.45
    rv_move = rv_pct * h_sqrt
    
    expected = (0.6 * atr_move) + (0.4 * rv_move)
    
    if expected <= 0:
        expected = float(self._expected_move_fallback_pct)
    
    expected = max(
        float(self._expected_move_min_pct),
        min(float(self._expected_move_max_pct), float(expected))
    )
    
    self.logger.debug(
        f"[{self.name}] Expected move: horizon={horizon_steps} "
        f"atr_pct={atr_pct:.4f} rv_pct={rv_pct:.4f} "
        f"expected={expected:.4f} (√{h:.1f}={h_sqrt:.2f})"
    )
    
    return float(expected)
```

### Step 5: Update Signal Emission

**File:** `agents/ml_forecaster.py` → Signal registration

```python
async def _emit_signal(self, symbol: str, action: str, confidence: float, ...):
    # Include horizon information in signal
    signal = {
        "agent": "MLForecaster",
        "action": action,
        "confidence": confidence,
        "symbol": symbol,
        "timestamp": time.time(),
        
        # NEW: Horizon-aware metadata
        "prediction_horizon_candles": self.prediction_horizon_candles,
        "prediction_horizon_minutes": self.prediction_horizon_candles * 5,
        "expected_move_pct": expected_move_pct,
        "min_edge_pct": self.horizon_min_edge_pct,
    }
    
    await self.meta_controller.receive_signal(
        self.name,
        symbol,
        signal,
        confidence,
    )
```

---

## 📊 Mathematical Justification

### Volatility Scaling

**Formula:**
```
σ(t) = σ₁ × √t

where:
  σ(t) = volatility over time period t
  σ₁ = 1-period volatility (5m)
  t = number of periods
```

**Example (BTC/USDT typical):**
```
1-period volatility (5m):    0.15%
────────────────────────────────

Horizon    Periods  √periods  Expected_Move  Cost   Edge    Ratio
─────────────────────────────────────────────────────────────────
5m         1        1.0       0.15%          0.12%  0.03%   1.25x
10m        2        1.41      0.21%          0.12%  0.09%   1.75x
15m        3        1.73      0.26%          0.12%  0.14%   2.17x
20m        4        2.0       0.30%          0.12%  0.18%   2.50x
30m        6        2.45      0.37%          0.12%  0.25%   3.08x
60m        12       3.46      0.52%          0.12%  0.40%   4.33x
```

### Why This Matters for EV Gating

**Before (5m prediction):**
```
Signal confidence = 0.75 (model output)
Expected move = 0.15%
Cost = 0.12%
Edge = 0.03%

EV = Edge × Confidence / Cost
   = 0.03% × 0.75 / 0.12%
   = 0.1875
   
Interpretation: Barely positive (1.25x cost)
Gate: Meaningless (0.75 confidence gates nothing)
```

**After (30m prediction):**
```
Signal confidence = 0.65 (lower, harder to predict 30m)
Expected move = 0.37% (sqrt scaling)
Cost = 0.12%
Edge = 0.25%

EV = Edge × Confidence / Cost
   = 0.25% × 0.65 / 0.12%
   = 1.354
   
Interpretation: Clear positive (3.1x cost)
Gate: Meaningful (0.65 confidence gates weak signals)
```

---

## 🧪 Testing Strategy

### Test 1: Target Distribution
```python
# Before implementing full model training
# Just check target generation

horizon_candles = 6
min_edge = 0.0020

positive_ratio = sum(targets) / len(targets)
print(f"Positive ratio: {positive_ratio:.1%}")  # Should be ~30-40%
```

**Expected:** 30-40% positive (not 50%, because edge threshold is real)

### Test 2: Expected Move Calculation
```python
# Verify sqrt scaling
for candles in [1, 3, 6, 12]:
    move = _estimate_expected_move_pct_from_rows(rows, candles)
    print(f"{candles} candles: {move:.4f} (expected ~{0.15*np.sqrt(candles):.4f})")
```

**Expected:** Linear increase with √candles

### Test 3: Model Performance Comparison
```python
# Train two models:
# 1. Original (1-candle, 5m)
# 2. Horizon-aware (6-candle, 30m)

# Compare on same test set:
- Prediction accuracy
- Calibration (confidence vs actual)
- EV consistency
```

---

## 🚀 Deployment

### Configuration

Add to config:

```json
{
  "ML_PREDICTION_HORIZON_CANDLES": 6,
  "ML_HORIZON_MIN_EDGE_PCT": 0.0020,
  "ML_EXPECTED_MOVE_MIN_PCT": 0.0015,
  "ML_EXPECTED_MOVE_MAX_PCT": 0.0500
}
```

### Migration Path

**Phase 1:** Run both models in parallel
- Existing: 5m prediction
- New: 30m prediction
- Compare signals

**Phase 2:** Switch to horizon-aware
- Retrain on new target definition
- Monitor EV gate effectiveness
- Verify confidence calibration

**Phase 3:** Optimize horizon
- Test different horizons (15m, 20m, 30m, 60m)
- Find optimal balance of:
  - Predictability (shorter = easier)
  - Edge size (longer = bigger)
  - Holding period (operational constraint)

---

## 💡 Why This Fixes Signal Quality

### Current Problem (Phase 8)
```
Model predicts: "Will BTC go up in next 5m?"
- Too noisy (1 candle is random)
- Expected move < cost
- Confidence not meaningful
- EV gate useless
```

### Solution (Phase 9+)
```
Model predicts: "Will BTC gain >0.20% over next 30m?"
- Clearer signal (6 candles aggregate noise)
- Expected move > cost (by 3x)
- Confidence reflects real edge
- EV gate provides real filtering
```

---

## 📈 Impact Summary

| Metric | Phase 8 | Phase 9 | Improvement |
|--------|---------|---------|-------------|
| **Prediction Horizon** | 5m | 30m | 6x longer |
| **Expected Move** | 0.15% | 0.37% | 2.5x larger |
| **Cost Coverage** | 1.25x | 3.1x | 2.5x better |
| **EV Gate** | Useless | Meaningful | Productive |
| **Signal Confidence** | Miscalibrated | Realistic | Reliable |
| **Model Difficulty** | Hard | Moderate | Better balance |

---

## ✅ Next Steps

1. **Implement** `_build_edge_feature_frame_with_horizon_target()`
2. **Update** `retrain()` to use new target generation
3. **Modify** `_estimate_expected_move_pct_from_rows()` for horizon
4. **Test** on historical data (backtest)
5. **Compare** Phase 8 vs Phase 9 signal quality
6. **Deploy** when confident

---

## 🎯 Summary

Your insight is **exactly right**: predicting 30m cumulative return is far more useful than predicting next 5m direction because:

1. ✅ Volatility scales with √time (gives bigger edge)
2. ✅ EV gate becomes meaningful (can actually filter signals)
3. ✅ Signal confidence is realistic (harder to predict 30m accurately)
4. ✅ Execution is simpler (30m is reasonable hold period)
5. ✅ Model is more trainable (6-candle aggregate = less noise)

This transforms MLForecaster from a "noisy directional predictor" to a "reliable edge detector."

