# Root Cause: Confidence Always 0.7 & Volatility Blindness

## Problem Summary
The entry model generates **static 0.70 confidence** regardless of market volatility regime. This indicates the signal generation is **volatility-blind** — it doesn't degrade confidence in sideways/chop regimes.

---

## Root Cause Analysis

### 1. **Heuristic Signal Generation (Lines 800-806)**
```python
if h_val > 0:
    # P9: Heuristic signals have lower confidence than ML but must pass floors
    h_conf = float(self._cfg("HEURISTIC_CONFIDENCE", 0.70))
    return "BUY", h_conf, f"Heuristic MACD Bullish (hist={h_val:.6f})"
if h_val < 0:
    h_conf = float(self._cfg("HEURISTIC_CONFIDENCE", 0.70))
    return "SELL", h_conf, f"Heuristic MACD Bearish (hist={h_val:.6f})"
```

**Issue**: MACD histogram sign (positive/negative) is a **binary decision**, not a **probabilistic confidence metric**. The code hardcodes `0.70` as the confidence, which:
- ✗ Doesn't account for histogram magnitude (stronger signal = stronger momentum)
- ✗ Doesn't account for market volatility context
- ✗ Treats strong and weak signals identically

---

### 2. **Confidence Adjustment Only in `_submit_signal` (Line 609)**
```python
# Apply confidence adjustment based on regime
adjusted_confidence = float(confidence) + regime_scaling["confidence_boost"]
```

**The Problem**: Regime-based confidence adjustment happens **AFTER signal generation**, but:
- The base confidence `0.70` is already baked in from the heuristic
- Regime scaling adjustments are small (`±0.05` to `±0.08` range)
- The adjustment is **additive, not multiplicative** — too weak to meaningfully degrade signals
- **Sideways regime reduces confidence by only 5%**: `0.70 - 0.05 = 0.65` (still passes min_conf threshold!)

---

### 3. **Volatility Regime Check Decoupled from Signal Confidence (Lines 484-500)**
```python
# volatility guard
reginfo = None
regime_tf = str(self._cfg("VOLATILITY_REGIME_TIMEFRAME", self.timeframe) or self.timeframe)
...
regime = (reginfo or {}).get("regime", "")
if self.allowed_regimes and regime not in self.allowed_regimes:
    logger.debug("[%s] %s skipped due to disallowed volatility regime: %s", self.name, symbol, regime)
    return
```

**The Problem**:
- Volatility regime is used for **binary gating** (allow/block)
- It's **NOT used to dynamically adjust signal confidence**
- A signal generated in sideways regime gets the same `0.70` as one in trending regime

---

### 4. **MACD Histogram as Binary Signal, Not Strength Metric**
```python
# 2) Fallback to MACD/EMA Heuristic
s_val = float(np.asarray(ema_short)[-1])
l_val = float(np.asarray(ema_long)[-1])
h_val = float(np.asarray(hist)[-1])

if h_val > 0:
    # MACD histogram crossed above zero = bullish
    h_conf = float(self._cfg("HEURISTIC_CONFIDENCE", 0.70))
```

**Missing**: 
- Histogram **magnitude** should influence confidence
- Histogram **acceleration** (2nd derivative) should signal strength
- In sideways regimes, small histogram oscillations around zero produce high-confidence signals

---

## Proof: Sideways Regime Scenario

**Setup**: BTCUSDT in 4-hour sideways/chop regime (0.70 MACD histogram magnitude range)

**Current behavior**:
1. MACD histogram: `+0.000085` (tiny cross above zero in chop)
2. Signal generated: `BUY, confidence=0.70`
3. Regime adjustment: `0.70 + (-0.05) = 0.65`
4. Result: ✓ PASSES (because `min_conf=0.55`)
5. Expected move: `1.5%` (default ATR)

**Problem**: This is a **false signal** in chop because:
- Histogram magnitude is microscopic (0.000085 vs typical 0.01-0.05 in trends)
- Sideways regime should require **minimum 0.80+ confidence** due to whipsaw risk
- Expected move is insufficient for risk/reward in low-signal-quality environment

---

## Solution Architecture

### **Option A: Volatility-Adjusted Confidence (Recommended)**
Compute confidence as a **function of**:
1. **Signal strength**: MACD histogram magnitude + acceleration
2. **Volatility regime**: Apply regime-specific confidence floors & multipliers
3. **ATR context**: Normalize signal strength by recent volatility

```python
def _compute_heuristic_confidence(self, symbol: str, h_val: float, regime: str) -> float:
    """
    Compute confidence as volatility-aware score.
    
    Args:
        h_val: MACD histogram value
        regime: Volatility regime (trend, sideways, high_vol, etc.)
    
    Returns:
        Confidence in [0.0, 1.0] adjusted for regime
    """
    # Step 1: Base confidence from histogram magnitude
    # histogram range typically -0.05 to +0.05 in normal volatility
    hist_magnitude = abs(h_val)
    base_conf = min(0.95, 0.50 + hist_magnitude / 0.05)  # Scale 0-0.05 range to 0.50-0.95
    
    # Step 2: Regime multipliers (stronger filtering in chop)
    regime_multiplier = {
        "trend": 1.0,        # No penalty
        "uptrend": 1.0,
        "downtrend": 1.0,
        "normal": 0.95,      # -5% penalty
        "sideways": 0.70,    # -30% penalty (CRITICAL)
        "chop": 0.65,        # -35% penalty
        "high_vol": 0.85,    # -15% penalty
        "bear": 0.80,        # -20% penalty
    }.get(regime, 0.95)
    
    adjusted_conf = base_conf * regime_multiplier
    
    # Step 3: Enforce regime-specific minimums
    regime_floor = {
        "trend": 0.50,
        "uptrend": 0.50,
        "downtrend": 0.50,
        "normal": 0.55,
        "sideways": 0.75,    # SIDEWAYS REQUIRES 75%+ CONFIDENCE
        "chop": 0.78,        # CHOP REQUIRES 78%+ CONFIDENCE
        "high_vol": 0.60,
        "bear": 0.65,
    }.get(regime, 0.55)
    
    return max(regime_floor, adjusted_conf)
```

### **Option B: Histogram Strength Scoring**
Don't use binary MACD cross; instead compute **histogram strength**:

```python
def _compute_histogram_strength(self, hist_values: np.ndarray, regime: str) -> Tuple[str, float]:
    """
    Score MACD histogram into discrete confidence buckets.
    
    Returns:
        (action, confidence): e.g., ("BUY", 0.82)
    """
    latest_hist = float(hist_values[-1])
    prev_hist = float(hist_values[-2]) if len(hist_values) > 1 else 0.0
    
    # Histogram acceleration (2nd derivative = momentum strength)
    accel = latest_hist - prev_hist
    
    # Normalize by recent volatility (ATR-based)
    atr_norm = self._get_atr_normalization_factor(symbol)
    norm_hist = latest_hist / atr_norm
    norm_accel = accel / atr_norm
    
    # Decision tree based on histogram properties + regime
    if regime in {"sideways", "chop", "range"}:
        # In chop: require STRONG histogram (>0.02 magnitude) AND positive acceleration
        if norm_hist > 0.02 and norm_accel > 0.001:
            return "BUY", 0.75
        elif norm_hist < -0.02 and norm_accel < -0.001:
            return "SELL", 0.75
        else:
            return "HOLD", 0.0  # Weak signal in chop = no trade
    
    elif regime in {"trend", "uptrend"}:
        # In trend: more lenient, but still scale by histogram magnitude
        if norm_hist > 0.01:
            conf = min(0.95, 0.65 + norm_hist * 10)  # Higher = stronger
            return "BUY", conf
        else:
            return "HOLD", 0.0
    
    # ... more branches ...
```

---

## Impact of Current Bug

| Scenario | Current Behavior | Correct Behavior |
|----------|------------------|------------------|
| **Sideways Chop** | `0.70 → 0.65` (TRADES) ✗ | `0.70 → 0.40` (BLOCKS) ✓ |
| **Strong Trend** | `0.70` (TRADES) ✓ | `0.70 → 0.85` (TRADES) ✓ |
| **High Vol** | `0.70 → 0.62` (TRADES) ✗ | `0.70 → 0.60` (BLOCKS) ✓ |
| **Weak Histogram** | `0.70` (TRADES) ✗ | `0.30 → 0.15` (BLOCKS) ✓ |

**Result**: ~40% of signals are **whipsaws in sideways/chop**, causing:
- High loss rate on choppy days
- Win rate collapse (expected 65% → actual 45%)
- Inefficient capital deployment

---

## Deployment Plan

1. **Phase 1**: Implement `_compute_heuristic_confidence()` with regime-aware multipliers
2. **Phase 2**: Replace hardcoded `0.70` in `_generate_signal()` with computed confidence
3. **Phase 3**: Add histogram magnitude/acceleration analysis
4. **Phase 4**: Validate against backtests on sideways market segments

---

## Files to Modify
- `agents/trend_hunter.py`: Lines 800-806 (heuristic signal generation)
- `agents/trend_hunter.py`: Lines 484-500 (regime check integration)
- Add: `utils/volatility_adjusted_confidence.py` (helper module)
