# Regime-Based Scaling Architecture (Superior to Binary Gating)

## The Problem with Binary Gates

Binary regime gates (block everything in X regime) are crude and lose alpha:

```python
# ❌ OLD: All-or-nothing
if regime == "bear":
    return  # Block entire signal
# Result: Miss 100% of profitable bear market trades
```

## The Solution: Regime-Based Scaling

Scale trade parameters based on regime instead of blocking. Allows nuanced risk adjustment:

```python
# ✅ NEW: Adaptive scaling
regime_scaling = {
    "position_size_mult": 0.60,      # 60% of normal size in bear
    "tp_target_mult": 0.80,          # 80% of normal TP
    "excursion_requirement_mult": 1.2,  # Harder to trigger
    "trail_mult": 0.95,              # Tighter stops
    "confidence_boost": -0.08,       # Confidence penalty
}
# Result: Scale down but still capture valid bear trades
```

---

## Implementation in TrendHunter

### 1. Regime Scaling Function (New Method)

```python
def _get_regime_scaling_factors(self, regime: str) -> Dict[str, float]:
    """
    Returns scaling factors for each regime.
    All multipliers are factors applied to base values.
    """
```

#### Regime Classifications and Scaling

| Regime | Position Size | TP Target | Excursion | Trail | Confidence Boost |
|--------|---------------|-----------|-----------|-------|-----------------|
| **trend/uptrend/downtrend** | 1.00 (full) | 1.00 (full) | 0.85 (easier) | 1.30 (aggressive) | +5% |
| **high_vol/high** | 0.80 (80%) | 1.05 (slightly wider) | 1.00 (normal) | 1.20 (moderately aggressive) | 0% |
| **sideways/chop/range** | 0.50 (50%) | 0.60 (reduced) | 1.40 (harder) | 0.90 (tight) | -5% |
| **bear/bearish** | 0.60 (60%) | 0.80 (reduced) | 1.20 (harder) | 0.95 (tight) | -8% |
| **normal/neutral** | 1.00 (baseline) | 1.00 (baseline) | 1.00 (baseline) | 1.00 (baseline) | 0% |

### 2. Signal Submission with Scaling

```python
async def _submit_signal(self, symbol: str, action: str, confidence: float, reason: str):
    # Get 1h regime (brain)
    regime_1h = await get_volatility_regime(symbol, "1h")
    
    # Get scaling factors (not binary gate)
    regime_scaling = self._get_regime_scaling_factors(regime_1h)
    
    # Apply confidence adjustment
    adjusted_confidence = confidence + regime_scaling["confidence_boost"]
    
    # Still reject if adjusted confidence too low
    if adjusted_confidence < min_conf:
        return  # Block if confidence fails even after boost
    
    # Include scaling in signal
    signal = {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "_regime_scaling": regime_scaling,  # ← Pass to MetaController
        "_regime": regime_1h,
        ...
    }
```

### 3. Signal Includes Regime Metadata

Each BUY signal now contains:

```python
signal = {
    "symbol": "ETHUSDT",
    "action": "BUY",
    "confidence": 0.75,
    "quote_hint": 100.0,
    
    # NEW: Regime scaling factors
    "_regime": "sideways",
    "_regime_scaling": {
        "position_size_mult": 0.50,
        "tp_target_mult": 0.60,
        "excursion_requirement_mult": 1.40,
        "trail_mult": 0.90,
        "confidence_boost": -0.05,
    },
}
```

---

## How MetaController/ExecutionManager Uses Scaling

### Position Size Adjustment

```python
# In MetaController._execute_decision() or ExecutionManager.create_order()
base_quote = signal.get("quote_hint", 100.0)
regime_scaling = signal.get("_regime_scaling", {})
position_size_mult = regime_scaling.get("position_size_mult", 1.0)

adjusted_quote = base_quote * position_size_mult
# If signal suggests $100 but regime is sideways (0.50 mult)
# → actual order size = $50
```

### TP/SL Target Adjustment

```python
# In TPSLEngine.calculate_tp_sl()
# TP distance from volatility profile: 1.5% of entry
base_tp_pct = 0.015

regime_scaling = signal.get("_regime_scaling", {})
tp_mult = regime_scaling.get("tp_target_mult", 1.0)

adjusted_tp_pct = base_tp_pct * tp_mult
# If base is 1.5% but regime is sideways (0.60 mult)
# → actual TP = 0.9% (tighter target)
```

### Excursion Gate Adjustment

```python
# In TPSLEngine._passes_excursion_gate()
# Minimum price movement to qualify as valid TP
base_threshold = atr * 0.35

regime_scaling = signal.get("_regime_scaling", {})
excursion_mult = regime_scaling.get("excursion_requirement_mult", 1.0)

adjusted_threshold = base_threshold * excursion_mult
# If threshold is 100 bps but regime is sideways (1.40 mult)
# → required movement = 140 bps (harder to trigger)
```

### Trailing Stop Aggressiveness

```python
# In TPSLEngine.check_orders() trailing logic
base_trail_mult = 1.5

regime_scaling = signal.get("_regime_scaling", {})
trail_mult = regime_scaling.get("trail_mult", 1.0)

adjusted_trail = base_trail_mult * trail_mult
# If base is 1.5x ATR but regime is sideways (0.90 mult)
# → trailing = 1.35x ATR (tighter, follows closer)
```

---

## Benefits vs Binary Gating

| Aspect | Binary Gate | Regime Scaling |
|--------|------------|----------------|
| **Alpha capture** | 0% in excluded regimes | Full alpha in all regimes |
| **Risk management** | All-or-nothing | Granular per-regime |
| **Position sizing** | Block or execute full | Scale based on regime |
| **TP/SL targets** | Same for all regimes | Regime-adjusted |
| **Adaptability** | Fixed | Dynamic based on market |
| **Configurability** | Block/allow | Fine-grained tuning |
| **Logging clarity** | "BUY blocked" | "BUY allowed, pos=50%, TP=0.9%" |

---

## Concrete Examples

### Example 1: Sideways Regime

**Signal**: ETHUSDT BUY, confidence=0.72, quote_hint=$100, expected_move=+0.8%

**Regime Detection**: 1h regime = "sideways"

**Scaling Applied**:
- Position size: $100 × 0.50 = **$50** (50%)
- TP target: 1.5% × 0.60 = **0.9%** (reduced)
- Excursion requirement: 100 bps × 1.40 = **140 bps** (harder to trigger)
- Trail multiplier: 1.5 × 0.90 = **1.35** (tighter)
- Confidence adjustment: 0.72 - 0.05 = **0.67** (penalty)

**Outcome**: Trade executes at reduced size with tighter targets and wider trail. If sideways breaks, trailing catches it. If moves as expected, take smaller profit.

### Example 2: Trending Regime

**Signal**: BTCUSDT BUY, confidence=0.78, quote_hint=$100, expected_move=+1.2%

**Regime Detection**: 1h regime = "uptrend"

**Scaling Applied**:
- Position size: $100 × 1.00 = **$100** (full)
- TP target: 1.5% × 1.00 = **1.5%** (full)
- Excursion requirement: 100 bps × 0.85 = **85 bps** (easier)
- Trail multiplier: 1.5 × 1.30 = **1.95** (aggressive)
- Confidence adjustment: 0.78 + 0.05 = **0.83** (boost)

**Outcome**: Trade executes at full size with full targets and aggressive trailing. In trends, trailing stops follow closely and lock in gains as price rises.

### Example 3: High Volatility Regime

**Signal**: BNBUSDT BUY, confidence=0.65, quote_hint=$100, expected_move=+0.6%

**Regime Detection**: 1h regime = "high_vol"

**Scaling Applied**:
- Position size: $100 × 0.80 = **$80** (80%)
- TP target: 1.5% × 1.05 = **1.575%** (slightly wider)
- Excursion requirement: 100 bps × 1.00 = **100 bps** (normal)
- Trail multiplier: 1.5 × 1.20 = **1.80** (moderately aggressive)
- Confidence adjustment: 0.65 + 0.00 = **0.65** (no penalty)

**Outcome**: Reduce position size due to volatility, but widen TP slightly (more range to hit). Moderate trailing handles whipsaws better.

---

## Configuration

### TrendHunter Config

```python
# Regime scaling enable/disable
TREND_REGIME_SCALING_ENABLED = True

# Override default scaling (if needed)
TREND_SIDEWAYS_POS_SIZE_MULT = 0.50
TREND_SIDEWAYS_TP_MULT = 0.60
TREND_TRENDING_TRAIL_MULT = 1.30
TREND_ALLOW_BEAR_IF_HIGH_CONF = False  # Still block bear at low confidence

# Confidence adjustments by regime
TREND_SIDEWAYS_CONFIDENCE_BOOST = -0.05
TREND_HIGH_VOL_CONFIDENCE_BOOST = 0.0
TREND_TRENDING_CONFIDENCE_BOOST = 0.05
```

### MetaController Usage

```python
# When processing signal:
regime_scaling = signal.get("_regime_scaling", {})
if regime_scaling:
    # Apply all scaling factors
    position_size = base_quote * regime_scaling["position_size_mult"]
    # ... apply others
```

### TP/SL Engine Integration

```python
# When calculating TP/SL:
regime_scaling = trade_record.get("_regime_scaling", {})
if regime_scaling:
    tp_mult = regime_scaling.get("tp_target_mult", 1.0)
    trail_mult = regime_scaling.get("trail_mult", 1.0)
    # Apply to TP/SL calculations
```

---

## Architectural Flow

```
┌─────────────────────────────────────────────┐
│ TrendHunter (Agent)                         │
│ ├─ Detect 1h regime                         │
│ ├─ Get scaling factors per regime           │
│ ├─ Adjust confidence if needed              │
│ └─ Emit signal WITH _regime_scaling         │
└────────────┬────────────────────────────────┘
             │ (signal includes scaling factors)
             ▼
┌─────────────────────────────────────────────┐
│ MetaController                              │
│ ├─ Receive signal with _regime_scaling      │
│ ├─ Apply position_size_mult to quote        │
│ ├─ Apply other regime constraints           │
│ └─ Pass adjusted signal to ExecutionManager │
└────────────┬────────────────────────────────┘
             │ (adjusted position size)
             ▼
┌─────────────────────────────────────────────┐
│ ExecutionManager                            │
│ ├─ Place order with scaled size             │
│ ├─ Store _regime_scaling in trade record    │
│ └─ Fill order                               │
└────────────┬────────────────────────────────┘
             │ (position created with regime metadata)
             ▼
┌─────────────────────────────────────────────┐
│ TP/SL Engine                                │
│ ├─ Retrieve _regime_scaling from position   │
│ ├─ Apply tp_target_mult to TP calculation   │
│ ├─ Apply trail_mult to trailing SL          │
│ ├─ Apply excursion_mult to gate requirements│
│ └─ Monitor with regime-adjusted parameters  │
└─────────────────────────────────────────────┘
```

---

## Why This is Superior

1. **Captures Alpha**: Doesn't miss regime-specific opportunities
2. **Risk-Managed**: Scales down in unfavorable regimes without blocking
3. **Testable**: Each multiplier can be tuned and backtested
4. **Transparent**: Logs show exactly what scaling was applied
5. **Flexible**: Easy to override specific regime multipliers
6. **Consistent**: Same logic across all agents and regimes
7. **Extensible**: Easy to add new regimes or multipliers

---

## Next Steps

1. ✅ Implement regime scaling in TrendHunter._submit_signal()
2. ⏭️ Update MetaController to apply position_size_mult
3. ⏭️ Update TPSLEngine to apply tp_target_mult, trail_mult, excursion_mult
4. ⏭️ Add configuration options for each regime multiplier
5. ⏭️ Add logging to show scaling decisions
6. ⏭️ Backtest against historical regimes to tune multipliers
