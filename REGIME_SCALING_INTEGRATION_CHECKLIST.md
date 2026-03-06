# Regime-Based Scaling Integration Checklist

## Status: PHASE 1 COMPLETE ✅

Regime-based scaling has been implemented in TrendHunter. Now integrating downstream.

---

## PHASE 1: TrendHunter Implementation ✅ COMPLETE

### ✅ 1. Added `_get_regime_scaling_factors()` Method
- **Location**: `agents/trend_hunter.py` lines 503-584
- **Implements**: 5 regime types with scaling multipliers
- **Regimes**:
  - ✅ trending (uptrend/downtrend)
  - ✅ high_vol (high volatility)
  - ✅ sideways (sideways/chop/range)
  - ✅ bear (defensive)
  - ✅ normal (default/unknown)
- **Returns**: Dict with position_size_mult, tp_target_mult, excursion_requirement_mult, trail_mult, confidence_boost

### ✅ 2. Modified `_submit_signal()` Method
- **Location**: `agents/trend_hunter.py` lines 586-720
- **Changes**:
  - ✅ Replaced binary `if regime == "bear": return` with scaling approach
  - ✅ Gets 1h regime from shared_state.get_volatility_regime()
  - ✅ Calls _get_regime_scaling_factors(regime_1h)
  - ✅ Applies confidence_boost adjustment
  - ✅ Re-filters on adjusted confidence (not absolute position blocking)
  - ✅ Only hard-blocks bear if high-confidence override not enabled
  - ✅ Logs all scaling factors being applied
  - ✅ Has fallback for missing regime data (uses "normal" baseline)

### ✅ 3. Updated Signal Emission
- **Location**: `agents/trend_hunter.py` lines 697-720
- **Changes**:
  - ✅ Added `_regime_scaling` dict to signal payload
  - ✅ Added `_regime` field to signal
  - ✅ Enhanced logging to show position_size_mult being applied

---

## PHASE 2: MetaController Integration ⏭️ PENDING

### Location: `core/meta_controller.py`

### Task: Apply position_size_mult to Quote Hint

**Objective**: When MetaController receives signal with regime scaling, apply the position_size_mult to the quote_hint before creating order.

**Code Changes Needed**:

```python
# In MetaController._execute_decision() or where signal is processed:

# Step 1: Extract regime scaling from signal
regime_scaling = signal.get("_regime_scaling", {})
position_size_mult = regime_scaling.get("position_size_mult", 1.0)

# Step 2: Apply to quote_hint
original_quote = signal.get("quote_hint")
if original_quote and regime_scaling:
    adjusted_quote = original_quote * position_size_mult
    signal["quote_hint"] = adjusted_quote
    
    logger.info(
        "[MetaController] Applied regime scaling: quote %s → %s (mult=%.2f)",
        original_quote, adjusted_quote, position_size_mult
    )
```

**Impact**: All BUY signals now execute with regime-adjusted position sizes:
- Trending: Full size (1.0x)
- Sideways: 50% size (0.5x)
- Bear: 60% size (0.6x)
- High Vol: 80% size (0.8x)

**Priority**: 🔴 HIGH - This is the primary lever for risk management across all signals

---

## PHASE 3: TP/SL Engine Integration ⏭️ PENDING

### Location: `core/tp_sl_engine.py`

### Task 1: Apply tp_target_mult to TP Calculation

**Objective**: Scale TP distance based on regime when calculating TP for a position.

**Code Changes Needed**:

```python
# In TPSLEngine.calculate_tp_sl() or _calculate_tp_distance():

# Step 1: Extract regime scaling from signal metadata
trade_metadata = position.get("_signal_metadata", {})
regime_scaling = trade_metadata.get("_regime_scaling", {})
tp_mult = regime_scaling.get("tp_target_mult", 1.0)

# Step 2: Apply to TP calculation
base_tp_distance = volatility_profile["tp_distance"]
adjusted_tp_distance = base_tp_distance * tp_mult

# Step 3: Calculate TP price
tp_price = entry_price + adjusted_tp_distance

logger.info(
    "[TP/SL] TP distance: %.2f → %.2f (mult=%.2f)",
    base_tp_distance, adjusted_tp_distance, tp_mult
)
```

**Impact**: TP targets are scaled per regime:
- Trending: Full TP (1.0x)
- Sideways: 60% of normal TP (0.6x)
- Bear: 80% of normal TP (0.8x)
- High Vol: 105% of normal TP (1.05x wider)

**Priority**: 🟡 MEDIUM - Refines profit targets for each regime

### Task 2: Apply excursion_requirement_mult to Excursion Gate

**Objective**: Scale the minimum price movement required to keep position valid.

**Code Changes Needed**:

```python
# In TPSLEngine._passes_excursion_gate():

# Step 1: Extract regime scaling
regime_scaling = position.get("_regime_scaling", {})
excursion_mult = regime_scaling.get("excursion_requirement_mult", 1.0)

# Step 2: Scale threshold
base_excursion_threshold = atr * 0.35
adjusted_threshold = base_excursion_threshold * excursion_mult

# Step 3: Check if position meets threshold
current_excursion = abs(current_price - entry_price)
is_valid = current_excursion >= adjusted_threshold

logger.debug(
    "[Excursion] %s: threshold %.2f → %.2f (mult=%.2f), current=%.2f %s",
    symbol, base_excursion_threshold, adjusted_threshold, excursion_mult,
    current_excursion, "✓ PASS" if is_valid else "✗ FAIL"
)
```

**Impact**: Position validity gates are scaled per regime:
- Trending: Easier to trigger (0.85x)
- Sideways: Harder to trigger (1.4x)
- Bear: Harder to trigger (1.2x)
- Normal: Standard gate (1.0x)

**Priority**: 🟡 MEDIUM - Prevents "false signal" exits in choppy regimes

---

## PHASE 4: ExecutionManager Trailing Integration ⏭️ PENDING

### Location: `core/execution_manager.py`

### Task: Apply trail_mult to Trailing Stop Aggressiveness

**Objective**: Scale the trailing stop multiplier based on regime when monitoring open positions.

**Code Changes Needed**:

```python
# In ExecutionManager.check_orders() or trailing SL update logic:

# Step 1: Extract regime scaling
regime_scaling = position.get("_regime_scaling", {})
trail_mult = regime_scaling.get("trail_mult", 1.0)

# Step 2: Apply to trailing configuration
base_trail_mult = TRAILING_ATR_MULT  # e.g., 1.5
adjusted_trail_mult = base_trail_mult * trail_mult

# Step 3: Use in trailing SL calculation
trailing_sl = current_high - (atr * adjusted_trail_mult)

logger.debug(
    "[Trailing] %s: multiplier %.2f → %.2f (mult=%.2f)",
    symbol, base_trail_mult, adjusted_trail_mult, trail_mult
)
```

**Impact**: Trailing stop aggressiveness per regime:
- Trending: Aggressive trailing (1.3x) - follows price closely as it rises
- Sideways: Tight trailing (0.9x) - quick stops on false breakouts
- Bear: Very tight (0.95x) - protective positioning
- High Vol: Moderate (1.2x) - balance whipsaw vs. trend

**Priority**: 🟡 MEDIUM - Controls exit velocity in each regime

---

## PHASE 5: Configuration ⏭️ PENDING

### Location: `config.py` or environment variables

### Task: Make Scaling Factors Configurable

**Configuration Needed**:

```python
# Regime scaling enable/disable
TREND_REGIME_SCALING_ENABLED = True

# Per-regime position size multipliers
TREND_POSITION_SIZE_MULT_TRENDING = 1.00
TREND_POSITION_SIZE_MULT_HIGH_VOL = 0.80
TREND_POSITION_SIZE_MULT_SIDEWAYS = 0.50
TREND_POSITION_SIZE_MULT_BEAR = 0.60

# Per-regime TP target multipliers
TREND_TP_TARGET_MULT_TRENDING = 1.00
TREND_TP_TARGET_MULT_HIGH_VOL = 1.05
TREND_TP_TARGET_MULT_SIDEWAYS = 0.60
TREND_TP_TARGET_MULT_BEAR = 0.80

# Per-regime excursion multipliers
TREND_EXCURSION_MULT_TRENDING = 0.85
TREND_EXCURSION_MULT_HIGH_VOL = 1.00
TREND_EXCURSION_MULT_SIDEWAYS = 1.40
TREND_EXCURSION_MULT_BEAR = 1.20

# Per-regime trailing multipliers
TREND_TRAIL_MULT_TRENDING = 1.30
TREND_TRAIL_MULT_HIGH_VOL = 1.20
TREND_TRAIL_MULT_SIDEWAYS = 0.90
TREND_TRAIL_MULT_BEAR = 0.95

# Per-regime confidence adjustments
TREND_CONFIDENCE_BOOST_TRENDING = 0.05
TREND_CONFIDENCE_BOOST_HIGH_VOL = 0.00
TREND_CONFIDENCE_BOOST_SIDEWAYS = -0.05
TREND_CONFIDENCE_BOOST_BEAR = -0.08

# Override: Allow high-confidence trades in bear regime
TREND_ALLOW_BEAR_IF_HIGH_CONF = False
```

**Benefits**:
- Easy A/B testing of different multiplier sets
- Per-environment tuning (prod vs. test)
- Fast iteration on scaling factors without code changes

**Priority**: 🟢 LOW - Implement after core integration works

---

## PHASE 6: Logging & Metrics ⏭️ PENDING

### Location: `core/metrics.py` or logging setup

### Task: Track Regime-Based Performance

**Metrics Needed**:

```python
# Per regime:
- Win rate by regime (%)
- Avg profit per regime (USDT, %)
- Profit factor by regime
- Sharpe ratio by regime
- Max drawdown by regime
- Position count by regime
- Avg position size by regime

# Comparisons:
- Binary gating vs. regime scaling (A/B)
- ROI by regime (trending > sideways > bear)
- Scalp trades in sideways vs. trending
```

**Logging Format**:
```
[REGIME SCALING] ETHUSDT | regime=sideways | 
  pos_size: 100 → 50 (0.50x) | 
  tp_target: 1.5% → 0.9% (0.60x) | 
  excursion: 100bp → 140bp (1.40x) | 
  trail: 1.5 → 1.35 (0.90x) | 
  conf: 0.72 → 0.67 (-0.05)
```

**Priority**: 🟢 LOW - Implement after all integration phases

---

## Verification Checklist

### Before Production Deployment ✓

- [ ] **Phase 2**: MetaController applies position_size_mult to quote_hint
  - [ ] BUY signals execute with regime-scaled sizes
  - [ ] Sideways BUY signals are 50% smaller
  - [ ] Trending BUY signals are full size

- [ ] **Phase 3a**: TP/SL Engine scales TP targets by regime
  - [ ] Sideways trades have smaller TP targets
  - [ ] Trending trades have normal TP targets
  - [ ] All TP calculations use regime scaling

- [ ] **Phase 3b**: Excursion gate scales per regime
  - [ ] Sideways positions have 1.4x harder excursion gates
  - [ ] Trending positions have 0.85x easier excursion gates
  - [ ] False signal exits are reduced in choppy regimes

- [ ] **Phase 4**: ExecutionManager applies trailing multipliers
  - [ ] Trending positions trail aggressively (1.3x)
  - [ ] Sideways positions trail tightly (0.9x)
  - [ ] Trailing stops are regime-aware

- [ ] **Phase 5**: Configuration is externalized
  - [ ] Scaling factors can be tuned without code changes
  - [ ] Different multiplier sets can be tested (A/B)
  - [ ] Environment-specific overrides work

- [ ] **Testing**: Backtest results show improvement
  - [ ] Regime-scaling vs. binary gating: win rate comparison
  - [ ] Profit factor improves in each regime
  - [ ] Max drawdown is reduced
  - [ ] Sharpe ratio improves or remains neutral

---

## Implementation Order

**MUST DO (Critical Path)**:
1. ⏭️ Phase 2: MetaController position_size_mult
2. ⏭️ Phase 3a: TP/SL Engine tp_target_mult
3. ⏭️ Testing: Backtest regime scaling system

**SHOULD DO (High Value)**:
4. ⏭️ Phase 3b: Excursion gate multipliers
5. ⏭️ Phase 4: ExecutionManager trailing multipliers

**NICE TO HAVE (Operational)**:
6. ⏭️ Phase 5: Configuration externalization
7. ⏭️ Phase 6: Logging and metrics

---

## Rollback Plan

If regime-based scaling causes issues:

**Option 1: Disable Scaling (Soft Rollback)**
```python
# In agents/trend_hunter.py:
TREND_REGIME_SCALING_ENABLED = False
# Falls back to 1.0x multipliers everywhere (baseline behavior)
```

**Option 2: Revert Binary Gating (Hard Rollback)**
- Comment out regime scaling methods
- Restore old `if regime == "bear": return` logic
- Requires code change, but maintains original behavior

**Option 3: Adjust Multipliers (Tuning)**
- Keep scaling enabled but change multiplier values
- Example: sideways 0.5x → 0.7x (less aggressive reduction)
- Can be done via config without code changes

---

## Success Criteria

✅ **Regime-based scaling is working when**:

1. BUY signals in sideways regime execute at 50% position size
2. BUY signals in trending regime execute at full position size
3. TP targets are adjusted per regime in TP/SL calculations
4. Excursion gates account for regime difficulty
5. Trailing stops adjust aggressiveness per regime
6. Confidence scores are adjusted based on regime
7. No signals are binary blocked (all gradient-scaled)
8. Performance metrics show improvement in sideways regimes
9. Profit factor is maintained across all regimes
10. System remains stable under all market conditions

---

## Documentation

- **Architecture**: `REGIME_BASED_SCALING_ARCHITECTURE.md`
- **Implementation**: This file
- **Reference**: `REGIME_FILTER_ENFORCEMENT_ANALYSIS.md`

---

## Questions?

Refer to the main architecture document for detailed examples and rationale.
