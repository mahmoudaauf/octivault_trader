# Regime-Based Scaling Integration Guide - Code Locations & Snippets

## Quick Reference

| Phase | Component | File | Lines | Status |
|-------|-----------|------|-------|--------|
| 1 | TrendHunter._get_regime_scaling_factors() | agents/trend_hunter.py | 503-584 | ✅ DONE |
| 1 | TrendHunter._submit_signal() | agents/trend_hunter.py | 586-720 | ✅ DONE |
| 2 | MetaController position scaling | core/meta_controller.py | TBD | ⏭️ NEXT |
| 3a | TP/SL Engine TP scaling | core/tp_sl_engine.py | TBD | ⏭️ PENDING |
| 3b | TP/SL Engine excursion scaling | core/tp_sl_engine.py | TBD | ⏭️ PENDING |
| 4 | ExecutionManager trailing scaling | core/execution_manager.py | TBD | ⏭️ PENDING |
| 5 | Configuration | config.py | TBD | ⏭️ PENDING |

---

## Phase 2: MetaController Integration (⏭️ NEXT)

### Objective
Apply `position_size_mult` from signal's `_regime_scaling` dict to the quote_hint before creating the order.

### Search for These Methods in meta_controller.py

```python
# Search for: "def _execute_decision" or "def execute_decision" or "def process_signal"
# These are where signals are converted to orders
```

### Implementation Template

Add this code where signals are processed (typically in execute_decision or similar):

```python
async def _execute_decision(self, symbol: str, signal: Dict[str, Any]) -> None:
    """
    Process signal and apply regime-based position scaling.
    """
    
    # ... existing code ...
    
    # ====== PHASE 2: Apply Regime Scaling to Position Size ======
    regime_scaling = signal.get("_regime_scaling", {})
    if regime_scaling:
        position_size_mult = regime_scaling.get("position_size_mult", 1.0)
        original_quote = signal.get("quote_hint")
        
        if original_quote and position_size_mult != 1.0:
            adjusted_quote = original_quote * position_size_mult
            signal["quote_hint"] = adjusted_quote
            
            logger.info(
                "[MetaController] Regime scaling applied for %s: "
                "quote_hint %.2f → %.2f (mult=%.2f, regime=%s)",
                symbol,
                original_quote,
                adjusted_quote,
                position_size_mult,
                signal.get("_regime", "unknown")
            )
    
    # ... continue with rest of execution ...
```

### Key Points

- ✅ Extract `_regime_scaling` dict from signal
- ✅ Get `position_size_mult` from the dict (default 1.0 if missing)
- ✅ Multiply `quote_hint` by this multiplier
- ✅ Update signal's `quote_hint` with adjusted value
- ✅ Log the adjustment for debugging
- ✅ Handle missing regime scaling gracefully (use 1.0 as default)

### Testing

```python
# Test case 1: Sideways regime → 50% position size
signal = {
    "quote_hint": 100.0,
    "_regime_scaling": {"position_size_mult": 0.5},
    "_regime": "sideways"
}
# Expected: signal["quote_hint"] becomes 50.0

# Test case 2: Trending regime → full position size
signal = {
    "quote_hint": 100.0,
    "_regime_scaling": {"position_size_mult": 1.0},
    "_regime": "trending"
}
# Expected: signal["quote_hint"] remains 100.0

# Test case 3: No regime scaling → full position size
signal = {
    "quote_hint": 100.0,
}
# Expected: signal["quote_hint"] remains 100.0 (fallback to 1.0)
```

---

## Phase 3a: TP/SL Engine - TP Target Scaling (⏭️ PENDING)

### Objective
Scale TP distance by `tp_target_mult` from regime scaling when calculating TP for a position.

### Search for These Methods in tp_sl_engine.py

```python
# Primary search: "def calculate_tp_sl" or "_calculate_tp_sl"
# Secondary search: "_build_volatility_profile" (already uses regime)
# Tertiary search: "tp_distance" or "tp_pct"
```

### Current Implementation Reference

Lines 453-473 show how regime is already used:

```python
def _build_volatility_profile(self, symbol: str, atr: float, rsi: int) -> Dict[str, float]:
    # ... existing code ...
    
    # Regime-based multipliers (ALREADY EXISTS)
    if regime_name in {"high_vol", "high"}:
        tp_pct *= 1.05
    elif regime_name == "sideways":
        tp_pct *= 0.6
    # ... etc
```

### Implementation Template

Modify the TP calculation to also use signal-provided scaling:

```python
def calculate_tp_sl(self, symbol: str, entry_price: float, signal: Dict[str, Any], 
                   position: Dict[str, Any]) -> Tuple[float, float]:
    """
    Calculate TP and SL with regime-based scaling from signal.
    """
    
    # Get volatility profile (existing code)
    vol_profile = self._build_volatility_profile(symbol, atr, rsi)
    base_tp_distance = vol_profile["tp_distance"]
    
    # ====== PHASE 3a: Apply TP Target Scaling ======
    regime_scaling = signal.get("_regime_scaling", {})
    if regime_scaling:
        tp_mult = regime_scaling.get("tp_target_mult", 1.0)
        adjusted_tp_distance = base_tp_distance * tp_mult
        
        logger.debug(
            "[TP/SL] %s: TP distance %.2f → %.2f (mult=%.2f, regime=%s)",
            symbol,
            base_tp_distance,
            adjusted_tp_distance,
            tp_mult,
            signal.get("_regime", "unknown")
        )
        
        tp_price = entry_price + adjusted_tp_distance
    else:
        tp_price = entry_price + base_tp_distance
    
    # ... rest of SL calculation ...
    
    return tp_price, sl_price
```

### Key Points

- ✅ Get `tp_target_mult` from `_regime_scaling` dict in signal
- ✅ Multiply calculated TP distance by this multiplier
- ✅ Use adjusted TP for price calculation
- ✅ Store multiplier information in position metadata for later reference
- ✅ Handle missing regime scaling (default 1.0)

### Testing

```python
# Test case 1: Sideways regime → 60% of normal TP
signal = {
    "_regime_scaling": {"tp_target_mult": 0.6},
    "_regime": "sideways"
}
base_tp = 1.5  # 1.5% move
# Expected: adjusted_tp = 0.9% (60% of 1.5%)

# Test case 2: Trending regime → full TP
signal = {
    "_regime_scaling": {"tp_target_mult": 1.0},
    "_regime": "trending"
}
base_tp = 1.5
# Expected: adjusted_tp = 1.5% (100% of 1.5%)

# Test case 3: High vol → slightly wider TP
signal = {
    "_regime_scaling": {"tp_target_mult": 1.05},
    "_regime": "high_vol"
}
base_tp = 1.5
# Expected: adjusted_tp = 1.575% (105% of 1.5%)
```

### Important Note

The TP/SL engine **already reads regime** from shared_state (see line 453-473). This phase adds an **additional layer** of scaling from the signal itself. Both will compound:

```
Final TP = Base TP × (regime_from_shared_state_mult) × (tp_target_mult_from_signal)
```

Example:
- Base TP: 1.5%
- Regime from shared_state (sideways): 0.6x → 0.9%
- TP target mult from signal (sideways): 0.6x → 0.54%
- Final: 0.54% of 1.5% entry = move to 1.54% entry price

---

## Phase 3b: TP/SL Engine - Excursion Gate Scaling (⏭️ PENDING)

### Objective
Scale minimum price movement requirement by `excursion_requirement_mult` from regime scaling.

### Search for These Methods in tp_sl_engine.py

```python
# Primary search: "_passes_excursion_gate" or "excursion"
# Secondary search: "price_movement" or "move_pct"
# Tertiary search: "atr" (excursion uses ATR multiples)
```

### Implementation Template

```python
def _passes_excursion_gate(self, symbol: str, entry_price: float, current_price: float,
                          signal: Dict[str, Any], atr: float) -> bool:
    """
    Check if position meets minimum price movement requirement (excursion gate).
    Gate is scaled by regime.
    """
    
    # Base excursion threshold (e.g., 0.35 ATR)
    base_threshold = atr * 0.35
    
    # ====== PHASE 3b: Apply Excursion Scaling ======
    regime_scaling = signal.get("_regime_scaling", {})
    if regime_scaling:
        excursion_mult = regime_scaling.get("excursion_requirement_mult", 1.0)
        adjusted_threshold = base_threshold * excursion_mult
        
        logger.debug(
            "[Excursion] %s: threshold %.2f → %.2f (mult=%.2f, regime=%s)",
            symbol,
            base_threshold,
            adjusted_threshold,
            excursion_mult,
            signal.get("_regime", "unknown")
        )
    else:
        adjusted_threshold = base_threshold
    
    # Calculate current excursion
    current_excursion = abs(current_price - entry_price)
    is_valid = current_excursion >= adjusted_threshold
    
    logger.debug(
        "[Excursion] %s: current=%.2f, threshold=%.2f %s",
        symbol,
        current_excursion,
        adjusted_threshold,
        "✓ PASS" if is_valid else "✗ FAIL"
    )
    
    return is_valid
```

### Key Points

- ✅ Get `excursion_requirement_mult` from `_regime_scaling` dict
- ✅ Multiply base threshold by this multiplier
- ✅ Higher multiplier = harder to pass (e.g., sideways 1.4x)
- ✅ Lower multiplier = easier to pass (e.g., trending 0.85x)
- ✅ Store threshold info in position metadata

### Testing

```python
# Test case 1: Sideways regime → 1.4x harder to trigger
signal = {
    "_regime_scaling": {"excursion_requirement_mult": 1.4},
    "_regime": "sideways"
}
base_threshold = 100  # bps
# Expected: adjusted = 140 bps (1.4x harder)

# Test case 2: Trending regime → 0.85x easier to trigger
signal = {
    "_regime_scaling": {"excursion_requirement_mult": 0.85},
    "_regime": "trending"
}
base_threshold = 100
# Expected: adjusted = 85 bps (0.85x easier)

# Test case 3: Normal regime → same threshold
signal = {
    "_regime_scaling": {"excursion_requirement_mult": 1.0},
    "_regime": "normal"
}
base_threshold = 100
# Expected: adjusted = 100 bps (1.0x normal)
```

### Why This Matters

Sideways markets have many false breakouts. A position that moves 100 bps might be noise in a sideways regime but a valid trend in a trending regime. By scaling the excursion gate:

- **Sideways (1.4x)**: Require 140 bps movement to confirm the signal was valid
- **Trending (0.85x)**: Only 85 bps needed to confirm (faster entry validation)

---

## Phase 4: ExecutionManager - Trailing Aggressiveness (⏭️ PENDING)

### Objective
Scale trailing stop aggressiveness by `trail_mult` from regime scaling.

### Search for These Methods in execution_manager.py

```python
# Primary search: "def check_orders" or "trailing"
# Secondary search: "TRAILING_ATR_MULT" or "trail_atr"
# Tertiary search: "highest" and "stop_loss"
```

### Implementation Template

```python
def check_orders(self, symbol: str, position: Dict[str, Any], current_price: float,
                high_price: float, atr: float) -> Optional[Dict[str, Any]]:
    """
    Monitor position and update trailing stop based on regime.
    """
    
    # Base trailing configuration
    TRAILING_ATR_MULT = 1.5  # Standard multiplier
    
    # ====== PHASE 4: Apply Trail Multiplier ======
    regime_scaling = position.get("_regime_scaling", {})
    if regime_scaling:
        trail_mult = regime_scaling.get("trail_mult", 1.0)
        adjusted_trail_mult = TRAILING_ATR_MULT * trail_mult
        
        logger.debug(
            "[Trailing] %s: multiplier %.2f → %.2f (mult=%.2f, regime=%s)",
            symbol,
            TRAILING_ATR_MULT,
            adjusted_trail_mult,
            trail_mult,
            position.get("_regime", "unknown")
        )
    else:
        adjusted_trail_mult = TRAILING_ATR_MULT
    
    # Calculate trailing stop
    trailing_sl = high_price - (atr * adjusted_trail_mult)
    
    # Check if stop should trigger
    if current_price <= trailing_sl:
        logger.info(
            "[Trailing] %s: STOP TRIGGERED (price=%.2f, sl=%.2f)",
            symbol, current_price, trailing_sl
        )
        return {"action": "exit", "reason": "trailing_stop"}
    
    # Update position's trailing SL
    position["trailing_sl"] = trailing_sl
    return None
```

### Key Points

- ✅ Get `trail_mult` from `_regime_scaling` dict in position metadata
- ✅ Multiply base TRAILING_ATR_MULT by this factor
- ✅ Lower multiplier = tighter trailing (sideways 0.9x)
- ✅ Higher multiplier = looser trailing (trending 1.3x)
- ✅ Update position metadata with current trailing stop

### Testing

```python
# Test case 1: Sideways regime → tighter trailing
position = {
    "_regime_scaling": {"trail_mult": 0.9},
    "_regime": "sideways"
}
base_trail = 1.5
atr = 100
# Expected: adjusted = 1.5 × 0.9 = 1.35 × atr = 135 bps from high
# Result: Trailing stops at +135 bps from highest point

# Test case 2: Trending regime → aggressive trailing
position = {
    "_regime_scaling": {"trail_mult": 1.3},
    "_regime": "trending"
}
base_trail = 1.5
atr = 100
# Expected: adjusted = 1.5 × 1.3 = 1.95 × atr = 195 bps from high
# Result: Trailing stops at +195 bps from highest point (follows loosely)

# Test case 3: High vol regime → moderate trailing
position = {
    "_regime_scaling": {"trail_mult": 1.2},
    "_regime": "high_vol"
}
base_trail = 1.5
atr = 100
# Expected: adjusted = 1.5 × 1.2 = 1.8 × atr = 180 bps from high
# Result: Trailing stops at +180 bps from highest point
```

### Why This Matters

Different regimes have different volatility and trend characteristics:

- **Sideways (0.9x tight)**: Protect profits quickly; false breakouts are likely
- **Trending (1.3x loose)**: Give room for price oscillation; let winners run
- **High Vol (1.2x moderate)**: Balance whipsaw protection vs. trend following
- **Bear (0.95x very tight)**: Conservative protection in risky regimes

---

## Phase 5: Configuration Externalization (⏭️ PENDING)

### Objective
Move hardcoded multipliers to config for easy tuning without code changes.

### Add to config.py or environment

```python
# ========== Regime-Based Scaling Configuration ==========

# Master enable/disable
TREND_REGIME_SCALING_ENABLED = True

# Position Size Multipliers (by regime)
TREND_POSITION_SIZE_MULT_TRENDING = 1.00      # Full size
TREND_POSITION_SIZE_MULT_HIGH_VOL = 0.80      # 80% size
TREND_POSITION_SIZE_MULT_SIDEWAYS = 0.50      # 50% size
TREND_POSITION_SIZE_MULT_BEAR = 0.60          # 60% size
TREND_POSITION_SIZE_MULT_NORMAL = 1.00        # Baseline

# TP Target Multipliers (by regime)
TREND_TP_TARGET_MULT_TRENDING = 1.00          # Full TP
TREND_TP_TARGET_MULT_HIGH_VOL = 1.05          # 105% TP
TREND_TP_TARGET_MULT_SIDEWAYS = 0.60          # 60% TP
TREND_TP_TARGET_MULT_BEAR = 0.80              # 80% TP
TREND_TP_TARGET_MULT_NORMAL = 1.00            # Baseline

# Excursion Requirement Multipliers (by regime)
TREND_EXCURSION_MULT_TRENDING = 0.85          # Easier to trigger
TREND_EXCURSION_MULT_HIGH_VOL = 1.00          # Normal
TREND_EXCURSION_MULT_SIDEWAYS = 1.40          # Harder to trigger
TREND_EXCURSION_MULT_BEAR = 1.20              # Harder
TREND_EXCURSION_MULT_NORMAL = 1.00            # Baseline

# Trailing Stop Multipliers (by regime)
TREND_TRAIL_MULT_TRENDING = 1.30              # Loose trailing
TREND_TRAIL_MULT_HIGH_VOL = 1.20              # Moderate
TREND_TRAIL_MULT_SIDEWAYS = 0.90              # Tight
TREND_TRAIL_MULT_BEAR = 0.95                  # Very tight
TREND_TRAIL_MULT_NORMAL = 1.00                # Baseline

# Confidence Adjustments (by regime)
TREND_CONFIDENCE_BOOST_TRENDING = 0.05        # +5%
TREND_CONFIDENCE_BOOST_HIGH_VOL = 0.00        # No change
TREND_CONFIDENCE_BOOST_SIDEWAYS = -0.05       # -5%
TREND_CONFIDENCE_BOOST_BEAR = -0.08           # -8%
TREND_CONFIDENCE_BOOST_NORMAL = 0.00          # Baseline

# Override flags
TREND_ALLOW_BEAR_IF_HIGH_CONF = False         # Still block bear unless very high confidence
TREND_MIN_BEAR_CONFIDENCE = 0.85              # Minimum conf to allow bear trades
```

### Usage in Code

```python
def _get_regime_scaling_factors(self, regime: str) -> Dict[str, float]:
    """Load scaling factors from config instead of hardcoding."""
    
    regime_norm = str(regime or "normal").lower()
    
    # Load from config with defaults
    return {
        "position_size_mult": float(self._cfg(
            f"TREND_POSITION_SIZE_MULT_{regime_norm.upper()}",
            1.0
        )),
        "tp_target_mult": float(self._cfg(
            f"TREND_TP_TARGET_MULT_{regime_norm.upper()}",
            1.0
        )),
        "excursion_requirement_mult": float(self._cfg(
            f"TREND_EXCURSION_MULT_{regime_norm.upper()}",
            1.0
        )),
        "trail_mult": float(self._cfg(
            f"TREND_TRAIL_MULT_{regime_norm.upper()}",
            1.0
        )),
        "confidence_boost": float(self._cfg(
            f"TREND_CONFIDENCE_BOOST_{regime_norm.upper()}",
            0.0
        )),
        "regime": regime_norm,
    }
```

---

## Summary Table

| Phase | Component | File | Primary Method | Secondary Search | Status |
|-------|-----------|------|---|---|---|
| 2 | MetaController | core/meta_controller.py | _execute_decision | process_signal, execute_decision | ⏭️ |
| 3a | TP/SL Engine | core/tp_sl_engine.py | calculate_tp_sl | _build_volatility_profile | ⏭️ |
| 3b | TP/SL Engine | core/tp_sl_engine.py | _passes_excursion_gate | excursion, price_movement | ⏭️ |
| 4 | ExecutionManager | core/execution_manager.py | check_orders | trailing, TRAILING_ATR_MULT | ⏭️ |
| 5 | Config | config.py | N/A | Environment variables | ⏭️ |

---

## How to Use This Guide

1. **Phase 2 Next**: Open `core/meta_controller.py` and search for `_execute_decision`
2. **Implement**: Copy the implementation template and adapt to existing code style
3. **Test**: Use the test cases provided
4. **Verify**: Ensure signals emit `_regime_scaling` dict and MetaController applies position_size_mult
5. **Repeat**: Move to Phase 3a, then 3b, then 4

---

## Version Control

Track changes:
```bash
# Branch for regime scaling work
git checkout -b feat/regime-based-scaling

# Commit phases incrementally
git commit -m "Phase 2: MetaController position size scaling"
git commit -m "Phase 3a: TP/SL Engine TP target scaling"
git commit -m "Phase 3b: TP/SL Engine excursion scaling"
git commit -m "Phase 4: ExecutionManager trailing scaling"
git commit -m "Phase 5: Configuration externalization"
```

---

## Debugging Tips

**Logging to add at each integration point**:

```python
# At each _regime_scaling extraction point:
if regime_scaling:
    logger.info(
        "[%s] Using regime scaling: pos=%.2f, tp=%.2f, excursion=%.2f, trail=%.2f",
        symbol,
        regime_scaling.get("position_size_mult", 1.0),
        regime_scaling.get("tp_target_mult", 1.0),
        regime_scaling.get("excursion_requirement_mult", 1.0),
        regime_scaling.get("trail_mult", 1.0),
    )
```

**Validation checklist**:
- [ ] Signal contains `_regime_scaling` dict
- [ ] Signal contains `_regime` field
- [ ] MetaController receives both fields
- [ ] Each component reads and applies appropriate multiplier
- [ ] Multipliers are logged for each trade
- [ ] No positions execute without regime awareness

