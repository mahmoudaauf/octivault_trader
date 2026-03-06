# ML Position Scaling Implementation

## Overview
Implemented a 4-step ML position scaling feature that allows the MLForecaster agent to adjust trade sizes based on model confidence levels. This creates dynamic position sizing where higher confidence predictions lead to larger positions.

## Implementation Details

### Step 1: MLForecaster Position Scale Calculation
**File:** `agents/ml_forecaster.py` (lines 3482-3519)

Added position scaling logic after the ML model makes a prediction:

```python
# Tiered position scaling based on confidence bands
if prob >= 0.75:
    position_scale = 1.5  # 50% larger position
elif prob >= 0.65:
    position_scale = 1.2  # 20% larger position
elif prob >= 0.55:
    position_scale = 1.0  # Standard position size
elif prob >= 0.45:
    position_scale = 0.8  # 20% smaller position
else:
    position_scale = 0.6  # 40% smaller position
```

**Key Points:**
- Scales are based on buy prediction confidence
- Range: 0.6x to 1.5x (60% to 150% of base position)
- Scale is calculated only for BUY actions
- Logged with confidence level for debugging

### Step 2: SharedState Storage
**File:** `core/shared_state.py`

#### Dictionary Addition (line 563)
```python
self.ml_position_scale = {}  # Symbol -> position scale multiplier from ML model
```

#### Setter Method (lines 4374-4381)
```python
async def set_ml_position_scale(self, symbol: str, scale: float) -> None:
    """Store ML model position scale multiplier for a symbol."""
    async with self._lock_context("signals"):
        self.ml_position_scale[symbol] = (float(scale), time.time())
```

#### Getter Method (lines 4383-4397)
```python
async def get_ml_position_scale(self, symbol: str, default: float = 1.0) -> float:
    """Get ML model position scale multiplier for a symbol."""
    s = self.ml_position_scale.get(symbol)
    if not s:
        return float(default)
    scale, ts = s
    return float(scale)
```

**Key Points:**
- Thread-safe access with `_lock_context("signals")`
- Default value of 1.0 (no scaling) if not found
- Stores timestamp with scale for potential future TTL management

### Step 3: MetaController Position Size Adjustment
**File:** `core/meta_controller.py` (lines 2883-2897)

In the `should_place_buy` method, added ML scaling before exchange validation:

```python
# ML POSITION SCALING: Apply ML-derived position scale to planned_quote
ml_scale = await self.shared_state.get_ml_position_scale(symbol)
original_planned_quote = planned_quote
planned_quote = float(planned_quote or 0.0) * float(ml_scale or 1.0)

if ml_scale != 1.0:
    self.logger.info(
        "[Meta:MLScaling] %s planned_quote scaled: %.2f → %.2f (ml_scale=%.2f)",
        symbol, original_planned_quote, planned_quote, ml_scale
    )
```

**Key Points:**
- Scaling applied after all initial validation but before exchange min checks
- Original quote preserved for logging
- Logs scaling events for transparency

## Data Flow

```
MLForecaster
  ├─ Model prediction: confidence = X
  ├─ Calculate position_scale based on confidence
  └─> await shared_state.set_ml_position_scale(symbol, scale)
        │
        └─> SharedState.ml_position_scale[symbol] = (scale, timestamp)

MetaController.should_place_buy()
  ├─ Get planned_quote from signal
  ├─> ml_scale = await shared_state.get_ml_position_scale(symbol)
  ├─ planned_quote *= ml_scale
  └─ Continue with exchange validation using scaled planned_quote
```

## Configuration & Customization

### Scaling Thresholds
The confidence bands can be adjusted by modifying the thresholds in `ml_forecaster.py`:

```python
if prob >= 0.75:      # Adjust thresholds here
    position_scale = 1.5
elif prob >= 0.65:
    position_scale = 1.2
# ... etc
```

### Default Scale
The default scale can be changed via the `default` parameter in SharedState getter:

```python
ml_scale = await self.shared_state.get_ml_position_scale(symbol, default=1.0)
```

## Examples

### Example 1: High Confidence Trade
- MLForecaster confidence: 0.78 (78%)
- Position scale: 1.5x (50% larger)
- Base planned_quote: 25 USDT
- Final planned_quote: 37.5 USDT

### Example 2: Moderate Confidence Trade
- MLForecaster confidence: 0.60 (60%)
- Position scale: 1.2x (20% larger)
- Base planned_quote: 25 USDT
- Final planned_quote: 30 USDT

### Example 3: Low Confidence Trade
- MLForecaster confidence: 0.48 (48%)
- Position scale: 0.8x (20% smaller)
- Base planned_quote: 25 USDT
- Final planned_quote: 20 USDT

## Logging

All scaling operations are logged with the `[Meta:MLScaling]` tag for easy monitoring:

```
[Meta:MLScaling] BTCUSDT planned_quote scaled: 25.00 → 37.50 (ml_scale=1.50)
[MLForecaster] ML position scale stored for BTCUSDT: 1.50x (confidence=0.78)
```

## Safety Features

1. **Thread Safety:** All access to `ml_position_scale` uses async locks
2. **Default Handling:** Missing scales default to 1.0 (no scaling)
3. **Validation:** MetaController applies scaling before exchange min checks
4. **Reversibility:** Scale stored per-symbol and can be easily reset
5. **Logging:** All scaling decisions are logged for audit trail

## Future Enhancements

Possible extensions:
- TTL-based scale expiration (refresh on each new signal)
- Regime-aware scaling (adjust multipliers based on volatility regime)
- Risk-adjusted scaling (limit total position exposure)
- A/B testing framework for different scaling curves
- Integration with capital governor for position limits

## Verification

To verify implementation:

1. Check MLForecaster logs for position scale storage messages
2. Monitor MetaController logs for MLScaling messages
3. Verify planned_quotes are scaled appropriately in execution records
4. Query SharedState.ml_position_scale directly for current scales
