# ML Position Scaling - Code Reference

## Change Summary

| File | Lines | Change Type | Status |
|------|-------|-------------|--------|
| `agents/ml_forecaster.py` | 3482-3519 | Added position scaling logic | ✅ Complete |
| `core/shared_state.py` | 563 | Added dictionary | ✅ Complete |
| `core/shared_state.py` | 4374-4381 | Added setter method | ✅ Complete |
| `core/shared_state.py` | 4383-4397 | Added getter method | ✅ Complete |
| `core/meta_controller.py` | 2883-2897 | Added scaling application | ✅ Complete |

---

## File 1: agents/ml_forecaster.py

### Location: Lines 3482-3519 (inserted before `await self._collect_signal()`)

**Before:**
```python
        if (sentiment < -0.5) or (regime in {"high_vol", "bear"}) or (cot_num < 0):
            action = "hold"

        self.logger.info(f"[{self.name}] Final decision for {cur_sym} => Action: {action.upper()}, Confidence: {confidence:.2f}")

        # Store a brief CoT/debug explanation for UI/debuggers
        try:
            await self.shared_state.set_cot_explanation(
                cur_sym,
                text=f"Pred={action} conf={confidence:.2f} on features shape={X.shape}",
                source=self.name,
            )
        except Exception:
            pass

        # --------- Emission (signal-only) ---------
        await self._collect_signal(
```

**After:**
```python
        if (sentiment < -0.5) or (regime in {"high_vol", "bear"}) or (cot_num < 0):
            action = "hold"

        self.logger.info(f"[{self.name}] Final decision for {cur_sym} => Action: {action.upper()}, Confidence: {confidence:.2f}")

        # Store a brief CoT/debug explanation for UI/debuggers
        try:
            await self.shared_state.set_cot_explanation(
                cur_sym,
                text=f"Pred={action} conf={confidence:.2f} on features shape={X.shape}",
                source=self.name,
            )
        except Exception:
            pass

        # ═══════════════════════════════════════════════════════════════════════
        # ML POSITION SCALING: Calculate position scale based on buy probability
        # ═══════════════════════════════════════════════════════════════════════
        position_scale = 1.0  # Default: no scaling
        
        if action.upper() == "BUY":
            # Extract buy probability from the model output
            # The confidence here represents our prediction strength
            prob = float(confidence)
            
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
            
            # Store ML position scale in SharedState for downstream use by MetaController
            try:
                if hasattr(self.shared_state, "set_ml_position_scale"):
                    await self.shared_state.set_ml_position_scale(cur_sym, position_scale)
                    self.logger.info(
                        f"[{self.name}] ML position scale stored for {cur_sym}: {position_scale:.2f}x "
                        f"(confidence={prob:.2f})"
                    )
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to store ML position scale: {e}")

        # --------- Emission (signal-only) ---------
        await self._collect_signal(
```

---

## File 2: core/shared_state.py

### Change 2a: Dictionary Initialization (Line 563)

**Location:** In `SharedState.__init__()`, agent state section

**Added:**
```python
        self.ml_position_scale = {}  # Symbol -> position scale multiplier from ML model
```

**Context:**
```python
        # Agent state
        self.volatility_regimes = {}
        self.sentiment_scores = {}
        self.agent_scores = {}
        self.cot_explanations = {}
        self.ml_position_scale = {}  # ← ADDED
        # Back-compat aliases for components expecting singular names
        self.volatility_state = self.volatility_regimes
        self.sentiment_score = self.sentiment_scores
```

---

### Change 2b: Setter Method (Lines 4374-4381)

**Location:** After `get_sentiment()` method

**Added:**
```python
    # -------- ML Position Scaling --------
    async def set_ml_position_scale(self, symbol: str, scale: float) -> None:
        """
        Store ML model position scale multiplier for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            scale: Position scale multiplier (1.0 = no change, 1.5 = 50% larger, 0.8 = 20% smaller)
        """
        async with self._lock_context("signals"):
            self.ml_position_scale[symbol] = (float(scale), time.time())
```

**Context:**
```python
    async def get_sentiment(self, symbol: str, max_age_seconds: int = 1800) -> Optional[float]:
        s = self.sentiment_scores.get(symbol)
        if not s: return None
        score, ts = s
        return score if time.time() - ts <= max_age_seconds else None

    # -------- ML Position Scaling --------  ← ADDED
    async def set_ml_position_scale(self, symbol: str, scale: float) -> None:
        """
        Store ML model position scale multiplier for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            scale: Position scale multiplier (1.0 = no change, 1.5 = 50% larger, 0.8 = 20% smaller)
        """
        async with self._lock_context("signals"):
            self.ml_position_scale[symbol] = (float(scale), time.time())
```

---

### Change 2c: Getter Method (Lines 4383-4397)

**Location:** Immediately after setter method

**Added:**
```python
    async def get_ml_position_scale(self, symbol: str, default: float = 1.0) -> float:
        """
        Get ML model position scale multiplier for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            default: Default scale if not found (default 1.0 = no scaling)
            
        Returns:
            Position scale multiplier as float
        """
        s = self.ml_position_scale.get(symbol)
        if not s:
            return float(default)
        scale, ts = s
        # Scale is valid; return it (no expiry check as scales are meant to persist per-signal)
        return float(scale)
```

**Full Context:**
```python
    # -------- ML Position Scaling --------
    async def set_ml_position_scale(self, symbol: str, scale: float) -> None:
        """Store ML model position scale multiplier for a symbol."""
        async with self._lock_context("signals"):
            self.ml_position_scale[symbol] = (float(scale), time.time())

    async def get_ml_position_scale(self, symbol: str, default: float = 1.0) -> float:
        """Get ML model position scale multiplier for a symbol."""
        s = self.ml_position_scale.get(symbol)
        if not s:
            return float(default)
        scale, ts = s
        return float(scale)

    async def push_signal(self, symbol: str, signal_data: Dict[str, Any]) -> None:
        """P9: Legacy compatibility shim for push_signal."""
        await self.add_strategy_signal(symbol, signal_data)
```

---

## File 3: core/meta_controller.py

### Location: Lines 2883-2897 (in `should_place_buy()` method)

**Position:** After ATR validation, before exchange minimum validation

**Before:**
```python
            atr_pct = float(signal.get("atr_pct", signal.get("_atr_pct", 0.0)) or 0.0)
            min_atr_pct = float(getattr(self.config, "MIN_ATR_PCT_FOR_ENTRY", 0.0015) or 0.0015)  # 0.15%
            if atr_pct > 0 and atr_pct < min_atr_pct:
                self.logger.info("[WHY_NO_TRADE] reason=ATR_TOO_LOW symbol=%s details=atr_%.4f_min_%.4f", symbol, atr_pct, min_atr_pct)
                await self._record_why_no_trade(symbol, "ATR_TOO_LOW", f"atr_{atr_pct:.4f}<min_{min_atr_pct:.4f}", signal=signal)
                return False

            # ═══════════════════════════════════════════════════════════════════════
            # GLOBAL ECONOMIC BUY GATE (SOP): economically sellable + profit lock
            # ═══════════════════════════════════════════════════════════════════════
            if planned_quote < exchange_min_with_buffer:
```

**After:**
```python
            atr_pct = float(signal.get("atr_pct", signal.get("_atr_pct", 0.0)) or 0.0)
            min_atr_pct = float(getattr(self.config, "MIN_ATR_PCT_FOR_ENTRY", 0.0015) or 0.0015)  # 0.15%
            if atr_pct > 0 and atr_pct < min_atr_pct:
                self.logger.info("[WHY_NO_TRADE] reason=ATR_TOO_LOW symbol=%s details=atr_%.4f_min_%.4f", symbol, atr_pct, min_atr_pct)
                await self._record_why_no_trade(symbol, "ATR_TOO_LOW", f"atr_{atr_pct:.4f}<min_{min_atr_pct:.4f}", signal=signal)
                return False

            # ═══════════════════════════════════════════════════════════════════════
            # ML POSITION SCALING: Apply ML-derived position scale to planned_quote
            # ═══════════════════════════════════════════════════════════════════════
            ml_scale = await self.shared_state.get_ml_position_scale(symbol)
            original_planned_quote = planned_quote
            planned_quote = float(planned_quote or 0.0) * float(ml_scale or 1.0)
            
            if ml_scale != 1.0:
                self.logger.info(
                    "[Meta:MLScaling] %s planned_quote scaled: %.2f → %.2f (ml_scale=%.2f)",
                    symbol, original_planned_quote, planned_quote, ml_scale
                )

            # ═══════════════════════════════════════════════════════════════════════
            # GLOBAL ECONOMIC BUY GATE (SOP): economically sellable + profit lock
            # ═══════════════════════════════════════════════════════════════════════
            if planned_quote < exchange_min_with_buffer:
```

---

## Integration Flow

```
Step 1: MLForecaster
└─> Line 3482-3519 in ml_forecaster.py
    ├─ Calculate position_scale (1.5, 1.2, 1.0, 0.8, or 0.6)
    └─ await shared_state.set_ml_position_scale(symbol, scale)

         ↓

Step 2: SharedState Storage
└─> Lines 563, 4374-4397 in shared_state.py
    ├─ Dictionary: self.ml_position_scale[symbol] = (scale, ts)
    ├─ Setter: set_ml_position_scale()
    └─ Getter: get_ml_position_scale(symbol, default=1.0)

         ↓

Step 3: MetaController Retrieval & Application
└─> Lines 2883-2897 in meta_controller.py
    ├─ ml_scale = await shared_state.get_ml_position_scale(symbol)
    ├─ planned_quote *= ml_scale
    └─ Log scaling operation

         ↓

Step 4: Trade Execution
└─> Position size = planned_quote / current_price
    (Now using scaled planned_quote)
```

---

## Quick Edit Guide

### To Modify Scaling Thresholds

Edit `ml_forecaster.py` lines 3495-3503:

```python
if prob >= 0.75:        # ← Increase/decrease this threshold
    position_scale = 1.5    # ← Increase/decrease this multiplier
elif prob >= 0.65:
    position_scale = 1.2
elif prob >= 0.55:
    position_scale = 1.0
elif prob >= 0.45:
    position_scale = 0.8
else:
    position_scale = 0.6
```

### To Disable Scaling

Comment out this line in `meta_controller.py` line 2886:

```python
# planned_quote = float(planned_quote or 0.0) * float(ml_scale or 1.0)
```

Or set all MLForecaster scales to 1.0:

```python
position_scale = 1.0  # Always 1.0 (no scaling)
```

### To Change Default Scale

In `meta_controller.py` line 2884:

```python
ml_scale = await self.shared_state.get_ml_position_scale(symbol, default=0.8)  # Changed from 1.0
```

---

## Validation Commands

### Check for syntax errors:
```bash
python -m py_compile agents/ml_forecaster.py
python -m py_compile core/shared_state.py
python -m py_compile core/meta_controller.py
```

### Check imports work:
```python
from agents.ml_forecaster import MLForecaster
from core.shared_state import SharedState
from core.meta_controller import MetaController
```

### Test scales in Python:
```python
import asyncio
from core.shared_state import SharedState

async def test_scales():
    ss = SharedState()
    
    # Store a scale
    await ss.set_ml_position_scale("BTCUSDT", 1.5)
    
    # Retrieve it
    scale = await ss.get_ml_position_scale("BTCUSDT")
    print(f"Scale: {scale}")  # Should print: Scale: 1.5
    
    # Test default
    scale = await ss.get_ml_position_scale("ETHUSDT")  # Not stored
    print(f"Default: {scale}")  # Should print: Default: 1.0

asyncio.run(test_scales())
```

---

## Documentation References

- `ML_POSITION_SCALING_IMPLEMENTATION.md` - Detailed implementation guide
- `ML_POSITION_SCALING_QUICK_REF.md` - Quick reference
- `ML_POSITION_SCALING_COMPLETION_REPORT.md` - Full completion report

---

**Last Updated:** 2026-03-04
**Status:** ✅ Complete and Verified
