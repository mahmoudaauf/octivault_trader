# 🔄 Lightweight Signal Outcome Tracking

## Overview
This implementation adds minimal signal outcome tracking across three components without touching execution logic. It tracks how well signals predict price movement and computes **realized edge vs cost** — the professional way to tune ML models.

## Quick Start

### What Gets Measured
For each BUY signal:
- **Price at signal time** (baseline)
- **Price at +5m, +15m, +30m** (tracking intervals)
- **Realized return** at each interval
- **Edge vs roundtrip cost** at each interval

### Example Log Output
```
[SIGNAL_OUTCOME:5m]  BTCUSDT ret=0.45% cost=0.12% edge=0.33% conf=0.85 ✅ OPTIMAL agent=MLForecaster
[SIGNAL_OUTCOME:15m] BTCUSDT ret=0.62% cost=0.12% edge=0.50% conf=0.85 ⚠️ TOO_CONSERVATIVE agent=MLForecaster
[SIGNAL_OUTCOME:30m] BTCUSDT ret=0.75% cost=0.12% edge=0.63% conf=0.85 ⚠️ TOO_CONSERVATIVE agent=MLForecaster
[SIGNAL_TUNING]     BTCUSDT avg_edge=0.49% → INCREASE_CONFIDENCE_FLOOR or RELAX_ENTRY_FILTERS
```

### Tuning Interpretation
- **edge > 0.4%** (TOO_CONSERVATIVE) → Model leaves upside on table, increase entry rate
- **edge 0.2%-0.4%** (OPTIMAL) → Well-tuned, maintain current parameters  
- **edge < 0.2%** (INSUFFICIENT) → Model doesn't justify costs, retrain or reduce position size

**➜ See [SIGNAL_EDGE_TUNING_GUIDE.md](./SIGNAL_EDGE_TUNING_GUIDE.md) for professional tuning methodology.**

## Architecture

### Step 1: MLForecaster Signal Registration
**File:** `agents/ml_forecaster.py`
**Location:** Right after `_collect_signal()` call in the `run()` method (line ~2536)

When MLForecaster emits a BUY signal, it now registers the outcome for tracking:
```python
if action == "buy":
    try:
        now = time.time()
        price_now = self.shared_state.get_price(cur_sym)
        if hasattr(self.shared_state, "register_signal_outcome"):
            self.shared_state.register_signal_outcome({
                "symbol": cur_sym,
                "timestamp": now,
                "price_at_signal": price_now,
                "confidence": confidence,
                "agent": "MLForecaster"
            })
    except Exception:
        pass
```

**Why:**
- Non-blocking: wrapped in try/except
- Minimal overhead: just appends to a list
- Extensible: can be added to other agents

### Step 2: SharedState Tracking
**File:** `core/shared_state.py`
**Additions:**
- `self._signal_outcomes = []` in `__init__()` (line ~523)
- `register_signal_outcome(record)` method (line ~3367)

```python
def register_signal_outcome(self, record: Dict[str, Any]) -> None:
    """Register a signal outcome for tracking price movement after emission."""
    try:
        self._signal_outcomes.append(record)
    except Exception:
        pass
```

**Why:**
- Single source of truth for all signal outcomes
- Easy to query/analyze later
- Thread-safe append operation

### Step 3: MetaController Periodic Evaluation
**File:** `core/meta_controller.py`
**Additions:**
- Call `self._evaluate_signal_outcomes()` in `evaluate_and_act()` after decisions built (line ~3658)
- Method `_evaluate_signal_outcomes()` implemented (line ~12648)

The evaluator runs every loop tick and computes **realized edge vs cost**:

1. **5m check:** After 300s, calculates return and edge
2. **15m check:** After 900s, logs the 15m return and edge
3. **30m check:** After 1800s, logs the 30m return, edge, and tuning recommendation
4. **Cleanup:** Removes old record after 30m evaluation

**Edge Calculation:**
```python
roundtrip_cost_pct = (maker_bps + taker_bps) / 10,000  # ~0.12% default
realized_edge = price_movement_pct - roundtrip_cost_pct
```

**Tuning Assessment:**
- `realized_edge > 0.4%` → TOO_CONSERVATIVE (model too cautious)
- `0.2% <= realized_edge <= 0.4%` → OPTIMAL (model well-tuned)
- `realized_edge < 0.2%` → INSUFFICIENT (edge doesn't justify costs)

## Benefits

✅ **Non-invasive:** No changes to execution, trading, or fill logic
✅ **Observable:** Real-time signal performance metrics
✅ **Extensible:** Easy to add other agents (TrendHunter, ArbitrageHunter, etc.)
✅ **Cleanup:** Old records auto-removed after 30m evaluation
✅ **Lightweight:** O(n) per tick where n = active signals (typically <100)

## Future Enhancements

1. **Agent-level aggregation:** Track signal accuracy by agent
2. **Confidence correlation:** Compare signal confidence vs actual edge/returns
3. **Regime adaptation:** Evaluate outcomes per volatility regime
4. **Alert integration:** Trigger alerts if agent performance drops below threshold
5. **Dashboard integration:** Display signal outcome stats and tuning recommendations in UI
6. **Profit-locked reentry:** Use measured edge to calibrate position sizing

## Professional Tuning Framework

This implementation provides the foundation for professional ML model tuning:

| Metric | Threshold | Action |
|--------|-----------|--------|
| `avg_edge` | > 0.40% | Increase entry rate (too conservative) |
| `avg_edge` | 0.20%-0.40% | Hold current parameters (optimal) |
| `avg_edge` | < 0.20% | Retrain model (insufficient edge) |

**Detailed tuning methodology:** See [SIGNAL_EDGE_TUNING_GUIDE.md](./SIGNAL_EDGE_TUNING_GUIDE.md)

## Testing

To verify the implementation:
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 -c "
from core.shared_state import SharedState
from core.meta_controller import MetaController
ss = SharedState()
ss.register_signal_outcome({'symbol': 'TEST', 'timestamp': 0, 'price_at_signal': 100})
print('✅ Signal outcome tracking working')
"
```
