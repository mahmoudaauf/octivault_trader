# 📊 Signal Edge vs Cost Tuning Guide

## Overview

This system measures **realized edge** — the actual profit potential remaining after costs. It's how professionals tune ML models.

### Formula
```
realized_edge = price_movement_pct - roundtrip_trading_cost_pct
```

Where:
- **price_movement_pct** = (price_at_+Xm - price_at_signal) / price_at_signal
- **roundtrip_trading_cost_pct** = (maker_bps + taker_bps) / 10,000

Default costs (Binance spot):
- Maker: 0.02% (2 bps)
- Taker: 0.10% (10 bps)
- **Roundtrip**: 0.12% (12 bps)

## Tuning Thresholds

### ✅ Optimal (0.2% - 0.4%)
Signal provides meaningful edge above costs. Model is well-calibrated.

**Action:** Monitor closely; maintain current strategy.

### ⚠️ Too Conservative (> 0.4%)
Model leaves >0.4% upside on table repeatedly. You're being too cautious.

**Tuning Options:**
1. **Increase confidence floor** (e.g., 0.70 → 0.65)
   - Captures more trades with decent conviction
   - Risk: May include lower-quality signals
   
2. **Relax entry filters** (sentiment, volatility regime)
   - Sentiment gate: Reduce threshold from -0.5 to -0.3
   - Volatility gate: Remove or raise "high" regime block
   
3. **Retrain with relaxed parameters**
   - Lower model threshold for action selection

**Example Log:**
```
[SIGNAL_OUTCOME:30m] BTCUSDT ret=0.52% cost=0.12% edge=0.40% conf=0.82 ⚠️ TOO_CONSERVATIVE
[SIGNAL_TUNING] BTCUSDT avg_edge=0.41% → INCREASE_CONFIDENCE_FLOOR or RELAX_ENTRY_FILTERS
```

### ❌ Insufficient (< 0.2%)
Model doesn't justify trading costs. Either the model is weak or entry is poor.

**Tuning Options:**
1. **Decrease confidence floor** (but with caution!)
   - Lower threshold captures poor signals
   - Risk: Increased whipsaws
   
2. **Retrain model** with better features
   - More lookback windows
   - Additional technical indicators
   - Market regime context
   
3. **Improve entry timing**
   - Current entry points are suboptimal
   - Consider broader market context
   
4. **Reduce position size** or skip trading
   - If edge < cost, don't trade
   - Preserve capital for better opportunities

**Example Log:**
```
[SIGNAL_OUTCOME:30m] ETHUSDT ret=0.15% cost=0.12% edge=0.03% conf=0.75 ❌ INSUFFICIENT
[SIGNAL_TUNING] ETHUSDT avg_edge=0.05% → DECREASE_CONFIDENCE_FLOOR or RETRAIN_MODEL
```

## Real-World Example

### Scenario 1: MLForecaster Too Conservative

**Observed Metrics:**
```
[SIGNAL_OUTCOME:5m]  BTCUSDT ret=0.35% cost=0.12% edge=0.23% ✅ OPTIMAL
[SIGNAL_OUTCOME:15m] BTCUSDT ret=0.45% cost=0.12% edge=0.33% ✅ OPTIMAL
[SIGNAL_OUTCOME:30m] BTCUSDT ret=0.55% cost=0.12% edge=0.43% ⚠️ TOO_CONSERVATIVE
[SIGNAL_TUNING]     BTCUSDT avg_edge=0.33% → INCREASE_CONFIDENCE_FLOOR or RELAX_ENTRY_FILTERS
```

**Action:**
1. Reduce confidence floor from 0.70 to 0.65
2. Monitor for 100 signals
3. If avg_edge still > 0.35%, reduce further to 0.60

### Scenario 2: MLForecaster Insufficient

**Observed Metrics:**
```
[SIGNAL_OUTCOME:5m]  ETHUSDT ret=0.10% cost=0.12% edge=-0.02% ❌ INSUFFICIENT
[SIGNAL_OUTCOME:15m] ETHUSDT ret=0.18% cost=0.12% edge=0.06% ❌ INSUFFICIENT
[SIGNAL_OUTCOME:30m] ETHUSDT ret=0.20% cost=0.12% edge=0.08% ❌ INSUFFICIENT
[SIGNAL_TUNING]     ETHUSDT avg_edge=0.04% → DECREASE_CONFIDENCE_FLOOR or RETRAIN_MODEL
```

**Action:**
1. **Do NOT trade ETHUSDT with this model** — edge < cost
2. Retrain with:
   - More lookback (e.g., 30 candles → 50)
   - Additional features (RSI, Bollinger Bands)
   - Cross-validation on recent data

## Implementation Details

### Where It Runs
**File:** `core/meta_controller.py`
**Method:** `_evaluate_signal_outcomes()`
**Frequency:** Every loop cycle (~1s)

### What Gets Tracked

For each BUY signal:
```python
{
    "symbol": "BTCUSDT",
    "timestamp": 1708519200.5,
    "price_at_signal": 47250.50,
    "confidence": 0.85,
    "agent": "MLForecaster",
    
    # At 5m:
    "ret_5m": 0.0045,
    "edge_vs_cost_5m": 0.0033,
    "evaluated_5m": True,
    
    # At 15m:
    "ret_15m": 0.0062,
    "edge_vs_cost_15m": 0.0050,
    "evaluated_15m": True,
    
    # At 30m:
    "ret_30m": 0.0075,
    "edge_vs_cost_30m": 0.0063,
    "evaluated_30m": True,
    
    # Summary:
    "avg_edge_vs_cost": 0.0049
}
```

### Log Format

**5-minute mark:**
```
[SIGNAL_OUTCOME:5m] SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅/⚠️/❌ agent=AGENT
```

**15-minute mark:**
```
[SIGNAL_OUTCOME:15m] SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅/⚠️/❌ agent=AGENT
```

**30-minute mark with tuning recommendation:**
```
[SIGNAL_OUTCOME:30m] SYMBOL ret=X.XX% cost=0.12% edge=X.XX% conf=0.XX ✅/⚠️/❌ agent=AGENT
[SIGNAL_TUNING] SYMBOL avg_edge=X.XX% → RECOMMENDATION
```

## Aggregating Results

To track performance over time:

```python
# Count signals by assessment
too_conservative = len([s for s in signals if s['avg_edge_vs_cost'] > 0.004])
insufficient = len([s for s in signals if s['avg_edge_vs_cost'] < 0.002])
optimal = len([s for s in signals if 0.002 <= s['avg_edge_vs_cost'] <= 0.004])

# Average edge by agent
agent_edges = defaultdict(list)
for s in signals:
    agent_edges[s['agent']].append(s['avg_edge_vs_cost'])

for agent, edges in agent_edges.items():
    print(f"{agent}: avg={np.mean(edges):.4%}, std={np.std(edges):.4%}")
```

## Best Practices

1. **Collect 100+ signals** before tuning (variance is high on small samples)
2. **Tune one parameter at a time** (confidence floor, filter threshold, etc.)
3. **Re-evaluate after each change** — give it 30+ new signals
4. **Watch for regime dependence** — some models work better in certain volatility regimes
5. **Log anomalies** — signals with extreme edge (>1%) or negative edge may be data quality issues

## Warnings

⚠️ **Overfitting Risk**
- Tuning to maximize edge can lead to overfitting recent market conditions
- Always validate on out-of-sample data

⚠️ **Fee Structure Changes**
- If exchange fees change, update `roundtrip_cost_pct` calculation
- Current hardcoded: 0.12% (maker 0.02% + taker 0.10%)

⚠️ **Slippage Not Included**
- This only accounts for exchange fees, not market impact
- Real edge should also subtract estimated slippage (~0.01% for small orders)

## Next Steps

1. **Enable signal outcome tracking** (already done ✅)
2. **Collect baseline metrics** (run for 1-2 weeks)
3. **Identify pattern** (too conservative vs insufficient)
4. **Apply one tuning change**
5. **Repeat from step 2**

**Expected timeline:** 3-4 iterations to find optimal tuning per agent.
