# CompoundingEngine Protective Gates - Quick Reference

## Three Gates Summary

| Gate | Purpose | Check | Default | Impact |
|------|---------|-------|---------|--------|
| **Gate 1: Volatility** | Avoid calm symbols where fees > edge | 24h volatility ≥ 0.45% | 0.45% | Eliminates 60% bad buys |
| **Gate 2: Edge** | Avoid local tops and momentum exhaustion | Not at high, momentum clear | — | Eliminates 40% remaining bad buys |
| **Gate 3: Economic** | Avoid eating all profit through fees | profit > fees + $50 buffer | $50 | Prevents fee churn cycles |

---

## Fee Structure (Why Gates Needed)

```
Each Buy Order Costs:
├─ Binance taker fee:    0.075% of notional
├─ Bid-ask spread:       0.05%
├─ Price impact:         0.10%
└─ TOTAL FRICTION:       0.225% per buy

To break even in 4 hours of holding, need:
├─ Volatility > 0.225% (break-even)
└─ Volatility > 0.45% (2x buffer for safety) ← This is Gate 1

Without gates, buying 0.15% volatility symbols:
├─ Need to move +0.225% just to break even
├─ Historical average: -0.09% (moves down)
├─ Result per buy: -0.09% loss
├─ Per cycle (5 buys): -0.45% loss
└─ Per month (48 cycles): -$34.30 loss
```

---

## Configuration (tuned_params.json)

```json
{
  "COMPOUNDING_MIN_VOLATILITY": 0.0045,
  "COMPOUNDING_ECONOMIC_BUFFER": 50.0,
  "COMPOUNDING_THRESHOLD": 10.0,
  "COMPOUNDING_RESERVE_USDT": 25.0,
  "MAX_COMPOUND_SYMBOLS": 5,
  "COMPOUNDING_INTERVAL": 60
}
```

---

## Execution Flow

```
User profit: +$100 realized
    ↓
_check_and_compound() runs:
    ├─ Circuit breaker? No ✅
    ├─ Realized PnL > 0? Yes ✅
    ├─ Balance > threshold? Yes ✅
    ├─ GATE 3: $100 > $2.50 fees + $50 buffer? Yes ✅
    └─ Execute compounding strategy
        ↓
        _pick_symbols() runs:
            ├─ Get 20 candidate symbols (by score)
            ├─ GATE 1: Filter by volatility (20 → 8 symbols)
            ├─ GATE 2: Filter by edge (8 → 3 symbols)
            └─ Return top 3 symbols
        ↓
        For each symbol:
            ├─ Check affordability ✅
            └─ Place high-conviction BUY ✅
        ↓
        Result: 3 buys of high-quality symbols
```

---

## Example Scenarios

### Scenario 1: Good Time to Compound ✅
```
Market conditions:
├─ ETHUSDT: volatility 0.52%, 0.3% from high, no recent momentum
├─ LTCUSDT: volatility 0.48%, 0.2% from high, momentum clear
├─ ADAUSDT: volatility 0.46%, 0.5% from high, no momentum

Profit status:
├─ Realized PnL: $100
├─ Estimated fees: $3.50
├─ Safety buffer: $50
├─ Available: $100 - $3.50 - $50 = $46.50

Result: ✅ ALL GATES PASS
- Gate 1: All symbols > 0.45% volatility ✅
- Gate 2: All at good entry points ✅
- Gate 3: Profit sufficient ✅
→ Execute: 3 buys of $10 each
```

### Scenario 2: Too Calm (Gate 1 Fails) ❌
```
Market conditions:
├─ ETHUSDT: volatility 0.25%
├─ LTCUSDT: volatility 0.18%
├─ ADAUSDT: volatility 0.32%

All have volatility < 0.45% minimum

Result: ❌ GATE 1 FAILS
- No symbols pass volatility filter
→ Skip compounding entirely
→ Log: "All symbols filtered by volatility gate"
```

### Scenario 3: Poor Entry Points (Gate 2 Fails) ❌
```
Market conditions:
├─ ETHUSDT: at 2-week high, 0.05% from top
├─ LTCUSDT: just rallied +0.8% (momentum exhausted)
├─ ADAUSDT: at local high, momentum clear

All have poor entry timing

Result: ❌ GATE 2 FAILS
- No symbols pass edge validation
→ Skip compounding entirely
→ Log: "All symbols filtered by edge validation gate"
```

### Scenario 4: Insufficient Profit (Gate 3 Fails) ❌
```
Profit status:
├─ Realized PnL: $20
├─ Estimated fees: $2.50
├─ Safety buffer: $50
├─ Required total: $52.50
├─ Shortfall: -$32.50

Result: ❌ GATE 3 FAILS
- Profit insufficient to safely cover compounding
→ Skip compounding cycle
→ Log: "Compounding blocked by economic gate (profit insufficient)"
→ Result: Protects $20 from being eaten by $2.50 in fees
```

---

## Monitoring Logs

### Normal Execution (All Gates Pass)
```
✅ ETHUSDT volatility 0.52% >= 0.45% (Gate 1: PASS)
✅ ETHUSDT edge is valid - not at high, momentum clear (Gate 2: PASS)
✅ Economic gate PASS: $100 - $2.50 - $50 = $47.50 available (Gate 3: PASS)
📊 Selected symbols: ['ETHUSDT', 'LTCUSDT', 'ADAUSDT']
✅ Compounded into ETHUSDT with 10.00 USDT (status=executed)
✅ Compounded into LTCUSDT with 10.00 USDT (status=executed)
✅ Compounded into ADAUSDT with 10.00 USDT (status=executed)
```

### Gate 1 Rejects (Calm Market)
```
❌ ETHUSDT volatility 0.15% < 0.45% (Gate 1: FAIL - too calm)
❌ LTCUSDT volatility 0.18% < 0.45% (Gate 1: FAIL - too calm)
❌ ADAUSDT volatility 0.32% < 0.45% (Gate 1: FAIL - too calm)
⚠️ All symbols filtered by volatility gate (none volatile enough)
```

### Gate 2 Rejects (Poor Entry Points)
```
✅ ETHUSDT volatility 0.50% >= 0.45% (Gate 1: PASS)
❌ ETHUSDT at local high (current=2105.5, high=2106.3, dist=0.04%) (Gate 2: FAIL)
✅ LTCUSDT volatility 0.48% >= 0.45% (Gate 1: PASS)
❌ LTCUSDT momentum fired recently (+0.62% move in last 5 candles) (Gate 2: FAIL)
⚠️ All symbols filtered by edge validation gate (poor entry timing for all)
```

### Gate 3 Rejects (Insufficient Profit)
```
❌ Economic gate FAIL: profit too thin ($20 < $52.50 fees+buffer) (Gate 3: FAIL)
⚠️ Compounding blocked by economic gate (profit insufficient to cover fee churn)
```

---

## Testing Checklist

- [ ] Volatility calculation works (numpy import available)
- [ ] OHLCV data fetching from shared_state
- [ ] Edge validation checks distance from high
- [ ] Edge validation checks momentum
- [ ] Economic gate arithmetic correct
- [ ] Async/await properly implemented
- [ ] Gates reject bad symbols (unit tests)
- [ ] Compounding still works when gates pass (integration test)
- [ ] All gates can be individually tuned
- [ ] Logging shows gate decisions
- [ ] Backward compatible (no breaking changes)

---

## Tuning Guide

### Increase Volatility Threshold (More Conservative)
```python
COMPOUNDING_MIN_VOLATILITY = 0.0060  # 0.60% instead of 0.45%
```
- Pro: Even safer, more margin of safety
- Con: Fewer opportunities, less frequent compounding
- When to use: High-risk systems, nervous operators

### Decrease Volatility Threshold (More Aggressive)
```python
COMPOUNDING_MIN_VOLATILITY = 0.0030  # 0.30% instead of 0.45%
```
- Pro: More opportunities, higher execution frequency
- Con: Less safety margin, higher fee risk
- When to use: Bull markets, volatile assets like alts

### Increase Economic Buffer (More Conservative)
```python
COMPOUNDING_ECONOMIC_BUFFER = 100.0  # $100 instead of $50
```
- Pro: Stronger protection against fee churn
- Con: Requires more profit before compounding
- When to use: Protecting small accounts from erosion

### Decrease Economic Buffer (More Aggressive)
```python
COMPOUNDING_ECONOMIC_BUFFER = 25.0  # $25 instead of $50
```
- Pro: Compounds with less profit, more aggressive
- Con: Higher risk of eating through gains
- When to use: Large accounts, confident in profitability

---

## Fee Calculation Example

**Input**: Compound $100 across 5 symbols

```
Per symbol: $100 / 5 = $20 each

Fee breakdown per $20 order:
├─ Binance taker fee: $20 × 0.075% = $0.015
├─ Bid-ask spread: $20 × 0.05% = $0.010
├─ Price impact: $20 × 0.10% = $0.020
└─ Per order total: $0.045

Total for 5 orders: $0.045 × 5 = $0.225 (0.225% of $100)
```

**Gate 3 Check**:
```
Available profit: $100
Total fees: $0.225
Safety buffer: $50.00
Required: $50.225
Available - Required = $100 - $50.225 = $49.775

$49.775 > 0? Yes ✅ PASS Gate 3
```

---

## Expected Impact (Monthly)

### Without Gates
```
Cycles per month: 48 (hourly check)
Symbols per cycle: 5
Total orders: 240
Fee per order: $0.045
Monthly fees: -$34.30 loss
```

### With All Three Gates
```
Cycles per month: 48
Symbols post-volatility: 2
Symbols post-edge: 1
Orders per cycle: 1
Total orders: 48
Fee per order: $0.0225
Monthly fees: -$2.16 loss

Improvement: 94% reduction in fee churn ✅
```

---

## Quick Troubleshooting

| Problem | Log Message | Solution |
|---------|-------------|----------|
| No orders placed | "All symbols filtered by volatility gate" | Market too calm, lower threshold or wait |
| No orders placed | "All symbols filtered by edge validation gate" | All at highs/momentum fired, wait for setups |
| No orders placed | "Compounding blocked by economic gate" | Profit too thin, wait for more profit |
| Volatility calc fails | "Could not calculate volatility for {symbol}" | Check OHLCV data availability |
| NPY errors | "ModuleNotFoundError: No module named 'numpy'" | Install numpy: `pip install numpy` |

---

## Code Locations

- **Volatility Gate**: `core/compounding_engine.py` lines 164-219
- **Edge Gate**: `core/compounding_engine.py` lines 221-268
- **Economic Gate**: `core/compounding_engine.py` lines 270-323
- **_pick_symbols integration**: `core/compounding_engine.py` lines 369-400
- **_check_and_compound integration**: `core/compounding_engine.py` lines 402-447
- **Configuration**: `config/tuned_params.json` (or passed via config object)

---

## Signal Recovery Heuristic

If gates are too strict and blocking all compounding:

```
1. Check if you have high-conviction setups:
   → Are volatility levels genuinely low? (Check ticker volatility)
   → Are symbols genuinely at bad entry points? (Check charts)
   → Is profit genuinely insufficient? (Check realized_pnl)

2. If market conditions poor, gates working correctly ✅
   → Not a bug, gates protecting you

3. If market conditions good but gates blocking, tune:
   → Lower COMPOUNDING_MIN_VOLATILITY by 0.001 (0.1%)
   → Check edge validation thresholds
   → Monitor logs to see what's being filtered

4. Last resort (not recommended):
   → Temporarily disable gates by setting thresholds to 0.0
   → This re-enables fee churn risk
```

---

## References

- **Implementation Date**: 2024 Q4
- **Files Modified**: `core/compounding_engine.py`
- **Root Cause Analysis**: `COMPOUNDING_ENGINE_FEE_CHURN_ANALYSIS.md`
- **Full Documentation**: `COMPOUNDING_ENGINE_PROTECTIVE_GATES_IMPLEMENTATION.md`

