# 📊 Signal Edge Tuning — Visual Guide

## The Core Concept

```
Signal emitted
    ↓
Price at +5m:  $47,250 → $47,462  → Return: +0.45%
Price at +15m: $47,250 → $47,543  → Return: +0.62%
Price at +30m: $47,250 → $47,605  → Return: +0.75%

Minus trading costs (buy + sell): 0.12%

Edge at +5m:  0.45% - 0.12% = 0.33% ✅ (profit remains after costs)
Edge at +15m: 0.62% - 0.12% = 0.50% ⚠️  (too much, model too cautious)
Edge at +30m: 0.75% - 0.12% = 0.63% ⚠️  (too much, model too cautious)
Average:      0.49% → INCREASE_CONFIDENCE_FLOOR
```

## The Tuning Decision Tree

```
                    ┌─ Collect 100+ signals
                    │
                    ├─ Measure avg_edge
                    │
        ┌───────────┼───────────┐
        │           │           │
        ↓           ↓           ↓
     >0.4%       0.2-0.4%     <0.2%
       │           │           │
       ↓           ↓           ↓
    ⚠️ TOO      ✅ OPTIMAL    ❌ INSUFF
   CONSERVATIVE              ICIENT
       │           │           │
       ↓           ↓           ↓
    Lower      Monitor      Retrain
    Conf       Current      Model
    Floor      Params
       │           │           │
       └────┬───────┴───────────┘
            │
            ↓
    Collect 30+ new signals
            │
            ↓
    Compare new avg_edge
            │
            ↓
    Better? Keep. Worse? Revert.
```

## Signal Confidence vs Edge

### Model Too Conservative (⚠️)
```
Confidence threshold too high
    │
    ↓
Only highest-confidence signals trigger
    │
    ↓
But these signals often continue to move favorably
    │
    ├─ You take 50% fewer trades
    ├─ But each has +0.50% edge
    └─ You leave money on the table
    
Solution: Lower confidence threshold
    conf_threshold = 0.70 → 0.65
    
Result: 
    • 50% more trades (opportunity)
    • Edge drops to 0.35% (more common price moves)
    • But overall profit ↑ because of volume
```

### Model Well-Tuned (✅)
```
Confidence threshold optimal
    │
    ↓
Right proportion of signals trigger
    │
    ├─ 70% have meaningful edge (>0.2%)
    ├─ Avg edge is 0.25-0.35%
    └─ Entry quality is consistent
    
Result:
    • Steady profit stream
    • Low variance in edge
    • Minimal over-trading or under-trading
```

### Model Insufficient (❌)
```
Confidence threshold too low or model weak
    │
    ├─ Taking mediocre signals
    └─ Or entry timing is poor
    
Result:
    • Most signals barely move in favorable direction
    • Avg edge < 0.2% (costs exceed benefit)
    • Some trades go negative immediately
    
Solution: Either
    A) Disable trading until model retrained
    B) Retrain with:
        • More historical data
        • Better features
        • Cross-validation
        • Regime awareness
```

## Time Window Progression

```
At Signal Time:
    Price = $50,000
    Record this as baseline

At +5m:
    Price = $50,100
    Return = 0.20%
    Edge = 0.20% - 0.12% = 0.08% ✅

At +15m:
    Price = $50,180
    Return = 0.36%
    Edge = 0.36% - 0.12% = 0.24% ✅

At +30m:
    Price = $50,250
    Return = 0.50%
    Edge = 0.50% - 0.12% = 0.38% ✅

Observation: Price keeps moving in our favor
→ Model gave us good signal
→ Maybe we're being TOO cautious about entry?
→ Consider lowering confidence threshold
```

## Three Scenarios

### Scenario 1: TOO_CONSERVATIVE

```
Signal fired 100 times:
    ✅ 80 times (80%)   — Avg edge = 0.42%
    ⚠️  15 times (15%)  — Avg edge = 0.35%
    ❌ 5 times  (5%)   — Avg edge = 0.08%

Average across all: 0.35% (heavily skewed to ✅)

Interpretation:
    "Model is right 95% of the time with +0.3-0.4% edge.
     But we're only trading in cases where confidence is VERY HIGH.
     Maybe we should trade at lower confidence levels too?"

Action:
    confidence_threshold = 0.70 → 0.65
    
Expected outcome:
    • More trades fire (captured medium-confidence signals)
    • Avg edge drops to 0.28% (more diverse prices)
    • But total P&L ↑ due to higher volume
    • Re-evaluate after 30+ new signals
```

### Scenario 2: OPTIMAL

```
Signal fired 100 times:
    ✅ 62 times (62%)  — Avg edge = 0.28%
    ⚠️  25 times (25%) — Avg edge = 0.22%
    ❌ 13 times (13%)  — Avg edge = 0.05%

Average across all: 0.22% (balanced distribution)

Interpretation:
    "Model has realistic edge distribution.
     Good signals are clearly separated from bad ones.
     Entry criteria are well-calibrated."

Action:
    Keep current parameters
    Monitor for regime changes (volatility, sector rotation)
    Adjust only if market structure changes
```

### Scenario 3: INSUFFICIENT

```
Signal fired 100 times:
    ✅ 25 times (25%)  — Avg edge = 0.12%
    ⚠️  35 times (35%) — Avg edge = 0.08%
    ❌ 40 times (40%)  — Avg edge = -0.02%

Average across all: 0.05% (heavily skewed to ❌)

Interpretation:
    "Model fires signals but they don't work.
     40% of trades lose money immediately.
     Trading costs exceed expected gains."

Action:
    DO NOT TRADE this pair until model retrained
    
    Retrain with:
    • 50 periods lookback (was 20)
    • Add 5 technical indicators (RSI, MATR, Bollinger Bands, etc.)
    • Include market regime (high volatility? trending?)
    • Use cross-validation on recent 3 months
    • Deploy to paper trading for 2 weeks before live
    
Expected outcome:
    • Better feature representation
    • Model learns more nuanced entry conditions
    • Avg edge returns to 0.25%+
```

## Quick Health Check

Run this mentally every day:

```
Q1: What's my average edge across all signals?
    <0.2%   → Retrain or stop trading
    0.2-0.4% → Keep going
    >0.4%   → More aggressive, relax entry

Q2: What's my optimal signal percentage?
    <50%    → Model is unreliable, retrain
    50-70%  → Good calibration
    >70%    → Very conservative, could be more aggressive

Q3: Any signals with negative edge?
    >20%    → Data quality issue, investigate
    10-20%  → Normal, keep going
    <10%    → Very rare, expected randomness

Q4: How consistent is edge?
    Std > 0.20% → High variance, hard to predict
    Std 0.10-0.20% → Moderate variance, tradeable
    Std < 0.10% → Very consistent, excellent model
```

## Pro Tip: Confidence Stratification

Instead of one threshold, use tiers:

```
Confidence Level    Min_Edge    Action
─────────────────────────────────────────
0.90+               0.3%        MUST_TRADE
0.80-0.89           0.25%       SHOULD_TRADE
0.70-0.79           0.20%       CAN_TRADE
0.60-0.69           0.15%       MAY_TRADE
<0.60               SKIP        DON'T_TRADE
```

This way you can trade more often while maintaining edge minimum.
Example: Lower tier has slightly smaller position, but more frequency.

## Remember

> **The edge is real. Measure it. Improve it. Profit from it.**
>
> If your model leaves >0.4% on the table, you're being too cautious.
> If your model costs more than it makes (<0.2%), you're wasting time.
> If your model nets 0.25%, you're doing it right.
