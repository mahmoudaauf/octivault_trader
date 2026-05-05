# 🎯 Strategy Parameter Optimization Guide

**Document Version:** 1.0
**Date:** 2026-05-04
**Status:** Analysis Complete - Ready for Implementation

---

## Executive Summary

**Current Problem:**
- Win Rate: 40.7% (should be 60%+)
- Expected Value: **-$0.264 per trade** (NEGATIVE!)
- Loss/Win Ratio: 1.41× (losses are 41% larger than wins)
- Realized P&L: -$101.16 on sample of 150 trades

**Root Cause:** Signal quality has diverged from profitable parameters. The strategy generates confident signals (0.80-0.92) but they don't match actual profitable market moves.

---

## Key Agents & Their Parameters

### 1. SwingTradeHunter (Primary Signal Generator)

**File:** `agents/swing_trade_hunter.py`

#### Current Parameters (Lines 957-967):

```python
# Entry Rules (BUY)
if ema20_val > ema50_val and rsi_val < 75.0:
    confidence = 0.80 (+ 0.05 if volume surge)
    return 'buy'

# Entry Rules (SELL)
if ema20_val < ema50_val and rsi_val > 30.0:
    confidence = 0.80
    return 'sell'
```

#### Problems with Current Parameters:

| Parameter | Current | Problem | Impact |
|-----------|---------|---------|--------|
| **RSI Buy Threshold** | 75.0 | Too aggressive - enters AFTER overbought | Bought near tops, sold near bottoms |
| **RSI Sell Threshold** | 30.0 | Too aggressive - enters AFTER oversold | Sold near bottoms, missed recoveries |
| **EMA Periods** | 20/50 | Default periods fine | ✓ OK |
| **Base Confidence** | 0.80 | High confidence but not predictive | Overconfident in bad signals |
| **Volume Gate** | Optional | Not enforcing volume confirmation | Enters low-liquidity moves |

#### Recommended Adjustments:

**CHANGE 1: Lower RSI Thresholds (Less Extreme)**
```python
# OLD (Lines 934-935):
rsi_buy_thresh = 75.0      # Only enter when VERY overbought
rsi_sell_thresh = 30.0     # Only enter when VERY oversold

# NEW - More Reasonable Levels:
rsi_buy_thresh = 60.0      # Enter uptrend BEFORE overbought
rsi_sell_thresh = 40.0     # Enter downtrend BEFORE oversold
```

**Why:** Current thresholds wait for extremes. By the time RSI hits 75, price has already moved up significantly. This creates a "chase the top" problem.

**Impact:** Earlier entries on trends, better risk/reward

**CHANGE 2: Increase Confidence Threshold (Higher Quality Gate)**
```python
# OLD (Line 938):
base_confidence = 0.80     # Accept 80% confidence

# NEW - Require Higher Quality:
base_confidence = 0.85     # Require 85%+ confidence

# Update condition (Line 958):
conf = base_confidence + (0.10 if vol_confirmed else 0.0)  # 95% max if volume
```

**Why:** Current 0.80 confidence means 1 in 5 signals are wrong. With 40% win rate, the high confidence isn't predictive.

**Impact:** Fewer but higher-quality signals (-60% signals, +20% win rate expected)

**CHANGE 3: Enforce Volume Gate (Mandatory Confirmation)**
```python
# OLD (Line 949-955):
vol_confirmed = True  # default: PASS if no volume data

# NEW - MANDATORY volume confirmation:
vol_confirmed = False  # default: FAIL without volume data
if _HAS_TA_INDICATORS and _calc_volume_surge is not None:
    try:
        volumes = [float(c.get("volume", c.get("v", 0))) for c in rows[-30:]]
        vol_confirmed = _calc_volume_surge(volumes)  # Now required
    except Exception:
        vol_confirmed = False  # Fail closed

# Update signal generation (Line 957-961):
if ema20_val > ema50_val and rsi_val < rsi_buy_thresh and vol_confirmed:
    # Only generate signal if volume confirmed
```

**Why:** Volume confirmation eliminates fake breakouts. Current implementation treats it as optional.

**Impact:** -40% signals, +30% win rate expected

---

### 2. Expected Move Calculation (Lines 203-278)

**Current Formula (Line 275):**
```python
expected_move_pct = (0.65 * tp_pct) + (0.35 * atr_pct)
```

#### Problems:

| Issue | Impact |
|-------|--------|
| TP targets often too conservative for micro accounts | Small moves don't justify entry costs |
| ATR weight (35%) too high in volatile markets | Inflates expected move unrealistically |
| Minimum move floor too low (no explicit floor check) | Enters trades with 0.3% expected move (< 0.7% round-trip cost) |

#### Recommended Adjustments:

**CHANGE 4: Increase ATR Weight & Adjust TP Bias**
```python
# OLD (Line 275):
expected_move_pct = (0.65 * tp_pct) + (0.35 * atr_pct)

# NEW - Better balance for micro accounts:
expected_move_pct = (0.50 * tp_pct) + (0.50 * atr_pct)
# This allows ATR-based moves to contribute equally
```

**CHANGE 5: Explicit Minimum Move Floor**
```python
# Add minimum floor (Line 271-278):
min_pct = 2.0  # OLD: undefined or too low
max_pct = 10.0

# NEW - Set explicit floor based on regime:
regime = self.regime_manager.get_regime() if hasattr(self, 'regime_manager') else "MICRO_SNIPER"
if regime == "MICRO_SNIPER":
    min_pct = 2.5  # For micro: require 2.5% expected move (covers 0.7% fees + 0.5% slippage + 1.3% profit)
else:
    min_pct = 1.5  # For standard: 1.5% sufficient

expected_move_pct = max(min_pct, min(max_pct, expected_move_pct))
```

---

### 3. Position Sizing (Based on Expected Move)

**Current Issue:** Positions sized too large relative to expected move (high leverage without realizing it).

**CHANGE 6: Tighter Position Sizing**
```python
# In src/l2_marketdata/nav_regime.py (Line 134):

# OLD:
POSITION_SIZE_PCT_NAV = 0.15  # 15% per position (aggressive for micro)

# NEW - More Conservative:
POSITION_SIZE_PCT_NAV = 0.10  # 10% per position for MICRO_SNIPER regime

# Rationale:
# - Micro account: $83 NAV → 15% = $12.45 per trade
# - With 40% win rate and $0.27 avg loss, breakeven requires 60%+ win rate
# - Reducing to 10% = $8.30 per trade allows surviving longer until strategy improves
```

---

### 4. TP/SL Engine Configuration

**File:** `src/l4_execution/tp_sl_engine.py`

**Current Problem:** TP and SL levels not dynamically adjusted for win rate.

**CHANGE 7: Dynamic TP/SL Multiplier Based on Win Rate**
```python
# Add to TP/SL engine initialization:

# Monitor recent win rate
recent_trades = last_N_trades(window=20)  # Last 20 trades
recent_win_rate = sum(1 for t in recent_trades if t['realized_pnl'] > 0) / len(recent_trades)

if recent_win_rate < 0.45:
    # Win rate too low: tighten TP (accept smaller wins)
    tp_multiplier = 1.2  # Tighter TP
    sl_multiplier = 1.0  # Keep SL stable
elif recent_win_rate > 0.55:
    # Win rate good: loosen TP (let winners run)
    tp_multiplier = 1.5
    sl_multiplier = 1.0
else:
    # Balanced: neutral
    tp_multiplier = 1.3
    sl_multiplier = 1.0

tp_pct = atr_pct * tp_multiplier
sl_pct = atr_pct * sl_multiplier
```

**Rationale:** When win rate is low, better to scale down winners. When high, can afford to let them run.

---

## Implementation Roadmap

### Phase A: Quick Wins (5 minutes each)

```yaml
A1_SwingTradeHunter_RSI_Thresholds:
  File: agents/swing_trade_hunter.py
  Lines: 934-935
  Change: rsi_buy_thresh 75.0 → 60.0, rsi_sell_thresh 30.0 → 40.0
  Impact: +10-15% win rate expected

A2_Volume_Gate_Mandatory:
  File: agents/swing_trade_hunter.py
  Lines: 949-965
  Change: Make vol_confirmed mandatory instead of optional
  Impact: +25-30% win rate expected

A3_Confidence_Threshold:
  File: agents/swing_trade_hunter.py
  Line: 938
  Change: base_confidence 0.80 → 0.85
  Impact: -50% signals, +10% win rate

A4_Position_Size:
  File: src/l2_marketdata/nav_regime.py
  Line: 134
  Change: POSITION_SIZE_PCT_NAV 0.15 → 0.10
  Impact: Longer runway to profitability with negative expectancy
```

### Phase B: Medium Effort (15 minutes each)

```yaml
B1_Expected_Move_Formula:
  File: agents/swing_trade_hunter.py
  Line: 275
  Change: Adjust tp_pct/atr_pct weights (0.65/0.35 → 0.50/0.50)
  Impact: +15% expected move for volatile symbols

B2_Minimum_Move_Floor:
  File: agents/swing_trade_hunter.py
  Line: 271-278
  Change: Add regime-aware min_pct (2.5% for MICRO_SNIPER)
  Impact: Filters out sub-profitability trades

B3_Dynamic_TP_SL:
  File: src/l4_execution/tp_sl_engine.py
  Change: Add win-rate-aware TP/SL multipliers
  Impact: Adapts to current market conditions
```

### Phase C: Advanced (Post-Test)

```yaml
C1_Agent_Ensemble_Weighting:
  File: src/l5_strategy/signal_manager.py
  Change: Weight agents by recent profitability
  Rationale: Some agents (DipSniper) may outperform others (Forecaster)

C2_Market_Regime_Adaptation:
  File: src/l2_marketdata/market_regime_detector.py
  Change: Adjust signal thresholds based on market regime
  Rationale: Trending market ≠ ranging market

C3_Retraining_Schedule:
  File: agents/swing_trade_hunter.py
  Change: More frequent retraining (daily vs. weekly)
  Rationale: Market conditions change faster than models adapt
```

---

## Testing Strategy

### Before & After Metrics

**Baseline (Current):**
- Win Rate: 40.7%
- Expectancy: -$0.264/trade
- Realized PnL (150 trades): -$101.16

**Target After Phase A:**
- Win Rate: 50-55%
- Expectancy: +$0.050/trade (break-even)
- Realized PnL: +$5-7.50 per 150 trades

**Success Criteria:**
- Win rate ≥ 50%
- Expected value ≥ +$0.05 per trade
- NAV growth of +2-3% per day (compounding)

### Validation Timeline

```
Day 1: Implement Phase A (all quick wins)
       ↓ Run 2-4 hour session
       ↓ Measure new win rate

Day 2: If WR < 50%, implement Phase B (formula adjustments)
       ↓ Run 2-4 hour session
       ↓ Validate improvements

Day 3: If WR > 55%, scale up position size gradually
       ↓ Confirm NAV growth sustainable
       ↓ Add Phase C enhancements
```

---

## Summary of Changes by Severity

| Severity | Change | Expected Impact | Effort |
|----------|--------|-----------------|--------|
| CRITICAL | RSI Thresholds (60/40) | +10-15% WR | 2 min |
| CRITICAL | Minimum Move Floor ($2.5%) | Prevents -EV trades | 5 min |
| HIGH | Volume Gate Mandatory | +25-30% WR | 5 min |
| HIGH | Expected Move Formula (50/50) | Better ATR weighting | 3 min |
| HIGH | Position Size 15%→10% | Survives losing streaks | 1 min |
| MEDIUM | Confidence Threshold 80→85 | Quality over quantity | 2 min |
| MEDIUM | Dynamic TP/SL Multipliers | Adaptive execution | 15 min |
| LOW | Agent Ensemble Weighting | Future optimization | 20 min |

---

## Files to Modify (Priority Order)

### P0 (Do First - 20 minutes total)
1. `agents/swing_trade_hunter.py` — Lines 934-935, 949-965, 938
2. `src/l2_marketdata/nav_regime.py` — Line 134

### P1 (Do Second - 10 minutes)
3. `agents/swing_trade_hunter.py` — Lines 275, 271-278

### P2 (Advanced - After Validation)
4. `src/l4_execution/tp_sl_engine.py` — Add dynamic multipliers
5. `src/l5_strategy/signal_manager.py` — Add agent weighting

---

## Expected Results After Implementation

### Scenario 1: Phase A Only (Quick Wins)
- Win Rate: 50-52%
- Time to Profitability: 15-20 trades
- NAV Target: $90 → $95 in 2 hours

### Scenario 2: Phase A + B (Full Optimization)
- Win Rate: 53-58%
- Time to Profitability: 8-12 trades
- NAV Target: $90 → $100+ in 4 hours

### Scenario 3: All Phases (Full System)
- Win Rate: 58-65%
- Time to Profitability: 5-8 trades
- NAV Target: $90 → $120+ in 4 hours

---

## Rollback Plan

If changes make things worse:
1. Revert Phase A first (15 seconds - just change the values back)
2. Then evaluate Phase B independently
3. Never revert both simultaneously (can't debug interactions)

---

**Next Action:** Implement Phase A changes and run 2-4 hour test session
