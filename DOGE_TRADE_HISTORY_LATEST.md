# 📊 DOGE TRADE HISTORY - LAST TRADES ANALYSIS

**Date:** April 28, 2026, 00:03-00:22 UTC  
**Status:** Recent trades analyzed from system logs  
**Log Source:** `system_restart_20260428_000327.log`

---

## 🎯 RECENT DOGE TRADES EXECUTED

### Trade #1 - POSITION CLOSED (Missed Fill Recovery)
```
Symbol:           DOGEUSDT
Type:             POSITION FULLY CLOSED
Timestamp:        2026-04-28 00:03:44
Current Qty:      1.00000000 DOGE
Executed Qty:     32.00000000 DOGE
Executed Price:   $0.15625000 per DOGE
Total Value:      $5.00 USDT (32 × $0.15625)
Reason:           TRUTH_AUDIT:missed_fill_recovery
Tag:              truth_auditor
Status:           ✅ CLOSED
```

**Analysis:**
- This was a recovery operation
- 1 DOGE position closed via execution of 32 DOGE worth
- Price per DOGE: $0.15625 (mid-range price)
- Reason: Truth Auditor detected missed fill and recovered it
- **Action Type:** Capital Recovery

---

### Trade #2 - POSITION CLOSED (Missed Fill Recovery)
```
Symbol:           DOGEUSDT
Type:             POSITION FULLY CLOSED
Timestamp:        2026-04-28 00:03:44 (same as Trade #1)
Current Qty:      2.00000000 DOGE
Executed Qty:     100.0000000 DOGE
Executed Price:   $0.09511000 per DOGE
Total Value:      $9.51 USDT (100 × $0.09511)
Reason:           TRUTH_AUDIT:missed_fill_recovery
Tag:              truth_auditor
Status:           ✅ CLOSED
```

**Analysis:**
- Another recovery operation (same batch)
- 2 DOGE position closed via execution of 100 DOGE worth
- Price per DOGE: $0.09511 (lower price than Trade #1)
- Reason: Truth Auditor detected missed fill and recovered it
- **Action Type:** Capital Recovery

---

## 📈 TRADING SIGNALS RECEIVED (After Positions Closed)

### Signal Timeline (04:57 onwards)

**04:57 UTC - SELL Signal #1**
- Agent: SwingTradeHunter
- Signal Type: **SELL**
- Confidence: 0.65 (65%)
- Status: ✅ PASSED ALL GATES
- Reason: EMA downtrend detected
- Position Check: ⚠️ NO POSITION (qty=0.00000000)
- Action: **SKIPPED** - Cannot sell without position

**05:05 UTC - SELL Signal #2**
- Agent: SwingTradeHunter
- Signal Type: **SELL**
- Confidence: 0.65 (65%)
- Status: ✅ PASSED ALL GATES
- Reason: EMA downtrend detected
- Position Check: ⚠️ NO POSITION
- Action: **SKIPPED** - Cannot sell without position

**Multiple SELL Signals (05:09, 05:15, 05:21, 05:27, 05:33+)**
- All from SwingTradeHunter
- All with 0.65 confidence
- All detecting EMA downtrend
- All SKIPPED due to no position

---

## 🔍 SIGNAL ANALYSIS & INDICATORS

### Technical Analysis for DOGE (Latest)
```
EMA20:      0.0982
EMA50:      0.0984
RSI:        51.55
MACD Hist:  -0.000036

Signal:     ✅ SELL (EMA20 < EMA50)
Status:     EMA downtrend detected
Confidence: 0.65 (65%)
```

**Interpretation:**
- EMA20 (0.0982) < EMA50 (0.0984) → Downtrend signal
- RSI 51.55 → Neutral (neither overbought nor oversold)
- MACD slightly negative → Momentum weakness
- **Conclusion:** Bearish short-term trend

---

## ⚠️ ML FORECASTER ANALYSIS

### ML Decision at 00:22:45
```
Schema:           scalar_hold_buy
Feature Mode:     full_ohlcv_edge
Lookback:         60 periods
Feature Dims:     29 features

Model Output:
  Buy Probability:   76.67%
  Hold Probability:  23.33%
  
Final Decision:   BUY
Confidence:       0.77 (77%)
Position Scale:   1.50x

Suppression Reason:
  Expected Move:  0.2350% < Required 0.6080%
  Multiplier:     1.60x
  Round Trip Cost: 0.3800%
  
Status:           ❌ BUY SUPPRESSED
```

**Analysis:**
- ML model predicts BUY with 77% confidence
- However, expected move (0.235%) is too small to justify trade
- After fees/slippage, would result in loss
- **Reason:** Risk/reward unfavorable
- **Action:** Signal suppressed (not traded)

---

## 📊 DOGE TRADING SUMMARY

### Current State
```
Position:         CLOSED ✅
Quantity:         0.00000000 DOGE
Status:           NO ACTIVE POSITION

Recent Exits:
  Trade #1: 1 DOGE @ $0.15625   ($5.00)
  Trade #2: 2 DOGE @ $0.09511   ($9.51)
  Total Recovered: $14.51 USDT
```

### Signal Status
```
SELL Signals:     6+ received (all skipped due to no position)
BUY Signal:       Suppressed (expected move too small)
Current Action:   MONITORING ONLY
```

### Market Condition
```
Trend:            Downtrend (EMA downtrend detected)
Volatility:       Low (regime="low")
RSI:              Neutral (51.55)
ML Forecast:      BUY (77% conf) but suppressed due to low expected move
```

---

## 💡 KEY OBSERVATIONS

### ✅ What Happened

1. **Recovery Operations (00:03:44)**
   - Two positions closed via Truth Auditor recovery
   - Recovered 32 DOGE (Trade #1) and 100 DOGE (Trade #2)
   - Total capital recovered: $14.51 USDT
   - Successfully freed capital from these positions

2. **After Recovery**
   - System generating continuous SELL signals
   - All signals SKIPPED because position is 0
   - Cannot sell what you don't hold

3. **No BUY Currently**
   - ML model shows 77% confidence for BUY
   - But signal suppressed due to unfavorable expected move
   - Risk/reward ratio too poor at current prices

### ⚠️ Why SELL Signals Are Skipped

```
Reason: Skip SELL for DOGEUSDT — no position (qty=0.00000000)

Logic:
├─ SELL signal received ✅
├─ Signal validation passed ✅
├─ Checks position quantity ❌
├─ qty = 0.00000000
├─ Cannot sell zero position
└─ Signal SKIPPED

Status: WAITING FOR ENTRY
```

### 🎯 Next Action

**What the system is waiting for:**
1. A favorable BUY signal (with good risk/reward)
2. Entry at low price
3. Then SELL at higher price (exit the downtrend)

**Current Blocker:**
- Expected move is 0.235%
- After trading costs (0.38%), would result in loss
- System correctly waits for better opportunity

---

## 📋 DOGE DUST EXIT STATUS

### Last Known State
```
Position:         CLOSED ✅
Remainder:        0 DOGE (clean exit)
Dust Detection:   N/A (no position)
Status:           CLEAN - No dust left
```

### Dust Exit Mechanism Applied
- When positions closed, dust detection ran
- Both positions fully closed (no remainder)
- No stuck dust scenarios
- Capital fully recovered

---

## 🎯 TRADING SETUP

### Signal Generators Active
1. **SwingTradeHunter** (Active)
   - EMA-based strategy
   - Currently generating SELL signals
   - Confidence: 0.65 (65%)
   
2. **TrendHunter** (Active)
   - ML-enhanced trend following
   - Currently low-confidence (0.41 < 0.60 min)
   - Prefilter: PASSED
   
3. **MLForecaster** (Active)
   - Full OHLCV feature analysis
   - Predicts BUY (77% confidence)
   - Suppressed due to expected move < required

### Configuration
```
Min Signal Confidence:   0.35
Current Signals:        0.65-0.77
ML Forecast:            BUY (suppressed)
Signal Status:          Ready to enter on opportunity
```

---

## ✅ CONCLUSION

### Status Summary
- ✅ Recent positions successfully closed (recovered $14.51)
- ✅ No dust left behind (clean exits)
- ⚠️ Currently no active DOGE position
- ⚠️ SELL signals skipped (no position to sell)
- ⏳ Waiting for favorable BUY opportunity

### Next Expected Action
System will execute a BUY when:
1. Expected move becomes favorable (> 0.608%)
2. Risk/reward improves
3. Either new signal with better confidence or better price action

### Current Market
- **Trend:** Downtrend (bearish)
- **Volatility:** Low
- **ML Forecast:** BUY (but suppressed)
- **Trading State:** IDLE/WAITING

---

**Report Generated:** April 28, 2026, 00:22:45 UTC  
**Data Source:** system_restart_20260428_000327.log  
**Status:** ✅ ANALYSIS COMPLETE

