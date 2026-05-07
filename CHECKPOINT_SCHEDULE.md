# Profit Compounding Checkpoint Schedule

**Monitoring Duration**: Continuous until $200 NAV (2x baseline)
**Baseline NAV**: $100 (when API throttle clears)
**Capital Mode**: Paper trading (simulated execution)

---

## Expected Checkpoint Timeline

### Checkpoint 1: NAV = $100 ✅ START
- **Status**: Waiting for API throttle to clear
- **Expected Time**: ~17 minutes from startup
- **Trigger**: First successful balance poll
- **Action**: Baseline NAV recorded, checkpoint monitor begins tracking

### Checkpoint 2: NAV = $110 (+10%)
- **Expected Time**: ~15-30 minutes from baseline
- **Requires**:
  - ≥ 2-3 profitable BUY/SELL cycles
  - Hybrid allocation (5% of $100 = $5 per trade)
  - Win rate ≥ 50%
- **Indicators**:
  - Signals ≥ 5
  - Executions ≥ 2-3
  - Realized PnL ≥ $10

### Checkpoint 3: NAV = $125 (+25%)
- **Expected Time**: ~45-90 minutes from baseline
- **Requires**:
  - ≥ 6-8 profitable cycles
  - Capital recycling from SELL winners
  - Adaptive sizing (ACE adjusting risk_fraction)
- **Indicators**:
  - Signals ≥ 15
  - Executions ≥ 6-8
  - Realized PnL ≥ $25

### Checkpoint 4: NAV = $150 (+50%)
- **Expected Time**: ~2-3 hours from baseline
- **Requires**:
  - ≥ 12-15 profitable cycles
  - OFC adjusting SIZE_MULTIPLIER upward
  - Compounding effect visible
- **Indicators**:
  - Signals ≥ 30
  - Executions ≥ 12-15
  - Realized PnL ≥ $50

### Checkpoint 5: NAV = $200 (+100%)
- **Expected Time**: ~4-6 hours from baseline (full profitability test)
- **Requires**:
  - ≥ 25-30 total profitable cycles
  - Sustained win rate ≥ 55%
  - Zero catastrophic losses
  - Smooth compounding curve
- **Indicators**:
  - Signals ≥ 60
  - Executions ≥ 25-30
  - Realized PnL ≥ $100
  - Drawdown ≤ 5%

---

## What Happens at Each Checkpoint

### Data Recorded
```json
{
  "timestamp": "2026-05-07T19:00:00Z",
  "target": 110.0,
  "actual": 110.25,
  "gain_pct": 10.25,
  "elapsed_sec": 900,
  "trades_executed": 3,
  "profit_realized": 10.25,
  "win_rate": 0.67
}
```

### Alerts Generated
```
🎯 CHECKPOINT REACHED: $110.00
    Actual NAV: $110.25
    Gain: +10.25% from baseline
    Time: 15m 0s
```

### Metrics Updated
- NAV history (for curve analysis)
- Trade execution count
- Profit realization amount
- Time-to-checkpoint

---

## System Behavior During Compounding

### Hybrid Capital Allocation
```
Initial Capital: $100
Trade 1: BUY 5% = $5 position
         SELL for +$1 profit → NAV = $101

Trade 2: BUY 5% = $5.05 position
         SELL for +$2 profit → NAV = $103.05

Trade 3: BUY 5% = $5.15 position
         SELL for +$1.50 profit → NAV = $104.55

... (repeat) → eventually hits $110
```

### ACE (Adaptive Capital Engine)
- Monitors win_rate in 200-trade window
- Adjusts risk_fraction based on historical performance
- If win_rate > 60%: increases position size
- If win_rate < 40%: decreases position size

### OFC (Objective Feedback Controller)
- Every 15 minutes, checks NAV progress
- Adjusts SIZE_MULTIPLIER to track daily target (+2% / 24h)
- If on track: leaves settings alone
- If ahead: reduces position size (lock in gains)
- If behind: increases position size (catch up)

---

## Failure Scenarios (and Automatic Responses)

### Scenario 1: No Signals Generated
- **Symptom**: Signals = 0, Executions = 0
- **Cause**: No high-probability opportunities in CHOPPY market
- **Response**: System waits (skips cycle, preserves capital)
- **Expected**: Next good regime arrives within hours

### Scenario 2: Signals But No Fills
- **Symptom**: Signals > 0, Executions = 0
- **Cause**: Paper mode or execution blocked by risk gates
- **Response**: Check arbitration gates (drawdown, fee tracking)
- **Expected**: Orders execute within 1-2 cycles

### Scenario 3: Trades Executed But Losing
- **Symptom**: Executions > 0, but Realized PnL negative
- **Cause**: Unlucky price action, not a system fault
- **Response**: SELL-for-profit gate blocks losses, preserves capital
- **Expected**: NAV doesn't decrease, win rate recovers

### Scenario 4: Drawdown Exceeds 5%
- **Symptom**: System state switches to DEFENSIVE
- **Cause**: Multiple losing trades or large slippage
- **Response**: OFC sets trading_halted=True, blocks BUY
- **Expected**: Once drawdown recovers, trading resumes

---

## Verifying Profit Compounding

### Check 1: NAV Curve (Should Be Monotonic Upward)
```bash
cat checkpoints.jsonl | jq '.actual'
# Expected: [100, 102.5, 105.3, 108.1, 110, 113, 116, 120, ...]
```

### Check 2: Execution Count (Should Increase)
```bash
grep "cycle.*exe=" live_run.log | awk '{print $NF}' | sort | uniq -c
# Expected: execution count increases over time
```

### Check 3: Realized Profit (Should Accumulate)
```bash
grep "realized_pnl" live_run.log | tail -10
# Expected: positive sums adding up
```

### Check 4: Win Rate (Should Be > 50%)
```bash
# Count SELL_FILLED with positive PnL vs negative
# Expected: profit_closes / total_closes > 0.5
```

---

## Success Criteria

**System is compounding successfully when:**
1. ✅ Baseline NAV visible (> $100)
2. ✅ Checkpoint 1 reached within 30 minutes
3. ✅ Checkpoint 2 reached (within reasonable time)
4. ✅ NAV curve monotonically increasing (no large drawdowns)
5. ✅ Profit recycling evident (capital from SELL winners reinvested)
6. ✅ No errors in startup sequence
7. ✅ Position Hydration Engine working (if restart occurs)
8. ✅ BUY gating enforced (no premature trading)

---

## Next Checkpoints Beyond $200

If system reaches $200:
- Continue monitoring to $300 (+200% gain)
- Measure volatility of NAV curve (smoothness)
- Verify no cascade failures (position count, fee drag)
- Compare actual vs expected compounding rate
- Record maximum drawdown observed

---

**Monitor continuously. Checkpoints auto-logged. No manual intervention needed.**

Start Monitoring: NOW
Expected First Checkpoint: ~18:10 UTC (API throttle recovery)
