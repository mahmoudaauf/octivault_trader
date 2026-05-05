# Capital Rebalancing Strategy

## Current State
- Total: $103.84
- Free: $10.02 (9.6%) ← TOO LOW
- Invested: $93.82 (90.4%)

## Goal State (after rebalance)
- Total: $103.84
- Free: $30-40 (30-40%) ← OPTIMAL
- Invested: $63-73 (60-70%)

## Action Plan

### Step 1: Identify Winners to Close (IMMEDIATE)
Look for positions with +2% or more profit:
- Example: If ETHUSDT was bought at $2420 and now at $2460, close it
- Capture: ~$3-5 profit per winner
- Target: Close 2-3 winners → Free $20-30

### Step 2: Use BalanceSync to Monitor
- Current NAV updates every 3 seconds
- Watch for delta > 0 = winning positions
- Close immediately when TP target hit

### Step 3: Reinvest Freed Capital
Once $30+ free:
- ✅ DOGEUSDT BUY signals will execute (now have capital)
- ✅ Confidence gate will pass (0.62 < 0.65)
- ✅ Position sizing will be adaptive ($10-15 trades)

### Step 4: Growth Loop
```
Close winner (+$5)
  → Free capital now $15.02
  → Can enter new trade with $10
  → Ride trend to +$3-5
  → Close and repeat
```

## Expected Timeline
- Rebalance: 10-30 min (wait for winning exit signals)
- First new trade: 5 min after rebalance
- First profit: 15-30 min after entry (trend-following strategy)
- Account growth: +5-10% per day once flowing

## Success Metrics
| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Free capital % | 9.6% | 35% | 🟡 In progress |
| Trades/hour | 0 | 2-4 | 🟡 Blocked by capital |
| Daily growth | 0% | +5-10% | 🟡 Pending |
