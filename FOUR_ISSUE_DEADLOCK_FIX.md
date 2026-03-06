# 🔴 CRITICAL 4-ISSUE DEADLOCK FIX

## Issues Identified

1. **TrendHunter BUY signals not reaching MetaController** - Strategy signals being dropped
2. **SOL position prevents buying** - ONE_POSITION_PER_SYMBOL gate too strict
3. **ProfitGate prevents selling SOL** - Loss position can't exit due to -29.768% PnL
4. **PortfolioAuthority keeps trying to rebalance SOL** - Stuck in loop trying to exit position

## Root Cause Analysis

### Issue #1: BUY Signals Not Reaching MetaController
- TrendHunter is generating signals (we see them in buffer)
- Signals aren't being cached in SignalManager
- Likely cause: Signal formatting or agent name mismatch

### Issue #2: SOL Position Prevents Buying
- ONE_POSITION_PER_SYMBOL gate at line ~9850 rejects ANY BUY if qty > 0
- SOL exists from earlier trades with -29.768% loss
- Bot can't average down (gate doesn't allow)
- Need to add exception for forced recovery

### Issue #3: ProfitGate Prevents Selling SOL
- Profit gate at line 2669 requires: `pnl_pct >= min_profit`
- SOL has `pnl_pct = -0.29768 < min_profit = 0.005` (0.5%)
- Forced recovery exit attempts blocked by this gate
- Gate needs override for PortfolioAuthority exits

### Issue #4: PortfolioAuthority Stuck in Loop
- PORTFOLIO_REBALANCE attempts to exit SOL every cycle
- Gets blocked by profit gate
- Never succeeds, tries again next cycle
- Need circuit breaker after N failures

## Solutions

### Fix #1: TrendHunter Signal Caching
- Add logging to verify signal transmission
- Check SignalManager.add_signal() is being called
- Verify signal format matches expected schema

### Fix #2: ONE_POSITION_PER_SYMBOL Override
- Add exception for FORCED_RECOVERY exits
- Allow rebalance/concentration exits to bypass position limit
- Keep limit for normal BUY signals

### Fix #3: ProfitGate Forced Exit Override
- Add flag: `_force_exit=True` to PortfolioAuthority signals
- Check for this flag in profit gate and bypass check
- Still enforce minimum loss limit (e.g., -5% max loss)

### Fix #4: Circuit Breaker for Rebalance Loop
- Track consecutive rebalance failures per symbol
- After 3 failures, disable rebalance for that symbol
- Reset on successful execution

## Implementation Status

Creating fixes now...
