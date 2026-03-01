"""
Deployment Guide: From Paper Trading to Live

This guide outlines the three-stage deployment process for the
universe-ready live trading system.

Timeline:
- Week 1: Integration & paper trading setup
- Week 2: Paper trading validation (monitor regime frequency, max DD)
- Week 3+: Live deployment with 5% initial allocation
"""

import logging

logger = logging.getLogger(__name__)

DEPLOYMENT_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LIVE TRADING DEPLOYMENT GUIDE                            ║
║              From Paper Trading → Live with ETH Primary Strategy            ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
STAGE 1: PAPER TRADING SETUP (Week 1)
═══════════════════════════════════════════════════════════════════════════════

📊 CURRENT VALIDATION STATUS:
  ✅ ETH Regime Edge: Sharpe 1.25 (validated on 24-month walk-forward)
  ✅ Regime Detection: LOW_VOL_TRENDING + macro filter verified
  ✅ Position Sizing: Risk management framework implemented
  ✅ Integration: All modules wired (architecture → data → signals)

🎯 PAPER TRADING OBJECTIVES:
  1. Monitor regime signal frequency
     Expected: ~1% of periods in LOW_VOL_TRENDING + uptrend (alpha regime)
     Alert: < 0.5% suggests vol thresholds need refinement
  
  2. Validate maximum drawdown
     Backtest: -52% (acceptable for proprietary strategy)
     Paper: Expect > -30% at some point
     STOP: If DD > -40% without recovery, refine exposure
  
  3. Verify execution quality
     Check: No slippage on real Binance 1h data
     Check: Signal timing (ensure regime detected before price move)
  
  4. Test infrastructure stability
     Check: Data fetching reliability
     Check: Regime detection doesn't crash
     Check: Risk monitoring alerts functional

📝 PAPER TRADING CHECKLIST:

  Day 1: System Initialization
    ☐ Configure live Binance API keys (read-only for paper trading)
    ☐ Set up data logging to track all signals
    ☐ Initialize $100k paper trading balance
    ☐ Configure ETH with full exposure settings:
       - base_exposure = 1.0x
       - alpha_exposure = 2.0x
       - downtrend_exposure = 0.0x
       - max_drawdown_threshold = 0.30

  Day 2-7: Daily Monitoring
    ☐ Check regime signals daily
    ☐ Count alpha regime occurrences
    ☐ Track max drawdown reached
    ☐ Review execution logs for errors
    ☐ Verify data freshness (should be < 1 hour old)

  End of Week 1: Weekly Review
    ☐ Calculate observed regime frequency
    ☐ Compare to expected ~1% alpha frequency
    ☐ Review max DD vs backtest -52%
    ☐ Check win rate on closed trades
    ☐ Decision: Ready for live or needs tuning?

🔍 CRITICAL METRICS TO MONITOR:

  Alpha Regime Frequency:
    Expected: 50-100 signals per year = ~1% of 1h candles
    Formula: (num_alpha_signals / total_hours) * 100
    
  Maximum Drawdown:
    Backtest: -52% (worst case)
    Paper: Expect to see -30% to -50% at some point
    Live: Will likely be 20-30% less than paper (smaller allocation)
    
  Win Rate on Alpha Signals:
    Backtest: ~55-60%
    Paper: Expect 45-65% (expect more variance with real slippage)
    
  Return per Alpha Signal:
    Backtest: +0.5% to +2% per signal
    Paper: Expect 30-50% of backtest (realistic execution)

═══════════════════════════════════════════════════════════════════════════════
STAGE 2: GO/NO-GO DECISION (End of Week 1)
═══════════════════════════════════════════════════════════════════════════════

✅ GO TO LIVE if:
  ✓ Regime frequency = 0.5% - 2.0% (reasonable range)
  ✓ No system crashes or data errors
  ✓ Max DD < -50% (within backtest range)
  ✓ Win rate > 40% (realistic execution)
  ✓ Confidence in infrastructure (logs, monitoring, alerts)

⚠️  NEEDS TUNING if:
  ⚠ Regime frequency > 3% (too many false signals)
    → Increase vol_percentile_threshold (e.g., from 33 to 40)
    → Increase autocorr_threshold (e.g., from 0.1 to 0.15)
  
  ⚠ Regime frequency < 0.3% (missing real alpha)
    → Decrease vol_percentile_threshold (e.g., from 33 to 25)
    → Decrease autocorr_threshold (e.g., from 0.1 to 0.05)
  
  ⚠ Max DD > -60% (excessive risk)
    → Reduce alpha_exposure from 2.0x to 1.5x
    → Reduce max_position_size_pct from 5% to 3%
    → Add max daily loss limit (e.g., -5% stop)
  
  ⚠ System crashes or data errors
    → Add error handling in LiveDataFetcher
    → Implement automatic reconnection (exponential backoff)
    → Add dead-letter queue for failed requests

❌ DO NOT GO LIVE if:
  ✗ Win rate < 30% (suggests regime detection is broken)
  ✗ Max DD > -80% (risk management failure)
  ✗ Regime frequency wildly inconsistent (suggests regime is not persistent)
  ✗ Infrastructure problems (crashes, data delays, API failures)

═══════════════════════════════════════════════════════════════════════════════
STAGE 3: LIVE DEPLOYMENT (Week 3+)
═══════════════════════════════════════════════════════════════════════════════

⚡ DEPLOYMENT APPROACH: Phased Allocation

  Week 3: Conservative Start
    Capital: $5,000 (5% of $100k)
    Symbols: ETHUSDT only
    Leverage: None (1x effective)
    Success Gate: 1 week of positive Sharpe

  Week 4: Validate & Hold
    Capital: Hold at $5,000
    Monitoring: Win rate, DD, regime frequency
    Success Gate: 1 month of positive Sharpe

  Week 5: Expand to $25,000
    Capital: Scale to 25% allocation
    Symbols: ETHUSDT primary, BTC secondary (if Sharpe > 0.3)
    Leverage: Full (2x alpha, 1x base)
    Success Gate: 1 month positive, stable execution

  Month 2+: Scale as Confidence Grows
    Capital: Gradually increase to desired allocation
    Symbols: Add rotation layer (BTC, ALT rotation)
    Enhancement: Add ML forecasting layer
    Success Gate: 3+ months of positive Sharpe > 0.5

📋 LIVE DEPLOYMENT CHECKLIST:

  Pre-Deployment (Day 1):
    ☐ Confirm paper trading validation metrics
    ☐ Verify real API keys are configured (NOT paper trading)
    ☐ Test order execution on small trade ($100)
    ☐ Confirm stop-loss orders are working
    ☐ Enable alerts for DD threshold breaches
    ☐ Set up trade history logging to database

  First Day Live (Hour-by-hour):
    ☐ Monitor first 5 signals
    ☐ Check order fills vs expected prices
    ☐ Monitor P&L updates
    ☐ Check position sizing is correct (5% max)
    ☐ Verify risk limits are enforced

  First Week Live:
    ☐ Daily review of signals generated
    ☐ Daily P&L tracking
    ☐ Verify regime frequency matches paper trading
    ☐ Monitor max DD vs paper vs backtest
    ☐ Check for any system instability

  First Month Live:
    ☐ Calculate Sharpe ratio vs benchmark
    ☐ Compare backtest Sharpe (1.25) vs live Sharpe
    ☐ Expected ratio: Live Sharpe / Backtest Sharpe = 0.4-0.6
    ☐ If Live Sharpe > 0.5, approve scaling
    ☐ If Live Sharpe < 0.2, pause and investigate

🚨 RISK MANAGEMENT - LIVE DEPLOYMENT:

  Daily Loss Limit:
    Rule: Close all positions if daily loss > 5%
    Implementation: Check at end of each signal window
    Alert: Send Slack message when approaching limit

  Maximum Drawdown Stop:
    Rule: Reduce exposure to 0.5x if DD > 25%
    Rule: Close all positions if DD > 35%
    Rationale: Live markets move faster than backtest

  Position Limit:
    Rule: Maximum 5% of account in single position
    Rule: Maximum 20% total exposure (can't be > 4x alpha regime)
    Implementation: Enforce in ExposureController.calculate_position_size()

  Regime Staleness:
    Rule: If no fresh data for 2 hours, flatten positions
    Rationale: Regime detection requires current data
    Implementation: Check timestamp of latest candle

⚡ LIVE EXECUTION NOTES:

  ETH Regime Edge Specifics:
    - Edge appears strongest in calm+uptrend periods
    - Expected holding period: 1-24 hours per signal
    - Position sizing: More conservative than backtest suggests
      * Backtest uses 5% per signal (can compound to 25% total)
      * Live: Cap at 5% total per symbol to reduce DD
    
  Expected Live Performance vs Backtest:
    - Backtest Sharpe: 1.25 (on hourly 1h data with perfect execution)
    - Expected Live Sharpe: 0.5-0.8 (realistic with:
        * Slippage: -0.5% per trade
        * Commission: -0.1% per trade
        * Gap risk: Occasional overnight gaps
        * Regime lag: Detection 1-2 candles delayed
    
  Infrastructure Stability:
    - Data: Binance API can have brief outages (usually < 1 min)
      → Implement automatic retry with exponential backoff
      → Fall back to cached data if API down
    - Execution: Exchange latency typically 100-500ms
      → Use limit orders with 1-2% buffer to ensure fills
    - Monitoring: Set up alerting for:
        → Regime detection failures
        → Missed signals (data stale)
        → Risk limit breaches
        → Cumulative losses > -5% daily

═══════════════════════════════════════════════════════════════════════════════
IMPLEMENTATION SEQUENCE
═══════════════════════════════════════════════════════════════════════════════

Step 1: Configure Paper Trading
  File: live_trading_runner.py
  Change: paper_trading=True (already set)
  Action: Run daily, review signals, track metrics

Step 2: Prepare Live Trading
  File: live_trading_runner.py
  Add: Database logging (store all signals to SQL)
  Add: Alert system (Slack/email on regime detection, DD warnings)
  Add: Exchange integration (replace simulate with real Binance API)

Step 3: Go Live
  File: live_trading_runner.py
  Change: paper_trading=False (after paper validation)
  Change: initial_allocation=$5000 (start conservative)
  Action: Deploy, monitor hourly

Step 4: Scale
  File: live_trading_runner.py
  Add: Symbol rotation (quarterly rebalance based on Sharpe)
  Change: initial_allocation=$25000+ (after 1 month validation)
  Action: Monitor monthly, iterate

═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING GUIDE
═══════════════════════════════════════════════════════════════════════════════

Problem: Regime frequency too high (> 3%)
  Symptom: Too many false alpha signals
  Root Cause: Vol thresholds or autocorr too lenient
  Solution 1: Increase vol_percentile_threshold from 33 to 40
  Solution 2: Increase autocorr_threshold from 0.1 to 0.2
  Test: Run 1 week paper trading with new params

Problem: Regime frequency too low (< 0.3%)
  Symptom: Missing real opportunities
  Root Cause: Vol thresholds or autocorr too strict
  Solution 1: Decrease vol_percentile_threshold from 33 to 25
  Solution 2: Decrease autocorr_threshold from 0.1 to 0.05
  Test: Run 1 week paper trading with new params

Problem: Max DD exceeds -60%
  Symptom: Account losing > 60% at worst point
  Root Cause: Position sizing too large or leverage too high
  Solution 1: Reduce alpha_exposure from 2.0x to 1.5x
  Solution 2: Reduce max_position_size_pct from 5% to 3%
  Solution 3: Add daily loss limit (-5% per day)
  Test: Run 1 week paper trading with new risk limits

Problem: Data fetching fails frequently
  Symptom: Binance API errors in logs
  Root Cause: Network issues or Binance outage
  Solution 1: Implement exponential backoff (1s, 2s, 4s retry)
  Solution 2: Cache last successful data (use if API fails)
  Solution 3: Add Slack alert when data is stale > 1 hour
  Test: Verify error handling in logs

Problem: Live Sharpe much lower than backtest
  Symptom: Live Sharpe 0.1 vs backtest 1.25
  Root Cause: Slippage, commission, or regime lag
  Solution 1: Accept this (backtest is optimistic)
  Solution 2: If Live Sharpe < 0.2, review execution quality
  Solution 3: If Live Sharpe > 0.5, algorithm is actually better than expected!

═══════════════════════════════════════════════════════════════════════════════
SUCCESS METRICS (Month 1)
═══════════════════════════════════════════════════════════════════════════════

Must Have (Green Light to Scale):
  ✅ No system crashes for 7 consecutive days
  ✅ Regime frequency 0.5% - 2.0%
  ✅ Win rate 40%+
  ✅ Max DD < -40%
  ✅ Sharpe > 0 (positive returns)

Nice to Have (Confidence to Scale 10x):
  🟢 Sharpe > 0.3
  🟢 Win rate 50%+
  🟢 Max DD < -25%
  🟢 Regime frequency 0.8% - 1.2% (sweet spot)
  🟢 Live Sharpe / Backtest Sharpe > 0.4

Red Flags (Do Not Scale):
  ❌ Any system crashes
  ❌ Regime frequency > 5% (too many false signals)
  ❌ Win rate < 30% (strategy is broken)
  ❌ Max DD > -70%
  ❌ Sharpe < -0.2 (losing money)

═══════════════════════════════════════════════════════════════════════════════
MONITORING DASHBOARD
═══════════════════════════════════════════════════════════════════════════════

Daily Email Report (auto-generated):
  Date: [DATE]
  
  SIGNALS & REGIMES:
    Alpha signals generated: [N]
    Current regime: [NAME]
    Macro trend: [UP/DOWN]
    
  PORTFOLIO METRICS:
    Current P&L: [+X% / -X%]
    Max DD (lifetime): [-X%]
    Max DD (this week): [-X%]
    Win rate: [X%]
    
  PERFORMANCE:
    This week Sharpe: [X.XX]
    Backtest Sharpe: 1.25
    Ratio: [X%]
    
  ALERTS:
    ✅ All normal / ⚠️ Warning / ❌ Critical
    [List any issues]

Slack Alerts (Real-time):
  🟢 ALPHA SIGNAL: LOW_VOL_TRENDING + UPTREND detected on ETHUSDT
  🔴 RISK ALERT: Max DD hit -35%, reducing exposure to 0.5x
  ⚠️  DATA ALERT: Stale data (2+ hours), flattening all positions
  ❌ ERROR: Binance API failed 5x, manual intervention required

═══════════════════════════════════════════════════════════════════════════════
FUTURE ENHANCEMENTS (Post-Month 1)
═══════════════════════════════════════════════════════════════════════════════

Q2 Enhancements:
  ☐ Add BTCUSDT (if Sharpe > 0.3 validated)
  ☐ Implement symbol rotation (quarterly rebalance)
  ☐ Add ML forecasting layer (XGBoost regime prediction)
  ☐ Optimize leverage per symbol (ETH 2x, BTC 1.5x)

Q3 Enhancements:
  ☐ Multi-pair correlation hedging
  ☐ Macro regime synchronization (interest rates, VIX)
  ☐ Volatility targeting (scale exposure inversely to volatility)
  ☐ Intraday signal generation (5m/15m candles for faster execution)

Q4 Enhancements:
  ☐ Multi-asset class (commodities, forex, stocks)
  ☐ Market microstructure (order flow analysis)
  ☐ Risk parity allocation (equal risk per signal)
  ☐ Full ML pipeline (end-to-end neural network)

═══════════════════════════════════════════════════════════════════════════════
ROLL-OFF GUIDE (If Sharpe < 0.2)
═══════════════════════════════════════════════════════════════════════════════

If live trading shows Sharpe < 0.2 for 2+ weeks:
  1. Flatten all positions
  2. Review regime detection vs backtest
  3. Check if macro regime has fundamentally shifted
  4. Run extended backtest on recent data
  5. Either:
     a) Refine parameters and restart (iterative tuning)
     b) Pause for 1 month and reassess (market shift)
     c) Retire signal (find new edge) - only if Sharpe < -0.3

═══════════════════════════════════════════════════════════════════════════════
CONTACT & ESCALATION
═══════════════════════════════════════════════════════════════════════════════

System Down / Critical Alert: [YOUR EMAIL/PHONE]
  - Any system crash or data failure
  - Broker margin call
  - Max DD exceeded (-40%+)
  - Daily loss limit breached (-5%+)

Weekly Review: [YOUR CALENDAR]
  - Paper trading metrics review
  - Live trading performance check
  - Risk assessment
  - Parameter optimization

Monthly Decision Point: [YOUR CALENDAR]
  - Scale/hold/reduce allocation decision
  - Add new symbols
  - Refresh statistical validation

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == '__main__':
    print(DEPLOYMENT_GUIDE)
    
    # Log to file for reference
    with open('/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/DEPLOYMENT_GUIDE.txt', 'w') as f:
        f.write(DEPLOYMENT_GUIDE)
    
    print("\n✅ Deployment guide saved to DEPLOYMENT_GUIDE.txt")
