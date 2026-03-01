# Quick Start Guide: Live Trading System

## TL;DR - Start Here

```bash
# 1. Run paper trading demo (5 iterations, ~10 seconds)
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 live_trading_runner.py

# Output shows:
# ✅ ETHUSDT detected as ALPHA REGIME or FLAT based on current market
# ✅ Exposure calculated (2.0x if alpha + uptrend, 0.0x if downtrend)
# ✅ Portfolio P&L tracked
```

---

## 3-Minute Overview

### What Is This System?

A **regime-based trading bot** that:
- 🎯 Detects when ETH is in "calm + trending" periods (alpha regime)
- 💰 Sizes positions 2x larger during alpha (more aggressive)
- 🔴 Stops trading during bear markets (0x leverage)
- 📊 Validated on 24 months of real market data (Sharpe 1.25)

### Why ETH?

- ✅ Sharpe ratio 1.25 (validated on 24-month walk-forward)
- ✅ Edge is consistent (mean Sharpe exceeds 0.3 threshold)
- ❌ BTC too weak (Sharpe 0.13 - not good enough)

### Why "Universe-Ready"?

- Can add more symbols (BTCUSDT, ALTUSDT, etc.) via config
- Per-symbol exposure control (different leverage per asset)
- Regime detection works on any OHLCV data
- Framework for future rotation layer

---

## Running the System

### Step 1: Paper Trading Demo (5 minutes)

```bash
python3 live_trading_runner.py
```

**What you'll see:**
```
ITERATION 1
  Fetching data for ['ETHUSDT']...
  Fetched 240 candles for ETHUSDT, latest: 2026-02-21 21:00:00
  
  Regime Summary:
    ETHUSDT: LOW_VOL_MEAN_REVERT | Macro: DOWNTREND | Price: $1979.46
  
  Trading Signals:
    → ETHUSDT: FLAT (0.0x exposure)  ← 0x because in downtrend
  
  Portfolio P&L: +0.00%
```

### Step 2: Enable Real Paper Trading (Week 1)

Edit `live_trading_runner.py`, change line 305:
```python
runner = LiveTradingRunner(
    account_balance=100000,
    paper_trading=True  # ← Keep True for now
)
```

Then run it hourly (set up cron job):
```bash
# Every hour at minute 0
0 * * * * cd /path/to/trader && python3 live_trading_runner.py >> logs/trader.log 2>&1
```

**Monitor these metrics:**
- Alpha regime frequency (expect 0.8-1.2% of candles)
- Maximum drawdown (expect -30% to -52% range)
- Win rate (expect 40%+)
- Current P&L

### Step 3: Go Live (After 1 week paper trading)

**Only if:**
- ✅ Regime frequency is 0.5-2.0%
- ✅ Max DD is reasonable (-40% OK, -70% is problem)
- ✅ No system crashes
- ✅ Data fetching works reliably

**Change to live:**
```python
runner = LiveTradingRunner(
    account_balance=100000,
    paper_trading=False  # ← Switch to False
)
```

**Start small:**
- Deploy with $5,000 only (5% of account)
- Monitor daily for 1 week
- Only scale after 1 week of positive returns

---

## Understanding Signals

### Alpha Regime Signal (Bullish)

```
Regime: LOW_VOL_TRENDING
Macro: UPTREND
Action: LONG
Exposure: 2.0x
Status: ⚡ ALPHA SIGNAL
```

**What it means:**
- Volatility is low (calm market)
- Price is trending up (momentum)
- This is when the strategy makes money
- Position size: 2x normal
- **Action:** Go long (increase position)

### Normal Regime Signal (Neutral)

```
Regime: NORMAL_MEAN_REVERT
Macro: UPTREND
Action: LONG
Exposure: 1.0x
Status: → Normal signal
```

**What it means:**
- Volatility is medium (normal)
- Price is still trending up
- This is OK but not the alpha edge
- Position size: 1x normal
- **Action:** Hold existing positions

### Downtrend Signal (Bearish)

```
Regime: LOW_VOL_MEAN_REVERT
Macro: DOWNTREND
Action: FLAT
Exposure: 0.0x
Status: → No trading
```

**What it means:**
- Volatility is low (calm)
- Price is trending down (declining)
- This is when we lose money
- Position size: 0x (close positions)
- **Action:** Exit all trades, wait for uptrend

---

## Key Files Explained

### `live_trading_runner.py`
**The main file you run.** Coordinates everything:
1. Fetches live data from Binance
2. Detects regime (calm+trending?)
3. Calculates exposure (2x or 1x or 0x?)
4. Manages positions
5. Monitors risk

**Run it:**
```bash
python3 live_trading_runner.py
```

### `live_trading_system_architecture.py`
**The brain.** Contains:
- `RegimeDetectionEngine`: Detects LOW_VOL_TRENDING regime
- `ExposureController`: Maps regime → leverage (2x, 1x, 0x)
- `PositionSizer`: Calculates safe position sizes
- `UniverseManager`: Manages multiple symbols (ETH, BTC, etc.)
- `LiveTradingOrchestrator`: Coordinates everything

**You don't run this directly** - it's imported by `live_trading_runner.py`

### `live_data_pipeline.py`
**The data fetcher.** Contains:
- `LiveDataFetcher`: Pulls real OHLCV data from Binance
- `LivePositionManager`: Tracks open positions and P&L

**You don't run this directly** - it's imported by `live_trading_runner.py`

### `extended_walk_forward_validator.py`
**Proof that it works.** Backtested on 24 months of real data:
- ETH: Sharpe 1.25 ✅ (validated)
- BTC: Sharpe 0.13 ❌ (too weak)

**You already ran this** - this is why we're confident

### `SYSTEM_ARCHITECTURE.md`
**Complete technical documentation.** Read this for:
- Detailed component descriptions
- Configuration examples
- Deployment phases
- Risk management rules

### `deployment_guide.py`
**Step-by-step deployment instructions.** Use this for:
- Paper trading checklist
- Go/no-go decision criteria
- Live deployment phases
- Risk monitoring alerts
- Troubleshooting guide

---

## Configuration Cheatsheet

### Enable a Symbol

In `live_trading_runner.py`, line 315:

```python
symbols_config = {
    'ETHUSDT': {
        'enabled': True,  # ← Active
        'alpha_exposure': 2.0,  # ← 2x leverage in alpha regime
    },
    'BTCUSDT': {
        'enabled': False,  # ← Disabled
        'alpha_exposure': 1.0,  # ← (would be 1x if enabled)
    },
}
```

### Change Leverage

For ETH, increase leverage in alpha regime from 2.0x to 3.0x:

```python
'ETHUSDT': {
    'alpha_exposure': 3.0,  # ← More aggressive
}
```

⚠️ **Warning:** Higher leverage = higher risk. Backtest used 2.0x. Test carefully.

### Change Max Drawdown Limit

Stop if account loses more than 40% (instead of 30%):

```python
'ETHUSDT': {
    'max_drawdown_threshold': 0.40,  # ← 40% instead of 30%
}
```

### Change Position Size

Allow up to 10% of account per position (instead of 5%):

```python
'ETHUSDT': {
    'max_position_size_pct': 0.10,  # ← 10% max
}
```

---

## Monitoring Dashboard

### What to Check Daily

```
1. Regime Summary
   → ETHUSDT is currently in: [REGIME NAME]
   → Macro trend: UPTREND or DOWNTREND
   → Is it alpha? ⚡ or →

2. Trading Signal
   → Action: LONG (if alpha) or FLAT (if downtrend)
   → Exposure: 2.0x (if alpha), 1.0x (if normal), 0.0x (if downtrend)

3. Portfolio P&L
   → Unrealized P&L: +X% or -X%
   → Open positions: N

4. Risk Check
   → Max drawdown: -X% (if approaching -30%, be concerned)
   → Daily loss: -X% (if approaching -5%, close positions)
```

### Expected Patterns

**Good Week:**
- Alpha regime: 5-8 times (good signal frequency)
- Max DD: -15% to -25% (within tolerance)
- P&L: +2% to +5% (positive)
- Win rate: 50%+ (more wins than losses)

**Concerning Week:**
- Alpha regime: 0-2 times (too few signals)
- Max DD: -40%+ (breaching limits)
- P&L: -5%+ (losing money)
- Win rate: <40% (more losses than wins)

**Red Flag Week:**
- System crashes (data errors, API failures)
- Regime frequency: >5% (too many false signals)
- P&L: <-10% (large losses)
- Max DD: >-50% (breaching stop-loss)

---

## Troubleshooting

### "Regime frequency is too high (5% alpha signals)"

**Problem:** Too many false alpha signals

**Fix Option 1:** Increase volatility threshold
```python
# In RegimeDetectionEngine, find:
if volatility_percentile < 33:  # ← Change to 40
    is_low_vol = True
```

**Fix Option 2:** Require stronger trend
```python
if autocorr > 0.1:  # ← Change to 0.2
    is_trending = True
```

### "Max drawdown hit -70%, that's too much"

**Problem:** System is too aggressive

**Fix Option 1:** Reduce alpha exposure
```python
'ETHUSDT': {
    'alpha_exposure': 1.5,  # ← From 2.0x to 1.5x
}
```

**Fix Option 2:** Reduce max position size
```python
'ETHUSDT': {
    'max_position_size_pct': 0.03,  # ← From 5% to 3%
}
```

**Fix Option 3:** Stop losses at 40% DD
```python
'ETHUSDT': {
    'max_drawdown_threshold': 0.40,  # ← Close if down 40%
}
```

### "Live Sharpe is 0.1, but backtest is 1.25"

**Problem:** Live performance much worse than backtest

**Root Causes (in order of likelihood):**
1. Slippage (0.5% cost per trade) - normal
2. Regime detection lag (detect 1-2 candles late) - normal
3. Commission (0.1% per trade) - normal
4. Market conditions changed (regime edge disappeared) - problem

**Check:**
- Is regime frequency still 0.8-1.2%? (If yes, market OK)
- Is win rate still 40%+? (If yes, decisions are right)
- Is max DD similar to paper trading? (If yes, execution OK)

**Action:**
- If first week is <0.1 Sharpe: This is normal, give it 4 weeks
- If after 4 weeks still <0.2 Sharpe: Something is wrong, investigate

---

## Decision Tree

```
START
  ↓
Run: python3 live_trading_runner.py
  ↓
Does it execute without errors?
  ├─ NO → Check error message, debug
  └─ YES ↓
    ↓
Does it fetch ETHUSDT data?
  ├─ NO → Check internet, Binance API status
  └─ YES ↓
    ↓
Does regime detection work?
  ├─ NO → Check SYSTEM_ARCHITECTURE.md
  └─ YES ↓
    ↓
Ready for paper trading?
  ├─ YES → Run it hourly for 1 week (set up cron)
  └─ NO → Fix issues first
    ↓
After 1 week paper trading:
  ├─ Good metrics (freq 0.8-1.2%, DD <-50%)
  │   ├─ YES → Ready for live deployment (start $5k)
  │   └─ NO → Adjust parameters, run 2nd week
  ↓
Live trading started ($5k allocation)
  ↓
After 1 week live:
  ├─ Positive Sharpe? 
  │   ├─ YES → Scale to $25k
  │   └─ NO → Pause, investigate
  ↓
After 1 month live:
  ├─ Sharpe > 0.3?
  │   ├─ YES → Scale to $100k
  │   └─ NO → Roll off, find new edge
```

---

## Success Checklist

- [ ] Can run `python3 live_trading_runner.py` without errors
- [ ] System fetches live data from Binance
- [ ] Regime detection returns valid regime names
- [ ] Signals are calculated for enabled symbols
- [ ] Paper trading runs for 1 week
- [ ] Regime frequency is 0.5-2.0% (expected 0.8-1.2%)
- [ ] Max drawdown is reasonable (<-50%)
- [ ] No system crashes or data errors
- [ ] Ready for live with $5k allocation
- [ ] Live trading runs for 1 week with positive Sharpe
- [ ] Scale to $25k after 1-week live validation

---

## Support Resources

1. **System Architecture** → Read `SYSTEM_ARCHITECTURE.md`
2. **Deployment Steps** → Read `deployment_guide.py`
3. **Backtest Validation** → Run `extended_walk_forward_validator.py`
4. **Code Overview** → Review `live_trading_system_architecture.py`
5. **Data Pipeline** → Review `live_data_pipeline.py`

---

## Final Thoughts

✅ **You have:**
- Validated regime edge (Sharpe 1.25 on ETH)
- Universe-ready architecture
- Complete implementation
- Paper trading framework
- Deployment guide

🚀 **Next step:**
Run paper trading for 1 week, then go live with $5k

**Timeline:**
- Week 1: Paper trading
- Week 2: Validate metrics
- Week 3: Live with $5k
- Month 2: Scale to $25k (if positive)
- Month 3+: Full deployment

**Key metric to watch:**
Regime frequency should be ~1% of hourly candles. If it's 0% or >5%, the edge is not working.

Good luck! 🎯
