╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              🎛️ CAPITAL SYMBOL GOVERNOR — QUICK REFERENCE                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝


═══════════════════════════════════════════════════════════════════════════════════════

THE PROBLEM SOLVED

Before Governor:
  System discovers 50+ symbols
  → All get validated
  → All added to accepted symbols
  → $172 account tries to manage 50 concurrent positions
  → Risk exposure too high for bootstrap test
  ❌ Unmanageable

After Governor:
  System discovers 50+ symbols
  → All get validated
  → Governor says: "You have $172, can only safely trade 2"
  → 2 symbols selected, others discarded
  → $172 account manages 2 positions
  → Risk contained and manageable
  ✅ Safe


═══════════════════════════════════════════════════════════════════════════════════════

THE FOUR RULES AT A GLANCE

┌─────────────────────────────────────────────────────────────────────────┐
│ Rule 1: Capital Floor (Always Applied)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Equity        Cap                                                       │
│ ─────────────────────────                                              │
│ < $250        2 symbols                                                │
│ $250–$800     3 symbols                                                │
│ $800–$2000    4 symbols                                                │
│ $2000+        dynamic = max(2, floor(usable / min_trade))             │
│                                                                         │
│ Example: $172 equity → cap = 2                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Rule 2: API Health Guard (Dynamic)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Trigger: Binance returns RateLimit error (-1003, -1015, -1021)        │
│ Action:  cap = max(1, cap - 1)                                         │
│ Effect:  Reduces system load when API is throttling                    │
│ Auto:    Notified by MarketDataFeed._classify_error()                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Rule 3: Retrain Stability Guard (Dynamic)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Trigger: ML retrain skipped >2 cycles                                  │
│ Action:  cap = max(1, cap - 1)                                         │
│ Effect:  Reduces complexity when model isn't learning                  │
│ Manual:  Call governor.record_retrain_skip() / reset_retrain_skips()  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Rule 4: Drawdown Guard (Emergency)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Trigger: Current drawdown > 8% (configurable)                          │
│ Action:  cap = 1  (single symbol only)                                 │
│ Effect:  Forces defensive posture when account is losing               │
│ Auto:    Checked by governor._get_drawdown_pct()                       │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════

HOW IT INTEGRATES (5 Touch Points)

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. APP CONTEXT                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ During init:                                                            │
│   governor = CapitalSymbolGovernor(shared_state, config)               │
│   app.capital_symbol_governor = governor                                │
│                                                                         │
│ Files:                                                                  │
│   core/app_context.py (line 62, 1000, 3320)                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. SYMBOL MANAGER                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ During initialize_symbols():                                            │
│   cap = await governor.compute_symbol_cap()                             │
│   validated = validated[:cap]  # Apply cap                              │
│   await _safe_set_accepted_symbols(validated)                           │
│                                                                         │
│ Files:                                                                  │
│   core/symbol_manager.py (line 81, 98, 235)                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. MARKET DATA FEED                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ When RateLimit error detected:                                          │
│   if kind == "RateLimit":                                               │
│       governor.mark_api_rate_limited()                                  │
│                                                                         │
│ Files:                                                                  │
│   core/market_data_feed.py (line 406)                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 4. ML FORECASTER (Optional Manual)                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ When retrain skipped:                                                   │
│   governor.record_retrain_skip()                                        │
│                                                                         │
│ When retrain succeeds:                                                  │
│   governor.reset_retrain_skips()                                        │
│                                                                         │
│ Files:                                                                  │
│   agents/ml_forecaster.py (optional, not yet integrated)               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CAPITAL SYMBOL GOVERNOR (Core Logic)                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Main entry point:                                                       │
│   cap = await governor.compute_symbol_cap()                             │
│                                                                         │
│ Internal flow:                                                          │
│   1. Fetch equity from SharedState.balances                             │
│   2. Apply capital floor rule                                           │
│   3. Check API health flag                                              │
│   4. Check retrain skip count                                           │
│   5. Check drawdown percentage                                          │
│   6. Return minimum cap after all rules applied                         │
│                                                                         │
│ Files:                                                                  │
│   core/capital_symbol_governor.py (entire module)                      │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════

CONFIGURATION

Set in config.json, config object, or environment:

  Key                              Default    Meaning
  ───────────────────────────────────────────────────────────────────────
  MAX_EXPOSURE_RATIO               0.6        Fraction of equity usable
  MIN_ECONOMIC_TRADE_USDT          30         Minimum position size
  MAX_DRAWDOWN_PCT                 8.0        Trigger defensive mode
  MAX_RETRAIN_SKIPS                2          Reduce cap after N skips

Example override:
  # In config.json
  {
    "MAX_EXPOSURE_RATIO": 0.8,
    "MIN_ECONOMIC_TRADE_USDT": 50,
    "MAX_DRAWDOWN_PCT": 5.0
  }


═══════════════════════════════════════════════════════════════════════════════════════

EXPECTED SYSTEM BEHAVIOR

$172 USDT Account Bootstrap Scenario:

  1. System starts
     → AppContext creates governor + symbol_manager

  2. Symbol discovery runs
     → Finds 50+ symbols (BTCUSDT, ETHUSDT, BNBUSDT, ...)

  3. Symbol validation passes
     → All 50 symbols are valid

  4. Governor computes cap
     → Equity = $172
     → Capital floor rule: $172 < $250 → cap = 2
     → No rate limits yet
     → No retrain skips yet
     → No drawdown yet
     → Final cap: 2 symbols

  5. Symbols capped
     → accepted = [BTCUSDT, ETHUSDT]
     → Other 48 symbols discarded

  6. System operates
     → MarketDataFeed polls 2 symbols
     → MLForecaster scans 2 symbols
     → ExecutionManager can trade 2 symbols

  ✅ Bootstrap completes safely


═══════════════════════════════════════════════════════════════════════════════════════

MONITORING

Watch logs for:

  ✅ Normal startup:
     "🎛️ Capital Floor: equity=172.00 USDT → cap=2 symbols"
     "🎛️ Governor capped symbols: 2 (was 50)"

  ⚠️ Rate limit detected:
     "⚠️ API Rate Limited → reduce cap to 1"

  🛡️ Drawdown triggered:
     "🛡️ Drawdown 9.5% > 8% → DEFENSIVE (cap=1)"

  🚀 Recovery:
     "✅ API rate limit cleared"
     "✅ Retrain skip counter reset"


═══════════════════════════════════════════════════════════════════════════════════════

METHOD REFERENCE

Governor Public API:

  async def compute_symbol_cap() -> int
    Compute current cap based on all rules
    Returns: int (1-N symbols)

  async def _get_equity() -> float
    Fetch USDT equity from SharedState
    Returns: float (USDT amount)

  async def _get_drawdown_pct() -> Optional[float]
    Fetch current drawdown percentage
    Returns: float (e.g., 5.2) or None

  def _capital_floor_cap(equity: float) -> int
    Apply Rule 1: Capital Floor
    Returns: int (2-4 for typical accounts)

  def mark_api_rate_limited()
    Called by MarketDataFeed on RateLimit error
    Effect: Triggers Rule 2 on next compute_symbol_cap()

  def clear_api_rate_limit()
    Reset rate limit flag after cooldown
    Effect: Allows cap to increase again

  def record_retrain_skip()
    Track ML retrain skip
    Effect: Increments skip counter

  def reset_retrain_skips()
    Reset skip counter after successful retrain
    Effect: Allows cap to increase again


═══════════════════════════════════════════════════════════════════════════════════════

QUICK START

1. System automatically uses governor
   → No code changes needed in most cases

2. Monitor via logs
   → grep "Governor\|governor\|capital" logs/*.log

3. Override defaults (if needed)
   → Set config values before AppContext init

4. Track retrain (optional)
   → Call record_retrain_skip() / reset_retrain_skips()

5. That's it!
   → Governor runs automatically on every symbol discovery


═══════════════════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING

Q: Governor not limiting symbols?
A: Check logs for "Governor capped symbols"
   If not seen, governor may not be initialized
   Verify app_context.py has governor attribute

Q: Rate limit not triggering cap reduction?
A: Verify MarketDataFeed has proper error handling
   Check that _classify_error() is being called
   Ensure governor reference is available

Q: Drawdown guard not activating?
A: Verify SharedState has current_drawdown or metrics
   Check that governor._get_drawdown_pct() returns value
   May need to add drawdown tracking to shared_state

Q: Cap stuck at 1?
A: Multiple guards may be active simultaneously
   Check logs for which rule is active
   Call clear_api_rate_limit() or reset_retrain_skips() as needed


═══════════════════════════════════════════════════════════════════════════════════════

NEXT STEPS

Immediate:
  ☐ Run full system test with governor active
  ☐ Verify logs show symbol capping
  ☐ Check that bootstrap uses 2 symbols max

Short term:
  ☐ Monitor drawdown behavior during trading
  ☐ Test rate limit handling
  ☐ Validate accounting (symbols managed vs. traded)

Future enhancements:
  ☐ Dynamic symbol weighting (instead of hard cap)
  ☐ Time-based cap relaxation (after 24h profit)
  ☐ Performance-based scaling (unlock if Sharpe > 1.0)
  ☐ Volatility-aware position sizing

═══════════════════════════════════════════════════════════════════════════════════════
