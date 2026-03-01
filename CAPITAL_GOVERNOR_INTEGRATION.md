"""
Capital & Symbol Governor — Integration Guide

This document explains how the CapitalSymbolGovernor integrates with the trading system
to ensure bootstrap trades don't overextend a small account through economic constraints.

═══════════════════════════════════════════════════════════════════════════════════════

ARCHITECTURE OVERVIEW

Component Placement:
  AppContext (orchestrator)
    ├── Creates CapitalSymbolGovernor (initialization)
    ├── Creates SymbolManager (initialization)
    ├── Wires CapitalSymbolGovernor → SymbolManager._app
    │
    └── At Runtime:
        ├── SymbolManager.initialize_symbols()
        │   ├── Discovers symbols via agents
        │   ├── Validates symbol format and viability
        │   ├── Calls: await governor.compute_symbol_cap()
        │   ├── Caps validated symbols: symbols = symbols[:symbol_cap]
        │   └── Finalizes accepted symbols in SharedState
        │
        └── MarketDataFeed (monitors API health)
            ├── Detects RateLimit (-1003, -1015, -1021)
            ├── Calls: governor.mark_api_rate_limited()
            └── Governor reduces cap on next compute_symbol_cap()

═══════════════════════════════════════════════════════════════════════════════════════

KEY RULES

Rule 1 — Capital Floor Mapping
  Equity      Symbol Cap
  ─────────────────────────
  < 250       2 symbols
  250–800     3 symbols
  800–2000    4 symbols
  2000+       dynamic (max(2, floor(usable / min_trade)))

Rule 2 — API Health Guard
  IF: RateLimit detected (ClassifyError returns "RateLimit")
  THEN: symbol_cap = max(1, symbol_cap - 1)
  EFFECT: Reduces load when API is stressed

Rule 3 — Retrain Stability Guard
  IF: ML retrain skipped > 2 cycles in a row
  THEN: symbol_cap = max(1, symbol_cap - 1)
  EFFECT: Reduces complexity when model is unstable

Rule 4 — Drawdown Guard
  IF: current_drawdown > 8% (or MAX_DRAWDOWN_PCT config)
  THEN: symbol_cap = 1
  EFFECT: Go defensive when account is losing

═══════════════════════════════════════════════════════════════════════════════════════

FILES MODIFIED & CREATED

[NEW] core/capital_symbol_governor.py (198 lines)
  Class CapitalSymbolGovernor:
    • compute_symbol_cap() — Main entry point, applies all rules
    • _capital_floor_cap(equity) — Rule 1: Static tier mapping
    • _get_equity() — Fetch USDT balance from SharedState
    • _get_drawdown_pct() — Rule 4: Fetch current drawdown
    • mark_api_rate_limited() — Rule 2: Called by MarketDataFeed
    • record_retrain_skip() — Rule 3: Called by MLForecaster
    • reset_retrain_skips() — Reset counter after successful retrain

[MODIFIED] core/app_context.py
  Changes:
    Line ~62: Added import: _governor_mod = _import_strict("core.capital_symbol_governor")
    Line ~1000: Added attribute: self.capital_symbol_governor: Optional[Any] = None
    Line ~3320: Added instantiation:
      CapitalSymbolGovernor = _get_cls(_governor_mod, "CapitalSymbolGovernor")
      self.capital_symbol_governor = _try_construct(...)

[MODIFIED] core/symbol_manager.py
  Changes:
    Line ~81: Added parameter to __init__: app: Optional[Any] = None
    Line ~98: Store app reference: self._app = app
    Line ~235: Added governor integration in initialize_symbols():
      symbol_cap = await governor.compute_symbol_cap()
      validated_list = validated_list[:symbol_cap]  # Apply cap

[MODIFIED] core/market_data_feed.py
  Changes:
    Line ~406: Added rate limit notification:
      if kind == "RateLimit":
          app.capital_symbol_governor.mark_api_rate_limited()

═══════════════════════════════════════════════════════════════════════════════════════

USAGE PATTERNS

Pattern 1 — Symbol Discovery (Automatic)
  1. AppContext initializes CapitalSymbolGovernor
  2. AppContext initializes SymbolManager with app=self
  3. SymbolManager.initialize_symbols() called
  4. Governor computes cap based on current equity
  5. Symbols capped and finalized

Pattern 2 — Rate Limit Detection (Automatic)
  1. MarketDataFeed.get_ohlcv() or similar hits rate limit
  2. _classify_error() detects code -1003, -1015, or -1021
  3. Calls governor.mark_api_rate_limited()
  4. Next symbol discovery uses reduced cap

Pattern 3 — Manual Retrain Tracking (Optional)
  In MLForecaster or similar:
    if depth_not_loading:
      governor.record_retrain_skip()
    else:
      governor.reset_retrain_skips()

═══════════════════════════════════════════════════════════════════════════════════════

CONFIGURATION

In config.json or via environment:

  # Economic constraints
  "MAX_EXPOSURE_RATIO": 0.6,          # Fraction of equity to use
  "MIN_ECONOMIC_TRADE_USDT": 30,      # Minimum position size

  # Health thresholds
  "MAX_DRAWDOWN_PCT": 8.0,            # Trigger defensive mode
  "MAX_RETRAIN_SKIPS": 2,             # Reduce cap after this many skips

Example:
  equity = 172 USDT
  exposure = 0.6 → 103.2 USDT usable
  min_trade = 30 USDT
  raw_cap = floor(103.2 / 30) = 3

  But equity < 250, so capital_floor_cap = 2
  → Final cap = 2 symbols

═══════════════════════════════════════════════════════════════════════════════════════

BOOTSTRAP FLOW EXAMPLE (172 USDT Account)

1. System starts
   ├── AppContext.__init__()
   │   ├── Creates SharedState
   │   ├── Creates CapitalSymbolGovernor(shared_state, config)
   │   └── Creates SymbolManager(shared_state, config, app=self)
   │
2. Discovery phase begins
   ├── SymbolDiscoverer agents scan Binance
   ├── Find 50 symbols (BTCUSDT, ETHUSDT, BNBUSDT, ..., LTOUSDT)
   │
3. Symbol validation
   ├── Format check ✓
   ├── Liquidity check ✓
   ├── Blacklist check ✓
   └── All 50 symbols pass → validated = {50 symbols}
   │
4. Governor applies cap
   ├── Current equity = 172 USDT
   ├── Capital floor rule: 172 < 250 → cap = 2
   ├── API health: no rate limit yet → cap stays 2
   ├── Retrain skips: none yet → cap stays 2
   ├── Drawdown: 0% → cap stays 2
   │
5. Final cap applied
   ├── validated symbols = [BTCUSDT, ETHUSDT]  (first 2 only)
   └── SharedState.set_accepted_symbols({BTCUSDT, ETHUSDT})
   │
6. System continues with 2 symbols
   ├── MarketDataFeed polls both
   ├── MLForecaster scans both
   ├── ExecutionManager executes for both
   └── Bootstrap completes with manageable position count

═══════════════════════════════════════════════════════════════════════════════════════

ADVANCED: FUTURE ENHANCEMENTS

Option 1 — Dynamic Symbol Weighting
  Instead of hard cap, assign weights:
    • 2 core symbols (always): weight=1.0
    • 1 alpha slot (if capital allows): weight=0.5
    • Symbols ranked by volatility/momentum

Option 2 — Time-Based Relaxation
  After 24 hours without drawdown:
    • Automatically increase cap by 1
    • Gradually re-expose to more symbols

Option 3 — Performance-Based Scaling
  If Sharpe > 0.8 for 100 trades:
    • Unlock next tier of symbols
    • Scale positions linearly with performance

Option 4 — Volatility-Aware Sizing
  In low-vol environments:
    • Increase position size
    • Reduce position count (stay within cap)
  In high-vol environments:
    • Decrease position size
    • Maintain tighter risk envelope

═══════════════════════════════════════════════════════════════════════════════════════

MONITORING & DEBUGGING

Logs to watch:

  ✅ Normal startup:
    [CapitalSymbolGovernor] 🎛️ Capital Floor: equity=172.00 USDT → cap=2 symbols
    [SymbolManager] 🎛️ Governor capped symbols: 2 (was 50)

  ⚠️ Rate limit detected:
    [MarketDataFeed] RateLimit error detected
    [CapitalSymbolGovernor] API rate limit flagged for governor
    [CapitalSymbolGovernor] ⚠️ API Rate Limited → reduce cap to 1

  🛡️ Drawdown trigger:
    [CapitalSymbolGovernor] 🛡️ Drawdown 9.5% > 8% → DEFENSIVE (cap=1)

  🚀 Recovery:
    [CapitalSymbolGovernor] ✅ API rate limit cleared
    [CapitalSymbolGovernor] ✅ Retrain skip counter reset

═══════════════════════════════════════════════════════════════════════════════════════

TESTING

Unit test example:

  async def test_capital_floor_cap():
      gov = CapitalSymbolGovernor(shared_state=None, config=None)
      assert gov._capital_floor_cap(100) == 2
      assert gov._capital_floor_cap(500) == 3
      assert gov._capital_floor_cap(1500) == 4

  async def test_api_rate_limit():
      gov = CapitalSymbolGovernor(shared_state=None, config=None)
      cap = await gov.compute_symbol_cap()  # returns 2 (no equity)
      gov.mark_api_rate_limited()
      cap_reduced = await gov.compute_symbol_cap()
      assert cap_reduced < cap

  async def test_drawdown_guard():
      ss = MockSharedState()
      ss.current_drawdown = 10.0  # 10% > 8%
      gov = CapitalSymbolGovernor(shared_state=ss, config=None)
      cap = await gov.compute_symbol_cap()
      assert cap == 1  # Defensive mode

═══════════════════════════════════════════════════════════════════════════════════════
"""
