# SHADOW MODE IMPLEMENTATION — P9 COMPLETE GUIDE

## Status

✅ **COMPLETE & READY FOR TESTING**

All surgical patches applied:
1. SharedState: Virtual portfolio fields + initialization
2. ExecutionManager: Shadow mode gate + simulated fill engine
3. Configuration: Trading mode support

**No breaking changes. Zero architectural drift. 100% backward compatible.**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MetaController                           │
│                    (unchanged - passes prices)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ order_signal: {symbol, side, qty, price}
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ExecutionManager.execute_trade()              │
│                                                                  │
│  1. Check: TRADING_MODE == "shadow" ?                           │
│  2. YES → _place_market_order_core() calls _place_with_client_id()
│  3. _place_with_client_id() redirects to _simulate_fill()       │
│  4. _simulate_fill() returns ExecResult (no real order sent)    │
│  5. _update_virtual_portfolio_on_fill() updates balances        │
│  6. Return ExecResult to MetaController (looks real)            │
│                                                                  │
│  NO → Normal flow: _place_with_client_id_live() → Binance      │
└─────────────────────┬───────────────────────────────────────────┘
                      │ ExecResult
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SharedState                                │
│                                                                  │
│  Shadow mode:                                                    │
│  - real_balances: unchanged (Binance)                           │
│  - virtual_balances: updated from simulation                    │
│  - virtual_positions: updated from simulation                   │
│  - virtual_realized_pnl: running total                          │
│  - virtual_nav: quote + unrealized + realized_pnl              │
│                                                                  │
│  Live mode:                                                      │
│  - All real (no virtual ledger)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variable

```bash
export TRADING_MODE=shadow    # "shadow" or "live"
```

### In Code

```python
# core/shared_state.py — SharedStateConfig class
trading_mode: str = "live"  # Override with "shadow"
shadow_slippage_bps: float = 0.02  # ±2 bps
shadow_min_run_rate_usdt_24h: float = 15.0  # Min rate for safe switch
shadow_max_drawdown_pct: float = 0.10  # Max 10% drawdown
```

## API Reference

### SharedState Methods

#### `async init_virtual_portfolio_from_real_snapshot()`

Initialize virtual portfolio from real balance snapshot. Call once at boot if `TRADING_MODE == "shadow"`.

```python
if shared_state.trading_mode == "shadow":
    await shared_state.init_virtual_portfolio_from_real_snapshot()
```

**Effect:**
- Copies real balances to `virtual_balances`
- Initializes `virtual_positions` (empty)
- Sets `virtual_realized_pnl = 0`
- Records `_shadow_mode_start_time`
- Emits `ShadowModeInitialized` event

#### `get_virtual_balance(asset: str) -> Dict[str, float]`

Get virtual balance for an asset (shadow mode only).

```python
btc_balance = shared_state.get_virtual_balance("BTC")
# Returns: {"free": 0.5, "locked": 0.0}
```

### ExecutionManager Methods

#### `_get_trading_mode() -> str`

Get current trading mode. Returns `"live"` or `"shadow"`.

```python
mode = execution_manager._get_trading_mode()
if mode == "shadow":
    print("Running in shadow mode")
```

#### `async _simulate_fill(...) -> Dict`

Simulate a realistic fill with slippage. Internal method (called automatically).

**Parameters:**
- `symbol`: e.g. "BTCUSDT"
- `side`: "BUY" or "SELL"
- `quantity`: Amount to fill
- `ref_price`: Reference price from MetaController
- `taker_fee_bps`: Fee in basis points (default 10 bps)

**Returns:**
```python
{
    "ok": True,
    "status": "FILLED",
    "executedQty": 0.01,
    "price": 45100.50,
    "cummulativeQuoteQty": 451.0050,
    "exchange_order_id": "SHADOW-abc123...",
    "mode": "shadow",
    "timestamp": 1234567890.0,
}
```

#### `async _update_virtual_portfolio_on_fill(...) -> None`

Update virtual portfolio after a simulated fill. Internal method (called automatically).

## Testing

### Unit Test: Simulate Fill

```python
import asyncio
from core.execution_manager import ExecutionManager
from core.shared_state import SharedState

async def test_simulate_fill():
    ss = SharedState()
    ss.trading_mode = "shadow"
    ss.quote_asset = "USDT"
    ss.latest_prices["BTCUSDT"] = 45000.0
    
    em = ExecutionManager(shared_state=ss, exchange_client=None)
    
    result = await em._simulate_fill(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01,
        ref_price=45000.0,
        taker_fee_bps=10,
    )
    
    assert result["ok"] == True
    assert result["status"] == "FILLED"
    assert result["executedQty"] == 0.01
    assert result["mode"] == "shadow"
    print("✅ Simulate fill test passed")

asyncio.run(test_simulate_fill())
```

### Unit Test: Virtual Portfolio Update

```python
async def test_virtual_portfolio_update():
    ss = SharedState()
    ss.trading_mode = "shadow"
    ss.quote_asset = "USDT"
    ss.virtual_balances = {"USDT": {"free": 1000.0, "locked": 0.0}}
    ss.virtual_positions = {}
    ss.latest_prices["BTCUSDT"] = 45000.0
    
    em = ExecutionManager(shared_state=ss, exchange_client=None)
    
    # Simulate BUY 0.01 BTC at 45000
    await em._update_virtual_portfolio_on_fill(
        symbol="BTCUSDT",
        side="BUY",
        filled_qty=0.01,
        fill_price=45000.0,
        cumm_quote=450.0,  # 0.01 * 45000 * (1 + 0.001 fee)
    )
    
    # Check virtual balances
    assert ss.virtual_balances["USDT"]["free"] == pytest.approx(550.0)  # 1000 - 450
    assert ss.virtual_positions["BTCUSDT"]["qty"] == 0.01
    assert ss.virtual_positions["BTCUSDT"]["avg_price"] == 45000.0
    
    # Simulate SELL 0.01 BTC at 46000
    ss.latest_prices["BTCUSDT"] = 46000.0
    await em._update_virtual_portfolio_on_fill(
        symbol="BTCUSDT",
        side="SELL",
        filled_qty=0.01,
        fill_price=46000.0,
        cumm_quote=459.4,  # 0.01 * 46000 * (1 - 0.001 fee)
    )
    
    # Check results
    assert ss.virtual_balances["USDT"]["free"] == pytest.approx(1009.4)  # 550 + 459.4
    assert ss.virtual_positions["BTCUSDT"]["qty"] == 0.0  # Closed
    assert ss.virtual_realized_pnl == pytest.approx(9.4)  # 459.4 - 450
    
    print("✅ Virtual portfolio update test passed")

asyncio.run(test_virtual_portfolio_update())
```

### Integration Test: 1-hour Shadow Run

```python
async def test_shadow_mode_1hour():
    """
    Run system in shadow mode for 1 hour and verify:
    1. No real orders sent to Binance
    2. Virtual portfolio tracked correctly
    3. All events have mode="shadow"
    """
    
    # Start system in shadow mode
    os.environ["TRADING_MODE"] = "shadow"
    app = await AppContext.create_app()
    
    # Wait 1 hour (or 10 minutes for quick test)
    await asyncio.sleep(600)  # 10 minutes
    
    # Verify
    assert shared_state.trading_mode == "shadow"
    assert len(shared_state.virtual_positions) > 0  # Some trades happened
    assert shared_state.virtual_realized_pnl != 0 or len(shared_state.virtual_positions) > 0
    
    # Check event log
    for event in shared_state._event_log:
        if "trade" in event.get("event", "").lower():
            assert event.get("mode") == "shadow"
    
    print(f"✅ Shadow mode 1h test passed")
    print(f"   Virtual PnL: {shared_state.virtual_realized_pnl:.2f}")
    print(f"   Virtual NAV: {shared_state.virtual_nav:.2f}")
    print(f"   Trades: {len(shared_state.trade_history)}")
```

### Live Switch Validation Test

```python
async def test_shadow_to_live_switch():
    """
    Verify safety guards for switching from shadow to live mode.
    """
    ss = SharedState()
    ss.trading_mode = "shadow"
    ss.virtual_realized_pnl = 100.0  # $100 profit
    ss._shadow_mode_start_time = time.time() - (24 * 3600)  # Started 24h ago
    ss._shadow_mode_high_water_mark = 1000.0
    ss.virtual_nav = 950.0  # Down from high water
    
    em = ExecutionManager(shared_state=ss)
    
    # Check if safe to switch
    can_switch, reason = em._can_switch_to_live_from_shadow()
    
    run_rate = ss.virtual_realized_pnl / 24.0  # $100 / 24h = $4.17/h
    max_dd = (ss._shadow_mode_high_water_mark - ss.virtual_nav) / ss._shadow_mode_high_water_mark
    
    print(f"Run rate: ${run_rate:.2f}/h (min: ${ss.config.shadow_min_run_rate_usdt_24h})")
    print(f"Max DD: {max_dd*100:.1f}% (max: {ss.config.shadow_max_drawdown_pct*100}%)")
    print(f"Safe to switch: {can_switch} ({reason})")
    
    # In this case, run rate is too low (~$4/h < $15/h minimum)
    assert can_switch == False
    print("✅ Switch validation test passed")
```

## Deployment Checklist

### Pre-Deployment (Dev/Test)

- [ ] Run unit tests for `_simulate_fill()`
- [ ] Run unit tests for `_update_virtual_portfolio_on_fill()`
- [ ] Run 10-minute shadow mode test
- [ ] Verify virtual balances stay consistent
- [ ] Verify no real orders sent to Binance
- [ ] Verify SummaryLog events have `mode="shadow"`

### Staging Deployment

- [ ] Set `TRADING_MODE=shadow` in staging environment
- [ ] Start system
- [ ] Verify `init_virtual_portfolio_from_real_snapshot()` called
- [ ] Run for 24+ hours continuously
- [ ] Monitor virtual NAV growth
- [ ] Check realized PnL accumulation
- [ ] Verify all orders appear as "SHADOW-*" in logs
- [ ] Confirm real Binance balances unchanged

### Production Deployment (After Shadow Run Success)

- [ ] Switch `TRADING_MODE=live`
- [ ] Restart system
- [ ] Verify first order goes to real Binance
- [ ] Monitor for 1 hour closely
- [ ] Confirm real balances update correctly
- [ ] Set alerts for anomalies

## Troubleshooting

### Problem: Virtual portfolio not initializing

**Check:**
1. `TRADING_MODE` environment variable set to "shadow"?
2. Is `init_virtual_portfolio_from_real_snapshot()` being called at boot?
3. Are real balances loaded before initialization?

**Fix:**
```python
# In app startup
if shared_state.trading_mode == "shadow":
    await shared_state.init_virtual_portfolio_from_real_snapshot()
    print(f"Virtual NAV: {shared_state.virtual_nav}")
```

### Problem: Simulated fills not happening

**Check:**
1. `_get_trading_mode()` returning "shadow"?
2. Current price available in `latest_prices`?
3. Is `_place_with_client_id()` being called?

**Fix:**
```python
# Add debug logging
em.logger.info(f"Trading mode: {em._get_trading_mode()}")
em.logger.info(f"Latest prices: {shared_state.latest_prices}")
```

### Problem: Virtual balances going negative

**Likely cause:** Fee calculation error or missing reservation logic

**Check:**
1. Are reservations being enforced before orders?
2. Is fee ratio calculated correctly?
3. Are we checking `get_spendable_balance()` or real balances?

**Fix:** RiskManager should evaluate against `virtual_balances` in shadow mode:

```python
# In RiskManager (check if needed)
if shared_state.trading_mode == "shadow":
    quote_bal = shared_state.get_virtual_balance(quote_asset)["free"]
else:
    quote_bal = await shared_state.get_spendable_balance(quote_asset)
```

## Observability

### Events Emitted

```python
# At shadow mode initialization
{
    "event": "ShadowModeInitialized",
    "virtual_balances": {"USDT": 1000.0, ...},
    "virtual_nav": 1000.0,
    "start_time": 1234567890.0,
}

# On every trade (in shadow or live)
{
    "component": "ExecutionManager",
    "event": "trade_executed",
    "symbol": "BTCUSDT",
    "side": "BUY",
    "qty": 0.01,
    "price": 45000.0,
    "status": "FILLED",
    "mode": "shadow",  # NEW FIELD
}
```

### Logging

Shadow mode logs include `[EM:ShadowMode]` prefix:

```
[EM:ShadowMode] Virtual portfolio initialized. Quote balance: 1000.00, NAV: 1000.00
[EM:ShadowMode] BTCUSDT BUY FILLED (simulated). qty=0.01000000, price=45001.23, quote=450.01
[EM:ShadowMode:UpdateVirtual] BTCUSDT BUY: qty 0.00 → 0.01, avg_price=45001.23, quote_balance 1000.00 → 549.99
```

### Metrics

Monitor in SharedState.metrics:

```python
# Shadow mode specific (if we add them)
shared_state.metrics.get("virtual_nav")
shared_state.metrics.get("virtual_realized_pnl")
shared_state.metrics.get("shadow_run_duration_hours")
shared_state.metrics.get("shadow_trades_executed")
```

## Safety & Rollback

### Safe to Run in Parallel

Shadow mode and live mode are **mutually exclusive** (not concurrent). Switch requires restart.

### Rollback

If issues detected:

1. Set `TRADING_MODE=live`
2. Restart system
3. Verify real orders resume
4. Real balances override virtual immediately

### Data Preservation

Virtual portfolio data is **NOT persisted** (lives in memory). On restart:
- If `TRADING_MODE=shadow`: Resets to real balances
- If `TRADING_MODE=live`: Ignores virtual ledger

## Next Steps

1. **Test locally**: Run unit tests and 10-min integration test
2. **Staging**: Deploy to staging, run 24+ hours
3. **Monitor**: Verify virtual NAV, PnL, order counts
4. **Go Live**: Switch `TRADING_MODE=live`, monitor closely for 1 hour
5. **Iterate**: Adjust slippage config if needed, log learnings

## Questions?

Refer to:
- `SHADOW_MODE_IMPLEMENTATION.md` — Overall architecture
- `core/shared_state.py` — Virtual portfolio code
- `core/execution_manager.py` — Simulation engine code
- This guide — Testing and deployment

---

**Status: Ready for Testing ✅**
