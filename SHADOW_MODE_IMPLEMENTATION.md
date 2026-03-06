# Shadow Mode Implementation — Surgical, Non-Breaking Patches

## Overview

Shadow Mode enables virtual trading without touching real Binance balances. All canonical invariants preserved:
- Single order path: ExecutionManager → ExchangeClient only (conditionally)
- SharedState remains authoritative over virtual portfolio
- No agent/MetaController/RiskManager modifications
- HYG remains final execution gate
- All contracts unchanged
- 100% backward compatible

## Architecture Decision

Shadow Mode is implemented via **two surgical patches**:

1. **SharedState**: Add virtual portfolio ledger
2. **ExecutionManager**: Simulate fills instead of placing real orders
3. **Config**: Add `TRADING_MODE` with environment override

No other component changes. No architectural drift.

## Files Modified

```
core/shared_state.py          — ADD virtual portfolio fields + init method
core/execution_manager.py     — ADD shadow mode simulation in order placement
config/defaults.py or main   — ADD TRADING_MODE configuration
```

## Implementation Steps

### STEP 1: SharedState Enhancements

**Add to `SharedStateConfig`**:
```python
trading_mode: str = "live"  # "live" | "shadow"
virtual_balances: Dict[str, float] = {}
virtual_positions: Dict[str, Dict[str, Any]] = {}
virtual_realized_pnl: float = 0.0
virtual_unrealized_pnl: float = 0.0
virtual_nav: float = 0.0
```

**Add to `SharedState.__init__`**:
```python
# Virtual portfolio (shadow mode only)
self.virtual_balances: Dict[str, float] = {}
self.virtual_positions: Dict[str, Dict[str, Any]] = {}
self.virtual_realized_pnl = 0.0
self.virtual_unrealized_pnl = 0.0
self.virtual_nav = 0.0
```

**Add new method**:
```python
async def init_virtual_portfolio_from_real_snapshot(self):
    """
    Initialize virtual portfolio from real snapshot.
    Called once at boot if TRADING_MODE == "shadow".
    """
```

### STEP 2: ExecutionManager Order Placement

**In `_place_market_order_core()`, before calling `exchange_client.place_market_order()`**:

```python
if self._get_trading_mode() == "shadow":
    # Simulate fill
    return await self._simulate_fill(
        symbol=symbol,
        side=side,
        quantity=quantity,
        ref_price=current_price,
    )
else:
    # Real order (existing code path)
    return await self._place_with_client_id(...)
```

### STEP 3: Simulated Fill Engine

**Add new method to ExecutionManager**:
```python
async def _simulate_fill(
    self,
    symbol: str,
    side: str,
    quantity: float,
    ref_price: float,
) -> Dict[str, Any]:
    """
    Simulate realistic fill with slippage.
    Returns ExecResult contract exactly as live version.
    """
```

Rules:
- Use ref_price from MetaController
- Apply random slippage: uniform(-0.0002, 0.0002) = ±2 bps
- Compute fee using configured taker fee rate
- Return exact ExecResult structure

### STEP 4: Virtual Balance Updates

**In ExecutionManager, after successful fill (shadow)**:

On BUY:
```python
quote_after_fee = qty * fill_price * (1 + taker_fee_bps / 10000)
shared_state.virtual_balances[quote_asset] -= quote_after_fee
shared_state.virtual_positions[symbol].qty += qty
shared_state.virtual_positions[symbol].avg_price = recalculate(...)
```

On SELL:
```python
quote_after_fee = qty * fill_price * (1 - taker_fee_bps / 10000)
shared_state.virtual_balances[quote_asset] += quote_after_fee
shared_state.virtual_positions[symbol].qty -= qty
shared_state.virtual_realized_pnl += realized_pnl
```

Emit:
- `RealizedPnlUpdated`
- `PortfolioSnapshot` (virtual)

### STEP 5: Observability

**Modify SummaryLog event**:
```python
{
    component: "ExecutionManager",
    event: "trade_executed",
    symbol: "BTCUSDT",
    side: "BUY",
    qty: 0.01,
    price: 45000.00,
    tag: "...",
    status: "FILLED",
    mode: "shadow"  # ← NEW FIELD
}
```

### STEP 6: Live Switch Safety

**Add validation method**:
```python
def _can_switch_to_live_from_shadow(self) -> Tuple[bool, str]:
    """
    Validate safety criteria for shadow → live switch.
    Returns (allowed: bool, reason: str)
    """
    run_rate_24h = self.virtual_realized_pnl / max(1, elapsed_hours)
    if run_rate_24h < 15:
        return False, f"Run rate {run_rate_24h:.2f} < 15 USDT/h"
    
    max_dd = self._compute_max_drawdown()
    if max_dd > 0.10:
        return False, f"Max drawdown {max_dd*100:.1f}% > 10%"
    
    return True, "Safe to switch to live"
```

## Invariants Guaranteed

✅ No component above ExecutionManager knows shadow vs live
✅ RiskManager consulted for all orders (real & virtual)
✅ HYG remains final execution gate
✅ MetaController logic unchanged
✅ Agent logic unchanged
✅ MarketDataFeed real
✅ All contracts unchanged (ExecOrder, ExecResult, PortfolioSnapshot, HealthStatus)
✅ 100% backward compatible (shadow mode defaults to OFF)

## Testing Strategy

1. Unit test: `_simulate_fill()` returns valid ExecResult
2. Unit test: Virtual balances update correctly on BUY/SELL
3. Integration test: Shadow mode runs 1h without hitting real exchange
4. Safety test: Switch validation blocks unsafe transitions
5. Regression test: Live mode still works exactly as before

## Configuration

```python
# In config/defaults.py or environment
TRADING_MODE = os.getenv("TRADING_MODE", "live")  # "live" | "shadow"

# Safety thresholds for shadow → live
SHADOW_MIN_RUN_RATE_USDT_24H = 15
SHADOW_MAX_DRAWDOWN_PCT = 10
```

## Minimal Diff Summary

| Component | Lines Added | Lines Modified | Breaking Changes |
|-----------|-------------|----------------|------------------|
| SharedState | ~100 | ~20 | 0 |
| ExecutionManager | ~200 | ~10 | 0 |
| Config | ~5 | 0 | 0 |
| **Total** | **~305** | **~30** | **0** |

## Deployment Checklist

- [ ] Add TRADING_MODE config
- [ ] Add virtual_* fields to SharedState
- [ ] Implement init_virtual_portfolio_from_real_snapshot()
- [ ] Add _simulate_fill() to ExecutionManager
- [ ] Add _get_trading_mode() helper
- [ ] Modify _place_market_order_core() to check mode
- [ ] Update virtual balances on fill
- [ ] Add mode field to SummaryLog
- [ ] Implement _can_switch_to_live_from_shadow()
- [ ] Unit tests for simulation
- [ ] Integration test (1h shadow run)
- [ ] Regression test (live mode unchanged)
- [ ] Update ARCHITECTURE.md

## Implementation Order

1. Add config
2. Modify SharedState (add fields + init method)
3. Modify ExecutionManager (add simulation + mode check)
4. Add observability (SummaryLog mode field)
5. Test
6. Deploy to staging
7. Monitor 24h+ before switching to live
