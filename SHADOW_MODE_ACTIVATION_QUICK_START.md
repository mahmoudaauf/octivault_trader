# Shadow Mode Activation Guide

## Overview

Shadow Mode is a **virtual trading mode** where all orders are simulated without touching real Binance balances. This allows 24+ hour testing before going live.

- **Default Mode**: `live` (real orders to Binance)
- **Shadow Mode**: `shadow` (virtual fills, no real capital risk)

---

## Method 1: Environment Variable (Fastest)

### Activate Shadow Mode
```bash
export TRADING_MODE=shadow
python3 launch_regime_trading.py --mode paper
```

### Deactivate Shadow Mode (Back to Live)
```bash
export TRADING_MODE=live
python3 launch_regime_trading.py --mode paper
```

### Verify Current Mode
```bash
echo $TRADING_MODE
```

---

## Method 2: Configuration File

Edit `core/shared_state.py` in `SharedStateConfig`:

```python
class SharedStateConfig(BaseModel):
    # ... other fields ...
    trading_mode: str = "shadow"  # Change "live" → "shadow"
    
    # Shadow mode configuration
    shadow_slippage_bps: float = 0.02           # ±2 basis points slippage
    shadow_min_run_rate_usdt_24h: float = 15.0  # Minimum hourly run rate
    shadow_max_drawdown_pct: float = 0.10       # Maximum 10% drawdown
```

---

## Method 3: Runtime Configuration (Through Meta Controller)

If you need to switch modes dynamically during runtime, modify through `MetaController`:

```python
# In your trading loop or signal handler
shared_state.trading_mode = "shadow"  # Switch to virtual
# OR
shared_state.trading_mode = "live"    # Switch to real
```

> ⚠️ **Warning**: Do NOT switch from "shadow" → "live" during active positions.

---

## How Shadow Mode Works

When activated (`TRADING_MODE=shadow`):

### 1. **Virtual Portfolio Initialization**
- Real balances copied to virtual ledger at boot
- Virtual positions initialized as empty
- Separate tracking of virtual PnL (real balances unchanged)

### 2. **Order Interception**
- All orders intercepted at `ExecutionManager._place_with_client_id()`
- Instead of sending to Binance, orders are simulated locally
- ExecResult format identical to real orders (no code changes upstream)

### 3. **Simulated Fills**
- Fill price = reference price ± random slippage (±2 bps default)
- Fee deduction applied (same as real orders)
- ExecResult returned with `mode: "shadow"` field

### 4. **Virtual Balance Updates**
- **On BUY**: USDT reduced, position qty increased
- **On SELL**: USDT increased, position qty decreased, PnL realized
- Virtual NAV recalculated after each fill

### 5. **Real Balances Unchanged**
- Real Binance balances: **NOT touched**
- Real positions: **NOT modified**
- Risk safety: **Maximum**

---

## Verification Checklist

### Before Starting Shadow Mode

```bash
# 1. Check current TRADING_MODE
echo "TRADING_MODE=$TRADING_MODE"

# 2. Verify compilation
python3 -m py_compile core/shared_state.py core/execution_manager.py

# 3. Review virtual portfolio config
grep -A 3 "trading_mode" core/shared_state.py | head -10
```

### After Activating Shadow Mode

```bash
# Start system with TRADING_MODE=shadow
export TRADING_MODE=shadow
python3 launch_regime_trading.py --mode paper

# Monitor logs for shadow mode indicators
tail -f *.log | grep -i "shadow\|virtual\|mode"

# Watch for messages like:
# [SS:ShadowMode] Initialized virtual portfolio from real snapshot
# [EM:ShadowMode] Order intercepted: SHADOW-<uuid>
# [EM:ShadowMode] Virtual portfolio updated
```

### Verify No Real Orders Placed

```bash
# Check Binance account (should be unchanged)
# OR check trading logs - should see "mode": "shadow" in ExecResult

# Real orders would appear as:
# "exchange_order_id": <actual Binance order ID>
# "mode": "live"

# Virtual orders appear as:
# "exchange_order_id": "SHADOW-<uuid>"
# "mode": "shadow"
```

---

## Configuration Parameters

### `trading_mode` (required)
- **Type**: `str`
- **Values**: `"live"` or `"shadow"`
- **Default**: `"live"`
- **Purpose**: Controls order routing (real vs virtual)

### `shadow_slippage_bps` (shadow mode only)
- **Type**: `float`
- **Default**: `0.02` (±2 basis points)
- **Range**: `0.0` to `0.5` (realistic for market orders)
- **Purpose**: Simulates realistic execution slippage

### `shadow_min_run_rate_usdt_24h` (shadow mode only)
- **Type**: `float`
- **Default**: `15.0` (minimum $15/hour profit)
- **Purpose**: Gate for shadow → live transition (optional check)

### `shadow_max_drawdown_pct` (shadow mode only)
- **Type**: `float`
- **Default**: `0.10` (10% maximum)
- **Purpose**: Safety bound for virtual losses in shadow mode

---

## Deployment Stages

### STAGE 1: Local Development (Laptop)
```bash
export TRADING_MODE=shadow
python3 launch_regime_trading.py --mode paper
# Test 30 minutes locally
```

### STAGE 2: Staging (24+ hours)
```bash
export TRADING_MODE=shadow
python3 launch_regime_trading.py --mode paper
# Monitor virtual NAV, trade counts, realized PnL
# Run for 24+ hours with real market data
# Verify no real Binance orders placed
```

### STAGE 3: Production Cutover
```bash
export TRADING_MODE=live
python3 launch_regime_trading.py --mode paper
# First real order should execute
# Monitor for 1 hour closely
# Then resume normal operations
```

---

## Troubleshooting

### Issue: Shadow mode not activating

**Check 1**: Environment variable not set
```bash
# Verify TRADING_MODE is set
echo $TRADING_MODE
# Should output: shadow

# If blank, set it:
export TRADING_MODE=shadow
```

**Check 2**: Configuration override in code
```python
# Check SharedState initialization
# Make sure config.trading_mode = "shadow"
grep -n "self.trading_mode" core/shared_state.py
```

**Check 3**: ExecutionManager checking wrong mode
```python
# Verify _get_trading_mode() is called
grep -n "_get_trading_mode" core/execution_manager.py
# Should show multiple references
```

---

### Issue: Real orders still being placed

**Check 1**: Verify shadow mode gate in ExecutionManager
```bash
# Look at _place_with_client_id() method
grep -A 10 "def _place_with_client_id" core/execution_manager.py | grep -i shadow
```

**Check 2**: Confirm TRADING_MODE=shadow in logs
```bash
tail -f *.log | grep -i "trading.*mode\|TRADING_MODE"
```

**Check 3**: Check ExecResult for "mode" field
```python
# Should see: "mode": "shadow"
# Not: "mode": "live"
```

---

### Issue: Virtual balances not updating

**Check 1**: Virtual portfolio initialized
```bash
tail -f *.log | grep -i "virtual portfolio.*init"
```

**Check 2**: Confirm _update_virtual_portfolio_on_fill called
```bash
tail -f *.log | grep -i "virtual.*update\|shadow.*fill"
```

**Check 3**: Check SharedState.virtual_balances
```python
# Verify balances exist:
print(shared_state.virtual_balances)
# Should show: {"USDT": <amount>, "BTC": <amount>, ...}
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Activate Shadow Mode | `export TRADING_MODE=shadow` |
| Deactivate Shadow Mode | `export TRADING_MODE=live` |
| Check Current Mode | `echo $TRADING_MODE` |
| View Virtual Balances | Check logs for "virtual_nav" |
| Monitor Shadow Orders | `tail -f *.log \| grep -i "SHADOW"` |
| Verify Config | `grep trading_mode core/shared_state.py` |
| Test Compilation | `python3 -m py_compile core/*.py` |

---

## Key Features

✅ **Virtual Orders**: All orders simulated with realistic slippage  
✅ **Real Balances Protected**: Real Binance balances never touched  
✅ **Transparent**: Upper layers see identical ExecResult format  
✅ **Traceable**: Orders marked with `SHADOW-<uuid>` for audit  
✅ **Safe Switching**: Default is "live" (opt-in for shadow)  
✅ **PnL Tracking**: Virtual realized/unrealized PnL calculated  
✅ **Backward Compatible**: Zero breaking changes  

---

## Safety Guarantees

- ✅ Real capital never at risk in shadow mode
- ✅ No Binance API calls for order placement (simulated only)
- ✅ All invariants preserved (RiskManager, HYG, MetaController unchanged)
- ✅ Default safe ("live" mode if TRADING_MODE not set)
- ✅ Clear audit trail (ExecResult.mode field indicates shadow vs live)
- ✅ No persistent state changes (purely virtual ledger)

---

## Next Steps

1. **Set Environment Variable**
   ```bash
   export TRADING_MODE=shadow
   ```

2. **Start Trading System**
   ```bash
   python3 launch_regime_trading.py --mode paper
   ```

3. **Monitor Logs**
   ```bash
   tail -f *.log | grep -i "shadow\|virtual"
   ```

4. **Verify Virtual Orders**
   - Check for "SHADOW-" order IDs in logs
   - Verify real Binance orders NOT placed
   - Monitor virtual NAV growth

5. **Run 24+ Hour Test**
   - Monitor performance metrics
   - Collect data on slippage, PnL, drawdown
   - Validate strategy behavior

6. **Switch to Live** (when ready)
   ```bash
   export TRADING_MODE=live
   ```

---

## Support

For questions or issues:
- Check logs: `tail -f *.log`
- Review code: See `SHADOW_MODE_CODE_PATCHES.md` for implementation details
- Test manually: Use examples in `SHADOW_MODE_GUIDE.md`
- Reference: Check `SHADOW_MODE_SUMMARY.md` for complete API

---

**Shadow Mode Status**: ✅ READY  
**Last Updated**: 2026-03-02  
**Version**: P9 Implementation
