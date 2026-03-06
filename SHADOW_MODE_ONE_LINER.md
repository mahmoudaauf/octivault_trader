# SHADOW MODE — ONE-LINER QUICK REFERENCE

## Activate Shadow Mode (Virtual Trading)
```bash
export TRADING_MODE=shadow && python3 launch_regime_trading.py --mode paper
```

## Return to Live Mode (Real Trading)
```bash
export TRADING_MODE=live && python3 launch_regime_trading.py --mode paper
```

## Check Current Mode
```bash
echo $TRADING_MODE
```

## Monitor Shadow Orders in Real-Time
```bash
tail -f *.log | grep -E "SHADOW|virtual|mode"
```

## Verify Configuration
```bash
grep -A 4 "trading_mode:" core/shared_state.py
```

## Test Compilation
```bash
python3 -m py_compile core/shared_state.py core/execution_manager.py && echo "✅ OK"
```

---

## What Happens When Shadow Mode Is Active

| Aspect | Shadow Mode | Live Mode |
|--------|-------------|-----------|
| Order Destination | Simulated locally | Sent to Binance |
| Real Balances | Never touched | Updated on execution |
| Risk Level | ZERO (virtual only) | REAL (capital at risk) |
| Order ID Format | `SHADOW-<uuid>` | Binance exchange ID |
| ExecResult.mode | `"shadow"` | `"live"` |
| Binance API Calls | None for fills | Yes, actual orders |
| PnL Tracking | Virtual ledger | Real portfolio |
| Time to Switch | Instant | Requires restart |

---

## Configuration Parameters (core/shared_state.py)

```python
class SharedStateConfig(BaseModel):
    # Trading Mode Selection
    trading_mode: str = "live"                    # ← Change to "shadow"
    
    # Shadow Mode Settings
    shadow_slippage_bps: float = 0.02             # ±2 basis points
    shadow_min_run_rate_usdt_24h: float = 15.0    # Minimum hourly rate
    shadow_max_drawdown_pct: float = 0.10         # 10% max drawdown
```

---

## 3-Minute Activation

```bash
# 1. Set shadow mode (30 seconds)
export TRADING_MODE=shadow

# 2. Start system (30 seconds)
python3 launch_regime_trading.py --mode paper

# 3. Verify in another terminal (30 seconds)
tail -f *.log | grep -i "shadow"

# Expected: Messages like "[SS:ShadowMode] Initialized virtual portfolio"
# Expected: "SHADOW-" order IDs (not real exchange IDs)
# Expected: Virtual balances updating in logs
```

---

## Verification Checklist

- [ ] `echo $TRADING_MODE` → outputs `shadow`
- [ ] Logs show `[SS:ShadowMode]` messages
- [ ] Orders have `SHADOW-` prefix (not real Binance IDs)
- [ ] ExecResult contains `"mode": "shadow"`
- [ ] Real Binance account balance unchanged
- [ ] Virtual portfolio NAV being tracked
- [ ] No real orders placed to Binance

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Shadow mode not activating | Check: `echo $TRADING_MODE` (should be "shadow") |
| Real orders still placed | Check: grep `"mode"` *.log (should see "shadow") |
| Virtual balances not updating | Check logs for `"virtual.*update"` messages |
| Compilation errors | Run: `python3 -m py_compile core/shared_state.py core/execution_manager.py` |
| Orders not simulated | Verify: `grep _get_trading_mode core/execution_manager.py` shows calls |

---

## Code Integration Points

**When TRADING_MODE=shadow:**

1. **ExecutionManager._get_trading_mode()** → Returns `"shadow"`
2. **ExecutionManager._place_with_client_id()** → Intercepts, simulates instead of sending to Binance
3. **ExecutionManager._simulate_fill()** → Creates realistic fill with slippage ±2 bps
4. **ExecutionManager._update_virtual_portfolio_on_fill()** → Updates virtual balances
5. **SharedState.virtual_balances** → Tracks simulated positions
6. **ExecResult.mode** → Set to `"shadow"` (no changes needed upstream)

**When TRADING_MODE=live:**

1. All above steps same, BUT
2. **ExecutionManager._place_with_client_id_live()** → Actual Binance API calls
3. **ExecResult.mode** → Set to `"live"`
4. Real orders placed to exchange, real capital at risk

---

## Full Deployment Path

```
Local Dev (30 min)
    ↓
    export TRADING_MODE=shadow
    Verify 5-10 orders simulated
    ↓
Staging (24+ hours)
    ↓
    Monitor virtual NAV, PnL, drawdown
    Real balances verify unchanged
    ↓
Production Cutover (monitored)
    ↓
    export TRADING_MODE=live
    Monitor first real order
    Close watch for 1 hour
    ↓
Ongoing Trading ✓
```

---

## Files Modified

- ✅ `core/shared_state.py` — Virtual portfolio fields + init method
- ✅ `core/execution_manager.py` — Shadow mode gate + simulation engine

## Documentation

- `SHADOW_MODE_ACTIVATION_QUICK_START.md` ← **YOU ARE HERE**
- `SHADOW_MODE_IMPLEMENTATION.md` — Architecture & design
- `SHADOW_MODE_CODE_PATCHES.md` — Exact code diffs
- `SHADOW_MODE_GUIDE.md` — Testing & deployment
- `SHADOW_MODE_SUMMARY.md` — Complete reference
- `00_SHADOW_MODE_INDEX.md` — Master index

---

**Status**: ✅ READY FOR IMMEDIATE ACTIVATION
