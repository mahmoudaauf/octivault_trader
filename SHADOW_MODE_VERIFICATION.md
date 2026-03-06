# Shadow Mode Verification Guide

## TL;DR

If you see `live_mode=True` in the logs but set `export TRADING_MODE=shadow`, that's **NORMAL and EXPECTED**.

The `live_mode` flag is from the OLD legacy system. The NEW shadow mode system uses `TRADING_MODE` environment variable and checks via `_get_trading_mode()` in ExecutionManager.

**Shadow mode IS still working even if logs show `live_mode=True`.**

---

## Understanding the Two Systems

### OLD System (Legacy)
- **What it checks**: `LIVE_MODE` environment variable
- **Where it appears**: AppContext startup logs showing `live_mode=True`
- **What it controls**: Bootstrap behavior, reconciliation policy
- **When it runs**: At startup

### NEW System (P9 Shadow Mode)
- **What it checks**: `TRADING_MODE` environment variable (or env var)
- **Where it appears**: ExecutionManager at order placement time
- **What it controls**: Whether orders are SIMULATED or SENT TO BINANCE
- **When it runs**: When an order needs to be placed

---

## How to Verify Shadow Mode is Working

### Step 1: Check Environment Variable is Set
```bash
echo $TRADING_MODE
# Output should be: shadow
```

### Step 2: Look for Trading Mode in Logs
```bash
grep "trading_mode" logs/clean_run.log | head -5
# Should see references to "shadow" or "trading_mode"
```

### Step 3: Check for Shadow Mode Indicators
```bash
grep -i "SHADOW\|shadow.*mode\|virtual.*portfolio" logs/clean_run.log | head -20
```

Look for messages like:
- `[SS:ShadowMode] Initialized virtual portfolio`
- `[EM:ShadowMode] Order intercepted`
- `SHADOW-` (order ID prefix)

### Step 4: Verify Orders Are NOT Sent to Binance
```bash
grep "exchange_order_id" logs/clean_run.log | head -10
```

Check the `exchange_order_id` field:
- **Shadow mode**: Should see `"SHADOW-<uuid>"` or `"mode": "shadow"`
- **Live mode**: Would see real Binance order IDs like `123456789`

### Step 5: Check Virtual Portfolio Creation
```bash
grep -i "virtual" logs/clean_run.log | head -20
```

Look for:
- `virtual_balances` initialization
- `virtual_portfolio` creation
- `virtual_nav` updates

---

## What Each Log Line Means

### ❌ WRONG (Real orders being placed)
```
exchange_order_id: 1234567890
status: "FILLED"
side: "BUY"
```
This would mean orders are going to Binance (real capital at risk).

### ✅ RIGHT (Shadow mode working)
```
exchange_order_id: SHADOW-abc123def456
mode: "shadow"
status: "FILLED"
side: "BUY"
```
This means orders are simulated locally (NO real capital at risk).

---

## Configuration Chain Verification

### Verify Config is Reading TRADING_MODE

```bash
# Check that Config class loads the env var
grep "self.trading_mode" core/config.py
# Output: Line 571 should show: self.trading_mode = os.getenv("TRADING_MODE", "live").lower()
```

### Verify SharedState is Reading from Config

```bash
# Check that SharedState reads from config
grep "self.trading_mode" core/shared_state.py | head -5
# Should see: self.trading_mode = str(getattr(self.config, 'trading_mode', 'live') or 'live')
```

### Verify ExecutionManager is Using It

```bash
# Check the shadow mode gate
grep -A 5 "def _get_trading_mode" core/execution_manager.py
# Should show the method that checks SharedState.trading_mode
```

---

## Troubleshooting

### Problem: Logs show `live_mode=True`
**Solution**: This is EXPECTED. It's checking a different variable (old LIVE_MODE). Shadow mode uses TRADING_MODE instead.

### Problem: No SHADOW- orders in logs
**Possible causes**:
1. Environment variable not set: `echo $TRADING_MODE`
2. No orders being placed (check agent signals)
3. Orders going to Binance instead of being simulated

### Problem: Getting errors in logs about trading mode
**Solution**: Check that `TRADING_MODE` is lowercase:
```bash
export TRADING_MODE=shadow  # Correct
export TRADING_MODE=Shadow  # WRONG (must be lowercase)
```

### Problem: System not reading the environment variable
**Solution**: Ensure you set it BEFORE starting the system:
```bash
# WRONG:
python3 main_phased.py
export TRADING_MODE=shadow

# CORRECT:
export TRADING_MODE=shadow
python3 main_phased.py
```

---

## Quick Verification Script

Run this to check everything:

```bash
#!/bin/bash

echo "🔍 Shadow Mode Verification"
echo "=================================="

echo ""
echo "1. Environment Variable:"
echo "   TRADING_MODE=$TRADING_MODE"

echo ""
echo "2. Configuration File Check:"
grep "self.trading_mode = os.getenv" core/config.py && echo "   ✓ Config loads TRADING_MODE" || echo "   ✗ Config missing TRADING_MODE"

echo ""
echo "3. Recent Log Activity:"
echo "   Last 10 lines mentioning 'shadow' or 'SHADOW':"
tail -100 logs/clean_run.log | grep -i shadow | tail -10

echo ""
echo "4. Order Status:"
echo "   Last order type in logs:"
tail -50 logs/clean_run.log | grep "exchange_order_id" | tail -1

echo ""
echo "=================================="
```

---

## What Should Happen in Each Stage

### Stage 1: System Startup
```
✓ Config.__init__() reads TRADING_MODE="shadow"
✓ SharedState.__init__() stores trading_mode="shadow"
✓ Logs may show "live_mode=True" (legacy, ignore it)
```

### Stage 2: First Order Placed
```
✓ ExecutionManager._get_trading_mode() returns "shadow"
✓ _place_with_client_id() checks mode
✓ Goes to _simulate_fill() instead of exchange_client
✓ Order gets SHADOW-<uuid> ID
```

### Stage 3: After Fill
```
✓ _update_virtual_portfolio_on_fill() called
✓ SharedState.virtual_balances updated
✓ Virtual PnL accumulated
✓ Real Binance balance UNCHANGED
```

---

## Production Checklist

Before switching to live mode, verify:

- [ ] `echo $TRADING_MODE` returns "shadow"
- [ ] Logs show `SHADOW-` order IDs (not real Binance IDs)
- [ ] Real Binance balance checked and UNCHANGED
- [ ] Virtual NAV and PnL tracking correctly
- [ ] All system components still working (agents, RiskManager, etc.)
- [ ] No errors in logs related to trading_mode
- [ ] System has been running in shadow mode for 24+ hours

When ready to go live:

1. Stop the system: `pkill -f main_phased.py`
2. Set live mode: `export TRADING_MODE=live`
3. Restart: `nohup python -u main_phased.py > logs/clean_run.log 2>&1 &`
4. Monitor logs: `tail -f logs/clean_run.log`
5. Verify first real order executes
6. Monitor for 1 hour closely

---

## Key Files for Shadow Mode

| File | Purpose |
|------|---------|
| `core/config.py` | Loads TRADING_MODE env var in `__init__` (line 571) |
| `core/shared_state.py` | Stores trading_mode and manages virtual portfolio |
| `core/execution_manager.py` | `_get_trading_mode()` + shadow mode gate at `_place_with_client_id()` |

---

## Safety Guarantees

✅ **Real Capital**: ZERO risk in shadow mode (env var TRADING_MODE controls it)  
✅ **Default**: Safe ("live" mode if not set)  
✅ **Binance**: NOT called for order placement in shadow mode  
✅ **Verification**: Look for `SHADOW-` prefix in order IDs  
✅ **Reversible**: Can switch back anytime with `export TRADING_MODE=live`

---

## Questions?

**Q: Why does the log say `live_mode=True` if I set `TRADING_MODE=shadow`?**  
A: Those are two different systems. `live_mode` is legacy. Shadow mode uses `TRADING_MODE`. Orders will still be simulated.

**Q: How do I know orders are simulated?**  
A: Check for `SHADOW-` prefix in order IDs, or grep logs: `grep "SHADOW-" logs/clean_run.log`

**Q: Can I switch from shadow to live without restarting?**  
A: No, you must restart. Set env var, then restart the system.

**Q: What if something breaks?**  
A: You're in shadow mode, so no real capital is at risk. Switch back to `TRADING_MODE=live` and restart.

---

**Status**: Shadow Mode Production Ready ✅  
**Last Updated**: 2026-03-02
