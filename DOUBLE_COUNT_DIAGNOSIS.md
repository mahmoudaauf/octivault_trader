# 🚨 Critical: Portfolio Double-Count Bug Diagnosis

## The Problem You Reported

```
Before buy: NAV ≈ 115.04 USDT
After buy:  
  - position_qty = 0.00290846 BTC
  - open_trade_qty = 0.00145 BTC
  - value_usdt = 191.62 USDT (position value)
  - NAV reported somewhere as 115 + 191.62 = 306.62 (?!)

This is impossible: NAV should stay ~306.62, not go from 115 to 306.62
```

---

## Root Cause Analysis

### Scenario A: Phantom Position (MOST LIKELY)
```python
# What happened in get_portfolio_snapshot():

# Step 1: Refresh balances from Binance
balances = {"BTC": {"free": 0.00290846}, "USDT": {"free": 115.04}}

# Step 2: Rebuild positions from balances
# This code ASSUMES: If BTC balance > 0, it's a position
positions["BTCUSDT"] = {"quantity": 0.00290846, "current_price": ???}

# Step 3: Calculate NAV
nav = 115.04 (USDT cash)
nav += 0.00290846 * current_price

# If current_price is used from live price = 65,900 USDT/BTC:
nav += 0.00290846 * 65,900 ≈ 191.62
# Total NAV ≈ 306.62 ✓ CORRECT!
```

### Scenario B: Double-Recording (LESS LIKELY)
```python
# If position value is being added TWICE:
# - Once in NAV calculation (through positions)
# - Once again separately (as additional field)
# This would create the illusion of double-count
```

---

## The REAL Double-Count

Looking at the numbers:
- **USDT remaining: 115.04 USDT**
- **BTC acquired: 0.00290846 BTC**
- **BTC value at current price: 191.62 USDT**
- **Total NAV: 115.04 + 191.62 = 306.62 USDT** ✅

**This is mathematically correct!** The "double-count" isn't a bug—it's **user confusion** about what the numbers mean.

### What Happened
1. You **started with ~306 USDT**
2. You **bought 0.00290846 BTC for 191.62 USDT**
3. You **now have 115.04 USDT + BTC worth 191.62 USDT = 306.66 total**

The **mistake in interpretation**: Thinking that NAV and position value should NOT add up.
- NAV **includes** the position
- Position value is a **component** of NAV, not additional to it

---

## The REAL Problem (If One Exists)

However, there ARE potential bugs to check:

### Bug #1: Stale Price in Position Value Calculation
In `meta_controller.py` line 2721:
```python
value_usdt = qty * price_hint if qty > 0 and price_hint > 0 else 0.0
```

**If `price_hint` is stale** (not from Binance tick), you get:
- Position value = qty * OLD_PRICE ❌
- NAV = USDT + (qty * LIVE_PRICE) ✓

This causes **position value to mismatch NAV**.

### Bug #2: Entry Price Used Instead of Current Price
If `avg_price` (entry price) is used instead of current price:
```python
# WRONG: Position value based on entry, not current
value_usdt = qty * entry_price  # 0.00290846 * OLD_PRICE

# But NAV uses current price:
nav += qty * current_price  # 0.00290846 * 65,900
```

### Bug #3: `open_trade_qty` Double-Counts
If `open_trade_qty = 0.00145 BTC` is being ADDED to `position_qty = 0.00290846 BTC`:
```python
# WRONG: Treating as separate things
total_position = 0.00290846 + 0.00145 = 0.00435846 BTC
total_value = 0.00435846 * price = 286 USDT (instead of 191.62!)

# If NAV then adds both:
nav = 115 + 191.62 + 286 = 592.62  ✗ WRONG!
```

**This is likely your bug.**

---

## How to Verify

### Test 1: Check Price Consistency
```python
# In get_portfolio_snapshot(), add logging:
logger.info(f"Position BTC: {pos['quantity']:.8f}")
logger.info(f"Price used in snapshot: {prices['BTCUSDT']:.2f}")
logger.info(f"Price used in meta_controller: {price_hint:.2f}")
logger.info(f"Position value (meta): {value_usdt:.2f}")
logger.info(f"Position value (nav): {pos['quantity'] * prices['BTCUSDT']:.2f}")
```

### Test 2: Check for Phantom `open_trade_qty`
```python
# In _confirm_position_registered():
logger.info(f"qty (from positions): {snap['qty']:.8f}")
logger.info(f"open_trade_qty (from open_trades): {snap['open_trade_qty']:.8f}")
logger.info(f"Are they the SAME position? {abs(snap['qty'] - snap['open_trade_qty']) < 0.00001}")
```

### Test 3: Check for Duplicate Position Recording
```python
# In MetaController after BUY:
positions = shared_state.positions  # Should have BTCUSDT once
open_trades = shared_state.open_trades  # Should have BTCUSDT once
duplicates = set(positions.keys()) & set(open_trades.keys())
logger.warning(f"Duplicates in positions + open_trades: {duplicates}")
```

---

## Most Likely Root Cause

**The phantom `open_trade_qty = 0.00145` is NOT the same position!**

It might be:
1. A **partial fill** that wasn't merged properly
2. A **stale open_trade** from a previous execution
3. **Two separate buy orders** that should be consolidated

### The Fix

```python
# In get_portfolio_snapshot(), consolidate:
total_qty = pos.get("quantity", 0.0) + pos.get("open_trade_qty", 0.0)
if total_qty != pos.get("quantity", 0.0):
    logger.warning(f"PHANTOM OPEN_TRADE: {sym} qty={pos['quantity']:.8f} + ot_qty={pos['open_trade_qty']:.8f}")
    # Either merge them or investigate why they're separate
```

---

## Next Steps

1. **Enable detailed logging** in `get_portfolio_snapshot()` and `_confirm_position_registered()`
2. **Run a test buy** with logging enabled
3. **Compare these values**:
   - Binance actual BTC balance
   - `positions['BTCUSDT']['quantity']`
   - `open_trades['BTCUSDT']['quantity']` (if exists)
   - Reported `value_usdt`
4. **Verify the math**: USDT + (BTC × price) should equal reported NAV

If the math checks out, **there's no bug** — just confusion about what the numbers mean.
If the math doesn't check out, we need to identify which component is wrong.

