# ✅ SHADOW MODE DUPLICATE EMISSION FIX - IMPLEMENTATION GUIDE

## Status
- Current File: `core/execution_manager.py` (7999 lines) - **Does NOT yet have shadow mode code**
- Documentation: Multiple files describe the shadow mode implementation with the duplicate emission bug
- Action Needed: Apply this fix once shadow mode code is merged

---

## The Issue

Once shadow mode code is added to `_place_with_client_id()`, it will have:

```python
# After _simulate_fill():
if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        try:
            # ❌ FIRST EMISSION (Lines ~7972)
            await self._emit_trade_executed_event(
                symbol=symbol,
                side=side,
                tag=tag,
                order=simulated,
            )
            self.logger.info(f"[EM:ShadowMode:Canonical] {symbol} {side} TRADE_EXECUTED event emitted...")
        except Exception as e:
            self.logger.error(f"[EM:ShadowMode:EmitFail]...")
            if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
                raise
        
        try:
            # ✅ SECOND EMISSION (inside this function, Lines ~7993)
            await self._handle_post_fill(
                symbol=symbol,
                side=side,
                order=simulated,
                tag=tag,
            )
            # _handle_post_fill() internally emits TRADE_EXECUTED at line 304:
            #   trade_event_emitted = bool(
            #       await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
            #   )
```

**Problem:** TRADE_EXECUTED is emitted **twice**
- Once before calling `_handle_post_fill()`
- Once inside `_handle_post_fill()`

**Result:** Every event listener fires twice
- virtual_balances updated twice
- virtual_positions updated twice
- realized_pnl compounded twice
- Position accounting doubled

Over multiple trades: **107 USDT → 557 USDT (5x NAV explosion)**

---

## The Root Cause

From `core/execution_manager.py` lines 301-304:

```python
# P9 event contract: every confirmed fill must emit TRADE_EXECUTED.
trade_event_emitted = bool(
    await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
)
```

`_handle_post_fill()` **already emits TRADE_EXECUTED internally**. The pre-emission before calling it is redundant and causes duplicate accounting.

---

## The Fix

**Remove the try-except block that contains the first `_emit_trade_executed_event()` call.**

### Before
```python
if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        try:
            # BUILD CANONICAL TRADE_EXECUTED EVENT (same as live path)
            await self._emit_trade_executed_event(
                symbol=symbol,
                side=side,
                tag=tag,
                order=simulated,
            )
            self.logger.info(
                f"[EM:ShadowMode:Canonical] {symbol} {side} TRADE_EXECUTED event emitted. "
                f"qty={exec_qty:.8f}, shadow_order_id={simulated.get('exchange_order_id')}"
            )
        except Exception as e:
            self.logger.error(
                f"[EM:ShadowMode:EmitFail] Failed to emit TRADE_EXECUTED for {symbol} {side}: {e}",
                exc_info=True,
            )
            if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
                raise
        
        # CRITICAL: Call canonical post-fill handler for accounting
        # This is the SAME handler as live path, ensuring:
        # - Virtual balances are updated (in shadow mode)
        # - Positions are recorded
        # - PnL is calculated
        # - Event emissions are consistent
        try:
            await self._handle_post_fill(
                symbol=symbol,
                side=side,
                order=simulated,
                tag=tag,
            )
            self.logger.info(
                f"[EM:ShadowMode:PostFill] {symbol} {side} post-fill accounting complete"
            )
        except Exception as e:
            self.logger.error(
                f"[EM:ShadowMode:PostFillFail] Failed to handle post-fill for {symbol} {side}: {e}",
                exc_info=True,
            )
            if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                raise
```

### After
```python
if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        # CRITICAL FIX: Do NOT emit TRADE_EXECUTED here
        # _handle_post_fill() will emit it internally
        # This ensures single emission point and prevents 2x accounting
        
        # Call canonical post-fill handler for accounting
        # This handler emits TRADE_EXECUTED internally (exactly once)
        try:
            await self._handle_post_fill(
                symbol=symbol,
                side=side,
                order=simulated,
                tag=tag,
            )
            self.logger.info(
                f"[EM:ShadowMode:PostFill] {symbol} {side} post-fill accounting complete"
            )
        except Exception as e:
            self.logger.error(
                f"[EM:ShadowMode:PostFillFail] Failed to handle post-fill for {symbol} {side}: {e}",
                exc_info=True,
            )
            if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                raise
```

### What Changed
1. **Removed:** First try-except block containing `_emit_trade_executed_event()`
2. **Kept:** Second try-except block containing `_handle_post_fill()`
3. **Why:** `_handle_post_fill()` handles emission internally (line 304), so pre-emitting is redundant

---

## Exact Lines to Delete

Once shadow mode is merged, search for:
```
[EM:ShadowMode:Canonical] {symbol} {side} TRADE_EXECUTED event emitted
```

Or search backwards from the line with:
```
[EM:ShadowMode:PostFill] {symbol} {side} post-fill accounting complete
```

The deletion target is the try-except block **above** this line that contains:
```python
await self._emit_trade_executed_event(
    symbol=symbol,
    side=side,
    tag=tag,
    order=simulated,
)
```

---

## Verification

After applying the fix:

### Test 1: Shadow Mode Fill Tracking
```python
# Create a shadow mode fill
await execution_manager._place_with_client_id(
    symbol="ETHUSDT",
    side="BUY",
    quantity=0.5,
    tag="test"
)

# Verify:
# ✅ virtual_balances reduced by USDT amount (once, not twice)
# ✅ virtual_positions increased by 0.5 ETH (once, not twice)
# ✅ Log shows: [EM:ShadowMode:PostFill] ETHUSDT BUY post-fill complete
# ✅ NAV stable at ~107 USDT (not exploded to 557)
```

### Test 2: Check Dedup Cache
```python
# Verify TRADE_EXECUTED emitted exactly once
# Search logs for:
# - [EM:ShadowMode:PostFill] → Should appear once per fill
# - Should NOT see both [EM:ShadowMode:Canonical] AND [EM:ShadowMode:PostFill]
```

### Test 3: NAV Over Multiple Fills
```
Cycle 1: BUY 0.5 ETH @ 1000 = NAV 107 → 105.99 ✓
Cycle 2: BUY 0.5 ETH @ 1000 = NAV 105.99 → 104.98 ✓
Cycle 3: SELL 0.5 ETH @ 1000 = NAV 104.98 → 105.49 ✓
# With bug: 107 → 557 (5x) ✗
# After fix: Stays stable ✓
```

---

## Deployment Checklist

- [ ] Shadow mode code merged into execution_manager.py
- [ ] Confirmed lines ~7972 contain `[EM:ShadowMode:Canonical]`
- [ ] Applied fix: Removed first try-except block (pre-emission)
- [ ] Kept second try-except block (`_handle_post_fill()`)
- [ ] Run unit tests: `pytest tests/test_shadow_mode.py -v`
- [ ] Verify NAV stays ~107 USDT (not 557)
- [ ] Check logs for single [EM:ShadowMode:PostFill] per fill
- [ ] Deploy to staging
- [ ] Monitor for 24h
- [ ] Deploy to production

---

## Why This Matters

**Architectural Invariant:** Every confirmed fill must emit TRADE_EXECUTED **exactly once**.

- **Live mode:** ✅ Emits once (in `_handle_post_fill()`)
- **Shadow mode before fix:** ❌ Emits twice (pre-emission + `_handle_post_fill()`)
- **Shadow mode after fix:** ✅ Emits once (in `_handle_post_fill()`)

This ensures:
1. Single accounting pass per fill
2. Dedup cache works correctly
3. Virtual portfolio stays in sync
4. NAV calculation is accurate
5. Shadow mode truly mirrors live mode

---

## Reference Materials

- Issue Details: `00_SHADOW_MODE_DUPLICATE_EMISSION_FIX_AWAITING_MERGE.md`
- Implementation Status: `IMPLEMENTATION_COMPLETE_SHADOW_MODE_FIX.md`
- User Request: See userRequest at top of this session
