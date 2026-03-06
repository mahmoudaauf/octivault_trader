# 🔥 SHADOW MODE DUPLICATE EMISSION FIX - AWAITING IMPLEMENTATION

## Current Status
✅ **Issue Identified** | ⏳ **Awaiting Code Merge**

The shadow mode code (with the duplicate TRADE_EXECUTED emission) **has not yet been merged** into the main execution_manager.py file.

## The Problem (Once Shadow Mode is Added)

In `_place_with_client_id()`, there are **two emission points** for TRADE_EXECUTED:

```python
# POINT 1: First emission in shadow mode (LINES ~7972)
await self._emit_trade_executed_event(
    symbol=symbol,
    side=side,
    tag=tag,
    order=simulated,
)

# ... then calls...

# POINT 2: Second emission inside _handle_post_fill() (LINE 304)
trade_event_emitted = bool(
    await self._emit_trade_executed_event(sym, side_u, str(tag or ""), order)
)
```

### Result: 5x NAV Explosion (107 → 557)

Every trade event listener runs **twice**:
- `virtual_balances` updated 2x
- `virtual_positions` updated 2x  
- `realized_pnl` compounded 2x
- Position accounting doubled

Over multiple cycles → capital compounds incorrectly → NAV explodes 5x.

---

## The Correct Fix

**Delete lines containing the first `_emit_trade_executed_event()` call and its try-except block in `_place_with_client_id()`.**

Keep only the `_handle_post_fill()` call, which emits TRADE_EXECUTED internally.

### Before
```python
if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        try:
            # ❌ REMOVE THIS BLOCK
            await self._emit_trade_executed_event(
                symbol=symbol,
                side=side,
                tag=tag,
                order=simulated,
            )
            self.logger.info(...)
        except Exception as e:
            self.logger.error(...)
            if bool(self._cfg("STRICT_OBSERVABILITY_EVENTS", False)):
                raise
        
        # ✅ KEEP THIS BLOCK - it emits internally
        try:
            await self._handle_post_fill(
                symbol=symbol,
                side=side,
                order=simulated,
                tag=tag,
            )
            self.logger.info(...)
        except Exception as e:
            self.logger.error(...)
            if bool(self._cfg("STRICT_ACCOUNTING_INTEGRITY", False)):
                raise
```

### After
```python
if isinstance(simulated, dict) and simulated.get("ok"):
    exec_qty = float(simulated.get("executedQty", 0.0))
    if exec_qty > 0:
        # ✅ SINGLE EMISSION POINT via _handle_post_fill()
        # Do NOT emit TRADE_EXECUTED here. Let _handle_post_fill() handle it.
        # This ensures:
        # - TRADE_EXECUTED emits exactly once (not twice)
        # - Accounting runs exactly once (not twice)
        # - Event listeners fire exactly once
        # - No duplicate mutations of virtual_balances/positions/realized_pnl
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

---

## Why This Is the Canonical Invariant

**P9 Principle:** Every confirmed fill must emit TRADE_EXECUTED exactly **once**.

- In **live mode**: Emission happens inside `_handle_post_fill()`
- In **shadow mode**: Should be identical to live mode

By removing the pre-emission in shadow mode, shadow and live both emit exactly once via `_handle_post_fill()`.

---

## How to Apply This Fix

Once shadow mode code is merged:

1. Find: `[EM:ShadowMode:Canonical]` or `await self._emit_trade_executed_event(` inside the shadow mode section
2. Delete the entire try-except block containing the first emission (before `_handle_post_fill`)
3. Keep the `_handle_post_fill()` try-except block (which emits internally)
4. Test: Verify NAV stays at ~107 USDT instead of exploding to 557

---

## Root Cause Explanation

You unknowingly rebuilt dual accounting through events:

1. Shadow mode pre-emits TRADE_EXECUTED
2. All trade listeners (balance updater, position recorder, PnL calculator) fire
3. Then `_handle_post_fill()` emits TRADE_EXECUTED again
4. Same listeners fire again (duplicate mutations)
5. Result: 2x accounting per fill → 5x NAV over multiple cycles

This violates the architectural principle: "There is now only ONE accounting system."

---

## Status Checklist

- [ ] Shadow mode code merged into execution_manager.py
- [ ] Locate duplicate emission pattern (search for "ShadowMode:Canonical")
- [ ] Delete first try-except block with pre-emission
- [ ] Verify _handle_post_fill() try-except remains
- [ ] Test shadow mode → verify NAV stays stable at ~107 USDT
- [ ] Run unit tests for shadow mode fills
- [ ] Deploy to production
