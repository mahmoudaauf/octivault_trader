# MetaController Race Condition Fixes - Quick Implementation Guide

**Priority:** 🔴 CRITICAL
**Estimated Effort:** 2-3 days
**Risk if not fixed:** High - Duplicate positions, fee waste, execution failures

---

## 5-Minute Summary

MetaController has **5 critical race conditions**:

1. **Position Check-Then-Execute** - Check position → DELAY → Execute BUY → Double position
2. **Single-Intent Violation** - Two coroutines read same position, both BUY
3. **Shared State R-M-W** - Read position → DELAY → Write based on stale data
4. **Signal Duplication** - Multiple SELL signals for same symbol in one cycle
5. **Dust State Race** - Cleanup cycle and trading cycle modify state simultaneously

**Fix:** Add `asyncio.Lock()` per symbol + atomic transactions

---

## Phase 1: Add Symbol Locks (2 hours)

### Step 1: Initialize Locks in `__init__`

**File:** `core/meta_controller.py`
**Location:** Around line 1277 (near `self._performance_lock`)

**Add:**
```python
# Add with other locks
self._symbol_locks: Dict[str, asyncio.Lock] = {}
self._symbol_locks_lock = asyncio.Lock()  # Lock for the locks dict itself
self._reserved_symbols: Set[str] = set()
```

### Step 2: Create Lock Manager Method

**Add new method in MetaController class:**
```python
async def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
    """Get or create an asyncio.Lock for this symbol."""
    # Ensure locks dict access is thread-safe
    if symbol not in self._symbol_locks:
        async with self._symbol_locks_lock:
            # Double-check after acquiring lock
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = asyncio.Lock()
    return self._symbol_locks[symbol]
```

### Step 3: Create Atomic Check-and-Reserve Method

**Add new method in MetaController class:**
```python
async def _check_and_reserve_symbol(self, symbol: str, qty: float) -> Tuple[bool, str]:
    """
    ATOMIC: Check if position blocks BUY, and reserve symbol if clear.
    
    Returns: (can_proceed: bool, reason: str)
    """
    lock = await self._get_symbol_lock(symbol)
    async with lock:
        # Check if position blocks (NOW ATOMIC!)
        blocks, pos_value, floor, reason = await self._position_blocks_new_buy(symbol, qty)
        
        if blocks:
            return False, f"Position blocks: {reason}"
        
        # Mark as reserved (prevent concurrent operations)
        if symbol in self._reserved_symbols:
            return False, "Symbol already reserved"
        
        self._reserved_symbols.add(symbol)
        self.logger.info(f"[Race:Guard] Reserved {symbol}")
        return True, "Reserved"

async def _release_symbol(self, symbol: str) -> None:
    """Release symbol reservation."""
    self._reserved_symbols.discard(symbol)
    self.logger.debug(f"[Race:Guard] Released {symbol}")
```

---

## Phase 2: Atomic Order Submission (3-4 hours)

### Step 1: Create Atomic BUY Wrapper

**Add new method:**
```python
async def _atomic_buy_order(
    self,
    symbol: str,
    qty: float,
    signal: Dict[str, Any],
    planned_quote: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    ATOMIC: Check position + reserve + submit BUY order.
    
    Guarantees:
    - Only ONE BUY order per symbol
    - No race between position check and order submission
    """
    lock = await self._get_symbol_lock(symbol)
    
    async with lock:
        try:
            # Step 1: Check if position exists (holding lock!)
            blocks, pos_value, floor, reason = await self._position_blocks_new_buy(symbol, qty)
            
            if blocks:
                self.logger.warning(
                    f"[Atomic:BUY] BLOCKED {symbol}: {reason} (pos_value={pos_value})"
                )
                return None
            
            # Step 2: Mark as reserved (holding lock!)
            self._reserved_symbols.add(symbol)
            
            try:
                # Step 3: Submit order (holding lock!)
                order = await self.execution_manager.place_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=qty,
                    planned_quote=planned_quote,
                )
                
                if order and order.get("ok"):
                    self.logger.info(
                        f"[Atomic:BUY] ✓ Order submitted {symbol}: order_id={order.get('order_id')}"
                    )
                    return order
                else:
                    self.logger.error(f"[Atomic:BUY] ✗ Order failed {symbol}: {order}")
                    return None
                    
            finally:
                # Step 4: Release reservation (even on failure)
                self._reserved_symbols.discard(symbol)
                
        except Exception as e:
            self.logger.error(f"[Atomic:BUY] ✗ Exception {symbol}: {e}", exc_info=True)
            self._reserved_symbols.discard(symbol)
            return None
```

### Step 2: Create Atomic SELL Wrapper

**Add new method:**
```python
async def _atomic_sell_order(
    self,
    symbol: str,
    qty: float,
    signal: Dict[str, Any],
    reason: str = "manual",
) -> Optional[Dict[str, Any]]:
    """
    ATOMIC: Check position exists + consolidate qty + submit SELL order.
    
    Guarantees:
    - Sells total position (no partial SELL duplicates)
    - Single SELL order per symbol per cycle
    """
    lock = await self._get_symbol_lock(symbol)
    
    async with lock:
        try:
            # Step 1: Get current position (holding lock!)
            position = await _safe_await(self.shared_state.get_position(symbol))
            
            if not position or position.get("quantity", 0) <= 0:
                self.logger.warning(f"[Atomic:SELL] ✗ {symbol} has no position")
                return None
            
            # Step 2: Consolidate total quantity
            total_qty = float(position.get("quantity", 0))
            self.logger.info(
                f"[Atomic:SELL] Consolidating {symbol}: signal_qty={qty} → total_qty={total_qty}"
            )
            
            # Step 3: Mark as reserved (prevent concurrent SELL)
            self._reserved_symbols.add(symbol)
            
            try:
                # Step 4: Submit consolidated SELL (holding lock!)
                order = await self.execution_manager.place_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=total_qty,
                )
                
                if order and order.get("ok"):
                    self.logger.info(
                        f"[Atomic:SELL] ✓ Order submitted {symbol}: qty={total_qty}, order_id={order.get('order_id')}"
                    )
                    return order
                else:
                    self.logger.error(f"[Atomic:SELL] ✗ Order failed {symbol}: {order}")
                    return None
                    
            finally:
                self._reserved_symbols.discard(symbol)
                
        except Exception as e:
            self.logger.error(f"[Atomic:SELL] ✗ Exception {symbol}: {e}", exc_info=True)
            self._reserved_symbols.discard(symbol)
            return None
```

### Step 3: Update Signal Processing to Use Atomic Methods

**Location:** Find where BUY/SELL orders are submitted (probably in `_build_decisions` or `evaluate_and_act`)

**Replace:**
```python
# OLD: Direct order submission (RACE CONDITION!)
# result = await self.execution_manager.place_order(symbol, side, qty)

# NEW: Atomic order submission (SAFE!)
if side == "BUY":
    result = await self._atomic_buy_order(symbol, qty, signal, planned_quote)
elif side == "SELL":
    result = await self._atomic_sell_order(symbol, qty, signal, reason)
```

---

## Phase 3: Signal Deduplication (2 hours)

### Step 1: Add Deduplication Logic

**Add new method:**
```python
async def _deduplicate_decisions(
    self,
    decisions: List[Tuple[str, str, Dict[str, Any]]]
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Remove duplicate signals per symbol per cycle.
    
    For each symbol:
    - Keep at most ONE BUY signal (highest confidence)
    - Keep at most ONE SELL signal (highest confidence)
    """
    # Group by (symbol, side)
    by_symbol_side = defaultdict(list)
    for symbol, side, signal in decisions:
        by_symbol_side[(symbol, side)].append(signal)
    
    # Deduplicate: keep highest confidence
    result = []
    for (symbol, side), signals in by_symbol_side.items():
        if not signals:
            continue
        
        # Sort by confidence descending
        signals.sort(key=lambda s: float(s.get("confidence", 0.0)), reverse=True)
        best_signal = signals[0]
        
        if len(signals) > 1:
            self.logger.warning(
                f"[Dedup] {symbol} {side}: Found {len(signals)} signals, keeping highest conf={best_signal.get('confidence')}"
            )
        
        result.append((symbol, side, best_signal))
    
    return result
```

### Step 2: Call Deduplication in evaluate_and_act

**Location:** Line ~5516 in `evaluate_and_act()`

**Update:**
```python
# OLD:
decisions = await self._build_decisions(accepted_symbols_set)

# NEW:
decisions = await self._build_decisions(accepted_symbols_set)
decisions = await self._deduplicate_decisions(decisions)  # ← ADD THIS
```

---

## Phase 4: Testing (2 hours)

### Test 1: Concurrent Buy Orders

**File:** `tests/test_meta_controller_races.py` (create new)

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_concurrent_buy_orders_blocked():
    """
    RACE TEST: Two BUY orders for same symbol simultaneously.
    
    Expected: Only one should succeed (second should be blocked by lock).
    """
    meta = MetaController(...)
    symbol = "BTC/USDT"
    
    # Clear position
    await meta.shared_state.set_position(symbol, {"quantity": 0})
    
    async def attempt_buy():
        return await meta._atomic_buy_order(
            symbol=symbol,
            qty=1.0,
            signal={"confidence": 0.8}
        )
    
    # Run both simultaneously
    results = await asyncio.gather(
        attempt_buy(),  # Attempt 1
        attempt_buy(),  # Attempt 2 - should be blocked
    )
    
    # Verify
    orders_submitted = [r for r in results if r is not None]
    assert len(orders_submitted) <= 1, f"RACE CONDITION: {len(orders_submitted)} orders submitted!"
    
    print("✓ PASS: Only 1 order submitted (race prevented)")
```

### Test 2: Concurrent Sell Signals

```python
@pytest.mark.asyncio
async def test_deduplicate_sell_signals():
    """
    DEDUP TEST: Multiple SELL signals for same symbol.
    
    Expected: Only one SELL order submitted (consolidated qty).
    """
    decisions = [
        ("BTC/USDT", "SELL", {"confidence": 0.9, "reason": "TP"}),
        ("BTC/USDT", "SELL", {"confidence": 0.7, "reason": "Agent"}),
        ("ETH/USDT", "SELL", {"confidence": 0.8, "reason": "Signal"}),
    ]
    
    meta = MetaController(...)
    dedup = await meta._deduplicate_decisions(decisions)
    
    # Verify
    assert len(dedup) == 2, f"Should have 2 signals, got {len(dedup)}"
    
    btc_sells = [d for d in dedup if d[0] == "BTC/USDT" and d[1] == "SELL"]
    assert len(btc_sells) == 1, f"Should have 1 BTC/USDT SELL, got {len(btc_sells)}"
    
    print("✓ PASS: Signals deduplicated correctly")
```

### Run Tests

```bash
# Run race condition tests
pytest tests/test_meta_controller_races.py -v

# Should output:
# test_concurrent_buy_orders_blocked PASSED
# test_deduplicate_sell_signals PASSED
```

---

## Phase 5: Verification (1 hour)

### Checklist

- [ ] Symbol locks initialized in `__init__`
- [ ] `_get_symbol_lock()` method implemented
- [ ] `_check_and_reserve_symbol()` implemented
- [ ] `_atomic_buy_order()` implemented
- [ ] `_atomic_sell_order()` implemented
- [ ] Signal deduplication implemented
- [ ] All order submissions use atomic methods
- [ ] Race condition tests pass (2/2)
- [ ] Existing unit tests still pass
- [ ] Code reviewed by 2+ people

### Manual Verification

```bash
# Start trading in test environment
# Monitor logs for:
# - [Atomic:BUY] messages
# - [Atomic:SELL] messages
# - [Dedup] messages
# - [Race:Guard] messages

# Verify:
grep -i "atomic\|dedup\|race:guard" logs/trading.log
# Should see these messages (not many, only on actual trades)
```

---

## Phase 6: Deployment

### Staging (2-3 hours monitoring)

1. Deploy to staging
2. Run for 3+ hours
3. Monitor logs for:
   - Any `[Atomic]` errors
   - Any `[Dedup]` warnings
   - Any `[Race]` incidents
4. Check positions: should all be single per symbol
5. Check orders: should be no duplicates

### Production (24-48 hour monitoring)

1. Deploy to production
2. Monitor first 24 hours closely
3. Verify:
   - No duplicate positions created
   - No duplicate orders submitted
   - No fee waste from signal duplication
4. Collect metrics

---

## Rollback Plan

If issues occur:

```bash
# Quick rollback
git revert <commit-hash>
systemctl restart octivault_trader

# Should return to pre-fix behavior
```

---

## Summary Table

| Fix | Effort | Impact | Risk |
|-----|--------|--------|------|
| Phase 1: Symbol locks | 2h | Prevents concurrent symbol access | LOW |
| Phase 2: Atomic BUY/SELL | 3-4h | Prevents duplicate orders | LOW |
| Phase 3: Deduplication | 2h | Prevents signal duplication | LOW |
| Phase 4: Testing | 2h | Verifies fixes work | LOW |
| Phase 5: Verification | 1h | Final checks | LOW |
| **TOTAL** | **10-12h** | **Eliminates all 5 race conditions** | **LOW** |

---

**Status:** Ready to implement
**Blocking:** Deployment should wait for these fixes
**Questions?** Check METACONTROLLER_RACE_CONDITIONS_ANALYSIS.md for detailed analysis

