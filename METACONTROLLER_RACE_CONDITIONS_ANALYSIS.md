# MetaController Race Condition Analysis

**Status:** CRITICAL RACE CONDITIONS IDENTIFIED
**Date:** March 2, 2026
**Severity:** HIGH - Can cause position/order conflicts

---

## Executive Summary

MetaController has **significant race conditions** in several critical areas:

1. **Position Check-Before-Execute Race** ⚠️ CRITICAL
   - Check position exists → delay → Execute SELL → Position already closed
   - Check position free → delay → Another thread reserves it → Execution fails

2. **Single-Intent Violation Race** 🔴 CRITICAL
   - Thread 1 checks `_position_blocks_new_buy()` → clear → Creates BUY order
   - Thread 2 creates second BUY simultaneously
   - Result: Multiple buy orders for same symbol

3. **Shared State Read-Modify-Write Race** 🔴 CRITICAL
   - No synchronization on `self.shared_state` reads/writes
   - Multiple methods read position, evaluate, then write asynchronously
   - State can change between read and write

4. **Signal Processing Race** ⚠️ HIGH
   - Exit signals processed without atomic transaction with order submission
   - Multiple exits possible for same symbol in same cycle

5. **Dust State Management Race** ⚠️ HIGH
   - Dust state checked, then modified without locks
   - Cleanup cycle can race with trading logic

---

## Critical Race Condition #1: Position Check-Then-Execute

### Location
- `_position_blocks_new_buy()` (line 1747)
- Signal evaluation → Order submission (async gap)

### Problem

```python
# RACE CONDITION EXAMPLE
async def process_symbol(symbol):
    # T=0: Check position
    blocks, value, floor, reason = await self._position_blocks_new_buy(symbol, existing_qty)
    # Result: blocks=False (no position exists)
    
    # T=100ms: CONTEXT SWITCH - Another thread/coroutine
    # Other coroutine creates a position!
    
    # T=200ms: Back to this coroutine - STALE DECISION
    if not blocks:  # Still thinks position is free!
        # Submit BUY order WITHOUT rechecking position
        await self.execution_manager.place_order(symbol, "BUY", qty)
        # ERROR: Position now exists! Should have been blocked!
```

### Impact
- ❌ Multiple BUY orders for same symbol
- ❌ Violates single-intent-per-symbol rule
- ❌ Duplicate positions can be opened

### Root Cause
- Async method with **no atomic guarantee**
- No lock between check and execution
- Shared state can change between operations

### Solution Needed
```python
# ADD: Atomic check-and-reserve operation
async def _check_and_reserve_symbol(self, symbol: str) -> bool:
    """Atomically check if symbol is free and reserve it."""
    async with asyncio.Lock():  # ← Need lock here!
        blocks, _, _, _ = await self._position_blocks_new_buy(symbol, qty)
        if not blocks:
            self._reserved_symbols.add(symbol)  # Mark as reserved
            return True
        return False
```

---

## Critical Race Condition #2: Single-Intent Violation

### Location
- `_position_blocks_new_buy()` (line 1747)
- `evaluate_and_act()` (line 5429)
- Concurrent `execute_trading_cycle()` calls

### Problem

```
Timeline:
T=0ms:   Cycle 1 reads position for BTC/USDT → position_qty = 0
T=1ms:   Cycle 2 reads position for BTC/USDT → position_qty = 0
         Both think they can BUY!
T=2ms:   Cycle 1 submits BUY order
T=3ms:   Cycle 2 submits BUY order  ← SECOND BUY!
T=4ms:   Both orders execute, two BTC/USDT positions created
```

### Code Evidence
```python
# Line 748: Sync read without lock
qty = float(self.shared_state.get_position_qty(symbol) or 0.0)

# Line 1747-1785: _position_blocks_new_buy logic
async def _position_blocks_new_buy(self, symbol: str, existing_qty: float):
    # ... evaluation logic ...
    return blocks, pos_value, significant_floor, reason
    # NO LOCK - Anyone can read and modify simultaneously!
```

### Why It Happens
- **No mutual exclusion** on position data
- **Async/await gaps** where context can switch
- **No atomic transaction** between check and execution
- Multiple coroutines can evaluate simultaneously

### Impact
- 🔴 Multiple positions per symbol
- 🔴 Violates invariant: "ONE position per symbol"
- 🔴 Can cause:
  - Duplicate liquidation
  - Doubled capital allocation
  - Conflict in TP/SL logic

---

## Critical Race Condition #3: Shared State Read-Modify-Write

### Location
- `evaluate_and_act()` → Signal processing → Execution
- Multiple reads of `self.shared_state` without locks

### Problem

```python
# Line 1678: Read position
pos_map = await _safe_await(self.shared_state.get_open_positions())

# Line ~1700: Evaluate decision based on snapshot

# Line ~1800: ASYNC GAP - another thread can modify state

# Line ~1900: Execute based on stale position data
await self._execute_exit(symbol, signal)
# ERROR: Position might have changed!
```

### Root Cause
- `self.shared_state` is shared reference with **no synchronization**
- Methods read different versions at different times
- No versioning or ACID guarantees

### Impact
- Decisions based on **stale position data**
- Exits for **positions that no longer exist**
- Entry on **positions being liquidated**

---

## Critical Race Condition #4: Signal Processing Race

### Location
- `evaluate_and_act()` (line 5429)
- `_build_decisions()` → `_flush_intents_to_cache()` → Execution

### Problem

```
Timeline:
T=0ms:   Signal 1 (SELL BTC): Loaded and evaluated
T=1ms:   Signal 2 (SELL BTC): Loaded and evaluated  ← Same symbol!
T=2ms:   Both marked for execution
T=3ms:   Signal 1 executes SELL order
T=4ms:   Signal 2 executes SELL order  ← DUPLICATE SELL!
         Position already flat - ERROR!
```

### Code Evidence
```python
# Line 5516: All signals loaded together
decisions = await self._build_decisions(accepted_symbols_set)

# Line 5545: Signals added to batcher
for symbol, side, signal in decisions:
    self.signal_batcher.add_signal(batched)
    # No duplicate detection across symbols!
```

### Impact
- Multiple SELL orders for same symbol
- Positions sold twice
- FEE WASTE

---

## Critical Race Condition #5: Dust State Management

### Location
- `_cleanup_symbol_dust_state()` (line 463)
- `_run_symbol_dust_cleanup_cycle()` (line 524)
- Trading logic simultaneously accessing dust state

### Problem

```python
# Trading cycle reads dust state
dust_state = self._symbol_dust_states.get(symbol)
if dust_state == "DUST_ACCUMULATING":
    # Decide to heal dust
    
    # Meanwhile, cleanup cycle modifies state!
    # self._symbol_dust_states[symbol] = "DUST_MATURED"
    
    # Execution based on stale state!
```

### Root Cause
- `_symbol_dust_states` dictionary accessed without lock
- Two different cycles can modify simultaneously
- No atomic read-modify-write

---

## Critical Race Condition #6: Missing Lock in Multiple Locations

### Locations Without Locks

```python
# Line 1723: Direct dictionary pop - NO LOCK
self.shared_state.open_trades.pop(sym, None)
#                                          ↑ RACE CONDITION

# Line 1475: Direct assignment - NO LOCK
self._profit_lock_checkpoint = float(...)
#                             ↑ RACE CONDITION

# Line 683-684: Dictionary manipulation - NO LOCK
self.shared_state.dynamic_config = {}
state = self.shared_state.dynamic_config
#       ↑ RACE CONDITION - state changed between lines!

# Line 1010: Reading without lock
active_symbols = set(self.shared_state.get_analysis_symbols() or [])
#           ↑ Set can change while iterating
```

---

## Severity Assessment

| Race Condition | Location | Severity | Impact |
|---|---|---|---|
| Position check-then-execute | `_position_blocks_new_buy()` | 🔴 CRITICAL | Multiple positions per symbol |
| Single-intent violation | `evaluate_and_act()` | 🔴 CRITICAL | Duplicate BUY orders |
| Shared state R-M-W | Signal processing | 🔴 CRITICAL | Stale position data |
| Signal duplication | `_build_decisions()` | 🟠 HIGH | Multiple SELL orders |
| Dust state race | `_cleanup_symbol_dust_state()` | 🟠 HIGH | Inconsistent state |
| Dictionary access | Various | 🟠 HIGH | Data corruption |

---

## Reproducing Race Condition #1

### Scenario: Concurrent BUY Orders

```python
import asyncio

async def test_race_condition():
    """Reproduce the two-BUY-orders race condition."""
    
    # Setup
    meta = MetaController(...)
    symbol = "BTC/USDT"
    
    # Clear position
    await meta.shared_state.set_position(symbol, {"quantity": 0})
    
    async def attempt_buy():
        # Both coroutines check position
        blocks, _, _, _ = await meta._position_blocks_new_buy(symbol, 0)
        
        # Both see: blocks=False (no position)
        if not blocks:
            # Both try to BUY
            result = await meta.execution_manager.place_order(
                symbol, "BUY", qty=1.0
            )
            return result
    
    # Run both simultaneously
    results = await asyncio.gather(
        attempt_buy(),  # Coroutine 1
        attempt_buy(),  # Coroutine 2 - RACE!
    )
    
    # Check result
    open_positions = await meta.shared_state.get_open_positions()
    position_count = len([p for p in open_positions if p["symbol"] == symbol])
    
    print(f"Positions created: {position_count}")
    # Expected: 1 position
    # Actual: 2 positions! ← RACE CONDITION
    assert position_count == 1, f"RACE CONDITION: {position_count} positions!"
```

---

## Proposed Fixes

### Fix #1: Add Symbol-Level Locks

```python
# In MetaController.__init__()
self._symbol_locks = {}  # Dict[str, asyncio.Lock]

async def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
    """Get or create a lock for this symbol."""
    if symbol not in self._symbol_locks:
        self._symbol_locks[symbol] = asyncio.Lock()
    return self._symbol_locks[symbol]

async def _check_and_reserve_symbol(self, symbol: str) -> bool:
    """Atomically check if symbol is free."""
    async with await self._get_symbol_lock(symbol):
        # Now atomic!
        blocks, _, _, _ = await self._position_blocks_new_buy(symbol, qty)
        if not blocks:
            # Reserve it
            self._reserved_symbols.add(symbol)
            return True
        return False
```

### Fix #2: Add Transaction Wrapper

```python
async def _atomic_buy_order(self, symbol: str, qty: float, signal: dict):
    """Submit BUY order atomically with position check."""
    async with await self._get_symbol_lock(symbol):
        # T=0: Check
        blocks, _, _, _ = await self._position_blocks_new_buy(symbol, qty)
        
        if blocks:
            self.logger.warning(f"[Atomic] BUY blocked for {symbol} (position exists)")
            return None
        
        # T=1: Reserve (still holding lock!)
        self._reserved_symbols.add(symbol)
        
        try:
            # T=2: Execute (still holding lock!)
            result = await self.execution_manager.place_order(symbol, "BUY", qty)
            return result
        finally:
            # T=3: Release
            self._reserved_symbols.discard(symbol)
```

### Fix #3: Deduplicate Signals

```python
async def _build_decisions_dedup(self, symbols: Set[str]):
    """Build decisions with deduplication per symbol."""
    decisions = await self._build_decisions(symbols)
    
    # Group by symbol
    by_symbol = defaultdict(list)
    for symbol, side, signal in decisions:
        by_symbol[symbol].append((side, signal))
    
    # Keep only highest-confidence per symbol
    result = []
    for symbol, candidates in by_symbol.items():
        # Filter duplicates
        by_side = {}
        for side, signal in candidates:
            if side not in by_side or signal["confidence"] > by_side[side]["confidence"]:
                by_side[side] = signal
        
        # Add deduplicated
        for side, signal in by_side.items():
            result.append((symbol, side, signal))
    
    return result
```

### Fix #4: Use asyncio.Lock for Shared State

```python
# In MetaController.__init__()
self._shared_state_lock = asyncio.Lock()

async def _safe_read_position(self, symbol: str):
    """Read position safely with lock."""
    async with self._shared_state_lock:
        return await _safe_await(self.shared_state.get_position(symbol))

async def _safe_write_position(self, symbol: str, position: dict):
    """Write position safely with lock."""
    async with self._shared_state_lock:
        return await _safe_await(self.shared_state.set_position(symbol, position))
```

---

## Testing Race Conditions

### Test Suite Recommendation

```python
import pytest
import asyncio

class TestMetaControllerRaceConditions:
    """Race condition test suite."""
    
    @pytest.mark.asyncio
    async def test_concurrent_buy_orders(self):
        """Test: Two BUY orders simultaneously for same symbol."""
        meta = MetaController(...)
        symbol = "BTC/USDT"
        
        # Clear position
        await meta.shared_state.set_position(symbol, {"quantity": 0})
        
        async def buy():
            return await meta.process_signal(symbol, "BUY", qty=1.0)
        
        # Run 10 times to increase race condition probability
        for _ in range(10):
            results = await asyncio.gather(buy(), buy())
            positions = await meta.shared_state.get_open_positions()
            count = len([p for p in positions if p["symbol"] == symbol])
            assert count <= 1, f"RACE: {count} positions for {symbol}"
    
    @pytest.mark.asyncio
    async def test_concurrent_exit_signals(self):
        """Test: Multiple SELL signals for same symbol."""
        # ... similar test ...
        pass
    
    @pytest.mark.asyncio
    async def test_stale_position_read(self):
        """Test: Position changes between check and execution."""
        # ... test for stale reads ...
        pass
```

---

## Impact on Safety Mechanisms

This analysis reveals why the SafetyMechanisms audit identified issues:

1. **Single-Intent Guard (70% complete)** - Race conditions prevent 30% effectiveness
   - Guard works in isolation
   - But concurrent operations bypass it

2. **Position Consolidation (40% complete)** - Enables duplicate positions
   - Can't consolidate if multiple positions exist
   - Multiple positions can form due to races

3. **Min Hold Time (100% complete)** - Vulnerable to bypass
   - Position age validated
   - But position could change after validation

---

## Recommendations

### Immediate (Critical)
1. ✅ Add `asyncio.Lock()` for each symbol
2. ✅ Make `_position_blocks_new_buy()` → `_check_and_reserve_symbol()` atomic
3. ✅ Deduplicate signals per symbol per cycle

### Short-Term (High Priority)
1. ✅ Wrap all `shared_state` access in locks
2. ✅ Add transaction wrapper for order submission
3. ✅ Implement signal deduplication

### Medium-Term (Architecture)
1. 🔄 Redesign signal processing with transaction semantics
2. 🔄 Add position versioning to detect stale reads
3. 🔄 Implement proper actor model for position management

---

## Conclusion

MetaController has **multiple critical race conditions** that can cause:
- ❌ Duplicate positions (violates single-intent rule)
- ❌ Multiple orders for same symbol
- ❌ Stale position data
- ❌ Fee waste and confusion

**These must be fixed before production deployment.**

The ExitArbitrator implementation helps with priority arbitration, but **does not solve these concurrency issues**. Both are needed:
1. ExitArbitrator (decision quality)
2. Concurrency fixes (execution safety)

---

**Severity: 🔴 CRITICAL**
**Timeline: URGENT (fix before next deployment)**
**Effort: 2-3 days for proper fixes**
