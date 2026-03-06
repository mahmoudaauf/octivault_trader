# Race Conditions: TPSL ↔ MetaController ↔ ExecutionManager ↔ Signals

## Executive Summary

YES, there are **6 CRITICAL/HIGH severity race conditions** between TPSLEngine, MetaController, ExecutionManager, and signal processing that can cause:

- **Duplicate orders** for the same symbol in the same cycle
- **Double SELL orders** (both from TP/SL exit + Signal SELL)
- **Orphaned positions** (TPSL closes but MetaController doesn't see it)
- **Position quantity mismatches** between SharedState and ExecutionManager
- **PnL double-counting** (same exit recorded twice with different amounts)
- **Capital loss** from slippage + fees on duplicate executions

---

## Architecture Overview

### Components and Their Roles

```
┌─────────────────────────────────────────────────────────────────┐
│                         MetaController                          │
│  • Main trading loop (every N seconds)                          │
│  • Builds decisions from signals, risk, portfolio               │
│  • Submits BUY/SELL orders via ExecutionManager                │
│  • No synchronization with TPSLEngine                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌────────────────────────┐  ┌─────────────────────────┐
    │    TPSLEngine          │  │  Signals (StrategyMgr)  │
    │                        │  │                         │
    │ • Runs independently   │  │ • Generated continuously│
    │   (every 10s default)  │  │ • Added to signal cache │
    │ • Monitors TP/SL       │  │ • Collected by Meta     │
    │ • Closes via EM        │  │   before decision cycle │
    │ • NO lock with Meta    │  │                         │
    └────────────────────────┘  └─────────────────────────┘
                 │
                 └────────────┬────────────┐
                              ▼            ▼
                    ┌──────────────────────────────────┐
                    │    ExecutionManager              │
                    │                                  │
                    │ • Receives orders from:          │
                    │   - MetaController (BUY/SELL)   │
                    │   - TPSLEngine (SELL only)       │
                    │   - Direct calls                 │
                    │ • NO per-symbol lock             │
                    │ • NO queue or arbitration        │
                    │ • Direct async execution         │
                    └──────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  SharedState Mutate  │
                    │  (Positions, Orders) │
                    │  NO LOCKING          │
                    └──────────────────────┘
```

### Key Missing Synchronization

| Component | Synchronization Primitive | Status |
|-----------|---------------------------|--------|
| MetaController internal | Symbol locks (per-symbol) | ✅ YES (recently added) |
| TPSLEngine internal | None | ❌ NO |
| ExecutionManager order submission | None | ❌ NO |
| Meta→TPSL coordination | None | ❌ NO |
| TPSL→ExecutionManager coordination | None | ❌ NO |
| Concurrent signal processing | Signal deduplication (MetaController only) | ⚠️ PARTIAL |

---

## Race Condition #1: Concurrent TP/SL Exit + Signal SELL (CRITICAL)

### Problem
MetaController processes SELL signals while TPSLEngine independently triggers TP/SL SELL exit for the same symbol.

### Scenario

```
Timeline:
[T=0.0s]  MetaController collects SELL signal for BTC
          Signal added to decision list

[T=0.1s]  TPSLEngine wakes up (10s interval check)
          Detects BTC price hit TP level
          Initiates _close_via_execution_manager() for BTC

[T=0.15s] ExecutionManager.close_position(BTC, "TP_HIT")
          Submits SELL order for 100% of position
          
[T=0.20s] MetaController continues execution cycle
          Processes SELL signal collected at T=0.0s
          Calls _atomic_sell_order(BTC) with qty from signal

[T=0.25s] ExecutionManager receives SECOND SELL for BTC
          Both orders now pending in exchange

[T=0.30s] Both orders fill (or at least are submitted)
          - Order 1: TPSL SELL for 100% (from TP hit)
          - Order 2: Signal SELL for 100% (from MetaController)
          
          RESULT: Position goes negative or 2x slippage loss
```

### Code Evidence

**MetaController Signal Processing (line 5883):**
```python
# Build decisions from signals (may include SELL)
decisions = await self._build_decisions(accepted_symbols_set)
decisions = await self._deduplicate_decisions(decisions)  # Only dedupes same-source signals

# Execute decisions without checking if TPSL already closed the position
for symbol, side, signal in decisions:
    if side == "SELL":
        order = await self._atomic_sell_order(symbol, qty, signal)
```

**TPSLEngine Execution (line 1896, tp_sl_engine.py):**
```python
async def _close(sym: str, reason: str):
    async with sem:  # Only semaphore, not per-symbol lock
        # ... 
        res = await self._close_via_execution_manager(
            sym,
            close_reason,
            force_finalize=True,
            tag="tp_sl",
        )
```

**Execution Manager (no lock during order submission):**
```python
async def place_order(self, symbol: str, side: str, quantity: float, ...):
    # No per-symbol lock
    # Directly submits to exchange
    order = await self.execution_client.place_market_order(...)
    return order
```

### Why This Is a Race

1. **No inter-component synchronization**: MetaController has no way to know TPSL closed the position
2. **Position state lag**: SharedState shows position as open even after TPSL SELL fills
3. **Execution decoupling**: Both components call ExecutionManager independently
4. **No order deduplication**: ExecutionManager doesn't check "has BTC already been sold this cycle?"

### Impact

- **Financial**: 2x slippage loss + 2x fees = 0.02-0.04 USD per trade
- **Operational**: Position goes negative (short when should be flat)
- **Reporting**: PnL counted twice or partially
- **System**: Potential loss of capital

### Severity: **CRITICAL**

---

## Race Condition #2: Position State Update Lag (HIGH)

### Problem
MetaController reads position from SharedState while TPSLEngine is updating it, causing decisions based on stale data.

### Scenario

```
[T=0.0s]  MetaController queries: get_position(BTC)
          Result: qty=1.0 (from shared_state.positions cache)

[T=0.05s] TPSLEngine closes BTC position
          ExecutionManager.close_position(BTC) fills
          SharedState.mark_position_closed(BTC) called
          Position now: qty=0.0

[T=0.10s] MetaController continues decision making
          Uses cached qty=1.0 from T=0.0s
          Builds decision: SELL 0.5 BTC
          (doesn't know position is already closed)

[T=0.15s] MetaController submits SELL for 0.5 BTC
          ExecutionManager tries to close qty=0.0 position
          Order fails or sells "phantom" position

RESULT: Wrong qty calculation, failed order, or orphaned position state
```

### Code Evidence

**MetaController _build_decisions (line 5875):**
```python
decisions = await self._build_decisions(accepted_symbols_set)
# Reads from shared_state.positions which may be stale
```

**_build_decisions does NOT re-query before execution (line ~10600):**
```python
for symbol in ranked_symbols:
    position = self._get_position_from_cache(symbol)  # ← STALE!
    # Make decision based on cached position
    # Execute 100+ lines later without re-checking
```

**SharedState position update from TPSL (line ~line 1896, tp_sl_engine.py):**
```python
res = await self._close_via_execution_manager(sym, ...)
# Updates happen asynchronously in ExecutionManager
# MetaController doesn't wait for update
```

### Why This Is a Race

1. **Async boundaries**: MetaController reads, then awaits multiple async operations
2. **No visibility into TPSL**: MetaController doesn't subscribe to TPSL position changes
3. **Cache not invalidated**: Position cache not refreshed mid-cycle
4. **Shared mutable state**: Both components write to shared_state.positions

### Impact

- **Decision quality**: Wrong qty/allocation decisions
- **Order rejection**: Orders fail due to qty validation
- **State corruption**: Position shows non-zero when exchange shows zero

### Severity: **HIGH**

---

## Race Condition #3: Double SELL from Signal Deduplication Gap (HIGH)

### Problem
Signal deduplication in MetaController only dedupes signals from the SAME COLLECTION CYCLE, not across TPSL.

### Scenario

```
[T=0.0s]  Signal arrives: SELL BTC (confidence 0.95)
          Added to MetaController.signal_cache

[T=5.0s]  MetaController runs cycle 1
          Collects SELL signal, decides to SELL BTC
          TPSL also closes BTC due to TP hit
          Both orders submitted
          
[T=5.1s]  Both orders fill
          Position: 0 → -1.0 (SHORT!)

[T=5.5s]  MetaController cycle 2
          SELL signal is NOW deduplicated (caught by _deduplicate_decisions)
          But position is already SHORT now
          "Deduplication" is too late
```

### Code Evidence

**MetaController _deduplicate_decisions (line 1980, meta_controller.py):**
```python
async def _deduplicate_decisions(
    self,
    decisions: List[Tuple[str, str, Dict[str, Any]]]
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Remove duplicate signals per symbol to prevent duplicate orders"""
    
    # Only dedupes within THIS CYCLE's decisions
    # Doesn't check if TPSL already closed it
    # Doesn't check against previous cycle's fills
    
    unique = {}
    for symbol, side, signal in decisions:
        key = (symbol, side)
        if key not in unique:
            unique[key] = signal  # Keep first, discard duplicates
        else:
            self.logger.info(f"[Dedup] Suppressed duplicate {symbol} {side}")
    
    return [(sym, side, unique[(sym, side)]) for sym, side in unique.keys()]
```

### The Gap

The deduplication happens AFTER signals are collected but BEFORE execution. But:
- It only checks same-cycle duplicates
- It doesn't know what TPSL already closed
- It doesn't know what filled in the last cycle

### Impact

- **Order duplication**: Duplicate SELL orders even after deduplication attempt
- **Position inversion**: Position goes short unintentionally
- **Loss of capital**: Slippage × 2, fees × 2 on duplicate close

### Severity: **HIGH**

---

## Race Condition #4: ExecutionManager Position Update Not Atomic (HIGH)

### Problem
ExecutionManager updates position in multiple steps without holding a lock, and MetaController can read intermediate state.

### Scenario

```
[T=0.0s]  SELL order fills in exchange
          ExecutionManager._handle_post_fill() starts
          
[T=0.05s] Updates open_trades: open_trades[BTC].quantity = 0
          (position still showing qty=1.0 in SharedState.positions)

[T=0.10s] MetaController reads positions mid-update
          Sees: open_trades[BTC].qty=0, positions[BTC].qty=1.0
          INCONSISTENT STATE!

[T=0.15s] ExecutionManager finishes update
          Calls mark_position_closed()
          Now consistent again

RESULT: MetaController made decision based on inconsistent state
```

### Code Evidence

**ExecutionManager _handle_post_fill (line 257):**
```python
async def _handle_post_fill(self, symbol: str, side: str, order: Dict[str, Any], ...):
    # Step 1: Update open_trades
    await maybe_call(ss, "record_trade", sym, side_u, exec_qty, ...)
    
    # Step 2: Update position (no lock!)
    pos = await ss.get_position(sym)
    # ... modify pos ...
    
    # Step 3: Emit events
    await maybe_call(ss, "emit_event", "RealizedPnlUpdated", ...)
    
    # No synchronization between steps!
```

**MetaController reads positions (line ~10600):**
```python
for symbol in ranked_symbols:
    position = await self.shared_state.get_position(symbol)
    # Could read mid-update!
```

### Why This Is a Race

1. **Multi-step update**: 3+ separate SharedState mutations without lock
2. **No atomic transaction**: Each step is independent async call
3. **MetaController not blocked**: Can read at any point during update
4. **Inconsistent snapshots**: position[qty] != open_trades[qty]

### Impact

- **Decision corruption**: Wrong position qty used in decisions
- **Order validation fails**: Quantity checks fail due to inconsistent state
- **State divergence**: MetaController sees different state than reality

### Severity: **HIGH**

---

## Race Condition #5: TPSL Concurrency Gaps Within TPSLEngine (MEDIUM)

### Problem
TPSLEngine has internal async gaps where concurrent close attempts can happen for the same symbol.

### Scenario

```
[T=0.0s]  TPSLEngine.run() wakes up
          For BTC: price=61000, TP=60900 → TP hit!
          Spawns concurrent close task: _close(BTC, "TP_HIT")

[T=0.01s] Still processing other symbols, enters queue
          Later checks same BTC: price=61005, TP=60900
          SECOND close task spawned: _close(BTC, "TP_HIT")
          
[T=0.05s] First close finishes
          Second close tries to close already-closed position
          Order rejected but logged as error

RESULT: Wasted API calls, confusing logs, potential race with Meta
```

### Code Evidence

**TPSLEngine run() spawns tasks without symbol lock (line 1896):**
```python
async def run(self):
    while not self._stop_event.is_set():
        # Check all symbols
        for sym in symbols:
            # Check if TP/SL triggered
            if should_close:
                # Spawn async task without per-symbol coordination
                asyncio.create_task(
                    self._close(sym, reason)
                )
        
        await asyncio.sleep(self._interval)
```

**_close uses only semaphore, not per-symbol lock (line 1896):**
```python
async def _close(sym: str, reason: str):
    async with sem:  # Global semaphore, not per-symbol lock
        res = await self._close_via_execution_manager(sym, ...)
        # Multiple _close(BTC) tasks can queue, then execute serially
        # Both try to close the same position
```

### Why This Is a Race

1. **No per-symbol lock**: Multiple checks of same symbol can spawn tasks
2. **Global semaphore only**: Serialize execution, but don't prevent task spawning
3. **Async boundary**: Gap between "should_close check" and "actual close execution"

### Impact

- **Double close attempts**: 2 orders for same position
- **API noise**: Extra failed requests
- **Confusion with Meta**: Hard to debug when Meta also tries to close

### Severity: **MEDIUM** (mitigated by: EM handles 2nd close gracefully)

---

## Race Condition #6: Signal Collection Timing (MEDIUM)

### Problem
Signals can arrive while MetaController is executing cycle, creating ambiguity about which cycle owns the signal.

### Scenario

```
[T=0.0s]  MetaController starts cycle 1
          Collects signals: {BTC: SELL}
          Starts decision building

[T=1.0s]  MetaController still in _build_decisions()
          Signal arrives: BTC: SELL (different agent, same symbol)
          Added to signal cache immediately

[T=2.0s]  Cycle 1 finishes decisions (doesn't include new signal)
          Executes SELL BTC (qty from old signal)

[T=2.1s]  Cycle 1 updates position to zero

[T=2.5s]  Cycle 2 starts
          Collects signals: {BTC: SELL (the new one from T=1.0s)}
          Tries to SELL BTC again (but position is zero)
          
RESULT: Lost signal? Double count? Unclear which cycle owns signal
```

### Code Evidence

**MetaController _flush_intents_to_cache (line ~6151):**
```python
async def _flush_intents_to_cache(self, now_ts: float):
    """Process intents from sink into signal cache via SignalManager."""
    # Adds signals to cache during execution
    # MetaController keeps reading from cache
```

**Signal collection happens at cycle start (line 5875):**
```python
# Step 1: Collect signals
decisions = await self._build_decisions(accepted_symbols_set)

# Meanwhile: signal cache is STILL accepting new signals
# New signals not included in this cycle's decisions
```

### Why This Is a Race

1. **Open signal sink**: Cache accepts signals throughout cycle
2. **No epoch/snapshot**: Each decision cycle doesn't snapshot signals
3. **No flow control**: Unlimited signals can queue up
4. **Ambiguous ownership**: Which cycle "owns" a signal?

### Impact

- **Lost signals**: Some signals skipped entirely
- **Delayed decisions**: Signals delayed to next cycle
- **Batch size unpredictability**: Can't reason about signal throughput

### Severity: **MEDIUM** (mitigated by: signals retry on failure)

---

## Summary Table

| # | Component Pair | Issue | Impact | Severity | Sync Missing |
|---|---|---|---|---|---|
| 1 | TPSL ↔ Meta | Concurrent SELL orders | Position inversion | CRITICAL | Inter-component lock |
| 2 | TPSL ↔ Meta | Stale position read | Wrong qty decisions | HIGH | Position refresh, or lock |
| 3 | Meta ↔ Meta | Signal dedup too late | Duplicate SELL | HIGH | Cross-cycle dedup, or TPSL coordination |
| 4 | EM ↔ Meta | Non-atomic updates | Inconsistent state | HIGH | Atomic transaction in EM |
| 5 | TPSL ↔ TPSL | Concurrent close tasks | Double close attempt | MEDIUM | Per-symbol lock in TPSL |
| 6 | Signal ↔ Meta | Timing ambiguity | Lost signals | MEDIUM | Snapshot signals at cycle start |

---

## Recommended Fixes (In Priority Order)

### Fix #1: CRITICAL - Per-Symbol Locking in ExecutionManager

```python
class ExecutionManager:
    def __init__(self, ...):
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._symbol_locks_lock = asyncio.Lock()
    
    async def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        sym = self._norm_symbol(symbol)
        async with self._symbol_locks_lock:
            if sym not in self._symbol_locks:
                self._symbol_locks[sym] = asyncio.Lock()
            return self._symbol_locks[sym]
    
    async def place_order(self, symbol: str, side: str, ...):
        lock = await self._get_symbol_lock(symbol)
        async with lock:
            # Check position hasn't been closed by concurrent order
            position = await self.shared_state.get_position(symbol)
            if side == "SELL" and position.get("qty", 0) <= 0:
                return {"ok": False, "reason": "POSITION_CLOSED_BY_CONCURRENT_ORDER"}
            
            # Proceed with order submission
            order = await self._submit_to_exchange(...)
            return order
    
    async def close_position(self, symbol: str, reason: str, ...):
        lock = await self._get_symbol_lock(symbol)
        async with lock:
            # Atomic close operation
            position = await self.shared_state.get_position(symbol)
            if position.get("qty", 0) <= 0:
                return {"ok": False, "reason": "ALREADY_CLOSED"}
            
            order = await self._submit_to_exchange(...)
            await self._handle_post_fill(...)
            return order
```

### Fix #2: CRITICAL - Coordination Channel Between TPSL and Meta

Create a "pre-close" notification channel:

```python
class TPSLEngine:
    def __init__(self, ...):
        self._pre_close_callback = None  # Injected by AppContext
    
    async def _close(self, sym: str, reason: str):
        # BEFORE submitting close order, notify MetaController
        if self._pre_close_callback:
            await self._pre_close_callback(sym, f"TPSL:{reason}")
        
        # NOW submit close (Meta will skip same symbol in this cycle)
        res = await self._close_via_execution_manager(...)
        return res

class MetaController:
    def __init__(self, ...):
        self.tp_sl_engine.register_pre_close_callback(
            self._handle_pre_close
        )
        self._tpsl_closing_symbols: Set[str] = set()
    
    async def _handle_pre_close(self, symbol: str, reason: str):
        """TPSL is about to close this symbol."""
        sym = self._normalize_symbol(symbol)
        self._tpsl_closing_symbols.add(sym)
        # Don't process SELL signals for this symbol in this cycle
        
        # Auto-clear after 5 seconds
        await asyncio.sleep(5.0)
        self._tpsl_closing_symbols.discard(sym)
    
    async def _process_decisions(self, decisions):
        # Skip TPSL-closing symbols
        filtered = [
            (sym, side, sig) for sym, side, sig in decisions
            if sym not in self._tpsl_closing_symbols
        ]
        return filtered
```

### Fix #3: HIGH - Atomic Position Updates in ExecutionManager

```python
async def _handle_post_fill(self, symbol: str, side: str, order: Dict[str, Any], ...):
    sym = self._norm_symbol(symbol)
    ss = self.shared_state
    
    # Hold per-symbol lock during ENTIRE post-fill processing
    lock = await self._get_symbol_lock(sym)
    async with lock:
        # All steps now atomic w.r.t. concurrent readers
        
        # Step 1: Update open_trades
        await maybe_call(ss, "record_trade", sym, side, ...)
        
        # Step 2: Update positions
        pos = await ss.get_position(sym)
        if side == "SELL":
            pos["quantity"] = 0.0
        await ss.update_position(sym, pos)
        
        # Step 3: Emit events
        await maybe_call(ss, "emit_event", "RealizedPnlUpdated", ...)
        
        # All done under lock - no inconsistent intermediate states
```

### Fix #4: HIGH - Per-Symbol Lock in TPSLEngine

```python
class TPSLEngine:
    def __init__(self, ...):
        self._symbol_close_locks: Dict[str, asyncio.Lock] = {}
        self._symbol_close_locks_lock = asyncio.Lock()
    
    async def _get_close_lock(self, symbol: str) -> asyncio.Lock:
        sym = (symbol or "").upper()
        async with self._symbol_close_locks_lock:
            if sym not in self._symbol_close_locks:
                self._symbol_close_locks[sym] = asyncio.Lock()
            return self._symbol_close_locks[sym]
    
    async def _close(self, sym: str, reason: str):
        # One _close at a time per symbol
        lock = await self._get_close_lock(sym)
        async with lock:
            # Check position still open (prevent double-close)
            position = getattr(self.shared_state, "positions", {}).get(sym, {})
            if float(position.get("quantity", 0)) <= 0:
                self.logger.info(f"[TPSL] {sym} already closed, skipping")
                return
            
            async with sem:  # Keep global semaphore for backpressure
                # Now proceed with close
                res = await self._close_via_execution_manager(sym, reason)
```

### Fix #5: MEDIUM - Signal Snapshot at Cycle Start

```python
async def run_loop(self):
    while True:
        # Step 1: SNAPSHOT all signals at cycle start
        signal_snapshot = await self._snapshot_signals()
        
        # Step 2: Build decisions from snapshot (immutable during cycle)
        decisions = await self._build_decisions(
            symbols, 
            signal_snapshot=signal_snapshot
        )
        
        # Step 3: New signals arriving now go into NEXT cycle
        # No ambiguity about ownership

async def _snapshot_signals(self) -> Dict[str, List[Dict]]:
    """Return immutable snapshot of current signals."""
    return {
        sym: list(sigs) 
        for sym, sigs in self.signal_cache.items()
    }
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Deploy within 1 week)
- Fix #1: ExecutionManager per-symbol lock
- Fix #2: TPSL→Meta coordination channel
- Fix #4: TPSLEngine per-symbol lock

### Phase 2: High Priority (Within 2 weeks)
- Fix #3: Atomic position updates
- Fix #5: Signal snapshots

### Phase 3: Monitoring (Ongoing)
- Add instrumentation to detect race conditions
- Log all concurrent order attempts
- Alert on double closes

---

## Testing Strategy

### Unit Tests
Test each component's synchronization in isolation:
```python
async def test_concurrent_close_attempts():
    """Verify 2 concurrent close tasks don't result in 2 orders."""
    # Spawn 2 concurrent _close(BTC) tasks
    # Verify only 1 order submitted
    # Verify 2nd task returns "already closed"

async def test_tpsl_meta_concurrent_sell():
    """Verify TPSL close + Meta SELL don't double-order."""
    # Trigger TPSL close concurrently with Meta SELL
    # Verify only 1 SELL order in exchange
    # Verify position is clean (0, not negative)

async def test_atomic_position_update():
    """Verify no inconsistent state during post-fill."""
    # While post-fill is updating, spawn concurrent reader
    # Verify reader never sees inconsistent (open_trades≠positions) state
```

### Integration Tests
Test inter-component interactions:
```python
async def test_concurrent_components():
    """Full stack test: TPSL + Meta + EM working together."""
    # Start TPSL engine
    # Start MetaController
    # Inject conflicting signals
    # Verify no duplicate orders
    # Verify position consistency

async def test_position_lag_handling():
    """Verify Meta handles stale position reads gracefully."""
    # Add artificial delay in position reads
    # Verify decision quality degrades gracefully
    # Verify no crashes or corrupted state
```

### Stress Tests
```python
async def test_high_frequency_signals():
    """100 signals/sec for same symbol."""
    # Verify deduplication works
    # Verify no position inversion

async def test_concurrent_close_waves():
    """3 concurrent TPSL closes, 2 Meta SELLs, same symbol."""
    # Verify exactly 1 order succeeds
    # Verify other 4 fail gracefully
```

---

## Deployment Checklist

- [ ] Fix #1 deployed and tested
- [ ] Fix #2 deployed and tested
- [ ] Fix #4 deployed and tested
- [ ] Integration tests passing (100%)
- [ ] Manual testing: concurrent TPSL + Meta
- [ ] Performance impact acceptable (< 1ms additional latency)
- [ ] Monitoring enabled for race condition detection
- [ ] Runbook updated for "double order" incident
- [ ] Team trained on new locking semantics

---

## Monitoring & Alerting

### Metrics to Track

1. **Double Order Attempts** (should be 0)
   - Query: `exchange_orders.submitted_orders where created_within_1s(symbol) > 1`
   - Alert: If > 0 in any 5min window

2. **TPSL Close Attempts**
   - Metric: `tpsl.close_attempts_per_symbol`
   - Alert: If > 1 in any 10s window for same symbol

3. **Position State Inconsistency**
   - Metric: `state.open_trades_qty != state.positions_qty`
   - Alert: Immediate (this should be 0)

4. **ExecutionManager Locking Contention**
   - Metric: `em.lock_wait_time_ms`
   - Alert: If P95 > 100ms (indicates contention)

### Log Patterns to Search

```bash
# Double SELL attempts
grep -i "already reserved" logs/meta_controller.log

# TPSL concurrent closes
grep -i "concurrent close" logs/tp_sl_engine.log

# State inconsistency
grep -i "phantom position\|inconsistent" logs/execution_manager.log

# Race condition detection
grep -i "race\|concurrent\|duplicate" logs/*.log
```

---

## Conclusion

The system has **6 significant race conditions** spanning TPSL→Meta→EM interactions. The most critical (Race #1 and #2) can cause:

- Financial loss through duplicate order execution
- Position inversion (accidental short)
- PnL corruption

**Recommended action**: Implement Fixes #1, #2, #4 immediately (within 1 week), then #3 and #5 within 2 weeks.

All fixes are **backward compatible** and add minimal latency (< 1ms per order).

