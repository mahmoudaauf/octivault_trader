# 🔴 CRITICAL BUG: Signal Pipeline Breakage - Root Cause Analysis

## Executive Summary

The signal pipeline is **broken at the signal cache layer**. Signals are being **generated and buffered** by TrendHunter but are **NOT reaching the Meta signal cache**, resulting in `decisions_count=0` every tick.

```
TrendHunter generates signals ✅
    ↓
Buffered in _collected_signals ✅
    ↓
Returned from generate_signals() ✅
    ↓
Normalized to TradeIntent ✅
    ↓
Published to event bus ✅ (or forwarded directly ✅)
    ↓
🔴 SIGNALS NEVER REACH SIGNAL CACHE
    ↓
_build_decisions() queries empty cache ❌
    ↓
decisions_count=0 every tick ❌
    ↓
No trades execute ❌
```

---

## Evidence from Logs

### Signal Generation Working
```
2026-03-03 21:59:26,639 - INFO - [TrendHunter] Buffered BUY for BTCUSDT (conf=0.70, ...)
2026-03-03 21:59:26,640 - INFO - [TrendHunter] Buffered BUY for ETHUSDT (conf=0.70, ...)
```
✅ TrendHunter is successfully generating signals

### Signal Cache Empty
```
2026-03-03 21:58:54,700 - WARNING - [Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
2026-03-03 21:58:56,780 - WARNING - [Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
```
❌ Despite TrendHunter buffering signals, Meta signal cache is ALWAYS empty

### Decisions Never Built
```
2026-03-03 21:58:54,700 - WARNING - [Meta:POST_BUILD] decisions_count=0 decisions=[]
2026-03-03 21:58:56,780 - WARNING - [Meta:POST_BUILD] decisions_count=0 decisions=[]
...
2026-03-03 22:00:37,910 - WARNING - [Meta:POST_BUILD] decisions_count=0 decisions=[]
```
❌ Every single cycle returns empty decisions list

### Missing Bridge Logs
```
NO LOGS FOR:
- "[AgentManager] Published %d trade intent events" (event_bus publish)
- "[AgentManager:BATCH] Submitted batch of..." (event bus forwarding)
- "[AgentManager:DIRECT] Forwarded %d signals directly" (direct path)
- "[Meta:EventBus] Drained %d trade intents" (event drain)
- "[IntentManager] Received %d intents" (intent reception)
- "[SignalManager] Signal ACCEPTED and cached" (signal caching)
```
❌ All critical bridge logs are missing!

---

## Root Cause Investigation

### Data Flow Path #1: Event Bus Route

```
TrendHunter.generate_signals()
    ↓ returns List[Dict]
AgentManager.collect_and_forward_signals()
    ├─ Line 399: fn = getattr(agent, "generate_signals")
    ├─ Line 400-401: res = fn(); if isawaitable: res = await res
    ├─ Line 402-408: Count raw signals
    ├─ Line 409: intents = self._normalize_to_intents(name, res)
    │   └─ Should convert List[Dict] → List[TradeIntent]
    ├─ Line 410-411: if intents: batch.extend(intents)
    │
    └─ IF BATCH NOT EMPTY (line 450):
        ├─ Line 449: await self.submit_trade_intents(batch)
        │   └─ Event bus publish route
        │
        └─ EXPECTED: Log "[AgentManager] Published %d trade intent events"
```

❌ **THIS LOG IS MISSING** → `submit_trade_intents()` is never called OR batch is empty

### Data Flow Path #2: Direct MetaController Route

```
AgentManager.collect_and_forward_signals()
    └─ Line 458-481: Direct path to MetaController
        ├─ if self.meta_controller:
        ├─ for intent in batch:
        │   └─ await self.meta_controller.receive_signal(agent, symbol, signal)
        │
        └─ EXPECTED: Log "[AgentManager:DIRECT] Forwarded %d signals directly to MetaController.signal_cache"
```

❌ **THIS LOG IS MISSING** → Direct path either not triggered OR code not executed

### Theory #1: `batch` is Empty

**Hypothesis:** `_normalize_to_intents()` is failing to convert signals

**Evidence Against:**
- Log shows: `[TrendHunter] Buffered BUY for BTCUSDT` → `generate_signals()` is returning data
- If normalization failed, should see: `FAILED to normalize any of the %d signals` (line 424)
- ❌ This log is also missing

**Conclusion:** Either normalization is silently failing OR the check `if intents:` is False

### Theory #2: Event Bus Subscription/Drain Not Working

**Hypothesis:** Signals are published but not being drained

**Evidence:**
- No logs for `[Meta:DRAIN] ⚠️ DRAINED %d events`
- Subscription IS working: `[Meta:EventBus] Subscribed to events.trade.intent as ...`
- But no subsequent drains logged

**Question:** Is `_drain_trade_intent_events()` actually being called in `evaluate_and_act()`?

**Code Location:** `meta_controller.py` line 5840-5851

```python
try:
    drained = await self._drain_trade_intent_events(
        max_items=int(self._cfg("TRADE_INTENT_EVENT_DRAIN_MAX", 1000) or 1000)
    )
    # 🔥 CRITICAL DEBUG: After drain
    self.logger.warning("[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus", drained)
```

❌ The warning `[Meta:DRAIN]` is NOT in logs → `_drain_trade_intent_events()` either not called OR drained=0 silently

### Theory #3: Direct Path Not Triggered

**Hypothesis:** Line 458 check `if self.meta_controller:` is False

**Evidence:**
- This would prevent direct signal forwarding
- But it wouldn't prevent event_bus publishing (line 449 is before this check)

**Check:**
- Is AgentManager.meta_controller properly initialized?
- Logs don't show initialization failures for AgentManager

---

## The Critical Missing Breadcrumb

The AgentManager has a debug log at line 453 that should always fire if batch is non-empty:

```python
if batch:
    await self.submit_trade_intents(batch)
    self.logger.info("Submitted %d TradeIntents to Meta", len(batch))
    
    # 🔥 CRITICAL DEBUG: Log submission
    self.logger.warning("[AgentManager:BATCH] Submitted batch of %d intents: %s", 
                       len(batch),
                       [f"{i.get('agent')}:{i.get('symbol')}" for i in batch])
```

❌ **Neither of these logs appear** → `batch` must be empty

This means line 409 (`intents = self._normalize_to_intents(name, res)`) is returning None or empty list.

---

## Probable Root Cause: Signal Normalization Failure

### Code Under Suspicion

**File:** `core/agent_manager.py` line 323-346

```python
def _normalize_to_intents(self, agent_name: str, raw_signals) -> List[Dict]:
    """Convert agent-generated signals to canonical TradeIntent format."""
    if not raw_signals:
        return None  # ❌ SUSPICIOUS: Returns None not []
    
    if isinstance(raw_signals, dict):
        raw_signals = [raw_signals]
    elif not isinstance(raw_signals, (list, tuple, set)):
        return None  # ❌ SUSPICIOUS: Returns None not []
    
    intents = []
    for sig in raw_signals:
        if not isinstance(sig, dict):
            self.logger.debug("Skipping non-dict signal from %s: %s", agent_name, type(sig))
            continue
        
        # Build canonical TradeIntent
        intent = {
            "symbol": (sig.get("symbol") or sig.get("sym") or "").upper(),
            "action": (sig.get("action") or sig.get("side") or "").upper(),
            "confidence": float(sig.get("confidence") or 0.0),
            "agent": agent_name,
            "reason": sig.get("reason", ""),
            "quote": float(sig.get("quote") or sig.get("quote_hint") or 0.0),
            "timestamp": sig.get("timestamp", time.time()),
            ...
        }
        
        # Validate
        if not intent.get("symbol") or intent["action"] not in ("BUY", "SELL"):
            self.logger.debug("Skipping invalid intent: %s %s", intent["symbol"], intent["action"])
            continue
        
        intents.append(intent)
    
    return intents  # ← Returns list (could be empty!)
```

**Problem 1:** Line 325 returns `None` instead of `[]`
- Line 428: `if intents:` treats None as falsy ✓ (correct)
- But line 412: `elif res:` would be true if res is a list
- This creates ambiguity

**Problem 2:** No logging when intents list is empty
- Could silently return empty list without any debug output

**Solution:** Add explicit logging to diagnose the issue

---

## Diagnostic Steps Needed

### Step 1: Verify Signal Generation

```bash
grep -a "generate_signals() returned" logs/clean_run.log
# Should show: "[TrendHunter] generate_signals() returned X raw signals"
```

### Step 2: Verify Signal Normalization

```bash
grep -a "_normalize_to_intents" logs/clean_run.log
# Should show: "Normalized X intents" OR "FAILED to normalize"
# Missing = normalization code has no logs
```

### Step 3: Verify Event Bus Publishing

```bash
grep -a "Published %d trade intent events" logs/clean_run.log
# Should show: "[AgentManager] Published X trade intent events"
# Missing = signals not published
```

### Step 4: Verify Direct Path

```bash
grep -a "DIRECT] Forwarded" logs/clean_run.log
# Should show: "[AgentManager:DIRECT] Forwarded X signals"
# Missing = direct path not taken
```

### Step 5: Verify Meta Signal Caching

```bash
grep -a "Signal ACCEPTED and cached" logs/clean_run.log
# Should show: "[SignalManager] Signal ACCEPTED and cached: BTCUSDT from TrendHunter"
# Missing = signals never reach signal_manager
```

---

## The Fix Required

### Fix #1: Add Diagnostic Logging to `_normalize_to_intents()`

**File:** `core/agent_manager.py` line 323

```python
def _normalize_to_intents(self, agent_name: str, raw_signals) -> List[Dict]:
    """Convert agent-generated signals to canonical TradeIntent format."""
    self.logger.warning("[AgentManager:NORMALIZE] Normalizing %d raw signals from %s", 
                       len(raw_signals) if raw_signals else 0, agent_name)
    
    if not raw_signals:
        self.logger.debug("[AgentManager:NORMALIZE] Empty/None raw_signals from %s", agent_name)
        return []  # ← CHANGE: None → []
    
    # ... rest of normalization code ...
    
    if not intents:
        self.logger.warning("[AgentManager:NORMALIZE] Normalized 0 intents from %s (raw had %d signals)", 
                           agent_name, len(raw_signals) if isinstance(raw_signals, (list, tuple)) else 1)
    else:
        self.logger.info("[AgentManager:NORMALIZE] Normalized %d intents from %s", len(intents), agent_name)
    
    return intents
```

### Fix #2: Add Logging to `submit_trade_intents()`

**File:** `core/agent_manager.py` line 255

```python
async def submit_trade_intents(self, intents: List[Dict[str, Any]]):
    """Bind Agent→Meta pipe through the event bus."""
    if not intents:
        self.logger.warning("[AgentManager:SUBMIT] submit_trade_intents called with empty list")
        return
    
    self.logger.warning("[AgentManager:SUBMIT] Publishing %d intents to event_bus", len(intents))
    
    # ... rest of code ...
```

### Fix #3: Verify MetaController.receive_signal() is Logging

**File:** `core/meta_controller.py` line 5044

```python
async def receive_signal(self, agent_name: str, symbol: str, signal: Dict[str, Any]):
    """Accept and cache signals with delegation to SignalManager."""
    self.logger.warning("[MetaController:RECV_SIGNAL] Received signal for %s from %s", symbol, agent_name)
    
    if not symbol or not isinstance(signal, dict):
        self.logger.warning("Invalid signal received: symbol=%s signal=%s", symbol, signal)
        return

    # Use SignalManager for core intake logic
    success = self.signal_manager.receive_signal(agent_name, symbol, signal)
    if not success:
        self.logger.warning("[MetaController:RECV_SIGNAL] SignalManager rejected signal for %s from %s", 
                           symbol, agent_name)
        return
    
    self.logger.info("[MetaController:RECV_SIGNAL] ✓ Signal cached for %s from %s", symbol, agent_name)
```

---

## Expected vs Actual

### Expected Logs (when TrendHunter has signals)
```
[TrendHunter] Buffered BUY for BTCUSDT
[TrendHunter] generate_signals() returned 2 raw signals
[AgentManager] generate_signals() returned 2 raw signals
[AgentManager:NORMALIZE] Normalizing 2 raw signals from TrendHunter
[AgentManager:NORMALIZE] Normalized 2 intents from TrendHunter
[AgentManager:BATCH] Submitted batch of 2 intents: TrendHunter:BTCUSDT, TrendHunter:ETHUSDT
[AgentManager:SUBMIT] Publishing 2 intents to event_bus
[AgentManager] Published 2 trade intent events
[AgentManager:DIRECT] Forwarded 2 signals directly to MetaController.signal_cache
[MetaController:RECV_SIGNAL] Received signal for BTCUSDT from TrendHunter
[MetaController:RECV_SIGNAL] ✓ Signal cached for BTCUSDT from TrendHunter
[MetaController:RECV_SIGNAL] Received signal for ETHUSDT from TrendHunter
[MetaController:RECV_SIGNAL] ✓ Signal cached for ETHUSDT from TrendHunter
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 2 signals: [BTCUSDT, ETHUSDT]
[Meta:POST_BUILD] decisions_count=2 decisions=[(BTCUSDT, BUY, ...), (ETHUSDT, BUY, ...)]
```

### Actual Logs
```
[TrendHunter] Buffered BUY for BTCUSDT
[TrendHunter] Buffered BUY for ETHUSDT
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
[Meta:POST_BUILD] decisions_count=0 decisions=[]
```

❌ **EVERYTHING BETWEEN TrendHunter buffering and Meta decision-building is missing**

---

## Next Steps

1. **Add the diagnostic logging** per Fix #1-3 above
2. **Re-run the clean_run test**
3. **Check logs for the new bridge logs** to identify exactly where signals are being lost
4. **Fix the broken link** based on which logs are missing

The signal pipeline is like a water pipe with a leak somewhere between TrendHunter output and Meta input. We need to turn on all the valves to see where the water stops flowing.
