# 🔍 DEBUG: Signal Pipeline Breakdown

## The Broken Pipeline

```
Agent → generate_signals() → _submit_signal() → buffer _collected_signals[]
  ↓
  return _collected_signals[] from generate_signals()
  ↓
AgentManager.collect_and_forward_signals() receives signals
  ↓
_normalize_to_intents() converts to TradeIntent
  ↓
submit_trade_intents() publishes to event_bus("events.trade.intent")
  ↓
MetaController._drain_trade_intent_events() reads from event bus
  ↓
intent_manager.receive_intents() stores intents
  ↓
_flush_intents_to_cache() calls signal_manager.flush_intents_to_cache()
  ↓
SignalManager stores signals in signal_cache
  ↓
_build_decisions() reads from signal_cache via signal_manager.get_all_signals()
```

## Potential Breakpoints

### 1. Agent Signal Generation
**Location:** `agents/trend_hunter.py:335`
```python
async def generate_signals(self) -> List[Any]:
    self._collected_signals = []
    await self.load_symbols()
    await self.run_once()
    return self._collected_signals  # Are signals here?
```

**Check:** Is `_submit_signal()` actually appending to `_collected_signals`?
**Line:** Line 760 in trend_hunter.py

### 2. AgentManager Collection
**Location:** `core/agent_manager.py:374-470`

**Problem Areas:**
- Line 381: Filter agents by `agent_type != "discovery"` and `hasattr(agent, "generate_signals")`
- Line 412: Call `generate_signals()` and check result
- Line 424: Log raw count
- Line 427: Normalize intents

**Check:** 
- Are strategy agents registered?
- Does `generate_signals()` return non-empty list?
- Does `_normalize_to_intents()` work?

### 3. Event Bus Publishing
**Location:** `core/agent_manager.py:260-280`

**Problem:** 
```python
event_bus = getattr(self.shared_state, "event_bus", None)
publish = getattr(event_bus, "publish", None)
if not callable(publish):
    logger.warning("...publish is unavailable")
    return
```

**Check:** 
- Is `shared_state.event_bus` available?
- Can we publish to it?

### 4. MetaController Event Drain
**Location:** `core/meta_controller.py:5827`

**Check:**
- Is `_drain_trade_intent_events()` being called?
- Is the event bus queue populated?
- Are events being drained successfully?

### 5. Intent Manager Storage
**Location:** `core/meta_controller.py:5025`

**Check:**
- Does `intent_manager.receive_intents()` store intents?
- Are intents queued for flush?

### 6. Signal Manager Cache Flush
**Location:** `core/meta_controller.py:5830` / `core/signal_manager.py:196-265`

**Check:**
- Does `flush_intents_to_cache()` convert intents to signals?
- Are signals actually stored in `signal_cache`?

### 7. Signal Retrieval
**Location:** `core/meta_controller.py:9216`

```python
all_signals = self.signal_manager.get_all_signals()
```

**Check:**
- Does `get_all_signals()` return non-empty list?
- Are signals in the cache?

## Root Cause Hypothesis

Based on the code review, the **MOST LIKELY** issue is:

### **Issue: Signals buffered in `_collected_signals` but not returned from `generate_signals()`**

The agents append to `_collected_signals[]` but there might be an exception or early return that prevents them from being returned to `collect_and_forward_signals()`.

### **Alternative: Event bus publish failing silently**

The `shared_state.event_bus.publish()` might not be available (logging warning but not failing loudly).

## Diagnostic Commands

### 1. Check if agents are generating signals
```python
# In logging, look for:
# "[AgentName] Normalized X intents (scanned Y symbols)"
# If you don't see this, agents aren't generating
```

### 2. Check if event bus is working
```python
# Look for:
# "[AgentManager] emit_to_meta failed for X: ..."
# "[AgentManager] Published Y trade intent events"
# If second one is missing, event bus publish failed
```

### 3. Check if MetaController is draining
```python
# Look for:
# "[Meta:EventBus] Subscribed to events.trade.intent as ..."
# "[Meta:EventBus] Drained N trade intents from events.trade.intent"
# If second is missing/zero, drain isn't working
```

### 4. Check if signals are in cache
```python
# At _build_decisions start:
# "all_signals = self.signal_manager.get_all_signals()"
# Look for:
# "signals_by_sym populated: N symbols with M signals"
# If zero, cache is empty
```

## Next Steps

1. Enable DEBUG logging for:
   - `[AgentManager]`
   - `[Meta:EventBus]`
   - `[SignalManager]`

2. Run one trading cycle and collect logs

3. Trace the signal through each step

4. Identify where it gets lost
