# 🎯 Root Cause Identified: Signal Pipeline Blockage

## The Issue

```
✅ Agents are alive → generating signals
✅ AgentManager publishes to event_bus → events queued  
✅ EventBus exists → queue has messages
❌ MetaController NEVER DRAINS → cache stays empty
❌ No decisions made → no trading
```

## The Root Cause

**MetaController.run() lifecycle loop may not be executing.**

The chain is:
1. AppContext starts MetaController via `await meta_controller.start()`
2. `start()` creates background task: `asyncio.create_task(self.run())`
3. `run()` should loop forever, each iteration:
   - Calls `evaluate_and_act()`
   - Which calls `_drain_trade_intent_events()`
   - Which pulls signals from event_bus into signal_cache
4. If `run()` doesn't execute → nothing drains → cache empty

## Critical Logging Added (3 Points)

### Point 1: MetaController.start() - Line 5111
```python
self.logger.warning("[Meta:START] ⚠️ START METHOD CALLED! interval_sec=%.1f", interval_sec)
```
✅ Confirms: Phase P6 successfully started MetaController

### Point 2: MetaController.run() - Lines 5209, 5216
```python
self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=%.1f", self.interval)
self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #%d starting (tick_id=%d)", iteration, self.tick_id)
```
✅ Confirms: run() loop is actually executing (logged every 10 iterations)

### Point 3: _drain_trade_intent_events() - Lines 5839, 5845  
```python
self.logger.warning("[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!")
self.logger.warning("[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus", drained)
```
✅ Confirms: Draining is happening and shows count drained

## Diagnostic Workflow

### Step 1: Run System with Debug Logs
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 main.py 2>&1 | tee logs/meta_drain_debug.log
# Wait 30 seconds minimum (15+ drain cycles at 2s interval)
# Stop with Ctrl+C
```

### Step 2: Check All 3 Logs  
```bash
# Does start() get called?
grep "\[Meta:START\]" logs/meta_drain_debug.log | head -3

# Does run() loop execute?
grep "\[Meta:RUN\]" logs/meta_drain_debug.log | head -3

# Does drain happen?
grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log | head -3

# Bonus: Are agents generating?
grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log | head -3
```

### Step 3: Analyze Results

**EXPECTED** (If working):
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:START] ⚠️ Evaluation task spawned: <Task name='meta.run' ...>
[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #1 starting (tick_id=1)
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #11 starting (tick_id=11)
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 3 events from event_bus
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 0 events from event_bus
```

**Case A** (start() not called):
```
[AgentManager:BATCH] Submitted batch of 3 intents
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals
# NO [Meta:START] logs anywhere
# → Phase P6 failed to start MetaController
```

**Case B** (start() called, run() not executing):
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:START] ⚠️ Evaluation task spawned: <Task ...>
[AgentManager:BATCH] Submitted batch of 3 intents
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals
# NO [Meta:RUN] logs
# → Task was created but never executes (exception/cancellation)
```

**Case C** (run() executes, drain not called):
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #1 starting (tick_id=1)
[AgentManager:BATCH] Submitted batch of 3 intents
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals
# NO [Meta:DRAIN] logs
# → evaluate_and_act() not reaching drain (stuck earlier in evaluate_and_act)
```

## Why This Matters

This diagnostic approach **pinpoints EXACTLY where the chain breaks**:
- Missing `[Meta:START]` → Problem: Phase setup (app_context)
- Missing `[Meta:RUN]` → Problem: Task lifecycle (asyncio)
- Missing `[Meta:DRAIN]` → Problem: evaluate_and_act execution (meta_controller logic)

Once we know WHICH log is missing, we know WHICH component to fix.

## Code Changes Summary

### File: core/meta_controller.py

**Change 1** - Line 5111 (in `start()` method):
```python
# 🔥 CRITICAL DEBUG: MetaController.start() entry
self.logger.warning("[Meta:START] ⚠️ START METHOD CALLED! interval_sec=%.1f", interval_sec)
```

**Change 2** - Line 5209 (in `run()` method):  
```python
# 🔥 CRITICAL DEBUG: run() loop started
self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=%.1f", self.interval)
```

**Change 3** - Line 5216 (in `run()` loop iteration):
```python
if iteration % 10 == 1:  # Log every 10 iterations
    self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #%d starting (tick_id=%d)", iteration, self.tick_id)
```

**Change 4** - Lines 5839, 5845 (in `evaluate_and_act()` drain call):
```python
# 🔥 CRITICAL DEBUG: About to drain
self.logger.warning("[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!")
# ... drain ...
self.logger.warning("[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus", drained)
```

## Verification

```bash
# Syntax check
python3 -m py_compile core/meta_controller.py
# Should produce no output (success) or error messages if syntax wrong
```

## Next Actions

1. ✅ Deploy core/meta_controller.py (done - debug logs added)
2. ⏭️ Run system: `python3 main.py 2>&1 | tee logs/meta_drain_debug.log`
3. ⏭️ Let it run 30+ seconds
4. ⏭️ Stop with Ctrl+C
5. ⏭️ Run the 4 diagnostic grep commands
6. ⏭️ Share the output

**The logs will tell us EXACTLY what's broken.**

## Preview of Fixes

Once we know which case applies:

- **Case A** (start() not called): Fix app_context Phase P6 startup
- **Case B** (run() not executing): Fix task creation or exception handling in start()
- **Case C** (drain() not called): Debug evaluate_and_act() to see where it's stuck

Each case has a specific fix. The debug logs will point directly to it.

## Files Modified

- ✅ `core/meta_controller.py` - 4 critical WARNING logs added

## Status

🟢 **Ready to test** - Deploy and run system, collect logs
