# 🔥 ROOT CAUSE: MetaController Never Drains Event Bus

## Current Symptom

```
✅ Agents are alive
✅ AgentManager publishes to event_bus
✅ EventBus exists and has messages
❌ Meta never drains → Signal cache stays empty
```

## Root Cause Chain

### Level 1: What Should Happen
1. AgentManager.collect_and_forward_signals() publishes to `events.trade.intent`
2. MetaController.run() lifecycle loop continuously calls `_drain_trade_intent_events()`
3. Drained events are converted to signals and stored in signal_cache
4. MetaController._build_decisions() reads signal_cache and makes decisions

### Level 2: What IS Happening
1. ✅ AgentManager publishes events (verified in logs)
2. ✅ event_bus has queue with messages
3. ❌ **MetaController.run() is NOT calling _drain_trade_intent_events()**
4. ❌ Signal cache stays empty forever

## Why Meta is Not Draining

### The Execution Path

```
AppContext
├─ Phase P6: call await _start_with_timeout("P6_meta_controller", self.meta_controller)
├─ This calls: await meta_controller.start()
│  ├─ Sets self._running = True
│  ├─ Spawns task: self._eval_task = asyncio.create_task(self.run())
│  └─ ✅ Returns (start() completes)
│
└─ Meanwhile, self.run() SHOULD be running in background
   ├─ Enters while not self._stop and self._running: loop
   ├─ Each iteration calls: await self.evaluate_and_act()
   │  ├─ Calls: await self._drain_trade_intent_events()
   │  └─ Should return num_drained
   └─ Sleeps interval seconds, repeat
```

### The Problem

**The question is: Is the run() loop actually executing?**

Three possibilities:
1. ❌ `start()` is never called (phase boot issue)
2. ❌ `start()` is called but task creation fails (exception during task creation)
3. ❌ `start()` completes but `run()` task is not actually running (task stuck/cancelled)

## Critical Logging Added

I've added CRITICAL WARNING logs at every step:

### MetaController.start() (Line ~5110)
```python
# 🔥 CRITICAL DEBUG: MetaController.start() entry
self.logger.warning("[Meta:START] ⚠️ START METHOD CALLED! interval_sec=%.1f", interval_sec)
# ... create tasks ...
self.logger.warning("[Meta:START] ⚠️ Evaluation task spawned: %s", self._eval_task)
```

**Expected log output**: 
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:START] ⚠️ Evaluation task spawned: <Task ...>
```

### MetaController.run() (Line ~5213)
```python
# 🔥 CRITICAL DEBUG: run() loop started
self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=%.1f", self.interval)
# ... each iteration ...
if iteration % 10 == 1:  # Every 10 iterations
    self.logger.warning("[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #%d starting (tick_id=%d)", iteration, self.tick_id)
```

**Expected log output** (every 10 iterations):
```
[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #1 starting (tick_id=1)
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #11 starting (tick_id=11)
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #21 starting (tick_id=21)
```

### MetaController._drain_trade_intent_events() (Line ~5844)
```python
# 🔥 CRITICAL DEBUG: About to drain
self.logger.warning("[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!")
# ... drain ...
self.logger.warning("[Meta:DRAIN] ⚠️ DRAINED %d events from event_bus", drained)
```

**Expected log output** (every 2 seconds):
```
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 0 events from event_bus
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 0 events from event_bus
```

Or if signals are being published:
```
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 5 events from event_bus
```

## How to Diagnose

**Run the system and immediately start searching for these logs:**

```bash
# Terminal 1: Run the system
python3 main.py 2>&1 | tee logs/meta_drain_debug.log

# Terminal 2 (while system runs): Check for logs
grep "\[Meta:START\]" logs/meta_drain_debug.log
grep "\[Meta:RUN\]" logs/meta_drain_debug.log
grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log
grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log
```

## Diagnostic Decision Tree

### Does `[Meta:START]` appear in logs?

**IF YES** → start() was called, proceed to next question
**IF NO** → Phase P6 failed to start MetaController. Check app_context.py phase setup

### Does `[Meta:RUN]` appear in logs?

**IF YES** → run() loop is executing! Check the drain logs
**IF NO** → run() task was created but never executed. Possible causes:
  - Task was cancelled immediately
  - Exception during task startup (check exception logs around `[Meta:START]` time)
  - Event loop not running

### Does `[Meta:DRAIN]` appear in logs?

**IF YES, with drained count > 0** → ✅ WORKING! Signals are flowing!
**IF YES, with drained count = 0** → Loop is running but no signals published. Check AgentManager logs
**IF NO** → evaluate_and_act() never called (shouldn't happen if RUN shows)

### Does `[AgentManager:BATCH]` appear in logs?

**IF YES** → Agents ARE generating signals, they're just not reaching Meta
**IF NO** → Agents are NOT generating signals (upstream issue)

## Expected vs Actual

### EXPECTED (Working System)

```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:START] ⚠️ Evaluation task spawned: <Task name='meta.run' coro=...>
[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #1 starting (tick_id=1)
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[AgentManager:BATCH] Submitted batch of 3 intents: TrendHunter:BTCUSDT, DipSniper:ETHUSDT, ...
[Meta:DRAIN] ⚠️ DRAINED 3 events from event_bus
[Meta] decisions_count=3 decisions=[...]
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #11 starting (tick_id=11)
[Meta:DRAIN] ⚠️ ABOUT TO DRAIN TRADE INTENT EVENTS!
[Meta:DRAIN] ⚠️ DRAINED 0 events from event_bus
```

### ACTUAL (Broken System)

One of these three patterns:

**Pattern A: start() never called**
```
[AgentManager:BATCH] Submitted batch of 3 intents: ...
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
(NO [Meta:START] ever appears)
```

**Pattern B: run() never started**
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:START] ⚠️ Evaluation task spawned: <Task ...>
[AgentManager:BATCH] Submitted batch of 3 intents: ...
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
(NO [Meta:RUN] ever appears)
```

**Pattern C: drain() never called**
```
[Meta:START] ⚠️ START METHOD CALLED! interval_sec=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING! interval=2.0
[Meta:RUN] ⚠️ LIFECYCLE LOOP ITERATION #1 starting (tick_id=1)
[AgentManager:BATCH] Submitted batch of 3 intents: ...
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []
(NO [Meta:DRAIN] ever appears)
```

## Code Status

### Files Modified
- `core/meta_controller.py`
  - Line ~5110: Added WARNING log to `start()`
  - Line ~5213: Added WARNING logs to `run()` lifecycle loop
  - Line ~5844: Added WARNING logs before/after drain

### Syntax Check
```bash
python3 -m py_compile core/meta_controller.py
# Should print nothing (success) or show line numbers if errors
```

## What This Proves

Once you run this with the new debug logs:

1. **If `[Meta:START]` + `[Meta:RUN]` + `[Meta:DRAIN]` all appear** → System works, just need to verify signal flow
2. **If `[Meta:START]` + `[Meta:RUN]` appear but NOT `[Meta:DRAIN]`** → Problem in evaluate_and_act() itself
3. **If `[Meta:START]` appears but NOT `[Meta:RUN]`** → Task creation failed or task is stuck
4. **If NO `[Meta:START]`** → Phase P6 never started MetaController

## Next Steps

1. Deploy updated `core/meta_controller.py` (debug logs added)
2. Run system: `python3 main.py 2>&1 | tee logs/meta_drain_debug.log`
3. Let it run for 30 seconds (at least 15 cycles if interval=2s)
4. Stop with Ctrl+C
5. Run diagnostic greps:
   ```bash
   grep "\[Meta:START\]" logs/meta_drain_debug.log | head -5
   grep "\[Meta:RUN\]" logs/meta_drain_debug.log | head -5
   grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log | head -5
   grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log | head -5
   ```
6. Share the output of these 4 grep commands

The logs will tell us EXACTLY where the chain breaks.

## Summary

**The direct path fix is correct and in place.**

**But it won't help if MetaController.run() is never executing.**

Once we know IF and WHERE the lifecycle loop is running, we can fix the real issue.

The debug logs I added will definitively answer: **Is MetaController.run() even running?**
