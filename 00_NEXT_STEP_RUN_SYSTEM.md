# 📋 Summary: Why Signals Don't Reach MetaController

## The Chain of Custody

Signal must flow through these steps to work:

```
Agent generates signal
    ↓
AgentManager.collect_and_forward_signals()
    ↓
event_bus.publish("events.trade.intent", signal)  ✅ HAPPENING
    ↓
[Signal sits in event_bus queue]  ✅ VERIFIED
    ↓
MetaController.run() lifecycle loop
    ↓
evaluate_and_act()
    ↓
_drain_trade_intent_events()  ❌ NOT HAPPENING
    ↓
Signal removed from event_bus, added to signal_cache
    ↓
_build_decisions() reads signal_cache
    ↓
Decisions made, trades executed
```

## Why It Breaks

The problem is at the **MetaController.run() lifecycle loop**.

**Three possible failure points:**

1. **start() never called** → loop never spawned
2. **start() called but task fails** → loop created but crashes/cancelled
3. **start() succeeds but loop stuck** → draining not reached

We don't know which one without seeing the logs.

## Solution: Add Diagnostic Logging

I've added CRITICAL WARNING logs at each point:

✅ Line 5111: `[Meta:START] ⚠️ START METHOD CALLED!` - Confirms start() executes
✅ Lines 5209+5216: `[Meta:RUN] ⚠️ LIFECYCLE LOOP STARTING!` - Confirms run() executes  
✅ Lines 5839+5845: `[Meta:DRAIN] ⚠️ ABOUT TO DRAIN...` - Confirms draining happens

## What To Do Now

1. **Deploy the updated code** (already done - meta_controller.py has debug logs)

2. **Run the system:**
   ```bash
   python3 main.py 2>&1 | tee logs/meta_drain_debug.log
   ```

3. **Wait 30 seconds**, then stop with Ctrl+C

4. **Run these 4 diagnostic greps:**
   ```bash
   grep "\[Meta:START\]" logs/meta_drain_debug.log | head
   grep "\[Meta:RUN\]" logs/meta_drain_debug.log | head
   grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log | head
   grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log | head
   ```

5. **Share the output** - It will show EXACTLY where the chain breaks

## What The Logs Mean

| Log | Meaning |
|-----|---------|
| `[Meta:START]` appears | ✅ Phase P6 worked - MetaController started |
| `[Meta:RUN]` appears | ✅ Lifecycle loop is executing |
| `[Meta:DRAIN]` appears | ✅ Draining is happening |
| `[AgentManager:BATCH]` appears | ✅ Agents generating signals |

**If all 4 appear** → Everything is working! Check signal_cache downstream issue
**If 3 appear, 1 missing** → That's the broken component

## Current Best Guess

Based on:
- ✅ Agents ARE generating (AgentManager logs show)
- ✅ Event bus HAS messages (verified in code)
- ✅ Signal cache EMPTY (confirmed in bootstrap debug)

**Most likely: MetaController.start() is called BUT run() loop is not executing**

This could be:
- Exception during task creation
- Task cancelled immediately
- Exception in run() on first iteration

The logs will tell us which.

## Previous Work Done

✅ **Direct path fix** (lines 460-480 in agent_manager.py) - Signal forwarding bypass implemented
✅ **AgentManager debug logs** (lines 384, 448 in agent_manager.py) - Agent status tracking
✅ **MetaController start() debug logs** (line 5111) - Startup confirmation  
✅ **MetaController run() debug logs** (lines 5209, 5216) - Loop execution tracking
✅ **MetaController drain() debug logs** (lines 5839, 5845) - Drain confirmation

## Files with Changes

1. `core/agent_manager.py` - Has debug logs + direct path
2. `core/meta_controller.py` - Has debug logs for lifecycle + drain

Both files are ready to deploy.

## Expected Timeline

1. **Now**: Deploy code (5 minutes)
2. **Run & collect logs**: 30-60 seconds
3. **Analyze**: 5 minutes to identify exact issue
4. **Fix**: 15-30 minutes depending on root cause
5. **Verify**: 30 seconds to confirm signals flowing

## The Smoking Gun

Once we see which log is MISSING, we know exactly what's broken:

- No `[Meta:START]` → Fix app_context Phase P6
- No `[Meta:RUN]` → Fix MetaController.start() task creation
- No `[Meta:DRAIN]` → Fix evaluate_and_act() body

Each case has a known fix.

## Let's Go

1. Run the system
2. Capture the logs
3. Run the 4 grep commands
4. Share output

**The logs will be the smoking gun. I'll fix it immediately after.**
