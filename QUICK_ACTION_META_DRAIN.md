# ⚡ Quick Action: Find Why Meta Never Drains

## The Question

**Is MetaController.run() lifecycle loop actually executing?**

## The Answer (4 Simple Greps)

```bash
# After running system for 30 seconds, stop it (Ctrl+C)
# Then run these 4 commands:

grep "\[Meta:START\]" logs/meta_drain_debug.log
grep "\[Meta:RUN\]" logs/meta_drain_debug.log  
grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log
grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log
```

## What Each Log Means

| Log Pattern | Meaning | Status |
|---|---|---|
| `[Meta:START]` appears | MetaController.start() was called | ✅ Phase boot worked |
| `[Meta:RUN]` appears | run() loop is executing | ✅ Task is alive |
| `[Meta:DRAIN]` appears | Draining is happening | ✅ Pipeline running |
| `[AgentManager:BATCH]` appears | Agents generating signals | ✅ Upstream OK |

## Run Right Now

```bash
# Terminal 1
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 main.py 2>&1 | tee logs/meta_drain_debug.log

# Wait 30 seconds, then Ctrl+C to stop

# Terminal 2  
grep "\[Meta:START\]" logs/meta_drain_debug.log
grep "\[Meta:RUN\]" logs/meta_drain_debug.log
grep "\[Meta:DRAIN\]" logs/meta_drain_debug.log
grep "\[AgentManager:BATCH\]" logs/meta_drain_debug.log
```

## Copy-Paste All 4 Outputs Here

Once you run the 4 greps above, share the output and I'll tell you exactly what's broken.

## What We Already Know

✅ Agents ARE generating signals (AgentManager publishes to event_bus)
✅ Event bus has the messages
❌ Meta is NOT pulling them out

**Possible causes**:
1. `start()` never called (Phase P6 issue)
2. `run()` task created but never runs (exception or cancellation)
3. `run()` runs but draining not happening (evaluate_and_act issue)

**The 4 grep commands will pinpoint EXACTLY which one.**

## Done?

Just share the grep output and we're one step closer to fixing this!
