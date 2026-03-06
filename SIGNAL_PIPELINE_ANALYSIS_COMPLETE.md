# Signal Pipeline Analysis - Complete Report

## Documents Created

I've created three comprehensive documents analyzing the signal pipeline:

### 1. **SIGNAL_PIPELINE_TRACE.md**
   - Complete end-to-end signal pipeline architecture
   - Detailed code flow from Strategy Agent → Signal Generation → Event Bus → Meta Signal Cache → Decision Building → Execution
   - Data structure schemas at each stage
   - Timing and lifecycle information
   - Configuration parameters
   - Error handling strategies
   - **USE THIS FOR:** Understanding the overall architecture and intended data flow

### 2. **SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md**
   - Root cause analysis showing signals are generated but never cached
   - Evidence from actual logs showing:
     - TrendHunter buffering signals ✅
     - Meta signal cache always empty ❌
     - Decisions never built ❌
   - Identifies missing bridge logs indicating broken links
   - Lists probable root causes (signal normalization failure)
   - Proposed diagnostic steps
   - Suggested fixes with code locations
   - **USE THIS FOR:** Understanding why the current system is failing

### 3. **DIAGNOSTIC_FIXES_APPLIED.md**
   - Details of diagnostic logging added to trace signals
   - Shows exact log patterns to look for at each pipeline stage
   - Provides diagnostic flowchart to identify which layer is broken
   - Lists expected full sequence when working correctly
   - Instructions for running diagnostic test
   - **USE THIS FOR:** Running the diagnostic test and interpreting results

---

## The Core Problem (Summarized)

**Signal Pipeline Breakage:**
```
TrendHunter generates & buffers signals ✅
    ↓
Returned from generate_signals() ✅
    ↓
???
    ↓
🔴 SIGNALS NEVER REACH SIGNAL CACHE
    ↓
_build_decisions() always returns empty []
```

**Evidence:**
- Logs show `[TrendHunter] Buffered BUY for BTCUSDT` ✅
- Logs show `[Meta:BOOTSTRAP_DEBUG] Signal cache contains 0 signals: []` ❌
- Logs show `[Meta:POST_BUILD] decisions_count=0` ❌
- All bridge logs are missing (Published, Submitted, Drained, RECV_SIGNAL)

---

## What I've Done

### 1. Traced the Complete Signal Pipeline
   - Documented all 7 stages from Strategy Agent to Execution
   - Identified data structures at each transformation
   - Located all critical code sections

### 2. Identified the Breakage
   - Signals ARE being generated (TrendHunter logs prove it)
   - Signals are NOT reaching Meta cache (empty cache logs prove it)
   - Something between generation and caching is broken

### 3. Added Diagnostic Instrumentation
   - Added WARNING-level logs at each critical handoff point
   - These logs will show exactly where signals stop flowing
   - Covers 4 critical sections:
     1. Signal normalization (agent_manager.py)
     2. Event bus publishing (agent_manager.py)
     3. Signal reception and caching (meta_controller.py)
     4. Event bus draining (meta_controller.py)

### 4. Created Diagnostic Flowchart
   - Shows decision tree for interpreting which logs are missing
   - Points to exact code section based on missing logs
   - Includes expected complete sequence

---

## How to Use These Documents

### For Understanding the Architecture
→ Start with **SIGNAL_PIPELINE_TRACE.md**
- Read the complete pipeline overview
- Understand each data transformation
- Learn the event flow and timing

### For Diagnosing the Problem
→ Use **SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md**
- Review the evidence of breakage
- Understand the three theories of what's wrong
- See the recommended diagnostic steps

### For Running the Diagnostic
→ Follow **DIAGNOSTIC_FIXES_APPLIED.md**
1. Run the diagnostic test:
   ```bash
   python -m pytest tests/test_clean_run.py -xvs > logs/diagnostic_run.log 2>&1
   ```

2. Check for the diagnostic logs:
   ```bash
   grep -a "\[AgentManager:NORMALIZE\]\|\[AgentManager:SUBMIT\]\|\[MetaController:RECV_SIGNAL\]\|\[Meta:DRAIN" logs/diagnostic_run.log
   ```

3. Use the diagnostic flowchart to identify which layer is broken

4. Apply the suggested fixes based on what logs are missing

---

## Critical Data Flow Paths

### Path #1: Event Bus Route
```
TrendHunter.generate_signals() → List[Dict]
    ↓
AgentManager.collect_and_forward_signals()
    ├─ normalize_to_intents() → List[TradeIntent]
    └─ submit_trade_intents() → publish to "events.trade.intent"
        ↓
    MetaController._drain_trade_intent_events()
        └─ Event appears in queue
            ↓
        IntentManager.receive_intents()
            └─ Signal cached
```

**Expected Log Bridge:**
- `[TrendHunter] Buffered BUY`
- `[AgentManager:NORMALIZE]` ← Checking for this
- `[AgentManager:SUBMIT]` ← Checking for this
- `[Meta:DRAIN] ⚠️ DRAINED` ← Checking for this
- `[Meta:BOOTSTRAP_DEBUG] Signal cache contains X` ← Should show X > 0

### Path #2: Direct MetaController Route
```
TrendHunter.generate_signals() → List[Dict]
    ↓
AgentManager.collect_and_forward_signals()
    ├─ normalize_to_intents() → List[TradeIntent]
    └─ [DIRECT PATH] await meta_controller.receive_signal() for each intent
        ↓
    MetaController.receive_signal()
        └─ signal_manager.receive_signal()
            └─ Signal cached directly in signal_cache
```

**Expected Log Bridge:**
- `[TrendHunter] Buffered BUY`
- `[AgentManager:NORMALIZE]` ← Checking for this
- `[MetaController:RECV_SIGNAL] Received signal` ← Checking for this
- `[MetaController:RECV_SIGNAL] ✓ Signal cached` ← Checking for this
- `[Meta:BOOTSTRAP_DEBUG] Signal cache contains X` ← Should show X > 0

---

## Files at Risk

The problem likely exists in one of these files:
1. **core/agent_manager.py** - `collect_and_forward_signals()` or `_normalize_to_intents()`
2. **core/meta_controller.py** - `receive_signal()` or signal_manager integration
3. **core/signal_manager.py** - Signal cache storage or retrieval

The diagnostic logging will pinpoint which file/function is broken.

---

## Summary

I've created a complete diagnostic toolkit to trace and fix the signal pipeline breakage:

1. **SIGNAL_PIPELINE_TRACE.md** - Architecture reference
2. **SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md** - Problem analysis
3. **DIAGNOSTIC_FIXES_APPLIED.md** - Diagnostic instructions and flowchart

The diagnostic logging has been added to the codebase. Next step is to run the diagnostic test and check the logs to see where exactly the signals are being lost.
