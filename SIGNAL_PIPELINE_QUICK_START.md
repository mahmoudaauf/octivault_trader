# 🚀 Quick Start - Signal Pipeline Diagnostic Guide

## TL;DR

Signals from TrendHunter are being **generated** but **NOT cached** in Meta. I've added diagnostic logging to find where they're being lost.

---

## Step 1: Run Diagnostic Test

```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python -m pytest tests/test_clean_run.py -xvs > logs/diagnostic_run.log 2>&1
```

---

## Step 2: Check Diagnostic Logs

Copy and run this command to see all diagnostic output:

```bash
grep -E "\[AgentManager:NORMALIZE\]|\[AgentManager:SUBMIT\]|\[MetaController:RECV_SIGNAL\]|\[Meta:DRAIN" logs/diagnostic_run.log | head -50
```

---

## Step 3: Identify Broken Link Using This Checklist

### ✅ Signals Being Generated
**Look for:** `[TrendHunter] Buffered BUY for BTCUSDT`
- If you see this → Generation is working ✅
- If NOT → TrendHunter not being called (separate issue)

### ✅ Signals Being Normalized
**Look for:** `[AgentManager:NORMALIZE] Normalizing 2 raw signals`
- If you see this → Normalization happening ✅
- If NOT → `collect_and_forward_signals()` not called

**Look for:** `[AgentManager:NORMALIZE] ✓ Successfully normalized 2 intents`
- If you see this → Normalization passed ✅
- If NOT → Normalization is filtering all signals (check constraints)

### ✅ Signals Being Published
**Look for:** `[AgentManager:SUBMIT] Publishing 2 intents to event_bus`
- If you see this → Publishing to bus ✅
- If NOT → Batch is empty after normalization

**Look for:** `[AgentManager] Published 2 trade intent events`
- If you see this → Event bus publishing succeeded ✅
- If NOT → Event bus failing

### ✅ Direct Path to Meta
**Look for:** `[MetaController:RECV_SIGNAL] Received signal for BTCUSDT from TrendHunter`
- If you see this → Direct path being used ✅
- If NOT → Direct path not triggered

**Look for:** `[MetaController:RECV_SIGNAL] ✓ Signal cached for BTCUSDT from TrendHunter (confidence=0.70)`
- If you see this → Signal cached successfully ✅
- If NOT → SignalManager rejected it (see next item)

### ✅ Event Bus Draining
**Look for:** `[Meta:DRAIN:ENTRY] Entering _drain_trade_intent_events(max_items=1000)`
- If you see this → Drain function is being called ✅
- If NOT → `evaluate_and_act()` not calling drain

**Look for:** `[Meta:DRAIN] ⚠️ DRAINED 2 events from event_bus`
- If you see this AND events > 0 → Queue has events ✅
- If you see "DRAINED 0" → No events in queue (publishing failed)

### ✅ Final Check
**Look for:** `[Meta:BOOTSTRAP_DEBUG] Signal cache contains 2 signals`
- If count > 0 → Everything working ✅
- If count = 0 → Something above is broken (use checklist)

---

## Step 4: Locate the Problem

Based on what logs you see:

| See This | Missing This | Problem |
|----------|--------------|---------|
| Buffered BUY | NORMALIZE | `collect_and_forward_signals()` not called |
| NORMALIZE | ✓ Successfully normalized | Signals failing validation |
| ✓ Successfully | SUBMIT | Batch not being passed |
| SUBMIT | Published | Event bus failing |
| Published | RECV_SIGNAL | Event not in queue |
| RECV_SIGNAL | ✓ Signal cached | SignalManager rejecting |
| DRAIN:ENTRY | DRAINED X events | Queue empty |
| DRAINED 2 events | Signal cache = 0 | Drain not updating cache |

---

## Step 5: Fix Based on Problem

### If "NORMALIZE" log is missing:
Check `core/agent_manager.py` line 410:
- Verify `batch` is non-empty
- Verify `intents` is being populated
- Add debug print before line 410

### If "Successfully normalized" is missing but "Normalizing" is there:
Check `core/agent_manager.py` lines 340-360:
- Symbol validation (must contain "USDT")
- Action validation (must be "BUY" or "SELL")
- Confidence validation (must be > 0.50)

### If "SUBMIT" is missing:
Check line 450 in `core/agent_manager.py`:
- Verify the `if batch:` condition is true
- Verify `submit_trade_intents()` is being called

### If "Published" is missing but "SUBMIT" is there:
Check `core/agent_manager.py` lines 260-282:
- Verify event_bus exists
- Verify publish method exists
- Check for exceptions in event publishing

### If "RECV_SIGNAL" is missing:
Check `core/agent_manager.py` lines 477-481:
- Verify `self.meta_controller` is not None
- Verify the direct path code is being executed

### If "✓ Signal cached" is missing:
Check `core/meta_controller.py` line 5053:
- Check SignalManager rejection reason
- Common issues:
  - Symbol base is quote token (like "USDT")
  - Confidence below minimum
  - Quote not recognized

---

## Command Reference

### View all diagnostic logs
```bash
grep -E "\[AgentManager:NORMALIZE\]|\[AgentManager:SUBMIT\]|\[MetaController:RECV_SIGNAL\]|\[Meta:DRAIN" logs/diagnostic_run.log
```

### View normalization diagnostics only
```bash
grep "\[AgentManager:NORMALIZE\]" logs/diagnostic_run.log
```

### View submission diagnostics only
```bash
grep "\[AgentManager:SUBMIT\]" logs/diagnostic_run.log
```

### View reception diagnostics only
```bash
grep "\[MetaController:RECV_SIGNAL\]" logs/diagnostic_run.log
```

### View draining diagnostics only
```bash
grep "\[Meta:DRAIN" logs/diagnostic_run.log
```

### View cached signal count
```bash
grep "\[Meta:BOOTSTRAP_DEBUG\] Signal cache contains" logs/diagnostic_run.log | head -5
```

### View all TrendHunter signals
```bash
grep "\[TrendHunter\] Buffered" logs/diagnostic_run.log | head -10
```

### Compare: signals vs cache vs decisions
```bash
echo "=== TrendHunter Signals ===" && \
grep "\[TrendHunter\] Buffered" logs/diagnostic_run.log | wc -l && \
echo "=== Meta Cache Count ===" && \
grep "\[Meta:BOOTSTRAP_DEBUG\] Signal cache contains" logs/diagnostic_run.log | tail -1 && \
echo "=== Meta Decisions ===" && \
grep "\[Meta:POST_BUILD\] decisions_count=" logs/diagnostic_run.log | tail -1
```

---

## Expected Output (When Fixed)

```
=== TrendHunter Signals ===
45

=== Meta Cache Count ===
[Meta:BOOTSTRAP_DEBUG] Signal cache contains 2 signals: [BTCUSDT, ETHUSDT]

=== Meta Decisions ===
[Meta:POST_BUILD] decisions_count=2 decisions=[...]
```

---

## Documents for Reference

1. **SIGNAL_PIPELINE_TRACE.md** - Full architecture (read first time)
2. **SIGNAL_PIPELINE_BREAKAGE_ROOT_CAUSE.md** - Problem analysis
3. **DIAGNOSTIC_FIXES_APPLIED.md** - Detailed diagnostic info
4. **This file** - Quick reference (what you're reading)

---

## Support Info

- **TrendHunter Agent:** `agents/trend_hunter.py` line 762 - "Buffered BUY/SELL" logs
- **Signal Normalization:** `core/agent_manager.py` line 313 - `_normalize_to_intents()`
- **Event Publishing:** `core/agent_manager.py` line 255 - `submit_trade_intents()`
- **Direct Meta Path:** `core/agent_manager.py` line 477 - direct `receive_signal()` call
- **Meta Reception:** `core/meta_controller.py` line 5044 - `receive_signal()`
- **Signal Caching:** `core/signal_manager.py` line 60 - `receive_signal()`
- **Event Draining:** `core/meta_controller.py` line 4992 - `_drain_trade_intent_events()`

---

**Next Action:** Run the diagnostic test and share the output of the grep commands above.
