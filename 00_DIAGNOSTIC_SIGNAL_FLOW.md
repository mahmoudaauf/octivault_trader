# 🔍 CRITICAL DIAGNOSTIC: Where Are The Signals Disappearing?

## The Evidence Chain

Your logs show:

```
✅ STEP 1: TrendHunter generates signals
[TrendHunter] Buffered BUY for BTCUSDT (conf=0.70, exp_move=0.99%, regime=normal, pos_scale=1.00)

❌ STEP 2: ??? MISSING ???

❌ STEP 3: MetaController never receives decisions
[Meta:POST_BUILD] decisions_count=0 decisions=[]

❌ STEP 4: Trades never execute
(No "Submitted X TradeIntents" messages found)
```

The signals are disappearing somewhere between **Step 1** (TrendHunter buffering) and **Step 3** (MetaController building decisions).

---

## The Missing Link: AgentManager Signal Collection

The supposed flow is:

```
TrendHunter.generate_signals()
    ↓ (returns _collected_signals)
AgentManager.collect_and_forward_signals()
    ↓ (normalizes to intents)
AgentManager.submit_trade_intents(batch)
    ↓ (forwards to MetaController)
MetaController.receive_signal()
    ↓ (stores in signal_cache)
MetaController._build_decisions()
    ↓ (reads from cache, builds decisions)
MetaController._execute_decision()
    ↓ (executes trades)
```

**But the logs show this chain is breaking at Step 2!**

---

## Diagnostic Steps (Run These Now)

### **Diagnostic 1: Check AgentManager Signal Collection**

```bash
grep -a "AgentManager:NORMALIZE" logs/clean_run.log | head -20
```

**Expected output:**
```
[AgentManager:NORMALIZE] Normalizing 2 raw signals from TrendHunter
[AgentManager:NORMALIZE] ✓ Successfully normalized 2 intents from TrendHunter
```

**If you see this:** AgentManager IS collecting signals ✅
**If NOT:** AgentManager is NOT collecting signals ❌ → **Problem is in Step 2**

---

### **Diagnostic 2: Check AgentManager Batch Submission**

```bash
grep -a "Submitted.*TradeIntents to Meta" logs/clean_run.log | head -20
```

**Expected output:**
```
Submitted 2 TradeIntents to Meta
[AgentManager:BATCH] Submitted batch of 2 intents: TrendHunter:BTCUSDT, TrendHunter:ETHUSDT
```

**If you see this:** Submission IS happening ✅
**If NOT:** Batch is empty or submission blocked ❌ → **Problem is in normalization or direct forwarding**

---

### **Diagnostic 3: Check AgentManager Direct Forwarding**

```bash
grep -a "AgentManager:DIRECT" logs/clean_run.log | head -20
```

**Expected output:**
```
[AgentManager:DIRECT] Forwarded 2 signals directly to MetaController.signal_cache
```

**If you see this:** Direct forwarding IS happening ✅
**If NOT:** Direct path failed ❌ → **Check receive_signal() method**

---

### **Diagnostic 4: Check MetaController Signal Reception**

```bash
grep -a "receive_signal" logs/clean_run.log | head -20
```

**Expected output:**
```
[MetaController:RECV_SIGNAL] Received signal for BTCUSDT from TrendHunter
[MetaController:RECV_SIGNAL] ✓ Signal cached for BTCUSDT from TrendHunter
```

**If you see this:** Signals ARE reaching MetaController ✅
**If NOT:** Signals stopped before reaching MetaController ❌

---

### **Diagnostic 5: Check Signal Cache Status**

```bash
grep -a "Signal cache" logs/clean_run.log | head -20
```

**Expected output:**
```
[Meta:SignalCache] Cleaned up 2 expired signals
```

This shows the cache is being used.

---

### **Diagnostic 6: Run Full Diagnostic Sequence**

```bash
echo "=== STEP 1: TrendHunter Buffering ===" && \
grep -a "Buffered BUY\|Buffered SELL" logs/clean_run.log | wc -l && \
echo "=== STEP 2: AgentManager Normalization ===" && \
grep -a "AgentManager:NORMALIZE" logs/clean_run.log | wc -l && \
echo "=== STEP 3: AgentManager Batch Submission ===" && \
grep -a "Submitted.*TradeIntents to Meta" logs/clean_run.log | wc -l && \
echo "=== STEP 4: Direct Forwarding ===" && \
grep -a "AgentManager:DIRECT" logs/clean_run.log | wc -l && \
echo "=== STEP 5: MetaController Reception ===" && \
grep -a "receive_signal" logs/clean_run.log | wc -l && \
echo "=== STEP 6: MetaController Decisions ===" && \
grep -a "decisions_count=" logs/clean_run.log | head -5
```

This will show the count at each step.

---

## Expected Sequence (Success Case)

```
TrendHunter: 10 buffered signals
   ↓
AgentManager: 10 normalized intents
   ↓
AgentManager: Submit 10 TradeIntents
   ↓
AgentManager: Forward 10 signals to Meta
   ↓
MetaController: Receive 10 signals
   ↓
MetaController: decisions_count=10 (or reduced due to deduplication)
   ↓
Trades execute
```

---

## Most Likely Failure Points

### **Failure Point 1: collect_and_forward_signals() Not Called**

**Symptoms:**
- `Buffered BUY` messages appear ✅
- NO `AgentManager:NORMALIZE` messages ❌
- `decisions_count=0` ❌

**Cause:** `_tick_loop()` may not be running

**Check:**
```bash
grep -a "_tick_loop\|AgentManager:tick" logs/clean_run.log | head -5
```

---

### **Failure Point 2: generate_signals() Returns Empty**

**Symptoms:**
- `Buffered BUY` messages appear ✅
- `AgentManager:NORMALIZE` shows "Empty/None raw signals" ❌
- `decisions_count=0` ❌

**Cause:** `generate_signals()` not returning `_collected_signals`

**Check:**
```bash
grep -a "Empty/None raw signals from TrendHunter" logs/clean_run.log
```

---

### **Failure Point 3: Normalization Fails**

**Symptoms:**
- `Buffered BUY` messages appear ✅
- `AgentManager:NORMALIZE` shows "Normalizing N signals" ✅
- `AgentManager:NORMALIZE` shows "FAILED to normalize any" ❌
- `decisions_count=0` ❌

**Cause:** Signal dict format doesn't match expected keys

**Check:**
```bash
grep -a "FAILED to normalize\|yielded non-dict" logs/clean_run.log
```

---

### **Failure Point 4: receive_signal() Fails**

**Symptoms:**
- `Buffered BUY` messages appear ✅
- `AgentManager:NORMALIZE` succeeds ✅
- `AgentManager:BATCH` shows "Submitted X TradeIntents" ✅
- NO `receive_signal` messages ❌
- `decisions_count=0` ❌

**Cause:** `MetaController.receive_signal()` rejecting signals

**Check:**
```bash
grep -a "receive_signal\|Signal cached" logs/clean_run.log | head -10
```

---

## Run This Diagnostic Now

Create a file `run_diagnostic.sh`:

```bash
#!/bin/bash

LOG_FILE="logs/clean_run.log"

echo "==============================================="
echo "SIGNAL FLOW DIAGNOSTIC"
echo "==============================================="
echo ""

echo "1️⃣  TrendHunter - Buffering Signals"
echo "---"
count=$(grep -c "Buffered BUY\|Buffered SELL" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Count: $count"
echo "Sample:"
grep -a "Buffered BUY" "$LOG_FILE" | head -3
echo ""

echo "2️⃣  AgentManager - Normalizing Signals"
echo "---"
count=$(grep -c "AgentManager:NORMALIZE" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Count: $count"
echo "Sample:"
grep -a "AgentManager:NORMALIZE" "$LOG_FILE" | head -3
echo ""

echo "3️⃣  AgentManager - Submitting Intents"
echo "---"
count=$(grep -c "Submitted.*TradeIntents to Meta" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Count: $count"
echo "Sample:"
grep -a "Submitted.*TradeIntents to Meta" "$LOG_FILE" | head -3
echo ""

echo "4️⃣  AgentManager - Direct Forwarding"
echo "---"
count=$(grep -c "AgentManager:DIRECT" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Count: $count"
echo "Sample:"
grep -a "AgentManager:DIRECT" "$LOG_FILE" | head -3
echo ""

echo "5️⃣  MetaController - Receiving Signals"
echo "---"
count=$(grep -c "receive_signal\|RECV_SIGNAL" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Count: $count"
echo "Sample:"
grep -a "receive_signal\|RECV_SIGNAL" "$LOG_FILE" | head -3
echo ""

echo "6️⃣  MetaController - Building Decisions"
echo "---"
echo "Sample:"
grep -a "decisions_count=" "$LOG_FILE" | head -5
echo ""

echo "==============================================="
echo "ANALYSIS"
echo "==============================================="
echo ""

BUFF=$(grep -c "Buffered BUY\|Buffered SELL" "$LOG_FILE" 2>/dev/null || echo 0)
NORM=$(grep -c "AgentManager:NORMALIZE" "$LOG_FILE" 2>/dev/null || echo 0)
SUBM=$(grep -c "Submitted.*TradeIntents to Meta" "$LOG_FILE" 2>/dev/null || echo 0)
RECV=$(grep -c "receive_signal\|RECV_SIGNAL" "$LOG_FILE" 2>/dev/null || echo 0)

echo "Buffered: $BUFF | Normalized: $NORM | Submitted: $SUBM | Received: $RECV"
echo ""

if [ "$BUFF" -gt 0 ] && [ "$NORM" -eq 0 ]; then
    echo "❌ PROBLEM: Signals buffered but NOT normalized"
    echo "   → AgentManager.collect_and_forward_signals() may not be running"
    echo "   → OR generate_signals() returning empty"
elif [ "$NORM" -gt 0 ] && [ "$SUBM" -eq 0 ]; then
    echo "❌ PROBLEM: Signals normalized but NOT submitted"
    echo "   → Batch construction failed"
elif [ "$SUBM" -gt 0 ] && [ "$RECV" -eq 0 ]; then
    echo "❌ PROBLEM: Signals submitted but NOT received by Meta"
    echo "   → receive_signal() failing or not called"
elif [ "$RECV" -gt 0 ]; then
    echo "✅ Signals reached MetaController"
    echo "   → Problem is in signal_cache or _build_decisions()"
else
    echo "❌ NO SIGNALS DETECTED IN ENTIRE PIPELINE"
fi
```

Run it:
```bash
bash run_diagnostic.sh
```

---

## What To Do Based On Results

### **If Step 1 → Step 2 breaks (Buffered ✅, Normalized ❌):**

Check if AgentManager is actually running. Look for:
```bash
grep -a "_tick_loop\|tick_all_once" logs/clean_run.log | head -10
```

If nothing, `_tick_loop()` is not executing.

### **If Step 2 → Step 3 breaks (Normalized ✅, Submitted ❌):**

The batch is empty or normalization is failing silently. Check:
```bash
grep -a "FAILED to normalize\|Empty/None raw signals" logs/clean_run.log
```

### **If Step 3 → Step 4 breaks (Submitted ✅, Received ❌):**

`MetaController.receive_signal()` is not being called or is failing. Check:
```bash
grep -a "receive_signal.*RECV_SIGNAL" logs/clean_run.log | tail -5
```

### **If Step 4 → Step 5 breaks (Received ✅, Decisions still 0 ❌):**

Signals are reaching MetaController but not being read by `_build_decisions()`. The signal_cache might not be connected properly.

---

## Run The Diagnostic Now!

Please run the diagnostic command and share the output. This will pinpoint exactly where the signals are disappearing!

