# 🎯 QUICK REFERENCE: The Consensus Gate Fix

## The Problem (In 1 Sentence)
**Tier-A signals from TrendHunter (1 agent) were blocked by a rule requiring 2+ agents.**

## The Location
**File**: `core/meta_controller.py`  
**Lines**: 12037-12055  
**Function**: `_build_decisions()`

## The Fix (In 3 Lines)
```python
# OLD: if len(agents_for_sym) < 2: skip signal
# NEW: if len(agents_for_sym) == 1 and conf < 0.65: require 2
#      if len(agents_for_sym) == 1 and conf >= 0.65: allow 1
```

## Test It (Copy/Paste)
```bash
# Terminal 1: Run system
python main_phased.py

# Terminal 2: Check fix is working
tail -f logs/app.log | grep "ConsensusCheck.*ALLOW\|Selected Tier-A"

# Should see:
# [Meta:ConsensusCheck] BTCUSDT: ... min_agents=1 decision=ALLOW
# [MetaController] Selected Tier-A: BTCUSDT BUY
```

## Expected Change in Logs

| Before | After |
|--------|-------|
| decisions_count=0 | decisions_count=1 |
| [INSUFFICIENT_AGENTS agents=1/2] | [decision=ALLOW agents=1/1] |
| NO trades | Trades execute |

## Verify With This Command
```bash
tail -n 200 logs/app.log | grep -c "POST_BUILD.*decisions_count=0"
# Result: Should be 0 (no zero-decision logs)

tail -n 200 logs/app.log | grep "POST_BUILD" | head -n 3
# Result: Should show decisions_count >= 1
```

## If It's Working
- ✅ Log contains: `[Meta:ConsensusCheck] ... decision=ALLOW`
- ✅ Log contains: `[Selected Tier-A: BTCUSDT BUY]`
- ✅ Trades are executing
- ✅ Account balance is changing

## If It's NOT Working
- ❌ Still seeing `[TierA:Readiness] ... INSUFFICIENT_AGENTS`
- ❌ Still seeing `decisions_count=0`
- ❌ No trades executing

**If NOT working**: The fix may not have applied. Check line 12042 contains:
```python
if len(agents_for_sym) == 1:
    if best_conf < 0.65:
```

---

## What Actually Changed

### Before (Broken)
```python
# Line 12042-12043
if tier == "A" and not getattr(self, "_focus_mode_active", False):
    min_agents = max(min_agents, 2)  # Always require 2
```

### After (Fixed)
```python
# Line 12041-12052
if tier == "A" and not getattr(self, "_focus_mode_active", False):
    if len(agents_for_sym) == 1:
        if best_conf < 0.65:
            min_agents = max(min_agents, 2)
    else:
        min_agents = max(min_agents, 2)
```

---

## Logic Table

| Scenario | Before | After |
|----------|--------|-------|
| 1 agent, conf=0.70 | ❌ BLOCK (1<2) | ✅ ALLOW (0.70≥0.65) |
| 1 agent, conf=0.60 | ❌ BLOCK (1<2) | ❌ BLOCK (require 2) |
| 2 agents, conf=0.60 | ✅ ALLOW | ✅ ALLOW (unchanged) |
| 2 agents, conf=0.70 | ✅ ALLOW | ✅ ALLOW (unchanged) |

---

## Why Your Logs Had This Problem

```
Your Setup:
  - 1 agent: TrendHunter
  - Signals: conf=0.70 (Tier-A)
  - Requirement: 2 agents
  - Result: 1 < 2 → BLOCKED ❌

After Fix:
  - 1 agent: TrendHunter ✓
  - Signals: conf=0.70 (Tier-A) ✓
  - Check: 0.70 >= 0.65? YES ✓
  - Result: ALLOWED ✅
```

---

## One-Line Summary

**Allow single-agent Tier-A signals if confidence ≥ 0.65 (was: always require 2 agents).**
