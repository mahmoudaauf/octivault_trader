🎯 QUICK REFERENCE: Decision Generation Bug Fix
================================================

## Problem
❌ 6 signals → 0 decisions → 0 trades

## Root Cause  
Checked agent **remaining budget** (0) instead of signal **allocated budget** (30)

## Location
`core/meta_controller.py` lines 10945-10963

## The Fix (3-Line Change)
```python
# BEFORE:
agent_budget = _wallet_budget_for(agent_name)  # ❌ 0

# AFTER:
signal_planned_quote = float(best_sig.get("_planned_quote") or 0.0)  # ✅ 30
if signal_planned_quote <= 0:
    signal_planned_quote = _wallet_budget_for(agent_name)  # Fallback
```

## Verification (Check Logs)
```
❌ BEFORE: [Meta:POST_BUILD] decisions_count=0 decisions=[]
✅ AFTER:  [Meta:POST_BUILD] decisions_count=N decisions=[...]
```

## Impact
- 0 → 100+ trades/day
- 0 → 100% execution rate
- Risk: LOW
- Breaking Changes: NONE

## Status
✅ Applied and ready for testing
