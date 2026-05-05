# ⚡ QUICK ACTION GUIDE - Enable Trading

**Current Issue:** Kill-switch blocking trades (portfolio protection mode)
**Solution:** Enable dust consolidation
**Time to Resume:** ~15-20 minutes

---

## What's Happening

```
✅ Fixes applied: PRETRADE gate now allows 0.04% moves
❌ Still blocked: Kill-switch (portfolio has losses + fragmentation)
📍 Root cause: 30 dust positions (each ~$0.30 value)
🎯 Solution: Consolidate positions → kill-switch auto-disables
```

---

## Steps to Enable Trading

### Option A: Force Dust Liquidation (Fastest - 2 minutes)
```bash
# Find and run the dust liquidation script
find . -name "*dust*liquidate*" -o -name "*consolidate*" | head -5

# Or manually in Python:
python3 -c "
import sys; sys.path.insert(0, '.')
from src.l4_execution.execution_manager import ExecutionManager
# Liquidate all dust positions
"
```

### Option B: Switch Regime (Allow Dust Healing - 1 minute)
```python
# Change MICRO_SNIPER regime to allow healing
# Look for: PRETRADE_REGIME or trading_regime config
# Change from: MICRO_SNIPER → NORMAL or HEALING_MODE
```

### Option C: Override Kill-Switch (Not Recommended)
```python
# In meta_controller.py, temporarily disable:
# if _cge.is_kill_switch_active(_ks_nav):
#     return...  # Comment out to bypass
```

---

## Monitoring After Fix

### Watch Logs For:
```
✅ "[Meta:PosCounts] Dust=30 → Dust=1-3" (consolidation happening)
✅ "[CompoundGrowthKS] Kill-switch inactive" (protection disabled)
✅ "PEPEUSDT BUY confidence=0.67" (signal retry)
✅ "TRADE_EXECUTED" (trading resumed)
```

### Command to Monitor:
```bash
tail -f logs/octivault_master_orchestrator.log | \
  grep -E "Dust=|Kill-switch|TRADE_|confidence"
```

---

## Expected Timeline

| Time | Event | Status |
|------|-------|--------|
| NOW | Fixes in code | ✅ |
| +2 min | Enable consolidation | ⏳ |
| +5 min | Dust positions liquidating | ⏳ |
| +10 min | Kill-switch auto-disables | ⏳ |
| +15 min | First PEPEUSDT BUY executes | ⏳ |
| +20 min | Trading normalized (3-5/cycle) | ⏳ |

---

## Verify Fixes Are In Place

```bash
# Check Fix #1 (threshold)
grep "0.0001" src/l8_lifecycle/meta_controller.py | head -1

# Check Fix #4 (costs)
grep "CR_FEE_BPS.*2.0" src/l8_lifecycle/meta_controller.py | head -1

# Should see both: 0.0001 and 2.0
```

---

## Questions?

See: `SYSTEM_PERFORMANCE_REPORT_MAY4.md` for detailed analysis
See: `FIX_EXECUTION_SUMMARY.md` for fix details
Check: `git log --oneline | head -5` for commits
