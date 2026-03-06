# ⚡_CAPITAL_ESCAPE_HATCH_QUICK_REFERENCE.md

## What Changed

**File**: `core/execution_manager.py`  
**Function**: `_execute_trade_impl()` (line 5398)  
**Lines Added**: 28 lines (5489-5516)  
**Changes**: Added escape hatch logic + modified 2 guard conditions  

---

## The Rule

```
IF position_value >= 85% NAV
   AND _forced_exit = True
   AND side = SELL
THEN bypass_checks = True
     → All execution guards skipped
     → Order proceeds to market
```

---

## The Code

```python
# ===== CAPITAL ESCAPE HATCH =====
bypass_checks = False
if side == "sell" and bool(policy_ctx.get("_forced_exit")):
    try:
        nav = float(await self._get_total_equity() or 0.0)
        position_value = float(policy_ctx.get("position_value", 0.0))
        
        if nav > 0 and position_value > 0:
            concentration = position_value / nav
            
            if concentration >= 0.85:
                self.logger.warning(
                    "[EscapeHatch] CAPITAL_ESCAPE_HATCH activated for %s (%.1f%% NAV concentration) - bypassing all execution checks",
                    sym,
                    concentration * 100
                )
                bypass_checks = True
                is_liq_full = True
    except Exception as e:
        self.logger.debug(f"[EscapeHatch] Error checking concentration: {e}")
```

---

## Guard Modifications

### Before
```python
if side == "sell" and is_real_mode and not is_liq_full:
```

### After
```python
if side == "sell" and is_real_mode and not is_liq_full and not bypass_checks:
```

Same for System Mode Guard (line 5527).

---

## Trigger Conditions (ALL Required)

✅ Side = SELL  
✅ `_forced_exit = True` (from authority layer)  
✅ concentration >= 85%  
✅ NAV > 0  

---

## Effects When Triggered

| Item | Before | After |
|------|--------|-------|
| bypass_checks | False | **True** ✅ |
| is_liq_full | True | **True** (enforced) |
| Real Mode Guard | May block | **Bypassed** ✅ |
| System Mode Guard | May block | **Bypassed** ✅ |
| Risk Checks | May block | **Bypassed** ✅ |
| Result | ❌ May reject | **✅ Executes** |

---

## Log Message

When activated:
```
[EscapeHatch] CAPITAL_ESCAPE_HATCH activated for BTCUSDT (87.3% NAV concentration) - bypassing all execution checks
```

---

## Safe Defaults

- ✅ If NAV = 0: No bypass (safe)
- ✅ If position_value missing: No bypass (safe)
- ✅ If `_forced_exit ≠ True`: No bypass (safe)
- ✅ If concentration < 85%: No bypass (safe)
- ✅ If error occurs: No bypass (safe)

---

## Use Cases

**Scenario 1: Normal rotation**
- Concentration = 30% → No escape hatch (normal flow)

**Scenario 2: Emergency liquidation**
- Concentration = 87% + _forced_exit → **Escape hatch activates** ✅
- Forced exit overcomes all blocks

**Scenario 3: Manual exit (no forced flag)**
- Concentration = 90% but _forced_exit ≠ True → No escape hatch
- Falls back to normal guards

---

## Integration

Data flows from authorities to ExecutionManager:
```
RotationExitAuthority
├─ Set _forced_exit = True
├─ Set position_value
└─ Call ExecutionManager

ExecutionManager
├─ Check concentration
├─ If >= 85%: bypass_checks = True
└─ Execute order
```

---

## Testing

### Test Case 1: Below Threshold
```python
concentration = 0.75 (75%)
_forced_exit = True
Result: bypass_checks = False (no escape)
```

### Test Case 2: At Threshold
```python
concentration = 0.85 (85%)
_forced_exit = True
Result: bypass_checks = True (escape triggered) ✅
```

### Test Case 3: Above Threshold
```python
concentration = 0.92 (92%)
_forced_exit = True
Result: bypass_checks = True (escape triggered) ✅
```

---

## Deployment Checklist

- [x] Code implemented (lines 5489-5516)
- [x] Guard conditions updated (2 locations)
- [x] Warning log added
- [x] Error handling included
- [x] Documentation complete
- [x] Safe defaults verified
- [x] No breaking changes

---

## Key Points

✅ **Only for high concentration** (>= 85%)  
✅ **Only for forced exits** (_forced_exit must be True)  
✅ **Only for SELL** (not BUY)  
✅ **Bypasses all guards** when triggered  
✅ **Logs every activation** for visibility  
✅ **Safe defaults** (errors don't trigger bypass)  

---

## Impact

**Before**: Concentration deadlock possible  
**After**: Always can liquidate at >= 85% concentration  

System now has **authority AND execution power**.
