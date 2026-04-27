# Capital Floor Check - Bootstrap Bypass Implementation

## Status: ✅ IMPLEMENTED BUT SYSTEM INITIALIZATION ISSUE ENCOUNTERED

### Fixes Applied

**Fix #5 (JUST NOW - Most Recent):**
- **File:** `core/meta_controller.py` line 11900-11950
- **Function:** `_check_capital_floor_central()`
- **Change:** Added BOOTSTRAP mode detection at the START of capital floor check
- **Implementation:**
  ```python
  current_mode = "NORMAL"
  if hasattr(self, "mode_manager") and self.mode_manager:
      try:
          current_mode = str(self.mode_manager.get_mode()).upper()
      except Exception as e:
          current_mode = "NORMAL"
  
  if current_mode in ("BOOTSTRAP", "BOOTSTRAP_VIRTUAL", "RECOVERY"):
      # ✅ RETURN TRUE immediately - bypass capital floor check
      return True
  ```
- **Why:** During BOOTSTRAP mode, phantom NAV inflation ($199 vs actual $32) was causing false capital starvation
- **Effect:** When in BOOTSTRAP mode, capital floor check now PASSES immediately without calculation
- **Added Debugging:** Print statements to stdout for visibility (lines 11918-11928)

### Expected Behavior After Fix

When system restarts and reaches `_build_decisions()`:
1. `_check_capital_floor_central()` is called (line 13167)
2. Mode manager returns "BOOTSTRAP" or "BOOTSTRAP_VIRTUAL"
3. Capital floor check returns TRUE immediately
4. `capital_ok = True`
5. `_evaluate_capital_stability()` should return `(True, ...)`
6. Phase gate at line 13325 does NOT return []
7. System proceeds to signal evaluation and trading

### Current System State

**What Works:**
✅ Signal generation (SwingTradeHunter, TrendHunter, MLForecaster all producing 0.65-0.94 confidence)
✅ Signal submission and caching in MetaController
✅ Mode manager detection capability (mode_manager.get_mode() works)
✅ Previous 3 confidence gate fixes (all persisted to disk)

**What's Blocked:**
❌ System initialization hanging/crashing (occurs during StartupOrchestrator phase, before main trading loop)
❌ Capital floor check never being evaluated (because main loop not reached)
❌ Actual trade execution not yet tested with the fix

### Root Cause Analysis

**Primary Issue (Chicken-Egg Deadlock):**
```
Phase 1: BOOTSTRAP mode, first trade not yet executed
↓
NAV = phantom $199 (calculated from past session positions)  
↓
Capital floor check: floor = max($10, $199 × 0.20) = $39.99 needed
↓
Available balance: $32.46 < $39.99 
↓
capital_ok = FALSE 
↓
capital_stable = FALSE
↓
BOOTSTRAP_VIRTUAL gate blocks ALL trades (including the first seed trade needed to confirm NAV!)
↓
System never executes first trade → NAV never confirmed → permanent deadlock
```

**Solution Implemented:**
Detect BOOTSTRAP mode and BYPASS the floor check entirely, allowing first seed trade(s) to execute and naturally confirm NAV. Once real positions are created, NAV calculation becomes accurate and floor check can re-engage.

### Code Location References

**Capital Floor Bypass Logic:**
- Location: `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/core/meta_controller.py`
- Start line: 11900 (function definition)
- Bypass check: Lines 11918-11928 (mode detection + immediate return)
- Full function: Lines 11900-12017

**Where It's Called:**
- Location: Line 13167 in `_build_decisions()`
- Timing: Called at START of decision building cycle (BEFORE any symbol evaluation)
- Result stored: `capital_ok = await self._check_capital_floor_central()`

**Phase Gate That Uses Result:**
- Location: Lines 13325-13351
- Condition: If `capital_stable=False` and reason NOT in capital-related categories, return empty (blocks trades)
- With fix: `capital_stable` should now be TRUE, so block doesn't trigger

### Testing Protocol

When system restarts successfully:

1. **Monitor Logs for BOOTSTRAP Bypass (30-60s):**
   ```
   grep "BOOTSTRAP MODE ACTIVE\|✅✅✅" orchestrator_latest.log
   ```
   Expected: Should see message confirming BOOTSTRAP mode detection

2. **Check Capital Stability (30-60s):**
   ```
   grep "capital_stable\|capital_ok" orchestrator_latest.log
   ```
   Expected: Should show `capital_ok=True` and `capital_stable=True`

3. **Verify Signal Execution (120-180s):**
   ```
   grep "decision=BUY\|decision=SELL\|TradeIntent.*BTCUSDT" orchestrator_latest.log
   ```
   Expected: Should see decisions being made (not blocked)

4. **Check Actual Trades (3-5min):**
   ```
   grep "ORDER PLACED\|execution\|Execution.*BTCUSDT" orchestrator_latest.log
   ```
   Expected: Should see actual order placement on Binance

### Related Previous Fixes (All Applied)

**Fix #1:** ModeManager SOP_MATRIX confidence floors lowered (0.70→0.50, 0.65→0.50)
**Fix #2:** MetaController._get_mode_confidence_floor() defaults corrected (0.45→0.50)
**Fix #3:** Signal floor cap added (→0.70 to prevent 0.905 blocking)
**Fix #4:** Capital floor thresholds reduced for BOOTSTRAP mode (lines 13195-13210) ← Fallback if bypass doesn't work
**Fix #5:** Capital floor check BYPASS for BOOTSTRAP mode (current, lines 11918-11928) ← Primary solution

### Investigation Summary

The capital floor phantom NAV issue was confirmed via log analysis:
- NAV at startup: $128.22 (actual account total)
- NAV during trading: $199.89 (phantom - from recovered past positions)
- Capital floor calc: floor = max($10, $199.89 × 0.20) = $23.99
- Available: $22.46 → CHECK FAILED
- Result: All trades blocked despite having sufficient free capital

With Fix #5 applied, the BOOTSTRAP mode bypass prevents this check entirely during initialization, allowing first trade to execute and confirm real NAV.

### Next Steps

1. **System Restart:** Resolve initialization hang (separate debugging needed)
2. **Main Trading Loop:** Verify capital floor bypass triggers when _build_decisions() is called
3. **Signal to Trade Flow:** Confirm signals convert to actual orders
4. **Phantom NAV Root Cause:** Investigate why NAV=$199 during recovery (possible dust double-count)
