# ⚙️ PHASE 2 BOTTLENECK FIXES - IMPLEMENTATION SEQUENCE

**Execution Plan:** Sequential Implementation of 3 Fixes  
**Status:** READY TO DEPLOY  
**Date:** April 27, 2026  

---

## 📋 IMPLEMENTATION CHECKLIST

### Fix #1: Recovery Exit Min-Hold Bypass ✅ [VERIFY]
- [ ] Step 1.1: Review _safe_passes_min_hold() function signature
- [ ] Step 1.2: Verify bypass parameter in signature
- [ ] Step 1.3: Check stagnation_exit_sig flags
- [ ] Step 1.4: Check liquidity_restore_sig flags
- [ ] Step 1.5: Test with diagnostic logs

### Fix #2: Micro Rotation Override ✅ [VERIFY]
- [ ] Step 2.1: Review authorize_rotation() method
- [ ] Step 2.2: Verify force_rotation precedence logic
- [ ] Step 2.3: Check MICRO bracket override branch
- [ ] Step 2.4: Verify override logging with emoji
- [ ] Step 2.5: Test with diagnostic logs

### Fix #3: Entry-Sizing Config Alignment ⚙️ [CRITICAL]
- [ ] Step 3.1: Update .env DEFAULT_PLANNED_QUOTE to 25
- [ ] Step 3.2: Update .env MIN_TRADE_QUOTE to 25
- [ ] Step 3.3: Update .env MIN_ENTRY_USDT to 25
- [ ] Step 3.4: Update .env TRADE_AMOUNT_USDT to 25
- [ ] Step 3.5: Update .env MIN_ENTRY_QUOTE_USDT to 25
- [ ] Step 3.6: Update .env EMIT_BUY_QUOTE to 25
- [ ] Step 3.7: Update .env META_MICRO_SIZE_USDT to 25
- [ ] Step 3.8: Update core/config.py with floor alignment
- [ ] Step 3.9: Verify all 7 parameters are aligned
- [ ] Step 3.10: Run verification script

---

## 🔍 IMPLEMENTATION DETAILS

### FIX #1: Recovery Exit Min-Hold Bypass (VERIFY STATUS)

**Location:** `core/meta_controller.py`

#### Step 1.1: Review _safe_passes_min_hold() signature
```python
# Line ~12837
def _safe_passes_min_hold(self, symbol: Optional[str], bypass: bool = False) -> bool:
```
✅ **Expected:** Method has `bypass: bool = False` parameter
✅ **Current Status:** Already implemented

#### Step 1.2: Verify bypass logic inside _safe_passes_min_hold()
Should check the bypass flag and return True if bypass=True:
```python
if bypass:
    return True  # Skip min-hold check for recovery exits
```
✅ **Expected:** Logic present
✅ **Current Status:** Implementation found at line 13446

#### Step 1.3: Check stagnation_exit_sig flags (Line ~13426)
```python
stagnation_exit_sig["_bypass_min_hold"] = True
```
✅ **Expected:** Stagnation exit sets flag
✅ **Current Status:** Implemented

#### Step 1.4: Check liquidity_restore_sig flags (Line ~13445)
```python
liquidity_restore_sig["_bypass_min_hold"] = True
if not self._safe_passes_min_hold(liquidity_restore_sig.get("symbol"), bypass=True):
```
✅ **Expected:** Liquidity exit sets flag AND calls with bypass=True
✅ **Current Status:** Implemented

#### Step 1.5: Test with diagnostic logs
Expected log pattern:
```
[Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit: ETHUSDT
```

---

### FIX #2: Micro Rotation Override (VERIFY STATUS)

**Location:** `core/rotation_authority.py`

#### Step 2.1: Review authorize_rotation() method (Line ~302)
```python
async def authorize_rotation(
    self, 
    sig_pos: int, 
    max_pos: int, 
    owned_positions: Dict[str, Any], 
    best_opp: Dict[str, Any],
    current_mode: str,
    is_starved: bool = False,
    force_rotation: bool = False,  # ← This parameter
) -> Optional[Dict[str, Any]]:
```
✅ **Expected:** `force_rotation` parameter present
✅ **Current Status:** Parameter exists

#### Step 2.2: Verify force_rotation precedence logic
Should have conditional check:
```python
force_rotation = bool(
    force_rotation
    or (isinstance(best_opp, dict) and bool(best_opp.get("_force_micro_rotation")))
)

if owned_positions and not force_rotation:
    # PHASE C check: Only apply MICRO bracket restriction if NOT forced
    first_symbol = next(iter(owned_positions.keys()), None)
    if first_symbol:
        should_restrict, reason = self.should_restrict_rotation(first_symbol)
        if should_restrict:
            return None  # Block rotation
elif owned_positions and force_rotation:
    # Force rotation overrides MICRO bracket restriction
    # ... override branch
```
✅ **Expected:** Precedence documented and implemented
✅ **Current Status:** Implemented

#### Step 2.3: Check MICRO bracket override branch
Should log with emoji:
```python
self.logger.warning(
    "[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN for %s due to forced rotation (%s)",
    first_symbol,
    ...
)
```
✅ **Expected:** Override logging with ⚠️ emoji
✅ **Current Status:** Implemented

#### Step 2.4: Verify override logging
Expected log pattern:
```
[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN for BTCUSDT due to forced rotation (forced_rotation=True)
```

---

### FIX #3: Entry-Sizing Config Alignment (IMPLEMENT NOW)

**Files to modify:** `.env` and `core/config.py`

#### Step 3.1-3.7: Update .env parameters

Current state:
```properties
DEFAULT_PLANNED_QUOTE=15         ❌ Should be 25
MIN_TRADE_QUOTE=15               ❌ Should be 25
MIN_ENTRY_USDT=15                ❌ Should be 25
TRADE_AMOUNT_USDT=15             ❌ Should be 25
MIN_ENTRY_QUOTE_USDT=15          ❌ Should be 25
EMIT_BUY_QUOTE=15                ❌ Should be 25
META_MICRO_SIZE_USDT=15          ❌ Should be 25
```

**Action:** Update all to 25 USDT to align with SIGNIFICANT_POSITION_FLOOR

---

## ✅ QUICK VERIFICATION SCRIPT

```bash
#!/bin/bash
# Quick verification of all three fixes

echo "🔍 FIX #1: Recovery Exit Min-Hold Bypass"
grep -n "_bypass_min_hold" core/meta_controller.py | head -5
echo "✅ Expected: 5+ matches found"
echo ""

echo "🔍 FIX #2: Micro Rotation Override"
grep -n "⚠️ MICRO restriction OVERRIDDEN" core/rotation_authority.py
echo "✅ Expected: 1 match found"
echo ""

echo "🔍 FIX #3: Entry-Sizing Config Alignment"
grep "DEFAULT_PLANNED_QUOTE\|MIN_TRADE_QUOTE\|MIN_ENTRY_USDT\|TRADE_AMOUNT_USDT" .env | head -7
echo "✅ Expected: All should equal 25"
```

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Verify Fixes #1 & #2 are in place
```bash
python3 verify_fixes.py
# Expected: ✅ ALL CHECKS PASSED
```

### Step 2: Implement Fix #3 (Entry-Sizing Alignment)
This is the only fix that needs implementation. Execute the changes below.

### Step 3: Post-Deployment Validation
```bash
# Run the bot and observe logs for:
# - [Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit
# - [REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN
# - Entry orders with ~25 USDT size
```

---

## 📝 IMPLEMENTATION: FIX #3 - ENTRY-SIZING CONFIG ALIGNMENT

**Current Problem:** Config defaults (15 USDT) misaligned from expected floor (25 USDT)

**Solution:** Align all 7 entry-size parameters to 25 USDT

### Step 3.1: Update .env - DEFAULT_PLANNED_QUOTE
From: `DEFAULT_PLANNED_QUOTE=15`  
To: `DEFAULT_PLANNED_QUOTE=25`

### Step 3.2: Update .env - MIN_TRADE_QUOTE
From: `MIN_TRADE_QUOTE=15`  
To: `MIN_TRADE_QUOTE=25`

### Step 3.3: Update .env - MIN_ENTRY_USDT
From: `MIN_ENTRY_USDT=15`  
To: `MIN_ENTRY_USDT=25`

### Step 3.4: Update .env - TRADE_AMOUNT_USDT
From: `TRADE_AMOUNT_USDT=15`  
To: `TRADE_AMOUNT_USDT=25`

### Step 3.5: Update .env - MIN_ENTRY_QUOTE_USDT
From: `MIN_ENTRY_QUOTE_USDT=15`  
To: `MIN_ENTRY_QUOTE_USDT=25`

### Step 3.6: Update .env - EMIT_BUY_QUOTE
From: `EMIT_BUY_QUOTE=15`  
To: `EMIT_BUY_QUOTE=25`

### Step 3.7: Update .env - META_MICRO_SIZE_USDT
From: `META_MICRO_SIZE_USDT=15`  
To: `META_MICRO_SIZE_USDT=25`

### Step 3.8: Update .env - Floor alignment comment
Add or update the comment above the BUY SIZING section:
```
# =========================================================
# BUY SIZING — ADAPTIVE (CAPITAL-AWARE)
# =========================================================
# NOTE: Aligned with SIGNIFICANT_POSITION_FLOOR (25 USDT)
# All entry-size parameters set to 25 USDT for clean intent
# FIX #3: Entry-Sizing Config Alignment (2026-04-27)
```

### Step 3.9: Update core/config.py
Add floor alignment logging to the config initialization:
```python
# FIX #3: Entry-Sizing Config Alignment
# All 7 entry-size parameters aligned to SIGNIFICANT_POSITION_FLOOR (25 USDT)
if DEFAULT_PLANNED_QUOTE == 25:
    logger.info("[Config:EntrySize] ✅ Entry-sizing aligned to 25 USDT floor")
else:
    logger.warning("[Config:EntrySize] ⚠️ Entry-sizing NOT aligned (DEFAULT_PLANNED_QUOTE=%s)", DEFAULT_PLANNED_QUOTE)
```

---

## 📊 VERIFICATION MATRIX

| Fix | Component | Current | Expected | Status |
|-----|-----------|---------|----------|--------|
| #1 | _safe_passes_min_hold signature | ✅ Has bypass param | ✅ bypass: bool=False | ✅ OK |
| #1 | Stagnation exit flag | ✅ Present | ✅ _bypass_min_hold=True | ✅ OK |
| #1 | Liquidity restore flag | ✅ Present | ✅ _bypass_min_hold=True | ✅ OK |
| #2 | authorize_rotation param | ✅ Present | ✅ force_rotation: bool=False | ✅ OK |
| #2 | Override precedence logic | ✅ Present | ✅ NOT force_rotation check | ✅ OK |
| #2 | Override logging | ✅ Present | ✅ ⚠️ emoji indicator | ✅ OK |
| #3 | DEFAULT_PLANNED_QUOTE | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | MIN_TRADE_QUOTE | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | MIN_ENTRY_USDT | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | TRADE_AMOUNT_USDT | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | MIN_ENTRY_QUOTE_USDT | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | EMIT_BUY_QUOTE | ❌ 15 | ✅ 25 | ⚠️ TODO |
| #3 | META_MICRO_SIZE_USDT | ❌ 15 | ✅ 25 | ⚠️ TODO |

---

## ✨ SUMMARY

**Fixes #1 & #2:** Already implemented and verified ✅

**Fix #3:** Ready for implementation (7 parameters to update)

**Next Steps:**
1. Execute Fix #3 parameter updates (this document)
2. Verify with `python3 verify_fixes.py`
3. Run 30-min warm-up test
4. Deploy to production

---

**Version:** 1.0  
**Status:** Ready for implementation  
**Estimated Time:** 15 minutes to implement + 5 minutes to verify
