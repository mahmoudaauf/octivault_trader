# Phase C: Symbol Rotation Manager Integration - Capital Governor Enforcement

**Status**: Implementation Guide  
**Component**: `core/rotation_authority.py` + `core/meta_controller.py`  
**Dependencies**: Capital Governor (Phase A & B complete)  
**Time Estimate**: 45 minutes  
**Scope**: Prevent rotation in MICRO bracket, restrict to core symbols  

---

## Overview

**Objective**: Use Capital Governor's rotation restrictions to gate symbol rotation

**Problem**: 
- MICRO bracket should have NO rotation (focused learning)
- Current rotation system allows all accounts to rotate freely
- No bracket-aware rotation enforcement

**Solution**:
- Query Capital Governor: `should_restrict_rotation(nav)`
- Block rotation attempts in MICRO bracket
- Log rotation blocks with reason
- Let SMALL+ brackets rotate normally

**Flow**:
```
Rotation Signal Arrives
    ↓
Capital Governor: Should restrict rotation?
    ├─ MICRO ($350): ✅ YES → Block rotation
    ├─ SMALL ($1,500): ❌ NO → Allow (1 rotating slot)
    ├─ MEDIUM ($5,000): ❌ NO → Allow (5 rotating slots)
    └─ LARGE ($10K+): ❌ NO → Allow (10 rotating slots)
    ↓
Continue with normal rotation authorization
```

---

## Implementation Steps

### Step 1: Understand Current Rotation Flow

**Current File**: `core/rotation_authority.py` (599 lines)

Key methods:
- `authorize_rotation()` - Main rotation authorization
- `authorize_stagnation_exit()` - Stagnation-based rotation
- `authorize_concentration_exit()` - Portfolio concentration management
- `authorize_starvation_efficiency_exit()` - Efficiency-based rotation

These are called from MetaController around lines 6943-7017.

---

### Step 2: Add Governor Check to RotationExitAuthority

**File**: `core/rotation_authority.py`  
**Location**: Add to `__init__` method (around line 14)

```python
def __init__(self, logger: logging.Logger, config: Any, shared_state: Any, capital_governor=None):
    self.logger = logger
    self.config = config
    self.ss = shared_state
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE C: Capital Governor Integration
    # Enforce bracket-based rotation restrictions
    # ═══════════════════════════════════════════════════════════════════
    self.capital_governor = capital_governor
    if self.capital_governor is None:
        # Try to import if not provided
        try:
            from core.capital_governor import CapitalGovernor
            self.capital_governor = CapitalGovernor(config)
            self.logger.info("[REA:Init] Capital Governor initialized for rotation enforcement")
        except ImportError:
            self.logger.warning("[REA:Init] Capital Governor not available, rotation will not be restricted by bracket")
            self.capital_governor = None
    
    # Rest of __init__ continues...
```

**Why here**: RotationExitAuthority is initialized in MetaController, so we can pass it or have it initialize itself.

---

### Step 3: Add Rotation Restriction Helper Method

**File**: `core/rotation_authority.py`  
**Location**: Add new method (after `__init__`, around line 70)

```python
def should_restrict_rotation(self, symbol: str) -> Tuple[bool, str]:
    """
    PHASE C: Check if rotation should be restricted for this symbol.
    
    Uses Capital Governor to enforce bracket-based rotation rules:
    - MICRO: ✅ Restrict (no rotation allowed)
    - SMALL+: ❌ Allow (rotation permitted)
    
    Args:
        symbol: Symbol being considered for rotation
        
    Returns:
        Tuple[bool, str]: (should_restrict, reason)
        - (True, "micro_bracket") if rotation should be blocked
        - (False, "") if rotation is allowed
    """
    if not self.capital_governor:
        # No Governor available, allow rotation
        return False, ""
    
    try:
        # Get current NAV from SharedState
        nav = float(getattr(self.ss, "nav", 0.0) or 
                   getattr(self.ss, "total_value", 0.0) or 0.0)
        
        if nav <= 0:
            # Default to allowing if NAV unavailable
            return False, ""
        
        # Check if rotation should be restricted
        should_restrict = self.capital_governor.should_restrict_rotation(nav)
        
        if should_restrict:
            self.logger.warning(
                "[REA:RotationRestriction] Rotation blocked for %s: "
                "MICRO bracket (NAV=$%.2f) - focused learning phase",
                symbol, nav
            )
            return True, "micro_bracket_restriction"
        else:
            # Rotation allowed for this bracket
            bracket = self.capital_governor.get_bracket(nav).value
            self.logger.debug(
                "[REA:RotationRestriction] Rotation allowed for %s: "
                "%s bracket (NAV=$%.2f)",
                symbol, bracket, nav
            )
            return False, ""
        
    except Exception as e:
        self.logger.error("[REA:RotationRestriction] Check failed: %s", e)
        # Graceful fallback: allow rotation on error
        return False, ""
```

**Why**: Centralized check that can be called before any rotation attempt

---

### Step 4: Integrate into `authorize_rotation()` Method

**File**: `core/rotation_authority.py`  
**Location**: Find `authorize_rotation()` method (around line ~200)

**Find this section**:
```python
async def authorize_rotation(self, symbol: str, candidate_symbol: str, ...) -> Optional[Dict[str, Any]]:
    """Authorize symbol rotation if conditions are met."""
    # ... existing logic ...
```

**Add PHASE C check right at the beginning**:
```python
async def authorize_rotation(self, symbol: str, candidate_symbol: str, ...) -> Optional[Dict[str, Any]]:
    """Authorize symbol rotation if conditions are met."""
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE C: Capital Governor Rotation Restriction Check
    # Block rotation in MICRO bracket for focused learning
    # ─────────────────────────────────────────────────────────────────
    should_restrict, reason = self.should_restrict_rotation(symbol)
    if should_restrict:
        self.logger.warning(
            "[REA:authorize_rotation] PHASE_C_BLOCK: Rotation denied for %s → %s: %s",
            symbol, candidate_symbol, reason
        )
        return None  # Block rotation
    
    # ... rest of existing logic continues ...
```

---

### Step 5: Integrate into `authorize_stagnation_exit()` Method

**File**: `core/rotation_authority.py`  
**Location**: Find `authorize_stagnation_exit()` method

**Add similar check**:
```python
def authorize_stagnation_exit(self, symbol: str, ...) -> Optional[Dict[str, Any]]:
    """Authorize exit due to stagnation."""
    
    # PHASE C: Check rotation restriction
    should_restrict, reason = self.should_restrict_rotation(symbol)
    if should_restrict:
        self.logger.warning(
            "[REA:authorize_stagnation_exit] PHASE_C_BLOCK: "
            "Stagnation-based rotation denied for %s: %s",
            symbol, reason
        )
        return None
    
    # ... rest of logic ...
```

---

### Step 6: Update MetaController RotationExitAuthority Initialization

**File**: `core/meta_controller.py`  
**Location**: Line ~795 where `rotation_authority` is initialized

**Find**:
```python
self.rotation_authority = RotationExitAuthority(self.logger, self.config, self.shared_state)
```

**Replace with**:
```python
# PHASE C: Pass Capital Governor to RotationExitAuthority for bracket-based restrictions
self.rotation_authority = RotationExitAuthority(
    self.logger, 
    self.config, 
    self.shared_state,
    capital_governor=self.capital_governor  # Already initialized above
)
self.logger.info("[Meta:Init] RotationExitAuthority initialized with Capital Governor for PHASE C")
```

---

## Testing Phase C

### Test 1: Rotation Blocked in MICRO

**Setup**:
- NAV: $350 (MICRO bracket)
- Current position: BTCUSDT
- Rotation candidate: ETHUSDT

**Action**: Call `rotation_authority.should_restrict_rotation("BTCUSDT")`

**Expected**:
```
[REA:RotationRestriction] Rotation blocked for BTCUSDT: 
MICRO bracket (NAV=$350.00) - focused learning phase
Returns: (True, "micro_bracket_restriction")
```

---

### Test 2: Rotation Allowed in SMALL

**Setup**:
- NAV: $1,500 (SMALL bracket)
- Current position: BTCUSDT
- Rotation candidate: ETHUSDT

**Action**: Call `rotation_authority.should_restrict_rotation("BTCUSDT")`

**Expected**:
```
[REA:RotationRestriction] Rotation allowed for BTCUSDT: 
small bracket (NAV=$1500.00)
Returns: (False, "")
```

---

### Test 3: Rotation in authorize_rotation() is Blocked

**Setup**:
- NAV: $350 (MICRO)
- Call: `await rotation_authority.authorize_rotation("BTCUSDT", "ETHUSDT", ...)`

**Expected**:
```
[REA:authorize_rotation] PHASE_C_BLOCK: 
Rotation denied for BTCUSDT → ETHUSDT: micro_bracket_restriction
Returns: None
```

Signal bubbles back up to MetaController as rejected.

---

## Integration Points

### Where Capital Governor Blocks Rotation

```python
# MetaController.evaluate_and_act()
#   → Build decisions
#   → For each decision:
#     └─ If SELL + rotation signal:
#       └─ rotation_authority.authorize_rotation()
#         └─ [NEW] should_restrict_rotation() check
#           ├─ MICRO: Block (return None)
#           └─ SMALL+: Allow (continue)
```

### Logging Output

**When Rotation Blocked** (MICRO):
```
[REA:RotationRestriction] Rotation blocked for BTCUSDT: MICRO bracket (NAV=$350.00)
[REA:authorize_rotation] PHASE_C_BLOCK: Rotation denied for BTCUSDT → ETHUSDT
```

**When Rotation Allowed** (SMALL+):
```
[REA:RotationRestriction] Rotation allowed for BTCUSDT: small bracket (NAV=$1500.00)
```

---

## Code Changes Summary

| File | Location | Change | Lines |
|------|----------|--------|-------|
| `rotation_authority.py` | ~14 | Update `__init__` signature + init Governor | +15 |
| `rotation_authority.py` | ~70 | Add `should_restrict_rotation()` helper | +45 |
| `rotation_authority.py` | ~200 | Add check in `authorize_rotation()` | +10 |
| `rotation_authority.py` | ~250 | Add check in `authorize_stagnation_exit()` | +10 |
| `meta_controller.py` | ~795 | Pass Governor to RotationExitAuthority | +5 |

**Total Changes**: ~85 lines  
**Files Modified**: 2  
**Backward Compatible**: ✅ Yes (Governor is optional)  

---

## Verification Checklist

### Pre-Implementation
- [ ] Read Phase B completion (rotation_authority already exists)
- [ ] Understand current rotation flow in MetaController
- [ ] Understand Capital Governor's `should_restrict_rotation()` method

### Implementation
- [ ] Update RotationExitAuthority `__init__` to accept capital_governor
- [ ] Add `should_restrict_rotation()` helper method
- [ ] Add check in `authorize_rotation()`
- [ ] Add check in `authorize_stagnation_exit()`
- [ ] Update MetaController initialization of RotationExitAuthority
- [ ] Add import if needed: `from core.capital_governor import CapitalGovernor`

### Testing
- [ ] Test rotation blocked in MICRO bracket
- [ ] Test rotation allowed in SMALL bracket
- [ ] Test rotation allowed in MEDIUM bracket
- [ ] Test NAV-based bracket changes
- [ ] Verify logs show PHASE_C_BLOCK messages
- [ ] Verify graceful fallback on error

### Post-Deployment
- [ ] Monitor logs for `[REA:RotationRestriction]` messages
- [ ] Verify MICRO accounts don't rotate
- [ ] Verify SMALL+ accounts can still rotate
- [ ] Check for any exceptions in rotation logic

---

## Architecture Diagram

```
Rotation Authorization Flow:

SELL Signal (with rotation intent)
    ↓
MetaController._build_decisions()
    ↓
rotation_authority.authorize_rotation()
    ├─ [NEW] PHASE C: should_restrict_rotation()
    │   ├─ Get NAV
    │   ├─ Check Capital Governor: should_restrict?
    │   ├─ MICRO: YES → Block (return True)
    │   └─ SMALL+: NO → Allow (return False)
    ├─ If blocked: Return None (rotation denied)
    └─ If allowed: Continue with existing checks
       ├─ Alpha gap check
       ├─ Stagnation check
       ├─ Concentration check
       └─ Return rotation signal (or None)
    ↓
MetaController receives result
    ├─ None: Rotation rejected
    └─ Signal: Rotation approved
```

---

## Why This Design

### 1. **Bracket-Aware Rotation**
   - Different accounts have different rotation allowances
   - MICRO: Focused learning (1 position, no rotation)
   - SMALL: Conservative rotation (1 slot)
   - LARGE: Aggressive rotation (10 slots)

### 2. **Reuses Capital Governor**
   - Single source of truth for bracket rules
   - No duplication of logic
   - One place to change rules

### 3. **Non-Blocking on Error**
   - If Governor unavailable: rotation still works
   - If NAV lookup fails: graceful fallback
   - System continues even if PHASE C disabled

### 4. **Clear Logging**
   - `PHASE_C_BLOCK` messages identify Governor blocks
   - Easy to distinguish from other rotation blocks
   - Helps with debugging and monitoring

---

## Example: Your $350 MICRO Account

### Scenario 1: Rotation Attempt (Blocked)

```
Signal: Stagnation detected, rotate BTCUSDT → ETHUSDT
    ↓
authorize_stagnation_exit(symbol="BTCUSDT", candidate="ETHUSDT")
    ↓
should_restrict_rotation("BTCUSDT")
    ├─ Get NAV: $350
    ├─ Check Governor: restrict_rotation($350)?
    ├─ Answer: YES (MICRO bracket)
    └─ Return: (True, "micro_bracket_restriction")
    ↓
Log: "[REA] PHASE_C_BLOCK: Stagnation rotation denied"
    ↓
Return: None (rotation blocked)
    ↓
MetaController: Signal rejected, BTCUSDT remains open
```

### Scenario 2: Account Grows to SMALL

```
Account NAV: $500 (now SMALL bracket)
Signal: Rotation attempt (same as before)
    ↓
should_restrict_rotation("BTCUSDT")
    ├─ Get NAV: $500
    ├─ Check Governor: restrict_rotation($500)?
    ├─ Answer: NO (SMALL bracket, 1 rotating slot allowed)
    └─ Return: (False, "")
    ↓
Log: "[REA] Rotation allowed: small bracket"
    ↓
Continue with existing rotation checks
    ↓
May return rotation signal (if other conditions met)
```

---

## Success Criteria

✅ **Phase C is successful when**:

1. Rotation is **blocked in MICRO** bracket
2. Rotation is **allowed in SMALL+** brackets
3. Logs show `PHASE_C_BLOCK` when rotation denied
4. No performance degradation
5. Graceful fallback if Governor unavailable
6. Stagnation-based rotation respects restrictions
7. Concentration-based rotation respects restrictions

---

## Next Steps (After Phase C)

### Phase D: Position Manager Integration
- Bracket-specific position sizing
- Apply EV multiplier per bracket
- Implement profit lock gates

### Phase E: End-to-End Testing
- Test Governor + Allocator together
- Test all 4 bracket levels
- Live trading validation

---

## Files to Create

After implementation is complete:
- `PHASE_C_ROTATION_MANAGER_INTEGRATION.md` - This guide (already created!)
- `test_phase_c_rotation_restriction.py` - Test suite
- `PHASE_C_COMPLETE.md` - Completion summary

---

## Summary

**Phase C Adds**:
- ✅ Bracket-aware rotation enforcement
- ✅ MICRO accounts stay focused (no rotation)
- ✅ SMALL+ accounts can rotate (within limits)
- ✅ Capital Governor as source of truth
- ✅ Clear logging for all decisions

**Result**: Your MICRO account won't rotate away from BTCUSDT/ETHUSDT core pairs during learning phase. Once account grows to SMALL ($500+), rotation becomes possible again.

---

**Created**: March 1, 2026  
**Phase**: C (Symbol Rotation Manager)  
**Status**: Ready for implementation  
**Estimate**: 45 minutes  
