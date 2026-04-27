# 🔧 Dust-Liquidation Flag Alignment & Entry Floor Guard Fix

**Date**: April 24, 2026
**Priority**: HIGH - Critical for reducing dust creation
**Scope**: Config, SharedState, ExecutionManager, CapitalAllocator

---

## 📋 Problem Statement

### Issue 1: Inconsistent Dust-Liquidation Flag Wiring
- **In `config.py`**: `DUST_LIQUIDATION_ENABLED` (snake_case)
- **In `shared_state.py`**: `dust_liquidation_enabled` (lowercase)
- **Used inconsistently**: Some code checks uppercase, some checks lowercase
- **Impact**: Guards may bypass unintentionally due to case mismatch

### Issue 2: Missing Entry Floor Guard
- Trades can be opened below `SIGNIFICANT_POSITION_FLOOR` (20 USDT)
- Creates new dust positions immediately upon entry
- No explicit check to prevent opening below floor unless explicitly allowed
- System should warn or block entries below significant floor

### Issue 3: Misaligned Min Entry Values
- `MIN_ENTRY_USDT = 24.0`
- `MIN_POSITION_USDT = 24.0`
- `SIGNIFICANT_POSITION_FLOOR = 20.0`
- `MIN_ENTRY_QUOTE_USDT = 10.0` (can be overridden, creating misalignment)
- No consistent ordering or relationship defined

---

## 🎯 Solution Overview

### Fix 1: Standardize Dust-Liquidation Flag Wiring
```
Config Layer:
├── DUST_LIQUIDATION_ENABLED (env var: DUST_LIQUIDATION_ENABLED)
│   └── Read from: os.getenv("DUST_LIQUIDATION_ENABLED", "false")
│   └── Storage: config.dust_liquidation_enabled (LOWERCASE in runtime)

SharedState Layer:
├── dust_liquidation_enabled (direct copy from config)
├── Use: getattr(shared_state, "dust_liquidation_enabled", True)

ExecutionManager/CapitalAllocator:
├── Always use: self.shared_state.dust_liquidation_enabled
├── Fallback: getattr(config, "dust_liquidation_enabled", True)
```

### Fix 2: Add Entry Floor Guard
```
Before opening NEW trade:
├─ Check: entry_size < SIGNIFICANT_POSITION_FLOOR
├─ If True:
│  ├─ Log WARNING (including reason)
│  ├─ Check: allow_below_floor_flag = False (default)
│  ├─ If flag not set → REJECT (guard blocks)
│  └─ If flag set → ALLOW (explicitly requested)
├─ This prevents dust creation on entry
```

### Fix 3: Align Entry Floor Hierarchy
```
Hierarchy (lowest to highest):
1. MIN_ENTRY_QUOTE_USDT = 10.0 (absolute minimum, for discovery)
2. SAFE_ENTRY_USDT = 12.0 (safe minimum)
3. SIGNIFICANT_POSITION_FLOOR = 20.0 (floor for "significant" positions)
4. MIN_ENTRY_USDT = 24.0 (normal trading floor)
5. DEFAULT_PLANNED_QUOTE = 24.0 (normal plan quote)

Relationship:
├─ New positions < 20 USDT → will become dust
├─ New positions >= 20 USDT → considered significant
├─ New positions >= 24 USDT → normal trading
├─ Entry below 20 → requires explicit allow flag
```

---

## 🔨 Implementation Changes

### Change 1: Config.py (Standardization)

**File**: `core/config.py`
**Lines**: ~1793-1810

```python
# BEFORE (mixed case):
self.DUST_LIQUIDATION_ENABLED = os.getenv("DUST_LIQUIDATION_ENABLED", "false").lower() == "true"
self.DUST_REENTRY_OVERRIDE = os.getenv("DUST_REENTRY_OVERRIDE", "true").lower() == "true"

# AFTER (consistent lowercase in runtime):
self.dust_liquidation_enabled = os.getenv("DUST_LIQUIDATION_ENABLED", "false").lower() == "true"
self.dust_reentry_override = os.getenv("DUST_REENTRY_OVERRIDE", "true").lower() == "true"
```

**Action**: Rename to lowercase for consistency with runtime convention.

---

### Change 2: SharedState.py (Direct Pass-Through)

**File**: `core/shared_state.py`
**Lines**: ~211-212

```python
# ENSURE (already correct):
dust_liquidation_enabled: bool = True  # allow listing dust as sellable inventory
dust_reentry_override: bool = True     # allow dust positions to bypass re-entry lock

# ADD:
allow_entry_below_significant_floor: bool = False  # guard to prevent new dust creation
```

**Action**: Add new flag for entry floor guard (default: False = block by default).

---

### Change 3: ExecutionManager.py (Entry Floor Guard)

**File**: `core/execution_manager.py`
**Location**: `_validate_buy_order()` or similar pre-execution check

```python
async def _check_entry_floor_guard(self, symbol: str, quote_amount: float) -> Tuple[bool, str]:
    """
    Guard: Prevent opening new trades below significant floor unless explicitly allowed.
    
    Args:
        symbol: Trading pair
        quote_amount: Entry amount in quote asset (USDT)
    
    Returns:
        (is_allowed, reason_message)
    """
    significant_floor = float(getattr(self.config, "SIGNIFICANT_POSITION_FLOOR", 20.0))
    allow_below_floor = bool(getattr(self.shared_state, "allow_entry_below_significant_floor", False))
    
    # Check if entry would be below significant floor
    if quote_amount < significant_floor:
        if not allow_below_floor:
            reason = (
                f"[EM:ENTRY_FLOOR_GUARD] {symbol} entry ${quote_amount:.2f} "
                f"below significant floor ${significant_floor:.2f}. "
                f"Set allow_entry_below_significant_floor=True to override."
            )
            self.logger.warning(reason)
            return False, reason
        else:
            reason = (
                f"[EM:ENTRY_FLOOR_GUARD_OVERRIDE] {symbol} entry ${quote_amount:.2f} "
                f"below significant floor ${significant_floor:.2f}, but override enabled."
            )
            self.logger.info(reason)
            return True, reason
    
    return True, "Entry floor check passed"
```

**Action**: Add this guard before any BUY order execution.

---

### Change 4: ExecutionManager.py (Dust-Liquidation Flag Consistency)

**File**: `core/execution_manager.py`
**Locations**: All references to dust_liquidation_enabled

```python
# BEFORE (inconsistent):
if getattr(self.config, "dust_liquidation_enabled", True):
if getattr(self.shared_state, "DUST_LIQUIDATION_ENABLED", True):

# AFTER (consistent lowercase):
dust_liq_enabled = getattr(self.shared_state, "dust_liquidation_enabled", True)
if dust_liq_enabled:
    # proceed with dust operations
```

**Action**: Standardize all checks to use lowercase `dust_liquidation_enabled` from shared_state.

---

### Change 5: CapitalAllocator.py (Dust-Liquidation Flag Consistency)

**File**: `core/capital_allocator.py`
**Similar changes as ExecutionManager**

```python
# Standardize all dust_liquidation_enabled checks
dust_liq_enabled = getattr(self.shared_state, "dust_liquidation_enabled", True)
```

---

### Change 6: SharedState.py (Init Method)

**File**: `core/shared_state.py`
**Location**: `__init__()` or property setter

```python
# ADD initialization from config:
@property
def dust_liquidation_enabled(self) -> bool:
    return getattr(self, "_dust_liquidation_enabled", True)

@dust_liquidation_enabled.setter
def dust_liquidation_enabled(self, value: bool) -> None:
    self._dust_liquidation_enabled = bool(value)

# OR during init:
self.dust_liquidation_enabled = getattr(config, "dust_liquidation_enabled", True)
```

---

## 📊 Testing Plan

### Test 1: Flag Wiring
```python
def test_dust_liquidation_flag_consistency():
    config = Config()
    shared_state = SharedState(config=config)
    
    # Check consistency
    assert hasattr(config, "dust_liquidation_enabled")
    assert hasattr(shared_state, "dust_liquidation_enabled")
    assert config.dust_liquidation_enabled == shared_state.dust_liquidation_enabled
    
    # Check env override works
    os.environ["DUST_LIQUIDATION_ENABLED"] = "true"
    config2 = Config()
    assert config2.dust_liquidation_enabled == True
```

### Test 2: Entry Floor Guard
```python
async def test_entry_floor_guard():
    # Test 1: Entry below floor without override → BLOCKED
    em = ExecutionManager(config=config)
    allowed, reason = await em._check_entry_floor_guard("BTCUSDT", 15.0)
    assert allowed == False
    assert "below significant floor" in reason
    
    # Test 2: Entry below floor with override → ALLOWED
    shared_state.allow_entry_below_significant_floor = True
    allowed, reason = await em._check_entry_floor_guard("BTCUSDT", 15.0)
    assert allowed == True
    assert "override enabled" in reason
    
    # Test 3: Entry above floor → ALLOWED
    allowed, reason = await em._check_entry_floor_guard("BTCUSDT", 25.0)
    assert allowed == True
    assert "passed" in reason
```

### Test 3: Integration (E2E)
```python
# Verify no new trades open below significant floor in normal operation
# Run 1-hour session, check:
# - No entries < SIGNIFICANT_POSITION_FLOOR (20 USDT) unless explicitly allowed
# - New positions >= 20 USDT
# - Dust positions only created from partial exits, not new entries
```

---

## 🚀 Rollout Plan

### Phase 1: Code Changes (Now)
- [ ] Standardize `DUST_LIQUIDATION_ENABLED` → `dust_liquidation_enabled` (all files)
- [ ] Add `allow_entry_below_significant_floor` flag to SharedState
- [ ] Implement entry floor guard in ExecutionManager
- [ ] Update all references to use consistent lowercase naming

### Phase 2: Testing (1 hour)
- [ ] Unit tests for flag consistency
- [ ] Unit tests for entry floor guard
- [ ] Integration test (1-hour trading session)

### Phase 3: Monitoring (Runtime)
- [ ] Log all entry floor guard triggers
- [ ] Track entries blocked vs. allowed
- [ ] Monitor new dust position creation rate

### Phase 4: Cleanup (Post-Session)
- [ ] Verify no regressions
- [ ] Document changes
- [ ] Update configuration docs

---

## 📝 Configuration Examples

### Environment Variables
```bash
# Enable dust liquidation (default: false)
export DUST_LIQUIDATION_ENABLED=true

# Enable dust re-entry override (default: true)
export DUST_REENTRY_OVERRIDE=true

# Allow entries below significant floor (default: false)
# Note: Set in code via shared_state, not env var
```

### Runtime Control
```python
# Block all entries below 20 USDT (default)
shared_state.allow_entry_below_significant_floor = False

# Allow entries below 20 USDT (dust healing mode)
shared_state.allow_entry_below_significant_floor = True
```

---

## 🎯 Expected Outcomes

After implementation:

### Metric 1: Reduced Dust Creation
- **Before**: New positions created at < $20 (dust-creating entries)
- **After**: New positions minimum $20 (significant floor)
- **Expected**: 70-80% reduction in new dust positions from entries

### Metric 2: Flag Consistency
- **Before**: Mixed uppercase/lowercase usage, inconsistent guards
- **After**: Unified lowercase, consistent guards across all modules
- **Expected**: 0 guard bypasses due to flag case mismatch

### Metric 3: Operational Control
- **Before**: No way to explicitly control sub-floor entries
- **After**: Configurable via `allow_entry_below_significant_floor` flag
- **Expected**: Precise control over dust creation strategy

---

## 🔗 Related Issues

- Dust creation from trades
- Capital efficiency with dust positions
- Liquidation timing and profitability
- Entry/exit balance mechanics

---

## ✅ Sign-Off

**Implementation Owner**: AI Agent
**Review Required**: Code review + 1-hour integration test
**Deployment**: After passing Phase 2 testing
**Rollback Plan**: Revert commits 1-6 in reverse order

