# 🔧 TECHNICAL REFERENCE - Architectural Fix Details

**Last Updated:** March 3, 2026
**Component:** core/shared_state.py
**Version:** 2.0.1+

---

## Implementation Details

### Core Pattern

```python
# The fix applies this pattern to 3 methods:
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
```

This single line makes each method trading-mode aware without coupling external code.

---

## Method-by-Method Breakdown

### 1. `classify_positions_by_size()` at Line 1546

**Purpose:** Classify positions into SIGNIFICANT and DUST categories

**Change Type:** MAJOR - Updates position store, must use correct source

**Implementation:**
```python
async def classify_positions_by_size(self) -> Dict[str, List[str]]:
    significant = []
    dust = []

    # FIX: Determine source based on mode (Line 1561)
    positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
    
    # FIX: Use source for iteration (Line 1563)
    position_keys = list(positions_source.keys())

    for symbol in position_keys:
        async with self._lock_context("positions"):
            # FIX: Read from source (Line 1569)
            position = positions_source.get(symbol)
            
            # ... classification logic ...
            
            # FIX: Write to source (Line 1584)
            positions_source[symbol] = position
```

**Why This Matters:**
- Writes position updates to correct store
- In shadow mode: updates virtual_positions
- In live mode: updates positions
- MetaController doesn't know which one is being updated ✅

**Async Considerations:**
- Uses `_lock_context("positions")` for thread safety
- Works with both sources (they're both simple dicts)

---

### 2. `get_positions_snapshot()` at Line 4910

**Purpose:** Return a shallow copy of all current positions

**Change Type:** SIMPLE - Pure getter, no side effects

**Implementation:**
```python
def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all positions, branching by trading mode."""
    if self.trading_mode == "shadow":
        return dict(self.virtual_positions)
    return dict(self.positions)
```

**Why This Matters:**
- Single point of access for "all positions"
- Guarantees correct dict based on mode
- No coupling to shadow mode logic needed by callers

**Call Sites Should Be:**
```python
# Good (uses abstraction):
snapshot = shared_state.get_positions_snapshot()

# Bad (bypasses abstraction):
snapshot = shared_state.positions  # May be wrong in shadow mode!
```

---

### 3. `get_open_positions()` at Line 4954

**Purpose:** Return only OPEN SIGNIFICANT positions with state healing

**Change Type:** MAJOR - Filters from store, must use correct source

**Implementation:**
```python
def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
    self._sync_heal_position_states()
    result = {}
    
    # FIX: Use correct source (Line 4967-4968)
    positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
    
    # FIX: Iterate correct source (Line 4970)
    for sym, pos_data in list(positions_source.items()):
        qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        if pos_data.get("is_significant", False) and pos_data.get("open_position", False):
            result[sym] = pos_data
    return result
```

**Key Features:**
- `_sync_heal_position_states()` updates states as side effect
- Healing affects correct dict (virtual or real)
- Filtering logic identical for both modes
- Only difference: which dict to iterate

---

## Data Flow Diagram

```
MetaController.decide_rebalance()
  │
  ├─> shared_state.get_positions_snapshot()
  │   │
  │   └─> if trading_mode == "shadow"
  │       ├─ YES: return dict(virtual_positions)
  │       └─ NO:  return dict(positions)
  │
  ├─> shared_state.get_open_positions()
  │   │
  │   └─> _sync_heal_position_states()
  │   └─> if trading_mode == "shadow"
  │       ├─ YES: iterate virtual_positions
  │       └─ NO:  iterate positions
  │
  └─> [MetaController makes decision using data]
```

**Key Invariant:**
```
For any symbol S in MetaController's decisions:
  - Position data came from correct source
  - Source determined by SharedState.trading_mode
  - MetaController never directly checks mode
```

---

## State Management During Fix

### Before Fix (BROKEN)
```
shared_state.positions
shared_state.virtual_positions
                │
                └─ External code doesn't know which to read
                └─ Risk of reading wrong dict
                └─ MetaController might use stale live positions in shadow mode
```

### After Fix (CORRECT)
```
shared_state.positions ──┐
shared_state.virtual_pos ├─ SharedState.get_positions_snapshot()
                         │  ├─ Checks trading_mode
                         │  └─ Returns correct one
                         └─ External code gets right dict
                         └─ MetaController uses correct positions
```

---

## Testing Strategy

### Unit Test: `classify_positions_by_size()` in Shadow Mode

```python
async def test_classify_positions_shadow_mode():
    ss = SharedState(config=MockConfig(trading_mode="shadow"))
    
    # Add virtual position
    ss.virtual_positions["BTC/USD"] = {
        "quantity": 1.0,
        "price": 50000.0
    }
    
    # Classify
    result = await ss.classify_positions_by_size()
    
    # Verify: position was updated in virtual_positions, NOT positions
    assert "BTC/USD" in ss.virtual_positions
    assert ss.virtual_positions["BTC/USD"]["status"] in ["SIGNIFICANT", "DUST"]
    assert "BTC/USD" not in ss.positions  # Still empty!
```

### Unit Test: `get_positions_snapshot()` Mode Switching

```python
def test_positions_snapshot_mode_switching():
    ss = SharedState(config=MockConfig(trading_mode="live"))
    
    # Add live position
    ss.positions["ETH/USD"] = {"qty": 10.0}
    ss.virtual_positions["ADA/USD"] = {"qty": 100.0}
    
    # In live mode
    snapshot = ss.get_positions_snapshot()
    assert "ETH/USD" in snapshot
    assert "ADA/USD" not in snapshot  # Virtual positions hidden!
    
    # Switch to shadow (if runtime switching allowed)
    ss.trading_mode = "shadow"
    snapshot = ss.get_positions_snapshot()
    assert "ADA/USD" in snapshot
    assert "ETH/USD" not in snapshot  # Live positions hidden!
```

---

## Error Prevention

### Potential Issue #1: Null/Missing Source

```python
# SAFE (fix prevents this):
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
# positions_source is ALWAYS one of the two dicts, never None

# UNSAFE (before fix):
for sym in self.positions:  # What if shadow mode?
```

### Potential Issue #2: Stale Mode Value

```python
# Fix assumes trading_mode is stable during method execution
# Should be true because:
# - trading_mode is set at boot
# - Only changes with explicit configuration
# - Not modified during normal trading

# If runtime mode switching is needed:
mode_snapshot = self.trading_mode  # Capture at entry
source = self.virtual_positions if mode_snapshot == "shadow" else self.positions
```

### Potential Issue #3: Bypassing Abstraction

```python
# BAD (should not do):
if self.trading_mode == "shadow":
    pos = self.virtual_positions
else:
    pos = self.positions

# GOOD (use abstraction):
snapshot = shared_state.get_positions_snapshot()
```

---

## Performance Considerations

**Time Complexity:**
- `classify_positions_by_size()`: O(n) where n = number of positions
  - No change from before (same iteration)
  - One additional if-check per iteration: negligible
  
- `get_positions_snapshot()`: O(n) where n = number of positions
  - Before: dict() call on one dict
  - After: one if-check, then dict() call
  - Performance impact: **negligible**

- `get_open_positions()`: O(n) where n = number of positions
  - One if-check at method entry
  - Then same filtering logic
  - Performance impact: **negligible**

**Memory Usage:** No change

**Recommendation:** No performance concerns. Fixes are safe for production.

---

## Future Refactoring Opportunities

### Possible Enhancement: Extract Helper Method

```python
def _get_positions_source(self) -> Dict[str, Dict[str, Any]]:
    """Get the authoritative positions dict for current mode."""
    return self.virtual_positions if self.trading_mode == "shadow" else self.positions
```

**Benefit:** DRY principle (reduces 3 identical if-checks to 1)
**Cost:** One extra function call
**Recommendation:** Could be done later if pattern repeats elsewhere

### Possible Enhancement: Explicit API Methods

```python
def get_virtual_positions(self) -> Dict[str, Dict[str, Any]]:
    """For testing: get virtual positions (shadow mode only)."""
    return dict(self.virtual_positions)

def get_live_positions(self) -> Dict[str, Dict[str, Any]]:
    """For testing: get live positions (live mode only)."""
    return dict(self.positions)
```

**Benefit:** Clearer test intent
**Current Status:** Not needed yet, but possible future enhancement

---

## Verification Checklist for Reviewers

- [ ] All three methods follow the same pattern
- [ ] Pattern is: `source = virtual if shadow else real`
- [ ] No direct `self.positions` access in shadow-aware methods
- [ ] No direct `self.virtual_positions` access except through source
- [ ] Comments clearly mark the fixes
- [ ] No new dependencies introduced
- [ ] No changes to method signatures
- [ ] Return types unchanged
- [ ] Performance unaffected

---

**Document Status:** ✅ COMPLETE
**Technical Debt:** ✅ ELIMINATED
**Code Quality:** ✅ IMPROVED
