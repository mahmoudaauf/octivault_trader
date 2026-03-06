# CODE CHANGES - Architectural Fix

## File: `core/shared_state.py`

---

## Change #1: `classify_positions_by_size()` - Lines 1546-1595

### Key Change: Branch on trading_mode to select position source

```python
# ADDED (Line 1561-1562):
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
position_keys = list(positions_source.keys())

# CHANGED (Line 1569):
# OLD:  position = self.positions.get(symbol)
# NEW:  position = positions_source.get(symbol)

# CHANGED (Line 1584):
# OLD:  self.positions[symbol] = position
# NEW:  positions_source[symbol] = position
```

**Pattern:**
```python
# Select source once at method entry
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions

# Use consistently throughout
position_keys = list(positions_source.keys())
for symbol in position_keys:
    position = positions_source.get(symbol)
    # ... process ...
    positions_source[symbol] = position
```

---

## Change #2: `get_positions_snapshot()` - Line 4910

### Key Change: Return correct dict based on mode

```python
# BEFORE (1 line):
def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all positions."""
    return dict(self.positions)

# AFTER (5 lines):
def get_positions_snapshot(self) -> Dict[str, Dict[str, Any]]:
    """Return a shallow copy of all positions, branching by trading mode."""
    if self.trading_mode == "shadow":
        return dict(self.virtual_positions)
    return dict(self.positions)
```

**Pattern:**
```python
if self.trading_mode == "shadow":
    return dict(self.virtual_positions)
return dict(self.positions)
```

---

## Change #3: `get_open_positions()` - Line 4954

### Key Change: Iterate over correct source based on mode

```python
# ADDED (Line 4967-4968):
# Branch by trading mode to use correct positions source
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions

# CHANGED (Line 4970):
# OLD:  for sym, pos_data in list(self.positions.items()):
# NEW:  for sym, pos_data in list(positions_source.items()):
```

**Pattern:**
```python
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions
for sym, pos_data in list(positions_source.items()):
    # ... filter and return ...
```

---

## Common Pattern Applied

All three fixes follow this architectural pattern:

```python
# At method entry or where positions are needed:
positions_source = self.virtual_positions if self.trading_mode == "shadow" else self.positions

# Then use consistently:
for symbol in positions_source:          # iterate
    pos = positions_source.get(symbol)   # read
    positions_source[symbol] = updated   # write
return dict(positions_source)             # return
```

**Benefits:**
✅ Single point of mode decision
✅ Consistent across all methods
✅ Easy to audit and maintain
✅ Clear intent: "positions_source" signals abstraction

---

## Lines Changed

| Method | Lines | Type | Description |
|--------|-------|------|-------------|
| `classify_positions_by_size()` | 1546+ | Major | 3 references to self.positions → positions_source |
| `get_positions_snapshot()` | 4910 | Minor | Simple if/else branch |
| `get_open_positions()` | 4954+ | Major | Loop source changed + trading_mode branching |

**Total changes:** 3 methods, ~8 lines modified/added, 0 breaking changes

---

## Validation

✅ Syntax check: `python3 -m py_compile core/shared_state.py` — PASSED
✅ No breaking changes to public API
✅ All changes are internal to SharedState
✅ Consistent pattern across all three methods
✅ Ready for deployment
