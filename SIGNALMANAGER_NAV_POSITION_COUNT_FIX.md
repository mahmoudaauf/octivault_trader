# SignalManager - NAV and Position Count Source Fix

## Summary

Fixed `SignalManager` class to properly support **NAV source** and **Position count source** parameters, enabling integration with `MetaController` and `CapitalGovernor`.

## Changes Made

### 1. Updated Constructor Signature

**Before:**
```python
def __init__(self, config, logger, signal_cache=None, intent_manager=None):
```

**After:**
```python
def __init__(self, config, logger, signal_cache=None, intent_manager=None, shared_state=None, position_count_source=None):
```

**New Parameters:**
- `shared_state`: Reference to SharedState instance (NAV source)
- `position_count_source`: Callable that returns current position count

### 2. Added NAV Retrieval Method

**New Method: `get_current_nav() -> float`**

```python
def get_current_nav(self) -> float:
    """
    Get current NAV from configured source (shared_state).
    
    Returns:
        Current NAV in USDT, or 0.0 if unavailable
    """
```

**Behavior:**
1. Returns 0.0 if no shared_state configured
2. Tries `nav` attribute first
3. Falls back to `portfolio_nav` attribute
4. Final fallback to `total_equity_usdt` attribute
5. Gracefully handles exceptions with debug logging
6. All numeric values properly validated and converted to float

### 3. Added Position Count Retrieval Method

**New Method: `get_position_count() -> int`**

```python
def get_position_count(self) -> int:
    """
    Get current position count from configured source.
    
    Returns:
        Number of open positions, or 0 if unavailable
    """
```

**Behavior:**
1. First tries `position_count_source` callable if provided
2. Falls back to counting positions from `shared_state.get_positions_snapshot()`
3. Only counts positions with `qty > 0` (ignores zero/dust positions)
4. Gracefully handles exceptions and returns 0 if unavailable
5. Returns integer count for safe integration with limits checking

### 4. Enhanced Initialization Logging

Added informational logging during initialization to confirm NAV and position count sources are properly wired:

```python
if shared_state or position_count_source:
    self.logger.info("[SignalManager] Initialized with NAV source: shared_state=%s, position_count_source=%s",
                   "yes" if shared_state else "no", "yes" if position_count_source else "no")
```

## Integration Points

### MetaController Integration

```python
from core.signal_manager import SignalManager

# In MetaController.__init__:
self.signal_manager = SignalManager(
    config=self.config,
    logger=self.logger,
    signal_cache=self.signal_cache,
    intent_manager=self.intent_manager,
    shared_state=self.shared_state,  # ← NAV source
    position_count_source=self._count_open_positions  # ← Position count source
)

# Later in code:
nav = self.signal_manager.get_current_nav()  # Get fresh NAV
positions = self.signal_manager.get_position_count()  # Get position count
```

### Direct Usage

```python
# Get NAV for decision-making
nav = signal_manager.get_current_nav()
if nav < 1000:
    # Apply MICRO_SNIPER constraints
    pass

# Get position count for limit enforcement
pos_count = signal_manager.get_position_count()
if pos_count >= max_positions:
    # Block new BUY orders
    return None
```

## Technical Details

### NAV Source Resolution

The `get_current_nav()` method tries three sources in order:

1. **Primary:** `shared_state.nav` (if present and non-zero)
2. **Secondary:** `shared_state.portfolio_nav` (if present and non-zero)
3. **Tertiary:** `shared_state.total_equity_usdt` (final fallback)

This provides flexibility across different SharedState implementations.

### Position Count Source Resolution

The `get_position_count()` method tries two sources in order:

1. **Primary:** `position_count_source()` callable (if provided)
2. **Fallback:** Count from `shared_state.get_positions_snapshot()` (if available)

This allows for:
- Custom position counting logic via callable
- Automatic fallback to SharedState snapshot enumeration
- Safe handling of missing positions data

### Error Handling

Both methods include comprehensive error handling:
- Try-except blocks wrap all external calls
- Exceptions logged at DEBUG level (non-fatal)
- Safe defaults returned (0.0 for NAV, 0 for count)
- No exception propagation to caller

## Backward Compatibility

✅ **Fully backward compatible**
- New parameters are optional (default to None)
- Methods gracefully handle missing sources
- Existing code that doesn't use these parameters continues to work
- No breaking changes to existing SignalManager interface

## Testing

### Basic Test

```python
signal_manager = SignalManager(
    config=config,
    logger=logger,
    shared_state=shared_state,
    position_count_source=lambda: 2
)

nav = signal_manager.get_current_nav()  # Returns NAV value
positions = signal_manager.get_position_count()  # Returns 2
```

### Without Sources

```python
signal_manager = SignalManager(
    config=config,
    logger=logger
)

nav = signal_manager.get_current_nav()  # Returns 0.0
positions = signal_manager.get_position_count()  # Returns 0
```

## Files Modified

- `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/core/signal_manager.py`
  - Added `shared_state` parameter to `__init__`
  - Added `position_count_source` parameter to `__init__`
  - Added `get_current_nav()` method
  - Added `get_position_count()` method
  - Enhanced initialization logging

## Validation

✅ Syntax check passed
✅ Type annotations correct
✅ Method signatures validated
✅ Backward compatibility maintained
✅ Error handling comprehensive
✅ Logging informative

## Next Steps

1. Update MetaController to pass `shared_state` to SignalManager
2. Update MetaController to pass `_count_open_positions` to SignalManager
3. Use `signal_manager.get_current_nav()` in decision-making
4. Use `signal_manager.get_position_count()` for limit enforcement
5. Monitor logs for NAV and position count accuracy
