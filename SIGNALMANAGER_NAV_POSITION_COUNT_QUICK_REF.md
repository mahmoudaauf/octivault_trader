# SignalManager NAV & Position Count - Quick Reference

## Fixed Issues

✅ **NAV Source** - SignalManager now retrieves current NAV from SharedState
✅ **Position Count Source** - SignalManager now retrieves position count from provided source

## Constructor

```python
SignalManager(
    config,                          # Config object
    logger,                          # Logger instance
    signal_cache=None,              # Optional cache
    intent_manager=None,            # Optional intent manager
    shared_state=None,              # ← NEW: NAV source
    position_count_source=None      # ← NEW: Position count source
)
```

## New Methods

### `get_current_nav() -> float`

Get current portfolio NAV in USDT.

```python
nav = signal_manager.get_current_nav()
# Returns: float (NAV in USDT, or 0.0 if unavailable)

if nav > 0:
    # Use fresh NAV for decision-making
    limits = capital_governor.get_position_limits(nav)
```

**Resolution Order:**
1. `shared_state.nav` (if non-zero)
2. `shared_state.portfolio_nav` (if non-zero)
3. `shared_state.total_equity_usdt` (final fallback)

### `get_position_count() -> int`

Get current open position count.

```python
positions = signal_manager.get_position_count()
# Returns: int (number of positions with qty > 0, or 0 if unavailable)

if positions >= max_positions:
    # Block new BUY orders
    return None
```

**Resolution Order:**
1. `position_count_source()` (if callable provided)
2. Count from `shared_state.get_positions_snapshot()` (fallback)

## Usage Examples

### Integration with MetaController

```python
class MetaController:
    def __init__(self, config, logger, shared_state):
        self.signal_manager = SignalManager(
            config=config,
            logger=logger,
            shared_state=shared_state,          # ← Pass SharedState
            position_count_source=self._count_open_positions  # ← Pass callable
        )
    
    async def run_cycle(self):
        # Get fresh NAV before decisions
        nav = self.signal_manager.get_current_nav()
        if nav <= 0:
            self.logger.warning("Invalid NAV, skipping cycle")
            return
        
        # Get position count
        pos_count = self.signal_manager.get_position_count()
        limits = self.capital_governor.get_position_limits(nav)
        
        if pos_count >= limits["max_concurrent_positions"]:
            self.logger.info("Position limit reached: %d/%d", 
                           pos_count, limits["max_concurrent_positions"])
```

### Direct Instantiation

```python
signal_manager = SignalManager(
    config=config,
    logger=logger,
    shared_state=shared_state,
    position_count_source=get_positions_callback
)

nav = signal_manager.get_current_nav()
positions = signal_manager.get_position_count()
```

### Minimal Setup (backward compatible)

```python
# Works without new parameters
signal_manager = SignalManager(config=config, logger=logger)

nav = signal_manager.get_current_nav()      # Returns 0.0
positions = signal_manager.get_position_count()  # Returns 0
```

## Error Handling

Both methods gracefully handle errors:

```python
try:
    # These never raise exceptions
    nav = signal_manager.get_current_nav()  # Returns 0.0 on error
    positions = signal_manager.get_position_count()  # Returns 0 on error
except Exception:
    # This won't happen - exceptions are caught internally
    pass
```

Errors are logged at DEBUG level:
```
[SignalManager] Failed to get NAV from shared_state: AttributeError
[SignalManager] Failed to count positions from shared_state: KeyError
```

## Testing Checklist

- [ ] Create instance with `shared_state`
- [ ] Create instance with `position_count_source` callable
- [ ] Create instance with both sources
- [ ] Create instance with neither (backward compatibility)
- [ ] Call `get_current_nav()` and verify value
- [ ] Call `get_position_count()` and verify count
- [ ] Check logs for initialization message
- [ ] Verify graceful fallback when sources unavailable
- [ ] Verify error messages appear at DEBUG level

## Integration Checklist

- [ ] Update MetaController to pass `shared_state` to SignalManager
- [ ] Update MetaController to pass `_count_open_positions` to SignalManager
- [ ] Update MetaController to call `signal_manager.get_current_nav()`
- [ ] Update MetaController to call `signal_manager.get_position_count()`
- [ ] Use fresh NAV in position limit checks
- [ ] Monitor logs for "Initialized with NAV source" message
- [ ] Verify position count is accurately tracked
- [ ] Test with live trading cycle

## Performance Notes

- `get_current_nav()`: O(1) - Direct attribute access
- `get_position_count()`: O(n) - Linear scan of positions (n = number of symbols)
- Both methods are synchronous (no async needed)
- No blocking I/O operations
- Safe to call frequently in decision loops

## Files Modified

- `core/signal_manager.py` - Constructor, new methods

## Related Documentation

- See `SIGNALMANAGER_NAV_POSITION_COUNT_FIX.md` for complete implementation details
- See `CAPITAL_GOVERNOR_COMPLETE_INDEX.md` for integration with CapitalGovernor
- See `NAV_SYNCHRONIZATION_FIX.md` for NAV freshness requirements
