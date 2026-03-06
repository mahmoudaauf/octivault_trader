# Symbol-Scoped Dust Cleanup Implementation Guide

**Date**: March 2, 2026  
**Feature**: Per-symbol dust state tracking and cleanup  
**Status**: Design & Implementation Plan  

---

## Overview

**Current State**: Global dust tracking
- `_bootstrap_dust_bypass_used` - set of symbols
- `_consolidated_dust_symbols` - set of symbols  
- `_dust_merge_attempts` - symbol -> attempt data
- `_dust_merge_bootstrap_epoch` - global epoch

**Desired State**: Symbol-scoped dust cleanup
- Each symbol maintains its own dust lifecycle
- Automatic cleanup of stale dust state per symbol
- TTL-based expiration (configurable timeout)
- Per-symbol cooldown management
- Prevents accumulation of stale dust metadata

---

## Implementation Strategy

### 1. Data Structure: Symbol-Scoped Dust State

```python
# NEW: Per-symbol dust state tracking
self._symbol_dust_state = {}  # symbol -> {
#     "bypass_used": bool,
#     "consolidated": bool,
#     "merge_attempts": [...],
#     "last_dust_tx": timestamp,
#     "state_created_at": timestamp,
# }

# NEW: Symbol-scoped dust state cleanup tracking
self._symbol_dust_cleanup_timeout = 3600.0  # 1 hour default (configurable)
```

### 2. Symbol-Scoped Dust Initialization

```python
def _init_symbol_dust_state(self, symbol: str) -> None:
    """Initialize dust state for a specific symbol."""
    if symbol not in self._symbol_dust_state:
        self._symbol_dust_state[symbol] = {
            "bypass_used": False,
            "consolidated": False,
            "merge_attempts": [],
            "last_dust_tx": None,
            "state_created_at": time.time(),
        }
```

### 3. Symbol-Scoped Dust Cleanup

```python
async def _cleanup_symbol_dust_state(self, symbol: str) -> bool:
    """
    Clean up stale dust state for a specific symbol.
    
    Args:
        symbol: The symbol to cleanup
        
    Returns:
        bool: True if state was cleaned, False if still active
    """
    if symbol not in self._symbol_dust_state:
        return False
    
    state = self._symbol_dust_state[symbol]
    created_at = state.get("state_created_at", time.time())
    age_sec = time.time() - created_at
    
    timeout_sec = float(
        getattr(self.config, "SYMBOL_DUST_STATE_TIMEOUT_SEC", 3600.0) or 3600.0
    )
    
    # Check if state is stale
    if age_sec > timeout_sec:
        # Check if there's recent activity
        last_dust_tx = state.get("last_dust_tx")
        if last_dust_tx is not None:
            activity_age = time.time() - last_dust_tx
            if activity_age < 300.0:  # Recent activity (< 5 min)
                return False
        
        # Clean up stale state
        self._symbol_dust_state.pop(symbol, None)
        self.logger.info(
            "[Meta:DustCleanup] Symbol %s: Cleaned up stale dust state "
            "(age=%d sec > timeout=%d sec)",
            symbol, int(age_sec), int(timeout_sec)
        )
        return True
    
    return False
```

### 4. Background Cleanup Loop

```python
async def _run_symbol_dust_cleanup_cycle(self) -> int:
    """
    Periodically clean up stale dust state for all symbols.
    
    Returns:
        int: Number of symbols with dust state cleaned
    """
    try:
        cleaned_count = 0
        for symbol in list(self._symbol_dust_state.keys()):
            if await self._cleanup_symbol_dust_state(symbol):
                cleaned_count += 1
        
        return cleaned_count
    except Exception as e:
        self.logger.error("[Meta:DustCleanup] Error cleaning up symbol dust state: %s", e)
        return 0
```

### 5. Integration into Cleanup Cycle

```python
# In _run_cleanup_cycle(), add:

# ═════════════════════════════════════════════════════════════════
# SYMBOL-SCOPED DUST STATE CLEANUP
# ═════════════════════════════════════════════════════════════════
try:
    dust_cleaned = await self._run_symbol_dust_cleanup_cycle()
    if dust_cleaned > 0:
        self.logger.info(
            "[Meta:Cleanup] Cleaned up dust state for %d symbols",
            dust_cleaned
        )
except Exception as e:
    self.logger.debug("[Meta:Cleanup] Dust state cleanup error: %s", e)
```

---

## Configuration Parameters

Add to `config.py`:

```python
# Symbol-scoped dust state timeout (seconds)
# After this duration, dust metadata for a symbol is cleaned up if inactive
SYMBOL_DUST_STATE_TIMEOUT_SEC = 3600.0  # 1 hour default

# Recent activity threshold (seconds)
# If dust transaction occurred within this time, state is preserved
SYMBOL_DUST_ACTIVITY_THRESHOLD_SEC = 300.0  # 5 minutes
```

---

## Benefits

✅ **Per-Symbol Isolation**: Each symbol's dust state managed independently  
✅ **Automatic Cleanup**: Stale dust metadata removed after timeout  
✅ **Activity Awareness**: Recent dust activity preserved  
✅ **Memory Efficient**: Prevents unbounded growth of dust tracking dicts  
✅ **Observable**: Logs track dust state cleanup events  

---

## Implementation Checklist

- [ ] Create symbol dust state dict in `_init_symbol_dust_state()`
- [ ] Implement `_cleanup_symbol_dust_state()` method
- [ ] Implement `_run_symbol_dust_cleanup_cycle()` method
- [ ] Integrate into `_run_cleanup_cycle()`
- [ ] Add configuration parameters
- [ ] Update dust merge retry logic to use symbol-scoped state
- [ ] Update dust consolidation tracking to use symbol-scoped state
- [ ] Add monitoring and logging
- [ ] Create comprehensive documentation
- [ ] Test with high symbol count scenarios

---

## Example: Before vs After

### Before (Global)
```python
self._dust_merge_attempts = {
    "BTCUSDT": {...},
    "ETHUSDT": {...},
    "BNBUSDT": {...},
    # ... 100+ symbols accumulate here forever
}
```

### After (Symbol-Scoped with Cleanup)
```python
self._symbol_dust_state = {
    "BTCUSDT": {
        "merge_attempts": [...],
        "state_created_at": 1709400000,
        # Automatically cleaned up after 1 hour if inactive
    },
    "ETHUSDT": {
        "merge_attempts": [...],
        "state_created_at": 1709400000,
        # Cleaned up automatically
    },
    # Old, inactive symbols are automatically pruned
}
```

---

## Next Steps

1. Implement symbol-scoped data structures
2. Create cleanup methods
3. Integrate into main cleanup cycle
4. Add configuration and observability
5. Test with real trading scenarios
6. Document behavior and configuration

