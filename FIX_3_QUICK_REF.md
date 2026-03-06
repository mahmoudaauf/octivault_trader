# ⚙️ FIX 3 Quick Reference — Bootstrap Loop Throttle

## Problem
```
[Meta] FLAT_PORTFOLIO detected. Enforcing BUY-only logic.
[Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
[Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
[Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
...repeats EVERY TICK...
```

## Solution
Throttle the "no valid signals" log message to **once per 60 seconds** instead of every tick.

## Implementation

### Location 1: Initialize throttle state
**File:** `core/meta_controller.py`, Lines 1307-1309
```python
# ⚙️ FIX 3: Bootstrap loop throttling (once per 60 seconds max)
self._last_bootstrap_no_signal_log_ts = 0.0  # Timestamp of last "no valid BUY" log
self._bootstrap_throttle_seconds = 60.0       # Throttle interval (configurable)
```

### Location 2: Apply throttle guard
**File:** `core/meta_controller.py`, Lines 10425-10432
```python
# ⚙️ FIX 3: Throttle bootstrap no-signal log to once per 60 seconds
now = time.time()
if (now - self._last_bootstrap_no_signal_log_ts) >= self._bootstrap_throttle_seconds:
    self.logger.warning(
        "[Meta:BootstrapThrottle] No valid BUY signals for FLAT_PORTFOLIO (throttled @ 60s intervals). "
        "Next update in ~%.0fs. This is not fatal—waiting for strategy to generate signals.",
        self._bootstrap_throttle_seconds
    )
    self._last_bootstrap_no_signal_log_ts = now
```

## Result

### Before FIX 3
```
10:00:00 [Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
10:00:00 [Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
10:00:00 [Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
10:00:00 [Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
10:00:00 [Meta] FLAT_PORTFOLIO but no valid & executable BUY signals found.
...FLOOD...
```

### After FIX 3
```
10:00:00 [Meta:BootstrapThrottle] No valid BUY signals... (throttled @ 60s)
10:01:00 [Meta:BootstrapThrottle] No valid BUY signals... (throttled @ 60s)
10:02:00 [Meta:BootstrapThrottle] No valid BUY signals... (throttled @ 60s)
```

## Configuration

Change throttle interval (line 1309):
```python
self._bootstrap_throttle_seconds = 30.0    # More frequent (every 30s)
self._bootstrap_throttle_seconds = 60.0    # Default (every 60s)
self._bootstrap_throttle_seconds = 300.0   # Silent (every 5 min)
self._bootstrap_throttle_seconds = 0.0     # Disabled (debug mode)
```

## Status

- **Implementation:** ✅ Complete
- **Verification:** ✅ Verified
- **Risk:** ✅ Zero (cosmetic only)
- **Functional Impact:** ✅ None (logging only)
- **Bootstrap Logic:** ✅ Unchanged

## Related Fixes

| Fix | Issue | Status |
|-----|-------|--------|
| FIX #1 | Shadow mode missing TRADE_EXECUTED | ✅ Done |
| FIX #2 | Dual accounting systems | ✅ Done |
| FIX #3 | Bootstrap loop flooding | ✅ Done (this) |

---

**Total fixes in session:** 3/3 ✅ Complete
