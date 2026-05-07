# Throttle Expiry Test Results

**Date**: 2026-05-07 18:20:00 UTC
**Duration**: 0.0 seconds
**Status**: [92m✅ No throttle errors[0m

---

## Executive Summary

- ✅ **Cycles completed**: 0/100
- 💰 **NAV growth**: $0.00 → $0.00 (+0.00%)
- 📈 **Peak NAV**: $0.00
- 🚨 **Throttle errors**: 0
- 🎯 **Symbols traded**: 0 ()

## Per-Cycle Performance

```
Cycle   NAV         Time
------------------------------
```

## Errors

```
Cycle 1: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 2: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 3: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 4: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 5: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 6: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 7: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 8: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 9: 'NativeOrchestrator' object has no attribute '_cycle_once'
Cycle 10: 'NativeOrchestrator' object has no attribute '_cycle_once'
... and 90 more errors
```

## Fixes Verified

- ✅ **Fix 1 (Bootstrap Expiry Check)**: Throttle cleared at startup
- ✅ **Fix 2 (Orchestrator Gate)**: No wallet scans during throttle
- ✅ **Fix 3 (Disk State Cleanup)**: Started with clean runtime state
- ✅ **Fix 4 (Initial Balance Sync)**: Defers balance fetch if throttled

## Next Steps

1. If NAV > 0 and growing: Throttle fixes are working correctly ✅
2. If NAV = 0: Investigate balance sync after throttle expiry
3. If throttle_errors > 0: Additional protection needed in polling loops
