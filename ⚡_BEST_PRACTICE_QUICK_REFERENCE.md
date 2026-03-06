# ⚡ BEST PRACTICE QUICK REFERENCE

## 5-Point Checklist for Production Stability

### ✅ 1. Short Idempotency Window (8 seconds)

**Configuration:**
```python
self._active_order_timeout_s = 8.0
self._client_order_id_timeout_s = 8.0
```

**Effect**: Orders outside 8-second window are automatically retried instead of permanently blocked.

---

### ✅ 2. Track Active Orders, Not Rejections

**Mechanism:**
- Symbol/side level: Blocks duplicate attempts for same symbol/side pair
- Client order ID level: Insurance against double-submission to exchange
- Both use 8-second windows for automatic recovery

**Effect**: Duplicates are prevented (good) but not penalized (also good).

---

### ✅ 3. IDEMPOTENT Rejections Don't Count

**Configuration:**
```python
self._ignore_idempotent_in_rejection_count = True
self._rejection_exempt_reasons = {"IDEMPOTENT", "ACTIVE_ORDER"}
```

**Effect**: Network glitches don't trigger dust retirement or position locks.

---

### ✅ 4. Auto-Reset Rejection Counters (60 seconds)

**Configuration:**
```python
self._rejection_reset_window_s = 60.0
self._maybe_auto_reset_rejections()  # Called during order placement
```

**Effect**: Stale rejection counts automatically clear, no manual intervention needed.

---

### ✅ 5. Bootstrap Trades Always Bypass Safety Gates

**Configuration:**
```python
allow_bootstrap_bypass = is_bootstrap_signal and self._is_bootstrap_allowed()

if not allow_bootstrap_bypass:
    # Apply idempotency checks
else:
    # Bypass for initial position opens
```

**Effect**: Portfolio initialization and restarts always work.

---

## Expected Behavior Under Network Glitches

### Scenario: Network timeout during order placement

```
Time 0.0s: Send BUY order → timeout but filled at exchange
Time 0.5s: Retry BUY order
└─ Status: SKIPPED (reason=IDEMPOTENT)
└─ Rejection count: NOT incremented ✅

Time 1.0s: Another retry
└─ Status: SKIPPED (reason=IDEMPOTENT)
└─ Rejection count: NOT incremented ✅

Time 8.5s: Next attempt
└─ Previous entry EXPIRED from cache
└─ Status: SUCCESS (fetches from exchange, confirms filled) ✅
└─ No manual intervention needed 🎉
```

---

## Configuration Tuning

### If network is very unstable (>50% timeouts):
```python
self._active_order_timeout_s = 10.0  # Increase to 10s
self._client_order_id_timeout_s = 10.0
```

### If network is very stable (<5% timeouts):
```python
self._active_order_timeout_s = 5.0   # Decrease to 5s
self._client_order_id_timeout_s = 5.0
```

### Default for normal conditions:
```python
self._active_order_timeout_s = 8.0   # Perfect for most networks
self._client_order_id_timeout_s = 8.0
```

---

## Monitoring Checklist

Before deploying to production:

- [ ] `_active_order_timeout_s = 8.0` (not 30.0)
- [ ] `_client_order_id_timeout_s = 8.0` (not 60.0)
- [ ] `_rejection_reset_window_s = 60.0`
- [ ] `_ignore_idempotent_in_rejection_count = True`
- [ ] `_maybe_auto_reset_rejections()` being called in order placement
- [ ] IDEMPOTENT responses NOT recording rejections (line 6265)
- [ ] Bootstrap flag working (test with `_bootstrap=True`)

---

## Logging Guide

### What to expect in logs (normal operation):

```
# Occasional ACTIVE_ORDER blocks (expected, not a problem)
[EM:ACTIVE_ORDER] Order in flight for AAPL BUY (1.2s ago); skipping.

# Automatic recovery (exactly what we want to see)
[EM:RETRY_ALLOWED] Previous attempt for AAPL BUY timed out (8.5s); allowing fresh retry.

# Garbage collection (infrequent, healthy)
[EM:DupIdGC] Garbage collected 523 stale client_order_ids, dict_size=3156

# Auto-reset working (confirming no manual intervention needed)
[EM:REJECTION_RESET] Auto-reset rejection counter for AAPL SELL (no rejections for 60s)
```

### Red flags (investigate if you see these):

```
# ❌ OLD WINDOW BEING USED (30s or 60s)
[EM:IDEMPOTENT] Active order exists for AAPL BUY (0.5s ago); skipping.

# ❌ CACHE GROWING UNBOUNDED (GC not running)
[EM:DupIdGC] dict_size=150000 (should be <5000)

# ❌ IDEMPOTENT BEING COUNTED (rejection counter incrementing)
[EXEC_REJECT] symbol=AAPL side=BUY reason=IDEMPOTENT count=1

# ❌ STALE ENTRIES NOT CLEARING (should see RETRY_ALLOWED)
[EM:ACTIVE_ORDER] Order in flight for AAPL BUY (45.2s ago); skipping.
```

---

## Code Locations

| Feature | File | Lines |
|---------|------|-------|
| Configuration | execution_manager.py | 1920-1945 |
| Symbol/side check | execution_manager.py | 7290-7315 |
| Client ID check | execution_manager.py | 4355-4390 |
| Auto-reset method | execution_manager.py | 4325-4350 |
| Auto-reset trigger | execution_manager.py | 7282-7287 |
| IDEMPOTENT skip | execution_manager.py | 6265-6270 |

---

## Summary

This configuration solves the permanent blocking problem by:

1. ✅ Using SHORT idempotency windows (8s, not 30-60s)
2. ✅ Not penalizing IDEMPOTENT rejections
3. ✅ Auto-clearing stale entries and retry locks
4. ✅ Auto-resetting rejection counters after inactivity
5. ✅ Allowing bootstrap trades to always work

**Result**: Production-grade stability with automatic recovery from network glitches.

---

**Last Updated**: 2026-03-04
**Status**: ✅ READY FOR DEPLOYMENT
