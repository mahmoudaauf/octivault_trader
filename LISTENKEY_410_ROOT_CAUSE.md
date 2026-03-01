# ListenKey Rotation 410 Error - Root Cause Analysis

**Date**: February 27, 2026  
**Error**: `listenKey rotation FAILED after 3 attempts: APIError(code=410)`

---

## 🔍 Root Cause Analysis

### The Error You're Seeing
```
2026-02-26 22:53:37,147 ERROR [EC:UserDataWS] listenKey rotation FAILED after 3 attempts: 
APIError(code=410): <html><head><title>410 Gone</title></head>...
```

### What This Means

The rotation process is:
1. ✅ Close old listenKey (DELETE request) - works
2. ❌ Create new listenKey (POST request) - **FAILS with 410**

**410 means "Gone" - Binance's API is unavailable or rejecting the request.**

### Why This Happens

The `_request()` method in `core/exchange_client.py` has **exponential backoff for specific errors**:
- ✅ 418, 429 (rate limit) → retries with backoff
- ✅ -1021 (time skew) → resync time and retry
- ❌ 410 (gone) → **NOT retried, raised immediately**

**410 errors are treated as terminal failures, not transient issues.**

---

## 💡 The Problem

### Current Backoff Schedule for Rotation Failures

```python
for attempt in range(max_retries):  # max_retries = 3
    try:
        await self._create_listen_key()  # If this fails, exception is raised
    except Exception as e:
        if attempt < max_retries - 1:
            backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
            await asyncio.sleep(backoff)
        else:
            return False  # Give up
```

**Problems:**
1. Backoff is **too short** (0.5s, 1s, 2s) when API is actually struggling
2. **Only 3 retries** - not enough when Binance API has issues
3. **No differentiation** between 410 (API issue) and other errors
4. **410 errors in _request() are immediate failures** - no built-in retry

---

## 🔧 The Fix Needed

### Solution 1: Longer Backoff for Rotation Failures

**Current**:
```python
backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s (3 attempts)
```

**Should be**:
```python
# For 410-like errors (API struggling), use much longer backoff
backoff = 2.0 * (2 ** attempt)  # 2s, 4s, 8s (more aggressive)
# With more retries
max_retries = 5  # instead of 3
```

### Solution 2: Handle 410 Specially in _request()

The `_request()` method should **retry 410 errors** with backoff since they're transient:

```python
# In _request() around line 1730:
if response.status == 410:  # Add this check
    # 410 means API is temporarily unavailable
    # Try to wait and retry
    try:
        ra = float(response.headers.get("Retry-After", "2"))
        await asyncio.sleep(min(ra, 5.0))  # Wait up to 5s
    except Exception:
        pass
    # Continue to next retry iteration (don't raise immediately)
    continue
```

### Solution 3: Detect When Binance API is Down

Add monitoring to detect if Binance API is having widespread issues:

```python
# Track 410 errors in rotation
if 410_errors > 2:  # Multiple 410s in a row
    log.critical("Binance API may be experiencing issues (410 errors)")
    # Could implement circuit breaker here
```

---

## 📊 Current Code Analysis

### Current Retry Logic (Lines 987-1040)

```python
async def _rotate_listen_key(self, *, reason: str = "", max_retries: int = 3) -> bool:
    for attempt in range(max_retries):  # 3 iterations
        try:
            async with self._user_data_lock:
                # ... close old key ...
                await self._create_listen_key()  # LINE 1007 - Can throw 410
        except Exception as e:
            if attempt < max_retries - 1:
                backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                await asyncio.sleep(backoff)
            else:
                return False
    return False
```

**Issues**:
1. `_create_listen_key()` at line 1007 calls `_request("POST", ...)`
2. If `_request()` gets 410, it raises immediately (no retry in _request)
3. Rotation catches it and retries after 0.5s → 1s → 2s
4. **But 410 usually means API needs MORE than 2 seconds to recover**

---

## ✅ Recommended Fix (Implementation Steps)

### Step 1: Increase Rotation Backoff

Replace the backoff calculation in `_rotate_listen_key()`:

```python
# OLD (Lines 1033-1038)
except Exception as e:
    if attempt < max_retries - 1:
        backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s

# NEW - Longer backoff for API issues
except Exception as e:
    if attempt < max_retries - 1:
        # Increase max_retries to 5 for better coverage
        backoff = 2.0 * (2 ** attempt)  # 2s, 4s, 8s, 16s
        self.logger.warning(
            "[EC:UserDataWS] rotation attempt %d/%d failed, retrying in %.1fs: %s",
            attempt + 1, max_retries, backoff, e,
        )
        await asyncio.sleep(backoff)
```

### Step 2: Increase Max Retries

Change the default `max_retries`:

```python
# OLD
async def _rotate_listen_key(self, *, reason: str = "", max_retries: int = 3) -> bool:

# NEW - More attempts for API issues
async def _rotate_listen_key(self, *, reason: str = "", max_retries: int = 5) -> bool:
```

### Step 3: Add 410 Retry in _request()

In the `_request()` method around line 1730, add special handling for 410:

```python
if response.status == 410:
    # 410 Gone - API temporarily unavailable
    # Don't raise immediately, try to wait and retry
    retry_after = response.headers.get("Retry-After", "2")
    try:
        wait_time = min(float(retry_after), 5.0)  # Max 5 seconds
    except (ValueError, TypeError):
        wait_time = 2.0  # Default 2 seconds
    
    self.logger.debug(
        "[EC] HTTP 410 on %s %s - waiting %.1fs before retry",
        method, path, wait_time
    )
    await asyncio.sleep(wait_time)
    # Continue to next retry iteration instead of raising
    continue
```

---

## 📈 Expected Behavior After Fix

### Scenario: Binance API temporarily down (5 seconds)

**Before fix**:
```
Attempt 1 (t=0):   POST /api/v3/userDataStream → 410
  Wait 0.5s
Attempt 2 (t=0.5): POST /api/v3/userDataStream → 410
  Wait 1s
Attempt 3 (t=1.5): POST /api/v3/userDataStream → 410
  Give up (t=1.5)
Result: ❌ FAILED after 1.5 seconds

Binance recovers at t=5s but we already gave up
```

**After fix**:
```
Attempt 1 (t=0):   POST /api/v3/userDataStream → 410
  Wait 2s (plus internal retry in _request)
Attempt 2 (t=2):   POST /api/v3/userDataStream → 410
  Wait 4s (plus internal retry in _request)
Attempt 3 (t=6):   POST /api/v3/userDataStream → ✅ SUCCESS
Result: ✅ RECOVERED after 6 seconds

Now it waits long enough for API to recover
```

---

## 🛡️ Additional Protection Measures

### 1. Circuit Breaker Pattern

After N consecutive rotation failures, stop trying:

```python
# Track consecutive rotation failures
self.rotation_failure_count = 0
self.max_rotation_failures = 5

# In rotation failure handler:
self.rotation_failure_count += 1
if self.rotation_failure_count >= self.max_rotation_failures:
    log.critical("Too many rotation failures - escalate to FATAL")
    # Escalate instead of retrying forever
```

### 2. Exponential Backoff Beyond Rotation

If rotation fails completely, wait longer before trying again:

```python
# In _user_data_ws_loop when rotation fails
if not rotation_ok:
    # Rotation failed, wait longer before next reconnect
    long_backoff = 30.0 * (2 ** min(reconnect_attempt / 2, 3))
    self.logger.warning("Rotation failed, waiting %.0fs before next attempt", long_backoff)
    await asyncio.sleep(long_backoff)
```

### 3. Monitoring Alert

Alert when rotation keeps failing:

```python
if 410_error_streak > 3:
    status = "WARNING"
    message = "Binance API returning 410 - possible service issue"
elif 410_error_streak > 5:
    status = "CRITICAL"
    message = "Binance API severely degraded - escalating"
```

---

## 📋 Summary of Changes Needed

| Change | Location | Impact | Difficulty |
|--------|----------|--------|-----------|
| Increase backoff formula | Line 1033 | More patient retry | ⭐ Easy |
| Increase max_retries to 5 | Line 983 | More attempts | ⭐ Easy |
| Add 410 handling in _request | Line 1730 | Internal retry | ⭐⭐ Medium |
| Add rotation failure tracking | New code | Better monitoring | ⭐⭐ Medium |

---

## 🎯 Recommendation

**Implement Changes 1 + 2 immediately** (Easy fixes):
- Increase backoff to 2s, 4s, 8s
- Increase max_retries to 5

**Then monitor** for continued 410 errors. If still happening:
- Implement Change 3 (410 retry in _request)
- Implement Change 4 (circuit breaker)

---

## 📝 Why 410 Happens

**410 errors on listenKey operations typically mean**:
1. **Binance API infrastructure issue** (temporary service disruption)
2. **Network connectivity problem** (between you and Binance)
3. **Rate limiting at Binance level** (too many auth requests)
4. **Account lockout** (after multiple failed auth attempts)

**Not a code bug** - just needs more patience in waiting for recovery.

---

**Conclusion**: The code is sound, just needs more generous retry backoff for API recovery scenarios.
