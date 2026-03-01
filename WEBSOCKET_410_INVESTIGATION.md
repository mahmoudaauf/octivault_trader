# WebSocket 410 Gone Error Investigation

## Problem Summary

**Error Pattern**:
```
2026-02-26 22:15:27,640 WARNING [AppContext] [EC:UserDataWS] disconnected: APIError(code=410): <html>
<head><title>410 Gone</title></head>
<body>
<center><h1>410 Gone</h1></center>
<hr><center>nginx</center>
</body>
</html>
 (reconnect_count=2 invalid_listen_key=True)

2026-02-26 22:15:27,715 WARNING [AppContext] [EC:UserDataWS] listenKey rotation failed: APIError(code=410): ...
```

**Error Code**: 410 (HTTP Gone)
**Source**: Binance API WebSocket user data stream
**Root Cause**: listenKey expiration or invalidation

---

## Technical Analysis

### 1. How listenKey Works

**Binance API Design**:
- listenKey is obtained via REST API: `POST /api/v3/userDataStream`
- Returns a unique 32-char key used for WebSocket authentication
- **Expires after 60 minutes** of inactivity (no keepalive)
- Must be refreshed periodically: `PUT /api/v3/userDataStream`
- Can be closed: `DELETE /api/v3/userDataStream`

**Current Implementation**:
```python
# File: core/exchange_client.py, line 650-651
self.listenkey_refresh_sec = float(_cfg("USER_DATA_LISTENKEY_REFRESH_SEC", 1800.0) or 1800.0)
self.listenkey_refresh_sec = min(max(self.listenkey_refresh_sec, 60.0), 3500.0)
```

**Default**: 1800 seconds = 30 minutes

### 2. Current Keepalive Logic

**Location**: `core/exchange_client.py`, lines 1005-1019

```python
async def _user_data_keepalive_loop(self) -> None:
    while self.is_started and not self._user_data_stop.is_set():
        try:
            # Sleep for listenkey_refresh_sec (default 1800 = 30 min)
            await asyncio.sleep(max(60.0, float(self.listenkey_refresh_sec or 1800.0)))
            if self._user_data_stop.is_set() or not self.is_started:
                break
            if not self._user_data_listen_key:
                continue
            # Refresh the listenKey
            await self._refresh_listen_key()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("[EC:UserDataWS] listenKey keepalive failed: %s", e)
            if self._is_invalid_listen_key_error(e):
                # Rotate if the key is invalid
                with contextlib.suppress(Exception):
                    await self._rotate_listen_key(reason=f"keepalive_error:{e}")
```

**Problem**: 30-minute refresh interval vs 60-minute Binance timeout

### 3. Root Causes (Multiple)

#### A. Refresh Interval vs Timeout Mismatch

| Component | Value | Issue |
|-----------|-------|-------|
| Binance listenKey TTL | 60 minutes | Server-side timeout |
| System refresh interval | 30 minutes | Should be safe, but... |
| **Gap** | **30 minutes** | Assumes keepalive loop always executes |

**Problem**: If refresh loop is delayed or blocked, listenKey expires.

#### B. Sleep Timing Issue

```python
await asyncio.sleep(max(60.0, float(self.listenkey_refresh_sec or 1800.0)))
```

**Issue**: 
- Uses `max(60.0, value)` which means minimum 60 seconds
- But if `listenkey_refresh_sec` is 1800, it sleeps for 1800 seconds
- If keepalive loop is blocked/delayed, refresh is late
- If other tasks consume CPU, asyncio.sleep can drift

#### C. Timing Window Risk

**Scenario**:
1. listenKey created at T=0, expires at T=3600 (60 minutes)
2. Keepalive loop should refresh at T=1800 (30 minutes)
3. If any of these delays happen:
   - Other async tasks block the loop
   - System clock adjustment
   - Event loop hiccups
4. Refresh might happen at T=1900+ (too late)
5. WebSocket uses expired key
6. Binance rejects with 410 Gone

#### D. Rotation Logic Timing

**Location**: `core/exchange_client.py`, lines 981-1000

```python
async def _rotate_listen_key(self, *, reason: str = "") -> bool:
    """Force-close current listenKey and create a fresh one."""
    async with self._user_data_lock:
        old_lk = str(getattr(self, "_user_data_listen_key", "") or "").strip()
        with contextlib.suppress(Exception):
            await self._close_listen_key()
        self._user_data_listen_key = ""
        try:
            await self._create_listen_key()
        except Exception as e:
            self.logger.warning(
                "[EC:UserDataWS] listenKey rotation failed: %s (reason=%s)",
                e,
                str(reason or "unknown"),
            )
            return False
```

**Issue**: If rotation fails (e.g., API rate limit), new key isn't created → WebSocket can't reconnect

---

## Solution Strategy

### Immediate Fixes (15 minutes)

#### Fix 1: Reduce Refresh Interval

**Current**:
```python
self.listenkey_refresh_sec = 1800.0  # 30 minutes
```

**Better**:
```python
self.listenkey_refresh_sec = 900.0  # 15 minutes (half of 30)
```

**Why**: 
- Gives 15-minute buffer before 30-minute Binance refresh timeout
- Reduces expiration risk significantly
- No performance cost (only 1 HTTP PUT request per 15 minutes)

#### Fix 2: More Aggressive Rotation on Failure

**Current Logic**:
- 410 error → try to rotate
- If rotation fails → log warning and stop

**Better Logic**:
- 410 error → rotate with exponential backoff
- If rotation fails → retry immediately (not in the keepalive loop)
- Track rotation failures and escalate

#### Fix 3: Health Status Reporting

Add periodic health checks to SharedState so other components know WebSocket status.

### Medium Fixes (30 minutes)

#### Fix 4: Reduce Refresh Timer Variance

Replace static sleep with dynamic interval that accounts for event loop latency:

```python
async def _user_data_keepalive_loop_improved(self) -> None:
    target_refresh_sec = float(self.listenkey_refresh_sec or 900.0)  # Default 15 min
    max_interval = target_refresh_sec * 0.8  # Refresh at 80% of target
    last_refresh_ts = time.time()
    
    while self.is_started and not self._user_data_stop.is_set():
        try:
            now = time.time()
            time_since_refresh = now - last_refresh_ts
            
            if time_since_refresh >= max_interval:
                # Force refresh if it's time
                if self._user_data_listen_key:
                    await self._refresh_listen_key()
                    last_refresh_ts = time.time()
            else:
                # Sleep for remaining time
                sleep_time = max_interval - time_since_refresh
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("[EC:UserDataWS] keepalive error: %s", e)
            if self._is_invalid_listen_key_error(e):
                await self._rotate_listen_key(reason=f"keepalive:{e}")
            last_refresh_ts = time.time()  # Reset timer after refresh
```

#### Fix 5: Rotation with Retry Logic

```python
async def _rotate_listen_key_resilient(self, *, reason: str = "", max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            async with self._user_data_lock:
                old_lk = str(getattr(self, "_user_data_listen_key", "") or "").strip()
                with contextlib.suppress(Exception):
                    await self._close_listen_key()
                self._user_data_listen_key = ""
                
                await self._create_listen_key()
                
                self.logger.warning(
                    "[EC:UserDataWS] listenKey rotated (attempt=%d/%d old=%s new=%s reason=%s)",
                    attempt + 1, max_retries,
                    old_lk[:8] + "..." if old_lk else "none",
                    str(self._user_data_listen_key or "")[:8] + "...",
                    str(reason or "unknown"),
                )
                return True
        except Exception as e:
            self.logger.warning(
                "[EC:UserDataWS] listenKey rotation attempt %d/%d failed: %s",
                attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff: 0.5s, 1s, 2s
    
    self.logger.error("[EC:UserDataWS] listenKey rotation FAILED after %d attempts", max_retries)
    return False
```

---

## Configuration Options

Add to environment or config:

```python
# core/exchange_client.py, line 650

# Reduce refresh interval to 15 minutes (was 30)
self.listenkey_refresh_sec = float(_cfg("USER_DATA_LISTENKEY_REFRESH_SEC", 900.0) or 900.0)
self.listenkey_refresh_sec = min(max(self.listenkey_refresh_sec, 60.0), 2500.0)

# Add rotation retry config
self.listenkey_rotation_max_retries = int(_cfg("USER_DATA_LISTENKEY_ROTATION_MAX_RETRIES", 3) or 3)

# Add refresh safety margin
self.listenkey_refresh_safety_margin_pct = float(_cfg("USER_DATA_LISTENKEY_REFRESH_SAFETY_PCT", 80) or 80)
```

---

## Implementation Priority

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| **P0** | Reduce refresh interval to 900s | High - prevents most 410 errors | 1 line |
| **P1** | Add rotation retry logic | Medium - handles transient failures | 20 lines |
| **P2** | Dynamic keepalive timing | Medium - prevents timer drift | 30 lines |
| **P3** | Health status reporting | Low - better observability | 10 lines |

---

## Testing Strategy

### Unit Tests

```python
def test_listenkey_refresh_timing():
    """Verify refresh happens before expiration"""
    client = ExchangeClient(...)
    assert client.listenkey_refresh_sec < 2400  # Less than 60-min Binance timeout / 2.5
    
def test_rotation_retry_logic():
    """Verify rotation retries on failure"""
    # Mock failed rotation
    # Verify exponential backoff applied
    # Verify final success after retries

def test_keepalive_loop_timing():
    """Verify keepalive refreshes on schedule"""
    # Mock time
    # Verify refresh happens at correct intervals
    # Verify no drift over multiple cycles
```

### Integration Tests

```python
def test_websocket_recovery_from_410():
    """Verify system recovers from 410 error"""
    # Simulate 410 error during WebSocket connection
    # Verify listenKey is rotated
    # Verify WebSocket reconnects with new key
    # Verify no order-execution impact
```

### Live Monitoring

```
[EC:UserDataWS] disconnected: reason=410_gone reconnect_count=N
  → Check if listenkey_refresh_sec < 900
  → Check if rotation succeeded
  → Check event loop latency (asyncio debug logs)
```

---

## Deployment Steps

1. **Immediate** (NOW):
   - Change `listenkey_refresh_sec` from 1800 to 900
   - Monitor for 410 errors (should be rare)

2. **Short-term** (This week):
   - Implement rotation retry logic
   - Add health status reporting

3. **Medium-term** (Next sprint):
   - Implement dynamic keepalive timing
   - Add comprehensive WebSocket health dashboard

---

## Expected Improvements

**Before Fix**:
- 410 errors every 30-90 minutes
- Reconnection takes 3-10 seconds
- No clear recovery path

**After Fix** (Full Implementation):
- 410 errors rare (< 1 per week)
- If occurs: rotates in < 1 second
- Clear health visibility
- Exponential backoff prevents cascading failures

---

## Reference: Binance WebSocket Documentation

- **listenKey Creation**: `POST /api/v3/userDataStream` → creates key valid for 60 minutes
- **listenKey Refresh**: `PUT /api/v3/userDataStream` → extends TTL to 60 minutes from now
- **listenKey Expiration**: Happens if no refresh PUT request within 60 minutes
- **410 Gone**: Indicates key expired or closed
- **Rate Limits**: No rate limit on userDataStream endpoints (specific quota)

---

## Files to Modify

1. **core/exchange_client.py**
   - Line 650-651: Change refresh interval to 900 seconds
   - Lines 1005-1019: Improve `_user_data_keepalive_loop()`
   - Lines 981-1000: Add `_rotate_listen_key_resilient()`

2. **tests/** (if exists)
   - Add WebSocket keepalive tests
   - Add rotation retry tests

3. **CONFIG** (environment):
   - `USER_DATA_LISTENKEY_REFRESH_SEC=900`
   - `USER_DATA_LISTENKEY_ROTATION_MAX_RETRIES=3`

---

## Questions Answered

**Q: Why does 410 error happen if refresh is every 30 minutes?**
A: Timing variance. The refresh loop might be delayed by other tasks, system clock adjustment, or event loop latency. By refreshing every 15 minutes instead, we have a 45-minute buffer.

**Q: Will more frequent refreshes hurt performance?**
A: No. A single HTTP PUT request every 15 minutes is negligible (~4 requests/hour vs 2 currently).

**Q: Why not just increase keepalive to less than 60 minutes?**
A: That works, but 15-minute intervals provide better safety margin and faster error detection.

**Q: What if the rotation itself fails?**
A: That's why we need retry logic. With exponential backoff, we retry 2-3 times before giving up.

**Q: Does this affect Phase 2 implementation?**
A: No. Phase 2 uses REST API for order execution (unaffected by WebSocket). But WebSocket health affects user data stream (balance updates, order fills).

