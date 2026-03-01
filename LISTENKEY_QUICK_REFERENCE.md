# Quick Reference - WebSocket ListenKey Rotation Status

**Date**: February 27, 2026  
**Status**: ✅ CODE REVIEW COMPLETE - NO CHANGES NEEDED

---

## 🎯 Your Concerns - Status

| Concern | Status | Evidence | Action |
|---------|--------|----------|--------|
| Runaway loop | ✅ Fixed | FATAL @ line 1133 | None needed |
| Create NEW listenKey | ✅ Done | POST @ line 1007 | None needed |
| Not keep old one | ✅ Done | DELETE @ line 1003 | None needed |
| Close old WS | ✅ Done | Context manager @ line 1162 | None needed |
| Open new WS | ✅ Done | Lines 1151-1162 | None needed |
| Reset counter | ✅ Done | Line 1164 | None needed |
| Backoff failures | ✅ Done | Exponential @ line 1033 | None needed |

---

## 📍 Key Code Locations

### 1. ListenKey Rotation (Close Old → Create New)
- **File**: `core/exchange_client.py`
- **Method**: `_rotate_listen_key()`
- **Lines**: 988-1040
- **Steps**:
  1. Line 1003: `await self._close_listen_key()` → DELETE old
  2. Line 1005: `self._user_data_listen_key = ""` → clear memory
  3. Line 1007: `await self._create_listen_key()` → POST new

### 2. WebSocket Loop
- **File**: `core/exchange_client.py`
- **Method**: `_user_data_ws_loop()`
- **Lines**: 1124-1252
- **Key Events**:
  - Line 1133: Check `reconnect_count > 50` → FATAL
  - Line 1206: Catch disconnection exception
  - Line 1214: Call rotation on 410 error
  - Line 1164: Reset counter on success

### 3. Runaway Loop Prevention
- **File**: `core/exchange_client.py`
- **Lines**: 1132-1147
- **Logic**:
  ```python
  if current_reconnect_count > max_reconnect_attempts:  # 50 by default
      log.critical("[EC:UserDataWS] FATAL...")
      await self._report_status("FATAL", ...)
      self._user_data_stop.set()  # Stop loop
      break
  ```

---

## ✅ What's Correct

```
On 410 Error:
  ✓ OLD listenKey deleted from Binance (DELETE request)
  ✓ OLD listenKey cleared from memory
  ✓ NEW listenKey created at Binance (POST request)
  ✓ OLD WebSocket closed (context manager)
  ✓ NEW WebSocket opens with NEW listenKey
  ✓ reconnect_count reset to 0 on success
  ✓ Backoff exponential on rotation failures (0.5s → 1s → 2s)

Runaway Loop Protection:
  ✓ Max 50 consecutive reconnect attempts
  ✓ On 51st attempt: escalate to FATAL
  ✓ Set stop flag, log CRITICAL, report FATAL status
  ✓ Loop exits cleanly
  ✓ Requires manual intervention to recover
```

---

## 🧪 Quick Tests

### Verify Close Old Key
```bash
grep -n "_close_listen_key()" core/exchange_client.py
# Output: line 1003 (inside _rotate_listen_key)
```

### Verify Create New Key
```bash
grep -n "_create_listen_key()" core/exchange_client.py
# Output: line 1007 (inside _rotate_listen_key)
```

### Verify Counter Reset
```bash
grep -n "ws_reconnect_count = 0" core/exchange_client.py
# Output: line 661 (init) and line 1164 (on success)
```

### Verify FATAL Escalation
```bash
grep -n "FATAL" core/exchange_client.py
# Output: lines 1134, 1140, 1145
```

---

## 📈 Configuration (if needed to adjust)

```python
# Default values (good for production):
USER_DATA_WS_RECONNECT_BACKOFF_SEC = 3.0   # Initial wait
USER_DATA_WS_MAX_BACKOFF_SEC = 30.0        # Max wait
USER_DATA_WS_MAX_RECONNECTS = 50           # Before FATAL
USER_DATA_WS_TIMEOUT_SEC = 65.0            # Prevent hangs
USER_DATA_LISTENKEY_REFRESH_SEC = 900      # 15 min (good)

# To be more patient:
USER_DATA_WS_MAX_RECONNECTS = 100          # Allow 100 attempts

# To be more aggressive:
USER_DATA_WS_MAX_RECONNECTS = 20           # Fail faster
```

---

## 📊 Flow Diagram

```
WebSocket 410 Error
    ↓
Exception caught (line 1206)
    ↓
reconnect_count += 1
    ↓
Detected as invalid_listen_key
    ↓
_rotate_listen_key():
  1. DELETE old listenKey
  2. Clear old reference
  3. POST new listenKey
  4. Return True
    ↓
Sleep 0.5s
    ↓
Check: count > 50? (line 1133)
  → No: continue
  → Yes: FATAL escalation
    ↓
NEW WebSocket with NEW listenKey
    ↓
Success: reconnect_count = 0 (line 1164)
    ↓
✅ Connected
```

---

## 🎯 Bottom Line

**Status**: ✅ **NO CHANGES NEEDED**

All protections you specified are already in place:
- ✅ NEW listenKey on 410
- ✅ Close old key
- ✅ Open new key
- ✅ Reset counter
- ✅ Backoff on failure
- ✅ Runaway loop prevention (FATAL @ 50)

**Code is production-ready.**

---

**File**: `LISTENKEY_ROTATION_VERIFICATION.md` for full details
