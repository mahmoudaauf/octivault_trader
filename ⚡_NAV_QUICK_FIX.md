# Quick Reference Card: NAV Zero Issue

## Your Question
**"Why not to fetch the real NAV?"**

## The Answer (1 sentence)
**You ARE fetching it, but SHADOW MODE intentionally prevents it from being applied.**

---

## The Fix (1 line)

```python
TRADING_MODE = "live"  # Change from "shadow"
```

---

## The Flow

```
Exchange API Call ✅
        ↓
Fetch Balances ✅
        ↓
Shadow Mode Check?
        ├─ YES (your case) → Discard data, NAV=0 ❌
        └─ NO (switch here) → Use data, NAV=real ✅
```

---

## Before & After

### BEFORE (Shadow Mode):
```
Config:    TRADING_MODE = "shadow"
Logs:      [SHADOW MODE - balances not updated]
NAV:       0.0
Positions: 3 (virtual)
```

### AFTER (Live Mode):
```
Config:    TRADING_MODE = "live"
Logs:      [SS:BalanceUpdate] USDT: free=1234.56
NAV:       1234.56
Positions: 3 (real)
```

---

## Where to Change It

```bash
# Find it:
grep -r "trading_mode" . --include="*.py" --include="*.yaml"

# Edit:
TRADING_MODE = "shadow"  →  TRADING_MODE = "live"

# Restart bot
```

---

## Verification

After restart, look for:
- ✅ `trading_mode = "live"` (not shadow)
- ✅ `[SS:BalanceUpdate]` with real balance
- ✅ `NAV is 1234.56` (non-zero)
- ✅ No `[SHADOW MODE]` messages

---

## Code Locations

| What | File | Line | Status |
|------|------|------|--------|
| Fetch | recovery_engine.py | 215 | ✅ Always called |
| Block | recovery_engine.py | 360 | ⏹️ In shadow mode |
| Calc | shared_state.py | 1057 | ✅ But finds 0 |

---

## Is This a Bug?
**No.** Shadow mode is intentional design for testing/simulation.

---

## Impact

- **Shadow Mode:** Safe testing, no real trades
- **Live Mode:** Real trades with real money

**Choose carefully based on your needs.**

---

## TL;DR

1. Change `TRADING_MODE` to `"live"`
2. Restart bot
3. NAV will be real
4. Done ✅

---

## If Still NAV=0 After Change

Check:
1. Config change was saved
2. Bot was restarted
3. Logs show `trading_mode = "live"`
4. Exchange API working (look for `[SS:BalanceUpdate]`)
5. Account actually has funds
