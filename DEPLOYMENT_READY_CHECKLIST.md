# ✅ STEP-BY-STEP DEPLOYMENT CHECKLIST

**Date**: March 1, 2026  
**Status**: READY FOR DEPLOYMENT  

---

## CRITICAL BUG SUMMARY

**Issue**: Portfolio accounting is 45% wrong  
**Root Cause**: Stale cached prices, no Binance sync  
**Impact**: Phantom losses, false alerts, broken limits  
**Fix**: Refresh from Binance every snapshot  
**Status**: Code deployed to `core/shared_state.py`  

---

## 5-STEP DEPLOYMENT COMMAND

```bash
# 1. Go to project
cd /path/to/octivault_trader

# 2. Pull latest
git pull origin main

# 3. Stop bot
pkill -f "python.*meta_controller"

# 4. Start bot  
python main.py

# 5. Verify
tail -f bot.log | grep -E "NavReady|nav="
# Expected: NavReady event, nav≈115.89
```

---

## VERIFICATION CHECKLIST

After deployment:
- [ ] Bot starts without errors
- [ ] NavReady event appears in logs
- [ ] NAV shows ~115.89 USDT
- [ ] No error messages
- [ ] Trading proceeds normally

---

## WHAT TO EXPECT

**Before**: NAV=213.65 (WRONG by 97 USDT)  
**After**: NAV=115.89 (CORRECT ✅)  

**Before**: Unrealized PnL=-784 (phantom loss)  
**After**: Unrealized PnL≈0 (accurate ✅)  

---

## IF SOMETHING GOES WRONG

Rollback:
```bash
pkill -f "python.*meta_controller"
git checkout HEAD~1 -- core/shared_state.py
python main.py
```

---

## DOCUMENTATION

Read these for details:
- `QUICK_REFERENCE_ACCOUNTING_FIX.md` (2 min)
- `CODE_DIFF_ACCOUNTING_FIX.md` (5 min)
- `ACCOUNTING_FIX_INDEX.md` (all docs list)

---

**Status**: ✅ READY - DEPLOY NOW 🚀
