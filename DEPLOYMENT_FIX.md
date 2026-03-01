# URGENT FIX: Sync Corrected Files to Ubuntu Server

## Problem

Your Ubuntu server at `ip-172-31-37-246` has an **older version** of `execution_manager.py` with syntax errors:

```
2026-02-18: SyntaxError at line 614
2026-02-19: IndentationError at line 8
2026-02-20: IndentationError at line 3585
```

Your **local macOS file** is clean and correct (verified with Python 3.9 syntax check).

## Solution: Deploy Corrected Files

### Quick Method (Recommended)

```bash
cd "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
./SYNC_TO_SERVER.sh ubuntu@ip-172-31-37-246
```

This script will:
1. ✅ Copy corrected `execution_manager.py` to server
2. ✅ Copy corrected `shared_state.py` to server
3. ✅ Copy corrected `meta_controller.py` to server
4. ✅ Verify imports on server
5. ✅ Show next steps

### Manual Method (If script doesn't work)

#### Step 1: Copy the corrected execution_manager.py

```bash
scp "/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/core/execution_manager.py" \
    ubuntu@ip-172-31-37-246:~/octivault_trader/core/execution_manager.py
```

#### Step 2: Verify on server

```bash
ssh ubuntu@ip-172-31-37-246
cd ~/octivault_trader
python3 -c "import core.execution_manager; print('✅ Import successful')"
```

Expected output:
```
✅ Import successful
```

#### Step 3: Restart Phase 9

```bash
# Kill old process if still running
pkill -f "main_phased.py --phase 9"

# Start fresh
nohup python3 -u main_phased.py --phase 9 > logs/clean_run.log 2>&1 &

# Verify startup
tail -f logs/octivault_trader.log
```

## Verification Checklist

After deployment, check the logs:

```bash
tail -n 100 logs/octivault_trader.log
```

Expected: **No SyntaxError, No IndentationError**

You should see:
- ✅ Bootstrap successful
- ✅ Core modules loaded
- ✅ Phase 9 starting
- ✅ [SIGNAL_OUTCOME] and [SIGNAL_TUNING] messages (within 5-15 minutes)

## Files Synchronized

| File | Status | Notes |
|------|--------|-------|
| core/execution_manager.py | ✅ Clean | 5,944 lines, no syntax errors |
| core/shared_state.py | ✅ Clean | _signal_outcomes tracking |
| core/meta_controller.py | ✅ Clean | _evaluate_signal_outcomes method |

## If Still Having Issues

1. **Verify your SSH connection:**
   ```bash
   ssh ubuntu@ip-172-31-37-246 "cd ~/octivault_trader && pwd && python3 --version"
   ```

2. **Check Python version on server:**
   ```bash
   ssh ubuntu@ip-172-31-37-246 "python3 --version"
   ```
   Should be: Python 3.9+

3. **Check file permissions:**
   ```bash
   ssh ubuntu@ip-172-31-37-246 "ls -la ~/octivault_trader/core/execution_manager.py"
   ```

4. **Force reimport by clearing cache:**
   ```bash
   ssh ubuntu@ip-172-31-37-246 "cd ~/octivault_trader && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo 'Cache cleared'"
   ```

## Support

- **Local file location:** `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/core/execution_manager.py`
- **Server location:** `~/octivault_trader/core/execution_manager.py`
- **Sync script:** `./SYNC_TO_SERVER.sh`

After sync, your Phase 9 should start cleanly! 🚀
