# LIVE DEPLOYMENT CHECKLIST ✅

## Pre-Deployment Verification

### 1. State Recovery System
- [ ] system_state_manager.py exists and is accessible
- [ ] auto_recovery.py exists and is accessible
- [ ] state/ directory created: `mkdir -p state`
- [ ] All 5 state files ready:
  - [ ] operational_state.json
  - [ ] session_memory.json
  - [ ] checkpoint.json
  - [ ] recovery_state.json
  - [ ] context.json

### 2. Core System Files
- [ ] core/meta_controller.py available (all 5 portfolio fixes active)
- [ ] monitoring/sandbox_monitor.py available
- [ ] All test suites passing (57 tests, 100% pass rate)

### 3. Live Integration Files
- [ ] live_integration.py exists (2.0 KB)
- [ ] PRODUCTION_STARTUP.py exists (main entry point)
- [ ] LIVE_DEPLOYMENT_GUIDE.md exists (reference)

### 4. Environment Setup
- [ ] Python 3.9+ available
- [ ] Async/await support verified
- [ ] File permissions: state/ directory writable
- [ ] logs/ directory exists (optional but recommended)

---

## Deployment Steps

### Step 1: Start Production System
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python3 PRODUCTION_STARTUP.py
```

**Expected Output:**
```
════════════════════════════════════════════════════════════════════════════════
OCTIVAULT TRADER - PRODUCTION STARTUP
Live Trading System with State Recovery & Auto-Recovery
════════════════════════════════════════════════════════════════════════════════

📋 Phase 1: Restart Recovery Check
────────────────────────────────────────────────────────────────────────────────
✅ Fresh start - no restart detected

🔄 Phase 2: Live Environment Initialization
────────────────────────────────────────────────────────────────────────────────
✅ Live environment initialized with state recovery

📊 Phase 3: State Persistence Verification
────────────────────────────────────────────────────────────────────────────────
✅ Current phase: live_trading
✅ Current task: initialization
✅ State files: state/operational_state.json

🚀 Phase 4: Starting Continuous Operation
────────────────────────────────────────────────────────────────────────────────
✅ Live trading system starting...
   Start time: 2026-04-26T15:00:00
   State recovery: ENABLED
   Auto-recovery: ENABLED
   State persistence: ENABLED (every 60 seconds)

════════════════════════════════════════════════════════════════════════════════
✅ LIVE SYSTEM OPERATIONAL
════════════════════════════════════════════════════════════════════════════════
```

### Step 2: Verify State Files Created
```bash
# Check state directory
ls -lh state/

# Expected files:
# -rw-r--r--  operational_state.json
# -rw-r--r--  session_memory.json
# -rw-r--r--  checkpoint.json
# (others as created)
```

### Step 3: Monitor State Accumulation
```bash
# Watch state file growth in real-time
watch -n 5 'ls -lh state/ && echo "---" && wc -l state/*.json 2>/dev/null'
```

### Step 4: Verify Recovery System Ready
```bash
# Check if recovery system is operational
python3 -c "from auto_recovery import check_and_recover; result = check_and_recover(); print(f'Recovery ready: {result[\"recovery_ready\"]}')"
```

---

## Post-Deployment Verification

### ✅ Success Criteria

1. **State Files Created**
   - [ ] Files exist in state/ directory
   - [ ] operational_state.json shows current phase/task
   - [ ] Files are growing over time (periodic saves happening)

2. **Recovery System Active**
   - [ ] check_and_recover() returns recovery_ready=True
   - [ ] No errors in state management
   - [ ] Auto-recovery ready for restart

3. **Continuous Operation**
   - [ ] System running without errors
   - [ ] State saves happening every 60 seconds
   - [ ] Checkpoints saved every 5 minutes

4. **Persistent Memory**
   - [ ] State persists across cycles
   - [ ] Context information accumulating in session_memory.json
   - [ ] System context retrievable via get_system_context()

---

## Monitoring Commands

### Real-Time State Monitoring
```bash
# Terminal 1: Watch state files
watch -n 5 'ls -lh state/'

# Terminal 2: Check current system context
watch -n 10 'python3 -c "from system_state_manager import SystemStateManager; mgr = SystemStateManager(); ctx = mgr.get_system_context(); print(f\"Phase: {ctx[\"system_status\"][\"current_phase\"]}, Task: {ctx[\"system_status\"][\"current_task\"]}, Progress: {ctx[\"system_status\"][\"progress\"]}\")"'

# Terminal 3: Tail error logs (if available)
tail -f logs/live_trading.log 2>/dev/null || echo "Logs not yet created"
```

### Periodic Verification
```bash
# Check state file line counts (accumulating data)
wc -l state/*.json

# Verify latest checkpoint
cat state/checkpoint.json | python3 -m json.tool

# Check recovery readiness
python3 -c "
from auto_recovery import AutoRecoveryAgent
agent = AutoRecoveryAgent()
agent.auto_recover()
print('✅ Recovery system tested successfully')
"
```

---

## Restart Verification

### Test Auto-Recovery (After at least 5 minutes of operation)

1. **Stop the system:**
   ```bash
   # Press Ctrl+C in the terminal running PRODUCTION_STARTUP.py
   ```

2. **Verify state was saved:**
   ```bash
   cat state/checkpoint.json
   # Should show recent timestamp and operation context
   ```

3. **Restart the system:**
   ```bash
   python3 PRODUCTION_STARTUP.py
   ```

4. **Verify recovery:**
   ```
   Expected output should show:
   ✅ Restart detected and recovered!
      Phase: [previous phase]
      Task: [previous task]
      Progress: [previous progress]
   ```

This confirms auto-recovery is working correctly.

---

## Rollback Plan (If Needed)

### To Revert to Previous System
```bash
# Stop current system (Ctrl+C)

# Backup current state (optional)
cp -r state state.backup.$(date +%s)

# Remove state recovery files
rm -f live_integration.py PRODUCTION_STARTUP.py

# Restart with original system
python3 [your_original_startup_script]
```

---

## System Architecture Deployed

```
PRODUCTION_STARTUP.py
    ↓
    ├→ check_and_recover() ..................... Auto-recovery detection
    ├→ initialize_live_environment() ........... State recovery initialization
    └→ main_trading_loop() .................... Continuous operation
            ↓
            ├→ system_state_manager.py ........ Persistent state management
            ├→ auto_recovery.py ............... Automatic restart recovery
            ├→ state/ ......................... Persistent storage (JSON files)
            └→ core/meta_controller.py ....... Trading logic (all 5 fixes)
```

---

## Files Deployed

| File | Size | Purpose |
|------|------|---------|
| PRODUCTION_STARTUP.py | 5.2 KB | Main production entry point |
| live_integration.py | 2.0 KB | Live environment initializer |
| system_state_manager.py | 16.6 KB | State persistence engine |
| auto_recovery.py | 7.8 KB | Auto-recovery system |
| state/ | — | Persistent storage directory |

---

## Success Indicators

✅ **Phase 1: Initialization**
- State recovery system loaded
- Live environment initialized
- State files created

✅ **Phase 2: Continuous Operation**
- System running without errors
- State saves happening every 60 seconds
- Checkpoints created every 5 minutes

✅ **Phase 3: Persistence**
- State files growing over time
- Context accumulating in session_memory.json
- Recovery system ready for restart

✅ **Phase 4: Auto-Recovery**
- System survives restart intact
- Previous phase/task recovered
- Continuous operation resumes

---

## Support & Debugging

### If State Files Not Creating
```bash
# Check permissions
ls -ld state/
chmod 755 state/

# Manually create state files
python3 -c "
from system_state_manager import SystemStateManager
mgr = SystemStateManager()
mgr.save_operational_state()
print('✅ State files created manually')
"
```

### If Recovery Not Working
```bash
# Check auto_recovery module
python3 -c "
from auto_recovery import check_and_recover
result = check_and_recover()
print(f'Recovery status: {result}')
"
```

### If System Crashes
```bash
# Check for recovery state
cat state/recovery_state.json

# View system context
cat state/context.json | python3 -m json.tool
```

---

## Next Steps

1. ✅ **Deploy**: Run PRODUCTION_STARTUP.py
2. ✅ **Verify**: Check state files created (Step 2 above)
3. ✅ **Monitor**: Watch state accumulation (Step 3 above)
4. ✅ **Test**: Verify recovery system works (Restart Verification)
5. ✅ **Confirm**: System ready for Phase 5 staged rollout

---

**Deployment Status: READY TO DEPLOY** ✅
