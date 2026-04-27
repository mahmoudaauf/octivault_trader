# ⚡ QUICK REFERENCE - LIVE PRODUCTION DEPLOYMENT

## 🚀 START LIVE TRADING - Choose One:

### Option 1: Automatic (Recommended)
```bash
bash start_production.sh
```
- Runs verification first
- Shows system info
- Starts production
- Displays final state on exit

### Option 2: Direct Start
```bash
python3 PRODUCTION_STARTUP.py
```
- Minimal wrapper
- Direct production startup
- Same functionality

### Option 3: Verify Then Start
```bash
python3 verify_deployment.py
python3 PRODUCTION_STARTUP.py
```
- Manual verification
- Then start production

---

## 📊 MONITOR WHILE RUNNING

### Terminal 1: Main Process
```
(Shows live system output)
```

### Terminal 2: Watch State Files
```bash
watch -n 5 'ls -lh state/'
```

### Terminal 3: Check Context
```bash
watch -n 10 'python3 -c "from system_state_manager import SystemStateManager; mgr = SystemStateManager(); ctx = mgr.get_system_context(); status = ctx[\"system_status\"]; print(f\"Phase: {status[\"current_phase\"]}, Task: {status[\"current_task\"]}, Recovery: {status[\"recovery_enabled\"]}\")"'
```

---

## 🔄 TEST AUTO-RECOVERY

After running for ~5 minutes:

1. **Stop the system** (Ctrl+C in main terminal)
2. **Verify state saved:**
   ```bash
   cat state/checkpoint.json | python3 -m json.tool
   ```
3. **Restart the system:**
   ```bash
   python3 PRODUCTION_STARTUP.py
   ```
4. **Look for:** `✅ Restart detected and recovered!`

✅ **Success** = Auto-recovery working!

---

## 📋 KEY FILES REFERENCE

| File | Purpose | Size |
|------|---------|------|
| PRODUCTION_STARTUP.py | Main entry point | 5.9 KB |
| start_production.sh | Bash wrapper | 2.9 KB |
| verify_deployment.py | Verification | 6.3 KB |
| MASTER_DEPLOYMENT_INDEX.md | Complete guide | 11 KB |
| LIVE_DEPLOYMENT_READY.md | Detailed guide | 13 KB |
| DEPLOY_CHECKLIST.md | Checklist | 9.6 KB |

---

## 🔍 VERIFICATION CHECKLIST

**Before Starting:**
- [ ] `python3 verify_deployment.py` shows all 5 checks passed
- [ ] `ls -ld state/` shows `state/` is writable
- [ ] No error messages in verification output

**After Starting:**
- [ ] System starts without errors
- [ ] See "✅ LIVE SYSTEM OPERATIONAL" message
- [ ] State files created in state/ directory
- [ ] Monitor shows state files growing

**After 5+ Minutes:**
- [ ] Multiple state saves occurred
- [ ] Checkpoint created
- [ ] Recovery system ready

---

## ⚙️ SYSTEM ARCHITECTURE

```
PRODUCTION_STARTUP.py
  ├─ check_and_recover() ........... Restart detection
  ├─ initialize_live_environment() . State recovery init
  └─ main_trading_loop() .......... Continuous operation
        ├─ system_state_manager.py . State persistence
        ├─ auto_recovery.py ....... Auto-recovery
        ├─ state/ ................ Disk storage
        └─ core/meta_controller.py  Trading logic (5 fixes)
```

---

## 🆘 TROUBLESHOOTING

### System Won't Start
```bash
python3 verify_deployment.py
# Check for failures and follow suggestions
```

### State Files Not Creating
```bash
python3 -c "from system_state_manager import SystemStateManager; mgr = SystemStateManager(); mgr.save_operational_state(); print('✅ Created')"
```

### Recovery Not Working
```bash
python3 -c "from auto_recovery import check_and_recover; r = check_and_recover(); print('Ready' if r.get('context', {}).get('recovery_info', {}).get('safe_to_continue') else 'Not ready')"
```

---

## 📈 EXPECTED TIMELINE

- **Minute 0-1:** System starts, recovery check runs
- **Minute 1-5:** Live environment initializes
- **Minute 5-30:** State files accumulating
- **Minute 30+:** Continuous monitoring

---

## 🎯 SUCCESS INDICATORS

✅ System running without errors
✅ State files created in state/ directory
✅ Files growing over time (state saves happening)
✅ No error messages in output
✅ Recovery system ready (tested on restart)

---

## 📞 QUICK COMMANDS

```bash
# Start production
bash start_production.sh

# Verify all systems
python3 verify_deployment.py

# Watch state files
watch -n 5 'ls -lh state/'

# Check current status
python3 -c "from system_state_manager import SystemStateManager; import json; mgr = SystemStateManager(); print(json.dumps(mgr.get_system_context()['system_status'], indent=2))"

# Check recovery ready
python3 -c "from auto_recovery import check_and_recover; r = check_and_recover(); print('Ready' if r.get('context', {}).get('recovery_info', {}).get('safe_to_continue') else 'Not ready')"
```

---

## ✨ BOTTOM LINE

```
✅ Phase 4 Complete: 30-min test passed with 0 errors
✅ Phase 5 Ready: All systems deployed and verified
✅ Live deployment: 100% ready to go

START LIVE TRADING:
  bash start_production.sh
```

---

**Date:** April 26, 2026  
**Status:** ✅ READY FOR PRODUCTION  
**Verification:** ✅ 5/5 Checks Passed
