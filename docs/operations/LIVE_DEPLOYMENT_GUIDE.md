# Live Environment Deployment Guide

## Prerequisites
✅ 30-minute test passed
✅ All metrics within specification
✅ No critical errors
✅ State recovery system verified

## Deployment Steps

### 1. Enable State Recovery
```bash
# In your live trading system startup:
python3 live_integration.py
```

### 2. Monitor State Files
```bash
# Watch state persistence in real-time
watch -n 10 'du -sh state/'
```

### 3. Verify Recovery Works
```bash
# Check current state
python3 -c "
from system_state_manager import SystemStateManager
mgr = SystemStateManager()
ctx = mgr.get_system_context()
print('Phase:', ctx['system_status']['current_phase'])
print('Task:', ctx['system_status']['current_task'])
"
```

### 4. Monitor Live Logs
```bash
# Real-time log monitoring
tail -f logs/live_trading.log
```

## Testing Recovery

To verify recovery works in live environment:

1. Let system run for 5 minutes with state persistence enabled
2. Manually restart system: `pkill -f live_trading`
3. System should auto-recover within 30 seconds
4. Verify context with command above

## Rollback Plan

If issues occur:
1. Stop live trading: `pkill -f live_trading`
2. Disable state recovery in startup script
3. Restart from checkpoint: `python3 auto_recovery.py`
4. Contact support with checkpoint.json

## Success Criteria

✅ System auto-starts with state recovery
✅ State files updated every 60 seconds
✅ No data loss after restart
✅ Recovery takes < 30 seconds
✅ Zero critical errors in logs
✅ All operations tracked in session memory
