# ⚡ QUICK START: Position Invariant Monitoring

## Activation (3 Steps)

### Step 1: Enable in .env
```properties
# Position Sync Monitoring (add to .env or core/.env)
POSITION_SYNC_CHECK_INTERVAL_SEC=60
POSITION_SYNC_TOLERANCE=0.00001
STRICT_POSITION_INVARIANTS=false
```

### Step 2: Start Background Monitor
In your app initialization (e.g., `app_context.py` or main startup):
```python
# Start the periodic position sync monitor
await execution_manager.start_position_sync_monitor()
```

### Step 3: Monitor the Logs
```bash
# Watch for CRITICAL alerts (they'll stand out)
tail -f logs/app.log | grep -E "INVARIANT|CRITICAL|DEGRADED"
```

---

## What Gets Logged

### Normal Operation (No Issues)
```
[EM:PosSyncMonitor] Started (interval=60.0s)
[EM:DelayedFill] Reconciled delayed fill symbol=BTCUSDT side=SELL qty=0.001 attempt=2/6
```

### Position Drift Detected (Minor)
```
⚠️ Position drift on BTCUSDT: exchange=1.000000 internal=0.999990 drift=0.000010
```

### Invariant Violation (CRITICAL)
```
🚨 INVARIANT VIOLATED: BTCUSDT position INCREASED during SELL 
   (before=1.0 after=1.1). This indicates double-execution or state corruption.
   → Hard stop triggered if STRICT_POSITION_INVARIANTS=true
```

---

## Alert Severity Levels

| Level | Message | Action Required |
|-------|---------|-----------------|
| INFO | `[EM:DelayedFill] Reconciled...` | None - working as designed |
| WARNING | `Position drift...` | Monitor - normal reconciliation |
| CRITICAL | `INVARIANT VIOLATED...` | **STOP TRADING** - state corruption |

---

## Ultra-Safe Mode

For production with critical capital:
```properties
# Ultra-safe: halt on ANY drift > tolerance
STRICT_POSITION_INVARIANTS=true

# Tighter tolerance (requires more exchange API calls)
POSITION_SYNC_TOLERANCE=0.000001

# More frequent checks
POSITION_SYNC_CHECK_INTERVAL_SEC=30
```

---

## Verify Installation

```python
# Test that the method exists
import inspect
from core.execution_manager import ExecutionManager

# Should return the new method
assert hasattr(ExecutionManager, '_verify_position_invariants')
assert hasattr(ExecutionManager, 'start_position_sync_monitor')

print("✅ State sync hardening installed correctly")
```

---

## Troubleshooting

### No alerts appearing?
Check that `POSITION_SYNC_CHECK_INTERVAL_SEC` is set (default 60s)

### Too many drift warnings?
Increase `POSITION_SYNC_TOLERANCE` (default 0.00001)

### Want to see detailed checks?
Add to logger config:
```python
logging.getLogger("ExecutionManager").setLevel(logging.DEBUG)
```

---

## Performance Monitoring

The periodic monitor adds ~10ms every 60s (negligible):

```python
# Monitor the monitor itself
journalctl -u octivault_trader | grep "PosSyncMonitor"
```

---

**Status:** Ready to activate ✅
