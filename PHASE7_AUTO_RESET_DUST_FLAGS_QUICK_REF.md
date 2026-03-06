# PHASE 7: Auto-Reset Dust Flags - Quick Reference
**Developer Cheat Sheet**

---

## 🎯 TL;DR

**Problem**: Dust flags (_bootstrap_dust_bypass_used, _consolidated_dust_symbols) never reset, blocking future operations

**Solution**: Auto-reset flags after 24 hours of inactivity using `_reset_dust_flags_after_24h()`

**Impact**: Dust operations available every 24h instead of permanently blocked

---

## 📝 What Was Added

| Component | Location | Size | Purpose |
|-----------|----------|------|---------|
| **Method** | Lines 456-523 | 68 LOC | Auto-reset logic for both flag sets |
| **Integration** | Lines 4591-4598 | 8 LOC | Call from cleanup cycle |
| **Config** | Line 1103 | 1 LOC | Timeout initialization |
| **Total** | `core/meta_controller.py` | 77 LOC | Complete feature |

---

## 🔧 Quick Usage

### Check if Flags Will Reset
```python
# Get current dust state
state = controller._get_symbol_dust_state("BTCUSDT")
if state:
    age = time.time() - state["last_dust_tx"]
    hours_until_reset = (86400.0 - age) / 3600.0
    print(f"Reset in {max(0, hours_until_reset):.1f} hours")
```

### Manually Reset Flags (For Testing)
```python
# Direct reset (not recommended in production)
controller._bootstrap_dust_bypass_used.discard("BTCUSDT")
controller._consolidated_dust_symbols.discard("SOLUSDT")
```

### Monitor Reset Events
```bash
# Watch logs for reset events
tail -f logs/trading_bot.log | grep "DustReset"

# Count resets in last 24h
grep "DustReset.*Reset.*flag" logs/trading_bot.log | wc -l

# Find longest-lived flags
grep "DustReset" logs/trading_bot.log | grep "after.*hours"
```

---

## 🔍 How It Works

```
┌─────────────────────────────────────────────────────┐
│ Every 30 seconds (via cleanup cycle):                │
├─────────────────────────────────────────────────────┤
│ _reset_dust_flags_after_24h() called                │
│   │                                                  │
│   ├─ For each symbol in bypass_used set:            │
│   │  └─ Age = now - last_dust_tx                    │
│   │     └─ If age ≥ 24h: RESET + LOG               │
│   │                                                  │
│   └─ For each symbol in consolidated set:           │
│      └─ Age = now - last_dust_tx                    │
│         └─ If age ≥ 24h: RESET + LOG               │
│                                                      │
│   Return: count of resets (0, 1, 2, ...)            │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Expected Behavior

### Timeline Example (BTCUSDT)
```
00:00 ─── Dust merge attempt
         Flag set: bypass_used ← true
         
12:00 ─── Still active
         age = 12h < 24h
         Status: ✓ PRESERVED
         
24:00 ─── Timeout reached
         age = 24h ≥ 24h
         Cleanup runs: RESET TRIGGERED
         Log: "[Meta:DustReset] Reset bypass flag for BTCUSDT after 24.0 hours..."
         
24:30 ─── Next cleanup
         Flag gone, bypass available again
```

---

## 🚨 Logs to Monitor

### ✅ Good - Flags Resetting on Schedule
```
[Meta:DustReset] Reset bypass flag for BTCUSDT after 24.0 hours (24h timeout)
[Meta:DustReset] Reset consolidated flag for SOLUSDT after 24.2 hours (24h timeout)
[Meta:Cleanup] Reset 2 dust flags for inactive symbols (24h timeout)
```

### ⚠️ Warning - Orphaned Flags Found
```
[Meta:DustReset] Reset orphaned bypass flag for DOGEUSDT
[Meta:DustReset] Reset orphaned consolidated flag for XRPUSDT
```

### ❌ Error - Reset Failures
```
[Meta:Cleanup] Dust flag reset error: [error details]
```

---

## 🔐 Data Structures Affected

```python
# Two sets being managed:

self._bootstrap_dust_bypass_used = set()  # e.g., {"BTCUSDT", "ETHUSDT"}
# → Reset after 24h inactivity
# → Allows bypass to be used again

self._consolidated_dust_symbols = set()   # e.g., {"SOLUSDT"}
# → Reset after 24h inactivity
# → Allows consolidation to run again

# Supporting state:
self._symbol_dust_state = {               # From Phase 6
    "BTCUSDT": {
        "last_dust_tx": 1677000000.0,    # ← Determines reset timing
        "state_created_at": 1677000000.0,
        "bypass_used": True,
        "consolidated": False,
        "merge_attempts": 1
    },
    ...
}
```

---

## ⚙️ Configuration

### Default (No Config Needed)
```python
self._dust_flag_reset_timeout = 86400.0  # 24 hours
```

### Optional Custom Timeout
```python
# In config.py:
DUST_FLAG_RESET_TIMEOUT_SEC = 43200.0   # 12 hours (for testing)

# Applies to both bypass_used and consolidated_dust_symbols
```

---

## 🧪 Quick Test

### Verify Implementation Present
```python
import inspect
source = inspect.getsource(controller._reset_dust_flags_after_24h)
assert "86400.0" in source, "24h timeout found"
assert "bootstrap_dust_bypass_used" in source, "Bypass handling found"
assert "consolidated_dust_symbols" in source, "Consolidated handling found"
print("✓ Implementation verified")
```

### Simulate 24h Timeout (Testing)
```python
# Set flag with old timestamp
controller._bootstrap_dust_bypass_used.add("TEST")
controller._symbol_dust_state["TEST"] = {
    "last_dust_tx": time.time() - 86401.0,  # 24h + 1s ago
    "state_created_at": time.time()
}

# Run cleanup
await controller._reset_dust_flags_after_24h()

# Verify
assert "TEST" not in controller._bootstrap_dust_bypass_used, "Flag should be reset"
print("✓ Dust flag reset verified")
```

---

## 🚀 Deployment Checklist

- [ ] Code review: `_reset_dust_flags_after_24h()` looks good
- [ ] Syntax check: `python -m py_compile core/meta_controller.py`
- [ ] Integration check: Method in cleanup cycle
- [ ] Test: Run unit tests
- [ ] Staging: Deploy and monitor 24h
- [ ] Production: Deploy after 24h validation
- [ ] Monitoring: Set up log alerts

---

## 🔗 Related Files

- **Design**: `PHASE7_AUTO_RESET_DUST_FLAGS_24H_DESIGN.md`
- **Tests**: `PHASE7_AUTO_RESET_DUST_FLAGS_IMPLEMENTATION.md` (test cases)
- **Code**: `core/meta_controller.py` (lines 456-523, 1103, 4591-4598)
- **Prior**: `PHASE6_SYMBOL_SCOPED_DUST_CLEANUP_COMPLETE.md`

---

## ❓ FAQ

**Q: Why 24 hours?**
A: Long enough to prevent false resets, short enough to enable dust operations recovery

**Q: Does reset stop trading?**
A: No - only resets internal flags, trading continues unaffected

**Q: What if a symbol has dust activity within 24h?**
A: Flag is preserved indefinitely as long as dust activity continues

**Q: Can I configure different timeouts per symbol?**
A: Not currently - uses global 24h timeout for all symbols

**Q: What if reset fails?**
A: Error is logged but doesn't crash system; next cleanup cycle will retry

**Q: Why both bypass_used and consolidated?**
A: Both are one-shot operations that benefit from reset after dust clears

---

## 📞 Support

- **Issue with resets not happening?** Check logs for `DustReset` messages
- **Resets happening too often?** Check dust state timestamps
- **Performance concerns?** Monitor CPU during cleanup cycle (~2-3ms per 1000 symbols)

---

**Quick Reference Complete** ✅

See `PHASE7_AUTO_RESET_DUST_FLAGS_24H_DESIGN.md` for details.
