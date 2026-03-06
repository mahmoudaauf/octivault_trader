# 🎯 TruthAuditor Mode-Based Design — Quick Reference

**Implementation Complete:** March 3, 2026

---

## 🔧 The Three Modes

```python
class TruthAuditorMode(Enum):
    DISABLED = "disabled"           # ❌ No operations (shadow)
    STARTUP_ONLY = "startup_only"   # ⚡ One-time recovery (future)
    CONTINUOUS = "continuous"       # 🟢 Passive governance (live)
```

---

## 📊 Mode Decision Matrix

| Scenario | trading_mode | Inferred Mode | Behavior |
|----------|--------------|---------------|----------|
| Backtest | `shadow` | `DISABLED` | No TruthAuditor operations |
| Live | `live` | `CONTINUOUS` | Full reconciliation loop |
| Paper | `paper` | `CONTINUOUS` | Full reconciliation loop |
| Custom | `<custom>` | `CONTINUOUS` | Default to full loop |

---

## 🚀 Implementation

### 1. Enum (exchange_truth_auditor.py)
```python
from enum import Enum

class TruthAuditorMode(Enum):
    DISABLED = "disabled"
    STARTUP_ONLY = "startup_only"
    CONTINUOUS = "continuous"
```

### 2. Constructor (exchange_truth_auditor.py)
```python
def __init__(self, ..., mode: Optional[str] = None, ...):
    # Auto-detect if not provided
    if mode:
        self.mode = TruthAuditorMode(str(mode).lower())
    else:
        trading_mode = str(getattr(self.config, "trading_mode", "live") or "live").lower()
        self.mode = TruthAuditorMode.DISABLED if trading_mode == "shadow" else TruthAuditorMode.CONTINUOUS
    
    self.logger.info(f"[TruthAuditor] Mode: {self.mode.value}")
```

### 3. Start Method (exchange_truth_auditor.py)
```python
async def start(self) -> None:
    # DISABLED: Skip entirely
    if self.mode == TruthAuditorMode.DISABLED:
        self.logger.info("[TruthAuditor] Mode=DISABLED; skipping")
        return
    
    # Startup reconciliation
    await self._restart_recovery()
    
    # STARTUP_ONLY: Stop after startup
    if self.mode == TruthAuditorMode.STARTUP_ONLY:
        self.logger.info("[TruthAuditor] Mode=STARTUP_ONLY; done")
        self._running = False
        return
    
    # CONTINUOUS: Full loop
    self._task = asyncio.create_task(self._run_loop())
    self.logger.info("[TruthAuditor] Mode=CONTINUOUS; started")
```

### 4. Bootstrap (app_context.py)
```python
# Infer mode from trading_mode
if trading_mode == "shadow":
    auditor_mode = "disabled"
else:
    auditor_mode = "continuous"

# Pass mode to constructor
self.exchange_truth_auditor = ExchangeTruthAuditor(
    config=self.config,
    logger=self.logger,
    app=self,
    shared_state=self.shared_state,
    exchange_client=self.exchange_client,
    mode=auditor_mode,  # 🔧 NEW
)
```

---

## 💡 Key Behaviors

### DISABLED (Shadow)
```
✅ No reconciliation loops started
✅ No exchange API calls
✅ No state mutations
✅ Pure simulation boundary maintained
```

### CONTINUOUS (Live)
```
✅ Startup Phase: Reconcile pre-startup fills
✅ Ongoing Phase: Passive monitoring only
✅ Never intercepts fresh lifecycle events
✅ Recovers truly missed fills
```

### STARTUP_ONLY (Future)
```
✅ Reconcile at startup
✅ Stop after initial recovery
✅ No ongoing monitoring
✅ Useful for resource-constrained systems
```

---

## 🔍 Research Integrity

**Shadow (DISABLED):**
- No exchange state queried
- No real-time reconciliation
- Completely isolated simulation
- ✅ Backtests remain deterministic

**Live (CONTINUOUS):**
- Startup recovery only (idempotent)
- Passive monitoring (non-invasive)
- Fresh fills handled by ExecutionManager
- ✅ Governance without interference

---

## 📝 Log Examples

### Bootstrap
```
[TruthAuditor] Mode: disabled        # Shadow
[TruthAuditor] Mode: continuous      # Live
```

### Startup
```
[TruthAuditor] Mode=DISABLED; skipping start (shadow/simulation)
[TruthAuditor] Mode=CONTINUOUS; started (interval=5.00s)
```

### Reconciliation
```
[TruthAuditor] Pre-startup SELL (order_id=12345) missing canonical – recovering
[TruthAuditor] Order filled after startup – skipped (ExecutionManager handles)
```

---

## ✅ Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Intent** | Implicit guards | Explicit mode |
| **Shadow Safety** | Partial (`None` client) | Complete (DISABLED) |
| **Maintainability** | Multiple checks | Single mode enum |
| **Future Flexibility** | N/A | STARTUP_ONLY available |
| **Testing** | Mode inference implicit | Mode explicit in constructor |

---

## 🎯 Final Architecture

```
AppContext.trading_mode
     ↓
("shadow" → mode="disabled")
("live"   → mode="continuous")
     ↓
ExchangeTruthAuditor(mode=...)
     ↓
start() respects mode
     ↓
Shadow: No operations ✅
Live:   Governance ✅
```

---

**Status:** ✅ READY FOR PRODUCTION

Simple, explicit, maintainable design that ensures:
- Research integrity for backtests
- Governance safety for live trading
- Clear operational intent
