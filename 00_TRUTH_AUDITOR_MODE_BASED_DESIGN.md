# 🎯 TruthAuditor Mode-Based Design (Advanced)

**Date:** March 3, 2026  
**Status:** ✅ COMPLETE  
**Architecture:** Mode-Based Operational Design

---

## 📋 Overview

TruthAuditor now uses a **mode-based architecture** for explicit operational intent and research integrity.

```python
class TruthAuditorMode(Enum):
    DISABLED = "disabled"           # No reconciliation (shadow mode)
    STARTUP_ONLY = "startup_only"   # Reconcile at startup, then stop
    CONTINUOUS = "continuous"       # Startup + passive ongoing monitoring
```

| Mode | Purpose | When Used |
|------|---------|-----------|
| **DISABLED** | No reconciliation | Shadow/backtesting (pure simulation) |
| **STARTUP_ONLY** | One-time recovery | Startup phase only (optional) |
| **CONTINUOUS** | Governance | Live trading (default) |

---

## 🏗️ Architecture

### Mode Selection Logic

```
trading_mode="shadow"  →  TruthAuditor.mode=DISABLED
trading_mode="live"    →  TruthAuditor.mode=CONTINUOUS
trading_mode="paper"   →  TruthAuditor.mode=CONTINUOUS (configurable)
```

### Initialization (AppContext)

**File:** `core/app_context.py` (lines 3415-3450)

```python
# MODE-BASED DESIGN: Determine TruthAuditor mode based on trading_mode
if trading_mode == "shadow":
    auditor_mode = "disabled"
else:
    auditor_mode = "continuous"  # Default for live/paper

self.exchange_truth_auditor = ExchangeTruthAuditor(
    config=self.config,
    logger=self.logger,
    app=self,
    shared_state=self.shared_state,
    exchange_client=self.exchange_client,
    mode=auditor_mode,  # 🔧 NEW: Pass mode explicitly
)
```

### Mode Handling (ExchangeTruthAuditor)

**File:** `core/exchange_truth_auditor.py` (lines 10-75)

```python
class TruthAuditorMode(Enum):
    DISABLED = "disabled"
    STARTUP_ONLY = "startup_only"
    CONTINUOUS = "continuous"

class ExchangeTruthAuditor:
    def __init__(self, ..., mode: Optional[str] = None, ...):
        # Auto-detect if not provided
        if mode:
            self.mode = TruthAuditorMode(str(mode).lower())
        else:
            trading_mode = str(getattr(self.config, "trading_mode", "live") or "live").lower()
            self.mode = TruthAuditorMode.DISABLED if trading_mode == "shadow" else TruthAuditorMode.CONTINUOUS
        
        self.logger.info(f"[TruthAuditor] Mode: {self.mode.value}")
```

### Startup Behavior

**File:** `core/exchange_truth_auditor.py` (lines 163-207)

```python
async def start(self) -> None:
    # MODE-BASED STARTUP
    if self.mode == TruthAuditorMode.DISABLED:
        self.logger.info("[TruthAuditor] Mode=DISABLED; skipping start")
        await self._set_status("Skipped", "disabled_mode")
        return
    
    if not self.exchange_client:
        self.logger.warning("[TruthAuditor] No exchange_client; cannot operate")
        return
    
    # Startup reconciliation
    await self._restart_recovery()
    
    if self.mode == TruthAuditorMode.STARTUP_ONLY:
        # One-time reconciliation, then stop
        self.logger.info("[TruthAuditor] Mode=STARTUP_ONLY; done")
        self._running = False
        return
    
    # CONTINUOUS: Full loop
    self._task = asyncio.create_task(self._run_loop())
    self.logger.info("[TruthAuditor] Mode=CONTINUOUS; started (interval=%.2fs)", self.interval_sec)
```

---

## 🔍 Research Integrity Guarantees

### Shadow Mode (DISABLED)

```
Shadow backtest:
  trading_mode="shadow"
       ↓
  TruthAuditor.mode=DISABLED
       ↓
  • No exchange calls
  • No reconciliation
  • No state interference
  • 100% deterministic
  ✅ Research validity preserved
```

**Why this matters:**
- Shadow is a **closed simulation**
- No real exchange state
- If TruthAuditor runs, it may query live exchange (corrupting backtest)
- Must be completely isolated

### Live Mode (CONTINUOUS)

```
Live trading:
  trading_mode="live"
       ↓
  TruthAuditor.mode=CONTINUOUS
       ↓
  Startup Phase:
    • Reconcile pre-startup fills
    • Repair phantom positions
    • Align open orders
  
  Ongoing Phase (passive):
    • Monitor for discrepancies
    • Detect websocket failures
    • Recover truly missed fills
    • Never intercept fresh fills
  
  ✅ Safety + Governance
```

---

## 📊 Execution Flows

### Shadow Backtest
```
┌──────────────────────────────┐
│ Bootstrap: trading_mode=shadow
├──────────────────────────────┤
│ TruthAuditor.mode=DISABLED   │
│ → start() → log "Skipping"   │
│ → no loops created           │
│ → no exchange calls          │
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│ Pure Simulation Loop         │
├──────────────────────────────┤
│ • ShadowExchangeSimulator    │
│ • ExecutionManager handles   │
│ • SharedState tracks trades  │
│ • No interference            │
└──────────────────────────────┘
         ↓
      ✅ Valid backtest metrics
```

### Live Trading (Happy Path)
```
1. Order Placed → Binance
       ↓
2. Websocket: FILL event
       ↓
3. ExecutionManager._handle_post_fill()
       ↓
4. record_trade() → SharedState
       ↓
5. TP/SL applied
       ↓
6. RealizedPnlUpdated emitted ✅
       ↓
7. TruthAuditor observes (CONTINUOUS mode, passive)
   → Logs: "Detected filled order (already applied)"
   → Skips processing (fresh event)
```

### Live Trading (Crash Recovery)
```
1. Process crashes mid-fill
       ↓
2. Restart with TruthAuditor.mode=CONTINUOUS
       ↓
3. Startup Phase: await _restart_recovery()
       ↓
4. Scan exchange for pre-startup fills
       ↓
5. Recover missing SELL fills:
   → Call record_trade()
   → Finalize positions
   → Emit RealizedPnlUpdated (recovery flow)
       ↓
6. Transition to Ongoing Phase
       ↓
7. Passive monitoring only
   ✅ Recovery + Safety
```

---

## 🔐 Key Principles

### 1. Determinism (Shadow)
```
Shadow must produce identical results every run:
  ✅ No live exchange state queried
  ✅ No real-time reconciliation
  ✅ TruthAuditor disabled completely
```

### 2. Isolation (Shadow)
```
Backtest boundaries must be clean:
  ✅ Simulation ≠ Real exchange
  ✅ No cross-contamination
  ✅ Mode enforces separation
```

### 3. Governance (Live)
```
Governance layer must be defensive:
  ✅ Startup recovery works
  ✅ Passive monitoring (non-invasive)
  ✅ Never override fresh events
  ✅ Only reconcile stale discrepancies
```

### 4. Clarity (Both)
```
Intent must be explicit:
  ✅ Mode name conveys purpose
  ✅ Logs clearly state mode
  ✅ No ambiguous guards
  ✅ Future maintainers understand
```

---

## 📝 Log Signatures

### Shadow Bootstrap
```
[TruthAuditor] Mode: disabled
[TruthAuditor] Mode=DISABLED; skipping start (shadow/simulation)
```

### Live Bootstrap
```
[TruthAuditor] Mode: continuous
[TruthAuditor] Mode=CONTINUOUS; started (interval=5.00s)
```

### Startup Recovery
```
[TruthAuditor] Mode=CONTINUOUS; reconciling now, then continuing...
[TruthAuditor] Pre-startup SELL (order_id=12345) missing canonical event – recovering silently
[TruthAuditor] Recovered fill: BTC/USDT SELL qty=0.5 price=42500
```

### Ongoing Monitoring
```
[TruthAuditor] Order filled after startup detected – skipped (ExecutionManager handles)
[Reconciliation] Detected filled order (already in event_log, skipping)
```

---

## 🚀 Configuration

### Default Behavior
```python
# Inferred from trading_mode
trading_mode="shadow"    → mode="disabled"
trading_mode="live"      → mode="continuous"
trading_mode="paper"     → mode="continuous"
```

### Explicit Override (if needed)
```python
# In config or via constructor
truth_auditor_mode = "startup_only"  # Reconcile once only
```

---

## ✅ Verification Checklist

**Shadow Mode:**
- [x] Logs show "Mode: disabled"
- [x] No "Mode=CONTINUOUS" messages
- [x] No exchange API calls from TruthAuditor
- [x] Backtest metrics deterministic across runs

**Live Mode:**
- [x] Logs show "Mode: continuous"
- [x] Startup reconciliation runs
- [x] Fresh fills skip (ExecutionManager primary)
- [x] Pre-startup fills recovered

---

## 🎓 Design Principles

| Principle | Implementation |
|-----------|-----------------|
| **Explicit Intent** | Mode enum conveys purpose |
| **Determinism** | Shadow=DISABLED, no exchange queries |
| **Isolation** | Clean separation by mode |
| **Governance** | Live=CONTINUOUS, passive monitoring |
| **Clarity** | Mode name + logs explain behavior |
| **Flexibility** | Easy to add STARTUP_ONLY for future use |

---

## 🔄 Migration Path (If Needed)

```
Current:  TruthAuditor always runs (with guards)
Future:   TruthAuditor respects mode
Benefit:  Explicit intent, cleaner code, research integrity
```

---

## 📚 Related Concepts

- **TruthAuditorMode.DISABLED** = No component behavior
- **TruthAuditorMode.CONTINUOUS** = Passive governance
- **Fresh Event Skip** = ExecutionManager primary
- **Pre-startup Recovery** = Crash recovery safety net

---

**Status:** ✅ PRODUCTION READY

Mode-based design provides:
- ✅ Clear operational intent
- ✅ Research integrity for shadow mode
- ✅ Governance safety for live mode
- ✅ Future flexibility (STARTUP_ONLY mode available)
- ✅ Maintainable, explicit code
