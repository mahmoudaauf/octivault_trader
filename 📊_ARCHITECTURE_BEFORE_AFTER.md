# Institutional Architecture: Before & After

## BEFORE (Current - Broken)

```
┌─────────────────────────────────────────────────────────────┐
│                     Wallet State                             │
│         {BTC: 0.5, ETH: 2.0, USDT: 1000}                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    RecoveryEngine            │
        │  _load_live()                │
        │  - Fetch balances ✅         │
        │  - Fetch positions ✅        │
        │  - Load raw state (dumb) ✅  │
        └──────────────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────────┐        ┌──────────────┐
        │ Balances:    │        │ Positions:   │
        │ {BTC: 0.5,   │        │ {} ← EMPTY!  │
        │  ETH: 2.0,   │        │              │
        │  USDT: 1000} │        │ ❌ Gap here!  │
        └──────────────┘        └──────────────┘
                           │
                           ▼
        ┌──────────────────────────────┐
        │    ExchangeTruthAuditor      │
        │  _restart_recovery()         │
        │  - Reconcile balances ✅     │
        │  - Close phantoms ✅         │
        │  - Hydrate missing ❌ MISSING│
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    SharedState               │
        │  NAV = free + Σ(positions)   │
        │      = 1000 + 0              │
        │      = 1000 ❌ WRONG!        │
        │  (missing BTC/ETH positions) │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  StartupOrchestrator         │
        │  Verify: free > 0 ✅         │
        │          NAV > 0 ❌ FAIL!    │
        │  Result: STARTUP FAILS       │
        └──────────────────────────────┘
```

**The Problem:**
```
Exchange has:    BTC=0.5,  ETH=2.0,  USDT=1000
State has:       (nothing) (nothing) USDT=1000 ✅
                 ❌        ❌        

Missing Link:    Wallet → Position hydration
```

---

## AFTER (Fixed - Correct)

```
┌─────────────────────────────────────────────────────────────┐
│                     Wallet State                             │
│         {BTC: 0.5, ETH: 2.0, USDT: 1000}                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    RecoveryEngine            │
        │  _load_live()                │
        │  - Fetch balances ✅         │
        │  - Fetch positions ✅        │
        │  - Load raw state (dumb) ✅  │
        └──────────────┬───────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────────┐   ┌──────────────┐
        │ Balances:    │   │ Positions:   │
        │ {BTC: 0.5,   │   │ {} ← EMPTY   │
        │  ETH: 2.0,   │   │ (no orders)  │
        │  USDT: 1000} │   └──────────────┘
        └──────┬───────┘
               │ Pass balances dict
               ▼
        ┌──────────────────────────────┐
        │    ExchangeTruthAuditor      │ ← FIX: HYDRATION HERE
        │  _restart_recovery()         │
        │  ├─ _reconcile_balances()    │
        │  │  └─ Close phantoms ✅     │
        │  │  └─ RETURN: stats + dict  │ ← Modified signature
        │  │                           │
        │  └─ _hydrate_missing_positions() ← NEW METHOD
        │     ├─ Loop balances ✅      │
        │     ├─ Skip USDT ✅          │
        │     ├─ Skip if notional < 30 ✅
        │     ├─ Skip if exists ✅     │
        │     └─ CREATE positions ✅   │
        └──────────────┬───────────────┘
                       │
                       ▼ Creates:
        ┌──────────────────────────────┐
        │    SharedState               │
        │  Positions NOW:              │
        │  - BTCUSDT: qty=0.5          │
        │  - ETHUSDT: qty=2.0 ✅       │
        │  - Free: 1000 USDT           │
        │                              │
        │  NAV = 1000 + (0.5*65k) +    │
        │        (2.0*3.5k)            │
        │      = 40500 USDT ✅         │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  StartupOrchestrator         │
        │  Verify: free=1000 ✅        │
        │          NAV=40500 ✅        │
        │          positions=2 ✅      │
        │  Result: STARTUP PASSES ✅   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  MetaController              │
        │  Portfolio READY ✅          │
        │  Trading starts ✅           │
        └──────────────────────────────┘
```

**The Solution:**
```
Exchange has:    BTC=0.5,  ETH=2.0,  USDT=1000
                  ▼         ▼
TruthAuditor hydrates...
                  ▼         ▼
State has:       BTC ✅    ETH ✅    USDT ✅

NAV = sum(all) = 40500 USDT ✅
Startup PASSES ✅
```

---

## Component Responsibility Evolution

### RecoveryEngine

**Before (and AFTER - no change):**
```
JOB: Load raw state
├─ Fetch balances from exchange ✅
├─ Fetch positions from exchange ✅
└─ NO processing, NO hydration ✅
```

### ExchangeTruthAuditor

**Before:**
```
JOB: Validate state (incomplete)
├─ Reconcile balances vs positions ✅
├─ Close phantom positions ✅
└─ Hydrate missing ❌
```

**After:**
```
JOB: Validate & hydrate state (complete)
├─ Reconcile balances vs positions ✅
├─ Close phantom positions ✅
├─ Hydrate missing positions from wallet ✅ ← NEW
└─ Use unified dust threshold ✅ ← IMPROVED
```

### PortfolioManager

**Before:**
```
JOB: Classify positions
├─ Use stablecoin threshold ⚠️
├─ Use notional threshold ⚠️
└─ Mixed logic ❌
```

**After:**
```
JOB: Classify positions (unified)
├─ Use single MIN_ECONOMIC_TRADE_USDT ✅
├─ Notional-only check ✅
└─ Clean, auditable logic ✅
```

### SharedState

**Before (and AFTER - no change):**
```
JOB: Calculate metrics
├─ NAV = Σ(positions) + free ✅
├─ Trust upstream validation ✅
└─ No filtering ✅
```

### StartupOrchestrator

**Before (and AFTER - no change):**
```
JOB: Orchestrate startup
├─ Call RecoveryEngine ✅
├─ Call TruthAuditor ✅
├─ Verify integrity ✅
└─ Signal ready ✅
```

---

## Data Flow Diagram

### Before (Broken)

```
Exchange API
    │
    ├─→ Balances: {BTC:0.5, ETH:2.0, USDT:1000}
    │        │
    │        └─→ RecoveryEngine
    │             └─→ SharedState.update_balances()
    │                  └─→ Stored but orphaned
    │
    └─→ Positions: {}
             │
             └─→ RecoveryEngine
                  └─→ SharedState.update_position()
                       └─→ EMPTY!

SharedState Status:
  balances = {BTC:0.5, ETH:2.0, USDT:1000} ✅
  positions = {} ❌
  
NAV calculation:
  NAV = 1000 + Σ({}) = 1000 ❌

Startup: FAIL ("NAV too low")
```

### After (Fixed)

```
Exchange API
    │
    ├─→ Balances: {BTC:0.5, ETH:2.0, USDT:1000}
    │        │
    │        ├─→ RecoveryEngine
    │        │    └─→ SharedState.update_balances()
    │        │
    │        └─→ TruthAuditor._hydrate_missing_positions()
    │             ├─→ Skip USDT (it's free capital)
    │             ├─→ Check notional: BTC=32500 > 30 ✓
    │             ├─→ Check notional: ETH=7000 > 30 ✓
    │             └─→ SharedState.create_position()
    │                  ├─→ BTCUSDT: qty=0.5
    │                  └─→ ETHUSDT: qty=2.0
    │
    └─→ Positions: {}
             │
             └─→ RecoveryEngine
                  └─→ SharedState.update_position()
                       └─→ (no positions from exchange)

SharedState Status:
  balances = {BTC:0.5, ETH:2.0, USDT:1000} ✅
  positions = {BTCUSDT:..., ETHUSDT:..., ...} ✅
  
NAV calculation:
  NAV = 1000 + (0.5×65k) + (2.0×3.5k) = 40500 ✅

Startup: PASS ✅
```

---

## Dust Threshold: Before vs After

### Before (3 different definitions!)

```
Location           Name                    Type        Value
─────────────────────────────────────────────────────────────
TruthAuditor       DUST_POSITION_QTY       qty-based   0.00001
PortfolioManager   (stablecoin)            fixed       5.0 USDT
PortfolioManager   (notional)              notional    10.0 USDT
StartupOrch        MIN_ECONOMIC_TRADE_USDT notional    30.0 USDT

❌ Inconsistent: Same asset may be dust in one place, viable in another
```

### After (1 unified definition!)

```
Config Setting: MIN_ECONOMIC_TRADE_USDT = 30.0

Usage Everywhere:
  notional = qty * price
  is_dust = (notional < MIN_ECONOMIC_TRADE_USDT)

Location              Implementation
──────────────────────────────────────────────
TruthAuditor hydrate  if notional < 30 → skip
PortfolioManager      if notional < 30 → dust
StartupOrchestrator   if notional < 30 → filter

✅ Consistent: Same logic everywhere
```

---

## Control Flow: Startup Sequence

### RecoveryEngine → TruthAuditor → PortfolioManager → Orchestrator

```
Phase 8.5: Startup Orchestration
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  Step 1: RecoveryEngine.rebuild_state()                  ║
║  ├─ Input:  Exchange API                                 ║
║  ├─ Load:   Balances, Positions (raw)                    ║
║  ├─ Output: SharedState (populated)                      ║
║  └─ Time:   ~0.5s                                        ║
║                                                           ║
║  Step 2: ExchangeTruthAuditor._restart_recovery()        ║
║  ├─ Input:  Balances dict from RecoveryEngine            ║
║  ├─ Action: Reconcile, Close phantoms, Hydrate           ║
║  │          ↓                                             ║
║  │          _hydrate_missing_positions()  ← NEW          ║
║  │          ├─ Loop balances                             ║
║  │          ├─ Skip USDT                                 ║
║  │          ├─ Check notional vs 30 USDT                 ║
║  │          └─ Create missing positions                  ║
║  ├─ Output: Positions created in SharedState             ║
║  └─ Time:   ~1s                                          ║
║                                                           ║
║  Step 3: PortfolioManager.refresh_positions()            ║
║  ├─ Input:  Positions from SharedState                   ║
║  ├─ Action: Update metadata (non-fatal)                  ║
║  └─ Time:   ~0.5s                                        ║
║                                                           ║
║  Step 4: SharedState (implicit)                          ║
║  ├─ Input:  All positions now populated                  ║
║  ├─ Action: Calculate NAV from all positions             ║
║  └─ NAV:    = free + Σ(positions)                        ║
║                                                           ║
║  Step 5: StartupOrchestrator.verify_integrity()          ║
║  ├─ Input:  NAV (now non-zero!), positions               ║
║  ├─ Check:  Capital balance, position count              ║
║  ├─ Filter: Viable vs dust (using 30 USDT threshold)     ║
║  └─ Result: PASS ✅ (all checks green)                   ║
║                                                           ║
║  Step 6: Emit StartupPortfolioReady                      ║
║  ├─ Output: Signal to MetaController                     ║
║  └─ Action: Trading starts ✅                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

---

## Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **Hydration** | ❌ Doesn't happen | ✅ In TruthAuditor |
| **Dust model** | 3 definitions | 1 definition (config) |
| **NAV calculation** | Incomplete positions | Complete positions |
| **Startup result** | Fails (NAV=0) | Passes (NAV=actual) |
| **Responsibility** | Fragmented | Clear boundaries |
| **Auditable** | Unclear why NAV=0 | Clear: wallet → positions → NAV |

---

## Test Scenario: Wallet with Assets but No Orders

### Input State
```
Exchange:
  Wallet: {BTC: 0.5, ETH: 2.0, USDT: 1000}
  Orders: NONE

Config:
  MIN_ECONOMIC_TRADE_USDT = 30.0
```

### Execution Path (Before = Broken)
```
RecoveryEngine:
  → balances = {BTC: 0.5, ETH: 2.0, USDT: 1000} ✓
  → positions = {} (no orders) ✓
  → SharedState updated

TruthAuditor:
  → _reconcile_balances() finds no mismatch ✓
  → No hydration ❌

NAV = 1000 (USDT only)

StartupOrchestrator:
  → Check: NAV=1000, free=1000, positions=0 ❓
  → Issue: "Positions detected but NAV might be stale"
  → Retry: Still NAV=0 after retry ❓
  → Decision: Allow startup but risky ⚠️
```

### Execution Path (After = Fixed)
```
RecoveryEngine:
  → balances = {BTC: 0.5, ETH: 2.0, USDT: 1000} ✓
  → positions = {} (no orders) ✓
  → SharedState updated

TruthAuditor:
  → _reconcile_balances() reconciles ✓
  → _hydrate_missing_positions():
    ├─ BTC: notional = 0.5 * 65000 = 32500 > 30 ✓
    ├─ ETH: notional = 2.0 * 3500 = 7000 > 30 ✓
    └─ Creates both positions ✓

NAV = 1000 + 32500 + 7000 = 40500 ✓

StartupOrchestrator:
  → Check: NAV=40500, free=1000, positions=2 ✓
  → All checks pass
  → READY ✅
```

---

## Institutional Design Pattern

This fix implements the **canonical layered validation pattern**:

```
Layer 1: DATA LOAD (RecoveryEngine)
  └─ Purpose: Fetch & normalize
     Action: Get raw state from exchange

Layer 2: STATE VALIDATION (TruthAuditor)  ← Hydration lives here
  └─ Purpose: Reconcile & hydrate
     Action: Close phantoms, hydrate missing, use unified thresholds

Layer 3: ECONOMIC CLASSIFICATION (PortfolioManager)
  └─ Purpose: Classify positions
     Action: Mark viable vs dust, rebalance

Layer 4: METRICS CALCULATION (SharedState)
  └─ Purpose: Calculate read-only metrics
     Action: NAV, PnL, leverage, etc.

Layer 5: STARTUP VERIFICATION (StartupOrchestrator)
  └─ Purpose: Gate & signal ready
     Action: Verify integrity, emit ready event
```

This pattern ensures:
- **Single Responsibility**: Each component has one job
- **Clear Data Flow**: Output of layer N → Input of layer N+1
- **Auditable**: Easy to trace issues
- **Testable**: Each layer can be tested in isolation
