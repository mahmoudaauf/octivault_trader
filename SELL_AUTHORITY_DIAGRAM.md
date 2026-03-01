# 🔐 SELL Execution Authority - Architecture Diagram

## SELL Order Decision & Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELL Decision Sources                          │
│  (Who can decide to SELL?)                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ✅ MetaController        → "TP Hit, need to SELL"              │
│  ✅ TPSLEngine            → "Stop-loss triggered, SELL now"     │
│  ✅ StrategyManager       → "Agent signal: SELL"                │
│  ✅ ExecutionLogic        → "Complex strategy: SELL"            │
│  ✅ RegimeTrading         → "Regime change: SELL"               │
│  ✅ CompoundingEngine     → "Harvest profits: SELL"             │
│                                                                   │
│  ❌ RecoveryEngine        → "I only rebuild state"              │
│  ❌ RiskManager           → "I only advise/gate"                │
│  ❌ ExchangeAuditor       → "I only monitor"                    │
│                                                                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RiskManager Gate   │
                    │  (Advisory)         │
                    │                     │
                    │ ✅ ALLOW            │
                    │ ❌ DENY (rare)      │
                    │ 🔴 KILL-SWITCH      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼─────────────────────────────┐
                    │  ExecutionManager                       │
                    │  (SOLE EXECUTION AUTHORITY)            │
                    │                                         │
                    │  ✅ close_position(symbol, reason)    │
                    │     └─ Handles: TP, SL, rotation     │
                    │     └─ Handles: Liquidation, dust    │
                    │                                         │
                    │  ✅ execute_trade(symbol, "SELL")     │
                    │     └─ Handles: Agent signals         │
                    │     └─ Handles: Strategy logic        │
                    │                                         │
                    │  Guarantees:                            │
                    │  • Notional floor check                │
                    │  • Fee calculations                    │
                    │  • Post-fill reconciliation            │
                    │  • Journal entries                     │
                    │  • Position invariants                 │
                    └──────────┬─────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  ExchangeClient    │
                    │  (Order Placement) │
                    │                    │
                    │  SELL order → API  │
                    └────────────────────┘
```

---

## Component Responsibilities

### ✅ SELL Decision Makers

```
MetaController
├─ Monitors: prices, indicators, time
├─ Decides: when to exit (TP, SL, rotation, liquidation)
└─ Action: calls ExecutionManager.close_position() or execute_trade()

TPSLEngine
├─ Monitors: TP and SL levels
├─ Decides: when thresholds hit
└─ Action: calls ExecutionManager.close_position()

StrategyManager
├─ Monitors: agent signals
├─ Decides: when to act on signals
└─ Action: calls ExecutionManager.execute_trade()
```

### ❌ Components that CANNOT SELL

```
RecoveryEngine
├─ Role: Rebuild state after crash
├─ Methods: rebuild_state(), verify_integrity()
├─ Access: Read-only (balances, positions, prices)
└─ 🚫 NO order placement methods

RiskManager
├─ Role: Advisory risk gating
├─ Methods: can_execute(), set_halt(), get_health()
├─ Authority: Approve/deny new trades (gate)
└─ 🚫 NO order execution methods
    └─ Can kill-switch (stops trading, doesn't SELL)
    └─ Can deny execution (doesn't force SELL)
```

---

## Execution Authority Hierarchy

```
ExecutionManager (Level 0 - Sole Authority)
│
├─ Tier 1 Callers:
│  ├─ MetaController (primary orchestrator)
│  └─ TPSLEngine (automatic mechanics)
│
├─ Tier 2 Callers:
│  ├─ StrategyManager
│  ├─ ExecutionLogic
│  └─ Other agents
│
└─ Tier 3 Callers:
   ├─ RegimeTrading
   ├─ CompoundingEngine
   ├─ BaselineKernel
   └─ SignalFusion

RecoveryEngine (Not in hierarchy - no orders)
RiskManager (Advisory - no orders)
```

---

## RiskManager Gate (Advisory Only)

```
Request: MetaController wants to SELL BTCUSDT

        ┌─────────────────────────┐
        │ RiskManager.can_execute │
        │ (Advisory Gate)          │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Check:                    │
        │ ✓ SYSTEM_HALTED?         │
        │ ✓ DAILY_LOSS exceeded?   │
        │ ✓ EXPOSURE limit?        │
        │ ✓ Kill-switch?           │
        └────────────┬─────────────┘
                     │
        ┌────────────▼─────────────┐
        │ Decision:                 │
        │                           │
        │ Case 1: ALLOW ✅          │
        │ └─ Return: True           │
        │    → Proceed to execute   │
        │                           │
        │ Case 2: DENY ❌           │
        │ └─ Return: False          │
        │    → Trade rejected       │
        │    → MetaController logs  │
        │    → Advisory (not veto)  │
        │                           │
        │ Case 3: KILL-SWITCH 🔴   │
        │ └─ Return: False          │
        │    → NO NEW trading       │
        │    → Existing SELL OK     │
        │    → (stops entry, not exit)
        │                           │
        └──────────────────────────┘

Result: RiskManager CANNOT force SELL
        RiskManager CAN only advise/refuse entry
```

---

## No Direct RecoveryEngine → SELL Path

```
RecoveryEngine (Boot-time component)

┌────────────────────────────────────────┐
│ Methods:                                │
│                                         │
│ ✅ rebuild_state()                      │
│    └─ Get exchange balances            │
│    └─ Get exchange positions           │
│    └─ Restore to SharedState           │
│                                         │
│ ✅ verify_integrity()                   │
│    └─ Check consistency                │
│    └─ Emit health status               │
│                                         │
│ ✅ Other read-only ops                 │
│    └─ Fetch filters, prices            │
│    └─ Compute unrealized PnL           │
│                                         │
│ ❌ ZERO order placement code            │
│    └─ No place_market_order()         │
│    └─ No execute_trade()              │
│    └─ No close_position()             │
│                                         │
└────────────────────────────────────────┘

Lifecycle:
t=0       Boot/restart
t=1-10s   RecoveryEngine.rebuild_state()
t=10s     ✅ Recovery complete
t>10s     🚫 RecoveryEngine dormant
          ✅ ExecutionManager active

→ No SELL possible from RecoveryEngine
  because it exits before runtime loops
```

---

## Proof: "NEVER places orders"

### Explicit Documentation

**recovery_engine.py line 22-25:**
```python
Purpose
-------
Self‑heal boot path that rebuilds in‑memory state after a crash/restart and re‑establishes
phase readiness before runtime loops start. It DOES NOT place orders.
```

### Code Inspection

**recovery_engine.py (522 lines total)**
```
Search: "place_market_order" → ❌ NOT FOUND
Search: "execute_trade" → ❌ NOT FOUND  
Search: "close_position" → ❌ NOT FOUND
Search: "execute_" → ❌ NOT FOUND
Search: "order" → Only in comments, never executed
```

### Method Signature

**recovery_engine.py key methods:**
```python
async def rebuild_state(...)
    → Reads exchange, restores state
    → No execution

async def verify_integrity(...)
    → Checks consistency
    → No execution

async def initialize(...)
    → Setup only
    → No execution
```

---

## Authority Chain Validation

```
┌─────────────────────────────────────────────────┐
│ SELL EXECUTION AUTHORITY CHAIN                  │
├─────────────────────────────────────────────────┤
│                                                  │
│ Level 1: Decision                               │
│ ├─ MetaController decides                      │
│ ├─ StrategyManager signals                     │
│ └─ TPSLEngine detects                          │
│                                                  │
│ Level 2: Approval Gate (RiskManager)           │
│ ├─ Advisory only                               │
│ ├─ Cannot veto final SELL                      │
│ └─ Can only deny initial execution             │
│                                                  │
│ Level 3: Execution (ExecutionManager SOLE)     │
│ ├─ close_position() entry point                │
│ ├─ execute_trade() entry point                 │
│ ├─ place_market_order() implementation         │
│ ├─ Notional verification                       │
│ ├─ Post-fill reconciliation                    │
│ └─ Journal audit trail                         │
│                                                  │
│ Level 4: Order Placement (ExchangeClient)      │
│ ├─ Market order API call                       │
│ └─ Exchange confirmation                       │
│                                                  │
│ Recovery/Monitoring (Separate Concerns):       │
│ ├─ RecoveryEngine: Boot-time state rebuild     │
│ ├─ RiskManager: Advisory gates                 │
│ ├─ ExchangeAuditor: Read-only monitoring       │
│ └─ (None execute orders)                       │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## Summary: Who Controls SELL?

| Component | Can Decide? | Can Execute? | Authority |
|-----------|------------|--------------|-----------|
| **MetaController** | ✅ YES | ✅ Via ExecutionManager | Primary decision-maker |
| **TPSLEngine** | ✅ YES | ✅ Via ExecutionManager | Automatic mechanics |
| **StrategyManager** | ✅ YES | ✅ Via ExecutionManager | Agent signals |
| **ExecutionManager** | ✅ YES | ✅ SOLE EXECUTOR | Gateway (only place that executes) |
| **ExchangeClient** | ❌ NO | ✅ API caller | Called by ExecutionManager |
| **RecoveryEngine** | ❌ NO | ❌ NO | Boot-time only, zero order code |
| **RiskManager** | ❌ NO | ❌ NO | Advisory gate only |
| **ExchangeAuditor** | ❌ NO | ❌ NO | Read-only monitoring |

---

## Conclusion

✅ **SELL orders flow through:**
```
Decision Maker → RiskManager Gate → ExecutionManager → ExchangeClient
```

❌ **SELL orders NEVER come from:**
- RecoveryEngine (no order code, boot-time only)
- RiskManager (advisory only, no execution)

🔐 **ExecutionManager is the SINGLE source of truth for all order execution.**
