# Visual Architecture: Phases 1-3 Complete System

---

## System Flow Diagram (All 3 Phases)

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TRADING SIGNAL GENERATION                       │
│                     (CompoundingEngine, Advisor, etc)                │
│                                                                      │
│  Analyze market → Check conditions → Generate signal/directive       │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│              PHASE 1: SOFT BOOTSTRAP LOCK                            │
│           (SymbolRotationManager)                                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Check soft lock status                              │             │
│  │                                                     │             │
│  │  Is soft_lock_enabled? YES                         │             │
│  │  ├─ time.now() - lock_time < 3600s? (1 hour)     │             │
│  │  │  ├─ YES → BLOCK rotation (return status)       │             │
│  │  │  └─ NO → Continue to multiplier check          │             │
│  │  └─ If soft_lock_enabled = FALSE: Skip check      │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Check replacement multiplier                        │             │
│  │                                                     │             │
│  │  candidate_score > current_score × 1.10?          │             │
│  │  ├─ YES (115 > 100 × 1.10 = 110) → Continue      │             │
│  │  └─ NO (105 > 100 × 1.10 = 110) → BLOCK          │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Enforce universe size                               │             │
│  │                                                     │             │
│  │  active_symbols count:                             │             │
│  │  ├─ < 3 (min) → Add candidates                     │             │
│  │  ├─ 3-5 (OK) → Continue                            │             │
│  │  └─ > 5 (max) → Remove worst performers           │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  Result: ✅ Approved for Phase 2  or  ❌ Blocked (rotation denied)  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│          PHASE 2: PROFESSIONAL APPROVAL HANDLER                      │
│         (MetaController.propose_exposure_directive)                  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Parse & Validate Directive                          │             │
│  │                                                     │             │
│  │  ├─ Symbol format valid?                           │             │
│  │  ├─ Amount valid (positive)?                       │             │
│  │  ├─ Action valid (BUY/SELL)?                       │             │
│  │  └─ Required fields present?                       │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Verify Gates Status                                 │             │
│  │                                                     │             │
│  │  gates_status = {                                  │             │
│  │    'volatility': ✅,    (volatility OK)            │             │
│  │    'edge': ✅,          (expected edge OK)         │             │
│  │    'economic': ✅       (economic filter OK)       │             │
│  │  }                                                 │             │
│  │                                                     │             │
│  │  All gates ✅? Continue                             │             │
│  │  Any gate ❌? REJECT (return status)                │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Run Signal Validation                               │             │
│  │                                                     │             │
│  │  For BUY:                                          │             │
│  │  ├─ should_place_buy(symbol) → Check indicators    │             │
│  │  └─ Return ✅ or ❌                                  │             │
│  │                                                     │             │
│  │  For SELL:                                         │             │
│  │  ├─ should_execute_sell(symbol) → Check signals    │             │
│  │  └─ Return ✅ or ❌                                  │             │
│  │                                                     │             │
│  │  Signal valid? Continue                             │             │
│  │  Signal invalid? REJECT (return reason)             │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Generate Trace ID                                   │             │
│  │                                                     │             │
│  │  trace_id = "mc_" + random(8) + "_" + timestamp     │             │
│  │  Example: mc_a1b2c3d4e5f6_1708950045               │             │
│  │                                                     │             │
│  │  Return: {                                         │             │
│  │    'approved': True,                               │             │
│  │    'trace_id': trace_id,                           │             │
│  │    'gates': gates_status,                          │             │
│  │    'timestamp': timestamp,                         │             │
│  │    'signal': signal_result                         │             │
│  │  }                                                 │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                      │
│  Result: ✅ Approved with trace_id  or  ❌ Rejected (with reason)    │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ↓
┌──────────────────────────────────────────────────────────────────────┐
│         PHASE 3: FILL-AWARE EXECUTION                                │
│          (ExecutionManager)                                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Verify Trace ID Present                             │             │
│  │                                                     │             │
│  │  trace_id in request?                              │             │
│  │  ├─ YES: trace_id = "mc_XXXXX_timestamp"          │             │
│  │  │   ├─ Format valid?                              │             │
│  │  │   └─ Contains MetaController signature?         │             │
│  │  └─ NO: ❌ REJECT with error                        │             │
│  │       ("missing_meta_trace_id")                    │             │
│  │       (SECURITY GATE - prevents unauthorized orders)│             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Begin Execution Order Scope                         │             │
│  │                                                     │             │
│  │  save_checkpoint()                                 │             │
│  │  ├─ Current liquidity state                        │             │
│  │  └─ Current portfolio state                        │             │
│  │                                                     │             │
│  │  (Will use checkpoint for rollback if needed)      │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Place Order on Binance                              │             │
│  │                                                     │             │
│  │  order = place_order(symbol, quantity, side)        │             │
│  │  └─ order_id, order_status                         │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│                       ↓                                               │
│  ┌─────────────────────────────────────────────────────┐             │
│  │ Check Order Fill Status                             │             │
│  │                                                     │             │
│  │  fill_status = query_fill_status(order_id)          │             │
│  │  ├─ FILLED (100%)                                  │             │
│  │  │  ├─ Quantity received ✅                         │             │
│  │  │  └─ Proceed to liquidity release                │             │
│  │  │                                                 │             │
│  │  ├─ PARTIALLY_FILLED (50%)                         │             │
│  │  │  ├─ Partial quantity received                   │             │
│  │  │  └─ Proceed to liquidity release (partial)      │             │
│  │  │                                                 │             │
│  │  └─ NEW (0%)                                       │             │
│  │     ├─ Order not filled yet                        │             │
│  │     ├─ TRIGGER ROLLBACK                            │             │
│  │     └─ Return to previous state                    │             │
│  └─────────────────────────────────────────────────────┘             │
│                       │                                               │
│          ┌────────────┼────────────┐                                  │
│          ↓            ↓            ↓                                  │
│      FILLED      PARTIAL       NOT FILLED                            │
│          │            │            │                                  │
│  ┌───────┴────┬───────┴────┬───────┴────┐                             │
│  │            │            │            │                             │
│  ↓            ↓            ↓            ↓                             │
│ ┌──────────────────────────┐  ┌────────────────────┐                  │
│ │ Release Liquidity        │  │ Rollback Liquidity │                  │
│ │ (Fill-Aware Release)     │  │ (Fill-Aware)       │                  │
│ │                          │  │                    │                  │
│ │ release_qty(amount)      │  │ rollback_qty()     │                  │
│ │ ├─ Update available      │  │ ├─ Restore from    │                  │
│ │ │  liquid                │  │ │  checkpoint      │                  │
│ │ └─ Release only if       │  │ └─ Return state    │                  │
│ │    fill confirmed        │  │    to pre-trade    │                  │
│ └──────────────────────────┘  └────────────────────┘                  │
│          │                            │                               │
│          ↓                            ↓                               │
│ ┌──────────────────────────┐  ┌────────────────────┐                  │
│ │ Log Complete Audit Trail │  │ Log Rollback Event │                  │
│ │                          │  │                    │                  │
│ │ {                        │  │ {                  │                  │
│ │  'trace_id': 'mc_...',  │  │  'trace_id': '...',│                  │
│ │  'order_id': '12345',   │  │  'reason': 'unfill'│                  │
│ │  'fill_status': 'FILL', │  │  'rolled_back': ✅ │                  │
│ │  'liquidity': 'release',│  │  'state': 'restore'│                  │
│ │  'timestamp': 'XXX'     │  │ }                  │                  │
│ │ }                        │  │                    │                  │
│ └──────────────────────────┘  └────────────────────┘                  │
└──────────────────────┬───────────────────────────┬──────────────────────┘
                       │                           │
                       ↓                           ↓
                   SUCCESS                    ROLLED BACK
                (Order filled,          (Order not filled,
                 liquidity released)     state restored)
```

---

## Triple-Layer Protection Summary

```
┌────────────────────────────────────────────────────────────┐
│                  TRADING REQUEST ENTERS                    │
│                                                            │
│  CompoundingEngine or any signal source                   │
└────────────────────┬───────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    PHASE 1                   PHASE 2
  GATING CHECK           APPROVAL REQUIRED
    (3 Gates)               (3 Steps)
         │                        │
    ┌────┴───┐              ┌─────┴──────┐
    │         │              │            │
   ❌    ✅ OK               ❌        ✅ APPROVED
 BLOCKED  CONTINUE        REJECTED    WITH TRACE_ID
    │         │              │            │
    └─────┤   │              └────┤   ┌───┘
          │   │                   │   │
          │   └────────┬──────────┘   │
          │            │              │
          │            ↓              │
          │        ┌────────────┐     │
          │        │ PHASE 3    │◄────┘
          │        │EXECUTION   │
          │        │ (Fill-aware)
          │        └─────┬──────┘
          │              │
          │        ┌─────┴──────┐
          │        │            │
          │       ❌            ✅
          │    REJECTED      EXECUTED
          │    (No trace)   (With audit)
          │        │            │
          └────────┼────────────┘
                   │
                   ↓
            FINAL DECISION
```

---

## Configuration State

```
BOOTSTRAP_SOFT_LOCK_ENABLED = True
├─ When: First trade triggers lock
├─ Duration: 3600 seconds (1 hour)
├─ Effect: Block rotation for 1 hour after first trade
└─ Override: Set to False to disable

SYMBOL_REPLACEMENT_MULTIPLIER = 1.10
├─ When: Checking if new symbol is better
├─ Calculation: current_score × 1.10 = threshold
├─ Example: If current = 100, need candidate > 110 (10% better)
└─ Override: Set to 1.05 for 5% threshold, 1.01 for 1%, etc.

MAX_ACTIVE_SYMBOLS = 5
├─ When: Enforcing universe size
├─ Effect: Remove underperformers if > 5 active
└─ Override: Increase to 7 for more symbols, decrease to 3 for fewer

MIN_ACTIVE_SYMBOLS = 3
├─ When: Enforcing universe size
├─ Effect: Add candidates if < 3 active
└─ Override: Decrease to 2 to allow fewer symbols
```

---

## Execution Timeline Example

```
10:00:00 — Trade Signal Arrives
            │
            ├─ Phase 1: ✅ Soft lock not active, multiplier OK, universe OK
            ├─ Phase 2: ✅ Gates passed, signal valid, trace_id generated
            ├─ Phase 3: ✅ Order placed, FILLED, liquidity released
            │
            └─→ SUCCESS: Order executed, audit trail complete
                trace_id: mc_a1b2c3d4_1708950000

10:15:00 — Another Trade Signal
            │
            ├─ Phase 1: ❌ Soft lock ACTIVE (elapsed 15 min < 60 min)
            │
            └─→ BLOCKED: Wait 45 more minutes before rotation allowed

11:00:00 — After Soft Lock Expires
            │
            ├─ Phase 1: ✅ Soft lock EXPIRED, check multiplier
            │           New candidate score 110, current 100
            │           110 > 100 × 1.10? YES ✅
            ├─ Phase 2: ✅ Gates passed, signal valid, trace_id generated
            ├─ Phase 3: ✅ Order placed, FILLED, liquidity released
            │
            └─→ SUCCESS: Rotation allowed, new symbol active
                trace_id: mc_b2c3d4e5_1708950600
```

---

## Safety Guarantees After All 3 Phases

```
✅ PHASE 1: Prevents rotation overload
   - Can't rotate more than once per hour
   - Must show 10% improvement to rotate
   - Keeps 3-5 symbols active (not too many, not too few)

✅ PHASE 2: Ensures only valid trades execute
   - All trades require MetaController approval
   - Gates must pass (volatility, edge, economic)
   - Signal must validate (technical indicators)
   - Every trade gets unique audit ID (trace_id)

✅ PHASE 3: Ensures liquidity safety
   - Orders only executed with Phase 2 approval
   - Liquidity only released if order actually fills
   - Unfilled orders trigger automatic rollback
   - Complete audit trail (trace_id + fill status + timestamps)

✅ COMBINED: Triple-layer protection
   - Bad rotation attempts blocked at Phase 1
   - Unauthorized orders blocked at Phase 2
   - Unsafe fills blocked at Phase 3
   - Every trade audited with trace_id + fill verification
```

---

## Files Involved

```
Phase 1: Symbol Rotation
├─ core/symbol_rotation.py       (306 lines, NEW)
├─ core/config.py                (+56 lines)
├─ core/meta_controller.py       (+17 lines)
└─ agents/symbol_screener.py     (reused)

Phase 2: Professional Approval
├─ core/meta_controller.py       (+270 lines)
└─ core/execution_manager.py     (guard already in place)

Phase 3: Fill-Aware Execution
├─ core/shared_state.py          (+25 lines)
└─ core/execution_manager.py     (+150 lines)

Total Changes: 824 lines across 5 files
```

---

## Key Takeaway

```
┌───────────────────────────────────────────────────────────┐
│ ALL 3 PHASES WORKING TOGETHER CREATE A TRIPLE-LAYER       │
│ PROTECTION SYSTEM WITH COMPLETE AUDIT TRAIL               │
│                                                           │
│ Phase 1: Prevents bad rotations (soft lock + multiplier)  │
│ Phase 2: Prevents bad trades (gates + validation)         │
│ Phase 3: Prevents bad fills (fill-aware execution)        │
│                                                           │
│ Result: Safe, auditable, profitable trading system        │
└───────────────────────────────────────────────────────────┘
```

