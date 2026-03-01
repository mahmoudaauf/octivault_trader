# 🔍 SELL Execution Authority Analysis

## Question
**Can a SELL be directly executed through RecoveryEngine or RiskManager?**

## Answer: ❌ NO

**SELL orders can ONLY be triggered through:**
1. ✅ **MetaController** (primary orchestrator)
2. ✅ **ExecutionManager** (execution authority)
3. ✅ **TPSLEngine** (take-profit/stop-loss)
4. ✅ **Strategy Manager** (agent signals)

**Never directly from:**
- ❌ RecoveryEngine (read-only, no order placement)
- ❌ RiskManager (advisory-only, no veto power)

---

## Source Analysis

### RecoveryEngine: Self-Healing, NOT Order-Placing

**File:** `core/recovery_engine.py` (lines 1-100)

```python
"""
RecoveryEngine (P9‑compliant)

Purpose
-------
Self‑heal boot path that rebuilds in‑memory state after a crash/restart and re‑establishes
phase readiness before runtime loops start. It DOES NOT place orders.
```

**Key Statement:**
```
- Never places orders (ExecutionManager remains the single order path)
```

**What RecoveryEngine Does:**
- ✅ Rebuilds in-memory state after crash
- ✅ Verifies integrity
- ✅ Restores balances and positions
- ✅ Recomputes unrealized PnL/NAV
- ✅ Sets phase gates (AcceptedSymbolsReady, MarketDataReady)
- ❌ **NEVER places orders**

**Why:** RecoveryEngine is a **boot-time component**, not a runtime trading agent.

---

### RiskManager: Advisory, Not Veto

**File:** `core/risk_manager.py` (line 35+)

```python
"""
P9 Canon: RiskManager بوابة استشارية—لا تُنفّذ أوامر. يمكنها الاستعلام
(Advisory gate—does not execute orders. Can query only)
```

**Architecture:**
```
RiskManager:
  ├─ Answers: "Is this intent allowed to exist at all?"
  ├─ Approves/denies trading via gates (DAILY_TRADING_HALT, etc.)
  ├─ Advisory for SELL (can warn, not veto)
  └─ ❌ NEVER executes orders
```

**Key Finding (ExecutionManager line 4754):**
```python
# [FIX #4] UNIFIED SELL AUTHORITY: RiskManager is advisory, not veto, for SELL
```

**RiskManager Functions:**
- ✅ Sets daily loss caps
- ✅ Monitors exposure limits
- ✅ Can trigger kill-switch
- ✅ Reports health status
- ❌ **CANNOT directly execute trades**
- ❌ **CANNOT force SELL orders**

---

## SELL Execution Authority Chain

### Primary Path: MetaController → ExecutionManager

```
MetaController (Orchestrator)
  │
  ├─ Calls: execution_manager.close_position()
  │   └─ Reason: "TP_HIT", "SL_HIT", "LIQUIDATION", etc.
  │
  └─ Calls: execution_manager.execute_trade()
      └─ Side: "SELL"
```

**MetaController SELL Triggers:**
- ✅ Rotation escapes (line 5533)
- ✅ Dust exits (line 10095)
- ✅ TP/SL exits (lines 11426, 12162)
- ✅ Risk liquidations (lines 10825, 10906, 11898, 11969, 12169)

### Secondary Path: TPSLEngine → ExecutionManager

```
TPSLEngine (Take-Profit/Stop-Loss Engine)
  │
  └─ Calls: execution_manager.close_position()
      (line 1331-1337)
```

**Code Reference (tp_sl_engine.py lines 1331-1337):**
```python
close_fn = getattr(em, "close_position", None)
if not close_fn:
    raise RuntimeError(
        "TPSLEngine requires ExecutionManager.close_position(); "
        "direct exchange execution is forbidden"
    )
```

**Key Insight:** TPSLEngine **ENFORCES** that SELL must go through ExecutionManager.close_position(), never directly to exchange.

### Tertiary Path: StrategyManager → ExecutionManager

```
StrategyManager (Agent Signals)
  │
  └─ Calls: execution_manager.execute_trade()
      (strategy_manager.py line 266, 270)
```

### Quaternary Path: Other Components

**Baseline Trading Kernel:**
```
baseline_trading_kernel.py line 160
└─ exec_mgr.execute_trade(...)
```

**Execution Logic:**
```
execution_logic.py lines 352, 408, 531, 547, 567
└─ execution_manager.execute_trade(...)
```

**Regime Trading Integration:**
```
regime_trading_integration.py line 256
└─ self._execute_trade(trade)
```

**Compounding Engine:**
```
compounding_engine.py line 267
└─ execution_manager.execute_trade(...)
```

---

## Who CAN Trigger SELL?

### ✅ CAN Execute SELL
1. **MetaController**
   - Centralized orchestrator
   - Routes all major trading decisions
   - Enforces policy checks

2. **TPSLEngine**
   - Monitors TP/SL levels
   - Automatically closes at hit prices
   - Integrated with ExecutionManager

3. **StrategyManager**
   - Processes agent signals
   - Executes sell recommendations
   - Subject to RiskManager approval

4. **ExecutionManager**
   - Single order execution authority
   - Enforces notional minimums
   - Handles post-fill accounting
   - **All paths converge here**

### ❌ CANNOT Execute SELL
1. **RecoveryEngine**
   - Boot-time component only
   - Rebuilds state post-crash
   - Zero order placement code

2. **RiskManager**
   - Advisory only
   - Can refuse execution via gates
   - Cannot force SELL orders
   - Can trigger kill-switch (stops trading)

3. **Exchange Truth Auditor**
   - Read-only monitoring
   - No order placement

4. **Any Database/Cache Component**
   - Persistence only
   - No execution capability

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────┐
│  SELL Decision Sources                       │
├─────────────────────────────────────────────┤
│                                              │
│  • MetaController (TP/SL, rotation, etc.)   │
│  • TPSLEngine (automatic TP/SL)              │
│  • StrategyManager (agent signals)           │
│  • ExecutionLogic (complex trading)          │
│  • Other components (regime, compounding)    │
│                                              │
└────────────────────┬────────────────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ RiskManager     │
            │ (Advisory Gate) │
            │                 │
            │ ✅ Allow?       │
            │ ❌ Deny?        │
            │ 🔴 Kill-switch? │
            └────────┬────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ ExecutionManager           │
        │ (SOLE EXECUTION AUTHORITY) │
        │                            │
        │ ✅ execute_trade()         │
        │ ✅ close_position()        │
        │ ✅ place_market_order()    │
        │ ✅ Notional checks         │
        │ ✅ Post-fill accounting    │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ ExchangeClient             │
        │                            │
        │ SELL order placed          │
        └────────────────────────────┘
```

---

## Can RecoveryEngine Indirectly Trigger SELL?

**Q:** Can RecoveryEngine somehow call ExecutionManager to trigger SELL?  
**A:** ❌ NO

**Evidence:**
1. **RecoveryEngine is ONE-TIME boot component:**
   - Runs during initialization
   - Runs during crash recovery
   - Exits after state rebuilt

2. **No ExecutionManager reference in RecoveryEngine:**
   - recovery_engine.py has no `execution_manager` attribute
   - No `close_position()` calls
   - No `execute_trade()` calls

3. **Architecture principle:**
   - RecoveryEngine: "rebuild state"
   - ExecutionManager: "execute trades"
   - These are separate concerns

4. **Code validation:**
   - recovery_engine.py line 22: "It DOES NOT place orders"
   - recovery_engine.py line 25: "Never places orders"

---

## Can RiskManager Force a SELL?

**Q:** Can RiskManager force a SELL order via ExecutionManager?  
**A:** ❌ NO (not directly)

**Evidence:**
1. **RiskManager is advisory, not executive:**
   - risk_manager.py line 35: "Advisory gate—does not execute orders"
   - RiskManager can **warn**, but cannot **veto** SELL

2. **RiskManager can only:**
   - ✅ Set daily loss caps
   - ✅ Monitor exposure
   - ✅ Trigger kill-switch (stops all trading)
   - ✅ Report health status
   - ❌ Force SELL orders
   - ❌ Override trading decisions

3. **Kill-Switch is NOT SELL:**
   ```python
   # risk_manager.py
   await self._safe_health("RiskManager", "FROZEN", f"Kill-switch: {reason}")
   # This stops NEW trading, doesn't trigger SELL of existing positions
   ```

4. **ExecutionManager line 4754:**
   ```python
   # [FIX #4] UNIFIED SELL AUTHORITY: RiskManager is advisory, not veto, for SELL
   ```

---

## Single Source of Truth (SSOT)

**ExecutionManager is the SOLE execution authority:**

```python
# execution_manager.py - ALL SELL paths converge here

async def close_position(
    self,
    symbol: str,
    reason: str = "",
    tag: str = "",
    force_finalize: bool = False,
) -> None:
    """Close a position via the canonical execution path."""

async def execute_trade(
    self,
    symbol: str,
    side: str,  # "BUY" or "SELL"
    decision: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute a single trade (BUY or SELL)."""

async def place_market_order(
    self,
    symbol: str,
    side: str,
    quantity: float,
) -> Optional[Dict[str, Any]]:
    """Place order directly on exchange."""
```

**All components use these methods:**
- ✅ MetaController → ExecutionManager.close_position()
- ✅ TPSLEngine → ExecutionManager.close_position()
- ✅ StrategyManager → ExecutionManager.execute_trade()
- ✅ RecoveryEngine → (none, read-only)
- ✅ RiskManager → (none, advisory only)

---

## Summary Table

| Component | Can SELL? | Method | Authority |
|-----------|-----------|--------|-----------|
| MetaController | ✅ YES | `close_position()` | Primary |
| TPSLEngine | ✅ YES | `close_position()` | Secondary |
| StrategyManager | ✅ YES | `execute_trade()` | Tertiary |
| ExecutionManager | ✅ YES | `close_position()` `execute_trade()` | **SOLE EXECUTOR** |
| RecoveryEngine | ❌ NO | (none) | Boot-time only |
| RiskManager | ❌ NO | (none) | Advisory only |

---

## Compliance Statement

**P9 Architecture (2025-08-20) Compliance:**
- ✅ ExecutionManager is single order path
- ✅ RecoveryEngine never places orders (read-only)
- ✅ RiskManager is advisory (not veto)
- ✅ TPSLEngine uses ExecutionManager exclusively
- ✅ MetaController is primary orchestrator
- ✅ All paths respect RiskManager gates (advisory)

---

## Conclusion

**SELL orders are triggered through:**
1. MetaController (primary decision-maker)
2. TPSLEngine (automatic TP/SL hits)
3. StrategyManager (agent signals)
4. ExecutionManager (execution authority)

**NEVER directly from:**
- ❌ RecoveryEngine (no order code)
- ❌ RiskManager (advisory only)

**Single execution path:** `ExecutionManager.close_position()` or `ExecutionManager.execute_trade()`

All SELL orders must pass through ExecutionManager's notional checks, fee calculations, and post-fill accounting.
