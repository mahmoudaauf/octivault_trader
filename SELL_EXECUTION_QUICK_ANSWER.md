# ⚡ Quick Answer: SELL Execution Authority

## Your Question
**Can a SELL be directly executed through RecoveryEngine or RiskManager?**

## Answer
### ❌ NO - NEITHER

**RecoveryEngine:** Boot-time component only, no order execution code  
**RiskManager:** Advisory gate only, cannot execute orders

**✅ SELL can ONLY be executed through:**
1. MetaController (calls ExecutionManager)
2. TPSLEngine (calls ExecutionManager)
3. StrategyManager (calls ExecutionManager)
4. ExecutionManager (sole executor)

---

## Who Executes SELL?

| Component | SELL? | Why |
|-----------|-------|-----|
| **ExecutionManager** | ✅ **YES** | Only component with `place_market_order()` |
| MetaController | ✅ Calls ExecutionManager | Primary decision-maker |
| TPSLEngine | ✅ Calls ExecutionManager | Automatic TP/SL hits |
| StrategyManager | ✅ Calls ExecutionManager | Agent signals |
| **RecoveryEngine** | ❌ **NO** | Zero order code, boot-time only |
| **RiskManager** | ❌ **NO** | Advisory only, cannot execute |

---

## Architecture

```
SELL Decision
  ↓
RiskManager Gate (Advisory)
  ↓
ExecutionManager (Only executor)
  ↓
ExchangeClient (API call)
```

---

## Evidence

**RecoveryEngine** (recovery_engine.py line 22-25):
> "It DOES NOT place orders."
> "Never places orders (ExecutionManager remains the single order path)"

**RiskManager** (risk_manager.py line 35):
> "Advisory gate—does not execute orders"

**ExecutionManager** (execution_manager.py line 4754):
> "[FIX #4] UNIFIED SELL AUTHORITY: RiskManager is advisory, not veto, for SELL"

---

## Key Insight

**RecoveryEngine:**
- What: Rebuilds state after crash
- When: Boot-time (one-time)
- Methods: `rebuild_state()`, `verify_integrity()`
- Orders: ❌ ZERO

**RiskManager:**
- What: Advisory risk gating
- When: Runtime (continuous)
- Methods: `can_execute()`, `get_health()`
- Orders: ❌ ZERO

**ExecutionManager:**
- What: Executes ALL trades
- When: Runtime (on-demand)
- Methods: `close_position()`, `execute_trade()`, `place_market_order()`
- Orders: ✅ **SOLE EXECUTOR**

---

## For Documentation

- See: `SELL_EXECUTION_AUTHORITY.md` (comprehensive analysis)
- See: `SELL_AUTHORITY_DIAGRAM.md` (visual architecture)
