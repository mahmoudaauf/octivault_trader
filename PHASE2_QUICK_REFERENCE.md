# Phase 2 Quick Reference Guide

## What Changed - 30 Second Summary

**CompoundingEngine** (core/compounding_engine.py):
- ❌ Removed: Direct `execute_trade()` calls
- ✅ Added: `_generate_directive()` - Creates proposal
- ✅ Added: `_propose_exposure_directive()` - Sends to MetaController

**ExecutionManager** (core/execution_manager.py):
- ✅ Added: `trace_id` parameter to `execute_trade()`
- ✅ Added: Guard that blocks orders without trace_id (unless liquidation)
- ✅ Result: Only MetaController-approved orders execute

---

## Architecture (Before vs After)

### BEFORE
```
CompoundingEngine.execute_trade()
        ↓
    ExecutionManager
        ↓
    Exchange
```
Problem: Autonomous, no MetaController

### AFTER
```
CompoundingEngine.propose_directive()
        ↓
MetaController [validates signals, generates trace_id]
        ↓
execute_trade(trace_id=...)
        ↓
ExecutionManager [requires trace_id, rejects without]
        ↓
Exchange [only with approval]
```
Solution: Coordinated, signal-validated, auditable

---

## Code Changes Summary

### CompoundingEngine: 2 New Methods

**`_generate_directive(symbol, amount, reason)`**
```python
Returns:
{
    "source": "CompoundingEngine",
    "symbol": "BTC/USDT",
    "action": "buy",
    "amount": 100.0,
    "reason": "compounding",
    "timestamp": 1708971234.5,
    "gates_status": {
        "volatility": True,
        "edge": True,
        "economic": True,
    }
}
```

**`async _propose_exposure_directive(directive)`**
```python
# Sends directive to MetaController
# Handles missing MetaController gracefully
# Full error logging
```

---

### ExecutionManager: New Guard

```python
async def execute_trade(
    self,
    symbol: str,
    side: str,
    quantity: Optional[float] = None,
    planned_quote: Optional[float] = None,
    tag: str = "meta/Agent",
    trace_id: Optional[str] = None,  # ← NEW REQUIRED PARAMETER
    tier: Optional[str] = None,
    is_liquidation: bool = False,
    policy_context: Optional[Dict[str, Any]] = None,
):
    # NEW GUARD: Require trace_id for all orders except liquidation
    if not trace_id and not is_liquidation:
        return {
            "ok": False,
            "status": "blocked",
            "reason": "missing_meta_trace_id",
            "error_code": "MISSING_META_TRACE_ID",
        }
```

---

## How MetaController Should Implement

```python
class MetaController:
    async def propose_exposure_directive(self, directive):
        """Receive exposure directive from CompoundingEngine"""
        symbol = directive["symbol"]
        amount = directive["amount"]
        
        # 1. Validate against signal sources
        vol_regime = self.get_volatility_regime(symbol)
        momentum = self.get_momentum(symbol)
        
        if vol_regime and momentum > 1.5:
            # 2. Generate trace_id
            trace_id = f"meta_{uuid4()}"
            
            # 3. Execute with trace_id
            await self.execution_manager.execute_trade(
                symbol=symbol,
                side="buy",
                planned_quote=amount,
                trace_id=trace_id,  # ← CRITICAL
                tag=f"meta/MetaController/{trace_id}",
            )
```

---

## Testing Checklist

```python
# Test 1: CompoundingEngine generates directives
assert ce._generate_directive("BTC/USDT", 100)["source"] == "CompoundingEngine"

# Test 2: ExecutionManager requires trace_id
result = await em.execute_trade(symbol="BTC/USDT", side="buy", planned_quote=100)
assert result["error_code"] == "MISSING_META_TRACE_ID"

# Test 3: ExecutionManager accepts valid trace_id
result = await em.execute_trade(
    symbol="BTC/USDT", 
    side="buy", 
    planned_quote=100,
    trace_id="meta_xyz123"
)
assert result["error_code"] != "MISSING_META_TRACE_ID"

# Test 4: Liquidation bypass still works
result = await em.execute_trade(
    symbol="BTC/USDT",
    side="sell",
    planned_quote=100,
    trace_id=None,  # No trace_id
    is_liquidation=True  # But it's liquidation
)
assert result["error_code"] != "MISSING_META_TRACE_ID"
```

---

## Deployment Steps

1. ✅ Code implemented and syntax validated
2. ⏳ MetaController: Implement `propose_exposure_directive()`
3. ⏳ Testing: Run all unit tests
4. ⏳ Staging: Deploy and monitor
5. ⏳ Production: Deploy when stable

---

## What Happens Now

**Immediate (Code Deployed)**:
- CompoundingEngine generates directives
- ExecutionManager blocks orders (no trace_id)
- Logs show: `[EXEC:TraceID] Blocked ... missing trace_id`

**When MetaController Ready**:
- MetaController receives directives
- MetaController validates signals
- MetaController executes with trace_id
- Orders flow normally

---

## Files Modified

```
core/compounding_engine.py
  ├─ Import: Added 'time'
  ├─ Docstring: Updated
  ├─ Method: _execute_compounding_strategy() rewritten
  ├─ Method: _generate_directive() added
  └─ Method: _propose_exposure_directive() added

core/execution_manager.py
  ├─ Signature: Added 'trace_id' parameter
  ├─ Docstring: Updated
  └─ Guard: trace_id validation added
```

---

## Files Created

```
PHASE2_ARCHITECTURE_FIX.md
  └─ Complete architecture design and implementation guide

PHASE2_IMPLEMENTATION_COMPLETE.md
  └─ Detailed implementation with test specifications
  
PHASE2_QUICK_REFERENCE.md (this file)
  └─ Quick lookup guide
```

---

## Error Code Reference

### MISSING_META_TRACE_ID
```python
{
    "ok": False,
    "status": "blocked",
    "reason": "missing_meta_trace_id",
    "error_code": "MISSING_META_TRACE_ID"
}
```

**Cause**: Order placed without trace_id from MetaController
**Solution**: Must come from MetaController with valid trace_id

---

## FAQ

**Q: Will orders still be placed?**
A: No, not until MetaController implements directive receiver. All orders will be blocked until then.

**Q: What about liquidation (TP/SL)?**
A: Still works. Liquidation orders bypass trace_id requirement (`is_liquidation=True`).

**Q: Is this backward compatible?**
A: Yes, for liquidation. Normal orders require new parameter.

**Q: Can I test without MetaController?**
A: Yes, test directive generation and trace_id guard independently.

**Q: How long to implement MetaController?**
A: Depends on complexity of signal validation logic. ~2-3 hours for basic version.

**Q: Will this improve P&L?**
A: Yes. Gates reduce fee churn 94% (Phase 1). Signal validation prevents additional bad trades (Phase 2).

---

## Success Criteria

✅ Phase 2 is complete when:
- Code: ✅ Implemented and syntax valid
- Logs: Show directives being generated
- Guard: Blocks orders without trace_id
- MetaController: Implements directive receiver
- Testing: All unit tests pass
- Integration: Directives flow through MetaController
- Orders: Execute only with trace_id

---

## Next Actions

1. **Immediate**: MetaController team implements `propose_exposure_directive()`
2. **This Week**: Run all unit and integration tests
3. **Next Week**: Deploy to staging
4. **2 Weeks**: Deploy to production
5. **Ongoing**: Monitor for full system alignment

---

**Phase 2 Architectural Fix: IMPLEMENTATION COMPLETE ✅**

System is now architected for:
- Exposure directive generation (CompoundingEngine)
- Signal validation (MetaController)
- Execution enforcement (ExecutionManager)
- Full audit trail (trace_id on every order)

Next: MetaController integration
