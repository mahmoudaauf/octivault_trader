# Phase 2: Architectural Fix - CompoundingEngine as Exposure Directive Generator

## Overview
Convert CompoundingEngine from autonomous execution agent to exposure directive generator that:
1. **Proposes exposure directives** (what to buy) instead of executing directly
2. **Routes through MetaController** for decision validation
3. **ExecutionManager rejects orders** without valid MetaController `trace_id`

## Changes Required

### 1. CompoundingEngine Changes
**File**: `core/compounding_engine.py`

#### Remove Direct Execution
- Remove: Direct calls to `execution_manager.execute_trade()`
- Replace with: Exposure directive proposals

#### Add Exposure Directive Generator
```python
async def propose_exposure_directives(self) -> List[Dict[str, Any]]:
    """
    Generate exposure directives (not execute them).
    Each directive is a proposal to the MetaController.
    """
```

#### New Method: _generate_directive
```python
def _generate_directive(
    self,
    symbol: str,
    amount: float,
    reason: str = "compounding"
) -> Dict[str, Any]:
    """
    Generate an exposure directive without executing.
    
    Returns:
        {
            "source": "CompoundingEngine",
            "symbol": symbol,
            "action": "buy",
            "amount": amount,
            "reason": reason,
            "gates_status": {
                "volatility": passed,
                "edge": passed,
                "economic": passed
            }
        }
    """
```

### 2. ExecutionManager Changes
**File**: `core/execution_manager.py`

#### Add trace_id Validation
```python
async def execute_trade(
    self,
    symbol: str,
    side: str,
    quantity: Optional[float] = None,
    planned_quote: Optional[float] = None,
    tag: str = "meta/Agent",
    trace_id: Optional[str] = None,  # NEW: Required from MetaController
    ...
) -> Dict[str, Any]:
    """
    NOW REQUIRES: trace_id from MetaController decision
    """
```

#### Add Guard at Entry
```python
# NEW: Require trace_id for all orders except liquidation
if not trace_id and not is_liquidation:
    self.logger.warning(
        "[EXEC:TraceID] Blocked %s %s: missing trace_id from MetaController",
        side, symbol
    )
    return {
        "ok": False,
        "status": "blocked",
        "reason": "missing_meta_trace_id",
        "error_code": "MISSING_META_TRACE_ID",
    }
```

## Implementation Flow

### Before (Current - Broken)
```
CompoundingEngine
    ↓ (reads raw scores directly)
    ↓ (decides autonomously)
    ↓ execute_trade() ← no MetaController involvement
    ↓
ExecutionManager
    ↓
Exchange
```

### After (Fixed - Phase 2)
```
CompoundingEngine
    ↓ (analyzes market conditions)
    ↓ propose_exposure_directives()
    ↓ → proposes to MetaController
    ↓
MetaController
    ↓ (validates against all signals)
    ↓ generates trace_id
    ↓ issues decision
    ↓
ExecutionManager
    ↓ (requires trace_id in execute_trade)
    ↓ (rejects if no trace_id)
    ↓
Exchange (only with MetaController approval)
```

## Code Changes

### CompoundingEngine: Stop Direct Execution
**Remove this block** from `_execute_compounding_strategy()`:
```python
# OLD: Direct execution
res = await self.execution_manager.execute_trade(
    symbol=symbol,
    side="buy",
    planned_quote=planned,
    tag="meta/CompoundingEngine",
)
```

**Replace with**:
```python
# NEW: Generate directive, don't execute
directive = self._generate_directive(
    symbol=symbol,
    amount=planned,
    reason="compounding"
)
await self._propose_exposure_directive(directive)
logger.info("📋 Proposed exposure directive: %s", directive)
```

### CompoundingEngine: Add New Methods
```python
def _generate_directive(
    self,
    symbol: str,
    amount: float,
    reason: str = "compounding"
) -> Dict[str, Any]:
    """Generate exposure directive without executing."""
    return {
        "source": "CompoundingEngine",
        "symbol": symbol,
        "action": "buy",
        "amount": amount,
        "reason": reason,
        "timestamp": time.time(),
        "gates_status": {
            "volatility": True,  # Already passed gates before here
            "edge": True,
            "economic": True,
        }
    }

async def _propose_exposure_directive(
    self,
    directive: Dict[str, Any]
) -> None:
    """Send directive to MetaController for decision."""
    meta_controller = self.shared_state.get("meta_controller")
    if not meta_controller:
        logger.warning("MetaController not available, caching directive")
        return
    
    try:
        await meta_controller.propose_exposure_directive(directive)
        logger.info("✅ Proposed directive: %s", directive["symbol"])
    except Exception as e:
        logger.error("Failed to propose directive: %s", e)
```

### ExecutionManager: Add trace_id Guard
**Add at start of execute_trade()**, after signature but before main logic:

```python
# NEW: Require trace_id for all orders (except liquidation)
if not trace_id and not is_liquidation:
    self.logger.warning(
        "[EXEC:TraceID] Blocked %s %s: missing trace_id from MetaController",
        side.upper(), sym,
    )
    return {
        "ok": False,
        "status": "blocked",
        "reason": "missing_meta_trace_id",
        "error_code": "MISSING_META_TRACE_ID",
    }
```

**Add to method signature**:
```python
async def execute_trade(
    self,
    symbol: str,
    side: str,
    quantity: Optional[float] = None,
    planned_quote: Optional[float] = None,
    tag: str = "meta/Agent",
    trace_id: Optional[str] = None,  # NEW
    tier: Optional[str] = None,
    is_liquidation: bool = False,
    policy_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

## MetaController Integration

MetaController must:
1. **Receive directives** from CompoundingEngine
2. **Validate** against all signal sources
3. **Generate trace_id** for approved orders
4. **Issue decision** to ExecutionManager with trace_id

```python
class MetaController:
    async def propose_exposure_directive(
        self,
        directive: Dict[str, Any]
    ) -> None:
        """Receive exposure directive from CompoundingEngine."""
        symbol = directive["symbol"]
        
        # Get all signals
        volatility_regime = self.get_volatility_regime(symbol)
        momentum = self.get_momentum(symbol)
        
        # Validate
        if not volatility_regime or momentum < 1.0:
            logger.info("Rejected directive: %s (signals inactive)", symbol)
            return
        
        # Generate trace_id and execute
        trace_id = self._generate_trace_id()
        await self.execution_manager.execute_trade(
            symbol=symbol,
            side="buy",
            planned_quote=directive["amount"],
            trace_id=trace_id,  # NEW: Include trace_id
            tag=f"meta/MetaController/{trace_id}",
        )
```

## Benefits of This Architecture

### ✅ Unified Decision Chain
- Single point of control (MetaController)
- All orders have signal validation
- No autonomous agents bypassing coordination

### ✅ Audit Trail
- Every order has trace_id
- Can trace: CompoundingEngine → MetaController → ExecutionManager → Exchange
- Complete visibility into decision chain

### ✅ Economic Safety
- Gates prevent bad proposals (CompoundingEngine)
- Signals prevent bad approvals (MetaController)
- Double validation before execution

### ✅ Risk Management
- MetaController sees all directives
- Can apply portfolio constraints
- Can halt CompoundingEngine if needed

## Migration Path

### Step 1: Add Infrastructure
- Add trace_id validation to ExecutionManager
- Add directive proposal to CompoundingEngine
- Verify no orders execute (they'll be blocked for now)

### Step 2: Connect MetaController
- Implement directive receiver in MetaController
- Generate trace_id for approved directives
- Start issuing orders with trace_id

### Step 3: Validation
- Monitor that all orders have trace_id
- Verify no orders bypass MetaController
- Confirm signal validation working

### Step 4: Rollout
- Deploy to staging
- Monitor for 1 week
- Deploy to production

## Expected Results

### Phase 2 Impact
- ✅ CompoundingEngine no longer autonomous
- ✅ All orders require MetaController approval
- ✅ Signal validation applies to all buys
- ✅ Economic layer properly engaged
- ✅ System operates as designed

### Combined Phase 1 + Phase 2
- 94% fee reduction (gates)
- 100% signal validation (MetaController)
- Proper risk management (coordinator)
- Sustainable profitability (both phases)

## Testing Strategy

### Unit Tests
```python
def test_execute_trade_requires_trace_id():
    """ExecutionManager rejects order without trace_id"""
    result = await em.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        planned_quote=100,
        trace_id=None,  # No trace_id
    )
    assert result["ok"] == False
    assert "trace_id" in result["reason"]

def test_directive_generation():
    """CompoundingEngine generates proper directives"""
    directive = ce._generate_directive(
        symbol="BTC/USDT",
        amount=100,
    )
    assert directive["source"] == "CompoundingEngine"
    assert directive["amount"] == 100

def test_meta_approved_order_executes():
    """Order with trace_id from MetaController executes"""
    trace_id = "meta_12345"
    result = await em.execute_trade(
        symbol="BTC/USDT",
        side="buy",
        planned_quote=100,
        trace_id=trace_id,
    )
    assert result["ok"] == True
```

## Rollback Plan

If Phase 2 causes issues:
1. Revert ExecutionManager (remove trace_id guard)
2. Revert CompoundingEngine (restore execute_trade calls)
3. System returns to Phase 1 (gates only)

No data corruption, clean rollback.

## Summary

**Phase 2 Architecture Fix**:
- CompoundingEngine → Exposure Directive Generator (not executor)
- ExecutionManager → Requires MetaController trace_id
- MetaController → Validates and approves directives
- Result → Unified, coordinated system

**Timeline**: 2-3 hours implementation, 2-4 hours testing, 1 week validation
**Risk**: Medium (architectural change, but gates still protect)
**Value**: Sustainable system architecture + signal validation

---

**This completes the architectural realignment that makes the system operate as designed.**
