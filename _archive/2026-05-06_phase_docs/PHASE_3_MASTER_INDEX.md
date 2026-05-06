# 🚀 PHASE 3 MASTER INDEX
*Complete reference for wiring all 16 methods*

---

## 📊 QUICK STATS

- **Duration:** ~2 hours 45 minutes
- **Methods to wire:** 16 across 5 engines
- **Components to connect:** 22 L0-L8 layers
- **Difficulty:** Easy (copy/paste pattern)
- **Risk:** Minimal (all patterns tested)
- **Critical feature:** FIX #2 guard in place_sell_order()

---

## 🎯 THE 5-STEP WIRING SEQUENCE

### STEP 1: MarketAccountEngine (30 min)
**File:** `core_engine/market_account_engine.py`
**Methods:** 4 to replace
**Complexity:** Low (direct exchange calls)

| Method | Components | Status |
|--------|-----------|--------|
| `get_account_state()` | exchange_client (L1) | ⏳ Ready |
| `get_market_prices()` | market_data_feed (L1/L2) | ⏳ Ready |
| `get_wallet_balance()` | balance_manager (L2) | ⏳ Ready |
| `get_ohlcv_data()` | market_data_feed (L2) | ⏳ Ready |

**Import needed:** `from core_engine.implementations import MarketAccountEngineImpl`

**Reference:** `PHASE_2_INTEGRATION_GUIDE.md` → Step 1

**Example pattern:**
```python
async def get_account_state(self) -> Dict[str, Any]:
    """Fetch account state from exchange_client (L1)."""
    return await MarketAccountEngineImpl.get_account_state(self.app_ctx)
```

---

### STEP 2: SituationEngine (30 min)
**File:** `core_engine/situation_engine.py`
**Methods:** 4 to replace
**Complexity:** Medium (multi-component synthesis)

| Method | Components | Status |
|--------|-----------|--------|
| `get_portfolio_snapshot()` | portfolio_manager (L3) | ⏳ Ready |
| `get_all_signals()` | signal_manager (L5) | ⏳ Ready |
| `get_fused_signal()` | signal_fusion (L5) | ⏳ Ready |
| `get_market_regime()` | regime_detector (L2) | ⏳ Ready |

**Import needed:** `from core_engine.implementations import SituationEngineImpl`

**Reference:** `PHASE_2_INTEGRATION_GUIDE.md` → Step 2

**Example pattern:**
```python
async def get_portfolio_snapshot(self) -> Dict[str, Any]:
    """Get portfolio state from portfolio_manager (L3)."""
    return await SituationEngineImpl.get_portfolio_snapshot(self.app_ctx)
```

---

### STEP 3: DecisionEngine (20 min)
**File:** `core_engine/decision_engine.py`
**Methods:** 3 to replace
**Complexity:** Medium (decision arbitration)

| Method | Components | Status |
|--------|-----------|--------|
| `get_current_mode()` | mode_manager (L5) | ⏳ Ready |
| `evaluate_signal()` | arbitration_engine (L6) | ⏳ Ready |
| `make_buy_decision()` | capital_allocator (L5/L6) | ⏳ Ready |

**Import needed:** `from core_engine.implementations import DecisionEngineImpl`

**Reference:** `PHASE_2_INTEGRATION_GUIDE.md` → Step 3

**Example pattern:**
```python
async def get_current_mode(self) -> str:
    """Get current trading mode from mode_manager (L5)."""
    return await DecisionEngineImpl.get_current_mode(self.app_ctx)
```

---

### STEP 4: SafeExecutionEngine (30 min) ⭐ INCLUDES FIX #2
**File:** `core_engine/safe_execution_engine.py`
**Methods:** 3 to replace
**Complexity:** High (critical safety gates + FIX #2)

| Method | Components | Status |
|--------|-----------|--------|
| `validate_order()` | order_validator (L4) | ⏳ Ready |
| `place_buy_order()` | execution_manager (L4) | ⏳ Ready |
| `place_sell_order()` ⭐ | bounded_cache (L0) + execution_manager (L4) | ⏳ Ready |

**Import needed:** `from core_engine.implementations import SafeExecutionEngineImpl`

**Reference:** `PHASE_2_INTEGRATION_GUIDE.md` → Step 4

**⭐ CRITICAL - FIX #2 Guard Details:**
- **Location:** `place_sell_order()` method body
- **Mechanism:** bounded_cache lookup with 5-minute TTL
- **Purpose:** Prevent duplicate SELL execution on recovery
- **Implementation:** Already included in SafeExecutionEngineImpl
- **Confidence:** 99% (per specification)

**Example pattern:**
```python
async def place_sell_order(self, symbol: str, quantity: float,
                          price: Optional[float] = None,
                          order_type: str = "LIMIT") -> Dict[str, Any]:
    """Place SELL order with FIX #2 idempotent guard. ⭐ CRITICAL"""
    return await SafeExecutionEngineImpl.place_sell_order(
        self.app_ctx, symbol, quantity, price, order_type)
```

---

### STEP 5: OperationsEngine (20 min)
**File:** `core_engine/operations_engine.py`
**Methods:** 2 to replace
**Complexity:** Low (system health + recovery)

| Method | Components | Status |
|--------|-----------|--------|
| `startup_system()` | All L0→L8 components | ⏳ Ready |
| `get_health_report()` | health_monitor (L7) | ⏳ Ready |

**Import needed:** `from core_engine.implementations import OperationsEngineImpl`

**Reference:** `PHASE_2_INTEGRATION_GUIDE.md` → Step 5

**Example pattern:**
```python
async def startup_system(self) -> bool:
    """Execute system startup (L0→L8)."""
    return await OperationsEngineImpl.startup_system(self.app_ctx)
```

---

## 📁 DOCUMENTATION MAP

### START HERE 👈
- **`PHASE_3_QUICK_START.md`** - The pattern to follow (read first!)
- **`PHASE_3_START.sh`** - Run this to verify readiness

### THEN USE THESE
- **`PHASE_3_WIRING_CHECKLIST.md`** - Check off each method as you complete
- **`core_engine/WIRING_EXAMPLES.py`** - Copy/paste examples for reference
- **`PHASE_2_INTEGRATION_GUIDE.md`** - Component interface specifications
- **`core_engine/implementations.py`** - Source code to wire from

### FOR REFERENCE
- **`COMPLETE_ARCHITECTURE_FLOW.md`** - System architecture overview
- **`PHASE_2_MASTER_INDEX.md`** - All Phase 2 documentation
- **`PHASE_2_DELIVERY_SUMMARY.txt`** - What Phase 2 delivered

---

## ⚙️ THE WIRING PATTERN (Use for all 16 methods)

### For each method:

1. **LOCATE** - Find in engine file
   ```
   File: core_engine/{engine_name}_engine.py
   Search for: def {method_name}(
   ```

2. **ADD IMPORT** - At top of file
   ```python
   from core_engine.implementations import {EngineImpl}
   ```

3. **REPLACE BODY** - Change the method
   ```python
   # FROM:
   async def method_name(self, ...) -> ReturnType:
       """Docstring."""
       await asyncio.sleep(0.1)
       return {}

   # TO:
   async def method_name(self, ...) -> ReturnType:
       """Docstring."""
       return await {EngineImpl}.method_name(self.app_ctx, ...)
   ```

4. **TEST** - Verify compilation
   ```bash
   python3 -m py_compile core_engine/{engine_name}_engine.py
   ```

5. **COMMIT** - Save progress
   ```bash
   git add core_engine/{engine_name}_engine.py
   git commit -m "[Phase 3] Wire {engine_name}.{method_name}()"
   ```

---

## 🧪 TESTING STRATEGY

### After each engine (5 checkpoint tests):
```bash
# Test compilation
python3 -m py_compile core_engine/{engine_name}_engine.py

# Run specific tests if available
pytest tests/test_{engine_name}.py -v
```

### Final validation (after all 5 engines):
```bash
# Check all engines compile
for engine in market_account situation decision safe_execution operations; do
    python3 -m py_compile core_engine/${engine}_engine.py
done

# Run full test suite
pytest tests/ -v

# Check that FIX #2 is in place
grep -n "bounded_cache" core_engine/safe_execution_engine.py
```

---

## ⏱️ DETAILED TIMELINE

| Step | Task | Duration | Cumulative |
|------|------|----------|-----------|
| Setup | Read docs, create branch | 5 min | 5 min |
| Step 1 | MarketAccountEngine (4 methods) | 30 min | 35 min |
| Step 2 | SituationEngine (4 methods) | 30 min | 65 min |
| Step 3 | DecisionEngine (3 methods) | 20 min | 85 min |
| Step 4 | SafeExecutionEngine (3 methods) | 30 min | 115 min |
| Step 5 | OperationsEngine (2 methods) | 20 min | 135 min |
| Final | Testing & verification | 30 min | **165 min** |

**Total: ~2 hours 45 minutes**

---

## ✅ SUCCESS CRITERIA

When Phase 3 is complete, you'll have:

- ✅ All 16 methods replaced with real implementations
- ✅ All 5 engines wired to 22 L0-L8 components
- ✅ FIX #2 guard active in place_sell_order()
- ✅ All Python files compile without errors
- ✅ All methods have correct type hints
- ✅ Error handling in place for all components
- ✅ System ready for integration testing
- ✅ Git history with clean commits for each method

---

## 🚨 TROUBLESHOOTING

### Issue: Import not found
**Solution:** Check line 1 of the file has the correct import
```python
from core_engine.implementations import {EngineImpl}
```

### Issue: Method signature doesn't match
**Solution:** Compare with WIRING_EXAMPLES.py for correct arguments
```python
# Check that method call includes all required arguments
await {EngineImpl}.method_name(self.app_ctx, arg1, arg2, ...)
```

### Issue: Indentation error
**Solution:** Make sure method body is indented properly (4 spaces)
```python
async def method_name(self) -> Type:
    """Docstring."""
    return await Implementation.method_name(self.app_ctx)  # ← 8 spaces
```

### Issue: Compilation fails
**Solution:** Check for syntax errors
```bash
# Try to compile and see error
python3 -m py_compile core_engine/engine_name_engine.py

# Or check syntax directly
python3 -m py_compile --verbose core_engine/engine_name_engine.py
```

### Issue: Can't find FIX #2 guard
**Solution:** It's automatic in SafeExecutionEngineImpl.place_sell_order()
```python
# Just paste the method - the guard is already there
return await SafeExecutionEngineImpl.place_sell_order(self.app_ctx, ...)
```

---

## 🎓 LEARNING PATH

If this is your first time:

1. **Read** `PHASE_3_QUICK_START.md` (5 min)
   - Understand the pattern

2. **Run** `./PHASE_3_START.sh` (1 min)
   - Verify everything is ready

3. **Do Step 1** - MarketAccountEngine (30 min)
   - 4 simple methods, direct exchange calls
   - Easiest to understand the pattern

4. **Do Step 2** - SituationEngine (30 min)
   - Same pattern, more components
   - You'll start getting comfortable

5. **Do Step 3** - DecisionEngine (20 min)
   - Pattern is identical, fewer methods
   - You'll be fast now

6. **Do Step 4** - SafeExecutionEngine (30 min)
   - CRITICAL: This has FIX #2 guard
   - Same pattern, just with extra responsibility
   - Don't skip or shortcut this

7. **Do Step 5** - OperationsEngine (20 min)
   - Last 2 methods, pattern is automatic
   - You're almost done!

8. **Final validation** (30 min)
   - Test everything works
   - Celebrate! 🎉

---

## 🎯 NEXT AFTER PHASE 3

Once Phase 3 is complete:

- **Phase 4:** Integration testing
  - Test all 5 engines together
  - Verify component connections
  - Validate FIX #2 guard behavior

- **Phase 5:** Performance profiling
  - Optimize hot paths
  - Monitor memory usage
  - Check async patterns

- **Phase 6:** Production deployment
  - Package the system
  - Deploy to live trading
  - Monitor in production

---

## 💡 TIPS FOR SUCCESS

1. **Don't skip testing** - Test after each engine
2. **Commit frequently** - After each engine (5 commits)
3. **Follow the pattern** - All 16 methods are identical except arguments
4. **Use the examples** - WIRING_EXAMPLES.py has every pattern you need
5. **Take your time** - 2h45m is achievable at a comfortable pace
6. **Reference the specs** - PHASE_2_INTEGRATION_GUIDE.md is your friend
7. **Check FIX #2** - Verify place_sell_order() includes the guard
8. **Celebrate milestones** - You complete an entire engine every 30 min!

---

## 🚀 LET'S GOOO!

You have:
- ✅ All code needed (implementations.py)
- ✅ All examples (WIRING_EXAMPLES.py)
- ✅ All specs (PHASE_2_INTEGRATION_GUIDE.md)
- ✅ All patterns (this checklist)
- ✅ All documentation (guides & references)

No guessing required. Just follow the pattern and you'll be done in under 3 hours!

**Start with:** `PHASE_3_QUICK_START.md`
**Track with:** `PHASE_3_WIRING_CHECKLIST.md`
**Reference:** All other files as needed

---

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  16 methods → 22 components → 5 engines → 1 complete system        ║
║                                                                      ║
║  You've got this! 💪                                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```
