"""
PHASE 3 - METHOD IMPLEMENTATION & WIRING
═════════════════════════════════════════════════════════════════════════════

🚀 PHASE 3 LAUNCH - Bring All 5 Engines to Life
Status: READY TO START
Expected Duration: 2.5-3 hours
Objective: Replace all 16 methods with real implementations

═════════════════════════════════════════════════════════════════════════════
PHASE 3 OVERVIEW
═════════════════════════════════════════════════════════════════════════════

Phase 1 (COMPLETE):  Created 5 façade engines + 10 data classes
Phase 2 (COMPLETE):  Created implementation layer + documentation
Phase 3 (NOW):       Wire engines to real components (16 methods)
Phase 4 (PENDING):   Integration testing & validation
Phase 5 (PENDING):   Production deployment

Phase 3 Objective:
  Replace placeholder methods in all 5 engines with real implementations
  that call actual L0-L8 components


═════════════════════════════════════════════════════════════════════════════
WHAT YOU HAVE (FROM PHASE 2)
═════════════════════════════════════════════════════════════════════════════

✅ Real Implementations (ready to use):
   /core_engine/implementations.py (550 lines)
   - MarketAccountEngineImpl
   - SituationEngineImpl
   - DecisionEngineImpl
   - SafeExecutionEngineImpl (+ FIX #2 guard)
   - OperationsEngineImpl

✅ Copy/Paste Examples (ready to use):
   /core_engine/WIRING_EXAMPLES.py (400 lines)
   - 7 complete code examples
   - Full pytest test patterns
   - FIX #2 guard test

✅ Step-by-Step Guide (ready to use):
   /PHASE_2_INTEGRATION_GUIDE.md (400 lines)
   - 5-step integration sequence
   - Component interface specifications
   - 25-item verification checklist

✅ Architecture Reference (ready to use):
   /COMPLETE_ARCHITECTURE_FLOW.md (500 lines)
   - System overview diagram
   - Function flows (READ/UNDERSTAND/DECIDE/EXECUTE/RECOVER)
   - Component dependencies matrix


═════════════════════════════════════════════════════════════════════════════
PHASE 3 IMPLEMENTATION PLAN
═════════════════════════════════════════════════════════════════════════════

Total Methods to Replace: 16
Estimated Time: 2.5-3 hours
Pattern: Import Impl class → Replace method body → Test

Step 1: MarketAccountEngine (4 methods)
───────────────────────────────────────

Location: /core_engine/market_account_engine.py

Methods to Replace:
  1. get_account_state()
  2. get_market_prices()
  3. get_wallet_balance()
  4. get_ohlcv_data()

Pattern:
  FROM: async def method_name(self, ...): await asyncio.sleep(0.1)
  TO:   async def method_name(self, ...):
        return await MarketAccountEngineImpl.method_name(self.app_ctx, ...)

Time Estimate: 30 minutes
Reference: PHASE_2_INTEGRATION_GUIDE.md → Step 1
Examples: core_engine/WIRING_EXAMPLES.py → MARKET_ACCOUNT_ENGINE_WIRING


Step 2: SituationEngine (4 methods)
────────────────────────────────────

Location: /core_engine/situation_engine.py

Methods to Replace:
  1. get_portfolio_snapshot()
  2. get_all_signals()
  3. get_fused_signal()
  4. get_market_regime()

Pattern: Same as Step 1 (import, replace, test)

Time Estimate: 30 minutes
Reference: PHASE_2_INTEGRATION_GUIDE.md → Step 2
Examples: core_engine/WIRING_EXAMPLES.py → SITUATION_ENGINE_WIRING


Step 3: DecisionEngine (3 methods)
───────────────────────────────────

Location: /core_engine/decision_engine.py

Methods to Replace:
  1. get_current_mode()
  2. evaluate_signal()
  3. make_buy_decision()

Pattern: Same as Step 1 (import, replace, test)

Time Estimate: 20 minutes
Reference: PHASE_2_INTEGRATION_GUIDE.md → Step 3
Examples: core_engine/WIRING_EXAMPLES.py → DECISION_ENGINE_WIRING


Step 4: SafeExecutionEngine (3 methods) ⭐ CRITICAL
─────────────────────────────────────────────────────

Location: /core_engine/safe_execution_engine.py

Methods to Replace:
  1. validate_order()
  2. place_buy_order()
  3. place_sell_order() ⭐ WITH FIX #2 GUARD

⭐ IMPORTANT: place_sell_order() includes FIX #2 guard:
   - Check bounded_cache for existing finalization
   - Prevent duplicate SELL on system recovery
   - Mark completion in cache with 5-minute TTL
   - Return "ALREADY_FINALIZED" if duplicate detected

Pattern: Same as Step 1 (import, replace, test)

Time Estimate: 30 minutes
Reference: PHASE_2_INTEGRATION_GUIDE.md → Step 4
Examples: core_engine/WIRING_EXAMPLES.py → SAFE_EXECUTION_ENGINE_WIRING
Test: WIRING_EXAMPLES.py → test_fix2_duplicate_sell_prevention()


Step 5: OperationsEngine (2 methods)
──────────────────────────────────────

Location: /core_engine/operations_engine.py

Methods to Replace:
  1. startup_system()
  2. get_health_report()

Pattern: Same as Step 1 (import, replace, test)

Time Estimate: 20 minutes
Reference: PHASE_2_INTEGRATION_GUIDE.md → Step 5
Examples: core_engine/WIRING_EXAMPLES.py → OPERATIONS_ENGINE_WIRING


═════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP EXECUTION FLOW
═════════════════════════════════════════════════════════════════════════════

For Each Method:

1. LOCATE
   └─ File: /core_engine/{engine_name}.py
   └─ Find: async def method_name(...)
   └─ Current: await asyncio.sleep(0.1)

2. UNDERSTAND
   └─ Read spec: PHASE_2_INTEGRATION_GUIDE.md → relevant step
   └─ Review: What components this method calls
   └─ Check: Parameter types and return types

3. COPY
   └─ Open: core_engine/implementations.py
   └─ Find: {EngineImpl}.method_name()
   └─ Copy: The entire method body

4. PASTE
   └─ Replace: Full method body in engine file
   └─ Add: Import statement at top of file
   └─ Result: async def method_name(...):
              return await EngineImpl.method_name(self.app_ctx, ...)

5. TEST
   └─ Run: pytest for that engine
   └─ Verify: Method returns expected type
   └─ Check: Error handling works

6. COMMIT
   └─ Add: Changes to git
   └─ Message: "[Phase 3] Implement {engine_name}.{method_name}()"
   └─ Continue: Next method


═════════════════════════════════════════════════════════════════════════════
DETAILED WIRING PATTERN
═════════════════════════════════════════════════════════════════════════════

BEFORE (Placeholder):
─────────────────────
async def get_account_state(self) -> Dict[str, Any]:
    \"\"\"Fetch account state from exchange_client (L1).\"\"\"
    await asyncio.sleep(0.1)  # ← Placeholder
    return {}


AFTER (Real Implementation):
────────────────────────────
from core_engine.implementations import MarketAccountEngineImpl

async def get_account_state(self) -> Dict[str, Any]:
    \"\"\"Fetch account state from exchange_client (L1).\"\"\"
    return await MarketAccountEngineImpl.get_account_state(self.app_ctx)


KEY POINTS:
  1. Import goes at the top: from core_engine.implementations import ...
  2. Method body becomes: return await EngineImpl.method_name(self.app_ctx, ...)
  3. All parameters pass through: (self.app_ctx, param1, param2, ...)
  4. Return type unchanged: Still Dict[str, Any] or Optional[...], etc.
  5. Docstring unchanged: Keep original docstring


═════════════════════════════════════════════════════════════════════════════
IMPORTS NEEDED
═════════════════════════════════════════════════════════════════════════════

Add to each engine file (one per file):

/core_engine/market_account_engine.py:
  from core_engine.implementations import MarketAccountEngineImpl

/core_engine/situation_engine.py:
  from core_engine.implementations import SituationEngineImpl

/core_engine/decision_engine.py:
  from core_engine.implementations import DecisionEngineImpl

/core_engine/safe_execution_engine.py:
  from core_engine.implementations import SafeExecutionEngineImpl

/core_engine/operations_engine.py:
  from core_engine.implementations import OperationsEngineImpl


═════════════════════════════════════════════════════════════════════════════
TESTING STRATEGY
═════════════════════════════════════════════════════════════════════════════

After Each Engine:
  1. Run: pytest core_engine/tests/test_{engine_name}.py -v
  2. Verify: All methods return expected types
  3. Check: No import errors
  4. Test: Error handling paths

After All Engines:
  1. Run: pytest core_engine/tests/ -v
  2. Verify: Full test suite passes
  3. Test: FIX #2 guard (duplicate sell prevention)
  4. Integration: Full READ→UNDERSTAND→DECIDE→EXECUTE→RECOVER cycle


FIX #2 Specific Test:
  pytest core_engine/tests/test_fix2_idempotent_guard.py -v
  └─ Verify: Duplicate SELL is prevented
  └─ Verify: Cache returns "ALREADY_FINALIZED"
  └─ Verify: 5-minute TTL works


═════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING GUIDE
═════════════════════════════════════════════════════════════════════════════

Problem: "AttributeError: module has no attribute"
Solution: Check import statement is added at top of file
  Example: from core_engine.implementations import MarketAccountEngineImpl

Problem: "TypeError: method takes X arguments but Y were given"
Solution: Check all parameters are being passed through
  Example: return await EngineImpl.method_name(self.app_ctx, param1, param2)

Problem: "Test fails with unexpected return type"
Solution: Verify return type in implementation matches expected
  Check: PHASE_2_INTEGRATION_GUIDE.md for expected type
  Fix: May need to adjust parameters or return mapping

Problem: "Component not found" warning in logs
Solution: This is expected if component not initialized
  Check: app_ctx has the component registered
  Verify: Implementation has fallback logic (graceful degradation)

Problem: "FIX #2 guard not working"
Solution: Check bounded_cache is in app_ctx
  Verify: set/get methods work correctly
  Test: test_fix2_duplicate_sell_prevention() passes


═════════════════════════════════════════════════════════════════════════════
COMMIT STRATEGY
═════════════════════════════════════════════════════════════════════════════

After Each Engine (5 commits total):

Commit 1: [Phase 3] Wire MarketAccountEngine (4 methods)
Commit 2: [Phase 3] Wire SituationEngine (4 methods)
Commit 3: [Phase 3] Wire DecisionEngine (3 methods)
Commit 4: [Phase 3] Wire SafeExecutionEngine (3 methods + FIX #2)
Commit 5: [Phase 3] Wire OperationsEngine (2 methods)

Each commit message should include:
  - What was changed
  - How many methods
  - Any special notes (e.g., FIX #2 implementation)


═════════════════════════════════════════════════════════════════════════════
ESTIMATED TIMELINE
═════════════════════════════════════════════════════════════════════════════

Task                                    Time      Cumulative
─────────────────────────────────────────────────────────────
1. Setup & Review                       5 min     5 min
2. MarketAccountEngine (4 methods)      30 min    35 min
3. SituationEngine (4 methods)          30 min    65 min
4. DecisionEngine (3 methods)           20 min    85 min
5. SafeExecutionEngine (3 + FIX #2)     30 min    115 min
6. OperationsEngine (2 methods)         20 min    135 min
7. Testing & Verification              30 min    165 min
──────────────────────────────────────────────────────────────
TOTAL:                                            ~2.75 hours


═════════════════════════════════════════════════════════════════════════════
SUCCESS CRITERIA
═════════════════════════════════════════════════════════════════════════════

Phase 3 is complete when:
  ✅ All 16 methods have real implementations (not placeholders)
  ✅ All imports are correct and files compile
  ✅ Each method calls appropriate L0-L8 component
  ✅ FIX #2 guard is fully implemented in place_sell_order()
  ✅ All error handling is in place
  ✅ All tests pass
  ✅ Full cycle works: READ→UNDERSTAND→DECIDE→EXECUTE→RECOVER


═════════════════════════════════════════════════════════════════════════════
CRITICAL NOTES
═════════════════════════════════════════════════════════════════════════════

⭐ FIX #2 GUARD (place_sell_order)
   This is CRITICAL for system safety
   1. Prevents duplicate SELL execution on crash recovery
   2. Uses bounded_cache with 5-minute TTL
   3. Must return "ALREADY_FINALIZED" if duplicate detected
   4. DO NOT SKIP - it's essential for production
   5. Test case: test_fix2_duplicate_sell_prevention()

⭐ ERROR HANDLING
   All implementations have try/except blocks
   - If component not found: graceful degradation
   - If call fails: log warning and continue
   - Never raise exception, return error response instead

⭐ COMPONENT DEPENDENCIES
   All components are passed through app_ctx
   - MarketAccountEngine uses: exchange_client, market_data_feed, balance_manager
   - SituationEngine uses: portfolio_manager, signal_manager, signal_fusion, market_regime_detector
   - DecisionEngine uses: arbitration_engine, mode_manager, capital_allocator
   - SafeExecutionEngine uses: execution_manager, bounded_cache (FIX #2)
   - OperationsEngine uses: startup_orchestrator, health_monitor


═════════════════════════════════════════════════════════════════════════════
PHASE 3 READY TO BEGIN
═════════════════════════════════════════════════════════════════════════════

All infrastructure is ready:
  ✅ Implementations provided
  ✅ Examples provided
  ✅ Documentation provided
  ✅ Testing framework provided
  ✅ Estimated time: 2.75 hours

Next Steps:
  1. Open: PHASE_3_WIRING_CHECKLIST.md (start there)
  2. Follow: Step-by-step instructions
  3. Commit: After each engine
  4. Test: Throughout implementation

GO TIME! 🚀
═════════════════════════════════════════════════════════════════════════════
"""
