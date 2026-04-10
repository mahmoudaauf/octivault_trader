#!/usr/bin/env python3
"""
PHASE 2D STEP 2: META_CONTROLLER.PY MIGRATION STARTER KIT

This file provides concrete, copy-paste ready code examples
for refactoring meta_controller.py (the largest file with 356 handlers).

Start here to get hands-on with the migration patterns.
"""

# ============================================================================
# PART 1: IMPORT BLOCK FOR meta_controller.py
# ============================================================================

"""
ADD THIS TO TOP OF meta_controller.py:

(Replace the existing error handling imports with this block)
"""

# Error handling imports
from core.error_types import (
    # Base exception
    TraderException,
    
    # Bootstrap errors
    BootstrapError,
    BootstrapTimeoutError,
    BootstrapValidationError,
    BootstrapResourceError,
    
    # Arbitration errors
    ArbitrationError,
    GateValidationError,
    SignalValidationError,
    ConfidenceThresholdError,
    
    # Lifecycle errors
    LifecycleError,
    StateTransitionError,
    SymbolNotReadyError,
    SymbolLockError,
    
    # Execution errors
    ExecutionError,
    OrderPlacementError,
    BalanceError,
    NotionalError,
    ExecutionValidationError,
    DuplicateOrderError,
    
    # Exchange errors
    ExchangeError,
    ExchangeAPIError,
    ExchangeRateLimitError,
    ExchangeAuthError,
    InvalidPairError,
    LiquidityError,
    
    # State errors
    StateError,
    StateSyncError,
    StateLockError,
    StateCorruptionError,
    StateConsistencyError,
    
    # Network errors
    NetworkError,
    NetworkTimeoutError,
    ConnectionRefusedError,
    ConnectionResetError,
    DNSError,
    
    # Validation errors
    ValidationError,
    InvalidParameterError,
    MissingFieldError,
    TypeMismatchError,
    RangeError,
    
    # Configuration & Resource errors
    ConfigurationError,
    ConfigurationInvalidError,
    ConfigurationMissingError,
    ResourceError,
    ResourceLimitError,
    
    # Error context & enums
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
    ErrorRecovery,
    create_error_context,
)

from core.error_handler import (
    get_error_handler,
    ErrorClassification,
)


# ============================================================================
# PART 2: ACTUAL MIGRATION EXAMPLES FOR meta_controller.py
# ============================================================================

"""
These are REAL patterns you'll find in meta_controller.py.
Each shows BEFORE (current anti-pattern) and AFTER (fixed).

Copy these and adapt to the actual code in your file.
"""


# ============================================================================
# EXAMPLE 1: Bootstrap Orchestration Error (Very Common)
# ============================================================================

EXAMPLE_1_BEFORE = """
async def orchestrate_bootstrap(self):
    try:
        logger.info("Starting bootstrap orchestration")
        await self._bootstrap_exchange_connection()
        await self._bootstrap_models()
        await self._bootstrap_arbitration()
        logger.info("Bootstrap complete")
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        return False
"""

EXAMPLE_1_AFTER = """
async def orchestrate_bootstrap(self):
    handler = get_error_handler()
    
    try:
        logger.info("Starting bootstrap orchestration")
        await self._bootstrap_exchange_connection()
        await self._bootstrap_models()
        await self._bootstrap_arbitration()
        logger.info("Bootstrap complete")
        return True
    except BootstrapTimeoutError as e:
        # Timeout: retry with backoff
        classification = handler.handle_exception(
            e,
            additional_context={"operation": "bootstrap_orchestration", "stage": "connection"}
        )
        if classification.should_retry():
            delay = handler.record_recovery_attempt(e, ErrorRecovery.RETRY)
            logger.warning(f"Bootstrap timeout, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            return await self.orchestrate_bootstrap()
        else:
            logger.critical("Bootstrap timeout after retries, cannot proceed")
            return False
    except BootstrapValidationError as e:
        # Validation error: escalate to operator
        logger.error(f"Bootstrap validation failed: {e.context.message}")
        return False
    except BootstrapError as e:
        # Generic bootstrap error
        logger.error(f"Bootstrap error: {e.context.error_code}")
        return False
    except TraderException as e:
        # Other trader errors during bootstrap
        logger.critical(f"Unexpected error during bootstrap: {e.context.error_code}")
        return False
    except Exception as e:
        # Non-trader exceptions
        logger.exception(f"Unexpected error during bootstrap: {type(e).__name__}")
        return False
"""

WHY_BETTER_1 = """
✓ Specific exception types instead of broad Exception
✓ Bootstrap timeout gets automatic retry with exponential backoff
✓ Different bootstrap errors handled differently
✓ Clear context logging (operation, stage)
✓ Non-trader exceptions are explicitly caught (not silently swallowed)
"""


# ============================================================================
# EXAMPLE 2: State Lifecycle Error (Very Common in meta_controller.py)
# ============================================================================

EXAMPLE_2_BEFORE = """
async def transition_symbol_lifecycle(self, symbol: str, new_state: str):
    try:
        logger.info(f"Transitioning {symbol} to {new_state}")
        self.state_manager.validate_transition(symbol, new_state)
        await self.state_manager.set_state(symbol, new_state)
        logger.info(f"Transition complete: {symbol} → {new_state}")
        return True
    except Exception as e:
        logger.error(f"Transition failed: {e}")
        return False
"""

EXAMPLE_2_AFTER = """
async def transition_symbol_lifecycle(self, symbol: str, new_state: str):
    handler = get_error_handler()
    
    try:
        logger.info(f"Transitioning {symbol} to {new_state}")
        self.state_manager.validate_transition(symbol, new_state)
        await self.state_manager.set_state(symbol, new_state)
        logger.info(f"Transition complete: {symbol} → {new_state}")
        return True
    except StateTransitionError as e:
        # Invalid state transition
        classification = handler.handle_exception(
            e,
            additional_context={
                "symbol": symbol,
                "new_state": new_state,
                "operation": "transition_symbol_lifecycle",
            }
        )
        logger.error(f"Invalid state transition for {symbol}: {e.context.message}")
        return False
    except SymbolLockError as e:
        # Symbol is locked by another operation
        classification = handler.handle_exception(
            e,
            additional_context={"symbol": symbol}
        )
        # Try again after brief delay
        if classification.should_retry():
            delay = handler.record_recovery_attempt(e, ErrorRecovery.RETRY)
            logger.warning(f"Symbol {symbol} locked, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            return await self.transition_symbol_lifecycle(symbol, new_state)
        else:
            logger.error(f"Cannot acquire lock for {symbol}")
            return False
    except LifecycleError as e:
        # Generic lifecycle error
        logger.error(f"Lifecycle error for {symbol}: {e.context.error_code}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error transitioning {symbol}: {type(e).__name__}")
        return False
"""

WHY_BETTER_2 = """
✓ State transition errors caught specifically
✓ Lock errors get automatic retry
✓ Rich context (symbol, new_state, operation)
✓ Different error types handled appropriately
✓ No silent failures - all paths logged
"""


# ============================================================================
# EXAMPLE 3: Signal Arbitration Error (Very Common)
# ============================================================================

EXAMPLE_3_BEFORE = """
async def arbitrate_signals(self, signals: List[Signal]) -> Optional[List[Signal]]:
    try:
        logger.info(f"Arbitrating {len(signals)} signals")
        
        # Validate gates
        for signal in signals:
            result = await self.arbitration_engine.validate_gates(signal)
            if not result:
                logger.warning(f"Gate validation failed for {signal.symbol}")
                continue
        
        # Filter by confidence
        filtered = [s for s in signals if s.confidence > self.min_confidence]
        logger.info(f"Filtered to {len(filtered)} signals")
        
        return filtered
    except Exception as e:
        logger.error(f"Arbitration failed: {e}")
        return None
"""

EXAMPLE_3_AFTER = """
async def arbitrate_signals(self, signals: List[Signal]) -> Optional[List[Signal]]:
    handler = get_error_handler()
    
    try:
        logger.info(f"Arbitrating {len(signals)} signals")
        
        # Validate gates
        validated_signals = []
        for signal in signals:
            try:
                result = await self.arbitration_engine.validate_gates(signal)
                if not result:
                    logger.warning(f"Gate validation failed for {signal.symbol}")
                    continue
                validated_signals.append(signal)
            except GateValidationError as e:
                classification = handler.handle_exception(
                    e,
                    additional_context={"symbol": signal.symbol, "operation": "validate_gates"}
                )
                logger.warning(f"Gate validation error for {signal.symbol}: {e.context.message}")
                continue
            except ArbitrationError as e:
                logger.error(f"Arbitration error: {e.context.error_code}")
                continue
        
        # Filter by confidence
        filtered = [s for s in validated_signals if s.confidence > self.min_confidence]
        
        # Check confidence threshold
        if len(filtered) < len(validated_signals):
            skipped = len(validated_signals) - len(filtered)
            logger.info(f"Confidence filter: kept {len(filtered)}, skipped {skipped} below threshold")
        
        logger.info(f"Arbitration complete: {len(filtered)} valid signals")
        return filtered if filtered else None
    except ArbitrationError as e:
        classification = handler.handle_exception(e)
        if classification.is_critical:
            logger.critical(f"Critical arbitration error: {e.context.error_code}")
            raise
        logger.error(f"Arbitration error: {e.context.message}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during arbitration: {type(e).__name__}")
        return None
"""

WHY_BETTER_3 = """
✓ Gate validation errors handled per-signal (no silent failures)
✓ Each error type logged with specific information
✓ Confidence threshold checks explicit (not silent filtering)
✓ Different arbitration errors have different severity levels
✓ Caller gets clear result (None only if completely failed)
"""


# ============================================================================
# EXAMPLE 4: Model Execution Error (Common retry scenario)
# ============================================================================

EXAMPLE_4_BEFORE = """
async def execute_model_predictions(self, symbol: str, data: Dict) -> Optional[float]:
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Running model for {symbol}, attempt {attempt + 1}/{max_attempts}")
            prediction = await self.model_manager.predict(symbol, data)
            logger.info(f"Model prediction for {symbol}: {prediction}")
            return prediction
        except Exception as e:
            if attempt < max_attempts - 1:
                delay = 2 ** attempt + random.random()
                logger.warning(f"Model error, retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Model failed after {max_attempts} attempts: {e}")
                return None
"""

EXAMPLE_4_AFTER = """
async def execute_model_predictions(self, symbol: str, data: Dict) -> Optional[float]:
    handler = get_error_handler()
    
    try:
        logger.info(f"Running model for {symbol}")
        prediction = await self.model_manager.predict(symbol, data)
        logger.info(f"Model prediction for {symbol}: {prediction}")
        return prediction
    except ExecutionError as e:
        # Execution error: may be retryable
        classification = handler.handle_exception(
            e,
            additional_context={
                "symbol": symbol,
                "operation": "execute_model_predictions",
                "data_size": len(data),
            }
        )
        
        if classification.should_retry():
            delay = handler.record_recovery_attempt(e, ErrorRecovery.RETRY)
            logger.warning(f"Model error for {symbol}, retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
            return await self.execute_model_predictions(symbol, data)
        else:
            logger.error(f"Model error for {symbol}, not retryable: {e.context.message}")
            return None
    except ResourceError as e:
        # Resource exhaustion: use fallback
        classification = handler.handle_exception(
            e,
            additional_context={"symbol": symbol, "operation": "execute_model_predictions"}
        )
        
        if classification.recovery_action == ErrorRecovery.FALLBACK:
            logger.warning(f"Resource exhausted for {symbol}, using fallback model")
            return await self.fallback_model_prediction(symbol, data)
        else:
            logger.error(f"Resource error for {symbol}: {e.context.message}")
            return None
    except Exception as e:
        logger.exception(f"Unexpected error in model execution: {type(e).__name__}")
        return None
"""

WHY_BETTER_4 = """
✓ Manual retry loop replaced with automated recovery system
✓ Automatic exponential backoff with jitter (handled by framework)
✓ Different error types (Execution vs Resource) handled differently
✓ Fallback path available for resource exhaustion
✓ Clean, readable code without manual retry logic
"""


# ============================================================================
# EXAMPLE 5: Multiple Exception Paths (Complex orchestration)
# ============================================================================

EXAMPLE_5_BEFORE = """
async def orchestrate_trading_cycle(self):
    try:
        # Bootstrap
        try:
            await self._ensure_bootstrap()
        except Exception as e:
            logger.error(f"Bootstrap error: {e}")
            return False
        
        # Collect signals
        try:
            signals = await self._collect_market_signals()
        except Exception as e:
            logger.error(f"Signal collection error: {e}")
            return False
        
        # Arbitrate signals
        try:
            valid_signals = await self._arbitrate_signals(signals)
        except Exception as e:
            logger.error(f"Arbitration error: {e}")
            return False
        
        # Execute trades
        try:
            results = await self._execute_trades(valid_signals)
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
        
        logger.info("Trading cycle complete")
        return True
    except Exception as e:
        logger.error(f"Unexpected error in trading cycle: {e}")
        return False
"""

EXAMPLE_5_AFTER = """
async def orchestrate_trading_cycle(self):
    handler = get_error_handler()
    
    try:
        logger.info("Starting trading cycle orchestration")
        
        # Bootstrap
        try:
            logger.info("Phase 1: Bootstrap")
            await self._ensure_bootstrap()
        except BootstrapError as e:
            classification = handler.handle_exception(
                e,
                additional_context={"phase": "bootstrap"}
            )
            if classification.is_critical:
                logger.critical("Bootstrap failed critically, cannot proceed")
                return False
            logger.error(f"Bootstrap error: {e.context.error_code}")
            return False
        
        # Collect signals
        try:
            logger.info("Phase 2: Collect signals")
            signals = await self._collect_market_signals()
            logger.info(f"Collected {len(signals)} market signals")
        except NetworkError as e:
            classification = handler.handle_exception(
                e,
                additional_context={"phase": "signal_collection"}
            )
            if classification.should_retry():
                delay = handler.record_recovery_attempt(e, ErrorRecovery.RETRY)
                logger.warning(f"Network error collecting signals, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                return await self.orchestrate_trading_cycle()
            logger.error(f"Cannot collect signals: {e.context.message}")
            return False
        
        # Arbitrate signals
        try:
            logger.info("Phase 3: Arbitrate signals")
            valid_signals = await self._arbitrate_signals(signals)
            if not valid_signals:
                logger.warning("No signals passed arbitration, skipping execution")
                return True  # Not an error, just no opportunities
        except ArbitrationError as e:
            classification = handler.handle_exception(
                e,
                additional_context={"phase": "arbitration", "signal_count": len(signals)}
            )
            logger.error(f"Arbitration error: {e.context.error_code}")
            return False
        
        # Execute trades
        try:
            logger.info("Phase 4: Execute trades")
            results = await self._execute_trades(valid_signals)
            logger.info(f"Executed {len(results)} trades")
        except ExecutionError as e:
            classification = handler.handle_exception(
                e,
                additional_context={"phase": "execution", "signal_count": len(valid_signals)}
            )
            
            if classification.should_retry():
                delay = handler.record_recovery_attempt(e, ErrorRecovery.RETRY)
                logger.warning(f"Execution error, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                return await self.orchestrate_trading_cycle()
            else:
                logger.error(f"Cannot execute trades: {e.context.message}")
                return False
        
        logger.info("Trading cycle complete successfully")
        return True
    except TraderException as e:
        classification = handler.handle_exception(e)
        logger.critical(f"Unexpected trader exception in trading cycle: {e.context.error_code}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in trading cycle: {type(e).__name__}")
        return False
"""

WHY_BETTER_5 = """
✓ Each phase has explicit error handling and recovery
✓ Bootstrap errors are critical (immediate failure)
✓ Network errors can be retried with backoff
✓ Arbitration errors are logged but handled
✓ Execution errors can trigger retry of entire cycle
✓ Each error includes phase and context
✓ Clear logging of each orchestration phase
✓ No silent failures - all paths are explicit
"""


# ============================================================================
# PART 3: MIGRATION CHECKLIST FOR meta_controller.py
# ============================================================================

"""
STEP-BY-STEP CHECKLIST FOR meta_controller.py
===============================================

Before Starting:
  □ Create backup branch: git checkout -b refactor/meta-controller
  □ Verify framework works: pytest tests/test_error_types.py tests/test_error_handler.py -v
  □ Have imports ready (see PART 1 above)

Phase 1: Identify All Handlers (30 min)
  □ Run: grep -n "except Exception" core/meta_controller.py
  □ Run: grep -n "except " core/meta_controller.py | wc -l
  □ Document count in commit messages
  □ Note which exceptions are used most

Phase 2: Import Block (10 min)
  □ Add import block (PART 1) at top of file
  □ Run: mypy core/meta_controller.py --strict
  □ Fix any import issues

Phase 3: Bootstrap Methods (1-2 hours)
  Target: orchestrate_bootstrap(), _bootstrap_exchange_connection(), etc.
  □ Identify: ~30-50 bootstrap-related exception handlers
  □ Refactor: Use EXAMPLE_1 as template
  □ Apply: BootstrapError, BootstrapTimeoutError, BootstrapValidationError
  □ Test: pytest tests/test_meta_controller.py::TestBootstrap -v
  □ Commit: "Refactor: Fix bootstrap exception handlers (30 handlers → typed)"

Phase 4: Lifecycle Methods (1-2 hours)
  Target: transition_symbol_lifecycle(), _ensure_symbol_ready(), etc.
  □ Identify: ~40-60 lifecycle-related exception handlers
  □ Refactor: Use EXAMPLE_2 as template
  □ Apply: StateTransitionError, SymbolLockError, LifecycleError
  □ Test: pytest tests/test_meta_controller.py::TestLifecycle -v
  □ Commit: "Refactor: Fix lifecycle exception handlers (45 handlers → typed)"

Phase 5: Arbitration Methods (1-2 hours)
  Target: arbitrate_signals(), _validate_gates(), etc.
  □ Identify: ~30-40 arbitration-related exception handlers
  □ Refactor: Use EXAMPLE_3 as template
  □ Apply: ArbitrationError, GateValidationError, SignalValidationError
  □ Test: pytest tests/test_meta_controller.py::TestArbitration -v
  □ Commit: "Refactor: Fix arbitration exception handlers (35 handlers → typed)"

Phase 6: Execution Methods (1-2 hours)
  Target: execute_model_predictions(), orchestrate_trading_cycle(), etc.
  □ Identify: ~60-80 execution-related exception handlers
  □ Refactor: Use EXAMPLE_4 and EXAMPLE_5 as templates
  □ Apply: ExecutionError, OrderPlacementError, BalanceError
  □ Test: pytest tests/test_meta_controller.py::TestExecution -v
  □ Commit: "Refactor: Fix execution exception handlers (70 handlers → typed)"

Phase 7: Remaining Handlers (1-2 hours)
  Target: Any remaining broad Exception handlers
  □ Identify: ~70-100 remaining handlers
  □ Classify: Which error type for each?
  □ Refactor: Apply appropriate pattern
  □ Test: pytest tests/test_meta_controller.py -v
  □ Commit: "Refactor: Fix remaining exception handlers (85 handlers → typed)"

Phase 8: Full Verification (1 hour)
  □ Run full test suite: pytest tests/ -v
  □ Type check: mypy core/meta_controller.py --strict
  □ Verify handlers: grep "except Exception" core/meta_controller.py → 0 results
  □ Review diffs: git diff HEAD~7 core/meta_controller.py
  □ Manual testing: Test key workflows

Final Commit:
  git commit --amend -m "Refactor: Complete meta_controller.py exception migration

  - Migrated 356 broad Exception handlers to typed errors
  - Added 7 recovery strategies (RETRY, FALLBACK, SKIP, etc)
  - Improved error context logging (operation, component, symbol)
  - All 45 test cases passing
  - Zero broad Exception handlers remaining
  - Type checking passes (mypy --strict)"
"""


# ============================================================================
# PART 4: HOW TO HANDLE TRICKY PATTERNS
# ============================================================================

"""
HANDLING TRICKY PATTERNS IN meta_controller.py
=================================================

Pattern A: Nested Try-Except Blocks
------------------------------------

BEFORE:
  try:
      try:
          result = await operation1()
      except Exception as e1:
          logger.error(f"Op1 failed: {e1}")
          return None
      
      try:
          result = await operation2(result)
      except Exception as e2:
          logger.error(f"Op2 failed: {e2}")
          return None
  except Exception as e:
      logger.error(f"Unknown error: {e}")
      return False

AFTER:
  handler = get_error_handler()
  
  try:
      try:
          result = await operation1()
      except ExecutionError as e:
          classification = handler.handle_exception(e, additional_context={"step": 1})
          logger.error(f"Operation 1 failed: {e.context.message}")
          return None
      
      try:
          result = await operation2(result)
      except ExecutionError as e:
          classification = handler.handle_exception(e, additional_context={"step": 2})
          logger.error(f"Operation 2 failed: {e.context.message}")
          return None
  except TraderException as e:
      classification = handler.handle_exception(e)
      logger.critical(f"Unexpected trader error: {e.context.error_code}")
      return False
  except Exception as e:
      logger.exception(f"Unexpected error: {type(e).__name__}")
      return False


Pattern B: Exception in Exception Handler
-------------------------------------------

BEFORE:
  try:
      result = await operation()
  except Exception as e:
      try:
          await cleanup(e)
      except Exception as cleanup_error:
          logger.error(f"Cleanup failed: {cleanup_error}")
      raise

AFTER:
  handler = get_error_handler()
  
  try:
      result = await operation()
  except TraderException as e:
      classification = handler.handle_exception(e)
      logger.error(f"Operation failed: {e.context.error_code}")
      
      try:
          await cleanup(e)
      except TraderException as cleanup_error:
          cleanup_classification = handler.handle_exception(cleanup_error)
          logger.error(f"Cleanup failed: {cleanup_error.context.error_code}")
      except Exception as cleanup_error:
          logger.exception(f"Cleanup failed: {type(cleanup_error).__name__}")
      
      raise


Pattern C: Exception in Loop
-----------------------------

BEFORE:
  for item in items:
      try:
          await process(item)
      except Exception as e:
          logger.error(f"Failed to process {item}: {e}")
          continue

AFTER:
  handler = get_error_handler()
  
  for item in items:
      try:
          await process(item)
      except ExecutionError as e:
          classification = handler.handle_exception(
              e,
              additional_context={"item": item}
          )
          logger.warning(f"Failed to process {item}: {e.context.message}")
          continue
      except TraderException as e:
          classification = handler.handle_exception(e)
          if classification.is_critical:
              logger.critical(f"Critical error processing {item}: {e.context.error_code}")
              break  # Exit loop on critical error
          logger.error(f"Error processing {item}: {e.context.message}")
          continue
      except Exception as e:
          logger.exception(f"Unexpected error processing {item}: {type(e).__name__}")
          continue


Pattern D: Conditional Exception Handling
--------------------------------------------

BEFORE:
  try:
      result = await operation()
  except Exception as e:
      if some_condition:
          logger.error(f"Error with condition: {e}")
          return None
      else:
          logger.warning(f"Error without condition: {e}")
          return False

AFTER:
  handler = get_error_handler()
  
  try:
      result = await operation()
  except ExecutionError as e:
      classification = handler.handle_exception(
          e,
          additional_context={"some_condition": some_condition}
      )
      
      if some_condition:
          logger.error(f"Execution error with condition: {e.context.message}")
          return None
      else:
          logger.warning(f"Execution error without condition: {e.context.message}")
          return False
"""


# ============================================================================
# PART 5: TESTING YOUR CHANGES
# ============================================================================

"""
TESTING COMMANDS FOR meta_controller.py REFACTORING
====================================================

After each migration phase, run these commands:

Type Checking:
  mypy core/meta_controller.py --strict

Bootstrap Tests:
  pytest tests/test_meta_controller.py::TestBootstrap -v

Lifecycle Tests:
  pytest tests/test_meta_controller.py::TestLifecycle -v

Arbitration Tests:
  pytest tests/test_meta_controller.py::TestArbitration -v

Execution Tests:
  pytest tests/test_meta_controller.py::TestExecution -v

All meta_controller tests:
  pytest tests/test_meta_controller.py -v

Integration tests:
  pytest tests/integration/ -v

Full test suite:
  pytest tests/ -v

Check for remaining broad exceptions:
  grep -n "except Exception" core/meta_controller.py

Count remaining handlers:
  grep -n "except " core/meta_controller.py | wc -l

Show diff from start:
  git diff HEAD~X core/meta_controller.py

Create summary:
  echo "Migrated X handlers from meta_controller.py"
  echo "Tests passing: $(pytest tests/test_meta_controller.py -q | tail -1)"
"""


# ============================================================================
# CONCLUSION
# ============================================================================

"""
YOU ARE READY TO BEGIN!

1. Copy the import block from PART 1
2. Choose your first method (suggest orchestrate_bootstrap)
3. Use EXAMPLE_1 as template
4. Commit after every 50-100 handlers
5. Run tests frequently
6. When done, move to app_context.py

Remember:
  ✓ Specific exception types, not Exception
  ✓ Always provide error context (operation, component, symbol)
  ✓ Use recovery strategies (RETRY, FALLBACK, SKIP, ESCALATE)
  ✓ Log everything - no silent failures
  ✓ Test after each phase

You've got this! 🚀
"""
