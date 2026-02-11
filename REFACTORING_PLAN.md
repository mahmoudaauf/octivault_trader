# Refactoring Plan for MetaController

## Current Issues
- 11K+ lines in a single file
- Mixed responsibilities (orchestration, policy, mode management, safety)
- Inline class definitions
- Hard to test and maintain

## Proposed Structure
1. Extract inline classes to `core/types.py` or separate files
2. Split MetaController into manager classes based on sections
3. Use composition in MetaController
4. Add unit tests and type checking

## Extraction Plan
- `core/meta_types.py`: ExecutionError, DustState, LiquidityPlan, etc.
- `core/meta_cache.py`: BoundedCache, ThreadSafeIntentSink
- `core/meta_lifecycle.py`: Lifecycle methods
- `core/meta_mode_manager.py`: Mode management
- `core/meta_signal_handler.py`: Signal intake
- `core/meta_dust_logic.py`: Dust and emergency logic
- `core/meta_execution.py`: Execution dispatch
- `core/meta_metrics.py`: Metrics and KPIs

## MetaController becomes orchestrator
- Imports managers
- Delegates calls
- Manages high-level flow

## Next Steps
1. Fix syntax errors
2. Extract classes
3. Extract managers
4. Update imports
5. Test incrementally