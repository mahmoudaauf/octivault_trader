# Legacy System Reusable Components Analysis (2026-05-07)

## Overview

The legacy system (src/, 153 Python files) contains production-grade components that have been battle-tested. This document identifies high-value components that could enhance the native stack.

## Tier 1: High-Priority Components (Ready to Reuse)

### 1. ModelManager (`src/l5_strategy/model_manager.py`)
**Status**: ✅ CRITICAL for MLForecaster
**Purpose**: Load, cache, and manage TensorFlow `.keras` models
**Key Features**:
- Automatic loading of `.keras` model files from disk
- Model caching with mtime-based invalidation
- Graceful handling of incompatible models (quarantine system)
- Support for both `.keras` and legacy `.h5` formats
- Error classification for deserialization failures

**Integration Path**:
```python
# In core_engine/native/ml_forecaster_integration.py
from src.l5_strategy.model_manager import ModelManager

mm = ModelManager(model_dir="models/", cache_size=10)
model = mm.load_model("mlforecaster_BTCUSDT_5m.keras")
predictions = model.predict(features_batch)
```

**Why Important**: Without ModelManager, we can't load the 100+ trained models. This is the blocking dependency for activating MLForecaster.

### 2. ProfitTargetEngine (`src/l4_execution/profit_target_engine.py`)
**Status**: ✅ READY TO ADAPT
**Purpose**: Daily profit target tracking and compounding throttle
**Key Features**:
- Daily NAV target tracking (default 2%)
- Per-cycle risk cap (default 0.5% of NAV)
- Compounding throttle (reinvest fraction of excess profit)
- Grace period at startup (30 min default)
- Global profit guard (prevents over-trading after daily target hit)

**Native Stack Benefit**:
- Add to NativeOrchestrator as `profit_target_guard()`
- Prevents excessive trading once daily target is met
- Enables autonomous compounding (only reinvest profits)

**Integration Code**:
```python
# In core_engine/native/orchestrator.py
from src.l4_execution.profit_target_engine import ProfitTargetEngine

self._pte = ProfitTargetEngine(config=cfg, shared_state=self._shared_state)

# In decision gate:
if not await self._pte.check_global_compliance(context):
    logger.info("Daily profit target met; skipping BUY decisions")
    return []
```

### 3. OpportunityRanker (`src/l5_strategy/opportunity_ranker.py`)
**Status**: ✅ READY TO REUSE
**Purpose**: Multi-factor signal scoring and ranking
**Key Features**:
- Capital-first opportunity scoring (tuned for small accounts)
- Multi-factor weighting: signal_strength (40%), regime_alignment (15%), liquidity (15%), volatility (10%), agent_confidence (10%), market_quality (10%)
- Recommended max positions based on NAV ($0-150 → 1 slot, $151-350 → 2 slots, etc.)
- Rank and prune decision list (keep top N BUYs, preserve all SELLs)

**Native Stack Benefit**:
- Replace flat 5% allocation with intelligent ranking
- Better capital allocation for micro accounts
- Natural position limiting based on account size

**Integration Code**:
```python
# In core_engine/native/decision_engine.py
from src.l5_strategy.opportunity_ranker import OpportunityRanker

ranker = OpportunityRanker(shared_state=shared_state)

# Filter decisions:
max_buys = ranker.recommended_max_positions(nav_usd)
filtered = ranker.rank_and_prune(decisions, max_buys=max_buys)
```

## Tier 2: Medium-Priority Components (Adapt & Reuse)

### 4. ModelTrainer (`src/l5_strategy/model_trainer.py`)
**Status**: ⚠️ ARCHITECTURE-DEPENDENT
**Purpose**: Train/retrain ML models on fresh OHLCV data
**Key Features**:
- Handles model versioning and checkpointing
- Feature engineering pipeline
- Data windowing (train/test split)
- Model evaluation and backtest

**When to Use**:
- If models become stale (expected after 1-2 weeks of trading)
- For online learning (retrain on new data)
- For model drift detection

**Integration Note**: Requires live OHLCV data pipeline (currently only have ticker prices).

### 5. ArbitrationEngine (`src/l5_strategy/arbitration_engine.py`)
**Status**: ⚠️ PARTIALLY USEFUL
**Purpose**: Multi-layer signal validation gates
**Key Features**:
- Symbol validation gate
- Confidence threshold gate
- Regime-specific gates
- Position limit gate
- Capital gate
- Risk gate

**Native Stack Use**:
- Could replace some of NativeDecisionEngine's gating logic
- Multi-stage evaluation with detailed rejection reasons
- Good for signal filtering

### 6. SignalBatcher (`src/l4_execution/signal_batcher.py`)
**Status**: ⚠️ USEFUL FOR RATE LIMITING
**Purpose**: Batch signals to avoid excessive order placement
**Key Features**:
- Groups signals by symbol
- Implements cooldown between orders
- Deduplicates signals
- Respects exchange rate limits

**Native Stack Benefit**:
- Could reduce Binance API rate-limiting issues
- Batch N signals into M orders (N > M)
- Stagger order placement

## Tier 3: Reference & Architectural Patterns

### 7. AgentManager (`src/l5_strategy/agent_manager.py`)
**Pattern**: Centralized agent lifecycle management
**Useful For**: Understanding how legacy system manages MLForecaster, SymbolScreener
**Native Use**: Could inform orchestrator's agent startup sequence

### 8. CapitalVelocityOptimizer (`src/l5_strategy/capital_velocity_optimizer.py`)
**Pattern**: Dynamic allocation based on win rate and draw-down
**Useful For**: Understanding how legacy system adjusts position size
**Native Use**: Could enhance NativeAdaptiveCapitalEngine

### 9. PerformanceEvaluator (`src/l5_strategy/performance_evaluator.py`)
**Pattern**: Compute win rate, Sharpe ratio, drawdown metrics
**Useful For**: Backtesting and live P&L tracking
**Native Use**: Could enhance observability/metrics in NativeTelemetry

## Tier 4: Don't Reuse (Why)

### ❌ ExecutionManager (`src/l4_execution/execution_manager.py`)
- **Why**: Native stack has NativeExecutor which is purpose-built for async/await
- **Size**: 1000+ lines (complex state machine)
- **Effort**: Rewriting cost > value for native stack

### ❌ MetaController (`src/l8_lifecycle/meta_controller.py`)
- **Why**: This is the legacy orchestrator; native stack replaces it entirely
- **Note**: We already extracted useful patterns (signal gating, risk gates)

### ❌ TradeJournal (`src/l0_core/trade_journal.py`)
- **Why**: Native stack has equivalent in NativeTradeJournal
- **Status**: Already ported

## Implementation Priority

### Phase 8.5 (Next 2 weeks)
1. **ModelManager** (CRITICAL) — Enables MLForecaster inference
2. **OpportunityRanker** (HIGH) — Better capital allocation
3. **ProfitTargetEngine** (HIGH) — Daily profit guard + compounding

### Phase 8.6 (Weeks 3-4)
4. **SignalBatcher** (MEDIUM) — Rate limit handling
5. **ModelTrainer** (MEDIUM) — Model retraining on fresh data

### Phase 9.0+ (Future)
6. ArbitrationEngine — Replace/enhance NativeDecisionEngine gates
7. CapitalVelocityOptimizer — Enhance ACE (Adaptive Capital Engine)
8. PerformanceEvaluator — Enhanced metrics/monitoring

## Quick Integration Checklist

### For ModelManager Integration
- [ ] Copy `src/l5_strategy/model_manager.py` → `core_engine/native/model_manager.py`
- [ ] Update imports (no external deps beyond TensorFlow)
- [ ] Add `load_model()` call to MLForecaster bootstrap
- [ ] Test model loading for one symbol
- [ ] Verify inference works end-to-end

### For OpportunityRanker Integration
- [ ] Copy `src/l5_strategy/opportunity_ranker.py` → `core_engine/native/opportunity_ranker.py`
- [ ] Integrate into NativeDecisionEngine decision filtering
- [ ] Test ranking on 10-symbol portfolio
- [ ] Validate max_positions recommendations vs NAV

### For ProfitTargetEngine Integration
- [ ] Copy `src/l4_execution/profit_target_engine.py` → `core_engine/native/profit_target_engine.py`
- [ ] Wire into NativeOrchestrator._phase_decide() as a gate
- [ ] Add config parameters to BootstrapConfig
- [ ] Test daily target tracking

## Code Quality Notes

**Good Practices Observed**:
- Comprehensive error handling with fallback defaults
- Configurability via environment variables
- Minimal external dependencies (graceful degradation)
- Logging at appropriate levels (info for major events, debug for detail)
- Docstrings explaining design choices

**Areas for Improvement**:
- Some components are 1000+ lines (consider breaking down)
- Heavy use of side-effects (prefer pure functions)
- Limited type hints in some files (Python 3.9 compatibility)

## Files to Review

| Component | File | Lines | Reuse Level |
|-----------|------|-------|------------|
| ModelManager | `src/l5_strategy/model_manager.py` | 400+ | ✅ HIGH |
| ProfitTargetEngine | `src/l4_execution/profit_target_engine.py` | 250+ | ✅ HIGH |
| OpportunityRanker | `src/l5_strategy/opportunity_ranker.py` | 150+ | ✅ HIGH |
| SignalBatcher | `src/l4_execution/signal_batcher.py` | 200+ | ⚠️ MEDIUM |
| ModelTrainer | `src/l5_strategy/model_trainer.py` | 600+ | ⚠️ MEDIUM |
| ArbitrationEngine | `src/l5_strategy/arbitration_engine.py` | 250+ | ⚠️ MEDIUM |
| CapitalVelocityOptimizer | `src/l5_strategy/capital_velocity_optimizer.py` | 300+ | ⚠️ REFERENCE |
| PerformanceEvaluator | `src/l5_strategy/performance_evaluator.py` | 200+ | ⚠️ REFERENCE |

## Testing Strategy

1. **Unit Test**: Test component in isolation
   ```python
   from core_engine.native.model_manager import ModelManager
   mm = ModelManager(model_dir="models/")
   model = mm.load_model("mlforecaster_BTCUSDT_5m.keras")
   assert model is not None
   ```

2. **Integration Test**: Test with native orchestrator
   ```python
   # Run 50 cycles with ModelManager enabled
   # Verify signals are produced from MLForecaster
   ```

3. **Performance Test**: Profile latency impact
   ```python
   # Time ModelManager.load_model()
   # Time ModelManager.predict() for batch of 100 features
   ```

## Dependencies Check

| Component | Requires | Status |
|-----------|----------|--------|
| ModelManager | TensorFlow | ✅ Installed |
| ProfitTargetEngine | None | ✅ No deps |
| OpportunityRanker | numpy (light) | ✅ Installed |
| SignalBatcher | None | ✅ No deps |
| ModelTrainer | TensorFlow, sklearn | ⚠️ Requires scikit-learn |
| ArbitrationEngine | None | ✅ No deps |

## Next Steps

1. **Immediate** (this week):
   - Copy ModelManager to native stack
   - Integrate into MLForecaster bootstrap
   - Test end-to-end inference on one symbol

2. **Short-term** (next week):
   - Integrate OpportunityRanker into decision filtering
   - Wire ProfitTargetEngine as decision gate
   - Run 100+ cycles with all three components

3. **Medium-term** (Phase 8.6):
   - Profile and optimize model loading latency
   - Consider model caching strategy
   - Implement SignalBatcher for rate limit handling

---

**Status**: Analysis Complete
**Date**: 2026-05-07
**Author**: Claude Code
**Next Review**: 2026-05-14
