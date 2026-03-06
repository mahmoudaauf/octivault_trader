# Regime-Based Scaling Implementation Roadmap

## Timeline & Effort Estimate

```
PHASE 1 ✅ [COMPLETE]
├─ Duration: 2 hours (already done)
├─ TrendHunter._get_regime_scaling_factors()
├─ TrendHunter._submit_signal() modifications
├─ Signal emission updates
└─ Status: READY FOR TESTING

         ↓↓↓ SIGNALS NOW CARRY REGIME SCALING ↓↓↓

PHASE 2 ⏭️ [NEXT - 1-2 hours]
├─ MetaController position_size_mult application
├─ Effort: ~10-15 lines of code
├─ Impact: ENABLES position sizing per regime
├─ Tests: Verify BUY signals scale correctly
└─ Status: BLOCKING OTHER PHASES

         ↓↓↓ POSITIONS NOW SIZED BY REGIME ↓↓↓

PHASE 3a ⏭️ [2-3 hours after Phase 2]
├─ TP/SL Engine tp_target_mult application
├─ Effort: ~10-15 lines
├─ Impact: TP targets scale per regime
└─ Status: MEDIUM PRIORITY

PHASE 3b ⏭️ [2-3 hours after Phase 3a]
├─ TP/SL Engine excursion_requirement_mult
├─ Effort: ~10-15 lines
├─ Impact: Excursion gates scale per regime
└─ Status: MEDIUM PRIORITY

         ↓↓↓ TP/SL NOW SCALED BY REGIME ↓↓↓

PHASE 4 ⏭️ [2-3 hours after Phase 3b]
├─ ExecutionManager trail_mult application
├─ Effort: ~10-15 lines
├─ Impact: Trailing stops scale per regime
└─ Status: MEDIUM PRIORITY

         ↓↓↓ TRAILING NOW SCALED BY REGIME ↓↓↓

PHASE 5 ⏭️ [1-2 hours after Phase 4]
├─ Config externalization
├─ Effort: Configuration only
├─ Impact: Easy tuning without code changes
└─ Status: LOW PRIORITY (nice to have)

TESTING & VALIDATION [2-4 days after all phases]
├─ Backtest against historical data
├─ Compare binary gating vs. regime scaling
├─ Measure win rate, profit factor, Sharpe ratio
├─ Live trading validation
└─ Monitor performance

TOTAL EFFORT: ~10-20 hours for all phases
(~50 lines of code total)
```

---

## Dependency Chain

```
                          Phase 5: Configuration
                                    ▲
                                    │
                                    └─ Optional (nice to have)

Phase 4: ExecutionManager
    ▲
    │ (depends on Phase 3b complete)
    │
Phase 3b: Excursion Gate
    ▲
    │ (depends on Phase 3a complete)
    │
Phase 3a: TP Target
    ▲
    │ (depends on Phase 2 complete)
    │
Phase 2: MetaController ◄─── START HERE (BLOCKING)
    ▲
    │ (depends on Phase 1 complete)
    │
Phase 1: TrendHunter ✅ DONE

Legend:
═══> Blocks next phase (critical path)
───> Optional or can be done in parallel
```

---

## Critical Path (Fastest Route to Feature)

```
1. Phase 2: MetaController (Position sizing)      [1-2 hours]
      │
      ├─ Test: Verify position sizes scale
      │
2. Phase 3a + 3b: TP/SL Engine (Parallel)        [2-3 hours]
      │
      ├─ Test: Verify TP and excursion scale
      │
3. Phase 4: ExecutionManager (Trailing)          [1-2 hours]
      │
      ├─ Test: Verify trailing scales
      │
4. Integration Testing (All together)            [2-3 hours]
      │
      ├─ Backtest regime-scaled vs binary gating
      │
5. Live Validation (1-2 days)
      │
      └─ Monitor performance across regimes

TOTAL: 7-13 hours of implementation + testing
```

---

## Work Breakdown Structure (WBS)

```
Regime-Based Scaling Implementation
│
├── Phase 1: TrendHunter [✅ COMPLETE]
│   ├── Method: _get_regime_scaling_factors() [✅]
│   ├── Method: _submit_signal() modifications [✅]
│   ├── Signal emission with scaling [✅]
│   └── Testing [✅]
│
├── Phase 2: MetaController [⏭️ NEXT]
│   ├── Find execute_decision method
│   ├── Extract _regime_scaling from signal
│   ├── Apply position_size_mult to quote_hint
│   ├── Add logging
│   ├── Unit test scaling application
│   └── Integration test with TrendHunter
│
├── Phase 3a: TP/SL Engine - TP Target [⏭️]
│   ├── Find calculate_tp_sl method
│   ├── Extract tp_target_mult from signal
│   ├── Apply to TP distance calculation
│   ├── Add logging
│   ├── Unit test TP scaling
│   └── Integration test
│
├── Phase 3b: TP/SL Engine - Excursion Gate [⏭️]
│   ├── Find _passes_excursion_gate method
│   ├── Extract excursion_requirement_mult
│   ├── Apply to threshold calculation
│   ├── Add logging
│   ├── Unit test excursion scaling
│   └── Integration test
│
├── Phase 4: ExecutionManager - Trailing [⏭️]
│   ├── Find check_orders method
│   ├── Extract trail_mult from position
│   ├── Apply to trailing calculation
│   ├── Add logging
│   ├── Unit test trailing scaling
│   └── Integration test
│
├── Phase 5: Configuration [⏭️]
│   ├── Add TREND_*_MULT_* config vars
│   ├── Update TrendHunter to read config
│   ├── Update integration points
│   ├── Default values
│   └── Testing
│
├── Integration Testing
│   ├── Full signal flow test
│   ├── Cross-regime comparison test
│   ├── Scaling accumulation test
│   ├── Fallback behavior test
│   └── Performance metrics test
│
└── Validation & Monitoring
    ├── Backtest comparison (binary vs scaling)
    ├── Historical win rate by regime
    ├── Profit factor analysis
    ├── Sharpe ratio calculation
    ├── Live performance monitoring
    └── Metrics dashboard
```

---

## Resource Requirements

### Phase 2: MetaController
- **Prerequisite**: Understanding of MetaController signal processing
- **Skills**: Python, AsyncIO, dict manipulation
- **Tools**: Text editor, Python IDE
- **Time**: 1-2 hours
- **Complexity**: Low

### Phase 3a & 3b: TP/SL Engine
- **Prerequisite**: Understanding of TP/SL calculation logic
- **Skills**: Python, ATR/volatility calculations, math
- **Tools**: Text editor, Python IDE
- **Time**: 2-3 hours for both
- **Complexity**: Low-Medium

### Phase 4: ExecutionManager
- **Prerequisite**: Understanding of trailing stop logic
- **Skills**: Python, order management
- **Tools**: Text editor, Python IDE
- **Time**: 1-2 hours
- **Complexity**: Low

### Phase 5: Configuration
- **Prerequisite**: Understanding of config system
- **Skills**: Python config management
- **Tools**: Text editor
- **Time**: 1-2 hours
- **Complexity**: Very Low

### Testing & Validation
- **Prerequisite**: Backtest framework knowledge
- **Skills**: Python, data analysis, statistical methods
- **Tools**: Backtest engine, matplotlib/pandas
- **Time**: 2-4 days
- **Complexity**: Medium

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Signal payload not reaching MetaController | Low | High | Check logging at each step |
| Multiplier values not reasonable | Low | Medium | Backtest before live |
| Breaking existing position sizing | Medium | High | Thorough integration testing |
| Regime data unavailable | Low | Low | Fallback to 1.0x (baseline) |
| Configuration override issues | Low | Low | Test config loading first |
| Performance regression | Medium | High | A/B test binary vs scaling |

---

## Rollback Strategy

### If Problems Arise at Any Phase

```
Immediate (5 minutes):
├─ Set TREND_REGIME_SCALING_ENABLED = False
├─ Multipliers revert to 1.0x (baseline)
└─ System continues with original behavior

Short-term (hours):
├─ Revert just the problematic phase
├─ Keep completed phases active
└─ Debug the issue

Long-term (hours to days):
├─ Fix issue and re-test
├─ Full integration testing
└─ Gradual rollout (test → paper → live)
```

---

## Success Metrics

### During Implementation

```
Phase 2 Success:
✓ Signal carries _regime_scaling dict
✓ MetaController applies position_size_mult
✓ BUY signal in sideways = 50% size
✓ BUY signal in trending = 100% size

Phase 3 Success:
✓ TP targets vary by regime
✓ Excursion gates vary by regime
✓ No change to position sizing (previous phase)

Phase 4 Success:
✓ Trailing multipliers apply
✓ Sideways trades trail tightly
✓ Trending trades trail loosely

Phase 5 Success:
✓ Multipliers externalized
✓ Config changes apply
✓ No code changes needed for tuning
```

### After Implementation

```
Performance Metrics:
├─ Win rate by regime (target: no regression)
├─ Profit factor by regime (target: improve)
├─ Sharpe ratio (target: improve or neutral)
├─ Max drawdown (target: reduce)
├─ Total return (target: improve)
├─ Consistency (target: smoother equity curve)
└─ Risk-adjusted return (target: improve)

Operational Metrics:
├─ Position size distribution (should match regime)
├─ Average TP target by regime (should scale)
├─ Average trailing SL by regime (should scale)
└─ Regime distribution (all regimes represented)
```

---

## Communication Plan

### Status Updates

```
Week 1:
├─ Phase 1 ✅ Complete (already done)
├─ Phase 2 Start: MetaController integration
└─ Checkpoint: Is position sizing working?

Week 2:
├─ Phase 3a & 3b: TP/SL scaling
├─ Phase 4: ExecutionManager trailing
└─ Checkpoint: Is full system integrated?

Week 3:
├─ Phase 5: Configuration
├─ Integration testing & validation
└─ Checkpoint: Ready for backtesting?

Week 4:
├─ Backtest analysis
├─ Live validation
└─ Final approval & monitoring
```

---

## Documentation Updates Needed

```
To be updated after implementation:

├─ ARCHITECTURE.md
│  └─ Add regime-based scaling section
├─ INTEGRATION_GUIDE.md
│  └─ Add regime scaling setup steps
├─ METRICS_GUIDE.md
│  └─ Add regime-specific metrics
├─ CONFIGURATION.md
│  └─ Document all TREND_*_MULT_* variables
└─ RUNBOOK.md
   └─ Add regime scaling troubleshooting
```

---

## Deployment Checklist

### Pre-Deployment (Development)

- [ ] All 5 phases implemented
- [ ] Unit tests for each phase pass
- [ ] Integration tests pass
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Configuration values reasonable

### Testing Environment

- [ ] Backtest results show improvement
- [ ] Profit factor by regime analyzed
- [ ] Max drawdown reduction confirmed
- [ ] No regression in favorable regimes
- [ ] Fallback behavior tested
- [ ] Edge cases handled (missing regime, etc.)

### Staging/Paper Trading

- [ ] Live config loading works
- [ ] Signals emit correct scaling
- [ ] Position sizes match expected
- [ ] TP/SL calculations correct
- [ ] Trailing stops working
- [ ] Performance metrics collected

### Production Rollout

- [ ] Gradual rollout (e.g., 25% of capital first)
- [ ] Monitor closely first 48 hours
- [ ] Verify all metrics as expected
- [ ] No error conditions triggered
- [ ] Rollout to 100% once confirmed
- [ ] Continuous monitoring active

---

## Go/No-Go Decision Criteria

### Go (Proceed to Next Phase)
- Phase tests pass
- No regression from previous phase
- Scaling factors reasonable
- Logging shows correct behavior
- Ready to integrate next component

### No-Go (Pause & Investigate)
- Phase tests fail
- Unexpected behavior in logs
- Multipliers not applied correctly
- Integration conflicts detected
- Need to debug before proceeding

---

## Timeline Gantt Chart (Simplified)

```
Week 1:
  Phase 1 ✅ |████████████| Complete
  Phase 2    |████████|---- In Progress
  
Week 2:
  Phase 3a   |████████|---- In Progress
  Phase 3b   |████████|---- In Progress
  Phase 4    |████|------ Blocked (wait for 3b)
  
Week 3:
  Phase 4    |████████|---- In Progress
  Phase 5    |████|------ Blocked (wait for 4)
  Testing    |██|-------- Blocked (wait for 5)
  
Week 4:
  Testing    |████████████| In Progress
  Live       |████|------ Pending test results

Legend:
████ = Completed work
|--- = Remaining work
---- = Blocked/Waiting
```

---

## Communication Channels

### Daily Standup
- What was done yesterday (which phase)
- What's being done today (which phase/task)
- Blockers (any issues preventing progress)

### Code Review
- Each phase goes through review before merge
- Focus: correctness, performance, logging
- Approval needed before moving to next

### Testing Results
- Backtest comparison (binary vs scaling)
- Metric analysis by regime
- Risk assessment

### Deployment Decision
- All tests passed
- Metrics show improvement
- Risk assessment complete
- Green light for production

---

## Conclusion

**Total Implementation Time**: 10-20 hours
**Critical Path**: Phases 1 → 2 → 3 → 4 → 5
**Effort Per Phase**: 5-50 lines of code
**Impact**: Fundamental improvement in regime-aware positioning

**Status**: Phase 1 ✅ Complete, Phases 2-5 Ready to Start

**Next Action**: Begin Phase 2 integration in MetaController

---

*Created: REGIME_BASED_SCALING_ROADMAP.md*
*Status: Implementation plan ready for execution*
*Owner: Development team*
*Last Updated: [Current Date]*
