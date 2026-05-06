# Phase 8.2: Native Component Migration Roadmap

**Date:** 2026-05-06  
**Status:** Planning  
**Bridge Status:** ✅ Phase 8.1 Complete (validated, 6/6 tests pass)

---

## Overview

Phase 8.2 replaces the legacy orchestrator components layer-by-layer with native implementations, maintaining production parity while improving performance.

**Current State:**
- Bridge reuses legacy MasterSystemOrchestrator (3,306 lines)
- Production cycle time: ~300-330ms (dominated by I/O + startup overhead)
- Mock mode cycle time: ~1-3ms (reference baseline)
- Real balance: $72-$86 USDT flowing through

**Target State:**
- 100% native components in new architecture
- Cycle time: ~100-150ms (3x improvement)
- Memory: -100MB (remove legacy orchestrator)
- Maintenance: Single architecture source-of-truth

---

## Layer Architecture (L0-L8)

### Layer Profiling Data (from startup logs)

| Layer | Component | Current Role | Status | Migration Priority |
|-------|-----------|--------------|--------|-------------------|
| **L0** | Core Utils, TimeUtils, SharedState | Utilities, central state hub | Legacy | 🟢 **P1** (impacts all layers) |
| **L1** | ExchangeClient, API | Binance connection, authentication | Legacy | 🟡 **P2** (high I/O) |
| **L2** | MarketDataFeed, BalanceCache, PriceFeed | Market data, real-time feeds | Legacy | 🟡 **P2** (streaming data) |
| **L3** | PortfolioManager, ExecutionManager | Position tracking, order execution | Legacy | 🟠 **P3** (business logic) |
| **L4** | Strategy engines (trend, swing, etc) | Signal generation, risk checks | Legacy | 🠤 **P4** (high complexity) |
| **L5** | Recovery, Governance, SafetyOrders | Startup, state recovery | Legacy | 🟠 **P3** (state safety) |
| **L6** | Observability, Tracing | Logging, APM | Legacy | 🟣 **P5** (nice-to-have) |
| **L7** | Advanced features (chaos monkey, etc) | Testing features | Legacy | 🟣 **P5** (optional) |

---

## Migration Strategy: "Adopt-then-Refactor"

### Phase 8.2.1: L0 (Core Utils) — 1-2 weeks

**Scope:**
- Replace `SharedState` with native in-memory state manager
- Replace `TimeUtils` with native time management
- Replace utility modules (config constants, retry manager, etc)

**Expected Outcome:**
- 10-20ms baseline improvement (less legacy code execution)
- Foundation for all other layers
- Bridge becomes thin adapter layer

**Test:**
```bash
# Should show 25/26 components mapped still, but L0 native
python3 main.py --mode=paper-trade --duration=30s --interval=2 --production
# Expected: nav=$72-86, cycle ~280-300ms (10-20ms improvement)
```

### Phase 8.2.2: L1 (Exchange Client) — 2-3 weeks

**Scope:**
- Wrap Binance REST API layer (live with polling)
- Keep exchange_client internal API compatible
- Test with real balance sync

**Expected Outcome:**
- 20-40ms improvement (native REST logic faster than legacy polling)
- Real balance still $72-86
- Orders still execute

**Test:**
```bash
python3 main.py --mode=paper-trade --duration=60s --interval=2 --production
# Expected: nav=$72-86, cycle ~260-280ms (40-60ms improvement cumulative)
# Monitor order execution, balance updates
```

### Phase 8.2.3: L2 (Market Data) — 2-3 weeks

**Scope:**
- Native OHLCV cache (currently legacy list)
- Native balance cache updater
- Price feed aggregation

**Expected Outcome:**
- 30-50ms improvement (streaming data collection faster)
- Indicator warmup still 5-10min (unchanged)

**Test:**
```bash
python3 main.py --mode=paper-trade --duration=5min --interval=2 --production
# Expected: nav=$72-86, cycle ~220-260ms (80-100ms improvement cumulative)
# Signals should start firing after 5min warmup
```

### Phase 8.2.4: L3 (Portfolio + Execution) — 3-4 weeks

**Scope:**
- Replace PortfolioManager (position tracking)
- Replace ExecutionManager (order placement)
- Native order state machine

**Expected Outcome:**
- 20-30ms improvement (business logic execution)
- Real orders placed/canceled
- Signals driving trades

**Test:**
```bash
python3 main.py --mode=paper-trade --duration=15min --interval=2 --production
# Expected: nav=$72-86+, cycle ~200-230ms (100-130ms improvement cumulative)
# Monitor: sigs>0, dec>0, exe>0 (should see real trading signals)
```

### Phase 8.2.5: L4 (Strategy Engines) — 4-6 weeks

**Scope:**
- Trend detection engine (trend signals)
- Swing detection engine (swing signals)
- Risk checks and profit-taking logic

**Expected Outcome:**
- 10-20ms improvement
- Signals now strategy-aware
- Full trading logic working

**Test:**
```bash
python3 main.py --mode=paper-trade --duration=30min --interval=2 --production
# Expected: nav=$72-100+, cycle ~190-220ms (110-140ms improvement cumulative)
# Monitor: P&L growth, strategy signal distribution
```

### Phase 8.2.6: L5 (Recovery + Governance) — 2-3 weeks

**Scope:**
- Replace StartupOrchestrator (state recovery)
- Replace position merger, rebalancer
- Ledger reconciliation

**Expected Outcome:**
- 5-10ms improvement
- Cleaner startup (no TruthAuditor delays)
- Graceful recovery

**Test:**
```bash
# Full production run with clean startup
python3 main.py --mode=paper-trade --duration=60min --interval=2 --production
# Expected: nav=$72-100+, cycle ~185-210ms (120-145ms improvement cumulative)
# Monitor startup duration, recovery correctness
```

### Phase 8.2.7: L6-L8 (Observability + Optional) — 1-2 weeks

**Scope:**
- APM/Tracing (native telemetry)
- Advanced features (chaos monkey, etc) → optional

**Expected Outcome:**
- <5ms improvement
- Full telemetry coverage
- No legacy dependencies

**Test:**
```bash
# Final integration test
python3 main.py --mode=paper-trade --duration=2hours --interval=2 --production
# Expected: nav=$72-120+, cycle ~180-200ms (150+ ms improvement)
# Monitor: P&L, stability, telemetry quality
```

### Phase 8.2.8: Bridge Deprecation — 1 week

**Scope:**
- Remove production_bridge.py
- Remove legacy emoji filename support
- Clean up ATTR_TO_CTX_KEY mapping
- Legacy file no longer needed

**Expected Outcome:**
- Clean codebase
- Single architecture
- No technical debt

---

## Performance Targets

| Milestone | Cycle Time | Improvement | Status |
|-----------|-----------|-------------|--------|
| Phase 8.1 (Current) | ~300ms | - | ✅ Active |
| L0 Native | ~280ms | -20ms | 🔄 Planning |
| L1 Native | ~260ms | -40ms | 🔄 Planning |
| L2 Native | ~220ms | -80ms | 🔄 Planning |
| L3 Native | ~200ms | -100ms | 🔄 Planning |
| L4 Native | ~190ms | -110ms | 🔄 Planning |
| L5 Native | ~185ms | -120ms | 🔄 Planning |
| L6-L8 Native | ~180ms | -150ms ✅ | 🔄 Planning |

---

## Risk Mitigation

### Equivalence Testing
- Before replacing each layer, run 1-hour equivalence test:
  - Legacy bridge version: baseline NAV, signals, trades
  - Native version: should match ±0.1% NAV, ±5% signals
  - Abort migration if variance exceeds thresholds

### Rollback Plan
- Keep bridge in code until L5 complete
- Feature flag `--use-bridge` to switch between implementations
- 2-layer rollback always available (L0 + current work)

### Smoke Test Checklist
After each layer migration:
```bash
✅ Startup completes (no crashes)
✅ NAV syncs from Binance ($72-120)
✅ Positions hydrated correctly (8-12 symbols)
✅ Market data streaming (OHLCV bars arriving)
✅ No error logs in 30-second run
✅ Clean shutdown
✅ Cycle time within -20% of target
```

---

## Success Criteria

| Criterion | Current | Target | Status |
|-----------|---------|--------|--------|
| Cycle time | 300ms | <200ms | 🔄 In Progress |
| Memory (MB) | +300 | +100 | 🔄 In Progress |
| Components (native) | 0/25 | 25/25 | 🔄 In Progress |
| Test coverage | 6/6 | 20+ | 🔄 Planning |
| P&L equivalence | - | ±0.1% | 🔄 Planning |
| Uptime (24h run) | - | 99.5% | 🔄 Planning |
| Production ready | ✅ (bridge) | ✅ (native) | 🔄 Planning |

---

## Implementation Guidance

### Adding Native L0 Components

1. Create `core_engine/native_shared_state.py`:
   ```python
   class NativeSharedState:
       """In-memory state replacement for legacy SharedState"""
       def __init__(self):
           self.nav = 0.0
           self.positions = {}
           self.balance = 0.0
       
       def update_nav(self, value: float):
           self.nav = value
   ```

2. Update `core_engine/integration.py` to opt-in:
   ```python
   if production and migration_phase >= 8.2:
       # Use native L0
       app_ctx["shared_state"] = NativeSharedState()
   else:
       # Use bridge (legacy)
       app_ctx = await build_production_app_ctx()
   ```

3. Test both in parallel:
   ```bash
   # Legacy (current)
   python3 main.py --production
   
   # Native L0
   python3 main.py --production --native-l0
   ```

### Feature Flag Pattern

```python
# In main.py
parser.add_argument("--native-l0", action="store_true", help="Use native L0 (SharedState)")
parser.add_argument("--native-l1", action="store_true", help="Use native L1 (ExchangeClient)")
# ... etc for L2-L8

# In integration.py
if args.native_l0:
    app_ctx["shared_state"] = NativeSharedState()
```

---

## Timeline Estimate

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| 8.2.1 (L0) | 1-2 weeks | 2026-05-06 | 2026-05-20 |
| 8.2.2 (L1) | 2-3 weeks | 2026-05-20 | 2026-06-10 |
| 8.2.3 (L2) | 2-3 weeks | 2026-06-10 | 2026-07-01 |
| 8.2.4 (L3) | 3-4 weeks | 2026-07-01 | 2026-08-01 |
| 8.2.5 (L4) | 4-6 weeks | 2026-08-01 | 2026-09-15 |
| 8.2.6 (L5) | 2-3 weeks | 2026-09-15 | 2026-10-06 |
| 8.2.7 (L6-L8) | 1-2 weeks | 2026-10-06 | 2026-10-20 |
| 8.2.8 (Deprecation) | 1 week | 2026-10-20 | 2026-10-27 |
| **Total** | **~18-25 weeks** | 2026-05-06 | **2026-10-27** |

---

## Checkpoint: After Phase 8.2 Complete

✅ **System State:**
- Zero legacy code in production path
- 25/25 components native and tested
- Cycle time: ~180-200ms (3x improvement)
- Memory: -100MB
- Full telemetry and observability

✅ **Operations Ready:**
- Production trading with real Binance balance
- Strategy signals firing and executing
- P&L compounding (real profit taking)
- 24h+ stable runs validated

✅ **Next Steps (Phase 9):**
- Performance optimization (further reduce cycle time)
- Advanced features (grid trading, DCA, etc)
- Multi-exchange support
- Risk management enhancements

---

## Questions / Decisions Needed

1. **L0-L1 Timing:** Migrate together or separately?
   - Recommendation: Together (L0 affects L1 anyway)

2. **Feature Flag Cleanup:** When to remove bridge code?
   - Recommendation: After L5 complete + 2-week stability run

3. **Testing Approach:** Unit vs Integration vs Equivalence?
   - Recommendation: All three (unit for new code, integration for wiring, equivalence for parity)

4. **Rollback Procedure:** Keep bridge active until what milestone?
   - Recommendation: Until L5 complete, then sunset over 2 weeks

---

## Next Action

**Immediate:**
1. Profile current bridge to identify top 3 slowest functions/layers
2. Create detailed L0 native implementation spec
3. Set up feature flag infrastructure (`--native-l0` CLI option)
4. Begin L0 implementation (target: 2026-05-20)

**By End of Week:**
1. L0 native partially working
2. Equivalence test framework in place
3. Performance baseline measurements documented

---

**Status:** Ready to proceed with Phase 8.2.1 (L0 Native)  
**Owner:** @mauf  
**Last Updated:** 2026-05-06 01:47  
