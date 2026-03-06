# Implementation Checklist: Eliminate Dust Loop

## Pre-Implementation

- [ ] Read EXECUTIVE_SUMMARY_DUST_LOOP.md (this validates all 14 points)
- [ ] Read ARCHITECTURAL_FIX_DUST_LOOP.md (detailed fixes)
- [ ] Read SIGNAL_THRASHING_AMPLIFICATION.md (amplification factor)
- [ ] Create feature branch: `feat/dust-loop-elimination`
- [ ] Set up test environment with 3 scenarios:
  - [ ] Scenario A: System starts with dust (should heal, not bootstrap)
  - [ ] Scenario B: System restarts after partial trade (should NOT re-bootstrap)
  - [ ] Scenario C: Multiple rotations happen (should NOT create dust)

---

## Phase 1: Portfolio State Machine (Day 1, ~2 hours)

### Task 1.1: Create PortfolioState Enum
- [ ] File: `core/shared_state.py`
- [ ] Add enum with 5 states:
  - [ ] `EMPTY_PORTFOLIO`
  - [ ] `PORTFOLIO_WITH_DUST`
  - [ ] `PORTFOLIO_ACTIVE`
  - [ ] `PORTFOLIO_RECOVERING`
  - [ ] `COLD_BOOTSTRAP`

### Task 1.2: Implement get_portfolio_state() Method
- [ ] File: `core/shared_state.py`
- [ ] New method after line 4979
- [ ] Logic:
  - [ ] If `is_cold_bootstrap()`: return `COLD_BOOTSTRAP`
  - [ ] Get all open positions
  - [ ] Filter into significant vs dust positions
  - [ ] If significant positions exist: return `PORTFOLIO_ACTIVE`
  - [ ] If only dust: return `PORTFOLIO_WITH_DUST`
  - [ ] If no positions/dust: return `EMPTY_PORTFOLIO`

### Task 1.3: Implement _is_position_significant() Helper
- [ ] File: `core/shared_state.py`
- [ ] New method
- [ ] Check: `qty * current_price >= PERMANENT_DUST_USDT_THRESHOLD`
- [ ] Default threshold: $1.0

### Task 1.4: Unit Tests
- [ ] File: `test_portfolio_state_machine.py` (new file)
- [ ] Test 1: `test_empty_portfolio_detection()`
- [ ] Test 2: `test_dust_only_portfolio_detection()`
- [ ] Test 3: `test_active_portfolio_detection()`
- [ ] Run: `pytest test_portfolio_state_machine.py -v`

### Task 1.5: Manual Testing
- [ ] Scenario A: Add dust manually, verify state = PORTFOLIO_WITH_DUST
- [ ] Scenario B: Add significant position, verify state = PORTFOLIO_ACTIVE
- [ ] Scenario C: Clear all, verify state = EMPTY_PORTFOLIO

---

## Phase 2: Bootstrap Metrics Persistence (Day 1, ~1 hour)

### Task 2.1: Create BootstrapMetrics Class
- [ ] File: `core/shared_state.py`
- [ ] New class `BootstrapMetrics`
- [ ] Properties:
  - [ ] `db_path`: location to store metrics
  - [ ] `metrics_file`: `{db_path}/bootstrap_metrics.json`

### Task 2.2: Implement Persistence Methods
- [ ] Method: `save_first_trade_at(timestamp: float)` → writes to JSON
- [ ] Method: `get_first_trade_at() -> Optional[float]` → reads from JSON
- [ ] Method: `save_trade_executed()` → increments counter and writes
- [ ] Method: `get_total_trades_executed() -> int` → reads counter
- [ ] Helper: `_load_or_empty() -> Dict` → loads JSON or returns {}
- [ ] Helper: `_write(data: Dict)` → atomically writes JSON

### Task 2.3: Update is_cold_bootstrap()
- [ ] File: `core/shared_state.py` line 4894
- [ ] Change line 4899:
  ```python
  # FROM:
  has_trade_history = (
      self.metrics.get("first_trade_at") is not None
      or self.metrics.get("total_trades_executed", 0) > 0
  )
  
  # TO:
  has_trade_history = (
      self.bootstrap_metrics.get_first_trade_at() is not None
      or self.bootstrap_metrics.get_total_trades_executed() > 0
  )
  ```

### Task 2.4: Persist on First Trade Fill
- [ ] File: `core/execution_manager.py` method `_on_trade_fill()`
- [ ] After successful fill:
  - [ ] `self.bootstrap_metrics.save_first_trade_at(now)` (only once)
  - [ ] `self.bootstrap_metrics.save_trade_executed()` (every fill)

### Task 2.5: Integration Tests
- [ ] File: `test_bootstrap_metrics_persistence.py` (new file)
- [ ] Test 1: `test_metrics_saved_after_trade()`
  - [ ] Simulate trade fill
  - [ ] Verify JSON file created
  - [ ] Verify `first_trade_at` set
- [ ] Test 2: `test_metrics_persist_on_restart()`
  - [ ] Simulate trade fill
  - [ ] Simulate restart (recreate SharedState)
  - [ ] Verify `is_cold_bootstrap()` returns False

### Task 2.6: Manual Testing
- [ ] Start system
- [ ] Execute first trade
- [ ] Check: `bootstrap_metrics.json` file created
- [ ] Restart system
- [ ] Verify: `is_cold_bootstrap()` = False

---

## Phase 3: Dust Registry Lifecycle (Day 2, ~3 hours)

### Task 3.1: Create DustRegistry Class
- [ ] File: `core/shared_state.py` or `core/dust_registry.py` (new file)
- [ ] Class `DustRegistry`
- [ ] Properties:
  - [ ] `dust_entries`: Dict[symbol] → {qty, notional, origin, created_at}
  - [ ] `healing_attempts`: Dict[symbol] → attempt count
  - [ ] `healing_completed`: Dict[symbol] → completion timestamp

### Task 3.2: Implement Lifecycle Methods
- [ ] Method: `record_dust(symbol, qty, origin, context)` → adds to dust_entries
- [ ] Method: `increment_healing_attempt(symbol)` → increments counter, returns count
- [ ] Method: `mark_healing_complete(symbol)` → **CRITICAL**
  - [ ] Clears `dust_healing_deficit[symbol]`
  - [ ] Clears `dust_operation_symbols[symbol]`
  - [ ] Records `healing_completed[symbol]`
  - [ ] Logs completion
- [ ] Method: `mark_permanent_dust(symbol)` → marks as permanent after max attempts
- [ ] Method: `get_is_dust_only(symbol)` → returns if dust is active
- [ ] Method: `is_healing_failed(symbol, max_attempts=3)` → checks attempt count

### Task 3.3: Integrate with _is_dust_operation_context()
- [ ] File: `core/execution_manager.py` line 2513
- [ ] Update to check: `self.shared_state.dust_registry.get_is_dust_only(sym)`
- [ ] Remove direct checks of `dust_healing_deficit` and `dust_operation_symbols`

### Task 3.4: Circuit Breaker Implementation
- [ ] In `execution_manager.py`:
  - [ ] New method: `attempt_dust_healing(symbol, qty, price)` → bool
  - [ ] Check: `is_healing_failed(symbol, max_attempts=3)`
  - [ ] If failed: `mark_permanent_dust(symbol)` and return False
  - [ ] If healing succeeds: `mark_healing_complete(symbol)` and return True
  - [ ] Increment attempts on failure: `increment_healing_attempt(symbol)`

### Task 3.5: Unit Tests
- [ ] File: `test_dust_registry.py` (new file)
- [ ] Test 1: `test_record_dust()`
  - [ ] Add dust entry
  - [ ] Verify in registry
- [ ] Test 2: `test_mark_healing_complete()`
  - [ ] Record dust
  - [ ] Mark healing complete
  - [ ] Verify `get_is_dust_only()` returns False
  - [ ] Verify markers cleared
- [ ] Test 3: `test_circuit_breaker()`
  - [ ] Record dust
  - [ ] Fail 3 healing attempts
  - [ ] Verify `is_healing_failed()` returns True
  - [ ] Attempt healing returns False

### Task 3.6: Integration Test
- [ ] File: `test_dust_healing_lifecycle.py` (new file)
- [ ] Scenario: Dust detected → Healing attempted → Success → Verify not re-detected

---

## Phase 4: Separate Override Flags (Day 2, ~4 hours)

### Task 4.1: Update policy_ctx Structure
- [ ] File: `core/execution_manager.py`
- [ ] Replace:
  ```python
  # OLD: Single flag for both
  "bootstrap_override": bool
  "is_dust_operation": bool
  
  # NEW: Context-specific flags
  "is_dust_healing": bool
  "is_bootstrap_trade": bool
  "is_rotation_trade": bool
  ```

### Task 4.2: Implement Context Validation
- [ ] File: `core/execution_manager.py` method `_place_limit_order()`
- [ ] Add validation at start:
  ```python
  is_dust_healing = bool(policy_ctx.get("is_dust_healing", False))
  is_bootstrap = bool(policy_ctx.get("is_bootstrap_trade", False))
  is_rotation = bool(policy_ctx.get("is_rotation_trade", False))
  
  # Mutually exclusive check
  active_contexts = sum([is_dust_healing, is_bootstrap, is_rotation])
  if active_contexts > 1:
      logger.error("Invalid: multiple contexts")
      return {"ok": False, "reason": "invalid_context"}
  ```

### Task 4.3: Apply Context-Specific Rules
- [ ] For `is_dust_healing`:
  - [ ] Set: `bypass_min_notional = True`
  - [ ] Set: `bypass_risk_sizing = False` ← ENFORCE RISK
  - [ ] Set: `min_pnl_check = False`
- [ ] For `is_bootstrap`:
  - [ ] Set: `bypass_min_notional = False`
  - [ ] Set: `bypass_risk_sizing = True`
  - [ ] Set: `min_pnl_check = False`
- [ ] For `is_rotation`:
  - [ ] Set: `bypass_min_notional = False`
  - [ ] Set: `bypass_risk_sizing = False`
  - [ ] Set: `min_pnl_check = True` ← REQUIRE PROFIT

### Task 4.4: Update All Policy_ctx Callers
- [ ] File: `agents/meta_controller.py`
  - [ ] Dust healing trades: `"is_dust_healing": True`
  - [ ] Bootstrap trades: `"is_bootstrap_trade": True`
  - [ ] Other: clear both flags
- [ ] File: `agents/trend_hunter.py`
  - [ ] Rotation trades: `"is_rotation_trade": True`
- [ ] File: `core/execution_manager.py`
  - [ ] All other orders: clear all 3 flags

### Task 4.5: Unit Tests
- [ ] File: `test_context_flags.py` (new file)
- [ ] Test 1: `test_dust_healing_context_rules()`
- [ ] Test 2: `test_bootstrap_context_rules()`
- [ ] Test 3: `test_rotation_context_rules()`
- [ ] Test 4: `test_multiple_contexts_rejected()`

### Task 4.6: Manual Testing
- [ ] Dust healing trade: should bypass notional only
- [ ] Bootstrap trade: should bypass risk only
- [ ] Rotation trade: should require profit
- [ ] Trade with 2 flags: should be rejected

---

## Phase 5: Trading Coordinator (Day 3, ~6 hours)

### Task 5.1: Create TradingCoordinator Class
- [ ] File: `core/trading_coordinator.py` (new file)
- [ ] Class `TradingCoordinator`
- [ ] Constructor:
  - [ ] `__init__(shared_state, execution_manager)`

### Task 5.2: Implement authorize_trade() Gate
- [ ] Method: `authorize_trade(symbol, side, qty, context)` → Dict
- [ ] Logic:
  - [ ] Get portfolio state
  - [ ] Check if context allowed in state
  - [ ] Validate context-specific constraints
  - [ ] Return {ok, reason, state, context}

### Task 5.3: Implement State-Context Decision Matrix
- [ ] Method: `_is_trade_allowed(context, state)` → bool
- [ ] Matrix (from ARCHITECTURAL_FIX document):

| State | Bootstrap | Strategy | Dust Healing | Rotation |
|-------|-----------|----------|--------------|----------|
| COLD_BOOTSTRAP | ✓ | ✗ | ✗ | ✗ |
| EMPTY_PORTFOLIO | ✓ | ✗ | ✗ | ✗ |
| PORTFOLIO_WITH_DUST | ✗ | ✗ | ✓ | ✗ |
| PORTFOLIO_ACTIVE | ✗ | ✓ | ✗ | ✓ |
| PORTFOLIO_RECOVERING | ✗ | ✗ | ✗ | ✗ |

### Task 5.4: Implement Context Validation
- [ ] Method: `_validate_context(symbol, side, qty, context, state)` → Dict
- [ ] For "dust_healing": verify is_dust_only(symbol)
- [ ] For "bootstrap": verify state in [COLD_BOOTSTRAP, EMPTY_PORTFOLIO]
- [ ] For "rotation": verify state == PORTFOLIO_ACTIVE

### Task 5.5: Update MetaController to Use Gate
- [ ] File: `agents/meta_controller.py`
- [ ] Method: `propose_trade(signal)` → Optional[TradeRequest]
- [ ] Before returning signal for execution:
  ```python
  auth = await self.trading_coordinator.authorize_trade(
      symbol=signal.symbol,
      side="buy",
      qty=signal.size,
      context="bootstrap"  # or "dust_healing", etc.
  )
  if not auth["ok"]:
      logger.debug(f"Trade blocked: {auth['reason']}")
      return None  # Don't execute
  ```

### Task 5.6: Update TrendHunter to Use Gate
- [ ] File: `agents/trend_hunter.py`
- [ ] Method: `should_trade(symbol, signal)` → Optional[Dict]
- [ ] Call gate:
  ```python
  auth = await self.trading_coordinator.authorize_trade(
      symbol=symbol,
      side="buy",
      qty=signal.size,
      context="strategy"  # or "rotation"
  )
  if not auth["ok"]:
      return None
  ```

### Task 5.7: Update DustHealer/LiquidationAgent to Use Gate
- [ ] File: `agents/dust_healer.py` or `agents/liquidation_agent.py`
- [ ] Method: `heal_dust(symbol)` → Optional[Dict]
- [ ] Call gate:
  ```python
  auth = await self.trading_coordinator.authorize_trade(
      symbol=symbol,
      side="sell",
      qty=dust_qty,
      context="dust_healing"
  )
  if not auth["ok"]:
      return None
  ```

### Task 5.8: Integration Tests
- [ ] File: `test_trading_coordinator.py` (new file)
- [ ] Test 1: `test_bootstrap_allowed_when_empty()`
- [ ] Test 2: `test_bootstrap_blocked_when_dust()`
- [ ] Test 3: `test_dust_healing_allowed_when_dust()`
- [ ] Test 4: `test_dust_healing_blocked_when_active()`
- [ ] Test 5: `test_strategy_blocked_when_empty()`
- [ ] Test 6: `test_strategy_allowed_when_active()`

### Task 5.9: Manual End-to-End Test
- [ ] Scenario A: System starts empty → Bootstrap allowed ✓
- [ ] Scenario B: Dust detected → Bootstrap blocked ✓, Healing allowed ✓
- [ ] Scenario C: Healing completes → Bootstrap allowed ✓
- [ ] Scenario D: Portfolio active → Strategy trades allowed ✓, Bootstrap blocked ✓

---

## Phase 6: Dynamic Position Limits (Day 3, ~3 hours)

### Task 6.1: Create DynamicPositionLimits Class
- [ ] File: `core/capital_governor.py` or `core/position_limits.py` (new file)
- [ ] Class `DynamicPositionLimits`
- [ ] NAV tiers:
  - [ ] $0-100: 1 position max
  - [ ] $100-500: 2 positions max
  - [ ] $500-2K: 3 positions max
  - [ ] $2K+: 4 positions max

### Task 6.2: Implement Position Limit Methods
- [ ] Method: `get_max_positions(nav)` → int
- [ ] Method: `can_add_position(symbol, nav, correlation, signal_strength)` → bool
- [ ] Logic for soft gate: high correlation (>0.7) requires signal_strength > 0.75

### Task 6.3: Update MetaController.should_add_position()
- [ ] File: `agents/meta_controller.py`
- [ ] New method or gate:
  ```python
  can_add = await self.position_limits.can_add_position(
      symbol=signal.symbol,
      nav=await self.shared_state.get_nav(),
      proposed_correlation=await self._estimate_correlation(symbol),
      signal_strength=signal.confidence
  )
  if not can_add:
      logger.info(f"Position rejected: {symbol}")
      return None
  ```

### Task 6.4: Update Rotation Logic with Hysteresis
- [ ] File: `agents/meta_controller.py`
- [ ] Method: `should_rotate_out(symbol, new_signal)` → bool
- [ ] Add hysteresis: only rotate if new signal significantly better (>20% stronger)
- [ ] Or if position age > 1 hour

### Task 6.5: Unit Tests
- [ ] File: `test_dynamic_position_limits.py` (new file)
- [ ] Test 1: `test_position_limit_by_nav_tier()`
- [ ] Test 2: `test_correlation_gate()`

### Task 6.6: Manual Testing
- [ ] NAV $80 → Max 1 position
- [ ] NAV $300 → Max 2 positions
- [ ] NAV $1000 → Max 3 positions
- [ ] NAV $5000 → Max 4 positions

---

## Testing: Complete Scenarios

### Scenario A: Dust Healing (Critical Path)
- [ ] Start system with dust in BTCUSDT (0.0001 BTC)
- [ ] System should:
  1. [ ] Detect state = PORTFOLIO_WITH_DUST
  2. [ ] Reject bootstrap trade
  3. [ ] Allow dust healing sell
  4. [ ] Execute sell order
  5. [ ] Clear dust markers
  6. [ ] Detect state = EMPTY_PORTFOLIO
- [ ] Verify no loop occurs

### Scenario B: Bootstrap on Empty (Happy Path)
- [ ] Start fresh system
- [ ] System should:
  1. [ ] Detect state = COLD_BOOTSTRAP
  2. [ ] Allow bootstrap trade
  3. [ ] Execute BUY order
  4. [ ] Persist first_trade_at
  5. [ ] No loop even with restart
- [ ] Restart and verify bootstrap doesn't re-trigger

### Scenario C: Normal Trading (Regression)
- [ ] Start with active position (BTCUSDT 1.0)
- [ ] System should:
  1. [ ] Detect state = PORTFOLIO_ACTIVE
  2. [ ] Allow strategy trades
  3. [ ] Allow rotation trades
  4. [ ] Block bootstrap/dust healing
  5. [ ] Normal trading continues
- [ ] Verify no functional change

### Scenario D: Multi-Position Portfolio
- [ ] NAV > $500, allow 3 concurrent positions
- [ ] Add BTCUSDT, ETHUSDT, BNBUSDT
- [ ] Reduce rotations by 80%
- [ ] Measure capital preservation

---

## Deployment Checklist

### Pre-Deployment
- [ ] All phases complete
- [ ] All unit tests pass (100% pass rate)
- [ ] All integration tests pass (100% pass rate)
- [ ] All scenarios pass
- [ ] Code review completed
- [ ] Lint check passed: `pylint core/ agents/`
- [ ] Type check passed: `mypy core/ agents/`

### Deployment
- [ ] Create release branch: `release/dust-loop-fix-v1.0`
- [ ] Merge to main with PR
- [ ] Tag: `v1.0.0-dust-loop-fix`
- [ ] Deploy to test environment
- [ ] Run 24-hour test with monitoring
- [ ] Check metrics:
  - [ ] Bootstrap triggers: 0 (unless truly empty)
  - [ ] Dust creation rate: <1% daily
  - [ ] Capital preservation: >95% daily
- [ ] Deploy to production

### Post-Deployment
- [ ] Monitor for 1 week
- [ ] Check error logs for regressions
- [ ] Verify all scenarios working
- [ ] Collect metrics on capital preservation

---

## Success Criteria

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Bootstrap triggers/day | 3-5 | 0 | ✅ 0 |
| Dust creation rate | High | Low | ✅ <1%/day |
| Daily capital loss | 6% | 0.1% | ✅ <0.2% |
| System survival time | ~16 days | 1000+ days | ✅ 1000+ |
| Rotation frequency | 48-64/day | 2-4/day | ✅ 80% reduction |

All metrics must be achieved for successful deployment.

---

## Rollback Plan

If critical issues arise:

1. Revert to last stable commit
2. Keep Phase 1 (state machine) if working
3. Keep Phase 2 (metrics) if working
4. Remove Phases 3-5 (registry, flags, coordinator)
5. Verify metrics return to baseline
6. Plan Phase 2 retry

---

## Final Sign-Off

Completion of this checklist guarantees:
- ✅ Dust loop eliminated
- ✅ Bootstrap prevents re-entry
- ✅ All three subsystems coordinated
- ✅ Capital preserved long-term
- ✅ Production-ready system

Estimated timeline: **3 days of focused engineering**

---
