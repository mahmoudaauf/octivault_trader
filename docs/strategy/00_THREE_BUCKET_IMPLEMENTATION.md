# 🎯 THREE-BUCKET IMPLEMENTATION GUIDE
## Complete Integration Instructions

**Date**: April 17, 2026  
**Status**: ✅ **READY TO INTEGRATE**

---

## FILES CREATED ✅

### 1. Core Data Structures
**File**: `core/portfolio_buckets.py`
- `BucketType` enum (OPERATING_CASH, PRODUCTIVE, DEAD_CAPITAL)
- `DeadPositionReason` enum (classification reasons)
- `PositionClassification` dataclass (single position)
- `PortfolioBucketState` dataclass (entire portfolio state)
- `HealingEvent` dataclass (healing operation record)
- `HealingReport` dataclass (healing summary)
- `BucketMetrics` dataclass (metrics for monitoring)

### 2. Bucket Classifier
**File**: `core/bucket_classifier.py`
- `BucketClassifier` class (position classification logic)
  - `classify_position()` - Classify single position
  - `classify_portfolio()` - Classify entire portfolio
  - Rules-based classification (5 dead position rules)

### 3. Dead Capital Healer
**File**: `core/dead_capital_healer.py`
- `DeadCapitalHealer` class (liquidation engine)
  - `identify_liquidation_candidates()` - Find dead positions
  - `create_liquidation_orders()` - Generate orders
  - `execute_liquidation_batch()` - Execute liquidations
- `HealingOrchestrator` class (healing coordinator)
  - `plan_healing_cycle()` - Decide if/what to heal

### 4. Integration Manager
**File**: `core/three_bucket_manager.py`
- `ThreeBucketPortfolioManager` class (main API)
  - `update_bucket_state()` - Classify portfolio
  - `should_execute_healing()` - Check if heal needed
  - `plan_healing_cycle()` - Plan healing
  - `execute_healing()` - Execute healing
  - `can_trade_new_position()` - Trading gate
  - `get_trading_decision_gates()` - Get all gates
  - `get_bucket_metrics()` - Get metrics
  - `log_bucket_status()` - Log summary
  - `log_trading_gates()` - Log gates

---

## INTEGRATION STEPS

### Step 1: Add to Master Orchestrator (🎯_MASTER_SYSTEM_ORCHESTRATOR.py)

**1.1 Add imports at top:**

```python
# Around line 80, with other core imports:
from core.three_bucket_manager import ThreeBucketPortfolioManager
```

**1.2 Initialize in OrchestratorConfig (around line 330):**

```python
class OrchestratorConfig:
    # ... existing code ...
    
    # Add after existing capital settings (around line 240):
    def __init__(self):
        # ... existing init code ...
        
        # Three-bucket configuration
        self.three_bucket_config = {
            'min_productive_size': 25.0,           # $25 minimum
            'stale_days_threshold': 7,              # 7 days max stale
            'performance_threshold': -15.0,         # -15% performance
            'max_productive_positions': 5,          # Max 5 positions
            'min_dead_to_heal': 50.0,              # Heal if > $50 dust
            'batch_heal_enabled': True,             # Batch liquidations
            'max_liquidations': 10,                 # Max 10 per cycle
        }
```

**1.3 Initialize in MainOrchestrator.__init__ (around line 370):**

```python
class MainOrchestrator:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add after other managers:
        self.three_bucket_manager = ThreeBucketPortfolioManager(
            self.config.three_bucket_config
        )
        logger.info("✅ Three-bucket portfolio manager initialized")
```

---

### Step 2: Integrate into Main Trading Loop

**2.1 Update `_run_trading_loop()` method (around line 850):**

Find the main loop where trading decisions are made. Add these steps:

```python
async def _run_trading_loop(self):
    """Main trading loop with three-bucket integration"""
    
    while self._should_continue():
        loop_id = self.loop_counter
        
        try:
            # ================================================================
            # STEP 0: CLASSIFICATION (New - Three-Bucket)
            # ================================================================
            logger.info(f"LOOP {loop_id}: Three-bucket classification...")
            
            # Get current positions from shared state
            current_positions = self.shared_state.get_portfolio_positions()
            total_equity = self.shared_state.get_total_equity()
            
            # Update bucket classification
            bucket_state = self.three_bucket_manager.update_bucket_state(
                positions=current_positions,
                total_equity=total_equity,
            )
            
            # Log bucket status
            self.three_bucket_manager.log_bucket_status()
            
            # ================================================================
            # STEP 1: HEALING (New - Priority)
            # ================================================================
            healing_priority = self.three_bucket_manager.should_execute_healing()
            
            if healing_priority:
                logger.info(f"LOOP {loop_id}: Executing dead capital healing...")
                
                should_heal, reason, orders = self.three_bucket_manager.plan_healing_cycle()
                
                if should_heal and orders:
                    logger.info(f"🔧 {reason}")
                    
                    # Execute healing (with exchange callback)
                    healing_report = self.three_bucket_manager.execute_healing(
                        execution_callback=self.execution_manager.execute_market_order
                    )
                    
                    if healing_report:
                        logger.info(
                            f"✅ Healed ${healing_report.total_amount_recovered:.2f} "
                            f"({healing_report.total_positions_healed} positions)"
                        )
                        
                        # Update shared state with healed cash
                        self.shared_state.update_from_healing(healing_report)
                else:
                    logger.debug(f"💭 Healing not needed: {reason}")
            
            # ================================================================
            # STEP 2: TRADING GATES (Enhanced - Bucket-Aware)
            # ================================================================
            logger.debug(f"LOOP {loop_id}: Checking trading gates...")
            
            gates = self.three_bucket_manager.get_trading_decision_gates()
            self.three_bucket_manager.log_trading_gates()
            
            # Check if all critical gates pass
            gates_pass, gate_reason = gates['all_gates_pass']
            
            if not gates_pass:
                logger.warning(
                    f"⚠️ Trading gates failed: {gate_reason}. "
                    f"Skipping trading this cycle."
                )
                # Skip to next iteration
                await self._sleep_until_next_poll()
                continue
            
            # ================================================================
            # STEP 3: SIGNAL PROCESSING (Existing - Unchanged)
            # ================================================================
            logger.info(f"LOOP {loop_id}: Processing signals...")
            
            # ... existing signal processing code ...
            
            # ================================================================
            # STEP 4: TRADING (Existing - Unchanged but gate-checked)
            # ================================================================
            logger.info(f"LOOP {loop_id}: Building trading decisions...")
            
            # ... existing trading decision code ...
            
        except Exception as e:
            logger.error(f"❌ Loop error: {str(e)}")
            # ... existing error handling ...
```

---

### Step 3: Add Metrics to Heartbeat/Health Monitor

**3.1 Update health monitor to include bucket metrics:**

```python
# In health_monitor.py or similar:

def get_portfolio_health(self):
    """Get health including three-bucket metrics"""
    
    metrics = self.three_bucket_manager.get_bucket_metrics()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'portfolio': {
            'total_equity': metrics.operating_cash_value + metrics.productive_value + metrics.dead_value,
            'operating_cash': metrics.operating_cash_value,
            'productive_inventory': metrics.productive_value,
            'dead_capital': metrics.dead_value,
        },
        'distribution': {
            'operating_cash_pct': metrics.operating_cash_pct,
            'productive_pct': metrics.productive_pct,
            'dead_pct': metrics.dead_pct,
        },
        'health': {
            'operating_cash_health': metrics.operating_cash_health,
            'portfolio_efficiency': metrics.portfolio_efficiency,
            'bucket_balance': 'HEALTHY' if metrics.portfolio_efficiency > 60 else 'IMBALANCED',
        },
        'healing': {
            'healing_potential': metrics.healing_potential,
            'total_healed_session': metrics.total_healed_this_session,
            'healing_events': metrics.healing_events_this_session,
        }
    }
```

---

### Step 4: Update SharedState to Track Buckets

**4.1 In `core/shared_state.py`, add fields:**

```python
class SharedState:
    # ... existing code ...
    
    # Add these fields:
    current_bucket_state: Optional[PortfolioBucketState] = None
    three_bucket_metrics: Optional[BucketMetrics] = None
    healing_history: List[HealingEvent] = field(default_factory=list)
    
    # Add these methods:
    def update_from_healing(self, healing_report):
        """Update state after healing operation"""
        self.healing_history.extend(healing_report.healing_events)
        # Mark positions as healed in portfolio
        
    def get_portfolio_positions(self):
        """Return positions formatted for three-bucket manager"""
        # Convert internal position format to bucket manager format
```

---

## TESTING CHECKLIST

### Unit Tests

- [ ] `test_bucket_classifier.py`
  - [ ] Classify productive position
  - [ ] Classify dead position (below min size)
  - [ ] Classify dead position (stale)
  - [ ] Classify dead position (orphaned)
  - [ ] Classify dead position (failed performer)
  - [ ] Classify entire portfolio
  - [ ] Balance score calculation

- [ ] `test_dead_capital_healer.py`
  - [ ] Identify liquidation candidates
  - [ ] Create liquidation orders
  - [ ] Execute liquidation batch
  - [ ] Healing report generation
  - [ ] Success rate calculation

- [ ] `test_three_bucket_manager.py`
  - [ ] Update bucket state
  - [ ] Check healing threshold
  - [ ] Plan healing cycle
  - [ ] Trading decision gates
  - [ ] Metrics calculation

### Integration Tests

- [ ] Run main orchestrator with three-bucket manager
- [ ] Verify bucket classification at start of cycle
- [ ] Verify healing execution when dead capital > threshold
- [ ] Verify trading gates block when operating cash low
- [ ] Verify healing recovers capital to operating cash bucket
- [ ] Verify metrics logged correctly

### Live Testing (Paper Trading)

- [ ] Run for 1-2 hours
- [ ] Verify no errors in logs
- [ ] Check bucket status reports every cycle
- [ ] Verify healing events recorded
- [ ] Check portfolio efficiency stays 60-80%

---

## METRICS TO MONITOR

### Dashboard Metrics

```
📊 THREE-BUCKET PORTFOLIO
├─ 💵 Operating Cash:      $60.81 (55%)  [✅ HEALTHY]
├─ 📈 Productive:          $34.50 (31%)  in 1 position
├─ 💀 Dead Capital:        $15.38 (14%)  in 3 positions
├─ 🎯 Portfolio Efficiency: 31.1% (target: 60-80%)
└─ 🔧 Healing Potential:   $15.38 (recover this cycle)

HEALING STATUS
├─ Healed this session: $0.00
├─ Positions liquidated: 0
└─ Next healing ETA: When dead > $50
```

### Alert Thresholds

- ⚠️ **WARNING**: Operating cash < 1.2x floor ($12)
- 🔴 **CRITICAL**: Operating cash < floor ($10)
- ⚠️ **WARNING**: Dead capital > 20% of portfolio
- 🔴 **CRITICAL**: Dead capital > 50% of operating cash

---

## LOGGING FORMAT

### Console Output Example

```
2026-04-17 01:25:37,123 [INFO    ] MetaController - LOOP 65: Three-bucket classification...
2026-04-17 01:25:37,456 [INFO    ] BucketClassifier - 🔍 Classifying 4 positions...
2026-04-17 01:25:37,457 [DEBUG   ] BucketClassifier -    ✅ ETHUSDT: PRODUCTIVE | Value=$34.50 | Confidence=90%
2026-04-17 01:25:37,457 [DEBUG   ] BucketClassifier -    💀 BTCUSDT: DEAD (below_min_size) | Value=$0.0592 | Confidence=95%
2026-04-17 01:25:37,458 [INFO    ] BucketClassifier - 
📊 PORTFOLIO CLASSIFICATION COMPLETE
├─ 💵 Operating Cash:  $60.81 (55.0%) [HEALTHY]
├─ 📈 Productive:      $34.50 (31.0%) in 1 positions (avg $34.50)
├─ 💀 Dead Capital:    $15.38 (14.0%) in 3 positions
├─ 📌 Portfolio Total: $110.69
└─ 🎯 Efficiency:      31.0% productive | Balance score: 68/100

2026-04-17 01:25:37,459 [INFO    ] ThreeBucketManager - 💪 Healing prioritized: dead capital $15.38 > threshold $50.00
2026-04-17 01:25:37,460 [INFO    ] DeadCapitalHealer - 🔍 Searching for liquidation candidates...
2026-04-17 01:25:37,461 [INFO    ] DeadCapitalHealer - 🎯 Found 3 liquidation candidates totaling $15.38
2026-04-17 01:25:37,462 [INFO    ] DeadCapitalHealer - 📋 Created 3 liquidation orders
2026-04-17 01:25:37,465 [INFO    ] DeadCapitalHealer - 🚀 Executing 3 liquidation orders...
2026-04-17 01:25:37,500 [INFO    ] DeadCapitalHealer - ✅ Healed BTCUSDT | Recovered: $0.0586
2026-04-17 01:25:37,502 [INFO    ] DeadCapitalHealer - ✅ Healed XRPUSDT | Recovered: $0.1150
2026-04-17 01:25:37,505 [INFO    ] DeadCapitalHealer - ✅ Healed SOLUSDT | Recovered: $0.0008
2026-04-17 01:25:37,506 [INFO    ] DeadCapitalHealer - 
🎯 HEALING CYCLE COMPLETE
├─ Positions healed: 3
├─ Amount recovered: $15.17
├─ Success rate: 100%
└─ Session total: $15.17

2026-04-17 01:25:37,507 [INFO    ] ThreeBucketManager - ✅ PASS       | all_gates_pass                 | Overall: ✅ SAFE TO TRADE
2026-04-17 01:25:37,508 [INFO    ] MetaController - LOOP 65: Processing signals...
```

---

## NEXT STEPS

### Immediate (Next 30 minutes)
1. ✅ Review the 4 created files
2. ✅ Integrate imports into main orchestrator
3. ✅ Add three-bucket-manager initialization
4. ⏳ Create test file: `tests/test_three_bucket_integration.py`

### Short-term (Next 2 hours)
1. ⏳ Integrate into main trading loop
2. ⏳ Update SharedState
3. ⏳ Test with paper trading
4. ⏳ Verify logging output

### Medium-term (Next 24 hours)
1. ⏳ Add dashboard metrics
2. ⏳ Create alerting for bucket health
3. ⏳ Deploy to live trading
4. ⏳ Monitor healing efficiency

---

## SUCCESS CRITERIA

✅ **System is working correctly when:**

1. Portfolio is classified into three buckets every cycle
2. Dead capital positions are automatically liquidated
3. Operating cash never goes below floor
4. Productive inventory stays 60-80% of portfolio
5. Trading only happens when all gates pass
6. Healing recovers dead capital to operating cash
7. Metrics logged every cycle
8. No errors in logs related to buckets

---

## TROUBLESHOOTING

### Problem: "No bucket state"
- Ensure `update_bucket_state()` is called before using manager
- Check that positions are in correct format

### Problem: "Healing not executing"
- Verify `min_dead_to_heal` threshold (default $50)
- Check that `execute_market_order()` callback is working
- Look for errors in healing logs

### Problem: "Trading gates always fail"
- Check operating cash level
- Verify bucket classification working
- Review gate logic in `get_trading_decision_gates()`

### Problem: "Metrics show 0%"
- Ensure positions have correct format
- Check that totals are calculating correctly
- Verify `update_bucket_state()` is being called

---

## DOCUMENTATION COMPLETE ✅

All 4 modules created and ready to integrate!

**Quick Start Integration**:
```python
# In main orchestrator:
from core.three_bucket_manager import ThreeBucketPortfolioManager

# In __init__:
self.three_bucket_manager = ThreeBucketPortfolioManager(config)

# In main loop:
self.three_bucket_manager.update_bucket_state(positions, total_equity)
if self.three_bucket_manager.should_execute_healing():
    healing_report = self.three_bucket_manager.execute_healing()
```

**Total Implementation Time**: 2-3 hours with testing
