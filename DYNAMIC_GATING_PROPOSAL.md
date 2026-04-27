# DYNAMIC GATING STRATEGY PROPOSAL
## Adaptive Filtering Based on System State

---

## EXECUTIVE SUMMARY

**Current Issue**: Static gates block ALL trades when system not fully ready
```
if not market_data_ready: BLOCK ALL
if not balances_ready: BLOCK ALL
if not ops_plane_ready: BLOCK ALL
```

**Proposed Solution**: Dynamic gates that adapt based on:
- System warm-up phase (0-5 min) → Strict gating
- Initialization phase (5-20 min) → Relaxed gating  
- Steady state (20+ min) → Risk-based gating
- Execution success rate → Confidence-based gating
- Capital utilization → Allocation-based gating

---

## CURRENT STATE ANALYSIS

### Why System Shows `decision=NONE`

From our diagnostics:
1. System makes trading decisions (`decision=BUY/SELL`)
2. But ALL signals are filtered out by gates
3. Result: `decision=NONE` in LOOP_SUMMARY

### Current Static Gates (Lines 9448-9453)

```python
gated_reasons = []
if not snap.get("market_data_ready", True): 
    gated_reasons.append("MarketData")
if not snap.get("balances_ready", True): 
    gated_reasons.append("Balances")
if not snap.get("ops_plane_ready", True): 
    gated_reasons.append("OpsPlane")

if gated_reasons:
    # BLOCKED - No trading allowed
    # Only allow SELL/liquidation (no BUY)
```

### Problem with Current Approach

| Phase | Current Behavior | Impact |
|-------|------------------|--------|
| Startup (0-5 min) | Super strict gates | ✅ Safe but kills trading |
| Warm-up (5-20 min) | Still strict | ❌ NO SIGNALS = NO DATA |
| Steady (20+ min) | Same strict gates | ❌ Never learns system is ready |

---

## PROPOSED DYNAMIC GATE FRAMEWORK

### Phase 1: System Warm-up (Elapsed < 5 minutes)

**Strategy**: STRICT gates, but allow bootstrap trades
```python
class SystemPhase:
    BOOTSTRAP = "bootstrap"      # 0-5 minutes
    INITIALIZATION = "init"      # 5-20 minutes  
    STEADY_STATE = "steady"      # 20+ minutes
    ERROR_RECOVERY = "error"     # After errors

def get_system_phase(elapsed_minutes):
    if elapsed_minutes < 5:
        return SystemPhase.BOOTSTRAP
    elif elapsed_minutes < 20:
        return SystemPhase.INITIALIZATION
    else:
        return SystemPhase.STEADY_STATE
```

**Gate Rules**:
- ❌ No normal BUY signals yet (wait for stability)
- ✅ Allow bootstrap/dust healing trades (guaranteed safe)
- ✅ Allow SELL/liquidation (risk management)

**Log**: `[Gate:BOOTSTRAP] Strict mode - bootstrap trades only`

---

### Phase 2: Initialization (5-20 minutes)

**Strategy**: Relax gates based on success metrics

**Success Metrics**:
- Count: Trade attempts in this phase
- Success rate: Filled / Attempted
- Error rate: Rejections / Attempted

**Gate Rules**:
```python
if success_rate > 80%:
    # System is executing well → Relax gates
    allow_normal_buys = True
    confidence_threshold = 0.50 (was 0.60)
    
elif success_rate > 50%:
    # System working but imperfect → Medium gates
    allow_normal_buys = True  
    confidence_threshold = 0.65
    
else:
    # System struggling → Keep strict gates
    allow_normal_buys = False
    bootstrap_only = True
```

**Log**: `[Gate:INIT] Success rate=75% → Relaxing gates`

---

### Phase 3: Steady State (20+ minutes)

**Strategy**: Risk-based dynamic gating

**Factors**:
1. **Execution health**: Recent fill rate
2. **Capital utilization**: % of capital deployed
3. **PnL trend**: Positive? Negative?
4. **Error frequency**: Recent rejections?

**Gate Adjustment Logic**:
```python
def compute_dynamic_confidence_threshold():
    """Adjust confidence threshold based on system health"""
    
    base_threshold = 0.55
    
    # Factor 1: Execution health (0 to -0.15)
    fill_rate = recent_fills / recent_attempts
    health_adjustment = -0.15 * (1 - fill_rate)  # Perfect = -0.15, Poor = 0
    
    # Factor 2: Capital utilization (0 to +0.10)
    capital_deployed = portfolio_value / total_capital
    if capital_deployed > 0.7:
        capital_adjustment = +0.10  # Too much deployed
    elif capital_deployed < 0.2:
        capital_adjustment = -0.05  # Underdeployed
    else:
        capital_adjustment = 0
    
    # Factor 3: PnL trend (0 to ±0.10)
    pnl_change = (current_pnl - previous_pnl) / abs(previous_pnl + 0.01)
    pnl_adjustment = -0.10 * pnl_change  # Good PnL = lower threshold
    
    # Factor 4: Recent errors (0 to +0.15)
    error_rate = recent_errors / recent_loops
    error_adjustment = +0.15 * error_rate
    
    threshold = base_threshold + health_adjustment + capital_adjustment + pnl_adjustment + error_adjustment
    
    # Clamp to reasonable range
    return max(0.40, min(0.80, threshold))
```

**Example Adjustments**:
- Executing well, profitable, low capital: threshold = 0.40 (AGGRESSIVE)
- Executing poorly, losses, overdeployed: threshold = 0.75 (CONSERVATIVE)

---

## IMPLEMENTATION PLAN

### Step 1: Add System Phase Tracking

**File**: `core/meta_controller.py` (early in init)

```python
class DynamicGatingSystem:
    def __init__(self):
        self.system_start_time = time.time()
        self.phase = SystemPhase.BOOTSTRAP
        
        # Metrics tracking
        self.recent_attempts = deque(maxlen=50)  # Last 50 loops
        self.recent_fills = deque(maxlen=50)
        self.recent_errors = deque(maxlen=50)
        self.recent_pnls = deque(maxlen=10)
        
    def update_phase(self):
        elapsed = (time.time() - self.system_start_time) / 60
        if elapsed < 5:
            self.phase = SystemPhase.BOOTSTRAP
        elif elapsed < 20:
            self.phase = SystemPhase.INITIALIZATION
        else:
            self.phase = SystemPhase.STEADY_STATE
            
    def record_loop_result(self, exec_attempted, exec_result, pnl):
        if exec_attempted:
            self.recent_attempts.append(1)
            if exec_result in ("FILLED", "PLACED"):
                self.recent_fills.append(1)
            elif exec_result == "REJECTED":
                self.recent_errors.append(1)
        self.recent_pnls.append(pnl)
```

### Step 2: Modify Gate Logic

**File**: `core/meta_controller.py` (lines 9448-9538)

```python
# BEFORE: Static gates
if gated_reasons:
    safe_decisions = [d for d in decisions if not is_budget_required(d.side)]

# AFTER: Dynamic gates
gate_policy = self._compute_dynamic_gate_policy()
if gated_reasons:
    if gate_policy.allow_bootstrap_trades:
        safe_decisions = get_bootstrap_trades(decisions)
    elif gate_policy.allow_normal_trades:
        safe_decisions = decisions  # Allow all
    else:
        safe_decisions = get_liquidation_only(decisions)  # SELL only
```

### Step 3: Compute Dynamic Thresholds

**File**: `core/meta_controller.py` (new method)

```python
def _compute_dynamic_confidence_threshold(self):
    """Compute adaptive confidence threshold based on system health"""
    
    # Phase 1: Bootstrap → strict
    if self.phase == SystemPhase.BOOTSTRAP:
        return 0.70
    
    # Phase 2: Initialization → based on success rate
    elif self.phase == SystemPhase.INITIALIZATION:
        success_rate = len(self.recent_fills) / max(len(self.recent_attempts), 1)
        if success_rate > 0.80:
            return 0.50
        elif success_rate > 0.50:
            return 0.60
        else:
            return 0.70
    
    # Phase 3: Steady state → comprehensive health-based
    else:
        return self._compute_steady_state_threshold()
```

---

## EXPECTED OUTCOMES

### Before Implementation
```
Loop 1-50: decision=NONE (system warming up, all gated)
Loop 51+: decision=NONE (gates never relax, system learns nothing)
Result: ZERO trades, PnL $0.00
```

### After Implementation
```
Loop 1-50:  decision=NONE (bootstrap phase, strict gates)
Loop 51-100: decision=BUY/SELL (init phase, gates relaxed as success rate improves)
Loop 101+: decision=BUY/SELL (steady state, risk-based thresholds)
Result: TRADES EXECUTE, PnL ACCUMULATES
```

---

## IMPLEMENTATION CHECKLIST

- [ ] Create `DynamicGatingSystem` class
- [ ] Add phase tracking (bootstrap/init/steady)
- [ ] Implement success rate metrics
- [ ] Implement dynamic confidence calculation
- [ ] Modify gate logic in execute loop
- [ ] Add telemetry/logging for phase changes
- [ ] Add telemetry/logging for threshold adjustments
- [ ] Test with dry run
- [ ] Deploy to live system
- [ ] Monitor for 24 hours

---

## BENEFITS

1. **System Learns**: Early gates don't prevent later trading
2. **Adaptive Safety**: Strict when warming up, flexible when stable
3. **Data-Driven**: Decisions based on actual execution success
4. **Feedback Loop**: System improves its own thresholds
5. **Transparency**: Logs show why gates relaxed/tightened
6. **Backward Compatible**: Can disable with `DYNAMIC_GATING_ENABLED=false`

---

## RISK MITIGATION

**Risk**: Over-relaxing gates could cause losses
**Mitigation**: 
- Conservative thresholds (0.40-0.80, default 0.55)
- Never allow infinite capital allocation
- Always allow SELL/liquidation (risk management)

**Risk**: Phase transitions could be abrupt
**Mitigation**:
- Smooth transitions using success rate
- Gradual threshold adjustments (max ±0.05 per loop)
- Revert to strict if error rate spikes

---

## RECOMMENDED START: Phase 2 (Initialization) Logic

Don't wait for full framework - just implement the success-rate-based relaxation:

```python
# Quick win: In initialization phase, relax gates based on success
elapsed = (time.time() - start_time) / 60
if 5 < elapsed < 20:
    # Count recent successes
    recent_success = count_filled_orders_last_10_loops()
    if recent_success >= 2:  # At least 2 trades executed
        allow_normal_buys = True
        log("[Gate:SUCCESS_BASED] Relaxing gates after 2 successful trades")
```

This single change could unlock trading immediately after first successful order!

---

**Proposal**: Implement dynamic gating framework to enable trading as system stabilizes?

✅ Recommended - This explains why system shows NO SIGNALS currently!
