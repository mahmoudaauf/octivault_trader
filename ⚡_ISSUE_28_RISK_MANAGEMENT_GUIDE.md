# Issue #28: Risk Management Framework - Complete Implementation Guide

**Date:** April 11, 2026  
**Status:** 🚀 IMPLEMENTATION IN PROGRESS  
**Target Tests:** 28  
**Estimated Completion:** 3-4 hours  

---

## Executive Summary

Issue #28 implements comprehensive risk management capabilities for the Octivault trading bot, enabling real-time monitoring and enforcement of position limits, Value-at-Risk (VaR) calculations, drawdown tracking, and circuit breaker logic to prevent catastrophic losses.

### Key Capabilities:
- **VaR Calculation**: 95% confidence level using historical returns
- **Position Limits**: Hard caps on notional exposure per symbol
- **Concentration Monitoring**: Percentage-based portfolio concentration limits
- **Drawdown Tracking**: Real-time monitoring of peak-to-valley drops
- **Circuit Breakers**: Automatic trading halts when thresholds breached
- **Risk Scoring**: Composite risk metrics across all dimensions

---

## Architecture Overview

### 1. Core Risk Monitoring System

#### Component: `RiskMonitor` Class
**Purpose**: Central orchestration of all risk management functions
**Dependencies**: Market data, portfolio state, historical analytics

**Key Methods:**
```python
def calculate_var(self, confidence_level: float = 0.95) -> float
def check_position_limits(self, symbol: str, proposed_size: float) -> bool
def check_concentration_limits(self, symbol: str, proposed_value: float) -> bool
def monitor_drawdown(self) -> float
def enforce_circuit_breakers(self) -> bool
def get_comprehensive_risk_report(self) -> Dict[str, Any]
```

**State Management:**
- `_position_limits`: Dict[str, float] - Hard caps per symbol
- `_concentration_limits`: Dict[str, float] - Max % of portfolio per symbol
- `_peak_portfolio_value`: float - Historical peak for drawdown
- `_circuit_breaker_triggered`: bool - Global trading halt flag
- `_var_history`: Deque - Rolling window of daily returns
- `_risk_events`: List - Audit trail of risk violations

---

### 2. VaR (Value-at-Risk) Engine

#### Component: `VaRCalculator` Class
**Purpose**: Probabilistic risk assessment using historical simulation
**Method**: Historical simulation (non-parametric)

**Implementation:**
```python
def calculate_historical_var(
    self, 
    returns: List[float], 
    confidence_level: float = 0.95
) -> float
```

**Algorithm:**
1. Collect daily portfolio returns (252-day window)
2. Sort returns in ascending order
3. Calculate percentile = (1 - confidence_level) * 100
4. Return worst-case loss at that percentile

**Example:**
- Portfolio value: $1,000,000
- Daily returns: 252-day history
- 95% VaR: $25,000 (worst 5% of days)
- Interpretation: On 95% of days, max loss < $25,000

**Why Historical Simulation:**
- Non-parametric (no distribution assumptions)
- Captures fat tails and regime changes
- Easy to explain to risk committees
- Works with skewed distributions

---

### 3. Position Limits Framework

#### Component: `PositionLimiter` Class
**Purpose**: Hard caps on individual positions to prevent concentration risk
**Strategy**: Notional exposure limits

**Implementation:**
```python
def set_position_limits(self, limits: Dict[str, float]) -> None
def check_position_limit(self, symbol: str, current_position: float) -> bool
def get_remaining_capacity(self, symbol: str) -> float
```

**Limit Types:**
1. **Notional Limits**: Max dollars per symbol
   - Example: BTC max $50,000, ETH max $30,000
   - Prevents single-security concentration

2. **Concentration Limits**: Max % of portfolio
   - Example: BTC max 15% of portfolio
   - Adapts to portfolio growth

3. **Sector Limits**: Max per market segment
   - Example: Layer-2 max 10% of portfolio
   - Diversification enforcement

**Enforcement Logic:**
```
IF proposed_position + current_position > limit
  THEN reject order
ELSE allow order
```

---

### 4. Drawdown Monitoring System

#### Component: `DrawdownMonitor` Class
**Purpose**: Track peak-to-valley declines for psychological and risk control
**Metrics**: Peak value, current drawdown %, max allowed drawdown

**Implementation:**
```python
def update_portfolio_value(self, current_value: float) -> None
def get_current_drawdown(self) -> float
def get_max_drawdown(self) -> float
def check_drawdown_threshold(self, max_allowed: float) -> bool
```

**Key Metrics:**
- **Current Drawdown**: (Peak - Current) / Peak
- **Max Drawdown**: Worst historical drawdown
- **Recovery Needed**: How much gain needed to break even

**Example:**
```
Peak Value: $1,000,000
Current Value: $850,000
Current Drawdown: 15%
Max Allowed: 20%
Status: OK (below threshold)
```

---

### 5. Circuit Breaker Logic

#### Component: `CircuitBreaker` Class
**Purpose**: Automated trading halts when risk thresholds breached
**Strategy**: Multi-level triggers with escalation

**Implementation:**
```python
def check_breach(
    self,
    var_breach: bool,
    drawdown_breach: bool,
    concentration_breach: bool,
    loss_threshold_breach: bool
) -> Tuple[bool, str]

def trigger_halt(self, reason: str) -> None
def clear_halt(self) -> None
def get_halt_status(self) -> Dict[str, Any]
```

**Trigger Levels:**

| Level | Condition | Action | Example |
|-------|-----------|--------|---------|
| 1 | Any limit breached | Log warning, flag for review | VaR exceeded |
| 2 | 2+ limits breached | Stop new trades | Drawdown + concentration |
| 3 | 3+ limits breached | Close 10% position | VaR + drawdown + concentration |
| 4 | Portfolio loss > 10% | Full trading halt | Circuit breaker engaged |

---

### 6. Comprehensive Risk Reporting

#### Component: `RiskReporter` Class
**Purpose**: Executive-level risk dashboard and compliance reporting
**Output**: Structured risk report with all metrics

**Implementation:**
```python
def generate_comprehensive_report(self) -> Dict[str, Any]
def get_risk_summary(self) -> Dict[str, Any]
def export_to_compliance_format(self) -> str
```

**Report Structure:**
```python
{
    "timestamp": "2026-04-11T14:30:00Z",
    "portfolio_metrics": {
        "total_value": 1000000.0,
        "cash_available": 250000.0,
        "invested": 750000.0,
    },
    "var_metrics": {
        "var_95": 25000.0,
        "var_99": 35000.0,
        "recent_var": [24500, 25100, 24800],
    },
    "position_metrics": {
        "BTC": {
            "notional": 45000.0,
            "limit": 50000.0,
            "utilization": 0.90,
            "status": "OK"
        },
        ...
    },
    "drawdown_metrics": {
        "peak": 1000000.0,
        "current": 900000.0,
        "drawdown_pct": 0.10,
        "max_allowed": 0.20,
        "status": "OK"
    },
    "circuit_breaker": {
        "triggered": False,
        "reason": None,
        "breach_count": 0
    },
    "risk_score": 0.35,  # 0-1 scale, 1 = maximum risk
}
```

---

## Method Specifications

### Main Methods (6)

#### 1. `calculate_var(confidence_level: float = 0.95) -> float`
**Purpose**: Calculate Value-at-Risk at specified confidence level
**Input**: Confidence level (0.90, 0.95, 0.99)
**Output**: Maximum expected loss in dollars
**Logic**:
1. Collect 252-day portfolio return history
2. Sort returns ascending
3. Find percentile = (1 - confidence_level) * 100
4. Return dollar amount = portfolio_value * returns[percentile]

**Error Handling**:
- Insufficient history: raise ValueError
- Invalid confidence: clamp to [0.5, 0.99]

---

#### 2. `check_position_limits(symbol: str, proposed_size: float) -> bool`
**Purpose**: Validate proposed trade against position limits
**Input**: Symbol code, proposed position size
**Output**: True if allowed, False if breached
**Logic**:
1. Get current position for symbol
2. Calculate proposed total = current + proposed
3. Check against notional limit
4. Check against concentration limit
5. Return True if all checks pass

**Example**:
```python
# Current BTC position: 0.5 BTC @ $45,000 = $22,500
# Limit: $50,000 notional
# Proposed: 0.3 BTC @ $45,000 = $13,500
# Total: $36,000 < $50,000 ✓ ALLOWED
check_position_limits("BTC", 0.3)  # Returns True
```

---

#### 3. `check_concentration_limits(symbol: str, proposed_value: float) -> bool`
**Purpose**: Validate proposed trade against portfolio concentration
**Input**: Symbol code, proposed trade value in dollars
**Output**: True if allowed, False if concentration exceeded
**Logic**:
1. Get portfolio total value
2. Calculate current symbol concentration = symbol_value / portfolio_total
3. Calculate proposed new concentration = (symbol_value + proposed_value) / portfolio_total
4. Check against concentration limit (e.g., 15%)
5. Return True if under limit

**Example**:
```python
# Portfolio: $1,000,000
# BTC current: $100,000 (10%)
# Concentration limit: 15%
# Proposed: $40,000 trade
# New concentration: ($100k + $40k) / $1M = 14% < 15% ✓ ALLOWED
check_concentration_limits("BTC", 40000)  # Returns True
```

---

#### 4. `monitor_drawdown() -> float`
**Purpose**: Calculate current drawdown from peak
**Input**: None (uses internal state)
**Output**: Drawdown percentage (0-1 scale)
**Logic**:
1. Get portfolio peak value (historical maximum)
2. Get current portfolio value
3. Calculate drawdown = (peak - current) / peak
4. Update max drawdown if exceeded
5. Return drawdown percentage

**Example**:
```python
# Peak: $1,000,000
# Current: $900,000
# Drawdown: 0.10 (10%)
# Max threshold: 0.20 (20%)
# Status: SAFE (under threshold)
monitor_drawdown()  # Returns 0.10
```

---

#### 5. `enforce_circuit_breakers() -> bool`
**Purpose**: Check all breach conditions and enforce trading halt if needed
**Input**: None (uses internal state)
**Output**: True if trading allowed, False if halted
**Logic**:
1. Check VaR breach (loss > yesterday's VaR)
2. Check drawdown breach (current > max allowed)
3. Check concentration breach (any symbol over limit)
4. Count breaches
5. IF breaches >= 2: trigger halt
6. IF breaches >= 3: start closing positions
7. IF loss > 10%: full halt
8. Return current trading status

**Breach Scoring**:
```
0 breaches: Trading allowed (green)
1 breach: Warning mode (yellow) - log but allow
2 breaches: Stop new trades (orange) - no new orders
3 breaches: Start closing (red) - exit 10% positions
Portfolio loss > 10%: Full halt (black) - all trading off
```

---

#### 6. `get_comprehensive_risk_report() -> Dict[str, Any]`
**Purpose**: Generate executive risk dashboard
**Input**: None (uses internal state)
**Output**: Structured risk report with all metrics
**Logic**:
1. Collect portfolio metrics
2. Calculate VaR at 95% and 99%
3. Gather position metrics and utilization
4. Calculate drawdown and max drawdown
5. Assess circuit breaker status
6. Calculate composite risk score (0-1)
7. Return structured report

**Risk Score Calculation**:
```
risk_score = (var_util * 0.3 + 
              drawdown_util * 0.3 + 
              concentration_util * 0.25 + 
              loss_util * 0.15)

Where utilization = current_metric / max_allowed_metric
Range: 0.0 (safe) to 1.0 (maximum risk)
```

---

### Helper Methods (4)

#### 1. `_aggregate_risk_metrics() -> Dict[str, Any]`
**Purpose**: Collect all raw risk data points
**Logic**:
1. Sum notional exposures
2. Collect daily returns
3. Get current positions
4. Calculate concentrations
5. Return aggregated metrics

---

#### 2. `_validate_risk_thresholds() -> List[str]`
**Purpose**: Cross-check all limits and return violations
**Logic**:
1. For each position: check notional limit
2. For each symbol: check concentration limit
3. Check portfolio drawdown
4. Check VaR threshold
5. Return list of violations

---

#### 3. `_calculate_risk_score() -> float`
**Purpose**: Compute composite risk metric (0-1 scale)
**Logic**:
1. Get utilization of each limit
2. Weight by importance: VaR 30%, Drawdown 30%, Concentration 25%, Loss 15%
3. Calculate weighted average
4. Clamp to [0.0, 1.0]

---

#### 4. `_format_risk_alerts() -> List[Dict[str, Any]]`
**Purpose**: Generate actionable risk alerts for traders
**Logic**:
1. For each breach: create alert dict
2. Include severity (warning/error/critical)
3. Include recommended action
4. Sort by severity
5. Return alert list

---

## Infrastructure Requirements

### In `MetaController.__init__()`

```python
# Risk management infrastructure
self._risk_monitor = None
self._position_limits = {}  # Dict[str, float]
self._concentration_limits = {}  # Dict[str, float]
self._var_history = collections.deque(maxlen=252)  # 252-day window
self._peak_portfolio_value = 0.0
self._circuit_breaker_triggered = False
self._breach_count = 0
self._risk_events = []  # Audit trail

# Threading locks for thread safety
self._risk_monitor_lock = threading.Lock()
self._position_limits_lock = threading.Lock()
self._circuit_breaker_lock = threading.Lock()
```

### Required Imports

```python
from collections import deque
import statistics
import threading
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
```

---

## Test Strategy

### Test Categories (28 total)

**Category 1: Infrastructure Tests (4 tests)**
- `test_risk_monitor_initialization`
- `test_position_limits_setup`
- `test_concentration_limits_setup`
- `test_circuit_breaker_initialization`

**Category 2: VaR Calculation Tests (4 tests)**
- `test_var_calculation_basic`
- `test_var_95_confidence`
- `test_var_99_confidence`
- `test_var_with_limited_history`

**Category 3: Position Limits Tests (4 tests)**
- `test_position_limit_allowed`
- `test_position_limit_breached`
- `test_position_limit_at_boundary`
- `test_multiple_position_limits`

**Category 4: Concentration Limits Tests (4 tests)**
- `test_concentration_limit_allowed`
- `test_concentration_limit_breached`
- `test_concentration_adapts_to_portfolio`
- `test_multiple_concentration_limits`

**Category 5: Drawdown Monitoring Tests (3 tests)**
- `test_drawdown_calculation`
- `test_max_drawdown_tracking`
- `test_drawdown_threshold_breach`

**Category 6: Circuit Breaker Tests (4 tests)**
- `test_circuit_breaker_single_breach`
- `test_circuit_breaker_multiple_breaches`
- `test_circuit_breaker_halt_escalation`
- `test_circuit_breaker_recovery`

**Category 7: Risk Reporting Tests (2 tests)**
- `test_comprehensive_risk_report`
- `test_risk_score_calculation`

**Category 8: Integration Tests (2 tests)**
- `test_end_to_end_risk_management`
- `test_concurrent_risk_checks`

**Category 9: Edge Cases (1 test)**
- `test_risk_under_stress_conditions`

---

## Implementation Timeline

### Phase 1: Core VaR Engine (45 min)
- [ ] Create `VaRCalculator` class
- [ ] Implement historical VaR calculation
- [ ] Add 95%, 99% confidence levels
- [ ] Write and pass 4 VaR tests

### Phase 2: Position & Concentration Limits (60 min)
- [ ] Create `PositionLimiter` class
- [ ] Implement notional limit checking
- [ ] Implement concentration limit checking
- [ ] Write and pass 8 limit tests

### Phase 3: Drawdown Monitoring (30 min)
- [ ] Create `DrawdownMonitor` class
- [ ] Implement peak tracking
- [ ] Implement drawdown calculation
- [ ] Write and pass 3 drawdown tests

### Phase 4: Circuit Breakers (45 min)
- [ ] Create `CircuitBreaker` class
- [ ] Implement multi-level breach logic
- [ ] Implement escalation logic
- [ ] Write and pass 4 circuit breaker tests

### Phase 5: Risk Reporting (30 min)
- [ ] Create `RiskReporter` class
- [ ] Implement comprehensive report generation
- [ ] Implement risk score calculation
- [ ] Write and pass 2 reporting tests

### Phase 6: Integration & Stress Testing (30 min)
- [ ] Create integration tests
- [ ] Test concurrent access patterns
- [ ] Test stress conditions
- [ ] Pass all 28 tests

---

## Success Criteria

✅ All 28 tests passing (100%)  
✅ 100% type hints on all methods  
✅ 100% docstrings on all methods  
✅ Zero regressions on previous tests (257/257)  
✅ Thread safety verified through concurrent tests  
✅ Performance: All operations < 50ms  
✅ Production-ready code quality  

---

## Deliverables

1. ✅ Issue #28 Implementation Guide (this document)
2. ✅ Test suite with 28 comprehensive tests
3. ✅ Implementation with all 6 main + 4 helper methods
4. ✅ Infrastructure components in MetaController
5. ✅ Completion report with metrics and analysis
6. ✅ Zero regressions verification

---

## Notes

- **VaR Interpretation**: 95% VaR of $25k means: on 95% of days, max daily loss < $25k
- **Circuit Breaker Philosophy**: Prevent catastrophic losses through layered controls
- **Concentration Limits**: Adapt as portfolio grows to maintain dollar-based and percentage-based limits
- **Drawdown Psychology**: 20% drawdown is typical max for professional traders
- **Risk Score**: 0 (completely safe) to 1 (maximum risk) - use as real-time dashboard

---

**Next Steps:** Ready to generate comprehensive test suite with 28 tests covering all scenarios.

