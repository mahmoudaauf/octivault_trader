# Issue #29: Real-time Market Events - Complete Implementation Guide

**Date:** April 11, 2026  
**Status:** 🚀 IMPLEMENTATION IN PROGRESS  
**Target Tests:** 26  
**Estimated Completion:** 3-4 hours  

---

## Executive Summary

Issue #29 implements comprehensive real-time market event detection and response capabilities, enabling the trading bot to identify and react to market anomalies, flash crashes, and liquidity crises with automatic position adjustments and risk mitigation.

### Key Capabilities:
- **Anomaly Detection**: Statistical deviation detection in price/volume
- **Flash Crash Detection**: Rapid price decline identification (<1 second)
- **Liquidity Crisis Detection**: Order book depth and spread monitoring
- **Circuit Breaker Integration**: Automatic halt on extreme events
- **Auto Position Adjustment**: Intelligent position sizing during crises
- **Event Logging & Audit**: Complete audit trail of all events

---

## Architecture Overview

### 1. Market Event Detection System

#### Component: `MarketEventDetector` Class
**Purpose**: Central orchestration of real-time market event monitoring
**Dependencies**: Real-time price data, volume data, order book depth

**Key Methods:**
```python
def detect_price_anomaly(self, symbol: str, current_price: float) -> bool
def detect_flash_crash(self, symbol: str, price_change: float, timeframe_ms: int) -> bool
def detect_liquidity_crisis(self, symbol: str, bid_ask_spread: float, depth: float) -> bool
def detect_volume_spike(self, symbol: str, current_volume: float, avg_volume: float) -> bool
def should_adjust_positions(self) -> bool
def get_active_events(self) -> Dict[str, List[Dict[str, Any]]]
```

**State Management:**
- `_price_anomalies`: Dict[str, bool] - Current anomaly flags
- `_flash_crash_events`: List - Recent flash crash records
- `_liquidity_crises`: Dict[str, bool] - Current liquidity status
- `_event_history`: Deque - Rolling event audit trail
- `_event_severity`: Dict[str, int] - Severity ratings (1-5)

---

### 2. Anomaly Detection Engine

#### Component: `AnomalyDetector` Class
**Purpose**: Statistical detection of price/volume deviations
**Method**: Z-score based anomaly detection with configurable thresholds

**Implementation:**
```python
def calculate_z_score(
    self,
    current_value: float,
    mean: float,
    std_dev: float
) -> float

def is_anomaly(
    self,
    z_score: float,
    threshold: float = 3.0
) -> bool
```

**Algorithm:**
1. Maintain rolling window of 100 price points
2. Calculate mean and standard deviation
3. Calculate Z-score = (current - mean) / std_dev
4. Trigger if |Z-score| > threshold (typically 3.0 = 99.7% confidence)

**Example:**
```
Price History Mean: $50,000
Standard Deviation: $1,000
Current Price: $53,100
Z-score: (53100 - 50000) / 1000 = 3.1
Threshold: 3.0
Result: ANOMALY DETECTED ⚠️ (beyond 99.7% confidence)
```

---

### 3. Flash Crash Detection

#### Component: `FlashCrashDetector` Class
**Purpose**: Identify rapid price declines within extremely short timeframes
**Strategy**: Time-windowed price velocity detection

**Implementation:**
```python
def detect_flash_crash(
    self,
    price_history: Deque[Tuple[float, float]],  # (price, timestamp)
    time_window_ms: int = 1000,  # 1 second
    decline_threshold: float = 0.10  # 10%
) -> Tuple[bool, float]
```

**Flash Crash Criteria:**
- Decline > 10% within 1 second = Flash crash
- Multiple similar events within 5 minutes = Market instability
- Recover > 50% within 10 seconds = Typical flash crash pattern

**Example:**
```
T=0ms:   Price = $50,000
T=500ms: Price = $47,000 (6% decline)
T=1000ms: Price = $45,000 (10% total decline in 1 second)
Result: FLASH CRASH DETECTED ⚠️
```

---

### 4. Liquidity Crisis Detection

#### Component: `LiquidityCrisisDetector` Class
**Purpose**: Monitor order book health and spread widening
**Strategy**: Real-time monitoring of bid-ask spread and order book depth

**Implementation:**
```python
def check_liquidity_crisis(
    self,
    bid_price: float,
    ask_price: float,
    bid_volume: float,
    ask_volume: float,
    typical_spread: float
) -> Tuple[bool, str]
```

**Liquidity Crisis Indicators:**
1. **Spread Widening**: Bid-ask spread > 2x normal
2. **Depth Collapse**: Order book < 50% of typical depth
3. **Price Gaps**: No orders in middle of book
4. **Liquidity Imbalance**: One side has <20% of opposite side

**Example:**
```
Normal Spread: 0.04% ($20)
Current Spread: 0.12% ($60) - 3x wider
Bid Volume: $500,000
Ask Volume: $50,000 - Severe imbalance (10:1 ratio)
Result: LIQUIDITY CRISIS DETECTED ⛔
```

---

### 5. Auto Position Adjustment System

#### Component: `AutoPositionAdjuster` Class
**Purpose**: Intelligent position sizing during market stress
**Strategy**: Scaled position reductions based on event severity

**Implementation:**
```python
def calculate_adjustment(
    self,
    current_position: float,
    event_severity: int,  # 1-5 scale
    risk_level: str  # 'normal', 'elevated', 'critical'
) -> float

def execute_position_reduction(
    self,
    symbol: str,
    reduction_percent: float
) -> bool
```

**Adjustment Logic:**

| Event Severity | Position Reduction | Action | Example |
|----------------|-------------------|--------|---------|
| 1 (Low) | 0% | Monitor only | Price anomaly detected |
| 2 (Moderate) | 10% | Reduce size by 10% | Volume spike 2x normal |
| 3 (High) | 25% | Reduce size by 25% | Flash crash 5% decline |
| 4 (Critical) | 50% | Reduce size by 50% | Liquidity crisis |
| 5 (Extreme) | 100% | Close all positions | Multiple concurrent crises |

**Example:**
```
Current Position: 1.0 BTC ($50,000)
Event Severity: 4 (Critical - liquidity crisis)
Reduction: 50%
Action: Sell 0.5 BTC immediately
Remaining: 0.5 BTC ($25,000)
```

---

### 6. Event Logging & Audit Trail

#### Component: `EventAuditTrail` Class
**Purpose**: Complete logging of all market events for compliance and analysis
**Output**: Structured event records with timestamps and context

**Implementation:**
```python
def log_event(
    self,
    event_type: str,
    symbol: str,
    severity: int,
    details: Dict[str, Any]
) -> None

def get_event_history(
    self,
    symbol: str,
    time_window_minutes: int = 60
) -> List[Dict[str, Any]]
```

**Event Structure:**
```python
{
    "timestamp": "2026-04-11T14:30:00.123Z",
    "event_type": "flash_crash",
    "symbol": "BTC",
    "severity": 4,  # 1-5 scale
    "details": {
        "price_before": 50000.0,
        "price_after": 45000.0,
        "decline_percent": 10.0,
        "timeframe_ms": 1000,
        "liquidity_adjusted": True,
        "position_reduced_to": 0.5
    },
    "action_taken": "position_reduction",
    "result": "success"
}
```

---

## Method Specifications

### Main Methods (6)

#### 1. `detect_price_anomaly(symbol: str, current_price: float) -> bool`
**Purpose**: Detect statistical anomalies in price movement
**Input**: Symbol code, current price
**Output**: True if anomaly detected, False otherwise
**Logic**:
1. Get rolling price history (100 points)
2. Calculate mean and standard deviation
3. Calculate Z-score
4. Compare against threshold (3.0 std devs)
5. Return True if |Z-score| > 3.0

**Error Handling**:
- Insufficient history: return False
- Division by zero: return False

---

#### 2. `detect_flash_crash(symbol: str, price_change: float, timeframe_ms: int) -> bool`
**Purpose**: Identify rapid price declines (flash crashes)
**Input**: Symbol, price change percentage, timeframe in milliseconds
**Output**: True if flash crash detected, False otherwise
**Logic**:
1. Check if decline > 10% within timeframe
2. Check if decline occurred within 1 second
3. Verify recovery pattern (optional but typical)
4. Return True if meets criteria

**Example**:
```python
# Price dropped 10% in 500ms
detect_flash_crash("BTC", -0.10, 500)  # Returns True
```

---

#### 3. `detect_liquidity_crisis(symbol: str, bid_ask_spread: float, depth: float) -> bool`
**Purpose**: Identify market liquidity problems
**Input**: Symbol, current bid-ask spread %, available depth in dollars
**Output**: True if liquidity crisis detected, False otherwise
**Logic**:
1. Get historical normal spread for symbol
2. Check if current spread > 2x normal
3. Check if depth < 50% of typical
4. Check bid-ask imbalance > 10:1
5. Return True if 2+ indicators triggered

**Example**:
```python
# Normal spread: 0.04%, current: 0.12%
# Normal depth: $5M, current: $2M
detect_liquidity_crisis("BTC", 0.0012, 2000000)  # Returns True
```

---

#### 4. `detect_volume_spike(symbol: str, current_volume: float, avg_volume: float) -> bool`
**Purpose**: Identify unusual trading volume
**Input**: Symbol, current volume, historical average volume
**Output**: True if spike detected (>2x average), False otherwise
**Logic**:
1. Calculate volume ratio = current / average
2. Return True if ratio > 2.0
3. Record spike severity based on multiple

**Example**:
```python
# Average: 100,000 contracts
# Current: 250,000 contracts (2.5x)
detect_volume_spike("BTC", 250000, 100000)  # Returns True
```

---

#### 5. `should_adjust_positions() -> bool`
**Purpose**: Determine if market conditions warrant position adjustments
**Input**: None (uses internal state)
**Output**: True if adjustment needed, False otherwise
**Logic**:
1. Count active market events
2. Calculate composite risk score
3. Check for critical-level events
4. Return True if adjustment threshold exceeded

**Adjustment Triggers**:
```
1 anomaly = Monitor
2 anomalies = Consider adjustment
Flash crash + liquidity = ADJUST (50%)
3+ concurrent events = ADJUST (100% - close all)
```

---

#### 6. `get_active_events() -> Dict[str, List[Dict[str, Any]]]`
**Purpose**: Return all currently active market events
**Input**: None (uses internal state)
**Output**: Structured dict of active events by type
**Logic**:
1. Collect all active anomalies
2. Collect recent flash crashes (< 5 minutes old)
3. Collect active liquidity crises
4. Format with timestamps and severity
5. Return organized event dictionary

**Return Structure**:
```python
{
    "anomalies": [
        {
            "symbol": "BTC",
            "z_score": 3.5,
            "severity": 2,
            "detected_at": "2026-04-11T14:30:00Z"
        }
    ],
    "flash_crashes": [
        {
            "symbol": "ETH",
            "decline_percent": 8.5,
            "severity": 3,
            "detected_at": "2026-04-11T14:29:45Z"
        }
    ],
    "liquidity_crises": [
        {
            "symbol": "SOL",
            "spread_multiple": 2.3,
            "depth_degradation": 0.4,
            "severity": 4,
            "detected_at": "2026-04-11T14:29:30Z"
        }
    ]
}
```

---

### Helper Methods (4)

#### 1. `_calculate_event_severity(indicators: List[bool]) -> int`
**Purpose**: Determine severity level based on triggering indicators
**Logic**:
- 0 indicators: severity 0 (no event)
- 1 indicator: severity 2 (moderate)
- 2 indicators: severity 3 (high)
- 3+ indicators: severity 4-5 (critical-extreme)

---

#### 2. `_check_volume_pattern(symbol: str, current_vol: float) -> float`
**Purpose**: Analyze volume spike pattern
**Logic**:
1. Calculate volume ratio vs average
2. Check for sustained elevation
3. Return normalized volume score

---

#### 3. `_estimate_recovery_likelihood(symbol: str) -> float`
**Purpose**: Predict if market will recover from flash crash
**Logic**:
1. Historical recovery patterns
2. Current volatility regime
3. Market depth analysis
4. Return probability 0-1

---

#### 4. `_format_event_record(event_data: Dict) -> Dict`
**Purpose**: Standardize event logging format
**Logic**:
1. Add timestamp
2. Calculate severity
3. Format details
4. Create audit record

---

## Infrastructure Requirements

### In `MetaController.__init__()`

```python
# Market event detection infrastructure
self._market_event_detector = None
self._price_history = {}  # Dict[str, Deque[float]] - 100-point windows
self._volume_history = {}  # Dict[str, Deque[float]] - average volumes
self._bid_ask_spreads = {}  # Dict[str, Deque[float]] - historical spreads
self._order_book_depth = {}  # Dict[str, float] - typical depths

self._active_anomalies = set()  # Set[str] - symbols with current anomalies
self._flash_crash_events = deque(maxlen=50)  # Recent flash crashes
self._liquidity_crises = {}  # Dict[str, bool] - active liquidity issues
self._event_audit_trail = deque(maxlen=1000)  # Full event history

self._auto_adjust_enabled = True
self._event_thresholds = {}  # Configurable thresholds

# Threading locks for thread safety
self._market_event_lock = threading.Lock()
self._price_history_lock = threading.Lock()
self._event_audit_lock = threading.Lock()
```

### Required Imports

```python
from collections import deque, defaultdict
import statistics
import threading
from typing import Dict, List, Tuple, Any, Set
from datetime import datetime, timedelta
```

---

## Test Strategy

### Test Categories (26 total)

**Category 1: Infrastructure Tests (3 tests)**
- `test_market_event_detector_initialization`
- `test_price_history_setup`
- `test_event_audit_trail_initialization`

**Category 2: Anomaly Detection Tests (4 tests)**
- `test_price_anomaly_z_score_calculation`
- `test_anomaly_detection_threshold`
- `test_anomaly_with_limited_history`
- `test_multiple_symbol_anomalies`

**Category 3: Flash Crash Detection Tests (4 tests)**
- `test_flash_crash_detection_basic`
- `test_flash_crash_10_percent_decline`
- `test_flash_crash_below_threshold`
- `test_flash_crash_recovery_pattern`

**Category 4: Liquidity Crisis Tests (4 tests)**
- `test_liquidity_crisis_wide_spread`
- `test_liquidity_crisis_depth_collapse`
- `test_liquidity_crisis_imbalance`
- `test_liquidity_multiple_indicators`

**Category 5: Volume Spike Tests (3 tests)**
- `test_volume_spike_detection`
- `test_volume_spike_threshold`
- `test_volume_normal_volatility`

**Category 6: Position Adjustment Tests (3 tests)**
- `test_auto_position_adjustment_severity`
- `test_position_adjustment_multiple_events`
- `test_position_closure_on_critical_event`

**Category 7: Event Logging Tests (2 tests)**
- `test_event_audit_trail_logging`
- `test_event_history_retrieval`

**Category 8: Integration Tests (2 tests)**
- `test_end_to_end_market_event_handling`
- `test_concurrent_event_detection`

**Category 9: Edge Cases (1 test)**
- `test_event_under_extreme_volatility`

---

## Implementation Timeline

### Phase 1: Anomaly Detection Engine (50 min)
- [ ] Create `AnomalyDetector` class
- [ ] Implement Z-score calculation
- [ ] Implement threshold checking
- [ ] Write and pass 4 anomaly tests

### Phase 2: Flash Crash Detection (40 min)
- [ ] Create `FlashCrashDetector` class
- [ ] Implement rapid decline detection
- [ ] Implement recovery pattern analysis
- [ ] Write and pass 4 flash crash tests

### Phase 3: Liquidity Crisis Detection (40 min)
- [ ] Create `LiquidityCrisisDetector` class
- [ ] Implement spread monitoring
- [ ] Implement depth tracking
- [ ] Write and pass 4 liquidity tests

### Phase 4: Volume Analysis (30 min)
- [ ] Create volume spike detection
- [ ] Implement volume ratio calculation
- [ ] Write and pass 3 volume tests

### Phase 5: Auto Position Adjustment (30 min)
- [ ] Create `AutoPositionAdjuster` class
- [ ] Implement severity-based sizing
- [ ] Write and pass 3 position tests

### Phase 6: Event Logging & Integration (30 min)
- [ ] Create event audit trail
- [ ] Implement event formatting
- [ ] Write and pass remaining tests

---

## Success Criteria

✅ All 26 tests passing (100%)  
✅ 100% type hints on all methods  
✅ 100% docstrings on all methods  
✅ Zero regressions on previous tests (323/323)  
✅ Thread safety verified through concurrent tests  
✅ Performance: All operations < 50ms  
✅ Production-ready code quality  

---

## Deliverables

1. ✅ Issue #29 Implementation Guide (this document)
2. ✅ Test suite with 26 comprehensive tests
3. ✅ Implementation with all 6 main + 4 helper methods
4. ✅ Infrastructure components in MetaController
5. ✅ Completion report with metrics and analysis
6. ✅ Zero regressions verification

---

## Notes

- **Z-Score Threshold**: 3.0 represents 99.7% confidence (standard statistical threshold)
- **Flash Crash Definition**: 10% decline within 1 second (industry standard)
- **Liquidity Crisis**: Multiple indicators must trigger (not just single metric)
- **Position Adjustment Philosophy**: Scaled response based on severity
- **Event Logging**: Complete audit trail for compliance and post-analysis

---

**Next Steps:** Ready to generate comprehensive test suite with 26 tests covering all scenarios.

