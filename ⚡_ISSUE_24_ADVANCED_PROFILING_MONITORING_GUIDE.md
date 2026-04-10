# ⚡ Issue #24: Advanced Profiling & Monitoring - Implementation Guide

**Status:** 🚀 READY TO START  
**Priority:** HIGH  
**Estimated Effort:** 2 hours  
**Expected Tests:** ~20 (target 100%)  
**Timeline:** April 11, 2026 (2 PM - 4 PM)  
**Prerequisite:** Issues #21-23 (✅ ALL COMPLETE)  
**Current Sprint Progress:** 3/5 (60%) → Target: 4/5 (80%)

---

## 🎯 Overview

Issue #24 adds **deep performance analysis tools** to the MetaController, enabling continuous CPU profiling, memory leak detection, and bottleneck identification in production.

**Key Deliverables:**
- 6 methods for profiling infrastructure
- CPU profiling with per-component breakdown
- Memory tracking with trend analysis
- Real-time bottleneck detection
- Comprehensive performance dashboards
- ~400 lines of code

**Expected Outcomes:**
- ✅ 20/20 tests passing
- ✅ 0 regressions (310→330 cumulative tests)
- ✅ Production-ready profiling system
- ✅ Performance insights dashboard

---

## 📊 Architecture: Phase-Based Implementation

### Phase 1: CPU Profiling Infrastructure (Method 1)
**Scope:** Setup and execution of CPU profiling

**Method 1: `_start_cpu_profiler()`**
```python
def _start_cpu_profiler(self, output_dir: str = "./profiles") -> None:
    """
    Start CPU profiling with cProfile.
    
    Creates periodic sampling to avoid blocking main loop.
    Stores results to disk for analysis.
    
    Args:
        output_dir: Directory for profile outputs
    
    Stores:
        - self._cpu_profiler: cProfile.Profile instance
        - self._profile_output_dir: Directory path
        - self._profile_start_time: Timestamp
        - self._profile_frames: Frame samples list
    """
```

**Implementation Approach:**
- Use `cProfile.Profile()` for CPU timing
- Start/stop profiling around main loop iteration
- Collect function call statistics
- Store frame samples for trend analysis

**Infrastructure Added:**
```python
self._cpu_profiler: Optional[cProfile.Profile] = None
self._profile_output_dir: str = "./profiles"
self._profile_start_time: Optional[float] = None
self._profile_frames: List[Dict[str, Any]] = []
self._profile_lock = threading.Lock()
```

---

### Phase 2: Memory Tracking with Trends (Methods 2-3)
**Scope:** Memory monitoring and leak detection

**Method 2: `_track_memory_usage()`**
```python
def _track_memory_usage(self) -> None:
    """
    Track memory consumption and detect leaks.
    
    Records:
    - Process memory (RSS, VMS)
    - Object counts (Python objects)
    - Garbage collection stats
    - Memory growth trends
    
    Stores:
        - self._memory_samples: List of (timestamp, memory_mb)
        - self._memory_metrics: Current memory statistics
        - self._memory_growth_trend: EMA of growth rate
    """
```

**Memory Metrics to Track:**
```python
memory_data = {
    "timestamp": time.time(),
    "rss_mb": process.memory_info().rss / 1024 / 1024,
    "vms_mb": process.memory_info().vms / 1024 / 1024,
    "percent": process.memory_percent(),
    "object_count": len(gc.get_objects()),
    "gc_collections": gc.get_count(),
}
```

**Method 3: `get_memory_trend_analysis()`**
```python
def get_memory_trend_analysis(self) -> Dict[str, Any]:
    """
    Analyze memory trends for leak detection.
    
    Returns:
        {
            "current_mb": float,
            "peak_mb": float,
            "trend": "stable" | "growing" | "shrinking",
            "growth_rate_mb_per_min": float,
            "leak_risk": "low" | "medium" | "high",
            "samples_in_window": int,
        }
    
    Leak Detection Logic:
    - Calculate growth rate (MB/min)
    - Compare to baseline growth
    - Flag if consistent positive trend
    """
```

**Infrastructure Added:**
```python
self._memory_samples: deque = deque(maxlen=1000)  # Last 1000 samples
self._memory_metrics: Dict[str, Any] = {}
self._memory_growth_trend: float = 0.0
self._memory_baseline: Optional[float] = None
self._memory_lock = threading.Lock()
```

---

### Phase 3: Bottleneck Detection (Methods 4-5)
**Scope:** Identify performance hotspots

**Method 4: `_identify_bottlenecks()`**
```python
def _identify_bottlenecks(self) -> Dict[str, Dict[str, Any]]:
    """
    Identify performance bottlenecks across system.
    
    Analyzes:
    - Signal processing latency (avg, p95, p99)
    - Guard evaluation time by gate
    - Decision making latency
    - Execution latency
    - Loop cycle time components
    
    Returns:
        {
            "signal_processing": {
                "avg_ms": float,
                "p95_ms": float,
                "p99_ms": float,
                "count": int,
            },
            "guard_evaluation": {
                "gate_name": {...metrics...}
            },
            "decision_making": {...},
            "execution": {...},
            "cycle_breakdown": {
                "evaluation": float,
                "decision": float,
                "execution": float,
                "overhead": float,
                "total": float,
            }
        }
    """
```

**Method 5: `get_performance_hotspots()`**
```python
def get_performance_hotspots(
    self, 
    threshold_percentile: float = 75.0
) -> List[Dict[str, Any]]:
    """
    Get top performance hotspots (P75+).
    
    Returns list of:
    {
        "component": str,
        "operation": str,
        "latency_ms": float,
        "count": int,
        "total_time": float,
        "percentage": float,
        "recommendation": str,
    }
    
    Sorted by total_time descending.
    """
```

**Infrastructure Added:**
```python
self._bottleneck_metrics: Dict[str, List[float]] = defaultdict(list)
self._hotspot_history: deque = deque(maxlen=100)
self._hotspot_lock = threading.Lock()
```

---

### Phase 4: Continuous Profiling Dashboard (Methods 6)
**Scope:** Real-time performance visualization

**Method 6: `generate_profiling_report()`**
```python
def generate_profiling_report(self) -> Dict[str, Any]:
    """
    Generate comprehensive profiling report.
    
    Combines all profiling data into actionable dashboard.
    
    Returns:
        {
            "timestamp": float,
            "duration_seconds": float,
            "cpu": {
                "profiler_overhead": float,
                "top_functions": [...],
                "samples_collected": int,
            },
            "memory": {
                "current_mb": float,
                "trend_analysis": {...},
                "leak_risk": str,
            },
            "bottlenecks": {
                "signal_processing": {...},
                "guard_evaluation": {...},
                "decision_making": {...},
                "execution": {...},
            },
            "hotspots": [...],
            "recommendations": [...],
            "health_score": float,  # 0-100
        }
    """
```

**Report Structure:**
```python
{
    "summary": {
        "overall_performance": "healthy" | "degraded" | "critical",
        "health_score": 85.5,
        "recommendation": "Consider profiling guard evaluation",
    },
    "cpu_profile": {...},
    "memory_profile": {...},
    "latency_breakdown": {...},
    "hotspots": [...],
    "trends": {...},
    "timestamp": "2026-04-11T14:30:00Z",
}
```

**Infrastructure Added:**
```python
self._report_history: deque = deque(maxlen=50)
self._profiling_active: bool = False
self._profile_start_time: float = 0.0
```

---

## 🧪 Test Strategy (20 tests)

### Test Categories:

**CPU Profiling Tests (3)**
```python
test_start_cpu_profiler_exists()
test_start_cpu_profiler_creates_directory()
test_start_cpu_profiler_initializes_profile()
```

**Memory Tracking Tests (4)**
```python
test_track_memory_usage_exists()
test_track_memory_usage_records_samples()
test_track_memory_usage_calculates_metrics()
test_get_memory_trend_analysis_returns_dict()
```

**Bottleneck Detection Tests (4)**
```python
test_identify_bottlenecks_exists()
test_identify_bottlenecks_returns_dict()
test_get_performance_hotspots_exists()
test_get_performance_hotspots_returns_list()
```

**Dashboard & Reporting Tests (4)**
```python
test_generate_profiling_report_exists()
test_generate_profiling_report_returns_dict()
test_profiling_report_has_all_sections()
test_profiling_report_health_score_in_range()
```

**Integration Tests (5)**
```python
test_profiling_data_structures_initialized()
test_profiling_methods_use_locks()
test_profiling_report_timestamp_recent()
test_profiling_overhead_minimal()
test_profiling_memory_bounded()
```

---

## 📋 Implementation Checklist

### Step 1: Infrastructure Setup (15 min)
- [ ] Add import statements (cProfile, psutil, gc, memory_profiler)
- [ ] Create data structure dictionaries and lists
- [ ] Add threading locks for thread safety
- [ ] Initialize in `__init__()` method

### Step 2: Phase 1 - CPU Profiling (25 min)
- [ ] Implement `_start_cpu_profiler()` method
- [ ] Add profile directory creation logic
- [ ] Implement profile frame collection
- [ ] Add cProfile integration

### Step 3: Phase 2 - Memory Tracking (25 min)
- [ ] Implement `_track_memory_usage()` method
- [ ] Add memory sampling logic
- [ ] Implement `get_memory_trend_analysis()` method
- [ ] Add leak detection algorithm

### Step 4: Phase 3 - Bottleneck Detection (25 min)
- [ ] Implement `_identify_bottlenecks()` method
- [ ] Add per-component latency tracking
- [ ] Implement `get_performance_hotspots()` method
- [ ] Add threshold-based filtering

### Step 5: Phase 4 - Dashboard & Reporting (20 min)
- [ ] Implement `generate_profiling_report()` method
- [ ] Add report structure and formatting
- [ ] Add health score calculation
- [ ] Add recommendations generation

### Step 6: Testing (30 min)
- [ ] Write 20 comprehensive tests
- [ ] Achieve 100% test pass rate
- [ ] Verify no regressions (310/310 pass)
- [ ] Run full test suite

---

## 🔧 Key Implementation Details

### Imports Required
```python
import cProfile
import pstats
import io
import gc
import psutil
import threading
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple
```

### Data Structures
```python
# CPU Profiling
self._cpu_profiler: Optional[cProfile.Profile] = None
self._profile_frames: List[Dict[str, Any]] = []
self._profile_lock = threading.Lock()

# Memory Tracking
self._memory_samples: deque = deque(maxlen=1000)
self._memory_metrics: Dict[str, Any] = {}
self._memory_lock = threading.Lock()

# Bottleneck Detection
self._bottleneck_metrics: Dict[str, List[float]] = defaultdict(list)
self._hotspot_history: deque = deque(maxlen=100)
self._hotspot_lock = threading.Lock()

# Dashboard
self._report_history: deque = deque(maxlen=50)
self._profiling_active: bool = False
```

### Critical Patterns

**Thread Safety (All operations):**
```python
def _track_memory_usage(self) -> None:
    try:
        with self._memory_lock:
            # Collect memory data
            sample = self._collect_memory_sample()
            self._memory_samples.append(sample)
            self._memory_metrics = self._calculate_memory_stats()
    except Exception as e:
        logger.error(f"Memory tracking error: {e}")
```

**Efficient Sampling:**
```python
# Use deque with maxlen for fixed-size circular buffer
self._memory_samples: deque = deque(maxlen=1000)

# Prevents unbounded memory growth
# Automatically pops oldest when full
```

**Percentile Calculation:**
```python
def _calculate_percentiles(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {}
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    return {
        "p50": sorted_samples[int(n * 0.50)],
        "p95": sorted_samples[int(n * 0.95)],
        "p99": sorted_samples[int(n * 0.99)],
    }
```

**Trend Detection:**
```python
def _calculate_trend(samples: List[float]) -> str:
    if len(samples) < 3:
        return "insufficient_data"
    
    # Calculate slope using linear regression
    recent = samples[-10:]
    slope = (recent[-1] - recent[0]) / len(recent)
    
    if slope > 0.5:
        return "growing"
    elif slope < -0.5:
        return "shrinking"
    else:
        return "stable"
```

---

## 📈 Performance Targets

**Profiling Overhead:**
- CPU profiling: < 2% overhead
- Memory tracking: < 1% overhead
- Bottleneck detection: < 1% overhead
- Report generation: < 100ms

**Data Storage:**
- Memory samples: 1000 max (≤10MB)
- Bottleneck metrics: Bounded (defaultdict)
- Hotspot history: 100 max reports
- Report history: 50 max reports
- **Total overhead: ≤20MB**

**Report Generation:**
- Should complete in < 100ms
- All sections populated
- Recommendations generated

---

## ✅ Success Criteria

**Code Quality:**
- ✅ All 6 methods implemented with full docstrings
- ✅ Type hints on all parameters and returns
- ✅ Comprehensive error handling
- ✅ Thread-safe throughout (locks on all shared state)

**Testing:**
- ✅ 20/20 tests passing (100%)
- ✅ All method existence tests passing
- ✅ All infrastructure tests passing
- ✅ All integration tests passing
- ✅ Zero regressions (310/310 total pass)

**Performance:**
- ✅ < 2% CPU profiling overhead
- ✅ < 1% memory tracking overhead
- ✅ < 100MB total memory usage
- ✅ Report generation < 100ms

**Integration:**
- ✅ Works with existing Issue #21-23 systems
- ✅ Metrics properly initialized
- ✅ Thread locks don't deadlock
- ✅ Exception handling prevents crashes

---

## 📝 File Modifications

**Primary File: `core/meta_controller.py`**
- Add 6 new methods (~400 lines)
- Add 5 new data structures in `__init__`
- Import cProfile, psutil, gc
- Integrate with existing APM system

**Test File: `tests/test_issue_24_advanced_profiling.py` (NEW)**
- Create comprehensive test suite (~500 lines)
- 20 test cases covering all scenarios
- Tests for infrastructure, methods, and integration

---

## 🎯 Timeline (2 hours)

```
0:00 - 0:10  Read architecture & understand MetaController
0:10 - 0:25  Implement Phase 1 (CPU Profiling)
0:25 - 0:50  Implement Phase 2 (Memory Tracking)
0:50 - 1:15  Implement Phase 3 (Bottleneck Detection)
1:15 - 1:35  Implement Phase 4 (Dashboard & Reporting)
1:35 - 1:50  Run tests & fix issues
1:50 - 2:00  Documentation & verification
```

---

## 🚀 Next Steps

After Issue #24 completion:
1. **Issue #25:** Production Scaling Validation (~2.5 hours)
   - Load testing scenarios
   - Horizontal scaling tests
   - Database connection pooling
   - Resource limit validation

2. **Sprint 2 Complete:** All 30 issues done
   - 310+ tests passing
   - 40+ days ahead of schedule
   - Production-ready v2.0

---

## 📚 References

**Related Issues:**
- Issue #21: MetaController Loop Optimization (145 tests ✅)
- Issue #22: Guard Evaluation Parallelization (54 tests ✅)
- Issue #23: Signal Processing Pipeline (67 tests ✅)

**Documentation:**
- APM Integration: Issue #19 (21 tests)
- Health Monitoring: Issue #20 (completed)
- Performance Recommendations: Comprehensive Code Review Plan

---

**Generated:** April 11, 2026 | Status: READY TO IMPLEMENT

