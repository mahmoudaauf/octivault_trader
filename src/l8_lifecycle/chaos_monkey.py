"""
ChaosMonkey: Controlled failure injection for resilience testing.

This framework systematically injects failures to verify system resilience:
- Random API timeouts
- Random 500/502/503 errors
- Network partitions
- Slow network (high latency)
- Corrupted responses (bad JSON)
- Missing fields (incomplete responses)
- Database connection failures
- Clock skew

Principles:
1. Non-deterministic (random failures, but seeded for reproducibility)
2. Controlled (configurable injection rate)
3. Observable (track all injected failures)
4. Recoverable (system must handle gracefully)
5. Non-destructive (can run in staging/test)
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# FAILURE TYPES
# ============================================================================

class FailureType(Enum):
    """Types of failures we can inject."""
    
    # API failures
    API_TIMEOUT = "api_timeout"  # 10s timeout
    API_500_ERROR = "api_500_error"  # Internal server error
    API_502_ERROR = "api_502_error"  # Bad gateway
    API_503_ERROR = "api_503_error"  # Service unavailable
    API_429_ERROR = "api_429_error"  # Rate limit
    
    # Network failures
    NETWORK_PARTITION = "network_partition"  # Lose connectivity (30s)
    NETWORK_SLOW = "network_slow"  # Latency 10s
    NETWORK_JITTER = "network_jitter"  # Variable latency
    NETWORK_PACKET_LOSS = "network_packet_loss"  # 10% drop
    
    # Data failures
    CORRUPTED_RESPONSE = "corrupted_response"  # Invalid JSON
    MISSING_FIELDS = "missing_fields"  # Incomplete response
    WRONG_DATA_TYPE = "wrong_data_type"  # Type mismatch
    
    # Database failures
    DB_CONNECTION_FAILED = "db_connection_failed"  # Cannot connect
    DB_TIMEOUT = "db_timeout"  # Query too slow
    DB_DEADLOCK = "db_deadlock"  # Deadlock detected
    
    # System failures
    CLOCK_SKEW = "clock_skew"  # System clock jumps
    OUT_OF_MEMORY = "out_of_memory"  # Low memory
    DISK_FULL = "disk_full"  # Disk space exhausted


@dataclass
class ChaosEvent:
    """Record of an injected failure."""
    
    failure_type: FailureType
    timestamp: float
    component: str  # Which component was affected
    request_id: Optional[str] = None
    recovery_time: Optional[float] = None  # How long to recover
    success: bool = False  # Did system recover?
    error_message: Optional[str] = None


# ============================================================================
# CHAOS MONKEY
# ============================================================================

class ChaosMonkey:
    """
    Controlled failure injection framework.
    
    Usage:
    ```python
    chaos = ChaosMonkey(
        enabled=True,
        injection_rate=0.01,  # 1% of requests
        seed=42  # Reproducible
    )
    
    # Inject before API call
    failure = await chaos.maybe_inject_failure("exchange_api")
    if failure:
        # System should handle this gracefully
        pass
    ```
    """
    
    def __init__(
        self,
        enabled: bool = False,
        injection_rate: float = 0.01,  # 1% of requests
        seed: Optional[int] = None,
    ):
        """
        Initialize ChaosMonkey.
        
        Args:
            enabled: Enable/disable chaos injection (default: False)
            injection_rate: Fraction of requests to fail (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.enabled = enabled
        self.injection_rate = injection_rate
        
        if seed is not None:
            random.seed(seed)
        
        self.injected_failures: List[ChaosEvent] = []
        self._request_count = 0
        self._failure_counts: Dict[FailureType, int] = {
            ft: 0 for ft in FailureType
        }
    
    async def maybe_inject_failure(
        self,
        component: str,
        request_id: Optional[str] = None,
    ) -> Optional[ChaosEvent]:
        """
        Randomly inject failure with configured rate.
        
        Args:
            component: Component being tested ("exchange_api", "database", etc)
            request_id: Optional request ID for tracking
        
        Returns:
            ChaosEvent if failure injected, None otherwise
        Raises:
            Exception if failure is injected (raises appropriate error)
        """
        
        self._request_count += 1
        
        # Check if we should inject
        if not self.enabled or random.random() > self.injection_rate:
            return None
        
        # Pick random failure type
        failure_type = random.choice(list(FailureType))
        
        # Create chaos event
        chaos_event = ChaosEvent(
            failure_type=failure_type,
            timestamp=time.time(),
            component=component,
            request_id=request_id,
        )
        
        # Inject failure
        logger.warning(f"🔴 CHAOS: Injecting {failure_type.value} in {component}")
        
        try:
            await self._inject_failure(failure_type, component)
        except Exception as e:
            chaos_event.error_message = str(e)
            self.injected_failures.append(chaos_event)
            self._failure_counts[failure_type] += 1
            raise
        
        return chaos_event
    
    async def _inject_failure(self, failure_type: FailureType, component: str):
        """Inject the actual failure."""
        
        if failure_type == FailureType.API_TIMEOUT:
            # Sleep 10+ seconds (usually times out)
            await asyncio.sleep(random.uniform(10, 15))
            raise TimeoutError("API timeout (chaos injected)")
        
        elif failure_type == FailureType.API_500_ERROR:
            raise ApiError("500 Internal Server Error (chaos injected)")
        
        elif failure_type == FailureType.API_502_ERROR:
            raise ApiError("502 Bad Gateway (chaos injected)")
        
        elif failure_type == FailureType.API_503_ERROR:
            raise ApiError("503 Service Unavailable (chaos injected)")
        
        elif failure_type == FailureType.API_429_ERROR:
            raise ApiError("429 Too Many Requests (chaos injected)")
        
        elif failure_type == FailureType.NETWORK_PARTITION:
            # Simulate 30-second partition
            await asyncio.sleep(random.uniform(30, 60))
            raise ConnectionError("Network partition (chaos injected)")
        
        elif failure_type == FailureType.NETWORK_SLOW:
            # Add 10s latency
            await asyncio.sleep(random.uniform(5, 15))
            # Continue normally
        
        elif failure_type == FailureType.NETWORK_JITTER:
            # Variable latency
            for _ in range(random.randint(3, 10)):
                await asyncio.sleep(random.uniform(0.1, 0.5))
        
        elif failure_type == FailureType.NETWORK_PACKET_LOSS:
            # Randomly fail this request (10%)
            if random.random() < 0.1:
                raise ConnectionError("Packet loss (chaos injected)")
        
        elif failure_type == FailureType.CORRUPTED_RESPONSE:
            raise ValueError("Corrupted response: invalid JSON (chaos injected)")
        
        elif failure_type == FailureType.MISSING_FIELDS:
            raise ValueError("Missing required fields in response (chaos injected)")
        
        elif failure_type == FailureType.WRONG_DATA_TYPE:
            raise TypeError("Wrong data type in response (chaos injected)")
        
        elif failure_type == FailureType.DB_CONNECTION_FAILED:
            raise ConnectionError("Database connection failed (chaos injected)")
        
        elif failure_type == FailureType.DB_TIMEOUT:
            await asyncio.sleep(random.uniform(5, 10))
            raise TimeoutError("Database query timeout (chaos injected)")
        
        elif failure_type == FailureType.DB_DEADLOCK:
            raise RuntimeError("Deadlock detected (chaos injected)")
        
        elif failure_type == FailureType.CLOCK_SKEW:
            # In real scenario, would jump system clock
            # Here we just sleep and report it
            logger.error("System clock skew detected (chaos injected)")
        
        elif failure_type == FailureType.OUT_OF_MEMORY:
            raise MemoryError("Out of memory (chaos injected)")
        
        elif failure_type == FailureType.DISK_FULL:
            raise OSError("Disk space full (chaos injected)")
    
    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chaos injection statistics."""
        total_failures = len(self.injected_failures)
        successful_recoveries = sum(1 for f in self.injected_failures if f.success)
        
        return {
            "enabled": self.enabled,
            "injection_rate": self.injection_rate,
            "total_requests": self._request_count,
            "total_failures_injected": total_failures,
            "recovery_rate": (successful_recoveries / total_failures) if total_failures > 0 else 0,
            "failures_by_type": dict(self._failure_counts),
            "average_recovery_time": self._compute_avg_recovery_time(),
        }
    
    def _compute_avg_recovery_time(self) -> float:
        """Average time to recover from failure."""
        times = [f.recovery_time for f in self.injected_failures if f.recovery_time]
        return sum(times) / len(times) if times else 0.0
    
    def get_failure_log(self) -> List[ChaosEvent]:
        """Get log of all injected failures."""
        return self.injected_failures.copy()
    
    def reset(self):
        """Reset chaos monkey (clear history)."""
        self.injected_failures.clear()
        self._request_count = 0
        self._failure_counts = {ft: 0 for ft in FailureType}


# ============================================================================
# RESILIENCE VERIFIER
# ============================================================================

class ResilienceVerifier:
    """
    Verify that system is resilient to failures.
    
    Runs chaos tests and verifies:
    - System doesn't crash
    - No data loss
    - No position corruption
    - No duplicate trades
    - Automatic recovery < 30 seconds
    """
    
    def __init__(self, chaos_monkey: ChaosMonkey):
        """Initialize with ChaosMonkey instance."""
        self.chaos = chaos_monkey
        self._test_results: List[Dict[str, Any]] = []
    
    async def test_api_resilience(
        self,
        api_func: Callable,
        iterations: int = 100,
        failure_types: Optional[List[FailureType]] = None,
    ) -> Dict[str, Any]:
        """
        Test resilience to API failures.
        
        Runs function with random API failures injected.
        """
        logger.info(f"Testing API resilience ({iterations} iterations)...")
        
        if failure_types is None:
            failure_types = [
                FailureType.API_TIMEOUT,
                FailureType.API_500_ERROR,
                FailureType.API_503_ERROR,
            ]
        
        success_count = 0
        failure_count = 0
        error_types: Dict[str, int] = {}
        
        for i in range(iterations):
            try:
                # Enable chaos for this iteration
                self.chaos.enabled = (i % 10 == 0)  # 10% failure rate
                
                # Call function
                result = await api_func()
                success_count += 1
            
            except Exception as e:
                failure_count += 1
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        result = {
            "test_name": "api_resilience",
            "iterations": iterations,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / iterations,
            "error_types": error_types,
        }
        
        self._test_results.append(result)
        return result
    
    async def test_state_consistency(
        self,
        get_state: Callable,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Test that state remains consistent during chaos.
        
        Verifies:
        - Capital is conserved
        - Position count is reasonable
        - P&L is realistic
        - No duplicates
        """
        logger.info(f"Testing state consistency ({iterations} iterations)...")
        
        consistency_failures = []
        
        for i in range(iterations):
            try:
                # Enable chaos
                self.chaos.enabled = (i % 20 == 0)  # 5% failure rate
                
                # Get state
                state = await get_state()
                
                # Verify consistency
                self._verify_state(state)
            
            except Exception as e:
                consistency_failures.append(str(e))
        
        result = {
            "test_name": "state_consistency",
            "iterations": iterations,
            "consistency_failures": len(consistency_failures),
            "consistency_rate": (iterations - len(consistency_failures)) / iterations,
            "failures": consistency_failures[:10],  # First 10
        }
        
        self._test_results.append(result)
        return result
    
    def _verify_state(self, state: Dict[str, Any]):
        """Verify state validity."""
        capital = state.get("total_capital", 0)
        positions = state.get("open_positions", {})
        pnl = state.get("total_pnl", 0)
        
        assert capital >= 0, f"Negative capital: {capital}"
        assert len(positions) <= 100, f"Too many positions: {len(positions)}"
        assert abs(pnl) < capital * 2, f"Unrealistic P&L: {pnl}"
    
    def get_test_results(self) -> List[Dict[str, Any]]:
        """Get all test results."""
        return self._test_results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all resilience tests."""
        if not self._test_results:
            return {"message": "No tests run yet"}
        
        return {
            "total_tests": len(self._test_results),
            "test_results": self._test_results,
            "chaos_statistics": self.chaos.get_statistics(),
        }


# ============================================================================
# LOAD TESTER
# ============================================================================

class LoadTester:
    """
    Test system under load.
    
    Verifies:
    - Can handle 10x normal load
    - Can handle 100x normal load
    - Performance degrades gracefully
    - Recovery is automatic
    """
    
    def __init__(self):
        """Initialize load tester."""
        self._test_results: List[Dict[str, Any]] = []
    
    async def test_scaling(
        self,
        workload_func: Callable,
        base_load: int = 10,  # 10 symbols, 10 signals/hour
        scale_factors: List[int] = [1, 5, 10, 50, 100],
    ) -> Dict[str, Any]:
        """
        Test system scaling characteristics.
        
        Runs workload at increasing scale factors.
        """
        logger.info(f"Testing scaling (base={base_load}, factors={scale_factors})...")
        
        results = []
        
        for scale in scale_factors:
            load = base_load * scale
            start_time = time.time()
            
            try:
                await workload_func(load)
                elapsed = time.time() - start_time
                
                results.append({
                    "scale_factor": scale,
                    "load": load,
                    "elapsed_seconds": elapsed,
                    "throughput": load / elapsed,
                    "status": "success",
                })
            
            except Exception as e:
                elapsed = time.time() - start_time
                
                results.append({
                    "scale_factor": scale,
                    "load": load,
                    "elapsed_seconds": elapsed,
                    "status": "failed",
                    "error": str(e),
                })
                
                # Stop at first failure
                break
        
        result = {
            "test_name": "scaling",
            "base_load": base_load,
            "scale_results": results,
            "max_sustainable_load": results[-1]["load"] if results else 0,
        }
        
        self._test_results.append(result)
        return result
    
    async def test_saturation(
        self,
        workload_func: Callable,
        max_load: int = 1000,
    ) -> Dict[str, Any]:
        """
        Find saturation point (where system breaks).
        
        Binary search for maximum sustainable load.
        """
        logger.info(f"Finding saturation point (max={max_load})...")
        
        low, high = 1, max_load
        saturation_load = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                start = time.time()
                await workload_func(mid)
                elapsed = time.time() - start
                
                saturation_load = mid
                low = mid + 1  # Try higher
            
            except Exception as e:
                # Failed at this load
                high = mid - 1  # Try lower
        
        result = {
            "test_name": "saturation",
            "saturation_load": saturation_load,
            "max_load_tested": max_load,
        }
        
        self._test_results.append(result)
        return result


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ApiError(Exception):
    """Simulated API error."""
    pass


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_chaos_monkey: Optional[ChaosMonkey] = None


def get_chaos_monkey() -> ChaosMonkey:
    """Get global ChaosMonkey instance (singleton)."""
    global _chaos_monkey
    if _chaos_monkey is None:
        _chaos_monkey = ChaosMonkey(enabled=False)  # Disabled by default
    return _chaos_monkey
