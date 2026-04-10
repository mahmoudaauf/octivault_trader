"""
Exponential backoff retry logic for balance synchronization.

This module implements smart retry logic for balance sync operations,
preventing rate limit errors from causing cascading API call storms.

When a rate limit is hit (APIError 1003), the system:
1. Backs off exponentially (1s → 2s → 4s → 8s → 16s → 30s max)
2. Retries up to 3 times
3. Falls back to cached balance if all retries fail
4. Automatically resets when balance sync succeeds

This prevents the "thundering herd" problem where multiple components
independently calling sync_authoritative_balance(force=True) create
a positive feedback loop of rate limit errors.
"""

import asyncio
import logging
import time
from typing import Callable, Optional, Any, Awaitable
try:
    from binance.exceptions import BinanceAPIException
except Exception:  # pragma: no cover - fallback for stripped environments
    class BinanceAPIException(Exception):
        pass


class BalanceSyncRetryManager:
    """
    Manages retry logic for balance synchronization with exponential backoff.
    
    Used by components that need authoritative balance data:
    - MetaController (decision points, execution gates)
    - CapitalGovernor (capital floor checks)
    - ExecutionManager (order execution)
    - RotationAuthority (position rotation)
    - ExchangeAuditor (balance audits)
    """

    def __init__(self, logger: logging.Logger, component_name: str = "BalanceSyncRetry"):
        """
        Initialize the retry manager.
        
        Args:
            logger: Logger instance
            component_name: Name of the component using this manager
        """
        self.logger = logger
        self.component_name = component_name
        
        # Retry configuration
        self.initial_backoff_sec = 1.0  # Start with 1 second
        self.max_backoff_sec = 30.0     # Cap at 30 seconds
        self.backoff_multiplier = 2.0   # Double each retry
        self.max_retries = 3            # Up to 3 attempts
        
        # State tracking
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.last_rate_limit_time = None
        self.last_rate_limit_code = None
        self.time_in_backoff = 0.0

    async def execute_with_backoff(
        self,
        sync_fn: Callable[..., Awaitable[Any]],
        call_point: str = "unknown",
        force: bool = False
    ) -> bool:
        """
        Execute balance sync with exponential backoff on rate limit errors.

        Args:
            sync_fn: Async function to call (typically shared_state.sync_authoritative_balance)
            call_point: Name of the call location (e.g., "MetaController:evaluate_signals")
            force: Whether to force sync (bypass 300s throttle)

        Returns:
            True if successful, False if failed after all retries
            
        Behavior:
        - On success: Returns True, resets consecutive_failures counter
        - On rate limit: Backs off 1→2→4→8→16→30s and retries up to 3 times
        - On rate limit after max retries: Returns False, increments failure counter
        - On other errors: Returns False immediately (no retry)
        """
        attempt = 0
        backoff_sec = self.initial_backoff_sec
        
        while attempt < self.max_retries:
            attempt += 1
            
            try:
                # Execute the sync function
                await sync_fn(force=force)
                
                # SUCCESS! Reset all failure tracking
                self.consecutive_failures = 0
                self.time_in_backoff = 0.0
                
                if attempt > 1:
                    # Only log if this was a retry
                    self.logger.info(
                        f"[BalanceSync] {self.component_name}:{call_point}: "
                        f"Recovered after {attempt} attempts. Rate limit cleared."
                    )
                
                return True

            except BinanceAPIException as e:
                is_rate_limit = self._is_rate_limit_error(e)
                if is_rate_limit:
                    self.last_rate_limit_code = 1003
                    self.last_rate_limit_time = time.time()
                    self.consecutive_failures += 1
                    
                    # If this is the last attempt, give up
                    if attempt >= self.max_retries:
                        self.logger.warning(
                            f"[BalanceSync] {self.component_name}:{call_point}: "
                            f"Rate limit persisted after {self.max_retries} attempts. "
                            f"Giving up. Consecutive failures: {self.consecutive_failures}. "
                            f"Using cached balance."
                        )
                        
                        # If too many consecutive failures, increase backoff window
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            self.logger.error(
                                f"[BalanceSync] {self.component_name}: "
                                f"Multiple consecutive failures ({self.consecutive_failures}). "
                                f"System may be rate limited globally. Recommend: "
                                f"1) Reduce polling frequency, "
                                f"2) Check concurrent API calls, "
                                f"3) Restart bot if issue persists."
                            )
                        
                        return False
                    
                    # Still have retries left - back off and try again
                    self.logger.warning(
                        f"[BalanceSync] {self.component_name}:{call_point}: "
                        f"Rate limit hit (attempt {attempt}/{self.max_retries}). "
                        f"Backing off {backoff_sec:.1f}s before retry..."
                    )
                    
                    self.time_in_backoff += backoff_sec
                    await asyncio.sleep(backoff_sec)
                    
                    # Increase backoff for next iteration
                    backoff_sec = min(
                        backoff_sec * self.backoff_multiplier,
                        self.max_backoff_sec
                    )
                    continue
                
                # === NON-RATE-LIMIT ERROR ===
                else:
                    self.logger.error(
                        f"[BalanceSync] {self.component_name}:{call_point}: "
                        f"Non-rate-limit error (attempt {attempt}/{self.max_retries}): {e}. "
                        f"Not retrying."
                    )
                    self.consecutive_failures = 0  # Reset on non-rate-limit error
                    return False

            except asyncio.CancelledError:
                # Task was cancelled, propagate
                self.logger.warning(
                    f"[BalanceSync] {self.component_name}:{call_point}: "
                    f"Sync cancelled (attempt {attempt}/{self.max_retries})."
                )
                raise
            
            except Exception as e:
                if self._is_rate_limit_error(e):
                    self.last_rate_limit_code = 1003
                    self.last_rate_limit_time = time.time()
                    self.consecutive_failures += 1
                    if attempt >= self.max_retries:
                        self.logger.warning(
                            f"[BalanceSync] {self.component_name}:{call_point}: "
                            f"Rate limit persisted after {self.max_retries} attempts "
                            f"(generic exception path). Using cached balance."
                        )
                        return False
                    self.logger.warning(
                        f"[BalanceSync] {self.component_name}:{call_point}: "
                        f"Rate limit detected (generic exception path) "
                        f"(attempt {attempt}/{self.max_retries}). "
                        f"Backing off {backoff_sec:.1f}s..."
                    )
                    self.time_in_backoff += backoff_sec
                    await asyncio.sleep(backoff_sec)
                    backoff_sec = min(backoff_sec * self.backoff_multiplier, self.max_backoff_sec)
                    continue

                # Unexpected non-rate-limit error - don't retry
                self.logger.error(
                    f"[BalanceSync] {self.component_name}:{call_point}: "
                    f"Unexpected exception (attempt {attempt}/{self.max_retries}): {type(e).__name__}: {e}. "
                    f"Not retrying."
                )
                self.consecutive_failures = 0  # Reset on unexpected error
                return False
        
        # Should not reach here, but just in case
        self.logger.error(
            f"[BalanceSync] {self.component_name}:{call_point}: "
            f"Exhausted retries ({self.max_retries}). Giving up."
        )
        return False

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Best-effort Binance rate-limit detection across wrapper variants."""
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        status = getattr(exc, "status", None)
        msg = str(exc).lower()
        return (
            code in (-1003, -1015, 429)
            or status_code in (429, -1003, -1015)
            or status in (429, -1003, -1015)
            or "apierror(code=-1003)" in msg
            or "too much request weight" in msg
            or "request weight used" in msg
            or "too many requests" in msg
        )

    def reset(self) -> None:
        """Manually reset retry state (called after successful operations)."""
        self.consecutive_failures = 0
        self.time_in_backoff = 0.0
        self.last_rate_limit_time = None

    def is_in_backoff(self) -> bool:
        """Check if system is currently backing off from rate limits."""
        return self.consecutive_failures > 0

    def get_status(self) -> dict:
        """
        Get current retry manager status.
        
        Returns:
            Dict with current state:
            - is_in_backoff: Whether we're currently experiencing rate limits
            - consecutive_failures: Number of consecutive failures
            - last_rate_limit_time: Timestamp of last rate limit error
            - total_backoff_time: Total time spent backing off
        """
        return {
            "is_in_backoff": self.is_in_backoff(),
            "consecutive_failures": self.consecutive_failures,
            "last_rate_limit_time": self.last_rate_limit_time,
            "last_rate_limit_code": self.last_rate_limit_code,
            "total_backoff_time_sec": self.time_in_backoff,
        }


class BalanceSyncCoordinator:
    """
    Coordinates balance sync across multiple components to prevent thundering herd.
    
    Problem: When multiple independent components (MetaController, CapitalGovernor,
    ExecutionManager, RotationAuthority, ExchangeAuditor) all call
    sync_authoritative_balance(force=True) within the same evaluation cycle,
    they create a "thundering herd" of 5-10+ concurrent API calls, overwhelming
    the rate limit.
    
    Solution: Use a shared coordinator that serializes critical balance syncs
    and prevents duplicate concurrent requests.
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize the coordinator."""
        self.logger = logger
        self._sync_lock = asyncio.Lock()
        self._inflight_task: Optional[asyncio.Task] = None
        self._last_successful_sync = 0.0
        self._cache_ttl_sec = 30.0  # Cache balance for 30 seconds
    
    async def sync_authoritative_balance_coordinated(
        self,
        sync_fn: Callable[..., Awaitable[Any]],
        component_name: str,
        call_point: str,
        force: bool = False,
        use_cache: bool = True
    ) -> bool:
        """
        Coordinate balance sync across components to prevent thundering herd.
        
        Args:
            sync_fn: The actual sync function (shared_state.sync_authoritative_balance)
            component_name: Component requesting sync (e.g., "MetaController")
            call_point: Location in component (e.g., "evaluate_signals")
            force: Whether to force sync
            use_cache: Whether to use cached balance if recent
        
        Returns:
            True if sync successful or cache valid, False if failed
        """
        now = time.time()
        if use_cache and not force and (now - self._last_successful_sync) < self._cache_ttl_sec:
            self.logger.debug(
                f"[BalanceSyncCoord] {component_name}:{call_point}: "
                f"Using cached balance (age: {now - self._last_successful_sync:.1f}s)"
            )
            return True

        async with self._sync_lock:
            now = time.time()
            if use_cache and not force and (now - self._last_successful_sync) < self._cache_ttl_sec:
                return True

            if self._inflight_task and not self._inflight_task.done():
                task = self._inflight_task
                self.logger.debug(
                    f"[BalanceSyncCoord] {component_name}:{call_point}: joining in-flight sync"
                )
            else:
                async def _runner() -> bool:
                    try:
                        await sync_fn(force=force)
                        return True
                    except Exception as e:
                        self.logger.error(
                            f"[BalanceSyncCoord] {component_name}:{call_point}: Sync failed: {e}"
                        )
                        return False
                task = asyncio.create_task(_runner())
                self._inflight_task = task

        result = await task

        async with self._sync_lock:
            if self._inflight_task is task:
                self._inflight_task = None
            if result:
                self._last_successful_sync = time.time()
        return bool(result)
