"""
Native L0: Retry manager with exponential backoff (replaces 100-line legacy RetryManager)

Handles retries with exponential backoff and optional jitter for API calls.
"""

import asyncio
import random
from collections.abc import Callable
from typing import Any


class NativeRetryManager:
    """Simple async retry with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_sec: float = 0.1,
        max_delay_sec: float = 10.0,
        jitter: bool = True,
        backoff_multiplier: float = 2.0,
    ):
        """
        Initialize retry manager

        Args:
            max_attempts: Maximum number of attempts (including first try)
            base_delay_sec: Initial delay in seconds
            max_delay_sec: Maximum delay in seconds
            jitter: Whether to add random jitter to delays
            backoff_multiplier: Exponential backoff multiplier
        """
        self.max_attempts = max_attempts
        self.base_delay_sec = base_delay_sec
        self.max_delay_sec = max_delay_sec
        self.jitter = jitter
        self.backoff_multiplier = backoff_multiplier

    async def call(self, coro_func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with retries

        Args:
            coro_func: Async function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Return value from function

        Raises:
            Last exception if all attempts fail
        """
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await coro_func(*args, **kwargs)

            except Exception as e:
                last_error = e

                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    await asyncio.sleep(delay)

        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for attempt using exponential backoff

        Args:
            attempt: Attempt number (1-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base * multiplier^(attempt-1)
        delay = self.base_delay_sec * (self.backoff_multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay_sec)

        # Add jitter (randomize between 50-100% of delay)
        if self.jitter:
            delay *= 0.5 + random.random()

        return delay

    async def call_with_fallback(
        self, coro_func: Callable, fallback_value: Any, *args, **kwargs
    ) -> Any:
        """
        Execute with retries, return fallback on failure

        Args:
            coro_func: Async function to call
            fallback_value: Value to return on all failures
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or fallback value
        """
        try:
            return await self.call(coro_func, *args, **kwargs)
        except Exception:
            return fallback_value

    async def call_with_default(
        self, coro_func: Callable, default_value: Any, *args, **kwargs
    ) -> Any:
        """
        Alias for call_with_fallback (clearer naming)

        Args:
            coro_func: Async function to call
            default_value: Value to return on all failures
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or default value
        """
        return await self.call_with_fallback(coro_func, default_value, *args, **kwargs)


# Predefined retry managers for common scenarios
RETRY_FAST = NativeRetryManager(
    max_attempts=2,
    base_delay_sec=0.05,
    max_delay_sec=1.0,
)

RETRY_STANDARD = NativeRetryManager(
    max_attempts=3,
    base_delay_sec=0.1,
    max_delay_sec=5.0,
)

RETRY_AGGRESSIVE = NativeRetryManager(
    max_attempts=5,
    base_delay_sec=0.1,
    max_delay_sec=10.0,
)

RETRY_NO_JITTER = NativeRetryManager(
    max_attempts=3,
    base_delay_sec=0.1,
    max_delay_sec=5.0,
    jitter=False,
)
