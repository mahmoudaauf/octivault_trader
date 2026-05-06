"""
Native L0: Lightweight time utilities (replaces 150-line legacy TimeUtils)

Provides essential time operations with minimal overhead.
Static methods (no instance creation).
"""

import time
from datetime import datetime, timezone


class NativeTimeUtils:
    """Lightweight time utilities as static methods"""

    @staticmethod
    def unix_now_ms() -> int:
        """
        Current Unix timestamp (milliseconds)

        Returns:
            int: Milliseconds since epoch
        """
        return int(time.time() * 1000)

    @staticmethod
    def unix_now_s() -> float:
        """
        Current Unix timestamp (seconds)

        Returns:
            float: Seconds since epoch (with sub-second precision)
        """
        return time.time()

    @staticmethod
    def iso_now() -> str:
        """
        ISO8601 timestamp (UTC)

        Returns:
            str: Format: 2026-05-06T12:34:56.789012+00:00
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def align_candle_time(unix_ms: int, interval_sec: int) -> int:
        """
        Align timestamp to candle boundary

        Args:
            unix_ms: Unix timestamp in milliseconds
            interval_sec: Interval in seconds (60 for 1m, 300 for 5m, etc)

        Returns:
            int: Aligned Unix timestamp (milliseconds)

        Example:
            >>> unix_ms = 1000000
            >>> NativeTimeUtils.align_candle_time(unix_ms, 60)  # 1-minute candle
            >>> # Returns timestamp at start of minute
        """
        interval_ms = interval_sec * 1000
        return (unix_ms // interval_ms) * interval_ms

    @staticmethod
    def seconds_until_next_candle(interval_sec: int) -> float:
        """
        Seconds until next candle opens

        Args:
            interval_sec: Interval in seconds

        Returns:
            float: Seconds until next candle (0.0 to interval_sec)

        Example:
            >>> NativeTimeUtils.seconds_until_next_candle(60)
            >>> # Returns 30.5 (roughly 30 seconds until next minute)
        """
        now_ms = NativeTimeUtils.unix_now_ms()
        next_candle_ms = NativeTimeUtils.align_candle_time(now_ms, interval_sec) + (
            interval_sec * 1000
        )
        return (next_candle_ms - now_ms) / 1000.0

    @staticmethod
    def format_duration_sec(seconds: float) -> str:
        """
        Format duration in human-readable format

        Args:
            seconds: Duration in seconds

        Returns:
            str: Formatted duration (e.g., "1h 30m 45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    @staticmethod
    def is_market_hours(_unix_ms: int) -> bool:
        """
        Check if timestamp is during market hours (24/7 crypto)

        Args:
            _unix_ms: Unix timestamp in milliseconds (unused — crypto is 24/7)

        Returns:
            bool: Always True for crypto (24/7 markets)
        """
        # Crypto markets are 24/7, so always true
        return True

    @staticmethod
    def unix_ms_to_datetime(unix_ms: int) -> datetime:
        """
        Convert Unix milliseconds to datetime

        Args:
            unix_ms: Unix timestamp in milliseconds

        Returns:
            datetime: UTC datetime object
        """
        return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)

    @staticmethod
    def datetime_to_unix_ms(dt: datetime) -> int:
        """
        Convert datetime to Unix milliseconds

        Args:
            dt: datetime object

        Returns:
            int: Unix timestamp in milliseconds
        """
        return int(dt.timestamp() * 1000)
