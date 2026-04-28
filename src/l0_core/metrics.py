"""
Quick Win Metrics Collection Module
Lightweight metrics tracking for OCTI AI BOT performance monitoring
"""

from collections import defaultdict, deque
from typing import Dict, Any, Optional
import time
import logging


class MetricsCollector:
    """
    Lightweight metrics collector for tracking system performance.
    Thread-safe for async operations.
    """
    
    def __init__(self, max_samples: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_samples = max_samples
        
        # Counters
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Gauges (current values)
        self.gauges: Dict[str, float] = {}
        
        # Histograms (time series data)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        
        # Timers (for measuring durations)
        self._timer_starts: Dict[str, float] = {}
        
        self.start_time = time.time()
        
    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[metric_name] += value
        
    def set_gauge(self, metric_name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        self.gauges[metric_name] = value
        
    def record_value(self, metric_name: str, value: float) -> None:
        """Record a value in a histogram."""
        self.histograms[metric_name].append({
            "timestamp": time.time(),
            "value": value
        })
        
    def start_timer(self, metric_name: str) -> None:
        """Start a timer for measuring duration."""
        self._timer_starts[metric_name] = time.time()
        
    def stop_timer(self, metric_name: str) -> Optional[float]:
        """Stop a timer and record the duration."""
        if metric_name not in self._timer_starts:
            return None
            
        duration = time.time() - self._timer_starts[metric_name]
        del self._timer_starts[metric_name]
        
        # Record in histogram
        self.record_value(f"{metric_name}_duration_seconds", duration)
        return duration
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "uptime_seconds": time.time() - self.start_time,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {}
        }
        
        # Add histogram statistics
        for name, values in self.histograms.items():
            if values:
                vals = [v["value"] for v in values]
                summary["histograms"][name] = {
                    "count": len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "avg": sum(vals) / len(vals),
                    "latest": vals[-1] if vals else None
                }
                
        return summary
        
    def get_metric(self, metric_name: str) -> Optional[Any]:
        """Get a specific metric value."""
        if metric_name in self.counters:
            return self.counters[metric_name]
        elif metric_name in self.gauges:
            return self.gauges[metric_name]
        elif metric_name in self.histograms:
            return list(self.histograms[metric_name])
        return None
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self._timer_starts.clear()
        self.start_time = time.time()
        
    def log_summary(self, level: int = logging.INFO) -> None:
        """Log a summary of metrics."""
        summary = self.get_summary()
        self.logger.log(level, f"[Metrics] Summary: {summary}")


# Global metrics instance (singleton pattern)
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


# Convenience functions
def increment_counter(metric_name: str, value: int = 1) -> None:
    """Increment a counter metric."""
    get_metrics().increment(metric_name, value)


def set_gauge(metric_name: str, value: float) -> None:
    """Set a gauge metric."""
    get_metrics().set_gauge(metric_name, value)


def record_value(metric_name: str, value: float) -> None:
    """Record a histogram value."""
    get_metrics().record_value(metric_name, value)


def start_timer(metric_name: str) -> None:
    """Start a timer."""
    get_metrics().start_timer(metric_name)


def stop_timer(metric_name: str) -> Optional[float]:
    """Stop a timer and return duration."""
    return get_metrics().stop_timer(metric_name)


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        record_value(f"{self.metric_name}_duration_seconds", duration)
        return False


def timer(metric_name: str) -> TimerContext:
    """Create a timer context manager."""
    return TimerContext(metric_name)
