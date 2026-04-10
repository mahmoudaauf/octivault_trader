"""
Anomaly Detection Module for Trading Signals

This module provides statistical anomaly detection for trading signals to prevent
garbage or corrupted data from entering the trading pipeline. Uses multi-window
baselines and 3-sigma rule detection.

Issue #9: Anomaly Detection System
- Detect statistical outliers in signal data
- Quarantine anomalous signals before execution
- Track anomaly history and patterns
- Alert operators to signal quality issues
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import statistics
import threading

logger = logging.getLogger(__name__)


class AnomalyStatus(Enum):
    """Signal anomaly status indicators."""
    ACCEPTED = "ACCEPTED"
    REJECTED_OUTLIER = "REJECTED_OUTLIER"
    REJECTED_EXTREME_OUTLIER = "REJECTED_EXTREME_OUTLIER"
    REJECTED_INVALID_RANGE = "REJECTED_INVALID_RANGE"
    QUARANTINED_REVIEW = "QUARANTINED_REVIEW"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


@dataclass
class SignalBaseline:
    """Statistical baseline for a signal."""
    signal_id: str
    mean: float
    std_dev: float
    min_val: float
    max_val: float
    sample_count: int
    window_seconds: int
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_outlier(self, value: float, sigma: float = 3.0) -> Tuple[bool, float]:
        """
        Check if value is statistical outlier.
        
        Args:
            value: Signal value to check
            sigma: Standard deviation threshold (default 3.0 = 99.7% confidence)
            
        Returns:
            (is_outlier, z_score)
        """
        if self.std_dev == 0:
            # Constant signal - any change is anomaly
            return value != self.mean, float('inf')
        
        z_score = abs((value - self.mean) / self.std_dev)
        return z_score > sigma, z_score
    
    def is_in_valid_range(self, value: float) -> bool:
        """Check if value is within historically observed range."""
        return self.min_val <= value <= self.max_val


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection check."""
    status: AnomalyStatus
    signal_id: str
    value: float
    timestamp: datetime
    z_score: Optional[float] = None
    reason: str = ""
    baseline: Optional[SignalBaseline] = None
    confidence: float = 0.95  # 95% confidence default


class SignalHistoryBuffer:
    """Rolling window buffer for signal history."""
    
    def __init__(self, signal_id: str, window_seconds: int = 3600):
        self.signal_id = signal_id
        self.window_seconds = window_seconds
        self.history: List[Tuple[datetime, float]] = []
        self.lock = threading.Lock()
    
    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add value to history, pruning old entries."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self.lock:
            self.history.append((timestamp, value))
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_seconds)
            self.history = [(ts, v) for ts, v in self.history if ts > cutoff_time]
    
    def get_values(self) -> List[float]:
        """Get all values in current window."""
        with self.lock:
            return [v for _, v in self.history]
    
    def calculate_baseline(self) -> Optional[SignalBaseline]:
        """Calculate statistical baseline from history."""
        values = self.get_values()
        
        if len(values) < 3:
            return None  # Insufficient data
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        
        return SignalBaseline(
            signal_id=self.signal_id,
            mean=mean,
            std_dev=std_dev,
            min_val=min_val,
            max_val=max_val,
            sample_count=len(values),
            window_seconds=self.window_seconds,
            updated_at=datetime.utcnow()
        )


class AnomalyDetector:
    """
    Statistical anomaly detector for trading signals.
    
    Detects:
    - Extreme outliers (>3 sigma)
    - Moderate outliers (2-3 sigma)
    - Range violations (outside observed bounds)
    - Constant signals (zero variance)
    - Suspicious patterns
    """
    
    def __init__(self, min_history_samples: int = 10, review_threshold_sigma: float = 2.5):
        self.min_history_samples = min_history_samples
        self.review_threshold_sigma = review_threshold_sigma  # Review at 2.5 sigma
        self.signal_buffers: Dict[str, SignalHistoryBuffer] = {}
        self.signal_baselines: Dict[str, SignalBaseline] = {}
        self.rejection_log: List[AnomalyDetectionResult] = []
        self.lock = threading.Lock()
        logger.info(f"AnomalyDetector initialized (min_samples={min_history_samples}, review_sigma={review_threshold_sigma})")
    
    def register_signal(self, signal_id: str, window_seconds: int = 3600) -> None:
        """Register new signal for anomaly tracking."""
        with self.lock:
            if signal_id not in self.signal_buffers:
                self.signal_buffers[signal_id] = SignalHistoryBuffer(signal_id, window_seconds)
                logger.debug(f"Registered signal {signal_id} for anomaly detection")
    
    def update_baseline(self, signal_id: str) -> bool:
        """
        Update baseline for signal.
        
        Returns:
            True if baseline updated, False if insufficient history
        """
        if signal_id not in self.signal_buffers:
            self.register_signal(signal_id)
        
        buffer = self.signal_buffers[signal_id]
        baseline = buffer.calculate_baseline()
        
        if baseline is None:
            return False
        
        with self.lock:
            self.signal_baselines[signal_id] = baseline
        
        logger.debug(f"Updated baseline for {signal_id}: mean={baseline.mean:.4f}, std_dev={baseline.std_dev:.4f}")
        return True
    
    def check_signal(self, signal_id: str, value: float) -> AnomalyDetectionResult:
        """
        Check if signal value is anomalous.
        
        Args:
            signal_id: Unique signal identifier
            value: Signal value to check
            
        Returns:
            AnomalyDetectionResult with status and details
        """
        # Register signal if not seen before
        if signal_id not in self.signal_buffers:
            self.register_signal(signal_id)
        
        # Add to history
        buffer = self.signal_buffers[signal_id]
        buffer.add(value)
        
        # Get current baseline
        with self.lock:
            baseline = self.signal_baselines.get(signal_id)
        
        # Insufficient history
        if baseline is None or baseline.sample_count < self.min_history_samples:
            result = AnomalyDetectionResult(
                status=AnomalyStatus.INSUFFICIENT_HISTORY,
                signal_id=signal_id,
                value=value,
                timestamp=datetime.utcnow(),
                reason=f"Insufficient history (n={baseline.sample_count if baseline else 0}, min={self.min_history_samples})"
            )
            self._log_result(result)
            return result
        
        # Check for extreme outlier (>3 sigma = 99.7% confidence)
        is_outlier, z_score = baseline.is_outlier(value, sigma=3.0)
        if is_outlier:
            result = AnomalyDetectionResult(
                status=AnomalyStatus.REJECTED_EXTREME_OUTLIER,
                signal_id=signal_id,
                value=value,
                z_score=z_score,
                timestamp=datetime.utcnow(),
                baseline=baseline,
                reason=f"Extreme outlier detected (z-score={z_score:.2f}, threshold=3.0)",
                confidence=0.997
            )
            self._log_result(result)
            logger.warning(f"ANOMALY EXTREME: {signal_id} value={value:.4f} z_score={z_score:.2f}")
            return result
        
        # Check for moderate outlier (2-3 sigma = recommend review)
        is_moderate, z_score = baseline.is_outlier(value, sigma=self.review_threshold_sigma)
        if is_moderate:
            result = AnomalyDetectionResult(
                status=AnomalyStatus.QUARANTINED_REVIEW,
                signal_id=signal_id,
                value=value,
                z_score=z_score,
                timestamp=datetime.utcnow(),
                baseline=baseline,
                reason=f"Moderate outlier - recommend review (z-score={z_score:.2f}, threshold={self.review_threshold_sigma})",
                confidence=0.95
            )
            self._log_result(result)
            logger.info(f"ANOMALY MODERATE: {signal_id} value={value:.4f} z_score={z_score:.2f} - recommend review")
            return result
        
        # Check if in valid range
        if not baseline.is_in_valid_range(value):
            result = AnomalyDetectionResult(
                status=AnomalyStatus.REJECTED_INVALID_RANGE,
                signal_id=signal_id,
                value=value,
                timestamp=datetime.utcnow(),
                baseline=baseline,
                reason=f"Outside observed range (min={baseline.min_val:.4f}, max={baseline.max_val:.4f})"
            )
            self._log_result(result)
            logger.warning(f"ANOMALY RANGE: {signal_id} value={value:.4f} outside range [{baseline.min_val:.4f}, {baseline.max_val:.4f}]")
            return result
        
        # Normal signal - accepted
        result = AnomalyDetectionResult(
            status=AnomalyStatus.ACCEPTED,
            signal_id=signal_id,
            value=value,
            z_score=z_score,
            timestamp=datetime.utcnow(),
            baseline=baseline,
            reason="Signal within acceptable parameters"
        )
        return result
    
    def _log_result(self, result: AnomalyDetectionResult) -> None:
        """Log anomaly detection result."""
        with self.lock:
            self.rejection_log.append(result)
            # Keep only last 1000 entries
            if len(self.rejection_log) > 1000:
                self.rejection_log = self.rejection_log[-1000:]
    
    def get_signal_stats(self, signal_id: str) -> Dict:
        """Get statistics for signal."""
        with self.lock:
            baseline = self.signal_baselines.get(signal_id)
        
        if baseline is None:
            return {"signal_id": signal_id, "status": "no_baseline"}
        
        buffer = self.signal_buffers[signal_id]
        values = buffer.get_values()
        
        return {
            "signal_id": signal_id,
            "mean": baseline.mean,
            "std_dev": baseline.std_dev,
            "min": baseline.min_val,
            "max": baseline.max_val,
            "sample_count": baseline.sample_count,
            "window_seconds": baseline.window_seconds,
            "updated_at": baseline.updated_at.isoformat(),
            "current_samples": len(values)
        }
    
    def get_anomaly_log(self, signal_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent anomalies."""
        with self.lock:
            logs = self.rejection_log[-limit:]
        
        results = []
        for log in logs:
            if signal_id and log.signal_id != signal_id:
                continue
            
            results.append({
                "signal_id": log.signal_id,
                "status": log.status.value,
                "value": log.value,
                "z_score": log.z_score,
                "timestamp": log.timestamp.isoformat(),
                "reason": log.reason,
                "confidence": log.confidence
            })
        
        return results
    
    def reset_baseline(self, signal_id: str) -> None:
        """Reset baseline for signal (useful after known issues)."""
        with self.lock:
            if signal_id in self.signal_baselines:
                del self.signal_baselines[signal_id]
            if signal_id in self.signal_buffers:
                self.signal_buffers[signal_id].history.clear()
        logger.info(f"Reset baseline for {signal_id}")


# Global instance
_anomaly_detector: Optional[AnomalyDetector] = None
_detector_lock = threading.Lock()


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        with _detector_lock:
            if _anomaly_detector is None:
                _anomaly_detector = AnomalyDetector(
                    min_history_samples=10,
                    review_threshold_sigma=2.5
                )
    return _anomaly_detector


def initialize_anomaly_detection() -> AnomalyDetector:
    """Initialize anomaly detection system at startup."""
    detector = get_anomaly_detector()
    logger.info("Anomaly detection system initialized")
    return detector


if __name__ == "__main__":
    # Example usage
    detector = get_anomaly_detector()
    
    # Register signals
    detector.register_signal("RSI_BTC")
    detector.register_signal("MOMENTUM_ETH")
    
    # Add some history (simulating normal operation)
    import random
    random.seed(42)
    
    for i in range(20):
        value = 50 + random.gauss(0, 3)  # Normal distribution, mean=50, std=3
        detector.check_signal("RSI_BTC", value)
    
    # Update baseline
    detector.update_baseline("RSI_BTC")
    
    # Check normal signal (should be accepted)
    result = detector.check_signal("RSI_BTC", 51.0)
    print(f"Normal signal: {result.status.value}")
    
    # Check anomaly (should be extreme outlier)
    result = detector.check_signal("RSI_BTC", 90.0)
    print(f"Anomaly signal: {result.status.value} - z_score={result.z_score:.2f}")
    
    # Print stats
    stats = detector.get_signal_stats("RSI_BTC")
    print(f"Signal stats: {stats}")
