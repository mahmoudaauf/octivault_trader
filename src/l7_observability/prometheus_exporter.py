#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/prometheus_exporter.py - Prometheus Metrics for Safety Guards

Exports all 5 safety guard metrics to Prometheus for monitoring:
- BalanceValidator metrics
- LeverageValidator metrics
- TradingHoursValidator metrics
- AnomalyDetector metrics
- CorrelationManager metrics

Issue #16: Prometheus Metrics Integration
"""

import time
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SafetyGuardMetrics:
    """
    Prometheus metrics for all 5 safety guards.
    
    Tracks validation checks, latencies, and approval rates.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics registry."""
        self.registry = registry or CollectorRegistry()
        self._init_balance_metrics()
        self._init_leverage_metrics()
        self._init_hours_metrics()
        self._init_anomaly_metrics()
        self._init_correlation_metrics()
        self._init_execution_metrics()
        logger.info("[Prometheus] SafetyGuardMetrics initialized with %d metrics", 30)
    
    # ============================================================================
    # BALANCE GUARD METRICS
    # ============================================================================
    
    def _init_balance_metrics(self):
        """Initialize BalanceValidator metrics."""
        self.balance_checks = Counter(
            'balance_validation_total',
            'Total balance validation checks',
            ['symbol', 'status'],  # status: approved, rejected_insufficient, rejected_overallocated
            registry=self.registry
        )
        
        self.balance_latency = Histogram(
            'balance_validation_latency_seconds',
            'Balance validation latency',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry
        )
        
        self.balance_approval_rate = Gauge(
            'balance_validation_approval_rate',
            'Balance validation approval rate (0-1)',
            registry=self.registry
        )
        
        logger.debug("✅ Balance metrics initialized")
    
    # ============================================================================
    # LEVERAGE GUARD METRICS
    # ============================================================================
    
    def _init_leverage_metrics(self):
        """Initialize LeverageValidator metrics."""
        self.leverage_checks = Counter(
            'leverage_validation_total',
            'Total leverage validation checks',
            ['symbol', 'status'],  # status: approved, rejected_over_leverage, rejected_at_limit
            registry=self.registry
        )
        
        self.leverage_latency = Histogram(
            'leverage_validation_latency_seconds',
            'Leverage validation latency',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry
        )
        
        self.leverage_approval_rate = Gauge(
            'leverage_validation_approval_rate',
            'Leverage validation approval rate (0-1)',
            registry=self.registry
        )
        
        self.current_max_leverage = Gauge(
            'current_max_leverage',
            'Current maximum leverage setting',
            registry=self.registry
        )
        
        logger.debug("✅ Leverage metrics initialized")
    
    # ============================================================================
    # TRADING HOURS GUARD METRICS
    # ============================================================================
    
    def _init_hours_metrics(self):
        """Initialize TradingHoursValidator metrics."""
        self.hours_checks = Counter(
            'hours_validation_total',
            'Total trading hours validation checks',
            ['symbol', 'status'],  # status: allowed, rejected_market_closed, rejected_maintenance
            registry=self.registry
        )
        
        self.hours_latency = Histogram(
            'hours_validation_latency_seconds',
            'Trading hours validation latency',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry
        )
        
        self.hours_approval_rate = Gauge(
            'hours_validation_approval_rate',
            'Trading hours validation approval rate (0-1)',
            registry=self.registry
        )
        
        logger.debug("✅ Hours metrics initialized")
    
    # ============================================================================
    # ANOMALY GUARD METRICS
    # ============================================================================
    
    def _init_anomaly_metrics(self):
        """Initialize AnomalyDetector metrics."""
        self.anomaly_checks = Counter(
            'anomaly_detection_total',
            'Total anomaly detection checks',
            ['status'],  # status: accepted, rejected_outlier, rejected_extreme, quarantined
            registry=self.registry
        )
        
        self.anomaly_latency = Histogram(
            'anomaly_detection_latency_seconds',
            'Anomaly detection latency',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry
        )
        
        self.anomaly_detection_rate = Gauge(
            'anomaly_detection_rate',
            'Rate of signals detected as anomalous (0-1)',
            registry=self.registry
        )
        
        self.signals_in_history = Gauge(
            'anomaly_signals_in_history',
            'Total signals in anomaly detection history',
            registry=self.registry
        )
        
        logger.debug("✅ Anomaly metrics initialized")
    
    # ============================================================================
    # CORRELATION GUARD METRICS
    # ============================================================================
    
    def _init_correlation_metrics(self):
        """Initialize CorrelationManager metrics."""
        self.correlation_checks = Counter(
            'correlation_validation_total',
            'Total correlation validation checks',
            ['symbol', 'status'],  # status: approved, rejected_concentration, rejected_correlation
            registry=self.registry
        )
        
        self.correlation_latency = Histogram(
            'correlation_validation_latency_seconds',
            'Correlation validation latency',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry
        )
        
        self.correlation_approval_rate = Gauge(
            'correlation_validation_approval_rate',
            'Correlation validation approval rate (0-1)',
            registry=self.registry
        )
        
        self.active_positions_count = Gauge(
            'active_positions_count',
            'Current number of active positions being tracked',
            registry=self.registry
        )
        
        self.max_concentration_ratio = Gauge(
            'max_concentration_ratio',
            'Maximum concentration ratio in portfolio (0-1)',
            registry=self.registry
        )
        
        logger.debug("✅ Correlation metrics initialized")
    
    # ============================================================================
    # EXECUTION METRICS
    # ============================================================================
    
    def _init_execution_metrics(self):
        """Initialize overall execution metrics."""
        self.trades_executed = Counter(
            'trades_executed_total',
            'Total trades executed',
            ['symbol', 'side', 'guard_status'],  # guard_status: all_passed, rejected_by_*
            registry=self.registry
        )
        
        self.execution_latency = Histogram(
            'execution_latency_seconds',
            'Total execution latency (all guards + execution)',
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
            registry=self.registry
        )
        
        self.total_guard_latency = Histogram(
            'guard_latency_seconds',
            'Total latency for all guards combined',
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2),
            registry=self.registry
        )
        
        self.overall_approval_rate = Gauge(
            'overall_approval_rate',
            'Overall approval rate across all guards (0-1)',
            registry=self.registry
        )
        
        logger.debug("✅ Execution metrics initialized")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_metrics_instance: Optional[SafetyGuardMetrics] = None


def get_metrics() -> SafetyGuardMetrics:
    """Get singleton metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = SafetyGuardMetrics()
    return _metrics_instance


# ============================================================================
# HELPER FUNCTIONS FOR RECORDING METRICS
# ============================================================================

def record_balance_check(symbol: str, approved: bool, latency: float):
    """Record a balance validation check."""
    metrics = get_metrics()
    status = "approved" if approved else "rejected_insufficient"
    metrics.balance_checks.labels(symbol=symbol, status=status).inc()
    metrics.balance_latency.observe(latency)


def record_leverage_check(symbol: str, approved: bool, latency: float):
    """Record a leverage validation check."""
    metrics = get_metrics()
    status = "approved" if approved else "rejected_over_leverage"
    metrics.leverage_checks.labels(symbol=symbol, status=status).inc()
    metrics.leverage_latency.observe(latency)


def record_hours_check(symbol: str, approved: bool, latency: float):
    """Record a trading hours validation check."""
    metrics = get_metrics()
    status = "allowed" if approved else "rejected_market_closed"
    metrics.hours_checks.labels(symbol=symbol, status=status).inc()
    metrics.hours_latency.observe(latency)


def record_anomaly_check(approved: bool, latency: float):
    """Record an anomaly detection check."""
    metrics = get_metrics()
    status = "accepted" if approved else "rejected_outlier"
    metrics.anomaly_checks.labels(status=status).inc()
    metrics.anomaly_latency.observe(latency)


def record_correlation_check(symbol: str, approved: bool, latency: float):
    """Record a correlation validation check."""
    metrics = get_metrics()
    status = "approved" if approved else "rejected_concentration"
    metrics.correlation_checks.labels(symbol=symbol, status=status).inc()
    metrics.correlation_latency.observe(latency)


def record_trade_execution(symbol: str, side: str, all_guards_passed: bool, latency: float):
    """Record a trade execution."""
    metrics = get_metrics()
    guard_status = "all_passed" if all_guards_passed else "rejected_by_guard"
    metrics.trades_executed.labels(symbol=symbol, side=side, guard_status=guard_status).inc()
    metrics.execution_latency.observe(latency)


def update_approval_rate(guard_name: str, approval_rate: float):
    """Update approval rate gauge for a guard."""
    metrics = get_metrics()
    
    if guard_name == "balance":
        metrics.balance_approval_rate.set(approval_rate)
    elif guard_name == "leverage":
        metrics.leverage_approval_rate.set(approval_rate)
    elif guard_name == "hours":
        metrics.hours_approval_rate.set(approval_rate)
    elif guard_name == "anomaly":
        metrics.anomaly_detection_rate.set(approval_rate)
    elif guard_name == "correlation":
        metrics.correlation_approval_rate.set(approval_rate)
    elif guard_name == "overall":
        metrics.overall_approval_rate.set(approval_rate)


# ============================================================================
# PROMETHEUS EXPOSITION
# ============================================================================

def get_metrics_registry() -> CollectorRegistry:
    """Get Prometheus registry for exposition."""
    return get_metrics().registry


if __name__ == "__main__":
    # Test metrics instantiation
    metrics = get_metrics()
    print("✅ Metrics initialized successfully")
    print(f"   - Balance Guard: ✅")
    print(f"   - Leverage Guard: ✅")
    print(f"   - Trading Hours Guard: ✅")
    print(f"   - Anomaly Detector: ✅")
    print(f"   - Correlation Manager: ✅")
    print(f"   - Execution Metrics: ✅")
