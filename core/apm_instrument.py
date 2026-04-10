# -*- coding: utf-8 -*-
"""
APM Instrumentation - Issue #19

Distributed tracing instrumentation for MetaController and safety guards.
Integrates with Jaeger backend for visualization and analysis.

Key Traces:
1. Guard Evaluation Traces
   - balance_guard_evaluation: Balance checking and threshold validation
   - leverage_guard_evaluation: Leverage checking and calculations
   - hours_guard_evaluation: Trading hours validation
   - anomaly_guard_evaluation: Anomaly detection signal processing
   - correlation_guard_evaluation: Position correlation checking
   - execution_guard_evaluation: Trade execution monitoring

2. Trade Execution Traces
   - trade_decision: Signal arrival through decision making
   - trade_execution: Order submission and confirmation
   - position_management: Position tracking and updates

3. System Traces
   - evaluate_and_act: Main controller loop iteration
   - mode_evaluation: Trading mode decision making
   - signal_processing: Incoming signal processing

Span Attributes (OpenTelemetry Standard Tags):
   - service.name: octivault-trader
   - span.kind: INTERNAL, SERVER, CLIENT
   - Guard-specific tags: guard_name, symbol, rejection_reason, approval_rate
"""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

try:
    from core.jaeger_tracer import get_tracer, JaegerTracer
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False


logger = logging.getLogger(__name__)


class APMInstrument:
    """Decorator and context manager for APM tracing."""
    
    def __init__(self, tracer: Optional['JaegerTracer'] = None):
        self.tracer = tracer or (get_tracer() if TRACING_AVAILABLE else None)
        self.enabled = TRACING_AVAILABLE and self.tracer is not None
    
    async def trace_guard_evaluation(
        self,
        guard_name: str,
        symbol: str,
        coro,
        additional_attrs: Optional[Dict[str, Any]] = None
    ):
        """
        Trace a guard evaluation.
        
        Args:
            guard_name: Name of guard (balance, leverage, hours, etc)
            symbol: Trading symbol being evaluated
            coro: Coroutine to trace
            additional_attrs: Additional span attributes
        
        Returns:
            Guard decision result
        """
        if not self.enabled:
            return await coro
        
        span_name = f"{guard_name}_evaluation"
        attributes = {
            "guard.name": guard_name,
            "symbol": symbol,
            "span.kind": "INTERNAL",
            "component": "guard_system",
        }
        
        if additional_attrs:
            attributes.update(additional_attrs)
        
        try:
            result = await self.tracer.trace_async(span_name, coro, attributes)
            return result
        except Exception as e:
            logger.error(f"Guard evaluation trace failed for {guard_name}: {e}")
            return await coro
    
    async def trace_trade_decision(
        self,
        symbol: str,
        agent_name: str,
        signal_type: str,
        coro,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trace trade decision making.
        
        Args:
            symbol: Trading symbol
            agent_name: Source agent name
            signal_type: Type of signal (buy, sell, hold)
            coro: Coroutine for decision logic
            metadata: Optional signal metadata
        
        Returns:
            Decision result
        """
        if not self.enabled:
            return await coro
        
        span_name = f"trade_decision_{signal_type}"
        attributes = {
            "symbol": symbol,
            "agent.name": agent_name,
            "signal.type": signal_type,
            "span.kind": "INTERNAL",
            "component": "arbitrator",
        }
        
        if metadata:
            attributes.update({f"signal.{k}": v for k, v in metadata.items()})
        
        try:
            result = await self.tracer.trace_async(span_name, coro, attributes)
            return result
        except Exception as e:
            logger.error(f"Trade decision trace failed for {symbol}: {e}")
            return await coro
    
    async def trace_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        coro,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trace trade execution.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY, SELL)
            quantity: Order quantity
            coro: Coroutine for execution logic
            metadata: Optional execution metadata
        
        Returns:
            Execution result
        """
        if not self.enabled:
            return await coro
        
        span_name = "trade_execution"
        attributes = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "span.kind": "CLIENT",
            "component": "execution",
        }
        
        if metadata:
            attributes.update({f"exec.{k}": v for k, v in metadata.items()})
        
        try:
            result = await self.tracer.trace_async(span_name, coro, attributes)
            return result
        except Exception as e:
            logger.error(f"Execution trace failed for {symbol}: {e}")
            return await coro
    
    async def trace_loop_iteration(self, coro, cycle_number: int = 0):
        """
        Trace main controller loop iteration.
        
        Args:
            coro: Main loop coroutine
            cycle_number: Optional iteration counter
        
        Returns:
            Loop iteration result
        """
        if not self.enabled:
            return await coro
        
        span_name = "evaluate_and_act_iteration"
        attributes = {
            "cycle.number": cycle_number,
            "timestamp": datetime.utcnow().isoformat(),
            "span.kind": "INTERNAL",
        }
        
        try:
            result = await self.tracer.trace_async(span_name, coro, attributes)
            return result
        except Exception as e:
            logger.error(f"Loop iteration trace failed: {e}")
            return await coro


class TraceContext:
    """Context manager for manual span creation."""
    
    def __init__(self, tracer: Optional['JaegerTracer'] = None):
        self.tracer = tracer or (get_tracer() if TRACING_AVAILABLE else None)
        self.enabled = TRACING_AVAILABLE and self.tracer is not None
        self.span = None
    
    async def __aenter__(self):
        """Enter async context."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self.span and self.tracer:
            if exc_type:
                self.tracer.set_span_status_error(self.span, str(exc_type))
            else:
                self.tracer.set_span_status_success(self.span)
    
    async def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a new span."""
        if not self.enabled:
            return None
        
        self.span = self.tracer.create_span(name, attributes)
        return self.span


# Instrumentation statistics (for diagnostics)
class InstrumentationMetrics:
    """Track instrumentation metrics."""
    
    def __init__(self):
        self.guards_traced = {}
        self.trades_traced = 0
        self.trace_errors = 0
        self.total_traces = 0
        self.start_time = time.time()
    
    def record_guard_trace(self, guard_name: str):
        """Record guard trace."""
        self.total_traces += 1
        if guard_name not in self.guards_traced:
            self.guards_traced[guard_name] = 0
        self.guards_traced[guard_name] += 1
    
    def record_trade_trace(self):
        """Record trade execution trace."""
        self.total_traces += 1
        self.trades_traced += 1
    
    def record_error(self):
        """Record trace error."""
        self.trace_errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        uptime = time.time() - self.start_time
        return {
            "total_traces": self.total_traces,
            "guards_traced": self.guards_traced,
            "trades_traced": self.trades_traced,
            "trace_errors": self.trace_errors,
            "uptime_seconds": uptime,
            "avg_traces_per_minute": (self.total_traces / uptime * 60) if uptime > 0 else 0,
        }


# Global instrumentation instance
_apm_instrument = None


def get_apm_instrument() -> APMInstrument:
    """Get or create the global APM instrument."""
    global _apm_instrument
    if _apm_instrument is None:
        _apm_instrument = APMInstrument()
    return _apm_instrument


# Global metrics
_instrumentation_metrics = None


def get_instrumentation_metrics() -> InstrumentationMetrics:
    """Get or create instrumentation metrics."""
    global _instrumentation_metrics
    if _instrumentation_metrics is None:
        _instrumentation_metrics = InstrumentationMetrics()
    return _instrumentation_metrics
