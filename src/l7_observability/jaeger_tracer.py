# -*- coding: utf-8 -*-
"""
Jaeger OpenTelemetry Tracer Configuration - Issue #19

Distributed tracing for APM instrumentation of the MetaController.
Provides span creation, context propagation, and integration with Grafana.

Features:
- Jaeger backend integration (UDP export, batching)
- Trace context propagation for async operations
- Structured span attributes for guard evaluation
- Integration with Prometheus metrics for trace correlation
- Automatic span lifecycle management

Configuration:
- Jaeger Agent: localhost:6831 (UDP, configurable)
- Service Name: octivault-trader
- Sampler: Const (all traces) or adaptive based on load
- Batch Processing: 512 spans per batch, 5-second flush
"""

import os
import logging
from typing import Optional, Dict, Any, TYPE_CHECKING
from contextlib import asynccontextmanager
import time

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span

try:
    # OpenTelemetry SDK & Jaeger exporter
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.context import Context
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("⚠️  OpenTelemetry not installed. APM tracing disabled.")
    print("   Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger")

logger = logging.getLogger(__name__)


class JaegerTracerConfig:
    """Configuration for Jaeger APM tracing."""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "octivault-trader")
        self.jaeger_host = os.getenv("JAEGER_HOST", "localhost")
        self.jaeger_port = int(os.getenv("JAEGER_PORT", "6831"))
        self.enabled = os.getenv("OTEL_ENABLED", "true").lower() == "true"
        self.sampler_type = os.getenv("OTEL_SAMPLER", "always_on")  # always_on, always_off, parentbased_always_on
        self.batch_size = int(os.getenv("OTEL_BATCH_SIZE", "512"))
        self.flush_interval = float(os.getenv("OTEL_FLUSH_INTERVAL", "5.0"))
        self.environment = os.getenv("OTEL_ENVIRONMENT", "production")
        self.version = os.getenv("OTEL_VERSION", "1.0.0")


class JaegerTracer:
    """Manages OpenTelemetry/Jaeger tracing for distributed tracing."""
    
    _instance: Optional['JaegerTracer'] = None
    _tracer_provider: Optional[Any] = None
    _tracer: Optional[Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._tracer is not None:
            return  # Already initialized
        
        self.config = JaegerTracerConfig()
        
        if not OTEL_AVAILABLE or not self.config.enabled:
            logger.info("APM tracing disabled or OpenTelemetry not available")
            self._tracer = None
            return
        
        try:
            self._initialize_tracer()
            logger.info(f"✅ Jaeger tracer initialized: {self.config.service_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Jaeger tracer: {e}")
            self._tracer = None
    
    def _initialize_tracer(self):
        """Initialize OpenTelemetry tracer with Jaeger exporter."""
        
        # Create resource with service metadata
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": self.config.version,
            "deployment.environment": self.config.environment,
            "host.name": os.getenv("HOSTNAME", "unknown"),
        })
        
        # Create Jaeger exporter (UDP to local agent)
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.config.jaeger_host,
            agent_port=self.config.jaeger_port,
        )
        
        # Create tracer provider with batch processing
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(
            BatchSpanProcessor(
                jaeger_exporter,
                max_queue_size=2048,
                max_export_batch_size=self.config.batch_size,
                schedule_delay_millis=int(self.config.flush_interval * 1000),
            )
        )
        
        # Set global tracer provider
        trace.set_tracer_provider(self._tracer_provider)
        
        # Get tracer instance
        self._tracer = trace.get_tracer(__name__)
        
        # Instrument standard libraries
        try:
            RequestsInstrumentor().instrument()
            SQLAlchemyInstrumentor().instrument()
        except Exception as e:
            logger.debug(f"Could not instrument standard libraries: {e}")
    
    def get_tracer(self):
        """Get the global tracer instance."""
        return self._tracer
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a new span.
        
        Args:
            name: Span name (e.g., "guard_evaluation", "trade_execution")
            attributes: Optional span attributes (tags)
        
        Returns:
            Span object or None if tracing disabled
        """
        if self._tracer is None:
            return None
        
        span = self._tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception as e:
                    logger.debug(f"Could not set span attribute {key}: {e}")
        
        return span
    
    async def trace_async(self, name: str, coro, attributes: Optional[Dict[str, Any]] = None):
        """
        Trace an async coroutine.
        
        Args:
            name: Span name
            coro: Coroutine to trace
            attributes: Optional span attributes
        
        Returns:
            Coroutine result
        """
        if self._tracer is None:
            return await coro
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass
            return await coro
    
    def trace_sync(self, name: str, func, *args, attributes: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Trace a synchronous function.
        
        Args:
            name: Span name
            func: Function to call
            *args: Function args
            attributes: Optional span attributes
            **kwargs: Function kwargs
        
        Returns:
            Function result
        """
        if self._tracer is None:
            return func(*args, **kwargs)
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass
            return func(*args, **kwargs)
    
    @asynccontextmanager
    async def span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Async context manager for creating spans.
        
        Usage:
            async with tracer.span("operation", {"symbol": "BTC"}) as span:
                # Do work
                span.set_attribute("result", "success")
        """
        if self._tracer is None:
            yield None
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass
            yield span
    
    def add_event(self, span, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span."""
        if span is None:
            return
        
        try:
            span.add_event(event_name, attributes or {})
        except Exception as e:
            logger.debug(f"Could not add event to span: {e}")
    
    def set_span_status_success(self, span):
        """Mark span as successful."""
        if span is None:
            return
        
        try:
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.debug(f"Could not set span status: {e}")
    
    def set_span_status_error(self, span, description: str = "Error"):
        """Mark span as error."""
        if span is None:
            return
        
        try:
            span.set_status(Status(StatusCode.ERROR, description))
        except Exception as e:
            logger.debug(f"Could not set span error status: {e}")
    
    def shutdown(self):
        """Shutdown tracer and flush pending spans."""
        if self._tracer_provider is not None:
            try:
                self._tracer_provider.force_flush(timeout_millis=5000)
                self._tracer_provider.shutdown()
                logger.info("✅ Jaeger tracer shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down tracer: {e}")


# Global tracer instance
_global_tracer = None


def get_tracer() -> JaegerTracer:
    """Get or create the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = JaegerTracer()
    return _global_tracer


def init_tracing():
    """Initialize global tracing."""
    return get_tracer()


def shutdown_tracing():
    """Shutdown global tracing."""
    tracer = get_tracer()
    tracer.shutdown()
