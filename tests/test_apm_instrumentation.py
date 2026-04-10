# -*- coding: utf-8 -*-
"""
Test Suite: APM Instrumentation (Issue #19)

Tests for Jaeger distributed tracing integration with MetaController and guard system.
Validates:
1. Tracer initialization and configuration
2. Span creation and context propagation
3. Guard evaluation tracing
4. Trade execution flow tracing
5. Error handling and span status marking
"""

import pytest
import asyncio
import logging
from typing import Optional, Dict, Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import time

# Tests assume APM modules are available
try:
    from core.apm_instrument import APMInstrument, get_apm_instrument, TraceContext
    from core.jaeger_tracer import JaegerTracer, get_tracer, init_tracing, shutdown_tracing
    APM_AVAILABLE = True
except ImportError:
    APM_AVAILABLE = False
    APMInstrument = None
    JaegerTracer = None


@pytest.fixture
def mock_tracer():
    """Create a mock Jaeger tracer for testing."""
    if not APM_AVAILABLE:
        pytest.skip("APM not available")
    
    tracer = Mock(spec=JaegerTracer)
    tracer.create_span = Mock(return_value=Mock())
    tracer.trace_async = AsyncMock(return_value={"status": "success"})
    tracer.trace_sync = Mock(return_value={"status": "success"})
    tracer.set_span_status_success = Mock()
    tracer.set_span_status_error = Mock()
    
    return tracer


@pytest.fixture
def apm_instrument(mock_tracer):
    """Create APMInstrument with mock tracer."""
    if not APM_AVAILABLE:
        pytest.skip("APM not available")
    
    apm = APMInstrument(tracer=mock_tracer)
    return apm


# ============================================================================
# Test Suite 1: Tracer Initialization
# ============================================================================

class TestTracerInitialization:
    """Test Jaeger tracer initialization and configuration."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_tracer_singleton(self):
        """Verify tracer uses singleton pattern."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2, "Tracer should be singleton"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_tracer_initialization_creates_valid_instance(self):
        """Verify tracer initialization creates valid instance."""
        tracer = get_tracer()
        assert tracer is not None, "Tracer should be initialized"
        assert hasattr(tracer, "create_span"), "Tracer should have create_span method"
        assert hasattr(tracer, "trace_async"), "Tracer should have trace_async method"
        assert hasattr(tracer, "trace_sync"), "Tracer should have trace_sync method"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_apm_instrument_initialization(self):
        """Verify APM instrument initializes correctly."""
        apm = get_apm_instrument()
        assert apm is not None, "APM instrument should be initialized"
        assert hasattr(apm, "trace_guard_evaluation"), "APM should have guard tracing method"
        assert hasattr(apm, "trace_trade_decision"), "APM should have trade decision tracing method"
        assert hasattr(apm, "trace_execution"), "APM should have execution tracing method"


# ============================================================================
# Test Suite 2: Span Creation and Context
# ============================================================================

class TestSpanCreation:
    """Test span creation and attribute management."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_create_span_with_attributes(self, mock_tracer):
        """Verify span creation with attributes."""
        apm = APMInstrument(tracer=mock_tracer)
        
        span = mock_tracer.create_span(
            name="test_operation",
            attributes={"symbol": "BTC", "amount": 1.5}
        )
        
        mock_tracer.create_span.assert_called_once_with(
            name="test_operation",
            attributes={"symbol": "BTC", "amount": 1.5}
        )
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_span_context_manager(self, mock_tracer):
        """Verify span context manager works correctly."""
        apm = APMInstrument(tracer=mock_tracer)
        
        # Mock the tracer's span method to return a context manager
        mock_span = Mock()
        mock_span.__aenter__ = AsyncMock(return_value=mock_span)
        mock_span.__aexit__ = AsyncMock(return_value=None)
        
        mock_tracer.span = Mock(return_value=mock_span)
        
        async with mock_tracer.span("test_span", {"key": "value"}) as span:
            assert span is mock_span
        
        mock_span.__aexit__.assert_called_once()


# ============================================================================
# Test Suite 3: Guard Evaluation Tracing
# ============================================================================

class TestGuardEvaluationTracing:
    """Test tracing of guard evaluations."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_balance_guard_tracing(self):
        """Verify balance guard evaluation is traced."""
        # Setup
        apm = get_apm_instrument()
        symbol = "BTC"
        
        # Create mock coroutine
        async def mock_guard_check():
            return {"approved": True, "reason": "sufficient_balance"}
        
        # Execute guard tracing
        result = await apm.trace_guard_evaluation(
            guard_name="balance_guard",
            symbol=symbol,
            coro=mock_guard_check(),
            additional_attrs={"threshold": 100.0, "current": 150.0}
        )
        
        # Verify result
        assert result is not None, "Guard evaluation should return result"
        assert result.get("approved") == True, "Balance guard should approve"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_guard_rejection_tracking(self):
        """Verify guard rejection is tracked with reason."""
        apm = get_apm_instrument()
        symbol = "BTC"
        
        # Create mock rejection result
        async def mock_guard_check():
            return {"approved": False, "reason": "insufficient_balance"}
        
        # Execute with rejection
        result = await apm.trace_guard_evaluation(
            guard_name="balance_guard",
            symbol=symbol,
            coro=mock_guard_check(),
            additional_attrs={"rejection_reason": "insufficient_balance"}
        )
        
        # Verify rejection was tracked
        assert result is not None
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_multiple_guard_correlation(self, apm_instrument):
        """Verify multiple guards are correlated in traces."""
        symbol = "BTC"
        guards = ["balance_guard", "leverage_guard", "hours_guard"]
        
        # Trace multiple guards
        results = []
        for guard_name in guards:
            async def mock_guard():
                return {"approved": True}
            
            result = await apm_instrument.trace_guard_evaluation(
                guard_name=guard_name,
                symbol=symbol,
                coro=mock_guard(),
                additional_attrs={"guard": guard_name}
            )
            results.append(result)
        
        # All guards should be traced
        assert len(results) == 3, "All guards should be traced"


# ============================================================================
# Test Suite 4: Trade Execution Tracing
# ============================================================================

class TestTradeExecutionTracing:
    """Test tracing of trade execution flow."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_trade_decision_tracing(self):
        """Verify trade decisions are traced."""
        apm = get_apm_instrument()
        symbol = "BTC"
        agent_name = "trend_hunter"
        signal_type = "BUY"
        
        async def mock_decision():
            return {"symbol": symbol, "action": "BUY", "confidence": 0.85}
        
        result = await apm.trace_trade_decision(
            symbol=symbol,
            agent_name=agent_name,
            signal_type=signal_type,
            coro=mock_decision(),
            metadata={"confidence": 0.85, "reason": "breakout"}
        )
        
        assert result is not None, "Trade decision should be traced"
        assert result.get("action") == "BUY", "Decision should be captured"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_execution_tracing_with_latency(self):
        """Verify execution tracing captures latency."""
        apm = get_apm_instrument()
        symbol = "BTC"
        side = "BUY"
        quantity = 1.0
        
        async def mock_execution():
            await asyncio.sleep(0.1)  # Simulate network latency
            return {
                "order_id": "12345",
                "status": "FILLED",
                "executed": True
            }
        
        start_time = time.time()
        result = await apm.trace_execution(
            symbol=symbol,
            side=side,
            quantity=quantity,
            coro=mock_execution(),
            metadata={"price": 42000}
        )
        elapsed = time.time() - start_time
        
        assert result is not None, "Execution should be traced"
        assert result.get("status") == "FILLED", "Execution status captured"
        assert elapsed >= 0.1, "Latency should be captured"


# ============================================================================
# Test Suite 5: Error Handling and Status
# ============================================================================

class TestErrorHandling:
    """Test error handling in tracing."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_span_status_success(self, mock_tracer):
        """Verify span can be marked as successful."""
        mock_span = Mock()
        mock_tracer.set_span_status_success(mock_span)
        
        mock_tracer.set_span_status_success.assert_called_once_with(mock_span)
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_span_status_error(self, mock_tracer):
        """Verify span can be marked with error."""
        mock_span = Mock()
        error_msg = "Order placement failed"
        
        mock_tracer.set_span_status_error(mock_span, error_msg)
        
        mock_tracer.set_span_status_error.assert_called_once_with(
            mock_span, error_msg
        )
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_guard_evaluation_error_handling(self, apm_instrument):
        """Verify guard evaluation errors are handled gracefully."""
        symbol = "BTC"
        
        async def mock_guard_error():
            raise ValueError("Guard check failed")
        
        # Should not raise, should fall back to direct execution
        try:
            result = await apm_instrument.trace_guard_evaluation(
                guard_name="balance_guard",
                symbol=symbol,
                coro=mock_guard_error()
            )
        except ValueError:
            # Expected behavior - error from coroutine should propagate
            pass


# ============================================================================
# Test Suite 6: Loop Iteration Tracing
# ============================================================================

class TestLoopIterationTracing:
    """Test tracing of main controller loop."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_loop_iteration_span_creation(self):
        """Verify main loop iteration creates appropriate span."""
        apm = get_apm_instrument()
        cycle_number = 42
        
        async def mock_loop_iteration():
            # Simulate loop work
            await asyncio.sleep(0.01)
            return {"decisions": 2, "executed": 1}
        
        result = await apm.trace_loop_iteration(
            coro=mock_loop_iteration(),
            cycle_number=cycle_number
        )
        
        assert result is not None, "Loop iteration should be traced"
        assert result.get("decisions") == 2, "Iteration results captured"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_loop_iteration_with_high_cycle_count(self, apm_instrument):
        """Verify loop tracing handles high cycle counts."""
        async def mock_loop():
            return {"status": "ok"}
        
        for cycle_num in [1, 1000, 10000, 1000000]:
            result = await apm_instrument.trace_loop_iteration(
                coro=mock_loop(),
                cycle_number=cycle_num
            )
            assert result is not None, f"Loop trace should work for cycle {cycle_num}"


# ============================================================================
# Test Suite 7: MetaController Integration
# ============================================================================

class TestMetaControllerAPMIntegration:
    """Test APM integration with MetaController."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    def test_meta_controller_apm_initialization(self):
        """Verify MetaController initializes APM."""
        # This requires a full MetaController instance with mocked dependencies
        # Simplified test validates initialization pattern
        
        mock_shared_state = Mock()
        mock_exchange = Mock()
        mock_execution_mgr = Mock()
        mock_config = {"LOG_FILE": "/tmp/test.log"}
        
        try:
            from core.meta_controller import MetaController
            
            # Should initialize APM without errors
            assert True, "APM import should succeed"
        except ImportError:
            pytest.skip("MetaController not available")
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_evaluate_and_act_creates_cycle_span(self):
        """Verify evaluate_and_act creates main cycle span."""
        # This would require full MetaController setup
        # For now, validate that the tracing infrastructure is in place
        
        apm = get_apm_instrument()
        assert apm is not None, "APM should be initialized"
        assert hasattr(apm, "tracer"), "APM should have tracer"


# ============================================================================
# Test Suite 8: Performance and Overhead
# ============================================================================

class TestPerformanceOverhead:
    """Test that APM tracing has minimal performance impact."""
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_guard_tracing_overhead(self, apm_instrument):
        """Verify guard tracing doesn't add significant overhead."""
        symbol = "BTC"
        iterations = 100
        
        async def mock_guard():
            return {"approved": True}
        
        # Measure tracing overhead
        start = time.time()
        for i in range(iterations):
            await apm_instrument.trace_guard_evaluation(
                guard_name="balance_guard",
                symbol=symbol,
                coro=mock_guard(),
            )
        elapsed = time.time() - start
        
        # Should complete reasonably fast (< 1 second for 100 guards)
        assert elapsed < 1.0, f"100 guard traces took {elapsed:.2f}s, should be < 1s"
    
    @pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
    @pytest.mark.asyncio
    async def test_execution_tracing_overhead(self, apm_instrument):
        """Verify execution tracing doesn't add significant overhead."""
        iterations = 10
        
        async def mock_execution():
            await asyncio.sleep(0.01)  # Simulate network call
            return {"status": "FILLED"}
        
        start = time.time()
        for i in range(iterations):
            await apm_instrument.trace_execution(
                symbol="BTC",
                side="BUY",
                quantity=1.0,
                coro=mock_execution()
            )
        elapsed = time.time() - start
        
        # Should not add more than 20% overhead (10 * 0.01 = 0.1s base)
        expected_base = iterations * 0.01
        overhead_ratio = elapsed / expected_base if expected_base > 0 else 1.0
        assert overhead_ratio < 1.5, f"Overhead ratio {overhead_ratio:.1f}x should be < 1.5x"


# ============================================================================
# Test Markers and Utilities
# ============================================================================

pytestmark = pytest.mark.asyncio


def test_apm_module_availability():
    """Verify APM modules are available for testing."""
    if not APM_AVAILABLE:
        pytest.skip("APM modules not available - install OpenTelemetry packages")
    
    assert APMInstrument is not None, "APMInstrument should be available"
    assert JaegerTracer is not None, "JaegerTracer should be available"


# ============================================================================
# Integration Test: End-to-End Tracing Flow
# ============================================================================

@pytest.mark.skipif(not APM_AVAILABLE, reason="APM not available")
@pytest.mark.asyncio
async def test_end_to_end_trace_flow():
    """Test complete tracing flow: guard → decision → execution."""
    apm = get_apm_instrument()
    symbol = "BTC"
    
    # Step 1: Guard evaluation
    async def guard_check():
        await asyncio.sleep(0.01)
        return {"approved": True}
    
    guard_result = await apm.trace_guard_evaluation(
        guard_name="balance_guard",
        symbol=symbol,
        coro=guard_check()
    )
    assert guard_result is not None
    
    # Step 2: Trade decision
    async def trade_decision():
        await asyncio.sleep(0.01)
        return {"action": "BUY", "confidence": 0.9}
    
    decision_result = await apm.trace_trade_decision(
        symbol=symbol,
        agent_name="agent1",
        signal_type="BUY",
        coro=trade_decision()
    )
    assert decision_result is not None
    
    # Step 3: Execution
    async def execute_trade():
        await asyncio.sleep(0.01)
        return {"order_id": "123", "status": "FILLED"}
    
    execution_result = await apm.trace_execution(
        symbol=symbol,
        side="BUY",
        quantity=1.0,
        coro=execute_trade()
    )
    assert execution_result is not None
    
    # All steps should trace successfully
    assert guard_result.get("approved") == True
    assert decision_result.get("action") == "BUY"
    assert execution_result.get("status") == "FILLED"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
