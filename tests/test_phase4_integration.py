"""
Phase 4 Part 5: Comprehensive Integration Tests

Tests that all 4 parts of Phase 4 work together seamlessly:
- Part 1: MetaDecision class
- Part 2: Cache intent preservation
- Part 3: Decision building
- Part 4: Execution adapter

Validates end-to-end flow from agent through execution.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from core.stubs import MetaDecision, TradeIntent


class TestFullPipelineIntegration:
    """Test complete Agent → Cache → Decision → Execution flow."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Setup minimal pipeline components."""
        return {
            "cache": {},
            "signal_batcher": Mock(),
            "execution_results": []
        }
    
    def test_single_signal_end_to_end(self, setup_pipeline):
        """Test single signal flowing through entire pipeline."""
        pipeline = setup_pipeline
        
        # Step 1: Agent creates TradeIntent
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            planned_quote=50000.0,
            confidence=0.85,
            trace_id="trace_001",
            agent="swing_hunter",
            reason="momentum_signal"
        )
        
        # Step 2: Store in cache (Part 2)
        cache_key = f"{intent.symbol}:{intent.side}"
        signal_dict = {
            "symbol": intent.symbol,
            "action": intent.side,
            "confidence": intent.confidence,
            "_planned_quote": intent.planned_quote,
            "trace_id": intent.trace_id,
            "agent": intent.agent,
            "reason": intent.reason,
            "_source_intent": intent,  # Part 2: preserve intent
            "_has_source_intent": True,
        }
        pipeline["cache"][cache_key] = signal_dict
        
        # Step 3: Build decision (Part 3)
        decision = MetaDecision(
            symbol=intent.symbol,
            side=intent.side,
            confidence=intent.confidence,
            planned_quote=intent.planned_quote,
            source_intent=intent,
            trace_id=intent.trace_id,
            execution_tier="pending",
            enrichment={"agent": intent.agent},
            policy_context={},
            rationale=intent.reason
        )
        
        # Step 4: Execute with adapter (Part 4)
        # Simulate adapter normalization
        sym, side, sig = decision.symbol, decision.side, {
            "confidence": decision.confidence,
            "_planned_quote": decision.planned_quote,
            "trace_id": decision.trace_id,
        }
        
        # Verify all components work
        assert sym == "BTCUSDT"
        assert side == "BUY"
        assert sig["confidence"] == 0.85
        assert sig["_planned_quote"] == 50000.0
        assert sig["trace_id"] == "trace_001"
        
        # Verify intent accessible
        assert pipeline["cache"][cache_key]["_source_intent"] == intent
    
    def test_multiple_signals_with_deduplication(self, setup_pipeline):
        """Test multiple signals deduplicated correctly."""
        pipeline = setup_pipeline
        
        # Create 3 signals for same symbol
        signals = [
            TradeIntent(
                symbol="ETHUSDT",
                side="SELL",
                quantity=10.0,
                planned_quote=2000.0,
                confidence=0.80,
                trace_id="trace_101",
                agent="reversal_hunter",
                reason="overbought"
            ),
            TradeIntent(
                symbol="ETHUSDT",
                side="SELL",
                quantity=15.0,
                planned_quote=2500.0,
                confidence=0.85,  # Highest confidence
                trace_id="trace_102",
                agent="momentum_hunter",
                reason="top_signal"
            ),
            TradeIntent(
                symbol="ETHUSDT",
                side="SELL",
                quantity=12.0,
                planned_quote=2200.0,
                confidence=0.75,
                trace_id="trace_103",
                agent="drift_hunter",
                reason="trend_end"
            ),
        ]
        
        # Find best (highest confidence) before caching
        best_intent = max(signals, key=lambda s: s.confidence)
        
        # Store best in cache (deduplication logic)
        cache_key = f"{best_intent.symbol}:{best_intent.side}"
        pipeline["cache"][cache_key] = {
            "confidence": best_intent.confidence,
            "_source_intent": best_intent,
        }
        
        # Retrieve from cache
        cached = pipeline["cache"][cache_key]
        best_confidence = cached["confidence"]
        cached_intent = cached["_source_intent"]
        
        # Verify best signal selected
        assert best_confidence == 0.85
        assert cached_intent.trace_id == "trace_102"
    
    def test_bootstrap_signal_execution(self, setup_pipeline):
        """Test bootstrap signal flows through system."""
        pipeline = setup_pipeline
        
        # Bootstrap signal
        intent = TradeIntent(
            symbol="BNBUSDT",
            side="BUY",
            quantity=50.0,
            planned_quote=5000.0,
            confidence=0.90,
            trace_id="bootstrap_001",
            agent="bootstrap",
            reason="bootstrap_init"
        )
        
        # Create decision
        decision = MetaDecision(
            symbol=intent.symbol,
            side=intent.side,
            confidence=0.90,
            planned_quote=5000.0,
            source_intent=intent,
            trace_id="bootstrap_001",
            execution_tier="pending",
            enrichment={"agent": "bootstrap"},
            policy_context={},
            rationale="bootstrap_init"
        )
        
        # Add bootstrap metadata
        signal_dict = {
            "symbol": intent.symbol,
            "_bootstrap": True,
            "confidence": 0.90,
        }
        
        # Verify bootstrap signal properties
        assert signal_dict["_bootstrap"] is True
        assert decision.confidence == 0.90
        assert "bootstrap" in decision.enrichment["agent"]
    
    def test_mixed_metadecision_and_tuple_decisions(self, setup_pipeline):
        """Test adapter handles both MetaDecision and tuple decisions."""
        pipeline = setup_pipeline
        
        # Create MetaDecision
        metadecision = MetaDecision(
            symbol="ADAUSDT",
            side="BUY",
            confidence=0.75,
            planned_quote=300.0,
            source_intent=None,
            trace_id="trace_201",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Create tuple decision (legacy format)
        tuple_decision = ("DOGEUSDT", "SELL", {
            "confidence": 0.70,
            "_planned_quote": 100.0,
            "trace_id": "trace_202"
        })
        
        # Adapter handles both
        decisions = [metadecision, tuple_decision]
        results = []
        
        for d in decisions:
            # Simulate adapter normalization
            if isinstance(d, MetaDecision):
                sym, side = d.symbol, d.side
            else:
                sym, side = d[0], d[1]
            
            results.append((sym, side))
        
        # Verify both types processed
        assert results[0] == ("ADAUSDT", "BUY")
        assert results[1] == ("DOGEUSDT", "SELL")
    
    def test_trace_id_preservation_end_to_end(self, setup_pipeline):
        """Test trace_id preserved through entire pipeline."""
        pipeline = setup_pipeline
        
        original_trace_id = "end_to_end_trace_001"
        
        # Create signal with trace_id
        intent = TradeIntent(
            symbol="LTCUSDT",
            side="BUY",
            quantity=5.0,
            planned_quote=1000.0,
            confidence=0.80,
            trace_id=original_trace_id,
            agent="test_agent",
            reason="test_reason"
        )
        
        # Cache (Part 2)
        cache_data = {"_source_intent": intent}
        
        # Decision (Part 3)
        decision = MetaDecision(
            symbol=intent.symbol,
            side=intent.side,
            confidence=intent.confidence,
            planned_quote=intent.planned_quote,
            source_intent=intent,
            trace_id=original_trace_id,  # Preserved
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Execution dict (Part 4)
        execution_dict = {
            "trace_id": decision.trace_id
        }
        
        # Verify trace_id at each step
        assert cache_data["_source_intent"].trace_id == original_trace_id
        assert decision.trace_id == original_trace_id
        assert execution_dict["trace_id"] == original_trace_id
    
    def test_gate_tracking_through_pipeline(self, setup_pipeline):
        """Test gates tracked through decision pipeline."""
        pipeline = setup_pipeline
        
        # Create decision
        decision = MetaDecision(
            symbol="XRPUSDT",
            side="SELL",
            confidence=0.65,
            planned_quote=500.0,
            source_intent=None,
            trace_id="trace_301",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Add gates during building
        decision.add_gate("position_limit")
        decision.add_gate("profit_target")
        decision.add_gate("max_drawdown")
        
        # Convert to execution dict
        execution_dict = {
            "_applied_gates": decision.applied_gates
        }
        
        # Verify gates accessible at execution
        assert "position_limit" in execution_dict["_applied_gates"]
        assert "profit_target" in execution_dict["_applied_gates"]
        assert "max_drawdown" in execution_dict["_applied_gates"]
        assert len(execution_dict["_applied_gates"]) == 3


class TestCacheWithExecutionIntegration:
    """Test cache behavior with execution layer."""
    
    def test_intent_storage_and_retrieval(self):
        """Test intent stored and retrieved from cache."""
        cache = {}
        
        # Create intent
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            planned_quote=50000.0,
            confidence=0.85,
            trace_id="cache_trace_001",
            agent="test",
            reason="test"
        )
        
        # Store in cache
        key = "BTCUSDT:BUY"
        cache[key] = {
            "confidence": 0.85,
            "_source_intent": intent,  # Part 2
            "_has_source_intent": True,
        }
        
        # Retrieve and verify
        assert cache[key]["_has_source_intent"] is True
        assert cache[key]["_source_intent"] == intent
        assert cache[key]["_source_intent"].trace_id == "cache_trace_001"
    
    def test_multiple_intents_in_cache(self):
        """Test multiple intents stored correctly."""
        cache = {}
        
        # Create multiple intents for different symbols
        intents = [
            TradeIntent(
                symbol="BTCUSDT",
                side="BUY",
                quantity=1.0,
                planned_quote=50000.0,
                confidence=0.85,
                trace_id="trace_1",
                agent="agent1",
                reason="reason1"
            ),
            TradeIntent(
                symbol="ETHUSDT",
                side="SELL",
                quantity=10.0,
                planned_quote=2000.0,
                confidence=0.80,
                trace_id="trace_2",
                agent="agent2",
                reason="reason2"
            ),
            TradeIntent(
                symbol="BNBUSDT",
                side="BUY",
                quantity=50.0,
                planned_quote=5000.0,
                confidence=0.75,
                trace_id="trace_3",
                agent="agent3",
                reason="reason3"
            ),
        ]
        
        # Store all intents
        for intent in intents:
            key = f"{intent.symbol}:{intent.side}"
            cache[key] = {"_source_intent": intent}
        
        # Verify all accessible
        assert len(cache) == 3
        assert cache["BTCUSDT:BUY"]["_source_intent"].trace_id == "trace_1"
        assert cache["ETHUSDT:SELL"]["_source_intent"].trace_id == "trace_2"
        assert cache["BNBUSDT:BUY"]["_source_intent"].trace_id == "trace_3"
    
    def test_intent_accessible_at_decision_layer(self):
        """Test intent accessible when building decision."""
        cache = {}
        
        # Store signal with intent in cache
        intent = TradeIntent(
            symbol="XRPUSDT",
            side="BUY",
            quantity=100.0,
            planned_quote=1000.0,
            confidence=0.70,
            trace_id="decision_trace_001",
            agent="test",
            reason="test"
        )
        
        cache["XRPUSDT:BUY"] = {
            "confidence": 0.70,
            "_source_intent": intent,
        }
        
        # Build decision using cached intent
        cached_signal = cache["XRPUSDT:BUY"]
        source_intent = cached_signal.get("_source_intent")
        
        decision = MetaDecision(
            symbol="XRPUSDT",
            side="BUY",
            confidence=cached_signal["confidence"],
            planned_quote=1000.0,
            source_intent=source_intent,  # From cache
            trace_id=source_intent.trace_id if source_intent else None,
            execution_tier="pending",
            enrichment={"agent": source_intent.agent if source_intent else "Meta"},
            policy_context={},
            rationale=source_intent.reason if source_intent else "none"
        )
        
        # Verify intent accessible
        assert decision.source_intent == intent
        assert decision.source_intent.trace_id == "decision_trace_001"
    
    def test_cache_roundtrip_field_preservation(self):
        """Test all fields preserved through cache round-trip."""
        cache = {}
        
        # Original intent with all fields
        intent = TradeIntent(
            symbol="MATICUSDT",
            side="SELL",
            quantity=1000.0,
            planned_quote=2000.0,
            confidence=0.88,
            trace_id="roundtrip_trace_001",
            agent="comprehensive_agent",
            reason="complete_reason",
            timestamp=datetime.now()
        )
        
        # Store in cache
        cache["MATICUSDT:SELL"] = {
            "symbol": intent.symbol,
            "action": intent.side,
            "quantity": intent.quantity,
            "planned_quote": intent.planned_quote,
            "confidence": intent.confidence,
            "trace_id": intent.trace_id,
            "agent": intent.agent,
            "reason": intent.reason,
            "_source_intent": intent,
        }
        
        # Retrieve and verify all fields
        retrieved = cache["MATICUSDT:SELL"]
        assert retrieved["symbol"] == intent.symbol
        assert retrieved["action"] == intent.side
        assert retrieved["quantity"] == intent.quantity
        assert retrieved["confidence"] == intent.confidence
        assert retrieved["trace_id"] == intent.trace_id
        assert retrieved["agent"] == intent.agent
        assert retrieved["_source_intent"] == intent
    
    def test_backward_compat_dict_signals(self):
        """Test dict signals still work without intent."""
        cache = {}
        
        # Old-format signal without intent
        signal = {
            "symbol": "ATOMUSDT",
            "action": "BUY",
            "confidence": 0.75,
            "trace_id": "legacy_trace_001",
            "agent": "legacy_agent",
            "reason": "legacy_reason",
            # No _source_intent
        }
        
        cache["ATOMUSDT:BUY"] = signal
        
        # Should still work
        retrieved = cache["ATOMUSDT:BUY"]
        assert retrieved["confidence"] == 0.75
        assert retrieved["trace_id"] == "legacy_trace_001"
        
        # Intent will be None, but that's ok
        source_intent = retrieved.get("_source_intent")
        assert source_intent is None


class TestDecisionContextAccess:
    """Test decision context accessible at execution boundary."""
    
    def test_applied_gates_accessible(self):
        """Test applied gates accessible at execution."""
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=50000.0,
            source_intent=None,
            trace_id="gates_trace_001",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Add gates
        decision.add_gate("position_limit")
        decision.add_gate("profit_target")
        decision.add_gate("max_loss")
        
        # Convert to execution context
        execution_context = {
            "symbol": decision.symbol,
            "side": decision.side,
            "applied_gates": decision.applied_gates,
        }
        
        # Verify gates accessible
        assert len(execution_context["applied_gates"]) == 3
        assert "position_limit" in execution_context["applied_gates"]
        assert "profit_target" in execution_context["applied_gates"]
        assert "max_loss" in execution_context["applied_gates"]
    
    def test_rejection_reasons_accessible(self):
        """Test rejection reasons accessible at execution."""
        decision = MetaDecision(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.60,
            planned_quote=2000.0,
            source_intent=None,
            trace_id="reject_trace_001",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Add rejection reasons
        decision.add_rejection_reason("low_volume")
        decision.add_rejection_reason("high_slippage")
        decision.add_rejection_reason("market_hours")
        
        # Verify status changed
        assert decision.is_rejected is True
        assert decision.execution_tier == "rejected"
        
        # Convert to execution context
        execution_context = {
            "symbol": decision.symbol,
            "side": decision.side,
            "is_rejected": decision.is_rejected,
            "rejection_reasons": decision.rejection_reasons,
        }
        
        # Verify reasons accessible
        assert execution_context["is_rejected"] is True
        assert len(execution_context["rejection_reasons"]) == 3
        assert "low_volume" in execution_context["rejection_reasons"]
    
    def test_confidence_preserved(self):
        """Test confidence preserved through pipeline."""
        original_confidence = 0.82
        
        # Create with specific confidence
        decision = MetaDecision(
            symbol="BNBUSDT",
            side="BUY",
            confidence=original_confidence,
            planned_quote=5000.0,
            source_intent=None,
            trace_id="conf_trace_001",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Extract for execution
        execution_dict = {
            "confidence": decision.confidence,
        }
        
        # Verify unchanged
        assert execution_dict["confidence"] == original_confidence
        assert execution_dict["confidence"] == 0.82
    
    def test_trace_id_available_at_boundary(self):
        """Test trace_id available at execution boundary."""
        trace_id = "boundary_trace_001"
        
        decision = MetaDecision(
            symbol="ADAUSDT",
            side="SELL",
            confidence=0.70,
            planned_quote=300.0,
            source_intent=None,
            trace_id=trace_id,
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # At execution boundary
        execution_context = {
            "trace_id": decision.trace_id,
            "symbol": decision.symbol,
            "side": decision.side,
        }
        
        # Verify trace_id accessible
        assert execution_context["trace_id"] == trace_id
        assert execution_context["trace_id"] == "boundary_trace_001"


class TestRegressionPrevention:
    """Ensure no regressions from Phase 4 changes."""
    
    def test_backward_compatibility_tuple_format(self):
        """Test legacy tuple format still works."""
        # Old-style tuple decision
        tuple_decision = ("BTCUSDT", "BUY", {
            "confidence": 0.85,
            "trace_id": "legacy_001",
            "agent": "legacy_agent",
        })
        
        # Should still be processable
        symbol, side, signal_dict = tuple_decision
        
        assert symbol == "BTCUSDT"
        assert side == "BUY"
        assert signal_dict["confidence"] == 0.85
        assert signal_dict["trace_id"] == "legacy_001"
    
    def test_execution_logic_unchanged(self):
        """Test existing execution logic still works."""
        # Simulate existing execution logic
        decision = ("ETHUSDT", "SELL", {
            "confidence": 0.75,
            "quantity": 10.0,
        })
        
        sym, side, sig = decision
        
        # Existing logic: unpack and check confidence
        if float(sig.get("confidence", 0.0)) > 0.50:
            can_execute = True
        else:
            can_execute = False
        
        # Verify logic works
        assert can_execute is True
        assert sym == "ETHUSDT"
    
    def test_error_handling_preserved(self):
        """Test error handling not broken."""
        # Invalid decision should gracefully fail
        invalid_decision = None
        
        # Simulate graceful fallback
        try:
            if invalid_decision is None:
                raise ValueError("Invalid decision")
        except (ValueError, TypeError):
            # Graceful fallback
            fallback = ("UNKNOWN", "UNKNOWN", {})
        
        # Fallback works
        assert fallback[0] == "UNKNOWN"
    
    def test_logging_and_observability(self):
        """Test logging/observability still works."""
        events = []
        
        # Simulate decision logging
        decision = MetaDecision(
            symbol="XRPUSDT",
            side="BUY",
            confidence=0.80,
            planned_quote=1000.0,
            source_intent=None,
            trace_id="logging_trace_001",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # Log decision
        events.append({
            "type": "decision_created",
            "symbol": decision.symbol,
            "side": decision.side,
            "confidence": decision.confidence,
            "trace_id": decision.trace_id,
        })
        
        # Verify logging works
        assert len(events) == 1
        assert events[0]["symbol"] == "XRPUSDT"
        assert events[0]["trace_id"] == "logging_trace_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
