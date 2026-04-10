"""
Tests for MetaDecision class (Phase 4 Part 1)

Test-Driven Development approach:
1. Write tests first
2. Run (should fail initially)
3. Implement code
4. Run (should pass)
"""

import pytest
import time
from core.stubs import MetaDecision, TradeIntent


class TestMetaDecisionCreation:
    """Test basic MetaDecision creation and validation"""
    
    def test_create_approved_decision(self):
        """Test creating an approved buy decision"""
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=100.0,
            agent="DipSniper",
            reason="Dip detected",
        )
        
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=100.0,
            source_intent=intent,
            trace_id="test-trace-001",
            applied_gates=["min_confidence", "concentration"],
            execution_tier="immediate",
            enrichment={"dip_percent": 2.5},
            rationale="Test decision",
        )
        
        assert decision.symbol == "BTCUSDT"
        assert decision.side == "BUY"
        assert decision.confidence == 0.85
        assert decision.planned_quote == 100.0
        assert decision.is_approved
        assert not decision.is_rejected
        assert len(decision.applied_gates) == 2
    
    def test_create_rejected_decision(self):
        """Test creating a rejected decision"""
        intent = TradeIntent(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.95,
        )
        
        decision = MetaDecision(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.95,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-trace-002",
            execution_tier="rejected",
            rejection_reasons=["position_limit_exceeded"],
        )
        
        assert decision.is_rejected
        assert not decision.is_approved
        assert "position_limit_exceeded" in decision.rejection_reasons
    
    def test_timestamp_auto_set(self):
        """Test that timestamp is auto-set if not provided"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        before = time.time()
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
        )
        after = time.time()
        
        assert before <= decision.timestamp <= after
    
    def test_default_collections_initialized(self):
        """Test that list/dict fields are initialized"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
        )
        
        assert isinstance(decision.applied_gates, list)
        assert isinstance(decision.rejection_reasons, list)
        assert isinstance(decision.enrichment, dict)
        assert isinstance(decision.policy_context, dict)


class TestMetaDecisionValidation:
    """Test MetaDecision validation"""
    
    def test_invalid_side(self):
        """Test that invalid side raises ValueError"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        with pytest.raises(ValueError, match="Invalid side"):
            MetaDecision(
                symbol="BTCUSDT",
                side="INVALID",
                confidence=0.5,
                planned_quote=50.0,
                source_intent=intent,
                trace_id="test-123",
            )
    
    def test_invalid_confidence(self):
        """Test that confidence outside 0-1 raises ValueError"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        with pytest.raises(ValueError, match="Confidence must be"):
            MetaDecision(
                symbol="BTCUSDT",
                side="BUY",
                confidence=1.5,  # > 1.0
                planned_quote=50.0,
                source_intent=intent,
                trace_id="test-123",
            )
    
    def test_invalid_execution_tier(self):
        """Test that invalid execution_tier raises ValueError"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        with pytest.raises(ValueError, match="Invalid execution_tier"):
            MetaDecision(
                symbol="BTCUSDT",
                side="BUY",
                confidence=0.5,
                planned_quote=50.0,
                source_intent=intent,
                trace_id="test-123",
                execution_tier="invalid_tier",
            )


class TestMetaDecisionMethods:
    """Test MetaDecision methods"""
    
    def test_add_gate(self):
        """Test adding gates to applied_gates"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
        )
        
        assert len(decision.applied_gates) == 0
        decision.add_gate("min_confidence")
        assert "min_confidence" in decision.applied_gates
        assert len(decision.applied_gates) == 1
        
        # Adding same gate twice should not duplicate
        decision.add_gate("min_confidence")
        assert len(decision.applied_gates) == 1
    
    def test_add_rejection_reason(self):
        """Test adding rejection reasons"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
        )
        
        assert not decision.is_rejected
        decision.add_rejection_reason("position_limit_exceeded")
        assert decision.is_rejected
        assert "position_limit_exceeded" in decision.rejection_reasons
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            agent="DipSniper",
        )
        
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=100.0,
            source_intent=intent,
            trace_id="test-123",
            applied_gates=["min_confidence"],
            enrichment={"dip_percent": 2.5},
            rationale="Test",
        )
        
        d = decision.to_dict()
        
        assert isinstance(d, dict)
        assert d["symbol"] == "BTCUSDT"
        assert d["side"] == "BUY"
        assert d["confidence"] == 0.85
        assert d["source_intent_agent"] == "DipSniper"
        assert d["trace_id"] == "test-123"
        assert d["enrichment"]["dip_percent"] == 2.5


class TestMetaDecisionProperties:
    """Test MetaDecision properties"""
    
    def test_is_approved_property(self):
        """Test is_approved property"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        # Approved
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
            execution_tier="immediate",
        )
        assert decision.is_approved
        
        # Rejected
        decision.execution_tier = "rejected"
        assert not decision.is_approved
    
    def test_is_rejected_property(self):
        """Test is_rejected property"""
        intent = TradeIntent(symbol="BTCUSDT", side="BUY")
        
        # Not rejected
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
            execution_tier="immediate",
        )
        assert not decision.is_rejected
        
        # Rejected
        decision.execution_tier = "rejected"
        assert decision.is_rejected


class TestMetaDecisionTraceability:
    """Test traceability features"""
    
    def test_trace_id_preserved(self):
        """Test that trace_id is preserved through decision"""
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            trace_id="intent-trace-001",
        )
        
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.5,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="intent-trace-001",  # Should match source intent
        )
        
        assert decision.trace_id == intent.trace_id
    
    def test_source_intent_reference(self):
        """Test that source intent is properly referenced"""
        intent = TradeIntent(
            symbol="BTCUSDT",
            side="BUY",
            agent="DipSniper",
            confidence=0.85,
        )
        
        decision = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=50.0,
            source_intent=intent,
            trace_id="test-123",
        )
        
        assert decision.source_intent == intent
        assert decision.source_intent.agent == "DipSniper"
        assert decision.source_intent.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
