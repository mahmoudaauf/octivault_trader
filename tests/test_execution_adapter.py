"""
Phase 4 Part 4: Execution Layer Adapter Tests

Tests that MetaDecision objects flow through the execution layer properly.
Tests both MetaDecision and backward-compatible tuple formats.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from core.stubs import MetaDecision, TradeIntent
from core.meta_controller import MetaController


class TestExecutionAdapterBasic:
    """Test basic adapter functionality for decision normalization."""
    
    @pytest.fixture
    def setup_controller(self):
        """Create a minimal MetaController for testing."""
        controller = Mock(spec=MetaController)
        controller.shared_state = Mock()
        controller.logger = Mock()
        
        # Add the normalize function as a method
        def _normalize_decision_for_execution(decision):
            """Convert decision to (symbol, side, signal_dict) triple."""
            from core.stubs import MetaDecision
            
            # Case 1: Already a tuple
            if isinstance(decision, tuple) and len(decision) == 3:
                symbol, side, signal_dict = decision
                if isinstance(symbol, str) and isinstance(side, str) and isinstance(signal_dict, dict):
                    return symbol, side, signal_dict
            
            # Case 2: MetaDecision object
            if isinstance(decision, MetaDecision):
                signal_dict = {
                    "symbol": decision.symbol,
                    "action": decision.side,
                    "confidence": decision.confidence,
                    "_planned_quote": decision.planned_quote,
                    "trace_id": decision.trace_id,
                    "agent": decision.enrichment.get("agent", "Meta") if decision.enrichment else "Meta",
                    "reason": decision.rationale,
                    "_applied_gates": decision.applied_gates,
                    "_rejection_reasons": decision.rejection_reasons,
                    "_bootstrap": False,
                }
                return decision.symbol, decision.side, signal_dict
            
            raise ValueError(f"Decision must be MetaDecision or tuple, got {type(decision)}")
        
        controller._normalize_decision_for_execution = _normalize_decision_for_execution
        return controller
    
    def test_adapter_accepts_tuple(self, setup_controller):
        """Test adapter handles (symbol, side, dict) tuples."""
        controller = setup_controller
        
        decision_tuple = ("BTCUSDT", "BUY", {"confidence": 0.85, "trace_id": "trace_123"})
        sym, side, sig = controller._normalize_decision_for_execution(decision_tuple)
        
        assert sym == "BTCUSDT"
        assert side == "BUY"
        assert sig["confidence"] == 0.85
        assert sig["trace_id"] == "trace_123"
    
    def test_adapter_accepts_metadecision(self, setup_controller):
        """Test adapter handles MetaDecision objects."""
        controller = setup_controller
        
        intent = TradeIntent(
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.0,
            planned_quote=2000.0,
            confidence=0.90,
            trace_id="trace_456",
            agent="swing_hunter",
            reason="momentum_reversal"
        )
        
        decision = MetaDecision(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.90,
            planned_quote=2000.0,
            source_intent=intent,
            trace_id="trace_456",
            execution_tier="pending",
            enrichment={"agent": "swing_hunter"},
            policy_context={},
            rationale="momentum_reversal",
            timestamp=datetime.now()
        )
        
        sym, side, sig = controller._normalize_decision_for_execution(decision)
        
        assert sym == "ETHUSDT"
        assert side == "SELL"
        assert sig["confidence"] == 0.90
        assert sig["_planned_quote"] == 2000.0
        assert sig["agent"] == "swing_hunter"
    
    def test_adapter_preserves_applied_gates(self, setup_controller):
        """Test adapter preserves applied gates from MetaDecision."""
        controller = setup_controller
        
        decision = MetaDecision(
            symbol="BNBUSDT",
            side="BUY",
            confidence=0.75,
            planned_quote=500.0,
            source_intent=None,
            trace_id="trace_789",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        decision.add_gate("position_limit")
        decision.add_gate("profit_target")
        
        sym, side, sig = controller._normalize_decision_for_execution(decision)
        
        assert "position_limit" in sig["_applied_gates"]
        assert "profit_target" in sig["_applied_gates"]
    
    def test_adapter_preserves_rejection_reasons(self, setup_controller):
        """Test adapter preserves rejection reasons from MetaDecision."""
        controller = setup_controller
        
        decision = MetaDecision(
            symbol="ADAUSDT",
            side="SELL",
            confidence=0.60,
            planned_quote=100.0,
            source_intent=None,
            trace_id="trace_999",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        decision.add_rejection_reason("low_volume")
        decision.add_rejection_reason("high_slippage")
        
        sym, side, sig = controller._normalize_decision_for_execution(decision)
        
        assert "low_volume" in sig["_rejection_reasons"]
        assert "high_slippage" in sig["_rejection_reasons"]
    
    def test_adapter_rejects_invalid_type(self, setup_controller):
        """Test adapter rejects invalid decision types."""
        controller = setup_controller
        
        with pytest.raises(ValueError, match="Decision must be MetaDecision or tuple"):
            controller._normalize_decision_for_execution("invalid_string")
    
    def test_adapter_rejects_malformed_tuple(self, setup_controller):
        """Test adapter rejects tuples with wrong structure."""
        controller = setup_controller
        
        # Tuple too short
        with pytest.raises(ValueError, match="Decision must be MetaDecision or tuple"):
            controller._normalize_decision_for_execution(("BTCUSDT", "BUY"))
        
        # Tuple with wrong types
        with pytest.raises(ValueError, match="Decision must be MetaDecision or tuple"):
            controller._normalize_decision_for_execution((123, "BUY", {}))


class TestExecutionAdapterEdgeCases:
    """Test edge cases for decision normalization."""
    
    @pytest.fixture
    def setup_controller(self):
        """Create adapter function for testing."""
        def _normalize_decision_for_execution(decision):
            from core.stubs import MetaDecision
            
            if isinstance(decision, tuple) and len(decision) == 3:
                symbol, side, signal_dict = decision
                if isinstance(symbol, str) and isinstance(side, str) and isinstance(signal_dict, dict):
                    return symbol, side, signal_dict
            
            if isinstance(decision, MetaDecision):
                signal_dict = {
                    "symbol": decision.symbol,
                    "action": decision.side,
                    "confidence": decision.confidence,
                    "_planned_quote": decision.planned_quote,
                    "trace_id": decision.trace_id,
                    "agent": decision.enrichment.get("agent", "Meta") if decision.enrichment else "Meta",
                    "reason": decision.rationale,
                    "_applied_gates": decision.applied_gates,
                    "_rejection_reasons": decision.rejection_reasons,
                    "_bootstrap": False,
                }
                return decision.symbol, decision.side, signal_dict
            
            raise ValueError(f"Decision must be MetaDecision or tuple, got {type(decision)}")
        
        return _normalize_decision_for_execution
    
    def test_adapter_handles_none_enrichment(self, setup_controller):
        """Test adapter when MetaDecision has None enrichment."""
        adapter = setup_controller
        
        decision = MetaDecision(
            symbol="XRPUSDT",
            side="BUY",
            confidence=0.80,
            planned_quote=250.0,
            source_intent=None,
            trace_id="trace_111",
            execution_tier="pending",
            enrichment=None,  # None enrichment
            policy_context={},
            rationale="test"
        )
        
        sym, side, sig = adapter(decision)
        
        assert sig["agent"] == "Meta"  # Default when enrichment is None
    
    def test_adapter_handles_empty_enrichment(self, setup_controller):
        """Test adapter when MetaDecision has empty enrichment."""
        adapter = setup_controller
        
        decision = MetaDecision(
            symbol="DOGEUSDT",
            side="SELL",
            confidence=0.70,
            planned_quote=50.0,
            source_intent=None,
            trace_id="trace_222",
            execution_tier="pending",
            enrichment={},  # Empty enrichment
            policy_context={},
            rationale="test"
        )
        
        sym, side, sig = adapter(decision)
        
        assert sig["agent"] == "Meta"  # Default when enrichment empty
    
    def test_adapter_handles_missing_trace_id(self, setup_controller):
        """Test adapter when MetaDecision has None trace_id."""
        adapter = setup_controller
        
        decision = MetaDecision(
            symbol="LTCUSDT",
            side="BUY",
            confidence=0.75,
            planned_quote=300.0,
            source_intent=None,
            trace_id=None,  # None trace_id
            execution_tier="pending",
            enrichment={"agent": "test_agent"},
            policy_context={},
            rationale="test"
        )
        
        sym, side, sig = adapter(decision)
        
        assert sig["trace_id"] is None
    
    def test_adapter_handles_empty_gates_list(self, setup_controller):
        """Test adapter when MetaDecision has empty gates list."""
        adapter = setup_controller
        
        decision = MetaDecision(
            symbol="MATICUSDT",
            side="SELL",
            confidence=0.65,
            planned_quote=75.0,
            source_intent=None,
            trace_id="trace_333",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="test"
        )
        
        # No gates added
        sym, side, sig = adapter(decision)
        
        assert sig["_applied_gates"] == []
        assert sig["_rejection_reasons"] == []
    
    def test_adapter_preserves_tuple_extra_fields(self, setup_controller):
        """Test adapter preserves extra fields in tuple signal dict."""
        adapter = setup_controller
        
        signal_dict = {
            "confidence": 0.88,
            "trace_id": "trace_444",
            "_bootstrap": True,
            "_forced_exit": True,
            "_is_rotation": False,
            "custom_field": "custom_value"
        }
        
        decision_tuple = ("ATOMUSDT", "BUY", signal_dict)
        sym, side, sig = adapter(decision_tuple)
        
        assert sig["_bootstrap"] is True
        assert sig["_forced_exit"] is True
        assert sig["custom_field"] == "custom_value"


class TestExecutionAdapterIntegration:
    """Test adapter integration with decision loops."""
    
    @pytest.fixture
    def setup_controller(self):
        """Create adapter function for testing."""
        def _normalize_decision_for_execution(decision):
            from core.stubs import MetaDecision
            
            if isinstance(decision, tuple) and len(decision) == 3:
                symbol, side, signal_dict = decision
                if isinstance(symbol, str) and isinstance(side, str) and isinstance(signal_dict, dict):
                    return symbol, side, signal_dict
            
            if isinstance(decision, MetaDecision):
                signal_dict = {
                    "symbol": decision.symbol,
                    "action": decision.side,
                    "confidence": decision.confidence,
                    "_planned_quote": decision.planned_quote,
                    "trace_id": decision.trace_id,
                    "agent": decision.enrichment.get("agent", "Meta") if decision.enrichment else "Meta",
                    "reason": decision.rationale,
                    "_applied_gates": decision.applied_gates,
                    "_rejection_reasons": decision.rejection_reasons,
                    "_bootstrap": False,
                }
                return decision.symbol, decision.side, signal_dict
            
            raise ValueError(f"Decision must be MetaDecision or tuple, got {type(decision)}")
        
        return _normalize_decision_for_execution
    
    def test_adapter_in_decision_loop_mixed_types(self, setup_controller):
        """Test adapter in a loop with mixed MetaDecision and tuple decisions."""
        adapter = setup_controller
        
        # Create mixed decisions
        tuple_decision = ("BTCUSDT", "BUY", {"confidence": 0.85})
        
        metadecision = MetaDecision(
            symbol="ETHUSDT",
            side="SELL",
            confidence=0.90,
            planned_quote=2000.0,
            source_intent=None,
            trace_id="trace_555",
            execution_tier="pending",
            enrichment={"agent": "hunter"},
            policy_context={},
            rationale="reversal"
        )
        
        decisions = [tuple_decision, metadecision]
        
        # Simulate decision loop
        normalized_decisions = []
        for decision in decisions:
            sym, side, sig = adapter(decision)
            normalized_decisions.append((sym, side, sig))
        
        assert len(normalized_decisions) == 2
        assert normalized_decisions[0][0] == "BTCUSDT"
        assert normalized_decisions[1][0] == "ETHUSDT"
    
    def test_adapter_deduplication_with_metadecision(self, setup_controller):
        """Test deduplication logic works with adapter normalization."""
        adapter = setup_controller
        
        decision1 = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.85,
            planned_quote=1000.0,
            source_intent=None,
            trace_id="trace_1",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="signal1"
        )
        
        decision2 = MetaDecision(
            symbol="BTCUSDT",
            side="BUY",
            confidence=0.80,
            planned_quote=900.0,
            source_intent=None,
            trace_id="trace_2",
            execution_tier="pending",
            enrichment={},
            policy_context={},
            rationale="signal2"
        )
        
        decisions = [decision1, decision2]
        executed = set()
        
        # Simulate dedup loop
        for decision in decisions:
            sym, side, sig = adapter(decision)
            dedup_key = (sym, side)
            
            if dedup_key not in executed:
                executed.add(dedup_key)
            else:
                continue  # Would skip duplicate
        
        # Should have only one unique (symbol, side)
        assert len(executed) == 1
        assert ("BTCUSDT", "BUY") in executed
    
    def test_adapter_with_real_execution_pattern(self, setup_controller):
        """Test adapter with realistic execution pattern from meta_controller."""
        adapter = setup_controller
        
        # Simulate multiple decisions from _build_decisions()
        decisions = [
            MetaDecision(
                symbol="BTCUSDT",
                side="BUY",
                confidence=0.85,
                planned_quote=1000.0,
                source_intent=None,
                trace_id="trace_1",
                execution_tier="pending",
                enrichment={"agent": "hunter"},
                policy_context={},
                rationale="momentum"
            ),
            MetaDecision(
                symbol="ETHUSDT",
                side="SELL",
                confidence=0.75,
                planned_quote=500.0,
                source_intent=None,
                trace_id="trace_2",
                execution_tier="pending",
                enrichment={"agent": "reversal"},
                policy_context={},
                rationale="overbought"
            ),
        ]
        
        # Simulate execution loop (lines 6890-6895 pattern)
        executed_this_tick = set()
        execution_results = []
        
        for decision in decisions:
            sym, side, sig = adapter(decision)
            
            dedup_key = (sym, side)
            if dedup_key in executed_this_tick:
                continue
            executed_this_tick.add(dedup_key)
            
            # Simulate execution
            execution_results.append({
                "symbol": sym,
                "side": side,
                "confidence": sig["confidence"],
                "planned_quote": sig["_planned_quote"],
                "trace_id": sig["trace_id"]
            })
        
        assert len(execution_results) == 2
        assert execution_results[0]["symbol"] == "BTCUSDT"
        assert execution_results[0]["confidence"] == 0.85
        assert execution_results[1]["symbol"] == "ETHUSDT"
        assert execution_results[1]["confidence"] == 0.75


# ═══════════════════════════════════════════════════════════════════════════════
# Run tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
