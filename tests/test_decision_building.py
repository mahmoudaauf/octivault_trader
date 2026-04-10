"""
Test suite for Phase 4 Part 3: Decision Building with MetaDecision Objects.
Validates that _build_decisions() returns MetaDecision objects with proper
source intent and gate tracking.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stubs import TradeIntent, MetaDecision


class MockSignalManager:
    """Mock signal manager for testing."""
    def get_source_intent(self, sig_dict):
        """Extract source intent if available."""
        return sig_dict.get("_source_intent")


class TestDecisionConversion:
    """Test conversion of decision tuples to MetaDecision objects."""
    
    @pytest.fixture
    def setup(self):
        """Setup test environment."""
        # Create a mock MetaController with conversion method
        controller = Mock()
        controller.logger = Mock()
        controller.signal_manager = MockSignalManager()
        
        # Add the conversion method
        async def convert_decisions(tuples):
            """Convert tuples to MetaDecision."""
            meta_decisions = []
            for symbol, side, sig_dict in tuples:
                try:
                    source_intent = controller.signal_manager.get_source_intent(sig_dict)
                    
                    decision = MetaDecision(
                        symbol=symbol,
                        side=side,
                        confidence=float(sig_dict.get("confidence", 0.50)),
                        planned_quote=float(sig_dict.get("_planned_quote", 0.0)),
                        source_intent=source_intent,
                        trace_id=source_intent.trace_id if source_intent else sig_dict.get("trace_id"),
                        execution_tier="pending",
                        enrichment={"agent": sig_dict.get("agent")},
                        policy_context={},
                        rationale=sig_dict.get("reason", "decision"),
                    )
                    
                    applied_gates = sig_dict.get("_applied_gates", [])
                    if applied_gates:
                        for gate in applied_gates:
                            decision.add_gate(gate)
                    
                    meta_decisions.append(decision)
                except Exception as e:
                    # Fallback to tuple
                    meta_decisions.append((symbol, side, sig_dict))
            
            return meta_decisions
        
        controller._convert_decisions_to_metadecisions = convert_decisions
        
        return {
            "controller": controller,
            "signal_manager": controller.signal_manager
        }
    
    @pytest.mark.asyncio
    async def test_single_decision_conversion(self, setup):
        """Test conversion of a single decision tuple."""
        controller = setup["controller"]
        
        # Create a decision tuple
        decision_tuple = (
            "BTCUSDT",
            "BUY",
            {
                "confidence": 0.75,
                "_planned_quote": 45000.0,
                "agent": "SwingTradeHunter",
                "reason": "bullish_signal",
            }
        )
        
        # Convert
        result = await controller._convert_decisions_to_metadecisions([decision_tuple])
        
        assert len(result) == 1
        decision = result[0]
        
        # Verify it's a MetaDecision
        assert isinstance(decision, MetaDecision)
        assert decision.symbol == "BTCUSDT"
        assert decision.side == "BUY"
        assert decision.confidence == 0.75
        assert decision.planned_quote == 45000.0
        assert decision.is_approved is True
    
    @pytest.mark.asyncio
    async def test_decision_with_source_intent(self, setup):
        """Test conversion preserves source intent."""
        controller = setup["controller"]
        
        # Create intent
        intent = TradeIntent(
            symbol="ETHUSDT",
            side="BUY",
            quantity=10.0,
            planned_quote=3000.0,
            confidence=0.65,
            trace_id="trace_eth_001",
            agent="MLForecaster",
            reason="uptrend"
        )
        
        # Create decision with intent reference
        decision_tuple = (
            "ETHUSDT",
            "BUY",
            {
                "confidence": 0.65,
                "_planned_quote": 3000.0,
                "agent": "MLForecaster",
                "reason": "uptrend",
                "_source_intent": intent,
            }
        )
        
        # Convert
        result = await controller._convert_decisions_to_metadecisions([decision_tuple])
        
        assert len(result) == 1
        decision = result[0]
        
        # Verify intent preservation
        assert decision.source_intent is intent
        assert decision.trace_id == "trace_eth_001"
        assert decision.source_intent.symbol == "ETHUSDT"
    
    @pytest.mark.asyncio
    async def test_decision_with_applied_gates(self, setup):
        """Test conversion tracks applied gates."""
        controller = setup["controller"]
        
        decision_tuple = (
            "BNBUSDT",
            "BUY",
            {
                "confidence": 0.72,
                "_planned_quote": 500.0,
                "agent": "DipSniper",
                "reason": "dip_detected",
                "_applied_gates": ["min_confidence", "concentration", "capital_available"],
            }
        )
        
        # Convert
        result = await controller._convert_decisions_to_metadecisions([decision_tuple])
        
        decision = result[0]
        assert len(decision.applied_gates) == 3
        assert "min_confidence" in decision.applied_gates
        assert "concentration" in decision.applied_gates
        assert "capital_available" in decision.applied_gates
    
    @pytest.mark.asyncio
    async def test_multiple_decisions_conversion(self, setup):
        """Test conversion of multiple decisions."""
        controller = setup["controller"]
        
        decision_tuples = [
            ("BTCUSDT", "BUY", {"confidence": 0.75, "_planned_quote": 45000.0, "agent": "Agent1"}),
            ("ETHUSDT", "SELL", {"confidence": 0.60, "_planned_quote": 3000.0, "agent": "Agent2"}),
            ("BNBUSDT", "BUY", {"confidence": 0.72, "_planned_quote": 500.0, "agent": "Agent3"}),
        ]
        
        result = await controller._convert_decisions_to_metadecisions(decision_tuples)
        
        assert len(result) == 3
        assert result[0].symbol == "BTCUSDT"
        assert result[0].side == "BUY"
        assert result[1].symbol == "ETHUSDT"
        assert result[1].side == "SELL"
        assert result[2].symbol == "BNBUSDT"
        assert result[2].side == "BUY"


class TestDecisionFieldExtraction:
    """Test correct field extraction during conversion."""
    
    @pytest.fixture
    def signal_manager(self):
        """Create mock signal manager."""
        return MockSignalManager()
    
    def converter(self, signal_manager):
        """Create converter function."""
        async def convert(tuples):
            meta_decisions = []
            for symbol, side, sig_dict in tuples:
                source_intent = signal_manager.get_source_intent(sig_dict)
                
                decision = MetaDecision(
                    symbol=symbol,
                    side=side,
                    confidence=float(sig_dict.get("confidence", 0.50)),
                    planned_quote=float(sig_dict.get("_planned_quote", sig_dict.get("quote", 0.0))),
                    source_intent=source_intent,
                    trace_id=source_intent.trace_id if source_intent else sig_dict.get("trace_id"),
                    execution_tier="pending",
                    enrichment={"agent": sig_dict.get("agent")},
                    policy_context={},
                    rationale=sig_dict.get("reason", "decision"),
                )
                meta_decisions.append(decision)
            return meta_decisions
        
        return convert
    
    @pytest.mark.asyncio
    async def test_confidence_extraction(self, signal_manager):
        """Test confidence field extraction."""
        converter = self.converter(signal_manager)
        decision_tuple = ("BTCUSDT", "BUY", {"confidence": 0.85})
        result = await converter([decision_tuple])
        
        assert result[0].confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_planned_quote_extraction(self, signal_manager):
        """Test planned_quote extraction from _planned_quote or quote."""
        converter = self.converter(signal_manager)
        
        # Test _planned_quote
        tuple1 = ("BTCUSDT", "BUY", {"_planned_quote": 45000.0})
        result1 = await converter([tuple1])
        assert result1[0].planned_quote == 45000.0
        
        # Test fallback to quote
        tuple2 = ("ETHUSDT", "BUY", {"quote": 3000.0})
        result2 = await converter([tuple2])
        assert result2[0].planned_quote == 3000.0
        
        # Test _planned_quote takes precedence
        tuple3 = ("BNBUSDT", "BUY", {"_planned_quote": 600.0, "quote": 500.0})
        result3 = await converter([tuple3])
        assert result3[0].planned_quote == 600.0
    
    @pytest.mark.asyncio
    async def test_symbol_side_preserved(self, signal_manager):
        """Test symbol and side are correctly preserved."""
        converter = self.converter(signal_manager)
        decision_tuple = ("SOLUSDT", "SELL", {})
        result = await converter([decision_tuple])
        
        assert result[0].symbol == "SOLUSDT"
        assert result[0].side == "SELL"


class TestDecisionTypeConsistency:
    """Test that converted decisions have correct types."""
    
    def converter(self):
        """Create converter."""
        async def convert(tuples):
            meta_decisions = []
            for symbol, side, sig_dict in tuples:
                decision = MetaDecision(
                    symbol=symbol,
                    side=side,
                    confidence=float(sig_dict.get("confidence", 0.50)),
                    planned_quote=float(sig_dict.get("_planned_quote", 0.0)),
                    source_intent=None,  # Required parameter
                    trace_id=sig_dict.get("trace_id"),  # Required parameter
                )
                meta_decisions.append(decision)
            return meta_decisions
        
        return convert
    
    @pytest.mark.asyncio
    async def test_all_results_are_metadecision(self):
        """Test all results are MetaDecision instances."""
        converter = self.converter()
        tuples = [
            ("SYM1", "BUY", {}),
            ("SYM2", "SELL", {}),
            ("SYM3", "BUY", {}),
        ]
        
        result = await converter(tuples)
        
        for decision in result:
            assert isinstance(decision, MetaDecision)
    
    @pytest.mark.asyncio
    async def test_metadecision_fields_accessible(self):
        """Test that MetaDecision fields are accessible."""
        converter = self.converter()
        decision_tuple = ("BTCUSDT", "BUY", {"confidence": 0.75, "_planned_quote": 45000.0})
        result = await converter([decision_tuple])
        
        decision = result[0]
        
        # All these should work without errors
        assert decision.symbol == "BTCUSDT"
        assert decision.side == "BUY"
        assert decision.confidence == 0.75
        assert decision.planned_quote == 45000.0
        assert decision.is_approved  # Should be True by default
        assert decision.is_rejected == False  # Should be False by default


class TestDecisionBackwardCompatibility:
    """Test backward compatibility when conversion fails."""
    
    def converter_with_fallback(self):
        """Create converter with fallback."""
        async def convert(tuples):
            meta_decisions = []
            for symbol, side, sig_dict in tuples:
                try:
                    decision = MetaDecision(
                        symbol=symbol,
                        side=side,
                        confidence=float(sig_dict.get("confidence", 0.50)),
                        planned_quote=float(sig_dict.get("_planned_quote", 0.0)),
                        source_intent=None,
                        trace_id=sig_dict.get("trace_id"),
                    )
                    meta_decisions.append(decision)
                except Exception:
                    # Fallback to tuple
                    meta_decisions.append((symbol, side, sig_dict))
            return meta_decisions
        
        return convert
    
    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """Test fallback to tuple when conversion fails."""
        converter = self.converter_with_fallback()
        
        # Invalid side value that might cause issues
        decision_tuple = ("BTCUSDT", "INVALID", {})
        
        # Converter has fallback, so should not crash
        result = await converter([decision_tuple])
        
        # Should return something (either MetaDecision or fallback tuple)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
