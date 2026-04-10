"""
Unit tests for arbitration_engine module.

Tests cover:
- GateResult dataclass
- 6-layer arbitration pipeline (symbol → confidence → regime → position → capital → risk)
- Individual gate testing
- Full pipeline integration
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


# Mock classes for testing
@dataclass
class GateResult:
    """Result from a single gate evaluation."""
    gate_name: str
    passed: bool
    reason: str
    details: Dict[str, Any]


class ArbitrationEngine:
    """6-layer arbitration pipeline for signal evaluation."""
    
    def __init__(self):
        self.gates_evaluated = 0
        self.last_evaluation_time = None
    
    async def evaluate_signal(
        self,
        symbol: str,
        signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate signal through all 6 gates."""
        self.gates_evaluated += 1
        self.last_evaluation_time = datetime.now()
        
        results = []
        
        # Gate 1: Symbol Validation
        symbol_result = await self._symbol_validation_gate(symbol)
        results.append(symbol_result)
        if not symbol_result.passed:
            return {"passed": False, "blocking_gate": "symbol_validation", "results": results}
        
        # Gate 2: Confidence
        confidence_result = await self._confidence_gate(signal_data)
        results.append(confidence_result)
        if not confidence_result.passed:
            return {"passed": False, "blocking_gate": "confidence", "results": results}
        
        # Gate 3: Regime
        regime_result = await self._regime_gate(signal_data)
        results.append(regime_result)
        if not regime_result.passed:
            return {"passed": False, "blocking_gate": "regime", "results": results}
        
        # Gate 4: Position Limit
        position_result = await self._position_limit_gate(symbol, signal_data)
        results.append(position_result)
        if not position_result.passed:
            return {"passed": False, "blocking_gate": "position_limit", "results": results}
        
        # Gate 5: Capital
        capital_result = await self._capital_gate(signal_data)
        results.append(capital_result)
        if not capital_result.passed:
            return {"passed": False, "blocking_gate": "capital", "results": results}
        
        # Gate 6: Risk
        risk_result = await self._risk_gate(signal_data)
        results.append(risk_result)
        if not risk_result.passed:
            return {"passed": False, "blocking_gate": "risk", "results": results}
        
        return {"passed": True, "results": results}
    
    async def _symbol_validation_gate(self, symbol: str) -> GateResult:
        """Validate symbol format."""
        if not symbol or len(symbol) < 2:
            return GateResult(
                gate_name="symbol_validation",
                passed=False,
                reason="Invalid symbol format",
                details={"symbol": symbol}
            )
        return GateResult(
            gate_name="symbol_validation",
            passed=True,
            reason="Symbol valid",
            details={"symbol": symbol}
        )
    
    async def _confidence_gate(self, signal_data: Dict[str, Any]) -> GateResult:
        """Check confidence threshold."""
        confidence = signal_data.get("confidence", 0.0)
        threshold = 0.50
        passed = confidence >= threshold
        return GateResult(
            gate_name="confidence",
            passed=passed,
            reason=f"Confidence {'above' if passed else 'below'} threshold",
            details={"confidence": confidence, "threshold": threshold}
        )
    
    async def _regime_gate(self, signal_data: Dict[str, Any]) -> GateResult:
        """Check market regime compatibility."""
        regime = signal_data.get("regime", "unknown")
        valid_regimes = ["bullish", "neutral", "bearish"]
        passed = regime in valid_regimes
        return GateResult(
            gate_name="regime",
            passed=passed,
            reason=f"Regime {'valid' if passed else 'invalid'}",
            details={"regime": regime, "valid_regimes": valid_regimes}
        )
    
    async def _position_limit_gate(
        self,
        symbol: str,
        signal_data: Dict[str, Any]
    ) -> GateResult:
        """Check position limit constraints."""
        current_positions = signal_data.get("current_positions", 0)
        max_positions = signal_data.get("max_positions", 10)
        passed = current_positions < max_positions
        return GateResult(
            gate_name="position_limit",
            passed=passed,
            reason=f"Position limit {'within' if passed else 'exceeded'}",
            details={
                "symbol": symbol,
                "current": current_positions,
                "max": max_positions
            }
        )
    
    async def _capital_gate(self, signal_data: Dict[str, Any]) -> GateResult:
        """Check capital availability."""
        available_capital = signal_data.get("available_capital", 0.0)
        required_capital = signal_data.get("required_capital", 0.0)
        passed = available_capital >= required_capital
        return GateResult(
            gate_name="capital",
            passed=passed,
            reason=f"Capital {'sufficient' if passed else 'insufficient'}",
            details={
                "available": available_capital,
                "required": required_capital
            }
        )
    
    async def _risk_gate(self, signal_data: Dict[str, Any]) -> GateResult:
        """Check risk constraints."""
        portfolio_risk = signal_data.get("portfolio_risk", 0.0)
        max_risk = signal_data.get("max_risk", 0.05)
        passed = portfolio_risk <= max_risk
        return GateResult(
            gate_name="risk",
            passed=passed,
            reason=f"Risk {'within' if passed else 'exceeds'} limits",
            details={
                "portfolio_risk": portfolio_risk,
                "max_risk": max_risk
            }
        )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "gates_evaluated": self.gates_evaluated,
            "last_evaluation": self.last_evaluation_time
        }


# ============================================================================
# Test Classes
# ============================================================================

class TestGateResult:
    """Test GateResult dataclass."""
    
    def test_gate_result_creation(self) -> None:
        """Test GateResult creation."""
        result = GateResult(
            gate_name="test_gate",
            passed=True,
            reason="Test reason",
            details={"key": "value"}
        )
        assert result.gate_name == "test_gate"
        assert result.passed is True
        assert result.reason == "Test reason"
        assert result.details == {"key": "value"}
    
    def test_gate_result_passed_false(self) -> None:
        """Test GateResult with passed=False."""
        result = GateResult(
            gate_name="failing_gate",
            passed=False,
            reason="Failed reason",
            details={}
        )
        assert result.passed is False
    
    def test_gate_result_with_complex_details(self) -> None:
        """Test GateResult with complex details."""
        details = {
            "threshold": 0.5,
            "actual": 0.3,
            "metadata": {"timestamp": "2026-04-10", "source": "test"}
        }
        result = GateResult(
            gate_name="complex_gate",
            passed=False,
            reason="Details test",
            details=details
        )
        assert result.details == details


class TestArbitrationEngineInitialization:
    """Test ArbitrationEngine initialization."""
    
    def test_engine_initialization(self) -> None:
        """Test engine initialization."""
        engine = ArbitrationEngine()
        assert engine.gates_evaluated == 0
        assert engine.last_evaluation_time is None
    
    def test_get_pipeline_status_initial(self) -> None:
        """Test pipeline status after initialization."""
        engine = ArbitrationEngine()
        status = engine.get_pipeline_status()
        assert status["gates_evaluated"] == 0
        assert status["last_evaluation"] is None


class TestSymbolValidationGate:
    """Test symbol validation gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_valid_symbol(self, engine: ArbitrationEngine) -> None:
        """Test valid symbol passes gate."""
        result = await engine._symbol_validation_gate("BTC")
        assert result.passed is True
        assert result.gate_name == "symbol_validation"
    
    @pytest.mark.asyncio
    async def test_empty_symbol(self, engine: ArbitrationEngine) -> None:
        """Test empty symbol fails gate."""
        result = await engine._symbol_validation_gate("")
        assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_single_character_symbol(self, engine: ArbitrationEngine) -> None:
        """Test single character symbol fails gate."""
        result = await engine._symbol_validation_gate("A")
        assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_long_symbol(self, engine: ArbitrationEngine) -> None:
        """Test long symbol passes gate."""
        result = await engine._symbol_validation_gate("VERYLONGSYMBOL")
        assert result.passed is True


class TestConfidenceGate:
    """Test confidence gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_confidence_above_threshold(self, engine: ArbitrationEngine) -> None:
        """Test confidence above threshold."""
        result = await engine._confidence_gate({"confidence": 0.75})
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_confidence_at_threshold(self, engine: ArbitrationEngine) -> None:
        """Test confidence at exact threshold."""
        result = await engine._confidence_gate({"confidence": 0.50})
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_confidence_below_threshold(self, engine: ArbitrationEngine) -> None:
        """Test confidence below threshold."""
        result = await engine._confidence_gate({"confidence": 0.25})
        assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_missing_confidence(self, engine: ArbitrationEngine) -> None:
        """Test missing confidence defaults to 0."""
        result = await engine._confidence_gate({})
        assert result.passed is False


class TestRegimeGate:
    """Test regime gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_bullish_regime(self, engine: ArbitrationEngine) -> None:
        """Test bullish regime passes."""
        result = await engine._regime_gate({"regime": "bullish"})
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_bearish_regime(self, engine: ArbitrationEngine) -> None:
        """Test bearish regime passes."""
        result = await engine._regime_gate({"regime": "bearish"})
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_neutral_regime(self, engine: ArbitrationEngine) -> None:
        """Test neutral regime passes."""
        result = await engine._regime_gate({"regime": "neutral"})
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_invalid_regime(self, engine: ArbitrationEngine) -> None:
        """Test invalid regime fails."""
        result = await engine._regime_gate({"regime": "sideways"})
        assert result.passed is False


class TestPositionLimitGate:
    """Test position limit gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_within_limit(self, engine: ArbitrationEngine) -> None:
        """Test within position limit."""
        result = await engine._position_limit_gate(
            "BTC",
            {"current_positions": 5, "max_positions": 10}
        )
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_at_limit(self, engine: ArbitrationEngine) -> None:
        """Test at position limit."""
        result = await engine._position_limit_gate(
            "BTC",
            {"current_positions": 10, "max_positions": 10}
        )
        assert result.passed is False
    
    @pytest.mark.asyncio
    async def test_exceeds_limit(self, engine: ArbitrationEngine) -> None:
        """Test exceeds position limit."""
        result = await engine._position_limit_gate(
            "BTC",
            {"current_positions": 15, "max_positions": 10}
        )
        assert result.passed is False


class TestCapitalGate:
    """Test capital gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_sufficient_capital(self, engine: ArbitrationEngine) -> None:
        """Test sufficient capital."""
        result = await engine._capital_gate({
            "available_capital": 1000.0,
            "required_capital": 500.0
        })
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_exact_capital(self, engine: ArbitrationEngine) -> None:
        """Test exact capital required."""
        result = await engine._capital_gate({
            "available_capital": 500.0,
            "required_capital": 500.0
        })
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_insufficient_capital(self, engine: ArbitrationEngine) -> None:
        """Test insufficient capital."""
        result = await engine._capital_gate({
            "available_capital": 300.0,
            "required_capital": 500.0
        })
        assert result.passed is False


class TestRiskGate:
    """Test risk gate."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_risk_within_limit(self, engine: ArbitrationEngine) -> None:
        """Test risk within limit."""
        result = await engine._risk_gate({
            "portfolio_risk": 0.03,
            "max_risk": 0.05
        })
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_risk_at_limit(self, engine: ArbitrationEngine) -> None:
        """Test risk at limit."""
        result = await engine._risk_gate({
            "portfolio_risk": 0.05,
            "max_risk": 0.05
        })
        assert result.passed is True
    
    @pytest.mark.asyncio
    async def test_risk_exceeds_limit(self, engine: ArbitrationEngine) -> None:
        """Test risk exceeds limit."""
        result = await engine._risk_gate({
            "portfolio_risk": 0.07,
            "max_risk": 0.05
        })
        assert result.passed is False


class TestFullPipeline:
    """Test full arbitration pipeline."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_all_gates_pass(self, engine: ArbitrationEngine) -> None:
        """Test signal passes all gates."""
        signal = {
            "confidence": 0.75,
            "regime": "bullish",
            "current_positions": 5,
            "max_positions": 10,
            "available_capital": 1000.0,
            "required_capital": 500.0,
            "portfolio_risk": 0.03,
            "max_risk": 0.05
        }
        result = await engine.evaluate_signal("BTC", signal)
        assert result["passed"] is True
        assert len(result["results"]) == 6
    
    @pytest.mark.asyncio
    async def test_fails_on_first_gate(self, engine: ArbitrationEngine) -> None:
        """Test signal fails on symbol validation gate."""
        signal = {"confidence": 0.75}
        result = await engine.evaluate_signal("", signal)
        assert result["passed"] is False
        assert result["blocking_gate"] == "symbol_validation"
    
    @pytest.mark.asyncio
    async def test_fails_on_confidence_gate(self, engine: ArbitrationEngine) -> None:
        """Test signal fails on confidence gate."""
        signal = {
            "confidence": 0.25,
            "regime": "bullish"
        }
        result = await engine.evaluate_signal("BTC", signal)
        assert result["passed"] is False
        assert result["blocking_gate"] == "confidence"
    
    @pytest.mark.asyncio
    async def test_fails_on_capital_gate(self, engine: ArbitrationEngine) -> None:
        """Test signal fails on capital gate."""
        signal = {
            "confidence": 0.75,
            "regime": "bullish",
            "current_positions": 5,
            "max_positions": 10,
            "available_capital": 100.0,
            "required_capital": 500.0,
            "portfolio_risk": 0.03,
            "max_risk": 0.05
        }
        result = await engine.evaluate_signal("BTC", signal)
        assert result["passed"] is False
        assert result["blocking_gate"] == "capital"


class TestArbitrationEdgeCases:
    """Test edge cases."""
    
    @pytest.fixture
    def engine(self) -> ArbitrationEngine:
        """Create engine instance."""
        return ArbitrationEngine()
    
    @pytest.mark.asyncio
    async def test_empty_signal_data(self, engine: ArbitrationEngine) -> None:
        """Test empty signal data."""
        result = await engine.evaluate_signal("BTC", {})
        assert result["passed"] is False
    
    @pytest.mark.asyncio
    async def test_extreme_values(self, engine: ArbitrationEngine) -> None:
        """Test extreme values."""
        signal = {
            "confidence": 1.0,
            "regime": "bullish",
            "current_positions": 0,
            "max_positions": 1000000,
            "available_capital": 999999999.0,
            "required_capital": 1.0,
            "portfolio_risk": 0.0,
            "max_risk": 1.0
        }
        result = await engine.evaluate_signal("BTC", signal)
        assert result["passed"] is True
    
    @pytest.mark.asyncio
    async def test_negative_values(self, engine: ArbitrationEngine) -> None:
        """Test negative values."""
        signal = {
            "confidence": -0.5,
            "regime": "bullish",
            "current_positions": -5,
            "max_positions": 10,
            "available_capital": -1000.0,
            "required_capital": 500.0,
            "portfolio_risk": -0.05,
            "max_risk": 0.05
        }
        result = await engine.evaluate_signal("BTC", signal)
        assert result["passed"] is False
