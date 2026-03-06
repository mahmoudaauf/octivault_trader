"""
Test suite for ExitArbitrator: Priority-based exit resolution.

Tests verify:
1. Risk exits always beat TP/SL and signal exits
2. TP/SL exits beat signal exits (when no risk)
3. Signal exits execute when no higher priority exists
4. Logging captures arbitration decisions
5. Priority modification works at runtime
6. Edge cases (no exits, single exit, etc.)
"""

import pytest
import logging
from typing import Dict, Any, List
from core.exit_arbitrator import ExitArbitrator, ExitPriority, get_arbitrator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger("test_exit_arbitrator")


@pytest.fixture
def arbitrator(logger):
    """Create an ExitArbitrator instance for testing."""
    return ExitArbitrator(logger=logger)


@pytest.fixture
def sample_position():
    """Sample position data for testing."""
    return {
        "symbol": "BTC/USDT",
        "amount": 0.5,
        "entry_price": 44000,
        "current_price": 45000,
        "quote": 22500,
    }


@pytest.fixture
def risk_exit_starvation():
    """Sample risk exit (starvation)."""
    return {
        "action": "SELL",
        "reason": "Capital starvation - insufficient quote to maintain position",
        "tag": "risk/starvation",
        "quantity": 0.5,
    }


@pytest.fixture
def risk_exit_dust():
    """Sample risk exit (dust position)."""
    return {
        "action": "SELL",
        "reason": "Dust position - below 0.60 threshold",
        "tag": "risk/dust",
        "quantity": 0.01,
    }


@pytest.fixture
def tp_exit():
    """Sample TP exit."""
    return {
        "action": "SELL",
        "reason": "Take-profit target reached at $45,000",
        "tag": "tp_sl",
        "quantity": 0.5,
        "price": 45000,
    }


@pytest.fixture
def sl_exit():
    """Sample SL exit."""
    return {
        "action": "SELL",
        "reason": "Stop-loss triggered at $42,000",
        "tag": "tp_sl",
        "quantity": 0.5,
        "price": 42000,
    }


@pytest.fixture
def signal_exit_agent():
    """Sample agent signal exit."""
    return {
        "action": "SELL",
        "reason": "Agent recommends sell - downtrend detected",
        "tag": "agent_signal",
        "confidence": 0.85,
        "quantity": 0.5,
    }


@pytest.fixture
def signal_exit_rotation():
    """Sample rotation exit."""
    return {
        "action": "SELL",
        "reason": "Symbol exiting rotation universe",
        "tag": "rotation_exit",
        "quantity": 0.5,
    }


@pytest.fixture
def signal_exit_rebalance():
    """Sample rebalance exit."""
    return {
        "action": "SELL",
        "reason": "Portfolio weight adjustment",
        "tag": "rebalance_exit",
        "quantity": 0.3,
    }


# ============================================================================
# BASIC ARBITRATION TESTS
# ============================================================================

class TestBasicArbitration:
    """Test basic arbitration logic and priority ordering."""
    
    @pytest.mark.asyncio
    async def test_no_exits_returns_none(self, arbitrator, sample_position):
        """When no exits available, should return None."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
        )
        assert exit_type is None
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_single_risk_exit(self, arbitrator, sample_position, risk_exit_starvation):
        """Single risk exit should execute."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            risk_exit=risk_exit_starvation,
        )
        assert exit_type == "RISK"
        assert signal == risk_exit_starvation
    
    @pytest.mark.asyncio
    async def test_single_tp_sl_exit(self, arbitrator, sample_position, tp_exit):
        """Single TP/SL exit should execute."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            tp_sl_exit=tp_exit,
        )
        assert exit_type == "TP_SL"
        assert signal == tp_exit
    
    @pytest.mark.asyncio
    async def test_single_signal_exit(self, arbitrator, sample_position, signal_exit_agent):
        """Single signal exit should execute."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent],
        )
        assert exit_type == "SIGNAL"
        assert signal == signal_exit_agent


# ============================================================================
# PRIORITY ORDERING TESTS
# ============================================================================

class TestPriorityOrdering:
    """Test that priority ordering is enforced correctly."""
    
    @pytest.mark.asyncio
    async def test_risk_beats_tp_sl(self, arbitrator, sample_position, risk_exit_starvation, tp_exit):
        """Risk exit should beat TP/SL exit."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            risk_exit=risk_exit_starvation,
            tp_sl_exit=tp_exit,
        )
        assert exit_type == "RISK"
        assert signal == risk_exit_starvation
    
    @pytest.mark.asyncio
    async def test_risk_beats_signal(self, arbitrator, sample_position, risk_exit_dust, signal_exit_agent):
        """Risk exit should beat signal exit."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            risk_exit=risk_exit_dust,
            signal_exits=[signal_exit_agent],
        )
        assert exit_type == "RISK"
        assert signal == risk_exit_dust
    
    @pytest.mark.asyncio
    async def test_tp_sl_beats_signal(self, arbitrator, sample_position, tp_exit, signal_exit_agent):
        """TP/SL exit should beat signal exit (when no risk)."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            tp_sl_exit=tp_exit,
            signal_exits=[signal_exit_agent],
        )
        assert exit_type == "TP_SL"
        assert signal == tp_exit
    
    @pytest.mark.asyncio
    async def test_risk_beats_tp_sl_and_signal(
        self, arbitrator, sample_position, risk_exit_starvation, tp_exit, signal_exit_agent
    ):
        """Risk should beat both TP/SL and signal."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            risk_exit=risk_exit_starvation,
            tp_sl_exit=tp_exit,
            signal_exits=[signal_exit_agent],
        )
        assert exit_type == "RISK"
        assert signal == risk_exit_starvation
    
    @pytest.mark.asyncio
    async def test_complete_hierarchy(
        self,
        arbitrator,
        sample_position,
        risk_exit_starvation,
        tp_exit,
        signal_exit_agent,
        signal_exit_rotation,
    ):
        """Complete hierarchy: RISK > TP_SL > SIGNAL > ROTATION."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            risk_exit=risk_exit_starvation,
            tp_sl_exit=tp_exit,
            signal_exits=[signal_exit_agent, signal_exit_rotation],
        )
        assert exit_type == "RISK"
        assert signal == risk_exit_starvation


# ============================================================================
# SIGNAL CATEGORIZATION TESTS
# ============================================================================

class TestSignalCategorization:
    """Test that signals are correctly categorized by type."""
    
    @pytest.mark.asyncio
    async def test_rotation_exit_categorized_correctly(
        self, arbitrator, sample_position, signal_exit_rotation
    ):
        """Rotation exit should be categorized as ROTATION."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_rotation],
        )
        assert exit_type == "ROTATION"
        assert signal == signal_exit_rotation
    
    @pytest.mark.asyncio
    async def test_rebalance_exit_categorized_correctly(
        self, arbitrator, sample_position, signal_exit_rebalance
    ):
        """Rebalance exit should be categorized as REBALANCE."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_rebalance],
        )
        assert exit_type == "REBALANCE"
        assert signal == signal_exit_rebalance
    
    @pytest.mark.asyncio
    async def test_agent_signal_categorized_as_signal(
        self, arbitrator, sample_position, signal_exit_agent
    ):
        """Agent signal should be categorized as SIGNAL."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent],
        )
        assert exit_type == "SIGNAL"
        assert signal == signal_exit_agent
    
    @pytest.mark.asyncio
    async def test_rotation_beats_rebalance(
        self, arbitrator, sample_position, signal_exit_rotation, signal_exit_rebalance
    ):
        """ROTATION (priority 4) should beat REBALANCE (priority 5)."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_rotation, signal_exit_rebalance],
        )
        assert exit_type == "ROTATION"
        assert signal == signal_exit_rotation
    
    @pytest.mark.asyncio
    async def test_signal_beats_rotation(
        self, arbitrator, sample_position, signal_exit_agent, signal_exit_rotation
    ):
        """SIGNAL (priority 3) should beat ROTATION (priority 4)."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent, signal_exit_rotation],
        )
        assert exit_type == "SIGNAL"
        assert signal == signal_exit_agent


# ============================================================================
# PRIORITY MODIFICATION TESTS
# ============================================================================

class TestPriorityModification:
    """Test runtime priority adjustment."""
    
    @pytest.mark.asyncio
    async def test_set_priority_invalid_type(self, arbitrator):
        """Setting priority for invalid type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown exit type"):
            arbitrator.set_priority("INVALID_TYPE", 1)
    
    @pytest.mark.asyncio
    async def test_set_priority_valid_type(self, arbitrator):
        """Setting priority for valid type should succeed."""
        arbitrator.set_priority("ROTATION", 1.5)
        assert arbitrator.priority_map["ROTATION"] == 1.5
    
    @pytest.mark.asyncio
    async def test_modified_priority_affects_resolution(
        self, arbitrator, sample_position, signal_exit_agent, signal_exit_rotation
    ):
        """Modified priority should affect exit resolution."""
        # Default: SIGNAL (3) beats ROTATION (4)
        exit_type, _ = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent, signal_exit_rotation],
        )
        assert exit_type == "SIGNAL"
        
        # After modification: ROTATION (1.5) should beat SIGNAL (3)
        arbitrator.set_priority("ROTATION", 1.5)
        exit_type, _ = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent, signal_exit_rotation],
        )
        assert exit_type == "ROTATION"
    
    def test_get_priority_order(self, arbitrator):
        """get_priority_order should return sorted priorities."""
        order = arbitrator.get_priority_order()
        
        # Should be sorted by priority value
        priorities = [p for _, p in order]
        assert priorities == sorted(priorities)
        
        # Default order should be RISK, TP_SL, SIGNAL, ROTATION, REBALANCE
        assert order[0][0] == "RISK"
        assert order[1][0] == "TP_SL"
        assert order[2][0] == "SIGNAL"
    
    def test_reset_priorities(self, arbitrator):
        """reset_priorities should restore default values."""
        # Modify priorities
        arbitrator.set_priority("ROTATION", 1.5)
        assert arbitrator.priority_map["ROTATION"] == 1.5
        
        # Reset
        arbitrator.reset_priorities()
        assert arbitrator.priority_map["ROTATION"] == ExitPriority.ROTATION


# ============================================================================
# MULTIPLE EXITS PER TIER TESTS
# ============================================================================

class TestMultipleExitsPerTier:
    """Test handling multiple exits within same tier."""
    
    @pytest.mark.asyncio
    async def test_multiple_signal_exits_first_wins(
        self, arbitrator, sample_position, signal_exit_agent, signal_exit_rotation
    ):
        """When multiple signal exits available, first in list wins (stable sort)."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_exit_agent, signal_exit_rotation],
        )
        # AGENT (SIGNAL) has priority 3, ROTATION has priority 4
        assert exit_type == "SIGNAL"
        assert signal == signal_exit_agent
    
    @pytest.mark.asyncio
    async def test_multiple_generic_signal_exits(self, arbitrator, sample_position):
        """Multiple generic signal exits should return first one."""
        signal1 = {
            "action": "SELL",
            "reason": "Signal 1",
            "tag": "signal_1",
            "confidence": 0.9,
        }
        signal2 = {
            "action": "SELL",
            "reason": "Signal 2",
            "tag": "signal_2",
            "confidence": 0.8,
        }
        
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal1, signal2],
        )
        assert exit_type == "SIGNAL"
        assert signal == signal1  # First one in list


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_signal_list(self, arbitrator, sample_position, tp_exit):
        """Empty signal list should not affect arbitration."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            tp_sl_exit=tp_exit,
            signal_exits=[],
        )
        assert exit_type == "TP_SL"
        assert signal == tp_exit
    
    @pytest.mark.asyncio
    async def test_none_signal_list(self, arbitrator, sample_position, tp_exit):
        """None signal list (default) should not affect arbitration."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            tp_sl_exit=tp_exit,
        )
        assert exit_type == "TP_SL"
        assert signal == tp_exit
    
    @pytest.mark.asyncio
    async def test_signal_without_tag(self, arbitrator, sample_position):
        """Signal without tag should default to SIGNAL category."""
        signal_no_tag = {
            "action": "SELL",
            "reason": "Some reason",
            # no tag field
        }
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=sample_position,
            signal_exits=[signal_no_tag],
        )
        assert exit_type == "SIGNAL"
        assert signal == signal_no_tag
    
    @pytest.mark.asyncio
    async def test_symbol_with_special_characters(self, arbitrator, sample_position):
        """Symbols with special characters should work."""
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="SOL/USDT:PERP",  # Futures symbol
            position=sample_position,
        )
        assert exit_type is None
        assert signal is None


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestLogging:
    """Test logging of arbitration decisions."""
    
    @pytest.mark.asyncio
    async def test_logging_multiple_candidates(self, caplog, sample_position, tp_exit, signal_exit_agent):
        """When multiple candidates, should log winner and suppressed."""
        arbitrator = ExitArbitrator(logger=logging.getLogger("test"))
        
        with caplog.at_level(logging.INFO):
            exit_type, signal = await arbitrator.resolve_exit(
                symbol="BTC/USDT",
                position=sample_position,
                tp_sl_exit=tp_exit,
                signal_exits=[signal_exit_agent],
            )
        
        # Should log the arbitration
        assert any("ExitArbitration" in record.message for record in caplog.records)
        assert any("BTC/USDT" in record.message for record in caplog.records)
        assert any("TP_SL" in record.message for record in caplog.records)
    
    @pytest.mark.asyncio
    async def test_logging_single_candidate(self, caplog, sample_position, tp_exit):
        """Single candidate should log at DEBUG level."""
        arbitrator = ExitArbitrator(logger=logging.getLogger("test"))
        
        with caplog.at_level(logging.DEBUG):
            exit_type, signal = await arbitrator.resolve_exit(
                symbol="BTC/USDT",
                position=sample_position,
                tp_sl_exit=tp_exit,
            )
        
        # Should still log the decision
        assert any("ExitArbitration" in record.message for record in caplog.records)
    
    @pytest.mark.asyncio
    async def test_priority_modification_logged(self, caplog):
        """Priority modification should be logged."""
        arbitrator = ExitArbitrator(logger=logging.getLogger("test"))
        
        with caplog.at_level(logging.INFO):
            arbitrator.set_priority("ROTATION", 2.0)
        
        assert any("Priority updated" in record.message for record in caplog.records)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests simulating real trading scenarios."""
    
    @pytest.mark.asyncio
    async def test_scenario_capital_emergency(self, arbitrator):
        """Scenario: Capital at critical level (starvation) with multiple signals."""
        position = {"symbol": "BTC/USDT", "amount": 0.1, "quote": 2.0}
        
        risk_exit = {
            "action": "SELL",
            "reason": "Capital starvation - insufficient quote",
            "tag": "risk/starvation",
        }
        tp_exit = {
            "action": "SELL",
            "reason": "Take-profit reached",
            "tag": "tp_sl",
        }
        agent_signal = {
            "action": "SELL",
            "reason": "Agent recommends hold",
            "tag": "agent_signal",
        }
        
        # Even though agent says hold, RISK should win
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="BTC/USDT",
            position=position,
            risk_exit=risk_exit,
            tp_sl_exit=tp_exit,
            signal_exits=[agent_signal],
        )
        
        assert exit_type == "RISK"
        assert signal == risk_exit
    
    @pytest.mark.asyncio
    async def test_scenario_normal_trading(self, arbitrator):
        """Scenario: Normal trading day - no risk, normal signals."""
        position = {"symbol": "ETH/USDT", "amount": 1.0, "quote": 2500}
        
        signal_exit = {
            "action": "SELL",
            "reason": "Agent detects downtrend",
            "tag": "agent_signal",
        }
        
        # With no risk or TP/SL, agent signal should execute
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="ETH/USDT",
            position=position,
            signal_exits=[signal_exit],
        )
        
        assert exit_type == "SIGNAL"
        assert signal == signal_exit
    
    @pytest.mark.asyncio
    async def test_scenario_take_profit_with_agent_conflict(self, arbitrator):
        """Scenario: TP triggered but agent recommends different action."""
        position = {"symbol": "SOL/USDT", "amount": 10.0, "quote": 1500}
        
        tp_exit = {
            "action": "SELL",
            "reason": "Take-profit target reached",
            "tag": "tp_sl",
            "price": 150,
        }
        agent_signal = {
            "action": "HOLD",
            "reason": "Agent expects further upside",
            "tag": "agent_signal",
        }
        
        # TP should win over agent's hold signal
        exit_type, signal = await arbitrator.resolve_exit(
            symbol="SOL/USDT",
            position=position,
            tp_sl_exit=tp_exit,
            signal_exits=[agent_signal],
        )
        
        assert exit_type == "TP_SL"
        assert signal == tp_exit


# ============================================================================
# MODULE-LEVEL SINGLETON TESTS
# ============================================================================

class TestModuleSingleton:
    """Test module-level singleton pattern."""
    
    def test_get_arbitrator_creates_instance(self):
        """get_arbitrator should create default instance."""
        # Note: This test may be affected by module state
        # In practice, use fresh instances for testing
        arbitrator = get_arbitrator()
        assert arbitrator is not None
        assert isinstance(arbitrator, ExitArbitrator)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
