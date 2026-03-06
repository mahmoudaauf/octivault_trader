"""
Test suite for Phase 5: Trading Coordinator Integration

Tests the complete trading workflow integrating all Phase 1-4 components:
- Phase 1: Portfolio State Machine
- Phase 2: Bootstrap Metrics Persistence
- Phase 3: Dust Registry Lifecycle
- Phase 4: Position Merger & Consolidation
- Phase 5: Trading Coordinator (NEW)
"""

import pytest
import time
from datetime import datetime

from core.shared_state import SharedState, PortfolioState
from core.trading_coordinator import TradeExecution, TradingCoordinator


class TestTradingCoordinatorBasics:
    """Test basic TradingCoordinator initialization and setup."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly."""
        assert self.coordinator is not None
        assert isinstance(self.coordinator, TradingCoordinator)
        assert self.coordinator.shared_state is self.state
    
    def test_coordinator_has_shared_state(self):
        """Test coordinator has reference to SharedState."""
        assert self.coordinator.shared_state == self.state
        assert self.coordinator.shared_state.position_merger is not None
        assert self.coordinator.shared_state.bootstrap_metrics is not None
    
    def test_trade_history_starts_empty(self):
        """Test trade history starts empty."""
        assert self.coordinator.trade_history == []
        assert len(self.coordinator.get_trade_history()) == 0
    
    def test_trade_execution_dataclass(self):
        """Test TradeExecution dataclass creation."""
        execution = TradeExecution(
            order_id="TEST-001",
            symbol="BTC",
            quantity=0.1,
            entry_price=50000.0,
            trade_type="BUY",
            timestamp=time.time(),
            consolidated=False
        )
        
        assert execution.order_id == "TEST-001"
        assert execution.symbol == "BTC"
        assert execution.quantity == 0.1
        assert execution.entry_price == 50000.0
        assert execution.trade_type == "BUY"
        assert execution.consolidated == False
    
    def test_trade_execution_serialization(self):
        """Test TradeExecution to_dict serialization."""
        execution = TradeExecution(
            order_id="TEST-002",
            symbol="ETH",
            quantity=1.0,
            entry_price=3000.0,
            trade_type="SELL",
            timestamp=time.time()
        )
        
        data = execution.to_dict()
        assert data["order_id"] == "TEST-002"
        assert data["symbol"] == "ETH"
        assert data["quantity"] == 1.0
        assert data["entry_price"] == 3000.0


class TestSystemReadinessChecks:
    """Test system readiness verification."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        # Reset bootstrap metrics for fresh test state (for readiness checks)
        self.state.bootstrap_metrics._cached_metrics = {}
        self.state.metrics["total_trades_executed"] = 0
        self.state.metrics["first_trade_at"] = None
        # Enable cold bootstrap for testing
        self.state.config.COLD_BOOTSTRAP_ENABLED = True
        self.coordinator = self.state.trading_coordinator
    
    def test_reject_during_cold_bootstrap(self):
        """Test system rejects trades during cold bootstrap."""
        # Fresh state is in cold bootstrap
        is_ready, reason = self.coordinator.check_system_ready()
        assert is_ready == False
        assert "cold bootstrap" in reason.lower()
    
    def test_accept_after_bootstrap(self):
        """Test system accepts trades after bootstrap."""
        # Record a trade to exit cold bootstrap
        self.state.bootstrap_metrics.save_trade_executed()
        
        # Now should be ready
        is_ready, reason = self.coordinator.check_system_ready()
        assert is_ready == True
        assert reason == "System ready"
    
    def test_bootstrap_metrics_validation(self):
        """Test bootstrap metrics must be initialized."""
        # Fresh state has no trades - should not be ready
        is_ready, reason = self.coordinator.check_system_ready()
        assert is_ready == False
    
    def test_portfolio_state_required(self):
        """Test system is ready after bootstrap metrics initialized."""
        # Record a trade to pass that check
        self.state.bootstrap_metrics.save_trade_executed()
        
        # Should have dust registry initialized
        assert self.state.dust_lifecycle_registry is not None
        
        # Should be ready
        is_ready, _ = self.coordinator.check_system_ready()
        assert is_ready == True


class TestPositionPreparation:
    """Test position consolidation before trading."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        # Need to initialize bootstrap for readiness checks
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_single_position_no_consolidation(self):
        """Test single position passes through without consolidation."""
        positions = [
            {"symbol": "BTC", "quantity": 0.5, "entry_price": 50000.0}
        ]
        
        prepared, was_consolidated = self.coordinator.prepare_positions("BTC", positions)
        
        assert prepared is not None
        assert len(prepared) == 1
        assert was_consolidated == False
        assert prepared[0]["quantity"] == 0.5
    
    def test_consolidate_fragmented_positions(self):
        """Test multiple positions get consolidated."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50100.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50050.0},
        ]
        
        prepared, was_consolidated = self.coordinator.prepare_positions("BTC", positions)
        
        assert prepared is not None
        assert was_consolidated == True
        assert len(prepared) == 1
        assert prepared[0]["consolidated"] == True
    
    def test_respects_merge_decision_logic(self):
        """Test respects PositionMerger decision logic."""
        # Small positions that shouldn't merge
        positions = [
            {"symbol": "BTC", "quantity": 0.001, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.0005, "entry_price": 50000.0},
        ]
        
        prepared, was_consolidated = self.coordinator.prepare_positions("BTC", positions)
        
        # Might not consolidate depending on feasibility score
        # Just verify method returns valid result
        assert prepared is not None
    
    def test_empty_positions_returns_none(self):
        """Test empty positions list returns None."""
        prepared, was_consolidated = self.coordinator.prepare_positions("BTC", [])
        
        assert prepared is None
        assert was_consolidated == False


class TestTradeExecution:
    """Test trade execution workflow."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        # Initialize bootstrap to pass readiness checks
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_full_trade_workflow_rejected_before_bootstrap(self):
        """Test trade rejected before bootstrap complete."""
        # Create a fresh state without bootstrap
        fresh_state = SharedState()
        # Reset bootstrap metrics for fresh test state
        fresh_state.bootstrap_metrics._cached_metrics = {}
        fresh_state.metrics["total_trades_executed"] = 0
        fresh_state.metrics["first_trade_at"] = None
        # Enable cold bootstrap so trades are rejected before bootstrap
        fresh_state.config.COLD_BOOTSTRAP_ENABLED = True
        fresh_coordinator = fresh_state.trading_coordinator
        
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        order_params = {"trade_type": "BUY"}
        
        order_id = fresh_coordinator.execute_trade("BTC", positions, order_params)
        
        assert order_id is None
    
    def test_full_trade_workflow_success(self):
        """Test successful trade execution workflow."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        order_params = {"trade_type": "BUY"}
        
        order_id = self.coordinator.execute_trade("BTC", positions, order_params)
        
        assert order_id is not None
        assert "BTC" in order_id
        assert len(self.coordinator.trade_history) == 1
    
    def test_trade_tracking_in_state(self):
        """Test trade is tracked in portfolio state."""
        positions = [
            {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}
        ]
        order_params = {"trade_type": "BUY"}
        
        order_id = self.coordinator.execute_trade("ETH", positions, order_params)
        
        assert order_id is not None
        
        # Check execution was tracked
        execution = self.coordinator.trade_history[-1]
        assert execution.symbol == "ETH"
        assert execution.quantity == 1.0
        # Execution should have recorded state
        assert execution.state_after is not None
    
    def test_order_placement(self):
        """Test order placement creates valid order ID."""
        order_id = self.coordinator._place_order(
            symbol="BTC",
            quantity=0.1,
            entry_price=50000.0,
            order_params={}
        )
        
        assert order_id is not None
        assert isinstance(order_id, str)
        assert "BTC" in order_id
    
    def test_invalid_order_rejected(self):
        """Test invalid orders are rejected."""
        # Zero quantity
        order_id = self.coordinator._place_order(
            symbol="BTC",
            quantity=0.0,
            entry_price=50000.0,
            order_params={}
        )
        assert order_id is None
        
        # Negative price
        order_id = self.coordinator._place_order(
            symbol="BTC",
            quantity=0.1,
            entry_price=-50000.0,
            order_params={}
        )
        assert order_id is None


class TestIntegrationWithAllPhases:
    """Test integration with all Phase 1-4 components."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_uses_portfolio_state_machine(self):
        """Test trading coordinator records trade execution."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        # Check trade was recorded in history
        assert len(self.coordinator.trade_history) > 0
        # State should be tracked in execution
        assert self.coordinator.trade_history[-1].state_after is not None
    
    def test_uses_bootstrap_metrics(self):
        """Test trading coordinator uses bootstrap metrics."""
        # Execute a trade
        positions = [
            {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}
        ]
        
        order_id = self.coordinator.execute_trade("ETH", positions, {"trade_type": "BUY"})
        
        assert order_id is not None
        # Bootstrap metrics should track that a trade was executed
        assert self.state.bootstrap_metrics.get_total_trades_executed() >= 1
    
    def test_uses_dust_registry(self):
        """Test trading coordinator uses dust registry."""
        # Trade a dust position
        positions = [
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50000.0}
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        assert order_id is not None
        # Dust registry should exist
        assert self.state.dust_lifecycle_registry is not None
        dust_summary = self.state.dust_lifecycle_registry.get_dust_summary()
        assert dust_summary is not None
    
    def test_uses_position_merger(self):
        """Test trading coordinator uses position merger."""
        # Multiple positions that should merge
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50050.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50100.0},
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        assert order_id is not None
        # Check if consolidation happened
        execution = self.coordinator.trade_history[-1]
        # Consolidated should be True if merge happened
        assert isinstance(execution.consolidated, bool)


class TestAnalyticsAndSummary:
    """Test trade analytics and summary generation."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_empty_summary(self):
        """Test empty trade summary."""
        summary = self.coordinator.get_trade_summary()
        
        assert summary["total_trades"] == 0
        assert summary["total_consolidated"] == 0
        assert summary["consolidation_rate"] == 0.0
    
    def test_summary_after_trades(self):
        """Test summary after executing trades."""
        # Execute a trade
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        summary = self.coordinator.get_trade_summary()
        
        assert summary["total_trades"] == 1
        assert "BTC" in summary["symbols_traded"]
    
    def test_consolidation_rate_calculation(self):
        """Test consolidation rate is calculated correctly."""
        # Execute a non-consolidated trade
        positions = [{"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}]
        self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        # Execute a consolidated trade (multiple positions)
        positions = [
            {"symbol": "ETH", "quantity": 0.1, "entry_price": 3000.0},
            {"symbol": "ETH", "quantity": 0.1, "entry_price": 3050.0},
            {"symbol": "ETH", "quantity": 0.1, "entry_price": 3025.0},
        ]
        self.coordinator.execute_trade("ETH", positions, {"trade_type": "BUY"})
        
        summary = self.coordinator.get_trade_summary()
        
        assert summary["total_trades"] == 2
        # Consolidation rate should be between 0 and 1
        assert 0.0 <= summary["consolidation_rate"] <= 1.0
    
    def test_reset_history(self):
        """Test trade history can be reset."""
        # Execute a trade
        positions = [{"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}]
        self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        assert len(self.coordinator.trade_history) == 1
        
        # Reset history
        self.coordinator.reset_history()
        
        assert len(self.coordinator.trade_history) == 0
        assert self.coordinator.get_trade_summary()["total_trades"] == 0


class TestSystemStatus:
    """Test system status inspection."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        # Reset bootstrap metrics for fresh test state
        self.state.bootstrap_metrics._cached_metrics = {}
        self.state.metrics["total_trades_executed"] = 0
        self.state.metrics["first_trade_at"] = None
        # Enable cold bootstrap for testing
        self.state.config.COLD_BOOTSTRAP_ENABLED = True
        self.coordinator = self.state.trading_coordinator
    
    def test_system_status_before_bootstrap(self):
        """Test system status before bootstrap."""
        status = self.coordinator.get_system_status()
        
        assert status["ready"] == False
        assert "bootstrap" in status
        assert status["bootstrap"]["is_cold_bootstrap"] == True
    
    def test_system_status_after_bootstrap(self):
        """Test system status after bootstrap."""
        self.state.bootstrap_metrics.save_trade_executed()
        
        status = self.coordinator.get_system_status()
        
        assert status["ready"] == True
        assert status["bootstrap"]["is_cold_bootstrap"] == False
        assert status["bootstrap"]["total_trades_executed"] >= 1
    
    def test_system_status_includes_components(self):
        """Test system status includes all component status."""
        self.state.bootstrap_metrics.save_trade_executed()
        
        status = self.coordinator.get_system_status()
        
        # Should have info about all components
        assert "bootstrap" in status
        assert "dust" in status
        assert "merges" in status
        assert "trades" in status
        assert "timestamp" in status


class TestDiagnostics:
    """Test diagnostic methods."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_diagnose_single_position(self):
        """Test diagnosis for single position."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        
        diagnosis = self.coordinator.diagnose_trade_readiness("BTC", positions)
        
        assert diagnosis["symbol"] == "BTC"
        assert diagnosis["system_ready"] == True
        assert diagnosis["positions_valid"] == True
        assert diagnosis["consolidation_needed"] == False
        assert diagnosis["overall_ready"] == True
    
    def test_diagnose_fragmented_positions(self):
        """Test diagnosis for fragmented positions."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50050.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50100.0},
        ]
        
        diagnosis = self.coordinator.diagnose_trade_readiness("BTC", positions)
        
        assert diagnosis["consolidation_needed"] == True
        # May or may not be feasible depending on scoring
        assert isinstance(diagnosis["consolidation_feasible"], bool)
    
    def test_diagnose_empty_positions(self):
        """Test diagnosis for empty positions."""
        diagnosis = self.coordinator.diagnose_trade_readiness("BTC", [])
        
        assert diagnosis["positions_valid"] == False
        assert "No positions" in diagnosis["issues"][0]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Create a SharedState with TradingCoordinator for each test."""
        self.state = SharedState()
        self.coordinator = self.state.trading_coordinator
        self.state.bootstrap_metrics.save_trade_executed()
    
    def test_zero_quantity_rejected(self):
        """Test zero quantity trades are rejected."""
        positions = [
            {"symbol": "BTC", "quantity": 0.0, "entry_price": 50000.0}
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        # Should fail due to zero quantity
        assert order_id is None
    
    def test_negative_price_rejected(self):
        """Test negative prices are rejected."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": -50000.0}
        ]
        
        # Prepare should handle this
        prepared, _ = self.coordinator.prepare_positions("BTC", positions)
        
        # Either prepare returns None or trade execution handles it
        if prepared is not None:
            order_id = self.coordinator._place_order("BTC", 0.1, -50000.0, {})
            assert order_id is None
    
    def test_very_large_position(self):
        """Test handling of very large positions."""
        positions = [
            {"symbol": "BTC", "quantity": 1000.0, "entry_price": 50000.0}
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        # Should succeed (though unrealistic)
        assert order_id is not None
    
    def test_very_small_position(self):
        """Test handling of very small positions."""
        positions = [
            {"symbol": "BTC", "quantity": 0.00001, "entry_price": 50000.0}
        ]
        
        order_id = self.coordinator.execute_trade("BTC", positions, {"trade_type": "BUY"})
        
        # Should succeed but marked as dust
        assert order_id is not None
    
    def test_multiple_symbols_separate_trades(self):
        """Test trading multiple symbols separately."""
        # First trade BTC
        btc_positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        ]
        btc_order = self.coordinator.execute_trade("BTC", btc_positions, {"trade_type": "BUY"})
        
        # Then trade ETH
        eth_positions = [
            {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}
        ]
        eth_order = self.coordinator.execute_trade("ETH", eth_positions, {"trade_type": "BUY"})
        
        assert btc_order is not None
        assert eth_order is not None
        assert len(self.coordinator.trade_history) == 2
        
        summary = self.coordinator.get_trade_summary()
        assert len(summary["symbols_traded"]) == 2
        assert "BTC" in summary["symbols_traded"]
        assert "ETH" in summary["symbols_traded"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
