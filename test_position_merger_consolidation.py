"""
Test suite for Phase 4: Position Merger & Consolidation implementation.
Tests position consolidation and merging functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from core.shared_state import SharedState, SharedStateConfig, PositionMerger, MergeOperation, MergeImpact


class TestPositionMergerBasics:
    """Test basic PositionMerger functionality."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_position_merger_initialization(self):
        """Test that PositionMerger initializes correctly."""
        assert self.merger is not None
        assert self.merger.merge_threshold_usd == 1.0
        assert self.merger.max_entry_price_deviation == 0.05
        assert len(self.merger.merge_history) == 0
    
    def test_merge_operation_creation(self):
        """Test creating a MergeOperation."""
        op = MergeOperation(
            symbol="BTC",
            source_quantity=0.001,
            target_quantity=0.002,
            source_entry_price=50000.0,
            target_entry_price=50100.0,
            merged_quantity=0.003,
            merged_entry_price=50050.0
        )
        assert op.symbol == "BTC"
        assert op.merged_quantity == 0.003
        assert op.merge_type == "POSITION_MERGE"
    
    def test_merge_impact_creation(self):
        """Test creating a MergeImpact."""
        impact = MergeImpact(
            symbol="BTC",
            cost_basis_change=10.0,
            new_average_entry=50050.0,
            quantity_change=0.003,
            order_count_reduction=2,
            estimated_slippage=2.5
        )
        assert impact.symbol == "BTC"
        assert impact.order_count_reduction == 2
        assert impact.feasibility_score == 1.0


class TestMergeCandidateDetection:
    """Test identification of merge candidates."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_no_candidates_single_position(self):
        """Test that single position returns no candidates."""
        positions = {
            "pos1": {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        }
        
        candidates = self.merger.identify_merge_candidates(positions)
        assert len(candidates) == 0
    
    def test_detect_multiple_same_symbol(self):
        """Test detection of multiple positions in same symbol."""
        positions = {
            "pos1": {"symbol": "BTC", "quantity": 0.001, "entry_price": 50000.0},
            "pos2": {"symbol": "BTC", "quantity": 0.002, "entry_price": 50100.0},
            "pos3": {"symbol": "BTC", "quantity": 0.0005, "entry_price": 49900.0},
        }
        
        candidates = self.merger.identify_merge_candidates(positions)
        assert "BTC" in candidates
        assert len(candidates["BTC"]) == 3
    
    def test_detect_multiple_symbols(self):
        """Test detection across multiple symbols."""
        positions = {
            "btc1": {"symbol": "BTC", "quantity": 0.001, "entry_price": 50000.0},
            "btc2": {"symbol": "BTC", "quantity": 0.002, "entry_price": 50100.0},
            "eth1": {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0},
            "eth2": {"symbol": "ETH", "quantity": 2.0, "entry_price": 3050.0},
            "ada": {"symbol": "ADA", "quantity": 100.0, "entry_price": 0.5},
        }
        
        candidates = self.merger.identify_merge_candidates(positions)
        assert "BTC" in candidates
        assert "ETH" in candidates
        assert "ADA" not in candidates
        assert len(candidates["BTC"]) == 2
        assert len(candidates["ETH"]) == 2


class TestEntryPriceCalculation:
    """Test weighted average entry price calculation."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_weighted_entry_equal_quantities(self):
        """Test weighted entry price with equal quantities."""
        positions = [
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 200.0},
        ]
        
        avg_price = self.merger.calculate_weighted_entry_price(positions)
        assert avg_price == 150.0
    
    def test_weighted_entry_unequal_quantities(self):
        """Test weighted entry price with unequal quantities."""
        positions = [
            {"quantity": 2.0, "entry_price": 100.0},  # 200 notional
            {"quantity": 1.0, "entry_price": 300.0},  # 300 notional
        ]
        
        avg_price = self.merger.calculate_weighted_entry_price(positions)
        # (2*100 + 1*300) / (2+1) = 500/3 = 166.67
        assert abs(avg_price - 166.67) < 0.01
    
    def test_weighted_entry_many_positions(self):
        """Test weighted entry price with many positions."""
        positions = [
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 200.0},
        ]
        
        avg_price = self.merger.calculate_weighted_entry_price(positions)
        # (1*100 + 1*100 + 1*200) / 3 = 400/3 = 133.33
        assert abs(avg_price - 133.33) < 0.01
    
    def test_weighted_entry_zero_quantity(self):
        """Test that zero quantity returns 0."""
        positions = [
            {"quantity": 0.0, "entry_price": 100.0},
        ]
        
        avg_price = self.merger.calculate_weighted_entry_price(positions)
        assert avg_price == 0.0


class TestMergeValidation:
    """Test merge validation logic."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_validate_different_symbols(self):
        """Test that different symbols fail validation."""
        pos1 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        pos2 = {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0}
        
        assert self.merger.validate_merge(pos1, pos2) == False
    
    def test_validate_entry_price_deviation(self):
        """Test that high entry price deviation fails."""
        pos1 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        pos2 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 60000.0}  # 20% deviation
        
        assert self.merger.validate_merge(pos1, pos2) == False
    
    def test_validate_similar_entry_prices(self):
        """Test that similar entry prices pass."""
        pos1 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0}
        pos2 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 51000.0}  # 2% deviation
        
        assert self.merger.validate_merge(pos1, pos2) == True
    
    def test_validate_zero_quantity(self):
        """Test that zero quantity fails."""
        pos1 = {"symbol": "BTC", "quantity": 0.0, "entry_price": 50000.0}
        pos2 = {"symbol": "BTC", "quantity": 0.1, "entry_price": 50100.0}
        
        assert self.merger.validate_merge(pos1, pos2) == False


class TestMergeImpactCalculation:
    """Test merge impact analysis."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_impact_two_positions(self):
        """Test impact calculation for two positions."""
        positions = [
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 120.0},
        ]
        
        impact = self.merger.calculate_merge_impact("TEST", positions)
        
        assert impact.symbol == "TEST"
        assert impact.quantity_change == 2.0
        assert impact.order_count_reduction == 1
        assert impact.new_average_entry == 110.0
        assert impact.feasibility_score > 0.0
    
    def test_impact_three_positions(self):
        """Test impact with three positions."""
        positions = [
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 100.0},
            {"quantity": 1.0, "entry_price": 100.0},
        ]
        
        impact = self.merger.calculate_merge_impact("TEST", positions)
        
        assert impact.quantity_change == 3.0
        assert impact.order_count_reduction == 2
        assert impact.new_average_entry == 100.0


class TestMergeExecution:
    """Test merge execution."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_merge_two_positions(self):
        """Test merging two positions."""
        import pytest
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50500.0},
        ]
        
        operation = self.merger.merge_positions("BTC", positions)
        
        assert operation is not None
        assert operation.symbol == "BTC"
        assert operation.merged_quantity == pytest.approx(0.15, rel=1e-9)
        assert len(self.merger.merge_history) == 1
    
    def test_merge_updates_history(self):
        """Test that merging updates history."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50500.0},
        ]
        
        assert len(self.merger.merge_history) == 0
        
        self.merger.merge_positions("BTC", positions)
        
        assert len(self.merger.merge_history) == 1
        assert self.merger.merge_history[0].symbol == "BTC"
    
    def test_merge_incompatible_positions(self):
        """Test merging incompatible positions returns None."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 65000.0},  # 30% deviation
        ]
        
        operation = self.merger.merge_positions("BTC", positions)
        
        assert operation is None


class TestMergeDecision:
    """Test merge decision logic."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_should_merge_good_candidates(self):
        """Test that good merge candidates are identified."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50100.0},
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50050.0},
        ]
        
        should_merge = self.merger.should_merge("BTC", positions)
        # Should merge: good feasibility, low cost basis change
        assert should_merge == True
    
    def test_should_not_merge_single_position(self):
        """Test that single position is not merged."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
        ]
        
        should_merge = self.merger.should_merge("BTC", positions)
        assert should_merge == False


class TestDustConsolidation:
    """Test dust-specific consolidation."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_consolidate_dust_positions(self):
        """Test consolidating dust positions."""
        positions = [
            {"symbol": "BTC", "quantity": 0.001, "entry_price": 50000.0},  # $50 notional
            {"symbol": "BTC", "quantity": 0.0005, "entry_price": 50000.0},  # $25 notional
            {"symbol": "BTC", "quantity": 0.5, "entry_price": 50000.0},    # $25,000 notional
        ]
        
        # Consolidate positions < $100
        operation = self.merger.consolidate_dust("BTC", positions, dust_threshold=100.0)
        
        assert operation is not None
        assert operation.merged_quantity == 0.0015
    
    def test_dust_consolidation_no_dust(self):
        """Test dust consolidation with no dust."""
        positions = [
            {"symbol": "BTC", "quantity": 1.0, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 2.0, "entry_price": 50000.0},
        ]
        
        operation = self.merger.consolidate_dust("BTC", positions, dust_threshold=1.0)
        
        # No dust positions, should return None
        assert operation is None


class TestMergeSummary:
    """Test merge summary and analytics."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_empty_summary(self):
        """Test summary with no merges."""
        summary = self.merger.get_merge_summary()
        
        assert summary["total_merges"] == 0
        assert summary["symbols_merged"] == 0
        assert summary["total_quantity_consolidated"] == 0.0
    
    def test_summary_after_merges(self):
        """Test summary after performing merges."""
        positions1 = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50500.0},
        ]
        positions2 = [
            {"symbol": "ETH", "quantity": 1.0, "entry_price": 3000.0},
            {"symbol": "ETH", "quantity": 2.0, "entry_price": 3100.0},
        ]
        
        self.merger.merge_positions("BTC", positions1)
        self.merger.merge_positions("ETH", positions2)
        
        summary = self.merger.get_merge_summary()
        
        assert summary["total_merges"] == 2
        assert summary["symbols_merged"] == 2
        assert summary["total_quantity_consolidated"] == 3.15  # 0.15 + 3.0
    
    def test_reset_history(self):
        """Test resetting merge history."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50500.0},
        ]
        
        self.merger.merge_positions("BTC", positions)
        assert len(self.merger.merge_history) == 1
        
        self.merger.reset_history()
        assert len(self.merger.merge_history) == 0


class TestPositionMergerIntegration:
    """Test PositionMerger integration with SharedState."""
    
    def test_shared_state_has_position_merger(self):
        """Test that SharedState has position_merger instance."""
        config = {
            "COLD_BOOTSTRAP_ENABLED": False,
            "LIVE_MODE": False,
        }
        shared_state = SharedState(config=config)
        
        assert hasattr(shared_state, 'position_merger')
        assert isinstance(shared_state.position_merger, PositionMerger)
    
    def test_position_merger_multiple_instances(self):
        """Test creating multiple SharedState instances with separate mergers."""
        config1 = {"COLD_BOOTSTRAP_ENABLED": False, "LIVE_MODE": False}
        config2 = {"COLD_BOOTSTRAP_ENABLED": False, "LIVE_MODE": False}
        
        shared_state1 = SharedState(config=config1)
        shared_state2 = SharedState(config=config2)
        
        # Each should have separate merger instance
        assert shared_state1.position_merger is not shared_state2.position_merger


class TestMergeEdgeCases:
    """Test edge cases in position merging."""
    
    def setup_method(self):
        """Create a position merger for each test."""
        self.merger = PositionMerger()
    
    def test_merge_operation_to_dict(self):
        """Test MergeOperation serialization."""
        op = MergeOperation(
            symbol="BTC",
            source_quantity=0.1,
            target_quantity=0.05,
            source_entry_price=50000.0,
            target_entry_price=50500.0,
            merged_quantity=0.15,
            merged_entry_price=50167.0
        )
        
        op_dict = op.to_dict()
        assert op_dict["symbol"] == "BTC"
        assert op_dict["merged_quantity"] == 0.15
        assert "timestamp" in op_dict
    
    def test_merge_impact_to_dict(self):
        """Test MergeImpact serialization."""
        impact = MergeImpact(
            symbol="BTC",
            cost_basis_change=50.0,
            new_average_entry=50100.0,
            quantity_change=0.15,
            order_count_reduction=1,
            estimated_slippage=2.5
        )
        
        impact_dict = impact.to_dict()
        assert impact_dict["symbol"] == "BTC"
        assert impact_dict["order_count_reduction"] == 1
    
    def test_merge_identical_prices(self):
        """Test merging positions with identical entry prices."""
        positions = [
            {"symbol": "BTC", "quantity": 0.1, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.05, "entry_price": 50000.0},
            {"symbol": "BTC", "quantity": 0.02, "entry_price": 50000.0},
        ]
        
        operation = self.merger.merge_positions("BTC", positions)
        
        assert operation is not None
        assert operation.merged_entry_price == 50000.0
        assert operation.merged_quantity == 0.17
    
    def test_merge_many_positions(self):
        """Test merging many positions."""
        import pytest
        positions = [
            {"symbol": "TEST", "quantity": 0.1, "entry_price": 100.0}
            for _ in range(10)
        ]
        
        operation = self.merger.merge_positions("TEST", positions)
        
        assert operation is not None
        assert operation.merged_quantity == pytest.approx(1.0, rel=1e-9)
        assert operation.merged_entry_price == pytest.approx(100.0, rel=1e-9)
