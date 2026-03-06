"""
Test suite for Phase 1: Portfolio State Machine implementation.
Tests the new PortfolioState enum and state detection logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from core.shared_state import SharedState, SharedStateConfig, PortfolioState


class TestPortfolioStateEnum:
    """Test the PortfolioState enum definition."""
    
    def test_portfolio_state_enum_exists(self):
        """Verify PortfolioState enum is defined."""
        assert hasattr(PortfolioState, 'EMPTY_PORTFOLIO')
        assert hasattr(PortfolioState, 'PORTFOLIO_WITH_DUST')
        assert hasattr(PortfolioState, 'PORTFOLIO_ACTIVE')
        assert hasattr(PortfolioState, 'PORTFOLIO_RECOVERING')
        assert hasattr(PortfolioState, 'COLD_BOOTSTRAP')
    
    def test_portfolio_state_values(self):
        """Verify PortfolioState enum values are correct."""
        assert PortfolioState.EMPTY_PORTFOLIO.value == "EMPTY_PORTFOLIO"
        assert PortfolioState.PORTFOLIO_WITH_DUST.value == "PORTFOLIO_WITH_DUST"
        assert PortfolioState.PORTFOLIO_ACTIVE.value == "PORTFOLIO_ACTIVE"
        assert PortfolioState.PORTFOLIO_RECOVERING.value == "PORTFOLIO_RECOVERING"
        assert PortfolioState.COLD_BOOTSTRAP.value == "COLD_BOOTSTRAP"


class TestPositionSignificanceHelper:
    """Test the _is_position_significant() helper method."""
    
    def setup_method(self):
        """Set up shared state mock for each test."""
        self.config = Mock(spec=SharedStateConfig)
        self.config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        self.config.COLD_BOOTSTRAP_ENABLED = False
        self.config.LIVE_MODE = False
        
        self.shared_state = Mock(spec=SharedState)
        self.shared_state.config = self.config
        self.shared_state.logger = Mock()
        
        # Bind the actual method to the mock
        self.shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(self.shared_state)
    
    def test_significant_position_above_threshold(self):
        """Test that position above threshold is marked significant."""
        self.shared_state.latest_price = Mock(return_value=50000.0)  # $50k per BTC
        
        # 1 BTC = $50,000 notional (>> $1.0 threshold)
        is_sig = self.shared_state._is_position_significant("BTCUSDT", 1.0)
        assert is_sig is True
    
    def test_dust_position_below_threshold(self):
        """Test that position below threshold is marked as dust."""
        self.shared_state.latest_price = Mock(return_value=50000.0)
        
        # 0.00001 BTC = $0.50 notional (< $1.0 threshold)
        is_sig = self.shared_state._is_position_significant("BTCUSDT", 0.00001)
        assert is_sig is False
    
    def test_position_at_threshold_boundary(self):
        """Test position exactly at threshold."""
        self.shared_state.latest_price = Mock(return_value=100.0)
        
        # 0.01 quantity * $100 price = $1.0 notional (exactly at threshold)
        is_sig = self.shared_state._is_position_significant("USDT", 0.01)
        assert is_sig is True
    
    def test_custom_threshold_configuration(self):
        """Test with custom dust threshold."""
        self.config.PERMANENT_DUST_USDT_THRESHOLD = 5.0  # Higher threshold
        self.shared_state.latest_price = Mock(return_value=100.0)
        
        # 0.04 quantity * $100 = $4.0 notional (< $5.0 threshold)
        is_sig = self.shared_state._is_position_significant("USDT", 0.04)
        assert is_sig is False
    
    def test_no_price_available(self):
        """Test when price is unavailable."""
        self.shared_state.latest_price = Mock(return_value=None)
        
        is_sig = self.shared_state._is_position_significant("UNKNOWN", 1.0)
        assert is_sig is False
    
    def test_price_zero_or_negative(self):
        """Test when price is zero or negative."""
        self.shared_state.latest_price = Mock(return_value=0.0)
        
        is_sig = self.shared_state._is_position_significant("BADPRICE", 1.0)
        assert is_sig is False
    
    def test_exception_during_price_lookup(self):
        """Test graceful handling when price lookup throws exception."""
        self.shared_state.latest_price = Mock(side_effect=Exception("Price API error"))
        
        # Should return True (assume significant) to avoid false positives
        is_sig = self.shared_state._is_position_significant("BTCUSDT", 0.0001)
        assert is_sig is True
    
    def test_negative_quantity_treated_as_absolute(self):
        """Test that negative quantities (short positions) are treated correctly."""
        self.shared_state.latest_price = Mock(return_value=100.0)
        
        # -0.01 quantity (short) * $100 = $1.0 notional value (should use abs)
        is_sig = self.shared_state._is_position_significant("USDT", -0.01)
        assert is_sig is True


class TestEmptyPortfolioDetection:
    """Test detection of empty portfolio state."""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_detection(self):
        """Test that completely empty portfolio is detected as EMPTY_PORTFOLIO."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        shared_state.get_open_positions = Mock(return_value=[])
        shared_state.latest_price = Mock(return_value=100.0)
        
        # Bind actual methods
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.EMPTY_PORTFOLIO.value


class TestDustOnlyPortfolioDetection:
    """Test detection of dust-only portfolio state."""
    
    @pytest.mark.asyncio
    async def test_dust_only_portfolio_detection(self):
        """Test that portfolio with only dust is detected as PORTFOLIO_WITH_DUST."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        
        # Set up dust position
        dust_position = {
            "symbol": "BTCUSDT",
            "qty": 0.00001,  # Dust amount
        }
        shared_state.get_open_positions = Mock(return_value=[dust_position])
        shared_state.latest_price = Mock(return_value=50000.0)
        
        # Bind actual methods
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.PORTFOLIO_WITH_DUST.value


class TestActivePortfolioDetection:
    """Test detection of active portfolio state."""
    
    @pytest.mark.asyncio
    async def test_active_portfolio_detection(self):
        """Test that portfolio with significant positions is detected as PORTFOLIO_ACTIVE."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        
        # Set up significant position
        position = {
            "symbol": "BTCUSDT",
            "qty": 1.0,  # 1 BTC = ~$50k
        }
        shared_state.get_open_positions = Mock(return_value=[position])
        shared_state.latest_price = Mock(return_value=50000.0)
        
        # Bind actual methods
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.PORTFOLIO_ACTIVE.value
    
    @pytest.mark.asyncio
    async def test_mixed_positions_with_significant_preferred(self):
        """Test that portfolio with mixed dust+significant is detected as ACTIVE."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        
        # Mixed positions: 1 significant, 1 dust
        positions = [
            {"symbol": "BTCUSDT", "qty": 1.0},      # Significant
            {"symbol": "ETHUSDT", "qty": 0.00001},  # Dust
        ]
        shared_state.get_open_positions = Mock(return_value=positions)
        
        def mock_latest_price(symbol):
            prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
            return prices.get(symbol, 100.0)
        
        shared_state.latest_price = Mock(side_effect=mock_latest_price)
        
        # Bind actual methods
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.PORTFOLIO_ACTIVE.value


class TestColdBootstrapDetection:
    """Test detection of cold bootstrap state."""
    
    @pytest.mark.asyncio
    async def test_cold_bootstrap_state_returned(self):
        """Test that cold bootstrap state is returned when is_cold_bootstrap() is True."""
        shared_state = Mock(spec=SharedState)
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=True)
        
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.COLD_BOOTSTRAP.value


class TestIsPortfolioFlat:
    """Test the updated is_portfolio_flat() method."""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_is_flat(self):
        """Test that empty portfolio is considered flat."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        shared_state.get_open_positions = Mock(return_value=[])
        shared_state.latest_price = Mock(return_value=100.0)
        
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        shared_state.is_portfolio_flat = \
            SharedState.is_portfolio_flat.__get__(shared_state)
        
        is_flat = await shared_state.is_portfolio_flat()
        assert is_flat is True
    
    @pytest.mark.asyncio
    async def test_dust_only_portfolio_is_not_flat(self):
        """Test that dust-only portfolio is NOT considered flat (critical fix!)."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        
        dust_position = {"symbol": "BTCUSDT", "qty": 0.00001}
        shared_state.get_open_positions = Mock(return_value=[dust_position])
        shared_state.latest_price = Mock(return_value=50000.0)
        
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        shared_state.is_portfolio_flat = \
            SharedState.is_portfolio_flat.__get__(shared_state)
        
        is_flat = await shared_state.is_portfolio_flat()
        # THIS IS THE CRITICAL FIX: dust-only should NOT be flat
        assert is_flat is False
    
    @pytest.mark.asyncio
    async def test_active_portfolio_is_not_flat(self):
        """Test that active portfolio is not flat."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.COLD_BOOTSTRAP_ENABLED = False
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        
        position = {"symbol": "BTCUSDT", "qty": 1.0}
        shared_state.get_open_positions = Mock(return_value=[position])
        shared_state.latest_price = Mock(return_value=50000.0)
        
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        shared_state.is_portfolio_flat = \
            SharedState.is_portfolio_flat.__get__(shared_state)
        
        is_flat = await shared_state.is_portfolio_flat()
        assert is_flat is False


class TestStateTransitionLogic:
    """Test the overall state transition logic."""
    
    @pytest.mark.asyncio
    async def test_cold_bootstrap_to_active_transition(self):
        """Test transition from COLD_BOOTSTRAP to PORTFOLIO_ACTIVE."""
        config = Mock(spec=SharedStateConfig)
        config.PERMANENT_DUST_USDT_THRESHOLD = 1.0
        config.LIVE_MODE = False
        
        shared_state = Mock(spec=SharedState)
        shared_state.config = config
        shared_state.logger = Mock()
        shared_state.latest_price = Mock(return_value=50000.0)
        
        shared_state._is_position_significant = \
            SharedState._is_position_significant.__get__(shared_state)
        shared_state.get_portfolio_state = \
            SharedState.get_portfolio_state.__get__(shared_state)
        
        # Start in cold bootstrap
        shared_state.is_cold_bootstrap = Mock(return_value=True)
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.COLD_BOOTSTRAP.value
        
        # Transition to active after first trade
        shared_state.is_cold_bootstrap = Mock(return_value=False)
        position = {"symbol": "BTCUSDT", "qty": 1.0}
        shared_state.get_open_positions = Mock(return_value=[position])
        
        state = await shared_state.get_portfolio_state()
        assert state == PortfolioState.PORTFOLIO_ACTIVE.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
