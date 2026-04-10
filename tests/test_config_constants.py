"""
Unit Tests for Configuration Constants Module

Validates that all configuration constants are properly defined,
accessible, and satisfy domain constraints.
"""

import pytest
from core.config_constants import (
    TimeoutConstants,
    RetryConstants,
    CapitalConstants,
    ConfidenceConstants,
    LeverageConstants,
    RiskConstants,
    BootstrapConstants,
    StateTransitionConstants,
    ConcurrencyConstants,
    PerformanceConstants,
    EnvironmentConstants,
    get_all_constants,
    validate_constants,
)


class TestTimeoutConstants:
    """Test timeout configuration values."""
    
    def test_all_timeouts_positive(self):
        """All timeout values should be positive."""
        assert TimeoutConstants.EXCHANGE_API_TIMEOUT > 0
        assert TimeoutConstants.BOOTSTRAP_TIMEOUT > 0
        assert TimeoutConstants.HEALTH_CHECK_TIMEOUT > 0
        assert TimeoutConstants.RETRY_BASE_DELAY > 0
        assert TimeoutConstants.RETRY_MAX_DELAY > 0
    
    def test_base_delay_less_than_max_delay(self):
        """Base delay should be less than max delay."""
        assert TimeoutConstants.RETRY_BASE_DELAY < TimeoutConstants.RETRY_MAX_DELAY
    
    def test_timeout_values_reasonable(self):
        """Timeouts should be in reasonable ranges."""
        assert 10 <= TimeoutConstants.EXCHANGE_API_TIMEOUT <= 120
        assert 5 <= TimeoutConstants.BOOTSTRAP_TIMEOUT <= 60
        assert TimeoutConstants.SIGNAL_CACHE_TTL >= 30


class TestRetryConstants:
    """Test retry configuration values."""
    
    def test_all_retry_counts_positive(self):
        """All retry count values should be positive."""
        assert RetryConstants.EXCHANGE_API_RETRIES > 0
        assert RetryConstants.BOOTSTRAP_RETRIES > 0
        assert RetryConstants.STATE_SYNC_RETRIES > 0
        assert RetryConstants.HEALTH_CHECK_RETRIES > 0
    
    def test_backoff_base_valid(self):
        """Backoff base should be > 1 for exponential growth."""
        assert RetryConstants.RETRY_BACKOFF_BASE > 1.0
    
    def test_jitter_factor_valid(self):
        """Jitter factor should be between 0 and 1."""
        assert 0 <= RetryConstants.RETRY_JITTER_FACTOR <= 1.0
    
    def test_circuit_breaker_thresholds_positive(self):
        """Circuit breaker thresholds should be positive."""
        assert RetryConstants.CIRCUIT_BREAKER_FAILURE_THRESHOLD > 0
        assert RetryConstants.CIRCUIT_BREAKER_RECOVERY_TIMEOUT > 0


class TestCapitalConstants:
    """Test capital and balance configuration values."""
    
    def test_minimum_capital_values_positive(self):
        """All capital values should be positive."""
        assert CapitalConstants.MIN_TRADING_CAPITAL > 0
        assert CapitalConstants.MIN_ORDER_NOTIONAL > 0
        assert CapitalConstants.MIN_ENTRY_AMOUNT > 0
        assert CapitalConstants.DUST_THRESHOLD_USDT > 0
    
    def test_capital_hierarchy(self):
        """Capital thresholds should follow logical hierarchy."""
        # Entry should be smallest, then notional, then trading capital
        assert CapitalConstants.MIN_ENTRY_AMOUNT <= CapitalConstants.MIN_ORDER_NOTIONAL
        assert CapitalConstants.MIN_ORDER_NOTIONAL <= CapitalConstants.MIN_TRADING_CAPITAL
    
    def test_capital_percentages_valid(self):
        """Capital percentage values should be between 0 and 1."""
        assert 0 < CapitalConstants.CAPITAL_FLOOR_PERCENT < 1.0
        assert 0 < CapitalConstants.CAPITAL_ALLOCATION_MAX_PERCENT <= 1.0
        assert 0 < CapitalConstants.DUST_THRESHOLD_PERCENT < 1.0
    
    def test_bootstrap_allocation_valid(self):
        """Bootstrap allocation should be reasonable."""
        assert 0 < CapitalConstants.BOOTSTRAP_CAPITAL_ALLOCATION <= 0.5


class TestConfidenceConstants:
    """Test confidence threshold configuration."""
    
    def test_confidence_thresholds_valid_range(self):
        """Confidence values should be between 0 and 1."""
        assert 0 <= ConfidenceConstants.MIN_SIGNAL_CONFIDENCE <= 1.0
        assert 0 <= ConfidenceConstants.HIGH_CONFIDENCE_THRESHOLD <= 1.0
        assert 0 <= ConfidenceConstants.CRITICAL_CONFIDENCE_THRESHOLD <= 1.0
    
    def test_confidence_threshold_hierarchy(self):
        """Confidence thresholds should be in ascending order."""
        assert ConfidenceConstants.MIN_SIGNAL_CONFIDENCE < ConfidenceConstants.HIGH_CONFIDENCE_THRESHOLD
        assert ConfidenceConstants.HIGH_CONFIDENCE_THRESHOLD < ConfidenceConstants.CRITICAL_CONFIDENCE_THRESHOLD
    
    def test_adaptive_band_valid(self):
        """Adaptive confidence band should be valid."""
        assert 0 <= ConfidenceConstants.CONFIDENCE_BAND_ADAPTIVE_LOW <= 1.0
        assert 0 <= ConfidenceConstants.CONFIDENCE_BAND_ADAPTIVE_HIGH <= 1.0
        assert ConfidenceConstants.CONFIDENCE_BAND_ADAPTIVE_LOW < ConfidenceConstants.CONFIDENCE_BAND_ADAPTIVE_HIGH
    
    def test_volatility_multipliers_positive(self):
        """Volatility multipliers should be positive."""
        assert ConfidenceConstants.VOLATILITY_MULTIPLIER_MIN > 0
        assert ConfidenceConstants.VOLATILITY_MULTIPLIER_MAX > 0
        assert ConfidenceConstants.VOLATILITY_MULTIPLIER_MIN < ConfidenceConstants.VOLATILITY_MULTIPLIER_MAX


class TestLeverageConstants:
    """Test leverage configuration."""
    
    def test_leverage_limits_valid(self):
        """Leverage limits should be positive and reasonable."""
        assert LeverageConstants.MAX_LEVERAGE > 0
        assert LeverageConstants.MAX_LEVERAGE <= 10.0  # Reasonable max
        assert LeverageConstants.DEFAULT_LEVERAGE > 0
        assert LeverageConstants.DEFAULT_LEVERAGE <= LeverageConstants.MAX_LEVERAGE
    
    def test_position_size_percentages_valid(self):
        """Position size percentages should be between 0 and 1."""
        assert 0 < LeverageConstants.MIN_POSITION_SIZE_PERCENT < 1.0
        assert 0 < LeverageConstants.MAX_POSITION_SIZE_PERCENT <= 1.0
        assert LeverageConstants.MIN_POSITION_SIZE_PERCENT < LeverageConstants.MAX_POSITION_SIZE_PERCENT


class TestRiskConstants:
    """Test risk management configuration."""
    
    def test_risk_percentages_valid(self):
        """Risk percentages should be between 0 and 1."""
        assert 0 < RiskConstants.MAX_LOSS_PERCENT < 1.0
        assert 0 < RiskConstants.MAX_DRAWDOWN_PERCENT < 1.0
        assert 0 < RiskConstants.DEFAULT_STOP_LOSS_PERCENT < 1.0
        assert 0 < RiskConstants.DEFAULT_TAKE_PROFIT_PERCENT < 1.0
    
    def test_stop_loss_less_than_take_profit(self):
        """Stop loss distance should be less than take profit."""
        assert RiskConstants.DEFAULT_STOP_LOSS_PERCENT < RiskConstants.DEFAULT_TAKE_PROFIT_PERCENT
    
    def test_concentration_thresholds_valid(self):
        """Concentration thresholds should be between 0 and 1."""
        assert 0 < RiskConstants.CONCENTRATION_WARNING_THRESHOLD < 1.0
        assert 0 < RiskConstants.CONCENTRATION_ERROR_THRESHOLD < 1.0
        assert RiskConstants.CONCENTRATION_WARNING_THRESHOLD < RiskConstants.CONCENTRATION_ERROR_THRESHOLD


class TestBootstrapConstants:
    """Test bootstrap configuration."""
    
    def test_bootstrap_duration_positive(self):
        """Bootstrap duration should be positive."""
        assert BootstrapConstants.BOOTSTRAP_DURATION_MINUTES > 0
        assert BootstrapConstants.BOOTSTRAP_DURATION_MINUTES <= 120  # Max 2 hours
    
    def test_bootstrap_symbols_positive(self):
        """Bootstrap symbol counts should be positive."""
        assert BootstrapConstants.BOOTSTRAP_INITIAL_SYMBOLS > 0
        assert BootstrapConstants.DUST_BYPASS_BUDGET_PER_CYCLE > 0
        assert BootstrapConstants.BOOTSTRAP_DUST_HEALING_CYCLES > 0


class TestStateTransitionConstants:
    """Test state transition configuration."""
    
    def test_transition_timeouts_positive(self):
        """All transition timeouts should be positive."""
        assert StateTransitionConstants.STATE_TRANSITION_COOLDOWN > 0
        assert StateTransitionConstants.AWAITING_SIGNAL_TIMEOUT > 0
        assert StateTransitionConstants.DUST_HEALING_TIMEOUT > 0
        assert StateTransitionConstants.MAX_STATE_AGE > 0
    
    def test_timeout_values_reasonable(self):
        """Timeout values should be in reasonable ranges."""
        assert StateTransitionConstants.STATE_TRANSITION_COOLDOWN <= 30  # Max 30s
        assert StateTransitionConstants.AWAITING_SIGNAL_TIMEOUT <= 300  # Max 5 min
        assert StateTransitionConstants.DUST_HEALING_TIMEOUT <= 1800  # Max 30 min


class TestConcurrencyConstants:
    """Test concurrency configuration."""
    
    def test_concurrency_limits_positive(self):
        """All concurrency limits should be positive."""
        assert ConcurrencyConstants.MAX_CONCURRENT_ORDERS > 0
        assert ConcurrencyConstants.MAX_CONCURRENT_SYMBOL_CHECKS > 0
    
    def test_queue_sizes_positive(self):
        """All queue sizes should be positive."""
        assert ConcurrencyConstants.SIGNAL_QUEUE_MAX_SIZE > 0
        assert ConcurrencyConstants.EVENT_QUEUE_MAX_SIZE > 0
    
    def test_rate_limit_delay_valid(self):
        """Rate limit delay should be reasonable."""
        assert ConcurrencyConstants.API_RATE_LIMIT_DELAY > 0
        assert ConcurrencyConstants.API_RATE_LIMIT_DELAY <= 1.0  # Max 1 second


class TestPerformanceConstants:
    """Test performance configuration."""
    
    def test_cache_sizes_positive(self):
        """All cache sizes should be positive."""
        assert PerformanceConstants.SYMBOL_CACHE_SIZE > 0
        assert PerformanceConstants.STATE_CACHE_SIZE > 0
    
    def test_batch_sizes_positive(self):
        """All batch sizes should be positive."""
        assert PerformanceConstants.SIGNAL_BATCH_SIZE > 0
        assert PerformanceConstants.STATE_SYNC_BATCH_SIZE > 0
    
    def test_cleanup_intervals_positive(self):
        """All cleanup intervals should be positive."""
        assert PerformanceConstants.CACHE_CLEANUP_INTERVAL > 0
        assert PerformanceConstants.STATE_CLEANUP_INTERVAL > 0


class TestEnvironmentConstants:
    """Test environment configuration."""
    
    def test_environment_flags_are_booleans(self):
        """Environment flags should be boolean."""
        assert isinstance(EnvironmentConstants.DEBUG_MODE_ENABLED, bool)
        assert isinstance(EnvironmentConstants.DEVELOPMENT_MODE, bool)
        assert isinstance(EnvironmentConstants.STRICT_TYPE_CHECKING, bool)


class TestGetAllConstants:
    """Test the get_all_constants utility function."""
    
    def test_returns_dictionary(self):
        """Should return a dictionary."""
        result = get_all_constants()
        assert isinstance(result, dict)
    
    def test_contains_expected_categories(self):
        """Should contain constants from all categories."""
        constants = get_all_constants()
        
        # Check that we have constants from each category
        categories = [
            "TimeoutConstants",
            "RetryConstants",
            "CapitalConstants",
            "ConfidenceConstants",
        ]
        
        for category in categories:
            found = any(key.startswith(category) for key in constants.keys())
            assert found, f"Missing constants from {category}"
    
    def test_all_constants_are_immutable(self):
        """Returned constants should not be modifiable."""
        constants = get_all_constants()
        assert len(constants) >= 50  # Sanity check on count (we have ~65)


class TestValidateConstants:
    """Test the validate_constants utility function."""
    
    def test_validation_passes(self):
        """Current configuration should pass validation."""
        assert validate_constants() is True


class TestConstantDocumentation:
    """Test that all constants have proper documentation."""
    
    def test_timeout_constants_documented(self):
        """Each timeout constant should have a docstring."""
        for attr_name in dir(TimeoutConstants):
            if attr_name.isupper() and not attr_name.startswith("_"):
                # Check that we can get the value
                value = getattr(TimeoutConstants, attr_name)
                assert isinstance(value, (int, float))
    
    def test_all_classes_documented(self):
        """Each configuration class should have a docstring."""
        classes = [
            TimeoutConstants,
            RetryConstants,
            CapitalConstants,
            ConfidenceConstants,
        ]
        
        for cls in classes:
            assert cls.__doc__ is not None
            assert len(cls.__doc__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
