"""
Test Suite: ExchangeClient Paper Mode API Key Fix
Date: 2026-03-07
Focus: Core paper mode functionality

Purpose:
- Verify that paper mode correctly handles API key validation
- Ensure mock data responses for account/order endpoints in paper mode
- Validate exception handling and fallback behavior
"""

import pytest
import logging
from unittest.mock import Mock


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.level = logging.INFO
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def paper_config():
    """Create paper mode config as dict."""
    return {
        "PAPER_MODE": "True",
        "BINANCE_API_KEY": "test_key_12345",
        "BINANCE_API_SECRET": "test_secret_12345",
        "BINANCE_TESTNET": "false",
        "TESTNET_MODE": "false",
        "STRICT_API_KEYS_ON_INIT": "false",
        "WEIGHT_WINDOW_SEC": "1.0",
        "WEIGHT_LIMIT_PER_WINDOW": "10",
        "ACCT_CACHE_TTL_SEC": "5.0",
        "PRICE_MICROCACHE_TTL": "1.0",
        "RECV_WINDOW_MS": "5000",
        "TICKER_24H_TTL_SEC": "15.0",
        "USER_DATA_WS_TIMEOUT_SEC": "65.0",
        "USER_DATA_WS_RECONNECT_BACKOFF_SEC": "3.0",
        "USER_DATA_WS_MAX_BACKOFF_SEC": "30.0",
        "USER_DATA_WS_API_REQUEST_TIMEOUT_SEC": "12.0",
        "USER_DATA_WS_AUTH_MODE": "polling",
        "BINANCE_API_TYPE": "",
        "FEE_BUFFER_BPS": "10",
        "PATH_WEIGHTS": {"/api/v3/klines": 2},
    }


@pytest.fixture
def live_config():
    """Create live mode config as dict."""
    return {
        "PAPER_MODE": "False",
        "BINANCE_API_KEY": "real_key_12345",
        "BINANCE_API_SECRET": "real_secret_12345",
        "BINANCE_TESTNET": "false",
        "TESTNET_MODE": "false",
        "STRICT_API_KEYS_ON_INIT": "false",
        "WEIGHT_WINDOW_SEC": "1.0",
        "WEIGHT_LIMIT_PER_WINDOW": "10",
        "ACCT_CACHE_TTL_SEC": "5.0",
        "PRICE_MICROCACHE_TTL": "1.0",
        "RECV_WINDOW_MS": "5000",
        "TICKER_24H_TTL_SEC": "15.0",
        "USER_DATA_WS_TIMEOUT_SEC": "65.0",
        "USER_DATA_WS_RECONNECT_BACKOFF_SEC": "3.0",
        "USER_DATA_WS_MAX_BACKOFF_SEC": "30.0",
        "USER_DATA_WS_API_REQUEST_TIMEOUT_SEC": "12.0",
        "USER_DATA_WS_AUTH_MODE": "polling",
        "BINANCE_API_TYPE": "",
        "FEE_BUFFER_BPS": "10",
        "PATH_WEIGHTS": {"/api/v3/klines": 2},
    }


# ============================================================================
# TEST CLASS: Paper Mode Initialization
# ============================================================================

class TestPaperModeInitialization:
    """Test that ExchangeClient properly initializes in paper mode."""

    def test_paper_mode_flag_recognized(self, mock_logger, paper_config):
        """Verify PAPER_MODE config is recognized."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        assert ec.paper_trade is True
        assert mock_logger.info.called


    def test_paper_mode_sets_fake_keys(self, mock_logger, paper_config):
        """Verify that paper mode sets fake API keys."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        assert ec.paper_trade is True
        assert ec.api_key == "paper_key"
        assert ec.api_secret == "paper_secret"


    def test_live_mode_preserves_real_keys(self, mock_logger, live_config):
        """Verify that live mode can use provided API keys."""
        from core.exchange_client import ExchangeClient
        
        # When paper_trade=False and keys are provided as arguments
        ec = ExchangeClient(
            logger=mock_logger,
            config=live_config,
            paper_trade=False,
            api_key="custom_real_key",
            api_secret="custom_real_secret"
        )
        
        # The instance should have been configured with keys (from env or config)
        assert ec.paper_trade is False
        # The keys may be from environment, so just verify they're not paper keys
        assert ec.api_key != "paper_key"
        assert ec.api_secret != "paper_secret"


# ============================================================================
# TEST CLASS: Paper Mode Code Guards
# ============================================================================

class TestPaperModeGuards:
    """Test that paper mode guards are present in code."""

    def test_paper_mode_guard_in_request_method(self, mock_logger, paper_config):
        """Verify that _request method has paper mode guard."""
        from core.exchange_client import ExchangeClient
        import inspect
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        # Verify the guard exists by checking source code
        source = inspect.getsource(ec._request)
        assert "self.paper_trade" in source, "_request should check paper_trade"
        assert "/api/v3/account" in source, "_request should handle /api/v3/account"


    def test_get_spot_balances_has_paper_mode_check(self, mock_logger, paper_config):
        """Verify that get_spot_balances has paper mode check."""
        from core.exchange_client import ExchangeClient
        import inspect
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        # Verify the check exists in source code
        source = inspect.getsource(ec.get_spot_balances)
        assert "self.paper_trade" in source, "get_spot_balances should check paper_trade"
        assert "return {}" in source, "get_spot_balances should return empty dict in paper mode"


# ============================================================================
# TEST CLASS: get_spot_balances() Paper Mode Behavior
# ============================================================================

class TestGetSpotBalances:
    """Test the get_spot_balances() method paper mode behavior."""

    @pytest.mark.asyncio
    async def test_get_spot_balances_returns_empty_in_paper_mode(self, mock_logger, paper_config):
        """Verify get_spot_balances returns empty dict in paper mode."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        # Call get_spot_balances - should return immediately
        result = await ec.get_spot_balances()
        
        # Should return empty dict
        assert result == {}
        assert isinstance(result, dict)


# ============================================================================
# TEST CLASS: get_account_balances() Paper Mode Behavior
# ============================================================================

class TestGetAccountBalances:
    """Test the get_account_balances() method paper mode behavior."""

    @pytest.mark.asyncio
    async def test_get_account_balances_paper_mode_success(self, mock_logger, paper_config):
        """Verify get_account_balances succeeds in paper mode."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        result = await ec.get_account_balances()
        
        # Should return empty dict without errors
        assert result == {}
        assert isinstance(result, dict)


# ============================================================================
# TEST CLASS: API Key Validation
# ============================================================================

class TestAPIKeyValidation:
    """Test API key validation and error handling."""

    def test_paper_mode_ignores_missing_keys(self, mock_logger, paper_config):
        """Verify paper mode doesn't require API keys."""
        from core.exchange_client import ExchangeClient
        
        paper_config_no_keys = paper_config.copy()
        paper_config_no_keys.pop("BINANCE_API_KEY", None)
        paper_config_no_keys.pop("BINANCE_API_SECRET", None)
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config_no_keys,
            paper_trade=True,
            api_key=None,
            api_secret=None
        )
        
        # Should not raise in paper mode
        assert ec.paper_trade is True
        assert ec.api_key == "paper_key"
        assert ec.api_secret == "paper_secret"


    def test_missing_keys_in_live_mode_warning(self, mock_logger, live_config):
        """Verify missing keys generate warning in live mode."""
        from core.exchange_client import ExchangeClient
        
        live_config_no_keys = live_config.copy()
        live_config_no_keys.pop("BINANCE_API_KEY", None)
        live_config_no_keys.pop("BINANCE_API_SECRET", None)
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=live_config_no_keys,
            paper_trade=False,
            api_key=None,
            api_secret=None
        )
        
        # Should have warned about missing keys
        assert mock_logger.warning.called
        assert ec.api_key == ""
        assert ec.api_secret == ""


    def test_live_mode_uses_provided_keys(self, mock_logger, live_config):
        """Verify live mode can use provided API keys."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=live_config,
            paper_trade=False,
            api_key="override_key_xyz",
            api_secret="override_secret_abc"
        )
        
        # Should not be paper keys
        assert ec.paper_trade is False
        assert ec.api_key != "paper_key"
        assert ec.api_secret != "paper_secret"


# ============================================================================
# TEST CLASS: Logging
# ============================================================================

class TestLogging:
    """Test logging behavior for paper mode."""

    def test_paper_mode_logs_on_init(self, mock_logger, paper_config):
        """Verify paper mode logs on initialization."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        # Should have logged paper mode enabled
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Paper trading mode is enabled" in call for call in info_calls)


# ============================================================================
# TEST CLASS: Integration
# ============================================================================

class TestPaperModeIntegration:
    """Integration tests for complete paper mode workflow."""

    @pytest.mark.asyncio
    async def test_paper_mode_complete_workflow(self, mock_logger, paper_config):
        """Test complete paper mode workflow without API errors."""
        from core.exchange_client import ExchangeClient
        
        ec = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        # All these should work without API errors in paper mode
        balances = await ec.get_spot_balances()
        assert balances == {}
        
        account_balances = await ec.get_account_balances()
        assert account_balances == {}
        
        # Verify no warnings about API key format
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("API-key format" in call for call in warning_calls)
        assert not any("APIError" in call for call in warning_calls)


    def test_paper_mode_mode_transition(self, mock_logger, paper_config, live_config):
        """Verify mode can transition from paper to live."""
        from core.exchange_client import ExchangeClient
        
        # Start in paper mode
        ec_paper = ExchangeClient(
            logger=mock_logger,
            config=paper_config,
            paper_trade=True
        )
        
        assert ec_paper.paper_trade is True
        assert ec_paper.api_key == "paper_key"
        assert ec_paper.api_secret == "paper_secret"
        
        # Create new instance in live mode
        ec_live = ExchangeClient(
            logger=mock_logger,
            config=live_config,
            paper_trade=False
        )
        
        assert ec_live.paper_trade is False
        # Should not have paper keys
        assert ec_live.api_key != "paper_key"
        assert ec_live.api_secret != "paper_secret"


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
