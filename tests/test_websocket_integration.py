"""
Test Suite: WebSocket Integration
Date: 2026-03-08
Focus: ExchangeClient user-data WS pipeline + WebSocketMarketData

Covers:
- _ingest_user_data_ws_payload: event extraction, routing to SharedState
- get_ws_health_snapshot: structure and gap calculations
- WS URL derivation: testnet vs mainnet (WS API v3 and listenKey streams)
- _has_signed_credentials: paper/empty/real key differentiation
- Auth mode selection & error classification (_is_user_data_ws_auth_error,
  _should_use_signature_fallback)
- mark_user_data_event / mark_any_ws_event: timestamp tracking
- start/stop_user_data_stream lifecycle
- reconnect_user_data_stream
- WebSocketMarketData: stream list, ticker/kline handling, readiness signaling
- HMAC signed-params structure (_ws_api_signed_params, _ws_api_signature_params)
- End-to-end: fill event reaches SharedState via event bus
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock
from typing import Any

import pytest


# ============================================================================
# Shared config / fixtures
# ============================================================================

_BASE_CFG = {
    "PAPER_MODE": "False",
    "BINANCE_TESTNET": "False",
    "WEIGHT_WINDOW_SEC": "1.0",
    "WEIGHT_LIMIT_PER_WINDOW": "100",
    "ACCT_CACHE_TTL_SEC": "5.0",
    "PRICE_MICROCACHE_TTL": "1.0",
    "RECV_WINDOW_MS": "5000",
    "TICKER_24H_TTL_SEC": "15.0",
    "USER_DATA_WS_TIMEOUT_SEC": "65.0",
    "USER_DATA_WS_RECONNECT_BACKOFF_SEC": "3.0",
    "USER_DATA_WS_MAX_BACKOFF_SEC": "30.0",
    "USER_DATA_WS_API_REQUEST_TIMEOUT_SEC": "12.0",
    "USER_DATA_WS_AUTH_MODE": "auto",
    "BINANCE_API_TYPE": "",
    "FEE_BUFFER_BPS": "10",
    "PATH_WEIGHTS": {"/api/v3/klines": 2},
    # Disable the order-path guard so tests can construct the client freely.
    "ENFORCE_EXECUTION_MANAGER_PATH": "False",
    "ALLOW_UNSAFE_DIRECT_ORDER_PATH": "True",
}


@pytest.fixture
def mock_logger():
    logger = Mock(spec=logging.Logger)
    logger.level = logging.INFO
    for m in ("info", "debug", "warning", "error", "critical"):
        setattr(logger, m, Mock())
    return logger


@pytest.fixture
def hmac_cfg():
    """HMAC-key client config (live, no testnet, no paper)."""
    cfg = dict(_BASE_CFG)
    cfg["BINANCE_API_KEY"] = "hmac_api_key_abc123"
    cfg["BINANCE_API_SECRET"] = "hmac_api_secret_def456"
    cfg["USER_DATA_WS_AUTH_MODE"] = "signature"
    return cfg


@pytest.fixture
def paper_cfg():
    cfg = dict(_BASE_CFG)
    cfg["PAPER_MODE"] = "True"
    cfg["USER_DATA_WS_AUTH_MODE"] = "polling"
    return cfg


@pytest.fixture
def testnet_cfg():
    cfg = dict(_BASE_CFG)
    cfg["BINANCE_TESTNET"] = "True"
    cfg["BINANCE_TESTNET_API_KEY"] = "testnet_key_xyz"
    cfg["BINANCE_TESTNET_API_SECRET"] = "testnet_secret_xyz"
    cfg["USER_DATA_WS_AUTH_MODE"] = "signature"
    return cfg


@pytest.fixture
def mock_shared_state():
    ss = AsyncMock()
    ss.emit_event = AsyncMock(return_value=None)
    return ss


def _make_ec(config, logger, shared_state=None):
    """Construct ExchangeClient with no network calls."""
    import os
    # Remove any real env vars that might interfere with key resolution.
    for key in (
        "BINANCE_API_KEY", "BINANCE_API_SECRET",
        "BINANCE_TESTNET_API_KEY", "BINANCE_TESTNET_API_SECRET",
    ):
        os.environ.pop(key, None)

    from core.exchange_client import ExchangeClient
    return ExchangeClient(config=config, logger=logger, shared_state=shared_state)


# ============================================================================
# 1. _ingest_user_data_ws_payload — extraction
# ============================================================================

class TestIngestPayloadExtraction:
    """Event-type extraction from raw WS payloads."""

    def test_returns_empty_for_non_dict(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._ingest_user_data_ws_payload("text") == ""
        assert ec._ingest_user_data_ws_payload(None) == ""
        assert ec._ingest_user_data_ws_payload([]) == ""

    def test_returns_empty_for_empty_dict(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._ingest_user_data_ws_payload({}) == ""

    def test_extracts_event_type_from_e_field(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._ingest_user_data_ws_payload({"e": "executionReport"}) == "executionReport"

    def test_extracts_event_type_from_eventType_field(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._ingest_user_data_ws_payload({"eventType": "balanceUpdate"}) == "balanceUpdate"

    def test_extracts_from_ws_api_v3_event_wrapper(self, mock_logger, hmac_cfg):
        """WS API v3 wraps events under the 'event' key."""
        ec = _make_ec(hmac_cfg, mock_logger)
        payload = {
            "subscriptionId": 1,
            "event": {"e": "outboundAccountPosition", "B": []},
        }
        assert ec._ingest_user_data_ws_payload(payload) == "outboundAccountPosition"

    def test_sets_subscription_id_from_wrapper(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec._ingest_user_data_ws_payload({
            "subscriptionId": 77,
            "event": {"e": "executionReport"},
        })
        assert ec._user_data_subscription_id == 77

    def test_malformed_subscription_id_does_not_crash(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        result = ec._ingest_user_data_ws_payload({
            "subscriptionId": "not_a_number",
            "event": {"e": "executionReport"},
        })
        assert result == "executionReport"

    def test_updates_last_user_data_event_ts(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        before = time.time()
        ec._ingest_user_data_ws_payload({"e": "executionReport"})
        assert ec.last_user_data_event_ts >= before

    def test_updates_last_any_ws_event_ts(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        before = time.time()
        ec._ingest_user_data_ws_payload({"e": "balanceUpdate"})
        assert ec.last_any_ws_event_ts >= before


# ============================================================================
# 2. _ingest_user_data_ws_payload — SharedState routing
# ============================================================================

class TestIngestPayloadRouting:
    """Critical: user-data events must reach the SharedState event bus."""

    @pytest.mark.asyncio
    async def test_execution_report_routed_to_shared_state(self, mock_logger, hmac_cfg, mock_shared_state):
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        payload = {"e": "executionReport", "s": "BTCUSDT", "X": "FILLED"}
        ec._ingest_user_data_ws_payload(payload)
        await asyncio.sleep(0)  # let the loop run the created task
        mock_shared_state.emit_event.assert_awaited_with("executionReport", payload)

    @pytest.mark.asyncio
    async def test_balance_update_routed_to_shared_state(self, mock_logger, hmac_cfg, mock_shared_state):
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        payload = {"e": "balanceUpdate", "a": "USDT", "d": "50.0"}
        ec._ingest_user_data_ws_payload(payload)
        await asyncio.sleep(0)
        mock_shared_state.emit_event.assert_awaited_with("balanceUpdate", payload)

    @pytest.mark.asyncio
    async def test_outbound_account_position_routed(self, mock_logger, hmac_cfg, mock_shared_state):
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        payload = {"e": "outboundAccountPosition", "B": []}
        ec._ingest_user_data_ws_payload(payload)
        await asyncio.sleep(0)
        mock_shared_state.emit_event.assert_awaited_with("outboundAccountPosition", payload)

    @pytest.mark.asyncio
    async def test_listen_key_expired_routed(self, mock_logger, hmac_cfg, mock_shared_state):
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        payload = {"e": "listenKeyExpired"}
        ec._ingest_user_data_ws_payload(payload)
        await asyncio.sleep(0)
        mock_shared_state.emit_event.assert_awaited_with("listenKeyExpired", payload)

    @pytest.mark.asyncio
    async def test_non_user_data_events_not_routed(self, mock_logger, hmac_cfg, mock_shared_state):
        """Session status events and UNKNOWN should not go to SharedState."""
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        ec._ingest_user_data_ws_payload({"e": "UNKNOWN"})
        await asyncio.sleep(0)
        mock_shared_state.emit_event.assert_not_awaited()

    def test_no_crash_when_shared_state_is_none(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=None)
        result = ec._ingest_user_data_ws_payload({"e": "executionReport"})
        assert result == "executionReport"

    @pytest.mark.asyncio
    async def test_ws_api_v3_wrapped_fill_is_routed(self, mock_logger, hmac_cfg, mock_shared_state):
        """A WS API v3 wrapped executionReport must also reach SharedState."""
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        payload = {
            "subscriptionId": 5,
            "event": {
                "e": "executionReport",
                "s": "ETHUSDT",
                "X": "FILLED",
                "z": "1.0",
            },
        }
        ec._ingest_user_data_ws_payload(payload)
        await asyncio.sleep(0)
        # The inner event dict is routed (not the outer wrapper)
        mock_shared_state.emit_event.assert_awaited()
        call_args = mock_shared_state.emit_event.call_args
        assert call_args[0][0] == "executionReport"


# ============================================================================
# 3. WS Health Snapshot
# ============================================================================

class TestWsHealthSnapshot:

    def test_all_required_keys_present(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        snap = ec.get_ws_health_snapshot()
        required = {
            "user_data_stream_enabled", "ws_connected", "ws_reconnect_count",
            "user_data_ws_auth_mode", "user_data_subscription_id",
            "last_user_data_event_ts", "last_any_ws_event_ts",
            "last_listenkey_refresh_ts", "last_successful_force_sync_ts",
            "user_data_gap_sec", "any_ws_gap_sec",
            "listenkey_refresh_gap_sec", "force_sync_gap_sec",
        }
        assert required.issubset(snap.keys())

    def test_gap_sec_is_minus_one_when_no_events(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.last_user_data_event_ts = 0.0
        ec.last_any_ws_event_ts = 0.0
        snap = ec.get_ws_health_snapshot()
        assert snap["user_data_gap_sec"] == -1.0
        assert snap["any_ws_gap_sec"] == -1.0

    def test_gap_sec_positive_after_event(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        past = time.time() - 10.0
        ec.last_user_data_event_ts = past
        ec.last_any_ws_event_ts = past
        snap = ec.get_ws_health_snapshot()
        assert snap["user_data_gap_sec"] >= 9.0
        assert snap["any_ws_gap_sec"] >= 9.0

    def test_ws_connected_reflects_attribute(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.ws_connected = True
        assert ec.get_ws_health_snapshot()["ws_connected"] is True
        ec.ws_connected = False
        assert ec.get_ws_health_snapshot()["ws_connected"] is False

    def test_reconnect_count_reflected(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.ws_reconnect_count = 7
        assert ec.get_ws_health_snapshot()["ws_reconnect_count"] == 7


# ============================================================================
# 4. WS URL Derivation
# ============================================================================

class TestWsUrlDerivation:

    def test_ws_api_url_mainnet(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.base_url_spot_api = "https://api.binance.com"
        url = ec._user_data_ws_api_url()
        assert "ws-api.binance.com" in url
        assert "/ws-api/v3" in url
        assert url.startswith("wss://")

    def test_ws_api_url_testnet(self, mock_logger, testnet_cfg):
        ec = _make_ec(testnet_cfg, mock_logger)
        ec.base_url_spot_api = "https://testnet.binance.vision"
        url = ec._user_data_ws_api_url()
        assert "testnet.binance.vision" in url
        assert "/ws-api/v3" in url

    def test_ws_stream_url_mainnet(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.base_url_spot_api = "https://api.binance.com"
        url = ec._user_data_ws_stream_url("listen_key_abc")
        assert "stream.binance.com" in url
        assert "listen_key_abc" in url
        assert url.startswith("wss://")

    def test_ws_stream_url_testnet(self, mock_logger, testnet_cfg):
        ec = _make_ec(testnet_cfg, mock_logger)
        ec.base_url_spot_api = "https://testnet.binance.vision"
        url = ec._user_data_ws_stream_url("testnet_lk")
        assert "testnet" in url
        assert "testnet_lk" in url

    def test_ws_stream_url_empty_for_missing_key(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._user_data_ws_stream_url("") == ""
        assert ec._user_data_ws_stream_url(None) == ""


# ============================================================================
# 5. _has_signed_credentials
# ============================================================================

class TestHasSignedCredentials:

    def test_false_for_paper_sentinel_keys(self, mock_logger, paper_cfg):
        ec = _make_ec(paper_cfg, mock_logger)
        assert ec._has_signed_credentials() is False

    def test_false_when_key_empty(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = ""
        ec.api_secret = "some_secret"
        assert ec._has_signed_credentials() is False

    def test_false_when_secret_empty(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "some_key"
        ec.api_secret = ""
        assert ec._has_signed_credentials() is False

    def test_true_for_real_key_pair(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "real_key"
        ec.api_secret = "real_secret"
        assert ec._has_signed_credentials() is True


# ============================================================================
# 6. Auth mode selection and error classification
# ============================================================================

class TestAuthModeAndErrorClassification:

    def test_hmac_key_auto_selects_signature_mode(self, mock_logger, hmac_cfg):
        cfg = dict(hmac_cfg)
        cfg["USER_DATA_WS_AUTH_MODE"] = "auto"
        ec = _make_ec(cfg, mock_logger)
        assert ec._api_key_type == "HMAC"
        assert ec.user_data_ws_auth_mode == "signature"

    def test_ed25519_explicit_selects_session_mode(self, mock_logger, hmac_cfg):
        cfg = dict(hmac_cfg)
        cfg["BINANCE_API_TYPE"] = "ED25519"
        cfg["USER_DATA_WS_AUTH_MODE"] = "auto"
        ec = _make_ec(cfg, mock_logger)
        assert ec._api_key_type == "ED25519"
        assert ec.user_data_ws_auth_mode == "session"

    def test_is_auth_error_on_code_minus_2015(self, mock_logger, hmac_cfg):
        from core.exchange_client import BinanceAPIException
        ec = _make_ec(hmac_cfg, mock_logger)
        err = BinanceAPIException("Unauthorized", code=-2015)
        assert ec._is_user_data_ws_auth_error(err) is True

    def test_is_auth_error_on_invalid_api_key_text(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._is_user_data_ws_auth_error(RuntimeError("Invalid API-key format")) is True

    def test_is_auth_error_on_signature_invalid_text(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        err = RuntimeError("Signature for this request is not valid")
        assert ec._is_user_data_ws_auth_error(err) is True

    def test_not_auth_error_on_timeout(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._is_user_data_ws_auth_error(TimeoutError("timeout")) is False

    def test_should_use_signature_fallback_on_auth_error(self, mock_logger, hmac_cfg):
        from core.exchange_client import BinanceAPIException
        ec = _make_ec(hmac_cfg, mock_logger)
        err = BinanceAPIException("Invalid API-key", code=-2015)
        assert ec._should_use_signature_fallback(err) is True

    def test_should_not_fallback_from_session_mode_on_non_auth_error(self, mock_logger, hmac_cfg):
        cfg = dict(hmac_cfg)
        cfg["USER_DATA_WS_AUTH_MODE"] = "session"
        ec = _make_ec(cfg, mock_logger)
        assert ec._should_use_signature_fallback(RuntimeError("timeout")) is False

    def test_signature_mode_always_fallbacks(self, mock_logger, hmac_cfg):
        cfg = dict(hmac_cfg)
        cfg["USER_DATA_WS_AUTH_MODE"] = "signature"
        ec = _make_ec(cfg, mock_logger)
        # In signature mode, _should_use_signature_fallback always returns True
        assert ec._should_use_signature_fallback(RuntimeError("any error")) is True


# ============================================================================
# 7. mark_user_data_event / mark_any_ws_event
# ============================================================================

class TestWsEventMarkers:

    def test_mark_user_data_event_updates_both_timestamps(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.last_user_data_event_ts = 0.0
        ec.last_any_ws_event_ts = 0.0
        before = time.time()
        ts = ec.mark_user_data_event("executionReport")
        assert ts >= before
        assert ec.last_user_data_event_ts >= before
        assert ec.last_any_ws_event_ts >= before

    def test_mark_any_ws_event_does_not_update_user_data_ts(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.last_user_data_event_ts = 0.0
        ec.last_any_ws_event_ts = 0.0
        before = time.time()
        ts = ec.mark_any_ws_event("ping")
        assert ts >= before
        assert ec.last_any_ws_event_ts >= before
        # user_data_event_ts must NOT be touched by mark_any_ws_event
        assert ec.last_user_data_event_ts == 0.0

    def test_record_successful_force_sync(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.last_successful_force_sync_ts = 0.0
        before = time.time()
        ts = ec.record_successful_force_sync(reason="test")
        assert ts >= before
        assert ec.last_successful_force_sync_ts >= before

    def test_mark_user_data_event_returns_timestamp(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ts = ec.mark_user_data_event("balanceUpdate")
        assert isinstance(ts, float)
        assert ts > 0

    def test_mark_any_ws_event_returns_timestamp(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ts = ec.mark_any_ws_event("pong")
        assert isinstance(ts, float)
        assert ts > 0


# ============================================================================
# 8. start/stop/reconnect lifecycle
# ============================================================================

class TestUserDataStreamLifecycle:

    @pytest.mark.asyncio
    async def test_start_returns_false_when_not_started(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        # No session/client → is_started is False → should not start
        result = await ec.start_user_data_stream()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_returns_false_when_stream_disabled(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.user_data_stream_enabled = False
        result = await ec.start_user_data_stream()
        assert result is False

    @pytest.mark.asyncio
    async def test_start_returns_false_for_paper_mode(self, mock_logger, paper_cfg):
        ec = _make_ec(paper_cfg, mock_logger)
        result = await ec.start_user_data_stream()
        assert result is False  # paper mode → no signed credentials

    @pytest.mark.asyncio
    async def test_start_returns_true_when_task_already_running(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        # Simulate a running task and a valid session

        async def _long_running():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long_running())
        ec._user_data_ws_task = task
        # Also fake is_started so the guard passes
        ec.session = Mock()
        ec.client = Mock()
        ec.api_key = "real_key"
        ec.api_secret = "real_secret"
        result = await ec.start_user_data_stream()
        assert result is True
        task.cancel()
        # Suppress CancelledError — we're only checking `result`
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_stop_cancels_ws_task(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)

        async def _dummy():
            await asyncio.sleep(100)

        task = asyncio.create_task(_dummy())
        ec._user_data_ws_task = task

        await ec.stop_user_data_stream()

        assert task.cancelled()
        assert ec._user_data_ws_task is None
        assert ec.ws_connected is False

    @pytest.mark.asyncio
    async def test_stop_is_idempotent_when_no_task(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec._user_data_ws_task = None
        # Must not raise
        await ec.stop_user_data_stream()

    @pytest.mark.asyncio
    async def test_reconnect_increments_reconnect_count(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.ws_reconnect_count = 3
        # Keep _has_signed_credentials True so reconnect proceeds past the early-exit guard.
        # Stub start_user_data_stream to prevent actual connection attempts.
        ec._has_signed_credentials = Mock(return_value=True)
        ec.start_user_data_stream = AsyncMock(return_value=False)
        await ec.reconnect_user_data_stream("unit_test")
        assert ec.ws_reconnect_count >= 4


# ============================================================================
# 9. HMAC signed-params correctness
# ============================================================================

class TestWsSignedParams:

    def test_ws_api_signed_params_required_fields(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "key123"
        ec.api_secret = "secret456"
        params = ec._ws_api_signed_params()
        assert "apiKey" in params
        assert "timestamp" in params
        assert "signature" in params

    def test_signature_is_64_char_hex(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "key123"
        ec.api_secret = "secret456"
        sig = ec._ws_api_signed_params()["signature"]
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_ws_api_signature_params_includes_recv_window(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "key123"
        ec.api_secret = "secret456"
        params = ec._ws_api_signature_params()
        assert "recvWindow" in params
        assert params["recvWindow"] == ec.recv_window_ms

    def test_signature_changes_with_different_secrets(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec.api_key = "same_key"
        ec.api_secret = "secret_A"
        sig_a = ec._ws_api_signed_params()["signature"]

        ec.api_secret = "secret_B"
        sig_b = ec._ws_api_signed_params()["signature"]
        assert sig_a != sig_b


# ============================================================================
# 10. WebSocketMarketData — stream list and message handling
# ============================================================================

@pytest.fixture
def ws_shared_state():
    # Use a plain object so hasattr(ss, "_lock_context") returns False,
    # exercising the non-locking code path in _update_shared_state_kline.
    class _SS:
        pass

    ss = _SS()
    ss.prices = {}
    ss.market_data = {}
    ss.market_data_ready_event = asyncio.Event()
    ss.update_component_health = Mock()
    return ss


@pytest.fixture
def ws_exchange_client():
    ec = Mock()
    ec.client = None
    ec.mark_any_ws_event = Mock(return_value=time.time())
    return ec


class TestWebSocketMarketDataStreamList:

    def test_stream_list_contains_ticker_and_klines(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(
            ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m", "5m"]
        )
        ws._symbols_subscribed = {"BTCUSDT", "ETHUSDT"}
        streams = ws._build_stream_list()
        assert "btcusdt@ticker" in streams
        assert "ethusdt@ticker" in streams
        assert "btcusdt@kline_1m" in streams
        assert "btcusdt@kline_5m" in streams
        assert "ethusdt@kline_5m" in streams

    def test_stream_list_empty_with_no_symbols(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        ws._symbols_subscribed = set()
        assert ws._build_stream_list() == []

    def test_stream_list_uses_lowercase_symbols(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m"])
        ws._symbols_subscribed = {"SOLUSDT"}
        streams = ws._build_stream_list()
        assert all(s == s.lower() for s in streams)


class TestWebSocketMarketDataSubscribe:

    @pytest.mark.asyncio
    async def test_subscribe_adds_symbols(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws.subscribe(["BTCUSDT", "ETHUSDT"])
        assert "BTCUSDT" in ws._symbols_subscribed
        assert "ETHUSDT" in ws._symbols_subscribed

    @pytest.mark.asyncio
    async def test_subscribe_deduplicates(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws.subscribe(["BTCUSDT"])
        await ws.subscribe(["BTCUSDT", "BTCUSDT"])
        assert len(ws._symbols_subscribed) == 1

    @pytest.mark.asyncio
    async def test_subscribe_normalizes_to_uppercase(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws.subscribe(["btcusdt"])
        assert "BTCUSDT" in ws._symbols_subscribed


class TestWebSocketMarketDataTickerHandling:

    @pytest.mark.asyncio
    async def test_ticker_updates_price_buffer_and_shared_state(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws._handle_ticker_message({"e": "24hrTicker", "s": "BTCUSDT", "c": "45000.00"})
        assert ws._price_buffer.get("BTCUSDT") == 45000.0
        assert ws_shared_state.prices.get("BTCUSDT") == 45000.0

    @pytest.mark.asyncio
    async def test_ticker_ignores_missing_price(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws._handle_ticker_message({"e": "24hrTicker", "s": "BTCUSDT"})
        assert "BTCUSDT" not in ws._price_buffer

    @pytest.mark.asyncio
    async def test_ticker_ignores_invalid_price(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        await ws._handle_ticker_message({"e": "24hrTicker", "s": "BTCUSDT", "c": "not_a_price"})
        assert "BTCUSDT" not in ws._price_buffer


class TestWebSocketMarketDataKlineHandling:

    @pytest.mark.asyncio
    async def test_open_kline_buffered_not_committed(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m"])
        msg = {
            "e": "kline", "s": "BTCUSDT",
            "k": {
                "t": 1700000000000, "i": "1m",
                "o": "45000", "h": "45100", "l": "44900", "c": "45050", "v": "10",
                "x": False,  # not closed
            },
        }
        await ws._handle_kline_message(msg)
        assert ("BTCUSDT", "1m") in ws._kline_buffer
        assert ws_shared_state.market_data == {}

    @pytest.mark.asyncio
    async def test_closed_kline_updates_market_data(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m"])
        msg = {
            "e": "kline", "s": "BTCUSDT",
            "k": {
                "t": 1700000000000, "i": "1m",
                "o": "45000", "h": "45100", "l": "44900", "c": "45050", "v": "10.5",
                "x": True,  # closed
            },
        }
        await ws._handle_kline_message(msg)
        key = ("BTCUSDT", "1m")
        assert key in ws_shared_state.market_data
        bars = ws_shared_state.market_data[key]
        assert len(bars) == 1
        assert bars[0][1] == 45000.0  # open
        assert bars[0][2] == 45100.0  # high
        assert bars[0][4] == 45050.0  # close
        assert bars[0][5] == 10.5    # volume

    @pytest.mark.asyncio
    async def test_kline_ignores_unsubscribed_timeframe(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m"])
        msg = {
            "e": "kline", "s": "BTCUSDT",
            "k": {"t": 1700000000000, "i": "4h", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "x": True},
        }
        await ws._handle_kline_message(msg)
        assert ws_shared_state.market_data == {}

    @pytest.mark.asyncio
    async def test_kline_does_not_duplicate_same_ts(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client, ohlcv_timeframes=["1m"])
        msg = {
            "e": "kline", "s": "BTCUSDT",
            "k": {"t": 1700000000000, "i": "1m", "o": "1", "h": "1", "l": "1", "c": "1", "v": "1", "x": True},
        }
        await ws._handle_kline_message(msg)
        await ws._handle_kline_message(msg)  # duplicate
        assert len(ws_shared_state.market_data[("BTCUSDT", "1m")]) == 1


class TestWebSocketMarketDataReadiness:

    @pytest.mark.asyncio
    async def test_ready_event_set_when_enough_bars(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(
            ws_shared_state, ws_exchange_client,
            ohlcv_timeframes=["1m"], readiness_min_bars=2,
        )
        ws._symbols_subscribed = {"BTCUSDT"}
        ws_shared_state.market_data = {
            ("BTCUSDT", "1m"): [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]],
        }
        await ws._maybe_set_ready()
        assert ws_shared_state.market_data_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_ready_event_not_set_with_insufficient_bars(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(
            ws_shared_state, ws_exchange_client,
            ohlcv_timeframes=["1m"], readiness_min_bars=50,
        )
        ws._symbols_subscribed = {"BTCUSDT"}
        ws_shared_state.market_data = {("BTCUSDT", "1m"): [[1, 1, 1, 1, 1, 1]]}
        await ws._maybe_set_ready()
        assert not ws_shared_state.market_data_ready_event.is_set()

    @pytest.mark.asyncio
    async def test_ready_event_not_set_if_symbol_missing(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(
            ws_shared_state, ws_exchange_client,
            ohlcv_timeframes=["1m"], readiness_min_bars=1,
        )
        ws._symbols_subscribed = {"BTCUSDT", "ETHUSDT"}
        ws_shared_state.market_data = {("BTCUSDT", "1m"): [[1, 1, 1, 1, 1, 1]]}
        # ETHUSDT missing → not ready
        await ws._maybe_set_ready()
        assert not ws_shared_state.market_data_ready_event.is_set()


class TestWebSocketMarketDataMessageRouter:

    @pytest.mark.asyncio
    async def test_handle_message_routes_ticker(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        handled = []

        async def fake_ticker(msg):
            handled.append(("ticker", msg))

        ws._handle_ticker_message = fake_ticker
        await ws._handle_message({"e": "24hrTicker", "s": "X", "c": "1"})
        assert len(handled) == 1 and handled[0][0] == "ticker"

    @pytest.mark.asyncio
    async def test_handle_message_routes_kline(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        handled = []

        async def fake_kline(msg):
            handled.append(("kline", msg))

        ws._handle_kline_message = fake_kline
        await ws._handle_message({"e": "kline", "s": "X", "k": {}})
        assert len(handled) == 1 and handled[0][0] == "kline"

    @pytest.mark.asyncio
    async def test_handle_message_ignores_unknown_event(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        # Should not raise
        await ws._handle_message({"e": "someUnknownEvent"})

    @pytest.mark.asyncio
    async def test_handle_message_ignores_no_event_key(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        # Must not raise
        await ws._handle_message({"data": "something"})


class TestWebSocketMarketDataStopLifecycle:

    @pytest.mark.asyncio
    async def test_stop_sets_running_false_and_stop_event(self, ws_shared_state, ws_exchange_client):
        from core.ws_market_data import WebSocketMarketData
        ws = WebSocketMarketData(ws_shared_state, ws_exchange_client)
        ws._running = True
        await ws.stop()
        assert ws._running is False
        assert ws._stop.is_set()


# ============================================================================
# 11. End-to-end: polling loop emits fills → SharedState receives them
# ============================================================================

class TestPollingFillRouting:
    """Validate that the polling reconciliation loop routes fills to SharedState."""

    @pytest.mark.asyncio
    async def test_ingest_payload_called_in_polling_context(self, mock_logger, hmac_cfg, mock_shared_state):
        """
        Simulate what the polling loop does: build an executionReport dict and
        call _ingest_user_data_ws_payload, then verify SharedState is notified.
        """
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)
        now = time.time()

        fill_event = {
            "e": "executionReport",
            "E": int(now * 1000),
            "s": "SOLUSDT",
            "c": "octi-test-order",
            "S": "BUY",
            "o": "MARKET",
            "x": "TRADE",
            "X": "FILLED",
            "i": 123456,
            "l": 10.0,
            "z": 10.0,
            "L": 25.00,
        }

        ec._ingest_user_data_ws_payload(fill_event)
        await asyncio.sleep(0)  # let the event loop run the task

        mock_shared_state.emit_event.assert_awaited()
        args = mock_shared_state.emit_event.call_args[0]
        assert args[0] == "executionReport"
        assert args[1]["s"] == "SOLUSDT"

    @pytest.mark.asyncio
    async def test_multiple_events_all_routed(self, mock_logger, hmac_cfg, mock_shared_state):
        """Multiple sequential WS events are all routed."""
        ec = _make_ec(hmac_cfg, mock_logger, shared_state=mock_shared_state)

        events = [
            {"e": "executionReport", "s": "BTCUSDT", "X": "FILLED"},
            {"e": "balanceUpdate", "a": "USDT", "d": "100.0"},
            {"e": "outboundAccountPosition", "B": []},
        ]

        for ev in events:
            ec._ingest_user_data_ws_payload(ev)

        await asyncio.sleep(0)
        assert mock_shared_state.emit_event.await_count == len(events)


# ============================================================================
# 12. Tier unavailability flags (1008 policy-close / 410 Gone)
# ============================================================================

class TestTierUnavailabilityFlags:
    """
    Verify that permanent environment failures (WS API v3 policy-close 1008,
    listenKey 410 Gone) are recorded as flags so the supervisor skips those
    tiers on every subsequent reconnect.
    """

    def test_initial_flags_are_false(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        assert ec._ws_v3_unavailable is False
        assert ec._listen_key_unavailable is False

    def test_health_snapshot_exposes_flags(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        snap = ec.get_ws_health_snapshot()
        assert "ws_v3_unavailable" in snap
        assert "listen_key_unavailable" in snap
        assert snap["ws_v3_unavailable"] is False
        assert snap["listen_key_unavailable"] is False

    def test_health_snapshot_reflects_flag_when_set(self, mock_logger, hmac_cfg):
        ec = _make_ec(hmac_cfg, mock_logger)
        ec._ws_v3_unavailable = True
        ec._listen_key_unavailable = True
        snap = ec.get_ws_health_snapshot()
        assert snap["ws_v3_unavailable"] is True
        assert snap["listen_key_unavailable"] is True

    @pytest.mark.asyncio
    async def test_supervisor_sets_ws_v3_flag_on_policy_close_1008(self, mock_logger, hmac_cfg):
        """1008 policy-close → _ws_v3_unavailable=True and tier is skipped thereafter."""
        ec = _make_ec(hmac_cfg, mock_logger)
        # Force real keys so _has_signed_credentials() → True and supervisor enters Tier 1
        ec.api_key = "real_key_for_test"
        ec.api_secret = "real_secret_for_test"

        call_count = {"v3": 0, "lk": 0, "poll": 0}

        async def fake_ws_v3():
            call_count["v3"] += 1
            raise RuntimeError("USER_DATA_WS_CLOSED ws_code=1008 reason='policy violation'")

        async def fake_listen_key():
            call_count["lk"] += 1
            # Signal stop so the supervisor exits after one iteration
            ec._user_data_stop.set()
            raise RuntimeError("listenKey error")

        async def fake_polling():
            call_count["poll"] += 1
            ec._user_data_stop.set()

        ec._user_data_ws_api_v3_direct = fake_ws_v3
        ec._user_data_listen_key_loop = fake_listen_key
        ec._user_data_polling_loop = fake_polling
        ec.session = Mock()
        ec.client = Mock()

        ec._user_data_stop.clear()
        await ec._user_data_ws_loop()

        # Flag must be set
        assert ec._ws_v3_unavailable is True
        # v3 called once (to discover it's unavailable), not retried
        assert call_count["v3"] == 1

    @pytest.mark.asyncio
    async def test_supervisor_sets_listen_key_flag_on_410_gone(self, mock_logger, hmac_cfg):
        """listenKey 410 Gone → _listen_key_unavailable=True and tier is skipped thereafter."""
        ec = _make_ec(hmac_cfg, mock_logger)
        # Force real keys so _has_signed_credentials() → True
        ec.api_key = "real_key_for_test"
        ec.api_secret = "real_secret_for_test"
        # Tier 1 is already known unavailable so the test focuses on Tier 2
        ec._ws_v3_unavailable = True

        call_count = {"lk": 0, "poll": 0}

        async def fake_listen_key():
            call_count["lk"] += 1
            raise RuntimeError(
                "Failed to create listenKey - likely HTTP 410 Gone "
                "(account doesn't support user-data streams)"
            )

        async def fake_polling():
            call_count["poll"] += 1
            ec._user_data_stop.set()

        ec._user_data_listen_key_loop = fake_listen_key
        ec._user_data_polling_loop = fake_polling
        ec.session = Mock()
        ec.client = Mock()

        ec._user_data_stop.clear()
        await ec._user_data_ws_loop()

        assert ec._listen_key_unavailable is True
        assert call_count["lk"] == 1  # tried once, then flagged
        assert call_count["poll"] >= 1  # polling took over

    @pytest.mark.asyncio
    async def test_supervisor_skips_flagged_tiers_immediately(self, mock_logger, hmac_cfg):
        """When both flags are set, the supervisor goes straight to polling."""
        ec = _make_ec(hmac_cfg, mock_logger)
        ec._ws_v3_unavailable = True
        ec._listen_key_unavailable = True

        call_count = {"v3": 0, "lk": 0, "poll": 0}

        async def fake_ws_v3():
            call_count["v3"] += 1

        async def fake_listen_key():
            call_count["lk"] += 1

        async def fake_polling():
            call_count["poll"] += 1
            ec._user_data_stop.set()

        ec._user_data_ws_api_v3_direct = fake_ws_v3
        ec._user_data_listen_key_loop = fake_listen_key
        ec._user_data_polling_loop = fake_polling
        ec.session = Mock()
        ec.client = Mock()

        ec._user_data_stop.clear()
        await ec._user_data_ws_loop()

        # Neither WS tier should have been called
        assert call_count["v3"] == 0
        assert call_count["lk"] == 0
        assert call_count["poll"] >= 1

    @pytest.mark.asyncio
    async def test_transient_v3_error_does_not_set_flag(self, mock_logger, hmac_cfg):
        """A non-1008 error (e.g., network timeout) must not set the flag."""
        ec = _make_ec(hmac_cfg, mock_logger)

        async def fake_ws_v3():
            raise RuntimeError("Connection timed out")

        async def fake_listen_key():
            ec._user_data_stop.set()
            raise RuntimeError("listenKey create failed")

        async def fake_polling():
            ec._user_data_stop.set()

        ec._user_data_ws_api_v3_direct = fake_ws_v3
        ec._user_data_listen_key_loop = fake_listen_key
        ec._user_data_polling_loop = fake_polling
        ec.session = Mock()
        ec.client = Mock()

        ec._user_data_stop.clear()
        await ec._user_data_ws_loop()

        # Transient error → flag must remain False (tier retried next reconnect)
        assert ec._ws_v3_unavailable is False

    @pytest.mark.asyncio
    async def test_transient_listen_key_error_does_not_set_flag(self, mock_logger, hmac_cfg):
        """A non-410 listenKey error must not set the permanent flag."""
        ec = _make_ec(hmac_cfg, mock_logger)
        ec._ws_v3_unavailable = True  # skip Tier 1

        async def fake_listen_key():
            raise RuntimeError("Connection reset by peer")

        async def fake_polling():
            ec._user_data_stop.set()

        ec._user_data_listen_key_loop = fake_listen_key
        ec._user_data_polling_loop = fake_polling
        ec.session = Mock()
        ec.client = Mock()

        ec._user_data_stop.clear()
        await ec._user_data_ws_loop()

        assert ec._listen_key_unavailable is False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
