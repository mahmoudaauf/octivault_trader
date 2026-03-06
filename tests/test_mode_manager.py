import pytest
import asyncio
import time
from unittest.mock import Mock
from core.mode_manager import ModeManager


def test_mode_manager_basic():
    """Basic test for ModeManager initialization."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10

    mock_logger = Mock()

    mode_manager = ModeManager(mock_logger, mock_config)

    # Test basic functionality
    assert mode_manager.get_mode() == "BOOTSTRAP"
    assert mode_manager.get_active_symbol_limit() == 1

    # Test mode switching
    mode_manager.set_mode("NORMAL")
    assert mode_manager.get_mode() == "NORMAL"
    assert mode_manager.get_active_symbol_limit() == 3

    # Test envelope
    envelope = mode_manager.get_envelope()
    assert envelope["max_trade_usdt"] == 150.0
    assert envelope["max_positions"] == 3
    assert envelope["confidence_floor"] == 0.65

    print("✅ Basic ModeManager tests passed!")


def test_mode_envelopes():
    """Test that each mode has correct envelope constraints."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    test_cases = [
        ("SAFE", {"max_trade_usdt": 30.0, "max_positions": 1, "confidence_floor": 0.85}),
        ("PROTECTIVE", {"max_trade_usdt": 0.0, "max_positions": 0, "confidence_floor": 0.95}),
        ("NORMAL", {"max_trade_usdt": 150.0, "max_positions": 3, "confidence_floor": 0.65}),
        ("AGGRESSIVE", {"max_trade_usdt": 300.0, "max_positions": 5, "confidence_floor": 0.55}),
        ("RECOVERY", {"max_trade_usdt": 50.0, "max_positions": 1, "confidence_floor": 0.80}),
        ("BOOTSTRAP", {"max_trade_usdt": 20.0, "max_positions": 5, "confidence_floor": 0.70}),
        ("PAUSED", {"max_trade_usdt": 0.0, "max_positions": 0, "confidence_floor": 1.0}),
        ("SIGNAL_ONLY", {"max_trade_usdt": 1000.0, "max_positions": 10, "confidence_floor": 0.40}),
    ]

    for mode, expected_envelope in test_cases:
        mode_manager.set_mode(mode)
        envelope = mode_manager.get_envelope()
        for key, expected_value in expected_envelope.items():
            assert envelope[key] == expected_value, f"Mode {mode} envelope {key} mismatch"


def test_invalid_mode_rejection():
    """Test that invalid modes are rejected."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    original_mode = mode_manager.get_mode()
    mode_manager.set_mode("INVALID_MODE")
    assert mode_manager.get_mode() == original_mode
    mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_to_normal_transition():
    """Test automatic transition from BOOTSTRAP to NORMAL."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    # Start in bootstrap
    assert mode_manager.get_mode() == "BOOTSTRAP"

    # Simulate first trade executed
    metrics = {"first_trade_executed": True}
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "NORMAL"


@pytest.mark.asyncio
async def test_bootstrap_flat_portfolio_restart():
    """Test that flat portfolio on restart triggers BOOTSTRAP."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    # Set to NORMAL first
    mode_manager.set_mode("NORMAL")

    # Simulate restart with flat portfolio
    metrics = {
        "has_positions": False,
        "is_restart": True,
        "idle_time_sec": 0
    }
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "BOOTSTRAP"


@pytest.mark.asyncio
async def test_circuit_breaker_safe_mode():
    """Test circuit breaker triggers SAFE mode."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    mode_manager.set_mode("NORMAL")

    metrics = {"circuit_breaker_open": True}
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "SAFE"


@pytest.mark.asyncio
async def test_manual_pause():
    """Test manual pause triggers PAUSED mode."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    mode_manager.set_mode("NORMAL")

    metrics = {"manual_pause": True}
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "PAUSED"


@pytest.mark.asyncio
async def test_drawdown_protective_mode():
    """Test drawdown triggers PROTECTIVE mode."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    mode_manager.set_mode("NORMAL")

    metrics = {"drawdown_pct": 2.5}  # > 2.0 threshold
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "PROTECTIVE"


@pytest.mark.asyncio
async def test_hard_drawdown_safe_mode():
    """Test hard drawdown triggers SAFE mode."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    mode_manager.set_mode("PROTECTIVE")  # Start from PROTECTIVE to test hard drawdown

    metrics = {"drawdown_pct": 6.0}  # > 5.0 threshold
    await mode_manager.evaluate_state_machine(metrics)

    assert mode_manager.get_mode() == "SAFE"


def test_mandatory_sell_mode():
    """Test mandatory sell mode functionality."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    assert not mode_manager.is_mandatory_sell_mode_active()

    mode_manager.set_mandatory_sell_mode(True)
    assert mode_manager.is_mandatory_sell_mode_active()

    mode_manager.set_mandatory_sell_mode(False)
    assert not mode_manager.is_mandatory_sell_mode_active()


def test_mode_info_structure():
    """Test that get_mode_info returns correct structure."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    info = mode_manager.get_mode_info()

    required_keys = [
        "current_mode", "last_mode", "mode_switch_count",
        "mode_switch_timestamps", "active_symbol_limit", "envelope"
    ]

    for key in required_keys:
        assert key in info

    assert info["current_mode"] == "BOOTSTRAP"
    assert info["active_symbol_limit"] == 1


def test_event_handlers():
    """Test event handler registration and emission."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    event_calls = []

    def test_handler(event_type, payload):
        event_calls.append((event_type, payload))

    mode_manager.register_event_handler(test_handler)

    # Trigger a mode change
    mode_manager.set_mode("NORMAL")

    assert len(event_calls) == 1
    event_type, payload = event_calls[0]
    assert event_type == "mode_changed"
    assert payload["old_mode"] == "BOOTSTRAP"
    assert payload["new_mode"] == "NORMAL"


def test_condition_persistence_tracking():
    """Test condition persistence timing logic."""
    mock_config = Mock()
    mock_config.MIN_MODE_DURATION_SEC = 30.0
    mock_config.PROLONGED_IDLE_SECONDS = 3600
    mock_config.PROTECTIVE_DRAWDOWN_LIMIT = 2.0
    mock_config.HARD_DRAWDOWN_LIMIT = 5.0
    mock_config.RECOVERY_STABILIZATION_MIN = 10
    mock_logger = Mock()
    mode_manager = ModeManager(mock_logger, mock_config)

    now = time.time()

    # First check - should start timer
    assert not mode_manager._check_condition_persistence("test_condition", now, 60)
    assert "test_condition" in mode_manager._condition_start_times

    # Check again before duration - should return False
    assert not mode_manager._check_condition_persistence("test_condition", now + 30, 60)

    # Check after duration - should return True
    assert mode_manager._check_condition_persistence("test_condition", now + 70, 60)

    # Reset condition
    mode_manager._reset_condition("test_condition")
    assert "test_condition" not in mode_manager._condition_start_times