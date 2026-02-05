import pytest
from core.shared_state import SharedState

def test_initial_balances():
    state = SharedState()
    assert isinstance(state.balances, dict)
    assert state.balances["USDT"] == 0.0
