import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.shared_state import SharedState, SharedStateConfig


class _FakeExchangeClient:
    def __init__(self):
        self._balances = {
            "USDT": {"free": 250.0, "locked": 0.0},
            "BTC": {"free": 1.0, "locked": 0.0},
        }

    async def get_spot_balances(self):
        return dict(self._balances)

    def has_symbol(self, symbol: str) -> bool:
        return symbol.upper() == "BTCUSDT"

    async def get_current_price(self, symbol: str):
        if symbol.upper() == "BTCUSDT":
            return 50000.0
        return 0.0


@pytest.mark.asyncio
async def test_authoritative_wallet_sync_rebuilds_and_publishes_nav():
    ss = SharedState(
        config=SharedStateConfig(),
        exchange_client=_FakeExchangeClient(),
    )

    result = await ss.authoritative_wallet_sync()

    assert result["balances"]["USDT"]["free"] == 250.0
    assert "BTCUSDT" in result["positions"]

    expected_nav = float(ss.get_nav_quote())
    assert expected_nav > 0.0

    assert ss.nav == expected_nav
    assert ss.portfolio_nav == expected_nav
    assert ss.total_equity == expected_nav
    assert ss.total_equity_usdt == expected_nav
    assert float(ss.metrics["nav"]) == expected_nav
    assert float(ss.metrics["total_equity"]) == expected_nav
    assert ss.nav_ready_event.is_set()
