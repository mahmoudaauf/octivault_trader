from __future__ import annotations

import pytest

from core_engine.market_account_engine import MarketAccountEngine


class _ThrottleExchange:
    def __init__(self) -> None:
        self.account_calls = 0
        self.price_calls = 0

    def is_throttled(self) -> bool:
        return True

    async def get_account(self):
        self.account_calls += 1
        raise AssertionError("REST account call should be skipped while throttled")

    async def get_prices(self, symbols=None):
        self.price_calls += 1
        raise AssertionError("REST price call should be skipped while throttled")


class _State:
    def __init__(self) -> None:
        self.exchange_throttled = True
        self.balance = {"USDT": 42.0, "BTC": 0.01}
        self.positions = {"BTCUSDT": object()}
        self.open_orders = []
        self.prices = {"BTCUSDT": 65000.0, "ETHUSDT": 3000.0}


@pytest.mark.asyncio
async def test_market_account_engine_uses_cached_state_while_throttled() -> None:
    exchange = _ThrottleExchange()
    state = _State()
    engine = MarketAccountEngine(
        {
            "exchange_client": exchange,
            "shared_state": state,
            "market_data_feed": None,
            "balance_manager": None,
        }
    )

    account = await engine.get_account_state()
    prices = await engine.get_market_prices(["BTCUSDT"])
    wallet = await engine.get_wallet_balance()

    assert account["balances"]["USDT"] == 42.0
    assert "BTCUSDT" in account["positions"]
    assert prices == {"BTCUSDT": 65000.0}
    assert wallet["available_usdt"] == 42.0
    assert exchange.account_calls == 0
    assert exchange.price_calls == 0
