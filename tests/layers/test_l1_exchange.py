"""L1 — Exchange I/O contract + retry + reconcile."""
import asyncio

from src.l0_core.layer_contracts import L1ExchangeContract
from tests.layers.fakes import FakeExchange


def test_l1_contract_validates_well_formed_output():
    c = L1ExchangeContract()
    assert c.validate_output(
        {
            "balances": {"USDT": {"free": 1.0, "locked": 0.0}},
            "open_positions": [],
            "exchange_time_ms": 1_700_000_000_000,
            "rate_limit_remaining": 1000,
        }
    )


def test_l1_contract_rejects_missing_fields():
    c = L1ExchangeContract()
    assert not c.validate_output({"balances": {}})
    assert not c.validate_output(
        {
            "balances": {},
            "open_positions": [],
            "exchange_time_ms": 0,
            # rate_limit_remaining missing
        }
    )


def test_l1_get_balances_succeeds_after_retry():
    """Simulate two transient failures, then success."""
    ex = FakeExchange(fail_first_n=2)

    async def with_retry():
        last_err = None
        for _ in range(3):
            try:
                return await ex.get_balances()
            except ConnectionError as e:
                last_err = e
        raise last_err

    out = asyncio.get_event_loop().run_until_complete(with_retry())
    assert "USDT" in out


def test_l1_order_cache_reconcile_zero_drift():
    ex = FakeExchange()
    ex.upsert({"id": "A", "symbol": "BTC", "qty": 1.0})
    ex.upsert({"id": "B", "symbol": "ETH", "qty": 2.0})
    diff = ex.reconcile([{"id": "A"}, {"id": "B"}])
    assert diff == {"missing_local": set(), "stale_local": set()}


def test_l1_place_order_marks_filled():
    ex = FakeExchange()
    out = asyncio.get_event_loop().run_until_complete(
        ex.place_order({"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.1, "price": 40000.0})
    )
    assert out["status"] == "FILLED"
    assert ex.placed_orders[0]["id"] == out["id"]
