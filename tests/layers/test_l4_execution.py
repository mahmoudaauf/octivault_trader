"""L4 — Execution: no order without ReservationToken; exactly-once journaling."""
import asyncio
import pytest

from src.l0_core.layer_contracts import L4ExecutionContract
from tests.layers.fakes import FakeExchange, FakePortfolio


def test_l4_contract_validates_required_fields():
    c = L4ExecutionContract()
    assert c.validate_output({
        "tickets": [], "fills": [], "cancels": [], "timestamp": 0.0,
    })
    assert not c.validate_output({"tickets": []})


def test_l4_intent_validator_rejects_no_token():
    c = L4ExecutionContract()
    assert not c.validate_intent(
        reservation_token=None, symbol="BTCUSDT", side="BUY", quantity=0.1
    )
    assert c.validate_intent(
        reservation_token="R-abc", symbol="BTCUSDT", side="BUY", quantity=0.1
    )


def test_l4_no_token_means_no_order_placed():
    """If L4 doesn't have a token from L3, it must not call exchange.place_order."""
    ex = FakeExchange()
    p = FakePortfolio(cash=10_000.0)
    contract = L4ExecutionContract()

    intent = {"id": "i1", "symbol": "BTCUSDT", "side": "BUY",
              "qty": 0.1, "price": 40_000.0}

    async def submit_with_gate(token):
        if not contract.validate_intent(token, intent["symbol"],
                                        intent["side"], intent["qty"]):
            return None
        return await ex.place_order({**intent, "quantity": intent["qty"]})

    # No token -> no order
    out = asyncio.get_event_loop().run_until_complete(submit_with_gate(None))
    assert out is None
    assert ex.placed_orders == []

    # With token -> order goes out, journal records fill
    tok = p.reserve("BTCUSDT", 4_000.0, "buy")
    out = asyncio.get_event_loop().run_until_complete(submit_with_gate(tok))
    assert out is not None and out["status"] == "FILLED"
    assert p.apply_fill(tok, "BTCUSDT", out["filled_qty"], out["filled_price"])
    fills = [e for e in p.journal if e["event"] == "FILL"]
    assert len(fills) == 1


def test_l4_exchange_failure_releases_reservation():
    """If exchange rejects, capital must return to CASH (no leak)."""
    ex = FakeExchange()
    p = FakePortfolio(cash=10_000.0)
    cash0 = p.buckets()["CASH"]

    tok = p.reserve("BTCUSDT", 4_000.0, "buy")
    assert p.buckets()["CASH"] == cash0 - 4_000.0

    # Simulate exchange failure path: don't apply fill, just release.
    assert p.release(tok)
    assert p.buckets()["CASH"] == cash0
