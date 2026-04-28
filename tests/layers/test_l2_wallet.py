"""L2 — Wallet & Market Data: snapshot + classification invariants."""
import asyncio
from src.l0_core.layer_contracts import L2WalletContract
from tests.layers.fakes import FakeExchange


def test_l2_contract_requires_classified_assets():
    c = L2WalletContract()
    assert not c.validate_output({"assets": {"BTC": {}},  # missing classification
                                  "positions": {}, "last_updated": 0.0})
    assert c.validate_output({
        "assets": {"BTC": {"classification": "BOT_POSITION"}},
        "positions": {},
        "last_updated": 0.0,
    })


def test_l2_classification_values_are_constrained():
    c = L2WalletContract()
    assert not c.validate_output({
        "assets": {"X": {"classification": "BOGUS"}},
        "positions": {}, "last_updated": 0.0,
    })


def test_l2_snapshot_is_independent_per_call():
    """Mutating a returned snapshot must not affect subsequent reads."""
    ex = FakeExchange()
    snap1 = asyncio.get_event_loop().run_until_complete(ex.get_balances())
    snap1["USDT"]["free"] = -1
    snap2 = asyncio.get_event_loop().run_until_complete(ex.get_balances())
    assert snap2["USDT"]["free"] == 10_000.0
