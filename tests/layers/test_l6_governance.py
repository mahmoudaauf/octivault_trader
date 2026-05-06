"""L6 — Governance: caps inviolable, vetoes are typed, no silent downsizing."""

from src.l0_core.layer_contracts import L6PolicyContract
from tests.layers.fakes import FakePolicyGate


def test_l6_contract_decision_consistency():
    c = L6PolicyContract()
    assert c.validate_decision({"id": "i"}, approved=True, veto_reason=None)
    assert c.validate_decision({"id": "i"}, approved=False, veto_reason="cap")
    # Inconsistent: approved + reason
    assert not c.validate_decision({"id": "i"}, approved=True, veto_reason="x")
    # Inconsistent: rejected with no reason
    assert not c.validate_decision({"id": "i"}, approved=False, veto_reason=None)


def test_l6_veto_includes_reason_and_no_silent_downsizing():
    gate = FakePolicyGate(max_position_usdt=1_000.0)
    big_intent = {"id": "i1", "symbol": "BTCUSDT", "qty": 1.0, "price": 40_000.0}  # notional 40 000
    res = gate.approve(big_intent)
    assert "reason" in res
    assert res.get("approved") is not True
    assert gate.approved == []  # not silently shrunk
    assert len(gate.vetoed) == 1
    assert "cap" in res["reason"]


def test_l6_caps_are_inviolable_under_fuzz():
    """Random fuzz: no approved order may breach max_position_usdt."""
    import random

    rng = random.Random(1234)
    gate = FakePolicyGate(max_position_usdt=1_000.0)
    for i in range(1000):
        qty = rng.uniform(0.0, 0.5)
        price = rng.uniform(100.0, 100_000.0)
        gate.approve({"id": f"i{i}", "symbol": "X", "qty": qty, "price": price})
    for ap in gate.approved:
        assert ap["qty"] * ap["price"] <= 1_000.0
