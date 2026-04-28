"""L3 — Portfolio: bucket conservation, journaling, EXTERNAL read-only."""
import pytest
from tests.layers.fakes import FakePortfolio


def test_l3_bucket_conservation_after_reserve_and_release():
    p = FakePortfolio(cash=10_000.0)
    initial_total = p.total
    token = p.reserve("BTCUSDT", 1_000.0, "test")
    assert token is not None
    assert p.total == pytest.approx(initial_total)        # invariant
    assert p.release(token)
    assert p.total == pytest.approx(initial_total)
    assert p.buckets()["CASH"] == pytest.approx(10_000.0)


def test_l3_apply_fill_creates_exactly_one_journal_entry():
    p = FakePortfolio(cash=10_000.0)
    token = p.reserve("BTCUSDT", 4_000.0, "buy")
    n_before = len(p.journal)
    assert p.apply_fill(token, "BTCUSDT", qty=0.1, price=40_000.0)
    fills = [e for e in p.journal[n_before:] if e["event"] == "FILL"]
    assert len(fills) == 1


def test_l3_double_fill_with_same_token_is_rejected():
    p = FakePortfolio(cash=10_000.0)
    token = p.reserve("BTCUSDT", 4_000.0, "buy")
    assert p.apply_fill(token, "BTCUSDT", 0.1, 40_000.0)
    # Second use of the same token must be refused (exactly-once)
    assert not p.apply_fill(token, "BTCUSDT", 0.1, 40_000.0)


def test_l3_external_position_is_read_only():
    p = FakePortfolio()
    p.add_external("USDC", qty=500.0, price=1.0)
    with pytest.raises(PermissionError):
        p.force_mutate_external("USDC")


def test_l3_overdraft_is_refused():
    p = FakePortfolio(cash=100.0)
    assert p.reserve("BTC", 1_000.0, "too big") is None
