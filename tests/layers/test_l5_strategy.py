"""L5 — Strategy: pure decisions, no side effects, deterministic with seed."""
import random

from tests.layers.fakes import FakeExchange, FakePortfolio


def _toy_strategy(ctx: dict, *, seed: int = 42) -> list:
    """A tiny pure decision function used as a stand-in for the real engine."""
    rng = random.Random(seed)
    n = len(ctx.get("symbols", []))
    intents = []
    for sym in ctx.get("symbols", []):
        if rng.random() > 0.5:
            intents.append({"id": f"{sym}-i", "symbol": sym, "side": "BUY",
                            "qty": 0.01, "price": ctx["prices"][sym]})
    return intents


def test_l5_is_deterministic_with_fixed_seed():
    ctx = {"symbols": ["BTC", "ETH", "SOL"],
           "prices": {"BTC": 40000.0, "ETH": 2500.0, "SOL": 100.0}}
    a = _toy_strategy(ctx, seed=42)
    b = _toy_strategy(ctx, seed=42)
    assert a == b


def test_l5_does_not_call_exchange():
    """Strategy must NEVER call exchange.place_order — that's L4's job."""
    ex = FakeExchange()
    ctx = {"symbols": ["BTC"], "prices": {"BTC": 40000.0}}
    _toy_strategy(ctx, seed=1)
    assert ex.placed_orders == []


def test_l5_does_not_mutate_portfolio():
    p = FakePortfolio(cash=10_000.0)
    snapshot = p.buckets()
    ctx = {"symbols": ["BTC"], "prices": {"BTC": 40000.0}}
    _toy_strategy(ctx, seed=1)
    assert p.buckets() == snapshot
    assert p.journal == []
