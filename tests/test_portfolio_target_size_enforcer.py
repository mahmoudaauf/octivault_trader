"""
Sanity tests for PortfolioTargetSizeEnforcer.

Covers:
  T1  Disabled by default → no-op, no SELLs, report.skipped_reason='disabled'.
  T2  Enabled but already at/below target → no-op, skipped_reason='below_target'.
  T3  Enabled with 8 tradable + target=5 → 3 lowest-value SELLs ordered ascending.
  T4  BOT_POSITION/RECOVERY classifications are excluded from candidates even when
      they push count above target.
  T5  Idempotency: second call to enforce_once() returns skipped_reason='already_ran_in_this_session'.
"""

from __future__ import annotations

import asyncio

from src.l3_portfolio.portfolio_target_size_enforcer import (
    PortfolioTargetSizeEnforcer,
)


class _FakeSharedState:
    def __init__(self, positions: dict) -> None:
        self._positions = positions

    def get_positions_snapshot(self, include_wallet_inventory: bool = False) -> dict:
        # Return a copy so the enforcer can't mutate our backing store
        return dict(self._positions)


class _FakeExecutionManager:
    from typing import Optional

    def __init__(self, sell_symbols: Optional[set] = None) -> None:
        self.calls: list[list[dict]] = []
        # Symbols that should be considered "sold" after execute_liquidation_plan
        self._sell_symbols = sell_symbols

    async def execute_liquidation_plan(self, exits):
        self.calls.append(list(exits))
        return True


def _pos(
    qty: float, price: float, *, classification: str = "EXTERNAL_POSITION", is_tradable: bool = True
) -> dict:
    return {
        "quantity": qty,
        "mark_price": price,
        "value_usdt": qty * price,
        "classification": classification,
        "is_tradable": is_tradable,
    }


def test_disabled_by_default_is_noop():
    ss = _FakeSharedState({"AAA/USDT": _pos(10, 100), "BBB/USDT": _pos(10, 100)})
    em = _FakeExecutionManager()
    enf = PortfolioTargetSizeEnforcer(ss, em, target_count=5, enable=False)

    report = asyncio.run(enf.enforce_once())

    assert report["skipped_reason"] == "disabled"
    assert report["ran"] is False
    assert em.calls == []


def test_below_target_is_noop():
    ss = _FakeSharedState(
        {
            "AAA/USDT": _pos(10, 10),  # $100
            "BBB/USDT": _pos(5, 20),  # $100
            "CCC/USDT": _pos(2, 50),  # $100
        }
    )
    em = _FakeExecutionManager()
    enf = PortfolioTargetSizeEnforcer(ss, em, target_count=5, enable=True)

    report = asyncio.run(enf.enforce_once())

    assert report["skipped_reason"] == "below_target"
    assert report["tradable_count"] == 3
    assert em.calls == []


def test_trim_above_target_sells_lowest_value_first():
    # 8 tradable positions, target=5 → must sell the 3 lowest-value
    positions = {
        "BIG1/USDT": _pos(1, 1000),  # $1000
        "BIG2/USDT": _pos(1, 900),  # $900
        "MID1/USDT": _pos(1, 500),  # $500
        "MID2/USDT": _pos(1, 300),  # $300
        "MID3/USDT": _pos(1, 200),  # $200
        "LOW1/USDT": _pos(1, 50),  # $50  ← cut
        "LOW2/USDT": _pos(1, 25),  # $25  ← cut
        "LOW3/USDT": _pos(1, 10),  # $10  ← cut
    }
    ss = _FakeSharedState(positions)
    em = _FakeExecutionManager()
    enf = PortfolioTargetSizeEnforcer(ss, em, target_count=5, enable=True)

    report = asyncio.run(enf.enforce_once())

    assert report["ran"] is True
    assert report["tradable_count"] == 8
    assert report["to_liquidate"] == 3
    assert report["exits_submitted"] == 3
    assert len(em.calls) == 1
    sold_symbols = [e["symbol"] for e in em.calls[0]]
    # Must sell the 3 lowest-value, ascending order
    assert sold_symbols == ["LOW3/USDT", "LOW2/USDT", "LOW1/USDT"]
    # All exits must carry the trim tag
    for exit_ in em.calls[0]:
        assert exit_["tag"] == PortfolioTargetSizeEnforcer.LIQUIDATION_TAG
        assert exit_["quantity"] > 0


def test_bot_managed_positions_are_protected():
    # Two BOT_POSITION + 6 EXTERNAL → must only consider the 6 externals,
    # so with target=5 only ONE external gets cut (the cheapest external).
    positions = {
        "BOT1/USDT": _pos(1, 10, classification="BOT_POSITION"),  # protected
        "BOT2/USDT": _pos(1, 5, classification="RECOVERY"),  # protected
        "EXT_BIG/USDT": _pos(1, 1000),
        "EXT_MID1/USDT": _pos(1, 500),
        "EXT_MID2/USDT": _pos(1, 300),
        "EXT_MID3/USDT": _pos(1, 200),
        "EXT_MID4/USDT": _pos(1, 100),
        "EXT_LOW/USDT": _pos(1, 20),  # ← only this one should be cut
    }
    ss = _FakeSharedState(positions)
    em = _FakeExecutionManager()
    enf = PortfolioTargetSizeEnforcer(ss, em, target_count=5, enable=True)

    report = asyncio.run(enf.enforce_once())

    assert report["tradable_count"] == 6  # only the 6 externals
    assert report["to_liquidate"] == 1
    sold = [e["symbol"] for e in em.calls[0]]
    assert sold == ["EXT_LOW/USDT"]


def test_idempotent_second_call_is_skipped():
    ss = _FakeSharedState(
        {
            "AAA/USDT": _pos(1, 100),
            "BBB/USDT": _pos(1, 50),
        }
    )
    em = _FakeExecutionManager()
    enf = PortfolioTargetSizeEnforcer(ss, em, target_count=5, enable=True)

    r1 = asyncio.run(enf.enforce_once())
    r2 = asyncio.run(enf.enforce_once())

    assert r1["skipped_reason"] == "below_target"
    assert r2["skipped_reason"] == "already_ran_in_this_session"
