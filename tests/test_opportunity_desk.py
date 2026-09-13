"""The opportunity desk: catch an asset ENTERING the tiered space, and never
imply a volatile token's rate is a return.

The largest gain this account ever made (+1.35pp, permanent) came from noticing
USD1 had a tier while the allocator scanned a hardcoded list without it. This
desk exists to catch the next one automatically, so these tests pin the event
that matters and the exclusions that keep it honest.
"""
from __future__ import annotations

import importlib

import pytest

od = importlib.import_module("opportunity_desk")


def P(base, tier, cap="0-1000X", open_=True):
    return {"base": base, "tier": tier, "cap": cap, "open": open_}


def test_a_tier_appearing_is_the_headline_event():
    """Exactly the USD1 shape: an asset that had no tier suddenly has one."""
    old = {"USD1": P(0.0159, 0.0), "USDC": P(0.0211, 0.05)}
    new = {"USD1": P(0.0159, 0.07), "USDC": P(0.0211, 0.05)}
    events = od.diff(old, new)
    assert any(e.startswith("★ TIER APPEARED on USD1") for e in events)
    assert any("8.59%" in e for e in events)


def test_a_tier_disappearing_is_reported_too():
    """The promo ending is as actionable as it starting — it is the moment the
    rate collapses and the money should move."""
    events = od.diff({"USD1": P(0.0159, 0.07)}, {"USD1": P(0.0159, 0.0)})
    assert any("TIER REMOVED from USD1" in e for e in events)


def test_a_volatile_token_gaining_a_tier_raises_nothing():
    """KGST pays 13.05%. It is a token, not a dollar. A rate on an asset that
    can halve is not a return, and the desk must never imply it is."""
    events = od.diff({"KGST": P(0.0305, 0.0)}, {"KGST": P(0.0305, 0.10)})
    assert events == []


def test_synthetic_dollars_are_excluded_despite_being_pegged():
    """USDe and BFUSD earn their yield from a basis trade — a short under the
    hood — which the principal excluded."""
    for a in ("USDE", "BFUSD"):
        assert od.actionable(a, P(0.04, 0.10)) is False
    assert od.actionable("USD1", P(0.0159, 0.07)) is True


def test_a_closed_product_is_not_actionable():
    """USDP showed 10.66% while isSoldOut — a closed product's rate during a
    borrow squeeze is not an opportunity."""
    assert od.actionable("USDP", P(0.1066, 0.0, open_=False)) is False


def test_small_rate_drift_does_not_generate_noise():
    """Rates move fractionally every cycle; only a move past MIN_EDGE_PP counts."""
    old = {"USDC": P(0.0211, 0.05)}
    new = {"USDC": P(0.0212, 0.05)}
    assert od.diff(old, new) == []


def test_a_closure_of_something_we_could_have_used_is_reported():
    events = od.diff({"USD1": P(0.0159, 0.07, open_=True)},
                     {"USD1": P(0.0159, 0.07, open_=False)})
    assert any("CLOSED USD1" in e for e in events)


def test_a_reopening_is_reported():
    events = od.diff({"USD1": P(0.0159, 0.07, open_=False)},
                     {"USD1": P(0.0159, 0.07, open_=True)})
    assert any("REOPENED USD1" in e for e in events)


def test_a_volatile_token_closing_raises_nothing():
    """The closure path must honour the same stable-only rule as everything else."""
    assert od.diff({"KGST": P(0.0305, 0.10, open_=True)},
                   {"KGST": P(0.0305, 0.10, open_=False)}) == []
