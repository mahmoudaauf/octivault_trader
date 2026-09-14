"""The wealth desk measures purchasing power. Sign errors here invert advice."""
from __future__ import annotations

import importlib

import pytest

wd = importlib.import_module("wealth_desk")


def test_a_weakening_pound_RAISES_the_real_value_of_a_dollar_asset():
    """Negative egp_move means the pound fell, which is good for a USD holder.
    Getting this sign backwards would invert every conclusion on the page."""
    weak = wd.real_return(0.086, -0.15, 0.145)
    flat = wd.real_return(0.086, 0.0, 0.145)
    strong = wd.real_return(0.086, 0.067, 0.145)
    assert weak > flat > strong
    assert weak == pytest.approx(0.1039, abs=1e-3)
    assert strong == pytest.approx(-0.1320, abs=1e-3)


def test_the_fx_move_compounds_with_the_yield_rather_than_adding():
    """A 15% devaluation on top of 8.6% is multiplicative, not 23.6%."""
    got = wd.real_return(0.086, -0.15, 0.0)
    assert got == pytest.approx(1.086 * 1.15 - 1, abs=1e-9)
    assert got > 0.086 + 0.15 - 1e-9      # strictly more than the naive sum


def test_inflation_is_the_hurdle_even_at_a_doubled_yield():
    """The finding the desk exists to state: 8.60% against 14.5% CPI is a real
    LOSS with the pound flat, and doubling the rate barely clears zero."""
    assert wd.real_return(0.086, 0.0, 0.145) < 0
    assert wd.real_return(0.172, 0.0, 0.145) == pytest.approx(0.027, abs=1e-3)


def test_fx_falls_back_visibly_rather_than_silently(monkeypatch, tmp_path):
    """There is no USDTEGP spot pair; an unavailable rate must be labelled, not
    quietly substituted, or the EGP figures look measured when they are guessed."""
    monkeypatch.setenv("P2P_STATE", str(tmp_path / "nope.jsonl"))
    rate, src = wd.fx_rate()
    assert rate == wd.FALLBACK_FX and "FALLBACK" in src


def test_fx_uses_the_p2p_mid_when_history_exists(monkeypatch, tmp_path):
    import json
    p = tmp_path / "h.jsonl"
    p.write_text(json.dumps({"ts": "x", "buy": 51.0, "sell": 53.0,
                             "spread_pct": 3.9, "spread_median_pct": 2.0,
                             "buy_median": 51.2, "sell_median": 52.8,
                             "buy_min_fiat": 5000.0, "sell_min_fiat": 5000.0}) + "\n")
    monkeypatch.setenv("P2P_STATE", str(p))
    rate, src = wd.fx_rate()
    assert rate == pytest.approx(52.0) and "P2P mid" in src
