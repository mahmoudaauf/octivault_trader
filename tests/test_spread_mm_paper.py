"""Tests for the paper market-maker.

The whole tool is a measurement instrument, so the things that must be right are
the fill model (must not invent fills we would not have got) and the mark-out
sign convention (an inverted sign would turn toxic flow into apparent profit --
exactly the false-positive this project keeps getting bitten by).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spread_mm_paper as mm


# ── quotes ───────────────────────────────────────────────────────────────────
def test_quotes_at_touch_by_default():
    b, a = mm.compute_quotes(100.0, 101.0, inside_frac=0.0)
    assert (b, a) == (100.0, 101.0)


def test_quotes_step_inside_the_spread():
    b, a = mm.compute_quotes(100.0, 101.0, inside_frac=0.25)
    assert b == pytest.approx(100.25)
    assert a == pytest.approx(100.75)
    assert b < a, "quoting inside must never cross the book"


# ── fill model ───────────────────────────────────────────────────────────────
# aggTrade "m"=True  -> buyer was maker -> the AGGRESSOR SOLD -> hits our BID
# aggTrade "m"=False -> aggressor BOUGHT -> lifts our ASK
def test_through_model_fills_only_when_market_trades_past_us():
    trades = [
        {"p": "99.5", "q": "1.0", "m": True},    # sold below our bid -> fills us
        {"p": "100.0", "q": "5.0", "m": True},   # exactly AT our bid -> queue unknown
    ]
    bought, sold = mm.simulate_fills(trades, 100.0, 101.0, queue_model="through")
    assert bought == pytest.approx(1.0), "must not claim the at-touch fill"
    assert sold == 0.0


def test_touch_model_also_fills_at_our_price():
    trades = [{"p": "100.0", "q": "5.0", "m": True}]
    bought, _ = mm.simulate_fills(trades, 100.0, 101.0, queue_model="touch")
    assert bought == pytest.approx(5.0)


def test_ask_side_fills_on_aggressive_buys():
    trades = [
        {"p": "101.5", "q": "2.0", "m": False},  # bought above our ask -> fills us
        {"p": "100.2", "q": "9.0", "m": False},  # inside spread, never reaches ask
    ]
    bought, sold = mm.simulate_fills(trades, 100.0, 101.0, queue_model="through")
    assert sold == pytest.approx(2.0)
    assert bought == 0.0


def test_no_fills_when_market_stays_inside_the_spread():
    trades = [{"p": "100.5", "q": "50.0", "m": True}, {"p": "100.4", "q": "50.0", "m": False}]
    assert mm.simulate_fills(trades, 100.0, 101.0, queue_model="through") == (0.0, 0.0)


# ── mark-outs: the sign convention is the whole experiment ───────────────────
# Measured MID-TO-MID. Comparing fill price to mid double counts the spread and
# manufactures a false edge -- the first smoke run reported mark-outs almost
# exactly equal to the half-spread on every fill because of that.
def test_markout_is_zero_when_price_does_not_move():
    """THE regression that matters: fill at the bid, market unchanged, mid still
    100 -> the mark-out must be 0, NOT +half-spread."""
    assert mm.markout_pct("buy", 100.0, 100.0) == pytest.approx(0.0)
    assert mm.markout_pct("sell", 100.0, 100.0) == pytest.approx(0.0)


def test_markout_positive_when_price_moves_our_way_after_a_buy():
    # bought while mid was 100, mid later 101 -> benign flow
    assert mm.markout_pct("buy", 100.0, 101.0) == pytest.approx(1.0)


def test_markout_negative_when_picked_off_on_a_buy():
    # bought while mid was 100, mid later 99 -> toxic flow, we were run over
    assert mm.markout_pct("buy", 100.0, 99.0) == pytest.approx(-1.0)


def test_markout_sign_flips_for_sells():
    assert mm.markout_pct("sell", 100.0, 99.0) == pytest.approx(1.0)
    assert mm.markout_pct("sell", 100.0, 101.0) == pytest.approx(-1.0)


def test_markout_handles_zero_mid():
    assert mm.markout_pct("buy", 0.0, 5.0) == 0.0


def test_markout_uses_mid_at_fill_not_fill_price():
    """Guard the caller wiring: the runtime must pass mid_at_fill."""
    import os as _os
    src = open(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "spread_mm_paper.py")).read()
    assert 'markout_pct(rec["side"], rec["mid_at_fill"], m)' in src
    assert 'markout_pct(rec["side"], rec["price"], m)' not in src


# ── the tool must never be able to trade ────────────────────────────────────
def test_module_has_no_order_placing_calls():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "spread_mm_paper.py")).read()
    for forbidden in ("order_market_buy", "order_market_sell", "create_order",
                      "create_oco_order", "subscribe_simple_earn", "redeem_simple_earn"):
        assert forbidden not in src, f"paper tool must not reference {forbidden}"


# ── causal ordering: a quote can only be filled by trades posted AFTER it ────
def test_no_fills_from_the_backlog_on_first_sight():
    """aggTrades returns the last 500 prints — hours on a thin pair. Matching
    them against a current book manufactured a full set of fake fills on cycle
    one and reported an edge that did not exist."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "spread_mm_paper.py")).read()
    assert "WARMUP" in src, "first sight of a symbol must only set the watermark"
    # fills must be simulated against the PREVIOUS cycle's quote
    assert 'simulate_fills(fresh, prev["bid_q"], prev["ask_q"])' in src
    # and the new quote must be posted after that matching
    assert src.index('simulate_fills(fresh, prev["bid_q"]') < src.index('state.setdefault("quote", {})[sym]')


def test_fill_record_carries_the_quotes_own_mid():
    """mark-outs must be measured from the mid that was live when the quote was
    posted, not from a later snapshot."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "spread_mm_paper.py")).read()
    assert '"mid_at_fill": mid_p' in src
