"""P2P maker fill probe: a fill you cannot prove is not a fill.

The taker side of this spread died because a number that looked like an edge was
never required to be executable — requiring both legs on one payment rail, 0 of
749/755/654 combinations were positive. The maker side rests on a different
assumption: that a posted ad actually gets taken. These tests pin the one thing
that keeps the measurement honest — only a surplus DECREASE on a still-listed ad
counts as a fill. An ad that vanished may have been cancelled, and counting it
would inflate the fill rate in exactly the direction that flatters the plan.
"""
from __future__ import annotations

import importlib
import json

import pytest

probe = importlib.import_module("p2p_maker_probe")


def _ad(adv_no, price, surplus, orders=200, min_fiat=100.0):
    return {"adv": {"advNo": adv_no, "price": str(price),
                    "surplusAmount": str(surplus),
                    "minSingleTransAmount": str(min_fiat)},
            "advertiser": {"monthOrderCount": orders}}


def _paths(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "SNAP", str(tmp_path / "snap.json"))
    monkeypatch.setattr(probe, "FILLS", str(tmp_path / "fills.jsonl"))


def _book(monkeypatch, buys, sells):
    monkeypatch.setattr(probe, "_ads",
                        lambda tt: buys if tt == "BUY" else sells)


def test_first_sample_records_no_fills(tmp_path, monkeypatch, capsys):
    """Nothing to diff against. A first observation must not invent a baseline of
    zero and then report the whole book as filled on the next pass."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch, [_ad("b1", 52.8, 100)], [_ad("s1", 53.3, 100)])
    assert probe.sample() is None
    assert "first snapshot" in capsys.readouterr().out
    assert not (tmp_path / "fills.jsonl").exists()


def test_a_fill_needs_both_witnesses(tmp_path, monkeypatch):
    """Inventory fell AND the advertiser completed an order."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch, [_ad("b1", 52.8, 100, orders=200)], [_ad("s1", 53.3, 100)])
    probe.sample()
    _book(monkeypatch, [_ad("b1", 52.8, 60, orders=201)], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert len(row["fills"]) == 1
    f = row["fills"][0]
    assert f["advNo"] == "b1" and f["filled_usdt"] == pytest.approx(40.0)
    assert f["side"] == "BUY" and f["bucket"] == "top 3" and f["n_orders"] == 1


def test_an_ad_edited_down_is_not_a_fill(tmp_path, monkeypatch):
    """The flaw that produced 28,593 USDT/day per ad. Inventory fell but the
    advertiser completed no order: they shrank the ad, they did not sell."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch, [_ad("b1", 52.8, 1200, orders=200)], [_ad("s1", 53.3, 100)])
    probe.sample()
    _book(monkeypatch, [_ad("b1", 52.8, 67, orders=200)], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert row["fills"] == []
    assert row["n_shrunk"] == 1


def test_a_vanished_ad_is_never_counted_as_a_fill(tmp_path, monkeypatch):
    """The ambiguity that would flatter the plan. An ad that disappeared may have
    been cancelled, repriced off the sampled pages, or filled. Unknown is not a
    fill, and undercounting is the correct bias: it can make a viable business
    look unviable, never the reverse."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch, [_ad("b1", 52.8, 100), _ad("b2", 52.9, 500)],
          [_ad("s1", 53.3, 100)])   # b2 disappears next pass
    probe.sample()
    _book(monkeypatch, [_ad("b1", 52.8, 100)], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert row["fills"] == []
    assert row["n_vanished"] == 1


def test_a_top_up_is_not_a_negative_fill(tmp_path, monkeypatch):
    """An advertiser refunding inventory must not net against real fills."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch, [_ad("b1", 52.8, 100)], [_ad("s1", 53.3, 100)])
    probe.sample()
    _book(monkeypatch, [_ad("b1", 52.8, 900)], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert row["fills"] == []
    assert row["n_topped_up"] == 1


def test_rank_is_competitiveness_not_api_order(tmp_path, monkeypatch):
    """Best BUY price is the LOWEST (cheapest to buy); best SELL is the HIGHEST.
    Getting this backwards would report the least competitive ads as top-of-book
    and invert the whole conclusion about where to price."""
    _paths(tmp_path, monkeypatch)
    _book(monkeypatch,
          [_ad("b_worst", 60.0, 100), _ad("b_best", 52.0, 100)],
          [_ad("s_worst", 50.0, 100), _ad("s_best", 55.0, 100)])
    probe.sample()
    snap = json.load(open(tmp_path / "snap.json"))
    assert snap["ads"]["b_best"]["rank"] == 0
    assert snap["ads"]["b_worst"]["rank"] == 1
    assert snap["ads"]["s_best"]["rank"] == 0
    assert snap["ads"]["s_worst"]["rank"] == 1


def test_report_states_the_target_in_absorbed_volume(tmp_path, monkeypatch, capsys):
    """The report has to connect fill rate to the operator's actual target, or it
    is trivia: $4/week at 0.8% a round trip needs ~$71/day absorbed per side."""
    _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "TARGET_WEEK", 4.0)
    monkeypatch.setattr(probe, "SPREAD_PCT", 0.8)
    monkeypatch.setattr(probe, "CAPITAL_USD", 60.94)
    rows = [{"ts": "2026-09-30T10:00:00+00:00", "hours": 0.5,
             "n_prev": 120, "n_now": 120, "n_topped_up": 1, "n_vanished": 3,
             "fills": [{"advNo": "b1", "side": "BUY", "rank": 0, "bucket": "top 3",
                        "px": 52.8, "filled_usdt": 25.0, "orders30": 300}]}]
    (tmp_path / "fills.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    probe.report()
    out = capsys.readouterr().out
    assert "$71/day absorbed on EACH side" in out
    assert "never counted" in out          # the ambiguity is stated, not hidden
    assert "Too short" in out or "h of diffs" in out


def test_report_needs_two_samples_before_it_says_anything(tmp_path, monkeypatch, capsys):
    _paths(tmp_path, monkeypatch)
    assert probe.report() == 1
    assert "No fill history" in capsys.readouterr().out


def test_a_near_zero_interval_cannot_become_a_rate(tmp_path, monkeypatch, capsys):
    """Two samples seconds apart annualised one fill into 28,593 USDT/day per ad.
    The fill is still a fact and stays in the record; the interval is not trusted
    as a measure of time, so the row is excluded from per-hour arithmetic."""
    _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "MIN_DIFF_MIN", 0.5)
    rows = [{"ts": "2026-09-30T04:57:00+00:00", "hours": 0.0011,   # 4 seconds
             "n_prev": 120, "n_now": 120, "n_topped_up": 0, "n_vanished": 0,
             "n_shrunk": 0, "ads_per_bucket": {"4-10": 7},
             "fills": [{"advNo": "b1", "side": "BUY", "rank": 5, "bucket": "4-10",
                        "px": 52.8, "filled_usdt": 1133.3, "n_orders": 1,
                        "orders30": 300}]}]
    (tmp_path / "fills.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert probe.report() == 1
    assert "none span" in capsys.readouterr().out


def test_the_per_ad_rate_divides_by_ads_actually_observed(tmp_path, monkeypatch, capsys):
    """The divisor must be counted, not inferred from the book size."""
    _paths(tmp_path, monkeypatch)
    monkeypatch.setattr(probe, "MIN_DIFF_MIN", 0.5)
    # 1 hour, 10 U filled in a bucket that held 10 ads -> 1 U per ad per hour
    # = 24 U per ad per day.
    rows = [{"ts": "2026-09-30T10:00:00+00:00", "hours": 1.0,
             "n_prev": 120, "n_now": 120, "n_topped_up": 0, "n_vanished": 0,
             "n_shrunk": 0, "ads_per_bucket": {"4-10": 10},
             "fills": [{"advNo": "b1", "side": "BUY", "rank": 5, "bucket": "4-10",
                        "px": 52.8, "filled_usdt": 10.0, "n_orders": 1,
                        "orders30": 300}]}]
    (tmp_path / "fills.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    probe.report()
    out = capsys.readouterr().out
    assert "24.0 U" in out


def test_one_advertiser_cannot_claim_more_fills_than_orders(tmp_path, monkeypatch):
    """monthOrderCount belongs to the ADVERTISER, not the ad. An advertiser running
    two ads who sells on one and trims the other would otherwise have both
    witnesses fire on the trimmed ad too. They may claim no more ads than orders
    they actually completed; the larger inventory drop is the likelier sale."""
    _paths(tmp_path, monkeypatch)
    a1 = _ad("x1", 52.8, 500, orders=200)
    a2 = _ad("x2", 52.9, 500, orders=200)
    for a in (a1, a2):
        a["advertiser"]["userNo"] = "same_person"
    _book(monkeypatch, [a1, a2], [_ad("s1", 53.3, 100)])
    probe.sample()
    # One order completed, but BOTH ads shrank: the 400 is the sale, 50 is a trim.
    b1 = _ad("x1", 52.8, 100, orders=201)
    b2 = _ad("x2", 52.9, 450, orders=201)
    for a in (b1, b2):
        a["advertiser"]["userNo"] = "same_person"
    _book(monkeypatch, [b1, b2], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert len(row["fills"]) == 1
    assert row["fills"][0]["advNo"] == "x1"          # the bigger drop
    assert row["fills"][0]["filled_usdt"] == pytest.approx(400.0)
    assert row["n_overclaimed"] == 1


def test_two_orders_let_a_multi_ad_advertiser_claim_both(tmp_path, monkeypatch):
    """The cap must not punish an advertiser who genuinely sold on both ads."""
    _paths(tmp_path, monkeypatch)
    a1, a2 = _ad("y1", 52.8, 500, orders=200), _ad("y2", 52.9, 500, orders=200)
    for a in (a1, a2):
        a["advertiser"]["userNo"] = "busy_person"
    _book(monkeypatch, [a1, a2], [_ad("s1", 53.3, 100)])
    probe.sample()
    b1, b2 = _ad("y1", 52.8, 100, orders=202), _ad("y2", 52.9, 300, orders=202)
    for a in (b1, b2):
        a["advertiser"]["userNo"] = "busy_person"
    _book(monkeypatch, [b1, b2], [_ad("s1", 53.3, 100)])
    row = probe.sample()
    assert len(row["fills"]) == 2
    assert row["n_overclaimed"] == 0
