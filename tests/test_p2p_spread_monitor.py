"""P2P spread monitor: a spread you cannot reach is not a spread.

The monitor exists because this project has an expensive lesson about
snapshots — delta_neutral_yield_scan ranked on one and 11 of 25 candidates were
negative on 90-day history. These tests pin the two things that make the
reported number honest: median-of-book rather than top-of-book, and the
minimum order size recorded alongside the price.
"""
from __future__ import annotations

import importlib
import json

import pytest

p2p = importlib.import_module("p2p_spread_monitor")


def _ad(price, min_fiat=5000.0, orders=300, rails=("Instapay",),
        user_type="merchant", fee=0.0):
    return {"adv": {"price": str(price), "minSingleTransAmount": str(min_fiat),
                    "commissionRate": str(fee), "takerCommissionRate": str(fee),
                    "tradeMethods": [{"tradeMethodName": r} for r in rails]},
            "advertiser": {"monthOrderCount": orders, "userType": user_type}}


def test_sample_records_both_top_and_median_spread(tmp_path, monkeypatch):
    """Top-of-book can be one outlier ad with an unreachable minimum; the
    median is the honest middle of the book, and both are recorded."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        [_ad(100.0), _ad(102.0), _ad(104.0)] if tt == "BUY"
                        else [_ad(106.0), _ad(108.0), _ad(110.0)])
    row = p2p.sample()
    assert row["buy"] == 100.0 and row["sell"] == 110.0
    assert row["spread_pct"] == pytest.approx(10.0)          # 100 -> 110
    assert row["buy_median"] == 102.0 and row["sell_median"] == 108.0
    assert row["spread_median_pct"] == pytest.approx(5.8824, abs=1e-3)
    assert row["spread_median_pct"] < row["spread_pct"]      # median is conservative


def test_minimum_order_size_is_recorded_not_dropped(tmp_path, monkeypatch):
    """At $60 this account sits below the smallest ad and can place no trade at
    all. A spread you cannot reach is not a spread, so the reach is recorded."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        [_ad(100.0, 20000.0)] if tt == "BUY" else [_ad(110.0, 15000.0)])
    row = p2p.sample()
    assert row["buy_min_fiat"] == 20000.0 and row["sell_min_fiat"] == 15000.0


def test_an_empty_book_records_nothing_rather_than_a_zero(tmp_path, monkeypatch, capsys):
    """Unknown is not zero — a fetch that returns nothing must not write a row
    that a later report would average in as a 0% spread."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1: [])
    assert p2p.sample() is None
    assert "empty book" in capsys.readouterr().out
    assert not (tmp_path / "h.jsonl").exists()


def test_a_failed_fetch_records_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    def boom(tt, rows=20, pages=1): raise RuntimeError("network down")
    monkeypatch.setattr(p2p, "_ads", boom)
    assert p2p.sample() is None
    assert "fetch failed" in capsys.readouterr().out


def test_report_warns_while_the_sample_is_still_thin(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    rows = [{"ts": "2026-09-13T19:00:00+00:00", "buy": 51.7, "sell": 52.4,
             "buy_min_fiat": 5000.0, "sell_min_fiat": 5000.0,
             "spread_pct": 1.35, "spread_median_pct": 0.9,
             "buy_median": 51.8, "sell_median": 52.3, "n_buy": 20, "n_sell": 20}]
    (tmp_path / "h.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    p2p.report()
    out = capsys.readouterr().out
    assert "Not a verdict yet" in out and "delta-neutral" in out


def test_report_prices_the_goal_off_the_TYPICAL_spread(tmp_path, monkeypatch, capsys):
    """A turnover business earns the median spread, not the best one ever seen."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    rows = [{"ts": f"2026-09-1{d}T19:00:00+00:00", "buy": 50.0, "sell": 55.0,
             "buy_min_fiat": 5000.0, "sell_min_fiat": 5000.0,
             "spread_pct": 10.0, "spread_median_pct": 1.0,
             "buy_median": 50.0, "sell_median": 50.5, "n_buy": 20, "n_sell": 20}
            for d in range(1, 5)]
    (tmp_path / "h.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    p2p.report()
    out = capsys.readouterr().out
    assert "$100 of turnover per hour" in out     # 1.0% median -> $100/hr, not $10


# --- reach, velocity and rails (2026-09-29) -----------------------------------
# The monitor measured the spread well and left the two things that decide
# whether it is income unmeasured: whether a $60 account can reach the book, and
# how many round trips a day the market supports. Both were in the payload
# already being fetched. Measured on the live book: the median minimum order is
# 5,000 EGP (~$95) but the floor is 70 EGP (~$1), 74 of 200 ads accept under
# $25, and the median merchant completes 286 orders per 30 days.

def test_top_of_book_is_not_the_reachable_spread(tmp_path, monkeypatch):
    """The headline spread is routinely posted by an ad this account is too small
    to touch. Recording only that number is how a screenshot becomes a plan."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "CAPITAL_USD", 60.0)
    # Best prices sit behind huge minimums; a reachable pair exists further in.
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        [_ad(100.0, 900_000.0), _ad(101.0, 1_000.0)] if tt == "BUY"
                        else [_ad(110.0, 900_000.0), _ad(105.0, 1_000.0)])
    row = p2p.sample()
    # Top of book: 100 -> 110 = 10%, but neither leg is reachable at $60.
    assert row["spread_pct"] == pytest.approx(10.0)
    # Reachable: 101 -> 105 = 3.96%, the number that could actually be traded.
    assert row["buy_reach"] == 101.0 and row["sell_reach"] == 105.0
    assert row["spread_reach_pct"] == pytest.approx(3.9604, abs=1e-3)
    assert row["spread_reach_pct"] < row["spread_pct"]
    assert row["min_fiat_floor"] == 1_000.0


def test_an_unreachable_book_records_none_not_a_number(tmp_path, monkeypatch):
    """When every ad is too big, the reachable spread is unknown. Unknown is not
    zero and it is not the top-of-book figure either."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "CAPITAL_USD", 60.0)
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        [_ad(100.0, 5_000_000.0)] if tt == "BUY"
                        else [_ad(110.0, 5_000_000.0)])
    row = p2p.sample()
    assert row["spread_reach_pct"] is None
    assert row["n_reach"] == 0
    assert row["spread_pct"] == pytest.approx(10.0)   # still recorded, still true


def test_throughput_and_rails_are_recorded(tmp_path, monkeypatch):
    """Velocity is the term that was implicitly 1. Rail overlap was listed as
    unmeasurable in the monitor's own docstring; it is not."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        [_ad(100.0, 100.0, orders=300, rails=("Instapay", "telda"))]
                        if tt == "BUY" else
                        [_ad(110.0, 100.0, orders=900, rails=("Instapay", "OrangeCash"))])
    row = p2p.sample()
    assert row["orders30_max"] == 900
    assert row["n_merchant"] == 2
    assert row["rails_both"] == ["Instapay"]          # the only capturable rail
    assert row["maker_fee_p50"] == 0.0


def test_the_median_stays_comparable_when_the_book_gets_deeper(tmp_path, monkeypatch):
    """Deepening the fetch changed what 'median-of-book' meant: the same market
    read -1.199% instead of +0.478%, because page 5 of the BUY side is priced far
    above page 1. 1,192 earlier observations are medians of the top 20, so the
    median must stay pinned to the first page however deep the fetch goes."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    shallow = [_ad(100.0 + i, 100.0) for i in range(20)]
    deep = shallow + [_ad(500.0, 100.0) for _ in range(80)]   # far-page outliers
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20, pages=1:
                        deep if tt == "BUY" else [_ad(200.0, 100.0)])
    row = p2p.sample()
    # Median of the first 20 only — 500s on later pages must not drag it.
    assert row["buy_median"] == pytest.approx(109.5)
    assert row["n_buy"] == 100          # depth WAS fetched, for reach
