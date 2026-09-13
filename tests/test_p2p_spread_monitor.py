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


def _ad(price, min_fiat=5000.0):
    return {"adv": {"price": str(price), "minSingleTransAmount": str(min_fiat)}}


def test_sample_records_both_top_and_median_spread(tmp_path, monkeypatch):
    """Top-of-book can be one outlier ad with an unreachable minimum; the
    median is the honest middle of the book, and both are recorded."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20:
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
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20:
                        [_ad(100.0, 20000.0)] if tt == "BUY" else [_ad(110.0, 15000.0)])
    row = p2p.sample()
    assert row["buy_min_fiat"] == 20000.0 and row["sell_min_fiat"] == 15000.0


def test_an_empty_book_records_nothing_rather_than_a_zero(tmp_path, monkeypatch, capsys):
    """Unknown is not zero — a fetch that returns nothing must not write a row
    that a later report would average in as a 0% spread."""
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    monkeypatch.setattr(p2p, "_ads", lambda tt, rows=20: [])
    assert p2p.sample() is None
    assert "empty book" in capsys.readouterr().out
    assert not (tmp_path / "h.jsonl").exists()


def test_a_failed_fetch_records_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p2p, "STATE", str(tmp_path / "h.jsonl"))
    def boom(tt, rows=20): raise RuntimeError("network down")
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
