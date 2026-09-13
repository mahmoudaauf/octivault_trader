"""The office brief must never flatter: no estimate dressed as a measurement,
no promising snapshot dressed as a tested result, no count dressed as a price."""
from __future__ import annotations

import importlib

import pytest

ob = importlib.import_module("office_brief")


def test_unreadable_exchange_reports_unknown_not_zero(capsys, monkeypatch):
    """A failed read must surface as UNKNOWN. Rendering it as $0.00 would show a
    wiped-out account, which is the most alarming possible lie."""
    monkeypatch.setattr(ob, "_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")))
    pos = ob.position()
    assert pos["err"] and pos["nav"] is None and pos["rate"] is None
    ob.performance(pos["nav"], pos["rate"])
    assert ob.UNKNOWN in capsys.readouterr().out


def test_blended_rate_uses_base_plus_tier(monkeypatch):
    """The tier is a bonus ON TOP OF the base — the lesson that took three days
    and a near-miss rotation to establish. The brief must not regress it."""
    def fake(path, params=None, signed=True):
        if "position" in path:
            return {"rows": [{"asset": "USD1", "totalAmount": "60.0",
                              "cumulativeBonusRewards": "0.04"}]}
        if "account" in path:
            return {"balances": []}
        return {"rows": [{"latestAnnualPercentageRate": "0.0159",
                          "tierAnnualPercentageRate": {"0-1500USD1": "0.07"}}]}
    monkeypatch.setattr(ob, "_get", fake)
    pos = ob.position()
    assert pos["rate"] == pytest.approx(0.0859)      # 1.59 base + 7.00 tier
    assert pos["nav"] == pytest.approx(60.0)


def test_a_long_cell_truncates_instead_of_shunting_the_row(capsys, monkeypatch):
    monkeypatch.setattr(ob, "_get", lambda *a, **k: {"tracking": [], "rows": []})
    monkeypatch.setattr(ob, "P2P_FILE", "/nonexistent")
    ob.pipeline()
    for line in capsys.readouterr().out.splitlines():
        if line.startswith("  ") and "candidate" not in line and "─" not in line:
            assert len(line) < 140          # no row explodes the page width


def test_dust_is_reported_as_a_count_not_a_dollar_amount(capsys, monkeypatch):
    """It rendered as '$9 coins stranded' — nine coins, not nine dollars."""
    monkeypatch.setattr(ob, "_get", lambda *a, **k: {"rows": []})
    ob.risks({"earn": {}, "spot": {"ADA": 0.1, "INJ": 0.01, "PEPE": 1.0}, "nav": 0.0})
    out = capsys.readouterr().out
    assert "3 coins stranded" in out and "$3 coins" not in out


def test_promo_tier_fragility_is_flagged_when_the_bonus_dominates(capsys, monkeypatch):
    """USD1 is 1.59% base + 7.00% tier: nearly all its value is the promo, so it
    degrades catastrophically if the promo ends. That must be on the brief."""
    def fake(path, params=None, signed=True):
        return {"rows": [{"latestAnnualPercentageRate": "0.0159",
                          "tierAnnualPercentageRate": {"0-1500USD1": "0.07"}}]}
    monkeypatch.setattr(ob, "_get", fake)
    ob.risks({"earn": {"USD1": {"amt": 60.0, "bonus_cum": 0.0}}, "spot": {}, "nav": 60.0})
    out = capsys.readouterr().out
    assert "promo-tier fragility" in out and "USD1" in out
    assert "concentration" in out           # 100% in one issuer


def test_a_graceful_rate_product_is_not_flagged_as_fragile(capsys, monkeypatch):
    """USDT at 2.83 base + 4.00 tier keeps most of its value if the promo ends."""
    def fake(path, params=None, signed=True):
        return {"rows": [{"latestAnnualPercentageRate": "0.0283",
                          "tierAnnualPercentageRate": {"0-800USDT": "0.04"}}]}
    monkeypatch.setattr(ob, "_get", fake)
    ob.risks({"earn": {"USDT": {"amt": 60.0, "bonus_cum": 0.0}}, "spot": {}, "nav": 60.0})
    assert "promo-tier fragility" not in capsys.readouterr().out
