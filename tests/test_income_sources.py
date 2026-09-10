"""Income-source classification: the axis a credit lands on decides what it means.

Blending capital-bound yield with capital-free commission would let $0.0005/hour
of interest and $0.00/hour of referral average into one number that hides which
lever is moving — which is the entire question behind the $1/hour goal.
"""
from __future__ import annotations

import importlib

import pytest

inc = importlib.import_module("income_sources")


@pytest.mark.parametrize("label", [
    "Flexible", "Simple Earn Flexible", "Locked Staking", "BFUSD Daily reward",
    "ETH2 staking rewards", "Savings Interest",
])
def test_yield_labels_are_capital_bound(label):
    assert inc.classify(label) == "capital_bound"


@pytest.mark.parametrize("label", [
    "Referral Commission", "Spot referral kickback", "Affiliate commission rebate",
    "Launchpool Airdrop", "HODLer Airdrops", "Megadrop reward",
    "Token swap distribution for OMNI", "Campaign reward voucher",
])
def test_capital_free_labels_are_detected(label):
    assert inc.classify(label) == "capital_free"


def test_an_unknown_label_is_flagged_not_guessed():
    """A credit type we have never seen is the thing most worth noticing, and
    guessing its axis would corrupt both totals."""
    assert inc.classify("Some Brand New 2027 Reward Type") == "unclassified"
    assert inc.classify("") == "unclassified"
    assert inc.classify(None) == "unclassified"


def test_classification_is_case_insensitive():
    assert inc.classify("REFERRAL COMMISSION") == "capital_free"
    assert inc.classify("flexible") == "capital_bound"


def test_yield_wins_when_a_label_could_read_either_way():
    """'Launchpool staking' is a return on held BNB, not a free-token payout —
    it must not be counted on the capital-free axis."""
    assert inc.classify("Launchpool staking interest") == "capital_bound"


def test_stablecoins_are_priced_without_a_network_call():
    for a in ("USDT", "USDC", "USD1", "FDUSD", "BFUSD"):
        assert inc._price(a) == 1.0
