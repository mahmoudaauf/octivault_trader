"""Guards for the carry daemons' sizing and booking.

Both were stopped and both strategies are falsified, but they retain live code
paths (CARRY_MODE=live in .env) and carried two bugs of the family that has bitten
this project repeatedly: a plausible value substituted for an unknown.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DAEMONS = ["carry_paper_trader.py", "negative_carry_paper_trader.py"]


def _src(name):
    return open(os.path.join(ROOT, name)).read()


@pytest.mark.parametrize("name", DAEMONS)
def test_never_falls_back_to_max_notional(name):
    """Sizing at MAXIMUM precisely when the balance is unreadable is the worst
    possible guess, and it ran once at startup so it stuck for the whole run."""
    s = _src(name)
    assert 'return CARRY_MAX_NOTIONAL, "fallback' not in s
    assert 'return MAX_NOTIONAL, "fallback' not in s
    assert 'return None, "balance unreadable"' in s


@pytest.mark.parametrize("name", DAEMONS)
def test_refuses_to_run_on_unreadable_balance(name):
    s = _src(name)
    assert "ABORT — refusing to run on an unreadable balance" in s
    assert "if NOTIONAL is None:" in s


@pytest.mark.parametrize("name", DAEMONS)
def test_unknown_funding_is_not_booked_as_zero(name):
    """A failed funding read returned 0.0, booking a FABRICATED number into the
    ledger which then fed the verdict and the drawdown halt."""
    s = _src(name)
    assert '"funding_unknown": True' in s
    assert "if _r.get(\"net_pct\") is None:" in s, "drawdown must skip unknowns"
    assert 'if t.get("net_pct") is not None' in s, "report must skip unknowns"


@pytest.mark.parametrize("name", DAEMONS)
def test_client_is_hardened_against_drift_and_dns(name):
    s = _src(name)
    assert "create_client_with_retry" in s, "must wait out an outage, not exit(1)"
    assert "resync_clock" in s, "must correct clock drift each cycle"
    assert not re.search(r"^\s*client = await AsyncClient\.create\(", s, re.M), \
        "raw AsyncClient.create leaves the daemon drift- and DNS-exposed"


@pytest.mark.parametrize("name", DAEMONS)
def test_resync_happens_before_the_cycle_body(name):
    s = _src(name)
    i = s.index("while True:")
    assert "resync_clock" in s[i:i + 400], "resync must be the first thing each cycle"


def test_resilience_helpers_are_not_duplicated():
    """Two copies of a fix is how one of them silently rots. Every daemon must
    delegate to exchange_resilience rather than carry its own copy."""
    import glob
    owners = []
    for path in glob.glob(os.path.join(ROOT, "*.py")):
        name = os.path.basename(path)
        if name == "exchange_resilience.py":
            continue
        src = open(path).read()
        # a real definition, not an import alias or a wrapper that delegates
        if "def dns_session_params" in src or "def resync_clock(" in src:
            owners.append(name)
        if "aiohttp.ThreadedResolver()" in src:
            owners.append(name)
    assert not owners, f"resilience logic duplicated in: {sorted(set(owners))}"


def test_every_daemon_uses_the_shared_resilience_module():
    for name in DAEMONS + ["hybrid_allocator.py", "delisting_exit_paper_trader.py",
                           "spread_mm_paper.py"]:
        s = _src(name)
        assert "exchange_resilience" in s, f"{name} does not use exchange_resilience"
