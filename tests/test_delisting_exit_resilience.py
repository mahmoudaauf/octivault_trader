"""Blindness guards for delisting_exit_paper_trader.py.

This daemon runs LIVE-ARMED with real sell authority, and its failure mode is
silent: get_account() is SIGNED so it fails on clock drift, while the article
feed is UNSIGNED plain HTTP and keeps working. The loop marks an article as
`seen` BEFORE checking holdings, and articles are only ever examined once — so
treating a failed account read as "holds nothing" permanently discarded the
delisting notice this strategy exists to act on, while printing pending=0 and
looking perfectly healthy.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mod():
    import delisting_exit_paper_trader as d
    return d


async def test_holdings_unreadable_is_none_not_empty():
    """None (UNKNOWN) must be distinguishable from {} (genuinely holds nothing)."""
    d = _mod()
    c = AsyncMock()
    c.get_account = AsyncMock(side_effect=Exception("APIError(code=-1021) recvWindow"))
    assert await d.get_real_holdings(c) is None


async def test_holdings_empty_account_is_empty_dict_not_none():
    d = _mod()
    c = AsyncMock()
    c.get_account = AsyncMock(return_value={"balances": [
        {"asset": "USDT", "free": "10.0"},          # stable, excluded
        {"asset": "BTC", "free": "0.0"},            # zero, excluded
    ]})
    assert await d.get_real_holdings(c) == {}       # readable AND empty


async def test_holdings_parsed_when_readable():
    d = _mod()
    c = AsyncMock()
    c.get_account = AsyncMock(return_value={"balances": [
        {"asset": "FOO", "free": "12.5"},
        {"asset": "USDT", "free": "9.0"},
    ]})
    h = await d.get_real_holdings(c)
    assert h == {"FOO": 12.5}


async def test_resync_clock_is_called_before_signed_calls():
    """The drift that causes the blindness must be corrected each cycle."""
    from exchange_resilience import resync_clock
    import time as _t
    c = AsyncMock()
    c.timestamp_offset = 0
    c.get_server_time = AsyncMock(return_value={"serverTime": int(_t.time() * 1000) + 30_000})
    assert await resync_clock(c, "test") is True
    assert c.timestamp_offset > 25_000


async def test_resync_failure_never_raises():
    from exchange_resilience import resync_clock
    c = AsyncMock()
    c.timestamp_offset = 0
    c.get_server_time = AsyncMock(side_effect=Exception("network down"))
    assert await resync_clock(c, "test") is False


async def test_client_create_waits_out_an_outage():
    from exchange_resilience import create_client_with_retry
    calls = {"n": 0}

    class _Cls:
        @staticmethod
        async def create(*a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("Could not contact DNS servers")
            return "CLIENT"

    got = await create_client_with_retry(_Cls, "k", "s", label="test", max_delay_s=0.001)
    assert got == "CLIENT" and calls["n"] == 3


async def test_dns_params_use_os_resolver():
    import aiohttp
    from exchange_resilience import dns_session_params
    p = dns_session_params()
    conn = p["connector"]
    try:
        assert isinstance(conn._resolver, aiohttp.ThreadedResolver)
        assert conn._cached_hosts._ttl == 300
    finally:
        await conn.close()


def test_loop_skips_article_scan_when_blind():
    """Source-level guard: the scan must be gated on holdings being readable,
    because consuming an article while blind loses it forever."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "delisting_exit_paper_trader.py")).read()
    scan = src.index("holdings = await get_real_holdings(client)")
    fetch = src.index("articles = fetch_recent_delisting_articles()")
    guard = src.index("if holdings is None:")
    assert scan < guard < fetch, "article fetch must be gated behind the holdings-known check"
