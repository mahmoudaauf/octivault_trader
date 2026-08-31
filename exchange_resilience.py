#!/usr/bin/env python3
"""Shared resilience helpers for long-lived python-binance daemons.

Every long-running client in this repo has hit the same three failure modes on
this machine. They are collected here so a daemon gets all three by construction
instead of rediscovering them in production:

  1. CLOCK DRIFT — python-binance sets `timestamp_offset` ONCE inside
     AsyncClient.create() and never refreshes it. Local drift (which macOS
     accrues across sleep/wake) eventually pushes every SIGNED request outside
     recvWindow, so they all fail -1021 permanently until restart, while
     UNSIGNED calls keep working. A daemon therefore looks perfectly healthy
     while every authenticated read fails. Observed live 2026-08-23: 12 hours.

  2. DNS — aiohttp defaults to AsyncResolver (aiodns/c-ares), which fixes its
     nameserver list at construction and copes badly with a laptop changing
     networks or waking. Three outages in six days, once ~45 minutes.

  3. STARTUP FRAGILITY — AsyncClient.create() pings Binance, so a DNS failure
     raises and kills the daemon before it starts. Crashing is worse than
     waiting: while the process is dead nothing manages open positions.

hybrid_allocator.py carries its own copies of (1) and (2), predating this
module; they are behaviourally identical. Prefer importing from here in new
code.
"""
from __future__ import annotations

import asyncio
import os
import time


def dns_session_params() -> dict:
    """aiohttp session params using the OS resolver plus a DNS cache.

    ThreadedResolver defers to the OS resolver, which tracks network changes and
    keeps its own cache; ttl_dns_cache then rides out brief resolver failures
    without a lookup at all. Returns a FRESH connector per call — create()
    closes the session (and its connector) when its ping fails, so a retry must
    not reuse one. Degrades to aiohttp defaults rather than blocking startup.
    """
    try:
        import aiohttp
        return {"connector": aiohttp.TCPConnector(
            resolver=aiohttp.ThreadedResolver(), ttl_dns_cache=300, limit=20)}
    except Exception as e:
        print(f"[resilience] DNS hardening unavailable ({str(e)[:60]}) — aiohttp defaults")
        return {}


async def resync_clock(client, label: str = "resilience") -> bool:
    """Re-derive the client's Binance timestamp offset. Cheap (one unsigned
    request); call once per poll BEFORE any signed call."""
    try:
        res = await client.get_server_time()
        before = getattr(client, "timestamp_offset", 0)
        client.timestamp_offset = res["serverTime"] - int(time.time() * 1000)
        if abs(client.timestamp_offset - before) > 500:
            print(f"[{label}] CLOCK-RESYNC offset {before}ms → {client.timestamp_offset}ms")
        return True
    except Exception as e:
        print(f"[{label}] CLOCK-RESYNC failed: {str(e)[:80]}")
        return False


async def create_client_with_retry(async_client_cls, api_key: str = None,
                                   api_secret: str = None, *, testnet: bool = False,
                                   label: str = "resilience", max_delay_s: float = 300.0):
    """Build the client, WAITING OUT a network outage instead of exiting(1).

    Retries with exponential backoff, logging every attempt so a supervisor's
    stall watchdog still sees a live process rather than a wedged one.
    """
    key = api_key if api_key is not None else (os.getenv("BINANCE_API_KEY") or "x")
    secret = api_secret if api_secret is not None else (os.getenv("BINANCE_API_SECRET") or "x")
    delay, attempt = min(15.0, max_delay_s), 0
    while True:
        attempt += 1
        try:
            kwargs = {"session_params": dns_session_params()}
            if testnet:
                kwargs["testnet"] = True
            client = await async_client_cls.create(key, secret, **kwargs)
            if attempt > 1:
                print(f"[{label}] client connected after {attempt} attempts")
            return client
        except Exception as e:
            print(f"[{label}] client create failed (attempt {attempt}): {str(e)[:80]} "
                  f"— retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
