#!/usr/bin/env python3
"""Income by SOURCE, against the $1.00/hour goal — the two-axis scoreboard.

WHY THIS EXISTS
---------------
Every instrument in this repo measures return on capital, and on $60 that axis
is finished: 8.53% is the best verified rate available, which is $0.0005/hour.
Reaching $1.00/hour that way needs ~$102,700. No technique changes it, because
profit is capital x return and the return side is maxed.

There is a second axis, and nothing was watching it. Some income does not scale
with what we hold at all:

    referral / affiliate commission   a share of OTHER people's trading fees,
                                      computed hourly, paid every 6h, identical
                                      whether our balance is $60 or $0
    airdrops, Launchpool rewards      paid per qualifying account
    one-off distributions             token swaps, migrations

$1.00/hour on THAT axis costs roughly 300 moderately active referred traders
(~$17.5M/yr of referred volume at the 50% commission tier) and zero capital.
That is not cheap, but it is a price in reach rather than an impossibility — and
it is the only honest answer to "stop being bound to capital".

So this reports both axes separately and never blends them. Blending would let
$0.0005/hour of yield and $0.00/hour of commission average into a single number
that hides which lever is actually moving.

WHERE THE DATA COMES FROM
-------------------------
`/sapi/v1/asset/assetDividend` is the channel every non-trade credit arrives
through — earn bonuses, referral kickbacks, airdrops, distributions — each row
carrying an `enInfo` label. Rows are classified by that label, and anything
unrecognised is reported in an UNCLASSIFIED bucket rather than silently dropped
or silently counted: a new income type we have never seen is exactly the thing
worth noticing, and guessing its axis would corrupt both.

Read-only. Moves no money.

Usage:
  python3 income_sources.py              # last 30 days
  python3 income_sources.py --days 90
  python3 income_sources.py --target 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("BINANCE_API_KEY")
SECRET = os.getenv("BINANCE_API_SECRET")
BASE = "https://api.binance.com"

# Label fragments, lowercased. Order matters only in that the first match wins.
# CAPITAL_BOUND income is a return ON what we hold; it scales with the balance
# and is therefore capped by it. CAPITAL_FREE income is paid for something other
# than holding — a referral, a qualifying account, a distribution — and does not
# scale with our balance at all. That distinction is the whole point of the file.
CAPITAL_BOUND = ("flexible", "simple earn", "locked", "staking", "savings",
                 "bfusd", "eth2", "launchpool staking")
CAPITAL_FREE = ("referral", "commission", "kickback", "rebate", "affiliate",
                "airdrop", "launchpool", "megadrop", "hodler", "distribution",
                "token swap", "bonus voucher", "reward voucher", "campaign",
                "promotion")


def _signed(path: str, params: dict | None = None):
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 10000
    qs = urllib.parse.urlencode(p)
    qs += "&signature=" + hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"X-MBX-APIKEY": KEY})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def _price(asset: str) -> float | None:
    """USD price, or None. Stablecoins are taken as $1 without a round trip."""
    a = asset.upper()
    if a in ("USDT", "USDC", "USD1", "FDUSD", "TUSD", "DAI", "USDP", "BUSD", "BFUSD"):
        return 1.0
    for quote in ("USDT", "USDC"):
        try:
            with urllib.request.urlopen(
                    f"{BASE}/api/v3/ticker/price?symbol={a}{quote}", timeout=15) as r:
                return float(json.loads(r.read().decode())["price"])
        except Exception:
            continue
    return None


def classify(label: str) -> str:
    low = (label or "").lower()
    for frag in CAPITAL_BOUND:
        if frag in low:
            return "capital_bound"
    for frag in CAPITAL_FREE:
        if frag in low:
            return "capital_free"
    return "unclassified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--target", type=float,
                    default=float(os.getenv("PROFIT_TARGET_HOURLY_USD", "1.00")))
    a = ap.parse_args()

    since = (time.time() - a.days * 86400) * 1000
    try:
        rows = _signed("/sapi/v1/asset/assetDividend", {"limit": 500}).get("rows", [])
    except Exception as e:
        print(f"Could not read distribution history: {str(e)[:120]}")
        return 1

    buckets: dict[str, dict] = defaultdict(lambda: {"usd": 0.0, "n": 0, "labels": set()})
    unpriced = []
    for r in rows:
        try:
            if float(r.get("divTime", 0)) < since:
                continue
            asset, amt = str(r.get("asset", "")).upper(), float(r.get("amount", 0) or 0)
        except (TypeError, ValueError):
            continue
        label = r.get("enInfo", "")
        kind = classify(label)
        px = _price(asset)
        if px is None:
            unpriced.append((asset, amt, label))
            continue
        b = buckets[kind]
        b["usd"] += amt * px
        b["n"] += 1
        b["labels"].add(label[:42])

    hours = a.days * 24.0
    print("=" * 72)
    print(f"INCOME BY SOURCE — last {a.days} days — goal ${a.target:.2f}/hour")
    print("=" * 72)
    for kind, title in (("capital_bound", "CAPITAL-BOUND (return on what we hold)"),
                        ("capital_free", "CAPITAL-FREE (paid regardless of balance)"),
                        ("unclassified", "UNCLASSIFIED (new type — check before trusting)")):
        b = buckets.get(kind)
        usd = b["usd"] if b else 0.0
        per_h = usd / hours
        print(f"\n  {title}")
        print(f"    credited        ${usd:>12.6f}   ({b['n'] if b else 0} rows)")
        print(f"    per hour        ${per_h:>12.6f}   = {per_h/a.target*100:.4f}% of goal")
        if b and b["labels"]:
            for lab in sorted(b["labels"]):
                print(f"      · {lab}")

    bound = buckets.get("capital_bound", {}).get("usd", 0.0) / hours
    free = buckets.get("capital_free", {}).get("usd", 0.0) / hours
    print("\n" + "-" * 72)
    print(f"  total            ${bound+free:>12.6f}/hour   "
          f"vs goal ${a.target:.2f}  ({(bound+free)/a.target*100:.4f}%)")
    print("\n  THE PRICE OF THE GOAL ON EACH AXIS")
    # Capital-bound: the rate is already the best available, so only size is free.
    print(f"    capital-bound : ${a.target*8760/0.0853:>12,.0f} of capital at 8.53%")
    # Capital-free: commission is a share of OTHER people's fees, so the
    # denominator is their volume, not our balance.
    for rate, lab in ((0.50, "50% tier"), (0.40, "40% tier")):
        vol = a.target * 8760 / rate / 0.001
        print(f"    capital-free  : ${vol:>12,.0f}/yr of referred volume ({lab}) "
              f"≈ {vol/12/5000:,.0f} traders at $5k/mo")
    if unpriced:
        print("\n  NOT PRICED (no USDT/USDC pair — reported, not ignored):")
        for asset, amt, label in unpriced[:8]:
            print(f"    {asset:<8} {amt:>16.8f}  {label[:40]}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
