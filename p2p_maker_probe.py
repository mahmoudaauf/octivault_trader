#!/usr/bin/env python3
"""P2P maker fill rate — the last unmeasured variable, measured without money.

WHY THIS EXISTS
---------------
The operator's target is $4/week on a $60.94 balance: 341%/yr, against a 7.55%
autonomous yield ceiling that pays $0.088/week. Every capital-bound route in this
repo is closed, and the taker side of the P2P spread closed too — requiring both
legs to settle on ONE payment rail, three snapshots gave 0 of 749/755/654
combinations positive, best -0.411%. The apparent ~1% spread IS the rail toll.

What survives is the MAKER side: post your own ad, set your own rail and price,
and be paid by takers who need it. You collect the toll instead of paying it.
The arithmetic is the first in this project that reaches the target at a
plausible activity level — at 0.8% a round trip, $60.94 needs 1.17 round trips a
day, against a median market merchant's 4.8.

But "post an ad and get filled" contains an assumption this repo has been burned
by before: that the fill happens. A quoted spread is not a trade, and an ad at
the back of the book is a price nobody takes. Spread market-making died exactly
here — 701 paper fills to learn that the wide-spread tail loses 0.59% net to
adverse selection.

WHAT IT MEASURES
----------------
Fills, from the public book, with no account and no money at risk. Each ad
carries `advNo` and `surplusAmount`. Sampling the book on a schedule and diffing
by advNo:

  - surplus DECREASED **and** the advertiser's 30-day order count went UP
        -> a fill. Two independent witnesses to the same event.
  - surplus DECREASED and the order count did NOT move
        -> the advertiser edited the ad size down. NOT a fill. A surplus drop
           alone cannot tell these apart, and the first version of this file
           counted them together: one 1,133 USDT "fill" in a 2-minute window
           annualised to 28,593 USDT/day per ad, which is what an unchallenged
           assumption looks like from the outside.
  - surplus INCREASED                -> the advertiser topped up
  - ad DISAPPEARED                   -> filled OR cancelled, genuinely ambiguous

Only the first is counted. The rest are recorded separately and never added in,
because "it shrank so it must have sold" is the kind of assumption that produces
a confident wrong answer. Undercounting is the right bias: it can make a viable
business look unviable, not the reverse.

THE QUESTION IT ANSWERS
-----------------------
How much volume is absorbed per hour by an ad at a given position in the book?
That is what decides whether the operator's own ad fills, and it is reported by
rank bucket, because the whole decision is how aggressively to price.

To clear $4/week at 0.8% a round trip, ~$71/day has to be absorbed on EACH side
at a price you set. The report says whether the book does that at a rank you
could hold.

WHAT IT CANNOT TELL YOU
-----------------------
  - Whether a non-merchant account may post ads for this pair at all. Checkable
    only in the app, by the operator.
  - Adverse selection: the takers who hit you first may be the ones you least
    want, which is precisely what killed spread market-making here.
  - That YOUR ad fills like an incumbent's. Advertisers carry completion counts
    and badges, and a new one with no history is not the same product.
  - The labour: every fill is a manual bank transfer and a chat.
  - Bank and regulatory risk on crypto-linked transfers in this market.

Read-only. Public endpoint, no credentials, no orders, no money.

Usage:
  python3 p2p_maker_probe.py sample    # one observation, diff against the last
  python3 p2p_maker_probe.py report    # fill rate by book rank
  python3 p2p_maker_probe.py watch     # sample every PROBE_SAMPLE_SEC seconds
Env: P2P_FIAT(EGP) P2P_ASSET(USDT) PROBE_SAMPLE_SEC(120) PROBE_PAGES(3)
     P2P_CAPITAL_USD(60) PROBE_TARGET_WEEK(4.0) PROBE_SPREAD_PCT(0.8)
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

FIAT = os.getenv("P2P_FIAT", "EGP")
ASSET = os.getenv("P2P_ASSET", "USDT")
PAGES = int(os.getenv("PROBE_PAGES", "3"))
SAMPLE_SEC = float(os.getenv("PROBE_SAMPLE_SEC", "120"))
CAPITAL_USD = float(os.getenv("P2P_CAPITAL_USD", "60"))
TARGET_WEEK = float(os.getenv("PROBE_TARGET_WEEK", "4.0"))
SPREAD_PCT = float(os.getenv("PROBE_SPREAD_PCT", "0.8"))
# Shortest gap between two snapshots that may be used as a RATE denominator.
# Two manual samples 4 seconds apart turned one fill into 28,593 USDT/day per ad.
# The diff is still recorded — a fill is a fact — but the interval is not trusted
# as a measure of time, so such rows are excluded from per-hour arithmetic.
MIN_DIFF_MIN = float(os.getenv("PROBE_MIN_DIFF_MIN", "0.5"))

SNAP = os.getenv("PROBE_SNAP", "logs/p2p_maker_snapshot.json")
FILLS = os.getenv("PROBE_FILLS", "logs/p2p_maker_fills.jsonl")
URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"

# Ranks are bucketed rather than reported per-position: rank 3 and rank 4 are not
# meaningfully different decisions, and a per-rank series at this sample size is
# noise wearing a number's clothes.
BUCKETS = ((0, 3, "top 3"), (3, 10, "4-10"), (10, 25, "11-25"), (25, 10_000, "26+"))


def _bucket(rank: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= rank < hi:
            return name
    return "26+"


def _ads(trade_type: str) -> list[dict]:
    out: list[dict] = []
    for page in range(1, PAGES + 1):
        body = json.dumps({"page": page, "rows": 20, "payTypes": [], "asset": ASSET,
                           "tradeType": trade_type, "fiat": FIAT,
                           "publisherType": None}).encode()
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            got = json.loads(r.read().decode()).get("data") or []
        if not got:
            break
        out += got
        if page < PAGES:
            time.sleep(0.35)
    return out


def _snapshot() -> dict | None:
    """The book as {advNo: {...}}, ranked per side by price competitiveness."""
    try:
        buys, sells = _ads("BUY"), _ads("SELL")
    except Exception as e:
        print(f"[probe] fetch failed: {str(e)[:90]}")
        return None
    if not buys or not sells:
        print("[probe] empty book — not recording")
        return None

    def rank(ads: list[dict], best_is_low: bool) -> list[dict]:
        return sorted(ads, key=lambda a: float(a["adv"]["price"]),
                      reverse=not best_is_low)

    out: dict[str, dict] = {}
    for side, ads in (("BUY", rank(buys, True)), ("SELL", rank(sells, False))):
        for i, a in enumerate(ads):
            d = a["adv"]
            try:
                out[d["advNo"]] = {
                    "side": side, "rank": i, "px": float(d["price"]),
                    "surplus": float(d.get("surplusAmount") or 0),
                    "min_fiat": float(d.get("minSingleTransAmount") or 0),
                    "orders": int(a["advertiser"].get("monthOrderCount") or 0),
                }
            except (TypeError, ValueError, KeyError):
                continue
    return {"ts": datetime.now(timezone.utc).isoformat(), "ads": out}


def sample() -> dict | None:
    now = _snapshot()
    if not now:
        return None
    try:
        with open(SNAP) as f:
            prev = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        prev = None

    os.makedirs(os.path.dirname(SNAP) or ".", exist_ok=True)
    with open(SNAP, "w") as f:
        json.dump(now, f)

    if not prev:
        print(f"[probe] first snapshot — {len(now['ads'])} ads recorded, "
              "nothing to diff yet")
        return None

    dt_h = ((datetime.fromisoformat(now["ts"]) -
             datetime.fromisoformat(prev["ts"])).total_seconds() / 3600)
    if dt_h <= 0:
        print("[probe] non-positive interval — skipping diff")
        return None

    filled, topped, vanished, shrunk = [], 0, 0, 0
    for adv_no, was in prev["ads"].items():
        nowad = now["ads"].get(adv_no)
        if nowad is None:
            vanished += 1
            continue
        delta = was["surplus"] - nowad["surplus"]
        d_orders = nowad["orders"] - was["orders"]
        if delta > 1e-9 and d_orders > 0:
            # Two witnesses: inventory fell AND the advertiser completed an order.
            filled.append({"advNo": adv_no, "side": was["side"],
                           "rank": was["rank"], "bucket": _bucket(was["rank"]),
                           "px": was["px"], "filled_usdt": round(delta, 8),
                           "n_orders": d_orders, "orders30": was["orders"]})
        elif delta > 1e-9:
            shrunk += 1          # edited down, not sold
        elif delta < -1e-9:
            topped += 1

    # Ads actually seen per bucket, so the per-ad rate divides by a counted
    # number instead of the guess the first version used.
    seen: dict[str, int] = {}
    for a in prev["ads"].values():
        b = _bucket(a["rank"])
        seen[b] = seen.get(b, 0) + 1

    row = {"ts": now["ts"], "hours": round(dt_h, 5),
           "n_prev": len(prev["ads"]), "n_now": len(now["ads"]),
           "fills": filled, "n_topped_up": topped, "n_vanished": vanished,
           "n_shrunk": shrunk, "ads_per_bucket": seen}
    os.makedirs(os.path.dirname(FILLS) or ".", exist_ok=True)
    with open(FILLS, "a") as f:
        f.write(json.dumps(row) + "\n")

    vol = sum(f["filled_usdt"] for f in filled)
    print(f"[probe {now['ts'][11:16]}] {dt_h*60:.1f}min: {len(filled)} confirmed "
          f"fills, {vol:,.1f} {ASSET}  (excluded: {shrunk} shrank without an "
          f"order, {topped} topped up, {vanished} vanished)")
    return row


def report() -> int:
    try:
        rows = [json.loads(l) for l in open(FILLS) if l.strip()]
    except FileNotFoundError:
        print(f"No fill history at {FILLS}. Run: p2p_maker_probe.py sample")
        return 1
    if not rows:
        print("No diffs recorded yet — needs at least two samples.")
        return 1

    # Rows whose interval is too short to be a denominator are dropped from the
    # rate arithmetic, not from the record.
    usable = [r for r in rows if r["hours"] * 60 >= MIN_DIFF_MIN]
    dropped = len(rows) - len(usable)
    if not usable:
        print(f"{len(rows)} diff(s) recorded but none span {MIN_DIFF_MIN:.1f} min "
              "— nothing can be expressed as a rate yet.")
        return 1

    hours = sum(r["hours"] for r in usable)
    per_bucket: dict[str, dict] = {}
    bucket_ads: dict[str, list] = {}
    for r in usable:
        for b, n in (r.get("ads_per_bucket") or {}).items():
            bucket_ads.setdefault(b, []).append(n)
        for f in r["fills"]:
            b = per_bucket.setdefault(f["bucket"], {"usdt": 0.0, "n": 0})
            b["usdt"] += f["filled_usdt"]
            b["n"] += 1
    total = sum(b["usdt"] for b in per_bucket.values())
    vanished = sum(r["n_vanished"] for r in usable)
    topped = sum(r["n_topped_up"] for r in usable)
    shrunk = sum(r.get("n_shrunk", 0) for r in usable)
    ads = statistics.median([r["n_now"] for r in usable]) or 1

    print("=" * 72)
    print(f"{ASSET}/{FIAT} P2P MAKER FILL RATE — {len(rows)} diffs over {hours:.1f}h")
    print("=" * 72)
    print(f"  CONFIRMED fills {total:,.1f} {ASSET} across ~{ads:.0f} live ads")
    print("  a fill needs TWO witnesses: inventory fell AND the advertiser's")
    print("  order count rose. Excluded, never counted:")
    print(f"    {shrunk} ads shrank with no new order  (edited down, not sold)")
    print(f"    {vanished} ads vanished                (filled OR cancelled)")
    print(f"    {topped} ads topped up")
    if dropped:
        print(f"    {dropped} diff(s) spanned under {MIN_DIFF_MIN:.1f} min "
              "— too short to be a rate")
    print()
    print(f"  {'book position':<16}{'fills':>8}{'volume':>14}{'per ad per day':>18}")
    for _lo, _hi, name in BUCKETS:
        b = per_bucket.get(name)
        if not b:
            print(f"  {name:<16}{'-':>8}{'-':>14}{'-':>18}")
            continue
        # Per-ad-per-day is what the operator's own ad would inherit sitting in
        # this bucket. The divisor is the number of ads actually OBSERVED there,
        # averaged over the samples — not inferred from the book size, which is
        # how the first version produced 28,593 U/day.
        obs = bucket_ads.get(name) or [1]
        in_bucket = max(1.0, statistics.mean(obs))
        print(f"  {name:<16}{b['n']:>8}{b['usdt']:>12,.1f} U"
              f"{b['usdt']/in_bucket/(hours/24):>16,.1f} U")

    need_day = TARGET_WEEK / 7 / (SPREAD_PCT / 100)
    print()
    print(f"  THE TARGET: ${TARGET_WEEK:.2f}/week at {SPREAD_PCT:.1f}% a round trip")
    print(f"  needs ${need_day:,.0f}/day absorbed on EACH side at a price you set")
    print(f"  (= {need_day/CAPITAL_USD:.2f} round trips/day of your ${CAPITAL_USD:.0f})")

    if hours < 24:
        print()
        print(f"  ⚠️  {hours:.1f}h of diffs. Fill rate varies by hour of day —")
        print("      Egyptian retail flow is not flat at 04:00. Needs a few full")
        print("      days before the per-bucket numbers mean anything, and this")
        print("      repo's own history says snapshots mislead: spread")
        print("      market-making needed 701 fills to reach its verdict.")
    print("=" * 72)
    return 0


def main() -> int:
    cmd = (sys.argv[1:] or ["sample"])[0]
    if cmd == "report":
        return report()
    if cmd == "watch":
        print(f"[probe] watching {ASSET}/{FIAT} every {SAMPLE_SEC:.0f}s "
              f"({PAGES} pages/side) — Ctrl-C to stop")
        while True:
            try:
                sample()
            except Exception as e:                      # never die on one bad cycle
                print(f"[probe] cycle failed: {str(e)[:90]}")
            time.sleep(SAMPLE_SEC)
    sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
