#!/usr/bin/env python3
"""USDT/EGP P2P spread — does the 1.4% persist, or was it a lucky snapshot?

WHY THIS EXISTS
---------------
On 2026-09-13 a single reading of Binance P2P showed USDT/EGP buying at 51.70
and selling at 52.43 — a 1.41% round-trip spread. That is the only mechanism
found in this project whose arithmetic reaches $1/hour without six figures:
profit is capital x spread x cycles, so ~$400 turned over a few times a day
clears the target where $192,700 of yield capital would be needed instead.

It is also ONE SNAPSHOT, and this repo has an expensive lesson about those.
delta_neutral_yield_scan.py ranked candidates on a snapshot and 11 of 25 were
NEGATIVE on 90-day history; the scan now ranks on the mean and says so in its
own docstring. A spread is not an edge until it persists.

So this records the book on a schedule and reports the DISTRIBUTION, not the
latest number. What matters for a turnover business is not the best spread ever
seen but the spread available at a typical moment — specifically the lower
quantiles, because those are the ones you will actually trade through.

WHAT IT DOES NOT TELL YOU
-------------------------
Persistence of the QUOTED spread is necessary, not sufficient. It cannot
measure:
  - whether a counterparty will actually fill you at the top-of-book price
  - payment-rail mismatch: the best buy ad may be InstaPay while the best sell
    ad is Vodafone Cash, and you need both rails to capture the pair
  - fraud, chargebacks and reversed transfers
  - the regulatory and bank-account risk of high-volume crypto-linked transfers
  - that capturing the spread means being the MAKER; as a taker you PAY it
Those are operator questions. This answers exactly one: is the spread real and
stable enough to be worth the rest of the diligence?

Read-only. No credentials, no orders, no money.

Usage:
  python3 p2p_spread_monitor.py sample     # record one observation
  python3 p2p_spread_monitor.py report     # distribution so far
  python3 p2p_spread_monitor.py watch      # sample every SAMPLE_MIN minutes
Env: P2P_FIAT (EGP)  P2P_ASSET (USDT)  P2P_SAMPLE_MIN (15)  P2P_STATE
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
STATE = os.getenv("P2P_STATE", "logs/p2p_spread_history.jsonl")
SAMPLE_MIN = float(os.getenv("P2P_SAMPLE_MIN", "15"))
PAGES = int(os.getenv("P2P_PAGES", "5"))   # 20 ads/page, both sides
# The account's own size, in USD. Everything the operator can actually trade is
# gated by this: an ad whose minimum exceeds it is a price on a screen, not an
# opportunity. Recorded per observation so the history stays honest if it changes.
CAPITAL_USD = float(os.getenv("P2P_CAPITAL_USD", "60"))
TARGET_PER_HOUR = float(os.getenv("P2P_TARGET_PER_HOUR", "1.0"))
URL = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"


def _ads(trade_type: str, rows: int = 20, pages: int = 1) -> list[dict]:
    out: list[dict] = []
    for page in range(1, pages + 1):
        body = json.dumps({"page": page, "rows": rows, "payTypes": [], "asset": ASSET,
                           "tradeType": trade_type, "fiat": FIAT,
                           "publisherType": None}).encode()
        req = urllib.request.Request(URL, data=body, headers={
            "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            got = json.loads(r.read().decode()).get("data") or []
        if not got:
            break
        out += got
        if page < pages:
            time.sleep(0.4)
    return out


def _q(xs: list[float], p: float) -> float:
    """Quantile of a list, empty-safe."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(int(p * (len(s) - 1)), len(s) - 1)]


def _reachable(ads: list[dict], capital_fiat: float, best):
    """The best price among ads this account is big enough to actually trade.

    The distinction the first version of this file missed. Top-of-book is the
    number a screenshot shows; it is frequently posted by an ad with a 9,000 EGP
    minimum, which a $60 account cannot touch. Measured 2026-09-29: the median
    minimum on the book is 5,000 EGP (~$95) but the 5th percentile is 70 EGP
    (~$1), and 74 of 200 ads accept under $25 — so the reachable book exists,
    it is just not the top of it. Returns None when nothing is reachable.
    """
    ok = [a for a in ads
          if float(a["adv"]["minSingleTransAmount"]) <= capital_fiat]
    return best(ok, key=lambda a: float(a["adv"]["price"])) if ok else None


def sample() -> dict | None:
    """One observation of both sides of the book.

    Records the MINIMUM ORDER SIZE alongside the price, because a spread you
    cannot reach is not a spread: at $60 this account sits below the smallest
    ad on the book and can place no trade at all.
    """
    try:
        buys, sells = _ads("BUY", pages=PAGES), _ads("SELL", pages=PAGES)
    except Exception as e:
        print(f"[p2p] fetch failed: {str(e)[:90]}")
        return None
    if not buys or not sells:
        print("[p2p] empty book — not recording")
        return None

    def px(a):
        return float(a["adv"]["price"])

    def minamt(a):
        return float(a["adv"]["minSingleTransAmount"])

    best_buy = min(buys, key=px)      # cheapest price we can BUY at
    best_sell = max(sells, key=px)    # highest price we can SELL at
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "buy": px(best_buy), "sell": px(best_sell),
        "buy_min_fiat": minamt(best_buy), "sell_min_fiat": minamt(best_sell),
        "spread_pct": round((px(best_sell) - px(best_buy)) / px(best_buy) * 100, 4),
        # Medians are the honest middle of the book; top-of-book can be one
        # outlier ad with an unreachable minimum.
        #
        # Over the FIRST PAGE ONLY, deliberately. On 2026-09-29 the fetch went
        # from 1 page to PAGES so reachability and throughput could be measured,
        # and a median taken over the deeper book is a different statistic: page
        # 5 of the BUY side is priced far above page 1, so the same market read
        # -1.199% instead of +0.477%. 1,192 earlier observations are medians of
        # the top 20, and a series whose definition changes silently underneath
        # it is worse than no series. Depth is used for reach, not for this.
        "buy_median": round(statistics.median(px(a) for a in buys[:20]), 4),
        "sell_median": round(statistics.median(px(a) for a in sells[:20]), 4),
        "n_buy": len(buys), "n_sell": len(sells),
    }
    row["spread_median_pct"] = round(
        (row["sell_median"] - row["buy_median"]) / row["buy_median"] * 100, 4)

    # --- what THIS account can reach, and how fast this market actually moves --
    # Added 2026-09-29. The spread was well measured and the two things that
    # decide whether it is income were not: whether a $60 account can reach it,
    # and how many round trips a day this market really supports. Both are in
    # the same payload that was already being fetched.
    fiat_cap = CAPITAL_USD * row["buy_median"]
    rb, rs = _reachable(buys, fiat_cap, min), _reachable(sells, fiat_cap, max)
    row["capital_usd"] = CAPITAL_USD
    if rb and rs:
        row["buy_reach"], row["sell_reach"] = px(rb), px(rs)
        row["spread_reach_pct"] = round((px(rs) - px(rb)) / px(rb) * 100, 4)
    else:
        row["buy_reach"] = row["sell_reach"] = row["spread_reach_pct"] = None
    row["n_reach"] = sum(1 for a in buys + sells
                         if minamt(a) <= fiat_cap)
    row["min_fiat_floor"] = min(minamt(a) for a in buys + sells)

    # Binance charges the ADVERTISER in most markets; zero here is worth
    # recording rather than assuming, because it is the whole net spread.
    def fee(a, k):
        try:
            return float(a["adv"].get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    row["maker_fee_p50"] = round(_q([fee(a, "commissionRate")
                                     for a in buys + sells], 0.50), 6)
    row["taker_fee_p50"] = round(_q([fee(a, "takerCommissionRate")
                                     for a in buys + sells], 0.50), 6)

    # Throughput of the merchants already doing this, from their own 30-day
    # order counts. Two orders make one round trip, so this is halved into
    # round trips per day — the multiplier that turns a spread into income.
    orders = [int(a["advertiser"].get("monthOrderCount") or 0)
              for a in buys + sells
              if (a["advertiser"].get("monthOrderCount") or 0) > 0]
    row["orders30_p50"] = _q(orders, 0.50)
    row["orders30_p90"] = _q(orders, 0.90)
    row["orders30_max"] = max(orders) if orders else 0
    row["n_merchant"] = sum(1 for a in buys + sells
                            if a["advertiser"].get("userType") == "merchant")
    row["n_ads"] = len(buys) + len(sells)

    # Rail mismatch was listed as unmeasurable in this file's own docstring. It
    # is measurable: count the payment methods quoted on BOTH sides, because a
    # spread you can only reach through two different rails is not capturable.
    def rails(ads):
        return {m.get("tradeMethodName") or m.get("identifier")
                for a in ads for m in (a["adv"].get("tradeMethods") or [])}
    row["rails_both"] = sorted(rails(buys) & rails(sells))
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "a") as f:
        f.write(json.dumps(row) + "\n")
    reach = ("n/a" if row["spread_reach_pct"] is None
             else f"{row['spread_reach_pct']:+.3f}%")
    print(f"[p2p {row['ts'][11:16]}] buy {row['buy']:.2f} sell {row['sell']:.2f} "
          f"top {row['spread_pct']:+.3f}%  median {row['spread_median_pct']:+.3f}%  "
          f"reachable@${CAPITAL_USD:.0f} {reach} ({row['n_reach']} ads)  "
          f"merchants {row['orders30_p50']:.0f} orders/30d")
    return row


def report() -> int:
    try:
        rows = [json.loads(l) for l in open(STATE) if l.strip()]
    except FileNotFoundError:
        print(f"No history yet at {STATE}. Run: p2p_spread_monitor.py sample")
        return 1
    if not rows:
        print("No observations recorded yet.")
        return 1
    tops = [r["spread_pct"] for r in rows]
    meds = [r.get("spread_median_pct", r["spread_pct"]) for r in rows]
    span_h = (datetime.fromisoformat(rows[-1]["ts"]) -
              datetime.fromisoformat(rows[0]["ts"])).total_seconds() / 3600

    def q(xs, p):
        s = sorted(xs)
        return s[min(int(len(s) * p), len(s) - 1)]

    print("=" * 70)
    print(f"{ASSET}/{FIAT} P2P SPREAD — {len(rows)} observations over {span_h:.1f}h")
    print("=" * 70)
    for label, xs in (("top-of-book", tops), ("median-of-book", meds)):
        print(f"  {label:<16} mean {statistics.mean(xs):+.3f}%   "
              f"p10 {q(xs,0.10):+.3f}%   p50 {q(xs,0.50):+.3f}%   "
              f"p90 {q(xs,0.90):+.3f}%   min {min(xs):+.3f}%")
    # What the operator actually earns depends on the TYPICAL spread, not the best.
    typical = q(meds, 0.50)
    print()
    print(f"  A turnover business earns the TYPICAL spread, not the best one.")
    print(f"  At the median-of-book p50 of {typical:+.3f}%, $1/hour needs "
          f"${100/typical if typical>0 else float('inf'):,.0f} of turnover per hour.")
    reach = statistics.median(r["buy_min_fiat"] for r in rows)
    print(f"  Median minimum order on the buy side: {reach:,.0f} {FIAT}"
          f"  (~${reach/statistics.median(r['buy'] for r in rows):,.0f})")
    if span_h < 48:
        print(f"\n  ⚠️  {span_h:.1f}h of data. This repo's own history says a snapshot")
        print("      misleads: 11 of 25 delta-neutral candidates looked positive on a")
        print("      snapshot and were negative on 90-day history. Not a verdict yet.")
    print("=" * 70)
    return 0


def frontier() -> int:
    """Capital required for the operator's target, from measured numbers only.

    The whole project has computed income as capital x rate, which is why every
    answer to "$1/hour" came back six figures: at the 8.60% Binance ceiling it
    needs $101,860. A turnover business has a third term:

        income/day = capital x spread x round_trips/day

    Velocity was never measured, so it was implicitly 1. It is not 1: the
    merchants in this market publish their own 30-day order counts, and the
    median runs ~286 orders (≈4.8 round trips/day), the 90th percentile ~992.
    That term is worth two orders of magnitude of capital, which is the entire
    difference between "impossible" and "arithmetically reachable".

    Every input below is measured, not assumed. What this does NOT measure is
    whether the operator can supply the labour, and that is the real constraint
    — see the warnings it prints.
    """
    try:
        rows = [json.loads(l) for l in open(STATE) if l.strip()]
    except FileNotFoundError:
        print(f"No history at {STATE}. Run: p2p_spread_monitor.py sample")
        return 1
    rows = [r for r in rows if r.get("spread_median_pct") is not None]
    if not rows:
        print("No usable observations yet.")
        return 1

    span_d = (datetime.fromisoformat(rows[-1]["ts"]) -
              datetime.fromisoformat(rows[0]["ts"])).total_seconds() / 86400
    reach = [r["spread_reach_pct"] for r in rows
             if r.get("spread_reach_pct") is not None]
    meds = [r["spread_median_pct"] for r in rows]
    orders = [r["orders30_p50"] for r in rows if r.get("orders30_p50")]
    o90 = [r["orders30_p90"] for r in rows if r.get("orders30_p90")]
    target_day = TARGET_PER_HOUR * 24

    print("=" * 72)
    print(f"{ASSET}/{FIAT} TURNOVER FRONTIER — ${TARGET_PER_HOUR:.2f}/hour "
          f"(= ${target_day:.0f}/day)")
    print(f"{len(rows)} observations over {span_d:.1f} days")
    print("=" * 72)

    print("  MEASURED INPUTS")
    print(f"    spread, median-of-book      p50 {_q(meds,0.50):+.3f}%   "
          f"p10 {_q(meds,0.10):+.3f}%")
    if reach:
        print(f"    spread reachable at ${CAPITAL_USD:.0f}   p50 "
              f"{_q(reach,0.50):+.3f}%   p10 {_q(reach,0.10):+.3f}%  "
              f"({len(reach)} obs)")
    else:
        print(f"    spread reachable at ${CAPITAL_USD:.0f}   NOT YET RECORDED "
              "— re-run `sample` on the current code")
    if orders:
        print(f"    merchant throughput        p50 {_q(orders,0.50):.0f} "
              f"orders/30d = {_q(orders,0.50)/60:.1f} round trips/day")
        print(f"                               p90 {_q(o90,0.50):.0f} "
              f"orders/30d = {_q(o90,0.50)/60:.1f} round trips/day")
    fees = [r.get("maker_fee_p50", 0) for r in rows if "maker_fee_p50" in r]
    if fees:
        print(f"    advertiser fee             {_q(fees,0.50)*100:.4f}% "
              "(measured, not assumed — it is the entire net spread)")

    # The frontier itself. Rows are spreads, columns are velocities, cells are
    # the capital that reaches the target. Nothing here is a projection: both
    # axes are quantiles of what was recorded.
    spreads = [("reachable p10", _q(reach, 0.10) if reach else None),
               ("reachable p50", _q(reach, 0.50) if reach else None),
               ("median-of-book p50", _q(meds, 0.50))]
    vels = [("median merchant", _q(orders, 0.50) / 60 if orders else 4.8),
            ("p90 merchant", _q(o90, 0.50) / 60 if o90 else 16.5),
            ("1 round trip/day", 1.0)]
    print()
    print("  CAPITAL NEEDED FOR THE TARGET")
    print(f"    {'spread used':<27}" + "".join(f"{n:>20}" for n, _ in vels))
    for label, sp in spreads:
        if not sp or sp <= 0:
            print(f"    {label:<22}" + "".join(f"{'n/a':>20}" for _ in vels))
            continue
        cells = [f"{'$' + format(target_day / (sp / 100 * v), ',.0f'):>20}"
                 for _, v in vels]
        print(f"    {label + f' ({sp:+.2f}%)':<27}" + "".join(cells))
    print()
    print("  For comparison, the same target from YIELD alone at the 8.60%")
    print(f"  Binance ceiling: ${target_day * 365 / 0.086:,.0f}.")

    print()
    print("  WHAT THIS DOES NOT SAY")
    print("    - A round trip is TWO manual bank transfers plus a chat. The")
    print("      median merchant's 4.8/day is ~10 transfers a day, every day.")
    print("      That is a job, and it breaks the standing autonomy mandate:")
    print("      there is no public order API for a non-merchant account.")
    print("    - The spread is QUOTED. As the maker you earn it and wait; these")
    print("      numbers assume you fill, which is not yet measured.")
    print("    - EGP inventory decays. Measured over this window the pound")
    print("      weakened, so every hour holding fiat instead of USDT is a cost")
    print("      the spread has to cover — see wealth_desk.py.")
    print("    - Bank and regulatory risk on high-volume crypto-linked")
    print("      transfers in this market is real and is not modelled here.")
    if span_d < 30:
        print(f"    - {span_d:.1f} days. Throughput has only just started being")
        print("      recorded; the velocity column firms up with more history.")
    print("=" * 72)
    return 0


def main() -> int:
    cmd = (sys.argv[1:] or ["sample"])[0]
    if cmd == "frontier":
        return frontier()
    if cmd == "report":
        return report()
    if cmd == "watch":
        print(f"[p2p] sampling {ASSET}/{FIAT} every {SAMPLE_MIN:.0f} min — Ctrl-C to stop")
        while True:
            sample()
            time.sleep(SAMPLE_MIN * 60)
    sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
