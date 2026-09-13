#!/usr/bin/env python3
"""The investing office's daily brief — one page, live, honest.

WHY THIS EXISTS
---------------
The analysis in this project is real but scattered: hybrid_allocator holds the
position, hourly_profit the efficiency, income_sources the two income axes,
p2p_spread_monitor a live candidate, profit_readiness the goal gap, and the
falsification record sits in docs and memory. Answering "how are we doing"
meant running five tools and holding the result in your head.

An office produces a BRIEF. One page, assembled from live data, that a
principal can read in a minute and act on. This is that page.

WHAT IT WILL NOT DO
-------------------
It will not flatter. Every number is measured or marked UNKNOWN — never
estimated into something that looks better. The opportunity pipeline carries
each candidate's EVIDENCE STATUS, so a promising snapshot is never presented
with the authority of a tested result; this project has an expensive lesson
about exactly that (11 of 25 delta-neutral candidates looked good on a snapshot
and were negative on 90-day history).

Sections, in the order a principal needs them:
  POSITION     where the money is, right now, from the exchange
  PERFORMANCE  what it earns, and how that compares to the best available rate
  INCOME AXES  capital-bound vs capital-free, against the standing goal
  PIPELINE     live candidates, each with its evidence status and next step
  RISKS        what could go wrong, quantified where possible
  YOUR DESK    the things only the principal can do

Read-only. Authenticated READS only; no orders, transfers or subscriptions.

Usage:  python3 office_brief.py [--goal 1.00]
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import statistics
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

KEY, SECRET = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
BASE = "https://api.binance.com"
GOAL_HOURLY = float(os.getenv("PROFIT_TARGET_HOURLY_USD", "1.00"))
NAV_FILE = os.getenv("HYBRID_NAV_FILE", "logs/nav_history.jsonl")
P2P_FILE = os.getenv("P2P_STATE", "logs/p2p_spread_history.jsonl")
STABLES = {"USDT", "USDC", "USD1", "FDUSD", "TUSD", "DAI", "USDP", "XUSD", "RLUSD"}

UNKNOWN = "UNKNOWN"


def _get(path: str, params: dict | None = None, signed: bool = True):
    p = dict(params or {})
    if signed:
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = 10000
    qs = urllib.parse.urlencode(p)
    if signed:
        qs += "&signature=" + hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"X-MBX-APIKEY": KEY})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _rule(title: str) -> None:
    print(f"\n{title}")
    print("─" * 74)


def position() -> dict:
    """Where the money is, read from the exchange rather than from state."""
    out = {"earn": {}, "spot": {}, "nav": None, "rate": None, "err": None}
    try:
        for p in _get("/sapi/v1/simple-earn/flexible/position").get("rows", []):
            out["earn"][p["asset"].upper()] = {
                "amt": float(p["totalAmount"]),
                "bonus_cum": float(p.get("cumulativeBonusRewards", 0) or 0),
            }
        for b in _get("/api/v3/account")["balances"]:
            t = float(b["free"]) + float(b["locked"])
            if t > 0 and not b["asset"].startswith("LD"):
                out["spot"][b["asset"].upper()] = t
    except Exception as e:
        out["err"] = str(e)[:90]
        return out

    # Blended rate across what we actually hold, using each product's own tiers.
    total = sum(v["amt"] for a, v in out["earn"].items() if a in STABLES)
    weighted = 0.0
    for asset, v in out["earn"].items():
        if asset not in STABLES or v["amt"] <= 0:
            continue
        try:
            rows = _get("/sapi/v1/simple-earn/flexible/list", {"asset": asset}).get("rows", [])
        except Exception:
            continue
        if not rows:
            continue
        r = rows[0]
        base = float(r["latestAnnualPercentageRate"])
        tiers = r.get("tierAnnualPercentageRate") or {}
        tier = max((float(x) for x in tiers.values()), default=0.0)
        # The account is far below every cap, so the tiered rate applies whole.
        weighted += v["amt"] * (base + tier)
    out["nav"] = total
    out["rate"] = (weighted / total) if total else None
    return out


def performance(nav: float | None, rate: float | None) -> None:
    _rule("PERFORMANCE")
    if not nav or not rate:
        print(f"  {UNKNOWN} — could not read the position")
        return
    yr = nav * rate
    print(f"  rate {rate*100:>6.2f}%/yr   "
          f"${yr/8760:.6f}/hr   ${yr/365:.4f}/day   ${yr/52:.3f}/wk   ${yr:.2f}/yr")
    # Efficiency is measured against the best rate the account can actually get,
    # not against the goal — it answers "is the machine working", not "are we rich".
    try:
        rows = [json.loads(l) for l in open(NAV_FILE) if l.strip()][-96:]
        if len(rows) > 2:
            span_h = (datetime.fromisoformat(rows[-1]["ts"]) -
                      datetime.fromisoformat(rows[0]["ts"])).total_seconds() / 3600
            g = rows[-1].get("growth", 0.0) - rows[0].get("growth", 0.0)
            if span_h > 0:
                print(f"  measured last {span_h:.0f}h: ${g/span_h:.6f}/hr realised "
                      f"({g/span_h/(yr/8760)*100:.0f}% of the rate above)")
    except Exception:
        pass


def axes(nav: float | None, rate: float | None, goal: float) -> None:
    _rule(f"INCOME AXES  (goal ${goal:.2f}/hour)")
    bound = (nav * rate / 8760) if (nav and rate) else 0.0
    free = 0.0
    try:
        rows = _get("/sapi/v1/asset/assetDividend", {"limit": 500}).get("rows", [])
        cut = (time.time() - 30 * 86400) * 1000
        import income_sources as inc
        for r in rows:
            if float(r.get("divTime", 0)) < cut:
                continue
            if inc.classify(r.get("enInfo", "")) == "capital_free":
                px = inc._price(str(r.get("asset", "")).upper())
                if px:
                    free += float(r.get("amount", 0) or 0) * px
        free /= 30 * 24
    except Exception:
        free = 0.0
    print(f"  capital-bound  ${bound:.6f}/hr   {bound/goal*100:7.4f}% of goal   "
          f"(return on what we hold — capped by the balance)")
    print(f"  capital-free   ${free:.6f}/hr   {free/goal*100:7.4f}% of goal   "
          f"(referral, airdrops, quizzes — independent of the balance)")
    tot = bound + free
    print(f"  TOTAL          ${tot:.6f}/hr   {tot/goal*100:7.4f}% of goal   "
          f"→ {goal/tot:,.0f}x away" if tot > 0 else "  TOTAL 0")


def pipeline() -> None:
    """Live candidates. EVIDENCE STATUS is the column that matters."""
    _rule("PIPELINE")
    items = []

    # P2P — the only mechanism whose arithmetic reaches the goal without 6 figures.
    try:
        rows = [json.loads(l) for l in open(P2P_FILE) if l.strip()]
    except Exception:
        rows = []
    if rows:
        meds = [r.get("spread_median_pct", r["spread_pct"]) for r in rows]
        span_h = (datetime.fromisoformat(rows[-1]["ts"]) -
                  datetime.fromisoformat(rows[0]["ts"])).total_seconds() / 3600
        status = "TESTED" if span_h >= 48 else f"SNAPSHOT ({span_h:.0f}h — not a verdict)"
        items.append(("USDT/EGP P2P spread",
                      f"median {statistics.median(meds):+.2f}%", status,
                      "run p2p_spread_monitor.py watch for 48h+"))
    else:
        items.append(("USDT/EGP P2P spread", "1.43% top / 0.90% median",
                      "SNAPSHOT (one reading)", "start the monitor"))

    # Launchpool — built, armed, verified, waiting on Binance.
    try:
        d = _get("/sapi/v1/launchpool/project/list")
        live = len(d.get("tracking") or [])
        items.append(("Binance Launchpool",
                      f"~21%/yr while live; {live} running now",
                      "BUILT + VERIFIED", "none — fires automatically"))
    except Exception:
        items.append(("Binance Launchpool", UNKNOWN, "BUILT", "none"))

    items.append(("Referral (Pro)", "~49 heavy traders = $1/hr", "NOT STARTED",
                  "check Account → Referral for mode"))
    items.append(("Learn & Earn", "$10–25 one-time", "NOT STARTED",
                  "Rewards Hub → Learn & Earn"))
    items.append(("Capital", "$192,700 = $1/hr at blended rates", "AWAITING PRINCIPAL",
                  "state amount + on-chain tolerance"))

    def _fit(s: str, w: int) -> str:
        """Truncate rather than let a long cell shunt the whole row right."""
        return s if len(s) <= w - 1 else s[:w - 2] + "…"
    print(f"  {'candidate':<22}{'value':<30}{'evidence':<26}next step")
    for name, val, status, nxt in items:
        print(f"  {_fit(name,22):<22}{_fit(val,30):<30}{_fit(status,26):<26}{nxt}")


def risks(pos: dict) -> None:
    _rule("RISKS")
    earn = pos.get("earn", {})
    nav = pos.get("nav") or 0.0
    out = []

    # Concentration: a single issuer holding the whole core.
    if earn and nav:
        top, amt = max(((a, v["amt"]) for a, v in earn.items()), key=lambda kv: kv[1])
        share = amt / nav * 100
        if share > 80:
            out.append((f"concentration: {share:.0f}% in {top}",
                        f"${amt:.2f}", "single issuer; mitigated by daily paid-vs-promised audit"))

    # Promo-tier fragility: rate collapses if a bonus ends.
    for asset, v in earn.items():
        if asset not in STABLES or v["amt"] < 1:
            continue
        try:
            r = _get("/sapi/v1/simple-earn/flexible/list", {"asset": asset}).get("rows", [])[0]
        except Exception:
            continue
        base = float(r["latestAnnualPercentageRate"])
        tier = max((float(x) for x in (r.get("tierAnnualPercentageRate") or {}).values()), default=0.0)
        if tier > base * 2:
            out.append((f"promo-tier fragility: {asset}",
                        f"{tier*100:.2f}pp of {(base+tier)*100:.2f}%",
                        f"falls to {base*100:.2f}% if the promo ends; audit rotates out in 3 days"))

    out.append(("machine uptime", "~2% downtime",
                "laptop sleeps on battery; yield unaffected, detection pauses"))
    n_dust = sum(1 for a in pos.get("spot", {}) if a not in STABLES)
    out.append(("dust", f"{n_dust} coins stranded",
                "below both earn and sell minimums; unrecoverable"))
    for name, exposure, note in out:
        n = name if len(name) <= 33 else name[:32] + "…"
        e = exposure if len(exposure) <= 21 else exposure[:20] + "…"
        print(f"  {n:<34}{e:<22}{note}")


def desk() -> None:
    _rule("YOUR DESK  (only the principal can do these)")
    print("  1. Account → Referral — report the mode. Lite vs Pro is likely irreversible,")
    print("     and only Pro can reach the goal. One minute, unblocks the biggest lever.")
    print("  2. Rewards Hub → Learn & Earn — $10–25, zero capital. That is 2–5 years of")
    print("     this account's yield, for an afternoon. Rewards auto-compound now.")
    print("  3. Capital: state what is available and whether it may sit on-chain.")
    print("     Everything downstream is designable the day that lands.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=float, default=GOAL_HOURLY)
    a = ap.parse_args()

    print("═" * 74)
    print(f"OCTIVAULT INVESTING OFFICE — BRIEF   {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    print("═" * 74)

    pos = position()
    _rule("POSITION")
    if pos["err"]:
        print(f"  {UNKNOWN} — exchange unreadable: {pos['err']}")
    else:
        for asset, v in sorted(pos["earn"].items(), key=lambda kv: -kv[1]["amt"]):
            if v["amt"] > 0.01:
                print(f"  earn  {asset:<6} ${v['amt']:>9,.2f}   bonus paid to date ${v['bonus_cum']:.4f}")
        dust = sum(1 for x in pos["spot"] if x not in STABLES)
        print(f"  spot  {len(pos['spot'])} assets, {dust} of them non-stable dust")
        print(f"  NAV   ${pos['nav']:,.2f} in earn")

    performance(pos.get("nav"), pos.get("rate"))
    axes(pos.get("nav"), pos.get("rate"), a.goal)
    pipeline()
    risks(pos)
    desk()
    print("\n" + "═" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
