#!/usr/bin/env python3
"""The opportunity desk — watches the whole earn board for the next USD1.

WHY THIS EXISTS
---------------
The single largest gain this account has ever made was not a trade. On
2026-09-08 a review found USD1 paying 8.54% while the allocator sat in USDC at
7.24%, because the scan list was ten hardcoded assets and USD1 was not one of
them. Widening the scan was worth a permanent +1.35pp — more than every
execution optimisation in this repo combined, and more than any strategy that
was ever tested and passed (none were).

That gain came from LOOKING WIDER, once, by hand. Binance rotates bonus tiers
across assets on roughly a monthly cadence — USDT, FDUSD, XUSD, USD1, U and
USDC have each carried one during 2026 — so there will be a next USD1. Nothing
was watching for it.

This desk watches. It snapshots all 413 flexible products, keeps the history,
and reports what CHANGED: a tier appearing on an asset that had none, a rate
moving, a product opening or closing. A promo caught on day one instead of week
three is the whole difference, and it costs nothing to watch.

WHY IT WATCHES THE WHOLE BOARD, NOT JUST STABLECOINS
----------------------------------------------------
Because the thing worth catching is an asset ENTERING the opportunity space,
and you cannot see that by watching the assets already in it. Of 413 products
only 8 carry a tier today; the interesting event is the ninth.

WHAT IT WILL NOT DO
-------------------
Move money, or recommend holding a volatile asset for its rate. LSK shows
101.04% and STEEM 44.12% — those are borrow-driven base rates on tokens whose
price can halve, and a 40% rate on an asset that falls 50% is a 30% loss. Only
dollar-stable assets are scored as actionable; everything else is reported as
context and explicitly marked NOT ACTIONABLE.

Read-only. Authenticated READS only; no orders, transfers or subscriptions.

Usage:
  python3 opportunity_desk.py scan     # snapshot + report changes since last
  python3 opportunity_desk.py report   # what the board looks like now
  python3 opportunity_desk.py watch    # every SCAN_MIN minutes, forever
Env: OPP_STATE  OPP_SCAN_MIN (60)  OPP_MIN_EDGE_PP (0.25)  HYBRID_ALERT_CMD
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

KEY, SECRET = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
BASE = "https://api.binance.com"
STATE = os.getenv("OPP_STATE", "logs/opportunity_board.jsonl")
SCAN_MIN = float(os.getenv("OPP_SCAN_MIN", "60"))
# How much better a new product must be before it is worth waking anyone.
MIN_EDGE_PP = float(os.getenv("OPP_MIN_EDGE_PP", "0.25"))
ALERT = os.getenv("HYBRID_ALERT_CMD", "./hybrid_alert.sh")

# Dollar-stable assets are the only ones whose RATE is the whole story. A 40%
# rate on a token that can halve is not a 40% return, and this desk must never
# imply otherwise. Kept deliberately broad — an asset we do not hold today may
# be the one that gets a tier tomorrow, and it still has to be dollar-stable to
# be actionable when it does.
STABLE = {"USDT", "USDC", "USD1", "FDUSD", "TUSD", "DAI", "USDP", "XUSD",
          "RLUSD", "PYUSD", "BUSD", "USDD", "U", "AEUR", "EURI", "USDS"}
# Excluded from actionable even though dollar-pegged: their yield comes from a
# basis trade (long spot / short perp), i.e. shorting under the hood.
SYNTHETIC = {"USDE", "SUSDE", "BFUSD"}


def _get(path: str, params: dict | None = None):
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 10000
    qs = urllib.parse.urlencode(p)
    qs += "&signature=" + hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(f"{BASE}{path}?{qs}", headers={"X-MBX-APIKEY": KEY})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def board() -> dict:
    """Every flexible product, as {ASSET: {base, tier, cap, open}}.

    Paginates to the end rather than taking the first page: the tiered products
    are scattered through 413 rows and a truncated read would silently miss the
    one that matters.
    """
    out, cur = {}, 1
    while cur <= 12:
        rows = _get("/sapi/v1/simple-earn/flexible/list",
                    {"current": cur, "size": 100}).get("rows", [])
        for r in rows:
            asset = str(r.get("asset", "")).upper()
            if not asset:
                continue
            tiers = r.get("tierAnnualPercentageRate") or {}
            try:
                tier = max((float(v) for v in tiers.values()), default=0.0)
                base = float(r.get("latestAnnualPercentageRate", 0) or 0)
            except (TypeError, ValueError):
                continue
            out[asset] = {"base": round(base, 6), "tier": round(tier, 6),
                          "cap": (list(tiers) or [None])[0],
                          "open": bool(r.get("canPurchase"))}
        if len(rows) < 100:
            break
        cur += 1
    return out


def actionable(asset: str, p: dict) -> bool:
    """Dollar-stable, purchasable, and not synthetic. Everything else is context."""
    return asset in STABLE and asset not in SYNTHETIC and p.get("open", False)


def _load_last() -> dict | None:
    try:
        rows = [json.loads(l) for l in open(STATE) if l.strip()]
    except FileNotFoundError:
        return None
    return rows[-1]["board"] if rows else None


def _alert(msg: str) -> None:
    print(f"  🔔 {msg}")
    if ALERT and os.path.exists(ALERT) and os.access(ALERT, os.X_OK):
        try:
            subprocess.Popen([ALERT, msg], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass


def diff(old: dict, new: dict) -> list[str]:
    """What changed, in the order a desk would care.

    A TIER APPEARING on an asset that had none is the headline event — that is
    exactly the USD1 shape, and it is the reason this desk exists.
    """
    events = []
    best_now = max((v["base"] + v["tier"] for a, v in new.items() if actionable(a, v)),
                   default=0.0)
    for asset, p in sorted(new.items()):
        o = old.get(asset)
        tot = p["base"] + p["tier"]
        act = actionable(asset, p)
        if o is None:
            if act and tot * 100 >= best_now * 100 - MIN_EDGE_PP:
                events.append(f"NEW PRODUCT {asset} at {tot*100:.2f}% "
                              f"(base {p['base']*100:.2f} + tier {p['tier']*100:.2f})")
            continue
        # The headline: an asset entering the tiered space.
        if o["tier"] == 0 and p["tier"] > 0 and act:
            events.append(f"★ TIER APPEARED on {asset}: +{p['tier']*100:.2f}pp "
                          f"→ {tot*100:.2f}% total, cap {p['cap']}")
        elif o["tier"] > 0 and p["tier"] == 0 and act:
            events.append(f"TIER REMOVED from {asset}: was {o['tier']*100:.2f}pp, "
                          f"now base {p['base']*100:.2f}% only")
        elif act and abs((p["tier"] - o["tier"]) * 100) >= MIN_EDGE_PP:
            events.append(f"TIER CHANGED {asset}: {o['tier']*100:.2f}pp → "
                          f"{p['tier']*100:.2f}pp ({tot*100:.2f}% total)")
        elif act and abs((tot - (o["base"] + o["tier"])) * 100) >= MIN_EDGE_PP:
            events.append(f"rate moved {asset}: {(o['base']+o['tier'])*100:.2f}% → {tot*100:.2f}%")
        # Closure is judged on what the product WAS, not what it is. `actionable`
        # requires open=True, so testing it here would suppress the very event
        # that matters: a product we could have used going away. That is the
        # moment money should move, and it must never be silent.
        was_act = asset in STABLE and asset not in SYNTHETIC and o.get("open", False)
        if was_act and not p.get("open"):
            events.append(f"CLOSED {asset} (was {(o['base']+o['tier'])*100:.2f}%) — sold out")
        if act and not o.get("open"):
            events.append(f"REOPENED {asset} at {tot*100:.2f}%")
    return events


def scan(quiet: bool = False) -> int:
    try:
        now = board()
    except Exception as e:
        print(f"[opp] board unreadable: {str(e)[:100]} — nothing recorded")
        return 1
    if not now:
        print("[opp] empty board — not recording")
        return 1

    old = _load_last()
    events = diff(old, now) if old else []
    tiered = {a: v for a, v in now.items() if v["tier"] > 0}
    act = {a: v for a, v in now.items() if actionable(a, v)}
    best = max(act.items(), key=lambda kv: kv[1]["base"] + kv[1]["tier"], default=(None, None))

    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            "n": len(now), "board": now}) + "\n")

    stamp = datetime.now(timezone.utc).strftime("%m-%d %H:%M")
    line = (f"[opp {stamp}] {len(now)} products · {len(tiered)} tiered · "
            f"best actionable {best[0]} {(best[1]['base']+best[1]['tier'])*100:.2f}%"
            if best[0] else f"[opp {stamp}] {len(now)} products · none actionable")
    print(line)
    for e in events:
        if e.startswith("★") or e.startswith("NEW"):
            _alert(f"opportunity desk: {e}")
        else:
            print(f"  · {e}")
    if old and not events and not quiet:
        print("  · no change")
    return 0


def report() -> int:
    try:
        now = board()
    except Exception as e:
        print(f"Board unreadable: {str(e)[:100]}")
        return 1
    tiered = sorted(((v["base"] + v["tier"], a, v) for a, v in now.items() if v["tier"] > 0),
                    reverse=True)
    print("=" * 72)
    print(f"OPPORTUNITY BOARD — {len(now)} flexible products, {len(tiered)} carry a bonus tier")
    print("=" * 72)
    print("  The tiered set is the whole opportunity space. Of 413 products only a")
    print("  handful carry a tier at any time, and Binance rotates which ones monthly.\n")
    print(f"  {'asset':<8}{'total':>9}{'base':>8}{'tier':>8}  {'cap':<18}status")
    for tot, a, v in tiered:
        if actionable(a, v):
            status = "ACTIONABLE"
        elif a in SYNTHETIC:
            status = "excluded — synthetic (short under the hood)"
        elif a in STABLE:
            status = "closed to new subscriptions"
        else:
            status = "NOT ACTIONABLE — price risk, not a stable"
        print(f"  {a:<8}{tot*100:>8.2f}%{v['base']*100:>7.2f}%{v['tier']*100:>7.2f}%  "
              f"{str(v['cap'] or '—'):<18}{status}")
    hi = sorted(((v["base"] + v["tier"], a) for a, v in now.items()), reverse=True)[:5]
    print("\n  Highest raw rates on the whole board — context only, NOT actionable:")
    for tot, a in hi:
        print(f"    {a:<8}{tot*100:>8.2f}%   "
              f"{'stable' if a in STABLE else 'volatile token — a 40% rate on an asset that halves is a 30% loss'}")
    print("=" * 72)
    return 0


def main() -> int:
    cmd = (sys.argv[1:] or ["scan"])[0]
    if cmd == "report":
        return report()
    if cmd == "watch":
        print(f"[opp] watching the board every {SCAN_MIN:.0f} min — Ctrl-C to stop", flush=True)
        while True:
            try:
                scan(quiet=True)
            except Exception as e:
                print(f"[opp] cycle failed: {str(e)[:90]}", flush=True)
            sys.stdout.flush()
            time.sleep(SCAN_MIN * 60)
    return scan()


if __name__ == "__main__":
    raise SystemExit(main())
