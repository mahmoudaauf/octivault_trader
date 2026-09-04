#!/usr/bin/env python3
"""
Delta-neutral STAKING-yield PAPER trader — forward proof-of-edge engine.

THE STRUCTURE
-------------
    long spot, subscribed to Simple Earn   (collects the staking APR)
  + short perpetual of the same asset      (removes the price risk)
  = delta neutral, earning  staking_APR + funding − fees

Different from `carry_paper_trader.py`, which chased EXTREME funding on majors
and came out at roughly −2%/yr over 166 days. This one is paid by STAKING and
merely tolerates funding. The scan (`delta_neutral_yield_scan.py backtest`) found
five assets on 2026-09-04 where a real staking APR coincides with funding that
has been neutral-to-positive over 90 days: GTC, COOKIE, F, BIGTIME, VELODROME.

WHY PAPER FIRST, AND WHAT IT MEASURES THAT THE SCAN CANNOT
----------------------------------------------------------
The scan has 90 days of FUNDING history and ZERO days of STAKING history —
Binance does not expose past APRs. Promotional rates collapse. This daemon
holds paper positions and accrues:
  - the REAL staking APR, re-read every cycle (so a collapse is felt, not assumed)
  - the REAL settled funding at each 8h window, from the exchange's own history
  - fees at open and close on both legs
  - BASIS: spot and perp prices are not identical, and the residual after the
    two legs' PnL is netted is a real cost/gain this structure carries
  - LIQUIDATION: the short is sized at 1x (margin == notional). If the mark
    rises past the liquidation threshold the position is marked DEAD and the
    margin is lost. This is the risk that makes the trade unrunnable at small
    size, and it must be counted, not hand-waved.

Every close is booked to a ledger with the split above, so `report` can say
which component actually produced the return — and whether the whole thing
beats the ~5% stablecoin baseline it is competing with.

ENTRY / EXIT
------------
Entry: assets the backtest marks 'survives' (net > 0 on history AND ≥ 80% of
funding windows positive), best first, up to MAX_POSITIONS.
Exit: staking APR + trailing funding drops below MIN_NET_APR; or liquidation;
or MAX_HOLD_D elapsed (forces a close/re-evaluate so churn cost is measured).

It places NO orders and moves NO money. Same live-arming path as the others
would apply later; it is not wired here on purpose — see the memory note on
feasibility: below ~$1,840 the gain over stablecoin earn does not clear the
risk, so the honest use of this file today is to build the forward record.

Usage:
  python3 delta_neutral_paper_trader.py           # daemon
  python3 delta_neutral_paper_trader.py report    # forward verdict
Env: DNP_NOTIONAL(100) DNP_MAX_POSITIONS(4) DNP_MIN_NET_APR(0.04)
     DNP_MAX_HOLD_D(30) DNP_POLL_MIN(60) DNP_FEE_SPOT_PCT(0.1) DNP_FEE_PERP_PCT(0.05)
     DNP_LIQ_MOVE_PCT(90)  — % price rise at which a 1x short is liquidated
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

STATE = os.getenv("DNP_STATE", "logs/dnp_state.json")
LEDGER = os.getenv("DNP_LEDGER", "logs/dnp_ledger.jsonl")
KILL = os.getenv("DNP_KILL_FILE", "logs/dnp.stop")

NOTIONAL = float(os.getenv("DNP_NOTIONAL", "100"))        # $ per leg, paper
MAX_POS = int(os.getenv("DNP_MAX_POSITIONS", "4"))
MIN_NET_APR = float(os.getenv("DNP_MIN_NET_APR", "0.04"))
MAX_HOLD_D = float(os.getenv("DNP_MAX_HOLD_D", "30"))
POLL_MIN = float(os.getenv("DNP_POLL_MIN", "60"))
FEE_SPOT = float(os.getenv("DNP_FEE_SPOT_PCT", "0.1")) / 100.0
FEE_PERP = float(os.getenv("DNP_FEE_PERP_PCT", "0.05")) / 100.0
LIQ_MOVE = float(os.getenv("DNP_LIQ_MOVE_PCT", "90")) / 100.0
MIN_PCT_POS = float(os.getenv("DNP_MIN_PCT_POSITIVE", "80"))


def _log(msg):
    print(f"[dnp {datetime.now(timezone.utc):%m-%d %H:%M}] {msg}", flush=True)


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _book(rec):
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ── market reads ─────────────────────────────────────────────────────────────

async def _staking_apr(client, asset: str) -> float | None:
    """Current flexible APR, or None if the product vanished (also a signal)."""
    try:
        r = await client.get_simple_earn_flexible_product_list(asset=asset)
        for p in (r.get("rows") or []):
            if str(p.get("asset", "")).upper() == asset and p.get("canPurchase") is not False:
                return float(p.get("latestAnnualPercentageRate", 0) or 0)
    except Exception as e:
        _log(f"  apr read failed {asset}: {str(e)[:60]}")
    return None


async def _prices(client, symbol: str) -> tuple[float, float] | None:
    """(spot, perp mark). Both, because their difference is the basis."""
    try:
        s = float((await client.get_symbol_ticker(symbol=symbol))["price"])
        m = float((await client.futures_mark_price(symbol=symbol))["markPrice"])
        return s, m
    except Exception as e:
        _log(f"  price read failed {symbol}: {str(e)[:60]}")
        return None


async def _settled_funding(client, symbol: str, since_ms: int) -> tuple[float, int]:
    """Sum of funding RATES settled since `since_ms` (positive = short receives),
    and the timestamp of the last one, so nothing is double counted."""
    total, last = 0.0, since_ms
    try:
        rows = await client.futures_funding_rate(symbol=symbol, startTime=since_ms + 1, limit=1000)
        for r in rows or []:
            total += float(r.get("fundingRate", 0) or 0)
            last = max(last, int(r.get("fundingTime", since_ms)))
    except Exception as e:
        _log(f"  funding read failed {symbol}: {str(e)[:60]}")
    return total, last


# ── candidates: reuse the scanner's backtest, do not re-derive it ────────────

async def _candidates(exclude: set[str]) -> list[dict]:
    import delta_neutral_yield_scan as dny
    rec = await dny.scan()
    assets = [r["asset"] for r in rec["rows"][:25] if r["asset"] not in exclude]
    hist = await dny.funding_history(assets, 90)
    aprs = {r["asset"]: r["staking_apr"] for r in rec["rows"]}
    out = []
    for a in assets:
        h = hist.get(a)
        if not h or h["pct_positive"] < MIN_PCT_POS:
            continue
        net = (aprs[a] + h["mean"]) * dny.DEPLOYED_FRACTION - dny.COST_PCT / 100.0
        if net > 0:
            out.append({"asset": a, "apr": aprs[a], "fund_mean": h["mean"], "net": net})
    return sorted(out, key=lambda r: -r["net"])


# ── position lifecycle ───────────────────────────────────────────────────────

def _open(state, cand, spot, mark):
    qty = NOTIONAL / spot
    fees = NOTIONAL * (FEE_SPOT + FEE_PERP)
    state["positions"][cand["asset"]] = {
        "symbol": f"{cand['asset']}USDT", "qty": qty,
        "spot_entry": spot, "perp_entry": mark,
        "opened": time.time(), "last_funding_ms": int(time.time() * 1000),
        "apr_at_open": cand["apr"], "apr_now": cand["apr"], "last_accrual": time.time(),
        "staking_usd": 0.0, "funding_usd": 0.0, "fees_usd": fees,
        "max_adverse_pct": 0.0, "funding_rates": [],
    }
    _log(f"  OPEN {cand['asset']}: ${NOTIONAL:.0f}/leg, apr {cand['apr']*100:.2f}%, "
         f"90d funding {cand['fund_mean']*100:+.1f}%, fees ${fees:.2f}")


def _close(state, asset, spot, mark, reason):
    p = state["positions"].pop(asset)
    p["fees_usd"] += NOTIONAL * (FEE_SPOT + FEE_PERP)
    spot_pnl = (spot - p["spot_entry"]) * p["qty"]
    perp_pnl = (p["perp_entry"] - mark) * p["qty"]
    basis = spot_pnl + perp_pnl                  # ~0 if perfectly hedged
    liquidated = reason == "LIQUIDATED"
    margin_lost = NOTIONAL if liquidated else 0.0
    net = p["staking_usd"] + p["funding_usd"] + basis - p["fees_usd"] - margin_lost
    days = (time.time() - p["opened"]) / 86400
    capital = 2 * NOTIONAL                        # spot + equal margin
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "asset": asset, "reason": reason,
           "days": round(days, 2), "staking": round(p["staking_usd"], 4),
           "funding": round(p["funding_usd"], 4), "basis": round(basis, 4),
           "fees": round(p["fees_usd"], 4), "margin_lost": margin_lost,
           "net": round(net, 4), "capital": capital,
           # Floored at one day: annualising a fee-only close held for seconds
           # prints a seven-figure percentage that means nothing.
           "net_apr_on_capital": round(net / capital * 365 / max(days, 1.0), 6),
           "apr_open": p["apr_at_open"], "apr_close": p["apr_now"],
           "max_adverse_pct": round(p["max_adverse_pct"] * 100, 2)}
    _book(rec)
    _log(f"  CLOSE {asset} [{reason}] {days:.1f}d net ${net:+.2f} "
         f"(stake ${p['staking_usd']:.2f} fund ${p['funding_usd']:+.2f} basis ${basis:+.2f} "
         f"fees ${p['fees_usd']:.2f}{' MARGIN LOST' if liquidated else ''}) "
         f"= {rec['net_apr_on_capital']*100:+.1f}%/yr on capital")


async def _tick(client, state):
    now = time.time()
    for asset in list(state["positions"]):
        p = state["positions"][asset]
        px = await _prices(client, p["symbol"])
        if not px:
            continue
        spot, mark = px
        # staking accrual on the SPOT leg at the CURRENT rate
        apr = await _staking_apr(client, asset)
        if apr is None:
            apr = 0.0                              # product gone: earns nothing now
        p["apr_now"] = apr
        hours = (now - p["last_accrual"]) / 3600
        p["staking_usd"] += NOTIONAL * apr * hours / (365 * 24)
        p["last_accrual"] = now
        # funding actually settled since last look, on the short's notional
        rate_sum, last_ms = await _settled_funding(client, p["symbol"], p["last_funding_ms"])
        if rate_sum:
            p["funding_usd"] += NOTIONAL * rate_sum
            p["funding_rates"].append(rate_sum)
            p["last_funding_ms"] = last_ms
        # liquidation watch on the short (1x margin): price UP is the danger
        adverse = mark / p["perp_entry"] - 1.0
        p["max_adverse_pct"] = max(p["max_adverse_pct"], adverse)
        if adverse >= LIQ_MOVE:
            _close(state, asset, spot, mark, "LIQUIDATED")
            continue
        # exit rules
        days = (now - p["opened"]) / 86400
        trailing = statistics.mean(p["funding_rates"][-90:]) * 3 * 365 if p["funding_rates"] else 0.0
        if apr + trailing < MIN_NET_APR:
            _close(state, asset, spot, mark, f"RATE_COLLAPSE apr={apr*100:.1f}% fund={trailing*100:+.1f}%")
        elif days >= MAX_HOLD_D:
            _close(state, asset, spot, mark, "MAX_HOLD")
    # entries
    if len(state["positions"]) < MAX_POS:
        for c in await _candidates(set(state["positions"])):
            if len(state["positions"]) >= MAX_POS:
                break
            px = await _prices(client, f"{c['asset']}USDT")
            if px:
                _open(state, c, *px)


# ── report ───────────────────────────────────────────────────────────────────

def _report():
    rows = [json.loads(l) for l in open(LEDGER)] if os.path.exists(LEDGER) else []
    state = _load(STATE, {"positions": {}})
    print("=" * 74)
    print("DELTA-NEUTRAL STAKING YIELD — forward paper record")
    print("=" * 74)
    if rows:
        cap_days = sum(r["capital"] * r["days"] for r in rows)
        net = sum(r["net"] for r in rows)
        print(f"  closed          : {len(rows)}  liquidations: {sum(r['reason']=='LIQUIDATED' for r in rows)}")
        print(f"  staking         : ${sum(r['staking'] for r in rows):+.2f}")
        print(f"  funding         : ${sum(r['funding'] for r in rows):+.2f}")
        print(f"  basis           : ${sum(r['basis'] for r in rows):+.2f}")
        print(f"  fees            : ${-sum(r['fees'] for r in rows):+.2f}")
        print(f"  margin lost     : ${-sum(r['margin_lost'] for r in rows):+.2f}")
        print(f"  NET             : ${net:+.2f}")
        if cap_days:
            print(f"  NET APR/capital : {net / cap_days * 365 * 100:+.2f}%   <- beat ~5% stablecoin or it is pointless")
        worst = max((r["max_adverse_pct"] for r in rows), default=0)
        print(f"  worst adverse   : {worst:.1f}% (liquidation at {LIQ_MOVE*100:.0f}%)")
    else:
        print("  no closed positions yet")
    if state["positions"]:
        print("  open:")
        for a, p in state["positions"].items():
            d = (time.time() - p["opened"]) / 86400
            print(f"    {a:<10} {d:5.1f}d  apr {p['apr_at_open']*100:.1f}%->{p['apr_now']*100:.1f}%  "
                  f"stake ${p['staking_usd']:.2f} fund ${p['funding_usd']:+.2f} "
                  f"adverse {p['max_adverse_pct']*100:.1f}%")
    print("=" * 74)


# ── daemon ───────────────────────────────────────────────────────────────────

async def run():
    from binance import AsyncClient
    from exchange_resilience import create_client_with_retry, resync_clock
    client = await create_client_with_retry(AsyncClient)
    state = _load(STATE, {"positions": {}})
    _log(f"start — PAPER. notional ${NOTIONAL:.0f}/leg, max {MAX_POS} positions, "
         f"poll {POLL_MIN:.0f}m. No orders, no money.")
    try:
        while True:
            if os.path.exists(KILL):
                _log("kill file present — idling")
                await asyncio.sleep(POLL_MIN * 60)
                continue
            try:
                await resync_clock(client, "dnp")
                await asyncio.wait_for(_tick(client, state), timeout=600)
                _save(STATE, state)
                _log(f"open={len(state['positions'])} "
                     + " ".join(f"{a}:${p['staking_usd']+p['funding_usd']-p['fees_usd']:+.2f}"
                                for a, p in state["positions"].items()))
            except asyncio.TimeoutError:
                _log("  tick exceeded 600s — abandoned, retrying next poll")
            except Exception as e:
                _log(f"  [ERROR] {str(e)[:120]}")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


if __name__ == "__main__":
    if (sys.argv[1:] or [""])[0] == "report":
        _report()
    else:
        asyncio.run(run())
