#!/usr/bin/env python3
"""
Ring-fenced BTC call bet — the operator's 10x shot, sized so a miss cannot
take the compounding core with it.

WHAT THIS IS
------------
Operator mandate 2026-09-04: grow $60 to $600 in 90 days with no deposits.
Every legitimate yield path in this repo tops out near 5-10%/yr, so a 10x is
not earned — it is a bet the market has to pay. The instrument with the best
odds AND a bounded loss is an out-of-the-money BTC call: pay a premium, lose at
most the premium, no liquidation, no funding bleed.

This script executes the RING-FENCED size only: it sells the idle BTC hold
(which earns 0% and is not part of the earn core), moves the proceeds to the
options wallet, and buys ONE contract. The $48 stablecoin core is never read,
never touched, and not even referenced here. If the bet misses, the machine in
hybrid_allocator.py is exactly as it was.

THE BET (priced 2026-09-04, BTC $79,590)
----------------------------------------
  contract  BTC-261127-105000-C   (expiry ~2026-11-27, strike $105,000)
  cost      ~$8.05 at min qty 0.01 (ask $805)
  10x needs BTC >= ~$113,000 at expiry (+42%)
  odds      market delta 0.11 (~11%); BTC history since 2017: +40% in 84d
            finished in 21% of windows, 10% over the last two years.

SAFETY
------
  - Refuses unless the API key reports enableVanillaOptions=True.
  - Refuses without --yes. Prints the full plan first.
  - Sells ONLY the BTC free balance; never USDT, never USDC, never earn.
  - Caps the option spend at the BTC proceeds. Will not top up from the core.
  - Limit order at the current ask (not market): options books are thin.
  - Every step is logged to logs/hybrid_ledger.jsonl with kind=option_bet.

TO ENABLE (operator, in the Binance app — this code cannot do it)
-----------------------------------------------------------------
  1. Derivatives -> Options -> activate the options account (regional).
  2. API Management -> this key -> Edit -> tick "Enable European Options".

Usage:
  python3 ringfenced_call_bet.py            # dry run: show the plan, send nothing
  python3 ringfenced_call_bet.py --yes      # execute
Env:  BET_SYMBOL (default BTC-261127-105000-C)  BET_QTY (0.01)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("BINANCE_API_KEY")
SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL = os.getenv("BET_SYMBOL", "BTC-261127-105000-C")
QTY = float(os.getenv("BET_QTY", "0.01"))
LEDGER = "logs/hybrid_ledger.jsonl"

SPOT = "https://api.binance.com"
EAPI = "https://eapi.binance.com"


def _req(base: str, path: str, method: str = "GET", params: dict | None = None,
         signed: bool = False):
    p = dict(params or {})
    if signed:
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = 10000
    q = urllib.parse.urlencode(p)
    if signed:
        q += "&signature=" + hmac.new(SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}" + (f"?{q}" if method == "GET" else "")
    data = q.encode() if method != "GET" else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-MBX-APIKEY": KEY,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {body[:200]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(f"{method} {path} -> non-JSON response (product not available?): {body[:120]}")


def _book(rec: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "kind": "option_bet", **rec}) + "\n")


def main(execute: bool) -> int:
    print("=" * 70)
    print("RING-FENCED CALL BET — " + ("EXECUTING" if execute else "DRY RUN (nothing sent)"))
    print("=" * 70)

    # Gate 1: the key must be allowed to trade options. Checked, not assumed.
    perms = _req(SPOT, "/sapi/v1/account/apiRestrictions", signed=True)
    if not perms.get("enableVanillaOptions"):
        print("  ❌ enableVanillaOptions is FALSE on this API key.")
        print("     Activate Options in the Binance app, then edit the key and tick")
        print("     'Enable European Options'. This script cannot do that for you.")
        return 2
    print("  ✅ key permits options trading")

    # What we are selling: the idle BTC hold, and only its FREE balance.
    bal = _req(SPOT, "/api/v3/account", signed=True)
    btc_free = next((float(b["free"]) for b in bal["balances"] if b["asset"] == "BTC"), 0.0)
    btc_px = float(_req(SPOT, "/api/v3/ticker/price", params={"symbol": "BTCUSDT"})["price"])
    btc_usd = btc_free * btc_px
    print(f"  BTC hold: {btc_free:.8f} ≈ ${btc_usd:.2f} at ${btc_px:,.0f}")
    if btc_usd < 5.0:
        print("  ❌ BTC hold below the $5 spot minimum — nothing to ring-fence")
        return 2

    # What we are buying: the contract at the live ASK, not the mark.
    depth = _req(EAPI, "/eapi/v1/depth", params={"symbol": SYMBOL, "limit": 10})
    ask = float(depth["asks"][0][0])
    mark = _req(EAPI, "/eapi/v1/mark", params={"symbol": SYMBOL})[0]
    cost = ask * QTY
    strike = float(SYMBOL.split("-")[2])
    print(f"  contract: {SYMBOL}  ask ${ask:,.0f}  mark ${float(mark['markPrice']):,.0f}  "
          f"delta {float(mark['delta']):.2f}  IV {float(mark['markIV'])*100:.0f}%")
    print(f"  buy {QTY} -> cost ${cost:.2f}   10x if BTC >= ${strike + 10*ask:,.0f} "
          f"({(strike + 10*ask)/btc_px*100-100:+.0f}%) at expiry")
    if cost > btc_usd * 0.98:
        print(f"  ❌ contract costs ${cost:.2f} but BTC proceeds are only ${btc_usd:.2f} — "
              f"not topping up from the core. Pick a cheaper strike (BET_SYMBOL).")
        return 2
    print(f"  remainder ≈ ${btc_usd - cost:.2f} stays in spot USDT -> allocator sweeps it to earn")
    if not execute:
        print("\n  Dry run complete. Re-run with --yes to execute.")
        return 0

    # 1. Sell the BTC hold. Market order; BTCUSDT is the deepest book on earth.
    qty = f"{btc_free:.5f}"
    sold = _req(SPOT, "/api/v3/order", "POST", signed=True,
                params={"symbol": "BTCUSDT", "side": "SELL", "type": "MARKET", "quantity": qty})
    got = float(sold.get("cummulativeQuoteQty", 0))
    print(f"  1. sold {qty} BTC -> ${got:.2f} USDT")
    _book({"step": "sell_btc", "qty": qty, "usdt": got})

    # 2. Move exactly the premium to the options wallet.
    amt = f"{min(cost * 1.02, got):.2f}"           # 2% headroom for fee, capped
    _req(SPOT, "/sapi/v1/asset/transfer", "POST", signed=True,
         params={"type": "MAIN_OPTION", "asset": "USDT", "amount": amt})
    print(f"  2. transferred ${amt} USDT spot -> options wallet")
    _book({"step": "transfer", "usdt": float(amt)})

    # 3. Limit buy at the ask.
    order = _req(EAPI, "/eapi/v1/order", "POST", signed=True,
                 params={"symbol": SYMBOL, "side": "BUY", "type": "LIMIT",
                         "quantity": f"{QTY:.2f}", "price": f"{ask:.0f}", "timeInForce": "GTC"})
    print(f"  3. order placed: id {order.get('orderId')} status {order.get('status')}")
    _book({"step": "buy_call", "symbol": SYMBOL, "qty": QTY, "price": ask,
           "order_id": order.get("orderId"), "status": order.get("status")})
    print("\n  ✅ Bet is on. Expiry settles automatically in USDT. Core untouched.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(execute="--yes" in sys.argv))
    except RuntimeError as e:
        print(f"  ❌ {e}")
        sys.exit(1)
