#!/usr/bin/env python3
"""
Option position tracker — values the ring-fenced BTC call from PUBLIC data.

WHY THIS EXISTS
---------------
The options API is not entitled on this account: `/eapi/v1/account` returns
Binance's error page even with funds in the options wallet, and neither API key
offers an "Enable European Options" checkbox. So the position cannot be read
from the account. It CAN be valued, because the option's mark price, order book
and greeks are PUBLIC endpoints that need no key at all.

The position is therefore recorded once, by hand, from the fill the operator
confirmed on the web ticket, and marked to market from public prices thereafter.
That is the honest arrangement: entry is asserted, valuation is measured.

WHAT IT REPORTS
---------------
  - current mark vs the entry premium, in dollars and percent
  - what BTC must reach for break-even and for the 10x that motivated the bet
  - delta as the market's own current probability of finishing in the money
  - theta as dollars per day of time decay, the cost of waiting
  - days left, and a reminder that a European option cannot be exercised early

It reads nothing private, moves no money, and places no orders.

Usage:
  python3 option_position_tracker.py            # value the open position
  python3 option_position_tracker.py record     # write the position file
Env: OPT_SYMBOL OPT_QTY OPT_ENTRY (used by `record`)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

STATE = os.getenv("OPT_STATE", "logs/option_position.json")
EAPI = "https://eapi.binance.com"
SPOT = "https://api.binance.com"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def record() -> None:
    """Write the position from the operator's confirmed fill."""
    pos = {
        "symbol": os.getenv("OPT_SYMBOL", "BTC-261127-105000-C"),
        "qty": float(os.getenv("OPT_QTY", "0.01")),
        "entry_premium": float(os.getenv("OPT_ENTRY", "845")),
        "opened": datetime.now(timezone.utc).isoformat(),
        "funded_usdt": 11.98,
        "source": "manual web fill; options API not entitled on this account",
    }
    pos["cost_usd"] = round(pos["qty"] * pos["entry_premium"], 4)
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(pos, f, indent=2)
    print(f"recorded {pos['symbol']} qty {pos['qty']} @ {pos['entry_premium']} "
          f"= ${pos['cost_usd']:.2f}")


def report() -> int:
    if not os.path.exists(STATE):
        print("No position recorded. Run: option_position_tracker.py record")
        return 1
    p = json.load(open(STATE))
    sym, qty, entry = p["symbol"], p["qty"], p["entry_premium"]
    strike = float(sym.split("-")[2])

    mark = _get(f"{EAPI}/eapi/v1/mark?symbol={sym}")[0]
    m = float(mark["markPrice"])
    delta, theta = float(mark["delta"]), float(mark["theta"])
    btc = float(_get(f"{SPOT}/api/v3/ticker/price?symbol=BTCUSDT")["price"])

    info = _get(f"{EAPI}/eapi/v1/exchangeInfo")
    exp = next((s["expiryDate"] for s in info["optionSymbols"] if s["symbol"] == sym), None)
    days = (exp / 1000 - time.time()) / 86400 if exp else float("nan")

    value = m * qty
    cost = entry * qty
    pnl = value - cost

    print("=" * 66)
    print(f"RING-FENCED CALL — {sym}")
    print("=" * 66)
    print(f"  BTC now         : ${btc:,.0f}")
    print(f"  premium         : paid {entry:,.0f}  ->  now {m:,.0f}")
    print(f"  position value  : ${value:.2f}  (cost ${cost:.2f})   "
          f"P&L ${pnl:+.2f}  {pnl/cost*100:+.0f}%")
    print(f"  days to expiry  : {days:.0f}   decay ${abs(theta)*qty:.2f}/day")
    print(f"  market's odds   : {delta*100:.0f}% (delta) of finishing in the money")
    print()
    print(f"  break-even  BTC >= ${strike + entry:,.0f}   ({(strike+entry)/btc*100-100:+.0f}% from here)")
    print(f"  10x         BTC >= ${strike + 10*entry:,.0f}   ({(strike+10*entry)/btc*100-100:+.0f}% from here)")
    print()
    # A European option cannot be exercised early; the only way out before
    # expiry is to SELL the contract, which is why the mark matters daily.
    print("  European: no early exercise. To take profit before 2026-11-27 you")
    print("  SELL the contract on the web ticket at the prevailing mark.")
    print(f"  Max loss stays ${cost:.2f}. The earn core is in a separate wallet.")
    return 0


if __name__ == "__main__":
    if (sys.argv[1:] or [""])[0] == "record":
        record()
    else:
        sys.exit(report())
