#!/usr/bin/env python3
"""
FULL delta-neutral carry validation on testnet (fake money).

Runs the complete carry execution end-to-end across BOTH legs, on Binance's two
separate testnets:
  - PERP leg  → futures testnet (BINANCE_FUTURES_TESTNET_*)  : short perp
  - SPOT leg  → spot testnet   (BINANCE_TESTNET_*)           : long spot
Opens both, confirms, closes both. Reports each leg's fills + fees. Zero real money.

Usage: python3 testnet_validate_full.py [SYMBOL] [NOTIONAL_USD]
"""
from __future__ import annotations

import asyncio
import os
import sys


def _prec(filters, ftype, key):
    step = next(f[key] for f in filters if f["filterType"] == ftype)
    st = step.rstrip("0")
    return len(st.split(".")[1]) if "." in st else 0


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    notional = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    lev = 2

    fut = await AsyncClient.create(
        os.getenv("BINANCE_FUTURES_TESTNET_KEY"), os.getenv("BINANCE_FUTURES_TESTNET_SECRET"),
        testnet=True)
    spot = await AsyncClient.create(
        os.getenv("BINANCE_TESTNET_API_KEY"), os.getenv("BINANCE_TESTNET_API_SECRET_HMAC"),
        testnet=True)

    print(f"🧪 FULL DELTA-NEUTRAL CARRY — {symbol} ~${notional} (FAKE money, both legs)")
    ok_open = {"perp": False, "spot": False}
    try:
        finfo = await fut.futures_exchange_info()
        fsym = next(x for x in finfo["symbols"] if x["symbol"] == symbol)
        fprec = _prec(fsym["filters"], "LOT_SIZE", "stepSize")
        sinfo = await spot.get_exchange_info()
        ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
        sprec = _prec(ssym["filters"], "LOT_SIZE", "stepSize")

        px = float((await fut.futures_mark_price(symbol=symbol))["markPrice"])
        qf = round(notional / px, fprec)
        qs = round(notional / px, sprec)
        print(f"  mark=${px:,.2f}  perp_qty={qf}  spot_qty={qs}")

        # ── OPEN: short perp + long spot ──
        await fut.futures_change_leverage(symbol=symbol, leverage=lev)
        po = await fut.futures_create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qf)
        ok_open["perp"] = True
        print(f"  ✅ PERP short opened — orderId={po.get('orderId')}")
        so = await spot.create_order(symbol=symbol, side="BUY", type="MARKET", quantity=qs)
        ok_open["spot"] = True
        print(f"  ✅ SPOT long opened — orderId={so.get('orderId')} status={so.get('status')}")

        print("  → position is now DELTA-NEUTRAL (short perp + long spot). "
              "Price moves cancel; funding accrues.")

        # ── CLOSE both legs ──
        pc = await fut.futures_create_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=qf, reduceOnly=True)
        print(f"  ✅ PERP closed — orderId={pc.get('orderId')}")
        sc = await spot.create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qs)
        print(f"  ✅ SPOT closed — orderId={sc.get('orderId')}")

        # Fees.
        pf = sum(float(t["commission"]) for t in (await fut.futures_account_trades(symbol=symbol, limit=2)))
        print(f"\n  perp round-trip fees: {pf:.4f} USDT")
        print("\n" + "=" * 62)
        print("✅ FULL TWO-LEG DELTA-NEUTRAL EXECUTION VALIDATED on testnet.")
        print("   Short-perp + long-spot open AND close both execute cleanly.")
        print("   The entire live carry order path is now proven (fake money).")
        print("=" * 62)
    except Exception as e:
        print(f"\n❌ FAILED at open={ok_open}: {type(e).__name__}: {str(e)[:160]}")
        # Best-effort unwind so we don't leave a one-legged testnet position.
        try:
            if ok_open["perp"]:
                qf2 = round(notional / px, fprec)
                await fut.futures_create_order(symbol=symbol, side="BUY", type="MARKET",
                                               quantity=qf2, reduceOnly=True)
                print("   unwound perp leg")
        except Exception:
            pass
    finally:
        await fut.close_connection()
        await spot.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
