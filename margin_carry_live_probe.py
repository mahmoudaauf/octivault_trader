#!/usr/bin/env python3
"""
ONE-SHOT real-money mechanics test for negative-funding carry's untested leg:
borrow -> short-spot (margin) -> long-perp -> close -> repay.

WHY THIS EXISTS
---------------
Every other piece of this project's execution was validated on Binance's
testnet before touching real capital (see testnet_validate_full.py). This
leg can't be: Binance's public spot testnet has no margin API at all
(confirmed directly — get_margin_account() returns "Invalid Api-Key ID" on
testnet while normal spot calls succeed with the same key). So the ONLY way
to prove the borrow/short/repay API sequence actually works is a real,
deliberately tiny, immediately-closed test. This is a MECHANICS test, not a
profitability test — it does not hold the position long enough to actually
collect funding.

Sequence (real money, ~$10 notional, held seconds):
  1. create_margin_loan(asset, qty)          — borrow the asset
  2. create_margin_order(SELL, qty)          — sell it (opens short-spot leg)
  3. futures_create_order(BUY, qty)          — open long-perp leg (delta-neutral)
  4. futures_create_order(SELL, reduceOnly)  — close the perp leg
  5. create_margin_order(BUY, AUTO_REPAY)    — buy back + repay the loan in one call

On any failure mid-sequence, best-effort unwinds whatever legs are open and
explicitly repays any outstanding loan — a real loan must never be left
dangling. Prints account state before/after for manual verification.

Usage: python3 margin_carry_live_probe.py [SYMBOL] [NOTIONAL_USD]
Default: CHIPUSDT $10 (confirmed real borrow supply as of 2026-07-20).
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

    symbol = sys.argv[1] if len(sys.argv) > 1 else "CHIPUSDT"
    notional = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    asset = symbol[:-4]
    lev = 2

    c = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

    print(f"🔴 REAL-MONEY MECHANICS TEST — {symbol} ~${notional} (held seconds, delta-neutral)")
    step_done = {"loan": False, "short_spot": False, "long_perp": False}
    qty = 0.0
    try:
        # Pre-flight: confirm real supply right now (paper daemon's own new check).
        maxb = await c.get_max_margin_loan(asset=asset)
        avail = float(maxb.get("amount", 0.0) or 0.0)
        print(f"  pre-flight: {avail:.4f} {asset} available to borrow")

        finfo = await c.futures_exchange_info()
        fsym = next(x for x in finfo["symbols"] if x["symbol"] == symbol)
        fprec = _prec(fsym["filters"], "LOT_SIZE", "stepSize")
        sinfo = await c.get_exchange_info()
        ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
        sprec = _prec(ssym["filters"], "LOT_SIZE", "stepSize")

        px = float((await c.futures_mark_price(symbol=symbol))["markPrice"])
        qty = round(notional / px, min(fprec, sprec))
        if qty > avail:
            print(f"  ❌ ABORT: need {qty} {asset} but only {avail:.4f} available. No orders sent.")
            return
        print(f"  mark=${px:.6f}  qty={qty} {asset}  (~${qty*px:.2f})")

        # ── 1. Borrow ──
        loan = await c.create_margin_loan(asset=asset, amount=str(qty))
        step_done["loan"] = True
        print(f"  ✅ 1/5 BORROWED {qty} {asset} — tranId={loan.get('tranId')}")

        # ── 2. Sell borrowed asset on margin (opens short-spot leg) ──
        so = await c.create_margin_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=qty,
            sideEffectType="NO_SIDE_EFFECT")
        step_done["short_spot"] = True
        print(f"  ✅ 2/5 SHORT-SPOT opened — orderId={so.get('orderId')} status={so.get('status')}")

        # ── 3. Long perp (delta-neutral hedge) ──
        await c.futures_change_leverage(symbol=symbol, leverage=lev)
        po = await c.futures_create_order(symbol=symbol, side="BUY", type="MARKET", quantity=qty)
        step_done["long_perp"] = True
        print(f"  ✅ 3/5 LONG-PERP opened — orderId={po.get('orderId')}")
        print("  → delta-neutral. Closing immediately (mechanics test, not a funding hold).")

        # ── 4. Close perp ──
        pc = await c.futures_create_order(
            symbol=symbol, side="SELL", type="MARKET", quantity=qty, reduceOnly=True)
        print(f"  ✅ 4/5 LONG-PERP closed — orderId={pc.get('orderId')}")

        # ── 5. Buy back + auto-repay loan in one call ──
        bo = await c.create_margin_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=qty,
            sideEffectType="AUTO_REPAY")
        print(f"  ✅ 5/5 SHORT-SPOT closed + loan repaid — orderId={bo.get('orderId')}")

        # ── Verify ──
        acc = await c.get_margin_account()
        row = next((a for a in acc.get("userAssets", []) if a["asset"] == asset), None)
        borrowed_after = float(row["borrowed"]) if row else 0.0
        pos = await c.futures_position_information(symbol=symbol)
        amt_after = float(next((p["positionAmt"] for p in pos if p["symbol"] == symbol), 0) or 0)
        print(f"\n  post-trade check: {asset} borrowed={borrowed_after} (want 0), "
              f"perp positionAmt={amt_after} (want 0)")
        if borrowed_after == 0 and amt_after == 0:
            print("\n" + "=" * 66)
            print("✅ FULL BORROW → SHORT-SPOT → LONG-PERP → CLOSE → REPAY VALIDATED.")
            print("   Real money, real API calls, clean round trip. Mechanics proven.")
            print("=" * 66)
        else:
            print("\n⚠️  Round trip completed but residuals remain — check manually.")
    except Exception as e:
        print(f"\n❌ FAILED at {step_done}: {type(e).__name__}: {str(e)[:200]}")
        print("  Attempting best-effort unwind...")
        try:
            if step_done["long_perp"]:
                await c.futures_create_order(
                    symbol=symbol, side="SELL", type="MARKET", quantity=qty, reduceOnly=True)
                print("  unwound perp leg")
        except Exception as e2:
            print(f"  perp unwind failed: {str(e2)[:120]}")
        try:
            if step_done["short_spot"]:
                await c.create_margin_order(
                    symbol=symbol, side="BUY", type="MARKET", quantity=qty,
                    sideEffectType="AUTO_REPAY")
                print("  unwound short-spot leg + repaid loan")
            elif step_done["loan"]:
                await c.repay_margin_loan(asset=asset, amount=str(qty))
                print("  repaid outstanding loan (no spot leg had opened)")
        except Exception as e3:
            print(f"  ⚠️  UNWIND FAILED — manual intervention needed: {str(e3)[:150]}")
    finally:
        await c.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
