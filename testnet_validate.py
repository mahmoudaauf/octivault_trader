#!/usr/bin/env python3
"""
Testnet execution validation — proves the carry perp-leg order path works with FAKE
money on Binance FUTURES testnet. Does a full open(short)→confirm→close cycle, exactly
the mechanics the live carry executor uses (set leverage, MARKET order, reduceOnly
close). Reports fills, realized PnL, fees. Zero real money.

Usage: python3 testnet_validate.py [SYMBOL] [NOTIONAL_USD]
"""
from __future__ import annotations

import asyncio
import os
import sys


async def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dotenv import load_dotenv

    load_dotenv()
    from binance import AsyncClient

    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    notional = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    lev = 2

    k = os.getenv("BINANCE_FUTURES_TESTNET_KEY")
    s = os.getenv("BINANCE_FUTURES_TESTNET_SECRET")
    if not k or not s:
        print("❌ futures testnet keys missing in .env")
        return
    c = await AsyncClient.create(k, s, testnet=True)
    print(f"🧪 FUTURES TESTNET — validating {symbol} carry perp leg (~${notional}, {lev}x, FAKE money)")
    try:
        # Quantity precision from exchange info.
        info = await c.futures_exchange_info()
        sym = next(x for x in info["symbols"] if x["symbol"] == symbol)
        step = next(f["stepSize"] for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
        prec = len(step.rstrip("0").split(".")[1]) if "." in step.rstrip("0") else 0

        mark = await c.futures_mark_price(symbol=symbol)
        px = float(mark["markPrice"])
        qty = round(notional / px, prec)
        print(f"  mark=${px:,.2f}  qty={qty} {symbol[:-4]}")

        await c.futures_change_leverage(symbol=symbol, leverage=lev)
        print(f"  ✅ leverage set to {lev}x")

        # OPEN short (the positive-funding carry perp leg).
        o = await c.futures_create_order(symbol=symbol, side="SELL", type="MARKET", quantity=qty)
        print(f"  ✅ OPEN short filled — orderId={o.get('orderId')} status={o.get('status')}")

        pos = await c.futures_position_information(symbol=symbol)
        p = next((x for x in pos if x["symbol"] == symbol), {})
        print(f"  position: amt={p.get('positionAmt')} entry={p.get('entryPrice')} "
              f"unrealized={p.get('unRealizedProfit')}")

        # CLOSE (reduceOnly buy).
        cl = await c.futures_create_order(
            symbol=symbol, side="BUY", type="MARKET", quantity=qty, reduceOnly=True)
        print(f"  ✅ CLOSE filled — orderId={cl.get('orderId')} status={cl.get('status')}")

        # Realized PnL + fees from trade history.
        trades = await c.futures_account_trades(symbol=symbol, limit=4)
        pnl = sum(float(t["realizedPnl"]) for t in trades[-2:])
        fees = sum(float(t["commission"]) for t in trades[-2:])
        print(f"\n  round-trip realized PnL: {pnl:+.4f} USDT   fees: {fees:.4f} USDT")
        print("\n" + "=" * 60)
        print("✅ TESTNET EXECUTION VALIDATED — perp open/close path works.")
        print("   The live carry executor's order mechanics are proven (fake money).")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {type(e).__name__}: {str(e)[:160]}")
    finally:
        await c.close_connection()


if __name__ == "__main__":
    asyncio.run(main())
