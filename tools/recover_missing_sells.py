"""One-shot admin tool to recover/emit canonical TRADE_EXECUTED for missing SELL fills.

Usage examples:
  # CSV input (symbol,order_id) one-per-line
  python tools/recover_missing_sells.py --input missing.csv

  # Inline single recovery by client order id (requires symbol)
  python tools/recover_missing_sells.py --symbol BTCUSDT --client-order-id octi-1771339153457-tp_sl

Notes:
- This is a best-effort admin script that calls the exchange to fetch the authoritative
  order payload and then asks ExchangeTruthAuditor to apply the recovered fill.
- Must be run where BINANCE API keys/config are available (same env as your running system).
- Script will NOT place orders; it only attempts recovery/emission for fills that already exist on the exchange.
"""
import argparse
import asyncio
import csv
import logging
import sys
from types import SimpleNamespace
from typing import List, Tuple, Optional

from core.exchange_client import get_global_exchange_client
from core.shared_state import SharedState
from core.execution_manager import ExecutionManager
from core.exchange_truth_auditor import ExchangeTruthAuditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recover_missing_sells")


def _parse_csv(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if not row:
                continue
            if len(row) == 1:
                # single token -> treat as order id only (symbol required separately)
                out.append(("", row[0].strip()))
            else:
                out.append((row[0].strip(), row[1].strip()))
    return out


async def _recover_one(auditor: ExchangeTruthAuditor, ex_client, ss: SharedState, symbol: str, token: str) -> bool:
    """token may be order_id (numeric) or clientOrderId (string)"""
    try:
        # Try numeric order id first
        try:
            oid = int(token)
        except Exception:
            oid = None

        order = None
        if oid is not None and symbol:
            with contextlib.suppress(Exception):
                order = await ex_client.get_order(symbol, order_id=oid)
        if order is None and symbol:
            with contextlib.suppress(Exception):
                order = await ex_client.get_order(symbol, client_order_id=token)

        if order is None:
            logger.warning("Order not found on exchange: symbol=%s token=%s", symbol or "(none)", token)
            return False

        # Ensure auditor can apply the recovered fill
        applied = await auditor._apply_recovered_fill(order, reason="admin_replay", synthetic=False)
        if applied:
            logger.info("Recovered and applied order: symbol=%s order_id=%s client_order_id=%s", order.get("symbol"), order.get("orderId") or order.get("clientOrderId"))
        else:
            logger.info("Order found but not applied (already recorded?): %s", order.get("orderId") or token)
        return bool(applied)
    except Exception as e:
        logger.exception("Recovery failed for %s/%s: %s", symbol, token, e)
        return False


async def main_async(args):
    # obtain global exchange client (will error if keys/config not present)
    ex = None
    try:
        ex = get_global_exchange_client()
    except Exception as e:
        logger.error("Failed to acquire global exchange client: %s", e)
        return 2

    ss = SharedState()
    em = ExecutionManager(SimpleNamespace(), ss, ex)
    auditor = ExchangeTruthAuditor(config=SimpleNamespace(), shared_state=ss, exchange_client=ex, app=SimpleNamespace(execution_manager=em))

    tokens: List[Tuple[str, str]] = []
    if args.input:
        tokens.extend(_parse_csv(args.input))
    if args.client_order_id:
        tokens.append((args.symbol or "", args.client_order_id))
    if args.order_id:
        tokens.append((args.symbol or "", args.order_id))

    if not tokens:
        logger.error("No tokens/orders provided. Use --input or --order-id/--client-order-id")
        return 2

    success = 0
    for sym, tok in tokens:
        res = await _recover_one(auditor, ex, ss, sym or args.symbol or "", tok)
        success += 1 if res else 0

    logger.info("Recovery done: %d/%d succeeded", success, len(tokens))
    return 0


def main():
    p = argparse.ArgumentParser(description="Recover missing SELL TRADE_EXECUTED events from exchange truth")
    p.add_argument("--input", help="CSV file with rows: SYMBOL,ORDER_ID or SYMBOL,CLIENT_ORDER_ID")
    p.add_argument("--symbol", help="Symbol to use when providing single --order-id/--client-order-id")
    p.add_argument("--order-id", help="Order ID (numeric)")
    p.add_argument("--client-order-id", help="clientOrderId string")
    ns = p.parse_args()
    return asyncio.run(main_async(ns))


if __name__ == "__main__":
    sys.exit(main())
