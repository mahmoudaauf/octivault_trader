#!/usr/bin/env python3
"""Read-only Binance spot-fill audit. No orders, transfers, or subscriptions.

Fetches complete available symbol histories using trade-ID pagination. FIFO is
an account cost-basis calculation, NOT strategy attribution. Unknown opening
inventory and commissions in other currencies remain unresolved, never zero.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

D = Decimal
ZERO = D("0")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


async def fetch_symbol(client, symbol: str, after_id: int = -1,
                       page_size: int = 1000, max_pages: int = 100) -> list[dict]:
    rows, cursor = [], after_id + 1
    for _ in range(max_pages):
        page = await client.get_my_trades(symbol=symbol, fromId=cursor, limit=page_size)
        if not isinstance(page, list):
            raise ValueError("unexpected trade-history response")
        if not page:
            return rows
        if any(r.get("symbol") != symbol or int(r["id"]) < cursor for r in page):
            raise ValueError("trade history did not advance or changed symbol")
        rows.extend(page)
        cursor = max(int(r["id"]) for r in page) + 1
        if len(page) < page_size:
            return rows
    raise ValueError("history pagination limit reached; refusing incomplete history")


def fifo_audit(fills: list[dict], start_ms: int, end_ms: int) -> dict:
    """USDT-quoted fills only, deduplicated by symbol/trade ID.

    Base-asset BUY fees reduce units received; their cost is already included
    in quote paid and must not be subtracted twice. Base-asset SELL fees consume
    extra inventory. Quote fees increase entry cost or reduce exit proceeds.
    Other-asset fees have no historical conversion here: affected P&L is null.
    Actual execution prices already include execution slippage.
    """
    unique = {}
    duplicates = 0
    for r in fills:
        key = (r["symbol"], int(r["id"]))
        if key in unique:
            if unique[key] != r:
                raise ValueError("conflicting duplicate exchange fill")
            duplicates += 1
        unique[key] = r
    lots = defaultdict(deque)
    fees = defaultdict(lambda: ZERO)
    orders = {}
    unknown = []
    for r in sorted(unique.values(), key=lambda r: (int(r["time"]), r["symbol"], int(r["id"]))):
        ts, sym = int(r["time"]), r["symbol"]
        if ts > end_ms:
            continue
        if not sym.endswith("USDT"):
            raise ValueError("FIFO audit supports USDT quote only")
        base = sym[:-4]
        qty, price = D(str(r["qty"])), D(str(r["price"]))
        quote = D(str(r.get("quoteQty", qty * price)))
        fee, asset = D(str(r["commission"])), r["commissionAsset"]
        if any(not n.is_finite() for n in (qty, price, quote, fee)) or min(qty, price, quote) <= 0 or fee < 0:
            raise ValueError("invalid exchange fill amounts")
        if type(r["isBuyer"]) is not bool:
            raise ValueError("isBuyer must be an exchange boolean")
        in_window = start_ms <= ts <= end_ms
        if in_window:
            fees[asset] += fee
        fee_known = fee == 0 or asset in (base, "USDT")
        if not fee_known:
            unknown.append({"symbol": sym, "trade_id": r["id"], "asset": asset,
                            "amount": str(fee), "time": ts})
        if r["isBuyer"]:
            received = qty - (fee if asset == base else ZERO)
            if received <= 0:
                raise ValueError("base commission consumed the purchase")
            cost = quote + (fee if asset == "USDT" else ZERO)
            lots[sym].append({"qty": received, "cost": cost, "known": fee_known})
            continue
        consumed = qty + (fee if asset == base else ZERO)
        remaining, cost, known = consumed, ZERO, fee_known
        while remaining > 0 and lots[sym]:
            lot = lots[sym][0]
            take = min(remaining, lot["qty"])
            portion = lot["cost"] * take / lot["qty"]
            cost += portion
            known = known and lot["known"]
            lot["qty"] -= take
            lot["cost"] -= portion
            remaining -= take
            if lot["qty"] == 0:
                lots[sym].popleft()
        known = known and remaining == 0
        if not in_window:
            continue
        proceeds = quote - (fee if asset == "USDT" else ZERO)
        key = (sym, int(r["orderId"]))
        order = orders.setdefault(key, {"symbol": sym, "order_id": key[1], "time": ts,
                                       "cost": ZERO, "proceeds": ZERO, "known": True,
                                       "unmatched_qty": ZERO, "fill_count": 0})
        order["time"] = max(ts, order["time"])
        order["cost"] += cost
        order["proceeds"] += proceeds
        order["known"] = order["known"] and known
        order["unmatched_qty"] += remaining
        order["fill_count"] += 1
    closed = []
    for order in sorted(orders.values(), key=lambda r: r["time"]):
        closed.append({"symbol": order["symbol"], "order_id": order["order_id"],
                       "ts": datetime.fromtimestamp(order["time"] / 1000, timezone.utc).isoformat(),
                       "cost_basis_usdt": float(order["cost"]),
                       "net_proceeds_usdt": float(order["proceeds"]),
                       "net_pnl_usdt": float(order["proceeds"] - order["cost"]) if order["known"] else None,
                       "unmatched_quantity": float(order["unmatched_qty"]),
                       "fill_count": order["fill_count"], "costs_resolved": order["known"]})
    resolved = [r for r in closed if r["costs_resolved"]]
    return {"scope": "Selected-symbol account FIFO; not strategy attribution or whole-account reconciliation",
            "unique_fills": len(unique), "duplicates_removed": duplicates,
            "sell_orders": closed, "resolved_sell_orders": len(resolved),
            "unresolved_sell_orders": len(closed) - len(resolved),
            "resolved_subset_net_usdt": sum(r["net_pnl_usdt"] for r in resolved),
            "commissions_in_window": {k: str(v) for k, v in fees.items()},
            "unconverted_commissions": unknown,
            "open_lots": {s: {"qty": float(sum((l["qty"] for l in ls), ZERO)),
                               "recorded_cost_usdt": float(sum((l["cost"] for l in ls), ZERO))}
                          for s, ls in lots.items() if ls},
            "limitations": ["Opening transfers, Earn movements and fee-token inventory movements are not reconstructed.",
                            "Available exchange history can omit older cost basis; unmatched sells are excluded.",
                            "Open inventory is not marked to market; this is not total account profit."]}


def reconcile_strategy(ledger: list[dict], fills: list[dict], fee_prices: dict) -> dict:
    """Match legacy ledger episodes to unique exchange orders by time/price.

    Legacy rows lack order IDs: matches remain inferred, and ambiguous matches
    are rejected. Fee-token valuation uses a historical minute's low/high,
    producing a P&L interval instead of invented exact USDT commissions.
    """
    orders = defaultdict(list)
    for fill in fills:
        orders[(fill["symbol"], int(fill["orderId"]))].append(fill)
    used, results = set(), []

    def order_time(fs):
        return min(int(f["time"]) for f in fs) / 1000

    def qty(fs):
        return sum(float(f["qty"]) for f in fs)

    def quote(fs):
        return sum(float(f.get("quoteQty", float(f["qty"]) * float(f["price"]))) for f in fs)

    def nonbase_fees(fs, base):
        low = high = 0.0
        for f in fs:
            amount, asset = float(f["commission"]), f["commissionAsset"]
            if amount == 0 or asset == base:
                continue
            if asset == "USDT":
                low += amount
                high += amount
            else:
                key = f"{asset}USDT:{int(f['time']) // 60000 * 60000}"
                price = fee_prices.get(key)
                if not price:
                    return None
                low += amount * price["low"]
                high += amount * price["high"]
        return low, high

    for row in ledger:
        if row.get("mode") != "live" or "net_pct" not in row or row.get("kind"):
            continue
        sym = row["symbol"]
        end = datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).timestamp()
        start = end - float(row["held_h"]) * 3600
        matches = []
        for buyer, when, price, tolerance in ((True, start, row["entry_price"], 180),
                                               (False, end, row["exit_price"], 1200)):
            recorded_id = row.get("entry_order_id" if buyer else "exit_order_id")
            candidates = [(key, fs) for key, fs in orders.items()
                          if key[0] == sym and key not in used and all(f["isBuyer"] == buyer for f in fs)
                          and ((recorded_id is not None and key[1] == int(recorded_id))
                               or (recorded_id is None and abs(order_time(fs) - when) <= tolerance
                                   and abs(quote(fs) / qty(fs) / float(price) - 1) <= .001))]
            matches.append(candidates)
        result = {"symbol": sym, "ledger_ts": row["ts"], "attribution": "inferred_unique_time_price_match"}
        if row.get("entry_order_id") is not None and row.get("exit_order_id") is not None:
            result["attribution"] = "persisted_exchange_order_ids"
        if any(len(m) != 1 for m in matches):
            result["status"] = "unresolved_order_match"
            results.append(result)
            continue
        (bk, buy), (sk, sell) = matches[0][0], matches[1][0]
        if order_time(sell) < order_time(buy):
            result["status"] = "invalid_order_sequence"
            results.append(result)
            continue
        used.update((bk, sk))
        result.update({"buy_order_id": bk[1], "sell_order_id": sk[1],
                       "entry_ts": datetime.fromtimestamp(order_time(buy), timezone.utc).isoformat(),
                       "exit_ts": datetime.fromtimestamp(order_time(sell), timezone.utc).isoformat(),
                       "buy_quantity": qty(buy), "sell_quantity": qty(sell),
                       "ledger_quantity": row["qty"],
                       "ledger_quantity_difference": float(row["qty"]) - qty(sell)})
        base = sym[:-4]
        received = qty(buy) - sum(float(f["commission"]) for f in buy if f["commissionAsset"] == base)
        consumed = qty(sell) + sum(float(f["commission"]) for f in sell if f["commissionAsset"] == base)
        result["remaining_base_quantity"] = received - consumed
        bf, sf = nonbase_fees(buy, base), nonbase_fees(sell, base)
        if (received > 0 and consumed > received + 1e-10
                and abs(qty(buy) - qty(sell)) < 1e-10 and bf is not None and sf is not None):
            # Selling the full gross BNB purchase while also paying BNB fees
            # draws those fees from an existing BNB buffer. Attribute that
            # expense at each actual BNBUSDT fill price; do not call it free.
            base_fee_value = sum(float(f["commission"]) * float(f["price"])
                                 for f in buy + sell if f["commissionAsset"] == base)
            result.update({"status": "matched_with_fee_valuation_bounds",
                           "fee_buffer_base_used": consumed - received,
                           "remaining_base_quantity": 0.0,
                           "allocated_entry_cost_usdt": quote(buy),
                           "realized_net_low_usdt": quote(sell) - quote(buy) - base_fee_value - bf[1] - sf[1],
                           "realized_net_high_usdt": quote(sell) - quote(buy) - base_fee_value - bf[0] - sf[0],
                           "remaining_cost_low_usdt": 0.0, "remaining_cost_high_usdt": 0.0})
        elif received <= 0 or consumed > received + 1e-10:
            result["status"] = "opening_inventory_used_for_sale_or_base_fees"
        elif bf is None or sf is None:
            result["status"] = "unconverted_fee_currency"
        else:
            proportion = consumed / received
            cost = quote(buy) * proportion
            result.update({"status": "matched_with_fee_valuation_bounds",
                           "allocated_entry_cost_usdt": cost,
                           "realized_net_low_usdt": quote(sell) - sf[1] - cost - bf[1] * proportion,
                           "realized_net_high_usdt": quote(sell) - sf[0] - cost - bf[0] * proportion,
                           "remaining_cost_low_usdt": (quote(buy) + bf[0]) * (1 - proportion),
                           "remaining_cost_high_usdt": (quote(buy) + bf[1]) * (1 - proportion)})
        results.append(result)
    resolved = [r for r in results if r["status"] == "matched_with_fee_valuation_bounds"]
    return {"trades": results, "matched_with_cost_bounds": len(resolved),
            "unresolved_trades": len(results) - len(resolved),
            "resolved_subset_net_low_usdt": sum(r["realized_net_low_usdt"] for r in resolved),
            "resolved_subset_net_high_usdt": sum(r["realized_net_high_usdt"] for r in resolved),
            "limitations": ["Legacy order attribution is inferred, not proven by persisted order IDs.",
                            "Fee-token historical minute ranges are valuation bounds, not exact conversion executions.",
                            "Residual inventory must be marked separately; unresolved trades are not counted as zero profit."]}


async def refresh_cache(path: Path, symbols: list[str], fee_start_ms: int = 0) -> dict:
    """Persist only a completely successful read. Do not expose API errors/keys."""
    from dotenv import load_dotenv
    from binance import AsyncClient
    from exchange_resilience import dns_session_params

    load_dotenv()
    key, secret = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    if not key or not secret:
        raise ValueError("Binance read credentials unavailable")
    cached = json.loads(path.read_text()) if path.exists() else {"fills": [], "covered_symbols": []}
    client = await AsyncClient.create(key, secret, requests_params={"timeout": 20},
                                      session_params=dns_session_params())
    try:
        fills = list(cached["fills"])
        for symbol in symbols:
            last_id = max((int(r["id"]) for r in fills if r["symbol"] == symbol), default=-1)
            fresh = await fetch_symbol(client, symbol, last_id)
            fills.extend(fresh)
        fee_prices = dict(cached.get("fee_prices", {}))
        for fill in fills:
            asset = fill["commissionAsset"]
            if (int(fill["time"]) < fee_start_ms or float(fill["commission"]) == 0
                    or asset in (fill["symbol"][:-4], "USDT")):
                continue
            minute = int(fill["time"]) // 60000 * 60000
            fee_symbol = asset + "USDT"
            price_key = f"{fee_symbol}:{minute}"
            if price_key not in fee_prices:
                candles = await client.get_klines(symbol=fee_symbol, interval="1m",
                                                  startTime=minute, endTime=minute + 59999, limit=1)
                if len(candles) != 1 or int(candles[0][0]) != minute:
                    raise ValueError("historical fee conversion candle unavailable")
                low, high = float(candles[0][3]), float(candles[0][2])
                if not 0 < low <= high or not all(map(math.isfinite, (low, high))):
                    raise ValueError("invalid historical fee price")
                fee_prices[price_key] = {"low": low, "high": high, "source": "historical_1m_range"}
        result = {"fetched_at": datetime.now(timezone.utc).isoformat(), "fills": fills,
                  "fee_prices": fee_prices,
                  "covered_symbols": sorted(set(symbols) | set(cached.get("covered_symbols", [])))}
        atomic_json(path, result)
        return result
    finally:
        await client.close_connection()
