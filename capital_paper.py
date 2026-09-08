#!/usr/bin/env python3
"""One forward-only research cycle, using public market data and virtual cash.

Frozen baseline: existing long-only Donchian(20, volume 2x) rule. Its earlier
backtest failed; this is a measured baseline, not a new profitable strategy.
No credentials, leverage, historical fill replay, or live order methods.
Snapshots include liquidation-value inventory and both execution costs.
"""
from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from exchange_trade_audit import atomic_json

STATE = Path("logs/profit_readiness/forward_paper.json")
CONFIG = {"version": "donchian20_volume2_long_baseline_v1", "initial_capital": 60.65,
          "sleeve_fraction": .20, "fee_per_side": .001, "slippage_per_side": .0005,
          "take_profit": .04, "stop_loss": .02, "max_hold_hours": 48,
          "max_daily_loss_fraction": .02, "max_drawdown_fraction": .10,
          "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT", "ADAUSDT", "INJUSDT"]}


def initial_state(now: float) -> dict:
    return {"config": copy.deepcopy(CONFIG), "started_at": now, "cash": CONFIG["initial_capital"],
            "position": None, "closed_trades": [], "seen_bars": {}, "last_cycle": None,
            "peak_nav": CONFIG["initial_capital"], "day": None, "day_start_nav": CONFIG["initial_capital"],
            "coverage_gaps": 0, "snapshots": [], "halted": False}


def breakout(candles: list, now: float):
    closed = [r for r in candles if int(r[6]) < now * 1000]
    if len(closed) < 21:
        return None, False
    bar, previous = closed[-1], closed[-21:-1]
    # Do not enter on a stale signal after downtime.
    recent = now * 1000 - int(bar[6]) <= 20 * 60 * 1000
    avg_volume = sum(float(r[5]) for r in previous) / 20
    signal = recent and avg_volume > 0 and float(bar[4]) > max(float(r[2]) for r in previous)
    signal = signal and float(bar[5]) >= 2 * avg_volume
    return int(bar[0]), bool(signal)


def liquidation_nav(state: dict, books: dict) -> float:
    pos = state["position"]
    if not pos:
        return state["cash"]
    bid = books[pos["symbol"]]["bid"]
    cfg = state["config"]
    return state["cash"] + pos["qty"] * bid * (1 - cfg["slippage_per_side"]) * (1 - cfg["fee_per_side"])


def advance(state: dict, books: dict, signals: dict, filters: dict, now: float) -> dict:
    """Pure deterministic state transition. Caller atomically persists it."""
    state = copy.deepcopy(state)
    cfg = state["config"]
    if state["last_cycle"] is not None and now <= state["last_cycle"]:
        return state
    for book in books.values():
        if not all(math.isfinite(v) for v in (book["bid"], book["ask"])) or not 0 < book["bid"] <= book["ask"]:
            raise ValueError("invalid book; refuse fabricated marks")
    if state["position"] and state["position"]["symbol"] not in books:
        raise ValueError("open position has no current mark")
    if state["last_cycle"] and now - state["last_cycle"] > 30 * 60:
        state["coverage_gaps"] += 1
    nav = liquidation_nav(state, books)
    day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    if state["day"] != day:
        state["day"], state["day_start_nav"] = day, nav
    state["peak_nav"] = max(state["peak_nav"], nav)
    drawdown = (state["peak_nav"] - nav) / state["peak_nav"]
    daily_loss = (state["day_start_nav"] - nav) / state["day_start_nav"]
    if drawdown >= cfg["max_drawdown_fraction"]:
        state["halted"] = True
    risk_stop = state["halted"] or daily_loss >= cfg["max_daily_loss_fraction"]
    pos = state["position"]
    if pos:
        bid = books[pos["symbol"]]["bid"]
        reason = None
        if risk_stop:
            reason = "risk_limit"
        elif bid <= pos["entry_price"] * (1 - cfg["stop_loss"]):
            reason = "observed_stop_loss"
        elif bid >= pos["entry_price"] * (1 + cfg["take_profit"]):
            reason = "observed_take_profit"
        elif now - pos["entry_time"] >= cfg["max_hold_hours"] * 3600:
            reason = "time_exit"
        if reason:
            price = bid * (1 - cfg["slippage_per_side"])
            f = filters[pos["symbol"]]
            if pos["qty"] < f["min_qty"] or pos["qty"] * price < max(5, f["min_notional"]):
                state["exit_blocked"] = "below_exchange_minimum; inventory remains marked"
            else:
                state.pop("exit_blocked", None)
                proceeds = pos["qty"] * price * (1 - cfg["fee_per_side"])
                state["cash"] += proceeds
                state["closed_trades"].append({"symbol": pos["symbol"], "entry_time": pos["entry_time"],
                                                "ts": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                                                "entry_price": pos["entry_price"], "exit_price": price,
                                                "qty": pos["qty"], "entry_cost_usdt": pos["entry_cost"],
                                                "net_pnl_usdt": proceeds - pos["entry_cost"],
                                                "mode": "forward_paper", "reason": reason})
                state["position"] = None
                state["last_exit"] = now
    for symbol in cfg["symbols"]:
        bar, signal = signals.get(symbol, (None, False))
        if bar is None or state["seen_bars"].get(symbol, -1) >= bar:
            continue
        state["seen_bars"][symbol] = bar
        if not signal or state["position"] or risk_stop or now - state.get("last_exit", 0) < 3600:
            continue
        f = filters[symbol]
        price = books[symbol]["ask"] * (1 + cfg["slippage_per_side"])
        # All positions share the same cash; no $100-per-symbol fictitious funds.
        budget = min(state["cash"], liquidation_nav(state, books) * cfg["sleeve_fraction"])
        raw = budget / (price * (1 + cfg["fee_per_side"]))
        step = Decimal(str(f["step"]))
        qty = float((Decimal(str(raw)) / step).to_integral_value(rounding=ROUND_DOWN) * step)
        if qty < f["min_qty"] or qty > f["max_qty"] or qty * price < max(5, f["min_notional"]):
            continue
        cost = qty * price * (1 + cfg["fee_per_side"])
        if cost > state["cash"]:
            continue
        state["cash"] -= cost
        state["position"] = {"symbol": symbol, "qty": qty, "entry_price": price,
                              "entry_cost": cost, "entry_time": now}
    state["last_cycle"] = now
    nav = liquidation_nav(state, books)
    state["snapshots"].append({"ts": now, "nav": nav, "cash": state["cash"]})
    state["latest_nav"] = nav
    return state


async def cycle(path: Path = STATE) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return await _cycle_locked(path)


async def _cycle_locked(path: Path) -> dict:
    from binance import AsyncClient
    from exchange_resilience import dns_session_params
    state = json.loads(path.read_text()) if path.exists() else initial_state(time.time())
    if state["config"] != CONFIG:
        raise ValueError("paper parameters changed; use a new experiment file")
    # Intentionally do not load .env or pass API credentials.
    client = await AsyncClient.create(requests_params={"timeout": 20}, session_params=dns_session_params())
    try:
        now = time.time()
        books, signals, filters = {}, {}, {}
        info = await client.get_exchange_info()
        by_symbol = {s["symbol"]: s for s in info["symbols"]}
        for symbol in CONFIG["symbols"]:
            s = by_symbol.get(symbol)
            if not s or s.get("status") != "TRADING":
                if state["position"] and state["position"]["symbol"] == symbol:
                    raise ValueError("held paper asset is not trading; manual research review needed")
                continue
            fs = {f["filterType"]: f for f in s["filters"]}
            lot = fs.get("MARKET_LOT_SIZE", fs["LOT_SIZE"])
            if float(lot["stepSize"]) <= 0:
                lot = fs["LOT_SIZE"]
            minimum = fs.get("NOTIONAL", fs.get("MIN_NOTIONAL", {}))
            filters[symbol] = {"step": float(lot["stepSize"]), "min_qty": float(lot["minQty"]),
                               "max_qty": float(lot["maxQty"]), "min_notional": float(minimum.get("minNotional", 5))}
            book = await client.get_order_book(symbol=symbol, limit=5)
            if not book.get("bids") or not book.get("asks"):
                raise ValueError("empty market book")
            books[symbol] = {"bid": float(book["bids"][0][0]), "ask": float(book["asks"][0][0])}
            candles = await client.get_klines(symbol=symbol, interval="1h", limit=25)
            signals[symbol] = breakout(candles, now)
        result = advance(state, books, signals, filters, now)
        atomic_json(path, result)
        return result
    finally:
        await client.close_connection()


if __name__ == "__main__":
    result = asyncio.run(asyncio.wait_for(cycle(), timeout=180))
    print(f"Forward paper: NAV=${result['latest_nav']:.4f}, closed={len(result['closed_trades'])}, gaps={result['coverage_gaps']}")
