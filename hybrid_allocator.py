#!/usr/bin/env python3
"""
Hybrid capital allocator — safe yield CORE + capped auto-speculative SATELLITE.

WHAT THIS IS (and honestly is not)
----------------------------------
A barbell / core-satellite structure, operator-chosen with full disclosure:
  • CORE (default 80%) stays in Binance Simple Earn flexible USDT — safe yield,
    the account's only real (slow) growth. Never traded, never at risk here.
  • SATELLITE (default 20%) runs a simple AUTOMATED speculative trader.

The satellite has NO EDGE. It is negative expected value. Its entry rule is a
Donchian breakout — one of the rules this very project already FALSIFIED
(volatility_breakout_backtest.py). It is used precisely BECAUSE any rule here
loses on average, and breakout is simple and transparent. Do not expect profit.

THE POINT OF THIS FILE IS THE GUARDRAILS, NOT THE TRADING:
  1. THE WALL — the satellite can lose its whole allocation but can NEVER pull
     from the core. `redeem_simple_earn_flexible_product` is called ONLY in the
     one-time live setup, NEVER in the trading loop. Money only flows
     satellite -> core (win-sweep), never core -> satellite.
  2. WIN-SWEEP — satellite gains above target are swept back into the safe core.
  3. FLOOR — below the floor the satellite stops (can't meet min-notional).
  4. Double-gated live (MODE=live AND arm file), kill-switch, drawdown-halt.

Production hardening (2026-08-17, gap audit G1-G8):
  G1 EXCHANGE-SIDE STOP — on every live open an OCO (take-profit + stop-loss)
     rests on Binance, so the stop is enforced even if this process is dead or
     the machine is asleep. The poll loop is only a backup + the time-stop.
  G2 RESTART RECONCILIATION — on live startup, verify the tracked position
     against real balances; adopt/alert on stray holdings; protect an
     unprotected adopted position.
  G3 SINGLE-INSTANCE LOCK — an flock pidfile prevents two daemons trading one
     account.
  G5 API RETRY — order/price calls retry with backoff; a filled-but-errored buy
     is reconciled instead of leaving an untracked position.
  G6 paper cash now applies fees (was inconsistent with the ledger).
  G7 drawdown-halt counts LIVE trades only (paper history no longer trips live).
  G8 live sizing is capped to the satellite budget, not all spot USDT.

Modes (HYBRID_MODE, default paper):
  paper  — simulate the satellite against a NOTIONAL balance; move NO real money.
  dryrun — log the exact real orders it would place; send nothing.
  live   — real spot orders; DOUBLE-GATED on MODE=live AND logs/hybrid_live_armed.

Objectives (HYBRID_OBJECTIVE, default satellite):
  satellite — the negative-EV breakout trader described above.
  allocate  — deploy/compound/account ONLY. Places no speculative trades and
              never calls redeem, so the wall is absolute rather than merely
              enforced. Positive-EV: it captures yield on capital that would
              otherwise sit at 0%. It does four things, all autonomous:
                • sweeps idle spot USDT into flexible earn
                • sweeps idle NON-USDT holdings (BTC) into their own flexible
                  products — added 2026-09-04; the BTC hold earned 0% until then
                • picks the BEST-APR product each cycle rather than whichever
                  one Binance listed first, and logs the rates it saw
                • detects external contributions from deposit history and
                  records an honest NAV curve keeping deposits out of returns

Usage:
  python3 hybrid_allocator.py          # run the daemon (mode + objective from env)
  python3 hybrid_allocator.py report   # satellite track record
  python3 hybrid_allocator.py nav      # NAV curve: contributions vs returns
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

STATE = os.getenv("HYBRID_STATE_FILE", "logs/hybrid_state.json")
LEDGER = os.getenv("HYBRID_LEDGER_FILE", "logs/hybrid_ledger.jsonl")

# ── Allocation ───────────────────────────────────────────────────────────────
CORE_FRACTION = float(os.getenv("HYBRID_CORE_FRACTION", "0.80"))   # 80% safe
SWEEP_THRESHOLD_USD = float(os.getenv("HYBRID_SWEEP_THRESHOLD_USD", "0.5"))
SATELLITE_FLOOR_USD = float(os.getenv("HYBRID_SATELLITE_FLOOR_USD", "5.0"))
TRADE_FRACTION = float(os.getenv("HYBRID_TRADE_FRACTION", "0.95"))  # of sat budget

# ── Satellite trading ────────────────────────────────────────────────────────
_DEFAULT_WATCH = "SOLUSDT,AVAXUSDT,LINKUSDT,DOGEUSDT,ADAUSDT,SUIUSDT,APTUSDT,ARBUSDT,INJUSDT,NEARUSDT"
# Assets the satellite must NEVER touch — long-term buy-and-hold buckets that
# live in the same spot wallet. Without this, `_reconcile`'s stray-holding
# adoption would grab the BTC hold and put a -2% stop on a position meant to be
# held for years. Filtered out of the watchlist, so it guards entry AND adoption.
PROTECTED_ASSETS = {s.strip().upper() for s in os.getenv("HYBRID_PROTECTED_ASSETS", "BTC").split(",") if s.strip()}
WATCHLIST = [s.strip().upper() for s in os.getenv("HYBRID_WATCHLIST", _DEFAULT_WATCH).split(",") if s.strip()]
WATCHLIST = [s for s in WATCHLIST if s[:-4] not in PROTECTED_ASSETS]
INTERVAL = os.getenv("HYBRID_INTERVAL", "1h")
BREAKOUT_LOOKBACK = int(os.getenv("HYBRID_BREAKOUT_LOOKBACK", "20"))
VOLUME_MULT = float(os.getenv("HYBRID_VOLUME_MULT", "2.0"))
TP_PCT = float(os.getenv("HYBRID_TP_PCT", "4.0"))   # take-profit %
SL_PCT = float(os.getenv("HYBRID_SL_PCT", "2.0"))   # stop-loss %
# OCO stop-limit sits this % below the stop TRIGGER, so a triggered stop still
# fills through a fast move instead of resting as an unfilled limit.
STOP_LIMIT_OFFSET_PCT = float(os.getenv("HYBRID_STOP_LIMIT_OFFSET_PCT", "0.8"))
MAX_HOLD_H = float(os.getenv("HYBRID_MAX_HOLD_H", "48"))
FEE_RT_PCT = float(os.getenv("HYBRID_FEE_RT_PCT", "0.2"))  # round-trip spot taker
COOLDOWN_H = float(os.getenv("HYBRID_COOLDOWN_H", "1"))
POLL_MIN = float(os.getenv("HYBRID_POLL_MIN", "15"))
# Hard ceiling on ONE allocate cycle. A healthy cycle is a few seconds; this is
# not a performance tuning knob but a hang-breaker. Deliberately well under the
# supervisor's 25m stall watchdog so the daemon recovers itself in-process and
# the watchdog stays a last resort. See the BOUNDED CYCLE note in run_allocate.
CYCLE_TIMEOUT_S = float(os.getenv("HYBRID_CYCLE_TIMEOUT_S", "300"))

_INTERVAL_HOURS = {"5m": 5 / 60, "15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0, "1d": 24.0}

# ── Execution mode + safety ──────────────────────────────────────────────────
MODE = os.getenv("HYBRID_MODE", "paper").lower()
LIVE_ARM_FILE = os.getenv("HYBRID_LIVE_ARM_FILE", "logs/hybrid_live_armed")
KILL_FILE = os.getenv("HYBRID_KILL_FILE", "logs/hybrid.stop")
PIDFILE = os.getenv("HYBRID_PIDFILE", "logs/hybrid.pid")          # G3
MAX_DD_PCT = float(os.getenv("HYBRID_MAX_DD_PCT", "100.0"))

# ── Objective: what job this daemon is doing ─────────────────────────────────
# "satellite" — the original negative-EV breakout trader (capped, disclosed).
# "allocate"  — deploy/compound/account only: sweep idle USDT into earn, detect
#               external contributions, and record an honest NAV curve. Places
#               NO speculative trades and NEVER redeems from earn, so it is
#               strictly positive-EV (it captures yield that would otherwise sit
#               idle) and scales linearly with capital.
OBJECTIVE = os.getenv("HYBRID_OBJECTIVE", "satellite").lower()
NAV_FILE = os.getenv("HYBRID_NAV_FILE", "logs/nav_history.jsonl")
# Cash left in spot rather than swept — covers fees/min-notional friction.
IDLE_BUFFER_USD = float(os.getenv("HYBRID_IDLE_BUFFER_USD", "0.10"))
# Minimum sweep worth doing (Binance flexible min purchase is 0.01 USDT).
MIN_SWEEP_USD = float(os.getenv("HYBRID_MIN_SWEEP_USD", "0.05"))
# Fraction of a detected contribution to route into BTC. Default 0: a bot must
# not take a directional position unless the operator explicitly asks.
CONTRIB_BTC_PCT = float(os.getenv("HYBRID_CONTRIB_BTC_PCT", "0.0"))
# ── Autonomous yield (2026-09-04) ────────────────────────────────────────────
# Assets auto-subscribed to flexible earn. USDT was the only one for months,
# which left the BTC hold sitting at 0% indefinitely. Subscribing an asset does
# NOT change NAV (see _nav_snapshot: earn quantities are counted alongside spot),
# and flexible earn redeems on demand, so this is yield with no lock-up.
EARN_ASSETS = [a.strip().upper() for a in
               os.getenv("HYBRID_EARN_ASSETS", "USDT,BTC").split(",") if a.strip()]
# Log every available flexible rate each cycle. Read-only; it is how a better
# promotional rate becomes visible instead of being silently missed.
RATE_SCAN = os.getenv("HYBRID_RATE_SCAN", "1") not in ("0", "false", "False")
# DELIBERATELY NOT IMPLEMENTED: migrating the existing core between earn
# products. It would require REDEEM, which the allocate objective never calls —
# that absence is the wall, not a policy that could be flipped by a flag. New
# money already routes to the best-APR product every cycle, so the rate is
# captured over time without ever unwinding a position. A knob here that did
# nothing would be worse than no knob at all.
NAV_ASSETS = [a.strip().upper() for a in
              os.getenv("HYBRID_NAV_ASSETS", "BTC,BNB,AVAX,SOL,LINK,INJ,NEAR,ADA,SUI,APT,ARB,DOGE").split(",") if a.strip()]
API_RETRIES = int(os.getenv("HYBRID_API_RETRIES", "3"))          # G5
API_RETRY_DELAY_S = float(os.getenv("HYBRID_API_RETRY_DELAY_S", "2.0"))

_EARN_PRODUCT_ID = None  # resolved once at startup (USDT flexible, e.g. "USDT001")
_filters_cache: dict[str, tuple] = {}  # symbol -> (qty_step, min_notional, tick)
_lock_handle = None      # keep the flock fd alive for the process lifetime


def _is_live() -> bool:
    return MODE == "live" and os.path.exists(LIVE_ARM_FILE)


def _killed() -> bool:
    return os.path.exists(KILL_FILE)


# ── G3: single-instance lock ─────────────────────────────────────────────────
def _acquire_lock() -> bool:
    """Exclusive flock on a pidfile. Prevents two daemons trading one account
    (double orders, racing sweeps, state clobbering). Returns False if held."""
    global _lock_handle
    import fcntl
    os.makedirs(os.path.dirname(PIDFILE) or ".", exist_ok=True)
    fh = open(PIDFILE, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fh.seek(0)
    fh.write(str(os.getpid()))
    fh.truncate()
    fh.flush()
    _lock_handle = fh  # held until process exit
    return True


def _current_drawdown_pct(live_only: bool = True) -> float:
    """Peak-to-current drawdown over summed net_pct. G7: by default counts only
    LIVE trades, so a long PAPER loss history can't trip the live drawdown-halt
    the instant you arm live."""
    if not os.path.exists(LEDGER):
        return 0.0
    cum = peak = 0.0
    with open(LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if live_only and rec.get("mode") != "live":
                continue
            try:
                cum += float(rec.get("net_pct", 0.0))
            except (TypeError, ValueError):
                continue
            peak = max(peak, cum)
    return max(0.0, peak - cum)


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


def _log_trade(rec: dict):
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return (int(qty / step)) * step


def _round_price(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(int(price / tick) * tick, 8)


# ── G5: retry wrapper for flaky API calls ────────────────────────────────────
async def _retry(fn, *args, **kwargs):
    """Await fn(*args, **kwargs) with bounded retry/backoff. Re-raises the last
    error so callers still see a genuine failure."""
    last = None
    for attempt in range(API_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt < API_RETRIES - 1:
                await asyncio.sleep(API_RETRY_DELAY_S * (attempt + 1))
    raise last


async def _symbol_filters(client, symbol: str) -> tuple[float, float, float]:
    """(qty_step, min_notional, price_tick) for a SPOT symbol. Cached."""
    if symbol in _filters_cache:
        return _filters_cache[symbol]
    sinfo = await client.get_exchange_info()
    ssym = next(x for x in sinfo["symbols"] if x["symbol"] == symbol)
    step = next(float(f["stepSize"]) for f in ssym["filters"] if f["filterType"] == "LOT_SIZE")
    tick = 0.0
    min_notional = 5.0
    for f in ssym["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = float(f["tickSize"])
        if f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
            min_notional = float(f.get("minNotional", f.get("notional", 5.0)))
    _filters_cache[symbol] = (step, min_notional, tick)
    return _filters_cache[symbol]


# ── Earn (core) helpers — READ + one-directional writes ──────────────────────
def _parse_tiers(row: dict) -> list[tuple[float, float, float]]:
    """Binance's balance-tier rates as sorted [(lo, hi, apr)].

    The payload keys look like "0-500USDT" / "500-1000USDT". A tier that cannot
    be parsed is dropped rather than guessed at: a wrong bound silently
    misprices every rate comparison that follows.
    """
    tiers = []
    for k, v in (row.get("tierAnnualPercentageRate") or {}).items():
        m = re.match(r"^\s*([\d.]+)\s*-\s*([\d.]+)", str(k))
        if not m:
            continue
        try:
            tiers.append((float(m.group(1)), float(m.group(2)), float(v)))
        except (TypeError, ValueError):
            continue
    return sorted(tiers)


def _effective_apr(base_apr: float, tiers: list, amount: float) -> float:
    """Blended APR actually earned on `amount`, applying balance tiers.

    THIS IS NOT THE HEADLINE RATE. Binance pays a bonus rate on a first slice of
    the balance and the base rate above it: USDT flexible advertises
    latestAnnualPercentageRate ~2.94% while paying 4.00% on the first 500 USDT.
    Ranking products by the headline field therefore both understates what a
    small balance earns and can pick the WRONG product — one with a better
    headline but no bonus tier. For a sub-500 USDT account the tier IS the rate.

    Falls back to the base rate when there are no tiers or the amount is unknown.
    """
    if amount <= 0 or not tiers:
        return base_apr
    weighted = covered = 0.0
    for lo, hi, apr in tiers:
        if amount <= lo:
            break
        portion = min(amount, hi) - lo
        if portion > 0:
            weighted += portion * apr
            covered += portion
    if covered < amount:                      # remainder earns the base rate
        weighted += (amount - covered) * base_apr
    return weighted / amount if amount else base_apr


async def _earn_products(client, asset: str, amount: float = 0.0) -> list[dict]:
    """Subscribable flexible products for an asset, best EFFECTIVE APR first.

    `amount` is the balance the rate would be earned on, so tiered products are
    ranked by what they would actually pay us rather than by their headline.

    Returns [] rather than raising: a product-list outage must degrade to "keep
    using the product we already resolved", never to a crash in the daemon loop.
    """
    try:
        lst = await _retry(client.get_simple_earn_flexible_product_list, asset=asset)
        rows = lst.get("rows", []) if isinstance(lst, dict) else (lst or [])
    except Exception as e:
        print(f"  [EARN-PRODUCTS-FAIL] {asset}: {str(e)[:70]}")
        return []
    out = []
    for p in rows:
        if str(p.get("asset", "")).upper() != asset.upper():
            continue
        if p.get("canPurchase") is False or str(p.get("status", "PURCHASING")).upper() == "END":
            continue
        try:
            base = float(p.get("latestAnnualPercentageRate", 0.0) or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        tiers = _parse_tiers(p)
        out.append({"productId": p.get("productId"), "asset": asset.upper(),
                    "apr": _effective_apr(base, tiers, amount),
                    "base_apr": base, "tiers": tiers,
                    "min": float(p.get("minPurchaseAmount", 0.0) or 0.0)})
    return sorted(out, key=lambda r: -r["apr"])


async def _resolve_product_id(client, asset: str = "USDT",
                             amount: float = 0.0) -> str | None:
    """Best-EFFECTIVE-APR flexible product id for an asset.

    Was "first product returned", which quietly pinned the core to whatever
    Binance happened to list first and ignored better promotional terms. Then it
    ranked on the headline rate, which understates a small balance: the tier
    bonus is the real rate below 500 USDT. Ranking on the effective rate for the
    amount we actually hold is the only comparison that means anything.
    """
    products = await _earn_products(client, asset, amount)
    if not products:
        return None
    best = products[0]
    if RATE_SCAN:
        tier_note = ""
        if best["tiers"] and abs(best["apr"] - best["base_apr"]) > 1e-9:
            tier_note = (f" [tier bonus: headline is {best['base_apr']*100:.2f}%, "
                         f"you earn {best['apr']*100:.2f}% on ${amount:,.2f}]")
        others = (" ; also offered "
                  + ", ".join(f"{p['apr']*100:.2f}%" for p in products[1:4])) if len(products) > 1 else " (only product)"
        print(f"  [RATE-SCAN] {asset}: using {best['apr']*100:.2f}% APR "
              f"(product {best['productId']}){others}{tier_note}")
    return best["productId"]


async def _earn_positions(client) -> dict | None:
    """{ASSET: quantity} across ALL flexible earn positions, or None if unread.

    Reads every asset, not just USDT, because the sweep now subscribes BTC too.
    A subscribed asset LEAVES the spot balance, so if NAV counted only spot the
    machine would book a ~$12 collapse the moment it did its job correctly.
    """
    try:
        pos = await _retry(client.get_simple_earn_flexible_product_position)
        rows = pos.get("rows", []) if isinstance(pos, dict) else (pos or [])
    except Exception as e:
        print(f"  [EARN-READ-FAIL] {str(e)[:80]} — UNKNOWN (not zero)")
        return None
    out: dict[str, float] = {}
    for p in rows:
        asset = str(p.get("asset", "")).upper()
        if not asset:
            continue
        try:
            out[asset] = out.get(asset, 0.0) + float(p.get("totalAmount", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return out


async def _earn_usdt(client):
    """Total USDT in flexible earn, or None if it could NOT be read.

    Never 0.0 on failure: this is the largest component of NAV, so a failed read
    would record a ~$48 NAV collapse in the equity curve and — on a first
    snapshot — anchor the baseline to it, manufacturing enormous phantom growth
    afterwards. Same failure this file already hit with sweep timing.
    """
    positions = await _earn_positions(client)
    if positions is None:
        return None
    return positions.get("USDT", 0.0)


# ── Resilience: shared with the other daemons ────────────────────────────────
# These three lived here first and were duplicated into exchange_resilience.py
# for the carry/delisting/market-maker daemons. Two copies of a fix is how one
# of them silently rots, so this file now delegates. Thin wrappers keep the
# existing private names and call signatures.
from exchange_resilience import (                       # noqa: E402
    create_client_with_retry as _shared_create_client,
    dns_session_params as _shared_dns_params,
    resync_clock as _shared_resync,
)


def _dns_session_params() -> dict:
    """OS DNS resolver + cache; see exchange_resilience.dns_session_params."""
    return _shared_dns_params()


async def _resync_clock(client) -> bool:
    """Re-derive the Binance timestamp offset; see exchange_resilience."""
    return await _shared_resync(client, "hybrid")


async def _create_client_with_retry(async_client_cls, max_delay_s: float = 300.0):
    """Build the client, waiting out an outage; see exchange_resilience."""
    return await _shared_create_client(async_client_cls, label="hybrid",
                                       max_delay_s=max_delay_s)


async def _read_balance(client, asset: str):
    """Raw balance dict for an asset, or None if it could NOT be read.

    None means UNKNOWN and must never be coerced to zero. Swallowing read errors
    into 0.0 caused two real failures: (1) the sleeve sat idle for 12h believing
    it had $0.00 while $9.62 was in spot; (2) far worse, `_detect_exchange_close`
    reads this to decide whether the coin is gone — a failed read looked like
    "OCO fired", which would book a FABRICATED close and abandon a live position.
    """
    try:
        bal = await _retry(client.get_asset_balance, asset=asset)
    except Exception as e:
        # A signature/timestamp rejection is self-healable — resync and retry
        # once before giving up, so drift costs one read instead of a whole day.
        if "-1021" in str(e) or "-1022" in str(e) or "recvWindow" in str(e):
            print(f"  [BAL-READ-FAIL] {asset}: {str(e)[:70]} — resyncing clock")
            if await _resync_clock(client):
                try:
                    bal = await _retry(client.get_asset_balance, asset=asset)
                    print(f"  [BAL-READ-OK] {asset}: recovered after clock resync")
                    return bal if isinstance(bal, dict) and "free" in bal else None
                except Exception as e2:
                    e = e2
        print(f"  [BAL-READ-FAIL] {asset}: {str(e)[:90]} — UNKNOWN (not zero)")
        return None
    if not isinstance(bal, dict) or "free" not in bal:
        print(f"  [BAL-READ-FAIL] {asset}: unexpected response {str(bal)[:60]} — UNKNOWN (not zero)")
        return None
    return bal


async def _spot_free_usdt(client):
    """Free spot USDT, or None if unreadable."""
    return await _asset_free(client, "USDT")


async def _asset_qty(client, asset: str):
    """Free + locked balance (locked because an OCO holds it), or None if unreadable."""
    bal = await _read_balance(client, asset)
    if bal is None:
        return None
    try:
        return float(bal.get("free") or 0.0) + float(bal.get("locked") or 0.0)
    except (TypeError, ValueError):
        return None


async def _asset_free(client, asset: str):
    """FREE balance only — what can actually be sold right now — or None if
    unreadable. A resting OCO locks its quantity, so free < total until that OCO
    is cancelled or fills."""
    bal = await _read_balance(client, asset)
    if bal is None:
        return None
    try:
        return float(bal.get("free") or 0.0)
    except (TypeError, ValueError):
        return None


# ── Breakout signal ──────────────────────────────────────────────────────────
async def _fetch_klines(client, symbol: str):
    try:
        lookback_days = max(2, int((BREAKOUT_LOOKBACK + 4) * _INTERVAL_HOURS[INTERVAL] / 24) + 1)
        return await client.get_historical_klines(
            symbol, INTERVAL, start_str=f"{lookback_days} days ago UTC")
    except Exception:
        return None


def _is_breakout(klines) -> bool:
    """LONG breakout on the last CLOSED bar: close > prior N-bar high AND volume
    >= VOLUME_MULT x average. (klines[-1] may be the still-forming bar → use -2.)"""
    n = BREAKOUT_LOOKBACK
    if not klines or len(klines) < n + 2:
        return False
    highs = [float(k[2]) for k in klines]
    closes = [float(k[4]) for k in klines]
    vols = [float(k[5]) for k in klines]
    i = len(klines) - 2  # last closed bar
    window_high = max(highs[i - n:i])
    avg_vol = sum(vols[i - n:i]) / n
    if avg_vol <= 0:
        return False
    return closes[i] > window_high and vols[i] >= VOLUME_MULT * avg_vol


async def _price(client, symbol: str) -> float:
    t = await _retry(client.get_symbol_ticker, symbol=symbol)
    return float(t["price"])


# ── Satellite budget ─────────────────────────────────────────────────────────
async def _satellite_cash(client, state) -> float:
    """Investable satellite USDT. G8: live is capped to the satellite target
    (never sizes off unrelated USDT parked in spot). Paper: simulated cash."""
    if _is_live():
        spot = await _spot_free_usdt(client)
        if spot is None:
            return None                      # UNKNOWN — caller must skip, not trade
        target = float(state.get("satellite_target", 0.0)) or spot
        return min(spot, target)
    return float(state.get("satellite_cash", 0.0))


# ── Core allocation: one-time setup + win-sweep (the WALL lives here) ─────────
async def _setup_allocation(client, state):
    """Establish the 80/20 split. In LIVE this redeems earn->spot ONCE. This is
    the ONLY place redeem is called — the loop never redeems, so satellite
    losses can never touch the core."""
    spot = await _spot_free_usdt(client)
    earn = await _earn_usdt(client)
    if spot is None or earn is None:
        # The one-time split (and its redeem) must never be sized off a failed
        # read; retry on the next start rather than move the wrong amount.
        print("[hybrid] setup deferred — balance unreadable this cycle")
        return
    total = spot + earn
    if "satellite_target" not in state:
        state["satellite_target"] = round(total * (1.0 - CORE_FRACTION), 2)
    sat_target = state["satellite_target"]

    if not _is_live():
        if "satellite_cash" not in state:
            state["satellite_cash"] = round(sat_target, 2)
        print(f"[hybrid] paper split — total=${total:.2f} core={total*CORE_FRACTION:.2f} "
              f"satellite=${state['satellite_cash']:.2f} (simulated, no real money moved)")
        return

    if state.get("setup_done"):
        print(f"[hybrid] live split already established — satellite_target=${sat_target:.2f}, spot=${spot:.2f}")
        return
    deficit = sat_target - spot
    if deficit > 0.01 and _EARN_PRODUCT_ID:
        try:
            await client.redeem_simple_earn_flexible_product(
                productId=_EARN_PRODUCT_ID, amount=str(round(deficit, 2)))
            print(f"[hybrid] live setup — redeemed ${deficit:.2f} earn->spot; core stays ~${total*CORE_FRACTION:.2f}")
        except Exception as e:
            print(f"[hybrid] live setup redeem failed: {str(e)[:90]} — satellite underfunded, continuing")
    state["setup_done"] = True


async def _win_sweep(client, state, sat_cash: float):
    """Ratchet: sweep satellite cash above target back into the safe core.
    Money only ever flows satellite -> core here — never the reverse."""
    target = float(state.get("satellite_target", 0.0))
    if target <= 0:
        return sat_cash
    excess = sat_cash - target - SWEEP_THRESHOLD_USD
    if excess < 0.5:
        return sat_cash
    if _is_live():
        if _EARN_PRODUCT_ID:
            try:
                await client.subscribe_simple_earn_flexible_product(
                    productId=_EARN_PRODUCT_ID, amount=str(round(excess, 2)))
                print(f"[hybrid] 🟢 WIN-SWEEP ${excess:.2f} satellite->core (banked to safe yield)")
            except Exception as e:
                print(f"[hybrid] win-sweep subscribe failed: {str(e)[:80]}")
                return sat_cash
    else:
        state["satellite_cash"] = round(sat_cash - excess, 2)
        print(f"[hybrid] 🟢 WIN-SWEEP ${excess:.2f} satellite->core (simulated)")
    return sat_cash - excess


# ── G1: exchange-side protective OCO (take-profit + stop-loss) ────────────────
async def _place_protective_oco(client, symbol, qty, tp_price, sl_price):
    """Rest an OCO SELL on Binance so the stop/target is enforced even if this
    process dies or the machine sleeps. Returns orderListId, or None on failure
    (the software poll loop then remains the only stop — logged loudly)."""
    step, _mn, tick = await _symbol_filters(client, symbol)
    q = _round_step(qty, step)
    tp = _round_price(tp_price, tick)
    stop_trig = _round_price(sl_price, tick)
    stop_limit = _round_price(sl_price * (1 - STOP_LIMIT_OFFSET_PCT / 100.0), tick)
    if q <= 0:
        return None
    def _fmt(p):
        return f"{p:.8f}".rstrip("0").rstrip(".")
    try:
        # Binance's current OCO API uses the above/below-order form (the old
        # price/stopPrice/stopLimitPrice form is rejected with -1102). For a SELL
        # OCO: ABOVE = take-profit (LIMIT_MAKER, price > last), BELOW = stop-loss
        # (STOP_LOSS_LIMIT, stopPrice < last).
        o = await _retry(
            client.create_oco_order, symbol=symbol, side="SELL", quantity=q,
            aboveType="LIMIT_MAKER", abovePrice=_fmt(tp),
            belowType="STOP_LOSS_LIMIT", belowStopPrice=_fmt(stop_trig),
            belowPrice=_fmt(stop_limit), belowTimeInForce="GTC")
        oid = o.get("orderListId")
        print(f"  [OCO] {symbol} protective stop+target resting on exchange (id={oid}) "
              f"tp={tp} stop={stop_trig}")
        return oid
    except Exception as e:
        print(f"  ⚠️  [OCO-FAIL] {symbol}: {str(e)[:120]} — SOFTWARE STOP ONLY until next open")
        return None


async def _cancel_oco(client, symbol, oco_id) -> bool:
    """Cancel a resting OCO. Returns True if the list is gone (cancelled now, or
    already filled/cancelled), False if it may STILL be resting — in which case
    the base asset is still locked and a market sell would fail.

    Must hit DELETE /api/v3/orderList (`v3_delete_order_list`), NOT
    DELETE /api/v3/order (`cancel_order`) — the latter cancels a single order by
    orderId and rejects `orderListId`, so it never cancelled the OCO at all.
    """
    if oco_id is None:
        return True
    canceller = getattr(client, "v3_delete_order_list", None)
    if canceller is None:  # very old python-binance — no spot order-list cancel
        print(f"  ⚠️  [OCO-CANCEL] {symbol}: client lacks v3_delete_order_list — cannot cancel list {oco_id}")
        return False
    try:
        await _retry(canceller, symbol=symbol, orderListId=oco_id)
        return True
    except Exception as e:
        # Already filled/cancelled → the list is gone, which is what we wanted.
        if "-2011" in str(e) or "Unknown order" in str(e) or "-2013" in str(e):
            return True
        print(f"  ⚠️  [OCO-CANCEL] {symbol} id={oco_id}: {str(e)[:80]} — qty may still be LOCKED")
        return False


# ── Satellite trade execution ────────────────────────────────────────────────
async def _open_position(client, state, symbol: str, sat_cash: float):
    step, min_notional, _tick = await _symbol_filters(client, symbol)
    price = await _price(client, symbol)
    notional = min(sat_cash * TRADE_FRACTION, sat_cash - 0.05)
    if notional < max(min_notional, SATELLITE_FLOOR_USD):
        return False
    qty = _round_step(notional / price, step)
    if qty <= 0 or qty * price < max(min_notional, 5.0):
        return False

    if MODE == "dryrun":
        print(f"  [DRYRUN] would BUY {qty} {symbol} @ ~{price:.6f} (~${qty*price:.2f}) "
              f"tp=+{TP_PCT}% sl=-{SL_PCT}% + resting OCO — sending nothing")
        return False

    fill_price = price
    oco_id = None
    if _is_live():
        try:
            o = await _retry(client.order_market_buy, symbol=symbol, quantity=qty)
            fills = o.get("fills", [])
            if fills:
                spent = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                got = sum(float(f["qty"]) for f in fills)
                fill_price = spent / got if got else price
                qty = got
        except Exception as e:
            # G5: a buy can fill AND raise (timeout after match). Reconcile the
            # real balance before assuming no fill, so we never leave an
            # untracked, unstopped position.
            print(f"  [SAT-OPEN-ERROR] {symbol}: {str(e)[:90]} — checking for a stray fill")
            held = await _asset_qty(client, symbol[:-4])
            if held is None:
                print(f"  [SAT-OPEN-ERROR] {symbol}: balance unreadable — cannot confirm fill; "
                      f"NOT tracking. Reconcile will adopt it on the next start if it filled.")
                return False
            if held * price >= max(min_notional, 5.0):
                qty = _round_step(held, step)
                print(f"  [SAT-OPEN-RECOVER] {symbol}: buy DID fill ({held}) — adopting + protecting")
            else:
                return False
        # G1: rest the protective OCO on the exchange (uses actual held qty).
        held = await _asset_qty(client, symbol[:-4])
        if held is None:
            held = qty          # unreadable → protect the qty we believe we bought
        tp_price = fill_price * (1 + TP_PCT / 100.0)
        sl_price = fill_price * (1 - SL_PCT / 100.0)
        oco_id = await _place_protective_oco(client, symbol, min(qty, held), tp_price, sl_price)
    else:
        state["satellite_cash"] = round(sat_cash - qty * fill_price * (1 + FEE_RT_PCT / 200.0), 2)

    state["position"] = {
        "symbol": symbol, "entry_ts": time.time(), "entry_price": fill_price,
        "qty": qty, "tp_price": fill_price * (1 + TP_PCT / 100.0),
        "sl_price": fill_price * (1 - SL_PCT / 100.0), "mode": MODE, "oco_id": oco_id,
    }
    state["last_entry_ts"] = time.time()
    print(f"  [SAT-OPEN] {symbol} {qty} @ {fill_price:.6f} (~${qty*fill_price:.2f}) "
          f"tp={state['position']['tp_price']:.6f} sl={state['position']['sl_price']:.6f} [{MODE}]")
    return True


def _book_close(state, symbol, entry, exit_price, reason, qty):
    """Write the ledger record and clear the position (shared by all close paths)."""
    gross_pct = (exit_price - entry) / entry * 100.0
    net_pct = gross_pct - FEE_RT_PCT
    pos = state.get("position") or {}
    held_h = (time.time() - pos.get("entry_ts", time.time())) / 3600.0
    _log_trade({
        "ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
        "held_h": round(held_h, 2), "entry_price": entry, "exit_price": round(exit_price, 8),
        "qty": qty, "gross_pct": round(gross_pct, 4), "net_pct": round(net_pct, 4),
        "reason": reason, "mode": pos.get("mode", MODE),
    })
    print(f"  [SAT-CLOSE] {symbol} @ {exit_price:.6f} net={net_pct:+.2f}% ({reason}) [{pos.get('mode', MODE)}]")
    state["position"] = None


async def _actual_exit_vwap(client, symbol: str, entry_ts: float):
    """Volume-weighted price of the SELL fills that closed this position, read
    from real trade history. Returns None if history is unavailable, so callers
    can fall back rather than book a fabricated price."""
    try:
        trades = await _retry(client.get_my_trades, symbol=symbol, limit=50)
        since_ms = int(entry_ts * 1000)
        sells = [t for t in (trades or [])
                 if not t.get("isBuyer") and int(t.get("time", 0) or 0) >= since_ms]
        total_qty = sum(float(t["qty"]) for t in sells)
        if total_qty <= 0:
            return None
        return sum(float(t["price"]) * float(t["qty"]) for t in sells) / total_qty
    except Exception as e:
        # Any shape surprise from the API must fall back, never crash the poll.
        print(f"  [EXIT-FILL] {symbol}: trade history unusable ({str(e)[:60]})")
        return None


async def _detect_exchange_close(client, state) -> bool:
    """G1: if the resting OCO already executed (stop or target hit) — possibly
    while this process was asleep or between polls — the coin balance is gone.
    Detect it and book the trade at the OCO price, so the exchange stop and our
    records stay consistent. Returns True if it closed the position."""
    pos = state.get("position")
    if not pos or pos.get("mode") != "live":
        return False
    symbol = pos["symbol"]
    held = await _asset_qty(client, symbol[:-4])
    if held is None:
        # Balance UNKNOWN. Never infer "the OCO fired" from a failed read — that
        # would book a fabricated close and abandon a live position.
        print(f"  [OCO-CHECK] {symbol}: balance unreadable — assuming still held, retrying next poll")
        return False
    # Position still meaningfully held → OCO hasn't fired.
    if held >= pos["qty"] * 0.5:
        return False
    # Coin is gone → the OCO closed it. Book the REAL fill, not the intended
    # trigger price: a STOP_LOSS_LIMIT fills near sl*(1-STOP_LIMIT_OFFSET_PCT),
    # not at sl, and labelling by the *current* price (up to POLL_MIN stale)
    # could book a rebounded stop-out as a take-profit.
    actual = await _actual_exit_vwap(client, symbol, pos["entry_ts"])
    if actual is not None:
        exit_price = actual
        reason = "take-profit-oco" if actual >= pos["entry_price"] else "stop-loss-oco"
    else:
        # Trade history unavailable — fall back to the intended prices, but mark
        # the record as an estimate so the ledger never claims false precision.
        price = await _price(client, symbol)
        if price >= pos["tp_price"] * 0.999:
            exit_price, reason = pos["tp_price"], "take-profit-oco-est"
        else:
            exit_price, reason = pos["sl_price"], "stop-loss-oco-est"
    print(f"  [OCO-FILLED] {symbol} closed on exchange (held={held}) → booking {reason} @ {exit_price:.6f}")
    _book_close(state, symbol, pos["entry_price"], exit_price, reason, pos["qty"])
    return True


async def _close_position(client, state, reason: str):
    """Software-initiated close (time-stop / kill-switch / paper). Cancels the
    resting OCO first so the market sell isn't blocked by the locked balance."""
    pos = state["position"]
    symbol, qty, entry = pos["symbol"], pos["qty"], pos["entry_price"]
    exit_price = await _price(client, symbol)
    if _is_live():
        try:
            cancelled = await _cancel_oco(client, symbol, pos.get("oco_id"))
            step, _mn, _tick = await _symbol_filters(client, symbol)
            asset = symbol[:-4]
            total = await _asset_qty(client, asset)      # free + locked
            free = await _asset_free(client, asset)      # what can actually be sold
            if total is None or free is None:
                print(f"  [SAT-CLOSE-BLOCKED] {symbol}: balance unreadable — keeping tracked, retrying next poll")
                return False
            if total <= 0:
                # Coin vanished between the pre-check and here — the OCO fired in
                # that window. BOOK it at the real fill; silently dropping the
                # position would lose the trade from the ledger entirely.
                actual = await _actual_exit_vwap(client, symbol, pos["entry_ts"])
                if actual is not None:
                    exit_price = actual
                    reason = "take-profit-oco" if actual >= entry else "stop-loss-oco"
                print(f"  [SAT-CLOSE] {symbol} nothing held — OCO closed it mid-close; "
                      f"booking {reason} @ {exit_price:.6f}")
                _book_close(state, symbol, entry, exit_price, reason, qty)
                return True
            if free <= 0:
                # Held but fully LOCKED — an OCO is still resting. Dropping the
                # position here would orphan a real holding, so keep tracking it.
                print(f"  [SAT-CLOSE-BLOCKED] {symbol} {total} held but LOCKED by OCO "
                      f"{pos.get('oco_id')} (cancel_ok={cancelled}) — keeping tracked, retrying next poll")
                return False
            sell_qty = _round_step(min(qty, free), step)
            if sell_qty <= 0:
                print(f"  [SAT-CLOSE] {symbol} free {free} below lot step {step} — keeping tracked")
                return False
            o = await _retry(client.order_market_sell, symbol=symbol, quantity=sell_qty)
            fills = o.get("fills", [])
            if fills:
                got = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                sold = sum(float(f["qty"]) for f in fills)
                exit_price = got / sold if sold else exit_price
        except Exception as e:
            print(f"  [SAT-CLOSE-ERROR] {symbol}: {str(e)[:100]} — keeping tracked, will retry")
            return False
    else:
        state["satellite_cash"] = round(
            float(state.get("satellite_cash", 0.0)) + qty * exit_price * (1 - FEE_RT_PCT / 200.0), 2)
    _book_close(state, symbol, entry, exit_price, reason, qty)
    return True


# ── G2: restart / crash reconciliation ───────────────────────────────────────
async def _reconcile(client, state):
    """On live startup, make the tracked state agree with the exchange:
      • tracked position but the coin is GONE → it closed while we were down; book it.
      • tracked position still held but NO resting OCO → protect it now.
      • state is FLAT but a stray coin balance ≥ min-notional exists → adopt +
        protect it (a buy whose state-save crashed) instead of leaving it
        unmanaged and unstopped.
    """
    if not _is_live():
        return
    pos = state.get("position")
    if pos and pos.get("mode") == "live":
        symbol = pos["symbol"]
        held = await _asset_qty(client, symbol[:-4])
        if held is None:
            print(f"[hybrid] reconcile — {symbol} balance unreadable; leaving tracked state untouched")
            return
        if held < pos["qty"] * 0.5:
            print(f"[hybrid] reconcile — tracked {symbol} no longer held; booking its exchange close")
            await _detect_exchange_close(client, state)
        elif not pos.get("oco_id"):
            print(f"[hybrid] reconcile — tracked {symbol} has NO resting stop; placing one now")
            pos["oco_id"] = await _place_protective_oco(
                client, symbol, min(pos["qty"], held), pos["tp_price"], pos["sl_price"])
        return
    # Flat: look for a stray holding to adopt.
    for symbol in WATCHLIST:
        try:
            held = await _asset_qty(client, symbol[:-4])
            price = await _price(client, symbol)
            if held is None:
                continue
            if held * price >= max(SATELLITE_FLOOR_USD, 5.0):
                print(f"[hybrid] reconcile — stray {symbol} holding ${held*price:.2f} found; adopting + protecting")
                tp, sl = price * (1 + TP_PCT / 100.0), price * (1 - SL_PCT / 100.0)
                oco = await _place_protective_oco(client, symbol, held, tp, sl)
                state["position"] = {
                    "symbol": symbol, "entry_ts": time.time(), "entry_price": price,
                    "qty": held, "tp_price": tp, "sl_price": sl, "mode": "live", "oco_id": oco,
                }
                return
        except Exception:
            continue


def _report():
    trades = []
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    if not trades:
        print("No closed satellite trades yet.")
        return
    nets = [t["net_pct"] for t in trades]
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    print("=" * 66)
    print(f"HYBRID SATELLITE — TRACK RECORD ({n} closed trades)")
    print("=" * 66)
    print(f"  Avg net/trade: {sum(nets)/n:+.3f}%   win-rate: {wins/n*100:.0f}%   cum: {sum(nets):+.2f}%")
    print("  NOTE: satellite is negative-EV by construction — capped 'action',")
    print("        not a growth engine. The core (80% in yield) is where growth lives.")
    print("=" * 66)


async def run():
    global _EARN_PRODUCT_ID
    from binance import AsyncClient

    if not _acquire_lock():                                   # G3
        print(f"[hybrid] another instance holds {PIDFILE} — refusing to start (single-instance lock)")
        return

    client = await _create_client_with_retry(AsyncClient)
    state = _load(STATE, {"position": None})
    armed = _is_live()
    banner = {"paper": "📝 PAPER (no real money)",
              "dryrun": "🧪 DRY-RUN (logs orders, sends none)",
              "live": ("🔴 LIVE-ARMED (real spot orders + exchange stops)" if armed
                       else "🔴 live (BLOCKED — arm file missing, no orders)")}
    print(f"[hybrid] start — MODE={MODE} → {banner.get(MODE, MODE)}")
    print(f"[hybrid] barbell — core={CORE_FRACTION*100:.0f}% (yield, untouchable) "
          f"satellite={(1-CORE_FRACTION)*100:.0f}% (auto-speculative, NEGATIVE-EV, capped)")
    print(f"[hybrid] satellite — breakout({BREAKOUT_LOOKBACK}/{INTERVAL}) tp=+{TP_PCT}% sl=-{SL_PCT}% "
          f"floor=${SATELLITE_FLOOR_USD} sweep>+${SWEEP_THRESHOLD_USD} kill={KILL_FILE}")
    try:
        _EARN_PRODUCT_ID = await _resolve_product_id(client)
        await _setup_allocation(client, state)
        await _reconcile(client, state)                       # G2
        _save(STATE, state)

        while True:
            now = time.time()
            # Keep the signing clock aligned BEFORE any signed call this cycle.
            # Cheap (one unsigned request) and prevents drift from silently
            # disabling every signed request until a restart.
            await _resync_clock(client)

            # ── Manage an open position ──
            if state.get("position"):
                try:
                    # G1: did the exchange OCO already close it (asleep/between polls)?
                    if not await _detect_exchange_close(client, state):
                        pos = state["position"]
                        price = await _price(client, pos["symbol"])
                        held_h = (now - pos["entry_ts"]) / 3600.0
                        reason = None
                        # Software TP/SL remain as a BACKUP to the exchange OCO
                        # (and are the only stop in paper mode).
                        if price >= pos["tp_price"]:
                            reason = "take-profit"
                        elif price <= pos["sl_price"]:
                            reason = "stop-loss"
                        elif held_h >= MAX_HOLD_H:
                            reason = "time-stop"
                        elif _killed():
                            reason = "kill-switch"
                        if reason:
                            await _close_position(client, state, reason)
                    _save(STATE, state)
                except Exception as e:
                    print(f"  [SAT-MANAGE-ERROR] {str(e)[:90]}")

            # ── Drawdown auto-halt (G7: live trades only) ──
            dd = _current_drawdown_pct(live_only=True)
            if dd >= MAX_DD_PCT and not _killed():
                open(KILL_FILE, "w").close()
                print(f"[hybrid] 🛑 DRAWDOWN HALT {dd:.1f}% >= {MAX_DD_PCT}% — kill-switch engaged")

            sat_cash = await _satellite_cash(client, state)
            if sat_cash is None:
                # Balance unreadable — do NOT report $0.00 and do NOT trade off a
                # guess. Skip the cycle; the next poll re-reads.
                print(f"[hybrid {datetime.now(timezone.utc).strftime('%H:%M')}] mode={MODE} "
                      f"sat_cash=UNKNOWN (balance read failed) — skipping this cycle")
                await asyncio.sleep(POLL_MIN * 60)
                continue

            # ── Win-sweep (only when flat) ──
            if not state.get("position"):
                sat_cash = await _win_sweep(client, state, sat_cash)
                _save(STATE, state)

            # ── Entry ──
            opened = False
            floor = max(SATELLITE_FLOOR_USD, 5.0)
            if (not state.get("position") and not _killed()
                    and sat_cash >= floor / TRADE_FRACTION   # G-low: no dead-zone busy-loop
                    and (now - float(state.get("last_entry_ts", 0))) >= COOLDOWN_H * 3600):
                if MODE == "live" and not _is_live():
                    print("  [LIVE-BLOCKED] arm file missing — no order")
                else:
                    for symbol in WATCHLIST:
                        try:
                            klines = await _fetch_klines(client, symbol)
                            if klines and _is_breakout(klines):
                                if await _open_position(client, state, symbol, sat_cash):
                                    opened = True
                                    _save(STATE, state)
                                    break
                        except Exception as e:
                            print(f"  [SCAN-ERROR] {symbol}: {str(e)[:70]}")
            elif sat_cash < floor and not state.get("position"):
                print(f"  [SATELLITE-FLOOR] cash ${sat_cash:.2f} < ${floor:.2f} — sleeve idle "
                      f"(never topped up from core)")

            ts = datetime.now(timezone.utc).strftime("%H:%M")
            held = "flat" if not state.get("position") else state["position"]["symbol"]
            print(f"[hybrid {ts}] mode={MODE} sat_cash=${sat_cash:.2f} pos={held} opened={int(opened)}")
            _save(STATE, state)
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


# ═════════════════════════════════════════════════════════════════════════════
# OBJECTIVE "allocate" — deploy, compound, and account honestly.
#
# No speculation. Money flows spot -> earn ONLY; `redeem_simple_earn_flexible_
# product` is never called here at all, so the wall is absolute rather than
# merely enforced. Positive-EV by construction: it captures yield on cash that
# would otherwise sit at 0%, and scales linearly with capital.
# ═════════════════════════════════════════════════════════════════════════════

async def _nav_snapshot(client) -> dict:
    """Total account value, broken down. Returns None on an unreadable balance
    so a failed read is never recorded as a NAV crash in the equity curve."""
    spot = await _asset_free(client, "USDT")
    if spot is None:
        return None
    earn_positions = await _earn_positions(client)
    if earn_positions is None:
        return None          # earn is the bulk of NAV; never record it as 0
    earn = earn_positions.get("USDT", 0.0)
    holdings, hold_usd = {}, 0.0
    for asset in NAV_ASSETS:
        spot_qty = await _asset_qty(client, asset)
        earn_qty = earn_positions.get(asset, 0.0)
        # An asset subscribed to earn LEAVES the spot balance but is still ours.
        # Counting spot alone would book a NAV collapse the moment the sweep
        # succeeded — the machine punishing itself for doing its job.
        if spot_qty is None and not earn_qty:
            continue                       # unreadable and not in earn: skip
        qty = (spot_qty or 0.0) + earn_qty
        if not qty:
            continue
        try:
            px = await _price(client, f"{asset}USDT")
        except Exception:
            continue
        holdings[asset] = {"qty": qty, "price": px, "usd": round(qty * px, 4),
                           "in_earn": round(earn_qty, 8)}
        hold_usd += qty * px
    return {"spot_usdt": round(spot, 4), "earn_usdt": round(earn, 4),
            "holdings_usd": round(hold_usd, 4), "holdings": holdings,
            "nav": round(spot + earn + hold_usd, 4)}


async def _detect_contributions(client, state) -> float:
    """New EXTERNAL money since last check, in USD.

    Uses deposit history, NOT a NAV jump — internal moves between your own
    wallets look identical to a deposit in NAV terms and would be counted as
    contributions. (This account round-tripped $20 spot->futures->spot in
    July/August; inferring from NAV would have booked $20 of phantom deposits.)
    Returns the new total and records the ids so a deposit is counted once.
    """
    seen = set(state.setdefault("seen_deposits", []))
    total = 0.0
    try:
        deposits = await _retry(client.get_deposit_history)
    except Exception as e:
        print(f"  [CONTRIB] deposit history unavailable ({str(e)[:60]})")
        return 0.0
    for d in (deposits or []):
        if int(d.get("status", -1)) != 1:            # 1 = completed
            continue
        key = f"{d.get('txId')}:{d.get('coin')}:{d.get('amount')}"
        if key in seen:
            continue
        amount, coin = float(d.get("amount", 0) or 0), (d.get("coin") or "").upper()
        usd = amount
        if coin not in ("USDT", "USDC", "BUSD", "FDUSD"):
            try:
                usd = amount * await _price(client, f"{coin}USDT")
            except Exception:
                print(f"  [CONTRIB] can't price {coin}; recording qty only")
                usd = 0.0
        seen.add(key)
        total += usd
        print(f"  [CONTRIB] +{amount} {coin} (~${usd:.2f}) detected")
    state["seen_deposits"] = sorted(seen)
    if total:
        state["cumulative_contributions"] = round(
            float(state.get("cumulative_contributions", 0.0)) + total, 4)
    return total


async def _sweep_idle_to_earn(client, state) -> float:
    """Move idle spot USDT into flexible earn. The ONLY money movement this
    objective makes, and it is one-directional by design.

    Runs only when flat: an open satellite position needs its cash buffer, and
    sweeping it would strand the position. Leaves IDLE_BUFFER_USD behind for
    fee/min-notional friction.
    """
    if state.get("position"):
        return 0.0
    spot = await _asset_free(client, "USDT")
    if spot is None:
        return 0.0
    amount = spot - IDLE_BUFFER_USD
    if amount < MIN_SWEEP_USD or not _EARN_PRODUCT_ID:
        return 0.0
    if not _is_live():
        print(f"  [SWEEP-{MODE.upper()}] would move ${amount:.2f} idle spot -> earn (nothing sent)")
        return 0.0
    try:
        await _retry(client.subscribe_simple_earn_flexible_product,
                     productId=_EARN_PRODUCT_ID, amount=str(round(amount, 2)))
        print(f"  [SWEEP] 🟢 ${amount:.2f} idle spot -> earn (now compounding)")
        return amount
    except Exception as e:
        print(f"  [SWEEP-FAIL] ${amount:.2f}: {str(e)[:80]}")
        return 0.0


async def _sweep_assets_to_earn(client, state) -> dict:
    """Subscribe idle NON-USDT holdings (BTC by default) to flexible earn.

    The USDT sweep has run for months while the BTC hold sat at 0% — the single
    largest avoidable gap in the account. Flexible earn redeems on demand, so
    this adds yield without a lock-up and without a directional decision: it
    changes the rate on an asset you already hold, nothing else.

    Skips PROTECTED_ASSETS only if a satellite position is open; a resting OCO
    holds its quantity as `locked`, and `_asset_free` already excludes that, so
    a protected position can never be swept out from under its own stop.
    """
    swept: dict[str, float] = {}
    if state.get("position"):
        return swept                       # satellite open: leave balances alone
    for asset in EARN_ASSETS:
        if asset == "USDT":
            continue                       # handled by _sweep_idle_to_earn
        free = await _asset_free(client, asset)
        if not free:
            continue
        products = await _earn_products(client, asset, free)
        if not products:
            continue
        best = products[0]
        if best["min"] and free < best["min"]:
            # Logged, not silent: "nothing happened" and "nothing COULD happen"
            # look identical in a quiet log, and the operator needs to know the
            # holding is stranded below the minimum rather than assume it earns.
            print(f"  [SWEEP-SKIP] {asset}: hold {free:.8f} < product minimum "
                  f"{best['min']:.8f} — stays at 0% (rate offered: "
                  f"{best['apr']*100:.2f}% APR)")
            continue
        if best["apr"] <= 0.0005:
            print(f"  [SWEEP-SKIP] {asset}: best rate {best['apr']*100:.2f}% APR "
                  f"is not worth the subscription — leaving it liquid")
            continue
        if not _is_live():
            print(f"  [SWEEP-{MODE.upper()}] would subscribe {free:.8f} {asset} "
                  f"-> earn @ {best['apr']*100:.2f}% APR (nothing sent)")
            continue
        try:
            await _retry(client.subscribe_simple_earn_flexible_product,
                         productId=best["productId"], amount=str(free))
            print(f"  [SWEEP] 🟢 {free:.8f} {asset} -> earn @ {best['apr']*100:.2f}% APR")
            swept[asset] = free
        except Exception as e:
            print(f"  [SWEEP-FAIL] {asset} {free:.8f}: {str(e)[:80]}")
    return swept


def _record_nav(state, snap: dict, contributed: float, swept: float):
    """Append one row to the equity curve, separating CONTRIBUTIONS from RETURNS.

    Without this split a deposit looks exactly like profit: balance goes up and
    the system appears to be working. `growth` below is the only number that
    reflects what the account actually earned.
    """
    contrib = float(state.get("cumulative_contributions", 0.0))
    baseline = state.get("nav_baseline")
    if baseline is None:                     # first ever snapshot anchors the curve
        baseline = snap["nav"] - contrib
        state["nav_baseline"] = round(baseline, 4)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "nav": snap["nav"], "spot_usdt": snap["spot_usdt"], "earn_usdt": snap["earn_usdt"],
        "holdings_usd": snap["holdings_usd"], "holdings": snap["holdings"],
        "cumulative_contributions": round(contrib, 4),
        "contributed_this_cycle": round(contributed, 4),
        "swept_to_earn": round(swept, 4),
        # growth = what the account earned, with deposits removed
        "growth": round(snap["nav"] - contrib - float(state["nav_baseline"]), 4),
    }
    os.makedirs(os.path.dirname(NAV_FILE) or ".", exist_ok=True)
    with open(NAV_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _nav_report():
    """Print the equity curve with contributions and returns kept apart."""
    if not os.path.exists(NAV_FILE):
        print("No NAV history yet.")
        return
    rows = []
    with open(NAV_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print("No NAV history yet.")
        return
    first, last = rows[0], rows[-1]
    print("=" * 68)
    print("NAV HISTORY — contributions and returns kept separate")
    print("=" * 68)
    print(f"  snapshots        : {len(rows)}  ({first['ts'][:16]} -> {last['ts'][:16]})")
    print(f"  NAV now          : ${last['nav']:.2f}   (was ${first['nav']:.2f})")
    print(f"  contributions    : ${last['cumulative_contributions']:.2f}  (deposits — NOT profit)")
    print(f"  GROWTH (earned)  : ${last['growth']:+.2f}   <- the only honest number")
    print(f"  breakdown        : earn ${last['earn_usdt']:.2f} | spot ${last['spot_usdt']:.2f} "
          f"| holdings ${last['holdings_usd']:.2f}")
    swept = sum(r.get("swept_to_earn", 0.0) for r in rows)
    print(f"  swept to earn    : ${swept:.2f} total (cash rescued from 0% idle)")
    print("=" * 68)


async def run_allocate():
    """The 'allocate' objective loop: sweep, detect contributions, record NAV."""
    global _EARN_PRODUCT_ID
    from binance import AsyncClient

    if not _acquire_lock():
        print(f"[hybrid] another instance holds {PIDFILE} — refusing to start")
        return
    client = await _create_client_with_retry(AsyncClient)
    state = _load(STATE, {"position": None})
    print(f"[alloc] start — OBJECTIVE=allocate MODE={MODE} "
          f"{'🟢 LIVE (real earn subscriptions)' if _is_live() else '📝 simulated (no money moves)'}")
    print("[alloc] job — deploy idle cash into yield, detect contributions, record honest NAV")
    print("[alloc] NO speculative trades. Money flows spot -> earn ONLY; redeem is never called.")
    try:
        # Resolve against the real USDT balance so tiered rates are compared on
        # what we actually hold, not on a headline for a hypothetical size.
        _held = await _earn_usdt(client)
        _EARN_PRODUCT_ID = await _resolve_product_id(client, "USDT", _held or 0.0)
        while True:
            if _killed():
                print("[alloc] kill-switch present — idling (no money moves)")
                await asyncio.sleep(POLL_MIN * 60)
                continue
            try:
                # BOUNDED CYCLE. Nothing in here had a timeout, so a single
                # stuck HTTP call hung the whole daemon until the supervisor's
                # 25-minute stall watchdog killed it — five times between
                # 2026-08-25 and 2026-09-04. aiohttp's own default only bounds
                # ONE request (300s), and a cycle makes a dozen of them behind a
                # 3x retry wrapper, so the ceiling was effectively unbounded.
                # Abandoning a slow cycle is always safe here: every value is
                # re-read from the exchange next cycle, nothing is carried over,
                # and the sweep re-derives free balance from scratch. This makes
                # the watchdog a true last resort instead of the primary
                # recovery path — the daemon now heals itself in-process.
                await asyncio.wait_for(_allocate_cycle(client, state),
                                       timeout=CYCLE_TIMEOUT_S)
            except asyncio.TimeoutError:
                print(f"  [ALLOC-TIMEOUT] cycle exceeded {CYCLE_TIMEOUT_S:.0f}s — "
                      f"abandoned; retrying next cycle (no state written)")
            except Exception as e:
                print(f"  [ALLOC-ERROR] {str(e)[:100]}")
            await asyncio.sleep(POLL_MIN * 60)
    finally:
        await client.close_connection()


async def _allocate_cycle(client, state):
    """One allocate cycle. Extracted so the caller can bound it with wait_for.

    Every step re-reads from the exchange, so abandoning this mid-flight loses
    nothing: the next cycle starts from the exchange's truth, not from memory.
    """
    await _resync_clock(client)
    contributed = await _detect_contributions(client, state)
    # Measure BEFORE sweeping. A spot->earn subscription debits spot immediately
    # but the earn position updates with a lag, so a snapshot taken straight
    # after a sweep undercounts NAV by the in-flight amount. Anchoring the
    # baseline on that would have manufactured ~$9.60 of phantom "growth" on the
    # next cycle. Moving money between your own wallets never changes NAV, so
    # measuring first is both correct and always consistent.
    snap = await _nav_snapshot(client)
    swept = await _sweep_idle_to_earn(client, state)
    # Non-USDT holdings (BTC) into their own flexible products. Same
    # measure-before-sweep ordering as above, and for the same reason: a
    # subscription debits the balance immediately while the earn position lags,
    # so snapshotting after would undercount.
    swept_assets = await _sweep_assets_to_earn(client, state)
    if swept_assets:
        print("  [SWEEP] non-USDT subscribed: "
              + ", ".join(f"{q:.8f} {a}" for a, q in swept_assets.items()))
    if snap is None:
        print(f"[alloc {datetime.now(timezone.utc):%H:%M}] NAV unreadable "
              f"— skipping this cycle (not recording a false zero)")
    else:
        row = _record_nav(state, snap, contributed, swept)
        print(f"[alloc {datetime.now(timezone.utc):%H:%M}] nav=${snap['nav']:.2f} "
              f"earn=${snap['earn_usdt']:.2f} spot=${snap['spot_usdt']:.2f} "
              f"hold=${snap['holdings_usd']:.2f} contrib=${row['cumulative_contributions']:.2f} "
              f"growth=${row['growth']:+.2f}")
    _save(STATE, state)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "report":
        _report()
    elif arg == "nav":
        _nav_report()
    elif OBJECTIVE == "allocate":
        asyncio.run(run_allocate())
    else:
        asyncio.run(run())
