#!/usr/bin/env python3
"""
IBKR thematic allocator — deposit router, drift monitor, and honest NAV ledger.

WHAT THIS IS (and honestly is not)
----------------------------------
A READ-ONLY accounting and planning daemon for an Interactive Brokers ETF
portfolio. It answers three questions every month:

  1. What is the book actually worth, and how much of that did I EARN rather
     than DEPOSIT?              -> `report`
  2. Where should this month's deposit go to pull the book back to target?
     -> `plan`
  3. Which companies do I really own, once fund overlap is unwound?
     -> `exposure`

IT PLACES NO ORDERS. It holds no credential that can place one. The Flex Web
Service token it uses is read-only by construction, so the worst case for a
leaked token is disclosure, never loss of funds. Execution is done by IBKR's own
Recurring Investment feature, edited by hand when `plan` says drift has moved
the split. Automating orders is a later, separately-gated stage and is not worth
its risk below roughly $20k (see docs/ibkr_allocator.md).

WHY THERE IS NO STRATEGY IN HERE
--------------------------------
This project falsified every trading edge it tested, on real money, including
the market-maker whose paper run ended 2026-09-04 at -0.59% per fill. The ETF
book is not an attempt to find another one. Its return is market return; the
only levers are the size of the deposit, staying invested, and not paying to
rebalance. So this file automates the deposit and the accounting, and contains
no signal, no timing rule, and no discretion.

THE RULES IT ENFORCES
---------------------
  1. NEVER SELL. Deposits alone rebalance the book, so drift is corrected for
     the price of a spread you were paying anyway.
  2. CONTRIBUTIONS ARE NOT RETURNS. Tracked separately in the shared ledger.
     A $600 deposit must never read as $600 of profit.
  3. CASH IS CAPPED IN DOLLARS, not as a percentage. A percentage reserve grows
     with the book forever and drags ~0.5%/yr on return.
  4. TARGETS CHANGE ON A SCHEDULE, not on news. `report` prints the age of the
     config and nags after `review_months`.

Usage:
  python3 ibkr_allocator.py report          # NAV, contributions, growth, drift
  python3 ibkr_allocator.py plan [AMOUNT]   # buy list for the next deposit
  python3 ibkr_allocator.py drift           # weights vs target, band breaches
  python3 ibkr_allocator.py exposure        # look-through company overlap
  python3 ibkr_allocator.py fetch           # refresh the cached Flex statement
  python3 ibkr_allocator.py                 # daemon: refresh + record + alert

Environment (.env):
  IBKR_FLEX_TOKEN      read-only Flex Web Service token
  IBKR_FLEX_QUERY_ID   the Activity Flex Query id
  IBKR_FLEX_FILE       optional: parse this local XML instead of calling the
                       web service (offline testing; skips the network entirely)
  IBKR_TARGETS         config path        (default config/ibkr_targets.json)
  IBKR_POLL_H          daemon cycle hours (default 6)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import capital_ledger as ledger

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:      # stdlib-only fallback so the CLI still runs bare
    pass

# ── config ───────────────────────────────────────────────────────────────────
TARGETS_FILE = os.getenv("IBKR_TARGETS", "config/ibkr_targets.json")
HOLDINGS_FILE = os.getenv("IBKR_HOLDINGS", "config/ibkr_fund_holdings.json")
STATE_FILE = os.getenv("IBKR_STATE", "logs/ibkr_state.json")
NAV_FILE = os.getenv("IBKR_NAV_FILE", "logs/ibkr_nav_history.jsonl")
CACHE_FILE = os.getenv("IBKR_CACHE", "logs/ibkr_flex_cache.xml")
ALERT_FILE = os.getenv("IBKR_ALERT_FILE", "logs/ibkr_alerts.log")
PIDFILE = os.getenv("IBKR_PIDFILE", "logs/ibkr.pid")
KILL_FILE = os.getenv("IBKR_KILL_FILE", "logs/ibkr.stop")
POLL_H = float(os.getenv("IBKR_POLL_H", "6"))

FLEX_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
FLEX_RETRIES = int(os.getenv("IBKR_FLEX_RETRIES", "4"))
FLEX_RETRY_DELAY_S = float(os.getenv("IBKR_FLEX_RETRY_DELAY_S", "10"))

_lock_handle = None


def _log(msg: str) -> None:
    print(f"[ibkr {datetime.now(timezone.utc):%H:%M}] {msg}", flush=True)


def _alert(msg: str) -> None:
    """Record something the operator must actually act on. Kept deliberately
    rare: an alert stream nobody reads is the same as no alerts at all."""
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    os.makedirs(os.path.dirname(ALERT_FILE) or ".", exist_ok=True)
    with open(ALERT_FILE, "a") as f:
        f.write(line + "\n")
    print(f"  ⚠️  ALERT: {msg}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# TARGETS
# ═════════════════════════════════════════════════════════════════════════════

def load_targets(path: str = None) -> dict:
    """Load and VALIDATE the target config.

    Validation is not ceremony here: a weights table that does not sum to 1.0
    silently under- or over-deploys every future deposit, and the error is
    invisible in the buy list because the numbers still look plausible.
    """
    path = path or TARGETS_FILE
    with open(path) as f:
        cfg = json.load(f)
    funds: dict[str, float] = {}
    for theme, spec in cfg.get("themes", {}).items():
        for sym, w in spec.get("funds", {}).items():
            if sym in funds:
                raise ValueError(f"{path}: fund {sym} appears in more than one theme")
            if not 0 < float(w) <= 1:
                raise ValueError(f"{path}: weight for {sym} is {w}, must be in (0, 1]")
            funds[sym.upper()] = float(w)
    if not funds:
        raise ValueError(f"{path}: no funds configured")
    total = sum(funds.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{path}: weights sum to {total:.6f}, must sum to 1.0")
    cfg["_funds"] = funds
    return cfg


def theme_weights(cfg: dict) -> dict[str, float]:
    return {t: sum(s.get("funds", {}).values()) for t, s in cfg.get("themes", {}).items()}


def fund_theme(cfg: dict, sym: str) -> str:
    for theme, spec in cfg.get("themes", {}).items():
        if sym.upper() in {k.upper() for k in spec.get("funds", {})}:
            return theme
    return "unassigned"


# ═════════════════════════════════════════════════════════════════════════════
# IBKR FLEX WEB SERVICE  (read-only; cannot place an order)
# ═════════════════════════════════════════════════════════════════════════════

def _flex_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "octivault-ibkr-allocator/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def fetch_flex_statement(token: str = None, query_id: str = None,
                         cache_path: str = None) -> str:
    """Two-step Flex Web Service fetch: SendRequest -> ReferenceCode -> GetStatement.

    IBKR generates the statement asynchronously and answers GetStatement with
    error 1019 ("statement generation in progress") until it is ready, so a
    single request is expected to fail on a cold query. Retrying that specific
    code is the documented flow, not a workaround.

    On success the raw XML is cached to disk. Every read path can then work from
    the cache, so a broker outage degrades to stale data rather than to no data.
    """
    token = token or os.getenv("IBKR_FLEX_TOKEN")
    query_id = query_id or os.getenv("IBKR_FLEX_QUERY_ID")
    cache_path = cache_path or CACHE_FILE
    if not token or not query_id:
        raise RuntimeError(
            "IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID are not set. Create a read-only "
            "Activity Flex Query in Client Portal (Performance & Reports -> Flex "
            "Queries) and enable the Flex Web Service, then put both in .env.")

    q = urllib.parse.urlencode({"t": token, "q": query_id, "v": "3"})
    resp = _flex_get(f"{FLEX_BASE}/SendRequest?{q}")
    root = ET.fromstring(resp)
    status = (root.findtext("Status") or "").strip()
    if status != "Success":
        raise RuntimeError(
            f"Flex SendRequest failed: code={root.findtext('ErrorCode')} "
            f"msg={root.findtext('ErrorMessage')}")
    ref = (root.findtext("ReferenceCode") or "").strip()
    base = (root.findtext("Url") or f"{FLEX_BASE}/GetStatement").strip()

    last_err = None
    for attempt in range(FLEX_RETRIES):
        q2 = urllib.parse.urlencode({"t": token, "q": ref, "v": "3"})
        xml = _flex_get(f"{base}?{q2}")
        if "<FlexQueryResponse" in xml:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(xml)
            return xml
        # Not ready yet (1019) or a real error — inspect and decide.
        try:
            err_root = ET.fromstring(xml)
            code = (err_root.findtext("ErrorCode") or "").strip()
            msg = (err_root.findtext("ErrorMessage") or "").strip()
        except ET.ParseError:
            code, msg = "?", xml[:120]
        last_err = f"code={code} msg={msg}"
        if code != "1019":                      # 1019 = still generating
            raise RuntimeError(f"Flex GetStatement failed: {last_err}")
        if attempt < FLEX_RETRIES - 1:
            _log(f"statement still generating; retrying in {FLEX_RETRY_DELAY_S:.0f}s")
            time.sleep(FLEX_RETRY_DELAY_S)
    raise RuntimeError(f"Flex GetStatement never became ready: {last_err}")


def load_statement_xml(refresh: bool = False) -> tuple[str, str]:
    """Return (xml, source). Prefers IBKR_FLEX_FILE, then a live fetch, then cache."""
    local = os.getenv("IBKR_FLEX_FILE")
    if local:
        with open(local) as f:
            return f.read(), f"file:{local}"
    if refresh or not os.path.exists(CACHE_FILE):
        try:
            return fetch_flex_statement(), "flex-web-service"
        except Exception as e:
            if not os.path.exists(CACHE_FILE):
                raise
            _alert(f"Flex fetch failed, using cached statement: {str(e)[:120]}")
    with open(CACHE_FILE) as f:
        return f.read(), f"cache:{CACHE_FILE}"


def _fnum(el, *names, default=0.0) -> float:
    for n in names:
        v = el.get(n)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                continue
    return default


def parse_statement(xml: str) -> dict:
    """Parse a Flex statement into positions, cash, deposits and NAV.

    Values are converted to the account base currency via `fxRateToBase` where
    the attribute is present, so a non-USD holding is not silently added to a
    USD total at face value.
    """
    root = ET.fromstring(xml)
    stmt = root.find(".//FlexStatement")
    account = stmt.get("accountId") if stmt is not None else None
    to_date = stmt.get("toDate") if stmt is not None else None

    positions: dict[str, dict] = {}
    for p in root.findall(".//OpenPosition"):
        if (p.get("assetCategory") or "STK").upper() not in ("STK", "FUND", "ETF"):
            continue
        sym = (p.get("symbol") or "").upper()
        if not sym:
            continue
        qty = _fnum(p, "position")
        mark = _fnum(p, "markPrice")
        fx = _fnum(p, "fxRateToBase", default=1.0) or 1.0
        val = _fnum(p, "positionValue", default=qty * mark) * fx
        # A Flex query can emit one row per lot; sum them into one position.
        if sym in positions:
            positions[sym]["qty"] += qty
            positions[sym]["value"] += val
        else:
            positions[sym] = {"qty": qty, "price": mark, "value": val,
                              "currency": p.get("currency") or "USD"}
    for sym, p in positions.items():
        p["qty"] = round(p["qty"], 8)
        p["value"] = round(p["value"], 2)

    cash = 0.0
    for c in root.findall(".//CashReportCurrency"):
        cur = (c.get("currency") or "").upper()
        if cur == "BASE_SUMMARY":
            cash = _fnum(c, "endingCash", "endingSettledCash")
            break
    else:
        for c in root.findall(".//CashReportCurrency"):
            cash += _fnum(c, "endingCash", "endingSettledCash") * (
                _fnum(c, "fxRateToBase", default=1.0) or 1.0)

    deposits = []
    for t in root.findall(".//CashTransaction"):
        ttype = (t.get("type") or "")
        if "deposit" not in ttype.lower() and "withdraw" not in ttype.lower():
            continue
        amt = _fnum(t, "amount") * (_fnum(t, "fxRateToBase", default=1.0) or 1.0)
        if amt == 0:
            continue
        deposits.append({
            "id": t.get("transactionID") or f"{t.get('dateTime')}:{amt}",
            "ts": t.get("dateTime") or t.get("reportDate") or "",
            "amount": round(amt, 2),
            "currency": t.get("currency") or "USD",
            "type": ttype,
        })

    reported_nav = None
    eq = root.findall(".//EquitySummaryByReportDateInBase")
    if eq:
        reported_nav = _fnum(eq[-1], "total")

    invested = round(sum(p["value"] for p in positions.values()), 2)
    nav = round(invested + cash, 2)
    return {"account": account, "as_of": to_date, "positions": positions,
            "cash": round(cash, 2), "invested": invested, "nav": nav,
            "reported_nav": reported_nav, "deposits": deposits}


# ═════════════════════════════════════════════════════════════════════════════
# THE ROUTER — where the next deposit goes
# ═════════════════════════════════════════════════════════════════════════════

def compute_drift(positions: dict, targets: dict[str, float]) -> dict:
    """Current weight vs target weight, in percentage points, per fund.

    Weights are measured against INVESTED value, not NAV. Including the cash
    reserve in the denominator makes every fund look permanently underweight,
    which would push the router to over-buy on every single deposit.
    """
    values = {s: float(positions.get(s, {}).get("value", 0.0)) for s in targets}
    for s, p in positions.items():
        values.setdefault(s.upper(), float(p.get("value", 0.0)))
    invested = sum(values.values())
    out = {}
    for sym in sorted(set(list(targets) + list(values))):
        cur_w = (values.get(sym, 0.0) / invested) if invested > 0 else 0.0
        tgt_w = targets.get(sym, 0.0)
        out[sym] = {
            "value": round(values.get(sym, 0.0), 2),
            "current_pct": round(100 * cur_w, 2),
            "target_pct": round(100 * tgt_w, 2),
            "drift_pp": round(100 * (cur_w - tgt_w), 2),
            "untracked": sym not in targets,
        }
    return {"invested": round(invested, 2), "funds": out}


def plan_deposit(positions: dict, cash: float, cfg: dict,
                 deposit: float = None) -> dict:
    """Turn a deposit into a buy list that pulls the book toward target.

    Gap-fill, not proportional: every dollar goes to whichever fund is furthest
    BELOW its target value, so the deposit does the rebalancing and nothing is
    ever sold. When the book is already on target this degenerates to the plain
    target split, which is the correct behaviour rather than a special case.

    Cash above `cash_cap_usd` is treated as deployable. The cap is in dollars on
    purpose: a percentage reserve grows with the book and drags on return
    forever.
    """
    targets = cfg["_funds"]
    deposit = float(cfg.get("monthly_deposit", 0.0)) if deposit is None else float(deposit)
    cash_cap = float(cfg.get("cash_cap_usd", 0.0))
    min_order = float(cfg.get("min_order_usd", 1.0))

    excess_cash = max(0.0, float(cash) - cash_cap)
    deployable = round(deposit + excess_cash, 2)
    values = {s: float(positions.get(s, {}).get("value", 0.0)) for s in targets}
    invested_now = sum(values.values())
    future = invested_now + deployable

    gaps = {s: max(0.0, future * targets[s] - values[s]) for s in targets}
    total_gap = sum(gaps.values())
    if deployable <= 0:
        alloc = {s: 0.0 for s in targets}
    elif total_gap <= 0:
        # Already at or above target everywhere (only reachable if a fund is
        # untracked or values are stale) — fall back to the plain split.
        alloc = {s: deployable * targets[s] for s in targets}
    else:
        alloc = {s: deployable * gaps[s] / total_gap for s in targets}

    # Drop sub-minimum tickets and redistribute, until every survivor clears the
    # minimum. IBKR fractional orders below ~$1 are rejected, and an order that
    # silently fails leaves cash undeployed for a month.
    while True:
        small = [s for s, a in alloc.items() if 0 < a < min_order]
        live = [s for s, a in alloc.items() if a >= min_order]
        if not small or not live:
            break
        freed = sum(alloc[s] for s in small)
        for s in small:
            alloc[s] = 0.0
        base = sum(gaps[s] for s in live) or sum(targets[s] for s in live)
        for s in live:
            share = (gaps[s] if sum(gaps[s] for s in live) > 0 else targets[s]) / base
            alloc[s] += freed * share

    alloc = {s: round(a, 2) for s, a in alloc.items()}
    # Push the rounding residual onto the largest ticket so the buy list sums
    # exactly to the deployable amount.
    residual = round(deployable - sum(alloc.values()), 2)
    if residual and alloc:
        biggest = max(alloc, key=lambda s: alloc[s])
        alloc[biggest] = round(alloc[biggest] + residual, 2)

    # Does this month's split differ enough from the standing recurring order
    # that the operator needs to go and edit it?
    band = float(cfg.get("drift_band_pct", 5.0))
    breaches = []
    for s in targets:
        this_w = 100 * alloc[s] / deployable if deployable else 0.0
        if abs(this_w - 100 * targets[s]) > band:
            breaches.append(s)

    return {"deposit": round(deposit, 2), "excess_cash": round(excess_cash, 2),
            "deployable": deployable, "cash_cap": cash_cap,
            "invested_now": round(invested_now, 2),
            "allocations": alloc, "needs_recurring_edit": breaches}


# ═════════════════════════════════════════════════════════════════════════════
# LOOK-THROUGH EXPOSURE
# ═════════════════════════════════════════════════════════════════════════════

def look_through(positions: dict, holdings: dict) -> dict:
    """Aggregate company-level exposure across funds.

    QQQM, SMH and VTI hold many of the same companies, so a fund-level weight
    understates single-name concentration badly. `holdings` maps
    fund -> {company: weight_pct} and is MAINTAINED BY HAND from issuer files;
    this function never invents it, and reports coverage so a partly-filled
    file cannot be mistaken for a complete picture.
    """
    total = sum(float(p.get("value", 0.0)) for p in positions.values())
    exposure: dict[str, float] = {}
    covered = 0.0
    for sym, p in positions.items():
        table = holdings.get(sym.upper(), {}).get("holdings")
        if not table:
            continue
        covered += float(p.get("value", 0.0))
        for name, pct in table.items():
            exposure[name] = exposure.get(name, 0.0) + float(p["value"]) * float(pct) / 100.0
    ranked = sorted(exposure.items(), key=lambda kv: -kv[1])
    return {
        "book_value": round(total, 2),
        "covered_value": round(covered, 2),
        "coverage_pct": round(100 * covered / total, 1) if total else 0.0,
        "top": [{"name": n, "usd": round(v, 2),
                 "pct_of_book": round(100 * v / total, 2) if total else 0.0}
                for n, v in ranked[:15]],
    }


# ═════════════════════════════════════════════════════════════════════════════
# DEPOSIT DETECTION + NAV RECORDING
# ═════════════════════════════════════════════════════════════════════════════

def detect_contributions(parsed: dict, state: dict) -> float:
    """New external money since the last check, from CASH TRANSACTIONS.

    Never inferred from a NAV jump: a market rally and a deposit are
    indistinguishable in NAV terms, and inferring would book every good week as
    a contribution (and so erase it from reported growth). Withdrawals count
    negatively so the contribution base stays correct.
    """
    seen = set(state.setdefault("seen_deposits", []))
    total = 0.0
    for d in parsed.get("deposits", []):
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        total += d["amount"]
        verb = "deposit" if d["amount"] > 0 else "withdrawal"
        _log(f"  [CONTRIB] {verb} ${d['amount']:+,.2f} on {str(d['ts'])[:10]} detected")
    state["seen_deposits"] = sorted(seen)
    if total:
        state["cumulative_contributions"] = round(
            float(state.get("cumulative_contributions", 0.0)) + total, 4)
    return total


def snapshot(parsed: dict, cfg: dict) -> dict:
    drift = compute_drift(parsed["positions"], cfg["_funds"])
    return {
        "nav": parsed["nav"],
        "cash": parsed["cash"],
        "invested": parsed["invested"],
        "as_of": parsed.get("as_of"),
        "positions": {s: {"qty": p["qty"], "value": p["value"]}
                      for s, p in parsed["positions"].items()},
        "weights": {s: d["current_pct"] for s, d in drift["funds"].items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _load_book(refresh: bool = False):
    cfg = load_targets()
    xml, source = load_statement_xml(refresh=refresh)
    parsed = parse_statement(xml)
    return cfg, parsed, source


def cmd_report(refresh: bool = False) -> None:
    cfg, parsed, source = _load_book(refresh)
    print("=" * 68)
    print("IBKR BOOK")
    print("=" * 68)
    print(f"  account          : {parsed['account']}   as of {parsed['as_of']}  ({source})")
    print(f"  NAV              : ${parsed['nav']:,.2f}")
    print(f"  invested / cash  : ${parsed['invested']:,.2f} / ${parsed['cash']:,.2f}")
    if parsed["reported_nav"] is not None and parsed["nav"]:
        diff = abs(parsed["reported_nav"] - parsed["nav"]) / parsed["nav"]
        if diff > 0.01:
            print(f"  ⚠️  computed NAV differs from IBKR's reported "
                  f"${parsed['reported_nav']:,.2f} by {100*diff:.1f}% "
                  f"— check the Flex query includes all asset classes")
    print()
    ledger.print_nav_report(NAV_FILE, "LEDGER")
    print()
    cmd_drift(parsed=parsed, cfg=cfg)
    age_days = None
    try:
        as_of = datetime.strptime(cfg.get("_as_of", ""), "%Y-%m-%d")
        age_days = (datetime.now() - as_of).days
    except ValueError:
        pass
    months = int(cfg.get("review_months", 12))
    if age_days is not None and age_days > months * 30:
        print(f"\n  📅 targets were set {age_days} days ago — due for the "
              f"{months}-month review (change them on a schedule, not on news)")


def cmd_drift(parsed=None, cfg=None, refresh: bool = False) -> None:
    if parsed is None or cfg is None:
        cfg, parsed, _ = _load_book(refresh)
    d = compute_drift(parsed["positions"], cfg["_funds"])
    band = float(cfg.get("drift_band_pct", 5.0))
    print("=" * 68)
    print(f"WEIGHTS vs TARGET   (invested ${d['invested']:,.2f}; band ±{band:.0f}pp)")
    print("=" * 68)
    print(f"  {'fund':<8}{'theme':<16}{'value':>12}{'now':>9}{'target':>9}{'drift':>9}")
    for sym, f in sorted(d["funds"].items(), key=lambda kv: -kv[1]["target_pct"]):
        flag = "  ⚠️" if abs(f["drift_pp"]) > band else ""
        theme = "UNTRACKED" if f["untracked"] else fund_theme(cfg, sym)
        print(f"  {sym:<8}{theme:<16}${f['value']:>11,.2f}"
              f"{f['current_pct']:>8.1f}%{f['target_pct']:>8.1f}%"
              f"{f['drift_pp']:>+8.1f}{flag}")
    untracked = [s for s, f in d["funds"].items() if f["untracked"] and f["value"] > 0]
    if untracked:
        print(f"\n  ⚠️  held but not in targets: {', '.join(untracked)} "
              f"— add them to the config or they are excluded from every plan")
    print("\n  themes:", ", ".join(
        f"{t} {100*w:.0f}%" for t, w in sorted(theme_weights(cfg).items(), key=lambda kv: -kv[1])))


def cmd_plan(amount: float = None, refresh: bool = False) -> None:
    cfg, parsed, _ = _load_book(refresh)
    p = plan_deposit(parsed["positions"], parsed["cash"], cfg, amount)
    print("=" * 68)
    print("NEXT DEPOSIT — BUY LIST")
    print("=" * 68)
    print(f"  deposit          : ${p['deposit']:,.2f}")
    if p["excess_cash"] > 0:
        print(f"  + idle cash      : ${p['excess_cash']:,.2f}  "
              f"(above the ${p['cash_cap']:,.0f} reserve)")
    print(f"  deployable       : ${p['deployable']:,.2f}")
    print()
    print(f"  {'fund':<8}{'theme':<16}{'buy':>12}{'% of deposit':>14}")
    for sym, amt in sorted(p["allocations"].items(), key=lambda kv: -kv[1]):
        pct = 100 * amt / p["deployable"] if p["deployable"] else 0.0
        print(f"  {sym:<8}{fund_theme(cfg, sym):<16}${amt:>11,.2f}{pct:>13.1f}%")
    print(f"  {'TOTAL':<24}${sum(p['allocations'].values()):>11,.2f}")
    print()
    print("  NO SELLS. Drift is corrected with new cash only.")
    if p["needs_recurring_edit"]:
        print(f"  ⚠️  drift has moved the split for {', '.join(p['needs_recurring_edit'])} "
              f"beyond ±{cfg.get('drift_band_pct')}pp —")
        print("      update the IBKR Recurring Investment amounts to match the list above.")
    else:
        print("  ✅ matches your standing recurring order within the band — nothing to edit.")


def cmd_exposure(refresh: bool = False) -> None:
    cfg, parsed, _ = _load_book(refresh)
    holdings = ledger.load_state(HOLDINGS_FILE, {})
    result = look_through(parsed["positions"], holdings)
    print("=" * 68)
    print("LOOK-THROUGH EXPOSURE — what you really own")
    print("=" * 68)
    if result["coverage_pct"] <= 0:
        print(f"  No fund holdings loaded. Populate {HOLDINGS_FILE} from the issuer")
        print("  holdings files (see the URLs in that file) to unwind fund overlap.")
        return
    print(f"  coverage: {result['coverage_pct']:.0f}% of ${result['book_value']:,.2f} "
          f"({'partial — unlisted funds excluded' if result['coverage_pct'] < 99 else 'full'})")
    print()
    for row in result["top"]:
        print(f"  {row['name']:<28}${row['usd']:>10,.2f}{row['pct_of_book']:>8.2f}% of book")


def cmd_fetch() -> None:
    xml = fetch_flex_statement()
    print(f"Fetched {len(xml):,} bytes -> {CACHE_FILE}")
    p = parse_statement(xml)
    print(f"account={p['account']} as_of={p['as_of']} nav=${p['nav']:,.2f} "
          f"positions={len(p['positions'])} cash_txns={len(p['deposits'])}")


def run_daemon() -> None:
    """Refresh the statement, record one honest NAV row, alert on drift."""
    global _lock_handle
    _lock_handle = ledger.acquire_lock(PIDFILE)
    if _lock_handle is None:
        print(f"[ibkr] another instance holds {PIDFILE} — refusing to start")
        return
    cfg = load_targets()
    _log(f"start — READ-ONLY. No orders are ever placed. targets={TARGETS_FILE} "
         f"poll={POLL_H}h")
    state = ledger.load_state(STATE_FILE, {})
    while True:
        if os.path.exists(KILL_FILE):
            _log("kill-switch present — idling")
            time.sleep(POLL_H * 3600)
            continue
        try:
            cfg = load_targets()                      # pick up operator edits live
            xml, source = load_statement_xml(refresh=True)
            parsed = parse_statement(xml)
            contributed = detect_contributions(parsed, state)
            snap = snapshot(parsed, cfg)
            row = ledger.record_nav(NAV_FILE, state, snap, contributed,
                                    extra={"source": source})
            _log(f"nav=${snap['nav']:,.2f} invested=${snap['invested']:,.2f} "
                 f"cash=${snap['cash']:,.2f} contrib=${row['cumulative_contributions']:,.2f} "
                 f"growth=${row['growth']:+,.2f}")
            band = float(cfg.get("drift_band_pct", 5.0))
            d = compute_drift(parsed["positions"], cfg["_funds"])
            for sym, f in d["funds"].items():
                if abs(f["drift_pp"]) > band:
                    _alert(f"{sym} drift {f['drift_pp']:+.1f}pp vs target "
                           f"{f['target_pct']:.1f}% — next deposit will correct it")
            if contributed:
                _alert(f"contribution ${contributed:+,.2f} detected — run "
                       f"`python3 ibkr_allocator.py plan` for the buy list")
            ledger.save_state(STATE_FILE, state)
        except Exception as e:
            _log(f"[ERROR] {str(e)[:160]}")
        time.sleep(POLL_H * 3600)


if __name__ == "__main__":
    arg = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    rest = sys.argv[2:]
    refresh = "--refresh" in rest
    try:
        if arg == "report":
            cmd_report(refresh)
        elif arg == "plan":
            amt = next((float(a) for a in rest if not a.startswith("-")), None)
            cmd_plan(amt, refresh)
        elif arg == "drift":
            cmd_drift(refresh=refresh)
        elif arg == "exposure":
            cmd_exposure(refresh)
        elif arg == "fetch":
            cmd_fetch()
        elif arg in ("", "run"):
            run_daemon()
        else:
            print(__doc__)
            sys.exit(2)
    except RuntimeError as e:
        print(f"error: {e}")
        sys.exit(1)
