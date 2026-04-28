#!/usr/bin/env python3
"""
OctiVault Live Monitor
======================
Real-time dashboard: balance, NAV, positions, regime, compounding, P&L.
Run alongside the trading bot:
    python3 LIVE_MONITOR.py

Reads from:
  - Live log file (parsed for events)
  - Binance REST (spot balances, current prices)
  - .env config

Refreshes every 10 seconds.
"""

import os, sys, time, json, re, subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── colour codes ──────────────────────────────────────────────────────────────
GRN  = '\033[92m';  YLW  = '\033[93m';  RED  = '\033[91m'
CYN  = '\033[96m';  BLU  = '\033[94m';  MAG  = '\033[95m'
BLD  = '\033[1m';   DIM  = '\033[2m';   RST  = '\033[0m'

LOG_FILE   = ROOT / "logs" / "octivault_master_orchestrator.log"
REFRESH_S  = 10   # dashboard refresh interval

# ── Binance REST fetch ─────────────────────────────────────────────────────────
def _fetch_balances():
    """Return {asset: free_float} from Binance via python-binance."""
    try:
        from binance.client import Client
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC", "")
        if not (api_key and api_secret):
            return {}
        c = Client(api_key, api_secret)
        info = c.get_account()
        return {
            b["asset"]: float(b["free"]) + float(b["locked"])
            for b in info["balances"]
            if float(b["free"]) + float(b["locked"]) > 0
        }
    except Exception as e:
        return {"_error": str(e)}


def _fetch_prices(symbols):
    """Return {symbol: float_price}."""
    try:
        from binance.client import Client
        api_key    = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET_HMAC", "")
        c = Client(api_key, api_secret)
        tickers = c.get_all_tickers()
        return {t["symbol"]: float(t["price"]) for t in tickers}
    except Exception:
        return {}


def _nav_from_balances(balances, prices):
    """Calculate total NAV in USDT."""
    nav = 0.0
    breakdown = {}
    for asset, qty in balances.items():
        if asset.startswith("_"):
            continue
        if asset == "USDT":
            nav += qty
            breakdown["USDT"] = qty
        else:
            sym = asset + "USDT"
            price = prices.get(sym, 0.0)
            if price > 0:
                value = qty * price
                nav += value
                breakdown[asset] = value
    return nav, breakdown


# ── log parser ────────────────────────────────────────────────────────────────
_LOG_TAIL = 500   # lines to scan

def _tail_log(n=_LOG_TAIL):
    if not LOG_FILE.exists():
        return []
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(LOG_FILE)],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.splitlines()
    except Exception:
        return []


def _parse_log_events(lines):
    """Extract key events from recent log lines."""
    events = {
        "trades":        [],  # (ts, side, symbol, quote)
        "regime":        "unknown",
        "compounding":   [],  # log lines about compounding
        "errors":        [],
        "gate_passed":   0,
        "gate_dropped":  0,
        "recovery":      None,
        "last_cycle_ts": None,
    }

    trade_re     = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(?:EXEC|BUY|SELL)\].*?(BUY|SELL)\s+(\w+).*?\$?([\d.]+)", re.I)
    regime_re    = re.compile(r"regime[=:]\s*([A-Z_]+)", re.I)
    gate_ok_re   = re.compile(r"Filtered intents.*out=(\d+)")
    gate_drop_re = re.compile(r"Filtered intents.*dropped=(\d+)")
    compound_re  = re.compile(r"compound|reinvest|CompoundingEngine", re.I)
    error_re     = re.compile(r"\bERROR\b|\bCRITICAL\b", re.I)
    cycle_re     = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[Meta.*cycle\]", re.I)
    recovery_re  = re.compile(r"StartupOrchestrator complete|positions hydrated", re.I)

    for line in lines[-_LOG_TAIL:]:
        # trades
        m = trade_re.search(line)
        if m:
            events["trades"].append((m.group(1), m.group(2), m.group(3), m.group(4)))
        # regime
        m = regime_re.search(line)
        if m:
            events["regime"] = m.group(1).upper()
        # gate
        m = gate_ok_re.search(line)
        if m:
            events["gate_passed"] += int(m.group(1))
        m = gate_drop_re.search(line)
        if m:
            events["gate_dropped"] += int(m.group(1))
        # compounding
        if compound_re.search(line):
            events["compounding"].append(line.strip()[-120:])
        # errors
        if error_re.search(line):
            events["errors"].append(line.strip()[-120:])
        # cycle ts
        m = cycle_re.search(line)
        if m:
            events["last_cycle_ts"] = m.group(1)
        # recovery
        if recovery_re.search(line):
            events["recovery"] = line.strip()[-100:]

    events["trades"] = events["trades"][-5:]        # last 5 trades
    events["compounding"] = events["compounding"][-3:]
    events["errors"] = events["errors"][-5:]
    return events


# ── process check ─────────────────────────────────────────────────────────────
def _bot_running():
    try:
        r = subprocess.run(
            "ps aux | grep 'MASTER_SYSTEM_ORCHESTRATOR' | grep -v grep | wc -l",
            shell=True, capture_output=True, text=True, timeout=3
        )
        return int(r.stdout.strip()) > 0
    except Exception:
        return False


# ── dashboard render ──────────────────────────────────────────────────────────
def _bar(pct, width=30, filled='█', empty='░'):
    pct = max(0.0, min(1.0, pct))
    n = int(pct * width)
    return filled * n + empty * (width - n)


def _render(balances, prices, events, session_start_nav, snapshot_ts):
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nav, breakdown = _nav_from_balances(balances, prices)
    usdt_free = balances.get("USDT", 0.0)

    # P&L vs session start
    if session_start_nav and session_start_nav > 0:
        pnl_abs = nav - session_start_nav
        pnl_pct = (pnl_abs / session_start_nav) * 100.0
        pnl_color = GRN if pnl_abs >= 0 else RED
        pnl_str = f"{pnl_color}{BLD}{pnl_abs:+.4f} USDT  ({pnl_pct:+.3f}%){RST}"
    else:
        pnl_str = f"{DIM}(awaiting data){RST}"

    bot_status = f"{GRN}🟢 RUNNING{RST}" if _bot_running() else f"{RED}🔴 STOPPED{RST}"

    regime = events["regime"]
    regime_color = YLW if "MICRO" in regime else GRN

    print(f"{BLD}{CYN}╔══════════════════════════════════════════════════════════════╗{RST}")
    print(f"{BLD}{CYN}║     OctiVault  ·  Autonomous Wealth Engine  ·  Live          ║{RST}")
    print(f"{BLD}{CYN}╚══════════════════════════════════════════════════════════════╝{RST}")
    print(f"  {DIM}Refreshed: {now}   Bot: {bot_status}{RST}")
    print()

    # ── NAV & Balance ──
    print(f"{BLD}  💰 Portfolio NAV{RST}")
    print(f"     Total NAV   : {BLD}{GRN}${nav:,.4f} USDT{RST}")
    print(f"     Free USDT   : {BLD}${usdt_free:,.4f}{RST}")
    print(f"     Session P&L : {pnl_str}")
    print()

    # ── Holdings ──
    if breakdown:
        print(f"{BLD}  📊 Holdings{RST}")
        sorted_hold = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        for asset, val in sorted_hold[:8]:
            qty = balances.get(asset, 0.0)
            pct = val / nav if nav > 0 else 0.0
            bar = _bar(pct, width=20)
            if asset == "USDT":
                print(f"     {BLD}{asset:<8}{RST}  {val:>10.4f} USDT  {DIM}{bar} {pct*100:.1f}%{RST}")
            else:
                price = prices.get(asset + "USDT", 0.0)
                print(f"     {BLD}{asset:<8}{RST}  {qty:>12.6f}  @ ${price:,.2f}  = ${val:>9.4f}  {DIM}{bar} {pct*100:.1f}%{RST}")
    print()

    # ── Regime & Engine Status ──
    print(f"{BLD}  🧠 Engine Status{RST}")
    print(f"     Market Regime   : {regime_color}{BLD}{regime}{RST}")
    print(f"     Last Cycle      : {DIM}{events['last_cycle_ts'] or 'none yet'}{RST}")
    passed  = events["gate_passed"]
    dropped = events["gate_dropped"]
    total   = passed + dropped
    gate_pct = (passed / total * 100) if total > 0 else 0
    gate_color = GRN if gate_pct > 50 else (YLW if gate_pct > 0 else RED)
    print(f"     Signal Gate     : {gate_color}{passed} passed / {dropped} dropped{RST}  (last {_LOG_TAIL} log lines)")
    print()

    # ── Recent Trades ──
    print(f"{BLD}  📈 Recent Trades{RST}")
    if events["trades"]:
        for ts, side, sym, quote in reversed(events["trades"]):
            sc = GRN if side.upper() == "BUY" else MAG
            print(f"     {DIM}{ts}{RST}  {sc}{BLD}{side:<4}{RST}  {CYN}{sym:<12}{RST}  ${quote}")
    else:
        print(f"     {DIM}No trades yet in this log window{RST}")
    print()

    # ── Compounding ──
    print(f"{BLD}  🔄 Compounding Engine{RST}")
    if events["compounding"]:
        for line in events["compounding"]:
            print(f"     {DIM}{line}{RST}")
    else:
        print(f"     {DIM}Waiting for first realized profit...{RST}")
    print()

    # ── Recovery ──
    if events["recovery"]:
        print(f"{BLD}  ✅ State Recovery{RST}")
        print(f"     {GRN}{events['recovery']}{RST}")
        print()

    # ── Errors ──
    if events["errors"]:
        print(f"{BLD}  ⚠  Recent Errors{RST}")
        for e in events["errors"][-3:]:
            print(f"     {RED}{e}{RST}")
        print()

    # ── Footer ──
    print(f"  {DIM}Log: {LOG_FILE}   Ctrl+C to exit monitor{RST}")
    print(f"{BLD}{CYN}{'─'*66}{RST}")


# ── main loop ─────────────────────────────────────────────────────────────────
def main():
    print(f"{BLD}{GRN}OctiVault Live Monitor starting...{RST}")
    print(f"{DIM}Fetching initial portfolio state from Binance...{RST}")

    session_start_nav = None
    snapshot_ts       = None

    try:
        while True:
            # Fetch live data
            balances = _fetch_balances()
            if not balances or "_error" in balances:
                err = balances.get("_error", "unknown") if balances else "no data"
                os.system("clear")
                print(f"{RED}❌ Cannot reach Binance: {err}{RST}")
                print(f"{DIM}Retrying in {REFRESH_S}s...{RST}")
                time.sleep(REFRESH_S)
                continue

            # Build price map for assets we hold
            symbols = [a + "USDT" for a in balances if a != "USDT" and not a.startswith("_")]
            prices  = _fetch_prices(symbols)

            nav, _ = _nav_from_balances(balances, prices)
            if session_start_nav is None and nav > 0:
                session_start_nav = nav
                snapshot_ts = datetime.now().isoformat()

            # Parse log
            lines  = _tail_log()
            events = _parse_log_events(lines)

            # Render
            _render(balances, prices, events, session_start_nav, snapshot_ts)

            time.sleep(REFRESH_S)

    except KeyboardInterrupt:
        print(f"\n{YLW}Monitor stopped.{RST}")


if __name__ == "__main__":
    main()
