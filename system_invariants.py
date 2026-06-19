#!/usr/bin/env python3
"""
Phase 0 — System Invariant Checker.

Makes the system catch its own bugs. Validates the invariants that, when broken,
caused every state/accounting bug we hit: NAV miscount, orphaned positions, unmanaged
positions (no stop), phantom positions, and saturated/dead decision weights.

Reads the bot's live believed state (runtime_state_snapshot.json + logs/tpsl_state.json
+ recent QUANT_LOOP/EV log lines) and reports PASS / WARN / FAIL per invariant.

    python3 system_invariants.py            # one-shot report (exit 1 if any FAIL)
    python3 system_invariants.py --watch    # re-check every 60s

Wire into the bot later by calling check_all() every N cycles and logging FAILs.
"""
from __future__ import annotations

import ast
import glob
import json
import sys
import time

SNAPSHOT = "runtime_state_snapshot.json"
TPSL = "logs/tpsl_state.json"
LOG = "logs/run_latest.log"

DUST_USD = 5.0          # below this notional an untracked holding is dust, not an orphan
NAV_TOL_USD = 1.0       # NAV identity tolerance
NAV_TOL_PCT = 0.03      # or 3%, whichever is larger
PASS, WARN, FAIL = "✅ PASS", "⚠️  WARN", "❌ FAIL"

# Quote/base assets that never count as tradable spot positions
_NON_TRADE = {"USDT", "BNB", "BTC"}


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _pos_qty(p) -> float:
    return float((p.get("qty", 0) if isinstance(p, dict) else getattr(p, "qty", 0)) or 0)


def _pos_field(p, k) -> float:
    return float((p.get(k, 0) if isinstance(p, dict) else getattr(p, k, 0)) or 0)


# ── Invariants ─────────────────────────────────────────────────────────────────

def inv_nav_identity(snap: dict) -> tuple:
    """NAV must equal free USDT + Σ position market value (no under/over-count)."""
    nav = float(snap.get("nav_usdt", 0) or 0)
    free = float(snap.get("free_balance_usdt", 0) or 0)
    positions = snap.get("positions", {}) or {}
    prices = snap.get("prices", {}) or {}
    pos_val = 0.0
    for sym, p in positions.items():
        mark = _pos_field(p, "mark_price") or float(prices.get(sym, 0) or 0)
        pos_val += _pos_qty(p) * mark
    derived = free + pos_val
    tol = max(NAV_TOL_USD, nav * NAV_TOL_PCT)
    if nav <= 0:
        return "NAV identity", WARN, f"nav_usdt={nav} (not set yet)"
    if abs(nav - derived) <= tol:
        return "NAV identity", PASS, f"nav={nav:.2f} ≈ free {free:.2f} + positions {pos_val:.2f}"
    return ("NAV identity", FAIL,
            f"nav={nav:.2f} but free {free:.2f} + positions {pos_val:.2f} = {derived:.2f} "
            f"(diff ${nav-derived:+.2f}) → mis-count")


def inv_tpsl_coverage(snap: dict, tpsl: dict) -> tuple:
    """Every tracked position must have a TP/SL armed (no unmanaged position)."""
    positions = snap.get("positions", {}) or {}
    sl = tpsl.get("sl_levels", {}) or {}
    tp = tpsl.get("tp_levels", {}) or {}
    unmanaged = [s for s in positions if not (sl.get(s) and tp.get(s))]
    if not positions:
        return "TP/SL coverage", PASS, "no open positions"
    if not unmanaged:
        return "TP/SL coverage", PASS, f"all {len(positions)} positions armed"
    return ("TP/SL coverage", FAIL,
            f"{len(unmanaged)} unmanaged (no stop): {unmanaged}")


def inv_positions_in_balance(snap: dict) -> tuple:
    """Every tracked position's base asset must exist in the exchange balance (no phantom)."""
    positions = snap.get("positions", {}) or {}
    balance = {k.upper(): float(v or 0) for k, v in (snap.get("balance", {}) or {}).items()}
    phantom = []
    for sym in positions:
        base = sym[:-4] if sym.endswith("USDT") else sym
        held = balance.get(base, 0.0)
        tracked = _pos_qty(positions[sym])
        if held <= 1e-8 or (tracked > 0 and held < tracked * 0.5):
            phantom.append(f"{sym}(tracked={tracked:.4g}, exch={held:.4g})")
    if not phantom:
        return "Positions ⊆ balance", PASS, "no phantom positions"
    return "Positions ⊆ balance", FAIL, f"phantom/oversized: {phantom}"


def inv_no_orphans(snap: dict) -> tuple:
    """Every held asset above dust must be tracked as a position (no orphaned holding)."""
    positions = {s.upper() for s in (snap.get("positions", {}) or {})}
    balance = {k.upper(): float(v or 0) for k, v in (snap.get("balance", {}) or {}).items()}
    prices = {k.upper(): float(v or 0) for k, v in (snap.get("prices", {}) or {}).items()}
    orphans = []
    for asset, qty in balance.items():
        if asset in _NON_TRADE or qty <= 1e-8:
            continue
        sym = asset + "USDT"
        if sym in positions:
            continue
        price = prices.get(sym, 0.0)
        notional = qty * price if price > 0 else 0.0
        if notional >= DUST_USD:
            orphans.append(f"{sym}(~${notional:.2f})")
    if not orphans:
        return "No orphaned holdings", PASS, "all >$5 holdings tracked"
    return ("No orphaned holdings", FAIL,
            f"{len(orphans)} held but untracked (no NAV/no stop): {orphans}")


def inv_confidence_not_saturated(log: str) -> tuple:
    """Recent emitted confidences must not be pinned/saturated at 1.0.

    Saturation signature = values clipped to 1.0 (the bug we fixed). 'All identical'
    is only suspicious when MULTIPLE distinct symbols share one value — a single active
    symbol legitimately repeats its own confidence and must not false-trip.
    """
    samples = []  # (symbol, conf)
    try:
        with open(log, errors="ignore") as f:
            for line in f:
                i = line.find("MLForecaster:EV] symbol=")
                if i == -1:
                    continue
                try:
                    sym = line[i + len("MLForecaster:EV] symbol="):].split()[0]
                    j = line.find("conf=", i)
                    samples.append((sym, float(line[j + 5:j + 10])))
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        return "Confidence calibration", WARN, "no log"
    samples = samples[-40:]
    if len(samples) < 5:
        return "Confidence calibration", WARN, f"only {len(samples)} samples"
    confs = [c for _, c in samples]
    symbols = {s for s, _ in samples}
    pinned_one = sum(1 for c in confs if c >= 0.999)
    distinct_conf = len(set(round(c, 2) for c in confs))
    if pinned_one > len(confs) * 0.8:
        return ("Confidence calibration", FAIL,
                f"{pinned_one}/{len(confs)} pinned at 1.0 → saturation regressed")
    # Only a dead-weight signal if many distinct symbols all share one confidence value.
    if len(symbols) >= 3 and distinct_conf <= 1:
        return ("Confidence calibration", FAIL,
                f"{len(symbols)} symbols all at one conf → dead weight")
    if len(symbols) == 1:
        return ("Confidence calibration", PASS,
                f"single active symbol ({list(symbols)[0]}) conf={confs[-1]:.2f}, not saturated")
    return ("Confidence calibration", PASS,
            f"{distinct_conf} distinct values across {len(symbols)} symbols (spread healthy)")


def inv_prices_for_positions(snap: dict) -> tuple:
    """Every open position must have a live price (else NAV/TP-SL run blind)."""
    positions = snap.get("positions", {}) or {}
    prices = {k.upper(): float(v or 0) for k, v in (snap.get("prices", {}) or {}).items()}
    missing = [s for s in positions if prices.get(s.upper(), 0) <= 0
               and _pos_field(positions[s], "mark_price") <= 0]
    if not positions:
        return "Position prices", PASS, "no positions"
    if not missing:
        return "Position prices", PASS, f"all {len(positions)} priced"
    return "Position prices", FAIL, f"no price for: {missing}"


def check_live(shared_state, tp_sl_engine=None) -> list:
    """In-process invariant check against live objects (for wiring into the bot loop).

    Returns list of (name, status, detail) for any non-PASS invariant. Empty list = healthy.
    Reads live shared_state instead of the snapshot file so it reflects the true cycle state.
    """
    try:
        positions = dict(getattr(shared_state, "positions", {}) or {})
        snap = {
            "nav_usdt": float(getattr(shared_state, "nav_usdt", 0) or 0),
            "free_balance_usdt": float(getattr(shared_state, "free_balance_usdt", 0) or 0),
            "positions": positions,
            "balance": dict(getattr(shared_state, "balance", {}) or {}),
            "prices": dict(getattr(shared_state, "prices", {}) or {}),
        }
        tpsl = {}
        if tp_sl_engine is not None:
            tpsl = {
                "sl_levels": dict(getattr(tp_sl_engine, "_sl_levels", {}) or {}),
                "tp_levels": dict(getattr(tp_sl_engine, "_tp_levels", {}) or {}),
            }
        results = [
            inv_nav_identity(snap),
            inv_tpsl_coverage(snap, tpsl),
            inv_positions_in_balance(snap),
            inv_no_orphans(snap),
            inv_prices_for_positions(snap),
        ]
        return [(n, s, d) for (n, s, d) in results if s != PASS]
    except Exception as e:
        return [("invariant_check_error", WARN, str(e))]


def check_all() -> list:
    snap = _load_json(SNAPSHOT)
    tpsl = _load_json(TPSL)
    age = time.time() - float(snap.get("ts", 0) or 0) if snap.get("ts") else None
    results = [
        inv_nav_identity(snap),
        inv_tpsl_coverage(snap, tpsl),
        inv_positions_in_balance(snap),
        inv_no_orphans(snap),
        inv_prices_for_positions(snap),
        inv_confidence_not_saturated(LOG),
    ]
    return results, age


def _print(results, age):
    import datetime as dt
    print("=" * 66)
    print(f"SYSTEM INVARIANTS — {dt.datetime.now():%H:%M:%S}"
          + (f"  (state age {age:.0f}s)" if age is not None else ""))
    print("=" * 66)
    worst = 0
    for name, status, detail in results:
        print(f"{status}  {name:24} {detail}")
        worst = max(worst, {PASS: 0, WARN: 1, FAIL: 2}[status])
    print("-" * 66)
    verdict = {0: "ALL INVARIANTS HOLD", 1: "WARNINGS (no hard violation)",
               2: "INVARIANT VIOLATION — investigate"}[worst]
    print(f"{['✅','⚠️','❌'][worst]} {verdict}")
    return worst


def main():
    if "--watch" in sys.argv:
        while True:
            results, age = check_all()
            _print(results, age)
            print()
            time.sleep(60)
    else:
        results, age = check_all()
        worst = _print(results, age)
        sys.exit(1 if worst == 2 else 0)


if __name__ == "__main__":
    main()
