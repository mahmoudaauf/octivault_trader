#!/usr/bin/env python3
"""
Venue-agnostic capital ledger — keeps CONTRIBUTIONS and RETURNS apart.

Extracted from `hybrid_allocator.py`'s allocate objective so the Binance daemon
and the IBKR daemon record growth the same way. The one idea it exists to
enforce:

    A deposit and a profit look IDENTICAL in a balance. Balance goes up, and
    the machine appears to be working.

Every function here exists to stop that confusion. `growth` is NAV minus
cumulative contributions minus the anchoring baseline, and it is the only
number in either daemon that reflects what the money actually earned.

Deliberately has no venue imports (no binance, no IBKR) and no network calls:
callers hand it a NAV snapshot and it appends a row. That keeps it testable
offline and reusable for a third venue later.

NOTE ON MIGRATION: `hybrid_allocator.py` still carries its own private copies of
`_load`/`_save`/`_record_nav`/`_nav_report`. They are behaviourally identical to
these. Switching it over is a small change but it requires restarting a daemon
that makes real earn subscriptions, so it is left as a deliberate follow-up
rather than done silently underneath a running process.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# ── state files ──────────────────────────────────────────────────────────────


def load_state(path: str, default: dict) -> dict:
    """Read a JSON state file, falling back to `default` if missing/corrupt.

    A corrupt state file must not crash a daemon on boot — it would take the
    machine offline for a parse error. Callers get the default and rebuild.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(default)


def save_state(path: str, obj: dict) -> None:
    """Atomically write a JSON state file (write-temp-then-rename).

    A daemon killed mid-write must never leave a half-written state file: the
    next boot would read it as corrupt and silently reset the ledger baseline.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def acquire_lock(pidfile: str):
    """Exclusive flock on a pidfile. Returns the held handle, or None if another
    instance holds it. The handle must stay referenced for the process lifetime
    — letting it be garbage collected releases the lock.
    """
    import fcntl

    os.makedirs(os.path.dirname(pidfile) or ".", exist_ok=True)
    fh = open(pidfile, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.seek(0)
    fh.write(str(os.getpid()))
    fh.truncate()
    fh.flush()
    return fh


# ── the ledger ───────────────────────────────────────────────────────────────


def record_nav(nav_file: str, state: dict, snap: dict,
               contributed: float = 0.0, extra: dict | None = None) -> dict:
    """Append one row to the equity curve, separating contributions from returns.

    `snap` must carry a "nav" key; everything else in it is copied through as
    the venue-specific breakdown. `state` carries "cumulative_contributions"
    and "nav_baseline" across runs.

    The baseline anchors on the first ever snapshot as (nav - contributions so
    far), so a ledger started on a funded account reports growth from zero
    rather than reporting the whole existing balance as profit.
    """
    contrib = float(state.get("cumulative_contributions", 0.0))
    if state.get("nav_baseline") is None:
        state["nav_baseline"] = round(float(snap["nav"]) - contrib, 4)
    baseline = float(state["nav_baseline"])

    row = dict(snap)
    row.update({
        "ts": datetime.now(timezone.utc).isoformat(),
        "cumulative_contributions": round(contrib, 4),
        "contributed_this_cycle": round(float(contributed), 4),
        # growth = what the account EARNED, with deposits removed.
        "growth": round(float(snap["nav"]) - contrib - baseline, 4),
    })
    if extra:
        row.update(extra)

    os.makedirs(os.path.dirname(nav_file) or ".", exist_ok=True)
    with open(nav_file, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_nav_rows(nav_file: str) -> list[dict]:
    """Load the equity curve, skipping any line that is not valid JSON.

    A truncated final line (daemon killed mid-append) must not make the whole
    history unreadable.
    """
    if not os.path.exists(nav_file):
        return []
    rows = []
    with open(nav_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def growth_summary(rows: list[dict]) -> dict | None:
    """Reduce an equity curve to the numbers worth reporting.

    `return_pct` is measured against capital actually put in (contributions
    plus whatever funded the account before tracking began), not against NAV —
    dividing by NAV would shrink a real return every time you paid money in,
    since NAV already includes the deposit.

    `invested` is derived as `nav - growth` rather than re-summed from
    `first.nav` + contributions: growth is DEFINED as
    `nav - cumulative_contributions - baseline`, so `nav - growth` is exactly
    `cumulative_contributions + baseline` by construction. Re-deriving it from
    `first.nav` double-counts whatever contribution landed in that first
    snapshot, since `first.nav` already reflects that same cash inflow.
    """
    if not rows:
        return None
    first, last = rows[0], rows[-1]
    contrib = float(last.get("cumulative_contributions", 0.0))
    nav = float(last.get("nav", 0.0))
    growth = float(last.get("growth", 0.0))
    invested = round(nav - growth, 4)
    return {
        "snapshots": len(rows),
        "first_ts": first.get("ts", ""),
        "last_ts": last.get("ts", ""),
        "nav": nav,
        "nav_first": float(first.get("nav", 0.0)),
        "contributions": contrib,
        "growth": growth,
        "invested": round(invested, 4),
        "return_pct": round(100.0 * growth / invested, 4) if invested > 0 else 0.0,
    }


def print_nav_report(nav_file: str, title: str = "NAV HISTORY") -> None:
    """Print the equity curve with contributions and returns kept apart."""
    rows = read_nav_rows(nav_file)
    s = growth_summary(rows)
    if not s:
        print("No NAV history yet.")
        return
    print("=" * 68)
    print(f"{title} — contributions and returns kept separate")
    print("=" * 68)
    print(f"  snapshots        : {s['snapshots']}  ({s['first_ts'][:16]} -> {s['last_ts'][:16]})")
    print(f"  NAV now          : ${s['nav']:,.2f}   (was ${s['nav_first']:,.2f})")
    print(f"  contributions    : ${s['contributions']:,.2f}  (deposits — NOT profit)")
    print(f"  GROWTH (earned)  : ${s['growth']:+,.2f}   <- the only honest number")
    print(f"  return on capital: {s['return_pct']:+.2f}%  (growth / money put in)")
    print("=" * 68)
