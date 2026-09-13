#!/usr/bin/env python3
"""Platform health — does the platform know when it is broken?

WHY THIS EXISTS
---------------
Six agents run unattended. Four of them are launchd StartInterval jobs with no
KeepAlive and no health check, and that combination has a specific failure mode:
a job that errors on EVERY run looks identical to one that works. launchd
reports the exit status of the last attempt, the plist stays loaded, `launchctl
list` shows it — and nothing produces output. An expired API key, a renamed
endpoint, a full disk, and the platform goes quiet without going down.

This project has already lost a week to exactly that shape. Three cron jobs died
in July 2026 when macOS TCC silently denied them disk access; they were noticed
six weeks later by reading a log by hand. The lesson recorded then was "check
output freshness, never assume a job runs". This is that check, automated.

WHAT MAKES A JOB HEALTHY
------------------------
Not "is it loaded" — loaded is nearly meaningless. Healthy means it WROTE
something recently. The budget for "recently" is derived from each job's own
StartInterval in its plist rather than hardcoded, so a cadence change cannot
leave a stale threshold behind silently claiming health. Three missed intervals
is the trigger: one missed run is a network blip, three is a pattern.

It also reads each job's stderr file. A job can write to stdout on schedule and
still be throwing on every cycle; the .err file is where that shows.

WHO WATCHES THIS
----------------
office_brief.py surfaces the summary, and this script is itself scheduled — so
if IT stops, its own log goes stale and the brief reports it. That is a loop
with no privileged node: every agent, including this one, is judged by whether
it produced output.

Read-only. Inspects plists and log timestamps; touches no money and no API.

Usage:
  python3 platform_health.py            # report, exit 1 if anything is unhealthy
  python3 platform_health.py --quiet    # only print problems (for scheduling)
Env: HEALTH_MISSED_INTERVALS (3)  HYBRID_ALERT_CMD
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
ROOT = Path(__file__).resolve().parent
MISSED = float(os.getenv("HEALTH_MISSED_INTERVALS", "3"))
ALERT = os.getenv("HYBRID_ALERT_CMD", str(ROOT / "hybrid_alert.sh"))

# The output each agent must be producing. A job is judged on its PRODUCT, not
# on whether launchd still has it loaded.
#   label -> (log file, fallback budget in minutes when the plist has no interval)
OUTPUTS = {
    "com.octivault.hybrid":          ("logs/hybrid.log", 25),
    "com.octivault.delisting":       ("logs/delisting_exit_paper.log", 90),
    "com.octivault.dnyscan":         ("logs/dny_scan.log", 180),
    "com.octivault.profit-research": ("logs/profit_research.log", 45),
    "com.octivault.opportunity":     ("logs/opportunity.log", 180),
    "com.octivault.p2pspread":       ("logs/p2p_spread.log", 45),
    "com.octivault.health":          ("logs/platform_health.log", 180),
}


def loaded() -> dict[str, str]:
    """{label: last exit status} for octivault jobs launchd currently holds."""
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return {}
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].startswith("com.octivault."):
            found[parts[2]] = parts[1]
    return found


def budget_minutes(label: str, fallback: int) -> tuple[float, str]:
    """Freshness budget from the job's OWN StartInterval, times MISSED.

    Derived rather than hardcoded: if someone changes a job's cadence, a
    hardcoded threshold would keep passing while the job ran at the wrong rate,
    or start failing for no reason. The plist is the source of truth.
    """
    p = AGENTS_DIR / f"{label}.plist"
    try:
        with open(p, "rb") as f:
            d = plistlib.load(f)
        iv = d.get("StartInterval")
        if iv:
            return (iv / 60.0) * MISSED, f"{iv/60:.0f}m × {MISSED:.0f}"
    except Exception:
        pass
    return float(fallback), f"{fallback}m default"


def stderr_tail(log: str, budget_min: float) -> str | None:
    """Last meaningful stderr line, but only if the job is erroring NOW.

    Two filters, both learned from the first run:

    1. The urllib3/LibreSSL warning is emitted by every script in this
       environment on every invocation. Surfacing it would mark all seven
       agents as erroring, permanently, which is the same as no signal at all.

    2. Age. The first run flagged dnyscan on a TimeoutError that turned out to
       be 2 occurrences against 201 completed scans — a ~1% transient network
       rate on a perfectly healthy job. An error only matters if the .err file
       was written within the job's own freshness budget; older than that and
       the job has since run clean.
    """
    err = ROOT / (log.rsplit(".", 1)[0] + ".err")
    try:
        if not err.exists() or err.stat().st_size == 0:
            return None
        if (time.time() - err.stat().st_mtime) / 60.0 > budget_min:
            return None                      # errored once, has run clean since
        lines = [l.strip() for l in err.read_text(errors="replace").splitlines() if l.strip()]
    except Exception:
        return None
    noise = ("NotOpenSSLWarning", "warnings.warn", "urllib3")
    real = [l for l in lines if not any(n in l for n in noise)]
    return real[-1][:110] if real else None


def check() -> list[dict]:
    live = loaded()
    now = time.time()
    rows = []
    for label, (log, fallback) in sorted(OUTPUTS.items()):
        budget, how = budget_minutes(label, fallback)
        path = ROOT / log
        age = (now - path.stat().st_mtime) / 60.0 if path.exists() else None
        is_loaded = label in live
        if not is_loaded:
            state, why = "DOWN", "not loaded in launchd"
        elif age is None:
            state, why = "DOWN", "no output file has ever been written"
        elif age > budget:
            state, why = "STALE", f"silent {age:.0f}m, budget {budget:.0f}m ({how})"
        else:
            state, why = "OK", f"wrote {age:.0f}m ago"
        rows.append({"label": label, "state": state, "why": why,
                     "age": age, "budget": budget,
                     "exit": live.get(label, "-"), "err": stderr_tail(log, budget)})
    return rows


def main() -> int:
    quiet = "--quiet" in sys.argv
    rows = check()
    bad = [r for r in rows if r["state"] != "OK"]
    erroring = [r for r in rows if r["err"]]

    if not quiet:
        print("=" * 72)
        print(f"PLATFORM HEALTH — {len(rows)} agents, "
              f"{len(rows)-len(bad)} healthy, {len(bad)} not")
        print("=" * 72)
        for r in rows:
            mark = {"OK": "  ok  ", "STALE": " STALE", "DOWN": " DOWN "}[r["state"]]
            print(f" [{mark}] {r['label']:<32}{r['why']}")
            if r["err"]:
                print(f"          ↳ stderr: {r['err']}")
        if not bad and not erroring:
            print("\n  Every agent is loaded AND producing output within its own cadence.")
        print("=" * 72)

    for r in bad:
        msg = f"platform health: {r['label']} is {r['state']} — {r['why']}"
        print(msg)
        if os.path.exists(ALERT) and os.access(ALERT, os.X_OK):
            try:
                subprocess.Popen([ALERT, msg], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except Exception:
                pass
    for r in erroring:
        print(f"platform health: {r['label']} is writing to stderr — {r['err']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
