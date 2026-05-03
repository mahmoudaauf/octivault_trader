#!/usr/bin/env python3
"""
objective_tracker.py — score gates G1..G6 from session artefacts.

Reads:
  • checkpoint_metrics.json
  • objective_controller_state.json (if present)

Writes:
  • OBJECTIVE_DASHBOARD.md (auto-generated status report)

Usage:
  python3 objective_tracker.py
  python3 objective_tracker.py --json    # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent
CP_FILE = ROOT / "checkpoint_metrics.json"
OFC_FILE = ROOT / "objective_controller_state.json"
DASHBOARD = ROOT / "OBJECTIVE_DASHBOARD.md"

# Set-points (must match objective_feedback_controller.DEFAULTS)
DAILY_TARGET_PCT = 2.0
HOURLY_TARGET_PCT = DAILY_TARGET_PCT / 24
ROLLING_4H_TARGET_PCT = HOURLY_TARGET_PCT * 4   # 0.333%
DD_MAX_PCT = 5.0
MIN_NET_EDGE_BPS = 5.0
MIN_CHECKPOINTS_PER_HOUR = 4   # one every 15 min


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def score_gates(cp: Dict[str, Any], ofc: Dict[str, Any]) -> List[Dict[str, Any]]:
    checkpoints = cp.get("checkpoints", []) or []
    session_start = cp.get("session_start")
    session_end = cp.get("session_end")

    elapsed_h = 0.0
    if session_start and session_end:
        try:
            t0 = datetime.fromisoformat(session_start)
            t1 = datetime.fromisoformat(session_end)
            elapsed_h = max((t1 - t0).total_seconds() / 3600.0, 1e-3)
        except Exception:
            pass

    gates: List[Dict[str, Any]] = []

    # ── G1: Telemetry cadence ────────────────────────────────────────
    cp_per_h = (len(checkpoints) / elapsed_h) if elapsed_h > 0 else 0.0
    gates.append({
        "id": "G1",
        "name": "Telemetry cadence",
        "metric": f"{cp_per_h:.2f} checkpoints/h",
        "threshold": f"≥ {MIN_CHECKPOINTS_PER_HOUR}",
        "passed": cp_per_h >= MIN_CHECKPOINTS_PER_HOUR,
        "fix_hint": "Ensure 2HOUR_CHECKPOINT_SESSION emits at least every 15 min",
    })

    # ── G2: 4h rolling pace (from latest checkpoint vs 4h-ago) ───────
    pace_4h = _rolling_pace_pct(checkpoints, hours=4)
    gates.append({
        "id": "G2",
        "name": "4h rolling NAV pace",
        "metric": f"{pace_4h:+.3f}% / 4h" if pace_4h is not None else "n/a",
        "threshold": f"≥ +{ROLLING_4H_TARGET_PCT:.3f}%",
        "passed": (pace_4h is not None) and pace_4h >= ROLLING_4H_TARGET_PCT,
        "fix_hint": "Loop will raise size_multiplier; check ENTRY logic if persistent",
    })

    # ── G3: Daily target ─────────────────────────────────────────────
    daily_pct = _session_change_pct(checkpoints)
    gates.append({
        "id": "G3",
        "name": "Session NAV change",
        "metric": f"{daily_pct:+.3f}%" if daily_pct is not None else "n/a",
        "threshold": f"≥ +{DAILY_TARGET_PCT * 0.75:.2f}% (75% of daily target)",
        "passed": (daily_pct is not None) and daily_pct >= DAILY_TARGET_PCT * 0.75,
        "fix_hint": "Increase throughput target or extend session length",
    })

    # ── G4: Max drawdown ─────────────────────────────────────────────
    max_dd = _max_drawdown_pct(checkpoints)
    gates.append({
        "id": "G4",
        "name": "Intra-session max drawdown",
        "metric": f"{max_dd:.2f}%" if max_dd is not None else "n/a",
        "threshold": f"≤ {DD_MAX_PCT}%",
        "passed": (max_dd is not None) and max_dd <= DD_MAX_PCT,
        "fix_hint": "Tighten size_multiplier; verify kill-switch fired when needed",
    })

    # ── G5: Net edge (from last checkpoint metrics) ──────────────────
    last_metrics = _last_checkpoint_metrics(checkpoints)
    net_edge = last_metrics.get("avg_net_profit_bps")
    gates.append({
        "id": "G5",
        "name": "Avg net profit per trade",
        "metric": f"{net_edge:.1f} bps" if net_edge is not None else "n/a",
        "threshold": f"> {MIN_NET_EDGE_BPS} bps",
        "passed": (net_edge is not None) and net_edge > MIN_NET_EDGE_BPS,
        "fix_hint": "Raise confidence_floor; revisit fee/slippage assumptions",
    })

    # ── G6: Knob convergence (last 5 OFC steps) ──────────────────────
    convergence = _knob_convergence(ofc)
    gates.append({
        "id": "G6",
        "name": "Controller convergence",
        "metric": (f"σ={convergence['sigma']:.4f} "
                   f"over {convergence['n']} steps")
                  if convergence else "n/a",
        "threshold": "σ ≤ 0.05 (knobs settling)",
        "passed": bool(convergence) and convergence["sigma"] <= 0.05,
        "fix_hint": "Lower OBJ_KP_* gains if oscillating",
    })

    return gates


from typing import Optional
def _rolling_pace_pct(cps: List[dict], hours: int) -> Optional[float]:
    if len(cps) < 2:
        return None
    last = cps[-1]
    last_ts = _parse_ts(last)
    last_bal = _bal(last)
    if last_ts is None or last_bal is None:
        return None
    cutoff = last_ts - timedelta(hours=hours)
    older = next((c for c in cps if (_parse_ts(c) or last_ts) >= cutoff), None)
    if older is None or _bal(older) in (None, 0):
        return None
    base = _bal(older)
    return (last_bal - base) / base * 100.0


def _session_change_pct(cps: List[dict]) -> Optional[float]:
    if len(cps) < 1:
        return None
    first_bal = _bal(cps[0])
    last_bal = _bal(cps[-1])
    if not first_bal or last_bal is None:
        return None
    return (last_bal - first_bal) / first_bal * 100.0


def _max_drawdown_pct(cps: List[dict]) -> Optional[float]:
    bals = [_bal(c) for c in cps]
    bals = [b for b in bals if b]
    if not bals:
        return None
    peak = bals[0]
    max_dd = 0.0
    for b in bals:
        peak = max(peak, b)
        dd = (peak - b) / peak * 100.0 if peak else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _last_checkpoint_metrics(cps: List[dict]) -> Dict[str, float]:
    if not cps:
        return {}
    last = cps[-1]
    out = {}
    for section in ("profit", "capital", "adaptation"):
        out.update((last.get("checks", {}).get(section, {}) or {}).get("metrics", {}) or {})
    return out


def _knob_convergence(ofc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hist = ofc.get("history_tail") or []
    if len(hist) < 3:
        return None
    series = []
    for h in hist[-5:]:
        k = h.get("knobs_after") or {}
        if "size_multiplier" in k:
            series.append(float(k["size_multiplier"]))
    if len(series) < 3:
        return None
    mean = sum(series) / len(series)
    var = sum((x - mean) ** 2 for x in series) / len(series)
    return {"sigma": var ** 0.5, "n": len(series)}


def _parse_ts(c: dict) -> Optional[datetime]:
    ts = c.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _bal(c: dict) -> Optional[float]:
    m = (c.get("checks", {}).get("profit", {}) or {}).get("metrics", {}) or {}
    b = m.get("balance")
    return float(b) if b not in (None, 0, 0.0) else None


def write_dashboard(gates: List[Dict[str, Any]]) -> None:
    n_pass = sum(1 for g in gates if g["passed"])
    overall = "🟢 ON-OBJECTIVE" if n_pass == len(gates) else (
        "🟡 PARTIAL" if n_pass >= 4 else "🔴 OFF-OBJECTIVE"
    )
    lines = [
        "# 🎯 Objective Dashboard",
        "",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        "",
        f"**Overall:** {overall}  ({n_pass}/{len(gates)} gates green)",
        "",
        "| Gate | Name | Metric | Threshold | Status | Fix hint |",
        "|---|---|---|---|---|---|",
    ]
    for g in gates:
        status = "✅" if g["passed"] else "❌"
        lines.append(
            f"| {g['id']} | {g['name']} | {g['metric']} | {g['threshold']} | "
            f"{status} | {g['fix_hint']} |"
        )
    lines += [
        "",
        "## How to read this",
        "* **G1–G2** are *prerequisites* — without telemetry & pace, the controller is blind.",
        "* **G3–G5** are *objective metrics* — the actual +2%/day contract.",
        "* **G6** is *stability* — confirms the auto-calibration is converging.",
        "",
        "Run `python3 objective_tracker.py` after each session to refresh.",
    ]
    DASHBOARD.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    args = parser.parse_args()

    cp = _load_json(CP_FILE)
    ofc = _load_json(OFC_FILE)
    gates = score_gates(cp, ofc)
    write_dashboard(gates)

    if args.json:
        json.dump({"gates": gates}, sys.stdout, indent=2, default=str)
        return

    print(f"\n🎯 Objective gates ({CP_FILE.name}):\n")
    for g in gates:
        mark = "✅" if g["passed"] else "❌"
        print(f"  {mark}  {g['id']}  {g['name']:<32s}  {g['metric']:<28s}  ({g['threshold']})")
    n_pass = sum(1 for g in gates if g["passed"])
    print(f"\n{n_pass}/{len(gates)} gates green — see {DASHBOARD.name}\n")


if __name__ == "__main__":
    main()
