#!/usr/bin/env python3
"""
Trade monitor — tracks progress toward a 50-trade goal with win/loss stats.

Replaces the old `simple_checkpoint_monitor.sh`, which read a non-existent
`live_run.log` and never recorded anything. This reads the real run log and
parses the `TRADE_CLOSED` lines emitted by core_engine/native/executor.py on
every sell fill, then tallies wins/losses and progress to the target.

Robust to log rotation/truncation: every closed trade is appended (deduped) to
`trade_ledger.jsonl`, so the running tally survives the supervisor truncating
run_latest.log at 100MB. Live snapshot is written to `trade_progress.json`.

Usage:   nohup python3 trade_monitor.py >> logs/trade_monitor.log 2>&1 &
Stop:    pkill -f trade_monitor.py
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_LOG = os.path.join(HERE, "logs", "run_latest.log")
LEDGER = os.path.join(HERE, "trade_ledger.jsonl")
STATE = os.path.join(HERE, "trade_progress.json")

TARGET_TRADES = int(os.getenv("TRADE_TARGET", "50"))
POLL_SECONDS = float(os.getenv("TRADE_MONITOR_POLL_S", "30"))
# Round-trip fee + spread (%). The executor logs GROSS pnl_pct; a "WIN" there can
# still be a NET loss once round-trip costs are paid. We reclassify on net so the
# win-rate reflects money, not gross ticks. Matches round_trip_cost in the logs.
ROUND_TRIP_FEE_PCT = float(os.getenv("ROUND_TRIP_FEE_PCT", "0.38"))

# Matches the executor's TRADE_CLOSED line. Leading group captures the log
# timestamp so each close has a stable dedup signature.
LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r".*TRADE_CLOSED (?P<symbol>\S+) (?P<outcome>WIN|LOSS|FLAT) "
    r"entry=(?P<entry>[-\d.]+) exit=(?P<exit>[-\d.]+) qty=(?P<qty>[-\d.]+) "
    r"gross_pnl=(?P<pnl>-?[\d.]+) USDT pnl_pct=(?P<pct>-?[\d.]+)%"
)


def _sig(rec: dict) -> str:
    """Stable signature for de-duplication across log rotations."""
    return f"{rec['ts']}|{rec['symbol']}|{rec['exit']}"


def load_ledger() -> tuple[list[dict], set[str]]:
    trades: list[dict] = []
    seen: set[str] = set()
    if os.path.exists(LEDGER):
        with open(LEDGER, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = _sig(rec)
                if s not in seen:
                    seen.add(s)
                    trades.append(rec)
    return trades, seen


def scan_run_log(seen: set[str]) -> list[dict]:
    """Return newly-seen TRADE_CLOSED records from the current run log."""
    fresh: list[dict] = []
    if not os.path.exists(RUN_LOG):
        return fresh
    try:
        with open(RUN_LOG, "r", errors="ignore") as fh:
            for line in fh:
                if "TRADE_CLOSED" not in line:
                    continue
                m = LINE_RE.search(line)
                if not m:
                    continue
                rec = {
                    "ts": m.group("ts"),
                    "symbol": m.group("symbol"),
                    "outcome": m.group("outcome"),
                    "entry": float(m.group("entry")),
                    "exit": float(m.group("exit")),
                    "qty": float(m.group("qty")),
                    "gross_pnl": float(m.group("pnl")),
                    "pnl_pct": float(m.group("pct")),
                }
                s = _sig(rec)
                if s not in seen:
                    seen.add(s)
                    fresh.append(rec)
    except OSError:
        pass
    return fresh


def append_ledger(records: list[dict]) -> None:
    if not records:
        return
    with open(LEDGER, "a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _net_pct(t: dict) -> float:
    """Net P&L %% after the round-trip fee — the figure that reflects actual money."""
    return float(t.get("pnl_pct", 0.0)) - ROUND_TRIP_FEE_PCT


def summarize(trades: list[dict]) -> dict:
    total = len(trades)
    # Gross outcomes (as logged by the executor).
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    flat = sum(1 for t in trades if t["outcome"] == "FLAT")
    decided = wins + losses
    win_rate = (wins / decided) if decided else 0.0
    cum_pnl = sum(t["gross_pnl"] for t in trades)
    # Net-of-fee outcomes — the truth. A gross "WIN" below the round-trip cost is
    # a net loss; this is what catches the fee-churn trap.
    net_wins = sum(1 for t in trades if _net_pct(t) > 0)
    net_losses = sum(1 for t in trades if _net_pct(t) < 0)
    net_decided = net_wins + net_losses
    net_win_rate = (net_wins / net_decided) if net_decided else 0.0
    # Convert each trade's net %% to USDT using its notional (gross_pnl/pnl_pct ratio).
    cum_net_pnl = 0.0
    for t in trades:
        pct = float(t.get("pnl_pct", 0.0))
        notional = (float(t.get("gross_pnl", 0.0)) / (pct / 100.0)) if pct else 0.0
        cum_net_pnl += notional * (_net_pct(t) / 100.0)
    return {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_trades": TARGET_TRADES,
        "total_trades": total,
        "round_trip_fee_pct": ROUND_TRIP_FEE_PCT,
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate": round(win_rate, 4),
        "net_wins": net_wins,
        "net_losses": net_losses,
        "net_win_rate": round(net_win_rate, 4),
        "cum_gross_pnl_usdt": round(cum_pnl, 4),
        "cum_net_pnl_usdt": round(cum_net_pnl, 4),
        "remaining_to_target": max(0, TARGET_TRADES - total),
        "target_reached": total >= TARGET_TRADES,
    }


def render(state: dict) -> str:
    done = state["total_trades"]
    tgt = state["target_trades"]
    filled = int(min(1.0, done / tgt) * 30) if tgt else 0
    bar = "█" * filled + "░" * (30 - filled)
    return (
        f"[{state['updated']}] TRADES {done}/{tgt} [{bar}] "
        f"gross[W:{state['wins']} L:{state['losses']} wr={state['win_rate'] * 100:.0f}% "
        f"{state['cum_gross_pnl_usdt']:+.4f}]  "
        f"NET[W:{state['net_wins']} L:{state['net_losses']} wr={state['net_win_rate'] * 100:.0f}% "
        f"{state['cum_net_pnl_usdt']:+.4f} USDT]"
        + ("  🎯 TARGET REACHED" if state["target_reached"] else "")
    )


def main() -> None:
    print("=" * 80)
    print(f"TRADE MONITOR — target {TARGET_TRADES} trades, win/loss tracking")
    print(f"  run log: {RUN_LOG}")
    print(f"  ledger : {LEDGER}")
    print("=" * 80, flush=True)

    trades, seen = load_ledger()
    last_render = ""
    while True:
        fresh = scan_run_log(seen)
        if fresh:
            append_ledger(fresh)
            trades.extend(fresh)
            for t in fresh:
                print(
                    f"  → NEW {t['outcome']} {t['symbol']} "
                    f"pnl={t['gross_pnl']:+.4f} USDT ({t['pnl_pct']:+.3f}%)",
                    flush=True,
                )
        state = summarize(trades)
        with open(STATE, "w") as fh:
            json.dump(state, fh, indent=2)
        line = render(state)
        # Only reprint the status bar when something changed or every loop tick.
        if line[20:] != last_render[20:] or fresh:
            print(line, flush=True)
            last_render = line
        if state["target_reached"]:
            # Keep running so post-target trades still tally, but note it once.
            pass
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
