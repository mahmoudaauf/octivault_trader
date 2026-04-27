#!/usr/bin/env python3
"""
6-hour execution-focused monitor.

Tracks:
- AgentManager intent throughput
- Pre-publish gate drops
- Trade journal submissions/fills/blockers

Outputs checkpoint summaries every 30 minutes (12 checkpoints over 6 hours).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


DEFAULT_DURATION_HOURS = 6.0
DEFAULT_CHECKPOINT_MINUTES = 30.0
DEFAULT_HEARTBEAT_MINUTES = 5.0
IDLE_SLEEP_SEC = 0.2


def _orchestrator_alive() -> bool:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "MASTER_SYSTEM_ORCHESTRATOR.py"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return False


def _latest_trade_journal() -> Optional[Path]:
    logs_dir = Path("logs")
    files = sorted(logs_dir.glob("trade_journal_*.jsonl"))
    return files[-1] if files else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execution-focused orchestrator monitor with periodic checkpoints."
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=DEFAULT_DURATION_HOURS,
        help=f"How long to monitor (default: {DEFAULT_DURATION_HOURS}h).",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=float,
        default=DEFAULT_CHECKPOINT_MINUTES,
        help=f"Checkpoint interval in minutes (default: {DEFAULT_CHECKPOINT_MINUTES}m).",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=DEFAULT_HEARTBEAT_MINUTES,
        help=f"Heartbeat interval in minutes (default: {DEFAULT_HEARTBEAT_MINUTES}m).",
    )
    args = parser.parse_args(argv)

    duration_sec = max(60.0, float(args.duration_hours) * 3600.0)
    checkpoint_sec = max(60.0, float(args.checkpoint_minutes) * 60.0)
    heartbeat_sec = max(30.0, float(args.heartbeat_minutes) * 60.0)
    expected_checkpoints = max(1, int(math.ceil(duration_sec / checkpoint_sec)))

    agent_log = Path("logs/core/agent_manager.log")
    trade_journal = _latest_trade_journal()
    submit_re = re.compile(r"Submitted\s+(\d+)\s+TradeIntents\s+to\s+Meta")

    counts: Dict[str, int] = {
        "submitted_intents_total": 0,
        "submitted_batches": 0,
        "prepublish_dropped": 0,
        "order_submitted": 0,
        "order_filled": 0,
        "buy_filled": 0,
        "sell_filled": 0,
        "not_submitted_or_blocked": 0,
    }
    blockers: Counter[str] = Counter()
    checkpoint_counts: Dict[str, int] = counts.copy()
    checkpoint_blockers: Counter[str] = Counter()

    print("=" * 120, flush=True)
    print(f"SESSION MONITOR START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Agent log: {agent_log}", flush=True)
    print(f"Trade journal: {trade_journal if trade_journal else 'NONE'}", flush=True)
    print(
        f"Checkpoints: every {checkpoint_sec / 60.0:.1f} minutes for {duration_sec / 3600.0:.2f} hours "
        f"({expected_checkpoints} checkpoints expected)",
        flush=True,
    )
    print("=" * 120, flush=True)

    if not agent_log.exists():
        print(f"ERROR: missing {agent_log}", flush=True)
        return 1

    if trade_journal is None:
        print("WARNING: no trade journal found at startup; waiting for file to appear.", flush=True)

    with open(agent_log, "r", encoding="utf-8", errors="ignore") as agent_fp:
        agent_fp.seek(0, os.SEEK_END)

        trade_fp = None
        if trade_journal and trade_journal.exists():
            trade_fp = open(trade_journal, "r", encoding="utf-8", errors="ignore")
            trade_fp.seek(0, os.SEEK_END)

        start_ts = time.time()
        next_checkpoint = start_ts + checkpoint_sec
        next_heartbeat = start_ts + heartbeat_sec

        try:
            while (time.time() - start_ts) < duration_sec:
                progressed = False

                # AgentManager stream
                line = agent_fp.readline()
                while line:
                    progressed = True
                    if "AgentManager:PrePublishGate] Dropped intent" in line:
                        counts["prepublish_dropped"] += 1
                    m = submit_re.search(line)
                    if m:
                        counts["submitted_batches"] += 1
                        try:
                            counts["submitted_intents_total"] += int(m.group(1))
                        except Exception:
                            pass
                    line = agent_fp.readline()

                # Trade journal stream (auto-open if appears later)
                if trade_fp is None:
                    new_journal = _latest_trade_journal()
                    if new_journal and new_journal.exists():
                        trade_journal = new_journal
                        trade_fp = open(trade_journal, "r", encoding="utf-8", errors="ignore")
                        trade_fp.seek(0, os.SEEK_END)
                        print(
                            f"[INFO] Attached trade journal: {trade_journal} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            flush=True,
                        )

                if trade_fp is not None:
                    line = trade_fp.readline()
                    while line:
                        progressed = True
                        raw = line.strip()
                        if raw:
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                obj = None

                            if isinstance(obj, dict):
                                event = str(obj.get("event", "") or "")
                                side = str(obj.get("side", "") or "").upper()
                                status = str(obj.get("status", "") or "").upper()
                                reason = str(
                                    obj.get("reason", "") or obj.get("error_code", "") or ""
                                ).strip()

                                if event == "ORDER_SUBMITTED":
                                    counts["order_submitted"] += 1

                                if event == "ORDER_FILLED":
                                    counts["order_filled"] += 1
                                    if side == "BUY":
                                        counts["buy_filled"] += 1
                                    elif side == "SELL":
                                        counts["sell_filled"] += 1

                                blocked = ("NOT_SUBMITTED" in event) or (status == "BLOCKED")
                                if blocked:
                                    counts["not_submitted_or_blocked"] += 1
                                    blockers[reason or event or "BLOCKED_UNKNOWN"] += 1

                        line = trade_fp.readline()

                now = time.time()
                if now >= next_heartbeat:
                    elapsed_min = int((now - start_ts) // 60)
                    print(
                        f"[HEARTBEAT] elapsed={elapsed_min}m alive={_orchestrator_alive()} intents={counts['submitted_intents_total']} filled={counts['order_filled']}",
                        flush=True,
                    )
                    next_heartbeat += heartbeat_sec

                if now >= next_checkpoint:
                    cp_idx = int((now - start_ts) // checkpoint_sec)
                    delta = {
                        k: counts[k] - checkpoint_counts.get(k, 0)
                        for k in counts
                    }

                    delta_blockers = Counter(blockers)
                    delta_blockers.subtract(checkpoint_blockers)
                    top_total = blockers.most_common(5)
                    top_delta = [(k, v) for k, v in delta_blockers.most_common(5) if v > 0]

                    print("-" * 120, flush=True)
                    print(
                        f"CHECKPOINT {cp_idx:02d}/{expected_checkpoints:02d} | "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"orchestrator_alive={_orchestrator_alive()}",
                        flush=True,
                    )
                    print(
                        "TOTAL  "
                        f"intents={counts['submitted_intents_total']} batches={counts['submitted_batches']} "
                        f"prepublish_drop={counts['prepublish_dropped']} submitted={counts['order_submitted']} "
                        f"filled={counts['order_filled']} buy_filled={counts['buy_filled']} "
                        f"sell_filled={counts['sell_filled']} blocked={counts['not_submitted_or_blocked']}",
                        flush=True,
                    )
                    print(
                        "DELTA  "
                        f"intents={delta['submitted_intents_total']} batches={delta['submitted_batches']} "
                        f"prepublish_drop={delta['prepublish_dropped']} submitted={delta['order_submitted']} "
                        f"filled={delta['order_filled']} buy_filled={delta['buy_filled']} "
                        f"sell_filled={delta['sell_filled']} blocked={delta['not_submitted_or_blocked']}",
                        flush=True,
                    )
                    print(f"TOP_BLOCKERS_TOTAL {top_total if top_total else 'none'}", flush=True)
                    print(f"TOP_BLOCKERS_DELTA {top_delta if top_delta else 'none'}", flush=True)

                    checkpoint_counts = counts.copy()
                    checkpoint_blockers = Counter(blockers)
                    next_checkpoint += checkpoint_sec

                if not progressed:
                    time.sleep(IDLE_SLEEP_SEC)

        finally:
            if trade_fp is not None:
                trade_fp.close()

    print("=" * 120, flush=True)
    print(f"SESSION MONITOR END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("FINAL_SUMMARY", flush=True)
    for key, value in counts.items():
        print(f"  {key}={value}", flush=True)
    print("TOP_BLOCKERS_FINAL", flush=True)
    if blockers:
        for key, value in blockers.most_common(10):
            print(f"  {key}={value}", flush=True)
    else:
        print("  none", flush=True)
    print("=" * 120, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("[FATAL] monitor_6h_session crashed with unhandled exception:", flush=True)
        traceback.print_exc()
        raise
