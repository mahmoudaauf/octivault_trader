#!/usr/bin/env python3
"""Scheduled public-data paper cycle and read-only profitability report."""
from __future__ import annotations

import asyncio
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from capital_paper import cycle
from profit_readiness import OPERATOR_TARGET_HOURLY_USD


def main():
    root = Path(__file__).resolve().parent
    output = root / "logs/profit_readiness"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "research.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            result = asyncio.run(asyncio.wait_for(cycle(output / "forward_paper.json"), timeout=180))
            print(f"Paper NAV=${result['latest_nav']:.4f}; trades={len(result['closed_trades'])}", flush=True)
        except Exception as exc:
            # Keep reporting even when a market read fails; don't expose URLs
            # or fabricate a fresh paper valuation from a failed fetch.
            print(f"Paper cycle failed: {type(exc).__name__}; state not advanced", flush=True)
        cache = output / "exchange_fills.json"
        refreshed = 0.0
        if cache.exists():
            refreshed = datetime.fromisoformat(json.loads(cache.read_text())["fetched_at"]).timestamp()
        command = [sys.executable, str(root / "profit_readiness.py"),
                   "--target-hourly", f"{OPERATOR_TARGET_HOURLY_USD:.6f}"]
        if time.time() - refreshed >= 3600:
            command.append("--refresh-exchange")
        return subprocess.run(command, cwd=root, timeout=240).returncode


if __name__ == "__main__":
    raise SystemExit(main())
