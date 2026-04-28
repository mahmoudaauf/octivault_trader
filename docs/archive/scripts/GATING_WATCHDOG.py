#!/usr/bin/env python3
"""
GATING_WATCHDOG.py
Watches trading.log for gating/filter conflicts, high rejection counts, and strategy signals.
Appends alerts to gating_alerts.log and prints concise alerts to stdout.
"""

import os
import time
import re
from collections import defaultdict, deque
from datetime import datetime

LOG = "trading.log"
ALERT_LOG = "gating_alerts.log"

# thresholds
REJECTION_ALERT_THRESHOLD = 50        # per-symbol rejections (absolute) to alert
RECENT_WINDOW_SECONDS = 300          # time window to track recent events
NO_SIGNALS_RATE_THRESHOLD = 5        # "NO SIGNALS" events within window to alert

# patterns
PAT_SKIP_REJECT = re.compile(r"Skipping\s+([A-Z0-9]+)\s+(?:BUY|SELL):\s*rejected\s+(\d+)", re.I)
PAT_SKIP_SHORT = re.compile(r"Skipping\s+([A-Z0-9]+)\s+(?:BUY|SELL)\b.*rejections", re.I)
PAT_NO_SIGNALS = re.compile(r"NO SIGNALS PASSED FILTERS|NO SIGNALS PASSED", re.I)
PAT_EXEC_REJECT = re.compile(r"EXEC_REJECT|MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD", re.I)
PAT_BYPASS = re.compile(r"Bypassing min-hold", re.I)
PAT_OVERRIDDEN = re.compile(r"OVERRIDDEN", re.I)
PAT_SIGNAL = re.compile(r"Received signal for\s+([A-Z0-9]+)\s+from\s+([A-Za-z0-9_]+)|Published TradeIntent:\s*([A-Z0-9]+)\s+(BUY|SELL)", re.I)
PAT_FLAT = re.compile(r"FLAT_PORTFOLIO", re.I)


def tail_f(filepath):
    """Generator that yields new lines as they are written to the file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.25)
                continue
            yield line


def now_ts():
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def alert(msg):
    timestamped = f"{now_ts()} ALERT: {msg}\n"
    print(timestamped, end="")
    with open(ALERT_LOG, 'a', encoding='utf-8') as a:
        a.write(timestamped)


def run_watchdog():
    if not os.path.exists(LOG):
        print(f"Error: {LOG} not found")
        return

    # per-symbol rejection counters (total and recent window)
    total_rejections = defaultdict(int)
    recent_rejections = defaultdict(deque)  # symbol -> deque of (ts, count)

    # recent NO SIGNALS events timestamps
    recent_no_signals = deque()

    # strategy counts
    strategy_counts = defaultdict(int)

    g = tail_f(LOG)
    for line in g:
        ts = time.time()
        # check patterns
        if PAT_SKIP_REJECT.search(line):
            m = PAT_SKIP_REJECT.search(line)
            sym = m.group(1)
            cnt = int(m.group(2)) if m.group(2).isdigit() else 1
            total_rejections[sym] += cnt
            recent_rejections[sym].append((ts, cnt))
            # prune old
            while recent_rejections[sym] and ts - recent_rejections[sym][0][0] > RECENT_WINDOW_SECONDS:
                recent_rejections[sym].popleft()
            recent_sum = sum(c for _, c in recent_rejections[sym])
            if total_rejections[sym] >= REJECTION_ALERT_THRESHOLD or recent_sum >= REJECTION_ALERT_THRESHOLD:
                alert(f"High rejection count for {sym}: total={total_rejections[sym]} recent({RECENT_WINDOW_SECONDS}s)={recent_sum}")

        elif PAT_SKIP_SHORT.search(line):
            m = PAT_SKIP_SHORT.search(line)
            sym = m.group(1)
            total_rejections[sym] += 1
            recent_rejections[sym].append((ts, 1))
            while recent_rejections[sym] and ts - recent_rejections[sym][0][0] > RECENT_WINDOW_SECONDS:
                recent_rejections[sym].popleft()
            recent_sum = sum(c for _, c in recent_rejections[sym])
            if total_rejections[sym] >= REJECTION_ALERT_THRESHOLD or recent_sum >= REJECTION_ALERT_THRESHOLD:
                alert(f"High rejection count for {sym}: total={total_rejections[sym]} recent({RECENT_WINDOW_SECONDS}s)={recent_sum}")

        if PAT_NO_SIGNALS.search(line):
            recent_no_signals.append(ts)
            # prune
            while recent_no_signals and ts - recent_no_signals[0] > RECENT_WINDOW_SECONDS:
                recent_no_signals.popleft()
            if len(recent_no_signals) >= NO_SIGNALS_RATE_THRESHOLD:
                alert(f"NO SIGNALS events high: {len(recent_no_signals)} occurrences in last {RECENT_WINDOW_SECONDS}s")

        if PAT_EXEC_REJECT.search(line):
            alert(f"Execution rejection detected: {line.strip()}")

        if PAT_BYPASS.search(line):
            alert(f"Recovery bypass event: {line.strip()}")

        if PAT_OVERRIDDEN.search(line):
            alert(f"Forced rotation override: {line.strip()}")

        if PAT_SIGNAL.search(line):
            m = PAT_SIGNAL.search(line)
            if m:
                if m.group(2):
                    strategy = m.group(2)
                    strategy_counts[strategy] += 1
                elif m.group(3):
                    # Published TradeIntent matched
                    strategy_counts['PublishedTradeIntent'] += 1

        if PAT_FLAT.search(line):
            alert("FLAT_PORTFOLIO detected — buy-only logic enforced")

        # Periodic summary every N seconds
        # We'll print a light-weight summary inline every 60 seconds
        try:
            if int(ts) % 60 == 0:
                top_rej = sorted(total_rejections.items(), key=lambda x: x[1], reverse=True)[:5]
                summary = f"SUMMARY top_rejections: {top_rej} | strategies: {dict(strategy_counts)}"
                with open(ALERT_LOG, 'a', encoding='utf-8') as a:
                    a.write(f"{now_ts()} SUMMARY: {summary}\n")
        except Exception:
            pass


if __name__ == '__main__':
    print(f"Starting gating watchdog — logging alerts to {ALERT_LOG}")
    run_watchdog()
