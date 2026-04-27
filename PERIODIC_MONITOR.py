#!/usr/bin/env python3
"""
PERIODIC_MONITOR.py
Writes a concise monitoring summary to MONITOR_SUMMARY.log every 10 minutes.
Also prints a short summary to stdout on each run.
"""
import os
import time
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

LOG = 'trading.log'
ALERT_LOG = 'gating_alerts.log'
SUMMARY_LOG = 'MONITOR_SUMMARY.log'

REFRESH_SECONDS = 600  # 10 minutes
TAIL_CHARS = 200000  # read last ~200KB of log to limit parsing

TS_RE = re.compile(r'^\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),')

KEY_PATTERNS = {
    'bypass': re.compile(r'Bypassing min-hold', re.I),
    'overridden': re.compile(r'OVERRIDDEN', re.I),
    'exec_reject': re.compile(r'EXEC_REJECT|MICRO_BACKTEST_WIN_RATE_BELOW_THRESHOLD', re.I),
    'no_signals': re.compile(r'NO SIGNALS PASSED FILTERS|NO SIGNALS PASSED', re.I),
    'entry_size': re.compile(r'ENTRY_SIZE_ENFORCEMENT', re.I),
    'flat_portfolio': re.compile(r'FLAT_PORTFOLIO', re.I)
}

SYM_REJ_RE = re.compile(r'Skipping\s+([A-Z0-9]+)\s+(?:BUY|SELL):\s*rejected\s+(\d+)', re.I)
SIG_RE = re.compile(r'Received signal for\s+([A-Z0-9]+)\s+from\s+([A-Za-z0-9_]+)|Published TradeIntent:\s*([A-Z0-9]+)\s+(BUY|SELL)', re.I)


def tail_bytes(path, num_bytes):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - num_bytes))
            data = f.read().decode('utf-8', errors='ignore')
            return data
    except Exception:
        return ''


def parse_recent(text, cutoff_dt):
    lines = text.splitlines()
    counts = Counter()
    rej_counts = Counter()
    strategy = Counter()
    for line in lines:
        # try extract timestamp
        m = TS_RE.match(line)
        if not m:
            # include if cannot parse ts (conservative)
            consider = True
        else:
            ts_str = m.group(1)
            try:
                ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            except:
                ts = None
            consider = (ts is None) or (ts >= cutoff_dt)
        if not consider:
            continue
        for k,pat in KEY_PATTERNS.items():
            if pat.search(line):
                counts[k]+=1
        m = SYM_REJ_RE.search(line)
        if m:
            sym=m.group(1)
            cnt=int(m.group(2) or 1)
            rej_counts[sym]+=cnt
        m2 = SIG_RE.search(line)
        if m2:
            if m2.group(2):
                strategy[m2.group(2)]+=1
            else:
                strategy['PublishedIntent']+=1
    return counts, rej_counts, strategy


def make_summary():
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=REFRESH_SECONDS)
    text = ''
    if os.path.exists(LOG):
        text = tail_bytes(LOG, TAIL_CHARS)
    else:
        text = ''
    counts, rej_counts, strategy = parse_recent(text, cutoff)

    # also collect alerts in alert log last refresh window
    alert_snip = ''
    if os.path.exists(ALERT_LOG):
        alert_snip = tail_bytes(ALERT_LOG, 20000)

    top_rej = rej_counts.most_common(5)

    summary = {
        'ts': now.strftime('%Y-%m-%d %H:%M:%S'),
        'window_minutes': REFRESH_SECONDS//60,
        'counts': counts,
        'top_rejections': top_rej,
        'strategy_counts': dict(strategy),
        'alerts_snip': (alert_snip.splitlines()[-10:] if alert_snip else [])
    }

    # write to SUMMARY_LOG
    with open(SUMMARY_LOG, 'a', encoding='utf-8') as f:
        f.write(f"=== {summary['ts']} (last {summary['window_minutes']}m) ===\n")
        f.write(f"counts: {summary['counts']}\n")
        f.write(f"top_rejections: {summary['top_rejections']}\n")
        f.write(f"strategy_counts: {summary['strategy_counts']}\n")
        if summary['alerts_snip']:
            f.write('alerts_recent:\n')
            for a in summary['alerts_snip']:
                f.write('  '+a+"\n")
        f.write('\n')

    # also print a compact summary to stdout
    print(f"{summary['ts']} summary - counts={dict(summary['counts'])} top_rejections={summary['top_rejections']} strategies={summary['strategy_counts']}")

    return summary


if __name__ == '__main__':
    print(f"Starting PERIODIC_MONITOR, writing summaries to {SUMMARY_LOG} every {REFRESH_SECONDS}s")
    try:
        while True:
            make_summary()
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print('Stopped by user')
