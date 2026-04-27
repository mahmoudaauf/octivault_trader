#!/usr/bin/env python3
"""
Lightweight monitor: watches `automation/proposed_rules.json` and logs whenever
proposals change. Writes summary lines to `trading.log` and `automation/proposal_monitor.log`.

Run this in the background (nohup) if you want continuous monitoring. It's safe and read-only.
"""
import time
import os
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
PROPOSAL_PATH = os.path.join(ROOT, 'automation', 'proposed_rules.json')
TRADING_LOG = os.path.join(ROOT, 'trading.log')
MON_LOG = os.path.join(ROOT, 'automation', 'proposal_monitor.log')

def _write_mon(msg):
    line = f"{datetime.utcnow().isoformat()}Z {msg}\n"
    with open(MON_LOG, 'a') as f:
        f.write(line)

def _write_trading(msg):
    line = f"{datetime.utcnow().isoformat()}Z [PROPOSAL_MONITOR] {msg}\n"
    with open(TRADING_LOG, 'a') as f:
        f.write(line)

def load(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def summarize(payload):
    if not isinstance(payload, dict):
        return 'invalid_payload'
    props = payload.get('proposals') or []
    lines = []
    for p in props:
        lines.append(f"{p.get('symbol')} cur={p.get('current_required_conf')} sug={p.get('suggested_required_conf')}")
    return '; '.join(lines)

def main(poll_sec=5, run_once=False):
    last_mtime = 0
    last_summary = ''
    while True:
        if os.path.exists(PROPOSAL_PATH):
            mtime = os.path.getmtime(PROPOSAL_PATH)
            if mtime != last_mtime:
                payload = load(PROPOSAL_PATH)
                summary = summarize(payload)
                msg = f'PROPOSAL_CHANGED: {summary}'
                _write_mon(msg)
                _write_trading(msg)
                last_mtime = mtime
                last_summary = summary
        else:
            if last_summary != 'MISSING':
                _write_mon('PROPOSAL_MISSING')
                _write_trading('PROPOSAL_MISSING')
                last_summary = 'MISSING'

        if run_once:
            break
        time.sleep(poll_sec)

if __name__ == '__main__':
    main(poll_sec=2, run_once=True)
