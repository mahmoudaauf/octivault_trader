#!/usr/bin/env python3
"""
Diagnostic: per-loop, per-symbol gate table for recent loops.
"""
import os
import re
import argparse
import json
import csv
from collections import defaultdict
from datetime import datetime

LOG = 'trading.log'
TAIL_BYTES = 300000  # read last ~300KB
MAX_LOOPS = 50

# regex
RE_LOOP = re.compile(r'LOOP_SUMMARY\].*?loop_id=(\d+).*?capital_free=([0-9]+\.?[0-9]*).*?exec_attempted=(True|False).*?exec_result=([A-Z_]+).*?rej_count=(\d+)', re.I | re.S)
RE_GATED = re.compile(r'GateDebug:TRADEABILITY\].*?([A-Z0-9]+)\s+(BUY|SELL).*?signal_conf=([0-9]*\.?[0-9]+).*?required_conf=([0-9]*\.?[0-9]+).*?result=([A-Z]+)', re.I | re.S)
RE_EXEC_REJECT = re.compile(r'EXEC_REJECT\].*symbol=([A-Z0-9]+).*count=(\d+)', re.I)
RE_EXEC_REJECT_ALT = re.compile(r'SharedState - \[\s*EXEC_REJECT\s*\].*symbol=([A-Z0-9]+).*count=(\d+)', re.I)
RE_SKIP_REJECT = re.compile(r'Skipping\s+([A-Z0-9]+)\s+(?:BUY|SELL):\s*rejected\s*(\d+)', re.I)
RE_SKIP_SHORT = re.compile(r'Skipping\s+([A-Z0-9]+)\s+(BUY|SELL)\b', re.I)


def tail_bytes(path, n):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print('Error reading log:', e)
        return ''


def parse(max_loops=MAX_LOOPS):
    text = tail_bytes(LOG, TAIL_BYTES)
    if not text:
        print('No log content')
        return []
    # Find all LOOP_SUMMARY matches and extract preceding text as raw events for that loop
    loops = []
    matches = list(RE_LOOP.finditer(text))
    if not matches:
        print('No LOOP_SUMMARY found in recent log window')
        return []

    for i, m in enumerate(matches):
        loop_id = int(m.group(1))
        capital_free = float(m.group(2))
        exec_attempted = m.group(3) == 'True'
        exec_result = m.group(4)
        rej_count = int(m.group(5))
        start_idx = matches[i-1].end() if i > 0 else 0
        raw_text = text[start_idx:m.start()]
        loops.append({
            'loop_id': loop_id,
            'capital_free': capital_free,
            'exec_attempted': exec_attempted,
            'exec_result': exec_result,
            'rej_count': rej_count,
            'raw_events': raw_text.splitlines()
        })

    if not loops:
        print('No LOOP_SUMMARY found in recent log window')
        return []

    # take last max_loops
    loops = loops[-max_loops:]

    out = []
    # analyze each loop
    for loop in loops:
        per_symbol = defaultdict(lambda: {'signal_conf': None, 'required_conf': None, 'result': None, 'exec_rejects': 0, 'skip_rejections':0})
        # scan raw_events for gatedebug and exec_reject
        for ev in loop['raw_events']:
            gm = RE_GATED.search(ev)
            if gm:
                sym = gm.group(1)
                side = gm.group(2)
                try:
                    sig = float(gm.group(3))
                except Exception:
                    sig = None
                try:
                    req = float(gm.group(4))
                except Exception:
                    req = None
                res = gm.group(5)
                per_symbol[sym]['signal_conf'] = sig
                per_symbol[sym]['required_conf'] = req
                per_symbol[sym]['result'] = res
            em = RE_EXEC_REJECT.search(ev) or RE_EXEC_REJECT_ALT.search(ev)
            if em:
                sym = em.group(1)
                cnt = int(em.group(2))
                per_symbol[sym]['exec_rejects'] += cnt
            sm = RE_SKIP_REJECT.search(ev)
            if sm:
                sym = sm.group(1)
                cnt = int(sm.group(2) or 1)
                per_symbol[sym]['skip_rejections'] += cnt

        # build output structure for this loop
        symbols = {}
        for sym, info in per_symbol.items():
            symbols[sym] = {
                'signal_conf': info['signal_conf'],
                'required_conf': info['required_conf'],
                'result': info['result'],
                'exec_rejects': info['exec_rejects'],
                'skip_rejections': info['skip_rejections']
            }
        out.append({
            'loop_id': loop['loop_id'],
            'capital_free': loop['capital_free'],
            'exec_attempted': loop['exec_attempted'],
            'exec_result': loop['exec_result'],
            'rej_count': loop['rej_count'],
            'symbols': symbols
        })
    return out


def pretty_print(out):
    for loop in out:
        print('\n' + '='*80)
        print(f"Loop {loop['loop_id']}: capital_free={loop['capital_free']:.2f} exec_attempted={loop['exec_attempted']} exec_result={loop['exec_result']} rej_count={loop['rej_count']}")
        if not loop['symbols']:
            print('  (no GateDebug or exec_reject events captured in this loop)')
            continue
        print(f"  {'SYMBOL':<10} {'sig_conf':>8} {'req_conf':>8} {'result':>8} {'exec_rej':>8} {'skip_rej':>8}")
        for sym, info in sorted(loop['symbols'].items(), key=lambda x: x[1].get('exec_rejects',0), reverse=True):
            sig = f"{info['signal_conf']:.3f}" if info['signal_conf'] is not None else '-'
            req = f"{info['required_conf']:.3f}" if info['required_conf'] is not None else '-'
            res = info['result'] or '-'
            ej = info['exec_rejects']
            sj = info['skip_rejections']
            print(f"  {sym:<10} {sig:>8} {req:>8} {res:>8} {ej:>8} {sj:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-loop, per-symbol diag')
    parser.add_argument('--loops', type=int, default=50, help='Number of recent loops to include')
    parser.add_argument('--out-json', type=str, default=None, help='Write output JSON to path')
    parser.add_argument('--out-csv', type=str, default=None, help='Write output CSV to path')
    args = parser.parse_args()

    if not os.path.exists(LOG):
        print('trading.log not found in cwd')
        raise SystemExit(1)

    out = parse(max_loops=args.loops)
    if args.out_json:
        try:
            with open(args.out_json, 'w') as f:
                json.dump(out, f, indent=2)
            print('Wrote JSON to', args.out_json)
        except Exception as e:
            print('Failed to write JSON:', e)
    if args.out_csv:
        try:
            # flat rows: loop_id,symbol,signal_conf,required_conf,result,exec_rejects,skip_rejections,capital_free,exec_attempted,exec_result,rej_count
            with open(args.out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['loop_id','symbol','signal_conf','required_conf','result','exec_rejects','skip_rejections','capital_free','exec_attempted','exec_result','rej_count'])
                for loop in out:
                    for sym, info in loop['symbols'].items():
                        writer.writerow([
                            loop['loop_id'], sym, info['signal_conf'], info['required_conf'], info['result'], info['exec_rejects'], info['skip_rejections'], loop['capital_free'], loop['exec_attempted'], loop['exec_result'], loop['rej_count']
                        ])
            print('Wrote CSV to', args.out_csv)
        except Exception as e:
            print('Failed to write CSV:', e)

    # if no outputs requested, pretty print
    if not args.out_json and not args.out_csv:
        pretty_print(out)
