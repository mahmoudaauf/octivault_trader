#!/usr/bin/env python3
"""
Minimal automated rule proposer.

Reads diagnostics/50loops.json (produced by diagnostics/per_loop_symbol_diag.py),
scores symbols by recent signal strength vs required_conf and rejection counts,
and proposes a small relaxation of required_conf for top-k candidates.

This is non-invasive: it only writes a JSON proposal file for human review.
"""
import json
import statistics
from collections import defaultdict
import argparse
import os

DEFAULT_INPUT = 'diagnostics/50loops.json'
DEFAULT_OUT = 'automation/proposed_rules.json'


def load_diagnostics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'diagnostics file not found: {path}')
    with open(path, 'r') as f:
        return json.load(f)


def score_symbols(data):
    # aggregate stats per symbol
    stats = defaultdict(lambda: {
        'n_signals': 0,
        'n_pass': 0,
        'deltas': [],
        'exec_rejects': 0,
        'skip_rejections': 0,
        'last_required_conf': None,
    })

    for loop in data:
        for sym, info in loop.get('symbols', {}).items():
            sig = info.get('signal_conf')
            req = info.get('required_conf')
            if sig is not None:
                stats[sym]['n_signals'] += 1
                if req is not None:
                    stats[sym]['deltas'].append(sig - req)
                    if sig >= req:
                        stats[sym]['n_pass'] += 1
                else:
                    # no required_conf observed, still count the signal
                    stats[sym]['deltas'].append(0.0)
            stats[sym]['exec_rejects'] += int(info.get('exec_rejects') or 0)
            stats[sym]['skip_rejections'] += int(info.get('skip_rejections') or 0)
            if req is not None:
                stats[sym]['last_required_conf'] = req

    # compute score
    scored = []
    for sym, s in stats.items():
        n = s['n_signals']
        if n == 0:
            continue
        avg_delta = statistics.mean(s['deltas']) if s['deltas'] else 0.0
        pass_rate = s['n_pass'] / n if n else 0.0
        rej_penalty = (s['exec_rejects'] + s['skip_rejections'])
        # heuristic score: favor pass_rate and avg_delta, penalize rejections
        score = pass_rate * 2.0 + avg_delta * 3.0 - rej_penalty * 0.01
        scored.append((sym, score, s))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def propose_rules(scored, top_k=5, relax_delta=0.05):
    proposals = []
    for sym, score, s in scored[:top_k]:
        cur_req = s.get('last_required_conf')
        if cur_req is None:
            suggested = max(0.0, 0.5 - relax_delta)  # fallback
        else:
            suggested = max(0.0, cur_req - relax_delta)
        proposals.append({
            'symbol': sym,
            'score': score,
            'n_signals': s['n_signals'],
            'n_pass': s['n_pass'],
            'avg_delta': statistics.mean(s['deltas']) if s['deltas'] else 0.0,
            'exec_rejects': s['exec_rejects'],
            'skip_rejections': s['skip_rejections'],
            'current_required_conf': cur_req,
            'suggested_required_conf': round(suggested, 3),
            'rationale': f"High pass_rate + avg_delta; lowish rejection penalty. Relax required_conf by {relax_delta}"
        })
    return proposals


def main():
    parser = argparse.ArgumentParser(description='Auto rule proposer — non-invasive proposals from diagnostics')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.05, help='how much to relax required_conf')
    parser.add_argument('--out', default=DEFAULT_OUT)
    args = parser.parse_args()

    data = load_diagnostics(args.input)
    scored = score_symbols(data)
    proposals = propose_rules(scored, top_k=args.top, relax_delta=args.delta)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'generated_at': __import__('datetime').datetime.utcnow().isoformat() + 'Z', 'proposals': proposals}, f, indent=2)

    print(f'Wrote proposals to {args.out}')
    for p in proposals:
        print(f"{p['symbol']}: current={p['current_required_conf']} -> suggested={p['suggested_required_conf']}  (score={p['score']:.3f})")


if __name__ == '__main__':
    main()
