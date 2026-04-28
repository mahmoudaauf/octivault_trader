#!/usr/bin/env python3
"""Show all detected symbols from the latest run log."""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/octivault_live_run_10.log"

with open(LOG) as f:
    lines = f.readlines()

# Dedupe: keep last seen value per symbol from CLASSIFY events
latest = {}
for l in lines:
    m = re.search(
        r"CLASSIFY\] (\w+) qty=([0-9.]+) value=([0-9.]+) floor=([0-9.]+) latest_price=([0-9.e\-+]+)",
        l,
    )
    if m:
        sym, qty, val, floor, px = m.groups()
        latest[sym] = (float(qty), float(val), float(floor), float(px))

rows = sorted(latest.items(), key=lambda kv: kv[1][1], reverse=True)
ACTIVE, DUST, ZERO = [], [], []
for sym, (q, v, fl, px) in rows:
    if v == 0:
        ZERO.append((sym, q, v, px))
    elif v >= fl:
        ACTIVE.append((sym, q, v, px))
    else:
        DUST.append((sym, q, v, px))

print()
print("=" * 74)
print("  OCTIVAULT — DETECTED SYMBOLS (run #10, latest snapshot)")
print("=" * 74)
print(f"  Total symbols with non-zero qty: {len(rows)}")
print(f"  Active positions (>= $25 floor): {len(ACTIVE)}")
print(f"  Dust positions   (< $25):        {len(DUST)}")
print(f"  Zero-value/unpriced:             {len(ZERO)}")
print("=" * 74)
print()

print(f"--- ACTIVE [{len(ACTIVE)}] ---")
print(f"  {'SYMBOL':<14} {'QTY':>16} {'PRICE':>14} {'VALUE':>10}")
for s, q, v, p in ACTIVE:
    print(f"  {s:<14} {q:>16.6f} {p:>14.4f}  ${v:>8.2f}")
print()

print(f"--- DUST [{len(DUST)}] (sorted by value desc) ---")
print(f"  {'SYMBOL':<14} {'QTY':>16} {'PRICE':>14} {'VALUE':>10}")
for s, q, v, p in DUST:
    print(f"  {s:<14} {q:>16.6f} {p:>14.6f}  ${v:>8.4f}")
print()

print(f"--- ZERO/UNPRICED [{len(ZERO)}] ---")
for s, q, v, p in ZERO:
    print(f"  {s:<14} qty={q:.6f}  px={p}")
print()

print("--- AGGREGATES ---")
print(f"  Active value:  ${sum(v for _, _, v, _ in ACTIVE):>10.2f}")
print(f"  Dust value:    ${sum(v for _, _, v, _ in DUST):>10.2f}  ({len(DUST)} fragmented symbols)")
print(f"  Total invested:${sum(v for _, _, v, _ in rows):>10.2f}")
print()
