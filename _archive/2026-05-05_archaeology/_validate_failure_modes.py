#!/usr/bin/env python3
"""Validate failure-mode handling: stop-loss, take-profit, time-decay logic."""
import json
from collections import Counter
from pathlib import Path

JOURNAL = Path("logs/trade_journal_20260504.jsonl")

events = Counter()
tags = Counter()
sl_events = []
tp_events = []
buys = []
sells = []

with open(JOURNAL) as f:
    for line in f:
        try:
            d = json.loads(line)
            ev = d.get("event", "")
            events[ev] += 1
            tag = str(d.get("tag", ""))
            if tag:
                tags[tag] += 1
            tl = tag.lower()
            el = ev.lower()
            if "stop" in el or "stop_loss" in tl or "_sl" in tl or tl.endswith("sl"):
                sl_events.append(d)
            if "tp" in tl or "take_profit" in tl:
                tp_events.append(d)
            if ev == "ORDER_FILLED":
                if d.get("side") == "BUY":
                    buys.append(d)
                elif d.get("side") == "SELL":
                    sells.append(d)
        except Exception:
            pass

print("=" * 70)
print("FAILURE MODE VALIDATION — Trade Journal Analysis")
print("=" * 70)

print("\n📊 EVENT TYPES (top 15):")
for e, c in events.most_common(15):
    print(f"   {c:4d}  {e}")

print("\n🏷️  TAGS (all):")
for t, c in tags.most_common():
    print(f"   {c:4d}  {t}")

print(f"\n🔴 STOP-LOSS RELATED EVENTS: {len(sl_events)}")
for e in sl_events[-8:]:
    print(
        f"   {e.get('ts','')[:19]} {e.get('event','')} | {e.get('symbol','')} | tag={e.get('tag','')}"
    )

print(f"\n🎯 TAKE-PROFIT RELATED EVENTS: {len(tp_events)}")
for e in tp_events[-8:]:
    print(
        f"   {e.get('ts','')[:19]} {e.get('event','')} | {e.get('symbol','')} | tag={e.get('tag','')}"
    )

print("\n🟢 LAST 5 BUYS:")
for e in buys[-5:]:
    print(
        f"   {e.get('ts','')[:19]} {e.get('symbol',''):<10} qty={e.get('executed_qty')} @ ${e.get('avg_price')} tag={e.get('tag')}"
    )

print("\n🔴 LAST 5 SELLS:")
for e in sells[-5:]:
    print(
        f"   {e.get('ts','')[:19]} {e.get('symbol',''):<10} qty={e.get('executed_qty')} @ ${e.get('avg_price')} tag={e.get('tag')}"
    )

# Determine current positions (BUY without matching SELL today for symbol)
print("\n📈 ACTIVE POSITIONS (today's buys not yet sold):")
for sym in ["ETHUSDT", "SOLUSDT", "XRPUSDT"]:
    sym_buys = [b for b in buys if b.get("symbol") == sym]
    sym_sells = [s for s in sells if s.get("symbol") == sym]
    print(f"   {sym}: {len(sym_buys)} buys, {len(sym_sells)} sells")
    for b in sym_buys[-2:]:
        print(
            f"      BUY  {b.get('ts','')[:19]} qty={b.get('executed_qty')} @ ${b.get('avg_price')} tag={b.get('tag')}"
        )
    for s in sym_sells[-2:]:
        print(
            f"      SELL {s.get('ts','')[:19]} qty={s.get('executed_qty')} @ ${s.get('avg_price')} tag={s.get('tag')}"
        )
