#!/usr/bin/env python3
"""
Next-level TP/SL diagnostics from runtime logs.

Outputs:
1) SL width optimality
2) Regime classification accuracy
3) RR curve vs volatility
4) Position sizing efficiency
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Dict, List, Optional, Tuple


VOL_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+-\s+INFO\s+-\s+🎯 VOL-ADAPTIVE TP/SL\s+"
    r"(?P<symbol>[A-Z0-9]+):.*?"
    r"ATR%=(?P<atr_pct>\d+(?:\.\d+)?)\s+RV%=(?P<rv_pct>\d+(?:\.\d+)?)\s+regime=(?P<regime>[a-z_]+).*?"
    r"RR target=(?P<rr_target>\d+(?:\.\d+)?)\s+final=(?P<rr_final>\d+(?:\.\d+)?)\s+\|\s+"
    r"SL%=(?P<sl_pct>\d+(?:\.\d+)?)\s+TP%=(?P<tp_pct>\d+(?:\.\d+)?)"
    r".*?RiskSize=(?P<risk_size>\d+(?:\.\d+)?)"
)

EXEC_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+-\s+INFO\s+-\s+Execution Event:\s+"
    r".*?\"event\":\s*\"TRADE_EXECUTED\".*?\"symbol\":\s*\"(?P<symbol>[A-Z0-9]+)\""
    r".*?\"side\":\s*\"(?P<side>BUY|SELL)\".*?"
    r"(?:\"cummulative_quote\":\s*(?P<cumq>\d+(?:\.\d+)?)|\"cumulative_quote\":\s*(?P<cumq2>\d+(?:\.\d+)?))"
)


@dataclass
class VolRow:
    ts: datetime
    symbol: str
    atr_pct: float
    rv_pct: float
    regime: str
    rr_target: float
    rr_final: float
    sl_pct: float
    tp_pct: float
    risk_size: float


@dataclass
class ExecRow:
    ts: datetime
    symbol: str
    side: str
    cquote: float


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _expected_regime(atr_pct_ratio: float, low: float, high: float) -> str:
    # atr_pct_ratio is ratio (0.01 == 1.0%)
    if atr_pct_ratio >= high:
        return "high_vol"
    if atr_pct_ratio <= low:
        return "sideways"
    return "trend"


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def parse_log(path: str) -> Tuple[List[VolRow], List[ExecRow]]:
    vols: List[VolRow] = []
    execs: List[ExecRow] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mv = VOL_RE.search(line)
            if mv:
                vols.append(
                    VolRow(
                        ts=_parse_ts(mv.group("ts")),
                        symbol=mv.group("symbol"),
                        atr_pct=float(mv.group("atr_pct")),  # percent units (1.00 == 1%)
                        rv_pct=float(mv.group("rv_pct")),    # percent units
                        regime=mv.group("regime"),
                        rr_target=float(mv.group("rr_target")),
                        rr_final=float(mv.group("rr_final")),
                        sl_pct=float(mv.group("sl_pct")),    # percent units
                        tp_pct=float(mv.group("tp_pct")),    # percent units
                        risk_size=float(mv.group("risk_size")),
                    )
                )
                continue

            me = EXEC_RE.search(line)
            if me:
                cq = me.group("cumq") or me.group("cumq2")
                if cq is None:
                    continue
                execs.append(
                    ExecRow(
                        ts=_parse_ts(me.group("ts")),
                        symbol=me.group("symbol"),
                        side=me.group("side"),
                        cquote=float(cq),
                    )
                )
    return vols, execs


def analyze(
    vols: List[VolRow],
    execs: List[ExecRow],
    low_atr_ratio: float,
    high_atr_ratio: float,
    sl_noise_floor_mult: float,
    match_window_sec: int,
) -> str:
    if not vols:
        return "No VOL-ADAPTIVE TP/SL rows found."

    out: List[str] = []
    out.append(f"Rows parsed: vol={len(vols)} exec={len(execs)}")

    # 1) SL width optimality
    sl_over_atr = []
    noise_violations = 0
    for r in vols:
        # convert percent units -> ratio units
        atr_r = r.atr_pct / 100.0
        sl_r = r.sl_pct / 100.0
        if atr_r > 0:
            ratio = sl_r / atr_r
            sl_over_atr.append(ratio)
            if sl_r < (sl_noise_floor_mult * atr_r):
                noise_violations += 1
    out.append("")
    out.append("1) SL width optimality")
    out.append(
        f"- mean(SL/ATR)={mean(sl_over_atr):.3f} "
        f"min={min(sl_over_atr):.3f} max={max(sl_over_atr):.3f}"
        if sl_over_atr
        else "- insufficient data"
    )
    out.append(
        f"- noise-floor violations (SL < {sl_noise_floor_mult:.2f} * ATR): "
        f"{noise_violations}/{len(vols)} ({(100*noise_violations/len(vols)):.1f}%)"
    )

    # 2) Regime accuracy
    correct = 0
    expected_counts: Dict[str, int] = defaultdict(int)
    actual_counts: Dict[str, int] = defaultdict(int)
    for r in vols:
        exp = _expected_regime(r.atr_pct / 100.0, low_atr_ratio, high_atr_ratio)
        expected_counts[exp] += 1
        actual_counts[r.regime] += 1
        if exp == r.regime:
            correct += 1
    out.append("")
    out.append("2) Regime accuracy")
    out.append(
        f"- expected-vs-logged accuracy={correct}/{len(vols)} ({(100*correct/len(vols)):.1f}%) "
        f"using low={low_atr_ratio:.4f}, high={high_atr_ratio:.4f}"
    )
    out.append(f"- expected counts: {dict(expected_counts)}")
    out.append(f"- actual counts:   {dict(actual_counts)}")

    # 3) RR curve vs volatility
    atr_vals = [r.atr_pct for r in vols]
    rr_t = [r.rr_target for r in vols]
    rr_f = [r.rr_final for r in vols]
    corr_t = _pearson(atr_vals, rr_t)
    corr_f = _pearson(atr_vals, rr_f)
    out.append("")
    out.append("3) RR curve vs volatility")
    out.append(f"- corr(ATR%, RR target)={corr_t:.3f}")
    out.append(f"- corr(ATR%, RR final)={corr_f:.3f}")

    bins = [(0.0, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 99.0)]
    out.append("- binned means (ATR% bin -> RR target / RR final / rows):")
    for lo, hi in bins:
        rows = [r for r in vols if lo <= r.atr_pct < hi]
        if not rows:
            continue
        out.append(
            f"  {lo:.1f}-{hi:.1f}% -> {mean([r.rr_target for r in rows]):.3f} / "
            f"{mean([r.rr_final for r in rows]):.3f} / n={len(rows)}"
        )

    # 4) Position sizing efficiency
    # Match nearest BUY execution by symbol/time against VOL risk_size.
    by_symbol_exec: Dict[str, List[ExecRow]] = defaultdict(list)
    for e in execs:
        if e.side == "BUY":
            by_symbol_exec[e.symbol].append(e)
    for sym in by_symbol_exec:
        by_symbol_exec[sym].sort(key=lambda x: x.ts)

    effs = []
    matched = 0
    for r in vols:
        cands = by_symbol_exec.get(r.symbol, [])
        if not cands or r.risk_size <= 0:
            continue
        best: Optional[ExecRow] = None
        best_dt = None
        for e in cands:
            dt = abs((e.ts - r.ts).total_seconds())
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = e
        if best is None or best_dt is None or best_dt > match_window_sec:
            continue
        eff = best.cquote / r.risk_size
        effs.append(eff)
        matched += 1

    out.append("")
    out.append("4) Position sizing efficiency")
    if effs:
        out.append(
            f"- matched rows={matched} (window={match_window_sec}s), "
            f"mean(exec_quote / risk_size)={mean(effs):.3f} "
            f"min={min(effs):.3f} max={max(effs):.3f}"
        )
    else:
        out.append("- no matched BUY executions near VOL rows; cannot score efficiency yet")

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/app.log")
    p.add_argument("--low-atr", type=float, default=0.0045, help="ATR low threshold ratio (0.0045 = 0.45%)")
    p.add_argument("--high-atr", type=float, default=0.0150, help="ATR high threshold ratio (0.015 = 1.5%)")
    p.add_argument("--noise-floor-mult", type=float, default=1.20, help="SL should be >= mult * ATR")
    p.add_argument("--match-window-sec", type=int, default=60)
    args = p.parse_args()

    vols, execs = parse_log(args.log)
    print(
        analyze(
            vols=vols,
            execs=execs,
            low_atr_ratio=args.low_atr,
            high_atr_ratio=args.high_atr,
            sl_noise_floor_mult=args.noise_floor_mult,
            match_window_sec=args.match_window_sec,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

