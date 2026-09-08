#!/usr/bin/env python3
"""Measure the $1/hour objective without placing or enabling trades.

Default is offline. --refresh-exchange performs authenticated READS only.
Reports estimated strategy P&L separately from account FIFO and NAV changes.
Idle calendar time counts. Statistical thresholds are screening criteria, not
a promise of returns or an automatic authorization to use real capital.
"""
from __future__ import annotations

import argparse
import os
import asyncio
import json
import math
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exchange_trade_audit import atomic_json, fifo_audit, reconcile_strategy, refresh_cache

# THE OPERATOR'S GOAL (set 2026-09-08): $4.39 per WEEK from $60.65, i.e. the
# account's whole annual yield, weekly. In per-hour terms, which is what every
# report and gate here is denominated in:
#
#     $4.39 / 168 h  =  $0.026131 / hour  =  376%/yr  =  1.034%/day
#
# For scale: the best verified rate this account can get is 7.24%/yr, so this
# asks 52x that; the best sustained record in fund history (Renaissance
# Medallion, ~66%/yr) is 5.7x short of it. The world can pay $228/yr — nobody
# offers it. This constant exists so the system is measured against the goal
# the operator actually set, and says plainly, every cycle, how far off it is.
# Override with PROFIT_TARGET_HOURLY_USD.
OPERATOR_TARGET_HOURLY_USD = float(os.getenv("PROFIT_TARGET_HOURLY_USD", str(4.39 / 168.0)))


def timestamp(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return ts.astimezone(timezone.utc)


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(value)
    return rows


def positive_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def strategy_summary(rows: list[dict], start: datetime, end: datetime,
                     capital: float, target_hourly: float, stress_bps: float = 10.0,
                     evidence_mode: str = "live") -> dict:
    if end <= start or capital <= 0 or target_hourly <= 0 or stress_bps < 0:
        raise ValueError("invalid evaluation window, capital, target, or costs")
    trades = []
    for row in rows:
        if row.get("mode") != evidence_mode or row.get("net_pct") is None or row.get("kind"):
            continue
        ts = timestamp(row["ts"])
        if not start <= ts < end:
            continue
        cost = float(row["entry_price"]) * float(row["qty"])
        net = float(row.get("net_pnl_usdt", cost * float(row["net_pct"]) / 100))
        if not all(math.isfinite(v) for v in (cost, net)) or cost <= 0:
            raise ValueError("nonfinite P&L or invalid trade notional")
        trades.append((ts, cost, net))
    trades.sort(key=lambda row: row[0])
    hours = (end - start).total_seconds() / 3600
    nets = [t[2] for t in trades]
    gains, losses = sum(max(n, 0) for n in nets), -sum(min(n, 0) for n in nets)
    n = len(trades)
    net = sum(nets)
    notional = sum(t[1] for t in trades)
    mean = net / n if n else 0
    # Complete UTC days only for daily resampling. Partial days are not padded
    # into complete days. All trades still count in the exact hourly result.
    first = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if first < start:
        first += timedelta(days=1)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    days = {}
    day = first
    while day < last:
        days[day.date()] = 0.0
        day += timedelta(days=1)
    for ts, _, pnl in trades:
        if ts.date() in days:
            days[ts.date()] += pnl
    daily = list(days.values())
    lower_hourly = None
    if len(daily) >= 2 and n:
        rng = random.Random(6065)
        samples = sorted(statistics.mean(rng.choices(daily, k=len(daily))) / 24
                         for _ in range(2000))
        lower_hourly = samples[49]  # lower 2.5% daily-block bootstrap quantile
    peak = equity = capital
    drawdown = 0.0
    for pnl in nets:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak * 100)
    pf = gains / losses if losses else None
    gates = {
        "at_least_30_complete_days": len(daily) >= 30,
        "at_least_100_closed_trades": n >= 100,
        "positive_estimated_expectancy": mean > 0,
        "profit_factor_at_least_1_2": pf >= 1.2 if pf is not None else gains > 0,
        "positive_daily_bootstrap_lower_bound": lower_hourly is not None and lower_hourly > 0,
        "positive_after_extra_10bps_cost": net - notional * stress_bps / 10000 > 0,
        "closed_trade_drawdown_under_10pct": drawdown <= 10,
    }
    return {"start": start.isoformat(), "end_exclusive": end.isoformat(),
            "calendar_hours_including_idle": hours, "complete_days": len(daily),
            "trades": n, "wins": sum(p > 0 for p in nets),
            "estimated_net_usdt": net, "estimated_mean_usdt_per_trade": mean,
            "estimated_net_usdt_per_hour": net / hours, "profit_factor": pf,
            "estimated_closed_trade_drawdown_pct": drawdown,
            "daily_bootstrap_lower_95pct_usdt_per_hour": lower_hourly,
            "extra_cost_stress_bps": stress_bps,
            "stressed_net_usdt": net - notional * stress_bps / 10000,
            "observed_trades_per_hour": n / hours,
            "required_trades_per_hour_at_estimated_mean": target_hourly / mean if mean > 0 else None,
            "target_average_met": net / hours >= target_hourly,
            "screening_gates": gates, "screening_passed": all(gates.values()),
            "daily": [{"date": str(day), "net_usdt": pnl} for day, pnl in days.items()],
            "limitations": ["Historical hybrid ledger uses estimated fees and lacks reliable order attribution.",
                            "Bootstrap is a small-sample diagnostic, assumes exchangeable days, and cannot prove an edge.",
                            "Drawdown uses closed trades only; open inventory and intraday losses are not covered."]}


def nav_windows(rows: list[dict], now: datetime) -> dict:
    ordered = sorted((r for r in rows if timestamp(r["ts"]) <= now), key=lambda r: timestamp(r["ts"]))
    if not ordered:
        return {"available": False, "windows": []}
    last = ordered[-1]
    end = timestamp(last["ts"])
    windows = []
    for h in (24, 72, 168):
        cutoff = end - timedelta(hours=h)
        before = [r for r in ordered if timestamp(r["ts"]) <= cutoff]
        if not before:
            continue
        first = before[-1]
        hours = (end - timestamp(first["ts"])).total_seconds() / 3600
        # NAV schema additions can appear as profits. Report, but do not
        # silently call differences across wallet-coverage migrations returns.
        schema_fields = {"spot_stable_usd", "earn_stable_usd", "other_wallets_usd", "holdings_usd"}
        same_schema = all((k in first) == (k in last) for k in schema_fields)
        flows_known = all("cumulative_contributions" in r for r in (first, last))
        flow = float(last["cumulative_contributions"]) - float(first["cumulative_contributions"]) if flows_known else None
        change = float(last["nav"]) - float(first["nav"])
        adjusted = change - flow if flows_known else None
        windows.append({"requested_hours": h, "actual_hours": hours, "start": first["ts"],
                        "end": last["ts"], "nav_change_usdt": change, "recorded_net_contributions_usdt": flow,
                        "contribution_adjusted_change_usdt": adjusted,
                        "same_wallet_schema": same_schema,
                        "comparable_usdt_per_hour": adjusted / hours if same_schema and flows_known else None})
    return {"available": True, "latest": last, "age_hours": (now - end).total_seconds() / 3600,
            "windows": windows,
            "limitation": "Uses locally recorded flows and wallet marks; not independently reconciled deposit/withdrawal profit."}


def render(report: dict) -> str:
    s = report["strategy"]
    failed = [k for k, v in report["readiness_gates"].items() if not v]
    status = ("Insufficient evidence for live strategy deployment or the hourly income target."
              if failed else "Screening gates passed; no live trading is enabled and future returns are not guaranteed.")
    lines = ["# Profit readiness", "", f"Generated: {report['generated_at']}", "",
             f"**Status: {status}**", "",
             f"Capital assumption: ${report['capital_usdt']:.2f}; target: ${report['target_usdt_per_hour']:.6f}/hour "
             f"(= ${report['target_usdt_per_hour']*168:.2f}/week).",
             f"Required net return: {report['required_hourly_return_pct']:.4f}% per hour.", "",
             f"P&L basis: {report['pnl_basis']}.", "",
             "| Historical live strategy metric | Result |", "|---|---:|",
             f"| Closed trades / wins | {s['trades']} / {s['wins']} |",
             f"| Total net under stated basis | ${s['estimated_net_usdt']:.6f} |",
             f"| Net/hour, including idle time | ${s['estimated_net_usdt_per_hour']:.8f} |",
             f"| Profit factor | {s['profit_factor']} |",
             f"| Net after extra {s['extra_cost_stress_bps']:g} bps execution cost | ${s['stressed_net_usdt']:.6f} |", "",
             "## Account value changes", ""]
    for w in report["nav"]["windows"]:
        rate = w["comparable_usdt_per_hour"]
        text = f"${rate:.8f}/hour" if rate is not None else "not comparable across wallet schema/flow gaps"
        lines.append(f"- {w['actual_hours']:.2f} hours: {text}. Includes price changes; not just realized trading profit.")
    audit = report.get("exchange_audit")
    lines += ["", "## Exchange reconciliation", "", f"Read status: {report['exchange_read_status']}."]
    if audit:
        lines += [f"- Last successful exchange snapshot: {audit['fetched_at']}.",
                  f"- Account FIFO audit: {audit['resolved_sell_orders']} resolved sell orders, {audit['unresolved_sell_orders']} unresolved. Detailed diagnostic is in latest.json.",
                  "- Account FIFO includes unrelated activity and incomplete fee-token inventory movements; do not use its subset total as strategy or whole-account profit."]
    reconciled = report.get("strategy_reconciliation")
    if reconciled:
        lines += [f"- Strategy trades matched with fee valuation bounds: {reconciled['matched_with_cost_bounds']}; unresolved: {reconciled['unresolved_trades']}.",
                  f"- Matched subset realized net: ${reconciled['resolved_subset_net_low_usdt']:.6f} to ${reconciled['resolved_subset_net_high_usdt']:.6f}.",
                  "- Historical order matches are inferred; residual holdings and unmatched trades still require reconciliation."]
    paper = report.get("forward_paper")
    if paper:
        lines += ["", "## Forward research baseline", "",
                  f"- Virtual initial capital: ${paper['initial_capital']:.2f}; latest liquidation-value NAV: ${paper['nav']:.4f}.",
                  f"- Completed trades: {paper['closed_trades']}; elapsed hours: {paper['elapsed_hours']:.2f}; coverage gaps: {paper['coverage_gaps']}.",
                  f"- Last successful market observation: {paper['last_observation']}; age: {paper['age_hours']:.2f} hours.",
                  "- Existing breakout rule, frozen as a baseline; its earlier research failed. New simulated data is not evidence of a profitable strategy yet.",
                  "- Uses observed bids/asks and costs. Stops are checked at observations; unseen intraday moves are not reconstructed."]
    lines += ["", "## Remaining gates", ""] + [f"- {g}" for g in failed]
    lines += ["", "## Interpretation", "",
              "Strategy screening and income-target achievement are separate. Passing a statistical screen does not enable trading.",
              "A new candidate needs a capital-constrained forward test, actual fee reconciliation, and marked open inventory.",
              "The current allocator remains in its configured mode. This report sends no orders and changes no allocation.", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, default=Path("logs/hybrid_ledger.jsonl"))
    p.add_argument("--nav", type=Path, default=Path("logs/nav_history.jsonl"))
    p.add_argument("--capital", type=positive_number, default=60.65)
    p.add_argument("--target-hourly", type=positive_number, default=OPERATOR_TARGET_HOURLY_USD,
                   help="net USDT per hour the operator is aiming for (default: $4.39/week)")
    p.add_argument("--start", type=timestamp, default=timestamp("2026-08-01T00:00:00+00:00"))
    p.add_argument("--end", type=timestamp)
    p.add_argument("--refresh-exchange", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("logs/profit_readiness"))
    args = p.parse_args()
    now = datetime.now(timezone.utc)
    end = args.end or now
    if end > now:
        p.error("end cannot be in the future")
    ledger, nav = read_rows(args.ledger), read_rows(args.nav)
    summary = strategy_summary(ledger, args.start, end, args.capital, args.target_hourly)
    symbols = sorted({r["symbol"] for r in ledger if r.get("mode") == "live" and "net_pct" in r})
    cache_path = args.output_dir / "exchange_fills.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    read_status = "cached" if cache else "not fetched"
    if args.refresh_exchange:
        try:
            fee_start = int((args.start - timedelta(days=3)).timestamp() * 1000)
            cache = asyncio.run(asyncio.wait_for(refresh_cache(cache_path, symbols, fee_start), timeout=180))
            read_status = "refreshed using read-only exchange calls"
        except Exception as exc:
            # API exception text may contain signed URLs; do not log it.
            read_status = f"refresh failed ({type(exc).__name__}); cached data is not current"
    audit = None
    reconciled = None
    if cache:
        audit = fifo_audit(cache["fills"], int(args.start.timestamp() * 1000), int(end.timestamp() * 1000))
        audit["fetched_at"] = cache.get("fetched_at")
        window_ledger = [r for r in ledger if args.start <= timestamp(r["ts"]) < end]
        reconciled = reconcile_strategy(window_ledger, cache["fills"], cache.get("fee_prices", {}))
    paper = None
    forward_passed = False
    paper_path = args.output_dir / "forward_paper.json"
    if paper_path.exists():
        state = json.loads(paper_path.read_text())
        last_observation = state.get("last_cycle") or state["started_at"]
        paper = {"initial_capital": state["config"]["initial_capital"],
                 "nav": state.get("latest_nav", state["cash"]),
                 "closed_trades": len(state["closed_trades"]),
                 "elapsed_hours": max(0, (now.timestamp() - state["started_at"]) / 3600),
                 "coverage_gaps": state["coverage_gaps"],
                 "last_observation": datetime.fromtimestamp(last_observation, timezone.utc).isoformat(),
                 "age_hours": max(0, (now.timestamp() - last_observation) / 3600)}
        paper_rows = [{**t, "net_pct": 100 * t["net_pnl_usdt"] / (t["qty"] * t["entry_price"])}
                      for t in state["closed_trades"]]
        began = datetime.fromtimestamp(state["started_at"], timezone.utc)
        if now > began:
            paper["screening"] = strategy_summary(paper_rows, began, now, paper["initial_capital"],
                                                   args.target_hourly, evidence_mode="forward_paper")
            paper["screening"]["limitations"][0] = "Forward simulated fills and explicit modeled costs; no actual exchange executions."
            marked_peak = paper["initial_capital"]
            marked_dd = 0.0
            for snap in state["snapshots"]:
                marked_peak = max(marked_peak, snap["nav"])
                marked_dd = max(marked_dd, (marked_peak - snap["nav"]) / marked_peak)
            paper["marked_drawdown_pct"] = marked_dd * 100
            forward_passed = (paper["screening"]["screening_passed"] and paper["coverage_gaps"] == 0
                              and paper["age_hours"] <= .5 and marked_dd <= .1
                              and paper["nav"] > paper["initial_capital"])
    exact_attribution = bool(reconciled and reconciled["trades"] and not reconciled["unresolved_trades"]
                             and all(t["attribution"] == "persisted_exchange_order_ids" for t in reconciled["trades"]))
    legacy_summary = summary
    pnl_basis = "legacy ledger with fixed estimated fees"
    if reconciled and reconciled["matched_with_cost_bounds"] and not reconciled["unresolved_trades"]:
        matched_rows = [{"mode": "live", "ts": t["exit_ts"], "entry_price": t["allocated_entry_cost_usdt"],
                         "qty": 1, "net_pct": 100 * t["realized_net_low_usdt"] / t["allocated_entry_cost_usdt"],
                         "net_pnl_usdt": t["realized_net_low_usdt"]} for t in reconciled["trades"]]
        summary = strategy_summary(matched_rows, args.start, end, args.capital, args.target_hourly)
        summary["limitations"][0] = "Exchange orders matched to strategy; legacy attribution is inferred. Fee conversions use the conservative end of historical minute ranges."
        pnl_basis = "matched exchange orders; conservative fee valuation bound; legacy attribution inferred"
    nav_summary = nav_windows(nav, end)
    fresh_exchange = bool(cache and cache.get("fetched_at") and
                          0 <= (now - timestamp(cache["fetched_at"])).total_seconds() <= 7200)
    report = {"generated_at": now.isoformat(), "capital_usdt": args.capital,
              "target_usdt_per_hour": args.target_hourly,
              "required_hourly_return_pct": args.target_hourly / args.capital * 100,
              "strategy": summary, "legacy_ledger_strategy": legacy_summary, "pnl_basis": pnl_basis,
              "nav": nav_summary,
              "exchange_read_status": read_status, "exchange_audit": audit,
              "strategy_reconciliation": reconciled,
              "forward_paper": paper,
              "readiness_gates": {**summary["screening_gates"],
                                  "recent_exchange_snapshot": fresh_exchange,
                                  "recent_nav_observation": nav_summary.get("available", False) and nav_summary.get("age_hours", 999) <= 1,
                                  "strategy_fills_and_actual_costs_attributed": exact_attribution,
                                  "capital_constrained_forward_test_with_open_inventory": forward_passed,
                                  "target_average_observed": summary["target_average_met"]}}
    atomic_json(args.output_dir / "latest.json", report)
    (args.output_dir / "latest.md").write_text(render(report))
    print(render(report))
    # Exit 0 means the diagnostic ran, not that a strategy passed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
