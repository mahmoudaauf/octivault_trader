#!/usr/bin/env python3
"""Validate a closed-trade ledger against OctiVault's daily strategy objective.

All profitability metrics are net of explicit round-trip fees and slippage.  The
validator never recommends or forces trades; zero-trade calendar days remain in
the evaluation window and therefore count against the frequency target.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AcceptanceCriteria:
    target_profitable_trades_per_day: float = 5.0
    minimum_calendar_days: int = 30
    minimum_profit_factor: float = 1.20
    minimum_expectancy_bps: float = 0.0
    maximum_daily_loss_pct: float = 2.0
    maximum_drawdown_pct: float = 10.0
    maximum_single_trade_loss_pct: float = 0.5


def _parse_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("trade is missing ts/timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid trade timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _notional(trade: dict[str, Any]) -> float:
    for key in ("notional_usdt", "notional", "entry_notional_usdt"):
        value = float(trade.get(key, 0.0) or 0.0)
        if value > 0:
            return value
    entry = float(trade.get("entry", trade.get("entry_price", 0.0)) or 0.0)
    qty = abs(float(trade.get("qty", trade.get("quantity", 0.0)) or 0.0))
    value = entry * qty
    if value <= 0:
        raise ValueError("trade needs positive notional or entry × quantity")
    return value


def _net_pnl(
    trade: dict[str, Any], *, round_trip_fee_bps: float, round_trip_slippage_bps: float
) -> tuple[float, float]:
    notional = _notional(trade)
    if trade.get("net_pnl_usdt") is not None:
        return float(trade["net_pnl_usdt"]), notional
    if trade.get("net_pct") is not None:
        return notional * float(trade["net_pct"]) / 100.0, notional
    gross = trade.get("gross_pnl", trade.get("gross_pnl_usdt"))
    if gross is None:
        pnl_pct = trade.get("pnl_pct", trade.get("gross_pnl_pct"))
        if pnl_pct is None:
            raise ValueError("trade needs net P&L or gross P&L")
        gross = notional * float(pnl_pct) / 100.0
    cost = notional * (round_trip_fee_bps + round_trip_slippage_bps) / 10_000.0
    return float(gross) - cost, notional


def _calendar_days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must not precede start date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def evaluate(
    trades: Iterable[dict[str, Any]],
    *,
    start: date,
    end: date,
    initial_capital: float,
    round_trip_fee_bps: float = 20.0,
    round_trip_slippage_bps: float = 10.0,
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> dict[str, Any]:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if round_trip_fee_bps < 0 or round_trip_slippage_bps < 0:
        raise ValueError("cost assumptions cannot be negative")

    days = _calendar_days(start, end)
    daily_pnl = {day: 0.0 for day in days}
    daily_wins = {day: 0 for day in days}
    normalized: list[tuple[datetime, float, float]] = []
    ignored_outside_window = 0
    for trade in trades:
        ts = _parse_timestamp(trade.get("ts", trade.get("timestamp")))
        if ts.date() < start or ts.date() > end:
            ignored_outside_window += 1
            continue
        pnl, notional = _net_pnl(
            trade,
            round_trip_fee_bps=round_trip_fee_bps,
            round_trip_slippage_bps=round_trip_slippage_bps,
        )
        normalized.append((ts, pnl, notional))
        daily_pnl[ts.date()] += pnl
        if pnl > 0:
            daily_wins[ts.date()] += 1

    normalized.sort(key=lambda row: row[0])
    total_net = sum(row[1] for row in normalized)
    total_notional = sum(row[2] for row in normalized)
    gross_profit = sum(max(0.0, row[1]) for row in normalized)
    gross_loss = abs(sum(min(0.0, row[1]) for row in normalized))
    # ``None`` represents an unbounded profit factor (profits with no losses) and
    # remains valid JSON, unlike Infinity.
    profit_factor = gross_profit / gross_loss if gross_loss else None
    expectancy_bps = total_net / total_notional * 10_000.0 if total_notional else 0.0

    equity = peak = initial_capital
    max_drawdown_pct = 0.0
    maximum_single_trade_loss_pct = 0.0
    for _, pnl, _ in normalized:
        equity_before = equity
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak - equity) / peak * 100.0)
        if pnl < 0:
            maximum_single_trade_loss_pct = max(
                maximum_single_trade_loss_pct,
                -pnl / max(equity_before, 1e-9) * 100.0,
            )

    daily_equity = initial_capital
    worst_daily_loss_pct = 0.0
    for day in days:
        pnl = daily_pnl[day]
        if pnl < 0:
            worst_daily_loss_pct = max(
                worst_daily_loss_pct, -pnl / max(daily_equity, 1e-9) * 100.0
            )
        daily_equity += pnl
    win_counts = list(daily_wins.values())
    average_profitable_trades_per_day = sum(win_counts) / len(days)
    median_profitable_trades_per_day = statistics.median(win_counts)

    gates = {
        "enough_calendar_days": len(days) >= criteria.minimum_calendar_days,
        "average_profitable_trades_per_day": average_profitable_trades_per_day
        >= criteria.target_profitable_trades_per_day,
        "median_profitable_trades_per_day": median_profitable_trades_per_day
        >= criteria.target_profitable_trades_per_day,
        "positive_net_pnl": total_net > 0,
        "profit_factor": (gross_profit > 0 if profit_factor is None else profit_factor >= criteria.minimum_profit_factor),
        "expectancy_bps": expectancy_bps > criteria.minimum_expectancy_bps,
        "daily_loss_limit": worst_daily_loss_pct <= criteria.maximum_daily_loss_pct,
        "drawdown_limit": max_drawdown_pct <= criteria.maximum_drawdown_pct,
        "single_trade_loss_limit": maximum_single_trade_loss_pct
        <= criteria.maximum_single_trade_loss_pct,
        "explicit_nonzero_costs": round_trip_fee_bps > 0 and round_trip_slippage_bps > 0,
    }
    return {
        "passed": all(gates.values()),
        "window": {"start": start.isoformat(), "end": end.isoformat(), "calendar_days": len(days)},
        "costs": {
            "round_trip_fee_bps": round_trip_fee_bps,
            "round_trip_slippage_bps": round_trip_slippage_bps,
        },
        "criteria": asdict(criteria),
        "metrics": {
            "trades": len(normalized),
            "ignored_outside_window": ignored_outside_window,
            "net_profitable_trades": sum(win_counts),
            "average_profitable_trades_per_day": round(average_profitable_trades_per_day, 4),
            "median_profitable_trades_per_day": median_profitable_trades_per_day,
            "net_pnl_usdt": round(total_net, 8),
            "profit_factor": profit_factor,
            "expectancy_bps": round(expectancy_bps, 4),
            "worst_daily_loss_pct": round(worst_daily_loss_pct, 4),
            "maximum_drawdown_pct": round(max_drawdown_pct, 4),
            "maximum_single_trade_loss_pct": round(maximum_single_trade_loss_pct, 4),
        },
        "daily": [
            {"date": day.isoformat(), "net_profitable_trades": daily_wins[day], "net_pnl_usdt": round(daily_pnl[day], 8)}
            for day in days
        ],
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--initial-capital", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, default=20.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(
        load_jsonl(args.ledger),
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        round_trip_fee_bps=args.fee_bps,
        round_trip_slippage_bps=args.slippage_bps,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
