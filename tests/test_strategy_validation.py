import json
from datetime import date, datetime, timedelta, timezone

from strategy_validation import evaluate


def _trade(day: date, index: int, gross_pnl: float) -> dict:
    ts = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(minutes=index)
    return {
        "ts": ts.isoformat(),
        "entry": 100.0,
        "qty": 1.0,
        "gross_pnl": gross_pnl,
    }


def test_validation_passes_sustained_net_edge_after_costs() -> None:
    start = date(2026, 1, 1)
    trades = []
    for offset in range(30):
        day = start + timedelta(days=offset)
        trades.extend(_trade(day, i, 1.0) for i in range(5))
        trades.append(_trade(day, 6, -0.5))

    report = evaluate(
        trades,
        start=start,
        end=start + timedelta(days=29),
        initial_capital=10_000.0,
        round_trip_fee_bps=20.0,
        round_trip_slippage_bps=10.0,
    )

    assert report["passed"] is True
    assert report["metrics"]["average_profitable_trades_per_day"] == 5.0


def test_zero_trade_days_count_against_frequency_target() -> None:
    start = date(2026, 1, 1)
    trades = [_trade(start, i, 1.0) for i in range(5)]

    report = evaluate(
        trades,
        start=start,
        end=start + timedelta(days=29),
        initial_capital=10_000.0,
    )

    assert report["metrics"]["average_profitable_trades_per_day"] == 0.1667
    assert report["gates"]["average_profitable_trades_per_day"] is False
    assert report["passed"] is False


def test_gross_winner_below_cost_is_a_net_loss() -> None:
    start = date(2026, 1, 1)
    report = evaluate(
        [_trade(start, 0, 0.20)],
        start=start,
        end=start + timedelta(days=29),
        initial_capital=1_000.0,
        round_trip_fee_bps=20.0,
        round_trip_slippage_bps=18.0,
    )

    assert report["metrics"]["net_profitable_trades"] == 0
    assert report["metrics"]["net_pnl_usdt"] < 0


def test_single_trade_risk_limit_is_enforced() -> None:
    start = date(2026, 1, 1)
    report = evaluate(
        [{"ts": start.isoformat(), "notional_usdt": 100.0, "net_pnl_usdt": -6.0}],
        start=start,
        end=start + timedelta(days=29),
        initial_capital=1_000.0,
    )

    assert report["metrics"]["maximum_single_trade_loss_pct"] == 0.6
    assert report["gates"]["single_trade_loss_limit"] is False


def test_all_winner_report_is_strict_json() -> None:
    start = date(2026, 1, 1)
    report = evaluate(
        [_trade(start, 0, 1.0)],
        start=start,
        end=start + timedelta(days=29),
        initial_capital=1_000.0,
    )

    assert report["metrics"]["profit_factor"] is None
    json.dumps(report, allow_nan=False)
