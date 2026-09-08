from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from exchange_trade_audit import fetch_symbol, fifo_audit, reconcile_strategy
from profit_readiness import nav_windows, strategy_summary


def fill(id, buy, qty="1", price="100", fee="0", asset="USDT", order=None):
    return {"id": id, "symbol": "BTCUSDT", "time": id * 1000, "orderId": order or id,
            "qty": qty, "price": price, "commission": fee,
            "commissionAsset": asset, "isBuyer": buy}


def test_quote_fees_and_partial_sales_preserve_cost_basis():
    r = fifo_audit([fill(1, True, "2", fee="0.2"),
                    fill(2, False, "1", "110", "0.11"),
                    fill(3, False, "1", "90", "0.09")], 0, 10000)
    assert r["sell_orders"][0]["net_pnl_usdt"] == pytest.approx(9.79)
    assert r["sell_orders"][1]["net_pnl_usdt"] == pytest.approx(-10.19)
    assert r["resolved_subset_net_usdt"] == pytest.approx(-0.4)
    assert not r["open_lots"]


def test_base_buy_fee_is_not_double_counted():
    r = fifo_audit([fill(1, True, fee="0.001", asset="BTC"),
                    fill(2, False, ".999", "110", ".10989")], 0, 10000)
    assert r["resolved_subset_net_usdt"] == pytest.approx(9.78011)
    assert not r["open_lots"]


def test_base_sell_fee_consumes_extra_inventory():
    r = fifo_audit([fill(1, True), fill(2, False, ".999", "110", ".001", "BTC")], 0, 10000)
    assert r["resolved_subset_net_usdt"] == pytest.approx(9.89)
    assert not r["open_lots"]


def test_unconverted_fee_and_missing_entry_never_become_profit():
    r = fifo_audit([fill(1, False), fill(2, True, fee=".001", asset="BNB"),
                    fill(3, False, price="110")], 0, 10000)
    assert r["unresolved_sell_orders"] == 2
    assert r["resolved_subset_net_usdt"] == 0
    assert all(t["net_pnl_usdt"] is None for t in r["sell_orders"])
    assert len(r["unconverted_commissions"]) == 1


def test_deduplication_and_order_grouping_do_not_inflate_samples():
    a, b, c = fill(1, True, "2"), fill(2, False, order=20), fill(3, False, order=20)
    r = fifo_audit([a, a, b, c], 0, 10000)
    assert r["duplicates_removed"] == 1
    assert r["resolved_sell_orders"] == 1
    assert r["sell_orders"][0]["fill_count"] == 2
    with pytest.raises(ValueError, match="conflicting"):
        fifo_audit([a, {**a, "price": "200"}], 0, 10000)


def test_history_before_window_supplies_cost_not_window_profit():
    r = fifo_audit([fill(1, True), fill(2, False, price="110")], 1500, 2500)
    assert r["resolved_subset_net_usdt"] == 10
    assert r["resolved_sell_orders"] == 1
    assert not fifo_audit([fill(1, True), fill(2, False)], 0, 1500)["sell_orders"]


async def test_trade_id_pagination_keeps_fills_at_identical_timestamps():
    a, b, c = fill(1, True), fill(2, False), fill(3, True)
    b["time"] = a["time"]
    client = AsyncMock()
    client.get_my_trades.side_effect = [[a, b], [c]]
    assert await fetch_symbol(client, "BTCUSDT", page_size=2) == [a, b, c]
    assert client.get_my_trades.call_args_list[1].kwargs["fromId"] == 3


async def test_pagination_fails_closed_instead_of_accepting_truncation():
    client = AsyncMock()
    client.get_my_trades.return_value = [fill(1, True), fill(2, False)]
    with pytest.raises(ValueError, match="pagination"):
        await fetch_symbol(client, "BTCUSDT", page_size=2, max_pages=1)
    with pytest.raises(ValueError, match="advance"):
        await fetch_symbol(client, "BTCUSDT", after_id=5)


START = datetime(2026, 8, 1, tzinfo=timezone.utc)


def trade(day, net=1.0, mode="live"):
    return {"ts": (START + timedelta(days=day, hours=1)).isoformat(),
            "entry_price": 100, "qty": .1, "net_pct": net,
            "mode": mode, "symbol": "BTCUSDT"}


def test_idle_time_modes_and_nontrade_rows():
    rows = [trade(0), trade(0, 1000, "paper"),
            {"ts": START.isoformat(), "kind": "stable_rotation", "mode": "live"}]
    r = strategy_summary(rows, START, START + timedelta(days=30), 60.65, 1)
    assert r["trades"] == 1
    assert r["estimated_net_usdt_per_hour"] == pytest.approx(.1 / 720)
    assert len(r["daily"]) == 30
    assert not r["screening_passed"]
    assert not r["target_average_met"]


def test_100_tiny_winners_can_pass_edge_screen_but_miss_income_target():
    rows = [trade(day) for day in range(30) for _ in range(4)]
    r = strategy_summary(rows, START, START + timedelta(days=30), 60.65, 1)
    assert r["screening_passed"]
    assert not r["target_average_met"]


def test_cost_stress_removes_marginal_apparent_edge():
    r = strategy_summary([trade(0, .01)], START, START + timedelta(days=30), 60.65, 1)
    assert r["estimated_net_usdt"] > 0
    assert r["stressed_net_usdt"] < 0


def test_partial_days_are_not_invented_and_window_end_is_exclusive():
    end = START + timedelta(days=30, hours=12)
    row = {**trade(1), "ts": end.isoformat()}
    r = strategy_summary([row], START + timedelta(hours=12), end, 60.65, 1)
    assert r["complete_days"] == 29
    assert r["trades"] == 0


def test_deposit_is_not_profit_and_schema_migrations_are_flagged():
    end = START + timedelta(days=2)
    first = {"ts": START.isoformat(), "nav": 60, "cumulative_contributions": 0}
    last = {"ts": end.isoformat(), "nav": 160.1, "cumulative_contributions": 100}
    r = nav_windows([first, last], end)
    assert r["windows"][0]["contribution_adjusted_change_usdt"] == pytest.approx(.1)
    migrated = {**last, "other_wallets_usd": 5}
    assert nav_windows([first, migrated], end)["windows"][0]["comparable_usdt_per_hour"] is None


def test_nan_trade_values_are_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        strategy_summary([trade(0, float("nan"))], START, START + timedelta(days=1), 60.65, 1)


def paired_episode(buy, sell):
    return {"symbol": "BTCUSDT", "mode": "live", "ts": datetime.fromtimestamp(sell["time"] / 1000, timezone.utc).isoformat(),
            "held_h": (sell["time"] - buy["time"]) / 3600000,
            "entry_price": float(buy["price"]), "exit_price": float(sell["price"]),
            "qty": float(buy["qty"]), "net_pct": 1}


def test_strategy_reconciliation_values_bnb_fees_as_an_interval():
    buy, sell = fill(1, True, fee=".001", asset="BNB"), fill(2, False, price="110", fee=".001", asset="BNB")
    prices = {"BNBUSDT:0": {"low": 500, "high": 510}}
    r = reconcile_strategy([paired_episode(buy, sell)], [buy, sell], prices)
    assert r["matched_with_cost_bounds"] == 1
    assert r["resolved_subset_net_low_usdt"] == pytest.approx(8.98)
    assert r["resolved_subset_net_high_usdt"] == pytest.approx(9)


def test_partial_exit_preserves_residual_cost_and_uses_actual_sold_quantity():
    buy, sell = fill(1, True), fill(2, False, ".9", "110")
    r = reconcile_strategy([paired_episode(buy, sell)], [buy, sell], {})["trades"][0]
    assert r["remaining_base_quantity"] == pytest.approx(.1)
    assert r["remaining_cost_low_usdt"] == pytest.approx(10)
    assert r["realized_net_low_usdt"] == pytest.approx(9)
    assert r["ledger_quantity_difference"] == pytest.approx(.1)


def test_base_fee_buffer_is_charged_at_execution_value():
    buy, sell = fill(1, True, fee=".001", asset="BTC"), fill(2, False, price="110", fee=".001", asset="BTC")
    r = reconcile_strategy([paired_episode(buy, sell)], [buy, sell], {})["trades"][0]
    assert r["fee_buffer_base_used"] == pytest.approx(.002)
    assert r["realized_net_low_usdt"] == pytest.approx(9.79)


def test_ambiguous_legacy_match_is_rejected_but_persisted_ids_disambiguate():
    buy, sell, other = fill(1, True), fill(3, False), fill(2, True)
    ledger = paired_episode(buy, sell)
    assert reconcile_strategy([ledger], [buy, sell, other], {})["unresolved_trades"] == 1
    ledger.update(entry_order_id=1, exit_order_id=3)
    r = reconcile_strategy([ledger], [buy, sell, other], {})["trades"][0]
    assert r["attribution"] == "persisted_exchange_order_ids"
    assert r["status"] == "matched_with_fee_valuation_bounds"


def test_forward_evidence_is_never_counted_as_live_trades():
    rows = [trade(0, mode="forward_paper")]
    end = START + timedelta(days=30)
    assert strategy_summary(rows, START, end, 60.65, 1)["trades"] == 0
    assert strategy_summary(rows, START, end, 60.65, 1, evidence_mode="forward_paper")["trades"] == 1
