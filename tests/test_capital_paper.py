import pytest

from capital_paper import advance, breakout, initial_state, liquidation_nav


NOW = 1788811200.0
BOOKS = {"BTCUSDT": {"bid": 99.9, "ask": 100.1}}
FILTERS = {"BTCUSDT": {"step": .001, "min_qty": .001, "max_qty": 100, "min_notional": 5}}


def entered():
    return advance(initial_state(NOW), BOOKS, {"BTCUSDT": (1, True)}, FILTERS, NOW)


def test_paper_budget_includes_fees_and_leaves_core_cash():
    state = entered()
    assert state["position"]
    assert state["position"]["entry_cost"] <= 60.65 * .2
    assert state["cash"] >= 60.65 * .8
    assert state["latest_nav"] < 60.65  # spread and exit/entry costs included
    assert state["cash"] + state["position"]["entry_cost"] == pytest.approx(60.65)


def test_exchange_minimum_blocks_unaffordable_entry():
    filters = {"BTCUSDT": {**FILTERS["BTCUSDT"], "min_notional": 20}}
    s = advance(initial_state(NOW), BOOKS, {"BTCUSDT": (1, True)}, filters, NOW)
    assert s["position"] is None
    assert s["cash"] == 60.65


def test_paper_stop_books_observed_gap_and_both_sides_costs():
    s = entered()
    pos = s["position"]
    books = {"BTCUSDT": {"bid": 90, "ask": 90.1}}
    s = advance(s, books, {}, FILTERS, NOW + 900)
    expected = pos["qty"] * 90 * .9995 * .999 - pos["entry_cost"]
    assert s["closed_trades"][0]["net_pnl_usdt"] == pytest.approx(expected)
    assert s["cash"] == pytest.approx(60.65 + expected)
    assert s["position"] is None


def test_no_duplicate_cycle_or_reentry_on_old_signal():
    s = entered()
    assert advance(s, BOOKS, {"BTCUSDT": (1, True)}, FILTERS, NOW) == s
    s = advance(s, {"BTCUSDT": {"bid": 105, "ask": 105.1}}, {}, FILTERS, NOW + 900)
    assert s["position"] is None
    s = advance(s, BOOKS, {"BTCUSDT": (1, True)}, FILTERS, NOW + 4500)
    assert s["position"] is None
    assert len(s["closed_trades"]) == 1


def test_missing_mark_fails_instead_of_hiding_inventory_loss():
    with pytest.raises(ValueError, match="current mark"):
        advance(entered(), {}, {}, FILTERS, NOW + 900)


def test_unrealized_loss_affects_nav_before_close_and_gaps_are_counted():
    s = entered()
    books = {"BTCUSDT": {"bid": 99, "ask": 99.1}}
    nav_before = s["latest_nav"]
    s = advance(s, books, {}, FILTERS, NOW + 3600)
    assert s["position"] is not None
    assert s["latest_nav"] < nav_before
    assert s["coverage_gaps"] == 1
    assert liquidation_nav(s, books) == s["latest_nav"]


def test_open_candle_and_stale_breakout_do_not_trigger():
    candles = [[i * 3600000, "100", "101", "99", "100", "10", (i + 1) * 3600000 - 1] for i in range(22)]
    candles[-1][4], candles[-1][5] = "110", "100"
    assert breakout(candles, 21.5 * 3600)[1] is False
    assert breakout(candles, 22 * 3600 + 60)[1] is True
    assert breakout(candles, 23 * 3600)[1] is False


def test_global_drawdown_halts_future_entries():
    s = initial_state(NOW)
    s["cash"] = 50
    s = advance(s, BOOKS, {"BTCUSDT": (1, True)}, FILTERS, NOW)
    assert s["halted"]
    assert s["position"] is None


def test_exit_below_exchange_minimum_stays_in_inventory_not_fabricated_cash():
    s = entered()
    old_cash = s["cash"]
    s = advance(s, {"BTCUSDT": {"bid": 20, "ask": 20.1}}, {}, FILTERS, NOW + 900)
    assert s["exit_blocked"]
    assert s["position"] is not None
    assert s["cash"] == old_cash
    assert not s["closed_trades"]
