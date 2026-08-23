"""
Guardrail tests for hybrid_allocator.py — the safe-core / capped-satellite barbell.

The satellite trades a negative-EV rule ON PURPOSE (operator chose "automated
speculative"); its P&L is not what these tests protect. What they protect is
THE WALL and the surrounding safety, which are the entire reason the system
exists:

  1. The trading loop NEVER redeems from the core — satellite losses can never
     reach the 80% safe yield. redeem is called ONLY in one-time live setup.
  2. Money flows satellite -> core (win-sweep) only, never core -> satellite.
  3. Below the floor the sleeve idles; it is never topped up from the core.
  4. paper/dryrun move NO real money.
  5. Sub-min-notional entries are skipped, not force-placed.

hybrid_allocator reads its config from env at import, so each test reloads the
module with a patched environment (tmp-isolated state/ledger/arm/kill paths).
"""
from __future__ import annotations

import importlib
import os
import time
from unittest.mock import AsyncMock

import pytest


def _load(tmp_path, **env):
    base = {
        "HYBRID_MODE": "paper",
        "HYBRID_STATE_FILE": str(tmp_path / "state.json"),
        "HYBRID_LEDGER_FILE": str(tmp_path / "ledger.jsonl"),
        "HYBRID_LIVE_ARM_FILE": str(tmp_path / "armed"),
        "HYBRID_KILL_FILE": str(tmp_path / "kill"),
        "HYBRID_CORE_FRACTION": "0.80",
        "HYBRID_SATELLITE_FLOOR_USD": "5.0",
        "HYBRID_SWEEP_THRESHOLD_USD": "0.5",
    }
    base.update({k: str(v) for k, v in env.items()})
    for k, v in base.items():
        os.environ[k] = v
    import hybrid_allocator
    importlib.reload(hybrid_allocator)  # load_dotenv(override=False) won't clobber the above
    return hybrid_allocator


def _mock_client(*, spot_free="0.01", earn="37.59", price=100.0,
                 step="0.001", min_notional="5.0"):
    c = AsyncMock()
    c.get_asset_balance.return_value = {"free": spot_free}
    c.get_simple_earn_flexible_product_position.return_value = {
        "rows": [{"asset": "USDT", "totalAmount": earn}]}
    c.get_simple_earn_flexible_product_list.return_value = {
        "rows": [{"asset": "USDT", "productId": "USDT001"}]}
    c.get_symbol_ticker.return_value = {"price": str(price)}
    c.get_exchange_info.return_value = {"symbols": [{
        "symbol": "BTCUSDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": "0.00001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "NOTIONAL", "minNotional": min_notional},
        ],
    }]}
    c.order_market_buy.return_value = {"fills": [{"price": str(price), "qty": "0.06"}]}
    c.order_market_sell.return_value = {"fills": [{"price": str(price), "qty": "0.06"}]}
    c.create_oco_order.return_value = {"orderListId": 123}
    c.cancel_order.return_value = {}
    return c


# ─────────────────────────────────────────────────────────────────────────────
# 1. THE WALL — the trading loop never redeems from the core
# ─────────────────────────────────────────────────────────────────────────────
async def test_paper_setup_never_redeems(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client()
    state = {"position": None}
    await h._setup_allocation(c, state)
    c.redeem_simple_earn_flexible_product.assert_not_called()
    # paper seeds a simulated satellite ~= 20% of $37.60
    assert state["satellite_cash"] == pytest.approx(37.60 * 0.20, abs=0.05)


async def test_win_sweep_never_redeems_and_only_flows_to_core(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client()
    state = {"position": None, "satellite_cash": 15.0, "satellite_target": 7.52}
    new_cash = await h._win_sweep(c, state, 15.0)
    # excess (15 - 7.52 - 0.5 = 6.98) swept to core; NEVER redeemed
    c.redeem_simple_earn_flexible_product.assert_not_called()
    assert new_cash < 15.0
    assert state["satellite_cash"] == pytest.approx(7.52 + 0.5, abs=0.01)


async def test_live_close_sells_min_of_recorded_and_actual_balance(tmp_path):
    """If the buy fee was taken in the base asset, the real balance is below the
    recorded (bought) qty. Selling the recorded amount would fail 'insufficient
    balance' — the close must sell min(recorded, actual), rounded down."""
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(price=100.0, step="0.001")

    def _bal(asset=None):
        return {"free": "0.055"} if asset == "BTC" else {"free": "0.40"}
    c.get_asset_balance = AsyncMock(side_effect=_bal)  # actual BTC 0.055 < recorded 0.06

    state = {"satellite_cash": 0.4, "position": {
        "symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
        "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live"}}
    await h._close_position(c, state, "take-profit")
    c.order_market_sell.assert_called_once()
    _, kwargs = c.order_market_sell.call_args
    assert float(kwargs["quantity"]) == pytest.approx(0.055, abs=0.0005)  # not 0.06
    assert state["position"] is None


async def test_close_position_never_redeems(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client(price=104.0)  # +4% vs entry 100
    state = {"satellite_cash": 0.5, "position": {
        "symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
        "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "paper"}}
    await h._close_position(c, state, "take-profit")
    c.redeem_simple_earn_flexible_product.assert_not_called()
    assert state["position"] is None
    # proceeds returned to satellite cash (paper)
    assert state["satellite_cash"] == pytest.approx(0.5 + 0.06 * 104.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIVE setup redeems ONCE with the right deficit; then is gated off
# ─────────────────────────────────────────────────────────────────────────────
async def test_live_setup_redeems_once_correct_amount(tmp_path):
    open(tmp_path / "armed", "w").close()  # arm file present
    h = _load(tmp_path, HYBRID_MODE="live")
    assert h._is_live() is True
    h._EARN_PRODUCT_ID = "USDT001"
    c = _mock_client(spot_free="0.01", earn="37.59")
    state = {"position": None}
    await h._setup_allocation(c, state)
    # total 37.60, sat target 7.52, deficit 7.51 redeemed once
    c.redeem_simple_earn_flexible_product.assert_called_once()
    _, kwargs = c.redeem_simple_earn_flexible_product.call_args
    assert float(kwargs["amount"]) == pytest.approx(7.51, abs=0.02)
    assert state["setup_done"] is True

    # second call must NOT redeem again (the wall: setup is one-time)
    c.redeem_simple_earn_flexible_product.reset_mock()
    await h._setup_allocation(c, state)
    c.redeem_simple_earn_flexible_product.assert_not_called()


async def test_satellite_target_stable_across_restart(tmp_path):
    """satellite_target must be set ONCE and never recomputed. A restart while a
    position is open (its value held as the coin, not in spot/earn) would
    otherwise shrink `total` and the target, corrupting the win-sweep baseline
    (observed live: it dropped $7.52 -> $6.10)."""
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    h._EARN_PRODUCT_ID = "USDT001"
    # mid-position restart: spot+earn ($30.48) understates the real account
    c = _mock_client(spot_free="0.40", earn="30.08")
    state = {"position": {"symbol": "BTCUSDT"}, "setup_done": True, "satellite_target": 7.52}
    await h._setup_allocation(c, state)
    assert state["satellite_target"] == 7.52          # NOT recomputed to ~6.10
    c.redeem_simple_earn_flexible_product.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Win-sweep in live goes satellite -> core via subscribe (never reverse)
# ─────────────────────────────────────────────────────────────────────────────
async def test_live_win_sweep_subscribes_to_core(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    h._EARN_PRODUCT_ID = "USDT001"
    c = _mock_client()
    state = {"position": None, "satellite_target": 7.52}
    await h._win_sweep(c, state, 20.0)
    c.subscribe_simple_earn_flexible_product.assert_called_once()   # sat -> core
    c.redeem_simple_earn_flexible_product.assert_not_called()       # never core -> sat


async def test_win_sweep_noop_when_at_target(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client()
    state = {"position": None, "satellite_cash": 7.5, "satellite_target": 7.52}
    await h._win_sweep(c, state, 7.5)
    c.subscribe_simple_earn_flexible_product.assert_not_called()
    assert state["satellite_cash"] == 7.5  # unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sub-min-notional / floor: entries skipped, not force-placed
# ─────────────────────────────────────────────────────────────────────────────
async def test_open_below_min_notional_skipped(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client(min_notional="5.0")
    state = {"position": None, "satellite_cash": 3.0}  # below $5 floor
    ok = await h._open_position(c, state, "BTCUSDT", 3.0)
    assert ok is False
    assert state["position"] is None
    c.order_market_buy.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 5. paper / dryrun move NO real money
# ─────────────────────────────────────────────────────────────────────────────
async def test_paper_open_places_no_real_order(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper")
    c = _mock_client(price=100.0)
    state = {"position": None, "satellite_cash": 7.5}
    ok = await h._open_position(c, state, "BTCUSDT", 7.5)
    assert ok is True
    c.order_market_buy.assert_not_called()          # paper: simulated only
    assert state["position"]["symbol"] == "BTCUSDT"
    assert state["satellite_cash"] < 7.5            # cash deducted (simulated)


async def test_dryrun_open_places_no_real_order(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="dryrun")
    c = _mock_client(price=100.0)
    state = {"position": None, "satellite_cash": 7.5}
    ok = await h._open_position(c, state, "BTCUSDT", 7.5)
    assert ok is False                              # dryrun holds no position
    c.order_market_buy.assert_not_called()
    assert state["position"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Breakout signal logic
# ─────────────────────────────────────────────────────────────────────────────
def _kline(close, high, vol):
    return [0, "0", str(high), "0", str(close), str(vol), 0, "0", 0, "0", "0", "0"]


async def test_breakout_true_on_new_high_with_volume(tmp_path):
    h = _load(tmp_path, HYBRID_BREAKOUT_LOOKBACK="5", HYBRID_VOLUME_MULT="1.5")
    # 5 flat bars (high 100, vol 10), then a breakout bar (close 105 > 100, vol 30),
    # then a trailing (forming) bar that must be ignored.
    kl = [_kline(99, 100, 10) for _ in range(5)]
    kl.append(_kline(105, 106, 30))   # last CLOSED bar = breakout
    kl.append(_kline(105, 106, 5))    # forming bar (ignored)
    assert h._is_breakout(kl) is True


async def test_no_breakout_without_volume(tmp_path):
    h = _load(tmp_path, HYBRID_BREAKOUT_LOOKBACK="5", HYBRID_VOLUME_MULT="1.5")
    kl = [_kline(99, 100, 10) for _ in range(5)]
    kl.append(_kline(105, 106, 11))   # new high but volume < 1.5x avg
    kl.append(_kline(105, 106, 5))
    assert h._is_breakout(kl) is False


async def test_no_breakout_below_range(tmp_path):
    h = _load(tmp_path, HYBRID_BREAKOUT_LOOKBACK="5", HYBRID_VOLUME_MULT="1.5")
    kl = [_kline(99, 100, 10) for _ in range(5)]
    kl.append(_kline(99, 99.5, 30))   # high volume but no new high
    kl.append(_kline(99, 99.5, 5))
    assert h._is_breakout(kl) is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Closed trade is logged with net_pct (drives the drawdown-halt backstop)
# ─────────────────────────────────────────────────────────────────────────────
async def test_close_logs_trade_with_net_pct(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper", HYBRID_FEE_RT_PCT="0.2")
    c = _mock_client(price=104.0)
    state = {"satellite_cash": 0.5, "position": {
        "symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
        "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "paper"}}
    await h._close_position(c, state, "take-profit")
    import json
    lines = [json.loads(x) for x in open(h.LEDGER) if x.strip()]
    assert len(lines) == 1
    assert lines[0]["net_pct"] == pytest.approx(4.0 - 0.2, abs=0.01)  # +4% gross - fees
    assert lines[0]["reason"] == "take-profit"


# ═════════════════════════════════════════════════════════════════════════════
# HARDENING (gap audit G1-G8, 2026-08-17)
# ═════════════════════════════════════════════════════════════════════════════

# G3 — single-instance lock
def test_single_instance_lock(tmp_path):
    import fcntl
    h = _load(tmp_path, HYBRID_PIDFILE=str(tmp_path / "h.pid"))
    assert h._acquire_lock() is True
    fh = open(tmp_path / "h.pid")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        second_got_it = True
    except OSError:
        second_got_it = False
    finally:
        fh.close()
    assert second_got_it is False  # lock held → a second instance is blocked


# G8 — live sizing capped to the satellite budget, not all spot USDT
async def test_live_satellite_cash_capped_to_target(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(spot_free="50.0")  # lots of unrelated USDT parked in spot
    state = {"satellite_target": 7.52}
    assert await h._satellite_cash(c, state) == pytest.approx(7.52)  # NOT $50


# G7 — drawdown-halt counts LIVE trades only
async def test_drawdown_halt_ignores_paper_trades(tmp_path):
    import json
    h = _load(tmp_path)
    with open(h.LEDGER, "w") as f:
        f.write(json.dumps({"net_pct": -50.0, "mode": "paper"}) + "\n")
        f.write(json.dumps({"net_pct": -2.0, "mode": "live"}) + "\n")
    assert h._current_drawdown_pct(live_only=True) == pytest.approx(2.0)   # only the live loss
    assert h._current_drawdown_pct(live_only=False) == pytest.approx(52.0)  # both


# G6 — paper round-trip now loses fees (was inconsistent with the ledger before)
async def test_paper_round_trip_loses_fees(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="paper", HYBRID_FEE_RT_PCT="0.2")
    c = _mock_client(price=100.0)
    state = {"position": None, "satellite_cash": 7.5}
    await h._open_position(c, state, "BTCUSDT", 7.5)
    await h._close_position(c, state, "time-stop")  # exit at same price
    assert state["satellite_cash"] < 7.5   # fee was deducted (used to be exactly 7.5)
    assert state["satellite_cash"] > 7.4   # but only a small fee


# G1 — live open places a protective OCO on the exchange
async def test_live_open_places_protective_oco(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(price=100.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.06", "locked": "0"} if asset == "BTC" else {"free": "7.5", "locked": "0"}))
    state = {"position": None, "satellite_target": 7.52}
    assert await h._open_position(c, state, "BTCUSDT", 7.5) is True
    c.order_market_buy.assert_called_once()
    c.create_oco_order.assert_called_once()       # exchange-side stop+target placed
    assert state["position"]["oco_id"] == 123


# G1 — software close cancels the resting OCO before the market sell
async def test_live_close_cancels_oco(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(price=100.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.06", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": 123}}
    await h._close_position(c, state, "time-stop")
    # Must cancel the LIST (DELETE /api/v3/orderList). cancel_order (DELETE
    # /api/v3/order) rejects orderListId and never cancels an OCO, which would
    # leave the qty locked and make the market sell fail with -2010.
    c.v3_delete_order_list.assert_called_once()
    assert c.cancel_order.call_count == 0
    c.order_market_sell.assert_called_once()
    assert state["position"] is None


# The time-stop must NOT drop a position whose qty is still locked by an OCO we
# failed to cancel — that would orphan a real holding the daemon stops managing.
async def test_live_close_blocked_when_qty_still_locked(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(price=100.0)
    c.v3_delete_order_list = AsyncMock(side_effect=Exception("APIError(code=-1102)"))
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.0", "locked": "0.06"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": 123}}
    assert await h._close_position(c, state, "time-stop") is False
    c.order_market_sell.assert_not_called()       # would have failed with -2010
    assert state["position"] is not None          # still tracked, retried next poll


# G1 — exchange OCO fired while we were away → detected and booked
async def test_detect_exchange_close_books_when_balance_gone(tmp_path):
    import json
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    entry_ts = time.time() - 3600
    c = _mock_client(price=104.5)  # at/above TP → labelled take-profit
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.0", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    c.get_my_trades = AsyncMock(return_value=[
        {"isBuyer": False, "qty": "0.06", "price": "104.0", "time": int((entry_ts + 60) * 1000)}])
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": entry_ts, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": 123}}
    assert await h._detect_exchange_close(c, state) is True
    assert state["position"] is None
    rec = json.loads(open(h.LEDGER).read().strip())
    assert rec["reason"] == "take-profit-oco"
    assert rec["exit_price"] == pytest.approx(104.0)


# G2 — reconcile protects a tracked position that has no resting stop
async def test_reconcile_protects_position_without_oco(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client(price=100.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.06", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": time.time() - 3600, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": None}}
    await h._reconcile(c, state)
    c.create_oco_order.assert_called_once()
    assert state["position"]["oco_id"] == 123


# G2 — reconcile adopts + protects a stray coin holding when state says flat
async def test_reconcile_adopts_stray_holding(tmp_path):
    open(tmp_path / "armed", "w").close()
    # PROTECTED_ASSETS cleared: this test is about adoption working at all, and
    # BTC is the only symbol the mock exchange-info defines filters for.
    h = _load(tmp_path, HYBRID_MODE="live", HYBRID_WATCHLIST="BTCUSDT",
              HYBRID_PROTECTED_ASSETS="")
    c = _mock_client(price=100.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.06", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    state = {"position": None, "satellite_target": 7.52}
    await h._reconcile(c, state)
    assert state["position"] is not None and state["position"]["symbol"] == "BTCUSDT"
    c.create_oco_order.assert_called_once()


# G5 — retry succeeds after transient failures, and re-raises after max
async def test_retry_succeeds_after_transient_failures(tmp_path):
    h = _load(tmp_path, HYBRID_API_RETRIES="3", HYBRID_API_RETRY_DELAY_S="0")
    calls = {"n": 0}
    async def flaky(**k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("transient")
        return {"ok": True}
    assert await h._retry(flaky, x=1) == {"ok": True}
    assert calls["n"] == 3


async def test_retry_reraises_after_max(tmp_path):
    h = _load(tmp_path, HYBRID_API_RETRIES="2", HYBRID_API_RETRY_DELAY_S="0")
    async def always_fail(**k):
        raise ValueError("nope")
    with pytest.raises(ValueError):
        await h._retry(always_fail)


# A long-term hold living in the same spot wallet must never be adopted or
# traded by the satellite, even if someone adds it to the watchlist.
async def test_protected_asset_excluded_from_watchlist_and_adoption(tmp_path):
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live",
              HYBRID_WATCHLIST="BTCUSDT,SOLUSDT", HYBRID_PROTECTED_ASSETS="BTC")
    assert "BTCUSDT" not in h.WATCHLIST          # filtered out of entry scanning
    assert "SOLUSDT" in h.WATCHLIST

    # Flat state + a big BTC balance => must NOT be adopted as a position.
    c = _mock_client(price=100.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "1.0", "locked": "0"} if asset == "BTC" else {"free": "0.0", "locked": "0"}))
    state = {"position": None}
    await h._reconcile(c, state)
    assert state["position"] is None
    c.create_oco_order.assert_not_called()


# An OCO that fired must be booked at its REAL fill price, not the intended
# trigger. A STOP_LOSS_LIMIT fills below its trigger, and labelling by the
# (stale) current price could book a rebounded stop-out as a take-profit.
async def test_exchange_close_books_real_fill_not_intended_price(tmp_path):
    import json
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    entry_ts = time.time() - 3600
    # Price has REBOUNDED above TP by poll time — the old code would have called
    # this a +4% take-profit. The fills say it was a stop-out at 97.2.
    c = _mock_client(price=105.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.0", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    c.get_my_trades = AsyncMock(return_value=[
        {"isBuyer": True,  "qty": "0.06", "price": "100.0", "time": int(entry_ts * 1000)},
        {"isBuyer": False, "qty": "0.06", "price": "97.2",  "time": int((entry_ts + 60) * 1000)},
    ])
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": entry_ts, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": 123}}
    assert await h._detect_exchange_close(c, state) is True
    rec = json.loads(open(h.LEDGER).read().strip())
    assert rec["reason"] == "stop-loss-oco"           # not take-profit
    assert rec["exit_price"] == pytest.approx(97.2)   # real fill, not sl_price 98.0
    assert rec["net_pct"] < 0


# Pre-entry sells must not pollute the exit VWAP.
async def test_exit_vwap_ignores_trades_before_entry(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="live")
    entry_ts = time.time() - 3600
    c = _mock_client()
    c.get_my_trades = AsyncMock(return_value=[
        {"isBuyer": False, "qty": "1.0", "price": "500.0", "time": int((entry_ts - 9999) * 1000)},
        {"isBuyer": False, "qty": "0.06", "price": "97.2", "time": int((entry_ts + 60) * 1000)},
    ])
    assert await h._actual_exit_vwap(c, "BTCUSDT", entry_ts) == pytest.approx(97.2)


# No usable history => return None so the caller falls back instead of booking 0.
async def test_exit_vwap_returns_none_without_history(tmp_path):
    h = _load(tmp_path, HYBRID_MODE="live")
    c = _mock_client()
    c.get_my_trades = AsyncMock(side_effect=Exception("api down"))
    assert await h._actual_exit_vwap(c, "BTCUSDT", time.time()) is None


# If the OCO fires between the pre-check and the market sell, the trade must
# still reach the ledger — dropping it silently loses a real round trip (the
# failure mode that lost the Aug-17 NEAR close).
async def test_close_race_with_oco_still_books_the_trade(tmp_path):
    import json
    open(tmp_path / "armed", "w").close()
    h = _load(tmp_path, HYBRID_MODE="live")
    entry_ts = time.time() - 3600
    c = _mock_client(price=98.0)
    c.get_asset_balance = AsyncMock(side_effect=lambda asset=None: (
        {"free": "0.0", "locked": "0"} if asset == "BTC" else {"free": "0.4", "locked": "0"}))
    c.get_my_trades = AsyncMock(return_value=[
        {"isBuyer": False, "qty": "0.06", "price": "97.2", "time": int((entry_ts + 60) * 1000)}])
    state = {"position": {"symbol": "BTCUSDT", "entry_ts": entry_ts, "entry_price": 100.0,
             "qty": 0.06, "tp_price": 104.0, "sl_price": 98.0, "mode": "live", "oco_id": 123}}
    assert await h._close_position(c, state, "time-stop") is True
    c.order_market_sell.assert_not_called()          # nothing left to sell
    assert state["position"] is None
    rec = json.loads(open(h.LEDGER).read().strip())  # but it IS in the ledger
    assert rec["reason"] == "stop-loss-oco"
    assert rec["exit_price"] == pytest.approx(97.2)
