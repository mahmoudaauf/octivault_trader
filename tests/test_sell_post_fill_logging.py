import asyncio
import time
from types import SimpleNamespace

import pytest

from core.execution_manager import ExecutionManager
from core.shared_state import SharedState


class _ExchangeStub:
    def place_market_order(self, *a, **k):
        return None


@pytest.mark.asyncio
async def test_sell_post_fill_records_trade_and_emits_event():
    ss = SharedState()
    exchange = _ExchangeStub()
    cfg = SimpleNamespace()
    em = ExecutionManager(cfg, ss, exchange)

    order = {
        "symbol": "BTCUSDT",
        "executedQty": 0.00018,
        "avgPrice": 67538.94,
        "orderId": "test-sell-1",
        "status": "FILLED",
        "fills": [
            {"qty": "0.00018", "price": "67538.94", "commission": "0.01215701", "commissionAsset": "USDT"}
        ],
        "cummulativeQuoteQty": "12.1570092",
    }

    res = await em._ensure_post_fill_handled("BTCUSDT", "SELL", order, tag="unit-test")

    # SharedState.trade_history should contain the SELL record
    history = list(getattr(ss, "trade_history", []))
    assert any(h.get("side") == "SELL" and abs(h.get("price") - 67538.94) < 1e-6 for h in history)

    # ExecutionManager should emit TRADE_EXECUTED event in shared_state's event log
    events = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("side") == "SELL" for e in events)


@pytest.mark.asyncio
async def test_multiple_small_sell_fills_get_recorded():
    ss = SharedState()
    exchange = _ExchangeStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    # Simulate multiple small fills coming as separate order payloads
    orders = [
        {"symbol": "BTCUSDT", "executedQty": 0.00009, "avgPrice": 67538.94, "orderId": "s1", "status": "FILLED", "cummulativeQuoteQty": "6.0785046"},
        {"symbol": "BTCUSDT", "executedQty": 0.00009, "avgPrice": 67538.94, "orderId": "s2", "status": "FILLED", "cummulativeQuoteQty": "6.0785046"},
    ]

    for o in orders:
        await em._ensure_post_fill_handled("BTCUSDT", "SELL", o, tag="unit-test")

    history = list(getattr(ss, "trade_history", []))
    sell_count = sum(1 for h in history if h.get("side") == "SELL")
    assert sell_count >= 2


class _FlakySharedState(SharedState):
    """SharedState that fails the first TRADE_EXECUTED emit, then succeeds."""
    def __init__(self):
        super().__init__()
        self._failed_once = False

    async def emit_event(self, event_name, payload):
        if event_name == "TRADE_EXECUTED" and not self._failed_once:
            self._failed_once = True
            raise RuntimeError("simulated transient emit failure")
        return await super().emit_event(event_name, payload)


@pytest.mark.asyncio
async def test_finalize_reemits_trade_executed_on_emit_failure():
    ss = _FlakySharedState()
    exchange = _ExchangeStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    order = {
        "symbol": "BTCUSDT",
        "executedQty": 0.00044,
        "avgPrice": 67538.94,
        "orderId": "flaky-sell-1",
        "status": "FILLED",
        "fills": [
            {"qty": "0.00018", "price": "67538.94", "commission": "0.01215701", "commissionAsset": "USDT"},
        ],
        "cummulativeQuoteQty": "29.7171336",
    }

    # First, run post-fill handler which will attempt to emit and fail once
    pf = await em._ensure_post_fill_handled("BTCUSDT", "SELL", order, tag="unit-test")
    assert isinstance(pf, dict)
    assert pf.get("trade_event_emitted") is False

    # Now finalize; the finalize logic should detect missing emission and re-emit
    await em._finalize_sell_post_fill(symbol="BTCUSDT", order=order, tag="unit-test", post_fill=pf)

    events = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("side") == "SELL" for e in events)


@pytest.mark.asyncio
async def test_emit_close_events_re_emits_trade_executed_when_missing():
    ss = SharedState()
    exchange = _ExchangeStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    raw = {
        "symbol": "BTCUSDT",
        "executedQty": 0.00018,
        "avgPrice": 67538.94,
        "orderId": "close-emit-1",
        "status": "FILLED",
        "fills": [{"qty": "0.00018", "price": "67538.94", "commission": "0.01215701", "commissionAsset": "USDT"}],
        "cummulativeQuoteQty": "12.1570092",
    }

    # Simulate post_fill indicating trade_event was not emitted earlier
    await em._emit_close_events("BTCUSDT", raw, post_fill={"emitted": False, "realized_committed": False})

    events = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("side") == "SELL" for e in events)


@pytest.mark.asyncio
async def test_emit_close_events_idempotent_when_already_emitted():
    ss = SharedState()
    exchange = _ExchangeStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    raw = {
        "symbol": "BTCUSDT",
        "executedQty": 0.00018,
        "avgPrice": 67538.94,
        "orderId": "close-emit-2",
        "status": "FILLED",
        "fills": [{"qty": "0.00018", "price": "67538.94", "commission": "0.01215701", "commissionAsset": "USDT"}],
        "cummulativeQuoteQty": "12.1570092",
    }

    # Emit canonical event first
    await em._emit_trade_executed_event("BTCUSDT", "SELL", "", raw)
    events_before = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("order_id") == raw["orderId"] for e in events_before)

    # Now call _emit_close_events (should be idempotent - not duplicate)
    await em._emit_close_events("BTCUSDT", raw, post_fill={"emitted": True, "realized_committed": False})
    events_after = await ss.get_recent_events(200)
    cnt = sum(1 for e in events_after if e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("order_id") == raw["orderId"])
    assert cnt == 1


@pytest.mark.asyncio
async def test_trade_executed_prefers_exchange_order_id_for_payload_and_cache():
    ss = SharedState()
    exchange = _ExchangeStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    raw = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "status": "FILLED",
        "executedQty": 0.00044,
        "avgPrice": 67538.94,
        # Normalized exchange-client payloads may carry internal order_id + exchange_order_id
        "order_id": "BTCUSDT:SELL:auto-123",
        "exchange_order_id": 57699521087,
        "client_order_id": "BTCUSDT:SELL:auto-123",
        "cummulativeQuoteQty": 29.7171336,
    }

    ok = await em._emit_trade_executed_event("BTCUSDT", "SELL", "unit-test", raw)
    assert ok is True

    events = await ss.get_recent_events(200)
    matched = [
        e for e in events
        if e.get("name") == "TRADE_EXECUTED"
        and str(e.get("data", {}).get("exchange_order_id")) == "57699521087"
    ]
    assert matched, "expected TRADE_EXECUTED with exchange_order_id in payload"
    assert str(matched[-1].get("data", {}).get("order_id")) == "57699521087"

    cache = getattr(em, "_trade_event_emit_cache", {}) or {}
    assert cache.get("BTCUSDT:SELL:OID:57699521087")


class _FilledNoExchangeIdStub:
    def place_market_order(self, *a, **k):
        return None

    async def get_order(self, symbol, *, order_id: int = None, client_order_id: str = None):
        if client_order_id != "BTCUSDT:SELL:auto-123":
            return None
        return {
            "symbol": symbol,
            "status": "FILLED",
            "executedQty": "0.00044",
            "avgPrice": "67538.94",
            "orderId": 57699521087,
            "clientOrderId": client_order_id,
            "cummulativeQuoteQty": "29.7171336",
        }


@pytest.mark.asyncio
async def test_reconcile_filled_order_enriches_missing_exchange_order_id():
    ss = SharedState()
    exchange = _FilledNoExchangeIdStub()
    em = ExecutionManager(SimpleNamespace(), ss, exchange)

    order = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "status": "FILLED",
        "executedQty": "0.00044",
        "avgPrice": "67538.94",
        # No exchange order ID yet; only internal/client identifiers.
        "order_id": "BTCUSDT:SELL:auto-123",
        "clientOrderId": "BTCUSDT:SELL:auto-123",
    }

    merged = await em._reconcile_delayed_fill(
        symbol="BTCUSDT",
        side="SELL",
        order=order,
        tag="unit-test",
        client_order_id_hint="BTCUSDT:SELL:auto-123",
    )

    assert isinstance(merged, dict)
    assert str(merged.get("orderId") or merged.get("exchange_order_id")) == "57699521087"


class _ExchangeStubProbe:
    async def get_current_price(self, symbol):
        return 67538.94

    async def ensure_symbol_filters_ready(self, symbol):
        # Minimal filter shape sufficient for EM validation
        return {
            "LOT_SIZE": {"stepSize": "0.000001", "minQty": "0.00001", "maxQty": "1000"},
            "PRICE_FILTER": {"tickSize": "0.01"},
            "MIN_NOTIONAL": {"minNotional": "1"},
        }

    def place_market_order(self, *a, **k):
        # Simulate client-side failure (e.g. network) despite exchange possibly creating the order
        return None

    async def get_order(self, symbol, *, order_id: int = None, client_order_id: str = None):
        # Simulate exchange returning a filled order for the provided client_order_id
        return {
            "symbol": symbol,
            "orderId": 57699521087,
            "clientOrderId": client_order_id,
            "status": "FILLED",
            "executedQty": "0.00044",
            "avgPrice": "67538.94",
            "fills": [
                {"qty": "0.00018", "price": "67538.94", "commission": "0.01215701", "commissionAsset": "USDT"},
                {"qty": "0.00009", "price": "67538.94", "commission": "0.0060785", "commissionAsset": "USDT"},
                {"qty": "0.00009", "price": "67538.94", "commission": "0.0060785", "commissionAsset": "USDT"},
                {"qty": "0.00008", "price": "67538.94", "commission": "0.00540312", "commissionAsset": "USDT"},
            ],
            "cummulativeQuoteQty": "29.7171336",
        }


@pytest.mark.asyncio
async def test_place_market_order_none_but_recovered_by_client_order_probe():
    ss = SharedState()
    exchange = _ExchangeStubProbe()
    cfg = SimpleNamespace()
    em = ExecutionManager(cfg, ss, exchange)

    # Execute SELL — place_market_order will return None, but get_order will show the fill
    res = await em.execute_trade(symbol="BTCUSDT", side="sell", quantity=0.00044, tag="tp_sl")

    # Ensure the execute_trade path returned a dict and recognized the fill
    assert isinstance(res, dict)
    assert float(res.get("executedQty") or res.get("executed_qty") or 0.0) > 0

    # SharedState should have recorded the SELL and emitted TRADE_EXECUTED
    history = list(getattr(ss, "trade_history", []))
    assert any(h.get("side") == "SELL" for h in history)
    events = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("side") == "SELL" for e in events)


@pytest.mark.asyncio
async def test_close_position_emits_trade_executed_after_finalize():
    ss = SharedState()
    # Pre-seed a position so close_position treats this as an exit
    ss.positions["BTCUSDT"] = {"quantity": 0.00044, "status": "SIGNIFICANT"}

    exchange = _ExchangeStubProbe()
    cfg = SimpleNamespace()
    em = ExecutionManager(cfg, ss, exchange)

    # Stub execute_trade to immediately return a filled order so close_position can finalize
    async def _stub_execute_trade(symbol, side, quantity=None, planned_quote=None, tag="meta/Agent", tier=None, is_liquidation=False, policy_context=None):
        return {
            "symbol": symbol,
            "status": "FILLED",
            "executedQty": "0.00044",
            "avgPrice": "67538.94",
            "orderId": "close-stub-1",
            "clientOrderId": "close-stub-client-1",
            "fills": [{"qty": "0.00044", "price": "67538.94", "commission": "0.01", "commissionAsset": "USDT"}],
            "cummulativeQuoteQty": "29.7171336",
        }

    em.execute_trade = _stub_execute_trade

    # Call close_position (will use the stubbed execute_trade)
    res = await em.close_position(symbol="BTCUSDT", reason="TP_HIT", force_finalize=False, tag="tp_sl")

    assert isinstance(res, dict)
    # SharedState should have recorded the SELL and emitted TRADE_EXECUTED
    history = list(getattr(ss, "trade_history", []))
    assert any(h.get("side") == "SELL" for h in history)
    events = await ss.get_recent_events(200)
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("side") == "SELL" for e in events)
