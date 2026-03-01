import asyncio
import time
from types import SimpleNamespace

from core.exchange_truth_auditor import ExchangeTruthAuditor
from core.shared_state import SharedState
from core.execution_manager import ExecutionManager


def _ensure_event_loop():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class _SharedStateStub:
    def __init__(self):
        self.events = []
        self.positions = {}
        self.open_orders = {}
        self.sync_calls = 0

    async def emit_event(self, event_name, payload):
        self.events.append((event_name, dict(payload)))

    async def sync_authoritative_balance(self, force: bool = False):
        self.sync_calls += 1


class _ExecutionManagerStub:
    def __init__(self, health_payload):
        self._health_payload = dict(health_payload or {})

    def health(self):
        return dict(self._health_payload)


def test_truth_auditor_emits_sell_finalize_gap_event_on_mismatch():
    _ensure_event_loop()
    ss = _SharedStateStub()
    em = _ExecutionManagerStub(
        {
            "sell_finalize_fills_seen": 7,
            "sell_finalize_finalized": 4,
            "sell_finalize_pending": 3,
            "sell_finalize_duplicate": 1,
            "sell_finalize_pending_timeout": 0,
        }
    )
    cfg = SimpleNamespace(
        TRUTH_AUDIT_SELL_FINALIZE_WARN_COOLDOWN_SEC=0.0,
        TRUTH_AUDIT_SELL_FINALIZE_GAP_WARN_THRESHOLD=1,
    )
    auditor = ExchangeTruthAuditor(
        config=cfg,
        shared_state=ss,
        app=SimpleNamespace(execution_manager=em),
    )

    snap = asyncio.run(auditor._validate_sell_finalize_mapping(startup=False))

    assert snap["sell_finalize_fills_seen"] == 7
    assert snap["sell_finalize_finalized"] == 4
    assert snap["sell_finalize_pending"] == 3
    assert snap["sell_finalize_gap"] == 3

    names = [name for name, _ in ss.events]
    assert "TRUTH_AUDIT_SELL_FINALIZE_GAP" in names


def test_truth_auditor_does_not_emit_gap_event_when_balanced():
    _ensure_event_loop()
    ss = _SharedStateStub()
    em = _ExecutionManagerStub(
        {
            "sell_finalize_fills_seen": 5,
            "sell_finalize_finalized": 5,
            "sell_finalize_pending": 0,
            "sell_finalize_duplicate": 0,
            "sell_finalize_pending_timeout": 0,
        }
    )
    cfg = SimpleNamespace(
        TRUTH_AUDIT_SELL_FINALIZE_WARN_COOLDOWN_SEC=0.0,
        TRUTH_AUDIT_SELL_FINALIZE_GAP_WARN_THRESHOLD=1,
    )
    auditor = ExchangeTruthAuditor(
        config=cfg,
        shared_state=ss,
        app=SimpleNamespace(execution_manager=em),
    )

    snap = asyncio.run(auditor._validate_sell_finalize_mapping(startup=False))

    assert snap["sell_finalize_gap"] == 0
    assert all(name != "TRUTH_AUDIT_SELL_FINALIZE_GAP" for name, _ in ss.events)


def test_truth_auditor_triggers_execution_manager_on_recovered_fill():
    _ensure_event_loop()
    ss = SharedState()
    # Minimal exchange stub required by ExecutionManager constructor
    class _Ex:
        def place_market_order(self, *a, **k):
            return None

        async def get_current_price(self, symbol):
            return 67538.94

    exchange_stub = _Ex()
    # ExecutionManager will emit canonical TRADE_EXECUTED into SharedState
    em = ExecutionManager(SimpleNamespace(), ss, exchange_stub)

    auditor = ExchangeTruthAuditor(config=SimpleNamespace(), shared_state=ss, app=SimpleNamespace(execution_manager=em))

    order = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "executedQty": "0.00044",
        "avgPrice": "67538.94",
        "orderId": "truth-recovered-1",
        "clientOrderId": "truth-recovered-client-1",
        "status": "FILLED",
        "fills": [{"qty": "0.00044", "price": "67538.94", "commission": "0.01", "commissionAsset": "USDT"}],
        "cummulativeQuoteQty": "29.7171336",
    }

    # Apply recovered fill via auditor (synchronous wrapper)
    asyncio.run(auditor._apply_recovered_fill(order, reason="unit-test", synthetic=False))

    # SharedState.trade_history must contain the SELL
    history = list(getattr(ss, "trade_history", []))
    assert any(h.get("side") == "SELL" and str(h.get("price")) == "67538.94" for h in history)

    # ExecutionManager should have emitted canonical TRADE_EXECUTED into SharedState
    events = asyncio.run(ss.get_recent_events(200))
    assert any(e.get("name") == "TRADE_EXECUTED" and e.get("data", {}).get("order_id") == str(order["orderId"]) for e in events)


def test_truth_auditor_recognizes_canonical_sell_via_cid_emit_cache():
    _ensure_event_loop()
    ss = SharedState()

    class _Ex:
        def place_market_order(self, *a, **k):
            return None

    em = ExecutionManager(SimpleNamespace(), ss, _Ex())
    em._trade_event_emit_cache = {
        "BTCUSDT:SELL:CID:BTCUSDT:SELL:auto-123": 1771890536.8,
    }

    auditor = ExchangeTruthAuditor(
        config=SimpleNamespace(),
        shared_state=ss,
        app=SimpleNamespace(execution_manager=em),
    )

    order = {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "status": "FILLED",
        "executedQty": "0.00044",
        "avgPrice": "67538.94",
        "orderId": "57699521087",
        "clientOrderId": "BTCUSDT:SELL:auto-123",
    }

    assert auditor._is_canonical_trade_event_present(order) is True


class _ExchangeClientWsStub:
    def __init__(self, snapshot):
        self._snapshot = dict(snapshot or {})
        self.api_key = "k"
        self.api_secret = "s"
        self.is_started = True
        self.last_successful_force_sync_ts = 0.0
        self.reconnect_called = False

    def get_ws_health_snapshot(self):
        return dict(self._snapshot)

    def record_successful_force_sync(self, *, reason="", ts=None):
        self.last_successful_force_sync_ts = float(ts or 0.0)

    async def reconnect_user_data_stream(self, reason=""):
        self.reconnect_called = True
        return True

    async def get_open_orders(self, symbol=None):
        return []

    async def get_all_orders(self, symbol, limit=50):
        return []

    async def get_my_trades(self, symbol, limit=50):
        return []


def test_truth_auditor_trigger_immediate_sync_debounced():
    _ensure_event_loop()
    ss = _SharedStateStub()
    ec = _ExchangeClientWsStub(
        {
            "user_data_stream_enabled": True,
            "last_user_data_event_ts": 0.0,
            "last_any_ws_event_ts": 0.0,
            "last_listenkey_refresh_ts": 0.0,
            "last_successful_force_sync_ts": 0.0,
        }
    )
    cfg = SimpleNamespace(FORCE_SYNC_COOLDOWN_SEC=60.0)
    auditor = ExchangeTruthAuditor(config=cfg, shared_state=ss, exchange_client=ec)

    first = asyncio.run(auditor.trigger_immediate_sync(reason="MANUAL_TRIGGER"))
    second = asyncio.run(auditor.trigger_immediate_sync(reason="MANUAL_TRIGGER"))

    assert first["status"] == "ok"
    assert second["status"] == "skipped_cooldown"
    assert ss.sync_calls == 1


def test_truth_auditor_soft_gap_requires_exposure():
    _ensure_event_loop()
    ss = _SharedStateStub()
    ss.positions = {"BTCUSDT": {"quantity": 0.12}}
    now = time.time()
    ec = _ExchangeClientWsStub(
        {
            "user_data_stream_enabled": True,
            "ws_connected": True,
            "last_user_data_event_ts": now - 30.0,  # stale on purpose
            "last_any_ws_event_ts": now - 30.0,
            "last_listenkey_refresh_ts": now - 30.0,
            "last_successful_force_sync_ts": 0.0,
            "user_data_gap_sec": 999.0,
            "any_ws_gap_sec": 999.0,
            "listenkey_refresh_gap_sec": 999.0,
        }
    )
    cfg = SimpleNamespace(
        USER_DATA_SOFT_GAP_SEC=20.0,
        USER_DATA_HARD_GAP_SEC=10_000.0,
        FORCE_SYNC_COOLDOWN_SEC=15.0,
    )
    auditor = ExchangeTruthAuditor(config=cfg, shared_state=ss, exchange_client=ec)

    reasons = []

    async def _fake_trigger(reason: str = ""):
        reasons.append(str(reason))
        return {"status": "ok", "reason": reason}

    auditor.trigger_immediate_sync = _fake_trigger  # type: ignore[assignment]
    asyncio.run(auditor._check_user_data_health())

    assert reasons == ["SOFT_GAP_EXPOSURE"]


def test_truth_auditor_hard_gap_overrides_exposure_gate():
    _ensure_event_loop()
    ss = _SharedStateStub()
    now = time.time()
    ec = _ExchangeClientWsStub(
        {
            "user_data_stream_enabled": True,
            "ws_connected": False,
            "last_user_data_event_ts": now - 120.0,  # stale on purpose
            "last_any_ws_event_ts": now - 120.0,
            "last_listenkey_refresh_ts": now - 120.0,
            "last_successful_force_sync_ts": 0.0,
            "user_data_gap_sec": 999.0,
            "any_ws_gap_sec": 999.0,
            "listenkey_refresh_gap_sec": 999.0,
        }
    )
    cfg = SimpleNamespace(
        USER_DATA_SOFT_GAP_SEC=20.0,
        USER_DATA_HARD_GAP_SEC=30.0,
        FORCE_SYNC_COOLDOWN_SEC=15.0,
    )
    auditor = ExchangeTruthAuditor(config=cfg, shared_state=ss, exchange_client=ec)

    reasons = []

    async def _fake_trigger(reason: str = ""):
        reasons.append(str(reason))
        return {"status": "ok", "reason": reason}

    auditor.trigger_immediate_sync = _fake_trigger  # type: ignore[assignment]
    asyncio.run(auditor._check_user_data_health())

    assert reasons == ["HARD_GAP_TIMEOUT"]
    assert ec.reconnect_called is True
