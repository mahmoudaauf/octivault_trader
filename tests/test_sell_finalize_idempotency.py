from __future__ import annotations

import types

from src.l4_execution.execution_manager import ExecutionManager


def _manager() -> ExecutionManager:
    em = ExecutionManager.__new__(ExecutionManager)
    em.logger = types.SimpleNamespace(
        error=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    em._norm_symbol = lambda s: str(s or "").upper()  # type: ignore[assignment]
    em._safe_float = lambda v, default=0.0: float(v if v not in (None, "") else default)  # type: ignore[assignment]
    em._sell_finalize_state = {}
    em._sell_finalize_stats = {
        "fills_seen": 0,
        "finalized": 0,
        "duplicate_finalize": 0,
        "finalize_without_fill": 0,
        "pending_timeout": 0,
        "fills_seen_duplicate": 0,
    }
    em._sell_finalize_pending = 0
    em._sell_finalize_assert_window_s = 30.0
    em._sell_finalize_track_ttl_s = 3600.0
    em._sell_finalize_log_every = 25
    em._sell_finalize_last_report_ts = 0.0
    em._sell_finalize_last_report_finalized = -1
    return em


def test_sell_finalize_tracker_returns_false_for_duplicate_order():
    em = _manager()
    order = {
        "status": "FILLED",
        "executedQty": 0.0127,
        "exchange_order_id": 46106873668,
        "client_order_id": "octi-fresh-client",
    }

    em._track_sell_fill_observed(symbol="ETHUSDT", order=order, tag="meta_exit")

    assert em._track_sell_finalize(symbol="ETHUSDT", order=order, tag="meta_exit") is True
    assert em._sell_finalize_already_done(symbol="ETHUSDT", order=order) is True

    assert (
        em._track_sell_finalize(
            symbol="ETHUSDT",
            order=dict(order),
            tag="meta_exit",
            duplicate_attempt=True,
        )
        is False
    )
    assert em._track_sell_finalize(symbol="ETHUSDT", order=dict(order), tag="meta_exit") is False
    assert em._sell_finalize_stats["finalized"] == 1
    assert em._sell_finalize_stats["duplicate_finalize"] == 2


def test_sell_finalize_key_prefers_exchange_order_id_across_payload_shapes():
    em = _manager()
    normalized = {"exchange_order_id": "46106873668", "client_order_id": "cid-a"}
    raw = {"orderId": "46106873668", "clientOrderId": "cid-b"}

    assert em._sell_finalize_key("ETHUSDT", normalized) == "ETHUSDT|oid:46106873668"
    assert em._sell_finalize_key("ETHUSDT", raw) == "ETHUSDT|oid:46106873668"
