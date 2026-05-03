from __future__ import annotations

from src.l1_exchange.exchange_client import ExchangeClient


def _client() -> ExchangeClient:
    c = ExchangeClient.__new__(ExchangeClient)
    c.config = {}
    return c


def test_live_client_order_id_is_fresh_even_with_deterministic_hint():
    c = _client()
    ts_ms = 1_777_777_777_000
    hint = "ETHUSDT:SELL:ETHUSDT:SELL:2:0"

    first = c._make_live_client_order_id(
        ts_ms=ts_ms,
        side="SELL",
        tag="meta_exit",
        client_order_id_hint=hint,
    )
    second = c._make_live_client_order_id(
        ts_ms=ts_ms,
        side="SELL",
        tag="meta_exit",
        client_order_id_hint=hint,
    )

    assert first != second
    assert first.startswith("octi")
    assert ":" not in first
    assert len(first) <= 36


def test_recovered_order_must_match_current_submit_window_and_identity():
    c = _client()
    submit_started_ms = 1_777_777_777_000
    raw = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "clientOrderId": "octi7777777000abc123Sdeadbeefmeta",
        "updateTime": submit_started_ms + 250,
        "status": "FILLED",
    }

    ok, reason = c._validate_recovered_order_for_submit(
        raw,
        submit_started_ms=submit_started_ms,
        expected_symbol="ETHUSDT",
        expected_side="SELL",
        expected_client_order_id=raw["clientOrderId"],
    )
    assert ok is True
    assert reason == "fresh"


def test_recovered_order_rejects_stale_exchange_record():
    c = _client()
    submit_started_ms = 1_777_777_777_000
    raw = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "clientOrderId": "octi7777777000abc123Sdeadbeefmeta",
        "updateTime": submit_started_ms - 120_000,
        "status": "FILLED",
    }

    ok, reason = c._validate_recovered_order_for_submit(
        raw,
        submit_started_ms=submit_started_ms,
        expected_symbol="ETHUSDT",
        expected_side="SELL",
        expected_client_order_id=raw["clientOrderId"],
        stale_grace_ms=30_000,
    )
    assert ok is False
    assert reason.startswith("stale_recovered_order")


def test_recovered_order_rejects_mismatched_client_id():
    c = _client()
    raw = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "clientOrderId": "old-client-id",
        "updateTime": 1_777_777_777_100,
        "status": "FILLED",
    }

    ok, reason = c._validate_recovered_order_for_submit(
        raw,
        submit_started_ms=1_777_777_777_000,
        expected_symbol="ETHUSDT",
        expected_side="SELL",
        expected_client_order_id="new-client-id",
    )
    assert ok is False
    assert reason.startswith("client_order_id_mismatch")
