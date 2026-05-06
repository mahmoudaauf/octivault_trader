"""
Tests for NativeSafetyOrderManager (Phase 8.3.10).

Covers the API contract consumed by SafeExecutionEngine.place_safety_order
plus the OCO intent store and best-effort exchange placement path.

Test groups
-----------
- Construction validation (1)
- place_oco input validation (5)
- place_oco SIMULATED path (no exchange wired) (2)
- place_oco ACTIVE path (exchange wired, success) (2)
- place_oco FAILED path (exchange raises) (1)
- cancel_oco (3)
- get_oco / list_active / health (3)
- Bootstrap wiring + compat-stub-doesn't-overwrite (3)
"""

from __future__ import annotations

from typing import Any

import pytest

from core_engine.native.app_context import build_native_app_ctx
from core_engine.native.bootstrap import BootstrapConfig, build_components, shutdown_components
from core_engine.native.safety_order_manager import NativeSafetyOrderManager


# ----------------------------------------------------------------------
# Stub plumbing
# ----------------------------------------------------------------------
class _StubExchangeClient:
    """Minimal stub used by build_components for wiring tests."""

    async def close(self) -> None:
        return None

    async def get_account(self) -> dict[str, Any]:
        return {"balances": []}

    async def get_ticker_prices(self) -> dict[str, float]:
        return {}

    async def get_klines(self, *a: Any, **kw: Any) -> list[Any]:
        return []


class _RecordingExchangeClient:
    """Stub that records place_order/cancel_order calls and returns canned ids."""

    def __init__(
        self,
        *,
        place_returns: dict[str, Any] | None = None,
        place_raises: BaseException | None = None,
    ) -> None:
        self.place_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self._place_returns = place_returns or {"orderId": "tp-123"}
        self._place_raises = place_raises

    async def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.place_calls.append(kwargs)
        if self._place_raises is not None:
            raise self._place_raises
        return self._place_returns

    async def cancel_order(self, **kwargs: Any) -> dict[str, Any]:
        self.cancel_calls.append(kwargs)
        return {"status": "CANCELED"}


def _min_cfg(**overrides: Any) -> BootstrapConfig:
    base: dict[str, Any] = {
        "api_key": "k",
        "api_secret": "s",
        "testnet": True,
        "symbols": ["BTCUSDT"],
    }
    base.update(overrides)
    return BootstrapConfig(**base)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_rejects_non_positive_min_order_usdt() -> None:
    with pytest.raises(ValueError):
        NativeSafetyOrderManager(min_order_usdt=0.0)


# ----------------------------------------------------------------------
# place_oco — input validation
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_place_oco_rejects_zero_quantity() -> None:
    mgr = NativeSafetyOrderManager()
    res = await mgr.place_oco("BTCUSDT", quantity=0.0, take_profit=110.0, stop_loss=95.0)
    assert res["success"] is False
    assert "quantity" in res["error_message"]
    assert mgr.health()["failed"] == 1


@pytest.mark.asyncio
async def test_place_oco_rejects_zero_tp_or_sl() -> None:
    mgr = NativeSafetyOrderManager()
    res1 = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=0.0, stop_loss=95.0)
    res2 = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=0.0)
    assert res1["success"] is False
    assert res2["success"] is False


@pytest.mark.asyncio
async def test_place_oco_rejects_invalid_side() -> None:
    mgr = NativeSafetyOrderManager()
    res = await mgr.place_oco(
        "BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0, side="HOLD"
    )
    assert res["success"] is False
    assert "side" in res["error_message"]


@pytest.mark.asyncio
async def test_place_oco_sell_requires_tp_above_sl() -> None:
    mgr = NativeSafetyOrderManager()
    res = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=90.0, stop_loss=100.0)
    assert res["success"] is False
    assert "take_profit > stop_loss" in res["error_message"]


@pytest.mark.asyncio
async def test_place_oco_rejects_below_min_notional() -> None:
    mgr = NativeSafetyOrderManager(min_order_usdt=10.0)
    # 0.001 * 5 = 0.005 USDT notional
    res = await mgr.place_oco("BTCUSDT", quantity=0.001, take_profit=5.0, stop_loss=4.0)
    assert res["success"] is False
    assert "min_order_usdt" in res["error_message"]


# ----------------------------------------------------------------------
# place_oco — SIMULATED path
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_place_oco_simulated_when_no_exchange_wired() -> None:
    mgr = NativeSafetyOrderManager()
    res = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    assert res["success"] is True
    assert res["status"] == "SIMULATED"
    assert res["order_id"] is None
    assert mgr.health()["simulated"] == 1
    assert mgr.health()["placed"] == 0


@pytest.mark.asyncio
async def test_place_oco_persists_simulated_intent() -> None:
    mgr = NativeSafetyOrderManager()
    await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    snap = mgr.get_oco("BTCUSDT")
    assert snap is not None
    assert snap["status"] == "SIMULATED"
    assert snap["take_profit"] == 110.0
    assert snap["stop_loss"] == 95.0


# ----------------------------------------------------------------------
# place_oco — ACTIVE path (exchange wired, success)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_place_oco_active_when_exchange_succeeds() -> None:
    ex = _RecordingExchangeClient(place_returns={"orderId": "tp-xyz"})
    mgr = NativeSafetyOrderManager(exchange_client=ex)  # type: ignore[arg-type]
    res = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    assert res["success"] is True
    assert res["status"] == "PENDING"
    assert res["order_id"] == "tp-xyz"
    assert mgr.health()["placed"] == 1
    assert ex.place_calls[0]["side"] == "SELL"
    assert ex.place_calls[0]["order_type"] == "LIMIT"
    assert ex.place_calls[0]["price"] == 110.0


@pytest.mark.asyncio
async def test_place_oco_intent_records_active_status() -> None:
    ex = _RecordingExchangeClient(place_returns={"orderId": "tp-1"})
    mgr = NativeSafetyOrderManager(exchange_client=ex)  # type: ignore[arg-type]
    await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    snap = mgr.get_oco("BTCUSDT")
    assert snap is not None
    assert snap["status"] == "ACTIVE"
    assert snap["tp_order_id"] == "tp-1"


# ----------------------------------------------------------------------
# place_oco — FAILED path (exchange raises)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_place_oco_failed_path_persists_intent_with_error() -> None:
    ex = _RecordingExchangeClient(place_raises=RuntimeError("boom"))
    mgr = NativeSafetyOrderManager(exchange_client=ex)  # type: ignore[arg-type]
    res = await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    assert res["success"] is False
    assert "boom" in res["error_message"]
    snap = mgr.get_oco("BTCUSDT")
    assert snap is not None
    assert snap["status"] == "FAILED"
    assert snap["error"] == "boom"
    assert mgr.health()["failed"] == 1


# ----------------------------------------------------------------------
# cancel_oco
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancel_oco_drops_intent_and_cancels_exchange_order() -> None:
    ex = _RecordingExchangeClient(place_returns={"orderId": "tp-1"})
    mgr = NativeSafetyOrderManager(exchange_client=ex)  # type: ignore[arg-type]
    await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    assert await mgr.cancel_oco("BTCUSDT") is True
    assert mgr.get_oco("BTCUSDT") is None
    assert len(ex.cancel_calls) == 1
    assert ex.cancel_calls[0]["order_id"] == "tp-1"
    assert mgr.health()["canceled"] == 1


@pytest.mark.asyncio
async def test_cancel_oco_returns_false_when_unknown_symbol() -> None:
    mgr = NativeSafetyOrderManager()
    assert await mgr.cancel_oco("UNKNOWN") is False


@pytest.mark.asyncio
async def test_cancel_oco_simulated_intent_does_not_call_exchange() -> None:
    ex = _RecordingExchangeClient()
    mgr = NativeSafetyOrderManager()  # no exchange wired → SIMULATED
    await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    # Now wire exchange and cancel — the original intent has no tp_order_id
    mgr._exchange_client = ex  # type: ignore[attr-defined]
    assert await mgr.cancel_oco("BTCUSDT") is True
    assert ex.cancel_calls == []  # no order_id was recorded for SIMULATED


# ----------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_oco_returns_none_for_unknown() -> None:
    mgr = NativeSafetyOrderManager()
    assert mgr.get_oco("UNKNOWN") is None


@pytest.mark.asyncio
async def test_list_active_filters_to_active_and_simulated() -> None:
    ex = _RecordingExchangeClient(place_raises=RuntimeError("boom"))
    mgr = NativeSafetyOrderManager(exchange_client=ex)  # type: ignore[arg-type]
    # one FAILED intent
    await mgr.place_oco("BTCUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    # one SIMULATED intent on a different mgr (no exchange)
    mgr2 = NativeSafetyOrderManager()
    await mgr2.place_oco("ETHUSDT", quantity=1.0, take_profit=110.0, stop_loss=95.0)
    assert mgr.list_active() == []  # FAILED is excluded
    assert mgr2.list_active() == ["ETHUSDT"]


def test_health_reports_exchange_wired_flag() -> None:
    assert NativeSafetyOrderManager().health()["exchange_wired"] is False
    assert (
        NativeSafetyOrderManager(exchange_client=_RecordingExchangeClient()).health()[  # type: ignore[arg-type]
            "exchange_wired"
        ]
        is True
    )


# ----------------------------------------------------------------------
# Bootstrap wiring
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bootstrap_attaches_native_safety_order_manager() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        assert isinstance(components.safety_order_manager, NativeSafetyOrderManager)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_native_safety_order_manager_visible_in_app_ctx() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components)
        assert isinstance(app_ctx["safety_order_manager"], NativeSafetyOrderManager)
    finally:
        await shutdown_components(components)


@pytest.mark.asyncio
async def test_compat_stub_does_not_overwrite_native_safety_order_manager() -> None:
    components = await build_components(
        _min_cfg(), exchange_client_factory=lambda _c: _StubExchangeClient()
    )
    try:
        app_ctx, _orch = build_native_app_ctx(components, compat=True)
        assert isinstance(app_ctx["safety_order_manager"], NativeSafetyOrderManager)
    finally:
        await shutdown_components(components)
