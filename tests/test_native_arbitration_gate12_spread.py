"""
gate_12_spread_check -- real-time spread guard, fail-CLOSED on missing data.

Added 2026-07-15 after a live incident: BBUSDT (0.53% spread, $197k/24h volume)
was bought by the live bot because gate_3/regime_gate.py's existing spread
check silently no-ops (passes) when the WS book cache has no data for a
symbol -- exactly the case for a just-rotated-in thin name outside the WS's
capped 12-symbol tracked set. gate_12 fixes this by failing CLOSED instead,
with a REST bookTicker fallback so a symbol merely outside the WS's tracked
set isn't blocked by default.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core_engine.native.arbitration_engine import NativeArbitrationEngine
from core_engine.native.shared_state import NativeSharedState


def _engine(tmp_path, *, exchange_client=None, shared_state=None) -> NativeArbitrationEngine:
    ss = shared_state if shared_state is not None else NativeSharedState()
    de = MagicMock()
    de.min_notional_usdt = 10.0
    de.max_concurrent_positions = 5
    de._resolve_mode = MagicMock(
        return_value={"max_positions": 5, "confidence_floor": 0.5}
    )
    engine = NativeArbitrationEngine(
        shared_state=ss, decision_engine=de, exchange_client=exchange_client
    )
    # Isolate from the real, live bot's logs/arb_state.json -- never read/write
    # production state from a unit test, even read-only.
    engine._arb_state_path = str(tmp_path / "arb_state.json")
    engine._load_streak_state()
    return engine


class TestLiveBookCache:
    @pytest.mark.asyncio
    async def test_passes_when_live_spread_is_tight(self, tmp_path) -> None:
        ss = NativeSharedState()
        ss.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.05, ask_qty=1.0)  # 0.05%
        engine = _engine(tmp_path, shared_state=ss)
        assert await engine.gate_12_spread_check("BTCUSDT") is True

    @pytest.mark.asyncio
    async def test_blocks_when_live_spread_is_wide(self, tmp_path) -> None:
        ss = NativeSharedState()
        ss.update_book("BBUSDT", bid=100.0, bid_qty=1.0, ask=100.53, ask_qty=1.0)  # 0.53%
        engine = _engine(tmp_path, shared_state=ss)
        assert await engine.gate_12_spread_check("BBUSDT") is False

    @pytest.mark.asyncio
    async def test_passes_at_exactly_the_threshold_boundary(self, tmp_path) -> None:
        ss = NativeSharedState()
        # spread exactly 0.5% -- strictly-greater-than semantics means this passes
        ss.update_book("ETHUSDT", bid=1000.0, bid_qty=1.0, ask=1005.0, ask_qty=1.0)
        engine = _engine(tmp_path, shared_state=ss)
        assert await engine.gate_12_spread_check("ETHUSDT") is True

    @pytest.mark.asyncio
    async def test_respects_custom_max_pct_threshold(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SPREAD_GATE_MAX_PCT", "0.01")  # 1% -- looser than default
        ss = NativeSharedState()
        ss.update_book("BBUSDT", bid=100.0, bid_qty=1.0, ask=100.53, ask_qty=1.0)  # 0.53%
        engine = _engine(tmp_path, shared_state=ss)
        assert await engine.gate_12_spread_check("BBUSDT") is True


class TestMissingOrStaleBookData:
    @pytest.mark.asyncio
    async def test_fails_closed_when_symbol_never_seen_and_no_exchange_client(self, tmp_path) -> None:
        ss = NativeSharedState()  # BBUSDT never updated -> get_book_age() == inf
        engine = _engine(tmp_path, shared_state=ss, exchange_client=None)
        assert await engine.gate_12_spread_check("BBUSDT") is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_book_data_is_stale(self, tmp_path) -> None:
        ss = NativeSharedState()
        ss.update_book("BBUSDT", bid=100.0, bid_qty=1.0, ask=100.05, ask_qty=1.0)
        # Age the snapshot past the freshness window without waiting in real time.
        ss.order_book["BBUSDT"]["ts"] -= 3600.0
        engine = _engine(tmp_path, shared_state=ss, exchange_client=None)
        assert await engine.gate_12_spread_check("BBUSDT") is False


class TestRestFallback:
    @pytest.mark.asyncio
    async def test_uses_rest_fallback_and_passes_on_tight_spread(self, tmp_path) -> None:
        ss = NativeSharedState()  # no live book data
        client = MagicMock()
        client.get_book_ticker = AsyncMock(
            return_value={"bid": 100.0, "bid_qty": 1.0, "ask": 100.05, "ask_qty": 1.0}
        )
        engine = _engine(tmp_path, shared_state=ss, exchange_client=client)
        assert await engine.gate_12_spread_check("SOMENEWUSDT") is True
        client.get_book_ticker.assert_awaited_once_with("SOMENEWUSDT")

    @pytest.mark.asyncio
    async def test_uses_rest_fallback_and_blocks_on_wide_spread(self, tmp_path) -> None:
        ss = NativeSharedState()
        client = MagicMock()
        client.get_book_ticker = AsyncMock(
            return_value={"bid": 100.0, "bid_qty": 1.0, "ask": 100.53, "ask_qty": 1.0}
        )
        engine = _engine(tmp_path, shared_state=ss, exchange_client=client)
        assert await engine.gate_12_spread_check("BBUSDT") is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_rest_fallback_also_unavailable(self, tmp_path) -> None:
        ss = NativeSharedState()
        client = MagicMock()
        client.get_book_ticker = AsyncMock(return_value=None)
        engine = _engine(tmp_path, shared_state=ss, exchange_client=client)
        assert await engine.gate_12_spread_check("BBUSDT") is False

    @pytest.mark.asyncio
    async def test_does_not_call_rest_fallback_when_live_data_is_fresh(self, tmp_path) -> None:
        """The REST call costs real API weight -- must only fire on a genuine cache miss."""
        ss = NativeSharedState()
        ss.update_book("BTCUSDT", bid=100.0, bid_qty=1.0, ask=100.05, ask_qty=1.0)
        client = MagicMock()
        client.get_book_ticker = AsyncMock(return_value={"bid": 1.0, "ask": 999.0})
        engine = _engine(tmp_path, shared_state=ss, exchange_client=client)
        assert await engine.gate_12_spread_check("BTCUSDT") is True
        client.get_book_ticker.assert_not_called()


class TestDisableFlag:
    @pytest.mark.asyncio
    async def test_disabled_gate_always_passes(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SPREAD_GATE_ENABLED", "false")
        ss = NativeSharedState()  # no book data at all, would otherwise fail-closed
        engine = _engine(tmp_path, shared_state=ss, exchange_client=None)
        assert await engine.gate_12_spread_check("BBUSDT") is True


class TestFullEvaluateIntegration:
    @pytest.mark.asyncio
    async def test_buy_blocked_by_gate_12_when_no_book_data_available(self, tmp_path) -> None:
        """End-to-end: a BUY that would otherwise clear every other gate is still
        blocked overall once gate_12 is in the mix, with no live/REST spread data."""
        ss = NativeSharedState()
        ss.current_mode = "NORMAL"
        engine = _engine(tmp_path, shared_state=ss, exchange_client=None)
        result = await engine.evaluate("BBUSDT", "BUY", 0.9)
        assert result["passed"] is False
        assert "gate_12_spread_check" in result["blocking_gates"]

    @pytest.mark.asyncio
    async def test_sell_is_not_gated_by_spread_check(self, tmp_path) -> None:
        """gate_12 is BUY-only -- exits must never be blocked by spread quality."""
        ss = NativeSharedState()
        engine = _engine(tmp_path, shared_state=ss, exchange_client=None)
        result = await engine.evaluate("BBUSDT", "SELL", 0.9)
        assert "gate_12_spread_check" not in result["gates_status"]
