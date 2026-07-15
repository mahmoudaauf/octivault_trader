"""
Tests for CarryPollingLoop (Phase 5 of the funding-carry native-wiring plan).

Uses a REAL CarryGateEngine + CarrySharedState (both lightweight and already
independently tested) so these tests exercise genuine end-to-end integration,
with only the futures client and executor mocked (network/order-placement
boundaries).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from core_engine.native.carry.executor import LegExecutionResult
from core_engine.native.carry.gates import CarryGateEngine
from core_engine.native.carry.poller import discover_carry_universe
from core_engine.native.carry.poller import CarryPollingLoop
from core_engine.native.carry.state import CarrySharedState, HedgePosition


def _carry_state(tmp_path) -> CarrySharedState:
    return CarrySharedState(
        state_path=str(tmp_path / "carry_state.json"),
        ledger_path=str(tmp_path / "carry_ledger.jsonl"),
    )


def _gates(carry_state, **overrides) -> CarryGateEngine:
    kwargs = dict(
        carry_state=carry_state,
        entry_bps=6.0,
        exit_bps=1.0,
        positive_only=True,
        max_positions=5,
        max_total_usd=5000.0,
        max_hold_h=360.0,
        max_drawdown_pct=5.0,
        liq_buffer_pct=15.0,
        kill_file=str(carry_state.state_path.parent / "carry.stop"),
    )
    kwargs.update(overrides)
    return CarryGateEngine(**kwargs)


def _poller(tmp_path, *, futures=None, executor=None, universe=None, **overrides) -> CarryPollingLoop:
    cs = _carry_state(tmp_path)
    gates = _gates(cs)
    kwargs = dict(
        futures_client=futures or AsyncMock(),
        carry_state=cs,
        carry_gates=gates,
        carry_executor=executor or AsyncMock(),
        universe=universe or {"BTCUSDT"},
        default_notional_usd=10.0,
        funding_poll_interval_sec=0.01,
        liq_check_interval_sec=0.01,
    )
    kwargs.update(overrides)
    return CarryPollingLoop(**kwargs)


class TestFetchFunding:
    @pytest.mark.asyncio
    async def test_filters_to_universe_and_parses_rates(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [
            {"symbol": "BTCUSDT", "lastFundingRate": "0.0007"},
            {"symbol": "ETHUSDT", "lastFundingRate": "0.0009"},  # not in universe
        ]
        p = _poller(tmp_path, futures=futures, universe={"BTCUSDT"})
        funding = await p.fetch_funding()
        assert funding == {"BTCUSDT": 0.0007}

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_empty_dict_not_raises(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.side_effect = RuntimeError("network down")
        p = _poller(tmp_path, futures=futures)
        funding = await p.fetch_funding()
        assert funding == {}


class TestRunFundingCycleOpens:
    @pytest.mark.asyncio
    async def test_opens_new_eligible_position(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0007"}]
        executor = AsyncMock()
        executor.open_hedge.return_value = LegExecutionResult(True, "ok")

        p = _poller(tmp_path, futures=futures, executor=executor)
        result = await p.run_funding_cycle()

        assert result["opened"] == 1
        executor.open_hedge.assert_awaited_once_with("BTCUSDT", 0.0007, 10.0)

    @pytest.mark.asyncio
    async def test_does_not_open_below_entry_threshold(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0002"}]
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor)
        result = await p.run_funding_cycle()

        assert result["opened"] == 0
        executor.open_hedge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_respects_notional_budget(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0007"}]
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor, default_notional_usd=10.0)
        p._gates.max_total_usd = 5.0  # smaller than the resolved notional
        result = await p.run_funding_cycle()

        assert result["opened"] == 0
        executor.open_hedge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uses_injected_notional_resolver(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0007"}]
        executor = AsyncMock()
        executor.open_hedge.return_value = LegExecutionResult(True, "ok")

        async def resolver():
            return 42.0

        p = _poller(tmp_path, futures=futures, executor=executor, resolve_notional_usd=resolver)
        await p.run_funding_cycle()
        executor.open_hedge.assert_awaited_once_with("BTCUSDT", 0.0007, 42.0)

    @pytest.mark.asyncio
    async def test_resolver_failure_falls_back_to_default(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0007"}]
        executor = AsyncMock()
        executor.open_hedge.return_value = LegExecutionResult(True, "ok")

        async def bad_resolver():
            raise RuntimeError("balance fetch failed")

        p = _poller(tmp_path, futures=futures, executor=executor, resolve_notional_usd=bad_resolver, default_notional_usd=15.0)
        await p.run_funding_cycle()
        executor.open_hedge.assert_awaited_once_with("BTCUSDT", 0.0007, 15.0)


class TestRunFundingCycleCloses:
    @pytest.mark.asyncio
    async def test_closes_when_funding_normalized_and_records_ledger(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.00001"}]
        futures.futures_funding_rate.return_value = [{"fundingRate": "0.0006"}, {"fundingRate": "0.0005"}]
        executor = AsyncMock()
        closed_pos = HedgePosition(
            symbol="BTCUSDT", entry_ts=0.0, entry_funding=0.0007, direction="short_perp",
            perp_qty=0.01, spot_qty=0.01, notional_usd=500.0,
        )
        executor.close_hedge.return_value = LegExecutionResult(True, "ok", position=closed_pos)

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.open_hedge("BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        result = await p.run_funding_cycle()

        assert result["closed"] == 1
        executor.close_hedge.assert_awaited_once_with("BTCUSDT")
        trades = p._carry_state.read_ledger()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"
        # accrued = 0.0006 + 0.0005 = 0.0011 -> 0.11%; net = (0.11% - fee_rt%)
        assert trades[0]["accrued_funding_pct"] == pytest.approx(0.11, abs=1e-6)

    @pytest.mark.asyncio
    async def test_does_not_close_while_funding_still_extreme(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = [{"symbol": "BTCUSDT", "lastFundingRate": "0.0009"}]
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.open_hedge("BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        result = await p.run_funding_cycle()

        assert result["closed"] == 0
        executor.close_hedge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_drawdown_halt_check_every_cycle(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = []
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.record_closed_trade("A", held_h=1, accrued_funding_pct=0, net_pct=-10.0, exit_funding=0.0001)
        p._gates.max_drawdown_pct = 1.0
        await p.run_funding_cycle()
        assert p._gates._killed() is True


class TestLiquidationCheck:
    @pytest.mark.asyncio
    async def test_force_closes_position_near_liquidation(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "-0.01", "markPrice": "50000.0", "liquidationPrice": "48000.0"}
        ]
        futures.futures_funding_rate.return_value = []
        executor = AsyncMock()
        closed_pos = HedgePosition(
            symbol="BTCUSDT", entry_ts=0.0, entry_funding=0.0007, direction="short_perp",
            perp_qty=0.01, spot_qty=0.01, notional_usd=500.0,
        )
        executor.close_hedge.return_value = LegExecutionResult(True, "ok", position=closed_pos)

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.open_hedge("BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        forced = await p.run_liquidation_check()

        assert forced == 1
        executor.close_hedge.assert_awaited_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_does_not_close_when_far_from_liquidation(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_position_information.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "-0.01", "markPrice": "50000.0", "liquidationPrice": "10000.0"}
        ]
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.open_hedge("BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        forced = await p.run_liquidation_check()

        assert forced == 0
        executor.close_hedge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_position_fetch_failure_skips_gracefully(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_position_information.side_effect = RuntimeError("network error")
        executor = AsyncMock()

        p = _poller(tmp_path, futures=futures, executor=executor)
        p._carry_state.open_hedge("BTCUSDT", entry_funding=0.0007, perp_qty=0.01, spot_qty=0.01, notional_usd=500.0)

        forced = await p.run_liquidation_check()
        assert forced == 0
        executor.close_hedge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_open_positions_is_a_noop(self, tmp_path) -> None:
        p = _poller(tmp_path)
        forced = await p.run_liquidation_check()
        assert forced == 0


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_spawns_two_tasks_and_stop_cancels_them(self, tmp_path) -> None:
        futures = AsyncMock()
        futures.futures_mark_price.return_value = []
        futures.futures_position_information.return_value = []
        p = _poller(tmp_path, futures=futures)

        await p.start()
        assert p._funding_task is not None
        assert p._liq_task is not None
        await asyncio.sleep(0.05)  # let at least one iteration run
        await p.stop()

        assert p._funding_task is None
        assert p._liq_task is None


class TestDiscoverCarryUniverse:
    @pytest.mark.asyncio
    async def test_filters_to_spot_hedgeable_liquid_perps(self) -> None:
        futures = AsyncMock()
        futures.futures_exchange_info.return_value = {
            "symbols": [
                {"symbol": "BTCUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "ETHUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "NOSPOTUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "DELISTEDUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "BREAK"},
            ]
        }
        futures.futures_ticker.return_value = [
            {"symbol": "BTCUSDT", "quoteVolume": "100000000"},
            {"symbol": "ETHUSDT", "quoteVolume": "1000"},  # below liquidity floor
            {"symbol": "NOSPOTUSDT", "quoteVolume": "100000000"},
        ]
        spot = AsyncMock()
        spot.get_exchange_info.return_value = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
                {"symbol": "ETHUSDT", "status": "TRADING"},
            ]
        }

        universe = await discover_carry_universe(futures, spot, min_vol_usd=50_000_000.0)
        assert universe == {"BTCUSDT"}

    @pytest.mark.asyncio
    async def test_failure_returns_empty_set_not_raises(self) -> None:
        futures = AsyncMock()
        futures.futures_exchange_info.side_effect = RuntimeError("network down")
        spot = AsyncMock()

        universe = await discover_carry_universe(futures, spot)
        assert universe == set()
