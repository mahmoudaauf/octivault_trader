"""
Tests for the price-overextension BUY guard in make_buy_decision.

The guard blocks a BUY when the last candle close is >0.8% above the
10-candle SMA — the root cause of all <20min SL hits on Jun 3 2026
(SEI, BIO, GIGGLE, DASH, AVAX all bought at local spike tops).
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core_engine.implementations import DecisionEngineImpl
from core_engine.native.quant_reasoning import Playbook


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_klines(closes: list[float]) -> list[dict]:
    return [{"close": c, "open": c, "high": c, "low": c} for c in closes]


def _permissive_playbook() -> Playbook:
    """A playbook that imposes no confidence floor and always allows BUYs."""
    return Playbook(
        name="NORMAL_TRADING",
        allow_buy=True,
        allow_sell=True,
        allow_rebalance=False,
        allow_dust_cleanup=False,
        max_trade_size_usdt=20.0,
        confidence_floor=0.0,
        reason="test",
    )


def _make_app_ctx(closes: list[float]) -> dict:
    ss = types.SimpleNamespace()
    ss.market_data = {("TESTUSDT", "1m"): _make_klines(closes)}
    ss.metrics = {
        "nav": 100.0,
        "locked_usdt": 20.0,
        "positions_count": 2,
        "session_elapsed_h": 1.0,
        "win_rate_window": 0.5,
        "avg_fee_bps": 10.0,
        "avg_slippage_bps": 0.0,
        "avg_net_profit_bps": 5.0,
        "last_update_ts": 0.0,
    }
    ss.positions = {}
    ss.prices = {"TESTUSDT": closes[-1]}
    ss.current_mode = "normal"
    ss.runtime_overrides = {}
    ss.session_anchor_nav = 100.0

    arb = MagicMock()
    arb.evaluate = AsyncMock(return_value={
        "passed": True,
        "blocking_gates": [],
        "gates_status": {},
        "symbol_size_mult": 1.0,
        "reason": "",
        "regime_floor_bump": 0.0,
    })
    arb.get_dynamic_floor_delta = MagicMock(return_value=0.0)

    cap_alloc = MagicMock()
    cap_alloc.allocate_for_buy = AsyncMock(return_value=15.0)

    return {
        "shared_state": ss,
        "arbitration_engine": arb,
        "capital_allocator": cap_alloc,
    }


async def _buy_decision(closes: list[float], edge: float = 0.85) -> object:
    ctx = _make_app_ctx(closes)
    pb = _permissive_playbook()

    with patch("core_engine.implementations.select_playbook", return_value=pb), \
         patch("core_engine.implementations.compute_probability_score", return_value=edge), \
         patch("core_engine.implementations.DecisionEngineImpl.get_current_mode",
               new_callable=AsyncMock, return_value="normal"):
        return await DecisionEngineImpl.make_buy_decision(ctx, "TESTUSDT", edge)


# ── close sequence builders ───────────────────────────────────────────────────

def _flat_then_spike(base: float, spike_pct: float, n: int = 10) -> list[float]:
    return [base] * (n - 1) + [base * (1 + spike_pct / 100)]


def _flat_then_dip(base: float, dip_pct: float, n: int = 10) -> list[float]:
    return [base] * (n - 1) + [base * (1 - dip_pct / 100)]


def _rising_trend(base: float, step_pct: float, n: int = 10) -> list[float]:
    return [base * (1 + step_pct / 100) ** i for i in range(n)]


# ── tests ─────────────────────────────────────────────────────────────────────

class TestOverextensionGuard:

    @pytest.mark.asyncio
    async def test_spike_0_9pct_above_sma_is_blocked(self):
        closes = _flat_then_spike(base=1.0, spike_pct=0.9)
        dec = await _buy_decision(closes)
        assert not dec.allowed, f"Expected BLOCKED, got allowed=True reason={dec.blocked_reason}"
        assert "PRICE_EXTENDED" in (dec.blocked_reason or ""), dec.blocked_reason

    @pytest.mark.asyncio
    async def test_spike_1_5pct_above_sma_is_blocked(self):
        closes = _flat_then_spike(base=0.0345, spike_pct=1.5)
        dec = await _buy_decision(closes)
        assert not dec.allowed
        assert "PRICE_EXTENDED" in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_spike_0_5pct_above_sma_is_not_blocked_by_extension(self):
        """0.5% above SMA — within tolerance, extension check must not fire."""
        closes = _flat_then_spike(base=1.0, spike_pct=0.5)
        dec = await _buy_decision(closes)
        assert "PRICE_EXTENDED" not in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_steady_uptrend_not_blocked_by_extension(self):
        """Gradual uptrend 0.05%/candle — price above SMA but never spiked."""
        closes = _rising_trend(base=1.0, step_pct=0.05)
        dec = await _buy_decision(closes)
        assert "PRICE_EXTENDED" not in (dec.blocked_reason or ""), dec.blocked_reason

    @pytest.mark.asyncio
    async def test_falling_price_blocked_by_momentum_check(self):
        """Falling price blocked by MOMENTUM_FALLING (not overextension)."""
        closes = _flat_then_dip(base=1.0, dip_pct=0.5)
        dec = await _buy_decision(closes)
        assert not dec.allowed
        reason = dec.blocked_reason or ""
        assert "MOMENTUM_FALLING" in reason or "MOMENTUM_FLAT" in reason

    @pytest.mark.asyncio
    async def test_jun3_biousdt_pattern(self):
        """
        BIO Jun 3: ~0.033 base, spike to 0.0345 (+3.6%) → -$0.31 SL in 11min.
        Must be blocked by overextension guard.
        """
        closes = [0.0333] * 9 + [0.0345]
        dec = await _buy_decision(closes)
        assert not dec.allowed
        assert "PRICE_EXTENDED" in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_jun3_dashusdt_pattern(self):
        """
        DASH Jun 3: ~39.5 base, spike to 40.10 (+1.5%) → -$0.24 SL in 20min.
        """
        closes = [39.5] * 9 + [40.10]
        dec = await _buy_decision(closes)
        assert not dec.allowed
        assert "PRICE_EXTENDED" in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_jun3_avaxusdt_pattern(self):
        """
        AVAX Jun 3: ~8.26 base, spike to 8.344 (+1.0%) → -$0.10 SL in 13min.
        """
        closes = [8.26] * 9 + [8.344]
        dec = await _buy_decision(closes)
        assert not dec.allowed
        assert "PRICE_EXTENDED" in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_no_kline_data_allows_through(self):
        """When kline data missing, gate must fail open (not block trading)."""
        ctx = _make_app_ctx([1.0] * 10)
        ctx["shared_state"].market_data = {}

        pb = _permissive_playbook()
        with patch("core_engine.implementations.select_playbook", return_value=pb), \
             patch("core_engine.implementations.compute_probability_score", return_value=0.85), \
             patch("core_engine.implementations.DecisionEngineImpl.get_current_mode",
                   new_callable=AsyncMock, return_value="normal"):
            dec = await DecisionEngineImpl.make_buy_decision(ctx, "TESTUSDT", 0.85)

        assert "PRICE_EXTENDED" not in (dec.blocked_reason or "")

    @pytest.mark.asyncio
    async def test_flat_price_below_sma_blocked_by_flat_check(self):
        """Flat price below SMA blocked by MOMENTUM_FLAT, not overextension."""
        closes = [1.0] * 8 + [0.998, 0.997]  # slight dip at end, below SMA
        dec = await _buy_decision(closes)
        assert not dec.allowed
        assert "PRICE_EXTENDED" not in (dec.blocked_reason or "")
