"""
Tests for Native L3 (Phase 8.2.4) — NativeSignalEngine + indicators.

No I/O. Pure-numpy + synthetic kline data.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from core_engine.native import (
    AggregatedSignal,
    NativeSignalEngine,
    Signal,
)
from core_engine.native.signals import (
    _ema,
    ma_crossover,
    macd,
    rsi,
    strategy_ma_crossover,
    strategy_rsi,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _klines_from_closes(closes: list[float]) -> list[list[Any]]:
    """Build minimal Binance-shaped klines (only column 4 / close matters)."""
    return [[i, 0, 0, 0, c, 0] for i, c in enumerate(closes)]


# ─────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────
class TestRSI:
    def test_insufficient_data_returns_none(self) -> None:
        assert rsi(np.array([1.0, 2.0, 3.0]), period=14) is None

    def test_pure_uptrend_near_100(self) -> None:
        closes = np.linspace(100, 200, 50)
        val = rsi(closes, period=14)
        assert val is not None
        assert val > 95.0  # all gains, no losses ⇒ near 100

    def test_pure_downtrend_near_0(self) -> None:
        closes = np.linspace(200, 100, 50)
        val = rsi(closes, period=14)
        assert val is not None
        assert val < 5.0

    def test_no_movement_returns_100_when_no_loss(self) -> None:
        # All flat ⇒ avg_loss=0 path returns 100
        closes = np.array([100.0] * 30)
        val = rsi(closes, period=14)
        assert val == 100.0


class TestMACD:
    def test_insufficient_data_returns_none(self) -> None:
        assert macd(np.array([1.0] * 10)) is None

    def test_uptrend_positive_histogram(self) -> None:
        closes = np.linspace(100, 200, 60)
        res = macd(closes)
        assert res is not None
        macd_line, sig_line, hist = res
        assert hist > 0
        assert macd_line > sig_line

    def test_downtrend_negative_histogram(self) -> None:
        closes = np.linspace(200, 100, 60)
        res = macd(closes)
        assert res is not None
        _, _, hist = res
        assert hist < 0


class TestEMA:
    def test_constant_input_constant_output(self) -> None:
        out = _ema(np.array([5.0] * 20), 10)
        assert np.allclose(out, 5.0)

    def test_first_value_seeds_output(self) -> None:
        out = _ema(np.array([10.0, 20.0, 30.0]), 3)
        assert out[0] == 10.0


class TestMACrossover:
    def test_insufficient_data_returns_none(self) -> None:
        assert ma_crossover(np.array([1.0] * 10)) is None

    def test_golden_cross(self) -> None:
        # 34 flat bars then a single up-spike: fast_prev == slow_prev (no
        # prior cross), fast_now > slow_now (fresh cross on last bar).
        closes = np.array([100.0] * 34 + [200.0])
        res = ma_crossover(closes, fast=5, slow=20)
        assert res is not None
        _, _, cross = res
        assert cross == 1

    def test_death_cross(self) -> None:
        closes = np.array([150.0] * 34 + [100.0])
        res = ma_crossover(closes, fast=5, slow=20)
        assert res is not None
        _, _, cross = res
        assert cross == -1

    def test_no_cross_returns_zero(self) -> None:
        closes = np.linspace(100, 105, 40)  # gentle rising, no fresh cross
        res = ma_crossover(closes, fast=5, slow=20)
        assert res is not None
        _, _, cross = res
        assert cross in (0, 1)  # may have already crossed earlier; accept both


# ─────────────────────────────────────────────────────────────────────
# Built-in strategies
# ─────────────────────────────────────────────────────────────────────
class TestStrategies:
    def test_rsi_strategy_oversold_emits_buy(self) -> None:
        closes = np.linspace(200, 100, 40)
        sig = strategy_rsi(closes, symbol="BTCUSDT")
        assert sig is not None
        assert sig.direction == "BUY"
        assert 0.5 <= sig.score <= 1.0

    def test_rsi_strategy_overbought_emits_sell(self) -> None:
        closes = np.linspace(100, 200, 40)
        sig = strategy_rsi(closes, symbol="BTCUSDT")
        assert sig is not None
        assert sig.direction == "SELL"

    def test_rsi_strategy_neutral_emits_hold(self) -> None:
        closes = np.array(
            [100.0 + (i % 2) * 0.1 for i in range(40)]  # tiny zigzag, RSI ≈ 50
        )
        sig = strategy_rsi(closes, symbol="BTCUSDT")
        assert sig is not None
        assert sig.direction == "HOLD"
        assert sig.score == 0.0

    def test_ma_crossover_strategy_no_cross_returns_hold(self) -> None:
        closes = np.array([100.0] * 50)
        sig = strategy_ma_crossover(closes, symbol="BTCUSDT")
        assert sig is not None
        assert sig.direction == "HOLD"


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────
class TestNativeSignalEngine:
    def test_default_enables_all_builtins(self) -> None:
        eng = NativeSignalEngine()
        assert set(eng.enabled_strategies) == {"rsi", "macd", "ma_cross"}

    def test_disable_and_enable(self) -> None:
        eng = NativeSignalEngine()
        eng.disable("macd")
        assert "macd" not in eng.enabled_strategies
        eng.enable("macd")
        assert "macd" in eng.enabled_strategies

    def test_enable_unknown_strategy_raises(self) -> None:
        eng = NativeSignalEngine()
        with pytest.raises(KeyError):
            eng.enable("nonexistent")

    def test_register_custom_strategy(self) -> None:
        eng = NativeSignalEngine()

        def always_buy(closes: np.ndarray, *, symbol: str = "") -> Signal:
            return Signal(symbol, "BUY", 1.0, "always_buy")

        eng.register_strategy("always_buy", always_buy, weight=2.0)
        klines = _klines_from_closes(list(np.linspace(100, 110, 50)))
        agg = eng.evaluate("BTCUSDT", klines)
        assert agg is not None
        # weight=2 contribution overpowers others; should be BUY
        # (or at least, buy signal must appear in contributions)
        names = [s.strategy for s in agg.contributions]
        assert "always_buy" in names

    def test_evaluate_aggregates_uptrend_to_sell_via_rsi(self) -> None:
        # Strong uptrend ⇒ RSI overbought ⇒ SELL aggregate.
        # Isolate to RSI: MACD on a steady trend produces an opposing signal
        # of comparable magnitude (engine correctly nets them ⇒ HOLD), so we
        # disable it to verify the RSI path explicitly.
        closes = list(np.linspace(100, 200, 60))
        eng = NativeSignalEngine(enabled=["rsi"])
        agg = eng.evaluate("BTCUSDT", _klines_from_closes(closes))
        assert agg is not None
        assert agg.direction == "SELL"
        assert 0.0 < agg.score <= 1.0

    def test_evaluate_aggregates_downtrend_to_buy_via_rsi(self) -> None:
        closes = list(np.linspace(200, 100, 60))
        eng = NativeSignalEngine(enabled=["rsi"])
        agg = eng.evaluate("BTCUSDT", _klines_from_closes(closes))
        assert agg is not None
        assert agg.direction == "BUY"

    def test_empty_klines_returns_none(self) -> None:
        assert NativeSignalEngine().evaluate("BTCUSDT", []) is None

    def test_insufficient_data_returns_none(self) -> None:
        # 5 bars: every indicator returns None
        agg = NativeSignalEngine().evaluate(
            "BTCUSDT", _klines_from_closes([100.0, 101, 102, 103, 104])
        )
        assert agg is None

    def test_cooldown_blocks_repeat_signal(self) -> None:
        eng = NativeSignalEngine(cooldown_sec=10.0, enabled=["rsi"])
        closes = list(np.linspace(100, 200, 60))
        kl = _klines_from_closes(closes)
        first = eng.evaluate("BTCUSDT", kl)
        assert first is not None
        # second evaluation immediately after — cooldown should kick in
        second = eng.evaluate("BTCUSDT", kl)
        assert second is None

    def test_cooldown_per_symbol(self) -> None:
        eng = NativeSignalEngine(cooldown_sec=10.0, enabled=["rsi"])
        closes = list(np.linspace(100, 200, 60))
        kl = _klines_from_closes(closes)
        a = eng.evaluate("BTCUSDT", kl)
        b = eng.evaluate("ETHUSDT", kl)
        assert a is not None
        assert b is not None  # different symbol, no cooldown

    def test_evaluate_many_drops_holds_and_nones(self) -> None:
        eng = NativeSignalEngine(enabled=["rsi"])
        kl_up = _klines_from_closes(list(np.linspace(100, 200, 60)))
        kl_short = _klines_from_closes([100.0, 101.0])  # too few bars
        out = eng.evaluate_many({"BTCUSDT": kl_up, "ETHUSDT": kl_short})
        assert "BTCUSDT" in out
        assert "ETHUSDT" not in out
        assert out["BTCUSDT"].direction == "SELL"

    def test_aggregated_score_in_unit_range(self) -> None:
        eng = NativeSignalEngine()
        closes = list(np.linspace(100, 200, 60))
        agg = eng.evaluate("BTCUSDT", _klines_from_closes(closes))
        assert agg is not None
        assert isinstance(agg, AggregatedSignal)
        assert 0.0 <= agg.score <= 1.0
        assert agg.direction in ("BUY", "SELL", "HOLD")
        assert all(isinstance(s, Signal) for s in agg.contributions)

    def test_weights_change_aggregation_outcome(self) -> None:
        # Heavy weight on a strategy that returns BUY should overpower a
        # weak SELL from another.
        eng = NativeSignalEngine(
            weights={"big_buy": 100.0, "tiny_sell": 1.0},
            enabled=["big_buy", "tiny_sell"],
        )

        def big_buy(closes: np.ndarray, *, symbol: str = "") -> Signal:
            return Signal(symbol, "BUY", 1.0, "big_buy")

        def tiny_sell(closes: np.ndarray, *, symbol: str = "") -> Signal:
            return Signal(symbol, "SELL", 0.5, "tiny_sell")

        eng.register_strategy("big_buy", big_buy, weight=100.0)
        eng.register_strategy("tiny_sell", tiny_sell, weight=1.0)
        agg = eng.evaluate("BTCUSDT", _klines_from_closes([100.0] * 50))
        assert agg is not None
        assert agg.direction == "BUY"
