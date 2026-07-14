"""
Tests for core_engine/native/math_utils.py's ATR functions (Priority 3 item #10).

These were moved out of NativeTPSLEngine so NativeCapitalAllocator could share
the same real ATR computation instead of a hardcoded 0.008 placeholder.
"""
from __future__ import annotations

from core_engine.native import math_utils


class _SS:
    def __init__(self, market_data=None, prices=None, klines=None):
        self.market_data = market_data or {}
        self.prices = prices or {}
        self.klines = klines or {}


def test_compute_atr_from_candles_dict_format():
    candles = [
        {"high": 101.0, "low": 99.0, "close": 100.0},
        {"high": 102.0, "low": 100.0, "close": 101.0},
        {"high": 103.0, "low": 101.0, "close": 102.0},
    ]
    atr = math_utils.compute_atr_from_candles(candles, lookback=3)
    assert atr > 0


def test_compute_atr_from_candles_legacy_list_format():
    # [ts, open, high, low, close, ...]
    candles = [
        [0, 100.0, 101.0, 99.0, 100.0],
        [1, 100.0, 102.0, 100.0, 101.0],
        [2, 101.0, 103.0, 101.0, 102.0],
    ]
    atr = math_utils.compute_atr_from_candles(candles, lookback=3)
    assert atr > 0


def test_compute_atr_from_candles_too_few_returns_zero():
    assert math_utils.compute_atr_from_candles([{"high": 1, "low": 1, "close": 1}]) == 0.0


def test_compute_atr_primary_source_websocket_candles():
    candles = [{"high": 101.0, "low": 99.0, "close": 100.0} for _ in range(20)]
    ss = _SS(market_data={("BTCUSDT", "1m"): candles})
    atr = math_utils.compute_atr(ss, "BTCUSDT")
    assert atr == 2.0  # constant high-low range of 2.0 every bar


def test_compute_atr_legacy_cached_scalar():
    ss = _SS(market_data={"BTCUSDT": {"atr": 5.0}})
    assert math_utils.compute_atr(ss, "BTCUSDT") == 5.0


def test_compute_atr_legacy_klines_attribute():
    candles = [{"high": 101.0, "low": 99.0, "close": 100.0} for _ in range(5)]
    ss = _SS(klines={"BTCUSDT": {"1m": candles}})
    atr = math_utils.compute_atr(ss, "BTCUSDT")
    assert atr == 2.0


def test_compute_atr_no_data_falls_back_to_price_pct():
    ss = _SS(prices={"BTCUSDT": 100.0})
    assert math_utils.compute_atr(ss, "BTCUSDT") == 0.8  # 0.8% of 100.0


def test_compute_atr_no_data_no_price_returns_zero():
    ss = _SS()
    assert math_utils.compute_atr(ss, "BTCUSDT") == 0.0


def test_compute_atr_pct_normalizes_by_current_price():
    candles = [{"high": 101.0, "low": 99.0, "close": 100.0} for _ in range(20)]
    ss = _SS(market_data={("BTCUSDT", "1m"): candles}, prices={"BTCUSDT": 100.0})
    assert math_utils.compute_atr_pct(ss, "BTCUSDT") == 0.02


def test_compute_atr_pct_zero_when_no_price_available():
    candles = [{"high": 101.0, "low": 99.0, "close": 100.0} for _ in range(20)]
    ss = _SS(market_data={("BTCUSDT", "1m"): candles})  # no prices
    assert math_utils.compute_atr_pct(ss, "BTCUSDT") == 0.0
