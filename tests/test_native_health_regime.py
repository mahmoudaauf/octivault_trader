from __future__ import annotations

import pytest

from core_engine.implementations import OperationsEngineImpl, SituationEngineImpl
from core_engine.native.health_monitor import NativeHealthMonitor
from core_engine.native.market_regime_detector import NativeMarketRegimeDetector


class _State:
    def __init__(self) -> None:
        self.trading_halted = False
        self.exchange_throttled = False
        self.exchange_throttle_reason = ""
        self.market_data_ready = True
        self.prices = {"BTCUSDT": 101.0}
        self.nav_usdt = 125.0
        self.current_mode = "NORMAL"
        self.metrics = {"peak_nav": 120.0}
        self.session_anchor_nav = 100.0


class _MarketData:
    def __init__(self, closes: list[float]) -> None:
        self._klines = {
            ("BTCUSDT", "1m", 64): (0.0, [[0, 0, 0, 0, c, 0] for c in closes]),
        }


@pytest.mark.asyncio
async def test_market_regime_detector_reports_uptrend_growth() -> None:
    detector = NativeMarketRegimeDetector(
        market_data=_MarketData([100 + i for i in range(30)]),
        shared_state=_State(),
    )
    regime = await detector.get_regime()
    assert regime["trend_regime"] == "UPTREND"
    assert regime["nav_regime"] == "GROWTH"
    assert regime["overall_health"] == "OK"


@pytest.mark.asyncio
async def test_situation_engine_impl_uses_native_market_regime_detector() -> None:
    app_ctx = {
        "market_regime_detector": NativeMarketRegimeDetector(
            market_data=_MarketData([100 + i for i in range(30)]),
            shared_state=_State(),
        )
    }
    regime = await SituationEngineImpl.get_market_regime(app_ctx)
    assert regime["trend_regime"] == "UPTREND"
    assert regime["overall_health"] == "OK"


@pytest.mark.asyncio
async def test_health_monitor_reports_warning_when_throttled() -> None:
    state = _State()
    state.exchange_throttled = True
    state.exchange_throttle_reason = "binance 418"
    monitor = NativeHealthMonitor(shared_state=state, watchdog=object())
    report = await monitor.get_report()
    assert report["overall_status"] == "WARN"
    assert "exchange_api:binance 418" in report["warnings"]


@pytest.mark.asyncio
async def test_operations_impl_uses_native_health_monitor() -> None:
    monitor = NativeHealthMonitor(shared_state=_State(), watchdog=object())
    report = await OperationsEngineImpl.get_health_report({"health_monitor": monitor})
    assert report["overall_status"] == "OK"
    assert "exchange_api" in report["components"]
