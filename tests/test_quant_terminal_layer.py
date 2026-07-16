from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from core_engine.decision_engine import DecisionEngine, TradeDecision
from core_engine.native.legacy_signal_adapter import LegacySignalAdapter
from core_engine.operations_engine import OperationsEngine
from core_engine.safe_execution_engine import SafeExecutionEngine
from core_engine.situation_engine import SituationEngine


class _PortfolioManager:
    def __init__(
        self, *, nav: float, free: float, locked: float, positions: dict[str, Any] | None = None
    ):
        self._nav = nav
        self._free = free
        self._locked = locked
        self._positions = positions or {}

    async def get_nav(self) -> float:
        return self._nav

    async def get_positions(self) -> dict[str, Any]:
        return self._positions

    async def get_capital_allocated(self) -> float:
        return self._locked

    async def get_capital_available(self) -> float:
        return self._free


class _HealthMonitor:
    def __init__(self, overall_status: str = "OK", exchange_status: str = "OK") -> None:
        self._overall_status = overall_status
        self._exchange_status = exchange_status

    async def get_report(self) -> dict[str, Any]:
        return {
            "overall_status": self._overall_status,
            "components": {
                "exchange_api": {"status": self._exchange_status},
            },
        }


class _RegimeDetector:
    def __init__(
        self, trend: str = "UPTREND", volatility: str = "NORMAL", health: str = "OK"
    ) -> None:
        self._trend = trend
        self._volatility = volatility
        self._health = health

    async def get_regime(self) -> dict[str, str]:
        return {
            "volatility_regime": self._volatility,
            "trend_regime": self._trend,
            "nav_regime": "GROWTH",
            "overall_health": self._health,
        }


class _ModeManager:
    async def get_current_mode(self) -> str:
        return "NORMAL"


class _CapitalAllocator:
    def __init__(self, quote: float = 50.0) -> None:
        self.quote = quote

    async def allocate_for_buy(self, symbol: str) -> float:
        return self.quote


class _Arbitration:
    async def evaluate(self, symbol: str, signal_type: str, edge_score: float) -> dict[str, Any]:
        return {"passed": True, "gates_status": {}, "blocking_gates": [], "reason": "ok"}


class _ExecutionManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def place_order(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float | None,
        action: str,
        order_type: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "action": action,
                "order_type": order_type,
            }
        )
        return {
            "orderId": "test-order-1",
            "executedQty": quantity,
            "price": price or 0.0,
            "status": "FILLED",
        }


class _SharedState:
    def __init__(
        self,
        *,
        nav_usdt: float = 100.0,
        free_balance_usdt: float = 50.0,
        invested_capital_usdt: float = 50.0,
        positions: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        current_mode: str = "NORMAL",
    ) -> None:
        self.nav_usdt = nav_usdt
        self.free_balance_usdt = free_balance_usdt
        self.invested_capital_usdt = invested_capital_usdt
        self.positions = positions or {}
        self.metrics = metrics or {}
        self.trade_history: dict[str, list[Any]] = {}
        self.current_mode = current_mode
        self.exchange_throttled = False

    def reserved_quote_total(self, asset: str) -> float:
        return 0.0


class _PositionManager:
    def __init__(self, position: Any | None = None) -> None:
        self._position = position

    async def get_position(self, symbol: str) -> Any | None:
        return self._position


class _TpSlEngine:
    """Minimal stub exposing only what make_sell_decision() consults: the
    armed entry-price registry that should override a stale/corrupted
    position.entry_price (see get_entry_price() on the real engine)."""

    def __init__(self, entry_prices: dict[str, float] | None = None) -> None:
        self._entry_prices = entry_prices or {}

    def get_entry_price(self, symbol: str) -> float:
        return float(self._entry_prices.get(symbol, 0.0) or 0.0)


def _make_position(*, qty: float = 1.0, entry: float = 100.0, mark: float = 110.0) -> Any:
    return SimpleNamespace(qty=qty, entry_price=entry, mark_price=mark)


def _make_app_ctx(
    *,
    nav: float = 100.0,
    free: float = 50.0,
    locked: float | None = None,
    positions: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    health_status: str = "OK",
    exchange_status: str = "OK",
    trend: str = "UPTREND",
    volatility: str = "NORMAL",
    regime_health: str = "OK",
    allocator_quote: float = 50.0,
    position_for_symbol: Any | None = None,
    tp_sl_engine: Any | None = None,
) -> dict[str, Any]:
    locked_capital = float(locked if locked is not None else max(nav - free, 0.0))
    shared_state = _SharedState(
        nav_usdt=nav,
        free_balance_usdt=free,
        invested_capital_usdt=locked_capital,
        positions=positions,
        metrics=metrics,
    )
    return {
        "portfolio_manager": _PortfolioManager(
            nav=nav,
            free=free,
            locked=locked_capital,
            positions=positions,
        ),
        "health_monitor": _HealthMonitor(health_status, exchange_status),
        "market_regime_detector": _RegimeDetector(trend, volatility, regime_health),
        "shared_state": shared_state,
        "mode_manager": _ModeManager(),
        "capital_allocator": _CapitalAllocator(allocator_quote),
        "arbitration_engine": _Arbitration(),
        "execution_manager": _ExecutionManager(),
        "bounded_cache": {},
        "position_manager": _PositionManager(position_for_symbol),
        "tp_sl_engine": tp_sl_engine,
    }


@pytest.mark.asyncio
async def test_situation_state_classifies_cash_heavy() -> None:
    engine = SituationEngine(_make_app_ctx(nav=100.0, free=80.0, locked=20.0))
    situation = await engine.get_situation_state()
    assert situation["portfolio_state"] == "CASH_HEAVY"


@pytest.mark.asyncio
async def test_situation_state_classifies_low_usdt() -> None:
    engine = SituationEngine(_make_app_ctx(nav=100.0, free=5.0, locked=95.0))
    situation = await engine.get_situation_state()
    assert situation["portfolio_state"] == "LOW_USDT"
    assert situation["capital_state"] == "LOW_FREE_USDT"


@pytest.mark.asyncio
async def test_situation_state_classifies_overexposed() -> None:
    engine = SituationEngine(_make_app_ctx(nav=100.0, free=15.0, locked=85.0))
    situation = await engine.get_situation_state()
    assert situation["portfolio_state"] == "OVEREXPOSED"


@pytest.mark.asyncio
async def test_situation_state_classifies_dust_heavy() -> None:
    # Dust cutoff is MIN_NOTIONAL_USDT (default 5.0, exclusive) -- all four
    # positions below it, "B" (5.0, notional) does not count.
    positions = {
        "A": SimpleNamespace(qty=1.0, mark_price=4.0, entry_price=3.0),
        "B": SimpleNamespace(qty=1.0, mark_price=4.5, entry_price=4.0),
        "C": SimpleNamespace(qty=1.0, mark_price=4.9, entry_price=4.0),
        "D": SimpleNamespace(qty=1.0, mark_price=4.99, entry_price=4.0),
    }
    engine = SituationEngine(_make_app_ctx(nav=100.0, free=40.0, locked=60.0, positions=positions))
    situation = await engine.get_situation_state()
    assert situation["portfolio_state"] == "DUST_HEAVY"


@pytest.mark.asyncio
async def test_situation_state_classifies_system_degraded() -> None:
    engine = SituationEngine(_make_app_ctx(health_status="WARN", exchange_status="WARN"))
    situation = await engine.get_situation_state()
    assert situation["system_state"] == "DEGRADED"
    assert situation["risk_state"] == "DEFENSIVE"


@pytest.mark.asyncio
async def test_decision_engine_selects_system_pause_when_degraded() -> None:
    engine = DecisionEngine(_make_app_ctx(health_status="WARN", exchange_status="WARN"))
    decision = await engine.make_buy_decision("BTCUSDT", 0.8)
    assert decision is not None
    assert decision.playbook == "SYSTEM_PAUSE"
    assert decision.allowed is False
    assert decision.blocked_reason == "BUY_BLOCKED_BY_PLAYBOOK"


@pytest.mark.asyncio
async def test_decision_engine_selects_low_usdt_recovery() -> None:
    engine = DecisionEngine(_make_app_ctx(nav=100.0, free=5.0, locked=95.0))
    decision = await engine.make_buy_decision("BTCUSDT", 0.8)
    assert decision is not None
    assert decision.playbook == "LOW_USDT_RECOVERY"
    assert decision.allowed is False


@pytest.mark.asyncio
async def test_decision_engine_selects_cash_deployment() -> None:
    engine = DecisionEngine(_make_app_ctx(nav=100.0, free=80.0, locked=20.0, trend="UPTREND"))
    decision = await engine.make_buy_decision("BTCUSDT", 0.8)
    assert decision is not None
    assert decision.playbook == "CASH_DEPLOYMENT"
    assert decision.allowed is True
    assert decision.quantity > 0


@pytest.mark.asyncio
async def test_decision_engine_selects_normal_trading() -> None:
    engine = DecisionEngine(_make_app_ctx(nav=100.0, free=40.0, locked=60.0, trend="RANGING"))
    decision = await engine.make_buy_decision("BTCUSDT", 0.8)
    assert decision is not None
    assert decision.playbook == "NORMAL_TRADING"
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_decision_engine_make_sell_decision_uses_real_position_quantity() -> None:
    position = _make_position(qty=0.5, entry=100.0, mark=120.0)
    engine = DecisionEngine(_make_app_ctx(position_for_symbol=position))
    decision = await engine.make_sell_decision("BTCUSDT", 0.9, "signal")
    assert decision is not None
    assert decision.action == "SELL"
    assert decision.quantity == 0.5


@pytest.mark.asyncio
async def test_decision_engine_make_sell_decision_supports_dict_backed_positions() -> None:
    position = {"qty": 0.75, "entry_price": 100.0, "mark_price": 125.0}
    engine = DecisionEngine(_make_app_ctx(position_for_symbol=position))
    decision = await engine.make_sell_decision("BTCUSDT", 0.9, "signal")
    assert decision is not None
    assert decision.action == "SELL"
    assert decision.quantity == 0.75


@pytest.mark.asyncio
async def test_decision_engine_make_sell_decision_prefers_armed_entry_price() -> None:
    """Live incident 2026-07-16: polling_coordinator's fill-reconciliation averaged
    together fills from a previous, already-closed round trip on the same symbol,
    inflating position.entry_price far from the real fill price. Relative to a
    corrupted entry (150) a mark of 101 looks like a ~33% loss and would be
    blocked by the ordinary (non-forced) profit-after-fees gate; relative to the
    REAL armed entry (100) the same mark is a genuine, sellable +1% gain. Confirms
    make_sell_decision() prefers tp_sl_engine.get_entry_price() over the
    position's own (corruptible) entry_price field.
    """
    position = _make_position(qty=1.0, entry=150.0, mark=101.0)  # corrupted entry
    tp_sl_engine = _TpSlEngine({"BTCUSDT": 100.0})  # real armed entry
    engine = DecisionEngine(_make_app_ctx(position_for_symbol=position, tp_sl_engine=tp_sl_engine))
    decision = await engine.make_sell_decision("BTCUSDT", 0.9, "signal")
    assert decision is not None, (
        "expected a real SELL decision using the armed entry price (100 -> 101 is "
        "a genuine gain); got None, suggesting the corrupted entry_price (150) was "
        "used instead, which would look like a ~33% loss and get blocked"
    )
    assert decision.action == "SELL"


@pytest.mark.asyncio
async def test_safe_execution_blocks_buy_in_low_usdt_recovery() -> None:
    engine = SafeExecutionEngine(_make_app_ctx())
    result = await engine.execute_decision(
        TradeDecision(
            symbol="BTCUSDT",
            action="BUY",
            quantity=25.0,
            playbook="LOW_USDT_RECOVERY",
            allowed=False,
            blocked_reason="BUY_BLOCKED_BY_PLAYBOOK",
        )
    )
    assert result.success is False
    assert result.error_message == "BUY_BLOCKED_BY_PLAYBOOK"


@pytest.mark.asyncio
async def test_safe_execution_blocks_buy_in_overexposed_protection() -> None:
    engine = SafeExecutionEngine(_make_app_ctx())
    result = await engine.execute_decision(
        TradeDecision(
            symbol="BTCUSDT",
            action="BUY",
            quantity=25.0,
            playbook="OVEREXPOSED_PROTECTION",
            allowed=False,
            blocked_reason="BUY_BLOCKED_BY_PLAYBOOK",
        )
    )
    assert result.success is False
    assert result.status == "REJECTED"


@pytest.mark.asyncio
async def test_safe_execution_blocks_all_in_system_pause() -> None:
    engine = SafeExecutionEngine(_make_app_ctx())
    result = await engine.execute_decision(
        TradeDecision(
            symbol="BTCUSDT",
            action="SELL",
            quantity=1.0,
            playbook="SYSTEM_PAUSE",
            allowed=False,
            blocked_reason="SYSTEM_PAUSE",
        )
    )
    assert result.success is False
    assert result.error_message == "SYSTEM_PAUSE"


@pytest.mark.asyncio
async def test_operations_engine_emits_quant_loop_summary(caplog: pytest.LogCaptureFixture) -> None:
    engine = OperationsEngine(_make_app_ctx())
    summary = {
        "timestamp": 1.0,
        "nav_usdt": 100.0,
        "free_usdt": 50.0,
        "free_ratio": 0.5,
        "exposure_ratio": 0.5,
        "market_regime": "TRENDING",
        "portfolio_state": "BALANCED",
        "capital_state": "HEALTHY",
        "risk_state": "NORMAL",
        "system_state": "HEALTHY",
        "playbook": "NORMAL_TRADING",
        "action": "BUY",
        "symbol": "BTCUSDT",
        "probability_score": 0.75,
        "confidence": 0.8,
        "edge_score": 0.8,
        "allowed": True,
        "blocked_reason": "",
        "execution_result": "FILLED",
        "loop_duration_ms": 12.5,
    }
    with caplog.at_level(logging.INFO):
        await engine.log_event("QUANT_LOOP_SUMMARY", summary)
    assert "QUANT_LOOP_SUMMARY" in caplog.text
    for key in summary:
        assert key in caplog.text


def test_legacy_signal_adapter_uses_action_when_signal_type_missing() -> None:
    adapter = LegacySignalAdapter()
    normalized = adapter._normalize_forecaster_signals(
        [{"symbol": "BNBUSDT", "action": "SELL", "confidence": 0.92, "edge_score": 0.8}]
    )
    assert normalized[0]["signal_type"] == "SELL"
    assert normalized[0]["action"] == "SELL"
