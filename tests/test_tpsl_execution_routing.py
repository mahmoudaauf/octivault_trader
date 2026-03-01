import asyncio
from types import SimpleNamespace

import pytest

from core.tp_sl_engine import TPSLEngine


class _ExecutionManagerStub:
    def __init__(self):
        self.calls = []

    async def close_position(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "ok": True,
            "status": "filled",
            "executedQty": 0.01,
            "symbol": kwargs.get("symbol"),
        }


def _shared_state_stub():
    return SimpleNamespace(
        open_trades={},
        latest_prices={},
        sentiment_score={},
        volatility_state={},
        dynamic_config={},
        market_data={},
        metrics={},
        total_equity=0.0,
        symbol_filters={},
    )


def _config_stub(enforce_em_only: bool = True):
    return SimpleNamespace(
        TPSL_ENFORCE_EXECUTION_MANAGER_ONLY=bool(enforce_em_only),
    )


def test_tpsl_close_routes_only_via_execution_manager():
    em = _ExecutionManagerStub()
    engine = TPSLEngine(
        shared_state=_shared_state_stub(),
        config=_config_stub(enforce_em_only=True),
        execution_manager=em,
    )

    res = asyncio.run(
        engine._close_via_execution_manager(
            "BTCUSDT",
            "TP_HIT",
            force_finalize=True,
            tag="tp_sl",
        )
    )

    assert res.get("ok") is True
    assert len(em.calls) == 1
    assert em.calls[0]["symbol"] == "BTCUSDT"
    assert em.calls[0]["reason"] == "TP_HIT"
    assert em.calls[0]["force_finalize"] is True
    assert em.calls[0]["tag"] == "tp_sl"


def test_tpsl_close_raises_when_execution_manager_missing():
    engine = TPSLEngine(
        shared_state=_shared_state_stub(),
        config=_config_stub(enforce_em_only=True),
        execution_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="ExecutionManager.close_position"):
        asyncio.run(
            engine._close_via_execution_manager(
                "BTCUSDT",
                "TP_HIT",
                force_finalize=True,
                tag="tp_sl",
            )
        )
