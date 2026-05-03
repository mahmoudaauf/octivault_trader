from __future__ import annotations

import types

import pytest

from src.l8_lifecycle.meta_controller import MetaController


def _controller() -> MetaController:
    mc = MetaController.__new__(MetaController)
    mc.cycles_no_trade = 42
    mc.portfolio_manager = types.SimpleNamespace(
        positions={
            "HIGHCONFUSDT": {
                "state": "DUST_LOCKED",
                "quantity": 2.0,
                "value_usdt": 2.0,
                "created_at": 100,
                "confidence": 0.90,
                "notional_usdt": 2.0,
            },
            "LOWCONFUSDT": {
                "state": "DUST_LOCKED",
                "quantity": 3.0,
                "value_usdt": 3.0,
                "created_at": 100,
                "confidence": 0.20,
                "notional_usdt": 3.0,
            },
            "NEWERUSDT": {
                "state": "DUST_LOCKED",
                "qty": 1.0,
                "value_usdt": 1.5,
                "created_at": 200,
                "confidence": 0.01,
                "notional_usdt": 1.5,
            },
        }
    )
    mc.shared_state = types.SimpleNamespace(is_permanent_dust=lambda symbol: False)
    mc.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    mc._cfg = lambda key, default=None: default  # type: ignore[assignment]
    return mc


@pytest.mark.asyncio
async def test_dust_exit_candidate_prefers_oldest_lowest_confidence_and_quantity_key():
    mc = _controller()

    selected = await mc._select_dust_exit_candidate()

    assert selected == "LOWCONFUSDT"
