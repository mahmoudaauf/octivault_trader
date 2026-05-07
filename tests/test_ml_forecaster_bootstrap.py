from __future__ import annotations

import pandas as pd
import pytest

from agents import ml_forecaster as ml_mod
from agents.ml_forecaster import MLForecaster


class _Cfg:
    MLF_TRAIN_TIMEOUT_S = 5.0
    MLF_MAX_BACKGROUND_TRAINS = 1


class _ModelManager:
    def build_model_path(self, **_: object) -> str:
        return "dummy-model.keras"

    def model_exists(self, _path: str) -> bool:
        return False


class _SharedState:
    pass


class _DummyTrainer:
    def __init__(self, **_: object) -> None:
        self.persisted = False

    def train_model(self, df: pd.DataFrame, **_: object) -> dict[str, object]:
        return {
            "ok": True,
            "rows": len(df),
            "epochs": 1,
            "val_accuracy": 0.7,
        }

    def persist_model(self, model_path: str) -> bool:
        self.persisted = bool(model_path)
        return True


@pytest.mark.asyncio
async def test_startup_full_training_allows_bootstrap_row_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ml_mod, "ModelTrainer", _DummyTrainer)

    forecaster = MLForecaster(
        shared_state=_SharedState(),
        execution_manager=None,
        config=_Cfg(),
        model_manager=_ModelManager(),
        symbols=["BNBUSDT"],
    )

    async def _fetch_training_ohlcv(*_: object, **__: object) -> list[list[int]]:
        return [[0, 1, 1, 1, 1, 1]] * 180

    forecaster._fetch_training_ohlcv = _fetch_training_ohlcv
    forecaster._build_train_df_from_ohlcv = lambda ohlcv, _max_rows: pd.DataFrame(
        {
            "open": [1.0] * len(ohlcv),
            "high": [1.0] * len(ohlcv),
            "low": [1.0] * len(ohlcv),
            "close": [1.0] * len(ohlcv),
            "volume": [1.0] * len(ohlcv),
        }
    )
    forecaster._passes_retrain_quality_guard = lambda *_, **__: (True, "ok")
    forecaster._refresh_model_cache_for_path = lambda _path: True

    scheduled, state = forecaster._schedule_startup_full_training(
        symbol="BNBUSDT",
        timeframe="5m",
        lookback=60,
        model_path="dummy-model.keras",
        reason="startup_missing_model",
    )

    assert scheduled is True
    assert state == "full_training_queued"

    task = forecaster._train_tasks["dummy-model.keras"]
    await task

    assert "dummy-model.keras" in forecaster._startup_full_train_done
