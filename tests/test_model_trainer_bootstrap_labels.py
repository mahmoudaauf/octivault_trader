from __future__ import annotations

import numpy as np
import pandas as pd

from src.l5_strategy.model_trainer import ModelTrainer
from src.l5_strategy import model_trainer as trainer_mod


class _DummyHistory:
    history = {
        "loss": [0.5],
        "accuracy": [0.6],
        "val_loss": [0.5],
        "val_accuracy": [0.6],
    }


class _DummyModel:
    name = "dummy"

    def fit(self, **_: object) -> _DummyHistory:
        return _DummyHistory()


def test_train_model_uses_median_split_when_labels_are_degenerate() -> None:
    rows = 180
    close = np.linspace(100.0, 101.79, rows)
    df = pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": np.linspace(1000.0, 1200.0, rows),
        }
    )

    trainer = ModelTrainer(symbol="TESTUSDT", input_lookback=60, epochs=1)
    trainer.use_triple_barrier_labels = False
    trainer.regime_aware_labels_enabled = False
    trainer.label_threshold_pct = 10.0
    trainer.persist_model = lambda *_, **__: True
    trainer._build_model = lambda *_: _DummyModel()

    result = trainer.train_model(
        df,
        task="supervised_learning",
        epochs=1,
        save_model_artifact=False,
        return_metrics=True,
    )

    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert result.get("reason") == "trained"


def test_train_model_uses_rank_split_when_returns_are_flat() -> None:
    rows = 180
    close = np.full(rows, 100.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.linspace(1000.0, 1200.0, rows),
        }
    )

    trainer = ModelTrainer(symbol="FLATUSDT", input_lookback=60, epochs=1)
    trainer.use_triple_barrier_labels = False
    trainer.regime_aware_labels_enabled = False
    trainer.label_threshold_pct = 10.0
    trainer.persist_model = lambda *_, **__: True
    trainer._build_model = lambda *_: _DummyModel()

    result = trainer.train_model(
        df,
        task="supervised_learning",
        epochs=1,
        save_model_artifact=False,
        return_metrics=True,
    )

    assert isinstance(result, dict)
    assert result.get("ok") is True
    assert result.get("reason") == "trained"


def test_build_optimizer_falls_back_when_legacy_adam_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = ModelTrainer(symbol="OPTUSDT", input_lookback=60, epochs=1)

    class _FallbackAdam:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def _boom(**_: object) -> object:
        raise ImportError("legacy unsupported")

    monkeypatch.setattr(trainer_mod, "Adam", _FallbackAdam)
    monkeypatch.setattr(trainer_mod, "LegacyAdam", _boom)
    monkeypatch.setenv("ML_TRAIN_USE_LEGACY_ADAM", "true")

    optimizer = trainer._build_optimizer()
    assert isinstance(optimizer, _FallbackAdam)
