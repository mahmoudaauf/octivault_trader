from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


class _PredictionOnlyModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.fit_called = False

    def fit(self, **_: object) -> _DummyHistory:
        self.fit_called = True
        raise AssertionError("holdout evaluation must not fit")

    def predict(self, windows: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.full((len(windows), 1), self.probability, dtype=np.float32)


def _holdout_frame(rows: int = 100, step_pct: float = 0.002) -> pd.DataFrame:
    close = 100.0 * np.power(1.0 + step_pct, np.arange(rows))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.linspace(1000.0, 1200.0, rows),
        },
        index=pd.date_range("2026-01-01", periods=rows, freq="5min"),
    )


def test_evaluate_holdout_is_prediction_only_and_cost_adjusted() -> None:
    trainer = ModelTrainer(symbol="HOLDOUTUSDT", input_lookback=20, epochs=1)
    trainer.feature_columns = ["open", "high", "low", "close", "volume"]
    trainer.feature_scalers = {
        col: {"mean": 0.0, "std": 1.0} for col in trainer.feature_columns
    }
    model = _PredictionOnlyModel(0.90)
    trainer.model = model
    trainer.triple_barrier_trend_filter = False

    metrics = trainer.evaluate_holdout(
        _holdout_frame(step_pct=0.002),
        probability_threshold=0.55,
        round_trip_cost_pct=0.003,
        horizon_bars=5,
    )

    assert metrics["ok"] is True
    assert metrics["signals"] > 20
    assert metrics["expectancy_pct"] > 0.0
    assert metrics["profit_factor"] == float("inf")
    assert model.fit_called is False


def test_evaluate_holdout_rejects_gross_moves_smaller_than_cost() -> None:
    trainer = ModelTrainer(symbol="COSTUSDT", input_lookback=20, epochs=1)
    trainer.feature_columns = ["open", "high", "low", "close", "volume"]
    trainer.feature_scalers = {
        col: {"mean": 0.0, "std": 1.0} for col in trainer.feature_columns
    }
    trainer.model = _PredictionOnlyModel(0.90)
    trainer.triple_barrier_trend_filter = False

    metrics = trainer.evaluate_holdout(
        _holdout_frame(step_pct=0.0002),
        probability_threshold=0.55,
        round_trip_cost_pct=0.003,
        horizon_bars=5,
    )

    assert metrics["signals"] > 20
    assert metrics["wins"] == 0
    assert metrics["expectancy_pct"] < 0.0
    assert metrics["profit_factor"] == 0.0


def test_persist_model_still_saves_after_holdout_method_added(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    trainer = ModelTrainer(symbol="SAVEUSDT", input_lookback=20, epochs=1)
    trainer.model = object()
    target = tmp_path / "model.keras"
    saved: list[object] = []

    monkeypatch.setattr(trainer_mod, "save_model", lambda model, path: saved.append((model, path)))
    monkeypatch.setattr(trainer_mod, "model_exists", lambda path: path == target)
    monkeypatch.setattr(trainer, "_save_training_metadata", lambda path: saved.append(path))

    assert trainer.persist_model(target) is True
    assert saved[0] == (trainer.model, target)


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
