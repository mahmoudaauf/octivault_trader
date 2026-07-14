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

    def evaluate_holdout(self, df: pd.DataFrame, **_: object) -> dict[str, object]:
        return {"ok": True, "signals": 25, "expectancy_pct": 0.5, "profit_factor": 1.5}


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

    def _split_train_holdout_df(ohlcv, _max_rows):
        train_df = pd.DataFrame(
            {
                "open": [1.0] * len(ohlcv),
                "high": [1.0] * len(ohlcv),
                "low": [1.0] * len(ohlcv),
                "close": [1.0] * len(ohlcv),
                "volume": [1.0] * len(ohlcv),
            }
        )
        holdout_df = train_df.copy()
        return train_df, holdout_df

    forecaster._fetch_training_ohlcv = _fetch_training_ohlcv
    forecaster._split_train_holdout_df = _split_train_holdout_df
    forecaster._passes_retrain_quality_guard = lambda *_, **__: (True, "ok")
    forecaster._passes_retrain_economic_guard = lambda *_, **__: (True, "ok")
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


# ── Regression tests: 2026-07-14 economic retrain-deployment gate ──────────
# (docs/audit strategic plan Phase 0a) -- the live retrain loop previously
# only checked classification accuracy before redeploying a model, with no
# concept of profitability. These tests cover the new
# _split_train_holdout_df / _passes_retrain_economic_guard pair added to
# close that gap, mirroring retrain_weekly.py's already-vetted PnL-based gate.


def _make_forecaster() -> MLForecaster:
    return MLForecaster(
        shared_state=_SharedState(),
        execution_manager=None,
        config=_Cfg(),
        model_manager=_ModelManager(),
        symbols=["BNBUSDT"],
    )


class TestSplitTrainHoldoutDf:
    def test_holdout_is_the_most_recent_rows_and_excluded_from_train(self) -> None:
        forecaster = _make_forecaster()
        full_df = pd.DataFrame({"close": list(range(100))})
        forecaster._build_edge_feature_frame = lambda _ohlcv: full_df
        forecaster._training_feature_columns = lambda: ["close"]
        forecaster._retrain_holdout_frac = 0.15

        train_df, holdout_df = forecaster._split_train_holdout_df(
            ohlcv=[[0]] * 100, row_cap=1000
        )

        assert holdout_df is not None and len(holdout_df) == 15
        assert train_df is not None and len(train_df) == 85
        # No overlap: holdout is the tail, train is everything strictly before it.
        assert holdout_df["close"].tolist() == list(range(85, 100))
        assert train_df["close"].tolist() == list(range(0, 85))

    def test_row_cap_applies_to_training_portion_only(self) -> None:
        forecaster = _make_forecaster()
        full_df = pd.DataFrame({"close": list(range(100))})
        forecaster._build_edge_feature_frame = lambda _ohlcv: full_df
        forecaster._training_feature_columns = lambda: ["close"]
        forecaster._retrain_holdout_frac = 0.15  # holdout = 15 rows

        train_df, holdout_df = forecaster._split_train_holdout_df(
            ohlcv=[[0]] * 100, row_cap=20
        )

        assert len(holdout_df) == 15
        assert len(train_df) == 20  # capped, even though 85 rows were available
        # Row cap keeps the MOST RECENT rows of the training portion (tail),
        # i.e. rows immediately preceding the holdout.
        assert train_df["close"].tolist() == list(range(65, 85))

    def test_empty_feature_frame_returns_none_none(self) -> None:
        forecaster = _make_forecaster()
        forecaster._build_edge_feature_frame = lambda _ohlcv: pd.DataFrame()
        train_df, holdout_df = forecaster._split_train_holdout_df(ohlcv=[], row_cap=100)
        assert train_df is None and holdout_df is None

    def test_too_few_rows_for_any_holdout_returns_train_only(self) -> None:
        """n=1 row: holdout_rows would be clamped to 0 -- must not crash, and
        must return no holdout (fail-closed for the economic gate) rather
        than silently treating the single row as both train and holdout."""
        forecaster = _make_forecaster()
        full_df = pd.DataFrame({"close": [1.0]})
        forecaster._build_edge_feature_frame = lambda _ohlcv: full_df
        forecaster._training_feature_columns = lambda: ["close"]
        forecaster._retrain_holdout_frac = 0.15

        train_df, holdout_df = forecaster._split_train_holdout_df(ohlcv=[[0]], row_cap=100)

        assert holdout_df is None
        assert train_df is not None and len(train_df) == 1


class TestPassesRetrainEconomicGuard:
    def _forecaster_with_thresholds(self) -> MLForecaster:
        f = _make_forecaster()
        f._retrain_min_signals = 20
        f._retrain_min_expectancy_pct = 0.0
        f._retrain_min_profit_factor = 1.20
        return f

    def test_accepts_when_all_thresholds_clear(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": True, "signals": 25, "expectancy_pct": 0.5, "profit_factor": 1.5}
        )
        assert ok is True
        assert reason == "economic_ok"

    def test_rejects_missing_result(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(None)
        assert ok is False
        assert reason == "missing_holdout_result"

    def test_rejects_when_holdout_not_ok(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": False, "reason": "insufficient_rows"}
        )
        assert ok is False
        assert "holdout_not_ok" in reason

    def test_rejects_insufficient_signals(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": True, "signals": 5, "expectancy_pct": 1.0, "profit_factor": 2.0}
        )
        assert ok is False
        assert "insufficient_holdout_signals" in reason

    def test_rejects_non_positive_expectancy_even_with_good_accuracy(self) -> None:
        """The exact bug scenario: a model whose accuracy improved but whose
        real net-of-fee expectancy on genuinely unseen data is <= 0 must be
        rejected -- accuracy alone (the old, sole gate) said nothing about this."""
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": True, "signals": 30, "expectancy_pct": -0.1, "profit_factor": 0.8}
        )
        assert ok is False
        assert "non_positive_expectancy" in reason

    def test_rejects_low_profit_factor(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": True, "signals": 30, "expectancy_pct": 0.1, "profit_factor": 1.0}
        )
        assert ok is False
        assert "profit_factor_below_guard" in reason

    def test_rejects_invalid_metric_types_gracefully(self) -> None:
        f = self._forecaster_with_thresholds()
        ok, reason = f._passes_retrain_economic_guard(
            {"ok": True, "signals": "not_a_number", "expectancy_pct": 1.0, "profit_factor": 2.0}
        )
        assert ok is False
        assert reason == "invalid_holdout_metrics"
