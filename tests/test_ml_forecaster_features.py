import math
from typing import List, Dict, Any

from agents.ml_forecaster import MLForecaster


class _DummySharedState:
    dynamic_config = {}


class _DummyConfig:
    pass


def _mk_forecaster() -> MLForecaster:
    return MLForecaster(
        shared_state=_DummySharedState(),
        execution_manager=None,
        config=_DummyConfig(),
        name="MLForecasterFeatureTest",
    )


def _sample_ohlcv(n: int = 180) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    price = 100.0
    vol = 1000.0
    ts = 1700000000000
    for i in range(n):
        drift = 0.15 if (i % 40) < 20 else -0.08
        wiggle = ((i % 7) - 3) * 0.03
        price = max(1.0, price + drift + wiggle)
        high = price * (1.0 + 0.003 + (0.0003 * (i % 3)))
        low = price * (1.0 - 0.003 - (0.0002 * (i % 2)))
        open_p = price * (1.0 - 0.001)
        close_p = price
        vol = max(1.0, vol * (1.0 + (((i % 9) - 4) * 0.01)))
        out.append(
            {
                "timestamp": ts + (i * 60_000),
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": vol,
            }
        )
    return out


def test_edge_feature_frame_contains_expected_columns_and_finite_values() -> None:
    f = _mk_forecaster()
    df = f._build_edge_feature_frame(_sample_ohlcv(180))

    assert not df.empty
    for col in f._legacy_feature_columns + f._edge_feature_columns:
        assert col in df.columns

    tail = df[f._edge_feature_columns].tail(40)
    for _, row in tail.iterrows():
        for col in f._edge_feature_columns:
            v = float(row[col])
            assert math.isfinite(v)

    # Regime flags must stay bounded
    for col in ("trend_flag", "sideways_flag", "high_vol_flag"):
        vals = df[col].tail(40).tolist()
        for v in vals:
            assert 0.0 <= float(v) <= 1.0


def test_resolve_input_columns_supports_legacy_and_edge_modes() -> None:
    f = _mk_forecaster()
    cols, mode = f._resolve_input_columns_for_model(len(f._legacy_feature_columns))
    assert mode == "legacy_ohlcv"
    assert cols == f._legacy_feature_columns

    cols, mode = f._resolve_input_columns_for_model(len(f._edge_feature_columns))
    assert mode == "edge_native"
    assert cols == f._edge_feature_columns
