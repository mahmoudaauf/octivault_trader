import numpy as np

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
        name="MLForecasterTest",
    )


def test_decode_two_output_q_values_maps_hold_buy() -> None:
    f = _mk_forecaster()

    action, confidence, probs, schema = f._decode_model_output(np.asarray([0.9, 0.1], dtype=np.float64))
    assert schema == "q2_hold_buy"
    assert action == "hold"
    assert 0.0 <= confidence <= 1.0
    assert probs.shape == (2,)

    action, confidence, probs, schema = f._decode_model_output(np.asarray([0.1, 0.9], dtype=np.float64))
    assert schema == "q2_hold_buy"
    assert action == "buy"
    assert 0.0 <= confidence <= 1.0
    assert probs.shape == (2,)


def test_decode_three_output_maps_buy_sell_hold() -> None:
    f = _mk_forecaster()

    action, _, _, schema = f._decode_model_output(np.asarray([0.9, 0.1, 0.0], dtype=np.float64))
    assert schema == "cls3_buy_sell_hold"
    assert action == "buy"

    action, _, _, schema = f._decode_model_output(np.asarray([0.1, 0.9, 0.0], dtype=np.float64))
    assert schema == "cls3_buy_sell_hold"
    assert action == "sell"

    action, _, _, schema = f._decode_model_output(np.asarray([0.1, 0.0, 0.9], dtype=np.float64))
    assert schema == "cls3_buy_sell_hold"
    assert action == "hold"


def test_decode_scalar_maps_hold_buy() -> None:
    f = _mk_forecaster()

    action, confidence, probs, schema = f._decode_model_output(np.asarray([0.2], dtype=np.float64))
    assert schema == "scalar_hold_buy"
    assert action == "hold"
    assert 0.0 <= confidence <= 1.0
    assert probs.shape == (2,)

    action, confidence, probs, schema = f._decode_model_output(np.asarray([0.8], dtype=np.float64))
    assert schema == "scalar_hold_buy"
    assert action == "buy"
    assert 0.0 <= confidence <= 1.0
    assert probs.shape == (2,)
